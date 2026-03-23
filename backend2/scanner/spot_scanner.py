"""
spot_scanner.py — KSU Smart Parking YOLOv8 Detection Engine
===========================================================
This is the core computer vision engine that processes the video stream.

Responsibilities:
  - Run YOLOv8 at high resolution (1280px) to detect vehicles far from the camera.
  - Use ByteTrack to assign stable IDs to cars, preventing flickering status.
  - Test for geometric overlap between car bounding boxes and spot polygons.
  - Manage "Permanent IDs" (P1, P2...) to survive tracking gaps (occlusions).
  - Implement "N_CONFIRM" logic: status only changes if N frames in a row agree.
  - Implement a "Row Guard" to ensure count of taken spots won't exceed cars in the row.

Coordinate System:
  - Raw annotations in spots.json are pixels on a 640x360 reference canvas.
  - Internal processing uses normalized 0.0 - 1.0 floats.
  - Visualization (annotated frame) uses the actual current video frame pixels.
"""

import asyncio, cv2, numpy as np, time, logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Classes for YOLOv8 (2=car, 3=motorcycle, 5=bus, 7=truck)
VEHICLE_CLASSES = {2, 3, 5, 7}
CONFIDENCE_MIN  = 0.15   # Low threshold allows detecting small/occluded cars
YOLO_MODEL      = "yolov8n.pt"  # Use nano model for speed, or "yolov8s.pt" for accuracy
OVERLAP_THRESH  = 0.10   # Car must cover 10% of a spot polygon to "claim" it
N_CONFIRM       = 3      # Number of consecutive frames for status change
REMATCH_PX      = 120    # Pixel radius to re-link a lost car to its ID
RELEASE_SECONDS = 8.0    # Wait this long before declaring a spot "open" after car leaves
MAX_RECONNECTS  = 3      # For live RTSP streams
MAX_FAILS       = 30

# Parking lot logical layout for row-guard logic
ROWS = {
    'A': ['11','10','9','8','7','6','5','4','3','2','1'],
    'B': ['22','21','20','19','18','17','16','15','14','13','12'],
    'C': ['23','24','25','26','27','28','29','30','31','32','33','34'],
    'D': ['44','43','42','41','40','39','38','37','36','35'],
}

# Consistent colours for permanent car IDs tracking
_ID_COLOURS = [
    (0,220,255),(0,180,255),(50,255,50),(255,180,0),
    (255,80,180),(180,255,80),(80,180,255),(255,255,80),
    (200,80,255),(80,255,200),(255,120,80),(120,255,80),
]

def _pid_colour(pid: str) -> tuple:
    """Return a consistent BGR colour for a permanent ID string (P1, P2...)."""
    n = int(pid[1:]) if pid[1:].isdigit() else 0
    return _ID_COLOURS[n % len(_ID_COLOURS)]


class SpotScanner:
    """
    Main scanner engine. Decoupled from the API, interacts via callbacks.
    """
    def __init__(self, spots, source, on_update: Callable,
                 debug_fn: Optional[Callable] = None,
                 frame_callback: Optional[Callable] = None):
        self.spots      = spots         # List of spot definitions
        self.source     = source        # MP4 path or RTSP URL
        self.on_update  = on_update     # Status change callback (FastAPI layer)
        self._dbg       = debug_fn or (lambda m: logger.info(m))
        self._frame_cb  = frame_callback # Annotated frame callback (for UI)
        self._running   = False
        self._model     = None
        self._fw = self._fh = None

        # Polygon data structures
        self._raw:   Dict[str, np.ndarray] = {}  # Normalized 0-1
        self._polys: Dict[str, np.ndarray] = {}  # Scaled to current frame
        self._build_raw()

        # Tracking state
        self._statuses: Dict[str, str]  = {str(s.id): "open" for s in spots}
        self._pending:  Dict[str, dict] = {}     # For N_CONFIRM consensus

        # Permanent car identity state
        self._next_pid:  int                           = 1
        self._bt_to_pid: Dict[int, str]                = {}  # ByteTrack ID -> P-ID
        self._pid_pos:   Dict[str, Tuple[float,float]] = {}  # Last seen center
        self._pid_seen:  Dict[str, float]              = {}  # Last seen timestamp
        self._pid_spots: Dict[str, Set[str]]           = defaultdict(set)
        self._pid_box:   Dict[str, Tuple[float,float,float,float]] = {} # Bounding box

    def update_polygons(self, spots):
        """Allows polygon definitions to be shared/updated without restarting the video."""
        self.spots = spots
        for s in spots:
            sid = str(s.id)
            if sid not in self._statuses:
                self._statuses[sid] = "open"
        self._build_raw()
        if self._fw and self._fh:
            self._scale(self._fw, self._fh)

    def _build_raw(self):
        """
        Interprets input coordinates. Normalizes them if they are in pixel space.
        Note: The reference frame is 640x360 as per annotations in spots.json.
        """
        self._raw = {}
        for s in self.spots:
            pts = np.array(s.pts, dtype=np.float32)
            if pts.size == 0: continue
            
            # If coordinates are larger than 1.0, they are in 640x360 pixel space
            if pts.max() > 1.0:
                pts[:, 0] /= 640.0
                pts[:, 1] /= 360.0
            self._raw[str(s.id)] = np.clip(pts, 0.0, 1.0)

    def _scale(self, w: int, h: int):
        """Scales normalized 0-1 coordinates to the actual video frame resolution."""
        self._polys = {}
        for sid, p in self._raw.items():
            sc = p.copy()
            # Standard scale-to-pixels logic
            sc[:, 0] *= w
            sc[:, 1] *= h
            self._polys[sid] = sc

    def _load_model(self):
        """Initializes YOLOv8 and ByteTrack."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(YOLO_MODEL)
            # Warm up with a dummy frame
            dummy = np.zeros((384, 640, 3), dtype=np.uint8)
            try:
                self._model.track(dummy, persist=True, verbose=False)
                self._dbg(f"YOLO + ByteTrack ready ({YOLO_MODEL})")
            except Exception:
                self._model(dummy, verbose=False)
                self._dbg(f"YOLO ready (tracking fallback mode)")
        except Exception as e:
            self._dbg(f"YOLO load failed: {e}")

    def _is_live(self) -> bool:
        """Determines if the source is a live stream or a file."""
        if isinstance(self.source, int): return True
        s = str(self.source).lower()
        return any(s.startswith(p) for p in ["rtsp://", "http://", "https://"])

    async def run(self):
        """Main entry point. Runs the loop in an executor thread."""
        self._running = True
        await asyncio.get_event_loop().run_in_executor(None, self._load_model)
        try:
            await self._scan_with_reconnect()
        finally:
            self._running = False
            self._dbg("Scanner stopped")

    async def _scan_with_reconnect(self):
        """Handles retries for live stream drops."""
        attempt = 0
        while self._running:
            try:
                await self._scan()
                break
            except _FeedLost as e:
                attempt += 1
                if attempt > MAX_RECONNECTS: raise
                wait = attempt * 2
                self._dbg(f"Feed dropped. Reconnect {attempt}/{MAX_RECONNECTS} in {wait}s...")
                await asyncio.sleep(wait)

    async def _scan(self):
        """The core frame-processing loop."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened(): raise _FeedLost(f"Cannot open: {self.source}")
        
        # Buffer size of 2 ensures we don't lag behind live cameras
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise _FeedLost("Empty video file/stream")

        # Initial dimension setup
        self._fh, self._fw = frame.shape[:2]
        self._scale(self._fw, self._fh)

        # Save a reference image for the polygon editor to use
        try:
            import os
            ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "reference.jpg")
            cv2.imwrite(ref_path, frame)
        except Exception: pass

        loop = asyncio.get_event_loop()
        frame_n = 0
        is_live = self._is_live()

        while self._running:
            ret, frame = cap.read()
            if not ret:
                if is_live: await asyncio.sleep(0.1); continue
                else: break

            ts = time.time()
            # Offload heavy CV work (YOLO/ByteTrack/Annotate) to a thread to keep the server alive
            changed = await loop.run_in_executor(None, self._process, frame, frame_n, ts)
            
            # Only trigger a state push if the parking lot counts actually changed
            if changed:
                self.on_update({"__feed_lost__": False, **dict(self._statuses)})

            frame_n += 1
            await asyncio.sleep(0) # Yield control

        cap.release()

    def _process(self, frame, frame_n: int, ts: float) -> bool:
        """Processes a single frame: Detect -> Update Logic -> Annotate -> Callback."""
        detections = self._detect(frame, frame_n)
        changed    = self._update(detections, ts, frame_n)

        # Send frame + data to the adapter layer (main.py)
        if self._frame_cb:
            try:
                annotated = self._annotate(frame, detections)
                owner_map = {pid: next(iter(spots)) for pid, spots in self._pid_spots.items() if spots}
                self._frame_cb(annotated, dict(self._statuses), dict(self._polys), owner_map)
            except Exception: pass

        return changed

    def _detect(self, frame, frame_n: int):
        """Runs YOLOv8 at 1280px (doubled standard imgsz) for far detections."""
        if not self._model: return []
        try:
            results = self._model.track(
                frame,
                classes=list(VEHICLE_CLASSES),
                conf=CONFIDENCE_MIN,
                persist=True,
                verbose=False,
                imgsz=1280,  # Critical for small objects far away
                tracker="bytetrack.yaml"
            )[0]
            
            out = []
            if results.boxes.id is not None:
                for box, tid in zip(results.boxes, results.boxes.id):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    out.append((int(tid), cx, cy, x1, y1, x2, y2))
            return out
        except Exception: return []

    def _get_pid(self, bt_id: int, cx: float, cy: float) -> str:
        """Converts ephemeral ByteTrack IDs to persistent Permanent IDs."""
        if bt_id in self._bt_to_pid: return self._bt_to_pid[bt_id]

        # Rematch based on spatial proximity if tracking was lost
        best_pid, best_d = None, float("inf")
        active_pids = set(self._bt_to_pid.values())
        for pid, (px, py) in self._pid_pos.items():
            if pid in active_pids: continue
            d = ((cx-px)**2 + (cy-py)**2)**0.5
            if d < REMATCH_PX and d < best_d:
                best_d, best_pid = d, pid

        if best_pid:
            self._bt_to_pid[bt_id] = best_pid
            return best_pid

        # Truly new car entering the lot
        pid = f"P{self._next_pid}"
        self._next_pid += 1
        self._bt_to_pid[bt_id] = pid
        return pid

    def _best_spot(self, x1, y1, x2, y2) -> Optional[str]:
        """Calculates area-based overlap to find which spot a car owns."""
        car_box = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
        best_sid, best_overlap = None, 0.0
        
        for sid, poly in self._polys.items():
            spot_area = float(cv2.contourArea(poly))
            if spot_area < 1: continue
            try:
                ok, inter = cv2.intersectConvexConvex(poly, car_box)
                if ok and inter is not None and len(inter) > 0:
                    overlap = float(cv2.contourArea(inter)) / spot_area
                    if overlap >= OVERLAP_THRESH and overlap > best_overlap:
                        best_overlap = overlap
                        best_sid     = sid
            except: pass
        return best_sid

    def _update(self, detections, ts: float, frame_n: int) -> bool:
        """Updates internal status based on current detections."""
        active_pids: Set[str] = set()
        new_status: Dict[str, str] = {sid: "open" for sid in self._polys}

        for (bt_id, cx, cy, x1, y1, x2, y2) in detections:
            pid = self._get_pid(bt_id, cx, cy)
            active_pids.add(pid)
            self._pid_pos[pid], self._pid_seen[pid], self._pid_box[pid] = (cx, cy), ts, (x1, y1, x2, y2)

            # Link car to spot
            best = self._best_spot(x1, y1, x2, y2)
            self._pid_spots[pid] = {best} if best else set()
            if best: new_status[best] = "taken"

        # Cleanup old cars that went out of frame
        for pid in list(self._pid_seen):
            if pid not in active_pids and (ts - self._pid_seen[pid]) >= RELEASE_SECONDS:
                self._pid_seen.pop(pid); self._pid_pos.pop(pid, None); self._pid_box.pop(pid, None); self._pid_spots.pop(pid, None)
                for bt in [k for k,v in self._bt_to_pid.items() if v == pid]: del self._bt_to_pid[bt]

        self._row_guard(new_status, active_pids)
        return self._confirm(new_status)

    def _annotate(self, frame, detections) -> np.ndarray:
        """Draws the visual debug overlays on the final frame."""
        vis = frame.copy()
        for sid, poly in self._polys.items():
            pts = poly.astype(np.int32)
            taken = self._statuses.get(sid) == "taken"
            if taken:
                overlay = vis.copy()
                cv2.fillPoly(overlay, [pts], (0, 60, 180))
                cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis) # Semi-transparent blue for taken
                cv2.polylines(vis, [pts], True, (0, 100, 255), 2)
            else:
                cv2.polylines(vis, [pts], True, (0, 200, 0), 1) # Thin green for open
            
            # Label spot number
            cx, cy = int(np.mean(poly[:, 0])), int(np.mean(poly[:, 1]))
            color = (0, 100, 255) if taken else (0, 200, 0)
            cv2.putText(vis, sid, (cx-8, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        # Draw car tracking dots
        for (bt_id, cx, cy, x1, y1, x2, y2) in detections:
            pid = self._bt_to_pid.get(bt_id, f"BT{bt_id}")
            color = _pid_colour(pid)
            cv2.circle(vis, (int(cx), int(cy)), 8, color, -1)
            cv2.circle(vis, (int(cx), int(cy)), 8, (0, 0, 0), 1)
            
            label = pid
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            lx, ly = int(cx) + 10, int(cy) + th // 2
            cv2.rectangle(vis, (lx-1, ly-th-1), (lx+tw+3, ly+2), (0, 0, 0), -1)
            cv2.putText(vis, label, (lx+1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return vis

    def _row_guard(self, new_status: Dict[str, str], active_pids: Set[str]):
        """Sanity check: ensure single spots aren't double-claimed by camera angles."""
        for row_name, row_ids in ROWS.items():
            cars_in_row = 0
            for pid, (px, py) in self._pid_pos.items():
                if pid not in active_pids: continue
                
                # Check which row this car center belongs to
                best_row, best_d = None, float("inf")
                for rn, rids in ROWS.items():
                    for rsid in rids:
                        if rsid not in self._polys: continue
                        poly = self._polys[rsid]
                        pcx, pcy = np.mean(poly[:, 0]), np.mean(poly[:, 1])
                        d = ((px-pcx)**2 + (py-pcy)**2)**0.5
                        if d < best_d: best_d, best_row = d, rn
                
                if best_row == row_name: cars_in_row += 1

            taken_in_row = [s for s in row_ids if new_status.get(s) == "taken"]
            if len(taken_in_row) > cars_in_row:
                # If AI claims more spots than cars exist in this row, release the excess
                for sid in taken_in_row[:len(taken_in_row)-cars_in_row]: new_status[sid] = "open"

    def _confirm(self, new_status: Dict[str, str]) -> bool:
        """Requires N consecutive frames of same state before switching."""
        changed = False
        for sid, new_state in new_status.items():
            current = self._statuses.get(sid, "open")
            if new_state == current: self._pending.pop(sid, None); continue
            
            p = self._pending.setdefault(sid, {"state": new_state, "count": 0})
            if p["state"] == new_state: p["count"] += 1
            else: p["state"], p["count"] = new_state, 1
            
            if p["count"] >= N_CONFIRM:
                self._statuses[sid] = new_state
                del self._pending[sid]
                changed = True
        return changed

class _FeedLost(RuntimeError): pass
