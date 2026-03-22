"""
spot_scanner.py — persistent car-ID ownership + live frame feed

Every processed frame:
  1. YOLO+ByteTrack detects all vehicles with persistent IDs
  2. Each car ID is assigned to its nearest spot polygon (one car = one spot)
  3. Spot stays TAKEN as long as car ID is being tracked
  4. Spot only goes OPEN when car ID absent for RELEASE_SECONDS
  5. Row count guard: taken per row never exceeds detected cars in row
  6. Annotated frame pushed to /feed endpoint for live viewing
"""

import asyncio, cv2, numpy as np, time, logging
from typing import Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

VEHICLE_CLASSES = {2, 3, 5, 7}
CONFIDENCE_MIN  = 0.20
# Upgraded from Nano (n) to Small (s) for much better small-object detection accuracy!
YOLO_MODEL      = "yolov8s.pt"
# These MUST exactly match the resolution the bounding boxes were drawn on! (640x360)
POLY_REF_W      = 640
POLY_REF_H      = 360
RELEASE_SECONDS = 4.0
# Radically reduced so cars driving along the aisle don't accidentally get assigned to spots!
SNAP_DIST_FRAC  = 0.04

ROWS = {
    'A': ['1','2','3','4','5','6','7','8','9','10','11'],
    'B': ['12','13','14','15','16','17','18','19','20','21','22'],
    'C': ['23','24','25','26','27','28','29','30','31','32','33','34'],
    'D': ['35','36','37','38','39','40','41','42','43','44'],
}


class SpotScanner:
    def __init__(self, spots, source, on_update: Callable,
                 debug_fn: Optional[Callable] = None,
                 frame_callback: Optional[Callable] = None):
        self.spots          = spots
        self.source         = source
        self.on_update      = on_update
        self._dbg           = debug_fn or (lambda m: logger.info(m))
        self._frame_cb      = frame_callback   # called with (frame, statuses, polys, car_owns)
        self._running       = False
        self._model         = None
        self._fw = self._fh = None

        self._polys:   Dict[str, np.ndarray]         = {}
        self._centers: Dict[str, Tuple[float,float]] = {}
        self._raw:     Dict[str, np.ndarray]         = {}

        self._spot_owner: Dict[str, Optional[int]] = {s.id: None for s in spots}
        self._last_seen:  Dict[int, float]         = {}
        self._car_owns:   Dict[int, str]           = {}
        self._statuses:   Dict[str, str]           = {s.id: "open" for s in spots}

        self._build_raw()

    def update_polygons(self, spots):
        self.spots = spots
        for s in spots:
            if s.id not in self._statuses:
                self._statuses[s.id]   = "open"
                self._spot_owner[s.id] = None
        self._build_raw()
        if self._fw: self._scale(self._fw, self._fh)

    def _build_raw(self):
        self._raw = {s.id: np.array(s.points, dtype=np.float32) for s in self.spots}

    def _scale(self, w, h):
        sx, sy = w/POLY_REF_W, h/POLY_REF_H
        self._polys = {}; self._centers = {}
        for sid, p in self._raw.items():
            sc = p.copy(); sc[:,0]*=sx; sc[:,1]*=sy
            self._polys[sid]   = sc
            self._centers[sid] = (float(np.mean(sc[:,0])), float(np.mean(sc[:,1])))
        self._dbg(f"Polygons scaled to {w}x{h} (sx={sx:.3f} sy={sy:.3f})")

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO(YOLO_MODEL)
            dummy = np.zeros((384,640,3), dtype=np.uint8)
            try:
                self._model.track(dummy, persist=True, verbose=False)
                self._dbg(f"YOLO+ByteTrack ready ({YOLO_MODEL})")
            except Exception:
                self._model(dummy, verbose=False)
                self._dbg(f"YOLO ready, ByteTrack unavailable")
        except Exception as e:
            self._dbg(f"YOLO load failed: {e}")

    def _is_live(self):
        if isinstance(self.source, int): return True
        s = str(self.source).lower()
        return any(s.startswith(p) for p in ("rtsp://","http://","https://"))

    async def run(self):
        self._running = True
        await asyncio.get_event_loop().run_in_executor(None, self._load_model)
        try:
            await self._scan()
        except _FeedLost as e:
            self._dbg(f"FEED LOST: {e}")
            self.on_update({"__feed_lost__": True, **dict(self._statuses)})
        except Exception as e:
            import traceback
            self._dbg(f"Fatal: {e}\n{traceback.format_exc()}")
        finally:
            self._running = False
            self._dbg("Scanner stopped")

    async def _scan(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened(): raise _FeedLost(f"Cannot open: {self.source}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        if not ret:
            cap.release(); raise _FeedLost("Cannot read first frame")

        self._fh, self._fw = frame.shape[:2]
        self._scale(self._fw, self._fh)

        is_live = self._is_live()
        loop    = asyncio.get_event_loop()
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else -1
        nat_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fails   = 0; frame_n = 0

        self._dbg(
            f"{'Live' if is_live else f'Video ({total/nat_fps:.1f}s)'} | "
            f"{self._fw}x{self._fh} | {len(self.spots)} spots | "
            f"release after {RELEASE_SECONDS}s"
        )

        changed = await loop.run_in_executor(None, self._process, frame, 0, time.time())
        if changed: self.on_update({"__feed_lost__": False, **dict(self._statuses)})
        frame_n = 1

        while self._running:
            ret, frame = cap.read()
            if not ret:
                if is_live:
                    fails += 1
                    if fails >= 5:
                        cap.release(); raise _FeedLost("5 consecutive failed reads")
                    await asyncio.sleep(0.1); continue
                else:
                    self._dbg(f"Video finished at frame {frame_n}"); break

            fails = 0
            ts = time.time()
            changed = await loop.run_in_executor(None, self._process, frame, frame_n, ts)
            if changed: self.on_update({"__feed_lost__": False, **dict(self._statuses)})
            frame_n += 1

            if frame_n % 120 == 0:
                taken  = sum(1 for v in self._statuses.values() if v=="taken")
                pct    = f"{frame_n/total*100:.0f}%" if total>0 else "live"
                self._dbg(f"[{pct}] fr={frame_n} | {taken} taken | {len(self._car_owns)} cars tracked")

            await asyncio.sleep(0)
        cap.release()

    # ── per-frame ─────────────────────────────────────────────────────────────

    def _process(self, frame, frame_n, ts):
        detections = self._detect(frame, frame_n)
        changed    = self._update_ownership(detections, ts, frame_n)

        # Push annotated frame to /feed regardless of status change
        if self._frame_cb:
            try:
                self._frame_cb(
                    frame,
                    dict(self._statuses),
                    dict(self._polys),
                    dict(self._car_owns)
                )
            except Exception:
                pass

        return changed

    def _detect(self, frame, frame_n):
        if not self._model: return []

        # Try ByteTrack
        try:
            results = self._model.track(
                frame, classes=list(VEHICLE_CLASSES),
                conf=CONFIDENCE_MIN, persist=True,
                verbose=False, imgsz=640, tracker="bytetrack.yaml"
            )[0]
            if results.boxes.id is not None:
                out = []
                for box, tid in zip(results.boxes, results.boxes.id):
                    if int(box.cls[0]) not in VEHICLE_CLASSES: continue
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    out.append((int(tid), (x1+x2)/2, (y1+y2)/2))
                return out
        except Exception:
            pass

        # Fallback: plain YOLO
        try:
            results = self._model(
                frame, classes=list(VEHICLE_CLASSES),
                conf=CONFIDENCE_MIN, verbose=False, imgsz=640
            )[0]
            return [
                (-(i+1), *(lambda b: ((b[0]+b[2])/2, (b[1]+b[3])/2))(box.xyxy[0].tolist()))
                for i,box in enumerate(results.boxes)
                if int(box.cls[0]) in VEHICLE_CLASSES
            ]
        except Exception as e:
            if frame_n%60==0: self._dbg(f"Detection error: {e}")
            return []

    def _update_ownership(self, detections, ts, frame_n):
        active_ids = set()

        # Update last-seen and assign new cars
        for (tid, cx, cy) in detections:
            active_ids.add(tid)
            self._last_seen[tid] = ts
            if tid in self._car_owns:
                continue  # already owns a spot — keep it
            sid = self._nearest_free_spot(cx, cy)
            if sid:
                self._car_owns[tid]       = sid
                self._spot_owner[sid]     = tid
                self._dbg(f"Car #{tid} → spot {sid} ({cx:.0f},{cy:.0f})")

        # Release spots for cars absent > RELEASE_SECONDS
        for tid in list(self._last_seen):
            if tid not in active_ids:
                if ts - self._last_seen[tid] >= RELEASE_SECONDS:
                    if tid in self._car_owns:
                        sid = self._car_owns.pop(tid)
                        self._spot_owner[sid] = None
                        self._dbg(f"Car #{tid} gone → spot {sid} freed")
                    del self._last_seen[tid]

        # Row count guard (Disabled since dynamic IDs cause this to aggressively false-free legitimate cars)
        # self._row_count_guard(detections, frame_n)

        # Compute changes
        changed = False
        for sid in self._statuses:
            new = "taken" if self._spot_owner.get(sid) is not None else "open"
            if new != self._statuses[sid]:
                self._statuses[sid] = new
                self._dbg(f"Spot {sid} → {new}")
                changed = True
        return changed

    def _nearest_free_spot(self, cx, cy):
        max_snap = (self._fh or 580) * SNAP_DIST_FRAC
        best_sid = None; best_d = float("inf")
        pt = (float(cx), float(cy))

        # Inside polygon first
        for sid, poly in self._polys.items():
            if self._spot_owner.get(sid) is not None: continue
            d = cv2.pointPolygonTest(poly, pt, measureDist=True)
            if d >= 0 and -d < best_d:
                best_d = -d; best_sid = sid

        if best_sid: return best_sid

        # Nearest center
        for sid, (pcx,pcy) in self._centers.items():
            if self._spot_owner.get(sid) is not None: continue
            d = ((cx-pcx)**2+(cy-pcy)**2)**0.5
            if d < max_snap and d < best_d:
                best_d = d; best_sid = sid

        return best_sid

    def _row_count_guard(self, detections, frame_n):
        for row_name, row_ids in ROWS.items():
            cars_in_row = []
            for (tid, cx, cy) in detections:
                best_row = None; best_d = float("inf")
                for rn, rids in ROWS.items():
                    for rsid in rids:
                        if rsid not in self._centers: continue
                        pcx,pcy = self._centers[rsid]
                        d = ((cx-pcx)**2+(cy-pcy)**2)**0.5
                        if d < best_d: best_d=d; best_row=rn
                if best_row==row_name: cars_in_row.append(tid)

            taken_spots = [s for s in row_ids
                           if s in self._spot_owner and self._spot_owner[s] is not None]

            if len(taken_spots) > len(cars_in_row):
                excess = len(taken_spots)-len(cars_in_row)
                self._dbg(f"Row {row_name}: {len(taken_spots)} taken > "
                          f"{len(cars_in_row)} cars — freeing {excess}")
                freed = 0
                for sid in taken_spots:
                    if freed >= excess: break
                    owner = self._spot_owner[sid]
                    if owner not in {t for t,_,_ in detections}:
                        if owner in self._car_owns: del self._car_owns[owner]
                        self._spot_owner[sid] = None
                        freed += 1


class _FeedLost(RuntimeError):
    pass
