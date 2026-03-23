"""
api/main.py — KSU Smart Parking FastAPI Server
================================================
This is the adapter layer between the YOLOv8 scanner engine and the React frontend.

Responsibilities:
  - Serve a WebSocket endpoint (/ws) that the React frontend connects to
  - On startup, auto-start the scanner on the demo video
  - Accept POST /stream/start to switch to a live RTSP camera or a new video
  - Accept GET/PUT /polygons so the polygon editor can hot-reload spot definitions
  - Receive each annotated frame from the scanner via frame_callback
  - Encode the frame as a Base64 JPEG and broadcast the full update payload to all WS clients

Adapter pattern:
  The scanner (spot_scanner.py) works in its own coordinate system — YOLOv8
  pixel space scaled to the raw video frame dimensions.
  This file bridges that internal model and the frontend's expected JSON shape
  WITHOUT the scanner needing to know anything about HTTP or React.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio, json, time, logging, threading, base64
from typing import Dict, List, Optional
from collections import deque
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="KSU Parking")
# Allow requests from the React dev server (localhost:5173) and any other origin
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── WebSocket connection manager ───────────────────────────────────────────────

class ConnectionManager:
    """Tracks all active WebSocket clients and fan-outs broadcast messages."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        """Send a JSON string to every connected frontend client."""
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception:
                pass  # Stale connections are silently dropped


manager = ConnectionManager()
# main_loop is captured at startup so background threads can schedule coroutines
main_loop = None

# ── File paths ─────────────────────────────────────────────────────────────────

DATA_DIR      = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
# spots.json is the canonical human-annotated polygon file (640×360 pixel coords)
POLYGONS_FILE = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "scripts", "annotation", "spots.json"))
STATUS_FILE   = os.path.join(DATA_DIR, "status.json")
# The annotation canvas resolution — spots.json was drawn at this size
POLY_REF_W, POLY_REF_H = 640, 360

# ── In-memory state ────────────────────────────────────────────────────────────

sse_q:   List[asyncio.Queue] = []
debug_q: List[asyncio.Queue] = []
scanner  = None
dbg_log: deque = deque(maxlen=500)   # ring buffer of debug log lines
feed_ok  = True

# Protected by _frame_lock — written from the scanner thread, read from async routes
_frame_lock   = threading.Lock()
_latest_frame = None   # JPEG bytes with polygon/car-ID overlays (for /feed endpoint)


# ── Frame callback (called by scanner on each processed frame) ─────────────────

def set_latest_frame(frame_bgr, statuses, polys, car_owns):
    """
    Adapter between the scanner engine and the WebSocket broadcast.

    Called from a background thread by SpotScanner._process() after every frame.
    Performs two jobs:
      1. Store the annotated frame as JPEG for the /feed MJPEG endpoint.
      2. Build the JSON payload the React frontend expects and broadcast it via /ws.

    Args:
        frame_bgr  — annotated BGR NumPy frame (polygons and car dots already drawn)
        statuses   — dict {spot_id: "open" | "taken"}
        polys      — dict {spot_id: np.ndarray of shape (N,2)} in normalized 0-1 coords
        car_owns   — dict {permanent_id: spot_id} mapping each car to the spot it owns
    """
    global _latest_frame, main_loop

    # ── 1. Store annotated frame for the /feed MJPEG endpoint ─────────────────
    # The scanner already drew the polygon outlines and car-ID dots onto frame_bgr.
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 72])
    with _frame_lock:
        _latest_frame = buf.tobytes()

    # ── 2. Resize for WebSocket broadcast (keeps payload small) ───────────────
    max_width = 1280
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame_resized = cv2.resize(frame_bgr, (max_width, int(h * scale)))
    else:
        frame_resized = frame_bgr

    # Encode the resized frame as a Base64 data URI for <img src="..."> in React
    _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 50])
    base64_img = base64.b64encode(buffer).decode('utf-8')
    image_uri = f"data:image/jpeg;base64,{base64_img}"

    # ── 3. Build the spots list for the frontend ───────────────────────────────
    spots_list = []
    occupied_count = 0
    available_count = 0

    for sid, poly in polys.items():
        is_occupied = (statuses.get(str(sid)) == "taken")
        if is_occupied:
            occupied_count += 1
        else:
            available_count += 1

        # poly is in normalized 0-1 coords from the scanner.
        # Scale to the resized frame's pixel dimensions so the SVG viewBox
        # in React (which equals the JPEG's naturalWidth/naturalHeight) aligns perfectly.
        scaled_poly = (poly * [frame_resized.shape[1], frame_resized.shape[0]]).astype(np.int32)

        spots_list.append({
            "id": int(sid) if str(sid).isdigit() else sid,
            "label": str(sid),
            "isOccupied": is_occupied,
            "polygon": scaled_poly.tolist()  # [[x1,y1],[x2,y2],...] in resized-frame pixels
        })

    # ── 4. Broadcast the full payload to all connected React clients ───────────
    payload = {
        "type": "update",           # React checks data.type === 'update'
        "lotName": "KSU Main Campus",
        "totalSpots": len(polys),
        "availableSpots": available_count,
        "occupiedSpots": occupied_count,
        "spots": spots_list,
        "frameImage": image_uri     # Base64 JPEG — React sets this as <img src>
    }
    if main_loop and main_loop.is_running():
        # We're in a background thread — use run_coroutine_threadsafe to schedule
        # the async broadcast on the server's event loop safely.
        asyncio.run_coroutine_threadsafe(manager.broadcast(json.dumps(payload)), loop=main_loop)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load(path, default):
    """Load JSON from disk, returning `default` on any error."""
    try:    return json.load(open(path))
    except: return default

def _save(path, data):
    """Serialize `data` to JSON on disk."""
    json.dump(data, open(path, "w"), indent=2)

def _broadcast(data: dict):
    """Put a message onto the SSE queues (legacy SSE clients)."""
    msg = json.dumps(data)
    for q in sse_q: q.put_nowait(msg)

def dbg(msg: str):
    """Log a message to the console, the ring buffer, and any debug SSE clients."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    logger.info(msg)
    dbg_log.append(line)
    for q in debug_q: q.put_nowait(line)


# ── Pydantic models ────────────────────────────────────────────────────────────

class Spot(BaseModel):
    """A single parking spot with a string ID and normalized 0-1 polygon points."""
    id: str
    pts: List[List[float]]

def convert_repo_spots_to_scanner_spots(raw_spots):
    """
    Convert the raw spots.json entries into Spot objects with normalized 0-1 coordinates.

    spots.json stores pixel coordinates drawn on a 640×360 canvas.
    If coords are already in 0-1 range they are passed through unchanged.
    If they are large pixel values they are divided by POLY_REF_W / POLY_REF_H.

    Returns a list of Spot objects ready to be passed to SpotScanner.__init__.
    """
    converted = []
    for s in raw_spots:
        sid = str(s.get("id"))
        pts = s.get("pts") or s.get("points") or []

        norm_pts = []
        for x, y in pts:
            if 0 <= x <= 1 and 0 <= y <= 1:
                # Already normalized
                norm_pts.append([float(x), float(y)])
            else:
                # Legacy absolute pixel coords — divide by the annotation canvas size
                norm_pts.append([
                    float(x) / POLY_REF_W,
                    float(y) / POLY_REF_H,
                ])
        converted.append(Spot(id=sid, pts=norm_pts))
    return converted

class StreamPayload(BaseModel):
    url: str


# ── Utility ────────────────────────────────────────────────────────────────────

def _resolve_url(url: str) -> str:
    """Resolve a YouTube URL to a direct stream URL using yt-dlp, or pass through as-is."""
    if "youtube.com" in url or "youtu.be" in url:
        dbg("Resolving YouTube URL...")
        try:
            import yt_dlp
            with yt_dlp.YoutubeDL({"quiet": True, "format": "best[ext=mp4]/best"}) as ydl:
                info   = ydl.extract_info(url, download=False)
                stream = info.get("url") or (info.get("formats") or [{}])[-1].get("url", "")
                if not stream: raise RuntimeError("No stream URL")
                dbg("YouTube URL resolved")
                return stream
        except ImportError:
            raise RuntimeError("yt-dlp not installed — run: pip install yt-dlp")
        except Exception as e:
            raise RuntimeError(f"yt-dlp error: {e}")
    return url


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Simple health check — returns feed status so the frontend status indicator works."""
    return {"status": "ok", "feed": "ok" if feed_ok else "lost"}


@app.get("/feed")
async def video_feed():
    """
    MJPEG streaming endpoint.
    Serves the latest annotated frame in a continuous multipart JPEG stream.
    Useful for opening directly in a browser tab to verify the scanner output.
    """
    async def gen():
        while True:
            with _frame_lock:
                frame = _latest_frame
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + frame + b"\r\n")
            await asyncio.sleep(0.05)   # ~20 fps cap
    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"}
    )


@app.get("/debug/stream")
async def debug_stream():
    """Server-Sent Events endpoint — streams debug log lines to the browser console."""
    q: asyncio.Queue = asyncio.Queue()
    debug_q.append(q)
    history = list(dbg_log)
    async def gen():
        try:
            for line in history: yield f"data: {json.dumps(line)}\n\n"
            yield f"data: {json.dumps('--- live ---')}\n\n"
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=30)
                    yield f"data: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"   # keep-alive
        finally:
            if q in debug_q: debug_q.remove(q)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@app.post("/stream/start")
async def stream_start(payload: StreamPayload):
    """
    Start (or restart) the scanner on a given video URL or file path.

    Steps:
      1. Resolve YouTube URLs to direct stream URLs via yt-dlp.
      2. Load polygon definitions from spots.json and normalize them.
      3. Stop any currently running scanner.
      4. Instantiate SpotScanner with the adapter callback (set_latest_frame).
      5. Launch the scanner as an async background task.
    """
    global scanner, feed_ok
    dbg(f"Stream start: {payload.url}")
    try:    resolved = _resolve_url(payload.url)
    except RuntimeError as e: raise HTTPException(400, str(e))

    # Load and convert the polygon definitions
    saved = _load(POLYGONS_FILE, [])
    spots_list = saved if isinstance(saved, list) else saved.get("spots", [])
    scanner_spots = convert_repo_spots_to_scanner_spots(spots_list)
    if not scanner_spots:
        raise HTTPException(400, "No polygons found")

    def on_update(data: dict):
        """Called by the scanner when a spot status changes."""
        global feed_ok
        lost    = data.pop("__feed_lost__", False)
        feed_ok = not lost
        _save(STATUS_FILE, data)
        _broadcast({"type":"status_update","statuses":data,
                    "feed":"lost" if lost else "ok"})

    # Stop any existing scanner before starting a new one
    if scanner and getattr(scanner, "_running", False):
        scanner._running = False
        await asyncio.sleep(0.3)

    from scanner.spot_scanner import SpotScanner
    scanner = SpotScanner(
        spots=scanner_spots,
        source=resolved,
        on_update=on_update,
        debug_fn=dbg,
        frame_callback=set_latest_frame,   # <-- adapter hook
    )
    feed_ok = True
    asyncio.create_task(scanner.run())
    dbg(f"Scanner started — {len(scanner_spots)} spots")
    return {"message": "Scanner started"}


@app.post("/stream/stop")
async def stream_stop():
    """Stop the currently running scanner."""
    global scanner, feed_ok
    if scanner:
        scanner._running = False
        scanner = None
    feed_ok = False
    return {"message": "Scanner stopped"}


@app.get("/polygons")
def get_polygons():
    """Return the current polygon definitions from spots.json."""
    return _load(POLYGONS_FILE, [])


@app.put("/polygons")
async def update_polygons(spots: List[Spot]):
    """
    Hot-reload polygon definitions without restarting the scanner.
    Called by the polygon editor when the user saves changes.
    """
    _save(POLYGONS_FILE, [s.dict() for s in spots])
    if scanner:
        scanner.update_polygons(spots)
        dbg(f"Polygons updated: {len(spots)} spots")
    return {"message": f"{len(spots)} polygons saved"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Primary WebSocket endpoint.
    React connects here on page load and receives `update` JSON payloads
    every time the scanner processes a new frame.
    """
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()   # keep connection alive; we push, not pull
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ── Auto-start on server startup ───────────────────────────────────────────────

# Path to the demo video — change this to an RTSP URL for a live camera
AUTO_START_URL = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                 "scripts", "demo", "demo-parking.mp4")
)

@app.on_event("startup")
async def startup():
    """
    Capture the event loop and auto-start the demo video feed.
    The loop reference is used by set_latest_frame to safely schedule
    async broadcasts from the background scanner thread.
    """
    global main_loop
    main_loop = asyncio.get_event_loop()
    dbg("Backend ready. Auto-starting YOLOv8 Tracking Feed...")
    # Kick off the scanner start without blocking startup
    asyncio.create_task(stream_start(StreamPayload(url=AUTO_START_URL)))
