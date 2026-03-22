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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()
main_loop = None

DATA_DIR      = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
POLYGONS_FILE = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "scripts", "annotation", "spots.json"))
STATUS_FILE   = os.path.join(DATA_DIR, "status.json")
POLY_REF_W, POLY_REF_H = 928, 580

sse_q:   List[asyncio.Queue] = []
debug_q: List[asyncio.Queue] = []
scanner  = None
dbg_log: deque = deque(maxlen=500)
feed_ok  = True

# Live frame store
_frame_lock   = threading.Lock()
_latest_frame = None   # JPEG bytes with annotations


def set_latest_frame(frame_bgr, statuses, polys, car_owns):
    """Annotate and store latest frame as JPEG for /feed endpoint AND broadcast to /ws for React Frontend."""
    global _latest_frame, main_loop
    
    # 1) Store standard vis for /feed
    vis = frame_bgr.copy()
    for sid, poly in polys.items():
        pts   = poly.astype(np.int32)
        taken = statuses.get(sid) == "taken"
        color = (0, 80, 200) if taken else (0, 200, 0)
        cv2.polylines(vis, [pts], True, color, 2)
        cx = int(np.mean(poly[:, 0]))
        cy = int(np.mean(poly[:, 1]))
        cv2.putText(vis, str(sid), (cx-8, cy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
    for car_id, sid in car_owns.items():
        if sid in polys:
            cx = int(np.mean(polys[sid][:, 0]))
            cy = int(np.mean(polys[sid][:, 1]))
            cv2.putText(vis, f"#{car_id}", (cx-10, cy-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)
    _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 72])
    with _frame_lock:
        _latest_frame = buf.tobytes()
        
    # 2) Broadcast Base64 clean frame to our /ws endpoint!
    max_width = 1280
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * scale)))
        
    _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 50])
    base64_img = base64.b64encode(buffer).decode('utf-8')
    image_uri = f"data:image/jpeg;base64,{base64_img}"
    
    spots_list = []
    occupied_count = 0
    available_count = 0
    
    for sid, poly in polys.items():
        is_occupied = (statuses.get(str(sid)) == "taken")
        if is_occupied:
            occupied_count += 1
        else:
            available_count += 1
        spots_list.append({
            "id": int(sid) if str(sid).isdigit() else sid,
            "label": str(sid),
            "isOccupied": is_occupied,
            "polygon": poly.tolist()  # Native list of [x, y] format the frontend parses via SVG
        })
        
    payload = {
        "type": "update",
        "lotName": "KSU Main Campus - East Deck (YOLOv8 ByteTrack)",
        "totalSpots": len(polys),
        "availableSpots": available_count,
        "occupiedSpots": occupied_count,
        "spots": spots_list,
        "frameImage": image_uri
    }
    if main_loop and main_loop.is_running():
        asyncio.run_coroutine_threadsafe(manager.broadcast(json.dumps(payload)), loop=main_loop)


def _load(path, default):
    try:    return json.load(open(path))
    except: return default

def _save(path, data):
    json.dump(data, open(path, "w"), indent=2)

def _broadcast(data: dict):
    msg = json.dumps(data)
    for q in sse_q: q.put_nowait(msg)

def dbg(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    logger.info(msg)
    dbg_log.append(line)
    for q in debug_q: q.put_nowait(line)


class Spot(BaseModel):
    id: int
    points: List[List[int]]

class StreamPayload(BaseModel):
    url: str


def _resolve_url(url: str) -> str:
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


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "feed": "ok" if feed_ok else "lost"}


@app.get("/feed")
async def video_feed():
    """MJPEG stream — shows live camera feed with polygon + car ID overlays."""
    async def gen():
        while True:
            with _frame_lock:
                frame = _latest_frame
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + frame + b"\r\n")
            await asyncio.sleep(0.05)
    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"}
    )


@app.get("/debug/stream")
async def debug_stream():
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
                    yield ": ping\n\n"
        finally:
            if q in debug_q: debug_q.remove(q)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@app.post("/stream/start")
async def stream_start(payload: StreamPayload):
    global scanner, feed_ok
    dbg(f"Stream start: {payload.url}")
    try:    resolved = _resolve_url(payload.url)
    except RuntimeError as e: raise HTTPException(400, str(e))

    saved = _load(POLYGONS_FILE, [])
    spots_list = saved if isinstance(saved, list) else saved.get("spots", [])
    spots = [Spot(**s) for s in spots_list]
    if not spots: raise HTTPException(400, "No polygons found")

    def on_update(data: dict):
        global feed_ok
        lost    = data.pop("__feed_lost__", False)
        feed_ok = not lost
        _save(STATUS_FILE, data)
        _broadcast({"type":"status_update","statuses":data,
                    "feed":"lost" if lost else "ok"})

    if scanner and getattr(scanner, "_running", False):
        scanner._running = False
        await asyncio.sleep(0.3)

    from scanner.spot_scanner import SpotScanner
    scanner = SpotScanner(spots=spots, source=resolved,
                          on_update=on_update, debug_fn=dbg,
                          frame_callback=set_latest_frame)
    feed_ok = True
    asyncio.create_task(scanner.run())
    dbg(f"Scanner started — {len(spots)} spots")
    return {"message": "Scanner started"}


@app.post("/stream/stop")
async def stream_stop():
    global scanner, feed_ok
    if scanner:
        scanner._running = False
        scanner = None
    feed_ok = True
    dbg("Scanner stopped")
    return {"message": "Stopped"}


@app.get("/stream/status")
def stream_status():
    return {
        "running": scanner is not None and getattr(scanner, "_running", False),
        "feed":    "ok" if feed_ok else "lost",
    }


@app.get("/polygons")
def get_polygons():
    saved = _load(POLYGONS_FILE, [])
    return {"spots": saved} if isinstance(saved, list) else saved


@app.get("/status")
def get_status():
    return _load(STATUS_FILE, {})


@app.get("/stream")
async def sse():
    q: asyncio.Queue = asyncio.Queue()
    sse_q.append(q)
    async def gen():
        try:
            yield 'data: {"type":"connected"}\n\n'
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=30)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            if q in sse_q: sse_q.remove(q)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@app.on_event("startup")
async def startup():
    global main_loop
    main_loop = asyncio.get_running_loop()
    dbg("Backend ready. Auto-starting YOLOv8 Tracking Feed...")
    url = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "scripts", "demo", "demo-parking.mp4"))
    await stream_start(StreamPayload(url=url))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
