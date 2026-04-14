# backend/app.py
"""
FastAPI application for Smart Parking KSU.

Run from project root with the venv active:
    uvicorn backend.app:app --reload --port 8000

Serves both the JSON API and the static frontend at http://localhost:8000/
"""

from pathlib import Path
from threading import Thread

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ml.yolo import config

from .state import state, state_lock
from .inference_worker import run_inference_loop, render_preview_frame


app = FastAPI(title="Smart Parking KSU Backend", version="0.3.0")

# CORS: allow everything. Safe for a demo, simplifies phone testing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class SelectVideoRequest(BaseModel):
    video_name: str   # filename only, e.g. "demo_angled.mp4"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _spots_path_for(video_name: str) -> Path:
    """Convention: videos/foo.mp4 -> spots/spots_foo.json"""
    stem = Path(video_name).stem
    return config.SPOTS_DIR / f"spots_{stem}.json"

def _list_spots_files_for(video_name: str) -> list[Path]:
    """
    Find all spots files for a video, including variants.
    Default file:  spots_<stem>.json
    Variant files: spots_<stem>_<anything>.json

    Returns a sorted list. The default file (if it exists) is always first.
    """
    stem = Path(video_name).stem
    default = config.SPOTS_DIR / f"spots_{stem}.json"
    variants = sorted(config.SPOTS_DIR.glob(f"spots_{stem}_*.json"))

    result = []
    if default.exists():
        result.append(default)
    result.extend(variants)
    return result


# ---------------------------------------------------------------------------
# API endpoints
# NOTE: no "/" route here — the static mount at the bottom serves index.html.
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """
    Lightweight health check. Returns immediately without touching the lock
    or doing any work. Used by the frontend to detect "backend is alive but
    just slow" vs "backend is down."
    """
    with state_lock:
        return {
            "status": "ok",
            "is_running": state.is_running,
            "has_video": state.video_path is not None,
        }

@app.get("/videos")
def list_videos():
    """List videos in ml/yolo/videos/ that have a matching spots_*.json."""
    if not config.VIDEOS_DIR.exists():
        return {"videos": []}

    videos = []
    for video_file in sorted(config.VIDEOS_DIR.iterdir()):
        if video_file.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
            continue
        spots_file = _spots_path_for(video_file.name)
        if spots_file.exists():
            videos.append({
                "name": video_file.name,
                "spots_file": spots_file.name,
            })
    return {"videos": videos}


@app.post("/select-video")
def select_video(req: SelectVideoRequest):
    """Set the active video, render and cache a preview frame, return its URL."""
    video_path = config.VIDEOS_DIR / req.video_name
    spots_path = _spots_path_for(req.video_name)

    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"video not found: {req.video_name}")
    if not spots_path.exists():
        raise HTTPException(status_code=404, detail=f"spots file missing: {spots_path.name}")

    # Refuse to switch videos while detection is running. Force a stop first.
    with state_lock:
        if state.is_running:
            raise HTTPException(
                status_code=409,
                detail="detection is currently running, call /stop-detection first",
            )

    # Render the preview frame (raises on failure).
    try:
        preview_bytes = render_preview_frame(video_path, spots_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"preview render failed: {e}")

    with state_lock:
        state.video_path = video_path
        state.spots_path = spots_path
        state.reset_outputs()
        # Stash the preview so /live-frame has something to show before
        # detection starts.
        state.latest_frame_jpeg = preview_bytes

    return {
        "selected": req.video_name,
        "spots_file": spots_path.name,
        "preview_url": "/preview-frame",
    }


@app.get("/preview-frame")
def preview_frame():
    """Serve the most recently rendered preview frame as JPEG."""
    with state_lock:
        data = state.latest_frame_jpeg
    if data is None:
        raise HTTPException(status_code=404, detail="no preview frame available")
    return Response(content=data, media_type="image/jpeg")


@app.post("/start-detection")
def start_detection():
    """Start the inference worker in a background thread."""
    with state_lock:
        if state.video_path is None or state.spots_path is None:
            raise HTTPException(status_code=400, detail="no video selected")
        if state.is_running:
            raise HTTPException(status_code=409, detail="detection already running")

        state.is_running = True
        state.stop_requested = False
        video_path = state.video_path
        spots_path = state.spots_path

    thread = Thread(
        target=run_inference_loop,
        args=(video_path, spots_path),
        daemon=True,
        name="inference-worker",
    )
    with state_lock:
        state.worker_thread = thread
    thread.start()

    return {"status": "started", "video": video_path.name}


@app.post("/stop-detection")
def stop_detection():
    """Signal the worker to stop and wait for it to finish."""
    with state_lock:
        if not state.is_running:
            return {"status": "not_running"}
        state.stop_requested = True
        thread = state.worker_thread

    if thread is not None:
        thread.join(timeout=10.0)
        if thread.is_alive():
            raise HTTPException(status_code=500, detail="worker did not stop in time")

    with state_lock:
        state.worker_thread = None

    return {"status": "stopped"}

@app.get("/spots-files")
def list_spots_files():
    """
    List all spots files (default + variants) for the currently selected video.
    Returns empty list if no video selected.
    """
    with state_lock:
        video_path = state.video_path
        active_spots = state.spots_path

    if video_path is None:
        return {"video": None, "active": None, "files": []}

    files = _list_spots_files_for(video_path.name)
    return {
        "video": video_path.name,
        "active": active_spots.name if active_spots else None,
        "files": [f.name for f in files],
    }


@app.post("/switch-spots")
def switch_spots(req: SelectVideoRequest):
    """
    Switch the active spots layout. Reuses SelectVideoRequest schema since
    we just need a filename — the field is repurposed as the spots filename
    (e.g. "spots_demo_angled_test.json"), not a video name.

    Works whether detection is running or stopped:
      - Running: queues the switch, worker picks it up next iteration
      - Stopped: applies immediately so the next /start-detection uses it
    """
    spots_filename = req.video_name  # field is repurposed, see docstring
    spots_path = config.SPOTS_DIR / spots_filename

    if not spots_path.exists():
        raise HTTPException(status_code=404, detail=f"spots file not found: {spots_filename}")

    with state_lock:
        if state.video_path is None:
            raise HTTPException(status_code=400, detail="no video selected")

        # Sanity check: the spots file should belong to the active video.
        # We allow it through anyway (user might know what they're doing) but
        # it's a strong hint of a mistake.
        expected_prefix = f"spots_{state.video_path.stem}"
        if not spots_filename.startswith(expected_prefix):
            raise HTTPException(
                status_code=400,
                detail=f"spots file '{spots_filename}' does not match active video '{state.video_path.name}'",
            )

        state.spots_path = spots_path
        if state.is_running:
            state.pending_spots_path = spots_path

    return {
        "active_spots": spots_path.name,
        "queued_for_worker": state.is_running,
    }


@app.get("/parking-status")
def parking_status():
    """Latest JSON output from the inference worker."""
    with state_lock:
        status = dict(state.latest_status)  # shallow copy under lock
        running = state.is_running
    return {
        "is_running": running,
        "status": status,
    }


@app.get("/live-frame")
def live_frame():
    """Latest annotated frame as JPEG."""
    with state_lock:
        data = state.latest_frame_jpeg
    if data is None:
        raise HTTPException(status_code=404, detail="no frame available")
    return Response(content=data, media_type="image/jpeg")


@app.post("/shutdown")
def shutdown():
    """
    Nuclear option. Stops the worker if running, then exits the process.
    Not used by the demo flow — kept for completeness.
    """
    import os
    import signal

    with state_lock:
        if state.is_running:
            state.stop_requested = True
            thread = state.worker_thread
        else:
            thread = None

    if thread is not None:
        thread.join(timeout=5.0)

    os.kill(os.getpid(), signal.SIGINT)
    return {"status": "shutting_down"}


# ---------------------------------------------------------------------------
# Static frontend mount — MUST be declared LAST so it doesn't shadow the API
# endpoints above. Serves frontend/index.html at the root URL.
# ---------------------------------------------------------------------------
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount(
    "/",
    StaticFiles(directory=str(_FRONTEND_DIR), html=True),
    name="frontend",
)