"""
Background inference worker. One thread, one video, in-memory outputs only.

Reuses the Day 1 modules unchanged:
    ml.yolo.detect_and_assign.process_frame
    ml.yolo.detect_and_assign.load_spots
    ml.yolo.smoothing.SpotSmoother
    ml.yolo.utils.draw_polygon / draw_bbox / draw_hud
    ml.yolo.config (for thresholds, frame skip, colors)

Writes results into backend.state, never to disk.
"""

import time
from pathlib import Path

import cv2

from ml.yolo import config
from ml.yolo.detect_and_assign import load_spots, process_frame
from ml.yolo.smoothing import SpotSmoother
from ml.yolo.utils import draw_polygon, draw_bbox, draw_hud

from .state import state, state_lock


def _build_status_payload(frame_index: int, timestamp_sec: float, smoothed: list[int]) -> dict:
    """Same JSON contract as Day 1's prediction_output.json."""
    return {
        "frame_index": frame_index,
        "timestamp_sec": round(timestamp_sec, 3),
        "total_spots": len(smoothed),
        "available_spots": sum(1 for s in smoothed if s == 0),
        "occupied_spots": sum(1 for s in smoothed if s == 1),
        "spots": smoothed,
    }


def _annotate_frame(frame, spots, smoothed, detections, payload: dict):
    """Draw polygons + bboxes + HUD onto a copy of the frame. Returns the copy."""
    annotated = frame.copy()
    for i, poly in enumerate(spots):
        color = config.COLOR_OCCUPIED if smoothed[i] == 1 else config.COLOR_EMPTY
        draw_polygon(annotated, poly, color, label=str(i))
    for bbox, conf, _cls in detections:
        draw_bbox(annotated, bbox, label=f"{conf:.2f}")
    draw_hud(
        annotated,
        total=payload["total_spots"],
        available=payload["available_spots"],
        occupied=payload["occupied_spots"],
    )
    return annotated


def run_inference_loop(video_path: Path, spots_path: Path) -> None:
    """
    The actual worker body. Runs in a background thread.

    NOTE: We import / load the YOLO model lazily inside this function (via
    process_frame -> load_model) so the model lives on the worker thread,
    not the main thread. Avoids edge-case torch threading weirdness.
    """
    print(f"[worker] starting on video={video_path.name} spots={spots_path.name}")

    spots = load_spots(spots_path)
    smoother = SpotSmoother(n_spots=len(spots))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[worker] ERROR: could not open {video_path}")
        with state_lock:
            state.is_running = False
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[worker] opened {video_path.name} fps={fps:.1f} frame_skip={config.FRAME_SKIP}")

    frame_index = 0
    processed = 0
    t0 = time.time()

    try:
        while True:
            # Check stop flag and pending spots-switch every iteration.
            with state_lock:
                if state.stop_requested:
                    print("[worker] stop requested, exiting loop")
                    break
                pending = state.pending_spots_path
                if pending is not None:
                    state.pending_spots_path = None

            # If a spots-switch is pending, reload polygons and reset the smoother.
            # This happens outside the lock because file I/O is slow.
            if pending is not None:
                try:
                    spots = load_spots(pending)
                    smoother = SpotSmoother(n_spots=len(spots))
                    print(f"[worker] switched to {pending.name} ({len(spots)} spots)")
                except Exception as e:
                    print(f"[worker] failed to switch spots: {e}")

            ok, frame = cap.read()
            if not ok:
                # Loop the video forever, same as Day 1.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_index = 0
                continue

            if frame_index % config.FRAME_SKIP == 0:
                raw, detections = process_frame(frame, spots)
                smoothed = smoother.update(raw)
                timestamp_sec = frame_index / fps

                payload = _build_status_payload(frame_index, timestamp_sec, smoothed)
                annotated = _annotate_frame(frame, spots, smoothed, detections, payload)

                # Encode once here, store bytes. The endpoint just serves them.
                ok_enc, buf = cv2.imencode(".jpg", annotated)
                jpeg_bytes = buf.tobytes() if ok_enc else None

                # Tiny critical section: just two field assignments.
                with state_lock:
                    state.latest_status = payload
                    if jpeg_bytes is not None:
                        state.latest_frame_jpeg = jpeg_bytes

                processed += 1
                if processed % 10 == 0:
                    elapsed = time.time() - t0
                    eff_fps = processed / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[worker] processed={processed} frame={frame_index} "
                        f"avail={payload['available_spots']}/{payload['total_spots']} "
                        f"speed={eff_fps:.1f} proc-fps"
                    )

            frame_index += 1
    finally:
        cap.release()
        with state_lock:
            state.is_running = False
            state.stop_requested = False
        print("[worker] stopped cleanly")


def render_preview_frame(video_path: Path, spots_path: Path) -> bytes:
    """
    Grab the first frame of the video, draw the spot polygons on it (no cars,
    no detection), return JPEG bytes. Used by POST /select-video so the user
    sees their annotation before starting full inference.
    """
    spots = load_spots(spots_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read first frame of {video_path}")

    for i, poly in enumerate(spots):
        draw_polygon(frame, poly, config.COLOR_EMPTY, label=str(i))

    ok_enc, buf = cv2.imencode(".jpg", frame)
    if not ok_enc:
        raise RuntimeError("failed to encode preview frame")
    return buf.tobytes()