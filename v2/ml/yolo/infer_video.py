"""
End-to-end runner for the Day 1 pipeline.

Reads ACTIVE_VIDEO from config, loops it forever (or once), runs YOLO every
FRAME_SKIP frames, assigns detections to spots, smooths, and writes:

    OUTPUT_JSON   — latest parking status (the contract for the frontend)
    OUTPUT_FRAME  — latest annotated JPEG (what the live feed will show)

Usage (from project root):
    python -m ml.yolo.infer_video
"""

import json
import time
from pathlib import Path

import cv2

from . import config
from .detect_and_assign import load_spots, process_frame
from .smoothing import SpotSmoother
from .utils import draw_polygon, draw_bbox, draw_hud


def write_outputs(
    frame,
    frame_index: int,
    timestamp_sec: float,
    smoothed: list[int],
    spots,
    detections,
) -> None:
    # JSON contract — this is what the backend/frontend will read on Day 2+.
    payload = {
        "frame_index": frame_index,
        "timestamp_sec": round(timestamp_sec, 3),
        "total_spots": len(smoothed),
        "available_spots": sum(1 for s in smoothed if s == 0),
        "occupied_spots": sum(1 for s in smoothed if s == 1),
        "spots": smoothed,
    }
    config.OUTPUT_JSON.write_text(json.dumps(payload, indent=2))

    # Annotated frame: polygons colored by smoothed state, bboxes for detections, HUD.
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
    cv2.imwrite(str(config.OUTPUT_FRAME), annotated)


def main():
    video_path: Path = config.ACTIVE_VIDEO
    if not video_path.exists():
        raise FileNotFoundError(f"ACTIVE_VIDEO not found: {video_path}")

    spots = load_spots(config.ACTIVE_SPOTS)
    print(f"[infer] loaded {len(spots)} spots from {config.ACTIVE_SPOTS.name}")

    smoother = SpotSmoother(n_spots=len(spots))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[infer] opened {video_path.name}  fps={fps:.1f}  frame_skip={config.FRAME_SKIP}")

    frame_index = 0
    processed = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if config.LOOP_VIDEO:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_index = 0
                    continue
                break

            if frame_index % config.FRAME_SKIP == 0:
                raw, detections = process_frame(frame, spots)
                smoothed = smoother.update(raw)
                timestamp_sec = frame_index / fps
                write_outputs(frame, frame_index, timestamp_sec, smoothed, spots, detections)

                processed += 1
                if processed % 10 == 0:
                    elapsed = time.time() - t0
                    eff_fps = processed / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[infer] processed={processed}  frame={frame_index}  "
                        f"avail={sum(1 for s in smoothed if s == 0)}/{len(smoothed)}  "
                        f"speed={eff_fps:.1f} proc-fps"
                    )

            frame_index += 1
    except KeyboardInterrupt:
        print("\n[infer] stopped by user")
    finally:
        cap.release()
        print(f"[infer] done. outputs at:\n  {config.OUTPUT_JSON}\n  {config.OUTPUT_FRAME}")


if __name__ == "__main__":
    main()