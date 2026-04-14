"""
CLI tool to define parking spot polygons for a video.

Usage (from project root):
    python -m ml.yolo.annotate_spots --video ml/yolo/videos/demo_angled.mp4

Controls:
    Left click       Add a vertex to the current spot (need 4)
    n                Save current spot, start a new one
    u                Undo last vertex
    r                Reset current spot
    s                Save all spots to spots/spots_<video_stem>.json and exit
    q                Quit without saving
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import config
from .utils import draw_polygon


VERTICES_PER_SPOT = 4


def grab_first_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from: {video_path}")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Optional variant suffix. Default file: spots_<stem>.json. "
             "With variant 'foo': spots_<stem>_foo.json",
    )
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    base_frame = grab_first_frame(video_path)

    spots: list[list[list[int]]] = []     # committed spots
    current: list[list[int]] = []         # vertices for the spot currently being drawn

    window = "Annotate spots — n=next  u=undo  r=reset  s=save  q=quit"

    def on_mouse(event, x, y, flags, param):
        nonlocal current
        if event == cv2.EVENT_LBUTTONDOWN and len(current) < VERTICES_PER_SPOT:
            current.append([x, y])

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    print(f"[annotate] Loaded {video_path.name}")
    print(f"[annotate] Click {VERTICES_PER_SPOT} corners per spot, then press 'n'.")

    while True:
        display = base_frame.copy()

        # Draw committed spots in green with their index.
        for i, poly in enumerate(spots):
            draw_polygon(display, [tuple(p) for p in poly], (0, 200, 0), label=str(i))

        # Draw the in-progress spot in yellow.
        for (x, y) in current:
            cv2.circle(display, (x, y), 4, (0, 255, 255), -1)
        if len(current) >= 2:
            cv2.polylines(
                display,
                [np.array(current, dtype=np.int32)],
                isClosed=(len(current) == VERTICES_PER_SPOT),
                color=(0, 255, 255),
                thickness=2,
            )

        cv2.putText(
            display,
            f"spots saved: {len(spots)}   current vertices: {len(current)}/{VERTICES_PER_SPOT}",
            (10, display.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("n"):
            if len(current) == VERTICES_PER_SPOT:
                spots.append(current)
                current = []
                print(f"[annotate] saved spot {len(spots) - 1}")
            else:
                print(f"[annotate] need {VERTICES_PER_SPOT} vertices, have {len(current)}")
        elif key == ord("u"):
            if current:
                current.pop()
        elif key == ord("r"):
            current = []
        elif key == ord("s"):
            if len(current) == VERTICES_PER_SPOT:
                spots.append(current)
                current = []
            # Build the base filename. If a variant was given, include it.
            if args.variant:
                base = f"spots_{video_path.stem}_{args.variant}"
            else:
                base = f"spots_{video_path.stem}"

            # Never silently overwrite an existing file — append a counter
            # until we find a free name. spots_foo.json, spots_foo_1.json, etc.
            out_path = config.SPOTS_DIR / f"{base}.json"
            counter = 1
            while out_path.exists():
                out_path = config.SPOTS_DIR / f"{base}_{counter}.json"
                counter += 1
            payload = {
                "video": video_path.name,
                "frame_size": [base_frame.shape[1], base_frame.shape[0]],
                "spots": spots,
            }
            out_path.write_text(json.dumps(payload, indent=2))
            print(f"[annotate] wrote {len(spots)} spots to {out_path}")
            break
        elif key == ord("q"):
            print("[annotate] quit without saving")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()