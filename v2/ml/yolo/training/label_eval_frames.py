"""
Hand-label ground-truth spot states for a video, for use by evaluate.py.

Samples N evenly-spaced frames from the video, shows each one with the
spot polygons numbered, and asks you to type the indices of OCCUPIED spots
(comma-separated). Anything you don't type is recorded as empty.

Output: ml/yolo/eval/ground_truth_<video_stem>.json

Usage from project root:
    python -m ml.yolo.training.label_eval_frames --video ml\yolo\videos\demo_angled.mp4 --n 30

Controls during labeling:
    - The frame appears in an OpenCV window with numbered polygons
    - Switch to the terminal, type occupied indices like:  0,3,5,12,18
    - Press Enter to save and advance to the next frame
    - Type 'skip' to skip a frame (won't be in ground truth)
    - Type 'quit' to stop early (saves what you've labeled so far)
    - Type 'back' to redo the previous frame
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ml.yolo import config
from ml.yolo.detect_and_assign import load_spots
from ml.yolo.utils import draw_polygon


EVAL_DIR = config.YOLO_DIR / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def parse_indices(text: str, n_spots: int) -> list[int] | None:
    """Parse '0,3,5,12' into [0,3,5,12]. Returns None on error."""
    text = text.strip()
    if not text:
        return []
    try:
        indices = [int(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError:
        return None
    for i in indices:
        if i < 0 or i >= n_spots:
            print(f"  ! index {i} out of range [0, {n_spots - 1}]")
            return None
    return sorted(set(indices))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--n", type=int, default=30, help="Number of frames to sample")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    spots_path = config.SPOTS_DIR / f"spots_{video_path.stem}.json"
    if not spots_path.exists():
        raise FileNotFoundError(f"No spots file at {spots_path}")

    spots = load_spots(spots_path)
    n_spots = len(spots)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError("Could not read frame count")

    # Evenly spaced frame indices.
    sample_indices = np.linspace(0, total_frames - 1, args.n, dtype=int).tolist()
    print(f"[label] video: {video_path.name}")
    print(f"[label] total frames: {total_frames}, sampling {len(sample_indices)}")
    print(f"[label] {n_spots} spots per frame")
    print(f"[label] convention: type OCCUPIED indices comma-separated, blank = all empty")
    print(f"[label] commands: 'skip', 'back', 'quit'")
    print()

    out_path = EVAL_DIR / f"ground_truth_{video_path.stem}.json"
    ground_truth: dict[str, list[int]] = {}

    # Resume support: if file already exists, load and skip already-labeled frames.
    if out_path.exists():
        ground_truth = json.loads(out_path.read_text())
        print(f"[label] resuming — {len(ground_truth)} frames already labeled\n")

    window = "Label frame — switch to terminal to type"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    i = 0
    while i < len(sample_indices):
        frame_idx = sample_indices[i]
        key = f"frame_{frame_idx}"

        if key in ground_truth:
            i += 1
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"  ! could not read frame {frame_idx}, skipping")
            i += 1
            continue

        # Draw polygons with their index number, all in yellow so you don't get
        # primed by green/red predictions.
        display = frame.copy()
        for idx, poly in enumerate(spots):
            draw_polygon(display, poly, (0, 255, 255), label=str(idx))

        cv2.imshow(window, display)
        cv2.waitKey(1)  # let the window actually render

        prompt = f"[{i + 1}/{len(sample_indices)}] frame {frame_idx} — occupied indices: "
        text = input(prompt)

        if text.strip().lower() == "quit":
            print("[label] quitting early")
            break
        if text.strip().lower() == "skip":
            i += 1
            continue
        if text.strip().lower() == "back":
            if i > 0:
                i -= 1
                prev_key = f"frame_{sample_indices[i]}"
                if prev_key in ground_truth:
                    del ground_truth[prev_key]
            continue

        occupied = parse_indices(text, n_spots)
        if occupied is None:
            print("  ! could not parse, try again")
            continue

        # Build the per-spot 0/1 list.
        states = [0] * n_spots
        for idx in occupied:
            states[idx] = 1
        ground_truth[key] = states
        print(f"  saved: {sum(states)}/{n_spots} occupied")

        # Save after every frame so an interrupt doesn't lose progress.
        out_path.write_text(json.dumps(ground_truth, indent=2))
        i += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[label] done. {len(ground_truth)} frames saved to {out_path}")


if __name__ == "__main__":
    main()