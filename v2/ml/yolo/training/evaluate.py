# ml/yolo/training/evaluate.py
"""
Evaluate a YOLO model against hand-labeled ground truth on a real video.

Loads a model, runs detection + spot assignment on each labeled frame,
compares predicted occupancy to ground truth, prints accuracy, precision,
recall, and a confusion matrix. Also lists the worst frames for visual
debugging.

Usage from project root:
    python -m ml.yolo.training.evaluate ^
        --model ml\yolo\weights\yolov8m_pklot.pt ^
        --video ml\yolo\videos\demo_angled.mp4 ^
        --gt    ml\yolo\eval\ground_truth_demo_angled.json
"""

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from ml.yolo import config
from ml.yolo.detect_and_assign import (
    assign_detections_to_spots,
    load_spots,
)
import ml.yolo.detect_and_assign as da


# Reuse the same vehicle classes from config so eval matches production.
VEHICLE_CLASS_IDS = config.VEHICLE_CLASS_IDS


def detect_with(model: YOLO, frame, vehicle_class_ids: set[int]):
    """
    Run inference with a SPECIFIC model instance (not the cached one in
    detect_and_assign). Returns list of (bbox, conf, cls).

    For the PKLot fine-tuned model, vehicle_class_ids is {1} (space-occupied).
    For COCO models, it's {2, 5, 7} (car, bus, truck).
    """
    results = model.predict(frame, conf=config.DETECTION_CONFIDENCE, verbose=False)
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls.item())
            if cls_id not in vehicle_class_ids:
                continue
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
            detections.append((bbox, conf, cls_id))
    return detections


def is_pklot_model(model: YOLO) -> bool:
    """Heuristic: PKLot fine-tune has 2 classes (space-empty, space-occupied)."""
    names = model.names
    return len(names) <= 5 and any("space" in str(n).lower() for n in names.values())


def evaluate(model_path: Path, video_path: Path, gt_path: Path, verbose: bool = True):
    print(f"\n[eval] model: {model_path.name}")
    print(f"[eval] video: {video_path.name}")
    print(f"[eval] gt:    {gt_path.name}")

    model = YOLO(str(model_path))

    # Decide which class IDs mean "occupied vehicle" for THIS model.
    if is_pklot_model(model):
        # PKLot: class 1 = space-occupied. We do NOT use class 0 (space-empty)
        # because we're detecting where cars are, not where empty spots are.
        vehicle_class_ids = {1}
        print(f"[eval] detected PKLot model, using class IDs: {vehicle_class_ids}")
        print(f"[eval] model classes: {model.names}")
    else:
        vehicle_class_ids = VEHICLE_CLASS_IDS
        print(f"[eval] detected COCO model, using class IDs: {vehicle_class_ids}")

    spots_path = config.SPOTS_DIR / f"spots_{video_path.stem}.json"
    spots = load_spots(spots_path)
    n_spots = len(spots)

    ground_truth = json.loads(gt_path.read_text())
    print(f"[eval] {len(ground_truth)} labeled frames, {n_spots} spots each")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    tp = fp = tn = fn = 0
    total_correct = 0
    total_compared = 0
    per_frame_errors = []  # (frame_idx, n_errors)

    for key in sorted(ground_truth.keys(), key=lambda k: int(k.split("_")[1])):
        frame_idx = int(key.split("_")[1])
        actual = ground_truth[key]

        if len(actual) != n_spots:
            print(f"  ! frame {frame_idx}: gt has {len(actual)} spots, expected {n_spots}, skipping")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"  ! could not read frame {frame_idx}")
            continue

        detections = detect_with(model, frame, vehicle_class_ids)
        predicted = assign_detections_to_spots(detections, spots)

        frame_errors = 0
        for p, a in zip(predicted, actual):
            total_compared += 1
            if p == a:
                total_correct += 1
            else:
                frame_errors += 1
            if   p == 1 and a == 1: tp += 1
            elif p == 1 and a == 0: fp += 1
            elif p == 0 and a == 0: tn += 1
            elif p == 0 and a == 1: fn += 1
        per_frame_errors.append((frame_idx, frame_errors))

    cap.release()

    accuracy  = total_correct / total_compared if total_compared else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("\n" + "=" * 60)
    print(f"RESULTS: {model_path.name} on {video_path.name}")
    print("=" * 60)
    print(f"  frames evaluated:     {len(per_frame_errors)}")
    print(f"  total spot decisions: {total_compared}")
    print(f"  accuracy:             {accuracy * 100:.2f}%")
    print(f"  precision (occupied): {precision * 100:.2f}%")
    print(f"  recall    (occupied): {recall * 100:.2f}%")
    print(f"  f1 score:             {f1 * 100:.2f}%")
    print(f"\n  confusion matrix:")
    print(f"    true positives  (correctly occupied):  {tp}")
    print(f"    true negatives  (correctly empty):     {tn}")
    print(f"    false positives (said occupied, was empty):  {fp}  ← over-detecting")
    print(f"    false negatives (said empty, was occupied):  {fn}  ← missing cars")

    if verbose and per_frame_errors:
        per_frame_errors.sort(key=lambda x: -x[1])
        print(f"\n  worst 5 frames (most errors):")
        for fi, ne in per_frame_errors[:5]:
            print(f"    frame {fi:5d}: {ne}/{n_spots} wrong")
    print()

    return {
        "model": model_path.name,
        "video": video_path.name,
        "frames": len(per_frame_errors),
        "decisions": total_compared,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--gt",    type=str, required=True)
    args = parser.parse_args()

    evaluate(
        model_path=Path(args.model).resolve(),
        video_path=Path(args.video).resolve(),
        gt_path=Path(args.gt).resolve(),
    )


if __name__ == "__main__":
    main()