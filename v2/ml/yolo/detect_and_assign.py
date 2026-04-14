"""
Core detection + assignment logic. Pure frame-in, status-out.
No file I/O, no video loop — that's infer_video.py's job.
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

from . import config
from .utils import bbox_polygon_iou, BBox, Polygon


# Cache the model so repeated calls don't reload weights.
_model: YOLO | None = None


def load_model(weights_path: Path = None) -> YOLO:
    """Load YOLO weights once and cache. Ultralytics auto-downloads yolov8n.pt if missing."""
    global _model
    if _model is None:
        path = weights_path or config.MODEL_WEIGHTS
        # If the local cached weights don't exist yet, fall back to the
        # ultralytics short name so it auto-downloads to cwd, then we move on.
        if not path.exists():
            print(f"[detect] weights not found at {path}, using 'yolov8n.pt' (auto-download)")
            _model = YOLO("yolov8n.pt")
        else:
            _model = YOLO(str(path))
    return _model


def load_spots(spots_path: Path = None) -> List[Polygon]:
    """Load spot polygons from JSON. Returns list of [(x,y), ...] tuples."""
    path = spots_path or config.ACTIVE_SPOTS
    if not path.exists():
        raise FileNotFoundError(
            f"No spots file at {path}. Run annotate_spots.py first."
        )
    payload = json.loads(path.read_text())
    return [[tuple(pt) for pt in poly] for poly in payload["spots"]]


def detect_vehicles(frame: np.ndarray) -> List[Tuple[BBox, float, int]]:
    """
    Run YOLO on a single frame, return only vehicle detections above threshold.

    Returns a list of (bbox, confidence, class_id) tuples. Only boxes whose
    class_id is in config.VEHICLE_CLASS_IDS (car/bus/truck by default) pass
    the filter. Callers don't need to know which model or which class IDs —
    that's all handled here.
    """
    model = load_model()
    results = model.predict(
        frame,
        conf=config.DETECTION_CONFIDENCE,
        verbose=False,
    )

    detections: List[Tuple[BBox, float, int]] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls.item())
            if cls_id not in config.VEHICLE_CLASS_IDS:
                continue
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            bbox: BBox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
            detections.append((bbox, conf, cls_id))
    return detections


def assign_detections_to_spots(
    detections: List[Tuple[BBox, float, int]],
    spots: List[Polygon],
) -> List[int]:
    """
    Match each detection to its single best-overlapping parking spot.

    Uses detection-to-spot assignment (not spot-to-detection): each car
    claims one spot, the one with highest IoU above SPOT_IOU_THRESHOLD.
    If two cars fight over the same spot, the higher-IoU one wins.

    Returns a list parallel to `spots` where 1 = occupied and 0 = empty.
    This is the raw per-frame result — pipe it through SpotSmoother before
    showing it to users.
    """
    n_spots = len(spots)
    statuses = [0] * n_spots
    best_iou_per_spot = [0.0] * n_spots

    for bbox, _conf, _cls in detections:
        best_spot = -1
        best_iou = config.SPOT_IOU_THRESHOLD  # must beat threshold to count
        for i, poly in enumerate(spots):
            iou = bbox_polygon_iou(bbox, poly)
            if iou > best_iou:
                best_iou = iou
                best_spot = i
        if best_spot >= 0 and best_iou > best_iou_per_spot[best_spot]:
            statuses[best_spot] = 1
            best_iou_per_spot[best_spot] = best_iou

    return statuses


def process_frame(
    frame: np.ndarray,
    spots: List[Polygon],
) -> Tuple[List[int], List[Tuple[BBox, float, int]]]:
    """
    Convenience wrapper: detect vehicles + assign to spots in one call.

    Returns (raw_statuses, detections) — statuses for the smoother,
    detections for drawing bounding boxes on the annotated frame.
    """
    detections = detect_vehicles(frame)
    statuses = assign_detections_to_spots(detections, spots)
    return statuses, detections