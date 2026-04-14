"""
Small reusable helpers for the YOLO pipeline.
Geometry (IoU, point-in-polygon) and drawing (polygons, bboxes).
No ML, no I/O — pure functions so they're trivial to test.
"""

from typing import List, Tuple
import numpy as np
import cv2

from . import config

BBox = Tuple[int, int, int, int]   # (x1, y1, x2, y2)
Polygon = List[Tuple[int, int]]    # list of (x, y) vertices


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def polygon_area(polygon: Polygon) -> float:
    """Shoelace formula. Returns absolute area in pixels^2."""
    pts = np.array(polygon, dtype=np.float32)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def bbox_polygon_iou(bbox: BBox, polygon: Polygon) -> float:
    """
    IoU between an axis-aligned bbox and an arbitrary polygon.
    Computed by rasterizing both into a tight mask and counting pixels —
    slower than analytic intersection but bulletproof for any polygon shape
    and fast enough at parking-lot scale (a few dozen spots per frame).
    """
    x1, y1, x2, y2 = bbox
    poly = np.array(polygon, dtype=np.int32)

    # Tight bounding region covering both shapes.
    px1 = min(x1, poly[:, 0].min())
    py1 = min(y1, poly[:, 1].min())
    px2 = max(x2, poly[:, 0].max())
    py2 = max(y2, poly[:, 1].max())

    w = max(1, px2 - px1 + 1)
    h = max(1, py2 - py1 + 1)

    bbox_mask = np.zeros((h, w), dtype=np.uint8)
    poly_mask = np.zeros((h, w), dtype=np.uint8)

    # Shift coordinates into the local mask space.
    cv2.rectangle(
        bbox_mask,
        (x1 - px1, y1 - py1),
        (x2 - px1, y2 - py1),
        color=1,
        thickness=-1,
    )
    shifted_poly = poly.copy()
    shifted_poly[:, 0] -= px1
    shifted_poly[:, 1] -= py1
    cv2.fillPoly(poly_mask, [shifted_poly], color=1)

    intersection = int(np.logical_and(bbox_mask, poly_mask).sum())
    union = int(np.logical_or(bbox_mask, poly_mask).sum())
    if union == 0:
        return 0.0
    return intersection / union


def point_in_polygon(point: Tuple[int, int], polygon: Polygon) -> bool:
    """OpenCV's pointPolygonTest. Returns True if point is inside or on edge."""
    poly = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly, point, measureDist=False) >= 0


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_polygon(
    frame: np.ndarray,
    polygon: Polygon,
    color: Tuple[int, int, int],
    thickness: int = config.POLYGON_THICKNESS,
    label: str = None,
) -> None:
    """Draw a closed polygon onto frame in place. Optional text label at centroid."""
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    if label is not None:
        cx = int(np.mean([p[0] for p in polygon]))
        cy = int(np.mean([p[1] for p in polygon]))
        cv2.putText(
            frame, label, (cx - 10, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )


def draw_bbox(
    frame: np.ndarray,
    bbox: BBox,
    color: Tuple[int, int, int] = config.COLOR_BBOX,
    thickness: int = config.BBOX_THICKNESS,
    label: str = None,
) -> None:
    """Draw a bounding box onto frame in place."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label is not None:
        cv2.putText(
            frame, label, (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )


def draw_hud(
    frame: np.ndarray,
    total: int,
    available: int,
    occupied: int,
) -> None:
    """Top-left overlay with the live counts."""
    lines = [
        f"Total:     {total}",
        f"Available: {available}",
        f"Occupied:  {occupied}",
    ]
    pad = 8
    line_h = 22
    box_w = 180
    box_h = pad * 2 + line_h * len(lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, text in enumerate(lines):
        cv2.putText(
            frame, text, (10 + pad, 10 + pad + line_h * (i + 1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )