"""
Single source of truth for the YOLO inference pipeline.
Every other module imports paths, thresholds, and model settings from here.
Edit this file to swap models, videos, or tuning knobs — never hardcode elsewhere.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# Anchor everything to this file's location so the pipeline runs the same
# whether you launch it from the repo root or from ml/yolo/.
PROJECT_ROOT = Path(r"C:\Users\juanj\smart-parking-ksu\v2")
YOLO_DIR     = PROJECT_ROOT / "ml" / "yolo"

VIDEOS_DIR   = YOLO_DIR / "videos"
SPOTS_DIR    = YOLO_DIR / "spots"
WEIGHTS_DIR  = YOLO_DIR / "weights"
OUTPUTS_DIR  = YOLO_DIR / "outputs"

# Make sure output dirs exist at import time so nothing downstream has to care.
for d in (SPOTS_DIR, WEIGHTS_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Day 1: COCO-pretrained YOLOv8n. Ultralytics auto-downloads on first use.
# Phase D: replace with fine-tuned PKLot weights — only this line changes.
MODEL_WEIGHTS = WEIGHTS_DIR / "yolov8n.pt"

# COCO class IDs we care about. Cars dominate; trucks/buses catch SUVs and
# the occasional misclassification. Motorcycles intentionally excluded —
# they don't occupy car spots in the way we want to count.
VEHICLE_CLASS_IDS = {2, 5, 7}  # car, bus, truck

# YOLO detection confidence floor. A box must score above this to count.
# Lower = catches more cars but more false positives (shadows, posts, etc).
# Higher = cleaner output but misses distant or partially-occluded cars.
# 0.35 is the tuned value for COCO YOLOv8n on angled parking footage.
# If you swap to a different model, expect to retune — fine-tuned models
# often output systematically lower confidences and need ~0.15–0.25.
DETECTION_CONFIDENCE = 0.35

# ---------------------------------------------------------------------------
# Spot assignment
# ---------------------------------------------------------------------------
# Minimum IoU between a detection's bbox and a spot polygon for that
# detection to be a candidate for that spot. Looks low, but is correct
# for angled footage: a car's axis-aligned bbox extends well outside its
# actual angled-trapezoid spot, so true overlap rarely exceeds 0.20–0.30.
# Raise this if you see false positives (cars in the aisle being assigned
# to spots). Lower it if cars in real spots aren't being claimed.
SPOT_IOU_THRESHOLD = 0.15

# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------
# Rolling window length per spot for temporal smoothing. A spot's committed
# state only flips when the last K raw observations all agree.
# K=5 with FRAME_SKIP=5 means a state change requires ~25 video frames of
# agreement (roughly 1 second of video at 30fps). Tradeoff:
#   Lower K → faster reaction to real changes, more flicker
#   Higher K → rock-solid stability, slower reaction
# 5 is the tuned value for the demo: visible flicker eliminated, state
# changes still feel responsive when a car arrives or leaves.
SMOOTHING_WINDOW_K = 5

# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------
# Day 1 active video. Swap to demo_overhead.mp4 once the angled one works.
ACTIVE_VIDEO = VIDEOS_DIR / "demo_angled.mp4"

# Spots file paired with the active video. Convention: spots_<video_stem>.json
ACTIVE_SPOTS = SPOTS_DIR / f"spots_{ACTIVE_VIDEO.stem}.json"

# Process every Nth frame. CPU inference can't keep up with real-time on
# YOLOv8n, and parking changes slowly anyway, so we skip frames aggressively.
# 5 = process every 5th frame (~1.2 fps on CPU, plenty for parking).
# If you have a GPU, you can drop this to 2 or 1 for smoother visuals.
# If your CPU is slow and the worker falls behind, raise to 10.
FRAME_SKIP = 5

# Loop the video forever for the demo so judges always see motion.
LOOP_VIDEO = True

# ---------------------------------------------------------------------------
# Output artifacts (what the backend/frontend will eventually consume)
# ---------------------------------------------------------------------------
OUTPUT_JSON  = OUTPUTS_DIR / "prediction_output.json"
OUTPUT_FRAME = OUTPUTS_DIR / "annotated_frame.jpg"

# ---------------------------------------------------------------------------
# Drawing (used by utils.py)
# ---------------------------------------------------------------------------
COLOR_EMPTY    = (0, 200, 0)      # BGR green
COLOR_OCCUPIED = (0, 0, 220)      # BGR red
COLOR_BBOX     = (255, 200, 0)    # BGR cyan-ish for car bboxes
POLYGON_THICKNESS = 2
BBOX_THICKNESS    = 2