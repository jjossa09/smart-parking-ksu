# Architecture — The "Config, Not Engineering" Story

The whole Phase 2 design is built around one idea: **when real KSU data arrives, we should not need to rewrite code.** Three pieces of the system are designed to be swapped out; the pipeline code itself is fixed.

## The three swappable pieces

| Piece | What it is | When to swap |
|---|---|---|
| **Model weights** (`.pt` file) | The brain that detects cars | New fine-tune, better public model, KSU-specific training |
| **Spot polygons** (`spots_*.json`) | Where the parking spots are | New camera angle, new lot, want to try tighter/looser annotation |
| **Test video** (`.mp4` file) | What the system watches | New footage, real KSU recording, different demo scenario |

All three live in known paths under `v2/ml/yolo/`:

```
ml/yolo/
├── weights/        <- .pt files
├── spots/          <- spots_*.json
└── videos/         <- .mp4 files
```

## The one fixed piece: the pipeline

The code in `ml/yolo/detect_and_assign.py`, `smoothing.py`, and the backend worker never changes when you swap the three pieces above. That's the whole point.

The pipeline runs the same four stages regardless of what data trained the model or what camera shot the video:

1. **Detection** — YOLO returns bounding boxes for vehicles
2. **Assignment** — each box claims its best-matching spot via IoU
3. **Smoothing** — per-spot rolling window kills single-frame flicker
4. **Output** — JSON + annotated frame, consumed by the backend

## What swapping actually looks like

**New model (e.g. fine-tuned on KSU footage):**
```python
# v2/ml/yolo/config.py
MODEL_WEIGHTS = WEIGHTS_DIR / "yolov8m_ksu.pt"   # was yolov8n.pt
```
Restart the backend. Done.

**New spots layout for the same video:**
Run `annotate_spots.py --variant foo`, then use the spots dropdown on the detection screen to switch between layouts live. No code touched, no restart.

**New video entirely:**
Drop the file in `videos/`, annotate once, reload the browser. New video appears in the dropdown.

## Why this matters for the presentation

The demo judges see is running COCO-pretrained YOLOv8n on `demo_angled.mp4`. **On demo day the pitch is:** "If KSU hands us footage tomorrow morning, we fine-tune on it by tomorrow afternoon, and the production model updates with a one-line config change. No architectural rewrite. No retraining the pipeline. Just swap the file."

That's not marketing — it's literally how the code is organized. See `ksu_data_integration.md` for the step-by-step runbook.