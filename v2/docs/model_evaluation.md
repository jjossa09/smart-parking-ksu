# Model Evaluation — COCO Baseline vs. PKLot Fine-Tune

**Production decision:** Ship YOLOv8n COCO baseline. Keep PKLot fine-tuned weights as a starting point for future KSU-specific fine-tuning.

---

## TL;DR

We compared a YOLOv8n model with COCO pretrained weights against a YOLOv8m model fine-tuned on the PKLot parking dataset. The fine-tuned model achieved **99.5% mAP50** on PKLot's own validation set but **failed almost completely** on our real demo footage — producing essentially zero detections. The COCO baseline, by contrast, achieved **94.6% per-spot accuracy** on the same footage. We ship the COCO baseline for the demo and treat the PKLot fine-tune as evidence that public parking datasets do not transfer cleanly to arbitrary camera angles — exactly the gap that real KSU footage will close in a future fine-tuning pass.

---

## Methodology

### Test footage
- **Video:** `demo_angled.mp4` — angled (non-overhead) view of a parking lot, 29 manually annotated parking spot polygons
- This footage is representative of what we expect from a real KSU surveillance camera: angled perspective, varied lighting, real cars, partial occlusions

### Ground truth
- 30 frames sampled at evenly-spaced intervals across the video
- Each frame hand-labeled by a human annotator: for each of the 29 spots, recorded as either occupied (1) or empty (0)
- Total ground truth decisions: **870** (30 frames × 29 spots)
- Ground truth file: `ml/yolo/eval/ground_truth_demo_angled.json`
- Convention: a spot is "occupied" if any vehicle is visibly inside it (matching the IoU-based assignment logic used in production)

### Models compared

| Model | Architecture | Training data | Source |
|---|---|---|---|
| Baseline | YOLOv8n | COCO (general-purpose, 80 classes) | Ultralytics auto-download |
| Fine-tuned | YOLOv8m | PKLot via Roboflow Universe, 24 epochs from COCO weights | Trained on Google Colab T4 GPU |

The fine-tuning achieved excellent in-distribution metrics on PKLot's own validation set:
- **mAP50: 0.995**
- **mAP50-95: 0.991**

These numbers are near the theoretical ceiling for the dataset.

### Evaluation procedure
For each labeled frame, run the model, assign detections to spot polygons via IoU, and compare predicted occupancy to ground truth. Aggregate over all frames. Report per-spot accuracy, precision, recall, F1, and confusion matrix. Implementation: `ml/yolo/training/evaluate.py`.

---

## Results

### COCO baseline (YOLOv8n) on `demo_angled.mp4`

| Metric | Value |
|---|---|
| Frames evaluated | 30 |
| Total spot decisions | 870 |
| **Accuracy** | **94.60%** |
| Precision (occupied class) | 99.84% |
| Recall (occupied class) | 93.32% |
| F1 score | 96.47% |

**Confusion matrix:**
- True positives (correctly marked occupied): 643
- True negatives (correctly marked empty): 180
- False positives (marked occupied, was empty): 1
- False negatives (marked empty, was actually occupied): 46

**Failure mode:** the COCO model very rarely hallucinates cars (1 false positive in 870 decisions), but it occasionally misses real ones — typically distant or partially occluded vehicles. This is the expected failure profile for a general-purpose detector applied to a specific lot at distance.

### PKLot fine-tune (YOLOv8m) on `demo_angled.mp4`

| Metric | Value |
|---|---|
| Frames evaluated | 30 |
| Total spot decisions | 870 |
| **Accuracy** | **20.80%** |
| Precision (occupied class) | 0.00% |
| Recall (occupied class) | 0.00% |
| F1 score | 0.00% |

**Confusion matrix:**
- True positives: 0
- True negatives: 181
- False positives: 0
- False negatives: 689

**Failure mode:** total non-detection. A diagnostic single-frame test confirmed that the model produces only 2–3 low-confidence detections (all below 0.06) on a frame that contains roughly 25 visible cars. The 20.80% accuracy figure is purely the result of guessing "all empty" on a lot where ~80% of spots are occupied.

---

## Interpretation: distribution shift

The fine-tuned model's collapse on real footage despite near-perfect PKLot validation scores is a textbook example of **distribution shift** (also called "out-of-distribution failure") in transfer learning.

PKLot is a fixed dataset shot from three specific cameras at three specific Brazilian parking lots. All images share:
- A roughly overhead camera angle
- Consistent (Brazilian outdoor) lighting
- The same lot geometry and painted line patterns
- A specific distribution of car models and colors

During fine-tuning, the YOLOv8m model became excellent at recognizing parking spots **that look like PKLot images**. It did not learn a general concept of "occupied parking spot" — it learned a specific concept of "PKLot-style occupied parking spot."

Our `demo_angled.mp4` is shot from a markedly different angle, in different lighting, of a differently-shaped lot, with different cars. To the fine-tuned model, this footage is out-of-distribution and the learned features do not activate.

The COCO model, by contrast, was trained on hundreds of thousands of images from highly varied sources (street scenes, traffic cameras, dashcam footage, parking lots from many continents). Its "car" concept is far more general and transfers cleanly to our footage.

**This is not a failure of training — the training succeeded by every metric we asked it to optimize. It is a failure of dataset selection. PKLot is the wrong proxy for arbitrary real-world parking footage.**

---

## Decision

**Ship the COCO baseline (`yolov8n.pt`) as the production model for the demo.**

`ml/yolo/config.py` already points at the COCO weights and was not changed during evaluation. The fine-tuned weights file (`ml/yolo/weights/yolov8m_pklot.pt`) is preserved in the repo as both an artifact of the experiment and as a starting point for future fine-tuning.

### Why keep the fine-tuned weights even though they failed
1. **Architectural validation:** the fact that we *can* swap models with one config-line change is a core claim of the architecture. The PKLot weights file proves the swap works in practice.
2. **Future progressive fine-tuning:** when KSU footage becomes available, we will fine-tune again starting from the PKLot weights, not from COCO. The PKLot fine-tune learned parking-domain features (lot geometry, car arrangements, painted lines) that are still useful as a starting point — they just need the final layer adapted to KSU's specific visual distribution. This is called *progressive transfer learning* and is the standard approach for adapting models across related domains.

---

## Forward plan

1. **Highest priority:** record real KSU parking footage. Even 10 minutes of phone video from a stable position would dramatically improve the demo's relevance and provide training data for a KSU-specific fine-tune.
2. **When KSU footage is available:** annotate ~50 frames using Roboflow's free annotation tool, fine-tune YOLOv8m starting from `yolov8m_pklot.pt` (not from scratch, not from COCO), evaluate against the same ground-truth methodology used here, and swap if it beats COCO. Estimated time: 2 hours of human work + 1 hour of Colab training.
3. **Stretch experiments (post-handoff):** ensemble the COCO and PKLot models to see if combining them recovers any of the PKLot model's parking-specific knowledge; try fine-tuning on CNRPark-EXT (which has angled surveillance footage) to test whether a different public dataset transfers better than PKLot did.

---

## What this finding is worth in the presentation

This evaluation is a strong technical talking point, not a weakness:

> "We trained a YOLOv8m model on PKLot and achieved 99.5% validation accuracy. Then we tested it honestly on our real demo footage — and it failed almost completely. The COCO baseline we built in week one outperformed it at 94.6%. This taught us that public parking datasets don't transfer to arbitrary camera angles, which is exactly why our architecture treats the model file as a swappable component. When we have KSU footage, we re-fine-tune from the PKLot weights as a starting point and the system updates with a one-line config change — no architectural rework, no retraining the entire pipeline."

This story demonstrates:
- Honest self-evaluation
- Understanding of in-distribution vs. out-of-distribution generalization
- Architectural foresight (designing for swappability before knowing we'd need it)
- A grounded, specific plan for what to do when real data arrives