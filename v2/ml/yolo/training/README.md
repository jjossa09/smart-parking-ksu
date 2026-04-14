# Training - PKLot Fine-Tuning

Fine-tune YOLOv8m on PKLot from COCO pretrained weights, then drop the
result into `ml/yolo/weights/yolov8m_pklot.pt`. Production model swap is
a one-line change in `ml/yolo/config.py` - but only after Day 5 evaluation.

## What lives here

- `dataset.yaml` - reference format only. Real config ships inside the Roboflow download.
- `train_finetune.py` - training script. Runs the same on Colab and locally.
- `evaluate.py` - Day 5 evaluation against hand-labeled demo frames.

## How to run training (Colab, free T4)

1. Create a free Roboflow account at roboflow.com, copy your API key from
   app.roboflow.com/settings/api.
2. Open a new Colab notebook at colab.research.google.com.
3. Runtime → Change runtime type → **T4 GPU** → Save.
4. Run the cells from `colab_train.ipynb` (or paste them - see the build chat
   for the cell contents).
5. Training takes ~30–90 minutes. Keep the tab open. Don't let the laptop sleep.
6. When done, the notebook downloads `best.pt` to your machine.
7. Move that file to `ml/yolo/weights/yolov8m_pklot.pt` in this repo.

## Swapping the production model (Day 5+, only after evaluation)

In `ml/yolo/config.py`, change:

```python
MODEL_WEIGHTS = WEIGHTS_DIR / "yolov8n.pt"
```

to:

```python
MODEL_WEIGHTS = WEIGHTS_DIR / "yolov8m_pklot.pt"
```

Restart the backend. Done. The pipeline doesn't know or care which model is loaded.

## Why these choices

- **YOLOv8m, not nano.** Day 1–3 used nano for CPU speed. Fine-tuning gives us
  headroom to use the medium model- better accuracy ceiling, "we fine-tuned a
  real model" credibility.
- **PKLot.** Bigger and better-cited than CNRPark-EXT. Roboflow has clean
  YOLO-formatted versions ready to download.
- **Colab over local.** Even on a fast laptop GPU, Colab is "just works" -
  zero environment debugging, zero risk to the working Day 1–3 system.