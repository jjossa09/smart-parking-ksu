"""
Fine-tune YOLOv8m on PKLot starting from COCO pretrained weights.

Designed to run identically on Colab and locally. On Colab, the dataset
lands at /content/PKLot-*/data.yaml after the Roboflow download cell.

Usage from a Colab cell or shell:
    python -m ml.yolo.training.train_finetune --data /content/PKLot-1/data.yaml

Or import and call train() from the notebook.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


# Hyperparameters — Ultralytics defaults are good. Tune on Day 5 if needed.
DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ  = 640
DEFAULT_BATCH  = 16
DEFAULT_MODEL  = "yolov8m.pt"   # COCO pretrained, auto-downloads on first use


def train(
    data_yaml: str,
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_BATCH,
    model_weights: str = DEFAULT_MODEL,
    project: str = "runs/train",
    name: str = "pklot_finetune",
) -> Path:
    """
    Run a fine-tuning job. Returns the path to the best.pt produced by Ultralytics.
    """
    print(f"[train] loading base model: {model_weights}")
    model = YOLO(model_weights)

    print(f"[train] starting fine-tune  data={data_yaml}  epochs={epochs}  imgsz={imgsz}  batch={batch}")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        exist_ok=True,
        patience=10,         # early-stop if no improvement for 10 epochs
        save=True,
        plots=True,
        verbose=True,
    )

    # Ultralytics writes to {project}/{name}/weights/best.pt
    best_pt = Path(project) / name / "weights" / "best.pt"
    print(f"[train] done. best weights: {best_pt}")
    return best_pt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset data.yaml")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_weights=args.model,
    )


if __name__ == "__main__":
    main()