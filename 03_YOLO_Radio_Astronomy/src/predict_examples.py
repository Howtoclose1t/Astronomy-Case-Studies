from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .config import load_yaml
from .paths import resolve_project_path
from .runtime import configure_project_cache


def save_prediction_examples(config_path: str, weights: str | Path, limit: int = 8) -> Path:
    """Save annotated validation predictions for qualitative inspection."""
    configure_project_cache()
    from ultralytics import YOLO

    cfg = load_yaml(config_path)
    dataset_yaml = load_yaml(cfg["dataset_yaml"])
    dataset_root = resolve_project_path(dataset_yaml["path"])
    val_dir = dataset_root / dataset_yaml["val"]
    images = sorted(val_dir.glob("*.png"))[:limit]
    output_dir = resolve_project_path("outputs/predictions/validation_examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(resolve_project_path(weights)))
    results = model.predict(source=[str(path) for path in images], imgsz=int(cfg.get("imgsz", 640)), conf=0.05, iou=0.5, verbose=False)
    for image_path, result in zip(images, results):
        plotted = result.plot()
        cv2.imwrite(str(output_dir / image_path.name), plotted)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Save validation prediction examples.")
    parser.add_argument("--config", default="configs/train_small.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    output_dir = save_prediction_examples(args.config, args.weights, args.limit)
    print(f"Prediction examples written to {output_dir}")


if __name__ == "__main__":
    main()
