from __future__ import annotations

import argparse

from .config import load_yaml
from .paths import resolve_project_path
from .runtime import configure_project_cache


def train(config_path: str):
    """Train an Ultralytics YOLO detector from a YAML configuration."""
    configure_project_cache()
    from ultralytics import YOLO

    cfg = load_yaml(config_path)
    model = YOLO(cfg["model"])
    optional_keys = [
        "lr0",
        "mosaic",
        "degrees",
        "translate",
        "scale",
        "fliplr",
        "flipud",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "mixup",
    ]
    augmentation_args = {key: cfg[key] for key in optional_keys if key in cfg}
    results = model.train(
        data=str(resolve_project_path(cfg["dataset_yaml"])),
        project=str(resolve_project_path(cfg.get("project", "outputs/runs"))),
        name=cfg.get("name", "yolo_sdc1_small"),
        imgsz=int(cfg.get("imgsz", 640)),
        epochs=int(cfg.get("epochs", 30)),
        batch=int(cfg.get("batch", 8)),
        device=cfg.get("device", 0),
        workers=int(cfg.get("workers", 2)),
        patience=int(cfg.get("patience", 10)),
        seed=int(cfg.get("seed", 42)),
        optimizer=cfg.get("optimizer", "auto"),
        cos_lr=bool(cfg.get("cos_lr", True)),
        close_mosaic=int(cfg.get("close_mosaic", 10)),
        cache=bool(cfg.get("cache", False)),
        plots=bool(cfg.get("plots", True)),
        exist_ok=bool(cfg.get("exist_ok", False)),
        **augmentation_args,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO on radio source tiles.")
    parser.add_argument("--config", default="configs/train_small.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
