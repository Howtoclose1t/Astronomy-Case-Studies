from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_yaml
from .paths import resolve_project_path
from .runtime import configure_project_cache


def evaluate(config_path: str, weights: str | Path, output_name: str = "evaluation_summary.json"):
    """Evaluate a trained YOLO detector and save compact metrics."""
    configure_project_cache()
    from ultralytics import YOLO

    cfg = load_yaml(config_path)
    model = YOLO(str(resolve_project_path(weights)))
    metrics = model.val(
        data=str(resolve_project_path(cfg["dataset_yaml"])),
        imgsz=int(cfg.get("imgsz", 640)),
        batch=int(cfg.get("batch", 8)),
        device=cfg.get("device", 0),
        max_det=int(cfg.get("max_det", 600)),
        plots=True,
    )
    box = metrics.box
    precision = float(box.mp)
    recall = float(box.mr)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }
    out_dir = resolve_project_path("outputs/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_name
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate a YOLO radio source detector.")
    parser.add_argument("--config", default="configs/train_small.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-name", default="evaluation_summary.json")
    args = parser.parse_args()
    evaluate(args.config, args.weights, args.output_name)


if __name__ == "__main__":
    main()
