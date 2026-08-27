from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime import configure_project_cache

configure_project_cache()

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_yaml
from .paths import resolve_project_path


def _iou_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return pairwise IoU values for two arrays of xyxy boxes."""
    if len(first) == 0 or len(second) == 0:
        return np.zeros((len(first), len(second)), dtype=float)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0.0, None)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    first_area = np.clip(first[:, 2] - first[:, 0], 0.0, None) * np.clip(first[:, 3] - first[:, 1], 0.0, None)
    second_area = np.clip(second[:, 2] - second[:, 0], 0.0, None) * np.clip(second[:, 3] - second[:, 1], 0.0, None)
    union = first_area[:, None] + second_area[None, :] - intersection_area
    return np.divide(intersection_area, union, out=np.zeros_like(intersection_area), where=union > 0)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Apply class-agnostic non-maximum suppression and return kept indices."""
    if len(boxes) == 0:
        return np.array([], dtype=int)
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        overlaps = _iou_matrix(boxes[current:current + 1], boxes[order[1:]])[0]
        order = order[1:][overlaps < iou_threshold]
    return np.asarray(keep, dtype=int)


def _match_boxes(gt_boxes: np.ndarray, pred_boxes: np.ndarray, scores: np.ndarray, iou_threshold: float):
    """Greedily match predictions to ground truth in descending confidence order."""
    gt_match = np.full(len(gt_boxes), -1, dtype=int)
    pred_match = np.full(len(pred_boxes), -1, dtype=int)
    matched_ious = []
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return gt_match, pred_match, matched_ious
    pairwise = _iou_matrix(pred_boxes, gt_boxes)
    for pred_index in np.argsort(scores)[::-1]:
        available = np.flatnonzero(gt_match < 0)
        if len(available) == 0:
            break
        best_local = int(np.argmax(pairwise[pred_index, available]))
        gt_index = int(available[best_local])
        best_iou = float(pairwise[pred_index, gt_index])
        if best_iou >= iou_threshold:
            gt_match[gt_index] = pred_index
            pred_match[pred_index] = gt_index
            matched_ious.append(best_iou)
    return gt_match, pred_match, matched_ious


def _metrics(gt_boxes: np.ndarray, pred_boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> dict:
    gt_match, pred_match, matched_ious = _match_boxes(gt_boxes, pred_boxes, scores, iou_threshold)
    true_positive = int(np.sum(pred_match >= 0))
    false_positive = int(len(pred_boxes) - true_positive)
    false_negative = int(len(gt_boxes) - true_positive)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "gt_match": gt_match,
        "pred_match": pred_match,
    }


def _ground_truth_boxes(objects: pd.DataFrame, tile_size: int) -> tuple[pd.DataFrame, np.ndarray]:
    frame = objects.copy()
    frame["width_px"] = frame["width"] * tile_size
    frame["height_px"] = frame["height"] * tile_size
    unique = frame.groupby("source_id", as_index=False).agg(
        x_abs=("x_abs", "first"),
        y_abs=("y_abs", "first"),
        width_px=("width_px", "max"),
        height_px=("height_px", "max"),
        flux_jy=("flux_jy", "first"),
        class_name=("class_name", "first"),
    )
    boxes = np.column_stack(
        [
            unique["x_abs"] - unique["width_px"] / 2,
            unique["y_abs"] - unique["height_px"] / 2,
            unique["x_abs"] + unique["width_px"] / 2,
            unique["y_abs"] + unique["height_px"] / 2,
        ]
    )
    return unique, boxes


def _threshold_table(gt_boxes: np.ndarray, pred_boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> pd.DataFrame:
    thresholds = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6])
    rows = []
    for threshold in thresholds:
        selected = scores >= threshold
        metric = _metrics(gt_boxes, pred_boxes[selected], scores[selected], iou_threshold)
        rows.append({"confidence": threshold, **{key: value for key, value in metric.items() if not key.endswith("match")}})
    return pd.DataFrame(rows)


def _flux_completeness(gt: pd.DataFrame, gt_match: np.ndarray) -> pd.DataFrame:
    edges = np.array([1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, np.inf])
    labels = ["10-20", "20-50", "50-100", "100-200", "200-500", "500-1000", ">=1000"]
    frame = gt.copy()
    frame["matched"] = gt_match >= 0
    frame["flux_bin_ujy"] = pd.cut(frame["flux_jy"] * 1e6, bins=edges * 1e6, labels=labels, right=False)
    rows = []
    for label, group in frame.groupby("flux_bin_ujy", observed=False):
        total = int(len(group))
        matched = int(group["matched"].sum())
        rows.append({"flux_bin_ujy": str(label), "sources": total, "matched": matched, "completeness": matched / total if total else np.nan})
    return pd.DataFrame(rows)


def _class_completeness(gt: pd.DataFrame, gt_match: np.ndarray) -> pd.DataFrame:
    frame = gt.copy()
    frame["matched"] = gt_match >= 0
    rows = []
    for class_name, group in frame.groupby("class_name"):
        total = int(len(group))
        matched = int(group["matched"].sum())
        rows.append({"class_name": class_name, "sources": total, "matched": matched, "completeness": matched / total if total else 0.0})
    return pd.DataFrame(rows)


def _draw_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int], width: int = 1):
    x1, y1, x2, y2 = np.rint(box).astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, width)


def _save_qualitative_examples(
    image_dir: Path,
    objects: pd.DataFrame,
    tile_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    tile_size: int,
    confidence: float,
    iou_threshold: float,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = []
    annotations = {}
    for image_name, group in objects.groupby("image"):
        gt_boxes = np.column_stack(
            [
                (group["xc"] - group["width"] / 2) * tile_size,
                (group["yc"] - group["height"] / 2) * tile_size,
                (group["xc"] + group["width"] / 2) * tile_size,
                (group["yc"] + group["height"] / 2) * tile_size,
            ]
        )
        pred_boxes, pred_scores = tile_predictions.get(image_name, (np.empty((0, 4)), np.empty(0)))
        selected = pred_scores >= confidence
        pred_boxes = pred_boxes[selected]
        pred_scores = pred_scores[selected]
        metric = _metrics(gt_boxes, pred_boxes, pred_scores, iou_threshold)
        if len(gt_boxes) >= 3:
            ranked.append((image_name, metric))
            annotations[image_name] = (gt_boxes, pred_boxes, metric)

    successes = sorted(ranked, key=lambda item: (item[1]["f1"], item[1]["true_positive"]), reverse=True)[:3]
    failures = sorted(ranked, key=lambda item: (item[1]["false_negative"] + item[1]["false_positive"], -item[1]["f1"]), reverse=True)[:3]
    for category, examples in (("success", successes), ("failure", failures)):
        for rank, (image_name, metric) in enumerate(examples, start=1):
            image = cv2.imread(str(image_dir / image_name))
            gt_boxes, pred_boxes, metric = annotations[image_name]
            for index, box in enumerate(gt_boxes):
                _draw_box(image, box, (40, 190, 40) if metric["gt_match"][index] >= 0 else (30, 30, 230), 1)
            for index, box in enumerate(pred_boxes):
                _draw_box(image, box, (230, 190, 30) if metric["pred_match"][index] >= 0 else (0, 170, 255), 1)
            label = f"TP {metric['true_positive']}  FP {metric['false_positive']}  FN {metric['false_negative']}"
            cv2.rectangle(image, (0, 0), (tile_size, 22), (0, 0, 0), -1)
            cv2.putText(image, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(output_dir / f"{category}_{rank}_{image_name}"), image)


def _plot_metrics(thresholds: pd.DataFrame, flux: pd.DataFrame, classes: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(flux["flux_bin_ujy"], flux["completeness"], marker="o", color="#2364aa")
    axes[0].set(title="Completeness by catalogue flux", xlabel="Flux bin (microJy)", ylabel="Completeness", ylim=(0, 1.05))
    axes[0].tick_params(axis="x", rotation=35)
    for metric, color in (("precision", "#2a9d8f"), ("recall", "#e76f51"), ("f1", "#264653")):
        axes[1].plot(thresholds["confidence"], thresholds[metric], marker="o", label=metric.title(), color=color)
    axes[1].set(title="Operating-point trade-off", xlabel="Confidence threshold", ylabel="Score", ylim=(0, 1.05))
    axes[1].legend(frameon=False)
    axes[2].bar(classes["class_name"], classes["completeness"], color=["#457b9d", "#e9c46a", "#e76f51"][: len(classes)])
    axes[2].set(title="Completeness by source population", xlabel="Catalogue class", ylabel="Completeness", ylim=(0, 1.05))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_scientifically(config_path: str, weights: str | Path) -> dict:
    """Evaluate source completeness and reliability on a spatial holdout region."""
    configure_project_cache()
    from ultralytics import YOLO

    cfg = load_yaml(config_path)
    dataset_cfg = load_yaml(cfg["dataset_yaml"])
    dataset_root = resolve_project_path(dataset_cfg["path"])
    image_dir = dataset_root / dataset_cfg["val"]
    objects = pd.read_csv(dataset_root / "objects.csv")
    objects = objects.loc[objects["split"] == "val"].copy()
    tiles = pd.read_csv(dataset_root / "tiles.csv").set_index("image")
    confidence_min = float(cfg.get("scientific_conf_min", 0.005))
    matching_iou = float(cfg.get("matching_iou", 0.3))
    global_nms_iou = float(cfg.get("global_nms_iou", 0.3))

    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No validation images found in {image_dir}")
    tile_size = int(cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE).shape[0])
    model = YOLO(str(resolve_project_path(weights)))
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=int(cfg.get("imgsz", 640)),
        conf=confidence_min,
        iou=0.5,
        max_det=int(cfg.get("max_det", 600)),
        device=cfg.get("device", 0),
        verbose=False,
    )

    global_boxes = []
    global_scores = []
    tile_predictions = {}
    for image_path, result in zip(image_paths, results):
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty(0)
        tile_predictions[image_path.name] = (boxes, scores)
        if len(boxes):
            x0 = float(tiles.loc[image_path.name, "x0"])
            y0 = float(tiles.loc[image_path.name, "y0"])
            absolute = boxes + np.array([x0, y0, x0, y0])
            global_boxes.append(absolute)
            global_scores.append(scores)

    pred_boxes = np.concatenate(global_boxes) if global_boxes else np.empty((0, 4))
    pred_scores = np.concatenate(global_scores) if global_scores else np.empty(0)
    keep = _nms(pred_boxes, pred_scores, global_nms_iou)
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    gt, gt_boxes = _ground_truth_boxes(objects, tile_size)

    thresholds = _threshold_table(gt_boxes, pred_boxes, pred_scores, matching_iou)
    best_row = thresholds.loc[thresholds["f1"].idxmax()]
    confidence = float(best_row["confidence"])
    selected = pred_scores >= confidence
    final_metric = _metrics(gt_boxes, pred_boxes[selected], pred_scores[selected], matching_iou)
    flux = _flux_completeness(gt, final_metric["gt_match"])
    classes = _class_completeness(gt, final_metric["gt_match"])

    metrics_dir = resolve_project_path("outputs/metrics/round2")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    thresholds.to_csv(metrics_dir / "reliability_by_confidence.csv", index=False)
    flux.to_csv(metrics_dir / "completeness_by_flux.csv", index=False)
    classes.to_csv(metrics_dir / "completeness_by_class.csv", index=False)
    summary = {
        "confidence_threshold": confidence,
        "matching_iou": matching_iou,
        "unique_catalogue_sources": int(len(gt_boxes)),
        "deduplicated_predictions": int(np.sum(selected)),
        **{key: value for key, value in final_metric.items() if not key.endswith("match")},
    }
    (metrics_dir / "scientific_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure_path = resolve_project_path("outputs/figures/round2_scientific_metrics.png")
    _plot_metrics(thresholds, flux, classes, figure_path)
    _save_qualitative_examples(
        image_dir,
        objects,
        tile_predictions,
        tile_size,
        confidence,
        matching_iou,
        resolve_project_path("outputs/predictions/round2_examples"),
    )
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate flux-dependent radio source detection performance.")
    parser.add_argument("--config", default="configs/train_round2.yaml")
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()
    evaluate_scientifically(args.config, args.weights)


if __name__ == "__main__":
    main()
