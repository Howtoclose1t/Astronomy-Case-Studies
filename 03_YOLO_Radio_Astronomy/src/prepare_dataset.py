from __future__ import annotations

import argparse
import math
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .catalog import read_sdc1_catalog
from .config import load_yaml
from .paths import project_root, resolve_project_path
from .radio_image import open_fits_image, pixel_scale_arcsec, robust_normalize, to_rgb


def _box_for_source(
    row,
    class_index: int,
    x0: int,
    y0: int,
    tile_size: int,
    pixscale: float,
    min_box: int,
    max_fraction: float,
):
    width = max(float(row.bmaj_arcsec) / pixscale, min_box)
    height = max(float(row.bmin_arcsec) / pixscale, min_box)
    max_box = tile_size * max_fraction
    width = min(width, max_box)
    height = min(height, max_box)
    cx = float(row.x) - x0
    cy = float(row.y) - y0
    if not (0 <= cx < tile_size and 0 <= cy < tile_size):
        return None
    x_min = max(0.0, cx - width / 2)
    x_max = min(float(tile_size), cx + width / 2)
    y_min = max(0.0, cy - height / 2)
    y_max = min(float(tile_size), cy + height / 2)
    if x_max <= x_min or y_max <= y_min:
        return None
    xc = ((x_min + x_max) / 2) / tile_size
    yc = ((y_min + y_max) / 2) / tile_size
    bw = (x_max - x_min) / tile_size
    bh = (y_max - y_min) / tile_size
    return class_index, xc, yc, bw, bh


def _tile_origins(width: int, height: int, tile_size: int, stride: int):
    for y0 in range(0, max(1, height - tile_size + 1), stride):
        for x0 in range(0, max(1, width - tile_size + 1), stride):
            yield x0, y0


def _origins_for_coordinate(coordinate: float, limit: int, tile_size: int, stride: int) -> range:
    max_origin = limit - tile_size
    first = max(0, math.ceil((coordinate - tile_size + 1e-9) / stride) * stride)
    last = min(max_origin, math.floor(coordinate / stride) * stride)
    if first > last:
        return range(0)
    return range(first, last + 1, stride)


def _positive_origins(table: pd.DataFrame, width: int, height: int, tile_size: int, stride: int) -> set[tuple[int, int]]:
    origins: set[tuple[int, int]] = set()
    for row in table.itertuples(index=False):
        x_origins = _origins_for_coordinate(float(row.x), width, tile_size, stride)
        y_origins = _origins_for_coordinate(float(row.y), height, tile_size, stride)
        origins.update((x0, y0) for y0 in y_origins for x0 in x_origins)
    return origins


def _sources_in_tile(table: pd.DataFrame, x0: int, y0: int, tile_size: int) -> pd.DataFrame:
    return table[
        (table["x"] >= x0)
        & (table["x"] < x0 + tile_size)
        & (table["y"] >= y0)
        & (table["y"] < y0 + tile_size)
    ]


def _spatial_split(x0: int, tile_size: int, boundary: float) -> str | None:
    if x0 + tile_size <= boundary:
        return "train"
    if x0 >= boundary:
        return "val"
    return None


def _select_tiles(
    table: pd.DataFrame,
    width: int,
    height: int,
    tile_size: int,
    stride: int,
    max_positive_tiles: int,
    max_sources_per_tile: int,
    negative_fraction: float,
    validation_fraction: float,
    split_strategy: str,
    rng: random.Random,
):
    positive_origin_set = _positive_origins(table, width, height, tile_size, stride)
    positive_origins = sorted(positive_origin_set)
    rng.shuffle(positive_origins)
    if max_positive_tiles > 0:
        positive_origins = positive_origins[:max_positive_tiles]

    boundary = float(table["x"].quantile(1.0 - validation_fraction))
    positives = []
    for x0, y0 in positive_origins:
        in_tile = _sources_in_tile(table, x0, y0, tile_size)
        if max_sources_per_tile > 0 and len(in_tile) > max_sources_per_tile:
            in_tile = in_tile.sort_values("flux_jy", ascending=False).head(max_sources_per_tile)
        if split_strategy == "spatial_x_holdout":
            split = _spatial_split(x0, tile_size, boundary)
        else:
            split = "val" if rng.random() < validation_fraction else "train"
        if split is not None:
            positives.append((x0, y0, in_tile, split))

    all_origins = list(_tile_origins(width, height, tile_size, stride))
    negative_origins = [origin for origin in all_origins if origin not in positive_origin_set]
    rng.shuffle(negative_origins)
    negative_candidates = {"train": [], "val": []}
    for x0, y0 in negative_origins:
        if split_strategy == "spatial_x_holdout":
            split = _spatial_split(x0, tile_size, boundary)
        else:
            split = "val" if rng.random() < validation_fraction else "train"
        if split is not None:
            negative_candidates[split].append((x0, y0, table.iloc[0:0], split))

    selected = list(positives)
    for split in ("train", "val"):
        positive_count = sum(item[3] == split for item in positives)
        negative_count = min(len(negative_candidates[split]), round(positive_count * negative_fraction))
        selected.extend(negative_candidates[split][:negative_count])
    rng.shuffle(selected)
    return selected, boundary


def prepare_dataset(config_path: str | Path) -> Path:
    """Build a YOLO-format tile dataset from an SDC1 FITS image and catalogue."""
    cfg = load_yaml(config_path)
    rng = random.Random(int(cfg.get("random_seed", 42)))
    fits_path = resolve_project_path(cfg["raw_fits"])
    catalog_path = resolve_project_path(cfg["catalog"])
    output_dir = resolve_project_path(cfg["output_dir"])
    tile_size = int(cfg.get("tile_size", 640))
    stride = int(cfg.get("tile_stride", tile_size))
    max_positive_tiles = int(cfg.get("max_positive_tiles", cfg.get("max_tiles", 160)))
    negative_fraction = float(cfg.get("negative_fraction", 0.25))
    validation_fraction = float(cfg.get("validation_fraction", 0.2))
    split_strategy = str(cfg.get("split_strategy", "random"))
    min_flux = float(cfg.get("min_flux_jy", 0.0))
    max_sources_per_tile = int(cfg.get("max_sources_per_tile", 0))
    min_box = int(cfg.get("min_box_pixels", 6))
    max_fraction = float(cfg.get("max_box_fraction", 0.6))
    class_mode = str(cfg.get("class_mode", "astrophysical"))
    norm_cfg = cfg.get("normalization", {})

    table = read_sdc1_catalog(catalog_path)
    if min_flux > 0:
        table = table.loc[table["flux_jy"] >= min_flux].copy()
    hdul, image, header = open_fits_image(fits_path)
    try:
        height, width = image.shape
        pixscale = pixel_scale_arcsec(header)
        selected, split_boundary = _select_tiles(
            table,
            width,
            height,
            tile_size,
            stride,
            max_positive_tiles,
            max_sources_per_tile,
            negative_fraction,
            validation_fraction,
            split_strategy,
            rng,
        )

        for split in ["train", "val"]:
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        tile_rows = []
        object_rows = []
        for x0, y0, in_tile, split in tqdm(selected, desc="Writing YOLO tiles"):
            stem = f"{cfg.get('band', 'B1').lower()}_x{x0:05d}_y{y0:05d}"
            tile = np.asarray(image[y0:y0 + tile_size, x0:x0 + tile_size], dtype=np.float32)
            if tile.shape != (tile_size, tile_size):
                continue
            gray = robust_normalize(
                tile,
                lower=float(norm_cfg.get("lower_percentile", 1.0)),
                upper=float(norm_cfg.get("upper_percentile", 99.7)),
                stretch=str(norm_cfg.get("stretch", "asinh")),
            )
            image_path = output_dir / "images" / split / f"{stem}.png"
            label_path = output_dir / "labels" / split / f"{stem}.txt"
            cv2.imwrite(str(image_path), cv2.cvtColor(to_rgb(gray), cv2.COLOR_RGB2BGR))

            labels = []
            class_counts = {name: 0 for name in ("SS-AGN", "FS-AGN", "SFG")}
            for row in in_tile.itertuples(index=False):
                yolo_class = 0 if class_mode == "detection" else int(row.class_index)
                box = _box_for_source(row, yolo_class, x0, y0, tile_size, pixscale, min_box, max_fraction)
                if box is None:
                    continue
                labels.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(*box))
                class_counts[str(row.class_name)] += 1
                object_rows.append(
                    {
                        "split": split,
                        "image": image_path.name,
                        "source_id": int(row.id),
                        "yolo_class": yolo_class,
                        "class_id": int(row.class_id),
                        "class_name": str(row.class_name),
                        "flux_jy": float(row.flux_jy),
                        "x_abs": float(row.x),
                        "y_abs": float(row.y),
                        "xc": box[1],
                        "yc": box[2],
                        "width": box[3],
                        "height": box[4],
                    }
                )
            label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            tile_rows.append(
                {
                    "split": split,
                    "image": image_path.name,
                    "sources": len(labels),
                    "ss_agn": class_counts["SS-AGN"],
                    "fs_agn": class_counts["FS-AGN"],
                    "sfg": class_counts["SFG"],
                    "x0": x0,
                    "y0": y0,
                    "min_flux_jy": min_flux,
                }
            )

        dataset_yaml = output_dir / "dataset.yaml"
        dataset_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(output_dir.relative_to(project_root())).replace("\\", "/"),
                    "train": "images/train",
                    "val": "images/val",
                    "names": cfg["classes"],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        tile_frame = pd.DataFrame(tile_rows)
        object_frame = pd.DataFrame(object_rows)
        tile_frame.to_csv(output_dir / "tiles.csv", index=False)
        object_frame.to_csv(output_dir / "objects.csv", index=False)
        metadata = {
            "class_mode": class_mode,
            "split_strategy": split_strategy,
            "split_boundary_x": split_boundary,
            "pixel_scale_arcsec": pixscale,
            "tile_size": tile_size,
            "tile_stride": stride,
            "min_flux_jy": min_flux,
            "tiles": tile_frame["split"].value_counts().to_dict(),
            "objects": object_frame["split"].value_counts().to_dict(),
        }
        (output_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
        return output_dir
    finally:
        hdul.close()


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO tiles from SKA SDC1 data.")
    parser.add_argument("--config", default="configs/sdc1_b1_small.yaml")
    args = parser.parse_args()
    output_dir = prepare_dataset(args.config)
    print(f"Dataset written to {output_dir.relative_to(project_root())}")


if __name__ == "__main__":
    main()
