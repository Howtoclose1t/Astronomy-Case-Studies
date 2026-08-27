from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def read_yolo_labels(path: str | Path) -> list[tuple[int, float, float, float, float]]:
    """Read one YOLO label file."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cls, xc, yc, w, h = line.split()
        labels.append((int(cls), float(xc), float(yc), float(w), float(h)))
    return labels


def draw_yolo_boxes(image_path: str | Path, label_path: str | Path, names: dict[int, str] | None = None) -> Image.Image:
    """Draw YOLO-format boxes on a PNG tile."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for cls, xc, yc, bw, bh in read_yolo_labels(label_path):
        x0 = (xc - bw / 2) * width
        x1 = (xc + bw / 2) * width
        y0 = (yc - bh / 2) * height
        y1 = (yc + bh / 2) * height
        draw.rectangle([x0, y0, x1, y1], outline=(255, 80, 40), width=2)
        if names:
            draw.text((x0 + 3, y0 + 3), names.get(cls, str(cls)), fill=(255, 220, 120))
    return image


def plot_tile_grid(items, columns: int = 3, figsize=(14, 7), title: str | None = None):
    """Plot image tiles in a horizontal grid for notebook display."""
    rows = int(np.ceil(len(items) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, item in zip(axes.ravel(), items):
        if isinstance(item, (str, Path)):
            image = Image.open(item).convert("RGB")
            subtitle = Path(item).name
        else:
            image, subtitle = item
        ax.imshow(image)
        ax.set_title(subtitle, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig
