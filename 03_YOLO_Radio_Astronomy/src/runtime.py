from __future__ import annotations

import os
from pathlib import Path

from .paths import project_root


def configure_project_cache() -> Path:
    """Route training caches and temporary files to the project drive."""
    cache_root = project_root() / ".cache"
    locations = {
        "TORCH_HOME": cache_root / "torch",
        "YOLO_CONFIG_DIR": cache_root / "ultralytics",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "HF_HOME": cache_root / "huggingface",
        "XDG_CACHE_HOME": cache_root,
        "CUDA_CACHE_PATH": cache_root / "cuda",
        "NUMBA_CACHE_DIR": cache_root / "numba",
        "PIP_CACHE_DIR": cache_root / "pip",
        "TEMP": cache_root / "tmp",
        "TMP": cache_root / "tmp",
    }
    for variable, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    return cache_root
