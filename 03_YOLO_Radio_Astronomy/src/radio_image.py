from pathlib import Path

import numpy as np
from astropy.io import fits


def open_fits_image(path: str | Path):
    """Open a FITS image with memory mapping and return image data plus header."""
    hdul = fits.open(path, memmap=True)
    data = hdul[0].data
    while data.ndim > 2:
        data = data[0]
    return hdul, data, hdul[0].header


def pixel_scale_arcsec(header) -> float:
    """Estimate pixel scale in arcseconds from a FITS header."""
    for key in ("CDELT1", "CD1_1"):
        if key in header:
            return abs(float(header[key])) * 3600.0
    raise KeyError("Could not infer pixel scale from CDELT1 or CD1_1")


def robust_normalize(tile: np.ndarray, lower: float = 1.0, upper: float = 99.7, stretch: str = "asinh") -> np.ndarray:
    """Convert a radio image tile into an 8-bit display image."""
    finite = np.isfinite(tile)
    if not finite.any():
        return np.zeros(tile.shape, dtype=np.uint8)
    lo, hi = np.nanpercentile(tile[finite], [lower, upper])
    if hi <= lo:
        return np.zeros(tile.shape, dtype=np.uint8)
    scaled = np.clip((tile - lo) / (hi - lo), 0.0, 1.0)
    if stretch == "asinh":
        scaled = np.arcsinh(10.0 * scaled) / np.arcsinh(10.0)
    return (255.0 * scaled).astype(np.uint8)


def to_rgb(gray: np.ndarray) -> np.ndarray:
    """Convert a single-channel image to three-channel RGB."""
    return np.repeat(gray[:, :, None], 3, axis=2)
