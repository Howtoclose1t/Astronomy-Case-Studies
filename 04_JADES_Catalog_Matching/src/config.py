"""Configuration for the current JADES DR4-DR5 catalog pipeline."""

from pathlib import Path


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"


# Catalog matching and sample-selection settings
MATCH_RADIUS_ARCSEC = 0.2

SECURE_SPEC_FLAGS = (
    "A",
    "B",
    "C",
)

MIN_VALID_FILTERS = 5


# Generated-output directories
OUTPUT_DIRECTORIES = (
    PROCESSED_DATA_DIR,
    FIGURE_DIR,
    METRICS_DIR,
)


def ensure_output_directories():
    """Create generated-output directories when they do not exist."""

    for directory in OUTPUT_DIRECTORIES:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )
