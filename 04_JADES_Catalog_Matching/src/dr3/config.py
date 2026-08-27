"""Central configuration for the JADES catalog-matching case study."""

from pathlib import Path


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"


# Official JADES DR3 GOODS-N catalogs
PHOTOMETRY_PATH = (
    RAW_DATA_DIR
    / "hlsp_jades_jwst_nircam_goods-n_photometry_v1.0_catalog.fits"
)

SPECTROSCOPY_PATH = (
    RAW_DATA_DIR
    / "hlsp_jades_jwst_nirspec_goods-n_prism-line-fluxes_v1.1_catalog.fits"
)


# FITS extensions and photometric measurement
PHOTOMETRY_FLAG_EXTENSION = "FLAG"
PHOTOMETRY_FLUX_EXTENSION = "CIRC_CONV"
PHOTOMETRY_PHOTOZ_EXTENSION = "PHOTOZ"
SPECTROSCOPY_EXTENSION = "Joined"

PHOTOMETRY_APERTURE = "CIRC2"


# NIRCam filters used as redshift features
FILTERS = (
    "F090W",
    "F115W",
    "F150W",
    "F200W",
    "F277W",
    "F335M",
    "F356W",
    "F410M",
    "F444W",
)


# Catalog-matching configuration
MATCH_RADIUS_ARCSEC = 0.2

RADIUS_SCAN_ARCSEC = (
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
)


# Spectroscopic and photometric quality selection
SECURE_SPEC_FLAGS = (
    "A",
    "B",
    "C",
)

MIN_VALID_FILTERS = 5


# Generated catalog paths
MATCHED_CATALOG_PATH = (
    PROCESSED_DATA_DIR
    / "jades_dr3_goodsn_phot_spec_matched_all.csv"
)

ML_READY_CATALOG_PATH = (
    PROCESSED_DATA_DIR
    / "jades_dr3_goodsn_ml_ready.csv"
)


# Generated metrics paths
INPUT_SUMMARY_PATH = (
    METRICS_DIR
    / "input_summary.csv"
)

MATCH_SUMMARY_PATH = (
    METRICS_DIR
    / "match_summary.json"
)

RADIUS_SCAN_PATH = (
    METRICS_DIR
    / "matching_radius_scan.csv"
)

SPEC_QUALITY_SUMMARY_PATH = (
    METRICS_DIR
    / "spectroscopic_quality_summary.csv"
)


# Output directories created by the pipeline
OUTPUT_DIRECTORIES = (
    PROCESSED_DATA_DIR,
    FIGURE_DIR,
    METRICS_DIR,
)


def ensure_output_directories():
    """Create generated-output directories if they do not exist."""

    for directory in OUTPUT_DIRECTORIES:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )
