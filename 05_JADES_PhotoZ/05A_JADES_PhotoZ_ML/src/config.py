"""Configuration for the expanded JADES DR4-DR5 Photo-z project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

UPSTREAM_PROJECT_ROOT = (
    PROJECT_ROOT.parents[1]
    / "04_JADES_Catalog_Matching"
)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "models"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

UPSTREAM_MATCHED_CATALOG_PATH = (
    UPSTREAM_PROJECT_ROOT
    / "data"
    / "processed"
    / "jades_dr4_dr5_matched_photometry.csv"
)

FIELD_COLUMN = "field"
PHOTOMETRY_ID_COLUMN = "phot_id"
SOURCE_KEY_COLUMN = "source_key"
TARGET_COLUMN = "z_spec"
QUALITY_COLUMN = "z_spec_quality"
MODEL_CANDIDATE_COLUMN = "is_model_candidate"
HYBRID_CANDIDATE_COLUMN = "is_hybrid_candidate"

REFERENCE_PREDICTION_COLUMN = "eazy_z_a"
EAZY_AUXILIARY_COLUMNS = (
    "eazy_z_ml",
    "eazy_chi_a",
    "eazy_l68",
    "eazy_u68",
    "eazy_nfilt",
    "eazy_z_peak",
    "eazy_z500",
)

CORE_FILTERS = (
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

OPTIONAL_FILTERS = (
    "F070W",
    "F162M",
    "F182M",
    "F210M",
    "F250M",
    "F300M",
    "F430M",
    "F460M",
    "F480M",
)

ALL_FILTERS = CORE_FILTERS + OPTIONAL_FILTERS

CORE_FLUX_COLUMNS = tuple(
    f"flux_{filter_name.lower()}_njy"
    for filter_name in CORE_FILTERS
)
CORE_ERROR_COLUMNS = tuple(
    f"fluxerr_{filter_name.lower()}_njy"
    for filter_name in CORE_FILTERS
)
CORE_VALIDITY_COLUMNS = tuple(
    f"valid_{filter_name.lower()}"
    for filter_name in CORE_FILTERS
)
OPTIONAL_FLUX_COLUMNS = tuple(
    f"flux_{filter_name.lower()}_njy"
    for filter_name in OPTIONAL_FILTERS
)
OPTIONAL_ERROR_COLUMNS = tuple(
    f"fluxerr_{filter_name.lower()}_njy"
    for filter_name in OPTIONAL_FILTERS
)
OPTIONAL_VALIDITY_COLUMNS = tuple(
    f"valid_{filter_name.lower()}"
    for filter_name in OPTIONAL_FILTERS
)
ALL_FLUX_COLUMNS = CORE_FLUX_COLUMNS + OPTIONAL_FLUX_COLUMNS
ALL_ERROR_COLUMNS = CORE_ERROR_COLUMNS + OPTIONAL_ERROR_COLUMNS
ALL_VALIDITY_COLUMNS = CORE_VALIDITY_COLUMNS + OPTIONAL_VALIDITY_COLUMNS
CORE_PHOTOMETRIC_COLUMNS = (
    CORE_FLUX_COLUMNS
    + CORE_ERROR_COLUMNS
    + CORE_VALIDITY_COLUMNS
)
ALL_PHOTOMETRIC_COLUMNS = (
    ALL_FLUX_COLUMNS
    + ALL_ERROR_COLUMNS
    + ALL_VALIDITY_COLUMNS
)

SECURE_REDSHIFT_QUALITIES = {"A", "B", "C"}
MINIMUM_VALID_FILTERS = 5

RANDOM_SEED = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
CROSS_VALIDATION_FOLDS = 5
REDSHIFT_STRATIFICATION_EDGES = (
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
    float("inf"),
)

MODELING_CATALOG_PATH = (
    PROCESSED_DATA_DIR
    / "jades_dr45_photoz_modeling_catalog.csv"
)
SPLIT_ASSIGNMENTS_PATH = (
    PROCESSED_DATA_DIR
    / "jades_dr45_split_assignments.csv"
)
TRAINING_CATALOG_PATH = (
    PROCESSED_DATA_DIR
    / "jades_dr45_training_catalog.csv"
)
VALIDATION_CATALOG_PATH = (
    PROCESSED_DATA_DIR
    / "jades_dr45_validation_catalog.csv"
)
TEST_CATALOG_PATH = (
    PROCESSED_DATA_DIR
    / "jades_dr45_test_catalog.csv"
)

EAZY_BENCHMARK_METRICS_PATH = (
    METRICS_DIR
    / "dr45_eazy_test_metrics.json"
)
EAZY_BENCHMARK_PREDICTIONS_PATH = (
    PREDICTION_DIR
    / "dr45_eazy_test_predictions.csv"
)

OUTPUT_DIRECTORIES = (
    PROCESSED_DATA_DIR,
    FIGURE_DIR,
    METRICS_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
)


def ensure_output_directories():
    """Create generated-output directories when they do not exist."""
    for directory in OUTPUT_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
