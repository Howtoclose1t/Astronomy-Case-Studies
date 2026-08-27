"""Central configuration for the JADES TabPFN foundation-model case study."""

from pathlib import Path


# ---------------------------------------------------------------------
# Project directories
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AST_ROOT = PROJECT_ROOT.parents[1]

TASK5_ROOT = PROJECT_ROOT.parent / "05A_JADES_PhotoZ_ML"
TASK5_DATA_DIR = TASK5_ROOT / "data" / "processed"
TASK5_OUTPUT_DIR = TASK5_ROOT / "outputs"

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "models"
PREDICTION_DIR = OUTPUT_DIR / "predictions"


# ---------------------------------------------------------------------
# Frozen inputs inherited from Task 5
# ---------------------------------------------------------------------

TRAIN_FEATURES_PATH = (
    TASK5_DATA_DIR
    / "jades_dr45_weighted_training_features.csv"
)

AUGMENTED_TRAIN_FEATURES_PATH = (
    TASK5_DATA_DIR
    / "jades_dr45_augmented_training_features.csv"
)

VALIDATION_FEATURES_PATH = (
    TASK5_DATA_DIR
    / "jades_dr45_validation_features.csv"
)

TEST_FEATURES_PATH = (
    TASK5_DATA_DIR
    / "jades_dr45_test_features.csv"
)

FEATURE_STATE_PATH = (
    TASK5_OUTPUT_DIR
    / "models"
    / "jades_dr45_feature_state.json"
)

TASK5_TEST_PREDICTIONS_PATH = (
    TASK5_OUTPUT_DIR
    / "predictions"
    / "dr45_final_test_predictions.csv"
)


# ---------------------------------------------------------------------
# Catalog schema
# ---------------------------------------------------------------------

SOURCE_KEY_COLUMN = "source_key"
TARGET_COLUMN = "z_spec"
SPLIT_COLUMN = "split"
SAMPLE_WEIGHT_COLUMN = "sample_weight"

EAZY_REDSHIFT_COLUMN = "eazy_z_a"
EAZY_MODEL_FEATURE = "x_eazy_z_a"

EAZY_AUXILIARY_COLUMNS = (
    "eazy_z_ml",
    "eazy_chi_a",
    "eazy_l68",
    "eazy_u68",
    "eazy_nfilt",
    "eazy_z_peak",
    "eazy_z500",
)

EAZY_DIAGNOSTIC_FEATURES = (
    "x_eazy_log1p_za",
    "x_eazy_log_chi",
    "x_eazy_nfilt",
    "x_eazy_interval_width",
    "x_eazy_normalized_interval_width",
    "x_eazy_delta_za_zml",
    "x_eazy_delta_za_zpeak",
    "x_eazy_delta_za_z500",
)


# ---------------------------------------------------------------------
# Expected dataset structure
# ---------------------------------------------------------------------

EXPECTED_SPLIT_COUNTS = {
    "train": 2219,
    "validation": 476,
    "test": 476,
}

EXPECTED_TOTAL_SOURCES = 3171
EXPECTED_FULL_FEATURE_COUNT = 57


# ---------------------------------------------------------------------
# TabPFN residual-learning experiment
# ---------------------------------------------------------------------

RANDOM_SEED = 42

TABPFN_MODEL_VERSION = "v3"
TABPFN_DEVICE = "cuda"
TABPFN_FIT_MODE = "low_memory"
TABPFN_N_ESTIMATORS = 8

RESIDUAL_TARGET_COLUMN = "eazy_log_residual"
TABPFN_REDSHIFT_COLUMN = "z_tabpfn_residual"

CATASTROPHIC_OUTLIER_THRESHOLD = 0.15
MINIMUM_PREDICTED_REDSHIFT = 0.0
MAXIMUM_PREDICTED_REDSHIFT = 20.0


# ---------------------------------------------------------------------
# Task 5B output files
# ---------------------------------------------------------------------

VALIDATION_PREDICTIONS_PATH = (
    PREDICTION_DIR
    / "tabpfn_validation_predictions.csv"
)

VALIDATION_METRICS_PATH = (
    METRICS_DIR
    / "tabpfn_validation_metrics.csv"
)

TEST_PREDICTIONS_PATH = (
    PREDICTION_DIR
    / "tabpfn_locked_test_predictions.csv"
)

TEST_METRICS_PATH = (
    METRICS_DIR
    / "tabpfn_locked_test_metrics.csv"
)

EXPERIMENT_MANIFEST_PATH = (
    MODEL_DIR
    / "tabpfn_experiment_manifest.json"
)


def ensure_output_directories():
    """Create Task 5B output directories when an experiment is run."""

    for directory in (
        PROCESSED_DATA_DIR,
        FIGURE_DIR,
        METRICS_DIR,
        MODEL_DIR,
        PREDICTION_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
