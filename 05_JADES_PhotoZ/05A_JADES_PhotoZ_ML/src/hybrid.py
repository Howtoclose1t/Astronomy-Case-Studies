"""Build an out-of-fold gated residual EAZY-XGBoost hybrid."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

from .config import (
    METRICS_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
    RANDOM_SEED,
    SOURCE_KEY_COLUMN,
    TARGET_COLUMN,
    ensure_output_directories,
)
from .features import (
    EAZY_HYBRID_FEATURE,
    FULL_FEATURE_COLUMNS,
    VALIDATION_FEATURES_PATH,
    WEIGHTED_TRAINING_FEATURES_PATH,
)
from .resampling import (
    SAMPLE_ID_COLUMN,
    SAMPLE_WEIGHT_COLUMN,
)
from .xgboost import (
    CATASTROPHIC_COLUMN,
    VALIDATION_PREDICTIONS_PATH,
    build_prediction_table,
    compute_photoz_metrics,
)


REDSHIFT_STRATUM_COLUMN = "redshift_stratum"
SPLIT_COLUMN = "split"

EAZY_AUXILIARY_INPUT_COLUMNS = (
    "eazy_z_ml",
    "eazy_chi_a",
    "eazy_l68",
    "eazy_u68",
    "eazy_nfilt",
    "eazy_z_peak",
    "eazy_z500",
)

EAZY_DIAGNOSTIC_FEATURE_COLUMNS = (
    "x_eazy_log1p_za",
    "x_eazy_log_chi",
    "x_eazy_nfilt",
    "x_eazy_interval_width",
    "x_eazy_normalized_interval_width",
    "x_eazy_delta_za_zml",
    "x_eazy_delta_za_zpeak",
    "x_eazy_delta_za_z500",
)

PREDICTED_RESIDUAL_FEATURE = (
    "x_predicted_eazy_log_residual"
)

ABSOLUTE_PREDICTED_RESIDUAL_FEATURE = (
    "x_absolute_predicted_eazy_log_residual"
)

RESIDUAL_FEATURE_COLUMNS = (
    FULL_FEATURE_COLUMNS
    + (EAZY_HYBRID_FEATURE,)
    + EAZY_DIAGNOSTIC_FEATURE_COLUMNS
)

GATE_FEATURE_COLUMNS = (
    RESIDUAL_FEATURE_COLUMNS
    + (
        PREDICTED_RESIDUAL_FEATURE,
        ABSOLUTE_PREDICTED_RESIDUAL_FEATURE,
    )
)

OOF_FOLDS = 5
MINIMUM_CORRECTION_IMPROVEMENT = 0.01
MAXIMUM_PREDICTED_REDSHIFT = 20.0

GATE_THRESHOLD_GRID = np.concatenate(
    [
        np.array([0.0]),
        np.linspace(0.05, 0.95, 19),
        np.array([1.01]),
    ]
)

RESIDUAL_MODEL_PATH = (
    MODEL_DIR
    / "xgb_eazy_residual.json"
)

GATE_MODEL_PATH = (
    MODEL_DIR
    / "xgb_eazy_residual_gate.json"
)

HYBRID_VALIDATION_METRICS_PATH = (
    METRICS_DIR
    / "dr45_hybrid_validation_metrics.csv"
)

GATE_THRESHOLD_METRICS_PATH = (
    METRICS_DIR
    / "dr45_hybrid_gate_thresholds.csv"
)

HYBRID_VALIDATION_PREDICTIONS_PATH = (
    PREDICTION_DIR
    / "dr45_hybrid_validation_predictions.csv"
)

HYBRID_OOF_PREDICTIONS_PATH = (
    PREDICTION_DIR
    / "dr45_hybrid_oof_training_predictions.csv"
)

HYBRID_MANIFEST_PATH = (
    METRICS_DIR
    / "dr45_hybrid_manifest.json"
)


OOF_RESIDUAL_PARAMETERS = {
    "objective": "reg:pseudohubererror",
    "n_estimators": 1200,
    "learning_rate": 0.02,
    "max_depth": 3,
    "min_child_weight": 8.0,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.10,
    "reg_lambda": 3.0,
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": 0,
}

FINAL_RESIDUAL_PARAMETERS = {
    **OOF_RESIDUAL_PARAMETERS,
    "n_estimators": 3000,
    "eval_metric": "mae",
    "early_stopping_rounds": 150,
}

GATE_PARAMETERS = {
    "objective": "binary:logistic",
    "n_estimators": 600,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 10.0,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.20,
    "reg_lambda": 3.0,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": 0,
}


def _numeric_series(series: pd.Series) -> pd.Series:
    """Convert a catalog column to finite numbers or missing values."""

    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )


def load_hybrid_catalog(
    path,
    expected_split: str,
    catalog_name: str,
) -> pd.DataFrame:
    """Load one catalog required by the hybrid model."""

    if not path.exists():
        raise FileNotFoundError(
            f"{catalog_name} was not found at {path}"
        )

    catalog = pd.read_csv(
        path,
        low_memory=False,
    )

    required_columns = {
        SOURCE_KEY_COLUMN,
        SAMPLE_ID_COLUMN,
        TARGET_COLUMN,
        REDSHIFT_STRATUM_COLUMN,
        SPLIT_COLUMN,
        SAMPLE_WEIGHT_COLUMN,
        EAZY_HYBRID_FEATURE,
        *FULL_FEATURE_COLUMNS,
        *EAZY_AUXILIARY_INPUT_COLUMNS,
    }

    missing_columns = sorted(
        required_columns - set(catalog.columns)
    )

    if missing_columns:
        raise KeyError(
            f"{catalog_name} is missing required columns: "
            f"{missing_columns}"
        )

    if not catalog[SPLIT_COLUMN].eq(
        expected_split
    ).all():
        raise ValueError(
            f"{catalog_name} does not contain only "
            f"the {expected_split} split."
        )

    if not catalog[SAMPLE_ID_COLUMN].is_unique:
        raise ValueError(
            f"{catalog_name} contains duplicated sample IDs."
        )

    catalog[TARGET_COLUMN] = _numeric_series(
        catalog[TARGET_COLUMN]
    )

    catalog[EAZY_HYBRID_FEATURE] = (
        _numeric_series(
            catalog[EAZY_HYBRID_FEATURE]
        )
    )

    finite_targets = np.isfinite(
        catalog[TARGET_COLUMN]
    )

    finite_eazy = np.isfinite(
        catalog[EAZY_HYBRID_FEATURE]
    )

    if not finite_targets.all():
        raise ValueError(
            f"{catalog_name} contains non-finite z_spec."
        )

    if not finite_eazy.all():
        raise ValueError(
            f"{catalog_name} contains non-finite EAZY z_a."
        )

    if (
        catalog[EAZY_HYBRID_FEATURE] < 0
    ).any():
        raise ValueError(
            f"{catalog_name} contains negative EAZY z_a."
        )

    return catalog


def load_hybrid_catalogs(
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and validation without opening the test catalog."""

    training_catalog = load_hybrid_catalog(
        WEIGHTED_TRAINING_FEATURES_PATH,
        expected_split="train",
        catalog_name="weighted training features",
    )

    validation_catalog = load_hybrid_catalog(
        VALIDATION_FEATURES_PATH,
        expected_split="validation",
        catalog_name="validation features",
    )

    overlap = (
        set(training_catalog[SOURCE_KEY_COLUMN])
        & set(validation_catalog[SOURCE_KEY_COLUMN])
    )

    if overlap:
        raise ValueError(
            "Training-validation source leakage was detected."
        )

    return training_catalog, validation_catalog


def add_eazy_diagnostic_features(
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Create EAZY confidence and estimator-disagreement features."""

    result = catalog.copy()

    z_a = _numeric_series(
        result[EAZY_HYBRID_FEATURE]
    )

    z_ml = _numeric_series(
        result["eazy_z_ml"]
    )

    chi_a = _numeric_series(
        result["eazy_chi_a"]
    )

    lower_68 = _numeric_series(
        result["eazy_l68"]
    )

    upper_68 = _numeric_series(
        result["eazy_u68"]
    )

    number_of_filters = _numeric_series(
        result["eazy_nfilt"]
    )

    z_peak = _numeric_series(
        result["eazy_z_peak"]
    )

    z_500 = _numeric_series(
        result["eazy_z500"]
    )

    interval_width = (
        upper_68 - lower_68
    ).clip(lower=0.0)

    result["x_eazy_log1p_za"] = np.log1p(
        z_a.clip(lower=0.0)
    )

    result["x_eazy_log_chi"] = np.log1p(
        chi_a.clip(lower=0.0)
    )

    result["x_eazy_nfilt"] = number_of_filters

    result["x_eazy_interval_width"] = (
        interval_width
    )

    result[
        "x_eazy_normalized_interval_width"
    ] = interval_width / (1.0 + z_a)

    result["x_eazy_delta_za_zml"] = (
        z_a - z_ml
    ) / (1.0 + z_a)

    result["x_eazy_delta_za_zpeak"] = (
        z_a - z_peak
    ) / (1.0 + z_a)

    result["x_eazy_delta_za_z500"] = (
        z_a - z_500
    ) / (1.0 + z_a)

    return result


def prepare_matrix(
    catalog: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Construct a numeric matrix while retaining genuine missing values."""

    missing_columns = sorted(
        set(feature_columns) - set(catalog.columns)
    )

    if missing_columns:
        raise KeyError(
            "Hybrid features are missing: "
            f"{missing_columns}"
        )

    feature_matrix = (
        catalog.loc[
            :,
            list(feature_columns),
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    completely_missing = (
        feature_matrix.columns[
            feature_matrix.isna().all()
        ].tolist()
    )

    if completely_missing:
        raise ValueError(
            "Hybrid features are completely missing: "
            f"{completely_missing}"
        )

    return feature_matrix


def calculate_log_residual_target(
    catalog: pd.DataFrame,
) -> np.ndarray:
    """Calculate log(1 + z_spec) minus log(1 + z_EAZY)."""

    z_spec = catalog[TARGET_COLUMN].to_numpy(
        dtype=float
    )

    z_eazy = catalog[
        EAZY_HYBRID_FEATURE
    ].to_numpy(dtype=float)

    return (
        np.log1p(z_spec)
        - np.log1p(z_eazy)
    )


def apply_log_residual_correction(
    eazy_redshift,
    predicted_log_residual,
) -> np.ndarray:
    """Apply a bounded residual correction to EAZY redshift."""

    eazy_values = np.asarray(
        eazy_redshift,
        dtype=float,
    )

    residual_values = np.asarray(
        predicted_log_residual,
        dtype=float,
    )

    corrected_log_redshift = (
        np.log1p(eazy_values)
        + residual_values
    )

    corrected_log_redshift = np.clip(
        corrected_log_redshift,
        a_min=0.0,
        a_max=np.log1p(
            MAXIMUM_PREDICTED_REDSHIFT
        ),
    )

    return np.expm1(
        corrected_log_redshift
    )


def build_residual_model(
    final_model: bool,
) -> XGBRegressor:
    """Create either an OOF or final residual regressor."""

    if final_model:
        return XGBRegressor(
            **FINAL_RESIDUAL_PARAMETERS
        )

    return XGBRegressor(
        **OOF_RESIDUAL_PARAMETERS
    )


def build_gate_model() -> XGBClassifier:
    """Create the classifier that decides whether to apply correction."""

    return XGBClassifier(
        **GATE_PARAMETERS
    )


def generate_residual_oof_predictions(
    training_catalog: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate residual predictions from models that never saw each source."""

    feature_matrix = prepare_matrix(
        training_catalog,
        RESIDUAL_FEATURE_COLUMNS,
    )

    residual_target = (
        calculate_log_residual_target(
            training_catalog
        )
    )

    redshift_strata = training_catalog[
        REDSHIFT_STRATUM_COLUMN
    ].to_numpy()

    source_weights = _numeric_series(
        training_catalog[
            SAMPLE_WEIGHT_COLUMN
        ]
    ).to_numpy(dtype=float)

    cross_validator = StratifiedKFold(
        n_splits=OOF_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    oof_prediction = np.full(
        len(training_catalog),
        np.nan,
        dtype=float,
    )

    oof_fold = np.full(
        len(training_catalog),
        -1,
        dtype=int,
    )

    for fold_number, (
        fit_indices,
        holdout_indices,
    ) in enumerate(
        cross_validator.split(
            feature_matrix,
            redshift_strata,
        ),
        start=1,
    ):
        model = build_residual_model(
            final_model=False
        )

        model.fit(
            feature_matrix.iloc[
                fit_indices
            ],
            residual_target[
                fit_indices
            ],
            sample_weight=source_weights[
                fit_indices
            ],
            verbose=False,
        )

        oof_prediction[
            holdout_indices
        ] = model.predict(
            feature_matrix.iloc[
                holdout_indices
            ]
        )

        oof_fold[
            holdout_indices
        ] = fold_number

    if not np.isfinite(oof_prediction).all():
        raise RuntimeError(
            "Residual OOF predictions are incomplete."
        )

    return oof_prediction, oof_fold


def build_gate_training_data(
    training_catalog: pd.DataFrame,
    residual_oof_prediction: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Create leakage-safe gate features and correction-benefit labels."""

    result = training_catalog.copy()

    result[PREDICTED_RESIDUAL_FEATURE] = (
        residual_oof_prediction
    )

    result[
        ABSOLUTE_PREDICTED_RESIDUAL_FEATURE
    ] = np.abs(
        residual_oof_prediction
    )

    z_spec = result[TARGET_COLUMN].to_numpy(
        dtype=float
    )

    z_eazy = result[
        EAZY_HYBRID_FEATURE
    ].to_numpy(dtype=float)

    candidate_prediction = (
        apply_log_residual_correction(
            z_eazy,
            residual_oof_prediction,
        )
    )

    eazy_error = np.abs(
        (z_eazy - z_spec)
        / (1.0 + z_spec)
    )

    candidate_error = np.abs(
        (candidate_prediction - z_spec)
        / (1.0 + z_spec)
    )

    improvement = (
        eazy_error - candidate_error
    )

    correction_is_beneficial = (
        improvement
        >= MINIMUM_CORRECTION_IMPROVEMENT
    ).astype("int8")

    gate_matrix = prepare_matrix(
        result,
        GATE_FEATURE_COLUMNS,
    )

    diagnostic_table = result.loc[
        :,
        [
            SOURCE_KEY_COLUMN,
            TARGET_COLUMN,
            REDSHIFT_STRATUM_COLUMN,
            EAZY_HYBRID_FEATURE,
        ],
    ].copy()

    diagnostic_table[
        "residual_oof_prediction"
    ] = residual_oof_prediction

    diagnostic_table[
        "candidate_residual_redshift"
    ] = candidate_prediction

    diagnostic_table[
        "eazy_absolute_normalized_error"
    ] = eazy_error

    diagnostic_table[
        "candidate_absolute_normalized_error"
    ] = candidate_error

    diagnostic_table[
        "candidate_error_improvement"
    ] = improvement

    diagnostic_table[
        "correction_is_beneficial"
    ] = correction_is_beneficial.astype(
        bool
    )

    return (
        gate_matrix,
        correction_is_beneficial,
        diagnostic_table,
    )


def build_gate_sample_weights(
    labels: np.ndarray,
    source_weights: np.ndarray,
) -> np.ndarray:
    """Combine redshift weights with binary-class balancing."""

    labels = np.asarray(
        labels,
        dtype=int,
    )

    source_weights = np.asarray(
        source_weights,
        dtype=float,
    )

    positive_count = int(
        labels.sum()
    )

    negative_count = int(
        len(labels) - positive_count
    )

    if positive_count == 0:
        raise ValueError(
            "No beneficial corrections exist for gate training."
        )

    if negative_count == 0:
        raise ValueError(
            "Every correction is beneficial; a gate is unnecessary."
        )

    positive_scale = (
        negative_count / positive_count
    )

    gate_weights = source_weights.copy()

    gate_weights[
        labels == 1
    ] *= positive_scale

    return gate_weights


def generate_gate_oof_probabilities(
    gate_matrix: pd.DataFrame,
    gate_labels: np.ndarray,
    gate_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate leakage-safe OOF probabilities for threshold selection."""

    cross_validator = StratifiedKFold(
        n_splits=OOF_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED + 1,
    )

    oof_probability = np.full(
        len(gate_matrix),
        np.nan,
        dtype=float,
    )

    oof_fold = np.full(
        len(gate_matrix),
        -1,
        dtype=int,
    )

    for fold_number, (
        fit_indices,
        holdout_indices,
    ) in enumerate(
        cross_validator.split(
            gate_matrix,
            gate_labels,
        ),
        start=1,
    ):
        gate_model = build_gate_model()

        gate_model.fit(
            gate_matrix.iloc[
                fit_indices
            ],
            gate_labels[
                fit_indices
            ],
            sample_weight=gate_weights[
                fit_indices
            ],
            verbose=False,
        )

        oof_probability[
            holdout_indices
        ] = gate_model.predict_proba(
            gate_matrix.iloc[
                holdout_indices
            ]
        )[:, 1]

        oof_fold[
            holdout_indices
        ] = fold_number

    if not np.isfinite(
        oof_probability
    ).all():
        raise RuntimeError(
            "Gate OOF probabilities are incomplete."
        )

    return oof_probability, oof_fold


def select_gate_threshold(
    training_catalog: pd.DataFrame,
    candidate_prediction: np.ndarray,
    gate_oof_probability: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    """Select a gate threshold using OOF training predictions only."""

    z_spec = training_catalog[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

    z_eazy = training_catalog[
        EAZY_HYBRID_FEATURE
    ].to_numpy(dtype=float)

    threshold_rows = []

    for threshold in GATE_THRESHOLD_GRID:
        correction_applied = (
            gate_oof_probability
            >= threshold
        )

        gated_prediction = np.where(
            correction_applied,
            candidate_prediction,
            z_eazy,
        )

        metrics = compute_photoz_metrics(
            z_spec,
            gated_prediction,
        )

        threshold_rows.append(
            {
                "gate_threshold": float(
                    threshold
                ),
                "correction_fraction": float(
                    correction_applied.mean()
                ),
                **metrics,
            }
        )

    threshold_table = pd.DataFrame(
        threshold_rows
    )

    threshold_table[
        "absolute_normalized_bias"
    ] = threshold_table[
        "normalized_bias"
    ].abs()

    threshold_table = threshold_table.sort_values(
        [
            "catastrophic_outlier_fraction",
            "sigma_nmad",
            "absolute_normalized_bias",
            "mean_absolute_redshift_error",
        ],
        ascending=True,
    ).reset_index(drop=True)

    selected_threshold = float(
        threshold_table.loc[
            0,
            "gate_threshold",
        ]
    )

    return selected_threshold, threshold_table


def fit_final_residual_model(
    training_catalog: pd.DataFrame,
    validation_catalog: pd.DataFrame,
) -> tuple[XGBRegressor, np.ndarray]:
    """Fit the final residual model with validation early stopping."""

    training_matrix = prepare_matrix(
        training_catalog,
        RESIDUAL_FEATURE_COLUMNS,
    )

    validation_matrix = prepare_matrix(
        validation_catalog,
        RESIDUAL_FEATURE_COLUMNS,
    )

    training_target = (
        calculate_log_residual_target(
            training_catalog
        )
    )

    validation_target = (
        calculate_log_residual_target(
            validation_catalog
        )
    )

    training_weights = _numeric_series(
        training_catalog[
            SAMPLE_WEIGHT_COLUMN
        ]
    ).to_numpy(dtype=float)

    model = build_residual_model(
        final_model=True
    )

    model.fit(
        training_matrix,
        training_target,
        sample_weight=training_weights,
        eval_set=[
            (
                validation_matrix,
                validation_target,
            )
        ],
        verbose=False,
    )

    validation_residual_prediction = (
        model.predict(
            validation_matrix
        )
    )

    return (
        model,
        validation_residual_prediction,
    )


def fit_final_gate_model(
    gate_matrix: pd.DataFrame,
    gate_labels: np.ndarray,
    gate_weights: np.ndarray,
) -> XGBClassifier:
    """Fit the final gate using all leakage-safe OOF training rows."""

    model = build_gate_model()

    model.fit(
        gate_matrix,
        gate_labels,
        sample_weight=gate_weights,
        verbose=False,
    )

    return model


def load_direct_hybrid_prediction(
    validation_catalog: pd.DataFrame,
) -> np.ndarray:
    """Load the previous direct-hybrid validation prediction."""

    if not VALIDATION_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Run the controlled XGBoost experiments before "
            "running the residual hybrid."
        )

    predictions = pd.read_csv(
        VALIDATION_PREDICTIONS_PATH,
        low_memory=False,
    )

    direct_hybrid = predictions.loc[
        predictions["experiment_id"]
        == "xgb_eazy_hybrid",
        [
            SOURCE_KEY_COLUMN,
            "z_prediction",
        ],
    ].copy()

    if not direct_hybrid[
        SOURCE_KEY_COLUMN
    ].is_unique:
        raise ValueError(
            "Direct-hybrid validation predictions are duplicated."
        )

    prediction_map = direct_hybrid.set_index(
        SOURCE_KEY_COLUMN
    )["z_prediction"]

    aligned_prediction = (
        validation_catalog[
            SOURCE_KEY_COLUMN
        ]
        .map(prediction_map)
        .to_numpy(dtype=float)
    )

    if not np.isfinite(
        aligned_prediction
    ).all():
        raise ValueError(
            "Direct-hybrid predictions could not be aligned "
            "with validation sources."
        )

    return aligned_prediction


def summarize_hybrid_method(
    validation_catalog: pd.DataFrame,
    experiment_id: str,
    method: str,
    prediction: np.ndarray,
    correction_applied: np.ndarray,
    gate_threshold: float,
) -> tuple[dict, pd.DataFrame]:
    """Calculate metrics and source-level predictions for one method."""

    metrics = compute_photoz_metrics(
        validation_catalog[TARGET_COLUMN],
        prediction,
    )

    metrics.update(
        {
            "experiment_id": experiment_id,
            "method": method,
            "correction_fraction": float(
                correction_applied.mean()
            ),
            "gate_threshold": gate_threshold,
        }
    )

    prediction_table = build_prediction_table(
        validation_catalog,
        experiment_id=experiment_id,
        method=method,
        predicted_redshift=prediction,
    )

    prediction_table[
        "correction_applied"
    ] = correction_applied

    return metrics, prediction_table


def save_hybrid_products(
    metrics_table: pd.DataFrame,
    prediction_table: pd.DataFrame,
    oof_diagnostics: pd.DataFrame,
    threshold_table: pd.DataFrame,
    selected_threshold: float,
    residual_model: XGBRegressor,
    gate_model: XGBClassifier,
) -> None:
    """Save hybrid models, predictions, metrics, and configuration."""

    ensure_output_directories()

    metrics_table.to_csv(
        HYBRID_VALIDATION_METRICS_PATH,
        index=False,
    )

    prediction_table.to_csv(
        HYBRID_VALIDATION_PREDICTIONS_PATH,
        index=False,
    )

    oof_diagnostics.to_csv(
        HYBRID_OOF_PREDICTIONS_PATH,
        index=False,
    )

    threshold_table.to_csv(
        GATE_THRESHOLD_METRICS_PATH,
        index=False,
    )

    residual_model.save_model(
        RESIDUAL_MODEL_PATH
    )

    gate_model.save_model(
        GATE_MODEL_PATH
    )

    manifest = {
        "selected_gate_threshold": (
            selected_threshold
        ),
        "oof_folds": OOF_FOLDS,
        "minimum_correction_improvement": (
            MINIMUM_CORRECTION_IMPROVEMENT
        ),
        "maximum_predicted_redshift": (
            MAXIMUM_PREDICTED_REDSHIFT
        ),
        "residual_feature_columns": list(
            RESIDUAL_FEATURE_COLUMNS
        ),
        "gate_feature_columns": list(
            GATE_FEATURE_COLUMNS
        ),
        "oof_residual_parameters": (
            OOF_RESIDUAL_PARAMETERS
        ),
        "final_residual_parameters": (
            FINAL_RESIDUAL_PARAMETERS
        ),
        "gate_parameters": GATE_PARAMETERS,
    }

    with HYBRID_MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            manifest,
            file_handle,
            indent=2,
        )


def run_gated_residual_hybrid(
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run residual learning and gating without opening test data."""

    ensure_output_directories()

    (
        training_catalog,
        validation_catalog,
    ) = load_hybrid_catalogs()

    training_catalog = (
        add_eazy_diagnostic_features(
            training_catalog
        )
    )

    validation_catalog = (
        add_eazy_diagnostic_features(
            validation_catalog
        )
    )

    (
        residual_oof_prediction,
        residual_oof_fold,
    ) = generate_residual_oof_predictions(
        training_catalog
    )

    (
        gate_training_matrix,
        gate_labels,
        oof_diagnostics,
    ) = build_gate_training_data(
        training_catalog,
        residual_oof_prediction,
    )

    source_weights = _numeric_series(
        training_catalog[
            SAMPLE_WEIGHT_COLUMN
        ]
    ).to_numpy(dtype=float)

    gate_weights = build_gate_sample_weights(
        gate_labels,
        source_weights,
    )

    (
        gate_oof_probability,
        gate_oof_fold,
    ) = generate_gate_oof_probabilities(
        gate_training_matrix,
        gate_labels,
        gate_weights,
    )

    oof_diagnostics[
        "residual_oof_fold"
    ] = residual_oof_fold

    oof_diagnostics[
        "gate_oof_fold"
    ] = gate_oof_fold

    oof_diagnostics[
        "gate_oof_probability"
    ] = gate_oof_probability

    candidate_oof_prediction = (
        oof_diagnostics[
            "candidate_residual_redshift"
        ].to_numpy(dtype=float)
    )

    (
        selected_threshold,
        threshold_table,
    ) = select_gate_threshold(
        training_catalog,
        candidate_oof_prediction,
        gate_oof_probability,
    )

    (
        residual_model,
        validation_residual_prediction,
    ) = fit_final_residual_model(
        training_catalog,
        validation_catalog,
    )

    residual_validation_prediction = (
        apply_log_residual_correction(
            validation_catalog[
                EAZY_HYBRID_FEATURE
            ],
            validation_residual_prediction,
        )
    )

    gate_model = fit_final_gate_model(
        gate_training_matrix,
        gate_labels,
        gate_weights,
    )

    validation_gate_catalog = (
        validation_catalog.copy()
    )

    validation_gate_catalog[
        PREDICTED_RESIDUAL_FEATURE
    ] = validation_residual_prediction

    validation_gate_catalog[
        ABSOLUTE_PREDICTED_RESIDUAL_FEATURE
    ] = np.abs(
        validation_residual_prediction
    )

    validation_gate_matrix = prepare_matrix(
        validation_gate_catalog,
        GATE_FEATURE_COLUMNS,
    )

    validation_gate_probability = (
        gate_model.predict_proba(
            validation_gate_matrix
        )[:, 1]
    )

    correction_applied = (
        validation_gate_probability
        >= selected_threshold
    )

    eazy_prediction = validation_catalog[
        EAZY_HYBRID_FEATURE
    ].to_numpy(dtype=float)

    gated_validation_prediction = np.where(
        correction_applied,
        residual_validation_prediction,
        eazy_prediction,
    )

    direct_hybrid_prediction = (
        load_direct_hybrid_prediction(
            validation_catalog
        )
    )

    method_definitions = (
        (
            "eazy_dr5_za",
            "EAZY DR5 z_a",
            eazy_prediction,
            np.zeros(
                len(validation_catalog),
                dtype=bool,
            ),
            np.nan,
        ),
        (
            "xgb_eazy_direct_hybrid",
            "XGBoost: direct EAZY hybrid",
            direct_hybrid_prediction,
            np.ones(
                len(validation_catalog),
                dtype=bool,
            ),
            np.nan,
        ),
        (
            "xgb_eazy_residual_hybrid",
            "XGBoost: EAZY residual hybrid",
            residual_validation_prediction,
            np.ones(
                len(validation_catalog),
                dtype=bool,
            ),
            0.0,
        ),
        (
            "xgb_eazy_gated_residual",
            "XGBoost: gated residual hybrid",
            gated_validation_prediction,
            correction_applied,
            selected_threshold,
        ),
    )

    all_metrics = []
    all_predictions = []

    for (
        experiment_id,
        method,
        prediction,
        method_correction_applied,
        method_threshold,
    ) in method_definitions:
        metrics, predictions = (
            summarize_hybrid_method(
                validation_catalog,
                experiment_id,
                method,
                prediction,
                method_correction_applied,
                method_threshold,
            )
        )

        if (
            experiment_id
            == "xgb_eazy_gated_residual"
        ):
            predictions[
                "gate_probability"
            ] = validation_gate_probability
        else:
            predictions[
                "gate_probability"
            ] = np.nan

        all_metrics.append(metrics)
        all_predictions.append(predictions)

    metrics_table = pd.DataFrame(
        all_metrics
    )

    metrics_table = metrics_table.sort_values(
        [
            "catastrophic_outlier_fraction",
            "sigma_nmad",
        ],
        ascending=True,
    ).reset_index(drop=True)

    metrics_table.insert(
        0,
        "validation_rank",
        np.arange(
            1,
            len(metrics_table) + 1,
        ),
    )

    prediction_table = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    save_hybrid_products(
        metrics_table,
        prediction_table,
        oof_diagnostics,
        threshold_table,
        selected_threshold,
        residual_model,
        gate_model,
    )

    return metrics_table, prediction_table