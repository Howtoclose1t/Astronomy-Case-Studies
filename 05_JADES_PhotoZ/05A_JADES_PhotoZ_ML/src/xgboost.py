"""Run controlled XGBoost Photo-z experiments on the validation set."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor

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
    AUGMENTED_TRAINING_FEATURES_PATH,
    EAZY_HYBRID_FEATURE,
    FULL_FEATURE_COLUMNS,
    HYBRID_FEATURE_COLUMNS,
    HYBRID_TRAINING_ALLOWED_COLUMN,
    RAW_FEATURE_COLUMNS,
    VALIDATION_FEATURES_PATH,
    WEIGHTED_TRAINING_FEATURES_PATH,
)
from .resampling import (
    SAMPLE_ID_COLUMN,
    SAMPLE_WEIGHT_COLUMN,
)


REDSHIFT_STRATUM_COLUMN = "redshift_stratum"
SPLIT_COLUMN = "split"

NORMALIZED_ERROR_COLUMN = "normalized_redshift_error"
ABSOLUTE_NORMALIZED_ERROR_COLUMN = (
    "absolute_normalized_redshift_error"
)
CATASTROPHIC_COLUMN = "is_catastrophic_outlier"

CATASTROPHIC_OUTLIER_THRESHOLD = 0.15

VALIDATION_METRICS_PATH = (
    METRICS_DIR
    / "dr45_xgboost_validation_metrics.csv"
)

VALIDATION_PREDICTIONS_PATH = (
    PREDICTION_DIR
    / "dr45_xgboost_validation_predictions.csv"
)

EXPERIMENT_MANIFEST_PATH = (
    METRICS_DIR
    / "dr45_xgboost_experiment_manifest.json"
)


@dataclass(frozen=True)
class ExperimentSpec:
    """Define one controlled XGBoost validation experiment."""

    experiment_id: str
    method: str
    training_catalog_key: str
    feature_columns: tuple[str, ...]
    use_sample_weights: bool
    uses_augmentation: bool
    uses_eazy: bool


EXPERIMENT_SPECS = (
    ExperimentSpec(
        experiment_id="xgb_raw_unweighted",
        method="XGBoost: raw photometry",
        training_catalog_key="weighted",
        feature_columns=RAW_FEATURE_COLUMNS,
        use_sample_weights=False,
        uses_augmentation=False,
        uses_eazy=False,
    ),
    ExperimentSpec(
        experiment_id="xgb_full_unweighted",
        method="XGBoost: full features",
        training_catalog_key="weighted",
        feature_columns=FULL_FEATURE_COLUMNS,
        use_sample_weights=False,
        uses_augmentation=False,
        uses_eazy=False,
    ),
    ExperimentSpec(
        experiment_id="xgb_full_weighted",
        method="XGBoost: full + weights",
        training_catalog_key="weighted",
        feature_columns=FULL_FEATURE_COLUMNS,
        use_sample_weights=True,
        uses_augmentation=False,
        uses_eazy=False,
    ),
    ExperimentSpec(
        experiment_id="xgb_full_augmented",
        method="XGBoost: full + weights + augmentation",
        training_catalog_key="augmented",
        feature_columns=FULL_FEATURE_COLUMNS,
        use_sample_weights=True,
        uses_augmentation=True,
        uses_eazy=False,
    ),
    ExperimentSpec(
        experiment_id="xgb_eazy_hybrid",
        method="XGBoost: EAZY hybrid",
        training_catalog_key="weighted",
        feature_columns=HYBRID_FEATURE_COLUMNS,
        use_sample_weights=True,
        uses_augmentation=False,
        uses_eazy=True,
    ),
)


MODEL_PARAMETERS = {
    "objective": "reg:pseudohubererror",
    "n_estimators": 5000,
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_child_weight": 5.0,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.05,
    "reg_lambda": 2.0,
    "gamma": 0.0,
    "max_bin": 256,
    "tree_method": "hist",
    "eval_metric": "mae",
    "early_stopping_rounds": 150,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": 0,
}


def load_feature_catalog(
    path,
    expected_split: str,
    catalog_name: str,
) -> pd.DataFrame:
    """Load and validate one feature catalog."""

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
        HYBRID_TRAINING_ALLOWED_COLUMN,
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
        unexpected_splits = (
            catalog[SPLIT_COLUMN]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{catalog_name} contains unexpected split values: "
            f"{unexpected_splits}"
        )

    if not catalog[SAMPLE_ID_COLUMN].is_unique:
        raise ValueError(
            f"{catalog_name} contains duplicated sample IDs."
        )

    catalog[TARGET_COLUMN] = pd.to_numeric(
        catalog[TARGET_COLUMN],
        errors="coerce",
    )

    if not np.isfinite(
        catalog[TARGET_COLUMN]
    ).all():
        raise ValueError(
            f"{catalog_name} contains invalid z_spec values."
        )

    return catalog


def load_experiment_catalogs() -> dict[str, pd.DataFrame]:
    """Load training alternatives and the untouched validation set."""

    catalogs = {
        "weighted": load_feature_catalog(
            WEIGHTED_TRAINING_FEATURES_PATH,
            expected_split="train",
            catalog_name="weighted training features",
        ),
        "augmented": load_feature_catalog(
            AUGMENTED_TRAINING_FEATURES_PATH,
            expected_split="train",
            catalog_name="augmented training features",
        ),
        "validation": load_feature_catalog(
            VALIDATION_FEATURES_PATH,
            expected_split="validation",
            catalog_name="validation features",
        ),
    }

    training_sources = set(
        catalogs["weighted"][SOURCE_KEY_COLUMN]
    )

    augmented_sources = set(
        catalogs["augmented"][SOURCE_KEY_COLUMN]
    )

    validation_sources = set(
        catalogs["validation"][SOURCE_KEY_COLUMN]
    )

    if training_sources != augmented_sources:
        raise ValueError(
            "The weighted and augmented training catalogs do not "
            "contain the same physical sources."
        )

    overlap = training_sources & validation_sources

    if overlap:
        overlap_examples = sorted(overlap)[:10]

        raise ValueError(
            "Training-validation source leakage was detected: "
            f"{overlap_examples}"
        )

    return catalogs


def prepare_feature_matrix(
    catalog: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Create a numeric XGBoost matrix while retaining genuine NaN values."""

    missing_columns = sorted(
        set(feature_columns) - set(catalog.columns)
    )

    if missing_columns:
        raise KeyError(
            "The requested model features are missing: "
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

    completely_missing_columns = (
        feature_matrix.columns[
            feature_matrix.isna().all()
        ].tolist()
    )

    if completely_missing_columns:
        raise ValueError(
            "Some model features are completely missing: "
            f"{completely_missing_columns}"
        )

    return feature_matrix


def transform_redshift_target(
    redshift: pd.Series,
) -> np.ndarray:
    """Transform redshift into log(1 + z) for robust normalized learning."""

    redshift_values = pd.to_numeric(
        redshift,
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(redshift_values).all():
        raise ValueError(
            "The target contains non-finite redshifts."
        )

    if (redshift_values < 0).any():
        raise ValueError(
            "The target contains negative redshifts."
        )

    return np.log1p(redshift_values)


def inverse_transform_redshift(
    transformed_prediction: np.ndarray,
) -> np.ndarray:
    """Convert model predictions from log(1 + z) back to redshift."""

    redshift_prediction = np.expm1(
        np.asarray(
            transformed_prediction,
            dtype=float,
        )
    )

    return np.clip(
        redshift_prediction,
        a_min=0.0,
        a_max=None,
    )


def compute_photoz_metrics(
    true_redshift,
    predicted_redshift,
) -> dict[str, float]:
    """Calculate standard point-estimate Photo-z validation metrics."""

    true_values = np.asarray(
        true_redshift,
        dtype=float,
    )

    predicted_values = np.asarray(
        predicted_redshift,
        dtype=float,
    )

    finite = (
        np.isfinite(true_values)
        & np.isfinite(predicted_values)
    )

    if not finite.any():
        raise ValueError(
            "No finite prediction-target pairs are available."
        )

    true_values = true_values[finite]
    predicted_values = predicted_values[finite]

    normalized_error = (
        predicted_values - true_values
    ) / (1.0 + true_values)

    normalized_bias = float(
        np.median(normalized_error)
    )

    normalized_median_absolute_error = float(
        np.median(
            np.abs(normalized_error)
        )
    )

    sigma_nmad = float(
        1.4826
        * np.median(
            np.abs(
                normalized_error
                - normalized_bias
            )
        )
    )

    absolute_redshift_error = np.abs(
        predicted_values - true_values
    )

    catastrophic = (
        np.abs(normalized_error)
        > CATASTROPHIC_OUTLIER_THRESHOLD
    )

    return {
        "sources_with_finite_prediction_and_z_spec": int(
            len(true_values)
        ),
        "normalized_bias": normalized_bias,
        "normalized_median_absolute_error": (
            normalized_median_absolute_error
        ),
        "sigma_nmad": sigma_nmad,
        "mean_absolute_redshift_error": float(
            absolute_redshift_error.mean()
        ),
        "catastrophic_outlier_threshold": (
            CATASTROPHIC_OUTLIER_THRESHOLD
        ),
        "catastrophic_outliers": int(
            catastrophic.sum()
        ),
        "catastrophic_outlier_fraction": float(
            catastrophic.mean()
        ),
    }


def build_prediction_table(
    validation_catalog: pd.DataFrame,
    experiment_id: str,
    method: str,
    predicted_redshift,
) -> pd.DataFrame:
    """Build source-level validation predictions and residual flags."""

    prediction_table = validation_catalog.loc[
        :,
        [
            SOURCE_KEY_COLUMN,
            TARGET_COLUMN,
            REDSHIFT_STRATUM_COLUMN,
        ],
    ].copy()

    prediction_table["experiment_id"] = (
        experiment_id
    )

    prediction_table["method"] = method

    prediction_table["z_prediction"] = np.asarray(
        predicted_redshift,
        dtype=float,
    )

    prediction_table[NORMALIZED_ERROR_COLUMN] = (
        (
            prediction_table["z_prediction"]
            - prediction_table[TARGET_COLUMN]
        )
        / (
            1.0
            + prediction_table[TARGET_COLUMN]
        )
    )

    prediction_table[
        ABSOLUTE_NORMALIZED_ERROR_COLUMN
    ] = prediction_table[
        NORMALIZED_ERROR_COLUMN
    ].abs()

    prediction_table[CATASTROPHIC_COLUMN] = (
        prediction_table[
            ABSOLUTE_NORMALIZED_ERROR_COLUMN
        ]
        > CATASTROPHIC_OUTLIER_THRESHOLD
    )

    return prediction_table


def build_xgboost_model() -> XGBRegressor:
    """Create the common XGBoost 3.2 regression configuration."""

    return XGBRegressor(**MODEL_PARAMETERS)


def run_single_experiment(
    spec: ExperimentSpec,
    training_catalog: pd.DataFrame,
    validation_catalog: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Train and evaluate one controlled validation experiment."""

    if spec.uses_eazy:
        hybrid_allowed = training_catalog[
            HYBRID_TRAINING_ALLOWED_COLUMN
        ].astype(bool)

        training_catalog = training_catalog.loc[
            hybrid_allowed
        ].copy()

    training_features = prepare_feature_matrix(
        training_catalog,
        spec.feature_columns,
    )

    validation_features = prepare_feature_matrix(
        validation_catalog,
        spec.feature_columns,
    )

    transformed_training_target = (
        transform_redshift_target(
            training_catalog[TARGET_COLUMN]
        )
    )

    transformed_validation_target = (
        transform_redshift_target(
            validation_catalog[TARGET_COLUMN]
        )
    )

    fit_arguments = {
        "X": training_features,
        "y": transformed_training_target,
        "eval_set": [
            (
                validation_features,
                transformed_validation_target,
            )
        ],
        "verbose": False,
    }

    if spec.use_sample_weights:
        sample_weights = pd.to_numeric(
            training_catalog[
                SAMPLE_WEIGHT_COLUMN
            ],
            errors="coerce",
        ).to_numpy(dtype=float)

        if (
            not np.isfinite(sample_weights).all()
            or (sample_weights <= 0).any()
        ):
            raise ValueError(
                f"{spec.experiment_id} has invalid sample weights."
            )

        fit_arguments["sample_weight"] = (
            sample_weights
        )

    model = build_xgboost_model()

    model.fit(**fit_arguments)

    transformed_prediction = model.predict(
        validation_features
    )

    predicted_redshift = (
        inverse_transform_redshift(
            transformed_prediction
        )
    )

    metrics = compute_photoz_metrics(
        validation_catalog[TARGET_COLUMN],
        predicted_redshift,
    )

    metrics.update(
        {
            "experiment_id": spec.experiment_id,
            "method": spec.method,
            "training_rows": len(
                training_catalog
            ),
            "training_physical_sources": (
                training_catalog[
                    SOURCE_KEY_COLUMN
                ].nunique()
            ),
            "features": len(
                spec.feature_columns
            ),
            "uses_sample_weights": (
                spec.use_sample_weights
            ),
            "uses_augmentation": (
                spec.uses_augmentation
            ),
            "uses_eazy": spec.uses_eazy,
            "trees_used": int(
                model.best_iteration + 1
            ),
            "early_stopping_score": float(
                model.best_score
            ),
        }
    )

    prediction_table = build_prediction_table(
        validation_catalog,
        experiment_id=spec.experiment_id,
        method=spec.method,
        predicted_redshift=predicted_redshift,
    )

    model_path = (
        MODEL_DIR
        / f"{spec.experiment_id}.json"
    )

    model.save_model(model_path)

    return metrics, prediction_table


def evaluate_eazy_benchmark(
    validation_catalog: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate the published DR5 EAZY z_a on the same validation set."""

    eazy_prediction = pd.to_numeric(
        validation_catalog[
            EAZY_HYBRID_FEATURE
        ],
        errors="coerce",
    ).to_numpy(dtype=float)

    metrics = compute_photoz_metrics(
        validation_catalog[TARGET_COLUMN],
        eazy_prediction,
    )

    metrics.update(
        {
            "experiment_id": "eazy_dr5_za",
            "method": "EAZY DR5 z_a",
            "training_rows": 0,
            "training_physical_sources": 0,
            "features": 0,
            "uses_sample_weights": False,
            "uses_augmentation": False,
            "uses_eazy": True,
            "trees_used": 0,
            "early_stopping_score": np.nan,
        }
    )

    prediction_table = build_prediction_table(
        validation_catalog,
        experiment_id="eazy_dr5_za",
        method="EAZY DR5 z_a",
        predicted_redshift=eazy_prediction,
    )

    return metrics, prediction_table


def save_experiment_products(
    metrics_table: pd.DataFrame,
    prediction_table: pd.DataFrame,
) -> None:
    """Save validation metrics, predictions, and experiment definitions."""

    ensure_output_directories()

    metrics_table.to_csv(
        VALIDATION_METRICS_PATH,
        index=False,
    )

    prediction_table.to_csv(
        VALIDATION_PREDICTIONS_PATH,
        index=False,
    )

    manifest = {
        "xgboost_version": xgboost.__version__,
        "selection_rule": (
            "Minimize catastrophic outlier fraction first; "
            "use sigma_nmad as the secondary criterion."
        ),
        "target_transformation": "log1p(z_spec)",
        "model_parameters": MODEL_PARAMETERS,
        "experiments": [
            {
                **asdict(spec),
                "feature_columns": list(
                    spec.feature_columns
                ),
            }
            for spec in EXPERIMENT_SPECS
        ],
    }

    with EXPERIMENT_MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            manifest,
            file_handle,
            indent=2,
        )


def run_validation_experiments(
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run EAZY and five XGBoost experiments without opening test data."""

    ensure_output_directories()

    catalogs = load_experiment_catalogs()

    validation_catalog = catalogs[
        "validation"
    ]

    all_metrics = []
    all_predictions = []

    eazy_metrics, eazy_predictions = (
        evaluate_eazy_benchmark(
            validation_catalog
        )
    )

    all_metrics.append(eazy_metrics)
    all_predictions.append(eazy_predictions)

    for spec in EXPERIMENT_SPECS:
        training_catalog = catalogs[
            spec.training_catalog_key
        ]

        metrics, predictions = (
            run_single_experiment(
                spec,
                training_catalog,
                validation_catalog,
            )
        )

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

    save_experiment_products(
        metrics_table,
        prediction_table,
    )

    return metrics_table, prediction_table