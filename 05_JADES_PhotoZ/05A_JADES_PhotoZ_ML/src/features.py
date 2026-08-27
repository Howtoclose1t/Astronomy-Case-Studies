"""Construct physically motivated JADES photometric-redshift features."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import (
    CORE_FILTERS,
    MODEL_DIR,
    REFERENCE_PREDICTION_COLUMN,
    SOURCE_KEY_COLUMN,
    TARGET_COLUMN,
    TEST_CATALOG_PATH,
    VALIDATION_CATALOG_PATH,
)
from .resampling import (
    AUGMENTED_TRAINING_CATALOG_PATH,
    AUGMENTATION_ITERATION_COLUMN,
    HYBRID_TRAINING_ALLOWED_COLUMN,
    IS_AUGMENTED_COLUMN,
    SAMPLE_ID_COLUMN,
    SAMPLE_WEIGHT_COLUMN,
    SOURCE_WEIGHT_COLUMN,
    WEIGHTED_TRAINING_CATALOG_PATH,
)


REDSHIFT_STRATUM_COLUMN = "redshift_stratum"
SPLIT_COLUMN = "split"

AB_ZERO_POINT_FLUX_NJY = 3.631e12
ASINH_MAGNITUDE_COEFFICIENT = 2.5 / np.log(10.0)

ADJACENT_COLOR_PAIRS = (
    ("F090W", "F115W"),
    ("F115W", "F150W"),
    ("F150W", "F200W"),
    ("F200W", "F277W"),
    ("F277W", "F335M"),
    ("F335M", "F356W"),
    ("F356W", "F410M"),
    ("F410M", "F444W"),
)

LONG_BASELINE_COLOR_PAIRS = (
    ("F090W", "F200W"),
    ("F150W", "F277W"),
    ("F200W", "F444W"),
)

COLOR_PAIRS = (
    ADJACENT_COLOR_PAIRS
    + LONG_BASELINE_COLOR_PAIRS
)


RAW_FLUX_FEATURE_COLUMNS = tuple(
    f"x_flux_{filter_name.lower()}_njy"
    for filter_name in CORE_FILTERS
)

FLUX_ERROR_FEATURE_COLUMNS = tuple(
    f"x_fluxerr_{filter_name.lower()}_njy"
    for filter_name in CORE_FILTERS
)

ASINH_MAGNITUDE_FEATURE_COLUMNS = tuple(
    f"x_asinh_mag_{filter_name.lower()}"
    for filter_name in CORE_FILTERS
)

LOG_SNR_FEATURE_COLUMNS = tuple(
    f"x_log_snr_{filter_name.lower()}"
    for filter_name in CORE_FILTERS
)

VALIDITY_FEATURE_COLUMNS = tuple(
    f"x_valid_{filter_name.lower()}"
    for filter_name in CORE_FILTERS
)

COLOR_FEATURE_COLUMNS = tuple(
    (
        f"x_color_{first_filter.lower()}"
        f"_{second_filter.lower()}"
    )
    for first_filter, second_filter in COLOR_PAIRS
)

CORE_FILTER_COUNT_FEATURE = "x_n_valid_core_filters"
EAZY_HYBRID_FEATURE = "x_eazy_z_a"

RAW_FEATURE_COLUMNS = (
    RAW_FLUX_FEATURE_COLUMNS
    + FLUX_ERROR_FEATURE_COLUMNS
    + VALIDITY_FEATURE_COLUMNS
    + (CORE_FILTER_COUNT_FEATURE,)
)

PHYSICAL_FEATURE_COLUMNS = (
    ASINH_MAGNITUDE_FEATURE_COLUMNS
    + LOG_SNR_FEATURE_COLUMNS
    + COLOR_FEATURE_COLUMNS
    + VALIDITY_FEATURE_COLUMNS
    + (CORE_FILTER_COUNT_FEATURE,)
)

FULL_FEATURE_COLUMNS = (
    RAW_FLUX_FEATURE_COLUMNS
    + FLUX_ERROR_FEATURE_COLUMNS
    + ASINH_MAGNITUDE_FEATURE_COLUMNS
    + LOG_SNR_FEATURE_COLUMNS
    + COLOR_FEATURE_COLUMNS
    + VALIDITY_FEATURE_COLUMNS
    + (CORE_FILTER_COUNT_FEATURE,)
)

HYBRID_FEATURE_COLUMNS = (
    FULL_FEATURE_COLUMNS
    + (EAZY_HYBRID_FEATURE,)
)


PROCESSED_DATA_DIRECTORY = (
    WEIGHTED_TRAINING_CATALOG_PATH.parent
)

WEIGHTED_TRAINING_FEATURES_PATH = (
    PROCESSED_DATA_DIRECTORY
    / "jades_dr45_weighted_training_features.csv"
)

AUGMENTED_TRAINING_FEATURES_PATH = (
    PROCESSED_DATA_DIRECTORY
    / "jades_dr45_augmented_training_features.csv"
)

VALIDATION_FEATURES_PATH = (
    PROCESSED_DATA_DIRECTORY
    / "jades_dr45_validation_features.csv"
)

TEST_FEATURES_PATH = (
    PROCESSED_DATA_DIRECTORY
    / "jades_dr45_test_features.csv"
)

FEATURE_SUMMARY_PATH = (
    PROCESSED_DATA_DIRECTORY
    / "jades_dr45_feature_summary.csv"
)

FEATURE_STATE_PATH = (
    MODEL_DIR
    / "jades_dr45_feature_state.json"
)


def _input_column_names() -> set[str]:
    """Return the input columns required for feature construction."""

    required_columns = {
        SOURCE_KEY_COLUMN,
        TARGET_COLUMN,
        REFERENCE_PREDICTION_COLUMN,
        REDSHIFT_STRATUM_COLUMN,
        SPLIT_COLUMN,
    }

    for filter_name in CORE_FILTERS:
        normalized_filter = filter_name.lower()

        required_columns.update(
            {
                f"flux_{normalized_filter}_njy",
                f"fluxerr_{normalized_filter}_njy",
                f"valid_{normalized_filter}",
            }
        )

    return required_columns


def _validate_input_columns(
    catalog: pd.DataFrame,
    catalog_name: str,
) -> None:
    """Check that a catalog has every feature-construction input."""

    missing_columns = sorted(
        _input_column_names() - set(catalog.columns)
    )

    if missing_columns:
        raise KeyError(
            f"{catalog_name} is missing required columns: "
            f"{missing_columns}"
        )


def _coerce_boolean_series(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """Convert boolean-like catalog values without silent failures."""

    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    converted = normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
    )

    if converted.isna().any():
        unexpected_values = (
            series.loc[converted.isna()]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{column_name} contains unsupported values: "
            f"{unexpected_values[:10]}"
        )

    return converted.astype(bool)


def _numeric_series(series: pd.Series) -> pd.Series:
    """Convert a column to finite numeric values or missing values."""

    numeric_values = pd.to_numeric(
        series,
        errors="coerce",
    )

    return numeric_values.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _ensure_sample_metadata(
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Add consistent metadata to unaugmented validation and test data."""

    result = catalog.copy()

    if SAMPLE_ID_COLUMN not in result.columns:
        result[SAMPLE_ID_COLUMN] = (
            result[SOURCE_KEY_COLUMN].astype(str)
            + "::original"
        )

    if SOURCE_WEIGHT_COLUMN not in result.columns:
        result[SOURCE_WEIGHT_COLUMN] = 1.0

    if SAMPLE_WEIGHT_COLUMN not in result.columns:
        result[SAMPLE_WEIGHT_COLUMN] = 1.0

    if IS_AUGMENTED_COLUMN not in result.columns:
        result[IS_AUGMENTED_COLUMN] = False

    if AUGMENTATION_ITERATION_COLUMN not in result.columns:
        result[AUGMENTATION_ITERATION_COLUMN] = 0

    if HYBRID_TRAINING_ALLOWED_COLUMN not in result.columns:
        result[HYBRID_TRAINING_ALLOWED_COLUMN] = True

    result[IS_AUGMENTED_COLUMN] = _coerce_boolean_series(
        result[IS_AUGMENTED_COLUMN],
        IS_AUGMENTED_COLUMN,
    )

    result[HYBRID_TRAINING_ALLOWED_COLUMN] = (
        _coerce_boolean_series(
            result[HYBRID_TRAINING_ALLOWED_COLUMN],
            HYBRID_TRAINING_ALLOWED_COLUMN,
        )
    )

    return result


def load_feature_input_catalog(
    path,
    catalog_name: str,
) -> pd.DataFrame:
    """Load one catalog used by the feature-construction stage."""

    if not path.exists():
        raise FileNotFoundError(
            f"{catalog_name} was not found at {path}"
        )

    catalog = pd.read_csv(
        path,
        low_memory=False,
    )

    _validate_input_columns(
        catalog,
        catalog_name,
    )

    catalog = _ensure_sample_metadata(catalog)

    return catalog


def fit_feature_state(
    weighted_training_catalog: pd.DataFrame,
) -> dict:
    """Estimate asinh softening scales from original training data only."""

    softening_flux_njy = {}

    for filter_name in CORE_FILTERS:
        normalized_filter = filter_name.lower()

        error_column = (
            f"fluxerr_{normalized_filter}_njy"
        )

        validity_column = (
            f"valid_{normalized_filter}"
        )

        error_values = _numeric_series(
            weighted_training_catalog[error_column]
        )

        validity_values = _coerce_boolean_series(
            weighted_training_catalog[validity_column],
            validity_column,
        )

        usable_errors = error_values.loc[
            validity_values
            & error_values.notna()
            & (error_values > 0)
        ]

        if usable_errors.empty:
            raise ValueError(
                f"No positive training uncertainties exist for "
                f"{filter_name}."
            )

        softening_flux_njy[filter_name] = float(
            usable_errors.median()
        )

    return {
        "ab_zero_point_flux_njy": AB_ZERO_POINT_FLUX_NJY,
        "softening_flux_njy": softening_flux_njy,
        "raw_feature_columns": list(
            RAW_FEATURE_COLUMNS
        ),
        "physical_feature_columns": list(
            PHYSICAL_FEATURE_COLUMNS
        ),
        "full_feature_columns": list(
            FULL_FEATURE_COLUMNS
        ),
        "hybrid_feature_columns": list(
            HYBRID_FEATURE_COLUMNS
        ),
    }


def calculate_asinh_magnitude(
    flux_njy: pd.Series,
    softening_flux_njy: float,
) -> pd.Series:
    """Convert nJy flux into a low-S/N-safe AB asinh magnitude."""

    if not np.isfinite(softening_flux_njy):
        raise ValueError(
            "The asinh softening flux must be finite."
        )

    if softening_flux_njy <= 0:
        raise ValueError(
            "The asinh softening flux must be positive."
        )

    dimensionless_flux = (
        flux_njy / AB_ZERO_POINT_FLUX_NJY
    )

    dimensionless_softening = (
        softening_flux_njy
        / AB_ZERO_POINT_FLUX_NJY
    )

    magnitude = -ASINH_MAGNITUDE_COEFFICIENT * (
        np.arcsinh(
            dimensionless_flux
            / (2.0 * dimensionless_softening)
        )
        + np.log(dimensionless_softening)
    )

    return pd.Series(
        magnitude,
        index=flux_njy.index,
        dtype=float,
    )


def transform_catalog(
    catalog: pd.DataFrame,
    feature_state: dict,
) -> pd.DataFrame:
    """Create flux, uncertainty, S/N, asinh-magnitude, and color features."""

    transformed = catalog.copy()

    usable_filter_masks = {}

    for filter_name in CORE_FILTERS:
        normalized_filter = filter_name.lower()

        flux_column = (
            f"flux_{normalized_filter}_njy"
        )

        error_column = (
            f"fluxerr_{normalized_filter}_njy"
        )

        validity_column = (
            f"valid_{normalized_filter}"
        )

        output_flux_column = (
            f"x_flux_{normalized_filter}_njy"
        )

        output_error_column = (
            f"x_fluxerr_{normalized_filter}_njy"
        )

        output_magnitude_column = (
            f"x_asinh_mag_{normalized_filter}"
        )

        output_snr_column = (
            f"x_log_snr_{normalized_filter}"
        )

        output_validity_column = (
            f"x_valid_{normalized_filter}"
        )

        flux_values = _numeric_series(
            transformed[flux_column]
        )

        error_values = _numeric_series(
            transformed[error_column]
        )

        validity_values = _coerce_boolean_series(
            transformed[validity_column],
            validity_column,
        )

        usable_measurement = (
            validity_values
            & flux_values.notna()
            & error_values.notna()
            & (error_values > 0)
        )

        usable_filter_masks[filter_name] = (
            usable_measurement
        )

        masked_flux = flux_values.where(
            usable_measurement
        )

        masked_error = error_values.where(
            usable_measurement
        )

        transformed[output_flux_column] = (
            masked_flux
        )

        transformed[output_error_column] = (
            masked_error
        )

        transformed[output_validity_column] = (
            usable_measurement.astype("int8")
        )

        transformed[output_magnitude_column] = (
            calculate_asinh_magnitude(
                masked_flux,
                feature_state[
                    "softening_flux_njy"
                ][filter_name],
            )
        )

        signal_to_noise = (
            masked_flux / masked_error
        )

        transformed[output_snr_column] = (
            np.sign(signal_to_noise)
            * np.log1p(
                np.abs(signal_to_noise)
            )
        )

    for first_filter, second_filter in COLOR_PAIRS:
        first_name = first_filter.lower()
        second_name = second_filter.lower()

        first_magnitude_column = (
            f"x_asinh_mag_{first_name}"
        )

        second_magnitude_column = (
            f"x_asinh_mag_{second_name}"
        )

        color_column = (
            f"x_color_{first_name}_{second_name}"
        )

        color_is_usable = (
            usable_filter_masks[first_filter]
            & usable_filter_masks[second_filter]
        )

        transformed[color_column] = (
            transformed[first_magnitude_column]
            - transformed[second_magnitude_column]
        ).where(color_is_usable)

    transformed[CORE_FILTER_COUNT_FEATURE] = (
        transformed.loc[
            :,
            list(VALIDITY_FEATURE_COLUMNS),
        ].sum(axis=1)
    )

    transformed[EAZY_HYBRID_FEATURE] = (
        _numeric_series(
            transformed[
                REFERENCE_PREDICTION_COLUMN
            ]
        )
    )

    feature_values = transformed.loc[
        :,
        list(FULL_FEATURE_COLUMNS),
    ]

    if np.isinf(
        feature_values.to_numpy(dtype=float)
    ).any():
        raise ValueError(
            "Infinite values remain in the constructed features."
        )

    return transformed


def build_feature_summary(
    named_catalogs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build a compact summary of the four modeling feature tables."""

    summary_rows = []

    for catalog_name, catalog in named_catalogs.items():
        finite_feature_counts = (
            catalog.loc[
                :,
                list(FULL_FEATURE_COLUMNS),
            ]
            .notna()
            .sum(axis=1)
        )

        summary_rows.append(
            {
                "catalog": catalog_name,
                "rows": len(catalog),
                "physical_sources": (
                    catalog[SOURCE_KEY_COLUMN].nunique()
                ),
                "augmented_rows": int(
                    catalog[IS_AUGMENTED_COLUMN].sum()
                ),
                "full_feature_columns": len(
                    FULL_FEATURE_COLUMNS
                ),
                "median_finite_features": float(
                    finite_feature_counts.median()
                ),
                "finite_eazy_fraction": float(
                    catalog[
                        EAZY_HYBRID_FEATURE
                    ].notna().mean()
                ),
            }
        )

    return pd.DataFrame(summary_rows)


def save_feature_products(
    named_catalogs: dict[str, pd.DataFrame],
    feature_state: dict,
    summary: pd.DataFrame,
) -> None:
    """Save feature tables, their fitted state, and summary."""

    output_paths = {
        "weighted_train": (
            WEIGHTED_TRAINING_FEATURES_PATH
        ),
        "augmented_train": (
            AUGMENTED_TRAINING_FEATURES_PATH
        ),
        "validation": VALIDATION_FEATURES_PATH,
        "test": TEST_FEATURES_PATH,
    }

    for catalog_name, output_path in output_paths.items():
        named_catalogs[catalog_name].to_csv(
            output_path,
            index=False,
        )

    summary.to_csv(
        FEATURE_SUMMARY_PATH,
        index=False,
    )

    FEATURE_STATE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with FEATURE_STATE_PATH.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            feature_state,
            file_handle,
            indent=2,
        )


def prepare_feature_data(
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Run the complete Task 5 feature-construction stage."""

    weighted_training_catalog = (
        load_feature_input_catalog(
            WEIGHTED_TRAINING_CATALOG_PATH,
            "weighted training catalog",
        )
    )

    augmented_training_catalog = (
        load_feature_input_catalog(
            AUGMENTED_TRAINING_CATALOG_PATH,
            "augmented training catalog",
        )
    )

    validation_catalog = load_feature_input_catalog(
        VALIDATION_CATALOG_PATH,
        "validation catalog",
    )

    test_catalog = load_feature_input_catalog(
        TEST_CATALOG_PATH,
        "test catalog",
    )

    feature_state = fit_feature_state(
        weighted_training_catalog
    )

    named_catalogs = {
        "weighted_train": transform_catalog(
            weighted_training_catalog,
            feature_state,
        ),
        "augmented_train": transform_catalog(
            augmented_training_catalog,
            feature_state,
        ),
        "validation": transform_catalog(
            validation_catalog,
            feature_state,
        ),
        "test": transform_catalog(
            test_catalog,
            feature_state,
        ),
    }

    summary = build_feature_summary(
        named_catalogs
    )

    save_feature_products(
        named_catalogs,
        feature_state,
        summary,
    )

    return named_catalogs, summary