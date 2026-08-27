"""Prepare deterministic train, validation, and test data for Task 5."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CORE_PHOTOMETRIC_COLUMNS,
    FIELD_COLUMN,
    HYBRID_CANDIDATE_COLUMN,
    MINIMUM_VALID_FILTERS,
    MODELING_CATALOG_PATH,
    MODEL_CANDIDATE_COLUMN,
    PHOTOMETRY_ID_COLUMN,
    QUALITY_COLUMN,
    RANDOM_SEED,
    REDSHIFT_STRATIFICATION_EDGES,
    REFERENCE_PREDICTION_COLUMN,
    SECURE_REDSHIFT_QUALITIES,
    SOURCE_KEY_COLUMN,
    SPLIT_ASSIGNMENTS_PATH,
    TARGET_COLUMN,
    TEST_CATALOG_PATH,
    TEST_FRACTION,
    TRAIN_FRACTION,
    TRAINING_CATALOG_PATH,
    UPSTREAM_MATCHED_CATALOG_PATH,
    VALIDATION_CATALOG_PATH,
    VALIDATION_FRACTION,
    ensure_output_directories,
)


N_VALID_FILTERS_COLUMN = "n_valid_filters"
REDSHIFT_STRATUM_COLUMN = "redshift_stratum"
SPLIT_COLUMN = "split"

SPLIT_ORDER = (
    "train",
    "validation",
    "test",
)

REQUIRED_COLUMNS = (
    FIELD_COLUMN,
    PHOTOMETRY_ID_COLUMN,
    TARGET_COLUMN,
    QUALITY_COLUMN,
    MODEL_CANDIDATE_COLUMN,
    HYBRID_CANDIDATE_COLUMN,
    REFERENCE_PREDICTION_COLUMN,
    N_VALID_FILTERS_COLUMN,
    *CORE_PHOTOMETRIC_COLUMNS,
)


def _validate_required_columns(catalog: pd.DataFrame) -> None:
    """Raise an informative error if required upstream columns are missing."""

    missing_columns = sorted(
        set(REQUIRED_COLUMNS) - set(catalog.columns)
    )

    if missing_columns:
        raise KeyError(
            "The upstream Task 4 catalog is missing required columns: "
            f"{missing_columns}"
        )


def _coerce_boolean_series(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """Convert a boolean-like column without silently accepting bad values."""

    if pd.api.types.is_bool_dtype(series.dtype):
        if series.isna().any():
            raise ValueError(
                f"{column_name} contains missing boolean values."
            )

        return series.astype(bool)

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
            f"{column_name} contains unsupported boolean values: "
            f"{unexpected_values[:10]}"
        )

    return converted.astype(bool)


def _build_source_key(catalog: pd.DataFrame) -> pd.Series:
    """Build a field-aware source identifier from field and photometric ID."""

    field_values = (
        catalog[FIELD_COLUMN]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if field_values.isna().any() or field_values.eq("").any():
        raise ValueError(
            f"{FIELD_COLUMN} contains missing or empty values."
        )

    photometric_ids = pd.to_numeric(
        catalog[PHOTOMETRY_ID_COLUMN],
        errors="coerce",
    )

    invalid_ids = (
        photometric_ids.isna()
        | (photometric_ids <= 0)
        | ~np.isclose(photometric_ids % 1, 0)
    )

    if invalid_ids.any():
        raise ValueError(
            f"{PHOTOMETRY_ID_COLUMN} contains invalid source IDs."
        )

    catalog[FIELD_COLUMN] = field_values
    catalog[PHOTOMETRY_ID_COLUMN] = photometric_ids.astype("int64")

    return (
        field_values
        + ":"
        + catalog[PHOTOMETRY_ID_COLUMN].astype(str)
    )


def load_upstream_catalog() -> pd.DataFrame:
    """Load and validate the final matched catalog produced by Task 4."""

    if not UPSTREAM_MATCHED_CATALOG_PATH.exists():
        raise FileNotFoundError(
            "The Task 4 matched catalog was not found at "
            f"{UPSTREAM_MATCHED_CATALOG_PATH}"
        )

    catalog = pd.read_csv(
        UPSTREAM_MATCHED_CATALOG_PATH,
        low_memory=False,
    )

    _validate_required_columns(catalog)

    catalog[TARGET_COLUMN] = pd.to_numeric(
        catalog[TARGET_COLUMN],
        errors="coerce",
    )

    catalog[REFERENCE_PREDICTION_COLUMN] = pd.to_numeric(
        catalog[REFERENCE_PREDICTION_COLUMN],
        errors="coerce",
    )

    catalog[N_VALID_FILTERS_COLUMN] = pd.to_numeric(
        catalog[N_VALID_FILTERS_COLUMN],
        errors="coerce",
    )

    catalog[QUALITY_COLUMN] = (
        catalog[QUALITY_COLUMN]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    catalog[MODEL_CANDIDATE_COLUMN] = _coerce_boolean_series(
        catalog[MODEL_CANDIDATE_COLUMN],
        MODEL_CANDIDATE_COLUMN,
    )

    catalog[HYBRID_CANDIDATE_COLUMN] = _coerce_boolean_series(
        catalog[HYBRID_CANDIDATE_COLUMN],
        HYBRID_CANDIDATE_COLUMN,
    )

    catalog[SOURCE_KEY_COLUMN] = _build_source_key(catalog)

    duplicated_keys = catalog[SOURCE_KEY_COLUMN].duplicated(
        keep=False
    )

    if duplicated_keys.any():
        duplicated_examples = (
            catalog.loc[duplicated_keys, SOURCE_KEY_COLUMN]
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            "The Task 4 catalog is not one-to-one. "
            f"Duplicated source keys include {duplicated_examples}"
        )

    eligible_model_source = (
        np.isfinite(catalog[TARGET_COLUMN])
        & catalog[QUALITY_COLUMN].isin(
            SECURE_REDSHIFT_QUALITIES
        )
        & (
            catalog[N_VALID_FILTERS_COLUMN]
            >= MINIMUM_VALID_FILTERS
        )
    )

    invalid_model_flags = (
        catalog[MODEL_CANDIDATE_COLUMN]
        & ~eligible_model_source
    )

    if invalid_model_flags.any():
        raise ValueError(
            "Some rows marked as model candidates do not satisfy "
            "the finite-z, quality, or filter-coverage requirements."
        )

    invalid_hybrid_flags = (
        catalog[HYBRID_CANDIDATE_COLUMN]
        & (
            ~catalog[MODEL_CANDIDATE_COLUMN]
            | ~np.isfinite(
                catalog[REFERENCE_PREDICTION_COLUMN]
            )
        )
    )

    if invalid_hybrid_flags.any():
        raise ValueError(
            "Some hybrid candidates are missing a valid model flag "
            "or a finite EAZY reference prediction."
        )

    return catalog


def select_modeling_sample(
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Select Task 4 model candidates and construct redshift strata."""

    modeling_catalog = catalog.loc[
        catalog[MODEL_CANDIDATE_COLUMN]
    ].copy()

    if modeling_catalog.empty:
        raise ValueError(
            "No model candidates were found in the Task 4 catalog."
        )

    modeling_catalog = modeling_catalog.sort_values(
        SOURCE_KEY_COLUMN
    ).reset_index(drop=True)

    redshift_strata = pd.cut(
        modeling_catalog[TARGET_COLUMN],
        bins=REDSHIFT_STRATIFICATION_EDGES,
        labels=False,
        include_lowest=True,
        right=False,
    )

    if redshift_strata.isna().any():
        raise ValueError(
            "At least one model candidate falls outside the "
            "configured redshift stratification edges."
        )

    modeling_catalog[REDSHIFT_STRATUM_COLUMN] = (
        redshift_strata.astype("int64")
    )

    stratum_counts = modeling_catalog[
        REDSHIFT_STRATUM_COLUMN
    ].value_counts()

    if (stratum_counts < 3).any():
        sparse_strata = (
            stratum_counts.loc[stratum_counts < 3]
            .sort_index()
            .to_dict()
        )

        raise ValueError(
            "Some redshift strata are too small for a three-way "
            f"split: {sparse_strata}"
        )

    return modeling_catalog


def assign_fixed_splits(
    modeling_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Assign reproducible stratified train, validation, and test splits."""

    split_fractions = np.array(
        [
            TRAIN_FRACTION,
            VALIDATION_FRACTION,
            TEST_FRACTION,
        ],
        dtype=float,
    )

    if not np.isclose(split_fractions.sum(), 1.0):
        raise ValueError(
            "Train, validation, and test fractions must sum to 1."
        )

    holdout_fraction = (
        VALIDATION_FRACTION + TEST_FRACTION
    )

    test_fraction_within_holdout = (
        TEST_FRACTION / holdout_fraction
    )

    all_indices = modeling_catalog.index.to_numpy()
    all_strata = modeling_catalog[
        REDSHIFT_STRATUM_COLUMN
    ].to_numpy()

    train_indices, holdout_indices = train_test_split(
        all_indices,
        test_size=holdout_fraction,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=all_strata,
    )

    validation_indices, test_indices = train_test_split(
        holdout_indices,
        test_size=test_fraction_within_holdout,
        random_state=RANDOM_SEED + 1,
        shuffle=True,
        stratify=modeling_catalog.loc[
            holdout_indices,
            REDSHIFT_STRATUM_COLUMN,
        ],
    )

    result = modeling_catalog.copy()
    result[SPLIT_COLUMN] = ""

    result.loc[train_indices, SPLIT_COLUMN] = "train"
    result.loc[validation_indices, SPLIT_COLUMN] = "validation"
    result.loc[test_indices, SPLIT_COLUMN] = "test"

    if result[SPLIT_COLUMN].eq("").any():
        raise RuntimeError(
            "At least one source was not assigned to a data split."
        )

    if not result[SOURCE_KEY_COLUMN].is_unique:
        raise RuntimeError(
            "Source keys are no longer unique after splitting."
        )

    split_stratum_counts = pd.crosstab(
        result[SPLIT_COLUMN],
        result[REDSHIFT_STRATUM_COLUMN],
    )

    missing_strata = split_stratum_counts.eq(0)

    if missing_strata.any().any():
        raise ValueError(
            "At least one data split does not contain every redshift "
            "stratum. Use coarser redshift strata before modeling."
        )

    return result


def build_split_summary(
    modeling_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Create the compact split summary displayed in the notebook."""

    summary_rows = []

    for split_name in SPLIT_ORDER:
        subset = modeling_catalog.loc[
            modeling_catalog[SPLIT_COLUMN] == split_name
        ]

        summary_rows.append(
            {
                "split": split_name,
                "sources": len(subset),
                "fraction": len(subset) / len(modeling_catalog),
                "median_z_spec": subset[TARGET_COLUMN].median(),
                "minimum_z_spec": subset[TARGET_COLUMN].min(),
                "maximum_z_spec": subset[TARGET_COLUMN].max(),
                "hybrid_sources": int(
                    subset[HYBRID_CANDIDATE_COLUMN].sum()
                ),
            }
        )

    return pd.DataFrame(summary_rows)


def save_modeling_data(
    modeling_catalog: pd.DataFrame,
) -> None:
    """Save the complete catalog, split assignments, and split tables."""

    ensure_output_directories()

    modeling_catalog.to_csv(
        MODELING_CATALOG_PATH,
        index=False,
    )

    assignment_columns = [
        SOURCE_KEY_COLUMN,
        FIELD_COLUMN,
        PHOTOMETRY_ID_COLUMN,
        TARGET_COLUMN,
        QUALITY_COLUMN,
        REFERENCE_PREDICTION_COLUMN,
        REDSHIFT_STRATUM_COLUMN,
        SPLIT_COLUMN,
    ]

    modeling_catalog.loc[
        :,
        assignment_columns,
    ].to_csv(
        SPLIT_ASSIGNMENTS_PATH,
        index=False,
    )

    output_paths = {
        "train": TRAINING_CATALOG_PATH,
        "validation": VALIDATION_CATALOG_PATH,
        "test": TEST_CATALOG_PATH,
    }

    for split_name, output_path in output_paths.items():
        split_catalog = modeling_catalog.loc[
            modeling_catalog[SPLIT_COLUMN] == split_name
        ]

        split_catalog.to_csv(
            output_path,
            index=False,
        )


def prepare_modeling_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete deterministic Task 5 data-entry stage."""

    upstream_catalog = load_upstream_catalog()

    modeling_catalog = select_modeling_sample(
        upstream_catalog
    )

    modeling_catalog = assign_fixed_splits(
        modeling_catalog
    )

    save_modeling_data(modeling_catalog)

    split_summary = build_split_summary(
        modeling_catalog
    )

    return modeling_catalog, split_summary