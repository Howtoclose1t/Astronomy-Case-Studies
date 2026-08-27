"""Build training-only weighting and photometric augmentation products."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    ALL_FILTERS,
    RANDOM_SEED,
    SOURCE_KEY_COLUMN,
    TARGET_COLUMN,
    TRAINING_CATALOG_PATH,
)


REDSHIFT_STRATUM_COLUMN = "redshift_stratum"
SPLIT_COLUMN = "split"
SAMPLE_ID_COLUMN = "sample_id"
SOURCE_WEIGHT_COLUMN = "source_weight"
SAMPLE_WEIGHT_COLUMN = "sample_weight"
IS_AUGMENTED_COLUMN = "is_augmented"
AUGMENTATION_ITERATION_COLUMN = "augmentation_iteration"
HYBRID_TRAINING_ALLOWED_COLUMN = "hybrid_training_allowed"

MAX_SOURCE_WEIGHT = 4.0
RARE_STRATUM_THRESHOLD = 0.50
AUGMENTED_COPIES_PER_RARE_SOURCE = 2
PHOTOMETRIC_NOISE_SCALE = 1.0

WEIGHTED_TRAINING_CATALOG_PATH = (
    TRAINING_CATALOG_PATH.parent
    / "jades_dr45_weighted_training_catalog.csv"
)

AUGMENTED_TRAINING_CATALOG_PATH = (
    TRAINING_CATALOG_PATH.parent
    / "jades_dr45_augmented_training_catalog.csv"
)

RESAMPLING_SUMMARY_PATH = (
    TRAINING_CATALOG_PATH.parent
    / "jades_dr45_resampling_summary.csv"
)


def load_training_catalog() -> pd.DataFrame:
    """Load and validate the original, unmodified training split."""

    if not TRAINING_CATALOG_PATH.exists():
        raise FileNotFoundError(
            "The fixed Task 5 training catalog was not found at "
            f"{TRAINING_CATALOG_PATH}"
        )

    training_catalog = pd.read_csv(
        TRAINING_CATALOG_PATH,
        low_memory=False,
    )

    required_columns = {
        SOURCE_KEY_COLUMN,
        TARGET_COLUMN,
        REDSHIFT_STRATUM_COLUMN,
        SPLIT_COLUMN,
    }

    for filter_name in ALL_FILTERS:
        normalized_filter = filter_name.lower()

        required_columns.update(
            {
                f"flux_{normalized_filter}_njy",
                f"fluxerr_{normalized_filter}_njy",
                f"valid_{normalized_filter}",
            }
        )

    missing_columns = sorted(
        required_columns - set(training_catalog.columns)
    )

    if missing_columns:
        raise KeyError(
            "The training catalog is missing required columns: "
            f"{missing_columns}"
        )

    if not training_catalog[SOURCE_KEY_COLUMN].is_unique:
        raise ValueError(
            "The original training catalog contains duplicated "
            "source keys."
        )

    if not training_catalog[SPLIT_COLUMN].eq("train").all():
        unexpected_splits = (
            training_catalog[SPLIT_COLUMN]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Resampling may only use the fixed training split. "
            f"Found split values: {unexpected_splits}"
        )

    training_catalog[TARGET_COLUMN] = pd.to_numeric(
        training_catalog[TARGET_COLUMN],
        errors="coerce",
    )

    training_catalog[REDSHIFT_STRATUM_COLUMN] = pd.to_numeric(
        training_catalog[REDSHIFT_STRATUM_COLUMN],
        errors="coerce",
    )

    if training_catalog[TARGET_COLUMN].isna().any():
        raise ValueError(
            "The training catalog contains missing z_spec values."
        )

    if training_catalog[REDSHIFT_STRATUM_COLUMN].isna().any():
        raise ValueError(
            "The training catalog contains missing redshift strata."
        )

    training_catalog[REDSHIFT_STRATUM_COLUMN] = (
        training_catalog[REDSHIFT_STRATUM_COLUMN]
        .astype("int64")
    )

    return training_catalog


def compute_source_weights(
    training_catalog: pd.DataFrame,
    maximum_weight: float = MAX_SOURCE_WEIGHT,
) -> pd.Series:
    """Calculate clipped inverse-frequency weights by redshift stratum."""

    if maximum_weight <= 0:
        raise ValueError(
            "maximum_weight must be positive."
        )

    stratum_counts = training_catalog[
        REDSHIFT_STRATUM_COLUMN
    ].value_counts()

    number_of_sources = len(training_catalog)
    number_of_strata = len(stratum_counts)

    raw_weight_by_stratum = (
        number_of_sources
        / (
            number_of_strata
            * stratum_counts
        )
    )

    clipped_weight_by_stratum = (
        raw_weight_by_stratum
        .clip(upper=maximum_weight)
    )

    source_weights = (
        training_catalog[REDSHIFT_STRATUM_COLUMN]
        .map(clipped_weight_by_stratum)
        .astype(float)
    )

    if not np.isfinite(source_weights).all():
        raise ValueError(
            "Non-finite source weights were generated."
        )

    if (source_weights <= 0).any():
        raise ValueError(
            "All source weights must be positive."
        )

    return source_weights


def build_weighted_training_catalog(
    training_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Attach source weights without creating synthetic observations."""

    weighted_catalog = training_catalog.copy()

    weighted_catalog[SOURCE_WEIGHT_COLUMN] = (
        compute_source_weights(weighted_catalog)
    )

    weighted_catalog[SAMPLE_WEIGHT_COLUMN] = (
        weighted_catalog[SOURCE_WEIGHT_COLUMN]
    )

    weighted_catalog[IS_AUGMENTED_COLUMN] = False
    weighted_catalog[AUGMENTATION_ITERATION_COLUMN] = 0
    weighted_catalog[HYBRID_TRAINING_ALLOWED_COLUMN] = True

    weighted_catalog[SAMPLE_ID_COLUMN] = (
        weighted_catalog[SOURCE_KEY_COLUMN]
        .astype(str)
        + "::original"
    )

    return weighted_catalog


def _coerce_validity_column(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """Convert a filter-validity column to strict boolean values."""

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


def perturb_photometry(
    catalog: pd.DataFrame,
    random_generator: np.random.Generator,
    noise_scale: float = PHOTOMETRIC_NOISE_SCALE,
) -> pd.DataFrame:
    """Perturb valid fluxes using their published measurement errors."""

    if noise_scale < 0:
        raise ValueError(
            "noise_scale cannot be negative."
        )

    perturbed_catalog = catalog.copy()

    for filter_name in ALL_FILTERS:
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

        flux_values = pd.to_numeric(
            perturbed_catalog[flux_column],
            errors="coerce",
        )

        error_values = pd.to_numeric(
            perturbed_catalog[error_column],
            errors="coerce",
        )

        validity_values = _coerce_validity_column(
            perturbed_catalog[validity_column],
            validity_column,
        )

        perturbable = (
            validity_values
            & np.isfinite(flux_values)
            & np.isfinite(error_values)
            & (error_values > 0)
        )

        random_noise = random_generator.normal(
            loc=0.0,
            scale=1.0,
            size=len(perturbed_catalog),
        )

        perturbed_flux = (
            flux_values
            + noise_scale
            * error_values
            * random_noise
        )

        perturbed_catalog.loc[
            perturbable,
            flux_column,
        ] = perturbed_flux.loc[perturbable]

    return perturbed_catalog


def build_augmented_training_catalog(
    weighted_catalog: pd.DataFrame,
    rare_threshold: float = RARE_STRATUM_THRESHOLD,
    copies_per_source: int = AUGMENTED_COPIES_PER_RARE_SOURCE,
    noise_scale: float = PHOTOMETRIC_NOISE_SCALE,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Augment sparse redshift strata using flux-error perturbations."""

    if not 0 < rare_threshold <= 1:
        raise ValueError(
            "rare_threshold must be in the interval (0, 1]."
        )

    if copies_per_source < 0:
        raise ValueError(
            "copies_per_source cannot be negative."
        )

    stratum_counts = weighted_catalog[
        REDSHIFT_STRATUM_COLUMN
    ].value_counts()

    median_stratum_count = float(
        stratum_counts.median()
    )

    rare_strata = stratum_counts.loc[
        stratum_counts
        < rare_threshold * median_stratum_count
    ].index

    augmented_base = weighted_catalog.copy()

    is_rare_source = augmented_base[
        REDSHIFT_STRATUM_COLUMN
    ].isin(rare_strata)

    number_of_variants = pd.Series(
        1,
        index=augmented_base.index,
        dtype="int64",
    )

    number_of_variants.loc[is_rare_source] = (
        copies_per_source + 1
    )

    augmented_base[SAMPLE_WEIGHT_COLUMN] = (
        augmented_base[SOURCE_WEIGHT_COLUMN]
        / number_of_variants
    )

    catalog_parts = [augmented_base]

    random_generator = np.random.default_rng(
        random_seed
    )

    rare_source_catalog = weighted_catalog.loc[
        is_rare_source
    ].copy()

    for iteration in range(
        1,
        copies_per_source + 1,
    ):
        synthetic_catalog = perturb_photometry(
            rare_source_catalog,
            random_generator=random_generator,
            noise_scale=noise_scale,
        )

        synthetic_catalog[IS_AUGMENTED_COLUMN] = True
        synthetic_catalog[AUGMENTATION_ITERATION_COLUMN] = iteration

        synthetic_catalog[
            HYBRID_TRAINING_ALLOWED_COLUMN
        ] = False

        synthetic_catalog[SAMPLE_WEIGHT_COLUMN] = (
            synthetic_catalog[SOURCE_WEIGHT_COLUMN]
            / (copies_per_source + 1)
        )

        synthetic_catalog[SAMPLE_ID_COLUMN] = (
            synthetic_catalog[SOURCE_KEY_COLUMN]
            .astype(str)
            + f"::augmented_{iteration}"
        )

        catalog_parts.append(synthetic_catalog)

    augmented_catalog = pd.concat(
        catalog_parts,
        ignore_index=True,
    )

    if not augmented_catalog[SAMPLE_ID_COLUMN].is_unique:
        raise RuntimeError(
            "Augmented sample IDs are not unique."
        )

    total_weight_by_source = (
        augmented_catalog.groupby(
            SOURCE_KEY_COLUMN,
            sort=False,
        )[SAMPLE_WEIGHT_COLUMN]
        .sum()
    )

    expected_weight_by_source = (
        weighted_catalog.set_index(
            SOURCE_KEY_COLUMN
        )[SOURCE_WEIGHT_COLUMN]
    )

    expected_weight_by_source = (
        expected_weight_by_source.loc[
            total_weight_by_source.index
        ]
    )

    if not np.allclose(
        total_weight_by_source.to_numpy(),
        expected_weight_by_source.to_numpy(),
    ):
        raise RuntimeError(
            "Photometric augmentation changed the total weight "
            "assigned to at least one physical source."
        )

    return augmented_catalog


def build_resampling_summary(
    weighted_catalog: pd.DataFrame,
    augmented_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the training distribution before and after augmentation."""

    original_counts = (
        weighted_catalog.groupby(
            REDSHIFT_STRATUM_COLUMN
        )
        .size()
        .rename("original_sources")
    )

    augmented_counts = (
        augmented_catalog.groupby(
            REDSHIFT_STRATUM_COLUMN
        )
        .size()
        .rename("total_training_rows")
    )

    source_weights = (
        weighted_catalog.groupby(
            REDSHIFT_STRATUM_COLUMN
        )[SOURCE_WEIGHT_COLUMN]
        .first()
        .rename("source_weight")
    )

    redshift_ranges = (
        weighted_catalog.groupby(
            REDSHIFT_STRATUM_COLUMN
        )[TARGET_COLUMN]
        .agg(
            minimum_z_spec="min",
            maximum_z_spec="max",
        )
    )

    summary = pd.concat(
        [
            original_counts,
            augmented_counts,
            source_weights,
            redshift_ranges,
        ],
        axis=1,
    ).reset_index()

    summary["synthetic_rows"] = (
        summary["total_training_rows"]
        - summary["original_sources"]
    )

    summary = summary[
        [
            REDSHIFT_STRATUM_COLUMN,
            "minimum_z_spec",
            "maximum_z_spec",
            "original_sources",
            "source_weight",
            "synthetic_rows",
            "total_training_rows",
        ]
    ]

    return summary


def save_resampling_products(
    weighted_catalog: pd.DataFrame,
    augmented_catalog: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Save reproducible local training-only resampling products."""

    WEIGHTED_TRAINING_CATALOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weighted_catalog.to_csv(
        WEIGHTED_TRAINING_CATALOG_PATH,
        index=False,
    )

    augmented_catalog.to_csv(
        AUGMENTED_TRAINING_CATALOG_PATH,
        index=False,
    )

    summary.to_csv(
        RESAMPLING_SUMMARY_PATH,
        index=False,
    )


def prepare_resampling_data(
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the complete training-only resampling stage."""

    training_catalog = load_training_catalog()

    weighted_catalog = build_weighted_training_catalog(
        training_catalog
    )

    augmented_catalog = build_augmented_training_catalog(
        weighted_catalog
    )

    summary = build_resampling_summary(
        weighted_catalog,
        augmented_catalog,
    )

    save_resampling_products(
        weighted_catalog,
        augmented_catalog,
        summary,
    )

    return (
        weighted_catalog,
        augmented_catalog,
        summary,
    )