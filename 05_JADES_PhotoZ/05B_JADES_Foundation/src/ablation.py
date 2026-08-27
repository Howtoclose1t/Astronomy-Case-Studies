# -*- coding: utf-8 -*-

"""Compare EAZY, pure TabPFN, and two TabPFN residual models."""

import gc
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import (
    AUGMENTED_TRAIN_FEATURES_PATH,
    CATASTROPHIC_OUTLIER_THRESHOLD,
    EAZY_AUXILIARY_COLUMNS,
    EAZY_MODEL_FEATURE,
    EXPERIMENT_MANIFEST_PATH,
    FIGURE_DIR,
    MAXIMUM_PREDICTED_REDSHIFT,
    METRICS_DIR,
    MINIMUM_PREDICTED_REDSHIFT,
    PREDICTION_DIR,
    SOURCE_KEY_COLUMN,
    SPLIT_COLUMN,
    TABPFN_REDSHIFT_COLUMN,
    TARGET_COLUMN,
    VALIDATION_PREDICTIONS_PATH,
    ensure_output_directories,
)
from .data import (
    TabPFNDevelopmentData,
    add_eazy_diagnostic_features,
    calculate_log_residual_target,
    load_feature_columns,
    prepare_feature_matrix,
    prepare_tabpfn_development_data,
)
from .tabpfn import (
    apply_log_residual_correction,
    build_tabpfn_regressor,
    calculate_photoz_metrics,
)


EXPECTED_AUGMENTED_ROWS = 2795
EXPECTED_PHYSICAL_TRAINING_SOURCES = 2219
EXPECTED_SYNTHETIC_ROWS = 576

DIRECT_TABPFN_COLUMN = "z_tabpfn_photometry_only"
RESAMPLED_RESIDUAL_COLUMN = "z_tabpfn_resampled_residual"

ABLATION_METRICS_PATH = (
    METRICS_DIR
    / "tabpfn_validation_ablation_metrics.csv"
)

ABLATION_PREDICTIONS_PATH = (
    PREDICTION_DIR
    / "tabpfn_validation_ablation_predictions.csv"
)

ABLATION_FIGURE_PATH = (
    FIGURE_DIR
    / "tabpfn_validation_ablation.png"
)

ABLATION_MANIFEST_PATH = (
    METRICS_DIR
    / "tabpfn_validation_ablation_manifest.json"
)


def _numeric_series(
    series: pd.Series,
) -> pd.Series:
    """Convert a column to numeric values."""

    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )


def load_existing_residual_predictions(
    data: TabPFNDevelopmentData,
) -> pd.DataFrame:
    """Reuse the already completed physical-source residual experiment."""

    if not VALIDATION_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "The existing TabPFN residual predictions were not found. "
            "Run the previous validation experiment first."
        )

    saved_predictions = pd.read_csv(
        VALIDATION_PREDICTIONS_PATH,
        low_memory=False,
    )

    required_columns = {
        SOURCE_KEY_COLUMN,
        "z_spec",
        "z_eazy",
        TABPFN_REDSHIFT_COLUMN,
    }

    missing_columns = sorted(
        required_columns
        - set(saved_predictions.columns)
    )

    if missing_columns:
        raise KeyError(
            "The saved residual predictions are missing: "
            f"{missing_columns}"
        )

    if not saved_predictions[
        SOURCE_KEY_COLUMN
    ].is_unique:
        raise ValueError(
            "The saved residual predictions contain duplicated sources."
        )

    saved_predictions = (
        saved_predictions
        .set_index(SOURCE_KEY_COLUMN)
    )

    validation_keys = data.validation_catalog[
        SOURCE_KEY_COLUMN
    ].tolist()

    missing_sources = sorted(
        set(validation_keys)
        - set(saved_predictions.index)
    )

    extra_sources = sorted(
        set(saved_predictions.index)
        - set(validation_keys)
    )

    if missing_sources or extra_sources:
        raise ValueError(
            "The saved residual predictions do not match "
            "the current validation sources."
        )

    saved_predictions = saved_predictions.loc[
        validation_keys
    ]

    current_z_spec = data.validation_catalog[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

    current_z_eazy = data.validation_catalog[
        EAZY_MODEL_FEATURE
    ].to_numpy(dtype=float)

    if not np.allclose(
        saved_predictions["z_spec"],
        current_z_spec,
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError(
            "The saved and current z_spec values disagree."
        )

    if not np.allclose(
        saved_predictions["z_eazy"],
        current_z_eazy,
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError(
            "The saved and current EAZY values disagree."
        )

    return pd.DataFrame(
        {
            SOURCE_KEY_COLUMN: validation_keys,
            "z_spec": current_z_spec,
            "z_eazy": current_z_eazy,
            TABPFN_REDSHIFT_COLUMN: (
                saved_predictions[
                    TABPFN_REDSHIFT_COLUMN
                ].to_numpy(dtype=float)
            ),
        }
    )


def load_augmented_training_catalog(
    full_feature_columns: tuple[str, ...],
    validation_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Load the Task 5 training catalog containing synthetic rows."""

    if not AUGMENTED_TRAIN_FEATURES_PATH.exists():
        raise FileNotFoundError(
            "The augmented Task 5 training catalog was not found: "
            f"{AUGMENTED_TRAIN_FEATURES_PATH}"
        )

    catalog = pd.read_csv(
        AUGMENTED_TRAIN_FEATURES_PATH,
        low_memory=False,
    )

    required_columns = {
        SOURCE_KEY_COLUMN,
        "sample_id",
        TARGET_COLUMN,
        SPLIT_COLUMN,
        "is_augmented",
        EAZY_MODEL_FEATURE,
        *EAZY_AUXILIARY_COLUMNS,
        *full_feature_columns,
    }

    missing_columns = sorted(
        required_columns - set(catalog.columns)
    )

    if missing_columns:
        raise KeyError(
            "The augmented training catalog is missing: "
            f"{missing_columns}"
        )

    if len(catalog) != EXPECTED_AUGMENTED_ROWS:
        raise ValueError(
            f"The augmented catalog contains {len(catalog)} rows; "
            f"{EXPECTED_AUGMENTED_ROWS} were expected."
        )

    if not catalog[SPLIT_COLUMN].eq("train").all():
        raise ValueError(
            "The augmented catalog contains rows outside "
            "the training split."
        )

    if not catalog["sample_id"].is_unique:
        raise ValueError(
            "The augmented catalog contains duplicated sample IDs."
        )

    raw_augmented_flag = (
        catalog["is_augmented"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    valid_flag_values = {
        "true",
        "false",
    }

    if not set(raw_augmented_flag.unique()).issubset(
        valid_flag_values
    ):
        raise ValueError(
            "The is_augmented column contains invalid values."
        )

    augmented_mask = raw_augmented_flag.eq("true")

    physical_rows = int(
        (~augmented_mask).sum()
    )

    synthetic_rows = int(
        augmented_mask.sum()
    )

    if physical_rows != EXPECTED_PHYSICAL_TRAINING_SOURCES:
        raise ValueError(
            f"Expected {EXPECTED_PHYSICAL_TRAINING_SOURCES} "
            f"physical rows, found {physical_rows}."
        )

    if synthetic_rows != EXPECTED_SYNTHETIC_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SYNTHETIC_ROWS} synthetic rows, "
            f"found {synthetic_rows}."
        )

    if (
        catalog[SOURCE_KEY_COLUMN].nunique()
        != EXPECTED_PHYSICAL_TRAINING_SOURCES
    ):
        raise ValueError(
            "The augmented catalog does not represent exactly "
            "2219 physical training sources."
        )

    validation_sources = set(
        validation_catalog[SOURCE_KEY_COLUMN]
    )

    training_sources = set(
        catalog[SOURCE_KEY_COLUMN]
    )

    if training_sources & validation_sources:
        raise ValueError(
            "Training-validation leakage was detected "
            "in the augmented catalog."
        )

    catalog[TARGET_COLUMN] = _numeric_series(
        catalog[TARGET_COLUMN]
    )

    catalog[EAZY_MODEL_FEATURE] = _numeric_series(
        catalog[EAZY_MODEL_FEATURE]
    )

    if not np.isfinite(
        catalog[TARGET_COLUMN]
    ).all():
        raise ValueError(
            "The augmented catalog contains invalid z_spec."
        )

    if not np.isfinite(
        catalog[EAZY_MODEL_FEATURE]
    ).all():
        raise ValueError(
            "The augmented catalog contains invalid EAZY redshifts."
        )

    return add_eazy_diagnostic_features(catalog)


def fit_tabpfn_context(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """Fit one TabPFN context and return validation predictions."""

    model = build_tabpfn_regressor()

    start_time = time.perf_counter()

    model.fit(
        x_train.to_numpy(dtype=np.float32),
        np.asarray(y_train, dtype=np.float32),
    )

    predictions = model.predict(
        x_validation.to_numpy(dtype=np.float32)
    )

    runtime_seconds = (
        time.perf_counter() - start_time
    )

    predictions = np.asarray(
        predictions,
        dtype=float,
    ).reshape(-1)

    if not np.isfinite(predictions).all():
        raise ValueError(
            "TabPFN produced non-finite predictions."
        )

    # 一次只保留一个模型，避免8 GB显存被多个模型同时占用
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return predictions, runtime_seconds


def convert_direct_log_prediction(
    predicted_log_redshift,
) -> np.ndarray:
    """Convert predicted log(1 + z) into redshift."""

    predicted_log_values = np.asarray(
        predicted_log_redshift,
        dtype=float,
    )

    predicted_log_values = np.clip(
        predicted_log_values,
        a_min=np.log1p(
            MINIMUM_PREDICTED_REDSHIFT
        ),
        a_max=np.log1p(
            MAXIMUM_PREDICTED_REDSHIFT
        ),
    )

    return np.expm1(predicted_log_values)


def build_ablation_predictions(
    data: TabPFNDevelopmentData,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run the two new TabPFN ablation configurations."""

    (
        full_feature_columns,
        residual_feature_columns,
    ) = load_feature_columns()

    if (
        tuple(data.feature_columns)
        != residual_feature_columns
    ):
        raise ValueError(
            "The development data use unexpected residual features."
        )

    predictions = load_existing_residual_predictions(
        data
    )

    # --------------------------------------------------------------
    # 1. Pure TabPFN：不使用 EAZY，只用57个测光和颜色特征
    # --------------------------------------------------------------

    direct_x_train = prepare_feature_matrix(
        data.training_catalog,
        full_feature_columns,
    )

    direct_x_validation = prepare_feature_matrix(
        data.validation_catalog,
        full_feature_columns,
    )

    # Pure TabPFN 直接学习 log(1 + z_spec)
    direct_y_train = np.log1p(
        data.training_catalog[
            TARGET_COLUMN
        ].to_numpy(dtype=float)
    ).astype(np.float32)

    (
        predicted_direct_log_redshift,
        direct_runtime,
    ) = fit_tabpfn_context(
        x_train=direct_x_train,
        y_train=direct_y_train,
        x_validation=direct_x_validation,
    )

    predictions[DIRECT_TABPFN_COLUMN] = (
        convert_direct_log_prediction(
            predicted_direct_log_redshift
        )
    )

    # --------------------------------------------------------------
    # 2. Resampled residual：使用2795行训练数据和 EAZY prior
    # --------------------------------------------------------------

    augmented_catalog = (
        load_augmented_training_catalog(
            full_feature_columns=full_feature_columns,
            validation_catalog=data.validation_catalog,
        )
    )

    augmented_x_train = prepare_feature_matrix(
        augmented_catalog,
        residual_feature_columns,
    )

    augmented_y_train = (
        calculate_log_residual_target(
            augmented_catalog
        )
    )

    (
        predicted_resampled_residual,
        resampled_runtime,
    ) = fit_tabpfn_context(
        x_train=augmented_x_train,
        y_train=augmented_y_train,
        x_validation=data.x_validation,
    )

    predictions[RESAMPLED_RESIDUAL_COLUMN] = (
        apply_log_residual_correction(
            eazy_redshift=predictions["z_eazy"],
            predicted_log_residual=(
                predicted_resampled_residual
            ),
        )
    )

    runtimes = {
        "EAZY DR5 z_a": np.nan,
        "TabPFN v3: photometry only": direct_runtime,
        "TabPFN v3: EAZY residual": (
            load_existing_residual_runtime()
        ),
        "TabPFN v3: resampled EAZY residual": (
            resampled_runtime
        ),
    }

    return predictions, runtimes


def load_existing_residual_runtime() -> float:
    """Read the runtime of the already completed residual experiment."""

    if not EXPERIMENT_MANIFEST_PATH.exists():
        return np.nan

    with EXPERIMENT_MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    return float(
        manifest.get(
            "runtime_seconds",
            np.nan,
        )
    )


def build_ablation_metrics(
    predictions: pd.DataFrame,
    runtimes: dict[str, float],
) -> pd.DataFrame:
    """Calculate four-method validation metrics."""

    method_definitions = (
        (
            "EAZY DR5 z_a",
            "z_eazy",
            0,
            0,
            False,
        ),
        (
            "TabPFN v3: photometry only",
            DIRECT_TABPFN_COLUMN,
            2219,
            57,
            False,
        ),
        (
            "TabPFN v3: EAZY residual",
            TABPFN_REDSHIFT_COLUMN,
            2219,
            66,
            True,
        ),
        (
            "TabPFN v3: resampled EAZY residual",
            RESAMPLED_RESIDUAL_COLUMN,
            2795,
            66,
            True,
        ),
    )

    metric_rows = []

    for (
        method,
        prediction_column,
        training_rows,
        feature_count,
        uses_eazy,
    ) in method_definitions:
        metric_row = calculate_photoz_metrics(
            true_redshift=predictions["z_spec"],
            predicted_redshift=predictions[
                prediction_column
            ],
            method=method,
        )

        metric_row["training_rows"] = training_rows
        metric_row["feature_count"] = feature_count
        metric_row["uses_eazy"] = uses_eazy
        metric_row["runtime_seconds"] = runtimes[method]

        metric_rows.append(metric_row)

    metrics = pd.DataFrame(metric_rows)

    metrics["validation_rank"] = (
        metrics[
            "normalized_median_absolute_error"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    return metrics


def _draw_redshift_panel(
    axis,
    z_spec,
    predicted,
    title: str,
    color: str,
) -> None:
    """Draw one predicted-versus-spectroscopic-redshift panel."""

    normalized_error = (
        predicted - z_spec
    ) / (1.0 + z_spec)

    outlier = (
        np.abs(normalized_error)
        > CATASTROPHIC_OUTLIER_THRESHOLD
    )

    plot_limit = max(
        1.0,
        float(
            np.ceil(
                np.nanmax(
                    np.concatenate(
                        [z_spec, predicted]
                    )
                )
                + 0.5
            )
        ),
    )

    axis.scatter(
        z_spec[~outlier],
        predicted[~outlier],
        s=15,
        alpha=0.6,
        color=color,
    )

    axis.scatter(
        z_spec[outlier],
        predicted[outlier],
        s=38,
        marker="x",
        linewidth=1.5,
        color="red",
    )

    axis.plot(
        [0.0, plot_limit],
        [0.0, plot_limit],
        linestyle="--",
        color="black",
        linewidth=1.0,
    )

    axis.set_xlim(0.0, plot_limit)
    axis.set_ylim(0.0, plot_limit)
    axis.set_xlabel("Spectroscopic redshift")
    axis.set_ylabel("Predicted redshift")
    axis.set_title(title)
    axis.grid(alpha=0.2)


def build_ablation_figure(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    output_path: Path = ABLATION_FIGURE_PATH,
) -> Path:
    """Create one scientifically focused four-method comparison."""

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11),
    )

    z_spec = predictions["z_spec"].to_numpy(
        dtype=float
    )

    method_specs = (
        (
            axes[0, 0],
            "z_eazy",
            "EAZY DR5",
            "#1f77b4",
        ),
        (
            axes[0, 1],
            DIRECT_TABPFN_COLUMN,
            "Pure TabPFN: photometry only",
            "#ff7f0e",
        ),
        (
            axes[0, 2],
            TABPFN_REDSHIFT_COLUMN,
            "TabPFN–EAZY residual",
            "#2ca02c",
        ),
        (
            axes[1, 0],
            RESAMPLED_RESIDUAL_COLUMN,
            "Resampled residual",
            "#9467bd",
        ),
    )

    for (
        axis,
        prediction_column,
        title,
        color,
    ) in method_specs:
        _draw_redshift_panel(
            axis=axis,
            z_spec=z_spec,
            predicted=predictions[
                prediction_column
            ].to_numpy(dtype=float),
            title=title,
            color=color,
        )

    comparison_metrics = (
        "normalized_median_absolute_error",
        "sigma_nmad",
        "mean_absolute_redshift_error",
        "catastrophic_outlier_fraction",
    )

    metric_labels = (
        "Median normalized\nabsolute error",
        "Sigma NMAD",
        "MAE",
        "Outlier fraction",
    )

    method_colors = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#9467bd",
    )

    eazy_row = metrics.iloc[0]
    x_positions = np.arange(
        len(comparison_metrics)
    )

    bar_width = 0.19

    for method_index, (_, row) in enumerate(
        metrics.iterrows()
    ):
        ratios = [
            row[metric]
            / eazy_row[metric]
            for metric in comparison_metrics
        ]

        axes[1, 1].bar(
            x_positions
            + (
                method_index - 1.5
            )
            * bar_width,
            ratios,
            width=bar_width,
            color=method_colors[method_index],
            label=row["method"],
        )

    axes[1, 1].axhline(
        1.0,
        linestyle="--",
        color="black",
        linewidth=1.0,
    )

    axes[1, 1].set_xticks(x_positions)
    axes[1, 1].set_xticklabels(
        metric_labels,
        fontsize=8,
    )

    axes[1, 1].set_ylabel(
        "Metric relative to EAZY\n(lower is better)"
    )

    axes[1, 1].set_title(
        "Relative validation performance"
    )

    axes[1, 1].grid(
        axis="y",
        alpha=0.2,
    )

    axes[1, 1].legend(fontsize=7)

    outlier_counts = metrics[
        "catastrophic_outliers"
    ].to_numpy()

    short_method_names = (
        "EAZY",
        "Pure\nTabPFN",
        "Residual",
        "Resampled\nresidual",
    )

    bars = axes[1, 2].bar(
        short_method_names,
        outlier_counts,
        color=method_colors,
    )

    for bar, count in zip(
        bars,
        outlier_counts,
    ):
        axes[1, 2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(int(count)),
            ha="center",
            va="bottom",
        )

    axes[1, 2].set_ylabel(
        "Catastrophic outliers"
    )

    axes[1, 2].set_title(
        "Validation outlier comparison"
    )

    axes[1, 2].grid(
        axis="y",
        alpha=0.2,
    )

    figure.suptitle(
        "JADES TabPFN validation ablation",
        fontsize=16,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def save_ablation_outputs(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Save the complete four-method comparison."""

    metrics.to_csv(
        ABLATION_METRICS_PATH,
        index=False,
    )

    predictions.to_csv(
        ABLATION_PREDICTIONS_PATH,
        index=False,
    )

    manifest = {
        "comparison": [
            "EAZY DR5 z_a",
            "TabPFN v3: photometry only",
            "TabPFN v3: EAZY residual",
            "TabPFN v3: resampled EAZY residual",
        ],
        "validation_sources": len(predictions),
        "test_set_opened": False,
        "selection_metric": (
            "normalized_median_absolute_error"
        ),
    }

    with ABLATION_MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )


def run_tabpfn_ablation(
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Run and save the complete validation ablation."""

    ensure_output_directories()

    data = prepare_tabpfn_development_data()

    predictions, runtimes = (
        build_ablation_predictions(data)
    )

    metrics = build_ablation_metrics(
        predictions=predictions,
        runtimes=runtimes,
    )

    figure_path = build_ablation_figure(
        predictions=predictions,
        metrics=metrics,
    )

    save_ablation_outputs(
        metrics=metrics,
        predictions=predictions,
    )

    return (
        metrics,
        predictions,
        figure_path,
    )