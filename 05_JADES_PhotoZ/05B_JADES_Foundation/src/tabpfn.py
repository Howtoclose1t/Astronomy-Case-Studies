# -*- coding: utf-8 -*-

"""Run the TabPFN–EAZY log-residual validation experiment."""

import json
import time
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from .config import (
    CATASTROPHIC_OUTLIER_THRESHOLD,
    EAZY_MODEL_FEATURE,
    FIGURE_DIR,
    MAXIMUM_PREDICTED_REDSHIFT,
    MINIMUM_PREDICTED_REDSHIFT,
    RANDOM_SEED,
    SOURCE_KEY_COLUMN,
    TABPFN_DEVICE,
    TABPFN_FIT_MODE,
    TABPFN_MODEL_VERSION,
    TABPFN_N_ESTIMATORS,
    TABPFN_REDSHIFT_COLUMN,
    TARGET_COLUMN,
    VALIDATION_METRICS_PATH,
    VALIDATION_PREDICTIONS_PATH,
    EXPERIMENT_MANIFEST_PATH,
    ensure_output_directories,
)
from .data import (
    TabPFNDevelopmentData,
    prepare_tabpfn_development_data,
)


VALIDATION_FIGURE_PATH = (
    FIGURE_DIR
    / "tabpfn_validation_comparison.png"
)


def calculate_photoz_metrics(
    true_redshift,
    predicted_redshift,
    method: str,
) -> dict:
    """Calculate the same Photo-z metrics used in Task 5."""

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
            f"No finite prediction pairs are available for {method}."
        )

    true_values = true_values[finite]
    predicted_values = predicted_values[finite]

    # Photo-z 标准化误差：
    # delta_z = (z_pred - z_spec) / (1 + z_spec)
    normalized_error = (
        predicted_values - true_values
    ) / (1.0 + true_values)

    normalized_bias = float(
        np.median(normalized_error)
    )

    normalized_median_absolute_error = float(
        np.median(np.abs(normalized_error))
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
        "method": method,
        "sources": int(len(true_values)),
        "normalized_bias": normalized_bias,
        "normalized_median_absolute_error": (
            normalized_median_absolute_error
        ),
        "sigma_nmad": sigma_nmad,
        "mean_absolute_redshift_error": float(
            absolute_redshift_error.mean()
        ),
        "catastrophic_outliers": int(
            catastrophic.sum()
        ),
        "catastrophic_outlier_fraction": float(
            catastrophic.mean()
        ),
    }


def apply_log_residual_correction(
    eazy_redshift,
    predicted_log_residual,
) -> np.ndarray:
    """Convert predicted log residuals into corrected redshifts."""

    eazy_values = np.asarray(
        eazy_redshift,
        dtype=float,
    )

    residual_values = np.asarray(
        predicted_log_residual,
        dtype=float,
    )

    # 把预测的 residual 加回 EAZY 的 log(1 + z)
    corrected_log_redshift = (
        np.log1p(eazy_values)
        + residual_values
    )

    # 防止模型产生不合理的负红移或极端红移
    corrected_log_redshift = np.clip(
        corrected_log_redshift,
        a_min=np.log1p(
            MINIMUM_PREDICTED_REDSHIFT
        ),
        a_max=np.log1p(
            MAXIMUM_PREDICTED_REDSHIFT
        ),
    )

    # expm1(x) = exp(x) - 1
    return np.expm1(corrected_log_redshift)


def build_tabpfn_regressor() -> TabPFNRegressor:
    """Create the frozen TabPFN v3 residual regressor."""

    model_version = ModelVersion(
        TABPFN_MODEL_VERSION
    )

    return TabPFNRegressor.create_default_for_version(
        model_version,
        n_estimators=TABPFN_N_ESTIMATORS,
        device=TABPFN_DEVICE,
        fit_mode=TABPFN_FIT_MODE,
        random_state=RANDOM_SEED,
        show_progress_bar=True,
    )


def build_validation_predictions(
    data: TabPFNDevelopmentData,
    predicted_log_residual,
) -> pd.DataFrame:
    """Build one row per validation source."""

    predicted_residual = np.asarray(
        predicted_log_residual,
        dtype=float,
    ).reshape(-1)

    if len(predicted_residual) != len(
        data.validation_catalog
    ):
        raise ValueError(
            "The number of TabPFN predictions does not match "
            "the validation catalog."
        )

    if not np.isfinite(predicted_residual).all():
        raise ValueError(
            "TabPFN produced non-finite residual predictions."
        )

    z_spec = data.validation_catalog[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

    z_eazy = data.validation_catalog[
        EAZY_MODEL_FEATURE
    ].to_numpy(dtype=float)

    z_tabpfn = apply_log_residual_correction(
        eazy_redshift=z_eazy,
        predicted_log_residual=predicted_residual,
    )

    eazy_normalized_error = (
        z_eazy - z_spec
    ) / (1.0 + z_spec)

    tabpfn_normalized_error = (
        z_tabpfn - z_spec
    ) / (1.0 + z_spec)

    predictions = pd.DataFrame(
        {
            SOURCE_KEY_COLUMN: (
                data.validation_catalog[
                    SOURCE_KEY_COLUMN
                ].to_numpy()
            ),
            "z_spec": z_spec,
            "z_eazy": z_eazy,
            "true_log_residual": data.y_validation,
            "predicted_log_residual": predicted_residual,
            TABPFN_REDSHIFT_COLUMN: z_tabpfn,
            "eazy_normalized_error": (
                eazy_normalized_error
            ),
            "tabpfn_normalized_error": (
                tabpfn_normalized_error
            ),
        }
    )

    predictions["eazy_is_catastrophic"] = (
        np.abs(predictions["eazy_normalized_error"])
        > CATASTROPHIC_OUTLIER_THRESHOLD
    )

    predictions["tabpfn_is_catastrophic"] = (
        np.abs(predictions["tabpfn_normalized_error"])
        > CATASTROPHIC_OUTLIER_THRESHOLD
    )

    predictions["absolute_error_improvement"] = (
        np.abs(predictions["eazy_normalized_error"])
        - np.abs(
            predictions["tabpfn_normalized_error"]
        )
    )

    return predictions


def build_validation_figure(
    predictions: pd.DataFrame,
    output_path: Path = VALIDATION_FIGURE_PATH,
) -> Path:
    """Create a direct EAZY-versus-TabPFN validation figure."""

    z_spec = predictions["z_spec"].to_numpy()
    z_eazy = predictions["z_eazy"].to_numpy()

    z_tabpfn = predictions[
        TABPFN_REDSHIFT_COLUMN
    ].to_numpy()

    eazy_outlier = predictions[
        "eazy_is_catastrophic"
    ].to_numpy(dtype=bool)

    tabpfn_outlier = predictions[
        "tabpfn_is_catastrophic"
    ].to_numpy(dtype=bool)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.5),
    )

    maximum_redshift = float(
        np.nanmax(
            np.concatenate(
                [z_spec, z_eazy, z_tabpfn]
            )
        )
    )

    plot_limit = max(
        1.0,
        np.ceil(maximum_redshift + 0.5),
    )

    methods = (
        (
            axes[0],
            z_eazy,
            eazy_outlier,
            "EAZY DR5",
            "#1f77b4",
        ),
        (
            axes[1],
            z_tabpfn,
            tabpfn_outlier,
            "TabPFN–EAZY residual hybrid",
            "#2ca02c",
        ),
    )

    for (
        axis,
        predicted,
        outlier_mask,
        title,
        color,
    ) in methods:
        axis.scatter(
            z_spec[~outlier_mask],
            predicted[~outlier_mask],
            s=18,
            alpha=0.65,
            color=color,
            label="Non-outlier",
        )

        axis.scatter(
            z_spec[outlier_mask],
            predicted[outlier_mask],
            s=45,
            marker="x",
            linewidth=1.6,
            color="red",
            label="Catastrophic outlier",
        )

        axis.plot(
            [0.0, plot_limit],
            [0.0, plot_limit],
            linestyle="--",
            color="black",
            linewidth=1.2,
            label="1:1 relation",
        )

        axis.set_xlim(0.0, plot_limit)
        axis.set_ylim(0.0, plot_limit)
        axis.set_xlabel("Spectroscopic redshift")
        axis.set_ylabel("Predicted redshift")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)

    absolute_eazy_error = np.maximum(
        np.abs(
            predictions[
                "eazy_normalized_error"
            ].to_numpy()
        ),
        1e-4,
    )

    absolute_tabpfn_error = np.maximum(
        np.abs(
            predictions[
                "tabpfn_normalized_error"
            ].to_numpy()
        ),
        1e-4,
    )

    error_limit = max(
        0.2,
        float(
            np.nanmax(
                np.concatenate(
                    [
                        absolute_eazy_error,
                        absolute_tabpfn_error,
                    ]
                )
            )
        ),
    )

    log_error_limit = 10 ** np.ceil(
        np.log10(error_limit)
    )

    axes[2].scatter(
        absolute_eazy_error,
        absolute_tabpfn_error,
        s=20,
        alpha=0.6,
        color="#9467bd",
    )

    axes[2].plot(
        [1e-4, log_error_limit],
        [1e-4, log_error_limit],
        linestyle="--",
        color="black",
        linewidth=1.2,
        label="Equal error",
    )

    axes[2].axvline(
        CATASTROPHIC_OUTLIER_THRESHOLD,
        linestyle=":",
        color="#1f77b4",
        label="EAZY outlier threshold",
    )

    axes[2].axhline(
        CATASTROPHIC_OUTLIER_THRESHOLD,
        linestyle=":",
        color="#2ca02c",
        label="TabPFN outlier threshold",
    )

    axes[2].set_xscale("log")
    axes[2].set_yscale("log")

    axes[2].set_xlim(
        1e-4,
        log_error_limit,
    )

    axes[2].set_ylim(
        1e-4,
        log_error_limit,
    )

    improved_fraction = float(
        np.mean(
            absolute_tabpfn_error
            < absolute_eazy_error
        )
    )

    fixed_outliers = int(
        (
            eazy_outlier
            & ~tabpfn_outlier
        ).sum()
    )

    new_outliers = int(
        (
            ~eazy_outlier
            & tabpfn_outlier
        ).sum()
    )

    axes[2].set_xlabel(
        "Absolute normalized error: EAZY"
    )

    axes[2].set_ylabel(
        "Absolute normalized error: TabPFN"
    )

    axes[2].set_title(
        "Per-source error comparison\n"
        f"Improved: {improved_fraction:.1%}, "
        f"fixed outliers: {fixed_outliers}, "
        f"new outliers: {new_outliers}"
    )

    axes[2].grid(
        alpha=0.2,
        which="both",
    )

    axes[2].legend(fontsize=8)

    figure.suptitle(
        "JADES validation: TabPFN log-residual correction",
        fontsize=15,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    # 关闭当前 figure，防止 Notebook 自动显示第二次
    plt.close(figure)

    return output_path


def save_validation_outputs(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    data: TabPFNDevelopmentData,
    runtime_seconds: float,
) -> None:
    """Save reproducible validation products."""

    metrics.to_csv(
        VALIDATION_METRICS_PATH,
        index=False,
    )

    predictions.to_csv(
        VALIDATION_PREDICTIONS_PATH,
        index=False,
    )

    manifest = {
        "experiment_id": (
            "tabpfn_v3_eazy_log_residual_validation"
        ),
        "tabpfn_package_version": version("tabpfn"),
        "tabpfn_model_version": TABPFN_MODEL_VERSION,
        "device": TABPFN_DEVICE,
        "fit_mode": TABPFN_FIT_MODE,
        "n_estimators": TABPFN_N_ESTIMATORS,
        "random_seed": RANDOM_SEED,
        "training_sources": len(data.training_catalog),
        "validation_sources": len(
            data.validation_catalog
        ),
        "feature_count": len(data.feature_columns),
        "feature_columns": list(data.feature_columns),
        "target_definition": (
            "log1p(z_spec) - log1p(z_eazy)"
        ),
        "runtime_seconds": runtime_seconds,
        "test_set_opened": False,
    }

    with EXPERIMENT_MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )


def run_tabpfn_validation_experiment(
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Run the complete TabPFN validation experiment."""

    ensure_output_directories()

    data = prepare_tabpfn_development_data()

    model = build_tabpfn_regressor()

    start_time = time.perf_counter()


    model.fit(
        data.x_train.to_numpy(dtype=np.float32),
        data.y_train,
    )

    predicted_log_residual = model.predict(
        data.x_validation.to_numpy(dtype=np.float32)
    )

    runtime_seconds = (
        time.perf_counter() - start_time
    )

    predictions = build_validation_predictions(
        data=data,
        predicted_log_residual=predicted_log_residual,
    )

    eazy_metrics = calculate_photoz_metrics(
        true_redshift=predictions["z_spec"],
        predicted_redshift=predictions["z_eazy"],
        method="EAZY DR5 z_a",
    )

    tabpfn_metrics = calculate_photoz_metrics(
        true_redshift=predictions["z_spec"],
        predicted_redshift=predictions[
            TABPFN_REDSHIFT_COLUMN
        ],
        method="TabPFN v3: EAZY log-residual",
    )

    metrics = pd.DataFrame(
        [
            eazy_metrics,
            tabpfn_metrics,
        ]
    )

    metrics["runtime_seconds"] = runtime_seconds

    figure_path = build_validation_figure(
        predictions
    )

    save_validation_outputs(
        metrics=metrics,
        predictions=predictions,
        data=data,
        runtime_seconds=runtime_seconds,
    )

    return (
        metrics,
        predictions,
        figure_path,
    )