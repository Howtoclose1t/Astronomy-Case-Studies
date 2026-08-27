# -*- coding: utf-8 -*-

"""Compare five frozen Photo-z methods on the same validation sources."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    CATASTROPHIC_OUTLIER_THRESHOLD,
    FIGURE_DIR,
    METRICS_DIR,
    PREDICTION_DIR,
    SOURCE_KEY_COLUMN,
    TASK5_OUTPUT_DIR,
    ensure_output_directories,
)
from .tabpfn import calculate_photoz_metrics


XGBOOST_PREDICTIONS_PATH = (
    TASK5_OUTPUT_DIR
    / "predictions"
    / "dr45_xgboost_validation_predictions.csv"
)

GATED_XGBOOST_PREDICTIONS_PATH = (
    TASK5_OUTPUT_DIR
    / "predictions"
    / "dr45_hybrid_validation_predictions.csv"
)

TABPFN_ABLATION_PREDICTIONS_PATH = (
    PREDICTION_DIR
    / "tabpfn_validation_ablation_predictions.csv"
)

COMPARISON_METRICS_PATH = (
    METRICS_DIR
    / "five_method_validation_metrics.csv"
)

COMPARISON_PREDICTIONS_PATH = (
    PREDICTION_DIR
    / "five_method_validation_predictions.csv"
)

COMPARISON_FIGURE_PATH = (
    FIGURE_DIR
    / "five_method_validation_comparison.png"
)

COMPARISON_MANIFEST_PATH = (
    METRICS_DIR
    / "five_method_validation_manifest.json"
)


METHOD_DEFINITIONS = (
    (
        "EAZY DR5 z_a",
        "z_eazy",
        "Template-fitting benchmark",
    ),
    (
        "XGBoost: photometry only",
        "z_xgboost",
        "Traditional ML baseline",
    ),
    (
        "XGBoost: gated EAZY residual",
        "z_gated_xgboost",
        "Traditional ML hybrid",
    ),
    (
        "TabPFN v3: photometry only",
        "z_tabpfn",
        "Foundation-model baseline",
    ),
    (
        "TabPFN v3: EAZY residual",
        "z_tabpfn_eazy",
        "Foundation-model hybrid",
    ),
)


METHOD_COLORS = {
    "EAZY DR5 z_a": "#1f77b4",
    "XGBoost: photometry only": "#ff7f0e",
    "XGBoost: gated EAZY residual": "#d62728",
    "TabPFN v3: photometry only": "#9467bd",
    "TabPFN v3: EAZY residual": "#2ca02c",
}


def load_tabpfn_base_predictions() -> pd.DataFrame:
    """Load EAZY, pure TabPFN, and TabPFN–EAZY predictions."""

    if not TABPFN_ABLATION_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "The TabPFN ablation predictions were not found. "
            "Run the TabPFN ablation Cell first."
        )

    predictions = pd.read_csv(
        TABPFN_ABLATION_PREDICTIONS_PATH,
        low_memory=False,
    )

    required_columns = {
        SOURCE_KEY_COLUMN,
        "z_spec",
        "z_eazy",
        "z_tabpfn_photometry_only",
        "z_tabpfn_residual",
    }

    missing_columns = sorted(
        required_columns - set(predictions.columns)
    )

    if missing_columns:
        raise KeyError(
            "The TabPFN predictions are missing: "
            f"{missing_columns}"
        )

    if not predictions[SOURCE_KEY_COLUMN].is_unique:
        raise ValueError(
            "The TabPFN predictions contain duplicated sources."
        )

    if len(predictions) != 476:
        raise ValueError(
            f"Expected 476 validation sources, "
            f"found {len(predictions)}."
        )

    numeric_columns = (
        "z_spec",
        "z_eazy",
        "z_tabpfn_photometry_only",
        "z_tabpfn_residual",
    )

    for column in numeric_columns:
        predictions[column] = pd.to_numeric(
            predictions[column],
            errors="coerce",
        )

        if not np.isfinite(
            predictions[column]
        ).all():
            raise ValueError(
                f"{column} contains non-finite predictions."
            )

    return predictions


def load_selected_task5_prediction(
    path: Path,
    experiment_id: str,
    source_keys: pd.Series,
    expected_z_spec,
) -> np.ndarray:
    """Load one frozen Task 5 experiment and align its source order."""

    if not path.exists():
        raise FileNotFoundError(
            f"Task 5 predictions were not found: {path}"
        )

    catalog = pd.read_csv(
        path,
        low_memory=False,
    )

    required_columns = {
        SOURCE_KEY_COLUMN,
        "z_spec",
        "experiment_id",
        "z_prediction",
    }

    missing_columns = sorted(
        required_columns - set(catalog.columns)
    )

    if missing_columns:
        raise KeyError(
            "Task 5 predictions are missing: "
            f"{missing_columns}"
        )

    selected = catalog.loc[
        catalog["experiment_id"].eq(experiment_id)
    ].copy()

    if len(selected) != len(source_keys):
        raise ValueError(
            f"{experiment_id} contains {len(selected)} rows; "
            f"{len(source_keys)} were expected."
        )

    if not selected[SOURCE_KEY_COLUMN].is_unique:
        raise ValueError(
            f"{experiment_id} contains duplicated sources."
        )

    selected = selected.set_index(
        SOURCE_KEY_COLUMN
    )

    missing_sources = sorted(
        set(source_keys) - set(selected.index)
    )

    extra_sources = sorted(
        set(selected.index) - set(source_keys)
    )

    if missing_sources or extra_sources:
        raise ValueError(
            f"{experiment_id} does not contain the same "
            "validation sources as TabPFN."
        )

    selected = selected.loc[
        source_keys.tolist()
    ]

    selected_z_spec = pd.to_numeric(
        selected["z_spec"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.allclose(
        selected_z_spec,
        np.asarray(expected_z_spec, dtype=float),
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError(
            f"{experiment_id} uses different z_spec values."
        )

    prediction = pd.to_numeric(
        selected["z_prediction"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(prediction).all():
        raise ValueError(
            f"{experiment_id} contains non-finite predictions."
        )

    return prediction


def build_five_method_predictions() -> pd.DataFrame:
    """Combine five already completed validation experiments."""

    tabpfn_predictions = (
        load_tabpfn_base_predictions()
    )

    source_keys = tabpfn_predictions[
        SOURCE_KEY_COLUMN
    ]

    z_spec = tabpfn_predictions[
        "z_spec"
    ].to_numpy(dtype=float)

    # Pure XGBoost：57个测光特征，使用 sample weights
    z_xgboost = load_selected_task5_prediction(
        path=XGBOOST_PREDICTIONS_PATH,
        experiment_id="xgb_full_weighted",
        source_keys=source_keys,
        expected_z_spec=z_spec,
    )

    # Gated XGBoost：EAZY residual + gate classifier
    z_gated_xgboost = (
        load_selected_task5_prediction(
            path=GATED_XGBOOST_PREDICTIONS_PATH,
            experiment_id="xgb_eazy_gated_residual",
            source_keys=source_keys,
            expected_z_spec=z_spec,
        )
    )

    predictions = pd.DataFrame(
        {
            SOURCE_KEY_COLUMN: source_keys.to_numpy(),
            "z_spec": z_spec,
            "z_eazy": tabpfn_predictions[
                "z_eazy"
            ].to_numpy(dtype=float),
            "z_xgboost": z_xgboost,
            "z_gated_xgboost": z_gated_xgboost,
            "z_tabpfn": tabpfn_predictions[
                "z_tabpfn_photometry_only"
            ].to_numpy(dtype=float),
            "z_tabpfn_eazy": tabpfn_predictions[
                "z_tabpfn_residual"
            ].to_numpy(dtype=float),
        }
    )

    # 保存每种方法逐源的标准化误差和 outlier 状态
    for _, prediction_column, _ in METHOD_DEFINITIONS:
        method_prefix = prediction_column.removeprefix(
            "z_"
        )

        normalized_error = (
            predictions[prediction_column]
            - predictions["z_spec"]
        ) / (
            1.0 + predictions["z_spec"]
        )

        predictions[
            f"{method_prefix}_normalized_error"
        ] = normalized_error

        predictions[
            f"{method_prefix}_absolute_normalized_error"
        ] = np.abs(normalized_error)

        predictions[
            f"{method_prefix}_is_catastrophic"
        ] = (
            np.abs(normalized_error)
            > CATASTROPHIC_OUTLIER_THRESHOLD
        )

    return predictions


def build_five_method_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate identical Photo-z metrics for all five methods."""

    metric_rows = []

    for (
        method,
        prediction_column,
        role,
    ) in METHOD_DEFINITIONS:
        metrics = calculate_photoz_metrics(
            true_redshift=predictions["z_spec"],
            predicted_redshift=predictions[
                prediction_column
            ],
            method=method,
        )

        metrics["role"] = role
        metric_rows.append(metrics)

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

    ordered_columns = (
        "validation_rank",
        "method",
        "role",
        "sources",
        "normalized_bias",
        "normalized_median_absolute_error",
        "sigma_nmad",
        "mean_absolute_redshift_error",
        "catastrophic_outliers",
        "catastrophic_outlier_fraction",
    )

    return metrics.loc[
        :,
        list(ordered_columns),
    ]


def draw_redshift_panel(
    axis,
    predictions: pd.DataFrame,
    method: str,
    prediction_column: str,
) -> None:
    """Draw one method using the same axes and outlier definition."""

    z_spec = predictions[
        "z_spec"
    ].to_numpy(dtype=float)

    z_prediction = predictions[
        prediction_column
    ].to_numpy(dtype=float)

    normalized_error = (
        z_prediction - z_spec
    ) / (1.0 + z_spec)

    outlier = (
        np.abs(normalized_error)
        > CATASTROPHIC_OUTLIER_THRESHOLD
    )

    axis.scatter(
        z_spec[~outlier],
        z_prediction[~outlier],
        s=15,
        alpha=0.6,
        color=METHOD_COLORS[method],
        label="Non-outlier",
    )

    axis.scatter(
        z_spec[outlier],
        z_prediction[outlier],
        s=38,
        marker="x",
        linewidth=1.5,
        color="red",
        label="Catastrophic outlier",
    )

    axis.plot(
        [0.0, 15.0],
        [0.0, 15.0],
        linestyle="--",
        linewidth=1.0,
        color="black",
        label="1:1 relation",
    )

    axis.set_xlim(0.0, 15.0)
    axis.set_ylim(0.0, 15.0)
    axis.set_xlabel("Spectroscopic redshift")
    axis.set_ylabel("Predicted redshift")
    axis.set_title(method)
    axis.grid(alpha=0.2)


def build_five_method_figure(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    output_path: Path = COMPARISON_FIGURE_PATH,
) -> Path:
    """Create a five-method prediction and performance comparison."""

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11),
    )

    panel_axes = (
        axes[0, 0],
        axes[0, 1],
        axes[0, 2],
        axes[1, 0],
        axes[1, 1],
    )

    for (
        axis,
        (
            method,
            prediction_column,
            _,
        ),
    ) in zip(
        panel_axes,
        METHOD_DEFINITIONS,
    ):
        draw_redshift_panel(
            axis=axis,
            predictions=predictions,
            method=method,
            prediction_column=prediction_column,
        )

    summary_axis = axes[1, 2]

    for _, row in metrics.iterrows():
        summary_axis.scatter(
            row[
                "normalized_median_absolute_error"
            ],
            row[
                "catastrophic_outlier_fraction"
            ],
            s=90,
            color=METHOD_COLORS[row["method"]],
        )

        summary_axis.annotate(
            row["method"],
            (
                row[
                    "normalized_median_absolute_error"
                ],
                row[
                    "catastrophic_outlier_fraction"
                ],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    eazy_metrics = metrics.loc[
        metrics["method"].eq("EAZY DR5 z_a")
    ].iloc[0]

    summary_axis.axvline(
        eazy_metrics[
            "normalized_median_absolute_error"
        ],
        linestyle="--",
        linewidth=1.0,
        color="#1f77b4",
    )

    summary_axis.axhline(
        eazy_metrics[
            "catastrophic_outlier_fraction"
        ],
        linestyle="--",
        linewidth=1.0,
        color="#1f77b4",
    )

    summary_axis.set_xscale("log")
    summary_axis.set_yscale("log")

    summary_axis.set_xlabel(
        "Median absolute normalized error"
    )

    summary_axis.set_ylabel(
        "Catastrophic outlier fraction"
    )

    summary_axis.set_title(
        "Central accuracy versus catastrophic failures\n"
        "Lower-left is better"
    )

    summary_axis.grid(
        alpha=0.2,
        which="both",
    )

    figure.suptitle(
        "JADES validation: EAZY, XGBoost and TabPFN",
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


def save_five_method_outputs(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Save the frozen five-method validation comparison."""

    metrics.to_csv(
        COMPARISON_METRICS_PATH,
        index=False,
    )

    predictions.to_csv(
        COMPARISON_PREDICTIONS_PATH,
        index=False,
    )

    manifest = {
        "methods": [
            method
            for method, _, _ in METHOD_DEFINITIONS
        ],
        "validation_sources": len(predictions),
        "ranking_metric": (
            "normalized_median_absolute_error"
        ),
        "catastrophic_outlier_threshold": (
            CATASTROPHIC_OUTLIER_THRESHOLD
        ),
        "models_retrained": False,
        "test_set_opened": False,
    }

    with COMPARISON_MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )


def run_five_method_validation_comparison(
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Run the complete saved-prediction comparison."""

    ensure_output_directories()

    predictions = build_five_method_predictions()

    metrics = build_five_method_metrics(
        predictions
    )

    figure_path = build_five_method_figure(
        predictions=predictions,
        metrics=metrics,
    )

    save_five_method_outputs(
        metrics=metrics,
        predictions=predictions,
    )

    return (
        metrics,
        predictions,
        figure_path,
    )