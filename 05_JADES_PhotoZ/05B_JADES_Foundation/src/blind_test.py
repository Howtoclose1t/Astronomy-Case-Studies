# -*- coding: utf-8 -*-

"""Run the frozen five-method TabPFN blind-test comparison."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ablation import (
    convert_direct_log_prediction,
    fit_tabpfn_context,
)
from .config import (
    CATASTROPHIC_OUTLIER_THRESHOLD,
    EAZY_MODEL_FEATURE,
    FIGURE_DIR,
    METRICS_DIR,
    PREDICTION_DIR,
    SOURCE_KEY_COLUMN,
    TASK5_OUTPUT_DIR,
    TARGET_COLUMN,
    TEST_FEATURES_PATH,
    ensure_output_directories,
)
from .data import (
    load_development_catalog,
    load_feature_columns,
    prepare_feature_matrix,
    prepare_tabpfn_development_data,
)
from .tabpfn import (
    apply_log_residual_correction,
    calculate_photoz_metrics,
)


TASK5_FINAL_TEST_PREDICTIONS_PATH = (
    TASK5_OUTPUT_DIR
    / "predictions"
    / "dr45_final_test_predictions.csv"
)

BLIND_TEST_METRICS_PATH = (
    METRICS_DIR
    / "tabpfn_blind_test_metrics.csv"
)

BLIND_TEST_PREDICTIONS_PATH = (
    PREDICTION_DIR
    / "tabpfn_blind_test_predictions.csv"
)

BLIND_TEST_FIGURE_PATH = (
    FIGURE_DIR
    / "tabpfn_blind_test_comparison.png"
)

BLIND_TEST_MANIFEST_PATH = (
    METRICS_DIR
    / "tabpfn_blind_test_manifest.json"
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
        "Frozen Task 5A traditional ML baseline",
    ),
    (
        "XGBoost: gated EAZY residual",
        "z_gated_xgboost",
        "Frozen Task 5A traditional ML hybrid",
    ),
    (
        "TabPFN v3: photometry only",
        "z_tabpfn",
        "Frozen foundation-model ablation",
    ),
    (
        "TabPFN v3: EAZY residual",
        "z_tabpfn_eazy",
        "Preselected primary foundation-model hybrid",
    ),
)


METHOD_COLORS = {
    "EAZY DR5 z_a": "#1f77b4",
    "XGBoost: photometry only": "#ff7f0e",
    "XGBoost: gated EAZY residual": "#d62728",
    "TabPFN v3: photometry only": "#9467bd",
    "TabPFN v3: EAZY residual": "#2ca02c",
}


def load_one_task5_test_prediction(
    catalog: pd.DataFrame,
    experiment_id: str,
    test_source_keys: pd.Series,
    expected_z_spec,
) -> np.ndarray:
    """Select and align one frozen Task 5A test result."""

    selected = catalog.loc[
        catalog["experiment_id"].eq(
            experiment_id
        )
    ].copy()

    if len(selected) != len(test_source_keys):
        raise ValueError(
            f"{experiment_id} contains {len(selected)} rows; "
            f"{len(test_source_keys)} were expected."
        )

    if not selected[SOURCE_KEY_COLUMN].is_unique:
        raise ValueError(
            f"{experiment_id} contains duplicated sources."
        )

    selected = selected.set_index(
        SOURCE_KEY_COLUMN
    )

    missing_sources = sorted(
        set(test_source_keys)
        - set(selected.index)
    )

    extra_sources = sorted(
        set(selected.index)
        - set(test_source_keys)
    )

    if missing_sources or extra_sources:
        raise ValueError(
            f"{experiment_id} does not contain the same "
            "blind-test sources."
        )

    selected = selected.loc[
        test_source_keys.tolist()
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


def load_task5_frozen_test_predictions(
    test_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Load EAZY and XGBoost test results without retraining."""

    if not TASK5_FINAL_TEST_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "The frozen Task 5A test predictions were not found: "
            f"{TASK5_FINAL_TEST_PREDICTIONS_PATH}"
        )

    catalog = pd.read_csv(
        TASK5_FINAL_TEST_PREDICTIONS_PATH,
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
            "The Task 5A test file is missing: "
            f"{missing_columns}"
        )

    test_source_keys = test_catalog[
        SOURCE_KEY_COLUMN
    ]

    z_spec = test_catalog[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

    z_eazy = load_one_task5_test_prediction(
        catalog=catalog,
        experiment_id="eazy_dr5_za",
        test_source_keys=test_source_keys,
        expected_z_spec=z_spec,
    )

    z_xgboost = load_one_task5_test_prediction(
        catalog=catalog,
        experiment_id="xgb_full_weighted",
        test_source_keys=test_source_keys,
        expected_z_spec=z_spec,
    )

    z_gated_xgboost = (
        load_one_task5_test_prediction(
            catalog=catalog,
            experiment_id="xgb_eazy_gated_residual",
            test_source_keys=test_source_keys,
            expected_z_spec=z_spec,
        )
    )


    catalog_eazy = test_catalog[
        EAZY_MODEL_FEATURE
    ].to_numpy(dtype=float)

    if not np.allclose(
        z_eazy,
        catalog_eazy,
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError(
            "The frozen Task 5A and current EAZY predictions disagree."
        )

    return pd.DataFrame(
        {
            SOURCE_KEY_COLUMN: test_source_keys.to_numpy(),
            "z_spec": z_spec,
            "z_eazy": z_eazy,
            "z_xgboost": z_xgboost,
            "z_gated_xgboost": z_gated_xgboost,
        }
    )


def run_frozen_tabpfn_models(
    test_catalog: pd.DataFrame,
    full_feature_columns: tuple[str, ...],
    residual_feature_columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit frozen TabPFN contexts and predict the blind-test sources."""

    development_data = (
        prepare_tabpfn_development_data()
    )

    context_catalog = pd.concat(
        [
            development_data.training_catalog,
            development_data.validation_catalog,
        ],
        axis=0,
        ignore_index=True,
    )

    context_sources = set(
        context_catalog[SOURCE_KEY_COLUMN]
    )

    test_sources = set(
        test_catalog[SOURCE_KEY_COLUMN]
    )

    if context_sources & test_sources:
        raise ValueError(
            "Context–blind-test source leakage was detected."
        )

    if len(context_catalog) != 2695:
        raise ValueError(
            f"Expected 2695 context sources, "
            f"found {len(context_catalog)}."
        )

    # --------------------------------------------------------------
    # Pure TabPFN：57 features
    # --------------------------------------------------------------

    direct_x_context = prepare_feature_matrix(
        context_catalog,
        full_feature_columns,
    )

    direct_x_test = prepare_feature_matrix(
        test_catalog,
        full_feature_columns,
    )

    direct_y_context = np.log1p(
        context_catalog[
            TARGET_COLUMN
        ].to_numpy(dtype=float)
    ).astype(np.float32)

    (
        predicted_direct_log_redshift,
        direct_runtime,
    ) = fit_tabpfn_context(
        x_train=direct_x_context,
        y_train=direct_y_context,
        x_validation=direct_x_test,
    )

    z_tabpfn = convert_direct_log_prediction(
        predicted_direct_log_redshift
    )

    # --------------------------------------------------------------
    # TabPFN–EAZY：66 features
    # --------------------------------------------------------------

    residual_x_context = prepare_feature_matrix(
        context_catalog,
        residual_feature_columns,
    )

    residual_x_test = prepare_feature_matrix(
        test_catalog,
        residual_feature_columns,
    )

    z_spec_context = context_catalog[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

    z_eazy_context = context_catalog[
        EAZY_MODEL_FEATURE
    ].to_numpy(dtype=float)

    residual_y_context = (
        np.log1p(z_spec_context)
        - np.log1p(z_eazy_context)
    ).astype(np.float32)

    (
        predicted_log_residual,
        residual_runtime,
    ) = fit_tabpfn_context(
        x_train=residual_x_context,
        y_train=residual_y_context,
        x_validation=residual_x_test,
    )

    z_tabpfn_eazy = (
        apply_log_residual_correction(
            eazy_redshift=test_catalog[
                EAZY_MODEL_FEATURE
            ].to_numpy(dtype=float),
            predicted_log_residual=(
                predicted_log_residual
            ),
        )
    )

    runtimes = {
        "TabPFN v3: photometry only": direct_runtime,
        "TabPFN v3: EAZY residual": residual_runtime,
    }

    return (
        z_tabpfn,
        z_tabpfn_eazy,
        runtimes,
    )


def add_error_columns(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Add identical normalized errors for all five methods."""

    result = predictions.copy()

    for _, prediction_column, _ in METHOD_DEFINITIONS:
        prefix = prediction_column.removeprefix(
            "z_"
        )

        normalized_error = (
            result[prediction_column]
            - result["z_spec"]
        ) / (
            1.0 + result["z_spec"]
        )

        result[
            f"{prefix}_normalized_error"
        ] = normalized_error

        result[
            f"{prefix}_absolute_normalized_error"
        ] = np.abs(normalized_error)

        result[
            f"{prefix}_is_catastrophic"
        ] = (
            np.abs(normalized_error)
            > CATASTROPHIC_OUTLIER_THRESHOLD
        )

    return result


def build_blind_test_metrics(
    predictions: pd.DataFrame,
    runtimes: dict[str, float],
) -> pd.DataFrame:
    """Calculate five-method blind-test metrics."""

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
        metrics["runtime_seconds"] = (
            runtimes.get(method, np.nan)
        )

        metric_rows.append(metrics)

    metrics = pd.DataFrame(metric_rows)

    metrics["blind_test_rank"] = (
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
        "blind_test_rank",
        "method",
        "role",
        "sources",
        "normalized_bias",
        "normalized_median_absolute_error",
        "sigma_nmad",
        "mean_absolute_redshift_error",
        "catastrophic_outliers",
        "catastrophic_outlier_fraction",
        "runtime_seconds",
    )

    return metrics.loc[
        :,
        list(ordered_columns),
    ]


def draw_prediction_panel(
    axis,
    predictions: pd.DataFrame,
    method: str,
    prediction_column: str,
) -> None:
    """Draw one blind-test redshift panel."""

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
    )

    axis.scatter(
        z_spec[outlier],
        z_prediction[outlier],
        s=38,
        marker="x",
        linewidth=1.5,
        color="red",
    )

    axis.plot(
        [0.0, 15.0],
        [0.0, 15.0],
        linestyle="--",
        color="black",
        linewidth=1.0,
    )

    axis.set_xlim(0.0, 15.0)
    axis.set_ylim(0.0, 15.0)
    axis.set_xlabel("Spectroscopic redshift")
    axis.set_ylabel("Predicted redshift")
    axis.set_title(method)
    axis.grid(alpha=0.2)


METHOD_COLORS = {
    "EAZY DR5 z_a": "#1f77b4",
    "XGBoost: photometry only": "#ff7f0e",
    "XGBoost: gated EAZY residual": "#d62728",
    "TabPFN v3: photometry only": "#9467bd",
    "TabPFN v3: EAZY residual": "#2ca02c",
}


def build_blind_test_figure(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    output_path: Path = BLIND_TEST_FIGURE_PATH,
) -> Path:
    """Create the final five-method blind-test figure."""

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
        draw_prediction_panel(
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

    summary_axis.set_xscale("log")
    summary_axis.set_yscale("log")

    summary_axis.set_xlabel(
        "Median absolute normalized error"
    )

    summary_axis.set_ylabel(
        "Catastrophic outlier fraction"
    )

    summary_axis.set_title(
        "Blind-test central accuracy versus failures\n"
        "Lower-left is better"
    )

    summary_axis.grid(
        alpha=0.2,
        which="both",
    )

    figure.suptitle(
        "JADES TabPFN blind-test comparison",
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


def save_blind_test_outputs(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    runtimes: dict[str, float],
) -> None:
    """Save the frozen blind-test products."""

    metrics.to_csv(
        BLIND_TEST_METRICS_PATH,
        index=False,
    )

    predictions.to_csv(
        BLIND_TEST_PREDICTIONS_PATH,
        index=False,
    )

    manifest = {
        "primary_method": (
            "TabPFN v3: EAZY residual"
        ),
        "secondary_tabpfn_ablation": (
            "TabPFN v3: photometry only"
        ),
        "tabpfn_context_sources": 2695,
        "blind_test_sources": len(predictions),
        "task5a_baselines_retrained": False,
        "task5a_predictions_reused": True,
        "blind_with_respect_to_tabpfn_development": True,
        "tabpfn_runtimes_seconds": runtimes,
        "further_test_tuning_allowed": False,
    }

    with BLIND_TEST_MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )


def run_tabpfn_blind_test(
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Run the complete frozen TabPFN blind test."""

    ensure_output_directories()

    (
        full_feature_columns,
        residual_feature_columns,
    ) = load_feature_columns()

    test_catalog = load_development_catalog(
        path=TEST_FEATURES_PATH,
        expected_split="test",
        catalog_name="TabPFN blind-test catalog",
        full_feature_columns=full_feature_columns,
    )

    predictions = (
        load_task5_frozen_test_predictions(
            test_catalog
        )
    )

    (
        z_tabpfn,
        z_tabpfn_eazy,
        runtimes,
    ) = run_frozen_tabpfn_models(
        test_catalog=test_catalog,
        full_feature_columns=full_feature_columns,
        residual_feature_columns=residual_feature_columns,
    )

    predictions["z_tabpfn"] = z_tabpfn
    predictions["z_tabpfn_eazy"] = (
        z_tabpfn_eazy
    )

    predictions = add_error_columns(
        predictions
    )

    metrics = build_blind_test_metrics(
        predictions=predictions,
        runtimes=runtimes,
    )

    figure_path = build_blind_test_figure(
        predictions=predictions,
        metrics=metrics,
    )

    save_blind_test_outputs(
        metrics=metrics,
        predictions=predictions,
        runtimes=runtimes,
    )

    return (
        metrics,
        predictions,
        figure_path,
    )