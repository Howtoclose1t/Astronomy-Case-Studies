"""Evaluate frozen Photo-z models once on the blind test split."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from .config import (
    FIGURE_DIR,
    METRICS_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
    SOURCE_KEY_COLUMN,
    TARGET_COLUMN,
    ensure_output_directories,
)
from .features import (
    EAZY_HYBRID_FEATURE,
    FULL_FEATURE_COLUMNS,
    HYBRID_FEATURE_COLUMNS,
    TEST_FEATURES_PATH,
)
from .hybrid import (
    ABSOLUTE_PREDICTED_RESIDUAL_FEATURE,
    GATE_FEATURE_COLUMNS,
    GATE_MODEL_PATH,
    HYBRID_MANIFEST_PATH,
    PREDICTED_RESIDUAL_FEATURE,
    RESIDUAL_FEATURE_COLUMNS,
    RESIDUAL_MODEL_PATH,
    add_eazy_diagnostic_features,
    apply_log_residual_correction,
    prepare_matrix,
)
from .xgboost import (
    CATASTROPHIC_COLUMN,
    build_prediction_table,
    compute_photoz_metrics,
    inverse_transform_redshift,
)


REDSHIFT_STRATUM_COLUMN = "redshift_stratum"
SPLIT_COLUMN = "split"

PURE_XGBOOST_MODEL_PATH = (
    MODEL_DIR
    / "xgb_full_weighted.json"
)

DIRECT_HYBRID_MODEL_PATH = (
    MODEL_DIR
    / "xgb_eazy_hybrid.json"
)

FINAL_TEST_METRICS_PATH = (
    METRICS_DIR
    / "dr45_final_test_metrics.csv"
)

FINAL_TEST_REDSHIFT_METRICS_PATH = (
    METRICS_DIR
    / "dr45_final_test_metrics_by_redshift.csv"
)

FINAL_TEST_PREDICTIONS_PATH = (
    PREDICTION_DIR
    / "dr45_final_test_predictions.csv"
)

FINAL_TEST_FIGURE_PATH = (
    FIGURE_DIR
    / "dr45_final_test_eazy_vs_hybrid.png"
)

REDSHIFT_STRATUM_LABELS = {
    0: "0 <= z < 1",
    1: "1 <= z < 2",
    2: "2 <= z < 3",
    3: "3 <= z < 4",
    4: "4 <= z < 6",
    5: "6 <= z < 8",
    6: "z >= 8",
}


def load_test_catalog() -> pd.DataFrame:
    """Open the blind test feature catalog for final evaluation."""

    if not TEST_FEATURES_PATH.exists():
        raise FileNotFoundError(
            "The test feature catalog was not found at "
            f"{TEST_FEATURES_PATH}"
        )

    test_catalog = pd.read_csv(
        TEST_FEATURES_PATH,
        low_memory=False,
    )

    required_columns = {
        SOURCE_KEY_COLUMN,
        TARGET_COLUMN,
        REDSHIFT_STRATUM_COLUMN,
        SPLIT_COLUMN,
        EAZY_HYBRID_FEATURE,
        *FULL_FEATURE_COLUMNS,
        *HYBRID_FEATURE_COLUMNS,
    }

    missing_columns = sorted(
        required_columns - set(test_catalog.columns)
    )

    if missing_columns:
        raise KeyError(
            "The test feature catalog is missing columns: "
            f"{missing_columns}"
        )

    if not test_catalog[SPLIT_COLUMN].eq(
        "test"
    ).all():
        raise ValueError(
            "The final evaluation catalog contains non-test rows."
        )

    if not test_catalog[
        SOURCE_KEY_COLUMN
    ].is_unique:
        raise ValueError(
            "The blind test catalog contains duplicated sources."
        )

    test_catalog[TARGET_COLUMN] = pd.to_numeric(
        test_catalog[TARGET_COLUMN],
        errors="coerce",
    )

    test_catalog[EAZY_HYBRID_FEATURE] = (
        pd.to_numeric(
            test_catalog[
                EAZY_HYBRID_FEATURE
            ],
            errors="coerce",
        )
    )

    if not np.isfinite(
        test_catalog[TARGET_COLUMN]
    ).all():
        raise ValueError(
            "The blind test catalog contains invalid z_spec."
        )

    if not np.isfinite(
        test_catalog[EAZY_HYBRID_FEATURE]
    ).all():
        raise ValueError(
            "The blind test catalog contains invalid EAZY z_a."
        )

    return test_catalog


def load_regression_model(
    model_path,
) -> XGBRegressor:
    """Load one previously fitted XGBoost regressor."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"The frozen model was not found at {model_path}"
        )

    model = XGBRegressor()
    model.load_model(model_path)

    return model


def load_gate_model() -> XGBClassifier:
    """Load the frozen residual-correction gate."""

    if not GATE_MODEL_PATH.exists():
        raise FileNotFoundError(
            "The frozen gate model was not found at "
            f"{GATE_MODEL_PATH}"
        )

    model = XGBClassifier()
    model.load_model(GATE_MODEL_PATH)

    return model


def load_selected_gate_threshold() -> float:
    """Load the gate threshold selected using training OOF results."""

    if not HYBRID_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "The hybrid manifest was not found at "
            f"{HYBRID_MANIFEST_PATH}"
        )

    with HYBRID_MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        manifest = json.load(file_handle)

    if "selected_gate_threshold" not in manifest:
        raise KeyError(
            "The hybrid manifest has no selected gate threshold."
        )

    return float(
        manifest["selected_gate_threshold"]
    )


def predict_standard_xgboost(
    model_path,
    test_catalog: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> np.ndarray:
    """Predict z from a frozen model trained on log(1 + z_spec)."""

    model = load_regression_model(
        model_path
    )

    feature_matrix = prepare_matrix(
        test_catalog,
        feature_columns,
    )

    transformed_prediction = model.predict(
        feature_matrix
    )

    return inverse_transform_redshift(
        transformed_prediction
    )


def predict_gated_residual_hybrid(
    test_catalog: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict the selected gated residual hybrid on blind test data."""

    diagnostic_catalog = (
        add_eazy_diagnostic_features(
            test_catalog
        )
    )

    residual_model = load_regression_model(
        RESIDUAL_MODEL_PATH
    )

    residual_matrix = prepare_matrix(
        diagnostic_catalog,
        RESIDUAL_FEATURE_COLUMNS,
    )

    predicted_log_residual = (
        residual_model.predict(
            residual_matrix
        )
    )

    residual_prediction = (
        apply_log_residual_correction(
            diagnostic_catalog[
                EAZY_HYBRID_FEATURE
            ],
            predicted_log_residual,
        )
    )

    gate_catalog = diagnostic_catalog.copy()

    gate_catalog[
        PREDICTED_RESIDUAL_FEATURE
    ] = predicted_log_residual

    gate_catalog[
        ABSOLUTE_PREDICTED_RESIDUAL_FEATURE
    ] = np.abs(
        predicted_log_residual
    )

    gate_matrix = prepare_matrix(
        gate_catalog,
        GATE_FEATURE_COLUMNS,
    )

    gate_model = load_gate_model()

    gate_probability = (
        gate_model.predict_proba(
            gate_matrix
        )[:, 1]
    )

    selected_threshold = (
        load_selected_gate_threshold()
    )

    correction_applied = (
        gate_probability
        >= selected_threshold
    )

    eazy_prediction = diagnostic_catalog[
        EAZY_HYBRID_FEATURE
    ].to_numpy(dtype=float)

    gated_prediction = np.where(
        correction_applied,
        residual_prediction,
        eazy_prediction,
    )

    return (
        gated_prediction,
        correction_applied,
        gate_probability,
    )


def summarize_test_method(
    test_catalog: pd.DataFrame,
    experiment_id: str,
    method: str,
    prediction: np.ndarray,
    pre_test_role: str,
    correction_applied=None,
    gate_probability=None,
) -> tuple[dict, pd.DataFrame]:
    """Calculate final metrics and source-level predictions."""

    metrics = compute_photoz_metrics(
        test_catalog[TARGET_COLUMN],
        prediction,
    )

    if correction_applied is None:
        correction_fraction = np.nan
    else:
        correction_fraction = float(
            np.asarray(
                correction_applied,
                dtype=bool,
            ).mean()
        )

    metrics.update(
        {
            "experiment_id": experiment_id,
            "method": method,
            "pre_test_role": pre_test_role,
            "correction_fraction": (
                correction_fraction
            ),
        }
    )

    prediction_table = build_prediction_table(
        test_catalog,
        experiment_id=experiment_id,
        method=method,
        predicted_redshift=prediction,
    )

    prediction_table[
        "pre_test_role"
    ] = pre_test_role

    if correction_applied is None:
        prediction_table[
            "correction_applied"
        ] = False
    else:
        prediction_table[
            "correction_applied"
        ] = np.asarray(
            correction_applied,
            dtype=bool,
        )

    if gate_probability is None:
        prediction_table[
            "gate_probability"
        ] = np.nan
    else:
        prediction_table[
            "gate_probability"
        ] = np.asarray(
            gate_probability,
            dtype=float,
        )

    return metrics, prediction_table


def build_redshift_metrics(
    test_catalog: pd.DataFrame,
    method_predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compare EAZY and the selected hybrid in each redshift stratum."""

    rows = []

    for method, prediction in (
        method_predictions.items()
    ):
        for stratum in sorted(
            REDSHIFT_STRATUM_LABELS
        ):
            selected = (
                test_catalog[
                    REDSHIFT_STRATUM_COLUMN
                ]
                == stratum
            )

            if not selected.any():
                continue

            subset_true = test_catalog.loc[
                selected,
                TARGET_COLUMN,
            ].to_numpy(dtype=float)

            subset_prediction = np.asarray(
                prediction,
                dtype=float,
            )[selected.to_numpy()]

            metrics = compute_photoz_metrics(
                subset_true,
                subset_prediction,
            )

            rows.append(
                {
                    "method": method,
                    "redshift_stratum": stratum,
                    "z_spec_bin": (
                        REDSHIFT_STRATUM_LABELS[
                            stratum
                        ]
                    ),
                    "sources": int(
                        selected.sum()
                    ),
                    "median_z_spec": float(
                        np.median(
                            subset_true
                        )
                    ),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def _plot_prediction_panel(
    axis,
    prediction_table: pd.DataFrame,
    title: str,
    color: str,
) -> None:
    """Draw one z_spec versus prediction panel."""

    inlier = ~prediction_table[
        CATASTROPHIC_COLUMN
    ].astype(bool)

    outlier = prediction_table[
        CATASTROPHIC_COLUMN
    ].astype(bool)

    axis.scatter(
        prediction_table.loc[
            inlier,
            TARGET_COLUMN,
        ],
        prediction_table.loc[
            inlier,
            "z_prediction",
        ],
        s=18,
        alpha=0.65,
        color=color,
        edgecolors="none",
        label="Non-outlier",
    )

    axis.scatter(
        prediction_table.loc[
            outlier,
            TARGET_COLUMN,
        ],
        prediction_table.loc[
            outlier,
            "z_prediction",
        ],
        s=40,
        alpha=0.90,
        color="crimson",
        marker="x",
        label="Catastrophic outlier",
    )

    maximum_value = max(
        float(
            prediction_table[
                TARGET_COLUMN
            ].max()
        ),
        float(
            prediction_table[
                "z_prediction"
            ].max()
        ),
    )

    plotting_limit = min(
        max(10.0, maximum_value * 1.03),
        20.0,
    )

    axis.plot(
        [0.0, plotting_limit],
        [0.0, plotting_limit],
        linestyle="--",
        linewidth=1.2,
        color="black",
        label="1:1 relation",
    )

    axis.set_xlim(0.0, plotting_limit)
    axis.set_ylim(0.0, plotting_limit)
    axis.set_xlabel(
        "Spectroscopic redshift"
    )
    axis.set_ylabel(
        "Predicted redshift"
    )
    axis.set_title(title)
    axis.grid(
        alpha=0.20
    )
    axis.legend(
        fontsize=8,
    )


def build_final_test_figure(
    prediction_table: pd.DataFrame,
    redshift_metrics: pd.DataFrame,
):
    """Build one scientific comparison figure for the final notebook."""

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        constrained_layout=True,
    )

    eazy_predictions = prediction_table.loc[
        prediction_table["experiment_id"]
        == "eazy_dr5_za"
    ]

    gated_predictions = prediction_table.loc[
        prediction_table["experiment_id"]
        == "xgb_eazy_gated_residual"
    ]

    _plot_prediction_panel(
        axes[0, 0],
        eazy_predictions,
        title="EAZY DR5",
        color="tab:blue",
    )

    _plot_prediction_panel(
        axes[0, 1],
        gated_predictions,
        title="Gated residual hybrid",
        color="tab:green",
    )

    colors = {
        "EAZY DR5 z_a": "tab:blue",
        "XGBoost: gated residual hybrid": (
            "tab:green"
        ),
    }

    for method, method_table in (
        redshift_metrics.groupby(
            "method",
            sort=False,
        )
    ):
        method_table = method_table.sort_values(
            "redshift_stratum"
        )

        axes[1, 0].plot(
            method_table[
                "median_z_spec"
            ],
            method_table[
                "sigma_nmad"
            ],
            marker="o",
            linewidth=2,
            color=colors[method],
            label=method,
        )

        axes[1, 1].plot(
            method_table[
                "median_z_spec"
            ],
            method_table[
                "catastrophic_outlier_fraction"
            ],
            marker="o",
            linewidth=2,
            color=colors[method],
            label=method,
        )

    axes[1, 0].set_xlabel(
        "Median spectroscopic redshift in bin"
    )
    axes[1, 0].set_ylabel(
        r"$\sigma_{\mathrm{NMAD}}$"
    )
    axes[1, 0].set_title(
        "Robust scatter by redshift"
    )
    axes[1, 0].grid(
        alpha=0.20
    )
    axes[1, 0].legend(
        fontsize=8
    )

    axes[1, 1].set_xlabel(
        "Median spectroscopic redshift in bin"
    )
    axes[1, 1].set_ylabel(
        "Catastrophic outlier fraction"
    )
    axes[1, 1].set_title(
        "Outlier rate by redshift"
    )
    axes[1, 1].grid(
        alpha=0.20
    )
    axes[1, 1].legend(
        fontsize=8
    )

    figure.suptitle(
        "JADES DR4-DR5 blind-test Photo-z evaluation",
        fontsize=15,
    )

    figure.savefig(
        FINAL_TEST_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    return figure


def save_final_test_products(
    metrics_table: pd.DataFrame,
    redshift_metrics: pd.DataFrame,
    prediction_table: pd.DataFrame,
) -> None:
    """Save the frozen blind-test outputs."""

    ensure_output_directories()

    metrics_table.to_csv(
        FINAL_TEST_METRICS_PATH,
        index=False,
    )

    redshift_metrics.to_csv(
        FINAL_TEST_REDSHIFT_METRICS_PATH,
        index=False,
    )

    prediction_table.to_csv(
        FINAL_TEST_PREDICTIONS_PATH,
        index=False,
    )


def run_final_test_evaluation():
    """Evaluate all pre-specified frozen models on the test split once."""

    ensure_output_directories()

    test_catalog = load_test_catalog()

    eazy_prediction = test_catalog[
        EAZY_HYBRID_FEATURE
    ].to_numpy(dtype=float)

    pure_xgboost_prediction = (
        predict_standard_xgboost(
            PURE_XGBOOST_MODEL_PATH,
            test_catalog,
            FULL_FEATURE_COLUMNS,
        )
    )

    direct_hybrid_prediction = (
        predict_standard_xgboost(
            DIRECT_HYBRID_MODEL_PATH,
            test_catalog,
            HYBRID_FEATURE_COLUMNS,
        )
    )

    (
        gated_hybrid_prediction,
        correction_applied,
        gate_probability,
    ) = predict_gated_residual_hybrid(
        test_catalog
    )

    method_definitions = (
        {
            "experiment_id": "eazy_dr5_za",
            "method": "EAZY DR5 z_a",
            "prediction": eazy_prediction,
            "pre_test_role": (
                "Published template-fitting benchmark"
            ),
        },
        {
            "experiment_id": (
                "xgb_full_weighted"
            ),
            "method": (
                "XGBoost: full + weights"
            ),
            "prediction": (
                pure_xgboost_prediction
            ),
            "pre_test_role": (
                "Best pure-ML validation configuration"
            ),
        },
        {
            "experiment_id": (
                "xgb_eazy_direct_hybrid"
            ),
            "method": (
                "XGBoost: direct EAZY hybrid"
            ),
            "prediction": (
                direct_hybrid_prediction
            ),
            "pre_test_role": (
                "Direct-hybrid ablation"
            ),
        },
        {
            "experiment_id": (
                "xgb_eazy_gated_residual"
            ),
            "method": (
                "XGBoost: gated residual hybrid"
            ),
            "prediction": (
                gated_hybrid_prediction
            ),
            "pre_test_role": (
                "Selected before opening test"
            ),
            "correction_applied": (
                correction_applied
            ),
            "gate_probability": (
                gate_probability
            ),
        },
    )

    all_metrics = []
    all_predictions = []

    for method_definition in method_definitions:
        metrics, predictions = (
            summarize_test_method(
                test_catalog,
                **method_definition,
            )
        )

        all_metrics.append(metrics)
        all_predictions.append(predictions)

    metrics_table = pd.DataFrame(
        all_metrics
    )

    prediction_table = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    redshift_metrics = build_redshift_metrics(
        test_catalog,
        {
            "EAZY DR5 z_a": eazy_prediction,
            (
                "XGBoost: gated residual hybrid"
            ): gated_hybrid_prediction,
        },
    )

    save_final_test_products(
        metrics_table,
        redshift_metrics,
        prediction_table,
    )

    final_figure = build_final_test_figure(
        prediction_table,
        redshift_metrics,
    )

    return (
        metrics_table,
        redshift_metrics,
        prediction_table,
        final_figure,
    )