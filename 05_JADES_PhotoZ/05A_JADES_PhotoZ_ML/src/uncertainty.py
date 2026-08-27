"""Quantify paired uncertainty in the frozen blind-test results."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .config import (
    FIGURE_DIR,
    METRICS_DIR,
    RANDOM_SEED,
    SOURCE_KEY_COLUMN,
    TARGET_COLUMN,
    ensure_output_directories,
)
from .test import (
    FINAL_TEST_PREDICTIONS_PATH,
)
from .xgboost import (
    CATASTROPHIC_OUTLIER_THRESHOLD,
)


BOOTSTRAP_REPLICATES = 10_000
CONFIDENCE_LEVEL = 0.95

EAZY_EXPERIMENT_ID = "eazy_dr5_za"
HYBRID_EXPERIMENT_ID = (
    "xgb_eazy_gated_residual"
)

BOOTSTRAP_SUMMARY_PATH = (
    METRICS_DIR
    / "dr45_paired_bootstrap_summary.csv"
)

BOOTSTRAP_DRAWS_PATH = (
    METRICS_DIR
    / "dr45_paired_bootstrap_draws.csv"
)

OUTLIER_COMPARISON_PATH = (
    METRICS_DIR
    / "dr45_paired_outlier_comparison.csv"
)

UNCERTAINTY_MANIFEST_PATH = (
    METRICS_DIR
    / "dr45_uncertainty_manifest.json"
)

UNCERTAINTY_FIGURE_PATH = (
    FIGURE_DIR
    / "dr45_paired_test_uncertainty.png"
)


METRIC_LABELS = {
    "absolute_normalized_bias": (
        "Absolute normalized bias"
    ),
    "normalized_median_absolute_error": (
        "Median absolute normalized error"
    ),
    "sigma_nmad": (
        "Sigma NMAD"
    ),
    "mean_absolute_redshift_error": (
        "Mean absolute redshift error"
    ),
    "catastrophic_outlier_fraction": (
        "Catastrophic outlier fraction"
    ),
}


def load_paired_predictions() -> pd.DataFrame:
    """Load and align EAZY and hybrid predictions by physical source."""

    if not FINAL_TEST_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "The frozen test predictions were not found at "
            f"{FINAL_TEST_PREDICTIONS_PATH}"
        )

    predictions = pd.read_csv(
        FINAL_TEST_PREDICTIONS_PATH,
        low_memory=False,
    )

    required_columns = {
        SOURCE_KEY_COLUMN,
        TARGET_COLUMN,
        "experiment_id",
        "z_prediction",
    }

    missing_columns = sorted(
        required_columns - set(predictions.columns)
    )

    if missing_columns:
        raise KeyError(
            "The frozen prediction table is missing columns: "
            f"{missing_columns}"
        )

    eazy = predictions.loc[
        predictions["experiment_id"]
        == EAZY_EXPERIMENT_ID,
        [
            SOURCE_KEY_COLUMN,
            TARGET_COLUMN,
            "z_prediction",
        ],
    ].copy()

    hybrid = predictions.loc[
        predictions["experiment_id"]
        == HYBRID_EXPERIMENT_ID,
        [
            SOURCE_KEY_COLUMN,
            TARGET_COLUMN,
            "z_prediction",
        ],
    ].copy()

    eazy = eazy.rename(
        columns={
            TARGET_COLUMN: "z_spec_eazy",
            "z_prediction": "z_eazy",
        }
    )

    hybrid = hybrid.rename(
        columns={
            TARGET_COLUMN: "z_spec_hybrid",
            "z_prediction": "z_hybrid",
        }
    )

    if not eazy[SOURCE_KEY_COLUMN].is_unique:
        raise ValueError(
            "EAZY test predictions contain duplicated sources."
        )

    if not hybrid[SOURCE_KEY_COLUMN].is_unique:
        raise ValueError(
            "Hybrid test predictions contain duplicated sources."
        )

    paired = eazy.merge(
        hybrid,
        on=SOURCE_KEY_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    if len(paired) != len(eazy):
        raise ValueError(
            "Some EAZY sources have no aligned hybrid prediction."
        )

    if len(paired) != len(hybrid):
        raise ValueError(
            "Some hybrid sources have no aligned EAZY prediction."
        )

    if not np.allclose(
        paired["z_spec_eazy"],
        paired["z_spec_hybrid"],
    ):
        raise ValueError(
            "Aligned predictions do not have identical z_spec."
        )

    paired = paired.rename(
        columns={
            "z_spec_eazy": TARGET_COLUMN,
        }
    ).drop(
        columns=["z_spec_hybrid"]
    )

    numeric_columns = (
        TARGET_COLUMN,
        "z_eazy",
        "z_hybrid",
    )

    for column in numeric_columns:
        paired[column] = pd.to_numeric(
            paired[column],
            errors="coerce",
        )

    if not np.isfinite(
        paired.loc[
            :,
            list(numeric_columns),
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "The paired predictions contain non-finite values."
        )

    return paired.sort_values(
        SOURCE_KEY_COLUMN
    ).reset_index(drop=True)


def calculate_metric_values(
    true_redshift,
    predicted_redshift,
) -> dict[str, float]:
    """Calculate lower-is-better Photo-z metrics for one sample."""

    true_values = np.asarray(
        true_redshift,
        dtype=float,
    )

    predicted_values = np.asarray(
        predicted_redshift,
        dtype=float,
    )

    normalized_error = (
        predicted_values - true_values
    ) / (1.0 + true_values)

    normalized_bias = float(
        np.median(normalized_error)
    )

    catastrophic = (
        np.abs(normalized_error)
        > CATASTROPHIC_OUTLIER_THRESHOLD
    )

    return {
        "absolute_normalized_bias": abs(
            normalized_bias
        ),
        "normalized_median_absolute_error": float(
            np.median(
                np.abs(normalized_error)
            )
        ),
        "sigma_nmad": float(
            1.4826
            * np.median(
                np.abs(
                    normalized_error
                    - normalized_bias
                )
            )
        ),
        "mean_absolute_redshift_error": float(
            np.mean(
                np.abs(
                    predicted_values
                    - true_values
                )
            )
        ),
        "catastrophic_outlier_fraction": float(
            catastrophic.mean()
        ),
    }


def run_paired_bootstrap(
    paired_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap paired metric differences using identical source draws."""

    random_generator = np.random.default_rng(
        RANDOM_SEED
    )

    true_redshift = paired_predictions[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

    eazy_prediction = paired_predictions[
        "z_eazy"
    ].to_numpy(dtype=float)

    hybrid_prediction = paired_predictions[
        "z_hybrid"
    ].to_numpy(dtype=float)

    number_of_sources = len(
        paired_predictions
    )

    metric_names = tuple(
        METRIC_LABELS
    )

    delta_draws = {
        metric_name: np.empty(
            BOOTSTRAP_REPLICATES,
            dtype=float,
        )
        for metric_name in metric_names
    }

    relative_improvement_draws = {
        metric_name: np.empty(
            BOOTSTRAP_REPLICATES,
            dtype=float,
        )
        for metric_name in metric_names
    }

    for replicate in range(
        BOOTSTRAP_REPLICATES
    ):
        sampled_indices = (
            random_generator.integers(
                low=0,
                high=number_of_sources,
                size=number_of_sources,
            )
        )

        sampled_true = true_redshift[
            sampled_indices
        ]

        sampled_eazy = eazy_prediction[
            sampled_indices
        ]

        sampled_hybrid = hybrid_prediction[
            sampled_indices
        ]

        eazy_metrics = calculate_metric_values(
            sampled_true,
            sampled_eazy,
        )

        hybrid_metrics = calculate_metric_values(
            sampled_true,
            sampled_hybrid,
        )

        for metric_name in metric_names:
            eazy_value = eazy_metrics[
                metric_name
            ]

            hybrid_value = hybrid_metrics[
                metric_name
            ]

            delta_draws[metric_name][
                replicate
            ] = (
                hybrid_value - eazy_value
            )

            if eazy_value > 0:
                relative_improvement = (
                    100.0
                    * (
                        eazy_value
                        - hybrid_value
                    )
                    / eazy_value
                )
            else:
                relative_improvement = np.nan

            relative_improvement_draws[
                metric_name
            ][replicate] = (
                relative_improvement
            )

    alpha = 1.0 - CONFIDENCE_LEVEL

    lower_percentile = (
        100.0 * alpha / 2.0
    )

    upper_percentile = (
        100.0 * (1.0 - alpha / 2.0)
    )

    point_eazy_metrics = (
        calculate_metric_values(
            true_redshift,
            eazy_prediction,
        )
    )

    point_hybrid_metrics = (
        calculate_metric_values(
            true_redshift,
            hybrid_prediction,
        )
    )

    summary_rows = []

    draw_rows = []

    for metric_name in metric_names:
        metric_delta_draws = (
            delta_draws[metric_name]
        )

        metric_relative_draws = (
            relative_improvement_draws[
                metric_name
            ]
        )

        finite_relative_draws = (
            metric_relative_draws[
                np.isfinite(
                    metric_relative_draws
                )
            ]
        )

        eazy_point = point_eazy_metrics[
            metric_name
        ]

        hybrid_point = point_hybrid_metrics[
            metric_name
        ]

        point_delta = (
            hybrid_point - eazy_point
        )

        point_relative_improvement = (
            100.0
            * (
                eazy_point - hybrid_point
            )
            / eazy_point
        )

        summary_rows.append(
            {
                "metric": metric_name,
                "metric_label": (
                    METRIC_LABELS[
                        metric_name
                    ]
                ),
                "eazy_point": eazy_point,
                "hybrid_point": hybrid_point,
                "hybrid_minus_eazy": (
                    point_delta
                ),
                "delta_ci_lower_95": float(
                    np.percentile(
                        metric_delta_draws,
                        lower_percentile,
                    )
                ),
                "delta_ci_upper_95": float(
                    np.percentile(
                        metric_delta_draws,
                        upper_percentile,
                    )
                ),
                "relative_improvement_percent": (
                    point_relative_improvement
                ),
                (
                    "relative_improvement_ci_lower_95"
                ): float(
                    np.percentile(
                        finite_relative_draws,
                        lower_percentile,
                    )
                ),
                (
                    "relative_improvement_ci_upper_95"
                ): float(
                    np.percentile(
                        finite_relative_draws,
                        upper_percentile,
                    )
                ),
                (
                    "bootstrap_probability_hybrid_better"
                ): float(
                    np.mean(
                        metric_delta_draws < 0
                    )
                ),
            }
        )

        for replicate in range(
            BOOTSTRAP_REPLICATES
        ):
            draw_rows.append(
                {
                    "replicate": replicate,
                    "metric": metric_name,
                    "hybrid_minus_eazy": (
                        metric_delta_draws[
                            replicate
                        ]
                    ),
                    (
                        "relative_improvement_percent"
                    ): metric_relative_draws[
                        replicate
                    ],
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    draws = pd.DataFrame(
        draw_rows
    )

    return summary, draws


def run_exact_outlier_comparison(
    paired_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compare paired catastrophic-outlier decisions exactly."""

    true_redshift = paired_predictions[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

    eazy_prediction = paired_predictions[
        "z_eazy"
    ].to_numpy(dtype=float)

    hybrid_prediction = paired_predictions[
        "z_hybrid"
    ].to_numpy(dtype=float)

    eazy_normalized_error = (
        eazy_prediction - true_redshift
    ) / (1.0 + true_redshift)

    hybrid_normalized_error = (
        hybrid_prediction - true_redshift
    ) / (1.0 + true_redshift)

    eazy_outlier = (
        np.abs(eazy_normalized_error)
        > CATASTROPHIC_OUTLIER_THRESHOLD
    )

    hybrid_outlier = (
        np.abs(hybrid_normalized_error)
        > CATASTROPHIC_OUTLIER_THRESHOLD
    )

    both_inlier = int(
        (
            ~eazy_outlier
            & ~hybrid_outlier
        ).sum()
    )

    eazy_only_outlier = int(
        (
            eazy_outlier
            & ~hybrid_outlier
        ).sum()
    )

    hybrid_only_outlier = int(
        (
            ~eazy_outlier
            & hybrid_outlier
        ).sum()
    )

    both_outlier = int(
        (
            eazy_outlier
            & hybrid_outlier
        ).sum()
    )

    discordant_pairs = (
        eazy_only_outlier
        + hybrid_only_outlier
    )

    if discordant_pairs == 0:
        exact_p_value = 1.0
    else:
        exact_p_value = float(
            binomtest(
                k=eazy_only_outlier,
                n=discordant_pairs,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    if hybrid_only_outlier == 0:
        fixed_to_introduced_ratio = np.inf
    else:
        fixed_to_introduced_ratio = (
            eazy_only_outlier
            / hybrid_only_outlier
        )

    return pd.DataFrame(
        [
            {
                "sources": len(
                    paired_predictions
                ),
                "both_non_outliers": (
                    both_inlier
                ),
                (
                    "eazy_outlier_hybrid_non_outlier"
                ): eazy_only_outlier,
                (
                    "eazy_non_outlier_hybrid_outlier"
                ): hybrid_only_outlier,
                "both_outliers": both_outlier,
                "discordant_pairs": (
                    discordant_pairs
                ),
                "fixed_to_introduced_ratio": (
                    fixed_to_introduced_ratio
                ),
                "exact_mcnemar_p_value": (
                    exact_p_value
                ),
            }
        ]
    )


def build_uncertainty_figure(
    bootstrap_summary: pd.DataFrame,
    outlier_comparison: pd.DataFrame,
):
    """Build one paired-uncertainty figure for the case study."""

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
        constrained_layout=True,
    )

    plotting_table = (
        bootstrap_summary.iloc[
            ::-1
        ].reset_index(drop=True)
    )

    y_positions = np.arange(
        len(plotting_table)
    )

    point_values = plotting_table[
        "relative_improvement_percent"
    ].to_numpy(dtype=float)

    lower_values = plotting_table[
        "relative_improvement_ci_lower_95"
    ].to_numpy(dtype=float)

    upper_values = plotting_table[
        "relative_improvement_ci_upper_95"
    ].to_numpy(dtype=float)

    lower_errors = (
        point_values - lower_values
    )

    upper_errors = (
        upper_values - point_values
    )

    point_colors = np.where(
        point_values >= 0,
        "tab:green",
        "tab:orange",
    )

    for position in y_positions:
        axes[0].errorbar(
            point_values[position],
            position,
            xerr=np.array(
                [
                    [
                        lower_errors[
                            position
                        ]
                    ],
                    [
                        upper_errors[
                            position
                        ]
                    ],
                ]
            ),
            fmt="o",
            color=point_colors[position],
            capsize=4,
            markersize=7,
        )

    axes[0].axvline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
    )

    axes[0].set_yticks(
        y_positions
    )

    axes[0].set_yticklabels(
        plotting_table[
            "metric_label"
        ]
    )

    axes[0].set_xlabel(
        "Relative improvement of hybrid over EAZY (%)"
    )

    axes[0].set_title(
        "Paired bootstrap 95% confidence intervals"
    )

    axes[0].grid(
        axis="x",
        alpha=0.20,
    )

    outlier_row = (
        outlier_comparison.iloc[0]
    )

    outlier_labels = (
        "EAZY outlier\nfixed by hybrid",
        "New outlier\nintroduced by hybrid",
        "Outlier in\nboth methods",
    )

    outlier_counts = (
        int(
            outlier_row[
                "eazy_outlier_hybrid_non_outlier"
            ]
        ),
        int(
            outlier_row[
                "eazy_non_outlier_hybrid_outlier"
            ]
        ),
        int(
            outlier_row[
                "both_outliers"
            ]
        ),
    )

    bars = axes[1].bar(
        outlier_labels,
        outlier_counts,
        color=(
            "tab:green",
            "tab:orange",
            "tab:red",
        ),
    )

    axes[1].bar_label(
        bars,
        padding=3,
    )

    axes[1].set_ylabel(
        "Number of blind-test sources"
    )

    axes[1].set_title(
        "Paired catastrophic-outlier changes\n"
        f"Exact McNemar p = "
        f"{outlier_row['exact_mcnemar_p_value']:.3f}"
    )

    axes[1].grid(
        axis="y",
        alpha=0.20,
    )

    figure.suptitle(
        "JADES DR4-DR5 blind-test result uncertainty",
        fontsize=15,
    )

    figure.savefig(
        UNCERTAINTY_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return figure


def save_uncertainty_products(
    bootstrap_summary: pd.DataFrame,
    bootstrap_draws: pd.DataFrame,
    outlier_comparison: pd.DataFrame,
) -> None:
    """Save uncertainty tables and reproducibility metadata."""

    ensure_output_directories()

    bootstrap_summary.to_csv(
        BOOTSTRAP_SUMMARY_PATH,
        index=False,
    )

    bootstrap_draws.to_csv(
        BOOTSTRAP_DRAWS_PATH,
        index=False,
    )

    outlier_comparison.to_csv(
        OUTLIER_COMPARISON_PATH,
        index=False,
    )

    manifest = {
        "bootstrap_replicates": (
            BOOTSTRAP_REPLICATES
        ),
        "confidence_level": CONFIDENCE_LEVEL,
        "random_seed": RANDOM_SEED,
        "comparison": (
            "Paired gated residual hybrid minus EAZY"
        ),
        "negative_delta_favors": (
            "Gated residual hybrid"
        ),
        "outlier_threshold": (
            CATASTROPHIC_OUTLIER_THRESHOLD
        ),
    }

    with UNCERTAINTY_MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            manifest,
            file_handle,
            indent=2,
        )


def run_uncertainty_analysis(
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Run paired bootstrap and exact outlier comparison."""

    paired_predictions = (
        load_paired_predictions()
    )

    (
        bootstrap_summary,
        bootstrap_draws,
    ) = run_paired_bootstrap(
        paired_predictions
    )

    outlier_comparison = (
        run_exact_outlier_comparison(
            paired_predictions
        )
    )

    save_uncertainty_products(
        bootstrap_summary,
        bootstrap_draws,
        outlier_comparison,
    )

    uncertainty_figure = (
        build_uncertainty_figure(
            bootstrap_summary,
            outlier_comparison,
        )
    )

    return (
        bootstrap_summary,
        outlier_comparison,
        uncertainty_figure,
    )