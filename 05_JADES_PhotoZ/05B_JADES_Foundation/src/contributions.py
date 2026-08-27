# -*- coding: utf-8 -*-

"""Visualize the contributions of EAZY and TabPFN on the blind test."""

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
    ensure_output_directories,
)


BLIND_TEST_PREDICTIONS_PATH = (
    PREDICTION_DIR
    / "tabpfn_blind_test_predictions.csv"
)

CONTRIBUTION_SUMMARY_PATH = (
    METRICS_DIR
    / "tabpfn_contribution_summary.csv"
)

CONTRIBUTION_CASES_PATH = (
    METRICS_DIR
    / "tabpfn_contribution_cases.csv"
)

CONTRIBUTION_CATALOG_PATH = (
    PREDICTION_DIR
    / "tabpfn_contribution_catalog.csv"
)

CONTRIBUTION_FIGURE_PATH = (
    FIGURE_DIR
    / "tabpfn_eazy_contributions.png"
)

CONTRIBUTION_MANIFEST_PATH = (
    METRICS_DIR
    / "tabpfn_contribution_manifest.json"
)


REDSHIFT_BIN_EDGES = (
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
    np.inf,
)

REDSHIFT_BIN_LABELS = (
    "0 <= z < 1",
    "1 <= z < 2",
    "2 <= z < 3",
    "3 <= z < 4",
    "4 <= z < 6",
    "6 <= z < 8",
    "z >= 8",
)


def load_blind_test_predictions() -> pd.DataFrame:
    """Load and validate the frozen blind-test predictions."""

    if not BLIND_TEST_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "The blind-test predictions were not found: "
            f"{BLIND_TEST_PREDICTIONS_PATH}"
        )

    predictions = pd.read_csv(
        BLIND_TEST_PREDICTIONS_PATH,
        low_memory=False,
    )

    required_columns = {
        SOURCE_KEY_COLUMN,
        "z_spec",
        "z_eazy",
        "z_tabpfn_eazy",
    }

    missing_columns = sorted(
        required_columns - set(predictions.columns)
    )

    if missing_columns:
        raise KeyError(
            "The blind-test predictions are missing: "
            f"{missing_columns}"
        )

    if len(predictions) != 476:
        raise ValueError(
            f"Expected 476 blind-test sources, "
            f"found {len(predictions)}."
        )

    if not predictions[SOURCE_KEY_COLUMN].is_unique:
        raise ValueError(
            "The blind-test predictions contain duplicated sources."
        )

    numeric_columns = (
        "z_spec",
        "z_eazy",
        "z_tabpfn_eazy",
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
                f"{column} contains non-finite values."
            )

    return predictions


def build_contribution_catalog(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the applied TabPFN correction for every source."""

    result = predictions.loc[
        :,
        [
            SOURCE_KEY_COLUMN,
            "z_spec",
            "z_eazy",
            "z_tabpfn_eazy",
        ],
    ].copy()

    result["true_log_residual"] = (
        np.log1p(result["z_spec"])
        - np.log1p(result["z_eazy"])
    )

    result["applied_tabpfn_log_correction"] = (
        np.log1p(result["z_tabpfn_eazy"])
        - np.log1p(result["z_eazy"])
    )

    result["redshift_correction"] = (
        result["z_tabpfn_eazy"]
        - result["z_eazy"]
    )

    result["eazy_normalized_error"] = (
        result["z_eazy"]
        - result["z_spec"]
    ) / (
        1.0 + result["z_spec"]
    )

    result["hybrid_normalized_error"] = (
        result["z_tabpfn_eazy"]
        - result["z_spec"]
    ) / (
        1.0 + result["z_spec"]
    )

    result["eazy_absolute_normalized_error"] = (
        np.abs(
            result["eazy_normalized_error"]
        )
    )

    result["hybrid_absolute_normalized_error"] = (
        np.abs(
            result["hybrid_normalized_error"]
        )
    )

    result["normalized_error_improvement"] = (
        result["eazy_absolute_normalized_error"]
        - result["hybrid_absolute_normalized_error"]
    )

    result["eazy_is_catastrophic"] = (
        result["eazy_absolute_normalized_error"]
        > CATASTROPHIC_OUTLIER_THRESHOLD
    )

    result["hybrid_is_catastrophic"] = (
        result["hybrid_absolute_normalized_error"]
        > CATASTROPHIC_OUTLIER_THRESHOLD
    )

    result["correction_direction_is_correct"] = (
        np.sign(
            result["applied_tabpfn_log_correction"]
        )
        == np.sign(
            result["true_log_residual"]
        )
    )

    transition_conditions = (
        (
            result["eazy_is_catastrophic"]
            & ~result["hybrid_is_catastrophic"]
        ),
        (
            ~result["eazy_is_catastrophic"]
            & result["hybrid_is_catastrophic"]
        ),
        (
            result["eazy_is_catastrophic"]
            & result["hybrid_is_catastrophic"]
        ),
    )

    transition_labels = (
        "Fixed EAZY outlier",
        "New hybrid outlier",
        "Shared outlier",
    )

    result["outlier_transition"] = np.select(
        transition_conditions,
        transition_labels,
        default="Non-outlier in both",
    )

    return result


def build_contribution_summary(
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the global effect of the TabPFN correction."""

    improved = (
        catalog["normalized_error_improvement"]
        > 0.0
    )

    worsened = (
        catalog["normalized_error_improvement"]
        < 0.0
    )

    fixed_outliers = catalog[
        "outlier_transition"
    ].eq("Fixed EAZY outlier")

    new_outliers = catalog[
        "outlier_transition"
    ].eq("New hybrid outlier")

    shared_outliers = catalog[
        "outlier_transition"
    ].eq("Shared outlier")

    usable_direction = (
        np.abs(catalog["true_log_residual"])
        > 1e-8
    )

    direction_agreement = float(
        catalog.loc[
            usable_direction,
            "correction_direction_is_correct",
        ].mean()
    )

    residual_correlation = float(
        np.corrcoef(
            catalog["true_log_residual"],
            catalog[
                "applied_tabpfn_log_correction"
            ],
        )[0, 1]
    )

    rows = (
        (
            "Blind-test sources",
            float(len(catalog)),
        ),
        (
            "Sources improved by TabPFN",
            float(improved.sum()),
        ),
        (
            "Fraction improved by TabPFN",
            float(improved.mean()),
        ),
        (
            "Sources worsened by TabPFN",
            float(worsened.sum()),
        ),
        (
            "Fraction worsened by TabPFN",
            float(worsened.mean()),
        ),
        (
            "EAZY outliers fixed by TabPFN",
            float(fixed_outliers.sum()),
        ),
        (
            "New outliers introduced by TabPFN",
            float(new_outliers.sum()),
        ),
        (
            "Outliers shared by both methods",
            float(shared_outliers.sum()),
        ),
        (
            "Correction-direction agreement",
            direction_agreement,
        ),
        (
            "True-versus-applied residual correlation",
            residual_correlation,
        ),
        (
            "Median absolute redshift correction",
            float(
                np.median(
                    np.abs(
                        catalog[
                            "redshift_correction"
                        ]
                    )
                )
            ),
        ),
        (
            "Median normalized-error improvement",
            float(
                np.median(
                    catalog[
                        "normalized_error_improvement"
                    ]
                )
            ),
        ),
    )

    return pd.DataFrame(
        rows,
        columns=(
            "metric",
            "value",
        ),
    )


def select_explanation_cases(
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Select representative corrected and harmed sources."""

    selected_groups = []
    used_sources = set()

    def add_group(
        mask,
        group_name: str,
        sort_column: str,
        ascending: bool,
        maximum_cases: int,
    ) -> None:
        candidates = catalog.loc[
            mask
            & ~catalog[SOURCE_KEY_COLUMN].isin(
                used_sources
            )
        ].sort_values(
            sort_column,
            ascending=ascending,
        )

        selected = candidates.head(
            maximum_cases
        ).copy()

        if selected.empty:
            return

        selected.insert(
            0,
            "case_group",
            group_name,
        )

        used_sources.update(
            selected[SOURCE_KEY_COLUMN]
        )

        selected_groups.append(selected)

    add_group(
        mask=catalog["outlier_transition"].eq(
            "Fixed EAZY outlier"
        ),
        group_name="Fixed EAZY outlier",
        sort_column="normalized_error_improvement",
        ascending=False,
        maximum_cases=2,
    )

    add_group(
        mask=catalog["outlier_transition"].eq(
            "Shared outlier"
        ),
        group_name="Shared outlier",
        sort_column=(
            "hybrid_absolute_normalized_error"
        ),
        ascending=False,
        maximum_cases=2,
    )

    add_group(
        mask=catalog["outlier_transition"].eq(
            "New hybrid outlier"
        ),
        group_name="New hybrid outlier",
        sort_column="normalized_error_improvement",
        ascending=True,
        maximum_cases=2,
    )

    both_non_outliers = catalog[
        "outlier_transition"
    ].eq("Non-outlier in both")

    add_group(
        mask=(
            both_non_outliers
            & (
                catalog[
                    "normalized_error_improvement"
                ]
                > 0.0
            )
        ),
        group_name="Largest ordinary improvement",
        sort_column="normalized_error_improvement",
        ascending=False,
        maximum_cases=2,
    )

    add_group(
        mask=(
            both_non_outliers
            & (
                catalog[
                    "normalized_error_improvement"
                ]
                < 0.0
            )
        ),
        group_name="Largest ordinary harm",
        sort_column="normalized_error_improvement",
        ascending=True,
        maximum_cases=2,
    )

    if not selected_groups:
        raise ValueError(
            "No representative contribution cases were selected."
        )

    selected_cases = pd.concat(
        selected_groups,
        axis=0,
        ignore_index=True,
    )

    output_columns = (
        "case_group",
        SOURCE_KEY_COLUMN,
        "z_spec",
        "z_eazy",
        "z_tabpfn_eazy",
        "redshift_correction",
        "eazy_absolute_normalized_error",
        "hybrid_absolute_normalized_error",
        "normalized_error_improvement",
        "outlier_transition",
    )

    return selected_cases.loc[
        :,
        list(output_columns),
    ]


def build_redshift_bin_summary(
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate EAZY and hybrid errors in redshift bins."""

    working = catalog.copy()

    working["z_spec_bin"] = pd.cut(
        working["z_spec"],
        bins=REDSHIFT_BIN_EDGES,
        labels=REDSHIFT_BIN_LABELS,
        right=False,
        include_lowest=True,
    )

    rows = []

    for redshift_bin, group in working.groupby(
        "z_spec_bin",
        observed=True,
    ):
        rows.append(
            {
                "z_spec_bin": str(redshift_bin),
                "sources": len(group),
                "median_z_spec": float(
                    group["z_spec"].median()
                ),
                "eazy_median_absolute_normalized_error": float(
                    group[
                        "eazy_absolute_normalized_error"
                    ].median()
                ),
                "hybrid_median_absolute_normalized_error": float(
                    group[
                        "hybrid_absolute_normalized_error"
                    ].median()
                ),
                "eazy_outlier_fraction": float(
                    group[
                        "eazy_is_catastrophic"
                    ].mean()
                ),
                "hybrid_outlier_fraction": float(
                    group[
                        "hybrid_is_catastrophic"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def draw_residual_panel(
    axis,
    catalog: pd.DataFrame,
) -> None:
    """Show whether TabPFN applies the required EAZY correction."""

    improved = (
        catalog["normalized_error_improvement"]
        > 0.0
    )

    axis.scatter(
        catalog.loc[
            improved,
            "true_log_residual",
        ],
        catalog.loc[
            improved,
            "applied_tabpfn_log_correction",
        ],
        s=18,
        alpha=0.6,
        color="#2ca02c",
        label="Improved source",
    )

    axis.scatter(
        catalog.loc[
            ~improved,
            "true_log_residual",
        ],
        catalog.loc[
            ~improved,
            "applied_tabpfn_log_correction",
        ],
        s=18,
        alpha=0.6,
        color="#ff7f0e",
        label="Worsened source",
    )

    maximum = float(
        np.nanmax(
            np.abs(
                np.concatenate(
                    [
                        catalog[
                            "true_log_residual"
                        ].to_numpy(),
                        catalog[
                            "applied_tabpfn_log_correction"
                        ].to_numpy(),
                    ]
                )
            )
        )
    )

    axis.plot(
        [-maximum, maximum],
        [-maximum, maximum],
        linestyle="--",
        color="black",
        linewidth=1.0,
        label="Ideal correction",
    )

    axis.axhline(
        0.0,
        color="gray",
        linewidth=0.8,
    )

    axis.axvline(
        0.0,
        color="gray",
        linewidth=0.8,
    )

    axis.set_xlim(
        -maximum,
        maximum,
    )

    axis.set_ylim(
        -maximum,
        maximum,
    )

    axis.set_xlabel(
        "Required EAZY log-residual correction"
    )

    axis.set_ylabel(
        "Applied TabPFN log correction"
    )

    axis.set_title(
        "Residual contribution of TabPFN"
    )

    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)


def draw_error_comparison_panel(
    axis,
    catalog: pd.DataFrame,
) -> None:
    """Compare per-source EAZY and hybrid errors."""

    category_styles = (
        (
            "Fixed EAZY outlier",
            "#2ca02c",
            "o",
        ),
        (
            "New hybrid outlier",
            "#d62728",
            "x",
        ),
        (
            "Shared outlier",
            "black",
            "s",
        ),
        (
            "Non-outlier in both",
            "#9467bd",
            "o",
        ),
    )

    minimum_error = 1e-4

    eazy_error = np.maximum(
        catalog[
            "eazy_absolute_normalized_error"
        ].to_numpy(),
        minimum_error,
    )

    hybrid_error = np.maximum(
        catalog[
            "hybrid_absolute_normalized_error"
        ].to_numpy(),
        minimum_error,
    )

    maximum_error = float(
        np.nanmax(
            np.concatenate(
                [eazy_error, hybrid_error]
            )
        )
    )

    plot_limit = 10 ** np.ceil(
        np.log10(
            max(
                maximum_error,
                CATASTROPHIC_OUTLIER_THRESHOLD,
            )
        )
    )

    for (
        category,
        color,
        marker,
    ) in category_styles:
        mask = catalog[
            "outlier_transition"
        ].eq(category)

        axis.scatter(
            eazy_error[mask],
            hybrid_error[mask],
            s=24,
            alpha=0.65,
            color=color,
            marker=marker,
            label=category,
        )

    axis.plot(
        [minimum_error, plot_limit],
        [minimum_error, plot_limit],
        linestyle="--",
        color="black",
        linewidth=1.0,
        label="Equal error",
    )

    axis.axvline(
        CATASTROPHIC_OUTLIER_THRESHOLD,
        linestyle=":",
        color="#1f77b4",
        linewidth=1.1,
    )

    axis.axhline(
        CATASTROPHIC_OUTLIER_THRESHOLD,
        linestyle=":",
        color="#2ca02c",
        linewidth=1.1,
    )

    axis.set_xscale("log")
    axis.set_yscale("log")

    axis.set_xlim(
        minimum_error,
        plot_limit,
    )

    axis.set_ylim(
        minimum_error,
        plot_limit,
    )

    axis.set_xlabel(
        "Absolute normalized error: EAZY"
    )

    axis.set_ylabel(
        "Absolute normalized error: TabPFN–EAZY"
    )

    axis.set_title(
        "Per-source effect of the TabPFN correction"
    )

    axis.grid(
        alpha=0.2,
        which="both",
    )

    axis.legend(fontsize=7)


def draw_redshift_bin_panel(
    axis,
    redshift_summary: pd.DataFrame,
) -> None:
    """Show where in redshift the hybrid model improves EAZY."""

    x_values = redshift_summary[
        "median_z_spec"
    ]

    axis.plot(
        x_values,
        redshift_summary[
            "eazy_median_absolute_normalized_error"
        ],
        marker="o",
        linewidth=1.8,
        color="#1f77b4",
        label="EAZY",
    )

    axis.plot(
        x_values,
        redshift_summary[
            "hybrid_median_absolute_normalized_error"
        ],
        marker="o",
        linewidth=1.8,
        color="#2ca02c",
        label="TabPFN–EAZY",
    )

    for _, row in redshift_summary.iterrows():
        maximum_error = max(
            row[
                "eazy_median_absolute_normalized_error"
            ],
            row[
                "hybrid_median_absolute_normalized_error"
            ],
        )

        axis.annotate(
            f"n={int(row['sources'])}",
            (
                row["median_z_spec"],
                maximum_error,
            ),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )

    axis.set_xlabel(
        "Median spectroscopic redshift in bin"
    )

    axis.set_ylabel(
        "Median absolute normalized error"
    )

    axis.set_title(
        "Contribution by spectroscopic-redshift range"
    )

    axis.grid(alpha=0.2)
    axis.legend()


def draw_case_panel(
    axis,
    cases: pd.DataFrame,
) -> None:
    """Show representative EAZY-to-hybrid corrections."""

    y_positions = np.arange(
        len(cases)
    )

    for y_position, (_, row) in zip(
        y_positions,
        cases.iterrows(),
    ):
        arrow_color = (
            "#2ca02c"
            if row[
                "normalized_error_improvement"
            ]
            > 0.0
            else "#d62728"
        )

        axis.annotate(
            "",
            xy=(
                row["z_tabpfn_eazy"],
                y_position,
            ),
            xytext=(
                row["z_eazy"],
                y_position,
            ),
            arrowprops={
                "arrowstyle": "->",
                "color": arrow_color,
                "linewidth": 1.8,
            },
        )

    axis.scatter(
        cases["z_spec"],
        y_positions,
        marker="D",
        s=45,
        color="black",
        label="z_spec",
        zorder=3,
    )

    axis.scatter(
        cases["z_eazy"],
        y_positions,
        marker="o",
        s=45,
        color="#1f77b4",
        label="EAZY",
        zorder=3,
    )

    axis.scatter(
        cases["z_tabpfn_eazy"],
        y_positions,
        marker="s",
        s=45,
        color="#2ca02c",
        label="TabPFN–EAZY",
        zorder=3,
    )

    case_labels = [
        f"{row['case_group']}: "
        f"{row[SOURCE_KEY_COLUMN]}"
        for _, row in cases.iterrows()
    ]

    axis.set_yticks(y_positions)
    axis.set_yticklabels(
        case_labels,
        fontsize=7,
    )

    axis.invert_yaxis()

    axis.set_xlabel("Redshift")
    axis.set_title(
        "Representative source-level corrections"
    )

    axis.grid(
        axis="x",
        alpha=0.2,
    )

    axis.legend(
        fontsize=8,
        loc="best",
    )


def build_contribution_figure(
    catalog: pd.DataFrame,
    cases: pd.DataFrame,
    redshift_summary: pd.DataFrame,
    output_path: Path = CONTRIBUTION_FIGURE_PATH,
) -> Path:
    """Create the complete EAZY–TabPFN contribution figure."""

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(18, 13),
    )

    draw_residual_panel(
        axes[0, 0],
        catalog,
    )

    draw_error_comparison_panel(
        axes[0, 1],
        catalog,
    )

    draw_redshift_bin_panel(
        axes[1, 0],
        redshift_summary,
    )

    draw_case_panel(
        axes[1, 1],
        cases,
    )

    figure.suptitle(
        "JADES blind test: EAZY and TabPFN contributions",
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


def save_contribution_outputs(
    summary: pd.DataFrame,
    cases: pd.DataFrame,
    catalog: pd.DataFrame,
    redshift_summary: pd.DataFrame,
) -> None:
    """Save contribution tables and the analysis manifest."""

    summary.to_csv(
        CONTRIBUTION_SUMMARY_PATH,
        index=False,
    )

    cases.to_csv(
        CONTRIBUTION_CASES_PATH,
        index=False,
    )

    catalog.to_csv(
        CONTRIBUTION_CATALOG_PATH,
        index=False,
    )

    manifest = {
        "analysis": (
            "EAZY baseline plus applied TabPFN log-residual correction"
        ),
        "blind_test_sources": len(catalog),
        "representative_cases": len(cases),
        "redshift_bins": (
            redshift_summary[
                "z_spec_bin"
            ].tolist()
        ),
        "model_retrained": False,
        "gpu_used": False,
    }

    with CONTRIBUTION_MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )


def run_contribution_analysis(
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Run the complete saved-prediction contribution analysis."""

    ensure_output_directories()

    predictions = load_blind_test_predictions()

    catalog = build_contribution_catalog(
        predictions
    )

    summary = build_contribution_summary(
        catalog
    )

    cases = select_explanation_cases(
        catalog
    )

    redshift_summary = (
        build_redshift_bin_summary(
            catalog
        )
    )

    figure_path = build_contribution_figure(
        catalog=catalog,
        cases=cases,
        redshift_summary=redshift_summary,
    )

    save_contribution_outputs(
        summary=summary,
        cases=cases,
        catalog=catalog,
        redshift_summary=redshift_summary,
    )

    return (
        summary,
        cases,
        figure_path,
    )