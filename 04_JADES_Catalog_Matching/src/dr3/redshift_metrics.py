"""Evaluation metrics for photometric-redshift predictions."""

import numpy as np
import pandas as pd


DEFAULT_OUTLIER_THRESHOLD = 0.15

DEFAULT_REDSHIFT_BIN_EDGES = (
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
    np.inf,
)


def _require_columns(
    dataframe,
    required_columns,
    dataframe_name,
):
    """Raise an informative error if required columns are missing."""

    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing columns in {dataframe_name}: "
            f"{missing_columns}"
        )


def add_redshift_diagnostics(
    catalog,
    prediction_column,
    truth_column="z_spec",
    outlier_threshold=DEFAULT_OUTLIER_THRESHOLD,
):
    """Add redshift residuals and catastrophic-outlier flags."""

    _require_columns(
        catalog,
        [
            prediction_column,
            truth_column,
        ],
        "catalog",
    )

    diagnostics = catalog.copy()

    predicted_redshift = pd.to_numeric(
        diagnostics[prediction_column],
        errors="coerce",
    )

    reference_redshift = pd.to_numeric(
        diagnostics[truth_column],
        errors="coerce",
    )

    finite_pair = (
        np.isfinite(predicted_redshift)
        & np.isfinite(reference_redshift)
        & (1.0 + reference_redshift != 0)
    )

    diagnostics["redshift_residual"] = (
        predicted_redshift
        - reference_redshift
    )

    diagnostics.loc[
        ~finite_pair,
        "redshift_residual",
    ] = np.nan

    diagnostics["normalized_redshift_error"] = (
        diagnostics["redshift_residual"]
        / (1.0 + reference_redshift)
    )

    diagnostics.loc[
        ~finite_pair,
        "normalized_redshift_error",
    ] = np.nan

    diagnostics[
        "absolute_normalized_redshift_error"
    ] = diagnostics[
        "normalized_redshift_error"
    ].abs()

    outlier_flag = pd.Series(
        pd.NA,
        index=diagnostics.index,
        dtype="boolean",
    )

    outlier_flag.loc[finite_pair] = (
        diagnostics.loc[
            finite_pair,
            "absolute_normalized_redshift_error",
        ]
        > outlier_threshold
    )

    diagnostics["is_catastrophic_outlier"] = (
        outlier_flag
    )

    return diagnostics


def summarize_redshift_performance(
    diagnostics,
    outlier_threshold=DEFAULT_OUTLIER_THRESHOLD,
):
    """Summarize redshift accuracy for finite prediction pairs."""

    required_columns = [
        "redshift_residual",
        "normalized_redshift_error",
        "absolute_normalized_redshift_error",
        "is_catastrophic_outlier",
    ]

    _require_columns(
        diagnostics,
        required_columns,
        "diagnostics",
    )

    finite_error = np.isfinite(
        diagnostics["normalized_redshift_error"]
    )

    valid_diagnostics = diagnostics.loc[
        finite_error
    ].copy()

    if valid_diagnostics.empty:
        raise ValueError(
            "No finite redshift prediction pairs are available."
        )

    normalized_error = valid_diagnostics[
        "normalized_redshift_error"
    ].to_numpy()

    median_bias = float(
        np.median(normalized_error)
    )

    nmad = float(
        1.4826
        * np.median(
            np.abs(
                normalized_error
                - median_bias
            )
        )
    )

    number_outliers = int(
        valid_diagnostics[
            "is_catastrophic_outlier"
        ].sum()
    )

    summary = pd.Series(
        {
            "sources_with_finite_prediction_and_z_spec": (
                len(valid_diagnostics)
            ),
            "normalized_bias": median_bias,
            "normalized_median_absolute_error": float(
                np.median(
                    np.abs(normalized_error)
                )
            ),
            "sigma_nmad": nmad,
            "mean_absolute_redshift_error": float(
                valid_diagnostics[
                    "redshift_residual"
                ]
                .abs()
                .mean()
            ),
            "catastrophic_outlier_threshold": (
                outlier_threshold
            ),
            "catastrophic_outliers": number_outliers,
            "catastrophic_outlier_fraction": (
                number_outliers
                / len(valid_diagnostics)
            ),
        },
        name="value",
    )

    return summary.to_frame()


def summarize_performance_by_redshift_bin(
    diagnostics,
    redshift_bin_edges=DEFAULT_REDSHIFT_BIN_EDGES,
):
    """Summarize redshift performance in spectroscopic-redshift bins."""

    required_columns = [
        "z_spec",
        "normalized_redshift_error",
        "is_catastrophic_outlier",
    ]

    _require_columns(
        diagnostics,
        required_columns,
        "diagnostics",
    )

    finite_rows = (
        np.isfinite(diagnostics["z_spec"])
        & np.isfinite(
            diagnostics[
                "normalized_redshift_error"
            ]
        )
    )

    binned_diagnostics = diagnostics.loc[
        finite_rows
    ].copy()

    if binned_diagnostics.empty:
        raise ValueError(
            "No finite redshift diagnostics are available."
        )

    binned_diagnostics["z_spec_bin"] = pd.cut(
        binned_diagnostics["z_spec"],
        bins=redshift_bin_edges,
        right=False,
        include_lowest=True,
    )

    records = []

    for redshift_bin, group in (
        binned_diagnostics.groupby(
            "z_spec_bin",
            observed=True,
        )
    ):
        normalized_error = group[
            "normalized_redshift_error"
        ].to_numpy()

        median_bias = float(
            np.median(normalized_error)
        )

        nmad = float(
            1.4826
            * np.median(
                np.abs(
                    normalized_error
                    - median_bias
                )
            )
        )

        number_outliers = int(
            group[
                "is_catastrophic_outlier"
            ].sum()
        )

        records.append(
            {
                "z_spec_bin": str(
                    redshift_bin
                ),
                "sources": len(group),
                "median_z_spec": float(
                    group["z_spec"].median()
                ),
                "normalized_bias": median_bias,
                "sigma_nmad": nmad,
                "catastrophic_outliers": (
                    number_outliers
                ),
                "catastrophic_outlier_fraction": (
                    number_outliers
                    / len(group)
                ),
            }
        )

    return pd.DataFrame(records)