"""Reusable diagnostic plots for the JADES catalog-matching case study."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    FILTERS,
    MATCH_RADIUS_ARCSEC,
    MIN_VALID_FILTERS,
)


MATCH_METHOD_ORDER = (
    "exact_id",
    "nearest_sky",
    "unmatched",
)

MATCH_METHOD_LABELS = {
    "exact_id": "Published ID",
    "nearest_sky": "Nearest-sky fallback",
    "unmatched": "Unmatched",
}

MATCH_METHOD_COLORS = {
    "exact_id": "#2A6FBB",
    "nearest_sky": "#F28E2B",
    "unmatched": "#C44E52",
}


def _require_columns(
    dataframe,
    required_columns,
    dataframe_name,
):
    """Raise an informative error when plotting columns are missing."""

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


def plot_match_overview(
    match_audit,
    radius_scan,
    match_radius_arcsec=MATCH_RADIUS_ARCSEC,
):
    """Plot match-method counts and matching-radius sensitivity."""

    _require_columns(
        match_audit,
        ["match_method"],
        "match_audit",
    )

    _require_columns(
        radius_scan,
        [
            "radius_arcsec",
            "nearest_matches",
        ],
        "radius_scan",
    )

    method_counts = (
        match_audit["match_method"]
        .value_counts()
        .reindex(
            MATCH_METHOD_ORDER,
            fill_value=0,
        )
    )

    radius_data = (
        radius_scan
        .sort_values(
            "radius_arcsec",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 4.5),
    )

    method_axis = axes[0]

    method_bars = method_axis.bar(
        [
            MATCH_METHOD_LABELS[method]
            for method in MATCH_METHOD_ORDER
        ],
        method_counts.to_numpy(),
        color=[
            MATCH_METHOD_COLORS[method]
            for method in MATCH_METHOD_ORDER
        ],
    )

    method_axis.bar_label(
        method_bars,
        labels=[
            f"{int(count):,}"
            for count in method_counts
        ],
        padding=3,
    )

    method_axis.set_title(
        "Final catalog-match decisions"
    )

    method_axis.set_ylabel(
        "Number of NIRSpec sources"
    )

    method_axis.tick_params(
        axis="x",
        rotation=15,
    )

    method_axis.grid(
        axis="y",
        alpha=0.25,
    )

    radius_axis = axes[1]

    radius_axis.plot(
        radius_data["radius_arcsec"],
        radius_data["nearest_matches"],
        marker="o",
        linewidth=2,
        color="#2A6FBB",
    )

    for radius, number_matches in zip(
        radius_data["radius_arcsec"],
        radius_data["nearest_matches"],
    ):
        radius_axis.annotate(
            f"{int(number_matches):,}",
            xy=(
                radius,
                number_matches,
            ),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    radius_axis.axvline(
        match_radius_arcsec,
        color="#C44E52",
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Adopted radius = "
            f"{match_radius_arcsec:.2f} arcsec"
        ),
    )

    radius_axis.set_title(
        "Nearest-neighbor radius sensitivity"
    )

    radius_axis.set_xlabel(
        "Matching radius (arcsec)"
    )

    radius_axis.set_ylabel(
        "Nearest neighbors within radius"
    )

    radius_axis.grid(
        alpha=0.25,
    )

    radius_axis.legend(
        frameon=False,
    )

    figure.suptitle(
        "JADES NIRCam–NIRSpec matching diagnostics",
        fontsize=14,
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.94)
    )

    return figure, axes


def plot_separation_distribution(
    match_audit,
    match_radius_arcsec=MATCH_RADIUS_ARCSEC,
):
    """Plot accepted counterpart-coordinate separations."""

    _require_columns(
        match_audit,
        [
            "selected_separation_arcsec",
            "match_method",
        ],
        "match_audit",
    )

    accepted_separations = pd.to_numeric(
        match_audit.loc[
            match_audit["match_method"]
            != "unmatched",
            "selected_separation_arcsec",
        ],
        errors="coerce",
    ).dropna()

    if accepted_separations.empty:
        raise ValueError(
            "No accepted match separations are available."
        )

    separation_values = (
        accepted_separations.to_numpy()
    )

    upper_limit = max(
        match_radius_arcsec * 1.02,
        float(separation_values.max()) * 1.02,
    )

    histogram_bins = np.linspace(
        0,
        upper_limit,
        26,
    )

    figure, axis = plt.subplots(
        figsize=(7, 4.5)
    )

    axis.hist(
        separation_values,
        bins=histogram_bins,
        color="#2A6FBB",
        edgecolor="white",
    )

    axis.set_yscale("log")

    axis.axvline(
        match_radius_arcsec,
        color="#C44E52",
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Acceptance limit = "
            f"{match_radius_arcsec:.2f} arcsec"
        ),
    )

    zero_separations = int(
        np.isclose(
            separation_values,
            0.0,
            atol=1e-12,
        ).sum()
    )

    axis.text(
        0.97,
        0.92,
        (
            f"Accepted: {len(separation_values):,}\n"
            f"Exactly zero: {zero_separations:,}"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
    )

    axis.set_title(
        "Accepted catalog-match separations"
    )

    axis.set_xlabel(
        "Selected angular separation (arcsec)"
    )

    axis.set_ylabel(
        "Number of sources"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend(
        frameon=False,
    )

    figure.tight_layout()

    return figure, axis


def plot_secure_sources_on_sky(
    matched_catalog,
):
    """Plot accepted matches and secure-redshift sources on the sky."""

    _require_columns(
        matched_catalog,
        [
            "ra_deg",
            "dec_deg",
            "z_spec",
            "is_secure_spec",
        ],
        "matched_catalog",
    )

    finite_coordinates = (
        np.isfinite(matched_catalog["ra_deg"])
        & np.isfinite(matched_catalog["dec_deg"])
    )

    plot_catalog = matched_catalog.loc[
        finite_coordinates
    ].copy()

    secure_catalog = plot_catalog.loc[
        plot_catalog["is_secure_spec"]
        & np.isfinite(plot_catalog["z_spec"])
    ].copy()

    if secure_catalog.empty:
        raise ValueError(
            "No secure spectroscopic sources are available."
        )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    axis.scatter(
        plot_catalog["ra_deg"],
        plot_catalog["dec_deg"],
        s=8,
        color="lightgray",
        alpha=0.45,
        label="All accepted matches",
    )

    secure_points = axis.scatter(
        secure_catalog["ra_deg"],
        secure_catalog["dec_deg"],
        c=secure_catalog["z_spec"],
        s=14,
        cmap="viridis",
        alpha=0.85,
        label="Secure A/B/C redshifts",
    )

    colorbar = figure.colorbar(
        secure_points,
        ax=axis,
    )

    colorbar.set_label(
        "Spectroscopic redshift"
    )

    axis.set_title(
        "Secure NIRSpec redshifts across GOODS-N"
    )

    axis.set_xlabel(
        "Right ascension (deg)"
    )

    axis.set_ylabel(
        "Declination (deg)"
    )

    axis.invert_xaxis()

    axis.grid(
        alpha=0.2,
    )

    axis.legend(
        frameon=False,
        loc="best",
    )

    figure.tight_layout()

    return figure, axis


def plot_redshift_comparison(
    catalog,
):
    """Plot photometric redshift against spectroscopic redshift."""

    _require_columns(
        catalog,
        [
            "z_phot",
            "z_spec",
        ],
        "catalog",
    )

    redshift_data = catalog[
        [
            "z_phot",
            "z_spec",
        ]
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    finite_redshifts = (
        np.isfinite(redshift_data["z_phot"])
        & np.isfinite(redshift_data["z_spec"])
    )

    redshift_data = redshift_data.loc[
        finite_redshifts
    ].copy()

    if redshift_data.empty:
        raise ValueError(
            "No finite redshift pairs are available."
        )

    minimum_redshift = min(
        0.0,
        float(redshift_data.min().min()),
    )

    maximum_redshift = max(
        1.0,
        float(redshift_data.max().max()),
    )

    figure, axis = plt.subplots(
        figsize=(6, 6)
    )

    axis.scatter(
        redshift_data["z_spec"],
        redshift_data["z_phot"],
        s=18,
        color="#2A6FBB",
        alpha=0.55,
        edgecolors="none",
    )

    axis.plot(
        [
            minimum_redshift,
            maximum_redshift,
        ],
        [
            minimum_redshift,
            maximum_redshift,
        ],
        color="#C44E52",
        linestyle="--",
        linewidth=1.5,
        label="One-to-one relation",
    )

    axis.text(
        0.04,
        0.95,
        f"N = {len(redshift_data):,}",
        transform=axis.transAxes,
        ha="left",
        va="top",
    )

    axis.set_xlim(
        minimum_redshift,
        maximum_redshift,
    )

    axis.set_ylim(
        minimum_redshift,
        maximum_redshift,
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    axis.set_title(
        "JADES photometric and spectroscopic redshifts"
    )

    axis.set_xlabel(
        "Spectroscopic redshift"
    )

    axis.set_ylabel(
        "Photometric redshift"
    )

    axis.grid(
        alpha=0.2,
    )

    axis.legend(
        frameon=False,
    )

    figure.tight_layout()

    return figure, axis


def plot_photometric_coverage(
    catalog,
):
    """Plot per-filter validity and the number of valid filters."""

    valid_filter_columns = [
        f"valid_{filter_name.lower()}"
        for filter_name in FILTERS
    ]

    _require_columns(
        catalog,
        valid_filter_columns
        + ["n_valid_filters"],
        "catalog",
    )

    coverage_percent = (
        catalog[valid_filter_columns]
        .mean(axis=0)
        .mul(100)
    )

    valid_filter_counts = pd.to_numeric(
        catalog["n_valid_filters"],
        errors="coerce",
    ).dropna()

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 4.5),
    )

    coverage_axis = axes[0]

    coverage_bars = coverage_axis.bar(
        FILTERS,
        coverage_percent.to_numpy(),
        color="#2A6FBB",
    )

    coverage_axis.bar_label(
        coverage_bars,
        labels=[
            f"{percentage:.1f}%"
            for percentage in coverage_percent
        ],
        padding=3,
        fontsize=8,
    )

    coverage_axis.set_ylim(
        0,
        108,
    )

    coverage_axis.set_title(
        "Valid photometry by NIRCam filter"
    )

    coverage_axis.set_xlabel(
        "NIRCam filter"
    )

    coverage_axis.set_ylabel(
        "Sources with valid measurement (%)"
    )

    coverage_axis.tick_params(
        axis="x",
        rotation=45,
    )

    coverage_axis.grid(
        axis="y",
        alpha=0.25,
    )

    count_axis = axes[1]

    count_axis.hist(
        valid_filter_counts,
        bins=np.arange(
            -0.5,
            len(FILTERS) + 1.5,
            1,
        ),
        color="#59A14F",
        edgecolor="white",
    )

    count_axis.axvline(
        MIN_VALID_FILTERS - 0.5,
        color="#C44E52",
        linestyle="--",
        linewidth=1.5,
        label=(
            f"ML threshold: at least "
            f"{MIN_VALID_FILTERS} filters"
        ),
    )

    count_axis.set_xticks(
        range(len(FILTERS) + 1)
    )

    count_axis.set_title(
        "Number of valid filters per source"
    )

    count_axis.set_xlabel(
        "Valid NIRCam filters"
    )

    count_axis.set_ylabel(
        "Number of matched sources"
    )

    count_axis.grid(
        axis="y",
        alpha=0.25,
    )

    count_axis.legend(
        frameon=False,
    )

    figure.tight_layout()

    return figure, axes