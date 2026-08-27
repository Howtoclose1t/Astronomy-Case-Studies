"""Match JADES photometric and spectroscopic catalogs."""

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from .config import (
    FILTERS,
    MATCH_RADIUS_ARCSEC,
    MIN_VALID_FILTERS,
    RADIUS_SCAN_ARCSEC,
)


QUALITY_RANK = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
}


def _require_dataframe_columns(
    dataframe,
    required_columns,
    dataframe_name,
):
    """Raise an informative error if DataFrame columns are missing."""

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


def select_best_spectrum(
    spectroscopy_catalog,
):
    """Select one best spectroscopic record per positive NIRCam ID."""

    required_columns = [
        "catalog_nircam_id",
        "z_spec",
        "z_spec_quality",
        "dr_problem",
        "prism_exposure_s",
    ]

    _require_dataframe_columns(
        spectroscopy_catalog,
        required_columns,
        "spectroscopy_catalog",
    )

    candidates = spectroscopy_catalog.loc[
        spectroscopy_catalog[
            "catalog_nircam_id"
        ] > 0
    ].copy()

    candidates["_has_valid_z"] = (
        candidates["z_spec"].notna()
    )

    candidates["_quality_rank"] = (
        candidates["z_spec_quality"]
        .map(QUALITY_RANK)
        .fillna(99)
        .astype(int)
    )

    duplicate_mask = candidates.duplicated(
        subset="catalog_nircam_id",
        keep=False,
    )

    duplicate_records = candidates.loc[
        duplicate_mask
    ].copy()

    candidates = candidates.sort_values(
        [
            "catalog_nircam_id",
            "dr_problem",
            "_has_valid_z",
            "_quality_rank",
            "prism_exposure_s",
        ],
        ascending=[
            True,
            True,
            False,
            True,
            False,
        ],
        kind="stable",
    )

    best_spectra = (
        candidates
        .drop_duplicates(
            subset="catalog_nircam_id",
            keep="first",
        )
        .drop(
            columns=[
                "_has_valid_z",
                "_quality_rank",
            ]
        )
        .reset_index(drop=True)
    )

    duplicate_records = (
        duplicate_records
        .drop(
            columns=[
                "_has_valid_z",
                "_quality_rank",
            ]
        )
        .sort_values(
            [
                "catalog_nircam_id",
                "z_spec_quality",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if not best_spectra[
        "catalog_nircam_id"
    ].is_unique:
        raise ValueError(
            "Spectroscopic deduplication failed."
        )

    return best_spectra, duplicate_records


def match_catalogs(
    photometry_catalog,
    best_spectra,
    match_radius_arcsec=MATCH_RADIUS_ARCSEC,
):
    """Match spectroscopy to photometry using ID and sky coordinates."""

    required_photometry_columns = [
        "phot_id",
        "ra_deg",
        "dec_deg",
        "has_finite_coordinates",
    ]

    required_spectroscopy_columns = [
        "catalog_nircam_id",
        "ra_nircam_deg",
        "dec_nircam_deg",
        "has_finite_nircam_coordinates",
    ]

    _require_dataframe_columns(
        photometry_catalog,
        required_photometry_columns,
        "photometry_catalog",
    )

    _require_dataframe_columns(
        best_spectra,
        required_spectroscopy_columns,
        "best_spectra",
    )

    if not photometry_catalog[
        "phot_id"
    ].is_unique:
        raise ValueError(
            "Photometric source IDs must be unique."
        )

    if not best_spectra[
        "catalog_nircam_id"
    ].is_unique:
        raise ValueError(
            "Spectroscopic NIRCam IDs must be unique."
        )

    valid_photometry = (
        photometry_catalog.loc[
            photometry_catalog[
                "has_finite_coordinates"
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    if valid_photometry.empty:
        raise ValueError(
            "No photometric sources have finite coordinates."
        )

    match_audit = best_spectra.copy()

    match_audit["published_id_in_photometry"] = (
        match_audit["catalog_nircam_id"].isin(
            photometry_catalog["phot_id"]
        )
    )

    match_audit["exact_id_separation_arcsec"] = np.nan
    match_audit["nearest_phot_id"] = pd.Series(
        pd.NA,
        index=match_audit.index,
        dtype="Int64",
    )
    match_audit["nearest_separation_arcsec"] = np.nan

    photometry_by_id = photometry_catalog.set_index(
        "phot_id"
    )

    match_audit["_exact_ra_deg"] = (
        match_audit["catalog_nircam_id"].map(
            photometry_by_id["ra_deg"]
        )
    )

    match_audit["_exact_dec_deg"] = (
        match_audit["catalog_nircam_id"].map(
            photometry_by_id["dec_deg"]
        )
    )

    finite_exact_coordinates = (
        match_audit[
            "has_finite_nircam_coordinates"
        ]
        & np.isfinite(
            match_audit["_exact_ra_deg"]
        )
        & np.isfinite(
            match_audit["_exact_dec_deg"]
        )
    )

    if finite_exact_coordinates.any():
        exact_photometric_coordinates = SkyCoord(
            ra=match_audit.loc[
                finite_exact_coordinates,
                "_exact_ra_deg",
            ].to_numpy()
            * u.deg,
            dec=match_audit.loc[
                finite_exact_coordinates,
                "_exact_dec_deg",
            ].to_numpy()
            * u.deg,
        )

        exact_spectroscopic_coordinates = SkyCoord(
            ra=match_audit.loc[
                finite_exact_coordinates,
                "ra_nircam_deg",
            ].to_numpy()
            * u.deg,
            dec=match_audit.loc[
                finite_exact_coordinates,
                "dec_nircam_deg",
            ].to_numpy()
            * u.deg,
        )

        exact_separation = (
            exact_photometric_coordinates.separation(
                exact_spectroscopic_coordinates
            )
        )

        match_audit.loc[
            finite_exact_coordinates,
            "exact_id_separation_arcsec",
        ] = exact_separation.arcsec

    all_photometric_coordinates = SkyCoord(
        ra=valid_photometry[
            "ra_deg"
        ].to_numpy()
        * u.deg,
        dec=valid_photometry[
            "dec_deg"
        ].to_numpy()
        * u.deg,
    )

    finite_spectroscopic_coordinates = (
        match_audit[
            "has_finite_nircam_coordinates"
        ]
    )

    if finite_spectroscopic_coordinates.any():
        spectroscopic_coordinates = SkyCoord(
            ra=match_audit.loc[
                finite_spectroscopic_coordinates,
                "ra_nircam_deg",
            ].to_numpy()
            * u.deg,
            dec=match_audit.loc[
                finite_spectroscopic_coordinates,
                "dec_nircam_deg",
            ].to_numpy()
            * u.deg,
        )

        nearest_rows, nearest_separation, _ = (
            spectroscopic_coordinates.match_to_catalog_sky(
                all_photometric_coordinates
            )
        )

        nearest_ids = valid_photometry.iloc[
            nearest_rows
        ]["phot_id"].to_numpy()

        match_audit.loc[
            finite_spectroscopic_coordinates,
            "nearest_phot_id",
        ] = nearest_ids

        match_audit.loc[
            finite_spectroscopic_coordinates,
            "nearest_separation_arcsec",
        ] = nearest_separation.arcsec

    match_audit["nearest_id_agrees"] = (
        match_audit["nearest_phot_id"]
        == match_audit["catalog_nircam_id"]
    )

    match_audit["accept_exact_id"] = (
        match_audit[
            "published_id_in_photometry"
        ]
        & match_audit[
            "exact_id_separation_arcsec"
        ].le(match_radius_arcsec)
    )

    match_audit["accept_sky_fallback"] = (
        ~match_audit["accept_exact_id"]
        & match_audit[
            "nearest_separation_arcsec"
        ].le(match_radius_arcsec)
    )

    match_audit["selected_phot_id"] = pd.Series(
        pd.NA,
        index=match_audit.index,
        dtype="Int64",
    )

    match_audit["selected_separation_arcsec"] = np.nan
    match_audit["match_method"] = "unmatched"

    exact_mask = match_audit["accept_exact_id"]

    match_audit.loc[
        exact_mask,
        "selected_phot_id",
    ] = match_audit.loc[
        exact_mask,
        "catalog_nircam_id",
    ].astype("Int64")

    match_audit.loc[
        exact_mask,
        "selected_separation_arcsec",
    ] = match_audit.loc[
        exact_mask,
        "exact_id_separation_arcsec",
    ]

    match_audit.loc[
        exact_mask,
        "match_method",
    ] = "exact_id"

    fallback_mask = match_audit[
        "accept_sky_fallback"
    ]

    match_audit.loc[
        fallback_mask,
        "selected_phot_id",
    ] = match_audit.loc[
        fallback_mask,
        "nearest_phot_id",
    ].astype("Int64")

    match_audit.loc[
        fallback_mask,
        "selected_separation_arcsec",
    ] = match_audit.loc[
        fallback_mask,
        "nearest_separation_arcsec",
    ]

    match_audit.loc[
        fallback_mask,
        "match_method",
    ] = "nearest_sky"

    accepted_matches = match_audit.loc[
        match_audit[
            "selected_phot_id"
        ].notna()
    ].copy()

    accepted_matches["selected_phot_id"] = (
        accepted_matches[
            "selected_phot_id"
        ].astype(np.int64)
    )

    if not accepted_matches[
        "selected_phot_id"
    ].is_unique:
        duplicated_ids = accepted_matches.loc[
            accepted_matches[
                "selected_phot_id"
            ].duplicated(keep=False),
            "selected_phot_id",
        ].unique()

        raise ValueError(
            "Multiple spectra selected the same "
            f"photometric source: {duplicated_ids}"
        )

    accepted_matches = accepted_matches.drop(
        columns=[
            "_exact_ra_deg",
            "_exact_dec_deg",
        ]
    )

    matched_catalog = photometry_catalog.merge(
        accepted_matches,
        left_on="phot_id",
        right_on="selected_phot_id",
        how="inner",
        validate="one_to_one",
    )

    matched_catalog["separation_arcsec"] = (
        matched_catalog[
            "selected_separation_arcsec"
        ]
    )

    matched_catalog["id_consistent"] = (
        matched_catalog["phot_id"]
        == matched_catalog["catalog_nircam_id"]
    )

    match_audit = match_audit.drop(
        columns=[
            "_exact_ra_deg",
            "_exact_dec_deg",
        ]
    )

    return matched_catalog, match_audit


def build_radius_scan(
    match_audit,
    radii_arcsec=RADIUS_SCAN_ARCSEC,
):
    """Count nearest-neighbor matches across candidate radii."""

    _require_dataframe_columns(
        match_audit,
        ["nearest_separation_arcsec"],
        "match_audit",
    )

    finite_separation = match_audit[
        "nearest_separation_arcsec"
    ].notna()

    number_with_finite_separation = int(
        finite_separation.sum()
    )

    records = []

    for radius_arcsec in radii_arcsec:
        accepted = (
            match_audit.loc[
                finite_separation,
                "nearest_separation_arcsec",
            ]
            <= radius_arcsec
        )

        if number_with_finite_separation > 0:
            accepted_fraction = (
                accepted.sum()
                / number_with_finite_separation
            )
        else:
            accepted_fraction = np.nan

        records.append(
            {
                "radius_arcsec": radius_arcsec,
                "nearest_matches": int(
                    accepted.sum()
                ),
                "fraction_of_finite_coordinates": (
                    accepted_fraction
                ),
            }
        )

    return pd.DataFrame(records)


def add_photometric_validity(
    matched_catalog,
):
    """Add per-filter validity and ML-sample selection flags."""

    required_columns = [
        "is_secure_spec",
        "is_flagged_star",
    ]

    for filter_name in FILTERS:
        filter_key = filter_name.lower()

        required_columns.extend(
            [
                f"flux_{filter_key}_njy",
                f"fluxerr_{filter_key}_nmad_njy",
            ]
        )

    _require_dataframe_columns(
        matched_catalog,
        required_columns,
        "matched_catalog",
    )

    catalog = matched_catalog.copy()
    valid_filter_columns = []

    for filter_name in FILTERS:
        filter_key = filter_name.lower()

        flux_column = (
            f"flux_{filter_key}_njy"
        )

        error_column = (
            f"fluxerr_{filter_key}_nmad_njy"
        )

        valid_column = (
            f"valid_{filter_key}"
        )

        catalog[valid_column] = (
            np.isfinite(catalog[flux_column])
            & np.isfinite(catalog[error_column])
            & catalog[error_column].gt(0)
        )

        valid_filter_columns.append(
            valid_column
        )

    catalog["n_valid_filters"] = (
        catalog[valid_filter_columns]
        .sum(axis=1)
        .astype(np.int8)
    )

    catalog["is_ml_ready"] = (
        catalog["is_secure_spec"]
        & ~catalog["is_flagged_star"]
        & catalog["n_valid_filters"].ge(
            MIN_VALID_FILTERS
        )
    )

    return catalog


def build_ml_ready_catalog(
    matched_catalog,
):
    """Build the modeling table for the photometric-redshift baseline."""

    _require_dataframe_columns(
        matched_catalog,
        ["is_ml_ready"],
        "matched_catalog",
    )

    identity_columns = [
        "phot_id",
        "ra_deg",
        "dec_deg",
        "nirspec_id",
        "tier",
        "program_id",
        "z_phot",
        "z_spec",
        "z_spec_quality",
        "match_method",
        "separation_arcsec",
        "flag_bright_star",
        "flag_bad_neighbor",
        "n_valid_filters",
    ]

    measurement_columns = []
    validity_columns = []

    for filter_name in FILTERS:
        filter_key = filter_name.lower()

        measurement_columns.extend(
            [
                f"flux_{filter_key}_njy",
                f"fluxerr_{filter_key}_nmad_njy",
            ]
        )

        validity_columns.append(
            f"valid_{filter_key}"
        )

    selected_columns = (
        identity_columns
        + measurement_columns
        + validity_columns
    )

    _require_dataframe_columns(
        matched_catalog,
        selected_columns,
        "matched_catalog",
    )

    ml_ready_catalog = (
        matched_catalog.loc[
            matched_catalog["is_ml_ready"],
            selected_columns,
        ]
        .sort_values(
            "phot_id",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if not ml_ready_catalog[
        "phot_id"
    ].is_unique:
        raise ValueError(
            "ML-ready photometric IDs are not unique."
        )

    return ml_ready_catalog