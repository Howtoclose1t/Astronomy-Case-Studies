"""Match the JADES DR4 spectroscopy and DR5 source catalogs."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from .config import (
    MATCH_RADIUS_ARCSEC,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    SECURE_SPEC_FLAGS,
)
from .dr45_staging import (
    VALID_FIELDS,
    load_dr45_staging,
)


MATCHED_CORE_PATH = (
    PROCESSED_DATA_DIR
    / "jades_dr4_dr5_matched_core.csv"
)

MATCH_AUDIT_PATH = (
    METRICS_DIR
    / "dr45_match_audit.csv"
)

DUPLICATE_SPECTRA_PATH = (
    METRICS_DIR
    / "dr45_duplicate_spectra.csv"
)

MATCH_SUMMARY_PATH = (
    METRICS_DIR
    / "dr45_match_summary.csv"
)

MATCH_METHOD_SUMMARY_PATH = (
    METRICS_DIR
    / "dr45_match_method_summary.csv"
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
    """Check that a DataFrame contains the required columns."""

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


def _quality_rank(
    quality_series,
):
    """Convert spectroscopic quality labels into sortable ranks."""

    return (
        quality_series
        .astype(str)
        .str.strip()
        .str.upper()
        .map(QUALITY_RANK)
        .fillna(99)
        .astype(np.int16)
    )


def select_best_spectra(
    spectroscopy_catalog,
):
    """Retain one best record for each positive field-DR5-ID key."""

    required_columns = [
        "field",
        "nircam_dr5_id",
        "z_spec",
        "z_spec_quality",
        "prism_exposure_s",
        "has_positive_dr5_id",
    ]

    _require_dataframe_columns(
        spectroscopy_catalog,
        required_columns,
        "spectroscopy_catalog",
    )

    catalog = (
        spectroscopy_catalog
        .copy()
        .reset_index(drop=True)
    )

    catalog["_input_row"] = np.arange(
        len(catalog),
        dtype=np.int64,
    )

    positive_id_records = catalog.loc[
        catalog["has_positive_dr5_id"]
    ].copy()

    no_positive_id_records = catalog.loc[
        ~catalog["has_positive_dr5_id"]
    ].copy()

    key_columns = [
        "field",
        "nircam_dr5_id",
    ]

    positive_id_records[
        "_is_repeated_key"
    ] = positive_id_records.duplicated(
        subset=key_columns,
        keep=False,
    )

    positive_id_records[
        "_has_finite_z"
    ] = np.isfinite(
        positive_id_records["z_spec"]
    )

    positive_id_records[
        "_quality_rank"
    ] = _quality_rank(
        positive_id_records[
            "z_spec_quality"
        ]
    )

    positive_id_records = (
        positive_id_records
        .sort_values(
            [
                "field",
                "nircam_dr5_id",
                "_has_finite_z",
                "_quality_rank",
                "prism_exposure_s",
                "_input_row",
            ],
            ascending=[
                True,
                True,
                False,
                True,
                False,
                True,
            ],
            kind="stable",
        )
    )

    positive_id_records[
        "selected_as_best"
    ] = ~positive_id_records.duplicated(
        subset=key_columns,
        keep="first",
    )

    duplicate_audit = (
        positive_id_records.loc[
            positive_id_records[
                "_is_repeated_key"
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    best_positive_id_records = (
        positive_id_records.loc[
            positive_id_records[
                "selected_as_best"
            ]
        ]
        .drop(
            columns=[
                "_is_repeated_key",
                "_has_finite_z",
                "_quality_rank",
                "selected_as_best",
            ]
        )
    )

    no_positive_id_records[
        "selected_as_best"
    ] = True

    no_positive_id_records = (
        no_positive_id_records
        .drop(
            columns=[
                "selected_as_best",
            ]
        )
    )

    selected_spectra = (
        pd.concat(
            [
                best_positive_id_records,
                no_positive_id_records,
            ],
            ignore_index=True,
        )
        .sort_values(
            "_input_row",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if selected_spectra.loc[
        selected_spectra[
            "has_positive_dr5_id"
        ]
    ].duplicated(
        subset=key_columns
    ).any():
        raise ValueError(
            "Spectroscopic deduplication failed."
        )

    return (
        selected_spectra,
        duplicate_audit,
    )


def _calculate_separation_arcsec(
    first_ra_deg,
    first_dec_deg,
    second_ra_deg,
    second_dec_deg,
):
    """Calculate angular separations between paired sky positions."""

    first_coordinates = SkyCoord(
        ra=np.asarray(
            first_ra_deg
        )
        * u.deg,
        dec=np.asarray(
            first_dec_deg
        )
        * u.deg,
    )

    second_coordinates = SkyCoord(
        ra=np.asarray(
            second_ra_deg
        )
        * u.deg,
        dec=np.asarray(
            second_dec_deg
        )
        * u.deg,
    )

    return first_coordinates.separation(
        second_coordinates
    ).arcsec


def match_dr45_catalogs(
    selected_spectra,
    source_catalog,
    match_radius_arcsec=(
        MATCH_RADIUS_ARCSEC
    ),
):
    """Match spectra by official DR5 ID, then use a sky fallback."""

    required_spectroscopy_columns = [
        "field",
        "nircam_dr5_id",
        "ra_target_deg",
        "dec_target_deg",
        "z_spec",
        "z_spec_quality",
        "prism_exposure_s",
        "has_positive_dr5_id",
        "has_finite_target_coordinates",
    ]

    required_source_columns = [
        "field",
        "phot_id",
        "ra_deg",
        "dec_deg",
        "has_finite_coordinates",
    ]

    _require_dataframe_columns(
        selected_spectra,
        required_spectroscopy_columns,
        "selected_spectra",
    )

    _require_dataframe_columns(
        source_catalog,
        required_source_columns,
        "source_catalog",
    )

    if source_catalog.duplicated(
        subset=[
            "field",
            "phot_id",
        ]
    ).any():
        raise ValueError(
            "DR5 field-phot_id keys must be unique."
        )

    exact_source_lookup = (
        source_catalog[
            [
                "field",
                "phot_id",
                "ra_deg",
                "dec_deg",
            ]
        ]
        .rename(
            columns={
                "phot_id": "exact_phot_id",
                "ra_deg": (
                    "exact_source_ra_deg"
                ),
                "dec_deg": (
                    "exact_source_dec_deg"
                ),
            }
        )
    )

    match_audit = selected_spectra.merge(
        exact_source_lookup,
        left_on=[
            "field",
            "nircam_dr5_id",
        ],
        right_on=[
            "field",
            "exact_phot_id",
        ],
        how="left",
        validate="many_to_one",
    )

    match_audit[
        "published_id_in_dr5"
    ] = (
        match_audit[
            "has_positive_dr5_id"
        ]
        & match_audit[
            "exact_phot_id"
        ].notna()
    )

    match_audit[
        "exact_id_separation_arcsec"
    ] = np.nan

    finite_exact_coordinates = (
        match_audit[
            "published_id_in_dr5"
        ]
        & match_audit[
            "has_finite_target_coordinates"
        ]
        & np.isfinite(
            match_audit[
                "exact_source_ra_deg"
            ]
        )
        & np.isfinite(
            match_audit[
                "exact_source_dec_deg"
            ]
        )
    )

    if finite_exact_coordinates.any():
        match_audit.loc[
            finite_exact_coordinates,
            "exact_id_separation_arcsec",
        ] = _calculate_separation_arcsec(
            match_audit.loc[
                finite_exact_coordinates,
                "ra_target_deg",
            ],
            match_audit.loc[
                finite_exact_coordinates,
                "dec_target_deg",
            ],
            match_audit.loc[
                finite_exact_coordinates,
                "exact_source_ra_deg",
            ],
            match_audit.loc[
                finite_exact_coordinates,
                "exact_source_dec_deg",
            ],
        )

    match_audit[
        "nearest_phot_id"
    ] = pd.Series(
        pd.NA,
        index=match_audit.index,
        dtype="Int64",
    )

    match_audit[
        "nearest_separation_arcsec"
    ] = np.nan

    fallback_search_rows = (
        ~match_audit[
            "published_id_in_dr5"
        ]
        & match_audit[
            "has_finite_target_coordinates"
        ]
    )

    for field in VALID_FIELDS:
        field_spectrum_mask = (
            fallback_search_rows
            & match_audit[
                "field"
            ].eq(field)
        )

        field_spectrum_rows = (
            match_audit.index[
                field_spectrum_mask
            ]
        )

        if len(field_spectrum_rows) == 0:
            continue

        field_sources = (
            source_catalog.loc[
                source_catalog[
                    "field"
                ].eq(field)
                & source_catalog[
                    "has_finite_coordinates"
                ]
            ]
            .reset_index(drop=True)
        )

        if field_sources.empty:
            continue

        spectroscopic_coordinates = (
            SkyCoord(
                ra=match_audit.loc[
                    field_spectrum_rows,
                    "ra_target_deg",
                ].to_numpy()
                * u.deg,
                dec=match_audit.loc[
                    field_spectrum_rows,
                    "dec_target_deg",
                ].to_numpy()
                * u.deg,
            )
        )

        photometric_coordinates = SkyCoord(
            ra=field_sources[
                "ra_deg"
            ].to_numpy()
            * u.deg,
            dec=field_sources[
                "dec_deg"
            ].to_numpy()
            * u.deg,
        )

        (
            nearest_source_rows,
            nearest_separations,
            _,
        ) = (
            spectroscopic_coordinates
            .match_to_catalog_sky(
                photometric_coordinates
            )
        )

        nearest_source_ids = (
            field_sources.iloc[
                nearest_source_rows
            ]["phot_id"]
            .to_numpy()
        )

        match_audit.loc[
            field_spectrum_rows,
            "nearest_phot_id",
        ] = nearest_source_ids

        match_audit.loc[
            field_spectrum_rows,
            "nearest_separation_arcsec",
        ] = nearest_separations.arcsec

    match_audit[
        "selected_phot_id"
    ] = pd.Series(
        pd.NA,
        index=match_audit.index,
        dtype="Int64",
    )

    match_audit[
        "selected_separation_arcsec"
    ] = np.nan

    match_audit[
        "match_method"
    ] = "unmatched"

    match_audit[
        "blocked_by_exact_match"
    ] = False

    match_audit[
        "fallback_conflict"
    ] = False

    exact_match_mask = match_audit[
        "published_id_in_dr5"
    ]

    match_audit.loc[
        exact_match_mask,
        "selected_phot_id",
    ] = match_audit.loc[
        exact_match_mask,
        "nircam_dr5_id",
    ].astype("Int64")

    match_audit.loc[
        exact_match_mask,
        "selected_separation_arcsec",
    ] = match_audit.loc[
        exact_match_mask,
        "exact_id_separation_arcsec",
    ]

    match_audit.loc[
        exact_match_mask,
        "match_method",
    ] = "exact_dr5_id"

    exact_source_keys = {
        (
            str(row.field),
            int(row.selected_phot_id),
        )
        for row in match_audit.loc[
            exact_match_mask
        ].itertuples()
    }

    fallback_candidate_mask = (
        ~exact_match_mask
        & match_audit[
            "nearest_phot_id"
        ].notna()
        & match_audit[
            "nearest_separation_arcsec"
        ].le(
            match_radius_arcsec
        )
    )

    fallback_candidate_rows = (
        match_audit.index[
            fallback_candidate_mask
        ]
    )

    for row_index in (
        fallback_candidate_rows
    ):
        candidate_key = (
            str(
                match_audit.at[
                    row_index,
                    "field",
                ]
            ),
            int(
                match_audit.at[
                    row_index,
                    "nearest_phot_id",
                ]
            ),
        )

        if candidate_key in exact_source_keys:
            match_audit.at[
                row_index,
                "blocked_by_exact_match",
            ] = True

    available_fallback_mask = (
        fallback_candidate_mask
        & ~match_audit[
            "blocked_by_exact_match"
        ]
    )

    fallback_candidates = (
        match_audit.loc[
            available_fallback_mask
        ]
        .copy()
    )

    if not fallback_candidates.empty:
        fallback_candidates[
            "_has_finite_z"
        ] = np.isfinite(
            fallback_candidates[
                "z_spec"
            ]
        )

        fallback_candidates[
            "_quality_rank"
        ] = _quality_rank(
            fallback_candidates[
                "z_spec_quality"
            ]
        )

        fallback_candidates = (
            fallback_candidates
            .sort_values(
                [
                    "field",
                    "nearest_phot_id",
                    "nearest_separation_arcsec",
                    "_has_finite_z",
                    "_quality_rank",
                    "prism_exposure_s",
                    "_input_row",
                ],
                ascending=[
                    True,
                    True,
                    True,
                    False,
                    True,
                    False,
                    True,
                ],
                kind="stable",
            )
        )

        fallback_candidates[
            "_selected_fallback"
        ] = ~fallback_candidates.duplicated(
            subset=[
                "field",
                "nearest_phot_id",
            ],
            keep="first",
        )

        accepted_fallback_rows = (
            fallback_candidates.index[
                fallback_candidates[
                    "_selected_fallback"
                ]
            ]
        )

        conflicting_fallback_rows = (
            fallback_candidates.index[
                ~fallback_candidates[
                    "_selected_fallback"
                ]
            ]
        )

        match_audit.loc[
            accepted_fallback_rows,
            "selected_phot_id",
        ] = match_audit.loc[
            accepted_fallback_rows,
            "nearest_phot_id",
        ].astype("Int64")

        match_audit.loc[
            accepted_fallback_rows,
            "selected_separation_arcsec",
        ] = match_audit.loc[
            accepted_fallback_rows,
            "nearest_separation_arcsec",
        ]

        match_audit.loc[
            accepted_fallback_rows,
            "match_method",
        ] = "nearest_sky"

        match_audit.loc[
            conflicting_fallback_rows,
            "fallback_conflict",
        ] = True

    accepted_match_mask = match_audit[
        "selected_phot_id"
    ].notna()

    accepted_keys = (
        match_audit.loc[
            accepted_match_mask,
            [
                "field",
                "selected_phot_id",
            ],
        ]
    )

    if accepted_keys.duplicated().any():
        raise ValueError(
            "Final matched source keys are not unique."
        )

    accepted_matches = (
        match_audit.loc[
            accepted_match_mask
        ]
        .copy()
    )

    accepted_matches[
        "selected_phot_id"
    ] = accepted_matches[
        "selected_phot_id"
    ].astype(np.int64)

    matched_catalog = (
        source_catalog.merge(
            accepted_matches,
            left_on=[
                "field",
                "phot_id",
            ],
            right_on=[
                "field",
                "selected_phot_id",
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    matched_catalog[
        "match_separation_arcsec"
    ] = matched_catalog[
        "selected_separation_arcsec"
    ]

    if len(matched_catalog) != len(
        accepted_matches
    ):
        raise ValueError(
            "Accepted matches were lost during the final merge."
        )

    return (
        matched_catalog,
        match_audit,
    )


def build_match_summaries(
    original_spectroscopy,
    selected_spectra,
    matched_catalog,
    match_audit,
):
    """Build compact tables describing the matching outcome."""

    number_all_rows = len(
        original_spectroscopy
    )

    number_positive_id_rows = int(
        original_spectroscopy[
            "has_positive_dr5_id"
        ].sum()
    )

    number_unique_positive_keys = (
        original_spectroscopy.loc[
            original_spectroscopy[
                "has_positive_dr5_id"
            ],
            [
                "field",
                "nircam_dr5_id",
            ],
        ]
        .drop_duplicates()
        .shape[0]
    )

    number_redundant_positive_rows = (
        number_positive_id_rows
        - number_unique_positive_keys
    )

    number_exact_matches = int(
        match_audit[
            "match_method"
        ].eq(
            "exact_dr5_id"
        ).sum()
    )

    number_sky_matches = int(
        match_audit[
            "match_method"
        ].eq(
            "nearest_sky"
        ).sum()
    )

    number_unmatched = int(
        match_audit[
            "match_method"
        ].eq(
            "unmatched"
        ).sum()
    )

    number_finite_z_spec = int(
        np.isfinite(
            matched_catalog[
                "z_spec"
            ]
        ).sum()
    )

    secure_quality_values = {
        str(value).strip().upper()
        for value in SECURE_SPEC_FLAGS
    }

    secure_spec_mask = (
        np.isfinite(
            matched_catalog[
                "z_spec"
            ]
        )
        & matched_catalog[
            "z_spec_quality"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(
            secure_quality_values
        )
    )

    number_secure_z_spec = int(
        secure_spec_mask.sum()
    )

    summary_records = [
        (
            "All DR4 spectroscopic rows",
            number_all_rows,
        ),
        (
            "Rows with a positive published DR5 ID",
            number_positive_id_rows,
        ),
        (
            "Unique positive field-DR5-ID keys",
            number_unique_positive_keys,
        ),
        (
            "Redundant positive-ID rows removed",
            number_redundant_positive_rows,
        ),
        (
            "Spectroscopic records entering the match",
            len(selected_spectra),
        ),
        (
            "Exact official DR5-ID matches",
            number_exact_matches,
        ),
        (
            "Accepted nearest-sky fallbacks",
            number_sky_matches,
        ),
        (
            "Unmatched spectroscopic records",
            number_unmatched,
        ),
        (
            "Final unique DR4-DR5 matches",
            len(matched_catalog),
        ),
        (
            "Matched sources with finite z_spec",
            number_finite_z_spec,
        ),
        (
            "Matched finite A/B/C z_spec",
            number_secure_z_spec,
        ),
    ]

    match_summary = pd.DataFrame(
        summary_records,
        columns=[
            "stage",
            "sources",
        ],
    )

    match_summary[
        "fraction_of_all_dr4_rows"
    ] = (
        match_summary["sources"]
        / number_all_rows
    )

    method_summary = (
        match_audit[
            "match_method"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "match_method"
        )
        .reset_index(
            name="sources"
        )
    )

    method_summary[
        "fraction_of_records_entering_match"
    ] = (
        method_summary["sources"]
        / len(selected_spectra)
    )

    return (
        match_summary,
        method_summary,
    )


def run_dr45_matching():
    """Run matching and save the compact DR4-DR5 products."""

    staging = load_dr45_staging()

    original_spectroscopy = (
        staging["spectroscopy"]
    )

    source_catalog = staging[
        "sources"
    ]

    (
        selected_spectra,
        duplicate_spectra,
    ) = select_best_spectra(
        original_spectroscopy
    )

    (
        matched_catalog,
        match_audit,
    ) = match_dr45_catalogs(
        selected_spectra,
        source_catalog,
    )

    (
        match_summary,
        method_summary,
    ) = build_match_summaries(
        original_spectroscopy,
        selected_spectra,
        matched_catalog,
        match_audit,
    )

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    matched_catalog.to_csv(
        MATCHED_CORE_PATH,
        index=False,
    )

    match_audit.to_csv(
        MATCH_AUDIT_PATH,
        index=False,
    )

    duplicate_spectra.to_csv(
        DUPLICATE_SPECTRA_PATH,
        index=False,
    )

    match_summary.to_csv(
        MATCH_SUMMARY_PATH,
        index=False,
    )

    method_summary.to_csv(
        MATCH_METHOD_SUMMARY_PATH,
        index=False,
    )

    return {
        "matched_catalog": matched_catalog,
        "match_audit": match_audit,
        "duplicate_spectra": duplicate_spectra,
        "match_summary": match_summary,
        "method_summary": method_summary,
    }


def main():
    """Run the matching pipeline from the command line."""

    results = run_dr45_matching()

    print(
        results[
            "match_summary"
        ].to_string(
            index=False
        )
    )

    print()

    print(
        results[
            "method_summary"
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()