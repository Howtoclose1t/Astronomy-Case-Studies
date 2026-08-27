"""Run the complete JADES catalog-matching and sample-preparation pipeline."""

import json

import pandas as pd

from .catalogs import (
    load_photometry_catalog,
    load_spectroscopy_catalog,
)
from .config import (
    INPUT_SUMMARY_PATH,
    MATCHED_CATALOG_PATH,
    MATCH_RADIUS_ARCSEC,
    MATCH_SUMMARY_PATH,
    METRICS_DIR,
    MIN_VALID_FILTERS,
    ML_READY_CATALOG_PATH,
    RADIUS_SCAN_PATH,
    SPEC_QUALITY_SUMMARY_PATH,
    ensure_output_directories,
)
from .matching import (
    add_photometric_validity,
    build_ml_ready_catalog,
    build_radius_scan,
    match_catalogs,
    select_best_spectrum,
)


MATCH_AUDIT_PATH = (
    METRICS_DIR
    / "match_audit.csv"
)

DUPLICATE_SPECTRA_PATH = (
    METRICS_DIR
    / "spectroscopic_duplicate_records.csv"
)


def _build_input_summary(
    photometry_catalog,
    spectroscopy_catalog,
):
    """Build a concise summary of the two input catalogs."""

    records = [
        {
            "catalog": "NIRCam photometry",
            "rows": len(photometry_catalog),
            "columns": photometry_catalog.shape[1],
            "rows_with_finite_coordinates": int(
                photometry_catalog[
                    "has_finite_coordinates"
                ].sum()
            ),
        },
        {
            "catalog": "NIRSpec spectroscopy",
            "rows": len(spectroscopy_catalog),
            "columns": spectroscopy_catalog.shape[1],
            "rows_with_finite_coordinates": int(
                spectroscopy_catalog[
                    "has_finite_nircam_coordinates"
                ].sum()
            ),
        },
    ]

    return pd.DataFrame(records)


def _build_spectroscopic_quality_summary(
    spectroscopy_catalog,
):
    """Summarize NIRSpec records by redshift-quality grade."""

    records = []

    grouped_catalog = spectroscopy_catalog.groupby(
        "z_spec_quality",
        dropna=False,
        sort=True,
    )

    for quality_grade, group in grouped_catalog:
        positive_id = (
            group["catalog_nircam_id"] > 0
        )

        if pd.isna(quality_grade):
            quality_label = "missing"
        else:
            quality_label = str(
                quality_grade
            )

        records.append(
            {
                "z_spec_quality": quality_label,
                "rows": len(group),
                "finite_z_spec": int(
                    group["z_spec"].notna().sum()
                ),
                "secure_without_dr_problem": int(
                    group["is_secure_spec"].sum()
                ),
                "reduction_problems": int(
                    group["dr_problem"].sum()
                ),
                "rows_with_positive_nircam_id": int(
                    positive_id.sum()
                ),
                "unique_positive_nircam_ids": int(
                    group.loc[
                        positive_id,
                        "catalog_nircam_id",
                    ].nunique()
                ),
            }
        )

    return pd.DataFrame(records)


def _build_match_summary(
    photometry_catalog,
    spectroscopy_catalog,
    best_spectra,
    duplicate_spectra,
    match_audit,
    quality_catalog,
    ml_ready_catalog,
):
    """Build headline pipeline counts for reports and notebooks."""

    positive_nircam_id = (
        spectroscopy_catalog[
            "catalog_nircam_id"
        ] > 0
    )

    secure_non_star = (
        quality_catalog["is_secure_spec"]
        & ~quality_catalog[
            "is_flagged_star"
        ]
    )

    summary = {
        "photometric_rows": len(
            photometry_catalog
        ),
        "spectroscopic_rows": len(
            spectroscopy_catalog
        ),
        "spectroscopic_rows_with_positive_nircam_id": int(
            positive_nircam_id.sum()
        ),
        "unique_positive_nircam_ids": int(
            spectroscopy_catalog.loc[
                positive_nircam_id,
                "catalog_nircam_id",
            ].nunique()
        ),
        "repeated_nircam_id_groups": int(
            duplicate_spectra[
                "catalog_nircam_id"
            ].nunique()
        ),
        "duplicate_spectroscopic_rows_removed": int(
            positive_nircam_id.sum()
            - len(best_spectra)
        ),
        "best_spectra_retained": len(
            best_spectra
        ),
        "published_ids_found_in_photometry": int(
            match_audit[
                "published_id_in_photometry"
            ].sum()
        ),
        "exact_id_matches": int(
            (
                match_audit["match_method"]
                == "exact_id"
            ).sum()
        ),
        "nearest_sky_fallback_matches": int(
            (
                match_audit["match_method"]
                == "nearest_sky"
            ).sum()
        ),
        "unmatched_spectroscopic_records": int(
            (
                match_audit["match_method"]
                == "unmatched"
            ).sum()
        ),
        "accepted_one_to_one_matches": len(
            quality_catalog
        ),
        "matches_with_finite_z_spec": int(
            quality_catalog[
                "z_spec"
            ].notna().sum()
        ),
        "secure_abc_matches": int(
            quality_catalog[
                "is_secure_spec"
            ].sum()
        ),
        "secure_non_star_matches": int(
            secure_non_star.sum()
        ),
        "ml_ready_sources": len(
            ml_ready_catalog
        ),
        "match_radius_arcsec": (
            MATCH_RADIUS_ARCSEC
        ),
        "minimum_valid_filters": (
            MIN_VALID_FILTERS
        ),
    }

    return summary


def run_matching():
    """Run the full pipeline, save outputs, and return generated objects."""

    ensure_output_directories()

    photometry_catalog = (
        load_photometry_catalog()
    )

    spectroscopy_catalog = (
        load_spectroscopy_catalog()
    )

    input_summary = _build_input_summary(
        photometry_catalog,
        spectroscopy_catalog,
    )

    spectroscopic_quality_summary = (
        _build_spectroscopic_quality_summary(
            spectroscopy_catalog
        )
    )

    best_spectra, duplicate_spectra = (
        select_best_spectrum(
            spectroscopy_catalog
        )
    )

    matched_catalog, match_audit = (
        match_catalogs(
            photometry_catalog=(
                photometry_catalog
            ),
            best_spectra=best_spectra,
            match_radius_arcsec=(
                MATCH_RADIUS_ARCSEC
            ),
        )
    )

    radius_scan = build_radius_scan(
        match_audit
    )

    quality_catalog = (
        add_photometric_validity(
            matched_catalog
        )
    )

    ml_ready_catalog = (
        build_ml_ready_catalog(
            quality_catalog
        )
    )

    match_summary = _build_match_summary(
        photometry_catalog=(
            photometry_catalog
        ),
        spectroscopy_catalog=(
            spectroscopy_catalog
        ),
        best_spectra=best_spectra,
        duplicate_spectra=(
            duplicate_spectra
        ),
        match_audit=match_audit,
        quality_catalog=(
            quality_catalog
        ),
        ml_ready_catalog=(
            ml_ready_catalog
        ),
    )

    quality_catalog.to_csv(
        MATCHED_CATALOG_PATH,
        index=False,
    )

    ml_ready_catalog.to_csv(
        ML_READY_CATALOG_PATH,
        index=False,
    )

    input_summary.to_csv(
        INPUT_SUMMARY_PATH,
        index=False,
    )

    spectroscopic_quality_summary.to_csv(
        SPEC_QUALITY_SUMMARY_PATH,
        index=False,
    )

    radius_scan.to_csv(
        RADIUS_SCAN_PATH,
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

    MATCH_SUMMARY_PATH.write_text(
        json.dumps(
            match_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "photometry_catalog": (
            photometry_catalog
        ),
        "spectroscopy_catalog": (
            spectroscopy_catalog
        ),
        "best_spectra": best_spectra,
        "duplicate_spectra": (
            duplicate_spectra
        ),
        "match_audit": match_audit,
        "matched_catalog": (
            quality_catalog
        ),
        "ml_ready_catalog": (
            ml_ready_catalog
        ),
        "radius_scan": radius_scan,
        "match_summary": match_summary,
    }


def main():
    """Run the pipeline from the command line."""

    results = run_matching()

    summary = results[
        "match_summary"
    ]

    print(
        "JADES catalog-matching pipeline complete."
    )

    print(
        "Accepted one-to-one matches: "
        f"{summary['accepted_one_to_one_matches']:,}"
    )

    print(
        "Secure A/B/C matches: "
        f"{summary['secure_abc_matches']:,}"
    )

    print(
        "ML-ready sources: "
        f"{summary['ml_ready_sources']:,}"
    )

    print(
        f"Matched catalog: {MATCHED_CATALOG_PATH}"
    )

    print(
        f"ML-ready catalog: {ML_READY_CATALOG_PATH}"
    )

    print(
        f"Summary: {MATCH_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()