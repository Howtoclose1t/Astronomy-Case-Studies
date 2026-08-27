"""Inspect the structure of the JADES DR5 and DR4 FITS catalogs."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from astropy.io import fits


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dr45"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
)

CATALOG_PATHS = {
    "DR5 GOODS-N photometry": (
        RAW_DATA_DIR
        / (
            "hlsp_jades_jwst_nircam_"
            "goods-n_photometry_v5.0_catalog.fits"
        )
    ),
    "DR5 GOODS-S photometry": (
        RAW_DATA_DIR
        / (
            "hlsp_jades_jwst_nircam_"
            "goods-s_photometry_v5.0_catalog.fits"
        )
    ),
    "DR4 spectroscopy": (
        RAW_DATA_DIR
        / "Combined_DR4_external_v1.2.1.fits"
    ),
}

HDU_SUMMARY_PATH = (
    METRICS_DIR
    / "dr45_hdu_summary.csv"
)

COLUMN_INVENTORY_PATH = (
    METRICS_DIR
    / "dr45_column_inventory.csv"
)

CORE_COLUMNS_PATH = (
    METRICS_DIR
    / "dr45_core_columns.csv"
)

FILTER_COVERAGE_PATH = (
    METRICS_DIR
    / "dr45_filter_coverage.csv"
)

FILTER_PATTERN = re.compile(
    r"F\d{3}[A-Z]",
    flags=re.IGNORECASE,
)


def classify_column(column_name):
    """Assign a scientific role to a FITS-table column."""

    normalized_name = (
        column_name
        .strip()
        .lower()
    )

    redshift_terms = (
        "z_spec",
        "zspec",
        "specz",
        "z_phot",
        "zphot",
        "photoz",
        "redshift",
    )

    coordinate_terms = (
        "ra",
        "dec",
        "right_ascension",
        "declination",
    )

    quality_terms = (
        "quality",
        "flag",
        "problem",
        "star",
        "stellar",
        "chi2",
        "chisq",
        "q_z",
        "odds",
        "use_phot",
    )

    identifier_terms = (
        "id",
        "source_id",
        "catalog_id",
        "target_id",
        "nircam_id",
        "nirspec_id",
    )

    photometry_terms = (
        "flux",
        "fnu",
        "mag",
        "error",
        "err",
        "snr",
        "aperture",
        "kron",
    )

    if any(
        term in normalized_name
        for term in redshift_terms
    ):
        return "redshift"

    if (
        normalized_name in coordinate_terms
        or normalized_name.startswith("ra_")
        or normalized_name.startswith("dec_")
        or normalized_name.endswith("_ra")
        or normalized_name.endswith("_dec")
    ):
        return "coordinate"

    if any(
        term in normalized_name
        for term in quality_terms
    ):
        return "quality"

    if (
        normalized_name in identifier_terms
        or normalized_name.endswith("_id")
        or normalized_name.startswith("id_")
    ):
        return "identifier"

    if any(
        term in normalized_name
        for term in photometry_terms
    ):
        return "photometry"

    return "other"


def inspect_catalog(
    catalog_name,
    catalog_path,
):
    """Inspect HDUs and columns without loading full catalog arrays."""

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Catalog not found: {catalog_path}"
        )

    file_size_mb = (
        catalog_path.stat().st_size
        / 1024**2
    )

    hdu_records = []
    column_records = []

    with fits.open(
        catalog_path,
        mode="readonly",
        memmap=True,
        lazy_load_hdus=True,
    ) as hdul:
        for hdu_index, hdu in enumerate(hdul):
            extension_name = (
                hdu.name
                if hdu.name
                else "PRIMARY"
            )

            is_table = isinstance(
                hdu,
                (
                    fits.BinTableHDU,
                    fits.TableHDU,
                ),
            )

            number_rows = (
                int(hdu.header.get("NAXIS2", 0))
                if is_table
                else 0
            )

            number_columns = (
                len(hdu.columns)
                if is_table
                else 0
            )

            hdu_records.append(
                {
                    "catalog": catalog_name,
                    "filename": catalog_path.name,
                    "file_size_mb": file_size_mb,
                    "hdu_index": hdu_index,
                    "extension": extension_name,
                    "hdu_type": type(hdu).__name__,
                    "rows": number_rows,
                    "columns": number_columns,
                }
            )

            if not is_table:
                continue

            for column in hdu.columns:
                column_name = str(
                    column.name
                )

                detected_filters = sorted(
                    {
                        match.upper()
                        for match
                        in FILTER_PATTERN.findall(
                            column_name
                        )
                    }
                )

                column_records.append(
                    {
                        "catalog": catalog_name,
                        "filename": catalog_path.name,
                        "hdu_index": hdu_index,
                        "extension": extension_name,
                        "column": column_name,
                        "format": str(
                            column.format
                        ),
                        "unit": (
                            ""
                            if column.unit is None
                            else str(column.unit)
                        ),
                        "category": classify_column(
                            column_name
                        ),
                        "filters": ";".join(
                            detected_filters
                        ),
                    }
                )

    return (
        pd.DataFrame(hdu_records),
        pd.DataFrame(column_records),
    )


def build_filter_coverage(
    column_inventory,
):
    """Summarize filters represented in each catalog extension."""

    filter_records = []

    for row in column_inventory.itertuples(
        index=False
    ):
        if not row.filters:
            continue

        for filter_name in row.filters.split(";"):
            filter_records.append(
                {
                    "catalog": row.catalog,
                    "extension": row.extension,
                    "filter": filter_name,
                    "column": row.column,
                }
            )

    if not filter_records:
        return pd.DataFrame(
            columns=[
                "catalog",
                "extension",
                "filter",
                "related_columns",
            ]
        )

    filter_columns = pd.DataFrame(
        filter_records
    )

    filter_coverage = (
        filter_columns
        .groupby(
            [
                "catalog",
                "extension",
                "filter",
            ],
            as_index=False,
        )
        .agg(
            related_columns=(
                "column",
                "nunique",
            )
        )
        .sort_values(
            [
                "catalog",
                "extension",
                "filter",
            ]
        )
        .reset_index(drop=True)
    )

    return filter_coverage


def run_inventory():
    """Inspect all three catalogs and save compact schema tables."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    hdu_tables = []
    column_tables = []

    for catalog_name, catalog_path in (
        CATALOG_PATHS.items()
    ):
        (
            hdu_summary,
            column_inventory,
        ) = inspect_catalog(
            catalog_name,
            catalog_path,
        )

        hdu_tables.append(
            hdu_summary
        )
        column_tables.append(
            column_inventory
        )

    combined_hdu_summary = pd.concat(
        hdu_tables,
        ignore_index=True,
    )

    combined_column_inventory = pd.concat(
        column_tables,
        ignore_index=True,
    )

    core_columns = (
        combined_column_inventory.loc[
            combined_column_inventory[
                "category"
            ].isin(
                [
                    "identifier",
                    "coordinate",
                    "redshift",
                    "quality",
                ]
            )
        ]
        .sort_values(
            [
                "catalog",
                "hdu_index",
                "category",
                "column",
            ]
        )
        .reset_index(drop=True)
    )

    filter_coverage = build_filter_coverage(
        combined_column_inventory
    )

    combined_hdu_summary.to_csv(
        HDU_SUMMARY_PATH,
        index=False,
    )

    combined_column_inventory.to_csv(
        COLUMN_INVENTORY_PATH,
        index=False,
    )

    core_columns.to_csv(
        CORE_COLUMNS_PATH,
        index=False,
    )

    filter_coverage.to_csv(
        FILTER_COVERAGE_PATH,
        index=False,
    )

    return {
        "hdu_summary": combined_hdu_summary,
        "core_columns": core_columns,
        "filter_coverage": filter_coverage,
        "column_inventory_path": (
            COLUMN_INVENTORY_PATH
        ),
    }


def main():
    """Run the inventory from a command line."""

    results = run_inventory()

    print(
        results["hdu_summary"].to_string(
            index=False
        )
    )

    print(
        "\nFull column inventory saved to:"
    )
    print(
        results["column_inventory_path"]
    )


if __name__ == "__main__":
    main()