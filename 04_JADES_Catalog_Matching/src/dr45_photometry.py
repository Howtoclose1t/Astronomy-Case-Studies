"""Extract DR5 NIRCam photometry for the matched DR4 sources."""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.io import fits

from .config import (
    METRICS_DIR,
    MIN_VALID_FILTERS,
    PROCESSED_DATA_DIR,
    SECURE_SPEC_FLAGS,
)
from .dr45_matching import (
    MATCHED_CORE_PATH,
)
from .dr45_staging import (
    DR5_CATALOG_PATHS,
    VALID_FIELDS,
)


PHOTOMETRY_EXTENSION = "CIRC"
FLAG_EXTENSION = "FLAG"
PHOTOMETRY_APERTURE = "CIRC1"

NIRCAM_FILTERS = (
    "F070W",
    "F090W",
    "F115W",
    "F150W",
    "F162M",
    "F182M",
    "F200W",
    "F210M",
    "F250M",
    "F277W",
    "F300M",
    "F335M",
    "F356W",
    "F410M",
    "F430M",
    "F444W",
    "F460M",
    "F480M",
)

MATCHED_PHOTOMETRY_PATH = (
    PROCESSED_DATA_DIR
    / "jades_dr4_dr5_matched_photometry.csv"
)

PHOTOMETRY_SUMMARY_PATH = (
    METRICS_DIR
    / "dr45_photometry_summary.csv"
)

FILTER_COVERAGE_PATH = (
    METRICS_DIR
    / "dr45_matched_filter_coverage.csv"
)


def _check_file(
    file_path,
):
    """Raise an informative error when an input file is missing."""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

    return file_path


def _require_dataframe_columns(
    dataframe,
    required_columns,
    dataframe_name,
):
    """Check that a DataFrame contains all required columns."""

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


def _require_fits_columns(
    table,
    required_columns,
    table_name,
):
    """Check that a FITS table contains all required columns."""

    available_columns = set(
        table.columns.names
    )

    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name not in available_columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing columns in {table_name}: "
            f"{missing_columns}"
        )


def _numeric_column(
    table,
    column_name,
    dtype=np.float64,
):
    """Return a native-endian numeric copy of a FITS column."""

    return np.asarray(
        table[column_name]
    ).astype(
        dtype,
        copy=True,
    )


def load_matched_core(
    matched_core_path=(
        MATCHED_CORE_PATH
    ),
):
    """Load and validate the core DR4-DR5 matched catalog."""

    matched_core_path = _check_file(
        matched_core_path
    )

    matched_core = pd.read_csv(
        matched_core_path
    )

    required_columns = [
        "field",
        "phot_id",
        "z_spec",
        "z_spec_quality",
        "eazy_z_a",
    ]

    _require_dataframe_columns(
        matched_core,
        required_columns,
        "matched_core",
    )

    matched_core["field"] = (
        matched_core["field"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    unknown_fields = sorted(
        set(
            matched_core["field"]
        )
        - set(VALID_FIELDS)
    )

    if unknown_fields:
        raise ValueError(
            "Unexpected field labels in matched core: "
            f"{unknown_fields}"
        )

    matched_core["phot_id"] = (
        matched_core["phot_id"]
        .astype(np.int64)
    )

    if matched_core.duplicated(
        subset=[
            "field",
            "phot_id",
        ]
    ).any():
        raise ValueError(
            "Matched core contains duplicate "
            "field-phot_id keys."
        )

    return matched_core


def extract_field_photometry(
    field,
    matched_core,
):
    """Extract CIRC1 measurements for matched sources in one field."""

    catalog_path = _check_file(
        DR5_CATALOG_PATHS[field]
    )

    matched_field = (
        matched_core.loc[
            matched_core[
                "field"
            ].eq(field),
            [
                "field",
                "phot_id",
            ],
        ]
        .copy()
        .reset_index(drop=True)
    )

    matched_ids = matched_field[
        "phot_id"
    ].to_numpy(
        dtype=np.int64
    )

    with fits.open(
        catalog_path,
        mode="readonly",
        memmap=True,
        lazy_load_hdus=True,
    ) as photometry_hdul:
        photometry_table = (
            photometry_hdul[
                PHOTOMETRY_EXTENSION
            ].data
        )

        flag_table = photometry_hdul[
            FLAG_EXTENSION
        ].data

        _require_fits_columns(
            photometry_table,
            ["ID"],
            (
                f"{field} "
                f"{PHOTOMETRY_EXTENSION}"
            ),
        )

        _require_fits_columns(
            flag_table,
            ["ID"],
            f"{field} {FLAG_EXTENSION}",
        )

        photometry_ids = _numeric_column(
            photometry_table,
            "ID",
            dtype=np.int64,
        )

        flag_ids = _numeric_column(
            flag_table,
            "ID",
            dtype=np.int64,
        )

        if not np.array_equal(
            photometry_ids,
            flag_ids,
        ):
            raise ValueError(
                f"{field} CIRC and FLAG extensions "
                "are not aligned by source ID."
            )

        catalog_id_index = pd.Index(
            photometry_ids
        )

        matched_row_positions = (
            catalog_id_index.get_indexer(
                matched_ids
            )
        )

        missing_id_mask = (
            matched_row_positions < 0
        )

        if missing_id_mask.any():
            missing_ids = matched_ids[
                missing_id_mask
            ]

            raise ValueError(
                f"{field} matched IDs missing "
                "from the CIRC extension: "
                f"{missing_ids[:10].tolist()}"
            )

        output_columns = {
            "field": np.repeat(
                field,
                len(matched_ids),
            ),
            "phot_id": matched_ids,
        }

        coverage_records = []

        available_photometry_columns = set(
            photometry_table.columns.names
        )

        available_flag_columns = set(
            flag_table.columns.names
        )

        for filter_name in (
            NIRCAM_FILTERS
        ):
            filter_key = (
                filter_name.lower()
            )

            flux_fits_column = (
                f"{filter_name}_"
                f"{PHOTOMETRY_APERTURE}"
            )

            error_fits_column = (
                f"{flux_fits_column}_e"
            )

            flag_fits_column = (
                f"{filter_name}_FLAG"
            )

            exposure_fits_column = (
                f"{filter_name}_TEXP"
            )

            flux_output_column = (
                f"flux_{filter_key}_njy"
            )

            error_output_column = (
                f"fluxerr_{filter_key}_njy"
            )

            flag_output_column = (
                f"phot_flag_{filter_key}"
            )

            exposure_output_column = (
                f"exposure_{filter_key}_s"
            )

            valid_output_column = (
                f"valid_{filter_key}"
            )

            has_flux_columns = (
                flux_fits_column
                in available_photometry_columns
                and error_fits_column
                in available_photometry_columns
            )

            has_flag_columns = (
                flag_fits_column
                in available_flag_columns
                and exposure_fits_column
                in available_flag_columns
            )

            catalog_has_filter = (
                has_flux_columns
                and has_flag_columns
            )

            if catalog_has_filter:
                flux_values = (
                    _numeric_column(
                        photometry_table,
                        flux_fits_column,
                    )[
                        matched_row_positions
                    ]
                )

                error_values = (
                    _numeric_column(
                        photometry_table,
                        error_fits_column,
                    )[
                        matched_row_positions
                    ]
                )

                flag_values = (
                    _numeric_column(
                        flag_table,
                        flag_fits_column,
                    )[
                        matched_row_positions
                    ]
                )

                exposure_values = (
                    _numeric_column(
                        flag_table,
                        exposure_fits_column,
                    )[
                        matched_row_positions
                    ]
                )
            else:
                number_matched = len(
                    matched_ids
                )

                flux_values = np.full(
                    number_matched,
                    np.nan,
                )

                error_values = np.full(
                    number_matched,
                    np.nan,
                )

                flag_values = np.full(
                    number_matched,
                    np.nan,
                )

                exposure_values = np.full(
                    number_matched,
                    np.nan,
                )

            valid_values = (
                np.isfinite(
                    flux_values
                )
                & np.isfinite(
                    error_values
                )
                & (error_values > 0)
                & np.isfinite(
                    exposure_values
                )
                & (exposure_values > 0)
            )

            output_columns[
                flux_output_column
            ] = flux_values

            output_columns[
                error_output_column
            ] = error_values

            output_columns[
                flag_output_column
            ] = flag_values

            output_columns[
                exposure_output_column
            ] = exposure_values

            output_columns[
                valid_output_column
            ] = valid_values

            number_valid = int(
                valid_values.sum()
            )

            number_matched = len(
                matched_ids
            )

            if number_matched > 0:
                valid_fraction = (
                    number_valid
                    / number_matched
                )
            else:
                valid_fraction = np.nan

            coverage_records.append(
                {
                    "field": field,
                    "filter": filter_name,
                    "catalog_has_filter": (
                        catalog_has_filter
                    ),
                    "matched_sources": (
                        number_matched
                    ),
                    "valid_measurements": (
                        number_valid
                    ),
                    "valid_fraction": (
                        valid_fraction
                    ),
                }
            )

    field_photometry = pd.DataFrame(
        output_columns
    )

    filter_coverage = pd.DataFrame(
        coverage_records
    )

    return (
        field_photometry,
        filter_coverage,
    )


def add_photometry_validity(
    matched_photometry,
):
    """Add valid-filter counts and model-candidate flags."""

    catalog = matched_photometry.copy()

    valid_columns = [
        f"valid_{filter_name.lower()}"
        for filter_name in NIRCAM_FILTERS
    ]

    _require_dataframe_columns(
        catalog,
        valid_columns,
        "matched_photometry",
    )

    catalog["n_valid_filters"] = (
        catalog[
            valid_columns
        ]
        .sum(axis=1)
        .astype(np.int16)
    )

    secure_quality_values = {
        str(value)
        .strip()
        .upper()
        for value in SECURE_SPEC_FLAGS
    }

    normalized_quality = (
        catalog[
            "z_spec_quality"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    catalog[
        "has_secure_z_spec"
    ] = (
        np.isfinite(
            catalog["z_spec"]
        )
        & normalized_quality.isin(
            secure_quality_values
        )
    )

    catalog[
        "has_eazy_prediction"
    ] = np.isfinite(
        catalog["eazy_z_a"]
    )

    catalog[
        "is_model_candidate"
    ] = (
        catalog[
            "has_secure_z_spec"
        ]
        & catalog[
            "n_valid_filters"
        ].ge(
            MIN_VALID_FILTERS
        )
    )

    catalog[
        "is_hybrid_candidate"
    ] = (
        catalog[
            "is_model_candidate"
        ]
        & catalog[
            "has_eazy_prediction"
        ]
    )

    return catalog


def build_photometry_summary(
    matched_photometry,
):
    """Build the sample-flow table after photometry extraction."""

    total_matched = len(
        matched_photometry
    )

    finite_z_mask = np.isfinite(
        matched_photometry[
            "z_spec"
        ]
    )

    secure_z_mask = (
        matched_photometry[
            "has_secure_z_spec"
        ]
    )

    model_candidate_mask = (
        matched_photometry[
            "is_model_candidate"
        ]
    )

    hybrid_candidate_mask = (
        matched_photometry[
            "is_hybrid_candidate"
        ]
    )

    stages = [
        (
            "Final unique DR4-DR5 matches",
            total_matched,
        ),
        (
            "Matched sources with finite z_spec",
            int(
                finite_z_mask.sum()
            ),
        ),
        (
            "Matched finite A/B/C z_spec",
            int(
                secure_z_mask.sum()
            ),
        ),
        (
            (
                "A/B/C z_spec with at least "
                f"{MIN_VALID_FILTERS} valid filters"
            ),
            int(
                model_candidate_mask.sum()
            ),
        ),
        (
            (
                "Model candidates with finite "
                "DR5 EAZY z_a"
            ),
            int(
                hybrid_candidate_mask.sum()
            ),
        ),
    ]

    summary = pd.DataFrame(
        stages,
        columns=[
            "stage",
            "sources",
        ],
    )

    summary[
        "removed_from_previous"
    ] = (
        summary["sources"]
        .shift(1)
        .sub(
            summary["sources"]
        )
        .fillna(0)
        .astype(int)
    )

    if total_matched > 0:
        summary[
            "fraction_of_matched_sources"
        ] = (
            summary["sources"]
            / total_matched
        )
    else:
        summary[
            "fraction_of_matched_sources"
        ] = np.nan

    return summary


def run_dr45_photometry():
    """Extract matched photometry and save compact products."""

    matched_core = load_matched_core()

    field_photometry_tables = []
    filter_coverage_tables = []

    for field in VALID_FIELDS:
        (
            field_photometry,
            field_filter_coverage,
        ) = extract_field_photometry(
            field,
            matched_core,
        )

        field_photometry_tables.append(
            field_photometry
        )

        filter_coverage_tables.append(
            field_filter_coverage
        )

    extracted_photometry = pd.concat(
        field_photometry_tables,
        ignore_index=True,
    )

    filter_coverage = pd.concat(
        filter_coverage_tables,
        ignore_index=True,
    )

    matched_photometry = (
        matched_core.merge(
            extracted_photometry,
            on=[
                "field",
                "phot_id",
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    if len(matched_photometry) != len(
        matched_core
    ):
        raise ValueError(
            "Matched sources were lost during "
            "photometry extraction."
        )

    matched_photometry = (
        add_photometry_validity(
            matched_photometry
        )
    )

    photometry_summary = (
        build_photometry_summary(
            matched_photometry
        )
    )

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    matched_photometry.to_csv(
        MATCHED_PHOTOMETRY_PATH,
        index=False,
    )

    photometry_summary.to_csv(
        PHOTOMETRY_SUMMARY_PATH,
        index=False,
    )

    filter_coverage.to_csv(
        FILTER_COVERAGE_PATH,
        index=False,
    )

    return {
        "matched_photometry": (
            matched_photometry
        ),
        "photometry_summary": (
            photometry_summary
        ),
        "filter_coverage": (
            filter_coverage
        ),
    }


def main():
    """Run matched-source photometry extraction."""

    results = run_dr45_photometry()

    print(
        results[
            "photometry_summary"
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()