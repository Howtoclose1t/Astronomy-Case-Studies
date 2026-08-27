"""Load and standardize core JADES DR4 and DR5 catalog columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.io import fits

from .config import PROJECT_ROOT


RAW_DR45_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dr45"
)

DR4_SPECTROSCOPY_PATH = (
    RAW_DR45_DIR
    / "Combined_DR4_external_v1.2.1.fits"
)

DR5_CATALOG_PATHS = {
    "GN": (
        RAW_DR45_DIR
        / (
            "hlsp_jades_jwst_nircam_"
            "goods-n_photometry_v5.0_catalog.fits"
        )
    ),
    "GS": (
        RAW_DR45_DIR
        / (
            "hlsp_jades_jwst_nircam_"
            "goods-s_photometry_v5.0_catalog.fits"
        )
    ),
}

DR4_EXTENSION = "Obs_info"

DR5_FLAG_EXTENSION = "FLAG"
DR5_PHOTOZ_EXTENSION = "PHOTOZ"

VALID_FIELDS = (
    "GN",
    "GS",
)

EAZY_REDSHIFT_COLUMNS = (
    "eazy_z_a",
    "eazy_z_peak",
    "eazy_z500",
    "eazy_l68",
    "eazy_u68",
)


def _check_catalog_file(
    catalog_path,
):
    """Raise an informative error when a catalog file is missing."""

    if not catalog_path.is_file():
        raise FileNotFoundError(
            f"Catalog file not found: {catalog_path}"
        )

    return catalog_path


def _require_columns(
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

    values = np.asarray(
        table[column_name]
    )

    return values.astype(
        dtype,
        copy=True,
    )


def _text_column(
    table,
    column_name,
):
    """Return a stripped text copy of a FITS column."""

    values = np.asarray(
        table[column_name]
    )

    return np.char.strip(
        values.astype(str)
    )


def _normalize_field_value(
    field_value,
):
    """Convert catalog field labels to GN or GS."""

    normalized_value = (
        str(field_value)
        .strip()
        .upper()
        .replace("_", "-")
        .replace(" ", "")
    )

    field_aliases = {
        "GN": "GN",
        "GOODSN": "GN",
        "GOODS-N": "GN",
        "GS": "GS",
        "GOODSS": "GS",
        "GOODS-S": "GS",
    }

    if normalized_value not in field_aliases:
        raise ValueError(
            "Unrecognized JADES field label: "
            f"{field_value!r}"
        )

    return field_aliases[
        normalized_value
    ]


def _normalize_field_column(
    field_values,
):
    """Normalize every field label in a catalog column."""

    return np.asarray(
        [
            _normalize_field_value(
                field_value
            )
            for field_value in field_values
        ],
        dtype=object,
    )


def load_dr5_source_index(
    field,
    catalog_path=None,
):
    """Load DR5 source identity, coordinates, flags, and EAZY output."""

    field = _normalize_field_value(
        field
    )

    if catalog_path is None:
        catalog_path = (
            DR5_CATALOG_PATHS[field]
        )

    catalog_path = _check_catalog_file(
        catalog_path
    )

    with fits.open(
        catalog_path,
        mode="readonly",
        memmap=True,
        lazy_load_hdus=True,
    ) as photometry_hdul:
        flag_table = photometry_hdul[
            DR5_FLAG_EXTENSION
        ].data

        photoz_table = photometry_hdul[
            DR5_PHOTOZ_EXTENSION
        ].data

        required_flag_columns = [
            "ID",
            "RA",
            "DEC",
            "FLAG_BN",
            "PARENT_ID",
            "PID_HASH",
        ]

        required_photoz_columns = [
            "ID",
            "z_a",
            "z_ml",
            "chi_a",
            "l68",
            "u68",
            "nfilt",
            "z_peak",
            "z500",
        ]

        _require_columns(
            flag_table,
            required_flag_columns,
            f"{field} {DR5_FLAG_EXTENSION}",
        )

        _require_columns(
            photoz_table,
            required_photoz_columns,
            f"{field} {DR5_PHOTOZ_EXTENSION}",
        )

        flag_ids = _numeric_column(
            flag_table,
            "ID",
            dtype=np.int64,
        )

        photoz_ids = _numeric_column(
            photoz_table,
            "ID",
            dtype=np.int64,
        )

        if not np.array_equal(
            flag_ids,
            photoz_ids,
        ):
            raise ValueError(
                f"{field} FLAG and PHOTOZ "
                "extensions are not aligned by ID."
            )

        number_sources = len(
            flag_ids
        )

        catalog_columns = {
            "field": np.repeat(
                field,
                number_sources,
            ),
            "phot_id": flag_ids,
            "ra_deg": _numeric_column(
                flag_table,
                "RA",
            ),
            "dec_deg": _numeric_column(
                flag_table,
                "DEC",
            ),
            "flag_bright_neighbor": (
                _numeric_column(
                    flag_table,
                    "FLAG_BN",
                    dtype=np.int64,
                )
            ),
            "parent_id": _numeric_column(
                flag_table,
                "PARENT_ID",
                dtype=np.int64,
            ),
            "program_hash": _numeric_column(
                flag_table,
                "PID_HASH",
                dtype=np.int64,
            ),
            "eazy_z_a": _numeric_column(
                photoz_table,
                "z_a",
            ),
            "eazy_z_ml": _numeric_column(
                photoz_table,
                "z_ml",
            ),
            "eazy_chi_a": _numeric_column(
                photoz_table,
                "chi_a",
            ),
            "eazy_l68": _numeric_column(
                photoz_table,
                "l68",
            ),
            "eazy_u68": _numeric_column(
                photoz_table,
                "u68",
            ),
            "eazy_nfilt": _numeric_column(
                photoz_table,
                "nfilt",
                dtype=np.int64,
            ),
            "eazy_z_peak": _numeric_column(
                photoz_table,
                "z_peak",
            ),
            "eazy_z500": _numeric_column(
                photoz_table,
                "z500",
            ),
        }

    catalog = pd.DataFrame(
        catalog_columns
    )

    for column_name in (
        EAZY_REDSHIFT_COLUMNS
    ):
        catalog.loc[
            catalog[column_name] < 0,
            column_name,
        ] = np.nan

    catalog.loc[
        catalog["eazy_z_ml"] < 0,
        "eazy_z_ml",
    ] = np.nan

    catalog["has_finite_coordinates"] = (
        np.isfinite(catalog["ra_deg"])
        & np.isfinite(catalog["dec_deg"])
    )

    catalog["field"] = pd.Categorical(
        catalog["field"],
        categories=VALID_FIELDS,
    )

    if catalog.duplicated(
        subset=[
            "field",
            "phot_id",
        ]
    ).any():
        raise ValueError(
            f"Duplicate DR5 source keys found in {field}."
        )

    return catalog


def load_dr4_spectroscopy(
    catalog_path=DR4_SPECTROSCOPY_PATH,
):
    """Load the DR4 spectroscopic information required for matching."""

    catalog_path = _check_catalog_file(
        catalog_path
    )

    with fits.open(
        catalog_path,
        mode="readonly",
        memmap=True,
        lazy_load_hdus=True,
    ) as spectroscopy_hdul:
        spectroscopy_table = (
            spectroscopy_hdul[
                DR4_EXTENSION
            ].data
        )

        required_columns = [
            "Unique_ID",
            "PID",
            "TIER",
            "NIRSpec_ID",
            "NIRCam_DR5_ID",
            "NIRCam_DR3_ID",
            "RA_TARG",
            "Dec_TARG",
            "Field",
            "GSa",
            "GSb",
            "tExp_PRISM",
            "z_phot",
            "z_Spec",
            "z_R1000",
            "z_PRISM",
            "z_Spec_flag",
        ]

        _require_columns(
            spectroscopy_table,
            required_columns,
            DR4_EXTENSION,
        )

        raw_field_values = _text_column(
            spectroscopy_table,
            "Field",
        )

        catalog_columns = {
            "unique_spec_id": _text_column(
                spectroscopy_table,
                "Unique_ID",
            ),
            "program_id": _numeric_column(
                spectroscopy_table,
                "PID",
                dtype=np.int64,
            ),
            "tier": _text_column(
                spectroscopy_table,
                "TIER",
            ),
            "nirspec_id": _numeric_column(
                spectroscopy_table,
                "NIRSpec_ID",
                dtype=np.int64,
            ),
            "nircam_dr5_id": _numeric_column(
                spectroscopy_table,
                "NIRCam_DR5_ID",
                dtype=np.int64,
            ),
            "nircam_dr3_id": _numeric_column(
                spectroscopy_table,
                "NIRCam_DR3_ID",
                dtype=np.int64,
            ),
            "ra_target_deg": _numeric_column(
                spectroscopy_table,
                "RA_TARG",
            ),
            "dec_target_deg": _numeric_column(
                spectroscopy_table,
                "Dec_TARG",
            ),
            "field": _normalize_field_column(
                raw_field_values
            ),
            "is_goods_s_a": _numeric_column(
                spectroscopy_table,
                "GSa",
                dtype=bool,
            ),
            "is_goods_s_b": _numeric_column(
                spectroscopy_table,
                "GSb",
                dtype=bool,
            ),
            "prism_exposure_s": _numeric_column(
                spectroscopy_table,
                "tExp_PRISM",
            ),
            "dr4_z_phot": _numeric_column(
                spectroscopy_table,
                "z_phot",
            ),
            "z_spec": _numeric_column(
                spectroscopy_table,
                "z_Spec",
            ),
            "z_r1000": _numeric_column(
                spectroscopy_table,
                "z_R1000",
            ),
            "z_prism": _numeric_column(
                spectroscopy_table,
                "z_PRISM",
            ),
            "z_spec_quality": _text_column(
                spectroscopy_table,
                "z_Spec_flag",
            ),
        }

    catalog = pd.DataFrame(
        catalog_columns
    )

    redshift_columns = [
        "dr4_z_phot",
        "z_spec",
        "z_r1000",
        "z_prism",
    ]

    for column_name in redshift_columns:
        catalog.loc[
            catalog[column_name] < 0,
            column_name,
        ] = np.nan

    catalog["has_positive_dr5_id"] = (
        catalog["nircam_dr5_id"] > 0
    )

    catalog["has_finite_z_spec"] = (
        np.isfinite(catalog["z_spec"])
    )

    catalog[
        "has_finite_target_coordinates"
    ] = (
        np.isfinite(
            catalog["ra_target_deg"]
        )
        & np.isfinite(
            catalog["dec_target_deg"]
        )
    )

    catalog["field"] = pd.Categorical(
        catalog["field"],
        categories=VALID_FIELDS,
    )

    return catalog


def load_dr45_staging():
    """Load the complete standardized input for DR4-DR5 matching."""

    spectroscopy_catalog = (
        load_dr4_spectroscopy()
    )

    source_catalogs = [
        load_dr5_source_index(
            field
        )
        for field in VALID_FIELDS
    ]

    source_catalog = pd.concat(
        source_catalogs,
        ignore_index=True,
    )

    duplicate_source_keys = (
        source_catalog.duplicated(
            subset=[
                "field",
                "phot_id",
            ],
            keep=False,
        )
    )

    if duplicate_source_keys.any():
        duplicated_rows = (
            source_catalog.loc[
                duplicate_source_keys,
                [
                    "field",
                    "phot_id",
                ],
            ]
            .drop_duplicates()
        )

        raise ValueError(
            "Duplicate field-phot_id keys found "
            "after combining GOODS-N and GOODS-S: "
            f"{duplicated_rows.head().to_dict('records')}"
        )

    return {
        "spectroscopy": spectroscopy_catalog,
        "sources": source_catalog,
    }