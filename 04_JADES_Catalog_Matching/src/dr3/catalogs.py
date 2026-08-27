"""Load and standardize the official JADES DR3 GOODS-N catalogs."""

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from .config import (
    FILTERS,
    PHOTOMETRY_APERTURE,
    PHOTOMETRY_FLAG_EXTENSION,
    PHOTOMETRY_FLUX_EXTENSION,
    PHOTOMETRY_PATH,
    PHOTOMETRY_PHOTOZ_EXTENSION,
    SECURE_SPEC_FLAGS,
    SPECTROSCOPY_EXTENSION,
    SPECTROSCOPY_PATH,
)


def _numeric_column(
    table,
    column_name,
    dtype=np.float64,
):
    """Return a native-endian numeric copy of a FITS table column."""

    values = np.asarray(table[column_name])

    return values.astype(
        dtype,
        copy=True,
    )


def _text_column(
    table,
    column_name,
):
    """Return a stripped text copy of a FITS table column."""

    values = np.asarray(table[column_name])

    return np.char.strip(
        values.astype(str)
    )


def _require_columns(
    table,
    required_columns,
    table_name,
):
    """Raise an informative error if required columns are missing."""

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


def _check_catalog_file(
    catalog_path,
):
    """Validate that a catalog path points to an existing file."""

    catalog_path = Path(catalog_path)

    if not catalog_path.is_file():
        raise FileNotFoundError(
            f"Catalog file not found: {catalog_path}"
        )

    return catalog_path


def load_photometry_catalog(
    catalog_path=PHOTOMETRY_PATH,
):
    """Load the NIRCam fields required for matching and ML preparation."""

    catalog_path = _check_catalog_file(
        catalog_path
    )

    with fits.open(
        catalog_path,
        memmap=True,
    ) as phot_hdul:
        flag_table = phot_hdul[
            PHOTOMETRY_FLAG_EXTENSION
        ].data

        flux_table = phot_hdul[
            PHOTOMETRY_FLUX_EXTENSION
        ].data

        photoz_table = phot_hdul[
            PHOTOMETRY_PHOTOZ_EXTENSION
        ].data

        required_flag_columns = [
            "ID",
            "RA",
            "DEC",
            "FLAG_ST",
            "FLAG_BS",
            "FLAG_BN",
        ]

        required_photoz_columns = [
            "ID",
            "EAZY_z_a",
            "EAZY_chisq_min",
            "EAZY_nfilt",
        ]

        required_flux_columns = [
            "ID",
        ]

        for filter_name in FILTERS:
            required_flux_columns.extend(
                [
                    (
                        f"{filter_name}_"
                        f"{PHOTOMETRY_APERTURE}"
                    ),
                    (
                        f"{filter_name}_"
                        f"{PHOTOMETRY_APERTURE}_e"
                    ),
                    (
                        f"{filter_name}_"
                        f"{PHOTOMETRY_APERTURE}_ei"
                    ),
                ]
            )

        _require_columns(
            flag_table,
            required_flag_columns,
            PHOTOMETRY_FLAG_EXTENSION,
        )

        _require_columns(
            flux_table,
            required_flux_columns,
            PHOTOMETRY_FLUX_EXTENSION,
        )

        _require_columns(
            photoz_table,
            required_photoz_columns,
            PHOTOMETRY_PHOTOZ_EXTENSION,
        )

        flag_ids = _numeric_column(
            flag_table,
            "ID",
            dtype=np.int64,
        )

        flux_ids = _numeric_column(
            flux_table,
            "ID",
            dtype=np.int64,
        )

        photoz_ids = _numeric_column(
            photoz_table,
            "ID",
            dtype=np.int64,
        )

        extensions_are_aligned = (
            np.array_equal(
                flag_ids,
                flux_ids,
            )
            and np.array_equal(
                flag_ids,
                photoz_ids,
            )
        )

        if not extensions_are_aligned:
            raise ValueError(
                "The NIRCam catalog extensions "
                "are not aligned by source ID."
            )

        star_flag_raw = _numeric_column(
            flag_table,
            "FLAG_ST",
            dtype=np.uint16,
        )

        catalog_columns = {
            "phot_id": flag_ids,
            "ra_deg": _numeric_column(
                flag_table,
                "RA",
            ),
            "dec_deg": _numeric_column(
                flag_table,
                "DEC",
            ),
            "flag_star_raw": star_flag_raw,
            "is_flagged_star": (
                star_flag_raw == 1
            ),
            "flag_bright_star": _numeric_column(
                flag_table,
                "FLAG_BS",
                dtype=np.int8,
            ),
            "flag_bad_neighbor": _numeric_column(
                flag_table,
                "FLAG_BN",
                dtype=np.int8,
            ),
            "z_phot": _numeric_column(
                photoz_table,
                "EAZY_z_a",
            ),
            "z_phot_chisq": _numeric_column(
                photoz_table,
                "EAZY_chisq_min",
            ),
            "z_phot_n_filters": _numeric_column(
                photoz_table,
                "EAZY_nfilt",
            ),
        }

        for filter_name in FILTERS:
            filter_key = filter_name.lower()

            flux_column = (
                f"{filter_name}_"
                f"{PHOTOMETRY_APERTURE}"
            )

            nmad_error_column = (
                f"{filter_name}_"
                f"{PHOTOMETRY_APERTURE}_e"
            )

            pipeline_error_column = (
                f"{filter_name}_"
                f"{PHOTOMETRY_APERTURE}_ei"
            )

            catalog_columns[
                f"flux_{filter_key}_njy"
            ] = _numeric_column(
                flux_table,
                flux_column,
            )

            catalog_columns[
                f"fluxerr_{filter_key}_nmad_njy"
            ] = _numeric_column(
                flux_table,
                nmad_error_column,
            )

            catalog_columns[
                f"fluxerr_{filter_key}_pipeline_njy"
            ] = _numeric_column(
                flux_table,
                pipeline_error_column,
            )

    catalog = pd.DataFrame(
        catalog_columns
    )

    if not catalog["phot_id"].is_unique:
        raise ValueError(
            "The NIRCam photometric source IDs "
            "are not unique."
        )

    catalog.loc[
        catalog["z_phot"] < 0,
        "z_phot",
    ] = np.nan

    catalog["has_finite_coordinates"] = (
        np.isfinite(catalog["ra_deg"])
        & np.isfinite(catalog["dec_deg"])
    )

    return catalog


def load_spectroscopy_catalog(
    catalog_path=SPECTROSCOPY_PATH,
):
    """Load the NIRSpec fields required for catalog matching."""

    catalog_path = _check_catalog_file(
        catalog_path
    )

    with fits.open(
        catalog_path,
        memmap=True,
    ) as spec_hdul:
        spec_table = spec_hdul[
            SPECTROSCOPY_EXTENSION
        ].data

        required_columns = [
            "NIRSpec_ID",
            "TIER",
            "PID",
            "Field",
            "NIRCam_ID",
            "RA_TARG",
            "Dec_TARG",
            "RA_NIRCam",
            "Dec_NIRCam",
            "z_Spec",
            "z_Spec_flag",
            "DR_flag",
            "PRISM_flux_flag",
            "tExp_Prism",
        ]

        _require_columns(
            spec_table,
            required_columns,
            SPECTROSCOPY_EXTENSION,
        )

        catalog_columns = {
            "nirspec_id": _numeric_column(
                spec_table,
                "NIRSpec_ID",
                dtype=np.int64,
            ),
            "tier": _text_column(
                spec_table,
                "TIER",
            ),
            "program_id": _numeric_column(
                spec_table,
                "PID",
                dtype=np.int32,
            ),
            "field": _text_column(
                spec_table,
                "Field",
            ),
            "catalog_nircam_id": _numeric_column(
                spec_table,
                "NIRCam_ID",
                dtype=np.int64,
            ),
            "ra_target_deg": _numeric_column(
                spec_table,
                "RA_TARG",
            ),
            "dec_target_deg": _numeric_column(
                spec_table,
                "Dec_TARG",
            ),
            "ra_nircam_deg": _numeric_column(
                spec_table,
                "RA_NIRCam",
            ),
            "dec_nircam_deg": _numeric_column(
                spec_table,
                "Dec_NIRCam",
            ),
            "z_spec": _numeric_column(
                spec_table,
                "z_Spec",
            ),
            "z_spec_quality": _text_column(
                spec_table,
                "z_Spec_flag",
            ),
            "dr_problem": _numeric_column(
                spec_table,
                "DR_flag",
                dtype=bool,
            ),
            "prism_flux_problem": _numeric_column(
                spec_table,
                "PRISM_flux_flag",
                dtype=bool,
            ),
            "prism_exposure_s": _numeric_column(
                spec_table,
                "tExp_Prism",
            ),
        }

    catalog = pd.DataFrame(
        catalog_columns
    )

    catalog.loc[
        catalog["z_spec"] < 0,
        "z_spec",
    ] = np.nan

    catalog["has_positive_nircam_id"] = (
        catalog["catalog_nircam_id"] > 0
    )

    catalog["has_finite_nircam_coordinates"] = (
        np.isfinite(catalog["ra_nircam_deg"])
        & np.isfinite(catalog["dec_nircam_deg"])
    )

    catalog["is_secure_spec"] = (
        catalog["z_spec_quality"].isin(
            SECURE_SPEC_FLAGS
        )
        & catalog["z_spec"].notna()
        & ~catalog["dr_problem"]
    )

    return catalog