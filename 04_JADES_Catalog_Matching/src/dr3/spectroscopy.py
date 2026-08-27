"""Load and summarize the JADES DR3 GOODS-N NIRSpec catalog."""

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


SECURE_SPEC_FLAGS = frozenset({"A", "B", "C"})


def _numeric_column(table, column_name: str, dtype=np.float64) -> np.ndarray:
    """Return a native-endian numeric copy of a FITS table column."""
    values = np.asarray(table[column_name])
    return values.astype(dtype, copy=True)


def _text_column(table, column_name: str) -> np.ndarray:
    """Return a stripped text copy of a FITS table column."""
    values = np.asarray(table[column_name])
    return np.char.strip(values.astype(str))


def load_spectroscopy_catalog(catalog_path: str | Path) -> pd.DataFrame:
    """Load the NIRSpec fields required for matching and redshift selection."""
    catalog_path = Path(catalog_path)

    with fits.open(catalog_path, memmap=True) as spec_hdul:
        spec_table = spec_hdul["Joined"].data

        columns = {
            "nirspec_id": _numeric_column(
                spec_table, "NIRSpec_ID", dtype=np.int64
            ),
            "tier": _text_column(spec_table, "TIER"),
            "program_id": _numeric_column(
                spec_table, "PID", dtype=np.int32
            ),
            "catalog_nircam_id": _numeric_column(
                spec_table, "NIRCam_ID", dtype=np.int64
            ),
            "ra_target_deg": _numeric_column(spec_table, "RA_TARG"),
            "dec_target_deg": _numeric_column(spec_table, "Dec_TARG"),
            "ra_nircam_deg": _numeric_column(spec_table, "RA_NIRCam"),
            "dec_nircam_deg": _numeric_column(spec_table, "Dec_NIRCam"),
            "z_spec": _numeric_column(spec_table, "z_Spec"),
            "z_spec_quality": _text_column(spec_table, "z_Spec_flag"),
            "dr_problem": _numeric_column(spec_table, "DR_flag", dtype=bool),
            "prism_flux_problem": _numeric_column(
                spec_table, "PRISM_flux_flag", dtype=bool
            ),
            "prism_exposure_s": _numeric_column(spec_table, "tExp_Prism"),
        }

    catalog = pd.DataFrame(columns)
    catalog.loc[catalog["z_spec"] < 0, "z_spec"] = np.nan
    catalog["is_secure_spec"] = (
        catalog["z_spec_quality"].isin(SECURE_SPEC_FLAGS)
        & catalog["z_spec"].notna()
        & ~catalog["dr_problem"]
    )
    return catalog


def summarize_spectroscopy_catalog(
    catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return overall and redshift-quality summaries for the catalog."""
    positive_nircam_id = catalog["catalog_nircam_id"] > 0
    finite_nircam_coordinates = (
        np.isfinite(catalog["ra_nircam_deg"])
        & np.isfinite(catalog["dec_nircam_deg"])
    )
    unique_positive_ids = catalog.loc[
        positive_nircam_id, "catalog_nircam_id"
    ].nunique()

    overall = pd.Series(
        {
            "spectroscopy_rows": len(catalog),
            "rows_with_positive_nircam_id": int(positive_nircam_id.sum()),
            "unique_positive_nircam_ids": int(unique_positive_ids),
            "duplicate_positive_id_rows": int(
                positive_nircam_id.sum() - unique_positive_ids
            ),
            "rows_with_finite_nircam_coordinates": int(
                finite_nircam_coordinates.sum()
            ),
            "rows_without_nearby_nircam_source": int(
                (catalog["catalog_nircam_id"] == -9999).sum()
            ),
            "rows_outside_nircam_footprint": int(
                (catalog["catalog_nircam_id"] == -1111).sum()
            ),
            "rows_with_valid_z_spec": int(catalog["z_spec"].notna().sum()),
            "secure_abc_rows_without_dr_problem": int(
                catalog["is_secure_spec"].sum()
            ),
        },
        name="value",
    ).to_frame()

    by_quality = catalog.groupby("z_spec_quality").agg(
        rows=("nirspec_id", "size"),
        valid_redshifts=("z_spec", "count"),
        reduction_problems=("dr_problem", "sum"),
        unique_positive_nircam_ids=(
            "catalog_nircam_id",
            lambda values: values[values > 0].nunique(),
        ),
    )
    return overall, by_quality

