from pathlib import Path

import pandas as pd

CATALOG_COLUMNS = [
    "id",
    "ra_core_deg",
    "dec_core_deg",
    "ra_centroid_deg",
    "dec_centroid_deg",
    "flux_jy",
    "core_fraction",
    "bmaj_arcsec",
    "bmin_arcsec",
    "pa_deg",
    "size_type",
    "class_id",
    "selection",
    "x",
    "y",
]

CLASS_ID_TO_NAME = {1: "SS-AGN", 2: "FS-AGN", 3: "SFG"}


def read_sdc1_catalog(path: str | Path, selected_only: bool = True) -> pd.DataFrame:
    """Read an SDC1 training catalogue into a tidy dataframe."""
    path = Path(path)
    header_rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.lstrip().startswith("ID"):
                break
            header_rows += 1
    table = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=header_rows + 1,
        names=CATALOG_COLUMNS,
        engine="python",
    )
    if selected_only:
        table = table.loc[table["selection"] == 1].copy()
    table["class_index"] = table["class_id"].astype(int) - 1
    table["class_name"] = table["class_id"].map(CLASS_ID_TO_NAME).fillna("unknown")
    return table.reset_index(drop=True)


def summarize_catalog(table: pd.DataFrame) -> dict:
    """Return compact catalogue statistics for notebooks and logs."""
    return {
        "sources": int(len(table)),
        "flux_min_jy": float(table["flux_jy"].min()),
        "flux_median_jy": float(table["flux_jy"].median()),
        "flux_max_jy": float(table["flux_jy"].max()),
        "classes": table["class_name"].value_counts().to_dict(),
    }
