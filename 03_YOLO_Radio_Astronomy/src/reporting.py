from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_training_results(run_dir: str | Path) -> pd.DataFrame:
    """Read Ultralytics results.csv if a training run exists."""
    path = Path(run_dir) / "results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def latest_metric_snapshot(results: pd.DataFrame) -> dict:
    """Return the last-row training and validation metrics for concise reporting."""
    if results.empty:
        return {}
    row = results.iloc[-1]
    wanted = [col for col in results.columns if any(key in col for key in ["loss", "precision", "recall", "mAP50"])]
    return {col.strip(): float(row[col]) for col in wanted if pd.notna(row[col])}
