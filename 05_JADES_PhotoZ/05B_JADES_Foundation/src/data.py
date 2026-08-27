# -*- coding: utf-8 -*-

"""Prepare frozen Task 5 data for the TabPFN residual experiment."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    EAZY_AUXILIARY_COLUMNS,
    EAZY_DIAGNOSTIC_FEATURES,
    EAZY_MODEL_FEATURE,
    EAZY_REDSHIFT_COLUMN,
    EXPECTED_FULL_FEATURE_COUNT,
    EXPECTED_SPLIT_COUNTS,
    FEATURE_STATE_PATH,
    RESIDUAL_TARGET_COLUMN,
    SOURCE_KEY_COLUMN,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    TRAIN_FEATURES_PATH,
    VALIDATION_FEATURES_PATH,
)


@dataclass(frozen=True)
class TabPFNDevelopmentData:
    """Container holding the training and validation data."""

    training_catalog: pd.DataFrame
    validation_catalog: pd.DataFrame
    feature_columns: tuple[str, ...]
    x_train: pd.DataFrame
    y_train: np.ndarray
    x_validation: pd.DataFrame
    y_validation: np.ndarray


def _numeric_series(series: pd.Series) -> pd.Series:
    """Convert one catalog column to numeric values."""

    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )


def load_feature_columns(
    feature_state_path: Path = FEATURE_STATE_PATH,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Restore the frozen feature definitions created in Task 5."""

    if not feature_state_path.exists():
        raise FileNotFoundError(
            f"Task 5 feature state was not found: "
            f"{feature_state_path}"
        )

    with feature_state_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        feature_state = json.load(file)

    full_feature_columns = tuple(
        feature_state.get("full_feature_columns", [])
    )

    stored_hybrid_columns = tuple(
        feature_state.get("hybrid_feature_columns", [])
    )

    if len(full_feature_columns) != EXPECTED_FULL_FEATURE_COUNT:
        raise ValueError(
            "The Task 5 feature state does not contain "
            f"{EXPECTED_FULL_FEATURE_COUNT} full features."
        )

    if len(set(full_feature_columns)) != len(full_feature_columns):
        raise ValueError(
            "Duplicated feature names were found in the "
            "Task 5 feature state."
        )

    expected_hybrid_columns = (
        full_feature_columns
        + (EAZY_MODEL_FEATURE,)
    )

    if stored_hybrid_columns != expected_hybrid_columns:
        raise ValueError(
            "The stored Task 5 hybrid features do not match "
            "the expected full-features-plus-EAZY definition."
        )

    model_feature_columns = (
        stored_hybrid_columns
        + EAZY_DIAGNOSTIC_FEATURES
    )

    return full_feature_columns, model_feature_columns


def add_eazy_diagnostic_features(
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Create EAZY confidence and estimator-disagreement features."""

    result = catalog.copy()

    z_a = _numeric_series(result[EAZY_MODEL_FEATURE])
    z_ml = _numeric_series(result["eazy_z_ml"])
    chi_a = _numeric_series(result["eazy_chi_a"])
    lower_68 = _numeric_series(result["eazy_l68"])
    upper_68 = _numeric_series(result["eazy_u68"])
    number_of_filters = _numeric_series(result["eazy_nfilt"])
    z_peak = _numeric_series(result["eazy_z_peak"])
    z_500 = _numeric_series(result["eazy_z500"])

    interval_width = (
        upper_68 - lower_68
    ).clip(lower=0.0)

    # EAZY 主红移的 log(1 + z) 表示
    result["x_eazy_log1p_za"] = np.log1p(
        z_a.clip(lower=0.0)
    )

    # 对数化 EAZY 拟合的 chi-square，压缩极端值
    result["x_eazy_log_chi"] = np.log1p(
        chi_a.clip(lower=0.0)
    )

    # EAZY 拟合过程中实际使用的滤镜数量
    result["x_eazy_nfilt"] = number_of_filters

    # EAZY 68% 红移置信区间的绝对宽度
    result["x_eazy_interval_width"] = interval_width

    # 相对于 1 + z_EAZY 的置信区间宽度
    result["x_eazy_normalized_interval_width"] = (
        interval_width / (1.0 + z_a)
    )

    # EAZY 不同红移估计之间的标准化分歧
    result["x_eazy_delta_za_zml"] = (
        (z_a - z_ml) / (1.0 + z_a)
    )

    result["x_eazy_delta_za_zpeak"] = (
        (z_a - z_peak) / (1.0 + z_a)
    )

    result["x_eazy_delta_za_z500"] = (
        (z_a - z_500) / (1.0 + z_a)
    )

    return result


def load_development_catalog(
    path: Path,
    expected_split: str,
    catalog_name: str,
    full_feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Load and validate one physical-source catalog."""

    if not path.exists():
        raise FileNotFoundError(
            f"{catalog_name} was not found: {path}"
        )

    catalog = pd.read_csv(
        path,
        low_memory=False,
    )

    required_columns = {
        SOURCE_KEY_COLUMN,
        TARGET_COLUMN,
        SPLIT_COLUMN,
        EAZY_REDSHIFT_COLUMN,
        EAZY_MODEL_FEATURE,
        *EAZY_AUXILIARY_COLUMNS,
        *full_feature_columns,
    }

    missing_columns = sorted(
        required_columns - set(catalog.columns)
    )

    if missing_columns:
        raise KeyError(
            f"{catalog_name} is missing columns: "
            f"{missing_columns}"
        )

    if not catalog[SPLIT_COLUMN].eq(expected_split).all():
        raise ValueError(
            f"{catalog_name} contains rows outside the "
            f"{expected_split} split."
        )

    expected_sources = EXPECTED_SPLIT_COUNTS[expected_split]

    if len(catalog) != expected_sources:
        raise ValueError(
            f"{catalog_name} contains {len(catalog)} rows; "
            f"{expected_sources} were expected."
        )

    if not catalog[SOURCE_KEY_COLUMN].is_unique:
        raise ValueError(
            f"{catalog_name} contains duplicated physical sources."
        )

    catalog[TARGET_COLUMN] = _numeric_series(
        catalog[TARGET_COLUMN]
    )

    catalog[EAZY_REDSHIFT_COLUMN] = _numeric_series(
        catalog[EAZY_REDSHIFT_COLUMN]
    )

    catalog[EAZY_MODEL_FEATURE] = _numeric_series(
        catalog[EAZY_MODEL_FEATURE]
    )

    if not np.isfinite(catalog[TARGET_COLUMN]).all():
        raise ValueError(
            f"{catalog_name} contains non-finite z_spec values."
        )

    if not np.isfinite(catalog[EAZY_MODEL_FEATURE]).all():
        raise ValueError(
            f"{catalog_name} contains non-finite EAZY redshifts."
        )

    if (catalog[TARGET_COLUMN] < 0).any():
        raise ValueError(
            f"{catalog_name} contains negative z_spec values."
        )

    if (catalog[EAZY_MODEL_FEATURE] < 0).any():
        raise ValueError(
            f"{catalog_name} contains negative EAZY redshifts."
        )

    eazy_columns_agree = np.allclose(
        catalog[EAZY_REDSHIFT_COLUMN],
        catalog[EAZY_MODEL_FEATURE],
        rtol=0.0,
        atol=1e-8,
        equal_nan=True,
    )

    if not eazy_columns_agree:
        raise ValueError(
            "The original and engineered EAZY redshift "
            f"columns disagree in {catalog_name}."
        )

    return add_eazy_diagnostic_features(catalog)


def prepare_feature_matrix(
    catalog: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Construct a numeric TabPFN feature matrix."""

    missing_columns = sorted(
        set(feature_columns) - set(catalog.columns)
    )

    if missing_columns:
        raise KeyError(
            "TabPFN features are missing: "
            f"{missing_columns}"
        )

    feature_matrix = (
        catalog.loc[:, list(feature_columns)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype(np.float32)
    )

    completely_missing_columns = (
        feature_matrix.columns[
            feature_matrix.isna().all()
        ].tolist()
    )

    if completely_missing_columns:
        raise ValueError(
            "Some TabPFN features are completely missing: "
            f"{completely_missing_columns}"
        )

    return feature_matrix


def calculate_log_residual_target(
    catalog: pd.DataFrame,
) -> np.ndarray:
    """Calculate the EAZY log-redshift residual."""

    z_spec = catalog[TARGET_COLUMN].to_numpy(
        dtype=np.float64
    )

    z_eazy = catalog[EAZY_MODEL_FEATURE].to_numpy(
        dtype=np.float64
    )

    # TabPFN 学习的目标：
    # r = log(1 + z_spec) - log(1 + z_EAZY)
    residual_target = (
        np.log1p(z_spec)
        - np.log1p(z_eazy)
    )

    if not np.isfinite(residual_target).all():
        raise ValueError(
            "The residual target contains non-finite values."
        )

    return residual_target.astype(np.float32)


def prepare_tabpfn_development_data(
) -> TabPFNDevelopmentData:
    """Prepare training and validation data without opening the test set."""

    (
        full_feature_columns,
        model_feature_columns,
    ) = load_feature_columns()

    training_catalog = load_development_catalog(
        path=TRAIN_FEATURES_PATH,
        expected_split="train",
        catalog_name="weighted physical training catalog",
        full_feature_columns=full_feature_columns,
    )

    validation_catalog = load_development_catalog(
        path=VALIDATION_FEATURES_PATH,
        expected_split="validation",
        catalog_name="validation catalog",
        full_feature_columns=full_feature_columns,
    )

    overlapping_sources = (
        set(training_catalog[SOURCE_KEY_COLUMN])
        & set(validation_catalog[SOURCE_KEY_COLUMN])
    )

    if overlapping_sources:
        raise ValueError(
            "Training-validation source leakage was detected."
        )

    x_train = prepare_feature_matrix(
        training_catalog,
        model_feature_columns,
    )

    x_validation = prepare_feature_matrix(
        validation_catalog,
        model_feature_columns,
    )

    y_train = calculate_log_residual_target(
        training_catalog
    )

    y_validation = calculate_log_residual_target(
        validation_catalog
    )

    return TabPFNDevelopmentData(
        training_catalog=training_catalog,
        validation_catalog=validation_catalog,
        feature_columns=model_feature_columns,
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
    )