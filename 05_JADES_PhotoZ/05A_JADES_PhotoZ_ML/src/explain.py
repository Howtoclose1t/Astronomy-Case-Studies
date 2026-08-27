"""Explain the frozen residual and gate models with native TreeSHAP."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

from .config import (
    FIGURE_DIR,
    METRICS_DIR,
    SOURCE_KEY_COLUMN,
    TARGET_COLUMN,
    ensure_output_directories,
)
from .features import (
    EAZY_HYBRID_FEATURE,
    TEST_FEATURES_PATH,
)
from .hybrid import (
    ABSOLUTE_PREDICTED_RESIDUAL_FEATURE,
    EAZY_AUXILIARY_INPUT_COLUMNS,
    GATE_FEATURE_COLUMNS,
    GATE_MODEL_PATH,
    PREDICTED_RESIDUAL_FEATURE,
    RESIDUAL_FEATURE_COLUMNS,
    RESIDUAL_MODEL_PATH,
    add_eazy_diagnostic_features,
    prepare_matrix,
)
from .test import (
    FINAL_TEST_PREDICTIONS_PATH,
)


EAZY_EXPERIMENT_ID = "eazy_dr5_za"
HYBRID_EXPERIMENT_ID = (
    "xgb_eazy_gated_residual"
)

CATASTROPHIC_OUTLIER_THRESHOLD = 0.15

GLOBAL_IMPORTANCE_PATH = (
    METRICS_DIR
    / "dr45_tree_shap_global_importance.csv"
)

EXPLANATION_CASES_PATH = (
    METRICS_DIR
    / "dr45_tree_shap_cases.csv"
)

EXPLANATION_MANIFEST_PATH = (
    METRICS_DIR
    / "dr45_tree_shap_manifest.json"
)

EXPLANATION_FIGURE_PATH = (
    FIGURE_DIR
    / "dr45_tree_shap_explanations.png"
)


FEATURE_LABELS = {
    "x_n_valid_core_filters": (
        "Number of valid core filters"
    ),
    "x_eazy_z_a": (
        "EAZY z_a"
    ),
    "x_eazy_log1p_za": (
        "log(1 + EAZY z_a)"
    ),
    "x_eazy_log_chi": (
        "log(1 + EAZY chi-square)"
    ),
    "x_eazy_nfilt": (
        "EAZY number of filters"
    ),
    "x_eazy_interval_width": (
        "EAZY 68% interval width"
    ),
    "x_eazy_normalized_interval_width": (
        "Normalized EAZY interval width"
    ),
    "x_eazy_delta_za_zml": (
        "EAZY z_a minus z_ml"
    ),
    "x_eazy_delta_za_zpeak": (
        "EAZY z_a minus z_peak"
    ),
    "x_eazy_delta_za_z500": (
        "EAZY z_a minus z500"
    ),
    PREDICTED_RESIDUAL_FEATURE: (
        "Predicted EAZY log-residual"
    ),
    ABSOLUTE_PREDICTED_RESIDUAL_FEATURE: (
        "Absolute predicted log-residual"
    ),
}


def readable_feature_label(
    feature_name: str,
) -> str:
    """Convert internal feature names into plot-friendly labels."""

    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]

    if feature_name.startswith(
        "x_color_"
    ):
        filter_names = (
            feature_name
            .removeprefix("x_color_")
            .upper()
            .split("_")
        )

        return (
            f"Color {filter_names[0]}"
            f" - {filter_names[1]}"
        )

    if feature_name.startswith(
        "x_asinh_mag_"
    ):
        filter_name = (
            feature_name
            .removeprefix("x_asinh_mag_")
            .upper()
        )

        return (
            f"Asinh magnitude {filter_name}"
        )

    if feature_name.startswith(
        "x_log_snr_"
    ):
        filter_name = (
            feature_name
            .removeprefix("x_log_snr_")
            .upper()
        )

        return (
            f"log S/N {filter_name}"
        )

    if feature_name.startswith(
        "x_fluxerr_"
    ):
        filter_name = (
            feature_name
            .removeprefix("x_fluxerr_")
            .removesuffix("_njy")
            .upper()
        )

        return (
            f"Flux uncertainty {filter_name}"
        )

    if feature_name.startswith(
        "x_flux_"
    ):
        filter_name = (
            feature_name
            .removeprefix("x_flux_")
            .removesuffix("_njy")
            .upper()
        )

        return f"Flux {filter_name}"

    if feature_name.startswith(
        "x_valid_"
    ):
        filter_name = (
            feature_name
            .removeprefix("x_valid_")
            .upper()
        )

        return (
            f"Valid measurement {filter_name}"
        )

    return feature_name


def _coerce_boolean_series(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """Convert boolean-like values without treating strings as True."""

    if pd.api.types.is_bool_dtype(
        series.dtype
    ):
        return series.fillna(False).astype(
            bool
        )

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    converted = normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }
    )

    if converted.isna().any():
        unexpected = (
            series.loc[converted.isna()]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{column_name} contains invalid values: "
            f"{unexpected[:10]}"
        )

    return converted.astype(bool)


def load_test_catalog() -> pd.DataFrame:
    """Load and validate the frozen test feature catalog."""

    if not TEST_FEATURES_PATH.exists():
        raise FileNotFoundError(
            "The test feature catalog was not found at "
            f"{TEST_FEATURES_PATH}"
        )

    catalog = pd.read_csv(
        TEST_FEATURES_PATH,
        low_memory=False,
    )

    raw_required_columns = {
        SOURCE_KEY_COLUMN,
        TARGET_COLUMN,
        EAZY_HYBRID_FEATURE,
        *EAZY_AUXILIARY_INPUT_COLUMNS,
    }

    missing_raw_columns = sorted(
        raw_required_columns - set(catalog.columns)
    )

    if missing_raw_columns:
        raise KeyError(
            "The test feature catalog is missing raw columns: "
            f"{missing_raw_columns}"
        )

    if not catalog[SOURCE_KEY_COLUMN].is_unique:
        raise ValueError(
            "The test feature catalog contains duplicate sources."
        )

    catalog = add_eazy_diagnostic_features(
        catalog
    )

    missing_model_features = sorted(
        set(RESIDUAL_FEATURE_COLUMNS)
        - set(catalog.columns)
    )

    if missing_model_features:
        raise KeyError(
            "The derived test catalog is missing model features: "
            f"{missing_model_features}"
        )

    return catalog


def load_residual_model() -> XGBRegressor:
    """Load the frozen EAZY residual regressor."""

    if not RESIDUAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            "The residual model was not found at "
            f"{RESIDUAL_MODEL_PATH}"
        )

    model = XGBRegressor()
    model.load_model(
        RESIDUAL_MODEL_PATH
    )

    return model


def load_gate_model() -> XGBClassifier:
    """Load the frozen correction gate."""

    if not GATE_MODEL_PATH.exists():
        raise FileNotFoundError(
            "The gate model was not found at "
            f"{GATE_MODEL_PATH}"
        )

    model = XGBClassifier()
    model.load_model(
        GATE_MODEL_PATH
    )

    return model


def model_iteration_range(
    model,
) -> tuple[int, int]:
    """Match TreeSHAP calculations to the model's best iteration."""

    booster_attributes = (
        model.get_booster().attributes()
    )

    best_iteration = (
        booster_attributes.get(
            "best_iteration"
        )
    )

    if best_iteration is None:
        return 0, 0

    return (
        0,
        int(best_iteration) + 1,
    )


def calculate_native_tree_shap(
    model,
    feature_matrix: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate exact native TreeSHAP values and verify additivity."""

    booster = model.get_booster()

    dmatrix = xgb.DMatrix(
        feature_matrix,
        feature_names=list(
            feature_matrix.columns
        ),
    )

    iteration_range = (
        model_iteration_range(model)
    )

    contributions = booster.predict(
        dmatrix,
        pred_contribs=True,
        iteration_range=iteration_range,
    )

    shap_values = contributions[
        :,
        :-1,
    ]

    base_values = contributions[
        :,
        -1,
    ]

    model_margin = booster.predict(
        dmatrix,
        output_margin=True,
        iteration_range=iteration_range,
    )

    reconstructed_margin = (
        base_values
        + shap_values.sum(axis=1)
    )

    if not np.allclose(
        model_margin,
        reconstructed_margin,
        rtol=1e-4,
        atol=1e-5,
    ):
        raise RuntimeError(
            "TreeSHAP values do not reconstruct model output."
        )

    return shap_values, base_values


def build_global_importance(
    shap_values: np.ndarray,
    feature_matrix: pd.DataFrame,
    model_name: str,
    output_unit: str,
) -> pd.DataFrame:
    """Aggregate local SHAP magnitudes into global importance."""

    mean_absolute_shap = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    mean_signed_shap = np.mean(
        shap_values,
        axis=0,
    )

    missing_fraction = (
        feature_matrix.isna()
        .mean(axis=0)
        .to_numpy(dtype=float)
    )

    importance = pd.DataFrame(
        {
            "model": model_name,
            "output_unit": output_unit,
            "feature": (
                feature_matrix.columns
            ),
            "feature_label": [
                readable_feature_label(
                    feature_name
                )
                for feature_name
                in feature_matrix.columns
            ],
            "mean_absolute_shap": (
                mean_absolute_shap
            ),
            "mean_signed_shap": (
                mean_signed_shap
            ),
            "missing_fraction": (
                missing_fraction
            ),
        }
    )

    importance = importance.sort_values(
        "mean_absolute_shap",
        ascending=False,
    ).reset_index(drop=True)

    importance.insert(
        2,
        "rank",
        np.arange(
            1,
            len(importance) + 1,
        ),
    )

    return importance


def load_paired_case_predictions() -> pd.DataFrame:
    """Align EAZY and gated-hybrid test predictions for case selection."""

    if not FINAL_TEST_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "The final test predictions were not found at "
            f"{FINAL_TEST_PREDICTIONS_PATH}"
        )

    predictions = pd.read_csv(
        FINAL_TEST_PREDICTIONS_PATH,
        low_memory=False,
    )

    eazy = predictions.loc[
        predictions["experiment_id"]
        == EAZY_EXPERIMENT_ID,
        [
            SOURCE_KEY_COLUMN,
            TARGET_COLUMN,
            "z_prediction",
        ],
    ].rename(
        columns={
            TARGET_COLUMN: "z_spec_eazy",
            "z_prediction": "z_eazy",
        }
    )

    hybrid = predictions.loc[
        predictions["experiment_id"]
        == HYBRID_EXPERIMENT_ID,
        [
            SOURCE_KEY_COLUMN,
            TARGET_COLUMN,
            "z_prediction",
            "correction_applied",
            "gate_probability",
        ],
    ].rename(
        columns={
            TARGET_COLUMN: "z_spec_hybrid",
            "z_prediction": "z_hybrid",
        }
    )

    paired = eazy.merge(
        hybrid,
        on=SOURCE_KEY_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    if not np.allclose(
        paired["z_spec_eazy"],
        paired["z_spec_hybrid"],
    ):
        raise ValueError(
            "EAZY and hybrid z_spec values are not aligned."
        )

    paired[TARGET_COLUMN] = (
        paired["z_spec_eazy"]
    )

    paired = paired.drop(
        columns=[
            "z_spec_eazy",
            "z_spec_hybrid",
        ]
    )

    paired[
        "correction_applied"
    ] = _coerce_boolean_series(
        paired["correction_applied"],
        "correction_applied",
    )

    numeric_columns = (
        TARGET_COLUMN,
        "z_eazy",
        "z_hybrid",
        "gate_probability",
    )

    for column in numeric_columns:
        paired[column] = pd.to_numeric(
            paired[column],
            errors="coerce",
        )

    paired["eazy_normalized_error"] = (
        (
            paired["z_eazy"]
            - paired[TARGET_COLUMN]
        )
        / (
            1.0
            + paired[TARGET_COLUMN]
        )
    )

    paired["hybrid_normalized_error"] = (
        (
            paired["z_hybrid"]
            - paired[TARGET_COLUMN]
        )
        / (
            1.0
            + paired[TARGET_COLUMN]
        )
    )

    paired[
        "eazy_absolute_normalized_error"
    ] = paired[
        "eazy_normalized_error"
    ].abs()

    paired[
        "hybrid_absolute_normalized_error"
    ] = paired[
        "hybrid_normalized_error"
    ].abs()

    paired[
        "normalized_error_improvement"
    ] = (
        paired[
            "eazy_absolute_normalized_error"
        ]
        - paired[
            "hybrid_absolute_normalized_error"
        ]
    )

    paired["eazy_outlier"] = (
        paired[
            "eazy_absolute_normalized_error"
        ]
        > CATASTROPHIC_OUTLIER_THRESHOLD
    )

    paired["hybrid_outlier"] = (
        paired[
            "hybrid_absolute_normalized_error"
        ]
        > CATASTROPHIC_OUTLIER_THRESHOLD
    )

    paired["case_type"] = "non_catastrophic"

    paired.loc[
        paired["eazy_outlier"]
        & ~paired["hybrid_outlier"],
        "case_type",
    ] = "fixed_catastrophic"

    paired.loc[
        ~paired["eazy_outlier"]
        & paired["hybrid_outlier"],
        "case_type",
    ] = "introduced_catastrophic"

    paired.loc[
        paired["eazy_outlier"]
        & paired["hybrid_outlier"],
        "case_type",
    ] = "shared_catastrophic"

    return paired


def select_explanation_cases(
    paired_cases: pd.DataFrame,
) -> pd.DataFrame:
    """Select fixed, shared, improved, and harmed examples."""

    selected_parts = []

    fixed_cases = paired_cases.loc[
        paired_cases["case_type"]
        == "fixed_catastrophic"
    ].sort_values(
        "normalized_error_improvement",
        ascending=False,
    )

    if not fixed_cases.empty:
        fixed_cases = fixed_cases.copy()
        fixed_cases[
            "selected_case_group"
        ] = "Fixed catastrophic outlier"

        selected_parts.append(
            fixed_cases
        )

    introduced_cases = paired_cases.loc[
        paired_cases["case_type"]
        == "introduced_catastrophic"
    ].sort_values(
        "hybrid_absolute_normalized_error",
        ascending=False,
    )

    if not introduced_cases.empty:
        introduced_cases = (
            introduced_cases.copy()
        )

        introduced_cases[
            "selected_case_group"
        ] = "Introduced catastrophic outlier"

        selected_parts.append(
            introduced_cases
        )

    shared_cases = paired_cases.loc[
        paired_cases["case_type"]
        == "shared_catastrophic"
    ].nlargest(
        2,
        "eazy_absolute_normalized_error",
    ).copy()

    if not shared_cases.empty:
        shared_cases[
            "selected_case_group"
        ] = "Shared catastrophic outlier"

        selected_parts.append(
            shared_cases
        )

    non_catastrophic = paired_cases.loc[
        paired_cases["case_type"]
        == "non_catastrophic"
    ]

    largest_improvements = (
        non_catastrophic.nlargest(
            2,
            "normalized_error_improvement",
        ).copy()
    )

    largest_improvements[
        "selected_case_group"
    ] = "Largest non-catastrophic improvement"

    selected_parts.append(
        largest_improvements
    )

    harmed_candidates = (
        non_catastrophic.loc[
            non_catastrophic[
                "correction_applied"
            ]
        ]
    )

    largest_harms = (
        harmed_candidates.nsmallest(
            2,
            "normalized_error_improvement",
        ).copy()
    )

    if not largest_harms.empty:
        largest_harms[
            "selected_case_group"
        ] = "Largest non-catastrophic harm"

        selected_parts.append(
            largest_harms
        )

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    selected = selected.drop_duplicates(
        subset=[SOURCE_KEY_COLUMN],
        keep="first",
    )

    display_columns = [
        "selected_case_group",
        SOURCE_KEY_COLUMN,
        TARGET_COLUMN,
        "z_eazy",
        "z_hybrid",
        "eazy_absolute_normalized_error",
        "hybrid_absolute_normalized_error",
        "normalized_error_improvement",
        "correction_applied",
        "gate_probability",
    ]

    return selected.loc[
        :,
        display_columns,
    ]


def representative_source_key(
    explanation_cases: pd.DataFrame,
) -> str:
    """Choose one fixed outlier for a local SHAP explanation."""

    fixed = explanation_cases.loc[
        explanation_cases[
            "selected_case_group"
        ]
        == "Fixed catastrophic outlier"
    ]

    if not fixed.empty:
        return str(
            fixed.iloc[0][
                SOURCE_KEY_COLUMN
            ]
        )

    return str(
        explanation_cases.sort_values(
            "normalized_error_improvement",
            ascending=False,
        ).iloc[0][SOURCE_KEY_COLUMN]
    )


def _plot_global_importance(
    axis,
    importance: pd.DataFrame,
    title: str,
    color: str,
    top_features: int = 12,
) -> None:
    """Draw a global mean-absolute-SHAP bar chart."""

    plotting_table = (
        importance.head(
            top_features
        )
        .iloc[::-1]
    )

    axis.barh(
        plotting_table[
            "feature_label"
        ],
        plotting_table[
            "mean_absolute_shap"
        ],
        color=color,
        alpha=0.85,
    )

    axis.set_xlabel(
        "Mean absolute SHAP contribution"
    )

    axis.set_title(title)

    axis.grid(
        axis="x",
        alpha=0.20,
    )


def _plot_local_contributions(
    axis,
    shap_values: np.ndarray,
    feature_matrix: pd.DataFrame,
    row_position: int,
    title: str,
    x_label: str,
    top_features: int = 10,
) -> None:
    """Draw the strongest signed SHAP contributions for one source."""

    source_shap = shap_values[
        row_position
    ]

    strongest_indices = np.argsort(
        np.abs(source_shap)
    )[
        -top_features:
    ]

    strongest_indices = (
        strongest_indices[
            np.argsort(
                np.abs(
                    source_shap[
                        strongest_indices
                    ]
                )
            )
        ]
    )

    contribution_values = (
        source_shap[
            strongest_indices
        ]
    )

    feature_labels = [
        readable_feature_label(
            feature_matrix.columns[index]
        )
        for index in strongest_indices
    ]

    colors = np.where(
        contribution_values >= 0,
        "tab:green",
        "tab:orange",
    )

    axis.barh(
        feature_labels,
        contribution_values,
        color=colors,
        alpha=0.85,
    )

    axis.axvline(
        0.0,
        color="black",
        linewidth=1.0,
    )

    axis.set_xlabel(x_label)
    axis.set_title(title)

    axis.grid(
        axis="x",
        alpha=0.20,
    )


def build_explanation_figure(
    residual_importance: pd.DataFrame,
    gate_importance: pd.DataFrame,
    residual_shap: np.ndarray,
    gate_shap: np.ndarray,
    residual_matrix: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    test_catalog: pd.DataFrame,
    paired_cases: pd.DataFrame,
    source_key: str,
):
    """Build global and local explanations in one final figure."""

    source_positions = np.flatnonzero(
        test_catalog[
            SOURCE_KEY_COLUMN
        ].astype(str).to_numpy()
        == source_key
    )

    if len(source_positions) != 1:
        raise ValueError(
            "The representative source could not be uniquely aligned."
        )

    row_position = int(
        source_positions[0]
    )

    source_case = paired_cases.loc[
        paired_cases[
            SOURCE_KEY_COLUMN
        ].astype(str)
        == source_key
    ].iloc[0]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15, 12),
        constrained_layout=True,
    )

    _plot_global_importance(
        axes[0, 0],
        residual_importance,
        title=(
            "Residual model: global importance"
        ),
        color="tab:blue",
    )

    _plot_global_importance(
        axes[0, 1],
        gate_importance,
        title=(
            "Gate model: global importance"
        ),
        color="tab:purple",
    )

    _plot_local_contributions(
        axes[1, 0],
        residual_shap,
        residual_matrix,
        row_position,
        title=(
            "Representative corrected source:\n"
            "residual contributions"
        ),
        x_label=(
            "Contribution to predicted log-redshift residual"
        ),
    )

    _plot_local_contributions(
        axes[1, 1],
        gate_shap,
        gate_matrix,
        row_position,
        title=(
            "Representative corrected source:\n"
            "gate contributions"
        ),
        x_label=(
            "Contribution to gate log-odds"
        ),
    )

    figure.suptitle(
        (
            f"TreeSHAP explanation for {source_key}\n"
            f"z_spec={source_case[TARGET_COLUMN]:.3f}, "
            f"EAZY={source_case['z_eazy']:.3f}, "
            f"hybrid={source_case['z_hybrid']:.3f}, "
            f"gate probability="
            f"{source_case['gate_probability']:.3f}"
        ),
        fontsize=14,
    )

    figure.savefig(
        EXPLANATION_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return figure


def save_explanation_products(
    global_importance: pd.DataFrame,
    explanation_cases: pd.DataFrame,
    representative_key: str,
) -> None:
    """Save explanation tables and their interpretation metadata."""

    ensure_output_directories()

    global_importance.to_csv(
        GLOBAL_IMPORTANCE_PATH,
        index=False,
    )

    explanation_cases.to_csv(
        EXPLANATION_CASES_PATH,
        index=False,
    )

    manifest = {
        "method": (
            "Native XGBoost TreeSHAP contributions"
        ),
        "residual_output_unit": (
            "Predicted log-redshift residual"
        ),
        "gate_output_unit": (
            "Binary gate log-odds"
        ),
        "representative_source_key": (
            representative_key
        ),
        "important_limitation": (
            "SHAP attributes model output; it does not "
            "establish physical causality."
        ),
    }

    with EXPLANATION_MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            manifest,
            file_handle,
            indent=2,
        )


def run_tree_shap_explanations(
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Explain the frozen residual and gate models on blind-test sources."""

    ensure_output_directories()

    test_catalog = load_test_catalog()

    residual_model = (
        load_residual_model()
    )

    gate_model = load_gate_model()

    residual_matrix = prepare_matrix(
        test_catalog,
        RESIDUAL_FEATURE_COLUMNS,
    )

    predicted_residual = (
        residual_model.predict(
            residual_matrix
        )
    )

    gate_catalog = test_catalog.copy()

    gate_catalog[
        PREDICTED_RESIDUAL_FEATURE
    ] = predicted_residual

    gate_catalog[
        ABSOLUTE_PREDICTED_RESIDUAL_FEATURE
    ] = np.abs(
        predicted_residual
    )

    gate_matrix = prepare_matrix(
        gate_catalog,
        GATE_FEATURE_COLUMNS,
    )

    (
        residual_shap,
        residual_base_values,
    ) = calculate_native_tree_shap(
        residual_model,
        residual_matrix,
    )

    (
        gate_shap,
        gate_base_values,
    ) = calculate_native_tree_shap(
        gate_model,
        gate_matrix,
    )

    residual_importance = (
        build_global_importance(
            residual_shap,
            residual_matrix,
            model_name="Residual regressor",
            output_unit=(
                "log-redshift residual"
            ),
        )
    )

    gate_importance = (
        build_global_importance(
            gate_shap,
            gate_matrix,
            model_name="Gate classifier",
            output_unit="log-odds",
        )
    )

    global_importance = pd.concat(
        [
            residual_importance,
            gate_importance,
        ],
        ignore_index=True,
    )

    paired_cases = (
        load_paired_case_predictions()
    )

    explanation_cases = (
        select_explanation_cases(
            paired_cases
        )
    )

    representative_key = (
        representative_source_key(
            explanation_cases
        )
    )

    explanation_figure = (
        build_explanation_figure(
            residual_importance,
            gate_importance,
            residual_shap,
            gate_shap,
            residual_matrix,
            gate_matrix,
            test_catalog,
            paired_cases,
            representative_key,
        )
    )

    save_explanation_products(
        global_importance,
        explanation_cases,
        representative_key,
    )

    return (
        global_importance,
        explanation_cases,
        explanation_figure,
    )