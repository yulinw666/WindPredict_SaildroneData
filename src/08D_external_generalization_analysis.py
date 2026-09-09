# -*- coding: utf-8 -*-
"""
08D_external_generalization_analysis.py

Paper-ready analysis of the frozen SD1033 external-generalization experiments.

This stage is ANALYSIS ONLY.
It does NOT train, fine-tune, select checkpoints, alter scalers, or modify
forecast outputs.

Inputs
------
From 08B:
    external_dataset_summary.csv
    feature_shift_summary.csv

From 08C:
    external_seed_summary.csv
    external_per_horizon_summary.csv
    external_paired_effects.csv
    external_inference_audit.csv

Default directories
-------------------
08B external root:
D:\\project\\WindPredict_SaildroneData\\data\\external\\SD1033_external_forecasting_v0_1

08C inference root:
D:\\project\\WindPredict_SaildroneData\\data\\external\\SD1033_external_forecasting_v0_1\\frozen_external_inference_v0_1

Primary external tests
----------------------
1. SD1033_2024_matched_SD1090
   Same mission / matched calendar period / different vehicle.
2. SD1033_2023_full
   Cross-year / cross-mission external test.
3. SD1033_2022_full
   Cross-year / cross-mission external test.

SD1033_2024_full is retained as a supplementary robustness test because it is
not statistically independent of the matched 2024 test.

Paper-level questions
---------------------
A. Does the frozen compact vessel-residual model improve frozen Ridge on
   unseen SD1033 data?
B. Is the improvement present at every forecast horizon?
C. Does the compact physics-guided model retain the same trend across vehicle
   and year shifts?
D. How do external feature distribution shifts relate descriptively to the
   observed residual-model gain?
E. Does the 2024 matched-period cross-vehicle experiment support transfer of
   the vessel-motion residual correction beyond SD1090?

Important statistical wording
-----------------------------
The neural-model standard deviations and paired CIs in 08C are across five
frozen model initializations/seeds. They are NOT confidence intervals over
independent ocean missions or the population of environmental conditions.

Distribution-shift correlations are DESCRIPTIVE ONLY. The primary correlation
uses only three external tests (2024 matched, 2023, 2022), so it must not be
interpreted as inferential evidence or causality.

Outputs
-------
tables/
    Table_External_Generalization_Main.csv
    Table_External_Generalization_Main.tex
    Table_External_Generalization_FullMetrics.csv
    Table_External_Generalization_FullMetrics.tex
    Table_PerHorizon_AW_Skill.csv
    Table_PerHorizon_AW_Skill.tex
    Table_Matched2024_CrossVehicle.csv
    Table_Matched2024_CrossVehicle.tex
    Table_DistributionShift.csv

analysis/
    generalization_summary.csv
    generalization_summary.json
    distribution_shift_vs_gain.csv
    descriptive_shift_gain_correlation.csv
    horizon_consistency_summary.csv
    model_effect_summary.csv

figures/
    01_primary_external_aw_rmse.png/pdf
    02_primary_per_horizon_aw_skill.png/pdf
    03_primary_aw_skill_heatmap.png/pdf
    04_matched2024_aw_rmse.png/pdf
    05_matched2024_vessel_rmse.png/pdf
    06_distribution_shift_vs_aw_gain.png/pdf
    07_primary_vessel_improvement_vs_ridge.png/pdf
    08_primary_physics_refinement.png/pdf

paper_ready_results.txt
external_generalization_analysis_manifest.json

Plot style
----------
Uses sci_plot_style.py from the same src directory when available.
Fallback figures use Times New Roman when installed and are saved as:
    600 dpi PNG
    vector PDF
No explicit plot colors are specified.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0-external-generalization-analysis"

DEFAULT_EXTERNAL_ROOT = (
    r"D:\project\WindPredict_SaildroneData\data\external"
    r"\SD1033_external_forecasting_v0_1"
)

PRIMARY_DATASETS = [
    "SD1033_2024_matched_SD1090",
    "SD1033_2023_full",
    "SD1033_2022_full",
]

SUPPLEMENTARY_DATASETS = [
    "SD1033_2024_full",
]

DATASET_LABELS = {
    "SD1033_2024_matched_SD1090": "SD1033-2024 matched",
    "SD1033_2024_full": "SD1033-2024 full",
    "SD1033_2023_full": "SD1033-2023",
    "SD1033_2022_full": "SD1033-2022",
}

MODEL_PERSISTENCE = "Persistence"
MODEL_RIDGE = "Frozen-Ridge"
MODEL_COMPACT = "Frozen-Compact-Vessel-Residual"
MODEL_PHYSICS = "Frozen-Physics-Compact-Vessel-Residual"

MODEL_ORDER = [
    MODEL_PERSISTENCE,
    MODEL_RIDGE,
    MODEL_COMPACT,
    MODEL_PHYSICS,
]

MODEL_LABELS = {
    MODEL_PERSISTENCE: "Persistence",
    MODEL_RIDGE: "Frozen Ridge",
    MODEL_COMPACT: "Compact residual",
    MODEL_PHYSICS: "Physics-compact residual",
}

HORIZONS = [1, 2, 3, 5, 10]

MAIN_METRICS = [
    "AW_RMSE_mps",
    "AWS_RMSE_mps",
    "AWA_MAE_deg",
    "vessel_RMSE_mps",
]

FULL_METRICS = [
    "wind_RMSE_mps",
    "wind_speed_RMSE_mps",
    "wind_direction_MAE_deg",
    "vessel_RMSE_mps",
    "SOG_RMSE_mps",
    "COG_MAE_deg",
    "HDG_MAE_deg",
    "AW_RMSE_mps",
    "AWS_RMSE_mps",
    "AWA_MAE_deg",
    "AW_skill_vs_Ridge",
]


def log(message=""):
    print(message, flush=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj):
    def conv(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, np.ndarray):
            return [conv(x) for x in v.tolist()]
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return None if np.isnan(v) else float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, dict):
            return {str(k): conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        return v

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            conv(obj),
            f,
            ensure_ascii=False,
            indent=2,
        )


def parse_dataset_list(text: str):
    out = []
    for token in text.split(","):
        token = token.strip()
        if token and token not in out:
            out.append(token)
    return out


def require_columns(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{name} missing columns: {missing}"
        )


def validate_inputs(
    seed_df,
    horizon_df,
    paired_df,
    shift_df,
    dataset_df,
    audit_df,
    primary_datasets,
):
    require_columns(
        seed_df,
        [
            "dataset_id",
            "model",
            "mean_AW_RMSE_mps",
            "std_AW_RMSE_mps",
            "mean_vessel_RMSE_mps",
            "std_vessel_RMSE_mps",
            "mean_AWS_RMSE_mps",
            "std_AWS_RMSE_mps",
            "mean_AWA_MAE_deg",
            "std_AWA_MAE_deg",
            "mean_AW_skill_vs_Ridge",
        ],
        "external_seed_summary.csv",
    )

    require_columns(
        horizon_df,
        [
            "dataset_id",
            "model",
            "horizon_min",
            "mean_apparent_vector_RMSE_mps",
            "std_apparent_vector_RMSE_mps",
            "mean_vessel_vector_RMSE_mps",
            "mean_AW_skill_vs_Ridge",
        ],
        "external_per_horizon_summary.csv",
    )

    require_columns(
        paired_df,
        [
            "dataset_id",
            "comparison",
            "mean_AW_reduction_B_vs_A_mps",
            "mean_AW_reduction_percent",
            "B_win_count",
            "n_paired_runs",
            "paired_95pct_CI_low_mps",
            "paired_95pct_CI_high_mps",
        ],
        "external_paired_effects.csv",
    )

    require_columns(
        shift_df,
        [
            "dataset_id",
            "feature",
            "mean_abs_z",
            "fraction_abs_z_gt_2",
            "fraction_abs_z_gt_3",
        ],
        "feature_shift_summary.csv",
    )

    require_columns(
        dataset_df,
        [
            "dataset_id",
            "eligible_segment_count",
            "eligible_source_row_count",
            "window_count",
        ],
        "external_dataset_summary.csv",
    )

    require_columns(
        audit_df,
        [
            "dataset_id",
            "audit_passed",
        ],
        "external_inference_audit.csv",
    )

    failed_audit = audit_df.loc[
        ~audit_df["audit_passed"].astype(bool)
    ]

    if not failed_audit.empty:
        raise RuntimeError(
            "08C external inference audit contains failed datasets: "
            + ", ".join(
                failed_audit["dataset_id"].astype(str).tolist()
            )
        )

    available_seed = set(
        seed_df["dataset_id"].astype(str)
    )

    for dataset_id in primary_datasets:
        if dataset_id not in available_seed:
            raise RuntimeError(
                f"Primary dataset missing from 08C summary: {dataset_id}"
            )

        models = set(
            seed_df.loc[
                seed_df["dataset_id"] == dataset_id,
                "model",
            ].astype(str)
        )

        missing_models = [
            m for m in MODEL_ORDER
            if m not in models
        ]

        if missing_models:
            raise RuntimeError(
                f"{dataset_id}: missing models {missing_models}"
            )


def get_row(df, dataset_id, model):
    row = df.loc[
        (df["dataset_id"] == dataset_id)
        & (df["model"] == model)
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Expected exactly one summary row for "
            f"{dataset_id} / {model}; found {len(row)}"
        )

    return row.iloc[0]


def dataset_label(dataset_id):
    return DATASET_LABELS.get(
        dataset_id,
        dataset_id.replace("_", "-"),
    )


def model_label(model):
    return MODEL_LABELS.get(model, model)


def is_neural(model):
    return model in {
        MODEL_COMPACT,
        MODEL_PHYSICS,
    }


def metric_value(row, metric):
    return float(row[f"mean_{metric}"])


def metric_std(row, metric):
    key = f"std_{metric}"
    if key not in row.index:
        return 0.0
    value = row[key]
    if pd.isna(value):
        return 0.0
    return float(value)


def format_mean_std(
    mean,
    std,
    decimals,
    show_std=True,
):
    if pd.isna(mean):
        return "--"

    if show_std and std is not None and not pd.isna(std):
        return (
            f"{mean:.{decimals}f} "
            f"\u00b1 {std:.{decimals}f}"
        )

    return f"{mean:.{decimals}f}"


def format_percent(value, decimals=2, signed=True):
    if pd.isna(value):
        return "--"

    if signed:
        return f"{value:+.{decimals}f}%"

    return f"{value:.{decimals}f}%"


def latex_escape(text: str):
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }

    out = str(text)

    for old, new in replacements.items():
        out = out.replace(old, new)

    return out


def build_main_table(
    seed_df,
    dataset_df,
    primary_datasets,
):
    rows = []

    for dataset_id in primary_datasets:
        ds_meta = dataset_df.loc[
            dataset_df["dataset_id"] == dataset_id
        ]

        windows = (
            int(ds_meta.iloc[0]["window_count"])
            if not ds_meta.empty
            else np.nan
        )

        for model in MODEL_ORDER:
            r = get_row(
                seed_df,
                dataset_id,
                model,
            )

            rows.append({
                "Dataset": dataset_label(dataset_id),
                "Dataset_ID": dataset_id,
                "Windows": windows,
                "Model": model_label(model),
                "Model_ID": model,
                "AW_RMSE_mps_mean": metric_value(
                    r,
                    "AW_RMSE_mps",
                ),
                "AW_RMSE_mps_std": metric_std(
                    r,
                    "AW_RMSE_mps",
                ),
                "AWS_RMSE_mps_mean": metric_value(
                    r,
                    "AWS_RMSE_mps",
                ),
                "AWS_RMSE_mps_std": metric_std(
                    r,
                    "AWS_RMSE_mps",
                ),
                "AWA_MAE_deg_mean": metric_value(
                    r,
                    "AWA_MAE_deg",
                ),
                "AWA_MAE_deg_std": metric_std(
                    r,
                    "AWA_MAE_deg",
                ),
                "Vessel_RMSE_mps_mean": metric_value(
                    r,
                    "vessel_RMSE_mps",
                ),
                "Vessel_RMSE_mps_std": metric_std(
                    r,
                    "vessel_RMSE_mps",
                ),
                "AW_skill_vs_Ridge_percent": (
                    100.0
                    * metric_value(
                        r,
                        "AW_skill_vs_Ridge",
                    )
                ),
            })

    return pd.DataFrame(rows)


def build_full_metrics_table(
    seed_df,
    dataset_ids,
):
    rows = []

    for dataset_id in dataset_ids:
        for model in MODEL_ORDER:
            if seed_df.loc[
                (seed_df["dataset_id"] == dataset_id)
                & (seed_df["model"] == model)
            ].empty:
                continue

            r = get_row(
                seed_df,
                dataset_id,
                model,
            )

            row = {
                "Dataset": dataset_label(dataset_id),
                "Dataset_ID": dataset_id,
                "Model": model_label(model),
                "Model_ID": model,
            }

            for metric in FULL_METRICS:
                row[f"{metric}_mean"] = metric_value(
                    r,
                    metric,
                )

                row[f"{metric}_std"] = metric_std(
                    r,
                    metric,
                )

            rows.append(row)

    return pd.DataFrame(rows)


def build_horizon_skill_table(
    horizon_df,
    primary_datasets,
):
    rows = []

    for dataset_id in primary_datasets:
        ridge = horizon_df.loc[
            (horizon_df["dataset_id"] == dataset_id)
            & (horizon_df["model"] == MODEL_RIDGE)
        ].set_index("horizon_min")

        compact = horizon_df.loc[
            (horizon_df["dataset_id"] == dataset_id)
            & (horizon_df["model"] == MODEL_COMPACT)
        ].set_index("horizon_min")

        physics = horizon_df.loc[
            (horizon_df["dataset_id"] == dataset_id)
            & (horizon_df["model"] == MODEL_PHYSICS)
        ].set_index("horizon_min")

        for horizon in HORIZONS:
            if (
                horizon not in ridge.index
                or horizon not in compact.index
                or horizon not in physics.index
            ):
                raise RuntimeError(
                    f"{dataset_id}: missing horizon {horizon}"
                )

            ridge_aw = float(
                ridge.loc[
                    horizon,
                    "mean_apparent_vector_RMSE_mps",
                ]
            )

            compact_aw = float(
                compact.loc[
                    horizon,
                    "mean_apparent_vector_RMSE_mps",
                ]
            )

            physics_aw = float(
                physics.loc[
                    horizon,
                    "mean_apparent_vector_RMSE_mps",
                ]
            )

            physics_std = float(
                physics.loc[
                    horizon,
                    "std_apparent_vector_RMSE_mps",
                ]
            )

            rows.append({
                "Dataset": dataset_label(dataset_id),
                "Dataset_ID": dataset_id,
                "Horizon_min": int(horizon),
                "Ridge_AW_RMSE_mps": ridge_aw,
                "Compact_AW_RMSE_mps": compact_aw,
                "PhysicsCompact_AW_RMSE_mps": physics_aw,
                "PhysicsCompact_AW_RMSE_std_mps": physics_std,
                "Compact_AW_skill_vs_Ridge_percent": (
                    100.0
                    * (
                        1.0
                        - compact_aw / ridge_aw
                    )
                ),
                "PhysicsCompact_AW_skill_vs_Ridge_percent": (
                    100.0
                    * (
                        1.0
                        - physics_aw / ridge_aw
                    )
                ),
                "Physics_refinement_vs_Compact_percent": (
                    100.0
                    * (
                        1.0
                        - physics_aw / compact_aw
                    )
                ),
            })

    return pd.DataFrame(rows)


def build_effect_summary(
    seed_df,
    paired_df,
    dataset_ids,
):
    rows = []

    for dataset_id in dataset_ids:
        ridge = get_row(
            seed_df,
            dataset_id,
            MODEL_RIDGE,
        )

        compact = get_row(
            seed_df,
            dataset_id,
            MODEL_COMPACT,
        )

        physics = get_row(
            seed_df,
            dataset_id,
            MODEL_PHYSICS,
        )

        for model_name, model_row in [
            (MODEL_COMPACT, compact),
            (MODEL_PHYSICS, physics),
        ]:
            rows.append({
                "Dataset": dataset_label(dataset_id),
                "Dataset_ID": dataset_id,
                "Model": model_label(model_name),
                "Model_ID": model_name,
                "AW_improvement_vs_Ridge_percent": (
                    100.0
                    * (
                        1.0
                        - metric_value(
                            model_row,
                            "AW_RMSE_mps",
                        )
                        / metric_value(
                            ridge,
                            "AW_RMSE_mps",
                        )
                    )
                ),
                "Vessel_improvement_vs_Ridge_percent": (
                    100.0
                    * (
                        1.0
                        - metric_value(
                            model_row,
                            "vessel_RMSE_mps",
                        )
                        / metric_value(
                            ridge,
                            "vessel_RMSE_mps",
                        )
                    )
                ),
                "AWS_improvement_vs_Ridge_percent": (
                    100.0
                    * (
                        1.0
                        - metric_value(
                            model_row,
                            "AWS_RMSE_mps",
                        )
                        / metric_value(
                            ridge,
                            "AWS_RMSE_mps",
                        )
                    )
                ),
                "AWA_improvement_vs_Ridge_percent": (
                    100.0
                    * (
                        1.0
                        - metric_value(
                            model_row,
                            "AWA_MAE_deg",
                        )
                        / metric_value(
                            ridge,
                            "AWA_MAE_deg",
                        )
                    )
                ),
            })

        pair = paired_df.loc[
            (paired_df["dataset_id"] == dataset_id)
            & (
                paired_df["comparison"]
                == "PhysicsCompact_vs_Ridge"
            )
        ]

        if not pair.empty:
            p = pair.iloc[0]

            rows[-1].update({
                "AW_paired_seed_reduction_mps": float(
                    p[
                        "mean_AW_reduction_B_vs_A_mps"
                    ]
                ),
                "AW_paired_95CI_low_mps": float(
                    p[
                        "paired_95pct_CI_low_mps"
                    ]
                ),
                "AW_paired_95CI_high_mps": float(
                    p[
                        "paired_95pct_CI_high_mps"
                    ]
                ),
                "AW_seed_wins": int(
                    p["B_win_count"]
                ),
                "AW_seed_runs": int(
                    p["n_paired_runs"]
                ),
            })

    return pd.DataFrame(rows)


def build_shift_summary(
    shift_df,
    dataset_ids,
):
    rows = []

    for dataset_id in dataset_ids:
        d = shift_df.loc[
            shift_df["dataset_id"] == dataset_id
        ].copy()

        if d.empty:
            raise RuntimeError(
                f"Missing feature-shift data for {dataset_id}"
            )

        # Equal-feature descriptive aggregate. This is intentionally simple
        # and transparent; it is not an inferential OOD metric.
        row = {
            "Dataset": dataset_label(dataset_id),
            "Dataset_ID": dataset_id,
            "Feature_count": int(
                d["feature"].nunique()
            ),
            "Mean_feature_mean_abs_z": float(
                d["mean_abs_z"].mean()
            ),
            "Median_feature_mean_abs_z": float(
                d["mean_abs_z"].median()
            ),
            "Mean_fraction_abs_z_gt_2": float(
                d["fraction_abs_z_gt_2"].mean()
            ),
            "Mean_fraction_abs_z_gt_3": float(
                d["fraction_abs_z_gt_3"].mean()
            ),
            "Max_feature_mean_abs_z": float(
                d["mean_abs_z"].max()
            ),
            "Max_shift_feature": str(
                d.loc[
                    d["mean_abs_z"].idxmax(),
                    "feature",
                ]
            ),
        }

        # Temperature is retained because it was a notable shift channel.
        temp = d.loc[
            d["feature"] == "TEMP_AIR_MEAN"
        ]

        if not temp.empty:
            row.update({
                "TEMP_mean_z": float(
                    temp.iloc[0]["mean_z"]
                ),
                "TEMP_mean_abs_z": float(
                    temp.iloc[0]["mean_abs_z"]
                ),
                "TEMP_fraction_abs_z_gt_2": float(
                    temp.iloc[0]["fraction_abs_z_gt_2"]
                ),
                "TEMP_fraction_abs_z_gt_3": float(
                    temp.iloc[0]["fraction_abs_z_gt_3"]
                ),
            })

        rows.append(row)

    return pd.DataFrame(rows)


def build_shift_gain_analysis(
    shift_summary,
    effect_summary,
    primary_datasets,
):
    physics_effect = effect_summary.loc[
        effect_summary["Model_ID"] == MODEL_PHYSICS
    ][
        [
            "Dataset_ID",
            "AW_improvement_vs_Ridge_percent",
            "Vessel_improvement_vs_Ridge_percent",
        ]
    ]

    merged = shift_summary.merge(
        physics_effect,
        on="Dataset_ID",
        how="inner",
        validate="one_to_one",
    )

    merged = merged.loc[
        merged["Dataset_ID"].isin(
            primary_datasets
        )
    ].copy()

    merged["Dataset"] = merged[
        "Dataset_ID"
    ].map(
        lambda x: dataset_label(x)
    )

    return merged


def descriptive_correlation_table(
    shift_gain_df,
):
    predictors = [
        "Mean_feature_mean_abs_z",
        "Mean_fraction_abs_z_gt_2",
        "Mean_fraction_abs_z_gt_3",
        "TEMP_mean_abs_z",
        "TEMP_fraction_abs_z_gt_2",
    ]

    outcomes = [
        "AW_improvement_vs_Ridge_percent",
        "Vessel_improvement_vs_Ridge_percent",
    ]

    rows = []

    for xcol in predictors:
        if xcol not in shift_gain_df.columns:
            continue

        for ycol in outcomes:
            d = shift_gain_df[
                [
                    xcol,
                    ycol,
                ]
            ].dropna()

            n = len(d)

            if n < 2:
                pearson = np.nan
                spearman = np.nan
            else:
                pearson = float(
                    d[xcol].corr(
                        d[ycol],
                        method="pearson",
                    )
                )

                spearman = float(
                    d[xcol].corr(
                        d[ycol],
                        method="spearman",
                    )
                )

            rows.append({
                "Predictor": xcol,
                "Outcome": ycol,
                "N_primary_external_tests": int(n),
                "Pearson_r_descriptive": pearson,
                "Spearman_rho_descriptive": spearman,
                "Interpretation_policy": (
                    "DESCRIPTIVE ONLY; N is too small for inferential "
                    "or causal interpretation."
                ),
            })

    return pd.DataFrame(rows)


def build_horizon_consistency(
    horizon_skill,
    primary_datasets,
    all_dataset_ids,
):
    rows = []

    for label, dataset_ids in [
        (
            "primary_external_tests",
            primary_datasets,
        ),
        (
            "all_external_tests_including_2024_full",
            all_dataset_ids,
        ),
    ]:
        d = horizon_skill.loc[
            horizon_skill[
                "Dataset_ID"
            ].isin(
                dataset_ids
            )
        ]

        positive_ridge = (
            d[
                "PhysicsCompact_AW_skill_vs_Ridge_percent"
            ]
            > 0
        )

        positive_compact = (
            d[
                "Physics_refinement_vs_Compact_percent"
            ]
            > 0
        )

        rows.append({
            "Scope": label,
            "Dataset_count": int(
                d["Dataset_ID"].nunique()
            ),
            "Horizon_count_per_dataset": int(
                d["Horizon_min"].nunique()
            ),
            "Dataset_horizon_pairs": int(
                len(d)
            ),
            "PhysicsCompact_better_than_Ridge_count": int(
                positive_ridge.sum()
            ),
            "PhysicsCompact_better_than_Ridge_fraction": float(
                positive_ridge.mean()
            ),
            "PhysicsCompact_better_than_Compact_count": int(
                positive_compact.sum()
            ),
            "PhysicsCompact_better_than_Compact_fraction": float(
                positive_compact.mean()
            ),
            "Minimum_PhysicsCompact_skill_vs_Ridge_percent": float(
                d[
                    "PhysicsCompact_AW_skill_vs_Ridge_percent"
                ].min()
            ),
            "Maximum_PhysicsCompact_skill_vs_Ridge_percent": float(
                d[
                    "PhysicsCompact_AW_skill_vs_Ridge_percent"
                ].max()
            ),
        })

    return pd.DataFrame(rows)


def generate_main_latex(
    table_df,
    path,
):
    lines = []

    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Frozen external-generalization performance on unseen "
        r"SD1033 missions. Neural-model values are reported as mean "
        r"$\pm$ standard deviation across five frozen initialization seeds.}"
    )
    lines.append(r"\label{tab:external_generalization}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & Model & AW RMSE (m/s) & AWS RMSE (m/s) & "
        r"AWA MAE ($^\circ$) & Vessel RMSE (m/s) & AW skill vs. Ridge (\%) \\"
    )
    lines.append(r"\midrule")

    last_dataset = None

    for _, row in table_df.iterrows():
        dataset = row["Dataset"]

        if (
            last_dataset is not None
            and dataset != last_dataset
        ):
            lines.append(r"\addlinespace")

        model_id = row["Model_ID"]

        show_std = is_neural(
            model_id
        )

        aw = format_mean_std(
            row["AW_RMSE_mps_mean"],
            row["AW_RMSE_mps_std"],
            4,
            show_std=show_std,
        ).replace("\u00b1", r"$\pm$")

        aws = format_mean_std(
            row["AWS_RMSE_mps_mean"],
            row["AWS_RMSE_mps_std"],
            4,
            show_std=show_std,
        ).replace("\u00b1", r"$\pm$")

        awa = format_mean_std(
            row["AWA_MAE_deg_mean"],
            row["AWA_MAE_deg_std"],
            2,
            show_std=show_std,
        ).replace("\u00b1", r"$\pm$")

        vessel = format_mean_std(
            row["Vessel_RMSE_mps_mean"],
            row["Vessel_RMSE_mps_std"],
            4,
            show_std=show_std,
        ).replace("\u00b1", r"$\pm$")

        skill = format_percent(
            row["AW_skill_vs_Ridge_percent"],
            decimals=2,
            signed=True,
        ).replace("%", r"\%")

        lines.append(
            f"{latex_escape(dataset)} & "
            f"{latex_escape(row['Model'])} & "
            f"{aw} & {aws} & {awa} & {vessel} & {skill} \\\\"
        )

        last_dataset = dataset

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\begin{minipage}{0.98\textwidth}\footnotesize "
        r"All models are frozen before evaluation on SD1033. "
        r"No external fitting, fine-tuning, scaler refitting, or checkpoint "
        r"selection is performed. The reported seed variability reflects "
        r"model initialization variability, not mission-level population uncertainty."
        r"\end{minipage}"
    )
    lines.append(r"\end{table*}")

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def generate_full_metrics_latex(
    table_df,
    path,
):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Detailed frozen external-generalization metrics.}",
        r"\label{tab:external_generalization_full}",
        r"\scriptsize",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        (
            r"Dataset & Model & Wind RMSE & Vessel RMSE & HDG MAE & "
            r"AW RMSE & AWS RMSE & AWA MAE \\"
        ),
        r"\midrule",
    ]

    last_dataset = None

    for _, row in table_df.iterrows():
        dataset = row["Dataset"]

        if last_dataset is not None and dataset != last_dataset:
            lines.append(r"\addlinespace")

        show_std = is_neural(
            row["Model_ID"]
        )

        values = []

        for metric, decimals in [
            ("wind_RMSE_mps", 4),
            ("vessel_RMSE_mps", 4),
            ("HDG_MAE_deg", 2),
            ("AW_RMSE_mps", 4),
            ("AWS_RMSE_mps", 4),
            ("AWA_MAE_deg", 2),
        ]:
            text = format_mean_std(
                row[f"{metric}_mean"],
                row[f"{metric}_std"],
                decimals,
                show_std=show_std,
            ).replace("\u00b1", r"$\pm$")

            values.append(text)

        lines.append(
            f"{latex_escape(dataset)} & "
            f"{latex_escape(row['Model'])} & "
            + " & ".join(values)
            + r" \\"
        )

        last_dataset = dataset

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def generate_horizon_latex(
    horizon_df,
    path,
):
    pivot = horizon_df.pivot(
        index="Dataset",
        columns="Horizon_min",
        values="PhysicsCompact_AW_skill_vs_Ridge_percent",
    )

    pivot = pivot.reindex(
        columns=HORIZONS
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-horizon apparent-wind RMSE improvement of the frozen "
        r"physics-compact model relative to frozen Ridge.}",
        r"\label{tab:external_horizon_skill}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Dataset & 1 min & 2 min & 3 min & 5 min & 10 min \\",
        r"\midrule",
    ]

    for dataset, row in pivot.iterrows():
        vals = [
            f"{float(row[h]):.2f}\\%"
            for h in HORIZONS
        ]

        lines.append(
            f"{latex_escape(dataset)} & "
            + " & ".join(vals)
            + r" \\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def generate_matched_latex(
    matched_df,
    path,
):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Matched-period cross-vehicle external validation on SD1033-2024.}",
        r"\label{tab:matched_cross_vehicle}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & AW RMSE & Vessel RMSE & AWS RMSE & AWA MAE \\",
        r"\midrule",
    ]

    for _, row in matched_df.iterrows():
        show_std = is_neural(
            row["Model_ID"]
        )

        aw = format_mean_std(
            row["AW_RMSE_mps_mean"],
            row["AW_RMSE_mps_std"],
            4,
            show_std,
        ).replace("\u00b1", r"$\pm$")

        vessel = format_mean_std(
            row["Vessel_RMSE_mps_mean"],
            row["Vessel_RMSE_mps_std"],
            4,
            show_std,
        ).replace("\u00b1", r"$\pm$")

        aws = format_mean_std(
            row["AWS_RMSE_mps_mean"],
            row["AWS_RMSE_mps_std"],
            4,
            show_std,
        ).replace("\u00b1", r"$\pm$")

        awa = format_mean_std(
            row["AWA_MAE_deg_mean"],
            row["AWA_MAE_deg_std"],
            2,
            show_std,
        ).replace("\u00b1", r"$\pm$")

        lines.append(
            f"{latex_escape(row['Model'])} & "
            f"{aw} & {vessel} & {aws} & {awa} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def load_style():
    path = (
        Path(__file__).resolve().parent
        / "sci_plot_style.py"
    )

    if not path.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        "sci_plot_style_08d",
        str(path),
    )

    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def make_figures(
    *,
    main_table,
    horizon_skill,
    effect_summary,
    shift_gain,
    output_dir,
    primary_datasets,
):
    import matplotlib.pyplot as plt

    style = load_style()

    if style is None:
        plt.rcParams.update({
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 10,
        })

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    def clean(ax, grid=True):
        if (
            style is not None
            and hasattr(style, "clean_axis")
        ):
            style.clean_axis(
                ax,
                grid=grid,
            )
        else:
            ax.tick_params(
                which="both",
                direction="in",
                top=True,
                right=True,
            )
            if grid:
                ax.grid(
                    True,
                    alpha=0.25,
                )

    def save(fig, name):
        base = fig_dir / name

        if (
            style is not None
            and hasattr(style, "save_figure")
        ):
            style.save_figure(
                fig,
                base,
            )
        else:
            fig.savefig(
                base.with_suffix(".png"),
                dpi=600,
                bbox_inches="tight",
            )
            fig.savefig(
                base.with_suffix(".pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

    primary_labels = [
        dataset_label(d)
        for d in primary_datasets
    ]

    # Figure 01: Primary external AW RMSE.
    fig, ax = plt.subplots(
        figsize=(7.0, 3.7)
    )

    x = np.arange(
        len(primary_datasets)
    )

    width = 0.8 / len(MODEL_ORDER)

    for mi, model in enumerate(
        MODEL_ORDER
    ):
        means = []
        stds = []

        for dataset_id in primary_datasets:
            r = main_table.loc[
                (main_table["Dataset_ID"] == dataset_id)
                & (main_table["Model_ID"] == model)
            ].iloc[0]

            means.append(
                r["AW_RMSE_mps_mean"]
            )

            stds.append(
                r["AW_RMSE_mps_std"]
                if is_neural(model)
                else 0.0
            )

        pos = (
            x
            + (
                mi
                - (
                    len(MODEL_ORDER)
                    - 1
                ) / 2.0
            ) * width
        )

        ax.bar(
            pos,
            means,
            width=width,
            yerr=stds,
            capsize=2,
            label=model_label(model),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        primary_labels,
        rotation=18,
        ha="right",
    )
    ax.set_ylabel(
        r"Apparent-wind vector RMSE (m s$^{-1}$)"
    )
    ax.legend(
        loc="best",
        ncol=2,
    )
    clean(ax)
    save(
        fig,
        "01_primary_external_aw_rmse",
    )

    # Figure 02: per-horizon skill lines.
    fig, ax = plt.subplots(
        figsize=(5.8, 3.5)
    )

    for dataset_id in primary_datasets:
        d = horizon_skill.loc[
            horizon_skill["Dataset_ID"] == dataset_id
        ].sort_values(
            "Horizon_min"
        )

        ax.plot(
            d["Horizon_min"],
            d[
                "PhysicsCompact_AW_skill_vs_Ridge_percent"
            ],
            marker="o",
            label=dataset_label(dataset_id),
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )
    ax.set_xticks(HORIZONS)
    ax.set_xlabel(
        "Forecast horizon (min)"
    )
    ax.set_ylabel(
        "AW RMSE improvement vs. Ridge (%)"
    )
    ax.legend(
        loc="best"
    )
    clean(ax)
    save(
        fig,
        "02_primary_per_horizon_aw_skill",
    )

    # Figure 03: skill heatmap.
    pivot = (
        horizon_skill
        .pivot(
            index="Dataset",
            columns="Horizon_min",
            values=(
                "PhysicsCompact_AW_skill_vs_Ridge_percent"
            ),
        )
        .reindex(
            index=primary_labels,
            columns=HORIZONS,
        )
    )

    fig, ax = plt.subplots(
        figsize=(5.5, 2.8)
    )

    im = ax.imshow(
        pivot.to_numpy(),
        aspect="auto",
    )

    ax.set_yticks(
        np.arange(len(pivot.index))
    )
    ax.set_yticklabels(
        pivot.index
    )
    ax.set_xticks(
        np.arange(len(HORIZONS))
    )
    ax.set_xticklabels(
        [str(h) for h in HORIZONS]
    )
    ax.set_xlabel(
        "Forecast horizon (min)"
    )

    for i in range(
        pivot.shape[0]
    ):
        for j in range(
            pivot.shape[1]
        ):
            value = float(
                pivot.iloc[i, j]
            )
            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    cbar = fig.colorbar(
        im,
        ax=ax,
    )
    cbar.set_label(
        "AW RMSE improvement vs. Ridge (%)"
    )

    clean(
        ax,
        grid=False,
    )
    save(
        fig,
        "03_primary_aw_skill_heatmap",
    )

    # Figure 04: matched-period AW.
    matched_id = (
        "SD1033_2024_matched_SD1090"
    )

    matched = main_table.loc[
        main_table["Dataset_ID"] == matched_id
    ].copy()

    if not matched.empty:
        matched[
            "_order"
        ] = matched[
            "Model_ID"
        ].map({
            m: i
            for i, m in enumerate(
                MODEL_ORDER
            )
        })

        matched = matched.sort_values(
            "_order"
        )

        fig, ax = plt.subplots(
            figsize=(5.7, 3.3)
        )

        x = np.arange(
            len(matched)
        )

        yerr = [
            (
                row[
                    "AW_RMSE_mps_std"
                ]
                if is_neural(
                    row["Model_ID"]
                )
                else 0.0
            )
            for _, row in matched.iterrows()
        ]

        ax.bar(
            x,
            matched[
                "AW_RMSE_mps_mean"
            ],
            yerr=yerr,
            capsize=2,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            matched["Model"],
            rotation=25,
            ha="right",
        )
        ax.set_ylabel(
            r"AW vector RMSE (m s$^{-1}$)"
        )
        clean(ax)
        save(
            fig,
            "04_matched2024_aw_rmse",
        )

        # Figure 05: matched vessel RMSE.
        fig, ax = plt.subplots(
            figsize=(5.7, 3.3)
        )

        yerr = [
            (
                row[
                    "Vessel_RMSE_mps_std"
                ]
                if is_neural(
                    row["Model_ID"]
                )
                else 0.0
            )
            for _, row in matched.iterrows()
        ]

        ax.bar(
            x,
            matched[
                "Vessel_RMSE_mps_mean"
            ],
            yerr=yerr,
            capsize=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            matched["Model"],
            rotation=25,
            ha="right",
        )
        ax.set_ylabel(
            r"Vessel vector RMSE (m s$^{-1}$)"
        )
        clean(ax)
        save(
            fig,
            "05_matched2024_vessel_rmse",
        )

    # Figure 06: shift versus gain.
    fig, ax = plt.subplots(
        figsize=(5.0, 3.4)
    )

    ax.scatter(
        shift_gain[
            "Mean_feature_mean_abs_z"
        ],
        shift_gain[
            "AW_improvement_vs_Ridge_percent"
        ],
        s=45,
    )

    for _, row in shift_gain.iterrows():
        ax.annotate(
            row["Dataset"],
            (
                row[
                    "Mean_feature_mean_abs_z"
                ],
                row[
                    "AW_improvement_vs_Ridge_percent"
                ],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(
        r"Mean feature $|z|$ under SD1090 TRAIN scaler"
    )
    ax.set_ylabel(
        "Physics-compact AW improvement vs. Ridge (%)"
    )
    clean(ax)
    save(
        fig,
        "06_distribution_shift_vs_aw_gain",
    )

    # Figure 07: vessel improvement.
    phys_effect = effect_summary.loc[
        (effect_summary["Model_ID"] == MODEL_PHYSICS)
        & (
            effect_summary["Dataset_ID"].isin(
                primary_datasets
            )
        )
    ].copy()

    phys_effect[
        "_order"
    ] = phys_effect[
        "Dataset_ID"
    ].map({
        d: i
        for i, d in enumerate(
            primary_datasets
        )
    })

    phys_effect = phys_effect.sort_values(
        "_order"
    )

    fig, ax = plt.subplots(
        figsize=(5.5, 3.2)
    )

    x = np.arange(
        len(phys_effect)
    )

    ax.bar(
        x,
        phys_effect[
            "Vessel_improvement_vs_Ridge_percent"
        ],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        phys_effect["Dataset"],
        rotation=18,
        ha="right",
    )
    ax.set_ylabel(
        "Vessel RMSE improvement vs. Ridge (%)"
    )
    clean(ax)
    save(
        fig,
        "07_primary_vessel_improvement_vs_ridge",
    )

    # Figure 08: small physics refinement over compact.
    physics_ref = (
        horizon_skill.groupby(
            [
                "Dataset",
                "Dataset_ID",
            ],
            as_index=False,
        )[
            "Physics_refinement_vs_Compact_percent"
        ]
        .mean()
    )

    physics_ref = physics_ref.loc[
        physics_ref["Dataset_ID"].isin(
            primary_datasets
        )
    ].copy()

    physics_ref[
        "_order"
    ] = physics_ref[
        "Dataset_ID"
    ].map({
        d: i
        for i, d in enumerate(
            primary_datasets
        )
    })

    physics_ref = physics_ref.sort_values(
        "_order"
    )

    fig, ax = plt.subplots(
        figsize=(5.3, 3.1)
    )

    x = np.arange(
        len(physics_ref)
    )

    ax.bar(
        x,
        physics_ref[
            "Physics_refinement_vs_Compact_percent"
        ],
    )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        physics_ref["Dataset"],
        rotation=18,
        ha="right",
    )
    ax.set_ylabel(
        "Physics refinement over compact (%)"
    )
    clean(ax)
    save(
        fig,
        "08_primary_physics_refinement",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--external-root",
        default=DEFAULT_EXTERNAL_ROOT,
    )

    parser.add_argument(
        "--inference-dir",
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--primary-datasets",
        default=",".join(
            PRIMARY_DATASETS
        ),
    )

    parser.add_argument(
        "--include-supplementary-2024-full",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
    )

    args = parser.parse_args()

    external_root = Path(
        args.external_root
    )

    inference_dir = (
        Path(
            args.inference_dir
        )
        if args.inference_dir
        else (
            external_root
            / "frozen_external_inference_v0_1"
        )
    )

    output_dir = (
        Path(
            args.output_dir
        )
        if args.output_dir
        else (
            inference_dir
            / "paper_analysis_v0_1"
        )
    )

    table_dir = (
        output_dir
        / "tables"
    )

    analysis_dir = (
        output_dir
        / "analysis"
    )

    for d in [
        output_dir,
        table_dir,
        analysis_dir,
    ]:
        d.mkdir(
            parents=True,
            exist_ok=True,
        )

    primary_datasets = parse_dataset_list(
        args.primary_datasets
    )

    all_dataset_ids = list(
        primary_datasets
    )

    if args.include_supplementary_2024_full:
        for dataset_id in SUPPLEMENTARY_DATASETS:
            if dataset_id not in all_dataset_ids:
                all_dataset_ids.append(
                    dataset_id
                )

    seed_path = (
        inference_dir
        / "external_seed_summary.csv"
    )

    horizon_path = (
        inference_dir
        / "external_per_horizon_summary.csv"
    )

    paired_path = (
        inference_dir
        / "external_paired_effects.csv"
    )

    audit_path = (
        inference_dir
        / "external_inference_audit.csv"
    )

    inference_manifest_path = (
        inference_dir
        / "external_inference_manifest.json"
    )

    shift_path = (
        external_root
        / "feature_shift_summary.csv"
    )

    dataset_path = (
        external_root
        / "external_dataset_summary.csv"
    )

    external_manifest_path = (
        external_root
        / "external_dataset_manifest.json"
    )

    log(
        "=" * 110
    )
    log(
        "08D - EXTERNAL GENERALIZATION PAPER ANALYSIS"
    )
    log(
        "=" * 110
    )
    log(
        f"script_version        : {SCRIPT_VERSION}"
    )
    log(
        f"external_root         : {external_root}"
    )
    log(
        f"inference_dir         : {inference_dir}"
    )
    log(
        f"primary datasets      : {primary_datasets}"
    )
    log(
        f"analysis output       : {output_dir}"
    )
    log("")

    seed_df = load_csv(
        seed_path
    )

    horizon_df = load_csv(
        horizon_path
    )

    paired_df = load_csv(
        paired_path
    )

    audit_df = load_csv(
        audit_path
    )

    shift_df = load_csv(
        shift_path
    )

    dataset_df = load_csv(
        dataset_path
    )

    inference_manifest = load_json(
        inference_manifest_path
    )

    external_manifest = load_json(
        external_manifest_path
    )

    validate_inputs(
        seed_df,
        horizon_df,
        paired_df,
        shift_df,
        dataset_df,
        audit_df,
        primary_datasets,
    )

    log(
        "[AUDIT] All requested external inference datasets passed 08C audit."
    )

    # --------------------------------------------------------------
    # Paper-ready tables
    # --------------------------------------------------------------
    main_table = build_main_table(
        seed_df,
        dataset_df,
        primary_datasets,
    )

    full_metrics_table = (
        build_full_metrics_table(
            seed_df,
            all_dataset_ids,
        )
    )

    # Build horizon table for all datasets first, then primary subset for table.
    horizon_all = (
        build_horizon_skill_table(
            horizon_df,
            all_dataset_ids,
        )
    )

    horizon_primary = horizon_all.loc[
        horizon_all["Dataset_ID"].isin(
            primary_datasets
        )
    ].copy()

    effect_summary = (
        build_effect_summary(
            seed_df,
            paired_df,
            all_dataset_ids,
        )
    )

    shift_summary = (
        build_shift_summary(
            shift_df,
            all_dataset_ids,
        )
    )

    shift_gain = (
        build_shift_gain_analysis(
            shift_summary,
            effect_summary,
            primary_datasets,
        )
    )

    corr_df = (
        descriptive_correlation_table(
            shift_gain
        )
    )

    horizon_consistency = (
        build_horizon_consistency(
            horizon_all,
            primary_datasets,
            all_dataset_ids,
        )
    )

    # Matched cross-vehicle table.
    matched_table = main_table.loc[
        main_table["Dataset_ID"]
        == "SD1033_2024_matched_SD1090"
    ].copy()

    # Save CSV tables.
    main_table.to_csv(
        table_dir
        / "Table_External_Generalization_Main.csv",
        index=False,
        encoding="utf-8-sig",
    )

    full_metrics_table.to_csv(
        table_dir
        / "Table_External_Generalization_FullMetrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_primary.to_csv(
        table_dir
        / "Table_PerHorizon_AW_Skill.csv",
        index=False,
        encoding="utf-8-sig",
    )

    matched_table.to_csv(
        table_dir
        / "Table_Matched2024_CrossVehicle.csv",
        index=False,
        encoding="utf-8-sig",
    )

    shift_summary.to_csv(
        table_dir
        / "Table_DistributionShift.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save LaTeX tables.
    generate_main_latex(
        main_table,
        table_dir
        / "Table_External_Generalization_Main.tex",
    )

    generate_full_metrics_latex(
        full_metrics_table,
        table_dir
        / "Table_External_Generalization_FullMetrics.tex",
    )

    generate_horizon_latex(
        horizon_primary,
        table_dir
        / "Table_PerHorizon_AW_Skill.tex",
    )

    generate_matched_latex(
        matched_table,
        table_dir
        / "Table_Matched2024_CrossVehicle.tex",
    )

    # --------------------------------------------------------------
    # Analysis outputs
    # --------------------------------------------------------------
    effect_summary.to_csv(
        analysis_dir
        / "model_effect_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    shift_gain.to_csv(
        analysis_dir
        / "distribution_shift_vs_gain.csv",
        index=False,
        encoding="utf-8-sig",
    )

    corr_df.to_csv(
        analysis_dir
        / "descriptive_shift_gain_correlation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_consistency.to_csv(
        analysis_dir
        / "horizon_consistency_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Main numerical findings
    # --------------------------------------------------------------
    primary_effect = effect_summary.loc[
        (effect_summary["Dataset_ID"].isin(
            primary_datasets
        ))
        & (
            effect_summary["Model_ID"] == MODEL_PHYSICS
        )
    ].copy()

    # Preserve requested primary order.
    primary_effect[
        "_order"
    ] = primary_effect[
        "Dataset_ID"
    ].map({
        d: i
        for i, d in enumerate(
            primary_datasets
        )
    })

    primary_effect = primary_effect.sort_values(
        "_order"
    )

    primary_horizon_row = (
        horizon_consistency.loc[
            horizon_consistency["Scope"]
            == "primary_external_tests"
        ].iloc[0]
    )

    all_horizon_row = (
        horizon_consistency.loc[
            horizon_consistency["Scope"]
            == "all_external_tests_including_2024_full"
        ].iloc[0]
    )

    matched_effect = primary_effect.loc[
        primary_effect["Dataset_ID"]
        == "SD1033_2024_matched_SD1090"
    ]

    matched_effect_row = (
        matched_effect.iloc[0]
        if not matched_effect.empty
        else None
    )

    # Ridge vs persistence in 2023 is useful context.
    ridge_2023 = get_row(
        seed_df,
        "SD1033_2023_full",
        MODEL_RIDGE,
    )

    persistence_2023 = get_row(
        seed_df,
        "SD1033_2023_full",
        MODEL_PERSISTENCE,
    )

    physics_2023 = get_row(
        seed_df,
        "SD1033_2023_full",
        MODEL_PHYSICS,
    )

    ridge_vs_persistence_2023 = (
        100.0
        * (
            metric_value(
                ridge_2023,
                "AW_RMSE_mps",
            )
            / metric_value(
                persistence_2023,
                "AW_RMSE_mps",
            )
            - 1.0
        )
    )

    physics_vs_persistence_2023 = (
        100.0
        * (
            1.0
            - metric_value(
                physics_2023,
                "AW_RMSE_mps",
            )
            / metric_value(
                persistence_2023,
                "AW_RMSE_mps",
            )
        )
    )

    physics_ref_pairs = paired_df.loc[
        (
            paired_df["dataset_id"].isin(
                primary_datasets
            )
        )
        & (
            paired_df["comparison"]
            == "PhysicsCompact_vs_Compact"
        )
    ]

    physics_ref_all_positive_mean = bool(
        (
            physics_ref_pairs[
                "mean_AW_reduction_B_vs_A_mps"
            ]
            > 0
        ).all()
    )

    physics_ref_ci_excludes_zero_count = int(
        (
            (
                physics_ref_pairs[
                    "paired_95pct_CI_low_mps"
                ]
                > 0
            )
            | (
                physics_ref_pairs[
                    "paired_95pct_CI_high_mps"
                ]
                < 0
            )
        ).sum()
    )

    summary_rows = []

    for _, row in primary_effect.iterrows():
        summary_rows.append({
            "Finding": (
                f"{row['Dataset']} Physics-Compact vs Ridge"
            ),
            "Value": (
                f"AW improvement "
                f"{row['AW_improvement_vs_Ridge_percent']:.3f}%; "
                f"vessel improvement "
                f"{row['Vessel_improvement_vs_Ridge_percent']:.3f}%"
            ),
        })

    summary_rows.extend([
        {
            "Finding": (
                "Primary dataset-horizon consistency"
            ),
            "Value": (
                f"{int(primary_horizon_row['PhysicsCompact_better_than_Ridge_count'])}/"
                f"{int(primary_horizon_row['Dataset_horizon_pairs'])} "
                f"dataset-horizon pairs improve over Ridge"
            ),
        },
        {
            "Finding": (
                "All dataset-horizon consistency including 2024 full"
            ),
            "Value": (
                f"{int(all_horizon_row['PhysicsCompact_better_than_Ridge_count'])}/"
                f"{int(all_horizon_row['Dataset_horizon_pairs'])} "
                f"dataset-horizon pairs improve over Ridge"
            ),
        },
        {
            "Finding": (
                "2023 frozen Ridge vs Persistence"
            ),
            "Value": (
                f"Ridge AW RMSE is "
                f"{ridge_vs_persistence_2023:.3f}% worse than Persistence"
            ),
        },
        {
            "Finding": (
                "2023 Physics-Compact vs Persistence"
            ),
            "Value": (
                f"Physics-Compact AW RMSE is "
                f"{physics_vs_persistence_2023:.3f}% better than Persistence"
            ),
        },
        {
            "Finding": (
                "Physics refinement direction"
            ),
            "Value": (
                "Positive mean Physics-Compact vs Compact AW refinement "
                f"on all primary tests = {physics_ref_all_positive_mean}; "
                f"paired seed CIs excluding zero = "
                f"{physics_ref_ci_excludes_zero_count}/"
                f"{len(physics_ref_pairs)}"
            ),
        },
    ])

    generalization_summary_df = (
        pd.DataFrame(
            summary_rows
        )
    )

    generalization_summary_df.to_csv(
        analysis_dir
        / "generalization_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    generalization_summary_json = {
        "primary_datasets": primary_datasets,
        "supplementary_datasets": [
            d
            for d in all_dataset_ids
            if d not in primary_datasets
        ],
        "physics_compact_primary_effects": (
            primary_effect[
                [
                    "Dataset",
                    "Dataset_ID",
                    "AW_improvement_vs_Ridge_percent",
                    "Vessel_improvement_vs_Ridge_percent",
                    "AWS_improvement_vs_Ridge_percent",
                    "AWA_improvement_vs_Ridge_percent",
                ]
            ].to_dict(
                orient="records"
            )
        ),
        "primary_horizon_consistency": (
            primary_horizon_row.to_dict()
        ),
        "all_horizon_consistency": (
            all_horizon_row.to_dict()
        ),
        "matched_2024_cross_vehicle": (
            matched_effect_row.to_dict()
            if matched_effect_row is not None
            else None
        ),
        "sd1033_2023": {
            "frozen_Ridge_AW_percent_worse_than_Persistence": (
                ridge_vs_persistence_2023
            ),
            "PhysicsCompact_AW_percent_better_than_Persistence": (
                physics_vs_persistence_2023
            ),
        },
        "physics_refinement": {
            "positive_mean_on_all_primary_tests": (
                physics_ref_all_positive_mean
            ),
            "paired_seed_CI_excludes_zero_count": (
                physics_ref_ci_excludes_zero_count
            ),
            "primary_test_count": int(
                len(
                    physics_ref_pairs
                )
            ),
            "interpretation": (
                "Physics supervision is a small directional refinement; "
                "the dominant external gain is attributed to the structured "
                "vessel-residual decomposition."
            ),
        },
        "statistical_policy": {
            "seed_variability": (
                "Five frozen initialization seeds; not mission-population CI."
            ),
            "shift_gain_correlation": (
                "Descriptive only; primary N=3."
            ),
        },
    }

    save_json(
        analysis_dir
        / "generalization_summary.json",
        generalization_summary_json,
    )

    # --------------------------------------------------------------
    # Figures
    # --------------------------------------------------------------
    if not args.no_plots:
        make_figures(
            main_table=main_table,
            horizon_skill=horizon_primary,
            effect_summary=effect_summary,
            shift_gain=shift_gain,
            output_dir=output_dir,
            primary_datasets=primary_datasets,
        )

    # --------------------------------------------------------------
    # Paper-ready text report
    # --------------------------------------------------------------
    report_path = (
        output_dir
        / "paper_ready_results.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "08D Paper-Ready External Generalization Results\n"
        )
        f.write("=" * 110 + "\n\n")

        f.write(
            "Recommended external-validation hierarchy\n"
        )
        f.write("-" * 110 + "\n")
        f.write(
            "Primary 1: SD1033-2024 matched SD1090 "
            "(same mission / matched time / different vehicle).\n"
        )
        f.write(
            "Primary 2: SD1033-2023 full "
            "(cross-year / cross-mission external generalization).\n"
        )
        f.write(
            "Primary 3: SD1033-2022 full "
            "(cross-year / cross-mission external generalization).\n"
        )
        f.write(
            "Supplementary: SD1033-2024 full "
            "(overlaps with the matched 2024 test and is therefore "
            "not treated as an independent primary external test).\n\n"
        )

        f.write(
            "Physics-Compact external effects\n"
        )
        f.write("-" * 110 + "\n")

        for _, row in primary_effect.iterrows():
            f.write(
                f"{row['Dataset']}: "
                f"AW RMSE improvement vs frozen Ridge = "
                f"{row['AW_improvement_vs_Ridge_percent']:.3f}%; "
                f"vessel RMSE improvement = "
                f"{row['Vessel_improvement_vs_Ridge_percent']:.3f}%; "
                f"AWS RMSE improvement = "
                f"{row['AWS_improvement_vs_Ridge_percent']:.3f}%; "
                f"AWA MAE improvement = "
                f"{row['AWA_improvement_vs_Ridge_percent']:.3f}%.\n"
            )

        f.write("\n")

        f.write(
            "Horizon consistency\n"
        )
        f.write("-" * 110 + "\n")

        f.write(
            f"Primary tests: "
            f"{int(primary_horizon_row['PhysicsCompact_better_than_Ridge_count'])}/"
            f"{int(primary_horizon_row['Dataset_horizon_pairs'])} "
            f"dataset-horizon combinations show lower AW RMSE than frozen Ridge.\n"
        )

        f.write(
            f"Including supplementary SD1033-2024 full: "
            f"{int(all_horizon_row['PhysicsCompact_better_than_Ridge_count'])}/"
            f"{int(all_horizon_row['Dataset_horizon_pairs'])} "
            f"dataset-horizon combinations show lower AW RMSE than frozen Ridge.\n"
        )

        f.write(
            f"Primary-test per-horizon skill range = "
            f"{primary_horizon_row['Minimum_PhysicsCompact_skill_vs_Ridge_percent']:.3f}% "
            f"to "
            f"{primary_horizon_row['Maximum_PhysicsCompact_skill_vs_Ridge_percent']:.3f}%.\n\n"
        )

        f.write(
            "Matched-period cross-vehicle result\n"
        )
        f.write("-" * 110 + "\n")

        if matched_effect_row is not None:
            f.write(
                f"On SD1033-2024 matched to the SD1090 valid period, "
                f"Physics-Compact reduces AW RMSE by "
                f"{matched_effect_row['AW_improvement_vs_Ridge_percent']:.3f}% "
                f"and vessel RMSE by "
                f"{matched_effect_row['Vessel_improvement_vs_Ridge_percent']:.3f}% "
                f"relative to frozen Ridge.\n"
            )

        f.write("\n")

        f.write(
            "Strong-shift 2023 result\n"
        )
        f.write("-" * 110 + "\n")

        f.write(
            f"Frozen Ridge is {ridge_vs_persistence_2023:.3f}% worse than "
            f"Persistence in AW RMSE on SD1033-2023, whereas the frozen "
            f"Physics-Compact model is {physics_vs_persistence_2023:.3f}% "
            f"better than Persistence. This supports a residual-correction "
            f"interpretation under a stronger external distribution shift.\n\n"
        )

        f.write(
            "Physics-supervision interpretation\n"
        )
        f.write("-" * 110 + "\n")

        f.write(
            "Physics-Compact has lower mean AW RMSE than Compact on all "
            "primary external tests, but the paired seed confidence intervals "
            f"exclude zero on only {physics_ref_ci_excludes_zero_count}/"
            f"{len(physics_ref_pairs)} primary tests. Therefore the dominant "
            "external benefit should be attributed to the structured "
            "vessel-residual decomposition, with physics supervision described "
            "as a small directional refinement rather than a statistically "
            "established dominant effect.\n\n"
        )

        f.write(
            "Distribution-shift analysis policy\n"
        )
        f.write("-" * 110 + "\n")

        f.write(
            "The shift-versus-gain analysis uses simple feature-wise z-score "
            "summaries under the frozen SD1090 TRAIN scaler. Correlations are "
            "descriptive only because only three primary external tests are "
            "available; no causal or inferential claim should be made.\n\n"
        )

        f.write(
            "Statistical wording\n"
        )
        f.write("-" * 110 + "\n")

        f.write(
            "All neural-model mean +/- standard deviation values and paired "
            "confidence intervals originate from five independently trained "
            "frozen initialization seeds. They quantify initialization "
            "variability and must not be described as mission-population "
            "confidence intervals.\n"
        )

    manifest = {
        "script_version": SCRIPT_VERSION,
        "analysis_only": True,
        "external_root": str(
            external_root
        ),
        "inference_dir": str(
            inference_dir
        ),
        "primary_datasets": primary_datasets,
        "supplementary_datasets": [
            d
            for d in all_dataset_ids
            if d not in primary_datasets
        ],
        "models": MODEL_ORDER,
        "input_files": {
            "external_seed_summary": str(
                seed_path
            ),
            "external_per_horizon_summary": str(
                horizon_path
            ),
            "external_paired_effects": str(
                paired_path
            ),
            "external_inference_audit": str(
                audit_path
            ),
            "feature_shift_summary": str(
                shift_path
            ),
            "external_dataset_summary": str(
                dataset_path
            ),
            "08C_manifest": str(
                inference_manifest_path
            ),
            "08B_manifest": str(
                external_manifest_path
            ),
        },
        "frozen_protocol_from_08C": (
            inference_manifest.get(
                "protocol",
                {}
            )
        ),
        "08B_frozen_protocol": (
            external_manifest.get(
                "frozen_protocol",
                {}
            )
        ),
        "paper_interpretation": {
            "dominant_gain": (
                "structured vessel-residual correction"
            ),
            "physics_loss_role": (
                "small directional refinement"
            ),
            "matched_2024_role": (
                "primary cross-vehicle same-mission/matched-period test"
            ),
            "2023_role": (
                "stronger cross-year distribution-shift robustness test"
            ),
            "2024_full_role": (
                "supplementary robustness test due to overlap with matched 2024"
            ),
            "shift_gain_correlation": (
                "descriptive only"
            ),
        },
        "outputs": {
            "tables_dir": "tables",
            "analysis_dir": "analysis",
            "figures_dir": "figures",
            "paper_ready_results": "paper_ready_results.txt",
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    save_json(
        output_dir
        / "external_generalization_analysis_manifest.json",
        manifest,
    )

    # --------------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------------
    log("")
    log(
        "=" * 110
    )
    log(
        "08D EXTERNAL GENERALIZATION ANALYSIS RESULTS"
    )
    log(
        "=" * 110
    )

    for _, row in primary_effect.iterrows():
        log(
            f"{row['Dataset']}:"
        )
        log(
            f"  Physics-Compact AW improvement vs Ridge     = "
            f"{row['AW_improvement_vs_Ridge_percent']:.3f}%"
        )
        log(
            f"  Physics-Compact vessel improvement vs Ridge = "
            f"{row['Vessel_improvement_vs_Ridge_percent']:.3f}%"
        )
        log(
            f"  Physics-Compact AWS improvement vs Ridge    = "
            f"{row['AWS_improvement_vs_Ridge_percent']:.3f}%"
        )

    log("")
    log(
        "[HORIZON CONSISTENCY - PRIMARY] "
        f"{int(primary_horizon_row['PhysicsCompact_better_than_Ridge_count'])}/"
        f"{int(primary_horizon_row['Dataset_horizon_pairs'])} "
        "dataset-horizon pairs improve over Ridge."
    )

    log(
        "[HORIZON CONSISTENCY - ALL] "
        f"{int(all_horizon_row['PhysicsCompact_better_than_Ridge_count'])}/"
        f"{int(all_horizon_row['Dataset_horizon_pairs'])} "
        "dataset-horizon pairs improve over Ridge."
    )

    log(
        "[2023 SHIFT TEST] Frozen Ridge vs Persistence AW change = "
        f"{ridge_vs_persistence_2023:+.3f}% "
        "(positive = Ridge worse)."
    )

    log(
        "[2023 SHIFT TEST] Physics-Compact vs Persistence AW improvement = "
        f"{physics_vs_persistence_2023:.3f}%."
    )

    log(
        "[PHYSICS REFINEMENT] Mean improvement over Compact is positive on "
        f"all primary tests = {physics_ref_all_positive_mean}; "
        f"paired seed CI excludes zero on "
        f"{physics_ref_ci_excludes_zero_count}/{len(physics_ref_pairs)}."
    )

    log(
        "[STATISTICAL POLICY] Seed CIs quantify frozen-initialization "
        "variability, not mission-population uncertainty."
    )

    log(
        "[SHIFT-GAIN POLICY] Correlations are descriptive only (primary N=3)."
    )

    log(
        f"[DONE] 08D outputs: {output_dir}"
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(
            main()
        )
    except Exception:
        print(
            "\n[FATAL ERROR]",
            flush=True,
        )
        traceback.print_exc()
        print(
            "\nPlease send the complete traceback and terminal output.",
            flush=True,
        )
        sys.exit(1)
