# -*- coding: utf-8 -*-
r"""
12E_generate_Nie_same_mission_paper_results.py

Stage 12E
=========
Paper-results / figure-generation stage for the frozen Stage-12D final
Tropical Atlantic evaluation.

NO MODEL TRAINING.
NO HYPERPARAMETER TUNING.
NO TEST-SET MODEL SELECTION.

This script reads ONLY already-generated Stage-12D final outputs and creates
manuscript-ready tables, figures, and conservative comparison wording.

Scientific positioning
----------------------
The comparison with Nie et al. (2026) is:
    - same raw Tropical Atlantic mission;
    - same 10-min temporal resolution;
    - same core wind outputs U/V and derived wind speed/direction;
but NOT:
    - exact sample-for-sample reproduction;
    - exact preprocessing reproduction;
    - exact official BP-STGNN rerun.

Therefore this script never reports a percentage "improvement over BP-STGNN".
It may report:
    "lower numerical errors than the published BP-STGNN values"
provided the final Stage-12D values actually satisfy that condition.

Complexity context
------------------
Published BP-STGNN:
    242,580 trainable parameters
    Bayesian physics-guided adjacency
    graph convolution
    GRU
    heteroscedastic mean/variance output
    uncertainty quantification

Physics-Compact-NieData v2:
    parameter count is read from the Stage-12D final checkpoints
    Persistence anchor
    GRU64
    Dense64
    gated wind/vessel residual heads
    deterministic kinematic apparent-wind consistency

Important caveat:
BP-STGNN additionally performs Bayesian uncertainty quantification, whereas the
present Physics-Compact-NieData v2 comparison is deterministic point forecasting
plus vessel/apparent-wind prediction. Parameter efficiency should therefore be
described for the deterministic forecasting role, not as an equivalent UQ model.

Default input
-------------
D:\project\WindPredict_SaildroneData\data\forecasting\
12D_PhysicsCompact_NieData_v2_TropicalAtlantic_FINAL_v0_1

Default output
--------------
D:\project\WindPredict_SaildroneData\data\forecasting\
12E_Nie_same_mission_paper_results_v0_1

Generated outputs
-----------------
tables/
    Table12E1_same_mission_wind_comparison.csv
    Table12E1_same_mission_wind_comparison.tex
    Table12E2_joint_forecast_comparison.csv
    Table12E2_joint_forecast_comparison.tex
    Table12E3_seed_stability.csv
    Table12E4_model_complexity.csv
    Table12E4_model_complexity.tex

figures/
    Fig12E1_same_mission_wind_RMSE.png/pdf
    Fig12E2_wind_direction_RMSE.png/pdf
    Fig12E3_parameter_count.png/pdf
    Fig12E4_U_scatter.png/pdf
    Fig12E5_V_scatter.png/pdf
    Fig12E6_wind_speed_scatter.png/pdf
    Fig12E7_apparent_wind_speed_timeseries_24h.png/pdf
    Fig12E8_apparent_wind_vector_error_CDF.png/pdf
    Fig12E9_joint_vector_RMSE.png/pdf

12E_manuscript_claims.md
12E_summary.json
12E_REPORT.txt

Figure style
------------
The script attempts to import:
    D:\project\WindPredict_SaildroneData\src\sci_plot_style.py

If unavailable/incompatible, it falls back to Times New Roman and publication
friendly Matplotlib defaults.

All figures:
    600 dpi PNG
    vector PDF
    no seaborn
    one figure per file (no multi-panel subplots)

Time-series figure policy
-------------------------
The apparent-wind time-series figure uses the FIRST 24 h-equivalent sequence
of accepted held-out test samples (144 samples at 10-min resolution), rather
than selecting a visually favorable interval.

Run
---
conda activate WindPredict
python "D:\project\WindPredict_SaildroneData\src\12E_generate_Nie_same_mission_paper_results.py"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_VERSION = "0.1.0-Nie-same-mission-paper-results"

DEFAULT_PROJECT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")

DEFAULT_12D_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12D_PhysicsCompact_NieData_v2_TropicalAtlantic_FINAL_v0_1"
)

DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12E_Nie_same_mission_paper_results_v0_1"
)

DEFAULT_STYLE_PATH = (
    DEFAULT_PROJECT_ROOT
    / "src"
    / "sci_plot_style.py"
)

FINAL_SEEDS = [500043, 501052, 502061, 503070, 504079]

BP_STGNN = {
    "method": "BP-STGNN (published)",
    "parameters": 242580,
    "U_RMSE_mps": 0.74864,
    "V_RMSE_mps": 0.70506,
    "WS_RMSE_mps": 0.69940,
    "WD_RMSE_deg": 5.58400,
    "architecture": (
        "Bayesian physics-guided adjacency + GCN + GRU + "
        "heteroscedastic mean/variance head"
    ),
    "uncertainty_quantification": True,
    "scope": "Published literature reference",
}

PROPOSED_NAME = "PhysicsCompactNieDataV2"

PLOT_LABELS = {
    "BP-STGNN (published)": "BP-STGNN",
    "Persistence": "Persistence",
    "Ridge": "Ridge",
    PROPOSED_NAME: "Physics-Compact v2",
}


def log(message=""):
    print(message, flush=True)


def save_json(path: Path, obj):
    def cv(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return None if np.isnan(v) else float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, dict):
            return {str(k): cv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [cv(x) for x in v]
        return v

    with path.open("w", encoding="utf-8") as f:
        json.dump(cv(obj), f, ensure_ascii=False, indent=2, sort_keys=True)


def apply_sci_style(style_path: Path):
    """
    Best-effort compatibility with the user's existing sci_plot_style.py.
    The project style API has changed across revisions, so we try common entry
    points and otherwise use a conservative local fallback.
    """
    used = False
    message = ""

    if style_path.exists():
        try:
            spec = importlib.util.spec_from_file_location(
                "sci_plot_style_12e",
                style_path,
            )
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)

            for name in [
                "apply_sci_plot_style",
                "set_sci_plot_style",
                "setup_sci_plot_style",
                "apply_style",
                "setup_style",
            ]:
                fn = getattr(module, name, None)

                if callable(fn):
                    fn()
                    used = True
                    message = f"Imported {style_path.name}:{name}()"
                    break

            if not used:
                message = (
                    f"{style_path.name} imported but no recognized style "
                    "function was found; local fallback applied."
                )

        except Exception as exc:
            message = (
                f"Could not apply {style_path}: {type(exc).__name__}: {exc}; "
                "local fallback applied."
            )

    if not used:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

        if not message:
            message = "Local Times/serif publication fallback applied."

    return used, message


def save_figure(fig, base_path: Path):
    fig.savefig(
        base_path.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        base_path.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)


def load_12d_tables(stage12d_dir: Path):
    method_path = stage12d_dir / "tropical_atlantic_method_summary.csv"
    seed_path = stage12d_dir / "tropical_atlantic_seed_results.csv"
    final_path = stage12d_dir / "final_results.json"

    for path in [method_path, seed_path, final_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    method_df = pd.read_csv(method_path)
    seed_df = pd.read_csv(seed_path)

    with final_path.open("r", encoding="utf-8") as f:
        final_json = json.load(f)

    return method_df, seed_df, final_json


def load_predictions(stage12d_dir: Path):
    prediction_dir = stage12d_dir / "test_predictions"
    base_path = prediction_dir / "baselines_TropicalAtlantic.npz"

    if not base_path.exists():
        raise FileNotFoundError(base_path)

    with np.load(base_path, allow_pickle=False) as d:
        required = [
            "y_true",
            "apparent_true",
            "persistence_y_pred",
            "ridge_y_pred",
            "context_end_time_ns",
            "target_time_ns",
        ]

        for key in required:
            if key not in d.files:
                raise KeyError(f"{base_path}: missing {key}")

        base = {
            key: np.asarray(d[key])
            for key in required
        }

    seed_preds = []

    for seed in FINAL_SEEDS:
        path = (
            prediction_dir
            / f"PhysicsCompactNieDataV2_seed_{seed}_TropicalAtlantic.npz"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as d:
            y_pred = np.asarray(d["y_pred"], dtype=np.float64)
            target_time = np.asarray(d["target_time_ns"], dtype=np.int64)

        if not np.array_equal(
            target_time,
            np.asarray(base["target_time_ns"], dtype=np.int64),
        ):
            raise RuntimeError(
                f"Prediction timestamp mismatch for seed={seed}"
            )

        seed_preds.append(y_pred)

    seed_stack = np.stack(seed_preds, axis=0)
    ensemble_mean = np.mean(seed_stack, axis=0)
    ensemble_std = np.std(seed_stack, axis=0, ddof=0)

    return {
        **base,
        "seed_pred_stack": seed_stack,
        "proposed_ensemble_mean": ensemble_mean,
        "proposed_ensemble_std": ensemble_std,
    }


def load_parameter_count(stage12d_dir: Path):
    """
    Read trainable parameter_count from all final checkpoint metadata.

    If torch is unavailable, fall back to the analytically known count for the
    frozen architecture and mark the source accordingly.
    """
    checkpoint_dir = stage12d_dir / "final_checkpoints"
    counts = []

    try:
        import torch

        for seed in FINAL_SEEDS:
            path = checkpoint_dir / f"PhysicsCompactNieDataV2_seed_{seed}.pt"

            if not path.exists():
                raise FileNotFoundError(path)

            try:
                ckpt = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                ckpt = torch.load(
                    path,
                    map_location="cpu",
                )

            if "parameter_count" in ckpt:
                counts.append(int(ckpt["parameter_count"]))
            elif "state_dict" in ckpt:
                counts.append(
                    int(
                        sum(
                            int(v.numel())
                            for v in ckpt["state_dict"].values()
                        )
                    )
                )

        if len(counts) != len(FINAL_SEEDS):
            raise RuntimeError(
                f"Expected {len(FINAL_SEEDS)} checkpoint counts, got {counts}"
            )

        if len(set(counts)) != 1:
            raise RuntimeError(
                f"Final checkpoints disagree on parameter count: {counts}"
            )

        return counts[0], "Stage-12D checkpoint metadata/state_dict"

    except Exception as exc:
        # Analytical count of the frozen architecture:
        # GRU(9,64): 3h*in + 3h*h + 2*(3h) = 14,400
        # Dense(64,64): 4,160
        # four Linear(64,2) heads: 4*130 = 520
        # total = 19,080.
        return 19080, (
            "Analytical frozen-architecture count fallback; "
            f"checkpoint read unavailable: {type(exc).__name__}: {exc}"
        )


def validate_method_summary(method_df: pd.DataFrame):
    needed = [
        "Persistence",
        "Ridge",
        PROPOSED_NAME,
    ]

    found = set(method_df["method"].astype(str).tolist())

    missing = [x for x in needed if x not in found]

    if missing:
        raise RuntimeError(
            f"12D method summary missing methods: {missing}"
        )


def row_for(method_df: pd.DataFrame, method: str):
    sub = method_df.loc[
        method_df["method"].astype(str) == method
    ]

    if len(sub) != 1:
        raise RuntimeError(
            f"Expected one row for {method}, found {len(sub)}"
        )

    return sub.iloc[0]


def build_same_mission_wind_table(
    method_df: pd.DataFrame,
    proposed_params: int,
):
    rows = [{
        "Method": "BP-STGNN (published)",
        "Trainable parameters": BP_STGNN["parameters"],
        "U RMSE (m/s)": BP_STGNN["U_RMSE_mps"],
        "V RMSE (m/s)": BP_STGNN["V_RMSE_mps"],
        "Wind-speed RMSE (m/s)": BP_STGNN["WS_RMSE_mps"],
        "Wind-direction RMSE (deg)": BP_STGNN["WD_RMSE_deg"],
        "Forecasting capability": "Wind + predictive uncertainty",
        "Comparison scope": (
            "Published same-mission literature reference; "
            "not exact sample-for-sample reproduction"
        ),
    }]

    for method in ["Persistence", "Ridge", PROPOSED_NAME]:
        r = row_for(method_df, method)

        params = (
            proposed_params
            if method == PROPOSED_NAME
            else 0
        )

        capability = {
            "Persistence": "Wind + vessel + derived apparent wind",
            "Ridge": "Wind + vessel + derived apparent wind",
            PROPOSED_NAME: (
                "Joint wind-vessel prediction + apparent-wind preview"
            ),
        }[method]

        rows.append({
            "Method": PLOT_LABELS[method],
            "Trainable parameters": params,
            "U RMSE (m/s)": float(r["wind_U_RMSE_mps_mean"]),
            "V RMSE (m/s)": float(r["wind_V_RMSE_mps_mean"]),
            "Wind-speed RMSE (m/s)": float(
                r["wind_speed_RMSE_mps_mean"]
            ),
            "Wind-direction RMSE (deg)": float(
                r["wind_direction_RMSE_deg_mean"]
            ),
            "Forecasting capability": capability,
            "Comparison scope": (
                "Stage-12B/12D Tropical Atlantic held-out samples"
            ),
        })

    return pd.DataFrame(rows)


def build_joint_table(method_df: pd.DataFrame):
    rows = []

    for method in ["Persistence", "Ridge", PROPOSED_NAME]:
        r = row_for(method_df, method)

        rows.append({
            "Method": PLOT_LABELS[method],
            "Wind-vector RMSE (m/s)": float(
                r["wind_vector_RMSE_mps_mean"]
            ),
            "Vessel-vector RMSE (m/s)": float(
                r["vessel_vector_RMSE_mps_mean"]
            ),
            "Apparent-wind-vector RMSE (m/s)": float(
                r["AW_vector_RMSE_mps_mean"]
            ),
            "Apparent-wind-speed RMSE (m/s)": float(
                r["AWS_RMSE_mps_mean"]
            ),
        })

    df = pd.DataFrame(rows)

    p = df.loc[df["Method"] == "Persistence"].iloc[0]
    r = df.loc[df["Method"] == "Ridge"].iloc[0]
    q = df.loc[df["Method"] == "Physics-Compact v2"].iloc[0]

    df["AW skill vs Persistence (%)"] = np.nan
    df["AW skill vs Ridge (%)"] = np.nan

    mask = df["Method"] == "Physics-Compact v2"

    df.loc[mask, "AW skill vs Persistence (%)"] = (
        100.0
        * (
            1.0
            - q["Apparent-wind-vector RMSE (m/s)"]
            / p["Apparent-wind-vector RMSE (m/s)"]
        )
    )

    df.loc[mask, "AW skill vs Ridge (%)"] = (
        100.0
        * (
            1.0
            - q["Apparent-wind-vector RMSE (m/s)"]
            / r["Apparent-wind-vector RMSE (m/s)"]
        )
    )

    return df


def build_complexity_table(proposed_params: int):
    reduction = (
        1.0
        - proposed_params / BP_STGNN["parameters"]
    ) * 100.0

    ratio = BP_STGNN["parameters"] / proposed_params

    rows = [
        {
            "Method": "BP-STGNN (published)",
            "Trainable parameters": BP_STGNN["parameters"],
            "Relative parameter count": 1.0,
            "Parameter reduction vs BP-STGNN (%)": 0.0,
            "BP-STGNN / method parameter ratio": 1.0,
            "Core structure": BP_STGNN["architecture"],
            "Predictive uncertainty": "Yes",
        },
        {
            "Method": "Physics-Compact v2",
            "Trainable parameters": proposed_params,
            "Relative parameter count": proposed_params / BP_STGNN["parameters"],
            "Parameter reduction vs BP-STGNN (%)": reduction,
            "BP-STGNN / method parameter ratio": ratio,
            "Core structure": (
                "Persistence anchor + GRU64 + Dense64 + "
                "gated wind/vessel residual heads + A=W-V"
            ),
            "Predictive uncertainty": "No",
        },
    ]

    return pd.DataFrame(rows)


def latex_escape(text):
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def write_simple_latex_table(
    df: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
):
    numeric_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{latex_escape(caption)}}}\n")
        f.write(f"\\label{{{latex_escape(label)}}}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write(
            "\\begin{tabular}{"
            + "l"
            + "r" * (len(df.columns) - 1)
            + "}\n"
        )
        f.write("\\hline\n")
        f.write(
            " & ".join(
                latex_escape(c)
                for c in df.columns
            )
            + " \\\\\n"
        )
        f.write("\\hline\n")

        for _, row in df.iterrows():
            cells = []

            for col in df.columns:
                value = row[col]

                if pd.isna(value):
                    cells.append("--")
                elif col in numeric_cols:
                    if "parameters" in col.lower():
                        cells.append(f"{int(round(float(value))):,}")
                    else:
                        cells.append(f"{float(value):.4f}")
                else:
                    cells.append(latex_escape(value))

            f.write(" & ".join(cells) + " \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")
        f.write("\\end{table}\n")


def wind_speed_from_y(y):
    return np.linalg.norm(
        np.asarray(y, dtype=np.float64)[:, 0, 0:2],
        axis=1,
    )


def apparent_speed_from_y(y):
    y = np.asarray(y, dtype=np.float64)
    aw = y[:, :, 0:2] - y[:, :, 2:4]
    return np.linalg.norm(aw[:, 0, :], axis=1)


def apparent_vector_error(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    at = yt[:, 0, 0:2] - yt[:, 0, 2:4]
    ap = yp[:, 0, 0:2] - yp[:, 0, 2:4]

    return np.linalg.norm(
        ap - at,
        axis=1,
    )


def scatter_identity_figure(
    x,
    y,
    xlabel,
    ylabel,
    title,
    out_base,
):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    lo = float(np.nanmin([np.nanmin(x), np.nanmin(y)]))
    hi = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))

    pad = 0.03 * max(
        hi - lo,
        1e-6,
    )

    lo -= pad
    hi += pad

    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    ax.scatter(
        x,
        y,
        s=7,
        alpha=0.35,
        linewidths=0,
    )

    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    save_figure(
        fig,
        out_base,
    )


def make_figures(
    figures_dir: Path,
    wind_table: pd.DataFrame,
    joint_table: pd.DataFrame,
    complexity_table: pd.DataFrame,
    predictions: Dict,
):
    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Fig 1: U/V/WS RMSE grouped bars
    # ------------------------------------------------------------------
    methods = wind_table["Method"].tolist()
    metrics = [
        "U RMSE (m/s)",
        "V RMSE (m/s)",
        "Wind-speed RMSE (m/s)",
    ]
    short = ["U", "V", "Wind speed"]

    x = np.arange(len(methods), dtype=float)
    width = 0.24

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    for j, (metric, label) in enumerate(zip(metrics, short)):
        values = wind_table[metric].to_numpy(dtype=float)

        ax.bar(
            x + (j - 1) * width,
            values,
            width=width,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=12, ha="right")
    ax.set_ylabel("RMSE (m/s)")
    ax.set_title(
        "Same-mission Tropical Atlantic wind forecasting"
    )
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)

    save_figure(
        fig,
        figures_dir / "Fig12E1_same_mission_wind_RMSE",
    )

    # ------------------------------------------------------------------
    # Fig 2: WD RMSE
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    ax.bar(
        methods,
        wind_table["Wind-direction RMSE (deg)"].to_numpy(dtype=float),
    )

    ax.set_ylabel("Wind-direction RMSE (deg)")
    ax.set_title(
        "Same-mission Tropical Atlantic wind-direction error"
    )
    ax.tick_params(axis="x", rotation=12)
    ax.grid(True, axis="y", alpha=0.25)

    save_figure(
        fig,
        figures_dir / "Fig12E2_wind_direction_RMSE",
    )

    # ------------------------------------------------------------------
    # Fig 3: parameter count
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.4, 4.0))

    ax.bar(
        complexity_table["Method"],
        complexity_table["Trainable parameters"].to_numpy(dtype=float),
    )

    ax.set_yscale("log")
    ax.set_ylabel("Trainable parameters (log scale)")
    ax.set_title("Model parameter count")
    ax.tick_params(axis="x", rotation=10)
    ax.grid(True, axis="y", alpha=0.25)

    for i, v in enumerate(
        complexity_table["Trainable parameters"].to_numpy(dtype=int)
    ):
        ax.text(
            i,
            float(v) * 1.10,
            f"{v:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    save_figure(
        fig,
        figures_dir / "Fig12E3_parameter_count",
    )

    y_true = np.asarray(predictions["y_true"], dtype=np.float64)
    proposed = np.asarray(
        predictions["proposed_ensemble_mean"],
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # Fig 4: U scatter
    # ------------------------------------------------------------------
    scatter_identity_figure(
        y_true[:, 0, 0],
        proposed[:, 0, 0],
        "Observed U (m/s)",
        "Predicted U (m/s)",
        "Physics-Compact v2: U-component prediction",
        figures_dir / "Fig12E4_U_scatter",
    )

    # ------------------------------------------------------------------
    # Fig 5: V scatter
    # ------------------------------------------------------------------
    scatter_identity_figure(
        y_true[:, 0, 1],
        proposed[:, 0, 1],
        "Observed V (m/s)",
        "Predicted V (m/s)",
        "Physics-Compact v2: V-component prediction",
        figures_dir / "Fig12E5_V_scatter",
    )

    # ------------------------------------------------------------------
    # Fig 6: wind-speed scatter
    # ------------------------------------------------------------------
    scatter_identity_figure(
        wind_speed_from_y(y_true),
        wind_speed_from_y(proposed),
        "Observed wind speed (m/s)",
        "Predicted wind speed (m/s)",
        "Physics-Compact v2: wind-speed prediction",
        figures_dir / "Fig12E6_wind_speed_scatter",
    )

    persistence = np.asarray(
        predictions["persistence_y_pred"],
        dtype=np.float64,
    )
    ridge = np.asarray(
        predictions["ridge_y_pred"],
        dtype=np.float64,
    )

    target_time = np.asarray(
        predictions["target_time_ns"],
        dtype=np.int64,
    ).reshape(-1)

    target_dt = pd.to_datetime(
        target_time,
        unit="ns",
    )

    # ------------------------------------------------------------------
    # Fig 7: first 24h-equivalent AW-speed time series
    # ------------------------------------------------------------------
    n_window = min(144, len(y_true))
    sl = slice(0, n_window)

    fig, ax = plt.subplots(figsize=(9.0, 4.0))

    ax.plot(
        target_dt[sl],
        apparent_speed_from_y(y_true)[sl],
        label="Observed",
        linewidth=1.3,
    )

    ax.plot(
        target_dt[sl],
        apparent_speed_from_y(persistence)[sl],
        label="Persistence",
        linewidth=1.0,
    )

    ax.plot(
        target_dt[sl],
        apparent_speed_from_y(ridge)[sl],
        label="Ridge",
        linewidth=1.0,
    )

    ax.plot(
        target_dt[sl],
        apparent_speed_from_y(proposed)[sl],
        label="Physics-Compact v2",
        linewidth=1.2,
    )

    ax.set_xlabel("UTC time")
    ax.set_ylabel("Apparent-wind speed (m/s)")
    ax.set_title(
        "Held-out Tropical Atlantic apparent-wind preview "
        "(first 24 h-equivalent accepted samples)"
    )
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()

    save_figure(
        fig,
        figures_dir / "Fig12E7_apparent_wind_speed_timeseries_24h",
    )

    # ------------------------------------------------------------------
    # Fig 8: AW vector-error empirical CDF
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 4.2))

    for label, pred in [
        ("Persistence", persistence),
        ("Ridge", ridge),
        ("Physics-Compact v2", proposed),
    ]:
        e = np.sort(
            apparent_vector_error(
                y_true,
                pred,
            )
        )

        cdf = (
            np.arange(
                1,
                len(e) + 1,
                dtype=float,
            )
            / len(e)
        )

        ax.plot(
            e,
            cdf,
            label=label,
        )

    ax.set_xlabel("Apparent-wind vector error (m/s)")
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        "Distribution of held-out apparent-wind vector error"
    )
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    save_figure(
        fig,
        figures_dir / "Fig12E8_apparent_wind_vector_error_CDF",
    )

    # ------------------------------------------------------------------
    # Fig 9: Wind/vessel/AW vector RMSE
    # ------------------------------------------------------------------
    methods_joint = joint_table["Method"].tolist()
    metrics_joint = [
        "Wind-vector RMSE (m/s)",
        "Vessel-vector RMSE (m/s)",
        "Apparent-wind-vector RMSE (m/s)",
    ]
    labels_joint = ["True wind", "Vessel", "Apparent wind"]

    x = np.arange(
        len(methods_joint),
        dtype=float,
    )
    width = 0.24

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    for j, (metric, label) in enumerate(
        zip(
            metrics_joint,
            labels_joint,
        )
    ):
        ax.bar(
            x + (j - 1) * width,
            joint_table[metric].to_numpy(dtype=float),
            width=width,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods_joint)
    ax.set_ylabel("Vector RMSE (m/s)")
    ax.set_title(
        "Joint wind-vessel-apparent-wind forecasting"
    )
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)

    save_figure(
        fig,
        figures_dir / "Fig12E9_joint_vector_RMSE",
    )


def build_claims(
    wind_table: pd.DataFrame,
    complexity_table: pd.DataFrame,
):
    proposed = wind_table.loc[
        wind_table["Method"]
        == "Physics-Compact v2"
    ].iloc[0]

    bp = wind_table.loc[
        wind_table["Method"]
        == "BP-STGNN (published)"
    ].iloc[0]

    c = complexity_table.loc[
        complexity_table["Method"]
        == "Physics-Compact v2"
    ].iloc[0]

    lower_all = all([
        proposed["U RMSE (m/s)"] < bp["U RMSE (m/s)"],
        proposed["V RMSE (m/s)"] < bp["V RMSE (m/s)"],
        proposed["Wind-speed RMSE (m/s)"] < bp["Wind-speed RMSE (m/s)"],
        proposed["Wind-direction RMSE (deg)"]
        < bp["Wind-direction RMSE (deg)"],
    ])

    reduction = float(
        c["Parameter reduction vs BP-STGNN (%)"]
    )
    ratio = float(
        c["BP-STGNN / method parameter ratio"]
    )
    params = int(
        c["Trainable parameters"]
    )

    if lower_all:
        english = (
            f"Despite using only {params:,} trainable parameters "
            f"({reduction:.1f}% fewer, or approximately {ratio:.1f}x fewer "
            "parameters than the 242,580 parameters reported for BP-STGNN), "
            "Physics-Compact-NieData v2 yielded lower numerical RMSE values "
            "for the U and V wind components, synthesized wind speed, and "
            "wind direction on the same Tropical Atlantic mission at a "
            "10-min temporal resolution. Because several preprocessing and "
            "development-split details of BP-STGNN are under-specified, this "
            "result is interpreted as a same-mission, paper-informed "
            "comparison rather than an exact sample-for-sample head-to-head "
            "benchmark."
        )

        chinese = (
            f"Physics-Compact-NieData v2仅包含{params:,}个可训练参数，"
            f"相比BP-STGNN公开报告的242,580个参数减少约{reduction:.1f}%"
            f"（约为其1/{ratio:.1f}）。在相同Tropical Atlantic航次和"
            "10 min时间尺度下，本模型在U/V风分量、合成风速和风向RMSE上"
            "均取得了低于BP-STGNN已发表结果的数值。由于BP-STGNN的若干"
            "预处理细节和开发集内部划分并未完全公开，因此该结果应表述为"
            "同航次、论文协议对齐的文献比较，而非严格逐样本的直接复现对比。"
        )

    else:
        english = (
            f"Physics-Compact-NieData v2 uses only {params:,} trainable "
            f"parameters ({reduction:.1f}% fewer than BP-STGNN), while "
            "remaining in the same forecasting-error regime on the same "
            "Tropical Atlantic mission. Because the protocols are not exactly "
            "sample-aligned, no categorical superiority claim is made."
        )

        chinese = (
            f"Physics-Compact-NieData v2仅包含{params:,}个可训练参数，"
            f"相比BP-STGNN减少约{reduction:.1f}%，并在同一Tropical "
            "Atlantic航次上保持相近的预测误差水平。由于两者预处理协议"
            "并非严格逐样本一致，不应作绝对优越性表述。"
        )

    not_allowed = [
        "Our model improves BP-STGNN by XX%.",
        "Our model strictly outperforms BP-STGNN under identical conditions.",
        "This experiment exactly reproduces BP-STGNN.",
        (
            "Our model is universally more efficient than BP-STGNN. "
            "BP-STGNN also performs Bayesian uncertainty quantification."
        ),
    ]

    return {
        "lower_all_four_published_wind_metrics": bool(lower_all),
        "recommended_english": english,
        "recommended_chinese": chinese,
        "not_recommended_claims": not_allowed,
        "parameter_reduction_percent": reduction,
        "BP_STGNN_to_proposed_parameter_ratio": ratio,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        default=str(DEFAULT_PROJECT_ROOT),
    )

    parser.add_argument(
        "--stage12d-dir",
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--style-path",
        default=None,
    )

    args = parser.parse_args()

    project_root = Path(args.project_root)

    stage12d_dir = (
        Path(args.stage12d_dir)
        if args.stage12d_dir
        else project_root
        / "data"
        / "forecasting"
        / "12D_PhysicsCompact_NieData_v2_TropicalAtlantic_FINAL_v0_1"
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root
        / "data"
        / "forecasting"
        / "12E_Nie_same_mission_paper_results_v0_1"
    )

    style_path = (
        Path(args.style_path)
        if args.style_path
        else project_root
        / "src"
        / "sci_plot_style.py"
    )

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    for d in [
        output_dir,
        tables_dir,
        figures_dir,
    ]:
        d.mkdir(
            parents=True,
            exist_ok=True,
        )

    log("=" * 122)
    log("12E - NIE SAME-MISSION PAPER RESULTS / FIGURES")
    log("=" * 122)
    log(f"script version        : {SCRIPT_VERSION}")
    log(f"Stage-12D dir         : {stage12d_dir}")
    log(f"output dir            : {output_dir}")
    log("model training        : NONE")
    log("model selection       : NONE")
    log("test predictions      : READ-ONLY")
    log("")

    style_used, style_message = apply_sci_style(
        style_path
    )
    log(f"plot style            : {style_message}")

    method_df, seed_df, final_json = load_12d_tables(
        stage12d_dir
    )
    validate_method_summary(
        method_df
    )

    predictions = load_predictions(
        stage12d_dir
    )

    proposed_params, param_source = load_parameter_count(
        stage12d_dir
    )

    log(
        f"Physics-Compact params: {proposed_params:,} "
        f"({param_source})"
    )
    log(
        f"BP-STGNN params       : {BP_STGNN['parameters']:,} "
        "(published reference)"
    )

    wind_table = build_same_mission_wind_table(
        method_df,
        proposed_params,
    )

    joint_table = build_joint_table(
        method_df
    )

    complexity_table = build_complexity_table(
        proposed_params
    )

    seed_table = seed_df.copy()

    wind_csv = (
        tables_dir
        / "Table12E1_same_mission_wind_comparison.csv"
    )
    joint_csv = (
        tables_dir
        / "Table12E2_joint_forecast_comparison.csv"
    )
    seed_csv = (
        tables_dir
        / "Table12E3_seed_stability.csv"
    )
    complexity_csv = (
        tables_dir
        / "Table12E4_model_complexity.csv"
    )

    wind_table.to_csv(
        wind_csv,
        index=False,
        encoding="utf-8-sig",
    )

    joint_table.to_csv(
        joint_csv,
        index=False,
        encoding="utf-8-sig",
    )

    seed_table.to_csv(
        seed_csv,
        index=False,
        encoding="utf-8-sig",
    )

    complexity_table.to_csv(
        complexity_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # Compact LaTeX tables containing the columns most useful in the paper.
    wind_tex_df = wind_table[[
        "Method",
        "Trainable parameters",
        "U RMSE (m/s)",
        "V RMSE (m/s)",
        "Wind-speed RMSE (m/s)",
        "Wind-direction RMSE (deg)",
    ]]

    joint_tex_df = joint_table[[
        "Method",
        "Wind-vector RMSE (m/s)",
        "Vessel-vector RMSE (m/s)",
        "Apparent-wind-vector RMSE (m/s)",
        "Apparent-wind-speed RMSE (m/s)",
    ]]

    complexity_tex_df = complexity_table[[
        "Method",
        "Trainable parameters",
        "Parameter reduction vs BP-STGNN (%)",
        "BP-STGNN / method parameter ratio",
        "Predictive uncertainty",
    ]]

    write_simple_latex_table(
        wind_tex_df,
        tables_dir / "Table12E1_same_mission_wind_comparison.tex",
        (
            "Same-mission Tropical Atlantic wind-forecasting comparison. "
            "Published BP-STGNN values are literature references rather than "
            "an exact sample-for-sample reproduction."
        ),
        "tab:same_mission_wind",
    )

    write_simple_latex_table(
        joint_tex_df,
        tables_dir / "Table12E2_joint_forecast_comparison.tex",
        "Held-out Tropical Atlantic joint forecasting performance.",
        "tab:joint_forecast",
    )

    write_simple_latex_table(
        complexity_tex_df,
        tables_dir / "Table12E4_model_complexity.tex",
        "Model complexity comparison for deterministic point forecasting.",
        "tab:model_complexity",
    )

    make_figures(
        figures_dir,
        wind_table,
        joint_table,
        complexity_table,
        predictions,
    )

    claims = build_claims(
        wind_table,
        complexity_table,
    )

    claims_path = (
        output_dir
        / "12E_manuscript_claims.md"
    )

    with claims_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("# Stage 12E manuscript-safe claims\n\n")

        f.write("## Recommended English wording\n\n")
        f.write(claims["recommended_english"] + "\n\n")

        f.write("## 推荐中文表述\n\n")
        f.write(claims["recommended_chinese"] + "\n\n")

        f.write("## Claims to avoid\n\n")

        for item in claims["not_recommended_claims"]:
            f.write(f"- {item}\n")

        f.write("\n## Complexity interpretation\n\n")
        f.write(
            f"- BP-STGNN published trainable parameters: "
            f"{BP_STGNN['parameters']:,}\n"
        )
        f.write(
            f"- Physics-Compact-NieData v2 trainable parameters: "
            f"{proposed_params:,}\n"
        )
        f.write(
            f"- Parameter reduction: "
            f"{claims['parameter_reduction_percent']:.2f}%\n"
        )
        f.write(
            f"- BP-STGNN has approximately "
            f"{claims['BP_STGNN_to_proposed_parameter_ratio']:.2f}x "
            "the trainable parameter count.\n"
        )
        f.write(
            "- Caveat: BP-STGNN includes Bayesian graph learning and "
            "predictive uncertainty quantification; Physics-Compact-NieData "
            "v2 is a compact deterministic point forecaster that additionally "
            "predicts vessel motion and derives future apparent wind.\n"
        )

    summary = {
        "stage": "12E",
        "script_version": SCRIPT_VERSION,
        "source_stage12d_dir": str(stage12d_dir),
        "test_predictions_retrained": False,
        "test_model_selection_performed": False,
        "proposed_parameter_count": proposed_params,
        "proposed_parameter_count_source": param_source,
        "BP_STGNN_published_parameter_count": BP_STGNN["parameters"],
        "parameter_reduction_percent": claims[
            "parameter_reduction_percent"
        ],
        "BP_STGNN_to_proposed_parameter_ratio": claims[
            "BP_STGNN_to_proposed_parameter_ratio"
        ],
        "lower_all_four_published_wind_metrics": claims[
            "lower_all_four_published_wind_metrics"
        ],
        "comparison_label": (
            "same-mission / paper-informed; not exact sample-for-sample "
            "BP-STGNN reproduction"
        ),
        "recommended_english_claim": claims[
            "recommended_english"
        ],
        "recommended_chinese_claim": claims[
            "recommended_chinese"
        ],
    }

    save_json(
        output_dir
        / "12E_summary.json",
        summary,
    )

    with (
        output_dir
        / "12E_REPORT.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "12E Nie Same-Mission Paper Results\n"
        )
        f.write(
            "="
            * 122
            + "\n\n"
        )

        f.write(
            "SCIENTIFIC STATUS\n"
        )
        f.write(
            "-"
            * 122
            + "\n"
        )
        f.write(
            "No model training, retuning, or test-set model selection was "
            "performed in Stage 12E.\n"
        )
        f.write(
            "All paper figures are generated from the already frozen Stage-12D "
            "held-out predictions.\n\n"
        )

        f.write(
            "SAME-MISSION WIND TABLE\n"
        )
        f.write(
            "-"
            * 122
            + "\n"
        )
        f.write(
            wind_table.to_string(
                index=False
            )
        )

        f.write(
            "\n\nJOINT FORECAST TABLE\n"
        )
        f.write(
            "-"
            * 122
            + "\n"
        )
        f.write(
            joint_table.to_string(
                index=False
            )
        )

        f.write(
            "\n\nCOMPLEXITY TABLE\n"
        )
        f.write(
            "-"
            * 122
            + "\n"
        )
        f.write(
            complexity_table.to_string(
                index=False
            )
        )

        f.write(
            "\n\nRECOMMENDED CLAIM\n"
        )
        f.write(
            "-"
            * 122
            + "\n"
        )
        f.write(
            claims[
                "recommended_english"
            ]
            + "\n"
        )

    log("")
    log("=" * 122)
    log("12E PAPER-RESULTS SUMMARY")
    log("=" * 122)

    proposed_wind = wind_table.loc[
        wind_table["Method"]
        == "Physics-Compact v2"
    ].iloc[0]

    log(
        "Physics-Compact v2 wind: "
        f"U={proposed_wind['U RMSE (m/s)']:.4f} | "
        f"V={proposed_wind['V RMSE (m/s)']:.4f} | "
        f"WS={proposed_wind['Wind-speed RMSE (m/s)']:.4f} | "
        f"WD={proposed_wind['Wind-direction RMSE (deg)']:.2f} deg"
    )

    log(
        "BP-STGNN published     : "
        f"U={BP_STGNN['U_RMSE_mps']:.4f} | "
        f"V={BP_STGNN['V_RMSE_mps']:.4f} | "
        f"WS={BP_STGNN['WS_RMSE_mps']:.4f} | "
        f"WD={BP_STGNN['WD_RMSE_deg']:.2f} deg"
    )

    log(
        f"Parameters             : proposed={proposed_params:,} | "
        f"BP-STGNN={BP_STGNN['parameters']:,}"
    )

    log(
        f"Parameter reduction    : "
        f"{claims['parameter_reduction_percent']:.2f}% "
        f"(BP-STGNN/proposed="
        f"{claims['BP_STGNN_to_proposed_parameter_ratio']:.2f}x)"
    )

    if claims[
        "lower_all_four_published_wind_metrics"
    ]:
        log(
            "[PASS] Proposed model has lower numerical errors than the "
            "published BP-STGNN values for U, V, WS, and WD."
        )
    else:
        log(
            "[NOTE] Proposed model does not have lower numerical values "
            "than BP-STGNN for all four wind metrics."
        )

    log(
        "[WORDING] Use 'same-mission, paper-informed comparison' and "
        "'lower numerical errors than the published values'."
    )

    log(
        "[AVOID] Do not claim exact BP-STGNN reproduction or an exact "
        "XX% improvement over BP-STGNN."
    )

    log(
        f"[DONE] 12E outputs: {output_dir}"
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
        sys.exit(
            1
        )
