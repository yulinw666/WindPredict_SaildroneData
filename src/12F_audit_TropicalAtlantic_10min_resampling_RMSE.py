# -*- coding: utf-8 -*-
r"""
12F_audit_TropicalAtlantic_10min_resampling_RMSE.py

Stage 12F
=========
Forensic audit of why the reconstructed Tropical Atlantic 10-min task gives
much lower Persistence RMSE than the published BP-STGNN values.

NO TRAINING.
NO MODEL SELECTION.
NO HYPERPARAMETER TUNING.
NO ACCESS TO DEVELOPMENT MISSIONS IS REQUIRED.

Scientific questions
--------------------
Q1. Can we exactly reproduce the current frozen 12B/12D Persistence metrics
    from the saved Tropical_Atlantic_TEST.npz?

Q2. On the EXACT SAME 6,591 frozen 12B test samples, how much does Persistence
    RMSE change if each 10-min block is represented by:
        - arithmetic mean of the 10 raw 1-min values;
        - minute phase 0, 1, ..., 9 inside each UTC-aligned 10-min block?

    This isolates the effect of the 10-min aggregation operator while keeping
    the evaluation timestamps identical.

Q3. How much does sample filtering matter?
    We additionally compare:
        - the frozen 12B sample set;
        - all strict UV-complete 10-min blocks with 7-block continuity;
        - all strict COMMON_BASE-complete 10-min blocks with 7-block continuity.

    The latter two do NOT reproduce the 12B dT/dP/dRH eligibility rule; they
    are diagnostic only.

Expected interpretation
-----------------------
If the 10-min arithmetic mean produces much lower Persistence RMSE than
single-minute phase sampling on the same timestamps, then the aggregation
operator is an important explanation for the low reconstructed error.

If changing the sample-eligibility rule changes the error substantially, then
sample filtering is also important.

If neither effect is large enough to approach the published BP-STGNN error
scale, then the remaining discrepancy is more likely to involve other
under-specified protocol details (gap treatment, exact sample selection,
history definition, metric aggregation, etc.).

Default project
---------------
D:\project\WindPredict_SaildroneData

Required existing files
-----------------------
src\12B_build_Nie_same_mission_benchmark_dataset.py

data\forecasting\12B_Nie_same_mission_benchmark_v0_1\
    track_J_joint_compatible\Tropical_Atlantic_TEST.npz

Raw data are discovered through the exact Stage-12B mission-discovery logic.

Outputs
-------
data\forecasting\12F_TropicalAtlantic_10min_resampling_audit_v0_1\
    exact_saved_persistence_metrics.csv
    same_frozen_samples_resampling_sensitivity.csv
    phase_sensitivity_same_frozen_samples.csv
    sample_filtering_sensitivity.csv
    frozen_sample_raw_reconstruction_audit.csv
    Fig12F1_same_samples_phase_sensitivity.png/pdf
    audit_report.txt

Run
---
conda activate WindPredict
python "D:\project\WindPredict_SaildroneData\src\12F_audit_TropicalAtlantic_10min_resampling_RMSE.py"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


SCRIPT_VERSION = "0.1.0-forensic-resampling-audit"

DEFAULT_PROJECT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")
DEFAULT_12B_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12B_Nie_same_mission_benchmark_v0_1"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12F_TropicalAtlantic_10min_resampling_audit_v0_1"
)

ONE_MIN_NS = 60 * 1_000_000_000
TEN_MIN_NS = 10 * ONE_MIN_NS
LOOKBACK_STEPS = 6

PUBLISHED_BPSTGNN = {
    "U_RMSE_mps": 0.74864,
    "V_RMSE_mps": 0.70506,
    "WS_RMSE_mps": 0.69940,
    "WD_RMSE_deg": 5.584,
}


def log(msg=""):
    print(msg, flush=True)


# =============================================================================
# Style
# =============================================================================

def apply_style(project_root: Path):
    style_path = project_root / "src" / "sci_plot_style.py"

    if style_path.exists():
        try:
            spec = importlib.util.spec_from_file_location(
                "sci_plot_style_12f",
                str(style_path),
            )
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)

            for name, kwargs in [
                ("apply_sci_style", {"base_font_size": 8.5}),
                ("apply_sci_style", {"font_size": 8.5}),
                ("apply_sci_style", {}),
                ("apply_sci_plot_style", {}),
                ("set_sci_plot_style", {}),
            ]:
                fn = getattr(mod, name, None)
                if callable(fn):
                    try:
                        fn(**kwargs)
                        log(f"[STYLE] {style_path.name}:{name}{kwargs}")
                        return
                    except TypeError:
                        continue
        except Exception as exc:
            log(f"[STYLE WARNING] {type(exc).__name__}: {exc}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def finish_axis(ax):
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
    )
    ax.grid(True, alpha=0.20, linewidth=0.6)


def save_figure(fig, base: Path):
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
    log(f"[SAVED] {base.with_suffix('.png')}")
    log(f"[SAVED] {base.with_suffix('.pdf')}")


# =============================================================================
# Load Stage 12B module so the raw mission discovery/variable mapping is exact.
# =============================================================================

def load_stage12b_module(project_root: Path):
    path = project_root / "src" / "12B_build_Nie_same_mission_benchmark_dataset.py"

    if not path.exists():
        raise FileNotFoundError(
            "Stage-12B builder not found:\n"
            f"  {path}\n"
            "12F intentionally reuses its exact raw-variable mapping and "
            "mission-discovery logic."
        )

    spec = importlib.util.spec_from_file_location("stage12b_builder", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    return mod


# =============================================================================
# Metrics
# =============================================================================

def wrap_deg180(x):
    x = np.asarray(x, dtype=np.float64)
    return (x + 180.0) % 360.0 - 180.0


def wind_direction_from_uv(U, V):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    return np.rad2deg(np.arctan2(-U, -V)) % 360.0


def compute_metrics(
    pred_u,
    pred_v,
    true_u,
    true_v,
):
    pred_u = np.asarray(pred_u, dtype=np.float64).reshape(-1)
    pred_v = np.asarray(pred_v, dtype=np.float64).reshape(-1)
    true_u = np.asarray(true_u, dtype=np.float64).reshape(-1)
    true_v = np.asarray(true_v, dtype=np.float64).reshape(-1)

    finite = (
        np.isfinite(pred_u)
        & np.isfinite(pred_v)
        & np.isfinite(true_u)
        & np.isfinite(true_v)
    )

    pred_u = pred_u[finite]
    pred_v = pred_v[finite]
    true_u = true_u[finite]
    true_v = true_v[finite]

    if len(true_u) == 0:
        raise RuntimeError("No finite samples available for metric computation.")

    du = pred_u - true_u
    dv = pred_v - true_v

    u_rmse = float(np.sqrt(np.mean(du ** 2)))
    v_rmse = float(np.sqrt(np.mean(dv ** 2)))
    vector_rmse = float(np.sqrt(np.mean(du ** 2 + dv ** 2)))

    pred_ws = np.hypot(pred_u, pred_v)
    true_ws = np.hypot(true_u, true_v)
    ws_rmse = float(np.sqrt(np.mean((pred_ws - true_ws) ** 2)))

    pred_wd = wind_direction_from_uv(pred_u, pred_v)
    true_wd = wind_direction_from_uv(true_u, true_v)
    wd_err = wrap_deg180(pred_wd - true_wd)
    wd_rmse = float(np.sqrt(np.mean(wd_err ** 2)))
    wd_mae = float(np.mean(np.abs(wd_err)))

    step_mag = np.hypot(true_u - pred_u, true_v - pred_v)

    return {
        "samples": int(len(true_u)),
        "U_RMSE_mps": u_rmse,
        "V_RMSE_mps": v_rmse,
        "wind_vector_RMSE_mps": vector_rmse,
        "WS_RMSE_mps": ws_rmse,
        "WD_RMSE_deg": wd_rmse,
        "WD_MAE_deg": wd_mae,
        "mean_step_vector_mps": float(np.mean(step_mag)),
        "median_step_vector_mps": float(np.median(step_mag)),
        "target_U_std_mps": float(np.std(true_u, ddof=0)),
        "target_V_std_mps": float(np.std(true_v, ddof=0)),
        "target_WS_std_mps": float(np.std(true_ws, ddof=0)),
    }


# =============================================================================
# Exact frozen 12B saved test
# =============================================================================

def load_frozen_track_j(benchmark_dir: Path):
    path = (
        benchmark_dir
        / "track_J_joint_compatible"
        / "Tropical_Atlantic_TEST.npz"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as z:
        X = np.asarray(z["X_raw"], dtype=np.float64)
        y = np.asarray(z["y_joint_raw"], dtype=np.float64)
        context_end = np.asarray(
            z["context_end_time_ns"],
            dtype=np.int64,
        ).reshape(-1)
        target_time = np.asarray(
            z["target_time_ns"],
            dtype=np.int64,
        ).reshape(-1)

        feature_names = [
            str(x)
            for x in np.asarray(z["feature_names"]).tolist()
        ]

    if X.ndim != 3:
        raise RuntimeError(f"Unexpected X shape: {X.shape}")

    if y.ndim != 3 or y.shape[1] != 1:
        raise RuntimeError(f"Unexpected y shape: {y.shape}")

    idx_u = feature_names.index("U")
    idx_v = feature_names.index("V")

    pred_u = X[:, -1, idx_u]
    pred_v = X[:, -1, idx_v]
    true_u = y[:, 0, 0]
    true_v = y[:, 0, 1]

    metrics = compute_metrics(
        pred_u,
        pred_v,
        true_u,
        true_v,
    )

    return {
        "path": path,
        "X": X,
        "y": y,
        "context_end_time_ns": context_end,
        "target_time_ns": target_time,
        "feature_names": feature_names,
        "metrics": metrics,
    }


# =============================================================================
# Raw block construction
# =============================================================================

def build_raw_uv_block_table(
    raw: pd.DataFrame,
    common_base_names,
):
    """
    Build a UTC-aligned raw block table.

    Each accepted row contains the 10 raw U values and 10 raw V values.
    We separately record:
      - uv_complete: exact 10 timestamps + all U/V finite
      - common_complete: exact 10 timestamps + all COMMON_BASE finite

    This table does NOT aggregate the values yet.
    """
    raw = raw.sort_values("time_ns").reset_index(drop=True)

    time_ns = raw["time_ns"].to_numpy(dtype=np.int64)
    block_start = (time_ns // TEN_MIN_NS) * TEN_MIN_NS

    unique, first, counts = np.unique(
        block_start,
        return_index=True,
        return_counts=True,
    )

    rows = []

    for b, s, c in zip(unique, first, counts):
        if int(c) != 10:
            continue

        s = int(s)
        e = s + 10

        actual = time_ns[s:e]
        expected = (
            int(b)
            + np.arange(10, dtype=np.int64) * ONE_MIN_NS
        )

        if not np.array_equal(actual, expected):
            continue

        block = raw.iloc[s:e]

        U = block["U"].to_numpy(dtype=np.float64)
        V = block["V"].to_numpy(dtype=np.float64)

        uv_complete = bool(
            np.isfinite(U).all()
            and np.isfinite(V).all()
        )

        common = block[list(common_base_names)].to_numpy(
            dtype=np.float64
        )
        common_complete = bool(np.isfinite(common).all())

        row = {
            "block_start_ns": int(b),
            "uv_complete": uv_complete,
            "common_complete": common_complete,
        }

        for k in range(10):
            row[f"U{k}"] = float(U[k])
            row[f"V{k}"] = float(V[k])

        row["U_mean"] = float(np.mean(U)) if uv_complete else np.nan
        row["V_mean"] = float(np.mean(V)) if uv_complete else np.nan

        rows.append(row)

    blocks = pd.DataFrame(rows)

    if blocks.empty:
        raise RuntimeError("No exact 10-minute raw blocks were reconstructed.")

    blocks = (
        blocks
        .sort_values("block_start_ns")
        .reset_index(drop=True)
    )

    return blocks


def representation_columns(name: str):
    if name == "mean":
        return "U_mean", "V_mean"

    if name.startswith("phase_"):
        phase = int(name.split("_")[1])

        if phase < 0 or phase > 9:
            raise ValueError(name)

        return f"U{phase}", f"V{phase}"

    raise ValueError(name)


# =============================================================================
# Evaluation on exact frozen timestamps
# =============================================================================

def eval_representation_on_frozen_samples(
    blocks: pd.DataFrame,
    context_times: np.ndarray,
    target_times: np.ndarray,
    representation: str,
):
    ub, vb = representation_columns(representation)

    lookup = blocks.set_index("block_start_ns")

    rows = []

    missing = 0
    nonfinite = 0

    for c, t in zip(context_times, target_times):
        if c not in lookup.index or t not in lookup.index:
            missing += 1
            continue

        rc = lookup.loc[int(c)]
        rt = lookup.loc[int(t)]

        vals = np.asarray(
            [rc[ub], rc[vb], rt[ub], rt[vb]],
            dtype=np.float64,
        )

        if not np.isfinite(vals).all():
            nonfinite += 1
            continue

        rows.append(vals)

    if not rows:
        raise RuntimeError(
            f"No usable frozen samples for representation={representation}"
        )

    arr = np.asarray(rows, dtype=np.float64)

    metrics = compute_metrics(
        arr[:, 0],
        arr[:, 1],
        arr[:, 2],
        arr[:, 3],
    )

    metrics.update({
        "representation": representation,
        "requested_frozen_samples": int(len(context_times)),
        "missing_blocks": int(missing),
        "nonfinite_representation_samples": int(nonfinite),
    })

    return metrics


# =============================================================================
# Sample-set sensitivity
# =============================================================================

def build_contiguous_pairs(
    blocks: pd.DataFrame,
    eligibility_col: str,
):
    """
    Diagnostic sample builder.

    Require 7 contiguous accepted blocks:
        6 context + 1 target

    Unlike 12B, this does NOT require a previous block for dT/dP/dRH.
    That difference is intentional and is reported explicitly.
    """
    good = blocks[blocks[eligibility_col].astype(bool)].copy()

    starts = good["block_start_ns"].to_numpy(dtype=np.int64)

    pairs = []

    for c in range(LOOKBACK_STEPS - 1, len(good) - 1):
        a = c - LOOKBACK_STEPS + 1
        t = c + 1

        expected = (
            starts[a]
            + np.arange(
                LOOKBACK_STEPS + 1,
                dtype=np.int64,
            ) * TEN_MIN_NS
        )

        actual = starts[a:t + 1]

        if not np.array_equal(actual, expected):
            continue

        pairs.append(
            (int(starts[c]), int(starts[t]))
        )

    return pairs


def eval_representation_on_pairs(
    blocks: pd.DataFrame,
    pairs,
    representation: str,
):
    ub, vb = representation_columns(representation)
    lookup = blocks.set_index("block_start_ns")

    arr = []

    for c, t in pairs:
        rc = lookup.loc[c]
        rt = lookup.loc[t]

        vals = np.asarray(
            [rc[ub], rc[vb], rt[ub], rt[vb]],
            dtype=np.float64,
        )

        if np.isfinite(vals).all():
            arr.append(vals)

    if not arr:
        raise RuntimeError(
            f"No usable pairs for representation={representation}"
        )

    arr = np.asarray(arr, dtype=np.float64)

    return compute_metrics(
        arr[:, 0],
        arr[:, 1],
        arr[:, 2],
        arr[:, 3],
    )


# =============================================================================
# Raw-mean reconstruction audit
# =============================================================================

def compare_saved_mean_to_raw_mean(
    frozen,
    blocks,
):
    lookup = blocks.set_index("block_start_ns")

    X = frozen["X"]
    y = frozen["y"]
    feature_names = frozen["feature_names"]

    idx_u = feature_names.index("U")
    idx_v = feature_names.index("V")

    saved_context_u = X[:, -1, idx_u]
    saved_context_v = X[:, -1, idx_v]
    saved_target_u = y[:, 0, 0]
    saved_target_v = y[:, 0, 1]

    raw_context_u = []
    raw_context_v = []
    raw_target_u = []
    raw_target_v = []
    kept = []

    for i, (c, t) in enumerate(
        zip(
            frozen["context_end_time_ns"],
            frozen["target_time_ns"],
        )
    ):
        if c not in lookup.index or t not in lookup.index:
            continue

        rc = lookup.loc[int(c)]
        rt = lookup.loc[int(t)]

        vals = [
            rc["U_mean"],
            rc["V_mean"],
            rt["U_mean"],
            rt["V_mean"],
        ]

        if not np.isfinite(vals).all():
            continue

        kept.append(i)
        raw_context_u.append(vals[0])
        raw_context_v.append(vals[1])
        raw_target_u.append(vals[2])
        raw_target_v.append(vals[3])

    kept = np.asarray(kept, dtype=np.int64)

    raw_context_u = np.asarray(raw_context_u)
    raw_context_v = np.asarray(raw_context_v)
    raw_target_u = np.asarray(raw_target_u)
    raw_target_v = np.asarray(raw_target_v)

    audit = {
        "frozen_samples": int(len(saved_context_u)),
        "raw_reconstructed_samples": int(len(kept)),
        "max_abs_context_U_diff_mps": float(
            np.max(np.abs(saved_context_u[kept] - raw_context_u))
        ),
        "max_abs_context_V_diff_mps": float(
            np.max(np.abs(saved_context_v[kept] - raw_context_v))
        ),
        "max_abs_target_U_diff_mps": float(
            np.max(np.abs(saved_target_u[kept] - raw_target_u))
        ),
        "max_abs_target_V_diff_mps": float(
            np.max(np.abs(saved_target_v[kept] - raw_target_v))
        ),
    }

    return audit


# =============================================================================
# Plot
# =============================================================================

def plot_phase_sensitivity(
    df: pd.DataFrame,
    output_dir: Path,
):
    phase = df[
        df["representation"].str.startswith("phase_")
    ].copy()

    phase["phase"] = (
        phase["representation"]
        .str.replace("phase_", "", regex=False)
        .astype(int)
    )

    phase = phase.sort_values("phase")

    mean_row = df[
        df["representation"].eq("mean")
    ].iloc[0]

    fig, ax = plt.subplots(figsize=(5.8, 3.6))

    ax.plot(
        phase["phase"],
        phase["U_RMSE_mps"],
        marker="o",
        label="$U$ RMSE",
    )
    ax.plot(
        phase["phase"],
        phase["V_RMSE_mps"],
        marker="s",
        label="$V$ RMSE",
    )
    ax.plot(
        phase["phase"],
        phase["WS_RMSE_mps"],
        marker="^",
        label="WS RMSE",
    )

    # Horizontal references = 10-min arithmetic mean.
    ax.axhline(
        float(mean_row["U_RMSE_mps"]),
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
    )
    ax.axhline(
        float(mean_row["V_RMSE_mps"]),
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
    )
    ax.axhline(
        float(mean_row["WS_RMSE_mps"]),
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
    )

    ax.set_xlabel(
        "Minute phase within UTC-aligned 10-min block"
    )
    ax.set_ylabel(r"Persistence RMSE (m s$^{-1}$)")
    ax.set_xticks(range(10))
    ax.legend(loc="best")
    finish_axis(ax)

    fig.tight_layout()
    save_figure(
        fig,
        output_dir / "Fig12F1_same_samples_phase_sensitivity",
    )


# =============================================================================
# Report
# =============================================================================

def format_metric_row(row):
    return (
        f"N={int(row['samples']):5d} | "
        f"U={row['U_RMSE_mps']:.4f} | "
        f"V={row['V_RMSE_mps']:.4f} | "
        f"vector={row['wind_vector_RMSE_mps']:.4f} | "
        f"WS={row['WS_RMSE_mps']:.4f} | "
        f"WD={row['WD_RMSE_deg']:.2f} deg"
    )


def write_report(
    output_dir: Path,
    exact_df,
    same_df,
    filter_df,
    reconstruction_df,
):
    path = output_dir / "audit_report.txt"

    mean_row = same_df[
        same_df["representation"].eq("mean")
    ].iloc[0]

    phases = same_df[
        same_df["representation"].str.startswith("phase_")
    ].copy()

    phase_u_min = phases.loc[phases["U_RMSE_mps"].idxmin()]
    phase_u_max = phases.loc[phases["U_RMSE_mps"].idxmax()]
    phase_v_min = phases.loc[phases["V_RMSE_mps"].idxmin()]
    phase_v_max = phases.loc[phases["V_RMSE_mps"].idxmax()]
    phase_ws_min = phases.loc[phases["WS_RMSE_mps"].idxmin()]
    phase_ws_max = phases.loc[phases["WS_RMSE_mps"].idxmax()]

    with path.open("w", encoding="utf-8") as f:
        f.write("12F Tropical Atlantic 10-min Resampling Forensic Audit\n")
        f.write("=" * 110 + "\n\n")

        f.write("PURPOSE\n")
        f.write("-" * 110 + "\n")
        f.write(
            "Audit whether the low reconstructed Persistence errors are "
            "mainly caused by the 10-min aggregation operator and/or "
            "sample filtering. No model training is performed.\n\n"
        )

        f.write("1. EXACT SAVED 12B PERSISTENCE REPRODUCTION\n")
        f.write("-" * 110 + "\n")
        f.write(exact_df.to_string(index=False))
        f.write("\n\n")

        f.write("Published BP-STGNN Tropical Atlantic reference:\n")
        f.write(
            f"U={PUBLISHED_BPSTGNN['U_RMSE_mps']:.5f}, "
            f"V={PUBLISHED_BPSTGNN['V_RMSE_mps']:.5f}, "
            f"WS={PUBLISHED_BPSTGNN['WS_RMSE_mps']:.5f}, "
            f"WD={PUBLISHED_BPSTGNN['WD_RMSE_deg']:.3f} deg\n\n"
        )

        f.write("2. RAW-MEAN RECONSTRUCTION AUDIT\n")
        f.write("-" * 110 + "\n")
        f.write(reconstruction_df.to_string(index=False))
        f.write("\n\n")

        f.write("3. SAME FROZEN SAMPLES: AGGREGATION SENSITIVITY\n")
        f.write("-" * 110 + "\n")
        f.write(same_df.to_string(index=False))
        f.write("\n\n")

        f.write("Arithmetic-mean persistence:\n")
        f.write("  " + format_metric_row(mean_row) + "\n\n")

        f.write("Phase sensitivity range on EXACT same sample timestamps:\n")
        f.write(
            f"  U  min={phase_u_min['U_RMSE_mps']:.4f} "
            f"({phase_u_min['representation']}), "
            f"max={phase_u_max['U_RMSE_mps']:.4f} "
            f"({phase_u_max['representation']})\n"
        )
        f.write(
            f"  V  min={phase_v_min['V_RMSE_mps']:.4f} "
            f"({phase_v_min['representation']}), "
            f"max={phase_v_max['V_RMSE_mps']:.4f} "
            f"({phase_v_max['representation']})\n"
        )
        f.write(
            f"  WS min={phase_ws_min['WS_RMSE_mps']:.4f} "
            f"({phase_ws_min['representation']}), "
            f"max={phase_ws_max['WS_RMSE_mps']:.4f} "
            f"({phase_ws_max['representation']})\n\n"
        )

        f.write("4. SAMPLE-FILTERING SENSITIVITY\n")
        f.write("-" * 110 + "\n")
        f.write(filter_df.to_string(index=False))
        f.write("\n\n")

        f.write("INTERPRETATION GUIDE\n")
        f.write("-" * 110 + "\n")
        f.write(
            "A. If single-minute phase RMSE values are much larger than the "
            "10-min arithmetic-mean values on the same frozen timestamps, "
            "the aggregation operator materially changes task difficulty.\n"
        )
        f.write(
            "B. If UV-only/common-complete sample sets materially change "
            "RMSE relative to the frozen 12B samples, sample filtering also "
            "contributes.\n"
        )
        f.write(
            "C. If neither effect brings the reconstructed Persistence "
            "errors close to the published BP-STGNN scale, do not infer a "
            "model bug. Remaining likely causes are other under-specified "
            "protocol differences such as spline-filled gaps, exact "
            "resampling semantics, test-interval/sample selection, history "
            "construction, or metric aggregation.\n"
        )

    log(f"[SAVED] {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )

    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    project_root = args.project_root

    benchmark_dir = (
        args.benchmark_dir
        if args.benchmark_dir is not None
        else project_root
        / "data"
        / "forecasting"
        / "12B_Nie_same_mission_benchmark_v0_1"
    )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else project_root
        / "data"
        / "forecasting"
        / "12F_TropicalAtlantic_10min_resampling_audit_v0_1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stage12b = load_stage12b_module(project_root)

    raw_dir = (
        args.raw_dir
        if args.raw_dir is not None
        else project_root / "data" / "raw" / "RawData"
    )

    apply_style(project_root)

    log("=" * 118)
    log("12F — TROPICAL ATLANTIC 10-MIN RESAMPLING FORENSIC AUDIT")
    log("=" * 118)
    log(f"script version : {SCRIPT_VERSION}")
    log(f"project root   : {project_root}")
    log(f"benchmark dir  : {benchmark_dir}")
    log(f"raw dir        : {raw_dir}")
    log(f"output dir     : {output_dir}")
    log("[POLICY] Reporting/audit only. No training, HPO, or checkpoint changes.")
    log("")

    # -----------------------------------------------------------------
    # 1. Exact frozen saved Persistence
    # -----------------------------------------------------------------
    frozen = load_frozen_track_j(benchmark_dir)

    exact_row = {
        "scenario": "12B_saved_exact_Persistence",
        "representation": "saved_10min_mean",
        **frozen["metrics"],
    }
    exact_df = pd.DataFrame([exact_row])

    exact_df.to_csv(
        output_dir / "exact_saved_persistence_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("[1/5] Exact frozen 12B Persistence")
    log("      " + format_metric_row(exact_row))
    log("")

    # -----------------------------------------------------------------
    # 2. Load raw Tropical Atlantic using exact 12B discovery/mapping.
    # -----------------------------------------------------------------
    xr = stage12b.import_xarray()
    grouped = stage12b.discover_mission_files(raw_dir)
    tropical_paths = grouped["Tropical Atlantic"]

    log(
        f"[2/5] Loading Tropical Atlantic raw data using Stage-12B logic "
        f"({len(tropical_paths)} file(s))"
    )

    raw, raw_audit, _ = stage12b.load_mission_raw(
        xr,
        "Tropical Atlantic",
        tropical_paths,
        args.progress_every,
    )

    log(
        f"      raw rows after dedup = "
        f"{raw_audit['rows_after_dedup']:,}"
    )

    blocks = build_raw_uv_block_table(
        raw,
        stage12b.COMMON_BASE,
    )

    log(
        f"      exact 10-min timestamp blocks = {len(blocks):,}"
    )
    log(
        f"      UV-complete blocks            = "
        f"{int(blocks['uv_complete'].sum()):,}"
    )
    log(
        f"      COMMON_BASE-complete blocks   = "
        f"{int(blocks['common_complete'].sum()):,}"
    )
    log("")

    # -----------------------------------------------------------------
    # 3. Reconstruct frozen mean from raw and audit exact equality.
    # -----------------------------------------------------------------
    reconstruction = compare_saved_mean_to_raw_mean(
        frozen,
        blocks,
    )

    reconstruction_df = pd.DataFrame([reconstruction])
    reconstruction_df.to_csv(
        output_dir / "frozen_sample_raw_reconstruction_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("[3/5] Raw arithmetic-mean reconstruction audit")
    for k, v in reconstruction.items():
        log(f"      {k}: {v}")
    log("")

    # -----------------------------------------------------------------
    # 4. Same frozen timestamps, different 10-min representations.
    # -----------------------------------------------------------------
    representations = ["mean"] + [
        f"phase_{i}"
        for i in range(10)
    ]

    same_rows = []

    for rep in representations:
        m = eval_representation_on_frozen_samples(
            blocks,
            frozen["context_end_time_ns"],
            frozen["target_time_ns"],
            rep,
        )

        same_rows.append({
            "scenario": "same_frozen_12B_samples",
            **m,
        })

    same_df = pd.DataFrame(same_rows)

    same_df.to_csv(
        output_dir / "same_frozen_samples_resampling_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    phase_df = same_df[
        same_df["representation"].str.startswith("phase_")
    ].copy()

    phase_df.to_csv(
        output_dir / "phase_sensitivity_same_frozen_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("[4/5] Same frozen timestamps — aggregation sensitivity")
    for _, row in same_df.iterrows():
        log(
            f"      {row['representation']:<8s} | "
            + format_metric_row(row)
        )
    log("")

    # -----------------------------------------------------------------
    # 5. Sample filtering sensitivity.
    # -----------------------------------------------------------------
    filtering_rows = []

    for eligibility_col, label in [
        ("uv_complete", "strict_UV_complete_7block"),
        ("common_complete", "strict_COMMON_BASE_complete_7block"),
    ]:
        pairs = build_contiguous_pairs(
            blocks,
            eligibility_col,
        )

        for rep in ["mean", "phase_0", "phase_9"]:
            m = eval_representation_on_pairs(
                blocks,
                pairs,
                rep,
            )

            filtering_rows.append({
                "sample_policy": label,
                "representation": rep,
                "pair_count_before_rep_finite_check": int(len(pairs)),
                **m,
            })

    # Include exact frozen row for direct reference.
    filtering_rows.insert(
        0,
        {
            "sample_policy": "frozen_12B_exact",
            "representation": "saved_10min_mean",
            "pair_count_before_rep_finite_check": int(
                frozen["metrics"]["samples"]
            ),
            **frozen["metrics"],
        },
    )

    filter_df = pd.DataFrame(filtering_rows)

    filter_df.to_csv(
        output_dir / "sample_filtering_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("[5/5] Sample-filtering sensitivity")
    for _, row in filter_df.iterrows():
        log(
            f"      {row['sample_policy']:<36s} | "
            f"{row['representation']:<16s} | "
            + format_metric_row(row)
        )
    log("")

    # Plot.
    plot_phase_sensitivity(
        same_df,
        output_dir,
    )

    # Report.
    write_report(
        output_dir,
        exact_df,
        same_df,
        filter_df,
        reconstruction_df,
    )

    # Machine-readable summary.
    summary = {
        "stage": "12F",
        "script_version": SCRIPT_VERSION,
        "scientific_status": "AUDIT_ONLY_NO_TRAINING",
        "frozen_test_samples": int(frozen["metrics"]["samples"]),
        "published_BPSTGNN_reference": PUBLISHED_BPSTGNN,
        "raw_rows_after_dedup": int(raw_audit["rows_after_dedup"]),
        "exact_10min_timestamp_blocks": int(len(blocks)),
        "uv_complete_blocks": int(blocks["uv_complete"].sum()),
        "common_complete_blocks": int(blocks["common_complete"].sum()),
        "raw_reconstruction_audit": reconstruction,
    }

    with (
        output_dir / "audit_manifest.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    log("=" * 118)
    log("12F AUDIT COMPLETE")
    log("=" * 118)
    log(f"Outputs: {output_dir}")
    log("")
    log("Please send back:")
    log("  1) the terminal output from [1/5], [4/5], and [5/5];")
    log("  2) audit_report.txt.")
    log("")
    log(
        "The key diagnostic is whether phase_0...phase_9 Persistence RMSE "
        "moves materially toward the published BP-STGNN scale while the "
        "sample timestamps remain exactly frozen."
    )


if __name__ == "__main__":
    main()
