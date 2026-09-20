# -*- coding: utf-8 -*-
r"""
10B_plot_SD1090_test_sliding_8h_windows.py

Purpose
-------
After 10A_run_SD1090_internal_evaluation_v3.py has finished, this script:

1) reports the exact calendar coverage of the SD1090 internal-evaluation
   (test) split for every horizon;
2) reports the exact 8-h interval previously selected by 10A-v3;
3) divides the whole common test period into consecutive 8-h windows
   (default stride = 8 h), appending a final tail-aligned window if needed;
4) optionally adds the original max-observed-variability 8-h window;
5) produces one SCI-style 2x3 figure per window:
       top row    : U_app at 1, 5, 10 min
       bottom row : V_app at 1, 5, 10 min
   with:
       Observed
       Physics-Compact
       DLinear
       TimeMixer
       iTransformer
       PatchTST
6) saves horizon-wise metrics for every plotted window.

The plot style follows the previous manuscript figures:
- sci_plot_style.py if available
- Times New Roman / serif
- same six colors
- inward ticks
- subtle grid
- 600 dpi PNG + vector PDF

IMPORTANT
---------
This script performs NO training and NO model selection. It only reads the
frozen TEST predictions created by 10A-v3.

Recommended PowerShell
----------------------
# Non-overlapping 8-h windows covering the whole test period:
python "D:\project\WindPredict_SaildroneData\src\10B_plot_SD1090_test_sliding_8h_windows.py" `
  --root "D:\project\WindPredict_SaildroneData"

# 50%-overlapping 8-h windows:
python "D:\project\WindPredict_SaildroneData\src\10B_plot_SD1090_test_sliding_8h_windows.py" `
  --root "D:\project\WindPredict_SaildroneData" `
  --stride-hours 4

# Dense 1-h sliding:
python "D:\project\WindPredict_SaildroneData\src\10B_plot_SD1090_test_sliding_8h_windows.py" `
  --root "D:\project\WindPredict_SaildroneData" `
  --stride-hours 1

Outputs
-------
<eval_dir>/figures_sliding_8h/
    TEST_TIME_COVERAGE.txt
    sliding_8h_window_index.csv
    sliding_8h_window_metrics.csv
    window_001_...png/.pdf
    window_002_...png/.pdf
    ...
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


HORIZONS = np.asarray([1, 2, 3, 5, 10], dtype=int)
PLOT_HORIZONS = [1, 5, 10]
SEEDS = [500043, 501052, 502061, 503070, 504079]

MODERN_MODELS = [
    "DLinear",
    "TimeMixer",
    "iTransformer",
    "PatchTST",
]

MODEL_ORDER = [
    "Physics-Compact",
    "DLinear",
    "TimeMixer",
    "iTransformer",
    "PatchTST",
]

COLORS = {
    "Observed": "#000000",
    "Physics-Compact": "#1f77b4",
    "DLinear": "#ff7f0e",
    "TimeMixer": "#2ca02c",
    "iTransformer": "#d62728",
    "PatchTST": "#9467bd",
}


# =============================================================================
# Utilities
# =============================================================================

def log(msg=""):
    print(msg, flush=True)


def load_sci_style(root: Path) -> None:
    style_path = root / "src" / "sci_plot_style.py"

    if style_path.exists():
        spec = importlib.util.spec_from_file_location(
            "sci_plot_style_10b",
            str(style_path),
        )
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)

            if hasattr(mod, "apply_sci_style"):
                try:
                    mod.apply_sci_style(base_font_size=8.5)
                except TypeError:
                    try:
                        mod.apply_sci_style(font_size=8.5)
                    except TypeError:
                        mod.apply_sci_style()
                return

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman No9 L",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "axes.unicode_minus": False,
        }
    )


def clean_axis(ax):
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.grid(True, which="major", linewidth=0.40, alpha=0.18)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def panel_label(ax, label):
    ax.text(
        0.012,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
    )


def save_both(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")

    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def apparent_from_joint(y):
    y = np.asarray(y, dtype=np.float64)
    return y[..., 0:2] - y[..., 2:4]


def vector_rmse(obs, pred):
    e = np.asarray(pred, float) - np.asarray(obs, float)
    return float(np.sqrt(np.mean(np.sum(e * e, axis=-1))))


def scalar_rmse(obs, pred):
    e = np.asarray(pred, float) - np.asarray(obs, float)
    return float(np.sqrt(np.mean(e * e)))


def ns_to_utc(ns: int) -> pd.Timestamp:
    return pd.to_datetime(int(ns), unit="ns", utc=True)


def utc_compact(ns: int) -> str:
    return ns_to_utc(ns).strftime("%Y%m%d_%H%M")


def utc_text(ns: int) -> str:
    return ns_to_utc(ns).strftime("%Y-%m-%d %H:%M UTC")


def duration_text(delta_ns: int) -> str:
    td = pd.to_timedelta(int(delta_ns), unit="ns")
    total_min = int(round(td.total_seconds() / 60.0))
    days, rem = divmod(total_min, 24 * 60)
    hours, mins = divmod(rem, 60)
    return f"{days} d {hours} h {mins} min ({total_min} min)"


# =============================================================================
# Data/prediction loading
# =============================================================================

def load_test(dataset_dir: Path):
    path = dataset_dir / "test.npz"
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as z:
        if "y_raw" not in z.files:
            raise KeyError(f"{path}: missing y_raw")
        if "target_time_ns" not in z.files:
            raise KeyError(
                f"{path}: missing target_time_ns; exact calendar-window "
                "alignment is required for this script."
            )

        y = np.asarray(z["y_raw"], dtype=np.float32)
        tt = np.asarray(z["target_time_ns"], dtype=np.int64)

    if y.ndim != 3 or y.shape[1:] != (5, 6):
        raise RuntimeError(f"Unexpected y_raw shape: {y.shape}")
    if tt.shape != (len(y), 5):
        raise RuntimeError(
            f"Unexpected target_time_ns shape: {tt.shape}; "
            f"expected {(len(y), 5)}"
        )

    return y, tt


def load_seed_prediction(path: Path, pred_keys):
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as z:
        for key in pred_keys:
            if key in z.files:
                arr = np.asarray(z[key], dtype=np.float32)
                break
        else:
            raise KeyError(
                f"{path}: none of prediction keys {pred_keys}; "
                f"available={list(z.files)}"
            )

    if arr.ndim != 3 or arr.shape[1:] != (5, 6):
        raise RuntimeError(f"{path}: bad prediction shape {arr.shape}")

    return arr


def load_predictions(eval_dir: Path, truth_shape):
    pred_dir = eval_dir / "predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(
            f"Prediction directory not found: {pred_dir}\n"
            "Run 10A_run_SD1090_internal_evaluation_v3.py first."
        )

    out: Dict[str, np.ndarray] = {}

    # Physics-Compact: v3 writes one joint TEST NPZ per seed.
    pc = []
    for seed in SEEDS:
        p = pred_dir / f"15E_VH_seed_{seed}_test_predictions.npz"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing Physics-Compact TEST prediction: {p}\n"
                "Run 10A-v3 to completion first."
            )

        with np.load(p, allow_pickle=False) as z:
            required = ["frozen_truewind", "joint_vessel_hdg"]
            missing = [k for k in required if k not in z.files]
            if missing:
                raise KeyError(f"{p}: missing {missing}")

            wind = np.asarray(z["frozen_truewind"], dtype=np.float32)
            motion = np.asarray(z["joint_vessel_hdg"], dtype=np.float32)

        joint = np.empty(truth_shape, dtype=np.float32)
        joint[..., 0:2] = wind
        joint[..., 2:6] = motion
        pc.append(joint)

    out["Physics-Compact"] = np.mean(
        np.stack(pc, axis=0),
        axis=0,
    ).astype(np.float32)

    # Modern baselines: v3 writes test_<model>_seed<seed>.npz.
    for model in MODERN_MODELS:
        seed_preds = []
        safe = model.replace("-", "_")

        for seed in SEEDS:
            p = pred_dir / f"test_{safe}_seed{seed}.npz"
            pred = load_seed_prediction(
                p,
                ["pred_raw", "y_pred", "prediction_raw", "pred"],
            )
            seed_preds.append(pred)

        out[model] = np.mean(
            np.stack(seed_preds, axis=0),
            axis=0,
        ).astype(np.float32)

    for name, arr in out.items():
        if arr.shape != truth_shape:
            raise RuntimeError(
                f"{name}: prediction shape {arr.shape} != truth {truth_shape}"
            )

    return out


# =============================================================================
# Time coverage + prior selected interval
# =============================================================================

def report_test_coverage(
    truth,
    tt,
    selected_trace_path: Path,
    out_dir: Path,
):
    lines: List[str] = []

    lines.append("=" * 100)
    lines.append("SD1090 INTERNAL-EVALUATION / TEST TIME COVERAGE")
    lines.append("=" * 100)
    lines.append(f"Number of forecasting windows: {len(truth):,}")
    lines.append(
        "At each fixed horizon, adjacent target timestamps are expected "
        "to be 1 min apart."
    )
    lines.append("")

    for j, h in enumerate(HORIZONS):
        first_ns = int(tt[0, j])
        last_ns = int(tt[-1, j])
        elapsed = last_ns - first_ns

        diffs = np.diff(tt[:, j])
        median_step_min = (
            np.median(diffs.astype(np.float64))
            / 60e9
        )

        lines.append(
            f"h={h:2d} min | first={utc_text(first_ns)} | "
            f"last={utc_text(last_ns)} | "
            f"elapsed={duration_text(elapsed)} | "
            f"median step={median_step_min:.3f} min"
        )

    lines.append("")

    common_first = max(
        int(tt[0, np.where(HORIZONS == h)[0][0]])
        for h in PLOT_HORIZONS
    )
    common_last = min(
        int(tt[-1, np.where(HORIZONS == h)[0][0]])
        for h in PLOT_HORIZONS
    )

    lines.append(
        "Common target-time interval shared by h={1,5,10}:"
    )
    lines.append(
        f"  {utc_text(common_first)} -> {utc_text(common_last)}"
    )
    lines.append(
        f"  elapsed = {duration_text(common_last - common_first)}"
    )

    prior = None

    if selected_trace_path.exists():
        with np.load(selected_trace_path, allow_pickle=False) as z:
            start_ns = int(z["start_time_ns"])
            end_ns = int(z["end_time_ns"])
            samples = int(z["window_samples"])
            variability = float(z["observed_variability_score"])

        prior = {
            "start_time_ns": start_ns,
            "end_time_ns": end_ns,
            "window_samples": samples,
            "variability": variability,
        }

        lines.append("")
        lines.append(
            "8-h interval previously selected by 10A-v3 "
            "(maximum observed apparent-wind variability):"
        )
        lines.append(
            f"  {utc_text(start_ns)} -> {utc_text(end_ns)}"
        )
        lines.append(
            f"  samples={samples}, "
            f"timestamp elapsed={duration_text(end_ns-start_ns)}, "
            f"variability score={variability:.6f}"
        )
    else:
        lines.append("")
        lines.append(
            f"Previous selected interval file not found: "
            f"{selected_trace_path}"
        )

    text = "\n".join(lines)
    log(text)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "TEST_TIME_COVERAGE.txt").write_text(
        text + "\n",
        encoding="utf-8",
    )

    return common_first, common_last, prior


# =============================================================================
# Window generation/alignment
# =============================================================================

def exact_index(times, ns):
    i = int(np.searchsorted(times, int(ns), side="left"))
    if i >= len(times) or int(times[i]) != int(ns):
        return None
    return i


def aligned_indices_for_window(tt, start_ns, window_samples):
    idx = {}

    for h in PLOT_HORIZONS:
        j = int(np.where(HORIZONS == h)[0][0])
        i = exact_index(tt[:, j], start_ns)

        if i is None:
            return None

        if i + window_samples > len(tt):
            return None

        idx[h] = (i, i + window_samples)

    return idx


def generate_windows(
    tt,
    common_first,
    common_last,
    window_samples,
    stride_samples,
    prior,
    cover_tail=True,
    include_prior=True,
):
    """
    Start from the first common timestamp, then step by stride_samples on
    the h=10 target-time grid. Add the last possible full window so the
    tail is represented.
    """
    j10 = int(np.where(HORIZONS == 10)[0][0])
    t10 = tt[:, j10]

    first_idx = exact_index(t10, common_first)
    last_idx = exact_index(t10, common_last)

    if first_idx is None or last_idx is None:
        raise RuntimeError(
            "Could not locate common interval on h=10 timestamp grid."
        )

    max_start_idx = last_idx - window_samples + 1
    if max_start_idx < first_idx:
        raise RuntimeError(
            "Common h={1,5,10} interval is shorter than one 8-h window."
        )

    start_indices = list(
        range(
            first_idx,
            max_start_idx + 1,
            stride_samples,
        )
    )

    if cover_tail and start_indices[-1] != max_start_idx:
        start_indices.append(max_start_idx)

    candidate_starts = [
        int(t10[i])
        for i in start_indices
    ]

    if include_prior and prior is not None:
        ps = int(prior["start_time_ns"])
        if ps not in candidate_starts:
            candidate_starts.append(ps)

    # Sort chronologically and keep unique.
    candidate_starts = sorted(set(candidate_starts))

    rows = []

    for start_ns in candidate_starts:
        aligned = aligned_indices_for_window(
            tt,
            start_ns,
            window_samples,
        )
        if aligned is None:
            continue

        # Use h=10 time for end timestamp because all panels contain the
        # same number of one-minute targets and the same start timestamp.
        i10a, i10b = aligned[10]
        end_ns = int(tt[i10b - 1, j10])

        is_prior = (
            prior is not None
            and int(prior["start_time_ns"]) == start_ns
        )

        rows.append(
            {
                "start_time_ns": start_ns,
                "end_time_ns": end_ns,
                "is_original_selected_window": bool(is_prior),
                "indices": aligned,
            }
        )

    return rows


# =============================================================================
# Metrics and plotting
# =============================================================================

def observed_variability(obs_uv):
    u = np.asarray(obs_uv[:, 0], dtype=np.float64)
    v = np.asarray(obs_uv[:, 1], dtype=np.float64)

    return float(
        np.sqrt(
            np.var(u, ddof=0)
            + np.var(v, ddof=0)
        )
    )


def window_metrics(
    window_id,
    truth,
    pred_mean,
    tt,
    window,
):
    rows = []
    start_ns = int(window["start_time_ns"])
    end_ns = int(window["end_time_ns"])

    for h in PLOT_HORIZONS:
        j = int(np.where(HORIZONS == h)[0][0])
        a, b = window["indices"][h]

        obs_joint = truth[a:b, j, :]
        obs_app = apparent_from_joint(obs_joint)

        for model in MODEL_ORDER:
            pred_joint = pred_mean[model][a:b, j, :]
            pred_app = apparent_from_joint(pred_joint)

            rows.append(
                {
                    "window_id": int(window_id),
                    "start_UTC": utc_text(start_ns),
                    "end_UTC": utc_text(end_ns),
                    "is_original_selected_window": int(
                        window["is_original_selected_window"]
                    ),
                    "horizon_min": int(h),
                    "model": model,
                    "Uapp_RMSE_mps": scalar_rmse(
                        obs_app[:, 0],
                        pred_app[:, 0],
                    ),
                    "Vapp_RMSE_mps": scalar_rmse(
                        obs_app[:, 1],
                        pred_app[:, 1],
                    ),
                    "AppVector_RMSE_mps": vector_rmse(
                        obs_app,
                        pred_app,
                    ),
                    "Observed_App_variability_mps": (
                        observed_variability(obs_app)
                    ),
                }
            )

    return rows


def plot_window_2x3(
    out_dir: Path,
    window_id: int,
    truth,
    pred_mean,
    tt,
    window,
):
    start_ns = int(window["start_time_ns"])
    end_ns = int(window["end_time_ns"])

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.15, 5.10),
        sharex="col",
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.992,
        bottom=0.105,
        top=0.875,
        wspace=0.16,
        hspace=0.13,
    )

    # Collect ranges so U panels share one y range and V panels share one.
    all_u = []
    all_v = []
    panel_data = {}

    for col, h in enumerate(PLOT_HORIZONS):
        j = int(np.where(HORIZONS == h)[0][0])
        a, b = window["indices"][h]

        obs = apparent_from_joint(
            truth[a:b, j, :]
        )
        pred_app = {
            m: apparent_from_joint(
                pred_mean[m][a:b, j, :]
            )
            for m in MODEL_ORDER
        }

        panel_data[h] = (obs, pred_app)

        all_u.append(obs[:, 0])
        all_v.append(obs[:, 1])
        for m in MODEL_ORDER:
            all_u.append(pred_app[m][:, 0])
            all_v.append(pred_app[m][:, 1])

    u_min = min(float(np.nanmin(x)) for x in all_u)
    u_max = max(float(np.nanmax(x)) for x in all_u)
    v_min = min(float(np.nanmin(x)) for x in all_v)
    v_max = max(float(np.nanmax(x)) for x in all_v)

    def padded(lo, hi):
        span = max(hi - lo, 1e-6)
        pad = 0.05 * span
        return lo - pad, hi + pad

    u_ylim = padded(u_min, u_max)
    v_ylim = padded(v_min, v_max)

    labels = [
        "(a)", "(b)", "(c)",
        "(d)", "(e)", "(f)",
    ]

    for col, h in enumerate(PLOT_HORIZONS):
        obs, pred_app = panel_data[h]
        n = len(obs)
        t = np.arange(n, dtype=float) / 60.0

        # U row
        ax = axes[0, col]
        ax.plot(
            t,
            obs[:, 0],
            color=COLORS["Observed"],
            linewidth=1.00,
            label="Observed",
            zorder=6,
        )
        for m in MODEL_ORDER:
            ax.plot(
                t,
                pred_app[m][:, 0],
                color=COLORS[m],
                linewidth=0.72,
                alpha=0.97,
                label=m,
                zorder=4,
            )
        ax.set_ylim(*u_ylim)
        ax.set_title(f"{h}-min horizon", pad=3.0)
        panel_label(ax, labels[col])
        clean_axis(ax)

        # V row
        ax = axes[1, col]
        ax.plot(
            t,
            obs[:, 1],
            color=COLORS["Observed"],
            linewidth=1.00,
            label="Observed",
            zorder=6,
        )
        for m in MODEL_ORDER:
            ax.plot(
                t,
                pred_app[m][:, 1],
                color=COLORS[m],
                linewidth=0.72,
                alpha=0.97,
                label=m,
                zorder=4,
            )
        ax.set_ylim(*v_ylim)
        ax.set_xlabel("Time within 8-h window (h)")
        panel_label(ax, labels[3 + col])
        clean_axis(ax)

    axes[0, 0].set_ylabel(
        r"$U_{\mathrm{app}}$ (m s$^{-1}$)"
    )
    axes[1, 0].set_ylabel(
        r"$V_{\mathrm{app}}$ (m s$^{-1}$)"
    )

    # Avoid repeated y tick labels in the middle/right columns.
    for col in [1, 2]:
        axes[0, col].tick_params(labelleft=False)
        axes[1, col].tick_params(labelleft=False)

    handles, labels_legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=6,
        frameon=False,
        fontsize=6.8,
        columnspacing=0.85,
        handlelength=1.9,
        handletextpad=0.34,
    )

    tag = "SELECTED_" if window["is_original_selected_window"] else ""
    fig.suptitle(
        f"{tag}SD1090 internal evaluation: "
        f"{utc_text(start_ns)} to {utc_text(end_ns)}",
        y=0.915,
        fontsize=8.2,
    )

    name = (
        f"window_{window_id:03d}_"
        f"{utc_compact(start_ns)}_to_{utc_compact(end_ns)}"
    )

    if window["is_original_selected_window"]:
        name += "_ORIGINAL_SELECTED"

    save_both(
        fig,
        out_dir / name,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        type=Path,
        default=Path(
            r"D:\project\WindPredict_SaildroneData"
        ),
    )

    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help=(
            "10A-v3 output directory. By default the script searches "
            "v0_3 first and then v0_1."
        ),
    )

    ap.add_argument(
        "--window-hours",
        type=float,
        default=8.0,
    )

    ap.add_argument(
        "--stride-hours",
        type=float,
        default=8.0,
        help=(
            "Window-start spacing. Default 8 h gives consecutive "
            "non-overlapping 8-h windows. Use 4 for 50%% overlap or "
            "1 for dense hourly sliding."
        ),
    )

    ap.add_argument(
        "--no-cover-tail",
        action="store_true",
        help="Do not append a final tail-aligned full 8-h window.",
    )

    ap.add_argument(
        "--exclude-original-selected",
        action="store_true",
        help=(
            "Do not add the 10A-v3 max-variability interval if its "
            "start is not already on the stride grid."
        ),
    )

    ap.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help=(
            "Optional safety cap. 0 means plot all generated windows."
        ),
    )

    args = ap.parse_args()

    root = args.root.resolve()

    dataset_dir = (
        args.dataset_dir.resolve()
        if args.dataset_dir is not None
        else root
        / "data"
        / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
    )

    if args.eval_dir is not None:
        eval_dir = args.eval_dir.resolve()
    else:
        cand = [
            dataset_dir
            / "10A_SD1090_internal_evaluation_v0_3",
            dataset_dir
            / "10A_SD1090_internal_evaluation_v0_1",
        ]
        existing = [p for p in cand if p.exists()]
        if not existing:
            raise FileNotFoundError(
                "Cannot find 10A internal-evaluation output. "
                "Run 10A-v3 first or pass --eval-dir."
            )
        eval_dir = existing[0]

    out_dir = eval_dir / "figures_sliding_8h"
    out_dir.mkdir(parents=True, exist_ok=True)

    load_sci_style(root)

    truth, tt = load_test(dataset_dir)
    pred_mean = load_predictions(
        eval_dir,
        truth.shape,
    )

    selected_trace = (
        eval_dir
        / "figures"
        / "selected_interval_traceability.npz"
    )

    common_first, common_last, prior = report_test_coverage(
        truth,
        tt,
        selected_trace,
        out_dir,
    )

    window_samples = int(
        round(args.window_hours * 60.0)
    )
    stride_samples = int(
        round(args.stride_hours * 60.0)
    )

    if window_samples < 2:
        raise ValueError("--window-hours is too small")
    if stride_samples < 1:
        raise ValueError("--stride-hours is too small")

    windows = generate_windows(
        tt,
        common_first,
        common_last,
        window_samples,
        stride_samples,
        prior,
        cover_tail=not args.no_cover_tail,
        include_prior=not args.exclude_original_selected,
    )

    if args.max_windows > 0:
        windows = windows[: args.max_windows]

    log("")
    log("=" * 100)
    log("SLIDING 8-H WINDOW PLAN")
    log("=" * 100)
    log(f"window length  : {args.window_hours:g} h = {window_samples} samples")
    log(f"stride         : {args.stride_hours:g} h = {stride_samples} samples")
    log(f"windows to plot: {len(windows)}")
    log(f"output         : {out_dir}")
    log("=" * 100)

    index_rows = []
    metric_rows = []

    # Observed variability is computed from h=10, matching 10A-v3 selection.
    j10 = int(np.where(HORIZONS == 10)[0][0])

    for k, window in enumerate(windows, start=1):
        a10, b10 = window["indices"][10]
        obs10 = apparent_from_joint(
            truth[a10:b10, j10, :]
        )
        variability = observed_variability(obs10)

        start_ns = int(window["start_time_ns"])
        end_ns = int(window["end_time_ns"])

        index_rows.append(
            {
                "window_id": k,
                "start_UTC": utc_text(start_ns),
                "end_UTC": utc_text(end_ns),
                "start_time_ns": start_ns,
                "end_time_ns": end_ns,
                "window_samples": window_samples,
                "stride_hours": float(args.stride_hours),
                "is_original_selected_window": int(
                    window["is_original_selected_window"]
                ),
                "observed_app_variability_h10_mps": variability,
            }
        )

        metric_rows.extend(
            window_metrics(
                k,
                truth,
                pred_mean,
                tt,
                window,
            )
        )

        log(
            f"[{k:03d}/{len(windows):03d}] "
            f"{utc_text(start_ns)} -> {utc_text(end_ns)}"
            + (
                "  [ORIGINAL SELECTED]"
                if window["is_original_selected_window"]
                else ""
            )
        )

        plot_window_2x3(
            out_dir,
            k,
            truth,
            pred_mean,
            tt,
            window,
        )

    index_df = pd.DataFrame(index_rows)
    metrics_df = pd.DataFrame(metric_rows)

    index_df.to_csv(
        out_dir / "sliding_8h_window_index.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics_df.to_csv(
        out_dir / "sliding_8h_window_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Ranking by observed variability for fast visual review.
    rank_df = index_df.sort_values(
        "observed_app_variability_h10_mps",
        ascending=False,
    ).reset_index(drop=True)
    rank_df.insert(
        0,
        "variability_rank",
        np.arange(1, len(rank_df) + 1),
    )

    rank_df.to_csv(
        out_dir / "sliding_8h_windows_ranked_by_observed_variability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log("=" * 100)
    log("DONE")
    log("=" * 100)
    log(f"Figures : {out_dir}")
    log(
        "Index   : "
        f"{out_dir / 'sliding_8h_window_index.csv'}"
    )
    log(
        "Metrics : "
        f"{out_dir / 'sliding_8h_window_metrics.csv'}"
    )
    log(
        "Ranking : "
        f"{out_dir / 'sliding_8h_windows_ranked_by_observed_variability.csv'}"
    )


if __name__ == "__main__":
    main()
