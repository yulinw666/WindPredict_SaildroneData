# -*- coding: utf-8 -*-
r"""
plot_fig03_appUV_modern_models_h1_h5_h10_separate.py

Purpose
-------
Redraw the 6-panel apparent-wind figure as THREE separate 2-panel figures
for the fixed SD1090 internal-evaluation interval:

    2025-01-07 12:14 UTC to 2025-01-07 20:13 UTC

Each figure contains:
    (a) Apparent-wind U_app
    (b) Apparent-wind V_app

for one horizon only:
    1 min, 5 min, or 10 min

Curves
------
Observed
Physics-Compact
DLinear
TimeMixer
iTransformer
PatchTST

All model curves are plotted as five-seed means.

Input assumptions
-----------------
1) SD1090 internal-evaluation dataset:
   <dataset_dir>/test.npz
   containing:
       y_raw
       target_time_ns

2) 10A internal-evaluation outputs:
   <eval_dir>/predictions/
       15E_VH_seed_<seed>_test_predictions.npz
       or
       15E_VH_seed_<seed>_internal_evaluation_predictions.npz

       test_DLinear_seed<seed>.npz
       test_TimeMixer_seed<seed>.npz
       test_iTransformer_seed<seed>.npz
       test_PatchTST_seed<seed>.npz

Outputs
-------
<output_dir>/
    Fig03_appUV_modern_h1.png/.pdf
    Fig03_appUV_modern_h5.png/.pdf
    Fig03_appUV_modern_h10.png/.pdf

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\plot_fig03_appUV_modern_models_h1_h5_h10_separate.py" `
  --root "D:\project\WindPredict_SaildroneData"
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# =============================================================================
# Constants
# =============================================================================

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

def log(msg: str = "") -> None:
    print(msg, flush=True)


def ns_to_utc(ns: int) -> pd.Timestamp:
    return pd.to_datetime(int(ns), unit="ns", utc=True)


def utc_text(ns: int) -> str:
    return ns_to_utc(ns).strftime("%Y-%m-%d %H:%M UTC")


def parse_utc_to_ns(text: str) -> int:
    # Accepts e.g. "2025-01-07 12:14 UTC"
    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.value)


def load_sci_style(root: Path) -> None:
    style_path = root / "src" / "sci_plot_style.py"

    if style_path.exists():
        spec = importlib.util.spec_from_file_location(
            "sci_plot_style_fig03_h1_h5_h10",
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
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
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


def clean_axis(ax) -> None:
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.grid(True, which="major", linewidth=0.40, alpha=0.18)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def panel_label(ax, label: str) -> None:
    ax.text(
        0.012,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        fontweight="bold",
    )


def save_both(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)

    png = Path(os.path.abspath(os.path.normpath(str(out_base.with_suffix(".png")))))
    pdf = Path(os.path.abspath(os.path.normpath(str(out_base.with_suffix(".pdf")))))

    fig.savefig(str(png), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(str(pdf), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    log(f"[SAVED] {png}")
    log(f"[SAVED] {pdf}")


# =============================================================================
# Physics / apparent wind
# =============================================================================

def apparent_from_joint(joint: np.ndarray) -> np.ndarray:
    """
    joint[..., 0:2] = true-wind U,V
    joint[..., 2:4] = vessel east,north velocity

    apparent = wind - vessel_velocity
    """
    joint = np.asarray(joint, dtype=np.float64)

    if joint.ndim < 2 or joint.shape[-1] < 4:
        raise ValueError(f"Unexpected joint shape: {joint.shape}")

    return joint[..., 0:2] - joint[..., 2:4]


def make_joint(wind_2: np.ndarray, motion_4: np.ndarray) -> np.ndarray:
    """
    wind_2   : (N,5,2)
    motion_4 : (N,5,4)
    return   : (N,5,6)
    """
    if wind_2.shape[:2] != motion_4.shape[:2]:
        raise ValueError("wind and motion horizons/samples do not match")

    out = np.empty((wind_2.shape[0], wind_2.shape[1], 6), dtype=np.float32)
    out[..., 0:2] = wind_2
    out[..., 2:6] = motion_4
    return out


# =============================================================================
# Data loading
# =============================================================================

def load_test(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    path = dataset_dir / "test.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file:\n{path}")

    with np.load(path, allow_pickle=False) as z:
        if "y_raw" not in z.files:
            raise KeyError(f"{path}: missing key y_raw")
        if "target_time_ns" not in z.files:
            raise KeyError(f"{path}: missing key target_time_ns")

        y = np.asarray(z["y_raw"], dtype=np.float32)
        tt = np.asarray(z["target_time_ns"], dtype=np.int64)

    if y.ndim != 3 or y.shape[1:] != (5, 6):
        raise RuntimeError(f"Unexpected y_raw shape: {y.shape}")
    if tt.shape != (len(y), 5):
        raise RuntimeError(f"Unexpected target_time_ns shape: {tt.shape}")

    return y, tt


def first_existing_key(z, keys: List[str]) -> np.ndarray:
    for k in keys:
        if k in z.files:
            return np.asarray(z[k], dtype=np.float32)
    raise KeyError(f"None of keys {keys} found; available={list(z.files)}")


def discover_pc_file(pred_dir: Path, seed: int) -> Optional[Path]:
    candidates = [
        pred_dir / f"15E_VH_seed_{seed}_test_predictions.npz",
        pred_dir / f"15E_VH_seed_{seed}_internal_evaluation_predictions.npz",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_predictions(eval_dir: Path, truth_ref: np.ndarray) -> Dict[str, np.ndarray]:
    pred_dir = eval_dir / "predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(
            f"Prediction directory not found:\n{pred_dir}\n"
            f"Please run 10A internal evaluation first."
        )

    out: Dict[str, np.ndarray] = {}

    # ---------------- Physics-Compact ----------------
    pc_preds = []

    for seed in SEEDS:
        p = discover_pc_file(pred_dir, seed)
        if p is None:
            raise FileNotFoundError(
                f"Physics-Compact prediction for seed {seed} not found under:\n{pred_dir}"
            )

        with np.load(p, allow_pickle=False) as z:
            truth = first_existing_key(z, ["y_raw", "truth_raw", "target_raw"])
            wind = first_existing_key(
                z,
                ["frozen_truewind", "truewind", "pred_wind_raw", "wind_raw"],
            )
            motion = first_existing_key(
                z,
                ["joint_vessel_hdg", "vessel_hdg", "pred_motion_hdg_raw", "motion_hdg_raw"],
            )

        if truth.shape != truth_ref.shape or not np.allclose(truth, truth_ref, atol=2e-5, rtol=0):
            raise RuntimeError(f"{p}: truth does not match the frozen SD1090 internal-evaluation split")

        if wind.shape != (len(truth_ref), 5, 2):
            raise RuntimeError(f"{p}: bad wind shape {wind.shape}")
        if motion.shape != (len(truth_ref), 5, 4):
            raise RuntimeError(f"{p}: bad motion shape {motion.shape}")

        pc_preds.append(make_joint(wind, motion))
        log(f"[LOAD] Physics-Compact seed={seed}: {p.name}")

    out["Physics-Compact"] = np.mean(np.stack(pc_preds, axis=0), axis=0).astype(np.float32)

    # ---------------- Modern baselines ----------------
    for model in MODERN_MODELS:
        safe = model.replace("-", "_")
        seed_preds = []

        for seed in SEEDS:
            p = pred_dir / f"test_{safe}_seed{seed}.npz"
            if not p.exists():
                raise FileNotFoundError(f"Missing modern prediction:\n{p}")

            with np.load(p, allow_pickle=False) as z:
                pred = first_existing_key(z, ["pred_raw", "y_pred", "prediction_raw", "pred"])
                truth = first_existing_key(z, ["truth_raw", "y_raw", "target_raw"])

            if truth.shape != truth_ref.shape or not np.allclose(truth, truth_ref, atol=2e-5, rtol=0):
                raise RuntimeError(f"{p}: truth does not match the frozen SD1090 internal-evaluation split")

            if pred.shape != truth_ref.shape:
                raise RuntimeError(f"{p}: bad pred shape {pred.shape}")

            seed_preds.append(pred)
            log(f"[LOAD] {model} seed={seed}: {p.name}")

        out[model] = np.mean(np.stack(seed_preds, axis=0), axis=0).astype(np.float32)

    return out


# =============================================================================
# Interval selection
# =============================================================================

def get_interval_slice(tt_h: np.ndarray, start_ns: int, end_ns: int) -> Tuple[int, int]:
    """
    Returns [a:b) such that target timestamps satisfy start_ns <= t <= end_ns.
    """
    tt_h = np.asarray(tt_h, dtype=np.int64)

    mask = (tt_h >= start_ns) & (tt_h <= end_ns)
    idx = np.where(mask)[0]

    if len(idx) == 0:
        raise RuntimeError(
            f"No samples found in interval {utc_text(start_ns)} to {utc_text(end_ns)}"
        )

    # Check contiguity
    if not np.all(np.diff(idx) == 1):
        raise RuntimeError(
            "Selected interval is not contiguous in sample index; "
            "please check target timestamps."
        )

    a = int(idx[0])
    b = int(idx[-1]) + 1
    return a, b


# =============================================================================
# Plotting
# =============================================================================

def plot_single_horizon(
    output_dir: Path,
    horizon: int,
    truth: np.ndarray,
    pred_mean: Dict[str, np.ndarray],
    target_time_ns: np.ndarray,
    start_ns: int,
    end_ns: int,
) -> None:
    h_idx = int(np.where(HORIZONS == horizon)[0][0])

    a, b = get_interval_slice(target_time_ns[:, h_idx], start_ns, end_ns)

    obs = apparent_from_joint(truth[a:b, h_idx, :])

    pred_app = {
        model: apparent_from_joint(pred_mean[model][a:b, h_idx, :])
        for model in MODEL_ORDER
    }

    # x-axis as elapsed hours within selected interval
    tt = target_time_ns[a:b, h_idx].astype(np.int64)
    t_hours = (tt - tt[0]) / 3600e9

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.15, 4.15),
        sharex=True,
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.095,
        right=0.992,
        bottom=0.12,
        top=0.865,
        hspace=0.10,
    )

    observed_lw = 1.00
    model_lw = 0.74

    # ---------------- (a) U_app ----------------
    ax = axes[0]

    ax.plot(
        t_hours, obs[:, 0],
        color=COLORS["Observed"],
        linewidth=observed_lw,
        linestyle="-",
        label="Observed",
        zorder=6,
    )

    for name in MODEL_ORDER:
        ax.plot(
            t_hours, pred_app[name][:, 0],
            color=COLORS[name],
            linewidth=model_lw,
            linestyle="-",
            label=name,
            alpha=0.97,
            zorder=4,
        )

    ax.set_ylabel(r"Apparent-wind $U_{\mathrm{app}}$ (m s$^{-1}$)")
    ax.text(
        0.985, 0.965,
        f"{horizon}-min horizon",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
    )
    panel_label(ax, "(a)")
    clean_axis(ax)

    # ---------------- (b) V_app ----------------
    ax = axes[1]

    ax.plot(
        t_hours, obs[:, 1],
        color=COLORS["Observed"],
        linewidth=observed_lw,
        linestyle="-",
        label="Observed",
        zorder=6,
    )

    for name in MODEL_ORDER:
        ax.plot(
            t_hours, pred_app[name][:, 1],
            color=COLORS[name],
            linewidth=model_lw,
            linestyle="-",
            label=name,
            alpha=0.97,
            zorder=4,
        )

    ax.set_xlabel("Time within selected internal-evaluation interval (h)")
    ax.set_ylabel(r"Apparent-wind $V_{\mathrm{app}}$ (m s$^{-1}$)")
    ax.text(
        0.985, 0.965,
        f"{horizon}-min horizon",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
    )
    panel_label(ax, "(b)")
    clean_axis(ax)

    # Keep exact interval width
    if len(t_hours) > 1:
        axes[-1].set_xlim(0.0, float(t_hours[-1]))

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.988),
        ncol=6,
        frameon=False,
        fontsize=6.8,
        columnspacing=0.95,
        handlelength=2.0,
        handletextpad=0.38,
    )

    fig.suptitle(
        f"SD1090 internal evaluation: {utc_text(start_ns)} to {utc_text(end_ns)}",
        y=0.915,
        fontsize=8.5,
    )

    out_base = output_dir / f"Fig03_appUV_modern_h{horizon}"
    save_both(fig, out_base)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        type=Path,
        default=Path(r"D:\project\WindPredict_SaildroneData"),
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
        help="10A internal-evaluation output directory",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    ap.add_argument(
        "--start-time",
        type=str,
        default="2025-01-07 12:14 UTC",
    )
    ap.add_argument(
        "--end-time",
        type=str,
        default="2025-01-07 20:13 UTC",
    )

    args = ap.parse_args()

    root = args.root.resolve()

    dataset_dir = (
        args.dataset_dir.resolve()
        if args.dataset_dir is not None
        else root / "data" / "forecasting" / "SD1090_TPOS2024_JointForecasting_v0_1"
    )

    eval_dir = (
        args.eval_dir.resolve()
        if args.eval_dir is not None
        else dataset_dir / "10A_SD1090_internal_evaluation_v0_4"
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "figures" / "final_paper"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    start_ns = parse_utc_to_ns(args.start_time)
    end_ns = parse_utc_to_ns(args.end_time)

    if end_ns <= start_ns:
        raise ValueError("end-time must be later than start-time")

    load_sci_style(root)

    log("=" * 100)
    log("PLOT THREE SEPARATE APPARENT-WIND FIGURES (1, 5, 10 min)")
    log("=" * 100)
    log(f"dataset_dir : {dataset_dir}")
    log(f"eval_dir    : {eval_dir}")
    log(f"output_dir  : {output_dir}")
    log(f"interval    : {utc_text(start_ns)} to {utc_text(end_ns)}")
    log("=" * 100)

    truth, target_time_ns = load_test(dataset_dir)
    pred_mean = load_predictions(eval_dir, truth)

    # Optional audit: confirm target spacing for the three horizons
    for h in PLOT_HORIZONS:
        j = int(np.where(HORIZONS == h)[0][0])
        diffs = np.diff(target_time_ns[:, j]).astype(np.float64) / 60e9
        log(f"[AUDIT] horizon={h} min | median target-time step = {np.median(diffs):.3f} min")

    for h in PLOT_HORIZONS:
        plot_single_horizon(
            output_dir=output_dir,
            horizon=h,
            truth=truth,
            pred_mean=pred_mean,
            target_time_ns=target_time_ns,
            start_ns=start_ns,
            end_ns=end_ns,
        )

    log("")
    log("=" * 100)
    log("DONE")
    log("=" * 100)


if __name__ == "__main__":
    main()