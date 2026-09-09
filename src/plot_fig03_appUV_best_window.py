# -*- coding: utf-8 -*-
r"""
plot_fig03_appUV_best_window.py

Final qualitative time-series figure for the SD1090 validation set.

Upper panel:
    Apparent-wind U component
Lower panel:
    Apparent-wind V component

Curves:
    Observed
    Physics-Compact
    LSTM
    GRU
    CNN-LSTM

Definitions
-----------
Earth-frame apparent-wind components are reconstructed consistently as

    U_app = U_true_wind - V_ship,east
    V_app = V_true_wind - V_ship,north

for both observations and predictions.

Window-selection modes
----------------------
1) advantage   [default, requested by user]
   Selects a sufficiently dynamic window where Physics-Compact has the
   largest relative vector-RMSE advantage over the mean of the comparison
   neural models.

2) variability
   Selects the window with the largest observed apparent-wind component
   variability. This does NOT use prediction error and is preferable if the
   figure is intended as an unbiased "representative" manuscript example.

3) fixed
   Uses --start-index directly.

IMPORTANT:
If "advantage" is used in the paper, disclose that the interval was selected
by a performance-based criterion. Do not describe it as a random or typical
segment.

Default paths assume:
D:\project\WindPredict_SaildroneData

Physics-Compact:
data\forecasting\15E_VH_AlphaSearch_v0_1\full_retrain\alpha_1500

Conventional baselines:
data\forecasting\16A_ConventionalDeepBaselines_Final_v0_1\predictions

Example
-------
python "D:\project\WindPredict_SaildroneData\src\plot_fig03_appUV_best_window.py" `
  --root "D:\project\WindPredict_SaildroneData" `
  --window-mode advantage `
  --window-hours 2 `
  --horizon 10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


HORIZONS = np.asarray([1, 2, 3, 5, 10], dtype=int)
SEEDS = [500043, 501052, 502061, 503070, 504079]

COLORS = {
    "Observed": "#000000",
    "Physics-Compact": "#1f77b4",
    "LSTM": "#ff7f0e",
    "GRU": "#2ca02c",
    "CNN-LSTM": "#d62728",
}

MARKERS = {
    "Physics-Compact": None,
    "LSTM": None,
    "GRU": None,
    "CNN-LSTM": None,
}


# =============================================================================
# Style
# =============================================================================

def load_sci_style(root: Path) -> None:
    style_path = root / "src" / "sci_plot_style.py"

    if style_path.exists():
        spec = importlib.util.spec_from_file_location(
            "sci_plot_style_fig03",
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
                    mod.apply_sci_style(font_size=8.5)
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
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.5,
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
    ax.grid(True, which="major", linewidth=0.42, alpha=0.20)

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
        fontsize=9.0,
        fontweight="bold",
    )


def save_both(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)

    png = Path(os.path.abspath(os.path.normpath(str(out_base.with_suffix(".png")))))
    pdf = Path(os.path.abspath(os.path.normpath(str(out_base.with_suffix(".pdf")))))

    fig.savefig(
        str(png),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        str(pdf),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print(f"[SAVED] {png}")
    print(f"[SAVED] {pdf}")


# =============================================================================
# Data loading
# =============================================================================

def app_uv_from_joint(y: np.ndarray) -> np.ndarray:
    """
    y shape: [N, 5, 6]
      0 U
      1 V
      2 vessel east
      3 vessel north
      4 HDG sin
      5 HDG cos

    Returns Earth-frame apparent wind [U_app, V_app].
    """
    return np.asarray(y[..., 0:2], dtype=np.float64) - np.asarray(
        y[..., 2:4],
        dtype=np.float64,
    )


def load_physics_predictions(
    physics_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    y_true : [N,5,6]
    y_pred : [N,5,6]
    averaged across the five final Physics-Compact seeds.
    """
    preds = []
    truths = []

    for seed in SEEDS:
        p = physics_dir / f"15E_VH_seed_{seed}_validation_predictions.npz"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing final Physics-Compact validation prediction file:\n{p}"
            )

        with np.load(p, allow_pickle=False) as z:
            required = ["y_raw", "frozen_truewind", "joint_vessel_hdg"]
            missing = [k for k in required if k not in z]
            if missing:
                raise KeyError(f"{p}: missing keys {missing}")

            y_true = np.asarray(z["y_raw"], dtype=np.float32)
            wind = np.asarray(z["frozen_truewind"], dtype=np.float32)
            motion = np.asarray(z["joint_vessel_hdg"], dtype=np.float32)

        y_pred = np.empty_like(y_true)
        y_pred[..., 0:2] = wind
        y_pred[..., 2:6] = motion

        truths.append(y_true)
        preds.append(y_pred)

    y0 = truths[0]
    for k, y in enumerate(truths[1:], start=1):
        if not np.allclose(y, y0, atol=1e-7, rtol=0):
            raise RuntimeError(
                f"Physics-Compact seed truth mismatch at seed index {k}."
            )

    pred_mean = np.mean(np.stack(preds, axis=0), axis=0)

    return y0.astype(np.float32), pred_mean.astype(np.float32)


def load_conventional_model_predictions(
    prediction_dir: Path,
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the final five-seed predictions generated by
    16A_run_conventional_neural_baselines_final.py
    and averages predictions across seeds.
    """
    safe_name = model_name.replace("-", "_")

    preds = []
    truths = []

    for seed in SEEDS:
        p = (
            prediction_dir
            / f"{safe_name}_seed_{seed}_validation_predictions.npz"
        )

        if not p.exists():
            raise FileNotFoundError(
                f"Missing {model_name} final validation prediction:\n{p}\n\n"
                "Run 16A_run_conventional_neural_baselines_final.py first."
            )

        with np.load(p, allow_pickle=False) as z:
            if "y_raw" not in z or "y_pred" not in z:
                raise KeyError(
                    f"{p} must contain y_raw and y_pred."
                )

            y_true = np.asarray(z["y_raw"], dtype=np.float32)
            y_pred = np.asarray(z["y_pred"], dtype=np.float32)

        truths.append(y_true)
        preds.append(y_pred)

    y0 = truths[0]
    for k, y in enumerate(truths[1:], start=1):
        if not np.allclose(y, y0, atol=1e-7, rtol=0):
            raise RuntimeError(
                f"{model_name}: truth mismatch at seed index {k}."
            )

    pred_mean = np.mean(np.stack(preds, axis=0), axis=0)

    return y0.astype(np.float32), pred_mean.astype(np.float32)


# =============================================================================
# Window selection
# =============================================================================

def sliding_sum(x: np.ndarray, window: int) -> np.ndarray:
    """
    Sliding sum over a 1D array.
    Returns length len(x)-window+1.
    """
    x = np.asarray(x, dtype=np.float64)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    return cs[window:] - cs[:-window]


def sliding_mean(x: np.ndarray, window: int) -> np.ndarray:
    return sliding_sum(x, window) / float(window)


def observed_variability_score(
    obs_uv: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    sqrt(var(U_app) + var(V_app)) in each window.
    """
    u = obs_uv[:, 0]
    v = obs_uv[:, 1]

    mu_u = sliding_mean(u, window)
    mu_v = sliding_mean(v, window)

    var_u = sliding_mean(u * u, window) - mu_u * mu_u
    var_v = sliding_mean(v * v, window) - mu_v * mu_v

    return np.sqrt(np.maximum(var_u + var_v, 0.0))


def vector_rmse_sliding(
    obs_uv: np.ndarray,
    pred_uv: np.ndarray,
    window: int,
) -> np.ndarray:
    err = np.asarray(pred_uv, dtype=np.float64) - np.asarray(
        obs_uv,
        dtype=np.float64,
    )
    sq = np.sum(err * err, axis=1)
    return np.sqrt(sliding_mean(sq, window))


def select_window(
    obs_uv: np.ndarray,
    predictions: Dict[str, np.ndarray],
    window: int,
    mode: str,
    start_index: int,
    variability_percentile: float,
) -> Tuple[int, pd.DataFrame]:
    n = len(obs_uv)

    if window > n:
        raise ValueError(
            f"window={window} exceeds available samples={n}"
        )

    variability = observed_variability_score(obs_uv, window)

    rows = {
        "start_index": np.arange(len(variability), dtype=int),
        "variability": variability,
    }

    rmse_by_model = {}

    for name, pred in predictions.items():
        rmse = vector_rmse_sliding(obs_uv, pred, window)
        rmse_by_model[name] = rmse
        rows[f"rmse_{name}"] = rmse

    diagnostic = pd.DataFrame(rows)

    if mode == "fixed":
        if start_index < 0 or start_index + window > n:
            raise ValueError(
                f"Invalid --start-index={start_index} for window={window}, N={n}"
            )
        selected = start_index

    elif mode == "variability":
        selected = int(np.nanargmax(variability))

    elif mode == "advantage":
        if "Physics-Compact" not in rmse_by_model:
            raise RuntimeError(
                "Physics-Compact is required for advantage mode."
            )

        comparator_names = [
            name
            for name in rmse_by_model
            if name != "Physics-Compact"
        ]

        if not comparator_names:
            raise RuntimeError(
                "At least one comparison model is required."
            )

        comp = np.mean(
            np.stack(
                [rmse_by_model[name] for name in comparator_names],
                axis=0,
            ),
            axis=0,
        )

        pc = rmse_by_model["Physics-Compact"]

        # Relative improvement over the mean comparator.
        skill = (comp - pc) / np.maximum(comp, 1e-12)

        # Avoid selecting an artificially easy/flat interval:
        # require the observed variability to be above a chosen percentile.
        threshold = np.nanpercentile(
            variability,
            variability_percentile,
        )
        eligible = variability >= threshold

        score = np.full_like(skill, -np.inf)
        score[eligible] = skill[eligible]

        selected = int(np.nanargmax(score))

        diagnostic["mean_comparator_rmse"] = comp
        diagnostic["physics_skill_fraction"] = skill
        diagnostic["eligible"] = eligible.astype(int)

    else:
        raise ValueError(mode)

    diagnostic["selected"] = 0
    diagnostic.loc[
        diagnostic["start_index"] == selected,
        "selected",
    ] = 1

    return selected, diagnostic


# =============================================================================
# Plot
# =============================================================================

def component_rmse(
    obs: np.ndarray,
    pred: np.ndarray,
) -> Tuple[float, float, float]:
    err = np.asarray(pred) - np.asarray(obs)

    u_rmse = float(np.sqrt(np.mean(err[:, 0] ** 2)))
    v_rmse = float(np.sqrt(np.mean(err[:, 1] ** 2)))
    vec_rmse = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))

    return u_rmse, v_rmse, vec_rmse


def plot_selected_window(
    output_dir: Path,
    obs_uv: np.ndarray,
    predictions: Dict[str, np.ndarray],
    start: int,
    window: int,
    horizon: int,
    selection_mode: str,
) -> pd.DataFrame:
    stop = start + window

    obs = obs_uv[start:stop]

    # Samples are one minute apart.
    t_hours = np.arange(window, dtype=float) / 60.0

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
        top=0.86,
        hspace=0.11,
    )

    # Thin SCI curves.
    obs_lw = 1.05
    model_lw = 0.90

    # ------------------------------------------------------------------
    # U_app
    # ------------------------------------------------------------------
    ax = axes[0]

    ax.plot(
        t_hours,
        obs[:, 0],
        color=COLORS["Observed"],
        linewidth=obs_lw,
        label="Observed",
        zorder=6,
    )

    for name, pred_all in predictions.items():
        pred = pred_all[start:stop]

        ax.plot(
            t_hours,
            pred[:, 0],
            color=COLORS[name],
            linewidth=model_lw,
            linestyle="-",
            label=name,
            zorder=4,
        )

    ax.set_ylabel(
        r"Apparent-wind $U_{\mathrm{app}}$ (m s$^{-1}$)"
    )
    panel_label(ax, "(a)")
    clean_axis(ax)

    # ------------------------------------------------------------------
    # V_app
    # ------------------------------------------------------------------
    ax = axes[1]

    ax.plot(
        t_hours,
        obs[:, 1],
        color=COLORS["Observed"],
        linewidth=obs_lw,
        label="Observed",
        zorder=6,
    )

    for name, pred_all in predictions.items():
        pred = pred_all[start:stop]

        ax.plot(
            t_hours,
            pred[:, 1],
            color=COLORS[name],
            linewidth=model_lw,
            linestyle="-",
            label=name,
            zorder=4,
        )

    ax.set_xlabel("Time within selected validation interval (h)")
    ax.set_ylabel(
        r"Apparent-wind $V_{\mathrm{app}}$ (m s$^{-1}$)"
    )
    panel_label(ax, "(b)")
    clean_axis(ax)

    # One legend above both panels.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        frameon=False,
        fontsize=7.3,
        columnspacing=1.2,
        handlelength=2.2,
        handletextpad=0.45,
    )

    # Small, transparent annotation of how this interval was selected.
    axes[1].text(
        0.995,
        0.025,
        f"{horizon}-min horizon; {window/60.0:g}-h window; "
        f"selection={selection_mode}",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.6,
    )

    out_base = output_dir / "Fig03_apparent_UV_selected_window"
    save_both(fig, out_base)

    # ------------------------------------------------------------------
    # Metrics for the exact plotted window.
    # ------------------------------------------------------------------
    rows = []

    for name, pred_all in predictions.items():
        pred = pred_all[start:stop]
        u_rmse, v_rmse, vec_rmse = component_rmse(obs, pred)

        rows.append(
            {
                "model": name,
                "start_index": start,
                "stop_index_exclusive": stop,
                "window_samples": window,
                "window_hours": window / 60.0,
                "horizon_min": horizon,
                "selection_mode": selection_mode,
                "Uapp_RMSE_mps": u_rmse,
                "Vapp_RMSE_mps": v_rmse,
                "AppVector_RMSE_mps": vec_rmse,
            }
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(
        output_dir / "Fig03_selected_window_metrics.csv",
        index=False,
    )

    return metrics


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"D:\project\WindPredict_SaildroneData"),
    )
    parser.add_argument(
        "--physics-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--conventional-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing final LSTM/GRU/CNN-LSTM validation "
            "prediction NPZ files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=[1, 2, 3, 5, 10],
        default=10,
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--window-mode",
        choices=["advantage", "variability", "fixed"],
        default="advantage",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--variability-percentile",
        type=float,
        default=60.0,
        help=(
            "In advantage mode, only windows at or above this observed "
            "variability percentile are eligible."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["LSTM", "GRU", "CNN-LSTM"],
        default=["LSTM", "GRU", "CNN-LSTM"],
    )

    args = parser.parse_args()

    root = args.root.resolve()

    physics_dir = (
        args.physics_dir.resolve()
        if args.physics_dir is not None
        else root
        / "data"
        / "forecasting"
        / "15E_VH_AlphaSearch_v0_1"
        / "full_retrain"
        / "alpha_1500"
    )

    conventional_dir = (
        args.conventional_dir.resolve()
        if args.conventional_dir is not None
        else root
        / "data"
        / "forecasting"
        / "16A_ConventionalDeepBaselines_Final_v0_1"
        / "predictions"
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root
        / "figures"
        / "final_paper"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    load_sci_style(root)

    print("=" * 100)
    print("FINAL FIG. 3 — APPARENT-WIND U/V TIME SERIES")
    print("=" * 100)
    print(f"root              : {root}")
    print(f"physics_dir       : {physics_dir}")
    print(f"conventional_dir  : {conventional_dir}")
    print(f"output_dir        : {output_dir}")
    print(f"horizon           : {args.horizon} min")
    print(f"window_hours      : {args.window_hours:g}")
    print(f"window_mode       : {args.window_mode}")
    print(f"comparison models : {args.models}")
    print("=" * 100)

    # ------------------------------------------------------------------
    # Load final Physics-Compact.
    # ------------------------------------------------------------------
    y_true, pc_joint = load_physics_predictions(physics_dir)

    h_idx = int(np.where(HORIZONS == args.horizon)[0][0])

    obs_app = app_uv_from_joint(y_true)[:, h_idx, :]
    pc_app = app_uv_from_joint(pc_joint)[:, h_idx, :]

    predictions: Dict[str, np.ndarray] = {
        "Physics-Compact": pc_app,
    }

    # ------------------------------------------------------------------
    # Load conventional neural models.
    # ------------------------------------------------------------------
    for model_name in args.models:
        y_cmp, pred_cmp = load_conventional_model_predictions(
            conventional_dir,
            model_name,
        )

        if not np.allclose(
            y_cmp,
            y_true,
            atol=1e-6,
            rtol=0,
        ):
            raise RuntimeError(
                f"{model_name}: y_raw does not match Physics-Compact truth. "
                "The models are not aligned on identical validation samples."
            )

        predictions[model_name] = (
            app_uv_from_joint(pred_cmp)[:, h_idx, :]
        )

    # ------------------------------------------------------------------
    # Select window.
    # ------------------------------------------------------------------
    window = int(round(args.window_hours * 60.0))

    if window < 10:
        raise ValueError(
            "--window-hours is too short; use at least ~0.17 h (10 min)."
        )

    start, diagnostic = select_window(
        obs_uv=obs_app,
        predictions=predictions,
        window=window,
        mode=args.window_mode,
        start_index=args.start_index,
        variability_percentile=args.variability_percentile,
    )

    stop = start + window

    diagnostic.to_csv(
        output_dir / "Fig03_window_selection_diagnostics.csv",
        index=False,
    )

    print("")
    print("=" * 100)
    print("SELECTED WINDOW")
    print("=" * 100)
    print(f"start index     : {start}")
    print(f"stop index      : {stop} (exclusive)")
    print(f"window samples  : {window}")
    print(f"window duration : {window / 60.0:.3f} h")
    print(f"selection mode  : {args.window_mode}")

    if args.window_mode == "advantage":
        row = diagnostic.loc[
            diagnostic["selected"] == 1
        ].iloc[0]
        print(
            f"observed variability      : {row['variability']:.6f}"
        )
        print(
            f"Physics skill vs mean cmp : "
            f"{100.0 * row['physics_skill_fraction']:.3f}%"
        )

    # ------------------------------------------------------------------
    # Plot.
    # ------------------------------------------------------------------
    metrics = plot_selected_window(
        output_dir=output_dir,
        obs_uv=obs_app,
        predictions=predictions,
        start=start,
        window=window,
        horizon=args.horizon,
        selection_mode=args.window_mode,
    )

    print("")
    print("=" * 100)
    print("EXACT METRICS IN PLOTTED WINDOW")
    print("=" * 100)
    print(
        metrics[
            [
                "model",
                "Uapp_RMSE_mps",
                "Vapp_RMSE_mps",
                "AppVector_RMSE_mps",
            ]
        ].to_string(index=False)
    )

    # Also save a concise selection manifest.
    manifest = {
        "horizon_min": int(args.horizon),
        "window_mode": args.window_mode,
        "window_hours": float(args.window_hours),
        "start_index": int(start),
        "stop_index_exclusive": int(stop),
        "comparison_models": args.models,
        "variability_percentile_if_advantage": float(
            args.variability_percentile
        ),
        "definition": {
            "U_app": "true-wind U minus vessel east velocity",
            "V_app": "true-wind V minus vessel north velocity",
        },
    }

    with open(
        output_dir / "Fig03_window_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2)

    print("")
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
