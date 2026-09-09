# -*- coding: utf-8 -*-
r"""
plot_fig03_appUV_modern_models_v2.py

Corrected Fig. 3 plotting script based directly on the user's Stage-09C output.

Why this version
----------------
09C_refit_modern_baselines_outer_validation.py ALREADY saves per-sample
validation predictions by default:

    <dataset_dir>\09C_modern_baseline_outer_validation_v0_1\predictions\
        validation_DLinear_seed500043.npz
        validation_iTransformer_seed500043.npz
        validation_TimeMixer_seed500043.npz
        validation_PatchTST_seed500043.npz
        ...

Each NPZ contains:
    pred_raw
    truth_raw
    apparent_earth_raw
    context_end_time_ns
    target_time_ns

The previous plotting script failed because it searched for generic keys such as
"y_pred" / "y_raw" and therefore did not recognize the actual 09C keys
"pred_raw" / "truth_raw".

This script uses the exact 09C output contract.

Figure
------
(a) Earth-frame apparent-wind U component
(b) Earth-frame apparent-wind V component

Curves
------
Observed
Physics-Compact
DLinear
TimeMixer
iTransformer
PatchTST

All model curves are five-seed means.

Default selection
-----------------
10-min horizon, 8-h window.

window-mode=advantage:
    among sufficiently dynamic intervals, select the interval where
    Physics-Compact has the largest relative apparent-wind vector-RMSE
    advantage over the mean of the four modern baselines.

window-mode=variability:
    select only by observed apparent-wind variability; no model error enters
    interval selection.

For manuscript use, "variability" is more neutral. If "advantage" is used,
the caption should explicitly state that the interval is performance-selected.

Example
-------
python "D:\project\WindPredict_SaildroneData\src\plot_fig03_appUV_modern_models_v2.py" `
  --root "D:\project\WindPredict_SaildroneData" `
  --horizon 10 `
  --window-hours 8 `
  --window-mode advantage
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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

MODERN_MODELS = [
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
# Style
# =============================================================================

def load_sci_style(root: Path) -> None:
    style_path = root / "src" / "sci_plot_style.py"

    if style_path.exists():
        spec = importlib.util.spec_from_file_location(
            "sci_plot_style_fig03_modern_v2",
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
            "legend.fontsize": 7.1,
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
        fontsize=9.0,
        fontweight="bold",
    )


def save_both(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)

    png = Path(
        os.path.abspath(
            os.path.normpath(str(out_base.with_suffix(".png")))
        )
    )
    pdf = Path(
        os.path.abspath(
            os.path.normpath(str(out_base.with_suffix(".pdf")))
        )
    )

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
# Physics / apparent-wind reconstruction
# =============================================================================

def apparent_from_joint(y_joint: np.ndarray) -> np.ndarray:
    """
    y_joint[...,0:2] = true-wind U,V
    y_joint[...,2:4] = vessel east,north velocity

    Returns Earth-frame apparent wind:
        A_E = U - V_ship,E
        A_N = V - V_ship,N
    """
    y_joint = np.asarray(y_joint, dtype=np.float64)

    if y_joint.ndim != 3 or y_joint.shape[1:] != (5, 6):
        raise ValueError(
            f"Expected joint array (N,5,6), got {y_joint.shape}"
        )

    return y_joint[..., 0:2] - y_joint[..., 2:4]


# =============================================================================
# Final Physics-Compact predictions
# =============================================================================

def load_physicscompact(
    physics_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    truth_joint : [N,5,6]
    pred_joint_mean : [N,5,6], five-seed mean
    """
    truths = []
    preds = []

    for seed in SEEDS:
        p = (
            physics_dir
            / f"15E_VH_seed_{seed}_validation_predictions.npz"
        )

        if not p.exists():
            raise FileNotFoundError(
                f"Missing final Physics-Compact validation prediction:\n{p}"
            )

        with np.load(p, allow_pickle=False) as z:
            required = [
                "y_raw",
                "frozen_truewind",
                "joint_vessel_hdg",
            ]
            missing = [k for k in required if k not in z]

            if missing:
                raise KeyError(
                    f"{p}: missing keys {missing}"
                )

            truth = np.asarray(
                z["y_raw"],
                dtype=np.float32,
            )
            wind = np.asarray(
                z["frozen_truewind"],
                dtype=np.float32,
            )
            motion = np.asarray(
                z["joint_vessel_hdg"],
                dtype=np.float32,
            )

        pred = np.empty_like(truth)
        pred[..., 0:2] = wind
        pred[..., 2:6] = motion

        truths.append(truth)
        preds.append(pred)

    truth0 = truths[0]

    for y in truths[1:]:
        if not np.allclose(
            y,
            truth0,
            atol=1e-7,
            rtol=0,
        ):
            raise RuntimeError(
                "Physics-Compact seed files do not share identical truth."
            )

    pred_mean = np.mean(
        np.stack(preds, axis=0),
        axis=0,
    ).astype(np.float32)

    return truth0, pred_mean


# =============================================================================
# Exact 09C modern predictions
# =============================================================================

def modern_prediction_path(
    predictions_dir: Path,
    model_name: str,
    seed: int,
) -> Path:
    safe_model = model_name.replace("-", "_")

    return (
        predictions_dir
        / f"validation_{safe_model}_seed{seed}.npz"
    )


def load_modern_model(
    predictions_dir: Path,
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads EXACT Stage-09C NPZ keys:
        pred_raw
        truth_raw
        apparent_earth_raw
        target_time_ns

    Returns
    -------
    truth_joint
    pred_joint_mean
    observed_apparent
    target_time_ns
    """
    truths = []
    preds = []
    apparent_refs = []
    target_times = []

    print("")
    print(f"[LOAD] {model_name}")

    for seed in SEEDS:
        p = modern_prediction_path(
            predictions_dir,
            model_name,
            seed,
        )

        if not p.exists():
            raise FileNotFoundError(
                "\n"
                + "=" * 100
                + f"\nMISSING 09C PREDICTION FOR {model_name}, seed={seed}\n"
                + "=" * 100
                + f"\nExpected:\n{p}\n\n"
                + "Your uploaded 09C script saves this file by default.\n"
                + "If the predictions directory is empty, 09C may have been run "
                  "with --no-save-predictions.\n"
            )

        with np.load(p, allow_pickle=False) as z:
            required = [
                "pred_raw",
                "truth_raw",
                "apparent_earth_raw",
                "target_time_ns",
            ]

            missing = [
                k
                for k in required
                if k not in z
            ]

            if missing:
                raise KeyError(
                    f"{p}: expected 09C keys {required}, missing {missing}. "
                    f"Available={list(z.files)}"
                )

            pred = np.asarray(
                z["pred_raw"],
                dtype=np.float32,
            )
            truth = np.asarray(
                z["truth_raw"],
                dtype=np.float32,
            )
            apparent = np.asarray(
                z["apparent_earth_raw"],
                dtype=np.float32,
            )
            target_time = np.asarray(
                z["target_time_ns"]
            )

        if pred.shape[1:] != (5, 6):
            raise RuntimeError(
                f"{p}: pred_raw shape must be (N,5,6), got {pred.shape}"
            )

        if truth.shape != pred.shape:
            raise RuntimeError(
                f"{p}: truth_raw shape {truth.shape} != pred_raw {pred.shape}"
            )

        if apparent.shape[:2] != pred.shape[:2] or apparent.shape[-1] != 2:
            raise RuntimeError(
                f"{p}: apparent_earth_raw has unexpected shape {apparent.shape}"
            )

        print(
            f"       seed={seed} | pred_raw={pred.shape} | "
            f"truth_raw={truth.shape}"
        )

        truths.append(truth)
        preds.append(pred)
        apparent_refs.append(apparent)
        target_times.append(target_time)

    truth0 = truths[0]
    apparent0 = apparent_refs[0]
    target_time0 = target_times[0]

    for i in range(1, len(SEEDS)):
        if not np.allclose(
            truths[i],
            truth0,
            atol=1e-6,
            rtol=0,
        ):
            raise RuntimeError(
                f"{model_name}: truth_raw differs across seeds."
            )

        if not np.allclose(
            apparent_refs[i],
            apparent0,
            atol=1e-6,
            rtol=0,
        ):
            raise RuntimeError(
                f"{model_name}: apparent_earth_raw differs across seeds."
            )

        if not np.array_equal(
            target_times[i],
            target_time0,
        ):
            raise RuntimeError(
                f"{model_name}: target_time_ns differs across seeds."
            )

    pred_mean = np.mean(
        np.stack(preds, axis=0),
        axis=0,
    ).astype(np.float32)

    return (
        truth0,
        pred_mean,
        apparent0,
        target_time0,
    )


# =============================================================================
# Sliding-window selection
# =============================================================================

def sliding_sum(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)

    cs = np.concatenate(
        [
            np.asarray([0.0]),
            np.cumsum(x),
        ]
    )

    return cs[window:] - cs[:-window]


def sliding_mean(x: np.ndarray, window: int) -> np.ndarray:
    return sliding_sum(x, window) / float(window)


def variability_score(
    observed_uv: np.ndarray,
    window: int,
) -> np.ndarray:
    u = observed_uv[:, 0]
    v = observed_uv[:, 1]

    mu_u = sliding_mean(u, window)
    mu_v = sliding_mean(v, window)

    var_u = (
        sliding_mean(u * u, window)
        - mu_u * mu_u
    )
    var_v = (
        sliding_mean(v * v, window)
        - mu_v * mu_v
    )

    return np.sqrt(
        np.maximum(
            var_u + var_v,
            0.0,
        )
    )


def sliding_vector_rmse(
    observed_uv: np.ndarray,
    predicted_uv: np.ndarray,
    window: int,
) -> np.ndarray:
    err = (
        np.asarray(predicted_uv, dtype=np.float64)
        - np.asarray(observed_uv, dtype=np.float64)
    )

    sq = np.sum(
        err * err,
        axis=1,
    )

    return np.sqrt(
        sliding_mean(sq, window)
    )


def select_window(
    observed_uv: np.ndarray,
    predictions: Dict[str, np.ndarray],
    window: int,
    mode: str,
    fixed_start: int,
    variability_percentile: float,
) -> Tuple[int, pd.DataFrame]:
    n = len(observed_uv)

    if window > n:
        raise ValueError(
            f"Requested window={window} exceeds N={n}"
        )

    var = variability_score(
        observed_uv,
        window,
    )

    diagnostic = pd.DataFrame(
        {
            "start_index": np.arange(
                len(var),
                dtype=int,
            ),
            "observed_variability": var,
        }
    )

    rmse_by_model = {}

    for name, pred in predictions.items():
        r = sliding_vector_rmse(
            observed_uv,
            pred,
            window,
        )

        rmse_by_model[name] = r
        diagnostic[
            f"RMSE_{name}"
        ] = r

    if mode == "fixed":
        if fixed_start < 0 or fixed_start + window > n:
            raise ValueError(
                f"Invalid fixed start index {fixed_start}"
            )

        selected = int(
            fixed_start
        )

    elif mode == "variability":
        selected = int(
            np.nanargmax(var)
        )

    elif mode == "advantage":
        pc = rmse_by_model[
            "Physics-Compact"
        ]

        modern_mean = np.mean(
            np.stack(
                [
                    rmse_by_model[m]
                    for m in MODERN_MODELS
                ],
                axis=0,
            ),
            axis=0,
        )

        skill = (
            modern_mean - pc
        ) / np.maximum(
            modern_mean,
            1e-12,
        )

        var_threshold = np.nanpercentile(
            var,
            variability_percentile,
        )

        eligible = (
            var >= var_threshold
        )

        score = np.full_like(
            skill,
            -np.inf,
        )
        score[eligible] = skill[eligible]

        selected = int(
            np.nanargmax(score)
        )

        diagnostic[
            "mean_modern_RMSE"
        ] = modern_mean
        diagnostic[
            "PhysicsCompact_skill_fraction"
        ] = skill
        diagnostic[
            "eligible"
        ] = eligible.astype(int)

    else:
        raise ValueError(mode)

    diagnostic[
        "selected"
    ] = 0

    diagnostic.loc[
        diagnostic["start_index"] == selected,
        "selected",
    ] = 1

    return selected, diagnostic


# =============================================================================
# Plot metrics
# =============================================================================

def component_rmse(
    obs: np.ndarray,
    pred: np.ndarray,
) -> Tuple[float, float, float]:
    err = (
        np.asarray(pred, dtype=np.float64)
        - np.asarray(obs, dtype=np.float64)
    )

    u_rmse = float(
        np.sqrt(
            np.mean(
                err[:, 0] ** 2
            )
        )
    )
    v_rmse = float(
        np.sqrt(
            np.mean(
                err[:, 1] ** 2
            )
        )
    )
    vector_rmse = float(
        np.sqrt(
            np.mean(
                np.sum(
                    err * err,
                    axis=1,
                )
            )
        )
    )

    return (
        u_rmse,
        v_rmse,
        vector_rmse,
    )


# =============================================================================
# Plot
# =============================================================================

def plot_window(
    output_dir: Path,
    observed_uv: np.ndarray,
    predictions: Dict[str, np.ndarray],
    target_time_ns_h: np.ndarray,
    start: int,
    window: int,
    horizon: int,
    selection_mode: str,
) -> pd.DataFrame:
    stop = start + window

    obs = observed_uv[
        start:stop
    ]

    # Plot elapsed time rather than absolute timestamp to keep the figure compact.
    t_hours = (
        np.arange(
            window,
            dtype=float,
        )
        / 60.0
    )

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
        hspace=0.10,
    )

    observed_lw = 1.00
    model_lw = 0.74

    model_order = [
        "Physics-Compact",
        "DLinear",
        "TimeMixer",
        "iTransformer",
        "PatchTST",
    ]

    # ------------------------------------------------------------------
    # U_app
    # ------------------------------------------------------------------
    ax = axes[0]

    ax.plot(
        t_hours,
        obs[:, 0],
        color=COLORS["Observed"],
        linewidth=observed_lw,
        linestyle="-",
        label="Observed",
        zorder=6,
    )

    for name in model_order:
        pred = predictions[
            name
        ][
            start:stop
        ]

        ax.plot(
            t_hours,
            pred[:, 0],
            color=COLORS[name],
            linewidth=model_lw,
            linestyle="-",
            label=name,
            alpha=0.97,
            zorder=4,
        )

    ax.set_ylabel(
        r"Apparent-wind $U_{\mathrm{app}}$ (m s$^{-1}$)"
    )

    panel_label(
        ax,
        "(a)",
    )

    clean_axis(
        ax
    )

    # ------------------------------------------------------------------
    # V_app
    # ------------------------------------------------------------------
    ax = axes[1]

    ax.plot(
        t_hours,
        obs[:, 1],
        color=COLORS["Observed"],
        linewidth=observed_lw,
        linestyle="-",
        label="Observed",
        zorder=6,
    )

    for name in model_order:
        pred = predictions[
            name
        ][
            start:stop
        ]

        ax.plot(
            t_hours,
            pred[:, 1],
            color=COLORS[name],
            linewidth=model_lw,
            linestyle="-",
            label=name,
            alpha=0.97,
            zorder=4,
        )

    ax.set_xlabel(
        "Time within selected validation interval (h)"
    )
    ax.set_ylabel(
        r"Apparent-wind $V_{\mathrm{app}}$ (m s$^{-1}$)"
    )

    panel_label(
        ax,
        "(b)",
    )

    clean_axis(
        ax
    )

    # ------------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------------
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=6,
        frameon=False,
        fontsize=6.8,
        columnspacing=0.95,
        handlelength=2.0,
        handletextpad=0.38,
    )

    # No "selection=advantage" text inside the figure.
    # Selection details are written to CSV/JSON and should be described
    # in the manuscript caption if necessary.

    save_both(
        fig,
        output_dir
        / "Fig03_apparent_UV_modern_models",
    )

    # ------------------------------------------------------------------
    # Exact metrics in the plotted interval
    # ------------------------------------------------------------------
    rows = []

    for name, pred_all in predictions.items():
        pred = pred_all[
            start:stop
        ]

        (
            u_rmse,
            v_rmse,
            vector_rmse,
        ) = component_rmse(
            obs,
            pred,
        )

        rows.append(
            {
                "model": name,
                "horizon_min": horizon,
                "selection_mode": selection_mode,
                "start_index": start,
                "stop_index_exclusive": stop,
                "window_hours": window / 60.0,
                "Uapp_RMSE_mps": u_rmse,
                "Vapp_RMSE_mps": v_rmse,
                "AppVector_RMSE_mps": vector_rmse,
            }
        )

    metrics = pd.DataFrame(
        rows
    )

    metrics.to_csv(
        output_dir
        / "Fig03_apparent_UV_modern_models_window_metrics.csv",
        index=False,
    )

    # Save exact timestamps for traceability.
    selected_time = target_time_ns_h[
        start:stop
    ]

    np.savez_compressed(
        output_dir
        / "Fig03_selected_interval_traceability.npz",
        target_time_ns=selected_time,
        start_index=np.asarray(
            start,
            dtype=np.int64,
        ),
        stop_index_exclusive=np.asarray(
            stop,
            dtype=np.int64,
        ),
        horizon_min=np.asarray(
            horizon,
            dtype=np.int64,
        ),
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
        default=Path(
            r"D:\project\WindPredict_SaildroneData"
        ),
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--physics-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--modern-predictions-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        choices=[
            1,
            2,
            3,
            5,
            10,
        ],
        default=10,
    )

    parser.add_argument(
        "--window-hours",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--window-mode",
        choices=[
            "advantage",
            "variability",
            "fixed",
        ],
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
    )

    args = parser.parse_args()

    root = args.root.resolve()

    dataset_dir = (
        args.dataset_dir.resolve()
        if args.dataset_dir is not None
        else root
        / "data"
        / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
    )

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

    # THIS is the exact default 09C output path.
    modern_predictions_dir = (
        args.modern_predictions_dir.resolve()
        if args.modern_predictions_dir is not None
        else dataset_dir
        / "09C_modern_baseline_outer_validation_v0_1"
        / "predictions"
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root
        / "figures"
        / "final_paper"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    load_sci_style(
        root
    )

    print("=" * 100)
    print("FIG. 3 — MODERN MODEL APPARENT-WIND U/V TIME SERIES")
    print("=" * 100)
    print(f"dataset_dir            : {dataset_dir}")
    print(f"physics_dir            : {physics_dir}")
    print(f"modern_predictions_dir : {modern_predictions_dir}")
    print(f"output_dir             : {output_dir}")
    print(f"horizon                : {args.horizon} min")
    print(f"window_hours           : {args.window_hours:g}")
    print(f"window_mode            : {args.window_mode}")
    print("=" * 100)

    if not modern_predictions_dir.exists():
        raise FileNotFoundError(
            "\n09C prediction directory does not exist:\n"
            f"{modern_predictions_dir}\n\n"
            "The uploaded 09C script creates this directory automatically and "
            "saves predictions unless --no-save-predictions was supplied."
        )

    # ------------------------------------------------------------------
    # Physics-Compact
    # ------------------------------------------------------------------
    pc_truth, pc_pred_joint = load_physicscompact(
        physics_dir
    )

    h_idx = int(
        np.where(
            HORIZONS == args.horizon
        )[0][0]
    )

    pc_app = apparent_from_joint(
        pc_pred_joint
    )[
        :,
        h_idx,
        :,
    ]

    predictions: Dict[str, np.ndarray] = {
        "Physics-Compact": pc_app,
    }

    # ------------------------------------------------------------------
    # Modern models
    # ------------------------------------------------------------------
    observed_app = None
    target_time_ns = None

    for model_name in MODERN_MODELS:
        (
            truth_joint,
            pred_joint,
            apparent_ref,
            target_times,
        ) = load_modern_model(
            modern_predictions_dir,
            model_name,
        )

        if truth_joint.shape != pc_truth.shape:
            raise RuntimeError(
                f"{model_name}: truth shape {truth_joint.shape} "
                f"!= Physics-Compact truth shape {pc_truth.shape}"
            )

        if not np.allclose(
            truth_joint,
            pc_truth,
            atol=1e-5,
            rtol=0,
        ):
            max_diff = float(
                np.max(
                    np.abs(
                        truth_joint.astype(np.float64)
                        - pc_truth.astype(np.float64)
                    )
                )
            )

            raise RuntimeError(
                f"{model_name}: truth does not align with Physics-Compact. "
                f"max absolute difference={max_diff:.3e}"
            )

        model_app = apparent_from_joint(
            pred_joint
        )[
            :,
            h_idx,
            :,
        ]

        predictions[
            model_name
        ] = model_app

        current_obs = apparent_ref[
            :,
            h_idx,
            :,
        ].astype(
            np.float64
        )

        if observed_app is None:
            observed_app = current_obs
        elif not np.allclose(
            current_obs,
            observed_app,
            atol=1e-6,
            rtol=0,
        ):
            raise RuntimeError(
                f"{model_name}: apparent_earth_raw reference differs "
                "from the other modern models."
            )

        current_time = target_times[
            :,
            h_idx,
        ]

        if target_time_ns is None:
            target_time_ns = current_time
        elif not np.array_equal(
            current_time,
            target_time_ns,
        ):
            raise RuntimeError(
                f"{model_name}: target timestamps differ across models."
            )

    assert observed_app is not None
    assert target_time_ns is not None

    # Audit that the stored 09C apparent-wind reference is consistent with
    # the joint truth reconstruction.
    reconstructed_truth_app = apparent_from_joint(
        pc_truth
    )[
        :,
        h_idx,
        :,
    ]

    reference_rmse = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (
                        observed_app
                        - reconstructed_truth_app
                    )
                    ** 2,
                    axis=1,
                )
            )
        )
    )

    print("")
    print(
        "[AUDIT] 09C apparent reference vs reconstructed joint truth "
        f"RMSE = {reference_rmse:.10e} m/s"
    )

    if reference_rmse > 1e-5:
        raise RuntimeError(
            "09C apparent_earth_raw is inconsistent with truth_raw."
        )

    # ------------------------------------------------------------------
    # Select long plotting window
    # ------------------------------------------------------------------
    window = int(
        round(
            args.window_hours
            * 60.0
        )
    )

    if window < 60:
        raise ValueError(
            "Use at least a 1-h interval for trend visualization."
        )

    start, diagnostic = select_window(
        observed_uv=observed_app,
        predictions=predictions,
        window=window,
        mode=args.window_mode,
        fixed_start=args.start_index,
        variability_percentile=args.variability_percentile,
    )

    stop = start + window

    diagnostic.to_csv(
        output_dir
        / "Fig03_modern_window_selection_diagnostics.csv",
        index=False,
    )

    print("")
    print("=" * 100)
    print("SELECTED WINDOW")
    print("=" * 100)
    print(f"start index     : {start}")
    print(f"stop index      : {stop}")
    print(f"duration        : {window / 60.0:.3f} h")
    print(f"selection mode  : {args.window_mode}")

    if args.window_mode == "advantage":
        selected_row = diagnostic.loc[
            diagnostic["selected"] == 1
        ].iloc[0]

        print(
            "Physics-Compact skill vs mean modern models "
            f"= {100.0 * selected_row['PhysicsCompact_skill_fraction']:.3f}%"
        )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    metrics = plot_window(
        output_dir=output_dir,
        observed_uv=observed_app,
        predictions=predictions,
        target_time_ns_h=target_time_ns,
        start=start,
        window=window,
        horizon=args.horizon,
        selection_mode=args.window_mode,
    )

    print("")
    print("=" * 100)
    print("METRICS IN THE EXACT PLOTTED WINDOW")
    print("=" * 100)

    print(
        metrics[
            [
                "model",
                "Uapp_RMSE_mps",
                "Vapp_RMSE_mps",
                "AppVector_RMSE_mps",
            ]
        ].to_string(
            index=False
        )
    )

    manifest = {
        "source": (
            "09C_modern_baseline_outer_validation_v0_1/predictions"
        ),
        "modern_models": MODERN_MODELS,
        "seeds": SEEDS,
        "horizon_min": int(
            args.horizon
        ),
        "window_hours": float(
            args.window_hours
        ),
        "window_mode": args.window_mode,
        "start_index": int(
            start
        ),
        "stop_index_exclusive": int(
            stop
        ),
        "observed_apparent_definition": (
            "09C apparent_earth_raw; audited against truth_raw wind minus vessel velocity"
        ),
        "prediction_apparent_definition": (
            "pred_raw wind U/V minus pred_raw vessel east/north velocity"
        ),
        "curves": [
            "Observed",
            "Physics-Compact",
            *MODERN_MODELS,
        ],
    }

    with open(
        output_dir
        / "Fig03_modern_window_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print("")
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
