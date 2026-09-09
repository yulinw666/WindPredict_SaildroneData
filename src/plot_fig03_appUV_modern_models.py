# -*- coding: utf-8 -*-
r"""
plot_fig03_appUV_modern_models.py

Final qualitative comparison for SD1090 validation:
Observed vs Physics-Compact vs DLinear vs TimeMixer vs iTransformer vs PatchTST.

Panels
------
(a) Earth-frame apparent-wind U component:
    U_app = U_wind - V_ship,east
(b) Earth-frame apparent-wind V component:
    V_app = V_wind - V_ship,north

Default plot
------------
- 10-min forecast horizon
- 8-h continuous validation window
- "advantage" window selection:
    among sufficiently dynamic windows, select the interval where
    Physics-Compact has the largest relative apparent-wind vector-RMSE
    advantage over the mean of the four modern baselines.
- all curves are thin solid lines
- paper palette:
    Observed         black
    Physics-Compact blue
    DLinear         orange
    TimeMixer       green
    iTransformer    red
    PatchTST        purple

IMPORTANT
---------
This script does NOT retrain any model.

It searches existing per-sample VALIDATION prediction NPZ files for the four
modern models. A valid prediction file must contain physical-unit arrays with
shape (N,5,6), typically keys such as:
    y_raw / y_true
    y_pred / prediction / pred

If modern per-sample predictions were not saved by the previous refit scripts,
the script stops and reports what it found. In that case, do not reconstruct
time series from per-horizon RMSE CSVs; per-sample predictions are required.

Example
-------
python "D:\project\WindPredict_SaildroneData\src\plot_fig03_appUV_modern_models.py" `
  --root "D:\project\WindPredict_SaildroneData" `
  --horizon 10 `
  --window-hours 8 `
  --window-mode advantage

For a non-performance-selected manuscript interval:
python "D:\project\WindPredict_SaildroneData\src\plot_fig03_appUV_modern_models.py" `
  --root "D:\project\WindPredict_SaildroneData" `
  --horizon 10 `
  --window-hours 8 `
  --window-mode variability
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
            "sci_plot_style_fig03_modern",
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
            "legend.fontsize": 7.2,
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
# Generic prediction loading
# =============================================================================

def norm_text(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def app_uv_from_joint(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)

    if y.ndim != 3 or y.shape[1:] != (5, 6):
        raise ValueError(
            f"Expected joint target/prediction shape (N,5,6), got {y.shape}"
        )

    return y[..., 0:2] - y[..., 2:4]


def find_array_key(
    z,
    aliases: List[str],
    required_shape_tail=(5, 6),
) -> Optional[str]:
    normalized = {norm_text(k): k for k in z.files}

    # Alias-first.
    for alias in aliases:
        na = norm_text(alias)
        if na in normalized:
            k = normalized[na]
            arr = np.asarray(z[k])
            if arr.ndim == 3 and arr.shape[1:] == required_shape_tail:
                return k

    # Shape fallback, but only for arrays whose names contain useful words.
    useful = [
        "pred",
        "truth",
        "true",
        "target",
        "yraw",
        "y",
    ]

    for k in z.files:
        arr = np.asarray(z[k])
        if arr.ndim == 3 and arr.shape[1:] == required_shape_tail:
            nk = norm_text(k)
            if any(word in nk for word in useful):
                return k

    return None


def extract_truth_prediction_from_npz(
    path: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray, str, str]]:
    """
    Returns physical-unit y_true/y_pred arrays when identifiable.
    Does not accept standardized predictions with obvious *_z names.
    """
    try:
        with np.load(path, allow_pickle=False) as z:
            truth_key = find_array_key(
                z,
                [
                    "y_raw",
                    "y_true",
                    "truth",
                    "target_raw",
                    "targets_raw",
                    "target",
                    "targets",
                ],
            )
            pred_key = find_array_key(
                z,
                [
                    "y_pred",
                    "prediction",
                    "predictions",
                    "pred",
                    "yhat",
                    "y_hat",
                ],
            )

            if truth_key is None or pred_key is None:
                return None

            # Reject obvious standardized-space prediction arrays.
            if norm_text(pred_key).endswith("z") or "standard" in norm_text(pred_key):
                return None

            y_true = np.asarray(z[truth_key], dtype=np.float32)
            y_pred = np.asarray(z[pred_key], dtype=np.float32)

            if y_true.shape != y_pred.shape:
                return None

            if y_true.ndim != 3 or y_true.shape[1:] != (5, 6):
                return None

            if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
                return None

            return y_true, y_pred, truth_key, pred_key

    except Exception:
        return None


def score_candidate_path(
    path: Path,
    model_name: str,
) -> int:
    s = norm_text(str(path))
    model = norm_text(model_name)

    score = 0

    if model in s:
        score += 20
    if "validation" in s or "val" in s:
        score += 10
    if "prediction" in s or "pred" in s:
        score += 8
    if "09c" in s:
        score += 4
    if "modern" in s:
        score += 3
    if "external" in s:
        score -= 8
    if "sd1033" in s:
        score -= 8
    if "test" in s:
        score -= 5

    return score


def discover_modern_prediction_files(
    search_root: Path,
    model_name: str,
) -> List[Path]:
    """
    Recursively discover valid per-sample validation NPZ prediction files.

    The model name MUST appear somewhere in the full path, preventing
    accidental cross-model loading.
    """
    if not search_root.exists():
        return []

    model_token = norm_text(model_name)
    found = []

    for p in search_root.rglob("*.npz"):
        full_norm = norm_text(str(p))

        if model_token not in full_norm:
            continue

        # Strongly prefer validation files and reject obvious external files.
        if "sd1033" in full_norm or "external" in full_norm:
            continue

        if "test" in full_norm and "validation" not in full_norm:
            continue

        parsed = extract_truth_prediction_from_npz(p)
        if parsed is None:
            continue

        found.append(p)

    found.sort(
        key=lambda p: (-score_candidate_path(p, model_name), str(p))
    )

    return found


def load_model_prediction_ensemble(
    search_root: Path,
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
    candidates = discover_modern_prediction_files(
        search_root=search_root,
        model_name=model_name,
    )

    if not candidates:
        raise FileNotFoundError(
            "\n"
            + "=" * 100
            + f"\nNO PER-SAMPLE VALIDATION PREDICTIONS FOUND FOR {model_name}\n"
            + "=" * 100
            + "\nThe current plot requires arrays with shape (N,5,6):\n"
            + "  y_raw / y_true\n"
            + "  y_pred / prediction\n\n"
            + f"Searched recursively under:\n  {search_root}\n\n"
            + "Per-horizon RMSE CSV files are NOT sufficient for a time-series plot.\n"
            + "If your modern refit script saved checkpoints but not predictions,\n"
            + "run inference once and save validation predictions first.\n"
        )

    # Use the best-scored family. Usually this means one file per seed.
    # Restrict to candidates sharing the same parent when possible.
    best_parent = candidates[0].parent
    same_parent = [p for p in candidates if p.parent == best_parent]

    chosen = same_parent if same_parent else [candidates[0]]

    truths = []
    preds = []

    for p in chosen:
        parsed = extract_truth_prediction_from_npz(p)
        if parsed is None:
            continue

        y_true, y_pred, truth_key, pred_key = parsed

        truths.append(y_true)
        preds.append(y_pred)

        print(
            f"[LOAD] {model_name:12s} <- {p}\n"
            f"       truth={truth_key}, pred={pred_key}, shape={y_pred.shape}"
        )

    if not preds:
        raise RuntimeError(
            f"Candidates were found for {model_name}, but none could be loaded."
        )

    y0 = truths[0]

    for i, y in enumerate(truths[1:], start=1):
        if y.shape != y0.shape or not np.allclose(y, y0, atol=1e-6, rtol=0):
            raise RuntimeError(
                f"{model_name}: discovered prediction files are not aligned "
                "to the same validation truth."
            )

    pred_mean = np.mean(
        np.stack(preds, axis=0),
        axis=0,
    ).astype(np.float32)

    return y0.astype(np.float32), pred_mean, chosen


def load_physics_predictions(
    physics_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    truths = []
    preds = []

    for seed in SEEDS:
        p = physics_dir / f"15E_VH_seed_{seed}_validation_predictions.npz"

        if not p.exists():
            raise FileNotFoundError(p)

        with np.load(p, allow_pickle=False) as z:
            required = ["y_raw", "frozen_truewind", "joint_vessel_hdg"]
            missing = [k for k in required if k not in z]

            if missing:
                raise KeyError(f"{p}: missing {missing}")

            y_true = np.asarray(z["y_raw"], dtype=np.float32)
            wind = np.asarray(z["frozen_truewind"], dtype=np.float32)
            motion = np.asarray(z["joint_vessel_hdg"], dtype=np.float32)

        y_pred = np.empty_like(y_true)
        y_pred[..., 0:2] = wind
        y_pred[..., 2:6] = motion

        truths.append(y_true)
        preds.append(y_pred)

    y0 = truths[0]

    for y in truths[1:]:
        if not np.allclose(y, y0, atol=1e-7, rtol=0):
            raise RuntimeError(
                "Physics-Compact seed files do not share identical y_raw."
            )

    pred_mean = np.mean(
        np.stack(preds, axis=0),
        axis=0,
    ).astype(np.float32)

    return y0, pred_mean


# =============================================================================
# Window selection
# =============================================================================

def sliding_sum(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    return cs[window:] - cs[:-window]


def sliding_mean(x: np.ndarray, window: int) -> np.ndarray:
    return sliding_sum(x, window) / float(window)


def observed_variability_score(
    obs_uv: np.ndarray,
    window: int,
) -> np.ndarray:
    u = obs_uv[:, 0]
    v = obs_uv[:, 1]

    mu_u = sliding_mean(u, window)
    mu_v = sliding_mean(v, window)

    var_u = sliding_mean(u * u, window) - mu_u * mu_u
    var_v = sliding_mean(v * v, window) - mu_v * mu_v

    return np.sqrt(
        np.maximum(var_u + var_v, 0.0)
    )


def vector_rmse_sliding(
    obs_uv: np.ndarray,
    pred_uv: np.ndarray,
    window: int,
) -> np.ndarray:
    err = (
        np.asarray(pred_uv, dtype=np.float64)
        - np.asarray(obs_uv, dtype=np.float64)
    )

    sq = np.sum(err * err, axis=1)

    return np.sqrt(
        sliding_mean(sq, window)
    )


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
            f"window={window} exceeds available validation samples={n}"
        )

    variability = observed_variability_score(
        obs_uv,
        window,
    )

    diagnostic = pd.DataFrame(
        {
            "start_index": np.arange(
                len(variability),
                dtype=int,
            ),
            "variability": variability,
        }
    )

    rmse = {}

    for name, pred in predictions.items():
        r = vector_rmse_sliding(
            obs_uv,
            pred,
            window,
        )
        rmse[name] = r
        diagnostic[f"rmse_{name}"] = r

    if mode == "fixed":
        if start_index < 0 or start_index + window > n:
            raise ValueError(
                f"Invalid fixed start_index={start_index}"
            )
        selected = int(start_index)

    elif mode == "variability":
        selected = int(
            np.nanargmax(variability)
        )

    elif mode == "advantage":
        pc = rmse["Physics-Compact"]

        modern_names = [
            n
            for n in MODERN_MODELS
            if n in rmse
        ]

        if not modern_names:
            raise RuntimeError(
                "No modern models available for advantage selection."
            )

        modern_mean = np.mean(
            np.stack(
                [rmse[name] for name in modern_names],
                axis=0,
            ),
            axis=0,
        )

        skill = (
            modern_mean - pc
        ) / np.maximum(modern_mean, 1e-12)

        # Avoid choosing an almost-flat easy interval.
        threshold = np.nanpercentile(
            variability,
            variability_percentile,
        )
        eligible = variability >= threshold

        score = np.full_like(
            skill,
            -np.inf,
        )
        score[eligible] = skill[eligible]

        selected = int(
            np.nanargmax(score)
        )

        diagnostic["modern_mean_rmse"] = modern_mean
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
# Metrics
# =============================================================================

def component_rmse(
    obs: np.ndarray,
    pred: np.ndarray,
) -> Tuple[float, float, float]:
    err = (
        np.asarray(pred, dtype=np.float64)
        - np.asarray(obs, dtype=np.float64)
    )

    u = float(
        np.sqrt(
            np.mean(err[:, 0] ** 2)
        )
    )
    v = float(
        np.sqrt(
            np.mean(err[:, 1] ** 2)
        )
    )
    vec = float(
        np.sqrt(
            np.mean(
                np.sum(err * err, axis=1)
            )
        )
    )

    return u, v, vec


# =============================================================================
# Plot
# =============================================================================

def plot_window(
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

    t = np.arange(window, dtype=float) / 60.0

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.15, 4.20),
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

    # Thin raw 1-min curves.
    observed_lw = 1.00
    model_lw = 0.78

    plot_order = [
        "Physics-Compact",
        "DLinear",
        "TimeMixer",
        "iTransformer",
        "PatchTST",
    ]

    # Panel (a): U_app
    ax = axes[0]
    ax.plot(
        t,
        obs[:, 0],
        color=COLORS["Observed"],
        linewidth=observed_lw,
        linestyle="-",
        label="Observed",
        zorder=6,
    )

    for name in plot_order:
        pred = predictions[name][start:stop]

        ax.plot(
            t,
            pred[:, 0],
            color=COLORS[name],
            linewidth=model_lw,
            linestyle="-",
            label=name,
            alpha=0.96,
            zorder=4,
        )

    ax.set_ylabel(
        r"Apparent-wind $U_{\rm app}$ (m s$^{-1}$)"
    )
    panel_label(ax, "(a)")
    clean_axis(ax)

    # Panel (b): V_app
    ax = axes[1]
    ax.plot(
        t,
        obs[:, 1],
        color=COLORS["Observed"],
        linewidth=observed_lw,
        linestyle="-",
        label="Observed",
        zorder=6,
    )

    for name in plot_order:
        pred = predictions[name][start:stop]

        ax.plot(
            t,
            pred[:, 1],
            color=COLORS[name],
            linewidth=model_lw,
            linestyle="-",
            label=name,
            alpha=0.96,
            zorder=4,
        )

    ax.set_xlabel(
        "Time within selected validation interval (h)"
    )
    ax.set_ylabel(
        r"Apparent-wind $V_{\rm app}$ (m s$^{-1}$)"
    )
    panel_label(ax, "(b)")
    clean_axis(ax)

    # Shared legend.
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=6,
        frameon=False,
        fontsize=6.8,
        columnspacing=1.0,
        handlelength=2.0,
        handletextpad=0.4,
    )

    axes[1].text(
        0.995,
        0.022,
        (
            f"{horizon}-min horizon; "
            f"{window / 60.0:g}-h window; "
            f"selection={selection_mode}"
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
    )

    save_both(
        fig,
        output_dir / "Fig03_apparent_UV_modern_models",
    )

    # Exact metrics for plotted interval.
    rows = []

    for name, pred_all in predictions.items():
        pred = pred_all[start:stop]

        u_rmse, v_rmse, vec_rmse = component_rmse(
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
                "AppVector_RMSE_mps": vec_rmse,
            }
        )

    metrics = pd.DataFrame(rows)

    metrics.to_csv(
        output_dir
        / "Fig03_apparent_UV_modern_models_window_metrics.csv",
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
        default=Path(
            r"D:\project\WindPredict_SaildroneData"
        ),
    )
    parser.add_argument(
        "--physics-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--modern-search-root",
        type=Path,
        default=None,
        help=(
            "Root directory searched recursively for modern-model "
            "per-sample validation prediction NPZ files."
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

    modern_search_root = (
        args.modern_search_root.resolve()
        if args.modern_search_root is not None
        else root
        / "data"
        / "forecasting"
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

    load_sci_style(root)

    print("=" * 100)
    print("FIG. 3 — MODERN MODEL APPARENT-WIND U/V TIME SERIES")
    print("=" * 100)
    print(f"physics_dir       : {physics_dir}")
    print(f"modern_search_root: {modern_search_root}")
    print(f"output_dir        : {output_dir}")
    print(f"horizon           : {args.horizon} min")
    print(f"window_hours      : {args.window_hours:g}")
    print(f"window_mode       : {args.window_mode}")
    print("=" * 100)

    # Physics-Compact.
    y_true, pc_joint = load_physics_predictions(
        physics_dir
    )

    h_idx = int(
        np.where(
            HORIZONS == args.horizon
        )[0][0]
    )

    obs_app = app_uv_from_joint(
        y_true
    )[:, h_idx, :]

    predictions: Dict[str, np.ndarray] = {
        "Physics-Compact": app_uv_from_joint(
            pc_joint
        )[:, h_idx, :]
    }

    discovery_manifest = {}

    # Modern models.
    for model_name in MODERN_MODELS:
        y_cmp, pred_cmp, paths = load_model_prediction_ensemble(
            modern_search_root,
            model_name,
        )

        if y_cmp.shape != y_true.shape:
            raise RuntimeError(
                f"{model_name}: truth shape {y_cmp.shape} does not match "
                f"Physics-Compact truth {y_true.shape}."
            )

        if not np.allclose(
            y_cmp,
            y_true,
            atol=1e-6,
            rtol=0,
        ):
            raise RuntimeError(
                f"{model_name}: validation truth arrays do not align with "
                "Physics-Compact. Do not plot misaligned time series."
            )

        predictions[model_name] = app_uv_from_joint(
            pred_cmp
        )[:, h_idx, :]

        discovery_manifest[model_name] = [
            str(p)
            for p in paths
        ]

    # Window selection.
    window = int(
        round(
            args.window_hours * 60.0
        )
    )

    if window < 60:
        raise ValueError(
            "For a trend figure, use at least a 1-h window."
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
        output_dir
        / "Fig03_modern_window_selection_diagnostics.csv",
        index=False,
    )

    with open(
        output_dir
        / "Fig03_modern_prediction_discovery.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            discovery_manifest,
            f,
            indent=2,
        )

    print("")
    print("=" * 100)
    print("SELECTED WINDOW")
    print("=" * 100)
    print(f"start index     : {start}")
    print(f"stop index      : {stop}")
    print(f"duration        : {window / 60.0:.3f} h")
    print(f"mode            : {args.window_mode}")

    if args.window_mode == "advantage":
        row = diagnostic.loc[
            diagnostic["selected"] == 1
        ].iloc[0]

        print(
            "Physics skill vs mean modern models: "
            f"{100.0 * row['physics_skill_fraction']:.3f}%"
        )

    metrics = plot_window(
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
        ].to_string(index=False)
    )

    print("")
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
