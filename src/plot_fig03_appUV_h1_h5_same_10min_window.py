# -*- coding: utf-8 -*-
r"""
plot_fig03_appUV_h1_h5_same_10min_window.py

Purpose
-------
Plot the 1-min or 5-min apparent-wind U/V time series using EXACTLY the
same physical-time interval as the already generated 10-min Fig. 3.

This script deliberately does NOT re-select a window at the new horizon.
Instead, it reads the exact 10-min target timestamps saved by
plot_fig03_appUV_modern_models_v2.py:

    <output_dir>/Fig03_selected_interval_traceability.npz

and finds the identical target-time interval in the requested 1-min or
5-min horizon. This is stricter than reusing the same sample start index,
because target timestamps shift with forecast horizon.

Curves, colors, line widths, figure size, axes, legend, and sci_plot_style
are kept identical to plot_fig03_appUV_modern_models_v2.py.

Examples (PowerShell)
---------------------
# 1-min horizon
python "D:\project\WindPredict_SaildroneData\src\plot_fig03_appUV_h1_h5_same_10min_window.py" `
  --root "D:\project\WindPredict_SaildroneData" `
  --horizon 1

# 5-min horizon
python "D:\project\WindPredict_SaildroneData\src\plot_fig03_appUV_h1_h5_same_10min_window.py" `
  --root "D:\project\WindPredict_SaildroneData" `
  --horizon 5

IMPORTANT
---------
Run the original 10-min plotting script first so that
Fig03_selected_interval_traceability.npz corresponds to the final 10-min
figure/window that is already used in the manuscript.
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
import matplotlib.pyplot as plt


# =============================================================================
# Load the user's original plotting module so data loading, physics,
# colors, seed handling, and scientific style remain exactly consistent.
# =============================================================================

def load_original_module(root: Path):
    candidates = [
        root / "src" / "plot_fig03_appUV_modern_models_v2.py",
        root / "src" / "plot_fig03_appUV_modern_models_v2(1).py",
    ]

    source = None
    for p in candidates:
        if p.exists():
            source = p
            break

    if source is None:
        raise FileNotFoundError(
            "Cannot find the original plotting script. Expected one of:\n"
            + "\n".join(str(p) for p in candidates)
        )

    spec = importlib.util.spec_from_file_location(
        "fig03_original_v2",
        str(source),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import original plotting script: {source}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    print(f"[SOURCE] original plotting module: {source}")
    return mod


# =============================================================================
# Exact 10-min time-window matching
# =============================================================================

def load_reference_10min_window(reference_file: Path) -> np.ndarray:
    if not reference_file.exists():
        raise FileNotFoundError(
            "The 10-min traceability file does not exist:\n"
            f"{reference_file}\n\n"
            "Run the original 10-min plotting script first. It automatically "
            "writes Fig03_selected_interval_traceability.npz."
        )

    with np.load(reference_file, allow_pickle=False) as z:
        required = ["target_time_ns", "horizon_min"]
        missing = [k for k in required if k not in z]
        if missing:
            raise KeyError(
                f"{reference_file}: missing keys {missing}; available={list(z.files)}"
            )

        ref_times = np.asarray(z["target_time_ns"]).reshape(-1)
        ref_horizon = int(np.asarray(z["horizon_min"]).reshape(()))

    if ref_horizon != 10:
        raise RuntimeError(
            f"Reference traceability file has horizon={ref_horizon} min, not 10 min.\n"
            "Please regenerate the final 10-min figure first or provide the correct "
            "file with --reference-traceability."
        )

    if ref_times.size < 2:
        raise RuntimeError("10-min reference window contains too few timestamps.")

    if not np.all(np.diff(ref_times.astype(np.int64)) > 0):
        raise RuntimeError("10-min reference timestamps are not strictly increasing.")

    return ref_times


def match_exact_time_window(
    target_time_ns_h: np.ndarray,
    reference_times: np.ndarray,
) -> Tuple[int, int]:
    current = np.asarray(target_time_ns_h).reshape(-1)
    reference = np.asarray(reference_times).reshape(-1)

    # Exact match on the first physical target timestamp.
    hits = np.flatnonzero(current == reference[0])
    if hits.size != 1:
        raise RuntimeError(
            "Could not uniquely locate the first 10-min reference timestamp "
            f"in the requested horizon. matches={hits.size}"
        )

    start = int(hits[0])
    stop = start + int(reference.size)

    if stop > current.size:
        raise RuntimeError(
            f"Matched interval [{start}:{stop}] exceeds current horizon length {current.size}."
        )

    selected = current[start:stop]
    if not np.array_equal(selected, reference):
        # Give a useful diagnostic rather than silently using a shifted index.
        mismatch = np.flatnonzero(selected != reference)
        first = int(mismatch[0]) if mismatch.size else -1
        raise RuntimeError(
            "The requested horizon does not reproduce the exact 10-min physical-time "
            "window.\n"
            f"First mismatch position={first}.\n"
            "This script intentionally refuses to substitute a merely equal-length or "
            "same-index interval."
        )

    return start, stop


# =============================================================================
# Plot: copied from the original v2 script except for horizon-specific filenames.
# =============================================================================

def plot_exact_window(
    mod,
    output_dir: Path,
    observed_uv: np.ndarray,
    predictions: Dict[str, np.ndarray],
    target_time_ns_h: np.ndarray,
    start: int,
    stop: int,
    horizon: int,
) -> pd.DataFrame:
    window = stop - start
    obs = observed_uv[start:stop]

    # IDENTICAL x-axis definition to the original 10-min plot.
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
        color=mod.COLORS["Observed"],
        linewidth=observed_lw,
        linestyle="-",
        label="Observed",
        zorder=6,
    )

    for name in model_order:
        pred = predictions[name][start:stop]
        ax.plot(
            t_hours,
            pred[:, 0],
            color=mod.COLORS[name],
            linewidth=model_lw,
            linestyle="-",
            label=name,
            alpha=0.97,
            zorder=4,
        )

    ax.set_ylabel(r"Apparent-wind $U_{\mathrm{app}}$ (m s$^{-1}$)")
    mod.panel_label(ax, "(a)")
    mod.clean_axis(ax)

    # ------------------------------------------------------------------
    # V_app
    # ------------------------------------------------------------------
    ax = axes[1]
    ax.plot(
        t_hours,
        obs[:, 1],
        color=mod.COLORS["Observed"],
        linewidth=observed_lw,
        linestyle="-",
        label="Observed",
        zorder=6,
    )

    for name in model_order:
        pred = predictions[name][start:stop]
        ax.plot(
            t_hours,
            pred[:, 1],
            color=mod.COLORS[name],
            linewidth=model_lw,
            linestyle="-",
            label=name,
            alpha=0.97,
            zorder=4,
        )

    ax.set_xlabel("Time within selected validation interval (h)")
    ax.set_ylabel(r"Apparent-wind $V_{\mathrm{app}}$ (m s$^{-1}$)")
    mod.panel_label(ax, "(b)")
    mod.clean_axis(ax)

    # ------------------------------------------------------------------
    # Shared legend -- IDENTICAL to original
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

    tag = f"h{horizon:02d}min_same10minwindow"
    mod.save_both(
        fig,
        output_dir / f"Fig03_apparent_UV_modern_models_{tag}",
    )

    # ------------------------------------------------------------------
    # Exact metrics in the plotted interval
    # ------------------------------------------------------------------
    rows = []
    for name, pred_all in predictions.items():
        pred = pred_all[start:stop]
        u_rmse, v_rmse, vector_rmse = mod.component_rmse(obs, pred)
        rows.append(
            {
                "model": name,
                "horizon_min": horizon,
                "selection_mode": "exact_physical_time_match_to_10min",
                "start_index_for_this_horizon": start,
                "stop_index_exclusive_for_this_horizon": stop,
                "window_hours": window / 60.0,
                "Uapp_RMSE_mps": u_rmse,
                "Vapp_RMSE_mps": v_rmse,
                "AppVector_RMSE_mps": vector_rmse,
            }
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(
        output_dir / f"Fig03_apparent_UV_modern_models_{tag}_window_metrics.csv",
        index=False,
    )

    selected_time = np.asarray(target_time_ns_h)[start:stop]
    np.savez_compressed(
        output_dir / f"Fig03_selected_interval_traceability_{tag}.npz",
        target_time_ns=selected_time,
        start_index=np.asarray(start, dtype=np.int64),
        stop_index_exclusive=np.asarray(stop, dtype=np.int64),
        horizon_min=np.asarray(horizon, dtype=np.int64),
        reference_horizon_min=np.asarray(10, dtype=np.int64),
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
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--physics-dir", type=Path, default=None)
    parser.add_argument("--modern-predictions-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--reference-traceability",
        type=Path,
        default=None,
        help=(
            "Traceability NPZ saved by the FINAL 10-min plot. Default: "
            "<output-dir>/Fig03_selected_interval_traceability.npz"
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=[1, 5],
        required=True,
    )

    args = parser.parse_args()
    root = args.root.resolve()

    mod = load_original_module(root)

    dataset_dir = (
        args.dataset_dir.resolve()
        if args.dataset_dir is not None
        else root / "data" / "forecasting" / "SD1090_TPOS2024_JointForecasting_v0_1"
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
    modern_predictions_dir = (
        args.modern_predictions_dir.resolve()
        if args.modern_predictions_dir is not None
        else dataset_dir / "09C_modern_baseline_outer_validation_v0_1" / "predictions"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "figures" / "final_paper"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_file = (
        args.reference_traceability.resolve()
        if args.reference_traceability is not None
        else output_dir / "Fig03_selected_interval_traceability.npz"
    )

    # Use EXACTLY the same scientific plot style as the original script.
    mod.load_sci_style(root)

    print("=" * 100)
    print(f"FIG. 3 — {args.horizon}-MIN APPARENT-WIND U/V, SAME PHYSICAL WINDOW AS 10-MIN FIGURE")
    print("=" * 100)
    print(f"dataset_dir            : {dataset_dir}")
    print(f"physics_dir            : {physics_dir}")
    print(f"modern_predictions_dir : {modern_predictions_dir}")
    print(f"output_dir             : {output_dir}")
    print(f"reference_traceability : {reference_file}")
    print(f"horizon                : {args.horizon} min")
    print("=" * 100)

    if not modern_predictions_dir.exists():
        raise FileNotFoundError(f"09C prediction directory does not exist:\n{modern_predictions_dir}")

    reference_times = load_reference_10min_window(reference_file)

    # ------------------------------------------------------------------
    # Physics-Compact
    # ------------------------------------------------------------------
    pc_truth, pc_pred_joint = mod.load_physicscompact(physics_dir)
    h_idx = int(np.where(mod.HORIZONS == args.horizon)[0][0])

    pc_app = mod.apparent_from_joint(pc_pred_joint)[:, h_idx, :]
    predictions: Dict[str, np.ndarray] = {"Physics-Compact": pc_app}

    # ------------------------------------------------------------------
    # Modern models + observed apparent wind + target times
    # ------------------------------------------------------------------
    observed_app = None
    target_time_ns = None

    for model_name in mod.MODERN_MODELS:
        truth_joint, pred_joint, apparent_ref, target_times = mod.load_modern_model(
            modern_predictions_dir,
            model_name,
        )

        if truth_joint.shape != pc_truth.shape:
            raise RuntimeError(
                f"{model_name}: truth shape {truth_joint.shape} != "
                f"Physics-Compact truth shape {pc_truth.shape}"
            )

        if not np.allclose(truth_joint, pc_truth, atol=1e-5, rtol=0):
            max_diff = float(
                np.max(
                    np.abs(
                        truth_joint.astype(np.float64) - pc_truth.astype(np.float64)
                    )
                )
            )
            raise RuntimeError(
                f"{model_name}: truth does not align with Physics-Compact; "
                f"max absolute difference={max_diff:.3e}"
            )

        predictions[model_name] = mod.apparent_from_joint(pred_joint)[:, h_idx, :]

        current_obs = apparent_ref[:, h_idx, :].astype(np.float64)
        if observed_app is None:
            observed_app = current_obs
        elif not np.allclose(current_obs, observed_app, atol=1e-6, rtol=0):
            raise RuntimeError(
                f"{model_name}: apparent_earth_raw reference differs from other models."
            )

        current_time = target_times[:, h_idx]
        if target_time_ns is None:
            target_time_ns = current_time
        elif not np.array_equal(current_time, target_time_ns):
            raise RuntimeError(f"{model_name}: target timestamps differ across models.")

    assert observed_app is not None
    assert target_time_ns is not None

    # Audit stored apparent wind against joint truth.
    reconstructed_truth_app = mod.apparent_from_joint(pc_truth)[:, h_idx, :]
    reference_rmse = float(
        np.sqrt(
            np.mean(
                np.sum((observed_app - reconstructed_truth_app) ** 2, axis=1)
            )
        )
    )
    print(
        "[AUDIT] apparent reference vs reconstructed joint truth "
        f"RMSE = {reference_rmse:.10e} m/s"
    )
    if reference_rmse > 1e-5:
        raise RuntimeError("apparent_earth_raw is inconsistent with truth_raw.")

    # ------------------------------------------------------------------
    # Match the EXACT target-time interval of the final 10-min figure.
    # ------------------------------------------------------------------
    start, stop = match_exact_time_window(target_time_ns, reference_times)
    window = stop - start

    print("")
    print("=" * 100)
    print("EXACT 10-MIN PHYSICAL-TIME WINDOW MATCHED")
    print("=" * 100)
    print(f"start index for {args.horizon}-min horizon : {start}")
    print(f"stop index exclusive                     : {stop}")
    print(f"samples                                  : {window}")
    print(f"duration                                 : {window / 60.0:.3f} h")
    print(f"first target timestamp (ns)              : {int(reference_times[0])}")
    print(f"last target timestamp (ns)               : {int(reference_times[-1])}")
    print("=" * 100)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    metrics = plot_exact_window(
        mod=mod,
        output_dir=output_dir,
        observed_uv=observed_app,
        predictions=predictions,
        target_time_ns_h=target_time_ns,
        start=start,
        stop=stop,
        horizon=args.horizon,
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

    tag = f"h{args.horizon:02d}min_same10minwindow"
    manifest = {
        "source": "09C_modern_baseline_outer_validation_v0_1/predictions",
        "modern_models": list(mod.MODERN_MODELS),
        "seeds": list(mod.SEEDS),
        "horizon_min": int(args.horizon),
        "reference_horizon_min": 10,
        "reference_traceability": str(reference_file),
        "selection_mode": "exact_physical_time_match_to_10min",
        "start_index_for_this_horizon": int(start),
        "stop_index_exclusive_for_this_horizon": int(stop),
        "window_hours": float(window / 60.0),
        "first_target_time_ns": int(reference_times[0]),
        "last_target_time_ns": int(reference_times[-1]),
        "curves": ["Observed", "Physics-Compact", *list(mod.MODERN_MODELS)],
    }

    with open(
        output_dir / f"Fig03_modern_window_manifest_{tag}.json",
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
