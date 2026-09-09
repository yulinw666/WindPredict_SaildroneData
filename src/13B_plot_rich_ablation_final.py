# -*- coding: utf-8 -*-
r"""
13B_plot_rich_ablation_final.py

Richer final ablation figure, using the SAME plotting format as
13A_fix_fig05_fig06_v6.py, but combining:

A) historical residual variants from the ORIGINAL source files discovered
   in 13A_fix_fig05_fig06_v6.py

B) the CURRENT FINAL branch ablations from the frozen alpha_M=1500 model:
   - Wind enhanced
   - Motion enhanced
   - Physics-Compact (proposed)

The code does NOT use fabricated fallback values. If a required source file
is missing, it stops and prints the exact expected path.

Historical source files
-----------------------
<dataset_dir>\confirm_and_compact_vessel_residual_v0_1\seed_summary.csv

<dataset_dir>\structured_residual_ablation_v0_1\
    structured_ablation_metrics_summary.csv

Current final sources
---------------------
<project_root>\data\forecasting\15E_VH_AlphaSearch_v0_1\
    full_retrain\alpha_1500\
    15E_VH_seed_<seed>_validation_predictions.npz

The final Dual-Ridge reference is rebuilt from:
    alpha_W = 0
    alpha_M = 1500

Figure style
------------
Exactly follows the user's old Fig05 format:
- broken x axis
- point + horizontal error bar
- proposed model = red
- every other model = blue
- Dual Ridge = blue dashed vertical line
- Times New Roman via sci_plot_style.py
- 600 dpi PNG + vector PDF

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\13B_plot_rich_ablation_final.py" `
  --project-root "D:\project\WindPredict_SaildroneData"

Output
------
<project_root>\figures\final_paper\
    Fig_rich_ablation_final.png
    Fig_rich_ablation_final.pdf
    Fig_rich_ablation_final_data.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.linear_model import Ridge


# =============================================================================
# Frozen protocol
# =============================================================================

DEFAULT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")

HORIZONS = np.asarray([1, 2, 3, 5, 10], dtype=int)
SEEDS = [500043, 501052, 502061, 503070, 504079]

ALPHA_W = 0.0
ALPHA_M = 1500.0


# =============================================================================
# Utilities — kept deliberately close to 13A
# =============================================================================

def norm_text(s) -> str:
    s = str(s).strip().lower()
    return re.sub(r"[\s_\-()/\[\],.:;]+", "", s)


def apply_style(root: Path):
    style_path = root / "src" / "sci_plot_style.py"

    if style_path.exists():
        try:
            spec = importlib.util.spec_from_file_location(
                "sci_plot_style_rich_final",
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
                        print(f"[STYLE] sci_plot_style.py:{name}{kwargs}")
                        return
                    except TypeError:
                        continue

        except Exception as exc:
            print(f"[STYLE WARNING] {type(exc).__name__}: {exc}")

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


def finish_axis(ax, grid=True):
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
    )

    if grid:
        ax.grid(
            True,
            alpha=0.20,
            linewidth=0.6,
        )


def save_figure(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)

    png = Path(
        os.path.abspath(
            os.path.normpath(str(base.with_suffix(".png")))
        )
    )
    pdf = Path(
        os.path.abspath(
            os.path.normpath(str(base.with_suffix(".pdf")))
        )
    )

    fig.savefig(
        str(png),
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        str(pdf),
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[SAVED] {png}")
    print(f"[SAVED] {pdf}")


# =============================================================================
# Historical ablation data — exact source paths from 13A
# =============================================================================

HISTORICAL_MAPPING = [
    (
        "Physics-Compact-Vessel-Residual",
        "Previous Physics-Compact",
    ),
    (
        "Physics-Vessel-Only-Residual",
        "Physics-vessel-only residual",
    ),
    (
        "Physics-Full-Residual",
        "Physics-full residual",
    ),
    (
        "Vessel-Only-Residual",
        "Vessel-only residual",
    ),
    (
        "Full-Residual",
        "Full residual",
    ),
    (
        "Compact-Vessel-Residual",
        "Compact-vessel residual",
    ),
]


def load_historical_ablation(
    dataset_dir: Path,
) -> pd.DataFrame:
    """
    Read the exact 07C source used by the old Fig05.

    Unlike 13A, there is NO fallback here.
    """
    p07c = (
        dataset_dir
        / "confirm_and_compact_vessel_residual_v0_1"
        / "seed_summary.csv"
    )

    if not p07c.exists():
        raise FileNotFoundError(
            "Historical 07C ablation source was not found:\n"
            f"{p07c}"
        )

    print(f"[READ] historical 07C seed summary:\n       {p07c}")

    df = pd.read_csv(p07c)

    required = {
        "model",
        "mean_AW_RMSE_mps",
        "std_AW_RMSE_mps",
    }
    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            f"{p07c}: missing required columns {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    if "split" in df.columns:
        df = df[
            df["split"]
            .astype(str)
            .str.lower()
            .eq("validation")
        ]

    rows = []

    for raw_name, label in HISTORICAL_MAPPING:
        q = df[
            df["model"].astype(str).eq(raw_name)
        ]

        if q.empty:
            raise RuntimeError(
                f"Historical model not found in 07C source: {raw_name}"
            )

        r = q.iloc[0]

        rows.append({
            "label": label,
            "mean": float(r["mean_AW_RMSE_mps"]),
            "std": float(r["std_AW_RMSE_mps"]),
            "group": "historical",
            "source_file": str(p07c),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Current final data
# =============================================================================

def load_primary_arrays(
    dataset_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    train_path = dataset_dir / "train.npz"
    val_path = dataset_dir / "validation.npz"

    if not train_path.exists():
        raise FileNotFoundError(train_path)

    if not val_path.exists():
        raise FileNotFoundError(val_path)

    with np.load(
        train_path,
        allow_pickle=False,
    ) as z:
        Xtr = np.asarray(
            z["X"],
            dtype=np.float32,
        )
        ytr = np.asarray(
            z["y_raw"],
            dtype=np.float32,
        )

    with np.load(
        val_path,
        allow_pickle=False,
    ) as z:
        Xva = np.asarray(
            z["X"],
            dtype=np.float32,
        )
        yva = np.asarray(
            z["y_raw"],
            dtype=np.float32,
        )

    if Xtr.shape[1:] != (60, 11):
        raise RuntimeError(
            f"Unexpected train X shape: {Xtr.shape}"
        )

    if Xva.shape[1:] != (60, 11):
        raise RuntimeError(
            f"Unexpected validation X shape: {Xva.shape}"
        )

    if ytr.shape[1:] != (5, 6):
        raise RuntimeError(
            f"Unexpected train y_raw shape: {ytr.shape}"
        )

    if yva.shape[1:] != (5, 6):
        raise RuntimeError(
            f"Unexpected validation y_raw shape: {yva.shape}"
        )

    return Xtr, ytr, Xva, yva


def rebuild_final_anchors(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    # Final TrueWind Ridge anchor: alpha_W = 0.
    Xw_tr = Xtr[:, :, 0:4].reshape(
        len(Xtr),
        -1,
    )
    Xw_va = Xva[:, :, 0:4].reshape(
        len(Xva),
        -1,
    )
    yw_tr = ytr[:, :, 0:2].reshape(
        len(ytr),
        -1,
    )

    wind_ridge = Ridge(
        alpha=ALPHA_W,
        fit_intercept=True,
    )
    wind_ridge.fit(
        Xw_tr,
        yw_tr,
    )

    ridge_wind = wind_ridge.predict(
        Xw_va
    ).reshape(
        -1,
        5,
        2,
    )

    # Final joint motion/HDG Ridge anchor: alpha_M = 1500.
    Xj_tr = Xtr.reshape(
        len(Xtr),
        -1,
    )
    Xj_va = Xva.reshape(
        len(Xva),
        -1,
    )
    yj_tr = ytr.reshape(
        len(ytr),
        -1,
    )

    joint_ridge = Ridge(
        alpha=ALPHA_M,
        fit_intercept=True,
    )
    joint_ridge.fit(
        Xj_tr,
        yj_tr,
    )

    ridge_joint = joint_ridge.predict(
        Xj_va
    ).reshape(
        -1,
        5,
        6,
    )

    ridge_motion = ridge_joint[:, :, 2:6].copy()

    # Normalize heading sin/cos.
    q = ridge_motion[:, :, 2:4]
    qn = np.sqrt(
        np.sum(
            q * q,
            axis=-1,
            keepdims=True,
        )
    )

    ridge_motion[:, :, 2:4] = (
        q
        / np.maximum(
            qn,
            1e-12,
        )
    )

    return (
        ridge_wind.astype(np.float32),
        ridge_motion.astype(np.float32),
    )


def load_final_seed_predictions(
    final_dir: Path,
):
    rows = []

    for seed in SEEDS:
        p = (
            final_dir
            / f"15E_VH_seed_{seed}_validation_predictions.npz"
        )

        if not p.exists():
            raise FileNotFoundError(
                "Final alpha_M=1500 validation prediction not found:\n"
                f"{p}"
            )

        with np.load(
            p,
            allow_pickle=False,
        ) as z:

            required = [
                "y_raw",
                "frozen_truewind",
                "joint_vessel_hdg",
            ]

            missing = [
                k
                for k in required
                if k not in z
            ]

            if missing:
                raise KeyError(
                    f"{p}: missing keys {missing}"
                )

            rows.append({
                "seed": seed,
                "truth": np.asarray(
                    z["y_raw"],
                    dtype=np.float32,
                ),
                "wind": np.asarray(
                    z["frozen_truewind"],
                    dtype=np.float32,
                ),
                "motion": np.asarray(
                    z["joint_vessel_hdg"],
                    dtype=np.float32,
                ),
                "source_file": str(p),
            })

    truth0 = rows[0]["truth"]

    for d in rows[1:]:
        if not np.allclose(
            d["truth"],
            truth0,
            atol=1e-7,
            rtol=0,
        ):
            raise RuntimeError(
                "Final validation seed files do not share identical truth."
            )

    return rows


def apparent_earth(
    wind: np.ndarray,
    motion: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(
            wind,
            dtype=np.float64,
        )
        - np.asarray(
            motion,
            dtype=np.float64,
        )[:, :, 0:2]
    )


def horizon_vector_rmse_mean(
    truth: np.ndarray,
    pred: np.ndarray,
) -> float:
    """
    RMSE at each of the five forecast horizons, then arithmetic mean.
    This matches the paper's mean multi-horizon AppVector reporting.
    """
    vals = []

    for h in range(5):
        err = (
            np.asarray(
                pred[:, h, :],
                dtype=np.float64,
            )
            - np.asarray(
                truth[:, h, :],
                dtype=np.float64,
            )
        )

        vals.append(
            float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            err * err,
                            axis=1,
                        )
                    )
                )
            )
        )

    return float(
        np.mean(vals)
    )


def compute_current_variants(
    yva: np.ndarray,
    ridge_wind: np.ndarray,
    ridge_motion: np.ndarray,
    final_seeds,
    final_dir: Path,
) -> Tuple[pd.DataFrame, float]:

    truth_wind = yva[:, :, 0:2]
    truth_motion = yva[:, :, 2:6]

    truth_app = apparent_earth(
        truth_wind,
        truth_motion,
    )

    ridge_app = apparent_earth(
        ridge_wind,
        ridge_motion,
    )

    dual_ridge = horizon_vector_rmse_mean(
        truth_app,
        ridge_app,
    )

    values: Dict[str, List[float]] = {
        "Wind enhanced": [],
        "Motion enhanced": [],
        "Physics-Compact (proposed)": [],
    }

    for d in final_seeds:
        # Only TrueWind residual branch ON.
        pred_wind_only = apparent_earth(
            d["wind"],
            ridge_motion,
        )

        # Only Motion-HDG residual branch ON.
        pred_motion_only = apparent_earth(
            ridge_wind,
            d["motion"],
        )

        # Both final branches ON.
        pred_full = apparent_earth(
            d["wind"],
            d["motion"],
        )

        values["Wind enhanced"].append(
            horizon_vector_rmse_mean(
                truth_app,
                pred_wind_only,
            )
        )

        values["Motion enhanced"].append(
            horizon_vector_rmse_mean(
                truth_app,
                pred_motion_only,
            )
        )

        values["Physics-Compact (proposed)"].append(
            horizon_vector_rmse_mean(
                truth_app,
                pred_full,
            )
        )

    rows = []

    for label in [
        "Physics-Compact (proposed)",
        "Motion enhanced",
        "Wind enhanced",
    ]:
        arr = np.asarray(
            values[label],
            dtype=float,
        )

        rows.append({
            "label": label,
            "mean": float(
                np.mean(arr)
            ),
            "std": float(
                np.std(
                    arr,
                    ddof=1,
                )
            ),
            "group": "current-final",
            "source_file": str(final_dir),
        })

    return (
        pd.DataFrame(rows),
        dual_ridge,
    )


# =============================================================================
# Combine current + historical variants
# =============================================================================

PLOT_ORDER = [
    # Final model and its current branch ablations first.
    "Physics-Compact (proposed)",
    "Motion enhanced",
    "Wind enhanced",

    # Historical/earlier structural ablations from 07C.
    "Previous Physics-Compact",
    "Physics-vessel-only residual",
    "Physics-full residual",
    "Vessel-only residual",
    "Full residual",
    "Compact-vessel residual",
]


def combine_sources(
    current_df: pd.DataFrame,
    historical_df: pd.DataFrame,
) -> pd.DataFrame:

    df = pd.concat(
        [
            current_df,
            historical_df,
        ],
        ignore_index=True,
    )

    rank = {
        name: i
        for i, name in enumerate(
            PLOT_ORDER
        )
    }

    df["rank"] = df["label"].map(
        lambda x: rank.get(
            str(x),
            999,
        )
    )

    df = (
        df.sort_values(
            ["rank", "label"]
        )
        .drop(
            columns="rank"
        )
        .reset_index(
            drop=True
        )
    )

    return df


# =============================================================================
# Plot — SAME format as 13A Fig05
# =============================================================================

def make_rich_fig(
    df: pd.DataFrame,
    dual_ridge: float,
    outdir: Path,
):
    y = np.arange(
        len(df)
    )[::-1]

    # Slightly taller than old Fig05 because there are more variants.
    fig = plt.figure(
        figsize=(7.2, 5.25)
    )

    gs = GridSpec(
        1,
        2,
        width_ratios=[
            5.0,
            1.15,
        ],
        wspace=0.06,
        figure=fig,
    )

    ax1 = fig.add_subplot(
        gs[0, 0]
    )
    ax2 = fig.add_subplot(
        gs[0, 1],
        sharey=ax1,
    )

    proposed_color = "#d62728"
    normal_color = "#1f77b4"

    # Exact old Fig05 point/error-bar format.
    for yi, row in zip(
        y,
        df.itertuples(index=False),
    ):
        is_proposed = (
            str(row.label)
            == "Physics-Compact (proposed)"
        )

        color = (
            proposed_color
            if is_proposed
            else normal_color
        )

        ax1.errorbar(
            row.mean,
            yi,
            xerr=row.std,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            markersize=6.5,
            zorder=3,
        )

    # Final Dual-Ridge reference.
    ax2.axvline(
        dual_ridge,
        linestyle="--",
        linewidth=1.15,
        color=normal_color,
        zorder=2,
    )

    # Main nonlinear range.
    left_min = float(
        np.min(
            df["mean"]
            - df["std"]
        )
    ) - 0.0010

    left_max = float(
        np.max(
            df["mean"]
            + df["std"]
        )
    ) + 0.0010

    ax1.set_xlim(
        left_min,
        left_max,
    )

    # Ridge window; automatically centered on current final Dual Ridge.
    ridge_pad = max(
        0.003,
        0.003
        * abs(
            dual_ridge
        ),
    )

    ax2.set_xlim(
        dual_ridge - ridge_pad,
        dual_ridge + ridge_pad,
    )

    # Y labels.
    ax1.set_yticks(
        y
    )
    ax1.set_yticklabels(
        df["label"]
    )

    for t in ax1.get_yticklabels():
        if (
            t.get_text()
            == "Physics-Compact (proposed)"
        ):
            t.set_fontweight(
                "bold"
            )

    plt.setp(
        ax2.get_yticklabels(),
        visible=False,
    )

    ax2.tick_params(
        labelleft=False
    )

    ax1.set_ylabel(
        "Variant"
    )

    finish_axis(
        ax1
    )
    finish_axis(
        ax2
    )

    # Broken-axis styling copied from 13A.
    ax1.spines[
        "right"
    ].set_visible(
        False
    )
    ax2.spines[
        "left"
    ].set_visible(
        False
    )

    ax1.tick_params(
        right=False
    )
    ax2.tick_params(
        left=False
    )

    d = 0.012

    kwargs1 = dict(
        transform=ax1.transAxes,
        color="k",
        clip_on=False,
        linewidth=0.8,
    )

    ax1.plot(
        (1-d, 1+d),
        (-d, +d),
        **kwargs1,
    )
    ax1.plot(
        (1-d, 1+d),
        (1-d, 1+d),
        **kwargs1,
    )

    kwargs2 = dict(
        transform=ax2.transAxes,
        color="k",
        clip_on=False,
        linewidth=0.8,
    )

    ax2.plot(
        (-d, +d),
        (-d, +d),
        **kwargs2,
    )
    ax2.plot(
        (-d, +d),
        (1-d, 1+d),
        **kwargs2,
    )

    # Same Ridge label style as user's 13A v6.
    ax2.text(
        0.10,
        0.97,
        f"Dual Ridge = {dual_ridge:.4f}",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="none",
            alpha=0.9,
        ),
        zorder=5,
    )

    # More left margin for the richer labels.
    fig.subplots_adjust(
        left=0.32,
        right=0.98,
        bottom=0.15,
        top=0.98,
    )

    x_center_main = (
        0.32
        + 0.98
    ) / 2.0

    fig.text(
        x_center_main,
        0.045,
        r"Validation apparent-wind vector RMSE (m s$^{-1}$)",
        ha="center",
        va="center",
    )

    save_figure(
        fig,
        outdir
        / "Fig_rich_ablation_final",
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_ROOT,
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--final-validation-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    root = args.project_root.resolve()

    dataset_dir = (
        args.dataset_dir.resolve()
        if args.dataset_dir is not None
        else root
        / "data"
        / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
    )

    final_dir = (
        args.final_validation_dir.resolve()
        if args.final_validation_dir is not None
        else root
        / "data"
        / "forecasting"
        / "15E_VH_AlphaSearch_v0_1"
        / "full_retrain"
        / "alpha_1500"
    )

    outdir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root
        / "figures"
        / "final_paper"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    apply_style(
        root
    )

    print("=" * 100)
    print("RICH FINAL ABLATION — OLD STRUCTURAL VARIANTS + FINAL BRANCH ABLATIONS")
    print("=" * 100)
    print(f"project_root         : {root}")
    print(f"dataset_dir          : {dataset_dir}")
    print(f"final_validation_dir : {final_dir}")
    print(f"output_dir           : {outdir}")
    print(f"alpha_W              : {ALPHA_W:g}")
    print(f"alpha_M              : {ALPHA_M:g}")
    print("=" * 100)

    # 1) Exact historical variants.
    historical_df = load_historical_ablation(
        dataset_dir
    )

    # 2) Rebuild current final anchors.
    Xtr, ytr, Xva, yva = load_primary_arrays(
        dataset_dir
    )

    print("[FIT] Rebuilding final Dual-Ridge anchors ...")
    ridge_wind, ridge_motion = rebuild_final_anchors(
        Xtr,
        ytr,
        Xva,
    )

    # 3) Final five-seed predictions.
    print("[LOAD] Final alpha_M=1500 five-seed validation predictions ...")
    final_seeds = load_final_seed_predictions(
        final_dir
    )

    # Truth audit.
    if not np.allclose(
        final_seeds[0]["truth"],
        yva,
        atol=1e-6,
        rtol=0,
    ):
        raise RuntimeError(
            "Final prediction truth does not align with validation.npz."
        )

    # 4) Current branch ablations + final Dual Ridge.
    current_df, dual_ridge = compute_current_variants(
        yva=yva,
        ridge_wind=ridge_wind,
        ridge_motion=ridge_motion,
        final_seeds=final_seeds,
        final_dir=final_dir,
    )

    # 5) Combine.
    combined = combine_sources(
        current_df,
        historical_df,
    )

    combined["dual_ridge_reference"] = dual_ridge

    source_csv = (
        outdir
        / "Fig_rich_ablation_final_data.csv"
    )

    combined.to_csv(
        source_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print("")
    print("=" * 100)
    print("FIGURE SOURCE DATA")
    print("=" * 100)
    print(
        combined[
            [
                "label",
                "mean",
                "std",
                "group",
            ]
        ].to_string(
            index=False
        )
    )
    print(
        f"\nFinal Dual Ridge reference = {dual_ridge:.6f} m/s"
    )
    print(
        f"[SAVED DATA] {source_csv}"
    )

    # 6) Plot using the exact old Fig05 style.
    make_rich_fig(
        df=combined,
        dual_ridge=dual_ridge,
        outdir=outdir,
    )

    print("")
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
