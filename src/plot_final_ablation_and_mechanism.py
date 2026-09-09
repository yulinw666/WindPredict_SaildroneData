# -*- coding: utf-8 -*-
r"""
plot_final_ablation_and_mechanism.py

Generate two final-paper diagnostic figures for the frozen
Branch-Aware Physics-Compact model.

FIGURE A — Branch ablation forest plot
--------------------------------------
Validation mean apparent-wind vector RMSE, five-seed mean ± std:

    Physics-Compact
    Motion enhanced
    Wind enhanced

with the final Dual Ridge shown as a vertical reference.

Definitions:
    Dual Ridge:
        final wind Ridge (alpha_W = 0)
        + final joint motion/HDG Ridge (alpha_M = 1500)

    Wind enhanced:
        final nonlinear TrueWind branch
        + Ridge motion/HDG

    Motion enhanced:
        Ridge wind
        + final nonlinear Motion-HDG gated residual branch

    Physics-Compact:
        final nonlinear TrueWind branch
        + final nonlinear Motion-HDG branch

The figure automatically uses a broken x-axis when the Dual-Ridge reference
is sufficiently far from the three nonlinear variants.

FIGURE B — Residual-correction mechanism under transfer
-------------------------------------------------------
To avoid mixing the FINAL alpha_M=1500 model with old alpha=100 persistence-
skill diagnostics, the default mechanism x-axis is defined entirely from the
FINAL frozen results:

    x = 100 * (RMSE_Ridge,d / RMSE_Ridge,SD1090 - 1)

which is the Dual-Ridge vessel-RMSE shift relative to SD1090 validation.

    y = 100 * (1 - RMSE_MotionEnhanced,d / RMSE_Ridge,d)

which is the vessel-RMSE reduction supplied by the nonlinear motion branch.

Thus:
    x > 0 : the frozen linear vessel anchor degrades relative to development;
    y > 0 : the nonlinear motion residual reduces the anchor error.

Datasets:
    SD1090 validation
    SD1033-2024 matched
    SD1033-2023
    SD1033-2022

This definition is intentionally based only on the FINAL alpha_M=1500
results, so no stale alpha=100 mechanism number is reused.

Publication style
-----------------
- Uses src/sci_plot_style.py when available.
- Times New Roman / serif fallback.
- 600 dpi PNG + vector PDF.
- Thin axes / markers / error bars.
- Classic blue-orange-green-red-purple journal palette.
- No figure title inside the plot.

Default project root
--------------------
D:\project\WindPredict_SaildroneData

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\plot_final_ablation_and_mechanism.py" `
  --root "D:\project\WindPredict_SaildroneData"

Generate only one figure:
    --only ablation
or
    --only mechanism
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from sklearn.linear_model import Ridge


# =============================================================================
# Frozen protocol
# =============================================================================

HORIZONS = np.asarray([1, 2, 3, 5, 10], dtype=int)
SEEDS = [500043, 501052, 502061, 503070, 504079]

ALPHA_W = 0.0
ALPHA_M = 1500.0

COLORS = {
    "Dual Ridge": "#ff7f0e",       # orange
    "Wind enhanced": "#2ca02c",    # green
    "Motion enhanced": "#9467bd",  # purple
    "Physics-Compact": "#d62728",  # red
    "SD1090 validation": "#1f77b4",
    "SD1033-2024 matched": "#ff7f0e",
    "SD1033-2023": "#2ca02c",
    "SD1033-2022": "#d62728",
}

MARKERS = {
    "Wind enhanced": "^",
    "Motion enhanced": "v",
    "Physics-Compact": "D",
    "SD1090 validation": "o",
    "SD1033-2024 matched": "s",
    "SD1033-2023": "^",
    "SD1033-2022": "D",
}


# =============================================================================
# Style
# =============================================================================

def load_sci_style(root: Path) -> None:
    style_path = root / "src" / "sci_plot_style.py"

    if style_path.exists():
        spec = importlib.util.spec_from_file_location(
            "sci_plot_style_final_diagnostics",
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
            "legend.fontsize": 7.3,
            "axes.linewidth": 0.8,
            "lines.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "axes.unicode_minus": False,
        }
    )


def clean_axis(ax, grid=True) -> None:
    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
    )
    ax.minorticks_on()

    if grid:
        ax.grid(
            True,
            which="major",
            linewidth=0.40,
            alpha=0.18,
        )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save_both(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)

    png = Path(
        os.path.abspath(
            os.path.normpath(
                str(out_base.with_suffix(".png"))
            )
        )
    )
    pdf = Path(
        os.path.abspath(
            os.path.normpath(
                str(out_base.with_suffix(".pdf"))
            )
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
# General helpers
# =============================================================================

def norm_text(x) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(x).lower(),
    )


def find_col(
    df: pd.DataFrame,
    aliases,
    required=True,
):
    lookup = {
        norm_text(c): c
        for c in df.columns
    }

    for alias in aliases:
        key = norm_text(alias)

        if key in lookup:
            return lookup[key]

    for alias in aliases:
        key = norm_text(alias)

        for normalized, original in lookup.items():
            if key in normalized or normalized in key:
                return original

    if required:
        raise KeyError(
            f"Could not find any of {aliases} in columns:\n"
            f"{list(df.columns)}"
        )

    return None


def canonical_variant(x: str) -> str:
    n = norm_text(x)

    if (
        "physicscompact" in n
        or "15evh" in n
        or n == "full"
    ):
        return "Physics-Compact"

    if (
        "ridgewindmotionenhanced" in n
        or "motionenhanced" in n
        or "motiononly" in n
    ):
        return "Motion enhanced"

    if (
        "windenhancedridgemotion" in n
        or "windenhanced" in n
        or "windonly" in n
    ):
        return "Wind enhanced"

    if (
        "dualridge" in n
        or n == "ridge"
        or "ridgeridge" in n
    ):
        return "Dual Ridge"

    return str(x)


def canonical_dataset(x: str) -> str:
    n = norm_text(x)

    if "sd1090" in n:
        return "SD1090 validation"

    if "sd1033" in n and "2024" in n:
        return "SD1033-2024 matched"

    if "sd1033" in n and "2023" in n:
        return "SD1033-2023"

    if "sd1033" in n and "2022" in n:
        return "SD1033-2022"

    return str(x)


# =============================================================================
# Primary SD1090 arrays and final linear anchors
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


def fit_final_dual_ridge(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Final anchors:
        true-wind Ridge alpha_W = 0
        joint motion/HDG Ridge alpha_M = 1500

    Returns:
        ridge_wind   [Nval,5,2]
        ridge_motion [Nval,5,4]
    """
    # True-wind anchor uses only U,V,T,RH history.
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

    # Joint motion/HDG anchor uses all 11 history channels and all 30 outputs.
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

    # Unit-normalize heading sin/cos.
    q = ridge_motion[:, :, 2:4]
    q_norm = np.sqrt(
        np.sum(
            q * q,
            axis=-1,
            keepdims=True,
        )
    )
    ridge_motion[:, :, 2:4] = (
        q
        / np.maximum(
            q_norm,
            1e-12,
        )
    )

    return (
        ridge_wind.astype(np.float32),
        ridge_motion.astype(np.float32),
    )


# =============================================================================
# Final five-seed Physics-Compact predictions
# =============================================================================

def load_final_validation_seeds(
    final_validation_dir: Path,
):
    rows = []

    for seed in SEEDS:
        p = (
            final_validation_dir
            / f"15E_VH_seed_{seed}_validation_predictions.npz"
        )

        if not p.exists():
            raise FileNotFoundError(
                f"Missing final validation prediction:\n{p}"
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

            rows.append(
                {
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
                }
            )

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


# =============================================================================
# Metrics
# =============================================================================

def vector_rmse(
    truth_xy: np.ndarray,
    pred_xy: np.ndarray,
) -> float:
    err = (
        np.asarray(pred_xy, dtype=np.float64)
        - np.asarray(truth_xy, dtype=np.float64)
    )

    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    err * err,
                    axis=-1,
                )
            )
        )
    )


def mean_vector_rmse_over_horizons(
    truth_xy: np.ndarray,
    pred_xy: np.ndarray,
) -> float:
    """
    Calculate vector RMSE separately at each horizon, then average over
    [1,2,3,5,10] min. This matches the paper's mean multi-horizon reporting.
    """
    values = []

    for h in range(5):
        values.append(
            vector_rmse(
                truth_xy[:, h, :],
                pred_xy[:, h, :],
            )
        )

    return float(
        np.mean(
            values
        )
    )


def apparent_earth(
    wind: np.ndarray,
    motion: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(wind, dtype=np.float64)
        - np.asarray(motion, dtype=np.float64)[:, :, 0:2]
    )


# =============================================================================
# Figure A data
# =============================================================================

def compute_validation_ablation(
    y_val: np.ndarray,
    ridge_wind: np.ndarray,
    ridge_motion: np.ndarray,
    seed_data,
) -> pd.DataFrame:
    truth_wind = y_val[:, :, 0:2]
    truth_motion = y_val[:, :, 2:6]
    truth_app = apparent_earth(
        truth_wind,
        truth_motion,
    )

    ridge_app = apparent_earth(
        ridge_wind,
        ridge_motion,
    )

    ridge_value = mean_vector_rmse_over_horizons(
        truth_app,
        ridge_app,
    )

    rows = []

    # Fixed Dual Ridge reference.
    rows.append(
        {
            "variant": "Dual Ridge",
            "seed": np.nan,
            "AppVector_RMSE_mps": ridge_value,
        }
    )

    # Five-seed branch combinations.
    for d in seed_data:
        wind_enhanced_app = apparent_earth(
            d["wind"],
            ridge_motion,
        )

        motion_enhanced_app = apparent_earth(
            ridge_wind,
            d["motion"],
        )

        full_app = apparent_earth(
            d["wind"],
            d["motion"],
        )

        rows.extend(
            [
                {
                    "variant": "Wind enhanced",
                    "seed": d["seed"],
                    "AppVector_RMSE_mps": (
                        mean_vector_rmse_over_horizons(
                            truth_app,
                            wind_enhanced_app,
                        )
                    ),
                },
                {
                    "variant": "Motion enhanced",
                    "seed": d["seed"],
                    "AppVector_RMSE_mps": (
                        mean_vector_rmse_over_horizons(
                            truth_app,
                            motion_enhanced_app,
                        )
                    ),
                },
                {
                    "variant": "Physics-Compact",
                    "seed": d["seed"],
                    "AppVector_RMSE_mps": (
                        mean_vector_rmse_over_horizons(
                            truth_app,
                            full_app,
                        )
                    ),
                },
            ]
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Figure A plotting
# =============================================================================

def summarize_ablation(
    ablation_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for variant in [
        "Physics-Compact",
        "Motion enhanced",
        "Wind enhanced",
    ]:
        values = (
            ablation_df.loc[
                ablation_df["variant"] == variant,
                "AppVector_RMSE_mps",
            ]
            .astype(float)
            .to_numpy()
        )

        rows.append(
            {
                "variant": variant,
                "mean": float(
                    np.mean(
                        values
                    )
                ),
                "std": float(
                    np.std(
                        values,
                        ddof=1,
                    )
                ),
            }
        )

    ridge_value = float(
        ablation_df.loc[
            ablation_df["variant"] == "Dual Ridge",
            "AppVector_RMSE_mps",
        ].iloc[0]
    )

    summary = pd.DataFrame(
        rows
    )
    summary.attrs["dual_ridge"] = ridge_value

    return summary


def should_use_broken_axis(
    summary: pd.DataFrame,
    ridge_value: float,
) -> bool:
    means = summary["mean"].to_numpy(float)
    stds = summary["std"].to_numpy(float)

    learned_min = float(
        np.min(
            means - stds
        )
    )
    learned_max = float(
        np.max(
            means + stds
        )
    )

    learned_span = max(
        learned_max - learned_min,
        1e-6,
    )

    gap = ridge_value - learned_max

    return bool(
        gap > 2.0 * learned_span
    )


def plot_ablation(
    output_dir: Path,
    ablation_df: pd.DataFrame,
) -> None:
    summary = summarize_ablation(
        ablation_df
    )
    ridge_value = float(
        summary.attrs[
            "dual_ridge"
        ]
    )

    variants = [
        "Physics-Compact",
        "Motion enhanced",
        "Wind enhanced",
    ]

    y_positions = {
        variant: len(variants) - 1 - i
        for i, variant in enumerate(
            variants
        )
    }

    broken = should_use_broken_axis(
        summary,
        ridge_value,
    )

    if broken:
        fig = plt.figure(
            figsize=(7.15, 2.70),
        )

        gs = GridSpec(
            1,
            2,
            figure=fig,
            width_ratios=[
                5.0,
                1.05,
            ],
            wspace=0.04,
            left=0.25,
            right=0.985,
            bottom=0.22,
            top=0.91,
        )

        ax_left = fig.add_subplot(
            gs[0, 0]
        )
        ax_right = fig.add_subplot(
            gs[0, 1],
            sharey=ax_left,
        )

        axes = [
            ax_left,
            ax_right,
        ]

        learned_low = float(
            np.min(
                summary["mean"]
                - 2.5 * summary["std"]
            )
        )
        learned_high = float(
            np.max(
                summary["mean"]
                + 2.5 * summary["std"]
            )
        )

        span = max(
            learned_high - learned_low,
            0.002,
        )

        ax_left.set_xlim(
            learned_low - 0.15 * span,
            learned_high + 0.15 * span,
        )

        ridge_pad = max(
            0.0025,
            0.015 * abs(
                ridge_value
            ),
        )

        ax_right.set_xlim(
            ridge_value - ridge_pad,
            ridge_value + ridge_pad,
        )

        # Broken-axis diagonal marks.
        d = 0.012
        kwargs = dict(
            transform=ax_left.transAxes,
            color="black",
            clip_on=False,
            linewidth=0.8,
        )
        ax_left.plot(
            (1 - d, 1 + d),
            (-d, +d),
            **kwargs,
        )
        ax_left.plot(
            (1 - d, 1 + d),
            (1 - d, 1 + d),
            **kwargs,
        )

        kwargs.update(
            transform=ax_right.transAxes
        )
        ax_right.plot(
            (-d, +d),
            (-d, +d),
            **kwargs,
        )
        ax_right.plot(
            (-d, +d),
            (1 - d, 1 + d),
            **kwargs,
        )

        ax_left.spines["right"].set_visible(
            False
        )
        ax_right.spines["left"].set_visible(
            False
        )

        ax_left.tick_params(
            right=False
        )
        ax_right.tick_params(
            left=False,
            labelleft=False,
        )

    else:
        fig, ax = plt.subplots(
            figsize=(5.65, 2.70),
        )

        fig.subplots_adjust(
            left=0.29,
            right=0.985,
            bottom=0.22,
            top=0.91,
        )

        ax_left = ax
        ax_right = None
        axes = [ax]

        all_low = min(
            float(
                np.min(
                    summary["mean"]
                    - 2.5 * summary["std"]
                )
            ),
            ridge_value,
        )
        all_high = max(
            float(
                np.max(
                    summary["mean"]
                    + 2.5 * summary["std"]
                )
            ),
            ridge_value,
        )

        span = max(
            all_high - all_low,
            0.01,
        )

        ax.set_xlim(
            all_low - 0.08 * span,
            all_high + 0.08 * span,
        )

    # Plot the three learned variants.
    for variant in variants:
        row = summary.loc[
            summary["variant"] == variant
        ].iloc[0]

        y = y_positions[
            variant
        ]

        ax_left.errorbar(
            float(
                row["mean"]
            ),
            y,
            xerr=float(
                row["std"]
            ),
            fmt=MARKERS[
                variant
            ],
            color=COLORS[
                variant
            ],
            markerfacecolor=COLORS[
                variant
            ],
            markeredgecolor="white",
            markeredgewidth=0.55,
            markersize=6.2
            if variant
            == "Physics-Compact"
            else 5.5,
            elinewidth=0.85,
            capsize=3.0,
            capthick=0.85,
            zorder=5,
        )

    # Dual-Ridge vertical reference.
    ref_ax = (
        ax_right
        if ax_right is not None
        else ax_left
    )

    ref_ax.axvline(
        ridge_value,
        color=COLORS[
            "Dual Ridge"
        ],
        linewidth=1.0,
        linestyle="--",
        zorder=2,
    )

    ref_ax.text(
        ridge_value,
        2.22,
        f"Dual Ridge = {ridge_value:.4f}",
        ha="center",
        va="bottom",
        fontsize=7.3,
        color=COLORS[
            "Dual Ridge"
        ],
    )

    # Common y-axis.
    ax_left.set_yticks(
        [
            y_positions[v]
            for v in variants
        ]
    )

    labels = [
        r"$\bf{Physics}$-$\bf{Compact}$ (proposed)",
        "Motion enhanced",
        "Wind enhanced",
    ]

    ax_left.set_yticklabels(
        labels
    )

    for tick, variant in zip(
        ax_left.get_yticklabels(),
        variants,
    ):
        if variant == "Physics-Compact":
            tick.set_fontweight(
                "bold"
            )

    ax_left.set_ylim(
        -0.55,
        2.45,
    )

    # Only horizontal reference gridlines; cleaner forest-plot appearance.
    ax_left.grid(
        True,
        axis="y",
        linewidth=0.40,
        alpha=0.18,
    )

    ax_left.minorticks_on()
    ax_left.tick_params(
        which="both",
        direction="in",
        top=True,
    )

    if ax_right is not None:
        ax_right.grid(
            True,
            axis="y",
            linewidth=0.40,
            alpha=0.18,
        )
        ax_right.minorticks_on()
        ax_right.tick_params(
            which="both",
            direction="in",
            top=True,
            right=True,
        )

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(
                0.8
            )

    fig.supxlabel(
        r"Validation apparent-wind vector RMSE (m s$^{-1}$)",
        y=0.055,
        fontsize=8.5,
    )

    save_both(
        fig,
        output_dir
        / "Fig_branch_ablation_final",
    )


# =============================================================================
# External branch-decomposition loader
# =============================================================================

def locate_external_branch_csv(
    external_dir: Path,
) -> Path:
    preferred = (
        external_dir
        / "15F_external_branch_decomposition.csv"
    )

    if preferred.exists():
        return preferred

    candidates = []

    for p in external_dir.rglob(
        "*.csv"
    ):
        name = norm_text(
            p.name
        )

        if (
            "branch" in name
            and (
                "decomposition" in name
                or "ablation" in name
            )
        ):
            candidates.append(
                p
            )

    if not candidates:
        raise FileNotFoundError(
            "Could not find final external branch-decomposition CSV under:\n"
            f"{external_dir}\n\n"
            "Expected something like:\n"
            "15F_external_branch_decomposition.csv"
        )

    candidates.sort(
        key=lambda p: (
            len(
                str(
                    p
                )
            ),
            str(
                p
            ),
        )
    )

    return candidates[0]


def load_external_mechanism_data(
    external_dir: Path,
) -> pd.DataFrame:
    csv_path = locate_external_branch_csv(
        external_dir
    )

    print(
        f"[LOAD] external branch decomposition: {csv_path}"
    )

    df = pd.read_csv(
        csv_path
    )

    dataset_col = find_col(
        df,
        [
            "dataset",
            "external_dataset",
            "mission",
        ],
    )

    variant_col = find_col(
        df,
        [
            "model",
            "variant",
            "method",
        ],
    )

    vessel_col = find_col(
        df,
        [
            "Vessel_RMSE_mps",
            "VesselVector_RMSE_mps",
            "vessel_vector_RMSE_mps",
            "vessel_RMSE_mps",
        ],
    )

    work = pd.DataFrame(
        {
            "dataset": df[
                dataset_col
            ].map(
                canonical_dataset
            ),
            "variant": df[
                variant_col
            ].map(
                canonical_variant
            ),
            "Vessel_RMSE_mps": pd.to_numeric(
                df[
                    vessel_col
                ],
                errors="coerce",
            ),
        }
    )

    work = work.dropna(
        subset=[
            "Vessel_RMSE_mps"
        ]
    )

    keep_datasets = [
        "SD1033-2024 matched",
        "SD1033-2023",
        "SD1033-2022",
    ]

    work = work[
        work["dataset"].isin(
            keep_datasets
        )
    ]

    return work


# =============================================================================
# Figure B data
# =============================================================================

def compute_sd1090_mechanism_row(
    y_val: np.ndarray,
    ridge_motion: np.ndarray,
    seed_data,
) -> Dict[str, float]:
    truth_vessel = y_val[:, :, 2:4]

    ridge_vessel = ridge_motion[:, :, 0:2]

    ridge_rmse = (
        mean_vector_rmse_over_horizons(
            truth_vessel,
            ridge_vessel,
        )
    )

    motion_values = []

    for d in seed_data:
        motion_values.append(
            mean_vector_rmse_over_horizons(
                truth_vessel,
                d["motion"][
                    :,
                    :,
                    0:2,
                ],
            )
        )

    motion_mean = float(
        np.mean(
            motion_values
        )
    )
    motion_std = float(
        np.std(
            motion_values,
            ddof=1,
        )
    )

    return {
        "dataset": "SD1090 validation",
        "RidgeVessel_RMSE_mps": ridge_rmse,
        "MotionEnhanced_RMSE_mps": motion_mean,
        "MotionEnhanced_RMSE_std_mps": motion_std,
    }


def summarize_external_mechanism(
    external_work: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for dataset in [
        "SD1033-2024 matched",
        "SD1033-2023",
        "SD1033-2022",
    ]:
        d = external_work[
            external_work[
                "dataset"
            ]
            == dataset
        ]

        ridge_values = (
            d.loc[
                d["variant"]
                == "Dual Ridge",
                "Vessel_RMSE_mps",
            ]
            .astype(float)
            .to_numpy()
        )

        # Vessel predictions are identical for Motion enhanced and the full
        # Physics-Compact model because the wind branch does not alter vessel.
        motion_values = (
            d.loc[
                d["variant"].isin(
                    [
                        "Motion enhanced",
                        "Physics-Compact",
                    ]
                ),
                "Vessel_RMSE_mps",
            ]
            .astype(float)
            .to_numpy()
        )

        if len(
            ridge_values
        ) == 0:
            raise RuntimeError(
                f"{dataset}: Dual Ridge vessel RMSE not found "
                "in final external branch-decomposition CSV."
            )

        if len(
            motion_values
        ) == 0:
            raise RuntimeError(
                f"{dataset}: Motion-enhanced / Physics-Compact vessel RMSE "
                "not found in final external branch-decomposition CSV."
            )

        rows.append(
            {
                "dataset": dataset,
                "RidgeVessel_RMSE_mps": float(
                    np.mean(
                        ridge_values
                    )
                ),
                "MotionEnhanced_RMSE_mps": float(
                    np.mean(
                        motion_values
                    )
                ),
                "MotionEnhanced_RMSE_std_mps": float(
                    np.std(
                        motion_values,
                        ddof=1,
                    )
                )
                if len(
                    motion_values
                ) > 1
                else 0.0,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_mechanism_table(
    sd1090_row: Dict[str, float],
    external_rows: pd.DataFrame,
) -> pd.DataFrame:
    all_rows = pd.concat(
        [
            pd.DataFrame(
                [
                    sd1090_row
                ]
            ),
            external_rows,
        ],
        ignore_index=True,
    )

    dev_ridge = float(
        all_rows.loc[
            all_rows["dataset"]
            == "SD1090 validation",
            "RidgeVessel_RMSE_mps",
        ].iloc[
            0
        ]
    )

    all_rows[
        "RidgeVessel_shift_vs_SD1090_pct"
    ] = (
        100.0
        * (
            all_rows[
                "RidgeVessel_RMSE_mps"
            ]
            / dev_ridge
            - 1.0
        )
    )

    all_rows[
        "MotionBranch_RMSE_reduction_pct"
    ] = (
        100.0
        * (
            1.0
            - all_rows[
                "MotionEnhanced_RMSE_mps"
            ]
            / all_rows[
                "RidgeVessel_RMSE_mps"
            ]
        )
    )

    # Approximate y-error propagated directly from the five-seed motion RMSE std.
    all_rows[
        "MotionBranch_reduction_std_pct"
    ] = (
        100.0
        * all_rows[
            "MotionEnhanced_RMSE_std_mps"
        ]
        / all_rows[
            "RidgeVessel_RMSE_mps"
        ]
    )

    return all_rows


# =============================================================================
# Figure B plotting
# =============================================================================

def plot_mechanism(
    output_dir: Path,
    mechanism_df: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(
        figsize=(6.25, 3.95),
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        bottom=0.18,
        top=0.94,
    )

    # Reference axes.
    ax.axvline(
        0.0,
        color="0.45",
        linewidth=0.8,
        linestyle="--",
        zorder=1,
    )

    ax.axhline(
        0.0,
        color="0.45",
        linewidth=0.8,
        linestyle="--",
        zorder=1,
    )

    # Lightly emphasize the desirable correction region y>0.
    ymin_current = min(
        -2.0,
        float(
            mechanism_df[
                "MotionBranch_RMSE_reduction_pct"
            ].min()
        )
        - 4.0,
    )

    ymax_current = max(
        10.0,
        float(
            mechanism_df[
                "MotionBranch_RMSE_reduction_pct"
            ].max()
        )
        + 6.0,
    )

    ax.axhspan(
        0.0,
        ymax_current,
        color="#2ca02c",
        alpha=0.025,
        zorder=0,
    )

    annotation_offsets = {
        "SD1090 validation": (
            7,
            8,
        ),
        "SD1033-2024 matched": (
            7,
            -18,
        ),
        "SD1033-2023": (
            8,
            8,
        ),
        "SD1033-2022": (
            8,
            8,
        ),
    }

    for row in mechanism_df.itertuples(
        index=False
    ):
        dataset = str(
            row.dataset
        )

        x = float(
            row.RidgeVessel_shift_vs_SD1090_pct
        )
        y = float(
            row.MotionBranch_RMSE_reduction_pct
        )
        yerr = float(
            row.MotionBranch_reduction_std_pct
        )

        ax.errorbar(
            x,
            y,
            yerr=yerr
            if yerr > 0
            else None,
            fmt=MARKERS[
                dataset
            ],
            color=COLORS[
                dataset
            ],
            markerfacecolor=COLORS[
                dataset
            ],
            markeredgecolor="white",
            markeredgewidth=0.60,
            markersize=7.2,
            elinewidth=0.75,
            capsize=2.5,
            zorder=5,
        )

        dx, dy = annotation_offsets[
            dataset
        ]

        short_label = (
            dataset.replace(
                "SD1033-",
                ""
            )
            .replace(
                "SD1090 ",
                ""
            )
        )

        ax.annotate(
            short_label,
            xy=(
                x,
                y,
            ),
            xytext=(
                dx,
                dy,
            ),
            textcoords="offset points",
            fontsize=7.4,
            color=COLORS[
                dataset
            ],
            ha="left",
            va="center",
        )

    # Dynamic limits with comfortable whitespace.
    xvals = mechanism_df[
        "RidgeVessel_shift_vs_SD1090_pct"
    ].to_numpy(
        float
    )

    xlow = float(
        np.min(
            xvals
        )
    )
    xhigh = float(
        np.max(
            xvals
        )
    )
    xspan = max(
        xhigh - xlow,
        20.0,
    )

    ax.set_xlim(
        xlow - 0.12 * xspan,
        xhigh + 0.16 * xspan,
    )

    ax.set_ylim(
        ymin_current,
        ymax_current,
    )

    ax.set_xlabel(
        "Dual-Ridge vessel RMSE shift relative to SD1090 validation (%)"
    )
    ax.set_ylabel(
        "Motion-branch vessel RMSE reduction vs Dual Ridge (%)"
    )

    # Small directional explanations outside the data cloud.
    ax.text(
        0.02,
        0.97,
        "lower anchor error",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color="0.40",
    )

    ax.text(
        0.98,
        0.97,
        "larger anchor degradation",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color="0.40",
    )

    ax.text(
        0.985,
        0.06,
        r"$y>0$: residual correction reduces vessel error",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="0.35",
    )

    clean_axis(
        ax,
        grid=True,
    )

    # Compact legend.
    handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[
                dataset
            ],
            color="none",
            markerfacecolor=COLORS[
                dataset
            ],
            markeredgecolor="white",
            markeredgewidth=0.55,
            markersize=6.2,
            label=dataset,
        )
        for dataset in [
            "SD1090 validation",
            "SD1033-2024 matched",
            "SD1033-2023",
            "SD1033-2022",
        ]
    ]

    ax.legend(
        handles=handles,
        loc="lower left",
        ncol=2,
        fontsize=6.8,
        columnspacing=1.0,
        handletextpad=0.35,
        borderaxespad=0.5,
    )

    save_both(
        fig,
        output_dir
        / "Fig_motion_residual_mechanism_final",
    )


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
        "--final-validation-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--external-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--only",
        choices=[
            "all",
            "ablation",
            "mechanism",
        ],
        default="all",
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

    final_validation_dir = (
        args.final_validation_dir.resolve()
        if args.final_validation_dir is not None
        else root
        / "data"
        / "forecasting"
        / "15E_VH_AlphaSearch_v0_1"
        / "full_retrain"
        / "alpha_1500"
    )

    external_dir = (
        args.external_dir.resolve()
        if args.external_dir is not None
        else root
        / "data"
        / "forecasting"
        / "15F_Frozen_15EVH_alpha1500_External_v0_1"
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
    print("FINAL BRANCH ABLATION + MOTION-RESIDUAL MECHANISM")
    print("=" * 100)
    print(f"root                 : {root}")
    print(f"dataset_dir          : {dataset_dir}")
    print(f"final_validation_dir : {final_validation_dir}")
    print(f"external_dir         : {external_dir}")
    print(f"output_dir           : {output_dir}")
    print(f"alpha_W              : {ALPHA_W:g}")
    print(f"alpha_M              : {ALPHA_M:g}")
    print(f"only                 : {args.only}")
    print("=" * 100)

    # ------------------------------------------------------------------
    # Rebuild FINAL SD1090 validation anchors and final five-seed outputs.
    # ------------------------------------------------------------------
    Xtr, ytr, Xva, yva = load_primary_arrays(
        dataset_dir
    )

    print("[FIT] Rebuilding FINAL dual anchors ...")
    ridge_wind, ridge_motion = fit_final_dual_ridge(
        Xtr,
        ytr,
        Xva,
    )

    print("[LOAD] FINAL alpha_M=1500 validation predictions ...")
    seed_data = load_final_validation_seeds(
        final_validation_dir
    )

    # ------------------------------------------------------------------
    # Figure A
    # ------------------------------------------------------------------
    ablation_df = compute_validation_ablation(
        y_val=yva,
        ridge_wind=ridge_wind,
        ridge_motion=ridge_motion,
        seed_data=seed_data,
    )

    ablation_df.to_csv(
        output_dir
        / "Fig_branch_ablation_final_source.csv",
        index=False,
    )

    summary = summarize_ablation(
        ablation_df
    )

    ridge_reference = float(
        summary.attrs[
            "dual_ridge"
        ]
    )

    print("")
    print("=" * 100)
    print("FINAL VALIDATION BRANCH ABLATION")
    print("=" * 100)
    print(
        summary.to_string(
            index=False
        )
    )
    print(
        f"Dual Ridge reference = {ridge_reference:.6f} m/s"
    )

    if args.only in [
        "all",
        "ablation",
    ]:
        plot_ablation(
            output_dir,
            ablation_df,
        )

    # ------------------------------------------------------------------
    # Figure B
    # ------------------------------------------------------------------
    if args.only in [
        "all",
        "mechanism",
    ]:
        if not external_dir.exists():
            raise FileNotFoundError(
                f"Final external directory not found:\n{external_dir}"
            )

        sd1090_row = compute_sd1090_mechanism_row(
            y_val=yva,
            ridge_motion=ridge_motion,
            seed_data=seed_data,
        )

        external_work = load_external_mechanism_data(
            external_dir
        )

        external_rows = summarize_external_mechanism(
            external_work
        )

        mechanism_df = build_mechanism_table(
            sd1090_row,
            external_rows,
        )

        mechanism_df.to_csv(
            output_dir
            / "Fig_motion_residual_mechanism_final_source.csv",
            index=False,
        )

        print("")
        print("=" * 100)
        print("FINAL MOTION-RESIDUAL MECHANISM")
        print("=" * 100)
        print(
            mechanism_df[
                [
                    "dataset",
                    "RidgeVessel_RMSE_mps",
                    "MotionEnhanced_RMSE_mps",
                    "RidgeVessel_shift_vs_SD1090_pct",
                    "MotionBranch_RMSE_reduction_pct",
                ]
            ].to_string(
                index=False
            )
        )

        plot_mechanism(
            output_dir,
            mechanism_df,
        )

    print("")
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
