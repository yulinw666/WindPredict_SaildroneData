# -*- coding: utf-8 -*-
r"""
plot_motion_residual_mechanism_optimized.py

Optimized final-paper version of the motion-residual mechanism figure.

Reads:
    D:\project\WindPredict_SaildroneData\figures\final_paper\
        Fig_motion_residual_mechanism_final_source.csv

Expected columns:
    dataset
    RidgeVessel_RMSE_mps
    MotionEnhanced_RMSE_mps
    RidgeVessel_shift_vs_SD1090_pct
    MotionBranch_RMSE_reduction_pct
    MotionBranch_reduction_std_pct   (optional)

Main changes relative to the previous version
---------------------------------------------
1. Remove the large bottom legend; use direct point labels only.
2. Remove verbose in-panel explanatory sentences.
3. Shorten the x/y-axis labels.
4. Keep only the physically meaningful x=0 and y=0 reference lines.
5. Use a very light background band for y>0 (beneficial correction).
6. Use compact, non-overlapping direct labels.
7. Keep thin y-error bars only when the source CSV contains nonzero std.
8. Use project sci_plot_style.py when available.
9. Save 600 dpi PNG + vector PDF.

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\plot_motion_residual_mechanism_optimized.py" `
  --root "D:\project\WindPredict_SaildroneData"
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Frozen visual settings
# =============================================================================

DATASET_ORDER = [
    "SD1090 validation",
    "SD1033-2024 matched",
    "SD1033-2023",
    "SD1033-2022",
]

DISPLAY_LABELS = {
    "SD1090 validation": "SD1090 validation",
    "SD1033-2024 matched": "SD1033-2024 matched",
    "SD1033-2023": "SD1033-2023",
    "SD1033-2022": "SD1033-2022",
}

COLORS = {
    "SD1090 validation": "#1f77b4",
    "SD1033-2024 matched": "#ff7f0e",
    "SD1033-2023": "#2ca02c",
    "SD1033-2022": "#d62728",
}

MARKERS = {
    "SD1090 validation": "o",
    "SD1033-2024 matched": "s",
    "SD1033-2023": "^",
    "SD1033-2022": "D",
}

# label offsets in display points: (dx, dy)
LABEL_OFFSETS = {
    "SD1090 validation": (10, 8),
    "SD1033-2024 matched": (10, -18),
    "SD1033-2023": (10, 10),
    "SD1033-2022": (10, 10),
}


# =============================================================================
# Style
# =============================================================================

def load_sci_style(root: Path) -> None:
    style_path = root / "src" / "sci_plot_style.py"

    if style_path.exists():
        spec = importlib.util.spec_from_file_location(
            "sci_plot_style_motion_mechanism_opt",
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

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
        "axes.unicode_minus": False,
    })


def clean_axis(ax) -> None:
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
    )
    ax.minorticks_on()
    ax.grid(
        True,
        which="major",
        linewidth=0.42,
        alpha=0.18,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save_both(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)

    png = out_base.with_suffix(".png")
    pdf = out_base.with_suffix(".pdf")

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

    print(f"[SAVED] {png}")
    print(f"[SAVED] {pdf}")


# =============================================================================
# Data
# =============================================================================

def load_source(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Mechanism source CSV not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    required = {
        "dataset",
        "RidgeVessel_shift_vs_SD1090_pct",
        "MotionBranch_RMSE_reduction_pct",
    }

    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()

    df["RidgeVessel_shift_vs_SD1090_pct"] = pd.to_numeric(
        df["RidgeVessel_shift_vs_SD1090_pct"],
        errors="coerce",
    )

    df["MotionBranch_RMSE_reduction_pct"] = pd.to_numeric(
        df["MotionBranch_RMSE_reduction_pct"],
        errors="coerce",
    )

    if "MotionBranch_reduction_std_pct" in df.columns:
        df["MotionBranch_reduction_std_pct"] = pd.to_numeric(
            df["MotionBranch_reduction_std_pct"],
            errors="coerce",
        ).fillna(0.0)
    else:
        df["MotionBranch_reduction_std_pct"] = 0.0

    df = df.dropna(
        subset=[
            "RidgeVessel_shift_vs_SD1090_pct",
            "MotionBranch_RMSE_reduction_pct",
        ]
    )

    missing_ds = [
        d for d in DATASET_ORDER
        if d not in set(df["dataset"].astype(str))
    ]
    if missing_ds:
        raise RuntimeError(
            f"Missing datasets in source CSV: {missing_ds}"
        )

    rank = {name: i for i, name in enumerate(DATASET_ORDER)}
    df["rank"] = df["dataset"].map(rank)
    df = df.sort_values("rank").drop(columns="rank").reset_index(drop=True)

    return df


# =============================================================================
# Plot
# =============================================================================

def plot_mechanism(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:

    x = df["RidgeVessel_shift_vs_SD1090_pct"].to_numpy(float)
    y = df["MotionBranch_RMSE_reduction_pct"].to_numpy(float)
    yerr = df["MotionBranch_reduction_std_pct"].to_numpy(float)

    # Compact journal-size figure.
    fig, ax = plt.subplots(
        figsize=(6.20, 3.75),
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.985,
        bottom=0.18,
        top=0.96,
    )

    # Dynamic axis limits with moderate whitespace.
    x_span = max(float(np.ptp(x)), 20.0)
    y_span = max(float(np.ptp(y)), 10.0)

    x_min = min(float(np.min(x)) - 0.10 * x_span, -1.5)
    x_max = max(float(np.max(x)) + 0.13 * x_span, 1.5)

    # Keep y=0 visible because it has physical meaning.
    y_min = min(-1.5, float(np.min(y)) - 0.10 * y_span)
    y_max = float(np.max(y)) + 0.16 * y_span

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Very light beneficial-correction background, not visually dominant.
    ax.axhspan(
        0.0,
        y_max,
        facecolor="#2ca02c",
        alpha=0.025,
        zorder=0,
    )

    # Reference lines.
    ax.axvline(
        0.0,
        color="0.45",
        linestyle="--",
        linewidth=0.8,
        zorder=1,
    )
    ax.axhline(
        0.0,
        color="0.45",
        linestyle="--",
        linewidth=0.8,
        zorder=1,
    )

    # Points.
    for row in df.itertuples(index=False):
        dataset = str(row.dataset)

        px = float(row.RidgeVessel_shift_vs_SD1090_pct)
        py = float(row.MotionBranch_RMSE_reduction_pct)
        pe = float(row.MotionBranch_reduction_std_pct)

        if pe > 0:
            ax.errorbar(
                px,
                py,
                yerr=pe,
                fmt=MARKERS[dataset],
                color=COLORS[dataset],
                markerfacecolor=COLORS[dataset],
                markeredgecolor="white",
                markeredgewidth=0.55,
                markersize=7.0,
                elinewidth=0.75,
                capsize=2.5,
                capthick=0.75,
                zorder=4,
            )
        else:
            ax.plot(
                px,
                py,
                marker=MARKERS[dataset],
                linestyle="None",
                color=COLORS[dataset],
                markerfacecolor=COLORS[dataset],
                markeredgecolor="white",
                markeredgewidth=0.55,
                markersize=7.0,
                zorder=4,
            )

        dx, dy = LABEL_OFFSETS[dataset]

        ax.annotate(
            DISPLAY_LABELS[dataset],
            xy=(px, py),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.3,
            color=COLORS[dataset],
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor="white",
                edgecolor="none",
                alpha=0.78,
            ),
            zorder=5,
        )

    # Short, precise axis labels.
    ax.set_xlabel(
        "Dual-Ridge vessel RMSE shift vs SD1090 validation (%)"
    )
    ax.set_ylabel(
        "Motion-branch RMSE reduction vs Dual Ridge (%)"
    )

    # Replace the previous verbose corner text with compact directional cues.
    ax.text(
        0.015,
        0.975,
        "Lower relative anchor error",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color="0.40",
    )

    ax.text(
        0.985,
        0.975,
        "Greater anchor degradation",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color="0.40",
    )

    # Only one concise mechanism cue.
    ax.text(
        0.985,
        0.055,
        r"$y>0$: residual branch reduces vessel error",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.7,
        color="0.38",
    )

    clean_axis(ax)

    # No legend: direct labels already identify all points.
    save_both(
        fig,
        output_dir / "Fig_motion_residual_mechanism_optimized",
    )


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
        "--input-csv",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    root = args.root.resolve()

    input_csv = (
        args.input_csv.resolve()
        if args.input_csv is not None
        else root
        / "figures"
        / "final_paper"
        / "Fig_motion_residual_mechanism_final_source.csv"
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root
        / "figures"
        / "final_paper"
    )

    load_sci_style(root)

    print("=" * 100)
    print("OPTIMIZED MOTION-RESIDUAL MECHANISM FIGURE")
    print("=" * 100)
    print(f"input_csv : {input_csv}")
    print(f"output    : {output_dir}")
    print("=" * 100)

    df = load_source(input_csv)

    print(df[
        [
            "dataset",
            "RidgeVessel_shift_vs_SD1090_pct",
            "MotionBranch_RMSE_reduction_pct",
            "MotionBranch_reduction_std_pct",
        ]
    ].to_string(index=False))

    plot_mechanism(
        df=df,
        output_dir=output_dir,
    )

    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
