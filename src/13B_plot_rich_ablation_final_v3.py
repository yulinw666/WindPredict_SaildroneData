# -*- coding: utf-8 -*-
"""
13B_plot_rich_ablation_final_v3.py

Final rich ablation figure for the paper.
- Remove "Previous Physics-Compact"
- Remove "Compact-vessel residual"
- Keep only the variants that are easy to explain in the paper
- Proposed model in red, all others in blue
- Horizontal error bars + broken x-axis
- Save PNG / PDF / CSV

Author: ChatGPT
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =============================================================================
# STYLE
# =============================================================================
def apply_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,

        "font.size": 16,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,

        "axes.linewidth": 1.8,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.minor.width": 1.2,
        "ytick.minor.width": 1.2,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,

        "figure.dpi": 160,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


# =============================================================================
# DATA
# =============================================================================
def build_dataframe() -> tuple[pd.DataFrame, float]:
    """
    Final paper version:
      - Remove Previous Physics-Compact
      - Remove Compact-vessel residual
      - Keep 6 variants
      - Dual Ridge shown as vertical reference line only
    """
    rows = [
        {
            "label": "Physics-Compact (proposed)",
            "mean": 1.056610,
            "std": 0.002870,
            "group": "current-final",
            "is_proposed": True,
        },
        {
            "label": "Motion enhanced",
            "mean": 1.072780,
            "std": 0.001739,
            "group": "current-final",
            "is_proposed": False,
        },
        {
            "label": "Wind enhanced",
            "mean": 1.104466,
            "std": 0.003316,
            "group": "current-final",
            "is_proposed": False,
        },
        {
            "label": "Physics-vessel-only residual",
            "mean": 1.078910,
            "std": 0.001314,
            "group": "historical",
            "is_proposed": False,
        },
        {
            "label": "Physics-full residual",
            "mean": 1.079015,
            "std": 0.002574,
            "group": "historical",
            "is_proposed": False,
        },
        {
            "label": "Full residual",
            "mean": 1.080020,
            "std": 0.001212,
            "group": "historical",
            "is_proposed": False,
        },
    ]

    dual_ridge_ref = 1.121805
    df = pd.DataFrame(rows)
    return df, dual_ridge_ref


# =============================================================================
# SAVE
# =============================================================================
def save_both(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base.with_suffix(".png")))
    fig.savefig(str(out_base.with_suffix(".pdf")))


# =============================================================================
# PLOT
# =============================================================================
def plot_rich_ablation(output_dir: Path) -> None:
    apply_plot_style()

    df, ridge_ref = build_dataframe()

    # Save source data
    csv_path = output_dir / "Fig_rich_ablation_final_v3_data.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    # Order shown from top to bottom
    order = [
        "Physics-Compact (proposed)",
        "Motion enhanced",
        "Wind enhanced",
        "Physics-vessel-only residual",
        "Physics-full residual",
        "Full residual",
    ]
    df["label"] = pd.Categorical(df["label"], categories=order, ordered=True)
    df = df.sort_values("label").reset_index(drop=True)

    n = len(df)
    y = np.arange(n)[::-1]  # top to bottom

    # Colors
    blue = "#1f77b4"
    red = "#d62728"

    # Broken x-axis:
    # left panel: proposed + motion + historical cluster
    # right panel: wind enhanced + Dual Ridge reference
    xlim_left = (1.053, 1.0845)
    xlim_right = (1.100, 1.1245)

    fig = plt.figure(figsize=(15.8, 7.6))
    gs = GridSpec(1, 2, width_ratios=[7.9, 1.8], wspace=0.05)

    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)

    # Common axis style
    for ax in (ax_left, ax_right):
        ax.grid(True, axis="both", alpha=0.25, linewidth=0.9)
        ax.minorticks_on()
        ax.tick_params(direction="in", which="both", top=True, right=True)

    # Main scatter + error bars
    for idx, row in df.iterrows():
        color = red if row["is_proposed"] else blue

        # Decide which axis to use based on x position
        if xlim_left[0] <= row["mean"] <= xlim_left[1]:
            ax = ax_left
        else:
            ax = ax_right

        ax.errorbar(
            row["mean"],
            y[idx],
            xerr=row["std"],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.1,
            capsize=6,
            capthick=2.1,
            markersize=12.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.9,
            zorder=4 if row["is_proposed"] else 3,
        )

    # Dual Ridge reference line only on right axis
    ax_right.axvline(
        ridge_ref,
        color=blue,
        linestyle="--",
        linewidth=2.6,
        alpha=0.95,
        zorder=2,
    )
    ax_right.text(
        ridge_ref,
        y.max() + 0.35,
        f"Ridge = {ridge_ref:.4f}",
        ha="center",
        va="bottom",
        fontsize=15,
    )

    # Y ticks / labels
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(order)
    ax_right.tick_params(axis="y", left=False, labelleft=False)

    # Remove inner spines and inner y ticks to avoid ugly "double marks"
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_left.tick_params(axis="y", right=False)
    ax_right.tick_params(axis="y", left=False)

    # Limits
    ax_left.set_xlim(*xlim_left)
    ax_right.set_xlim(*xlim_right)
    ax_left.set_ylim(-0.6, n - 0.4)

    # Labels
    ax_left.set_ylabel("Variant")
    fig.supxlabel("Validation apparent-wind vector RMSE (m s$^{-1}$)", fontsize=20)

    # Make proposed label bold (best done after set_yticklabels)
    for t in ax_left.get_yticklabels():
        if t.get_text() == "Physics-Compact (proposed)":
            t.set_fontweight("bold")

    # Diagonal break marks only (cleaner than extra inner ticks)
    d = 0.012
    kwargs_left = dict(transform=ax_left.transAxes, color="k", clip_on=False, linewidth=1.7)
    kwargs_right = dict(transform=ax_right.transAxes, color="k", clip_on=False, linewidth=1.7)

    # left axis right-side diagonals
    ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs_left)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_left)

    # right axis left-side diagonals
    ax_right.plot((-d, +d), (-d, +d), **kwargs_right)
    ax_right.plot((-d, +d), (1 - d, 1 + d), **kwargs_right)

    # Optional fine ticks
    ax_left.set_xticks([1.055, 1.060, 1.065, 1.070, 1.075, 1.080])
    ax_right.set_xticks([1.10, 1.11, 1.12])

    # Save
    out_base = output_dir / "Fig_rich_ablation_final_v3"
    save_both(fig, out_base)
    plt.close(fig)

    # Print source data summary
    print("=" * 100)
    print("FIGURE SOURCE DATA")
    print("=" * 100)
    print(df[["label", "mean", "std", "group"]].to_string(index=False))
    print()
    print(f"Final Dual Ridge reference = {ridge_ref:.6f} m/s")
    print(f"[SAVED DATA] {csv_path}")
    print(f"[SAVED] {out_base.with_suffix('.png')}")
    print(f"[SAVED] {out_base.with_suffix('.pdf')}")
    print("=" * 100)


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot final rich ablation figure for the paper."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"D:\project\WindPredict_SaildroneData\figures\final_paper",
        help="Directory to save figure outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plot_rich_ablation(output_dir)


if __name__ == "__main__":
    main()