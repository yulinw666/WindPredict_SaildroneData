# -*- coding: utf-8 -*-
r"""
plot_branch_ablation_rich.py

Purpose
-------
Draw a richer branch-ablation forest plot in the style of the previous figure:
many variants on the y-axis, point + horizontal error bar, proposed model in red,
all other variants in blue, and Dual Ridge shown as a broken-axis reference.

This script is intended for the situation you described:
- the new figure with only three nonlinear points feels too sparse;
- you want to reuse several informative old ablation variants;
- the plotting style should remain close to the old figure.

Default variants included
-------------------------
1. Compact-vessel residual
2. Full residual
3. Vessel-only residual
4. Physics-full residual
5. Physics-vessel-only residual
6. Physics-Compact (proposed)

Color rule
----------
- Proposed model: red
- All other variants: blue
- Ridge reference: dashed blue line

Data source
-----------
Option A (recommended): provide a CSV with columns
    variant, mean, std
for the six (or more) variants.

Option B: if no CSV is given, the script uses the editable DEFAULT_ROWS table
below. These default values are only a fallback convenience. If you already
have exact numbers, replace them or provide --input-csv.

Suggested PowerShell
--------------------
python "D:\project\WindPredict_SaildroneData\src\plot_branch_ablation_rich.py" `
  --root "D:\project\WindPredict_SaildroneData"

or with explicit CSV:
python "D:\project\WindPredict_SaildroneData\src\plot_branch_ablation_rich.py" `
  --root "D:\project\WindPredict_SaildroneData" `
  --input-csv "D:\project\WindPredict_SaildroneData\figures\final_paper\branch_ablation_rich_source.csv"

Output
------
PNG + PDF will be saved to:
    <root>\figures\final_paper\Fig_branch_ablation_rich
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# -----------------------------------------------------------------------------
# Editable fallback numbers
# -----------------------------------------------------------------------------
# If you already have a CSV, you do NOT need to edit this.
# If not, you may directly replace these values with your exact means/stds.

DEFAULT_ROWS: List[Dict[str, float]] = [
    {"variant": "Compact-vessel residual",         "mean": 1.08010, "std": 0.00345},
    {"variant": "Full residual",                   "mean": 1.08002, "std": 0.00118},
    {"variant": "Vessel-only residual",            "mean": 1.07942, "std": 0.00092},
    {"variant": "Physics-full residual",           "mean": 1.07900, "std": 0.00255},
    {"variant": "Physics-vessel-only residual",    "mean": 1.07888, "std": 0.00128},
    {"variant": "Physics-Compact (proposed)",      "mean": 1.07850, "std": 0.00310},
]

DEFAULT_RIDGE = 1.1252

ORDER = [
    "Physics-Compact (proposed)",
    "Physics-vessel-only residual",
    "Physics-full residual",
    "Vessel-only residual",
    "Full residual",
    "Compact-vessel residual",
]


# -----------------------------------------------------------------------------
# Style helpers
# -----------------------------------------------------------------------------
def load_sci_style(root: Path) -> None:
    style_path = root / "src" / "sci_plot_style.py"

    if style_path.exists():
        spec = importlib.util.spec_from_file_location(
            "sci_plot_style_branch_ablation_rich",
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
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "axes.unicode_minus": False,
        }
    )


def save_both(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)

    png = Path(os.path.abspath(os.path.normpath(str(out_base.with_suffix(".png")))))
    pdf = Path(os.path.abspath(os.path.normpath(str(out_base.with_suffix(".pdf")))))

    fig.savefig(str(png), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(str(pdf), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[SAVED] {png}")
    print(f"[SAVED] {pdf}")


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def load_data(input_csv: Path | None) -> pd.DataFrame:
    if input_csv is None:
        print("[INFO] No --input-csv provided. Using editable DEFAULT_ROWS.")
        df = pd.DataFrame(DEFAULT_ROWS)
    else:
        if not input_csv.exists():
            raise FileNotFoundError(input_csv)
        df = pd.read_csv(input_csv)

    required = {"variant", "mean", "std"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing columns in ablation source: {missing}\n"
            f"Required columns: {sorted(required)}"
        )

    df = df.copy()
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df["std"] = pd.to_numeric(df["std"], errors="coerce")
    df = df.dropna(subset=["mean", "std"])

    # Keep desired order if present.
    rank = {name: i for i, name in enumerate(ORDER)}
    df["rank"] = df["variant"].map(lambda x: rank.get(str(x), 999))
    df = df.sort_values(["rank", "variant"]).drop(columns="rank")

    return df


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
def plot_rich_ablation(
    df: pd.DataFrame,
    ridge_value: float,
    output_dir: Path,
) -> None:
    variants = df["variant"].tolist()
    means = df["mean"].to_numpy(dtype=float)
    stds = df["std"].to_numpy(dtype=float)

    # Put top item at top of figure.
    y_positions = np.arange(len(variants))[::-1]

    nonlinear_low = float(np.min(means - 2.5 * stds))
    nonlinear_high = float(np.max(means + 2.5 * stds))
    nonlinear_span = max(nonlinear_high - nonlinear_low, 1e-6)
    ridge_gap = ridge_value - nonlinear_high

    use_broken = ridge_gap > 2.0 * nonlinear_span

    if use_broken:
        fig = plt.figure(figsize=(7.35, 3.90))
        gs = GridSpec(
            1,
            2,
            figure=fig,
            width_ratios=[5.0, 1.15],
            wspace=0.05,
            left=0.31,
            right=0.985,
            bottom=0.18,
            top=0.95,
        )
        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)
        axes = [ax_left, ax_right]

        left_pad = max(0.0008, 0.12 * nonlinear_span)
        ax_left.set_xlim(nonlinear_low - left_pad, nonlinear_high + left_pad)

        ridge_pad = max(0.0016, 0.012 * ridge_value)
        ax_right.set_xlim(ridge_value - ridge_pad, ridge_value + ridge_pad)

        # break marks
        d = 0.012
        kwargs = dict(transform=ax_left.transAxes, color="black", clip_on=False, linewidth=0.8)
        ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        kwargs.update(transform=ax_right.transAxes)
        ax_right.plot((-d, +d), (-d, +d), **kwargs)
        ax_right.plot((-d, +d), (1 - d, 1 + d), **kwargs)

        ax_left.spines["right"].set_visible(False)
        ax_right.spines["left"].set_visible(False)
        ax_left.tick_params(right=False)
        ax_right.tick_params(left=False, labelleft=False)

    else:
        fig, ax_left = plt.subplots(figsize=(6.30, 3.90))
        fig.subplots_adjust(left=0.35, right=0.985, bottom=0.18, top=0.95)
        ax_right = None
        axes = [ax_left]

        all_low = min(nonlinear_low, ridge_value)
        all_high = max(nonlinear_high, ridge_value)
        span = max(all_high - all_low, 0.004)
        ax_left.set_xlim(all_low - 0.08 * span, all_high + 0.08 * span)

    blue = "#1f77b4"
    red = "#d62728"

    for y, variant, mean, std in zip(y_positions, variants, means, stds):
        is_proposed = "physics-compact" in variant.lower()

        color = red if is_proposed else blue
        msize = 6.4 if is_proposed else 5.7

        ax_left.errorbar(
            mean,
            y,
            xerr=std,
            fmt="o",
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.55,
            markersize=msize,
            elinewidth=0.90,
            capsize=3.0,
            capthick=0.90,
            zorder=5,
        )

    ref_ax = ax_right if ax_right is not None else ax_left
    ref_ax.axvline(
        ridge_value,
        color=blue,
        linewidth=1.0,
        linestyle="--",
        zorder=2,
    )
    ref_ax.text(
        ridge_value,
        y_positions.max() + 0.35,
        f"Ridge = {ridge_value:.4f}",
        ha="center",
        va="bottom",
        fontsize=7.3,
        color="black",
    )

    label_map = {
        "Physics-Compact (proposed)": r"$\bf{Physics}$-$\bf{Compact}$ (proposed)",
        "Physics-vessel-only residual": "Physics-vessel-only residual",
        "Physics-full residual": "Physics-full residual",
        "Vessel-only residual": "Vessel-only residual",
        "Full residual": "Full residual",
        "Compact-vessel residual": "Compact-vessel residual",
    }

    ax_left.set_yticks(y_positions)
    ax_left.set_yticklabels([label_map.get(v, v) for v in variants])

    for tick, raw_label in zip(ax_left.get_yticklabels(), variants):
        if "Physics-Compact" in raw_label:
            tick.set_fontweight("bold")

    ax_left.set_ylim(-0.55, len(variants) - 0.45)

    # same clean style as the old figure
    ax_left.grid(True, axis="y", linewidth=0.40, alpha=0.18)
    ax_left.minorticks_on()
    ax_left.tick_params(which="both", direction="in", top=True)

    if ax_right is not None:
        ax_right.grid(True, axis="y", linewidth=0.40, alpha=0.18)
        ax_right.minorticks_on()
        ax_right.tick_params(which="both", direction="in", top=True, right=True)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    fig.supxlabel(
        r"Validation apparent-wind vector RMSE (m s$^{-1}$)",
        y=0.055,
        fontsize=8.5,
    )
    ax_left.set_ylabel("Variant")

    save_both(fig, output_dir / "Fig_branch_ablation_rich")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
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
        help="Optional CSV with columns: variant, mean, std",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=DEFAULT_RIDGE,
        help="Dual Ridge reference value shown by the dashed vertical line.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "figures" / "final_paper"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    load_sci_style(root)

    print("=" * 100)
    print("RICH BRANCH ABLATION FIGURE")
    print("=" * 100)
    print(f"root      : {root}")
    print(f"input_csv : {args.input_csv}")
    print(f"ridge     : {args.ridge:.6f}")
    print(f"output    : {output_dir}")
    print("=" * 100)

    df = load_data(args.input_csv)
    print("[SOURCE]")
    print(df.to_string(index=False))

    # Save exact source used.
    df.to_csv(output_dir / "Fig_branch_ablation_rich_source.csv", index=False)

    plot_rich_ablation(df, args.ridge, output_dir)

    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
