# -*- coding: utf-8 -*-
"""
13A_fix_fig06_v3.py

Redraw ONLY Fig06 using the correct 10A sources.

The previous v2 failed because it tried to read Ridge vessel skill from:
    physics_vs_compact_regime_effect_summary.csv
That file does not contain Ridge branch skill.

The correct source is:
    ridge_branch_residual_diagnostics.csv

This script is plotting-only:
- no training
- no model selection
- no metric recomputation from raw data
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


DEFAULT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")


def apply_style(root: Path):
    style_path = root / "src" / "sci_plot_style.py"
    if style_path.exists():
        try:
            spec = importlib.util.spec_from_file_location(
                "sci_plot_style_fig06v3",
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
                        print(f"[STYLE] {style_path.name}:{name}{kwargs}")
                        return
                    except TypeError:
                        continue
        except Exception as exc:
            print(f"[STYLE WARNING] {type(exc).__name__}: {exc}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
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


def finish_axis(ax):
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.grid(True, alpha=0.20, linewidth=0.6)


def save_figure(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {base.with_suffix('.png')}")
    print(f"[SAVED] {base.with_suffix('.pdf')}")


def read_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n{path}")
    print(f"[READ] {label}: {path}")
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.project_root
    outdir = (
        args.output_dir
        if args.output_dir is not None
        else root / "data" / "forecasting" / "13A_final_manuscript_figures_v0_1"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    apply_style(root)

    mech_dir = (
        root
        / "data"
        / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
        / "10A_mechanism_analysis_v0_1"
    )

    ridge_path = mech_dir / "ridge_branch_residual_diagnostics.csv"
    key_path = mech_dir / "mechanism_key_findings.csv"

    ridge = read_required(ridge_path, "10A ridge_branch_residual_diagnostics")
    key = read_required(key_path, "10A mechanism_key_findings")

    required_ridge = {
        "dataset_id",
        "branch",
        "ridge_skill_vs_persistence",
    }
    missing = required_ridge - set(ridge.columns)
    if missing:
        raise RuntimeError(
            f"ridge_branch_residual_diagnostics.csv missing columns: {sorted(missing)}"
        )

    required_key = {
        "dataset_id",
        "mean_PhysicsCompact_correction_RMS_mps",
        "mean_correction_cosine_alignment",
        "mean_vessel_SSE_reduction_fraction_vs_Ridge",
    }
    missing = required_key - set(key.columns)
    if missing:
        raise RuntimeError(
            f"mechanism_key_findings.csv missing columns: {sorted(missing)}"
        )

    # Mean Ridge vessel skill across the five horizons.
    vessel = ridge[
        ridge["branch"].astype(str).str.lower().eq("vessel")
    ].copy()

    vessel = (
        vessel
        .groupby("dataset_id", as_index=False)["ridge_skill_vs_persistence"]
        .mean()
    )

    # Stored as fraction -> percentage.
    vessel["ridge_vessel_skill_pct"] = (
        100.0 * vessel["ridge_skill_vs_persistence"].astype(float)
    )

    df = vessel.merge(key, on="dataset_id", how="inner", validate="one_to_one")

    if df.empty:
        raise RuntimeError("No matching dataset IDs between Ridge diagnostics and key findings.")

    label_map = {
        "SD1090_validation": "SD1090 validation",
        "SD1033_2024_matched_SD1090": "SD1033-2024 matched",
        "SD1033_2023_full": "SD1033-2023",
        "SD1033_2022_full": "SD1033-2022",
    }
    df["label"] = df["dataset_id"].map(label_map).fillna(df["dataset_id"])

    # Keep a deterministic manuscript order.
    order = [
        "SD1090_validation",
        "SD1033_2024_matched_SD1090",
        "SD1033_2023_full",
        "SD1033_2022_full",
    ]
    rank = {k: i for i, k in enumerate(order)}
    df["_order"] = df["dataset_id"].map(rank).fillna(999)
    df = df.sort_values("_order").reset_index(drop=True)

    x = df["ridge_vessel_skill_pct"].to_numpy(float)
    y = (
        100.0
        * df["mean_vessel_SSE_reduction_fraction_vs_Ridge"].to_numpy(float)
    )
    corr_mag = df["mean_PhysicsCompact_correction_RMS_mps"].to_numpy(float)
    align = df["mean_correction_cosine_alignment"].to_numpy(float)

    # Use only modest size variation.
    if np.ptp(corr_mag) > 0:
        sizes = 85.0 + 120.0 * (corr_mag - corr_mag.min()) / np.ptp(corr_mag)
    else:
        sizes = np.full_like(corr_mag, 120.0)

    print("\n[DATA USED FOR FIG06]")
    for _, r in df.iterrows():
        print(
            f"{r['label']:<24s} | "
            f"Ridge vessel skill={r['ridge_vessel_skill_pct']:+7.2f}% | "
            f"SSE reduction={100*r['mean_vessel_SSE_reduction_fraction_vs_Ridge']:6.2f}% | "
            f"alignment={r['mean_correction_cosine_alignment']:.3f} | "
            f"correction={r['mean_PhysicsCompact_correction_RMS_mps']:.4f} m/s"
        )

    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    ax.scatter(x, y, s=sizes, zorder=3)

    # Reference line: Ridge equal to Persistence.
    ax.axvline(0.0, linestyle="--", linewidth=1.0, zorder=1)

    # Carefully separated manuscript annotations.
    # xytext is in POINTS, so it remains stable across image dimensions.
    annotation_offsets = {
        "SD1090 validation": (18, 16),
        "SD1033-2024 matched": (18, -23),
        "SD1033-2023": (18, 5),
        "SD1033-2022": (18, 7),
    }

    for xi, yi, lab, ai, ci in zip(
        x, y, df["label"], align, corr_mag
    ):
        dx, dy = annotation_offsets.get(lab, (15, 8))

        txt = (
            f"{lab}\n"
            + rf"$\gamma$={ai:.2f}, $|c|$={ci:.2f} m s$^{{-1}}$"
        )

        ax.annotate(
            txt,
            xy=(xi, yi),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left",
            va="bottom" if dy >= 0 else "top",
            fontsize=7.4,
            arrowprops=dict(
                arrowstyle="-",
                linewidth=0.65,
                shrinkA=1,
                shrinkB=4,
            ),
            annotation_clip=False,
        )

    # Deliberately leave room for the labels on both sides.
    x_min = min(-55.0, float(np.min(x)) - 4.0)
    x_max = max(12.0, float(np.max(x)) + 9.0)
    y_min = min(27.5, float(np.min(y)) - 2.5)
    y_max = max(61.0, float(np.max(y)) + 3.0)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Ridge vessel skill vs Persistence (%)")
    ax.set_ylabel("Physics-Compact vessel SSE reduction (%)")

    # Explanation of marker area without adding a bulky legend.
    ax.text(
        0.02,
        0.035,
        r"Marker area $\propto$ correction magnitude $|c|$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
    )

    finish_axis(ax)
    fig.tight_layout()

    base = outdir / "Fig06_mechanism_vessel_shift_correction_v3"
    save_figure(fig, base)

    # Save exactly what was plotted, useful for manuscript audit.
    audit = df[[
        "dataset_id",
        "label",
        "ridge_vessel_skill_pct",
        "mean_vessel_SSE_reduction_fraction_vs_Ridge",
        "mean_correction_cosine_alignment",
        "mean_PhysicsCompact_correction_RMS_mps",
    ]].copy()
    audit["vessel_SSE_reduction_pct"] = (
        100.0 * audit["mean_vessel_SSE_reduction_fraction_vs_Ridge"]
    )
    audit.to_csv(
        outdir / "Fig06_mechanism_v3_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"[SAVED] {outdir / 'Fig06_mechanism_v3_data.csv'}")

    print("\n[DONE] Fig06 v3 generated successfully.")


if __name__ == "__main__":
    main()
