# -*- coding: utf-8 -*-
"""
13A_fix_fig05_fig06_v2.py

Purpose
-------
Redraw Fig05 and Fig06 with cleaner manuscript-quality layouts.

Why this revision?
------------------
- Fig05 in the previous draft looked visually odd because the six variants were
  packed into a very narrow vertical range with crowded x tick labels.
- Fig06 had overlapping text annotations and oversized markers, which reduced
  readability.

This script:
1) redraws Fig05 as a horizontal point-range plot;
2) redraws Fig06 as a scatter plot with manual label placement + callout arrows.

No training.
No model selection.
Only plotting from existing frozen CSV outputs.
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
    style_module = None
    if style_path.exists():
        try:
            spec = importlib.util.spec_from_file_location(
                "sci_plot_style_fix56",
                str(style_path),
            )
            style_module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(style_module)

            for name, kwargs in [
                ("apply_sci_style", {"base_font_size": 8.5}),
                ("apply_sci_style", {"font_size": 8.5}),
                ("apply_sci_style", {}),
                ("apply_sci_plot_style", {}),
                ("set_sci_plot_style", {}),
            ]:
                fn = getattr(style_module, name, None)
                if callable(fn):
                    try:
                        fn(**kwargs)
                        print(f"[STYLE] {style_path.name}:{name}{kwargs}")
                        return style_module
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
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return None


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


def find_named(root: Path, filename: str, prefer_tokens=()):
    hits = [p for p in root.rglob(filename) if p.is_file()]
    if not hits:
        return None

    def score(p: Path):
        s = 0
        low = str(p).lower()
        for token in prefer_tokens:
            if token.lower() in low:
                s += 10
        return (s, -len(str(p)))

    hits.sort(key=score, reverse=True)
    return hits[0]


def read_required(path: Path | None, label: str) -> pd.DataFrame:
    if path is None or not path.exists():
        raise FileNotFoundError(f"{label} not found")
    print(f"[READ] {label}: {path}")
    return pd.read_csv(path)


def make_fig05(root: Path, outdir: Path):
    dataset_dir = (
        root / "data" / "forecasting" / "SD1090_TPOS2024_JointForecasting_v0_1"
    )

    p07c = find_named(
        dataset_dir,
        "seed_summary.csv",
        prefer_tokens=("confirm_and_compact", "07C"),
    )
    p07b = find_named(
        dataset_dir,
        "structured_ablation_metrics_summary.csv",
        prefer_tokens=("structured_residual_ablation", "07B"),
    )

    df07c = read_required(p07c, "07C seed_summary")
    df07b = read_required(p07b, "07B structured_ablation_metrics_summary")

    if "split" in df07c.columns:
        df07c = df07c[df07c["split"].astype(str).str.lower().eq("validation")]
    if "split" in df07b.columns:
        df07b = df07b[df07b["split"].astype(str).str.lower().eq("validation")]

    mapping = {
        "Full-Residual": "Full residual",
        "Vessel-Only-Residual": "Vessel-only residual",
        "Physics-Full-Residual": "Physics-full residual",
        "Physics-Vessel-Only-Residual": "Physics-vessel-only residual",
        "Compact-Vessel-Residual": "Compact-vessel residual",
        "Physics-Compact-Vessel-Residual": "Physics-Compact (proposed)",
    }

    rows = []
    for raw_model, label in mapping.items():
        sub = df07c[df07c["model"].eq(raw_model)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        rows.append({
            "label": label,
            "mean_aw": float(r["mean_AW_RMSE_mps"]),
            "std_aw": float(r["std_AW_RMSE_mps"]),
        })

    plot_df = pd.DataFrame(rows).sort_values("mean_aw", ascending=False).reset_index(drop=True)
    ridge = df07b[df07b["model"].eq("Ridge")]
    ridge_aw = float(ridge.iloc[0]["mean_apparent_vector_RMSE_mps"])

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    y = np.arange(len(plot_df))

    ax.errorbar(
        plot_df["mean_aw"],
        y,
        xerr=plot_df["std_aw"],
        fmt="o",
        capsize=3,
        linestyle="none",
        label="Residual variant",
    )

    ax.axvline(
        ridge_aw,
        linestyle="--",
        linewidth=1.1,
        label=f"Ridge = {ridge_aw:.3f}",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel(r"Validation apparent-wind vector RMSE (m s$^{-1}$)")
    ax.set_ylabel("Variant")

    # Add value labels.
    xspan = plot_df["mean_aw"].max() - plot_df["mean_aw"].min()
    xpad = max(0.0012, 0.10 * xspan)
    for xi, yi, si in zip(plot_df["mean_aw"], y, plot_df["std_aw"]):
        ax.text(
            xi + xpad,
            yi,
            f"{xi:.4f} ± {si:.4f}",
            va="center",
            ha="left",
            fontsize=7.2,
        )

    xmin = min(plot_df["mean_aw"].min() - 0.004, ridge_aw - 0.006)
    xmax = max(plot_df["mean_aw"].max() + 0.012, ridge_aw + 0.004)
    ax.set_xlim(xmin, xmax)

    finish_axis(ax)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save_figure(fig, outdir / "Fig05_ablation_validation_AW_RMSE_v2")


def make_fig06(root: Path, outdir: Path):
    forecast_root = (
        root / "data" / "forecasting" / "SD1090_TPOS2024_JointForecasting_v0_1"
    )
    mechanism_dir = forecast_root / "10A_mechanism_analysis_v0_1"

    p_key = find_named(mechanism_dir, "mechanism_key_findings.csv", prefer_tokens=("10A_mechanism",))
    p_regime = find_named(mechanism_dir, "physics_vs_compact_regime_effect_summary.csv", prefer_tokens=("10A_mechanism",))

    key = read_required(p_key, "10A mechanism_key_findings")
    reg = read_required(p_regime, "10A physics_vs_compact_regime_effect_summary")

    # Expect at least:
    # key: dataset_id, mean_vessel_SSE_reduction_fraction_vs_Ridge,
    #      mean_PhysicsCompact_correction_RMS_mps, mean_correction_cosine_alignment
    # reg: dataset_id, ridge_vessel_skill_vs_persistence_pct
    possible_skill_cols = [
        "ridge_vessel_skill_vs_persistence_pct",
        "Ridge_vessel_skill_vs_Persistence_percent",
        "ridge_vessel_skill_percent",
    ]
    skill_col = None
    for c in possible_skill_cols:
        if c in reg.columns:
            skill_col = c
            break
    if skill_col is None:
        # fallback: maybe a column already named descriptively
        candidates = [c for c in reg.columns if "ridge" in c.lower() and "skill" in c.lower()]
        if not candidates:
            raise RuntimeError("No Ridge vessel skill column found in mechanism/regime summary.")
        skill_col = candidates[0]

    merged = key.merge(reg[["dataset_id", skill_col]], on="dataset_id", how="inner")
    merged = merged.rename(columns={skill_col: "ridge_skill_pct"})
    merged = merged.copy()

    # Clean labels
    label_map = {
        "SD1090_validation": "SD1090 validation",
        "SD1033_2024_matched_SD1090": "SD1033-2024 matched",
        "SD1033_2023_full": "SD1033-2023",
        "SD1033_2022_full": "SD1033-2022",
    }
    merged["label"] = merged["dataset_id"].map(label_map).fillna(merged["dataset_id"])

    x = merged["ridge_skill_pct"].to_numpy(float)
    y = 100.0 * merged["mean_vessel_SSE_reduction_fraction_vs_Ridge"].to_numpy(float)
    corr_mag = merged["mean_PhysicsCompact_correction_RMS_mps"].to_numpy(float)
    align = merged["mean_correction_cosine_alignment"].to_numpy(float)

    # Moderate marker sizes to avoid overwhelming the panel
    if np.ptp(corr_mag) > 0:
        sizes = 120.0 + 380.0 * (corr_mag - corr_mag.min()) / np.ptp(corr_mag)
    else:
        sizes = np.full_like(corr_mag, 220.0)

    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    ax.scatter(x, y, s=sizes, alpha=0.90)

    ax.axvline(0.0, linestyle="--", linewidth=1.1)

    # Manual annotation offsets to avoid overlap
    offsets = {
        "SD1033-2023": (12, 12),
        "SD1033-2022": (12, 8),
        "SD1090 validation": (12, 18),
        "SD1033-2024 matched": (12, -20),
    }

    for xi, yi, lab, ai, ci in zip(x, y, merged["label"], align, corr_mag):
        dx, dy = offsets.get(lab, (10, 8))
        txt = f"{lab}\n" + rf"$\gamma$={ai:.2f}, $|c|$={ci:.2f}"
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
                lw=0.7,
                shrinkA=0,
                shrinkB=4,
            ),
        )

    # Give extra room on the right and top for labels
    xpad_left = 3.0
    xpad_right = 8.0
    ypad = 2.0
    ax.set_xlim(np.min(x) - xpad_left, np.max(x) + xpad_right)
    ax.set_ylim(np.min(y) - 1.5, np.max(y) + ypad)

    ax.set_xlabel("Ridge vessel skill vs Persistence (%)")
    ax.set_ylabel("Physics-Compact vessel SSE reduction (%)")

    # Small in-panel note for marker meaning
    ax.text(
        0.02,
        0.03,
        "Marker size ∝ correction magnitude $|c|$",
        transform=ax.transAxes,
        fontsize=7.2,
        ha="left",
        va="bottom",
    )

    finish_axis(ax)
    fig.tight_layout()
    save_figure(fig, outdir / "Fig06_mechanism_vessel_shift_correction_v2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "data" / "forecasting" / "13A_final_manuscript_figures_v0_1",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_style(args.project_root)

    print("=" * 100)
    print("REDRAW FIG05 + FIG06")
    print("=" * 100)
    print(f"project root : {args.project_root}")
    print(f"output dir   : {args.output_dir}")

    make_fig05(args.project_root, args.output_dir)
    make_fig06(args.project_root, args.output_dir)

    print("[DONE] Fig05 and Fig06 v2 completed.")


if __name__ == "__main__":
    main()
