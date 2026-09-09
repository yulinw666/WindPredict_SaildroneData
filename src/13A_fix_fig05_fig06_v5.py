# -*- coding: utf-8 -*-
"""
13A_fix_fig05_fig06_v5.py

Final-layout redraw for manuscript Fig05 and Fig06.

Fig05
-----
Use a broken x-axis:
- left panel: zoomed residual variants (where all six error bars lie)
- right narrow panel: Ridge reference
This removes the large empty region between ~1.08 and ~1.125.
No numeric text is drawn on top of error bars because the exact values are
already reported in the manuscript table.

Fig06
-----
Use:
- one color per dataset
- marker AREA proportional to correction magnitude |c|
- no text annotations inside the scatter panel
- a compact, aligned right-side key containing only:
      dataset, gamma, |c|
  Ridge skill and SSE reduction are NOT repeated in the key because they are
  already encoded by the x/y coordinates.

No training. Plotting only.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


DEFAULT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")


# =============================================================================
# Utility
# =============================================================================

def norm_text(s) -> str:
    s = str(s).strip().lower()
    return re.sub(r"[\s_\-()/\[\],.:;]+", "", s)


def try_read(path: Path, tag: str):
    if not path.exists():
        print(f"[WARN] {tag} not found: {path}")
        return None
    print(f"[READ] {tag}: {path}")
    return pd.read_csv(path)


def find_col(df: pd.DataFrame, candidates):
    cmap = {norm_text(c): c for c in df.columns}
    for cand in candidates:
        key = norm_text(cand)
        if key in cmap:
            return cmap[key]
    return None


def apply_style(root: Path):
    style_path = root / "src" / "sci_plot_style.py"
    if style_path.exists():
        try:
            spec = importlib.util.spec_from_file_location(
                "sci_plot_style_fig56v5",
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
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    if grid:
        ax.grid(True, alpha=0.20, linewidth=0.6)


def save_figure(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {png}")
    print(f"[SAVED] {pdf}")


# =============================================================================
# Fig05 data
# =============================================================================

def fallback_fig05():
    data = [
        ("Physics-Compact (proposed)", 1.078514, 0.003204),
        ("Physics-vessel-only residual", 1.078910, 0.001314),
        ("Physics-full residual", 1.079015, 0.002574),
        ("Vessel-only residual", 1.079438, 0.000844),
        ("Full residual", 1.080020, 0.001212),
        ("Compact-vessel residual", 1.080117, 0.003441),
    ]
    return pd.DataFrame(data, columns=["label", "mean", "std"]), 1.125203


def extract_fig05(root: Path):
    base = (
        root / "data" / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
    )
    p07c = base / "confirm_and_compact_vessel_residual_v0_1" / "seed_summary.csv"
    p07b = base / "structured_residual_ablation_v0_1" / "structured_ablation_metrics_summary.csv"

    a = try_read(p07c, "07C seed_summary")
    b = try_read(p07b, "07B structured_ablation_metrics_summary")

    if a is None or b is None:
        return fallback_fig05()

    try:
        if "split" in a.columns:
            a = a[a["split"].astype(str).str.lower().eq("validation")]
        if "split" in b.columns:
            b = b[b["split"].astype(str).str.lower().eq("validation")]

        rows = []
        mapping = [
            ("Physics-Compact-Vessel-Residual", "Physics-Compact (proposed)"),
            ("Physics-Vessel-Only-Residual", "Physics-vessel-only residual"),
            ("Physics-Full-Residual", "Physics-full residual"),
            ("Vessel-Only-Residual", "Vessel-only residual"),
            ("Full-Residual", "Full residual"),
            ("Compact-Vessel-Residual", "Compact-vessel residual"),
        ]

        for raw, label in mapping:
            q = a[a["model"].eq(raw)]
            if q.empty:
                raise RuntimeError(f"missing model {raw}")
            r = q.iloc[0]
            rows.append(
                (label, float(r["mean_AW_RMSE_mps"]), float(r["std_AW_RMSE_mps"]))
            )

        ridge = b[b["model"].eq("Ridge")]
        if ridge.empty:
            raise RuntimeError("missing Ridge")
        ridge_aw = float(ridge.iloc[0]["mean_apparent_vector_RMSE_mps"])

        return pd.DataFrame(rows, columns=["label", "mean", "std"]), ridge_aw
    except Exception as exc:
        print(f"[WARN] Fig05 parse failed: {exc}; using frozen fallback values.")
        return fallback_fig05()


# =============================================================================
# Fig05 broken-axis plot
# =============================================================================

def make_fig05(root: Path, outdir: Path):
    df, ridge = extract_fig05(root)

    # Manuscript order: best at top.
    df = df.reset_index(drop=True)
    y = np.arange(len(df))[::-1]

    fig = plt.figure(figsize=(7.2, 4.1))
    gs = GridSpec(
        1, 2,
        width_ratios=[5.0, 1.15],
        wspace=0.06,
        figure=fig,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)

    proposed_color = "#d62728"
    normal_color = "#1f77b4"

    for yi, row in zip(y, df.itertuples(index=False)):
        color = proposed_color if "proposed" in row.label.lower() else normal_color
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

    # Ridge lives only in the right panel.
    ax2.axvline(
        ridge,
        linestyle="--",
        linewidth=1.15,
        color=normal_color,
        label=f"Ridge = {ridge:.4f}",
    )

    # Zoomed residual window and Ridge window.
    left_min = float(np.min(df["mean"] - df["std"])) - 0.0008
    left_max = float(np.max(df["mean"] + df["std"])) + 0.0008
    ax1.set_xlim(left_min, left_max)
    ax2.set_xlim(ridge - 0.003, ridge + 0.003)

    ax1.set_yticks(y)
    ax1.set_yticklabels(df["label"])
    for t in ax1.get_yticklabels():
        if "proposed" in t.get_text().lower():
            t.set_fontweight("bold")

    # no duplicated y labels on right
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.tick_params(labelleft=False)

    ax1.set_ylabel("Variant")
    fig.supxlabel(
        r"Validation apparent-wind vector RMSE (m s$^{-1}$)",
        y=0.02,
    )

    finish_axis(ax1)
    finish_axis(ax2)

    # broken-axis visual marks
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.tick_params(right=False)
    ax2.tick_params(left=False)

    d = 0.012
    kwargs1 = dict(transform=ax1.transAxes, color="k", clip_on=False, linewidth=0.8)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs1)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs1)

    kwargs2 = dict(transform=ax2.transAxes, color="k", clip_on=False, linewidth=0.8)
    ax2.plot((-d, +d), (-d, +d), **kwargs2)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs2)

    ax2.legend(loc="lower center", frameon=False)

    fig.subplots_adjust(left=0.31, right=0.98, bottom=0.18, top=0.98)
    save_figure(fig, outdir / "Fig05_ablation_validation_AW_RMSE_v5")

    out = df.copy()
    out["ridge_reference"] = ridge
    out.to_csv(
        outdir / "Fig05_ablation_validation_AW_RMSE_v5_data.csv",
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# Fig06 data
# =============================================================================

def fallback_fig06():
    rows = [
        ("SD1090 validation", 3.76, 31.71, 0.478, 0.3069),
        ("SD1033-2024 matched", 4.62, 29.69, 0.446, 0.2729),
        ("SD1033-2023", -50.90, 58.08, 0.722, 0.5153),
        ("SD1033-2022", -5.34, 39.20, 0.567, 0.3326),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "label",
            "ridge_skill_pct",
            "sse_reduction_pct",
            "gamma",
            "corr_mag",
        ],
    )


def extract_fig06(root: Path):
    mech = (
        root / "data" / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
        / "10A_mechanism_analysis_v0_1"
    )

    ridge = try_read(
        mech / "ridge_branch_residual_diagnostics.csv",
        "10A ridge_branch_residual_diagnostics",
    )
    key = try_read(
        mech / "mechanism_key_findings.csv",
        "10A mechanism_key_findings",
    )

    if ridge is None or key is None:
        return fallback_fig06()

    try:
        vessel = ridge[
            ridge["branch"].astype(str).str.lower().eq("vessel")
        ].copy()
        vessel = (
            vessel
            .groupby("dataset_id", as_index=False)["ridge_skill_vs_persistence"]
            .mean()
        )
        vessel["ridge_skill_pct"] = 100.0 * vessel["ridge_skill_vs_persistence"]

        cols = [
            "dataset_id",
            "mean_PhysicsCompact_correction_RMS_mps",
            "mean_correction_cosine_alignment",
            "mean_vessel_SSE_reduction_fraction_vs_Ridge",
        ]
        k = key[cols].copy()
        k["sse_reduction_pct"] = (
            100.0 * k["mean_vessel_SSE_reduction_fraction_vs_Ridge"]
        )

        df = vessel.merge(k, on="dataset_id", how="inner")

        label_map = {
            "SD1090_validation": "SD1090 validation",
            "SD1033_2024_matched_SD1090": "SD1033-2024 matched",
            "SD1033_2023_full": "SD1033-2023",
            "SD1033_2022_full": "SD1033-2022",
        }

        df["label"] = df["dataset_id"].map(label_map)
        df["gamma"] = df["mean_correction_cosine_alignment"]
        df["corr_mag"] = df["mean_PhysicsCompact_correction_RMS_mps"]

        order = [
            "SD1090 validation",
            "SD1033-2024 matched",
            "SD1033-2023",
            "SD1033-2022",
        ]
        df["ord"] = df["label"].map({v: i for i, v in enumerate(order)})
        df = df.sort_values("ord")

        return df[
            ["label", "ridge_skill_pct", "sse_reduction_pct", "gamma", "corr_mag"]
        ].reset_index(drop=True)
    except Exception as exc:
        print(f"[WARN] Fig06 parse failed: {exc}; using frozen fallback values.")
        return fallback_fig06()


# =============================================================================
# Fig06 scatter + compact aligned key
# =============================================================================

def make_fig06(root: Path, outdir: Path):
    df = extract_fig06(root)

    color_map = {
        "SD1090 validation": "#1f77b4",
        "SD1033-2024 matched": "#ff7f0e",
        "SD1033-2023": "#2ca02c",
        "SD1033-2022": "#d62728",
    }

    # Marker area proportional to correction magnitude.
    # Keep ratio visible but not overwhelming.
    cmin = float(df["corr_mag"].min())
    cmax = float(df["corr_mag"].max())
    if cmax > cmin:
        df["marker_area"] = 90.0 + 210.0 * (df["corr_mag"] - cmin) / (cmax - cmin)
    else:
        df["marker_area"] = 150.0

    fig = plt.figure(figsize=(8.0, 4.1))
    gs = GridSpec(
        1, 2,
        width_ratios=[1.9, 1.10],
        wspace=0.10,
        figure=fig,
    )
    ax = fig.add_subplot(gs[0, 0])
    info = fig.add_subplot(gs[0, 1])

    # Main plot: only points + zero reference.
    for row in df.itertuples(index=False):
        ax.scatter(
            row.ridge_skill_pct,
            row.sse_reduction_pct,
            s=row.marker_area,
            color=color_map[row.label],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        color="#1f77b4",
        zorder=1,
    )

    ax.set_xlabel("Ridge vessel skill vs Persistence (%)")
    ax.set_ylabel("Physics-Compact vessel SSE reduction (%)")

    ax.set_xlim(
        min(-55.0, float(df["ridge_skill_pct"].min()) - 4.0),
        max(8.0, float(df["ridge_skill_pct"].max()) + 3.0),
    )
    ax.set_ylim(
        min(27.0, float(df["sse_reduction_pct"].min()) - 2.0),
        max(60.5, float(df["sse_reduction_pct"].max()) + 2.0),
    )
    finish_axis(ax)

    # -----------------------------------------------------------------
    # Right-side compact aligned key.
    # Only show gamma and |c| because x/y already show skill and SSE.
    # -----------------------------------------------------------------
    info.axis("off")
    info.set_xlim(0, 1)
    info.set_ylim(0, 1)

    # Header
    info.text(
        0.02, 0.96,
        "Dataset",
        ha="left", va="top",
        fontsize=8.2, fontweight="bold",
    )
    info.text(
        0.73, 0.96,
        r"$\gamma$",
        ha="center", va="top",
        fontsize=8.2, fontweight="bold",
    )
    info.text(
        0.93, 0.96,
        r"$|c|$",
        ha="center", va="top",
        fontsize=8.2, fontweight="bold",
    )
    info.plot([0.02, 0.98], [0.90, 0.90], color="black", linewidth=0.7)

    row_y = [0.79, 0.62, 0.45, 0.28]

    for yy, row in zip(row_y, df.itertuples(index=False)):
        color = color_map[row.label]

        info.scatter(
            0.055, yy,
            s=55,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            clip_on=False,
        )
        info.text(
            0.11, yy,
            row.label,
            ha="left", va="center",
            fontsize=7.8,
        )
        info.text(
            0.73, yy,
            f"{row.gamma:.2f}",
            ha="center", va="center",
            fontsize=7.8,
        )
        info.text(
            0.93, yy,
            f"{row.corr_mag:.2f}",
            ha="center", va="center",
            fontsize=7.8,
        )

    # Units and compact legend notes below the table.
    info.text(
        0.93, 0.16,
        r"m s$^{-1}$",
        ha="center", va="center",
        fontsize=7.2,
    )

    info.plot([0.02, 0.98], [0.20, 0.20], color="black", linewidth=0.6)

    info.text(
        0.02, 0.12,
        r"Marker area $\propto |c|$",
        ha="left", va="center",
        fontsize=7.2,
    )
    info.text(
        0.02, 0.055,
        "Dashed line: Ridge = Persistence",
        ha="left", va="center",
        fontsize=7.2,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.17,
        top=0.98,
    )

    save_figure(fig, outdir / "Fig06_mechanism_vessel_shift_correction_v5")

    df.to_csv(
        outdir / "Fig06_mechanism_vessel_shift_correction_v5_data.csv",
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.project_root
    outdir = (
        args.output_dir
        if args.output_dir is not None
        else (
            root / "data" / "forecasting"
            / "13A_final_manuscript_figures_v0_1"
        )
    )
    outdir.mkdir(parents=True, exist_ok=True)

    apply_style(root)

    print("=" * 100)
    print("REDRAW FIG05 + FIG06 — v5")
    print("=" * 100)

    make_fig05(root, outdir)
    make_fig06(root, outdir)

    print("\n[DONE] Fig05 v5 and Fig06 v5 generated.")


if __name__ == "__main__":
    main()
