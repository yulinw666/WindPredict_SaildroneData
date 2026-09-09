# -*- coding: utf-8 -*-
"""
13A_fix_fig05_fig06_v4.py

Purpose
-------
Redraw Fig05 and Fig06 for the manuscript with cleaner layout:

Fig05:
    Residual-ablation validation apparent-wind RMSE with error bars.
    Numeric values are moved to a dedicated right-side value column
    so that they do NOT overlap the plotted error bars.

Fig06:
    Mechanism-analysis scatter plot redesigned with:
    - one color per dataset
    - no long overlapping annotations inside the plot
    - a separate right-side information panel listing the dataset-specific
      metrics associated with each color

Style
-----
Strictly tries to use:
    src/sci_plot_style.py
If unavailable, falls back to a Times-like scientific style.

Notes
-----
1) This script only draws figures.
2) It does NOT retrain or change any model.
3) It tries to read your real Stage-07/10 CSV outputs first.
4) If a CSV schema mismatch occurs, it falls back to the numeric values
   already printed in your terminal logs, so you can still generate figures.
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
from matplotlib import transforms as mtransforms


# ======================================================================================
# Paths
# ======================================================================================

DEFAULT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")


# ======================================================================================
# Utility
# ======================================================================================

def norm_text(s) -> str:
    s = str(s).strip().lower()
    s = s.replace("±", "+-")
    s = re.sub(r"[\s_\-()/\[\],.:;]+", "", s)
    return s


def read_required(path: Path, tag: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[MISSING] {tag}: {path}")
    print(f"[READ] {tag}: {path}")
    return pd.read_csv(path)


def try_read(path: Path, tag: str):
    if not path.exists():
        print(f"[WARN] {tag} not found: {path}")
        return None
    print(f"[READ] {tag}: {path}")
    return pd.read_csv(path)


def find_col(df: pd.DataFrame, candidates):
    colmap = {norm_text(c): c for c in df.columns}
    for cand in candidates:
        key = norm_text(cand)
        if key in colmap:
            return colmap[key]
    return None


def apply_style(root: Path):
    style_path = root / "src" / "sci_plot_style.py"
    if style_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("sci_plot_style_dynamic", str(style_path))
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)

            # try a few common signatures
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
            print(f"[STYLE WARNING] sci_plot_style.py load failed: {type(exc).__name__}: {exc}")

    # fallback
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.20,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    print("[STYLE] fallback scientific style applied")


def finish_axis(ax):
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
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


# ======================================================================================
# FIG05
# ======================================================================================

def fallback_fig05_data():
    # from your 07C terminal output
    rows = [
        ("Physics-Compact (proposed)", 1.0785, 0.0032),
        ("Physics-vessel-only residual", 1.0789, 0.0013),
        ("Physics-full residual", 1.0790, 0.0026),
        ("Vessel-only residual", 1.0794, 0.0008),
        ("Full residual", 1.0800, 0.0012),
        ("Compact-vessel residual", 1.0801, 0.0034),
    ]
    ridge = 1.1252
    return pd.DataFrame(rows, columns=["label", "mean", "std"]), ridge


def extract_fig05_from_csv(root: Path):
    """
    Prefer reading:
      07C seed_summary.csv
      07B structured_ablation_metrics_summary.csv
    but allow fallback if schema mismatch occurs.
    """
    dir07c = (
        root / "data" / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
        / "confirm_and_compact_vessel_residual_v0_1"
    )
    dir07b = (
        root / "data" / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
        / "structured_residual_ablation_v0_1"
    )

    seed_summary = try_read(dir07c / "seed_summary.csv", "07C seed_summary")
    ablation_summary = try_read(dir07b / "structured_ablation_metrics_summary.csv", "07B structured_ablation_metrics_summary")

    # If either is absent, use fallback.
    if seed_summary is None or ablation_summary is None:
        print("[WARN] Fig05 using fallback values from terminal output.")
        return fallback_fig05_data()

    try:
        # Flexible column detection
        name_col = find_col(seed_summary, [
            "variant", "model", "name", "method", "configuration"
        ])
        mean_col = find_col(seed_summary, [
            "validation_aw_mean_mps",
            "validation_aw_rmse_mean_mps",
            "mean_validation_aw_rmse_mps",
            "aw_mean_mps",
            "aw_rmse_mean",
            "validation_aw"
        ])
        std_col = find_col(seed_summary, [
            "validation_aw_std_mps",
            "validation_aw_rmse_std_mps",
            "std_validation_aw_rmse_mps",
            "aw_std_mps",
            "aw_rmse_std",
            "validation_aw_std"
        ])

        ridge_name_col = find_col(ablation_summary, ["variant", "model", "name", "method"])
        ridge_aw_col = find_col(ablation_summary, [
            "aw_rmse", "aw_rmse_mps", "validation_aw_rmse", "validation_aw_rmse_mps"
        ])

        if None in [name_col, mean_col, std_col, ridge_name_col, ridge_aw_col]:
            raise RuntimeError("unresolved CSV columns")

        seed_df = seed_summary[[name_col, mean_col, std_col]].copy()
        seed_df.columns = ["name", "mean", "std"]

        ridge_df = ablation_summary[[ridge_name_col, ridge_aw_col]].copy()
        ridge_df.columns = ["name", "aw"]

        # Build robust name map
        wanted = {
            "Physics-Compact (proposed)": [
                "physicscompactvesselresidual",
                "physics-compact-vessel-residual",
                "physicscompact",
                "physicscompactvessel",
            ],
            "Physics-vessel-only residual": [
                "physicsvesselonlyresidual",
                "physics-vessel-only-residual",
                "physicsmaskedvesselonly",
                "physicsvesselonly",
            ],
            "Physics-full residual": [
                "physicsfullresidual",
                "physics-full-residual",
                "physicsfull",
            ],
            "Vessel-only residual": [
                "vesselonlyresidual",
                "vessel-only-residual",
                "maskedvesselonly",
                "vesselonly",
            ],
            "Full residual": [
                "fullresidual",
                "full-residual",
                "full",
            ],
            "Compact-vessel residual": [
                "compactvesselresidual",
                "compact-vessel-residual",
                "compact",
                "compactvessel",
            ],
        }

        def match_row(df, aliases):
            df2 = df.copy()
            df2["_n"] = df2["name"].map(norm_text)
            for a in aliases:
                key = norm_text(a)
                hit = df2[df2["_n"] == key]
                if len(hit) > 0:
                    return hit.iloc[0]
            return None

        rows = []
        for label, aliases in wanted.items():
            row = match_row(seed_df, aliases)
            if row is None:
                raise RuntimeError(f"variant not found for {label}")
            rows.append((label, float(row["mean"]), float(row["std"])))

        # Ridge
        ridge_df["_n"] = ridge_df["name"].map(norm_text)
        ridge_candidates = ["ridge", "ridgebaseline", "frozenridge"]
        ridge_val = None
        for a in ridge_candidates:
            hit = ridge_df[ridge_df["_n"] == norm_text(a)]
            if len(hit) > 0:
                ridge_val = float(hit.iloc[0]["aw"])
                break
        if ridge_val is None:
            raise RuntimeError("ridge row not found")

        out = pd.DataFrame(rows, columns=["label", "mean", "std"])
        return out, ridge_val

    except Exception as exc:
        print(f"[WARN] Fig05 CSV parse failed ({type(exc).__name__}: {exc}). Using fallback values.")
        return fallback_fig05_data()


def make_fig05(root: Path, outdir: Path):
    df, ridge = extract_fig05_from_csv(root)

    # fixed manuscript order
    order = [
        "Physics-Compact (proposed)",
        "Physics-vessel-only residual",
        "Physics-full residual",
        "Vessel-only residual",
        "Full residual",
        "Compact-vessel residual",
    ]
    df["order"] = df["label"].map({k: i for i, k in enumerate(order)})
    df = df.sort_values("order").reset_index(drop=True)

    y = np.arange(len(df))[::-1]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    plt.subplots_adjust(right=0.78)  # reserve right margin for value column

    # colors
    proposed_color = "#d62728"
    normal_color = "#1f77b4"

    for yi, (_, row) in zip(y, df.iterrows()):
        color = proposed_color if row["label"] == "Physics-Compact (proposed)" else normal_color
        ax.errorbar(
            row["mean"], yi,
            xerr=row["std"],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            markersize=6.5,
            zorder=3,
        )

    ax.axvline(ridge, linestyle="--", linewidth=1.0, color=normal_color, label=f"Ridge = {ridge:.4f}", zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    # bold the proposed label
    for t in ax.get_yticklabels():
        if "proposed" in t.get_text().lower():
            t.set_fontweight("bold")

    ax.set_xlabel(r"Validation apparent-wind vector RMSE (m s$^{-1}$)")
    ax.set_ylabel("Variant")

    xmin = min(df["mean"] - df["std"]) - 0.001
    xmax = max(ridge + 0.004, max(df["mean"] + df["std"]) + 0.003)
    ax.set_xlim(xmin, xmax)

    finish_axis(ax)

    # dedicated right-side value column: no overlap with error bars
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        1.01, y[0] + 0.40,
        r"$\mathrm{Mean}\ \pm\ \mathrm{Std}$",
        transform=trans, ha="left", va="center", fontsize=7.8
    )
    for yi, (_, row) in zip(y, df.iterrows()):
        color = proposed_color if row["label"] == "Physics-Compact (proposed)" else "black"
        weight = "bold" if row["label"] == "Physics-Compact (proposed)" else "normal"
        ax.text(
            1.01, yi,
            rf"{row['mean']:.4f} $\pm$ {row['std']:.4f}",
            transform=trans,
            ha="left", va="center",
            fontsize=7.7,
            color=color,
            fontweight=weight,
        )

    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()

    save_figure(fig, outdir / "Fig05_ablation_validation_AW_RMSE_v4")

    # also save plotted data
    audit = df[["label", "mean", "std"]].copy()
    audit["ridge_reference"] = ridge
    audit.to_csv(outdir / "Fig05_ablation_validation_AW_RMSE_v4_data.csv", index=False, encoding="utf-8-sig")
    print(f"[SAVED] {outdir / 'Fig05_ablation_validation_AW_RMSE_v4_data.csv'}")


# ======================================================================================
# FIG06
# ======================================================================================

def fallback_fig06_data():
    # from your 10A terminal output
    rows = [
        {
            "dataset_id": "SD1090_validation",
            "label": "SD1090 validation",
            "ridge_vessel_skill_pct": 3.76,
            "physics_compact_vessel_sse_reduction_pct": 31.71,
            "gamma": 0.478,
            "corr_mag": 0.3069,
        },
        {
            "dataset_id": "SD1033_2024_matched_SD1090",
            "label": "SD1033-2024 matched",
            "ridge_vessel_skill_pct": 4.62,
            "physics_compact_vessel_sse_reduction_pct": 29.69,
            "gamma": 0.446,
            "corr_mag": 0.2729,
        },
        {
            "dataset_id": "SD1033_2023_full",
            "label": "SD1033-2023",
            "ridge_vessel_skill_pct": -50.90,
            "physics_compact_vessel_sse_reduction_pct": 58.08,
            "gamma": 0.722,
            "corr_mag": 0.5153,
        },
        {
            "dataset_id": "SD1033_2022_full",
            "label": "SD1033-2022",
            "ridge_vessel_skill_pct": -5.34,
            "physics_compact_vessel_sse_reduction_pct": 39.20,
            "gamma": 0.567,
            "corr_mag": 0.3326,
        },
    ]
    return pd.DataFrame(rows)


def extract_fig06_from_csv(root: Path):
    mech_dir = (
        root / "data" / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
        / "10A_mechanism_analysis_v0_1"
    )

    ridge_path = mech_dir / "ridge_branch_residual_diagnostics.csv"
    key_path = mech_dir / "mechanism_key_findings.csv"

    ridge = try_read(ridge_path, "10A ridge_branch_residual_diagnostics")
    key = try_read(key_path, "10A mechanism_key_findings")

    if ridge is None or key is None:
        print("[WARN] Fig06 using fallback values from terminal output.")
        return fallback_fig06_data()

    try:
        # ridge diagnostics
        dataset_col_r = find_col(ridge, ["dataset_id", "dataset"])
        branch_col_r = find_col(ridge, ["branch"])
        skill_col_r = find_col(ridge, ["ridge_skill_vs_persistence", "skill_vs_persistence"])

        # mechanism key findings
        dataset_col_k = find_col(key, ["dataset_id", "dataset"])
        gamma_col_k = find_col(key, [
            "mean_correction_cosine_alignment",
            "correction_cosine_alignment",
            "alignment"
        ])
        corr_col_k = find_col(key, [
            "mean_physicscompact_correction_rms_mps",
            "mean_correction_rms_mps",
            "correction_rms_mps",
            "correction_magnitude_mps"
        ])
        sse_col_k = find_col(key, [
            "mean_vessel_sse_reduction_fraction_vs_ridge",
            "vessel_sse_reduction_fraction_vs_ridge",
            "sse_reduction_fraction_vs_ridge"
        ])

        if None in [dataset_col_r, branch_col_r, skill_col_r, dataset_col_k, gamma_col_k, corr_col_k, sse_col_k]:
            raise RuntimeError("unresolved CSV columns")

        ridge2 = ridge[[dataset_col_r, branch_col_r, skill_col_r]].copy()
        ridge2.columns = ["dataset_id", "branch", "skill"]
        ridge2 = ridge2[ridge2["branch"].astype(str).str.lower() == "vessel"].copy()
        ridge2 = ridge2.groupby("dataset_id", as_index=False)["skill"].mean()
        ridge2["ridge_vessel_skill_pct"] = 100.0 * ridge2["skill"].astype(float)

        key2 = key[[dataset_col_k, gamma_col_k, corr_col_k, sse_col_k]].copy()
        key2.columns = ["dataset_id", "gamma", "corr_mag", "sse_frac"]
        key2["physics_compact_vessel_sse_reduction_pct"] = 100.0 * key2["sse_frac"].astype(float)

        df = ridge2.merge(key2, on="dataset_id", how="inner", validate="one_to_one")

        label_map = {
            "SD1090_validation": "SD1090 validation",
            "SD1033_2024_matched_SD1090": "SD1033-2024 matched",
            "SD1033_2023_full": "SD1033-2023",
            "SD1033_2022_full": "SD1033-2022",
        }
        df["label"] = df["dataset_id"].map(label_map).fillna(df["dataset_id"])

        order = [
            "SD1090_validation",
            "SD1033_2024_matched_SD1090",
            "SD1033_2023_full",
            "SD1033_2022_full",
        ]
        df["order"] = df["dataset_id"].map({k: i for i, k in enumerate(order)})
        df = df.sort_values("order").reset_index(drop=True)

        return df[[
            "dataset_id", "label",
            "ridge_vessel_skill_pct",
            "physics_compact_vessel_sse_reduction_pct",
            "gamma", "corr_mag"
        ]]

    except Exception as exc:
        print(f"[WARN] Fig06 CSV parse failed ({type(exc).__name__}: {exc}). Using fallback values.")
        return fallback_fig06_data()


def make_fig06(root: Path, outdir: Path):
    df = extract_fig06_from_csv(root)

    # manuscript colors: clearly distinct
    color_map = {
        "SD1090 validation": "#1f77b4",       # blue
        "SD1033-2024 matched": "#ff7f0e",     # orange
        "SD1033-2023": "#2ca02c",             # green
        "SD1033-2022": "#d62728",             # red
    }
    df["color"] = df["label"].map(color_map).fillna("#1f77b4")

    fig, (ax, ax_info) = plt.subplots(
        1, 2,
        figsize=(7.8, 4.1),
        gridspec_kw={"width_ratios": [1.65, 1.0]}
    )

    # main scatter
    for _, row in df.iterrows():
        ax.scatter(
            row["ridge_vessel_skill_pct"],
            row["physics_compact_vessel_sse_reduction_pct"],
            s=105,
            color=row["color"],
            edgecolors="black",
            linewidths=0.5,
            zorder=3
        )

    ax.axvline(0.0, linestyle="--", linewidth=1.0, color="#1f77b4", zorder=1)

    ax.set_xlabel("Ridge vessel skill vs Persistence (%)")
    ax.set_ylabel("Physics-Compact vessel SSE reduction (%)")

    xmin = min(-55.0, df["ridge_vessel_skill_pct"].min() - 4.0)
    xmax = max(8.0, df["ridge_vessel_skill_pct"].max() + 4.0)
    ymin = min(27.0, df["physics_compact_vessel_sse_reduction_pct"].min() - 2.0)
    ymax = max(60.5, df["physics_compact_vessel_sse_reduction_pct"].max() + 2.0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    finish_axis(ax)

    # right-side information panel
    ax_info.axis("off")
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)

    ax_info.text(
        0.02, 0.98, "Dataset-wise statistics",
        ha="left", va="top", fontsize=8.2, fontweight="bold"
    )

    y0 = 0.84
    dy = 0.22

    for i, (_, row) in enumerate(df.iterrows()):
        yy = y0 - i * dy

        ax_info.scatter(
            0.05, yy + 0.02,
            s=58,
            color=row["color"],
            edgecolors="black",
            linewidths=0.5,
            clip_on=False
        )

        ax_info.text(
            0.11, yy + 0.05,
            row["label"],
            ha="left", va="center",
            fontsize=8.0, fontweight="bold"
        )
        ax_info.text(
            0.11, yy - 0.01,
            f"Ridge vessel skill: {row['ridge_vessel_skill_pct']:+.2f}%",
            ha="left", va="center",
            fontsize=7.5
        )
        ax_info.text(
            0.11, yy - 0.07,
            f"SSE reduction: {row['physics_compact_vessel_sse_reduction_pct']:.2f}%",
            ha="left", va="center",
            fontsize=7.5
        )
        ax_info.text(
            0.11, yy - 0.13,
            rf"$\gamma$={row['gamma']:.2f},  $|c|$={row['corr_mag']:.2f} m s$^{{-1}}$",
            ha="left", va="center",
            fontsize=7.5
        )

    ax_info.text(
        0.02, 0.03,
        "Dashed line: Ridge = Persistence\nfor vessel prediction skill",
        ha="left", va="bottom", fontsize=7.1
    )

    fig.tight_layout()
    save_figure(fig, outdir / "Fig06_mechanism_vessel_shift_correction_v4")

    # save plotted data
    df.to_csv(outdir / "Fig06_mechanism_vessel_shift_correction_v4_data.csv", index=False, encoding="utf-8-sig")
    print(f"[SAVED] {outdir / 'Fig06_mechanism_vessel_shift_correction_v4_data.csv'}")


# ======================================================================================
# Main
# ======================================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If omitted, figures are written to data/forecasting/13A_final_manuscript_figures_v0_1"
    )
    args = parser.parse_args()

    root = args.project_root
    outdir = args.output_dir or (root / "data" / "forecasting" / "13A_final_manuscript_figures_v0_1")
    outdir.mkdir(parents=True, exist_ok=True)

    apply_style(root)

    print("=" * 100)
    print("REDRAW FIG05 + FIG06 (v4)")
    print("=" * 100)
    print(f"project root : {root}")
    print(f"output dir   : {outdir}")

    make_fig05(root, outdir)
    make_fig06(root, outdir)

    print("\n[DONE] Fig05 v4 and Fig06 v4 generated successfully.")


if __name__ == "__main__":
    main()