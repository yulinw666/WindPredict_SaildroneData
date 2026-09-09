# -*- coding: utf-8 -*-
"""
13A_generate_final_manuscript_figures.py

Purpose
-------
Generate the FINAL manuscript figures for the current Saildrone forecasting
paper from already frozen CSV outputs.

NO TRAINING.
NO MODEL SELECTION.
NO METRIC RECOMPUTATION FROM RAW PREDICTIONS.
NO EXTERNAL DATA FITTING.

Default project root
--------------------
D:\project\WindPredict_SaildroneData

Outputs
-------
data\forecasting\13A_final_manuscript_figures_v0_1\

Main manuscript candidates:
    Fig02_primary_multihorizon_AW_RMSE.png/pdf
    Fig03_modern_baselines_horizon_AW_RMSE.png/pdf
    Fig04a_external_2024_matched_AW_RMSE.png/pdf
    Fig04b_external_2023_AW_RMSE.png/pdf
    Fig04c_external_2022_AW_RMSE.png/pdf
    Fig05_ablation_validation_AW_RMSE.png/pdf
    Fig06_mechanism_vessel_shift_correction.png/pdf
    Fig07_BPSTGNN_same_mission_wind_RMSE.png/pdf   [copied from 12E if present]
    Fig08_complexity_performance.png/pdf

It also writes:
    figure_generation_status.txt

Scientific policy
-----------------
- 07B/07C values are only used for SD1090 validation/ablation.
- 09C values are only used for frozen SD1090 outer validation.
- 09D values are only used for frozen SD1033 external evaluation.
- 10A values are only used for post-hoc mechanism analysis.
- 12E same-mission BP-STGNN figure is copied only if it already exists.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


DEFAULT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")
HORIZONS = [1, 2, 3, 5, 10]

PRIMARY_DATASETS = [
    "SD1033_2024_matched_SD1090",
    "SD1033_2023_full",
    "SD1033_2022_full",
]

DATASET_SHORT = {
    "SD1090_validation": "SD1090 val.",
    "SD1033_2024_matched_SD1090": "SD1033-2024\nmatched",
    "SD1033_2023_full": "SD1033-2023",
    "SD1033_2022_full": "SD1033-2022",
}

MODEL_SHORT = {
    "Physics-Compact-Vessel-Residual": "Physics-Compact",
    "Compact-Vessel-Residual": "Compact",
    "Full-Residual": "Full",
    "Vessel-Only-Residual": "Vessel only",
    "Physics-Full-Residual": "Physics full",
    "Physics-Vessel-Only-Residual": "Physics vessel only",
    "Frozen-Physics-Compact-Vessel-Residual": "Physics-Compact",
    "Frozen-Compact-Vessel-Residual": "Compact",
    "Frozen-Ridge": "Ridge",
    "Frozen-DLinear": "DLinear",
    "Frozen-TimeMixer": "TimeMixer",
    "Frozen-iTransformer": "iTransformer",
    "Frozen-PatchTST": "PatchTST",
}


# ---------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------

def apply_style(root: Path):
    style_path = root / "src" / "sci_plot_style.py"
    style_module = None

    if style_path.exists():
        try:
            spec = importlib.util.spec_from_file_location(
                "sci_plot_style_13a",
                str(style_path),
            )
            style_module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(style_module)

            candidates = [
                ("apply_sci_style", {"base_font_size": 8.5}),
                ("apply_sci_style", {"font_size": 8.5}),
                ("apply_sci_style", {}),
                ("apply_sci_plot_style", {}),
                ("set_sci_plot_style", {}),
            ]

            for name, kwargs in candidates:
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
    print("[STYLE] fallback Times New Roman/serif")
    return style_module


def finish_axis(ax, style_module=None):
    if style_module is not None:
        for name in ["clean_axis", "finish_axes"]:
            fn = getattr(style_module, name, None)
            if callable(fn):
                try:
                    fn(ax, grid=True)
                    return
                except TypeError:
                    try:
                        fn(ax)
                        return
                    except Exception:
                        pass

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
    )
    ax.grid(True, alpha=0.20, linewidth=0.6)


def save_figure(fig, base: Path, style_module=None):
    base.parent.mkdir(parents=True, exist_ok=True)

    # Use project helper if available, but still enforce both formats.
    helper = getattr(style_module, "save_figure", None) if style_module else None
    if callable(helper):
        try:
            helper(fig, base)
        except Exception:
            pass

    fig.savefig(
        base.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        base.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"[SAVED] {base.name}.png/pdf")


# ---------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------

def all_csvs(root: Path):
    return list(root.rglob("*.csv"))


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
        # Prefer shorter, cleaner paths after token score.
        return (s, -len(str(p)))

    hits.sort(key=score, reverse=True)
    return hits[0]


def find_csv_with_columns(root: Path, required_cols, prefer_tokens=()):
    candidates = []
    for path in all_csvs(root):
        try:
            df = pd.read_csv(path, nrows=3)
        except Exception:
            continue
        if all(c in df.columns for c in required_cols):
            score = 0
            low = str(path).lower()
            for token in prefer_tokens:
                if token.lower() in low:
                    score += 10
            candidates.append((score, -len(str(path)), path))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][2]


def read(path: Path | None, label: str):
    if path is None or not path.exists():
        raise FileNotFoundError(f"{label}: file not found")
    print(f"[INPUT] {label}: {path}")
    return pd.read_csv(path)


def choose_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def validate_horizons(df, label):
    if "horizon_min" not in df.columns:
        raise RuntimeError(f"{label}: missing horizon_min")
    got = sorted(set(pd.to_numeric(df["horizon_min"], errors="coerce").dropna().astype(int)))
    if not set(HORIZONS).issubset(got):
        raise RuntimeError(f"{label}: expected horizons {HORIZONS}, got {got}")


# ---------------------------------------------------------------------
# Figure 2: primary multi-horizon
# ---------------------------------------------------------------------

def make_fig02(root, outdir, style):
    dataset_dir = (
        root / "data" / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
    )

    p07c = find_named(
        dataset_dir,
        "seed_per_horizon_summary.csv",
        prefer_tokens=("confirm_and_compact", "07C"),
    )
    p07b = find_named(
        dataset_dir,
        "structured_ablation_metrics_per_horizon.csv",
        prefer_tokens=("structured_residual_ablation", "07B"),
    )
    p09c = find_named(
        dataset_dir,
        "persistence_validation_metrics.csv",
        prefer_tokens=("09C_modern_baseline_outer_validation", "09C"),
    )

    df07c = read(p07c, "07C per-horizon summary")
    df07b = read(p07b, "07B per-horizon metrics")
    dfp = read(p09c, "09C persistence validation")

    fig, ax = plt.subplots(figsize=(5.4, 3.35))

    # Persistence
    p = dfp.copy()
    if "split" in p.columns:
        p = p[p["split"].astype(str).str.lower().eq("validation")]
    p = p[p["horizon_min"].isin(HORIZONS)].sort_values("horizon_min")
    pcol = choose_col(p, [
        "apparent_vector_RMSE_mps",
        "mean_apparent_vector_RMSE_mps",
    ])
    if pcol is None:
        raise RuntimeError("Persistence file has no apparent-wind RMSE column.")
    ax.plot(
        p["horizon_min"],
        p[pcol],
        marker="o",
        label="Persistence",
    )

    # Ridge
    r = df07b.copy()
    if "split" in r.columns:
        r = r[r["split"].astype(str).str.lower().eq("validation")]
    r = r[r["model"].eq("Ridge")]
    r = r[r["horizon_min"].isin(HORIZONS)].sort_values("horizon_min")
    rcol = choose_col(r, [
        "apparent_vector_RMSE_mps",
        "mean_apparent_vector_RMSE_mps",
    ])
    if rcol is None:
        raise RuntimeError("07B file has no Ridge apparent-wind RMSE column.")
    ax.plot(
        r["horizon_min"],
        r[rcol],
        marker="s",
        label="Ridge",
    )

    # Compact and Physics-Compact
    mean_col = "mean_apparent_vector_RMSE_mps"
    std_col = "std_apparent_vector_RMSE_mps"
    for model, label, marker in [
        ("Compact-Vessel-Residual", "Compact", "^"),
        ("Physics-Compact-Vessel-Residual", "Physics-Compact", "D"),
    ]:
        g = df07c.copy()
        if "split" in g.columns:
            g = g[g["split"].astype(str).str.lower().eq("validation")]
        g = g[g["model"].eq(model)]
        g = g[g["horizon_min"].isin(HORIZONS)].sort_values("horizon_min")
        ax.errorbar(
            g["horizon_min"],
            g[mean_col],
            yerr=g[std_col] if std_col in g.columns else None,
            marker=marker,
            capsize=2,
            label=label,
        )

    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel(r"Apparent-wind vector RMSE (m s$^{-1}$)")
    ax.set_xticks(HORIZONS)
    ax.legend(loc="best")
    finish_axis(ax, style)
    fig.tight_layout()
    save_figure(fig, outdir / "Fig02_primary_multihorizon_AW_RMSE", style)


# ---------------------------------------------------------------------
# Figure 3: modern baselines
# ---------------------------------------------------------------------

def make_fig03(root, outdir, style):
    dataset_dir = (
        root / "data" / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
    )

    p09c = find_named(
        dataset_dir,
        "final_per_horizon_summary.csv",
        prefer_tokens=("09C_modern_baseline_outer_validation", "09C"),
    )
    p07c = find_named(
        dataset_dir,
        "seed_per_horizon_summary.csv",
        prefer_tokens=("confirm_and_compact", "07C"),
    )

    modern = read(p09c, "09C modern per-horizon summary")
    compact = read(p07c, "07C Physics-Compact per-horizon summary")

    fig, ax = plt.subplots(figsize=(5.4, 3.35))

    pc = compact[
        compact["model"].eq("Physics-Compact-Vessel-Residual")
    ].copy()
    if "split" in pc.columns:
        pc = pc[pc["split"].astype(str).str.lower().eq("validation")]
    pc = pc.sort_values("horizon_min")
    ax.errorbar(
        pc["horizon_min"],
        pc["mean_apparent_vector_RMSE_mps"],
        yerr=pc.get("std_apparent_vector_RMSE_mps"),
        marker="D",
        capsize=2,
        label="Physics-Compact",
    )

    for model, marker in [
        ("DLinear", "o"),
        ("TimeMixer", "s"),
        ("iTransformer", "^"),
        ("PatchTST", "v"),
    ]:
        g = modern[modern["model"].eq(model)].copy()
        g = g.sort_values("horizon_min")

        mean_col = choose_col(g, [
            "mean_apparent_vector_RMSE_mps",
            "mean_AW_RMSE_mps",
        ])
        std_col = choose_col(g, [
            "std_apparent_vector_RMSE_mps",
            "std_AW_RMSE_mps",
        ])
        if mean_col is None:
            raise RuntimeError(f"09C modern summary missing AW RMSE for {model}")

        ax.errorbar(
            g["horizon_min"],
            g[mean_col],
            yerr=g[std_col] if std_col else None,
            marker=marker,
            capsize=2,
            label=model,
        )

    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel(r"Apparent-wind vector RMSE (m s$^{-1}$)")
    ax.set_xticks(HORIZONS)
    ax.legend(loc="best", ncol=2)
    finish_axis(ax, style)
    fig.tight_layout()
    save_figure(fig, outdir / "Fig03_modern_baselines_horizon_AW_RMSE", style)


# ---------------------------------------------------------------------
# Figure 4a-c: external transfer
# ---------------------------------------------------------------------

def make_fig04(root, outdir, style):
    external_root = (
        root / "data" / "external"
        / "SD1033_external_forecasting_v0_1"
    )

    path = find_named(
        external_root,
        "integrated_external_per_horizon_summary.csv",
        prefer_tokens=("09D_modern_baseline_external_inference", "09D"),
    )
    df = read(path, "09D integrated external per-horizon summary")

    desired_models = [
        "Frozen-Physics-Compact-Vessel-Residual",
        "Frozen-DLinear",
        "Frozen-TimeMixer",
        "Frozen-iTransformer",
        "Frozen-PatchTST",
    ]

    output_names = {
        "SD1033_2024_matched_SD1090": "Fig04a_external_2024_matched_AW_RMSE",
        "SD1033_2023_full": "Fig04b_external_2023_AW_RMSE",
        "SD1033_2022_full": "Fig04c_external_2022_AW_RMSE",
    }

    for dataset_id in PRIMARY_DATASETS:
        fig, ax = plt.subplots(figsize=(5.3, 3.25))
        sub = df[df["dataset_id"].eq(dataset_id)].copy()

        for model, marker in zip(
            desired_models,
            ["D", "o", "s", "^", "v"],
        ):
            g = sub[sub["model"].eq(model)].sort_values("horizon_min")
            if g.empty:
                print(f"[WARNING] missing {dataset_id}/{model}")
                continue

            mean_col = choose_col(g, [
                "mean_apparent_vector_RMSE_mps",
                "mean_AW_RMSE_mps",
            ])
            std_col = choose_col(g, [
                "std_apparent_vector_RMSE_mps",
                "std_AW_RMSE_mps",
            ])

            ax.errorbar(
                g["horizon_min"],
                g[mean_col],
                yerr=g[std_col] if std_col else None,
                marker=marker,
                capsize=2,
                label=MODEL_SHORT.get(model, model),
            )

        ax.set_xlabel("Forecast horizon (min)")
        ax.set_ylabel(r"Apparent-wind vector RMSE (m s$^{-1}$)")
        ax.set_xticks(HORIZONS)
        ax.legend(loc="best", ncol=2)
        finish_axis(ax, style)
        fig.tight_layout()
        save_figure(fig, outdir / output_names[dataset_id], style)


# ---------------------------------------------------------------------
# Figure 5: final ablation
# ---------------------------------------------------------------------

def make_fig05(root, outdir, style):
    dataset_dir = (
        root / "data" / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
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

    df = read(p07c, "07C seed summary")
    df07b = read(p07b, "07B structured ablation summary")

    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower().eq("validation")]

    order = [
        "Full-Residual",
        "Vessel-Only-Residual",
        "Physics-Full-Residual",
        "Physics-Vessel-Only-Residual",
        "Compact-Vessel-Residual",
        "Physics-Compact-Vessel-Residual",
    ]
    labels = [
        "Full",
        "Vessel\nonly",
        "Physics\nfull",
        "Physics\nvessel only",
        "Compact",
        "Physics-\nCompact",
    ]

    rows = []
    for model, label in zip(order, labels):
        q = df[df["model"].eq(model)]
        if q.empty:
            continue
        r = q.iloc[0]
        rows.append((
            label,
            float(r["mean_AW_RMSE_mps"]),
            float(r["std_AW_RMSE_mps"]),
        ))

    ridge = df07b[
        (df07b["model"].eq("Ridge"))
        & (
            df07b["split"].astype(str).str.lower().eq("validation")
            if "split" in df07b.columns
            else True
        )
    ]
    ridge_aw = (
        float(ridge.iloc[0]["mean_apparent_vector_RMSE_mps"])
        if not ridge.empty else np.nan
    )

    fig, ax = plt.subplots(figsize=(6.1, 3.5))
    x = np.arange(len(rows))
    means = [r[1] for r in rows]
    stds = [r[2] for r in rows]

    ax.errorbar(
        x,
        means,
        yerr=stds,
        marker="o",
        linestyle="none",
        capsize=3,
        label="Residual variants",
    )
    if np.isfinite(ridge_aw):
        ax.axhline(
            ridge_aw,
            linestyle="--",
            linewidth=1.0,
            label=f"Ridge = {ridge_aw:.3f}",
        )

    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel(r"Validation AW vector RMSE (m s$^{-1}$)")
    ax.legend(loc="best")
    finish_axis(ax, style)
    fig.tight_layout()
    save_figure(fig, outdir / "Fig05_ablation_validation_AW_RMSE", style)


# ---------------------------------------------------------------------
# Figure 6: mechanism summary
# ---------------------------------------------------------------------

def make_fig06(root, outdir, style):
    dataset_dir = (
        root / "data" / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
    )
    mech_dir = dataset_dir / "10A_mechanism_analysis_v0_1"

    key_path = find_named(
        mech_dir,
        "mechanism_key_findings.csv",
        prefer_tokens=("10A_mechanism",),
    )

    ridge_path = find_csv_with_columns(
        mech_dir,
        required_cols=[
            "dataset_id",
            "branch",
            "ridge_skill_vs_persistence",
        ],
        prefer_tokens=("ridge", "diagnostic"),
    )

    key = read(key_path, "10A mechanism key findings")
    ridge = read(ridge_path, "10A Ridge branch diagnostics")

    vessel = (
        ridge[ridge["branch"].astype(str).str.lower().eq("vessel")]
        .groupby("dataset_id", as_index=False)
        ["ridge_skill_vs_persistence"]
        .mean()
    )

    merged = vessel.merge(key, on="dataset_id", how="inner")
    if merged.empty:
        raise RuntimeError("Mechanism merge is empty.")

    x = 100.0 * merged["ridge_skill_vs_persistence"].to_numpy(float)
    y = (
        100.0
        * merged["mean_vessel_SSE_reduction_fraction_vs_Ridge"]
        .to_numpy(float)
    )
    corr = merged["mean_PhysicsCompact_correction_RMS_mps"].to_numpy(float)
    align = merged["mean_correction_cosine_alignment"].to_numpy(float)

    # Marker area encodes correction magnitude.
    if np.ptp(corr) > 0:
        sizes = 55.0 + 160.0 * (corr - corr.min()) / np.ptp(corr)
    else:
        sizes = np.full_like(corr, 100.0)

    fig, ax = plt.subplots(figsize=(5.3, 3.45))
    ax.scatter(x, y, s=sizes)

    for xi, yi, dataset_id, ai, ci in zip(
        x, y, merged["dataset_id"], align, corr
    ):
        label = DATASET_SHORT.get(dataset_id, dataset_id)
        ax.annotate(
            f"{label}\n$\\gamma$={ai:.2f}, $|c|$={ci:.2f}",
            (xi, yi),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7.2,
        )

    ax.axvline(0.0, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Ridge vessel skill vs Persistence (%)")
    ax.set_ylabel("Physics-Compact vessel SSE reduction (%)")
    finish_axis(ax, style)
    fig.tight_layout()
    save_figure(fig, outdir / "Fig06_mechanism_vessel_shift_correction", style)


# ---------------------------------------------------------------------
# Figure 7: same-mission BP-STGNN
# ---------------------------------------------------------------------

def make_fig07(root, outdir, status_lines):
    source = (
        root / "data" / "forecasting"
        / "12E_Nie_same_mission_paper_results_v0_1"
        / "figures"
        / "Fig12E1_same_mission_wind_RMSE.pdf"
    )
    source_png = source.with_suffix(".png")

    if source.exists():
        shutil.copy2(source, outdir / "Fig07_BPSTGNN_same_mission_wind_RMSE.pdf")
        if source_png.exists():
            shutil.copy2(
                source_png,
                outdir / "Fig07_BPSTGNN_same_mission_wind_RMSE.png",
            )
        print("[COPIED] Fig07 from frozen 12E output")
        status_lines.append(
            f"Fig07: copied existing 12E output from {source}"
        )
    else:
        status_lines.append(
            "Fig07: 12E figure not found. Run "
            r'python "D:\project\WindPredict_SaildroneData\src'
            r'\12E_generate_Nie_same_mission_paper_results.py"'
        )
        print("[WARNING] 12E Fig12E1 not found; see status file.")


# ---------------------------------------------------------------------
# Figure 8: complexity-performance
# ---------------------------------------------------------------------

def make_fig08(root, outdir, style):
    dataset_dir = (
        root / "data" / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
    )

    p09c = find_named(
        dataset_dir,
        "final_seed_summary.csv",
        prefer_tokens=("09C_modern_baseline_outer_validation", "09C"),
    )
    p07c = find_named(
        dataset_dir,
        "seed_summary.csv",
        prefer_tokens=("confirm_and_compact", "07C"),
    )

    modern = read(p09c, "09C modern seed summary")
    compact = read(p07c, "07C Physics-Compact seed summary")

    rows = []

    for model in ["DLinear", "TimeMixer", "iTransformer", "PatchTST"]:
        g = modern[modern["model"].eq(model)]
        if g.empty:
            continue
        r = g.iloc[0]

        pcol = choose_col(g, ["parameters"])
        awcol = choose_col(g, [
            "mean_AW_RMSE_mps",
            "mean_apparent_vector_RMSE_mps",
        ])
        rows.append({
            "model": model,
            "parameters": float(r[pcol]),
            "aw": float(r[awcol]),
        })

    pc = compact[
        compact["model"].eq("Physics-Compact-Vessel-Residual")
    ].copy()
    if "split" in pc.columns:
        pc = pc[pc["split"].astype(str).str.lower().eq("validation")]
    if not pc.empty:
        r = pc.iloc[0]
        rows.append({
            "model": "Physics-Compact",
            "parameters": float(r["parameters"]),
            "aw": float(r["mean_AW_RMSE_mps"]),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No complexity-performance rows found.")

    fig, ax = plt.subplots(figsize=(5.2, 3.35))
    ax.scatter(df["parameters"], df["aw"], s=45)

    for _, row in df.iterrows():
        ax.annotate(
            row["model"],
            (row["parameters"], row["aw"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7.5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel(r"Validation AW vector RMSE (m s$^{-1}$)")
    finish_axis(ax, style)
    fig.tight_layout()
    save_figure(fig, outdir / "Fig08_complexity_performance", style)


# ---------------------------------------------------------------------
# Existing-figure inventory
# ---------------------------------------------------------------------

def inventory_existing(root: Path):
    expected = {
        "Figure 1 framework schematic":
            None,  # user-managed manuscript artwork
        "07C ablation":
            find_named(root, "01_validation_aw_mean_std.pdf",
                       prefer_tokens=("confirm_and_compact",)),
        "07C per-horizon ablation":
            find_named(root, "02_validation_aw_per_horizon_mean_std.pdf",
                       prefer_tokens=("confirm_and_compact",)),
        "08C external AW":
            find_named(root, "01_external_aw_rmse.pdf",
                       prefer_tokens=("frozen_external_inference",)),
        "10A Ridge branch skill":
            find_named(root, "01_ridge_branch_skill.pdf",
                       prefer_tokens=("10A_mechanism",)),
        "10A gate by horizon":
            find_named(root, "02_physicscompact_gate_by_horizon.pdf",
                       prefer_tokens=("10A_mechanism",)),
        "10A correction alignment":
            find_named(root, "03_correction_alignment_by_horizon.pdf",
                       prefer_tokens=("10A_mechanism",)),
        "12E same-mission wind RMSE":
            find_named(root, "Fig12E1_same_mission_wind_RMSE.pdf",
                       prefer_tokens=("12E_Nie_same_mission",)),
        "12E parameter count":
            find_named(root, "Fig12E3_parameter_count.pdf",
                       prefer_tokens=("12E_Nie_same_mission",)),
        "12E AW 24h time series":
            find_named(root, "Fig12E7_apparent_wind_speed_timeseries_24h.pdf",
                       prefer_tokens=("12E_Nie_same_mission",)),
        "12E AW error CDF":
            find_named(root, "Fig12E8_apparent_wind_vector_error_CDF.pdf",
                       prefer_tokens=("12E_Nie_same_mission",)),
    }
    return expected


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

    style = apply_style(root)

    status_lines = [
        "13A FINAL MANUSCRIPT FIGURE STATUS",
        "=" * 90,
        "",
        "Existing figures discovered before final replot:",
    ]

    inventory = inventory_existing(root)
    for label, path in inventory.items():
        if label == "Figure 1 framework schematic":
            status_lines.append(
                "- Figure 1 framework schematic: already available in manuscript; "
                "not regenerated by this script."
            )
        elif path is not None:
            status_lines.append(f"- {label}: FOUND -> {path}")
        else:
            status_lines.append(f"- {label}: NOT FOUND")

    tasks = [
        ("Fig02 primary multi-horizon", make_fig02),
        ("Fig03 modern baselines", make_fig03),
        ("Fig04 external transfer", make_fig04),
        ("Fig05 ablation", make_fig05),
        ("Fig06 mechanism", make_fig06),
        ("Fig08 complexity-performance", make_fig08),
    ]

    status_lines += ["", "Final-manuscript replot status:"]

    for label, fn in tasks:
        try:
            fn(root, outdir, style)
            status_lines.append(f"- {label}: SUCCESS")
        except Exception as exc:
            status_lines.append(
                f"- {label}: FAILED -> {type(exc).__name__}: {exc}"
            )
            print(f"[FAILED] {label}: {type(exc).__name__}: {exc}")

    make_fig07(root, outdir, status_lines)

    status_lines += [
        "",
        "Recommended main-paper figure order:",
        "Figure 1  Physics-Compact framework schematic",
        "Figure 2  Primary multi-horizon AW RMSE",
        "Figure 3  Modern-baseline multi-horizon comparison",
        "Figure 4  External generalization (4a/4b/4c)",
        "Figure 5  Ablation study",
        "Figure 6  Mechanism analysis",
        "Figure 7  Same-mission BP-STGNN literature-aligned comparison",
        "Figure 8  Complexity-performance tradeoff [optional; table-only is acceptable]",
        "",
        "12E scatter/time-series/CDF figures are better treated as supplementary "
        "unless manuscript space permits.",
    ]

    status_path = outdir / "figure_generation_status.txt"
    status_path.write_text("\n".join(status_lines), encoding="utf-8")
    print(f"[DONE] status: {status_path}")
    print(f"[DONE] figures: {outdir}")


if __name__ == "__main__":
    main()
