# -*- coding: utf-8 -*-
r"""
11A_plot_SD1090_internal_evaluation_figures.py

Generate three paper figures from the corrected 10A-v4 SD1090
internal-evaluation results.

All three figures use the SAME structure:
    (a) Apparent-wind vector RMSE
    (b) True-wind vector RMSE
    (c) Vessel-velocity vector RMSE

and the SAME five forecast horizons:
    H = {1, 2, 3, 5, 10} min

Figures
-------
1) Fig03_SD1090_internal_branch_horizon
   Persistence, Dual Ridge, Wind enhanced, Motion enhanced, Physics-Compact

2) Fig04_SD1090_internal_modern_horizon
   DLinear, TimeMixer, iTransformer, PatchTST, Physics-Compact

3) Fig05_SD1090_internal_conventional_horizon
   Persistence, Dual Ridge, GRU, LSTM, CNN-LSTM, Physics-Compact

The Physics-Compact curve has the SAME red color and diamond marker in
all three figures.

Input
-----
Corrected 10A-v4:
    internal_ALL_models_ALL_horizons_summary.csv

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\11A_plot_SD1090_internal_evaluation_figures.py" `
  --root "D:\project\WindPredict_SaildroneData"

Optional explicit CSV:
python "...11A_plot_SD1090_internal_evaluation_figures.py" `
  --csv "...\10A_SD1090_internal_evaluation_v0_4\tables\internal_ALL_models_ALL_horizons_summary.csv"
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


HORIZONS = np.asarray([1, 2, 3, 5, 10], dtype=int)

# Global model styles. Repeated models keep exactly the same appearance
# across all three figures.
MODEL_STYLE = {
    "Persistence": {
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
    },
    "Dual Ridge": {
        "color": "#ff7f0e",
        "marker": "s",
        "linestyle": "-",
    },
    "Wind enhanced": {
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "-",
    },
    "Motion enhanced": {
        "color": "#9467bd",
        "marker": "v",
        "linestyle": "-",
    },
    "Physics-Compact": {
        "color": "#d62728",
        "marker": "D",
        "linestyle": "-",
    },
    "DLinear": {
        "color": "#ff7f0e",
        "marker": "s",
        "linestyle": "-",
    },
    "TimeMixer": {
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "-",
    },
    "iTransformer": {
        "color": "#9467bd",
        "marker": "v",
        "linestyle": "-",
    },
    "PatchTST": {
        "color": "#8c564b",
        "marker": "P",
        "linestyle": "-",
    },
    "GRU": {
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "-",
    },
    "LSTM": {
        "color": "#9467bd",
        "marker": "v",
        "linestyle": "-",
    },
    "CNN-LSTM": {
        "color": "#8c564b",
        "marker": "P",
        "linestyle": "-",
    },
}

PANEL_METRICS = [
    (
        "Apparent-wind vector RMSE",
        "mean_apparent_vector_RMSE_mps",
        "std_apparent_vector_RMSE_mps",
    ),
    (
        "True-wind vector RMSE",
        "mean_wind_vector_RMSE_mps",
        "std_wind_vector_RMSE_mps",
    ),
    (
        "Vessel-velocity vector RMSE",
        "mean_vessel_vector_RMSE_mps",
        "std_vessel_vector_RMSE_mps",
    ),
]


def log(msg=""):
    print(msg, flush=True)


def load_sci_style(root: Path):
    style_path = root / "src" / "sci_plot_style.py"

    if style_path.exists():
        try:
            spec = importlib.util.spec_from_file_location(
                "sci_plot_style_11a",
                str(style_path),
            )
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)

                if hasattr(mod, "apply_sci_style"):
                    try:
                        mod.apply_sci_style(base_font_size=9.0)
                    except TypeError:
                        try:
                            mod.apply_sci_style(font_size=9.0)
                        except TypeError:
                            mod.apply_sci_style()
                    return
        except Exception as e:
            log(f"[WARNING] sci_plot_style.py could not be applied: {e}")

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
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.2,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "axes.unicode_minus": False,
        }
    )


def clean_axis(ax):
    ax.tick_params(
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


def resolve_csv(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        p = explicit.resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    candidates = [
        root
        / "data"
        / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
        / "10A_SD1090_internal_evaluation_v0_4"
        / "tables"
        / "internal_ALL_models_ALL_horizons_summary.csv",
        root
        / "data"
        / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
        / "10A_SD1090_internal_evaluation_v0_3"
        / "tables"
        / "internal_ALL_models_ALL_horizons_summary.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Cannot find corrected 10A horizon summary. "
        "Pass --csv explicitly."
    )


def resolve_output(root: Path, csv_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()

    # If CSV is inside <eval>/tables/, write to <eval>/figures_paper_test/.
    if csv_path.parent.name.lower() == "tables":
        return csv_path.parent.parent / "figures_paper_test"

    return root / "figures_paper_test"


def validate_df(df: pd.DataFrame):
    required = {
        "model",
        "horizon_min",
        "mean_apparent_vector_RMSE_mps",
        "std_apparent_vector_RMSE_mps",
        "mean_wind_vector_RMSE_mps",
        "std_wind_vector_RMSE_mps",
        "mean_vessel_vector_RMSE_mps",
        "std_vessel_vector_RMSE_mps",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"CSV missing columns: {missing}")

    got_h = sorted(df["horizon_min"].unique().tolist())
    if got_h != HORIZONS.tolist():
        raise RuntimeError(
            f"Expected horizons {HORIZONS.tolist()}, got {got_h}"
        )


def plot_three_panel(
    df: pd.DataFrame,
    models: list[str],
    output_base: Path,
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.4, 3.65),
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.065,
        right=0.992,
        bottom=0.18,
        top=0.80,
        wspace=0.22,
    )

    labels = ["(a)", "(b)", "(c)"]

    for panel_idx, (
        panel_title,
        mean_col,
        std_col,
    ) in enumerate(PANEL_METRICS):
        ax = axes[panel_idx]

        for model in models:
            g = (
                df[df["model"] == model]
                .sort_values("horizon_min")
            )

            if len(g) != len(HORIZONS):
                raise RuntimeError(
                    f"{model}: expected {len(HORIZONS)} rows, got {len(g)}"
                )

            x = g["horizon_min"].to_numpy(dtype=float)
            y = g[mean_col].to_numpy(dtype=float)
            yerr = g[std_col].to_numpy(dtype=float)

            style = MODEL_STYLE[model]

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.25,
                markersize=5.4,
                markeredgewidth=0.7,
                capsize=2.4,
                capthick=0.8,
                elinewidth=0.8,
                label=model,
                zorder=4,
            )

        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Forecast horizon (min)")
        ax.set_ylabel(r"RMSE (m s$^{-1}$)")
        ax.set_title(panel_title, pad=4.0)

        ax.text(
            0.018,
            0.975,
            labels[panel_idx],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.4,
            fontweight="bold",
        )

        clean_axis(ax)

    # Shared legend, so all three panels have the same plotting area.
    handles, legend_labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=min(len(models), 6),
        frameon=False,
        fontsize=7.4,
        columnspacing=1.0,
        handlelength=2.1,
        handletextpad=0.4,
    )

    output_base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_base.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    log(f"[SAVED] {output_base.with_suffix('.png')}")
    log(f"[SAVED] {output_base.with_suffix('.pdf')}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        type=Path,
        default=Path(r"D:\project\WindPredict_SaildroneData"),
    )

    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    args = ap.parse_args()

    root = args.root.resolve()
    csv_path = resolve_csv(root, args.csv)
    out_dir = resolve_output(
        root,
        csv_path,
        args.output_dir,
    )

    load_sci_style(root)

    df = pd.read_csv(csv_path)
    validate_df(df)

    log("=" * 110)
    log("11A — SD1090 INTERNAL-EVALUATION PAPER FIGURES")
    log("=" * 110)
    log(f"CSV        : {csv_path}")
    log(f"output_dir : {out_dir}")
    log(
        "Panels     : apparent-wind vector / true-wind vector / "
        "vessel-velocity vector"
    )
    log(f"Horizons   : {HORIZONS.tolist()} min")
    log("Physics-Compact: red diamond in all three figures")

    # Figure 3: branch decomposition.
    plot_three_panel(
        df,
        [
            "Persistence",
            "Dual Ridge",
            "Wind enhanced",
            "Motion enhanced",
            "Physics-Compact",
        ],
        out_dir / "Fig03_SD1090_internal_branch_horizon",
    )

    # Figure 4: modern forecasting baselines.
    plot_three_panel(
        df,
        [
            "DLinear",
            "TimeMixer",
            "iTransformer",
            "PatchTST",
            "Physics-Compact",
        ],
        out_dir / "Fig04_SD1090_internal_modern_horizon",
    )

    # Additional conventional-baseline figure.
    # This becomes the next figure number wherever inserted in LaTeX.
    plot_three_panel(
        df,
        [
            "Persistence",
            "Dual Ridge",
            "GRU",
            "LSTM",
            "CNN-LSTM",
            "Physics-Compact",
        ],
        out_dir / "Fig05_SD1090_internal_conventional_horizon",
    )

    log("")
    log("[DONE]")


if __name__ == "__main__":
    main()
