# -*- coding: utf-8 -*-
"""
One-panel grouped bar chart:
RMSE comparison of U, V, and WS across different forecasting models.

Data source:
1) Physics-Compact (current work): from your final terminal result
   U/V/vector/WS/WD = 0.713950 / 0.781024 / 1.058170 / 0.679781 / 5.939701

2) Other models: from Nie et al. tables
   Table 2: RMSE of U and V
   Table 3: RMSE of wind speed (WS)

Output:
- Fig_BPSTGNN_like_grouped_RMSE.png
- Fig_BPSTGNN_like_grouped_RMSE.pdf
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =============================================================================
# Optional style loading
# =============================================================================
def try_apply_user_style():
    """
    Try to load user's sci_plot_style.py if available.
    If not found, fall back to local matplotlib settings.
    """
    candidate_paths = [
        Path.cwd() / "sci_plot_style.py",
        Path(__file__).resolve().parent / "sci_plot_style.py",
    ]
    loaded = False
    for p in candidate_paths:
        if p.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("sci_plot_style", str(p))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "apply_style"):
                    module.apply_style()
                    loaded = True
                    print(f"[STYLE] Loaded apply_style() from: {p}")
                    break
            except Exception as e:
                print(f"[STYLE] Failed to load {p}: {e}")

    if not loaded:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 16,
            "axes.labelsize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "figure.dpi": 200,
            "savefig.dpi": 600,
            "axes.linewidth": 1.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
            "xtick.minor.width": 1.2,
            "ytick.minor.width": 1.2,
            "xtick.major.size": 7,
            "ytick.major.size": 7,
            "xtick.minor.size": 4,
            "ytick.minor.size": 4,
        })
        print("[STYLE] Using built-in matplotlib style.")


# =============================================================================
# Data
# =============================================================================
def build_data():
    """
    Returns model names and RMSE values for U, V, WS.
    """
    model_names = [
        "Physics-Compact",
        "BP-STGNN",
        "HD-Meta",
        "GNN-Random",
        "CNN-LSTM",
        "BiLSTM",
        "StackedLSTM",
        "Transformer",
    ]

    # RMSE values:
    # Physics-Compact -> from your own final result
    # others -> from Nie et al. Table 2 & Table 3
    U_rmse = np.array([
        0.713950,  # Physics-Compact (your current model)
        0.74864,   # BP-STGNN
        0.83720,   # HD-Meta
        0.81500,   # GNN-Random
        0.86063,   # CNN-LSTM
        0.87692,   # BiLSTM
        0.70380,   # StackedLSTM
        1.32790,   # Transformer
    ], dtype=float)

    V_rmse = np.array([
        0.781024,  # Physics-Compact
        0.70506,   # BP-STGNN
        0.90203,   # HD-Meta
        0.73008,   # GNN-Random
        1.23870,   # CNN-LSTM
        1.18100,   # BiLSTM
        1.34510,   # StackedLSTM
        1.06350,   # Transformer
    ], dtype=float)

    WS_rmse = np.array([
        0.679781,  # Physics-Compact
        0.69940,   # BP-STGNN
        0.88488,   # HD-Meta
        0.71847,   # GNN-Random
        0.87696,   # CNN-LSTM
        0.92981,   # BiLSTM
        0.79286,   # StackedLSTM
        1.29220,   # Transformer
    ], dtype=float)

    return model_names, U_rmse, V_rmse, WS_rmse


# =============================================================================
# Plot helpers
# =============================================================================
def beautify_axes(ax):
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.grid(axis="y", linestyle="-", linewidth=0.8, alpha=0.25)


def add_value_labels(ax, bars, fmt="{:.3f}", dy=0.012):
    """
    Horizontal numerical labels above each bar.
    """
    for b in bars:
        h = b.get_height()

        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=10.5,
            rotation=0,          # 改为横向
            clip_on=False,
        )


def plot_grouped_rmse(output_dir: Path):
    model_names, U_rmse, V_rmse, WS_rmse = build_data()

    x = np.arange(len(model_names))
    width = 0.23

    # 与现在论文图统一的色系
    color_u = "#4C78A8"     # blue
    color_v = "#F58518"     # orange
    color_ws = "#54A24B"    # green

    fig, ax = plt.subplots(
        figsize=(16, 7.4),
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.19,
        top=0.91,
    )

    # ------------------------------------------------------------------
    # Highlight proposed model
    # ------------------------------------------------------------------
    # 只保留淡背景，不再写 "Current model"，避免和图例发生遮挡
    ax.axvspan(
        -0.50,
        0.50,
        color="#F6D6BF",
        alpha=0.22,
        zorder=0,
    )

    # ------------------------------------------------------------------
    # Bars
    # ------------------------------------------------------------------
    bars_u = ax.bar(
        x - width,
        U_rmse,
        width,
        label="U",
        color=color_u,
        edgecolor="black",
        linewidth=0.75,
        zorder=3,
    )

    bars_v = ax.bar(
        x,
        V_rmse,
        width,
        label="V",
        color=color_v,
        edgecolor="black",
        linewidth=0.75,
        zorder=3,
    )

    bars_ws = ax.bar(
        x + width,
        WS_rmse,
        width,
        label="WS",
        color=color_ws,
        edgecolor="black",
        linewidth=0.75,
        zorder=3,
    )

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------
    ax.set_ylabel(
        r"RMSE (m s$^{-1}$)",
    )

    ax.set_xlabel(
        "Forecasting model",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        model_names,
        rotation=18,
        ha="right",
    )

    # Physics-Compact label加粗
    xticklabels = ax.get_xticklabels()
    if len(xticklabels) > 0:
        xticklabels[0].set_fontweight("bold")

    all_values = np.concatenate(
        [
            U_rmse,
            V_rmse,
            WS_rmse,
        ]
    )

    ymax = max(all_values) * 1.14

    ax.set_ylim(
        0.0,
        ymax,
    )

    # ------------------------------------------------------------------
    # Grid and ticks
    # ------------------------------------------------------------------
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
    )

    ax.minorticks_on()

    ax.grid(
        axis="y",
        linestyle="-",
        linewidth=0.65,
        alpha=0.20,
        zorder=0,
    )
    ax.text(
        0,
        ymax * 0.98,
        "Current model",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
        color="#A64B00"
    )

    # ------------------------------------------------------------------
    # Legend -> upper right
    # ------------------------------------------------------------------
    ax.legend(
        loc="upper right",
        ncol=3,
        frameon=False,
        fontsize=13.5,
        handlelength=1.8,
        columnspacing=1.4,
        borderaxespad=0.7,
    )

    # ------------------------------------------------------------------
    # Horizontal numerical labels
    # ------------------------------------------------------------------
    label_offset = ymax * 0.012

    add_value_labels(
        ax,
        bars_u,
        fmt="{:.3f}",
        dy=label_offset,
    )

    add_value_labels(
        ax,
        bars_v,
        fmt="{:.3f}",
        dy=label_offset,
    )

    add_value_labels(
        ax,
        bars_ws,
        fmt="{:.3f}",
        dy=label_offset,
    )

    # ------------------------------------------------------------------
    # 不建议论文图内部放标题
    # 交给 figure caption 即可
    # ------------------------------------------------------------------
    # ax.set_title(
    #     "RMSE comparison of U, V, and wind speed across forecasting models"
    # )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        output_dir
        / "Fig_BPSTGNN_grouped_RMSE_final.png"
    )

    pdf_path = (
        output_dir
        / "Fig_BPSTGNN_grouped_RMSE_final.pdf"
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[SAVED] {png_path}")
    print(f"[SAVED] {pdf_path}")


# =============================================================================
# Entry
# =============================================================================
def main():
    try_apply_user_style()

    # 默认输出到当前目录下 figures/final_paper
    output_dir = Path.cwd() / "figures" / "final_paper"

    print("=" * 100)
    print("PLOT — GROUPED BAR CHART FOR U / V / WS RMSE COMPARISON")
    print("=" * 100)
    print(f"Output directory: {output_dir}")
    print("=" * 100)

    plot_grouped_rmse(output_dir)

    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()