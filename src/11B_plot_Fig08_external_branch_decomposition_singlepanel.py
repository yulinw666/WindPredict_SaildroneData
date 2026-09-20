# -*- coding: utf-8 -*-
r"""
11B_plot_Fig08_external_branch_decomposition_singlepanel.py

目的
----
重绘 Fig. 8，只保留 apparent-wind vector RMSE 的单面板柱状图。

新图结构
--------
4 个柱状区域：
1) SD1090 internal evaluation
2) SD1033-2024 matched
3) SD1033-2023
4) SD1033-2022

每个区域 4 个柱：
- Dual Ridge
- Wind enhanced
- Motion enhanced
- Physics-Compact

与旧版相比
----------
- 删除原来的 (b) True-wind 和 (c) Vessel-motion 两个 panel
- 增加 SD1090 internal evaluation 这一组
- 颜色改为更柔和、适合论文的风格
- 默认不显示 error bar；如需显示可加 --show-error-bars

数据来源
--------
1) SD1090 internal：
   优先读取 Table_internal_branch_ablation.csv
   若不存在，则从 Branch_ablation_AW_RMSE_ALL_horizons.csv 自动求平均

2) SD1033 external：
   读取 15F_external_branch_decomposition.csv
   自动汇总三个外部数据集的 apparent-wind vector RMSE

推荐运行
--------
python "D:\project\WindPredict_SaildroneData\src\11B_plot_Fig08_external_branch_decomposition_singlepanel.py" ^
  --root "D:\project\WindPredict_SaildroneData"

可选参数
--------
--show-error-bars
    若提供，则对随机模型画 std；否则只画均值柱状图
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
# 基本设置
# =============================================================================

MODEL_ORDER = [
    "Dual Ridge",
    "Wind enhanced",
    "Motion enhanced",
    "Physics-Compact",
]

# 更柔和、适合论文的配色（对比度降低）
COLORS = {
    "Dual Ridge": "#C9965A",       # muted amber
    "Wind enhanced": "#7FAE8A",    # muted sage green
    "Motion enhanced": "#A18BC3",  # muted violet
    "Physics-Compact": "#C76D6D",  # muted brick red
}

DATASET_ORDER = [
    "SD1090 internal evaluation",
    "SD1033-2024 matched",
    "SD1033-2023",
    "SD1033-2022",
]

DATASET_LABELS = [
    "SD1090\ninternal",
    "SD1033-2024\nmatched",
    "SD1033-2023",
    "SD1033-2022",
]


# =============================================================================
# 工具函数
# =============================================================================

def log(msg: str = "") -> None:
    print(msg, flush=True)


def norm_text(x) -> str:
    return (
        str(x)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
        .replace("+", "")
    )


def canonical_model(x: str) -> str:
    n = norm_text(x)

    if n in {"dualridge", "ridge", "dualridgemodel"}:
        return "Dual Ridge"

    if "windenhanced" in n:
        return "Wind enhanced"

    if "motionenhanced" in n:
        return "Motion enhanced"

    if n in {"15evh", "physicscompact"} or "physicscompact" in n:
        return "Physics-Compact"

    return str(x)


def canonical_dataset(x: str) -> str:
    n = norm_text(x)

    if "sd1090" in n and ("internal" in n or "validation" in n):
        return "SD1090 internal evaluation"

    if "sd1033" in n and "2024" in n and "matched" in n:
        return "SD1033-2024 matched"

    if "sd1033" in n and "2023" in n:
        return "SD1033-2023"

    if "sd1033" in n and "2022" in n:
        return "SD1033-2022"

    return str(x)


def find_col(df: pd.DataFrame, aliases) -> str:
    lookup = {norm_text(c): c for c in df.columns}
    for alias in aliases:
        key = norm_text(alias)
        if key in lookup:
            return lookup[key]
    raise KeyError(
        f"Could not find any of {aliases}; available columns={list(df.columns)}"
    )


def mean_std(values):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        return float(x[0]), 0.0
    return float(np.mean(x)), float(np.std(x, ddof=1))


def load_sci_style(root: Path) -> None:
    """
    如果项目里有 sci_plot_style.py，就沿用你之前的绘图格式。
    """
    style_path = root / "src" / "sci_plot_style.py"

    if not style_path.exists():
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "mathtext.fontset": "stix",
        })
        return

    try:
        spec = importlib.util.spec_from_file_location(
            "sci_plot_style_fig08_singlepanel",
            str(style_path),
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to load sci_plot_style.py")

        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        if hasattr(mod, "apply_sci_style"):
            try:
                mod.apply_sci_style(base_font_size=8.8)
            except TypeError:
                try:
                    mod.apply_sci_style(font_size=8.8)
                except TypeError:
                    mod.apply_sci_style()
    except Exception as e:
        log(f"[WARN] Failed to apply sci_plot_style.py: {e}")
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "mathtext.fontset": "stix",
        })


# =============================================================================
# 读取 SD1090 internal 数据
# =============================================================================

def load_internal_summary(internal_csv: Path, internal_horizon_csv: Path) -> pd.DataFrame:
    """
    优先读取已经汇总好的：
        Table_internal_branch_ablation.csv

    若不存在，则从：
        Branch_ablation_AW_RMSE_ALL_horizons.csv
    计算对 5 个 horizon 的平均值。
    """
    if internal_csv.exists():
        df = pd.read_csv(internal_csv)

        variant_col = find_col(df, ["Variant", "model"])
        mean_col = find_col(df, ["AW RMSE (m/s)", "AW_RMSE", "AW RMSE mean (m/s)"])
        std_col = find_col(df, ["AW RMSE std (m/s)", "AW_RMSE_std"])

        rows = []
        for _, r in df.iterrows():
            model = canonical_model(r[variant_col])
            if model not in MODEL_ORDER:
                continue
            rows.append({
                "dataset": "SD1090 internal evaluation",
                "model": model,
                "mean": float(r[mean_col]),
                "std": float(r[std_col]),
            })

        out = pd.DataFrame(rows)
        if not out.empty:
            return out

    if internal_horizon_csv.exists():
        df = pd.read_csv(internal_horizon_csv)

        variant_col = find_col(df, ["Variant", "model"])
        mean_col = find_col(df, ["AW RMSE mean (m/s)", "AW RMSE (m/s)"])
        std_col = find_col(df, ["AW RMSE std (m/s)"])

        rows = []
        for model_name in MODEL_ORDER:
            d = df[df[variant_col].map(canonical_model) == model_name]
            if len(d) == 0:
                continue

            mean_val = float(pd.to_numeric(d[mean_col], errors="coerce").mean())
            std_val = float(pd.to_numeric(d[std_col], errors="coerce").mean())

            rows.append({
                "dataset": "SD1090 internal evaluation",
                "model": model_name,
                "mean": mean_val,
                "std": std_val,
            })

        out = pd.DataFrame(rows)
        if not out.empty:
            return out

    raise FileNotFoundError(
        "Could not find internal branch-ablation data.\n"
        f"Tried:\n  {internal_csv}\n  {internal_horizon_csv}"
    )


# =============================================================================
# 读取 SD1033 external 数据
# =============================================================================

def load_external_summary(external_dir: Path) -> pd.DataFrame:
    """
    从 15F_external_branch_decomposition.csv 中读取 external 结果，
    只保留 apparent-wind vector RMSE。
    """
    p = external_dir / "15F_external_branch_decomposition.csv"
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)

    dataset_col = find_col(df, ["dataset"])
    model_col = find_col(df, ["model"])
    aw_col = find_col(df, ["AppVector_RMSE_mps", "apparent_vector_RMSE_mps"])

    df["_dataset"] = df[dataset_col].map(canonical_dataset)
    df["_model"] = df[model_col].map(canonical_model)
    df["_aw"] = pd.to_numeric(df[aw_col], errors="coerce")

    rows = []
    for ds in ["SD1033-2024 matched", "SD1033-2023", "SD1033-2022"]:
        for model in MODEL_ORDER:
            d = df[(df["_dataset"] == ds) & (df["_model"] == model)]
            if len(d) == 0:
                raise RuntimeError(f"No rows for dataset={ds}, model={model}")

            mean_val, std_val = mean_std(d["_aw"].to_numpy())

            rows.append({
                "dataset": ds,
                "model": model,
                "mean": mean_val,
                "std": std_val,
            })

    return pd.DataFrame(rows)


# =============================================================================
# 绘图
# =============================================================================

def build_plot_table(
    internal_df: pd.DataFrame,
    external_df: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.concat([internal_df, external_df], ignore_index=True)

    df["dataset"] = pd.Categorical(
        df["dataset"],
        categories=DATASET_ORDER,
        ordered=True,
    )
    df["model"] = pd.Categorical(
        df["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    df = df.sort_values(["dataset", "model"]).reset_index(drop=True)
    return df


def plot_fig8_singlepanel(
    table: pd.DataFrame,
    output_dir: Path,
    show_error_bars: bool = False,
) -> None:
    fig, ax = plt.subplots(
        1, 1,
        figsize=(7.0, 2.9),
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.995,
        bottom=0.24,
        top=0.80,
    )

    x = np.arange(len(DATASET_ORDER), dtype=float)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(MODEL_ORDER))

    for i, model in enumerate(MODEL_ORDER):
        d = table[table["model"] == model].copy()
        d = d.set_index("dataset").reindex(DATASET_ORDER)

        means = d["mean"].to_numpy(dtype=float)
        stds = d["std"].to_numpy(dtype=float)

        if show_error_bars:
            ax.bar(
                x + offsets[i],
                means,
                width=width,
                yerr=stds,
                capsize=2.2,
                color=COLORS[model],
                edgecolor="black",
                linewidth=0.6,
                error_kw=dict(
                    lw=0.8,
                    capthick=0.8,
                    ecolor="black",
                ),
                label=model,
                zorder=3,
            )
        else:
            ax.bar(
                x + offsets[i],
                means,
                width=width,
                color=COLORS[model],
                edgecolor="black",
                linewidth=0.6,
                label=model,
                zorder=3,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_LABELS)
    ax.set_ylabel(r"Apparent-wind vector RMSE (m s$^{-1}$)")
    ax.set_ylim(0.75, 1.18)

    ax.grid(axis="y", alpha=0.22, linewidth=0.6, zorder=0)

    # panel 标记
    ax.text(
        0.012, 0.985,
        "(a)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    ax.legend(
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        frameon=False,
        columnspacing=1.6,
        handlelength=2.3,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = "Fig08_external_branch_decomposition_singlepanel"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    log(f"[OK] Saved PNG: {png_path}")
    log(f"[OK] Saved PDF: {pdf_path}")


# =============================================================================
# 主函数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default=r"D:\project\WindPredict_SaildroneData",
        help="Project root directory.",
    )

    parser.add_argument(
        "--internal-csv",
        type=str,
        default=None,
        help="Optional path to Table_internal_branch_ablation.csv",
    )

    parser.add_argument(
        "--internal-horizon-csv",
        type=str,
        default=None,
        help="Optional path to Branch_ablation_AW_RMSE_ALL_horizons.csv",
    )

    parser.add_argument(
        "--external-dir",
        type=str,
        default=None,
        help="Directory containing 15F_external_branch_decomposition.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Figure output directory.",
    )

    parser.add_argument(
        "--show-error-bars",
        action="store_true",
        help="If set, draw standard-deviation error bars.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)

    load_sci_style(root)

    internal_csv = (
        Path(args.internal_csv)
        if args.internal_csv is not None
        else root / "Table_internal_branch_ablation.csv"
    )

    internal_horizon_csv = (
        Path(args.internal_horizon_csv)
        if args.internal_horizon_csv is not None
        else root / "Branch_ablation_AW_RMSE_ALL_horizons.csv"
    )

    external_dir = (
        Path(args.external_dir)
        if args.external_dir is not None
        else root / "data" / "forecasting" / "15F_Frozen_15EVH_alpha1500_External_v0_1"
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else root / "figures" / "final_paper"
    )

    log("=" * 90)
    log("BUILDING FIG. 8 (SINGLE-PANEL EXTERNAL/INTERNAL BRANCH COMPARISON)")
    log("=" * 90)

    log(f"[INFO] root                 : {root}")
    log(f"[INFO] internal_csv         : {internal_csv}")
    log(f"[INFO] internal_horizon_csv : {internal_horizon_csv}")
    log(f"[INFO] external_dir         : {external_dir}")
    log(f"[INFO] output_dir           : {output_dir}")
    log(f"[INFO] show_error_bars      : {args.show_error_bars}")

    internal_df = load_internal_summary(
        internal_csv=internal_csv,
        internal_horizon_csv=internal_horizon_csv,
    )
    external_df = load_external_summary(external_dir=external_dir)

    table = build_plot_table(
        internal_df=internal_df,
        external_df=external_df,
    )

    log("\n[SUMMARY TABLE]")
    log(table.to_string(index=False))

    plot_fig8_singlepanel(
        table=table,
        output_dir=output_dir,
        show_error_bars=args.show_error_bars,
    )

    log("\n[DONE]")


if __name__ == "__main__":
    main()