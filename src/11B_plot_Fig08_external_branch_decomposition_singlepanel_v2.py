# -*- coding: utf-8 -*-
r"""
11B_plot_Fig08_external_branch_decomposition_singlepanel_v2.py

Revised Fig. 8:
- one panel only: apparent-wind vector RMSE
- four dataset groups:
    SD1090 internal evaluation
    SD1033-2024 matched
    SD1033-2023
    SD1033-2022
- four bars per group:
    Dual Ridge
    Wind enhanced
    Motion enhanced
    Physics-Compact
- muted SCI-style palette
- no error bars by default
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


MODEL_ORDER = [
    "Dual Ridge",
    "Wind enhanced",
    "Motion enhanced",
    "Physics-Compact",
]

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

# Muted palette: lower saturation than the previous orange/green/purple/red.
COLORS = {
    "Dual Ridge": "#C7A16B",       # muted ochre
    "Wind enhanced": "#7FA58A",    # muted sage
    "Motion enhanced": "#9A8FB8",  # muted lavender
    "Physics-Compact": "#C87572",  # muted brick red
}


def log(msg=""):
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
    if "physicscompact" in n or n == "15evh":
        return "Physics-Compact"

    return str(x)


def canonical_dataset(x: str) -> str:
    n = norm_text(x)

    if "sd1090" in n and ("internal" in n or "test" in n or "evaluation" in n):
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


def load_sci_style(root: Path):
    style_path = root / "src" / "sci_plot_style.py"

    if style_path.exists():
        try:
            spec = importlib.util.spec_from_file_location(
                "sci_plot_style_fig08_singlepanel_v2",
                str(style_path),
            )
            if spec is not None and spec.loader is not None:
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
            "font.size": 8.8,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 7.5,
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


def discover_internal_csv(dataset_dir: Path, explicit: Path | None) -> Path:
    """
    Correct location from 10A-v4 is normally:
      <dataset_dir>/10A_SD1090_internal_evaluation_v0_4/tables/
          Table_internal_branch_ablation.csv
    """
    if explicit is not None:
        p = explicit.resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    candidates = [
        dataset_dir
        / "10A_SD1090_internal_evaluation_v0_4"
        / "tables"
        / "Table_internal_branch_ablation.csv",

        dataset_dir
        / "10A_SD1090_internal_evaluation_v0_3"
        / "tables"
        / "Table_internal_branch_ablation.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    # Robust fallback: search recursively, preferring v0_4.
    found = list(dataset_dir.rglob("Table_internal_branch_ablation.csv"))

    if found:
        found = sorted(
            found,
            key=lambda p: (
                "v0_4" not in str(p),
                "v0_3" not in str(p),
                len(str(p)),
            ),
        )
        return found[0]

    raise FileNotFoundError(
        "Could not find Table_internal_branch_ablation.csv under:\n"
        f"  {dataset_dir}\n"
        "Expected location is usually:\n"
        f"  {dataset_dir / '10A_SD1090_internal_evaluation_v0_4' / 'tables' / 'Table_internal_branch_ablation.csv'}"
    )


def discover_external_csv(external_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        p = explicit.resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    p = external_dir / "15F_external_branch_decomposition.csv"
    if p.exists():
        return p

    found = list(external_dir.rglob("15F_external_branch_decomposition.csv"))
    if found:
        return sorted(found)[0]

    raise FileNotFoundError(
        "Could not find 15F_external_branch_decomposition.csv under:\n"
        f"  {external_dir}"
    )


def load_internal(internal_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(internal_csv)

    model_col = find_col(df, ["Variant", "model"])
    mean_col = find_col(
        df,
        [
            "AW RMSE (m/s)",
            "AW RMSE mean (m/s)",
            "AppVector_RMSE_mps",
            "apparent_vector_RMSE_mps",
        ],
    )

    rows = []

    for _, r in df.iterrows():
        model = canonical_model(r[model_col])

        if model not in MODEL_ORDER:
            continue

        rows.append(
            {
                "dataset": "SD1090 internal evaluation",
                "model": model,
                "rmse": float(r[mean_col]),
            }
        )

    out = pd.DataFrame(rows)

    missing = [m for m in MODEL_ORDER if m not in set(out["model"])]

    if missing:
        raise RuntimeError(
            f"Internal CSV is missing branch variants: {missing}\n"
            f"File: {internal_csv}"
        )

    return out


def load_external(external_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(external_csv)

    dataset_col = find_col(df, ["dataset"])
    model_col = find_col(df, ["model"])
    app_col = find_col(
        df,
        [
            "AppVector_RMSE_mps",
            "apparent_vector_RMSE_mps",
        ],
    )

    df["_dataset"] = df[dataset_col].map(canonical_dataset)
    df["_model"] = df[model_col].map(canonical_model)
    df["_app"] = pd.to_numeric(df[app_col], errors="coerce")

    rows = []

    for ds in DATASET_ORDER[1:]:
        for model in MODEL_ORDER:
            d = df[
                (df["_dataset"] == ds)
                & (df["_model"] == model)
            ]["_app"].dropna()

            if len(d) == 0:
                raise RuntimeError(
                    f"No external rows for dataset={ds}, model={model}\n"
                    f"File: {external_csv}"
                )

            rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "rmse": float(d.mean()),
                }
            )

    return pd.DataFrame(rows)


def build_table(internal_df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
    table = pd.concat(
        [internal_df, external_df],
        ignore_index=True,
    )

    table["dataset"] = pd.Categorical(
        table["dataset"],
        categories=DATASET_ORDER,
        ordered=True,
    )

    table["model"] = pd.Categorical(
        table["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    return (
        table
        .sort_values(["dataset", "model"])
        .reset_index(drop=True)
    )


def plot_figure(table: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(
        figsize=(7.15, 3.25),
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.105,
        right=0.992,
        bottom=0.22,
        top=0.80,
    )

    x = np.arange(len(DATASET_ORDER), dtype=float)

    width = 0.18

    offsets = np.array(
        [-1.5, -0.5, 0.5, 1.5],
        dtype=float,
    ) * width

    for i, model in enumerate(MODEL_ORDER):
        vals = []

        for ds in DATASET_ORDER:
            q = table[
                (table["dataset"] == ds)
                & (table["model"] == model)
            ]

            if len(q) != 1:
                raise RuntimeError(
                    f"Expected exactly one row for {ds}/{model}, got {len(q)}"
                )

            vals.append(float(q.iloc[0]["rmse"]))

        ax.bar(
            x + offsets[i],
            vals,
            width=width,
            color=COLORS[model],
            edgecolor="#333333",
            linewidth=0.55,
            label=model,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_LABELS)

    ax.set_ylabel(
        r"Apparent-wind vector RMSE (m s$^{-1}$)"
    )

    # Bar charts should retain a zero baseline.
    ymax = float(table["rmse"].max())
    ax.set_ylim(0.0, ymax * 1.12)

    ax.grid(
        True,
        axis="y",
        linewidth=0.45,
        alpha=0.18,
        zorder=0,
    )

    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
    )

    ax.minorticks_on()

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.2,
        handletextpad=0.45,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base = (
        output_dir
        / "Fig08_branch_transfer_AW_RMSE"
    )

    fig.savefig(
        base.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        base.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    log(f"[SAVED] {base.with_suffix('.png')}")
    log(f"[SAVED] {base.with_suffix('.pdf')}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        type=Path,
        default=Path(r"D:\project\WindPredict_SaildroneData"),
    )

    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--internal-csv",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--external-dir",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--external-csv",
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

    dataset_dir = (
        args.dataset_dir.resolve()
        if args.dataset_dir is not None
        else root
        / "data"
        / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
    )

    external_dir = (
        args.external_dir.resolve()
        if args.external_dir is not None
        else root
        / "data"
        / "forecasting"
        / "15F_Frozen_15EVH_alpha1500_External_v0_1"
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root
        / "figures"
        / "final_paper"
    )

    load_sci_style(root)

    internal_csv = discover_internal_csv(
        dataset_dir,
        args.internal_csv,
    )

    external_csv = discover_external_csv(
        external_dir,
        args.external_csv,
    )

    log("=" * 100)
    log("FIG. 8 — INTERNAL + CROSS-VOYAGE BRANCH COMPARISON")
    log("=" * 100)
    log(f"root         : {root}")
    log(f"dataset_dir  : {dataset_dir}")
    log(f"internal_csv : {internal_csv}")
    log(f"external_csv : {external_csv}")
    log(f"output_dir   : {output_dir}")

    internal_df = load_internal(
        internal_csv
    )

    external_df = load_external(
        external_csv
    )

    table = build_table(
        internal_df,
        external_df,
    )

    log("")
    log("[PLOT DATA]")
    log(table.to_string(index=False))

    table.to_csv(
        output_dir
        / "Fig08_branch_transfer_AW_RMSE_data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_figure(
        table,
        output_dir,
    )

    log("")
    log("[DONE]")


if __name__ == "__main__":
    main()
