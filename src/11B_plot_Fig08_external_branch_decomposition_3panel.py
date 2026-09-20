# -*- coding: utf-8 -*-
r"""
11B_plot_Fig08_external_branch_decomposition_3panel.py

Generate the revised external branch-decomposition figure for the paper.

New Fig. 8 structure
--------------------
(a) Apparent-wind vector RMSE
(b) True-wind vector RMSE
(c) Vessel-velocity vector RMSE

Datasets
--------
SD1033-2024 matched
SD1033-2023
SD1033-2022

Branch variants
---------------
Dual Ridge
Wind enhanced
Motion enhanced
Physics-Compact

Important interpretation
------------------------
The three panels are intentionally branch-specific:

- True-wind panel:
    Dual Ridge == Motion enhanced
    Wind enhanced == Physics-Compact

  because the motion correction does not change the atmospheric prediction.

- Vessel-velocity panel:
    Dual Ridge == Wind enhanced
    Motion enhanced == Physics-Compact

  because the atmospheric correction does not change vessel prediction.

- Apparent-wind panel:
    all four variants can differ because apparent wind couples both branches.

This makes the physical role of each residual branch directly visible.

Data sources
------------
The script reads the existing frozen 15F outputs:

    15F_external_branch_decomposition.csv
    15F_external_per_seed_summary.csv

The branch CSV already contains apparent-wind and vessel-vector metrics.
The per-seed summary contains the enhanced/full true-wind metrics.

The deterministic dedicated TrueWind Ridge metric is reconstructed exactly
from:
    SD1090 train.npz
and the frozen SD1033 external NPZ products.

No neural model is retrained and no external parameter is fitted.

Default project paths
---------------------
root:
    D:\project\WindPredict_SaildroneData

dataset:
    data\forecasting\SD1090_TPOS2024_JointForecasting_v0_1

15F results:
    data\forecasting\15F_Frozen_15EVH_alpha1500_External_v0_1

Output:
    figures\final_paper\Fig08_external_branch_decomposition_3panel.png
    figures\final_paper\Fig08_external_branch_decomposition_3panel.pdf

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\11B_plot_Fig08_external_branch_decomposition_3panel.py" `
  --root "D:\project\WindPredict_SaildroneData"
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


HORIZONS = np.asarray([1, 2, 3, 5, 10], dtype=int)

DATASET_ORDER = [
    "SD1033-2024-matched",
    "SD1033-2023-full",
    "SD1033-2022-full",
]

DATASET_LABELS = [
    "2024 matched",
    "2023",
    "2022",
]

EXPECTED_EXTERNAL_COUNTS = {
    "SD1033-2024-matched": 111245,
    "SD1033-2023-full": 174458,
    "SD1033-2022-full": 106584,
}

MODEL_ORDER = [
    "Dual Ridge",
    "Wind enhanced",
    "Motion enhanced",
    "Physics-Compact",
]

# Keep branch colors consistent with the SD1090 internal branch figure.
COLORS = {
    "Dual Ridge": "#ff7f0e",
    "Wind enhanced": "#2ca02c",
    "Motion enhanced": "#9467bd",
    "Physics-Compact": "#d62728",
}


# =============================================================================
# Utilities
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

    if (
        "windenhanced" in n
        or "windenhancedridgemotion" in n
    ):
        return "Wind enhanced"

    if (
        "motionenhanced" in n
        or "ridgewindmotionenhanced" in n
    ):
        return "Motion enhanced"

    if (
        n in {"15evh", "physicscompact"}
        or "physicscompact" in n
    ):
        return "Physics-Compact"

    return str(x)


def canonical_dataset(x: str) -> str:
    n = norm_text(x)

    if "sd1033" in n and "2024" in n and "matched" in n:
        return "SD1033-2024-matched"

    if "sd1033" in n and "2023" in n:
        return "SD1033-2023-full"

    if "sd1033" in n and "2022" in n:
        return "SD1033-2022-full"

    return str(x)


def load_sci_style(root: Path) -> None:
    style_path = root / "src" / "sci_plot_style.py"

    if style_path.exists():
        try:
            spec = importlib.util.spec_from_file_location(
                "sci_plot_style_fig08_external_3panel",
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
                        try:
                            mod.apply_sci_style(font_size=8.5)
                        except TypeError:
                            mod.apply_sci_style()
                    return

        except Exception as exc:
            log(f"[WARNING] sci_plot_style.py could not be applied: {exc}")

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
            "axes.titlesize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "axes.unicode_minus": False,
        }
    )


def clean_axis(ax) -> None:
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
        axis="y",
        linewidth=0.40,
        alpha=0.18,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def panel_label(ax, label: str) -> None:
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
    )


def save_both(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)

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


# =============================================================================
# Exact dedicated TrueWind Ridge
# =============================================================================

def compute_stats(X: np.ndarray, Y: np.ndarray) -> dict:
    """
    Exact sufficient statistics used by the final Ridge implementation.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    return {
        "n": len(X),
        "sum_x": np.sum(X, axis=0),
        "sum_y": np.sum(Y, axis=0),
        "sum_x2": X.T @ X,
        "sum_xy": X.T @ Y,
    }


def fit_truewind_ridge_exact(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """
    Exact final dedicated TrueWind Ridge:
        60 x [U,V,T,RH] -> 5 x [U,V]
        alpha_W = 0

    This follows the centered Gram-matrix / standardized-target formulation
    used in the final 15F pipeline.
    """
    Xf = np.asarray(
        X_train[:, :, 0:4],
        dtype=np.float32,
    ).reshape(len(X_train), -1)

    yf = np.asarray(
        y_train[:, :, 0:2],
        dtype=np.float32,
    ).reshape(len(y_train), -1)

    y_mean = np.mean(
        yf.astype(np.float64),
        axis=0,
    )

    y_std = np.std(
        yf.astype(np.float64),
        axis=0,
        ddof=0,
    )

    y_std = np.where(
        y_std < 1e-8,
        1.0,
        y_std,
    )

    yz = (
        (yf - y_mean[None, :])
        / y_std[None, :]
    ).astype(np.float32)

    stats = compute_stats(
        Xf,
        yz,
    )

    n = float(stats["n"])

    x_mean = stats["sum_x"] / n
    y_mean_z = stats["sum_y"] / n

    G = (
        stats["sum_x2"]
        - np.outer(
            stats["sum_x"],
            stats["sum_x"],
        ) / n
    )

    G = 0.5 * (G + G.T)

    C = (
        stats["sum_xy"]
        - np.outer(
            stats["sum_x"],
            y_mean_z,
        )
    )

    try:
        B = np.linalg.solve(G, C)
        jitter = 0.0
    except np.linalg.LinAlgError:
        jitter = 1e-10
        B = np.linalg.solve(
            G
            + jitter
            * np.eye(
                G.shape[0],
                dtype=np.float64,
            ),
            C,
        )

    intercept = y_mean_z - x_mean @ B

    return {
        "coef_z": B,
        "intercept_z": intercept,
        "y_mean_raw": y_mean,
        "y_std_raw": y_std,
        "alpha": 0.0,
        "numerical_jitter": jitter,
    }


def predict_truewind_ridge(
    model: dict,
    X: np.ndarray,
) -> np.ndarray:
    Xf = np.asarray(
        X[:, :, 0:4],
        dtype=np.float64,
    ).reshape(len(X), -1)

    z = (
        Xf @ model["coef_z"]
        + model["intercept_z"][None, :]
    )

    raw = (
        z * model["y_std_raw"][None, :]
        + model["y_mean_raw"][None, :]
    )

    return raw.reshape(
        -1,
        5,
        2,
    ).astype(np.float32)


def truewind_vector_rmse_mean_over_horizons(
    y_true_uv: np.ndarray,
    y_pred_uv: np.ndarray,
) -> float:
    vals = []

    for j, _h in enumerate(HORIZONS):
        e = (
            y_pred_uv[:, j, :].astype(np.float64)
            - y_true_uv[:, j, :].astype(np.float64)
        )

        vals.append(
            float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            e * e,
                            axis=1,
                        )
                    )
                )
            )
        )

    return float(np.mean(vals))


# =============================================================================
# NPZ loading / discovery
# =============================================================================

def load_train(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    p = dataset_dir / "train.npz"

    if not p.exists():
        raise FileNotFoundError(p)

    with np.load(p, allow_pickle=False) as z:
        if "X" not in z.files:
            raise KeyError(f"{p}: missing X")

        if "y_raw" in z.files:
            y_key = "y_raw"
        elif "y" in z.files:
            y_key = "y"
        else:
            raise KeyError(
                f"{p}: neither y_raw nor y exists; "
                f"available={list(z.files)}"
            )

        X = np.asarray(
            z["X"],
            dtype=np.float32,
        )

        y = np.asarray(
            z[y_key],
            dtype=np.float32,
        )

    if X.shape[1:] != (60, 11):
        raise RuntimeError(
            f"{p}: unexpected X shape {X.shape}"
        )

    if y.shape[1:] != (5, 6):
        raise RuntimeError(
            f"{p}: unexpected y shape {y.shape}"
        )

    return X, y


def inspect_external_npz(path: Path):
    try:
        with np.load(
            path,
            allow_pickle=False,
            mmap_mode="r",
        ) as z:
            if (
                "X" not in z.files
                or "y_raw" not in z.files
            ):
                return None

            x = z["X"]
            y = z["y_raw"]

            if (
                x.ndim != 3
                or x.shape[1:] != (60, 11)
                or y.ndim != 3
                or y.shape[1:] != (5, 6)
                or len(x) != len(y)
            ):
                return None

            return {
                "n": int(len(x)),
            }

    except Exception:
        return None


def discover_external_products(
    external_root: Path,
) -> Dict[str, Path]:
    """
    Match the frozen SD1033 products by their known sample counts.
    """
    wanted_by_n = {
        n: name
        for name, n
        in EXPECTED_EXTERNAL_COUNTS.items()
    }

    found: Dict[str, Path] = {}

    log("")
    log("[DISCOVER] Searching frozen SD1033 NPZ products ...")

    for p in external_root.rglob("*.npz"):
        name_lower = p.name.lower()

        # Fast name gate before inspecting large NPZs.
        full_lower = str(p).lower()

        if (
            "sd1033" not in full_lower
            and "1033" not in full_lower
        ):
            continue

        info = inspect_external_npz(p)

        if info is None:
            continue

        n = info["n"]

        if n not in wanted_by_n:
            continue

        ds = wanted_by_n[n]

        # Prefer a file whose name/path also contains the expected year.
        if ds in found:
            year = ds.split("-")[1]
            old_score = int(year in str(found[ds]))
            new_score = int(year in str(p))

            if new_score <= old_score:
                continue

        found[ds] = p

        log(
            f"  {ds:24s} -> {p} "
            f"(N={n:,})"
        )

    missing = [
        ds
        for ds in DATASET_ORDER
        if ds not in found
    ]

    if missing:
        raise RuntimeError(
            "Could not auto-discover all required SD1033 products:\n"
            + "\n".join(missing)
            + "\nUse --external-2024-matched, --external-2023, "
              "--external-2022 to specify them explicitly."
        )

    return found


def load_external_product(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        if "X" not in z.files:
            raise KeyError(f"{path}: missing X")

        if "y_raw" not in z.files:
            raise KeyError(f"{path}: missing y_raw")

        X = np.asarray(
            z["X"],
            dtype=np.float32,
        )

        y = np.asarray(
            z["y_raw"],
            dtype=np.float32,
        )

    return X, y


# =============================================================================
# Read existing 15F results
# =============================================================================

def find_col(
    df: pd.DataFrame,
    aliases,
) -> str:
    lookup = {
        norm_text(c): c
        for c in df.columns
    }

    for alias in aliases:
        key = norm_text(alias)

        if key in lookup:
            return lookup[key]

    raise KeyError(
        f"Could not find any of {aliases}; "
        f"available columns={list(df.columns)}"
    )


def load_branch_results(
    final_ext_dir: Path,
) -> pd.DataFrame:
    p = (
        final_ext_dir
        / "15F_external_branch_decomposition.csv"
    )

    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)

    dataset_col = find_col(
        df,
        ["dataset"],
    )

    model_col = find_col(
        df,
        ["model"],
    )

    df["_dataset"] = df[dataset_col].map(
        canonical_dataset
    )

    df["_model"] = df[model_col].map(
        canonical_model
    )

    return df


def load_full_truewind_results(
    final_ext_dir: Path,
) -> pd.DataFrame:
    p = (
        final_ext_dir
        / "15F_external_per_seed_summary.csv"
    )

    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)

    dataset_col = find_col(
        df,
        ["dataset"],
    )

    tw_col = find_col(
        df,
        ["TrueWind_vector_RMSE_mps"],
    )

    df["_dataset"] = df[dataset_col].map(
        canonical_dataset
    )

    df["_tw"] = pd.to_numeric(
        df[tw_col],
        errors="coerce",
    )

    return df


def mean_std(values) -> Tuple[float, float]:
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return np.nan, np.nan

    mean = float(np.mean(x))

    std = (
        float(np.std(x, ddof=1))
        if len(x) > 1
        else 0.0
    )

    return mean, std


# =============================================================================
# Build new 3-panel metric table
# =============================================================================

def build_plot_table(
    branch_df: pd.DataFrame,
    full_tw_df: pd.DataFrame,
    ridge_tw_by_dataset: Dict[str, float],
) -> pd.DataFrame:
    app_col = find_col(
        branch_df,
        [
            "AppVector_RMSE_mps",
            "apparent_vector_RMSE_mps",
        ],
    )

    vessel_col = find_col(
        branch_df,
        [
            "Vessel_vector_RMSE_mps",
            "VesselVector_RMSE_mps",
        ],
    )

    rows = []

    for ds in DATASET_ORDER:
        # Full/wind-enhanced true-wind result is the same atmospheric
        # branch; obtain its five-seed distribution from 15F per-seed output.
        tw_seed_vals = (
            full_tw_df.loc[
                full_tw_df["_dataset"] == ds,
                "_tw",
            ]
            .dropna()
            .to_numpy(
                dtype=np.float64
            )
        )

        tw_enh_mean, tw_enh_std = mean_std(
            tw_seed_vals
        )

        ridge_tw = float(
            ridge_tw_by_dataset[ds]
        )

        for model in MODEL_ORDER:
            d = branch_df[
                (branch_df["_dataset"] == ds)
                & (branch_df["_model"] == model)
            ]

            if len(d) == 0:
                raise RuntimeError(
                    f"No branch rows for dataset={ds}, model={model}"
                )

            app_mean, app_std = mean_std(
                pd.to_numeric(
                    d[app_col],
                    errors="coerce",
                ).to_numpy()
            )

            vessel_mean, vessel_std = mean_std(
                pd.to_numeric(
                    d[vessel_col],
                    errors="coerce",
                ).to_numpy()
            )

            # Exact branch identity for true-wind prediction.
            if model in {
                "Dual Ridge",
                "Motion enhanced",
            }:
                tw_mean = ridge_tw
                tw_std = 0.0

            elif model in {
                "Wind enhanced",
                "Physics-Compact",
            }:
                tw_mean = tw_enh_mean
                tw_std = tw_enh_std

            else:
                raise RuntimeError(
                    f"Unexpected model {model}"
                )

            rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "AppVector_RMSE_mean": app_mean,
                    "AppVector_RMSE_std": app_std,
                    "TrueWind_vector_RMSE_mean": tw_mean,
                    "TrueWind_vector_RMSE_std": tw_std,
                    "Vessel_vector_RMSE_mean": vessel_mean,
                    "Vessel_vector_RMSE_std": vessel_std,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Plot
# =============================================================================

def plot_fig8(
    table: pd.DataFrame,
    output_dir: Path,
) -> None:
    metrics = [
        (
            "AppVector_RMSE_mean",
            "AppVector_RMSE_std",
            r"Apparent-wind vector RMSE (m s$^{-1}$)",
        ),
        (
            "TrueWind_vector_RMSE_mean",
            "TrueWind_vector_RMSE_std",
            r"True-wind vector RMSE (m s$^{-1}$)",
        ),
        (
            "Vessel_vector_RMSE_mean",
            "Vessel_vector_RMSE_std",
            r"Vessel-velocity vector RMSE (m s$^{-1}$)",
        ),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(9.6, 2.85),
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.065,
        right=0.995,
        bottom=0.19,
        top=0.80,
        wspace=0.25,
    )

    x = np.arange(
        len(DATASET_ORDER),
        dtype=float,
    )

    width = 0.19

    offsets = np.linspace(
        -1.5 * width,
        1.5 * width,
        len(MODEL_ORDER),
    )

    for panel_idx, (
        mean_col,
        std_col,
        ylabel,
    ) in enumerate(metrics):
        ax = axes[panel_idx]

        for model_idx, model in enumerate(
            MODEL_ORDER
        ):
            means = []
            stds = []

            for ds in DATASET_ORDER:
                q = table[
                    (table["dataset"] == ds)
                    & (table["model"] == model)
                ]

                if len(q) != 1:
                    raise RuntimeError(
                        f"Expected one aggregated row for {ds}/{model}; "
                        f"got {len(q)}"
                    )

                means.append(
                    float(
                        q.iloc[0][mean_col]
                    )
                )

                stds.append(
                    float(
                        q.iloc[0][std_col]
                    )
                )

            ax.bar(
                x + offsets[model_idx],
                means,
                width=width,
                yerr=stds,
                capsize=2.2,
                color=COLORS[model],
                edgecolor="black",
                linewidth=0.45,
                label=model,
                zorder=3,
                error_kw={
                    "elinewidth": 0.8,
                    "capthick": 0.8,
                },
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            DATASET_LABELS
        )

        ax.set_ylabel(
            ylabel
        )

        panel_label(
            ax,
            f"({chr(ord('a') + panel_idx)})",
        )

        clean_axis(
            ax
        )

    # One shared legend.
    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
        fontsize=7.2,
        columnspacing=1.15,
        handlelength=2.2,
        handletextpad=0.45,
    )

    save_both(
        fig,
        output_dir
        / "Fig08_external_branch_decomposition_3panel",
    )


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        type=Path,
        default=Path(
            r"D:\project\WindPredict_SaildroneData"
        ),
    )

    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--final-ext-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the 15F external CSV outputs."
        ),
    )

    ap.add_argument(
        "--external-root",
        type=Path,
        default=None,
        help=(
            "Root under which SD1033 external NPZ products are searched."
        ),
    )

    ap.add_argument(
        "--external-2024-matched",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--external-2023",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--external-2022",
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

    final_ext_dir = (
        args.final_ext_dir.resolve()
        if args.final_ext_dir is not None
        else root
        / "data"
        / "forecasting"
        / "15F_Frozen_15EVH_alpha1500_External_v0_1"
    )

    external_root = (
        args.external_root.resolve()
        if args.external_root is not None
        else root / "data"
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root
        / "figures"
        / "final_paper"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    load_sci_style(root)

    log("=" * 108)
    log("REVISED FIG. 8 — EXTERNAL BRANCH DECOMPOSITION")
    log("=" * 108)
    log(f"dataset_dir  : {dataset_dir}")
    log(f"15F results  : {final_ext_dir}")
    log(f"external_root: {external_root}")
    log(f"output_dir   : {output_dir}")
    log("")
    log("Panels:")
    log("  (a) apparent-wind vector RMSE")
    log("  (b) true-wind vector RMSE")
    log("  (c) vessel-velocity vector RMSE")
    log("=" * 108)

    # -----------------------------------------------------------------
    # Existing 15F results.
    # -----------------------------------------------------------------
    branch_df = load_branch_results(
        final_ext_dir
    )

    full_tw_df = load_full_truewind_results(
        final_ext_dir
    )

    # -----------------------------------------------------------------
    # Reconstruct the deterministic atmospheric Ridge exactly.
    # -----------------------------------------------------------------
    X_train, y_train = load_train(
        dataset_dir
    )

    log("")
    log("[TRUE-WIND RIDGE] rebuilding frozen alpha_W=0 anchor ...")

    ridge_model = fit_truewind_ridge_exact(
        X_train,
        y_train,
    )

    log(
        "  numerical jitter = "
        f"{ridge_model['numerical_jitter']:.3e}"
    )

    # Resolve external NPZ products.
    explicit = {
        "SD1033-2024-matched":
            args.external_2024_matched,
        "SD1033-2023-full":
            args.external_2023,
        "SD1033-2022-full":
            args.external_2022,
    }

    ext_paths: Dict[str, Path] = {}

    for ds, p in explicit.items():
        if p is not None:
            pp = p.resolve()

            if not pp.exists():
                raise FileNotFoundError(pp)

            ext_paths[ds] = pp

    if len(ext_paths) < len(DATASET_ORDER):
        discovered = discover_external_products(
            external_root
        )

        for ds, p in discovered.items():
            if ds not in ext_paths:
                ext_paths[ds] = p

    ridge_tw_by_dataset: Dict[
        str,
        float,
    ] = {}

    log("")
    log("[TRUE-WIND RIDGE] frozen external evaluation:")

    for ds in DATASET_ORDER:
        p = ext_paths[ds]

        X_ext, y_ext = load_external_product(
            p
        )

        expected_n = (
            EXPECTED_EXTERNAL_COUNTS[ds]
        )

        if len(X_ext) != expected_n:
            raise RuntimeError(
                f"{ds}: N={len(X_ext):,}, "
                f"expected {expected_n:,}"
            )

        ridge_pred = predict_truewind_ridge(
            ridge_model,
            X_ext,
        )

        value = (
            truewind_vector_rmse_mean_over_horizons(
                y_ext[:, :, 0:2],
                ridge_pred,
            )
        )

        ridge_tw_by_dataset[ds] = value

        log(
            f"  {ds:24s} | "
            f"TrueWind vector RMSE = {value:.6f} m/s"
        )

        del X_ext, y_ext, ridge_pred

    # -----------------------------------------------------------------
    # Build aggregated 3-panel data.
    # -----------------------------------------------------------------
    plot_table = build_plot_table(
        branch_df,
        full_tw_df,
        ridge_tw_by_dataset,
    )

    csv_out = (
        output_dir
        / "Fig08_external_branch_decomposition_3panel_data.csv"
    )

    plot_table.to_csv(
        csv_out,
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log("[FIGURE DATA]")
    log(
        plot_table.to_string(
            index=False
        )
    )

    log("")
    log(f"[SAVED] {csv_out}")

    # -----------------------------------------------------------------
    # Plot.
    # -----------------------------------------------------------------
    plot_fig8(
        plot_table,
        output_dir,
    )

    log("")
    log("[DONE]")


if __name__ == "__main__":
    main()
