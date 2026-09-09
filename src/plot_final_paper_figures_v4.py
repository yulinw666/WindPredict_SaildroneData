# -*- coding: utf-8 -*-
r"""
plot_final_paper_figures.py

Generate the final paper figures except Fig. 1 for the frozen
Branch-Aware Physics-Compact model.

Default project root
--------------------
D:\project\WindPredict_SaildroneData

Figures
-------
Fig. 2  SD1090 validation: branch-resolved RMSE vs forecast horizon
        (true wind / vessel motion / apparent wind)

Fig. 3  Representative 10-min-horizon time-series comparison
        (observed vs Dual Ridge vs Physics-Compact) for WS and AWS

Fig. 4  SD1090 validation: Physics-Compact vs modern baselines
        apparent-wind vector RMSE vs forecast horizon

Fig. 5  SD1033 external transfer: Physics-Compact vs modern baselines
        apparent-wind vector RMSE vs forecast horizon, three datasets

Fig. 6  External branch decomposition:
        Dual Ridge / Wind-enhanced only / Motion-enhanced only /
        Full Physics-Compact

Fig. 7  10-min BP-STGNN comparison:
        U/V/WS accuracy, WD accuracy, and parameter complexity

Publication style
-----------------
- Uses src/sci_plot_style.py when available.
- Times New Roman / serif fallback.
- 600 dpi PNG + vector PDF.
- Compact SCI-style axes, inward ticks, no figure titles.
- Panel labels (a), (b), ...
- Color-blind-safe palette.

Important
---------
1. Fig. 2 and Fig. 3 are reconstructed directly from the FINAL
   alpha_M=1500 validation predictions and the original train/validation
   arrays. This avoids using obsolete alpha=100 results.

2. Fig. 4 and Fig. 5 need the previously generated modern-baseline
   horizon-level CSVs. The script searches the project automatically.
   If discovery is ambiguous, supply:
       --modern-validation-csv <path>
       --modern-external-csv <path>

3. Modern baselines are NOT recomputed by this plotting script.

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\plot_final_paper_figures.py" `
  --root "D:\project\WindPredict_SaildroneData"

If automatic modern-baseline discovery fails:
python "D:\project\WindPredict_SaildroneData\src\plot_final_paper_figures.py" `
  --root "D:\project\WindPredict_SaildroneData" `
  --modern-validation-csv "PATH_TO_VALIDATION_PER_HORIZON.csv" `
  --modern-external-csv "PATH_TO_EXTERNAL_PER_HORIZON.csv"
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import Ridge


# =============================================================================
# Constants
# =============================================================================

HORIZONS = np.asarray([1, 2, 3, 5, 10], dtype=int)
SEEDS = [500043, 501052, 502061, 503070, 504079]

FINAL_MOTION_ALPHA = 1500.0

BP_VALUES = {
    "Physics-Compact": {
        "U": 0.713950,
        "V": 0.781024,
        "WS": 0.679781,
        "WD": 5.939701,
        "params": 25876,
    },
    "BP-STGNN": {
        "U": 0.748640,
        "V": 0.705060,
        "WS": 0.699400,
        "WD": 5.584000,
        "params": 242580,
    },
}

MODEL_LABELS = {
    "15evh": "Physics-Compact",
    "physiccompact": "Physics-Compact",
    "physicscompact": "Physics-Compact",
    "physicscompactproposed": "Physics-Compact",
    "proposed": "Physics-Compact",
    "persistence": "Persistence",
    "dualridge": "Dual Ridge",
    "ridge": "Dual Ridge",
    "windenhancedridgemotion": "Wind enhanced",
    "windenhanced": "Wind enhanced",
    "ridgewindmotionenhanced": "Motion enhanced",
    "motionenhanced": "Motion enhanced",
    "dlinear": "DLinear",
    "timemixer": "TimeMixer",
    "itransformer": "iTransformer",
    "patchtst": "PatchTST",
}

# Classic/tab10 palette, matching the previously used paper figures.
COLORS = {
    "Observed": "#000000",
    "Persistence": "#1f77b4",
    "Dual Ridge": "#ff7f0e",
    "Physics-Compact": "#d62728",
    "Wind enhanced": "#2ca02c",
    "Motion enhanced": "#9467bd",
    "DLinear": "#ff7f0e",
    "TimeMixer": "#2ca02c",
    "iTransformer": "#d62728",
    "PatchTST": "#9467bd",
    "BP-STGNN": "#1f77b4",
}

MARKERS = {
    "Persistence": "o",
    "Dual Ridge": "s",
    "Physics-Compact": "D",
    "Wind enhanced": "^",
    "Motion enhanced": "v",
    "DLinear": "o",
    "TimeMixer": "s",
    "iTransformer": "^",
    "PatchTST": "v",
}

# Final paper figures use solid lines. Markers/colors distinguish methods.
LINESTYLES = {
    "Persistence": "-",
    "Dual Ridge": "-",
    "Physics-Compact": "-",
    "Wind enhanced": "-",
    "Motion enhanced": "-",
    "DLinear": "-",
    "TimeMixer": "-",
    "iTransformer": "-",
    "PatchTST": "-",
}


# =============================================================================
# Style
# =============================================================================

def load_sci_style(root: Path):
    """Use the project's existing sci_plot_style.py; fallback if unavailable."""
    style_path = root / "src" / "sci_plot_style.py"

    if style_path.exists():
        spec = importlib.util.spec_from_file_location(
            "sci_plot_style_final_paper",
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
                    mod.apply_sci_style(font_size=8.5)

            return mod

    # Fallback reproduces the previously frozen paper style.
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
            "axes.titlesize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "axes.unicode_minus": False,
        }
    )
    return None


def clean_axis(ax):
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.grid(True, which="major", linewidth=0.45, alpha=0.22)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def panel_label(ax, label):
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=9.0,
    )


def save_both(fig, out_base: Path):
    """
    Robust Windows-safe figure saving.
    """
    import os

    out_base = Path(out_base)

    def _norm(base: Path, suffix: str) -> Path:
        p = base.with_suffix(suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        return Path(os.path.abspath(os.path.normpath(str(p))))

    png_path = _norm(out_base, ".png")
    pdf_path = _norm(out_base, ".pdf")

    # Probe that the destination directory is writable.
    probe = png_path.parent / "__write_probe__.tmp"
    with open(str(probe), "wb") as f:
        f.write(b"ok")
    probe.unlink(missing_ok=True)

    try:
        fig.savefig(
            str(png_path),
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
        fig.savefig(
            str(pdf_path),
            bbox_inches="tight",
            facecolor="white",
        )

    except OSError as exc:
        # Retry in a shorter, project-local directory.
        project_root = None
        for parent in [out_base.parent, *out_base.parents]:
            if parent.name.lower() == "windpredict_saildronedata":
                project_root = parent
                break

        if project_root is None:
            project_root = Path.cwd()

        fallback_dir = project_root / "paper_figures"
        fallback_dir.mkdir(parents=True, exist_ok=True)

        fb_png = Path(
            os.path.abspath(
                os.path.normpath(str(fallback_dir / f"{out_base.name}.png"))
            )
        )
        fb_pdf = Path(
            os.path.abspath(
                os.path.normpath(str(fallback_dir / f"{out_base.name}.pdf"))
            )
        )

        print(
            "[WARN] Windows/Pillow rejected the original save path:\n"
            f"       {png_path}\n"
            f"       {exc}\n"
            "[WARN] Retrying in:\n"
            f"       {fallback_dir}"
        )

        fig.savefig(
            str(fb_png),
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
        fig.savefig(
            str(fb_pdf),
            bbox_inches="tight",
            facecolor="white",
        )

        png_path = fb_png
        pdf_path = fb_pdf

    plt.close(fig)
    print(f"[SAVED] {png_path}")
    print(f"[SAVED] {pdf_path}")


# =============================================================================
# General helpers
# =============================================================================

def norm_text(x) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def canonical_model(x) -> str:
    n = norm_text(x)

    if n in MODEL_LABELS:
        return MODEL_LABELS[n]

    # Tolerant matching for filenames / model strings with suffixes.
    for key, value in MODEL_LABELS.items():
        if key and key in n:
            return value

    return str(x)


def find_col(df: pd.DataFrame, aliases, required=True):
    """Find a column by normalized aliases."""
    lookup = {norm_text(c): c for c in df.columns}

    # exact normalized match
    for a in aliases:
        na = norm_text(a)
        if na in lookup:
            return lookup[na]

    # substring match
    for a in aliases:
        na = norm_text(a)
        for nc, c in lookup.items():
            if na in nc or nc in na:
                return c

    if required:
        raise KeyError(
            f"Could not find column for aliases={aliases}. "
            f"Available columns={list(df.columns)}"
        )
    return None


def model_horizon_table(
    df: pd.DataFrame,
    metric_aliases,
    *,
    split=None,
    dataset=None,
    required_models=None,
):
    """
    Standardize arbitrary per-horizon result tables to:
        model, horizon_min, value, std
    """
    work = df.copy()

    model_col = find_col(work, ["model", "method", "architecture"])
    horizon_col = find_col(
        work,
        ["horizon_min", "horizon", "forecast_horizon_min", "forecast_horizon"],
    )
    metric_col = find_col(work, metric_aliases)

    split_col = find_col(work, ["split"], required=False)
    dataset_col = find_col(
        work,
        ["dataset", "external_dataset", "mission"],
        required=False,
    )

    if split is not None and split_col is not None:
        mask = work[split_col].astype(str).str.lower().str.contains(
            str(split).lower(),
            regex=False,
        )
        if mask.any():
            work = work.loc[mask].copy()

    if dataset is not None and dataset_col is not None:
        wanted = norm_text(dataset)
        mask = work[dataset_col].map(norm_text).map(
            lambda x: wanted in x or x in wanted
        )
        work = work.loc[mask].copy()

    work["_model"] = work[model_col].map(canonical_model)
    work["_horizon"] = pd.to_numeric(work[horizon_col], errors="coerce")
    work["_value"] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna(subset=["_horizon", "_value"])

    if required_models is not None:
        work = work[work["_model"].isin(required_models)]

    if work.empty:
        return pd.DataFrame(
            columns=["model", "horizon_min", "value", "std"]
        )

    # Handles raw seed rows and already-aggregated single rows.
    agg = (
        work.groupby(["_model", "_horizon"], as_index=False)["_value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)

    return pd.DataFrame(
        {
            "model": agg["_model"],
            "horizon_min": agg["_horizon"].astype(float),
            "value": agg["mean"].astype(float),
            "std": agg["std"].astype(float),
        }
    )


def discover_horizon_csv(
    search_root: Path,
    *,
    required_models,
    external=False,
):
    """
    Discover a per-horizon CSV containing the modern baselines.
    Only metadata-sized CSVs are inspected.
    """
    if not search_root.exists():
        return None

    candidates = []
    for p in search_root.rglob("*.csv"):
        name = p.name.lower()

        # Strong filename preference.
        score = 0
        if "horizon" in name:
            score += 4
        if "per_horizon" in name:
            score += 4
        if "modern" in name or "baseline" in name:
            score += 2
        if "external" in name:
            score += 2 if external else -1

        try:
            if p.stat().st_size > 50_000_000:
                continue

            head = pd.read_csv(p, nrows=5000)
        except Exception:
            continue

        try:
            model_col = find_col(
                head,
                ["model", "method", "architecture"],
                required=False,
            )
            horizon_col = find_col(
                head,
                ["horizon_min", "horizon", "forecast_horizon"],
                required=False,
            )
            metric_col = find_col(
                head,
                [
                    "AppVector_RMSE_mps",
                    "apparent_vector_RMSE_mps",
                    "apparent_wind_vector_RMSE_mps",
                    "AW_RMSE_mps",
                ],
                required=False,
            )
        except Exception:
            continue

        if model_col is None or horizon_col is None or metric_col is None:
            continue

        names = {canonical_model(v) for v in head[model_col].astype(str)}
        hits = len(set(required_models) & names)
        if hits:
            score += hits * 3
            candidates.append((score, p))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[0], str(x[1])))
    best = candidates[0][1]
    print(f"[AUTO-DISCOVER] modern horizon CSV -> {best}")
    return best


# =============================================================================
# Load final validation predictions and rebuild alpha=1500 linear anchors
# =============================================================================

def load_primary_arrays(dataset_dir: Path):
    train_path = dataset_dir / "train.npz"
    val_path = dataset_dir / "validation.npz"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Expected train.npz and validation.npz in "
            f"{dataset_dir}"
        )

    with np.load(train_path, allow_pickle=False) as z:
        Xtr = np.asarray(z["X"], dtype=np.float32)
        ytr = np.asarray(z["y_raw"], dtype=np.float32)

    with np.load(val_path, allow_pickle=False) as z:
        Xva = np.asarray(z["X"], dtype=np.float32)
        yva = np.asarray(z["y_raw"], dtype=np.float32)

    if Xtr.shape[1:] != (60, 11) or Xva.shape[1:] != (60, 11):
        raise RuntimeError(
            f"Expected X (*,60,11), got train={Xtr.shape}, val={Xva.shape}"
        )

    if ytr.shape[1:] != (5, 6) or yva.shape[1:] != (5, 6):
        raise RuntimeError(
            f"Expected y_raw (*,5,6), got train={ytr.shape}, val={yva.shape}"
        )

    return Xtr, ytr, Xva, yva


def fit_dual_ridge(Xtr, ytr, Xva):
    """
    Reconstruct the FINAL linear anchors:
      - dedicated true-wind anchor: alpha_W = 0
      - joint platform-state anchor: alpha_M = 1500

    X is already expressed using the frozen training-set scaling convention.
    """
    Xw_tr = Xtr[:, :, 0:4].reshape(len(Xtr), -1)
    Xw_va = Xva[:, :, 0:4].reshape(len(Xva), -1)

    yw_tr = ytr[:, :, 0:2].reshape(len(ytr), -1)

    wind_ridge = Ridge(alpha=0.0, fit_intercept=True)
    wind_ridge.fit(Xw_tr, yw_tr)
    pred_wind = wind_ridge.predict(Xw_va).reshape(-1, 5, 2)

    Xj_tr = Xtr.reshape(len(Xtr), -1)
    Xj_va = Xva.reshape(len(Xva), -1)
    yj_tr = ytr.reshape(len(ytr), -1)

    joint_ridge = Ridge(alpha=FINAL_MOTION_ALPHA, fit_intercept=True)
    joint_ridge.fit(Xj_tr, yj_tr)
    pred_joint = joint_ridge.predict(Xj_va).reshape(-1, 5, 6)

    pred_motion = pred_joint[:, :, 2:6]

    # Normalize predicted HDG pair.
    q = pred_motion[:, :, 2:4]
    qn = np.sqrt(np.sum(q * q, axis=-1, keepdims=True))
    pred_motion[:, :, 2:4] = q / np.maximum(qn, 1e-12)

    return pred_wind.astype(np.float32), pred_motion.astype(np.float32)


def load_final_seed_predictions(final_vh_dir: Path):
    seed_data = []

    for seed in SEEDS:
        p = final_vh_dir / f"15E_VH_seed_{seed}_validation_predictions.npz"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing final alpha=1500 prediction file: {p}"
            )

        with np.load(p, allow_pickle=False) as z:
            required = ["y_raw", "frozen_truewind", "joint_vessel_hdg"]
            missing = [k for k in required if k not in z]
            if missing:
                raise KeyError(f"{p}: missing keys {missing}")

            seed_data.append(
                {
                    "seed": seed,
                    "y_raw": np.asarray(z["y_raw"], dtype=np.float32),
                    "wind": np.asarray(
                        z["frozen_truewind"], dtype=np.float32
                    ),
                    "motion": np.asarray(
                        z["joint_vessel_hdg"], dtype=np.float32
                    ),
                }
            )

    # Assert identical truth.
    y0 = seed_data[0]["y_raw"]
    for d in seed_data[1:]:
        if not np.allclose(d["y_raw"], y0, atol=1e-7, rtol=0):
            raise RuntimeError("Seed prediction files do not share identical y_raw.")

    return seed_data


def vector_rmse_per_horizon(true, pred):
    err = np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    return np.sqrt(np.mean(np.sum(err * err, axis=-1), axis=0))


def app_earth(wind, motion):
    return (
        np.asarray(wind, dtype=np.float64)
        - np.asarray(motion, dtype=np.float64)[:, :, 0:2]
    )


def metric_bundle(y_raw, wind, motion):
    tw_true = y_raw[:, :, 0:2]
    motion_true = y_raw[:, :, 2:6]
    vessel_true = motion_true[:, :, 0:2]

    true_app = app_earth(tw_true, motion_true)
    pred_app = app_earth(wind, motion)

    return {
        "TrueWind": vector_rmse_per_horizon(tw_true, wind),
        "Vessel": vector_rmse_per_horizon(
            vessel_true,
            motion[:, :, 0:2],
        ),
        "AppVector": vector_rmse_per_horizon(true_app, pred_app),
    }


def seed_metric_stats(seed_data):
    metrics = [metric_bundle(d["y_raw"], d["wind"], d["motion"])
               for d in seed_data]

    out = {}
    for key in ["TrueWind", "Vessel", "AppVector"]:
        arr = np.stack([m[key] for m in metrics], axis=0)
        out[key] = {
            "mean": np.mean(arr, axis=0),
            "std": np.std(arr, axis=0, ddof=1),
        }

    return out


def load_persistence_horizon_metrics(dataset_dir: Path):
    """
    Read the frozen Persistence horizon metrics from the original joint
    baseline table. Persistence is model-independent, so these values remain
    valid after changing the final Ridge anchors to alpha_W=0 and alpha_M=1500.
    """
    p = dataset_dir / "joint_baseline_v0_1" / "metrics_per_horizon.csv"
    if not p.exists():
        raise FileNotFoundError(
            "Persistence metrics are required for Fig. 2, but the frozen "
            f"baseline table was not found: {p}"
        )

    df = pd.read_csv(p)

    metric_aliases = {
        "TrueWind": [
            "wind_vector_RMSE_mps",
            "true_wind_vector_RMSE_mps",
            "TrueWind_vector_RMSE_mps",
            "WindVector_RMSE_mps",
        ],
        "Vessel": [
            "vessel_vector_RMSE_mps",
            "Vessel_RMSE_mps",
            "vessel_RMSE_mps",
        ],
        "AppVector": [
            "apparent_vector_RMSE_mps",
            "apparent_wind_vector_RMSE_mps",
            "AppVector_RMSE_mps",
            "AW_RMSE_mps",
        ],
    }

    out = {}
    for key, aliases in metric_aliases.items():
        tab = model_horizon_table(
            df,
            aliases,
            split="validation",
            required_models=["Persistence"],
        )
        if tab.empty:
            raise RuntimeError(
                f"Could not read Persistence {key} horizon metrics from {p}. "
                f"Columns are: {list(df.columns)}"
            )

        tab = tab.sort_values("horizon_min")
        wanted = []
        for h in HORIZONS:
            row = tab[np.isclose(tab["horizon_min"].to_numpy(float), float(h))]
            if row.empty:
                raise RuntimeError(
                    f"Persistence metric {key}: missing horizon {h} min in {p}"
                )
            wanted.append(float(row.iloc[0]["value"]))
        out[key] = np.asarray(wanted, dtype=float)

    print("[FIG. 2] Persistence loaded from frozen baseline metrics:")
    for key in ["TrueWind", "Vessel", "AppVector"]:
        print(f"         {key:9s}: {np.array2string(out[key], precision=6)}")

    return out


# =============================================================================
# Fig. 2 — branch-resolved horizon performance
# =============================================================================

def plot_fig2(
    output_dir: Path,
    dataset_dir: Path,
    y_val,
    seed_data,
    ridge_wind,
    ridge_motion,
):
    """
    Final Fig. 2:
      (a) true-wind vector RMSE
      (b) vessel-vector RMSE
      (c) apparent-wind vector RMSE / branch decomposition

    Visual refinements:
      - thinner solid lines and smaller markers;
      - one common legend above the three panels;
      - a compact inset in panel (a) showing RMSE reduction relative
        to Persistence, because the three absolute curves nearly overlap.
    """
    proposed = seed_metric_stats(seed_data)
    ridge = metric_bundle(y_val, ridge_wind, ridge_motion)
    persistence = load_persistence_horizon_metrics(dataset_dir)

    # Branch combinations use five-seed means / stds.
    app_wind_only = []
    app_motion_only = []
    app_full = []

    for d in seed_data:
        y = d["y_raw"]
        true_app = app_earth(y[:, :, 0:2], y[:, :, 2:6])

        app_w = app_earth(d["wind"], ridge_motion)
        app_m = app_earth(ridge_wind, d["motion"])
        app_f = app_earth(d["wind"], d["motion"])

        app_wind_only.append(vector_rmse_per_horizon(true_app, app_w))
        app_motion_only.append(vector_rmse_per_horizon(true_app, app_m))
        app_full.append(vector_rmse_per_horizon(true_app, app_f))

    app_wind_only = np.stack(app_wind_only)
    app_motion_only = np.stack(app_motion_only)
    app_full = np.stack(app_full)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.15, 2.55),
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.992,
        bottom=0.19,
        top=0.80,
        wspace=0.34,
    )

    common_kw = dict(
        linewidth=1.15,
        markersize=4.15,
        markeredgewidth=0.55,
    )

    # ----------------------------- (a) True wind -----------------------------
    ax = axes[0]
    ax.plot(
        HORIZONS,
        persistence["TrueWind"],
        color=COLORS["Persistence"],
        marker=MARKERS["Persistence"],
        linestyle="-",
        label="Persistence",
        zorder=2,
        **common_kw,
    )
    ax.plot(
        HORIZONS,
        ridge["TrueWind"],
        color=COLORS["Dual Ridge"],
        marker=MARKERS["Dual Ridge"],
        linestyle="-",
        label="Dual Ridge",
        zorder=3,
        **common_kw,
    )
    ax.errorbar(
        HORIZONS,
        proposed["TrueWind"]["mean"],
        yerr=proposed["TrueWind"]["std"],
        color=COLORS["Physics-Compact"],
        marker=MARKERS["Physics-Compact"],
        linestyle="-",
        capsize=1.8,
        elinewidth=0.75,
        label="Physics-Compact",
        zorder=4,
        **common_kw,
    )
    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel(r"True-wind vector RMSE (m s$^{-1}$)")
    ax.set_xticks(HORIZONS)
    panel_label(ax, "(a)")
    clean_axis(ax)

    # Small skill inset: the absolute RMSE curves are intentionally close,
    # so showing the percentage reduction relative to Persistence makes the
    # separation visible without artificially shifting x-coordinates or
    # distorting the main y-axis.
    p = persistence["TrueWind"]
    skill_ridge = 100.0 * (p - ridge["TrueWind"]) / p
    skill_prop = 100.0 * (p - proposed["TrueWind"]["mean"]) / p

    axins = ax.inset_axes([0.47, 0.10, 0.49, 0.31])
    axins.plot(
        HORIZONS,
        skill_ridge,
        color=COLORS["Dual Ridge"],
        marker=MARKERS["Dual Ridge"],
        linestyle="-",
        linewidth=0.85,
        markersize=2.8,
    )
    axins.plot(
        HORIZONS,
        skill_prop,
        color=COLORS["Physics-Compact"],
        marker=MARKERS["Physics-Compact"],
        linestyle="-",
        linewidth=0.85,
        markersize=2.8,
    )
    axins.axhline(0.0, color="0.45", linewidth=0.55)
    axins.set_xticks(HORIZONS)
    axins.tick_params(
        axis="both",
        which="major",
        labelsize=5.4,
        length=2.0,
        width=0.55,
        direction="in",
    )
    axins.minorticks_off()
    axins.grid(True, linewidth=0.3, alpha=0.18)
    axins.text(
        0.03,
        0.94,
        "RMSE reduction vs Persistence (%)",
        transform=axins.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
    )
    for spine in axins.spines.values():
        spine.set_linewidth(0.6)

    # ----------------------------- (b) Vessel -----------------------------
    ax = axes[1]
    ax.plot(
        HORIZONS,
        persistence["Vessel"],
        color=COLORS["Persistence"],
        marker=MARKERS["Persistence"],
        linestyle="-",
        label="Persistence",
        zorder=2,
        **common_kw,
    )
    ax.plot(
        HORIZONS,
        ridge["Vessel"],
        color=COLORS["Dual Ridge"],
        marker=MARKERS["Dual Ridge"],
        linestyle="-",
        label="Dual Ridge",
        zorder=3,
        **common_kw,
    )
    ax.errorbar(
        HORIZONS,
        proposed["Vessel"]["mean"],
        yerr=proposed["Vessel"]["std"],
        color=COLORS["Physics-Compact"],
        marker=MARKERS["Physics-Compact"],
        linestyle="-",
        capsize=1.8,
        elinewidth=0.75,
        label="Physics-Compact",
        zorder=4,
        **common_kw,
    )
    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel(r"Vessel-vector RMSE (m s$^{-1}$)")
    ax.set_xticks(HORIZONS)
    panel_label(ax, "(b)")
    clean_axis(ax)

    # ----------------------------- (c) Apparent wind -----------------------------
    ax = axes[2]

    branch_series = {
        "Persistence": (
            persistence["AppVector"],
            np.zeros(5),
        ),
        "Dual Ridge": (
            ridge["AppVector"],
            np.zeros(5),
        ),
        "Wind enhanced": (
            np.mean(app_wind_only, axis=0),
            np.std(app_wind_only, axis=0, ddof=1),
        ),
        "Motion enhanced": (
            np.mean(app_motion_only, axis=0),
            np.std(app_motion_only, axis=0, ddof=1),
        ),
        "Physics-Compact": (
            np.mean(app_full, axis=0),
            np.std(app_full, axis=0, ddof=1),
        ),
    }

    for name, (mean, std) in branch_series.items():
        if np.allclose(std, 0):
            ax.plot(
                HORIZONS,
                mean,
                color=COLORS[name],
                marker=MARKERS[name],
                linestyle="-",
                label=name,
                zorder=2,
                **common_kw,
            )
        else:
            ax.errorbar(
                HORIZONS,
                mean,
                yerr=std,
                color=COLORS[name],
                marker=MARKERS[name],
                linestyle="-",
                capsize=1.7,
                elinewidth=0.70,
                label=name,
                zorder=3,
                **common_kw,
            )

    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel(r"Apparent-wind vector RMSE (m s$^{-1}$)")
    ax.set_xticks(HORIZONS)
    panel_label(ax, "(c)")
    clean_axis(ax)

    # One common legend above all panels: no curve is covered.
    legend_order = [
        "Persistence",
        "Dual Ridge",
        "Wind enhanced",
        "Motion enhanced",
        "Physics-Compact",
    ]
    legend_handles = [
        Line2D(
            [0], [0],
            color=COLORS[name],
            marker=MARKERS[name],
            linestyle="-",
            linewidth=1.15,
            markersize=4.1,
            markeredgewidth=0.55,
            label=name,
        )
        for name in legend_order
    ]
    fig.legend(
        handles=legend_handles,
        labels=legend_order,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        frameon=False,
        fontsize=7.0,
        handlelength=2.0,
        columnspacing=1.15,
        handletextpad=0.45,
    )

    save_both(fig, output_dir / "Fig02_SD1090_branch_horizon")


# =============================================================================
# Fig. 3 — observed vs predicted time series
# =============================================================================

def rolling_std_best_segment(x, window):
    x = np.asarray(x, dtype=float)
    if len(x) <= window:
        return 0, len(x)

    s = pd.Series(x)
    roll = s.rolling(window=window, min_periods=window).std()
    end = int(np.nanargmax(roll.to_numpy()))
    start = max(0, end - window + 1)
    return start, min(len(x), start + window)


def plot_fig3(
    output_dir: Path,
    y_val,
    seed_data,
    ridge_wind,
    ridge_motion,
    *,
    horizon_min=10,
    window_hours=12,
):
    hidx = int(np.where(HORIZONS == horizon_min)[0][0])

    proposed_wind = np.mean(
        np.stack([d["wind"] for d in seed_data], axis=0),
        axis=0,
    )
    proposed_motion = np.mean(
        np.stack([d["motion"] for d in seed_data], axis=0),
        axis=0,
    )

    true_w = y_val[:, hidx, 0:2]
    true_v = y_val[:, hidx, 2:4]

    ridge_w = ridge_wind[:, hidx, :]
    ridge_v = ridge_motion[:, hidx, 0:2]

    prop_w = proposed_wind[:, hidx, :]
    prop_v = proposed_motion[:, hidx, 0:2]

    ws_true = np.linalg.norm(true_w, axis=1)
    ws_ridge = np.linalg.norm(ridge_w, axis=1)
    ws_prop = np.linalg.norm(prop_w, axis=1)

    aws_true = np.linalg.norm(true_w - true_v, axis=1)
    aws_ridge = np.linalg.norm(ridge_w - ridge_v, axis=1)
    aws_prop = np.linalg.norm(prop_w - prop_v, axis=1)

    # Select a representative high-variability segment using ONLY observed WS,
    # never using model error. This avoids performance-based cherry-picking.
    window = max(60, int(round(window_hours * 60)))
    start, stop = rolling_std_best_segment(ws_true, window)

    t = np.arange(stop - start, dtype=float) / 60.0

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.15, 4.25),
        sharex=True,
        constrained_layout=True,
    )

    ax = axes[0]
    ax.plot(
        t,
        ws_true[start:stop],
        color=COLORS["Observed"],
        linewidth=1.25,
        label="Observed",
    )
    ax.plot(
        t,
        ws_ridge[start:stop],
        color=COLORS["Dual Ridge"],
        linestyle=LINESTYLES["Dual Ridge"],
        linewidth=1.05,
        label="Dual Ridge",
    )
    ax.plot(
        t,
        ws_prop[start:stop],
        color=COLORS["Physics-Compact"],
        linewidth=1.15,
        label="Physics-Compact",
    )
    ax.set_ylabel(r"True wind speed (m s$^{-1}$)")
    panel_label(ax, "(a)")
    clean_axis(ax)
    ax.legend(ncol=3, loc="upper right")

    ax = axes[1]
    ax.plot(
        t,
        aws_true[start:stop],
        color=COLORS["Observed"],
        linewidth=1.25,
        label="Observed",
    )
    ax.plot(
        t,
        aws_ridge[start:stop],
        color=COLORS["Dual Ridge"],
        linestyle=LINESTYLES["Dual Ridge"],
        linewidth=1.05,
        label="Dual Ridge",
    )
    ax.plot(
        t,
        aws_prop[start:stop],
        color=COLORS["Physics-Compact"],
        linewidth=1.15,
        label="Physics-Compact",
    )
    ax.set_xlabel("Time within selected validation interval (h)")
    ax.set_ylabel(r"Apparent wind speed (m s$^{-1}$)")
    panel_label(ax, "(b)")
    clean_axis(ax)

    # Small non-intrusive note; selection uses truth variability only.
    axes[1].text(
        0.995,
        0.03,
        f"{horizon_min}-min horizon; {window_hours:g}-h high-variability segment",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
    )

    save_both(fig, output_dir / "Fig03_observed_vs_predicted_timeseries")


# =============================================================================
# Fig. 4 — modern baselines on SD1090 validation
# =============================================================================

def plot_fig4(
    output_dir: Path,
    modern_csv: Path,
    final_vh_dir: Path,
):
    """
    Final Fig. 4: modern baselines on SD1090 validation.

    IMPORTANT:
    The Physics-Compact curve is reconstructed directly from the final
    alpha_M=1500 five-seed validation prediction files. It is not inferred
    from a model-name string in a CSV, so it cannot silently disappear.

    Palette follows the previously used modern-baseline figure:
      Physics-Compact  blue
      DLinear          orange
      TimeMixer        green
      iTransformer     red
      PatchTST         purple
    All curves are solid.
    """
    modern = pd.read_csv(modern_csv)

    req = ["DLinear", "TimeMixer", "iTransformer", "PatchTST"]
    tab = model_horizon_table(
        modern,
        [
            "AppVector_RMSE_mps",
            "apparent_vector_RMSE_mps",
            "apparent_wind_vector_RMSE_mps",
            "AW_RMSE_mps",
        ],
        split="validation",
        required_models=req,
    )

    missing = sorted(set(req) - set(tab["model"]))
    if missing:
        raise RuntimeError(
            f"{modern_csv} does not contain all required modern models. "
            f"Missing: {missing}"
        )

    # Rebuild the FINAL proposed curve directly from five alpha=1500 seeds.
    seed_data = load_final_seed_predictions(final_vh_dir)
    prop_arr = []
    for d in seed_data:
        true_app = app_earth(
            d["y_raw"][:, :, 0:2],
            d["y_raw"][:, :, 2:6],
        )
        pred_app = app_earth(d["wind"], d["motion"])
        prop_arr.append(vector_rmse_per_horizon(true_app, pred_app))

    prop_arr = np.stack(prop_arr, axis=0)
    prop_mean = np.mean(prop_arr, axis=0)
    prop_std = np.std(prop_arr, axis=0, ddof=1)

    print("[FIG. 4] FINAL Physics-Compact AppVector RMSE by horizon:")
    print("         mean =", np.array2string(prop_mean, precision=6))
    print("         std  =", np.array2string(prop_std, precision=6))

    # Local palette deliberately matches the earlier modern-baseline figure.
    fig4_colors = {
        "Physics-Compact": "#1f77b4",
        "DLinear": "#ff7f0e",
        "TimeMixer": "#2ca02c",
        "iTransformer": "#d62728",
        "PatchTST": "#9467bd",
    }
    fig4_markers = {
        "Physics-Compact": "D",
        "DLinear": "o",
        "TimeMixer": "s",
        "iTransformer": "^",
        "PatchTST": "v",
    }

    fig, ax = plt.subplots(
        figsize=(6.15, 3.65),
        constrained_layout=True,
    )

    # Proposed model first, with five-seed error bars.
    ax.errorbar(
        HORIZONS,
        prop_mean,
        yerr=prop_std,
        color=fig4_colors["Physics-Compact"],
        marker=fig4_markers["Physics-Compact"],
        linestyle="-",
        linewidth=1.25,
        markersize=4.4,
        markeredgewidth=0.7,
        elinewidth=0.80,
        capsize=1.8,
        label="Physics-Compact",
        zorder=5,
    )

    # Modern baselines.
    for name in ["DLinear", "TimeMixer", "iTransformer", "PatchTST"]:
        d = tab[tab["model"] == name].sort_values("horizon_min")
        if d.empty:
            raise RuntimeError(f"Fig. 4: no data for {name}")

        ax.plot(
            d["horizon_min"],
            d["value"],
            color=fig4_colors[name],
            marker=fig4_markers[name],
            linestyle="-",
            linewidth=1.20,
            markersize=4.2,
            markeredgewidth=0.7,
            label=name,
        )

        print(
            f"[FIG. 4] {name:16s}: "
            f"{np.array2string(d['value'].to_numpy(float), precision=6)}"
        )

    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel(r"Apparent-wind vector RMSE (m s$^{-1}$)")
    ax.set_xticks(HORIZONS)
    clean_axis(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=7.5,
        columnspacing=1.2,
        handlelength=2.0,
        frameon=False,
    )

    save_both(fig, output_dir / "Fig04_modern_baselines_SD1090")


# =============================================================================
# Fig. 5 — external modern-baseline horizon curves
# =============================================================================

def load_final_external_horizon(final_ext_dir: Path):
    p = final_ext_dir / "15F_external_per_horizon.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def plot_fig5(
    output_dir: Path,
    modern_external_csv: Path,
    final_ext_dir: Path,
):
    modern = pd.read_csv(modern_external_csv)
    final_ext = load_final_external_horizon(final_ext_dir)

    datasets = [
        ("SD1033-2024-matched", "SD1033-2024 matched"),
        ("SD1033-2023-full", "SD1033-2023"),
        ("SD1033-2022-full", "SD1033-2022"),
    ]

    modern_models = ["DLinear", "TimeMixer", "iTransformer", "PatchTST"]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.15, 2.45),
        constrained_layout=True,
    )

    for j, (dataset_key, title) in enumerate(datasets):
        ax = axes[j]

        prop = model_horizon_table(
            final_ext,
            [
                "AppVector_RMSE_mps",
                "apparent_vector_RMSE_mps",
                "AW_RMSE_mps",
            ],
            dataset=dataset_key,
            required_models=["Physics-Compact"],
        )

        if prop.empty:
            # Explicit 15E-VH model name if canonical detection failed.
            model_col = find_col(final_ext, ["model"])
            subset = final_ext[
                final_ext[model_col].astype(str).str.contains(
                    "15E-VH",
                    regex=False,
                )
            ]
            prop = model_horizon_table(
                subset,
                ["AppVector_RMSE_mps"],
                dataset=dataset_key,
            )
            prop["model"] = "Physics-Compact"

        mod = model_horizon_table(
            modern,
            [
                "AppVector_RMSE_mps",
                "apparent_vector_RMSE_mps",
                "AW_RMSE_mps",
            ],
            dataset=dataset_key,
            required_models=modern_models,
        )

        missing = sorted(set(modern_models) - set(mod["model"]))
        if missing:
            raise RuntimeError(
                f"{modern_external_csv}: dataset={dataset_key} "
                f"missing modern models {missing}"
            )

        combined = pd.concat([prop, mod], ignore_index=True)

        for name in [
            "Physics-Compact",
            "DLinear",
            "TimeMixer",
            "iTransformer",
            "PatchTST",
        ]:
            d = combined[combined["model"] == name].sort_values(
                "horizon_min"
            )
            if d.empty:
                continue

            if name == "Physics-Compact":
                ax.errorbar(
                    d["horizon_min"],
                    d["value"],
                    yerr=d["std"],
                    color=COLORS[name],
                    marker=MARKERS[name],
                    linestyle=LINESTYLES[name],
                    capsize=1.8,
                    label=name,
                )
            else:
                ax.plot(
                    d["horizon_min"],
                    d["value"],
                    color=COLORS[name],
                    marker=MARKERS[name],
                    linestyle=LINESTYLES[name],
                    label=name,
                )

        ax.set_xlabel("Forecast horizon (min)")
        if j == 0:
            ax.set_ylabel(r"Apparent-wind vector RMSE (m s$^{-1}$)")
        ax.set_xticks(HORIZONS)
        ax.set_title(title, pad=4)
        panel_label(ax, f"({chr(ord('a') + j)})")
        clean_axis(ax)

    # One legend for the entire figure.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.075),
        ncol=5,
        frameon=False,
        fontsize=7.3,
    )

    save_both(fig, output_dir / "Fig05_external_modern_horizons")


# =============================================================================
# Fig. 6 — external branch decomposition
# =============================================================================

def plot_fig6(output_dir: Path, final_ext_dir: Path):
    p = final_ext_dir / "15F_external_branch_decomposition.csv"
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)

    dataset_col = find_col(df, ["dataset"])
    model_col = find_col(df, ["model"])

    df["_dataset"] = df[dataset_col].astype(str)
    df["_model"] = df[model_col].map(canonical_model)

    dataset_order = [
        "SD1033-2024-matched",
        "SD1033-2023-full",
        "SD1033-2022-full",
    ]
    dataset_labels = [
        "2024 matched",
        "2023",
        "2022",
    ]

    models = [
        "Dual Ridge",
        "Wind enhanced",
        "Motion enhanced",
        "Physics-Compact",
    ]

    metrics = [
        (
            ["AppVector_RMSE_mps", "apparent_vector_RMSE_mps"],
            r"Apparent-wind vector RMSE (m s$^{-1}$)",
        ),
        (
            ["AWS_RMSE_mps"],
            r"AWS RMSE (m s$^{-1}$)",
        ),
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.65),
        constrained_layout=True,
    )

    x = np.arange(len(dataset_order), dtype=float)
    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(models))

    for j, (metric_aliases, ylabel) in enumerate(metrics):
        ax = axes[j]
        metric_col = find_col(df, metric_aliases)

        for k, model in enumerate(models):
            means = []
            stds = []

            for ds in dataset_order:
                wanted = norm_text(ds)
                d = df[
                    df["_dataset"].map(norm_text).map(
                        lambda s: wanted in s or s in wanted
                    )
                    & (df["_model"] == model)
                ]

                vals = pd.to_numeric(
                    d[metric_col],
                    errors="coerce",
                ).dropna().to_numpy()

                if len(vals) == 0:
                    means.append(np.nan)
                    stds.append(np.nan)
                else:
                    means.append(float(np.mean(vals)))
                    stds.append(
                        float(np.std(vals, ddof=1))
                        if len(vals) > 1 else 0.0
                    )

            ax.bar(
                x + offsets[k],
                means,
                width=width,
                yerr=stds,
                capsize=2.0,
                color=COLORS[model],
                edgecolor="black",
                linewidth=0.45,
                label=model,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels)
        ax.set_ylabel(ylabel)
        panel_label(ax, f"({chr(ord('a') + j)})")
        clean_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
        frameon=False,
    )

    save_both(fig, output_dir / "Fig06_external_branch_decomposition")


# =============================================================================
# Fig. 7 — BP-STGNN accuracy / complexity
# =============================================================================

def plot_fig7(output_dir: Path):
    methods = ["Physics-Compact", "BP-STGNN"]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.15, 2.55),
        constrained_layout=True,
    )

    # (a) U/V/WS
    ax = axes[0]
    metrics = ["U", "V", "WS"]
    x = np.arange(len(metrics))
    width = 0.36

    for j, method in enumerate(methods):
        vals = [BP_VALUES[method][m] for m in metrics]
        bars = ax.bar(
            x + (j - 0.5) * width,
            vals,
            width=width,
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.5,
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel(r"RMSE (m s$^{-1}$)")
    panel_label(ax, "(a)")
    clean_axis(ax)
    ax.legend(loc="upper right", fontsize=7.0)

    # (b) WD
    ax = axes[1]
    vals = [BP_VALUES[m]["WD"] for m in methods]
    bars = ax.bar(
        np.arange(2),
        vals,
        width=0.55,
        color=[COLORS[m] for m in methods],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["Physics-\nCompact", "BP-STGNN"])
    ax.set_ylabel(r"Wind-direction RMSE ($^\circ$)")
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.035,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )
    panel_label(ax, "(b)")
    clean_axis(ax)

    # (c) parameter count
    ax = axes[2]
    params = [BP_VALUES[m]["params"] for m in methods]
    bars = ax.bar(
        np.arange(2),
        params,
        width=0.55,
        color=[COLORS[m] for m in methods],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["Physics-\nCompact", "BP-STGNN"])
    ax.set_ylabel("Trainable neural parameters")
    for b, v in zip(bars, params):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v * 1.08,
            f"{v:,}",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )

    reduction = (
        1.0
        - BP_VALUES["Physics-Compact"]["params"]
        / BP_VALUES["BP-STGNN"]["params"]
    ) * 100.0

    ax.text(
        0.5,
        0.94,
        f"{reduction:.2f}% fewer",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        fontweight="bold",
    )
    panel_label(ax, "(c)")
    clean_axis(ax)

    save_both(fig, output_dir / "Fig07_BPSTGNN_accuracy_complexity")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"D:\project\WindPredict_SaildroneData"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--modern-validation-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--modern-external-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--timeseries-horizon",
        type=int,
        choices=[1, 2, 3, 5, 10],
        default=10,
    )
    parser.add_argument(
        "--timeseries-hours",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--only",
        type=int,
        nargs="*",
        choices=[2, 3, 4, 5, 6, 7],
        default=None,
        help="Generate only selected figure numbers.",
    )

    args = parser.parse_args()
    root = args.root.resolve()

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "figures" / "final_paper"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    load_sci_style(root)

    dataset_dir = (
        root
        / "data"
        / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
    )
    final_vh_dir = (
        root
        / "data"
        / "forecasting"
        / "15E_VH_AlphaSearch_v0_1"
        / "full_retrain"
        / "alpha_1500"
    )
    final_ext_dir = (
        root
        / "data"
        / "forecasting"
        / "15F_Frozen_15EVH_alpha1500_External_v0_1"
    )

    selected = set(args.only or [2, 3, 4, 5, 6, 7])

    print("=" * 100)
    print("FINAL PAPER FIGURE GENERATION")
    print("=" * 100)
    print(f"root          : {root}")
    print(f"output_dir    : {output_dir}")
    print(f"dataset_dir   : {dataset_dir}")
    print(f"final_vh_dir  : {final_vh_dir}")
    print(f"final_ext_dir : {final_ext_dir}")
    print(f"figures       : {sorted(selected)}")
    print("=" * 100)

    # Figures 2 and 3 need final validation predictions and final linear anchors.
    if 2 in selected or 3 in selected:
        print("[LOAD] Primary train/validation arrays.")
        Xtr, ytr, Xva, yva = load_primary_arrays(dataset_dir)

        print("[FIT] Rebuilding final dual linear anchors:")
        print("      alpha_W = 0")
        print(f"      alpha_M = {FINAL_MOTION_ALPHA:g}")
        ridge_wind, ridge_motion = fit_dual_ridge(Xtr, ytr, Xva)

        print("[LOAD] Final alpha=1500 five-seed validation predictions.")
        seed_data = load_final_seed_predictions(final_vh_dir)

        if 2 in selected:
            print("[FIG. 2] SD1090 branch-resolved horizon performance.")
            plot_fig2(
                output_dir,
                dataset_dir,
                yva,
                seed_data,
                ridge_wind,
                ridge_motion,
            )

        if 3 in selected:
            print("[FIG. 3] Observed vs predicted WS/AWS time series.")
            plot_fig3(
                output_dir,
                yva,
                seed_data,
                ridge_wind,
                ridge_motion,
                horizon_min=args.timeseries_horizon,
                window_hours=args.timeseries_hours,
            )

    # Fig. 4: discover modern validation horizon table.
    if 4 in selected:
        modern_val = args.modern_validation_csv
        if modern_val is None:
            modern_val = discover_horizon_csv(
                root / "data" / "forecasting",
                required_models=[
                    "DLinear",
                    "TimeMixer",
                    "iTransformer",
                    "PatchTST",
                ],
                external=False,
            )

        if modern_val is None:
            warnings.warn(
                "Fig. 4 skipped: could not auto-discover a validation "
                "per-horizon CSV containing all four modern baselines. "
                "Re-run with --modern-validation-csv PATH.",
                RuntimeWarning,
            )
        else:
            print(f"[FIG. 4] modern validation source = {modern_val}")
            plot_fig4(output_dir, modern_val, final_vh_dir)

    # Fig. 5: discover modern external horizon table.
    if 5 in selected:
        modern_ext = args.modern_external_csv
        if modern_ext is None:
            modern_ext = discover_horizon_csv(
                root / "data",
                required_models=[
                    "DLinear",
                    "TimeMixer",
                    "iTransformer",
                    "PatchTST",
                ],
                external=True,
            )

        if modern_ext is None:
            warnings.warn(
                "Fig. 5 skipped: could not auto-discover an external "
                "per-horizon CSV containing all four modern baselines. "
                "Re-run with --modern-external-csv PATH.",
                RuntimeWarning,
            )
        else:
            print(f"[FIG. 5] modern external source = {modern_ext}")
            plot_fig5(output_dir, modern_ext, final_ext_dir)

    if 6 in selected:
        print("[FIG. 6] External branch decomposition.")
        plot_fig6(output_dir, final_ext_dir)

    if 7 in selected:
        print("[FIG. 7] BP-STGNN accuracy-complexity comparison.")
        plot_fig7(output_dir)

    print("")
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
