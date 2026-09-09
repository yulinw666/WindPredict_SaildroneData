# -*- coding: utf-8 -*-
"""
10A_mechanism_analysis_branch_gate_regime.py

Post-hoc mechanism analysis for the FROZEN Physics-Compact-Vessel-Residual
model. No training, fine-tuning, seed selection, scaler fitting, architecture
change, or loss-weight change is performed.

Questions:
1) How different are Ridge wind-branch and vessel-branch residuals?
2) When does the compact vessel gate become larger?
3) Does the learned correction point toward the vessel residual Ridge needs?
4) Is the small Physics-Compact vs Compact gain concentrated in specific
   vessel/wind regimes?

Regime thresholds are quartiles estimated ONCE from frozen SD1090 OUTER TRAIN
history features, then applied unchanged to validation and SD1033 datasets.

The gate is an amplitude-modulation variable, NOT calibrated confidence/UQ.
All correlations are descriptive associations, NOT causal evidence.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import platform
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "0.1.0-mechanism"
SHARED_08C = "08C_frozen_external_inference.py"
SHARED_07C = "07C_confirm_and_compact_vessel_residual.py"

DEFAULT_DATASET_DIR = (
    r"D:\project\WindPredict_SaildroneData\data\forecasting"
    r"\SD1090_TPOS2024_JointForecasting_v0_1"
)
DEFAULT_COMPACT_DIR = (
    DEFAULT_DATASET_DIR
    + r"\confirm_and_compact_vessel_residual_v0_1"
)
DEFAULT_EXTERNAL_ROOT = (
    r"D:\project\WindPredict_SaildroneData\data\external"
    r"\SD1033_external_forecasting_v0_1"
)
DEFAULT_DATASETS = [
    "SD1090_validation",
    "SD1033_2024_matched_SD1090",
    "SD1033_2023_full",
    "SD1033_2022_full",
]

MODEL_COMPACT = "Frozen-Compact-Vessel-Residual"
MODEL_PHYSICS = "Frozen-Physics-Compact-Vessel-Residual"
MODEL_FILES = {
    MODEL_COMPACT: "Compact_Vessel_Residual.pt",
    MODEL_PHYSICS: "Physics_Compact_Vessel_Residual.pt",
}
HORIZONS = [1, 2, 3, 5, 10]
FEATURES = [
    "UWND_MEAN", "VWND_MEAN", "TEMP_AIR_MEAN", "RH_MEAN", "SOG",
    "COG_sin", "COG_cos", "HDG_sin", "HDG_cos",
    "WING_ANGLE_sin", "WING_ANGLE_cos",
]
TARGETS = [
    "UWND_MEAN", "VWND_MEAN", "VESSEL_EAST_MPS", "VESSEL_NORTH_MPS",
    "HDG_sin", "HDG_cos",
]
REGIME_FEATURES = [
    "current_SOG_mps",
    "current_wind_speed_mps",
    "wind_variability_10min_mps",
    "SOG_variability_10min_mps",
    "COG_turn_rate_10min_deg_per_min",
    "HDG_turn_rate_10min_deg_per_min",
    "wing_change_rate_10min_deg_per_min",
]
REGIME_LABELS = ["Q1", "Q2", "Q3", "Q4"]
DATASET_LABELS = {
    "SD1090_validation": "SD1090 validation",
    "SD1033_2024_matched_SD1090": "SD1033-2024 matched",
    "SD1033_2023_full": "SD1033-2023",
    "SD1033_2022_full": "SD1033-2022",
}


def log(s=""):
    print(s, flush=True)


def load_module(filename, name):
    path = Path(__file__).resolve().parent / filename
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, path


def save_json(path, obj):
    def cv(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return None if np.isnan(v) else float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, dict):
            return {str(k): cv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [cv(x) for x in v]
        return v
    path.write_text(json.dumps(cv(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def wrap_deg(a):
    return (np.asarray(a, dtype=np.float64) + 180.0) % 360.0 - 180.0


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    x, y = x[m], y[m]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    xr = pd.Series(x[m]).rank(method="average").to_numpy()
    yr = pd.Series(y[m]).rank(method="average").to_numpy()
    return safe_corr(xr, yr)


def vec_rmse(err):
    e = np.asarray(err, dtype=np.float64)
    m = np.isfinite(e).all(axis=-1)
    if not m.any():
        return np.nan
    return float(np.sqrt(np.mean(np.sum(e[m] ** 2, axis=-1))))


def moments(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) < 4 or np.std(x) == 0:
        return np.nan, np.nan
    z = (x - np.mean(x)) / np.std(x)
    return float(np.mean(z ** 3)), float(np.mean(z ** 4) - 3.0)


def lag1(err, times, segment=None):
    err = np.asarray(err, dtype=np.float64)
    times = np.asarray(times, dtype=np.int64)
    good = np.diff(times) == 60_000_000_000
    if segment is not None:
        segment = np.asarray(segment)
        good &= segment[1:] == segment[:-1]
    if not good.any():
        return np.nan
    r1 = safe_corr(err[:-1, 0][good], err[1:, 0][good])
    r2 = safe_corr(err[:-1, 1][good], err[1:, 1][good])
    return float(np.nanmean([r1, r2]))


def raw_feature(X, name, mean, std, last_n):
    i = FEATURES.index(name)
    z = np.asarray(X[:, -last_n:, i], dtype=np.float64)
    return z * float(std[i]) + float(mean[i])


def angle_from_pair(s, c):
    return np.degrees(np.arctan2(s, c))


def context_features(X, mean, std):
    u = raw_feature(X, "UWND_MEAN", mean, std, 10)
    v = raw_feature(X, "VWND_MEAN", mean, std, 10)
    sog = raw_feature(X, "SOG", mean, std, 10)
    cs = raw_feature(X, "COG_sin", mean, std, 11)
    cc = raw_feature(X, "COG_cos", mean, std, 11)
    hs = raw_feature(X, "HDG_sin", mean, std, 11)
    hc = raw_feature(X, "HDG_cos", mean, std, 11)
    ws = raw_feature(X, "WING_ANGLE_sin", mean, std, 11)
    wc = raw_feature(X, "WING_ANGLE_cos", mean, std, 11)

    cog = angle_from_pair(cs, cc)
    hdg = angle_from_pair(hs, hc)
    wing = angle_from_pair(ws, wc)

    cog_rate = np.mean(np.abs(wrap_deg(np.diff(cog, axis=1))), axis=1)
    hdg_rate = np.mean(np.abs(wrap_deg(np.diff(hdg, axis=1))), axis=1)
    wing_rate = np.mean(np.abs(wrap_deg(np.diff(wing, axis=1))), axis=1)

    um = np.mean(u, axis=1)
    vm = np.mean(v, axis=1)
    wind_var = np.sqrt(np.mean((u - um[:, None])**2 + (v - vm[:, None])**2, axis=1))

    return pd.DataFrame({
        "current_SOG_mps": sog[:, -1],
        "current_wind_speed_mps": np.sqrt(u[:, -1]**2 + v[:, -1]**2),
        "wind_variability_10min_mps": wind_var,
        "SOG_variability_10min_mps": np.std(sog, axis=1),
        "COG_turn_rate_10min_deg_per_min": cog_rate,
        "HDG_turn_rate_10min_deg_per_min": hdg_rate,
        "wing_change_rate_10min_deg_per_min": wing_rate,
    })


def make_thresholds(train_context):
    rows, out = [], {}
    for feature in REGIME_FEATURES:
        x = train_context[feature].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        q = np.quantile(x, [0.25, 0.50, 0.75])
        out[feature] = q
        rows.append({
            "feature": feature,
            "q25": q[0], "q50": q[1], "q75": q[2],
            "source": "SD1090_outer_train_only",
        })
    return out, pd.DataFrame(rows)


def load_dataset(dataset_id, core06, shared08c, dataset_dir, external_root):
    if dataset_id == "SD1090_validation":
        d = core06.load_npz_split(dataset_dir, "validation")
        d["segment_id"] = np.zeros(d["X"].shape[0], dtype=np.int64)
        return d
    return shared08c.load_external_npz(
        external_root / "datasets" / f"{dataset_id}.npz"
    )


def infer_detailed(core06, torch, model, X, ridge_z, device, batch_size, amp):
    n, H = X.shape[0], ridge_z.shape[1]
    pred = np.empty_like(ridge_z, dtype=np.float32)
    gate = np.empty((n, H, 2), dtype=np.float32)
    delta = np.empty_like(gate)
    corr = np.empty_like(gate)

    model.eval()
    with torch.no_grad():
        for a in range(0, n, batch_size):
            b = min(n, a + batch_size)
            xb = torch.from_numpy(np.asarray(X[a:b], dtype=np.float32)).to(device)
            rb = torch.from_numpy(np.asarray(ridge_z[a:b], dtype=np.float32)).to(device)
            with core06.AutocastContext(torch, amp, device.type):
                combined, delta_full, gate_full, corr_full, _ = model(xb, rb)
            pred[a:b] = combined.detach().float().cpu().numpy()
            delta[a:b] = delta_full[:, :, 2:4].detach().float().cpu().numpy()
            gate[a:b] = gate_full[:, :, 2:4].detach().float().cpu().numpy()
            corr[a:b] = corr_full[:, :, 2:4].detach().float().cpu().numpy()
    return pred, gate, delta, corr


def ridge_diagnostics(dataset_id, y, ridge, persistence, times, segment, context):
    rows, corr_rows = [], []
    for j, h in enumerate(HORIZONS):
        for branch, sl in [("wind", slice(0, 2)), ("vessel", slice(2, 4))]:
            e = ridge[:, j, sl].astype(float) - y[:, j, sl].astype(float)
            ep = persistence[:, j, sl].astype(float) - y[:, j, sl].astype(float)
            r = vec_rmse(e)
            rp = vec_rmse(ep)
            skew, kurt = moments(e)
            mag = np.sqrt(np.sum(e**2, axis=1))
            rows.append({
                "dataset_id": dataset_id,
                "branch": branch,
                "horizon_min": h,
                "ridge_vector_RMSE_mps": r,
                "persistence_vector_RMSE_mps": rp,
                "ridge_skill_vs_persistence": 1.0 - r/rp if rp > 0 else np.nan,
                "lag1_autocorrelation": lag1(e, times[:, j], segment),
                "pooled_component_skewness": skew,
                "pooled_component_excess_kurtosis": kurt,
            })
            for f in REGIME_FEATURES:
                corr_rows.append({
                    "dataset_id": dataset_id,
                    "branch": branch,
                    "horizon_min": h,
                    "context_feature": f,
                    "spearman_abs_error_vs_context": spearman(
                        mag, context[f].to_numpy(dtype=float)
                    ),
                })
    return pd.DataFrame(rows), pd.DataFrame(corr_rows)


def per_seed_analysis(dataset_id, model_name, seed, y, ridge_raw, pred_raw,
                      gate, delta_z, corr_z, target_std, context, thresholds):
    gate_rows, align_rows, state_rows, regime_rows = [], [], [], []

    corr_raw = corr_z.astype(float) * target_std[None, None, 2:4]
    delta_raw = delta_z.astype(float) * target_std[None, None, 2:4]
    needed = y[:, :, 2:4].astype(float) - ridge_raw[:, :, 2:4].astype(float)

    for j, h in enumerate(HORIZONS):
        g = gate[:, j, :].astype(float)
        gm = np.mean(g, axis=1)
        c = corr_raw[:, j, :]
        d = delta_raw[:, j, :]
        n = needed[:, j, :]
        cmag = np.sqrt(np.sum(c**2, axis=1))
        dmag = np.sqrt(np.sum(d**2, axis=1))
        nmag = np.sqrt(np.sum(n**2, axis=1))
        denom = cmag * nmag
        valid = denom > 1e-10
        cos = np.full(len(cmag), np.nan)
        cos[valid] = np.sum(c[valid] * n[valid], axis=1) / denom[valid]

        re = ridge_raw[:, j, 2:4].astype(float) - y[:, j, 2:4].astype(float)
        me = pred_raw[:, j, 2:4].astype(float) - y[:, j, 2:4].astype(float)
        ridge_sse = np.sum(re**2)
        model_sse = np.sum(me**2)

        gate_rows.append({
            "dataset_id": dataset_id, "model": model_name, "seed": seed,
            "horizon_min": h,
            "gate_mean": np.mean(g), "gate_std": np.std(g),
            "gate_q10": np.quantile(g, 0.10), "gate_q50": np.quantile(g, 0.50),
            "gate_q90": np.quantile(g, 0.90),
            "delta_vector_RMS_mps": np.sqrt(np.mean(dmag**2)),
            "correction_vector_RMS_mps": np.sqrt(np.mean(cmag**2)),
            "spearman_gate_vs_correction_magnitude": spearman(gm, cmag),
        })
        align_rows.append({
            "dataset_id": dataset_id, "model": model_name, "seed": seed,
            "horizon_min": h,
            "mean_correction_cosine_alignment": np.nanmean(cos),
            "median_correction_cosine_alignment": np.nanmedian(cos),
            "positive_alignment_fraction": np.nanmean(cos > 0),
            "correction_to_needed_RMS_ratio":
                np.sqrt(np.mean(cmag**2)) / max(np.sqrt(np.mean(nmag**2)), 1e-12),
            "vessel_SSE_reduction_fraction_vs_Ridge":
                1.0 - model_sse/ridge_sse if ridge_sse > 0 else np.nan,
        })

        for f in REGIME_FEATURES:
            x = context[f].to_numpy(dtype=float)
            state_rows.append({
                "dataset_id": dataset_id, "model": model_name, "seed": seed,
                "horizon_min": h, "context_feature": f,
                "spearman_gate_vs_context": spearman(gm, x),
                "spearman_correction_magnitude_vs_context": spearman(cmag, x),
            })

            bins = np.digitize(x, thresholds[f], right=True)
            for bi, label in enumerate(REGIME_LABELS):
                m = bins == bi
                if not m.any():
                    continue
                tw = y[m, j, 0:2].astype(float)
                tv = y[m, j, 2:4].astype(float)
                rw = ridge_raw[m, j, 0:2].astype(float)
                rv = ridge_raw[m, j, 2:4].astype(float)
                mw = pred_raw[m, j, 0:2].astype(float)
                mv = pred_raw[m, j, 2:4].astype(float)
                true_aw = tw - tv
                ridge_aw = rw - rv
                model_aw = mw - mv
                rvr = vec_rmse(rv - tv)
                mvr = vec_rmse(mv - tv)
                rar = vec_rmse(ridge_aw - true_aw)
                mar = vec_rmse(model_aw - true_aw)
                regime_rows.append({
                    "dataset_id": dataset_id, "model": model_name, "seed": seed,
                    "regime_feature": f, "regime_bin": label,
                    "regime_bin_index": bi, "horizon_min": h,
                    "n_samples": int(m.sum()),
                    "context_mean": np.mean(x[m]),
                    "mean_gate": np.mean(g[m]),
                    "correction_RMS_mps": np.sqrt(np.mean(np.sum(c[m]**2, axis=1))),
                    "ridge_vessel_RMSE_mps": rvr,
                    "model_vessel_RMSE_mps": mvr,
                    "vessel_improvement_vs_Ridge_percent":
                        100*(rvr-mvr)/rvr if rvr > 0 else np.nan,
                    "ridge_AW_RMSE_mps": rar,
                    "model_AW_RMSE_mps": mar,
                    "AW_improvement_vs_Ridge_percent":
                        100*(rar-mar)/rar if rar > 0 else np.nan,
                })

    return (
        pd.DataFrame(gate_rows),
        pd.DataFrame(align_rows),
        pd.DataFrame(state_rows),
        pd.DataFrame(regime_rows),
    )


def aggregate(df, groups, metrics):
    rows = []
    for keys, g in df.groupby(groups, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        r = {k: v for k, v in zip(groups, keys)}
        r["n_seed_rows"] = len(g)
        for m in metrics:
            x = g[m].to_numpy(dtype=float)
            r["mean_" + m] = np.nanmean(x)
            r["std_" + m] = np.nanstd(x, ddof=1) if np.isfinite(x).sum() > 1 else 0.0
        rows.append(r)
    return pd.DataFrame(rows)


def t95(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = np.mean(x)
    if len(x) == 1:
        return mean, 0.0, np.nan, np.nan
    std = np.std(x, ddof=1)
    try:
        from scipy.stats import t
        half = t.ppf(0.975, len(x)-1) * std / math.sqrt(len(x))
        return mean, std, mean-half, mean+half
    except Exception:
        return mean, std, np.nan, np.nan


def physics_regime_effect(regime):
    a = regime[regime["model"] == MODEL_COMPACT].copy()
    b = regime[regime["model"] == MODEL_PHYSICS].copy()
    keys = ["dataset_id", "seed", "regime_feature", "regime_bin",
            "regime_bin_index", "horizon_min"]
    m = a.merge(b, on=keys, suffixes=("_compact", "_physics"),
                validate="one_to_one")
    m["physics_AW_reduction_vs_compact_mps"] = (
        m["model_AW_RMSE_mps_compact"] - m["model_AW_RMSE_mps_physics"]
    )
    m["physics_AW_reduction_vs_compact_percent"] = (
        100*m["physics_AW_reduction_vs_compact_mps"]
        / m["model_AW_RMSE_mps_compact"]
    )

    rows = []
    gcols = ["dataset_id", "regime_feature", "regime_bin",
             "regime_bin_index", "horizon_min"]
    for keys_v, g in m.groupby(gcols, sort=False):
        mean, std, lo, hi = t95(g["physics_AW_reduction_vs_compact_mps"])
        r = {k: v for k, v in zip(gcols, keys_v)}
        r.update({
            "n_paired_seeds": len(g),
            "mean_physics_AW_reduction_vs_compact_mps": mean,
            "std_physics_AW_reduction_mps": std,
            "paired_95pct_CI_low_mps": lo,
            "paired_95pct_CI_high_mps": hi,
            "mean_physics_AW_reduction_vs_compact_percent":
                g["physics_AW_reduction_vs_compact_percent"].mean(),
            "physics_win_count": int(
                (g["physics_AW_reduction_vs_compact_mps"] > 0).sum()
            ),
        })
        rows.append(r)
    return m, pd.DataFrame(rows)


def load_style():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    p = Path(__file__).resolve().parent / "sci_plot_style.py"
    if not p.exists():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location("sci_plot_style_10a", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        selected = mod.apply_sci_style(base_font_size=8.5)
    except TypeError:
        try:
            selected = mod.apply_sci_style(font_size=8.5)
        except TypeError:
            selected = mod.apply_sci_style()
    log(f"[PLOT] SCI style applied; selected_font={selected}")
    return mod, plt


def make_plots(outdir, ridge_diag, gate_summary, align_summary,
               regime_summary, physics_effect):
    style, plt = load_style()
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    palette = list(getattr(style, "PALETTE", {}).values()) or [f"C{i}" for i in range(8)]

    def finish(ax):
        if hasattr(style, "clean_axis"):
            style.clean_axis(ax, grid=True)
        elif hasattr(style, "finish_axes"):
            style.finish_axes(ax, grid=True)

    def save(fig, name):
        base = figdir / name
        if hasattr(style, "save_figure"):
            style.save_figure(fig, base)
        else:
            fig.savefig(base.with_suffix(".png"), dpi=600)
            fig.savefig(base.with_suffix(".pdf"))
            plt.close(fig)

    # Ridge branch skill
    d = ridge_diag.groupby(["dataset_id", "branch"], as_index=False)[
        "ridge_skill_vs_persistence"].mean()
    ds = [x for x in DEFAULT_DATASETS if x in set(d["dataset_id"])]
    x = np.arange(len(ds)); w = 0.36
    wind = [d[(d.dataset_id==z)&(d.branch=="wind")].iloc[0]["ridge_skill_vs_persistence"] for z in ds]
    vessel = [d[(d.dataset_id==z)&(d.branch=="vessel")].iloc[0]["ridge_skill_vs_persistence"] for z in ds]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.bar(x-w/2, wind, width=w, label="Wind branch", color=palette[0])
    ax.bar(x+w/2, vessel, width=w, label="Vessel branch", color=palette[1 % len(palette)])
    ax.axhline(0, linewidth=0.7)
    ax.set_ylabel("Ridge skill vs persistence")
    ax.set_xticks(x, [DATASET_LABELS.get(z,z) for z in ds], rotation=15, ha="right")
    ax.legend(loc="best", ncol=2)
    finish(ax); save(fig, "01_ridge_branch_skill")

    # Gate by horizon
    d = gate_summary[gate_summary["model"] == MODEL_PHYSICS]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for i, z in enumerate(ds):
        g = d[d.dataset_id==z].sort_values("horizon_min")
        if len(g):
            ax.errorbar(g.horizon_min, g.mean_gate_mean, yerr=g.std_gate_mean,
                        marker="o", capsize=2, label=DATASET_LABELS.get(z,z),
                        color=palette[i % len(palette)])
    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel("Mean vessel-residual gate")
    ax.set_xticks(HORIZONS); ax.legend(loc="best", fontsize=7)
    finish(ax); save(fig, "02_physicscompact_gate_by_horizon")

    # Alignment by horizon
    d = align_summary[align_summary["model"] == MODEL_PHYSICS]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for i, z in enumerate(ds):
        g = d[d.dataset_id==z].sort_values("horizon_min")
        if len(g):
            ax.errorbar(
                g.horizon_min,
                g.mean_mean_correction_cosine_alignment,
                yerr=g.std_mean_correction_cosine_alignment,
                marker="o", capsize=2, label=DATASET_LABELS.get(z,z),
                color=palette[i % len(palette)]
            )
    ax.axhline(0, linewidth=0.7)
    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel("Correction-residual cosine")
    ax.set_xticks(HORIZONS); ax.legend(loc="best", fontsize=7)
    finish(ax); save(fig, "03_correction_alignment_by_horizon")

    feature = "COG_turn_rate_10min_deg_per_min"
    for z in ds:
        g = regime_summary[
            (regime_summary.dataset_id==z)
            & (regime_summary.model==MODEL_PHYSICS)
            & (regime_summary.regime_feature==feature)
        ]
        if len(g):
            q = g.groupby(["regime_bin","regime_bin_index"], as_index=False)[
                "mean_AW_improvement_vs_Ridge_percent"].mean().sort_values("regime_bin_index")
            fig, ax = plt.subplots(figsize=(3.6,2.8))
            ax.plot(q.regime_bin, q.mean_AW_improvement_vs_Ridge_percent,
                    marker="o", color=palette[0])
            ax.axhline(0, linewidth=0.7)
            ax.set_xlabel("COG-turn regime")
            ax.set_ylabel("AW improvement vs Ridge (%)")
            finish(ax); save(fig, "04_"+z+"_AW_gain_vs_COG_turn_regime")

        e = physics_effect[
            (physics_effect.dataset_id==z)
            & (physics_effect.regime_feature==feature)
        ]
        if len(e):
            q = e.groupby(["regime_bin","regime_bin_index"], as_index=False)[
                "mean_physics_AW_reduction_vs_compact_percent"].mean().sort_values("regime_bin_index")
            fig, ax = plt.subplots(figsize=(3.6,2.8))
            ax.plot(q.regime_bin, q.mean_physics_AW_reduction_vs_compact_percent,
                    marker="o", color=palette[1 % len(palette)])
            ax.axhline(0, linewidth=0.7)
            ax.set_xlabel("COG-turn regime")
            ax.set_ylabel("Physics refinement vs compact (%)")
            finish(ax); save(fig, "05_"+z+"_physics_refinement_vs_COG_turn_regime")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    ap.add_argument("--compact-dir", default=DEFAULT_COMPACT_DIR)
    ap.add_argument("--external-root", default=DEFAULT_EXTERNAL_ROOT)
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--ridge-chunk-size", type=int, default=16384)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    compact_dir = Path(args.compact_dir)
    external_root = Path(args.external_root)
    outdir = Path(args.output_dir) if args.output_dir else dataset_dir / "10A_mechanism_analysis_v0_1"
    outdir.mkdir(parents=True, exist_ok=True)
    dataset_ids = [x.strip() for x in args.datasets.split(",") if x.strip()]

    shared08c, p08c = load_module(SHARED_08C, "shared08c_10a")
    shared07c, p07c = load_module(SHARED_07C, "shared07c_10a")
    shared07b = shared07c.load_shared_07b()
    shared07a = shared07b.load_shared_07a()
    core06 = shared07a.load_core()
    torch, nn, _, _ = core06.import_torch()

    frozen = shared08c.load_frozen_context(
        shared07c, shared07a, core06, compact_dir, dataset_dir
    )
    seeds = [int(x) for x in frozen["seed_schedule"]]
    device = torch.device("cpu") if args.force_cpu or not torch.cuda.is_available() else torch.device("cuda:0")
    amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        core06.configure_tf32(torch, True)
    CompactClass = shared07c.build_compact_model_class(torch, nn)

    log("="*108)
    log("10A MECHANISM ANALYSIS")
    log("="*108)
    log(f"datasets               : {dataset_ids}")
    log(f"frozen seeds           : {seeds}")
    log(f"device / AMP           : {device} / {amp}")
    log("regime threshold source: SD1090 OUTER TRAIN only")
    log("training/fine-tuning   : NONE")
    log("model/loss change      : NONE")
    log("")

    # Frozen regime thresholds from OUTER TRAIN only.
    train = core06.load_npz_split(dataset_dir, "train")
    train_context = context_features(
        train["X"], frozen["feature_mean"], frozen["feature_std"]
    )
    thresholds, threshold_df = make_thresholds(train_context)
    threshold_df.to_csv(outdir/"frozen_regime_thresholds_from_train.csv",
                        index=False, encoding="utf-8-sig")
    del train, train_context
    gc.collect()

    ridge_frames, ridge_corr_frames = [], []
    gate_frames, align_frames, state_frames, regime_frames = [], [], [], []

    for di, dataset_id in enumerate(dataset_ids, 1):
        log("#"*108)
        log(f"[DATASET {di}/{len(dataset_ids)}] {dataset_id}")
        log("#"*108)

        d = load_dataset(dataset_id, core06, shared08c, dataset_dir, external_root)
        X = d["X"].astype(np.float32, copy=False)
        y = d["y_raw"].astype(np.float32, copy=False)
        times = d["target_time_ns"]
        segment = d.get("segment_id", np.zeros(X.shape[0], dtype=np.int64))
        context = context_features(X, frozen["feature_mean"], frozen["feature_std"])

        ridge_z = shared07a.ridge_predict_z(
            frozen["ridge_model"], X, len(HORIZONS), len(TARGETS), args.ridge_chunk_size
        )
        ridge_raw = shared07a.inverse_target(
            ridge_z, frozen["target_mean"], frozen["target_std"]
        )
        persistence = core06.persistence_joint_prediction(
            X, FEATURES, frozen["feature_mean"], frozen["feature_std"], len(HORIZONS)
        )

        rd, rc = ridge_diagnostics(
            dataset_id, y, ridge_raw, persistence, times, segment, context
        )
        ridge_frames.append(rd); ridge_corr_frames.append(rc)

        mean_rd = rd.groupby("branch")[["ridge_vector_RMSE_mps",
                                        "ridge_skill_vs_persistence",
                                        "lag1_autocorrelation"]].mean()
        for branch in ["wind", "vessel"]:
            r = mean_rd.loc[branch]
            log(f"  Ridge {branch:<6s}: RMSE={r.ridge_vector_RMSE_mps:.4f} | "
                f"skill={r.ridge_skill_vs_persistence:.4f} | lag1={r.lag1_autocorrelation:.4f}")

        for model_name in [MODEL_COMPACT, MODEL_PHYSICS]:
            log(f"  [{model_name}]")
            for si, seed in enumerate(seeds, 1):
                ckpt = compact_dir/"models"/f"seed_{seed}"/MODEL_FILES[model_name]
                if not ckpt.exists():
                    raise FileNotFoundError(ckpt)

                model = shared08c.instantiate_compact_model(
                    CompactClass,
                    feature_count=len(FEATURES),
                    horizon_count=len(HORIZONS),
                    target_count=len(TARGETS),
                    gru_config=frozen["gru_config"],
                    device=device,
                )
                state = torch.load(ckpt, map_location=device, weights_only=False)
                model.load_state_dict(state["state_dict"], strict=True)

                pred_z, gate, delta_z, corr_z = infer_detailed(
                    core06, torch, model, X, ridge_z, device, args.batch_size, amp
                )

                fixed = [0,1,4,5]
                err = np.max(np.abs(pred_z[:,:,fixed]-ridge_z[:,:,fixed]))
                if err > 1e-7:
                    raise RuntimeError(f"Frozen-channel structural audit failed: {err}")

                pred_raw = shared07a.inverse_target(
                    pred_z, frozen["target_mean"], frozen["target_std"]
                )

                g,a,s,r = per_seed_analysis(
                    dataset_id, model_name, seed, y, ridge_raw, pred_raw,
                    gate, delta_z, corr_z, frozen["target_std"], context, thresholds
                )
                gate_frames.append(g); align_frames.append(a)
                state_frames.append(s); regime_frames.append(r)

                log(
                    f"    seed {si}/{len(seeds)} ({seed}) | "
                    f"gate={g.gate_mean.mean():.3f} | "
                    f"alignment={a.mean_correction_cosine_alignment.mean():.3f} | "
                    f"vessel SSE reduction={100*a.vessel_SSE_reduction_fraction_vs_Ridge.mean():.2f}%"
                )
                del model, state, pred_z, gate, delta_z, corr_z, pred_raw
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        del d, X, y, times, segment, context, ridge_z, ridge_raw, persistence
        gc.collect()

    ridge_diag = pd.concat(ridge_frames, ignore_index=True)
    ridge_corr = pd.concat(ridge_corr_frames, ignore_index=True)
    gate = pd.concat(gate_frames, ignore_index=True)
    align = pd.concat(align_frames, ignore_index=True)
    state_corr = pd.concat(state_frames, ignore_index=True)
    regime = pd.concat(regime_frames, ignore_index=True)

    gate_summary = aggregate(
        gate, ["dataset_id","model","horizon_min"],
        ["gate_mean","gate_std","gate_q10","gate_q50","gate_q90",
         "delta_vector_RMS_mps","correction_vector_RMS_mps",
         "spearman_gate_vs_correction_magnitude"]
    )
    align_summary = aggregate(
        align, ["dataset_id","model","horizon_min"],
        ["mean_correction_cosine_alignment","median_correction_cosine_alignment",
         "positive_alignment_fraction","correction_to_needed_RMS_ratio",
         "vessel_SSE_reduction_fraction_vs_Ridge"]
    )
    state_summary = aggregate(
        state_corr, ["dataset_id","model","horizon_min","context_feature"],
        ["spearman_gate_vs_context","spearman_correction_magnitude_vs_context"]
    )
    regime_summary = aggregate(
        regime,
        ["dataset_id","model","regime_feature","regime_bin","regime_bin_index","horizon_min"],
        ["context_mean","mean_gate","correction_RMS_mps",
         "ridge_vessel_RMSE_mps","model_vessel_RMSE_mps",
         "vessel_improvement_vs_Ridge_percent",
         "ridge_AW_RMSE_mps","model_AW_RMSE_mps",
         "AW_improvement_vs_Ridge_percent"]
    )
    physics_seed, physics_summary = physics_regime_effect(regime)

    files = {
        "ridge_branch_residual_diagnostics.csv": ridge_diag,
        "ridge_residual_state_correlations.csv": ridge_corr,
        "compact_gate_correction_summary.csv": gate,
        "compact_gate_correction_seed_summary.csv": gate_summary,
        "correction_alignment.csv": align,
        "correction_alignment_seed_summary.csv": align_summary,
        "gate_state_correlations.csv": state_corr,
        "gate_state_correlation_seed_summary.csv": state_summary,
        "regime_bin_metrics.csv": regime,
        "regime_bin_seed_summary.csv": regime_summary,
        "physics_vs_compact_regime_seed_effects.csv": physics_seed,
        "physics_vs_compact_regime_effect_summary.csv": physics_summary,
    }
    for name, df in files.items():
        df.to_csv(outdir/name, index=False, encoding="utf-8-sig")

    if not args.no_plots:
        make_plots(outdir, ridge_diag, gate_summary, align_summary,
                   regime_summary, physics_summary)

    # Key terminal findings.
    ridge_mean = ridge_diag.groupby(["dataset_id","branch"], as_index=False)[
        ["ridge_vector_RMSE_mps","ridge_skill_vs_persistence",
         "lag1_autocorrelation","pooled_component_excess_kurtosis"]
    ].mean()

    key_rows = []
    for dataset_id in dataset_ids:
        phys_g = gate_summary[
            (gate_summary.dataset_id==dataset_id)&(gate_summary.model==MODEL_PHYSICS)
        ]
        phys_a = align_summary[
            (align_summary.dataset_id==dataset_id)&(align_summary.model==MODEL_PHYSICS)
        ]
        if len(phys_g) and len(phys_a):
            key_rows.append({
                "dataset_id": dataset_id,
                "mean_PhysicsCompact_gate": phys_g.mean_gate_mean.mean(),
                "mean_PhysicsCompact_correction_RMS_mps":
                    phys_g.mean_correction_vector_RMS_mps.mean(),
                "mean_correction_cosine_alignment":
                    phys_a.mean_mean_correction_cosine_alignment.mean(),
                "mean_vessel_SSE_reduction_fraction_vs_Ridge":
                    phys_a.mean_vessel_SSE_reduction_fraction_vs_Ridge.mean(),
            })
    key_df = pd.DataFrame(key_rows)
    key_df.to_csv(outdir/"mechanism_key_findings.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "version": VERSION,
        "datasets": dataset_ids,
        "seed_schedule": seeds,
        "regime_threshold_source": "SD1090_outer_train_only",
        "regime_features": REGIME_FEATURES,
        "policy": {
            "training": False,
            "fine_tuning": False,
            "external_scaler_fit": False,
            "seed_selection": False,
            "architecture_change": False,
            "loss_change": False,
            "all_frozen_seeds_used": True,
            "gate_is_not_uncertainty": True,
            "correlations_are_descriptive": True,
        },
        "shared_scripts": {"08C": str(p08c), "07C": str(p07c)},
        "software": {
            "python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "device": str(device),
        },
    }
    save_json(outdir/"mechanism_manifest.json", manifest)

    with (outdir/"mechanism_report.txt").open("w", encoding="utf-8") as f:
        f.write("10A Mechanism Analysis Report\n"+"="*100+"\n\n")
        f.write("POST-HOC DESCRIPTIVE ANALYSIS ONLY.\n")
        f.write("No training/fine-tuning/seed selection/model change/loss change.\n")
        f.write("Regime thresholds frozen from SD1090 OUTER TRAIN only.\n")
        f.write("Gate is amplitude modulation, not confidence/uncertainty.\n")
        f.write("Correlations are descriptive, not causal.\n\n")
        f.write("Ridge branch diagnostics (mean across horizons)\n"+"-"*100+"\n")
        f.write(ridge_mean.to_string(index=False))
        f.write("\n\nPhysics-Compact key findings\n"+"-"*100+"\n")
        f.write(key_df.to_string(index=False))
        f.write("\n")

    log("")
    log("="*108)
    log("10A MECHANISM ANALYSIS RESULTS")
    log("="*108)
    for dataset_id in dataset_ids:
        log(dataset_id + ":")
        for branch in ["wind","vessel"]:
            q = ridge_mean[
                (ridge_mean.dataset_id==dataset_id)&(ridge_mean.branch==branch)
            ]
            if len(q):
                r = q.iloc[0]
                log(f"  Ridge {branch:<6s}: RMSE={r.ridge_vector_RMSE_mps:.4f} | "
                    f"skill={r.ridge_skill_vs_persistence:.4f} | "
                    f"lag1={r.lag1_autocorrelation:.4f} | "
                    f"kurt={r.pooled_component_excess_kurtosis:.3f}")
        q = key_df[key_df.dataset_id==dataset_id]
        if len(q):
            r = q.iloc[0]
            log(f"  PhysicsCompact: gate={r.mean_PhysicsCompact_gate:.3f} | "
                f"correction={r.mean_PhysicsCompact_correction_RMS_mps:.4f} m/s | "
                f"alignment={r.mean_correction_cosine_alignment:.3f} | "
                f"vessel SSE reduction={100*r.mean_vessel_SSE_reduction_fraction_vs_Ridge:.2f}%")

    d = state_summary[state_summary.model==MODEL_PHYSICS].copy()
    d["abs_rho"] = np.abs(d["mean_spearman_gate_vs_context"])
    log("")
    log("Strongest descriptive gate-state associations:")
    for r in d.sort_values("abs_rho", ascending=False).head(10).itertuples():
        log(f"  {r.dataset_id} | h={r.horizon_min} min | "
            f"{r.context_feature} | rho={r.mean_spearman_gate_vs_context:.3f}")

    log("")
    log("[POLICY] No model was changed or retrained.")
    log("[POLICY] External data were not used to define regime thresholds.")
    log("[POLICY] Gate/state correlations are descriptive, not causal.")
    log(f"[DONE] 10A outputs: {outdir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n[FATAL ERROR]", flush=True)
        traceback.print_exc()
        print("\nPlease send the complete traceback and terminal output.", flush=True)
        sys.exit(1)
