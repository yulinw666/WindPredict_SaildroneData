# -*- coding: utf-8 -*-
"""
11C2A_common_fold_sanity_benchmark.py

Stage 11C2A
============
Common-fold sanity benchmark for the frozen BP-STGNN-F configuration.

Purpose
-------
Calibrate the absolute difficulty of the EXACT SAME three rolling-origin folds
used in Stage 11C2, using two simple references:

    1) Persistence
    2) Wind-only Ridge regression

This stage does NOT:
    - change the selected BP-STGNN configuration B01;
    - retune BP-STGNN;
    - read SD1090 outer validation;
    - read SD1090 outer test;
    - read any SD1033 external data;
    - modify Physics-Compact-Vessel-Residual v1.0.

The goal is diagnostic only:

    If BP-STGNN B01 is materially better than / competitive with Persistence
    and Ridge on the SAME folds and SAME accepted samples, the implementation
    is plausibly learning useful structure.

    If simple baselines are dramatically better, Stage 11C3 should be delayed
    and the BP-STGNN task adaptation should be audited before final refit.

Data
----
Reads ONLY:
    11C0_BP_STGNN_comparison_datasets_v0_1/
        track_F_common_task/
            SD1090_train.npz

and, if available:
    11C2_BP_STGNN_train_only_rolling_cv_v0_1/
        rolling_folds.csv
        fold_results.csv
        selected_config.json

The script independently reconstructs the three folds from timestamps and
cross-checks them against rolling_folds.csv when present.

Track-F schema
--------------
X_raw: [N, 60, 10]
y_wind_raw: [N, 5, 2]
horizons: [1,2,3,5,10] min
features:
    U,V,SOG,COG,T,H,P,dT,dP,dH

Benchmark 1: Persistence
------------------------
Future U/V at every horizon equals the last observed U/V in the 60-min context.

Benchmark 2: Ridge
------------------
A direct multi-output Ridge regression predicts all 10 targets:
    [U1,V1,U2,V2,U3,V3,U5,V5,U10,V10]

Input representation:
    flattened full 60 x 10 history = 600 predictors

This intentionally gives Ridge access to the SAME BP-STGNN-F information.

Each fold:
    - uses ONLY that fold's effective training samples;
    - fits X scaler on fold train only;
    - fits y scaler on fold train only;
    - fits Ridge on standardized data;
    - evaluates on the exact fold validation samples.

Ridge alpha
-----------
This stage is a sanity benchmark, not a new HPO stage. Therefore alpha is NOT
selected from validation. It is fixed a priori at:

    alpha = 1.0

This is the same simple regularized-linear spirit already used throughout the
project and prevents model-shopping during this diagnostic.

Primary metric
--------------
For each horizon h:

    vector_RMSE(h) =
        sqrt(mean((Uhat-U)^2 + (Vhat-V)^2))

Primary fold score:

    mean over h=[1,2,3,5,10] of vector_RMSE(h)

BP-STGNN reference
------------------
The script reads the completed B01 rows from Stage 11C2 fold_results.csv and
reports them side-by-side. It does NOT recompute or alter B01.

Outputs
-------
11C2A_common_fold_sanity_benchmark_v0_1/
    fold_definition_audit.csv
    sanity_fold_results.csv
    sanity_summary.csv
    sanity_decision.json
    sanity_report.txt

Recommended run
---------------
conda activate WindPredict
python "D:\\project\\WindPredict_SaildroneData\\src\\11C2A_common_fold_sanity_benchmark.py"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import Ridge
except Exception as exc:
    raise RuntimeError(
        "scikit-learn is required. Activate the WindPredict environment."
    ) from exc


SCRIPT_VERSION = "0.1.0-common-fold-sanity"

DEFAULT_PROJECT_ROOT = Path(
    r"D:\project\WindPredict_SaildroneData"
)

HORIZONS_MIN = [1, 2, 3, 5, 10]
NODE_NAMES = ["U", "V", "SOG", "COG", "T", "H", "P", "dT", "dP", "dH"]

PURGE_SAMPLES = 10
RIDGE_ALPHA = 1.0
SELECTED_BP_CONFIG_ID = "B01"


def log(msg=""):
    print(msg, flush=True)


def save_json(path: Path, obj):
    def cv(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, dict):
            return {str(k): cv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [cv(x) for x in v]
        return v

    with path.open("w", encoding="utf-8") as f:
        json.dump(cv(obj), f, ensure_ascii=False, indent=2)


def ns_iso(v):
    return str(np.datetime64(int(v), "ns"))


def load_track_f_train(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        p = {k: data[k] for k in data.files}

    required = [
        "X_raw",
        "y_wind_raw",
        "context_end_time_ns",
        "target_time_ns",
        "feature_names",
        "horizons_min",
    ]
    for k in required:
        if k not in p:
            raise KeyError(f"{path}: missing {k}")

    X = np.asarray(p["X_raw"], dtype=np.float32)
    y = np.asarray(p["y_wind_raw"], dtype=np.float32)
    ce = np.asarray(p["context_end_time_ns"], dtype=np.int64).reshape(-1)
    tt = np.asarray(p["target_time_ns"], dtype=np.int64)

    feature_names = [str(x) for x in np.asarray(p["feature_names"]).tolist()]
    horizons = [int(x) for x in np.asarray(p["horizons_min"]).tolist()]

    if X.shape[1:] != (60, 10):
        raise RuntimeError(f"Unexpected X shape: {X.shape}")
    if y.shape[1:] != (5, 2):
        raise RuntimeError(f"Unexpected y shape: {y.shape}")
    if feature_names != NODE_NAMES:
        raise RuntimeError(f"Feature mismatch: {feature_names}")
    if horizons != HORIZONS_MIN:
        raise RuntimeError(f"Horizon mismatch: {horizons}")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise RuntimeError("Non-finite values found in Track-F train data.")

    return {
        "X": X,
        "y": y,
        "context_end_time_ns": ce,
        "target_time_ns": tt,
    }


def build_rolling_folds(context_end_time_ns, target_time_ns):
    n = len(context_end_time_ns)
    val_size = n // 6
    initial_origin = n - 3 * val_size

    folds = []

    for k in range(3):
        origin = initial_origin + k * val_size
        train_end = origin - PURGE_SAMPLES
        val_start = origin
        val_end = n if k == 2 else origin + val_size

        train_idx = np.arange(0, train_end, dtype=np.int64)
        val_idx = np.arange(val_start, val_end, dtype=np.int64)

        max_train_target = int(np.max(target_time_ns[train_idx]))
        min_val_context = int(np.min(context_end_time_ns[val_idx]))

        firewall = max_train_target < min_val_context
        if not firewall:
            raise RuntimeError(
                f"Fold {k+1} timestamp firewall failed: "
                f"{ns_iso(max_train_target)} >= {ns_iso(min_val_context)}"
            )

        folds.append({
            "fold": k + 1,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "train_max_target_ns": max_train_target,
            "val_first_context_end_ns": min_val_context,
            "timestamp_firewall_pass": firewall,
        })

    return folds


def crosscheck_saved_folds(folds, path: Path):
    rows = []

    if not path.exists():
        for f in folds:
            rows.append({
                "fold": f["fold"],
                "saved_fold_file_present": False,
                "train_count_match": None,
                "val_count_match": None,
                "timestamp_match": None,
                "crosscheck_pass": True,
            })
        return rows

    saved = pd.read_csv(path)

    for f in folds:
        sub = saved.loc[saved["fold"].astype(int) == int(f["fold"])]
        if len(sub) != 1:
            raise RuntimeError(f"Saved rolling_folds.csv has invalid fold {f['fold']}")

        r = sub.iloc[0]

        train_match = int(r["train_samples"]) == int(f["train_samples"])
        val_match = int(r["validation_samples"]) == int(f["val_samples"])
        timestamp_match = (
            str(r["train_max_target"]) == ns_iso(f["train_max_target_ns"])
            and str(r["validation_first_context_end"])
            == ns_iso(f["val_first_context_end_ns"])
        )

        passed = train_match and val_match and timestamp_match
        if not passed:
            raise RuntimeError(
                f"Fold {f['fold']} does not match Stage 11C2 rolling_folds.csv."
            )

        rows.append({
            "fold": f["fold"],
            "saved_fold_file_present": True,
            "train_count_match": train_match,
            "val_count_match": val_match,
            "timestamp_match": timestamp_match,
            "crosscheck_pass": passed,
        })

    return rows


def fit_standardizer_2d(x):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def vector_metrics(y_true, y_pred):
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)

    per_h = []
    for hi, h in enumerate(HORIZONS_MIN):
        e = err[:, hi, :]

        u_rmse = float(np.sqrt(np.mean(e[:, 0] ** 2)))
        v_rmse = float(np.sqrt(np.mean(e[:, 1] ** 2)))
        vec_rmse = float(np.sqrt(np.mean(np.sum(e ** 2, axis=1))))

        per_h.append({
            "horizon_min": h,
            "U_RMSE_mps": u_rmse,
            "V_RMSE_mps": v_rmse,
            "vector_RMSE_mps": vec_rmse,
        })

    return {
        "mean_vector_RMSE_mps": float(
            np.mean([r["vector_RMSE_mps"] for r in per_h])
        ),
        "mean_U_RMSE_mps": float(
            np.mean([r["U_RMSE_mps"] for r in per_h])
        ),
        "mean_V_RMSE_mps": float(
            np.mean([r["V_RMSE_mps"] for r in per_h])
        ),
        "per_horizon": per_h,
    }


def persistence_predict(X_val):
    last_uv = np.asarray(X_val[:, -1, 0:2], dtype=np.float64)
    return np.repeat(last_uv[:, None, :], len(HORIZONS_MIN), axis=1)


def ridge_predict(X_train, y_train, X_val):
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]

    Xtr = np.asarray(X_train, dtype=np.float64).reshape(n_train, -1)
    Xva = np.asarray(X_val, dtype=np.float64).reshape(n_val, -1)
    ytr = np.asarray(y_train, dtype=np.float64).reshape(n_train, -1)

    x_mean, x_std = fit_standardizer_2d(Xtr)
    y_mean, y_std = fit_standardizer_2d(ytr)

    Xtr_z = (Xtr - x_mean[None, :]) / x_std[None, :]
    Xva_z = (Xva - x_mean[None, :]) / x_std[None, :]
    ytr_z = (ytr - y_mean[None, :]) / y_std[None, :]

    model = Ridge(
        alpha=RIDGE_ALPHA,
        fit_intercept=True,
    )
    model.fit(Xtr_z, ytr_z)

    pred_z = model.predict(Xva_z)
    pred = pred_z * y_std[None, :] + y_mean[None, :]
    return pred.reshape(n_val, len(HORIZONS_MIN), 2)


def metrics_to_row(method, fold_id, metrics):
    row = {
        "method": method,
        "fold": int(fold_id),
        "mean_vector_RMSE_mps": metrics["mean_vector_RMSE_mps"],
        "mean_U_RMSE_mps": metrics["mean_U_RMSE_mps"],
        "mean_V_RMSE_mps": metrics["mean_V_RMSE_mps"],
    }
    for hrow in metrics["per_horizon"]:
        h = hrow["horizon_min"]
        row[f"vector_RMSE_{h}min_mps"] = hrow["vector_RMSE_mps"]
        row[f"U_RMSE_{h}min_mps"] = hrow["U_RMSE_mps"]
        row[f"V_RMSE_{h}min_mps"] = hrow["V_RMSE_mps"]
    return row


def load_bp_fold_results(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Stage 11C2 fold_results.csv not found: {path}"
        )

    df = pd.read_csv(path)

    sub = df.loc[
        (df["config_id"].astype(str) == SELECTED_BP_CONFIG_ID)
        & (df["status"].astype(str) == "completed_finite")
    ].copy()

    if sorted(sub["fold"].astype(int).tolist()) != [1, 2, 3]:
        raise RuntimeError(
            f"Expected B01 completed rows for folds 1,2,3, got "
            f"{sub[['config_id','fold','status']].to_dict('records')}"
        )

    rows = []
    for r in sub.itertuples():
        row = {
            "method": "BP-STGNN-B01",
            "fold": int(r.fold),
            "mean_vector_RMSE_mps": float(r.val_mean_vector_RMSE_mps),
            "mean_U_RMSE_mps": float(r.val_mean_U_RMSE_mps),
            "mean_V_RMSE_mps": float(r.val_mean_V_RMSE_mps),
        }
        for h in HORIZONS_MIN:
            row[f"vector_RMSE_{h}min_mps"] = float(
                getattr(r, f"val_vector_RMSE_{h}min_mps")
            )
        rows.append(row)

    return rows


def summarize(fold_df):
    rows = []

    for method, sub in fold_df.groupby("method", sort=False):
        vals = sub["mean_vector_RMSE_mps"].to_numpy(float)

        row = {
            "method": method,
            "folds": int(len(sub)),
            "CV_mean_vector_RMSE_mps": float(np.mean(vals)),
            "CV_std_vector_RMSE_mps": float(np.std(vals, ddof=0)),
            "CV_mean_U_RMSE_mps": float(sub["mean_U_RMSE_mps"].mean()),
            "CV_mean_V_RMSE_mps": float(sub["mean_V_RMSE_mps"].mean()),
        }

        for h in HORIZONS_MIN:
            col = f"vector_RMSE_{h}min_mps"
            if col in sub.columns and sub[col].notna().all():
                row[f"CV_mean_vector_RMSE_{h}min_mps"] = float(sub[col].mean())

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["CV_mean_vector_RMSE_mps", "method"]
    ).reset_index(drop=True)

    best_simple = summary.loc[
        summary["method"].isin(["Persistence", "Ridge"])
    ].sort_values("CV_mean_vector_RMSE_mps").iloc[0]

    bp = summary.loc[
        summary["method"] == "BP-STGNN-B01"
    ].iloc[0]

    bp_vs_best_simple_pct = 100.0 * (
        best_simple["CV_mean_vector_RMSE_mps"]
        - bp["CV_mean_vector_RMSE_mps"]
    ) / best_simple["CV_mean_vector_RMSE_mps"]

    return summary, {
        "best_simple_baseline": str(best_simple["method"]),
        "best_simple_CV_vector_RMSE_mps": float(
            best_simple["CV_mean_vector_RMSE_mps"]
        ),
        "BP_STGNN_B01_CV_vector_RMSE_mps": float(
            bp["CV_mean_vector_RMSE_mps"]
        ),
        "BP_STGNN_improvement_vs_best_simple_percent": float(
            bp_vs_best_simple_pct
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        default=str(DEFAULT_PROJECT_ROOT),
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
    )
    parser.add_argument(
        "--stage11c2-dir",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )

    args = parser.parse_args()

    project_root = Path(args.project_root)

    frozen_root = (
        project_root
        / "data"
        / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
    )

    dataset_dir = (
        Path(args.dataset_dir)
        if args.dataset_dir
        else frozen_root / "11C0_BP_STGNN_comparison_datasets_v0_1"
    )

    stage11c2_dir = (
        Path(args.stage11c2_dir)
        if args.stage11c2_dir
        else frozen_root / "11C2_BP_STGNN_train_only_rolling_cv_v0_1"
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else frozen_root / "11C2A_common_fold_sanity_benchmark_v0_1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = (
        dataset_dir
        / "track_F_common_task"
        / "SD1090_train.npz"
    )

    data = load_track_f_train(train_path)

    folds = build_rolling_folds(
        data["context_end_time_ns"],
        data["target_time_ns"],
    )

    fold_audit = crosscheck_saved_folds(
        folds,
        stage11c2_dir / "rolling_folds.csv",
    )

    log("=" * 118)
    log("11C2A - COMMON-FOLD SANITY BENCHMARK")
    log("=" * 118)
    log(f"script version        : {SCRIPT_VERSION}")
    log(f"train dataset         : {train_path}")
    log(f"X / y                 : {data['X'].shape} / {data['y'].shape}")
    log(f"ridge alpha           : {RIDGE_ALPHA}")
    log(f"BP-STGNN reference    : {SELECTED_BP_CONFIG_ID}")
    log("outer validation      : NOT ACCESSED")
    log("outer test            : NOT ACCESSED")
    log("SD1033 external       : NOT ACCESSED")
    log("BP-STGNN retuning     : NONE")
    log("PhysicsCompact change : NONE")
    log("")

    fold_rows = []

    for fold in folds:
        k = fold["fold"]

        tr = fold["train_idx"]
        va = fold["val_idx"]

        Xtr = data["X"][tr]
        ytr = data["y"][tr]
        Xva = data["X"][va]
        yva = data["y"][va]

        log(
            f"[FOLD {k}] train={len(tr)} | val={len(va)} | "
            f"firewall={fold['timestamp_firewall_pass']}"
        )

        p_pred = persistence_predict(Xva)
        p_metrics = vector_metrics(yva, p_pred)
        fold_rows.append(
            metrics_to_row("Persistence", k, p_metrics)
        )

        log(
            f"  Persistence: mean vector RMSE="
            f"{p_metrics['mean_vector_RMSE_mps']:.4f} m/s"
        )

        r_pred = ridge_predict(Xtr, ytr, Xva)
        r_metrics = vector_metrics(yva, r_pred)
        fold_rows.append(
            metrics_to_row("Ridge", k, r_metrics)
        )

        log(
            f"  Ridge      : mean vector RMSE="
            f"{r_metrics['mean_vector_RMSE_mps']:.4f} m/s"
        )

    bp_rows = load_bp_fold_results(
        stage11c2_dir / "fold_results.csv"
    )
    fold_rows.extend(bp_rows)

    fold_df = pd.DataFrame(fold_rows)
    fold_df = fold_df.sort_values(
        ["fold", "method"]
    ).reset_index(drop=True)

    summary, decision_core = summarize(fold_df)

    p = summary.loc[summary["method"] == "Persistence"].iloc[0]
    r = summary.loc[summary["method"] == "Ridge"].iloc[0]
    b = summary.loc[summary["method"] == "BP-STGNN-B01"].iloc[0]

    bp_vs_persist = 100.0 * (
        p["CV_mean_vector_RMSE_mps"] - b["CV_mean_vector_RMSE_mps"]
    ) / p["CV_mean_vector_RMSE_mps"]

    bp_vs_ridge = 100.0 * (
        r["CV_mean_vector_RMSE_mps"] - b["CV_mean_vector_RMSE_mps"]
    ) / r["CV_mean_vector_RMSE_mps"]

    # Decision rule is intentionally diagnostic rather than model-selective.
    # Positive means BP-STGNN is better.
    if bp_vs_ridge >= 0.0:
        recommendation = "PROCEED_TO_11C3"
        interpretation = (
            "Frozen B01 is at least as good as the stronger simple linear "
            "reference on the exact same folds; proceed to five-seed final refit."
        )
    elif bp_vs_ridge > -10.0:
        recommendation = "PROCEED_WITH_CAUTION_TO_11C3"
        interpretation = (
            "Frozen B01 is slightly worse than Ridge (<10%) but remains in the "
            "same performance regime; retain B01 and proceed, documenting the result."
        )
    else:
        recommendation = "AUDIT_TASK_ADAPTATION_BEFORE_11C3"
        interpretation = (
            "Frozen B01 is more than 10% worse than same-sample Ridge. Do not "
            "retune B01. Audit the BP-STGNN task-adaptation mechanics (especially "
            "node-to-GRU aggregation / multi-horizon adaptation) before final refit."
        )

    decision = {
        "stage": "11C2A",
        "script_version": SCRIPT_VERSION,
        "selected_BP_STGNN_config_remains_frozen": SELECTED_BP_CONFIG_ID,
        "ridge_alpha_fixed_a_priori": RIDGE_ALPHA,
        **decision_core,
        "BP_STGNN_improvement_vs_Persistence_percent": float(bp_vs_persist),
        "BP_STGNN_improvement_vs_Ridge_percent": float(bp_vs_ridge),
        "recommendation": recommendation,
        "interpretation": interpretation,
        "data_firewall": {
            "SD1090_outer_train_accessed": True,
            "SD1090_outer_validation_accessed": False,
            "SD1090_outer_test_accessed": False,
            "SD1033_external_accessed": False,
        },
        "model_selection_policy": {
            "B01_changed": False,
            "BP_STGNN_retuned": False,
            "Ridge_alpha_tuned": False,
        },
    }

    pd.DataFrame(fold_audit).to_csv(
        output_dir / "fold_definition_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_df.to_csv(
        output_dir / "sanity_fold_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        output_dir / "sanity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        output_dir / "sanity_decision.json",
        decision,
    )

    with (output_dir / "sanity_report.txt").open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("11C2A Common-Fold Sanity Benchmark\n")
        f.write("=" * 118 + "\n\n")
        f.write("DATA FIREWALL\n")
        f.write("-" * 118 + "\n")
        f.write("SD1090 outer TRAIN accessed: YES\n")
        f.write("SD1090 outer VALIDATION accessed: NO\n")
        f.write("SD1090 outer TEST accessed: NO\n")
        f.write("SD1033 external accessed: NO\n")
        f.write("BP-STGNN B01 retuned: NO\n")
        f.write("Ridge alpha tuned: NO (fixed alpha=1.0)\n\n")
        f.write("FOLD RESULTS\n")
        f.write("-" * 118 + "\n")
        f.write(fold_df.to_string(index=False))
        f.write("\n\nSUMMARY\n")
        f.write("-" * 118 + "\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nDECISION\n")
        f.write("-" * 118 + "\n")
        f.write(json.dumps(decision, ensure_ascii=False, indent=2))
        f.write("\n")

    log("")
    log("=" * 118)
    log("11C2A COMMON-FOLD SANITY RESULTS")
    log("=" * 118)

    for row in summary.itertuples():
        log(
            f"{row.method}: CV vector RMSE="
            f"{row.CV_mean_vector_RMSE_mps:.4f} +/- "
            f"{row.CV_std_vector_RMSE_mps:.4f} m/s"
        )
        hs = []
        for h in HORIZONS_MIN:
            col = f"CV_mean_vector_RMSE_{h}min_mps"
            if hasattr(row, col):
                hs.append(
                    f"{h}min={getattr(row, col):.4f}"
                )
        if hs:
            log("  " + " | ".join(hs))

    log("")
    log(
        f"BP-STGNN vs Persistence: {bp_vs_persist:+.2f}% "
        "(positive = BP-STGNN better)"
    )
    log(
        f"BP-STGNN vs Ridge      : {bp_vs_ridge:+.2f}% "
        "(positive = BP-STGNN better)"
    )
    log(f"[DECISION] {recommendation}")
    log(f"  {interpretation}")
    log("")
    log("[POLICY] B01 remains frozen regardless of this sanity benchmark.")
    log("[POLICY] No outer validation/test or SD1033 external data were accessed.")
    log(f"[DONE] 11C2A outputs: {output_dir}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n[FATAL ERROR]", flush=True)
        traceback.print_exc()
        print("\nPlease send the complete traceback and terminal output.", flush=True)
        sys.exit(1)
