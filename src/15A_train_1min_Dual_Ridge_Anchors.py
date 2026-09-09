# -*- coding: utf-8 -*-
"""
15A_train_1min_Dual_Ridge_Anchors.py

STEP 1 of the staged 1-min apparent-wind forecasting method.

Anchor A: True-wind Ridge
    input  : [U,V,T,RH] x 60 min
    output : future [U,V] at [1,2,3,5,10] min

Anchor B: Vessel+HDG Ridge
    input  : all 11 joint-dataset features x 60 min
    output : future [VESSEL_EAST_MPS, VESSEL_NORTH_MPS, HDG_sin, HDG_cos]
             at [1,2,3,5,10] min

Each anchor:
    - selects its own Ridge alpha on VALIDATION only
    - final fit uses TRAIN only
    - generates purged blocked 5-fold OOF predictions on TRAIN

The OOF outputs are the leakage-safe residual targets for later stages.

Expected dataset:
    SD1090_TPOS2024_JointForecasting_v0_1
    X     : (N,60,11)
    y_raw : (N,5,6)

Target order:
    0 UWND_MEAN
    1 VWND_MEAN
    2 VESSEL_EAST_MPS
    3 VESSEL_NORTH_MPS
    4 HDG_sin
    5 HDG_cos
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ROOT = Path(r"D:\project\WindPredict_SaildroneData")
DEFAULT_DATASET_DIR = ROOT / "data" / "forecasting" / "SD1090_TPOS2024_JointForecasting_v0_1"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "forecasting" / "15A_1min_Dual_Ridge_Anchors_v0_1"

HORIZONS = [1, 2, 3, 5, 10]

FEATURE_NAMES = [
    "UWND_MEAN",
    "VWND_MEAN",
    "TEMP_AIR_MEAN",
    "RH_MEAN",
    "SOG",
    "COG_sin",
    "COG_cos",
    "HDG_sin",
    "HDG_cos",
    "WING_ANGLE_sin",
    "WING_ANGLE_cos",
]

TARGET_NAMES = [
    "UWND_MEAN",
    "VWND_MEAN",
    "VESSEL_EAST_MPS",
    "VESSEL_NORTH_MPS",
    "HDG_sin",
    "HDG_cos",
]

TRUEWIND_FEATURE_IDX = [0, 1, 2, 3]
ALL_FEATURE_IDX = list(range(11))
TRUEWIND_TARGET_IDX = [0, 1]
VESSEL_HDG_TARGET_IDX = [2, 3, 4, 5]

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

OOF_FOLDS = 5
OOF_PURGE = 10

RIDGE_SOLVER = "lsqr"
RIDGE_TOL = 1e-4
RIDGE_MAX_ITER = 5000


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
            return None if not np.isfinite(v) else float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, dict):
            return {str(k): cv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [cv(x) for x in v]
        return v

    with path.open("w", encoding="utf-8") as f:
        json.dump(cv(obj), f, ensure_ascii=False, indent=2, sort_keys=True)


def load_split(dataset_dir: Path, split: str):
    path = dataset_dir / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as z:
        if "X" not in z.files or "y_raw" not in z.files:
            raise RuntimeError(f"{path}: required keys X/y_raw missing. Keys={z.files}")

        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y_raw"], dtype=np.float32)

    if X.ndim != 3 or X.shape[1:] != (60, 11):
        raise RuntimeError(f"{split}: expected X=(N,60,11), got {X.shape}")
    if y.ndim != 3 or y.shape[1:] != (5, 6):
        raise RuntimeError(f"{split}: expected y_raw=(N,5,6), got {y.shape}")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise RuntimeError(f"{split}: nonfinite values.")

    return {"X": X, "y": y, "path": path}


def select_X(split, idx):
    return np.asarray(split["X"][:, :, idx], dtype=np.float32)


def select_y(split, idx):
    return np.asarray(split["y"][:, :, idx], dtype=np.float32)


def flatten_X(X):
    return np.asarray(X, dtype=np.float32).reshape(len(X), -1)


def flatten_y(y):
    return np.asarray(y, dtype=np.float32).reshape(len(y), -1)


def unflatten_y(y_flat, n_components):
    return np.asarray(y_flat, dtype=np.float32).reshape(len(y_flat), len(HORIZONS), n_components)


def fit_target_scaler(y):
    y64 = np.asarray(y, dtype=np.float64)
    mean = np.mean(y64, axis=0)
    std = np.std(y64, axis=0, ddof=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def scale_y(y, scaler):
    return (
        (np.asarray(y, dtype=np.float32) - scaler["mean"][None, :, :])
        / scaler["std"][None, :, :]
    ).astype(np.float32)


def inverse_y(yz, scaler):
    return (
        np.asarray(yz, dtype=np.float32) * scaler["std"][None, :, :]
        + scaler["mean"][None, :, :]
    ).astype(np.float32)


def build_ridge(alpha):
    return Ridge(
        alpha=float(alpha),
        solver=RIDGE_SOLVER,
        tol=RIDGE_TOL,
        max_iter=RIDGE_MAX_ITER,
    )


def fit_ridge(X, y, scaler, alpha):
    model = build_ridge(alpha)
    model.fit(flatten_X(X), flatten_y(scale_y(y, scaler)))
    return model


def predict_ridge(model, X, scaler, n_components):
    z = np.asarray(model.predict(flatten_X(X)), dtype=np.float32)
    z = unflatten_y(z, n_components)
    return inverse_y(z, scaler)


def standardized_mse(y_true, y_pred, scaler):
    t = scale_y(y_true, scaler)
    p = scale_y(y_pred, scaler)
    return float(np.mean((p - t) ** 2))


def parameter_count(model):
    return int(np.asarray(model.coef_).size + np.asarray(model.intercept_).size)


def select_alpha(name, Xtr, ytr, Xva, yva, scaler, n_components):
    rows = []
    best_alpha = None
    best_loss = np.inf

    log(f"[ALPHA SEARCH] {name}")
    for alpha in ALPHAS:
        model = fit_ridge(Xtr, ytr, scaler, alpha)
        pred = predict_ridge(model, Xva, scaler, n_components)
        loss = standardized_mse(yva, pred, scaler)

        rows.append(
            {
                "anchor": name,
                "alpha": float(alpha),
                "validation_standardized_MSE": loss,
            }
        )

        log(f"  alpha={alpha:8g} | validation stdMSE={loss:.8f}")

        if loss < best_loss:
            best_loss = loss
            best_alpha = float(alpha)

    log(f"  -> selected alpha={best_alpha:g}")
    return best_alpha, pd.DataFrame(rows)


def make_oof_folds(n):
    blocks = np.array_split(np.arange(n, dtype=np.int64), OOF_FOLDS)
    all_idx = np.arange(n, dtype=np.int64)
    folds = []

    for k, va_idx in enumerate(blocks):
        a = int(va_idx[0])
        b = int(va_idx[-1]) + 1
        pa = max(0, a - OOF_PURGE)
        pb = min(n, b + OOF_PURGE)

        mask = np.ones(n, dtype=bool)
        mask[pa:pb] = False
        tr_idx = all_idx[mask]

        folds.append(
            {
                "fold": k,
                "train_idx": tr_idx,
                "val_idx": va_idx,
                "validation_start_index": a,
                "validation_end_index_exclusive": b,
                "purge_start_index": pa,
                "purge_end_index_exclusive": pb,
            }
        )

    return folds


def oof_predict(X, y, scaler, alpha, n_components, folds):
    pred = np.full(y.shape, np.nan, dtype=np.float32)

    for fold in folds:
        tr = fold["train_idx"]
        va = fold["val_idx"]

        model = fit_ridge(X[tr], y[tr], scaler, alpha)
        pred[va] = predict_ridge(model, X[va], scaler, n_components)

    if not np.isfinite(pred).all():
        raise RuntimeError("OOF predictions incomplete/nonfinite.")

    return pred


def wind_direction_from_uv(uv):
    uv = np.asarray(uv, dtype=np.float64)
    return np.degrees(np.arctan2(-uv[..., 0], -uv[..., 1])) % 360.0


def circular_diff_deg(a, b):
    return ((np.asarray(a) - np.asarray(b) + 180.0) % 360.0) - 180.0


def truewind_metrics(y_true, y_pred, split):
    rows = []

    for j, h in enumerate(HORIZONS):
        t = np.asarray(y_true[:, j, :], dtype=np.float64)
        p = np.asarray(y_pred[:, j, :], dtype=np.float64)
        e = p - t

        ws_t = np.linalg.norm(t, axis=1)
        ws_p = np.linalg.norm(p, axis=1)
        wd_err = circular_diff_deg(
            wind_direction_from_uv(p),
            wind_direction_from_uv(t),
        )

        rows.append(
            {
                "split": split,
                "horizon_min": h,
                "U_RMSE_mps": float(np.sqrt(np.mean(e[:, 0] ** 2))),
                "V_RMSE_mps": float(np.sqrt(np.mean(e[:, 1] ** 2))),
                "vector_RMSE_mps": float(np.sqrt(np.mean(np.sum(e ** 2, axis=1)))),
                "WS_RMSE_mps": float(np.sqrt(np.mean((ws_p - ws_t) ** 2))),
                "WD_RMSE_deg": float(np.sqrt(np.mean(wd_err ** 2))),
            }
        )

    return pd.DataFrame(rows)


def normalize_heading_sincos(pred):
    out = np.asarray(pred, dtype=np.float32).copy()
    s = out[:, :, 2]
    c = out[:, :, 3]
    n = np.sqrt(s ** 2 + c ** 2)

    good = n > 1e-8
    s2 = np.zeros_like(s)
    c2 = np.ones_like(c)

    s2[good] = s[good] / n[good]
    c2[good] = c[good] / n[good]

    out[:, :, 2] = s2
    out[:, :, 3] = c2
    return out


def hdg_deg(s, c):
    return np.degrees(np.arctan2(s, c)) % 360.0


def vessel_hdg_metrics(y_true, y_pred, split):
    yp = normalize_heading_sincos(y_pred)
    rows = []

    for j, h in enumerate(HORIZONS):
        t = np.asarray(y_true[:, j, :], dtype=np.float64)
        p = np.asarray(yp[:, j, :], dtype=np.float64)

        ev = p[:, 0:2] - t[:, 0:2]

        sog_t = np.linalg.norm(t[:, 0:2], axis=1)
        sog_p = np.linalg.norm(p[:, 0:2], axis=1)

        h_t = hdg_deg(t[:, 2], t[:, 3])
        h_p = hdg_deg(p[:, 2], p[:, 3])
        h_err = circular_diff_deg(h_p, h_t)

        rows.append(
            {
                "split": split,
                "horizon_min": h,
                "Vessel_East_RMSE_mps": float(np.sqrt(np.mean(ev[:, 0] ** 2))),
                "Vessel_North_RMSE_mps": float(np.sqrt(np.mean(ev[:, 1] ** 2))),
                "Vessel_vector_RMSE_mps": float(np.sqrt(np.mean(np.sum(ev ** 2, axis=1)))),
                "SOG_RMSE_mps": float(np.sqrt(np.mean((sog_p - sog_t) ** 2))),
                "HDG_MAE_deg": float(np.mean(np.abs(h_err))),
                "HDG_RMSE_deg": float(np.sqrt(np.mean(h_err ** 2))),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 140)
    log("15A — STEP 1: DUAL RIDGE ANCHORS")
    log("=" * 140)
    log("TrueWind Ridge  : [U,V,T,RH] x 60")
    log("Vessel+HDG Ridge: all 11 features x 60")
    log("Horizons        : 1/2/3/5/10 min")

    log("")
    log("[STAGE 1/8] Loading train / validation / test.")
    train = load_split(dataset_dir, "train")
    val = load_split(dataset_dir, "validation")
    test = load_split(dataset_dir, "test")

    log(f"  train      X={train['X'].shape} y={train['y'].shape}")
    log(f"  validation X={val['X'].shape} y={val['y'].shape}")
    log(f"  test       X={test['X'].shape} y={test['y'].shape}")

    Xw_tr = select_X(train, TRUEWIND_FEATURE_IDX)
    Xw_va = select_X(val, TRUEWIND_FEATURE_IDX)
    Xw_te = select_X(test, TRUEWIND_FEATURE_IDX)

    Xvh_tr = select_X(train, ALL_FEATURE_IDX)
    Xvh_va = select_X(val, ALL_FEATURE_IDX)
    Xvh_te = select_X(test, ALL_FEATURE_IDX)

    yw_tr = select_y(train, TRUEWIND_TARGET_IDX)
    yw_va = select_y(val, TRUEWIND_TARGET_IDX)
    yw_te = select_y(test, TRUEWIND_TARGET_IDX)

    yvh_tr = select_y(train, VESSEL_HDG_TARGET_IDX)
    yvh_va = select_y(val, VESSEL_HDG_TARGET_IDX)
    yvh_te = select_y(test, VESSEL_HDG_TARGET_IDX)

    wind_scaler = fit_target_scaler(yw_tr)
    vh_scaler = fit_target_scaler(yvh_tr)

    log("")
    log("[STAGE 2/8] Separate validation alpha selection.")
    wind_alpha, wind_alpha_df = select_alpha(
        "TrueWind-Ridge",
        Xw_tr,
        yw_tr,
        Xw_va,
        yw_va,
        wind_scaler,
        2,
    )

    vh_alpha, vh_alpha_df = select_alpha(
        "VesselHDG-Ridge",
        Xvh_tr,
        yvh_tr,
        Xvh_va,
        yvh_va,
        vh_scaler,
        4,
    )

    wind_alpha_df.to_csv(
        output_dir / "15A_truewind_alpha_search.csv",
        index=False,
        encoding="utf-8-sig",
    )
    vh_alpha_df.to_csv(
        output_dir / "15A_vessel_hdg_alpha_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log("[STAGE 3/8] Final TRAIN-only Ridge fits.")
    wind_model = fit_ridge(Xw_tr, yw_tr, wind_scaler, wind_alpha)
    vh_model = fit_ridge(Xvh_tr, yvh_tr, vh_scaler, vh_alpha)

    wind_params = parameter_count(wind_model)
    vh_params = parameter_count(vh_model)

    log(f"  TrueWind Ridge params   = {wind_params:,}")
    log(f"  Vessel+HDG Ridge params = {vh_params:,}")

    log("")
    log("[STAGE 4/8] Train/validation/test anchor predictions.")
    pred_w_tr = predict_ridge(wind_model, Xw_tr, wind_scaler, 2)
    pred_w_va = predict_ridge(wind_model, Xw_va, wind_scaler, 2)
    pred_w_te = predict_ridge(wind_model, Xw_te, wind_scaler, 2)

    pred_vh_tr = predict_ridge(vh_model, Xvh_tr, vh_scaler, 4)
    pred_vh_va = predict_ridge(vh_model, Xvh_va, vh_scaler, 4)
    pred_vh_te = predict_ridge(vh_model, Xvh_te, vh_scaler, 4)

    log("")
    log("[STAGE 5/8] Shared purged blocked 5-fold OOF predictions.")
    folds = make_oof_folds(len(train["X"]))

    oof_w = oof_predict(
        Xw_tr,
        yw_tr,
        wind_scaler,
        wind_alpha,
        2,
        folds,
    )

    oof_vh = oof_predict(
        Xvh_tr,
        yvh_tr,
        vh_scaler,
        vh_alpha,
        4,
        folds,
    )

    fold_df = pd.DataFrame(
        [
            {
                "fold": f["fold"],
                "validation_start_index": f["validation_start_index"],
                "validation_end_index_exclusive": f["validation_end_index_exclusive"],
                "purge_start_index": f["purge_start_index"],
                "purge_end_index_exclusive": f["purge_end_index_exclusive"],
                "train_samples": len(f["train_idx"]),
                "validation_samples": len(f["val_idx"]),
            }
            for f in folds
        ]
    )
    fold_df.to_csv(
        output_dir / "15A_oof_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log("[STAGE 6/8] Metrics.")
    wind_metrics = pd.concat(
        [
            truewind_metrics(yw_tr, pred_w_tr, "train_in_sample"),
            truewind_metrics(yw_tr, oof_w, "train_OOF"),
            truewind_metrics(yw_va, pred_w_va, "validation"),
            truewind_metrics(yw_te, pred_w_te, "test"),
        ],
        ignore_index=True,
    )

    vh_metrics = pd.concat(
        [
            vessel_hdg_metrics(yvh_tr, pred_vh_tr, "train_in_sample"),
            vessel_hdg_metrics(yvh_tr, oof_vh, "train_OOF"),
            vessel_hdg_metrics(yvh_va, pred_vh_va, "validation"),
            vessel_hdg_metrics(yvh_te, pred_vh_te, "test"),
        ],
        ignore_index=True,
    )

    wind_metrics.to_csv(
        output_dir / "15A_truewind_metrics_per_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )
    vh_metrics.to_csv(
        output_dir / "15A_vessel_hdg_metrics_per_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log("[STAGE 7/8] Saving OOF anchors and frozen Ridge models.")

    np.savez_compressed(
        output_dir / "15A_train_oof_anchors.npz",
        true_wind_y=yw_tr.astype(np.float32),
        true_wind_ridge_oof=oof_w.astype(np.float32),
        true_wind_residual_oof=(yw_tr - oof_w).astype(np.float32),
        vessel_hdg_y=yvh_tr.astype(np.float32),
        vessel_hdg_ridge_oof=oof_vh.astype(np.float32),
        vessel_hdg_ridge_oof_unit_heading=normalize_heading_sincos(oof_vh).astype(np.float32),
        vessel_hdg_residual_oof=(yvh_tr - oof_vh).astype(np.float32),
        horizons_min=np.asarray(HORIZONS, dtype=np.int64),
    )

    np.savez_compressed(
        output_dir / "15A_validation_anchor_predictions.npz",
        true_wind_y=yw_va.astype(np.float32),
        true_wind_ridge=pred_w_va.astype(np.float32),
        true_wind_residual=(yw_va - pred_w_va).astype(np.float32),
        vessel_hdg_y=yvh_va.astype(np.float32),
        vessel_hdg_ridge=pred_vh_va.astype(np.float32),
        vessel_hdg_ridge_unit_heading=normalize_heading_sincos(pred_vh_va).astype(np.float32),
        vessel_hdg_residual=(yvh_va - pred_vh_va).astype(np.float32),
        horizons_min=np.asarray(HORIZONS, dtype=np.int64),
    )

    np.savez_compressed(
        output_dir / "15A_test_anchor_predictions.npz",
        true_wind_y=yw_te.astype(np.float32),
        true_wind_ridge=pred_w_te.astype(np.float32),
        true_wind_residual=(yw_te - pred_w_te).astype(np.float32),
        vessel_hdg_y=yvh_te.astype(np.float32),
        vessel_hdg_ridge=pred_vh_te.astype(np.float32),
        vessel_hdg_ridge_unit_heading=normalize_heading_sincos(pred_vh_te).astype(np.float32),
        vessel_hdg_residual=(yvh_te - pred_vh_te).astype(np.float32),
        horizons_min=np.asarray(HORIZONS, dtype=np.int64),
    )

    wind_model_path = output_dir / "15A_truewind_ridge.joblib"
    vh_model_path = output_dir / "15A_vessel_hdg_ridge.joblib"

    joblib.dump(
        {
            "model": wind_model,
            "target_scaler": wind_scaler,
            "alpha": wind_alpha,
            "feature_names": [FEATURE_NAMES[i] for i in TRUEWIND_FEATURE_IDX],
            "feature_indices": TRUEWIND_FEATURE_IDX,
            "target_names": [TARGET_NAMES[i] for i in TRUEWIND_TARGET_IDX],
            "target_indices": TRUEWIND_TARGET_IDX,
            "horizons_min": HORIZONS,
            "input_shape": (60, 4),
        },
        wind_model_path,
    )

    joblib.dump(
        {
            "model": vh_model,
            "target_scaler": vh_scaler,
            "alpha": vh_alpha,
            "feature_names": FEATURE_NAMES,
            "feature_indices": ALL_FEATURE_IDX,
            "target_names": [TARGET_NAMES[i] for i in VESSEL_HDG_TARGET_IDX],
            "target_indices": VESSEL_HDG_TARGET_IDX,
            "horizons_min": HORIZONS,
            "input_shape": (60, 11),
            "heading_representation": "sin/cos; normalize pair before angle use",
        },
        vh_model_path,
    )

    log("")
    log("[STAGE 8/8] Final report.")

    test_w = wind_metrics.loc[wind_metrics["split"] == "test"].reset_index(drop=True)
    test_vh = vh_metrics.loc[vh_metrics["split"] == "test"].reset_index(drop=True)
    oof_wm = wind_metrics.loc[wind_metrics["split"] == "train_OOF"].reset_index(drop=True)
    oof_vhm = vh_metrics.loc[vh_metrics["split"] == "train_OOF"].reset_index(drop=True)

    manifest = {
        "stage": "15A",
        "script_version": "0.2.0",
        "dataset": str(dataset_dir),
        "horizons_min": HORIZONS,
        "truewind_ridge": {
            "inputs": [FEATURE_NAMES[i] for i in TRUEWIND_FEATURE_IDX],
            "input_shape": [60, 4],
            "targets": [TARGET_NAMES[i] for i in TRUEWIND_TARGET_IDX],
            "alpha": wind_alpha,
            "params": wind_params,
        },
        "vessel_hdg_ridge": {
            "inputs": FEATURE_NAMES,
            "input_shape": [60, 11],
            "targets": [TARGET_NAMES[i] for i in VESSEL_HDG_TARGET_IDX],
            "alpha": vh_alpha,
            "params": vh_params,
        },
        "oof": {
            "method": "purged blocked 5-fold",
            "purge_each_side_samples": OOF_PURGE,
            "purpose": "leakage-safe residual targets for Stage15B/15C",
        },
        "next": {
            "15B": (
                "freeze both Ridge anchors; train only true-wind residual branch "
                "from true_wind_residual_oof; optimize/freeze alpha"
            ),
            "15C": (
                "freeze Stage15B true-wind predictor; train vessel+HDG residual branch; "
                "apparent wind uses Stage15B true wind instead of raw Ridge true wind"
            ),
        },
    }

    save_json(output_dir / "15A_manifest.json", manifest)

    with (output_dir / "15A_REPORT.txt").open("w", encoding="utf-8") as f:
        f.write("15A — STEP 1: DUAL RIDGE ANCHORS\n")
        f.write("=" * 120 + "\n\n")

        f.write("TRUE-WIND RIDGE TEST\n")
        f.write("-" * 120 + "\n")
        f.write(f"alpha={wind_alpha:g}, params={wind_params}\n")
        f.write(test_w.to_string(index=False))

        f.write("\n\nVESSEL+HDG RIDGE TEST\n")
        f.write("-" * 120 + "\n")
        f.write(f"alpha={vh_alpha:g}, params={vh_params}\n")
        f.write(test_vh.to_string(index=False))

        f.write("\n\nTRUE-WIND TRAIN OOF\n")
        f.write("-" * 120 + "\n")
        f.write(oof_wm.to_string(index=False))

        f.write("\n\nVESSEL+HDG TRAIN OOF\n")
        f.write("-" * 120 + "\n")
        f.write(oof_vhm.to_string(index=False))

        f.write("\n\nNEXT\n")
        f.write("-" * 120 + "\n")
        f.write(
            "15B: freeze both Ridge anchors and train only the true-wind residual branch. "
            "Use true_wind_residual_oof as training target; optimize/freeze alpha. "
            "Its final true-wind prediction will replace Ridge true wind in Stage15C.\n"
        )

    log("")
    log("=" * 140)
    log("15A FINAL — TRUE-WIND RIDGE TEST")
    log("=" * 140)
    log(test_w.to_string(index=False))

    log("")
    log("=" * 140)
    log("15A FINAL — VESSEL+HDG RIDGE TEST")
    log("=" * 140)
    log(test_vh.to_string(index=False))

    log("")
    log(f"TrueWind alpha={wind_alpha:g} | params={wind_params:,}")
    log(f"Vessel+HDG alpha={vh_alpha:g} | params={vh_params:,}")
    log("")
    log("[NEXT] 15B = true-wind residual branch only.")
    log("       15C = vessel+HDG residual branch using 15B final true wind in apparent-wind physics.")
    log("")
    log(f"[SAVED] {wind_model_path}")
    log(f"[SAVED] {vh_model_path}")
    log(f"[SAVED] {output_dir / '15A_train_oof_anchors.npz'}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n[FATAL ERROR]", flush=True)
        traceback.print_exc()
        sys.exit(1)
