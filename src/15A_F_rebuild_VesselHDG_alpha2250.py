# -*- coding: utf-8 -*-
"""
15A_F_rebuild_VesselHDG_alpha2250.py

Final rebuild of the Stage-1 Vessel+HDG Ridge anchor.

Frozen decision
---------------
Vessel+HDG Ridge alpha = 2250

This value was selected by TRAIN/VALIDATION only:
    coarse search -> refinement over 1000...10000
    best validation standardized MSE at alpha = 2250

TEST is NOT used for alpha selection.

Input
-----
All 11 existing joint-dataset features over 60 consecutive 1-min points:

    UWND_MEAN
    VWND_MEAN
    TEMP_AIR_MEAN
    RH_MEAN
    SOG
    COG_sin
    COG_cos
    HDG_sin
    HDG_cos
    WING_ANGLE_sin
    WING_ANGLE_cos

Input shape:
    60 x 11 = 660

Output
------
At horizons [1,2,3,5,10] min:

    VESSEL_EAST_MPS
    VESSEL_NORTH_MPS
    HDG_sin
    HDG_cos

Output shape:
    5 x 4 = 20

This script
-----------
1. Fits the final TRAIN-only Ridge with alpha=2250.
2. Evaluates validation/test.
3. Rebuilds purged blocked 5-fold TRAIN OOF predictions.
4. Uses fold-local target normalization for OOF.
5. Saves Stage15C-ready Vessel+HDG anchor files.

No neural network is trained here.

Recommended PowerShell
----------------------
python "D:\\project\\WindPredict_SaildroneData\\src\\15A_F_rebuild_VesselHDG_alpha2250.py" --dataset-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\SD1090_TPOS2024_JointForecasting_v0_1" --output-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15A_F_VesselHDG_alpha2250_v0_3"
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


ROOT = Path(r"D:\project\WindPredict_SaildroneData")

DEFAULT_DATASET_DIR = (
    ROOT
    / "data"
    / "forecasting"
    / "SD1090_TPOS2024_JointForecasting_v0_1"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "data"
    / "forecasting"
    / "15A_F_VesselHDG_alpha2250_v0_3"
)

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

ALL_FEATURE_IDX = list(range(11))
VESSEL_HDG_TARGET_IDX = [2, 3, 4, 5]

FROZEN_ALPHA = 2250.0

OOF_FOLDS = 5
OOF_PURGE = 10
STATS_CHUNK = 8192


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
            raise RuntimeError(
                f"{path}: required X/y_raw missing. Keys={z.files}"
            )

        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y_raw"], dtype=np.float32)

    if X.ndim != 3 or X.shape[1:] != (60, 11):
        raise RuntimeError(
            f"{split}: expected X=(N,60,11), got {X.shape}"
        )

    if y.ndim != 3 or y.shape[1:] != (5, 6):
        raise RuntimeError(
            f"{split}: expected y_raw=(N,5,6), got {y.shape}"
        )

    if not (
        np.isfinite(X).all()
        and np.isfinite(y).all()
    ):
        raise RuntimeError(
            f"{split}: nonfinite values."
        )

    return {
        "X": X,
        "y": y,
        "path": path,
    }


def flatten_X(split):
    return np.asarray(
        split["X"][:, :, ALL_FEATURE_IDX],
        dtype=np.float32,
    ).reshape(
        len(split["X"]),
        -1,
    )


def flatten_y(split):
    return np.asarray(
        split["y"][:, :, VESSEL_HDG_TARGET_IDX],
        dtype=np.float32,
    ).reshape(
        len(split["y"]),
        -1,
    )


def unflatten_y(y_flat):
    return np.asarray(
        y_flat,
        dtype=np.float32,
    ).reshape(
        len(y_flat),
        len(HORIZONS),
        4,
    )


# =============================================================================
# Fast primal Ridge via sufficient statistics
# =============================================================================

def empty_stats(p, q):
    return {
        "n": 0,
        "sum_x": np.zeros(p, dtype=np.float64),
        "sum_x2": np.zeros((p, p), dtype=np.float64),
        "sum_y": np.zeros(q, dtype=np.float64),
        "sum_y2": np.zeros(q, dtype=np.float64),
        "sum_xy": np.zeros((p, q), dtype=np.float64),
    }


def compute_stats(X, y, start=0, stop=None):
    if stop is None:
        stop = len(X)

    p = X.shape[1]
    q = y.shape[1]

    out = empty_stats(p, q)

    for a in range(start, stop, STATS_CHUNK):
        b = min(a + STATS_CHUNK, stop)

        xb = np.asarray(
            X[a:b],
            dtype=np.float64,
        )

        yb = np.asarray(
            y[a:b],
            dtype=np.float64,
        )

        out["n"] += len(xb)
        out["sum_x"] += xb.sum(axis=0)
        out["sum_x2"] += xb.T @ xb
        out["sum_y"] += yb.sum(axis=0)
        out["sum_y2"] += np.sum(yb * yb, axis=0)
        out["sum_xy"] += xb.T @ yb

    return out


def subtract_stats(full, excluded):
    out = {
        "n": int(full["n"] - excluded["n"]),
        "sum_x": full["sum_x"] - excluded["sum_x"],
        "sum_x2": full["sum_x2"] - excluded["sum_x2"],
        "sum_y": full["sum_y"] - excluded["sum_y"],
        "sum_y2": full["sum_y2"] - excluded["sum_y2"],
        "sum_xy": full["sum_xy"] - excluded["sum_xy"],
    }

    if out["n"] <= 0:
        raise RuntimeError("Invalid stats subtraction.")

    return out


def build_system(stats):
    n = float(stats["n"])

    x_mean = stats["sum_x"] / n
    y_mean = stats["sum_y"] / n

    y_var = (
        stats["sum_y2"] / n
        - y_mean ** 2
    )

    y_var = np.maximum(
        y_var,
        0.0,
    )

    y_std = np.sqrt(y_var)

    y_std = np.where(
        y_std < 1e-8,
        1.0,
        y_std,
    )

    G = (
        stats["sum_x2"]
        - np.outer(
            stats["sum_x"],
            stats["sum_x"],
        )
        / n
    )

    G = 0.5 * (
        G + G.T
    )

    C = (
        stats["sum_xy"]
        - np.outer(
            stats["sum_x"],
            y_mean,
        )
    ) / y_std[
        None,
        :
    ]

    return {
        "x_mean": x_mean,
        "y_mean": y_mean,
        "y_std": y_std,
        "G": G,
        "C": C,
    }


def fit_fast_ridge_from_stats(stats, alpha):
    sysm = build_system(
        stats
    )

    p = sysm["G"].shape[0]

    A = (
        sysm["G"]
        + float(alpha)
        * np.eye(
            p,
            dtype=np.float64,
        )
    )

    Bz = np.linalg.solve(
        A,
        sysm["C"],
    )

    intercept_z = (
        -sysm["x_mean"]
        @ Bz
    )

    B_raw = (
        Bz
        * sysm["y_std"][
            None,
            :
        ]
    )

    intercept_raw = (
        intercept_z
        * sysm["y_std"]
        + sysm["y_mean"]
    )

    return {
        "coef_raw_p_by_q": B_raw,
        "intercept_raw_q": intercept_raw,
        "alpha": float(alpha),
        "x_mean": sysm["x_mean"],
        "y_mean": sysm["y_mean"],
        "y_std": sysm["y_std"],
    }


def predict_fast(model, X):
    return (
        np.asarray(
            X,
            dtype=np.float64,
        )
        @ model[
            "coef_raw_p_by_q"
        ]
        + model[
            "intercept_raw_q"
        ][
            None,
            :
        ]
    ).astype(
        np.float32
    )


# =============================================================================
# OOF
# =============================================================================

def make_folds(n):
    blocks = np.array_split(
        np.arange(
            n,
            dtype=np.int64,
        ),
        OOF_FOLDS,
    )

    folds = []

    for k, va_idx in enumerate(blocks):
        a = int(va_idx[0])
        b = int(va_idx[-1]) + 1

        pa = max(
            0,
            a - OOF_PURGE,
        )

        pb = min(
            n,
            b + OOF_PURGE,
        )

        folds.append(
            {
                "fold": k,
                "val_idx": va_idx,
                "val_start": a,
                "val_stop": b,
                "purge_start": pa,
                "purge_stop": pb,
            }
        )

    return folds


def rebuild_oof(X, y, full_stats):
    pred = np.full(
        y.shape,
        np.nan,
        dtype=np.float32,
    )

    rows = []

    folds = make_folds(
        len(X)
    )

    for fold in folds:
        log(
            f"  [OOF fold {fold['fold']}] "
            f"holdout={len(fold['val_idx']):,}"
        )

        excluded = compute_stats(
            X,
            y,
            start=fold["purge_start"],
            stop=fold["purge_stop"],
        )

        train_stats = subtract_stats(
            full_stats,
            excluded,
        )

        model = fit_fast_ridge_from_stats(
            train_stats,
            FROZEN_ALPHA,
        )

        va = fold["val_idx"]

        pred[va] = predict_fast(
            model,
            X[va],
        )

        rows.append(
            {
                "fold": int(fold["fold"]),
                "alpha": FROZEN_ALPHA,
                "validation_start_index": int(
                    fold["val_start"]
                ),
                "validation_end_index_exclusive": int(
                    fold["val_stop"]
                ),
                "purge_start_index": int(
                    fold["purge_start"]
                ),
                "purge_end_index_exclusive": int(
                    fold["purge_stop"]
                ),
                "training_samples": int(
                    train_stats["n"]
                ),
                "validation_samples": int(
                    len(va)
                ),
            }
        )

    if not np.isfinite(pred).all():
        raise RuntimeError(
            "OOF prediction incomplete/nonfinite."
        )

    return (
        pred,
        pd.DataFrame(rows),
    )


# =============================================================================
# Heading + metrics
# =============================================================================

def normalize_heading_sincos(pred_3d):
    out = np.asarray(
        pred_3d,
        dtype=np.float32,
    ).copy()

    s = out[:, :, 2]
    c = out[:, :, 3]

    norm = np.sqrt(
        s ** 2
        + c ** 2
    )

    good = norm > 1e-8

    s2 = np.zeros_like(s)
    c2 = np.ones_like(c)

    s2[good] = (
        s[good]
        / norm[good]
    )

    c2[good] = (
        c[good]
        / norm[good]
    )

    out[:, :, 2] = s2
    out[:, :, 3] = c2

    return out


def circular_diff_deg(a, b):
    return (
        (
            np.asarray(a)
            - np.asarray(b)
            + 180.0
        )
        % 360.0
        - 180.0
    )


def hdg_deg(s, c):
    return (
        np.degrees(
            np.arctan2(
                s,
                c,
            )
        )
        % 360.0
    )


def metrics(y_true_flat, y_pred_flat, split):
    yt = unflatten_y(
        y_true_flat
    )

    yp = normalize_heading_sincos(
        unflatten_y(
            y_pred_flat
        )
    )

    rows = []

    for j, h in enumerate(HORIZONS):
        t = np.asarray(
            yt[:, j, :],
            dtype=np.float64,
        )

        p = np.asarray(
            yp[:, j, :],
            dtype=np.float64,
        )

        ev = (
            p[:, 0:2]
            - t[:, 0:2]
        )

        sog_t = np.linalg.norm(
            t[:, 0:2],
            axis=1,
        )

        sog_p = np.linalg.norm(
            p[:, 0:2],
            axis=1,
        )

        hdg_t = hdg_deg(
            t[:, 2],
            t[:, 3],
        )

        hdg_p = hdg_deg(
            p[:, 2],
            p[:, 3],
        )

        he = circular_diff_deg(
            hdg_p,
            hdg_t,
        )

        rows.append(
            {
                "split": split,
                "horizon_min": h,
                "Vessel_East_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            ev[:, 0] ** 2
                        )
                    )
                ),
                "Vessel_North_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            ev[:, 1] ** 2
                        )
                    )
                ),
                "Vessel_vector_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                ev ** 2,
                                axis=1,
                            )
                        )
                    )
                ),
                "SOG_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            (
                                sog_p
                                - sog_t
                            )
                            ** 2
                        )
                    )
                ),
                "HDG_MAE_deg": float(
                    np.mean(
                        np.abs(
                            he
                        )
                    )
                ),
                "HDG_RMSE_deg": float(
                    np.sqrt(
                        np.mean(
                            he ** 2
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("=" * 140)
    log(
        "15A-F — FINAL VESSEL+HDG RIDGE REBUILD, alpha=2250"
    )
    log("=" * 140)
    log(
        "Input = all 11 features x 60 one-minute points"
    )
    log(
        "Output = Vessel East/North + HDG sin/cos @ 1/2/3/5/10 min"
    )
    log(
        "Alpha was selected using TRAIN/VALIDATION only."
    )

    log("")
    log(
        "[STAGE 1/6] Loading train/validation/test."
    )

    train = load_split(
        args.dataset_dir,
        "train",
    )

    val = load_split(
        args.dataset_dir,
        "validation",
    )

    test = load_split(
        args.dataset_dir,
        "test",
    )

    Xtr = flatten_X(
        train
    )

    Xva = flatten_X(
        val
    )

    Xte = flatten_X(
        test
    )

    Ytr = flatten_y(
        train
    )

    Yva = flatten_y(
        val
    )

    Yte = flatten_y(
        test
    )

    log(
        f"  TRAIN X={Xtr.shape} Y={Ytr.shape}"
    )

    log(
        f"  VAL   X={Xva.shape} Y={Yva.shape}"
    )

    log(
        f"  TEST  X={Xte.shape} Y={Yte.shape}"
    )

    log("")
    log(
        "[STAGE 2/6] Computing full TRAIN sufficient statistics."
    )

    full_stats = compute_stats(
        Xtr,
        Ytr,
    )

    log("")
    log(
        "[STAGE 3/6] Fitting frozen TRAIN-only Ridge alpha=2250."
    )

    final_model = fit_fast_ridge_from_stats(
        full_stats,
        FROZEN_ALPHA,
    )

    pred_val = predict_fast(
        final_model,
        Xva,
    )

    pred_test = predict_fast(
        final_model,
        Xte,
    )

    log("")
    log(
        "[STAGE 4/6] Rebuilding clean 5-fold purged TRAIN OOF."
    )

    oof_pred, fold_df = rebuild_oof(
        Xtr,
        Ytr,
        full_stats,
    )

    fold_df.to_csv(
        out
        / "15A_F_vessel_hdg_oof_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log(
        "[STAGE 5/6] Metrics + Stage15C-ready prediction files."
    )

    metric_df = pd.concat(
        [
            metrics(
                Ytr,
                oof_pred,
                "train_OOF",
            ),
            metrics(
                Yva,
                pred_val,
                "validation",
            ),
            metrics(
                Yte,
                pred_test,
                "test",
            ),
        ],
        ignore_index=True,
    )

    metric_df.to_csv(
        out
        / "15A_F_vessel_hdg_metrics_per_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ytr3 = unflatten_y(
        Ytr
    )

    yva3 = unflatten_y(
        Yva
    )

    yte3 = unflatten_y(
        Yte
    )

    oof3 = unflatten_y(
        oof_pred
    )

    val3 = unflatten_y(
        pred_val
    )

    test3 = unflatten_y(
        pred_test
    )

    np.savez_compressed(
        out
        / "15A_F_train_vessel_hdg_oof.npz",
        vessel_hdg_y=ytr3.astype(
            np.float32
        ),
        vessel_hdg_ridge_oof=oof3.astype(
            np.float32
        ),
        vessel_hdg_ridge_oof_unit_heading=(
            normalize_heading_sincos(
                oof3
            ).astype(
                np.float32
            )
        ),
        vessel_hdg_residual_oof=(
            ytr3
            - oof3
        ).astype(
            np.float32
        ),
        alpha=np.asarray(
            [FROZEN_ALPHA],
            dtype=np.float32,
        ),
        horizons_min=np.asarray(
            HORIZONS,
            dtype=np.int64,
        ),
    )

    np.savez_compressed(
        out
        / "15A_F_validation_vessel_hdg.npz",
        vessel_hdg_y=yva3.astype(
            np.float32
        ),
        vessel_hdg_ridge=val3.astype(
            np.float32
        ),
        vessel_hdg_ridge_unit_heading=(
            normalize_heading_sincos(
                val3
            ).astype(
                np.float32
            )
        ),
        vessel_hdg_residual=(
            yva3
            - val3
        ).astype(
            np.float32
        ),
        alpha=np.asarray(
            [FROZEN_ALPHA],
            dtype=np.float32,
        ),
        horizons_min=np.asarray(
            HORIZONS,
            dtype=np.int64,
        ),
    )

    np.savez_compressed(
        out
        / "15A_F_test_vessel_hdg.npz",
        vessel_hdg_y=yte3.astype(
            np.float32
        ),
        vessel_hdg_ridge=test3.astype(
            np.float32
        ),
        vessel_hdg_ridge_unit_heading=(
            normalize_heading_sincos(
                test3
            ).astype(
                np.float32
            )
        ),
        vessel_hdg_residual=(
            yte3
            - test3
        ).astype(
            np.float32
        ),
        alpha=np.asarray(
            [FROZEN_ALPHA],
            dtype=np.float32,
        ),
        horizons_min=np.asarray(
            HORIZONS,
            dtype=np.int64,
        ),
    )

    model_path = (
        out
        / "15A_F_vessel_hdg_ridge_alpha2250.joblib"
    )

    joblib.dump(
        {
            "model_type": (
                "fast_primal_ridge"
            ),
            "coef_raw_p_by_q": final_model[
                "coef_raw_p_by_q"
            ],
            "intercept_raw_q": final_model[
                "intercept_raw_q"
            ],
            "alpha": FROZEN_ALPHA,
            "feature_names": FEATURE_NAMES,
            "feature_indices": ALL_FEATURE_IDX,
            "target_names": [
                TARGET_NAMES[
                    i
                ]
                for i in VESSEL_HDG_TARGET_IDX
            ],
            "target_indices": VESSEL_HDG_TARGET_IDX,
            "horizons_min": HORIZONS,
            "input_shape": (
                60,
                11,
            ),
            "output_shape": (
                5,
                4,
            ),
        },
        model_path,
    )

    log("")
    log(
        "[STAGE 6/6] Final report."
    )

    test_table = metric_df.loc[
        metric_df[
            "split"
        ]
        == "test"
    ].reset_index(
        drop=True
    )

    manifest = {
        "stage": "15A-F",
        "purpose": (
            "Final frozen Vessel+HDG Ridge anchor for Stage15C"
        ),
        "dataset": str(
            args.dataset_dir
        ),
        "alpha": FROZEN_ALPHA,
        "alpha_selection": (
            "TRAIN/VALIDATION coarse-to-fine search; TEST not used"
        ),
        "input": {
            "sampling_interval_min": 1,
            "points": 60,
            "features": FEATURE_NAMES,
            "input_dim": 660,
        },
        "output": {
            "horizons_min": HORIZONS,
            "targets": [
                TARGET_NAMES[
                    i
                ]
                for i in VESSEL_HDG_TARGET_IDX
            ],
            "output_dim": 20,
        },
        "OOF": {
            "method": (
                "purged blocked 5-fold"
            ),
            "purge_each_side_samples": (
                OOF_PURGE
            ),
            "fold_local_target_scaling": True,
        },
        "stage15c_ready": True,
    }

    save_json(
        out
        / "15A_F_manifest.json",
        manifest,
    )

    with (
        out
        / "15A_F_REPORT.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "15A-F — FINAL VESSEL+HDG RIDGE alpha=2250\n"
        )

        f.write(
            "="
            * 120
            + "\n\n"
        )

        f.write(
            "TEST\n"
        )

        f.write(
            "-"
            * 120
            + "\n"
        )

        f.write(
            test_table.to_string(
                index=False
            )
        )

        f.write(
            "\n\nStage15C input files:\n"
        )

        f.write(
            "15A_F_train_vessel_hdg_oof.npz\n"
        )

        f.write(
            "15A_F_validation_vessel_hdg.npz\n"
        )

        f.write(
            "15A_F_test_vessel_hdg.npz\n"
        )

    log("")
    log(
        "="
        * 140
    )

    log(
        "15A-F FINAL TEST — VESSEL+HDG RIDGE alpha=2250"
    )

    log(
        "="
        * 140
    )

    log(
        test_table.to_string(
            index=False
        )
    )

    log("")
    log(
        f"[SAVED] {out / '15A_F_train_vessel_hdg_oof.npz'}"
    )

    log(
        f"[SAVED] {out / '15A_F_validation_vessel_hdg.npz'}"
    )

    log(
        f"[SAVED] {out / '15A_F_test_vessel_hdg.npz'}"
    )

    log("")
    log(
        "[NEXT] Stage15C can now combine:"
    )

    log(
        "  1) Stage15B-F final true-wind predictions"
    )

    log(
        "  2) this alpha=2250 Vessel+HDG Ridge anchor"
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(
            main()
        )
    except Exception:
        print(
            "\n[FATAL ERROR]",
            flush=True,
        )
        traceback.print_exc()
        sys.exit(1)
