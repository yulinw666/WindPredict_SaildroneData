# -*- coding: utf-8 -*-
"""
15A_R2_refine_VesselHDG_alpha.py

Purpose
-------
Refine ONLY the Vessel+HDG Ridge alpha after Stage15A-R found a coarse optimum
near alpha=3000.

This script:
    - uses TRAIN and VALIDATION only
    - does NOT touch TEST
    - does NOT rebuild OOF yet
    - performs a transparent dense alpha sweep between 1000 and 10000
    - prints/saves the best validation alpha

Once the best alpha is frozen, rebuild the final Vessel+HDG Ridge + OOF once.

Expected dataset:
    SD1090_TPOS2024_JointForecasting_v0_1

Vessel+HDG Ridge input:
    all 11 features x 60 one-minute points

Output:
    future [VESSEL_EAST_MPS, VESSEL_NORTH_MPS, HDG_sin, HDG_cos]
    at [1,2,3,5,10] min

Selection metric:
    validation standardized MSE

Recommended PowerShell
----------------------
python "D:\\project\\WindPredict_SaildroneData\\src\\15A_R2_refine_VesselHDG_alpha.py" --dataset-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\SD1090_TPOS2024_JointForecasting_v0_1" --output-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15A_R2_VesselHDG_AlphaRefine_v0_1"
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

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
    / "15A_R2_VesselHDG_AlphaRefine_v0_1"
)

HORIZONS = [1, 2, 3, 5, 10]

ALL_FEATURE_IDX = list(range(11))
VESSEL_HDG_TARGET_IDX = [2, 3, 4, 5]

# Transparent dense search around the coarse optimum.
ALPHAS = [
    1000.0,
    1250.0,
    1500.0,
    1750.0,
    2000.0,
    2250.0,
    2500.0,
    2750.0,
    3000.0,
    3250.0,
    3500.0,
    3750.0,
    4000.0,
    4500.0,
    5000.0,
    6000.0,
    7500.0,
    10000.0,
]

STATS_CHUNK = 8192
EIG_EPS_REL = 1e-12


def log(msg=""):
    print(msg, flush=True)


def load_split(dataset_dir: Path, split: str):
    path = dataset_dir / f"{split}.npz"

    with np.load(path, allow_pickle=False) as z:
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

    return X, y


def flatten_selected_X(X, idx):
    return np.asarray(
        X[:, :, idx],
        dtype=np.float32,
    ).reshape(
        len(X),
        -1,
    )


def flatten_selected_y(y, idx):
    return np.asarray(
        y[:, :, idx],
        dtype=np.float32,
    ).reshape(
        len(y),
        -1,
    )


def compute_stats(X, y):
    p = X.shape[1]
    q = y.shape[1]

    n = 0
    sum_x = np.zeros(p, dtype=np.float64)
    sum_x2 = np.zeros((p, p), dtype=np.float64)
    sum_y = np.zeros(q, dtype=np.float64)
    sum_y2 = np.zeros(q, dtype=np.float64)
    sum_xy = np.zeros((p, q), dtype=np.float64)

    for a in range(0, len(X), STATS_CHUNK):
        b = min(
            a + STATS_CHUNK,
            len(X),
        )

        xb = np.asarray(
            X[a:b],
            dtype=np.float64,
        )

        yb = np.asarray(
            y[a:b],
            dtype=np.float64,
        )

        n += len(xb)
        sum_x += xb.sum(axis=0)
        sum_x2 += xb.T @ xb
        sum_y += yb.sum(axis=0)
        sum_y2 += np.sum(yb * yb, axis=0)
        sum_xy += xb.T @ yb

    return {
        "n": n,
        "sum_x": sum_x,
        "sum_x2": sum_x2,
        "sum_y": sum_y,
        "sum_y2": sum_y2,
        "sum_xy": sum_xy,
    }


def build_eigensystem(stats):
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

    y_std = np.sqrt(
        y_var
    )

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

    evals, Q = np.linalg.eigh(
        G
    )

    evals = np.maximum(
        evals,
        0.0,
    )

    QtC = Q.T @ C

    return {
        "x_mean": x_mean,
        "y_mean": y_mean,
        "y_std": y_std,
        "evals": evals,
        "Q": Q,
        "QtC": QtC,
    }


def solve(eig, alpha):
    inv = 1.0 / (
        eig["evals"]
        + float(alpha)
    )

    Bz = eig["Q"] @ (
        inv[:, None]
        * eig["QtC"]
    )

    intercept_z = (
        -eig["x_mean"]
        @ Bz
    )

    B_raw = (
        Bz
        * eig["y_std"][
            None,
            :
        ]
    )

    intercept_raw = (
        intercept_z
        * eig["y_std"]
        + eig["y_mean"]
    )

    return (
        B_raw,
        intercept_raw,
    )


def predict(B, b, X):
    return (
        np.asarray(
            X,
            dtype=np.float64,
        )
        @ B
        + b[
            None,
            :
        ]
    )


def standardized_mse(
    y_true,
    y_pred,
    y_mean,
    y_std,
):
    t = (
        np.asarray(
            y_true,
            dtype=np.float64,
        )
        - y_mean[
            None,
            :
        ]
    ) / y_std[
        None,
        :
    ]

    p = (
        np.asarray(
            y_pred,
            dtype=np.float64,
        )
        - y_mean[
            None,
            :
        ]
    ) / y_std[
        None,
        :
    ]

    return float(
        np.mean(
            (
                p - t
            )
            ** 2
        )
    )


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

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log(
        "=" * 120
    )

    log(
        "15A-R2 — VESSEL+HDG RIDGE ALPHA REFINEMENT"
    )

    log(
        "=" * 120
    )

    log(
        "Search interval: 1000 ... 10000"
    )

    log(
        "TEST is NOT loaded."
    )

    Xtr_raw, ytr_raw = load_split(
        args.dataset_dir,
        "train",
    )

    Xva_raw, yva_raw = load_split(
        args.dataset_dir,
        "validation",
    )

    Xtr = flatten_selected_X(
        Xtr_raw,
        ALL_FEATURE_IDX,
    )

    Xva = flatten_selected_X(
        Xva_raw,
        ALL_FEATURE_IDX,
    )

    ytr = flatten_selected_y(
        ytr_raw,
        VESSEL_HDG_TARGET_IDX,
    )

    yva = flatten_selected_y(
        yva_raw,
        VESSEL_HDG_TARGET_IDX,
    )

    log(
        f"TRAIN X={Xtr.shape}, Y={ytr.shape}"
    )

    log(
        f"VAL   X={Xva.shape}, Y={yva.shape}"
    )

    log("")
    log(
        "[1/3] Computing TRAIN sufficient statistics..."
    )

    stats = compute_stats(
        Xtr,
        ytr,
    )

    log(
        "[2/3] Eigendecomposition..."
    )

    eig = build_eigensystem(
        stats
    )

    log("")
    log(
        "[3/3] Dense validation alpha sweep..."
    )

    rows = []

    best_alpha = None
    best_loss = np.inf

    for alpha in ALPHAS:
        B, b = solve(
            eig,
            alpha,
        )

        pred = predict(
            B,
            b,
            Xva,
        )

        loss = standardized_mse(
            yva,
            pred,
            eig[
                "y_mean"
            ],
            eig[
                "y_std"
            ],
        )

        rows.append(
            {
                "alpha": float(
                    alpha
                ),
                "validation_standardized_MSE": float(
                    loss
                ),
            }
        )

        log(
            f"  alpha={alpha:8.0f} | "
            f"validation stdMSE={loss:.10f}"
        )

        if loss < best_loss:
            best_loss = float(
                loss
            )
            best_alpha = float(
                alpha
            )

    df = pd.DataFrame(
        rows
    )

    df[
        "delta_vs_best"
    ] = (
        df[
            "validation_standardized_MSE"
        ]
        - best_loss
    )

    df.to_csv(
        args.output_dir
        / "15A_R2_vessel_hdg_alpha_refinement.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (
        args.output_dir
        / "15A_R2_best_alpha.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "best_alpha": (
                    best_alpha
                ),
                "best_validation_standardized_MSE": (
                    best_loss
                ),
                "search_min": min(
                    ALPHAS
                ),
                "search_max": max(
                    ALPHAS
                ),
                "test_loaded": False,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    log("")
    log(
        "=" * 120
    )

    log(
        f"BEST VESSEL+HDG ALPHA = {best_alpha:g}"
    )

    log(
        f"BEST VALIDATION stdMSE = {best_loss:.10f}"
    )

    log(
        "=" * 120
    )

    log("")
    log(
        "Paste the full alpha table back before rebuilding final OOF."
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
