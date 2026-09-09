# -*- coding: utf-8 -*-
"""
15A_R_refine_1min_Dual_Ridge_Anchors.py

15A-R
=====
Fast alpha refinement + final rebuild of the two 1-min Ridge anchors.

Context window
--------------
The dataset contains 60 consecutive 1-minute records:
    t-59, t-58, ..., t
so there are exactly 60 time points in the model input.
The first-to-last timestamp span is 59 min, while the context contains
60 one-minute observations.

Anchor A — TrueWind Ridge
    input : [U,V,T,RH] x 60 points
    output: [U,V] at 1/2/3/5/10 min

Anchor B — Vessel+HDG Ridge
    input : all 11 features x 60 points
    output:
        VESSEL_EAST_MPS
        VESSEL_NORTH_MPS
        HDG_sin
        HDG_cos
    at 1/2/3/5/10 min

Why 15A-R?
----------
The original 15A search selected:
    TrueWind alpha  = 0.01  (lower search boundary)
    VesselHDG alpha = 1000  (upper search boundary)

15A-R expands/refines the alpha search and uses a fast primal Ridge solver:
    G = Xc^T Xc
    C = Xc^T Yz

For each alpha:
    B = (G + alpha I)^(-1) C

An eigendecomposition of G is computed once, so scanning many alpha values is
much faster than repeatedly calling sklearn Ridge.fit().

Search grids
------------
TrueWind:
    0,
    1e-6, 3e-6,
    1e-5, 3e-5,
    1e-4, 3e-4,
    1e-3, 3e-3,
    1e-2, 3e-2,
    1e-1, 3e-1,
    1, 3, 10

VesselHDG initial:
    1, 3, 10, 30, 100, 300,
    1000, 3000, 10000, 30000, 100000

If the VesselHDG optimum remains at the upper boundary, the grid is
automatically extended by x10 up to 1e8.

Selection metric:
    validation standardized MSE

After alpha freeze:
    - final models are rebuilt on TRAIN only
    - purged blocked 5-fold OOF anchors are rebuilt
    - validation/test predictions are rebuilt
    - outputs remain compatible with Stage 15B

Important OOF improvement
-------------------------
Each OOF fold uses a fold-local target mean/std computed from that fold's
training subset, avoiding target-scaler leakage from the held-out block.

Output directory should be NEW, for example:
    15A_R_1min_Dual_Ridge_Anchors_v0_2

Then Stage15B should point --stage15a-dir to this 15A-R directory.

Recommended PowerShell
----------------------
python "D:\\project\\WindPredict_SaildroneData\\src\\15A_R_refine_1min_Dual_Ridge_Anchors.py" --dataset-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\SD1090_TPOS2024_JointForecasting_v0_1" --output-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15A_R_1min_Dual_Ridge_Anchors_v0_2"
"""

from __future__ import annotations

import argparse
import json
import math
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
    / "15A_R_1min_Dual_Ridge_Anchors_v0_2"
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

TRUEWIND_FEATURE_IDX = [0, 1, 2, 3]
ALL_FEATURE_IDX = list(range(11))
TRUEWIND_TARGET_IDX = [0, 1]
VESSEL_HDG_TARGET_IDX = [2, 3, 4, 5]

TRUEWIND_ALPHAS = [
    0.0,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
    3.0,
    10.0,
]

VESSEL_ALPHAS_INITIAL = [
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
    3000.0,
    10000.0,
    30000.0,
    100000.0,
]

VESSEL_ALPHA_MAX = 1e8

OOF_FOLDS = 5
OOF_PURGE = 10

STATS_CHUNK = 8192

EIG_EPS_REL = 1e-12


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
    }


def flatten_selected_X(split, feature_idx):
    return np.asarray(
        split["X"][:, :, feature_idx],
        dtype=np.float32,
    ).reshape(
        len(split["X"]),
        -1,
    )


def flatten_selected_y(split, target_idx):
    return np.asarray(
        split["y"][:, :, target_idx],
        dtype=np.float32,
    ).reshape(
        len(split["y"]),
        -1,
    )


def unflatten_y(y_flat, n_components):
    return np.asarray(
        y_flat,
        dtype=np.float32,
    ).reshape(
        len(y_flat),
        len(HORIZONS),
        n_components,
    )


# =============================================================================
# Sufficient statistics
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


def add_stats(a, b, sign=1.0):
    out = {
        "n": int(a["n"] + sign * b["n"]),
        "sum_x": a["sum_x"] + sign * b["sum_x"],
        "sum_x2": a["sum_x2"] + sign * b["sum_x2"],
        "sum_y": a["sum_y"] + sign * b["sum_y"],
        "sum_y2": a["sum_y2"] + sign * b["sum_y2"],
        "sum_xy": a["sum_xy"] + sign * b["sum_xy"],
    }

    if out["n"] <= 0:
        raise RuntimeError("Invalid sufficient-statistics subtraction.")

    return out


def compute_stats(X, y, start=0, stop=None, chunk=STATS_CHUNK):
    if stop is None:
        stop = len(X)

    p = X.shape[1]
    q = y.shape[1]

    out = empty_stats(p, q)

    for a in range(start, stop, chunk):
        b = min(a + chunk, stop)

        xb = np.asarray(
            X[a:b],
            dtype=np.float64,
        )

        yb = np.asarray(
            y[a:b],
            dtype=np.float64,
        )

        out["n"] += int(len(xb))
        out["sum_x"] += xb.sum(axis=0)
        out["sum_x2"] += xb.T @ xb
        out["sum_y"] += yb.sum(axis=0)
        out["sum_y2"] += np.sum(yb * yb, axis=0)
        out["sum_xy"] += xb.T @ yb

    return out


def stats_to_centered_system(stats):
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
        G
        + G.T
    )

    # X_centered^T Y_standardized
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


def eigensystem_from_stats(stats):
    sysm = stats_to_centered_system(
        stats
    )

    evals, Q = np.linalg.eigh(
        sysm["G"]
    )

    evals = np.maximum(
        evals,
        0.0,
    )

    QtC = Q.T @ sysm["C"]

    return {
        **sysm,
        "evals": evals,
        "Q": Q,
        "QtC": QtC,
    }


def solve_from_eigensystem(eigsys, alpha):
    evals = eigsys["evals"]
    QtC = eigsys["QtC"]

    if float(alpha) == 0.0:
        cutoff = max(
            float(np.max(evals))
            * EIG_EPS_REL,
            1e-15,
        )

        inv = np.where(
            evals > cutoff,
            1.0 / evals,
            0.0,
        )
    else:
        inv = 1.0 / (
            evals
            + float(alpha)
        )

    Bz = eigsys["Q"] @ (
        inv[:, None]
        * QtC
    )

    intercept_z = (
        -eigsys["x_mean"]
        @ Bz
    )

    # Convert standardized-target prediction directly back to raw units.
    B_raw = (
        Bz
        * eigsys["y_std"][
            None,
            :
        ]
    )

    intercept_raw = (
        intercept_z
        * eigsys["y_std"]
        + eigsys["y_mean"]
    )

    return {
        "coef_raw_p_by_q": B_raw,
        "intercept_raw_q": intercept_raw,
        "x_mean": eigsys["x_mean"],
        "y_mean": eigsys["y_mean"],
        "y_std": eigsys["y_std"],
        "alpha": float(alpha),
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


def standardized_mse_raw(y_true, y_pred, model):
    t = (
        np.asarray(
            y_true,
            dtype=np.float64,
        )
        - model[
            "y_mean"
        ][
            None,
            :
        ]
    ) / model[
        "y_std"
    ][
        None,
        :
    ]

    p = (
        np.asarray(
            y_pred,
            dtype=np.float64,
        )
        - model[
            "y_mean"
        ][
            None,
            :
        ]
    ) / model[
        "y_std"
    ][
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


# =============================================================================
# Alpha refinement
# =============================================================================

def evaluate_alpha_grid(name, eigsys, X_val, y_val, alphas):
    rows = []

    best_alpha = None
    best_loss = np.inf

    log(
        f"[FAST ALPHA SEARCH] {name}"
    )

    for alpha in alphas:
        model = solve_from_eigensystem(
            eigsys,
            alpha,
        )

        pred = predict_fast(
            model,
            X_val,
        )

        loss = standardized_mse_raw(
            y_val,
            pred,
            model,
        )

        rows.append(
            {
                "anchor": name,
                "alpha": float(alpha),
                "validation_standardized_MSE": float(loss),
            }
        )

        log(
            f"  alpha={alpha:12g} | "
            f"validation stdMSE={loss:.10f}"
        )

        if loss < best_loss:
            best_loss = float(loss)
            best_alpha = float(alpha)

    return (
        best_alpha,
        best_loss,
        rows,
    )


def refine_vessel_grid(eigsys, X_val, y_val):
    grid = list(
        VESSEL_ALPHAS_INITIAL
    )

    all_rows = []

    while True:
        alpha, loss, rows = evaluate_alpha_grid(
            "VesselHDG-Ridge",
            eigsys,
            X_val,
            y_val,
            grid,
        )

        # De-duplicate repeated rows after expansion.
        seen = {
            float(r["alpha"])
            for r in all_rows
        }

        all_rows.extend(
            [
                r
                for r in rows
                if float(
                    r["alpha"]
                )
                not in seen
            ]
        )

        if alpha < max(
            grid
        ):
            break

        new_max = min(
            max(
                grid
            )
            * 10.0,
            VESSEL_ALPHA_MAX,
        )

        if new_max <= max(
            grid
        ):
            break

        log(
            f"  [AUTO-EXTEND] best alpha is still upper boundary; "
            f"extending to {new_max:g}"
        )

        grid = sorted(
            set(
                grid
                + [
                    max(
                        grid
                    )
                    * 3.0,
                    new_max,
                ]
            )
        )

    all_df = (
        pd.DataFrame(
            all_rows
        )
        .drop_duplicates(
            subset=[
                "alpha"
            ]
        )
        .sort_values(
            "alpha"
        )
        .reset_index(
            drop=True
        )
    )

    best_row = all_df.loc[
        all_df[
            "validation_standardized_MSE"
        ].idxmin()
    ]

    return (
        float(
            best_row[
                "alpha"
            ]
        ),
        float(
            best_row[
                "validation_standardized_MSE"
            ]
        ),
        all_df,
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

    for k, va in enumerate(
        blocks
    ):
        a = int(
            va[
                0
            ]
        )

        b = int(
            va[
                -1
            ]
        ) + 1

        pa = max(
            0,
            a
            - OOF_PURGE,
        )

        pb = min(
            n,
            b
            + OOF_PURGE,
        )

        folds.append(
            {
                "fold": k,
                "val_idx": va,
                "val_start": a,
                "val_stop": b,
                "purge_start": pa,
                "purge_stop": pb,
            }
        )

    return folds


def oof_fast(
    X,
    y,
    alpha,
    folds,
    full_stats,
    label,
):
    pred = np.full(
        y.shape,
        np.nan,
        dtype=np.float32,
    )

    rows = []

    for fold in folds:
        log(
            f"  [{label} OOF fold {fold['fold']}]"
        )

        excl_stats = compute_stats(
            X,
            y,
            fold[
                "purge_start"
            ],
            fold[
                "purge_stop"
            ],
        )

        train_stats = add_stats(
            full_stats,
            excl_stats,
            sign=-1.0,
        )

        eigsys = eigensystem_from_stats(
            train_stats
        )

        model = solve_from_eigensystem(
            eigsys,
            alpha,
        )

        va = fold[
            "val_idx"
        ]

        pred[
            va
        ] = predict_fast(
            model,
            X[
                va
            ],
        )

        rows.append(
            {
                "anchor": label,
                "fold": int(
                    fold[
                        "fold"
                    ]
                ),
                "alpha": float(
                    alpha
                ),
                "validation_start_index": int(
                    fold[
                        "val_start"
                    ]
                ),
                "validation_end_index_exclusive": int(
                    fold[
                        "val_stop"
                    ]
                ),
                "purge_start_index": int(
                    fold[
                        "purge_start"
                    ]
                ),
                "purge_end_index_exclusive": int(
                    fold[
                        "purge_stop"
                    ]
                ),
                "training_samples": int(
                    train_stats[
                        "n"
                    ]
                ),
                "validation_samples": int(
                    len(
                        va
                    )
                ),
            }
        )

    if not np.isfinite(
        pred
    ).all():
        raise RuntimeError(
            f"{label}: OOF prediction incomplete."
        )

    return (
        pred,
        pd.DataFrame(
            rows
        ),
    )


# =============================================================================
# Metrics
# =============================================================================

def wd_from_uv(uv):
    uv = np.asarray(
        uv,
        dtype=np.float64,
    )

    return (
        np.degrees(
            np.arctan2(
                -uv[
                    ...,
                    0
                ],
                -uv[
                    ...,
                    1
                ],
            )
        )
        % 360.0
    )


def circular_diff_deg(a, b):
    return (
        (
            np.asarray(
                a
            )
            - np.asarray(
                b
            )
            + 180.0
        )
        % 360.0
        - 180.0
    )


def truewind_metrics(y_true_flat, y_pred_flat, split):
    yt = unflatten_y(
        y_true_flat,
        2,
    )

    yp = unflatten_y(
        y_pred_flat,
        2,
    )

    rows = []

    for j, h in enumerate(
        HORIZONS
    ):
        t = yt[
            :,
            j,
            :
        ].astype(
            np.float64
        )

        p = yp[
            :,
            j,
            :
        ].astype(
            np.float64
        )

        e = p - t

        ws_t = np.linalg.norm(
            t,
            axis=1,
        )

        ws_p = np.linalg.norm(
            p,
            axis=1,
        )

        wd_err = circular_diff_deg(
            wd_from_uv(
                p
            ),
            wd_from_uv(
                t
            ),
        )

        rows.append(
            {
                "split": split,
                "horizon_min": h,
                "U_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            e[
                                :,
                                0
                            ]
                            ** 2
                        )
                    )
                ),
                "V_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            e[
                                :,
                                1
                            ]
                            ** 2
                        )
                    )
                ),
                "vector_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                e
                                ** 2,
                                axis=1,
                            )
                        )
                    )
                ),
                "WS_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            (
                                ws_p
                                - ws_t
                            )
                            ** 2
                        )
                    )
                ),
                "WD_RMSE_deg": float(
                    np.sqrt(
                        np.mean(
                            wd_err
                            ** 2
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def normalize_heading_flat(pred_flat):
    p = unflatten_y(
        pred_flat,
        4,
    )

    out = p.copy()

    s = out[
        :,
        :,
        2
    ]

    c = out[
        :,
        :,
        3
    ]

    n = np.sqrt(
        s
        ** 2
        + c
        ** 2
    )

    good = n > 1e-8

    s2 = np.zeros_like(
        s
    )

    c2 = np.ones_like(
        c
    )

    s2[
        good
    ] = (
        s[
            good
        ]
        / n[
            good
        ]
    )

    c2[
        good
    ] = (
        c[
            good
        ]
        / n[
            good
        ]
    )

    out[
        :,
        :,
        2
    ] = s2

    out[
        :,
        :,
        3
    ] = c2

    return out.reshape(
        len(out),
        -1,
    )


def vessel_hdg_metrics(y_true_flat, y_pred_flat, split):
    yt = unflatten_y(
        y_true_flat,
        4,
    )

    yp = unflatten_y(
        normalize_heading_flat(
            y_pred_flat
        ),
        4,
    )

    rows = []

    for j, h in enumerate(
        HORIZONS
    ):
        t = yt[
            :,
            j,
            :
        ].astype(
            np.float64
        )

        p = yp[
            :,
            j,
            :
        ].astype(
            np.float64
        )

        ev = (
            p[
                :,
                0:
                2
            ]
            - t[
                :,
                0:
                2
            ]
        )

        sog_t = np.linalg.norm(
            t[
                :,
                0:
                2
            ],
            axis=1,
        )

        sog_p = np.linalg.norm(
            p[
                :,
                0:
                2
            ],
            axis=1,
        )

        h_t = (
            np.degrees(
                np.arctan2(
                    t[
                        :,
                        2
                    ],
                    t[
                        :,
                        3
                    ],
                )
            )
            % 360.0
        )

        h_p = (
            np.degrees(
                np.arctan2(
                    p[
                        :,
                        2
                    ],
                    p[
                        :,
                        3
                    ],
                )
            )
            % 360.0
        )

        he = circular_diff_deg(
            h_p,
            h_t,
        )

        rows.append(
            {
                "split": split,
                "horizon_min": h,
                "Vessel_East_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            ev[
                                :,
                                0
                            ]
                            ** 2
                        )
                    )
                ),
                "Vessel_North_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            ev[
                                :,
                                1
                            ]
                            ** 2
                        )
                    )
                ),
                "Vessel_vector_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                ev
                                ** 2,
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
                            he
                            ** 2
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

    log(
        "="
        * 140
    )

    log(
        "15A-R — FAST RIDGE ALPHA REFINEMENT + FINAL DUAL ANCHOR REBUILD"
    )

    log(
        "="
        * 140
    )

    log(
        "Input context = 60 consecutive 1-min points: t-59 ... t"
    )

    log(
        "TrueWind input = [U,V,T,RH] x 60"
    )

    log(
        "Vessel+HDG input = all 11 features x 60"
    )

    # -----------------------------------------------------------------
    # Load.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 1/7] Loading dataset."
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

    log(
        f"  train      X={train['X'].shape} y={train['y'].shape}"
    )

    log(
        f"  validation X={val['X'].shape} y={val['y'].shape}"
    )

    log(
        f"  test       X={test['X'].shape} y={test['y'].shape}"
    )

    # Flatten selected views.
    Xw_tr = flatten_selected_X(
        train,
        TRUEWIND_FEATURE_IDX,
    )

    Xw_va = flatten_selected_X(
        val,
        TRUEWIND_FEATURE_IDX,
    )

    Xw_te = flatten_selected_X(
        test,
        TRUEWIND_FEATURE_IDX,
    )

    Yw_tr = flatten_selected_y(
        train,
        TRUEWIND_TARGET_IDX,
    )

    Yw_va = flatten_selected_y(
        val,
        TRUEWIND_TARGET_IDX,
    )

    Yw_te = flatten_selected_y(
        test,
        TRUEWIND_TARGET_IDX,
    )

    Xvh_tr = flatten_selected_X(
        train,
        ALL_FEATURE_IDX,
    )

    Xvh_va = flatten_selected_X(
        val,
        ALL_FEATURE_IDX,
    )

    Xvh_te = flatten_selected_X(
        test,
        ALL_FEATURE_IDX,
    )

    Yvh_tr = flatten_selected_y(
        train,
        VESSEL_HDG_TARGET_IDX,
    )

    Yvh_va = flatten_selected_y(
        val,
        VESSEL_HDG_TARGET_IDX,
    )

    Yvh_te = flatten_selected_y(
        test,
        VESSEL_HDG_TARGET_IDX,
    )

    # -----------------------------------------------------------------
    # Full sufficient statistics.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 2/7] Computing TRAIN sufficient statistics once."
    )

    log(
        "  TrueWind stats..."
    )

    stats_w = compute_stats(
        Xw_tr,
        Yw_tr,
    )

    log(
        "  Vessel+HDG stats..."
    )

    stats_vh = compute_stats(
        Xvh_tr,
        Yvh_tr,
    )

    log(
        "  eigendecomposition: TrueWind..."
    )

    eig_w = eigensystem_from_stats(
        stats_w
    )

    log(
        "  eigendecomposition: Vessel+HDG..."
    )

    eig_vh = eigensystem_from_stats(
        stats_vh
    )

    # -----------------------------------------------------------------
    # Alpha refinement.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 3/7] Fast alpha refinement."
    )

    wind_alpha, wind_loss, wind_rows = (
        evaluate_alpha_grid(
            "TrueWind-Ridge",
            eig_w,
            Xw_va,
            Yw_va,
            TRUEWIND_ALPHAS,
        )
    )

    wind_df = pd.DataFrame(
        wind_rows
    )

    vessel_alpha, vessel_loss, vessel_df = (
        refine_vessel_grid(
            eig_vh,
            Xvh_va,
            Yvh_va,
        )
    )

    wind_df.to_csv(
        out
        / "15A_R_truewind_alpha_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    vessel_df.to_csv(
        out
        / "15A_R_vessel_hdg_alpha_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log(
        f"[FROZEN ALPHA] TrueWind  = {wind_alpha:g} "
        f"| val stdMSE={wind_loss:.10f}"
    )

    log(
        f"[FROZEN ALPHA] VesselHDG = {vessel_alpha:g} "
        f"| val stdMSE={vessel_loss:.10f}"
    )

    # -----------------------------------------------------------------
    # Final models / predictions.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 4/7] Final TRAIN-only models + validation/test predictions."
    )

    model_w = solve_from_eigensystem(
        eig_w,
        wind_alpha,
    )

    model_vh = solve_from_eigensystem(
        eig_vh,
        vessel_alpha,
    )

    pred_w_va = predict_fast(
        model_w,
        Xw_va,
    )

    pred_w_te = predict_fast(
        model_w,
        Xw_te,
    )

    pred_vh_va = predict_fast(
        model_vh,
        Xvh_va,
    )

    pred_vh_te = predict_fast(
        model_vh,
        Xvh_te,
    )

    # -----------------------------------------------------------------
    # OOF.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 5/7] Rebuilding purged blocked 5-fold OOF anchors."
    )

    folds = make_folds(
        len(
            train[
                "X"
            ]
        )
    )

    oof_w, folds_w = oof_fast(
        Xw_tr,
        Yw_tr,
        wind_alpha,
        folds,
        stats_w,
        "TrueWind",
    )

    oof_vh, folds_vh = oof_fast(
        Xvh_tr,
        Yvh_tr,
        vessel_alpha,
        folds,
        stats_vh,
        "VesselHDG",
    )

    pd.concat(
        [
            folds_w,
            folds_vh,
        ],
        ignore_index=True,
    ).to_csv(
        out
        / "15A_R_oof_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Metrics.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 6/7] Metrics + downstream files."
    )

    wind_metrics = pd.concat(
        [
            truewind_metrics(
                Yw_tr,
                oof_w,
                "train_OOF",
            ),
            truewind_metrics(
                Yw_va,
                pred_w_va,
                "validation",
            ),
            truewind_metrics(
                Yw_te,
                pred_w_te,
                "test",
            ),
        ],
        ignore_index=True,
    )

    vessel_metrics = pd.concat(
        [
            vessel_hdg_metrics(
                Yvh_tr,
                oof_vh,
                "train_OOF",
            ),
            vessel_hdg_metrics(
                Yvh_va,
                pred_vh_va,
                "validation",
            ),
            vessel_hdg_metrics(
                Yvh_te,
                pred_vh_te,
                "test",
            ),
        ],
        ignore_index=True,
    )

    wind_metrics.to_csv(
        out
        / "15A_R_truewind_metrics_per_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    vessel_metrics.to_csv(
        out
        / "15A_R_vessel_hdg_metrics_per_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Stage15B-compatible files.
    np.savez_compressed(
        out
        / "15A_train_oof_anchors.npz",
        true_wind_y=unflatten_y(
            Yw_tr,
            2,
        ).astype(
            np.float32
        ),
        true_wind_ridge_oof=unflatten_y(
            oof_w,
            2,
        ).astype(
            np.float32
        ),
        true_wind_residual_oof=(
            unflatten_y(
                Yw_tr,
                2,
            )
            - unflatten_y(
                oof_w,
                2,
            )
        ).astype(
            np.float32
        ),
        vessel_hdg_y=unflatten_y(
            Yvh_tr,
            4,
        ).astype(
            np.float32
        ),
        vessel_hdg_ridge_oof=unflatten_y(
            oof_vh,
            4,
        ).astype(
            np.float32
        ),
        vessel_hdg_ridge_oof_unit_heading=unflatten_y(
            normalize_heading_flat(
                oof_vh
            ),
            4,
        ).astype(
            np.float32
        ),
        vessel_hdg_residual_oof=(
            unflatten_y(
                Yvh_tr,
                4,
            )
            - unflatten_y(
                oof_vh,
                4,
            )
        ).astype(
            np.float32
        ),
        horizons_min=np.asarray(
            HORIZONS,
            dtype=np.int64,
        ),
    )

    np.savez_compressed(
        out
        / "15A_validation_anchor_predictions.npz",
        true_wind_y=unflatten_y(
            Yw_va,
            2,
        ).astype(
            np.float32
        ),
        true_wind_ridge=unflatten_y(
            pred_w_va,
            2,
        ).astype(
            np.float32
        ),
        true_wind_residual=(
            unflatten_y(
                Yw_va,
                2,
            )
            - unflatten_y(
                pred_w_va,
                2,
            )
        ).astype(
            np.float32
        ),
        vessel_hdg_y=unflatten_y(
            Yvh_va,
            4,
        ).astype(
            np.float32
        ),
        vessel_hdg_ridge=unflatten_y(
            pred_vh_va,
            4,
        ).astype(
            np.float32
        ),
        vessel_hdg_ridge_unit_heading=unflatten_y(
            normalize_heading_flat(
                pred_vh_va
            ),
            4,
        ).astype(
            np.float32
        ),
        vessel_hdg_residual=(
            unflatten_y(
                Yvh_va,
                4,
            )
            - unflatten_y(
                pred_vh_va,
                4,
            )
        ).astype(
            np.float32
        ),
        horizons_min=np.asarray(
            HORIZONS,
            dtype=np.int64,
        ),
    )

    np.savez_compressed(
        out
        / "15A_test_anchor_predictions.npz",
        true_wind_y=unflatten_y(
            Yw_te,
            2,
        ).astype(
            np.float32
        ),
        true_wind_ridge=unflatten_y(
            pred_w_te,
            2,
        ).astype(
            np.float32
        ),
        true_wind_residual=(
            unflatten_y(
                Yw_te,
                2,
            )
            - unflatten_y(
                pred_w_te,
                2,
            )
        ).astype(
            np.float32
        ),
        vessel_hdg_y=unflatten_y(
            Yvh_te,
            4,
        ).astype(
            np.float32
        ),
        vessel_hdg_ridge=unflatten_y(
            pred_vh_te,
            4,
        ).astype(
            np.float32
        ),
        vessel_hdg_ridge_unit_heading=unflatten_y(
            normalize_heading_flat(
                pred_vh_te
            ),
            4,
        ).astype(
            np.float32
        ),
        vessel_hdg_residual=(
            unflatten_y(
                Yvh_te,
                4,
            )
            - unflatten_y(
                pred_vh_te,
                4,
            )
        ).astype(
            np.float32
        ),
        horizons_min=np.asarray(
            HORIZONS,
            dtype=np.int64,
        ),
    )

    # Save fast model dictionaries. No later stage needs sklearn to reproduce
    # the anchor prediction.
    wind_model_path = (
        out
        / "15A_truewind_ridge.joblib"
    )

    vessel_model_path = (
        out
        / "15A_vessel_hdg_ridge.joblib"
    )

    joblib.dump(
        {
            "model_type": "fast_primal_ridge",
            "coef_raw_p_by_q": model_w[
                "coef_raw_p_by_q"
            ],
            "intercept_raw_q": model_w[
                "intercept_raw_q"
            ],
            "alpha": wind_alpha,
            "feature_indices": TRUEWIND_FEATURE_IDX,
            "feature_names": [
                FEATURE_NAMES[
                    i
                ]
                for i in TRUEWIND_FEATURE_IDX
            ],
            "target_indices": TRUEWIND_TARGET_IDX,
            "target_names": [
                TARGET_NAMES[
                    i
                ]
                for i in TRUEWIND_TARGET_IDX
            ],
            "horizons_min": HORIZONS,
            "input_shape": (
                60,
                4,
            ),
        },
        wind_model_path,
    )

    joblib.dump(
        {
            "model_type": "fast_primal_ridge",
            "coef_raw_p_by_q": model_vh[
                "coef_raw_p_by_q"
            ],
            "intercept_raw_q": model_vh[
                "intercept_raw_q"
            ],
            "alpha": vessel_alpha,
            "feature_indices": ALL_FEATURE_IDX,
            "feature_names": FEATURE_NAMES,
            "target_indices": VESSEL_HDG_TARGET_IDX,
            "target_names": [
                TARGET_NAMES[
                    i
                ]
                for i in VESSEL_HDG_TARGET_IDX
            ],
            "horizons_min": HORIZONS,
            "input_shape": (
                60,
                11,
            ),
        },
        vessel_model_path,
    )

    # -----------------------------------------------------------------
    # Report.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 7/7] Final report."
    )

    test_w = wind_metrics.loc[
        wind_metrics[
            "split"
        ]
        == "test"
    ].reset_index(
        drop=True
    )

    test_vh = vessel_metrics.loc[
        vessel_metrics[
            "split"
        ]
        == "test"
    ].reset_index(
        drop=True
    )

    manifest = {
        "stage": "15A-R",
        "context": {
            "sampling_interval_min": 1,
            "input_points": 60,
            "timestamps": "t-59 ... t",
            "first_to_last_span_min": 59,
            "common_description": "60-min context / 60 one-minute records",
        },
        "truewind": {
            "alpha": wind_alpha,
            "validation_standardized_MSE": wind_loss,
            "input_features": [
                FEATURE_NAMES[
                    i
                ]
                for i in TRUEWIND_FEATURE_IDX
            ],
            "input_dim": 240,
            "output_dim": 10,
        },
        "vessel_hdg": {
            "alpha": vessel_alpha,
            "validation_standardized_MSE": vessel_loss,
            "input_features": FEATURE_NAMES,
            "input_dim": 660,
            "output_dim": 20,
        },
        "oof": {
            "folds": OOF_FOLDS,
            "purge_each_side_samples": OOF_PURGE,
            "fold_local_target_scaler": True,
        },
        "stage15b_compatible": True,
    }

    save_json(
        out
        / "15A_R_manifest.json",
        manifest,
    )

    with (
        out
        / "15A_R_REPORT.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "15A-R — FAST RIDGE ALPHA REFINEMENT\n"
        )

        f.write(
            "="
            * 120
            + "\n\n"
        )

        f.write(
            "CONTEXT\n"
        )

        f.write(
            "60 consecutive 1-min samples: t-59 ... t\n"
        )

        f.write(
            "TrueWind input: [U,V,T,RH] x 60\n"
        )

        f.write(
            "Vessel+HDG input: all 11 features x 60\n\n"
        )

        f.write(
            f"Frozen TrueWind alpha = {wind_alpha:g}\n"
        )

        f.write(
            f"Frozen VesselHDG alpha = {vessel_alpha:g}\n\n"
        )

        f.write(
            "TRUE-WIND TEST\n"
        )

        f.write(
            "-"
            * 120
            + "\n"
        )

        f.write(
            test_w.to_string(
                index=False
            )
        )

        f.write(
            "\n\nVESSEL+HDG TEST\n"
        )

        f.write(
            "-"
            * 120
            + "\n"
        )

        f.write(
            test_vh.to_string(
                index=False
            )
        )

        f.write(
            "\n\nNEXT\n"
        )

        f.write(
            "-"
            * 120
            + "\n"
        )

        f.write(
            "Use this 15A-R directory as --stage15a-dir for Stage15B.\n"
        )

    log("")
    log(
        "="
        * 140
    )

    log(
        "15A-R FINAL — TRUE-WIND RIDGE TEST"
    )

    log(
        "="
        * 140
    )

    log(
        test_w.to_string(
            index=False
        )
    )

    log("")
    log(
        "="
        * 140
    )

    log(
        "15A-R FINAL — VESSEL+HDG RIDGE TEST"
    )

    log(
        "="
        * 140
    )

    log(
        test_vh.to_string(
            index=False
        )
    )

    log("")
    log(
        f"FINAL TrueWind alpha  = {wind_alpha:g}"
    )

    log(
        f"FINAL VesselHDG alpha = {vessel_alpha:g}"
    )

    log("")
    log(
        "[NEXT] Stage15B should use this directory:"
    )

    log(
        f"       {out}"
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
