# -*- coding: utf-8 -*-
r"""
12Q_train_L15_RidgeAnchoredSafeResidual_1min_10minHorizon_LOMO.py

Stage 12Q
=========
Development-only LOMO screening of a SAFE Ridge-anchored residual model after
Stage 12P-A established that a continuous 15-minute, 1-minute-resolution
historical window is the strongest lookback among {6,15,30,60} for prediction
exactly +10 minutes after the context end.

Frozen task
-----------
Raw resolution:
    1 minute

Lookback:
    15 consecutive minutes

Target:
    true wind [U,V] exactly +10 minutes after context end

Dataset:
    Reuse Stage-12P-A common-sample dataset:
        <12P-A-dir>/dataset_1min_common60/

This guarantees the exact same target timestamps used in 12P-A/12P-B.

Scientific hypothesis
---------------------
12P-B showed that:
    - hand-engineered temporal summaries did not improve Ridge-L15;
    - direct MLP/GRU prediction of the full future U/V generalized poorly.

12Q therefore restores the strong Ridge-L15 forecast as an immutable anchor
and asks only whether the remaining error can be predicted from the richer
continuous 15-minute context.

For each fold:

    W_R = Ridge_L15(X)

    r_true = W_true - W_R

A compact network predicts only a correction:

    W_hat = W_R + r_hat

Residuals are normalized by THEIR OWN fold-training standard deviation:

    r_norm = r_true / std_train(r_true)

The neural output head is initialized to zero, so:

    epoch 0 == exact Ridge-L15

Therefore each neural run is allowed to fall back to Ridge if learning a
residual correction is harmful.

Candidates
----------
1) L15-ResidualMLP
   Global standardized 15x9 history + standardized Ridge anchor
   -> normalized global [r_U, r_V]

2) L15-ResidualGRU
   Global standardized 15x9 history -> GRU
   + standardized Ridge anchor
   -> normalized global [r_U, r_V]

3) L15-ResidualLocalGRU
   Sample-specific wind-aligned local frame based on LAST observed wind:
       e_parallel = W_t / ||W_t||
       e_perp     = [-e_parallel_N, e_parallel_E]
   Local-frame input:
       [W_parallel, W_perp, T, RH, SOG,
        COG_parallel, COG_perp,
        WING_sin, WING_cos]
   The Ridge residual is also rotated into the same frame and normalized by
   fold-training local residual standard deviations.
   Predicted correction is rotated back to East/North before evaluation.

No Stage-12P-B engineered summary features are used.

Development-only LOMO
---------------------
Holdout Antarctic  <- train Atlantic + West Coast
Holdout Atlantic   <- train Antarctic + West Coast
Holdout West Coast <- train Antarctic + Atlantic

Tropical Atlantic is NEVER loaded.

Ridge anchor
------------
Ridge(alpha=1.0) is fit independently in each LOMO fold using only the two
training missions:
    - fold-train feature standardization;
    - fold-train target standardization;
    - flattened 15x9 standardized history;
    - direct standardized U/V target.

Neural training
---------------
Loss:
    normalized residual MSE + 1e-4 * correction magnitude penalty

Optimizer:
    AdamW

Checkpoint:
    selected by the held-out DEVELOPMENT mission wind score.

Epoch 0 is evaluated before training and is a valid checkpoint.

Selection score
---------------
Within each fold Base-Ridge-L15 = 1:

    Score =
        0.25 * U_RMSE  / Ridge_U
      + 0.35 * V_RMSE  / Ridge_V
      + 0.25 * WS_RMSE / Ridge_WS
      + 0.15 * WD_RMSE / Ridge_WD

Predeclared continuation rule
-----------------------------
A residual candidate replaces Base-Ridge-L15 only if ALL are true:

    mean score < 0.99
    >= 2 of 3 held-out missions have mean score < 1.0
    mean vector RMSE improves > 0.5%
    at least one of:
        V RMSE improves > 1%
        WD RMSE improves > 1%

Otherwise:
    RETAIN_BASE_RIDGE_L15

Residual diagnostics
--------------------
For every selected checkpoint:
    corr(true Ridge residual U, predicted correction U)
    corr(true Ridge residual V, predicted correction V)
    sign accuracy U/V
    true residual RMS U/V
    predicted correction RMS U/V

These diagnostics reveal whether the richer continuous L15 context finally
makes the Ridge residual cross-mission predictable.

BP-STGNN literature reference
-----------------------------
Published values are stored for CONTEXT ONLY and never used for development
selection:
    U  = 0.74864 m/s
    V  = 0.70506 m/s
    WS = 0.69940 m/s
    WD = 5.584 deg

Comparability caveat:
    same +10-minute wall-clock forecast horizon,
    but not preprocessing-identical because BP-STGNN first resamples the
    original 1-minute data to a 10-minute modeling interval.

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\12Q_train_L15_RidgeAnchoredSafeResidual_1min_10minHorizon_LOMO.py" --pa-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12P_A_1minLookback_10minHorizon_Ridge_LOMO_v0_1" --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12Q_L15_RidgeAnchoredSafeResidual_1min_10minHorizon_LOMO_v0_1"

Smoke test
----------
Add:
    --debug-fast
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import Ridge
except Exception as exc:
    raise RuntimeError(
        "scikit-learn is required. Activate the WindPredict environment."
    ) from exc


SCRIPT_VERSION = "0.1.0-12Q-L15-RidgeAnchoredSafeResidual"

DEFAULT_PROJECT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")
DEFAULT_PA_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12P_A_1minLookback_10minHorizon_Ridge_LOMO_v0_1"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12Q_L15_RidgeAnchoredSafeResidual_1min_10minHorizon_LOMO_v0_1"
)

DEVELOPMENT_MISSIONS = ["Antarctic", "Atlantic", "West Coast"]
SCREENING_SEEDS = [500043, 501052, 502061]

LOOKBACK_MIN = 15
RAW_RESOLUTION_MIN = 1
FORECAST_HORIZON_MIN = 10

FEATURE_NAMES = [
    "U",
    "V",
    "T",
    "RH",
    "SOG",
    "COG_sin",
    "COG_cos",
    "WING_ANGLE_sin",
    "WING_ANGLE_cos",
]

IDX_U = 0
IDX_V = 1
IDX_T = 2
IDX_RH = 3
IDX_SOG = 4
IDX_COG_SIN = 5
IDX_COG_COS = 6
IDX_WING_SIN = 7
IDX_WING_COS = 8

RIDGE_ALPHA = 1.0
EPS = 1e-12
RESIDUAL_STD_FLOOR_MPS = 1e-4
FRAME_MIN_SPEED_MPS = 0.50

SCORE_WEIGHTS = {
    "U": 0.25,
    "V": 0.35,
    "WS": 0.25,
    "WD": 0.15,
}

CONTINUE_SCORE_THRESHOLD = 0.99
MIN_MISSION_WINS = 2
MIN_VECTOR_IMPROVEMENT = 0.005
MIN_V_OR_WD_IMPROVEMENT = 0.01

BP_STGNN_PUBLISHED_REFERENCE = {
    "U_RMSE_mps": 0.74864,
    "V_RMSE_mps": 0.70506,
    "WS_RMSE_mps": 0.69940,
    "WD_RMSE_deg": 5.58400,
}


@dataclass(frozen=True)
class TrainFreeze:
    mlp_hidden_1: int = 64
    mlp_hidden_2: int = 32

    gru_hidden: int = 48
    dense_hidden: int = 48

    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 100
    early_stop_patience: int = 15
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-5
    grad_clip_norm: float = 1.0

    correction_penalty: float = 1e-4
    use_amp: bool = True


FREEZE = TrainFreeze()


CANDIDATES = {
    "L15-ResidualMLP": "mlp",
    "L15-ResidualGRU": "gru",
    "L15-ResidualLocalGRU": "local_gru",
}


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
        json.dump(
            cv(obj),
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def import_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required. Activate the WindPredict environment."
        ) from exc

    return torch, nn


def configure_cuda(torch):
    if not torch.cuda.is_available():
        return

    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def seed_everything(torch, seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def amp_context(torch, enabled: bool, device):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()

    try:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        )
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def create_grad_scaler(torch, enabled: bool):
    if not enabled:
        return None

    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        try:
            return torch.cuda.amp.GradScaler(enabled=True)
        except Exception:
            return None


def state_dict_cpu(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def count_parameters(model):
    return int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )


def sequential_batches(*arrays, batch_size: int):
    if not arrays:
        return

    n = len(arrays[0])

    if any(len(x) != n for x in arrays):
        raise ValueError("Batch arrays have inconsistent first dimension.")

    for start in range(0, n, int(batch_size)):
        stop = min(start + int(batch_size), n)
        yield tuple(x[start:stop] for x in arrays)


# =============================================================================
# Frozen Stage-12P-A dataset loading
# =============================================================================

def load_frozen_lookback(pa_dir: Path):
    path = pa_dir / "selected_lookback.json"

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)

    selected = int(
        d["selected_lookback_minutes"]
    )

    if selected != LOOKBACK_MIN:
        raise RuntimeError(
            f"12Q is frozen for L={LOOKBACK_MIN} min, "
            f"but 12P-A selected L={selected} min."
        )

    if int(
        d["task"]["forecast_horizon_minutes"]
    ) != FORECAST_HORIZON_MIN:
        raise RuntimeError(
            "12P-A forecast horizon does not match 12Q."
        )

    if not bool(
        d.get("development_only", False)
    ):
        raise RuntimeError(
            "12P-A selection is not marked development_only."
        )

    if bool(
        d.get("Tropical_Atlantic_loaded", True)
    ):
        raise RuntimeError(
            "Scientific firewall violation in 12P-A selection file."
        )

    return d


def load_mission(pa_dir: Path, mission: str):
    path = (
        pa_dir
        / "dataset_1min_common60"
        / f"{mission.replace(' ', '_')}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(
        path,
        allow_pickle=False,
    ) as z:
        X60 = np.asarray(
            z["X60_raw"],
            dtype=np.float32,
        )
        y = np.asarray(
            z["y_joint_raw"],
            dtype=np.float32,
        )
        aw = np.asarray(
            z["apparent_ref"],
            dtype=np.float32,
        )
        context = np.asarray(
            z["context_end_time_ns"],
            dtype=np.int64,
        ).reshape(-1)
        target = np.asarray(
            z["target_time_ns"],
            dtype=np.int64,
        ).reshape(-1)
        feature_names = [
            str(x)
            for x in z["feature_names"].tolist()
        ]
        raw_step = int(
            np.asarray(
                z["raw_step_minutes"]
            ).reshape(-1)[0]
        )
        horizon = int(
            np.asarray(
                z["forecast_horizon_minutes"]
            ).reshape(-1)[0]
        )

    if feature_names != FEATURE_NAMES:
        raise RuntimeError(
            f"{path.name}: feature schema mismatch."
        )

    if raw_step != RAW_RESOLUTION_MIN:
        raise RuntimeError(
            f"{path.name}: expected 1-min raw step, found {raw_step}."
        )

    if horizon != FORECAST_HORIZON_MIN:
        raise RuntimeError(
            f"{path.name}: expected +10-min horizon, found {horizon}."
        )

    X = X60[:, -LOOKBACK_MIN:, :].astype(
        np.float32,
        copy=True,
    )

    if X.shape[1:] != (
        LOOKBACK_MIN,
        len(FEATURE_NAMES),
    ):
        raise RuntimeError(
            f"{path.name}: bad X shape {X.shape}."
        )

    if not (
        np.isfinite(X).all()
        and np.isfinite(y).all()
        and np.isfinite(aw).all()
    ):
        raise RuntimeError(
            f"{path.name}: non-finite values."
        )

    return {
        "X": X,
        "y": y,
        "aw": aw,
        "context_end_time_ns": context,
        "target_time_ns": target,
        "mission": mission,
    }


def concat_missions(items):
    return {
        "X": np.concatenate(
            [x["X"] for x in items],
            axis=0,
        ),
        "y": np.concatenate(
            [x["y"] for x in items],
            axis=0,
        ),
        "aw": np.concatenate(
            [x["aw"] for x in items],
            axis=0,
        ),
    }


# =============================================================================
# Fold-only standardization + Ridge anchor
# =============================================================================

def fit_global_scaler(X, y_uv):
    X64 = np.asarray(
        X,
        dtype=np.float64,
    )
    y64 = np.asarray(
        y_uv,
        dtype=np.float64,
    )

    x_mean = np.mean(
        X64,
        axis=(0, 1),
    )
    x_std = np.std(
        X64,
        axis=(0, 1),
        ddof=0,
    )
    x_std = np.where(
        x_std < 1e-8,
        1.0,
        x_std,
    )

    y_mean = np.mean(
        y64,
        axis=0,
    )
    y_std = np.std(
        y64,
        axis=0,
        ddof=0,
    )
    y_std = np.where(
        y_std < 1e-8,
        1.0,
        y_std,
    )

    return {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
    }


def standardize_X(X, scaler):
    return (
        (
            np.asarray(
                X,
                dtype=np.float32,
            )
            - scaler["x_mean"][None, None, :]
        )
        / scaler["x_std"][None, None, :]
    ).astype(np.float32)


def standardize_y(y, scaler):
    return (
        (
            np.asarray(
                y,
                dtype=np.float32,
            )
            - scaler["y_mean"][None, :]
        )
        / scaler["y_std"][None, :]
    ).astype(np.float32)


def inverse_y(yz, scaler):
    return (
        np.asarray(
            yz,
            dtype=np.float32,
        )
        * scaler["y_std"][None, :]
        + scaler["y_mean"][None, :]
    ).astype(np.float32)


def fit_ridge_anchor(
    X_train,
    y_train_uv,
    scaler,
):
    Xz = standardize_X(
        X_train,
        scaler,
    )
    yz = standardize_y(
        y_train_uv,
        scaler,
    )

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        Xz.reshape(
            len(Xz),
            -1,
        ),
        yz,
    )

    return model


def ridge_predict_raw(
    model,
    X,
    scaler,
):
    Xz = standardize_X(
        X,
        scaler,
    )

    pred_z = np.asarray(
        model.predict(
            Xz.reshape(
                len(Xz),
                -1,
            )
        ),
        dtype=np.float32,
    )

    pred_raw = inverse_y(
        pred_z,
        scaler,
    )

    return pred_raw, pred_z


def ridge_parameter_count(model):
    return int(
        np.asarray(model.coef_).size
        + np.asarray(model.intercept_).size
    )


# =============================================================================
# Local wind frame for candidate 3
# =============================================================================

def build_local_frame(X):
    X64 = np.asarray(
        X,
        dtype=np.float64,
    )

    last_w = X64[:, -1, [IDX_U, IDX_V]]
    last_speed = np.linalg.norm(
        last_w,
        axis=1,
    )

    mean_w = np.mean(
        X64[:, :, [IDX_U, IDX_V]],
        axis=1,
    )
    mean_speed = np.linalg.norm(
        mean_w,
        axis=1,
    )

    frame_w = last_w.copy()

    use_mean = (
        last_speed
        < FRAME_MIN_SPEED_MPS
    )
    frame_w[
        use_mean
    ] = mean_w[
        use_mean
    ]

    use_east = (
        use_mean
        & (
            mean_speed
            < FRAME_MIN_SPEED_MPS
        )
    )
    frame_w[
        use_east,
        0,
    ] = 1.0
    frame_w[
        use_east,
        1,
    ] = 0.0

    norm = np.maximum(
        np.linalg.norm(
            frame_w,
            axis=1,
        ),
        EPS,
    )

    epar = (
        frame_w
        / norm[:, None]
    )
    eperp = np.stack(
        [
            -epar[:, 1],
            epar[:, 0],
        ],
        axis=1,
    )

    return (
        epar.astype(np.float32),
        eperp.astype(np.float32),
    )


def rotate_to_local(vec, epar, eperp):
    v = np.asarray(
        vec,
        dtype=np.float64,
    )
    ep = np.asarray(
        epar,
        dtype=np.float64,
    )
    eq = np.asarray(
        eperp,
        dtype=np.float64,
    )

    if v.ndim == 3:
        par = np.sum(
            v * ep[:, None, :],
            axis=-1,
        )
        perp = np.sum(
            v * eq[:, None, :],
            axis=-1,
        )

        return np.stack(
            [par, perp],
            axis=-1,
        ).astype(np.float32)

    if v.ndim == 2:
        par = np.sum(
            v * ep,
            axis=-1,
        )
        perp = np.sum(
            v * eq,
            axis=-1,
        )

        return np.stack(
            [par, perp],
            axis=-1,
        ).astype(np.float32)

    raise ValueError(
        f"Unsupported vector shape: {v.shape}"
    )


def rotate_to_global(vec_local, epar, eperp):
    q = np.asarray(
        vec_local,
        dtype=np.float64,
    )
    ep = np.asarray(
        epar,
        dtype=np.float64,
    )
    eq = np.asarray(
        eperp,
        dtype=np.float64,
    )

    out = (
        q[:, 0:1]
        * ep
        + q[:, 1:2]
        * eq
    )

    return out.astype(np.float32)


def make_local_X(X):
    X = np.asarray(
        X,
        dtype=np.float32,
    )

    epar, eperp = build_local_frame(
        X
    )

    out = np.empty_like(
        X,
        dtype=np.float32,
    )

    wind_local = rotate_to_local(
        X[:, :, [IDX_U, IDX_V]],
        epar,
        eperp,
    )
    out[:, :, 0:2] = wind_local

    out[:, :, 2] = X[:, :, IDX_T]
    out[:, :, 3] = X[:, :, IDX_RH]
    out[:, :, 4] = X[:, :, IDX_SOG]

    cog_en = np.stack(
        [
            X[:, :, IDX_COG_SIN],
            X[:, :, IDX_COG_COS],
        ],
        axis=-1,
    )

    cog_local = rotate_to_local(
        cog_en,
        epar,
        eperp,
    )
    out[:, :, 5:7] = cog_local

    out[:, :, 7] = X[:, :, IDX_WING_SIN]
    out[:, :, 8] = X[:, :, IDX_WING_COS]

    return {
        "X_local": out,
        "epar": epar,
        "eperp": eperp,
    }


def fit_local_input_scaler(X_local):
    X64 = np.asarray(
        X_local,
        dtype=np.float64,
    )

    mean = np.mean(
        X64,
        axis=(0, 1),
    )
    std = np.std(
        X64,
        axis=(0, 1),
        ddof=0,
    )
    std = np.where(
        std < 1e-8,
        1.0,
        std,
    )

    return {
        "x_mean": mean.astype(np.float32),
        "x_std": std.astype(np.float32),
    }


def standardize_local_X(X_local, scaler):
    return (
        (
            np.asarray(
                X_local,
                dtype=np.float32,
            )
            - scaler["x_mean"][None, None, :]
        )
        / scaler["x_std"][None, None, :]
    ).astype(np.float32)


def fit_context_scaler(context):
    c = np.asarray(
        context,
        dtype=np.float64,
    )

    mean = np.mean(
        c,
        axis=0,
    )
    std = np.std(
        c,
        axis=0,
        ddof=0,
    )
    std = np.where(
        std < 1e-8,
        1.0,
        std,
    )

    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }


def standardize_context(context, scaler):
    return (
        (
            np.asarray(
                context,
                dtype=np.float32,
            )
            - scaler["mean"][None, :]
        )
        / scaler["std"][None, :]
    ).astype(np.float32)


# =============================================================================
# Metrics / diagnostics
# =============================================================================

def circular_diff_deg(pred_deg, true_deg):
    return (
        (
            np.asarray(
                pred_deg,
                dtype=np.float64,
            )
            - np.asarray(
                true_deg,
                dtype=np.float64,
            )
            + 180.0
        )
        % 360.0
        - 180.0
    )


def meteorological_wd(uv):
    uv = np.asarray(
        uv,
        dtype=np.float64,
    )

    U = uv[:, 0]
    V = uv[:, 1]

    return (
        np.degrees(
            np.arctan2(
                -U,
                -V,
            )
        )
        % 360.0
    )


def evaluate_wind(y_true, y_pred):
    yt = np.asarray(
        y_true,
        dtype=np.float64,
    )
    yp = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    err = yp - yt

    u_rmse = float(
        np.sqrt(
            np.mean(
                err[:, 0] ** 2
            )
        )
    )
    v_rmse = float(
        np.sqrt(
            np.mean(
                err[:, 1] ** 2
            )
        )
    )
    vector_rmse = float(
        np.sqrt(
            np.mean(
                np.sum(
                    err ** 2,
                    axis=1,
                )
            )
        )
    )

    ws_true = np.linalg.norm(
        yt,
        axis=1,
    )
    ws_pred = np.linalg.norm(
        yp,
        axis=1,
    )
    ws_rmse = float(
        np.sqrt(
            np.mean(
                (
                    ws_pred
                    - ws_true
                ) ** 2
            )
        )
    )

    wd_true = meteorological_wd(
        yt
    )
    wd_pred = meteorological_wd(
        yp
    )
    wd_err = circular_diff_deg(
        wd_pred,
        wd_true,
    )

    return {
        "wind_U_RMSE_mps": u_rmse,
        "wind_V_RMSE_mps": v_rmse,
        "wind_vector_RMSE_mps": vector_rmse,
        "wind_speed_RMSE_mps": ws_rmse,
        "wind_direction_MAE_deg": float(
            np.mean(
                np.abs(
                    wd_err
                )
            )
        ),
        "wind_direction_RMSE_deg": float(
            np.sqrt(
                np.mean(
                    wd_err ** 2
                )
            )
        ),
    }


def corrcoef_safe(a, b):
    a = np.asarray(
        a,
        dtype=np.float64,
    ).reshape(-1)
    b = np.asarray(
        b,
        dtype=np.float64,
    ).reshape(-1)

    mask = (
        np.isfinite(a)
        & np.isfinite(b)
    )

    a = a[mask]
    b = b[mask]

    if len(a) < 3:
        return np.nan

    if (
        np.std(a) < EPS
        or np.std(b) < EPS
    ):
        return np.nan

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def residual_diagnostics(
    y_true,
    pred,
    ridge_pred,
):
    yt = np.asarray(
        y_true,
        dtype=np.float64,
    )
    yp = np.asarray(
        pred,
        dtype=np.float64,
    )
    yr = np.asarray(
        ridge_pred,
        dtype=np.float64,
    )

    true_r = (
        yt - yr
    )
    corr = (
        yp - yr
    )

    def sign_acc(a, b):
        return float(
            np.mean(
                np.sign(a)
                == np.sign(b)
            )
        )

    return {
        "residual_corr_U": corrcoef_safe(
            true_r[:, 0],
            corr[:, 0],
        ),
        "residual_corr_V": corrcoef_safe(
            true_r[:, 1],
            corr[:, 1],
        ),
        "residual_sign_accuracy_U": sign_acc(
            true_r[:, 0],
            corr[:, 0],
        ),
        "residual_sign_accuracy_V": sign_acc(
            true_r[:, 1],
            corr[:, 1],
        ),
        "true_residual_RMS_U_mps": float(
            np.sqrt(
                np.mean(
                    true_r[:, 0] ** 2
                )
            )
        ),
        "true_residual_RMS_V_mps": float(
            np.sqrt(
                np.mean(
                    true_r[:, 1] ** 2
                )
            )
        ),
        "pred_correction_RMS_U_mps": float(
            np.sqrt(
                np.mean(
                    corr[:, 0] ** 2
                )
            )
        ),
        "pred_correction_RMS_V_mps": float(
            np.sqrt(
                np.mean(
                    corr[:, 1] ** 2
                )
            )
        ),
    }


def selection_score(metrics, ridge_metrics):
    ratios = {
        "U_ratio": (
            metrics["wind_U_RMSE_mps"]
            / max(
                ridge_metrics["wind_U_RMSE_mps"],
                EPS,
            )
        ),
        "V_ratio": (
            metrics["wind_V_RMSE_mps"]
            / max(
                ridge_metrics["wind_V_RMSE_mps"],
                EPS,
            )
        ),
        "WS_ratio": (
            metrics["wind_speed_RMSE_mps"]
            / max(
                ridge_metrics["wind_speed_RMSE_mps"],
                EPS,
            )
        ),
        "WD_ratio": (
            metrics["wind_direction_RMSE_deg"]
            / max(
                ridge_metrics["wind_direction_RMSE_deg"],
                EPS,
            )
        ),
    }

    score = (
        SCORE_WEIGHTS["U"]
        * ratios["U_ratio"]
        + SCORE_WEIGHTS["V"]
        * ratios["V_ratio"]
        + SCORE_WEIGHTS["WS"]
        * ratios["WS_ratio"]
        + SCORE_WEIGHTS["WD"]
        * ratios["WD_ratio"]
    )

    return float(score), ratios


# =============================================================================
# Candidate preparation
# =============================================================================

def prepare_candidate_data(
    mode,
    X_train,
    X_val,
    y_train,
    y_val,
    ridge_train_raw,
    ridge_val_raw,
    global_scaler,
):
    """
    Returns candidate-specific standardized inputs, residual targets, and
    metadata needed to reconstruct corrections.
    """
    if mode in {"mlp", "gru"}:
        X_train_z = standardize_X(
            X_train,
            global_scaler,
        )
        X_val_z = standardize_X(
            X_val,
            global_scaler,
        )

        ridge_context_train = standardize_y(
            ridge_train_raw,
            global_scaler,
        )
        ridge_context_val = standardize_y(
            ridge_val_raw,
            global_scaler,
        )

        train_residual = (
            y_train
            - ridge_train_raw
        ).astype(np.float32)

        residual_std = np.std(
            train_residual.astype(
                np.float64
            ),
            axis=0,
            ddof=0,
        )
        residual_std = np.where(
            residual_std
            < RESIDUAL_STD_FLOOR_MPS,
            1.0,
            residual_std,
        ).astype(np.float32)

        target_norm = (
            train_residual
            / residual_std[None, :]
        ).astype(np.float32)

        return {
            "X_train_z": X_train_z,
            "X_val_z": X_val_z,
            "ridge_context_train_z": (
                ridge_context_train
            ),
            "ridge_context_val_z": (
                ridge_context_val
            ),
            "target_norm": target_norm,
            "residual_std": residual_std,
            "val_frame": None,
            "mode": mode,
        }

    if mode == "local_gru":
        train_local = make_local_X(
            X_train
        )
        val_local = make_local_X(
            X_val
        )

        local_scaler = fit_local_input_scaler(
            train_local["X_local"]
        )

        X_train_z = standardize_local_X(
            train_local["X_local"],
            local_scaler,
        )
        X_val_z = standardize_local_X(
            val_local["X_local"],
            local_scaler,
        )

        ridge_train_local = rotate_to_local(
            ridge_train_raw,
            train_local["epar"],
            train_local["eperp"],
        )
        ridge_val_local = rotate_to_local(
            ridge_val_raw,
            val_local["epar"],
            val_local["eperp"],
        )

        context_scaler = fit_context_scaler(
            ridge_train_local
        )

        ridge_context_train_z = standardize_context(
            ridge_train_local,
            context_scaler,
        )
        ridge_context_val_z = standardize_context(
            ridge_val_local,
            context_scaler,
        )

        residual_train_global = (
            y_train
            - ridge_train_raw
        )

        residual_train_local = rotate_to_local(
            residual_train_global,
            train_local["epar"],
            train_local["eperp"],
        )

        residual_std = np.std(
            residual_train_local.astype(
                np.float64
            ),
            axis=0,
            ddof=0,
        )
        residual_std = np.where(
            residual_std
            < RESIDUAL_STD_FLOOR_MPS,
            1.0,
            residual_std,
        ).astype(np.float32)

        target_norm = (
            residual_train_local
            / residual_std[None, :]
        ).astype(np.float32)

        return {
            "X_train_z": X_train_z,
            "X_val_z": X_val_z,
            "ridge_context_train_z": (
                ridge_context_train_z
            ),
            "ridge_context_val_z": (
                ridge_context_val_z
            ),
            "target_norm": target_norm,
            "residual_std": residual_std,
            "val_frame": {
                "epar": val_local["epar"],
                "eperp": val_local["eperp"],
            },
            "mode": mode,
        }

    raise ValueError(
        f"Unknown candidate mode: {mode}"
    )


# =============================================================================
# Neural models
# =============================================================================

def build_model_class(
    torch,
    nn,
    mode,
):
    feature_count = len(
        FEATURE_NAMES
    )

    if mode == "mlp":
        class ResidualMLP(nn.Module):
            def __init__(self):
                super().__init__()

                self.backbone = nn.Sequential(
                    nn.Linear(
                        LOOKBACK_MIN
                        * feature_count
                        + 2,
                        FREEZE.mlp_hidden_1,
                    ),
                    nn.LayerNorm(
                        FREEZE.mlp_hidden_1
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        FREEZE.dropout
                    ),
                    nn.Linear(
                        FREEZE.mlp_hidden_1,
                        FREEZE.mlp_hidden_2,
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        FREEZE.dropout
                    ),
                )

                self.out = nn.Linear(
                    FREEZE.mlp_hidden_2,
                    2,
                )

                # Exact Ridge at epoch 0.
                nn.init.zeros_(
                    self.out.weight
                )
                nn.init.zeros_(
                    self.out.bias
                )

            def forward(
                self,
                x,
                ridge_context,
            ):
                z = torch.cat(
                    [
                        x.flatten(
                            start_dim=1
                        ),
                        ridge_context,
                    ],
                    dim=1,
                )

                return self.out(
                    self.backbone(
                        z
                    )
                )

        return ResidualMLP

    if mode in {
        "gru",
        "local_gru",
    }:
        class ResidualGRU(nn.Module):
            def __init__(self):
                super().__init__()

                self.gru = nn.GRU(
                    input_size=feature_count,
                    hidden_size=FREEZE.gru_hidden,
                    num_layers=1,
                    batch_first=True,
                )

                self.backbone = nn.Sequential(
                    nn.Linear(
                        FREEZE.gru_hidden
                        + 2,
                        FREEZE.dense_hidden,
                    ),
                    nn.LayerNorm(
                        FREEZE.dense_hidden
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        FREEZE.dropout
                    ),
                )

                self.out = nn.Linear(
                    FREEZE.dense_hidden,
                    2,
                )

                # Exact Ridge at epoch 0.
                nn.init.zeros_(
                    self.out.weight
                )
                nn.init.zeros_(
                    self.out.bias
                )

            def forward(
                self,
                x,
                ridge_context,
            ):
                seq, _ = self.gru(
                    x
                )

                h = seq[
                    :,
                    -1,
                    :,
                ]

                z = torch.cat(
                    [
                        h,
                        ridge_context,
                    ],
                    dim=1,
                )

                return self.out(
                    self.backbone(
                        z
                    )
                )

        return ResidualGRU

    raise ValueError(
        f"Unknown model mode: {mode}"
    )


def reconstruct_prediction(
    pred_norm,
    candidate_data,
    ridge_val_raw,
):
    correction_native = (
        np.asarray(
            pred_norm,
            dtype=np.float32,
        )
        * candidate_data[
            "residual_std"
        ][None, :]
    )

    if candidate_data[
        "mode"
    ] in {
        "mlp",
        "gru",
    }:
        correction_global = (
            correction_native
        )
    else:
        correction_global = (
            rotate_to_global(
                correction_native,
                candidate_data[
                    "val_frame"
                ]["epar"],
                candidate_data[
                    "val_frame"
                ]["eperp"],
            )
        )

    pred = (
        np.asarray(
            ridge_val_raw,
            dtype=np.float32,
        )
        + correction_global
    )

    return pred.astype(
        np.float32
    )


def evaluate_model(
    *,
    torch,
    model,
    candidate_data,
    y_val,
    ridge_val_raw,
    device,
    amp_enabled,
):
    model.eval()

    chunks = []

    with torch.no_grad():
        for xb_np, cb_np in sequential_batches(
            candidate_data[
                "X_val_z"
            ],
            candidate_data[
                "ridge_context_val_z"
            ],
            batch_size=FREEZE.batch_size,
        ):
            xb = torch.from_numpy(
                xb_np
            ).to(
                device,
                non_blocking=True,
            )

            cb = torch.from_numpy(
                cb_np
            ).to(
                device,
                non_blocking=True,
            )

            with amp_context(
                torch,
                amp_enabled,
                device,
            ):
                pred_norm = model(
                    xb,
                    cb,
                )

            chunks.append(
                pred_norm.detach()
                .float()
                .cpu()
                .numpy()
            )

    pred_norm = np.concatenate(
        chunks,
        axis=0,
    ).astype(np.float32)

    pred = reconstruct_prediction(
        pred_norm,
        candidate_data,
        ridge_val_raw,
    )

    metrics = evaluate_wind(
        y_val,
        pred,
    )

    metrics.update(
        residual_diagnostics(
            y_val,
            pred,
            ridge_val_raw,
        )
    )

    metrics[
        "pred_residual_norm_RMS"
    ] = float(
        np.sqrt(
            np.mean(
                pred_norm ** 2
            )
        )
    )

    return metrics, pred_norm, pred


def train_one_run(
    *,
    torch,
    nn,
    candidate,
    mode,
    seed,
    held_out,
    candidate_data,
    y_val,
    ridge_val_raw,
    ridge_val_metrics,
    output_dir,
    device,
    debug_fast,
):
    seed_everything(
        torch,
        seed,
    )

    ModelClass = build_model_class(
        torch,
        nn,
        mode,
    )

    model = ModelClass().to(
        device
    )

    parameter_count = count_parameters(
        model
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FREEZE.learning_rate,
        weight_decay=FREEZE.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=FREEZE.scheduler_factor,
        patience=FREEZE.scheduler_patience,
        min_lr=FREEZE.min_learning_rate,
    )

    amp_enabled = bool(
        FREEZE.use_amp
        and device.type == "cuda"
    )

    grad_scaler = create_grad_scaler(
        torch,
        amp_enabled,
    )

    # Epoch 0 = exact Ridge.
    initial_metrics, _, _ = evaluate_model(
        torch=torch,
        model=model,
        candidate_data=candidate_data,
        y_val=y_val,
        ridge_val_raw=ridge_val_raw,
        device=device,
        amp_enabled=amp_enabled,
    )

    best_score, best_parts = selection_score(
        initial_metrics,
        ridge_val_metrics,
    )

    best_epoch = 0
    best_state = state_dict_cpu(
        model
    )
    best_metrics = dict(
        initial_metrics
    )
    best_parts_saved = dict(
        best_parts
    )

    history = [{
        "candidate": candidate,
        "held_out_mission": held_out,
        "seed": int(seed),
        "epoch_one_based": 0,
        "learning_rate": float(
            FREEZE.learning_rate
        ),
        "train_total_loss": np.nan,
        "train_residual_mse": np.nan,
        "train_correction_penalty": np.nan,
        "selection_score": float(
            best_score
        ),
        "checkpoint_improved": True,
        **{
            f"val_{k}": v
            for k, v
            in initial_metrics.items()
        },
        **{
            f"score_{k}": v
            for k, v
            in best_parts.items()
        },
    }]

    max_epochs = (
        8
        if debug_fast
        else FREEZE.max_epochs
    )

    patience_limit = (
        4
        if debug_fast
        else FREEZE.early_stop_patience
    )

    patience = 0
    started = time.perf_counter()

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        model.train()

        sum_total = 0.0
        sum_mse = 0.0
        sum_penalty = 0.0
        n_seen = 0

        for (
            xb_np,
            cb_np,
            tb_np,
        ) in sequential_batches(
            candidate_data[
                "X_train_z"
            ],
            candidate_data[
                "ridge_context_train_z"
            ],
            candidate_data[
                "target_norm"
            ],
            batch_size=FREEZE.batch_size,
        ):
            xb = torch.from_numpy(
                xb_np
            ).to(
                device,
                non_blocking=True,
            )

            cb = torch.from_numpy(
                cb_np
            ).to(
                device,
                non_blocking=True,
            )

            tb = torch.from_numpy(
                tb_np
            ).to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with amp_context(
                torch,
                amp_enabled,
                device,
            ):
                pred_norm = model(
                    xb,
                    cb,
                )

                mse = torch.mean(
                    (
                        pred_norm
                        - tb
                    ) ** 2
                )

                penalty = torch.mean(
                    pred_norm ** 2
                )

                total = (
                    mse
                    + FREEZE.correction_penalty
                    * penalty
                )

            if not torch.isfinite(
                total
            ):
                raise FloatingPointError(
                    "Non-finite 12Q residual loss."
                )

            if grad_scaler is not None:
                grad_scaler.scale(
                    total
                ).backward()

                grad_scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    FREEZE.grad_clip_norm,
                )

                grad_scaler.step(
                    optimizer
                )
                grad_scaler.update()
            else:
                total.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    FREEZE.grad_clip_norm,
                )

                optimizer.step()

            bn = len(
                xb_np
            )

            sum_total += float(
                total.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            sum_mse += float(
                mse.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            sum_penalty += float(
                penalty.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            n_seen += bn

        val_metrics, _, _ = evaluate_model(
            torch=torch,
            model=model,
            candidate_data=candidate_data,
            y_val=y_val,
            ridge_val_raw=ridge_val_raw,
            device=device,
            amp_enabled=amp_enabled,
        )

        score, parts = selection_score(
            val_metrics,
            ridge_val_metrics,
        )

        improved = (
            score
            < best_score - 1e-8
        )

        if improved:
            best_score = float(
                score
            )
            best_epoch = int(
                epoch
            )
            best_state = state_dict_cpu(
                model
            )
            best_metrics = dict(
                val_metrics
            )
            best_parts_saved = dict(
                parts
            )
            patience = 0
        else:
            patience += 1

        lr = float(
            optimizer.param_groups[0][
                "lr"
            ]
        )

        scheduler.step(
            score
        )

        history.append({
            "candidate": candidate,
            "held_out_mission": held_out,
            "seed": int(seed),
            "epoch_one_based": int(
                epoch
            ),
            "learning_rate": lr,
            "train_total_loss": (
                sum_total
                / max(
                    n_seen,
                    1,
                )
            ),
            "train_residual_mse": (
                sum_mse
                / max(
                    n_seen,
                    1,
                )
            ),
            "train_correction_penalty": (
                sum_penalty
                / max(
                    n_seen,
                    1,
                )
            ),
            "selection_score": float(
                score
            ),
            "checkpoint_improved": bool(
                improved
            ),
            **{
                f"val_{k}": v
                for k, v
                in val_metrics.items()
            },
            **{
                f"score_{k}": v
                for k, v
                in parts.items()
            },
        })

        log(
            f"      ep={epoch:03d} | "
            f"score={score:.5f} | "
            f"U={val_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={val_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={val_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={val_metrics['wind_direction_RMSE_deg']:.3f} | "
            f"corr=({val_metrics['residual_corr_U']:+.3f},"
            f"{val_metrics['residual_corr_V']:+.3f})"
            + (
                " *"
                if improved
                else ""
            )
        )

        if patience >= patience_limit:
            log(
                f"      early stop after {patience} "
                "epochs without score improvement."
            )
            break

    elapsed = (
        time.perf_counter()
        - started
    )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    confirmed, _, _ = evaluate_model(
        torch=torch,
        model=model,
        candidate_data=candidate_data,
        y_val=y_val,
        ridge_val_raw=ridge_val_raw,
        device=device,
        amp_enabled=amp_enabled,
    )

    confirmed_score, confirmed_parts = (
        selection_score(
            confirmed,
            ridge_val_metrics,
        )
    )

    if abs(
        confirmed_score
        - best_score
    ) > 1e-6:
        raise RuntimeError(
            "Best checkpoint re-evaluation mismatch."
        )

    histories_dir = (
        output_dir
        / "histories"
    )
    checkpoints_dir = (
        output_dir
        / "checkpoints"
    )

    histories_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    checkpoints_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tag = (
        f"{candidate}"
        f"__holdout_{held_out.replace(' ', '_')}"
        f"__seed_{seed}"
    )

    history_path = (
        histories_dir
        / f"{tag}.csv"
    )
    checkpoint_path = (
        checkpoints_dir
        / f"{tag}.pt"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False,
        encoding="utf-8-sig",
    )

    torch.save({
        "stage": "12Q",
        "script_version": SCRIPT_VERSION,
        "candidate": candidate,
        "mode": mode,
        "held_out_mission": held_out,
        "seed": int(seed),
        "lookback_minutes": int(
            LOOKBACK_MIN
        ),
        "raw_resolution_minutes": int(
            RAW_RESOLUTION_MIN
        ),
        "forecast_horizon_minutes": int(
            FORECAST_HORIZON_MIN
        ),
        "parameter_count": int(
            parameter_count
        ),
        "best_epoch_one_based": int(
            best_epoch
        ),
        "selection_score": float(
            confirmed_score
        ),
        "validation_metrics": (
            confirmed
        ),
        "score_parts": (
            confirmed_parts
        ),
        "state_dict": (
            best_state
        ),
        "candidate_data_scaling": {
            "residual_std": (
                candidate_data[
                    "residual_std"
                ]
            ),
        },
        "train_freeze": asdict(
            FREEZE
        ),
        "Tropical_Atlantic_used": False,
    }, checkpoint_path)

    result = {
        "candidate": candidate,
        "held_out_mission": held_out,
        "seed": int(seed),
        "status": "completed_finite",
        "parameter_count": int(
            parameter_count
        ),
        "best_epoch_one_based": int(
            best_epoch
        ),
        "selection_score": float(
            confirmed_score
        ),
        "elapsed_seconds": float(
            elapsed
        ),
        "residual_std_channel_0": float(
            candidate_data[
                "residual_std"
            ][0]
        ),
        "residual_std_channel_1": float(
            candidate_data[
                "residual_std"
            ][1]
        ),
        **confirmed,
        **{
            f"score_{k}": v
            for k, v
            in confirmed_parts.items()
        },
        "history_path": str(
            history_path
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
    }

    del (
        model,
        optimizer,
        scheduler,
        grad_scaler,
    )

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# =============================================================================
# CSV / summary
# =============================================================================

def append_csv(path: Path, row: dict):
    new = pd.DataFrame(
        [row]
    )

    if path.exists():
        old = pd.read_csv(
            path
        )

        keys = [
            "candidate",
            "held_out_mission",
            "seed",
        ]

        if all(
            k in old.columns
            for k in keys
        ):
            mask = (
                (
                    old["candidate"].astype(str)
                    == str(
                        row["candidate"]
                    )
                )
                & (
                    old["held_out_mission"].astype(str)
                    == str(
                        row["held_out_mission"]
                    )
                )
                & (
                    old["seed"].astype(int)
                    == int(
                        row["seed"]
                    )
                )
            )

            old = old.loc[
                ~mask
            ].copy()

        new = pd.concat(
            [
                old,
                new,
            ],
            ignore_index=True,
        )

    new.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def summarize_candidate(
    runs,
    candidate,
):
    sub = runs.loc[
        (
            runs["candidate"].astype(str)
            == candidate
        )
        & (
            runs["status"].astype(str)
            == "completed_finite"
        )
    ].copy()

    if sub.empty:
        raise RuntimeError(
            f"No completed rows for {candidate}."
        )

    row = {
        "candidate": candidate,
        "runs": int(
            len(sub)
        ),
        "mission_count": int(
            sub[
                "held_out_mission"
            ].nunique()
        ),
        "selection_score_mean": float(
            sub[
                "selection_score"
            ].mean()
        ),
        "selection_score_std": float(
            sub[
                "selection_score"
            ].std(
                ddof=0
            )
        ),
        "parameter_count": int(
            round(
                sub[
                    "parameter_count"
                ].mean()
            )
        ),
        "best_epoch_median": float(
            np.median(
                sub[
                    "best_epoch_one_based"
                ].to_numpy(
                    dtype=float
                )
            )
        ),
    }

    for col in [
        "wind_U_RMSE_mps",
        "wind_V_RMSE_mps",
        "wind_vector_RMSE_mps",
        "wind_speed_RMSE_mps",
        "wind_direction_RMSE_deg",
        "residual_corr_U",
        "residual_corr_V",
        "residual_sign_accuracy_U",
        "residual_sign_accuracy_V",
        "pred_correction_RMS_U_mps",
        "pred_correction_RMS_V_mps",
    ]:
        vals = sub[
            col
        ].to_numpy(
            dtype=float
        )

        row[
            f"{col}_mean"
        ] = float(
            np.nanmean(
                vals
            )
        )
        row[
            f"{col}_std"
        ] = float(
            np.nanstd(
                vals,
                ddof=0,
            )
        )

    mission_scores = (
        sub.groupby(
            "held_out_mission"
        )[
            "selection_score"
        ]
        .mean()
    )

    row[
        "mission_wins_vs_BaseRidgeL15"
    ] = int(
        np.sum(
            mission_scores.to_numpy()
            < 1.0
        )
    )

    return row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pa-dir",
        type=Path,
        default=DEFAULT_PA_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
    )
    parser.add_argument(
        "--debug-fast",
        action="store_true",
        help=(
            "Smoke test: Antarctic holdout, one seed, "
            "L15-ResidualGRU only, <=8 epochs."
        ),
    )

    args = parser.parse_args()

    pa_dir = args.pa_dir
    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pa_selection = load_frozen_lookback(
        pa_dir
    )

    missions = {
        m: load_mission(
            pa_dir,
            m,
        )
        for m in DEVELOPMENT_MISSIONS
    }

    torch, nn = import_torch()
    configure_cuda(
        torch
    )

    if (
        args.force_cpu
        or not torch.cuda.is_available()
    ):
        device = torch.device(
            "cpu"
        )
    else:
        device = torch.device(
            "cuda"
        )

    active_holdouts = (
        ["Antarctic"]
        if args.debug_fast
        else DEVELOPMENT_MISSIONS
    )

    active_candidates = (
        {
            "L15-ResidualGRU": "gru"
        }
        if args.debug_fast
        else CANDIDATES
    )

    active_seeds = (
        [SCREENING_SEEDS[0]]
        if args.debug_fast
        else SCREENING_SEEDS
    )

    log("=" * 132)
    log(
        "12Q — L15 RIDGE-ANCHORED SAFE RESIDUAL "
        "DEVELOPMENT-ONLY LOMO"
    )
    log("=" * 132)
    log(
        f"12P-A dir       : {pa_dir}"
    )
    log(
        f"lookback        : {LOOKBACK_MIN} consecutive 1-min samples"
    )
    log(
        f"forecast target : exactly +{FORECAST_HORIZON_MIN} min"
    )
    log(
        f"candidates      : {list(active_candidates)}"
    )
    log(
        f"device          : {device}"
    )

    if device.type == "cuda":
        log(
            f"GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )

    log(
        "[SAFETY] Epoch 0 of every neural run is exact Base-Ridge-L15."
    )
    log(
        "[FIREWALL] Antarctic / Atlantic / West Coast only. "
        "Tropical Atlantic is not loaded."
    )
    log(
        "[CAVEAT] Same +10-min wall-clock horizon as BP-STGNN, "
        "not preprocessing-identical."
    )
    log("")

    run_path = (
        output_dir
        / "candidate_run_results.csv"
    )

    baseline_rows = []

    for held_out in (
        active_holdouts
    ):
        train_names = [
            m
            for m in DEVELOPMENT_MISSIONS
            if m != held_out
        ]

        train = concat_missions(
            [
                missions[m]
                for m in train_names
            ]
        )

        val = missions[
            held_out
        ]

        y_train = train[
            "y"
        ][:, 0, 0:2].astype(
            np.float32
        )

        y_val = val[
            "y"
        ][:, 0, 0:2].astype(
            np.float32
        )

        global_scaler = fit_global_scaler(
            train["X"],
            y_train,
        )

        ridge = fit_ridge_anchor(
            train["X"],
            y_train,
            global_scaler,
        )

        (
            ridge_train_raw,
            ridge_train_z,
        ) = ridge_predict_raw(
            ridge,
            train["X"],
            global_scaler,
        )

        (
            ridge_val_raw,
            ridge_val_z,
        ) = ridge_predict_raw(
            ridge,
            val["X"],
            global_scaler,
        )

        ridge_metrics = evaluate_wind(
            y_val,
            ridge_val_raw,
        )

        ridge_diag = residual_diagnostics(
            y_val,
            ridge_val_raw,
            ridge_val_raw,
        )

        ridge_row = {
            "candidate": "Base-Ridge-L15",
            "held_out_mission": held_out,
            "seed": -1,
            "status": "completed_finite",
            "parameter_count": int(
                ridge_parameter_count(
                    ridge
                )
            ),
            "best_epoch_one_based": 0,
            "selection_score": 1.0,
            "elapsed_seconds": 0.0,
            "residual_std_channel_0": np.nan,
            "residual_std_channel_1": np.nan,
            **ridge_metrics,
            **ridge_diag,
            "score_U_ratio": 1.0,
            "score_V_ratio": 1.0,
            "score_WS_ratio": 1.0,
            "score_WD_ratio": 1.0,
            "history_path": "",
            "checkpoint_path": "",
        }

        append_csv(
            run_path,
            ridge_row,
        )

        baseline_rows.append(
            ridge_row
        )

        log("-" * 132)
        log(
            f"[{held_out}] "
            f"Ntrain={len(train['X']):,} | "
            f"Nval={len(val['X']):,}"
        )
        log(
            "  Base-Ridge-L15 : "
            f"U={ridge_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={ridge_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={ridge_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={ridge_metrics['wind_direction_RMSE_deg']:.3f}"
        )

        for candidate, mode in (
            active_candidates.items()
        ):
            candidate_data = prepare_candidate_data(
                mode=mode,
                X_train=train["X"],
                X_val=val["X"],
                y_train=y_train,
                y_val=y_val,
                ridge_train_raw=(
                    ridge_train_raw
                ),
                ridge_val_raw=(
                    ridge_val_raw
                ),
                global_scaler=(
                    global_scaler
                ),
            )

            for seed in (
                active_seeds
            ):
                log(
                    f"  [TRAIN] {candidate} | seed={seed}"
                )

                result = train_one_run(
                    torch=torch,
                    nn=nn,
                    candidate=candidate,
                    mode=mode,
                    seed=seed,
                    held_out=held_out,
                    candidate_data=(
                        candidate_data
                    ),
                    y_val=y_val,
                    ridge_val_raw=(
                        ridge_val_raw
                    ),
                    ridge_val_metrics=(
                        ridge_metrics
                    ),
                    output_dir=(
                        output_dir
                    ),
                    device=device,
                    debug_fast=(
                        args.debug_fast
                    ),
                )

                append_csv(
                    run_path,
                    result,
                )

                log(
                    f"  [DONE] "
                    f"score={result['selection_score']:.5f} | "
                    f"best_ep={result['best_epoch_one_based']} | "
                    f"U={result['wind_U_RMSE_mps']:.4f} | "
                    f"V={result['wind_V_RMSE_mps']:.4f} | "
                    f"WS={result['wind_speed_RMSE_mps']:.4f} | "
                    f"WD={result['wind_direction_RMSE_deg']:.3f} | "
                    f"corr=({result['residual_corr_U']:+.3f},"
                    f"{result['residual_corr_V']:+.3f})"
                )

            del candidate_data
            gc.collect()

            if device.type == "cuda":
                torch.cuda.empty_cache()

        del (
            train,
            val,
            y_train,
            y_val,
            global_scaler,
            ridge,
            ridge_train_raw,
            ridge_train_z,
            ridge_val_raw,
            ridge_val_z,
        )

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.debug_fast:
        log("")
        log(
            "[DONE] debug-fast completed. "
            "No scientific final selection written."
        )
        return 0

    # =================================================================
    # Summary
    # =================================================================
    runs = pd.read_csv(
        run_path
    )

    expected = {
        "Base-Ridge-L15": 3,
        "L15-ResidualMLP": 9,
        "L15-ResidualGRU": 9,
        "L15-ResidualLocalGRU": 9,
    }

    for candidate, n_expected in (
        expected.items()
    ):
        got = int(
            np.sum(
                (
                    runs["candidate"].astype(str)
                    == candidate
                )
                & (
                    runs["status"].astype(str)
                    == "completed_finite"
                )
            )
        )

        if got != n_expected:
            raise RuntimeError(
                f"{candidate}: expected {n_expected} completed rows, got {got}."
            )

    candidate_order = [
        "Base-Ridge-L15",
        "L15-ResidualMLP",
        "L15-ResidualGRU",
        "L15-ResidualLocalGRU",
    ]

    summary = pd.DataFrame([
        summarize_candidate(
            runs,
            c,
        )
        for c in candidate_order
    ]).sort_values(
        [
            "selection_score_mean",
            "parameter_count",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    base_runs = runs.loc[
        runs["candidate"].astype(str)
        == "Base-Ridge-L15"
    ].copy()

    base_means = {
        k: float(
            base_runs[
                k
            ].mean()
        )
        for k in [
            "wind_U_RMSE_mps",
            "wind_V_RMSE_mps",
            "wind_vector_RMSE_mps",
            "wind_speed_RMSE_mps",
            "wind_direction_RMSE_deg",
        ]
    }

    for idx in summary.index:
        candidate = str(
            summary.loc[
                idx,
                "candidate",
            ]
        )

        sub = runs.loc[
            runs[
                "candidate"
            ].astype(str)
            == candidate
        ]

        for k, base_value in (
            base_means.items()
        ):
            candidate_value = float(
                sub[k].mean()
            )

            summary.loc[
                idx,
                f"{k}_improvement_vs_BaseRidge_fraction",
            ] = (
                base_value
                - candidate_value
            ) / max(
                base_value,
                EPS,
            )

    summary_path = (
        output_dir
        / "candidate_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    raw_best = summary.iloc[
        0
    ]

    raw_best_name = str(
        raw_best[
            "candidate"
        ]
    )

    raw_best_score = float(
        raw_best[
            "selection_score_mean"
        ]
    )

    raw_best_wins = int(
        raw_best[
            "mission_wins_vs_BaseRidgeL15"
        ]
    )

    raw_best_vector_imp = float(
        raw_best[
            "wind_vector_RMSE_mps_improvement_vs_BaseRidge_fraction"
        ]
    )

    raw_best_v_imp = float(
        raw_best[
            "wind_V_RMSE_mps_improvement_vs_BaseRidge_fraction"
        ]
    )

    raw_best_wd_imp = float(
        raw_best[
            "wind_direction_RMSE_deg_improvement_vs_BaseRidge_fraction"
        ]
    )

    raw_best_is_residual = (
        raw_best_name
        != "Base-Ridge-L15"
    )

    meets_rule = bool(
        raw_best_is_residual
        and raw_best_score
        < CONTINUE_SCORE_THRESHOLD
        and raw_best_wins
        >= MIN_MISSION_WINS
        and raw_best_vector_imp
        > MIN_VECTOR_IMPROVEMENT
        and (
            raw_best_v_imp
            > MIN_V_OR_WD_IMPROVEMENT
            or raw_best_wd_imp
            > MIN_V_OR_WD_IMPROVEMENT
        )
    )

    if meets_rule:
        selected_name = (
            raw_best_name
        )
        decision = (
            "CONTINUE_L15_SAFE_RESIDUAL"
        )
        reason = (
            f"{raw_best_name} clears the frozen rule: "
            f"score={raw_best_score:.6f}, "
            f"mission wins={raw_best_wins}/3, "
            f"vector improvement={100.0*raw_best_vector_imp:.3f}%, "
            f"V improvement={100.0*raw_best_v_imp:.3f}%, "
            f"WD improvement={100.0*raw_best_wd_imp:.3f}%."
        )
    else:
        selected_name = (
            "Base-Ridge-L15"
        )
        decision = (
            "RETAIN_BASE_RIDGE_L15"
        )
        reason = (
            f"Raw best {raw_best_name} does not clear every "
            "predeclared residual continuation requirement. "
            "Retain Base-Ridge-L15."
        )

    selected_runs = runs.loc[
        runs[
            "candidate"
        ].astype(str)
        == selected_name
    ].copy()

    selected_runs_path = (
        output_dir
        / "selected_candidate_lomo_runs.csv"
    )

    selected_runs.to_csv(
        selected_runs_path,
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "stage": "12Q",
        "script_version": SCRIPT_VERSION,
        "task": {
            "lookback_minutes": LOOKBACK_MIN,
            "raw_resolution_minutes": RAW_RESOLUTION_MIN,
            "forecast_horizon_minutes": FORECAST_HORIZON_MIN,
        },
        "candidates": CANDIDATES,
        "train_freeze": asdict(
            FREEZE
        ),
        "score_weights": SCORE_WEIGHTS,
        "predeclared_continuation_rule": {
            "score_lt": CONTINUE_SCORE_THRESHOLD,
            "mission_wins_ge": MIN_MISSION_WINS,
            "vector_improvement_gt": MIN_VECTOR_IMPROVEMENT,
            "V_or_WD_improvement_gt": MIN_V_OR_WD_IMPROVEMENT,
        },
        "raw_best_candidate": raw_best_name,
        "selected_candidate": selected_name,
        "decision": decision,
        "reason": reason,
        "development_only": True,
        "development_missions": DEVELOPMENT_MISSIONS,
        "Tropical_Atlantic_used": False,
        "published_BP_STGNN_reference_context_only": (
            BP_STGNN_PUBLISHED_REFERENCE
        ),
        "BP_STGNN_comparability_caveat": (
            "same +10-min wall-clock horizon, not preprocessing-identical"
        ),
        "scientific_note": (
            "Epoch 0 of every residual network is exact Ridge-L15. "
            "A residual model is retained only if it clears the predeclared "
            "development-only LOMO improvement rule."
        ),
    }

    selected_json_path = (
        output_dir
        / "selected_config.json"
    )

    save_json(
        selected_json_path,
        payload,
    )

    report_path = (
        output_dir
        / "12Q_REPORT.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "12Q L15 Ridge-Anchored Safe Residual "
            "(1-min history -> exact +10-min wind)\n"
        )
        f.write(
            "=" * 132
            + "\n\n"
        )

        f.write(
            "FROZEN TASK\n"
        )
        f.write(
            "-" * 132
            + "\n"
        )
        f.write(
            f"lookback: {LOOKBACK_MIN} consecutive 1-min samples\n"
        )
        f.write(
            f"forecast horizon: +{FORECAST_HORIZON_MIN} min exactly\n"
        )
        f.write(
            "Tropical Atlantic accessed: NO\n"
        )
        f.write(
            "Epoch 0 of every residual network = exact Base-Ridge-L15\n\n"
        )

        f.write(
            "CANDIDATE SUMMARY\n"
        )
        f.write(
            "-" * 132
            + "\n"
        )
        f.write(
            summary.to_string(
                index=False
            )
        )
        f.write(
            "\n\n"
        )

        f.write(
            "DECISION\n"
        )
        f.write(
            "-" * 132
            + "\n"
        )
        f.write(
            f"raw best: {raw_best_name}\n"
        )
        f.write(
            f"selected: {selected_name}\n"
        )
        f.write(
            f"decision: {decision}\n"
        )
        f.write(
            f"reason: {reason}\n\n"
        )

        f.write(
            "BP-STGNN REFERENCE — CONTEXT ONLY\n"
        )
        f.write(
            "-" * 132
            + "\n"
        )

        for k, v in (
            BP_STGNN_PUBLISHED_REFERENCE.items()
        ):
            f.write(
                f"{k}: {v}\n"
            )

        f.write(
            "\nCAVEAT: same +10-min wall-clock horizon, "
            "not preprocessing-identical to BP-STGNN.\n"
        )

    log("")
    log("=" * 132)
    log(
        "12Q CANDIDATE SUMMARY"
    )
    log("=" * 132)
    log(
        summary.to_string(
            index=False
        )
    )
    log("")
    log(
        f"[RAW BEST] {raw_best_name}"
    )
    log(
        f"[SELECTED] {selected_name}"
    )
    log(
        f"[DECISION] {decision}"
    )
    log(
        f"[REASON] {reason}"
    )
    log("")

    if raw_best_name != "Base-Ridge-L15":
        raw_best_runs = runs.loc[
            runs[
                "candidate"
            ].astype(str)
            == raw_best_name
        ]

        log(
            "Raw-best residual diagnostic means: "
            f"corr_U={raw_best_runs['residual_corr_U'].mean():+.4f} | "
            f"corr_V={raw_best_runs['residual_corr_V'].mean():+.4f} | "
            f"sign_U={raw_best_runs['residual_sign_accuracy_U'].mean():.3f} | "
            f"sign_V={raw_best_runs['residual_sign_accuracy_V'].mean():.3f}"
        )

    log("")
    log(
        "Published BP-STGNN reference (context only): "
        "U=0.74864 | V=0.70506 | WS=0.69940 | WD=5.584 deg"
    )
    log(
        "[CAVEAT] Same +10-min wall-clock horizon, "
        "not preprocessing-identical."
    )
    log("")
    log(
        f"[SAVED] {run_path}"
    )
    log(
        f"[SAVED] {summary_path}"
    )
    log(
        f"[SAVED] {selected_runs_path}"
    )
    log(
        f"[SAVED] {selected_json_path}"
    )
    log(
        f"[SAVED] {report_path}"
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
