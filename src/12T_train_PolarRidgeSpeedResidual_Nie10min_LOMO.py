# -*- coding: utf-8 -*-
r"""
12T_train_PolarRidgeSpeedResidual_Nie10min_LOMO.py

Stage 12T
=========
Development-only wind-SPEED residual experiment under the frozen Nie-aligned
forecasting protocol:

    resolution: 10 min
    history:    6 time steps = 60 min
    target:     immediately following +10-min wind

Scientific purpose
------------------
Stage 12R showed that a dedicated polar Ridge speed head predicts WS better
than obtaining WS indirectly from Cartesian U/V Ridge.

Stage 12S showed that a Local-Linear-Trend Kalman speed forecast is a poor
anchor under the 6x10-min protocol.

Therefore Stage 12T isolates ONE question:

    Is the remaining error of the strong Polar-Ridge speed head
    cross-mission predictable?

Anchor
------
A fold-specific Polar Ridge speed head is trained only on the two LOMO
training missions:

    [WS, sin(WD), cos(WD), T, RH] over 6 historical steps
        -> WS(t+10)

The direction Ridge head is also fit, but is FROZEN and used only to provide
secondary reconstructed U/V metrics. Direction is never changed by the
Stage-12T residual models.

Residual target
---------------
    r_WS = WS_true - WS_Ridge

The residual is normalized with its own fold-training standard deviation:

    r_norm = r_WS / std_train(r_WS)

Residual-model inputs
---------------------
At each of the same 6 historical 10-min time steps:

    [WS, dWS, T, RH]

where:
    dWS[0] = 0
    dWS[k] = WS[k] - WS[k-1]

The standardized Polar-Ridge speed anchor is concatenated after the sequence.

Candidates
----------
1) PolarSpeed-Ridge
   No residual correction. This is the frozen speed anchor.

2) Speed-ResidualMLP
   Flattened 6x4 sequence + Ridge speed anchor
       -> hidden 32 -> hidden 16 -> normalized scalar residual

3) Speed-ResidualGRU
   GRU hidden 32 over 6x4 sequence + Ridge speed anchor
       -> dense 32 -> normalized scalar residual

The final residual output layer is initialized to ZERO, so:

    epoch 0 == exact PolarSpeed-Ridge

Training
--------
Loss:
    normalized residual MSE
    + 1e-4 * normalized correction magnitude penalty

Checkpoint selection:
    held-out DEVELOPMENT mission WS RMSE only

This is intentional: Stage 12T is a speed-only mechanism experiment.

Secondary vector evaluation
---------------------------
For interpretation only, every speed prediction is combined with the SAME
frozen Polar-Ridge direction prediction:

    U_hat = -WS_hat * sin(WD_Ridge)
    V_hat = -WS_hat * cos(WD_Ridge)

Therefore any change in reconstructed U/V comes solely from changing speed.

Development-only LOMO
---------------------
Holdout Antarctic  <- train Atlantic + West Coast
Holdout Atlantic   <- train Antarctic + West Coast
Holdout West Coast <- train Antarctic + Atlantic

Tropical Atlantic is NEVER loaded.

Residual diagnostics
--------------------
For every selected checkpoint:

    corr(r_WS_true, r_WS_pred)
    residual sign accuracy
    true residual RMS
    predicted correction RMS

Also diagnose the true Ridge speed residual against simple historical
quantities on each held-out mission:

    corr(r_WS, dWS_last10)
    corr(r_WS, dWS_last20)
    corr(r_WS, slope_WS_6)
    corr(|r_WS|, std_WS_6)

These correlations are diagnostics only and are not used for training or
selection.

Predeclared continuation rule
-----------------------------
A learned speed residual is retained only if ALL are true:

    mean WS score < 0.99
        where score = candidate_WS_RMSE / PolarSpeedRidge_WS_RMSE

    >= 2/3 held-out mission wins

    no held-out mission has mean WS score > 1.01

    mean residual correction correlation > 0.10

Otherwise:
    RETAIN_POLAR_SPEED_RIDGE

A stronger residual-learning signal is separately flagged when:
    mean residual correction correlation >= 0.15

Published BP-STGNN reference
----------------------------
Context only, never used for development selection:
    WS RMSE = 0.699400 m/s
    U RMSE  = 0.748640 m/s
    V RMSE  = 0.705060 m/s
    WD RMSE = 5.584 deg

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\12T_train_PolarRidgeSpeedResidual_Nie10min_LOMO.py" --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12T_PolarRidgeSpeedResidual_Nie10min_LOMO_v0_1"

Smoke test:
    add --debug-fast
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


SCRIPT_VERSION = "0.1.0-12T-PolarRidge-SpeedResidual"

DEFAULT_PROJECT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")
DEFAULT_DATASET_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12G_Nie_point_sampled_benchmark_v0_1"
    / "dataset"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12T_PolarRidgeSpeedResidual_Nie10min_LOMO_v0_1"
)

DEVELOPMENT_MISSIONS = [
    "Antarctic",
    "Atlantic",
    "West Coast",
]

SCREENING_SEEDS = [
    500043,
    501052,
    502061,
]

LOOKBACK_STEPS = 6
STEP_MINUTES = 10
FORECAST_MINUTES = 10
TEN_MIN_NS = 10 * 60 * 1_000_000_000

FULL9_FEATURE_NAMES = [
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

RIDGE_ALPHA = 1.0
EPS = 1e-12
RESIDUAL_STD_FLOOR_MPS = 1e-4

CONTINUE_WS_SCORE_THRESHOLD = 0.99
MIN_MISSION_WINS = 2
MAX_SINGLE_MISSION_SCORE = 1.01
MIN_RESIDUAL_CORR = 0.10
STRONG_RESIDUAL_CORR = 0.15

BP_STGNN_PUBLISHED_REFERENCE = {
    "U_RMSE_mps": 0.748640,
    "V_RMSE_mps": 0.705060,
    "WS_RMSE_mps": 0.699400,
    "WD_RMSE_deg": 5.584000,
}


@dataclass(frozen=True)
class TrainFreeze:
    mlp_hidden_1: int = 32
    mlp_hidden_2: int = 16

    gru_hidden: int = 32
    dense_hidden: int = 32

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
    "Speed-ResidualMLP": "mlp",
    "Speed-ResidualGRU": "gru",
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


# =============================================================================
# Torch utilities
# =============================================================================

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
    if (
        not enabled
        or device.type != "cuda"
    ):
        return contextlib.nullcontext()

    try:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        )
    except Exception:
        return torch.cuda.amp.autocast(
            enabled=True
        )


def create_grad_scaler(torch, enabled: bool):
    if not enabled:
        return None

    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )
    except Exception:
        try:
            return torch.cuda.amp.GradScaler(
                enabled=True
            )
        except Exception:
            return None


def state_dict_cpu(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def count_parameters(model):
    return int(
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )


def sequential_batches(*arrays, batch_size: int):
    if not arrays:
        return

    n = len(arrays[0])

    if any(len(a) != n for a in arrays):
        raise ValueError(
            "Batch arrays have inconsistent first dimension."
        )

    for start in range(0, n, int(batch_size)):
        stop = min(start + int(batch_size), n)

        yield tuple(
            a[start:stop]
            for a in arrays
        )


# =============================================================================
# Frozen Track-J data
# =============================================================================

def load_track_j(dataset_dir: Path, mission: str):
    safe = mission.replace(" ", "_")

    path = (
        dataset_dir
        / "track_J_joint_compatible"
        / f"{safe}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(
        path,
        allow_pickle=False,
    ) as z:
        X = np.asarray(
            z["X_raw"],
            dtype=np.float32,
        )

        y = np.asarray(
            z["y_wind_raw"],
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

        names = [
            str(x)
            for x in z["feature_names"].tolist()
        ]

        lookback = (
            int(
                np.asarray(
                    z["lookback_steps"]
                ).reshape(-1)[0]
            )
            if "lookback_steps" in z.files
            else X.shape[1]
        )

        step_minutes = (
            int(
                np.asarray(
                    z["step_minutes"]
                ).reshape(-1)[0]
            )
            if "step_minutes" in z.files
            else STEP_MINUTES
        )

    if names != FULL9_FEATURE_NAMES:
        raise RuntimeError(
            f"{mission}: unexpected feature names {names}"
        )

    if (
        X.ndim != 3
        or X.shape[1:] != (
            LOOKBACK_STEPS,
            len(FULL9_FEATURE_NAMES),
        )
    ):
        raise RuntimeError(
            f"{mission}: bad X shape {X.shape}"
        )

    if y.ndim == 3:
        if y.shape[1:] != (1, 2):
            raise RuntimeError(
                f"{mission}: bad y shape {y.shape}"
            )
        y = y[:, 0, :]

    if (
        y.ndim != 2
        or y.shape[1] != 2
    ):
        raise RuntimeError(
            f"{mission}: bad y shape {y.shape}"
        )

    if lookback != LOOKBACK_STEPS:
        raise RuntimeError(
            f"{mission}: lookback={lookback}, expected {LOOKBACK_STEPS}"
        )

    if step_minutes != STEP_MINUTES:
        raise RuntimeError(
            f"{mission}: step_minutes={step_minutes}, expected {STEP_MINUTES}"
        )

    if not np.all(
        target - context
        == TEN_MIN_NS
    ):
        raise RuntimeError(
            f"{mission}: target is not exactly +10 min."
        )

    if not (
        np.isfinite(X).all()
        and np.isfinite(y).all()
    ):
        raise RuntimeError(
            f"{mission}: non-finite values."
        )

    return {
        "X_full9": X,
        "y_uv": y,
        "mission": mission,
    }


def concat_missions(items):
    return {
        "X_full9": np.concatenate(
            [x["X_full9"] for x in items],
            axis=0,
        ),
        "y_uv": np.concatenate(
            [x["y_uv"] for x in items],
            axis=0,
        ),
    }


# =============================================================================
# Wind geometry
# =============================================================================

def uv_to_ws_dirunit(uv):
    uv = np.asarray(
        uv,
        dtype=np.float64,
    )

    U = uv[..., 0]
    V = uv[..., 1]

    ws = np.hypot(U, V)

    # Meteorological FROM direction.
    theta = np.arctan2(
        -U,
        -V,
    )

    direction = np.stack(
        [
            np.sin(theta),
            np.cos(theta),
        ],
        axis=-1,
    )

    return (
        ws.astype(np.float32),
        direction.astype(np.float32),
    )


def reconstruct_uv(speed, direction_unit):
    speed = np.asarray(
        speed,
        dtype=np.float64,
    ).reshape(-1)

    d = np.asarray(
        direction_unit,
        dtype=np.float64,
    )

    U = -speed * d[:, 0]
    V = -speed * d[:, 1]

    return np.stack(
        [U, V],
        axis=1,
    ).astype(np.float32)


def normalize_direction_unit(raw, fallback):
    raw = np.asarray(
        raw,
        dtype=np.float64,
    )

    norm = np.linalg.norm(
        raw,
        axis=1,
    )

    bad = norm < 1e-8
    safe = np.maximum(
        norm,
        1e-8,
    )

    out = (
        raw
        / safe[:, None]
    )

    if np.any(bad):
        fb = np.asarray(
            fallback,
            dtype=np.float64,
        )

        fb_norm = np.maximum(
            np.linalg.norm(
                fb,
                axis=1,
            ),
            1e-8,
        )

        out[bad] = (
            fb[bad]
            / fb_norm[bad, None]
        )

    return (
        out.astype(np.float32),
        float(np.mean(bad)),
    )


def make_polar5(X_full9):
    X = np.asarray(
        X_full9,
        dtype=np.float32,
    )

    wind_uv = X[
        :,
        :,
        [
            IDX_U,
            IDX_V,
        ]
    ]

    ws, d = uv_to_ws_dirunit(
        wind_uv
    )

    T = X[:, :, IDX_T]
    RH = X[:, :, IDX_RH]

    return np.stack(
        [
            ws,
            d[:, :, 0],
            d[:, :, 1],
            T,
            RH,
        ],
        axis=-1,
    ).astype(np.float32)


def make_speed_residual_sequence(X_full9):
    """
    [WS, dWS, T, RH] at each of 6 historical steps.
    """
    X = np.asarray(
        X_full9,
        dtype=np.float32,
    )

    wind_uv = X[
        :,
        :,
        [
            IDX_U,
            IDX_V,
        ]
    ]

    ws, _ = uv_to_ws_dirunit(
        wind_uv
    )

    dws = np.zeros_like(
        ws,
        dtype=np.float32,
    )

    dws[:, 1:] = (
        ws[:, 1:]
        - ws[:, :-1]
    )

    T = X[:, :, IDX_T]
    RH = X[:, :, IDX_RH]

    seq = np.stack(
        [
            ws,
            dws,
            T,
            RH,
        ],
        axis=-1,
    ).astype(np.float32)

    return seq


# =============================================================================
# Fold-train scalers
# =============================================================================

def fit_x_scaler(X):
    x = np.asarray(
        X,
        dtype=np.float64,
    )

    mean = np.mean(
        x,
        axis=(0, 1),
    )

    std = np.std(
        x,
        axis=(0, 1),
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


def transform_X(X, scaler):
    return (
        (
            np.asarray(
                X,
                dtype=np.float32,
            )
            - scaler["mean"][None, None, :]
        )
        / scaler["std"][None, None, :]
    ).astype(np.float32)


def fit_scalar_scaler(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    ).reshape(-1)

    mean = float(
        np.mean(x)
    )

    std = float(
        np.std(
            x,
            ddof=0,
        )
    )

    if std < 1e-8:
        std = 1.0

    return {
        "mean": mean,
        "std": std,
    }


def transform_scalar(x, scaler):
    return (
        (
            np.asarray(
                x,
                dtype=np.float32,
            ).reshape(-1)
            - float(scaler["mean"])
        )
        / float(scaler["std"])
    ).astype(np.float32)


def inverse_scalar(xz, scaler):
    return (
        np.asarray(
            xz,
            dtype=np.float32,
        ).reshape(-1)
        * float(scaler["std"])
        + float(scaler["mean"])
    ).astype(np.float32)


# =============================================================================
# Polar Ridge anchor
# =============================================================================

def fit_polar_ridge_anchor(
    X_train_full9,
    y_train_uv,
):
    Xp = make_polar5(
        X_train_full9
    )

    xs = fit_x_scaler(
        Xp
    )

    Xz = transform_X(
        Xp,
        xs,
    ).reshape(
        len(Xp),
        -1,
    )

    ws_train, dir_train = (
        uv_to_ws_dirunit(
            y_train_uv
        )
    )

    ws_scaler = fit_scalar_scaler(
        ws_train
    )

    ws_train_z = transform_scalar(
        ws_train,
        ws_scaler,
    )

    speed_model = Ridge(
        alpha=RIDGE_ALPHA
    )

    direction_model = Ridge(
        alpha=RIDGE_ALPHA
    )

    speed_model.fit(
        Xz,
        ws_train_z,
    )

    direction_model.fit(
        Xz,
        dir_train,
    )

    return {
        "x_scaler": xs,
        "ws_scaler": ws_scaler,
        "speed_model": speed_model,
        "direction_model": direction_model,
    }


def polar_ridge_predict(
    anchor,
    X_full9,
):
    Xp = make_polar5(
        X_full9
    )

    Xz = transform_X(
        Xp,
        anchor["x_scaler"],
    ).reshape(
        len(Xp),
        -1,
    )

    ws_pred_z = anchor[
        "speed_model"
    ].predict(
        Xz
    )

    ws_pred_raw = inverse_scalar(
        ws_pred_z,
        anchor["ws_scaler"],
    )

    negative_speed_fraction = float(
        np.mean(
            ws_pred_raw < 0.0
        )
    )

    ws_pred = np.maximum(
        ws_pred_raw,
        0.0,
    ).astype(np.float32)

    direction_raw = np.asarray(
        anchor[
            "direction_model"
        ].predict(
            Xz
        ),
        dtype=np.float32,
    )

    _, last_dir = uv_to_ws_dirunit(
        X_full9[
            :,
            -1,
            [
                IDX_U,
                IDX_V,
            ]
        ]
    )

    direction_pred, fallback_fraction = (
        normalize_direction_unit(
            direction_raw,
            fallback=last_dir,
        )
    )

    return {
        "speed": ws_pred,
        "direction": direction_pred,
        "negative_speed_fraction_preclip": (
            negative_speed_fraction
        ),
        "direction_fallback_fraction": (
            fallback_fraction
        ),
    }


def polar_anchor_component_parameter_counts(anchor):
    speed_model = anchor["speed_model"]
    direction_model = anchor["direction_model"]

    speed_count = int(
        np.asarray(
            speed_model.coef_
        ).size
        + np.asarray(
            speed_model.intercept_
        ).size
    )

    direction_count = int(
        np.asarray(
            direction_model.coef_
        ).size
        + np.asarray(
            direction_model.intercept_
        ).size
    )

    return {
        "speed_anchor_parameter_count": speed_count,
        "direction_anchor_parameter_count": direction_count,
        "polar_anchor_total_parameter_count": (
            speed_count
            + direction_count
        ),
    }


def polar_anchor_parameter_count(anchor):
    return int(
        polar_anchor_component_parameter_counts(
            anchor
        )[
            "polar_anchor_total_parameter_count"
        ]
    )


# =============================================================================
# Metrics / diagnostics
# =============================================================================

def circular_diff_deg(a, b):
    return (
        (
            np.asarray(
                a,
                dtype=np.float64,
            )
            - np.asarray(
                b,
                dtype=np.float64,
            )
            + 180.0
        )
        % 360.0
        - 180.0
    )


def meteorological_wd_deg_from_uv(uv):
    uv = np.asarray(
        uv,
        dtype=np.float64,
    )

    theta = np.arctan2(
        -uv[:, 0],
        -uv[:, 1],
    )

    return (
        np.degrees(theta)
        % 360.0
    )


def evaluate_vector(y_true_uv, y_pred_uv):
    yt = np.asarray(
        y_true_uv,
        dtype=np.float64,
    )

    yp = np.asarray(
        y_pred_uv,
        dtype=np.float64,
    )

    err = yp - yt

    ws_true = np.linalg.norm(
        yt,
        axis=1,
    )

    ws_pred = np.linalg.norm(
        yp,
        axis=1,
    )

    wd_true = meteorological_wd_deg_from_uv(
        yt
    )

    wd_pred = meteorological_wd_deg_from_uv(
        yp
    )

    wd_err = circular_diff_deg(
        wd_pred,
        wd_true,
    )

    return {
        "wind_U_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    err[:, 0] ** 2
                )
            )
        ),
        "wind_V_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    err[:, 1] ** 2
                )
            )
        ),
        "wind_vector_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        err ** 2,
                        axis=1,
                    )
                )
            )
        ),
        "wind_speed_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    (
                        ws_pred
                        - ws_true
                    ) ** 2
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


def speed_rmse(ws_true, ws_pred):
    return float(
        np.sqrt(
            np.mean(
                (
                    np.asarray(
                        ws_pred,
                        dtype=np.float64,
                    )
                    - np.asarray(
                        ws_true,
                        dtype=np.float64,
                    )
                ) ** 2
            )
        )
    )


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
    ws_true,
    ws_anchor,
    ws_pred,
):
    ws_true = np.asarray(
        ws_true,
        dtype=np.float64,
    ).reshape(-1)

    ws_anchor = np.asarray(
        ws_anchor,
        dtype=np.float64,
    ).reshape(-1)

    ws_pred = np.asarray(
        ws_pred,
        dtype=np.float64,
    ).reshape(-1)

    r_true = (
        ws_true
        - ws_anchor
    )

    r_pred = (
        ws_pred
        - ws_anchor
    )

    return {
        "residual_corr_WS": (
            corrcoef_safe(
                r_true,
                r_pred,
            )
        ),
        "residual_sign_accuracy_WS": float(
            np.mean(
                np.sign(r_true)
                == np.sign(r_pred)
            )
        ),
        "true_residual_RMS_WS_mps": float(
            np.sqrt(
                np.mean(
                    r_true ** 2
                )
            )
        ),
        "pred_correction_RMS_WS_mps": float(
            np.sqrt(
                np.mean(
                    r_pred ** 2
                )
            )
        ),
        "pred_correction_bias_WS_mps": float(
            np.mean(
                r_pred
            )
        ),
    }


def linear_slope_per_sample(values):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    t = np.arange(
        x.shape[1],
        dtype=np.float64,
    )

    t0 = (
        t
        - np.mean(t)
    )

    denom = float(
        np.sum(
            t0 ** 2
        )
    )

    centered = (
        x
        - np.mean(
            x,
            axis=1,
            keepdims=True,
        )
    )

    return (
        centered
        @ t0
        / max(
            denom,
            EPS,
        )
    )


def true_residual_feature_diagnostics(
    X_full9,
    ws_true,
    ws_anchor,
):
    seq = make_speed_residual_sequence(
        X_full9
    )

    ws_hist = seq[:, :, 0]

    r = (
        np.asarray(
            ws_true,
            dtype=np.float64,
        ).reshape(-1)
        - np.asarray(
            ws_anchor,
            dtype=np.float64,
        ).reshape(-1)
    )

    dws_last10 = (
        ws_hist[:, -1]
        - ws_hist[:, -2]
    )

    dws_last20 = (
        ws_hist[:, -1]
        - ws_hist[:, -3]
    )

    slope6 = linear_slope_per_sample(
        ws_hist
    )

    std6 = np.std(
        ws_hist,
        axis=1,
        ddof=0,
    )

    return {
        "true_residual_corr_dWS_last10": (
            corrcoef_safe(
                r,
                dws_last10,
            )
        ),
        "true_residual_corr_dWS_last20": (
            corrcoef_safe(
                r,
                dws_last20,
            )
        ),
        "true_residual_corr_slopeWS6": (
            corrcoef_safe(
                r,
                slope6,
            )
        ),
        "abs_true_residual_corr_stdWS6": (
            corrcoef_safe(
                np.abs(r),
                std6,
            )
        ),
        "true_residual_mean_WS_mps": float(
            np.mean(r)
        ),
        "true_residual_std_WS_mps": float(
            np.std(
                r,
                ddof=0,
            )
        ),
    }


# =============================================================================
# Candidate data preparation
# =============================================================================

def prepare_residual_data(
    X_train_full9,
    X_val_full9,
    ws_train_true,
    ws_val_true,
    ws_train_anchor,
    ws_val_anchor,
):
    seq_train = make_speed_residual_sequence(
        X_train_full9
    )

    seq_val = make_speed_residual_sequence(
        X_val_full9
    )

    x_scaler = fit_x_scaler(
        seq_train
    )

    X_train_z = transform_X(
        seq_train,
        x_scaler,
    )

    X_val_z = transform_X(
        seq_val,
        x_scaler,
    )

    anchor_scaler = fit_scalar_scaler(
        ws_train_anchor
    )

    anchor_train_z = transform_scalar(
        ws_train_anchor,
        anchor_scaler,
    )

    anchor_val_z = transform_scalar(
        ws_val_anchor,
        anchor_scaler,
    )

    residual_train = (
        np.asarray(
            ws_train_true,
            dtype=np.float32,
        )
        - np.asarray(
            ws_train_anchor,
            dtype=np.float32,
        )
    )

    residual_std = float(
        np.std(
            residual_train.astype(
                np.float64
            ),
            ddof=0,
        )
    )

    if residual_std < RESIDUAL_STD_FLOOR_MPS:
        residual_std = 1.0

    target_norm = (
        residual_train
        / residual_std
    ).astype(np.float32)

    return {
        "X_train_z": X_train_z,
        "X_val_z": X_val_z,
        "anchor_train_z": anchor_train_z,
        "anchor_val_z": anchor_val_z,
        "target_norm": target_norm,
        "residual_std": float(
            residual_std
        ),
        "x_scaler": x_scaler,
        "anchor_scaler": anchor_scaler,
    }


# =============================================================================
# Neural residual models
# =============================================================================

def build_model_class(
    torch,
    nn,
    mode: str,
):
    input_dim = 4

    if mode == "mlp":
        class SpeedResidualMLP(nn.Module):
            def __init__(self):
                super().__init__()

                self.backbone = nn.Sequential(
                    nn.Linear(
                        LOOKBACK_STEPS
                        * input_dim
                        + 1,
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
                    1,
                )

                # Epoch 0 = exact PolarSpeed-Ridge.
                nn.init.zeros_(
                    self.out.weight
                )
                nn.init.zeros_(
                    self.out.bias
                )

            def forward(
                self,
                x,
                anchor_z,
            ):
                z = torch.cat(
                    [
                        x.flatten(
                            start_dim=1
                        ),
                        anchor_z.unsqueeze(
                            -1
                        ),
                    ],
                    dim=1,
                )

                return self.out(
                    self.backbone(z)
                ).squeeze(-1)

        return SpeedResidualMLP

    if mode == "gru":
        class SpeedResidualGRU(nn.Module):
            def __init__(self):
                super().__init__()

                self.gru = nn.GRU(
                    input_size=input_dim,
                    hidden_size=(
                        FREEZE.gru_hidden
                    ),
                    num_layers=1,
                    batch_first=True,
                )

                self.backbone = nn.Sequential(
                    nn.Linear(
                        FREEZE.gru_hidden
                        + 1,
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
                    1,
                )

                # Epoch 0 = exact PolarSpeed-Ridge.
                nn.init.zeros_(
                    self.out.weight
                )
                nn.init.zeros_(
                    self.out.bias
                )

            def forward(
                self,
                x,
                anchor_z,
            ):
                seq, _ = self.gru(x)

                h = seq[:, -1, :]

                z = torch.cat(
                    [
                        h,
                        anchor_z.unsqueeze(
                            -1
                        ),
                    ],
                    dim=1,
                )

                return self.out(
                    self.backbone(z)
                ).squeeze(-1)

        return SpeedResidualGRU

    raise ValueError(
        f"Unknown mode: {mode}"
    )


def reconstruct_speed(
    pred_norm,
    residual_std,
    ws_anchor,
):
    correction = (
        np.asarray(
            pred_norm,
            dtype=np.float32,
        ).reshape(-1)
        * float(
            residual_std
        )
    )

    raw = (
        np.asarray(
            ws_anchor,
            dtype=np.float32,
        ).reshape(-1)
        + correction
    )

    negative_fraction = float(
        np.mean(
            raw < 0.0
        )
    )

    pred = np.maximum(
        raw,
        0.0,
    ).astype(np.float32)

    return (
        pred,
        correction.astype(
            np.float32
        ),
        negative_fraction,
    )


def evaluate_model(
    *,
    torch,
    model,
    residual_data,
    ws_true,
    ws_anchor,
    fixed_direction,
    y_true_uv,
    device,
    amp_enabled,
):
    model.eval()

    chunks = []

    with torch.no_grad():
        for xb_np, ab_np in sequential_batches(
            residual_data[
                "X_val_z"
            ],
            residual_data[
                "anchor_val_z"
            ],
            batch_size=(
                FREEZE.batch_size
            ),
        ):
            xb = torch.from_numpy(
                xb_np
            ).to(
                device,
                non_blocking=True,
            )

            ab = torch.from_numpy(
                ab_np
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
                    ab,
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
    )

    (
        ws_pred,
        correction,
        negative_fraction,
    ) = reconstruct_speed(
        pred_norm,
        residual_data[
            "residual_std"
        ],
        ws_anchor,
    )

    uv_pred = reconstruct_uv(
        ws_pred,
        fixed_direction,
    )

    metrics = {
        "wind_speed_RMSE_mps": (
            speed_rmse(
                ws_true,
                ws_pred,
            )
        ),
        "negative_speed_fraction_preclip": (
            negative_fraction
        ),
    }

    metrics.update(
        residual_diagnostics(
            ws_true,
            ws_anchor,
            ws_pred,
        )
    )

    vector_metrics = evaluate_vector(
        y_true_uv,
        uv_pred,
    )

    # WS is identical to the dedicated speed metric, but keep one source.
    for key in [
        "wind_U_RMSE_mps",
        "wind_V_RMSE_mps",
        "wind_vector_RMSE_mps",
        "wind_direction_RMSE_deg",
    ]:
        metrics[key] = (
            vector_metrics[key]
        )

    return (
        metrics,
        pred_norm,
        ws_pred,
        uv_pred,
    )


def train_one_run(
    *,
    torch,
    nn,
    candidate,
    mode,
    seed,
    held_out,
    residual_data,
    ws_train_true,
    ws_val_true,
    ws_val_anchor,
    fixed_direction_val,
    y_val_uv,
    anchor_ws_rmse,
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

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=(
                FREEZE.scheduler_factor
            ),
            patience=(
                FREEZE.scheduler_patience
            ),
            min_lr=(
                FREEZE.min_learning_rate
            ),
        )
    )

    amp_enabled = bool(
        FREEZE.use_amp
        and device.type == "cuda"
    )

    grad_scaler = create_grad_scaler(
        torch,
        amp_enabled,
    )

    # Epoch 0 == exact PolarSpeed-Ridge.
    initial_metrics, _, _, _ = (
        evaluate_model(
            torch=torch,
            model=model,
            residual_data=(
                residual_data
            ),
            ws_true=(
                ws_val_true
            ),
            ws_anchor=(
                ws_val_anchor
            ),
            fixed_direction=(
                fixed_direction_val
            ),
            y_true_uv=(
                y_val_uv
            ),
            device=device,
            amp_enabled=(
                amp_enabled
            ),
        )
    )

    best_score = float(
        initial_metrics[
            "wind_speed_RMSE_mps"
        ]
        / max(
            anchor_ws_rmse,
            EPS,
        )
    )

    best_epoch = 0
    best_state = state_dict_cpu(
        model
    )
    best_metrics = dict(
        initial_metrics
    )

    history = [
        {
            "candidate": candidate,
            "held_out_mission": (
                held_out
            ),
            "seed": int(seed),
            "epoch_one_based": 0,
            "learning_rate": float(
                FREEZE.learning_rate
            ),
            "train_total_loss": np.nan,
            "train_residual_mse": np.nan,
            "train_correction_penalty": np.nan,
            "speed_score_vs_anchor": (
                best_score
            ),
            "checkpoint_improved": True,
            **{
                f"val_{k}": v
                for k, v
                in initial_metrics.items()
            },
        }
    ]

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

        total_sum = 0.0
        mse_sum = 0.0
        penalty_sum = 0.0
        n_seen = 0

        for (
            xb_np,
            ab_np,
            tb_np,
        ) in sequential_batches(
            residual_data[
                "X_train_z"
            ],
            residual_data[
                "anchor_train_z"
            ],
            residual_data[
                "target_norm"
            ],
            batch_size=(
                FREEZE.batch_size
            ),
        ):
            xb = torch.from_numpy(
                xb_np
            ).to(
                device,
                non_blocking=True,
            )

            ab = torch.from_numpy(
                ab_np
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
                    ab,
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

            if not torch.isfinite(total):
                raise FloatingPointError(
                    "Non-finite Stage-12T loss."
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

            bn = len(xb_np)

            total_sum += float(
                total.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            mse_sum += float(
                mse.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            penalty_sum += float(
                penalty.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            n_seen += bn

        val_metrics, _, _, _ = (
            evaluate_model(
                torch=torch,
                model=model,
                residual_data=(
                    residual_data
                ),
                ws_true=(
                    ws_val_true
                ),
                ws_anchor=(
                    ws_val_anchor
                ),
                fixed_direction=(
                    fixed_direction_val
                ),
                y_true_uv=(
                    y_val_uv
                ),
                device=device,
                amp_enabled=(
                    amp_enabled
                ),
            )
        )

        score = float(
            val_metrics[
                "wind_speed_RMSE_mps"
            ]
            / max(
                anchor_ws_rmse,
                EPS,
            )
        )

        improved = (
            score
            < best_score
            - 1e-8
        )

        if improved:
            best_score = score
            best_epoch = int(epoch)
            best_state = state_dict_cpu(
                model
            )
            best_metrics = dict(
                val_metrics
            )
            patience = 0
        else:
            patience += 1

        lr = float(
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        scheduler.step(
            score
        )

        history.append(
            {
                "candidate": candidate,
                "held_out_mission": (
                    held_out
                ),
                "seed": int(seed),
                "epoch_one_based": int(
                    epoch
                ),
                "learning_rate": lr,
                "train_total_loss": (
                    total_sum
                    / max(
                        n_seen,
                        1,
                    )
                ),
                "train_residual_mse": (
                    mse_sum
                    / max(
                        n_seen,
                        1,
                    )
                ),
                "train_correction_penalty": (
                    penalty_sum
                    / max(
                        n_seen,
                        1,
                    )
                ),
                "speed_score_vs_anchor": (
                    score
                ),
                "checkpoint_improved": (
                    bool(
                        improved
                    )
                ),
                **{
                    f"val_{k}": v
                    for k, v
                    in val_metrics.items()
                },
            }
        )

        log(
            f"      ep={epoch:03d} | "
            f"WSscore={score:.5f} | "
            f"WS={val_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"corr={val_metrics['residual_corr_WS']:+.3f} | "
            f"corrRMS={val_metrics['pred_correction_RMS_WS_mps']:.4f}"
            + (
                " *"
                if improved
                else ""
            )
        )

        if patience >= patience_limit:
            log(
                f"      early stop after {patience} epochs "
                "without WS-score improvement."
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

    confirmed, _, _, _ = (
        evaluate_model(
            torch=torch,
            model=model,
            residual_data=(
                residual_data
            ),
            ws_true=(
                ws_val_true
            ),
            ws_anchor=(
                ws_val_anchor
            ),
            fixed_direction=(
                fixed_direction_val
            ),
            y_true_uv=(
                y_val_uv
            ),
            device=device,
            amp_enabled=(
                amp_enabled
            ),
        )
    )

    confirmed_score = float(
        confirmed[
            "wind_speed_RMSE_mps"
        ]
        / max(
            anchor_ws_rmse,
            EPS,
        )
    )

    if abs(
        confirmed_score
        - best_score
    ) > 1e-6:
        raise RuntimeError(
            "Best Stage-12T checkpoint re-evaluation mismatch."
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

    torch.save(
        {
            "stage": "12T",
            "script_version": (
                SCRIPT_VERSION
            ),
            "candidate": candidate,
            "mode": mode,
            "held_out_mission": (
                held_out
            ),
            "seed": int(seed),
            "task": {
                "resolution_minutes": (
                    STEP_MINUTES
                ),
                "lookback_steps": (
                    LOOKBACK_STEPS
                ),
                "forecast_minutes": (
                    FORECAST_MINUTES
                ),
            },
            "parameter_count": int(
                parameter_count
            ),
            "best_epoch_one_based": int(
                best_epoch
            ),
            "speed_score_vs_anchor": float(
                confirmed_score
            ),
            "validation_metrics": (
                confirmed
            ),
            "residual_std_mps": float(
                residual_data[
                    "residual_std"
                ]
            ),
            "state_dict": (
                best_state
            ),
            "train_freeze": asdict(
                FREEZE
            ),
            "Tropical_Atlantic_used": False,
            "direction_branch_changed": False,
        },
        checkpoint_path,
    )

    result = {
        "candidate": candidate,
        "held_out_mission": (
            held_out
        ),
        "seed": int(seed),
        "status": (
            "completed_finite"
        ),
        "parameter_count": int(
            parameter_count
        ),
        "best_epoch_one_based": int(
            best_epoch
        ),
        "speed_score_vs_anchor": float(
            confirmed_score
        ),
        "elapsed_seconds": float(
            elapsed
        ),
        "residual_std_train_mps": float(
            residual_data[
                "residual_std"
            ]
        ),
        **confirmed,
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
# CSV / reporting
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
                    old[
                        "candidate"
                    ].astype(str)
                    == str(
                        row[
                            "candidate"
                        ]
                    )
                )
                & (
                    old[
                        "held_out_mission"
                    ].astype(str)
                    == str(
                        row[
                            "held_out_mission"
                        ]
                    )
                )
                & (
                    old[
                        "seed"
                    ].astype(int)
                    == int(
                        row[
                            "seed"
                        ]
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
            runs[
                "candidate"
            ].astype(str)
            == candidate
        )
        & (
            runs[
                "status"
            ].astype(str)
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
        "speed_score_mean": float(
            sub[
                "speed_score_vs_anchor"
            ].mean()
        ),
        "speed_score_std": float(
            sub[
                "speed_score_vs_anchor"
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
        "effective_total_parameter_count": int(
            round(
                sub[
                    "effective_total_parameter_count"
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
        "wind_speed_RMSE_mps",
        "wind_U_RMSE_mps",
        "wind_V_RMSE_mps",
        "wind_vector_RMSE_mps",
        "wind_direction_RMSE_deg",
        "residual_corr_WS",
        "residual_sign_accuracy_WS",
        "true_residual_RMS_WS_mps",
        "pred_correction_RMS_WS_mps",
        "pred_correction_bias_WS_mps",
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
            "speed_score_vs_anchor"
        ]
        .mean()
    )

    row[
        "mission_wins_vs_PolarSpeedRidge"
    ] = int(
        np.sum(
            mission_scores.to_numpy()
            < 1.0
        )
    )

    row[
        "max_mission_speed_score"
    ] = float(
        np.max(
            mission_scores.to_numpy()
        )
    )

    return row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=(
            DEFAULT_DATASET_DIR
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            DEFAULT_OUTPUT_DIR
        ),
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
            "Speed-ResidualGRU only, <=8 epochs."
        ),
    )

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    missions = {
        m: load_track_j(
            dataset_dir,
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
        [
            "Antarctic"
        ]
        if args.debug_fast
        else DEVELOPMENT_MISSIONS
    )

    active_candidates = (
        {
            "Speed-ResidualGRU": (
                "gru"
            )
        }
        if args.debug_fast
        else CANDIDATES
    )

    active_seeds = (
        [
            SCREENING_SEEDS[
                0
            ]
        ]
        if args.debug_fast
        else SCREENING_SEEDS
    )

    log("=" * 132)
    log(
        "12T — POLAR-RIDGE WIND-SPEED RESIDUAL SPECIALIST"
    )
    log("=" * 132)
    log(
        f"dataset         : {dataset_dir}"
    )
    log(
        f"output          : {output_dir}"
    )
    log(
        "task            : 6 x 10-min history -> next +10-min wind"
    )
    log(
        "anchor          : dedicated Polar-Ridge speed head"
    )
    log(
        "residual inputs : [WS,dWS,T,RH] x 6 + Ridge speed anchor"
    )
    log(
        "direction       : frozen Polar-Ridge direction; not trained in 12T"
    )
    log(
        f"device          : {device}"
    )

    if device.type == "cuda":
        log(
            f"GPU             : {torch.cuda.get_device_name(0)}"
        )

    log(
        "[SAFETY] Epoch 0 of every neural run is exact PolarSpeed-Ridge."
    )
    log(
        "[FIREWALL] Antarctic / Atlantic / West Coast only. "
        "Tropical Atlantic is not loaded."
    )
    log("")

    run_path = (
        output_dir
        / "candidate_run_results.csv"
    )

    diagnostic_rows = []

    for held_out in active_holdouts:
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

        Xtr = train[
            "X_full9"
        ]
        Xva = val[
            "X_full9"
        ]

        ytr = train[
            "y_uv"
        ]
        yva = val[
            "y_uv"
        ]

        ws_train_true, _ = (
            uv_to_ws_dirunit(
                ytr
            )
        )

        ws_val_true, _ = (
            uv_to_ws_dirunit(
                yva
            )
        )

        anchor = fit_polar_ridge_anchor(
            Xtr,
            ytr,
        )

        anchor_param_counts = (
            polar_anchor_component_parameter_counts(
                anchor
            )
        )

        train_anchor = polar_ridge_predict(
            anchor,
            Xtr,
        )

        val_anchor = polar_ridge_predict(
            anchor,
            Xva,
        )

        ws_train_anchor = train_anchor[
            "speed"
        ]

        ws_val_anchor = val_anchor[
            "speed"
        ]

        fixed_dir_val = val_anchor[
            "direction"
        ]

        anchor_uv_val = reconstruct_uv(
            ws_val_anchor,
            fixed_dir_val,
        )

        anchor_vector_metrics = (
            evaluate_vector(
                yva,
                anchor_uv_val,
            )
        )

        anchor_ws_rmse = (
            anchor_vector_metrics[
                "wind_speed_RMSE_mps"
            ]
        )

        anchor_diag = (
            residual_diagnostics(
                ws_val_true,
                ws_val_anchor,
                ws_val_anchor,
            )
        )

        true_feature_diag = (
            true_residual_feature_diagnostics(
                Xva,
                ws_val_true,
                ws_val_anchor,
            )
        )

        diagnostic_rows.append(
            {
                "held_out_mission": (
                    held_out
                ),
                "anchor_WS_RMSE_mps": (
                    anchor_ws_rmse
                ),
                **true_feature_diag,
            }
        )

        anchor_row = {
            "candidate": (
                "PolarSpeed-Ridge"
            ),
            "held_out_mission": (
                held_out
            ),
            "seed": -1,
            "status": (
                "completed_finite"
            ),
            "parameter_count": int(
                anchor_param_counts[
                    "polar_anchor_total_parameter_count"
                ]
            ),
            "speed_anchor_parameter_count": int(
                anchor_param_counts[
                    "speed_anchor_parameter_count"
                ]
            ),
            "direction_anchor_parameter_count": int(
                anchor_param_counts[
                    "direction_anchor_parameter_count"
                ]
            ),
            "polar_anchor_total_parameter_count": int(
                anchor_param_counts[
                    "polar_anchor_total_parameter_count"
                ]
            ),
            "effective_total_parameter_count": int(
                anchor_param_counts[
                    "polar_anchor_total_parameter_count"
                ]
            ),
            "best_epoch_one_based": 0,
            "speed_score_vs_anchor": 1.0,
            "elapsed_seconds": 0.0,
            "residual_std_train_mps": float(
                np.std(
                    (
                        ws_train_true
                        - ws_train_anchor
                    ).astype(
                        np.float64
                    ),
                    ddof=0,
                )
            ),
            **anchor_vector_metrics,
            **anchor_diag,
            "history_path": "",
            "checkpoint_path": "",
        }

        append_csv(
            run_path,
            anchor_row,
        )

        residual_data = prepare_residual_data(
            X_train_full9=Xtr,
            X_val_full9=Xva,
            ws_train_true=(
                ws_train_true
            ),
            ws_val_true=(
                ws_val_true
            ),
            ws_train_anchor=(
                ws_train_anchor
            ),
            ws_val_anchor=(
                ws_val_anchor
            ),
        )

        log("-" * 132)
        log(
            f"[{held_out}] "
            f"Ntrain={len(Xtr):,} | "
            f"Nval={len(Xva):,} | "
            f"Anchor WS={anchor_ws_rmse:.4f} | "
            f"resStd={residual_data['residual_std']:.4f}"
        )

        log(
            "  True residual diagnostics: "
            f"corr(dWS10)={true_feature_diag['true_residual_corr_dWS_last10']:+.3f} | "
            f"corr(dWS20)={true_feature_diag['true_residual_corr_dWS_last20']:+.3f} | "
            f"corr(slope6)={true_feature_diag['true_residual_corr_slopeWS6']:+.3f} | "
            f"corr(|r|,std6)={true_feature_diag['abs_true_residual_corr_stdWS6']:+.3f}"
        )

        for (
            candidate,
            mode,
        ) in active_candidates.items():
            for seed in active_seeds:
                log(
                    f"  [TRAIN] {candidate} | seed={seed}"
                )

                result = train_one_run(
                    torch=torch,
                    nn=nn,
                    candidate=(
                        candidate
                    ),
                    mode=(
                        mode
                    ),
                    seed=(
                        seed
                    ),
                    held_out=(
                        held_out
                    ),
                    residual_data=(
                        residual_data
                    ),
                    ws_train_true=(
                        ws_train_true
                    ),
                    ws_val_true=(
                        ws_val_true
                    ),
                    ws_val_anchor=(
                        ws_val_anchor
                    ),
                    fixed_direction_val=(
                        fixed_dir_val
                    ),
                    y_val_uv=(
                        yva
                    ),
                    anchor_ws_rmse=(
                        anchor_ws_rmse
                    ),
                    output_dir=(
                        output_dir
                    ),
                    device=(
                        device
                    ),
                    debug_fast=(
                        args.debug_fast
                    ),
                )

                result[
                    "speed_anchor_parameter_count"
                ] = int(
                    anchor_param_counts[
                        "speed_anchor_parameter_count"
                    ]
                )

                result[
                    "direction_anchor_parameter_count"
                ] = int(
                    anchor_param_counts[
                        "direction_anchor_parameter_count"
                    ]
                )

                result[
                    "polar_anchor_total_parameter_count"
                ] = int(
                    anchor_param_counts[
                        "polar_anchor_total_parameter_count"
                    ]
                )

                result[
                    "effective_total_parameter_count"
                ] = int(
                    result[
                        "parameter_count"
                    ]
                    + anchor_param_counts[
                        "polar_anchor_total_parameter_count"
                    ]
                )

                append_csv(
                    run_path,
                    result,
                )

                log(
                    f"  [DONE] "
                    f"WSscore={result['speed_score_vs_anchor']:.5f} | "
                    f"best_ep={result['best_epoch_one_based']} | "
                    f"WS={result['wind_speed_RMSE_mps']:.4f} | "
                    f"corr={result['residual_corr_WS']:+.3f} | "
                    f"sign={result['residual_sign_accuracy_WS']:.3f}"
                )

        del (
            train,
            val,
            Xtr,
            Xva,
            ytr,
            yva,
            anchor,
            train_anchor,
            val_anchor,
            residual_data,
        )

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    diagnostic_df = pd.DataFrame(
        diagnostic_rows
    )

    diagnostic_path = (
        output_dir
        / "anchor_residual_feature_diagnostics.csv"
    )

    diagnostic_df.to_csv(
        diagnostic_path,
        index=False,
        encoding="utf-8-sig",
    )

    if args.debug_fast:
        log("")
        log(
            "[DONE] debug-fast completed. "
            "No scientific final selection written."
        )
        return 0

    runs = pd.read_csv(
        run_path
    )

    expected_counts = {
        "PolarSpeed-Ridge": 3,
        "Speed-ResidualMLP": 9,
        "Speed-ResidualGRU": 9,
    }

    for candidate, expected in expected_counts.items():
        got = int(
            np.sum(
                (
                    runs[
                        "candidate"
                    ].astype(str)
                    == candidate
                )
                & (
                    runs[
                        "status"
                    ].astype(str)
                    == "completed_finite"
                )
            )
        )

        if got != expected:
            raise RuntimeError(
                f"{candidate}: expected {expected} completed rows, got {got}."
            )

    candidate_order = [
        "PolarSpeed-Ridge",
        "Speed-ResidualMLP",
        "Speed-ResidualGRU",
    ]

    summary = pd.DataFrame(
        [
            summarize_candidate(
                runs,
                c,
            )
            for c in candidate_order
        ]
    ).sort_values(
        [
            "speed_score_mean",
            "parameter_count",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    anchor_runs = runs.loc[
        runs[
            "candidate"
        ].astype(str)
        == "PolarSpeed-Ridge"
    ].copy()

    anchor_ws_mean = float(
        anchor_runs[
            "wind_speed_RMSE_mps"
        ].mean()
    )

    for idx in summary.index:
        ws_mean = float(
            summary.loc[
                idx,
                "wind_speed_RMSE_mps_mean",
            ]
        )

        summary.loc[
            idx,
            "wind_speed_RMSE_improvement_vs_anchor_fraction",
        ] = (
            anchor_ws_mean
            - ws_mean
        ) / max(
            anchor_ws_mean,
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

    raw_best = summary.iloc[0]

    raw_best_name = str(
        raw_best[
            "candidate"
        ]
    )

    raw_best_score = float(
        raw_best[
            "speed_score_mean"
        ]
    )

    raw_best_wins = int(
        raw_best[
            "mission_wins_vs_PolarSpeedRidge"
        ]
    )

    raw_best_max_mission_score = float(
        raw_best[
            "max_mission_speed_score"
        ]
    )

    raw_best_corr = float(
        raw_best[
            "residual_corr_WS_mean"
        ]
    )

    is_residual = (
        raw_best_name
        != "PolarSpeed-Ridge"
    )

    meets_rule = bool(
        is_residual
        and raw_best_score
        < CONTINUE_WS_SCORE_THRESHOLD
        and raw_best_wins
        >= MIN_MISSION_WINS
        and raw_best_max_mission_score
        <= MAX_SINGLE_MISSION_SCORE
        and raw_best_corr
        > MIN_RESIDUAL_CORR
    )

    strong_signal = bool(
        is_residual
        and raw_best_corr
        >= STRONG_RESIDUAL_CORR
    )

    if meets_rule:
        selected_name = (
            raw_best_name
        )

        decision = (
            "CONTINUE_SPEED_RESIDUAL"
        )

        reason = (
            f"{raw_best_name} clears the frozen speed-residual rule: "
            f"mean score={raw_best_score:.6f}, "
            f"mission wins={raw_best_wins}/3, "
            f"max mission score={raw_best_max_mission_score:.6f}, "
            f"mean residual corr={raw_best_corr:+.4f}."
        )
    else:
        selected_name = (
            "PolarSpeed-Ridge"
        )

        decision = (
            "RETAIN_POLAR_SPEED_RIDGE"
        )

        reason = (
            f"Raw best {raw_best_name} does not clear every predeclared "
            "speed-residual requirement. Retain PolarSpeed-Ridge."
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

    diag_mean = {
        col: float(
            diagnostic_df[
                col
            ].mean()
        )
        for col in [
            "true_residual_corr_dWS_last10",
            "true_residual_corr_dWS_last20",
            "true_residual_corr_slopeWS6",
            "abs_true_residual_corr_stdWS6",
        ]
    }

    if decision == "CONTINUE_SPEED_RESIDUAL":
        next_stage = (
            "FREEZE_SPEED_RESIDUAL_THEN_BUILD_DIRECTION_SPECIALIST"
        )
    else:
        strongest_diag_name = max(
            diag_mean,
            key=lambda k: (
                abs(
                    diag_mean[k]
                )
                if np.isfinite(
                    diag_mean[k]
                )
                else -1.0
            ),
        )

        strongest_diag_value = float(
            diag_mean[
                strongest_diag_name
            ]
        )

        if (
            np.isfinite(
                strongest_diag_value
            )
            and abs(
                strongest_diag_value
            )
            >= 0.15
        ):
            next_stage = (
                "USE_RESIDUAL_DIAGNOSTIC_TO_DESIGN_SPEED_REGIME_OR_TREND_MODEL"
            )
        else:
            next_stage = (
                "STOP_GENERIC_SPEED_RESIDUAL_AND_ADD_NEW_INFORMATION"
            )

    payload = {
        "stage": "12T",
        "script_version": (
            SCRIPT_VERSION
        ),
        "task": {
            "resolution_minutes": (
                STEP_MINUTES
            ),
            "lookback_steps": (
                LOOKBACK_STEPS
            ),
            "forecast_minutes": (
                FORECAST_MINUTES
            ),
        },
        "anchor": (
            "Polar-Ridge dedicated speed head"
        ),
        "residual_inputs": (
            "[WS,dWS,T,RH] x 6 + standardized Ridge speed anchor"
        ),
        "direction_branch": (
            "frozen Polar-Ridge direction head; unchanged by Stage 12T"
        ),
        "train_freeze": asdict(
            FREEZE
        ),
        "predeclared_rule": {
            "mean_speed_score_lt": (
                CONTINUE_WS_SCORE_THRESHOLD
            ),
            "mission_wins_ge": (
                MIN_MISSION_WINS
            ),
            "max_single_mission_score_le": (
                MAX_SINGLE_MISSION_SCORE
            ),
            "mean_residual_corr_gt": (
                MIN_RESIDUAL_CORR
            ),
            "strong_residual_corr_ge": (
                STRONG_RESIDUAL_CORR
            ),
        },
        "raw_best_candidate": (
            raw_best_name
        ),
        "selected_candidate": (
            selected_name
        ),
        "decision": decision,
        "reason": reason,
        "strong_residual_learning_signal": (
            strong_signal
        ),
        "anchor_residual_feature_diagnostic_means": (
            diag_mean
        ),
        "next_stage_recommendation": (
            next_stage
        ),
        "development_only": True,
        "development_missions": (
            DEVELOPMENT_MISSIONS
        ),
        "Tropical_Atlantic_used": False,
        "published_BP_STGNN_reference_context_only": (
            BP_STGNN_PUBLISHED_REFERENCE
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
        / "12T_REPORT.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "12T Polar-Ridge Wind-Speed Residual Specialist\n"
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
            "resolution: 10 min\n"
        )
        f.write(
            "history: 6 points = 60 min\n"
        )
        f.write(
            "target: next +10-min wind speed\n"
        )
        f.write(
            "anchor: Polar-Ridge dedicated speed head\n"
        )
        f.write(
            "direction branch changed: NO\n"
        )
        f.write(
            "Tropical Atlantic accessed: NO\n\n"
        )

        f.write(
            "ANCHOR TRUE-RESIDUAL FEATURE DIAGNOSTICS\n"
        )
        f.write(
            "-" * 132
            + "\n"
        )
        f.write(
            diagnostic_df.to_string(
                index=False
            )
        )
        f.write(
            "\n\n"
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
            f"reason: {reason}\n"
        )
        f.write(
            f"strong residual-learning signal: {strong_signal}\n"
        )
        f.write(
            f"NEXT_STAGE_RECOMMENDATION: {next_stage}\n\n"
        )

        f.write(
            "Published BP-STGNN reference values are context only:\n"
        )

        for k, v in (
            BP_STGNN_PUBLISHED_REFERENCE.items()
        ):
            f.write(
                f"{k}: {v}\n"
            )

    log("")
    log("=" * 132)
    log(
        "12T CANDIDATE SUMMARY"
    )
    log("=" * 132)
    log(
        summary.to_string(
            index=False
        )
    )
    log("")
    log(
        "12T ANCHOR TRUE-RESIDUAL FEATURE DIAGNOSTIC MEANS"
    )
    log("=" * 132)

    for k, v in diag_mean.items():
        log(
            f"{k}={v:+.4f}"
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
    log(
        f"[STRONG RESIDUAL SIGNAL] {strong_signal}"
    )
    log(
        f"[NEXT STAGE] {next_stage}"
    )
    log("")
    log(
        "Published BP-STGNN reference (context only): "
        "WS=0.69940 | U=0.74864 | V=0.70506 | WD=5.584 deg"
    )
    log("")
    log(
        f"[SAVED] {run_path}"
    )
    log(
        f"[SAVED] {summary_path}"
    )
    log(
        f"[SAVED] {diagnostic_path}"
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
