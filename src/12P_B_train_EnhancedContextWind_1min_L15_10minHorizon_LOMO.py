# -*- coding: utf-8 -*-
r"""
12P_B_train_EnhancedContextWind_1min_L15_10minHorizon_LOMO.py

Stage 12P-B
===========
Development-only enhanced-context wind forecasting after Stage 12P-A selected
a 15-minute ONE-MINUTE historical window for prediction exactly +10 minutes
after the context end.

Stage 12P-A result to be frozen
-------------------------------
The selected lookback is read from:
    <12P-A-dir>/selected_lookback.json

This script refuses to run if the selected lookback is not 15 min unless the
user explicitly supplies --allow-non15-selected-lookback.

The Stage-12P-A common sample dataset is reused:
    <12P-A-dir>/dataset_1min_common60/

Therefore:
    - raw resolution remains 1 min;
    - every candidate uses the same target timestamps;
    - target remains exactly context_end + 10 min;
    - Antarctic / Atlantic / West Coast are the only missions accessed;
    - Tropical Atlantic remains untouched.

Scientific question
-------------------
Can physically interpretable short-term tendency / turning / variability
features extract additional predictive information beyond the already improved
15-min Ridge baseline?

Base sequence features (15 x 9)
-------------------------------
    [U, V, T, RH, SOG,
     COG_sin, COG_cos,
     WING_ANGLE_sin, WING_ANGLE_cos]

Engineered summary features
---------------------------
All are computed ONLY from the historical 15-min context.

Wind tendency:
    dU_5, dV_5, dWS_5
    dU_10, dV_10, dWS_10
    slope_U_15, slope_V_15, slope_WS_15

Wind-direction turning:
    confidence-weighted sin(turn_5), 1-cos(turn_5)
    confidence-weighted sin(turn_10), 1-cos(turn_10)
    confidence-weighted angular_slope_15

Wind variability:
    std_U_5, std_V_5, std_WS_5
    std_U_15, std_V_15, std_WS_15

Vessel dynamics:
    dSOG_5, dSOG_10, slope_SOG_15
    sin(COG_turn_5), 1-cos(COG_turn_5)
    sin(COG_turn_10), 1-cos(COG_turn_10)
    COG_angular_slope_15

Wing dynamics:
    sin(WING_turn_5), 1-cos(WING_turn_5)
    sin(WING_turn_10), 1-cos(WING_turn_10)
    WING_angular_slope_15

Observed apparent-wind interaction features:
    dAWS_5, dAWS_10
    confidence-weighted sin(AW_turn_5), 1-cos(AW_turn_5)
    confidence-weighted sin(AW_turn_10), 1-cos(AW_turn_10)

Here apparent wind at historical times is deterministically reconstructed as:
    AW = true_wind - vessel_velocity
using observed historical wind, SOG and COG. No future information is used.

Candidates
----------
1) Base-Ridge-L15
   Exact 12P-A 15-min Ridge reference.

2) Enhanced-Ridge-L15
   Ridge on [flattened standardized 15x9 sequence,
             standardized engineered summaries].

3) Enhanced-MLP-L15
   Compact MLP on the same flattened sequence + summaries.

4) Enhanced-GRU-L15
   Compact GRU sequence encoder + summary vector.

Targets
-------
Direct future wind:
    [U(t+10), V(t+10)]

All reported U/V/WS/WD metrics are computed after inverse standardization in
the original East/North coordinates.

Development-only LOMO
---------------------
Holdout Antarctic  <- train Atlantic + West Coast
Holdout Atlantic   <- train Antarctic + West Coast
Holdout West Coast <- train Antarctic + Atlantic

Predeclared model-selection score
---------------------------------
Within each fold, Base-Ridge-L15 = 1.0:

    Score =
        0.25 * U_RMSE  / BaseRidge_U
      + 0.35 * V_RMSE  / BaseRidge_V
      + 0.25 * WS_RMSE / BaseRidge_WS
      + 0.15 * WD_RMSE / BaseRidge_WD

An enhanced candidate replaces Base-Ridge-L15 only if:
    mean score < 0.99
    AND >= 2 of 3 held-out missions have mean score < 1.0
    AND mean vector RMSE improves > 0.5%
    AND at least one of:
            V RMSE improves > 1%
            WD RMSE improves > 1%

Otherwise retain Base-Ridge-L15.

Published BP-STGNN values
-------------------------
The published BP-STGNN values are saved for context only:
    U  = 0.74864 m/s
    V  = 0.70506 m/s
    WS = 0.69940 m/s
    WD = 5.584 deg

They are NEVER used for development selection.

Comparability caveat
--------------------
This enhanced experiment uses the same +10-min wall-clock horizon but is NOT
preprocessing-identical to BP-STGNN, because the published BP-STGNN first
resamples the original 1-min data to a 10-min modeling interval.

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\12P_B_train_EnhancedContextWind_1min_L15_10minHorizon_LOMO.py" --pa-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12P_A_1minLookback_10minHorizon_Ridge_LOMO_v0_1" --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12P_B_EnhancedContextWind_1min_L15_10minHorizon_LOMO_v0_1"

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


SCRIPT_VERSION = "0.1.0-12P-B-EnhancedContext-L15-1min-10minHorizon"

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
    / "12P_B_EnhancedContextWind_1min_L15_10minHorizon_LOMO_v0_1"
)

DEVELOPMENT_MISSIONS = ["Antarctic", "Atlantic", "West Coast"]
SCREENING_SEEDS = [500043, 501052, 502061]

FROZEN_EXPECTED_LOOKBACK_MIN = 15
FORECAST_HORIZON_MIN = 10
RAW_RESOLUTION_MIN = 1
RIDGE_ALPHA = 1.0

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

EPS = 1e-12

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

TURN_CONFIDENCE_SPEED_SCALE_MPS = 5.0

BP_STGNN_PUBLISHED_REFERENCE = {
    "U_RMSE_mps": 0.74864,
    "V_RMSE_mps": 0.70506,
    "WS_RMSE_mps": 0.69940,
    "WD_RMSE_deg": 5.58400,
}


@dataclass(frozen=True)
class TrainFreeze:
    mlp_hidden_1: int = 96
    mlp_hidden_2: int = 48

    gru_hidden: int = 48
    dense_hidden: int = 64

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

    use_amp: bool = True


FREEZE = TrainFreeze()


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
# Dataset loading / scientific audit
# =============================================================================

def load_frozen_selected_lookback(pa_dir: Path):
    path = pa_dir / "selected_lookback.json"

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)

    selected = int(
        d["selected_lookback_minutes"]
    )

    horizon = int(
        d["task"]["forecast_horizon_minutes"]
    )

    if horizon != FORECAST_HORIZON_MIN:
        raise RuntimeError(
            f"12P-A horizon mismatch: expected {FORECAST_HORIZON_MIN}, "
            f"found {horizon}"
        )

    if not bool(
        d.get("development_only", False)
    ):
        raise RuntimeError(
            "12P-A selected_lookback.json does not declare development_only."
        )

    if bool(
        d.get("Tropical_Atlantic_loaded", True)
    ):
        raise RuntimeError(
            "12P-A selection file indicates Tropical Atlantic was loaded."
        )

    return selected, d


def load_mission(pa_dir: Path, mission: str, lookback: int):
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
        horizon = int(
            np.asarray(
                z["forecast_horizon_minutes"]
            ).reshape(-1)[0]
        )
        raw_step = int(
            np.asarray(
                z["raw_step_minutes"]
            ).reshape(-1)[0]
        )

    if feature_names != FEATURE_NAMES:
        raise RuntimeError(
            f"{path.name}: feature schema mismatch."
        )

    if raw_step != RAW_RESOLUTION_MIN:
        raise RuntimeError(
            f"{path.name}: raw step {raw_step} != {RAW_RESOLUTION_MIN} min."
        )

    if horizon != FORECAST_HORIZON_MIN:
        raise RuntimeError(
            f"{path.name}: horizon {horizon} != {FORECAST_HORIZON_MIN} min."
        )

    X = X60[:, -lookback:, :].astype(
        np.float32,
        copy=True,
    )

    if X.shape[1:] != (
        lookback,
        len(FEATURE_NAMES),
    ):
        raise RuntimeError(
            f"{path.name}: bad suffix shape {X.shape}"
        )

    if not (
        np.isfinite(X).all()
        and np.isfinite(y).all()
        and np.isfinite(aw).all()
    ):
        raise RuntimeError(
            f"{path.name}: non-finite data."
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
# Physically interpretable engineered temporal features
# =============================================================================

def wrap_angle_rad(x):
    return (
        (
            np.asarray(x, dtype=np.float64)
            + np.pi
        )
        % (2.0 * np.pi)
        - np.pi
    )


def vector_angle(u, v):
    return np.arctan2(
        np.asarray(v, dtype=np.float64),
        np.asarray(u, dtype=np.float64),
    )


def angular_difference(a_now, a_prev):
    return wrap_angle_rad(
        np.asarray(a_now, dtype=np.float64)
        - np.asarray(a_prev, dtype=np.float64)
    )


def linear_slope(values):
    """
    Per-sample least-squares slope per minute.
    values: [N,T]
    """
    x = np.asarray(values, dtype=np.float64)
    t = np.arange(
        x.shape[1],
        dtype=np.float64,
    )
    t0 = t - np.mean(t)
    denom = float(
        np.sum(t0 ** 2)
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
        centered @ t0
        / max(denom, EPS)
    )


def circular_slope(angles):
    """
    Per-sample unwrapped angular slope [rad/min].
    """
    a = np.asarray(
        angles,
        dtype=np.float64,
    )
    au = np.unwrap(
        a,
        axis=1,
    )
    return linear_slope(
        au
    )


def turn_features(
    angle_now,
    angle_prev,
    confidence,
):
    d = angular_difference(
        angle_now,
        angle_prev,
    )

    c = np.asarray(
        confidence,
        dtype=np.float64,
    )

    return (
        c * np.sin(d),
        c * (1.0 - np.cos(d)),
    )


def build_summary_features(X: np.ndarray):
    """
    X: [N,15,9] (or frozen selected lookback).

    Returns
    -------
    summary: [N,K]
    names: list[str]
    """
    X = np.asarray(
        X,
        dtype=np.float64,
    )

    n, T, F = X.shape

    if T < 11:
        raise RuntimeError(
            "Enhanced feature set requires at least 11 historical minutes."
        )

    U = X[:, :, IDX_U]
    V = X[:, :, IDX_V]
    SOG = X[:, :, IDX_SOG]

    COG = np.arctan2(
        X[:, :, IDX_COG_SIN],
        X[:, :, IDX_COG_COS],
    )
    WING = np.arctan2(
        X[:, :, IDX_WING_SIN],
        X[:, :, IDX_WING_COS],
    )

    WS = np.sqrt(
        U ** 2
        + V ** 2
    )
    WANG = vector_angle(
        U,
        V,
    )

    vessel_e = (
        SOG
        * X[:, :, IDX_COG_SIN]
    )
    vessel_n = (
        SOG
        * X[:, :, IDX_COG_COS]
    )

    AW_u = U - vessel_e
    AW_v = V - vessel_n
    AWS = np.sqrt(
        AW_u ** 2
        + AW_v ** 2
    )
    AWANG = vector_angle(
        AW_u,
        AW_v,
    )

    # Current and lag indices for exact 5- and 10-min changes.
    i0 = -1
    i5 = -6
    i10 = -11

    wind_conf_5 = np.clip(
        np.minimum(
            WS[:, i0],
            WS[:, i5],
        )
        / TURN_CONFIDENCE_SPEED_SCALE_MPS,
        0.0,
        1.0,
    )
    wind_conf_10 = np.clip(
        np.minimum(
            WS[:, i0],
            WS[:, i10],
        )
        / TURN_CONFIDENCE_SPEED_SCALE_MPS,
        0.0,
        1.0,
    )

    aw_conf_5 = np.clip(
        np.minimum(
            AWS[:, i0],
            AWS[:, i5],
        )
        / TURN_CONFIDENCE_SPEED_SCALE_MPS,
        0.0,
        1.0,
    )
    aw_conf_10 = np.clip(
        np.minimum(
            AWS[:, i0],
            AWS[:, i10],
        )
        / TURN_CONFIDENCE_SPEED_SCALE_MPS,
        0.0,
        1.0,
    )

    wt5_sin, wt5_mag = turn_features(
        WANG[:, i0],
        WANG[:, i5],
        wind_conf_5,
    )
    wt10_sin, wt10_mag = turn_features(
        WANG[:, i0],
        WANG[:, i10],
        wind_conf_10,
    )

    cog5_sin, cog5_mag = turn_features(
        COG[:, i0],
        COG[:, i5],
        1.0,
    )
    cog10_sin, cog10_mag = turn_features(
        COG[:, i0],
        COG[:, i10],
        1.0,
    )

    wing5_sin, wing5_mag = turn_features(
        WING[:, i0],
        WING[:, i5],
        1.0,
    )
    wing10_sin, wing10_mag = turn_features(
        WING[:, i0],
        WING[:, i10],
        1.0,
    )

    aw5_sin, aw5_mag = turn_features(
        AWANG[:, i0],
        AWANG[:, i5],
        aw_conf_5,
    )
    aw10_sin, aw10_mag = turn_features(
        AWANG[:, i0],
        AWANG[:, i10],
        aw_conf_10,
    )

    # Confidence-weight the full-window wind/AW angular slope.
    wind_conf_full = np.clip(
        np.mean(WS, axis=1)
        / TURN_CONFIDENCE_SPEED_SCALE_MPS,
        0.0,
        1.0,
    )
    aw_conf_full = np.clip(
        np.mean(AWS, axis=1)
        / TURN_CONFIDENCE_SPEED_SCALE_MPS,
        0.0,
        1.0,
    )

    feats = []
    names = []

    def add(name, value):
        value = np.asarray(
            value,
            dtype=np.float64,
        ).reshape(n)

        feats.append(
            value
        )
        names.append(
            name
        )

    # Wind tendency.
    add("dU_5", U[:, i0] - U[:, i5])
    add("dV_5", V[:, i0] - V[:, i5])
    add("dWS_5", WS[:, i0] - WS[:, i5])

    add("dU_10", U[:, i0] - U[:, i10])
    add("dV_10", V[:, i0] - V[:, i10])
    add("dWS_10", WS[:, i0] - WS[:, i10])

    add("slope_U_full", linear_slope(U))
    add("slope_V_full", linear_slope(V))
    add("slope_WS_full", linear_slope(WS))

    # Wind directional turning.
    add("wind_turn_sin_5_conf", wt5_sin)
    add("wind_turn_mag_5_conf", wt5_mag)
    add("wind_turn_sin_10_conf", wt10_sin)
    add("wind_turn_mag_10_conf", wt10_mag)
    add(
        "wind_angular_slope_full_conf",
        wind_conf_full * circular_slope(WANG),
    )

    # Wind variability.
    add("std_U_5", np.std(U[:, -5:], axis=1, ddof=0))
    add("std_V_5", np.std(V[:, -5:], axis=1, ddof=0))
    add("std_WS_5", np.std(WS[:, -5:], axis=1, ddof=0))

    add("std_U_full", np.std(U, axis=1, ddof=0))
    add("std_V_full", np.std(V, axis=1, ddof=0))
    add("std_WS_full", np.std(WS, axis=1, ddof=0))

    # Vessel dynamics.
    add("dSOG_5", SOG[:, i0] - SOG[:, i5])
    add("dSOG_10", SOG[:, i0] - SOG[:, i10])
    add("slope_SOG_full", linear_slope(SOG))

    add("COG_turn_sin_5", cog5_sin)
    add("COG_turn_mag_5", cog5_mag)
    add("COG_turn_sin_10", cog10_sin)
    add("COG_turn_mag_10", cog10_mag)
    add("COG_angular_slope_full", circular_slope(COG))

    # Wing dynamics.
    add("WING_turn_sin_5", wing5_sin)
    add("WING_turn_mag_5", wing5_mag)
    add("WING_turn_sin_10", wing10_sin)
    add("WING_turn_mag_10", wing10_mag)
    add("WING_angular_slope_full", circular_slope(WING))

    # Historical observed apparent-wind interactions.
    add("dAWS_5", AWS[:, i0] - AWS[:, i5])
    add("dAWS_10", AWS[:, i0] - AWS[:, i10])
    add("AW_turn_sin_5_conf", aw5_sin)
    add("AW_turn_mag_5_conf", aw5_mag)
    add("AW_turn_sin_10_conf", aw10_sin)
    add("AW_turn_mag_10_conf", aw10_mag)
    add(
        "AW_angular_slope_full_conf",
        aw_conf_full * circular_slope(AWANG),
    )

    summary = np.stack(
        feats,
        axis=1,
    ).astype(np.float32)

    if not np.isfinite(
        summary
    ).all():
        raise FloatingPointError(
            "Non-finite engineered summary features."
        )

    if len(names) != summary.shape[1]:
        raise RuntimeError(
            "Engineered feature-name count mismatch."
        )

    return summary, names


# =============================================================================
# Fold-train scalers
# =============================================================================

def fit_scaler(X, S, y_uv):
    X64 = np.asarray(
        X,
        dtype=np.float64,
    )
    S64 = np.asarray(
        S,
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

    s_mean = np.mean(
        S64,
        axis=0,
    )
    s_std = np.std(
        S64,
        axis=0,
        ddof=0,
    )
    s_std = np.where(
        s_std < 1e-8,
        1.0,
        s_std,
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
        "s_mean": s_mean.astype(np.float32),
        "s_std": s_std.astype(np.float32),
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


def standardize_S(S, scaler):
    return (
        (
            np.asarray(
                S,
                dtype=np.float32,
            )
            - scaler["s_mean"][None, :]
        )
        / scaler["s_std"][None, :]
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


# =============================================================================
# Metrics
# =============================================================================

def circular_diff_deg(pred_deg, true_deg):
    return (
        (
            np.asarray(pred_deg, dtype=np.float64)
            - np.asarray(true_deg, dtype=np.float64)
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

    U_rmse = float(
        np.sqrt(
            np.mean(
                err[:, 0] ** 2
            )
        )
    )
    V_rmse = float(
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

    ws_t = np.linalg.norm(
        yt,
        axis=1,
    )
    ws_p = np.linalg.norm(
        yp,
        axis=1,
    )
    ws_rmse = float(
        np.sqrt(
            np.mean(
                (
                    ws_p
                    - ws_t
                ) ** 2
            )
        )
    )

    wd_t = meteorological_wd(
        yt
    )
    wd_p = meteorological_wd(
        yp
    )
    wd_err = circular_diff_deg(
        wd_p,
        wd_t,
    )

    wd_mae = float(
        np.mean(
            np.abs(
                wd_err
            )
        )
    )
    wd_rmse = float(
        np.sqrt(
            np.mean(
                wd_err ** 2
            )
        )
    )

    return {
        "wind_U_RMSE_mps": U_rmse,
        "wind_V_RMSE_mps": V_rmse,
        "wind_vector_RMSE_mps": vector_rmse,
        "wind_speed_RMSE_mps": ws_rmse,
        "wind_direction_MAE_deg": wd_mae,
        "wind_direction_RMSE_deg": wd_rmse,
    }


def selection_score(metrics, baseline):
    ratios = {
        "U_ratio": (
            metrics["wind_U_RMSE_mps"]
            / max(
                baseline["wind_U_RMSE_mps"],
                EPS,
            )
        ),
        "V_ratio": (
            metrics["wind_V_RMSE_mps"]
            / max(
                baseline["wind_V_RMSE_mps"],
                EPS,
            )
        ),
        "WS_ratio": (
            metrics["wind_speed_RMSE_mps"]
            / max(
                baseline["wind_speed_RMSE_mps"],
                EPS,
            )
        ),
        "WD_ratio": (
            metrics["wind_direction_RMSE_deg"]
            / max(
                baseline["wind_direction_RMSE_deg"],
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
# Ridge candidates
# =============================================================================

def fit_base_ridge(
    X_train,
    y_train,
    X_val,
    scaler,
):
    Xtr_z = standardize_X(
        X_train,
        scaler,
    )
    Xva_z = standardize_X(
        X_val,
        scaler,
    )
    ytr_z = standardize_y(
        y_train,
        scaler,
    )

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        Xtr_z.reshape(
            len(Xtr_z),
            -1,
        ),
        ytr_z,
    )

    pred_z = model.predict(
        Xva_z.reshape(
            len(Xva_z),
            -1,
        )
    )

    pred = inverse_y(
        pred_z,
        scaler,
    )

    params = int(
        np.asarray(
            model.coef_
        ).size
        + np.asarray(
            model.intercept_
        ).size
    )

    return pred, params


def fit_enhanced_ridge(
    X_train,
    S_train,
    y_train,
    X_val,
    S_val,
    scaler,
):
    Xtr_z = standardize_X(
        X_train,
        scaler,
    )
    Xva_z = standardize_X(
        X_val,
        scaler,
    )
    Str_z = standardize_S(
        S_train,
        scaler,
    )
    Sva_z = standardize_S(
        S_val,
        scaler,
    )
    ytr_z = standardize_y(
        y_train,
        scaler,
    )

    Ztr = np.concatenate(
        [
            Xtr_z.reshape(
                len(Xtr_z),
                -1,
            ),
            Str_z,
        ],
        axis=1,
    )

    Zva = np.concatenate(
        [
            Xva_z.reshape(
                len(Xva_z),
                -1,
            ),
            Sva_z,
        ],
        axis=1,
    )

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        Ztr,
        ytr_z,
    )

    pred_z = model.predict(
        Zva
    )

    pred = inverse_y(
        pred_z,
        scaler,
    )

    params = int(
        np.asarray(
            model.coef_
        ).size
        + np.asarray(
            model.intercept_
        ).size
    )

    return pred, params


# =============================================================================
# Neural models
# =============================================================================

def build_model_class(
    torch,
    nn,
    mode: str,
    lookback: int,
    summary_dim: int,
):
    seq_dim = len(FEATURE_NAMES)

    if mode == "mlp":
        class EnhancedMLP(nn.Module):
            def __init__(self):
                super().__init__()

                self.net = nn.Sequential(
                    nn.Linear(
                        lookback * seq_dim
                        + summary_dim,
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
                    nn.Linear(
                        FREEZE.mlp_hidden_2,
                        2,
                    ),
                )

            def forward(
                self,
                x,
                s,
            ):
                z = torch.cat(
                    [
                        x.flatten(
                            start_dim=1
                        ),
                        s,
                    ],
                    dim=1,
                )
                return self.net(
                    z
                )

        return EnhancedMLP

    if mode == "gru":
        class EnhancedGRU(nn.Module):
            def __init__(self):
                super().__init__()

                self.gru = nn.GRU(
                    input_size=seq_dim,
                    hidden_size=FREEZE.gru_hidden,
                    num_layers=1,
                    batch_first=True,
                )

                self.head = nn.Sequential(
                    nn.Linear(
                        FREEZE.gru_hidden
                        + summary_dim,
                        FREEZE.dense_hidden,
                    ),
                    nn.LayerNorm(
                        FREEZE.dense_hidden
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        FREEZE.dropout
                    ),
                    nn.Linear(
                        FREEZE.dense_hidden,
                        2,
                    ),
                )

            def forward(
                self,
                x,
                s,
            ):
                seq, _ = self.gru(
                    x
                )
                h = seq[:, -1, :]

                z = torch.cat(
                    [
                        h,
                        s,
                    ],
                    dim=1,
                )

                return self.head(
                    z
                )

        return EnhancedGRU

    raise ValueError(
        f"Unknown mode: {mode}"
    )


def evaluate_neural(
    *,
    torch,
    model,
    X,
    S,
    y_true,
    scaler,
    device,
    amp_enabled,
):
    model.eval()

    Xz = standardize_X(
        X,
        scaler,
    )
    Sz = standardize_S(
        S,
        scaler,
    )

    chunks = []

    with torch.no_grad():
        for xb_np, sb_np in sequential_batches(
            Xz,
            Sz,
            batch_size=FREEZE.batch_size,
        ):
            xb = torch.from_numpy(
                xb_np
            ).to(
                device,
                non_blocking=True,
            )
            sb = torch.from_numpy(
                sb_np
            ).to(
                device,
                non_blocking=True,
            )

            with amp_context(
                torch,
                amp_enabled,
                device,
            ):
                yz = model(
                    xb,
                    sb,
                )

            chunks.append(
                yz.detach()
                .float()
                .cpu()
                .numpy()
            )

    pred_z = np.concatenate(
        chunks,
        axis=0,
    )

    pred = inverse_y(
        pred_z,
        scaler,
    )

    return evaluate_wind(
        y_true,
        pred,
    )


def train_neural_run(
    *,
    torch,
    nn,
    candidate,
    mode,
    seed,
    held_out,
    X_train,
    S_train,
    y_train,
    X_val,
    S_val,
    y_val,
    scaler,
    baseline_metrics,
    lookback,
    summary_dim,
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
        lookback,
        summary_dim,
    )

    model = ModelClass().to(
        device
    )

    params = count_parameters(
        model
    )

    Xtr_z = standardize_X(
        X_train,
        scaler,
    )
    Str_z = standardize_S(
        S_train,
        scaler,
    )
    ytr_z = standardize_y(
        y_train,
        scaler,
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

    best_score = math.inf
    best_epoch = -1
    best_state = None
    best_metrics = None
    best_parts = None

    history = []

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

        total_loss = 0.0
        n_seen = 0

        for (
            xb_np,
            sb_np,
            yb_np,
        ) in sequential_batches(
            Xtr_z,
            Str_z,
            ytr_z,
            batch_size=FREEZE.batch_size,
        ):
            xb = torch.from_numpy(
                xb_np
            ).to(
                device,
                non_blocking=True,
            )
            sb = torch.from_numpy(
                sb_np
            ).to(
                device,
                non_blocking=True,
            )
            yb = torch.from_numpy(
                yb_np
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
                pred = model(
                    xb,
                    sb,
                )
                loss = torch.mean(
                    (
                        pred
                        - yb
                    ) ** 2
                )

            if not torch.isfinite(
                loss
            ):
                raise FloatingPointError(
                    "Non-finite enhanced neural loss."
                )

            if grad_scaler is not None:
                grad_scaler.scale(
                    loss
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
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    FREEZE.grad_clip_norm,
                )
                optimizer.step()

            bn = len(
                xb_np
            )
            total_loss += float(
                loss.detach()
                .float()
                .cpu()
                .item()
            ) * bn
            n_seen += bn

        val_metrics = evaluate_neural(
            torch=torch,
            model=model,
            X=X_val,
            S=S_val,
            y_true=y_val,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
        )

        score, parts = selection_score(
            val_metrics,
            baseline_metrics,
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
            best_parts = dict(
                parts
            )
            patience = 0
        else:
            patience += 1

        lr = float(
            optimizer.param_groups[0]["lr"]
        )
        scheduler.step(
            score
        )

        history.append({
            "candidate": candidate,
            "held_out_mission": held_out,
            "seed": int(seed),
            "epoch_one_based": int(epoch),
            "learning_rate": lr,
            "train_mse_z": (
                total_loss
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
            f"WD={val_metrics['wind_direction_RMSE_deg']:.3f}"
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

    if best_state is None:
        raise RuntimeError(
            "Neural candidate did not produce a valid checkpoint."
        )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    confirmed = evaluate_neural(
        torch=torch,
        model=model,
        X=X_val,
        S=S_val,
        y_true=y_val,
        scaler=scaler,
        device=device,
        amp_enabled=amp_enabled,
    )

    confirmed_score, confirmed_parts = selection_score(
        confirmed,
        baseline_metrics,
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
        "stage": "12P-B",
        "script_version": SCRIPT_VERSION,
        "candidate": candidate,
        "mode": mode,
        "held_out_mission": held_out,
        "seed": int(seed),
        "lookback_minutes": int(
            lookback
        ),
        "forecast_horizon_minutes": int(
            FORECAST_HORIZON_MIN
        ),
        "parameter_count": int(
            params
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
        "scaler": scaler,
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
            params
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
                    old["candidate"].astype(str)
                    == str(row["candidate"])
                )
                & (
                    old["held_out_mission"].astype(str)
                    == str(row["held_out_mission"])
                )
                & (
                    old["seed"].astype(int)
                    == int(row["seed"])
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
            sub["held_out_mission"].nunique()
        ),
        "selection_score_mean": float(
            sub["selection_score"].mean()
        ),
        "selection_score_std": float(
            sub["selection_score"].std(
                ddof=0
            )
        ),
        "parameter_count": int(
            round(
                sub["parameter_count"].mean()
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
    ]:
        vals = sub[
            col
        ].to_numpy(
            dtype=float
        )

        row[
            f"{col}_mean"
        ] = float(
            np.mean(
                vals
            )
        )
        row[
            f"{col}_std"
        ] = float(
            np.std(
                vals,
                ddof=0,
            )
        )

    mission_scores = (
        sub.groupby(
            "held_out_mission"
        )["selection_score"]
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
            "Enhanced-GRU only, <=8 epochs."
        ),
    )
    parser.add_argument(
        "--allow-non15-selected-lookback",
        action="store_true",
        help=(
            "Allow the script to use a frozen 12P-A selected lookback "
            "other than 15 min. Not recommended for the current experiment."
        ),
    )

    args = parser.parse_args()

    pa_dir = args.pa_dir
    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_lookback, pa_selection = (
        load_frozen_selected_lookback(
            pa_dir
        )
    )

    if (
        selected_lookback
        != FROZEN_EXPECTED_LOOKBACK_MIN
        and not args.allow_non15_selected_lookback
    ):
        raise RuntimeError(
            f"Stage 12P-A selected {selected_lookback} min, but this "
            f"scientific 12P-B script is frozen for "
            f"{FROZEN_EXPECTED_LOOKBACK_MIN} min. "
            "Use --allow-non15-selected-lookback only if intentionally "
            "starting a different experiment."
        )

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

    missions = {
        m: load_mission(
            pa_dir,
            m,
            selected_lookback,
        )
        for m in DEVELOPMENT_MISSIONS
    }

    # Build summary features once per development mission.
    summaries = {}
    summary_names = None

    for m in DEVELOPMENT_MISSIONS:
        s, names = build_summary_features(
            missions[m]["X"]
        )

        summaries[m] = s

        if summary_names is None:
            summary_names = names
        elif names != summary_names:
            raise RuntimeError(
                "Engineered summary feature schema changed across missions."
            )

    summary_dim = int(
        len(summary_names)
    )

    save_json(
        output_dir
        / "engineered_feature_schema.json",
        {
            "stage": "12P-B",
            "lookback_minutes": int(
                selected_lookback
            ),
            "forecast_horizon_minutes": int(
                FORECAST_HORIZON_MIN
            ),
            "raw_resolution_minutes": int(
                RAW_RESOLUTION_MIN
            ),
            "base_feature_names": (
                FEATURE_NAMES
            ),
            "engineered_summary_names": (
                summary_names
            ),
            "engineered_summary_dim": int(
                summary_dim
            ),
            "Tropical_Atlantic_used": False,
        },
    )

    active_holdouts = (
        ["Antarctic"]
        if args.debug_fast
        else DEVELOPMENT_MISSIONS
    )

    active_neural = (
        {
            "Enhanced-GRU-L15": "gru"
        }
        if args.debug_fast
        else {
            "Enhanced-MLP-L15": "mlp",
            "Enhanced-GRU-L15": "gru",
        }
    )

    active_seeds = (
        [SCREENING_SEEDS[0]]
        if args.debug_fast
        else SCREENING_SEEDS
    )

    log("=" * 132)
    log(
        "12P-B — ENHANCED TEMPORAL CONTEXT WIND FORECASTING"
    )
    log("=" * 132)
    log(
        f"12P-A dir       : {pa_dir}"
    )
    log(
        f"selected L      : {selected_lookback} min"
    )
    log(
        f"raw resolution  : 1 min"
    )
    log(
        f"forecast target : exactly +{FORECAST_HORIZON_MIN} min"
    )
    log(
        f"summary features: {summary_dim}"
    )
    log(
        f"device          : {device}"
    )
    if device.type == "cuda":
        log(
            f"GPU             : {torch.cuda.get_device_name(0)}"
        )
    log(
        "[FIREWALL] Antarctic / Atlantic / West Coast only."
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

        S_train = np.concatenate(
            [
                summaries[m]
                for m in train_names
            ],
            axis=0,
        )

        val = missions[
            held_out
        ]
        S_val = summaries[
            held_out
        ]

        y_train = train[
            "y"
        ][:, 0, 0:2]
        y_val = val[
            "y"
        ][:, 0, 0:2]

        scaler = fit_scaler(
            train["X"],
            S_train,
            y_train,
        )

        # -------------------------------------------------------------
        # Base-Ridge-L15: exact reference.
        # -------------------------------------------------------------
        base_pred, base_params = fit_base_ridge(
            train["X"],
            y_train,
            val["X"],
            scaler,
        )
        base_metrics = evaluate_wind(
            y_val,
            base_pred,
        )

        base_score, base_parts = selection_score(
            base_metrics,
            base_metrics,
        )

        base_row = {
            "candidate": "Base-Ridge-L15",
            "held_out_mission": held_out,
            "seed": -1,
            "status": "completed_finite",
            "parameter_count": int(
                base_params
            ),
            "best_epoch_one_based": 0,
            "selection_score": float(
                base_score
            ),
            "elapsed_seconds": 0.0,
            **base_metrics,
            **{
                f"score_{k}": v
                for k, v
                in base_parts.items()
            },
            "history_path": "",
            "checkpoint_path": "",
        }

        append_csv(
            run_path,
            base_row,
        )

        baseline_rows.append(
            base_row
        )

        # -------------------------------------------------------------
        # Enhanced-Ridge-L15.
        # -------------------------------------------------------------
        enhanced_pred, enhanced_params = fit_enhanced_ridge(
            train["X"],
            S_train,
            y_train,
            val["X"],
            S_val,
            scaler,
        )
        enhanced_metrics = evaluate_wind(
            y_val,
            enhanced_pred,
        )
        enhanced_score, enhanced_parts = selection_score(
            enhanced_metrics,
            base_metrics,
        )

        enhanced_row = {
            "candidate": "Enhanced-Ridge-L15",
            "held_out_mission": held_out,
            "seed": -1,
            "status": "completed_finite",
            "parameter_count": int(
                enhanced_params
            ),
            "best_epoch_one_based": 0,
            "selection_score": float(
                enhanced_score
            ),
            "elapsed_seconds": 0.0,
            **enhanced_metrics,
            **{
                f"score_{k}": v
                for k, v
                in enhanced_parts.items()
            },
            "history_path": "",
            "checkpoint_path": "",
        }

        append_csv(
            run_path,
            enhanced_row,
        )

        log("-" * 132)
        log(
            f"[{held_out}] "
            f"Ntrain={len(train['X']):,} | "
            f"Nval={len(val['X']):,}"
        )
        log(
            "  Base-Ridge-L15     : "
            f"U={base_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={base_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={base_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={base_metrics['wind_direction_RMSE_deg']:.3f}"
        )
        log(
            "  Enhanced-Ridge-L15 : "
            f"score={enhanced_score:.5f} | "
            f"U={enhanced_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={enhanced_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={enhanced_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={enhanced_metrics['wind_direction_RMSE_deg']:.3f}"
        )

        # -------------------------------------------------------------
        # Neural enhanced candidates.
        # -------------------------------------------------------------
        for candidate, mode in active_neural.items():
            for seed in active_seeds:
                log(
                    f"  [TRAIN] {candidate} | seed={seed}"
                )

                result = train_neural_run(
                    torch=torch,
                    nn=nn,
                    candidate=candidate,
                    mode=mode,
                    seed=seed,
                    held_out=held_out,
                    X_train=train["X"],
                    S_train=S_train,
                    y_train=y_train,
                    X_val=val["X"],
                    S_val=S_val,
                    y_val=y_val,
                    scaler=scaler,
                    baseline_metrics=base_metrics,
                    lookback=selected_lookback,
                    summary_dim=summary_dim,
                    output_dir=output_dir,
                    device=device,
                    debug_fast=args.debug_fast,
                )

                append_csv(
                    run_path,
                    result,
                )

                log(
                    f"  [DONE] score={result['selection_score']:.5f} | "
                    f"best_ep={result['best_epoch_one_based']} | "
                    f"U={result['wind_U_RMSE_mps']:.4f} | "
                    f"V={result['wind_V_RMSE_mps']:.4f} | "
                    f"WS={result['wind_speed_RMSE_mps']:.4f} | "
                    f"WD={result['wind_direction_RMSE_deg']:.3f}"
                )

        del (
            train,
            S_train,
            val,
            S_val,
            y_train,
            y_val,
            scaler,
            base_pred,
            enhanced_pred,
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
    # Summary / predeclared selection
    # =================================================================
    runs = pd.read_csv(
        run_path
    )

    expected = {
        "Base-Ridge-L15": 3,
        "Enhanced-Ridge-L15": 3,
        "Enhanced-MLP-L15": 9,
        "Enhanced-GRU-L15": 9,
    }

    for candidate, n_expected in expected.items():
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
                f"{candidate}: expected {n_expected} completed runs, got {got}."
            )

    candidate_order = [
        "Base-Ridge-L15",
        "Enhanced-Ridge-L15",
        "Enhanced-MLP-L15",
        "Enhanced-GRU-L15",
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

    # Baseline means.
    base_runs = runs.loc[
        runs["candidate"].astype(str)
        == "Base-Ridge-L15"
    ].copy()

    base_means = {
        k: float(
            base_runs[k].mean()
        )
        for k in [
            "wind_U_RMSE_mps",
            "wind_V_RMSE_mps",
            "wind_vector_RMSE_mps",
            "wind_speed_RMSE_mps",
            "wind_direction_RMSE_deg",
        ]
    }

    # Add improvements vs frozen Base Ridge.
    for idx in summary.index:
        c = str(
            summary.loc[
                idx,
                "candidate",
            ]
        )

        sub = runs.loc[
            runs["candidate"].astype(str)
            == c
        ].copy()

        for k, b in base_means.items():
            v = float(
                sub[k].mean()
            )

            summary.loc[
                idx,
                f"{k}_improvement_vs_BaseRidge_fraction",
            ] = (
                b - v
            ) / max(
                b,
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
        raw_best["candidate"]
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

    raw_best_is_enhanced = (
        raw_best_name
        != "Base-Ridge-L15"
    )

    meets_rule = bool(
        raw_best_is_enhanced
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
        selected_name = raw_best_name
        decision = "CONTINUE_ENHANCED_CONTEXT"
        reason = (
            f"{raw_best_name} clears the frozen rule: score={raw_best_score:.6f}, "
            f"mission wins={raw_best_wins}/3, vector improvement="
            f"{100.0*raw_best_vector_imp:.3f}%, V improvement="
            f"{100.0*raw_best_v_imp:.3f}%, WD improvement="
            f"{100.0*raw_best_wd_imp:.3f}%."
        )
    else:
        selected_name = "Base-Ridge-L15"
        decision = "RETAIN_BASE_RIDGE_L15"
        reason = (
            f"Raw best {raw_best_name} does not clear every predeclared "
            "enhanced-context continuation requirement. Retain Base-Ridge-L15."
        )

    selected_runs = runs.loc[
        runs["candidate"].astype(str)
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
        "stage": "12P-B",
        "script_version": SCRIPT_VERSION,
        "frozen_12P_A_selected_lookback_minutes": int(
            selected_lookback
        ),
        "raw_resolution_minutes": int(
            RAW_RESOLUTION_MIN
        ),
        "forecast_horizon_minutes": int(
            FORECAST_HORIZON_MIN
        ),
        "engineered_summary_names": (
            summary_names
        ),
        "engineered_summary_dim": int(
            summary_dim
        ),
        "score_weights": (
            SCORE_WEIGHTS
        ),
        "predeclared_continuation_rule": {
            "score_lt": (
                CONTINUE_SCORE_THRESHOLD
            ),
            "mission_wins_ge": (
                MIN_MISSION_WINS
            ),
            "vector_improvement_gt": (
                MIN_VECTOR_IMPROVEMENT
            ),
            "V_or_WD_improvement_gt": (
                MIN_V_OR_WD_IMPROVEMENT
            ),
        },
        "raw_best_candidate": (
            raw_best_name
        ),
        "selected_candidate": (
            selected_name
        ),
        "decision": (
            decision
        ),
        "reason": (
            reason
        ),
        "development_only": True,
        "development_missions": (
            DEVELOPMENT_MISSIONS
        ),
        "Tropical_Atlantic_used": False,
        "published_BP_STGNN_reference_context_only": (
            BP_STGNN_PUBLISHED_REFERENCE
        ),
        "BP_STGNN_comparability_caveat": (
            "same +10-min wall-clock horizon, not preprocessing-identical"
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
        / "12P_B_REPORT.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "12P-B Enhanced Context Wind Forecasting "
            "(1-min L15 -> exact +10-min target)\n"
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
            f"lookback: {selected_lookback} min\n"
        )
        f.write(
            "raw resolution: 1 min\n"
        )
        f.write(
            f"forecast horizon: +{FORECAST_HORIZON_MIN} min exactly\n"
        )
        f.write(
            f"engineered summary dimension: {summary_dim}\n"
        )
        f.write(
            "Tropical Atlantic accessed: NO\n\n"
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
        for k, v in BP_STGNN_PUBLISHED_REFERENCE.items():
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
        "12P-B CANDIDATE SUMMARY"
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
