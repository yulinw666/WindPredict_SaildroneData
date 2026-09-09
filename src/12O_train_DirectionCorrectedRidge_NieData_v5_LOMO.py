# -*- coding: utf-8 -*-
r"""
12O_train_DirectionCorrectedRidge_NieData_v5_LOMO.py

Stage 12O
=========
Development-only LOMO screening of a DIRECTION-ONLY correction on top of the
strong Stage-12C global Ridge wind forecast.

Core idea
---------
The Stage-12K comparison showed that the existing framework was already strong
in U and wind-speed (WS), while the remaining gap was concentrated in V and
wind-direction (WD). Stage 12N further showed that wind-aligned local-frame
changes have materially higher cross-mission correlation than global U/V
residuals.

Therefore 12O DOES NOT relearn wind speed.

For every sample, let the fold-specific Stage-12C Ridge wind prediction be

    W_R = [U_R, V_R]
    S_R = ||W_R||
    theta_R = atan2(V_R, U_R)

A compact direction model predicts only a correction

    dtheta_hat

and the final prediction is

    theta_hat = theta_R + dtheta_hat

    U_hat = S_R * cos(theta_hat)
    V_hat = S_R * sin(theta_hat)

Hence, up to floating-point roundoff,

    WS_hat == WS_Ridge

for every sample.

This creates a clean mechanistic test:
    Can we improve V/WD by rotating an already-strong Ridge vector without
    sacrificing its wind-speed magnitude?

Scientific firewall
-------------------
This script reads ONLY the three development missions:
    Antarctic.npz
    Atlantic.npz
    West_Coast.npz

It NEVER reads:
    Tropical_Atlantic_TEST.npz
    Stage-12K outputs
    Stage-12K metrics

All model fitting, checkpoint selection, and candidate selection are
development-only.

Frozen Stage-12C protocol
-------------------------
X: [N,6,9]
   [U, V, T, RH, SOG,
    COG_sin, COG_cos,
    WING_ANGLE_sin, WING_ANGLE_cos]

y: [N,1,4]
   [U, V, VESSEL_E, VESSEL_N]

The exact Stage-12C fold-specific Global Ridge remains the anchor and the
reference baseline.

Direction input representation
------------------------------
For each sample, construct the same wind-aligned local frame used in Stage 12N
from the LAST observed wind:

    e_parallel = W_t / ||W_t||
    e_perp     = [-e_parallel_N, e_parallel_E]

Weak-wind fallback:
    last wind -> recent mean wind -> fixed East axis

Transform the sequence:
    [U,V]                 -> [W_parallel, W_perp]
    [COG_sin,COG_cos]     -> [COG_parallel, COG_perp]

Preserve:
    T, RH, SOG, WING_ANGLE_sin, WING_ANGLE_cos

The resulting input still has 9 features per time step.

Direction target
----------------
For each fold:

    dtheta_true = wrap(theta_true - theta_Ridge)

The correction is clipped only for the TRAINING TARGET to a predeclared
physical/regularization range of +/-45 deg. Evaluation always uses the full
unmodified ground-truth U/V.

Low-wind direction confidence
-----------------------------
Direction is poorly defined at very low wind speed. Training samples receive a
fold-local deterministic weight based on wind-speed magnitude:

    speed_ref = min(WS_true, WS_Ridge)
    w = clip(speed_ref / 5 m/s, 0, 1)^2

The weights are normalized to mean 1 on the fold training set.
No validation information is used in these weights.

Candidates
----------
A) Direction-Ridge
   Weighted Ridge(alpha=1.0) predicts normalized dtheta from flattened
   standardized local-frame history plus the normalized Ridge direction
   residual context.

B) Direction-MLP
   Compact MLP predicts bounded dtheta.

C) Direction-GRU
   Compact GRU predicts bounded dtheta.

All neural outputs are bounded with tanh to +/-45 deg.

Neural loss
-----------
Let e = wrap(theta_hat - theta_true).

    L_circ  = weighted mean(1 - cos(e))
    L_huber = weighted SmoothL1(dtheta_pred_norm, dtheta_target_norm)
    L_mag   = mean(dtheta_pred_norm^2)

    L = 0.60 * L_circ + 0.40 * L_huber + 1e-4 * L_mag

Since wind speed is frozen to Ridge, minimizing angular error also directly
reduces the direction-dependent part of vector error.

Predeclared development selection score
---------------------------------------
Because WS is mathematically frozen, the score focuses on U/V/WD:

    DirectionScore =
        0.20 * U_RMSE / Ridge_U_RMSE
      + 0.45 * V_RMSE / Ridge_V_RMSE
      + 0.35 * WD_RMSE / Ridge_WD_RMSE

Global Ridge = 1.0.

Predeclared continuation rule
-----------------------------
Continue this mechanism toward final integration only if ALL are true on
development-only LOMO means:

    mean DirectionScore < 0.98
    mean V_RMSE improves > 2%
    mean WD_RMSE improves > 2%
    mean vector_RMSE does not worsen by more than 0.5%
    >= 2 of 3 held-out missions have mean DirectionScore < 1.0
    WS equality audit passes

Otherwise:
    STOP_DIRECTION_CORRECTION_ESCALATION

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\12O_train_DirectionCorrectedRidge_NieData_v5_LOMO.py" --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12O_DirectionCorrectedRidge_NieData_v5_LOMO_point_v0_1"

Smoke test
----------
Add:
    --debug-fast
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
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


SCRIPT_VERSION = "0.1.0-DirectionCorrectedRidge-NieData-v5-LOMO"

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
    / "12O_DirectionCorrectedRidge_NieData_v5_LOMO_point_v0_1"
)

DEVELOPMENT_MISSIONS = ["Antarctic", "Atlantic", "West Coast"]
SCREENING_SEEDS = [500043, 501052, 502061]

# Frozen Stage-12C feature indices.
IDX_U = 0
IDX_V = 1
IDX_T = 2
IDX_RH = 3
IDX_SOG = 4
IDX_COG_SIN = 5
IDX_COG_COS = 6
IDX_WING_SIN = 7
IDX_WING_COS = 8

EXPECTED_FEATURE_NAMES = [
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
EXPECTED_TARGET_NAMES = ["U", "V", "VESSEL_E", "VESSEL_N"]

LOCAL_FEATURE_NAMES = [
    "W_parallel",
    "W_perp",
    "T",
    "RH",
    "SOG",
    "COG_parallel",
    "COG_perp",
    "WING_ANGLE_sin",
    "WING_ANGLE_cos",
]

EPS = 1e-12
FRAME_MIN_SPEED_MPS = 0.50
DIRECTION_WEIGHT_SPEED_SCALE_MPS = 5.0
MAX_CORRECTION_DEG = 45.0
MAX_CORRECTION_RAD = math.radians(MAX_CORRECTION_DEG)

DIRECTION_SCORE_WEIGHTS = {
    "U": 0.20,
    "V": 0.45,
    "WD": 0.35,
}

CONTINUE_SCORE_THRESHOLD = 0.98
CONTINUE_V_IMPROVEMENT = 0.02
CONTINUE_WD_IMPROVEMENT = 0.02
MAX_VECTOR_WORSENING = 0.005
MIN_MISSION_WINS = 2
WS_EQUALITY_TOL_MPS = 1e-5


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

    circular_loss_weight: float = 0.60
    huber_loss_weight: float = 0.40
    correction_magnitude_penalty: float = 1e-4

    use_amp: bool = True


FREEZE = TrainFreeze()


def log(msg=""):
    print(msg, flush=True)


def save_json(path: Path, obj):
    def cv(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return None if not np.isfinite(x) else float(x)
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, dict):
            return {str(k): cv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [cv(v) for v in x]
        return x

    with path.open("w", encoding="utf-8") as f:
        json.dump(cv(obj), f, ensure_ascii=False, indent=2, sort_keys=True)


def load_base_module(project_root: Path):
    path = project_root / "src" / "12C_train_PhysicsCompact_NieData_LOMO.py"
    if not path.exists():
        raise FileNotFoundError(
            "Required Stage-12C base script not found:\n"
            f"  {path}"
        )

    name = "stage12c_base_for_12o"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for: {path}")

    mod = importlib.util.module_from_spec(spec)

    # Required for Python 3.11 dataclass/type introspection.
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise

    return mod, path


def audit_base_protocol(base):
    if list(base.FEATURE_NAMES) != EXPECTED_FEATURE_NAMES:
        raise RuntimeError(
            "Unexpected Stage-12C feature order.\n"
            f"Expected: {EXPECTED_FEATURE_NAMES}\n"
            f"Found   : {list(base.FEATURE_NAMES)}"
        )
    if list(base.TARGET_NAMES) != EXPECTED_TARGET_NAMES:
        raise RuntimeError(
            "Unexpected Stage-12C target order.\n"
            f"Expected: {EXPECTED_TARGET_NAMES}\n"
            f"Found   : {list(base.TARGET_NAMES)}"
        )


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
    if any(len(a) != n for a in arrays):
        raise ValueError("Batch arrays have inconsistent first dimensions.")

    for start in range(0, n, int(batch_size)):
        stop = min(start + int(batch_size), n)
        yield tuple(a[start:stop] for a in arrays)


def wrap_angle_rad(x):
    x = np.asarray(x, dtype=np.float64)
    return ((x + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def angle_from_uv(w):
    w = np.asarray(w, dtype=np.float64)
    return np.arctan2(w[..., 1], w[..., 0]).astype(np.float32)


# =============================================================================
# Local-frame input transform
# =============================================================================

def build_local_frame(X_raw: np.ndarray):
    X = np.asarray(X_raw, dtype=np.float64)

    if X.ndim != 3 or X.shape[1:] != (6, 9):
        raise ValueError(f"Expected X [N,6,9], got {X.shape}")

    last_w = X[:, -1, [IDX_U, IDX_V]]
    last_speed = np.linalg.norm(last_w, axis=1)

    mean_w = np.mean(
        X[:, :, [IDX_U, IDX_V]],
        axis=1,
    )
    mean_speed = np.linalg.norm(mean_w, axis=1)

    frame_w = last_w.copy()

    use_mean = last_speed < FRAME_MIN_SPEED_MPS
    frame_w[use_mean] = mean_w[use_mean]

    use_east = use_mean & (mean_speed < FRAME_MIN_SPEED_MPS)
    frame_w[use_east, 0] = 1.0
    frame_w[use_east, 1] = 0.0

    norm = np.maximum(
        np.linalg.norm(frame_w, axis=1),
        EPS,
    )

    epar = frame_w / norm[:, None]
    eperp = np.stack(
        [-epar[:, 1], epar[:, 0]],
        axis=1,
    )

    audit = {
        "n": int(len(X)),
        "use_recent_mean_fraction": float(np.mean(use_mean)),
        "use_fixed_east_fraction": float(np.mean(use_east)),
        "orthogonality_max_abs_error": float(
            np.max(np.abs(np.sum(epar * eperp, axis=1)))
        ),
        "epar_unit_error_max": float(
            np.max(
                np.abs(np.linalg.norm(epar, axis=1) - 1.0)
            )
        ),
    }

    return (
        epar.astype(np.float32),
        eperp.astype(np.float32),
        audit,
    )


def rotate_to_local(vec_en, epar, eperp):
    vec = np.asarray(vec_en, dtype=np.float64)
    ep = np.asarray(epar, dtype=np.float64)
    eq = np.asarray(eperp, dtype=np.float64)

    if vec.ndim == 3:
        par = np.sum(vec * ep[:, None, :], axis=-1)
        perp = np.sum(vec * eq[:, None, :], axis=-1)
        return np.stack([par, perp], axis=-1).astype(np.float32)

    if vec.ndim == 2:
        par = np.sum(vec * ep, axis=-1)
        perp = np.sum(vec * eq, axis=-1)
        return np.stack([par, perp], axis=-1).astype(np.float32)

    raise ValueError(f"Unsupported vector shape: {vec.shape}")


def make_local_X(X_raw: np.ndarray):
    X = np.asarray(X_raw, dtype=np.float32)

    epar, eperp, audit = build_local_frame(X)

    out = np.empty_like(X, dtype=np.float32)

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

    if not np.isfinite(out).all():
        raise FloatingPointError("Non-finite local-frame input.")

    return {
        "X_local": out,
        "epar": epar,
        "eperp": eperp,
        "audit": audit,
    }


# =============================================================================
# Fold-only local feature standardization
# =============================================================================

def fit_local_x_scaler(X_local):
    X = np.asarray(X_local, dtype=np.float64)
    mean = np.mean(X, axis=(0, 1))
    std = np.std(X, axis=(0, 1), ddof=0)
    std = np.where(std < 1e-8, 1.0, std)

    return {
        "x_mean": mean.astype(np.float32),
        "x_std": std.astype(np.float32),
    }


def standardize_local_X(X_local, scaler):
    return (
        (
            np.asarray(X_local, dtype=np.float32)
            - scaler["x_mean"][None, None, :]
        )
        / scaler["x_std"][None, None, :]
    ).astype(np.float32)


# =============================================================================
# Direction targets / weights / reconstruction
# =============================================================================

def make_direction_target_and_weights(
    y_true,
    global_ridge_raw,
    normalize_weights: bool,
):
    true_w = np.asarray(
        y_true[:, 0, 0:2],
        dtype=np.float64,
    )
    ridge_w = np.asarray(
        global_ridge_raw[:, 0, 0:2],
        dtype=np.float64,
    )

    theta_true = np.arctan2(
        true_w[:, 1], true_w[:, 0]
    )
    theta_ridge = np.arctan2(
        ridge_w[:, 1], ridge_w[:, 0]
    )

    delta_raw = wrap_angle_rad(
        theta_true - theta_ridge
    ).astype(np.float64)

    delta_clipped = np.clip(
        delta_raw,
        -MAX_CORRECTION_RAD,
        MAX_CORRECTION_RAD,
    )

    target_norm = (
        delta_clipped / MAX_CORRECTION_RAD
    ).astype(np.float32)

    true_speed = np.linalg.norm(
        true_w, axis=1
    )
    ridge_speed = np.linalg.norm(
        ridge_w, axis=1
    )
    speed_ref = np.minimum(
        true_speed, ridge_speed
    )

    weights = np.clip(
        speed_ref
        / DIRECTION_WEIGHT_SPEED_SCALE_MPS,
        0.0,
        1.0,
    ) ** 2

    if normalize_weights:
        mean_w = float(np.mean(weights))
        if mean_w > EPS:
            weights = weights / mean_w

    audit = {
        "target_abs_deg_mean_raw": float(
            np.degrees(np.mean(np.abs(delta_raw)))
        ),
        "target_abs_deg_p90_raw": float(
            np.degrees(np.quantile(np.abs(delta_raw), 0.90))
        ),
        "target_abs_deg_p99_raw": float(
            np.degrees(np.quantile(np.abs(delta_raw), 0.99))
        ),
        "target_clipped_fraction": float(
            np.mean(np.abs(delta_raw) > MAX_CORRECTION_RAD)
        ),
        "weight_mean": float(np.mean(weights)),
        "weight_zero_fraction": float(
            np.mean(weights <= 1e-12)
        ),
        "true_speed_mean": float(np.mean(true_speed)),
        "ridge_speed_mean": float(np.mean(ridge_speed)),
    }

    return {
        "theta_true": theta_true.astype(np.float32),
        "theta_ridge": theta_ridge.astype(np.float32),
        "delta_raw": delta_raw.astype(np.float32),
        "target_norm": target_norm,
        "weights": weights.astype(np.float32),
        "audit": audit,
    }


def reconstruct_direction_corrected(
    global_ridge_raw,
    correction_rad,
):
    ridge = np.asarray(
        global_ridge_raw,
        dtype=np.float64,
    )
    correction = np.asarray(
        correction_rad,
        dtype=np.float64,
    ).reshape(-1)

    ridge_w = ridge[:, 0, 0:2]
    speed = np.linalg.norm(
        ridge_w, axis=1
    )
    theta = np.arctan2(
        ridge_w[:, 1], ridge_w[:, 0]
    )

    theta_new = theta + correction

    wind_new = np.stack(
        [
            speed * np.cos(theta_new),
            speed * np.sin(theta_new),
        ],
        axis=1,
    )

    out = ridge.copy()
    out[:, 0, 0:2] = wind_new

    return out.astype(np.float32)


def ws_equality_audit(
    pred_raw,
    ridge_raw,
):
    pred_w = np.asarray(
        pred_raw[:, 0, 0:2],
        dtype=np.float64,
    )
    ridge_w = np.asarray(
        ridge_raw[:, 0, 0:2],
        dtype=np.float64,
    )

    pred_s = np.linalg.norm(
        pred_w, axis=1
    )
    ridge_s = np.linalg.norm(
        ridge_w, axis=1
    )

    diff = np.abs(pred_s - ridge_s)

    return {
        "WS_equality_max_abs_mps": float(
            np.max(diff)
        ),
        "WS_equality_mean_abs_mps": float(
            np.mean(diff)
        ),
        "WS_equality_pass": bool(
            np.max(diff)
            <= WS_EQUALITY_TOL_MPS
        ),
    }


def direction_correction_diagnostics(
    correction_rad,
    target_info,
):
    pred = np.asarray(
        correction_rad,
        dtype=np.float64,
    ).reshape(-1)
    true_raw = np.asarray(
        target_info["delta_raw"],
        dtype=np.float64,
    ).reshape(-1)

    circular_error = wrap_angle_rad(
        pred - true_raw
    ).astype(np.float64)

    return {
        "direction_correction_abs_deg_mean": float(
            np.degrees(np.mean(np.abs(pred)))
        ),
        "direction_correction_abs_deg_p90": float(
            np.degrees(np.quantile(np.abs(pred), 0.90))
        ),
        "direction_correction_target_corr": corrcoef_safe(
            pred, true_raw
        ),
        "direction_correction_error_MAE_deg": float(
            np.degrees(
                np.mean(np.abs(circular_error))
            )
        ),
        "direction_correction_error_RMSE_deg": float(
            np.degrees(
                np.sqrt(
                    np.mean(circular_error ** 2)
                )
            )
        ),
    }


def corrcoef_safe(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    if len(a) < 3:
        return np.nan
    if np.std(a) < EPS or np.std(b) < EPS:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def direction_selection_score(
    metrics,
    ridge_metrics,
):
    u = metrics["wind_U_RMSE_mps"] / max(
        ridge_metrics["wind_U_RMSE_mps"],
        EPS,
    )
    v = metrics["wind_V_RMSE_mps"] / max(
        ridge_metrics["wind_V_RMSE_mps"],
        EPS,
    )
    wd = metrics["wind_direction_RMSE_deg"] / max(
        ridge_metrics["wind_direction_RMSE_deg"],
        EPS,
    )

    score = (
        DIRECTION_SCORE_WEIGHTS["U"] * u
        + DIRECTION_SCORE_WEIGHTS["V"] * v
        + DIRECTION_SCORE_WEIGHTS["WD"] * wd
    )

    return float(score), {
        "U_ratio": float(u),
        "V_ratio": float(v),
        "WD_ratio": float(wd),
    }


def metric_subset(metrics):
    keys = [
        "wind_U_RMSE_mps",
        "wind_V_RMSE_mps",
        "wind_vector_RMSE_mps",
        "wind_speed_RMSE_mps",
        "wind_direction_MAE_deg",
        "wind_direction_RMSE_deg",
        "vessel_vector_RMSE_mps",
        "AW_vector_RMSE_mps",
        "AWS_RMSE_mps",
    ]
    return {
        k: float(metrics[k])
        for k in keys
        if k in metrics
    }


# =============================================================================
# Direction-Ridge
# =============================================================================

def build_direction_ridge_features(
    X_local_z,
    global_ridge_raw,
    epar,
    eperp,
):
    Xf = np.asarray(
        X_local_z,
        dtype=np.float32,
    ).reshape(len(X_local_z), -1)

    ridge_w_en = np.asarray(
        global_ridge_raw[:, 0, 0:2],
        dtype=np.float64,
    )
    ridge_speed = np.linalg.norm(
        ridge_w_en, axis=1
    )

    # IMPORTANT:
    # Keep the direction context rotation-aware.  The Ridge vector is expressed
    # in the SAME sample-specific local wind frame as the transformed history,
    # rather than leaking absolute East/North orientation back into the model.
    ridge_w_local = rotate_to_local(
        ridge_w_en,
        epar,
        eperp,
    ).astype(np.float64)

    denom = np.maximum(
        ridge_speed,
        1e-6,
    )
    ridge_unit_local = (
        ridge_w_local
        / denom[:, None]
    )

    speed_context = (
        ridge_speed[:, None]
        / DIRECTION_WEIGHT_SPEED_SCALE_MPS
    )

    return np.concatenate(
        [
            Xf,
            ridge_unit_local.astype(np.float32),
            speed_context.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def fit_direction_ridge(
    X_train_features,
    target_norm,
    sample_weights,
):
    model = Ridge(alpha=1.0)
    model.fit(
        X_train_features,
        target_norm,
        sample_weight=sample_weights,
    )
    return model


def predict_direction_ridge(
    model,
    X_features,
):
    pred_norm = np.asarray(
        model.predict(X_features),
        dtype=np.float64,
    ).reshape(-1)

    pred_norm = np.clip(
        pred_norm,
        -1.0,
        1.0,
    )

    correction = (
        pred_norm * MAX_CORRECTION_RAD
    )

    return (
        pred_norm.astype(np.float32),
        correction.astype(np.float32),
    )


def direction_ridge_parameter_count(model):
    return int(
        np.asarray(model.coef_).size
        + np.asarray(model.intercept_).size
    )


# =============================================================================
# Neural direction models
# =============================================================================

def build_neural_model_class(torch, nn, mode: str):
    feature_count = len(LOCAL_FEATURE_NAMES)

    if mode == "mlp":
        class DirectionMLP(nn.Module):
            def __init__(self):
                super().__init__()

                # +3: Ridge unit East/North + speed context.
                self.backbone = nn.Sequential(
                    nn.Linear(
                        6 * feature_count + 3,
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

                # Exact zero direction correction at epoch 0.
                nn.init.zeros_(self.out.weight)
                nn.init.zeros_(self.out.bias)

            def forward(self, x, ridge_context):
                h = torch.cat(
                    [
                        x.flatten(start_dim=1),
                        ridge_context,
                    ],
                    dim=1,
                )
                raw = self.out(
                    self.backbone(h)
                ).squeeze(-1)

                return torch.tanh(raw)

        return DirectionMLP

    if mode == "gru":
        class DirectionGRU(nn.Module):
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
                        FREEZE.gru_hidden + 3,
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

                nn.init.zeros_(self.out.weight)
                nn.init.zeros_(self.out.bias)

            def forward(self, x, ridge_context):
                seq, _ = self.gru(x)
                h = seq[:, -1, :]

                z = self.backbone(
                    torch.cat(
                        [h, ridge_context],
                        dim=1,
                    )
                )
                raw = self.out(z).squeeze(-1)

                return torch.tanh(raw)

        return DirectionGRU

    raise ValueError(f"Unknown neural mode: {mode}")


def build_ridge_context(
    global_ridge_raw,
    epar,
    eperp,
):
    ridge_w_en = np.asarray(
        global_ridge_raw[:, 0, 0:2],
        dtype=np.float64,
    )
    speed = np.linalg.norm(
        ridge_w_en, axis=1
    )

    ridge_w_local = rotate_to_local(
        ridge_w_en,
        epar,
        eperp,
    ).astype(np.float64)

    denom = np.maximum(
        speed,
        1e-6,
    )
    unit_local = (
        ridge_w_local
        / denom[:, None]
    )

    speed_ctx = (
        speed[:, None]
        / DIRECTION_WEIGHT_SPEED_SCALE_MPS
    )

    return np.concatenate(
        [
            unit_local.astype(np.float32),
            speed_ctx.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def evaluate_neural(
    *,
    base,
    torch,
    model,
    X_local,
    local_x_scaler,
    ridge_context,
    y_true,
    aw,
    global_ridge_raw,
    target_info,
    device,
    amp_enabled,
):
    model.eval()

    Xz = standardize_local_X(
        X_local,
        local_x_scaler,
    )

    pred_chunks = []

    with torch.no_grad():
        for xb_np, cb_np in sequential_batches(
            Xz,
            ridge_context,
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
                    xb, cb
                )

            pred_chunks.append(
                pred_norm.detach()
                .float()
                .cpu()
                .numpy()
            )

    pred_norm = np.concatenate(
        pred_chunks,
        axis=0,
    ).reshape(-1)

    correction = (
        pred_norm.astype(np.float64)
        * MAX_CORRECTION_RAD
    ).astype(np.float32)

    pred_raw = reconstruct_direction_corrected(
        global_ridge_raw,
        correction,
    )

    metrics = base.evaluate_raw(
        y_true,
        pred_raw,
        aw,
    )

    metrics.update(
        ws_equality_audit(
            pred_raw,
            global_ridge_raw,
        )
    )
    metrics.update(
        direction_correction_diagnostics(
            correction,
            target_info,
        )
    )

    return (
        metrics,
        pred_norm.astype(np.float32),
        correction,
        pred_raw,
    )


def weighted_mean_torch(torch, values, weights):
    num = torch.sum(values * weights)
    den = torch.sum(weights).clamp_min(1e-8)
    return num / den


def train_neural_run(
    *,
    base,
    torch,
    nn,
    candidate_name,
    mode,
    seed,
    held_out,
    train_local,
    val_local,
    train_target,
    val_target,
    y_val,
    aw_val,
    global_ridge_train_raw,
    global_ridge_val_raw,
    global_ridge_val_metrics,
    local_x_scaler,
    device,
    output_dir,
    debug_fast,
):
    seed_everything(
        torch, seed
    )

    ModelClass = build_neural_model_class(
        torch,
        nn,
        mode,
    )
    model = ModelClass().to(device)

    parameter_count = count_parameters(
        model
    )

    X_train_z = standardize_local_X(
        train_local["X_local"],
        local_x_scaler,
    )

    ridge_context_train = build_ridge_context(
        global_ridge_train_raw,
        train_local["epar"],
        train_local["eperp"],
    )
    ridge_context_val = build_ridge_context(
        global_ridge_val_raw,
        val_local["epar"],
        val_local["eperp"],
    )

    target_norm_train = np.asarray(
        train_target["target_norm"],
        dtype=np.float32,
    )
    weights_train = np.asarray(
        train_target["weights"],
        dtype=np.float32,
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

    # Epoch 0 = exact global Ridge, because the direction correction head is zero.
    initial_metrics, _, _, _ = evaluate_neural(
        base=base,
        torch=torch,
        model=model,
        X_local=val_local["X_local"],
        local_x_scaler=local_x_scaler,
        ridge_context=ridge_context_val,
        y_true=y_val,
        aw=aw_val,
        global_ridge_raw=global_ridge_val_raw,
        target_info=val_target,
        device=device,
        amp_enabled=amp_enabled,
    )

    best_score, best_parts = direction_selection_score(
        initial_metrics,
        global_ridge_val_metrics,
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
        "candidate": candidate_name,
        "held_out_mission": held_out,
        "seed": int(seed),
        "epoch_one_based": 0,
        "learning_rate": float(
            FREEZE.learning_rate
        ),
        "selection_score": float(
            best_score
        ),
        "checkpoint_improved": True,
        "train_loss": np.nan,
        "train_circular_loss": np.nan,
        "train_huber_loss": np.nan,
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

    huber_fn = nn.SmoothL1Loss(
        reduction="none",
        beta=0.15,
    )

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        model.train()

        sum_loss = 0.0
        sum_circ = 0.0
        sum_huber = 0.0
        n_seen = 0

        for (
            xb_np,
            cb_np,
            tb_np,
            wb_np,
        ) in sequential_batches(
            X_train_z,
            ridge_context_train,
            target_norm_train,
            weights_train,
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
            wb = torch.from_numpy(
                wb_np
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
                    xb, cb
                )

                pred_angle = (
                    pred_norm
                    * MAX_CORRECTION_RAD
                )
                true_angle = (
                    tb
                    * MAX_CORRECTION_RAD
                )

                angle_err = (
                    pred_angle
                    - true_angle
                )

                circ_per = (
                    1.0
                    - torch.cos(
                        angle_err
                    )
                )

                huber_per = huber_fn(
                    pred_norm,
                    tb,
                )

                circ_loss = weighted_mean_torch(
                    torch,
                    circ_per,
                    wb,
                )
                huber_loss = weighted_mean_torch(
                    torch,
                    huber_per,
                    wb,
                )
                mag_penalty = torch.mean(
                    pred_norm ** 2
                )

                loss = (
                    FREEZE.circular_loss_weight
                    * circ_loss
                    + FREEZE.huber_loss_weight
                    * huber_loss
                    + FREEZE.correction_magnitude_penalty
                    * mag_penalty
                )

            if not torch.isfinite(
                loss
            ):
                raise FloatingPointError(
                    "Non-finite direction loss."
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

            bn = len(xb_np)

            sum_loss += float(
                loss.detach()
                .float()
                .cpu()
                .item()
            ) * bn
            sum_circ += float(
                circ_loss.detach()
                .float()
                .cpu()
                .item()
            ) * bn
            sum_huber += float(
                huber_loss.detach()
                .float()
                .cpu()
                .item()
            ) * bn
            n_seen += bn

        val_metrics, _, _, _ = evaluate_neural(
            base=base,
            torch=torch,
            model=model,
            X_local=val_local["X_local"],
            local_x_scaler=local_x_scaler,
            ridge_context=ridge_context_val,
            y_true=y_val,
            aw=aw_val,
            global_ridge_raw=global_ridge_val_raw,
            target_info=val_target,
            device=device,
            amp_enabled=amp_enabled,
        )

        score, parts = direction_selection_score(
            val_metrics,
            global_ridge_val_metrics,
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

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )
        scheduler.step(
            score
        )

        history.append({
            "candidate": candidate_name,
            "held_out_mission": held_out,
            "seed": int(seed),
            "epoch_one_based": int(epoch),
            "learning_rate": current_lr,
            "selection_score": float(
                score
            ),
            "checkpoint_improved": bool(
                improved
            ),
            "train_loss": (
                sum_loss
                / max(n_seen, 1)
            ),
            "train_circular_loss": (
                sum_circ
                / max(n_seen, 1)
            ),
            "train_huber_loss": (
                sum_huber
                / max(n_seen, 1)
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
            f"WD={val_metrics['wind_direction_RMSE_deg']:.2f} | "
            f"dCorr={val_metrics['direction_correction_target_corr']:+.3f}"
            + (
                " *"
                if improved
                else ""
            )
        )

        if patience >= patience_limit:
            log(
                f"      early stop after "
                f"{patience} epochs "
                "without score improvement."
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

    confirmed, _, _, _ = evaluate_neural(
        base=base,
        torch=torch,
        model=model,
        X_local=val_local["X_local"],
        local_x_scaler=local_x_scaler,
        ridge_context=ridge_context_val,
        y_true=y_val,
        aw=aw_val,
        global_ridge_raw=global_ridge_val_raw,
        target_info=val_target,
        device=device,
        amp_enabled=amp_enabled,
    )

    confirmed_score, confirmed_parts = (
        direction_selection_score(
            confirmed,
            global_ridge_val_metrics,
        )
    )

    if abs(
        confirmed_score
        - best_score
    ) > 1e-6:
        raise RuntimeError(
            "Best direction checkpoint "
            "re-evaluation mismatch."
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
        f"{candidate_name}"
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
        "stage": "12O",
        "script_version": SCRIPT_VERSION,
        "candidate": candidate_name,
        "mode": mode,
        "held_out_mission": held_out,
        "seed": int(seed),
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
        "local_x_scaler": (
            local_x_scaler
        ),
        "local_feature_names": (
            LOCAL_FEATURE_NAMES
        ),
        "direction_score_weights": (
            DIRECTION_SCORE_WEIGHTS
        ),
        "max_correction_deg": (
            MAX_CORRECTION_DEG
        ),
        "frame_min_speed_mps": (
            FRAME_MIN_SPEED_MPS
        ),
        "training_weight_speed_scale_mps": (
            DIRECTION_WEIGHT_SPEED_SCALE_MPS
        ),
        "Tropical_Atlantic_used": False,
        "Stage_12K_used": False,
    }, checkpoint_path)

    result = {
        "candidate": candidate_name,
        "mode": mode,
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
# Reporting
# =============================================================================

def append_csv(path: Path, row: dict):
    new = pd.DataFrame([row])

    if path.exists():
        old = pd.read_csv(path)

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
            [old, new],
            ignore_index=True,
        )

    new.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def boolean_series_all_true(series):
    """Robustly interpret CSV-loaded boolean values."""
    vals = []
    for x in series:
        if isinstance(x, (bool, np.bool_)):
            vals.append(bool(x))
        else:
            vals.append(
                str(x).strip().lower()
                in {"true", "1", "yes", "y"}
            )
    return bool(all(vals))


def summarize_candidate(
    runs: pd.DataFrame,
    candidate: str,
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
        "runs": int(len(sub)),
        "mission_count": int(
            sub["held_out_mission"].nunique()
        ),
        "selection_score_mean": float(
            sub["selection_score"].mean()
        ),
        "selection_score_std": float(
            sub["selection_score"].std(ddof=0)
        ),
        "parameter_count": int(
            round(
                sub["parameter_count"].mean()
            )
        ),
        "best_epoch_median": float(
            np.median(
                sub["best_epoch_one_based"]
                .to_numpy(dtype=float)
            )
        ),
    }

    for col in [
        "wind_U_RMSE_mps",
        "wind_V_RMSE_mps",
        "wind_vector_RMSE_mps",
        "wind_speed_RMSE_mps",
        "wind_direction_RMSE_deg",
        "AW_vector_RMSE_mps",
        "AWS_RMSE_mps",
        "direction_correction_target_corr",
        "direction_correction_abs_deg_mean",
        "direction_correction_error_MAE_deg",
        "WS_equality_max_abs_mps",
    ]:
        vals = sub[col].to_numpy(
            dtype=float
        )
        row[f"{col}_mean"] = float(
            np.nanmean(vals)
        )
        row[f"{col}_std"] = float(
            np.nanstd(vals, ddof=0)
        )

    mission_scores = (
        sub.groupby(
            "held_out_mission"
        )["selection_score"]
        .mean()
    )

    row["mission_wins_vs_ridge"] = int(
        np.sum(
            mission_scores.to_numpy()
            < 1.0
        )
    )

    row["ws_audit_all_pass"] = boolean_series_all_true(
        sub["WS_equality_pass"]
    )

    return row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )
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
    parser.add_argument(
        "--force-cpu",
        action="store_true",
    )
    parser.add_argument(
        "--debug-fast",
        action="store_true",
        help=(
            "NON-SCIENTIFIC smoke: Antarctic holdout, "
            "one seed, Direction-GRU, <=8 epochs."
        ),
    )

    args = parser.parse_args()

    project_root = (
        args.project_root
    )
    dataset_dir = (
        args.dataset_dir
    )
    output_dir = (
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base, base_path = (
        load_base_module(
            project_root
        )
    )
    audit_base_protocol(
        base
    )

    torch, nn, _ = (
        base.import_torch()
    )
    base.configure_cuda(
        torch
    )

    if (
        args.force_cpu
        or not torch.cuda.is_available()
    ):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    dev_paths = (
        base.explicit_development_paths(
            dataset_dir
        )
    )

    # Hard scientific firewall.
    for mission, path in (
        dev_paths.items()
    ):
        if "tropical" in str(
            path
        ).lower():
            raise RuntimeError(
                "FIREWALL VIOLATION: "
                f"{mission} -> {path}"
            )

    missions = {
        m: base.load_development_mission(
            dev_paths[m],
            m,
        )
        for m in DEVELOPMENT_MISSIONS
    }

    active_holdouts = (
        ["Antarctic"]
        if args.debug_fast
        else DEVELOPMENT_MISSIONS
    )
    active_seeds = (
        [SCREENING_SEEDS[0]]
        if args.debug_fast
        else SCREENING_SEEDS
    )
    active_neural = (
        {
            "Direction-GRU": "gru"
        }
        if args.debug_fast
        else {
            "Direction-MLP": "mlp",
            "Direction-GRU": "gru",
        }
    )

    log("=" * 132)
    log(
        "12O — DIRECTION-CORRECTED RIDGE WIND "
        "DEVELOPMENT-ONLY LOMO"
    )
    log("=" * 132)
    log(
        f"script version : "
        f"{SCRIPT_VERSION}"
    )
    log(
        f"dataset        : "
        f"{dataset_dir}"
    )
    log(
        f"output         : "
        f"{output_dir}"
    )
    log(
        f"base script    : "
        f"{base_path}"
    )
    log(
        f"device         : "
        f"{device}"
    )
    if device.type == "cuda":
        log(
            f"GPU            : "
            f"{torch.cuda.get_device_name(0)}"
        )
    log(
        f"max correction : "
        f"+/-{MAX_CORRECTION_DEG:.1f} deg"
    )
    log(
        f"score weights  : "
        f"{DIRECTION_SCORE_WEIGHTS}"
    )
    log(
        "[HARD PROPERTY] "
        "Wind-speed magnitude is frozen to Global Ridge."
    )
    log(
        "[FIREWALL] Tropical Atlantic and Stage 12K "
        "are NOT read."
    )
    log("")

    run_results_path = (
        output_dir
        / "candidate_run_results.csv"
    )
    baseline_rows = []
    audit_rows = []

    for held_out in (
        active_holdouts
    ):
        train_names = [
            m
            for m in DEVELOPMENT_MISSIONS
            if m != held_out
        ]

        train = base.concat_missions(
            [
                missions[m]
                for m in train_names
            ]
        )
        val = missions[
            held_out
        ]

        # -------------------------------------------------------------
        # Exact Stage-12C Global Ridge anchor.
        # -------------------------------------------------------------
        global_scaler = (
            base.fit_scalers(
                train["X"],
                train["y"],
                train["aw"],
            )
        )

        global_ridge = (
            base.fit_ridge(
                train["X"],
                train["y"],
                global_scaler,
            )
        )

        global_ridge_train_z = (
            base.ridge_predict_z(
                global_ridge,
                train["X"],
                global_scaler,
            )
        )
        global_ridge_val_z = (
            base.ridge_predict_z(
                global_ridge,
                val["X"],
                global_scaler,
            )
        )

        global_ridge_train_raw = (
            base.inverse_y(
                global_ridge_train_z,
                global_scaler,
            )
        )
        global_ridge_val_raw = (
            base.inverse_y(
                global_ridge_val_z,
                global_scaler,
            )
        )

        global_ridge_val_metrics = (
            base.evaluate_raw(
                val["y"],
                global_ridge_val_raw,
                val["aw"],
            )
        )

        baseline_rows.append({
            "method": "Global-Ridge",
            "held_out_mission": held_out,
            **metric_subset(
                global_ridge_val_metrics
            ),
        })

        # -------------------------------------------------------------
        # Local-frame input transform.
        # -------------------------------------------------------------
        train_local = (
            make_local_X(
                train["X"]
            )
        )
        val_local = (
            make_local_X(
                val["X"]
            )
        )

        local_x_scaler = (
            fit_local_x_scaler(
                train_local["X_local"]
            )
        )

        # -------------------------------------------------------------
        # Direction target & train-only weights.
        # -------------------------------------------------------------
        train_target = (
            make_direction_target_and_weights(
                train["y"],
                global_ridge_train_raw,
                normalize_weights=True,
            )
        )

        val_target = (
            make_direction_target_and_weights(
                val["y"],
                global_ridge_val_raw,
                normalize_weights=False,
            )
        )

        audit_rows.extend([
            {
                "held_out_mission": held_out,
                "split": "train",
                **train_local["audit"],
                **train_target["audit"],
            },
            {
                "held_out_mission": held_out,
                "split": "validation",
                **val_local["audit"],
                **val_target["audit"],
            },
        ])

        # -------------------------------------------------------------
        # Candidate A: Direction-Ridge.
        # -------------------------------------------------------------
        X_train_z = (
            standardize_local_X(
                train_local["X_local"],
                local_x_scaler,
            )
        )
        X_val_z = (
            standardize_local_X(
                val_local["X_local"],
                local_x_scaler,
            )
        )

        direction_ridge_X_train = (
            build_direction_ridge_features(
                X_train_z,
                global_ridge_train_raw,
                train_local["epar"],
                train_local["eperp"],
            )
        )
        direction_ridge_X_val = (
            build_direction_ridge_features(
                X_val_z,
                global_ridge_val_raw,
                val_local["epar"],
                val_local["eperp"],
            )
        )

        direction_ridge = (
            fit_direction_ridge(
                direction_ridge_X_train,
                train_target["target_norm"],
                train_target["weights"],
            )
        )

        (
            ridge_pred_norm,
            ridge_correction,
        ) = predict_direction_ridge(
            direction_ridge,
            direction_ridge_X_val,
        )

        direction_ridge_raw = (
            reconstruct_direction_corrected(
                global_ridge_val_raw,
                ridge_correction,
            )
        )

        direction_ridge_metrics = (
            base.evaluate_raw(
                val["y"],
                direction_ridge_raw,
                val["aw"],
            )
        )
        direction_ridge_metrics.update(
            ws_equality_audit(
                direction_ridge_raw,
                global_ridge_val_raw,
            )
        )
        direction_ridge_metrics.update(
            direction_correction_diagnostics(
                ridge_correction,
                val_target,
            )
        )

        (
            direction_ridge_score,
            direction_ridge_parts,
        ) = direction_selection_score(
            direction_ridge_metrics,
            global_ridge_val_metrics,
        )

        direction_ridge_row = {
            "candidate": "Direction-Ridge",
            "mode": "ridge",
            "held_out_mission": held_out,
            "seed": -1,
            "status": "completed_finite",
            "parameter_count": (
                direction_ridge_parameter_count(
                    direction_ridge
                )
            ),
            "best_epoch_one_based": 0,
            "selection_score": float(
                direction_ridge_score
            ),
            "elapsed_seconds": 0.0,
            **direction_ridge_metrics,
            **{
                f"score_{k}": v
                for k, v
                in direction_ridge_parts.items()
            },
            "history_path": "",
            "checkpoint_path": "",
        }

        append_csv(
            run_results_path,
            direction_ridge_row,
        )

        log("-" * 132)
        log(
            f"[{held_out}] "
            f"Ntrain={len(train['X']):,} "
            f"Nval={len(val['X']):,}"
        )
        log(
            "  Global Ridge    : "
            f"U={global_ridge_val_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={global_ridge_val_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={global_ridge_val_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={global_ridge_val_metrics['wind_direction_RMSE_deg']:.2f}"
        )
        log(
            "  Direction-Ridge : "
            f"score={direction_ridge_score:.5f} | "
            f"U={direction_ridge_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={direction_ridge_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={direction_ridge_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={direction_ridge_metrics['wind_direction_RMSE_deg']:.2f} | "
            f"dCorr={direction_ridge_metrics['direction_correction_target_corr']:+.3f}"
        )

        # -------------------------------------------------------------
        # Neural candidates.
        # -------------------------------------------------------------
        for candidate_name, mode in (
            active_neural.items()
        ):
            for seed in (
                active_seeds
            ):
                log(
                    f"  [TRAIN] "
                    f"{candidate_name} | "
                    f"seed={seed}"
                )

                result = (
                    train_neural_run(
                        base=base,
                        torch=torch,
                        nn=nn,
                        candidate_name=(
                            candidate_name
                        ),
                        mode=mode,
                        seed=seed,
                        held_out=held_out,
                        train_local=(
                            train_local
                        ),
                        val_local=(
                            val_local
                        ),
                        train_target=(
                            train_target
                        ),
                        val_target=(
                            val_target
                        ),
                        y_val=val["y"],
                        aw_val=val["aw"],
                        global_ridge_train_raw=(
                            global_ridge_train_raw
                        ),
                        global_ridge_val_raw=(
                            global_ridge_val_raw
                        ),
                        global_ridge_val_metrics=(
                            global_ridge_val_metrics
                        ),
                        local_x_scaler=(
                            local_x_scaler
                        ),
                        device=device,
                        output_dir=(
                            output_dir
                        ),
                        debug_fast=(
                            args.debug_fast
                        ),
                    )
                )

                append_csv(
                    run_results_path,
                    result,
                )

                log(
                    f"  [DONE] "
                    f"score={result['selection_score']:.5f} | "
                    f"best_ep={result['best_epoch_one_based']} | "
                    f"U={result['wind_U_RMSE_mps']:.4f} | "
                    f"V={result['wind_V_RMSE_mps']:.4f} | "
                    f"WD={result['wind_direction_RMSE_deg']:.2f} | "
                    f"dCorr={result['direction_correction_target_corr']:+.3f}"
                )

        del (
            train,
            val,
            global_scaler,
            global_ridge,
            global_ridge_train_z,
            global_ridge_val_z,
            global_ridge_train_raw,
            global_ridge_val_raw,
            train_local,
            val_local,
            local_x_scaler,
            train_target,
            val_target,
            X_train_z,
            X_val_z,
            direction_ridge_X_train,
            direction_ridge_X_val,
            direction_ridge,
            ridge_pred_norm,
            ridge_correction,
            direction_ridge_raw,
        )

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    baseline_df = (
        pd.DataFrame(
            baseline_rows
        )
    )
    baseline_path = (
        output_dir
        / "global_ridge_fold_results.csv"
    )
    baseline_df.to_csv(
        baseline_path,
        index=False,
        encoding="utf-8-sig",
    )

    audit_df = pd.DataFrame(
        audit_rows
    )
    audit_path = (
        output_dir
        / "direction_transform_target_audit.csv"
    )
    audit_df.to_csv(
        audit_path,
        index=False,
        encoding="utf-8-sig",
    )

    if args.debug_fast:
        log("")
        log(
            "[DONE] debug-fast completed. "
            "No scientific selection/decision freeze was written."
        )
        return 0

    # =================================================================
    # Full development-only summary.
    # =================================================================
    runs = pd.read_csv(
        run_results_path
    )

    expected_counts = {
        "Direction-Ridge": (
            len(DEVELOPMENT_MISSIONS)
        ),
        "Direction-MLP": (
            len(DEVELOPMENT_MISSIONS)
            * len(SCREENING_SEEDS)
        ),
        "Direction-GRU": (
            len(DEVELOPMENT_MISSIONS)
            * len(SCREENING_SEEDS)
        ),
    }

    for candidate, expected in (
        expected_counts.items()
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

        if got != expected:
            raise RuntimeError(
                f"{candidate}: expected "
                f"{expected} completed runs, "
                f"found {got}."
            )

    summary = pd.DataFrame([
        summarize_candidate(
            runs,
            candidate,
        )
        for candidate in [
            "Direction-Ridge",
            "Direction-MLP",
            "Direction-GRU",
        ]
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

    summary_path = (
        output_dir
        / "candidate_summary.csv"
    )
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    selected_name = str(
        summary.iloc[0][
            "candidate"
        ]
    )

    selected_runs = runs.loc[
        (
            runs["candidate"].astype(str)
            == selected_name
        )
        & (
            runs["status"].astype(str)
            == "completed_finite"
        )
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

    baseline_means = {
        k: float(
            baseline_df[k].mean()
        )
        for k in [
            "wind_U_RMSE_mps",
            "wind_V_RMSE_mps",
            "wind_vector_RMSE_mps",
            "wind_speed_RMSE_mps",
            "wind_direction_RMSE_deg",
            "AW_vector_RMSE_mps",
            "AWS_RMSE_mps",
        ]
    }

    selected_means = {
        k: float(
            selected_runs[k].mean()
        )
        for k in baseline_means
    }

    relative_improvement = {
        k: float(
            (
                baseline_means[k]
                - selected_means[k]
            )
            / max(
                baseline_means[k],
                EPS,
            )
        )
        for k in baseline_means
    }

    selected_score_mean = float(
        summary.iloc[0][
            "selection_score_mean"
        ]
    )

    mission_mean_scores = (
        selected_runs.groupby(
            "held_out_mission"
        )["selection_score"]
        .mean()
        .to_dict()
    )

    mission_wins = int(
        np.sum(
            np.asarray(
                list(
                    mission_mean_scores.values()
                ),
                dtype=float,
            )
            < 1.0
        )
    )

    ws_all_pass = boolean_series_all_true(
        selected_runs["WS_equality_pass"]
    )

    vector_relative_change = (
        -relative_improvement[
            "wind_vector_RMSE_mps"
        ]
    )
    # Positive vector_relative_change = worsening.

    meets_rule = bool(
        selected_score_mean
        < CONTINUE_SCORE_THRESHOLD
        and relative_improvement[
            "wind_V_RMSE_mps"
        ] > CONTINUE_V_IMPROVEMENT
        and relative_improvement[
            "wind_direction_RMSE_deg"
        ] > CONTINUE_WD_IMPROVEMENT
        and vector_relative_change
        <= MAX_VECTOR_WORSENING
        and mission_wins
        >= MIN_MISSION_WINS
        and ws_all_pass
    )

    decision = (
        "CONTINUE_DIRECTION_CORRECTION_TO_FINAL_INTEGRATION"
        if meets_rule
        else "STOP_DIRECTION_CORRECTION_ESCALATION"
    )

    selected_payload = {
        "stage": "12O",
        "script_version": SCRIPT_VERSION,
        "model_name": (
            "DirectionCorrectedRidge-NieData-v5"
        ),
        "selected_candidate": (
            selected_name
        ),
        "train_freeze": (
            asdict(FREEZE)
        ),
        "local_feature_names": (
            LOCAL_FEATURE_NAMES
        ),
        "direction_score_weights": (
            DIRECTION_SCORE_WEIGHTS
        ),
        "max_correction_deg": (
            MAX_CORRECTION_DEG
        ),
        "frame_min_speed_mps": (
            FRAME_MIN_SPEED_MPS
        ),
        "training_weight_speed_scale_mps": (
            DIRECTION_WEIGHT_SPEED_SCALE_MPS
        ),
        "continuation_rule": {
            "score_lt": (
                CONTINUE_SCORE_THRESHOLD
            ),
            "V_improvement_gt": (
                CONTINUE_V_IMPROVEMENT
            ),
            "WD_improvement_gt": (
                CONTINUE_WD_IMPROVEMENT
            ),
            "vector_worsening_le": (
                MAX_VECTOR_WORSENING
            ),
            "mission_wins_ge": (
                MIN_MISSION_WINS
            ),
            "ws_equality_required": True,
        },
        "selected_selection_score_mean": (
            selected_score_mean
        ),
        "selected_mission_mean_scores": (
            mission_mean_scores
        ),
        "mission_wins_vs_ridge": (
            mission_wins
        ),
        "global_ridge_development_means": (
            baseline_means
        ),
        "selected_development_means": (
            selected_means
        ),
        "relative_improvement_vs_global_ridge": (
            relative_improvement
        ),
        "ws_equality_all_pass": (
            ws_all_pass
        ),
        "meets_continue_rule": (
            meets_rule
        ),
        "decision": (
            decision
        ),
        "Tropical_Atlantic_used": False,
        "Stage_12K_used": False,
        "scientific_note": (
            "All fitting, epoch selection, candidate selection, and "
            "continuation decisions use Antarctic, Atlantic, and "
            "West Coast development-only LOMO. The mechanism must "
            "not be altered using Tropical Atlantic."
        ),
    }

    selected_json_path = (
        output_dir
        / "selected_config.json"
    )
    save_json(
        selected_json_path,
        selected_payload,
    )

    report_path = (
        output_dir
        / "12O_REPORT.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "12O Direction-Corrected Ridge Wind "
            "Development-Only LOMO\n"
        )
        f.write(
            "=" * 132 + "\n\n"
        )

        f.write(
            "SCIENTIFIC FIREWALL\n"
        )
        f.write(
            "-" * 132 + "\n"
        )
        f.write(
            "Antarctic / Atlantic / West Coast: ACCESSED\n"
        )
        f.write(
            "Tropical Atlantic: NOT ACCESSED\n"
        )
        f.write(
            "Stage 12K outputs: NOT ACCESSED\n\n"
        )

        f.write(
            "MECHANISM\n"
        )
        f.write(
            "-" * 132 + "\n"
        )
        f.write(
            "Ridge wind-speed magnitude is frozen exactly.\n"
        )
        f.write(
            "Only wind-vector direction may be rotated.\n"
        )
        f.write(
            f"Maximum correction: +/-{MAX_CORRECTION_DEG:.1f} deg\n\n"
        )

        f.write(
            "CANDIDATE SUMMARY\n"
        )
        f.write(
            "-" * 132 + "\n"
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
            "SELECTED\n"
        )
        f.write(
            "-" * 132 + "\n"
        )
        f.write(
            f"candidate: {selected_name}\n"
        )
        f.write(
            f"mean DirectionScore: "
            f"{selected_score_mean:.6f}\n"
        )
        f.write(
            "relative improvement vs Global Ridge:\n"
        )

        for k, v in (
            relative_improvement.items()
        ):
            f.write(
                f"  {k}: "
                f"{100.0*v:+.4f}%\n"
            )

        f.write(
            "\nMean score by held-out mission:\n"
        )
        for mission, score in (
            mission_mean_scores.items()
        ):
            f.write(
                f"  {mission}: "
                f"{score:.6f}\n"
            )

        f.write(
            "\nPREDECLARED DECISION RULE\n"
        )
        f.write(
            "-" * 132 + "\n"
        )
        f.write(
            f"score < {CONTINUE_SCORE_THRESHOLD:.3f}\n"
        )
        f.write(
            f"V improvement > "
            f"{100.0*CONTINUE_V_IMPROVEMENT:.1f}%\n"
        )
        f.write(
            f"WD improvement > "
            f"{100.0*CONTINUE_WD_IMPROVEMENT:.1f}%\n"
        )
        f.write(
            f"vector worsening <= "
            f"{100.0*MAX_VECTOR_WORSENING:.1f}%\n"
        )
        f.write(
            f"mission wins >= "
            f"{MIN_MISSION_WINS}\n"
        )
        f.write(
            "WS equality audit must pass\n"
        )
        f.write(
            f"meets rule: "
            f"{meets_rule}\n"
        )
        f.write(
            f"decision: "
            f"{decision}\n"
        )

    log("")
    log(
        "=" * 132
    )
    log(
        "12O CANDIDATE SUMMARY"
    )
    log(
        "=" * 132
    )
    log(
        summary.to_string(
            index=False
        )
    )
    log("")
    log(
        f"[SELECTED] "
        f"{selected_name}"
    )
    log(
        f"Mean DirectionScore = "
        f"{selected_score_mean:.6f}"
    )
    log(
        "Development relative improvement vs Global Ridge: "
        + " | ".join(
            f"{k}={100.0*v:+.3f}%"
            for k, v
            in relative_improvement.items()
        )
    )
    log(
        "Selected direction-correction corr mean = "
        f"{selected_runs['direction_correction_target_corr'].mean():+.4f}"
    )
    log(
        "Mission mean scores: "
        + " | ".join(
            f"{m}={s:.5f}"
            for m, s
            in mission_mean_scores.items()
        )
    )
    log(
        f"WS equality audit all pass: "
        f"{ws_all_pass}"
    )
    log("")
    log(
        f"[DECISION] "
        f"{decision}"
    )
    log(
        "Rule: score < 0.980 AND "
        "V improvement > 2% AND "
        "WD improvement > 2% AND "
        "vector worsening <= 0.5% AND "
        ">=2 mission wins AND WS equality pass"
    )
    log("")
    log(
        f"[SAVED] "
        f"{run_results_path}"
    )
    log(
        f"[SAVED] "
        f"{summary_path}"
    )
    log(
        f"[SAVED] "
        f"{selected_runs_path}"
    )
    log(
        f"[SAVED] "
        f"{selected_json_path}"
    )
    log(
        f"[SAVED] "
        f"{audit_path}"
    )
    log(
        f"[SAVED] "
        f"{report_path}"
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
