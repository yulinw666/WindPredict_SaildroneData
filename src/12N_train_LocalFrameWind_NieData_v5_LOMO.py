# -*- coding: utf-8 -*-
r"""
12N_train_LocalFrameWind_NieData_v5_LOMO.py

Stage 12N
=========
Development-only leave-one-mission-out (LOMO) screening of a rotation-aware
local-frame wind representation for the Stage-12G Nie point-sampled benchmark.

Scientific motivation
---------------------
Stages 12L and 12M showed that directly learning the remaining global-East/North
Ridge wind residual does not transfer across missions. The next hypothesis is
therefore REPRESENTATIONAL rather than architectural:

    Similar physical wind changes can appear in different U/V components when
    the prevailing wind direction differs between missions.

For each sample, define a local orthonormal frame from the LAST observed wind:

    e_parallel = W_t / ||W_t||
    e_perp     = [-e_parallel_N, e_parallel_E]

For very weak last-step wind, the recent-context mean wind is used as a
predeclared fallback; if that is also degenerate, the East axis is used.

The input sequence is transformed sample-by-sample:

    [U, V]                 -> [W_parallel, W_perp]
    [COG_sin, COG_cos]     -> [COG_parallel, COG_perp]

while T, RH, SOG, WING_ANGLE_sin, WING_ANGLE_cos are preserved.

The prediction target is the FUTURE WIND INCREMENT in that current local frame:

    dW = W_{t+1} - W_t

    target_parallel = dW dot e_parallel
    target_perp     = dW dot e_perp

Predictions are rotated back to East/North before ALL reported benchmark
metrics are computed.

Frozen feature protocol
-----------------------
Original Stage-12C/12G:
    X: [N, 6, 9]
       [U, V, T, RH, SOG,
        COG_sin, COG_cos,
        WING_ANGLE_sin, WING_ANGLE_cos]

    y: [N, 1, 4]
       [U, V, VESSEL_E, VESSEL_N]

Local-frame X remains [N, 6, 9]:
       [W_parallel, W_perp, T, RH, SOG,
        COG_parallel, COG_perp,
        WING_ANGLE_sin, WING_ANGLE_cos]

Development-only candidates
---------------------------
A) LocalFrame-Ridge
   - Ridge(alpha=1.0) on flattened standardized local-frame history
   - predicts standardized [dW_parallel, dW_perp]
   - tests the representation change with minimal model complexity

B) LocalFrame-MLP
   - compact MLP on the same flattened standardized local-frame history
   - direct local-increment prediction

C) LocalFrame-GRU
   - compact GRU on the standardized local-frame sequence
   - direct local-increment prediction

Global Stage-12C Ridge is recomputed in every fold and used ONLY as the
reference baseline. For fair apparent-wind diagnostics, all local-frame wind
candidates retain the SAME global-Ridge vessel prediction; therefore any AW
difference is caused by the wind prediction only.

Selection score
---------------
The predeclared Stage-12M wind score is retained unchanged:

    WindScore =
        0.25 * U_RMSE / GlobalRidge_U_RMSE
      + 0.35 * V_RMSE / GlobalRidge_V_RMSE
      + 0.25 * WS_RMSE / GlobalRidge_WS_RMSE
      + 0.15 * WD_RMSE / GlobalRidge_WD_RMSE

Global Ridge = 1.0. Lower is better.

Predeclared continuation rule
-----------------------------
A local-frame mechanism is considered strong enough to justify integration
only if, on development-only LOMO:

    selected mean WindScore < 0.99
    AND mean V_RMSE improvement versus Global Ridge > 1%

Otherwise, stop trying to force a nonlinear true-wind correction and retain
the strong global Ridge wind anchor.

SCIENTIFIC FIREWALL
-------------------
This script reads ONLY:
    track_J_joint_compatible/Antarctic.npz
    track_J_joint_compatible/Atlantic.npz
    track_J_joint_compatible/West_Coast.npz

It NEVER reads:
    Tropical_Atlantic_TEST.npz
    Stage-12K outputs
    any held-out Tropical Atlantic metric

LOMO folds
----------
Holdout Antarctic : train Atlantic + West Coast
Holdout Atlantic  : train Antarctic + West Coast
Holdout West Coast: train Antarctic + Atlantic

No validation-mission sample enters its fold scaler or model fitting.

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\12N_train_LocalFrameWind_NieData_v5_LOMO.py" --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12N_LocalFrameWind_NieData_v5_LOMO_point_v0_1"

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


SCRIPT_VERSION = "0.1.0-LocalFrameWind-NieData-v5-LOMO"

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
    / "12N_LocalFrameWind_NieData_v5_LOMO_point_v0_1"
)

DEVELOPMENT_MISSIONS = ["Antarctic", "Atlantic", "West Coast"]
SCREENING_SEEDS = [500043, 501052, 502061]

GLOBAL_RIDGE_ALPHA = 1.0
LOCAL_RIDGE_ALPHA = 1.0

# Stage-12C frozen feature order.
IDX_U = 0
IDX_V = 1
IDX_T = 2
IDX_RH = 3
IDX_SOG = 4
IDX_COG_SIN = 5   # East component of course unit vector
IDX_COG_COS = 6   # North component of course unit vector
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
LOCAL_TARGET_NAMES = ["dW_parallel", "dW_perp"]

WIND_SELECTION_WEIGHTS = {
    "U": 0.25,
    "V": 0.35,
    "WS": 0.25,
    "WD": 0.15,
}

# Predeclared scientific continuation rule.
CONTINUE_SCORE_THRESHOLD = 0.99
CONTINUE_V_IMPROVEMENT_FRACTION = 0.01

EPS = 1e-12
FRAME_MIN_SPEED_MPS = 0.50


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

    use_amp: bool = True
    train_shuffle: bool = False


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

    name = "stage12c_base_for_12n"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for: {path}")

    mod = importlib.util.module_from_spec(spec)

    # Important for Python 3.11 dataclass/type introspection.
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise

    return mod, path


def audit_base_protocol(base):
    feature_names = list(base.FEATURE_NAMES)
    target_names = list(base.TARGET_NAMES)

    if feature_names != EXPECTED_FEATURE_NAMES:
        raise RuntimeError(
            "Unexpected Stage-12C feature protocol.\n"
            f"Expected: {EXPECTED_FEATURE_NAMES}\n"
            f"Found   : {feature_names}"
        )

    if target_names != EXPECTED_TARGET_NAMES:
        raise RuntimeError(
            "Unexpected Stage-12C target protocol.\n"
            f"Expected: {EXPECTED_TARGET_NAMES}\n"
            f"Found   : {target_names}"
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
        raise ValueError("All batch arrays must have the same first dimension.")

    for start in range(0, n, int(batch_size)):
        stop = min(start + int(batch_size), n)
        yield tuple(a[start:stop] for a in arrays)


# =============================================================================
# Local-frame transform
# =============================================================================

def build_local_frame(X_raw: np.ndarray):
    """
    Build one local wind frame per sample.

    Coordinate convention:
        physical vectors are [East, North].

    e_parallel:
        direction of the last observed wind vector.

    e_perp:
        +90 deg counterclockwise in the mathematical East/North plane:
        [-e_parallel_N, e_parallel_E]

    Weak-wind fallback:
        last wind -> recent mean wind -> fixed East axis.
    """
    X = np.asarray(X_raw, dtype=np.float64)

    if X.ndim != 3 or X.shape[1:] != (6, 9):
        raise ValueError(
            f"Expected X [N,6,9], got {X.shape}"
        )

    last_w = X[:, -1, [IDX_U, IDX_V]]
    last_speed = np.linalg.norm(last_w, axis=1)

    recent_mean_w = np.mean(
        X[:, :, [IDX_U, IDX_V]],
        axis=1,
    )
    recent_speed = np.linalg.norm(recent_mean_w, axis=1)

    frame_w = last_w.copy()
    fallback_recent = last_speed < FRAME_MIN_SPEED_MPS

    frame_w[fallback_recent] = recent_mean_w[fallback_recent]

    fallback_east = (
        fallback_recent
        & (recent_speed < FRAME_MIN_SPEED_MPS)
    )
    frame_w[fallback_east, 0] = 1.0
    frame_w[fallback_east, 1] = 0.0

    norm = np.linalg.norm(frame_w, axis=1)
    norm = np.maximum(norm, EPS)

    epar = frame_w / norm[:, None]
    eperp = np.stack(
        [-epar[:, 1], epar[:, 0]],
        axis=1,
    )

    orth_err = np.max(
        np.abs(np.sum(epar * eperp, axis=1))
    )
    unit_err = max(
        float(np.max(np.abs(np.linalg.norm(epar, axis=1) - 1.0))),
        float(np.max(np.abs(np.linalg.norm(eperp, axis=1) - 1.0))),
    )

    audit = {
        "n": int(len(X)),
        "last_speed_below_threshold_fraction": float(
            np.mean(fallback_recent)
        ),
        "fixed_east_fallback_fraction": float(
            np.mean(fallback_east)
        ),
        "orthogonality_max_abs_error": float(orth_err),
        "unit_norm_max_abs_error": float(unit_err),
        "frame_min_speed_mps": float(FRAME_MIN_SPEED_MPS),
    }

    return (
        epar.astype(np.float32),
        eperp.astype(np.float32),
        last_w.astype(np.float32),
        audit,
    )


def rotate_vectors_to_local(
    vec_en: np.ndarray,
    epar: np.ndarray,
    eperp: np.ndarray,
):
    """
    vec_en: [N,T,2] or [N,2].
    epar, eperp: [N,2].
    """
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


def rotate_vectors_to_global(
    vec_local: np.ndarray,
    epar: np.ndarray,
    eperp: np.ndarray,
):
    """
    vec_local: [N,2] = [parallel, perp]
    Return East/North [N,2].
    """
    q = np.asarray(vec_local, dtype=np.float64)
    ep = np.asarray(epar, dtype=np.float64)
    eq = np.asarray(eperp, dtype=np.float64)

    out = (
        q[:, 0:1] * ep
        + q[:, 1:2] * eq
    )
    return out.astype(np.float32)


def make_local_dataset(X_raw: np.ndarray, y_raw: np.ndarray):
    """
    Transform one mission/fold array into:
        X_local: [N,6,9]
        dW_local: [N,2]
    plus frame metadata needed for reconstruction.
    """
    X = np.asarray(X_raw, dtype=np.float32)
    y = np.asarray(y_raw, dtype=np.float32)

    epar, eperp, current_w, frame_audit = build_local_frame(X)

    X_local = np.empty_like(X, dtype=np.float32)

    wind_hist = X[:, :, [IDX_U, IDX_V]]
    wind_local = rotate_vectors_to_local(
        wind_hist, epar, eperp
    )

    X_local[:, :, 0:2] = wind_local
    X_local[:, :, 2] = X[:, :, IDX_T]
    X_local[:, :, 3] = X[:, :, IDX_RH]
    X_local[:, :, 4] = X[:, :, IDX_SOG]

    # COG_sin = East component, COG_cos = North component.
    cog_en = np.stack(
        [
            X[:, :, IDX_COG_SIN],
            X[:, :, IDX_COG_COS],
        ],
        axis=-1,
    )
    cog_local = rotate_vectors_to_local(
        cog_en, epar, eperp
    )
    X_local[:, :, 5:7] = cog_local

    X_local[:, :, 7] = X[:, :, IDX_WING_SIN]
    X_local[:, :, 8] = X[:, :, IDX_WING_COS]

    future_w = y[:, 0, 0:2]
    dW_en = future_w - current_w
    dW_local = rotate_vectors_to_local(
        dW_en, epar, eperp
    )

    if not np.isfinite(X_local).all():
        raise FloatingPointError("Non-finite X_local.")
    if not np.isfinite(dW_local).all():
        raise FloatingPointError("Non-finite dW_local.")

    # Reconstruction audit.
    dW_recon = rotate_vectors_to_global(
        dW_local, epar, eperp
    )
    recon_err = float(
        np.max(np.abs(dW_recon - dW_en))
    )
    frame_audit["target_rotation_reconstruction_max_abs_error"] = recon_err

    return {
        "X_local": X_local,
        "dW_local": dW_local,
        "epar": epar,
        "eperp": eperp,
        "current_w": current_w,
        "frame_audit": frame_audit,
    }


# =============================================================================
# Fold-only standardization
# =============================================================================

def fit_local_scaler(X_local: np.ndarray, dW_local: np.ndarray):
    X = np.asarray(X_local, dtype=np.float64)
    y = np.asarray(dW_local, dtype=np.float64)

    x_mean = np.mean(X, axis=(0, 1))
    x_std = np.std(X, axis=(0, 1), ddof=0)

    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0, ddof=0)

    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)

    return {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
    }


def standardize_local_X(X_local, scaler):
    return (
        (
            np.asarray(X_local, dtype=np.float32)
            - scaler["x_mean"][None, None, :]
        )
        / scaler["x_std"][None, None, :]
    ).astype(np.float32)


def standardize_local_y(dW_local, scaler):
    return (
        (
            np.asarray(dW_local, dtype=np.float32)
            - scaler["y_mean"][None, :]
        )
        / scaler["y_std"][None, :]
    ).astype(np.float32)


def inverse_local_y(yz, scaler):
    return (
        np.asarray(yz, dtype=np.float32)
        * scaler["y_std"][None, :]
        + scaler["y_mean"][None, :]
    ).astype(np.float32)


# =============================================================================
# Prediction reconstruction + metrics
# =============================================================================

def build_full_prediction(
    *,
    base,
    local_pred_z,
    local_scaler,
    local_meta,
    global_ridge_raw,
):
    """
    Keep the fold-specific global Ridge vessel prediction fixed.
    Replace only wind U/V with local-frame candidate wind.
    """
    dW_local_pred = inverse_local_y(
        local_pred_z, local_scaler
    )
    dW_en_pred = rotate_vectors_to_global(
        dW_local_pred,
        local_meta["epar"],
        local_meta["eperp"],
    )

    future_w_pred = (
        local_meta["current_w"] + dW_en_pred
    )

    pred_raw = np.asarray(
        global_ridge_raw, dtype=np.float32
    ).copy()
    pred_raw[:, 0, 0:2] = future_w_pred

    return pred_raw


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


def local_delta_diagnostics(
    pred_z,
    true_local_raw,
    scaler,
):
    pred_local = inverse_local_y(pred_z, scaler)
    true_local = np.asarray(
        true_local_raw, dtype=np.float32
    )

    return {
        "delta_parallel_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    (pred_local[:, 0] - true_local[:, 0]) ** 2
                )
            )
        ),
        "delta_perp_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    (pred_local[:, 1] - true_local[:, 1]) ** 2
                )
            )
        ),
        "delta_parallel_corr": corrcoef_safe(
            pred_local[:, 0], true_local[:, 0]
        ),
        "delta_perp_corr": corrcoef_safe(
            pred_local[:, 1], true_local[:, 1]
        ),
        "pred_delta_parallel_RMS_mps": float(
            np.sqrt(np.mean(pred_local[:, 0] ** 2))
        ),
        "pred_delta_perp_RMS_mps": float(
            np.sqrt(np.mean(pred_local[:, 1] ** 2))
        ),
        "true_delta_parallel_RMS_mps": float(
            np.sqrt(np.mean(true_local[:, 0] ** 2))
        ),
        "true_delta_perp_RMS_mps": float(
            np.sqrt(np.mean(true_local[:, 1] ** 2))
        ),
    }


def wind_selection_score(metrics, global_ridge_metrics):
    eps = 1e-12

    u = metrics["wind_U_RMSE_mps"] / max(
        global_ridge_metrics["wind_U_RMSE_mps"],
        eps,
    )
    v = metrics["wind_V_RMSE_mps"] / max(
        global_ridge_metrics["wind_V_RMSE_mps"],
        eps,
    )
    ws = metrics["wind_speed_RMSE_mps"] / max(
        global_ridge_metrics["wind_speed_RMSE_mps"],
        eps,
    )
    wd = metrics["wind_direction_RMSE_deg"] / max(
        global_ridge_metrics["wind_direction_RMSE_deg"],
        eps,
    )

    score = (
        WIND_SELECTION_WEIGHTS["U"] * u
        + WIND_SELECTION_WEIGHTS["V"] * v
        + WIND_SELECTION_WEIGHTS["WS"] * ws
        + WIND_SELECTION_WEIGHTS["WD"] * wd
    )

    return float(score), {
        "U_ratio": float(u),
        "V_ratio": float(v),
        "WS_ratio": float(ws),
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
# LocalFrame-Ridge
# =============================================================================

def fit_local_ridge(
    X_train_local,
    dW_train_local,
    scaler,
):
    Xz = standardize_local_X(
        X_train_local, scaler
    )
    yz = standardize_local_y(
        dW_train_local, scaler
    )

    model = Ridge(alpha=LOCAL_RIDGE_ALPHA)
    model.fit(
        Xz.reshape(len(Xz), -1),
        yz,
    )
    return model


def predict_local_ridge_z(
    model,
    X_local,
    scaler,
):
    Xz = standardize_local_X(
        X_local, scaler
    )
    pred = model.predict(
        Xz.reshape(len(Xz), -1)
    )
    return np.asarray(pred, dtype=np.float32)


def ridge_parameter_count(model):
    count = int(np.asarray(model.coef_).size)
    count += int(np.asarray(model.intercept_).size)
    return count


# =============================================================================
# Neural local-frame models
# =============================================================================

def build_model_class(torch, nn, mode: str):
    feature_count = len(LOCAL_FEATURE_NAMES)

    if mode == "mlp":
        class LocalFrameMLP(nn.Module):
            def __init__(self):
                super().__init__()

                self.backbone = nn.Sequential(
                    nn.Linear(
                        6 * feature_count,
                        FREEZE.mlp_hidden_1,
                    ),
                    nn.LayerNorm(FREEZE.mlp_hidden_1),
                    nn.GELU(),
                    nn.Dropout(FREEZE.dropout),
                    nn.Linear(
                        FREEZE.mlp_hidden_1,
                        FREEZE.mlp_hidden_2,
                    ),
                    nn.GELU(),
                    nn.Dropout(FREEZE.dropout),
                )
                self.out = nn.Linear(
                    FREEZE.mlp_hidden_2, 2
                )

                # Zero standardized target initially = fold-train mean increment.
                nn.init.zeros_(self.out.weight)
                nn.init.zeros_(self.out.bias)

            def forward(self, x):
                return self.out(
                    self.backbone(
                        x.flatten(start_dim=1)
                    )
                )

        return LocalFrameMLP

    if mode == "gru":
        class LocalFrameGRU(nn.Module):
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
                        FREEZE.gru_hidden,
                        FREEZE.dense_hidden,
                    ),
                    nn.LayerNorm(FREEZE.dense_hidden),
                    nn.GELU(),
                    nn.Dropout(FREEZE.dropout),
                )
                self.out = nn.Linear(
                    FREEZE.dense_hidden, 2
                )

                nn.init.zeros_(self.out.weight)
                nn.init.zeros_(self.out.bias)

            def forward(self, x):
                seq, _ = self.gru(x)
                h = seq[:, -1, :]
                return self.out(
                    self.backbone(h)
                )

        return LocalFrameGRU

    raise ValueError(f"Unknown mode: {mode}")


def evaluate_neural(
    *,
    base,
    torch,
    model,
    X_local,
    dW_local_true,
    local_scaler,
    local_meta,
    y_true,
    apparent_ref,
    global_ridge_raw,
    device,
    amp_enabled,
):
    model.eval()

    Xz = standardize_local_X(
        X_local, local_scaler
    )

    pred_chunks = []

    with torch.no_grad():
        for (xb_np,) in sequential_batches(
            Xz,
            batch_size=FREEZE.batch_size,
        ):
            xb = torch.from_numpy(xb_np).to(
                device, non_blocking=True
            )

            with amp_context(
                torch, amp_enabled, device
            ):
                pred_z = model(xb)

            pred_chunks.append(
                pred_z.detach().float().cpu().numpy()
            )

    pred_z = np.concatenate(
        pred_chunks, axis=0
    ).astype(np.float32)

    pred_raw = build_full_prediction(
        base=base,
        local_pred_z=pred_z,
        local_scaler=local_scaler,
        local_meta=local_meta,
        global_ridge_raw=global_ridge_raw,
    )

    metrics = base.evaluate_raw(
        y_true, pred_raw, apparent_ref
    )
    metrics.update(
        local_delta_diagnostics(
            pred_z,
            dW_local_true,
            local_scaler,
        )
    )

    return metrics, pred_z, pred_raw


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
    y_train,
    y_val,
    aw_val,
    global_ridge_val_raw,
    global_ridge_val_metrics,
    local_scaler,
    device,
    output_dir,
    debug_fast,
):
    seed_everything(torch, seed)

    ModelClass = build_model_class(
        torch, nn, mode
    )
    model = ModelClass().to(device)
    parameter_count = count_parameters(model)

    X_train_z = standardize_local_X(
        train_local["X_local"],
        local_scaler,
    )
    y_train_z = standardize_local_y(
        train_local["dW_local"],
        local_scaler,
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
        FREEZE.use_amp and device.type == "cuda"
    )
    grad_scaler = create_grad_scaler(
        torch, amp_enabled
    )

    # Pre-training evaluation is allowed and is not a model-selection leak:
    # it uses the same held-out DEVELOPMENT mission as every later epoch.
    initial_metrics, _, _ = evaluate_neural(
        base=base,
        torch=torch,
        model=model,
        X_local=val_local["X_local"],
        dW_local_true=val_local["dW_local"],
        local_scaler=local_scaler,
        local_meta=val_local,
        y_true=y_val,
        apparent_ref=aw_val,
        global_ridge_raw=global_ridge_val_raw,
        device=device,
        amp_enabled=amp_enabled,
    )
    best_score, best_parts = wind_selection_score(
        initial_metrics,
        global_ridge_val_metrics,
    )
    best_epoch = 0
    best_state = state_dict_cpu(model)
    best_metrics = dict(initial_metrics)
    best_parts_saved = dict(best_parts)

    history = [{
        "candidate": candidate_name,
        "held_out_mission": held_out,
        "seed": int(seed),
        "epoch_one_based": 0,
        "learning_rate": float(FREEZE.learning_rate),
        "selection_score": float(best_score),
        "checkpoint_improved": True,
        "train_mse_z": np.nan,
        **{
            f"val_{k}": v
            for k, v in initial_metrics.items()
        },
        **{
            f"score_{k}": v
            for k, v in best_parts.items()
        },
    }]

    max_epochs = (
        8 if debug_fast
        else FREEZE.max_epochs
    )
    patience_limit = (
        4 if debug_fast
        else FREEZE.early_stop_patience
    )

    patience = 0
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()

        total_mse = 0.0
        n_seen = 0

        for xb_np, yb_np in sequential_batches(
            X_train_z,
            y_train_z,
            batch_size=FREEZE.batch_size,
        ):
            xb = torch.from_numpy(xb_np).to(
                device, non_blocking=True
            )
            yb = torch.from_numpy(yb_np).to(
                device, non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            with amp_context(
                torch, amp_enabled, device
            ):
                pred_z = model(xb)
                loss = torch.mean(
                    (pred_z - yb) ** 2
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite LocalFrame neural loss."
                )

            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    FREEZE.grad_clip_norm,
                )
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    FREEZE.grad_clip_norm,
                )
                optimizer.step()

            bn = len(xb_np)
            total_mse += float(
                loss.detach().float().cpu().item()
            ) * bn
            n_seen += bn

        val_metrics, _, _ = evaluate_neural(
            base=base,
            torch=torch,
            model=model,
            X_local=val_local["X_local"],
            dW_local_true=val_local["dW_local"],
            local_scaler=local_scaler,
            local_meta=val_local,
            y_true=y_val,
            apparent_ref=aw_val,
            global_ridge_raw=global_ridge_val_raw,
            device=device,
            amp_enabled=amp_enabled,
        )
        score, parts = wind_selection_score(
            val_metrics,
            global_ridge_val_metrics,
        )

        improved = (
            score < best_score - 1e-8
        )

        if improved:
            best_score = float(score)
            best_epoch = int(epoch)
            best_state = state_dict_cpu(model)
            best_metrics = dict(val_metrics)
            best_parts_saved = dict(parts)
            patience = 0
        else:
            patience += 1

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )
        scheduler.step(score)

        history.append({
            "candidate": candidate_name,
            "held_out_mission": held_out,
            "seed": int(seed),
            "epoch_one_based": int(epoch),
            "learning_rate": current_lr,
            "selection_score": float(score),
            "checkpoint_improved": bool(improved),
            "train_mse_z": (
                total_mse / max(n_seen, 1)
            ),
            **{
                f"val_{k}": v
                for k, v in val_metrics.items()
            },
            **{
                f"score_{k}": v
                for k, v in parts.items()
            },
        })

        log(
            f"      ep={epoch:03d} | "
            f"score={score:.5f} | "
            f"U={val_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={val_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={val_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={val_metrics['wind_direction_RMSE_deg']:.2f} | "
            f"dCorr=({val_metrics['delta_parallel_corr']:+.3f},"
            f"{val_metrics['delta_perp_corr']:+.3f})"
            + (" *" if improved else "")
        )

        if patience >= patience_limit:
            log(
                f"      early stop after {patience} "
                "epochs without improvement."
            )
            break

    elapsed = time.perf_counter() - started

    model.load_state_dict(
        best_state, strict=True
    )
    confirmed, _, _ = evaluate_neural(
        base=base,
        torch=torch,
        model=model,
        X_local=val_local["X_local"],
        dW_local_true=val_local["dW_local"],
        local_scaler=local_scaler,
        local_meta=val_local,
        y_true=y_val,
        apparent_ref=aw_val,
        global_ridge_raw=global_ridge_val_raw,
        device=device,
        amp_enabled=amp_enabled,
    )
    confirmed_score, confirmed_parts = (
        wind_selection_score(
            confirmed,
            global_ridge_val_metrics,
        )
    )

    if abs(
        confirmed_score - best_score
    ) > 1e-6:
        raise RuntimeError(
            "Best-checkpoint score re-evaluation mismatch."
        )

    histories_dir = output_dir / "histories"
    checkpoints_dir = output_dir / "checkpoints"
    histories_dir.mkdir(
        parents=True, exist_ok=True
    )
    checkpoints_dir.mkdir(
        parents=True, exist_ok=True
    )

    tag = (
        f"{candidate_name}"
        f"__holdout_{held_out.replace(' ', '_')}"
        f"__seed_{seed}"
    )

    history_path = (
        histories_dir / f"{tag}.csv"
    )
    checkpoint_path = (
        checkpoints_dir / f"{tag}.pt"
    )

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
        encoding="utf-8-sig",
    )

    torch.save({
        "stage": "12N",
        "script_version": SCRIPT_VERSION,
        "candidate": candidate_name,
        "mode": mode,
        "held_out_mission": held_out,
        "seed": int(seed),
        "parameter_count": int(parameter_count),
        "best_epoch_one_based": int(best_epoch),
        "selection_score": float(confirmed_score),
        "validation_metrics": confirmed,
        "score_parts": confirmed_parts,
        "state_dict": best_state,
        "local_scaler": local_scaler,
        "local_feature_names": LOCAL_FEATURE_NAMES,
        "local_target_names": LOCAL_TARGET_NAMES,
        "wind_selection_weights": WIND_SELECTION_WEIGHTS,
        "frame_min_speed_mps": FRAME_MIN_SPEED_MPS,
        "Tropical_Atlantic_used": False,
        "Stage_12K_used": False,
    }, checkpoint_path)

    result = {
        "candidate": candidate_name,
        "mode": mode,
        "held_out_mission": held_out,
        "seed": int(seed),
        "status": "completed_finite",
        "parameter_count": int(parameter_count),
        "best_epoch_one_based": int(best_epoch),
        "selection_score": float(confirmed_score),
        "elapsed_seconds": float(elapsed),
        **confirmed,
        **{
            f"score_{k}": v
            for k, v in confirmed_parts.items()
        },
        "history_path": str(history_path),
        "checkpoint_path": str(checkpoint_path),
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
# Reporting helpers
# =============================================================================

def append_csv(path: Path, row: dict):
    new = pd.DataFrame([row])

    if path.exists():
        old = pd.read_csv(path)

        key_cols = [
            "candidate",
            "held_out_mission",
            "seed",
        ]
        if all(k in old.columns for k in key_cols):
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
            old = old.loc[~mask].copy()

        new = pd.concat(
            [old, new],
            ignore_index=True,
        )

    new.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


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
        "delta_parallel_RMSE_mps",
        "delta_perp_RMSE_mps",
        "delta_parallel_corr",
        "delta_perp_corr",
    ]:
        vals = sub[col].to_numpy(dtype=float)
        row[f"{col}_mean"] = float(
            np.nanmean(vals)
        )
        row[f"{col}_std"] = float(
            np.nanstd(vals, ddof=0)
        )

    # Per-mission mean score, then count true cross-mission wins.
    mission_scores = (
        sub.groupby("held_out_mission")[
            "selection_score"
        ]
        .mean()
    )
    row["mission_wins_vs_global_ridge"] = int(
        np.sum(mission_scores.to_numpy() < 1.0)
    )
    row["mission_strong_wins_score_lt_0_99"] = int(
        np.sum(
            mission_scores.to_numpy()
            < CONTINUE_SCORE_THRESHOLD
        )
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
            "one seed, LocalFrame-GRU, <=8 epochs."
        ),
    )

    args = parser.parse_args()

    project_root = args.project_root
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    base, base_path = load_base_module(
        project_root
    )
    audit_base_protocol(base)

    torch, nn, _ = base.import_torch()
    base.configure_cuda(torch)

    device = (
        torch.device("cpu")
        if args.force_cpu
        or not torch.cuda.is_available()
        else torch.device("cuda")
    )

    dev_paths = base.explicit_development_paths(
        dataset_dir
    )

    # Hard scientific firewall.
    for mission, path in dev_paths.items():
        if "tropical" in str(path).lower():
            raise RuntimeError(
                "FIREWALL VIOLATION: development path "
                f"contains 'tropical': {mission} -> {path}"
            )

    missions = {
        m: base.load_development_mission(
            dev_paths[m], m
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
        {"LocalFrame-GRU": "gru"}
        if args.debug_fast
        else {
            "LocalFrame-MLP": "mlp",
            "LocalFrame-GRU": "gru",
        }
    )

    log("=" * 132)
    log(
        "12N — Rotation-Aware LOCAL-FRAME WIND "
        "DEVELOPMENT-ONLY LOMO"
    )
    log("=" * 132)
    log(f"script version : {SCRIPT_VERSION}")
    log(f"dataset        : {dataset_dir}")
    log(f"output         : {output_dir}")
    log(f"base script    : {base_path}")
    log(f"device         : {device}")
    if device.type == "cuda":
        log(
            f"GPU            : "
            f"{torch.cuda.get_device_name(0)}"
        )
    log(
        f"local features : {LOCAL_FEATURE_NAMES}"
    )
    log(
        f"local targets  : {LOCAL_TARGET_NAMES}"
    )
    log(
        f"frame fallback : last wind < "
        f"{FRAME_MIN_SPEED_MPS:.2f} m/s -> recent mean -> East"
    )
    log(
        "[FIREWALL] Tropical Atlantic and Stage 12K "
        "are NOT read."
    )
    log("")

    run_results_path = (
        output_dir / "candidate_run_results.csv"
    )
    baseline_rows = []
    transform_audit_rows = []

    for held_out in active_holdouts:
        train_names = [
            m
            for m in DEVELOPMENT_MISSIONS
            if m != held_out
        ]

        train = base.concat_missions(
            [missions[m] for m in train_names]
        )
        val = missions[held_out]

        # -------------------------------------------------------------
        # Recompute the exact Stage-12C GLOBAL Ridge reference.
        # -------------------------------------------------------------
        global_scaler = base.fit_scalers(
            train["X"],
            train["y"],
            train["aw"],
        )
        global_ridge = base.fit_ridge(
            train["X"],
            train["y"],
            global_scaler,
        )
        global_ridge_val_z = (
            base.ridge_predict_z(
                global_ridge,
                val["X"],
                global_scaler,
            )
        )
        global_ridge_val_raw = (
            base.inverse_y(
                global_ridge_val_z,
                global_scaler,
            )
        )
        global_ridge_metrics = (
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
                global_ridge_metrics
            ),
        })

        # -------------------------------------------------------------
        # Local-frame transform. Train and validation are transformed
        # independently per sample, but local scalers fit on TRAIN only.
        # -------------------------------------------------------------
        train_local = make_local_dataset(
            train["X"],
            train["y"],
        )
        val_local = make_local_dataset(
            val["X"],
            val["y"],
        )

        transform_audit_rows.extend([
            {
                "held_out_mission": held_out,
                "split": "train",
                **train_local["frame_audit"],
            },
            {
                "held_out_mission": held_out,
                "split": "validation",
                **val_local["frame_audit"],
            },
        ])

        local_scaler = fit_local_scaler(
            train_local["X_local"],
            train_local["dW_local"],
        )

        # -------------------------------------------------------------
        # Candidate A: LocalFrame-Ridge.
        # -------------------------------------------------------------
        local_ridge = fit_local_ridge(
            train_local["X_local"],
            train_local["dW_local"],
            local_scaler,
        )
        local_ridge_val_z = (
            predict_local_ridge_z(
                local_ridge,
                val_local["X_local"],
                local_scaler,
            )
        )
        local_ridge_val_raw = (
            build_full_prediction(
                base=base,
                local_pred_z=local_ridge_val_z,
                local_scaler=local_scaler,
                local_meta=val_local,
                global_ridge_raw=(
                    global_ridge_val_raw
                ),
            )
        )
        local_ridge_metrics = (
            base.evaluate_raw(
                val["y"],
                local_ridge_val_raw,
                val["aw"],
            )
        )
        local_ridge_metrics.update(
            local_delta_diagnostics(
                local_ridge_val_z,
                val_local["dW_local"],
                local_scaler,
            )
        )
        lf_score, lf_parts = (
            wind_selection_score(
                local_ridge_metrics,
                global_ridge_metrics,
            )
        )

        lf_row = {
            "candidate": "LocalFrame-Ridge",
            "mode": "ridge",
            "held_out_mission": held_out,
            "seed": -1,
            "status": "completed_finite",
            "parameter_count": (
                ridge_parameter_count(
                    local_ridge
                )
            ),
            "best_epoch_one_based": 0,
            "selection_score": lf_score,
            "elapsed_seconds": 0.0,
            **local_ridge_metrics,
            **{
                f"score_{k}": v
                for k, v in lf_parts.items()
            },
            "history_path": "",
            "checkpoint_path": "",
        }
        append_csv(
            run_results_path,
            lf_row,
        )

        log("-" * 132)
        log(
            f"[{held_out}] "
            f"Ntrain={len(train['X']):,} "
            f"Nval={len(val['X']):,}"
        )
        log(
            f"  Global Ridge     : "
            f"U={global_ridge_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={global_ridge_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={global_ridge_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={global_ridge_metrics['wind_direction_RMSE_deg']:.2f}"
        )
        log(
            f"  LocalFrame-Ridge : "
            f"score={lf_score:.5f} | "
            f"U={local_ridge_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={local_ridge_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={local_ridge_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={local_ridge_metrics['wind_direction_RMSE_deg']:.2f} | "
            f"dCorr=({local_ridge_metrics['delta_parallel_corr']:+.3f},"
            f"{local_ridge_metrics['delta_perp_corr']:+.3f})"
        )

        # -------------------------------------------------------------
        # Candidates B/C: direct local-increment neural models.
        # -------------------------------------------------------------
        for candidate_name, mode in active_neural.items():
            for seed in active_seeds:
                log(
                    f"  [TRAIN] {candidate_name} "
                    f"| seed={seed}"
                )

                result = train_neural_run(
                    base=base,
                    torch=torch,
                    nn=nn,
                    candidate_name=candidate_name,
                    mode=mode,
                    seed=seed,
                    held_out=held_out,
                    train_local=train_local,
                    val_local=val_local,
                    y_train=train["y"],
                    y_val=val["y"],
                    aw_val=val["aw"],
                    global_ridge_val_raw=(
                        global_ridge_val_raw
                    ),
                    global_ridge_val_metrics=(
                        global_ridge_metrics
                    ),
                    local_scaler=local_scaler,
                    device=device,
                    output_dir=output_dir,
                    debug_fast=args.debug_fast,
                )

                append_csv(
                    run_results_path,
                    result,
                )

                log(
                    f"  [DONE] score="
                    f"{result['selection_score']:.5f} | "
                    f"best_ep="
                    f"{result['best_epoch_one_based']} | "
                    f"U={result['wind_U_RMSE_mps']:.4f} | "
                    f"V={result['wind_V_RMSE_mps']:.4f} | "
                    f"dCorr=("
                    f"{result['delta_parallel_corr']:+.3f},"
                    f"{result['delta_perp_corr']:+.3f})"
                )

        del (
            train,
            val,
            global_scaler,
            global_ridge,
            global_ridge_val_z,
            global_ridge_val_raw,
            train_local,
            val_local,
            local_scaler,
            local_ridge,
            local_ridge_val_z,
            local_ridge_val_raw,
        )

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    baseline_df = pd.DataFrame(
        baseline_rows
    )
    baseline_path = (
        output_dir / "global_ridge_fold_results.csv"
    )
    baseline_df.to_csv(
        baseline_path,
        index=False,
        encoding="utf-8-sig",
    )

    transform_audit_df = pd.DataFrame(
        transform_audit_rows
    )
    transform_audit_path = (
        output_dir / "local_frame_transform_audit.csv"
    )
    transform_audit_df.to_csv(
        transform_audit_path,
        index=False,
        encoding="utf-8-sig",
    )

    if args.debug_fast:
        log("")
        log(
            "[DONE] debug-fast completed. "
            "No scientific freeze/decision written."
        )
        return 0

    # =================================================================
    # Full scientific summary.
    # =================================================================
    runs = pd.read_csv(
        run_results_path
    )

    expected_counts = {
        "LocalFrame-Ridge": len(
            DEVELOPMENT_MISSIONS
        ),
        "LocalFrame-MLP": (
            len(DEVELOPMENT_MISSIONS)
            * len(SCREENING_SEEDS)
        ),
        "LocalFrame-GRU": (
            len(DEVELOPMENT_MISSIONS)
            * len(SCREENING_SEEDS)
        ),
    }

    for candidate, expected in expected_counts.items():
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
                f"{expected} completed runs, got {got}."
            )

    summary_rows = [
        summarize_candidate(
            runs, candidate
        )
        for candidate
        in [
            "LocalFrame-Ridge",
            "LocalFrame-MLP",
            "LocalFrame-GRU",
        ]
    ]

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(
            [
                "selection_score_mean",
                "parameter_count",
            ],
            ascending=[True, True],
        )
        .reset_index(drop=True)
    )

    summary_path = (
        output_dir / "candidate_summary.csv"
    )
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    selected_name = str(
        summary.iloc[0]["candidate"]
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
        k: float(baseline_df[k].mean())
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
    v_improvement = float(
        relative_improvement[
            "wind_V_RMSE_mps"
        ]
    )

    meets_continue_rule = bool(
        selected_score_mean
        < CONTINUE_SCORE_THRESHOLD
        and v_improvement
        > CONTINUE_V_IMPROVEMENT_FRACTION
    )

    # Additional descriptive cross-mission stability.
    selected_mission_scores = (
        selected_runs
        .groupby("held_out_mission")[
            "selection_score"
        ]
        .mean()
        .to_dict()
    )

    decision = (
        "CONTINUE_TO_V5_INTEGRATION"
        if meets_continue_rule
        else "STOP_TRUE_WIND_NONLINEAR_ESCALATION"
    )

    selected_payload = {
        "stage": "12N",
        "script_version": SCRIPT_VERSION,
        "model_name": "LocalFrameWind-NieData-v5",
        "selected_candidate": selected_name,
        "train_freeze": asdict(FREEZE),
        "local_feature_names": LOCAL_FEATURE_NAMES,
        "local_target_names": LOCAL_TARGET_NAMES,
        "frame_definition": {
            "e_parallel": (
                "last observed wind direction; "
                "recent-mean fallback below threshold"
            ),
            "e_perp": (
                "[-e_parallel_N, e_parallel_E]"
            ),
            "frame_min_speed_mps": (
                FRAME_MIN_SPEED_MPS
            ),
        },
        "wind_selection_weights": (
            WIND_SELECTION_WEIGHTS
        ),
        "continuation_rule": {
            "score_threshold_strictly_less_than": (
                CONTINUE_SCORE_THRESHOLD
            ),
            "V_RMSE_improvement_fraction_strictly_greater_than": (
                CONTINUE_V_IMPROVEMENT_FRACTION
            ),
        },
        "selected_selection_score_mean": (
            selected_score_mean
        ),
        "selected_mission_mean_scores": (
            selected_mission_scores
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
        "meets_continue_rule": (
            meets_continue_rule
        ),
        "decision": decision,
        "Tropical_Atlantic_used": False,
        "Stage_12K_used": False,
        "scientific_note": (
            "All candidate selection and the continuation decision "
            "use Antarctic, Atlantic, and West Coast LOMO only. "
            "Do not modify this decision using Tropical Atlantic."
        ),
    }

    selected_json_path = (
        output_dir / "selected_config.json"
    )
    save_json(
        selected_json_path,
        selected_payload,
    )

    report_path = (
        output_dir / "12N_REPORT.txt"
    )
    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "12N Rotation-Aware Local-Frame Wind "
            "Development-Only LOMO\n"
        )
        f.write("=" * 132 + "\n\n")

        f.write("SCIENTIFIC FIREWALL\n")
        f.write("-" * 132 + "\n")
        f.write(
            "Antarctic / Atlantic / West Coast: ACCESSED\n"
        )
        f.write(
            "Tropical Atlantic: NOT ACCESSED\n"
        )
        f.write(
            "Stage 12K outputs: NOT ACCESSED\n\n"
        )

        f.write("LOCAL FRAME\n")
        f.write("-" * 132 + "\n")
        f.write(
            "e_parallel = last observed wind direction\n"
        )
        f.write(
            f"Weak-wind threshold = "
            f"{FRAME_MIN_SPEED_MPS:.3f} m/s\n"
        )
        f.write(
            "fallback = recent mean wind -> fixed East axis\n"
        )
        f.write(
            "target = future wind increment "
            "[dW_parallel, dW_perp]\n\n"
        )

        f.write("CANDIDATE SUMMARY\n")
        f.write("-" * 132 + "\n")
        f.write(
            summary.to_string(index=False)
        )
        f.write("\n\n")

        f.write("SELECTED\n")
        f.write("-" * 132 + "\n")
        f.write(
            f"candidate: {selected_name}\n"
        )
        f.write(
            f"mean WindScore: "
            f"{selected_score_mean:.6f}\n"
        )
        f.write(
            "relative improvement vs Global Ridge:\n"
        )
        for k, v in relative_improvement.items():
            f.write(
                f"  {k}: {100.0*v:+.4f}%\n"
            )

        f.write(
            "\nMean score by held-out mission:\n"
        )
        for mission, value in (
            selected_mission_scores.items()
        ):
            f.write(
                f"  {mission}: {value:.6f}\n"
            )

        f.write("\nPREDECLARED DECISION RULE\n")
        f.write("-" * 132 + "\n")
        f.write(
            f"score < {CONTINUE_SCORE_THRESHOLD:.3f} "
            f"AND V improvement > "
            f"{100.0*CONTINUE_V_IMPROVEMENT_FRACTION:.1f}%\n"
        )
        f.write(
            f"meets rule: {meets_continue_rule}\n"
        )
        f.write(
            f"decision: {decision}\n"
        )

    log("")
    log("=" * 132)
    log("12N CANDIDATE SUMMARY")
    log("=" * 132)
    log(
        summary.to_string(index=False)
    )
    log("")
    log(f"[SELECTED] {selected_name}")
    log(
        f"Mean WindScore = "
        f"{selected_score_mean:.6f}"
    )
    log(
        "Development relative improvement vs Global Ridge: "
        + " | ".join(
            f"{k}={100.0*v:+.3f}%"
            for k, v in relative_improvement.items()
        )
    )
    log(
        "Selected local-delta corr mean: "
        f"parallel="
        f"{selected_runs['delta_parallel_corr'].mean():+.4f} | "
        f"perp="
        f"{selected_runs['delta_perp_corr'].mean():+.4f}"
    )
    log(
        "Mission mean scores: "
        + " | ".join(
            f"{m}={s:.5f}"
            for m, s
            in selected_mission_scores.items()
        )
    )
    log("")
    log(
        f"[DECISION] {decision}"
    )
    log(
        f"Rule: score < "
        f"{CONTINUE_SCORE_THRESHOLD:.3f} "
        f"AND V improvement > "
        f"{100.0*CONTINUE_V_IMPROVEMENT_FRACTION:.1f}%"
    )
    log("")
    log(
        f"[SAVED] {run_results_path}"
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
        f"[SAVED] {transform_audit_path}"
    )
    log(
        f"[SAVED] {report_path}"
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print(
            "\n[FATAL ERROR]",
            flush=True,
        )
        traceback.print_exc()
        sys.exit(1)
