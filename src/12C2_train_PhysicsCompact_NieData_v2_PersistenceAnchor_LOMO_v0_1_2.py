# -*- coding: utf-8 -*-
r"""
12C2_train_PhysicsCompact_NieData_v2_PersistenceAnchor_LOMO.py

Stage 12C2
=========
Development-only leave-one-mission-out (LOMO) training and validation for the
Nie-data-compatible Physics-Compact model.

SCIENTIFIC FIREWALL
-------------------
This script reads ONLY:
    track_J_joint_compatible/Antarctic.npz
    track_J_joint_compatible/Atlantic.npz
    track_J_joint_compatible/West_Coast.npz

It NEVER reads:
    track_J_joint_compatible/Tropical_Atlantic_TEST.npz
    any Tropical Atlantic resampled CSV
    any held-out test metric

No directory glob is used for NPZ input. Development file paths are explicit.

DATA
----
Stage-12B frozen Track J:
    X: [N, 6, 9]
       [U, V, T, RH, SOG,
        COG_sin, COG_cos,
        WING_ANGLE_sin, WING_ANGLE_cos]

    y: [N, 1, 4]
       [U, V, VESSEL_E, VESSEL_N]

    apparent_ref: [N, 1, 2]
       AW_E = U - VESSEL_E
       AW_N = V - VESSEL_N

LOMO FOLDS
----------
Fold Antarctic:
    train = Atlantic + West Coast
    validation = Antarctic

Fold Atlantic:
    train = Antarctic + West Coast
    validation = Atlantic

Fold West Coast:
    train = Antarctic + Atlantic
    validation = West Coast

No sample from a validation mission enters its fold scaler or Ridge baseline.

MODEL FREEZE -- Physics-Compact-NieData v2.0 (Persistence Anchor)
----------------------------------------------------------------
This is a mechanism-only revision declared BEFORE held-out Tropical Atlantic
evaluation.

Relative to v1, the ONLY modeling change is:

    Ridge residual anchor  ->  sample-wise Persistence residual anchor

All neural dimensions, loss weights, folds, seeds, optimizer settings, and
checkpoint rules remain unchanged.

For standardized outputs:

    W_hat_z = W_persist_z + g_W * DeltaW_z
    V_hat_z = V_persist_z + g_V * DeltaV_z

After inverse scaling to physical units:

    A_hat = W_hat - V_hat

Architecture:
    shared GRU: 64 hidden units, 1 layer
    dense:      64 units + GELU
    dropout:    0.10
    wind delta head:   2
    wind gate head:    2, sigmoid
    vessel delta head: 2
    vessel gate head:  2, sigmoid

Delta heads are initialized to zero so the neural model starts EXACTLY from
the sample-wise Persistence prediction.

Two neural variants are trained with identical architecture:

    CompactNoPhysics:
        L = L_direct + 0.01 * L_res

    PhysicsCompactNieData:
        L = L_direct + 2.0 * L_AW + 0.01 * L_res

where:
    L_direct = 0.5 * MSE_z(W) + 0.5 * MSE_z(V)

    L_AW is an apparent-wind loss in physical units normalized by the
    fold-training apparent-wind component standard deviations.

    L_res = mean((g_W * DeltaW_z)^2) + mean((g_V * DeltaV_z)^2)

The proposed Stage-12D candidate is PhysicsCompactNieDataV2. CompactNoPhysicsV2 is an
ablation and does not replace the proposed model based on LOMO results.

RIDGE BASELINE
--------------
For each fold:
    input = flattened 6 x 9 = 54 features
    output = 4 joint targets
    alpha = 1.0, fixed a priori
    X and y scalers fit on fold TRAIN only

PERSISTENCE
-----------
Wind:
    last observed U,V in the context

Vessel:
    reconstruct from last observed SOG/COG:
        V_E = SOG * sin(COG)
        V_N = SOG * cos(COG)

Apparent:
    W_persistence - V_persistence

TRAINING
--------
LOMO screening seeds:
    500043, 501052, 502061

The SAME seeds are used for every held-out mission and both neural variants.

Fixed training settings:
    optimizer        = Adam
    learning rate    = 1e-3
    batch size       = 256
    max epochs       = 100
    early patience   = 15
    LR scheduler     = ReduceLROnPlateau(factor=0.5, patience=5)
    min LR           = 1e-5
    grad clip        = 1.0
    AMP              = enabled on CUDA
    chronological batches; no random sample shuffling

CHECKPOINT RULE
---------------
The model is an apparent-wind-preview model, so each run is checkpointed by
validation apparent-wind vector RMSE.

Wind U/V, wind-speed, and wind-direction metrics are still recorded at the same
checkpoint for later same-mission comparison with Nie et al.

METRICS
-------
Wind:
    U RMSE
    V RMSE
    vector RMSE = sqrt(mean(dU^2+dV^2))
    wind-speed RMSE
    meteorological wind-direction MAE / RMSE (circular degrees)

Vessel:
    E RMSE
    N RMSE
    vector RMSE

Apparent wind:
    E RMSE
    N RMSE
    vector RMSE
    apparent-wind-speed RMSE

Averages are reported by:
    mission x method
    method across the three held-out missions

No body-frame AWA is computed because Atlantic lacks true vessel HDG.

STAGE-12D POLICY
----------------
Stage 12C2 does NOT fit the final all-development model and does NOT touch the
held-out Tropical Atlantic mission.

If Stage 12C2 completes cleanly, Stage 12D will:
    - keep this architecture/loss frozen;
    - fit on Antarctic + Atlantic + West Coast;
    - use five final project seeds:
        500043, 501052, 502061, 503070, 504079
    - evaluate Tropical Atlantic exactly once after the final protocol is frozen.

OUTPUTS
-------
12C_PhysicsCompact_NieData_LOMO_v0_1/
    fold_definition.csv
    baseline_fold_results.csv
    neural_run_results.csv
    lomo_method_summary.csv
    seed_level_summary.csv
    proposed_model_freeze.json
    training_manifest.json
    training_report.txt
    histories/
    checkpoints/
    fold_scalers/

RUN
---
conda activate WindPredict
python "D:\project\WindPredict_SaildroneData\src\12C2_train_PhysicsCompact_NieData_v2_PersistenceAnchor_LOMO.py"

Optional NON-SCIENTIFIC smoke:
python "...12C2_train_PhysicsCompact_NieData_v2_PersistenceAnchor_LOMO.py" --debug-fast
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import Ridge
except Exception as exc:
    raise RuntimeError(
        "scikit-learn is required. Activate the WindPredict environment."
    ) from exc


SCRIPT_VERSION = "0.1.2-PhysicsCompact-NieData-v2-PersistenceAnchor-LOMO"

DEFAULT_PROJECT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")
DEFAULT_12B_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12B_Nie_same_mission_benchmark_v0_1"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12C2_PhysicsCompact_NieData_v2_PersistenceAnchor_LOMO_v0_1"
)

DEVELOPMENT_MISSIONS = ["Antarctic", "Atlantic", "West Coast"]
FORBIDDEN_TEST_MISSION = "Tropical Atlantic"

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

TARGET_NAMES = ["U", "V", "VESSEL_E", "VESSEL_N"]
APPARENT_NAMES = ["AW_E", "AW_N"]

SCREENING_SEEDS = [500043, 501052, 502061]
FINAL_STAGE12D_SEEDS = [500043, 501052, 502061, 503070, 504079]

RIDGE_ALPHA = 1.0


@dataclass(frozen=True)
class ModelFreeze:
    gru_hidden: int = 64
    gru_layers: int = 1
    dense_hidden: int = 64
    dropout: float = 0.10

    lambda_aw_physics: float = 2.0
    lambda_residual: float = 0.01

    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 100
    early_stop_patience: int = 15
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-5
    grad_clip_norm: float = 1.0

    use_amp: bool = True
    train_shuffle: bool = False

    checkpoint_metric: str = "validation_apparent_vector_RMSE_mps"


FREEZE = ModelFreeze()

NEURAL_VARIANTS = {
    "CompactNoPhysicsV2": 0.0,
    "PhysicsCompactNieDataV2": FREEZE.lambda_aw_physics,
}


def log(message=""):
    print(message, flush=True)


def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required. Activate the WindPredict environment."
        ) from exc

    return torch, nn, F


def seed_everything(torch, seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def configure_cuda(torch):
    if not torch.cuda.is_available():
        return

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def save_json(path: Path, obj):
    def cv(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return None if np.isnan(v) else float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, dict):
            return {str(k): cv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [cv(x) for x in v]
        return v

    with path.open("w", encoding="utf-8") as f:
        json.dump(cv(obj), f, ensure_ascii=False, indent=2, sort_keys=True)


def mission_filename(mission: str):
    return mission.replace(" ", "_") + ".npz"


def explicit_development_paths(dataset_dir: Path):
    """
    IMPORTANT: explicit paths only.

    No glob/rglob is used here, so the Tropical_Atlantic_TEST file is not even
    enumerated by this script.
    """
    j_dir = dataset_dir / "track_J_joint_compatible"

    return {
        "Antarctic": j_dir / "Antarctic.npz",
        "Atlantic": j_dir / "Atlantic.npz",
        "West Coast": j_dir / "West_Coast.npz",
    }


def load_development_mission(path: Path, mission: str):
    if FORBIDDEN_TEST_MISSION.replace(" ", "_").lower() in path.name.lower():
        raise RuntimeError(
            f"Forbidden held-out test path was requested: {path}"
        )

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        required = [
            "X_raw",
            "y_joint_raw",
            "apparent_ref",
            "context_end_time_ns",
            "target_time_ns",
            "feature_names",
            "target_names",
            "apparent_names",
        ]

        for key in required:
            if key not in data.files:
                raise KeyError(
                    f"{path}: missing '{key}', available={data.files}"
                )

        X = np.asarray(data["X_raw"], dtype=np.float32)
        y = np.asarray(data["y_joint_raw"], dtype=np.float32)
        aw = np.asarray(data["apparent_ref"], dtype=np.float32)
        context = np.asarray(
            data["context_end_time_ns"],
            dtype=np.int64,
        ).reshape(-1)
        target_time = np.asarray(data["target_time_ns"], dtype=np.int64)

        feature_names = [str(x) for x in data["feature_names"].tolist()]
        target_names = [str(x) for x in data["target_names"].tolist()]
        apparent_names = [str(x) for x in data["apparent_names"].tolist()]

    if X.ndim != 3 or X.shape[1:] != (6, 9):
        raise RuntimeError(
            f"{mission}: expected X [N,6,9], got {X.shape}"
        )

    if y.ndim != 3 or y.shape[1:] != (1, 4):
        raise RuntimeError(
            f"{mission}: expected y [N,1,4], got {y.shape}"
        )

    if aw.ndim != 3 or aw.shape[1:] != (1, 2):
        raise RuntimeError(
            f"{mission}: expected apparent_ref [N,1,2], got {aw.shape}"
        )

    if feature_names != FEATURE_NAMES:
        raise RuntimeError(
            f"{mission}: feature schema mismatch: {feature_names}"
        )

    if target_names != TARGET_NAMES:
        raise RuntimeError(
            f"{mission}: target schema mismatch: {target_names}"
        )

    if apparent_names != APPARENT_NAMES:
        raise RuntimeError(
            f"{mission}: apparent schema mismatch: {apparent_names}"
        )

    if not (
        len(X)
        == len(y)
        == len(aw)
        == len(context)
        == len(target_time)
    ):
        raise RuntimeError(f"{mission}: array-length mismatch.")

    if not (
        np.isfinite(X).all()
        and np.isfinite(y).all()
        and np.isfinite(aw).all()
    ):
        raise RuntimeError(f"{mission}: non-finite values detected.")

    physics = y[:, :, 0:2] - y[:, :, 2:4]
    physics_max = float(np.max(np.abs(physics - aw)))

    if physics_max > 1e-5:
        raise RuntimeError(
            f"{mission}: AW physics mismatch, max abs error={physics_max:.3e}"
        )

    if np.any(np.diff(context) <= 0):
        raise RuntimeError(
            f"{mission}: context timestamps are not strictly increasing."
        )

    return {
        "mission": mission,
        "X": X,
        "y": y,
        "aw": aw,
        "context_end_time_ns": context,
        "target_time_ns": target_time,
        "physics_max_abs_error": physics_max,
    }


def concat_missions(items: Sequence[Dict]):
    return {
        "X": np.concatenate([x["X"] for x in items], axis=0),
        "y": np.concatenate([x["y"] for x in items], axis=0),
        "aw": np.concatenate([x["aw"] for x in items], axis=0),
        "mission": np.concatenate([
            np.asarray([x["mission"]] * len(x["X"]))
            for x in items
        ]),
    }


def fit_scalers(X_train: np.ndarray, y_train: np.ndarray, aw_train: np.ndarray):
    X_flat = X_train.astype(np.float64).reshape(-1, X_train.shape[-1])
    y_flat = y_train.astype(np.float64).reshape(-1, y_train.shape[-1])
    aw_flat = aw_train.astype(np.float64).reshape(-1, aw_train.shape[-1])

    x_mean = X_flat.mean(axis=0)
    x_std = X_flat.std(axis=0)
    y_mean = y_flat.mean(axis=0)
    y_std = y_flat.std(axis=0)
    aw_mean = aw_flat.mean(axis=0)
    aw_std = aw_flat.std(axis=0)

    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)
    aw_std = np.where(aw_std < 1e-8, 1.0, aw_std)

    return {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
        "aw_mean": aw_mean.astype(np.float32),
        "aw_std": aw_std.astype(np.float32),
    }


def standardize_X(X, scaler):
    return (
        np.asarray(X, dtype=np.float32)
        - scaler["x_mean"][None, None, :]
    ) / scaler["x_std"][None, None, :]


def standardize_y(y, scaler):
    return (
        np.asarray(y, dtype=np.float32)
        - scaler["y_mean"][None, None, :]
    ) / scaler["y_std"][None, None, :]


def inverse_y(yz, scaler):
    return (
        np.asarray(yz, dtype=np.float64)
        * scaler["y_std"][None, None, :]
        + scaler["y_mean"][None, None, :]
    )


def flatten_X(X):
    return np.asarray(X, dtype=np.float64).reshape(len(X), -1)


def fit_ridge(X_train, y_train, scaler):
    Xz = standardize_X(X_train, scaler)
    yz = standardize_y(y_train, scaler)

    model = Ridge(
        alpha=RIDGE_ALPHA,
        fit_intercept=True,
    )
    model.fit(
        flatten_X(Xz),
        yz.reshape(len(yz), -1),
    )
    return model


def ridge_predict_z(model, X, scaler):
    Xz = standardize_X(X, scaler)
    pred = model.predict(flatten_X(Xz))
    return pred.reshape(len(X), 1, 4).astype(np.float32)


def persistence_predict(X):
    """
    X last step:
        U,V,T,RH,SOG,COG_sin,COG_cos,WING_sin,WING_cos
    """
    last = np.asarray(X[:, -1, :], dtype=np.float64)

    U = last[:, 0]
    V = last[:, 1]
    SOG = last[:, 4]
    cog_sin = last[:, 5]
    cog_cos = last[:, 6]

    norm = np.sqrt(cog_sin**2 + cog_cos**2)
    norm = np.where(norm < 1e-12, 1.0, norm)

    cog_sin = cog_sin / norm
    cog_cos = cog_cos / norm

    vessel_e = SOG * cog_sin
    vessel_n = SOG * cog_cos

    y = np.stack(
        [U, V, vessel_e, vessel_n],
        axis=1,
    )[:, None, :]

    aw = y[:, :, 0:2] - y[:, :, 2:4]

    return y.astype(np.float32), aw.astype(np.float32)


def persistence_predict_z(X, scaler):
    """
    Build the v2 persistence anchor and express it in the SAME standardized
    target space used by the residual heads.

    Because the delta heads are initialized to zero, epoch-0 prediction is
    exactly this anchor.
    """
    y_raw, _ = persistence_predict(X)
    return standardize_y(y_raw, scaler).astype(np.float32)


def wrap_deg180(x):
    return (x + 180.0) % 360.0 - 180.0


def meteorological_direction_from(U, V):
    return np.rad2deg(np.arctan2(-U, -V)) % 360.0


def evaluate_raw(y_true, y_pred, aw_true=None):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if aw_true is None:
        aw_true = y_true[:, :, 0:2] - y_true[:, :, 2:4]
    else:
        aw_true = np.asarray(aw_true, dtype=np.float64)

    aw_pred = y_pred[:, :, 0:2] - y_pred[:, :, 2:4]

    wt = y_true[:, 0, 0:2]
    wp = y_pred[:, 0, 0:2]
    vt = y_true[:, 0, 2:4]
    vp = y_pred[:, 0, 2:4]
    at = aw_true[:, 0, :]
    ap = aw_pred[:, 0, :]

    wind_err = wp - wt
    vessel_err = vp - vt
    aw_err = ap - at

    wind_speed_true = np.linalg.norm(wt, axis=1)
    wind_speed_pred = np.linalg.norm(wp, axis=1)

    wind_dir_true = meteorological_direction_from(wt[:, 0], wt[:, 1])
    wind_dir_pred = meteorological_direction_from(wp[:, 0], wp[:, 1])
    wind_dir_err = wrap_deg180(wind_dir_pred - wind_dir_true)

    aws_true = np.linalg.norm(at, axis=1)
    aws_pred = np.linalg.norm(ap, axis=1)

    return {
        "wind_U_RMSE_mps": float(
            np.sqrt(np.mean(wind_err[:, 0] ** 2))
        ),
        "wind_V_RMSE_mps": float(
            np.sqrt(np.mean(wind_err[:, 1] ** 2))
        ),
        "wind_vector_RMSE_mps": float(
            np.sqrt(np.mean(np.sum(wind_err**2, axis=1)))
        ),
        "wind_speed_RMSE_mps": float(
            np.sqrt(np.mean((wind_speed_pred - wind_speed_true) ** 2))
        ),
        "wind_direction_MAE_deg": float(
            np.mean(np.abs(wind_dir_err))
        ),
        "wind_direction_RMSE_deg": float(
            np.sqrt(np.mean(wind_dir_err**2))
        ),
        "vessel_E_RMSE_mps": float(
            np.sqrt(np.mean(vessel_err[:, 0] ** 2))
        ),
        "vessel_N_RMSE_mps": float(
            np.sqrt(np.mean(vessel_err[:, 1] ** 2))
        ),
        "vessel_vector_RMSE_mps": float(
            np.sqrt(np.mean(np.sum(vessel_err**2, axis=1)))
        ),
        "AW_E_RMSE_mps": float(
            np.sqrt(np.mean(aw_err[:, 0] ** 2))
        ),
        "AW_N_RMSE_mps": float(
            np.sqrt(np.mean(aw_err[:, 1] ** 2))
        ),
        "AW_vector_RMSE_mps": float(
            np.sqrt(np.mean(np.sum(aw_err**2, axis=1)))
        ),
        "AWS_RMSE_mps": float(
            np.sqrt(np.mean((aws_pred - aws_true) ** 2))
        ),
    }


def build_model_class(torch, nn, F):
    class PhysicsCompactNieDataV2(nn.Module):
        def __init__(self):
            super().__init__()

            self.gru = nn.GRU(
                input_size=len(FEATURE_NAMES),
                hidden_size=FREEZE.gru_hidden,
                num_layers=FREEZE.gru_layers,
                batch_first=True,
                dropout=(
                    FREEZE.dropout
                    if FREEZE.gru_layers > 1
                    else 0.0
                ),
            )

            self.shared = nn.Sequential(
                nn.Linear(FREEZE.gru_hidden, FREEZE.dense_hidden),
                nn.GELU(),
                nn.Dropout(FREEZE.dropout),
            )

            self.wind_delta = nn.Linear(FREEZE.dense_hidden, 2)
            self.wind_gate = nn.Linear(FREEZE.dense_hidden, 2)

            self.vessel_delta = nn.Linear(FREEZE.dense_hidden, 2)
            self.vessel_gate = nn.Linear(FREEZE.dense_hidden, 2)

            # Start exactly from Persistence.
            nn.init.zeros_(self.wind_delta.weight)
            nn.init.zeros_(self.wind_delta.bias)
            nn.init.zeros_(self.vessel_delta.weight)
            nn.init.zeros_(self.vessel_delta.bias)

            # Neutral initial gate = sigmoid(0) = 0.5.
            nn.init.zeros_(self.wind_gate.bias)
            nn.init.zeros_(self.vessel_gate.bias)

        def forward(self, x, persistence_pred_z):
            seq, _ = self.gru(x)
            h = self.shared(seq[:, -1, :])

            dw = self.wind_delta(h)
            gw = torch.sigmoid(self.wind_gate(h))

            dv = self.vessel_delta(h)
            gv = torch.sigmoid(self.vessel_gate(h))

            pred_w = persistence_pred_z[:, 0, 0:2] + gw * dw
            pred_v = persistence_pred_z[:, 0, 2:4] + gv * dv

            pred = torch.cat([pred_w, pred_v], dim=1)[:, None, :]

            return {
                "pred_z": pred,
                "wind_delta_z": dw,
                "wind_gate": gw,
                "vessel_delta_z": dv,
                "vessel_gate": gv,
                "wind_correction_z": gw * dw,
                "vessel_correction_z": gv * dv,
            }

    return PhysicsCompactNieDataV2


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def create_grad_scaler(torch, enabled):
    if not enabled:
        return None

    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        try:
            return torch.cuda.amp.GradScaler(enabled=True)
        except Exception:
            return None


def autocast_context(torch, enabled):
    if not enabled:
        return torch.autocast(
            device_type="cuda",
            enabled=False,
        )

    try:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        )
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def sequential_batches(*arrays, batch_size):
    n = len(arrays[0])

    for start in range(0, n, int(batch_size)):
        end = min(n, start + int(batch_size))
        yield tuple(arr[start:end] for arr in arrays)


def torch_scaler_constants(torch, scaler, device):
    return {
        "y_mean": torch.as_tensor(
            scaler["y_mean"],
            dtype=torch.float32,
            device=device,
        ),
        "y_std": torch.as_tensor(
            scaler["y_std"],
            dtype=torch.float32,
            device=device,
        ),
        "aw_std": torch.as_tensor(
            scaler["aw_std"],
            dtype=torch.float32,
            device=device,
        ),
    }


def compute_training_loss(
    torch,
    out,
    y_true_z,
    aw_true_raw,
    const,
    lambda_aw,
):
    pred_z = out["pred_z"]

    wind_mse = torch.mean(
        (pred_z[:, :, 0:2] - y_true_z[:, :, 0:2]) ** 2
    )
    vessel_mse = torch.mean(
        (pred_z[:, :, 2:4] - y_true_z[:, :, 2:4]) ** 2
    )

    direct = 0.5 * wind_mse + 0.5 * vessel_mse

    y_mean = const["y_mean"][None, None, :]
    y_std = const["y_std"][None, None, :]

    pred_raw = pred_z * y_std + y_mean
    aw_pred_raw = pred_raw[:, :, 0:2] - pred_raw[:, :, 2:4]

    aw_scale = const["aw_std"][None, None, :].clamp_min(1e-6)

    aw_loss = torch.mean(
        ((aw_pred_raw - aw_true_raw) / aw_scale) ** 2
    )

    residual = (
        torch.mean(out["wind_correction_z"] ** 2)
        + torch.mean(out["vessel_correction_z"] ** 2)
    )

    total = (
        direct
        + float(lambda_aw) * aw_loss
        + FREEZE.lambda_residual * residual
    )

    return {
        "total": total,
        "direct": direct,
        "wind_mse_z": wind_mse,
        "vessel_mse_z": vessel_mse,
        "aw_loss_norm": aw_loss,
        "residual_penalty": residual,
    }


def train_one_epoch(
    torch,
    model,
    optimizer,
    grad_scaler,
    Xz,
    yz,
    aw_raw,
    persistence_pred_z,
    const,
    lambda_aw,
    device,
    amp_enabled,
):
    model.train()

    sums = {
        "total": 0.0,
        "direct": 0.0,
        "wind_mse_z": 0.0,
        "vessel_mse_z": 0.0,
        "aw_loss_norm": 0.0,
        "residual_penalty": 0.0,
    }
    count = 0

    for xb_np, yb_np, awb_np, rb_np in sequential_batches(
        Xz,
        yz,
        aw_raw,
        persistence_pred_z,
        batch_size=FREEZE.batch_size,
    ):
        xb = torch.from_numpy(xb_np).to(device, non_blocking=True)
        yb = torch.from_numpy(yb_np).to(device, non_blocking=True)
        awb = torch.from_numpy(awb_np).to(device, non_blocking=True)
        rb = torch.from_numpy(rb_np).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(torch, amp_enabled):
            out = model(xb, rb)
            losses = compute_training_loss(
                torch,
                out,
                yb,
                awb,
                const,
                lambda_aw,
            )

        total = losses["total"]

        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite neural training loss.")

        if grad_scaler is not None:
            grad_scaler.scale(total).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                FREEZE.grad_clip_norm,
            )
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                FREEZE.grad_clip_norm,
            )
            optimizer.step()

        bn = len(xb_np)

        for key in sums:
            sums[key] += float(
                losses[key].detach().float().cpu().item()
            ) * bn

        count += bn

        del xb, yb, awb, rb, out, losses, total

    return {
        f"train_{key}": value / max(count, 1)
        for key, value in sums.items()
    }


def evaluate_neural(
    torch,
    model,
    X,
    y,
    aw,
    persistence_pred_z,
    scaler,
    device,
    amp_enabled,
):
    model.eval()

    Xz = standardize_X(X, scaler)

    preds = []
    wind_gates = []
    vessel_gates = []
    wind_corr = []
    vessel_corr = []

    with torch.no_grad():
        for xb_np, rb_np in sequential_batches(
            Xz,
            persistence_pred_z,
            batch_size=FREEZE.batch_size,
        ):
            xb = torch.from_numpy(xb_np).to(device, non_blocking=True)
            rb = torch.from_numpy(rb_np).to(device, non_blocking=True)

            with autocast_context(torch, amp_enabled):
                out = model(xb, rb)

            preds.append(
                out["pred_z"].detach().float().cpu().numpy()
            )
            wind_gates.append(
                out["wind_gate"].detach().float().cpu().numpy()
            )
            vessel_gates.append(
                out["vessel_gate"].detach().float().cpu().numpy()
            )
            wind_corr.append(
                out["wind_correction_z"].detach().float().cpu().numpy()
            )
            vessel_corr.append(
                out["vessel_correction_z"].detach().float().cpu().numpy()
            )

            del xb, rb, out

    pred_z = np.concatenate(preds, axis=0)
    pred_raw = inverse_y(pred_z, scaler)

    metrics = evaluate_raw(y, pred_raw, aw)

    wg = np.concatenate(wind_gates, axis=0)
    vg = np.concatenate(vessel_gates, axis=0)
    wc = np.concatenate(wind_corr, axis=0)
    vc = np.concatenate(vessel_corr, axis=0)

    metrics.update({
        "wind_gate_mean": float(np.mean(wg)),
        "wind_gate_std": float(np.std(wg)),
        "vessel_gate_mean": float(np.mean(vg)),
        "vessel_gate_std": float(np.std(vg)),
        "wind_correction_z_RMS": float(np.sqrt(np.mean(wc**2))),
        "vessel_correction_z_RMS": float(np.sqrt(np.mean(vc**2))),
    })

    return metrics, pred_raw


def state_dict_cpu(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def train_neural_run(
    *,
    torch,
    nn,
    F,
    variant_name,
    lambda_aw,
    seed,
    held_out_mission,
    X_train,
    y_train,
    aw_train,
    X_val,
    y_val,
    aw_val,
    persistence_train_z,
    persistence_val_z,
    scaler,
    device,
    output_dir,
    debug_fast,
):
    seed_everything(torch, seed)

    ModelClass = build_model_class(torch, nn, F)
    model = ModelClass().to(device)

    parameter_count = count_parameters(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FREEZE.learning_rate,
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

    const = torch_scaler_constants(
        torch,
        scaler,
        device,
    )

    X_train_z = standardize_X(X_train, scaler)
    y_train_z = standardize_y(y_train, scaler)

    max_epochs = 8 if debug_fast else FREEZE.max_epochs
    patience = 4 if debug_fast else FREEZE.early_stop_patience

    best_metric = math.inf
    best_epoch = None
    best_state = None
    best_metrics = None

    patience_counter = 0
    history_rows = []

    start_time = time.perf_counter()

    for epoch in range(max_epochs):
        train_stats = train_one_epoch(
            torch,
            model,
            optimizer,
            grad_scaler,
            X_train_z,
            y_train_z,
            aw_train,
            persistence_train_z,
            const,
            lambda_aw,
            device,
            amp_enabled,
        )

        val_metrics, _ = evaluate_neural(
            torch,
            model,
            X_val,
            y_val,
            aw_val,
            persistence_val_z,
            scaler,
            device,
            amp_enabled,
        )

        current = float(
            val_metrics["AW_vector_RMSE_mps"]
        )

        improved = current < best_metric - 1e-8

        if improved:
            best_metric = current
            best_epoch = int(epoch)
            best_state = state_dict_cpu(model)
            best_metrics = val_metrics
            patience_counter = 0
        else:
            patience_counter += 1

        lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(current)

        history_rows.append({
            "variant": variant_name,
            "held_out_mission": held_out_mission,
            "seed": int(seed),
            "epoch_zero_based": int(epoch),
            "epoch_one_based": int(epoch + 1),
            "learning_rate": lr,
            "checkpoint_improved": bool(improved),
            **train_stats,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        log(
            f"      epoch={epoch+1:03d}/{max_epochs} | "
            f"AW={val_metrics['AW_vector_RMSE_mps']:.4f} | "
            f"W={val_metrics['wind_vector_RMSE_mps']:.4f} | "
            f"Vship={val_metrics['vessel_vector_RMSE_mps']:.4f} | "
            f"lr={lr:.2e}"
            + (" *" if improved else "")
        )

        if patience_counter >= patience:
            log(
                f"      early stop: {patience_counter} epochs "
                "without AW improvement."
            )
            break

    elapsed = time.perf_counter() - start_time

    if best_state is None:
        raise RuntimeError(
            f"{variant_name}/{held_out_mission}/seed{seed}: "
            "no finite checkpoint."
        )

    model.load_state_dict(best_state, strict=True)

    confirmed, _ = evaluate_neural(
        torch,
        model,
        X_val,
        y_val,
        aw_val,
        persistence_val_z,
        scaler,
        device,
        amp_enabled,
    )

    if abs(
        confirmed["AW_vector_RMSE_mps"]
        - best_metrics["AW_vector_RMSE_mps"]
    ) > 1e-5:
        raise RuntimeError(
            "Checkpoint re-evaluation mismatch."
        )

    histories_dir = output_dir / "histories"
    checkpoints_dir = output_dir / "checkpoints"

    histories_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    tag = (
        f"{variant_name}__holdout_{held_out_mission.replace(' ', '_')}"
        f"__seed_{seed}"
    )

    history_path = histories_dir / f"{tag}.csv"
    checkpoint_path = checkpoints_dir / f"{tag}.pt"

    pd.DataFrame(history_rows).to_csv(
        history_path,
        index=False,
        encoding="utf-8-sig",
    )

    torch.save({
        "stage": "12C2",
        "script_version": SCRIPT_VERSION,
        "variant": variant_name,
        "held_out_mission": held_out_mission,
        "seed": int(seed),
        "model_freeze": asdict(FREEZE),
        "lambda_aw": float(lambda_aw),
        "parameter_count": int(parameter_count),
        "best_epoch_zero_based": int(best_epoch),
        "best_epoch_one_based": int(best_epoch + 1),
        "validation_metrics": confirmed,
        "state_dict": best_state,
        "scaler": scaler,
        "ridge_alpha": RIDGE_ALPHA,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "scientific_role": "development_only_LOMO_persistence_anchor_checkpoint",
        "held_out_test_accessed": False,
    }, checkpoint_path)

    result = {
        "method": variant_name,
        "held_out_mission": held_out_mission,
        "seed": int(seed),
        "status": "completed_finite",
        "lambda_AW": float(lambda_aw),
        "parameter_count": int(parameter_count),
        "best_epoch_zero_based": int(best_epoch),
        "best_epoch_one_based": int(best_epoch + 1),
        "elapsed_seconds": float(elapsed),
        **confirmed,
        "history_path": str(history_path),
        "checkpoint_path": str(checkpoint_path),
    }

    del (
        model,
        optimizer,
        scheduler,
        grad_scaler,
        X_train_z,
        y_train_z,
    )

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def append_result(path: Path, row: Dict):
    new = pd.DataFrame([row])

    if path.exists():
        old = pd.read_csv(path)

        if "seed" in row:
            mask = (
                (old["method"].astype(str) == str(row["method"]))
                & (
                    old["held_out_mission"].astype(str)
                    == str(row["held_out_mission"])
                )
                & (old["seed"].astype(int) == int(row["seed"]))
            )
        else:
            mask = (
                (old["method"].astype(str) == str(row["method"]))
                & (
                    old["held_out_mission"].astype(str)
                    == str(row["held_out_mission"])
                )
            )

        old = old.loc[~mask]

        combined = pd.concat(
            [old, new],
            ignore_index=True,
        )
    else:
        combined = new

    sort_cols = [
        c
        for c in ["method", "held_out_mission", "seed"]
        if c in combined.columns
    ]

    combined = combined.sort_values(sort_cols).reset_index(drop=True)

    combined.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def completed_neural_run_exists(
    path: Path,
    method: str,
    held_out_mission: str,
    seed: int,
):
    if not path.exists():
        return False

    df = pd.read_csv(path)

    required = {
        "method",
        "held_out_mission",
        "seed",
        "status",
        "AW_vector_RMSE_mps",
    }

    if not required.issubset(df.columns):
        return False

    sub = df.loc[
        (df["method"].astype(str) == method)
        & (df["held_out_mission"].astype(str) == held_out_mission)
        & (df["seed"].astype(int) == int(seed))
        & (df["status"].astype(str) == "completed_finite")
    ]

    if sub.empty:
        return False

    return bool(
        np.isfinite(
            float(sub.iloc[-1]["AW_vector_RMSE_mps"])
        )
    )


def summarize_seed_level(neural_df: pd.DataFrame):
    if neural_df.empty:
        return pd.DataFrame()

    metrics = [
        "wind_U_RMSE_mps",
        "wind_V_RMSE_mps",
        "wind_vector_RMSE_mps",
        "wind_speed_RMSE_mps",
        "wind_direction_MAE_deg",
        "wind_direction_RMSE_deg",
        "vessel_vector_RMSE_mps",
        "AW_vector_RMSE_mps",
        "AWS_RMSE_mps",
        "wind_gate_mean",
        "vessel_gate_mean",
        "wind_correction_z_RMS",
        "vessel_correction_z_RMS",
    ]

    rows = []

    for (method, mission), sub in neural_df.groupby(
        ["method", "held_out_mission"]
    ):
        row = {
            "method": method,
            "held_out_mission": mission,
            "seeds": int(len(sub)),
        }

        for metric in metrics:
            if metric in sub.columns:
                vals = sub[metric].to_numpy(dtype=float)
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals, ddof=0))

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["method", "held_out_mission"]
    ).reset_index(drop=True)


def summarize_methods(
    baseline_df: pd.DataFrame,
    seed_summary_df: pd.DataFrame,
):
    rows = []

    # Deterministic baselines: average over held-out missions.
    for method, sub in baseline_df.groupby("method"):
        row = {
            "method": method,
            "result_type": "deterministic_baseline",
            "mission_count": int(len(sub)),
            "seed_count": 0,
        }

        for metric in [
            "wind_U_RMSE_mps",
            "wind_V_RMSE_mps",
            "wind_vector_RMSE_mps",
            "wind_speed_RMSE_mps",
            "wind_direction_MAE_deg",
            "wind_direction_RMSE_deg",
            "vessel_vector_RMSE_mps",
            "AW_vector_RMSE_mps",
            "AWS_RMSE_mps",
        ]:
            row[f"{metric}_LOMO_mean"] = float(
                sub[metric].mean()
            )
            row[f"{metric}_LOMO_std"] = float(
                sub[metric].std(ddof=0)
            )

        rows.append(row)

    # Neural: first seed-mean per mission, then mission-average.
    if not seed_summary_df.empty:
        for method, sub in seed_summary_df.groupby("method"):
            row = {
                "method": method,
                "result_type": "neural_seed_mean_then_mission_mean",
                "mission_count": int(len(sub)),
                "seed_count": int(
                    sub["seeds"].min()
                ),
            }

            for metric in [
                "wind_U_RMSE_mps",
                "wind_V_RMSE_mps",
                "wind_vector_RMSE_mps",
                "wind_speed_RMSE_mps",
                "wind_direction_MAE_deg",
                "wind_direction_RMSE_deg",
                "vessel_vector_RMSE_mps",
                "AW_vector_RMSE_mps",
                "AWS_RMSE_mps",
            ]:
                col = f"{metric}_mean"

                if col in sub.columns:
                    values = sub[col].to_numpy(dtype=float)
                    row[f"{metric}_LOMO_mean"] = float(
                        np.mean(values)
                    )
                    row[f"{metric}_LOMO_std"] = float(
                        np.std(values, ddof=0)
                    )

            rows.append(row)

    return pd.DataFrame(rows).sort_values(
        "AW_vector_RMSE_mps_LOMO_mean"
    ).reset_index(drop=True)


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
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
    )

    parser.add_argument(
        "--debug-fast",
        action="store_true",
        help=(
            "NON-SCIENTIFIC: Antarctic holdout, one seed, short training."
        ),
    )

    args = parser.parse_args()

    project_root = Path(args.project_root)

    dataset_dir = (
        Path(args.dataset_dir)
        if args.dataset_dir
        else project_root
        / "data"
        / "forecasting"
        / "12B_Nie_same_mission_benchmark_v0_1"
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root
        / "data"
        / "forecasting"
        / (
            "12C2_PhysicsCompact_NieData_v2_PersistenceAnchor_LOMO_v0_1_DEBUG_FAST"
            if args.debug_fast
            else "12C2_PhysicsCompact_NieData_v2_PersistenceAnchor_LOMO_v0_1"
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = explicit_development_paths(dataset_dir)

    data = {
        mission: load_development_mission(path, mission)
        for mission, path in paths.items()
    }

    torch, nn, F = import_torch()
    configure_cuda(torch)

    device = (
        torch.device("cpu")
        if args.force_cpu or not torch.cuda.is_available()
        else torch.device("cuda:0")
    )

    active_missions = (
        ["Antarctic"]
        if args.debug_fast
        else DEVELOPMENT_MISSIONS
    )

    active_seeds = (
        [SCREENING_SEEDS[0]]
        if args.debug_fast
        else SCREENING_SEEDS
    )

    log("=" * 124)
    log("12C2 - PHYSICS-COMPACT-NIEDATA v2 PERSISTENCE-ANCHOR LOMO")
    log("=" * 124)
    log(f"script version        : {SCRIPT_VERSION}")
    log(f"dataset dir           : {dataset_dir}")
    log(f"output dir            : {output_dir}")
    log(f"device                : {device}")

    if device.type == "cuda":
        log(f"GPU                   : {torch.cuda.get_device_name(0)}")

    log(f"development missions  : {DEVELOPMENT_MISSIONS}")
    log(f"active holdouts       : {active_missions}")
    log(f"screening seeds       : {active_seeds}")
    log(f"Ridge alpha           : {RIDGE_ALPHA}")
    log(f"proposed lambda_AW    : {FREEZE.lambda_aw_physics}")
    log("Tropical Atlantic     : NOT ACCESSED")
    log("held-out test metrics : NONE")
    log("")

    for mission in DEVELOPMENT_MISSIONS:
        log(
            f"{mission:12s}: X={data[mission]['X'].shape} | "
            f"y={data[mission]['y'].shape} | "
            f"AW physics={data[mission]['physics_max_abs_error']:.3e}"
        )

    log("")

    # Freeze model definition before any LOMO results are used.
    freeze_payload = {
        "stage": "12C2",
        "model_name": "Physics-Compact-NieData v2.0 (Persistence Anchor)",
        "script_version": SCRIPT_VERSION,
        "scientific_status": "FROZEN_BEFORE_TROPICAL_ATLANTIC_TEST",
        "architecture": {
            "equations": [
                "W_hat_z = W_persist_z + sigmoid(gW)*DeltaW_z",
                "V_hat_z = V_persist_z + sigmoid(gV)*DeltaV_z",
                "A_hat = W_hat - V_hat in physical units",
            ],
            "input_features": FEATURE_NAMES,
            "targets": TARGET_NAMES,
            "model_freeze": asdict(FREEZE),
            "ridge_alpha": RIDGE_ALPHA,
            "delta_head_initialization": "all zeros; neural model starts exactly at Persistence",
        },
        "variants": {
            "CompactNoPhysicsV2": {
                "lambda_AW": 0.0,
                "role": "ablation_only",
            },
            "PhysicsCompactNieDataV2": {
                "lambda_AW": FREEZE.lambda_aw_physics,
                "role": "proposed_model_for_Stage12D",
            },
        },
        "LOMO_screening_seeds": SCREENING_SEEDS,
        "Stage12D_final_seeds": FINAL_STAGE12D_SEEDS,
        "checkpoint_metric": FREEZE.checkpoint_metric,
        "held_out_test_policy": (
            "Tropical Atlantic is not read by Stage 12C2. "
            "Architecture and loss remain frozen for Stage 12D."
        ),
    }

    save_json(
        output_dir / "proposed_model_freeze.json",
        freeze_payload,
    )

    fold_rows = []

    for held_out in DEVELOPMENT_MISSIONS:
        train_missions = [
            m for m in DEVELOPMENT_MISSIONS if m != held_out
        ]

        train_count = sum(len(data[m]["X"]) for m in train_missions)
        val_count = len(data[held_out]["X"])

        fold_rows.append({
            "held_out_mission": held_out,
            "training_missions": "+".join(train_missions),
            "train_samples": int(train_count),
            "validation_samples": int(val_count),
            "validation_mission_disjoint": True,
            "Tropical_Atlantic_accessed": False,
        })

    fold_df = pd.DataFrame(fold_rows)

    fold_df.to_csv(
        output_dir / "fold_definition.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline_path = output_dir / "baseline_fold_results.csv"
    neural_path = output_dir / "neural_run_results.csv"

    for held_out in active_missions:
        train_missions = [
            m for m in DEVELOPMENT_MISSIONS if m != held_out
        ]

        train = concat_missions([data[m] for m in train_missions])
        val = data[held_out]

        scaler = fit_scalers(
            train["X"],
            train["y"],
            train["aw"],
        )

        # Save fold scaler for audit/reproducibility.
        scaler_dir = output_dir / "fold_scalers"
        scaler_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            scaler_dir
            / f"holdout_{held_out.replace(' ', '_')}_train_only_scaler.npz",
            **scaler,
            train_missions=np.asarray(train_missions),
            validation_mission=np.asarray([held_out]),
            ridge_alpha=np.asarray([RIDGE_ALPHA], dtype=np.float64),
        )

        ridge = fit_ridge(
            train["X"],
            train["y"],
            scaler,
        )

        ridge_train_z = ridge_predict_z(
            ridge,
            train["X"],
            scaler,
        )

        ridge_val_z = ridge_predict_z(
            ridge,
            val["X"],
            scaler,
        )

        ridge_val_raw = inverse_y(
            ridge_val_z,
            scaler,
        )

        # v2 neural anchor: Persistence, expressed in train-fitted target z-space.
        persistence_train_z = persistence_predict_z(
            train["X"],
            scaler,
        )

        persistence_val_z = persistence_predict_z(
            val["X"],
            scaler,
        )

        p_y, p_aw = persistence_predict(
            val["X"]
        )

        p_metrics = evaluate_raw(
            val["y"],
            p_y,
            val["aw"],
        )

        r_metrics = evaluate_raw(
            val["y"],
            ridge_val_raw,
            val["aw"],
        )

        for method, metrics in [
            ("Persistence", p_metrics),
            ("Ridge", r_metrics),
        ]:
            row = {
                "method": method,
                "held_out_mission": held_out,
                "status": "completed_finite",
                "train_missions": "+".join(train_missions),
                "train_samples": int(len(train["X"])),
                "validation_samples": int(len(val["X"])),
                "ridge_alpha": (
                    RIDGE_ALPHA
                    if method == "Ridge"
                    else np.nan
                ),
                **metrics,
            }

            append_result(
                baseline_path,
                row,
            )

        log("-" * 124)
        log(
            f"[HOLDOUT {held_out}] train={train_missions} | "
            f"Ntrain={len(train['X']):,} | Nval={len(val['X']):,}"
        )
        log(
            f"  Persistence: W={p_metrics['wind_vector_RMSE_mps']:.4f} | "
            f"V={p_metrics['vessel_vector_RMSE_mps']:.4f} | "
            f"AW={p_metrics['AW_vector_RMSE_mps']:.4f}"
        )
        log(
            f"  Ridge      : W={r_metrics['wind_vector_RMSE_mps']:.4f} | "
            f"V={r_metrics['vessel_vector_RMSE_mps']:.4f} | "
            f"AW={r_metrics['AW_vector_RMSE_mps']:.4f}"
        )

        for variant_name, lambda_aw in NEURAL_VARIANTS.items():
            for seed in active_seeds:
                if completed_neural_run_exists(
                    neural_path,
                    variant_name,
                    held_out,
                    seed,
                ):
                    log(
                        f"  [RESUME] {variant_name} | "
                        f"holdout={held_out} | seed={seed}"
                    )
                    continue

                log(
                    f"  [TRAIN] {variant_name} | "
                    f"lambda_AW={lambda_aw:g} | seed={seed}"
                )

                result = train_neural_run(
                    torch=torch,
                    nn=nn,
                    F=F,
                    variant_name=variant_name,
                    lambda_aw=lambda_aw,
                    seed=seed,
                    held_out_mission=held_out,
                    X_train=train["X"],
                    y_train=train["y"],
                    aw_train=train["aw"],
                    X_val=val["X"],
                    y_val=val["y"],
                    aw_val=val["aw"],
                    persistence_train_z=persistence_train_z,
                    persistence_val_z=persistence_val_z,
                    scaler=scaler,
                    device=device,
                    output_dir=output_dir,
                    debug_fast=args.debug_fast,
                )

                append_result(
                    neural_path,
                    result,
                )

                log(
                    f"  [DONE] {variant_name} seed={seed}: "
                    f"best_epoch={result['best_epoch_one_based']} | "
                    f"W={result['wind_vector_RMSE_mps']:.4f} | "
                    f"V={result['vessel_vector_RMSE_mps']:.4f} | "
                    f"AW={result['AW_vector_RMSE_mps']:.4f} | "
                    f"WD_RMSE={result['wind_direction_RMSE_deg']:.2f} deg"
                )

        del (
            train,
            val,
            scaler,
            ridge,
            ridge_train_z,
            ridge_val_z,
            persistence_train_z,
            persistence_val_z,
            ridge_val_raw,
            p_y,
            p_aw,
        )

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    baseline_df = (
        pd.read_csv(baseline_path)
        if baseline_path.exists()
        else pd.DataFrame()
    )

    neural_df = (
        pd.read_csv(neural_path)
        if neural_path.exists()
        else pd.DataFrame()
    )

    seed_summary = summarize_seed_level(
        neural_df
    )

    if not seed_summary.empty:
        seed_summary.to_csv(
            output_dir / "seed_level_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # Scientific summary only when the full non-debug LOMO matrix is complete.
    expected_neural = {
        (method, mission, seed)
        for method in NEURAL_VARIANTS
        for mission in DEVELOPMENT_MISSIONS
        for seed in SCREENING_SEEDS
    }

    completed_neural = set()

    if not neural_df.empty:
        good = neural_df.loc[
            neural_df["status"].astype(str) == "completed_finite"
        ]

        for row in good.itertuples():
            completed_neural.add(
                (
                    str(row.method),
                    str(row.held_out_mission),
                    int(row.seed),
                )
            )

    full_neural_complete = expected_neural.issubset(
        completed_neural
    )

    expected_baseline = {
        (method, mission)
        for method in ["Persistence", "Ridge"]
        for mission in DEVELOPMENT_MISSIONS
    }

    completed_baseline = set()

    if not baseline_df.empty:
        good = baseline_df.loc[
            baseline_df["status"].astype(str) == "completed_finite"
        ]

        for row in good.itertuples():
            completed_baseline.add(
                (
                    str(row.method),
                    str(row.held_out_mission),
                )
            )

    full_baseline_complete = expected_baseline.issubset(
        completed_baseline
    )

    full_complete = bool(
        full_neural_complete
        and full_baseline_complete
        and not args.debug_fast
    )

    method_summary = pd.DataFrame()

    if full_complete:
        method_summary = summarize_methods(
            baseline_df,
            seed_summary,
        )

        method_summary.to_csv(
            output_dir / "lomo_method_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    manifest = {
        "stage": "12C2",
        "script_version": SCRIPT_VERSION,
        "debug_fast": bool(args.debug_fast),
        "scientific_full_LOMO_complete": bool(full_complete),
        "development_missions": DEVELOPMENT_MISSIONS,
        "held_out_test_mission": FORBIDDEN_TEST_MISSION,
        "held_out_test_accessed": False,
        "explicit_development_paths": {
            k: str(v) for k, v in paths.items()
        },
        "LOMO_screening_seeds": SCREENING_SEEDS,
        "active_seeds": active_seeds,
        "neural_variants": NEURAL_VARIANTS,
        "model_freeze": asdict(FREEZE),
        "ridge_alpha": RIDGE_ALPHA,
        "neural_anchor": "Persistence",
        "v1_to_v2_only_mechanism_change": (
            "Ridge anchor replaced by Persistence anchor; architecture, losses, "
            "folds, seeds, optimizer, and checkpoint rule unchanged"
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "platform": platform.platform(),
        "next_stage_policy": (
            "If full LOMO completes cleanly, keep PhysicsCompactNieData "
            "architecture/loss frozen. Stage12D fits all three development "
            "missions with five final seeds and only then evaluates Tropical "
            "Atlantic once."
        ),
    }

    save_json(
        output_dir / "training_manifest.json",
        manifest,
    )

    with (
        output_dir / "training_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "12C2 Physics-Compact-NieData v2 Persistence-Anchor Development-Only LOMO\n"
        )
        f.write("=" * 124 + "\n\n")

        f.write("DATA FIREWALL\n")
        f.write("-" * 124 + "\n")
        f.write("Antarctic accessed: YES\n")
        f.write("Atlantic accessed: YES\n")
        f.write("West Coast accessed: YES\n")
        f.write("Tropical Atlantic accessed: NO\n\n")

        f.write("MODEL FREEZE\n")
        f.write("-" * 124 + "\n")
        f.write(
            json.dumps(
                freeze_payload,
                ensure_ascii=False,
                indent=2,
            )
        )

        f.write("\n\nFOLD DEFINITION\n")
        f.write("-" * 124 + "\n")
        f.write(fold_df.to_string(index=False))

        f.write("\n\nBASELINES\n")
        f.write("-" * 124 + "\n")
        if not baseline_df.empty:
            f.write(baseline_df.to_string(index=False))

        f.write("\n\nNEURAL SEED SUMMARY\n")
        f.write("-" * 124 + "\n")
        if not seed_summary.empty:
            f.write(seed_summary.to_string(index=False))

        f.write("\n\nLOMO METHOD SUMMARY\n")
        f.write("-" * 124 + "\n")
        if full_complete and not method_summary.empty:
            f.write(method_summary.to_string(index=False))
        else:
            f.write(
                "Scientific LOMO summary not written yet because the complete "
                "non-debug matrix is not finished.\n"
            )

    log("")
    log("=" * 124)
    log("12C2 LOMO RESULTS")
    log("=" * 124)

    if full_complete and not method_summary.empty:
        for row in method_summary.itertuples():
            log(
                f"{row.method}: "
                f"W={row.wind_vector_RMSE_mps_LOMO_mean:.4f} | "
                f"WS={row.wind_speed_RMSE_mps_LOMO_mean:.4f} | "
                f"WD_RMSE={row.wind_direction_RMSE_deg_LOMO_mean:.2f} deg | "
                f"V={row.vessel_vector_RMSE_mps_LOMO_mean:.4f} | "
                f"AW={row.AW_vector_RMSE_mps_LOMO_mean:.4f} | "
                f"AWS={row.AWS_RMSE_mps_LOMO_mean:.4f}"
            )

        proposed = method_summary.loc[
            method_summary["method"] == "PhysicsCompactNieDataV2"
        ]

        ridge_summary = method_summary.loc[
            method_summary["method"] == "Ridge"
        ]

        persistence_summary = method_summary.loc[
            method_summary["method"] == "Persistence"
        ]

        if (
            not proposed.empty
            and not ridge_summary.empty
            and not persistence_summary.empty
        ):
            p = proposed.iloc[0]
            r = ridge_summary.iloc[0]
            q = persistence_summary.iloc[0]

            wind_skill_vs_ridge = 1.0 - (
                p["wind_vector_RMSE_mps_LOMO_mean"]
                / r["wind_vector_RMSE_mps_LOMO_mean"]
            )

            aw_skill_vs_ridge = 1.0 - (
                p["AW_vector_RMSE_mps_LOMO_mean"]
                / r["AW_vector_RMSE_mps_LOMO_mean"]
            )

            wind_skill_vs_persistence = 1.0 - (
                p["wind_vector_RMSE_mps_LOMO_mean"]
                / q["wind_vector_RMSE_mps_LOMO_mean"]
            )

            aw_skill_vs_persistence = 1.0 - (
                p["AW_vector_RMSE_mps_LOMO_mean"]
                / q["AW_vector_RMSE_mps_LOMO_mean"]
            )

            log("")
            log(
                f"PhysicsCompactNieDataV2 skill vs Persistence: "
                f"wind={100*wind_skill_vs_persistence:+.2f}% | "
                f"AW={100*aw_skill_vs_persistence:+.2f}%"
            )
            log(
                f"PhysicsCompactNieDataV2 skill vs Ridge      : "
                f"wind={100*wind_skill_vs_ridge:+.2f}% | "
                f"AW={100*aw_skill_vs_ridge:+.2f}%"
            )

        log("")
        log(
            "[READY] Full development-only LOMO is complete. "
            "Architecture/loss/persistence anchor remain frozen for Stage 12D."
        )

    else:
        missing_neural = sorted(
            expected_neural - completed_neural
        )

        log(
            f"[INCOMPLETE/DEBUG] Missing full scientific neural runs: "
            f"{len(missing_neural)}"
        )

        if len(missing_neural) <= 20:
            log(f"  missing={missing_neural}")

    log(
        "[POLICY] Tropical Atlantic held-out test was NOT accessed."
    )
    log(
        "[POLICY] PhysicsCompactNieDataV2 is the proposed Stage-12D candidate; "
        "CompactNoPhysicsV2 is ablation only."
    )
    log(f"[DONE] 12C2 outputs: {output_dir}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n[FATAL ERROR]", flush=True)
        traceback.print_exc()
        print(
            "\nPlease send the complete traceback and terminal output.",
            flush=True,
        )
        sys.exit(1)
