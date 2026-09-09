# -*- coding: utf-8 -*-
r"""
12D_final_PhysicsCompact_NieData_v2_TropicalAtlantic.py

Stage 12D
=========
FINAL frozen evaluation of Physics-Compact-NieData v2.0 on the held-out
Tropical Atlantic mission used in the Nie/BP-STGNN same-mission benchmark.

SCIENTIFIC STATUS
-----------------
Physics-Compact-NieData v2.0 was frozen at the end of Stage 12C2, BEFORE any
Tropical Atlantic model performance was accessed.

The only v1 -> v2 mechanism change was:
    Ridge residual anchor -> Persistence residual anchor.

Stage 12D does NOT tune architecture, loss weights, learning rate, batch size,
checkpoint metric, feature set, or prediction target.

FINAL TRAINING PROTOCOL
-----------------------
Development missions:
    Antarctic
    Atlantic
    West Coast

Held-out final test:
    Tropical Atlantic

Five final seeds:
    500043, 501052, 502061, 503070, 504079

There is no validation mission left during final all-development fitting, so
Stage 12D MUST NOT use Tropical Atlantic for early stopping.

Instead, the fixed number of training epochs is derived ONLY from the completed
Stage-12C2 development-only LOMO runs of the frozen proposed model:

For the three screening seeds:
    E(seed) = median of that seed's three LOMO best_epoch_one_based values.

For the two new final seeds:
    E(global) = median of all 3 x 3 = 9 LOMO best_epoch_one_based values.

Because there are 3 or 9 values, these medians are exact observed integer
epochs; no arbitrary rounding is required.

PERSISTENCE-ANCHORED PROPOSED MODEL
-----------------------------------
Track-J input:
    X in R^(6 x 9)
    [U, V, T, RH, SOG,
     COG_sin, COG_cos,
     WING_ANGLE_sin, WING_ANGLE_cos]

Target:
    [U, V, VESSEL_E, VESSEL_N] at the next 10-min block.

Persistence anchor:
    W_persist = last observed U,V
    V_E,persist = last SOG * last COG_sin
    V_N,persist = last SOG * last COG_cos

Standardized residual equations:
    W_hat_z = W_persist_z + sigmoid(g_W) * DeltaW_z
    V_hat_z = V_persist_z + sigmoid(g_V) * DeltaV_z

Physical consistency:
    A_hat = W_hat - V_hat

Frozen architecture:
    GRU hidden = 64
    GRU layers = 1
    Dense hidden = 64
    Dropout = 0.10

Frozen loss:
    L = L_direct + 2.0 * L_AW + 0.01 * L_res

    L_direct = 0.5*MSE_z(W) + 0.5*MSE_z(V)

    L_AW = normalized physical-unit apparent-wind component MSE

    L_res = mean((gW*DeltaW_z)^2) + mean((gV*DeltaV_z)^2)

Frozen optimizer/training:
    Adam
    LR = 1e-3
    batch = 256
    grad clip = 1.0
    chronological batches
    AMP on CUDA

Delta heads are initialized to zero, so epoch 0 equals Persistence exactly.

TEST-ACCESS FIREWALL
--------------------
The script has TWO phases.

Phase A -- before final test access:
    1. load ONLY Antarctic/Atlantic/West Coast;
    2. read Stage-12C2 development-only results;
    3. derive the frozen epoch plan;
    4. fit all-development scaler and Ridge baseline;
    5. train/save all five proposed-model checkpoints.

Only after ALL five final checkpoints exist and pass audit does the script:
    - write test_access_audit.json;
    - open Tropical_Atlantic_TEST.npz for the first final evaluation.

Use --preflight-only to validate paths, schemas, LOMO epoch plan, and the
development dataset WITHOUT opening Tropical Atlantic and WITHOUT training.

If test_access_audit.json already exists, a normal rerun refuses to access the
test again. Use --allow-repeat-test-access only for a genuine technical rerun
after the test had already been opened. The repeat is logged.

FINAL TEST METHODS
------------------
On the exact same Stage-12B held-out samples:
    Persistence
    Ridge(alpha=1.0)
    PhysicsCompactNieDataV2 -- five seeds

The proposed result is reported as mean +/- std across the five independently
trained final seeds. Persistence and Ridge are deterministic.

METRICS
-------
Wind:
    U RMSE
    V RMSE
    vector RMSE = sqrt(mean(dU^2+dV^2))
    wind-speed RMSE
    meteorological wind-direction MAE
    meteorological wind-direction RMSE

Vessel:
    E RMSE
    N RMSE
    vector RMSE

Apparent wind:
    E RMSE
    N RMSE
    vector RMSE
    apparent-wind-speed RMSE

No body-frame AWA is reported because Atlantic lacks true vessel HDG.

PUBLISHED NIE REFERENCE
-----------------------
The script writes a literature-reference comparison row using the published
BP-STGNN Tropical-Atlantic results:

    U RMSE  = 0.74864 m/s
    V RMSE  = 0.70506 m/s
    WS RMSE = 0.69940 m/s
    WD RMSE = 5.58400 deg

These values are NOT treated as sample-for-sample reproduced metrics.
The output explicitly labels the comparison as:
    same-mission / paper-informed, not exact official reproduction.

OUTPUTS
-------
12D_PhysicsCompact_NieData_v2_TropicalAtlantic_FINAL_v0_1/
    final_epoch_plan.csv
    final_training_manifest.json
    final_train_scaler.npz
    final_ridge_baseline.npz
    final_checkpoints/
    final_histories/
    test_access_audit.json
    test_predictions/
    tropical_atlantic_seed_results.csv
    tropical_atlantic_method_summary.csv
    published_BP_STGNN_comparison.csv
    final_results.json
    FINAL_REPORT.txt

RECOMMENDED FIRST COMMAND
-------------------------
Preflight only; DOES NOT OPEN Tropical Atlantic:

conda activate WindPredict
python "D:\project\WindPredict_SaildroneData\src\12D_final_PhysicsCompact_NieData_v2_TropicalAtlantic.py" --preflight-only

Then, if preflight passes, run the final experiment ONCE:

python "D:\project\WindPredict_SaildroneData\src\12D_final_PhysicsCompact_NieData_v2_TropicalAtlantic.py"
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import Ridge
except Exception as exc:
    raise RuntimeError(
        "scikit-learn is required. Activate the WindPredict environment."
    ) from exc


SCRIPT_VERSION = "0.1.0-PhysicsCompact-NieData-v2-FINAL-TropicalAtlantic"

DEFAULT_PROJECT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")

DEFAULT_12B_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12B_Nie_same_mission_benchmark_v0_1"
)

DEFAULT_12C2_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12C2_PhysicsCompact_NieData_v2_PersistenceAnchor_LOMO_v0_1"
)

DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12D_PhysicsCompact_NieData_v2_TropicalAtlantic_FINAL_v0_1"
)

DEVELOPMENT_MISSIONS = ["Antarctic", "Atlantic", "West Coast"]
TEST_MISSION = "Tropical Atlantic"

SCREENING_SEEDS = [500043, 501052, 502061]
FINAL_SEEDS = [500043, 501052, 502061, 503070, 504079]

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

RIDGE_ALPHA = 1.0

NIE_BP_STGNN_REFERENCE = {
    "wind_U_RMSE_mps": 0.74864,
    "wind_V_RMSE_mps": 0.70506,
    "wind_speed_RMSE_mps": 0.69940,
    "wind_direction_RMSE_deg": 5.58400,
}


@dataclass(frozen=True)
class ModelFreeze:
    gru_hidden: int = 64
    gru_layers: int = 1
    dense_hidden: int = 64
    dropout: float = 0.10

    lambda_aw: float = 2.0
    lambda_residual: float = 0.01

    learning_rate: float = 1e-3
    batch_size: int = 256
    grad_clip_norm: float = 1.0

    use_amp: bool = True
    train_shuffle: bool = False


FREEZE = ModelFreeze()


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
        json.dump(
            cv(obj),
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def explicit_development_paths(dataset_dir: Path):
    j_dir = dataset_dir / "track_J_joint_compatible"

    return {
        "Antarctic": j_dir / "Antarctic.npz",
        "Atlantic": j_dir / "Atlantic.npz",
        "West Coast": j_dir / "West_Coast.npz",
    }


def explicit_test_path(dataset_dir: Path):
    return (
        dataset_dir
        / "track_J_joint_compatible"
        / "Tropical_Atlantic_TEST.npz"
    )


def load_track_j(path: Path, mission: str):
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

        missing = [k for k in required if k not in data.files]

        if missing:
            raise KeyError(
                f"{path}: missing keys={missing}, available={data.files}"
            )

        X = np.asarray(data["X_raw"], dtype=np.float32)
        y = np.asarray(data["y_joint_raw"], dtype=np.float32)
        aw = np.asarray(data["apparent_ref"], dtype=np.float32)

        context = np.asarray(
            data["context_end_time_ns"],
            dtype=np.int64,
        ).reshape(-1)

        target_time = np.asarray(
            data["target_time_ns"],
            dtype=np.int64,
        )

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
    physics_max_abs_error = float(
        np.max(np.abs(physics - aw))
    )

    if physics_max_abs_error > 1e-5:
        raise RuntimeError(
            f"{mission}: AW physics mismatch, "
            f"max abs error={physics_max_abs_error:.3e}"
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
        "physics_max_abs_error": physics_max_abs_error,
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


def fit_scalers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    aw_train: np.ndarray,
):
    X_flat = X_train.astype(np.float64).reshape(
        -1,
        X_train.shape[-1],
    )
    y_flat = y_train.astype(np.float64).reshape(
        -1,
        y_train.shape[-1],
    )
    aw_flat = aw_train.astype(np.float64).reshape(
        -1,
        aw_train.shape[-1],
    )

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
    return np.asarray(
        X,
        dtype=np.float64,
    ).reshape(len(X), -1)


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

    pred = model.predict(
        flatten_X(Xz)
    )

    return pred.reshape(
        len(X),
        1,
        4,
    ).astype(np.float32)


def persistence_predict(X):
    last = np.asarray(
        X[:, -1, :],
        dtype=np.float64,
    )

    U = last[:, 0]
    V = last[:, 1]
    SOG = last[:, 4]

    cog_sin = last[:, 5]
    cog_cos = last[:, 6]

    norm = np.sqrt(
        cog_sin**2 + cog_cos**2
    )
    norm = np.where(
        norm < 1e-12,
        1.0,
        norm,
    )

    cog_sin = cog_sin / norm
    cog_cos = cog_cos / norm

    vessel_e = SOG * cog_sin
    vessel_n = SOG * cog_cos

    y = np.stack(
        [
            U,
            V,
            vessel_e,
            vessel_n,
        ],
        axis=1,
    )[:, None, :]

    aw = y[:, :, 0:2] - y[:, :, 2:4]

    return (
        y.astype(np.float32),
        aw.astype(np.float32),
    )


def persistence_predict_z(X, scaler):
    y_raw, _ = persistence_predict(X)

    return standardize_y(
        y_raw,
        scaler,
    ).astype(np.float32)


def wrap_deg180(x):
    return (
        x + 180.0
    ) % 360.0 - 180.0


def meteorological_direction_from(U, V):
    return (
        np.rad2deg(
            np.arctan2(
                -U,
                -V,
            )
        )
        % 360.0
    )


def evaluate_raw(
    y_true,
    y_pred,
    aw_true=None,
):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if aw_true is None:
        aw_true = (
            y_true[:, :, 0:2]
            - y_true[:, :, 2:4]
        )
    else:
        aw_true = np.asarray(
            aw_true,
            dtype=np.float64,
        )

    aw_pred = (
        y_pred[:, :, 0:2]
        - y_pred[:, :, 2:4]
    )

    wt = y_true[:, 0, 0:2]
    wp = y_pred[:, 0, 0:2]

    vt = y_true[:, 0, 2:4]
    vp = y_pred[:, 0, 2:4]

    at = aw_true[:, 0, :]
    ap = aw_pred[:, 0, :]

    wind_err = wp - wt
    vessel_err = vp - vt
    aw_err = ap - at

    wind_speed_true = np.linalg.norm(
        wt,
        axis=1,
    )
    wind_speed_pred = np.linalg.norm(
        wp,
        axis=1,
    )

    wind_dir_true = meteorological_direction_from(
        wt[:, 0],
        wt[:, 1],
    )

    wind_dir_pred = meteorological_direction_from(
        wp[:, 0],
        wp[:, 1],
    )

    wind_dir_err = wrap_deg180(
        wind_dir_pred - wind_dir_true
    )

    aws_true = np.linalg.norm(
        at,
        axis=1,
    )

    aws_pred = np.linalg.norm(
        ap,
        axis=1,
    )

    return {
        "wind_U_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    wind_err[:, 0] ** 2
                )
            )
        ),
        "wind_V_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    wind_err[:, 1] ** 2
                )
            )
        ),
        "wind_vector_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        wind_err**2,
                        axis=1,
                    )
                )
            )
        ),
        "wind_speed_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    (
                        wind_speed_pred
                        - wind_speed_true
                    )
                    ** 2
                )
            )
        ),
        "wind_direction_MAE_deg": float(
            np.mean(
                np.abs(
                    wind_dir_err
                )
            )
        ),
        "wind_direction_RMSE_deg": float(
            np.sqrt(
                np.mean(
                    wind_dir_err**2
                )
            )
        ),
        "vessel_E_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    vessel_err[:, 0] ** 2
                )
            )
        ),
        "vessel_N_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    vessel_err[:, 1] ** 2
                )
            )
        ),
        "vessel_vector_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        vessel_err**2,
                        axis=1,
                    )
                )
            )
        ),
        "AW_E_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    aw_err[:, 0] ** 2
                )
            )
        ),
        "AW_N_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    aw_err[:, 1] ** 2
                )
            )
        ),
        "AW_vector_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        aw_err**2,
                        axis=1,
                    )
                )
            )
        ),
        "AWS_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    (
                        aws_pred
                        - aws_true
                    )
                    ** 2
                )
            )
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
                nn.Linear(
                    FREEZE.gru_hidden,
                    FREEZE.dense_hidden,
                ),
                nn.GELU(),
                nn.Dropout(
                    FREEZE.dropout
                ),
            )

            self.wind_delta = nn.Linear(
                FREEZE.dense_hidden,
                2,
            )
            self.wind_gate = nn.Linear(
                FREEZE.dense_hidden,
                2,
            )

            self.vessel_delta = nn.Linear(
                FREEZE.dense_hidden,
                2,
            )
            self.vessel_gate = nn.Linear(
                FREEZE.dense_hidden,
                2,
            )

            # Frozen v2 initialization: epoch 0 == Persistence.
            nn.init.zeros_(
                self.wind_delta.weight
            )
            nn.init.zeros_(
                self.wind_delta.bias
            )
            nn.init.zeros_(
                self.vessel_delta.weight
            )
            nn.init.zeros_(
                self.vessel_delta.bias
            )

            nn.init.zeros_(
                self.wind_gate.bias
            )
            nn.init.zeros_(
                self.vessel_gate.bias
            )

        def forward(
            self,
            x,
            persistence_pred_z,
        ):
            seq, _ = self.gru(x)

            h = self.shared(
                seq[:, -1, :]
            )

            dw = self.wind_delta(h)
            gw = torch.sigmoid(
                self.wind_gate(h)
            )

            dv = self.vessel_delta(h)
            gv = torch.sigmoid(
                self.vessel_gate(h)
            )

            pred_w = (
                persistence_pred_z[:, 0, 0:2]
                + gw * dw
            )

            pred_v = (
                persistence_pred_z[:, 0, 2:4]
                + gv * dv
            )

            pred = torch.cat(
                [
                    pred_w,
                    pred_v,
                ],
                dim=1,
            )[:, None, :]

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
    return int(
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )


def create_grad_scaler(torch, enabled):
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
        return torch.cuda.amp.autocast(
            enabled=True
        )


def sequential_batches(
    *arrays,
    batch_size,
):
    n = len(arrays[0])

    for start in range(
        0,
        n,
        int(batch_size),
    ):
        end = min(
            n,
            start + int(batch_size),
        )

        yield tuple(
            arr[start:end]
            for arr in arrays
        )


def torch_scaler_constants(
    torch,
    scaler,
    device,
):
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
):
    pred_z = out["pred_z"]

    wind_mse = torch.mean(
        (
            pred_z[:, :, 0:2]
            - y_true_z[:, :, 0:2]
        )
        ** 2
    )

    vessel_mse = torch.mean(
        (
            pred_z[:, :, 2:4]
            - y_true_z[:, :, 2:4]
        )
        ** 2
    )

    direct = (
        0.5 * wind_mse
        + 0.5 * vessel_mse
    )

    y_mean = const["y_mean"][
        None,
        None,
        :,
    ]

    y_std = const["y_std"][
        None,
        None,
        :,
    ]

    pred_raw = (
        pred_z * y_std
        + y_mean
    )

    aw_pred_raw = (
        pred_raw[:, :, 0:2]
        - pred_raw[:, :, 2:4]
    )

    aw_scale = (
        const["aw_std"][
            None,
            None,
            :,
        ]
        .clamp_min(1e-6)
    )

    aw_loss = torch.mean(
        (
            (
                aw_pred_raw
                - aw_true_raw
            )
            / aw_scale
        )
        ** 2
    )

    residual = (
        torch.mean(
            out[
                "wind_correction_z"
            ]
            ** 2
        )
        + torch.mean(
            out[
                "vessel_correction_z"
            ]
            ** 2
        )
    )

    total = (
        direct
        + FREEZE.lambda_aw * aw_loss
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
    persistence_z,
    const,
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

    for (
        xb_np,
        yb_np,
        awb_np,
        pb_np,
    ) in sequential_batches(
        Xz,
        yz,
        aw_raw,
        persistence_z,
        batch_size=FREEZE.batch_size,
    ):
        xb = torch.from_numpy(
            xb_np
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

        awb = torch.from_numpy(
            awb_np
        ).to(
            device,
            non_blocking=True,
        )

        pb = torch.from_numpy(
            pb_np
        ).to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast_context(
            torch,
            amp_enabled,
        ):
            out = model(
                xb,
                pb,
            )

            losses = compute_training_loss(
                torch,
                out,
                yb,
                awb,
                const,
            )

        total = losses["total"]

        if not torch.isfinite(total):
            raise FloatingPointError(
                "Non-finite final-training loss."
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

        for key in sums:
            sums[key] += (
                float(
                    losses[key]
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )
                * bn
            )

        count += bn

        del (
            xb,
            yb,
            awb,
            pb,
            out,
            losses,
            total,
        )

    return {
        key: value / max(
            count,
            1,
        )
        for key, value
        in sums.items()
    }


def evaluate_neural(
    torch,
    model,
    X,
    y,
    aw,
    persistence_z,
    scaler,
    device,
    amp_enabled,
):
    model.eval()

    Xz = standardize_X(
        X,
        scaler,
    )

    preds = []
    wind_gates = []
    vessel_gates = []
    wind_corr = []
    vessel_corr = []

    with torch.no_grad():
        for xb_np, pb_np in sequential_batches(
            Xz,
            persistence_z,
            batch_size=FREEZE.batch_size,
        ):
            xb = torch.from_numpy(
                xb_np
            ).to(
                device,
                non_blocking=True,
            )

            pb = torch.from_numpy(
                pb_np
            ).to(
                device,
                non_blocking=True,
            )

            with autocast_context(
                torch,
                amp_enabled,
            ):
                out = model(
                    xb,
                    pb,
                )

            preds.append(
                out["pred_z"]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            wind_gates.append(
                out["wind_gate"]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            vessel_gates.append(
                out["vessel_gate"]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            wind_corr.append(
                out["wind_correction_z"]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            vessel_corr.append(
                out["vessel_correction_z"]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            del xb, pb, out

    pred_z = np.concatenate(
        preds,
        axis=0,
    )

    pred_raw = inverse_y(
        pred_z,
        scaler,
    )

    metrics = evaluate_raw(
        y,
        pred_raw,
        aw,
    )

    wg = np.concatenate(
        wind_gates,
        axis=0,
    )

    vg = np.concatenate(
        vessel_gates,
        axis=0,
    )

    wc = np.concatenate(
        wind_corr,
        axis=0,
    )

    vc = np.concatenate(
        vessel_corr,
        axis=0,
    )

    metrics.update({
        "wind_gate_mean": float(
            np.mean(wg)
        ),
        "wind_gate_std": float(
            np.std(wg)
        ),
        "vessel_gate_mean": float(
            np.mean(vg)
        ),
        "vessel_gate_std": float(
            np.std(vg)
        ),
        "wind_correction_z_RMS": float(
            np.sqrt(
                np.mean(
                    wc**2
                )
            )
        ),
        "vessel_correction_z_RMS": float(
            np.sqrt(
                np.mean(
                    vc**2
                )
            )
        ),
    })

    return metrics, pred_raw


def derive_epoch_plan(
    c2_results_path: Path,
):
    if not c2_results_path.exists():
        raise FileNotFoundError(
            c2_results_path
        )

    df = pd.read_csv(
        c2_results_path
    )

    required = {
        "method",
        "held_out_mission",
        "seed",
        "status",
        "best_epoch_one_based",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            f"12C2 result missing columns: {sorted(missing)}"
        )

    sub = df.loc[
        (
            df["method"].astype(str)
            == "PhysicsCompactNieDataV2"
        )
        & (
            df["status"].astype(str)
            == "completed_finite"
        )
    ].copy()

    expected = {
        (mission, seed)
        for mission in DEVELOPMENT_MISSIONS
        for seed in SCREENING_SEEDS
    }

    observed = {
        (
            str(row.held_out_mission),
            int(row.seed),
        )
        for row in sub.itertuples()
    }

    missing_runs = expected - observed

    if missing_runs:
        raise RuntimeError(
            "Stage-12C2 proposed-model LOMO matrix incomplete. "
            f"Missing={sorted(missing_runs)}"
        )

    # Enforce exactly one completed row per required mission/seed.
    dedup_check = (
        sub.groupby(
            [
                "held_out_mission",
                "seed",
            ]
        )
        .size()
    )

    duplicates = dedup_check.loc[
        dedup_check > 1
    ]

    if len(duplicates):
        raise RuntimeError(
            "Duplicate completed Stage-12C2 rows detected for "
            f"mission/seed pairs:\n{duplicates}"
        )

    required_sub = sub.loc[
        sub["held_out_mission"].isin(
            DEVELOPMENT_MISSIONS
        )
        & sub["seed"].astype(int).isin(
            SCREENING_SEEDS
        )
    ].copy()

    epochs = required_sub[
        "best_epoch_one_based"
    ].astype(int)

    if np.any(epochs <= 0):
        raise RuntimeError(
            "Stage-12C2 best epochs must be positive."
        )

    global_median = int(
        np.median(
            epochs.to_numpy(
                dtype=int
            )
        )
    )

    rows = []

    for seed in FINAL_SEEDS:
        if seed in SCREENING_SEEDS:
            seed_rows = required_sub.loc[
                required_sub["seed"].astype(int)
                == int(seed)
            ]

            values = (
                seed_rows[
                    "best_epoch_one_based"
                ]
                .astype(int)
                .to_numpy()
            )

            if len(values) != 3:
                raise RuntimeError(
                    f"Seed {seed}: expected 3 LOMO best epochs, "
                    f"got {len(values)}."
                )

            final_epoch = int(
                np.median(
                    values
                )
            )

            source = (
                "median_of_same_seed_three_development_LOMO_best_epochs"
            )

            source_values = ",".join(
                str(int(x))
                for x in sorted(
                    values.tolist()
                )
            )

        else:
            final_epoch = global_median

            source = (
                "global_median_of_9_proposed_LOMO_best_epochs"
            )

            source_values = ",".join(
                str(int(x))
                for x in sorted(
                    epochs.to_numpy(
                        dtype=int
                    ).tolist()
                )
            )

        rows.append({
            "seed": int(seed),
            "final_train_epochs": int(final_epoch),
            "epoch_source": source,
            "source_best_epochs": source_values,
            "Tropical_Atlantic_used_for_epoch_selection": False,
        })

    plan = pd.DataFrame(
        rows
    )

    audit = {
        "proposed_LOMO_rows_used": int(
            len(
                required_sub
            )
        ),
        "global_median_epoch": int(
            global_median
        ),
        "all_9_best_epochs": sorted(
            epochs.to_numpy(
                dtype=int
            ).tolist()
        ),
        "min_best_epoch": int(
            epochs.min()
        ),
        "max_best_epoch": int(
            epochs.max()
        ),
    }

    return plan, audit


def validate_12c2_freeze(
    freeze_path: Path,
):
    if not freeze_path.exists():
        raise FileNotFoundError(
            freeze_path
        )

    with freeze_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        obj = json.load(f)

    model_name = str(
        obj.get(
            "model_name",
            "",
        )
    )

    if "v2.0" not in model_name.lower():
        raise RuntimeError(
            "12C2 freeze file does not identify Physics-Compact-NieData v2.0."
        )

    variants = obj.get(
        "variants",
        {}
    )

    proposed = variants.get(
        "PhysicsCompactNieDataV2"
    )

    if proposed is None:
        raise RuntimeError(
            "12C2 freeze file does not contain PhysicsCompactNieDataV2."
        )

    lambda_aw = float(
        proposed.get(
            "lambda_AW",
            np.nan,
        )
    )

    if not np.isfinite(lambda_aw) or abs(
        lambda_aw - FREEZE.lambda_aw
    ) > 1e-12:
        raise RuntimeError(
            f"12C2 lambda_AW mismatch: {lambda_aw} vs {FREEZE.lambda_aw}"
        )

    architecture = obj.get(
        "architecture",
        {}
    )

    init_text = str(
        architecture.get(
            "delta_head_initialization",
            "",
        )
    ).lower()

    if "persistence" not in init_text:
        raise RuntimeError(
            "12C2 freeze does not confirm Persistence initialization."
        )

    return obj


def state_dict_cpu(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def train_final_seed(
    *,
    torch,
    nn,
    F,
    seed,
    epochs,
    train,
    scaler,
    persistence_train_z,
    device,
    output_dir,
):
    seed_everything(
        torch,
        seed,
    )

    ModelClass = build_model_class(
        torch,
        nn,
        F,
    )

    model = ModelClass().to(
        device
    )

    parameter_count = count_parameters(
        model
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FREEZE.learning_rate,
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

    Xz = standardize_X(
        train["X"],
        scaler,
    )

    yz = standardize_y(
        train["y"],
        scaler,
    )

    history_rows = []

    log(
        f"  [FINAL TRAIN] seed={seed} | fixed_epochs={epochs}"
    )

    start_time = time.perf_counter()

    for epoch in range(
        int(epochs)
    ):
        stats = train_one_epoch(
            torch,
            model,
            optimizer,
            grad_scaler,
            Xz,
            yz,
            train["aw"],
            persistence_train_z,
            const,
            device,
            amp_enabled,
        )

        history_rows.append({
            "seed": int(seed),
            "epoch_zero_based": int(epoch),
            "epoch_one_based": int(epoch + 1),
            "learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            **{
                f"train_{k}": v
                for k, v
                in stats.items()
            },
        })

        log(
            f"      epoch={epoch+1:03d}/{epochs} | "
            f"L={stats['total']:.6f} | "
            f"direct={stats['direct']:.6f} | "
            f"AWloss={stats['aw_loss_norm']:.6f}"
        )

    elapsed = time.perf_counter() - start_time

    checkpoint_dir = (
        output_dir
        / "final_checkpoints"
    )

    history_dir = (
        output_dir
        / "final_histories"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_dir
        / f"PhysicsCompactNieDataV2_seed_{seed}.pt"
    )

    history_path = (
        history_dir
        / f"PhysicsCompactNieDataV2_seed_{seed}.csv"
    )

    pd.DataFrame(
        history_rows
    ).to_csv(
        history_path,
        index=False,
        encoding="utf-8-sig",
    )

    torch.save({
        "stage": "12D",
        "script_version": SCRIPT_VERSION,
        "model_name": "Physics-Compact-NieData v2.0",
        "anchor": "Persistence",
        "seed": int(seed),
        "fixed_train_epochs": int(epochs),
        "model_freeze": asdict(FREEZE),
        "parameter_count": int(parameter_count),
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "state_dict": state_dict_cpu(model),
        "scientific_role": (
            "final_all_development_checkpoint_before_Tropical_Atlantic_test"
        ),
        "Tropical_Atlantic_used_for_training": False,
        "Tropical_Atlantic_used_for_epoch_selection": False,
    }, checkpoint_path)

    result = {
        "seed": int(seed),
        "fixed_train_epochs": int(epochs),
        "parameter_count": int(parameter_count),
        "elapsed_seconds": float(elapsed),
        "checkpoint_path": str(
            checkpoint_path
        ),
        "history_path": str(
            history_path
        ),
    }

    del (
        model,
        optimizer,
        grad_scaler,
        Xz,
        yz,
    )

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def final_checkpoint_valid(
    torch,
    checkpoint_path: Path,
    seed: int,
    expected_epochs: int,
):
    if not checkpoint_path.exists():
        return False

    try:
        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    checks = [
        str(
            ckpt.get(
                "model_name",
                "",
            )
        )
        == "Physics-Compact-NieData v2.0",
        str(
            ckpt.get(
                "anchor",
                "",
            )
        )
        == "Persistence",
        int(
            ckpt.get(
                "seed",
                -1,
            )
        )
        == int(seed),
        int(
            ckpt.get(
                "fixed_train_epochs",
                -1,
            )
        )
        == int(expected_epochs),
        bool(
            ckpt.get(
                "Tropical_Atlantic_used_for_training",
                True,
            )
        )
        is False,
        bool(
            ckpt.get(
                "Tropical_Atlantic_used_for_epoch_selection",
                True,
            )
        )
        is False,
        "state_dict" in ckpt,
    ]

    return bool(
        all(
            checks
        )
    )


def load_final_model(
    torch,
    nn,
    F,
    checkpoint_path: Path,
    device,
):
    try:
        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    ModelClass = build_model_class(
        torch,
        nn,
        F,
    )

    model = ModelClass().to(
        device
    )

    model.load_state_dict(
        ckpt["state_dict"],
        strict=True,
    )

    return model, ckpt


def summarize_seed_results(
    seed_df: pd.DataFrame,
):
    metric_cols = [
        "wind_U_RMSE_mps",
        "wind_V_RMSE_mps",
        "wind_vector_RMSE_mps",
        "wind_speed_RMSE_mps",
        "wind_direction_MAE_deg",
        "wind_direction_RMSE_deg",
        "vessel_E_RMSE_mps",
        "vessel_N_RMSE_mps",
        "vessel_vector_RMSE_mps",
        "AW_E_RMSE_mps",
        "AW_N_RMSE_mps",
        "AW_vector_RMSE_mps",
        "AWS_RMSE_mps",
        "wind_gate_mean",
        "wind_gate_std",
        "vessel_gate_mean",
        "vessel_gate_std",
        "wind_correction_z_RMS",
        "vessel_correction_z_RMS",
    ]

    out = {
        "method": "PhysicsCompactNieDataV2",
        "seed_count": int(
            len(
                seed_df
            )
        ),
    }

    for col in metric_cols:
        vals = seed_df[
            col
        ].to_numpy(
            dtype=float
        )

        out[
            f"{col}_mean"
        ] = float(
            np.mean(
                vals
            )
        )

        out[
            f"{col}_std"
        ] = float(
            np.std(
                vals,
                ddof=0,
            )
        )

    return out


def make_method_summary(
    persistence_metrics,
    ridge_metrics,
    proposed_summary,
):
    rows = []

    for method, metrics in [
        (
            "Persistence",
            persistence_metrics,
        ),
        (
            "Ridge",
            ridge_metrics,
        ),
    ]:
        row = {
            "method": method,
            "seed_count": 0,
        }

        for key, value in metrics.items():
            row[
                f"{key}_mean"
            ] = float(
                value
            )

            row[
                f"{key}_std"
            ] = 0.0

        rows.append(
            row
        )

    rows.append(
        proposed_summary
    )

    return pd.DataFrame(
        rows
    )


def create_published_comparison(
    method_summary: pd.DataFrame,
):
    rows = []

    rows.append({
        "method": "BP-STGNN (Nie et al., published)",
        "U_RMSE_mps_mean": NIE_BP_STGNN_REFERENCE[
            "wind_U_RMSE_mps"
        ],
        "U_RMSE_mps_std": np.nan,
        "V_RMSE_mps_mean": NIE_BP_STGNN_REFERENCE[
            "wind_V_RMSE_mps"
        ],
        "V_RMSE_mps_std": np.nan,
        "WS_RMSE_mps_mean": NIE_BP_STGNN_REFERENCE[
            "wind_speed_RMSE_mps"
        ],
        "WS_RMSE_mps_std": np.nan,
        "WD_RMSE_deg_mean": NIE_BP_STGNN_REFERENCE[
            "wind_direction_RMSE_deg"
        ],
        "WD_RMSE_deg_std": np.nan,
        "comparison_scope": (
            "published same-mission reference; paper-informed comparison, "
            "not exact sample-for-sample reproduction"
        ),
    })

    for method in [
        "Persistence",
        "Ridge",
        "PhysicsCompactNieDataV2",
    ]:
        sub = method_summary.loc[
            method_summary["method"]
            == method
        ]

        if sub.empty:
            continue

        row = sub.iloc[
            0
        ]

        rows.append({
            "method": method,
            "U_RMSE_mps_mean": row[
                "wind_U_RMSE_mps_mean"
            ],
            "U_RMSE_mps_std": row[
                "wind_U_RMSE_mps_std"
            ],
            "V_RMSE_mps_mean": row[
                "wind_V_RMSE_mps_mean"
            ],
            "V_RMSE_mps_std": row[
                "wind_V_RMSE_mps_std"
            ],
            "WS_RMSE_mps_mean": row[
                "wind_speed_RMSE_mps_mean"
            ],
            "WS_RMSE_mps_std": row[
                "wind_speed_RMSE_mps_std"
            ],
            "WD_RMSE_deg_mean": row[
                "wind_direction_RMSE_deg_mean"
            ],
            "WD_RMSE_deg_std": row[
                "wind_direction_RMSE_deg_std"
            ],
            "comparison_scope": (
                "Stage-12B Tropical Atlantic held-out samples; "
                "same raw mission and 10-min time scale"
            ),
        })

    return pd.DataFrame(
        rows
    )


def skill_percent(
    proposed,
    baseline,
):
    return 100.0 * (
        1.0
        - float(proposed)
        / float(baseline)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        default=str(
            DEFAULT_PROJECT_ROOT
        ),
    )

    parser.add_argument(
        "--dataset-dir",
        default=None,
    )

    parser.add_argument(
        "--c2-dir",
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
        "--preflight-only",
        action="store_true",
        help=(
            "Validate development data and frozen epoch plan. "
            "DOES NOT open Tropical Atlantic and DOES NOT train."
        ),
    )

    parser.add_argument(
        "--allow-repeat-test-access",
        action="store_true",
        help=(
            "Allow rerunning the final test after test_access_audit.json "
            "already exists. Use only for a genuine technical rerun."
        ),
    )

    args = parser.parse_args()

    project_root = Path(
        args.project_root
    )

    dataset_dir = (
        Path(
            args.dataset_dir
        )
        if args.dataset_dir
        else project_root
        / "data"
        / "forecasting"
        / "12B_Nie_same_mission_benchmark_v0_1"
    )

    c2_dir = (
        Path(
            args.c2_dir
        )
        if args.c2_dir
        else project_root
        / "data"
        / "forecasting"
        / "12C2_PhysicsCompact_NieData_v2_PersistenceAnchor_LOMO_v0_1"
    )

    output_dir = (
        Path(
            args.output_dir
        )
        if args.output_dir
        else project_root
        / "data"
        / "forecasting"
        / "12D_PhysicsCompact_NieData_v2_TropicalAtlantic_FINAL_v0_1"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    development_paths = explicit_development_paths(
        dataset_dir
    )

    test_path = explicit_test_path(
        dataset_dir
    )

    # Path existence is allowed during preflight; the held-out NPZ is NOT opened.
    if not test_path.exists():
        raise FileNotFoundError(
            f"Held-out test file path does not exist: {test_path}"
        )

    c2_results_path = (
        c2_dir
        / "neural_run_results.csv"
    )

    c2_freeze_path = (
        c2_dir
        / "proposed_model_freeze.json"
    )

    log("=" * 126)
    log("12D - FINAL PHYSICS-COMPACT-NIEDATA v2 TROPICAL ATLANTIC EVALUATION")
    log("=" * 126)
    log(
        f"script version        : {SCRIPT_VERSION}"
    )
    log(
        f"dataset dir           : {dataset_dir}"
    )
    log(
        f"12C2 dir              : {c2_dir}"
    )
    log(
        f"output dir            : {output_dir}"
    )
    log(
        f"final seeds           : {FINAL_SEEDS}"
    )
    log(
        "model                 : Physics-Compact-NieData v2.0"
    )
    log(
        "anchor                : Persistence"
    )
    log(
        "Tropical Atlantic     : NOT OPENED YET"
    )
    log("")

    # ------------------------------------------------------------------
    # PHASE A: DEVELOPMENT-ONLY PREPARATION
    # ------------------------------------------------------------------
    log(
        "[PHASE A] Development-only final-training preparation"
    )

    freeze_obj = validate_12c2_freeze(
        c2_freeze_path
    )

    epoch_plan, epoch_audit = derive_epoch_plan(
        c2_results_path
    )

    epoch_plan_path = (
        output_dir
        / "final_epoch_plan.csv"
    )

    epoch_plan.to_csv(
        epoch_plan_path,
        index=False,
        encoding="utf-8-sig",
    )

    log(
        f"  Stage-12C2 proposed LOMO best epochs: "
        f"{epoch_audit['all_9_best_epochs']}"
    )

    log(
        f"  global median epoch : "
        f"{epoch_audit['global_median_epoch']}"
    )

    for row in epoch_plan.itertuples():
        log(
            f"  seed={row.seed}: final_epochs={row.final_train_epochs} | "
            f"{row.epoch_source}"
        )

    data = {}

    for mission, path in development_paths.items():
        item = load_track_j(
            path,
            mission,
        )

        data[
            mission
        ] = item

        log(
            f"  {mission:12s}: "
            f"X={item['X'].shape} | "
            f"y={item['y'].shape} | "
            f"AW physics={item['physics_max_abs_error']:.3e}"
        )

    development = concat_missions(
        [
            data[m]
            for m in DEVELOPMENT_MISSIONS
        ]
    )

    log(
        f"  all development     : "
        f"X={development['X'].shape} | "
        f"y={development['y'].shape}"
    )

    scaler = fit_scalers(
        development["X"],
        development["y"],
        development["aw"],
    )

    scaler_path = (
        output_dir
        / "final_train_scaler.npz"
    )

    np.savez_compressed(
        scaler_path,
        **scaler,
        training_missions=np.asarray(
            DEVELOPMENT_MISSIONS
        ),
        Tropical_Atlantic_used=False,
    )

    ridge = fit_ridge(
        development["X"],
        development["y"],
        scaler,
    )

    ridge_path = (
        output_dir
        / "final_ridge_baseline.npz"
    )

    np.savez_compressed(
        ridge_path,
        coef=np.asarray(
            ridge.coef_,
            dtype=np.float64,
        ),
        intercept=np.asarray(
            ridge.intercept_,
            dtype=np.float64,
        ),
        alpha=np.asarray(
            [RIDGE_ALPHA],
            dtype=np.float64,
        ),
        feature_names=np.asarray(
            FEATURE_NAMES
        ),
        target_names=np.asarray(
            TARGET_NAMES
        ),
        Tropical_Atlantic_used_for_fit=np.asarray(
            [False]
        ),
    )

    persistence_train_z = persistence_predict_z(
        development["X"],
        scaler,
    )

    preflight_manifest = {
        "stage": "12D",
        "script_version": SCRIPT_VERSION,
        "phase": "development_only_pre_final_test",
        "model_name": "Physics-Compact-NieData v2.0",
        "anchor": "Persistence",
        "model_freeze": asdict(
            FREEZE
        ),
        "development_missions": DEVELOPMENT_MISSIONS,
        "development_samples": int(
            len(
                development["X"]
            )
        ),
        "held_out_test_mission": TEST_MISSION,
        "held_out_test_path_exists": bool(
            test_path.exists()
        ),
        "held_out_test_opened": False,
        "final_seeds": FINAL_SEEDS,
        "epoch_plan": epoch_plan.to_dict(
            orient="records"
        ),
        "epoch_audit": epoch_audit,
        "Stage12C2_freeze_path": str(
            c2_freeze_path
        ),
        "Stage12C2_results_path": str(
            c2_results_path
        ),
        "comparison_scope": (
            "same raw Tropical Atlantic mission and 10-min scale; "
            "paper-informed, not exact official reproduction"
        ),
    }

    save_json(
        output_dir
        / "final_training_manifest.json",
        preflight_manifest,
    )

    if args.preflight_only:
        log("")
        log("=" * 126)
        log("12D PREFLIGHT PASSED")
        log("=" * 126)
        log(
            "[PASS] Stage-12C2 freeze validated."
        )
        log(
            "[PASS] Final epoch schedule derived only from development LOMO."
        )
        log(
            "[PASS] Antarctic/Atlantic/West Coast schemas and physics passed."
        )
        log(
            "[PASS] Final all-development scaler and Ridge were fit without test data."
        )
        log(
            "[POLICY] Tropical Atlantic NPZ was NOT opened."
        )
        log(
            "[NEXT] Run the same script without --preflight-only "
            "for the final experiment."
        )
        return 0

    # ------------------------------------------------------------------
    # PHASE B: TRAIN ALL FIVE FINAL MODELS BEFORE OPENING TEST
    # ------------------------------------------------------------------
    torch, nn, F = import_torch()
    configure_cuda(
        torch
    )

    device = (
        torch.device("cpu")
        if (
            args.force_cpu
            or not torch.cuda.is_available()
        )
        else torch.device("cuda:0")
    )

    log("")
    log(
        "[PHASE B] Final all-development training -- "
        "Tropical Atlantic still NOT OPENED"
    )

    log(
        f"  device                : {device}"
    )

    if device.type == "cuda":
        log(
            f"  GPU                   : {torch.cuda.get_device_name(0)}"
        )

    training_rows = []

    checkpoint_dir = (
        output_dir
        / "final_checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for plan_row in epoch_plan.itertuples():
        seed = int(
            plan_row.seed
        )

        epochs = int(
            plan_row.final_train_epochs
        )

        checkpoint_path = (
            checkpoint_dir
            / f"PhysicsCompactNieDataV2_seed_{seed}.pt"
        )

        if final_checkpoint_valid(
            torch,
            checkpoint_path,
            seed,
            epochs,
        ):
            log(
                f"  [RESUME] seed={seed}: valid final checkpoint exists."
            )

            try:
                ckpt = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                ckpt = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                )

            training_rows.append({
                "seed": seed,
                "fixed_train_epochs": epochs,
                "parameter_count": int(
                    ckpt[
                        "parameter_count"
                    ]
                ),
                "elapsed_seconds": np.nan,
                "checkpoint_path": str(
                    checkpoint_path
                ),
                "history_path": str(
                    output_dir
                    / "final_histories"
                    / f"PhysicsCompactNieDataV2_seed_{seed}.csv"
                ),
                "status": "resumed_valid_checkpoint",
            })

            continue

        result = train_final_seed(
            torch=torch,
            nn=nn,
            F=F,
            seed=seed,
            epochs=epochs,
            train=development,
            scaler=scaler,
            persistence_train_z=persistence_train_z,
            device=device,
            output_dir=output_dir,
        )

        result[
            "status"
        ] = "trained_now"

        training_rows.append(
            result
        )

        if not final_checkpoint_valid(
            torch,
            Path(
                result[
                    "checkpoint_path"
                ]
            ),
            seed,
            epochs,
        ):
            raise RuntimeError(
                f"Final checkpoint audit failed after training seed={seed}."
            )

    final_training_df = pd.DataFrame(
        training_rows
    )

    final_training_df.to_csv(
        output_dir
        / "final_training_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Audit all five checkpoints one more time BEFORE test access.
    for plan_row in epoch_plan.itertuples():
        seed = int(
            plan_row.seed
        )

        epochs = int(
            plan_row.final_train_epochs
        )

        checkpoint_path = (
            checkpoint_dir
            / f"PhysicsCompactNieDataV2_seed_{seed}.pt"
        )

        if not final_checkpoint_valid(
            torch,
            checkpoint_path,
            seed,
            epochs,
        ):
            raise RuntimeError(
                f"Pre-test checkpoint audit failed: seed={seed}"
            )

    log(
        "[PASS] All five final model checkpoints exist and passed audit."
    )
    log(
        "[PASS] No Tropical Atlantic data have been opened during final training."
    )

    # ------------------------------------------------------------------
    # PHASE C: ONE-SHOT HELD-OUT TEST ACCESS
    # ------------------------------------------------------------------
    test_access_path = (
        output_dir
        / "test_access_audit.json"
    )

    if (
        test_access_path.exists()
        and not args.allow_repeat_test_access
    ):
        raise RuntimeError(
            "test_access_audit.json already exists. This indicates that "
            "Tropical Atlantic was already opened by a previous final run. "
            "The script refuses repeat test access by default. If this is a "
            "genuine technical rerun, explicitly pass --allow-repeat-test-access."
        )

    repeat_access = bool(
        test_access_path.exists()
    )

    access_record = {
        "stage": "12D",
        "event": "Tropical_Atlantic_FINAL_TEST_OPEN",
        "UTC_time": datetime.now(
            timezone.utc
        ).isoformat(),
        "repeat_access": repeat_access,
        "repeat_access_explicitly_allowed": bool(
            args.allow_repeat_test_access
        ),
        "all_five_final_checkpoints_completed_before_access": True,
        "epoch_selection_used_test": False,
        "scaler_fit_used_test": False,
        "ridge_fit_used_test": False,
        "model_training_used_test": False,
        "test_path": str(
            test_path
        ),
    }

    save_json(
        test_access_path,
        access_record,
    )

    log("")
    log("=" * 126)
    log("[PHASE C] OPENING HELD-OUT TROPICAL ATLANTIC FINAL TEST")
    log("=" * 126)
    log(
        f"  repeat access        : {repeat_access}"
    )
    log(
        "  model selection      : COMPLETE BEFORE TEST"
    )
    log(
        "  scaler/Ridge fitting : COMPLETE BEFORE TEST"
    )

    test = load_track_j(
        test_path,
        TEST_MISSION,
    )

    log(
        f"  Tropical Atlantic    : "
        f"X={test['X'].shape} | "
        f"y={test['y'].shape} | "
        f"AW physics={test['physics_max_abs_error']:.3e}"
    )

    persistence_test_raw, _ = persistence_predict(
        test["X"]
    )

    persistence_metrics = evaluate_raw(
        test["y"],
        persistence_test_raw,
        test["aw"],
    )

    ridge_test_z = ridge_predict_z(
        ridge,
        test["X"],
        scaler,
    )

    ridge_test_raw = inverse_y(
        ridge_test_z,
        scaler,
    )

    ridge_metrics = evaluate_raw(
        test["y"],
        ridge_test_raw,
        test["aw"],
    )

    persistence_test_z = persistence_predict_z(
        test["X"],
        scaler,
    )

    prediction_dir = (
        output_dir
        / "test_predictions"
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        prediction_dir
        / "baselines_TropicalAtlantic.npz",
        y_true=test["y"],
        apparent_true=test["aw"],
        persistence_y_pred=persistence_test_raw,
        ridge_y_pred=ridge_test_raw,
        context_end_time_ns=test[
            "context_end_time_ns"
        ],
        target_time_ns=test[
            "target_time_ns"
        ],
    )

    seed_rows = []

    for plan_row in epoch_plan.itertuples():
        seed = int(
            plan_row.seed
        )

        epochs = int(
            plan_row.final_train_epochs
        )

        checkpoint_path = (
            checkpoint_dir
            / f"PhysicsCompactNieDataV2_seed_{seed}.pt"
        )

        model, ckpt = load_final_model(
            torch,
            nn,
            F,
            checkpoint_path,
            device,
        )

        amp_enabled = bool(
            FREEZE.use_amp
            and device.type == "cuda"
        )

        metrics, pred_raw = evaluate_neural(
            torch,
            model,
            test["X"],
            test["y"],
            test["aw"],
            persistence_test_z,
            scaler,
            device,
            amp_enabled,
        )

        row = {
            "method": "PhysicsCompactNieDataV2",
            "seed": seed,
            "fixed_train_epochs": epochs,
            "test_mission": TEST_MISSION,
            "test_samples": int(
                len(
                    test["X"]
                )
            ),
            **metrics,
        }

        seed_rows.append(
            row
        )

        np.savez_compressed(
            prediction_dir
            / f"PhysicsCompactNieDataV2_seed_{seed}_TropicalAtlantic.npz",
            y_true=test["y"],
            y_pred=pred_raw.astype(
                np.float32
            ),
            apparent_true=test["aw"],
            apparent_pred=(
                pred_raw[:, :, 0:2]
                - pred_raw[:, :, 2:4]
            ).astype(
                np.float32
            ),
            context_end_time_ns=test[
                "context_end_time_ns"
            ],
            target_time_ns=test[
                "target_time_ns"
            ],
            seed=np.asarray(
                [seed],
                dtype=np.int64,
            ),
            fixed_train_epochs=np.asarray(
                [epochs],
                dtype=np.int64,
            ),
        )

        log(
            f"  seed={seed}: "
            f"U={metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={metrics['wind_direction_RMSE_deg']:.2f} deg | "
            f"Vship={metrics['vessel_vector_RMSE_mps']:.4f} | "
            f"AW={metrics['AW_vector_RMSE_mps']:.4f}"
        )

        del (
            model,
            ckpt,
            pred_raw,
        )

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    seed_df = pd.DataFrame(
        seed_rows
    )

    seed_df.to_csv(
        output_dir
        / "tropical_atlantic_seed_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    proposed_summary = summarize_seed_results(
        seed_df
    )

    method_summary = make_method_summary(
        persistence_metrics,
        ridge_metrics,
        proposed_summary,
    )

    method_summary.to_csv(
        output_dir
        / "tropical_atlantic_method_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    published_comparison = create_published_comparison(
        method_summary
    )

    published_comparison.to_csv(
        output_dir
        / "published_BP_STGNN_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    proposed_row = method_summary.loc[
        method_summary["method"]
        == "PhysicsCompactNieDataV2"
    ].iloc[0]

    persistence_row = method_summary.loc[
        method_summary["method"]
        == "Persistence"
    ].iloc[0]

    ridge_row = method_summary.loc[
        method_summary["method"]
        == "Ridge"
    ].iloc[0]

    skill_summary = {
        "wind_vector_skill_vs_Persistence_percent": skill_percent(
            proposed_row[
                "wind_vector_RMSE_mps_mean"
            ],
            persistence_row[
                "wind_vector_RMSE_mps_mean"
            ],
        ),
        "AW_vector_skill_vs_Persistence_percent": skill_percent(
            proposed_row[
                "AW_vector_RMSE_mps_mean"
            ],
            persistence_row[
                "AW_vector_RMSE_mps_mean"
            ],
        ),
        "vessel_vector_skill_vs_Persistence_percent": skill_percent(
            proposed_row[
                "vessel_vector_RMSE_mps_mean"
            ],
            persistence_row[
                "vessel_vector_RMSE_mps_mean"
            ],
        ),
        "wind_vector_skill_vs_Ridge_percent": skill_percent(
            proposed_row[
                "wind_vector_RMSE_mps_mean"
            ],
            ridge_row[
                "wind_vector_RMSE_mps_mean"
            ],
        ),
        "AW_vector_skill_vs_Ridge_percent": skill_percent(
            proposed_row[
                "AW_vector_RMSE_mps_mean"
            ],
            ridge_row[
                "AW_vector_RMSE_mps_mean"
            ],
        ),
        "vessel_vector_skill_vs_Ridge_percent": skill_percent(
            proposed_row[
                "vessel_vector_RMSE_mps_mean"
            ],
            ridge_row[
                "vessel_vector_RMSE_mps_mean"
            ],
        ),
    }

    published_reference_delta = {
        "U_RMSE_difference_ours_minus_published_BPSTGNN_mps": float(
            proposed_row[
                "wind_U_RMSE_mps_mean"
            ]
            - NIE_BP_STGNN_REFERENCE[
                "wind_U_RMSE_mps"
            ]
        ),
        "V_RMSE_difference_ours_minus_published_BPSTGNN_mps": float(
            proposed_row[
                "wind_V_RMSE_mps_mean"
            ]
            - NIE_BP_STGNN_REFERENCE[
                "wind_V_RMSE_mps"
            ]
        ),
        "WS_RMSE_difference_ours_minus_published_BPSTGNN_mps": float(
            proposed_row[
                "wind_speed_RMSE_mps_mean"
            ]
            - NIE_BP_STGNN_REFERENCE[
                "wind_speed_RMSE_mps"
            ]
        ),
        "WD_RMSE_difference_ours_minus_published_BPSTGNN_deg": float(
            proposed_row[
                "wind_direction_RMSE_deg_mean"
            ]
            - NIE_BP_STGNN_REFERENCE[
                "wind_direction_RMSE_deg"
            ]
        ),
        "IMPORTANT": (
            "These are descriptive same-mission differences only. "
            "Do not report them as exact percentage improvement over BP-STGNN "
            "because preprocessing/sample protocols are not identical."
        ),
    }

    final_results = {
        "stage": "12D",
        "script_version": SCRIPT_VERSION,
        "model": "Physics-Compact-NieData v2.0",
        "anchor": "Persistence",
        "test_mission": TEST_MISSION,
        "test_samples": int(
            len(
                test["X"]
            )
        ),
        "final_seed_count": int(
            len(
                seed_df
            )
        ),
        "epoch_plan": epoch_plan.to_dict(
            orient="records"
        ),
        "persistence_metrics": persistence_metrics,
        "ridge_metrics": ridge_metrics,
        "proposed_seed_mean_std": proposed_summary,
        "skill_summary": skill_summary,
        "published_BP_STGNN_reference": NIE_BP_STGNN_REFERENCE,
        "published_reference_descriptive_differences": published_reference_delta,
        "comparison_scope": (
            "Same Tropical Atlantic raw mission and 10-min time scale as the "
            "published BP-STGNN study, but paper-informed rather than exact "
            "sample-for-sample reproduction because several original "
            "preprocessing/modeling details are under-specified."
        ),
        "test_access_record": access_record,
    }

    save_json(
        output_dir
        / "final_results.json",
        final_results,
    )

    with (
        output_dir
        / "FINAL_REPORT.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "12D FINAL Physics-Compact-NieData v2.0 "
            "Tropical Atlantic Evaluation\n"
        )
        f.write(
            "="
            * 126
            + "\n\n"
        )

        f.write(
            "SCIENTIFIC FIREWALL\n"
        )
        f.write(
            "-"
            * 126
            + "\n"
        )
        f.write(
            "All five final checkpoints were trained on "
            "Antarctic + Atlantic + West Coast before Tropical Atlantic "
            "was opened.\n"
        )
        f.write(
            "Final training epochs were derived only from Stage-12C2 "
            "development-only LOMO best epochs.\n"
        )
        f.write(
            "Tropical Atlantic was not used for epoch selection, scaling, "
            "Ridge fitting, model training, or hyperparameter tuning.\n\n"
        )

        f.write(
            "FINAL EPOCH PLAN\n"
        )
        f.write(
            "-"
            * 126
            + "\n"
        )
        f.write(
            epoch_plan.to_string(
                index=False
            )
        )

        f.write(
            "\n\nTROPICAL ATLANTIC METHOD SUMMARY\n"
        )
        f.write(
            "-"
            * 126
            + "\n"
        )
        f.write(
            method_summary.to_string(
                index=False
            )
        )

        f.write(
            "\n\nPUBLISHED BP-STGNN REFERENCE COMPARISON\n"
        )
        f.write(
            "-"
            * 126
            + "\n"
        )
        f.write(
            published_comparison.to_string(
                index=False
            )
        )

        f.write(
            "\n\nSKILL SUMMARY\n"
        )
        f.write(
            "-"
            * 126
            + "\n"
        )
        for key, value in skill_summary.items():
            f.write(
                f"{key}: {value:+.4f}%\n"
            )

        f.write(
            "\nCOMPARISON CAVEAT\n"
        )
        f.write(
            "-"
            * 126
            + "\n"
        )
        f.write(
            final_results[
                "comparison_scope"
            ]
            + "\n"
        )

    log("")
    log("=" * 126)
    log("12D FINAL TROPICAL ATLANTIC RESULTS")
    log("=" * 126)

    log(
        "Persistence: "
        f"U={persistence_metrics['wind_U_RMSE_mps']:.4f} | "
        f"V={persistence_metrics['wind_V_RMSE_mps']:.4f} | "
        f"W={persistence_metrics['wind_vector_RMSE_mps']:.4f} | "
        f"WS={persistence_metrics['wind_speed_RMSE_mps']:.4f} | "
        f"WD={persistence_metrics['wind_direction_RMSE_deg']:.2f} deg | "
        f"Vship={persistence_metrics['vessel_vector_RMSE_mps']:.4f} | "
        f"AW={persistence_metrics['AW_vector_RMSE_mps']:.4f}"
    )

    log(
        "Ridge      : "
        f"U={ridge_metrics['wind_U_RMSE_mps']:.4f} | "
        f"V={ridge_metrics['wind_V_RMSE_mps']:.4f} | "
        f"W={ridge_metrics['wind_vector_RMSE_mps']:.4f} | "
        f"WS={ridge_metrics['wind_speed_RMSE_mps']:.4f} | "
        f"WD={ridge_metrics['wind_direction_RMSE_deg']:.2f} deg | "
        f"Vship={ridge_metrics['vessel_vector_RMSE_mps']:.4f} | "
        f"AW={ridge_metrics['AW_vector_RMSE_mps']:.4f}"
    )

    log(
        "PhysicsCompactNieDataV2: "
        f"U={proposed_row['wind_U_RMSE_mps_mean']:.4f}"
        f"+/-{proposed_row['wind_U_RMSE_mps_std']:.4f} | "
        f"V={proposed_row['wind_V_RMSE_mps_mean']:.4f}"
        f"+/-{proposed_row['wind_V_RMSE_mps_std']:.4f} | "
        f"W={proposed_row['wind_vector_RMSE_mps_mean']:.4f}"
        f"+/-{proposed_row['wind_vector_RMSE_mps_std']:.4f} | "
        f"WS={proposed_row['wind_speed_RMSE_mps_mean']:.4f}"
        f"+/-{proposed_row['wind_speed_RMSE_mps_std']:.4f} | "
        f"WD={proposed_row['wind_direction_RMSE_deg_mean']:.2f}"
        f"+/-{proposed_row['wind_direction_RMSE_deg_std']:.2f} deg | "
        f"Vship={proposed_row['vessel_vector_RMSE_mps_mean']:.4f}"
        f"+/-{proposed_row['vessel_vector_RMSE_mps_std']:.4f} | "
        f"AW={proposed_row['AW_vector_RMSE_mps_mean']:.4f}"
        f"+/-{proposed_row['AW_vector_RMSE_mps_std']:.4f}"
    )

    log("")
    log(
        "PhysicsCompactNieDataV2 skill vs Persistence: "
        f"wind={skill_summary['wind_vector_skill_vs_Persistence_percent']:+.2f}% | "
        f"vessel={skill_summary['vessel_vector_skill_vs_Persistence_percent']:+.2f}% | "
        f"AW={skill_summary['AW_vector_skill_vs_Persistence_percent']:+.2f}%"
    )

    log(
        "PhysicsCompactNieDataV2 skill vs Ridge      : "
        f"wind={skill_summary['wind_vector_skill_vs_Ridge_percent']:+.2f}% | "
        f"vessel={skill_summary['vessel_vector_skill_vs_Ridge_percent']:+.2f}% | "
        f"AW={skill_summary['AW_vector_skill_vs_Ridge_percent']:+.2f}%"
    )

    log("")
    log(
        "Published BP-STGNN reference: "
        f"U={NIE_BP_STGNN_REFERENCE['wind_U_RMSE_mps']:.5f} | "
        f"V={NIE_BP_STGNN_REFERENCE['wind_V_RMSE_mps']:.5f} | "
        f"WS={NIE_BP_STGNN_REFERENCE['wind_speed_RMSE_mps']:.5f} | "
        f"WD={NIE_BP_STGNN_REFERENCE['wind_direction_RMSE_deg']:.3f} deg"
    )

    log(
        "[CAUTION] Published BP-STGNN values are a same-mission literature "
        "reference, not an exact sample-for-sample reproduction. "
        "Do not compute or claim exact cross-paper percentage improvement."
    )

    log(
        f"[DONE] 12D final outputs: {output_dir}"
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
        print(
            "\nPlease send the complete traceback and terminal output.",
            flush=True,
        )
        sys.exit(
            1
        )
