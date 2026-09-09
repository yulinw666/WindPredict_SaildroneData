# -*- coding: utf-8 -*-
r"""
14A_train_RidgeAnchored_ComponentSafeResidual_FinalTropical.py

Stage 14A
=========
Ridge-Anchored Component-Safe Residual Network

This is the formal two-stage forecasting architecture selected after
Stages 12Z/12Z-O/13A/13B/13C.

CORE IDEA
---------
ROUND 1 — Ridge baseline
    X_base = [U,V,T,RH] over six 10-min point samples

    Ridge(X_base) -> [U_R, V_R]

ROUND 2 — residual neural network
    Ridge is FROZEN.

    The residual network learns only:

        r_U = U_true - U_R
        r_V = V_true - V_R

    Final prediction:

        U_final = U_R + alpha_U * rhat_U
        V_final = V_R + alpha_V * rhat_V

The neural network NEVER replaces Ridge.

COMPONENT-SAFE PRINCIPLE
------------------------
U and V corrections are independently accepted or rejected.

For component j in {U,V}, development-only evidence must satisfy:

    1) residual correction wins on >= 2 of 3 LOMO missions
    2) residual correlation is positive on >= 2 of 3 missions
    3) pooled development residual correlation > 0
    4) pooled development RMSE improvement >= 0.10%
    5) robust alpha > 0

Otherwise:

    alpha_j = 0

and that component EXACTLY falls back to Ridge.

This does NOT mathematically guarantee unseen Tropical performance.
It guarantees only that Ridge is contained in the model family and that
development selection can explicitly reject an unhelpful correction.

ROBUST ALPHA
------------
For each development holdout mission:

    alpha_j^(m) =
        clip(
            sum(e_j * rhat_j) / sum(rhat_j^2),
            0,
            0.75
        )

Also compute pooled development alpha:

    alpha_j^(pool)

The frozen final coefficient is deliberately conservative:

    alpha_j^(robust)
        = min(
            alpha_j^(pool),
            25th percentile of missionwise alpha_j
          )

If the component-safe acceptance rules fail:

    alpha_j^(final) = 0

This is intentionally safer than Stage 12Z's pooled-only alpha.

STRICT RESIDUAL TARGET
----------------------
NEVER use in-sample Ridge residuals to train the residual network.

Within each training set, use mission-preserving blocked 5-fold OOF Ridge:

    r_OOF = y - Ridge_OOF(X)

This makes the Stage-2 target closer to the residual encountered at inference.

TEMPORAL PROTOCOL
-----------------
Stage-12G point-sampled benchmark:

    t-50, t-40, t-30, t-20, t-10, t
        -> t+10 min

No 10-min arithmetic averaging.
No intermediate 1-min samples.
No pressure.
No SOG/COG.
No wing angle.
No absolute position.

The proposed model intentionally uses the reduced raw information budget:

    U, V, T, RH

because Stage 13A found that adding the BP-owned Motion/P/dP/etc. variables
did not provide stable Tropical improvement.

RESIDUAL FEATURES
-----------------
Per time step, derived only from U,V,T,RH:

    U
    V
    T
    RH
    dU
    dV
    ddU
    ddV
    WS
    dWS

Summary features:

    latest_U
    latest_V
    latest_dU
    latest_dV
    std_U
    std_V
    std_WS
    slope_U
    slope_V
    slope_WS

Ridge anchor context:

    U_R
    V_R
    WS_R

Residual network:
    2-layer GRU, hidden=48
    last hidden + summaries + Ridge anchor
    shared MLP  -> 32
    separate U/V heads

The final residual output layers are ZERO INITIALIZED.

Therefore at epoch 0:

    rhat_U = 0
    rhat_V = 0

and:

    final prediction = Ridge

TRAINING LOSS
-------------
The loss is intentionally simple:

    L =
        0.5 * MSE(r_U_norm, rhat_U_norm)
      + 0.5 * MSE(r_V_norm, rhat_V_norm)

No WS loss.
No WD loss.
No direction loss.
No gate loss.

WS and WD are always derived from the final U/V vector.

DEVELOPMENT
-----------
Missions:
    Antarctic
    Atlantic
    West Coast

Outer LOMO:
    hold out each mission once.

Seeds:
    500043
    501052
    502061

For each fold/seed:
    - Stage 1: fit Ridge on the other two missions
    - build blocked-OOF Ridge residual targets on the training missions
    - Stage 2: train residual NN only
    - epoch 0 (= Ridge) is retained as a valid checkpoint
    - evaluate validation correction using component-wise analytical alpha

Seed selection:
    lowest mean development score

        0.5 * U_RMSE / Ridge_U_RMSE
      + 0.5 * V_RMSE / Ridge_V_RMSE

Final epoch:
    median best epoch for selected seed.

Then pool selected-seed out-of-mission predictions and freeze:
    active_U
    active_V
    alpha_U
    alpha_V

Save:
    FROZEN_BEFORE_TROPICAL.json

Only after that:
    load Tropical_Atlantic_TEST.npz

FINAL TRAINING
--------------
1) Fit final Base4-Ridge on all three development missions.
2) Create all-development blocked-OOF Ridge residual targets.
3) Train the residual network for the frozen number of epochs.
4) Apply frozen component-safe alpha_U/V.
5) Evaluate Tropical once.

Tropical Atlantic has already been viewed elsewhere in the project, so this
is a benchmark evaluation rather than a globally pristine test.

PUBLISHED REFERENCE
-------------------
BP-STGNN:
    U  = 0.748640 m/s
    V  = 0.705060 m/s
    WS = 0.699400 m/s
    WD = 5.584 deg
    params = 242580

PRIMARY SUCCESS CRITERION
-------------------------
Stage 14A is NOT required to beat BP-STGNN on all four metrics.

The primary criterion is:

    preserve or improve the strong Base4-Ridge benchmark

with a compact model.

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\14A_train_RidgeAnchored_ComponentSafeResidual_FinalTropical.py" `
  --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" `
  --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\14A_RidgeAnchored_ComponentSafeResidual_v0_1"

Smoke:
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


SCRIPT_VERSION = "0.1.0-14A-RidgeAnchored-ComponentSafeResidual"

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
    / "14A_RidgeAnchored_ComponentSafeResidual_v0_1"
)

DEV_MISSIONS = [
    "Antarctic",
    "Atlantic",
    "West Coast",
]
FINAL_MISSION = "Tropical Atlantic"

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

BASE4_INDICES = [
    0,
    1,
    2,
    3,
]

LOOKBACK_STEPS = 6
STEP_MINUTES = 10
TEN_MIN_NS = STEP_MINUTES * 60 * 1_000_000_000

RIDGE_ALPHA = 1.0
INNER_OOF_FOLDS = 5

SEEDS = [
    500043,
    501052,
    502061,
]

EPS = 1e-12

BP_REFERENCE = {
    "U_RMSE_mps": 0.748640,
    "V_RMSE_mps": 0.705060,
    "WS_RMSE_mps": 0.699400,
    "WD_RMSE_deg": 5.584000,
    "params": 242580,
}


@dataclass(frozen=True)
class Freeze:
    gru_hidden: int = 48
    gru_layers: int = 2
    shared_hidden: int = 32
    head_hidden: int = 16
    dropout: float = 0.10

    learning_rate: float = 6e-4
    weight_decay: float = 2e-5
    batch_size: int = 256
    max_epochs: int = 220
    early_stop_patience: int = 28
    scheduler_patience: int = 8
    scheduler_factor: float = 0.5
    min_lr: float = 1e-5
    grad_clip_norm: float = 1.0

    residual_tanh_cap_std: float = 1.0
    alpha_max: float = 0.75

    minimum_mission_wins: int = 2
    minimum_positive_corr_missions: int = 2
    minimum_pooled_improvement_fraction: float = 0.001

    robust_alpha_quantile: float = 0.25

    use_amp: bool = True


FREEZE = Freeze()


# =============================================================================
# General utilities
# =============================================================================

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


def seed_everything(torch, seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def amp_context(torch, enabled, device):
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


# =============================================================================
# Dataset
# =============================================================================

def resolve_dataset_root(dataset_dir: Path):
    for root in [
        dataset_dir,
        dataset_dir / "dataset",
    ]:
        if (
            root
            / "track_J_joint_compatible"
        ).exists():
            return root

    raise FileNotFoundError(
        "track_J_joint_compatible not found."
    )


def mission_path(
    dataset_root: Path,
    mission: str,
):
    track_dir = (
        dataset_root
        / "track_J_joint_compatible"
    )

    safe = mission.replace(
        " ",
        "_",
    )

    if mission == FINAL_MISSION:
        candidates = [
            track_dir
            / f"{safe}_TEST.npz",
            track_dir
            / f"{safe}.npz",
        ]
    else:
        candidates = [
            track_dir
            / f"{safe}.npz",
        ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Missing mission {mission}. Tried: {candidates}"
    )


def load_mission(
    dataset_root: Path,
    mission: str,
):
    path = mission_path(
        dataset_root,
        mission,
    )

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
            for x in z[
                "feature_names"
            ].tolist()
        ]

    if names != FEATURE_NAMES:
        raise RuntimeError(
            f"{mission}: unexpected feature names:\n{names}"
        )

    if (
        X.ndim != 3
        or X.shape[1] != LOOKBACK_STEPS
        or X.shape[2] != len(
            FEATURE_NAMES
        )
    ):
        raise RuntimeError(
            f"{mission}: unexpected X shape {X.shape}"
        )

    if y.ndim == 3:
        y = y[:, 0, :]

    if (
        y.ndim != 2
        or y.shape[1] != 2
    ):
        raise RuntimeError(
            f"{mission}: unexpected y shape {y.shape}"
        )

    if not np.all(
        target
        - context
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
            f"{mission}: nonfinite values."
        )

    return {
        "X_full9": X,
        "y_uv": y,
        "context": context,
        "target": target,
        "mission": mission,
        "path": path,
    }


def concat_missions(items):
    return {
        "X_full9": np.concatenate(
            [
                item["X_full9"]
                for item in items
            ],
            axis=0,
        ),
        "y_uv": np.concatenate(
            [
                item["y_uv"]
                for item in items
            ],
            axis=0,
        ),
        "mission_labels": np.concatenate(
            [
                np.asarray(
                    [
                        item["mission"]
                    ]
                    * len(
                        item["y_uv"]
                    ),
                    dtype=object,
                )
                for item in items
            ],
            axis=0,
        ),
    }


def build_base4(X_full9):
    return np.asarray(
        X_full9[
            :,
            :,
            BASE4_INDICES
        ],
        dtype=np.float32,
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
                -uv[:, 0],
                -uv[:, 1],
            )
        )
        % 360.0
    )


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


def evaluate_wind(
    y_true,
    y_pred,
):
    yt = np.asarray(
        y_true,
        dtype=np.float64,
    )

    yp = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    err = yp - yt

    ws_t = np.linalg.norm(
        yt,
        axis=1,
    )

    ws_p = np.linalg.norm(
        yp,
        axis=1,
    )

    wd_err = circular_diff_deg(
        wd_from_uv(yp),
        wd_from_uv(yt),
    )

    return {
        "U_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    err[:, 0] ** 2
                )
            )
        ),
        "V_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    err[:, 1] ** 2
                )
            )
        ),
        "vector_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        err ** 2,
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
                    ) ** 2
                )
            )
        ),
        "WD_RMSE_deg": float(
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

    if (
        len(a) < 3
        or np.std(a) < EPS
        or np.std(b) < EPS
    ):
        return np.nan

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def rel_improve(old, new):
    return float(
        (
            float(old)
            - float(new)
        )
        / max(
            abs(float(old)),
            EPS,
        )
    )


# =============================================================================
# Ridge
# =============================================================================

def fit_xy_scaler(
    X,
    y,
):
    X64 = np.asarray(
        X,
        dtype=np.float64,
    )

    y64 = np.asarray(
        y,
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
        "x_mean": x_mean.astype(
            np.float32
        ),
        "x_std": x_std.astype(
            np.float32
        ),
        "y_mean": y_mean.astype(
            np.float32
        ),
        "y_std": y_std.astype(
            np.float32
        ),
    }


def transform_X_ridge(
    X,
    scaler,
):
    return (
        (
            np.asarray(
                X,
                dtype=np.float32,
            )
            - scaler[
                "x_mean"
            ][
                None,
                None,
                :
            ]
        )
        / scaler[
            "x_std"
        ][
            None,
            None,
            :
        ]
    ).astype(
        np.float32
    )


def transform_y_ridge(
    y,
    scaler,
):
    return (
        (
            np.asarray(
                y,
                dtype=np.float32,
            )
            - scaler[
                "y_mean"
            ][
                None,
                :
            ]
        )
        / scaler[
            "y_std"
        ][
            None,
            :
        ]
    ).astype(
        np.float32
    )


def inverse_y_ridge(
    yz,
    scaler,
):
    return (
        np.asarray(
            yz,
            dtype=np.float32,
        )
        * scaler[
            "y_std"
        ][
            None,
            :
        ]
        + scaler[
            "y_mean"
        ][
            None,
            :
        ]
    ).astype(
        np.float32
    )


def fit_ridge(
    X,
    y,
):
    scaler = fit_xy_scaler(
        X,
        y,
    )

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        transform_X_ridge(
            X,
            scaler,
        ).reshape(
            len(X),
            -1,
        ),
        transform_y_ridge(
            y,
            scaler,
        ),
    )

    return (
        model,
        scaler,
    )


def ridge_predict(
    model,
    scaler,
    X,
):
    pred_z = np.asarray(
        model.predict(
            transform_X_ridge(
                X,
                scaler,
            ).reshape(
                len(X),
                -1,
            )
        ),
        dtype=np.float32,
    )

    return inverse_y_ridge(
        pred_z,
        scaler,
    )


def ridge_parameter_count(model):
    return int(
        np.asarray(
            model.coef_
        ).size
        + np.asarray(
            model.intercept_
        ).size
    )


def blocked_fold_ids(
    mission_labels,
):
    labels = np.asarray(
        mission_labels,
        dtype=object,
    )

    fold_ids = np.full(
        len(labels),
        -1,
        dtype=np.int64,
    )

    for mission in np.unique(
        labels
    ):
        idx = np.flatnonzero(
            labels == mission
        )

        chunks = np.array_split(
            idx,
            INNER_OOF_FOLDS,
        )

        for k, chunk in enumerate(
            chunks
        ):
            fold_ids[
                chunk
            ] = k

    if np.any(
        fold_ids < 0
    ):
        raise RuntimeError(
            "Blocked fold assignment failed."
        )

    return fold_ids


def ridge_oof_predict(
    X,
    y,
    mission_labels,
):
    folds = blocked_fold_ids(
        mission_labels
    )

    pred = np.full_like(
        y,
        np.nan,
        dtype=np.float32,
    )

    for k in range(
        INNER_OOF_FOLDS
    ):
        va = folds == k
        tr = ~va

        model, scaler = fit_ridge(
            X[tr],
            y[tr],
        )

        pred[va] = ridge_predict(
            model,
            scaler,
            X[va],
        )

    if not np.isfinite(
        pred
    ).all():
        raise RuntimeError(
            "OOF Ridge predictions contain nonfinite values."
        )

    return pred


# =============================================================================
# Residual feature engineering
# =============================================================================

def linear_slope(arr):
    arr = np.asarray(
        arr,
        dtype=np.float64,
    )

    n = arr.shape[1]

    t = np.arange(
        n,
        dtype=np.float64,
    )

    tc = t - t.mean()

    denom = float(
        np.dot(tc, tc)
    )

    centered = (
        arr
        - arr.mean(
            axis=1,
            keepdims=True,
        )
    )

    return (
        centered
        @ tc
        / max(
            denom,
            EPS,
        )
    )


def build_residual_features(
    X_full9,
    ridge_anchor,
):
    X = build_base4(
        X_full9
    ).astype(
        np.float64
    )

    U = X[:, :, 0]
    V = X[:, :, 1]
    T = X[:, :, 2]
    RH = X[:, :, 3]

    dU = np.zeros_like(U)
    dV = np.zeros_like(V)

    dU[:, 1:] = (
        U[:, 1:]
        - U[:, :-1]
    )

    dV[:, 1:] = (
        V[:, 1:]
        - V[:, :-1]
    )

    ddU = np.zeros_like(U)
    ddV = np.zeros_like(V)

    ddU[:, 2:] = (
        dU[:, 2:]
        - dU[:, 1:-1]
    )

    ddV[:, 2:] = (
        dV[:, 2:]
        - dV[:, 1:-1]
    )

    WS = np.sqrt(
        U ** 2
        + V ** 2
    )

    dWS = np.zeros_like(
        WS
    )

    dWS[:, 1:] = (
        WS[:, 1:]
        - WS[:, :-1]
    )

    seq = np.stack(
        [
            U,
            V,
            T,
            RH,
            dU,
            dV,
            ddU,
            ddV,
            WS,
            dWS,
        ],
        axis=-1,
    ).astype(
        np.float32
    )

    summary = np.stack(
        [
            U[:, -1],
            V[:, -1],
            dU[:, -1],
            dV[:, -1],
            np.std(
                U,
                axis=1,
            ),
            np.std(
                V,
                axis=1,
            ),
            np.std(
                WS,
                axis=1,
            ),
            linear_slope(U),
            linear_slope(V),
            linear_slope(WS),
        ],
        axis=-1,
    ).astype(
        np.float32
    )

    anchor = np.asarray(
        ridge_anchor,
        dtype=np.float32,
    )

    anchor_context = np.concatenate(
        [
            anchor,
            np.linalg.norm(
                anchor,
                axis=1,
                keepdims=True,
            ),
        ],
        axis=1,
    ).astype(
        np.float32
    )

    if not (
        np.isfinite(seq).all()
        and np.isfinite(summary).all()
        and np.isfinite(anchor_context).all()
    ):
        raise RuntimeError(
            "Residual features contain nonfinite values."
        )

    return {
        "seq": seq,
        "summary": summary,
        "anchor": anchor_context,
    }


def fit_residual_scalers(
    features,
    residual_target,
):
    seq = np.asarray(
        features["seq"],
        dtype=np.float64,
    )

    summary = np.asarray(
        features["summary"],
        dtype=np.float64,
    )

    anchor = np.asarray(
        features["anchor"],
        dtype=np.float64,
    )

    residual = np.asarray(
        residual_target,
        dtype=np.float64,
    )

    seq_mean = np.mean(
        seq,
        axis=(0, 1),
    )

    seq_std = np.std(
        seq,
        axis=(0, 1),
        ddof=0,
    )

    seq_std = np.where(
        seq_std < 1e-8,
        1.0,
        seq_std,
    )

    summary_mean = np.mean(
        summary,
        axis=0,
    )

    summary_std = np.std(
        summary,
        axis=0,
        ddof=0,
    )

    summary_std = np.where(
        summary_std < 1e-8,
        1.0,
        summary_std,
    )

    anchor_mean = np.mean(
        anchor,
        axis=0,
    )

    anchor_std = np.std(
        anchor,
        axis=0,
        ddof=0,
    )

    anchor_std = np.where(
        anchor_std < 1e-8,
        1.0,
        anchor_std,
    )

    residual_std = np.std(
        residual,
        axis=0,
        ddof=0,
    )

    residual_std = np.where(
        residual_std < 1e-6,
        1.0,
        residual_std,
    )

    return {
        "seq_mean": seq_mean.astype(
            np.float32
        ),
        "seq_std": seq_std.astype(
            np.float32
        ),
        "summary_mean": summary_mean.astype(
            np.float32
        ),
        "summary_std": summary_std.astype(
            np.float32
        ),
        "anchor_mean": anchor_mean.astype(
            np.float32
        ),
        "anchor_std": anchor_std.astype(
            np.float32
        ),
        "residual_std": residual_std.astype(
            np.float32
        ),
    }


def transform_residual_features(
    features,
    scalers,
):
    return {
        "seq": (
            (
                features[
                    "seq"
                ]
                - scalers[
                    "seq_mean"
                ][
                    None,
                    None,
                    :
                ]
            )
            / scalers[
                "seq_std"
            ][
                None,
                None,
                :
            ]
        ).astype(
            np.float32
        ),
        "summary": (
            (
                features[
                    "summary"
                ]
                - scalers[
                    "summary_mean"
                ][
                    None,
                    :
                ]
            )
            / scalers[
                "summary_std"
            ][
                None,
                :
            ]
        ).astype(
            np.float32
        ),
        "anchor": (
            (
                features[
                    "anchor"
                ]
                - scalers[
                    "anchor_mean"
                ][
                    None,
                    :
                ]
            )
            / scalers[
                "anchor_std"
            ][
                None,
                :
            ]
        ).astype(
            np.float32
        ),
    }


# =============================================================================
# Residual model
# =============================================================================

def build_model_class(
    torch,
    nn,
):
    class ComponentSafeResidualGRU(
        nn.Module
    ):
        def __init__(self):
            super().__init__()

            self.gru = nn.GRU(
                input_size=10,
                hidden_size=(
                    FREEZE.gru_hidden
                ),
                num_layers=(
                    FREEZE.gru_layers
                ),
                batch_first=True,
                dropout=(
                    FREEZE.dropout
                    if FREEZE.gru_layers
                    > 1
                    else 0.0
                ),
            )

            self.shared = nn.Sequential(
                nn.Linear(
                    FREEZE.gru_hidden
                    + 10
                    + 3,
                    FREEZE.shared_hidden,
                ),
                nn.GELU(),
                nn.Dropout(
                    FREEZE.dropout
                ),
            )

            self.u_hidden = nn.Sequential(
                nn.Linear(
                    FREEZE.shared_hidden,
                    FREEZE.head_hidden,
                ),
                nn.GELU(),
            )

            self.v_hidden = nn.Sequential(
                nn.Linear(
                    FREEZE.shared_hidden,
                    FREEZE.head_hidden,
                ),
                nn.GELU(),
            )

            self.u_out = nn.Linear(
                FREEZE.head_hidden,
                1,
            )

            self.v_out = nn.Linear(
                FREEZE.head_hidden,
                1,
            )

            # Critical: epoch 0 = zero residual = Ridge.
            nn.init.zeros_(
                self.u_out.weight
            )
            nn.init.zeros_(
                self.u_out.bias
            )

            nn.init.zeros_(
                self.v_out.weight
            )
            nn.init.zeros_(
                self.v_out.bias
            )

        def forward(
            self,
            seq,
            summary,
            anchor,
        ):
            z, _ = self.gru(
                seq
            )

            last = z[
                :,
                -1,
                :
            ]

            h = self.shared(
                torch.cat(
                    [
                        last,
                        summary,
                        anchor,
                    ],
                    dim=1,
                )
            )

            zu = self.u_out(
                self.u_hidden(
                    h
                )
            ).squeeze(
                -1
            )

            zv = self.v_out(
                self.v_hidden(
                    h
                )
            ).squeeze(
                -1
            )

            z = torch.stack(
                [
                    zu,
                    zv,
                ],
                dim=1,
            )

            return torch.tanh(
                z
            ) * float(
                FREEZE.residual_tanh_cap_std
            )

    return ComponentSafeResidualGRU


# =============================================================================
# Residual training helpers
# =============================================================================

def normalized_residual_target(
    residual_target,
    residual_std,
):
    return (
        np.asarray(
            residual_target,
            dtype=np.float32,
        )
        / np.asarray(
            residual_std,
            dtype=np.float32,
        )[
            None,
            :
        ]
    ).astype(
        np.float32
    )


def predict_raw_residual(
    *,
    torch,
    model,
    transformed,
    residual_std,
    device,
    amp_enabled,
):
    model.eval()

    outputs = []

    with torch.no_grad():
        for start in range(
            0,
            len(
                transformed[
                    "seq"
                ]
            ),
            FREEZE.batch_size,
        ):
            stop = min(
                start
                + FREEZE.batch_size,
                len(
                    transformed[
                        "seq"
                    ]
                ),
            )

            seq_t = torch.from_numpy(
                transformed[
                    "seq"
                ][
                    start:stop
                ]
            ).to(
                device,
                non_blocking=True,
            )

            summary_t = torch.from_numpy(
                transformed[
                    "summary"
                ][
                    start:stop
                ]
            ).to(
                device,
                non_blocking=True,
            )

            anchor_t = torch.from_numpy(
                transformed[
                    "anchor"
                ][
                    start:stop
                ]
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
                    seq_t,
                    summary_t,
                    anchor_t,
                )

            outputs.append(
                pred_norm.detach()
                .float()
                .cpu()
                .numpy()
            )

    pred_norm = np.concatenate(
        outputs,
        axis=0,
    )

    return (
        pred_norm
        * np.asarray(
            residual_std,
            dtype=np.float32,
        )[
            None,
            :
        ]
    ).astype(
        np.float32
    )


def analytic_alpha_1d(
    true_residual,
    predicted_residual,
):
    e = np.asarray(
        true_residual,
        dtype=np.float64,
    ).reshape(-1)

    r = np.asarray(
        predicted_residual,
        dtype=np.float64,
    ).reshape(-1)

    denom = float(
        np.dot(
            r,
            r,
        )
    )

    if denom <= EPS:
        return 0.0

    alpha = float(
        np.dot(
            e,
            r,
        )
        / denom
    )

    return float(
        np.clip(
            alpha,
            0.0,
            FREEZE.alpha_max,
        )
    )


def analytic_alphas(
    y_true,
    ridge_pred,
    raw_residual,
):
    true_res = (
        np.asarray(
            y_true,
            dtype=np.float64,
        )
        - np.asarray(
            ridge_pred,
            dtype=np.float64,
        )
    )

    raw = np.asarray(
        raw_residual,
        dtype=np.float64,
    )

    return np.asarray(
        [
            analytic_alpha_1d(
                true_res[:, 0],
                raw[:, 0],
            ),
            analytic_alpha_1d(
                true_res[:, 1],
                raw[:, 1],
            ),
        ],
        dtype=np.float32,
    )


def apply_alphas(
    ridge_pred,
    raw_residual,
    alphas,
):
    return (
        np.asarray(
            ridge_pred,
            dtype=np.float32,
        )
        + np.asarray(
            raw_residual,
            dtype=np.float32,
        )
        * np.asarray(
            alphas,
            dtype=np.float32,
        )[
            None,
            :
        ]
    ).astype(
        np.float32
    )


def fold_score(
    y_true,
    ridge_pred,
    raw_residual,
):
    alphas = analytic_alphas(
        y_true,
        ridge_pred,
        raw_residual,
    )

    final = apply_alphas(
        ridge_pred,
        raw_residual,
        alphas,
    )

    base = evaluate_wind(
        y_true,
        ridge_pred,
    )

    m = evaluate_wind(
        y_true,
        final,
    )

    score = (
        0.5
        * (
            m[
                "U_RMSE_mps"
            ]
            / max(
                base[
                    "U_RMSE_mps"
                ],
                EPS,
            )
        )
        + 0.5
        * (
            m[
                "V_RMSE_mps"
            ]
            / max(
                base[
                    "V_RMSE_mps"
                ],
                EPS,
            )
        )
    )

    return (
        float(
            score
        ),
        alphas,
        m,
        base,
    )


# =============================================================================
# Train one LOMO fold
# =============================================================================

def train_one_fold(
    *,
    torch,
    nn,
    train_data,
    val_data,
    held_out,
    seed,
    output_dir,
    device,
    debug_fast,
):
    # ---------------------------------------------------------
    # ROUND 1: Ridge
    # ---------------------------------------------------------
    Xtr_base = build_base4(
        train_data[
            "X_full9"
        ]
    )

    Xva_base = build_base4(
        val_data[
            "X_full9"
        ]
    )

    ridge_model, ridge_scaler = fit_ridge(
        Xtr_base,
        train_data[
            "y_uv"
        ],
    )

    ridge_val = ridge_predict(
        ridge_model,
        ridge_scaler,
        Xva_base,
    )

    # OOF Ridge targets for ROUND 2.
    ridge_oof = ridge_oof_predict(
        Xtr_base,
        train_data[
            "y_uv"
        ],
        train_data[
            "mission_labels"
        ],
    )

    residual_target = (
        train_data[
            "y_uv"
        ]
        - ridge_oof
    ).astype(
        np.float32
    )

    # ---------------------------------------------------------
    # ROUND 2: residual NN only
    # ---------------------------------------------------------
    features_train = build_residual_features(
        train_data[
            "X_full9"
        ],
        ridge_oof,
    )

    features_val = build_residual_features(
        val_data[
            "X_full9"
        ],
        ridge_val,
    )

    scalers = fit_residual_scalers(
        features_train,
        residual_target,
    )

    transformed_train = transform_residual_features(
        features_train,
        scalers,
    )

    transformed_val = transform_residual_features(
        features_val,
        scalers,
    )

    target_norm = normalized_residual_target(
        residual_target,
        scalers[
            "residual_std"
        ],
    )

    seed_everything(
        torch,
        seed,
    )

    ModelClass = build_model_class(
        torch,
        nn,
    )

    model = ModelClass().to(
        device
    )

    n_params = count_parameters(
        model
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FREEZE.learning_rate,
        weight_decay=(
            FREEZE.weight_decay
        ),
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
                FREEZE.min_lr
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

    # epoch0 = Ridge fallback
    best_score = 1.0
    best_epoch = 0
    best_state = state_dict_cpu(
        model
    )

    zero_raw_val = np.zeros_like(
        ridge_val,
        dtype=np.float32,
    )

    (
        _,
        best_alphas,
        best_metrics,
        base_metrics,
    ) = fold_score(
        val_data[
            "y_uv"
        ],
        ridge_val,
        zero_raw_val,
    )

    best_raw_val = zero_raw_val

    patience = 0

    n_train = len(
        target_norm
    )

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

    history = []

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        model.train()

        order = np.arange(
            n_train
        )

        rng = np.random.default_rng(
            seed
            + epoch
            * 1009
        )

        rng.shuffle(
            order
        )

        running = 0.0
        seen = 0

        for start in range(
            0,
            n_train,
            FREEZE.batch_size,
        ):
            idx = order[
                start:
                start
                + FREEZE.batch_size
            ]

            seq_t = torch.from_numpy(
                transformed_train[
                    "seq"
                ][
                    idx
                ]
            ).to(
                device,
                non_blocking=True,
            )

            summary_t = torch.from_numpy(
                transformed_train[
                    "summary"
                ][
                    idx
                ]
            ).to(
                device,
                non_blocking=True,
            )

            anchor_t = torch.from_numpy(
                transformed_train[
                    "anchor"
                ][
                    idx
                ]
            ).to(
                device,
                non_blocking=True,
            )

            target_t = torch.from_numpy(
                target_norm[
                    idx
                ]
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
                    seq_t,
                    summary_t,
                    anchor_t,
                )

                loss_u = torch.mean(
                    (
                        pred_norm[:, 0]
                        - target_t[:, 0]
                    ) ** 2
                )

                loss_v = torch.mean(
                    (
                        pred_norm[:, 1]
                        - target_t[:, 1]
                    ) ** 2
                )

                loss = (
                    0.5
                    * loss_u
                    + 0.5
                    * loss_v
                )

            if not torch.isfinite(
                loss
            ):
                raise FloatingPointError(
                    "Nonfinite residual loss."
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
                idx
            )

            running += (
                float(
                    loss.detach()
                    .float()
                    .cpu()
                    .item()
                )
                * bn
            )

            seen += bn

        raw_val = predict_raw_residual(
            torch=torch,
            model=model,
            transformed=transformed_val,
            residual_std=scalers[
                "residual_std"
            ],
            device=device,
            amp_enabled=amp_enabled,
        )

        (
            score,
            alphas,
            metrics,
            _,
        ) = fold_score(
            val_data[
                "y_uv"
            ],
            ridge_val,
            raw_val,
        )

        scheduler.step(
            score
        )

        improved = (
            score
            < best_score
            - 1e-8
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

            best_alphas = alphas.copy()
            best_metrics = metrics.copy()
            best_raw_val = raw_val.copy()

            patience = 0
        else:
            patience += 1

        history.append(
            {
                "epoch": int(
                    epoch
                ),
                "train_loss": (
                    running
                    / max(
                        seen,
                        1,
                    )
                ),
                "selection_score": float(
                    score
                ),
                "alpha_U": float(
                    alphas[0]
                ),
                "alpha_V": float(
                    alphas[1]
                ),
                **{
                    f"final_{k}": v
                    for k, v
                    in metrics.items()
                },
            }
        )

        if (
            epoch <= 3
            or epoch % 10 == 0
            or improved
        ):
            log(
                f"      ep={epoch:03d} | "
                f"score={score:.6f} | "
                f"aU={alphas[0]:.3f} | "
                f"aV={alphas[1]:.3f} | "
                f"U={metrics['U_RMSE_mps']:.5f} | "
                f"V={metrics['V_RMSE_mps']:.5f}"
                + (
                    " *"
                    if improved
                    else ""
                )
            )

        if patience >= patience_limit:
            break

    model.load_state_dict(
        best_state,
        strict=True,
    )

    true_res_val = (
        val_data[
            "y_uv"
        ]
        - ridge_val
    )

    corr_u = corrcoef_safe(
        true_res_val[:, 0],
        best_raw_val[:, 0],
    )

    corr_v = corrcoef_safe(
        true_res_val[:, 1],
        best_raw_val[:, 1],
    )

    final_best = apply_alphas(
        ridge_val,
        best_raw_val,
        best_alphas,
    )

    # Save prediction for pooled development calibration.
    pred_dir = (
        output_dir
        / "development_predictions"
    )

    hist_dir = (
        output_dir
        / "histories"
    )

    pred_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    hist_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tag = (
        f"holdout_{held_out.replace(' ', '_')}"
        f"__seed{seed}"
    )

    prediction_path = (
        pred_dir
        / f"{tag}.npz"
    )

    np.savez_compressed(
        prediction_path,
        y_true=val_data[
            "y_uv"
        ].astype(
            np.float32
        ),
        ridge_pred=ridge_val.astype(
            np.float32
        ),
        raw_residual_pred=best_raw_val.astype(
            np.float32
        ),
        fold_alphas=best_alphas.astype(
            np.float32
        ),
        final_pred=final_best.astype(
            np.float32
        ),
    )

    pd.DataFrame(
        history
    ).to_csv(
        hist_dir
        / f"{tag}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    row = {
        "held_out_mission": held_out,
        "seed": int(
            seed
        ),
        "best_epoch": int(
            best_epoch
        ),
        "selection_score": float(
            best_score
        ),
        "parameter_count_nn": int(
            n_params
        ),
        "parameter_count_total": int(
            n_params
            + ridge_parameter_count(
                ridge_model
            )
        ),
        "alpha_U_fold": float(
            best_alphas[0]
        ),
        "alpha_V_fold": float(
            best_alphas[1]
        ),
        "residual_corr_U": float(
            corr_u
        ),
        "residual_corr_V": float(
            corr_v
        ),
        "U_Ridge": float(
            base_metrics[
                "U_RMSE_mps"
            ]
        ),
        "V_Ridge": float(
            base_metrics[
                "V_RMSE_mps"
            ]
        ),
        "U_Final": float(
            best_metrics[
                "U_RMSE_mps"
            ]
        ),
        "V_Final": float(
            best_metrics[
                "V_RMSE_mps"
            ]
        ),
        "U_improve_fraction": (
            rel_improve(
                base_metrics[
                    "U_RMSE_mps"
                ],
                best_metrics[
                    "U_RMSE_mps"
                ],
            )
        ),
        "V_improve_fraction": (
            rel_improve(
                base_metrics[
                    "V_RMSE_mps"
                ],
                best_metrics[
                    "V_RMSE_mps"
                ],
            )
        ),
        "prediction_path": str(
            prediction_path
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

    return row


# =============================================================================
# Development freeze
# =============================================================================

def summarize_seeds(runs):
    rows = []

    for seed in SEEDS:
        sub = runs.loc[
            runs[
                "seed"
            ]
            == seed
        ].copy()

        rows.append(
            {
                "seed": int(
                    seed
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
                "U_Final_mean": float(
                    sub[
                        "U_Final"
                    ].mean()
                ),
                "V_Final_mean": float(
                    sub[
                        "V_Final"
                    ].mean()
                ),
                "U_improve_mean": float(
                    sub[
                        "U_improve_fraction"
                    ].mean()
                ),
                "V_improve_mean": float(
                    sub[
                        "V_improve_fraction"
                    ].mean()
                ),
                "residual_corr_U_mean": float(
                    sub[
                        "residual_corr_U"
                    ].mean()
                ),
                "residual_corr_V_mean": float(
                    sub[
                        "residual_corr_V"
                    ].mean()
                ),
                "alpha_U_mean": float(
                    sub[
                        "alpha_U_fold"
                    ].mean()
                ),
                "alpha_V_mean": float(
                    sub[
                        "alpha_V_fold"
                    ].mean()
                ),
                "best_epoch_median": float(
                    np.median(
                        sub[
                            "best_epoch"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                ),
                "mission_U_wins": int(
                    (
                        sub[
                            "U_improve_fraction"
                        ]
                        > 0
                    ).sum()
                ),
                "mission_V_wins": int(
                    (
                        sub[
                            "V_improve_fraction"
                        ]
                        > 0
                    ).sum()
                ),
                "parameter_count_total": int(
                    round(
                        sub[
                            "parameter_count_total"
                        ].mean()
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "selection_score_mean",
            "V_Final_mean",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )


def pooled_component_freeze(
    selected_runs,
):
    y_all = []
    ridge_all = []
    raw_all = []

    mission_alphas = {
        "U": [],
        "V": [],
    }

    mission_corrs = {
        "U": [],
        "V": [],
    }

    mission_wins = {
        "U": 0,
        "V": 0,
    }

    mission_rows = []

    for _, row in selected_runs.iterrows():
        path = Path(
            str(
                row[
                    "prediction_path"
                ]
            )
        )

        with np.load(
            path,
            allow_pickle=False,
        ) as z:
            y = np.asarray(
                z[
                    "y_true"
                ],
                dtype=np.float32,
            )

            ridge = np.asarray(
                z[
                    "ridge_pred"
                ],
                dtype=np.float32,
            )

            raw = np.asarray(
                z[
                    "raw_residual_pred"
                ],
                dtype=np.float32,
            )

        alphas = analytic_alphas(
            y,
            ridge,
            raw,
        )

        final = apply_alphas(
            ridge,
            raw,
            alphas,
        )

        base_m = evaluate_wind(
            y,
            ridge,
        )

        final_m = evaluate_wind(
            y,
            final,
        )

        true_res = (
            y
            - ridge
        )

        corr_u = corrcoef_safe(
            true_res[:, 0],
            raw[:, 0],
        )

        corr_v = corrcoef_safe(
            true_res[:, 1],
            raw[:, 1],
        )

        imp_u = rel_improve(
            base_m[
                "U_RMSE_mps"
            ],
            final_m[
                "U_RMSE_mps"
            ],
        )

        imp_v = rel_improve(
            base_m[
                "V_RMSE_mps"
            ],
            final_m[
                "V_RMSE_mps"
            ],
        )

        mission_alphas[
            "U"
        ].append(
            float(
                alphas[0]
            )
        )

        mission_alphas[
            "V"
        ].append(
            float(
                alphas[1]
            )
        )

        mission_corrs[
            "U"
        ].append(
            float(
                corr_u
            )
        )

        mission_corrs[
            "V"
        ].append(
            float(
                corr_v
            )
        )

        mission_wins[
            "U"
        ] += int(
            imp_u > 0
        )

        mission_wins[
            "V"
        ] += int(
            imp_v > 0
        )

        mission_rows.append(
            {
                "held_out_mission": (
                    row[
                        "held_out_mission"
                    ]
                ),
                "alpha_U": float(
                    alphas[0]
                ),
                "alpha_V": float(
                    alphas[1]
                ),
                "corr_U": float(
                    corr_u
                ),
                "corr_V": float(
                    corr_v
                ),
                "U_improve_fraction": float(
                    imp_u
                ),
                "V_improve_fraction": float(
                    imp_v
                ),
            }
        )

        y_all.append(
            y
        )

        ridge_all.append(
            ridge
        )

        raw_all.append(
            raw
        )

    y_pool = np.concatenate(
        y_all,
        axis=0,
    )

    ridge_pool = np.concatenate(
        ridge_all,
        axis=0,
    )

    raw_pool = np.concatenate(
        raw_all,
        axis=0,
    )

    pooled_alpha = analytic_alphas(
        y_pool,
        ridge_pool,
        raw_pool,
    )

    true_res_pool = (
        y_pool
        - ridge_pool
    )

    pooled_corr = np.asarray(
        [
            corrcoef_safe(
                true_res_pool[:, 0],
                raw_pool[:, 0],
            ),
            corrcoef_safe(
                true_res_pool[:, 1],
                raw_pool[:, 1],
            ),
        ],
        dtype=np.float64,
    )

    component_info = {}

    frozen_alphas = np.zeros(
        2,
        dtype=np.float32,
    )

    active = np.zeros(
        2,
        dtype=bool,
    )

    for j, name in enumerate(
        [
            "U",
            "V",
        ]
    ):
        mission_a = np.asarray(
            mission_alphas[
                name
            ],
            dtype=np.float64,
        )

        robust_q = float(
            np.quantile(
                mission_a,
                FREEZE.robust_alpha_quantile,
            )
        )

        robust_alpha = float(
            min(
                float(
                    pooled_alpha[
                        j
                    ]
                ),
                robust_q,
            )
        )

        candidate_alpha = np.zeros(
            2,
            dtype=np.float32,
        )

        candidate_alpha[
            j
        ] = float(
            robust_alpha
        )

        candidate_final = apply_alphas(
            ridge_pool,
            raw_pool,
            candidate_alpha,
        )

        base_pool = evaluate_wind(
            y_pool,
            ridge_pool,
        )

        final_pool = evaluate_wind(
            y_pool,
            candidate_final,
        )

        metric_name = (
            "U_RMSE_mps"
            if name == "U"
            else "V_RMSE_mps"
        )

        pooled_improve = rel_improve(
            base_pool[
                metric_name
            ],
            final_pool[
                metric_name
            ],
        )

        positive_corr_missions = int(
            np.sum(
                np.asarray(
                    mission_corrs[
                        name
                    ],
                    dtype=np.float64,
                )
                > 0
            )
        )

        accepted = bool(
            mission_wins[
                name
            ]
            >= FREEZE.minimum_mission_wins
            and positive_corr_missions
            >= FREEZE.minimum_positive_corr_missions
            and pooled_corr[
                j
            ]
            > 0
            and pooled_improve
            >= FREEZE.minimum_pooled_improvement_fraction
            and robust_alpha
            > 0
        )

        if accepted:
            active[
                j
            ] = True

            frozen_alphas[
                j
            ] = float(
                robust_alpha
            )

        component_info[
            name
        ] = {
            "mission_wins": int(
                mission_wins[
                    name
                ]
            ),
            "positive_corr_missions": int(
                positive_corr_missions
            ),
            "mission_alphas": (
                mission_a.tolist()
            ),
            "pooled_alpha": float(
                pooled_alpha[
                    j
                ]
            ),
            "alpha_quantile25": float(
                robust_q
            ),
            "robust_alpha_before_acceptance": float(
                robust_alpha
            ),
            "pooled_corr": float(
                pooled_corr[
                    j
                ]
            ),
            "pooled_improvement_fraction_at_robust_alpha": float(
                pooled_improve
            ),
            "accepted": bool(
                accepted
            ),
            "frozen_alpha": float(
                frozen_alphas[
                    j
                ]
            ),
        }

    return {
        "active": active,
        "frozen_alphas": frozen_alphas,
        "component_info": component_info,
        "mission_table": pd.DataFrame(
            mission_rows
        ),
        "pooled_y": y_pool,
        "pooled_ridge": ridge_pool,
        "pooled_raw": raw_pool,
    }


# =============================================================================
# Final fixed-epoch training
# =============================================================================

def train_final_network(
    *,
    torch,
    nn,
    dev_all,
    final_seed,
    final_epochs,
    device,
):
    # ROUND 1 — final Ridge
    Xbase = build_base4(
        dev_all[
            "X_full9"
        ]
    )

    ridge_model, ridge_scaler = fit_ridge(
        Xbase,
        dev_all[
            "y_uv"
        ],
    )

    ridge_oof = ridge_oof_predict(
        Xbase,
        dev_all[
            "y_uv"
        ],
        dev_all[
            "mission_labels"
        ],
    )

    residual_target = (
        dev_all[
            "y_uv"
        ]
        - ridge_oof
    ).astype(
        np.float32
    )

    # ROUND 2 — final residual network.
    features = build_residual_features(
        dev_all[
            "X_full9"
        ],
        ridge_oof,
    )

    scalers = fit_residual_scalers(
        features,
        residual_target,
    )

    transformed = transform_residual_features(
        features,
        scalers,
    )

    target_norm = normalized_residual_target(
        residual_target,
        scalers[
            "residual_std"
        ],
    )

    seed_everything(
        torch,
        final_seed,
    )

    ModelClass = build_model_class(
        torch,
        nn,
    )

    model = ModelClass().to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FREEZE.learning_rate,
        weight_decay=(
            FREEZE.weight_decay
        ),
    )

    amp_enabled = bool(
        FREEZE.use_amp
        and device.type == "cuda"
    )

    grad_scaler = create_grad_scaler(
        torch,
        amp_enabled,
    )

    n = len(
        target_norm
    )

    for epoch in range(
        1,
        int(
            final_epochs
        )
        + 1,
    ):
        model.train()

        order = np.arange(
            n
        )

        rng = np.random.default_rng(
            final_seed
            + epoch
            * 1009
        )

        rng.shuffle(
            order
        )

        running = 0.0
        seen = 0

        for start in range(
            0,
            n,
            FREEZE.batch_size,
        ):
            idx = order[
                start:
                start
                + FREEZE.batch_size
            ]

            seq_t = torch.from_numpy(
                transformed[
                    "seq"
                ][
                    idx
                ]
            ).to(
                device,
                non_blocking=True,
            )

            summary_t = torch.from_numpy(
                transformed[
                    "summary"
                ][
                    idx
                ]
            ).to(
                device,
                non_blocking=True,
            )

            anchor_t = torch.from_numpy(
                transformed[
                    "anchor"
                ][
                    idx
                ]
            ).to(
                device,
                non_blocking=True,
            )

            target_t = torch.from_numpy(
                target_norm[
                    idx
                ]
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
                    seq_t,
                    summary_t,
                    anchor_t,
                )

                loss_u = torch.mean(
                    (
                        pred_norm[:, 0]
                        - target_t[:, 0]
                    ) ** 2
                )

                loss_v = torch.mean(
                    (
                        pred_norm[:, 1]
                        - target_t[:, 1]
                    ) ** 2
                )

                loss = (
                    0.5
                    * loss_u
                    + 0.5
                    * loss_v
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
                idx
            )

            running += (
                float(
                    loss.detach()
                    .float()
                    .cpu()
                    .item()
                )
                * bn
            )

            seen += bn

        if (
            epoch <= 3
            or epoch % 10 == 0
            or epoch == final_epochs
        ):
            log(
                f"  final epoch={epoch:03d}/{final_epochs:03d} | "
                f"loss={running/max(seen,1):.6f}"
            )

    return {
        "ridge_model": ridge_model,
        "ridge_scaler": ridge_scaler,
        "residual_model": model,
        "residual_scalers": scalers,
        "parameter_count_nn": count_parameters(
            model
        ),
        "parameter_count_ridge": ridge_parameter_count(
            ridge_model
        ),
    }


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

    parser.add_argument(
        "--force-cpu",
        action="store_true",
    )

    parser.add_argument(
        "--debug-fast",
        action="store_true",
    )

    args = parser.parse_args()

    dataset_root = resolve_dataset_root(
        args.dataset_dir
    )

    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch, nn = import_torch()

    configure_cuda(
        torch
    )

    device = torch.device(
        "cpu"
        if (
            args.force_cpu
            or not torch.cuda.is_available()
        )
        else "cuda"
    )

    log("=" * 140)
    log(
        "14A — RIDGE-ANCHORED COMPONENT-SAFE RESIDUAL NETWORK"
    )
    log("=" * 140)
    log(
        "ROUND 1 = Ridge"
    )
    log(
        "ROUND 2 = frozen-Ridge residual neural network"
    )
    log(
        f"dataset = {dataset_root}"
    )
    log(
        f"output  = {output_dir}"
    )
    log(
        f"device  = {device}"
    )

    if device.type == "cuda":
        log(
            f"GPU     = {torch.cuda.get_device_name(0)}"
        )

    # ---------------------------------------------------------
    # Development only.
    # ---------------------------------------------------------
    log("")
    log(
        "[STAGE A] Loading development missions only."
    )

    dev_missions = {
        mission: load_mission(
            dataset_root,
            mission,
        )
        for mission in DEV_MISSIONS
    }

    for mission in DEV_MISSIONS:
        log(
            f"  {mission:<12} N={len(dev_missions[mission]['y_uv']):,}"
        )

    log(
        "[FIREWALL] Tropical Atlantic NOT loaded."
    )

    active_holdouts = (
        [
            "Antarctic"
        ]
        if args.debug_fast
        else DEV_MISSIONS
    )

    active_seeds = (
        [
            SEEDS[0]
        ]
        if args.debug_fast
        else SEEDS
    )

    rows = []

    for held_out in active_holdouts:
        train_data = concat_missions(
            [
                dev_missions[
                    m
                ]
                for m in DEV_MISSIONS
                if m != held_out
            ]
        )

        val_data = dev_missions[
            held_out
        ]

        log("")
        log(
            "-" * 140
        )

        log(
            f"[LOMO HOLDOUT] {held_out} | "
            f"Ntrain={len(train_data['y_uv']):,} | "
            f"Nval={len(val_data['y_uv']):,}"
        )

        for seed in active_seeds:
            log(
                f"  [SEED {seed}]"
            )

            row = train_one_fold(
                torch=torch,
                nn=nn,
                train_data=train_data,
                val_data=val_data,
                held_out=held_out,
                seed=seed,
                output_dir=output_dir,
                device=device,
                debug_fast=(
                    args.debug_fast
                ),
            )

            rows.append(
                row
            )

            log(
                f"  [DONE] bestEp={row['best_epoch']} | "
                f"score={row['selection_score']:.6f} | "
                f"corrU/V={row['residual_corr_U']:+.4f}/"
                f"{row['residual_corr_V']:+.4f} | "
                f"U={row['U_Ridge']:.5f}->{row['U_Final']:.5f} | "
                f"V={row['V_Ridge']:.5f}->{row['V_Final']:.5f}"
            )

    runs = pd.DataFrame(
        rows
    )

    runs_path = (
        output_dir
        / "development_runs.csv"
    )

    runs.to_csv(
        runs_path,
        index=False,
        encoding="utf-8-sig",
    )

    if args.debug_fast:
        log("")
        log(
            "[DONE] --debug-fast completed. Tropical Atlantic was NOT loaded."
        )
        return 0

    # ---------------------------------------------------------
    # Seed selection.
    # ---------------------------------------------------------
    seed_summary = summarize_seeds(
        runs
    )

    seed_summary_path = (
        output_dir
        / "development_seed_summary.csv"
    )

    seed_summary.to_csv(
        seed_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log("=" * 140)
    log(
        "14A DEVELOPMENT SEED SUMMARY"
    )
    log("=" * 140)
    log(
        seed_summary.to_string(
            index=False
        )
    )

    selected_seed = int(
        seed_summary.iloc[
            0
        ][
            "seed"
        ]
    )

    selected_runs = runs.loc[
        runs[
            "seed"
        ]
        == selected_seed
    ].copy()

    final_epoch = int(
        max(
            0,
            round(
                float(
                    np.median(
                        selected_runs[
                            "best_epoch"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                )
            ),
        )
    )

    freeze_info = pooled_component_freeze(
        selected_runs
    )

    active = freeze_info[
        "active"
    ]

    frozen_alphas = freeze_info[
        "frozen_alphas"
    ]

    freeze_info[
        "mission_table"
    ].to_csv(
        output_dir
        / "development_component_evidence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log(
        "[COMPONENT-SAFE DEVELOPMENT FREEZE]"
    )

    for j, name in enumerate(
        [
            "U",
            "V",
        ]
    ):
        info = freeze_info[
            "component_info"
        ][
            name
        ]

        log(
            f"  {name}: active={bool(active[j])} | "
            f"alpha={frozen_alphas[j]:.6f} | "
            f"wins={info['mission_wins']}/3 | "
            f"+corr missions={info['positive_corr_missions']}/3 | "
            f"pooled corr={info['pooled_corr']:+.4f} | "
            f"pooled improve={100*info['pooled_improvement_fraction_at_robust_alpha']:+.3f}%"
        )

    frozen = {
        "stage": "14A",
        "script_version": (
            SCRIPT_VERSION
        ),
        "training_scheme": {
            "round_1": (
                "Base4 Ridge [U,V,T,RH]x6 -> U,V"
            ),
            "round_2": (
                "Ridge frozen; residual GRU learns OOF Ridge residual"
            ),
        },
        "selected_seed": (
            selected_seed
        ),
        "final_epoch": (
            final_epoch
        ),
        "active_U": bool(
            active[0]
        ),
        "active_V": bool(
            active[1]
        ),
        "alpha_U": float(
            frozen_alphas[0]
        ),
        "alpha_V": float(
            frozen_alphas[1]
        ),
        "component_info": (
            freeze_info[
                "component_info"
            ]
        ),
        "freeze": asdict(
            FREEZE
        ),
        "Tropical_Atlantic_used_for_selection": False,
        "Tropical_Atlantic_used_for_alpha": False,
    }

    frozen_path = (
        output_dir
        / "FROZEN_BEFORE_TROPICAL.json"
    )

    save_json(
        frozen_path,
        frozen,
    )

    log("")
    log(
        f"[FROZEN BEFORE TROPICAL] seed={selected_seed} | "
        f"epochs={final_epoch} | "
        f"activeU/V={bool(active[0])}/{bool(active[1])} | "
        f"alphaU/V={frozen_alphas[0]:.6f}/{frozen_alphas[1]:.6f}"
    )

    log(
        f"[SAVED BEFORE TROPICAL LOAD] {frozen_path}"
    )

    # ---------------------------------------------------------
    # Final training on all development missions.
    # ---------------------------------------------------------
    dev_all = concat_missions(
        [
            dev_missions[
                m
            ]
            for m in DEV_MISSIONS
        ]
    )

    final_bundle = train_final_network(
        torch=torch,
        nn=nn,
        dev_all=dev_all,
        final_seed=selected_seed,
        final_epochs=final_epoch,
        device=device,
    )

    # Save model BEFORE Tropical loading.
    model_dir = (
        output_dir
        / "final_model"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        model_dir
        / "14A_final.pt"
    )

    torch.save(
        {
            "stage": "14A-final",
            "script_version": (
                SCRIPT_VERSION
            ),
            "selected_seed": int(
                selected_seed
            ),
            "final_epoch": int(
                final_epoch
            ),
            "active_components": (
                active.astype(
                    bool
                )
            ),
            "frozen_alphas": (
                frozen_alphas.astype(
                    np.float32
                )
            ),
            "residual_state_dict": (
                state_dict_cpu(
                    final_bundle[
                        "residual_model"
                    ]
                )
            ),
            "residual_scalers": (
                final_bundle[
                    "residual_scalers"
                ]
            ),
            "freeze": asdict(
                FREEZE
            ),
        },
        checkpoint_path,
    )

    log("")
    log(
        "[STAGE C] Everything frozen. Loading Tropical Atlantic NOW."
    )

    tropical = load_mission(
        dataset_root,
        FINAL_MISSION,
    )

    log(
        f"[TROPICAL DATASET] {tropical['path']}"
    )

    # ---------------------------------------------------------
    # Tropical inference.
    # ---------------------------------------------------------
    Xtrop_base = build_base4(
        tropical[
            "X_full9"
        ]
    )

    ridge_trop = ridge_predict(
        final_bundle[
            "ridge_model"
        ],
        final_bundle[
            "ridge_scaler"
        ],
        Xtrop_base,
    )

    residual_features_trop = build_residual_features(
        tropical[
            "X_full9"
        ],
        ridge_trop,
    )

    transformed_trop = transform_residual_features(
        residual_features_trop,
        final_bundle[
            "residual_scalers"
        ],
    )

    amp_enabled = bool(
        FREEZE.use_amp
        and device.type == "cuda"
    )

    raw_trop = predict_raw_residual(
        torch=torch,
        model=final_bundle[
            "residual_model"
        ],
        transformed=transformed_trop,
        residual_std=final_bundle[
            "residual_scalers"
        ][
            "residual_std"
        ],
        device=device,
        amp_enabled=amp_enabled,
    )

    final_trop = apply_alphas(
        ridge_trop,
        raw_trop,
        frozen_alphas,
    )

    ridge_metrics = evaluate_wind(
        tropical[
            "y_uv"
        ],
        ridge_trop,
    )

    final_metrics = evaluate_wind(
        tropical[
            "y_uv"
        ],
        final_trop,
    )

    true_res_trop = (
        tropical[
            "y_uv"
        ]
        - ridge_trop
    )

    corr_u_trop = corrcoef_safe(
        true_res_trop[:, 0],
        raw_trop[:, 0],
    )

    corr_v_trop = corrcoef_safe(
        true_res_trop[:, 1],
        raw_trop[:, 1],
    )

    total_params = int(
        final_bundle[
            "parameter_count_nn"
        ]
        + final_bundle[
            "parameter_count_ridge"
        ]
    )

    final_table = pd.DataFrame(
        [
            {
                "model": (
                    "BP-STGNN-published"
                ),
                "U_RMSE_mps": (
                    BP_REFERENCE[
                        "U_RMSE_mps"
                    ]
                ),
                "V_RMSE_mps": (
                    BP_REFERENCE[
                        "V_RMSE_mps"
                    ]
                ),
                "vector_RMSE_mps": (
                    np.nan
                ),
                "WS_RMSE_mps": (
                    BP_REFERENCE[
                        "WS_RMSE_mps"
                    ]
                ),
                "WD_RMSE_deg": (
                    BP_REFERENCE[
                        "WD_RMSE_deg"
                    ]
                ),
                "parameter_count": int(
                    BP_REFERENCE[
                        "params"
                    ]
                ),
            },
            {
                "model": (
                    "Base4-Ridge-final"
                ),
                **ridge_metrics,
                "parameter_count": int(
                    final_bundle[
                        "parameter_count_ridge"
                    ]
                ),
            },
            {
                "model": (
                    "Ridge+ComponentSafeResidual-final"
                ),
                **final_metrics,
                "parameter_count": int(
                    total_params
                ),
            },
        ]
    )

    final_path = (
        output_dir
        / "FINAL_TROPICAL_14A.csv"
    )

    final_table.to_csv(
        final_path,
        index=False,
        encoding="utf-8-sig",
    )

    improvement = {
        "U_fraction": rel_improve(
            ridge_metrics[
                "U_RMSE_mps"
            ],
            final_metrics[
                "U_RMSE_mps"
            ],
        ),
        "V_fraction": rel_improve(
            ridge_metrics[
                "V_RMSE_mps"
            ],
            final_metrics[
                "V_RMSE_mps"
            ],
        ),
        "vector_fraction": rel_improve(
            ridge_metrics[
                "vector_RMSE_mps"
            ],
            final_metrics[
                "vector_RMSE_mps"
            ],
        ),
        "WS_fraction": rel_improve(
            ridge_metrics[
                "WS_RMSE_mps"
            ],
            final_metrics[
                "WS_RMSE_mps"
            ],
        ),
        "WD_fraction": rel_improve(
            ridge_metrics[
                "WD_RMSE_deg"
            ],
            final_metrics[
                "WD_RMSE_deg"
            ],
        ),
    }

    log("")
    log("=" * 140)
    log(
        "FINAL TROPICAL ATLANTIC — 14A TWO-STAGE COMPONENT-SAFE RESIDUAL"
    )
    log("=" * 140)
    log(
        final_table.to_string(
            index=False
        )
    )

    log("")
    log(
        "Frozen component-safe correction:"
    )

    log(
        f"  active U/V = {bool(active[0])} / {bool(active[1])}"
    )

    log(
        f"  alpha U/V  = {frozen_alphas[0]:.6f} / {frozen_alphas[1]:.6f}"
    )

    log(
        f"  Tropical raw residual corr U/V = "
        f"{corr_u_trop:+.4f} / {corr_v_trop:+.4f}"
    )

    log("")
    log(
        "Improvement vs Base4-Ridge:"
    )

    for key, value in improvement.items():
        log(
            f"  {key:<16}: {100*value:+.3f}%"
        )

    # Save prediction vectors for later ablation.
    np.savez_compressed(
        output_dir
        / "FINAL_TROPICAL_PREDICTIONS.npz",
        y_true=tropical[
            "y_uv"
        ].astype(
            np.float32
        ),
        ridge_pred=ridge_trop.astype(
            np.float32
        ),
        raw_residual_pred=raw_trop.astype(
            np.float32
        ),
        final_pred=final_trop.astype(
            np.float32
        ),
        active_components=active.astype(
            bool
        ),
        frozen_alphas=frozen_alphas.astype(
            np.float32
        ),
    )

    report = {
        "stage": "14A",
        "training_scheme": (
            "two-stage: Ridge first; Ridge frozen; residual NN second"
        ),
        "frozen_before_tropical": (
            frozen
        ),
        "tropical_raw_residual_corr": {
            "U": float(
                corr_u_trop
            ),
            "V": float(
                corr_v_trop
            ),
        },
        "ridge_metrics": (
            ridge_metrics
        ),
        "final_metrics": (
            final_metrics
        ),
        "improvement_vs_ridge": (
            improvement
        ),
        "parameter_count": {
            "ridge": int(
                final_bundle[
                    "parameter_count_ridge"
                ]
            ),
            "residual_nn": int(
                final_bundle[
                    "parameter_count_nn"
                ]
            ),
            "total": int(
                total_params
            ),
            "BP_STGNN": int(
                BP_REFERENCE[
                    "params"
                ]
            ),
        },
        "primary_success_criterion": (
            "repeatable improvement over Base4-Ridge; "
            "all-four BP dominance is not required"
        ),
    }

    save_json(
        output_dir
        / "14A_REPORT.json",
        report,
    )

    with (
        output_dir
        / "14A_REPORT.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "14A RIDGE-ANCHORED COMPONENT-SAFE RESIDUAL NETWORK\n"
        )

        f.write(
            "=" * 120
            + "\n\n"
        )

        f.write(
            "TWO-STAGE TRAINING\n"
        )

        f.write(
            "Round 1: Base4-Ridge\n"
        )

        f.write(
            "Round 2: Ridge frozen; residual GRU learns blocked-OOF Ridge error\n\n"
        )

        f.write(
            "DEVELOPMENT SEED SUMMARY\n"
        )

        f.write(
            "-" * 120
            + "\n"
        )

        f.write(
            seed_summary.to_string(
                index=False
            )
        )

        f.write(
            "\n\nCOMPONENT FREEZE\n"
        )

        f.write(
            "-" * 120
            + "\n"
        )

        f.write(
            json.dumps(
                {
                    "active_U": bool(
                        active[0]
                    ),
                    "active_V": bool(
                        active[1]
                    ),
                    "alpha_U": float(
                        frozen_alphas[0]
                    ),
                    "alpha_V": float(
                        frozen_alphas[1]
                    ),
                    "component_info": freeze_info[
                        "component_info"
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        f.write(
            "\n\nFINAL TROPICAL\n"
        )

        f.write(
            "-" * 120
            + "\n"
        )

        f.write(
            final_table.to_string(
                index=False
            )
        )

        f.write(
            "\n\nIMPROVEMENT VS RIDGE\n"
        )

        f.write(
            "-" * 120
            + "\n"
        )

        for key, value in improvement.items():
            f.write(
                f"{key}: {100*value:+.6f}%\n"
            )

    log("")
    log(
        f"[SAVED] {final_path}"
    )

    log(
        f"[SAVED] {checkpoint_path}"
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
