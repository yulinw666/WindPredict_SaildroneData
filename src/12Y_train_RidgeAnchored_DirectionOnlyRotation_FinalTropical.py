# -*- coding: utf-8 -*-
r"""
12Y_train_RidgeAnchored_DirectionOnlyRotation_FinalTropical.py

Stage 12Y
=========
BP-aligned, compact DIRECTION-ONLY correction on top of the strong Base4-Ridge.

Motivation from Stage 12X-v2 Tropical Atlantic
----------------------------------------------
Base4-Ridge:
    U  RMSE = 0.717330 m/s   < BP 0.748640
    V  RMSE = 0.791994 m/s   > BP 0.705060
    WS RMSE = 0.685559 m/s   < BP 0.699400
    WD RMSE = 5.998720 deg   > BP 5.584000

Thus:
    - U is already strong.
    - WS is already strong.
    - V and WD are the remaining weaknesses.

Stage 12X-v2 local m/s residual correction degraded Tropical V/WS/WD and had:
    corr_parallel = +0.0696
    corr_cross    = -0.0569

Therefore Stage 12Y does NOT predict any speed/magnitude correction.

FINAL MODEL
-----------
1) Global Base4-Ridge predicts:
       W_R = [U_R, V_R]

2) A tiny neural network predicts ONLY a bounded angular correction:
       delta_theta

3) Rotate the Ridge vector without changing its magnitude:
       W_final = R(delta_theta) W_R

Therefore, mathematically:
       |W_final| == |W_R|

and:
       WS_final == WS_Ridge

up to floating-point roundoff.

This protects the already-BP-beating WS result by construction.

BP-ALIGNED INPUT/TARGET
-----------------------
Stage 12G point-sampled benchmark.

History:
    six original 1-min-mean records sampled every 10 min:
        t-50, t-40, t-30, t-20, t-10, t

Target:
    original 1-min-mean U,V at t+10 min.

No:
    pressure
    wing angle
    extra 1-min records
    10-min arithmetic averaging
    absolute position
    speed residual head

DIRECTION REPRESENTATION
------------------------
For every sample, the latest reliable historical wind defines a local frame.

Historical wind is represented by:
    speed ratio relative to latest reliable speed
    sin(relative direction)
    cos(relative direction)
    direction increment sin/cos
    temperature
    relative humidity

Boat motion is represented only as:
    boat_parallel / speed_scale
    boat_cross    / speed_scale

All are deterministic transforms of the same BP-aligned U,V,T,RH,SOG,COG.

RESIDUAL TARGET
---------------
The network is NOT asked to predict target direction from scratch.

The relevant angular residual is:

    delta_theta_true
      = wrap(theta_true - theta_Ridge)

Training, however, minimizes the reconstructed final U/V, V-heavy loss and
circular direction loss directly.

SAFE ANGULAR CORRECTION
-----------------------
    delta_theta =
        cap_deg
        * sigmoid(gate_logit)
        * tanh(angle_logit)

The angle head is zero initialized:
    epoch 0 = EXACT Base4-Ridge.

Development cap candidates:
    +/- 2 deg
    +/- 4 deg
    +/- 6 deg

selected only on Antarctic / Atlantic / West Coast LOMO.

LOSS
----
Because the objective is now specifically V + WD while protecting U:

    L =
        0.25 * normalized U MSE
      + 0.60 * normalized V MSE
      + 0.12 * circular direction cosine loss
      + 0.03 * normalized angle-correction penalty

There is NO WS loss because rotation preserves predicted WS exactly.

NETWORK
-------
Compact MLP:
    flattened 6 x 10 sequence
    + 10 summary
    + 4 Ridge/current context
        -> 64
        -> 32
        -> 16
        -> angle_logit, gate_logit

Expected parameter count: only a few thousand.

DEVELOPMENT
-----------
LOMO using only:
    Antarctic
    Atlantic
    West Coast

Architecture is fixed.
Select:
    angle cap
    seed
    final epoch count

using development only.

FINAL
-----
After FROZEN_BEFORE_TROPICAL.json is written:
    fit Base4-Ridge on all three development missions
    train direction-only correction for frozen epoch count
    load Tropical Atlantic
    evaluate once

Tropical filename resolver supports:
    Tropical_Atlantic_TEST.npz
and legacy:
    Tropical_Atlantic.npz

Published BP-STGNN reference:
    U  0.748640
    V  0.705060
    WS 0.699400
    WD 5.584
    params 242580
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
        "scikit-learn is required."
    ) from exc


SCRIPT_VERSION = "0.1.0-12Y-RidgeDirectionOnlyRotation"

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
    / "12Y_RidgeDirectionOnlyRotation_FinalTropical_v0_1"
)

DEV_MISSIONS = [
    "Antarctic",
    "Atlantic",
    "West Coast",
]
FINAL_MISSION = "Tropical Atlantic"

LOOKBACK_STEPS = 6
STEP_MINUTES = 10
TEN_MIN_NS = STEP_MINUTES * 60 * 1_000_000_000

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
IDX_SOG = 4
IDX_COG_SIN = 5
IDX_COG_COS = 6

RIDGE_ALPHA = 1.0
INNER_CROSSFIT_FOLDS = 5
REFERENCE_WIND_MIN_MPS = 0.5
EPS = 1e-12

SEEDS = [
    500043,
    501052,
    502061,
]

ANGLE_CAPS_DEG = [
    2.0,
    4.0,
    6.0,
]

BP_REFERENCE = {
    "U_RMSE_mps": 0.748640,
    "V_RMSE_mps": 0.705060,
    "WS_RMSE_mps": 0.699400,
    "WD_RMSE_deg": 5.584000,
    "params": 242580,
}


@dataclass(frozen=True)
class Freeze:
    hidden1: int = 64
    hidden2: int = 32
    hidden3: int = 16
    dropout: float = 0.10

    learning_rate: float = 7e-4
    weight_decay: float = 1e-5
    batch_size: int = 256

    max_epochs: int = 220
    early_stop_patience: int = 25
    scheduler_patience: int = 8
    scheduler_factor: float = 0.5
    min_learning_rate: float = 8e-6
    grad_clip_norm: float = 1.0

    weight_u: float = 0.25
    weight_v: float = 0.60
    weight_direction: float = 0.12
    weight_angle_penalty: float = 0.03

    use_amp: bool = True


FREEZE = Freeze()

SEQ_FEATURE_NAMES = [
    "speed_ratio",
    "rel_dir_sin",
    "rel_dir_cos",
    "turn_sin",
    "turn_cos",
    "T",
    "RH",
    "boat_parallel_norm",
    "boat_cross_norm",
    "speed_change_norm",
]

SUMMARY_FEATURE_NAMES = [
    "latest_speed",
    "std_speed_ratio",
    "mean_abs_turn_sin",
    "last_turn_sin",
    "last_turn_cos",
    "speed_slope_norm",
    "T_last",
    "RH_last",
    "boat_parallel_last_norm",
    "boat_cross_last_norm",
]


# =============================================================================
# Utilities
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


def append_csv(path: Path, row: dict):
    new = pd.DataFrame([row])

    if path.exists():
        old = pd.read_csv(path)

        keys = [
            "candidate",
            "held_out_mission",
            "seed",
        ]

        if all(k in old.columns for k in keys):
            mask = (
                (old["candidate"].astype(str) == str(row["candidate"]))
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


def import_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required."
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

    for p in candidates:
        if p.exists():
            return p

    attempted = "\n".join(
        f"  - {p}"
        for p in candidates
    )

    raise FileNotFoundError(
        f"Missing mission {mission}. Tried:\n{attempted}"
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
            for x in z["feature_names"].tolist()
        ]

    if names != FULL9_FEATURE_NAMES:
        raise RuntimeError(
            f"{mission}: unexpected feature names."
        )

    if (
        X.ndim != 3
        or X.shape[1:] != (
            LOOKBACK_STEPS,
            len(FULL9_FEATURE_NAMES),
        )
    ):
        raise RuntimeError(
            f"{mission}: X shape {X.shape}"
        )

    if y.ndim == 3:
        y = y[:, 0, :]

    if (
        y.ndim != 2
        or y.shape[1] != 2
    ):
        raise RuntimeError(
            f"{mission}: y shape {y.shape}"
        )

    if not np.all(
        target - context
        == TEN_MIN_NS
    ):
        raise RuntimeError(
            f"{mission}: not exactly +10 min."
        )

    if not (
        np.isfinite(X).all()
        and np.isfinite(y).all()
    ):
        raise RuntimeError(
            f"{mission}: nonfinite."
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
        "mission_labels": np.concatenate(
            [
                np.asarray(
                    [x["mission"]] * len(x["X_full9"]),
                    dtype=object,
                )
                for x in items
            ]
        ),
    }


# =============================================================================
# Ridge
# =============================================================================

def build_base4(X_full9):
    return np.asarray(
        X_full9[
            :,
            :,
            [
                IDX_U,
                IDX_V,
                IDX_T,
                IDX_RH,
            ]
        ],
        dtype=np.float32,
    )


def fit_scaler(X, y):
    x_mean = np.mean(
        X.astype(np.float64),
        axis=(0, 1),
    )

    x_std = np.std(
        X.astype(np.float64),
        axis=(0, 1),
    )

    x_std = np.where(
        x_std < 1e-8,
        1.0,
        x_std,
    )

    y_mean = np.mean(
        y.astype(np.float64),
        axis=0,
    )

    y_std = np.std(
        y.astype(np.float64),
        axis=0,
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


def transform_X(X, scaler):
    return (
        (
            X.astype(np.float32)
            - scaler["x_mean"][None, None, :]
        )
        / scaler["x_std"][None, None, :]
    ).astype(np.float32)


def transform_y(y, scaler):
    return (
        (
            y.astype(np.float32)
            - scaler["y_mean"][None, :]
        )
        / scaler["y_std"][None, :]
    ).astype(np.float32)


def inverse_y(yz, scaler):
    return (
        yz.astype(np.float32)
        * scaler["y_std"][None, :]
        + scaler["y_mean"][None, :]
    ).astype(np.float32)


def fit_ridge(X, y):
    scaler = fit_scaler(
        X,
        y,
    )

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        transform_X(
            X,
            scaler,
        ).reshape(
            len(X),
            -1,
        ),
        transform_y(
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
    pred_z = model.predict(
        transform_X(
            X,
            scaler,
        ).reshape(
            len(X),
            -1,
        )
    )

    return inverse_y(
        np.asarray(
            pred_z,
            dtype=np.float32,
        ),
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


# =============================================================================
# Angles / metrics
# =============================================================================

def vector_angle_rad(uv):
    uv = np.asarray(
        uv,
        dtype=np.float64,
    )

    return np.arctan2(
        uv[:, 1],
        uv[:, 0],
    )


def wrap_rad(x):
    return (
        np.asarray(
            x,
            dtype=np.float64,
        )
        + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


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
                        ws_p - ws_t
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


def score_direction_task(
    metrics,
    ridge_metrics,
):
    # Main development objective:
    # V + WD, while U is protected.
    # WS is omitted because it is invariant to rotation.
    u_ratio = (
        metrics[
            "wind_U_RMSE_mps"
        ]
        / ridge_metrics[
            "wind_U_RMSE_mps"
        ]
    )

    v_ratio = (
        metrics[
            "wind_V_RMSE_mps"
        ]
        / ridge_metrics[
            "wind_V_RMSE_mps"
        ]
    )

    wd_ratio = (
        metrics[
            "wind_direction_RMSE_deg"
        ]
        / ridge_metrics[
            "wind_direction_RMSE_deg"
        ]
    )

    return float(
        0.20 * u_ratio
        + 0.55 * v_ratio
        + 0.25 * wd_ratio
    )


def corrcoef_safe(a, b):
    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    if (
        len(a) < 3
        or np.std(a) < EPS
        or np.std(b) < EPS
    ):
        return np.nan

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


# =============================================================================
# OOF Ridge
# =============================================================================

def blocked_fold_ids(
    labels,
):
    labels = np.asarray(
        labels,
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
            INNER_CROSSFIT_FOLDS,
        )

        for k, chunk in enumerate(
            chunks
        ):
            fold_ids[
                chunk
            ] = k

    return fold_ids


def crossfit_ridge(
    X,
    y,
    labels,
):
    fold_ids = blocked_fold_ids(
        labels
    )

    pred = np.full_like(
        y,
        np.nan,
        dtype=np.float32,
    )

    for k in range(
        INNER_CROSSFIT_FOLDS
    ):
        va = fold_ids == k
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

    if not np.isfinite(pred).all():
        raise RuntimeError(
            "Nonfinite OOF Ridge."
        )

    return pred


# =============================================================================
# Direction feature representation
# =============================================================================

def build_reference_basis(
    X_full9,
):
    X = np.asarray(
        X_full9,
        dtype=np.float64,
    )

    wind = X[
        :,
        :,
        [
            IDX_U,
            IDX_V,
        ]
    ]

    speed = np.linalg.norm(
        wind,
        axis=2,
    )

    n = len(X)

    ep = np.zeros(
        (n, 2),
        dtype=np.float64,
    )

    ref_speed = np.ones(
        n,
        dtype=np.float64,
    )

    unresolved = np.ones(
        n,
        dtype=bool,
    )

    for k in range(
        LOOKBACK_STEPS - 1,
        -1,
        -1,
    ):
        choose = (
            unresolved
            & (
                speed[:, k]
                >= REFERENCE_WIND_MIN_MPS
            )
        )

        if np.any(choose):
            s = speed[choose, k]

            ep[
                choose,
                :
            ] = (
                wind[
                    choose,
                    k,
                    :
                ]
                / s[:, None]
            )

            ref_speed[
                choose
            ] = s

            unresolved[
                choose
            ] = False

    if np.any(
        unresolved
    ):
        ep[
            unresolved,
            0
        ] = 1.0

        ep[
            unresolved,
            1
        ] = 0.0

        ref_speed[
            unresolved
        ] = 1.0

    ec = np.stack(
        [
            -ep[:, 1],
            ep[:, 0],
        ],
        axis=1,
    )

    return (
        ep.astype(
            np.float32
        ),
        ec.astype(
            np.float32
        ),
        ref_speed.astype(
            np.float32
        ),
    )


def rotate_history_local(
    wind,
    ep,
    ec,
):
    p = (
        wind[:, :, 0]
        * ep[:, None, 0]
        + wind[:, :, 1]
        * ep[:, None, 1]
    )

    c = (
        wind[:, :, 0]
        * ec[:, None, 0]
        + wind[:, :, 1]
        * ec[:, None, 1]
    )

    return np.stack(
        [p, c],
        axis=-1,
    )


def slope6(x):
    t = np.arange(
        LOOKBACK_STEPS,
        dtype=np.float64,
    )

    t -= t.mean()

    centered = (
        x
        - np.mean(
            x,
            axis=1,
            keepdims=True,
        )
    )

    return (
        np.sum(
            centered
            * t[None, :],
            axis=1,
        )
        / np.sum(
            t ** 2
        )
    )


def build_direction_features(
    X_full9,
    ridge_anchor,
):
    X = np.asarray(
        X_full9,
        dtype=np.float64,
    )

    ridge_anchor = np.asarray(
        ridge_anchor,
        dtype=np.float64,
    )

    wind = X[
        :,
        :,
        [
            IDX_U,
            IDX_V,
        ]
    ]

    (
        ep,
        ec,
        ref_speed,
    ) = build_reference_basis(
        X_full9
    )

    local = rotate_history_local(
        wind,
        ep,
        ec,
    )

    speed = np.linalg.norm(
        local,
        axis=2,
    )

    speed_scale = np.maximum(
        ref_speed.astype(
            np.float64
        ),
        1.0,
    )

    speed_ratio = (
        speed
        / speed_scale[
            :,
            None
        ]
    )

    local_angle = np.arctan2(
        local[:, :, 1],
        local[:, :, 0],
    )

    rel_sin = np.sin(
        local_angle
    )

    rel_cos = np.cos(
        local_angle
    )

    dtheta = np.zeros_like(
        local_angle
    )

    dtheta[:, 1:] = wrap_rad(
        local_angle[:, 1:]
        - local_angle[:, :-1]
    )

    turn_sin = np.sin(
        dtheta
    )

    turn_cos = np.cos(
        dtheta
    )

    T = X[:, :, IDX_T]
    RH = X[:, :, IDX_RH]

    SOG = X[:, :, IDX_SOG]

    boat_E = (
        SOG
        * X[
            :,
            :,
            IDX_COG_SIN,
        ]
    )

    boat_N = (
        SOG
        * X[
            :,
            :,
            IDX_COG_COS,
        ]
    )

    boat_parallel = (
        boat_E
        * ep[
            :,
            None,
            0
        ]
        + boat_N
        * ep[
            :,
            None,
            1
        ]
    )

    boat_cross = (
        boat_E
        * ec[
            :,
            None,
            0
        ]
        + boat_N
        * ec[
            :,
            None,
            1
        ]
    )

    boat_parallel_norm = (
        boat_parallel
        / speed_scale[
            :,
            None
        ]
    )

    boat_cross_norm = (
        boat_cross
        / speed_scale[
            :,
            None
        ]
    )

    dspeed = np.zeros_like(
        speed
    )

    dspeed[:, 1:] = (
        speed[:, 1:]
        - speed[:, :-1]
    )

    speed_change_norm = (
        dspeed
        / speed_scale[
            :,
            None
        ]
    )

    seq = np.stack(
        [
            speed_ratio,
            rel_sin,
            rel_cos,
            turn_sin,
            turn_cos,
            T,
            RH,
            boat_parallel_norm,
            boat_cross_norm,
            speed_change_norm,
        ],
        axis=-1,
    ).astype(
        np.float32
    )

    summary = np.stack(
        [
            speed[:, -1],
            np.std(
                speed_ratio,
                axis=1,
            ),
            np.mean(
                np.abs(
                    turn_sin
                ),
                axis=1,
            ),
            turn_sin[:, -1],
            turn_cos[:, -1],
            slope6(
                speed_ratio
            ),
            T[:, -1],
            RH[:, -1],
            boat_parallel_norm[
                :,
                -1
            ],
            boat_cross_norm[
                :,
                -1
            ],
        ],
        axis=1,
    ).astype(
        np.float32
    )

    ridge_angle = vector_angle_rad(
        ridge_anchor
    )

    ridge_speed = np.linalg.norm(
        ridge_anchor,
        axis=1,
    )

    latest_angle_global = vector_angle_rad(
        wind[:, -1, :]
    )

    angle_from_latest = wrap_rad(
        ridge_angle
        - latest_angle_global
    )

    context = np.stack(
        [
            ridge_speed,
            np.sin(
                angle_from_latest
            ),
            np.cos(
                angle_from_latest
            ),
            speed[:, -1],
        ],
        axis=1,
    ).astype(
        np.float32
    )

    if not (
        np.isfinite(seq).all()
        and np.isfinite(summary).all()
        and np.isfinite(context).all()
    ):
        raise RuntimeError(
            "Nonfinite direction features."
        )

    return {
        "seq": seq,
        "summary": summary,
        "context": context,
    }


# =============================================================================
# Feature scaling
# =============================================================================

def fit_feature_scalers(
    features,
    y,
):
    seq = features["seq"].astype(
        np.float64
    )

    summary = features[
        "summary"
    ].astype(
        np.float64
    )

    context = features[
        "context"
    ].astype(
        np.float64
    )

    def ms(arr, axis):
        mean = np.mean(
            arr,
            axis=axis,
        )
        std = np.std(
            arr,
            axis=axis,
        )
        std = np.where(
            std < 1e-8,
            1.0,
            std,
        )
        return (
            mean.astype(
                np.float32
            ),
            std.astype(
                np.float32
            ),
        )

    seq_mean, seq_std = ms(
        seq,
        (0, 1),
    )

    summary_mean, summary_std = ms(
        summary,
        0,
    )

    context_mean, context_std = ms(
        context,
        0,
    )

    y_std = np.std(
        y.astype(
            np.float64
        ),
        axis=0,
    )

    y_std = np.where(
        y_std < 1e-4,
        1.0,
        y_std,
    )

    return {
        "seq_mean": seq_mean,
        "seq_std": seq_std,
        "summary_mean": summary_mean,
        "summary_std": summary_std,
        "context_mean": context_mean,
        "context_std": context_std,
        "y_std": y_std.astype(
            np.float32
        ),
    }


def transform_features(
    features,
    scalers,
):
    return {
        "seq_z": (
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
        "summary_z": (
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
        "context_z": (
            (
                features[
                    "context"
                ]
                - scalers[
                    "context_mean"
                ][
                    None,
                    :
                ]
            )
            / scalers[
                "context_std"
            ][
                None,
                :
            ]
        ).astype(
            np.float32
        ),
    }


# =============================================================================
# Model
# =============================================================================

def build_model_class(
    torch,
    nn,
    angle_cap_deg,
):
    input_dim = (
        LOOKBACK_STEPS
        * len(
            SEQ_FEATURE_NAMES
        )
        + len(
            SUMMARY_FEATURE_NAMES
        )
        + 4
    )

    class DirectionOnlyMLP(nn.Module):
        def __init__(self):
            super().__init__()

            self.backbone = nn.Sequential(
                nn.Linear(
                    input_dim,
                    FREEZE.hidden1,
                ),
                nn.LayerNorm(
                    FREEZE.hidden1
                ),
                nn.GELU(),
                nn.Dropout(
                    FREEZE.dropout
                ),
                nn.Linear(
                    FREEZE.hidden1,
                    FREEZE.hidden2,
                ),
                nn.GELU(),
                nn.Dropout(
                    FREEZE.dropout
                ),
                nn.Linear(
                    FREEZE.hidden2,
                    FREEZE.hidden3,
                ),
                nn.GELU(),
            )

            self.out = nn.Linear(
                FREEZE.hidden3,
                2,
            )

            nn.init.zeros_(
                self.out.weight
            )

            with torch.no_grad():
                self.out.bias[
                    0
                ] = 0.0

                self.out.bias[
                    1
                ] = -2.0

        def forward(
            self,
            seq_z,
            summary_z,
            context_z,
        ):
            x = torch.cat(
                [
                    seq_z.flatten(
                        start_dim=1
                    ),
                    summary_z,
                    context_z,
                ],
                dim=1,
            )

            raw = self.out(
                self.backbone(x)
            )

            gate = torch.sigmoid(
                raw[:, 1]
            )

            angle_unit = torch.tanh(
                raw[:, 0]
            )

            cap_rad = (
                float(
                    angle_cap_deg
                )
                * math.pi
                / 180.0
            )

            delta = (
                cap_rad
                * gate
                * angle_unit
            )

            return (
                delta,
                gate,
            )

    return DirectionOnlyMLP


def rotate_uv_torch(
    torch,
    uv,
    delta_rad,
):
    c = torch.cos(
        delta_rad
    )

    s = torch.sin(
        delta_rad
    )

    u = uv[:, 0]
    v = uv[:, 1]

    u2 = (
        c * u
        - s * v
    )

    v2 = (
        s * u
        + c * v
    )

    return torch.stack(
        [
            u2,
            v2,
        ],
        dim=1,
    )


def rotate_uv_numpy(
    uv,
    delta_rad,
):
    uv = np.asarray(
        uv,
        dtype=np.float64,
    )

    d = np.asarray(
        delta_rad,
        dtype=np.float64,
    )

    c = np.cos(d)
    s = np.sin(d)

    out = np.stack(
        [
            c * uv[:, 0]
            - s * uv[:, 1],
            s * uv[:, 0]
            + c * uv[:, 1],
        ],
        axis=1,
    )

    return out.astype(
        np.float32
    )


# =============================================================================
# Loss / prediction
# =============================================================================

def direction_loss(
    torch,
    final_uv,
    true_uv,
    delta_rad,
    y_std,
    angle_cap_deg,
):
    err = (
        final_uv
        - true_uv
    )

    err_z = (
        err
        / y_std[
            None,
            :
        ]
    )

    comp = (
        FREEZE.weight_u
        * torch.mean(
            err_z[:, 0] ** 2
        )
        + FREEZE.weight_v
        * torch.mean(
            err_z[:, 1] ** 2
        )
    )

    sp = torch.linalg.vector_norm(
        final_uv,
        dim=1,
    )

    st = torch.linalg.vector_norm(
        true_uv,
        dim=1,
    )

    denom = (
        sp * st
    ).clamp_min(
        1e-4
    )

    cos_sim = (
        torch.sum(
            final_uv
            * true_uv,
            dim=1,
        )
        / denom
    )

    cos_sim = torch.clamp(
        cos_sim,
        -1.0,
        1.0,
    )

    mask = (
        st
        >= REFERENCE_WIND_MIN_MPS
    )

    if torch.any(mask):
        ldir = torch.mean(
            1.0
            - cos_sim[
                mask
            ]
        )
    else:
        ldir = (
            comp * 0.0
        )

    cap_rad = max(
        float(
            angle_cap_deg
        )
        * math.pi
        / 180.0,
        1e-6,
    )

    langle = torch.mean(
        (
            delta_rad
            / cap_rad
        ) ** 2
    )

    return (
        comp
        + FREEZE.weight_direction
        * ldir
        + FREEZE.weight_angle_penalty
        * langle
    )


def predict(
    *,
    torch,
    model,
    transformed,
    ridge_anchor,
    device,
    amp_enabled,
):
    model.eval()

    delta_chunks = []
    gate_chunks = []

    n = len(
        ridge_anchor
    )

    with torch.no_grad():
        for start in range(
            0,
            n,
            FREEZE.batch_size,
        ):
            stop = min(
                start
                + FREEZE.batch_size,
                n,
            )

            seq_t = torch.from_numpy(
                transformed[
                    "seq_z"
                ][
                    start:stop
                ]
            ).to(
                device
            )

            summary_t = torch.from_numpy(
                transformed[
                    "summary_z"
                ][
                    start:stop
                ]
            ).to(
                device
            )

            context_t = torch.from_numpy(
                transformed[
                    "context_z"
                ][
                    start:stop
                ]
            ).to(
                device
            )

            with amp_context(
                torch,
                amp_enabled,
                device,
            ):
                delta, gate = model(
                    seq_t,
                    summary_t,
                    context_t,
                )

            delta_chunks.append(
                delta.detach()
                .float()
                .cpu()
                .numpy()
            )

            gate_chunks.append(
                gate.detach()
                .float()
                .cpu()
                .numpy()
            )

    delta = np.concatenate(
        delta_chunks
    )

    gate = np.concatenate(
        gate_chunks
    )

    final = rotate_uv_numpy(
        ridge_anchor,
        delta,
    )

    return (
        final,
        delta,
        gate,
    )


# =============================================================================
# Development run
# =============================================================================

def train_dev_run(
    *,
    torch,
    nn,
    angle_cap_deg,
    seed,
    held_out,
    train_data,
    val_data,
    output_dir,
    device,
    debug_fast,
):
    candidate = (
        f"AngleCap{int(angle_cap_deg)}deg"
    )

    Xtr = build_base4(
        train_data[
            "X_full9"
        ]
    )

    Xva = build_base4(
        val_data[
            "X_full9"
        ]
    )

    ytr = train_data[
        "y_uv"
    ]

    yva = val_data[
        "y_uv"
    ]

    ridge_model, ridge_scaler = fit_ridge(
        Xtr,
        ytr,
    )

    ridge_va = ridge_predict(
        ridge_model,
        ridge_scaler,
        Xva,
    )

    ridge_metrics = evaluate_wind(
        yva,
        ridge_va,
    )

    ridge_oof = crossfit_ridge(
        Xtr,
        ytr,
        train_data[
            "mission_labels"
        ],
    )

    feat_tr = build_direction_features(
        train_data[
            "X_full9"
        ],
        ridge_oof,
    )

    feat_va = build_direction_features(
        val_data[
            "X_full9"
        ],
        ridge_va,
    )

    scalers = fit_feature_scalers(
        feat_tr,
        ytr,
    )

    tr = transform_features(
        feat_tr,
        scalers,
    )

    va = transform_features(
        feat_va,
        scalers,
    )

    seed_everything(
        torch,
        seed,
    )

    ModelClass = build_model_class(
        torch,
        nn,
        angle_cap_deg,
    )

    model = ModelClass().to(
        device
    )

    nn_params = count_parameters(
        model
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FREEZE.learning_rate,
        weight_decay=(
            FREEZE.weight_decay
        ),
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

    y_std_t = torch.as_tensor(
        scalers[
            "y_std"
        ],
        dtype=torch.float32,
        device=device,
    )

    best_score = 1.0
    best_state = state_dict_cpu(
        model
    )
    best_epoch = 0
    patience = 0

    max_epochs = (
        12
        if debug_fast
        else FREEZE.max_epochs
    )

    patience_limit = (
        5
        if debug_fast
        else FREEZE.early_stop_patience
    )

    history = []

    n = len(ytr)

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        model.train()

        loss_sum = 0.0
        n_seen = 0

        for start in range(
            0,
            n,
            FREEZE.batch_size,
        ):
            stop = min(
                start
                + FREEZE.batch_size,
                n,
            )

            seq_t = torch.from_numpy(
                tr[
                    "seq_z"
                ][
                    start:stop
                ]
            ).to(
                device
            )

            summary_t = torch.from_numpy(
                tr[
                    "summary_z"
                ][
                    start:stop
                ]
            ).to(
                device
            )

            context_t = torch.from_numpy(
                tr[
                    "context_z"
                ][
                    start:stop
                ]
            ).to(
                device
            )

            ridge_t = torch.from_numpy(
                ridge_oof[
                    start:stop
                ]
            ).to(
                device
            )

            y_t = torch.from_numpy(
                ytr[
                    start:stop
                ]
            ).to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with amp_context(
                torch,
                amp_enabled,
                device,
            ):
                delta, _ = model(
                    seq_t,
                    summary_t,
                    context_t,
                )

                final_t = rotate_uv_torch(
                    torch,
                    ridge_t,
                    delta,
                )

                loss = direction_loss(
                    torch,
                    final_t,
                    y_t,
                    delta,
                    y_std_t,
                    angle_cap_deg,
                )

            if not torch.isfinite(
                loss
            ):
                raise FloatingPointError(
                    "Nonfinite direction loss."
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

            bn = stop - start
            loss_sum += float(
                loss.detach()
                .float()
                .cpu()
                .item()
            ) * bn
            n_seen += bn

        final_va, delta_va, gate_va = predict(
            torch=torch,
            model=model,
            transformed=va,
            ridge_anchor=ridge_va,
            device=device,
            amp_enabled=amp_enabled,
        )

        metrics = evaluate_wind(
            yva,
            final_va,
        )

        score = score_direction_task(
            metrics,
            ridge_metrics,
        )

        true_angle_res = wrap_rad(
            vector_angle_rad(
                yva
            )
            - vector_angle_rad(
                ridge_va
            )
        )

        angle_corr = corrcoef_safe(
            true_angle_res,
            delta_va,
        )

        improved = (
            score
            < best_score
            - 1e-8
        )

        if improved:
            best_score = float(score)
            best_state = state_dict_cpu(
                model
            )
            best_epoch = int(epoch)
            patience = 0
        else:
            patience += 1

        scheduler.step(score)

        if (
            epoch <= 5
            or epoch % 10 == 0
            or improved
        ):
            log(
                f"      ep={epoch:03d} | "
                f"score={score:.5f} | "
                f"U={metrics['wind_U_RMSE_mps']:.4f} | "
                f"V={metrics['wind_V_RMSE_mps']:.4f} | "
                f"WS={metrics['wind_speed_RMSE_mps']:.4f} | "
                f"WD={metrics['wind_direction_RMSE_deg']:.3f} | "
                f"angleCorr={angle_corr:+.3f}"
                + (
                    " *"
                    if improved
                    else ""
                )
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": (
                    loss_sum
                    / max(
                        n_seen,
                        1,
                    )
                ),
                "selection_score": score,
                "angle_residual_corr": angle_corr,
                "gate_mean": float(
                    np.mean(
                        gate_va
                    )
                ),
                "angle_correction_RMS_deg": float(
                    np.degrees(
                        np.sqrt(
                            np.mean(
                                delta_va ** 2
                            )
                        )
                    )
                ),
                **metrics,
            }
        )

        if patience >= patience_limit:
            break

    model.load_state_dict(
        best_state,
        strict=True,
    )

    final_va, delta_va, gate_va = predict(
        torch=torch,
        model=model,
        transformed=va,
        ridge_anchor=ridge_va,
        device=device,
        amp_enabled=amp_enabled,
    )

    metrics = evaluate_wind(
        yva,
        final_va,
    )

    score = score_direction_task(
        metrics,
        ridge_metrics,
    )

    true_angle_res = wrap_rad(
        vector_angle_rad(yva)
        - vector_angle_rad(ridge_va)
    )

    angle_corr = corrcoef_safe(
        true_angle_res,
        delta_va,
    )

    ws_identity_error = float(
        np.max(
            np.abs(
                np.linalg.norm(
                    final_va,
                    axis=1,
                )
                - np.linalg.norm(
                    ridge_va,
                    axis=1,
                )
            )
        )
    )

    hist_dir = (
        output_dir
        / "histories"
    )

    hist_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    hist_path = (
        hist_dir
        / (
            f"{candidate}"
            f"__{held_out.replace(' ', '_')}"
            f"__seed{seed}.csv"
        )
    )

    pd.DataFrame(
        history
    ).to_csv(
        hist_path,
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "candidate": candidate,
        "angle_cap_deg": float(
            angle_cap_deg
        ),
        "held_out_mission": held_out,
        "seed": int(seed),
        "status": "completed_finite",
        "selection_score": float(score),
        "best_epoch": int(
            best_epoch
        ),
        "parameter_count": int(
            nn_params
            + ridge_parameter_count(
                ridge_model
            )
        ),
        "nn_parameter_count": int(
            nn_params
        ),
        "ridge_parameter_count": int(
            ridge_parameter_count(
                ridge_model
            )
        ),
        "angle_residual_corr": float(
            angle_corr
        ),
        "gate_mean": float(
            np.mean(
                gate_va
            )
        ),
        "angle_correction_RMS_deg": float(
            np.degrees(
                np.sqrt(
                    np.mean(
                        delta_va ** 2
                    )
                )
            )
        ),
        "WS_identity_max_abs_mps": (
            ws_identity_error
        ),
        **metrics,
    }


# =============================================================================
# Final training
# =============================================================================

def train_final(
    *,
    torch,
    nn,
    cap_deg,
    seed,
    final_epochs,
    dev,
    tropical,
    output_dir,
    device,
):
    Xdev = build_base4(
        dev[
            "X_full9"
        ]
    )

    Xtrop = build_base4(
        tropical[
            "X_full9"
        ]
    )

    ydev = dev[
        "y_uv"
    ]

    ytrop = tropical[
        "y_uv"
    ]

    ridge_model, ridge_scaler = fit_ridge(
        Xdev,
        ydev,
    )

    ridge_trop = ridge_predict(
        ridge_model,
        ridge_scaler,
        Xtrop,
    )

    ridge_metrics = evaluate_wind(
        ytrop,
        ridge_trop,
    )

    ridge_oof = crossfit_ridge(
        Xdev,
        ydev,
        dev[
            "mission_labels"
        ],
    )

    feat_dev = build_direction_features(
        dev[
            "X_full9"
        ],
        ridge_oof,
    )

    feat_trop = build_direction_features(
        tropical[
            "X_full9"
        ],
        ridge_trop,
    )

    scalers = fit_feature_scalers(
        feat_dev,
        ydev,
    )

    tr = transform_features(
        feat_dev,
        scalers,
    )

    te = transform_features(
        feat_trop,
        scalers,
    )

    seed_everything(
        torch,
        seed,
    )

    ModelClass = build_model_class(
        torch,
        nn,
        cap_deg,
    )

    model = ModelClass().to(
        device
    )

    nn_params = count_parameters(
        model
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FREEZE.learning_rate,
        weight_decay=FREEZE.weight_decay,
    )

    amp_enabled = bool(
        FREEZE.use_amp
        and device.type == "cuda"
    )

    grad_scaler = create_grad_scaler(
        torch,
        amp_enabled,
    )

    y_std_t = torch.as_tensor(
        scalers["y_std"],
        dtype=torch.float32,
        device=device,
    )

    n = len(ydev)

    log(
        f"[FINAL TRAIN] angle cap={cap_deg:.1f} deg | "
        f"seed={seed} | epochs={final_epochs}"
    )

    for epoch in range(
        1,
        final_epochs + 1,
    ):
        model.train()

        for start in range(
            0,
            n,
            FREEZE.batch_size,
        ):
            stop = min(
                start + FREEZE.batch_size,
                n,
            )

            seq_t = torch.from_numpy(
                tr["seq_z"][
                    start:stop
                ]
            ).to(device)

            summary_t = torch.from_numpy(
                tr["summary_z"][
                    start:stop
                ]
            ).to(device)

            context_t = torch.from_numpy(
                tr["context_z"][
                    start:stop
                ]
            ).to(device)

            ridge_t = torch.from_numpy(
                ridge_oof[
                    start:stop
                ]
            ).to(device)

            y_t = torch.from_numpy(
                ydev[
                    start:stop
                ]
            ).to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            with amp_context(
                torch,
                amp_enabled,
                device,
            ):
                delta, _ = model(
                    seq_t,
                    summary_t,
                    context_t,
                )

                final_t = rotate_uv_torch(
                    torch,
                    ridge_t,
                    delta,
                )

                loss = direction_loss(
                    torch,
                    final_t,
                    y_t,
                    delta,
                    y_std_t,
                    cap_deg,
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

    final_trop, delta_trop, gate_trop = predict(
        torch=torch,
        model=model,
        transformed=te,
        ridge_anchor=ridge_trop,
        device=device,
        amp_enabled=amp_enabled,
    )

    final_metrics = evaluate_wind(
        ytrop,
        final_trop,
    )

    true_angle_res = wrap_rad(
        vector_angle_rad(
            ytrop
        )
        - vector_angle_rad(
            ridge_trop
        )
    )

    angle_corr = corrcoef_safe(
        true_angle_res,
        delta_trop,
    )

    ws_identity_error = float(
        np.max(
            np.abs(
                np.linalg.norm(
                    final_trop,
                    axis=1,
                )
                - np.linalg.norm(
                    ridge_trop,
                    axis=1,
                )
            )
        )
    )

    checkpoint_dir = (
        output_dir
        / "final_model"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_dir
        / "12Y_final.pt"
    )

    torch.save(
        {
            "stage": "12Y-final",
            "cap_deg": float(
                cap_deg
            ),
            "seed": int(seed),
            "epochs": int(
                final_epochs
            ),
            "state_dict": (
                state_dict_cpu(
                    model
                )
            ),
            "scalers": scalers,
            "freeze": asdict(
                FREEZE
            ),
        },
        checkpoint_path,
    )

    return {
        "ridge_metrics": (
            ridge_metrics
        ),
        "final_metrics": (
            final_metrics
        ),
        "angle_residual_corr": float(
            angle_corr
        ),
        "angle_correction_RMS_deg": float(
            np.degrees(
                np.sqrt(
                    np.mean(
                        delta_trop ** 2
                    )
                )
            )
        ),
        "gate_mean": float(
            np.mean(
                gate_trop
            )
        ),
        "WS_identity_max_abs_mps": (
            ws_identity_error
        ),
        "nn_parameter_count": int(
            nn_params
        ),
        "ridge_parameter_count": int(
            ridge_parameter_count(
                ridge_model
            )
        ),
        "effective_parameter_count": int(
            nn_params
            + ridge_parameter_count(
                ridge_model
            )
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
    configure_cuda(torch)

    device = torch.device(
        "cpu"
        if (
            args.force_cpu
            or not torch.cuda.is_available()
        )
        else "cuda"
    )

    log("=" * 120)
    log(
        "12Y — RIDGE-ANCHORED DIRECTION-ONLY ROTATION"
    )
    log("=" * 120)
    log(
        "Guarantee: predicted WS is identical to Base4-Ridge WS."
    )
    log(
        "Development: Antarctic / Atlantic / West Coast only."
    )

    dev_missions = {
        m: load_mission(
            dataset_root,
            m,
        )
        for m in DEV_MISSIONS
    }

    log(
        "[FIREWALL] Tropical Atlantic NOT loaded."
    )

    active_holdouts = (
        ["Antarctic"]
        if args.debug_fast
        else DEV_MISSIONS
    )

    active_caps = (
        [2.0]
        if args.debug_fast
        else ANGLE_CAPS_DEG
    )

    active_seeds = (
        [SEEDS[0]]
        if args.debug_fast
        else SEEDS
    )

    run_path = (
        output_dir
        / "development_runs.csv"
    )

    for held_out in active_holdouts:
        train = concat_missions(
            [
                dev_missions[m]
                for m in DEV_MISSIONS
                if m != held_out
            ]
        )

        val = dev_missions[
            held_out
        ]

        log(
            f"\n[HOLDOUT] {held_out}"
        )

        for cap in active_caps:
            for seed in active_seeds:
                result = train_dev_run(
                    torch=torch,
                    nn=nn,
                    angle_cap_deg=cap,
                    seed=seed,
                    held_out=held_out,
                    train_data=train,
                    val_data=val,
                    output_dir=output_dir,
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
                    f"  DONE cap={cap:.1f} | "
                    f"seed={seed} | "
                    f"score={result['selection_score']:.5f} | "
                    f"V={result['wind_V_RMSE_mps']:.4f} | "
                    f"WD={result['wind_direction_RMSE_deg']:.3f} | "
                    f"angleCorr={result['angle_residual_corr']:+.3f}"
                )

    if args.debug_fast:
        log(
            "\n[DONE] debug-fast. Tropical not loaded."
        )
        return 0

    runs = pd.read_csv(
        run_path
    )

    summary_rows = []

    for cap in ANGLE_CAPS_DEG:
        name = (
            f"AngleCap{int(cap)}deg"
        )

        sub = runs.loc[
            runs[
                "candidate"
            ]
            == name
        ]

        summary_rows.append(
            {
                "candidate": name,
                "angle_cap_deg": cap,
                "runs": len(sub),
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
                "U_RMSE_mean": float(
                    sub[
                        "wind_U_RMSE_mps"
                    ].mean()
                ),
                "V_RMSE_mean": float(
                    sub[
                        "wind_V_RMSE_mps"
                    ].mean()
                ),
                "WS_RMSE_mean": float(
                    sub[
                        "wind_speed_RMSE_mps"
                    ].mean()
                ),
                "WD_RMSE_mean": float(
                    sub[
                        "wind_direction_RMSE_deg"
                    ].mean()
                ),
                "angle_residual_corr_mean": float(
                    sub[
                        "angle_residual_corr"
                    ].mean()
                ),
                "best_epoch_median": float(
                    np.median(
                        sub[
                            "best_epoch"
                        ]
                    )
                ),
                "parameter_count_mean": float(
                    sub[
                        "parameter_count"
                    ].mean()
                ),
                "mission_wins": int(
                    np.sum(
                        sub.groupby(
                            "held_out_mission"
                        )[
                            "selection_score"
                        ]
                        .mean()
                        .to_numpy()
                        < 1.0
                    )
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "selection_score_mean",
            "parameter_count_mean",
        ]
    ).reset_index(
        drop=True
    )

    summary_path = (
        output_dir
        / "development_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    selected_cap = float(
        summary.iloc[
            0
        ][
            "angle_cap_deg"
        ]
    )

    selected_name = (
        f"AngleCap{int(selected_cap)}deg"
    )

    selected_runs = runs.loc[
        runs[
            "candidate"
        ]
        == selected_name
    ]

    seed_summary = (
        selected_runs.groupby(
            "seed",
            as_index=False,
        )[
            "selection_score"
        ]
        .mean()
        .sort_values(
            "selection_score"
        )
    )

    selected_seed = int(
        seed_summary.iloc[
            0
        ][
            "seed"
        ]
    )

    final_epochs = int(
        max(
            1,
            round(
                float(
                    np.median(
                        selected_runs.loc[
                            selected_runs[
                                "seed"
                            ]
                            == selected_seed,
                            "best_epoch",
                        ]
                    )
                )
            ),
        )
    )

    frozen = {
        "selected_angle_cap_deg": (
            selected_cap
        ),
        "selected_seed": (
            selected_seed
        ),
        "final_epochs": (
            final_epochs
        ),
        "selection_source": (
            "Antarctic/Atlantic/West Coast LOMO only"
        ),
        "Tropical_used_for_selection": False,
        "freeze": asdict(
            FREEZE
        ),
    }

    frozen_path = (
        output_dir
        / "FROZEN_BEFORE_TROPICAL.json"
    )

    save_json(
        frozen_path,
        frozen,
    )

    log("\n" + "=" * 120)
    log(
        "12Y DEVELOPMENT SUMMARY"
    )
    log("=" * 120)
    log(
        summary.to_string(
            index=False
        )
    )

    log(
        f"\n[FROZEN BEFORE TROPICAL] "
        f"cap={selected_cap:.1f} deg | "
        f"seed={selected_seed} | "
        f"epochs={final_epochs}"
    )

    log(
        "\n[STAGE C] Configuration frozen. "
        "Loading Tropical Atlantic NOW."
    )

    tropical = load_mission(
        dataset_root,
        FINAL_MISSION,
    )

    log(
        f"[TROPICAL DATASET] "
        f"{mission_path(dataset_root, FINAL_MISSION)}"
    )

    dev_all = concat_missions(
        [
            dev_missions[m]
            for m in DEV_MISSIONS
        ]
    )

    final = train_final(
        torch=torch,
        nn=nn,
        cap_deg=selected_cap,
        seed=selected_seed,
        final_epochs=final_epochs,
        dev=dev_all,
        tropical=tropical,
        output_dir=output_dir,
        device=device,
    )

    ridge = final[
        "ridge_metrics"
    ]

    ours = final[
        "final_metrics"
    ]

    rows = [
        {
            "model": "BP-STGNN-published",
            "U_RMSE_mps": BP_REFERENCE[
                "U_RMSE_mps"
            ],
            "V_RMSE_mps": BP_REFERENCE[
                "V_RMSE_mps"
            ],
            "WS_RMSE_mps": BP_REFERENCE[
                "WS_RMSE_mps"
            ],
            "WD_RMSE_deg": BP_REFERENCE[
                "WD_RMSE_deg"
            ],
            "parameter_count": BP_REFERENCE[
                "params"
            ],
        },
        {
            "model": "Base4-Ridge-final",
            "U_RMSE_mps": ridge[
                "wind_U_RMSE_mps"
            ],
            "V_RMSE_mps": ridge[
                "wind_V_RMSE_mps"
            ],
            "WS_RMSE_mps": ridge[
                "wind_speed_RMSE_mps"
            ],
            "WD_RMSE_deg": ridge[
                "wind_direction_RMSE_deg"
            ],
            "parameter_count": final[
                "ridge_parameter_count"
            ],
        },
        {
            "model": (
                "Ridge+DirectionOnly-final"
            ),
            "U_RMSE_mps": ours[
                "wind_U_RMSE_mps"
            ],
            "V_RMSE_mps": ours[
                "wind_V_RMSE_mps"
            ],
            "WS_RMSE_mps": ours[
                "wind_speed_RMSE_mps"
            ],
            "WD_RMSE_deg": ours[
                "wind_direction_RMSE_deg"
            ],
            "parameter_count": final[
                "effective_parameter_count"
            ],
        },
    ]

    final_df = pd.DataFrame(
        rows
    )

    final_csv = (
        output_dir
        / "FINAL_TROPICAL_BENCHMARK.csv"
    )

    final_df.to_csv(
        final_csv,
        index=False,
        encoding="utf-8-sig",
    )

    beats = {
        "U": ours[
            "wind_U_RMSE_mps"
        ] < BP_REFERENCE[
            "U_RMSE_mps"
        ],
        "V": ours[
            "wind_V_RMSE_mps"
        ] < BP_REFERENCE[
            "V_RMSE_mps"
        ],
        "WS": ours[
            "wind_speed_RMSE_mps"
        ] < BP_REFERENCE[
            "WS_RMSE_mps"
        ],
        "WD": ours[
            "wind_direction_RMSE_deg"
        ] < BP_REFERENCE[
            "WD_RMSE_deg"
        ],
    }

    report = {
        "frozen": frozen,
        "ridge": ridge,
        "ours": ours,
        "angle_residual_corr": final[
            "angle_residual_corr"
        ],
        "angle_correction_RMS_deg": final[
            "angle_correction_RMS_deg"
        ],
        "gate_mean": final[
            "gate_mean"
        ],
        "WS_identity_max_abs_mps": final[
            "WS_identity_max_abs_mps"
        ],
        "beats_BP": beats,
        "effective_parameter_count": final[
            "effective_parameter_count"
        ],
        "parameter_fraction_vs_BP": (
            final[
                "effective_parameter_count"
            ]
            / BP_REFERENCE[
                "params"
            ]
        ),
    }

    save_json(
        output_dir
        / "FINAL_TROPICAL_REPORT.json",
        report,
    )

    log("\n" + "=" * 120)
    log(
        "FINAL TROPICAL ATLANTIC BENCHMARK"
    )
    log("=" * 120)
    log(
        final_df.to_string(
            index=False
        )
    )

    log(
        f"\nAngle residual corr = "
        f"{final['angle_residual_corr']:+.4f}"
    )

    log(
        f"Angle correction RMS = "
        f"{final['angle_correction_RMS_deg']:.4f} deg"
    )

    log(
        f"Gate mean = "
        f"{final['gate_mean']:.4f}"
    )

    log(
        f"WS identity max error = "
        f"{final['WS_identity_max_abs_mps']:.3e} m/s"
    )

    log(
        "\nBeats BP-STGNN:"
    )

    for k, v in beats.items():
        log(
            f"  {k}: {bool(v)}"
        )

    log(
        f"Params ours/BP = "
        f"{final['effective_parameter_count']}/"
        f"{BP_REFERENCE['params']} "
        f"({100*report['parameter_fraction_vs_BP']:.2f}%)"
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
