# -*- coding: utf-8 -*-
r"""
12W_train_BPAligned_RotationEquivariant_LocalWindFrame_LOMO.py

Stage 12W
=========
BP-aligned point-sampled true-wind forecasting using a rotation-equivariant
LOCAL WIND FRAME.

Why Stage 12W?
--------------
Stage 12V showed:
    - raw Boat7 / reconstructed Trajectory8 did not materially improve Base4;
    - first-order advection geometry worsened generalization;
    - generic U/V residual learning remained weak;
    - radial wind-speed residual showed only weak correlation and did not beat
      Base4 overall.

Therefore Stage 12W stops escalating the trajectory branch in the strict BP
benchmark and tests a different physical representation.

Core idea
---------
Global East/North coordinates make the same physical wind evolution look
different when the prevailing wind direction differs between missions.

For every forecasting sample, define an orthonormal local reference frame from
the latest reliable historical wind vector:

    e_parallel = unit(latest reliable W)
    e_cross    = [-e_parallel_N, e_parallel_E]

Every historical wind vector is rotated into this SAME frame:

    W_parallel,k = W_k dot e_parallel
    W_cross,k    = W_k dot e_cross

The future target is also represented in this local frame:

    y_parallel = W_future dot e_parallel
    y_cross    = W_future dot e_cross

The predicted local vector is reconstructed exactly back to global U,V.

This is NOT independent speed/direction prediction:
    - the model predicts a 2-D vector;
    - vector consistency is exact;
    - WS and WD are always derived from the reconstructed U,V.

FROZEN BP-ALIGNED TASK
----------------------
Source:
    Stage 12G point-sampled benchmark.

History:
    six original 1-min-mean records sampled every 10 min:
        t-50, t-40, t-30, t-20, t-10, t

Target:
    original 1-min-mean U,V record at t+10.

Fairness:
    NO 10-min arithmetic means.
    NO intermediate 1-min observations.
    NO pressure.
    NO wing angle.
    NO absolute position.
    Tropical Atlantic is NEVER loaded in development.

Feature groups
--------------
Global4:
    [U, V, T, RH]

Rot4:
    [W_parallel, W_cross, T, RH]

RotBoat6:
    Rot4 + [boat_parallel, boat_cross]

RotDyn8:
    RotBoat6 + [dW_parallel, dW_cross]

All added features are deterministic transforms of:
    U,V,T,RH,SOG,COG
at the same six BP-aligned samples.

Candidates
----------
Linear representation ablations:
    Global4-Ridge
    Rot4-Ridge
    RotBoat6-Ridge
    RotDyn8-Ridge

Direct compact nonlinear models:
    RotDyn8-MLP
    RotDyn8-GRU
    RotDyn8-TCN
    RotDyn8-GRU-Coupled

Local-frame safe residual:
    RotDyn8-RidgeResidualMLP

The residual candidate is retained as ONE diagnostic because local-frame
residuals may be more stationary than East/North residuals.  Its final head is
zero-initialized, therefore epoch 0 = exact RotDyn8-Ridge.

Coupled GRU loss
----------------
The direct GRU-Coupled predicts the same 2-D local vector but trains with:

    L = L_component
        + 0.15 * L_speed
        + 0.05 * L_direction

where direction uses cosine loss on the predicted and true local vectors.
This keeps one vector output and only adds auxiliary geometry-aware losses.

Near-calm reference direction
-----------------------------
If the latest historical wind magnitude is below 0.5 m/s, Stage 12W searches
backward for the most recent historical sample >= 0.5 m/s.  If the entire
six-point history is below threshold, East is used as a deterministic fallback.

LOMO development
----------------
Holdout Antarctic  <- Atlantic + West Coast
Holdout Atlantic   <- Antarctic + West Coast
Holdout West Coast <- Antarctic + Atlantic

Selection score vs Global4-Ridge:
    0.25 * U ratio
  + 0.35 * V ratio
  + 0.25 * WS ratio
  + 0.15 * WD ratio

Meaningful improvement:
    mean score < 0.99
    wins >= 2/3 missions
    vector RMSE improves >0.5%
    and U or V improves >1%

Published BP-STGNN reference (context only during development):
    U  = 0.748640 m/s
    V  = 0.705060 m/s
    WS = 0.699400 m/s
    WD = 5.584 deg
    params = 242580

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\12W_train_BPAligned_RotationEquivariant_LocalWindFrame_LOMO.py" --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12W_BPAligned_LocalWindFrame_LOMO_v0_1"

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


SCRIPT_VERSION = "0.1.0-12W-BPAligned-LocalWindFrame"

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
    / "12W_BPAligned_LocalWindFrame_LOMO_v0_1"
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
TEN_MIN_NS = (
    STEP_MINUTES
    * 60
    * 1_000_000_000
)

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

REFERENCE_WIND_MIN_MPS = 0.5
EPS = 1e-12
RIDGE_ALPHA = 1.0
INNER_CROSSFIT_FOLDS = 5

SCORE_WEIGHTS = {
    "U": 0.25,
    "V": 0.35,
    "WS": 0.25,
    "WD": 0.15,
}

CONTINUE_SCORE_THRESHOLD = 0.99
MIN_MISSION_WINS = 2
MIN_VECTOR_IMPROVEMENT = 0.005
MIN_U_OR_V_IMPROVEMENT = 0.01

BP_STGNN_REFERENCE = {
    "U_RMSE_mps": 0.748640,
    "V_RMSE_mps": 0.705060,
    "WS_RMSE_mps": 0.699400,
    "WD_RMSE_deg": 5.584000,
    "params": 242580,
}

FEATURE_GROUPS = {
    "Global4": [
        "U",
        "V",
        "T",
        "RH",
    ],
    "Rot4": [
        "W_parallel",
        "W_cross",
        "T",
        "RH",
    ],
    "RotBoat6": [
        "W_parallel",
        "W_cross",
        "T",
        "RH",
        "boat_parallel",
        "boat_cross",
    ],
    "RotDyn8": [
        "W_parallel",
        "W_cross",
        "T",
        "RH",
        "boat_parallel",
        "boat_cross",
        "dW_parallel",
        "dW_cross",
    ],
}


@dataclass(frozen=True)
class NeuralFreeze:
    mlp_hidden_1: int = 64
    mlp_hidden_2: int = 32

    gru_hidden: int = 48
    dense_hidden: int = 48

    tcn_channels: int = 48
    tcn_kernel_size: int = 3

    dropout: float = 0.10
    learning_rate: float = 8e-4
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 100
    early_stop_patience: int = 15
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-5
    grad_clip_norm: float = 1.0
    correction_penalty: float = 1e-4

    coupled_speed_weight: float = 0.15
    coupled_direction_weight: float = 0.05

    use_amp: bool = True


NEURAL_FREEZE = NeuralFreeze()


# =============================================================================
# General
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
            return {
                str(k): cv(x)
                for k, x in v.items()
            }
        if isinstance(v, (list, tuple)):
            return [
                cv(x)
                for x in v
            ]
        return v

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
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


# =============================================================================
# Torch
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

    for start in range(
        0,
        n,
        int(batch_size),
    ):
        stop = min(
            start + int(batch_size),
            n,
        )

        yield tuple(
            a[start:stop]
            for a in arrays
        )


# =============================================================================
# Data loading
# =============================================================================

def resolve_dataset_root(dataset_dir: Path):
    candidates = [
        dataset_dir,
        dataset_dir / "dataset",
    ]

    for root in candidates:
        if (
            root
            / "track_J_joint_compatible"
        ).exists():
            return root

    raise FileNotFoundError(
        "Could not find track_J_joint_compatible under "
        f"{dataset_dir} or {dataset_dir / 'dataset'}"
    )


def load_track_j(dataset_root: Path, mission: str):
    path = (
        dataset_root
        / "track_J_joint_compatible"
        / f"{mission.replace(' ', '_')}.npz"
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
            f"{mission}: unexpected Track-J feature names: {names}"
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
            f"{mission}: lookback={lookback}"
        )

    if step_minutes != STEP_MINUTES:
        raise RuntimeError(
            f"{mission}: step_minutes={step_minutes}"
        )

    if not np.all(
        target - context
        == TEN_MIN_NS
    ):
        raise RuntimeError(
            f"{mission}: target not exactly +10 min."
        )

    if not (
        np.isfinite(X).all()
        and np.isfinite(y).all()
    ):
        raise RuntimeError(
            f"{mission}: nonfinite data."
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
            ],
            axis=0,
        ),
    }


# =============================================================================
# Local wind frame
# =============================================================================

def build_reference_basis(X_full9):
    """
    Return orthonormal reference frame [e_parallel, e_cross] per sample.

    Reference = latest historical wind with speed >= threshold.
    If all six are near-calm, deterministic East direction is used.
    """
    X = np.asarray(
        X_full9,
        dtype=np.float64,
    )

    U = X[:, :, IDX_U]
    V = X[:, :, IDX_V]

    speed = np.hypot(U, V)

    n = len(X)

    e_parallel = np.zeros(
        (n, 2),
        dtype=np.float64,
    )

    fallback_all_calm = np.ones(
        n,
        dtype=bool,
    )

    chosen_index = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    for k in range(
        LOOKBACK_STEPS - 1,
        -1,
        -1,
    ):
        choose = (
            fallback_all_calm
            & (
                speed[:, k]
                >= REFERENCE_WIND_MIN_MPS
            )
        )

        if np.any(choose):
            s = speed[choose, k]

            e_parallel[choose, 0] = (
                U[choose, k] / s
            )

            e_parallel[choose, 1] = (
                V[choose, k] / s
            )

            chosen_index[choose] = k
            fallback_all_calm[choose] = False

    if np.any(fallback_all_calm):
        e_parallel[
            fallback_all_calm,
            0,
        ] = 1.0

        e_parallel[
            fallback_all_calm,
            1,
        ] = 0.0

    e_cross = np.stack(
        [
            -e_parallel[:, 1],
            e_parallel[:, 0],
        ],
        axis=1,
    )

    return (
        e_parallel.astype(np.float32),
        e_cross.astype(np.float32),
        {
            "all_calm_fallback_fraction": float(
                np.mean(
                    fallback_all_calm
                )
            ),
            "reference_latest_fraction": float(
                np.mean(
                    chosen_index
                    == (
                        LOOKBACK_STEPS - 1
                    )
                )
            ),
            "reference_mean_history_index": float(
                np.mean(
                    np.where(
                        chosen_index >= 0,
                        chosen_index,
                        np.nan,
                    )
                )
            ),
        },
    )


def rotate_vectors_to_local(
    vectors,
    e_parallel,
    e_cross,
):
    v = np.asarray(
        vectors,
        dtype=np.float64,
    )

    ep = np.asarray(
        e_parallel,
        dtype=np.float64,
    )

    ec = np.asarray(
        e_cross,
        dtype=np.float64,
    )

    if v.ndim == 3:
        parallel = (
            v[:, :, 0]
            * ep[:, None, 0]
            + v[:, :, 1]
            * ep[:, None, 1]
        )

        cross = (
            v[:, :, 0]
            * ec[:, None, 0]
            + v[:, :, 1]
            * ec[:, None, 1]
        )

        return np.stack(
            [parallel, cross],
            axis=-1,
        ).astype(np.float32)

    if v.ndim == 2:
        parallel = (
            v[:, 0] * ep[:, 0]
            + v[:, 1] * ep[:, 1]
        )

        cross = (
            v[:, 0] * ec[:, 0]
            + v[:, 1] * ec[:, 1]
        )

        return np.stack(
            [parallel, cross],
            axis=-1,
        ).astype(np.float32)

    raise ValueError(
        f"Unexpected vector shape: {v.shape}"
    )


def reconstruct_local_to_global(
    local_vectors,
    e_parallel,
    e_cross,
):
    local = np.asarray(
        local_vectors,
        dtype=np.float64,
    )

    ep = np.asarray(
        e_parallel,
        dtype=np.float64,
    )

    ec = np.asarray(
        e_cross,
        dtype=np.float64,
    )

    global_uv = (
        local[:, 0:1]
        * ep
        + local[:, 1:2]
        * ec
    )

    return global_uv.astype(
        np.float32
    )


def build_local_representation(
    X_full9,
    y_uv,
):
    X = np.asarray(
        X_full9,
        dtype=np.float64,
    )

    y = np.asarray(
        y_uv,
        dtype=np.float64,
    )

    e_parallel, e_cross, audit = (
        build_reference_basis(X)
    )

    wind_history = X[
        :,
        :,
        [
            IDX_U,
            IDX_V,
        ]
    ]

    wind_local = rotate_vectors_to_local(
        wind_history,
        e_parallel,
        e_cross,
    )

    target_local = rotate_vectors_to_local(
        y,
        e_parallel,
        e_cross,
    )

    T = X[:, :, IDX_T].astype(
        np.float32
    )

    RH = X[:, :, IDX_RH].astype(
        np.float32
    )

    SOG = X[:, :, IDX_SOG]

    boat_E = (
        SOG
        * X[:, :, IDX_COG_SIN]
    )

    boat_N = (
        SOG
        * X[:, :, IDX_COG_COS]
    )

    boat_local = rotate_vectors_to_local(
        np.stack(
            [boat_E, boat_N],
            axis=-1,
        ),
        e_parallel,
        e_cross,
    )

    d_local = np.zeros_like(
        wind_local,
        dtype=np.float32,
    )

    d_local[:, 1:, :] = (
        wind_local[:, 1:, :]
        - wind_local[:, :-1, :]
    )

    Rot4 = np.stack(
        [
            wind_local[:, :, 0],
            wind_local[:, :, 1],
            T,
            RH,
        ],
        axis=-1,
    ).astype(np.float32)

    RotBoat6 = np.stack(
        [
            wind_local[:, :, 0],
            wind_local[:, :, 1],
            T,
            RH,
            boat_local[:, :, 0],
            boat_local[:, :, 1],
        ],
        axis=-1,
    ).astype(np.float32)

    RotDyn8 = np.stack(
        [
            wind_local[:, :, 0],
            wind_local[:, :, 1],
            T,
            RH,
            boat_local[:, :, 0],
            boat_local[:, :, 1],
            d_local[:, :, 0],
            d_local[:, :, 1],
        ],
        axis=-1,
    ).astype(np.float32)

    Global4 = X[
        :,
        :,
        [
            IDX_U,
            IDX_V,
            IDX_T,
            IDX_RH,
        ]
    ].astype(np.float32)

    for arr in [
        Global4,
        Rot4,
        RotBoat6,
        RotDyn8,
        target_local,
    ]:
        if not np.isfinite(arr).all():
            raise RuntimeError(
                "Nonfinite local-frame representation."
            )

    return {
        "Global4": Global4,
        "Rot4": Rot4,
        "RotBoat6": RotBoat6,
        "RotDyn8": RotDyn8,
        "target_global": y.astype(np.float32),
        "target_local": target_local,
        "e_parallel": e_parallel,
        "e_cross": e_cross,
        "audit": audit,
    }


# =============================================================================
# Scaling / Ridge
# =============================================================================

def fit_scaler(X, y):
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


def transform_X(X, scaler):
    return (
        (
            np.asarray(
                X,
                dtype=np.float32,
            )
            - scaler[
                "x_mean"
            ][None, None, :]
        )
        / scaler[
            "x_std"
        ][None, None, :]
    ).astype(np.float32)


def transform_y(y, scaler):
    return (
        (
            np.asarray(
                y,
                dtype=np.float32,
            )
            - scaler[
                "y_mean"
            ][None, :]
        )
        / scaler[
            "y_std"
        ][None, :]
    ).astype(np.float32)


def inverse_y(yz, scaler):
    return (
        np.asarray(
            yz,
            dtype=np.float32,
        )
        * scaler[
            "y_std"
        ][None, :]
        + scaler[
            "y_mean"
        ][None, :]
    ).astype(np.float32)


def fit_ridge(X_train, y_train):
    scaler = fit_scaler(
        X_train,
        y_train,
    )

    Xz = transform_X(
        X_train,
        scaler,
    ).reshape(
        len(X_train),
        -1,
    )

    yz = transform_y(
        y_train,
        scaler,
    )

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        Xz,
        yz,
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
    Xz = transform_X(
        X,
        scaler,
    ).reshape(
        len(X),
        -1,
    )

    pred_z = np.asarray(
        model.predict(Xz),
        dtype=np.float32,
    )

    return inverse_y(
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


# =============================================================================
# Metrics
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

    err = (
        yp - yt
    )

    ws_t = np.linalg.norm(
        yt,
        axis=1,
    )

    ws_p = np.linalg.norm(
        yp,
        axis=1,
    )

    wd_t = wd_from_uv(yt)
    wd_p = wd_from_uv(yp)

    wd_err = circular_diff_deg(
        wd_p,
        wd_t,
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


def local_component_metrics(
    y_true_local,
    y_pred_local,
):
    err = (
        np.asarray(
            y_pred_local,
            dtype=np.float64,
        )
        - np.asarray(
            y_true_local,
            dtype=np.float64,
        )
    )

    return {
        "local_parallel_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    err[:, 0] ** 2
                )
            )
        ),
        "local_cross_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    err[:, 1] ** 2
                )
            )
        ),
    }


def selection_score(
    metrics,
    base_metrics,
):
    ratios = {
        "U_ratio": (
            metrics[
                "wind_U_RMSE_mps"
            ]
            / max(
                base_metrics[
                    "wind_U_RMSE_mps"
                ],
                EPS,
            )
        ),
        "V_ratio": (
            metrics[
                "wind_V_RMSE_mps"
            ]
            / max(
                base_metrics[
                    "wind_V_RMSE_mps"
                ],
                EPS,
            )
        ),
        "WS_ratio": (
            metrics[
                "wind_speed_RMSE_mps"
            ]
            / max(
                base_metrics[
                    "wind_speed_RMSE_mps"
                ],
                EPS,
            )
        ),
        "WD_ratio": (
            metrics[
                "wind_direction_RMSE_deg"
            ]
            / max(
                base_metrics[
                    "wind_direction_RMSE_deg"
                ],
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

    return (
        float(score),
        ratios,
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


def residual_diagnostics(
    y_true_local,
    ridge_local,
    final_local,
):
    r_true = (
        np.asarray(
            y_true_local,
            dtype=np.float64,
        )
        - np.asarray(
            ridge_local,
            dtype=np.float64,
        )
    )

    r_pred = (
        np.asarray(
            final_local,
            dtype=np.float64,
        )
        - np.asarray(
            ridge_local,
            dtype=np.float64,
        )
    )

    return {
        "residual_corr_parallel": (
            corrcoef_safe(
                r_true[:, 0],
                r_pred[:, 0],
            )
        ),
        "residual_corr_cross": (
            corrcoef_safe(
                r_true[:, 1],
                r_pred[:, 1],
            )
        ),
        "residual_sign_accuracy_parallel": float(
            np.mean(
                np.sign(r_true[:, 0])
                == np.sign(r_pred[:, 0])
            )
        ),
        "residual_sign_accuracy_cross": float(
            np.mean(
                np.sign(r_true[:, 1])
                == np.sign(r_pred[:, 1])
            )
        ),
        "pred_correction_RMS_parallel_mps": float(
            np.sqrt(
                np.mean(
                    r_pred[:, 0] ** 2
                )
            )
        ),
        "pred_correction_RMS_cross_mps": float(
            np.sqrt(
                np.mean(
                    r_pred[:, 1] ** 2
                )
            )
        ),
    }


# =============================================================================
# Blocked OOF for local-frame Ridge
# =============================================================================

def blocked_fold_ids(
    mission_labels,
    n_folds=INNER_CROSSFIT_FOLDS,
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

    for mission in np.unique(labels):
        idx = np.flatnonzero(
            labels == mission
        )

        chunks = np.array_split(
            idx,
            n_folds,
        )

        for k, chunk in enumerate(chunks):
            fold_ids[chunk] = k

    if np.any(
        fold_ids < 0
    ):
        raise RuntimeError(
            "Blocked OOF fold assignment failed."
        )

    return fold_ids


def crossfit_ridge_local_predictions(
    X_train,
    y_train_local,
    mission_labels,
    n_folds=INNER_CROSSFIT_FOLDS,
):
    fold_ids = blocked_fold_ids(
        mission_labels,
        n_folds,
    )

    pred = np.full_like(
        y_train_local,
        np.nan,
        dtype=np.float32,
    )

    for k in range(n_folds):
        val_mask = (
            fold_ids == k
        )

        train_mask = (
            ~val_mask
        )

        model, scaler = fit_ridge(
            X_train[train_mask],
            y_train_local[train_mask],
        )

        pred[val_mask] = ridge_predict(
            model,
            scaler,
            X_train[val_mask],
        )

    if not np.isfinite(pred).all():
        raise RuntimeError(
            "Nonfinite local OOF Ridge predictions."
        )

    return pred


# =============================================================================
# Neural data
# =============================================================================

def prepare_direct_data(
    X_train,
    X_val,
    y_train_local,
):
    scaler = fit_scaler(
        X_train,
        y_train_local,
    )

    return {
        "X_train_z": transform_X(
            X_train,
            scaler,
        ),
        "X_val_z": transform_X(
            X_val,
            scaler,
        ),
        "y_train_z": transform_y(
            y_train_local,
            scaler,
        ),
        "scaler": scaler,
    }


def prepare_residual_data(
    X_train,
    X_val,
    y_train_local,
    ridge_oof_local,
    ridge_val_local,
):
    base = prepare_direct_data(
        X_train,
        X_val,
        y_train_local,
    )

    anchor_mean = np.mean(
        ridge_oof_local,
        axis=0,
    )

    anchor_std = np.std(
        ridge_oof_local,
        axis=0,
        ddof=0,
    )

    anchor_std = np.where(
        anchor_std < 1e-8,
        1.0,
        anchor_std,
    )

    A_tr = (
        (
            ridge_oof_local
            - anchor_mean[None, :]
        )
        / anchor_std[None, :]
    ).astype(np.float32)

    A_va = (
        (
            ridge_val_local
            - anchor_mean[None, :]
        )
        / anchor_std[None, :]
    ).astype(np.float32)

    residual = (
        y_train_local
        - ridge_oof_local
    ).astype(np.float32)

    residual_std = np.std(
        residual.astype(
            np.float64
        ),
        axis=0,
        ddof=0,
    )

    residual_std = np.where(
        residual_std < 1e-4,
        1.0,
        residual_std,
    ).astype(np.float32)

    target_norm = (
        residual
        / residual_std[None, :]
    ).astype(np.float32)

    base.update(
        {
            "anchor_train_z": A_tr,
            "anchor_val_z": A_va,
            "residual_target_norm": target_norm,
            "residual_std": residual_std,
        }
    )

    return base


# =============================================================================
# Neural models
# =============================================================================

def build_direct_model_class(
    torch,
    nn,
    mode,
):
    input_dim = 8

    if mode == "mlp":
        class DirectMLP(nn.Module):
            def __init__(self):
                super().__init__()

                self.net = nn.Sequential(
                    nn.Linear(
                        LOOKBACK_STEPS
                        * input_dim,
                        NEURAL_FREEZE.mlp_hidden_1,
                    ),
                    nn.LayerNorm(
                        NEURAL_FREEZE.mlp_hidden_1
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        NEURAL_FREEZE.dropout
                    ),
                    nn.Linear(
                        NEURAL_FREEZE.mlp_hidden_1,
                        NEURAL_FREEZE.mlp_hidden_2,
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        NEURAL_FREEZE.dropout
                    ),
                    nn.Linear(
                        NEURAL_FREEZE.mlp_hidden_2,
                        2,
                    ),
                )

            def forward(self, x):
                return self.net(
                    x.flatten(
                        start_dim=1
                    )
                )

        return DirectMLP

    if mode in [
        "gru",
        "gru_coupled",
    ]:
        class DirectGRU(nn.Module):
            def __init__(self):
                super().__init__()

                self.gru = nn.GRU(
                    input_size=input_dim,
                    hidden_size=(
                        NEURAL_FREEZE.gru_hidden
                    ),
                    num_layers=1,
                    batch_first=True,
                )

                self.head = nn.Sequential(
                    nn.Linear(
                        NEURAL_FREEZE.gru_hidden,
                        NEURAL_FREEZE.dense_hidden,
                    ),
                    nn.LayerNorm(
                        NEURAL_FREEZE.dense_hidden
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        NEURAL_FREEZE.dropout
                    ),
                    nn.Linear(
                        NEURAL_FREEZE.dense_hidden,
                        2,
                    ),
                )

            def forward(self, x):
                seq, _ = self.gru(x)

                return self.head(
                    seq[:, -1, :]
                )

        return DirectGRU

    if mode == "tcn":
        class DirectTCN(nn.Module):
            def __init__(self):
                super().__init__()

                c = (
                    NEURAL_FREEZE.tcn_channels
                )

                k = (
                    NEURAL_FREEZE.tcn_kernel_size
                )

                self.conv1 = nn.Conv1d(
                    input_dim,
                    c,
                    kernel_size=k,
                    padding=0,
                )

                self.conv2 = nn.Conv1d(
                    c,
                    c,
                    kernel_size=k,
                    padding=0,
                )

                self.act = nn.GELU()

                self.drop = nn.Dropout(
                    NEURAL_FREEZE.dropout
                )

                # Two valid k=3 convs reduce length 6 -> 4 -> 2.
                self.head = nn.Sequential(
                    nn.Linear(
                        c * 2,
                        NEURAL_FREEZE.dense_hidden,
                    ),
                    nn.LayerNorm(
                        NEURAL_FREEZE.dense_hidden
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        NEURAL_FREEZE.dropout
                    ),
                    nn.Linear(
                        NEURAL_FREEZE.dense_hidden,
                        2,
                    ),
                )

            def forward(self, x):
                z = x.transpose(1, 2)

                z = self.drop(
                    self.act(
                        self.conv1(z)
                    )
                )

                z = self.drop(
                    self.act(
                        self.conv2(z)
                    )
                )

                return self.head(
                    z.flatten(
                        start_dim=1
                    )
                )

        return DirectTCN

    raise ValueError(
        f"Unknown direct mode: {mode}"
    )


def build_residual_model_class(
    torch,
    nn,
):
    input_dim = 8

    class LocalResidualMLP(nn.Module):
        def __init__(self):
            super().__init__()

            self.backbone = nn.Sequential(
                nn.Linear(
                    LOOKBACK_STEPS
                    * input_dim
                    + 2,
                    NEURAL_FREEZE.mlp_hidden_1,
                ),
                nn.LayerNorm(
                    NEURAL_FREEZE.mlp_hidden_1
                ),
                nn.GELU(),
                nn.Dropout(
                    NEURAL_FREEZE.dropout
                ),
                nn.Linear(
                    NEURAL_FREEZE.mlp_hidden_1,
                    NEURAL_FREEZE.mlp_hidden_2,
                ),
                nn.GELU(),
                nn.Dropout(
                    NEURAL_FREEZE.dropout
                ),
            )

            self.out = nn.Linear(
                NEURAL_FREEZE.mlp_hidden_2,
                2,
            )

            # Safe anchor:
            # epoch 0 == exact RotDyn8-Ridge.
            nn.init.zeros_(
                self.out.weight
            )
            nn.init.zeros_(
                self.out.bias
            )

        def forward(
            self,
            x,
            anchor,
        ):
            z = torch.cat(
                [
                    x.flatten(
                        start_dim=1
                    ),
                    anchor,
                ],
                dim=1,
            )

            return self.out(
                self.backbone(z)
            )

    return LocalResidualMLP


# =============================================================================
# Neural evaluation
# =============================================================================

def direct_predict_local(
    *,
    torch,
    model,
    data,
    device,
    amp_enabled,
):
    model.eval()

    chunks = []

    with torch.no_grad():
        for (xb_np,) in sequential_batches(
            data["X_val_z"],
            batch_size=(
                NEURAL_FREEZE.batch_size
            ),
        ):
            xb = torch.from_numpy(
                xb_np
            ).to(
                device,
                non_blocking=True,
            )

            with amp_context(
                torch,
                amp_enabled,
                device,
            ):
                pred_z = model(xb)

            chunks.append(
                pred_z.detach()
                .float()
                .cpu()
                .numpy()
            )

    pred_z = np.concatenate(
        chunks,
        axis=0,
    )

    return inverse_y(
        pred_z,
        data["scaler"],
    )


def residual_predict_local(
    *,
    torch,
    model,
    data,
    ridge_val_local,
    device,
    amp_enabled,
):
    model.eval()

    chunks = []

    with torch.no_grad():
        for xb_np, ab_np in sequential_batches(
            data["X_val_z"],
            data["anchor_val_z"],
            batch_size=(
                NEURAL_FREEZE.batch_size
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

    correction = (
        pred_norm
        * data[
            "residual_std"
        ][None, :]
    )

    return (
        ridge_val_local
        + correction
    ).astype(np.float32)


# =============================================================================
# Direct training
# =============================================================================

def train_direct_run(
    *,
    torch,
    nn,
    candidate,
    mode,
    seed,
    held_out,
    data,
    y_val_local,
    y_val_global,
    e_parallel_val,
    e_cross_val,
    base_metrics,
    output_dir,
    device,
    debug_fast,
):
    seed_everything(
        torch,
        seed,
    )

    ModelClass = build_direct_model_class(
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
        lr=NEURAL_FREEZE.learning_rate,
        weight_decay=(
            NEURAL_FREEZE.weight_decay
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=(
                NEURAL_FREEZE.scheduler_factor
            ),
            patience=(
                NEURAL_FREEZE.scheduler_patience
            ),
            min_lr=(
                NEURAL_FREEZE.min_learning_rate
            ),
        )
    )

    amp_enabled = bool(
        NEURAL_FREEZE.use_amp
        and device.type == "cuda"
    )

    grad_scaler = create_grad_scaler(
        torch,
        amp_enabled,
    )

    best_score = np.inf
    best_state = None
    best_epoch = -1
    patience = 0
    history = []

    max_epochs = (
        8
        if debug_fast
        else NEURAL_FREEZE.max_epochs
    )

    patience_limit = (
        4
        if debug_fast
        else NEURAL_FREEZE.early_stop_patience
    )

    started = time.perf_counter()

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        model.train()

        total_sum = 0.0
        n_seen = 0

        for xb_np, yb_np in sequential_batches(
            data["X_train_z"],
            data["y_train_z"],
            batch_size=(
                NEURAL_FREEZE.batch_size
            ),
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

            optimizer.zero_grad(
                set_to_none=True
            )

            with amp_context(
                torch,
                amp_enabled,
                device,
            ):
                pred_z = model(xb)

                component_loss = torch.mean(
                    (
                        pred_z - yb
                    ) ** 2
                )

                if mode == "gru_coupled":
                    y_mean = torch.as_tensor(
                        data["scaler"]["y_mean"],
                        dtype=pred_z.dtype,
                        device=device,
                    )

                    y_std = torch.as_tensor(
                        data["scaler"]["y_std"],
                        dtype=pred_z.dtype,
                        device=device,
                    )

                    pred_phys = (
                        pred_z
                        * y_std[None, :]
                        + y_mean[None, :]
                    )

                    true_phys = (
                        yb
                        * y_std[None, :]
                        + y_mean[None, :]
                    )

                    pred_speed = torch.linalg.vector_norm(
                        pred_phys,
                        dim=1,
                    )

                    true_speed = torch.linalg.vector_norm(
                        true_phys,
                        dim=1,
                    )

                    speed_scale = torch.mean(
                        torch.abs(
                            true_speed
                        )
                    ).detach().clamp_min(
                        1.0
                    )

                    speed_loss = torch.mean(
                        (
                            (
                                pred_speed
                                - true_speed
                            )
                            / speed_scale
                        ) ** 2
                    )

                    denom = (
                        pred_speed
                        * true_speed
                    ).clamp_min(
                        1e-4
                    )

                    cos_sim = torch.sum(
                        pred_phys
                        * true_phys,
                        dim=1,
                    ) / denom

                    cos_sim = torch.clamp(
                        cos_sim,
                        -1.0,
                        1.0,
                    )

                    direction_mask = (
                        true_speed
                        >= REFERENCE_WIND_MIN_MPS
                    )

                    if torch.any(
                        direction_mask
                    ):
                        direction_loss = torch.mean(
                            1.0
                            - cos_sim[
                                direction_mask
                            ]
                        )
                    else:
                        direction_loss = (
                            component_loss
                            * 0.0
                        )

                    loss = (
                        component_loss
                        + NEURAL_FREEZE.coupled_speed_weight
                        * speed_loss
                        + NEURAL_FREEZE.coupled_direction_weight
                        * direction_loss
                    )
                else:
                    loss = component_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Nonfinite direct local-frame loss."
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
                    NEURAL_FREEZE.grad_clip_norm,
                )

                grad_scaler.step(
                    optimizer
                )

                grad_scaler.update()
            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    NEURAL_FREEZE.grad_clip_norm,
                )

                optimizer.step()

            bn = len(xb_np)

            total_sum += float(
                loss.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            n_seen += bn

        pred_local = direct_predict_local(
            torch=torch,
            model=model,
            data=data,
            device=device,
            amp_enabled=amp_enabled,
        )

        pred_global = reconstruct_local_to_global(
            pred_local,
            e_parallel_val,
            e_cross_val,
        )

        metrics = evaluate_wind(
            y_val_global,
            pred_global,
        )

        metrics.update(
            local_component_metrics(
                y_val_local,
                pred_local,
            )
        )

        score, _ = selection_score(
            metrics,
            base_metrics,
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

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": (
                    total_sum
                    / max(n_seen, 1)
                ),
                "selection_score": float(score),
                "checkpoint_improved": bool(
                    improved
                ),
                **{
                    f"val_{k}": v
                    for k, v
                    in metrics.items()
                },
            }
        )

        log(
            f"      ep={epoch:03d} | "
            f"score={score:.5f} | "
            f"U={metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={metrics['wind_direction_RMSE_deg']:.3f}"
            + (
                " *"
                if improved
                else ""
            )
        )

        if patience >= patience_limit:
            break

    elapsed = (
        time.perf_counter()
        - started
    )

    if best_state is None:
        raise RuntimeError(
            "No valid direct checkpoint."
        )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    pred_local = direct_predict_local(
        torch=torch,
        model=model,
        data=data,
        device=device,
        amp_enabled=amp_enabled,
    )

    pred_global = reconstruct_local_to_global(
        pred_local,
        e_parallel_val,
        e_cross_val,
    )

    metrics = evaluate_wind(
        y_val_global,
        pred_global,
    )

    metrics.update(
        local_component_metrics(
            y_val_local,
            pred_local,
        )
    )

    final_score, parts = selection_score(
        metrics,
        base_metrics,
    )

    history_dir = (
        output_dir / "histories"
    )

    checkpoint_dir = (
        output_dir / "checkpoints"
    )

    history_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tag = (
        f"{candidate}"
        f"__holdout_{held_out.replace(' ', '_')}"
        f"__seed_{seed}"
    )

    history_path = (
        history_dir / f"{tag}.csv"
    )

    checkpoint_path = (
        checkpoint_dir / f"{tag}.pt"
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
            "stage": "12W",
            "candidate": candidate,
            "held_out_mission": held_out,
            "seed": int(seed),
            "best_epoch": int(
                best_epoch
            ),
            "parameter_count": int(
                parameter_count
            ),
            "state_dict": best_state,
            "metrics": metrics,
            "feature_names": (
                FEATURE_GROUPS["RotDyn8"]
            ),
            "target": (
                "local parallel/cross vector"
            ),
            "Tropical_Atlantic_used": False,
        },
        checkpoint_path,
    )

    del (
        model,
        optimizer,
        scheduler,
        grad_scaler,
    )

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "candidate": candidate,
        "held_out_mission": held_out,
        "seed": int(seed),
        "status": "completed_finite",
        "parameter_count": int(
            parameter_count
        ),
        "best_epoch": int(
            best_epoch
        ),
        "selection_score": float(
            final_score
        ),
        "elapsed_seconds": float(
            elapsed
        ),
        **metrics,
        **parts,
        "residual_corr_parallel": np.nan,
        "residual_corr_cross": np.nan,
        "history_path": str(
            history_path
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
    }


# =============================================================================
# Residual training
# =============================================================================

def train_residual_run(
    *,
    torch,
    nn,
    seed,
    held_out,
    data,
    y_val_local,
    y_val_global,
    ridge_val_local,
    e_parallel_val,
    e_cross_val,
    ridge_metrics,
    base_metrics,
    output_dir,
    device,
    debug_fast,
):
    candidate = (
        "RotDyn8-RidgeResidualMLP"
    )

    seed_everything(
        torch,
        seed,
    )

    ModelClass = build_residual_model_class(
        torch,
        nn,
    )

    model = ModelClass().to(
        device
    )

    parameter_count = count_parameters(
        model
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=NEURAL_FREEZE.learning_rate,
        weight_decay=(
            NEURAL_FREEZE.weight_decay
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=(
                NEURAL_FREEZE.scheduler_factor
            ),
            patience=(
                NEURAL_FREEZE.scheduler_patience
            ),
            min_lr=(
                NEURAL_FREEZE.min_learning_rate
            ),
        )
    )

    amp_enabled = bool(
        NEURAL_FREEZE.use_amp
        and device.type == "cuda"
    )

    grad_scaler = create_grad_scaler(
        torch,
        amp_enabled,
    )

    # Epoch 0 exact Ridge anchor.
    initial_local = residual_predict_local(
        torch=torch,
        model=model,
        data=data,
        ridge_val_local=(
            ridge_val_local
        ),
        device=device,
        amp_enabled=amp_enabled,
    )

    initial_global = reconstruct_local_to_global(
        initial_local,
        e_parallel_val,
        e_cross_val,
    )

    initial_metrics = evaluate_wind(
        y_val_global,
        initial_global,
    )

    best_anchor_score, _ = selection_score(
        initial_metrics,
        ridge_metrics,
    )

    best_state = state_dict_cpu(
        model
    )

    best_epoch = 0
    patience = 0

    history = [
        {
            "epoch": 0,
            "anchor_score": float(
                best_anchor_score
            ),
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
        else NEURAL_FREEZE.max_epochs
    )

    patience_limit = (
        4
        if debug_fast
        else NEURAL_FREEZE.early_stop_patience
    )

    started = time.perf_counter()

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        model.train()

        total_sum = 0.0
        n_seen = 0

        for xb_np, ab_np, tb_np in sequential_batches(
            data["X_train_z"],
            data["anchor_train_z"],
            data[
                "residual_target_norm"
            ],
            batch_size=(
                NEURAL_FREEZE.batch_size
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
                        pred_norm - tb
                    ) ** 2
                )

                penalty = torch.mean(
                    pred_norm ** 2
                )

                loss = (
                    mse
                    + NEURAL_FREEZE.correction_penalty
                    * penalty
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Nonfinite local residual loss."
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
                    NEURAL_FREEZE.grad_clip_norm,
                )

                grad_scaler.step(
                    optimizer
                )

                grad_scaler.update()
            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    NEURAL_FREEZE.grad_clip_norm,
                )

                optimizer.step()

            bn = len(xb_np)

            total_sum += float(
                loss.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            n_seen += bn

        pred_local = residual_predict_local(
            torch=torch,
            model=model,
            data=data,
            ridge_val_local=(
                ridge_val_local
            ),
            device=device,
            amp_enabled=amp_enabled,
        )

        pred_global = reconstruct_local_to_global(
            pred_local,
            e_parallel_val,
            e_cross_val,
        )

        metrics = evaluate_wind(
            y_val_global,
            pred_global,
        )

        metrics.update(
            residual_diagnostics(
                y_val_local,
                ridge_val_local,
                pred_local,
            )
        )

        anchor_score, _ = selection_score(
            metrics,
            ridge_metrics,
        )

        improved = (
            anchor_score
            < best_anchor_score
            - 1e-8
        )

        if improved:
            best_anchor_score = float(
                anchor_score
            )
            best_state = state_dict_cpu(
                model
            )
            best_epoch = int(epoch)
            patience = 0
        else:
            patience += 1

        scheduler.step(
            anchor_score
        )

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": (
                    total_sum
                    / max(n_seen, 1)
                ),
                "anchor_score": float(
                    anchor_score
                ),
                "checkpoint_improved": bool(
                    improved
                ),
                **{
                    f"val_{k}": v
                    for k, v
                    in metrics.items()
                },
            }
        )

        log(
            f"      ep={epoch:03d} | "
            f"anchorScore={anchor_score:.5f} | "
            f"U={metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={metrics['wind_direction_RMSE_deg']:.3f} | "
            f"corrLocal=({metrics['residual_corr_parallel']:+.3f},"
            f"{metrics['residual_corr_cross']:+.3f})"
            + (
                " *"
                if improved
                else ""
            )
        )

        if patience >= patience_limit:
            break

    elapsed = (
        time.perf_counter()
        - started
    )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    pred_local = residual_predict_local(
        torch=torch,
        model=model,
        data=data,
        ridge_val_local=(
            ridge_val_local
        ),
        device=device,
        amp_enabled=amp_enabled,
    )

    pred_global = reconstruct_local_to_global(
        pred_local,
        e_parallel_val,
        e_cross_val,
    )

    metrics = evaluate_wind(
        y_val_global,
        pred_global,
    )

    metrics.update(
        local_component_metrics(
            y_val_local,
            pred_local,
        )
    )

    metrics.update(
        residual_diagnostics(
            y_val_local,
            ridge_val_local,
            pred_local,
        )
    )

    final_score, parts = selection_score(
        metrics,
        base_metrics,
    )

    history_dir = (
        output_dir / "histories"
    )

    checkpoint_dir = (
        output_dir / "checkpoints"
    )

    history_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tag = (
        f"{candidate}"
        f"__holdout_{held_out.replace(' ', '_')}"
        f"__seed_{seed}"
    )

    history_path = (
        history_dir / f"{tag}.csv"
    )

    checkpoint_path = (
        checkpoint_dir / f"{tag}.pt"
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
            "stage": "12W",
            "candidate": candidate,
            "held_out_mission": held_out,
            "seed": int(seed),
            "best_epoch": int(
                best_epoch
            ),
            "parameter_count": int(
                parameter_count
            ),
            "state_dict": best_state,
            "metrics": metrics,
            "Tropical_Atlantic_used": False,
        },
        checkpoint_path,
    )

    del (
        model,
        optimizer,
        scheduler,
        grad_scaler,
    )

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "candidate": candidate,
        "held_out_mission": held_out,
        "seed": int(seed),
        "status": "completed_finite",
        "parameter_count": int(
            parameter_count
        ),
        "best_epoch": int(
            best_epoch
        ),
        "selection_score": float(
            final_score
        ),
        "elapsed_seconds": float(
            elapsed
        ),
        **metrics,
        **parts,
        "history_path": str(
            history_path
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
    }


# =============================================================================
# Summary
# =============================================================================

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
            f"No rows for {candidate}"
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
            sub["selection_score"].std(
                ddof=0
            )
        ),
        "parameter_count_mean": float(
            sub["parameter_count"].mean()
        ),
        "best_epoch_median": float(
            np.nanmedian(
                sub["best_epoch"].to_numpy(
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
        "local_parallel_RMSE_mps",
        "local_cross_RMSE_mps",
        "residual_corr_parallel",
        "residual_corr_cross",
    ]:
        if col not in sub.columns:
            continue

        vals = sub[col].to_numpy(
            dtype=float
        )

        if np.all(
            ~np.isfinite(vals)
        ):
            row[f"{col}_mean"] = np.nan
            row[f"{col}_std"] = np.nan
        else:
            row[f"{col}_mean"] = float(
                np.nanmean(vals)
            )
            row[f"{col}_std"] = float(
                np.nanstd(
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
        "mission_wins_vs_Global4"
    ] = int(
        np.sum(
            mission_scores.to_numpy()
            < 1.0
        )
    )

    return row


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
        help=(
            "Smoke test: Antarctic holdout, one seed, "
            "RotDyn8 Ridge + GRU-Coupled + local residual MLP."
        ),
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

    missions = {
        m: load_track_j(
            dataset_root,
            m,
        )
        for m in DEVELOPMENT_MISSIONS
    }

    torch, nn = import_torch()
    configure_cuda(torch)

    if (
        args.force_cpu
        or not torch.cuda.is_available()
    ):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

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

    run_path = (
        output_dir
        / "candidate_run_results.csv"
    )

    basis_audits = []

    log("=" * 132)
    log(
        "12W — BP-ALIGNED ROTATION-EQUIVARIANT LOCAL WIND FRAME"
    )
    log("=" * 132)
    log(
        f"dataset : {dataset_root}"
    )
    log(
        f"output  : {output_dir}"
    )
    log(
        "task    : six 10-min-spaced 1-min-mean samples -> next +10-min U/V sample"
    )
    log(
        "inputs  : U,V,T,RH and optional SOG/COG transforms; no P, no wing, no extra 1-min data"
    )
    log(
        "output  : one coupled 2-D wind vector, reconstructed exactly to U/V"
    )
    log(
        f"device  : {device}"
    )

    if device.type == "cuda":
        log(
            f"GPU     : {torch.cuda.get_device_name(0)}"
        )

    log(
        "[FIREWALL] Antarctic / Atlantic / West Coast only. Tropical Atlantic not loaded."
    )
    log("")

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

        val = missions[held_out]

        train_rep = build_local_representation(
            train["X_full9"],
            train["y_uv"],
        )

        val_rep = build_local_representation(
            val["X_full9"],
            val["y_uv"],
        )

        basis_audits.append(
            {
                "held_out_mission": held_out,
                **{
                    f"train_{k}": v
                    for k, v
                    in train_rep[
                        "audit"
                    ].items()
                },
                **{
                    f"val_{k}": v
                    for k, v
                    in val_rep[
                        "audit"
                    ].items()
                },
            }
        )

        ytr_global = train[
            "y_uv"
        ]

        yva_global = val[
            "y_uv"
        ]

        ytr_local = train_rep[
            "target_local"
        ]

        yva_local = val_rep[
            "target_local"
        ]

        log("-" * 132)
        log(
            f"[{held_out}] "
            f"Ntrain={len(ytr_global):,} | "
            f"Nval={len(yva_global):,} | "
            f"val latest-ref={val_rep['audit']['reference_latest_fraction']:.3f} | "
            f"all-calm fallback={val_rep['audit']['all_calm_fallback_fraction']:.5f}"
        )

        ridge_info = {}

        # Global4 predicts global U,V.
        global_model, global_scaler = fit_ridge(
            train_rep["Global4"],
            ytr_global,
        )

        global_pred = ridge_predict(
            global_model,
            global_scaler,
            val_rep["Global4"],
        )

        global_metrics = evaluate_wind(
            yva_global,
            global_pred,
        )

        ridge_info["Global4"] = {
            "model": global_model,
            "scaler": global_scaler,
            "pred_global": global_pred,
            "metrics": global_metrics,
        }

        base_metrics = global_metrics

        score, parts = selection_score(
            global_metrics,
            base_metrics,
        )

        append_csv(
            run_path,
            {
                "candidate": "Global4-Ridge",
                "held_out_mission": held_out,
                "seed": -1,
                "status": "completed_finite",
                "parameter_count": (
                    ridge_parameter_count(
                        global_model
                    )
                ),
                "best_epoch": 0,
                "selection_score": float(
                    score
                ),
                "elapsed_seconds": 0.0,
                **global_metrics,
                **parts,
                "local_parallel_RMSE_mps": np.nan,
                "local_cross_RMSE_mps": np.nan,
                "residual_corr_parallel": np.nan,
                "residual_corr_cross": np.nan,
            },
        )

        log(
            f"  {'Global4-Ridge':27s} | "
            f"score={score:.5f} | "
            f"U={global_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={global_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={global_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={global_metrics['wind_direction_RMSE_deg']:.3f}"
        )

        for group in [
            "Rot4",
            "RotBoat6",
            "RotDyn8",
        ]:
            model, scaler = fit_ridge(
                train_rep[group],
                ytr_local,
            )

            pred_local = ridge_predict(
                model,
                scaler,
                val_rep[group],
            )

            pred_global = reconstruct_local_to_global(
                pred_local,
                val_rep["e_parallel"],
                val_rep["e_cross"],
            )

            metrics = evaluate_wind(
                yva_global,
                pred_global,
            )

            metrics.update(
                local_component_metrics(
                    yva_local,
                    pred_local,
                )
            )

            ridge_info[group] = {
                "model": model,
                "scaler": scaler,
                "pred_local": pred_local,
                "pred_global": pred_global,
                "metrics": metrics,
            }

            score, parts = selection_score(
                metrics,
                base_metrics,
            )

            append_csv(
                run_path,
                {
                    "candidate": (
                        f"{group}-Ridge"
                    ),
                    "held_out_mission": held_out,
                    "seed": -1,
                    "status": "completed_finite",
                    "parameter_count": (
                        ridge_parameter_count(
                            model
                        )
                    ),
                    "best_epoch": 0,
                    "selection_score": float(
                        score
                    ),
                    "elapsed_seconds": 0.0,
                    **metrics,
                    **parts,
                    "residual_corr_parallel": np.nan,
                    "residual_corr_cross": np.nan,
                },
            )

            log(
                f"  {group + '-Ridge':27s} | "
                f"score={score:.5f} | "
                f"U={metrics['wind_U_RMSE_mps']:.4f} | "
                f"V={metrics['wind_V_RMSE_mps']:.4f} | "
                f"WS={metrics['wind_speed_RMSE_mps']:.4f} | "
                f"WD={metrics['wind_direction_RMSE_deg']:.3f} | "
                f"local=({metrics['local_parallel_RMSE_mps']:.4f},"
                f"{metrics['local_cross_RMSE_mps']:.4f})"
            )

        direct_data = prepare_direct_data(
            train_rep["RotDyn8"],
            val_rep["RotDyn8"],
            ytr_local,
        )

        direct_specs = (
            {
                "RotDyn8-GRU-Coupled": (
                    "gru_coupled"
                ),
            }
            if args.debug_fast
            else {
                "RotDyn8-MLP": "mlp",
                "RotDyn8-GRU": "gru",
                "RotDyn8-TCN": "tcn",
                "RotDyn8-GRU-Coupled": (
                    "gru_coupled"
                ),
            }
        )

        for candidate, mode in direct_specs.items():
            for seed in active_seeds:
                log(
                    f"  [TRAIN] {candidate} | seed={seed}"
                )

                result = train_direct_run(
                    torch=torch,
                    nn=nn,
                    candidate=candidate,
                    mode=mode,
                    seed=seed,
                    held_out=held_out,
                    data=direct_data,
                    y_val_local=yva_local,
                    y_val_global=(
                        yva_global
                    ),
                    e_parallel_val=(
                        val_rep[
                            "e_parallel"
                        ]
                    ),
                    e_cross_val=(
                        val_rep[
                            "e_cross"
                        ]
                    ),
                    base_metrics=(
                        base_metrics
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
                    f"  [DONE] score={result['selection_score']:.5f} | "
                    f"U={result['wind_U_RMSE_mps']:.4f} | "
                    f"V={result['wind_V_RMSE_mps']:.4f} | "
                    f"WS={result['wind_speed_RMSE_mps']:.4f} | "
                    f"WD={result['wind_direction_RMSE_deg']:.3f}"
                )

        # One local-frame residual diagnostic.
        ridge_oof_local = (
            crossfit_ridge_local_predictions(
                train_rep["RotDyn8"],
                ytr_local,
                train[
                    "mission_labels"
                ],
            )
        )

        residual_data = prepare_residual_data(
            train_rep["RotDyn8"],
            val_rep["RotDyn8"],
            ytr_local,
            ridge_oof_local,
            ridge_info[
                "RotDyn8"
            ][
                "pred_local"
            ],
        )

        for seed in active_seeds:
            log(
                "  [TRAIN] RotDyn8-RidgeResidualMLP "
                f"| seed={seed}"
            )

            result = train_residual_run(
                torch=torch,
                nn=nn,
                seed=seed,
                held_out=held_out,
                data=residual_data,
                y_val_local=yva_local,
                y_val_global=(
                    yva_global
                ),
                ridge_val_local=(
                    ridge_info[
                        "RotDyn8"
                    ][
                        "pred_local"
                    ]
                ),
                e_parallel_val=(
                    val_rep[
                        "e_parallel"
                    ]
                ),
                e_cross_val=(
                    val_rep[
                        "e_cross"
                    ]
                ),
                ridge_metrics=(
                    ridge_info[
                        "RotDyn8"
                    ][
                        "metrics"
                    ]
                ),
                base_metrics=(
                    base_metrics
                ),
                output_dir=(
                    output_dir
                ),
                device=device,
                debug_fast=(
                    args.debug_fast
                ),
            )

            result[
                "parameter_count"
            ] += ridge_parameter_count(
                ridge_info[
                    "RotDyn8"
                ][
                    "model"
                ]
            )

            append_csv(
                run_path,
                result,
            )

            log(
                f"  [DONE] score={result['selection_score']:.5f} | "
                f"U={result['wind_U_RMSE_mps']:.4f} | "
                f"V={result['wind_V_RMSE_mps']:.4f} | "
                f"WS={result['wind_speed_RMSE_mps']:.4f} | "
                f"WD={result['wind_direction_RMSE_deg']:.3f} | "
                f"corrLocal=({result['residual_corr_parallel']:+.3f},"
                f"{result['residual_corr_cross']:+.3f})"
            )

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    audit_df = pd.DataFrame(
        basis_audits
    )

    audit_path = (
        output_dir
        / "local_frame_audit.csv"
    )

    audit_df.to_csv(
        audit_path,
        index=False,
        encoding="utf-8-sig",
    )

    if args.debug_fast:
        log("")
        log(
            "[DONE] debug-fast completed. No scientific final selection written."
        )
        return 0

    runs = pd.read_csv(
        run_path
    )

    candidate_order = [
        "Global4-Ridge",
        "Rot4-Ridge",
        "RotBoat6-Ridge",
        "RotDyn8-Ridge",
        "RotDyn8-MLP",
        "RotDyn8-GRU",
        "RotDyn8-TCN",
        "RotDyn8-GRU-Coupled",
        "RotDyn8-RidgeResidualMLP",
    ]

    expected = {
        "Global4-Ridge": 3,
        "Rot4-Ridge": 3,
        "RotBoat6-Ridge": 3,
        "RotDyn8-Ridge": 3,
        "RotDyn8-MLP": 9,
        "RotDyn8-GRU": 9,
        "RotDyn8-TCN": 9,
        "RotDyn8-GRU-Coupled": 9,
        "RotDyn8-RidgeResidualMLP": 9,
    }

    for candidate, n_expected in expected.items():
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

        if got != n_expected:
            raise RuntimeError(
                f"{candidate}: expected {n_expected} rows, got {got}."
            )

    summary = pd.DataFrame(
        [
            summarize_candidate(
                runs,
                candidate,
            )
            for candidate
            in candidate_order
        ]
    )

    base_runs = runs.loc[
        runs[
            "candidate"
        ].astype(str)
        == "Global4-Ridge"
    ].copy()

    base_means = {
        metric: float(
            base_runs[
                metric
            ].mean()
        )
        for metric in [
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

        for metric, base_value in base_means.items():
            candidate_value = float(
                sub[metric].mean()
            )

            summary.loc[
                idx,
                f"{metric}_improvement_vs_Global4_fraction",
            ] = (
                base_value
                - candidate_value
            ) / max(
                base_value,
                EPS,
            )

    summary = summary.sort_values(
        [
            "selection_score_mean",
            "parameter_count_mean",
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

    raw_best = summary.iloc[0]

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
            "mission_wins_vs_Global4"
        ]
    )

    vector_imp = float(
        raw_best[
            "wind_vector_RMSE_mps_improvement_vs_Global4_fraction"
        ]
    )

    u_imp = float(
        raw_best[
            "wind_U_RMSE_mps_improvement_vs_Global4_fraction"
        ]
    )

    v_imp = float(
        raw_best[
            "wind_V_RMSE_mps_improvement_vs_Global4_fraction"
        ]
    )

    meaningful = bool(
        raw_best_name
        != "Global4-Ridge"
        and raw_best_score
        < CONTINUE_SCORE_THRESHOLD
        and raw_best_wins
        >= MIN_MISSION_WINS
        and vector_imp
        > MIN_VECTOR_IMPROVEMENT
        and (
            u_imp
            > MIN_U_OR_V_IMPROVEMENT
            or v_imp
            > MIN_U_OR_V_IMPROVEMENT
        )
    )

    if meaningful:
        selected = raw_best_name
        decision = (
            "CONTINUE_LOCAL_WIND_FRAME_ROUTE"
        )
    else:
        selected = (
            "Global4-Ridge"
        )
        decision = (
            "LOCAL_WIND_FRAME_NOT_YET_STRONG_ENOUGH"
        )

    selected_runs = runs.loc[
        runs[
            "candidate"
        ].astype(str)
        == selected
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

    numeric_bp = {
        "best_U_minus_BP": float(
            raw_best[
                "wind_U_RMSE_mps_mean"
            ]
            - BP_STGNN_REFERENCE[
                "U_RMSE_mps"
            ]
        ),
        "best_V_minus_BP": float(
            raw_best[
                "wind_V_RMSE_mps_mean"
            ]
            - BP_STGNN_REFERENCE[
                "V_RMSE_mps"
            ]
        ),
        "best_WS_minus_BP": float(
            raw_best[
                "wind_speed_RMSE_mps_mean"
            ]
            - BP_STGNN_REFERENCE[
                "WS_RMSE_mps"
            ]
        ),
        "best_WD_minus_BP": float(
            raw_best[
                "wind_direction_RMSE_deg_mean"
            ]
            - BP_STGNN_REFERENCE[
                "WD_RMSE_deg"
            ]
        ),
    }

    payload = {
        "stage": "12W",
        "script_version": (
            SCRIPT_VERSION
        ),
        "protocol": {
            "source": (
                "Stage12G BP-aligned point-sampled dataset"
            ),
            "history": (
                "six 10-min-spaced original 1-min mean observations"
            ),
            "target": (
                "next +10-min point-sampled original 1-min mean U,V"
            ),
            "pressure_used": False,
            "wing_used": False,
            "extra_1min_between_samples_used": False,
            "absolute_position_used": False,
        },
        "representation": {
            "reference": (
                "latest historical wind >=0.5 m/s, otherwise backward fallback"
            ),
            "output": (
                "one local 2-D vector reconstructed exactly to global U,V"
            ),
            "independent_speed_direction_heads": False,
        },
        "raw_best_candidate": (
            raw_best_name
        ),
        "selected_candidate": selected,
        "decision": decision,
        "meaningful_improvement": (
            meaningful
        ),
        "numeric_BP_reference_difference_context_only": (
            numeric_bp
        ),
        "development_only": True,
        "Tropical_Atlantic_used": False,
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
        / "12W_REPORT.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "12W BP-Aligned Rotation-Equivariant Local Wind Frame\n"
        )
        f.write(
            "=" * 132
            + "\n\n"
        )

        f.write(
            "LOCAL FRAME AUDIT\n"
        )
        f.write(
            "-" * 132
            + "\n"
        )
        f.write(
            audit_df.to_string(
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
            f"selected: {selected}\n"
        )
        f.write(
            f"decision: {decision}\n\n"
        )

        f.write(
            "PUBLISHED BP-STGNN NUMERICAL REFERENCE — CONTEXT ONLY\n"
        )
        f.write(
            "-" * 132
            + "\n"
        )
        f.write(
            f"U={BP_STGNN_REFERENCE['U_RMSE_mps']:.6f}\n"
        )
        f.write(
            f"V={BP_STGNN_REFERENCE['V_RMSE_mps']:.6f}\n"
        )
        f.write(
            f"WS={BP_STGNN_REFERENCE['WS_RMSE_mps']:.6f}\n"
        )
        f.write(
            f"WD={BP_STGNN_REFERENCE['WD_RMSE_deg']:.3f}\n"
        )

    log("")
    log("=" * 132)
    log(
        "12W CANDIDATE SUMMARY"
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
        f"[SELECTED] {selected}"
    )
    log(
        f"[DECISION] {decision}"
    )
    log("")
    log(
        "Best numerical difference vs published BP reference "
        "(development context):"
    )
    log(
        f"  U  : {numeric_bp['best_U_minus_BP']:+.6f} m/s"
    )
    log(
        f"  V  : {numeric_bp['best_V_minus_BP']:+.6f} m/s"
    )
    log(
        f"  WS : {numeric_bp['best_WS_minus_BP']:+.6f} m/s"
    )
    log(
        f"  WD : {numeric_bp['best_WD_minus_BP']:+.3f} deg"
    )
    log("")
    log(
        f"[SAVED] {summary_path}"
    )
    log(
        f"[SAVED] {audit_path}"
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
