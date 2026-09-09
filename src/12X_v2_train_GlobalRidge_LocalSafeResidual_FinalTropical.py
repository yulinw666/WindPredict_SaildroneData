# -*- coding: utf-8 -*-
r"""
12X_v2_train_GlobalRidge_LocalSafeResidual_FinalTropical.py

Stage 12X-v2
============
Strict BP-aligned point-sampled benchmark.

FINAL MODEL
-----------
    Global Base4-Ridge anchor
    +
    bounded local-wind-frame residual MLP

The Ridge anchor predicts global East/North wind:
    [U,V,T,RH] x 6  ->  [U_R, V_R]

A local orthonormal wind frame is built independently for every sample from
the latest reliable historical wind vector:

    e_parallel = unit(latest reliable historical W)
    e_cross    = [-e_parallel_N, e_parallel_E]

The global Ridge residual is rotated into this local frame:

    r_parallel = (W_true - W_Ridge) dot e_parallel
    r_cross    = (W_true - W_Ridge) dot e_cross

The neural network predicts ONLY a bounded correction in this local frame:

    delta_parallel
    delta_cross

which is reconstructed exactly to global U,V:

    delta_W = delta_parallel * e_parallel + delta_cross * e_cross
    W_final = W_Ridge + delta_W

Why?
----
Stage 12W found:
    residual corr_parallel ~ 0.123
    residual corr_cross    ~ 0.081

which was materially stronger than generic global U/V residual correlations.

Therefore 12X-v2:
    - keeps the strong global Ridge anchor;
    - learns the residual in a more stationary local frame;
    - allows larger correction along wind than across wind;
    - remains compact and conservative.

FROZEN BP-ALIGNED TASK
----------------------
Source:
    Stage 12G point-sampled dataset.

History:
    six original 1-min-mean observations sampled every 10 min:
        t-50, t-40, t-30, t-20, t-10, t

Target:
    original 1-min-mean U,V sample at t+10 min.

NO:
    10-min arithmetic block means
    intermediate 1-min observations
    pressure
    wing angle
    absolute position
    trajectory reconstruction
    independent WS/WD heads

DEVELOPMENT / FINAL TEST
------------------------
Development only:
    Antarctic
    Atlantic
    West Coast

LOMO:
    holdout Antarctic  <- Atlantic + West Coast
    holdout Atlantic   <- Antarctic + West Coast
    holdout West Coast <- Antarctic + Atlantic

ONLY AFTER configuration is frozen:
    train final model on Antarctic + Atlantic + West Coast
    evaluate once on Tropical Atlantic

Important integrity note:
Tropical Atlantic was viewed in earlier project experiments, so it should not
be described as a never-seen pristine test set in the paper.  Stage 12X-v2
itself nevertheless excludes Tropical Atlantic from architecture, cap, seed,
and epoch selection.

GLOBAL RIDGE ANCHOR
-------------------
Input:
    Base4 = [U,V,T,RH] x 6

Output:
    [U_R, V_R]

RESIDUAL NETWORK INPUT
----------------------
Per-step local sequence, 8 features:
    W_parallel
    W_cross
    T
    RH
    boat_parallel
    boat_cross
    dW_parallel
    dW_cross

SOG/COG are used only through deterministic boat_parallel/boat_cross
reparameterization. No extra observations are introduced.

Per-sample summary, 10 features:
    latest_W_parallel
    latest_W_cross
    latest_dW_parallel
    latest_dW_cross
    std_W_parallel_6
    std_W_cross_6
    std_WS_6
    slope_W_parallel_6
    slope_W_cross_6
    slope_WS_6

Anchor context, 3 features:
    Ridge_parallel
    Ridge_cross
    Ridge_speed

NETWORK
-------
One frozen compact MLP:

    input
      -> 64
      -> 48
      -> 32
      -> [delta_parallel_logit,
          delta_cross_logit,
          gate_parallel_logit,
          gate_cross_logit]

The delta logits are zero initialized:
    epoch 0 = EXACT global Base4-Ridge.

SAFE BOUNDED CORRECTION
-----------------------
    gate = sigmoid(gate_logit)

    delta_parallel =
        cap_parallel
        * gate_parallel
        * tanh(delta_parallel_logit)
        * sigma(r_parallel_train)

    delta_cross =
        cap_cross
        * gate_cross
        * tanh(delta_cross_logit)
        * sigma(r_cross_train)

Candidate cap pairs selected only from development LOMO:
    A: parallel=0.20, cross=0.10
    B: parallel=0.30, cross=0.15
    C: parallel=0.45, cross=0.15

This explicitly encodes the Stage-12W finding that the along-wind residual is
more predictable than the cross-wind residual.

LOSS
----
The network only predicts local residual correction, but training loss is
computed on the reconstructed FINAL global U/V vector:

    L =
        0.45 * normalized U MSE
      + 0.55 * normalized V MSE
      + 0.10 * normalized WS MSE
      + 0.03 * circular vector cosine loss
      + 0.02 * normalized correction penalty

No separate wind-speed residual head.
No separate wind-direction head.

OOF RIDGE FOR RESIDUAL TRAINING
-------------------------------
Correction-network training anchors are generated with mission-preserving,
chronologically blocked 5-fold OOF Ridge predictions.

This prevents residual training from using artificially optimistic in-sample
Ridge residuals.

DEVELOPMENT TRAINING
--------------------
Seeds:
    500043
    501052
    502061

max epochs:
    250

early stop:
    patience 25

Candidate cap pair:
    selected by mean LOMO score across 3 missions x 3 seeds.

Seed:
    selected by mean LOMO score across the three held-out missions.

Final epoch:
    median best epoch of selected cap + selected seed across three holdouts.

FINAL TROPICAL OUTPUT
---------------------
The script reports:
    BP-STGNN published
    Base4-Ridge-final
    Ridge+LocalSafeResidual-final

Metrics:
    U RMSE
    V RMSE
    vector RMSE
    WS RMSE
    WD RMSE
    parameter count

PUBLISHED BP-STGNN REFERENCE
----------------------------
Tropical Atlantic:
    U  = 0.748640 m/s
    V  = 0.705060 m/s
    WS = 0.699400 m/s
    WD = 5.584 deg
    parameters = 242580

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\12X_v2_train_GlobalRidge_LocalSafeResidual_FinalTropical.py" --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12X_v2_GlobalRidge_LocalSafeResidual_FinalTropical_v0_1"

Smoke test:
    add --debug-fast
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
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


SCRIPT_VERSION = "0.2.0-12Xv2-GlobalRidge-LocalSafeResidual"

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
    / "12X_v2_GlobalRidge_LocalSafeResidual_FinalTropical_v0_1"
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

CAP_CONFIGS = [
    {
        "name": "P20_C10",
        "parallel": 0.20,
        "cross": 0.10,
    },
    {
        "name": "P30_C15",
        "parallel": 0.30,
        "cross": 0.15,
    },
    {
        "name": "P45_C15",
        "parallel": 0.45,
        "cross": 0.15,
    },
]

BP_REFERENCE = {
    "U_RMSE_mps": 0.748640,
    "V_RMSE_mps": 0.705060,
    "WS_RMSE_mps": 0.699400,
    "WD_RMSE_deg": 5.584000,
    "params": 242580,
}

SCORE_WEIGHTS = {
    "U": 0.25,
    "V": 0.35,
    "WS": 0.25,
    "WD": 0.15,
}


@dataclass(frozen=True)
class Freeze:
    hidden1: int = 64
    hidden2: int = 48
    hidden3: int = 32
    dropout: float = 0.10

    learning_rate: float = 7e-4
    weight_decay: float = 1e-5
    batch_size: int = 256

    max_epochs: int = 250
    early_stop_patience: int = 25
    scheduler_patience: int = 8
    scheduler_factor: float = 0.5
    min_learning_rate: float = 8e-6
    grad_clip_norm: float = 1.0

    weight_u: float = 0.45
    weight_v: float = 0.55
    weight_ws: float = 0.10
    weight_direction: float = 0.03
    weight_correction: float = 0.02

    use_amp: bool = True


FREEZE = Freeze()

SEQ_FEATURE_NAMES = [
    "W_parallel",
    "W_cross",
    "T",
    "RH",
    "boat_parallel",
    "boat_cross",
    "dW_parallel",
    "dW_cross",
]

SUMMARY_FEATURE_NAMES = [
    "latest_W_parallel",
    "latest_W_cross",
    "latest_dW_parallel",
    "latest_dW_cross",
    "std_W_parallel_6",
    "std_W_cross_6",
    "std_WS_6",
    "slope_W_parallel_6",
    "slope_W_cross_6",
    "slope_WS_6",
]


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
            return {
                str(k): cv(x)
                for k, x in v.items()
            }
        if isinstance(v, (list, tuple)):
            return [cv(x) for x in v]
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
        torch.set_float32_matmul_precision(
            "high"
        )
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


def sequential_batches(
    *arrays,
    batch_size: int,
):
    n = len(arrays[0])

    if any(len(a) != n for a in arrays):
        raise ValueError(
            "Batch arrays have inconsistent length."
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
# Dataset
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
        "Could not find track_J_joint_compatible."
    )


def mission_path(
    dataset_root: Path,
    mission: str,
):
    return (
        dataset_root
        / "track_J_joint_compatible"
        / f"{mission.replace(' ', '_')}.npz"
    )


def load_mission(
    dataset_root: Path,
    mission: str,
):
    path = mission_path(
        dataset_root,
        mission,
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
            for x
            in z["feature_names"].tolist()
        ]

    if names != FULL9_FEATURE_NAMES:
        raise RuntimeError(
            f"{mission}: unexpected feature names: {names}"
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
            f"{mission}: nonfinite values."
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
# Base4 Ridge
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


def fit_ridge_scaler(X, y):
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
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
    }


def ridge_transform_X(X, scaler):
    return (
        (
            np.asarray(
                X,
                dtype=np.float32,
            )
            - scaler["x_mean"][
                None,
                None,
                :,
            ]
        )
        / scaler["x_std"][
            None,
            None,
            :,
        ]
    ).astype(np.float32)


def ridge_transform_y(y, scaler):
    return (
        (
            np.asarray(
                y,
                dtype=np.float32,
            )
            - scaler["y_mean"][
                None,
                :,
            ]
        )
        / scaler["y_std"][
            None,
            :,
        ]
    ).astype(np.float32)


def ridge_inverse_y(yz, scaler):
    return (
        np.asarray(
            yz,
            dtype=np.float32,
        )
        * scaler["y_std"][
            None,
            :,
        ]
        + scaler["y_mean"][
            None,
            :,
        ]
    ).astype(np.float32)


def fit_ridge(X, y):
    scaler = fit_ridge_scaler(
        X,
        y,
    )

    Xz = ridge_transform_X(
        X,
        scaler,
    ).reshape(
        len(X),
        -1,
    )

    yz = ridge_transform_y(
        y,
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
    Xz = ridge_transform_X(
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

    return ridge_inverse_y(
        pred_z,
        scaler,
    )


def ridge_parameter_count(model):
    return int(
        np.asarray(model.coef_).size
        + np.asarray(model.intercept_).size
    )


# =============================================================================
# Local wind-frame representation
# =============================================================================

def slope6(x):
    arr = np.asarray(
        x,
        dtype=np.float64,
    )

    t = np.arange(
        LOOKBACK_STEPS,
        dtype=np.float64,
    )

    t = t - np.mean(t)

    centered = (
        arr
        - np.mean(
            arr,
            axis=1,
            keepdims=True,
        )
    )

    return (
        np.sum(
            centered * t[None, :],
            axis=1,
        )
        / max(
            float(
                np.sum(t ** 2)
            ),
            EPS,
        )
    )


def build_reference_basis(X_full9):
    X = np.asarray(
        X_full9,
        dtype=np.float64,
    )

    U = X[:, :, IDX_U]
    V = X[:, :, IDX_V]

    WS = np.hypot(
        U,
        V,
    )

    n = len(X)

    e_parallel = np.zeros(
        (n, 2),
        dtype=np.float64,
    )

    unresolved = np.ones(
        n,
        dtype=bool,
    )

    chosen_idx = np.full(
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
            unresolved
            & (
                WS[:, k]
                >= REFERENCE_WIND_MIN_MPS
            )
        )

        if np.any(choose):
            s = WS[choose, k]

            e_parallel[
                choose,
                0,
            ] = (
                U[choose, k]
                / s
            )

            e_parallel[
                choose,
                1,
            ] = (
                V[choose, k]
                / s
            )

            chosen_idx[
                choose
            ] = k

            unresolved[
                choose
            ] = False

    if np.any(unresolved):
        e_parallel[
            unresolved,
            0,
        ] = 1.0

        e_parallel[
            unresolved,
            1,
        ] = 0.0

    e_cross = np.stack(
        [
            -e_parallel[:, 1],
            e_parallel[:, 0],
        ],
        axis=1,
    )

    audit = {
        "reference_latest_fraction": float(
            np.mean(
                chosen_idx
                == (
                    LOOKBACK_STEPS
                    - 1
                )
            )
        ),
        "all_calm_fallback_fraction": float(
            np.mean(unresolved)
        ),
    }

    return (
        e_parallel.astype(
            np.float32
        ),
        e_cross.astype(
            np.float32
        ),
        audit,
    )


def rotate_to_local(
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
        p = (
            v[:, :, 0]
            * ep[:, None, 0]
            + v[:, :, 1]
            * ep[:, None, 1]
        )

        c = (
            v[:, :, 0]
            * ec[:, None, 0]
            + v[:, :, 1]
            * ec[:, None, 1]
        )

    elif v.ndim == 2:
        p = (
            v[:, 0]
            * ep[:, 0]
            + v[:, 1]
            * ep[:, 1]
        )

        c = (
            v[:, 0]
            * ec[:, 0]
            + v[:, 1]
            * ec[:, 1]
        )

    else:
        raise ValueError(
            f"Unexpected vector shape {v.shape}"
        )

    return np.stack(
        [p, c],
        axis=-1,
    ).astype(np.float32)


def local_to_global(
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


def build_local_features(X_full9):
    X = np.asarray(
        X_full9,
        dtype=np.float64,
    )

    (
        e_parallel,
        e_cross,
        audit,
    ) = build_reference_basis(
        X_full9
    )

    wind_global = X[
        :,
        :,
        [
            IDX_U,
            IDX_V,
        ]
    ]

    wind_local = rotate_to_local(
        wind_global,
        e_parallel,
        e_cross,
    )

    SOG = X[:, :, IDX_SOG]

    boat_global = np.stack(
        [
            SOG
            * X[
                :,
                :,
                IDX_COG_SIN,
            ],
            SOG
            * X[
                :,
                :,
                IDX_COG_COS,
            ],
        ],
        axis=-1,
    )

    boat_local = rotate_to_local(
        boat_global,
        e_parallel,
        e_cross,
    )

    dW = np.zeros_like(
        wind_local,
        dtype=np.float32,
    )

    dW[:, 1:, :] = (
        wind_local[:, 1:, :]
        - wind_local[:, :-1, :]
    )

    T = X[
        :,
        :,
        IDX_T,
    ].astype(np.float32)

    RH = X[
        :,
        :,
        IDX_RH,
    ].astype(np.float32)

    seq = np.stack(
        [
            wind_local[:, :, 0],
            wind_local[:, :, 1],
            T,
            RH,
            boat_local[:, :, 0],
            boat_local[:, :, 1],
            dW[:, :, 0],
            dW[:, :, 1],
        ],
        axis=-1,
    ).astype(np.float32)

    WS = np.linalg.norm(
        wind_local.astype(
            np.float64
        ),
        axis=2,
    )

    summary = np.stack(
        [
            wind_local[:, -1, 0],
            wind_local[:, -1, 1],
            dW[:, -1, 0],
            dW[:, -1, 1],
            np.std(
                wind_local[:, :, 0],
                axis=1,
                ddof=0,
            ),
            np.std(
                wind_local[:, :, 1],
                axis=1,
                ddof=0,
            ),
            np.std(
                WS,
                axis=1,
                ddof=0,
            ),
            slope6(
                wind_local[:, :, 0]
            ),
            slope6(
                wind_local[:, :, 1]
            ),
            slope6(WS),
        ],
        axis=1,
    ).astype(np.float32)

    if not (
        np.isfinite(seq).all()
        and np.isfinite(summary).all()
        and np.isfinite(e_parallel).all()
        and np.isfinite(e_cross).all()
    ):
        raise RuntimeError(
            "Nonfinite local-frame features."
        )

    return {
        "seq": seq,
        "summary": summary,
        "e_parallel": e_parallel,
        "e_cross": e_cross,
        "audit": audit,
    }


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
    *,
    y_true_global,
    ridge_global,
    final_global,
    e_parallel,
    e_cross,
):
    true_res_global = (
        np.asarray(
            y_true_global,
            dtype=np.float64,
        )
        - np.asarray(
            ridge_global,
            dtype=np.float64,
        )
    )

    pred_corr_global = (
        np.asarray(
            final_global,
            dtype=np.float64,
        )
        - np.asarray(
            ridge_global,
            dtype=np.float64,
        )
    )

    true_res_local = rotate_to_local(
        true_res_global,
        e_parallel,
        e_cross,
    )

    pred_corr_local = rotate_to_local(
        pred_corr_global,
        e_parallel,
        e_cross,
    )

    return {
        "residual_corr_parallel": (
            corrcoef_safe(
                true_res_local[:, 0],
                pred_corr_local[:, 0],
            )
        ),
        "residual_corr_cross": (
            corrcoef_safe(
                true_res_local[:, 1],
                pred_corr_local[:, 1],
            )
        ),
        "correction_RMS_parallel_mps": float(
            np.sqrt(
                np.mean(
                    pred_corr_local[:, 0] ** 2
                )
            )
        ),
        "correction_RMS_cross_mps": float(
            np.sqrt(
                np.mean(
                    pred_corr_local[:, 1] ** 2
                )
            )
        ),
        "correction_RMS_U_mps": float(
            np.sqrt(
                np.mean(
                    pred_corr_global[:, 0] ** 2
                )
            )
        ),
        "correction_RMS_V_mps": float(
            np.sqrt(
                np.mean(
                    pred_corr_global[:, 1] ** 2
                )
            )
        ),
    }


# =============================================================================
# OOF Ridge
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
            fold_ids[
                chunk
            ] = k

    if np.any(
        fold_ids < 0
    ):
        raise RuntimeError(
            "OOF fold assignment failed."
        )

    return fold_ids


def crossfit_ridge_predictions(
    X_base4,
    y,
    mission_labels,
):
    fold_ids = blocked_fold_ids(
        mission_labels
    )

    pred = np.full_like(
        y,
        np.nan,
        dtype=np.float32,
    )

    for k in range(
        INNER_CROSSFIT_FOLDS
    ):
        val_mask = (
            fold_ids == k
        )

        train_mask = (
            ~val_mask
        )

        model, scaler = fit_ridge(
            X_base4[
                train_mask
            ],
            y[
                train_mask
            ],
        )

        pred[
            val_mask
        ] = ridge_predict(
            model,
            scaler,
            X_base4[
                val_mask
            ],
        )

    if not np.isfinite(
        pred
    ).all():
        raise RuntimeError(
            "Nonfinite OOF Ridge prediction."
        )

    return pred


# =============================================================================
# Residual scalers
# =============================================================================

def fit_residual_scalers(
    local_features,
    ridge_oof_global,
    y_true_global,
):
    seq = np.asarray(
        local_features["seq"],
        dtype=np.float64,
    )

    summary = np.asarray(
        local_features["summary"],
        dtype=np.float64,
    )

    ridge_local = rotate_to_local(
        ridge_oof_global,
        local_features[
            "e_parallel"
        ],
        local_features[
            "e_cross"
        ],
    ).astype(
        np.float64
    )

    y_local = rotate_to_local(
        y_true_global,
        local_features[
            "e_parallel"
        ],
        local_features[
            "e_cross"
        ],
    ).astype(
        np.float64
    )

    residual_local = (
        y_local
        - ridge_local
    )

    anchor_context = np.concatenate(
        [
            ridge_local,
            np.linalg.norm(
                ridge_oof_global,
                axis=1,
                keepdims=True,
            ),
        ],
        axis=1,
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
        anchor_context,
        axis=0,
    )

    anchor_std = np.std(
        anchor_context,
        axis=0,
        ddof=0,
    )

    anchor_std = np.where(
        anchor_std < 1e-8,
        1.0,
        anchor_std,
    )

    residual_std = np.std(
        residual_local,
        axis=0,
        ddof=0,
    )

    residual_std = np.where(
        residual_std < 1e-4,
        1.0,
        residual_std,
    )

    y_std = np.std(
        np.asarray(
            y_true_global,
            dtype=np.float64,
        ),
        axis=0,
        ddof=0,
    )

    y_std = np.where(
        y_std < 1e-4,
        1.0,
        y_std,
    )

    return {
        "seq_mean": (
            seq_mean.astype(
                np.float32
            )
        ),
        "seq_std": (
            seq_std.astype(
                np.float32
            )
        ),
        "summary_mean": (
            summary_mean.astype(
                np.float32
            )
        ),
        "summary_std": (
            summary_std.astype(
                np.float32
            )
        ),
        "anchor_mean": (
            anchor_mean.astype(
                np.float32
            )
        ),
        "anchor_std": (
            anchor_std.astype(
                np.float32
            )
        ),
        "residual_std": (
            residual_std.astype(
                np.float32
            )
        ),
        "y_std": (
            y_std.astype(
                np.float32
            )
        ),
    }


def transform_residual_inputs(
    local_features,
    ridge_global,
    scalers,
):
    seq = np.asarray(
        local_features["seq"],
        dtype=np.float32,
    )

    summary = np.asarray(
        local_features["summary"],
        dtype=np.float32,
    )

    ridge_local = rotate_to_local(
        ridge_global,
        local_features[
            "e_parallel"
        ],
        local_features[
            "e_cross"
        ],
    )

    anchor_context = np.concatenate(
        [
            ridge_local,
            np.linalg.norm(
                np.asarray(
                    ridge_global,
                    dtype=np.float32,
                ),
                axis=1,
                keepdims=True,
            ),
        ],
        axis=1,
    ).astype(
        np.float32
    )

    seq_z = (
        (
            seq
            - scalers[
                "seq_mean"
            ][
                None,
                None,
                :,
            ]
        )
        / scalers[
            "seq_std"
        ][
            None,
            None,
            :,
        ]
    ).astype(
        np.float32
    )

    summary_z = (
        (
            summary
            - scalers[
                "summary_mean"
            ][
                None,
                :,
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
    )

    anchor_z = (
        (
            anchor_context
            - scalers[
                "anchor_mean"
            ][
                None,
                :,
            ]
        )
        / scalers[
            "anchor_std"
        ][
            None,
            :,
        ]
    ).astype(
        np.float32
    )

    return {
        "seq_z": seq_z,
        "summary_z": summary_z,
        "anchor_z": anchor_z,
        "ridge_local": (
            ridge_local.astype(
                np.float32
            )
        ),
    }


# =============================================================================
# Model
# =============================================================================

def build_model_class(
    torch,
    nn,
    cap_parallel,
    cap_cross,
):
    input_dim = (
        LOOKBACK_STEPS
        * len(
            SEQ_FEATURE_NAMES
        )
        + len(
            SUMMARY_FEATURE_NAMES
        )
        + 3
    )

    class LocalSafeResidualMLP(nn.Module):
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
                4,
            )

            # delta logits exactly zero -> epoch 0 exact Ridge
            nn.init.zeros_(
                self.out.weight
            )

            with torch.no_grad():
                self.out.bias[
                    :2
                ].zero_()

                # initial gate ~ 0.119
                self.out.bias[
                    2:
                ].fill_(
                    -2.0
                )

        def forward(
            self,
            seq_z,
            summary_z,
            anchor_z,
            residual_std,
        ):
            x = torch.cat(
                [
                    seq_z.flatten(
                        start_dim=1
                    ),
                    summary_z,
                    anchor_z,
                ],
                dim=1,
            )

            raw = self.out(
                self.backbone(x)
            )

            delta_logits = raw[
                :,
                :2
            ]

            gate = torch.sigmoid(
                raw[
                    :,
                    2:
                ]
            )

            unit = torch.tanh(
                delta_logits
            )

            caps = torch.as_tensor(
                [
                    float(
                        cap_parallel
                    ),
                    float(
                        cap_cross
                    ),
                ],
                dtype=unit.dtype,
                device=unit.device,
            )

            correction_local = (
                caps[
                    None,
                    :
                ]
                * gate
                * unit
                * residual_std[
                    None,
                    :
                ]
            )

            return (
                correction_local,
                gate,
            )

    return LocalSafeResidualMLP


# =============================================================================
# Loss and prediction
# =============================================================================

def local_correction_to_global_torch(
    torch,
    correction_local,
    e_parallel,
    e_cross,
):
    return (
        correction_local[
            :,
            0:1
        ]
        * e_parallel
        + correction_local[
            :,
            1:2
        ]
        * e_cross
    )


def safe_loss(
    torch,
    final_global,
    y_true_global,
    correction_local,
    residual_std,
    y_std_global,
):
    err = (
        final_global
        - y_true_global
    )

    err_z = (
        err
        / y_std_global[
            None,
            :
        ]
    )

    component = (
        FREEZE.weight_u
        * torch.mean(
            err_z[
                :,
                0
            ] ** 2
        )
        + FREEZE.weight_v
        * torch.mean(
            err_z[
                :,
                1
            ] ** 2
        )
    )

    pred_speed = torch.linalg.vector_norm(
        final_global,
        dim=1,
    )

    true_speed = torch.linalg.vector_norm(
        y_true_global,
        dim=1,
    )

    speed_scale = (
        torch.mean(
            torch.abs(
                true_speed
            )
        )
        .detach()
        .clamp_min(
            1.0
        )
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

    cos_sim = (
        torch.sum(
            final_global
            * y_true_global,
            dim=1,
        )
        / denom
    )

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
            component * 0.0
        )

    correction_norm = (
        correction_local
        / residual_std[
            None,
            :
        ]
    )

    correction_loss = torch.mean(
        correction_norm ** 2
    )

    total = (
        component
        + FREEZE.weight_ws
        * speed_loss
        + FREEZE.weight_direction
        * direction_loss
        + FREEZE.weight_correction
        * correction_loss
    )

    return total


def predict_model(
    *,
    torch,
    model,
    transformed,
    local_features,
    ridge_global,
    residual_std,
    device,
    amp_enabled,
):
    model.eval()

    corrections_local = []
    gates = []

    with torch.no_grad():
        for (
            seq_np,
            summary_np,
            anchor_np,
        ) in sequential_batches(
            transformed[
                "seq_z"
            ],
            transformed[
                "summary_z"
            ],
            transformed[
                "anchor_z"
            ],
            batch_size=(
                FREEZE.batch_size
            ),
        ):
            seq_t = torch.from_numpy(
                seq_np
            ).to(
                device,
                non_blocking=True,
            )

            summary_t = torch.from_numpy(
                summary_np
            ).to(
                device,
                non_blocking=True,
            )

            anchor_t = torch.from_numpy(
                anchor_np
            ).to(
                device,
                non_blocking=True,
            )

            residual_std_t = torch.as_tensor(
                residual_std,
                dtype=seq_t.dtype,
                device=device,
            )

            with amp_context(
                torch,
                amp_enabled,
                device,
            ):
                corr_local, gate = model(
                    seq_t,
                    summary_t,
                    anchor_t,
                    residual_std_t,
                )

            corrections_local.append(
                corr_local.detach()
                .float()
                .cpu()
                .numpy()
            )

            gates.append(
                gate.detach()
                .float()
                .cpu()
                .numpy()
            )

    correction_local = (
        np.concatenate(
            corrections_local,
            axis=0,
        )
    )

    gate = np.concatenate(
        gates,
        axis=0,
    )

    correction_global = local_to_global(
        correction_local,
        local_features[
            "e_parallel"
        ],
        local_features[
            "e_cross"
        ],
    )

    final_global = (
        np.asarray(
            ridge_global,
            dtype=np.float32,
        )
        + correction_global
    ).astype(
        np.float32
    )

    return (
        final_global,
        correction_local,
        gate,
    )


# =============================================================================
# One development run
# =============================================================================

def train_dev_run(
    *,
    torch,
    nn,
    cap_config,
    seed,
    held_out,
    train_data,
    val_data,
    output_dir,
    device,
    debug_fast,
):
    candidate = cap_config["name"]

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

    ytr = train_data[
        "y_uv"
    ]

    yva = val_data[
        "y_uv"
    ]

    ridge_model, ridge_scaler = fit_ridge(
        Xtr_base,
        ytr,
    )

    ridge_val = ridge_predict(
        ridge_model,
        ridge_scaler,
        Xva_base,
    )

    base_metrics = evaluate_wind(
        yva,
        ridge_val,
    )

    ridge_oof = crossfit_ridge_predictions(
        Xtr_base,
        ytr,
        train_data[
            "mission_labels"
        ],
    )

    local_tr = build_local_features(
        train_data[
            "X_full9"
        ]
    )

    local_va = build_local_features(
        val_data[
            "X_full9"
        ]
    )

    scalers = fit_residual_scalers(
        local_tr,
        ridge_oof,
        ytr,
    )

    tr = transform_residual_inputs(
        local_tr,
        ridge_oof,
        scalers,
    )

    va = transform_residual_inputs(
        local_va,
        ridge_val,
        scalers,
    )

    seed_everything(
        torch,
        seed,
    )

    ModelClass = build_model_class(
        torch,
        nn,
        cap_config[
            "parallel"
        ],
        cap_config[
            "cross"
        ],
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

    best_score = 1.0
    best_state = state_dict_cpu(
        model
    )
    best_epoch = 0
    patience = 0

    history = [
        {
            "epoch": 0,
            "selection_score": 1.0,
            "checkpoint_improved": True,
            **{
                f"val_{k}": v
                for k, v
                in base_metrics.items()
            },
        }
    ]

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

    residual_std_t = torch.as_tensor(
        scalers[
            "residual_std"
        ],
        dtype=torch.float32,
        device=device,
    )

    y_std_t = torch.as_tensor(
        scalers[
            "y_std"
        ],
        dtype=torch.float32,
        device=device,
    )

    ep_tr = torch.from_numpy(
        local_tr[
            "e_parallel"
        ]
    ).to(
        device
    )

    ec_tr = torch.from_numpy(
        local_tr[
            "e_cross"
        ]
    ).to(
        device
    )

    started = time.perf_counter()

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        model.train()

        total_loss = 0.0
        n_seen = 0

        n = len(
            ytr
        )

        # Sequential batching preserves temporal block ordering.
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
                device,
                non_blocking=True,
            )

            summary_t = torch.from_numpy(
                tr[
                    "summary_z"
                ][
                    start:stop
                ]
            ).to(
                device,
                non_blocking=True,
            )

            anchor_t = torch.from_numpy(
                tr[
                    "anchor_z"
                ][
                    start:stop
                ]
            ).to(
                device,
                non_blocking=True,
            )

            ridge_t = torch.from_numpy(
                ridge_oof[
                    start:stop
                ]
            ).to(
                device,
                non_blocking=True,
            )

            y_t = torch.from_numpy(
                ytr[
                    start:stop
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
                corr_local, _ = model(
                    seq_t,
                    summary_t,
                    anchor_t,
                    residual_std_t,
                )

                corr_global = (
                    local_correction_to_global_torch(
                        torch,
                        corr_local,
                        ep_tr[
                            start:stop
                        ],
                        ec_tr[
                            start:stop
                        ],
                    )
                )

                final_t = (
                    ridge_t
                    + corr_global
                )

                loss = safe_loss(
                    torch,
                    final_t,
                    y_t,
                    corr_local,
                    residual_std_t,
                    y_std_t,
                )

            if not torch.isfinite(
                loss
            ):
                raise FloatingPointError(
                    "Nonfinite 12X-v2 loss."
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

            bn = (
                stop - start
            )

            total_loss += float(
                loss.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            n_seen += bn

        final_val, _, gate_val = (
            predict_model(
                torch=torch,
                model=model,
                transformed=va,
                local_features=(
                    local_va
                ),
                ridge_global=(
                    ridge_val
                ),
                residual_std=(
                    scalers[
                        "residual_std"
                    ]
                ),
                device=device,
                amp_enabled=(
                    amp_enabled
                ),
            )
        )

        metrics = evaluate_wind(
            yva,
            final_val,
        )

        score, _ = selection_score(
            metrics,
            base_metrics,
        )

        diag = residual_diagnostics(
            y_true_global=yva,
            ridge_global=ridge_val,
            final_global=final_val,
            e_parallel=(
                local_va[
                    "e_parallel"
                ]
            ),
            e_cross=(
                local_va[
                    "e_cross"
                ]
            ),
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
            best_state = state_dict_cpu(
                model
            )
            best_epoch = int(
                epoch
            )
            patience = 0
        else:
            patience += 1

        scheduler.step(score)

        history.append(
            {
                "epoch": int(
                    epoch
                ),
                "train_loss": (
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
                "gate_parallel_mean": float(
                    np.mean(
                        gate_val[
                            :,
                            0
                        ]
                    )
                ),
                "gate_cross_mean": float(
                    np.mean(
                        gate_val[
                            :,
                            1
                        ]
                    )
                ),
                **{
                    f"val_{k}": v
                    for k, v
                    in metrics.items()
                },
                **{
                    f"val_{k}": v
                    for k, v
                    in diag.items()
                },
            }
        )

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
                f"corrLocal=({diag['residual_corr_parallel']:+.3f},"
                f"{diag['residual_corr_cross']:+.3f})"
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

    final_val, _, gate_val = predict_model(
        torch=torch,
        model=model,
        transformed=va,
        local_features=local_va,
        ridge_global=ridge_val,
        residual_std=(
            scalers[
                "residual_std"
            ]
        ),
        device=device,
        amp_enabled=(
            amp_enabled
        ),
    )

    metrics = evaluate_wind(
        yva,
        final_val,
    )

    score, ratios = selection_score(
        metrics,
        base_metrics,
    )

    diag = residual_diagnostics(
        y_true_global=yva,
        ridge_global=ridge_val,
        final_global=final_val,
        e_parallel=(
            local_va[
                "e_parallel"
            ]
        ),
        e_cross=(
            local_va[
                "e_cross"
            ]
        ),
    )

    history_dir = (
        output_dir
        / "histories"
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints_dev"
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
        history_dir
        / f"{tag}.csv"
    )

    checkpoint_path = (
        checkpoint_dir
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
            "stage": "12X-v2-development",
            "candidate": candidate,
            "cap_parallel": float(
                cap_config[
                    "parallel"
                ]
            ),
            "cap_cross": float(
                cap_config[
                    "cross"
                ]
            ),
            "seed": int(seed),
            "held_out_mission": held_out,
            "best_epoch": int(
                best_epoch
            ),
            "nn_parameter_count": int(
                nn_params
            ),
            "ridge_parameter_count": int(
                ridge_parameter_count(
                    ridge_model
                )
            ),
            "state_dict": (
                best_state
            ),
            "freeze": asdict(
                FREEZE
            ),
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
        "cap_parallel": float(
            cap_config[
                "parallel"
            ]
        ),
        "cap_cross": float(
            cap_config[
                "cross"
            ]
        ),
        "held_out_mission": held_out,
        "seed": int(seed),
        "status": "completed_finite",
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
        "best_epoch": int(
            best_epoch
        ),
        "selection_score": float(
            score
        ),
        "elapsed_seconds": float(
            elapsed
        ),
        **metrics,
        **ratios,
        **diag,
        "gate_parallel_mean": float(
            np.mean(
                gate_val[
                    :,
                    0
                ]
            )
        ),
        "gate_cross_mean": float(
            np.mean(
                gate_val[
                    :,
                    1
                ]
            )
        ),
        "reference_latest_fraction": float(
            local_va[
                "audit"
            ][
                "reference_latest_fraction"
            ]
        ),
        "all_calm_fallback_fraction": float(
            local_va[
                "audit"
            ][
                "all_calm_fallback_fraction"
            ]
        ),
        "history_path": str(
            history_path
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
    }


# =============================================================================
# Final fit and Tropical evaluation
# =============================================================================

def train_final_model(
    *,
    torch,
    nn,
    selected_cap,
    selected_seed,
    final_epochs,
    dev_all,
    tropical,
    output_dir,
    device,
):
    Xdev_base = build_base4(
        dev_all[
            "X_full9"
        ]
    )

    ydev = dev_all[
        "y_uv"
    ]

    Xtrop_base = build_base4(
        tropical[
            "X_full9"
        ]
    )

    ytrop = tropical[
        "y_uv"
    ]

    ridge_model, ridge_scaler = fit_ridge(
        Xdev_base,
        ydev,
    )

    ridge_trop = ridge_predict(
        ridge_model,
        ridge_scaler,
        Xtrop_base,
    )

    ridge_metrics = evaluate_wind(
        ytrop,
        ridge_trop,
    )

    ridge_oof = crossfit_ridge_predictions(
        Xdev_base,
        ydev,
        dev_all[
            "mission_labels"
        ],
    )

    local_dev = build_local_features(
        dev_all[
            "X_full9"
        ]
    )

    local_trop = build_local_features(
        tropical[
            "X_full9"
        ]
    )

    scalers = fit_residual_scalers(
        local_dev,
        ridge_oof,
        ydev,
    )

    tr = transform_residual_inputs(
        local_dev,
        ridge_oof,
        scalers,
    )

    te = transform_residual_inputs(
        local_trop,
        ridge_trop,
        scalers,
    )

    seed_everything(
        torch,
        selected_seed,
    )

    ModelClass = build_model_class(
        torch,
        nn,
        selected_cap[
            "parallel"
        ],
        selected_cap[
            "cross"
        ],
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

    amp_enabled = bool(
        FREEZE.use_amp
        and device.type == "cuda"
    )

    grad_scaler = create_grad_scaler(
        torch,
        amp_enabled,
    )

    residual_std_t = torch.as_tensor(
        scalers[
            "residual_std"
        ],
        dtype=torch.float32,
        device=device,
    )

    y_std_t = torch.as_tensor(
        scalers[
            "y_std"
        ],
        dtype=torch.float32,
        device=device,
    )

    ep_dev = torch.from_numpy(
        local_dev[
            "e_parallel"
        ]
    ).to(
        device
    )

    ec_dev = torch.from_numpy(
        local_dev[
            "e_cross"
        ]
    ).to(
        device
    )

    log("")
    log(
        f"[FINAL TRAIN] cap={selected_cap['name']} "
        f"(parallel={selected_cap['parallel']:.2f}, "
        f"cross={selected_cap['cross']:.2f}) | "
        f"seed={selected_seed} | epochs={final_epochs}"
    )

    n = len(
        ydev
    )

    for epoch in range(
        1,
        int(
            final_epochs
        )
        + 1,
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
                device,
                non_blocking=True,
            )

            summary_t = torch.from_numpy(
                tr[
                    "summary_z"
                ][
                    start:stop
                ]
            ).to(
                device,
                non_blocking=True,
            )

            anchor_t = torch.from_numpy(
                tr[
                    "anchor_z"
                ][
                    start:stop
                ]
            ).to(
                device,
                non_blocking=True,
            )

            ridge_t = torch.from_numpy(
                ridge_oof[
                    start:stop
                ]
            ).to(
                device,
                non_blocking=True,
            )

            y_t = torch.from_numpy(
                ydev[
                    start:stop
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
                corr_local, _ = model(
                    seq_t,
                    summary_t,
                    anchor_t,
                    residual_std_t,
                )

                corr_global = (
                    local_correction_to_global_torch(
                        torch,
                        corr_local,
                        ep_dev[
                            start:stop
                        ],
                        ec_dev[
                            start:stop
                        ],
                    )
                )

                final_t = (
                    ridge_t
                    + corr_global
                )

                loss = safe_loss(
                    torch,
                    final_t,
                    y_t,
                    corr_local,
                    residual_std_t,
                    y_std_t,
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

            bn = (
                stop - start
            )

            loss_sum += float(
                loss.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            n_seen += bn

        if (
            epoch <= 5
            or epoch % 20 == 0
            or epoch == int(
                final_epochs
            )
        ):
            log(
                f"  epoch={epoch:03d}/{int(final_epochs):03d} | "
                f"train_loss={loss_sum/max(n_seen,1):.6f}"
            )

    final_trop, _, gate_trop = predict_model(
        torch=torch,
        model=model,
        transformed=te,
        local_features=local_trop,
        ridge_global=ridge_trop,
        residual_std=(
            scalers[
                "residual_std"
            ]
        ),
        device=device,
        amp_enabled=(
            amp_enabled
        ),
    )

    final_metrics = evaluate_wind(
        ytrop,
        final_trop,
    )

    diag = residual_diagnostics(
        y_true_global=ytrop,
        ridge_global=ridge_trop,
        final_global=final_trop,
        e_parallel=(
            local_trop[
                "e_parallel"
            ]
        ),
        e_cross=(
            local_trop[
                "e_cross"
            ]
        ),
    )

    diag[
        "gate_parallel_mean"
    ] = float(
        np.mean(
            gate_trop[
                :,
                0
            ]
        )
    )

    diag[
        "gate_cross_mean"
    ] = float(
        np.mean(
            gate_trop[
                :,
                1
            ]
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
        / "12X_v2_final_model.pt"
    )

    torch.save(
        {
            "stage": "12X-v2-final",
            "selected_cap": (
                selected_cap
            ),
            "selected_seed": int(
                selected_seed
            ),
            "final_epochs": int(
                final_epochs
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
        "diagnostics": diag,
        "ridge_parameter_count": int(
            ridge_parameter_count(
                ridge_model
            )
        ),
        "nn_parameter_count": int(
            nn_params
        ),
        "effective_parameter_count": int(
            nn_params
            + ridge_parameter_count(
                ridge_model
            )
        ),
        "checkpoint_path": (
            checkpoint_path
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
            "Smoke test: Antarctic holdout, first cap, first seed. "
            "Tropical Atlantic is NOT loaded."
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

    log("=" * 132)
    log(
        "12X-v2 — GLOBAL RIDGE + LOCAL PARALLEL/CROSS SAFE RESIDUAL"
    )
    log("=" * 132)
    log(
        f"dataset : {dataset_root}"
    )
    log(
        f"output  : {output_dir}"
    )
    log(
        "task    : Stage12G point-sampled; 6 sampled points -> +10-min sampled U/V"
    )
    log(
        "anchor  : Global Base4-Ridge [U,V,T,RH]"
    )
    log(
        "residual: local parallel/cross MLP; bounded; no pressure, no wing, no trajectory"
    )
    log(
        f"device  : {device}"
    )

    if device.type == "cuda":
        log(
            f"GPU     : {torch.cuda.get_device_name(0)}"
        )

    # -------------------------------------------------------------
    # Development firewall
    # -------------------------------------------------------------
    log("")
    log(
        "[STAGE A] Loading development missions only."
    )

    dev_missions = {
        m: load_mission(
            dataset_root,
            m,
        )
        for m in DEV_MISSIONS
    }

    log(
        "[FIREWALL] Tropical Atlantic NOT loaded during development."
    )

    active_holdouts = (
        [
            "Antarctic"
        ]
        if args.debug_fast
        else DEV_MISSIONS
    )

    active_caps = (
        [
            CAP_CONFIGS[
                0
            ]
        ]
        if args.debug_fast
        else CAP_CONFIGS
    )

    active_seeds = (
        [
            SEEDS[
                0
            ]
        ]
        if args.debug_fast
        else SEEDS
    )

    run_path = (
        output_dir
        / "development_runs.csv"
    )

    for held_out in active_holdouts:
        train_names = [
            m
            for m in DEV_MISSIONS
            if m != held_out
        ]

        train_data = concat_missions(
            [
                dev_missions[
                    m
                ]
                for m in train_names
            ]
        )

        val_data = dev_missions[
            held_out
        ]

        log("-" * 132)
        log(
            f"[DEV HOLDOUT] {held_out} | "
            f"Ntrain={len(train_data['y_uv']):,} | "
            f"Nval={len(val_data['y_uv']):,}"
        )

        for cap in active_caps:
            for seed in active_seeds:
                log(
                    f"  [TRAIN] {cap['name']} "
                    f"(P={cap['parallel']:.2f}, C={cap['cross']:.2f}) "
                    f"| seed={seed}"
                )

                result = train_dev_run(
                    torch=torch,
                    nn=nn,
                    cap_config=cap,
                    seed=seed,
                    held_out=held_out,
                    train_data=train_data,
                    val_data=val_data,
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
                    f"  [DONE] score={result['selection_score']:.5f} | "
                    f"U={result['wind_U_RMSE_mps']:.4f} | "
                    f"V={result['wind_V_RMSE_mps']:.4f} | "
                    f"WS={result['wind_speed_RMSE_mps']:.4f} | "
                    f"WD={result['wind_direction_RMSE_deg']:.3f} | "
                    f"corrLocal=({result['residual_corr_parallel']:+.3f},"
                    f"{result['residual_corr_cross']:+.3f}) | "
                    f"bestEp={result['best_epoch']}"
                )

    if args.debug_fast:
        log("")
        log(
            "[DONE] debug-fast completed. Tropical Atlantic was NOT loaded."
        )
        return 0

    # -------------------------------------------------------------
    # Freeze cap / seed / final epoch from development only
    # -------------------------------------------------------------
    log("")
    log(
        "[STAGE B] Freezing configuration from development LOMO only."
    )

    runs = pd.read_csv(
        run_path
    )

    summary_rows = []

    for cap in CAP_CONFIGS:
        sub = runs.loc[
            runs[
                "candidate"
            ].astype(str)
            == cap[
                "name"
            ]
        ].copy()

        summary_rows.append(
            {
                "candidate": (
                    cap[
                        "name"
                    ]
                ),
                "cap_parallel": float(
                    cap[
                        "parallel"
                    ]
                ),
                "cap_cross": float(
                    cap[
                        "cross"
                    ]
                ),
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
                "vector_RMSE_mean": float(
                    sub[
                        "wind_vector_RMSE_mps"
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
                "residual_corr_parallel_mean": float(
                    sub[
                        "residual_corr_parallel"
                    ].mean()
                ),
                "residual_corr_cross_mean": float(
                    sub[
                        "residual_corr_cross"
                    ].mean()
                ),
                "parameter_count_mean": float(
                    sub[
                        "parameter_count"
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
        / "development_candidate_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    selected_name = str(
        summary.iloc[
            0
        ][
            "candidate"
        ]
    )

    selected_cap = next(
        cap
        for cap in CAP_CONFIGS
        if cap[
            "name"
        ]
        == selected_name
    )

    selected_runs = runs.loc[
        runs[
            "candidate"
        ].astype(str)
        == selected_name
    ].copy()

    seed_summary = (
        selected_runs.groupby(
            "seed",
            as_index=False,
        )
        .agg(
            selection_score_mean=(
                "selection_score",
                "mean",
            ),
            U_RMSE_mean=(
                "wind_U_RMSE_mps",
                "mean",
            ),
            V_RMSE_mean=(
                "wind_V_RMSE_mps",
                "mean",
            ),
            WS_RMSE_mean=(
                "wind_speed_RMSE_mps",
                "mean",
            ),
            WD_RMSE_mean=(
                "wind_direction_RMSE_deg",
                "mean",
            ),
            best_epoch_median=(
                "best_epoch",
                "median",
            ),
        )
        .sort_values(
            "selection_score_mean",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
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

    selected_seed = int(
        seed_summary.iloc[
            0
        ][
            "seed"
        ]
    )

    selected_seed_runs = (
        selected_runs.loc[
            selected_runs[
                "seed"
            ].astype(int)
            == selected_seed
        ]
    )

    final_epochs = int(
        max(
            1,
            round(
                float(
                    np.median(
                        selected_seed_runs[
                            "best_epoch"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                )
            ),
        )
    )

    frozen = {
        "selected_cap": (
            selected_cap
        ),
        "selected_seed": int(
            selected_seed
        ),
        "final_epochs": int(
            final_epochs
        ),
        "selection_source": (
            "Antarctic/Atlantic/West Coast LOMO only"
        ),
        "Tropical_Atlantic_used_for_selection": False,
        "freeze": asdict(
            FREEZE
        ),
        "network": (
            "LocalSafeResidualMLP 64->48->32"
        ),
        "anchor": (
            "Global Base4-Ridge [U,V,T,RH]"
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

    log("")
    log("=" * 132)
    log(
        "12X-v2 DEVELOPMENT SUMMARY"
    )
    log("=" * 132)
    log(
        summary.to_string(
            index=False
        )
    )
    log("")
    log(
        f"[FROZEN BEFORE TROPICAL] "
        f"cap={selected_cap['name']} "
        f"(P={selected_cap['parallel']:.2f}, "
        f"C={selected_cap['cross']:.2f}) | "
        f"seed={selected_seed} | "
        f"epochs={final_epochs}"
    )
    log(
        f"[SAVED BEFORE TROPICAL LOAD] {frozen_path}"
    )

    # -------------------------------------------------------------
    # Tropical Atlantic final benchmark
    # -------------------------------------------------------------
    log("")
    log(
        "[STAGE C] Configuration frozen. Loading Tropical Atlantic NOW."
    )

    tropical = load_mission(
        dataset_root,
        FINAL_MISSION,
    )

    dev_all = concat_missions(
        [
            dev_missions[
                m
            ]
            for m in DEV_MISSIONS
        ]
    )

    final = train_final_model(
        torch=torch,
        nn=nn,
        selected_cap=selected_cap,
        selected_seed=selected_seed,
        final_epochs=final_epochs,
        dev_all=dev_all,
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

    diag = final[
        "diagnostics"
    ]

    final_rows = [
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
            "vector_RMSE_mps": np.nan,
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
            "U_RMSE_mps": (
                ridge[
                    "wind_U_RMSE_mps"
                ]
            ),
            "V_RMSE_mps": (
                ridge[
                    "wind_V_RMSE_mps"
                ]
            ),
            "vector_RMSE_mps": (
                ridge[
                    "wind_vector_RMSE_mps"
                ]
            ),
            "WS_RMSE_mps": (
                ridge[
                    "wind_speed_RMSE_mps"
                ]
            ),
            "WD_RMSE_deg": (
                ridge[
                    "wind_direction_RMSE_deg"
                ]
            ),
            "parameter_count": int(
                final[
                    "ridge_parameter_count"
                ]
            ),
        },
        {
            "model": (
                "Ridge+LocalSafeResidual-final"
            ),
            "U_RMSE_mps": (
                ours[
                    "wind_U_RMSE_mps"
                ]
            ),
            "V_RMSE_mps": (
                ours[
                    "wind_V_RMSE_mps"
                ]
            ),
            "vector_RMSE_mps": (
                ours[
                    "wind_vector_RMSE_mps"
                ]
            ),
            "WS_RMSE_mps": (
                ours[
                    "wind_speed_RMSE_mps"
                ]
            ),
            "WD_RMSE_deg": (
                ours[
                    "wind_direction_RMSE_deg"
                ]
            ),
            "parameter_count": int(
                final[
                    "effective_parameter_count"
                ]
            ),
        },
    ]

    final_df = pd.DataFrame(
        final_rows
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
        "U": bool(
            ours[
                "wind_U_RMSE_mps"
            ]
            < BP_REFERENCE[
                "U_RMSE_mps"
            ]
        ),
        "V": bool(
            ours[
                "wind_V_RMSE_mps"
            ]
            < BP_REFERENCE[
                "V_RMSE_mps"
            ]
        ),
        "WS": bool(
            ours[
                "wind_speed_RMSE_mps"
            ]
            < BP_REFERENCE[
                "WS_RMSE_mps"
            ]
        ),
        "WD": bool(
            ours[
                "wind_direction_RMSE_deg"
            ]
            < BP_REFERENCE[
                "WD_RMSE_deg"
            ]
        ),
    }

    improvements_vs_ridge = {
        "U_fraction": float(
            (
                ridge[
                    "wind_U_RMSE_mps"
                ]
                - ours[
                    "wind_U_RMSE_mps"
                ]
            )
            / max(
                ridge[
                    "wind_U_RMSE_mps"
                ],
                EPS,
            )
        ),
        "V_fraction": float(
            (
                ridge[
                    "wind_V_RMSE_mps"
                ]
                - ours[
                    "wind_V_RMSE_mps"
                ]
            )
            / max(
                ridge[
                    "wind_V_RMSE_mps"
                ],
                EPS,
            )
        ),
        "WS_fraction": float(
            (
                ridge[
                    "wind_speed_RMSE_mps"
                ]
                - ours[
                    "wind_speed_RMSE_mps"
                ]
            )
            / max(
                ridge[
                    "wind_speed_RMSE_mps"
                ],
                EPS,
            )
        ),
        "WD_fraction": float(
            (
                ridge[
                    "wind_direction_RMSE_deg"
                ]
                - ours[
                    "wind_direction_RMSE_deg"
                ]
            )
            / max(
                ridge[
                    "wind_direction_RMSE_deg"
                ],
                EPS,
            )
        ),
    }

    report = {
        "stage": "12X-v2",
        "script_version": (
            SCRIPT_VERSION
        ),
        "frozen_before_tropical": (
            frozen
        ),
        "final_tropical": {
            "ridge_metrics": ridge,
            "local_safe_residual_metrics": (
                ours
            ),
            "diagnostics": diag,
            "improvement_vs_Ridge": (
                improvements_vs_ridge
            ),
            "beats_BP_metricwise": (
                beats
            ),
            "beats_BP_all_four": bool(
                all(
                    beats.values()
                )
            ),
            "effective_parameter_count": int(
                final[
                    "effective_parameter_count"
                ]
            ),
            "parameter_fraction_vs_BP": float(
                final[
                    "effective_parameter_count"
                ]
                / BP_REFERENCE[
                    "params"
                ]
            ),
        },
        "integrity_note": (
            "Tropical Atlantic was excluded from Stage-12X-v2 selection, "
            "but had been observed in earlier project experiments."
        ),
    }

    report_json = (
        output_dir
        / "FINAL_TROPICAL_REPORT.json"
    )

    save_json(
        report_json,
        report,
    )

    report_txt = (
        output_dir
        / "FINAL_TROPICAL_REPORT.txt"
    )

    with report_txt.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "12X-v2 FINAL TROPICAL ATLANTIC BENCHMARK\n"
        )
        f.write(
            "=" * 110
            + "\n\n"
        )
        f.write(
            final_df.to_string(
                index=False
            )
        )
        f.write(
            "\n\n"
        )

        f.write(
            "LOCAL RESIDUAL DIAGNOSTICS\n"
        )
        f.write(
            "-" * 110
            + "\n"
        )

        for k, v in diag.items():
            f.write(
                f"{k}: {v}\n"
            )

        f.write(
            "\nIMPROVEMENT VS FINAL RIDGE\n"
        )
        f.write(
            "-" * 110
            + "\n"
        )

        for k, v in improvements_vs_ridge.items():
            f.write(
                f"{k}: {100.0*v:+.4f}%\n"
            )

        f.write(
            "\nBEATS PUBLISHED BP-STGNN\n"
        )
        f.write(
            "-" * 110
            + "\n"
        )

        for k, v in beats.items():
            f.write(
                f"{k}: {v}\n"
            )

        f.write(
            f"ALL FOUR: {all(beats.values())}\n"
        )

        f.write(
            f"Params ours/BP: "
            f"{final['effective_parameter_count']}/{BP_REFERENCE['params']} "
            f"({100.0*report['final_tropical']['parameter_fraction_vs_BP']:.3f}%)\n"
        )

    log("")
    log("=" * 132)
    log(
        "FINAL TROPICAL ATLANTIC BENCHMARK"
    )
    log("=" * 132)
    log(
        final_df.to_string(
            index=False
        )
    )
    log("")
    log(
        "Local residual diagnostics:"
    )
    log(
        f"  corr parallel = {diag['residual_corr_parallel']:+.4f}"
    )
    log(
        f"  corr cross    = {diag['residual_corr_cross']:+.4f}"
    )
    log(
        f"  correction RMS parallel = {diag['correction_RMS_parallel_mps']:.5f} m/s"
    )
    log(
        f"  correction RMS cross    = {diag['correction_RMS_cross_mps']:.5f} m/s"
    )
    log(
        f"  gate mean parallel = {diag['gate_parallel_mean']:.4f}"
    )
    log(
        f"  gate mean cross    = {diag['gate_cross_mean']:.4f}"
    )
    log("")
    log(
        "Improvement vs final Base4-Ridge:"
    )

    for k, v in improvements_vs_ridge.items():
        log(
            f"  {k}: {100.0*v:+.3f}%"
        )

    log("")
    log(
        "Beats published BP-STGNN:"
    )
    log(
        f"  U  : {beats['U']}"
    )
    log(
        f"  V  : {beats['V']}"
    )
    log(
        f"  WS : {beats['WS']}"
    )
    log(
        f"  WD : {beats['WD']}"
    )
    log(
        f"  ALL FOUR : {all(beats.values())}"
    )
    log(
        f"  Params ours/BP = "
        f"{final['effective_parameter_count']}/{BP_REFERENCE['params']} "
        f"({100.0*report['final_tropical']['parameter_fraction_vs_BP']:.2f}%)"
    )
    log("")
    log(
        f"[SAVED] {final_csv}"
    )
    log(
        f"[SAVED] {report_json}"
    )
    log(
        f"[SAVED] {report_txt}"
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
