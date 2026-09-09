# -*- coding: utf-8 -*-
r"""
12V_train_BPAligned_TrajectoryAdvection_UV_LOMO.py

Stage 12V
=========
BP-aligned, point-sampled, moving-platform / trajectory-advection-aware
true-wind forecasting for Saildrone.

FROZEN MAIN TASK
----------------
Input data source:
    Stage 12G point-sampled benchmark.

Raw Saildrone records are already 1-min means.
The benchmark keeps one record every 10 min.

History:
    6 point samples:
        t-50, t-40, t-30, t-20, t-10, t

Target:
    point sample at t+10 min

Output:
    direct true-wind Cartesian components [U,V]

IMPORTANT FAIRNESS RULES
------------------------
1) NO 10-min arithmetic block averaging.
2) NO use of the 1-min records between the six 10-min samples.
3) NO pressure P or dP.
4) NO wing angle.
5) NO absolute latitude/longitude.
6) Trajectory information is reconstructed ONLY from the SOG/COG information
   already present in the BP-aligned sampled history.
7) Tropical Atlantic is NEVER loaded for development/tuning.

PHYSICAL MOTIVATION
-------------------
A moving platform observes

    W_s(t) = W(X_s(t), t)

and along its trajectory

    dW_s/dt = partial(W)/partial(t) + V_boat dot grad(W)

This script does NOT claim to separately identify the temporal derivative and
the spatial gradient from one trajectory.

Instead, it conditions the forecast on deterministic moving-platform geometry
derived from the same historical information.

BOAT VELOCITY
-------------
Saildrone COG is treated as degrees clockwise from North, consistent with the
existing project convention:

    boat_E = SOG * sin(COG)
    boat_N = SOG * cos(COG)

The Stage-12G NPZ already stores COG_sin and COG_cos, so:

    boat_E = SOG * COG_sin
    boat_N = SOG * COG_cos

RELATIVE TRAJECTORY
-------------------
The latest sampled position is set to local origin:

    X_t = (0,0)

Past positions are reconstructed backward with trapezoidal integration over
each 10-min interval:

    X_k = X_{k+1} - 0.5*(V_k + V_{k+1})*600 s

Only relative positions are used. No absolute geolocation is introduced.

FUTURE QUERY POSITION
---------------------
The next Saildrone position is estimated using current-velocity persistence:

    X_future = X_t + V_boat(t)*600 s

This is not claimed to be a navigation forecast. It is a deterministic
trajectory query derived from available history.

ADVECTION GEOMETRY
------------------
As a first-order frozen-advection approximation, the local wind vector itself
is used as the convection velocity:

    U_c,k ~= W_k = [U_k,V_k]

A wind structure sampled at historical time k would be transported to:

    X_adv,k = X_k + W_k * (t_future - t_k)

The spatial mismatch to the estimated future Saildrone sampling position is:

    d_k = X_future - X_adv,k

For each historical sample, Stage 12V includes:
    - along-wind mismatch
    - cross-wind mismatch
    - mismatch distance

It also includes source-to-future geometry before advection.

FEATURE GROUPS
--------------
Base4:
    U, V, T, RH

Boat7:
    Base4 + SOG + COG_sin + COG_cos

Trajectory8:
    Base4 +
    boat_E, boat_N,
    relative_position_E_km, relative_position_N_km

Advect18:
    Base4
    boat_E, boat_N
    relative_position_E_km, relative_position_N_km
    boat_parallel_to_wind, boat_crosswind
    relative_sampling_E, relative_sampling_N, relative_sampling_speed
    source_to_future_parallel_km, source_to_future_cross_km
    advected_mismatch_parallel_km
    advected_mismatch_cross_km
    advected_mismatch_distance_km

No feature above introduces observations unavailable to Base4+SOG+COG history.

CANDIDATES
----------
Linear ablations:
    Base4-Ridge
    Boat7-Ridge
    Trajectory8-Ridge
    Advect18-Ridge

Nonlinear alternatives:
    Advect18-HGBR-Direct
    Advect18-RidgeResidual-HGBR

Safe residual neural models:
    Advect18-RidgeResidualMLP
    Advect18-RidgeResidualGRU

Speed-residual ablation:
    Advect18-RidgeRadialSpeedResidualMLP

The speed-residual candidate predicts only:

    delta_S = |W_true| - |W_Ridge|

and applies a radial correction to the Ridge vector:

    W_hat = unit(W_Ridge) * max(|W_Ridge| + delta_S, 0)

Therefore it changes wind magnitude but preserves the Ridge-predicted
direction. It is included because Stage 12T showed a weak but repeatable
speed-residual signal. It is an ABLATION, not the default architecture.

SAFE RESIDUAL TRAINING
----------------------
Residual targets for MLP/GRU/HGBR/radial-speed models are built from
mission-preserving 5-fold BLOCKED out-of-fold Ridge predictions on the two
outer-training missions.

The final heads of neural residual models are initialized to zero:

    epoch 0 = exact Advect18-Ridge

This prevents a neural correction from being forced when the residual is not
cross-mission predictable.

LOMO DEVELOPMENT
----------------
Holdout Antarctic  <- train Atlantic + West Coast
Holdout Atlantic   <- train Antarctic + West Coast
Holdout West Coast <- train Antarctic + Atlantic

Tropical Atlantic:
    NEVER loaded.

PRIMARY DEVELOPMENT SCORE
-------------------------
Relative to Base4-Ridge:

    Score =
        0.25 * U_RMSE  / Base_U
      + 0.35 * V_RMSE  / Base_V
      + 0.25 * WS_RMSE / Base_WS
      + 0.15 * WD_RMSE / Base_WD

A candidate is considered meaningfully better only if:
    mean Score < 0.99
    wins >= 2/3 held-out missions
    mean vector RMSE improves > 0.5%
    and U or V improves > 1%

PUBLISHED BP-STGNN REFERENCE
----------------------------
Context only during development:
    U  = 0.748640 m/s
    V  = 0.705060 m/s
    WS = 0.699400 m/s
    WD = 5.584 deg
    params = 242580

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\12V_train_BPAligned_TrajectoryAdvection_UV_LOMO.py" --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12V_BPAligned_TrajectoryAdvection_UV_LOMO_v0_1"

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
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
except Exception as exc:
    raise RuntimeError(
        "scikit-learn is required. Activate the WindPredict environment."
    ) from exc


SCRIPT_VERSION = "0.1.0-12V-BPAligned-TrajectoryAdvection-UV"

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
    / "12V_BPAligned_TrajectoryAdvection_UV_LOMO_v0_1"
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
STEP_SECONDS = float(
    STEP_MINUTES * 60
)
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

EPS = 1e-12
WIND_SPEED_FLOOR = 0.20
ANCHOR_SPEED_FLOOR = 0.05
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

BASE4_FEATURES = [
    "U",
    "V",
    "T",
    "RH",
]

BOAT7_FEATURES = BASE4_FEATURES + [
    "SOG",
    "COG_sin",
    "COG_cos",
]

TRAJECTORY8_FEATURES = BASE4_FEATURES + [
    "boat_E",
    "boat_N",
    "rel_pos_E_km",
    "rel_pos_N_km",
]

ADVECT18_FEATURES = TRAJECTORY8_FEATURES + [
    "boat_parallel",
    "boat_cross",
    "rel_sampling_E",
    "rel_sampling_N",
    "rel_sampling_speed",
    "source_future_parallel_km",
    "source_future_cross_km",
    "adv_mismatch_parallel_km",
    "adv_mismatch_cross_km",
    "adv_mismatch_distance_km",
]

FEATURE_GROUPS = {
    "Base4": BASE4_FEATURES,
    "Boat7": BOAT7_FEATURES,
    "Trajectory8": TRAJECTORY8_FEATURES,
    "Advect18": ADVECT18_FEATURES,
}

ALL_DERIVED_FEATURES = ADVECT18_FEATURES


@dataclass(frozen=True)
class NeuralFreeze:
    mlp_hidden_1: int = 64
    mlp_hidden_2: int = 32

    gru_hidden: int = 48
    dense_hidden: int = 48

    speed_hidden_1: int = 48
    speed_hidden_2: int = 24

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
    use_amp: bool = True


@dataclass(frozen=True)
class HGBFreeze:
    learning_rate: float = 0.05
    max_iter: int = 250
    max_leaf_nodes: int = 15
    max_depth: int = 4
    min_samples_leaf: int = 50
    l2_regularization: float = 1.0


NEURAL_FREEZE = NeuralFreeze()
HGB_FREEZE = HGBFreeze()


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


def append_csv(
    path: Path,
    row: dict,
):
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


def seed_everything(
    torch,
    seed: int,
):
    random.seed(
        int(seed)
    )
    np.random.seed(
        int(seed)
    )
    torch.manual_seed(
        int(seed)
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            int(seed)
        )


def amp_context(
    torch,
    enabled: bool,
    device,
):
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


def create_grad_scaler(
    torch,
    enabled: bool,
):
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
        k: v.detach()
        .cpu()
        .clone()
        for k, v
        in model.state_dict().items()
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
    if not arrays:
        return

    n = len(
        arrays[0]
    )

    if any(
        len(a) != n
        for a in arrays
    ):
        raise ValueError(
            "Batch arrays have inconsistent first dimension."
        )

    for start in range(
        0,
        n,
        int(batch_size),
    ):
        stop = min(
            start
            + int(batch_size),
            n,
        )

        yield tuple(
            a[
                start:stop
            ]
            for a in arrays
        )


# =============================================================================
# Stage-12G loading
# =============================================================================

def resolve_dataset_root(
    dataset_dir: Path,
):
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


def load_track_j(
    dataset_root: Path,
    mission: str,
):
    safe = mission.replace(
        " ",
        "_",
    )

    path = (
        dataset_root
        / "track_J_joint_compatible"
        / f"{safe}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
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
            for x
            in z[
                "feature_names"
            ].tolist()
        ]

        lookback = (
            int(
                np.asarray(
                    z["lookback_steps"]
                ).reshape(-1)[0]
            )
            if "lookback_steps"
            in z.files
            else X.shape[1]
        )

        step_minutes = (
            int(
                np.asarray(
                    z["step_minutes"]
                ).reshape(-1)[0]
            )
            if "step_minutes"
            in z.files
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
            len(
                FULL9_FEATURE_NAMES
            ),
        )
    ):
        raise RuntimeError(
            f"{mission}: bad X shape {X.shape}"
        )

    if y.ndim == 3:
        if y.shape[1:] != (
            1,
            2,
        ):
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
            f"{mission}: lookback={lookback}; expected {LOOKBACK_STEPS}"
        )

    if step_minutes != STEP_MINUTES:
        raise RuntimeError(
            f"{mission}: step_minutes={step_minutes}; expected {STEP_MINUTES}"
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
        "mission": mission,
    }


def concat_missions(items):
    return {
        "X_full9": np.concatenate(
            [
                x["X_full9"]
                for x in items
            ],
            axis=0,
        ),
        "y_uv": np.concatenate(
            [
                x["y_uv"]
                for x in items
            ],
            axis=0,
        ),
        "mission_labels": np.concatenate(
            [
                np.asarray(
                    [
                        x["mission"]
                    ]
                    * len(
                        x["X_full9"]
                    ),
                    dtype=object,
                )
                for x in items
            ],
            axis=0,
        ),
    }


# =============================================================================
# Moving-platform / advection feature construction
# =============================================================================

def build_derived_features(
    X_full9,
):
    """
    Construct all 18 derived features from the same point-sampled history.

    Returns:
        X18: [N,6,18]
        audit: simple sanity diagnostics
    """
    X = np.asarray(
        X_full9,
        dtype=np.float64,
    )

    U = X[
        :,
        :,
        IDX_U
    ]

    V = X[
        :,
        :,
        IDX_V
    ]

    T = X[
        :,
        :,
        IDX_T
    ]

    RH = X[
        :,
        :,
        IDX_RH
    ]

    SOG = X[
        :,
        :,
        IDX_SOG
    ]

    COG_sin = X[
        :,
        :,
        IDX_COG_SIN
    ]

    COG_cos = X[
        :,
        :,
        IDX_COG_COS
    ]

    boat_E = (
        SOG
        * COG_sin
    )

    boat_N = (
        SOG
        * COG_cos
    )

    # -------------------------------------------------------------
    # Relative trajectory. Current sampled location is local origin.
    # Backward trapezoidal integration over 10-min intervals.
    # -------------------------------------------------------------
    pos_E = np.zeros_like(
        boat_E,
        dtype=np.float64,
    )

    pos_N = np.zeros_like(
        boat_N,
        dtype=np.float64,
    )

    for k in range(
        LOOKBACK_STEPS - 2,
        -1,
        -1,
    ):
        pos_E[
            :,
            k
        ] = (
            pos_E[
                :,
                k + 1
            ]
            - 0.5
            * (
                boat_E[
                    :,
                    k
                ]
                + boat_E[
                    :,
                    k + 1
                ]
            )
            * STEP_SECONDS
        )

        pos_N[
            :,
            k
        ] = (
            pos_N[
                :,
                k + 1
            ]
            - 0.5
            * (
                boat_N[
                    :,
                    k
                ]
                + boat_N[
                    :,
                    k + 1
                ]
            )
            * STEP_SECONDS
        )

    rel_pos_E_km = (
        pos_E
        / 1000.0
    )

    rel_pos_N_km = (
        pos_N
        / 1000.0
    )

    # -------------------------------------------------------------
    # Local wind basis.
    # U,V are treated as Earth-frame velocity components.
    # -------------------------------------------------------------
    WS = np.hypot(
        U,
        V,
    )

    denom = np.maximum(
        WS,
        WIND_SPEED_FLOOR,
    )

    wind_e_E = (
        U
        / denom
    )

    wind_e_N = (
        V
        / denom
    )

    # A perpendicular unit direction, 90 degrees CCW in E-N plane.
    wind_perp_E = (
        -wind_e_N
    )

    wind_perp_N = (
        wind_e_E
    )

    boat_parallel = (
        boat_E
        * wind_e_E
        + boat_N
        * wind_e_N
    )

    boat_cross = (
        boat_E
        * wind_perp_E
        + boat_N
        * wind_perp_N
    )

    # -------------------------------------------------------------
    # Relative sampling velocity.
    # First-order convection velocity U_c ~= local true-wind vector.
    # -------------------------------------------------------------
    rel_sampling_E = (
        boat_E
        - U
    )

    rel_sampling_N = (
        boat_N
        - V
    )

    rel_sampling_speed = np.hypot(
        rel_sampling_E,
        rel_sampling_N,
    )

    # -------------------------------------------------------------
    # Estimated future Saildrone query location:
    # current local origin + current boat velocity * 10 min.
    # -------------------------------------------------------------
    future_E = (
        boat_E[
            :,
            -1
        ]
        * STEP_SECONDS
    )

    future_N = (
        boat_N[
            :,
            -1
        ]
        * STEP_SECONDS
    )

    # Historical sample k -> target has (6-k) 10-min steps.
    lag_steps = (
        LOOKBACK_STEPS
        - np.arange(
            LOOKBACK_STEPS,
            dtype=np.float64,
        )
    )

    lag_seconds = (
        lag_steps
        * STEP_SECONDS
    )[
        None,
        :
    ]

    # Raw source-to-future displacement.
    source_future_E = (
        future_E[
            :,
            None
        ]
        - pos_E
    )

    source_future_N = (
        future_N[
            :,
            None
        ]
        - pos_N
    )

    source_future_parallel_km = (
        (
            source_future_E
            * wind_e_E
            + source_future_N
            * wind_e_N
        )
        / 1000.0
    )

    source_future_cross_km = (
        (
            source_future_E
            * wind_perp_E
            + source_future_N
            * wind_perp_N
        )
        / 1000.0
    )

    # Advect the historical wind structure to target time.
    adv_E = (
        pos_E
        + U
        * lag_seconds
    )

    adv_N = (
        pos_N
        + V
        * lag_seconds
    )

    mismatch_E = (
        future_E[
            :,
            None
        ]
        - adv_E
    )

    mismatch_N = (
        future_N[
            :,
            None
        ]
        - adv_N
    )

    adv_mismatch_parallel_km = (
        (
            mismatch_E
            * wind_e_E
            + mismatch_N
            * wind_e_N
        )
        / 1000.0
    )

    adv_mismatch_cross_km = (
        (
            mismatch_E
            * wind_perp_E
            + mismatch_N
            * wind_perp_N
        )
        / 1000.0
    )

    adv_mismatch_distance_km = (
        np.hypot(
            mismatch_E,
            mismatch_N,
        )
        / 1000.0
    )

    X18 = np.stack(
        [
            U,
            V,
            T,
            RH,
            boat_E,
            boat_N,
            rel_pos_E_km,
            rel_pos_N_km,
            boat_parallel,
            boat_cross,
            rel_sampling_E,
            rel_sampling_N,
            rel_sampling_speed,
            source_future_parallel_km,
            source_future_cross_km,
            adv_mismatch_parallel_km,
            adv_mismatch_cross_km,
            adv_mismatch_distance_km,
        ],
        axis=-1,
    ).astype(
        np.float32
    )

    if X18.shape[2] != len(
        ADVECT18_FEATURES
    ):
        raise RuntimeError(
            f"Derived feature-count mismatch: {X18.shape}"
        )

    if not np.isfinite(
        X18
    ).all():
        raise RuntimeError(
            "Nonfinite derived trajectory/advection features."
        )

    audit = {
        "SOG_mean": float(
            np.mean(
                SOG
            )
        ),
        "SOG_p99": float(
            np.percentile(
                SOG,
                99
            )
        ),
        "trajectory_span_km_mean": float(
            np.mean(
                np.hypot(
                    rel_pos_E_km[
                        :,
                        0
                    ],
                    rel_pos_N_km[
                        :,
                        0
                    ],
                )
            )
        ),
        "future_displacement_km_mean": float(
            np.mean(
                np.hypot(
                    future_E,
                    future_N,
                )
                / 1000.0
            )
        ),
        "relative_sampling_speed_mean": float(
            np.mean(
                rel_sampling_speed
            )
        ),
        "adv_mismatch_distance_km_mean": float(
            np.mean(
                adv_mismatch_distance_km
            )
        ),
    }

    return (
        X18,
        audit,
    )


def select_feature_group(
    X_full9,
    group_name,
):
    if group_name == "Base4":
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

    if group_name == "Boat7":
        return np.asarray(
            X_full9[
                :,
                :,
                [
                    IDX_U,
                    IDX_V,
                    IDX_T,
                    IDX_RH,
                    IDX_SOG,
                    IDX_COG_SIN,
                    IDX_COG_COS,
                ]
            ],
            dtype=np.float32,
        )

    X18, _ = build_derived_features(
        X_full9
    )

    if group_name == "Trajectory8":
        return X18[
            :,
            :,
            :8
        ]

    if group_name == "Advect18":
        return X18

    raise ValueError(
        f"Unknown feature group: {group_name}"
    )


# =============================================================================
# Scalers / Ridge
# =============================================================================

def fit_scaler(
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
        "x_mean": (
            x_mean.astype(
                np.float32
            )
        ),
        "x_std": (
            x_std.astype(
                np.float32
            )
        ),
        "y_mean": (
            y_mean.astype(
                np.float32
            )
        ),
        "y_std": (
            y_std.astype(
                np.float32
            )
        ),
    }


def transform_X(
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
                :,
            ]
        )
        / scaler[
            "x_std"
        ][
            None,
            None,
            :,
        ]
    ).astype(
        np.float32
    )


def transform_y(
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
                :,
            ]
        )
        / scaler[
            "y_std"
        ][
            None,
            :,
        ]
    ).astype(
        np.float32
    )


def inverse_y(
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
            :,
        ]
        + scaler[
            "y_mean"
        ][
            None,
            :,
        ]
    ).astype(
        np.float32
    )


def fit_ridge(
    X_train,
    y_train,
):
    scaler = fit_scaler(
        X_train,
        y_train,
    )

    Xz = transform_X(
        X_train,
        scaler,
    ).reshape(
        len(
            X_train
        ),
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
        len(
            X
        ),
        -1,
    )

    pred_z = np.asarray(
        model.predict(
            Xz
        ),
        dtype=np.float32,
    )

    return inverse_y(
        pred_z,
        scaler,
    )


def ridge_parameter_count(
    model,
):
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

def circular_diff_deg(
    a,
    b,
):
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


def wd_from_uv(
    uv,
):
    uv = np.asarray(
        uv,
        dtype=np.float64,
    )

    return (
        np.degrees(
            np.arctan2(
                -uv[
                    :,
                    0
                ],
                -uv[
                    :,
                    1
                ],
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
        yp
        - yt
    )

    ws_true = np.linalg.norm(
        yt,
        axis=1,
    )

    ws_pred = np.linalg.norm(
        yp,
        axis=1,
    )

    wd_true = wd_from_uv(
        yt
    )

    wd_pred = wd_from_uv(
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
                    err[
                        :,
                        0
                    ] ** 2
                )
            )
        ),
        "wind_V_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    err[
                        :,
                        1
                    ] ** 2
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
        SCORE_WEIGHTS[
            "U"
        ]
        * ratios[
            "U_ratio"
        ]
        + SCORE_WEIGHTS[
            "V"
        ]
        * ratios[
            "V_ratio"
        ]
        + SCORE_WEIGHTS[
            "WS"
        ]
        * ratios[
            "WS_ratio"
        ]
        + SCORE_WEIGHTS[
            "WD"
        ]
        * ratios[
            "WD_ratio"
        ]
    )

    return (
        float(
            score
        ),
        ratios,
    )


def corrcoef_safe(
    a,
    b,
):
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
        or np.std(
            a
        ) < EPS
        or np.std(
            b
        ) < EPS
    ):
        return np.nan

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def uv_residual_diagnostics(
    y_true,
    anchor,
    final,
):
    r_true = (
        np.asarray(
            y_true,
            dtype=np.float64,
        )
        - np.asarray(
            anchor,
            dtype=np.float64,
        )
    )

    r_pred = (
        np.asarray(
            final,
            dtype=np.float64,
        )
        - np.asarray(
            anchor,
            dtype=np.float64,
        )
    )

    return {
        "residual_corr_U": (
            corrcoef_safe(
                r_true[
                    :,
                    0
                ],
                r_pred[
                    :,
                    0
                ],
            )
        ),
        "residual_corr_V": (
            corrcoef_safe(
                r_true[
                    :,
                    1
                ],
                r_pred[
                    :,
                    1
                ],
            )
        ),
        "residual_sign_accuracy_U": float(
            np.mean(
                np.sign(
                    r_true[
                        :,
                        0
                    ]
                )
                == np.sign(
                    r_pred[
                        :,
                        0
                    ]
                )
            )
        ),
        "residual_sign_accuracy_V": float(
            np.mean(
                np.sign(
                    r_true[
                        :,
                        1
                    ]
                )
                == np.sign(
                    r_pred[
                        :,
                        1
                    ]
                )
            )
        ),
        "pred_correction_RMS_U_mps": float(
            np.sqrt(
                np.mean(
                    r_pred[
                        :,
                        0
                    ] ** 2
                )
            )
        ),
        "pred_correction_RMS_V_mps": float(
            np.sqrt(
                np.mean(
                    r_pred[
                        :,
                        1
                    ] ** 2
                )
            )
        ),
    }


def speed_residual_diagnostics(
    y_true,
    anchor,
    final,
):
    s_true = np.linalg.norm(
        np.asarray(
            y_true,
            dtype=np.float64,
        ),
        axis=1,
    )

    s_anchor = np.linalg.norm(
        np.asarray(
            anchor,
            dtype=np.float64,
        ),
        axis=1,
    )

    s_final = np.linalg.norm(
        np.asarray(
            final,
            dtype=np.float64,
        ),
        axis=1,
    )

    r_true = (
        s_true
        - s_anchor
    )

    r_pred = (
        s_final
        - s_anchor
    )

    return {
        "speed_residual_corr": (
            corrcoef_safe(
                r_true,
                r_pred,
            )
        ),
        "speed_residual_sign_accuracy": float(
            np.mean(
                np.sign(
                    r_true
                )
                == np.sign(
                    r_pred
                )
            )
        ),
        "speed_pred_correction_RMS_mps": float(
            np.sqrt(
                np.mean(
                    r_pred ** 2
                )
            )
        ),
    }


# =============================================================================
# Blocked mission-preserving OOF Ridge
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
        len(
            labels
        ),
        -1,
        dtype=np.int64,
    )

    for mission in np.unique(
        labels
    ):
        idx = np.flatnonzero(
            labels
            == mission
        )

        chunks = np.array_split(
            idx,
            n_folds,
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
            "Failed blocked OOF fold assignment."
        )

    return fold_ids


def crossfit_ridge_predictions(
    X_train,
    y_train,
    mission_labels,
    n_folds=INNER_CROSSFIT_FOLDS,
):
    fold_ids = blocked_fold_ids(
        mission_labels,
        n_folds,
    )

    pred = np.full_like(
        y_train,
        np.nan,
        dtype=np.float32,
    )

    for k in range(
        n_folds
    ):
        val_mask = (
            fold_ids
            == k
        )

        train_mask = (
            ~val_mask
        )

        model, scaler = fit_ridge(
            X_train[
                train_mask
            ],
            y_train[
                train_mask
            ],
        )

        pred[
            val_mask
        ] = ridge_predict(
            model,
            scaler,
            X_train[
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
# Residual data
# =============================================================================

def prepare_uv_residual_data(
    X_train,
    X_val,
    y_train,
    ridge_train_oof,
    ridge_val,
):
    scaler = fit_scaler(
        X_train,
        y_train,
    )

    Xtr_z = transform_X(
        X_train,
        scaler,
    )

    Xva_z = transform_X(
        X_val,
        scaler,
    )

    anchor_mean = np.mean(
        ridge_train_oof,
        axis=0,
    )

    anchor_std = np.std(
        ridge_train_oof,
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
            ridge_train_oof
            - anchor_mean[
                None,
                :,
            ]
        )
        / anchor_std[
            None,
            :,
        ]
    ).astype(
        np.float32
    )

    A_va = (
        (
            ridge_val
            - anchor_mean[
                None,
                :,
            ]
        )
        / anchor_std[
            None,
            :,
        ]
    ).astype(
        np.float32
    )

    residual = (
        y_train
        - ridge_train_oof
    ).astype(
        np.float32
    )

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
    ).astype(
        np.float32
    )

    target_norm = (
        residual
        / residual_std[
            None,
            :,
        ]
    ).astype(
        np.float32
    )

    return {
        "X_train_z": Xtr_z,
        "X_val_z": Xva_z,
        "anchor_train_z": A_tr,
        "anchor_val_z": A_va,
        "target_norm": target_norm,
        "residual_std": residual_std,
    }


def prepare_speed_residual_data(
    X_train,
    X_val,
    y_train,
    ridge_train_oof,
    ridge_val,
):
    scaler = fit_scaler(
        X_train,
        y_train,
    )

    Xtr_z = transform_X(
        X_train,
        scaler,
    )

    Xva_z = transform_X(
        X_val,
        scaler,
    )

    anchor_speed_train = np.linalg.norm(
        ridge_train_oof,
        axis=1,
    ).astype(
        np.float32
    )

    anchor_speed_val = np.linalg.norm(
        ridge_val,
        axis=1,
    ).astype(
        np.float32
    )

    anchor_mean = float(
        np.mean(
            anchor_speed_train
        )
    )

    anchor_std = float(
        np.std(
            anchor_speed_train,
            ddof=0,
        )
    )

    if anchor_std < 1e-8:
        anchor_std = 1.0

    A_tr = (
        (
            anchor_speed_train
            - anchor_mean
        )
        / anchor_std
    ).astype(
        np.float32
    )

    A_va = (
        (
            anchor_speed_val
            - anchor_mean
        )
        / anchor_std
    ).astype(
        np.float32
    )

    true_speed_train = np.linalg.norm(
        y_train,
        axis=1,
    ).astype(
        np.float32
    )

    residual = (
        true_speed_train
        - anchor_speed_train
    )

    residual_std = float(
        np.std(
            residual.astype(
                np.float64
            ),
            ddof=0,
        )
    )

    if residual_std < 1e-4:
        residual_std = 1.0

    target_norm = (
        residual
        / residual_std
    ).astype(
        np.float32
    )

    return {
        "X_train_z": Xtr_z,
        "X_val_z": Xva_z,
        "anchor_train_z": A_tr,
        "anchor_val_z": A_va,
        "target_norm": target_norm,
        "residual_std": residual_std,
    }


# =============================================================================
# Neural model definitions
# =============================================================================

def build_uv_residual_model_class(
    torch,
    nn,
    mode,
):
    input_dim = len(
        ADVECT18_FEATURES
    )

    if mode == "mlp":
        class UVResidualMLP(nn.Module):
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
                    self.backbone(
                        z
                    )
                )

        return UVResidualMLP

    if mode == "gru":
        class UVResidualGRU(nn.Module):
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

                self.backbone = nn.Sequential(
                    nn.Linear(
                        NEURAL_FREEZE.gru_hidden
                        + 2,
                        NEURAL_FREEZE.dense_hidden,
                    ),
                    nn.LayerNorm(
                        NEURAL_FREEZE.dense_hidden
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        NEURAL_FREEZE.dropout
                    ),
                )

                self.out = nn.Linear(
                    NEURAL_FREEZE.dense_hidden,
                    2,
                )

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
                seq, _ = self.gru(
                    x
                )

                h = seq[
                    :,
                    -1,
                    :,
                ]

                z = torch.cat(
                    [
                        h,
                        anchor,
                    ],
                    dim=1,
                )

                return self.out(
                    self.backbone(
                        z
                    )
                )

        return UVResidualGRU

    raise ValueError(
        f"Unknown UV residual mode: {mode}"
    )


def build_speed_residual_model_class(
    torch,
    nn,
):
    input_dim = len(
        ADVECT18_FEATURES
    )

    class RadialSpeedResidualMLP(nn.Module):
        def __init__(self):
            super().__init__()

            self.backbone = nn.Sequential(
                nn.Linear(
                    LOOKBACK_STEPS
                    * input_dim
                    + 1,
                    NEURAL_FREEZE.speed_hidden_1,
                ),
                nn.LayerNorm(
                    NEURAL_FREEZE.speed_hidden_1
                ),
                nn.GELU(),
                nn.Dropout(
                    NEURAL_FREEZE.dropout
                ),
                nn.Linear(
                    NEURAL_FREEZE.speed_hidden_1,
                    NEURAL_FREEZE.speed_hidden_2,
                ),
                nn.GELU(),
                nn.Dropout(
                    NEURAL_FREEZE.dropout
                ),
            )

            self.out = nn.Linear(
                NEURAL_FREEZE.speed_hidden_2,
                1,
            )

            nn.init.zeros_(
                self.out.weight
            )
            nn.init.zeros_(
                self.out.bias
            )

        def forward(
            self,
            x,
            anchor_speed_z,
        ):
            z = torch.cat(
                [
                    x.flatten(
                        start_dim=1
                    ),
                    anchor_speed_z.unsqueeze(
                        -1
                    ),
                ],
                dim=1,
            )

            return self.out(
                self.backbone(
                    z
                )
            ).squeeze(-1)

    return RadialSpeedResidualMLP


# =============================================================================
# Neural evaluation
# =============================================================================

def evaluate_uv_residual_model(
    *,
    torch,
    model,
    data,
    y_val,
    ridge_val,
    device,
    amp_enabled,
):
    model.eval()

    chunks = []

    with torch.no_grad():
        for xb_np, ab_np in sequential_batches(
            data[
                "X_val_z"
            ],
            data[
                "anchor_val_z"
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
        ][
            None,
            :,
        ]
    )

    pred = (
        ridge_val
        + correction
    ).astype(
        np.float32
    )

    metrics = evaluate_wind(
        y_val,
        pred,
    )

    metrics.update(
        uv_residual_diagnostics(
            y_val,
            ridge_val,
            pred,
        )
    )

    metrics.update(
        speed_residual_diagnostics(
            y_val,
            ridge_val,
            pred,
        )
    )

    return (
        metrics,
        pred,
    )


def apply_radial_speed_correction(
    *,
    ridge_uv,
    speed_correction,
    fallback_last_uv,
):
    ridge_uv = np.asarray(
        ridge_uv,
        dtype=np.float64,
    )

    speed_correction = np.asarray(
        speed_correction,
        dtype=np.float64,
    ).reshape(-1)

    fallback_last_uv = np.asarray(
        fallback_last_uv,
        dtype=np.float64,
    )

    anchor_speed = np.linalg.norm(
        ridge_uv,
        axis=1,
    )

    raw_speed = (
        anchor_speed
        + speed_correction
    )

    negative_fraction = float(
        np.mean(
            raw_speed < 0.0
        )
    )

    final_speed = np.maximum(
        raw_speed,
        0.0,
    )

    direction = np.zeros_like(
        ridge_uv,
        dtype=np.float64,
    )

    good = (
        anchor_speed
        >= ANCHOR_SPEED_FLOOR
    )

    direction[
        good
    ] = (
        ridge_uv[
            good
        ]
        / anchor_speed[
            good,
            None,
        ]
    )

    if np.any(
        ~good
    ):
        fb_speed = np.linalg.norm(
            fallback_last_uv[
                ~good
            ],
            axis=1,
        )

        fb_speed = np.maximum(
            fb_speed,
            WIND_SPEED_FLOOR,
        )

        direction[
            ~good
        ] = (
            fallback_last_uv[
                ~good
            ]
            / fb_speed[
                :,
                None,
            ]
        )

    final_uv = (
        direction
        * final_speed[
            :,
            None,
        ]
    ).astype(
        np.float32
    )

    return (
        final_uv,
        negative_fraction,
        float(
            np.mean(
                ~good
            )
        ),
    )


def evaluate_speed_residual_model(
    *,
    torch,
    model,
    data,
    y_val,
    ridge_val,
    fallback_last_uv,
    device,
    amp_enabled,
):
    model.eval()

    chunks = []

    with torch.no_grad():
        for xb_np, ab_np in sequential_batches(
            data[
                "X_val_z"
            ],
            data[
                "anchor_val_z"
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

    speed_correction = (
        pred_norm
        * float(
            data[
                "residual_std"
            ]
        )
    )

    (
        pred,
        negative_fraction,
        fallback_fraction,
    ) = apply_radial_speed_correction(
        ridge_uv=(
            ridge_val
        ),
        speed_correction=(
            speed_correction
        ),
        fallback_last_uv=(
            fallback_last_uv
        ),
    )

    metrics = evaluate_wind(
        y_val,
        pred,
    )

    metrics.update(
        speed_residual_diagnostics(
            y_val,
            ridge_val,
            pred,
        )
    )

    metrics.update(
        {
            "radial_negative_speed_fraction_preclip": (
                negative_fraction
            ),
            "radial_direction_fallback_fraction": (
                fallback_fraction
            ),
            "residual_corr_U": np.nan,
            "residual_corr_V": np.nan,
        }
    )

    return (
        metrics,
        pred,
    )


# =============================================================================
# Neural training
# =============================================================================

def train_uv_residual_run(
    *,
    torch,
    nn,
    candidate,
    mode,
    seed,
    held_out,
    data,
    y_val,
    ridge_val,
    ridge_metrics,
    base_metrics,
    output_dir,
    device,
    debug_fast,
):
    seed_everything(
        torch,
        seed,
    )

    ModelClass = build_uv_residual_model_class(
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
        lr=(
            NEURAL_FREEZE.learning_rate
        ),
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

    initial_metrics, _ = evaluate_uv_residual_model(
        torch=torch,
        model=model,
        data=data,
        y_val=y_val,
        ridge_val=ridge_val,
        device=device,
        amp_enabled=(
            amp_enabled
        ),
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

    history.append(
        {
            "epoch": 0,
            "anchor_score": (
                best_anchor_score
            ),
            **{
                f"val_{k}": v
                for k, v
                in initial_metrics.items()
            },
        }
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
            data[
                "X_train_z"
            ],
            data[
                "anchor_train_z"
            ],
            data[
                "target_norm"
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
                        pred_norm
                        - tb
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

            if not torch.isfinite(
                loss
            ):
                raise FloatingPointError(
                    "Nonfinite UV residual loss."
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

            bn = len(
                xb_np
            )

            total_sum += float(
                loss.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            n_seen += bn

        val_metrics, _ = evaluate_uv_residual_model(
            torch=torch,
            model=model,
            data=data,
            y_val=y_val,
            ridge_val=ridge_val,
            device=device,
            amp_enabled=(
                amp_enabled
            ),
        )

        anchor_score, _ = selection_score(
            val_metrics,
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

            best_epoch = int(
                epoch
            )

            patience = 0
        else:
            patience += 1

        scheduler.step(
            anchor_score
        )

        history.append(
            {
                "epoch": int(
                    epoch
                ),
                "train_loss": (
                    total_sum
                    / max(
                        n_seen,
                        1,
                    )
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
                    in val_metrics.items()
                },
            }
        )

        log(
            f"      ep={epoch:03d} | "
            f"anchorScore={anchor_score:.5f} | "
            f"U={val_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={val_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={val_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={val_metrics['wind_direction_RMSE_deg']:.3f} | "
            f"corrUV=({val_metrics['residual_corr_U']:+.3f},"
            f"{val_metrics['residual_corr_V']:+.3f})"
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

    confirmed, _ = evaluate_uv_residual_model(
        torch=torch,
        model=model,
        data=data,
        y_val=y_val,
        ridge_val=ridge_val,
        device=device,
        amp_enabled=(
            amp_enabled
        ),
    )

    final_score, parts = selection_score(
        confirmed,
        base_metrics,
    )

    history_dir = (
        output_dir
        / "histories"
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints"
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
            "stage": "12V",
            "candidate": candidate,
            "held_out_mission": (
                held_out
            ),
            "seed": int(seed),
            "best_epoch": int(
                best_epoch
            ),
            "neural_parameter_count": int(
                parameter_count
            ),
            "state_dict": (
                best_state
            ),
            "metrics": confirmed,
            "feature_names": (
                ADVECT18_FEATURES
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
        "best_epoch": int(
            best_epoch
        ),
        "selection_score": float(
            final_score
        ),
        "elapsed_seconds": float(
            elapsed
        ),
        **confirmed,
        **parts,
        "history_path": str(
            history_path
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
    }


def train_speed_residual_run(
    *,
    torch,
    nn,
    seed,
    held_out,
    data,
    y_val,
    ridge_val,
    fallback_last_uv,
    ridge_metrics,
    base_metrics,
    output_dir,
    device,
    debug_fast,
):
    candidate = (
        "Advect18-RidgeRadialSpeedResidualMLP"
    )

    seed_everything(
        torch,
        seed,
    )

    ModelClass = build_speed_residual_model_class(
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
        lr=(
            NEURAL_FREEZE.learning_rate
        ),
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

    initial_metrics, _ = evaluate_speed_residual_model(
        torch=torch,
        model=model,
        data=data,
        y_val=y_val,
        ridge_val=ridge_val,
        fallback_last_uv=(
            fallback_last_uv
        ),
        device=device,
        amp_enabled=(
            amp_enabled
        ),
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

    history = [
        {
            "epoch": 0,
            "anchor_score": (
                best_anchor_score
            ),
            **{
                f"val_{k}": v
                for k, v
                in initial_metrics.items()
            },
        }
    ]

    started = time.perf_counter()

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        model.train()

        loss_sum = 0.0
        n_seen = 0

        for xb_np, ab_np, tb_np in sequential_batches(
            data[
                "X_train_z"
            ],
            data[
                "anchor_train_z"
            ],
            data[
                "target_norm"
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
                        pred_norm
                        - tb
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

            if not torch.isfinite(
                loss
            ):
                raise FloatingPointError(
                    "Nonfinite speed residual loss."
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

            bn = len(
                xb_np
            )

            loss_sum += float(
                loss.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            n_seen += bn

        val_metrics, _ = evaluate_speed_residual_model(
            torch=torch,
            model=model,
            data=data,
            y_val=y_val,
            ridge_val=ridge_val,
            fallback_last_uv=(
                fallback_last_uv
            ),
            device=device,
            amp_enabled=(
                amp_enabled
            ),
        )

        anchor_score, _ = selection_score(
            val_metrics,
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

            best_epoch = int(
                epoch
            )

            patience = 0
        else:
            patience += 1

        scheduler.step(
            anchor_score
        )

        history.append(
            {
                "epoch": int(
                    epoch
                ),
                "train_loss": (
                    loss_sum
                    / max(
                        n_seen,
                        1,
                    )
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
                    in val_metrics.items()
                },
            }
        )

        log(
            f"      ep={epoch:03d} | "
            f"anchorScore={anchor_score:.5f} | "
            f"U={val_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={val_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={val_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={val_metrics['wind_direction_RMSE_deg']:.3f} | "
            f"speedCorr={val_metrics['speed_residual_corr']:+.3f}"
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

    confirmed, _ = evaluate_speed_residual_model(
        torch=torch,
        model=model,
        data=data,
        y_val=y_val,
        ridge_val=ridge_val,
        fallback_last_uv=(
            fallback_last_uv
        ),
        device=device,
        amp_enabled=(
            amp_enabled
        ),
    )

    final_score, parts = selection_score(
        confirmed,
        base_metrics,
    )

    history_dir = (
        output_dir
        / "histories"
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints"
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
            "stage": "12V",
            "candidate": candidate,
            "held_out_mission": (
                held_out
            ),
            "seed": int(seed),
            "best_epoch": int(
                best_epoch
            ),
            "neural_parameter_count": int(
                parameter_count
            ),
            "state_dict": (
                best_state
            ),
            "metrics": confirmed,
            "feature_names": (
                ADVECT18_FEATURES
            ),
            "radial_speed_only": True,
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
        "best_epoch": int(
            best_epoch
        ),
        "selection_score": float(
            final_score
        ),
        "elapsed_seconds": float(
            elapsed
        ),
        **confirmed,
        **parts,
        "history_path": str(
            history_path
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
    }


# =============================================================================
# HGB models
# =============================================================================

def make_hgb():
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=(
            HGB_FREEZE.learning_rate
        ),
        max_iter=(
            HGB_FREEZE.max_iter
        ),
        max_leaf_nodes=(
            HGB_FREEZE.max_leaf_nodes
        ),
        max_depth=(
            HGB_FREEZE.max_depth
        ),
        min_samples_leaf=(
            HGB_FREEZE.min_samples_leaf
        ),
        l2_regularization=(
            HGB_FREEZE.l2_regularization
        ),
        early_stopping=False,
        random_state=42,
    )


def hgb_leaf_count(
    model,
):
    predictors = getattr(
        model,
        "_predictors",
        None,
    )

    if predictors is None:
        return -1

    total = 0

    for stage in predictors:
        for predictor in stage:
            nodes = getattr(
                predictor,
                "nodes",
                None,
            )

            if nodes is None:
                continue

            names = getattr(
                nodes.dtype,
                "names",
                None,
            )

            if (
                names is not None
                and "is_leaf" in names
            ):
                total += int(
                    np.sum(
                        nodes[
                            "is_leaf"
                        ]
                    )
                )

    return int(
        total
    )


def fit_hgb_direct(
    X_train,
    y_train,
    X_val,
):
    Xtr = np.asarray(
        X_train,
        dtype=np.float32,
    ).reshape(
        len(
            X_train
        ),
        -1,
    )

    Xva = np.asarray(
        X_val,
        dtype=np.float32,
    ).reshape(
        len(
            X_val
        ),
        -1,
    )

    pred = np.zeros(
        (
            len(
                X_val
            ),
            2,
        ),
        dtype=np.float32,
    )

    models = []

    for j in range(2):
        model = make_hgb()

        model.fit(
            Xtr,
            y_train[
                :,
                j
            ],
        )

        pred[
            :,
            j
        ] = model.predict(
            Xva
        ).astype(
            np.float32
        )

        models.append(
            model
        )

    leaves = sum(
        hgb_leaf_count(
            m
        )
        for m in models
    )

    return (
        pred,
        int(
            leaves
        ),
    )


def fit_hgb_residual(
    X_train,
    ridge_train_oof,
    y_train,
    X_val,
    ridge_val,
):
    Xtr_flat = np.asarray(
        X_train,
        dtype=np.float32,
    ).reshape(
        len(
            X_train
        ),
        -1,
    )

    Xva_flat = np.asarray(
        X_val,
        dtype=np.float32,
    ).reshape(
        len(
            X_val
        ),
        -1,
    )

    anchor_mean = np.mean(
        ridge_train_oof,
        axis=0,
    )

    anchor_std = np.std(
        ridge_train_oof,
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
            ridge_train_oof
            - anchor_mean[
                None,
                :,
            ]
        )
        / anchor_std[
            None,
            :,
        ]
    )

    A_va = (
        (
            ridge_val
            - anchor_mean[
                None,
                :,
            ]
        )
        / anchor_std[
            None,
            :,
        ]
    )

    Xtr = np.concatenate(
        [
            Xtr_flat,
            A_tr,
        ],
        axis=1,
    )

    Xva = np.concatenate(
        [
            Xva_flat,
            A_va,
        ],
        axis=1,
    )

    residual = (
        y_train
        - ridge_train_oof
    )

    correction = np.zeros(
        (
            len(
                X_val
            ),
            2,
        ),
        dtype=np.float32,
    )

    models = []

    for j in range(2):
        model = make_hgb()

        model.fit(
            Xtr,
            residual[
                :,
                j
            ],
        )

        correction[
            :,
            j
        ] = model.predict(
            Xva
        ).astype(
            np.float32
        )

        models.append(
            model
        )

    pred = (
        ridge_val
        + correction
    ).astype(
        np.float32
    )

    leaves = sum(
        hgb_leaf_count(
            m
        )
        for m in models
    )

    return (
        pred,
        int(
            leaves
        ),
    )


# =============================================================================
# Summary
# =============================================================================

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
            f"No rows for {candidate}"
        )

    row = {
        "candidate": candidate,
        "runs": int(
            len(
                sub
            )
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
        "parameter_count_mean": float(
            sub[
                "parameter_count"
            ].mean()
        ),
    }

    for col in [
        "wind_U_RMSE_mps",
        "wind_V_RMSE_mps",
        "wind_vector_RMSE_mps",
        "wind_speed_RMSE_mps",
        "wind_direction_RMSE_deg",
        "residual_corr_U",
        "residual_corr_V",
        "speed_residual_corr",
    ]:
        if col not in sub.columns:
            continue

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
            "selection_score"
        ]
        .mean()
    )

    row[
        "mission_wins_vs_Base4"
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
            "Advect18 Ridge + UV residual MLP + radial speed residual MLP; "
            "HGB skipped."
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
        "12V — BP-ALIGNED TRAJECTORY / ADVECTION-AWARE POINT-SAMPLED U/V FORECASTING"
    )
    log("=" * 132)
    log(
        f"dataset : {dataset_root}"
    )
    log(
        f"output  : {output_dir}"
    )
    log(
        "task    : six 10-min-spaced 1-min-mean samples -> next +10-min sampled U/V"
    )
    log(
        "inputs  : NO pressure, NO wing, NO extra 1-min sub-window records"
    )
    log(
        "space   : relative trajectory reconstructed only from sampled SOG/COG"
    )
    log(
        "speed residual: included only as radial-correction ablation"
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

    run_path = (
        output_dir
        / "candidate_run_results.csv"
    )

    trajectory_audits = []

    for held_out in active_holdouts:
        train_names = [
            m
            for m in DEVELOPMENT_MISSIONS
            if m != held_out
        ]

        train = concat_missions(
            [
                missions[
                    m
                ]
                for m
                in train_names
            ]
        )

        val = missions[
            held_out
        ]

        Xtr_full9 = train[
            "X_full9"
        ]

        Xva_full9 = val[
            "X_full9"
        ]

        ytr = train[
            "y_uv"
        ]

        yva = val[
            "y_uv"
        ]

        Xtr18, train_audit = (
            build_derived_features(
                Xtr_full9
            )
        )

        Xva18, val_audit = (
            build_derived_features(
                Xva_full9
            )
        )

        trajectory_audits.append(
            {
                "held_out_mission": (
                    held_out
                ),
                **{
                    f"train_{k}": v
                    for k, v
                    in train_audit.items()
                },
                **{
                    f"val_{k}": v
                    for k, v
                    in val_audit.items()
                },
            }
        )

        log("-" * 132)
        log(
            f"[{held_out}] Ntrain={len(ytr):,} | Nval={len(yva):,}"
        )
        log(
            "  trajectory audit | "
            f"val span={val_audit['trajectory_span_km_mean']:.3f} km | "
            f"future disp={val_audit['future_displacement_km_mean']:.3f} km | "
            f"rel sampling={val_audit['relative_sampling_speed_mean']:.3f} m/s | "
            f"adv mismatch={val_audit['adv_mismatch_distance_km_mean']:.3f} km"
        )

        ridge_info = {}

        group_names = [
            "Base4",
            "Boat7",
            "Trajectory8",
            "Advect18",
        ]

        for group in group_names:
            if group == "Advect18":
                Xtr = Xtr18
                Xva = Xva18
            else:
                Xtr = select_feature_group(
                    Xtr_full9,
                    group,
                )

                Xva = select_feature_group(
                    Xva_full9,
                    group,
                )

            model, scaler = fit_ridge(
                Xtr,
                ytr,
            )

            pred = ridge_predict(
                model,
                scaler,
                Xva,
            )

            metrics = evaluate_wind(
                yva,
                pred,
            )

            ridge_info[
                group
            ] = {
                "Xtr": Xtr,
                "Xva": Xva,
                "model": model,
                "scaler": scaler,
                "pred": pred,
                "metrics": metrics,
            }

        base_metrics = ridge_info[
            "Base4"
        ][
            "metrics"
        ]

        for group in group_names:
            info = ridge_info[
                group
            ]

            score, parts = selection_score(
                info[
                    "metrics"
                ],
                base_metrics,
            )

            candidate = (
                f"{group}-Ridge"
            )

            row = {
                "candidate": candidate,
                "held_out_mission": (
                    held_out
                ),
                "seed": -1,
                "status": (
                    "completed_finite"
                ),
                "parameter_count": int(
                    ridge_parameter_count(
                        info[
                            "model"
                        ]
                    )
                ),
                "best_epoch": 0,
                "selection_score": float(
                    score
                ),
                "elapsed_seconds": 0.0,
                **info[
                    "metrics"
                ],
                **parts,
                "residual_corr_U": np.nan,
                "residual_corr_V": np.nan,
                "speed_residual_corr": np.nan,
            }

            append_csv(
                run_path,
                row,
            )

            log(
                f"  {candidate:30s} | "
                f"score={score:.5f} | "
                f"U={info['metrics']['wind_U_RMSE_mps']:.4f} | "
                f"V={info['metrics']['wind_V_RMSE_mps']:.4f} | "
                f"WS={info['metrics']['wind_speed_RMSE_mps']:.4f} | "
                f"WD={info['metrics']['wind_direction_RMSE_deg']:.3f}"
            )

        advect_anchor = ridge_info[
            "Advect18"
        ]

        ridge_oof = crossfit_ridge_predictions(
            Xtr18,
            ytr,
            train[
                "mission_labels"
            ],
        )

        uv_residual_data = prepare_uv_residual_data(
            Xtr18,
            Xva18,
            ytr,
            ridge_oof,
            advect_anchor[
                "pred"
            ],
        )

        speed_residual_data = prepare_speed_residual_data(
            Xtr18,
            Xva18,
            ytr,
            ridge_oof,
            advect_anchor[
                "pred"
            ],
        )

        # ---------------------------------------------------------
        # Safe U/V residual neural candidates
        # ---------------------------------------------------------
        uv_specs = (
            {
                "Advect18-RidgeResidualMLP": "mlp",
            }
            if args.debug_fast
            else {
                "Advect18-RidgeResidualMLP": "mlp",
                "Advect18-RidgeResidualGRU": "gru",
            }
        )

        for candidate, mode in uv_specs.items():
            for seed in active_seeds:
                log(
                    f"  [TRAIN] {candidate} | seed={seed}"
                )

                result = train_uv_residual_run(
                    torch=torch,
                    nn=nn,
                    candidate=candidate,
                    mode=mode,
                    seed=seed,
                    held_out=(
                        held_out
                    ),
                    data=(
                        uv_residual_data
                    ),
                    y_val=yva,
                    ridge_val=(
                        advect_anchor[
                            "pred"
                        ]
                    ),
                    ridge_metrics=(
                        advect_anchor[
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
                ] += int(
                    ridge_parameter_count(
                        advect_anchor[
                            "model"
                        ]
                    )
                )

                append_csv(
                    run_path,
                    result,
                )

                log(
                    f"  [DONE] "
                    f"score={result['selection_score']:.5f} | "
                    f"U={result['wind_U_RMSE_mps']:.4f} | "
                    f"V={result['wind_V_RMSE_mps']:.4f} | "
                    f"WS={result['wind_speed_RMSE_mps']:.4f} | "
                    f"WD={result['wind_direction_RMSE_deg']:.3f}"
                )

        # ---------------------------------------------------------
        # Radial speed-residual ablation
        # ---------------------------------------------------------
        fallback_last_uv = (
            Xva_full9[
                :,
                -1,
                [
                    IDX_U,
                    IDX_V,
                ]
            ]
        ).astype(
            np.float32
        )

        for seed in active_seeds:
            log(
                "  [TRAIN] Advect18-RidgeRadialSpeedResidualMLP "
                f"| seed={seed}"
            )

            result = train_speed_residual_run(
                torch=torch,
                nn=nn,
                seed=seed,
                held_out=(
                    held_out
                ),
                data=(
                    speed_residual_data
                ),
                y_val=yva,
                ridge_val=(
                    advect_anchor[
                        "pred"
                    ]
                ),
                fallback_last_uv=(
                    fallback_last_uv
                ),
                ridge_metrics=(
                    advect_anchor[
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
            ] += int(
                ridge_parameter_count(
                    advect_anchor[
                        "model"
                    ]
                )
            )

            append_csv(
                run_path,
                result,
            )

            log(
                f"  [DONE] "
                f"score={result['selection_score']:.5f} | "
                f"U={result['wind_U_RMSE_mps']:.4f} | "
                f"V={result['wind_V_RMSE_mps']:.4f} | "
                f"WS={result['wind_speed_RMSE_mps']:.4f} | "
                f"WD={result['wind_direction_RMSE_deg']:.3f} | "
                f"speedCorr={result['speed_residual_corr']:+.3f}"
            )

        # ---------------------------------------------------------
        # HGB nonlinear alternatives
        # ---------------------------------------------------------
        if not args.debug_fast:
            started = time.perf_counter()

            hgb_direct_pred, direct_leaves = fit_hgb_direct(
                Xtr18,
                ytr,
                Xva18,
            )

            direct_metrics = evaluate_wind(
                yva,
                hgb_direct_pred,
            )

            direct_score, parts = selection_score(
                direct_metrics,
                base_metrics,
            )

            direct_row = {
                "candidate": (
                    "Advect18-HGBR-Direct"
                ),
                "held_out_mission": (
                    held_out
                ),
                "seed": -1,
                "status": (
                    "completed_finite"
                ),
                "parameter_count": int(
                    direct_leaves
                ),
                "best_epoch": int(
                    HGB_FREEZE.max_iter
                ),
                "selection_score": float(
                    direct_score
                ),
                "elapsed_seconds": float(
                    time.perf_counter()
                    - started
                ),
                **direct_metrics,
                **parts,
                "residual_corr_U": np.nan,
                "residual_corr_V": np.nan,
                "speed_residual_corr": np.nan,
            }

            append_csv(
                run_path,
                direct_row,
            )

            log(
                "  Advect18-HGBR-Direct           | "
                f"score={direct_score:.5f} | "
                f"U={direct_metrics['wind_U_RMSE_mps']:.4f} | "
                f"V={direct_metrics['wind_V_RMSE_mps']:.4f} | "
                f"WS={direct_metrics['wind_speed_RMSE_mps']:.4f} | "
                f"WD={direct_metrics['wind_direction_RMSE_deg']:.3f}"
            )

            started = time.perf_counter()

            hgb_res_pred, res_leaves = fit_hgb_residual(
                Xtr18,
                ridge_oof,
                ytr,
                Xva18,
                advect_anchor[
                    "pred"
                ],
            )

            res_metrics = evaluate_wind(
                yva,
                hgb_res_pred,
            )

            res_metrics.update(
                uv_residual_diagnostics(
                    yva,
                    advect_anchor[
                        "pred"
                    ],
                    hgb_res_pred,
                )
            )

            res_metrics.update(
                speed_residual_diagnostics(
                    yva,
                    advect_anchor[
                        "pred"
                    ],
                    hgb_res_pred,
                )
            )

            res_score, parts = selection_score(
                res_metrics,
                base_metrics,
            )

            res_row = {
                "candidate": (
                    "Advect18-RidgeResidual-HGBR"
                ),
                "held_out_mission": (
                    held_out
                ),
                "seed": -1,
                "status": (
                    "completed_finite"
                ),
                "parameter_count": int(
                    res_leaves
                    + ridge_parameter_count(
                        advect_anchor[
                            "model"
                        ]
                    )
                ),
                "best_epoch": int(
                    HGB_FREEZE.max_iter
                ),
                "selection_score": float(
                    res_score
                ),
                "elapsed_seconds": float(
                    time.perf_counter()
                    - started
                ),
                **res_metrics,
                **parts,
            }

            append_csv(
                run_path,
                res_row,
            )

            log(
                "  Advect18-RidgeResidual-HGBR    | "
                f"score={res_score:.5f} | "
                f"U={res_metrics['wind_U_RMSE_mps']:.4f} | "
                f"V={res_metrics['wind_V_RMSE_mps']:.4f} | "
                f"WS={res_metrics['wind_speed_RMSE_mps']:.4f} | "
                f"WD={res_metrics['wind_direction_RMSE_deg']:.3f} | "
                f"corrUV=({res_metrics['residual_corr_U']:+.3f},"
                f"{res_metrics['residual_corr_V']:+.3f})"
            )

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    audit_df = pd.DataFrame(
        trajectory_audits
    )

    audit_path = (
        output_dir
        / "trajectory_feature_audit.csv"
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
        "Base4-Ridge",
        "Boat7-Ridge",
        "Trajectory8-Ridge",
        "Advect18-Ridge",
        "Advect18-RidgeResidualMLP",
        "Advect18-RidgeResidualGRU",
        "Advect18-RidgeRadialSpeedResidualMLP",
        "Advect18-HGBR-Direct",
        "Advect18-RidgeResidual-HGBR",
    ]

    expected = {
        "Base4-Ridge": 3,
        "Boat7-Ridge": 3,
        "Trajectory8-Ridge": 3,
        "Advect18-Ridge": 3,
        "Advect18-RidgeResidualMLP": 9,
        "Advect18-RidgeResidualGRU": 9,
        "Advect18-RidgeRadialSpeedResidualMLP": 9,
        "Advect18-HGBR-Direct": 3,
        "Advect18-RidgeResidual-HGBR": 3,
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
        == "Base4-Ridge"
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
                sub[
                    metric
                ].mean()
            )

            summary.loc[
                idx,
                f"{metric}_improvement_vs_Base4_fraction",
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

    raw_best = summary.iloc[
        0
    ]

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
            "mission_wins_vs_Base4"
        ]
    )

    vector_imp = float(
        raw_best[
            "wind_vector_RMSE_mps_improvement_vs_Base4_fraction"
        ]
    )

    u_imp = float(
        raw_best[
            "wind_U_RMSE_mps_improvement_vs_Base4_fraction"
        ]
    )

    v_imp = float(
        raw_best[
            "wind_V_RMSE_mps_improvement_vs_Base4_fraction"
        ]
    )

    meaningful = bool(
        raw_best_name
        != "Base4-Ridge"
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
            "CONTINUE_TRAJECTORY_ADVECTION_ROUTE"
        )
    else:
        selected = (
            "Base4-Ridge"
        )
        decision = (
            "TRAJECTORY_ADVECTION_NOT_YET_STRONG_ENOUGH"
        )

    speed_row = summary.loc[
        summary[
            "candidate"
        ]
        == "Advect18-RidgeRadialSpeedResidualMLP"
    ].iloc[
        0
    ]

    speed_score = float(
        speed_row[
            "selection_score_mean"
        ]
    )

    speed_ws_imp = float(
        speed_row[
            "wind_speed_RMSE_mps_improvement_vs_Base4_fraction"
        ]
    )

    speed_corr = float(
        speed_row.get(
            "speed_residual_corr_mean",
            np.nan,
        )
    )

    advect_row = summary.loc[
        summary[
            "candidate"
        ]
        == "Advect18-Ridge"
    ].iloc[
        0
    ]

    # Separate test: does radial speed residual improve its OWN Advect18 anchor?
    advect_ws = float(
        advect_row[
            "wind_speed_RMSE_mps_mean"
        ]
    )

    speed_ws = float(
        speed_row[
            "wind_speed_RMSE_mps_mean"
        ]
    )

    radial_improvement_vs_advect = (
        advect_ws
        - speed_ws
    ) / max(
        advect_ws,
        EPS,
    )

    if (
        radial_improvement_vs_advect
        >= 0.005
        and np.isfinite(
            speed_corr
        )
        and speed_corr
        >= 0.10
    ):
        speed_residual_decision = (
            "KEEP_RADIAL_SPEED_RESIDUAL_AS_AUXILIARY_BRANCH"
        )
    else:
        speed_residual_decision = (
            "DROP_RADIAL_SPEED_RESIDUAL_FROM_MAIN_MODEL"
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
        "stage": "12V",
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
        "trajectory_physics": {
            "relative_position_source": (
                "deterministic integration of sampled SOG/COG only"
            ),
            "future_position_query": (
                "current SOG/COG persistence for 10 min"
            ),
            "advection_velocity_approximation": (
                "local true-wind vector U,V"
            ),
            "claim_boundary": (
                "conditioning on moving-platform geometry; "
                "not identification of separate temporal/spatial derivatives"
            ),
        },
        "feature_groups": (
            FEATURE_GROUPS
        ),
        "neural_freeze": asdict(
            NEURAL_FREEZE
        ),
        "hgb_freeze": asdict(
            HGB_FREEZE
        ),
        "raw_best_candidate": (
            raw_best_name
        ),
        "selected_candidate": (
            selected
        ),
        "decision": (
            decision
        ),
        "meaningful_improvement": (
            meaningful
        ),
        "radial_speed_residual": {
            "decision": (
                speed_residual_decision
            ),
            "improvement_vs_Advect18_Ridge_fraction": (
                radial_improvement_vs_advect
            ),
            "mean_speed_residual_corr": (
                speed_corr
            ),
        },
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
        / "12V_REPORT.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "12V BP-Aligned Trajectory / Advection-Aware U/V Forecasting\n"
        )
        f.write(
            "=" * 132
            + "\n\n"
        )

        f.write(
            "FROZEN PROTOCOL\n"
        )
        f.write(
            "-" * 132
            + "\n"
        )
        f.write(
            "Stage12G point-sampled benchmark\n"
        )
        f.write(
            "Six 10-min-spaced original 1-min mean observations -> next +10-min sample\n"
        )
        f.write(
            "Pressure: NO\n"
        )
        f.write(
            "Wing: NO\n"
        )
        f.write(
            "Extra 1-min records between samples: NO\n"
        )
        f.write(
            "Absolute latitude/longitude: NO\n"
        )
        f.write(
            "Tropical Atlantic accessed: NO\n\n"
        )

        f.write(
            "TRAJECTORY FEATURE AUDIT\n"
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
            f"decision: {decision}\n"
        )
        f.write(
            f"radial speed residual: {speed_residual_decision}\n"
        )
        f.write(
            f"radial speed residual improvement vs Advect18-Ridge: "
            f"{100.0*radial_improvement_vs_advect:+.3f}%\n"
        )
        f.write(
            f"mean speed residual corr: {speed_corr:+.4f}\n\n"
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
        "12V CANDIDATE SUMMARY"
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
    log(
        f"[RADIAL SPEED RESIDUAL] {speed_residual_decision}"
    )
    log(
        "Radial speed residual improvement vs Advect18-Ridge: "
        f"{100.0*radial_improvement_vs_advect:+.3f}%"
    )
    log(
        f"Mean speed residual corr: {speed_corr:+.4f}"
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
        sys.exit(
            1
        )
