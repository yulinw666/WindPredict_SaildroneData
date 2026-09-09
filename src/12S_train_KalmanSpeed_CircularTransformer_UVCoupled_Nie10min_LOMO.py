# -*- coding: utf-8 -*-
r"""
12S_train_KalmanSpeed_CircularTransformer_UVCoupled_Nie10min_LOMO.py

Stage 12S
=========
Development-only LOMO experiment for a branch-separated but vector-coupled
true-wind predictor under the frozen Nie-aligned protocol:

    10-min resolution
    6 historical points (60 min)
    predict the immediately following +10-min wind

Stage 12R findings motivating this stage
----------------------------------------
1) True-wind-only meteorological input [U,V,T,RH] was at least as strong as
   the old 9-feature input containing vessel/wing variables.

2) A polar Ridge representation improved mean WS and slightly improved mean WD,
   but worsened U/V after independent speed/direction reconstruction.

Therefore Stage 12S asks:

    Can we predict wind SPEED and DIRECTION with specialized mechanisms,
    while coupling them again through reconstructed U/V during training?

IMPORTANT:
    Stage 12S intentionally DOES NOT add a learned wind-speed residual.
    The speed branch is a pure local-linear-trend Kalman forecast.
    Whether a learned speed residual is warranted is diagnosed at the end and
    is reserved for the NEXT stage.

Frozen information
------------------
Base meteorological information:
    U, V, T, RH

Polar sequence supplied to the direction model:
    WS
    sin(WD)
    cos(WD)
    dWS
    dSinWD
    dCosWD
    T
    RH

where:
    WS = sqrt(U^2 + V^2)
    WD is meteorological FROM direction:
        U = -WS*sin(WD)
        V = -WS*cos(WD)

All polar quantities are DERIVED FROM U/V. No additional raw sensor information
is introduced.

Speed branch: Local Linear Trend Kalman
---------------------------------------
State:
    x_k = [speed_level, speed_slope]^T

Transition for one 10-min modeling step:
    F = [[1, 1],
         [0, 1]]

Measurement:
    z_k = [1,0] x_k + measurement_noise

Process covariance:
    Q = q * [[1/3, 1/2],
             [1/2, 1  ]]

Measurement variance:
    R = r

The two positive parameters q and r are estimated ONLY from the two LOMO
training missions by minimizing the one-step Gaussian Kalman negative
log-likelihood over the 6-point historical WS sequences.

After assimilating all six historical observations, the filter forecasts one
more 10-min step:
    WS_hat_KF(t+10)

Negative speed forecasts are clipped to zero and the pre-clip fraction is
reported.

Direction branch
----------------
The direction branch uses a tiny Transformer:

    input dimension = 8
    d_model = 32
    heads = 4
    encoder layers = 1
    FFN dimension = 64

The network predicts a 2-D correction around the LAST OBSERVED direction unit
vector. The output head is initialized to zero:

    raw_dir = last_dir + delta_dir
    dir_hat = raw_dir / ||raw_dir||

Therefore epoch 0 is exact DIRECTION PERSISTENCE, combined with the Kalman speed
forecast.

This avoids an in-sample learned direction anchor and gives a causal, physically
interpretable initialization: the network only needs to learn short-term
turning away from the current direction.

Candidates
----------
A) Met4-UV-Ridge
   [U,V,T,RH] -> [U,V]
   Frozen Stage-12R Cartesian reference.

B) Polar5-Ridge
   [WS,sinWD,cosWD,T,RH]
   separate Ridge speed and circular direction heads.
   Frozen Stage-12R polar reference.

C) KF-DirPersistence
   Kalman speed + last observed wind direction.

D) KF-CircularRidge
   Kalman speed + circular Ridge direction trained on the 8-feature polar
   sequence.

E) KF-TinyTransformer-Circular
   Kalman speed + tiny Transformer direction.
   Loss:
       L = circular_direction_loss + small correction penalty

F) KF-TinyTransformer-Coupled
   Same architecture and speed branch as E.
   Loss:
       L = 0.45 * circular_direction_loss
         + 0.55 * standardized_UV_loss
         + 1e-4 * correction_penalty

The only difference between E and F is the U/V coupling term, allowing a clean
ablation of vector-consistent training.

Development-only LOMO
---------------------
Holdout Antarctic  <- train Atlantic + West Coast
Holdout Atlantic   <- train Antarctic + West Coast
Holdout West Coast <- train Antarctic + Atlantic

Tropical Atlantic is NEVER loaded.

Model-selection score
---------------------
Met4-UV-Ridge is the fold reference:

    Score =
        0.25 * U_RMSE  / Met4_U
      + 0.35 * V_RMSE  / Met4_V
      + 0.25 * WS_RMSE / Met4_WS
      + 0.15 * WD_RMSE / Met4_WD

Predeclared continuation rule
-----------------------------
A Stage-12S specialized model is considered strong enough to continue only if:

    mean Score < 0.99
    >= 2/3 mission wins vs Met4
    mean vector RMSE improves > 0.5%
    and at least one of:
        V RMSE improves > 1%
        WD RMSE improves > 1%

Speed-residual diagnostic
-------------------------
Stage 12S also compares the fixed Kalman speed branch against:
    Met4-UV-Ridge implied WS
    Polar5-Ridge dedicated speed head

If the best direction mechanism is promising but Kalman WS is a clear
bottleneck, the report recommends:
    NEXT_STAGE = ADD_SPEED_RESIDUAL_ON_KALMAN

No speed residual is trained in this script.

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\12S_train_KalmanSpeed_CircularTransformer_UVCoupled_Nie10min_LOMO.py" --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12S_KalmanSpeed_CircularTransformer_Nie10min_LOMO_v0_1"

Smoke test:
    add --debug-fast
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
    from scipy.optimize import minimize
except Exception as exc:
    raise RuntimeError(
        "SciPy is required. Activate the WindPredict environment."
    ) from exc

try:
    from sklearn.linear_model import Ridge
except Exception as exc:
    raise RuntimeError(
        "scikit-learn is required. Activate the WindPredict environment."
    ) from exc


SCRIPT_VERSION = "0.1.0-12S-KalmanSpeed-CircularTransformer-UVCoupled"

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
    / "12S_KalmanSpeed_CircularTransformer_Nie10min_LOMO_v0_1"
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
TEN_MIN_NS = 10 * 60 * 1_000_000_000

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

MET4_INDICES = [
    IDX_U,
    IDX_V,
    IDX_T,
    IDX_RH,
]

DIRECTION_INPUT_NAMES = [
    "WS",
    "WD_sin",
    "WD_cos",
    "dWS",
    "dWD_sin",
    "dWD_cos",
    "T",
    "RH",
]

RIDGE_ALPHA = 1.0
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

# If KF WS is more than 0.5% worse than the Polar dedicated speed head,
# the fixed speed branch is considered a meaningful bottleneck.
SPEED_RESIDUAL_TRIGGER_WORSENING = 0.005

BP_STGNN_PUBLISHED_REFERENCE = {
    "U_RMSE_mps": 0.748640,
    "V_RMSE_mps": 0.705060,
    "WS_RMSE_mps": 0.699400,
    "WD_RMSE_deg": 5.584000,
}


@dataclass(frozen=True)
class TransformerFreeze:
    input_dim: int = 8
    d_model: int = 32
    nhead: int = 4
    num_layers: int = 1
    dim_feedforward: int = 64
    head_hidden: int = 32
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

    circular_weight_coupled: float = 0.45
    uv_weight_coupled: float = 0.55
    correction_penalty: float = 1e-4

    use_amp: bool = True


FREEZE = TransformerFreeze()


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


def state_dict_cpu(
    model,
):
    return {
        k: v.detach()
        .cpu()
        .clone()
        for k, v
        in model.state_dict().items()
    }


def count_parameters(
    model,
):
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
# Frozen Track-J data
# =============================================================================

def load_track_j(
    dataset_dir: Path,
    mission: str,
):
    safe = mission.replace(
        " ",
        "_",
    )

    path = (
        dataset_dir
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
            z[
                "X_raw"
            ],
            dtype=np.float32,
        )

        y = np.asarray(
            z[
                "y_wind_raw"
            ],
            dtype=np.float32,
        )

        context = np.asarray(
            z[
                "context_end_time_ns"
            ],
            dtype=np.int64,
        ).reshape(
            -1
        )

        target = np.asarray(
            z[
                "target_time_ns"
            ],
            dtype=np.int64,
        ).reshape(
            -1
        )

        names = [
            str(x)
            for x in z[
                "feature_names"
            ].tolist()
        ]

        lookback = (
            int(
                np.asarray(
                    z[
                        "lookback_steps"
                    ]
                ).reshape(
                    -1
                )[0]
            )
            if "lookback_steps"
            in z.files
            else X.shape[
                1
            ]
        )

        step_minutes = (
            int(
                np.asarray(
                    z[
                        "step_minutes"
                    ]
                ).reshape(
                    -1
                )[0]
            )
            if "step_minutes"
            in z.files
            else STEP_MINUTES
        )

    if names != FULL9_FEATURE_NAMES:
        raise RuntimeError(
            f"{mission}: unexpected feature names {names}"
        )

    if (
        X.ndim != 3
        or X.shape[
            1:
        ] != (
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
        if y.shape[
            1:
        ] != (
            1,
            2,
        ):
            raise RuntimeError(
                f"{mission}: bad y shape {y.shape}"
            )
        y = y[
            :,
            0,
            :,
        ]

    if (
        y.ndim != 2
        or y.shape[
            1
        ] != 2
    ):
        raise RuntimeError(
            f"{mission}: bad y shape {y.shape}"
        )

    if lookback != LOOKBACK_STEPS:
        raise RuntimeError(
            f"{mission}: lookback={lookback}, expected {LOOKBACK_STEPS}"
        )

    if step_minutes != STEP_MINUTES:
        raise RuntimeError(
            f"{mission}: step_minutes={step_minutes}, expected {STEP_MINUTES}"
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
        np.isfinite(
            X
        ).all()
        and np.isfinite(
            y
        ).all()
    ):
        raise RuntimeError(
            f"{mission}: non-finite values."
        )

    return {
        "X_full9": X,
        "y_uv": y,
        "mission": mission,
        "context_end_time_ns": (
            context
        ),
        "target_time_ns": (
            target
        ),
    }


def concat_missions(
    items,
):
    return {
        "X_full9": np.concatenate(
            [
                x[
                    "X_full9"
                ]
                for x
                in items
            ],
            axis=0,
        ),
        "y_uv": np.concatenate(
            [
                x[
                    "y_uv"
                ]
                for x
                in items
            ],
            axis=0,
        ),
    }


# =============================================================================
# Wind geometry / representations
# =============================================================================

def meteorological_wd_rad_from_uv(
    U,
    V,
):
    return np.arctan2(
        -np.asarray(
            U,
            dtype=np.float64,
        ),
        -np.asarray(
            V,
            dtype=np.float64,
        ),
    )


def circular_diff_rad(
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
            + np.pi
        )
        % (
            2.0
            * np.pi
        )
        - np.pi
    )


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


def make_met4(
    X_full9,
):
    return np.asarray(
        X_full9[
            :,
            :,
            MET4_INDICES
        ],
        dtype=np.float32,
    )


def uv_to_ws_dirunit(
    uv,
):
    uv = np.asarray(
        uv,
        dtype=np.float64,
    )

    U = uv[
        ...,
        0
    ]
    V = uv[
        ...,
        1
    ]

    WS = np.hypot(
        U,
        V,
    )

    theta = (
        meteorological_wd_rad_from_uv(
            U,
            V,
        )
    )

    s = np.sin(
        theta
    )
    c = np.cos(
        theta
    )

    return (
        WS.astype(
            np.float32
        ),
        np.stack(
            [
                s,
                c,
            ],
            axis=-1,
        ).astype(
            np.float32
        ),
    )


def make_polar5(
    X_full9,
):
    X = np.asarray(
        X_full9,
        dtype=np.float32,
    )

    wind_uv = X[
        :,
        :,
        [
            IDX_U,
            IDX_V,
        ]
    ]

    ws, d = (
        uv_to_ws_dirunit(
            wind_uv
        )
    )

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

    return np.stack(
        [
            ws,
            d[
                :,
                :,
                0
            ],
            d[
                :,
                :,
                1
            ],
            T,
            RH,
        ],
        axis=-1,
    ).astype(
        np.float32
    )


def make_direction_sequence(
    X_full9,
):
    """
    8 features:
        WS, sinWD, cosWD, dWS, dSinWD, dCosWD, T, RH
    """
    X = np.asarray(
        X_full9,
        dtype=np.float32,
    )

    wind_uv = X[
        :,
        :,
        [
            IDX_U,
            IDX_V,
        ]
    ]

    ws, d = (
        uv_to_ws_dirunit(
            wind_uv
        )
    )

    dws = np.zeros_like(
        ws,
        dtype=np.float32,
    )

    dd = np.zeros_like(
        d,
        dtype=np.float32,
    )

    dws[
        :,
        1:
    ] = (
        ws[
            :,
            1:
        ]
        - ws[
            :,
            :-1
        ]
    )

    dd[
        :,
        1:,
        :
    ] = (
        d[
            :,
            1:,
            :
        ]
        - d[
            :,
            :-1,
            :
        ]
    )

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

    seq = np.stack(
        [
            ws,
            d[
                :,
                :,
                0
            ],
            d[
                :,
                :,
                1
            ],
            dws,
            dd[
                :,
                :,
                0
            ],
            dd[
                :,
                :,
                1
            ],
            T,
            RH,
        ],
        axis=-1,
    ).astype(
        np.float32
    )

    if seq.shape[
        2
    ] != FREEZE.input_dim:
        raise RuntimeError(
            f"Direction input dimension mismatch: {seq.shape}"
        )

    return seq


def reconstruct_uv(
    speed,
    direction_unit,
):
    speed = np.asarray(
        speed,
        dtype=np.float64,
    ).reshape(
        -1
    )

    direction_unit = np.asarray(
        direction_unit,
        dtype=np.float64,
    )

    s = direction_unit[
        :,
        0
    ]

    c = direction_unit[
        :,
        1
    ]

    U = (
        -speed
        * s
    )

    V = (
        -speed
        * c
    )

    return np.stack(
        [
            U,
            V,
        ],
        axis=1,
    ).astype(
        np.float32
    )


def normalize_direction_unit(
    raw,
    fallback=None,
):
    x = np.asarray(
        raw,
        dtype=np.float64,
    )

    norm = np.linalg.norm(
        x,
        axis=1,
    )

    bad = (
        norm
        < 1e-8
    )

    safe = np.maximum(
        norm,
        1e-8,
    )

    out = (
        x
        / safe[
            :,
            None,
        ]
    )

    if (
        fallback is not None
        and np.any(
            bad
        )
    ):
        fb = np.asarray(
            fallback,
            dtype=np.float64,
        )

        fb_norm = np.maximum(
            np.linalg.norm(
                fb,
                axis=1,
            ),
            1e-8,
        )

        out[
            bad
        ] = (
            fb[
                bad
            ]
            / fb_norm[
                bad,
                None,
            ]
        )

    return (
        out.astype(
            np.float32
        ),
        float(
            np.mean(
                bad
            )
        ),
    )


# =============================================================================
# Fold-only standardization
# =============================================================================

def fit_x_scaler(
    X,
):
    x = np.asarray(
        X,
        dtype=np.float64,
    )

    mean = np.mean(
        x,
        axis=(
            0,
            1,
        ),
    )

    std = np.std(
        x,
        axis=(
            0,
            1,
        ),
        ddof=0,
    )

    std = np.where(
        std
        < 1e-8,
        1.0,
        std,
    )

    return {
        "mean": mean.astype(
            np.float32
        ),
        "std": std.astype(
            np.float32
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
                "mean"
            ][
                None,
                None,
                :,
            ]
        )
        / scaler[
            "std"
        ][
            None,
            None,
            :,
        ]
    ).astype(
        np.float32
    )


def fit_y_scaler(
    y,
):
    y = np.asarray(
        y,
        dtype=np.float64,
    )

    mean = np.mean(
        y,
        axis=0,
    )

    std = np.std(
        y,
        axis=0,
        ddof=0,
    )

    std = np.where(
        std
        < 1e-8,
        1.0,
        std,
    )

    return {
        "mean": mean.astype(
            np.float32
        ),
        "std": std.astype(
            np.float32
        ),
    }


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
                "mean"
            ][
                None,
                :,
            ]
        )
        / scaler[
            "std"
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
            "std"
        ][
            None,
            :,
        ]
        + scaler[
            "mean"
        ][
            None,
            :,
        ]
    ).astype(
        np.float32
    )


def fit_scalar_scaler(
    x,
):
    a = np.asarray(
        x,
        dtype=np.float64,
    ).reshape(
        -1
    )

    mean = float(
        np.mean(
            a
        )
    )

    std = float(
        np.std(
            a,
            ddof=0,
        )
    )

    if std < 1e-8:
        std = 1.0

    return {
        "mean": mean,
        "std": std,
    }


def transform_scalar(
    x,
    scaler,
):
    return (
        (
            np.asarray(
                x,
                dtype=np.float32,
            ).reshape(
                -1
            )
            - float(
                scaler[
                    "mean"
                ]
            )
        )
        / float(
            scaler[
                "std"
            ]
        )
    ).astype(
        np.float32
    )


# =============================================================================
# Metrics
# =============================================================================

def meteorological_wd_deg_from_uv(
    uv,
):
    uv = np.asarray(
        uv,
        dtype=np.float64,
    )

    return (
        np.degrees(
            meteorological_wd_rad_from_uv(
                uv[
                    :,
                    0
                ],
                uv[
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

    u_rmse = float(
        np.sqrt(
            np.mean(
                err[
                    :,
                    0
                ] ** 2
            )
        )
    )

    v_rmse = float(
        np.sqrt(
            np.mean(
                err[
                    :,
                    1
                ] ** 2
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

    ws_true = np.linalg.norm(
        yt,
        axis=1,
    )

    ws_pred = np.linalg.norm(
        yp,
        axis=1,
    )

    ws_rmse = float(
        np.sqrt(
            np.mean(
                (
                    ws_pred
                    - ws_true
                ) ** 2
            )
        )
    )

    wd_true = (
        meteorological_wd_deg_from_uv(
            yt
        )
    )

    wd_pred = (
        meteorological_wd_deg_from_uv(
            yp
        )
    )

    wd_err = (
        circular_diff_deg(
            wd_pred,
            wd_true,
        )
    )

    return {
        "wind_U_RMSE_mps": (
            u_rmse
        ),
        "wind_V_RMSE_mps": (
            v_rmse
        ),
        "wind_vector_RMSE_mps": (
            vector_rmse
        ),
        "wind_speed_RMSE_mps": (
            ws_rmse
        ),
        "wind_direction_MAE_deg": float(
            np.mean(
                np.abs(
                    wd_err
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
    reference,
):
    ratios = {
        "U_ratio": (
            metrics[
                "wind_U_RMSE_mps"
            ]
            / max(
                reference[
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
                reference[
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
                reference[
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
                reference[
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
    ).reshape(
        -1
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    ).reshape(
        -1
    )

    mask = (
        np.isfinite(
            a
        )
        & np.isfinite(
            b
        )
    )

    a = a[
        mask
    ]

    b = b[
        mask
    ]

    if len(
        a
    ) < 3:
        return np.nan

    if (
        np.std(
            a
        )
        < EPS
        or np.std(
            b
        )
        < EPS
    ):
        return np.nan

    return float(
        np.corrcoef(
            a,
            b,
        )[
            0,
            1
        ]
    )


def direction_turn_diagnostics(
    X_full9,
    y_true,
    y_pred,
):
    X = np.asarray(
        X_full9,
        dtype=np.float64,
    )

    last_theta = (
        meteorological_wd_rad_from_uv(
            X[
                :,
                -1,
                IDX_U
            ],
            X[
                :,
                -1,
                IDX_V
            ],
        )
    )

    true_theta = (
        meteorological_wd_rad_from_uv(
            y_true[
                :,
                0
            ],
            y_true[
                :,
                1
            ],
        )
    )

    pred_theta = (
        meteorological_wd_rad_from_uv(
            y_pred[
                :,
                0
            ],
            y_pred[
                :,
                1
            ],
        )
    )

    true_turn = (
        circular_diff_rad(
            true_theta,
            last_theta,
        )
    )

    pred_turn = (
        circular_diff_rad(
            pred_theta,
            last_theta,
        )
    )

    err = (
        circular_diff_rad(
            pred_turn,
            true_turn,
        )
    )

    return {
        "direction_turn_corr": (
            corrcoef_safe(
                true_turn,
                pred_turn,
            )
        ),
        "direction_turn_MAE_deg": float(
            np.degrees(
                np.mean(
                    np.abs(
                        err
                    )
                )
            )
        ),
        "direction_turn_RMSE_deg": float(
            np.degrees(
                np.sqrt(
                    np.mean(
                        err ** 2
                    )
                )
            )
        ),
        "pred_turn_abs_deg_mean": float(
            np.degrees(
                np.mean(
                    np.abs(
                        pred_turn
                    )
                )
            )
        ),
        "true_turn_abs_deg_mean": float(
            np.degrees(
                np.mean(
                    np.abs(
                        true_turn
                    )
                )
            )
        ),
    }


# =============================================================================
# Ridge baselines
# =============================================================================

def fit_predict_cartesian_ridge(
    X_train,
    y_train,
    X_val,
):
    xs = fit_x_scaler(
        X_train
    )

    ys = fit_y_scaler(
        y_train
    )

    Xtr = transform_X(
        X_train,
        xs,
    ).reshape(
        len(
            X_train
        ),
        -1,
    )

    Xva = transform_X(
        X_val,
        xs,
    ).reshape(
        len(
            X_val
        ),
        -1,
    )

    ytr = transform_y(
        y_train,
        ys,
    )

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        Xtr,
        ytr,
    )

    pred = inverse_y(
        model.predict(
            Xva
        ),
        ys,
    )

    params = int(
        np.asarray(
            model.coef_
        ).size
        + np.asarray(
            model.intercept_
        ).size
    )

    return (
        pred.astype(
            np.float32
        ),
        params,
    )


def fit_predict_polar5_ridge(
    X_train_full9,
    y_train,
    X_val_full9,
):
    Xtr_p = make_polar5(
        X_train_full9
    )

    Xva_p = make_polar5(
        X_val_full9
    )

    xs = fit_x_scaler(
        Xtr_p
    )

    Xtr = transform_X(
        Xtr_p,
        xs,
    ).reshape(
        len(
            Xtr_p
        ),
        -1,
    )

    Xva = transform_X(
        Xva_p,
        xs,
    ).reshape(
        len(
            Xva_p
        ),
        -1,
    )

    ws_train, dir_train = (
        uv_to_ws_dirunit(
            y_train
        )
    )

    ws_scaler = fit_y_scaler(
        ws_train.reshape(
            -1,
            1,
        )
    )

    speed_model = Ridge(
        alpha=RIDGE_ALPHA
    )

    direction_model = Ridge(
        alpha=RIDGE_ALPHA
    )

    speed_model.fit(
        Xtr,
        transform_y(
            ws_train.reshape(
                -1,
                1,
            ),
            ws_scaler,
        ),
    )

    direction_model.fit(
        Xtr,
        dir_train,
    )

    ws_pred = inverse_y(
        speed_model.predict(
            Xva
        ),
        ws_scaler,
    ).reshape(
        -1
    )

    ws_pred = np.maximum(
        ws_pred,
        0.0,
    )

    last_dir = (
        uv_to_ws_dirunit(
            X_val_full9[
                :,
                -1,
                [
                    IDX_U,
                    IDX_V,
                ]
            ]
        )[
            1
        ]
    )

    direction_raw = (
        direction_model.predict(
            Xva
        )
    )

    direction_pred, fallback_fraction = (
        normalize_direction_unit(
            direction_raw,
            fallback=(
                last_dir
            ),
        )
    )

    pred_uv = reconstruct_uv(
        ws_pred,
        direction_pred,
    )

    params = int(
        np.asarray(
            speed_model.coef_
        ).size
        + np.asarray(
            speed_model.intercept_
        ).size
        + np.asarray(
            direction_model.coef_
        ).size
        + np.asarray(
            direction_model.intercept_
        ).size
    )

    return (
        pred_uv,
        params,
        {
            "polar_direction_fallback_fraction": (
                fallback_fraction
            ),
            "polar_speed_prediction": (
                ws_pred.astype(
                    np.float32
                )
            ),
        },
    )


# =============================================================================
# Local-linear-trend Kalman speed branch
# =============================================================================

KALMAN_F = np.asarray(
    [
        [
            1.0,
            1.0,
        ],
        [
            0.0,
            1.0,
        ],
    ],
    dtype=np.float64,
)

KALMAN_Q_SHAPE = np.asarray(
    [
        [
            1.0
            / 3.0,
            0.5,
        ],
        [
            0.5,
            1.0,
        ],
    ],
    dtype=np.float64,
)


def speed_history_from_X(
    X_full9,
):
    uv = np.asarray(
        X_full9[
            :,
            :,
            [
                IDX_U,
                IDX_V,
            ]
        ],
        dtype=np.float64,
    )

    return np.linalg.norm(
        uv,
        axis=2,
    )


def kalman_nll_batch(
    log_params,
    ws_sequences,
):
    q = float(
        np.exp(
            log_params[
                0
            ]
        )
    )

    r = float(
        np.exp(
            log_params[
                1
            ]
        )
    )

    y = np.asarray(
        ws_sequences,
        dtype=np.float64,
    )

    if (
        not np.isfinite(
            q
        )
        or not np.isfinite(
            r
        )
        or q <= 0.0
        or r <= 0.0
    ):
        return 1e30

    d = np.diff(
        y,
        axis=1,
    )

    base_var = max(
        float(
            np.var(
                d
            )
        ),
        1e-4,
    )

    level = y[
        :,
        0
    ].copy()

    slope = np.zeros(
        len(
            y
        ),
        dtype=np.float64,
    )

    # Shared covariance because q/r and observation schedule are shared.
    P = np.asarray(
        [
            [
                max(
                    r,
                    1e-6,
                ),
                0.0,
            ],
            [
                0.0,
                max(
                    base_var,
                    1e-6,
                ),
            ],
        ],
        dtype=np.float64,
    )

    Q = (
        q
        * KALMAN_Q_SHAPE
    )

    nll = 0.0
    count = 0

    for k in range(
        1,
        y.shape[
            1
        ],
    ):
        # Predict state.
        level_pred = (
            level
            + slope
        )

        slope_pred = (
            slope
        )

        P_pred = (
            KALMAN_F
            @ P
            @ KALMAN_F.T
            + Q
        )

        innovation = (
            y[
                :,
                k
            ]
            - level_pred
        )

        S = float(
            P_pred[
                0,
                0
            ]
            + r
        )

        if (
            not np.isfinite(
                S
            )
            or S <= 1e-12
        ):
            return 1e30

        nll += 0.5 * float(
            np.sum(
                np.log(
                    2.0
                    * np.pi
                    * S
                )
                + (
                    innovation ** 2
                    / S
                )
            )
        )

        count += len(
            y
        )

        K0 = (
            P_pred[
                0,
                0
            ]
            / S
        )

        K1 = (
            P_pred[
                1,
                0
            ]
            / S
        )

        level = (
            level_pred
            + K0
            * innovation
        )

        slope = (
            slope_pred
            + K1
            * innovation
        )

        # Joseph form is unnecessary for scalar H here; use analytic update.
        P = P_pred.copy()

        first_col = (
            P_pred[
                :,
                0
            ].copy()
        )

        P[
            0,
            :
        ] = (
            P_pred[
                0,
                :
            ]
            - K0
            * P_pred[
                0,
                :
            ]
        )

        P[
            1,
            :
        ] = (
            P_pred[
                1,
                :
            ]
            - K1
            * P_pred[
                0,
                :
            ]
        )

        # Symmetrize to suppress numerical drift.
        P = (
            0.5
            * (
                P
                + P.T
            )
        )

    return float(
        nll
        / max(
            count,
            1,
        )
    )


def fit_kalman_speed_params(
    ws_train,
):
    y = np.asarray(
        ws_train,
        dtype=np.float64,
    )

    if (
        y.ndim != 2
        or y.shape[
            1
        ] != LOOKBACK_STEPS
    ):
        raise ValueError(
            f"Bad Kalman WS training shape: {y.shape}"
        )

    d = np.diff(
        y,
        axis=1,
    )

    var_d = max(
        float(
            np.var(
                d
            )
        ),
        1e-4,
    )

    q0 = max(
        0.10
        * var_d,
        1e-5,
    )

    r0 = max(
        0.50
        * var_d,
        1e-5,
    )

    initial = np.log(
        [
            q0,
            r0,
        ]
    )

    bounds = [
        (
            math.log(
                1e-8
            ),
            math.log(
                100.0
            ),
        ),
        (
            math.log(
                1e-8
            ),
            math.log(
                100.0
            ),
        ),
    ]

    result = minimize(
        kalman_nll_batch,
        initial,
        args=(
            y,
        ),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 100,
            "ftol": 1e-10,
        },
    )

    if (
        not result.success
        or not np.isfinite(
            result.fun
        )
    ):
        log(
            "  [KALMAN WARNING] optimizer did not fully converge; "
            "using best finite parameters returned."
        )

    q = float(
        np.exp(
            result.x[
                0
            ]
        )
    )

    r = float(
        np.exp(
            result.x[
                1
            ]
        )
    )

    return {
        "q": q,
        "r": r,
        "objective_nll_per_obs": float(
            result.fun
        ),
        "optimizer_success": bool(
            result.success
        ),
        "optimizer_message": str(
            result.message
        ),
        "optimizer_nit": int(
            getattr(
                result,
                "nit",
                -1,
            )
        ),
        "train_diff_variance": (
            var_d
        ),
    }


def kalman_forecast_speed(
    ws_sequences,
    params,
):
    y = np.asarray(
        ws_sequences,
        dtype=np.float64,
    )

    q = float(
        params[
            "q"
        ]
    )

    r = float(
        params[
            "r"
        ]
    )

    d = np.diff(
        y,
        axis=1,
    )

    base_var = max(
        float(
            np.var(
                d
            )
        ),
        1e-4,
    )

    level = y[
        :,
        0
    ].copy()

    slope = np.zeros(
        len(
            y
        ),
        dtype=np.float64,
    )

    P = np.asarray(
        [
            [
                max(
                    r,
                    1e-6,
                ),
                0.0,
            ],
            [
                0.0,
                max(
                    base_var,
                    1e-6,
                ),
            ],
        ],
        dtype=np.float64,
    )

    Q = (
        q
        * KALMAN_Q_SHAPE
    )

    for k in range(
        1,
        y.shape[
            1
        ],
    ):
        level_pred = (
            level
            + slope
        )

        slope_pred = (
            slope
        )

        P_pred = (
            KALMAN_F
            @ P
            @ KALMAN_F.T
            + Q
        )

        innovation = (
            y[
                :,
                k
            ]
            - level_pred
        )

        S = float(
            P_pred[
                0,
                0
            ]
            + r
        )

        K0 = (
            P_pred[
                0,
                0
            ]
            / S
        )

        K1 = (
            P_pred[
                1,
                0
            ]
            / S
        )

        level = (
            level_pred
            + K0
            * innovation
        )

        slope = (
            slope_pred
            + K1
            * innovation
        )

        P = P_pred.copy()

        P[
            0,
            :
        ] = (
            P_pred[
                0,
                :
            ]
            - K0
            * P_pred[
                0,
                :
            ]
        )

        P[
            1,
            :
        ] = (
            P_pred[
                1,
                :
            ]
            - K1
            * P_pred[
                0,
                :
            ]
        )

        P = (
            0.5
            * (
                P
                + P.T
            )
        )

    # One step into the future, no measurement update.
    forecast_raw = (
        level
        + slope
    )

    negative_fraction = float(
        np.mean(
            forecast_raw
            < 0.0
        )
    )

    forecast = np.maximum(
        forecast_raw,
        0.0,
    )

    return (
        forecast.astype(
            np.float32
        ),
        {
            "kalman_negative_speed_fraction_preclip": (
                negative_fraction
            ),
            "kalman_forecast_speed_mean_mps": float(
                np.mean(
                    forecast
                )
            ),
            "kalman_filtered_slope_mean_mps_per_10min": float(
                np.mean(
                    slope
                )
            ),
            "kalman_filtered_slope_std_mps_per_10min": float(
                np.std(
                    slope,
                    ddof=0,
                )
            ),
        },
    )


# =============================================================================
# Circular Ridge direction branch
# =============================================================================

def fit_direction_ridge(
    X_train_direction,
    y_train_uv,
):
    xs = fit_x_scaler(
        X_train_direction
    )

    Xtr = transform_X(
        X_train_direction,
        xs,
    ).reshape(
        len(
            X_train_direction
        ),
        -1,
    )

    _, target_dir = (
        uv_to_ws_dirunit(
            y_train_uv
        )
    )

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        Xtr,
        target_dir,
    )

    return (
        model,
        xs,
    )


def predict_direction_ridge(
    model,
    xs,
    X_direction,
    last_dir,
):
    Xz = transform_X(
        X_direction,
        xs,
    ).reshape(
        len(
            X_direction
        ),
        -1,
    )

    raw = np.asarray(
        model.predict(
            Xz
        ),
        dtype=np.float32,
    )

    unit, fallback = (
        normalize_direction_unit(
            raw,
            fallback=(
                last_dir
            ),
        )
    )

    return (
        unit,
        fallback,
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
# Tiny Transformer direction correction model
# =============================================================================

def build_transformer_model_class(
    torch,
    nn,
):
    class TinyDirectionTransformer(nn.Module):
        def __init__(
            self,
        ):
            super().__init__()

            self.input_proj = (
                nn.Linear(
                    FREEZE.input_dim,
                    FREEZE.d_model,
                )
            )

            self.positional = (
                nn.Parameter(
                    torch.zeros(
                        1,
                        LOOKBACK_STEPS,
                        FREEZE.d_model,
                    )
                )
            )

            layer = (
                nn.TransformerEncoderLayer(
                    d_model=FREEZE.d_model,
                    nhead=FREEZE.nhead,
                    dim_feedforward=(
                        FREEZE.dim_feedforward
                    ),
                    dropout=(
                        FREEZE.dropout
                    ),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )

            self.encoder = (
                nn.TransformerEncoder(
                    layer,
                    num_layers=(
                        FREEZE.num_layers
                    ),
                )
            )

            # Context:
            #   final transformer token
            #   last observed direction unit (2)
            #   standardized Kalman speed (1)
            self.head = nn.Sequential(
                nn.Linear(
                    FREEZE.d_model
                    + 2
                    + 1,
                    FREEZE.head_hidden,
                ),
                nn.LayerNorm(
                    FREEZE.head_hidden
                ),
                nn.GELU(),
                nn.Dropout(
                    FREEZE.dropout
                ),
            )

            self.delta_out = (
                nn.Linear(
                    FREEZE.head_hidden,
                    2,
                )
            )

            # Epoch 0 = exact direction persistence.
            nn.init.zeros_(
                self.delta_out.weight
            )
            nn.init.zeros_(
                self.delta_out.bias
            )

        def forward(
            self,
            x,
            last_dir,
            kf_speed_z,
        ):
            h = (
                self.input_proj(
                    x
                )
                + self.positional
            )

            h = self.encoder(
                h
            )

            h_last = h[
                :,
                -1,
                :
            ]

            z = torch.cat(
                [
                    h_last,
                    last_dir,
                    kf_speed_z.unsqueeze(
                        -1
                    ),
                ],
                dim=1,
            )

            hidden = self.head(
                z
            )

            delta = self.delta_out(
                hidden
            )

            raw_dir = (
                last_dir
                + delta
            )

            norm = torch.linalg.vector_norm(
                raw_dir,
                dim=1,
                keepdim=True,
            ).clamp_min(
                1e-8
            )

            unit_dir = (
                raw_dir
                / norm
            )

            return (
                unit_dir,
                delta,
            )

    return TinyDirectionTransformer


def evaluate_transformer(
    *,
    torch,
    model,
    X_direction,
    direction_scaler,
    last_dir,
    kf_speed,
    kf_speed_scaler,
    y_true,
    X_full9,
    device,
    amp_enabled,
):
    model.eval()

    Xz = transform_X(
        X_direction,
        direction_scaler,
    )

    kf_z = transform_scalar(
        kf_speed,
        kf_speed_scaler,
    )

    chunks_dir = []
    chunks_delta = []

    with torch.no_grad():
        for (
            xb_np,
            lb_np,
            kb_np,
        ) in sequential_batches(
            Xz,
            last_dir,
            kf_z,
            batch_size=(
                FREEZE.batch_size
            ),
        ):
            xb = torch.from_numpy(
                xb_np
            ).to(
                device,
                non_blocking=True,
            )

            lb = torch.from_numpy(
                lb_np
            ).to(
                device,
                non_blocking=True,
            )

            kb = torch.from_numpy(
                kb_np
            ).to(
                device,
                non_blocking=True,
            )

            with amp_context(
                torch,
                amp_enabled,
                device,
            ):
                pred_dir, delta = (
                    model(
                        xb,
                        lb,
                        kb,
                    )
                )

            chunks_dir.append(
                pred_dir.detach()
                .float()
                .cpu()
                .numpy()
            )

            chunks_delta.append(
                delta.detach()
                .float()
                .cpu()
                .numpy()
            )

    pred_dir = np.concatenate(
        chunks_dir,
        axis=0,
    ).astype(
        np.float32
    )

    delta = np.concatenate(
        chunks_delta,
        axis=0,
    ).astype(
        np.float32
    )

    pred_uv = reconstruct_uv(
        kf_speed,
        pred_dir,
    )

    metrics = evaluate_wind(
        y_true,
        pred_uv,
    )

    metrics.update(
        direction_turn_diagnostics(
            X_full9,
            y_true,
            pred_uv,
        )
    )

    metrics[
        "direction_delta_RMS"
    ] = float(
        np.sqrt(
            np.mean(
                delta ** 2
            )
        )
    )

    return (
        metrics,
        pred_dir,
        pred_uv,
    )


def train_transformer_run(
    *,
    torch,
    nn,
    candidate,
    coupled,
    seed,
    held_out,
    X_train_full9,
    X_val_full9,
    X_train_direction,
    X_val_direction,
    y_train,
    y_val,
    kf_speed_train,
    kf_speed_val,
    met4_val_metrics,
    output_dir,
    device,
    debug_fast,
):
    seed_everything(
        torch,
        seed,
    )

    direction_scaler = (
        fit_x_scaler(
            X_train_direction
        )
    )

    Xtr_z = transform_X(
        X_train_direction,
        direction_scaler,
    )

    _, train_dir = (
        uv_to_ws_dirunit(
            y_train
        )
    )

    _, val_dir = (
        uv_to_ws_dirunit(
            y_val
        )
    )

    _, hist_dir_train = (
        uv_to_ws_dirunit(
            X_train_full9[
                :,
                -1,
                [
                    IDX_U,
                    IDX_V,
                ]
            ]
        )
    )

    _, hist_dir_val = (
        uv_to_ws_dirunit(
            X_val_full9[
                :,
                -1,
                [
                    IDX_U,
                    IDX_V,
                ]
            ]
        )
    )

    kf_scaler = fit_scalar_scaler(
        kf_speed_train
    )

    kf_train_z = transform_scalar(
        kf_speed_train,
        kf_scaler,
    )

    # Fold-train U/V scale for vector-coupled loss.
    y_scale = np.std(
        np.asarray(
            y_train,
            dtype=np.float64,
        ),
        axis=0,
        ddof=0,
    )

    y_scale = np.where(
        y_scale
        < 1e-6,
        1.0,
        y_scale,
    ).astype(
        np.float32
    )

    ModelClass = (
        build_transformer_model_class(
            torch,
            nn,
        )
    )

    model = ModelClass().to(
        device
    )

    parameter_count = (
        count_parameters(
            model
        )
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=(
                FREEZE.learning_rate
            ),
            weight_decay=(
                FREEZE.weight_decay
            ),
        )
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
        and device.type
        == "cuda"
    )

    grad_scaler = (
        create_grad_scaler(
            torch,
            amp_enabled,
        )
    )

    # Epoch 0: exact direction persistence + Kalman speed.
    initial_metrics, _, _ = (
        evaluate_transformer(
            torch=torch,
            model=model,
            X_direction=(
                X_val_direction
            ),
            direction_scaler=(
                direction_scaler
            ),
            last_dir=(
                hist_dir_val
            ),
            kf_speed=(
                kf_speed_val
            ),
            kf_speed_scaler=(
                kf_scaler
            ),
            y_true=y_val,
            X_full9=(
                X_val_full9
            ),
            device=device,
            amp_enabled=(
                amp_enabled
            ),
        )
    )

    best_score, best_parts = (
        selection_score(
            initial_metrics,
            met4_val_metrics,
        )
    )

    best_epoch = 0
    best_state = (
        state_dict_cpu(
            model
        )
    )
    best_metrics = dict(
        initial_metrics
    )
    best_parts_saved = dict(
        best_parts
    )

    history = [
        {
            "candidate": (
                candidate
            ),
            "held_out_mission": (
                held_out
            ),
            "seed": int(
                seed
            ),
            "epoch_one_based": 0,
            "learning_rate": float(
                FREEZE.learning_rate
            ),
            "train_total_loss": np.nan,
            "train_circular_loss": np.nan,
            "train_uv_loss": np.nan,
            "train_delta_penalty": np.nan,
            "selection_score": float(
                best_score
            ),
            "checkpoint_improved": True,
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
        }
    ]

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
    started = (
        time.perf_counter()
    )

    y_scale_t = torch.from_numpy(
        y_scale
    ).to(
        device
    )

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        model.train()

        total_sum = 0.0
        circ_sum = 0.0
        uv_sum = 0.0
        penalty_sum = 0.0
        n_seen = 0

        for (
            xb_np,
            lb_np,
            kb_np,
            td_np,
            yb_np,
            kfraw_np,
        ) in sequential_batches(
            Xtr_z,
            hist_dir_train,
            kf_train_z,
            train_dir,
            y_train,
            kf_speed_train,
            batch_size=(
                FREEZE.batch_size
            ),
        ):
            xb = torch.from_numpy(
                xb_np
            ).to(
                device,
                non_blocking=True,
            )

            lb = torch.from_numpy(
                lb_np
            ).to(
                device,
                non_blocking=True,
            )

            kb = torch.from_numpy(
                kb_np
            ).to(
                device,
                non_blocking=True,
            )

            td = torch.from_numpy(
                td_np
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

            kfraw = torch.from_numpy(
                np.asarray(
                    kfraw_np,
                    dtype=np.float32,
                )
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
                pred_dir, delta = (
                    model(
                        xb,
                        lb,
                        kb,
                    )
                )

                dot = torch.sum(
                    pred_dir
                    * td,
                    dim=1,
                )

                circ_loss = torch.mean(
                    1.0
                    - dot
                )

                pred_u = (
                    -kfraw
                    * pred_dir[
                        :,
                        0
                    ]
                )

                pred_v = (
                    -kfraw
                    * pred_dir[
                        :,
                        1
                    ]
                )

                pred_uv = torch.stack(
                    [
                        pred_u,
                        pred_v,
                    ],
                    dim=1,
                )

                uv_loss = torch.mean(
                    (
                        (
                            pred_uv
                            - yb
                        )
                        / y_scale_t[
                            None,
                            :
                        ]
                    ) ** 2
                )

                delta_penalty = (
                    torch.mean(
                        delta ** 2
                    )
                )

                if coupled:
                    total = (
                        FREEZE.circular_weight_coupled
                        * circ_loss
                        + FREEZE.uv_weight_coupled
                        * uv_loss
                        + FREEZE.correction_penalty
                        * delta_penalty
                    )
                else:
                    total = (
                        circ_loss
                        + FREEZE.correction_penalty
                        * delta_penalty
                    )

            if not torch.isfinite(
                total
            ):
                raise FloatingPointError(
                    "Non-finite Stage-12S loss."
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

            bn = len(
                xb_np
            )

            total_sum += float(
                total.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            circ_sum += float(
                circ_loss.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            uv_sum += float(
                uv_loss.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            penalty_sum += float(
                delta_penalty.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            n_seen += bn

        val_metrics, _, _ = (
            evaluate_transformer(
                torch=torch,
                model=model,
                X_direction=(
                    X_val_direction
                ),
                direction_scaler=(
                    direction_scaler
                ),
                last_dir=(
                    hist_dir_val
                ),
                kf_speed=(
                    kf_speed_val
                ),
                kf_speed_scaler=(
                    kf_scaler
                ),
                y_true=y_val,
                X_full9=(
                    X_val_full9
                ),
                device=device,
                amp_enabled=(
                    amp_enabled
                ),
            )
        )

        score, parts = (
            selection_score(
                val_metrics,
                met4_val_metrics,
            )
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

            best_state = (
                state_dict_cpu(
                    model
                )
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

        lr = float(
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        scheduler.step(
            score
        )

        history.append(
            {
                "candidate": (
                    candidate
                ),
                "held_out_mission": (
                    held_out
                ),
                "seed": int(
                    seed
                ),
                "epoch_one_based": int(
                    epoch
                ),
                "learning_rate": lr,
                "train_total_loss": (
                    total_sum
                    / max(
                        n_seen,
                        1,
                    )
                ),
                "train_circular_loss": (
                    circ_sum
                    / max(
                        n_seen,
                        1,
                    )
                ),
                "train_uv_loss": (
                    uv_sum
                    / max(
                        n_seen,
                        1,
                    )
                ),
                "train_delta_penalty": (
                    penalty_sum
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
            }
        )

        log(
            f"      ep={epoch:03d} | "
            f"score={score:.5f} | "
            f"U={val_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={val_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={val_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={val_metrics['wind_direction_RMSE_deg']:.3f} | "
            f"turnCorr={val_metrics['direction_turn_corr']:+.3f}"
            + (
                " *"
                if improved
                else ""
            )
        )

        if patience >= patience_limit:
            log(
                f"      early stop after {patience} epochs "
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

    confirmed, _, _ = (
        evaluate_transformer(
            torch=torch,
            model=model,
            X_direction=(
                X_val_direction
            ),
            direction_scaler=(
                direction_scaler
            ),
            last_dir=(
                hist_dir_val
            ),
            kf_speed=(
                kf_speed_val
            ),
            kf_speed_scaler=(
                kf_scaler
            ),
            y_true=y_val,
            X_full9=(
                X_val_full9
            ),
            device=device,
            amp_enabled=(
                amp_enabled
            ),
        )
    )

    confirmed_score, confirmed_parts = (
        selection_score(
            confirmed,
            met4_val_metrics,
        )
    )

    if abs(
        confirmed_score
        - best_score
    ) > 1e-6:
        raise RuntimeError(
            "Best Stage-12S checkpoint re-evaluation mismatch."
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

    torch.save(
        {
            "stage": "12S",
            "script_version": (
                SCRIPT_VERSION
            ),
            "candidate": (
                candidate
            ),
            "coupled": bool(
                coupled
            ),
            "held_out_mission": (
                held_out
            ),
            "seed": int(
                seed
            ),
            "task": {
                "resolution_minutes": (
                    STEP_MINUTES
                ),
                "lookback_steps": (
                    LOOKBACK_STEPS
                ),
                "forecast_minutes": (
                    FORECAST_MINUTES
                ),
            },
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
            "direction_scaler": (
                direction_scaler
            ),
            "kf_speed_scaler": (
                kf_scaler
            ),
            "transformer_freeze": (
                asdict(
                    FREEZE
                )
            ),
            "Tropical_Atlantic_used": False,
            "speed_residual_used": False,
        },
        checkpoint_path,
    )

    result = {
        "candidate": (
            candidate
        ),
        "held_out_mission": (
            held_out
        ),
        "seed": int(
            seed
        ),
        "status": (
            "completed_finite"
        ),
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
# CSV / summary
# =============================================================================

def append_csv(
    path: Path,
    row: dict,
):
    new = pd.DataFrame(
        [
            row
        ]
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
                    ].astype(
                        str
                    )
                    == str(
                        row[
                            "candidate"
                        ]
                    )
                )
                & (
                    old[
                        "held_out_mission"
                    ].astype(
                        str
                    )
                    == str(
                        row[
                            "held_out_mission"
                        ]
                    )
                )
                & (
                    old[
                        "seed"
                    ].astype(
                        int
                    )
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


def summarize_candidate(
    runs,
    candidate,
):
    sub = runs.loc[
        (
            runs[
                "candidate"
            ].astype(
                str
            )
            == candidate
        )
        & (
            runs[
                "status"
            ].astype(
                str
            )
            == "completed_finite"
        )
    ].copy()

    if sub.empty:
        raise RuntimeError(
            f"No completed rows for {candidate}."
        )

    row = {
        "candidate": (
            candidate
        ),
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
        "parameter_count": int(
            round(
                sub[
                    "parameter_count"
                ].mean()
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
        "direction_turn_corr",
        "direction_turn_MAE_deg",
        "direction_turn_RMSE_deg",
        "pred_turn_abs_deg_mean",
        "true_turn_abs_deg_mean",
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
        "mission_wins_vs_Met4"
    ] = int(
        np.sum(
            mission_scores.to_numpy()
            < 1.0
        )
    )

    return row


# =============================================================================
# Main experiment
# =============================================================================

def main():
    parser = (
        argparse.ArgumentParser()
    )

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
            "coupled Transformer only, <=8 epochs."
        ),
    )

    args = parser.parse_args()

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

    missions = {
        m: load_track_j(
            dataset_dir,
            m,
        )
        for m in DEVELOPMENT_MISSIONS
    }

    torch, nn = (
        import_torch()
    )

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

    log(
        "="
        * 132
    )
    log(
        "12S — KALMAN SPEED + CIRCULAR TINY TRANSFORMER + U/V COUPLING"
    )
    log(
        "="
        * 132
    )
    log(
        f"dataset         : {dataset_dir}"
    )
    log(
        f"output          : {output_dir}"
    )
    log(
        "task            : 6 x 10-min history -> next +10-min wind"
    )
    log(
        "true-wind info  : U,V,T,RH only (polar equivalents in direction branch)"
    )
    log(
        "speed branch    : local-linear-trend Kalman only; NO learned speed residual"
    )
    log(
        "direction branch: tiny Transformer, d_model=32, heads=4, layers=1"
    )
    log(
        f"device          : {device}"
    )

    if device.type == "cuda":
        log(
            f"GPU             : {torch.cuda.get_device_name(0)}"
        )

    log(
        "[FIREWALL] Antarctic / Atlantic / West Coast only. "
        "Tropical Atlantic is not loaded."
    )
    log("")

    run_path = (
        output_dir
        / "candidate_run_results.csv"
    )

    kalman_rows = []

    for held_out in (
        active_holdouts
    ):
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

        val = (
            missions[
                held_out
            ]
        )

        Xtr_full = train[
            "X_full9"
        ]

        Xva_full = val[
            "X_full9"
        ]

        ytr = train[
            "y_uv"
        ]

        yva = val[
            "y_uv"
        ]

        log(
            "-"
            * 132
        )

        log(
            f"[{held_out}] "
            f"Ntrain={len(Xtr_full):,} | "
            f"Nval={len(Xva_full):,}"
        )

        # -------------------------------------------------------------
        # A) Met4-UV-Ridge reference
        # -------------------------------------------------------------
        Xtr_met4 = make_met4(
            Xtr_full
        )

        Xva_met4 = make_met4(
            Xva_full
        )

        met4_pred, met4_params = (
            fit_predict_cartesian_ridge(
                Xtr_met4,
                ytr,
                Xva_met4,
            )
        )

        met4_metrics = (
            evaluate_wind(
                yva,
                met4_pred,
            )
        )

        met4_metrics.update(
            direction_turn_diagnostics(
                Xva_full,
                yva,
                met4_pred,
            )
        )

        met4_row = {
            "candidate": (
                "Met4-UV-Ridge"
            ),
            "held_out_mission": (
                held_out
            ),
            "seed": -1,
            "status": (
                "completed_finite"
            ),
            "parameter_count": int(
                met4_params
            ),
            "best_epoch_one_based": 0,
            "selection_score": 1.0,
            "elapsed_seconds": 0.0,
            **met4_metrics,
            "score_U_ratio": 1.0,
            "score_V_ratio": 1.0,
            "score_WS_ratio": 1.0,
            "score_WD_ratio": 1.0,
            "history_path": "",
            "checkpoint_path": "",
        }

        append_csv(
            run_path,
            met4_row,
        )

        # -------------------------------------------------------------
        # B) Polar5-Ridge reference
        # -------------------------------------------------------------
        (
            polar_pred,
            polar_params,
            polar_extra,
        ) = fit_predict_polar5_ridge(
            Xtr_full,
            ytr,
            Xva_full,
        )

        polar_metrics = (
            evaluate_wind(
                yva,
                polar_pred,
            )
        )

        polar_metrics.update(
            direction_turn_diagnostics(
                Xva_full,
                yva,
                polar_pred,
            )
        )

        polar_score, polar_parts = (
            selection_score(
                polar_metrics,
                met4_metrics,
            )
        )

        polar_row = {
            "candidate": (
                "Polar5-Ridge"
            ),
            "held_out_mission": (
                held_out
            ),
            "seed": -1,
            "status": (
                "completed_finite"
            ),
            "parameter_count": int(
                polar_params
            ),
            "best_epoch_one_based": 0,
            "selection_score": float(
                polar_score
            ),
            "elapsed_seconds": 0.0,
            **polar_metrics,
            **{
                f"score_{k}": v
                for k, v
                in polar_parts.items()
            },
            "history_path": "",
            "checkpoint_path": "",
        }

        append_csv(
            run_path,
            polar_row,
        )

        # -------------------------------------------------------------
        # Speed branch: fit Kalman on fold training missions only.
        # -------------------------------------------------------------
        ws_train_history = (
            speed_history_from_X(
                Xtr_full
            )
        )

        ws_val_history = (
            speed_history_from_X(
                Xva_full
            )
        )

        kalman_params = (
            fit_kalman_speed_params(
                ws_train_history
            )
        )

        (
            kf_speed_train,
            kf_train_diag,
        ) = kalman_forecast_speed(
            ws_train_history,
            kalman_params,
        )

        (
            kf_speed_val,
            kf_val_diag,
        ) = kalman_forecast_speed(
            ws_val_history,
            kalman_params,
        )

        true_ws_val = (
            np.linalg.norm(
                yva,
                axis=1,
            )
        )

        kf_ws_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        kf_speed_val
                        - true_ws_val
                    ) ** 2
                )
            )
        )

        polar_ws_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        polar_extra[
                            "polar_speed_prediction"
                        ]
                        - true_ws_val
                    ) ** 2
                )
            )
        )

        kalman_rows.append(
            {
                "held_out_mission": (
                    held_out
                ),
                "q": (
                    kalman_params[
                        "q"
                    ]
                ),
                "r": (
                    kalman_params[
                        "r"
                    ]
                ),
                "objective_nll_per_obs": (
                    kalman_params[
                        "objective_nll_per_obs"
                    ]
                ),
                "optimizer_success": (
                    kalman_params[
                        "optimizer_success"
                    ]
                ),
                "optimizer_nit": (
                    kalman_params[
                        "optimizer_nit"
                    ]
                ),
                "train_diff_variance": (
                    kalman_params[
                        "train_diff_variance"
                    ]
                ),
                "KF_WS_RMSE_mps": (
                    kf_ws_rmse
                ),
                "PolarRidge_WS_RMSE_mps": (
                    polar_ws_rmse
                ),
                "Met4_WS_RMSE_mps": (
                    met4_metrics[
                        "wind_speed_RMSE_mps"
                    ]
                ),
                "KF_vs_PolarRidge_WS_improvement_fraction": (
                    polar_ws_rmse
                    - kf_ws_rmse
                )
                / max(
                    polar_ws_rmse,
                    EPS,
                ),
                "KF_vs_Met4_WS_improvement_fraction": (
                    met4_metrics[
                        "wind_speed_RMSE_mps"
                    ]
                    - kf_ws_rmse
                )
                / max(
                    met4_metrics[
                        "wind_speed_RMSE_mps"
                    ],
                    EPS,
                ),
                **{
                    f"train_{k}": v
                    for k, v
                    in kf_train_diag.items()
                },
                **{
                    f"val_{k}": v
                    for k, v
                    in kf_val_diag.items()
                },
            }
        )

        log(
            "  Speed branch       : "
            f"KF_WS={kf_ws_rmse:.4f} | "
            f"PolarRidge_WS={polar_ws_rmse:.4f} | "
            f"Met4_WS={met4_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"q={kalman_params['q']:.5g} | "
            f"r={kalman_params['r']:.5g}"
        )

        # -------------------------------------------------------------
        # C) KF + direction persistence
        # -------------------------------------------------------------
        _, last_dir_val = (
            uv_to_ws_dirunit(
                Xva_full[
                    :,
                    -1,
                    [
                        IDX_U,
                        IDX_V,
                    ]
                ]
            )
        )

        persistence_pred = (
            reconstruct_uv(
                kf_speed_val,
                last_dir_val,
            )
        )

        persistence_metrics = (
            evaluate_wind(
                yva,
                persistence_pred,
            )
        )

        persistence_metrics.update(
            direction_turn_diagnostics(
                Xva_full,
                yva,
                persistence_pred,
            )
        )

        (
            persistence_score,
            persistence_parts,
        ) = selection_score(
            persistence_metrics,
            met4_metrics,
        )

        persistence_row = {
            "candidate": (
                "KF-DirPersistence"
            ),
            "held_out_mission": (
                held_out
            ),
            "seed": -1,
            "status": (
                "completed_finite"
            ),
            "parameter_count": 2,
            "best_epoch_one_based": 0,
            "selection_score": float(
                persistence_score
            ),
            "elapsed_seconds": 0.0,
            **persistence_metrics,
            **{
                f"score_{k}": v
                for k, v
                in persistence_parts.items()
            },
            "history_path": "",
            "checkpoint_path": "",
        }

        append_csv(
            run_path,
            persistence_row,
        )

        # -------------------------------------------------------------
        # D) KF + circular Ridge direction
        # -------------------------------------------------------------
        Xtr_dir = (
            make_direction_sequence(
                Xtr_full
            )
        )

        Xva_dir = (
            make_direction_sequence(
                Xva_full
            )
        )

        (
            dir_ridge,
            dir_ridge_scaler,
        ) = fit_direction_ridge(
            Xtr_dir,
            ytr,
        )

        _, last_dir_train = (
            uv_to_ws_dirunit(
                Xtr_full[
                    :,
                    -1,
                    [
                        IDX_U,
                        IDX_V,
                    ]
                ]
            )
        )

        dir_ridge_val, dir_fallback = (
            predict_direction_ridge(
                dir_ridge,
                dir_ridge_scaler,
                Xva_dir,
                last_dir_val,
            )
        )

        kf_ridge_pred = (
            reconstruct_uv(
                kf_speed_val,
                dir_ridge_val,
            )
        )

        kf_ridge_metrics = (
            evaluate_wind(
                yva,
                kf_ridge_pred,
            )
        )

        kf_ridge_metrics.update(
            direction_turn_diagnostics(
                Xva_full,
                yva,
                kf_ridge_pred,
            )
        )

        (
            kf_ridge_score,
            kf_ridge_parts,
        ) = selection_score(
            kf_ridge_metrics,
            met4_metrics,
        )

        kf_ridge_row = {
            "candidate": (
                "KF-CircularRidge"
            ),
            "held_out_mission": (
                held_out
            ),
            "seed": -1,
            "status": (
                "completed_finite"
            ),
            "parameter_count": int(
                ridge_parameter_count(
                    dir_ridge
                )
                + 2
            ),
            "best_epoch_one_based": 0,
            "selection_score": float(
                kf_ridge_score
            ),
            "elapsed_seconds": 0.0,
            "direction_fallback_fraction": (
                dir_fallback
            ),
            **kf_ridge_metrics,
            **{
                f"score_{k}": v
                for k, v
                in kf_ridge_parts.items()
            },
            "history_path": "",
            "checkpoint_path": "",
        }

        append_csv(
            run_path,
            kf_ridge_row,
        )

        log(
            "  Met4-UV-Ridge      : "
            f"score=1.00000 | "
            f"U={met4_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={met4_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={met4_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={met4_metrics['wind_direction_RMSE_deg']:.3f}"
        )

        log(
            "  Polar5-Ridge       : "
            f"score={polar_score:.5f} | "
            f"U={polar_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={polar_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={polar_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={polar_metrics['wind_direction_RMSE_deg']:.3f}"
        )

        log(
            "  KF-Persistence     : "
            f"score={persistence_score:.5f} | "
            f"U={persistence_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={persistence_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={persistence_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={persistence_metrics['wind_direction_RMSE_deg']:.3f}"
        )

        log(
            "  KF-CircularRidge   : "
            f"score={kf_ridge_score:.5f} | "
            f"U={kf_ridge_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={kf_ridge_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={kf_ridge_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={kf_ridge_metrics['wind_direction_RMSE_deg']:.3f}"
        )

        # -------------------------------------------------------------
        # E/F) Tiny Transformer direction candidates.
        # -------------------------------------------------------------
        neural_specs = (
            [
                (
                    "KF-TinyTransformer-Coupled",
                    True,
                )
            ]
            if args.debug_fast
            else [
                (
                    "KF-TinyTransformer-Circular",
                    False,
                ),
                (
                    "KF-TinyTransformer-Coupled",
                    True,
                ),
            ]
        )

        for (
            candidate,
            coupled,
        ) in neural_specs:
            for seed in (
                active_seeds
            ):
                log(
                    f"  [TRAIN] {candidate} | seed={seed}"
                )

                result = (
                    train_transformer_run(
                        torch=torch,
                        nn=nn,
                        candidate=(
                            candidate
                        ),
                        coupled=(
                            coupled
                        ),
                        seed=(
                            seed
                        ),
                        held_out=(
                            held_out
                        ),
                        X_train_full9=(
                            Xtr_full
                        ),
                        X_val_full9=(
                            Xva_full
                        ),
                        X_train_direction=(
                            Xtr_dir
                        ),
                        X_val_direction=(
                            Xva_dir
                        ),
                        y_train=(
                            ytr
                        ),
                        y_val=(
                            yva
                        ),
                        kf_speed_train=(
                            kf_speed_train
                        ),
                        kf_speed_val=(
                            kf_speed_val
                        ),
                        met4_val_metrics=(
                            met4_metrics
                        ),
                        output_dir=(
                            output_dir
                        ),
                        device=(
                            device
                        ),
                        debug_fast=(
                            args.debug_fast
                        ),
                    )
                )

                append_csv(
                    run_path,
                    result,
                )

                log(
                    f"  [DONE] "
                    f"score={result['selection_score']:.5f} | "
                    f"best_ep={result['best_epoch_one_based']} | "
                    f"U={result['wind_U_RMSE_mps']:.4f} | "
                    f"V={result['wind_V_RMSE_mps']:.4f} | "
                    f"WS={result['wind_speed_RMSE_mps']:.4f} | "
                    f"WD={result['wind_direction_RMSE_deg']:.3f} | "
                    f"turnCorr={result['direction_turn_corr']:+.3f}"
                )

        del (
            train,
            val,
            Xtr_full,
            Xva_full,
            ytr,
            yva,
            Xtr_met4,
            Xva_met4,
            met4_pred,
            polar_pred,
            ws_train_history,
            ws_val_history,
            kf_speed_train,
            kf_speed_val,
            Xtr_dir,
            Xva_dir,
            dir_ridge,
        )

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    kalman_df = pd.DataFrame(
        kalman_rows
    )

    kalman_path = (
        output_dir
        / "kalman_fold_parameters.csv"
    )

    kalman_df.to_csv(
        kalman_path,
        index=False,
        encoding="utf-8-sig",
    )

    if args.debug_fast:
        log("")
        log(
            "[DONE] debug-fast completed. "
            "No scientific final selection written."
        )
        return 0

    # =================================================================
    # Final development-only summary.
    # =================================================================
    runs = pd.read_csv(
        run_path
    )

    expected_counts = {
        "Met4-UV-Ridge": 3,
        "Polar5-Ridge": 3,
        "KF-DirPersistence": 3,
        "KF-CircularRidge": 3,
        "KF-TinyTransformer-Circular": 9,
        "KF-TinyTransformer-Coupled": 9,
    }

    for candidate, expected in (
        expected_counts.items()
    ):
        got = int(
            np.sum(
                (
                    runs[
                        "candidate"
                    ].astype(
                        str
                    )
                    == candidate
                )
                & (
                    runs[
                        "status"
                    ].astype(
                        str
                    )
                    == "completed_finite"
                )
            )
        )

        if got != expected:
            raise RuntimeError(
                f"{candidate}: expected {expected} completed rows, got {got}."
            )

    candidate_order = [
        "Met4-UV-Ridge",
        "Polar5-Ridge",
        "KF-DirPersistence",
        "KF-CircularRidge",
        "KF-TinyTransformer-Circular",
        "KF-TinyTransformer-Coupled",
    ]

    summary = pd.DataFrame(
        [
            summarize_candidate(
                runs,
                c,
            )
            for c in candidate_order
        ]
    )

    met4_runs = runs.loc[
        runs[
            "candidate"
        ].astype(
            str
        )
        == "Met4-UV-Ridge"
    ].copy()

    met4_means = {
        k: float(
            met4_runs[
                k
            ].mean()
        )
        for k in [
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
            ].astype(
                str
            )
            == candidate
        ]

        for metric, reference_value in (
            met4_means.items()
        ):
            mean_value = float(
                sub[
                    metric
                ].mean()
            )

            summary.loc[
                idx,
                f"{metric}_improvement_vs_Met4_fraction",
            ] = (
                reference_value
                - mean_value
            ) / max(
                reference_value,
                EPS,
            )

    summary = summary.sort_values(
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

    raw_best = (
        summary.iloc[
            0
        ]
    )

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
            "mission_wins_vs_Met4"
        ]
    )

    raw_best_vector_imp = float(
        raw_best[
            "wind_vector_RMSE_mps_improvement_vs_Met4_fraction"
        ]
    )

    raw_best_v_imp = float(
        raw_best[
            "wind_V_RMSE_mps_improvement_vs_Met4_fraction"
        ]
    )

    raw_best_wd_imp = float(
        raw_best[
            "wind_direction_RMSE_deg_improvement_vs_Met4_fraction"
        ]
    )

    specialized = (
        raw_best_name
        not in {
            "Met4-UV-Ridge",
            "Polar5-Ridge",
        }
    )

    meets_rule = bool(
        specialized
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
        selected_name = (
            raw_best_name
        )

        decision = (
            "CONTINUE_SPECIALIZED_SPEED_DIRECTION_MODEL"
        )
    else:
        selected_name = (
            "Met4-UV-Ridge"
        )

        decision = (
            "RETAIN_MET4_RIDGE_REFERENCE"
        )

    # -----------------------------------------------------------------
    # Diagnose whether speed residual should be the NEXT mechanism.
    # -----------------------------------------------------------------
    kf_ws_mean = float(
        kalman_df[
            "KF_WS_RMSE_mps"
        ].mean()
    )

    polar_ws_mean = float(
        kalman_df[
            "PolarRidge_WS_RMSE_mps"
        ].mean()
    )

    met4_ws_mean = float(
        kalman_df[
            "Met4_WS_RMSE_mps"
        ].mean()
    )

    kf_vs_polar_improvement = (
        polar_ws_mean
        - kf_ws_mean
    ) / max(
        polar_ws_mean,
        EPS,
    )

    kf_vs_met4_improvement = (
        met4_ws_mean
        - kf_ws_mean
    ) / max(
        met4_ws_mean,
        EPS,
    )

    speed_is_bottleneck = bool(
        kf_vs_polar_improvement
        < -SPEED_RESIDUAL_TRIGGER_WORSENING
    )

    # Direction progress is judged from the best model using the KF speed.
    kf_family = summary.loc[
        summary[
            "candidate"
        ].astype(
            str
        ).str.startswith(
            "KF-"
        )
    ].copy()

    best_kf_row = (
        kf_family.sort_values(
            [
                "selection_score_mean",
                "parameter_count",
            ]
        ).iloc[
            0
        ]
    )

    best_kf_name = str(
        best_kf_row[
            "candidate"
        ]
    )

    best_kf_wd_imp = float(
        best_kf_row[
            "wind_direction_RMSE_deg_improvement_vs_Met4_fraction"
        ]
    )

    direction_has_signal = bool(
        best_kf_wd_imp
        > 0.005
        or (
            "direction_turn_corr_mean"
            in best_kf_row.index
            and np.isfinite(
                best_kf_row[
                    "direction_turn_corr_mean"
                ]
            )
            and float(
                best_kf_row[
                    "direction_turn_corr_mean"
                ]
            )
            > 0.10
        )
    )

    if (
        speed_is_bottleneck
        and direction_has_signal
    ):
        next_recommendation = (
            "ADD_SPEED_RESIDUAL_ON_KALMAN"
        )
    elif speed_is_bottleneck:
        next_recommendation = (
            "SPEED_BRANCH_IS_WEAK_BUT_DIRECTION_SIGNAL_IS_ALSO_LIMITED"
        )
    elif direction_has_signal:
        next_recommendation = (
            "KEEP_KALMAN_SPEED_AND_REFINE_DIRECTION_OR_COUPLING"
        )
    else:
        next_recommendation = (
            "RETHINK_BOTH_SPEED_AND_DIRECTION_MECHANISMS"
        )

    selected_runs = runs.loc[
        runs[
            "candidate"
        ].astype(
            str
        )
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
        "stage": "12S",
        "script_version": (
            SCRIPT_VERSION
        ),
        "task": {
            "resolution_minutes": (
                STEP_MINUTES
            ),
            "lookback_steps": (
                LOOKBACK_STEPS
            ),
            "lookback_minutes": (
                LOOKBACK_STEPS
                * STEP_MINUTES
            ),
            "forecast_minutes": (
                FORECAST_MINUTES
            ),
        },
        "input_information": (
            "true-wind meteorological information only: U,V,T,RH; "
            "polar/gradient quantities are deterministic transforms"
        ),
        "transformer_freeze": (
            asdict(
                FREEZE
            )
        ),
        "score_weights": (
            SCORE_WEIGHTS
        ),
        "predeclared_rule": {
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
        "best_KF_family_candidate": (
            best_kf_name
        ),
        "speed_branch_diagnostic": {
            "KF_WS_RMSE_mean": (
                kf_ws_mean
            ),
            "PolarRidge_WS_RMSE_mean": (
                polar_ws_mean
            ),
            "Met4_WS_RMSE_mean": (
                met4_ws_mean
            ),
            "KF_improvement_vs_PolarRidge_fraction": (
                kf_vs_polar_improvement
            ),
            "KF_improvement_vs_Met4_fraction": (
                kf_vs_met4_improvement
            ),
            "speed_is_bottleneck": (
                speed_is_bottleneck
            ),
            "speed_residual_trigger_worsening_fraction": (
                SPEED_RESIDUAL_TRIGGER_WORSENING
            ),
        },
        "direction_has_signal": (
            direction_has_signal
        ),
        "next_stage_recommendation": (
            next_recommendation
        ),
        "development_only": True,
        "development_missions": (
            DEVELOPMENT_MISSIONS
        ),
        "Tropical_Atlantic_used": False,
        "speed_residual_used": False,
        "published_BP_STGNN_reference_context_only": (
            BP_STGNN_PUBLISHED_REFERENCE
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
        / "12S_REPORT.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "12S Kalman Speed + Circular Tiny Transformer + U/V Coupling\n"
        )
        f.write(
            "="
            * 132
            + "\n\n"
        )

        f.write(
            "FROZEN TASK\n"
        )
        f.write(
            "-"
            * 132
            + "\n"
        )
        f.write(
            "resolution: 10 min\n"
        )
        f.write(
            "history: 6 points = 60 min\n"
        )
        f.write(
            "target: next +10-min wind\n"
        )
        f.write(
            "input information: U,V,T,RH only\n"
        )
        f.write(
            "speed residual used: NO\n"
        )
        f.write(
            "Tropical Atlantic accessed: NO\n\n"
        )

        f.write(
            "KALMAN SPEED PARAMETERS\n"
        )
        f.write(
            "-"
            * 132
            + "\n"
        )
        f.write(
            kalman_df.to_string(
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
            "-"
            * 132
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
            "-"
            * 132
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
            f"best KF-family candidate: {best_kf_name}\n"
        )
        f.write(
            f"direction has signal: {direction_has_signal}\n\n"
        )

        f.write(
            "SPEED-BRANCH DIAGNOSTIC\n"
        )
        f.write(
            "-"
            * 132
            + "\n"
        )
        f.write(
            f"KF WS RMSE mean: {kf_ws_mean:.6f}\n"
        )
        f.write(
            f"Polar-Ridge WS RMSE mean: {polar_ws_mean:.6f}\n"
        )
        f.write(
            f"Met4 WS RMSE mean: {met4_ws_mean:.6f}\n"
        )
        f.write(
            f"KF vs Polar improvement: "
            f"{100.0*kf_vs_polar_improvement:+.3f}%\n"
        )
        f.write(
            f"KF vs Met4 improvement: "
            f"{100.0*kf_vs_met4_improvement:+.3f}%\n"
        )
        f.write(
            f"speed branch bottleneck: {speed_is_bottleneck}\n"
        )
        f.write(
            f"NEXT_STAGE_RECOMMENDATION: {next_recommendation}\n\n"
        )

        f.write(
            "Published BP-STGNN reference values are context only:\n"
        )
        for k, v in (
            BP_STGNN_PUBLISHED_REFERENCE.items()
        ):
            f.write(
                f"{k}: {v}\n"
            )

    log("")
    log(
        "="
        * 132
    )
    log(
        "12S CANDIDATE SUMMARY"
    )
    log(
        "="
        * 132
    )
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
        f"[BEST KF FAMILY] {best_kf_name}"
    )
    log("")
    log(
        "Speed branch diagnostic: "
        f"KF_WS={kf_ws_mean:.6f} | "
        f"PolarRidge_WS={polar_ws_mean:.6f} | "
        f"Met4_WS={met4_ws_mean:.6f}"
    )
    log(
        "KF relative to Polar-Ridge speed head: "
        f"{100.0*kf_vs_polar_improvement:+.3f}%"
    )
    log(
        f"[SPEED BOTTLENECK] {speed_is_bottleneck}"
    )
    log(
        f"[DIRECTION SIGNAL] {direction_has_signal}"
    )
    log(
        f"[NEXT STAGE] {next_recommendation}"
    )
    log("")
    log(
        "Published BP-STGNN reference (context only): "
        "U=0.74864 | V=0.70506 | WS=0.69940 | WD=5.584 deg"
    )
    log("")
    log(
        f"[SAVED] {run_path}"
    )
    log(
        f"[SAVED] {summary_path}"
    )
    log(
        f"[SAVED] {kalman_path}"
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
        sys.exit(
            1
        )
