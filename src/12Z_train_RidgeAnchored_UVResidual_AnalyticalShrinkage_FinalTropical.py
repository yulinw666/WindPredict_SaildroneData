# -*- coding: utf-8 -*-
r"""
12Z_train_RidgeAnchored_UVResidual_AnalyticalShrinkage_FinalTropical.py

Stage 12Z
=========
Strict BP-aligned Ridge-anchored U/V residual learning with analytical
component-wise shrinkage.

CORE IDEA
---------
The strong Base4-Ridge is the shortcut / anchor:

    W_R = [U_R, V_R]

The neural network NEVER learns U,V from scratch.
It only learns:

    r_U = U_true - U_R
    r_V = V_true - V_R

Final:

    U_hat = U_R + alpha_U * rhat_U
    V_hat = V_R + alpha_V * rhat_V

This is the exact residual-learning idea motivated by ResNet:
learning a correction near zero is easier than relearning the whole mapping.

ZERO-SAFE INITIALIZATION
------------------------
Both final residual heads are initialized to exactly zero.

Therefore:
    epoch 0 raw residual prediction = [0,0]
    epoch 0 final prediction = exact Ridge

ANALYTICAL SHRINKAGE
--------------------
For a frozen set of residual predictions rhat on development data:

    alpha_U* =
        clip(
            sum((U_true-U_R) * rhat_U)
            / sum(rhat_U^2),
            0,
            ALPHA_MAX
        )

and similarly for V.

Because alpha=0 is always allowed, on the DATA USED FOR CALIBRATION the
component MSE cannot be worse than Ridge.

IMPORTANT:
This is NOT a mathematical guarantee on an unseen distribution such as another
mission.  It is a conservative development-calibrated safeguard.

STRICT BP-ALIGNED TASK
----------------------
Source:
    Stage 12G point-sampled dataset.

Input history:
    six original 1-min-mean observations sampled every 10 min:
        t-50, t-40, t-30, t-20, t-10, t

Target:
    original 1-min-mean U,V observation at t+10 min.

No:
    10-min arithmetic block averaging
    intermediate 1-min records
    pressure
    wing angle
    absolute position
    trajectory reconstruction

RIDGE ANCHOR
------------
Input:
    [U,V,T,RH] x 6

Output:
    [U_R,V_R]

RESIDUAL NETWORK INPUT
----------------------
Per-step sequence, 10 features:
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

Per-sample summary, 10 features:
    latest_U
    latest_V
    latest_dU
    latest_dV
    std_U6
    std_V6
    std_WS6
    slope_U6
    slope_V6
    slope_WS6

Ridge context, 3 features:
    U_R
    V_R
    WS_R

All residual inputs are either BP-owned U,V,T,RH samples or deterministic
transforms of them. No extra observation is introduced.

RESIDUAL NETWORK
----------------
One fixed compact 2-layer GRU:
    sequence -> GRU(hidden=48, layers=2)

Then:
    GRU last state + 10 summary + 3 Ridge context
        -> 48
        -> 32

Separate residual heads:
    U head: 32 -> 16 -> 1
    V head: 32 -> 16 -> 1

The two final head layers are zero-initialized.

Raw bounded residual:
    rhat_j =
        RESIDUAL_CAP_STD
        * sigma_residual_j
        * tanh(z_j)

Final correction:
    alpha_j * rhat_j

Frozen:
    RESIDUAL_CAP_STD = 1.0
    0 <= alpha_U,alpha_V <= 0.75

TRAINING TARGET
---------------
Residual targets are generated with MISSION-PRESERVING 5-fold blocked OOF
Ridge predictions:

    r_train =
        y_train - Ridge_OOF(X_train)

Never:
    y_train - Ridge_in_sample(X_train)

LOSS
----
Train directly on standardized U and V residuals:

    L =
        0.45 * MSE(rhat_U/std_U, r_U/std_U)
      + 0.55 * MSE(rhat_V/std_V, r_V/std_V)
      + 0.002 * mean((rhat/std)^2)

The V branch receives modestly higher weight because it has been the more
difficult component in the development experiments.

DEVELOPMENT
-----------
Only:
    Antarctic
    Atlantic
    West Coast

Outer LOMO:
    Antarctic holdout
    Atlantic holdout
    West Coast holdout

Seeds:
    500043
    501052
    502061

Every epoch:
    - predict raw residual on the held-out development mission
    - analytically compute fold alpha_U, alpha_V
    - evaluate Ridge + alpha * residual
    - checkpoint best validation score

Validation score:
    0.35 * U_ratio
  + 0.50 * V_ratio
  + 0.10 * WS_ratio
  + 0.05 * WD_ratio

Since the requested primary targets are U,V, they dominate selection.

FROZEN FINAL CALIBRATION
------------------------
1) Select seed from mean 3-mission LOMO score.
2) Final epoch = median best epoch for selected seed.
3) Concatenate the three held-out development predictions of selected seed.
4) Analytically calculate ONE frozen:
       alpha_U
       alpha_V
   from the full development out-of-mission predictions.
5) Save:
       FROZEN_BEFORE_TROPICAL.json
6) Only then load Tropical Atlantic.

FINAL TRAIN / TROPICAL
----------------------
Final Base4-Ridge:
    train Antarctic + Atlantic + West Coast.

Final residual GRU:
    train on all three development missions using OOF Ridge anchors,
    for the frozen epoch count.

Tropical:
    raw neural residual is multiplied by the FROZEN development alpha_U/V.
    Tropical labels are never used to alter alpha, network, epoch, or seed.

Tropical filename:
    first tries Tropical_Atlantic_TEST.npz
    then legacy Tropical_Atlantic.npz.

PUBLISHED BP-STGNN REFERENCE
----------------------------
    U  = 0.748640 m/s
    V  = 0.705060 m/s
    WS = 0.699400 m/s
    WD = 5.584 deg
    params = 242580

INTEGRITY
---------
Tropical Atlantic was already observed in earlier project experiments.
Therefore it must not later be described as a never-seen pristine test set.
Stage 12Z itself does not use Tropical labels for selection/calibration.

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\12Z_train_RidgeAnchored_UVResidual_AnalyticalShrinkage_FinalTropical.py" --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12Z_RidgeAnchored_UVResidual_Shrinkage_FinalTropical_v0_1"

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


SCRIPT_VERSION = "0.1.0-12Z-RidgeUVResidual-AnalyticalShrinkage"

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
    / "12Z_RidgeAnchored_UVResidual_Shrinkage_FinalTropical_v0_1"
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

RIDGE_ALPHA = 1.0
INNER_CROSSFIT_FOLDS = 5
EPS = 1e-12

SEEDS = [
    500043,
    501052,
    502061,
]

RESIDUAL_CAP_STD = 1.0
ALPHA_MAX = 0.75

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
    dense1: int = 48
    dense2: int = 32
    head_hidden: int = 16
    dropout: float = 0.10

    learning_rate: float = 6e-4
    weight_decay: float = 2e-5
    batch_size: int = 256

    max_epochs: int = 250
    early_stop_patience: int = 30
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    min_learning_rate: float = 8e-6
    grad_clip_norm: float = 1.0

    weight_u: float = 0.45
    weight_v: float = 0.55
    weight_residual_size: float = 0.002

    use_amp: bool = True


FREEZE = Freeze()

SEQ_FEATURE_NAMES = [
    "U",
    "V",
    "T",
    "RH",
    "dU",
    "dV",
    "ddU",
    "ddV",
    "WS",
    "dWS",
]

SUMMARY_FEATURE_NAMES = [
    "latest_U",
    "latest_V",
    "latest_dU",
    "latest_dV",
    "std_U6",
    "std_V6",
    "std_WS6",
    "slope_U6",
    "slope_V6",
    "slope_WS6",
]


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
        torch.cuda.manual_seed_all(
            int(seed)
        )


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
        "Could not find track_J_joint_compatible."
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

    attempted = "\n".join(
        f"  - {p}"
        for p in candidates
    )

    raise FileNotFoundError(
        f"Missing mission dataset for {mission}. Tried:\n{attempted}"
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
            f"{mission}: unexpected feature names {names}"
        )

    if (
        X.ndim != 3
        or X.shape[1:] != (
            LOOKBACK_STEPS,
            len(FULL9_FEATURE_NAMES),
        )
    ):
        raise RuntimeError(
            f"{mission}: unexpected X shape {X.shape}"
        )

    if y.ndim == 3:
        if y.shape[1:] != (1, 2):
            raise RuntimeError(
                f"{mission}: unexpected y shape {y.shape}"
            )
        y = y[:, 0, :]

    if (
        y.ndim != 2
        or y.shape[1] != 2
    ):
        raise RuntimeError(
            f"{mission}: unexpected y shape {y.shape}"
        )

    if not np.all(
        target - context
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
                    [x["mission"]]
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


def ridge_transform_X(X, scaler):
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


def ridge_transform_y(y, scaler):
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


def ridge_inverse_y(yz, scaler):
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
    pred_z = np.asarray(
        model.predict(
            ridge_transform_X(
                X,
                scaler,
            ).reshape(
                len(X),
                -1,
            )
        ),
        dtype=np.float32,
    )

    return ridge_inverse_y(
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
# OOF Ridge
# =============================================================================

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
            INNER_CROSSFIT_FOLDS,
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
        va = (
            fold_ids == k
        )

        tr = ~va

        model, scaler = fit_ridge(
            X_base4[
                tr
            ],
            y[
                tr
            ],
        )

        pred[
            va
        ] = ridge_predict(
            model,
            scaler,
            X_base4[
                va
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
# Residual features
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

    t -= np.mean(t)

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
            centered
            * t[
                None,
                :
            ],
            axis=1,
        )
        / max(
            float(
                np.sum(
                    t ** 2
                )
            ),
            EPS,
        )
    )


def build_residual_features(
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

    U = X[
        :,
        :,
        IDX_U,
    ]

    V = X[
        :,
        :,
        IDX_V,
    ]

    T = X[
        :,
        :,
        IDX_T,
    ]

    RH = X[
        :,
        :,
        IDX_RH,
    ]

    WS = np.hypot(
        U,
        V,
    )

    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dWS = np.zeros_like(WS)

    dU[
        :,
        1:
    ] = (
        U[
            :,
            1:
        ]
        - U[
            :,
            :-1
        ]
    )

    dV[
        :,
        1:
    ] = (
        V[
            :,
            1:
        ]
        - V[
            :,
            :-1
        ]
    )

    dWS[
        :,
        1:
    ] = (
        WS[
            :,
            1:
        ]
        - WS[
            :,
            :-1
        ]
    )

    ddU = np.zeros_like(U)
    ddV = np.zeros_like(V)

    ddU[
        :,
        2:
    ] = (
        dU[
            :,
            2:
        ]
        - dU[
            :,
            1:-1
        ]
    )

    ddV[
        :,
        2:
    ] = (
        dV[
            :,
            2:
        ]
        - dV[
            :,
            1:-1
        ]
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
            U[
                :,
                -1
            ],
            V[
                :,
                -1
            ],
            dU[
                :,
                -1
            ],
            dV[
                :,
                -1
            ],
            np.std(
                U,
                axis=1,
                ddof=0,
            ),
            np.std(
                V,
                axis=1,
                ddof=0,
            ),
            np.std(
                WS,
                axis=1,
                ddof=0,
            ),
            slope6(U),
            slope6(V),
            slope6(WS),
        ],
        axis=1,
    ).astype(
        np.float32
    )

    anchor_context = np.concatenate(
        [
            ridge_anchor,
            np.linalg.norm(
                ridge_anchor,
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
        and np.isfinite(
            anchor_context
        ).all()
    ):
        raise RuntimeError(
            "Nonfinite residual features."
        )

    return {
        "seq": seq,
        "summary": summary,
        "anchor": anchor_context,
    }


# =============================================================================
# Scaling
# =============================================================================

def fit_residual_scalers(
    features,
    y_true,
    ridge_anchor,
):
    seq = np.asarray(
        features[
            "seq"
        ],
        dtype=np.float64,
    )

    summary = np.asarray(
        features[
            "summary"
        ],
        dtype=np.float64,
    )

    anchor = np.asarray(
        features[
            "anchor"
        ],
        dtype=np.float64,
    )

    residual = (
        np.asarray(
            y_true,
            dtype=np.float64,
        )
        - np.asarray(
            ridge_anchor,
            dtype=np.float64,
        )
    )

    def fit_ms(
        arr,
        axis,
    ):
        mean = np.mean(
            arr,
            axis=axis,
        )

        std = np.std(
            arr,
            axis=axis,
            ddof=0,
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

    seq_mean, seq_std = fit_ms(
        seq,
        (0, 1),
    )

    summary_mean, summary_std = fit_ms(
        summary,
        0,
    )

    anchor_mean, anchor_std = fit_ms(
        anchor,
        0,
    )

    residual_std = np.std(
        residual,
        axis=0,
        ddof=0,
    )

    residual_std = np.where(
        residual_std < 1e-4,
        1.0,
        residual_std,
    )

    return {
        "seq_mean": seq_mean,
        "seq_std": seq_std,
        "summary_mean": (
            summary_mean
        ),
        "summary_std": (
            summary_std
        ),
        "anchor_mean": (
            anchor_mean
        ),
        "anchor_std": (
            anchor_std
        ),
        "residual_std": (
            residual_std.astype(
                np.float32
            )
        ),
    }


def transform_residual_features(
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
        "anchor_z": (
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
# Metrics / shrinkage
# =============================================================================

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

    wd_err = circular_diff_deg(
        wd_from_uv(yp),
        wd_from_uv(yt),
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
                        ws_p
                        - ws_t
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
    ridge_metrics,
):
    return float(
        0.35
        * (
            metrics[
                "wind_U_RMSE_mps"
            ]
            / max(
                ridge_metrics[
                    "wind_U_RMSE_mps"
                ],
                EPS,
            )
        )
        + 0.50
        * (
            metrics[
                "wind_V_RMSE_mps"
            ]
            / max(
                ridge_metrics[
                    "wind_V_RMSE_mps"
                ],
                EPS,
            )
        )
        + 0.10
        * (
            metrics[
                "wind_speed_RMSE_mps"
            ]
            / max(
                ridge_metrics[
                    "wind_speed_RMSE_mps"
                ],
                EPS,
            )
        )
        + 0.05
        * (
            metrics[
                "wind_direction_RMSE_deg"
            ]
            / max(
                ridge_metrics[
                    "wind_direction_RMSE_deg"
                ],
                EPS,
            )
        )
    )


def analytical_alpha_1d(
    true_residual,
    predicted_residual,
    alpha_max=ALPHA_MAX,
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

    if (
        denom <= EPS
        or not np.isfinite(
            denom
        )
    ):
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
            float(
                alpha_max
            ),
        )
    )


def analytical_alphas(
    y_true,
    ridge_pred,
    raw_residual_pred,
):
    residual_true = (
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
        raw_residual_pred,
        dtype=np.float64,
    )

    return np.asarray(
        [
            analytical_alpha_1d(
                residual_true[
                    :,
                    0
                ],
                raw[
                    :,
                    0
                ],
            ),
            analytical_alpha_1d(
                residual_true[
                    :,
                    1
                ],
                raw[
                    :,
                    1
                ],
            ),
        ],
        dtype=np.float32,
    )


def apply_shrinkage(
    ridge_pred,
    raw_residual_pred,
    alphas,
):
    return (
        np.asarray(
            ridge_pred,
            dtype=np.float32,
        )
        + np.asarray(
            raw_residual_pred,
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
    y_true,
    ridge_pred,
    raw_residual_pred,
    alphas,
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
        raw_residual_pred,
        dtype=np.float64,
    )

    final_corr = (
        raw
        * np.asarray(
            alphas,
            dtype=np.float64,
        )[
            None,
            :
        ]
    )

    return {
        "raw_residual_corr_U": (
            corrcoef_safe(
                true_res[
                    :,
                    0
                ],
                raw[
                    :,
                    0
                ],
            )
        ),
        "raw_residual_corr_V": (
            corrcoef_safe(
                true_res[
                    :,
                    1
                ],
                raw[
                    :,
                    1
                ],
            )
        ),
        "alpha_U": float(
            alphas[
                0
            ]
        ),
        "alpha_V": float(
            alphas[
                1
            ]
        ),
        "final_correction_RMS_U_mps": float(
            np.sqrt(
                np.mean(
                    final_corr[
                        :,
                        0
                    ] ** 2
                )
            )
        ),
        "final_correction_RMS_V_mps": float(
            np.sqrt(
                np.mean(
                    final_corr[
                        :,
                        1
                    ] ** 2
                )
            )
        ),
    }


# =============================================================================
# Network
# =============================================================================

def build_model_class(
    torch,
    nn,
):
    seq_dim = len(
        SEQ_FEATURE_NAMES
    )

    summary_dim = len(
        SUMMARY_FEATURE_NAMES
    )

    class ResidualGRU(nn.Module):
        def __init__(self):
            super().__init__()

            self.gru = nn.GRU(
                input_size=seq_dim,
                hidden_size=(
                    FREEZE.gru_hidden
                ),
                num_layers=(
                    FREEZE.gru_layers
                ),
                dropout=(
                    FREEZE.dropout
                    if FREEZE.gru_layers
                    > 1
                    else 0.0
                ),
                batch_first=True,
            )

            fused_dim = (
                FREEZE.gru_hidden
                + summary_dim
                + 3
            )

            self.shared = nn.Sequential(
                nn.Linear(
                    fused_dim,
                    FREEZE.dense1,
                ),
                nn.LayerNorm(
                    FREEZE.dense1
                ),
                nn.GELU(),
                nn.Dropout(
                    FREEZE.dropout
                ),
                nn.Linear(
                    FREEZE.dense1,
                    FREEZE.dense2,
                ),
                nn.GELU(),
            )

            self.u_hidden = nn.Sequential(
                nn.Linear(
                    FREEZE.dense2,
                    FREEZE.head_hidden,
                ),
                nn.GELU(),
            )

            self.v_hidden = nn.Sequential(
                nn.Linear(
                    FREEZE.dense2,
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

            # Critical ResNet-style safe start:
            # residual prediction at epoch 0 == exactly zero.
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
            seq_z,
            summary_z,
            anchor_z,
            residual_std,
        ):
            z, _ = self.gru(
                seq_z
            )

            h = z[
                :,
                -1,
                :
            ]

            fused = torch.cat(
                [
                    h,
                    summary_z,
                    anchor_z,
                ],
                dim=1,
            )

            shared = self.shared(
                fused
            )

            u_logit = self.u_out(
                self.u_hidden(
                    shared
                )
            )

            v_logit = self.v_out(
                self.v_hidden(
                    shared
                )
            )

            logits = torch.cat(
                [
                    u_logit,
                    v_logit,
                ],
                dim=1,
            )

            raw_residual = (
                float(
                    RESIDUAL_CAP_STD
                )
                * torch.tanh(
                    logits
                )
                * residual_std[
                    None,
                    :
                ]
            )

            return raw_residual

    return ResidualGRU


# =============================================================================
# Training / prediction
# =============================================================================

def residual_loss(
    torch,
    raw_residual,
    true_residual,
    residual_std,
):
    pred_z = (
        raw_residual
        / residual_std[
            None,
            :
        ]
    )

    true_z = (
        true_residual
        / residual_std[
            None,
            :
        ]
    )

    mse_u = torch.mean(
        (
            pred_z[
                :,
                0
            ]
            - true_z[
                :,
                0
            ]
        ) ** 2
    )

    mse_v = torch.mean(
        (
            pred_z[
                :,
                1
            ]
            - true_z[
                :,
                1
            ]
        ) ** 2
    )

    size_penalty = torch.mean(
        pred_z ** 2
    )

    return (
        FREEZE.weight_u
        * mse_u
        + FREEZE.weight_v
        * mse_v
        + FREEZE.weight_residual_size
        * size_penalty
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

    chunks = []

    n = len(
        transformed[
            "seq_z"
        ]
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
                device,
                non_blocking=True,
            )

            summary_t = torch.from_numpy(
                transformed[
                    "summary_z"
                ][
                    start:stop
                ]
            ).to(
                device,
                non_blocking=True,
            )

            anchor_t = torch.from_numpy(
                transformed[
                    "anchor_z"
                ][
                    start:stop
                ]
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
                pred = model(
                    seq_t,
                    summary_t,
                    anchor_t,
                    residual_std_t,
                )

            chunks.append(
                pred.detach()
                .float()
                .cpu()
                .numpy()
            )

    return np.concatenate(
        chunks,
        axis=0,
    ).astype(
        np.float32
    )


def train_dev_run(
    *,
    torch,
    nn,
    seed,
    held_out,
    train_data,
    val_data,
    output_dir,
    device,
    debug_fast,
):
    candidate = (
        "ResidualGRU-AnalyticalShrinkage"
    )

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

    ridge_va = ridge_predict(
        ridge_model,
        ridge_scaler,
        Xva_base,
    )

    ridge_metrics = evaluate_wind(
        yva,
        ridge_va,
    )

    ridge_oof = crossfit_ridge_predictions(
        Xtr_base,
        ytr,
        train_data[
            "mission_labels"
        ],
    )

    feat_tr = build_residual_features(
        train_data[
            "X_full9"
        ],
        ridge_oof,
    )

    feat_va = build_residual_features(
        val_data[
            "X_full9"
        ],
        ridge_va,
    )

    scalers = fit_residual_scalers(
        feat_tr,
        ytr,
        ridge_oof,
    )

    tr = transform_residual_features(
        feat_tr,
        scalers,
    )

    va = transform_residual_features(
        feat_va,
        scalers,
    )

    true_res_tr = (
        ytr
        - ridge_oof
    ).astype(
        np.float32
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

    residual_std_t = torch.as_tensor(
        scalers[
            "residual_std"
        ],
        dtype=torch.float32,
        device=device,
    )

    # Epoch 0 = exact Ridge.
    best_score = 1.0
    best_state = state_dict_cpu(
        model
    )
    best_epoch = 0
    best_alphas = np.zeros(
        2,
        dtype=np.float32,
    )
    best_raw_va = np.zeros_like(
        yva,
        dtype=np.float32,
    )

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

    n = len(
        ytr
    )

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

            true_res_t = torch.from_numpy(
                true_res_tr[
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
                raw_res = model(
                    seq_t,
                    summary_t,
                    anchor_t,
                    residual_std_t,
                )

                loss = residual_loss(
                    torch,
                    raw_res,
                    true_res_t,
                    residual_std_t,
                )

            if not torch.isfinite(
                loss
            ):
                raise FloatingPointError(
                    "Nonfinite Stage12Z residual loss."
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

        raw_va = predict_raw_residual(
            torch=torch,
            model=model,
            transformed=va,
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

        fold_alphas = analytical_alphas(
            yva,
            ridge_va,
            raw_va,
        )

        final_va = apply_shrinkage(
            ridge_va,
            raw_va,
            fold_alphas,
        )

        metrics = evaluate_wind(
            yva,
            final_va,
        )

        score = selection_score(
            metrics,
            ridge_metrics,
        )

        diag = residual_diagnostics(
            yva,
            ridge_va,
            raw_va,
            fold_alphas,
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
            best_alphas = (
                fold_alphas.copy()
            )
            best_raw_va = (
                raw_va.copy()
            )
            patience = 0
        else:
            patience += 1

        scheduler.step(
            score
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
                "selection_score": float(
                    score
                ),
                **metrics,
                **diag,
            }
        )

        if (
            epoch <= 5
            or epoch % 10 == 0
            or improved
        ):
            log(
                f"      ep={epoch:03d} | "
                f"score={score:.6f} | "
                f"alpha=({fold_alphas[0]:.3f},{fold_alphas[1]:.3f}) | "
                f"U={metrics['wind_U_RMSE_mps']:.4f} | "
                f"V={metrics['wind_V_RMSE_mps']:.4f} | "
                f"WS={metrics['wind_speed_RMSE_mps']:.4f} | "
                f"WD={metrics['wind_direction_RMSE_deg']:.3f} | "
                f"corr=({diag['raw_residual_corr_U']:+.3f},"
                f"{diag['raw_residual_corr_V']:+.3f})"
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

    final_va = apply_shrinkage(
        ridge_va,
        best_raw_va,
        best_alphas,
    )

    metrics = evaluate_wind(
        yva,
        final_va,
    )

    diag = residual_diagnostics(
        yva,
        ridge_va,
        best_raw_va,
        best_alphas,
    )

    pred_dir = (
        output_dir
        / "development_predictions"
    )

    hist_dir = (
        output_dir
        / "histories"
    )

    ckpt_dir = (
        output_dir
        / "checkpoints_dev"
    )

    for d in [
        pred_dir,
        hist_dir,
        ckpt_dir,
    ]:
        d.mkdir(
            parents=True,
            exist_ok=True,
        )

    tag = (
        f"{held_out.replace(' ', '_')}"
        f"__seed{seed}"
    )

    pred_path = (
        pred_dir
        / f"{tag}.npz"
    )

    hist_path = (
        hist_dir
        / f"{tag}.csv"
    )

    ckpt_path = (
        ckpt_dir
        / f"{tag}.pt"
    )

    np.savez_compressed(
        pred_path,
        y_true=yva.astype(
            np.float32
        ),
        ridge_pred=ridge_va.astype(
            np.float32
        ),
        raw_residual_pred=best_raw_va.astype(
            np.float32
        ),
        fold_alphas=best_alphas.astype(
            np.float32
        ),
    )

    pd.DataFrame(
        history
    ).to_csv(
        hist_path,
        index=False,
        encoding="utf-8-sig",
    )

    torch.save(
        {
            "stage": "12Z-development",
            "held_out_mission": held_out,
            "seed": int(seed),
            "best_epoch": int(
                best_epoch
            ),
            "fold_alphas": (
                best_alphas
            ),
            "state_dict": (
                best_state
            ),
            "nn_parameter_count": int(
                nn_params
            ),
            "ridge_parameter_count": int(
                ridge_parameter_count(
                    ridge_model
                )
            ),
            "freeze": asdict(
                FREEZE
            ),
        },
        ckpt_path,
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
        "candidate": (
            "ResidualGRU-AnalyticalShrinkage"
        ),
        "held_out_mission": held_out,
        "seed": int(seed),
        "status": "completed_finite",
        "selection_score": float(
            selection_score(
                metrics,
                ridge_metrics,
            )
        ),
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
        **metrics,
        **diag,
        "prediction_path": str(
            pred_path
        ),
        "history_path": str(
            hist_path
        ),
        "checkpoint_path": str(
            ckpt_path
        ),
    }


# =============================================================================
# Final training
# =============================================================================

def train_final_model(
    *,
    torch,
    nn,
    selected_seed,
    final_epochs,
    frozen_alphas,
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

    feat_dev = build_residual_features(
        dev_all[
            "X_full9"
        ],
        ridge_oof,
    )

    feat_trop = build_residual_features(
        tropical[
            "X_full9"
        ],
        ridge_trop,
    )

    scalers = fit_residual_scalers(
        feat_dev,
        ydev,
        ridge_oof,
    )

    tr = transform_residual_features(
        feat_dev,
        scalers,
    )

    te = transform_residual_features(
        feat_trop,
        scalers,
    )

    true_res_dev = (
        ydev
        - ridge_oof
    ).astype(
        np.float32
    )

    seed_everything(
        torch,
        selected_seed,
    )

    ModelClass = build_model_class(
        torch,
        nn,
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

    n = len(
        ydev
    )

    log(
        f"[FINAL TRAIN] seed={selected_seed} | "
        f"epochs={final_epochs} | "
        f"frozen alpha_U={frozen_alphas[0]:.4f}, "
        f"alpha_V={frozen_alphas[1]:.4f}"
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

            true_res_t = torch.from_numpy(
                true_res_dev[
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
                raw_res = model(
                    seq_t,
                    summary_t,
                    anchor_t,
                    residual_std_t,
                )

                loss = residual_loss(
                    torch,
                    raw_res,
                    true_res_t,
                    residual_std_t,
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

        if (
            epoch <= 5
            or epoch % 20 == 0
            or epoch
            == int(
                final_epochs
            )
        ):
            log(
                f"  epoch={epoch:03d}/{int(final_epochs):03d} | "
                f"train_loss={loss_sum/max(n_seen,1):.6f}"
            )

    raw_trop = predict_raw_residual(
        torch=torch,
        model=model,
        transformed=te,
        residual_std=(
            scalers[
                "residual_std"
            ]
        ),
        device=device,
        amp_enabled=amp_enabled,
    )

    final_trop = apply_shrinkage(
        ridge_trop,
        raw_trop,
        frozen_alphas,
    )

    final_metrics = evaluate_wind(
        ytrop,
        final_trop,
    )

    diag = residual_diagnostics(
        ytrop,
        ridge_trop,
        raw_trop,
        frozen_alphas,
    )

    ckpt_dir = (
        output_dir
        / "final_model"
    )

    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ckpt_path = (
        ckpt_dir
        / "12Z_final.pt"
    )

    torch.save(
        {
            "stage": "12Z-final",
            "selected_seed": int(
                selected_seed
            ),
            "final_epochs": int(
                final_epochs
            ),
            "frozen_alphas": np.asarray(
                frozen_alphas,
                dtype=np.float32,
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
            "nn_parameter_count": int(
                nn_params
            ),
            "ridge_parameter_count": int(
                ridge_parameter_count(
                    ridge_model
                )
            ),
        },
        ckpt_path,
    )

    return {
        "ridge_metrics": (
            ridge_metrics
        ),
        "final_metrics": (
            final_metrics
        ),
        "diagnostics": diag,
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
        "checkpoint_path": (
            ckpt_path
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
        help=(
            "Smoke test: Antarctic holdout, one seed, short training. "
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

    device = torch.device(
        "cpu"
        if (
            args.force_cpu
            or not torch.cuda.is_available()
        )
        else "cuda"
    )

    log("=" * 132)
    log(
        "12Z — RIDGE-ANCHORED U/V RESIDUAL + ANALYTICAL SHRINKAGE"
    )
    log("=" * 132)
    log(
        f"dataset : {dataset_root}"
    )
    log(
        f"output  : {output_dir}"
    )
    log(
        "anchor  : Base4-Ridge [U,V,T,RH]"
    )
    log(
        "network : compact GRU learns ONLY [true U,V - OOF Ridge U,V]"
    )
    log(
        f"alpha   : analytical component shrinkage, 0 <= alpha <= {ALPHA_MAX:.2f}"
    )
    log(
        f"device  : {device}"
    )

    if device.type == "cuda":
        log(
            f"GPU     : {torch.cuda.get_device_name(0)}"
        )

    # ---------------------------------------------------------
    # Development only
    # ---------------------------------------------------------
    log("")
    log(
        "[STAGE A] Loading Antarctic / Atlantic / West Coast only."
    )

    dev_missions = {
        mission: load_mission(
            dataset_root,
            mission,
        )
        for mission
        in DEV_MISSIONS
    }

    log(
        "[FIREWALL] Tropical Atlantic NOT loaded."
    )

    active_holdouts = (
        ["Antarctic"]
        if args.debug_fast
        else DEV_MISSIONS
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
        train_data = concat_missions(
            [
                dev_missions[m]
                for m in DEV_MISSIONS
                if m != held_out
            ]
        )

        val_data = dev_missions[
            held_out
        ]

        log("-" * 132)
        log(
            f"[HOLDOUT] {held_out} | "
            f"Ntrain={len(train_data['y_uv']):,} | "
            f"Nval={len(val_data['y_uv']):,}"
        )

        for seed in active_seeds:
            log(
                f"  [TRAIN] seed={seed}"
            )

            result = train_dev_run(
                torch=torch,
                nn=nn,
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
                f"  [DONE] score={result['selection_score']:.6f} | "
                f"alpha=({result['alpha_U']:.3f},{result['alpha_V']:.3f}) | "
                f"U={result['wind_U_RMSE_mps']:.4f} | "
                f"V={result['wind_V_RMSE_mps']:.4f} | "
                f"corr=({result['raw_residual_corr_U']:+.3f},"
                f"{result['raw_residual_corr_V']:+.3f}) | "
                f"bestEp={result['best_epoch']}"
            )

    if args.debug_fast:
        log("")
        log(
            "[DONE] debug-fast completed. Tropical Atlantic was NOT loaded."
        )
        return 0

    # ---------------------------------------------------------
    # Select seed from LOMO only
    # ---------------------------------------------------------
    runs = pd.read_csv(
        run_path
    )

    seed_summary = (
        runs.groupby(
            "seed",
            as_index=False,
        )
        .agg(
            selection_score_mean=(
                "selection_score",
                "mean",
            ),
            selection_score_std=(
                "selection_score",
                "std",
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
            residual_corr_U_mean=(
                "raw_residual_corr_U",
                "mean",
            ),
            residual_corr_V_mean=(
                "raw_residual_corr_V",
                "mean",
            ),
            alpha_U_mean=(
                "alpha_U",
                "mean",
            ),
            alpha_V_mean=(
                "alpha_V",
                "mean",
            ),
            best_epoch_median=(
                "best_epoch",
                "median",
            ),
            parameter_count_mean=(
                "parameter_count",
                "mean",
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

    selected_runs = runs.loc[
        runs[
            "seed"
        ].astype(int)
        == selected_seed
    ].copy()

    final_epochs = int(
        max(
            1,
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

    # ---------------------------------------------------------
    # Pool held-out development predictions for FINAL alpha
    # ---------------------------------------------------------
    pooled_y = []
    pooled_ridge = []
    pooled_raw = []

    for _, row in selected_runs.iterrows():
        pred_path = Path(
            str(
                row[
                    "prediction_path"
                ]
            )
        )

        with np.load(
            pred_path,
            allow_pickle=False,
        ) as z:
            pooled_y.append(
                np.asarray(
                    z[
                        "y_true"
                    ],
                    dtype=np.float32,
                )
            )

            pooled_ridge.append(
                np.asarray(
                    z[
                        "ridge_pred"
                    ],
                    dtype=np.float32,
                )
            )

            pooled_raw.append(
                np.asarray(
                    z[
                        "raw_residual_pred"
                    ],
                    dtype=np.float32,
                )
            )

    pooled_y = np.concatenate(
        pooled_y,
        axis=0,
    )

    pooled_ridge = np.concatenate(
        pooled_ridge,
        axis=0,
    )

    pooled_raw = np.concatenate(
        pooled_raw,
        axis=0,
    )

    frozen_alphas = analytical_alphas(
        pooled_y,
        pooled_ridge,
        pooled_raw,
    )

    pooled_final = apply_shrinkage(
        pooled_ridge,
        pooled_raw,
        frozen_alphas,
    )

    pooled_ridge_metrics = evaluate_wind(
        pooled_y,
        pooled_ridge,
    )

    pooled_final_metrics = evaluate_wind(
        pooled_y,
        pooled_final,
    )

    pooled_diag = residual_diagnostics(
        pooled_y,
        pooled_ridge,
        pooled_raw,
        frozen_alphas,
    )

    calibration_npz = (
        output_dir
        / "development_pooled_calibration.npz"
    )

    np.savez_compressed(
        calibration_npz,
        y_true=pooled_y,
        ridge_pred=pooled_ridge,
        raw_residual_pred=pooled_raw,
        frozen_alphas=frozen_alphas,
    )

    frozen = {
        "stage": "12Z",
        "selected_seed": int(
            selected_seed
        ),
        "final_epochs": int(
            final_epochs
        ),
        "alpha_U": float(
            frozen_alphas[
                0
            ]
        ),
        "alpha_V": float(
            frozen_alphas[
                1
            ]
        ),
        "alpha_max": float(
            ALPHA_MAX
        ),
        "residual_cap_std": float(
            RESIDUAL_CAP_STD
        ),
        "selection_source": (
            "Antarctic/Atlantic/West Coast LOMO only"
        ),
        "Tropical_Atlantic_used_for_selection": False,
        "development_pooled_Ridge_metrics": (
            pooled_ridge_metrics
        ),
        "development_pooled_Final_metrics": (
            pooled_final_metrics
        ),
        "development_pooled_residual_diagnostics": (
            pooled_diag
        ),
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

    log("")
    log("=" * 132)
    log(
        "12Z DEVELOPMENT SEED SUMMARY"
    )
    log("=" * 132)
    log(
        seed_summary.to_string(
            index=False
        )
    )

    log("")
    log(
        f"[FROZEN BEFORE TROPICAL] "
        f"seed={selected_seed} | "
        f"epochs={final_epochs} | "
        f"alpha_U={frozen_alphas[0]:.4f} | "
        f"alpha_V={frozen_alphas[1]:.4f}"
    )

    log(
        "Development pooled calibration:"
    )

    log(
        f"  Ridge U/V = "
        f"{pooled_ridge_metrics['wind_U_RMSE_mps']:.6f} / "
        f"{pooled_ridge_metrics['wind_V_RMSE_mps']:.6f}"
    )

    log(
        f"  Final U/V = "
        f"{pooled_final_metrics['wind_U_RMSE_mps']:.6f} / "
        f"{pooled_final_metrics['wind_V_RMSE_mps']:.6f}"
    )

    log(
        f"  raw residual corr U/V = "
        f"{pooled_diag['raw_residual_corr_U']:+.4f} / "
        f"{pooled_diag['raw_residual_corr_V']:+.4f}"
    )

    log(
        f"[SAVED BEFORE TROPICAL LOAD] {frozen_path}"
    )

    # ---------------------------------------------------------
    # Final Tropical benchmark
    # ---------------------------------------------------------
    log("")
    log(
        "[STAGE C] Configuration frozen. Loading Tropical Atlantic NOW."
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

    final = train_final_model(
        torch=torch,
        nn=nn,
        selected_seed=selected_seed,
        final_epochs=final_epochs,
        frozen_alphas=frozen_alphas,
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

    final_df = pd.DataFrame(
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
                    "Ridge+UVResidualShrink-final"
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

    improvements = {
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
        "stage": "12Z",
        "script_version": (
            SCRIPT_VERSION
        ),
        "frozen_before_tropical": (
            frozen
        ),
        "final_tropical": {
            "ridge_metrics": ridge,
            "residual_model_metrics": ours,
            "residual_diagnostics": diag,
            "improvement_vs_Ridge": improvements,
            "beats_BP_metricwise": beats,
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
            "Stage12Z excludes Tropical Atlantic labels from all "
            "selection/shrinkage calibration, but Tropical Atlantic had "
            "already been viewed in earlier project experiments."
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
            "12Z FINAL TROPICAL ATLANTIC BENCHMARK\n"
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
            "FROZEN ANALYTICAL SHRINKAGE\n"
        )
        f.write(
            "-" * 110
            + "\n"
        )
        f.write(
            f"alpha_U = {frozen_alphas[0]:.6f}\n"
        )
        f.write(
            f"alpha_V = {frozen_alphas[1]:.6f}\n"
        )

        f.write(
            "\nTROPICAL RESIDUAL DIAGNOSTICS\n"
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
            "\nIMPROVEMENT VS RIDGE\n"
        )
        f.write(
            "-" * 110
            + "\n"
        )

        for k, v in improvements.items():
            f.write(
                f"{k}: {100.0*v:+.4f}%\n"
            )

        f.write(
            "\nBEATS BP-STGNN\n"
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
        "Frozen analytical shrinkage:"
    )
    log(
        f"  alpha_U = {frozen_alphas[0]:.4f}"
    )
    log(
        f"  alpha_V = {frozen_alphas[1]:.4f}"
    )

    log("")
    log(
        "Tropical residual diagnostics:"
    )
    log(
        f"  raw corr U = {diag['raw_residual_corr_U']:+.4f}"
    )
    log(
        f"  raw corr V = {diag['raw_residual_corr_V']:+.4f}"
    )
    log(
        f"  correction RMS U = {diag['final_correction_RMS_U_mps']:.5f} m/s"
    )
    log(
        f"  correction RMS V = {diag['final_correction_RMS_V_mps']:.5f} m/s"
    )

    log("")
    log(
        "Improvement vs Base4-Ridge:"
    )

    for k, v in improvements.items():
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
        f"{final['effective_parameter_count']}/"
        f"{BP_REFERENCE['params']} "
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
