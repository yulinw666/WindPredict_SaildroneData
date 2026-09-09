# -*- coding: utf-8 -*-
r"""
12X_train_RidgeAnchoredSafeUVResidual_FinalTropical.py

Stage 12X
=========
Ridge-anchored conservative U/V residual correction for the strict
BP-aligned point-sampled Saildrone benchmark.

GOAL
----
Keep the strong, low-parameter Ridge model as the primary predictor and let a
small neural network make ONLY bounded corrections to U and V.

The architecture / correction limit / seed / final epoch count are selected
ONLY from Antarctic + Atlantic + West Coast development LOMO.

Only AFTER the configuration is frozen does the script load Tropical Atlantic,
fit the final model on ALL THREE development missions, and evaluate once on
Tropical Atlantic.

This script therefore does not use Tropical Atlantic for:
    - architecture selection
    - correction-cap selection
    - seed selection
    - early stopping
    - epoch selection
    - feature selection

IMPORTANT
---------
Tropical Atlantic has been inspected in earlier exploratory project stages, so
it should not be described as a never-seen pristine test set in the final
paper.  This script nevertheless keeps it completely outside Stage-12X model
selection and uses it only as the same-region benchmark evaluation against the
published BP-STGNN numbers.

FROZEN BP-ALIGNED TASK
----------------------
Source:
    Stage 12G point-sampled dataset.

History:
    six original 1-min mean observations sampled every 10 min:
        t-50, t-40, t-30, t-20, t-10, t

Target:
    original 1-min mean U,V at t+10 min.

NO:
    10-min block means
    intermediate 1-min records
    pressure
    wing angle
    absolute position

RIDGE ANCHOR
------------
Input:
    [U,V,T,RH] x 6

Output:
    [U_Ridge,V_Ridge]

Ridge is always part of the final model.

RESIDUAL FEATURES
-----------------
The correction network uses ONLY deterministic transforms of the same six
sampled U,V,T,RH observations:

Per-step sequence (10 features):
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

Per-sample summary:
    latest U
    latest V
    latest dU
    latest dV
    std(U_6)
    std(V_6)
    std(WS_6)
    slope(U_6)
    slope(V_6)
    slope(WS_6)

Anchor context:
    U_Ridge
    V_Ridge
    WS_Ridge

SAFE CORRECTION
---------------
The network predicts delta logits and gate logits.

    gate = sigmoid(gate_logits)

    delta_U =
        correction_cap
        * gate_U
        * tanh(delta_logit_U)
        * sigma_residual_U

    delta_V =
        correction_cap
        * gate_V
        * tanh(delta_logit_V)
        * sigma_residual_V

Final:
    U_hat = U_Ridge + delta_U
    V_hat = V_Ridge + delta_V

The correction is therefore bounded and cannot replace the Ridge anchor.

The delta head is zero initialized:
    epoch 0 = EXACT Ridge prediction.

CANDIDATE CORRECTION CAPS
-------------------------
    0.15
    0.30

These are selected only on development LOMO.

NETWORK CANDIDATES
------------------
1) SafeDeepMLP
   Shared:
       flattened 6x10 sequence + 10 summary + 3 anchor
       96 -> 64 -> 48 -> 32
   Head:
       [delta_U, delta_V, gate_U, gate_V]

2) SafeGRU
   2-layer GRU, hidden=48
   + summary + anchor
   -> 64 -> 32
   -> [delta_U, delta_V, gate_U, gate_V]

Both remain far smaller than BP-STGNN.

LOSS
----
The network is NOT trained to reproduce the full residual.

It directly minimizes the error of:
    Ridge + bounded correction

with:
    weighted U/V loss
    + auxiliary WS loss
    + circular vector-direction cosine loss
    + correction-size regularization

V receives slightly larger component weight because V has consistently been
the harder component in the development experiments.

    component:
        0.45 * U error
      + 0.55 * V error

    total:
        L_component
      + 0.10 * L_WS
      + 0.03 * L_direction
      + 0.02 * L_correction

All loss constants are frozen before Tropical evaluation.

OOF RESIDUAL TRAINING
---------------------
For each development outer LOMO fold, Ridge anchors for correction-network
training are generated using 5-fold mission-preserving BLOCKED cross-fitting.

This prevents the correction network from learning against artificially small
in-sample Ridge errors.

DEVELOPMENT
-----------
Holdout Antarctic  <- train Atlantic + West Coast
Holdout Atlantic   <- train Antarctic + West Coast
Holdout West Coast <- train Antarctic + Atlantic

For each architecture / cap / seed:
    train max 300 epochs
    early stop patience 30
    LR scheduler patience 10

Candidate architecture/cap:
    selected by mean LOMO score over all three seeds.

Seed:
    selected by mean LOMO score across the three held-out missions.

Final epoch count:
    median best epoch of the selected candidate+seed across the three
    development holdouts.

FINAL FIT
---------
1) Fit Base4-Ridge on Antarctic + Atlantic + West Coast.
2) Build OOF Ridge anchors on these three development missions.
3) Train selected correction network for frozen final epoch count.
4) Load Tropical Atlantic only now.
5) Evaluate:
       Ridge
       Ridge + Safe U/V Residual
       BP-STGNN published reference

PUBLISHED BP-STGNN REFERENCE
----------------------------
Tropical Atlantic:
    U RMSE  = 0.748640 m/s
    V RMSE  = 0.705060 m/s
    WS RMSE = 0.699400 m/s
    WD RMSE = 5.584 deg
    params  = 242580

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\12X_train_RidgeAnchoredSafeUVResidual_FinalTropical.py" --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12X_RidgeAnchoredSafeUVResidual_FinalTropical_v0_1"

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


SCRIPT_VERSION = "0.1.0-12X-RidgeAnchoredSafeUVResidual"

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
    / "12X_RidgeAnchoredSafeUVResidual_FinalTropical_v0_1"
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

CORRECTION_CAPS = [
    0.15,
    0.30,
]

NETWORK_MODES = [
    "deep_mlp",
    "gru",
]

SCORE_WEIGHTS = {
    "U": 0.25,
    "V": 0.35,
    "WS": 0.25,
    "WD": 0.15,
}

BP_REFERENCE = {
    "U_RMSE_mps": 0.748640,
    "V_RMSE_mps": 0.705060,
    "WS_RMSE_mps": 0.699400,
    "WD_RMSE_deg": 5.584000,
    "params": 242580,
}


@dataclass(frozen=True)
class Freeze:
    mlp_h1: int = 96
    mlp_h2: int = 64
    mlp_h3: int = 48
    mlp_h4: int = 32

    gru_hidden: int = 48
    gru_layers: int = 2
    gru_head1: int = 64
    gru_head2: int = 32

    dropout: float = 0.10

    learning_rate: float = 6e-4
    weight_decay: float = 2e-5

    batch_size: int = 256
    max_epochs: int = 300
    early_stop_patience: int = 30
    scheduler_patience: int = 10
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
        "Cannot find track_J_joint_compatible under "
        f"{dataset_dir}"
    )


def mission_npz_path(
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
    path = mission_npz_path(
        dataset_root,
        mission,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing mission dataset: {path}"
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
            f"{mission}: Track-J feature mismatch: {names}"
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
        y = y[:, 0, :]

    if (
        y.ndim != 2
        or y.shape[1] != 2
    ):
        raise RuntimeError(
            f"{mission}: unexpected y shape {y.shape}"
        )

    if not np.all(
        target - context == TEN_MIN_NS
    ):
        raise RuntimeError(
            f"{mission}: target is not exactly +10 min."
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
        "path": path,
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
# Feature construction
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

    t = (
        t - np.mean(t)
    )

    denom = float(
        np.sum(t ** 2)
    )

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
            * t[None, :],
            axis=1,
        )
        / max(
            denom,
            EPS,
        )
    )


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


def build_residual_features(X_full9):
    X = np.asarray(
        X_full9,
        dtype=np.float64,
    )

    U = X[:, :, IDX_U]
    V = X[:, :, IDX_V]
    T = X[:, :, IDX_T]
    RH = X[:, :, IDX_RH]

    WS = np.hypot(
        U,
        V,
    )

    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dWS = np.zeros_like(WS)

    dU[:, 1:] = (
        U[:, 1:]
        - U[:, :-1]
    )

    dV[:, 1:] = (
        V[:, 1:]
        - V[:, :-1]
    )

    dWS[:, 1:] = (
        WS[:, 1:]
        - WS[:, :-1]
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

    if not (
        np.isfinite(seq).all()
        and np.isfinite(summary).all()
    ):
        raise RuntimeError(
            "Nonfinite residual features."
        )

    return (
        seq,
        summary,
    )


# =============================================================================
# Ridge
# =============================================================================

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
            ][None, None, :]
        )
        / scaler[
            "x_std"
        ][None, None, :]
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
            ][None, :]
        )
        / scaler[
            "y_std"
        ][None, :]
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
        ][None, :]
        + scaler[
            "y_mean"
        ][None, :]
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


def selection_score(metrics, base):
    ratios = {
        "U_ratio": (
            metrics[
                "wind_U_RMSE_mps"
            ]
            / max(
                base[
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
                base[
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
                base[
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
                base[
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


def correction_diagnostics(
    y_true,
    ridge_pred,
    final_pred,
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

    pred_corr = (
        np.asarray(
            final_pred,
            dtype=np.float64,
        )
        - np.asarray(
            ridge_pred,
            dtype=np.float64,
        )
    )

    return {
        "corr_residual_U": (
            corrcoef_safe(
                true_res[:, 0],
                pred_corr[:, 0],
            )
        ),
        "corr_residual_V": (
            corrcoef_safe(
                true_res[:, 1],
                pred_corr[:, 1],
            )
        ),
        "correction_RMS_U_mps": float(
            np.sqrt(
                np.mean(
                    pred_corr[:, 0] ** 2
                )
            )
        ),
        "correction_RMS_V_mps": float(
            np.sqrt(
                np.mean(
                    pred_corr[:, 1] ** 2
                )
            )
        ),
        "correction_abs_p95_U_mps": float(
            np.percentile(
                np.abs(
                    pred_corr[:, 0]
                ),
                95,
            )
        ),
        "correction_abs_p95_V_mps": float(
            np.percentile(
                np.abs(
                    pred_corr[:, 1]
                ),
                95,
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
            fold_ids[chunk] = k

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
        mission_labels,
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
            X_base4[train_mask],
            y[train_mask],
        )

        pred[val_mask] = ridge_predict(
            model,
            scaler,
            X_base4[val_mask],
        )

    if not np.isfinite(pred).all():
        raise RuntimeError(
            "Nonfinite OOF Ridge prediction."
        )

    return pred


# =============================================================================
# Correction feature scaling
# =============================================================================

def fit_correction_scalers(
    seq,
    summary,
    anchor,
    y,
):
    seq64 = np.asarray(
        seq,
        dtype=np.float64,
    )

    summary64 = np.asarray(
        summary,
        dtype=np.float64,
    )

    anchor64 = np.asarray(
        anchor,
        dtype=np.float64,
    )

    residual = (
        np.asarray(
            y,
            dtype=np.float64,
        )
        - anchor64
    )

    seq_mean = np.mean(
        seq64,
        axis=(0, 1),
    )

    seq_std = np.std(
        seq64,
        axis=(0, 1),
        ddof=0,
    )

    seq_std = np.where(
        seq_std < 1e-8,
        1.0,
        seq_std,
    )

    summary_mean = np.mean(
        summary64,
        axis=0,
    )

    summary_std = np.std(
        summary64,
        axis=0,
        ddof=0,
    )

    summary_std = np.where(
        summary_std < 1e-8,
        1.0,
        summary_std,
    )

    anchor_context = np.concatenate(
        [
            anchor64,
            np.linalg.norm(
                anchor64,
                axis=1,
                keepdims=True,
            ),
        ],
        axis=1,
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
        residual,
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
            y,
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
        "y_std": y_std.astype(
            np.float32
        ),
    }


def transform_correction_inputs(
    seq,
    summary,
    anchor,
    scalers,
):
    seq_z = (
        (
            np.asarray(
                seq,
                dtype=np.float32,
            )
            - scalers[
                "seq_mean"
            ][None, None, :]
        )
        / scalers[
            "seq_std"
        ][None, None, :]
    ).astype(
        np.float32
    )

    summary_z = (
        (
            np.asarray(
                summary,
                dtype=np.float32,
            )
            - scalers[
                "summary_mean"
            ][None, :]
        )
        / scalers[
            "summary_std"
        ][None, :]
    ).astype(
        np.float32
    )

    anchor_context = np.concatenate(
        [
            np.asarray(
                anchor,
                dtype=np.float32,
            ),
            np.linalg.norm(
                np.asarray(
                    anchor,
                    dtype=np.float32,
                ),
                axis=1,
                keepdims=True,
            ),
        ],
        axis=1,
    )

    anchor_z = (
        (
            anchor_context
            - scalers[
                "anchor_mean"
            ][None, :]
        )
        / scalers[
            "anchor_std"
        ][None, :]
    ).astype(
        np.float32
    )

    return (
        seq_z,
        summary_z,
        anchor_z,
    )


# =============================================================================
# Models
# =============================================================================

def build_model_class(
    torch,
    nn,
    mode,
    correction_cap,
):
    seq_dim = len(
        SEQ_FEATURE_NAMES
    )
    summary_dim = len(
        SUMMARY_FEATURE_NAMES
    )
    anchor_dim = 3

    class SafeCorrectionMixin:
        def bounded_correction(
            self,
            raw,
            residual_std,
        ):
            delta_logits = raw[:, :2]
            gate_logits = raw[:, 2:]

            gate = torch.sigmoid(
                gate_logits
            )

            delta_unit = torch.tanh(
                delta_logits
            )

            correction = (
                float(
                    correction_cap
                )
                * gate
                * delta_unit
                * residual_std[
                    None,
                    :
                ]
            )

            return (
                correction,
                gate,
            )

    if mode == "deep_mlp":
        class SafeDeepMLP(
            nn.Module,
            SafeCorrectionMixin,
        ):
            def __init__(self):
                super().__init__()

                input_dim = (
                    LOOKBACK_STEPS
                    * seq_dim
                    + summary_dim
                    + anchor_dim
                )

                self.net = nn.Sequential(
                    nn.Linear(
                        input_dim,
                        FREEZE.mlp_h1,
                    ),
                    nn.LayerNorm(
                        FREEZE.mlp_h1
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        FREEZE.dropout
                    ),
                    nn.Linear(
                        FREEZE.mlp_h1,
                        FREEZE.mlp_h2,
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        FREEZE.dropout
                    ),
                    nn.Linear(
                        FREEZE.mlp_h2,
                        FREEZE.mlp_h3,
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        FREEZE.dropout
                    ),
                    nn.Linear(
                        FREEZE.mlp_h3,
                        FREEZE.mlp_h4,
                    ),
                    nn.GELU(),
                )

                self.out = nn.Linear(
                    FREEZE.mlp_h4,
                    4,
                )

                nn.init.zeros_(
                    self.out.weight
                )
                nn.init.zeros_(
                    self.out.bias[
                        :2
                    ]
                )

                with torch.no_grad():
                    self.out.bias[
                        2:
                    ].fill_(
                        -2.0
                    )

            def forward(
                self,
                seq,
                summary,
                anchor_z,
                residual_std,
            ):
                x = torch.cat(
                    [
                        seq.flatten(
                            start_dim=1
                        ),
                        summary,
                        anchor_z,
                    ],
                    dim=1,
                )

                raw = self.out(
                    self.net(x)
                )

                return self.bounded_correction(
                    raw,
                    residual_std,
                )

        return SafeDeepMLP

    if mode == "gru":
        class SafeGRU(
            nn.Module,
            SafeCorrectionMixin,
        ):
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
                        if FREEZE.gru_layers > 1
                        else 0.0
                    ),
                    batch_first=True,
                )

                self.head = nn.Sequential(
                    nn.Linear(
                        FREEZE.gru_hidden
                        + summary_dim
                        + anchor_dim,
                        FREEZE.gru_head1,
                    ),
                    nn.LayerNorm(
                        FREEZE.gru_head1
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        FREEZE.dropout
                    ),
                    nn.Linear(
                        FREEZE.gru_head1,
                        FREEZE.gru_head2,
                    ),
                    nn.GELU(),
                )

                self.out = nn.Linear(
                    FREEZE.gru_head2,
                    4,
                )

                nn.init.zeros_(
                    self.out.weight
                )
                nn.init.zeros_(
                    self.out.bias[
                        :2
                    ]
                )

                with torch.no_grad():
                    self.out.bias[
                        2:
                    ].fill_(
                        -2.0
                    )

            def forward(
                self,
                seq,
                summary,
                anchor_z,
                residual_std,
            ):
                z, _ = self.gru(
                    seq
                )

                h = z[:, -1, :]

                h = torch.cat(
                    [
                        h,
                        summary,
                        anchor_z,
                    ],
                    dim=1,
                )

                raw = self.out(
                    self.head(h)
                )

                return self.bounded_correction(
                    raw,
                    residual_std,
                )

        return SafeGRU

    raise ValueError(
        f"Unknown mode: {mode}"
    )


# =============================================================================
# Loss / prediction
# =============================================================================

def safe_loss(
    torch,
    final_pred,
    y_true,
    correction,
    residual_std,
    y_std,
):
    err = (
        final_pred
        - y_true
    )

    err_z = (
        err
        / y_std[
            None,
            :
        ]
    )

    loss_u = torch.mean(
        err_z[:, 0] ** 2
    )

    loss_v = torch.mean(
        err_z[:, 1] ** 2
    )

    component = (
        FREEZE.weight_u
        * loss_u
        + FREEZE.weight_v
        * loss_v
    )

    pred_speed = torch.linalg.vector_norm(
        final_pred,
        dim=1,
    )

    true_speed = torch.linalg.vector_norm(
        y_true,
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

    cos_sim = torch.sum(
        final_pred
        * y_true,
        dim=1,
    ) / denom

    cos_sim = torch.clamp(
        cos_sim,
        -1.0,
        1.0,
    )

    direction_mask = (
        true_speed
        >= 0.5
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
        correction
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

    return (
        total,
        {
            "component": component,
            "speed": speed_loss,
            "direction": direction_loss,
            "correction": correction_loss,
        },
    )


def predict_model(
    *,
    torch,
    model,
    seq_z,
    summary_z,
    anchor_z,
    ridge_anchor,
    residual_std,
    device,
    amp_enabled,
):
    model.eval()

    chunks = []
    gates = []

    with torch.no_grad():
        for (
            seq_np,
            summary_np,
            anchor_np,
            ridge_np,
        ) in sequential_batches(
            seq_z,
            summary_z,
            anchor_z,
            ridge_anchor,
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
                correction, gate = model(
                    seq_t,
                    summary_t,
                    anchor_t,
                    residual_std_t,
                )

            chunks.append(
                correction.detach()
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

    correction = np.concatenate(
        chunks,
        axis=0,
    )

    gate = np.concatenate(
        gates,
        axis=0,
    )

    final = (
        np.asarray(
            ridge_anchor,
            dtype=np.float32,
        )
        + correction
    ).astype(
        np.float32
    )

    return (
        final,
        correction,
        gate,
    )


# =============================================================================
# Train one development run
# =============================================================================

def train_dev_run(
    *,
    torch,
    nn,
    mode,
    correction_cap,
    seed,
    held_out,
    train_data,
    val_data,
    output_dir,
    device,
    debug_fast,
):
    candidate = (
        f"{mode}"
        f"_cap{int(round(100*correction_cap)):02d}"
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

    seq_tr, summary_tr = (
        build_residual_features(
            train_data[
                "X_full9"
            ]
        )
    )

    seq_va, summary_va = (
        build_residual_features(
            val_data[
                "X_full9"
            ]
        )
    )

    scalers = fit_correction_scalers(
        seq_tr,
        summary_tr,
        ridge_oof,
        ytr,
    )

    (
        seq_tr_z,
        summary_tr_z,
        anchor_tr_z,
    ) = transform_correction_inputs(
        seq_tr,
        summary_tr,
        ridge_oof,
        scalers,
    )

    (
        seq_va_z,
        summary_va_z,
        anchor_va_z,
    ) = transform_correction_inputs(
        seq_va,
        summary_va,
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
        mode,
        correction_cap,
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

    # epoch 0 exact Ridge
    initial_metrics = dict(
        base_metrics
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

    history = [
        {
            "epoch": 0,
            "selection_score": 1.0,
            "checkpoint_improved": True,
            **{
                f"val_{k}": v
                for k, v
                in initial_metrics.items()
            },
        }
    ]

    started = time.perf_counter()

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

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        model.train()

        loss_sum = 0.0
        n_seen = 0

        for (
            seq_np,
            summary_np,
            anchor_z_np,
            anchor_phys_np,
            y_np,
        ) in sequential_batches(
            seq_tr_z,
            summary_tr_z,
            anchor_tr_z,
            ridge_oof,
            ytr,
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

            anchor_z_t = torch.from_numpy(
                anchor_z_np
            ).to(
                device,
                non_blocking=True,
            )

            anchor_phys_t = torch.from_numpy(
                anchor_phys_np
            ).to(
                device,
                non_blocking=True,
            )

            y_t = torch.from_numpy(
                y_np
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
                correction, _ = model(
                    seq_t,
                    summary_t,
                    anchor_z_t,
                    residual_std_t,
                )

                final_pred = (
                    anchor_phys_t
                    + correction
                )

                loss, _ = safe_loss(
                    torch,
                    final_pred,
                    y_t,
                    correction,
                    residual_std_t,
                    y_std_t,
                )

            if not torch.isfinite(
                loss
            ):
                raise FloatingPointError(
                    "Nonfinite Stage12X loss."
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

            bn = len(seq_np)

            loss_sum += float(
                loss.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            n_seen += bn

        final_val, correction_val, gate_val = (
            predict_model(
                torch=torch,
                model=model,
                seq_z=seq_va_z,
                summary_z=summary_va_z,
                anchor_z=anchor_va_z,
                ridge_anchor=ridge_val,
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

        scheduler.step(
            score
        )

        diag = correction_diagnostics(
            yva,
            ridge_val,
            final_val,
        )

        history.append(
            {
                "epoch": int(epoch),
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
                "checkpoint_improved": bool(
                    improved
                ),
                "gate_U_mean": float(
                    np.mean(
                        gate_val[:, 0]
                    )
                ),
                "gate_V_mean": float(
                    np.mean(
                        gate_val[:, 1]
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
                f"corr=({diag['corr_residual_U']:+.3f},"
                f"{diag['corr_residual_V']:+.3f})"
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

    final_val, correction_val, gate_val = (
        predict_model(
            torch=torch,
            model=model,
            seq_z=seq_va_z,
            summary_z=summary_va_z,
            anchor_z=anchor_va_z,
            ridge_anchor=ridge_val,
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

    score, parts = selection_score(
        metrics,
        base_metrics,
    )

    diag = correction_diagnostics(
        yva,
        ridge_val,
        final_val,
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
            "stage": "12X-development",
            "candidate": candidate,
            "mode": mode,
            "correction_cap": float(
                correction_cap
            ),
            "held_out_mission": held_out,
            "seed": int(seed),
            "best_epoch": int(
                best_epoch
            ),
            "nn_params": int(
                nn_params
            ),
            "ridge_params": int(
                ridge_parameter_count(
                    ridge_model
                )
            ),
            "state_dict": best_state,
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
        "mode": mode,
        "correction_cap": float(
            correction_cap
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
        **parts,
        **diag,
        "gate_U_mean": float(
            np.mean(
                gate_val[:, 0]
            )
        ),
        "gate_V_mean": float(
            np.mean(
                gate_val[:, 1]
            )
        ),
        "history_path": str(
            history_path
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
    }


# =============================================================================
# Final fit on all development missions
# =============================================================================

def train_final_fixed_epochs(
    *,
    torch,
    nn,
    mode,
    correction_cap,
    seed,
    final_epochs,
    dev_data,
    tropical_data,
    output_dir,
    device,
):
    Xdev_base = build_base4(
        dev_data[
            "X_full9"
        ]
    )

    ydev = dev_data[
        "y_uv"
    ]

    Xtrop_base = build_base4(
        tropical_data[
            "X_full9"
        ]
    )

    ytrop = tropical_data[
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
        dev_data[
            "mission_labels"
        ],
    )

    seq_dev, summary_dev = (
        build_residual_features(
            dev_data[
                "X_full9"
            ]
        )
    )

    seq_trop, summary_trop = (
        build_residual_features(
            tropical_data[
                "X_full9"
            ]
        )
    )

    scalers = fit_correction_scalers(
        seq_dev,
        summary_dev,
        ridge_oof,
        ydev,
    )

    (
        seq_dev_z,
        summary_dev_z,
        anchor_dev_z,
    ) = transform_correction_inputs(
        seq_dev,
        summary_dev,
        ridge_oof,
        scalers,
    )

    (
        seq_trop_z,
        summary_trop_z,
        anchor_trop_z,
    ) = transform_correction_inputs(
        seq_trop,
        summary_trop,
        ridge_trop,
        scalers,
    )

    seed_everything(
        torch,
        seed,
    )

    ModelClass = build_model_class(
        torch,
        nn,
        mode,
        correction_cap,
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

    log("")
    log(
        f"[FINAL TRAIN] mode={mode} | "
        f"cap={correction_cap:.2f} | "
        f"seed={seed} | "
        f"epochs={final_epochs}"
    )

    for epoch in range(
        1,
        int(final_epochs) + 1,
    ):
        model.train()

        loss_sum = 0.0
        n_seen = 0

        for (
            seq_np,
            summary_np,
            anchor_z_np,
            anchor_phys_np,
            y_np,
        ) in sequential_batches(
            seq_dev_z,
            summary_dev_z,
            anchor_dev_z,
            ridge_oof,
            ydev,
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

            anchor_z_t = torch.from_numpy(
                anchor_z_np
            ).to(
                device,
                non_blocking=True,
            )

            anchor_phys_t = torch.from_numpy(
                anchor_phys_np
            ).to(
                device,
                non_blocking=True,
            )

            y_t = torch.from_numpy(
                y_np
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
                correction, _ = model(
                    seq_t,
                    summary_t,
                    anchor_z_t,
                    residual_std_t,
                )

                final_pred = (
                    anchor_phys_t
                    + correction
                )

                loss, _ = safe_loss(
                    torch,
                    final_pred,
                    y_t,
                    correction,
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

            bn = len(seq_np)

            loss_sum += float(
                loss.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            n_seen += bn

        if (
            epoch <= 5
            or epoch % 25 == 0
            or epoch == int(final_epochs)
        ):
            log(
                f"  final epoch={epoch:03d}/{int(final_epochs):03d} | "
                f"train_loss={loss_sum/max(n_seen,1):.6f}"
            )

    final_trop, correction_trop, gate_trop = (
        predict_model(
            torch=torch,
            model=model,
            seq_z=seq_trop_z,
            summary_z=summary_trop_z,
            anchor_z=anchor_trop_z,
            ridge_anchor=ridge_trop,
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

    final_metrics = evaluate_wind(
        ytrop,
        final_trop,
    )

    diag = correction_diagnostics(
        ytrop,
        ridge_trop,
        final_trop,
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
        / "12X_final_model.pt"
    )

    torch.save(
        {
            "stage": "12X-final",
            "mode": mode,
            "correction_cap": float(
                correction_cap
            ),
            "seed": int(seed),
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
            "state_dict": state_dict_cpu(
                model
            ),
            "correction_scalers": scalers,
            "freeze": asdict(
                FREEZE
            ),
        },
        checkpoint_path,
    )

    return {
        "ridge_metrics": ridge_metrics,
        "final_metrics": final_metrics,
        "diagnostics": {
            **diag,
            "gate_U_mean": float(
                np.mean(
                    gate_trop[:, 0]
                )
            ),
            "gate_V_mean": float(
                np.mean(
                    gate_trop[:, 1]
                )
            ),
        },
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
        "checkpoint_path": checkpoint_path,
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
            "Smoke test only. Antarctic holdout, one seed, deep_mlp cap0.15. "
            "Tropical Atlantic is NOT loaded in debug-fast."
        ),
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_root = resolve_dataset_root(
        args.dataset_dir
    )

    torch, nn = import_torch()
    configure_cuda(torch)

    if (
        args.force_cpu
        or not torch.cuda.is_available()
    ):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    log("=" * 132)
    log(
        "12X — RIDGE-ANCHORED SAFE U/V RESIDUAL + FINAL TROPICAL BENCHMARK"
    )
    log("=" * 132)
    log(
        f"dataset : {dataset_root}"
    )
    log(
        f"output  : {output_dir}"
    )
    log(
        "protocol : Stage12G point sampled; 6 x 10-min-spaced original 1-min means -> +10 min"
    )
    log(
        "anchor   : Base4-Ridge [U,V,T,RH]"
    )
    log(
        "residual : bounded U/V-only correction; no pressure, no wing, no trajectory"
    )
    log(
        f"device   : {device}"
    )

    if device.type == "cuda":
        log(
            f"GPU      : {torch.cuda.get_device_name(0)}"
        )

    # -------------------------------------------------------------
    # DEVELOPMENT DATA ONLY
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
        "[FIREWALL] Tropical Atlantic has NOT been loaded."
    )

    if args.debug_fast:
        active_holdouts = [
            "Antarctic"
        ]
        active_modes = [
            "deep_mlp"
        ]
        active_caps = [
            0.15
        ]
        active_seeds = [
            SEEDS[0]
        ]
    else:
        active_holdouts = DEV_MISSIONS
        active_modes = NETWORK_MODES
        active_caps = CORRECTION_CAPS
        active_seeds = SEEDS

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
                dev_missions[m]
                for m in train_names
            ]
        )

        val_data = dev_missions[
            held_out
        ]

        log("-" * 132)
        log(
            f"[DEV HOLDOUT] {held_out} | "
            f"train={train_names} | "
            f"Ntrain={len(train_data['y_uv']):,} | "
            f"Nval={len(val_data['y_uv']):,}"
        )

        for mode in active_modes:
            for cap in active_caps:
                for seed in active_seeds:
                    candidate = (
                        f"{mode}"
                        f"_cap{int(round(100*cap)):02d}"
                    )

                    log(
                        f"  [TRAIN] {candidate} | seed={seed}"
                    )

                    result = train_dev_run(
                        torch=torch,
                        nn=nn,
                        mode=mode,
                        correction_cap=cap,
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
                        f"  [DONE] "
                        f"score={result['selection_score']:.5f} | "
                        f"U={result['wind_U_RMSE_mps']:.4f} | "
                        f"V={result['wind_V_RMSE_mps']:.4f} | "
                        f"WS={result['wind_speed_RMSE_mps']:.4f} | "
                        f"WD={result['wind_direction_RMSE_deg']:.3f} | "
                        f"corr=({result['corr_residual_U']:+.3f},"
                        f"{result['corr_residual_V']:+.3f}) | "
                        f"bestEp={result['best_epoch']}"
                    )

    if args.debug_fast:
        log("")
        log(
            "[DONE] debug-fast completed. Tropical Atlantic was NOT loaded."
        )
        return 0

    # -------------------------------------------------------------
    # FREEZE CONFIGURATION FROM DEVELOPMENT ONLY
    # -------------------------------------------------------------
    log("")
    log(
        "[STAGE B] Freeze architecture/cap/seed/epochs from development LOMO only."
    )

    runs = pd.read_csv(
        run_path
    )

    summary_rows = []

    candidate_names = sorted(
        runs[
            "candidate"
        ].astype(str)
        .unique()
        .tolist()
    )

    for candidate in candidate_names:
        sub = runs.loc[
            runs[
                "candidate"
            ].astype(str)
            == candidate
        ].copy()

        summary_rows.append(
            {
                "candidate": candidate,
                "mode": str(
                    sub[
                        "mode"
                    ].iloc[0]
                ),
                "correction_cap": float(
                    sub[
                        "correction_cap"
                    ].iloc[0]
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
                "vector_RMSE_mean": float(
                    sub[
                        "wind_vector_RMSE_mps"
                    ].mean()
                ),
                "residual_corr_U_mean": float(
                    sub[
                        "corr_residual_U"
                    ].mean()
                ),
                "residual_corr_V_mean": float(
                    sub[
                        "corr_residual_V"
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

    selected_candidate = str(
        summary.iloc[0][
            "candidate"
        ]
    )

    selected_mode = str(
        summary.iloc[0][
            "mode"
        ]
    )

    selected_cap = float(
        summary.iloc[0][
            "correction_cap"
        ]
    )

    selected_runs = runs.loc[
        runs[
            "candidate"
        ].astype(str)
        == selected_candidate
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
        seed_summary.iloc[0][
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
        "selected_candidate": (
            selected_candidate
        ),
        "selected_mode": (
            selected_mode
        ),
        "selected_correction_cap": (
            selected_cap
        ),
        "selected_seed": (
            selected_seed
        ),
        "final_epochs": (
            final_epochs
        ),
        "selection_source": (
            "Antarctic/Atlantic/West Coast development LOMO only"
        ),
        "Tropical_Atlantic_used_for_selection": False,
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
        "DEVELOPMENT SUMMARY"
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
        f"candidate={selected_candidate} | "
        f"mode={selected_mode} | "
        f"cap={selected_cap:.2f} | "
        f"seed={selected_seed} | "
        f"epochs={final_epochs}"
    )
    log(
        f"[SAVED BEFORE TROPICAL LOAD] {frozen_path}"
    )

    # -------------------------------------------------------------
    # FINAL TROPICAL BENCHMARK
    # -------------------------------------------------------------
    log("")
    log(
        "[STAGE C] Configuration is frozen. Loading Tropical Atlantic NOW."
    )

    tropical = load_mission(
        dataset_root,
        FINAL_MISSION,
    )

    dev_all = concat_missions(
        [
            dev_missions[m]
            for m in DEV_MISSIONS
        ]
    )

    final = train_final_fixed_epochs(
        torch=torch,
        nn=nn,
        mode=selected_mode,
        correction_cap=selected_cap,
        seed=selected_seed,
        final_epochs=final_epochs,
        dev_data=dev_all,
        tropical_data=tropical,
        output_dir=output_dir,
        device=device,
    )

    ridge_metrics = final[
        "ridge_metrics"
    ]

    model_metrics = final[
        "final_metrics"
    ]

    diag = final[
        "diagnostics"
    ]

    rows = [
        {
            "model": "BP-STGNN-published",
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
            "model": "Base4-Ridge-final",
            "U_RMSE_mps": (
                ridge_metrics[
                    "wind_U_RMSE_mps"
                ]
            ),
            "V_RMSE_mps": (
                ridge_metrics[
                    "wind_V_RMSE_mps"
                ]
            ),
            "WS_RMSE_mps": (
                ridge_metrics[
                    "wind_speed_RMSE_mps"
                ]
            ),
            "WD_RMSE_deg": (
                ridge_metrics[
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
                "Ridge+SafeUVResidual-final"
            ),
            "U_RMSE_mps": (
                model_metrics[
                    "wind_U_RMSE_mps"
                ]
            ),
            "V_RMSE_mps": (
                model_metrics[
                    "wind_V_RMSE_mps"
                ]
            ),
            "WS_RMSE_mps": (
                model_metrics[
                    "wind_speed_RMSE_mps"
                ]
            ),
            "WD_RMSE_deg": (
                model_metrics[
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
        rows
    )

    final_path = (
        output_dir
        / "FINAL_TROPICAL_BENCHMARK.csv"
    )

    final_df.to_csv(
        final_path,
        index=False,
        encoding="utf-8-sig",
    )

    beats = {
        "U": bool(
            model_metrics[
                "wind_U_RMSE_mps"
            ]
            < BP_REFERENCE[
                "U_RMSE_mps"
            ]
        ),
        "V": bool(
            model_metrics[
                "wind_V_RMSE_mps"
            ]
            < BP_REFERENCE[
                "V_RMSE_mps"
            ]
        ),
        "WS": bool(
            model_metrics[
                "wind_speed_RMSE_mps"
            ]
            < BP_REFERENCE[
                "WS_RMSE_mps"
            ]
        ),
        "WD": bool(
            model_metrics[
                "wind_direction_RMSE_deg"
            ]
            < BP_REFERENCE[
                "WD_RMSE_deg"
            ]
        ),
    }

    report = {
        "stage": "12X",
        "script_version": (
            SCRIPT_VERSION
        ),
        "frozen_before_tropical": (
            frozen
        ),
        "final_tropical": {
            "ridge_metrics": (
                ridge_metrics
            ),
            "safe_residual_metrics": (
                model_metrics
            ),
            "safe_residual_diagnostics": (
                diag
            ),
            "beats_BP_metricwise": (
                beats
            ),
            "beats_BP_all_four": bool(
                all(
                    beats.values()
                )
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
        "paper_integrity_note": (
            "Tropical Atlantic was excluded from Stage-12X selection, "
            "but it was observed in earlier project experiments; do not "
            "describe it as a never-seen pristine test set."
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
            "12X FINAL TROPICAL ATLANTIC BENCHMARK\n"
        )
        f.write(
            "=" * 100
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
            "SAFE RESIDUAL DIAGNOSTICS\n"
        )
        f.write(
            "-" * 100
            + "\n"
        )
        for k, v in diag.items():
            f.write(
                f"{k}: {v}\n"
            )

        f.write(
            "\nBEATS BP-STGNN?\n"
        )
        f.write(
            "-" * 100
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
            f"Effective parameter fraction vs BP: "
            f"{100.0*report['final_tropical']['parameter_fraction_vs_BP']:.3f}%\n"
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
        "Safe residual diagnostics:"
    )
    log(
        f"  corr residual U = {diag['corr_residual_U']:+.4f}"
    )
    log(
        f"  corr residual V = {diag['corr_residual_V']:+.4f}"
    )
    log(
        f"  correction RMS U = {diag['correction_RMS_U_mps']:.5f} m/s"
    )
    log(
        f"  correction RMS V = {diag['correction_RMS_V_mps']:.5f} m/s"
    )
    log(
        f"  gate mean U = {diag['gate_U_mean']:.4f}"
    )
    log(
        f"  gate mean V = {diag['gate_V_mean']:.4f}"
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
        f"[SAVED] {final_path}"
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
