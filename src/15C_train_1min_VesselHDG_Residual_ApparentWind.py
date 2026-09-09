# -*- coding: utf-8 -*-
"""
15C_train_1min_VesselHDG_Residual_ApparentWind.py

STEP 3 — Vessel/HDG residual correction + apparent-wind physical coupling.

Frozen upstream components
--------------------------
Stage15B-F:
    final true-wind prediction
        W_true^(2)

Stage15A-F:
    Vessel+HDG Ridge anchor with alpha_Ridge = 2250

Stage15C learns only the residual of the Vessel+HDG Ridge anchor.

TRAIN inputs
------------
1) Sequence:
    all 11 existing joint-dataset features over 60 one-minute points

2) Future context (leakage-safe):
    Stage15B-F OOF true-wind prediction:
        [U_true^(2), V_true^(2)] x 5 horizons

    Stage15A-F OOF Vessel+HDG Ridge:
        [VE_R, VN_R, HDG_sin_R, HDG_cos_R] x 5 horizons

The future context therefore contains only upstream model predictions, not
future observations.

Residual targets
----------------
TRAIN uses OOF Ridge residuals:

    r = y_vessel_hdg - y_vessel_hdg_Ridge^OOF

Network
-------
    2-layer GRU
    hidden = 48
    dropout = 0.10

    last GRU state + standardized upstream future context
        -> shared MLP(64)
        -> 5 horizons x 4 residual channels

Output channels per horizon:
    dVE
    dVN
    dHDG_sin
    dHDG_cos

The residual output layer is zero initialized, so epoch 0 = Ridge.

Final adaptive shrinkage
------------------------
For each horizon:

    VE = VE_R + alpha_E(h) * dVE
    VN = VN_R + alpha_N(h) * dVN

Heading uses ONE shared alpha_psi(h) for the sin/cos residual pair:

    s = s_R + alpha_psi(h) * ds
    c = c_R + alpha_psi(h) * dc

then normalize:
    [s,c] <- [s,c] / sqrt(s^2+c^2)

Thus each horizon has:
    alpha_E(h), alpha_N(h), alpha_psi(h)

All alpha values are calibrated from VALIDATION only and clipped to [0,1].

Apparent-wind physics
---------------------
Earth-frame air-relative velocity:

    W_app^earth = W_true^(2) - V_vessel^(3)

Body axes:
    forward  = [sin(HDG),  cos(HDG)]
    starboard= [cos(HDG), -sin(HDG)]

Then:
    app_forward   = E*sin(HDG) + N*cos(HDG)
    app_starboard = E*cos(HDG) - N*sin(HDG)

AWS:
    sqrt(app_forward^2 + app_starboard^2)

Signed meteorological apparent-wind angle FROM relative to bow:
    AWA = atan2(-app_starboard, -app_forward)
    range [-180,180) deg

Apparent reference
------------------
The script first tries to locate an existing future apparent-vector reference
inside train/validation/test .npz files.

Candidate keys include:
    apparent_ref
    y_apparent_ref
    apparent_wind_ref
    y_apparent
    apparent_uv_ref

If a (N,5,2) candidate exists, TRAIN only is used to audit four coordinate
hypotheses:
    earth
    -earth
    body
    -body

The best convention is frozen and reused for validation/test.

If no apparent-vector array exists, the reference is derived from the future
targets:
    W_true_target - V_vessel_target

This fallback is explicitly reported.

Training loss
-------------
Primary residual loss:
    normalized OOF residual MSE

Auxiliary coupled losses:
    apparent earth-vector consistency
    heading unit-vector consistency
    body-frame apparent direction consistency

These auxiliary terms use Stage15B-F true wind, not the old TrueWind Ridge.

Model selection
---------------
Early stopping uses a validation-only dimensionless score:

    0.25 * VesselVectorRMSE / RidgeVesselVectorRMSE
  + 0.25 * HDG_MAE / RidgeHDG_MAE
  + 0.25 * AWS_RMSE / AnchorAWS_RMSE
  + 0.25 * AWA_MAE / AnchorAWA_MAE

where the apparent anchor is:
    Stage15B-F true wind + Stage15A-F Vessel/HDG Ridge

TEST is not used for:
    training
    epoch selection
    alpha calibration
    apparent convention selection

Recommended PowerShell
----------------------
python "D:\\project\\WindPredict_SaildroneData\\src\\15C_train_1min_VesselHDG_Residual_ApparentWind.py" --dataset-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\SD1090_TPOS2024_JointForecasting_v0_1" --stage15b-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15B_F_1min_TrueWind_Residual_v0_3" --stage15a-vessel-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15A_F_VesselHDG_alpha2250_v0_3" --output-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15C_1min_VesselHDG_ApparentResidual_v0_1"
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\project\WindPredict_SaildroneData")

DEFAULT_DATASET_DIR = (
    ROOT
    / "data"
    / "forecasting"
    / "SD1090_TPOS2024_JointForecasting_v0_1"
)

DEFAULT_STAGE15B_DIR = (
    ROOT
    / "data"
    / "forecasting"
    / "15B_F_1min_TrueWind_Residual_v0_3"
)

DEFAULT_STAGE15A_VESSEL_DIR = (
    ROOT
    / "data"
    / "forecasting"
    / "15A_F_VesselHDG_alpha2250_v0_3"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "data"
    / "forecasting"
    / "15C_1min_VesselHDG_ApparentResidual_v0_1"
)

HORIZONS = [1, 2, 3, 5, 10]

SEED = 500043

GRU_HIDDEN = 48
GRU_LAYERS = 2
SHARED_HIDDEN = 64
DROPOUT = 0.10

LR = 8e-4
WEIGHT_DECAY = 2e-5
BATCH_SIZE = 512
MAX_EPOCHS = 120
EARLY_STOP_PATIENCE = 18
GRAD_CLIP = 1.0

# Auxiliary loss weights. Residual MSE remains the dominant term.
LAMBDA_APP_VECTOR = 0.25
LAMBDA_HDG_CIRCLE = 0.15
LAMBDA_AWA_CIRCLE = 0.15

ALPHA_CAP = 1.0
EPS = 1e-12

APPARENT_CANDIDATE_KEYS = [
    "apparent_ref",
    "y_apparent_ref",
    "apparent_wind_ref",
    "y_apparent",
    "apparent_uv_ref",
    "apparent_wind",
    "y_app",
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
            return {str(k): cv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [cv(x) for x in v]
        return v

    with path.open("w", encoding="utf-8") as f:
        json.dump(cv(obj), f, ensure_ascii=False, indent=2, sort_keys=True)


def import_torch():
    import torch
    import torch.nn as nn
    return torch, nn


def seed_everything(torch, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch(torch):
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def amp_context(torch, device):
    if device.type != "cuda":
        return contextlib.nullcontext()

    try:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        )
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def create_grad_scaler(torch, device):
    if device.type != "cuda":
        return None

    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        try:
            return torch.cuda.amp.GradScaler(enabled=True)
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
# Dataset / upstream prediction loading
# =============================================================================

def load_dataset_split(dataset_dir: Path, split: str):
    path = dataset_dir / f"{split}.npz"

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as z:
        keys = list(z.files)

        X = np.asarray(
            z["X"],
            dtype=np.float32,
        )

        y = np.asarray(
            z["y_raw"],
            dtype=np.float32,
        )

        apparent = None
        apparent_key = None

        # First try explicit candidate names.
        for key in APPARENT_CANDIDATE_KEYS:
            if key in z.files:
                arr = np.asarray(
                    z[key],
                    dtype=np.float32,
                )

                if (
                    arr.ndim == 3
                    and arr.shape[1:] == (5, 2)
                    and len(arr) == len(X)
                ):
                    apparent = arr
                    apparent_key = key
                    break

        # Then try semantic key search.
        if apparent is None:
            for key in z.files:
                if "apparent" not in key.lower():
                    continue

                arr = np.asarray(
                    z[key],
                    dtype=np.float32,
                )

                if (
                    arr.ndim == 3
                    and arr.shape[1:] == (5, 2)
                    and len(arr) == len(X)
                ):
                    apparent = arr
                    apparent_key = key
                    break

    if X.ndim != 3 or X.shape[1:] != (60, 11):
        raise RuntimeError(
            f"{split}: expected X=(N,60,11), got {X.shape}"
        )

    if y.ndim != 3 or y.shape[1:] != (5, 6):
        raise RuntimeError(
            f"{split}: expected y_raw=(N,5,6), got {y.shape}"
        )

    if not (
        np.isfinite(X).all()
        and np.isfinite(y).all()
    ):
        raise RuntimeError(
            f"{split}: nonfinite X/y."
        )

    if apparent is not None and not np.isfinite(apparent).all():
        raise RuntimeError(
            f"{split}: apparent reference contains nonfinite values."
        )

    return {
        "X": X,
        "y": y,
        "apparent_candidate": apparent,
        "apparent_key": apparent_key,
        "npz_keys": keys,
        "path": path,
    }


def load_stage15b(stage15b_dir: Path):
    train_path = (
        stage15b_dir
        / "15B_F_train_stage2_oof_predictions.npz"
    )

    val_path = (
        stage15b_dir
        / "15B_F_validation_predictions.npz"
    )

    test_path = (
        stage15b_dir
        / "15B_F_test_predictions.npz"
    )

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    with np.load(train_path, allow_pickle=False) as z:
        train = np.asarray(
            z["stage2_truewind_oof"],
            dtype=np.float32,
        )

    with np.load(val_path, allow_pickle=False) as z:
        val = np.asarray(
            z["stage2_truewind"],
            dtype=np.float32,
        )

    with np.load(test_path, allow_pickle=False) as z:
        test = np.asarray(
            z["stage2_truewind"],
            dtype=np.float32,
        )

    for name, arr in [
        ("train", train),
        ("validation", val),
        ("test", test),
    ]:
        if arr.ndim != 3 or arr.shape[1:] != (5, 2):
            raise RuntimeError(
                f"Stage15B {name}: expected (N,5,2), got {arr.shape}"
            )

    return {
        "train": train,
        "validation": val,
        "test": test,
    }


def load_stage15a_vessel(stage15a_dir: Path):
    train_path = (
        stage15a_dir
        / "15A_F_train_vessel_hdg_oof.npz"
    )

    val_path = (
        stage15a_dir
        / "15A_F_validation_vessel_hdg.npz"
    )

    test_path = (
        stage15a_dir
        / "15A_F_test_vessel_hdg.npz"
    )

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    with np.load(train_path, allow_pickle=False) as z:
        train = {
            "y": np.asarray(
                z["vessel_hdg_y"],
                dtype=np.float32,
            ),
            "ridge": np.asarray(
                z["vessel_hdg_ridge_oof"],
                dtype=np.float32,
            ),
            "residual": np.asarray(
                z["vessel_hdg_residual_oof"],
                dtype=np.float32,
            ),
        }

    with np.load(val_path, allow_pickle=False) as z:
        val = {
            "y": np.asarray(
                z["vessel_hdg_y"],
                dtype=np.float32,
            ),
            "ridge": np.asarray(
                z["vessel_hdg_ridge"],
                dtype=np.float32,
            ),
            "residual": np.asarray(
                z["vessel_hdg_residual"],
                dtype=np.float32,
            ),
        }

    with np.load(test_path, allow_pickle=False) as z:
        test = {
            "y": np.asarray(
                z["vessel_hdg_y"],
                dtype=np.float32,
            ),
            "ridge": np.asarray(
                z["vessel_hdg_ridge"],
                dtype=np.float32,
            ),
            "residual": np.asarray(
                z["vessel_hdg_residual"],
                dtype=np.float32,
            ),
        }

    for name, pack in [
        ("train", train),
        ("validation", val),
        ("test", test),
    ]:
        for key in ["y", "ridge", "residual"]:
            arr = pack[key]

            if arr.ndim != 3 or arr.shape[1:] != (5, 4):
                raise RuntimeError(
                    f"Stage15A-F {name}/{key}: "
                    f"expected (N,5,4), got {arr.shape}"
                )

    return {
        "train": train,
        "validation": val,
        "test": test,
    }


# =============================================================================
# Geometry
# =============================================================================

def normalize_heading_np(sc):
    """
    sc[...,0] = sin(HDG)
    sc[...,1] = cos(HDG)
    """
    sc = np.asarray(
        sc,
        dtype=np.float64,
    )

    n = np.linalg.norm(
        sc,
        axis=-1,
        keepdims=True,
    )

    out = np.zeros_like(
        sc
    )

    good = (
        n[..., 0]
        > 1e-8
    )

    out[..., 0] = 0.0
    out[..., 1] = 1.0

    out[good] = (
        sc[good]
        / n[good]
    )

    return out


def earth_to_body_np(earth, heading_sc):
    """
    Earth vector = [East, North]
    Heading = [sin(psi), cos(psi)]

    body[...,0] = forward
    body[...,1] = starboard/right
    """
    e = np.asarray(
        earth[..., 0],
        dtype=np.float64,
    )

    n = np.asarray(
        earth[..., 1],
        dtype=np.float64,
    )

    hs = normalize_heading_np(
        heading_sc
    )

    s = hs[..., 0]
    c = hs[..., 1]

    fwd = (
        e * s
        + n * c
    )

    stb = (
        e * c
        - n * s
    )

    return np.stack(
        [
            fwd,
            stb,
        ],
        axis=-1,
    )


def body_to_earth_np(body, heading_sc):
    """
    Inverse of earth_to_body_np.
    """
    fwd = np.asarray(
        body[..., 0],
        dtype=np.float64,
    )

    stb = np.asarray(
        body[..., 1],
        dtype=np.float64,
    )

    hs = normalize_heading_np(
        heading_sc
    )

    s = hs[..., 0]
    c = hs[..., 1]

    e = (
        fwd * s
        + stb * c
    )

    n = (
        fwd * c
        - stb * s
    )

    return np.stack(
        [
            e,
            n,
        ],
        axis=-1,
    )


def awa_deg_from_body_np(body):
    """
    Signed meteorological FROM angle relative to bow.
    [-180,180) deg.
    """
    fwd = np.asarray(
        body[..., 0],
        dtype=np.float64,
    )

    stb = np.asarray(
        body[..., 1],
        dtype=np.float64,
    )

    return (
        (
            np.degrees(
                np.arctan2(
                    -stb,
                    -fwd,
                )
            )
            + 180.0
        )
        % 360.0
        - 180.0
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


# =============================================================================
# Apparent-reference convention audit
# =============================================================================

def true_future_components(dataset_split):
    y = dataset_split["y"]

    true_wind = y[:, :, 0:2]
    vessel = y[:, :, 2:4]
    heading = y[:, :, 4:6]

    return (
        true_wind.astype(
            np.float64
        ),
        vessel.astype(
            np.float64
        ),
        heading.astype(
            np.float64
        ),
    )


def audit_apparent_reference(train_ds):
    """
    Returns a frozen convention descriptor.

    Canonical target used by Stage15C is always earth-frame apparent velocity:
        Wtrue - Vvessel
    """
    true_wind, vessel, heading = true_future_components(
        train_ds
    )

    derived_earth = (
        true_wind
        - vessel
    )

    candidate = train_ds[
        "apparent_candidate"
    ]

    if candidate is None:
        return {
            "source": "derived_from_y_raw",
            "key": None,
            "convention": "derived_earth",
            "train_audit_rmse": 0.0,
            "use_measured_candidate": False,
        }

    cand = np.asarray(
        candidate,
        dtype=np.float64,
    )

    derived_body = earth_to_body_np(
        derived_earth,
        heading,
    )

    hypotheses = {
        "candidate_is_earth": cand,
        "candidate_is_negative_earth": -cand,
        "candidate_is_body": body_to_earth_np(
            cand,
            heading,
        ),
        "candidate_is_negative_body": body_to_earth_np(
            -cand,
            heading,
        ),
    }

    rows = []

    for name, canonical in hypotheses.items():
        rmse = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (
                            canonical
                            - derived_earth
                        )
                        ** 2,
                        axis=-1,
                    )
                )
            )
        )

        rows.append(
            (
                name,
                rmse,
            )
        )

    rows.sort(
        key=lambda x: x[1]
    )

    best_name, best_rmse = rows[0]

    # If the existing vector is wildly inconsistent, use the physically
    # derived target rather than silently accepting an unknown 2-column format.
    use_candidate = bool(
        np.isfinite(
            best_rmse
        )
        and best_rmse
        <= 2.0
    )

    return {
        "source": (
            "dataset_apparent_candidate"
            if use_candidate
            else "derived_from_y_raw_fallback"
        ),
        "key": train_ds[
            "apparent_key"
        ],
        "convention": (
            best_name
            if use_candidate
            else "derived_earth"
        ),
        "train_audit_rmse": float(
            best_rmse
        ),
        "all_hypotheses": [
            {
                "name": name,
                "rmse": rmse,
            }
            for name, rmse in rows
        ],
        "use_measured_candidate": use_candidate,
    }


def canonical_apparent_reference(dataset_split, audit):
    true_wind, vessel, heading = true_future_components(
        dataset_split
    )

    derived_earth = (
        true_wind
        - vessel
    )

    if not audit[
        "use_measured_candidate"
    ]:
        return derived_earth.astype(
            np.float32
        )

    candidate = dataset_split[
        "apparent_candidate"
    ]

    if candidate is None:
        raise RuntimeError(
            "TRAIN selected dataset apparent reference, "
            "but this split does not contain the candidate."
        )

    cand = np.asarray(
        candidate,
        dtype=np.float64,
    )

    convention = audit[
        "convention"
    ]

    if convention == "candidate_is_earth":
        earth = cand

    elif convention == "candidate_is_negative_earth":
        earth = -cand

    elif convention == "candidate_is_body":
        earth = body_to_earth_np(
            cand,
            heading,
        )

    elif convention == "candidate_is_negative_body":
        earth = body_to_earth_np(
            -cand,
            heading,
        )

    else:
        raise RuntimeError(
            f"Unknown apparent convention: {convention}"
        )

    return earth.astype(
        np.float32
    )


# =============================================================================
# Feature/context scaling
# =============================================================================

def build_sequence_features(X):
    """
    Keep the vessel branch close to the original joint model:
    all 11 input channels are used directly.

    X is already standardized by the original dataset builder.
    """
    seq = np.asarray(
        X,
        dtype=np.float32,
    )

    if not np.isfinite(seq).all():
        raise RuntimeError(
            "Nonfinite sequence feature."
        )

    return seq


def build_future_context(stage2_truewind, vessel_ridge):
    """
    Per horizon:
        Stage2 U,V
        Ridge VE,VN,HDG_sin,HDG_cos
    -> 6 channels x 5 horizons = 30 static context values.
    """
    tw = np.asarray(
        stage2_truewind,
        dtype=np.float32,
    )

    vr = np.asarray(
        vessel_ridge,
        dtype=np.float32,
    )

    context = np.concatenate(
        [
            tw,
            vr,
        ],
        axis=-1,
    )

    return context.reshape(
        len(context),
        -1,
    ).astype(
        np.float32
    )


def fit_scalers(seq, context, residual):
    seq64 = np.asarray(
        seq,
        dtype=np.float64,
    )

    ctx64 = np.asarray(
        context,
        dtype=np.float64,
    )

    res64 = np.asarray(
        residual,
        dtype=np.float64,
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

    context_mean = np.mean(
        ctx64,
        axis=0,
    )

    context_std = np.std(
        ctx64,
        axis=0,
        ddof=0,
    )

    context_std = np.where(
        context_std < 1e-8,
        1.0,
        context_std,
    )

    residual_std = np.std(
        res64,
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
        "context_mean": context_mean.astype(
            np.float32
        ),
        "context_std": context_std.astype(
            np.float32
        ),
        "residual_std": residual_std.astype(
            np.float32
        ),
    }


def transform_inputs(seq, context, scaler):
    seq_t = (
        (
            seq
            - scaler[
                "seq_mean"
            ][
                None,
                None,
                :
            ]
        )
        / scaler[
            "seq_std"
        ][
            None,
            None,
            :
        ]
    ).astype(
        np.float32
    )

    ctx_t = (
        (
            context
            - scaler[
                "context_mean"
            ][
                None,
                :
            ]
        )
        / scaler[
            "context_std"
        ][
            None,
            :
        ]
    ).astype(
        np.float32
    )

    return (
        seq_t,
        ctx_t,
    )


# =============================================================================
# Alpha calibration
# =============================================================================

def analytic_alphas(y_true, ridge, raw_residual):
    """
    alpha_E(h), alpha_N(h), alpha_psi(h).

    Heading alpha is shared by sin/cos and minimizes their combined
    pre-normalization squared error.
    """
    y = np.asarray(
        y_true,
        dtype=np.float64,
    )

    r = np.asarray(
        ridge,
        dtype=np.float64,
    )

    d = np.asarray(
        raw_residual,
        dtype=np.float64,
    )

    true_res = (
        y - r
    )

    alpha_motion = np.zeros(
        (
            len(HORIZONS),
            2,
        ),
        dtype=np.float64,
    )

    alpha_heading = np.zeros(
        len(HORIZONS),
        dtype=np.float64,
    )

    for h in range(
        len(HORIZONS)
    ):
        for c in range(2):
            rr = d[
                :,
                h,
                c
            ]

            ee = true_res[
                :,
                h,
                c
            ]

            denom = float(
                np.dot(
                    rr,
                    rr,
                )
            )

            a = (
                0.0
                if denom <= EPS
                else float(
                    np.dot(
                        ee,
                        rr,
                    )
                    / denom
                )
            )

            alpha_motion[
                h,
                c
            ] = np.clip(
                a,
                0.0,
                ALPHA_CAP,
            )

        rr_sc = d[
            :,
            h,
            2:4
        ].reshape(
            -1
        )

        ee_sc = true_res[
            :,
            h,
            2:4
        ].reshape(
            -1
        )

        denom = float(
            np.dot(
                rr_sc,
                rr_sc,
            )
        )

        a = (
            0.0
            if denom <= EPS
            else float(
                np.dot(
                    ee_sc,
                    rr_sc,
                )
                / denom
            )
        )

        alpha_heading[
            h
        ] = np.clip(
            a,
            0.0,
            ALPHA_CAP,
        )

    return (
        alpha_motion.astype(
            np.float32
        ),
        alpha_heading.astype(
            np.float32
        ),
    )


def apply_alphas_np(
    ridge,
    raw_residual,
    alpha_motion,
    alpha_heading,
):
    r = np.asarray(
        ridge,
        dtype=np.float64,
    )

    d = np.asarray(
        raw_residual,
        dtype=np.float64,
    )

    out = r.copy()

    out[
        :,
        :,
        0:2
    ] = (
        r[
            :,
            :,
            0:2
        ]
        + d[
            :,
            :,
            0:2
        ]
        * np.asarray(
            alpha_motion,
            dtype=np.float64,
        )[
            None,
            :,
            :
        ]
    )

    sc = (
        r[
            :,
            :,
            2:4
        ]
        + d[
            :,
            :,
            2:4
        ]
        * np.asarray(
            alpha_heading,
            dtype=np.float64,
        )[
            None,
            :,
            None
        ]
    )

    out[
        :,
        :,
        2:4
    ] = normalize_heading_np(
        sc
    )

    return out.astype(
        np.float32
    )


# =============================================================================
# Metrics
# =============================================================================

def apparent_outputs(
    stage2_truewind,
    vessel_hdg_pred,
):
    vh = np.asarray(
        vessel_hdg_pred,
        dtype=np.float64,
    )

    wind = np.asarray(
        stage2_truewind,
        dtype=np.float64,
    )

    vessel = vh[
        :,
        :,
        0:2
    ]

    heading = vh[
        :,
        :,
        2:4
    ]

    app_earth = (
        wind
        - vessel
    )

    app_body = earth_to_body_np(
        app_earth,
        heading,
    )

    aws = np.linalg.norm(
        app_body,
        axis=-1,
    )

    awa = awa_deg_from_body_np(
        app_body
    )

    return {
        "earth": app_earth,
        "body": app_body,
        "AWS": aws,
        "AWA": awa,
    }


def reference_apparent_outputs(
    app_ref_earth,
    y_vessel_hdg,
):
    y = np.asarray(
        y_vessel_hdg,
        dtype=np.float64,
    )

    true_heading = y[
        :,
        :,
        2:4
    ]

    body = earth_to_body_np(
        app_ref_earth,
        true_heading,
    )

    return {
        "earth": np.asarray(
            app_ref_earth,
            dtype=np.float64,
        ),
        "body": body,
        "AWS": np.linalg.norm(
            body,
            axis=-1,
        ),
        "AWA": awa_deg_from_body_np(
            body
        ),
    }


def metrics_per_horizon(
    y_true_vh,
    pred_vh,
    stage2_truewind,
    app_ref_earth,
    split,
    model,
):
    y = np.asarray(
        y_true_vh,
        dtype=np.float64,
    )

    p = np.asarray(
        pred_vh,
        dtype=np.float64,
    )

    p[
        :,
        :,
        2:4
    ] = normalize_heading_np(
        p[
            :,
            :,
            2:4
        ]
    )

    app_pred = apparent_outputs(
        stage2_truewind,
        p,
    )

    app_ref = reference_apparent_outputs(
        app_ref_earth,
        y,
    )

    rows = []

    for j, h in enumerate(
        HORIZONS
    ):
        t = y[
            :,
            j,
            :
        ]

        q = p[
            :,
            j,
            :
        ]

        ev = (
            q[
                :,
                0:2
            ]
            - t[
                :,
                0:2
            ]
        )

        sog_t = np.linalg.norm(
            t[
                :,
                0:2
            ],
            axis=1,
        )

        sog_p = np.linalg.norm(
            q[
                :,
                0:2
            ],
            axis=1,
        )

        hdg_t = (
            np.degrees(
                np.arctan2(
                    t[
                        :,
                        2
                    ],
                    t[
                        :,
                        3
                    ],
                )
            )
            % 360.0
        )

        hdg_p = (
            np.degrees(
                np.arctan2(
                    q[
                        :,
                        2
                    ],
                    q[
                        :,
                        3
                    ],
                )
            )
            % 360.0
        )

        hdg_err = circular_diff_deg(
            hdg_p,
            hdg_t,
        )

        app_e = (
            app_pred[
                "earth"
            ][
                :,
                j,
                :
            ]
            - app_ref[
                "earth"
            ][
                :,
                j,
                :
            ]
        )

        aws_err = (
            app_pred[
                "AWS"
            ][
                :,
                j
            ]
            - app_ref[
                "AWS"
            ][
                :,
                j
            ]
        )

        awa_err = circular_diff_deg(
            app_pred[
                "AWA"
            ][
                :,
                j
            ],
            app_ref[
                "AWA"
            ][
                :,
                j
            ],
        )

        rows.append(
            {
                "split": split,
                "model": model,
                "horizon_min": h,
                "Vessel_East_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            ev[
                                :,
                                0
                            ]
                            ** 2
                        )
                    )
                ),
                "Vessel_North_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            ev[
                                :,
                                1
                            ]
                            ** 2
                        )
                    )
                ),
                "Vessel_vector_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                ev
                                ** 2,
                                axis=1,
                            )
                        )
                    )
                ),
                "SOG_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            (
                                sog_p
                                - sog_t
                            )
                            ** 2
                        )
                    )
                ),
                "HDG_MAE_deg": float(
                    np.mean(
                        np.abs(
                            hdg_err
                        )
                    )
                ),
                "HDG_RMSE_deg": float(
                    np.sqrt(
                        np.mean(
                            hdg_err
                            ** 2
                        )
                    )
                ),
                "AppVector_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                app_e
                                ** 2,
                                axis=1,
                            )
                        )
                    )
                ),
                "AWS_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            aws_err
                            ** 2
                        )
                    )
                ),
                "AWA_MAE_deg": float(
                    np.mean(
                        np.abs(
                            awa_err
                        )
                    )
                ),
                "AWA_RMSE_deg": float(
                    np.sqrt(
                        np.mean(
                            awa_err
                            ** 2
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def validation_score(
    y_true,
    ridge,
    final_pred,
    stage2_truewind,
    app_ref_earth,
):
    base = metrics_per_horizon(
        y_true,
        ridge,
        stage2_truewind,
        app_ref_earth,
        "validation",
        "Anchor",
    )

    final = metrics_per_horizon(
        y_true,
        final_pred,
        stage2_truewind,
        app_ref_earth,
        "validation",
        "Stage15C",
    )

    def ratio(col):
        denom = np.maximum(
            base[
                col
            ].to_numpy(
                dtype=np.float64
            ),
            1e-8,
        )

        num = final[
            col
        ].to_numpy(
            dtype=np.float64
        )

        return float(
            np.mean(
                num
                / denom
            )
        )

    parts = {
        "vessel_vector_ratio": ratio(
            "Vessel_vector_RMSE_mps"
        ),
        "hdg_mae_ratio": ratio(
            "HDG_MAE_deg"
        ),
        "aws_ratio": ratio(
            "AWS_RMSE_mps"
        ),
        "awa_mae_ratio": ratio(
            "AWA_MAE_deg"
        ),
    }

    score = float(
        0.25
        * (
            parts[
                "vessel_vector_ratio"
            ]
            + parts[
                "hdg_mae_ratio"
            ]
            + parts[
                "aws_ratio"
            ]
            + parts[
                "awa_mae_ratio"
            ]
        )
    )

    return (
        score,
        parts,
    )


# =============================================================================
# Torch model / differentiable geometry
# =============================================================================

def build_model_class(torch, nn):
    class VesselHDGResidualGRU(nn.Module):
        def __init__(self):
            super().__init__()

            self.gru = nn.GRU(
                input_size=11,
                hidden_size=GRU_HIDDEN,
                num_layers=GRU_LAYERS,
                batch_first=True,
                dropout=(
                    DROPOUT
                    if GRU_LAYERS > 1
                    else 0.0
                ),
            )

            self.shared = nn.Sequential(
                nn.Linear(
                    GRU_HIDDEN
                    + len(HORIZONS)
                    * 6,
                    SHARED_HIDDEN,
                ),
                nn.GELU(),
                nn.Dropout(
                    DROPOUT
                ),
            )

            self.out = nn.Linear(
                SHARED_HIDDEN,
                len(HORIZONS)
                * 4,
            )

            # Epoch 0 = exact Ridge.
            nn.init.zeros_(
                self.out.weight
            )

            nn.init.zeros_(
                self.out.bias
            )

        def forward(
            self,
            seq,
            context,
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
                        context,
                    ],
                    dim=1,
                )
            )

            out = torch.tanh(
                self.out(
                    h
                )
            )

            return out.reshape(
                -1,
                len(HORIZONS),
                4,
            )

    return VesselHDGResidualGRU


def torch_normalize_heading(torch, sc):
    n = torch.sqrt(
        torch.sum(
            sc
            * sc,
            dim=-1,
            keepdim=True,
        )
        + 1e-8
    )

    return sc / n


def torch_earth_to_body(torch, earth, heading_sc):
    hs = torch_normalize_heading(
        torch,
        heading_sc,
    )

    e = earth[
        ...,
        0
    ]

    n = earth[
        ...,
        1
    ]

    s = hs[
        ...,
        0
    ]

    c = hs[
        ...,
        1
    ]

    fwd = (
        e * s
        + n * c
    )

    stb = (
        e * c
        - n * s
    )

    return torch.stack(
        [
            fwd,
            stb,
        ],
        dim=-1,
    )


def torch_direction_cosine_loss(
    torch,
    a,
    b,
):
    an = a / (
        torch.sqrt(
            torch.sum(
                a * a,
                dim=-1,
                keepdim=True,
            )
            + 1e-8
        )
    )

    bn = b / (
        torch.sqrt(
            torch.sum(
                b * b,
                dim=-1,
                keepdim=True,
            )
            + 1e-8
        )
    )

    cos = torch.sum(
        an * bn,
        dim=-1,
    )

    return torch.mean(
        1.0 - cos
    )


# =============================================================================
# Prediction
# =============================================================================

def predict_raw(
    torch,
    model,
    seq_t,
    ctx_t,
    residual_std,
    device,
):
    model.eval()

    outputs = []

    with torch.no_grad():
        for start in range(
            0,
            len(seq_t),
            BATCH_SIZE,
        ):
            stop = min(
                start
                + BATCH_SIZE,
                len(seq_t),
            )

            s = torch.from_numpy(
                seq_t[
                    start:
                    stop
                ]
            ).to(
                device,
                non_blocking=True,
            )

            c = torch.from_numpy(
                ctx_t[
                    start:
                    stop
                ]
            ).to(
                device,
                non_blocking=True,
            )

            with amp_context(
                torch,
                device,
            ):
                pred_norm = model(
                    s,
                    c,
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
        * residual_std[
            None,
            :,
            :
        ]
    ).astype(
        np.float32
    )


# =============================================================================
# Training
# =============================================================================

def train_stage15c(
    *,
    torch,
    nn,
    train_seq,
    train_context,
    train_residual,
    train_ridge,
    train_stage2,
    train_y,
    train_app_ref,
    val_seq,
    val_context,
    val_ridge,
    val_stage2,
    val_y,
    val_app_ref,
):
    seed_everything(
        torch,
        SEED,
    )

    scaler = fit_scalers(
        train_seq,
        train_context,
        train_residual,
    )

    tr_seq, tr_ctx = transform_inputs(
        train_seq,
        train_context,
        scaler,
    )

    va_seq, va_ctx = transform_inputs(
        val_seq,
        val_context,
        scaler,
    )

    target_norm = (
        np.asarray(
            train_residual,
            dtype=np.float32,
        )
        / scaler[
            "residual_std"
        ][
            None,
            :,
            :
        ]
    ).astype(
        np.float32
    )

    # Normalize apparent vector consistency by TRAIN anchor error per horizon.
    train_anchor_app = (
        train_stage2
        - train_ridge[
            :,
            :,
            0:2
        ]
    )

    train_app_error = (
        np.asarray(
            train_anchor_app,
            dtype=np.float64,
        )
        - np.asarray(
            train_app_ref,
            dtype=np.float64,
        )
    )

    app_scale = np.sqrt(
        np.mean(
            np.sum(
                train_app_error
                ** 2,
                axis=-1,
            ),
            axis=0,
        )
    )

    app_scale = np.where(
        app_scale < 1e-3,
        1.0,
        app_scale,
    ).astype(
        np.float32
    )

    Model = build_model_class(
        torch,
        nn,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = Model().to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    grad_scaler = create_grad_scaler(
        torch,
        device,
    )

    best_state = state_dict_cpu(
        model
    )

    best_epoch = 0

    # Epoch 0 baseline = Ridge anchor.
    best_score, best_parts = validation_score(
        val_y,
        val_ridge,
        val_ridge,
        val_stage2,
        val_app_ref,
    )

    best_alpha_motion = np.zeros(
        (
            len(HORIZONS),
            2,
        ),
        dtype=np.float32,
    )

    best_alpha_heading = np.zeros(
        len(HORIZONS),
        dtype=np.float32,
    )

    patience = 0

    history = []

    n = len(
        train_seq
    )

    residual_std_t = torch.from_numpy(
        scaler[
            "residual_std"
        ]
    ).to(
        device
    )

    app_scale_t = torch.from_numpy(
        app_scale
    ).to(
        device
    )

    # Raw train arrays used by coupled losses.
    tr_ridge_np = np.asarray(
        train_ridge,
        dtype=np.float32,
    )

    tr_stage2_np = np.asarray(
        train_stage2,
        dtype=np.float32,
    )

    tr_y_np = np.asarray(
        train_y,
        dtype=np.float32,
    )

    tr_app_ref_np = np.asarray(
        train_app_ref,
        dtype=np.float32,
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        rng = np.random.default_rng(
            SEED
            + epoch
            * 1009
        )

        order = np.arange(
            n
        )

        rng.shuffle(
            order
        )

        running_total = 0.0
        running_res = 0.0
        running_app = 0.0
        running_hdg = 0.0
        running_awa = 0.0
        seen = 0

        for start in range(
            0,
            n,
            BATCH_SIZE,
        ):
            idx = order[
                start:
                start
                + BATCH_SIZE
            ]

            seq_b = torch.from_numpy(
                tr_seq[
                    idx
                ]
            ).to(
                device,
                non_blocking=True,
            )

            ctx_b = torch.from_numpy(
                tr_ctx[
                    idx
                ]
            ).to(
                device,
                non_blocking=True,
            )

            target_b = torch.from_numpy(
                target_norm[
                    idx
                ]
            ).to(
                device,
                non_blocking=True,
            )

            ridge_b = torch.from_numpy(
                tr_ridge_np[
                    idx
                ]
            ).to(
                device,
                non_blocking=True,
            )

            stage2_b = torch.from_numpy(
                tr_stage2_np[
                    idx
                ]
            ).to(
                device,
                non_blocking=True,
            )

            y_b = torch.from_numpy(
                tr_y_np[
                    idx
                ]
            ).to(
                device,
                non_blocking=True,
            )

            app_ref_b = torch.from_numpy(
                tr_app_ref_np[
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
                device,
            ):
                pred_norm = model(
                    seq_b,
                    ctx_b,
                )

                loss_res = torch.mean(
                    (
                        pred_norm
                        - target_b
                    )
                    ** 2
                )

                raw_res = (
                    pred_norm
                    * residual_std_t[
                        None,
                        :,
                        :
                    ]
                )

                vessel_pred = (
                    ridge_b[
                        :,
                        :,
                        0:2
                    ]
                    + raw_res[
                        :,
                        :,
                        0:2
                    ]
                )

                hdg_pred = torch_normalize_heading(
                    torch,
                    (
                        ridge_b[
                            :,
                            :,
                            2:4
                        ]
                        + raw_res[
                            :,
                            :,
                            2:4
                        ]
                    ),
                )

                app_pred_earth = (
                    stage2_b
                    - vessel_pred
                )

                app_err = (
                    app_pred_earth
                    - app_ref_b
                )

                loss_app = torch.mean(
                    torch.sum(
                        app_err
                        * app_err,
                        dim=-1,
                    )
                    / (
                        app_scale_t[
                            None,
                            :
                        ]
                        ** 2
                    )
                )

                true_hdg = torch_normalize_heading(
                    torch,
                    y_b[
                        :,
                        :,
                        2:4
                    ],
                )

                loss_hdg = torch.mean(
                    1.0
                    - torch.sum(
                        hdg_pred
                        * true_hdg,
                        dim=-1,
                    )
                )

                app_pred_body = torch_earth_to_body(
                    torch,
                    app_pred_earth,
                    hdg_pred,
                )

                app_ref_body = torch_earth_to_body(
                    torch,
                    app_ref_b,
                    true_hdg,
                )

                # Direction is the same whether using TO or FROM vectors,
                # because both sides are multiplied by -1 for FROM.
                loss_awa = torch_direction_cosine_loss(
                    torch,
                    app_pred_body,
                    app_ref_body,
                )

                loss = (
                    loss_res
                    + LAMBDA_APP_VECTOR
                    * loss_app
                    + LAMBDA_HDG_CIRCLE
                    * loss_hdg
                    + LAMBDA_AWA_CIRCLE
                    * loss_awa
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
                    GRAD_CLIP,
                )

                grad_scaler.step(
                    optimizer
                )

                grad_scaler.update()

            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                optimizer.step()

            bn = len(
                idx
            )

            running_total += float(
                loss.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            running_res += float(
                loss_res.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            running_app += float(
                loss_app.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            running_hdg += float(
                loss_hdg.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            running_awa += float(
                loss_awa.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            seen += bn

        raw_val = predict_raw(
            torch,
            model,
            va_seq,
            va_ctx,
            scaler[
                "residual_std"
            ],
            device,
        )

        alpha_motion, alpha_heading = analytic_alphas(
            val_y,
            val_ridge,
            raw_val,
        )

        final_val = apply_alphas_np(
            val_ridge,
            raw_val,
            alpha_motion,
            alpha_heading,
        )

        score, parts = validation_score(
            val_y,
            val_ridge,
            final_val,
            val_stage2,
            val_app_ref,
        )

        improved = (
            score
            < best_score
            - 1e-7
        )

        if improved:
            best_score = score
            best_parts = parts
            best_epoch = epoch
            best_state = state_dict_cpu(
                model
            )
            best_alpha_motion = alpha_motion.copy()
            best_alpha_heading = alpha_heading.copy()
            patience = 0

        else:
            patience += 1

        history.append(
            {
                "epoch": epoch,
                "train_total_loss": running_total / seen,
                "train_residual_loss": running_res / seen,
                "train_app_vector_loss": running_app / seen,
                "train_hdg_circle_loss": running_hdg / seen,
                "train_awa_circle_loss": running_awa / seen,
                "validation_score": score,
                "best_validation_score": best_score,
                **{
                    f"val_{k}": v
                    for k, v in parts.items()
                },
            }
        )

        if (
            epoch <= 3
            or epoch % 10 == 0
            or improved
        ):
            log(
                f"  epoch={epoch:03d} | "
                f"loss={running_total/seen:.6f} | "
                f"val_score={score:.6f}"
                + (
                    " *"
                    if improved
                    else ""
                )
            )

        if patience >= EARLY_STOP_PATIENCE:
            break

    model.load_state_dict(
        best_state,
        strict=True,
    )

    return {
        "model": model,
        "scaler": scaler,
        "device": device,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_parts": best_parts,
        "alpha_motion": best_alpha_motion,
        "alpha_heading": best_alpha_heading,
        "history": pd.DataFrame(
            history
        ),
        "app_scale": app_scale,
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
        "--stage15b-dir",
        type=Path,
        default=DEFAULT_STAGE15B_DIR,
    )

    parser.add_argument(
        "--stage15a-vessel-dir",
        type=Path,
        default=DEFAULT_STAGE15A_VESSEL_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    out = args.output_dir

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch, nn = import_torch()

    configure_torch(
        torch
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    log("=" * 150)
    log(
        "15C — STEP 3: VESSEL/HDG RESIDUAL + APPARENT-WIND PHYSICAL COUPLING"
    )
    log("=" * 150)

    log(
        "Frozen true wind: Stage15B-F"
    )

    log(
        "Frozen Vessel+HDG Ridge alpha: 2250"
    )

    log(
        f"Residual network: 2-layer GRU, hidden={GRU_HIDDEN}"
    )

    log(
        f"device = {device}"
    )

    # -----------------------------------------------------------------
    # Load.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 1/8] Loading dataset + frozen upstream predictions."
    )

    ds = {
        split: load_dataset_split(
            args.dataset_dir,
            split,
        )
        for split in [
            "train",
            "validation",
            "test",
        ]
    }

    stage2 = load_stage15b(
        args.stage15b_dir
    )

    vessel_pack = load_stage15a_vessel(
        args.stage15a_vessel_dir
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:
        n = len(
            ds[
                split
            ][
                "X"
            ]
        )

        if len(
            stage2[
                split
            ]
        ) != n:
            raise RuntimeError(
                f"{split}: Stage15B length mismatch."
            )

        if len(
            vessel_pack[
                split
            ][
                "y"
            ]
        ) != n:
            raise RuntimeError(
                f"{split}: Stage15A-F length mismatch."
            )

    # -----------------------------------------------------------------
    # Apparent reference audit.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 2/8] TRAIN-only apparent-vector convention audit."
    )

    apparent_audit = audit_apparent_reference(
        ds[
            "train"
        ]
    )

    log(
        f"  source     = {apparent_audit['source']}"
    )

    log(
        f"  key        = {apparent_audit['key']}"
    )

    log(
        f"  convention = {apparent_audit['convention']}"
    )

    log(
        f"  TRAIN audit vector RMSE = "
        f"{apparent_audit['train_audit_rmse']:.6f}"
    )

    app_ref = {
        split: canonical_apparent_reference(
            ds[
                split
            ],
            apparent_audit,
        )
        for split in [
            "train",
            "validation",
            "test",
        ]
    }

    # -----------------------------------------------------------------
    # Build model inputs.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 3/8] Building leakage-safe Stage15C inputs."
    )

    seq = {
        split: build_sequence_features(
            ds[
                split
            ][
                "X"
            ]
        )
        for split in [
            "train",
            "validation",
            "test",
        ]
    }

    context = {
        split: build_future_context(
            stage2[
                split
            ],
            vessel_pack[
                split
            ][
                "ridge"
            ],
        )
        for split in [
            "train",
            "validation",
            "test",
        ]
    }

    log(
        f"  TRAIN sequence={seq['train'].shape}"
    )

    log(
        f"  TRAIN future context={context['train'].shape}"
    )

    # -----------------------------------------------------------------
    # Train.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 4/8] Training 2-layer Vessel/HDG residual GRU."
    )

    bundle = train_stage15c(
        torch=torch,
        nn=nn,
        train_seq=seq[
            "train"
        ],
        train_context=context[
            "train"
        ],
        train_residual=vessel_pack[
            "train"
        ][
            "residual"
        ],
        train_ridge=vessel_pack[
            "train"
        ][
            "ridge"
        ],
        train_stage2=stage2[
            "train"
        ],
        train_y=vessel_pack[
            "train"
        ][
            "y"
        ],
        train_app_ref=app_ref[
            "train"
        ],
        val_seq=seq[
            "validation"
        ],
        val_context=context[
            "validation"
        ],
        val_ridge=vessel_pack[
            "validation"
        ][
            "ridge"
        ],
        val_stage2=stage2[
            "validation"
        ],
        val_y=vessel_pack[
            "validation"
        ][
            "y"
        ],
        val_app_ref=app_ref[
            "validation"
        ],
    )

    bundle[
        "history"
    ].to_csv(
        out
        / "15C_training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log(
        f"[FROZEN] best epoch = {bundle['best_epoch']}"
    )

    log(
        f"[FROZEN] validation score = {bundle['best_score']:.6f}"
    )

    alpha_motion = bundle[
        "alpha_motion"
    ]

    alpha_heading = bundle[
        "alpha_heading"
    ]

    log("")
    log(
        "[FROZEN ALPHA]"
    )

    for j, h in enumerate(
        HORIZONS
    ):
        log(
            f"  {h:2d} min | "
            f"alpha_E={alpha_motion[j,0]:.6f} | "
            f"alpha_N={alpha_motion[j,1]:.6f} | "
            f"alpha_HDG={alpha_heading[j]:.6f}"
        )

    # -----------------------------------------------------------------
    # Frozen inference.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 5/8] Frozen validation/test inference."
    )

    transformed = {}

    for split in [
        "validation",
        "test",
    ]:
        transformed[
            split
        ] = transform_inputs(
            seq[
                split
            ],
            context[
                split
            ],
            bundle[
                "scaler"
            ],
        )

    raw_val = predict_raw(
        torch,
        bundle[
            "model"
        ],
        transformed[
            "validation"
        ][
            0
        ],
        transformed[
            "validation"
        ][
            1
        ],
        bundle[
            "scaler"
        ][
            "residual_std"
        ],
        bundle[
            "device"
        ],
    )

    raw_test = predict_raw(
        torch,
        bundle[
            "model"
        ],
        transformed[
            "test"
        ][
            0
        ],
        transformed[
            "test"
        ][
            1
        ],
        bundle[
            "scaler"
        ][
            "residual_std"
        ],
        bundle[
            "device"
        ],
    )

    final_val = apply_alphas_np(
        vessel_pack[
            "validation"
        ][
            "ridge"
        ],
        raw_val,
        alpha_motion,
        alpha_heading,
    )

    final_test = apply_alphas_np(
        vessel_pack[
            "test"
        ][
            "ridge"
        ],
        raw_test,
        alpha_motion,
        alpha_heading,
    )

    # -----------------------------------------------------------------
    # Metrics.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 6/8] Vessel/HDG + apparent-wind metrics."
    )

    metrics = pd.concat(
        [
            metrics_per_horizon(
                vessel_pack[
                    "validation"
                ][
                    "y"
                ],
                vessel_pack[
                    "validation"
                ][
                    "ridge"
                ],
                stage2[
                    "validation"
                ],
                app_ref[
                    "validation"
                ],
                "validation",
                "Stage15BWind+VesselRidge",
            ),
            metrics_per_horizon(
                vessel_pack[
                    "validation"
                ][
                    "y"
                ],
                final_val,
                stage2[
                    "validation"
                ],
                app_ref[
                    "validation"
                ],
                "validation",
                "Stage15C-Final",
            ),
            metrics_per_horizon(
                vessel_pack[
                    "test"
                ][
                    "y"
                ],
                vessel_pack[
                    "test"
                ][
                    "ridge"
                ],
                stage2[
                    "test"
                ],
                app_ref[
                    "test"
                ],
                "test",
                "Stage15BWind+VesselRidge",
            ),
            metrics_per_horizon(
                vessel_pack[
                    "test"
                ][
                    "y"
                ],
                final_test,
                stage2[
                    "test"
                ],
                app_ref[
                    "test"
                ],
                "test",
                "Stage15C-Final",
            ),
        ],
        ignore_index=True,
    )

    metrics.to_csv(
        out
        / "15C_metrics_per_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Save.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 7/8] Saving final predictions/model."
    )

    np.savez_compressed(
        out
        / "15C_validation_predictions.npz",
        y_vessel_hdg=vessel_pack[
            "validation"
        ][
            "y"
        ].astype(
            np.float32
        ),
        vessel_hdg_ridge=vessel_pack[
            "validation"
        ][
            "ridge"
        ].astype(
            np.float32
        ),
        raw_residual=raw_val.astype(
            np.float32
        ),
        vessel_hdg_final=final_val.astype(
            np.float32
        ),
        stage2_truewind=stage2[
            "validation"
        ].astype(
            np.float32
        ),
        apparent_ref_earth=app_ref[
            "validation"
        ].astype(
            np.float32
        ),
        alpha_motion=alpha_motion.astype(
            np.float32
        ),
        alpha_heading=alpha_heading.astype(
            np.float32
        ),
        horizons_min=np.asarray(
            HORIZONS,
            dtype=np.int64,
        ),
    )

    np.savez_compressed(
        out
        / "15C_test_predictions.npz",
        y_vessel_hdg=vessel_pack[
            "test"
        ][
            "y"
        ].astype(
            np.float32
        ),
        vessel_hdg_ridge=vessel_pack[
            "test"
        ][
            "ridge"
        ].astype(
            np.float32
        ),
        raw_residual=raw_test.astype(
            np.float32
        ),
        vessel_hdg_final=final_test.astype(
            np.float32
        ),
        stage2_truewind=stage2[
            "test"
        ].astype(
            np.float32
        ),
        apparent_ref_earth=app_ref[
            "test"
        ].astype(
            np.float32
        ),
        alpha_motion=alpha_motion.astype(
            np.float32
        ),
        alpha_heading=alpha_heading.astype(
            np.float32
        ),
        horizons_min=np.asarray(
            HORIZONS,
            dtype=np.int64,
        ),
    )

    checkpoint_path = (
        out
        / "15C_vessel_hdg_residual.pt"
    )

    torch.save(
        {
            "state_dict": state_dict_cpu(
                bundle[
                    "model"
                ]
            ),
            "scaler": bundle[
                "scaler"
            ],
            "alpha_motion": alpha_motion,
            "alpha_heading": alpha_heading,
            "best_epoch": bundle[
                "best_epoch"
            ],
            "best_validation_score": bundle[
                "best_score"
            ],
            "seed": SEED,
            "horizons_min": HORIZONS,
            "network": {
                "gru_layers": GRU_LAYERS,
                "gru_hidden": GRU_HIDDEN,
                "shared_hidden": SHARED_HIDDEN,
                "dropout": DROPOUT,
            },
            "loss_weights": {
                "app_vector": LAMBDA_APP_VECTOR,
                "hdg_circle": LAMBDA_HDG_CIRCLE,
                "awa_circle": LAMBDA_AWA_CIRCLE,
            },
            "apparent_audit": apparent_audit,
        },
        checkpoint_path,
    )

    report = {
        "stage": "15C",
        "architecture": {
            "GRU_layers": GRU_LAYERS,
            "GRU_hidden": GRU_HIDDEN,
            "shared_hidden": SHARED_HIDDEN,
            "dropout": DROPOUT,
            "trainable_parameters": count_parameters(
                bundle[
                    "model"
                ]
            ),
        },
        "upstream": {
            "true_wind": (
                "Stage15B-F final true wind"
            ),
            "vessel_hdg_anchor": (
                "Stage15A-F Ridge alpha=2250"
            ),
            "train_upstream_predictions": (
                "OOF/cross-fitted"
            ),
        },
        "best_epoch": bundle[
            "best_epoch"
        ],
        "best_validation_score": bundle[
            "best_score"
        ],
        "best_validation_score_parts": bundle[
            "best_parts"
        ],
        "alpha_motion": alpha_motion,
        "alpha_heading": alpha_heading,
        "alpha_cap": ALPHA_CAP,
        "apparent_reference_audit": apparent_audit,
        "test_used_for_training": False,
        "test_used_for_epoch_selection": False,
        "test_used_for_alpha": False,
        "metrics": metrics.to_dict(
            orient="records"
        ),
    }

    save_json(
        out
        / "15C_REPORT.json",
        report,
    )

    # -----------------------------------------------------------------
    # Final report.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 8/8] Final report."
    )

    test_table = metrics.loc[
        metrics[
            "split"
        ]
        == "test"
    ].copy()

    with (
        out
        / "15C_REPORT.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "15C — VESSEL/HDG RESIDUAL + APPARENT-WIND COUPLING\n"
        )

        f.write(
            "="
            * 140
            + "\n\n"
        )

        f.write(
            f"Best epoch: {bundle['best_epoch']}\n"
        )

        f.write(
            f"Trainable params: {count_parameters(bundle['model'])}\n"
        )

        f.write(
            f"Apparent reference source: {apparent_audit['source']}\n"
        )

        f.write(
            f"Apparent convention: {apparent_audit['convention']}\n\n"
        )

        f.write(
            "FROZEN ALPHA\n"
        )

        f.write(
            "-"
            * 140
            + "\n"
        )

        for j, h in enumerate(
            HORIZONS
        ):
            f.write(
                f"{h:2d} min | "
                f"E={alpha_motion[j,0]:.8f} | "
                f"N={alpha_motion[j,1]:.8f} | "
                f"HDG={alpha_heading[j]:.8f}\n"
            )

        f.write(
            "\nTEST\n"
        )

        f.write(
            "-"
            * 140
            + "\n"
        )

        f.write(
            test_table.to_string(
                index=False
            )
        )

    log("")
    log("=" * 150)
    log(
        "15C FINAL TEST — VESSEL/HDG + APPARENT WIND"
    )
    log("=" * 150)

    log(
        test_table.to_string(
            index=False
        )
    )

    log("")
    log(
        f"[PARAMS] Stage15C residual NN = "
        f"{count_parameters(bundle['model']):,}"
    )

    log("")
    log(
        "[INTERPRETATION]"
    )

    log(
        "  Apparent wind uses Stage15B-F final true wind."
    )

    log(
        "  It does NOT revert to the original TrueWind Ridge."
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
