# -*- coding: utf-8 -*-
"""
15E_hybrid_NewWind_OldCompactVessel_5seed_validation.py

15E — Hybrid confirmation:
    NEW Stage15B true-wind branch
    +
    OLD manuscript Physics-Compact vessel-only gated residual branch

Purpose
-------
The previous 15C-S single-GRU experiment was NOT the original manuscript
vessel model. It only changed the NEW Stage15C GRU depth from 2 to 1.

This script reconstructs the documented OLD primary vessel branch:

    full joint Ridge anchor:
        60 x 11 -> 5 x 6
        alpha = 100

    old compact vessel residual network:
        single-layer GRU, hidden=64
        full 30-value Ridge forecast as decoder context
        Dense(64), ReLU, Dropout(0.1)
        vessel residual head: 10 outputs
        sigmoid gate head:   10 outputs
        gate bias initialized to -1
        residual head zero initialized

    final OLD vessel:
        Vship = Vship_Ridge + gate * deltaV

    OLD wind and heading:
        remain frozen to the joint Ridge anchor

    documented old loss:
        L = L_direct_vessel
            + 2.0 * L_apparent_earth
            + 0.01 * L_residual_regularization

    optimizer:
        Adam, lr=1e-3

    batch:
        256

    max epochs:
        80

    early stopping:
        validation apparent-wind vector RMSE

The network architecture above has exactly 22,164 trainable parameters,
matching the manuscript.

15E hybrid
----------
The old vessel prediction and old Ridge heading are kept unchanged.

Only the wind source changes:

    OLD reconstructed:
        A_old = W_joint_Ridge - V_old_compact

    15E hybrid:
        A_15E = W_Stage15B - V_old_compact

Therefore OLD -> 15E isolates the value of the NEW true-wind branch.

Five seeds
----------
500043, 501052, 502061, 503070, 504079

IMPORTANT
---------
TRAIN/VALIDATION only.
The internal TEST split is NOT loaded.

Recommended PowerShell
----------------------
python "D:\\project\\WindPredict_SaildroneData\\src\\15E_hybrid_NewWind_OldCompactVessel_5seed_validation.py" --dataset-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\SD1090_TPOS2024_JointForecasting_v0_1" --stage15a-truewind-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15A_R_1min_Dual_Ridge_Anchors_v0_2" --src-dir "D:\\project\\WindPredict_SaildroneData\\src" --output-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15E_Hybrid_NewWind_OldCompactVessel_5seed_v0_1"
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
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

DEFAULT_STAGE15A_TRUEWIND_DIR = (
    ROOT
    / "data"
    / "forecasting"
    / "15A_R_1min_Dual_Ridge_Anchors_v0_2"
)

DEFAULT_SRC_DIR = (
    ROOT
    / "src"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "data"
    / "forecasting"
    / "15E_Hybrid_NewWind_OldCompactVessel_5seed_v0_1"
)

SEEDS = [
    500043,
    501052,
    502061,
    503070,
    504079,
]

HORIZONS = [
    1,
    2,
    3,
    5,
    10,
]

# ---------------------------------------------------------------------
# Frozen OLD manuscript primary-model settings.
# ---------------------------------------------------------------------
OLD_RIDGE_ALPHA = 100.0

OLD_GRU_HIDDEN = 64
OLD_GRU_LAYERS = 1
OLD_DENSE_HIDDEN = 64
OLD_DROPOUT = 0.10

OLD_LR = 1e-3
OLD_BATCH_SIZE = 256
OLD_MAX_EPOCHS = 80
OLD_PATIENCE = 12
OLD_GRAD_CLIP = 1.0

OLD_LAMBDA_APP = 2.0
OLD_LAMBDA_RES = 0.01

# Joint target:
# [U, V, VesselEast, VesselNorth, HDG_sin, HDG_cos]
WIND_IDX = [
    0,
    1,
]

VESSEL_IDX = [
    2,
    3,
]

HDG_IDX = [
    4,
    5,
]

EPS = 1e-12
STATS_CHUNK = 8192

OLD_MANUSCRIPT_TABLE6 = {
    "AppVector_RMSE_mps": 1.0785,
    "AppVector_RMSE_std_mps": 0.0032,
    "Vessel_vector_RMSE_mps": 0.4853,
    "Vessel_vector_RMSE_std_mps": 0.0078,
    "AWS_RMSE_mps": 0.8032,
    "AWA_MAE_deg": 8.64,
    "NN_parameters": 22164,
}

STAGE15B_NN_PARAMETERS = 26074
OLD_VESSEL_NN_PARAMETERS = 22164

HYBRID_NEURAL_PARAMETERS = (
    STAGE15B_NN_PARAMETERS
    + OLD_VESSEL_NN_PARAMETERS
)

TRUEWIND_RIDGE_COEFFICIENTS = 2410

OLD_JOINT_RIDGE_COEFFICIENTS = (
    660
    * 30
    + 30
)

HYBRID_TOTAL_FITTED_COEFFICIENTS = (
    TRUEWIND_RIDGE_COEFFICIENTS
    + OLD_JOINT_RIDGE_COEFFICIENTS
    + HYBRID_NEURAL_PARAMETERS
)


# =============================================================================
# Utilities
# =============================================================================

def log(msg=""):
    print(
        msg,
        flush=True,
    )


def save_json(
    path: Path,
    obj,
):
    def cv(v):
        if isinstance(
            v,
            Path,
        ):
            return str(
                v
            )

        if isinstance(
            v,
            np.ndarray,
        ):
            return v.tolist()

        if isinstance(
            v,
            np.integer,
        ):
            return int(
                v
            )

        if isinstance(
            v,
            np.floating,
        ):
            return (
                None
                if not np.isfinite(
                    v
                )
                else float(
                    v
                )
            )

        if isinstance(
            v,
            np.bool_,
        ):
            return bool(
                v
            )

        if isinstance(
            v,
            dict,
        ):
            return {
                str(
                    k
                ): cv(
                    x
                )
                for k, x in v.items()
            }

        if isinstance(
            v,
            (
                list,
                tuple,
            ),
        ):
            return [
                cv(
                    x
                )
                for x in v
            ]

        return v

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            cv(
                obj
            ),
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def load_module(
    name: str,
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            name,
            str(
                path
            ),
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Cannot import {path}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules[
        name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def import_torch():
    import torch
    import torch.nn as nn

    return (
        torch,
        nn,
    )


def seed_everything(
    torch,
    seed,
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def configure_torch(
    torch,
):
    if torch.cuda.is_available():
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


def amp_context(
    torch,
    device,
):
    if device.type != "cuda":
        return contextlib.nullcontext()

    try:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        )
    except Exception:
        return (
            torch.cuda.amp
            .autocast(
                enabled=True
            )
        )


def create_grad_scaler(
    torch,
    device,
):
    if device.type != "cuda":
        return None

    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )
    except Exception:
        try:
            return (
                torch.cuda.amp
                .GradScaler(
                    enabled=True
                )
            )
        except Exception:
            return None


def state_dict_cpu(
    model,
):
    return {
        k: (
            v.detach()
            .cpu()
            .clone()
        )
        for k, v in (
            model
            .state_dict()
            .items()
        )
    }


def count_parameters(
    model,
):
    return int(
        sum(
            p.numel()
            for p in (
                model
                .parameters()
            )
            if p.requires_grad
        )
    )


# =============================================================================
# Dataset
# =============================================================================

def load_split(
    dataset_dir: Path,
    split: str,
):
    path = (
        dataset_dir
        / f"{split}.npz"
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
                "X"
            ],
            dtype=np.float32,
        )

        y = np.asarray(
            z[
                "y_raw"
            ],
            dtype=np.float32,
        )

        apparent = None
        apparent_key = None

        for key in z.files:
            if (
                "apparent"
                in key.lower()
            ):
                arr = np.asarray(
                    z[
                        key
                    ],
                    dtype=np.float32,
                )

                if (
                    arr.ndim == 3
                    and arr.shape[
                        1:
                    ] == (
                        5,
                        2,
                    )
                    and len(
                        arr
                    ) == len(
                        X
                    )
                ):
                    apparent = arr
                    apparent_key = key
                    break

    if (
        X.ndim != 3
        or X.shape[
            1:
        ] != (
            60,
            11,
        )
    ):
        raise RuntimeError(
            f"{split}: invalid X {X.shape}"
        )

    if (
        y.ndim != 3
        or y.shape[
            1:
        ] != (
            5,
            6,
        )
    ):
        raise RuntimeError(
            f"{split}: invalid y {y.shape}"
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
            f"{split}: nonfinite data."
        )

    return {
        "X": X,
        "y": y,
        "apparent": apparent,
        "apparent_key": apparent_key,
        "path": path,
    }


def load_truewind_anchor(
    stage15a_truewind_dir: Path,
):
    train_path = (
        stage15a_truewind_dir
        / "15A_train_oof_anchors.npz"
    )

    val_path = (
        stage15a_truewind_dir
        / "15A_validation_anchor_predictions.npz"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            train_path
        )

    if not val_path.exists():
        raise FileNotFoundError(
            val_path
        )

    with np.load(
        train_path,
        allow_pickle=False,
    ) as z:
        train = {
            "y": np.asarray(
                z[
                    "true_wind_y"
                ],
                dtype=np.float32,
            ),
            "ridge": np.asarray(
                z[
                    "true_wind_ridge_oof"
                ],
                dtype=np.float32,
            ),
            "residual": np.asarray(
                z[
                    "true_wind_residual_oof"
                ],
                dtype=np.float32,
            ),
        }

    with np.load(
        val_path,
        allow_pickle=False,
    ) as z:
        val = {
            "y": np.asarray(
                z[
                    "true_wind_y"
                ],
                dtype=np.float32,
            ),
            "ridge": np.asarray(
                z[
                    "true_wind_ridge"
                ],
                dtype=np.float32,
            ),
            "residual": np.asarray(
                z[
                    "true_wind_residual"
                ],
                dtype=np.float32,
            ),
        }

    return (
        train,
        val,
    )


# =============================================================================
# Old joint Ridge
# =============================================================================

def empty_stats(
    p,
    q,
):
    return {
        "n": 0,
        "sum_x": np.zeros(
            p,
            dtype=np.float64,
        ),
        "sum_x2": np.zeros(
            (
                p,
                p,
            ),
            dtype=np.float64,
        ),
        "sum_y": np.zeros(
            q,
            dtype=np.float64,
        ),
        "sum_y2": np.zeros(
            q,
            dtype=np.float64,
        ),
        "sum_xy": np.zeros(
            (
                p,
                q,
            ),
            dtype=np.float64,
        ),
    }


def compute_stats(
    X,
    y,
):
    p = X.shape[
        1
    ]

    q = y.shape[
        1
    ]

    out = empty_stats(
        p,
        q,
    )

    for a in range(
        0,
        len(
            X
        ),
        STATS_CHUNK,
    ):
        b = min(
            a
            + STATS_CHUNK,
            len(
                X
            ),
        )

        xb = np.asarray(
            X[
                a:
                b
            ],
            dtype=np.float64,
        )

        yb = np.asarray(
            y[
                a:
                b
            ],
            dtype=np.float64,
        )

        out[
            "n"
        ] += len(
            xb
        )

        out[
            "sum_x"
        ] += xb.sum(
            axis=0
        )

        out[
            "sum_x2"
        ] += (
            xb.T
            @ xb
        )

        out[
            "sum_y"
        ] += yb.sum(
            axis=0
        )

        out[
            "sum_y2"
        ] += np.sum(
            yb
            * yb,
            axis=0,
        )

        out[
            "sum_xy"
        ] += (
            xb.T
            @ yb
        )

    return out


def fit_old_joint_ridge(
    X_train,
    y_train_raw,
):
    """
    Output-specific target standardization for the flattened 30 outputs.
    Ridge is fitted in standardized target space, with alpha=100.
    """
    Xf = np.asarray(
        X_train,
        dtype=np.float32,
    ).reshape(
        len(
            X_train
        ),
        -1,
    )

    yf = np.asarray(
        y_train_raw,
        dtype=np.float32,
    ).reshape(
        len(
            y_train_raw
        ),
        -1,
    )

    y_mean = np.mean(
        yf.astype(
            np.float64
        ),
        axis=0,
    )

    y_std = np.std(
        yf.astype(
            np.float64
        ),
        axis=0,
        ddof=0,
    )

    y_std = np.where(
        y_std
        < 1e-8,
        1.0,
        y_std,
    )

    yz = (
        (
            yf
            - y_mean[
                None,
                :
            ]
        )
        / y_std[
            None,
            :
        ]
    ).astype(
        np.float32
    )

    stats = compute_stats(
        Xf,
        yz,
    )

    n = float(
        stats[
            "n"
        ]
    )

    x_mean = (
        stats[
            "sum_x"
        ]
        / n
    )

    y_mean_z = (
        stats[
            "sum_y"
        ]
        / n
    )

    G = (
        stats[
            "sum_x2"
        ]
        - np.outer(
            stats[
                "sum_x"
            ],
            stats[
                "sum_x"
            ],
        )
        / n
    )

    G = (
        0.5
        * (
            G
            + G.T
        )
    )

    C = (
        stats[
            "sum_xy"
        ]
        - np.outer(
            stats[
                "sum_x"
            ],
            y_mean_z,
        )
    )

    A = (
        G
        + OLD_RIDGE_ALPHA
        * np.eye(
            G.shape[
                0
            ],
            dtype=np.float64,
        )
    )

    B = np.linalg.solve(
        A,
        C,
    )

    intercept = (
        y_mean_z
        - x_mean
        @ B
    )

    return {
        "coef_z": B,
        "intercept_z": intercept,
        "y_mean_raw": y_mean,
        "y_std_raw": y_std,
    }


def old_ridge_predict(
    model,
    X,
):
    Xf = np.asarray(
        X,
        dtype=np.float64,
    ).reshape(
        len(
            X
        ),
        -1,
    )

    z = (
        Xf
        @ model[
            "coef_z"
        ]
        + model[
            "intercept_z"
        ][
            None,
            :
        ]
    )

    raw = (
        z
        * model[
            "y_std_raw"
        ][
            None,
            :
        ]
        + model[
            "y_mean_raw"
        ][
            None,
            :
        ]
    )

    return (
        z.reshape(
            -1,
            5,
            6,
        ).astype(
            np.float32
        ),
        raw.reshape(
            -1,
            5,
            6,
        ).astype(
            np.float32
        ),
    )


def target_to_old_z(
    old_ridge_model,
    y_raw,
):
    yf = np.asarray(
        y_raw,
        dtype=np.float64,
    ).reshape(
        len(
            y_raw
        ),
        -1,
    )

    z = (
        (
            yf
            - old_ridge_model[
                "y_mean_raw"
            ][
                None,
                :
            ]
        )
        / old_ridge_model[
            "y_std_raw"
        ][
            None,
            :
        ]
    )

    return z.reshape(
        -1,
        5,
        6,
    ).astype(
        np.float32
    )


# =============================================================================
# Old compact vessel model
# =============================================================================

def build_old_model_class(
    torch,
    nn,
):
    class OldCompactVesselResidual(
        nn.Module
    ):
        def __init__(
            self,
        ):
            super().__init__()

            self.gru = nn.GRU(
                input_size=11,
                hidden_size=OLD_GRU_HIDDEN,
                num_layers=1,
                batch_first=True,
            )

            self.history_dropout = nn.Dropout(
                OLD_DROPOUT
            )

            self.fusion = nn.Sequential(
                nn.Linear(
                    OLD_GRU_HIDDEN
                    + 30,
                    OLD_DENSE_HIDDEN,
                ),
                nn.ReLU(),
                nn.Dropout(
                    OLD_DROPOUT
                ),
            )

            self.residual_head = nn.Linear(
                OLD_DENSE_HIDDEN,
                10,
            )

            self.gate_head = nn.Linear(
                OLD_DENSE_HIDDEN,
                10,
            )

            # Old manuscript initialization:
            # near-zero residual + gate bias -1.
            nn.init.zeros_(
                self.residual_head.weight
            )

            nn.init.zeros_(
                self.residual_head.bias
            )

            nn.init.zeros_(
                self.gate_head.weight
            )

            nn.init.constant_(
                self.gate_head.bias,
                -1.0,
            )

        def forward(
            self,
            seq,
            ridge_z_flat,
        ):
            out, _ = self.gru(
                seq
            )

            h = out[
                :,
                -1,
                :
            ]

            h = self.history_dropout(
                h
            )

            z = self.fusion(
                torch.cat(
                    [
                        h,
                        ridge_z_flat,
                    ],
                    dim=1,
                )
            )

            residual = (
                self.residual_head(
                    z
                )
                .reshape(
                    -1,
                    5,
                    2,
                )
            )

            gate = (
                torch.sigmoid(
                    self.gate_head(
                        z
                    )
                )
                .reshape(
                    -1,
                    5,
                    2,
                )
            )

            correction = (
                gate
                * residual
            )

            return (
                residual,
                gate,
                correction,
            )

    return OldCompactVesselResidual


def vessel_scaler_arrays(
    old_ridge_model,
):
    ym = (
        old_ridge_model[
            "y_mean_raw"
        ]
        .reshape(
            5,
            6,
        )
    )

    ys = (
        old_ridge_model[
            "y_std_raw"
        ]
        .reshape(
            5,
            6,
        )
    )

    return (
        ym[
            :,
            2:
            4
        ].astype(
            np.float32
        ),
        ys[
            :,
            2:
            4
        ].astype(
            np.float32
        ),
    )


def z_vessel_to_raw_torch(
    torch,
    vessel_z,
    vessel_mean_t,
    vessel_std_t,
):
    return (
        vessel_z
        * vessel_std_t[
            None,
            :,
            :
        ]
        + vessel_mean_t[
            None,
            :,
            :
        ]
    )


def predict_old_vessel(
    *,
    torch,
    model,
    X,
    ridge_z,
    ridge_raw,
    old_ridge_model,
    device,
):
    model.eval()

    mean_v, std_v = (
        vessel_scaler_arrays(
            old_ridge_model
        )
    )

    preds = []
    corr_all = []
    gate_all = []

    with torch.no_grad():
        for a in range(
            0,
            len(
                X
            ),
            OLD_BATCH_SIZE,
        ):
            b = min(
                a
                + OLD_BATCH_SIZE,
                len(
                    X
                ),
            )

            seq_b = (
                torch.from_numpy(
                    X[
                        a:
                        b
                    ]
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            rz_b = (
                torch.from_numpy(
                    ridge_z[
                        a:
                        b
                    ]
                    .reshape(
                        b
                        - a,
                        -1,
                    )
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            with amp_context(
                torch,
                device,
            ):
                _, gate, corr = model(
                    seq_b,
                    rz_b,
                )

            corr_np = (
                corr.detach()
                .float()
                .cpu()
                .numpy()
            )

            gate_np = (
                gate.detach()
                .float()
                .cpu()
                .numpy()
            )

            vessel_z = (
                ridge_z[
                    a:
                    b,
                    :,
                    2:
                    4
                ]
                + corr_np
            )

            vessel_raw = (
                vessel_z
                * std_v[
                    None,
                    :,
                    :
                ]
                + mean_v[
                    None,
                    :,
                    :
                ]
            )

            pred = (
                ridge_raw[
                    a:
                    b
                ]
                .copy()
            )

            pred[
                :,
                :,
                2:
                4
            ] = vessel_raw

            preds.append(
                pred.astype(
                    np.float32
                )
            )

            corr_all.append(
                corr_np.astype(
                    np.float32
                )
            )

            gate_all.append(
                gate_np.astype(
                    np.float32
                )
            )

    return (
        np.concatenate(
            preds,
            axis=0,
        ),
        np.concatenate(
            corr_all,
            axis=0,
        ),
        np.concatenate(
            gate_all,
            axis=0,
        ),
    )


def apparent_vector_rmse_mean(
    cmod,
    y_true,
    pred_vh,
    wind_pred,
    app_ref,
):
    df = cmod.metrics_per_horizon(
        y_true[
            :,
            :,
            2:
            6
        ],
        pred_vh[
            :,
            :,
            2:
            6
        ],
        wind_pred,
        app_ref,
        "validation",
        "tmp",
    )

    return float(
        df[
            "AppVector_RMSE_mps"
        ].mean()
    )


def train_old_vessel_model(
    *,
    torch,
    nn,
    cmod,
    seed,
    X_train,
    y_train_raw,
    y_train_z,
    ridge_z_train,
    ridge_raw_train,
    app_ref_train,
    X_val,
    y_val_raw,
    ridge_z_val,
    ridge_raw_val,
    app_ref_val,
    old_ridge_model,
):
    seed_everything(
        torch,
        seed,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    Model = build_old_model_class(
        torch,
        nn,
    )

    model = Model().to(
        device
    )

    param_count = count_parameters(
        model
    )

    if (
        param_count
        != OLD_VESSEL_NN_PARAMETERS
    ):
        raise RuntimeError(
            f"Old model parameter mismatch: "
            f"{param_count} != {OLD_VESSEL_NN_PARAMETERS}"
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=OLD_LR,
    )

    scaler = create_grad_scaler(
        torch,
        device,
    )

    vessel_mean, vessel_std = (
        vessel_scaler_arrays(
            old_ridge_model
        )
    )

    vessel_mean_t = (
        torch.from_numpy(
            vessel_mean
        )
        .to(
            device
        )
    )

    vessel_std_t = (
        torch.from_numpy(
            vessel_std
        )
        .to(
            device
        )
    )

    # Training-only apparent-wind normalization.
    app_std = np.std(
        np.asarray(
            app_ref_train,
            dtype=np.float64,
        ).reshape(
            -1,
            2,
        ),
        axis=0,
        ddof=0,
    )

    app_std = np.where(
        app_std
        < 1e-6,
        1.0,
        app_std,
    ).astype(
        np.float32
    )

    app_std_t = (
        torch.from_numpy(
            app_std
        )
        .to(
            device
        )
    )

    best_state = state_dict_cpu(
        model
    )

    # Epoch 0 = Ridge because residual head is zero.
    best_score = (
        apparent_vector_rmse_mean(
            cmod,
            y_val_raw,
            ridge_raw_val,
            ridge_raw_val[
                :,
                :,
                0:
                2
            ],
            app_ref_val,
        )
    )

    best_epoch = 0
    patience = 0
    history = []

    n = len(
        X_train
    )

    for epoch in range(
        1,
        OLD_MAX_EPOCHS
        + 1,
    ):
        model.train()

        rng = np.random.default_rng(
            seed
            + epoch
            * 1009
        )

        order = np.arange(
            n
        )

        rng.shuffle(
            order
        )

        total_loss = 0.0
        total_direct = 0.0
        total_app = 0.0
        total_res = 0.0
        seen = 0

        for a in range(
            0,
            n,
            OLD_BATCH_SIZE,
        ):
            idx = order[
                a:
                a
                + OLD_BATCH_SIZE
            ]

            seq_b = (
                torch.from_numpy(
                    X_train[
                        idx
                    ]
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            ridge_z_b = (
                torch.from_numpy(
                    ridge_z_train[
                        idx
                    ]
                    .reshape(
                        len(
                            idx
                        ),
                        -1,
                    )
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            ridge_v_z_b = (
                torch.from_numpy(
                    ridge_z_train[
                        idx,
                        :,
                        2:
                        4
                    ]
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            y_v_z_b = (
                torch.from_numpy(
                    y_train_z[
                        idx,
                        :,
                        2:
                        4
                    ]
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            ridge_w_raw_b = (
                torch.from_numpy(
                    ridge_raw_train[
                        idx,
                        :,
                        0:
                        2
                    ]
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            app_ref_b = (
                torch.from_numpy(
                    app_ref_train[
                        idx
                    ]
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with amp_context(
                torch,
                device,
            ):
                _, gate, correction = model(
                    seq_b,
                    ridge_z_b,
                )

                vessel_pred_z = (
                    ridge_v_z_b
                    + correction
                )

                loss_direct = torch.mean(
                    (
                        vessel_pred_z
                        - y_v_z_b
                    )
                    ** 2
                )

                vessel_pred_raw = (
                    z_vessel_to_raw_torch(
                        torch,
                        vessel_pred_z,
                        vessel_mean_t,
                        vessel_std_t,
                    )
                )

                app_pred = (
                    ridge_w_raw_b
                    - vessel_pred_raw
                )

                app_err_norm = (
                    (
                        app_pred
                        - app_ref_b
                    )
                    / app_std_t[
                        None,
                        None,
                        :
                    ]
                )

                loss_app = torch.mean(
                    torch.sum(
                        app_err_norm
                        * app_err_norm,
                        dim=-1,
                    )
                )

                loss_res = torch.mean(
                    correction
                    * correction
                )

                loss = (
                    loss_direct
                    + OLD_LAMBDA_APP
                    * loss_app
                    + OLD_LAMBDA_RES
                    * loss_res
                )

            if scaler is not None:
                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    OLD_GRAD_CLIP,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    OLD_GRAD_CLIP,
                )

                optimizer.step()

            bn = len(
                idx
            )

            total_loss += float(
                loss.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            total_direct += float(
                loss_direct.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            total_app += float(
                loss_app.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            total_res += float(
                loss_res.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            seen += bn

        pred_val, _, _ = (
            predict_old_vessel(
                torch=torch,
                model=model,
                X=X_val,
                ridge_z=ridge_z_val,
                ridge_raw=ridge_raw_val,
                old_ridge_model=old_ridge_model,
                device=device,
            )
        )

        score = (
            apparent_vector_rmse_mean(
                cmod,
                y_val_raw,
                pred_val,
                ridge_raw_val[
                    :,
                    :,
                    0:
                    2
                ],
                app_ref_val,
            )
        )

        improved = (
            score
            < best_score
            - 1e-7
        )

        if improved:
            best_score = score
            best_epoch = epoch
            best_state = state_dict_cpu(
                model
            )
            patience = 0

        else:
            patience += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": (
                    total_loss
                    / seen
                ),
                "train_direct_loss": (
                    total_direct
                    / seen
                ),
                "train_app_loss": (
                    total_app
                    / seen
                ),
                "train_residual_reg": (
                    total_res
                    / seen
                ),
                "validation_old_aw_vector_rmse": score,
                "best_validation_old_aw_vector_rmse": best_score,
            }
        )

        if (
            epoch <= 3
            or epoch
            % 10 == 0
            or improved
        ):
            log(
                f"      old-vessel epoch={epoch:03d} | "
                f"loss={total_loss/seen:.6f} | "
                f"old-AW={score:.6f}"
                + (
                    " *"
                    if improved
                    else ""
                )
            )

        if (
            patience
            >= OLD_PATIENCE
        ):
            break

    model.load_state_dict(
        best_state,
        strict=True,
    )

    return {
        "model": model,
        "device": device,
        "best_epoch": best_epoch,
        "best_old_aw_rmse": best_score,
        "history": pd.DataFrame(
            history
        ),
        "app_std": app_std,
    }


# =============================================================================
# Metrics
# =============================================================================

def mean_metrics(
    df,
):
    cols = [
        "Vessel_vector_RMSE_mps",
        "SOG_RMSE_mps",
        "HDG_MAE_deg",
        "AppVector_RMSE_mps",
        "AWS_RMSE_mps",
        "AWA_MAE_deg",
    ]

    return {
        col: float(
            df[
                col
            ].mean()
        )
        for col in cols
    }


def aggregate(
    df,
):
    rows = []

    numeric_cols = [
        c
        for c in df.columns
        if (
            c
            not in {
                "seed",
                "model",
            }
            and np.issubdtype(
                df[
                    c
                ].dtype,
                np.number,
            )
        )
    ]

    for col in numeric_cols:
        vals = df[
            col
        ].to_numpy(
            dtype=np.float64
        )

        rows.append(
            {
                "metric": col,
                "mean": float(
                    np.mean(
                        vals
                    )
                ),
                "std": float(
                    np.std(
                        vals,
                        ddof=1,
                    )
                ),
                "min": float(
                    np.min(
                        vals
                    )
                ),
                "max": float(
                    np.max(
                        vals
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


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
        "--stage15a-truewind-dir",
        type=Path,
        default=DEFAULT_STAGE15A_TRUEWIND_DIR,
    )

    parser.add_argument(
        "--src-dir",
        type=Path,
        default=DEFAULT_SRC_DIR,
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

    bmod = load_module(
        "stage15b_15e",
        args.src_dir
        / "15B_train_1min_TrueWind_Residual_Correction_v2.py",
    )

    cmod = load_module(
        "stage15c_geometry_15e",
        args.src_dir
        / "15C_train_1min_VesselHDG_Residual_ApparentWind.py",
    )

    # Final frozen Stage15B alpha design.
    bmod.ALPHA_MAX = 1.0

    torch, nn = import_torch()

    configure_torch(
        torch
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    log("=" * 154)
    log(
        "15E — NEW TRUE-WIND + OLD COMPACT VESSEL HYBRID, FIVE-SEED VALIDATION"
    )
    log("=" * 154)
    log(
        "TRAIN/VALIDATION ONLY — internal TEST is NOT loaded."
    )
    log(
        f"device = {device}"
    )
    log(
        f"seeds = {SEEDS}"
    )
    log(
        "OLD vessel NN must have exactly 22,164 parameters."
    )

    # -----------------------------------------------------------------
    # Load only train/validation.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 1/7] Loading TRAIN/VALIDATION."
    )

    train = load_split(
        args.dataset_dir,
        "train",
    )

    val = load_split(
        args.dataset_dir,
        "validation",
    )

    tw_anchor_train, tw_anchor_val = (
        load_truewind_anchor(
            args.stage15a_truewind_dir
        )
    )

    if (
        len(
            train[
                "X"
            ]
        )
        != len(
            tw_anchor_train[
                "y"
            ]
        )
    ):
        raise RuntimeError(
            "TRAIN true-wind alignment mismatch."
        )

    if (
        len(
            val[
                "X"
            ]
        )
        != len(
            tw_anchor_val[
                "y"
            ]
        )
    ):
        raise RuntimeError(
            "VALIDATION true-wind alignment mismatch."
        )

    log(
        f"  TRAIN      = {len(train['X']):,}"
    )

    log(
        f"  VALIDATION = {len(val['X']):,}"
    )

    # -----------------------------------------------------------------
    # Apparent reference.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 2/7] Apparent-wind reference audit."
    )

    audit = cmod.audit_apparent_reference(
        {
            "X": train[
                "X"
            ],
            "y": train[
                "y"
            ],
            "apparent_candidate": train[
                "apparent"
            ],
            "apparent_key": train[
                "apparent_key"
            ],
        }
    )

    app_ref_train = (
        cmod.canonical_apparent_reference(
            {
                "X": train[
                    "X"
                ],
                "y": train[
                    "y"
                ],
                "apparent_candidate": train[
                    "apparent"
                ],
                "apparent_key": train[
                    "apparent_key"
                ],
            },
            audit,
        )
    )

    app_ref_val = (
        cmod.canonical_apparent_reference(
            {
                "X": val[
                    "X"
                ],
                "y": val[
                    "y"
                ],
                "apparent_candidate": val[
                    "apparent"
                ],
                "apparent_key": val[
                    "apparent_key"
                ],
            },
            audit,
        )
    )

    log(
        f"  source     = {audit['source']}"
    )

    log(
        f"  key        = {audit['key']}"
    )

    log(
        f"  convention = {audit['convention']}"
    )

    log(
        f"  TRAIN audit RMSE = "
        f"{audit['train_audit_rmse']:.9f}"
    )

    # -----------------------------------------------------------------
    # Rebuild old joint Ridge once.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 3/7] Rebuilding documented OLD joint Ridge, alpha=100."
    )

    old_ridge = fit_old_joint_ridge(
        train[
            "X"
        ],
        train[
            "y"
        ],
    )

    ridge_z_train, ridge_raw_train = (
        old_ridge_predict(
            old_ridge,
            train[
                "X"
            ],
        )
    )

    ridge_z_val, ridge_raw_val = (
        old_ridge_predict(
            old_ridge,
            val[
                "X"
            ],
        )
    )

    y_z_train = target_to_old_z(
        old_ridge,
        train[
            "y"
        ],
    )

    # Check X standardization diagnostics.
    x_mean = np.mean(
        train[
            "X"
        ].astype(
            np.float64
        ),
        axis=(
            0,
            1,
        ),
    )

    x_std = np.std(
        train[
            "X"
        ].astype(
            np.float64
        ),
        axis=(
            0,
            1,
        ),
        ddof=0,
    )

    log(
        f"  mean(|TRAIN X channel mean|) = "
        f"{np.mean(np.abs(x_mean)):.6f}"
    )

    log(
        f"  mean(TRAIN X channel std)    = "
        f"{np.mean(x_std):.6f}"
    )

    # Validation target for cmod vessel metrics.
    y_vh_val = val[
        "y"
    ][
        :,
        :,
        2:
        6
    ]

    # Ridge vessel+heading prediction.
    ridge_vh_val = ridge_raw_val[
        :,
        :,
        2:
        6
    ]

    # Deterministic Ridge baseline with old joint wind.
    old_ridge_metrics = (
        cmod.metrics_per_horizon(
            y_vh_val,
            ridge_vh_val,
            ridge_raw_val[
                :,
                :,
                0:
                2
            ],
            app_ref_val,
            "validation",
            "OldJointRidge",
        )
    )

    old_ridge_metrics.to_csv(
        out
        / "15E_old_joint_ridge_validation_per_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Prebuild Stage15B features.
    # -----------------------------------------------------------------
    feat_train = bmod.build_features(
        train[
            "X"
        ]
    )

    feat_val = bmod.build_features(
        val[
            "X"
        ]
    )

    per_seed_rows = []
    horizon_frames = []
    alpha_rows = []

    # -----------------------------------------------------------------
    # Five seeds.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 4/7] Five-seed training."
    )

    for seed_idx, seed in enumerate(
        SEEDS,
        start=1,
    ):
        log("")
        log(
            "-" * 154
        )
        log(
            f"[SEED {seed_idx}/5] {seed}"
        )
        log(
            "-" * 154
        )

        # =============================================================
        # NEW Stage15B true-wind.
        # =============================================================
        log(
            "  [NEW WIND] Training frozen Stage15B."
        )

        b_bundle = bmod.train_model(
            torch=torch,
            nn=nn,
            X_train_features=feat_train,
            residual_train=tw_anchor_train[
                "residual"
            ],
            X_val_features=feat_val,
            y_val=tw_anchor_val[
                "y"
            ],
            ridge_val=tw_anchor_val[
                "ridge"
            ],
            seed=seed,
            max_epochs=bmod.MAX_EPOCHS,
            fixed_epochs=None,
            verbose=True,
        )

        b_alpha = np.asarray(
            b_bundle[
                "best_alpha"
            ],
            dtype=np.float32,
        )

        raw_tw_val = bmod.predict_raw(
            torch,
            b_bundle[
                "model"
            ],
            bmod.transform_features(
                feat_val,
                b_bundle[
                    "scaler"
                ],
            ),
            b_bundle[
                "scaler"
            ],
            b_bundle[
                "device"
            ],
        )

        stage2_wind_val = (
            bmod.apply_alpha(
                tw_anchor_val[
                    "ridge"
                ],
                raw_tw_val,
                b_alpha,
            )
        )

        for j, h in enumerate(
            HORIZONS
        ):
            alpha_rows.append(
                {
                    "seed": seed,
                    "horizon_min": h,
                    "alpha_U": float(
                        b_alpha[
                            j,
                            0
                        ]
                    ),
                    "alpha_V": float(
                        b_alpha[
                            j,
                            1
                        ]
                    ),
                }
            )

        log(
            f"  [NEW WIND] best epoch = "
            f"{int(b_bundle['best_epoch'])}"
        )

        # =============================================================
        # OLD vessel compact residual.
        # =============================================================
        log(
            "  [OLD VESSEL] Training documented single-GRU64 gated residual."
        )

        old_bundle = train_old_vessel_model(
            torch=torch,
            nn=nn,
            cmod=cmod,
            seed=seed,
            X_train=train[
                "X"
            ],
            y_train_raw=train[
                "y"
            ],
            y_train_z=y_z_train,
            ridge_z_train=ridge_z_train,
            ridge_raw_train=ridge_raw_train,
            app_ref_train=app_ref_train,
            X_val=val[
                "X"
            ],
            y_val_raw=val[
                "y"
            ],
            ridge_z_val=ridge_z_val,
            ridge_raw_val=ridge_raw_val,
            app_ref_val=app_ref_val,
            old_ridge_model=old_ridge,
        )

        old_vessel_full_val, corr_val, gate_val = (
            predict_old_vessel(
                torch=torch,
                model=old_bundle[
                    "model"
                ],
                X=val[
                    "X"
                ],
                ridge_z=ridge_z_val,
                ridge_raw=ridge_raw_val,
                old_ridge_model=old_ridge,
                device=old_bundle[
                    "device"
                ],
            )
        )

        # =============================================================
        # OLD reconstructed model:
        # old joint Ridge wind + old compact vessel + old Ridge heading.
        # =============================================================
        old_metrics = (
            cmod.metrics_per_horizon(
                y_vh_val,
                old_vessel_full_val[
                    :,
                    :,
                    2:
                    6
                ],
                ridge_raw_val[
                    :,
                    :,
                    0:
                    2
                ],
                app_ref_val,
                "validation",
                "OldPhysicsCompact-Reconstructed",
            )
        )

        old_metrics.insert(
            0,
            "seed",
            seed,
        )

        horizon_frames.append(
            old_metrics
        )

        # =============================================================
        # 15E:
        # NEW Stage15B wind + SAME old vessel + SAME old Ridge heading.
        # =============================================================
        hybrid_metrics = (
            cmod.metrics_per_horizon(
                y_vh_val,
                old_vessel_full_val[
                    :,
                    :,
                    2:
                    6
                ],
                stage2_wind_val,
                app_ref_val,
                "validation",
                "15E-Hybrid",
            )
        )

        hybrid_metrics.insert(
            0,
            "seed",
            seed,
        )

        horizon_frames.append(
            hybrid_metrics
        )

        old_mean = mean_metrics(
            old_metrics
        )

        hybrid_mean = mean_metrics(
            hybrid_metrics
        )

        row_old = {
            "seed": seed,
            "model": "OldPhysicsCompact-Reconstructed",
            "Stage15B_best_epoch": np.nan,
            "OldVessel_best_epoch": int(
                old_bundle[
                    "best_epoch"
                ]
            ),
            **old_mean,
        }

        row_hybrid = {
            "seed": seed,
            "model": "15E-Hybrid",
            "Stage15B_best_epoch": int(
                b_bundle[
                    "best_epoch"
                ]
            ),
            "OldVessel_best_epoch": int(
                old_bundle[
                    "best_epoch"
                ]
            ),
            **hybrid_mean,
        }

        per_seed_rows.extend(
            [
                row_old,
                row_hybrid,
            ]
        )

        log("")
        log(
            "  [SEED COMPARISON]"
        )

        log(
            f"    OLD  AW vector = "
            f"{old_mean['AppVector_RMSE_mps']:.6f}"
        )

        log(
            f"    15E  AW vector = "
            f"{hybrid_mean['AppVector_RMSE_mps']:.6f}"
        )

        log(
            f"    OLD  AWS       = "
            f"{old_mean['AWS_RMSE_mps']:.6f}"
        )

        log(
            f"    15E  AWS       = "
            f"{hybrid_mean['AWS_RMSE_mps']:.6f}"
        )

        log(
            f"    Vessel vector  = "
            f"{hybrid_mean['Vessel_vector_RMSE_mps']:.6f} "
            f"(identical OLD/15E)"
        )

        log(
            f"    AWA MAE        = "
            f"{hybrid_mean['AWA_MAE_deg']:.6f}"
        )

        # Save validation predictions per seed for later diagnostics.
        np.savez_compressed(
            out
            / f"15E_seed_{seed}_validation_predictions.npz",
            y_raw=val[
                "y"
            ].astype(
                np.float32
            ),
            old_joint_ridge=(
                ridge_raw_val
                .astype(
                    np.float32
                )
            ),
            old_compact_vessel_hdg=(
                old_vessel_full_val[
                    :,
                    :,
                    2:
                    6
                ]
                .astype(
                    np.float32
                )
            ),
            stage15b_truewind=(
                stage2_wind_val
                .astype(
                    np.float32
                )
            ),
            vessel_correction_z=(
                corr_val
                .astype(
                    np.float32
                )
            ),
            vessel_gate=(
                gate_val
                .astype(
                    np.float32
                )
            ),
            alpha_truewind=(
                b_alpha
                .astype(
                    np.float32
                )
            ),
            horizons_min=np.asarray(
                HORIZONS,
                dtype=np.int64,
            ),
        )

        old_bundle[
            "history"
        ].to_csv(
            out
            / f"15E_seed_{seed}_old_vessel_training_history.csv",
            index=False,
            encoding="utf-8-sig",
        )

        del old_bundle
        del b_bundle

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Aggregate.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 5/7] Aggregating results."
    )

    per_seed_df = pd.DataFrame(
        per_seed_rows
    )

    horizon_df = pd.concat(
        horizon_frames,
        ignore_index=True,
    )

    alpha_df = pd.DataFrame(
        alpha_rows
    )

    old_seed_df = (
        per_seed_df.loc[
            per_seed_df[
                "model"
            ]
            == "OldPhysicsCompact-Reconstructed"
        ]
        .reset_index(
            drop=True
        )
    )

    hybrid_seed_df = (
        per_seed_df.loc[
            per_seed_df[
                "model"
            ]
            == "15E-Hybrid"
        ]
        .reset_index(
            drop=True
        )
    )

    old_agg = aggregate(
        old_seed_df
    )

    old_agg.insert(
        0,
        "model",
        "OldPhysicsCompact-Reconstructed",
    )

    hybrid_agg = aggregate(
        hybrid_seed_df
    )

    hybrid_agg.insert(
        0,
        "model",
        "15E-Hybrid",
    )

    aggregate_df = pd.concat(
        [
            old_agg,
            hybrid_agg,
        ],
        ignore_index=True,
    )

    # Paired old -> 15E wind-only effect.
    paired_rows = []

    for metric in [
        "AppVector_RMSE_mps",
        "AWS_RMSE_mps",
        "AWA_MAE_deg",
        "Vessel_vector_RMSE_mps",
    ]:
        old_vals = old_seed_df[
            metric
        ].to_numpy(
            dtype=np.float64
        )

        new_vals = hybrid_seed_df[
            metric
        ].to_numpy(
            dtype=np.float64
        )

        diff = (
            old_vals
            - new_vals
        )

        paired_rows.append(
            {
                "metric": metric,
                "old_mean": float(
                    np.mean(
                        old_vals
                    )
                ),
                "hybrid_mean": float(
                    np.mean(
                        new_vals
                    )
                ),
                "mean_reduction_old_minus_hybrid": float(
                    np.mean(
                        diff
                    )
                ),
                "hybrid_improvement_pct": float(
                    (
                        np.mean(
                            old_vals
                        )
                        - np.mean(
                            new_vals
                        )
                    )
                    / np.mean(
                        old_vals
                    )
                    * 100.0
                ),
                "paired_diff_std": float(
                    np.std(
                        diff,
                        ddof=1,
                    )
                ),
            }
        )

    paired_df = pd.DataFrame(
        paired_rows
    )

    # -----------------------------------------------------------------
    # Save CSV.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 6/7] Saving outputs."
    )

    per_seed_df.to_csv(
        out
        / "15E_per_seed_validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    aggregate_df.to_csv(
        out
        / "15E_validation_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired_df.to_csv(
        out
        / "15E_old_vs_hybrid_paired_effect.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_df.to_csv(
        out
        / "15E_per_horizon_all_seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    alpha_df.to_csv(
        out
        / "15E_truewind_alpha_all_seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Final report.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 7/7] Final report."
    )

    def mstd(
        df,
        col,
    ):
        vals = df[
            col
        ].to_numpy(
            dtype=np.float64
        )

        return (
            float(
                np.mean(
                    vals
                )
            ),
            float(
                np.std(
                    vals,
                    ddof=1,
                )
            ),
        )

    old_aw = mstd(
        old_seed_df,
        "AppVector_RMSE_mps",
    )

    hybrid_aw = mstd(
        hybrid_seed_df,
        "AppVector_RMSE_mps",
    )

    old_vessel = mstd(
        old_seed_df,
        "Vessel_vector_RMSE_mps",
    )

    hybrid_vessel = mstd(
        hybrid_seed_df,
        "Vessel_vector_RMSE_mps",
    )

    old_aws = mstd(
        old_seed_df,
        "AWS_RMSE_mps",
    )

    hybrid_aws = mstd(
        hybrid_seed_df,
        "AWS_RMSE_mps",
    )

    old_awa = mstd(
        old_seed_df,
        "AWA_MAE_deg",
    )

    hybrid_awa = mstd(
        hybrid_seed_df,
        "AWA_MAE_deg",
    )

    log("")
    log("=" * 154)
    log(
        "15E FINAL — NEW WIND + OLD COMPACT VESSEL, SD1090 VALIDATION, FIVE SEEDS"
    )
    log("=" * 154)

    log(
        f"OLD reconstructed AW vector = "
        f"{old_aw[0]:.6f} ± {old_aw[1]:.6f}"
    )

    log(
        f"15E hybrid      AW vector = "
        f"{hybrid_aw[0]:.6f} ± {hybrid_aw[1]:.6f}"
    )

    log(
        f"OLD reconstructed Vessel    = "
        f"{old_vessel[0]:.6f} ± {old_vessel[1]:.6f}"
    )

    log(
        f"15E hybrid      Vessel       = "
        f"{hybrid_vessel[0]:.6f} ± {hybrid_vessel[1]:.6f}"
    )

    log(
        f"OLD reconstructed AWS       = "
        f"{old_aws[0]:.6f} ± {old_aws[1]:.6f}"
    )

    log(
        f"15E hybrid      AWS          = "
        f"{hybrid_aws[0]:.6f} ± {hybrid_aws[1]:.6f}"
    )

    log(
        f"OLD reconstructed AWA MAE   = "
        f"{old_awa[0]:.6f} ± {old_awa[1]:.6f}"
    )

    log(
        f"15E hybrid      AWA MAE      = "
        f"{hybrid_awa[0]:.6f} ± {hybrid_awa[1]:.6f}"
    )

    log("")
    log(
        "[PAIRED OLD -> 15E EFFECT]"
    )

    log(
        paired_df.to_string(
            index=False
        )
    )

    log("")
    log(
        f"Old vessel NN parameters  = "
        f"{OLD_VESSEL_NN_PARAMETERS:,}"
    )

    log(
        f"Stage15B NN parameters     = "
        f"{STAGE15B_NN_PARAMETERS:,}"
    )

    log(
        f"15E neural parameters      = "
        f"{HYBRID_NEURAL_PARAMETERS:,}"
    )

    log(
        f"15E total fitted coefficients incl. Ridge = "
        f"{HYBRID_TOTAL_FITTED_COEFFICIENTS:,}"
    )

    report = {
        "experiment": "15E",
        "internal_test_loaded": False,
        "purpose": (
            "Isolate the effect of replacing the old joint-Ridge wind "
            "with the new Stage15B true-wind forecast while preserving "
            "the documented old compact vessel branch."
        ),
        "seeds": SEEDS,
        "old_documented_architecture": {
            "joint_ridge_alpha": OLD_RIDGE_ALPHA,
            "GRU_layers": OLD_GRU_LAYERS,
            "GRU_hidden": OLD_GRU_HIDDEN,
            "dense_hidden": OLD_DENSE_HIDDEN,
            "dropout": OLD_DROPOUT,
            "residual_outputs": 10,
            "gate_outputs": 10,
            "gate_bias_init": -1.0,
            "lambda_apparent_earth": OLD_LAMBDA_APP,
            "lambda_residual_reg": OLD_LAMBDA_RES,
            "optimizer": "Adam",
            "learning_rate": OLD_LR,
            "batch_size": OLD_BATCH_SIZE,
            "max_epochs": OLD_MAX_EPOCHS,
            "patience": OLD_PATIENCE,
            "NN_parameters": OLD_VESSEL_NN_PARAMETERS,
        },
        "hybrid": {
            "true_wind": (
                "new Stage15B final"
            ),
            "vessel": (
                "old compact vessel-only gated residual"
            ),
            "heading": (
                "old joint Ridge"
            ),
            "neural_parameters": HYBRID_NEURAL_PARAMETERS,
            "total_fitted_coefficients": HYBRID_TOTAL_FITTED_COEFFICIENTS,
        },
        "old_manuscript_table6_reference": OLD_MANUSCRIPT_TABLE6,
        "old_reconstructed": {
            "AW_vector_mean": old_aw[
                0
            ],
            "AW_vector_std": old_aw[
                1
            ],
            "Vessel_vector_mean": old_vessel[
                0
            ],
            "Vessel_vector_std": old_vessel[
                1
            ],
            "AWS_mean": old_aws[
                0
            ],
            "AWS_std": old_aws[
                1
            ],
            "AWA_MAE_mean": old_awa[
                0
            ],
            "AWA_MAE_std": old_awa[
                1
            ],
        },
        "hybrid_result": {
            "AW_vector_mean": hybrid_aw[
                0
            ],
            "AW_vector_std": hybrid_aw[
                1
            ],
            "Vessel_vector_mean": hybrid_vessel[
                0
            ],
            "Vessel_vector_std": hybrid_vessel[
                1
            ],
            "AWS_mean": hybrid_aws[
                0
            ],
            "AWS_std": hybrid_aws[
                1
            ],
            "AWA_MAE_mean": hybrid_awa[
                0
            ],
            "AWA_MAE_std": hybrid_awa[
                1
            ],
        },
        "paired_effect": paired_df.to_dict(
            orient="records"
        ),
    }

    save_json(
        out
        / "15E_REPORT.json",
        report,
    )

    with (
        out
        / "15E_REPORT.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "15E — NEW WIND + OLD COMPACT VESSEL\n"
        )

        f.write(
            "="
            * 140
            + "\n\n"
        )

        f.write(
            "TRAIN/VALIDATION ONLY. INTERNAL TEST NOT LOADED.\n\n"
        )

        f.write(
            f"OLD reconstructed AW vector = "
            f"{old_aw[0]:.8f} ± {old_aw[1]:.8f}\n"
        )

        f.write(
            f"15E hybrid AW vector         = "
            f"{hybrid_aw[0]:.8f} ± {hybrid_aw[1]:.8f}\n"
        )

        f.write(
            f"OLD reconstructed Vessel    = "
            f"{old_vessel[0]:.8f} ± {old_vessel[1]:.8f}\n"
        )

        f.write(
            f"15E hybrid Vessel           = "
            f"{hybrid_vessel[0]:.8f} ± {hybrid_vessel[1]:.8f}\n"
        )

        f.write(
            f"OLD reconstructed AWS       = "
            f"{old_aws[0]:.8f} ± {old_aws[1]:.8f}\n"
        )

        f.write(
            f"15E hybrid AWS              = "
            f"{hybrid_aws[0]:.8f} ± {hybrid_aws[1]:.8f}\n"
        )

        f.write(
            f"OLD reconstructed AWA MAE   = "
            f"{old_awa[0]:.8f} ± {old_awa[1]:.8f}\n"
        )

        f.write(
            f"15E hybrid AWA MAE          = "
            f"{hybrid_awa[0]:.8f} ± {hybrid_awa[1]:.8f}\n\n"
        )

        f.write(
            "PAIRED EFFECT\n"
        )

        f.write(
            "-"
            * 140
            + "\n"
        )

        f.write(
            paired_df.to_string(
                index=False
            )
        )

    log("")
    log(
        f"[SAVED] {out}"
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
