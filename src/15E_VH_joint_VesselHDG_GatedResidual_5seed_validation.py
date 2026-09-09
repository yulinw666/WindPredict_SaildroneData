# -*- coding: utf-8 -*-
"""
15E_VH_joint_VesselHDG_GatedResidual_5seed_validation.py

15E-VH
======
Replace the separate HDG residual head of 15E-H with one unified
Vessel-HDG motion residual branch.

Frozen TrueWind:
    Use the already generated per-seed Stage15B true-wind predictions
    stored by 15E-H.

Joint motion target:
    [VesselEast, VesselNorth, HDG_sin, HDG_cos]
    for 5 horizons -> 20 outputs.

Unified residual formulation:
    Y_motion = Y_Ridge + Gate * DeltaY

where the SAME single-GRU64 + Dense64 representation predicts:
    20 residual outputs
    20 sigmoid gate outputs

There is NO separate HDG network and NO HDG-specific global alpha.

Heading pair is unit-normalized after residual correction.

This is a controlled comparison against 15E-H:
    15E-H:
        old vessel gated branch + separate 1300-param HDG head

    15E-VH:
        one integrated vessel+HDG gated branch

Both have exactly the SAME total neural parameter count:
    Stage15B TrueWind = 26,074
    15E-H motion      = 22,164 + 1,300 = 23,464
    15E-VH motion     = 23,464
    total             = 49,538

TRAIN/VALIDATION only.
Internal TEST is NOT loaded.

Dependencies in --src-dir:
    15E_hybrid_NewWind_OldCompactVessel_5seed_validation.py
    15C_train_1min_VesselHDG_Residual_ApparentWind.py

Required frozen baseline directory:
    15E_H_HDG_GatedResidual_5seed_v0_1

It must contain:
    15E_H_seed_<seed>_validation_predictions.npz

Recommended PowerShell
----------------------
python "D:\\project\\WindPredict_SaildroneData\\src\\15E_VH_joint_VesselHDG_GatedResidual_5seed_validation.py" --dataset-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\SD1090_TPOS2024_JointForecasting_v0_1" --stage15eh-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15E_H_HDG_GatedResidual_5seed_v0_1" --src-dir "D:\\project\\WindPredict_SaildroneData\\src" --output-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15E_VH_JointVesselHDG_5seed_v0_1"
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

DEFAULT_STAGE15EH_DIR = (
    ROOT
    / "data"
    / "forecasting"
    / "15E_H_HDG_GatedResidual_5seed_v0_1"
)

DEFAULT_SRC_DIR = ROOT / "src"

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "data"
    / "forecasting"
    / "15E_VH_JointVesselHDG_5seed_v0_1"
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
# Frozen old motion-branch settings.
# ---------------------------------------------------------------------
OLD_RIDGE_ALPHA = 100.0

GRU_HIDDEN = 64
GRU_LAYERS = 1
DENSE_HIDDEN = 64
DROPOUT = 0.10

LR = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 80
PATIENCE = 12
GRAD_CLIP = 1.0

LAMBDA_APP = 2.0
LAMBDA_RES = 0.01

# Equal standardized direct losses for vessel and heading.
LAMBDA_HDG = 1.0

GATE_BIAS_INIT = -1.0
EPS = 1e-8

# Outputs:
# 5 horizons x [VE, VN, HDG_sin, HDG_cos] = 20
MOTION_OUTPUTS = 20

# Original old vessel network:
# GRU 11->64 one layer                    = 14,784
# fusion Linear(64+30 ->64)               =  6,080
# vessel residual Linear(64->10)          =    650
# vessel gate Linear(64->10)              =    650
# total                                   = 22,164
#
# Extending each head from 10 -> 20:
# adds 2 * (64*10 + 10) = 1,300
EXPECTED_MOTION_PARAMS = 23464

STAGE15B_PARAMS = 26074
EXPECTED_TOTAL_NEURAL_PARAMS = (
    STAGE15B_PARAMS
    + EXPECTED_MOTION_PARAMS
)  # 49,538

# Same as 15E-H.
BASELINE_15EH_TOTAL_NEURAL_PARAMS = 49538


# =============================================================================
# Generic utilities
# =============================================================================

def log(msg=""):
    print(msg, flush=True)


def load_module(
    name: str,
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Cannot import module from {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def save_json(
    path: Path,
    obj,
):
    def cv(v):
        if isinstance(v, Path):
            return str(v)

        if isinstance(v, np.ndarray):
            return v.tolist()

        if isinstance(v, np.integer):
            return int(v)

        if isinstance(v, np.floating):
            return (
                None
                if not np.isfinite(v)
                else float(v)
            )

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


def seed_everything(
    torch,
    seed,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
                model.parameters()
            )
            if p.requires_grad
        )
    )


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


# =============================================================================
# Frozen 15E-H validation baseline
# =============================================================================

def load_15eh_seed(
    stage15eh_dir: Path,
    seed: int,
):
    path = (
        stage15eh_dir
        / f"15E_H_seed_{seed}_validation_predictions.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen 15E-H artifact: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as z:
        required = [
            "y_raw",
            "stage15b_truewind",
            "old_vessel_hdg",
            "final_hdg_q",
        ]

        missing = [
            k
            for k in required
            if k not in z.files
        ]

        if missing:
            raise RuntimeError(
                f"{path.name}: missing keys {missing}"
            )

        y_raw = np.asarray(
            z["y_raw"],
            dtype=np.float32,
        )

        wind = np.asarray(
            z["stage15b_truewind"],
            dtype=np.float32,
        )

        old_vh = np.asarray(
            z["old_vessel_hdg"],
            dtype=np.float32,
        )

        final_q = np.asarray(
            z["final_hdg_q"],
            dtype=np.float32,
        )

    baseline_vh = old_vh.copy()
    baseline_vh[:, :, 2:4] = final_q

    return {
        "y_raw": y_raw,
        "truewind": wind,
        "baseline_vh": baseline_vh,
    }


# =============================================================================
# Joint Vessel-HDG gated residual network
# =============================================================================

def build_joint_motion_model_class(
    torch,
    nn,
):
    class JointVesselHDGGatedResidual(
        nn.Module
    ):
        def __init__(self):
            super().__init__()

            self.gru = nn.GRU(
                input_size=11,
                hidden_size=GRU_HIDDEN,
                num_layers=1,
                batch_first=True,
            )

            self.history_dropout = nn.Dropout(
                DROPOUT
            )

            self.fusion = nn.Sequential(
                nn.Linear(
                    GRU_HIDDEN + 30,
                    DENSE_HIDDEN,
                ),
                nn.ReLU(),
                nn.Dropout(
                    DROPOUT
                ),
            )

            self.residual_head = nn.Linear(
                DENSE_HIDDEN,
                MOTION_OUTPUTS,
            )

            self.gate_head = nn.Linear(
                DENSE_HIDDEN,
                MOTION_OUTPUTS,
            )

            # Epoch 0 = exact Ridge.
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
                GATE_BIAS_INIT,
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
                    4,
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
                    4,
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

    return JointVesselHDGGatedResidual


# =============================================================================
# Old joint-Ridge target conversion helpers
# =============================================================================

def motion_scaler_arrays(
    old_ridge_model,
):
    mean_all = np.asarray(
        old_ridge_model[
            "y_mean_raw"
        ],
        dtype=np.float64,
    ).reshape(
        5,
        6,
    )

    std_all = np.asarray(
        old_ridge_model[
            "y_std_raw"
        ],
        dtype=np.float64,
    ).reshape(
        5,
        6,
    )

    return (
        mean_all[
            :,
            2:
            6
        ].astype(
            np.float32
        ),
        std_all[
            :,
            2:
            6
        ].astype(
            np.float32
        ),
    )


def normalize_heading_pair_np(
    q,
):
    q = np.asarray(
        q,
        dtype=np.float32,
    )

    n = np.sqrt(
        np.sum(
            q * q,
            axis=-1,
            keepdims=True,
        )
    )

    n = np.maximum(
        n,
        EPS,
    )

    return (
        q / n
    ).astype(
        np.float32
    )


def motion_z_to_final_raw(
    motion_z,
    old_ridge_model,
):
    mean_m, std_m = motion_scaler_arrays(
        old_ridge_model
    )

    raw = (
        np.asarray(
            motion_z,
            dtype=np.float32,
        )
        * std_m[
            None,
            :,
            :
        ]
        + mean_m[
            None,
            :,
            :
        ]
    )

    out = raw.copy()

    out[
        :,
        :,
        2:
        4
    ] = normalize_heading_pair_np(
        out[
            :,
            :,
            2:
            4
        ]
    )

    return out.astype(
        np.float32
    )


def predict_joint_motion(
    *,
    torch,
    model,
    X,
    ridge_z,
    old_ridge_model,
    device,
):
    model.eval()

    pred_all = []
    gate_all = []
    corr_all = []
    residual_all = []

    ridge_motion_z = ridge_z[
        :,
        :,
        2:
        6
    ]

    with torch.no_grad():
        for a in range(
            0,
            len(X),
            BATCH_SIZE,
        ):
            b = min(
                a + BATCH_SIZE,
                len(X),
            )

            seq_t = (
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

            ridge_flat_t = (
                torch.from_numpy(
                    ridge_z[
                        a:
                        b
                    ]
                    .reshape(
                        b - a,
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
                residual, gate, correction = model(
                    seq_t,
                    ridge_flat_t,
                )

            residual_np = (
                residual.detach()
                .float()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            gate_np = (
                gate.detach()
                .float()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            correction_np = (
                correction.detach()
                .float()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            motion_z = (
                ridge_motion_z[
                    a:
                    b
                ]
                + correction_np
            )

            pred_raw = motion_z_to_final_raw(
                motion_z,
                old_ridge_model,
            )

            pred_all.append(
                pred_raw
            )

            gate_all.append(
                gate_np
            )

            corr_all.append(
                correction_np
            )

            residual_all.append(
                residual_np
            )

    return {
        "pred_raw": np.concatenate(
            pred_all,
            axis=0,
        ),
        "gate": np.concatenate(
            gate_all,
            axis=0,
        ),
        "correction_z": np.concatenate(
            corr_all,
            axis=0,
        ),
        "residual_z": np.concatenate(
            residual_all,
            axis=0,
        ),
    }


# =============================================================================
# Validation score
# =============================================================================

def motion_validation_score(
    cmod,
    y_motion_raw,
    pred_motion_raw,
    truewind_for_metrics,
    app_ref,
    ridge_motion_raw,
):
    """
    Checkpoint selection uses ONLY direct motion-target quality:
        50% vessel-vector ratio
        50% HDG-MAE ratio

    It does not directly optimize AWA.
    """
    cand = cmod.metrics_per_horizon(
        y_motion_raw,
        pred_motion_raw,
        truewind_for_metrics,
        app_ref,
        "validation",
        "candidate",
    )

    base = cmod.metrics_per_horizon(
        y_motion_raw,
        ridge_motion_raw,
        truewind_for_metrics,
        app_ref,
        "validation",
        "ridge",
    )

    cand_v = float(
        cand[
            "Vessel_vector_RMSE_mps"
        ].mean()
    )

    base_v = float(
        base[
            "Vessel_vector_RMSE_mps"
        ].mean()
    )

    cand_h = float(
        cand[
            "HDG_MAE_deg"
        ].mean()
    )

    base_h = float(
        base[
            "HDG_MAE_deg"
        ].mean()
    )

    score = 0.5 * (
        cand_v
        / max(
            base_v,
            EPS,
        )
    ) + 0.5 * (
        cand_h
        / max(
            base_h,
            EPS,
        )
    )

    return (
        float(score),
        cand,
        base,
    )


# =============================================================================
# Training
# =============================================================================

def train_joint_motion(
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
    truewind_val,
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

    Model = build_joint_motion_model_class(
        torch,
        nn,
    )

    model = Model().to(
        device
    )

    params = count_parameters(
        model
    )

    if (
        params
        != EXPECTED_MOTION_PARAMS
    ):
        raise RuntimeError(
            f"Joint motion params = {params}, "
            f"expected {EXPECTED_MOTION_PARAMS}."
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    grad_scaler = create_grad_scaler(
        torch,
        device,
    )

    mean_m, std_m = motion_scaler_arrays(
        old_ridge_model
    )

    mean_m_t = (
        torch.from_numpy(
            mean_m
        )
        .to(
            device
        )
    )

    std_m_t = (
        torch.from_numpy(
            std_m
        )
        .to(
            device
        )
    )

    # Apparent-wind scaling from TRAIN only.
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
        app_std < 1e-6,
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

    ridge_motion_raw_val = ridge_raw_val[
        :,
        :,
        2:
        6
    ].copy()

    ridge_motion_raw_val[
        :,
        :,
        2:
        4
    ] = normalize_heading_pair_np(
        ridge_motion_raw_val[
            :,
            :,
            2:
            4
        ]
    )

    # Epoch 0 = Ridge.
    best_score, _, _ = motion_validation_score(
        cmod,
        y_val_raw[
            :,
            :,
            2:
            6
        ],
        ridge_motion_raw_val,
        truewind_val,
        app_ref_val,
        ridge_motion_raw_val,
    )

    best_state = state_dict_cpu(
        model
    )

    best_epoch = 0
    patience = 0
    history = []

    n = len(
        X_train
    )

    ridge_motion_z_train = ridge_z_train[
        :,
        :,
        2:
        6
    ]

    target_motion_z_train = y_train_z[
        :,
        :,
        2:
        6
    ]

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        rng = np.random.default_rng(
            seed + epoch * 1009
        )

        order = np.arange(
            n
        )

        rng.shuffle(
            order
        )

        total = 0.0
        total_v = 0.0
        total_h = 0.0
        total_app = 0.0
        total_reg = 0.0
        seen = 0

        for a in range(
            0,
            n,
            BATCH_SIZE,
        ):
            idx = order[
                a:
                a + BATCH_SIZE
            ]

            seq_t = (
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

            ridge_flat_t = (
                torch.from_numpy(
                    ridge_z_train[
                        idx
                    ]
                    .reshape(
                        len(idx),
                        -1,
                    )
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            ridge_motion_t = (
                torch.from_numpy(
                    ridge_motion_z_train[
                        idx
                    ]
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            target_motion_t = (
                torch.from_numpy(
                    target_motion_z_train[
                        idx
                    ]
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            # Same old-branch physics input as 15E:
            # old joint-Ridge wind, not future truth.
            ridge_wind_raw_t = (
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

            app_ref_t = (
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
                _, _, correction = model(
                    seq_t,
                    ridge_flat_t,
                )

                pred_motion_z = (
                    ridge_motion_t
                    + correction
                )

                pred_v_z = pred_motion_z[
                    :,
                    :,
                    0:
                    2
                ]

                target_v_z = target_motion_t[
                    :,
                    :,
                    0:
                    2
                ]

                pred_h_z = pred_motion_z[
                    :,
                    :,
                    2:
                    4
                ]

                target_h_z = target_motion_t[
                    :,
                    :,
                    2:
                    4
                ]

                loss_vessel = torch.mean(
                    (
                        pred_v_z
                        - target_v_z
                    )
                    ** 2
                )

                loss_hdg = torch.mean(
                    (
                        pred_h_z
                        - target_h_z
                    )
                    ** 2
                )

                # Convert vessel standardized outputs to m/s.
                pred_v_raw = (
                    pred_v_z
                    * std_m_t[
                        None,
                        :,
                        0:
                        2
                    ]
                    + mean_m_t[
                        None,
                        :,
                        0:
                        2
                    ]
                )

                app_pred = (
                    ridge_wind_raw_t
                    - pred_v_raw
                )

                app_err = (
                    (
                        app_pred
                        - app_ref_t
                    )
                    / app_std_t[
                        None,
                        None,
                        :
                    ]
                )

                loss_app = torch.mean(
                    torch.sum(
                        app_err * app_err,
                        dim=-1,
                    )
                )

                loss_reg = torch.mean(
                    correction
                    * correction
                )

                loss = (
                    loss_vessel
                    + LAMBDA_HDG
                    * loss_hdg
                    + LAMBDA_APP
                    * loss_app
                    + LAMBDA_RES
                    * loss_reg
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

            total += float(
                loss.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            total_v += float(
                loss_vessel.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            total_h += float(
                loss_hdg.detach()
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

            total_reg += float(
                loss_reg.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            seen += bn

        pred_val = predict_joint_motion(
            torch=torch,
            model=model,
            X=X_val,
            ridge_z=ridge_z_val,
            old_ridge_model=old_ridge_model,
            device=device,
        )

        score, cand_metrics, _ = motion_validation_score(
            cmod,
            y_val_raw[
                :,
                :,
                2:
                6
            ],
            pred_val[
                "pred_raw"
            ],
            truewind_val,
            app_ref_val,
            ridge_motion_raw_val,
        )

        cand_v = float(
            cand_metrics[
                "Vessel_vector_RMSE_mps"
            ].mean()
        )

        cand_h = float(
            cand_metrics[
                "HDG_MAE_deg"
            ].mean()
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
                "train_loss": total / seen,
                "train_vessel_loss": total_v / seen,
                "train_hdg_loss": total_h / seen,
                "train_app_loss": total_app / seen,
                "train_residual_reg": total_reg / seen,
                "validation_motion_score": score,
                "validation_vessel_vector_RMSE": cand_v,
                "validation_HDG_MAE_deg": cand_h,
                "best_validation_motion_score": best_score,
            }
        )

        if (
            epoch <= 3
            or epoch % 10 == 0
            or improved
        ):
            log(
                f"      VH epoch={epoch:03d} | "
                f"loss={total/seen:.6f} | "
                f"score={score:.6f} | "
                f"V={cand_v:.6f} | "
                f"HDG={cand_h:.6f}"
                + (
                    " *"
                    if improved
                    else ""
                )
            )

        if (
            patience >= PATIENCE
        ):
            break

    model.load_state_dict(
        best_state,
        strict=True,
    )

    final_pred = predict_joint_motion(
        torch=torch,
        model=model,
        X=X_val,
        ridge_z=ridge_z_val,
        old_ridge_model=old_ridge_model,
        device=device,
    )

    final_score, final_metrics, ridge_metrics = (
        motion_validation_score(
            cmod,
            y_val_raw[
                :,
                :,
                2:
                6
            ],
            final_pred[
                "pred_raw"
            ],
            truewind_val,
            app_ref_val,
            ridge_motion_raw_val,
        )
    )

    return {
        "model": model,
        "device": device,
        "parameters": params,
        "best_epoch": int(
            best_epoch
        ),
        "best_score": float(
            final_score
        ),
        "pred_val": final_pred,
        "metrics_val": final_metrics,
        "ridge_metrics_val": ridge_metrics,
        "history": pd.DataFrame(
            history
        ),
    }


# =============================================================================
# Reporting helpers
# =============================================================================

def mean_metrics(
    df,
):
    cols = [
        "Vessel_East_RMSE_mps",
        "Vessel_North_RMSE_mps",
        "Vessel_vector_RMSE_mps",
        "SOG_RMSE_mps",
        "HDG_MAE_deg",
        "HDG_RMSE_deg",
        "AppVector_RMSE_mps",
        "AWS_RMSE_mps",
        "AWA_MAE_deg",
        "AWA_RMSE_deg",
    ]

    return {
        c: float(
            df[
                c
            ].mean()
        )
        for c in cols
    }


def aggregate_df(
    df,
):
    rows = []

    for col in df.columns:
        if col in {
            "seed",
            "model",
        }:
            continue

        if not np.issubdtype(
            df[
                col
            ].dtype,
            np.number,
        ):
            continue

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


def paired_effect(
    base,
    candidate,
    metrics,
):
    rows = []

    for metric in metrics:
        a = base[
            metric
        ].to_numpy(
            dtype=np.float64
        )

        b = candidate[
            metric
        ].to_numpy(
            dtype=np.float64
        )

        d = a - b

        rows.append(
            {
                "metric": metric,
                "15E_H_mean": float(
                    np.mean(
                        a
                    )
                ),
                "15E_VH_mean": float(
                    np.mean(
                        b
                    )
                ),
                "mean_reduction_15E_H_minus_15E_VH": float(
                    np.mean(
                        d
                    )
                ),
                "improvement_pct": float(
                    (
                        np.mean(
                            a
                        )
                        - np.mean(
                            b
                        )
                    )
                    / max(
                        abs(
                            np.mean(
                                a
                            )
                        ),
                        EPS,
                    )
                    * 100.0
                ),
                "paired_diff_std": float(
                    np.std(
                        d,
                        ddof=1,
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
        "--stage15eh-dir",
        type=Path,
        default=DEFAULT_STAGE15EH_DIR,
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

    emod = load_module(
        "stage15e_helper_for_15evh",
        args.src_dir
        / "15E_hybrid_NewWind_OldCompactVessel_5seed_validation.py",
    )

    cmod = load_module(
        "stage15c_geometry_for_15evh",
        args.src_dir
        / "15C_train_1min_VesselHDG_Residual_ApparentWind.py",
    )

    torch, nn = emod.import_torch()

    emod.configure_torch(
        torch
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    log("=" * 166)

    log(
        "15E-VH — JOINT VESSEL+HDG GATED RESIDUAL, "
        "FIVE-SEED VALIDATION"
    )

    log("=" * 166)

    log(
        "TrueWind is frozen from the completed 15E-H run."
    )

    log(
        "Vessel and HDG are trained jointly in ONE "
        "single-GRU64 gated motion branch."
    )

    log(
        "No separate HDG head. No HDG alpha."
    )

    log(
        "TRAIN/VALIDATION ONLY — INTERNAL TEST IS NOT LOADED."
    )

    log(
        f"device = {device}"
    )

    log(
        f"seeds = {SEEDS}"
    )

    log(
        f"joint motion parameters = "
        f"{EXPECTED_MOTION_PARAMS:,}"
    )

    log(
        f"total neural parameters = "
        f"{EXPECTED_TOTAL_NEURAL_PARAMS:,}"
    )

    if (
        EXPECTED_TOTAL_NEURAL_PARAMS
        != BASELINE_15EH_TOTAL_NEURAL_PARAMS
    ):
        raise RuntimeError(
            "15E-VH and 15E-H must have equal neural parameter counts."
        )

    # -----------------------------------------------------------------
    # Data.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 1/6] Loading TRAIN/VALIDATION."
    )

    train = emod.load_split(
        args.dataset_dir,
        "train",
    )

    val = emod.load_split(
        args.dataset_dir,
        "validation",
    )

    log(
        f"  TRAIN      = {len(train['X']):,}"
    )

    log(
        f"  VALIDATION = {len(val['X']):,}"
    )

    # -----------------------------------------------------------------
    # Apparent-wind audit.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 2/6] Apparent-wind reference audit."
    )

    train_for_c = {
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

    val_for_c = {
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
    }

    audit = cmod.audit_apparent_reference(
        train_for_c
    )

    app_ref_train = (
        cmod.canonical_apparent_reference(
            train_for_c,
            audit,
        )
    )

    app_ref_val = (
        cmod.canonical_apparent_reference(
            val_for_c,
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
    # Rebuild deterministic old joint Ridge alpha=100.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 3/6] Rebuilding old joint Ridge alpha=100."
    )

    old_ridge = emod.fit_old_joint_ridge(
        train[
            "X"
        ],
        train[
            "y"
        ],
    )

    ridge_z_train, ridge_raw_train = (
        emod.old_ridge_predict(
            old_ridge,
            train[
                "X"
            ],
        )
    )

    ridge_z_val, ridge_raw_val = (
        emod.old_ridge_predict(
            old_ridge,
            val[
                "X"
            ],
        )
    )

    y_z_train = emod.target_to_old_z(
        old_ridge,
        train[
            "y"
        ],
    )

    y_motion_val = val[
        "y"
    ][
        :,
        :,
        2:
        6
    ]

    per_seed_rows = []
    horizon_frames = []
    gate_rows = []

    # -----------------------------------------------------------------
    # Five seeds.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 4/6] Five-seed integrated motion training."
    )

    for seed_idx, seed in enumerate(
        SEEDS,
        start=1,
    ):
        log("")
        log(
            "-" * 166
        )

        log(
            f"[SEED {seed_idx}/5] {seed}"
        )

        log(
            "-" * 166
        )

        # Exact frozen 15E-H wind + baseline outputs.
        frozen = load_15eh_seed(
            args.stage15eh_dir,
            seed,
        )

        if not np.allclose(
            frozen[
                "y_raw"
            ],
            val[
                "y"
            ],
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"Seed {seed}: frozen 15E-H artifact "
                f"does not align with validation data."
            )

        frozen_wind = frozen[
            "truewind"
        ]

        baseline_vh = frozen[
            "baseline_vh"
        ]

        baseline_metrics = (
            cmod.metrics_per_horizon(
                y_motion_val,
                baseline_vh,
                frozen_wind,
                app_ref_val,
                "validation",
                "15E-H",
            )
        )

        log(
            "  [JOINT MOTION] Training one GRU64 branch "
            "for [VE,VN,HDG_sin,HDG_cos]."
        )

        bundle = train_joint_motion(
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
            truewind_val=frozen_wind,
            old_ridge_model=old_ridge,
        )

        pred_vh = bundle[
            "pred_val"
        ][
            "pred_raw"
        ]

        candidate_metrics = (
            cmod.metrics_per_horizon(
                y_motion_val,
                pred_vh,
                frozen_wind,
                app_ref_val,
                "validation",
                "15E-VH",
            )
        )

        base_mean = mean_metrics(
            baseline_metrics
        )

        cand_mean = mean_metrics(
            candidate_metrics
        )

        bf = baseline_metrics.copy()

        bf.insert(
            0,
            "seed",
            seed,
        )

        horizon_frames.append(
            bf
        )

        cf = candidate_metrics.copy()

        cf.insert(
            0,
            "seed",
            seed,
        )

        horizon_frames.append(
            cf
        )

        gate = bundle[
            "pred_val"
        ][
            "gate"
        ]

        for j, h in enumerate(
            HORIZONS
        ):
            for c, name in enumerate(
                [
                    "VesselEast",
                    "VesselNorth",
                    "HDG_sin",
                    "HDG_cos",
                ]
            ):
                g = gate[
                    :,
                    j,
                    c
                ]

                gate_rows.append(
                    {
                        "seed": seed,
                        "horizon_min": h,
                        "component": name,
                        "mean_gate": float(
                            np.mean(
                                g
                            )
                        ),
                        "std_gate": float(
                            np.std(
                                g,
                                ddof=0,
                            )
                        ),
                        "p10_gate": float(
                            np.percentile(
                                g,
                                10,
                            )
                        ),
                        "p50_gate": float(
                            np.percentile(
                                g,
                                50,
                            )
                        ),
                        "p90_gate": float(
                            np.percentile(
                                g,
                                90,
                            )
                        ),
                    }
                )

        row_base = {
            "seed": seed,
            "model": "15E-H",
            "best_epoch": np.nan,
            "motion_parameters": (
                EXPECTED_MOTION_PARAMS
            ),
            **base_mean,
        }

        row_cand = {
            "seed": seed,
            "model": "15E-VH",
            "best_epoch": int(
                bundle[
                    "best_epoch"
                ]
            ),
            "motion_parameters": int(
                bundle[
                    "parameters"
                ]
            ),
            **cand_mean,
        }

        per_seed_rows.extend(
            [
                row_base,
                row_cand,
            ]
        )

        log("")
        log(
            "  [SEED SUMMARY]"
        )

        log(
            f"      Vessel vector | "
            f"15E-H={base_mean['Vessel_vector_RMSE_mps']:.6f} | "
            f"15E-VH={cand_mean['Vessel_vector_RMSE_mps']:.6f}"
        )

        log(
            f"      HDG MAE       | "
            f"15E-H={base_mean['HDG_MAE_deg']:.6f} | "
            f"15E-VH={cand_mean['HDG_MAE_deg']:.6f}"
        )

        log(
            f"      AW vector      | "
            f"15E-H={base_mean['AppVector_RMSE_mps']:.6f} | "
            f"15E-VH={cand_mean['AppVector_RMSE_mps']:.6f}"
        )

        log(
            f"      AWS            | "
            f"15E-H={base_mean['AWS_RMSE_mps']:.6f} | "
            f"15E-VH={cand_mean['AWS_RMSE_mps']:.6f}"
        )

        log(
            f"      AWA MAE        | "
            f"15E-H={base_mean['AWA_MAE_deg']:.6f} | "
            f"15E-VH={cand_mean['AWA_MAE_deg']:.6f}"
        )

        log(
            f"      best epoch     = "
            f"{bundle['best_epoch']}"
        )

        bundle[
            "history"
        ].to_csv(
            out
            / f"15E_VH_seed_{seed}_training_history.csv",
            index=False,
            encoding="utf-8-sig",
        )

        np.savez_compressed(
            out
            / f"15E_VH_seed_{seed}_validation_predictions.npz",
            y_raw=val[
                "y"
            ].astype(
                np.float32
            ),
            frozen_truewind=frozen_wind.astype(
                np.float32
            ),
            baseline_15EH_vessel_hdg=baseline_vh.astype(
                np.float32
            ),
            joint_vessel_hdg=pred_vh.astype(
                np.float32
            ),
            gate_motion=bundle[
                "pred_val"
            ][
                "gate"
            ].astype(
                np.float32
            ),
            correction_motion_z=bundle[
                "pred_val"
            ][
                "correction_z"
            ].astype(
                np.float32
            ),
            residual_motion_z=bundle[
                "pred_val"
            ][
                "residual_z"
            ].astype(
                np.float32
            ),
            horizons_min=np.asarray(
                HORIZONS,
                dtype=np.int64,
            ),
        )

        checkpoint = {
            "seed": int(
                seed
            ),
            "model_state_dict": state_dict_cpu(
                bundle[
                    "model"
                ]
            ),
            "best_epoch": int(
                bundle[
                    "best_epoch"
                ]
            ),
            "best_validation_motion_score": float(
                bundle[
                    "best_score"
                ]
            ),
            "ridge_alpha": float(
                OLD_RIDGE_ALPHA
            ),
            "GRU_hidden": int(
                GRU_HIDDEN
            ),
            "GRU_layers": int(
                GRU_LAYERS
            ),
            "dense_hidden": int(
                DENSE_HIDDEN
            ),
            "dropout": float(
                DROPOUT
            ),
            "lambda_hdg": float(
                LAMBDA_HDG
            ),
            "lambda_app": float(
                LAMBDA_APP
            ),
            "lambda_res": float(
                LAMBDA_RES
            ),
            "parameters": int(
                bundle[
                    "parameters"
                ]
            ),
        }

        torch.save(
            checkpoint,
            out
            / f"15E_VH_seed_{seed}_joint_motion_model.pt",
        )

        del bundle

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Aggregate.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 5/6] Aggregating and saving results."
    )

    per_seed_df = pd.DataFrame(
        per_seed_rows
    )

    horizon_df = pd.concat(
        horizon_frames,
        ignore_index=True,
    )

    gate_df = pd.DataFrame(
        gate_rows
    )

    df_base = (
        per_seed_df.loc[
            per_seed_df[
                "model"
            ]
            == "15E-H"
        ]
        .reset_index(
            drop=True
        )
    )

    df_cand = (
        per_seed_df.loc[
            per_seed_df[
                "model"
            ]
            == "15E-VH"
        ]
        .reset_index(
            drop=True
        )
    )

    agg_base = aggregate_df(
        df_base
    )

    agg_base.insert(
        0,
        "model",
        "15E-H",
    )

    agg_cand = aggregate_df(
        df_cand
    )

    agg_cand.insert(
        0,
        "model",
        "15E-VH",
    )

    aggregate = pd.concat(
        [
            agg_base,
            agg_cand,
        ],
        ignore_index=True,
    )

    paired_metrics = [
        "Vessel_East_RMSE_mps",
        "Vessel_North_RMSE_mps",
        "Vessel_vector_RMSE_mps",
        "SOG_RMSE_mps",
        "HDG_MAE_deg",
        "HDG_RMSE_deg",
        "AppVector_RMSE_mps",
        "AWS_RMSE_mps",
        "AWA_MAE_deg",
        "AWA_RMSE_deg",
    ]

    paired = paired_effect(
        df_base,
        df_cand,
        paired_metrics,
    )

    per_seed_df.to_csv(
        out
        / "15E_VH_per_seed_validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    aggregate.to_csv(
        out
        / "15E_VH_validation_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired.to_csv(
        out
        / "15E_VH_paired_effect.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_df.to_csv(
        out
        / "15E_VH_per_horizon_all_seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gate_df.to_csv(
        out
        / "15E_VH_gate_diagnostics_all_seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Final print.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 6/6] Final five-seed comparison."
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

    final_metrics = [
        "Vessel_vector_RMSE_mps",
        "SOG_RMSE_mps",
        "HDG_MAE_deg",
        "HDG_RMSE_deg",
        "AppVector_RMSE_mps",
        "AWS_RMSE_mps",
        "AWA_MAE_deg",
        "AWA_RMSE_deg",
    ]

    log("")
    log("=" * 166)

    log(
        "15E-VH FINAL — JOINT VESSEL+HDG GATED RESIDUAL, "
        "SD1090 VALIDATION, FIVE SEEDS"
    )

    log("=" * 166)

    for metric in final_metrics:
        a = mstd(
            df_base,
            metric,
        )

        b = mstd(
            df_cand,
            metric,
        )

        log(
            f"{metric:28s} | "
            f"15E-H={a[0]:.6f} ± {a[1]:.6f} | "
            f"15E-VH={b[0]:.6f} ± {b[1]:.6f}"
        )

    log("")
    log(
        "[PAIRED 15E-H -> 15E-VH EFFECT]"
    )

    log(
        paired.to_string(
            index=False
        )
    )

    log("")
    log(
        f"15E-H total neural params = "
        f"{BASELINE_15EH_TOTAL_NEURAL_PARAMS:,}"
    )

    log(
        f"15E-VH motion params       = "
        f"{EXPECTED_MOTION_PARAMS:,}"
    )

    log(
        f"15E-VH total neural params = "
        f"{EXPECTED_TOTAL_NEURAL_PARAMS:,}"
    )

    log(
        "Parameter count is exactly matched to 15E-H."
    )

    report = {
        "experiment": "15E-VH",
        "internal_test_loaded": False,
        "baseline": "15E-H",
        "purpose": (
            "Integrate HDG residual learning into the same "
            "single-GRU64 gated motion branch as vessel velocity."
        ),
        "formula": (
            "Y_motion=[VE,VN,sinHDG,cosHDG]; "
            "Y_hat=Y_Ridge+Gate*DeltaY"
        ),
        "separate_HDG_head": False,
        "HDG_global_alpha": False,
        "truewind": (
            "frozen Stage15B predictions from completed 15E-H"
        ),
        "architecture": {
            "GRU_layers": GRU_LAYERS,
            "GRU_hidden": GRU_HIDDEN,
            "dense_hidden": DENSE_HIDDEN,
            "dropout": DROPOUT,
            "motion_outputs": MOTION_OUTPUTS,
            "motion_parameters": EXPECTED_MOTION_PARAMS,
            "total_neural_parameters": EXPECTED_TOTAL_NEURAL_PARAMS,
            "same_total_parameters_as_15E_H": True,
        },
        "loss": {
            "lambda_hdg": LAMBDA_HDG,
            "lambda_app": LAMBDA_APP,
            "lambda_res": LAMBDA_RES,
            "AWA_used_for_checkpoint_selection": False,
            "checkpoint_score": (
                "0.5*(VesselVector/RidgeVesselVector)"
                "+0.5*(HDG_MAE/RidgeHDG_MAE)"
            ),
        },
        "seeds": SEEDS,
        "paired_effect": paired.to_dict(
            orient="records"
        ),
    }

    save_json(
        out
        / "15E_VH_REPORT.json",
        report,
    )

    with (
        out
        / "15E_VH_REPORT.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "15E-VH — JOINT VESSEL+HDG GATED RESIDUAL\n"
        )

        f.write(
            "=" * 150
            + "\n\n"
        )

        f.write(
            "One shared single-GRU64 branch predicts "
            "[VE,VN,HDG_sin,HDG_cos] residuals and gates.\n"
        )

        f.write(
            "No separate HDG head. No HDG alpha.\n"
        )

        f.write(
            "Total neural parameters exactly match 15E-H: 49,538.\n\n"
        )

        for metric in final_metrics:
            a = mstd(
                df_base,
                metric,
            )

            b = mstd(
                df_cand,
                metric,
            )

            f.write(
                f"{metric:28s} | "
                f"15E-H={a[0]:.8f} ± {a[1]:.8f} | "
                f"15E-VH={b[0]:.8f} ± {b[1]:.8f}\n"
            )

        f.write(
            "\nPAIRED EFFECT\n"
        )

        f.write(
            "-" * 150
            + "\n"
        )

        f.write(
            paired.to_string(
                index=False
            )
        )

        f.write(
            "\n\nGATE DIAGNOSTICS\n"
        )

        f.write(
            "-" * 150
            + "\n"
        )

        f.write(
            gate_df.to_string(
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
