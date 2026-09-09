# -*- coding: utf-8 -*-
"""
05C_tune_joint_deep_baselines_pytorch.py

Controlled hyperparameter tuning for the frozen Saildrone joint wind-vessel
forecasting dataset.

Scientific protocol
-------------------
This script is designed to tune the conventional deep-learning baselines
without leaking information from the original test set.

Outer frozen dataset:
    train / validation / test

Hyperparameter search:
    ONLY the original train split is used.
    The original train split is further divided chronologically into:

        inner_train -> first (1 - inner_val_fraction)
        inner_val   -> last inner_val_fraction

    Every hyperparameter trial:
        - trains only on inner_train
        - early-stops only on inner_val
        - is ranked only by inner_val mean apparent-wind vector RMSE

Final refit after hyperparameters are selected:
    - selected hyperparameters are fixed
    - model trains on the FULL original train split
    - original validation is used only for final early stopping/checkpointing
    - original test is evaluated only after the final model is frozen

This gives a much cleaner protocol than repeatedly tuning directly against
the original validation/test periods.

Models
------
1. LSTM
2. GRU
3. CNN-LSTM

Search spaces
-------------
LSTM / GRU:
    recurrent_units  : 32, 64, 128
    recurrent_layers : 1, 2
    dropout          : 0.0, 0.1, 0.2
    learning_rate    : 3e-4, 1e-3

CNN-LSTM:
    recurrent_units  : 32, 64, 128
    recurrent_layers : 1, 2
    dropout          : 0.0, 0.1, 0.2
    learning_rate    : 3e-4, 1e-3
    cnn_filters      : 16, 32, 64
    cnn_kernel_size  : 3, 5
    cnn_layers       : 1, 2

The script uses a controlled random search without requiring Optuna.
The current 05B configuration is always included as an anchor trial.

Primary tuning objective
------------------------
Mean apparent-wind vector RMSE over horizons [1, 2, 3, 5, 10] min:

    A_E = U_w - V_s,E
    A_N = V_w - V_s,N

The search objective is intentionally aligned with the control-oriented
apparent-wind preview target rather than standardized joint MSE.

Training loss for conventional baselines
----------------------------------------
Standardized joint MSE over all 5 horizons x 6 target channels.

GPU
---
Automatically uses CUDA when available.
Supports:
    AMP
    TF32
    pinned memory
    non-blocking transfers
    gradient clipping
    cuDNN

Outputs
-------
hyperparameter_trials.csv
best_hyperparameters.json
final_retrain_summary.csv
tuned_metrics_per_horizon.csv
tuned_metrics_summary.csv
tuning_manifest.json
tuning_report.txt

models/
    best_LSTM.pt
    best_GRU.pt
    best_CNN_LSTM.pt

histories/
    final_history_LSTM.csv
    final_history_GRU.csv
    final_history_CNN_LSTM.csv

predictions/
    validation_predictions_LSTM.npz
    validation_predictions_GRU.npz
    validation_predictions_CNN_LSTM.npz
    test_predictions_LSTM.npz
    test_predictions_GRU.npz
    test_predictions_CNN_LSTM.npz

figures/
    01_search_best_score_by_trial.png/pdf
    02_validation_apparent_rmse.png/pdf
    03_test_apparent_rmse.png/pdf
    04_test_aws_rmse.png/pdf
    05_test_awa_mae.png/pdf
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import itertools
import json
import math
import platform
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0-physics-gru"
EXPECTED_TASK_NAME = "joint_wind_vessel_forecasting"

EXPECTED_FEATURE_NAMES = [
    "UWND_MEAN",
    "VWND_MEAN",
    "TEMP_AIR_MEAN",
    "RH_MEAN",
    "SOG",
    "COG_sin",
    "COG_cos",
    "HDG_sin",
    "HDG_cos",
    "WING_ANGLE_sin",
    "WING_ANGLE_cos",
]

EXPECTED_TARGET_NAMES = [
    "UWND_MEAN",
    "VWND_MEAN",
    "VESSEL_EAST_MPS",
    "VESSEL_NORTH_MPS",
    "HDG_sin",
    "HDG_cos",
]

MODEL_ORDER = [
    "Persistence",
    "Ridge",
    "LSTM",
    "GRU",
    "CNN-LSTM",
]

ANCHOR_CONFIG = {
    "recurrent_units": 64,
    "recurrent_layers": 1,
    "dropout": 0.1,
    "learning_rate": 1e-3,
    "cnn_filters": 32,
    "cnn_kernel_size": 3,
    "cnn_layers": 1,
}


# =============================================================================
# Basic helpers
# =============================================================================

def log(msg: str = "") -> None:
    print(msg, flush=True)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_scaler(
    scalers: dict,
    key: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    obj = scalers[key]
    return (
        list(obj["names"]),
        np.asarray(obj["mean"], dtype=np.float64),
        np.asarray(obj["std"], dtype=np.float64),
    )


def find_validation_flags(dataset_dir: Path) -> list[Path]:
    return sorted(
        dataset_dir.glob("validation*/VALIDATION_PASSED.flag")
    )


def load_npz_split(
    dataset_dir: Path,
    split_name: str,
) -> dict[str, np.ndarray]:
    path = dataset_dir / f"{split_name}.npz"
    require_file(path)

    required = [
        "X",
        "y",
        "y_raw",
        "apparent_earth_raw",
        "context_end_time_ns",
        "target_time_ns",
        "anchor_row_index",
    ]

    log(f"[LOAD] {path}")

    with np.load(path, allow_pickle=False) as npz:
        missing = [
            key for key in required
            if key not in npz.files
        ]
        if missing:
            raise RuntimeError(
                f"{path.name} missing arrays: {missing}"
            )

        return {
            key: npz[key]
            for key in required
        }


def inverse_joint_target(
    y_z: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    y_z = np.asarray(y_z, dtype=np.float64)
    return (
        y_z
        * target_std[None, None, :]
        + target_mean[None, None, :]
    ).astype(np.float32)


def recover_last_raw_features(
    X: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    return (
        X[:, -1, :].astype(np.float64)
        * feature_std[None, :]
        + feature_mean[None, :]
    )


def persistence_joint_prediction(
    X: np.ndarray,
    feature_names: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    n_horizons: int,
) -> np.ndarray:
    last_raw = recover_last_raw_features(
        X=X,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    idx = {
        name: feature_names.index(name)
        for name in [
            "UWND_MEAN",
            "VWND_MEAN",
            "SOG",
            "COG_sin",
            "COG_cos",
            "HDG_sin",
            "HDG_cos",
        ]
    }

    u = last_raw[:, idx["UWND_MEAN"]]
    v = last_raw[:, idx["VWND_MEAN"]]
    sog = last_raw[:, idx["SOG"]]
    cog_sin = last_raw[:, idx["COG_sin"]]
    cog_cos = last_raw[:, idx["COG_cos"]]
    hdg_sin = last_raw[:, idx["HDG_sin"]]
    hdg_cos = last_raw[:, idx["HDG_cos"]]

    state = np.column_stack(
        [
            u,
            v,
            sog * cog_sin,
            sog * cog_cos,
            hdg_sin,
            hdg_cos,
        ]
    )

    return np.repeat(
        state[:, None, :],
        repeats=n_horizons,
        axis=1,
    ).astype(np.float32)


# =============================================================================
# Physics / metrics
# =============================================================================

def wrap_deg(angle_deg: np.ndarray) -> np.ndarray:
    return (angle_deg + 180.0) % 360.0 - 180.0


def vector_speed(
    east: np.ndarray,
    north: np.ndarray,
) -> np.ndarray:
    return np.sqrt(
        east.astype(np.float64) ** 2
        + north.astype(np.float64) ** 2
    )


def direction_from_uv_deg(
    u_east: np.ndarray,
    v_north: np.ndarray,
) -> np.ndarray:
    """
    Meteorological wind-from direction:
        0° = from North
        90° = from East
    """
    ang = np.degrees(
        np.arctan2(
            -u_east.astype(np.float64),
            -v_north.astype(np.float64),
        )
    )
    return (ang + 360.0) % 360.0


def cog_from_en_deg(
    east: np.ndarray,
    north: np.ndarray,
) -> np.ndarray:
    ang = np.degrees(
        np.arctan2(
            east.astype(np.float64),
            north.astype(np.float64),
        )
    )
    return (ang + 360.0) % 360.0


def heading_from_sincos_deg(
    hdg_sin: np.ndarray,
    hdg_cos: np.ndarray,
) -> np.ndarray:
    ang = np.degrees(
        np.arctan2(
            hdg_sin.astype(np.float64),
            hdg_cos.astype(np.float64),
        )
    )
    return (ang + 360.0) % 360.0


def apparent_earth_from_joint(
    joint_raw: np.ndarray,
) -> np.ndarray:
    app = np.empty(
        joint_raw.shape[:-1] + (2,),
        dtype=np.float64,
    )
    app[..., 0] = (
        joint_raw[..., 0].astype(np.float64)
        - joint_raw[..., 2].astype(np.float64)
    )
    app[..., 1] = (
        joint_raw[..., 1].astype(np.float64)
        - joint_raw[..., 3].astype(np.float64)
    )
    return app


def apparent_wind_angle_deg(
    apparent_east: np.ndarray,
    apparent_north: np.ndarray,
    hdg_sin: np.ndarray,
    hdg_cos: np.ndarray,
) -> np.ndarray:
    """
    Body-frame apparent wind-from angle:
        0°      : from directly ahead
        +90°    : from starboard
        -90°    : from port
        +/-180° : from astern
    """
    ae = apparent_east.astype(np.float64)
    an = apparent_north.astype(np.float64)
    s = hdg_sin.astype(np.float64)
    c = hdg_cos.astype(np.float64)

    norm = np.sqrt(s ** 2 + c ** 2)
    safe = norm > 1e-12
    s = np.where(safe, s / norm, 0.0)
    c = np.where(safe, c / norm, 1.0)

    a_forward = ae * s + an * c
    a_starboard = ae * c - an * s

    wind_from_forward = -a_forward
    wind_from_starboard = -a_starboard

    return wrap_deg(
        np.degrees(
            np.arctan2(
                wind_from_starboard,
                wind_from_forward,
            )
        )
    )


def safe_r2(
    truth: np.ndarray,
    pred: np.ndarray,
) -> float:
    truth = truth.astype(np.float64)
    pred = pred.astype(np.float64)
    ss_res = float(np.sum((truth - pred) ** 2))
    ss_tot = float(
        np.sum(
            (truth - np.mean(truth)) ** 2
        )
    )
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def masked_circular_metrics(
    pred_deg: np.ndarray,
    true_deg: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float, int]:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(pred_deg)
        & np.isfinite(true_deg)
    )
    n = int(valid.sum())

    if n == 0:
        return float("nan"), float("nan"), 0

    err = wrap_deg(
        pred_deg[valid] - true_deg[valid]
    )

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return mae, rmse, n


def unit_norm_error(
    sin_values: np.ndarray,
    cos_values: np.ndarray,
) -> float:
    norm = np.sqrt(
        sin_values.astype(np.float64) ** 2
        + cos_values.astype(np.float64) ** 2
    )
    return float(
        np.mean(np.abs(norm - 1.0))
    )


def mean_apparent_vector_rmse(
    truth_joint_raw: np.ndarray,
    pred_joint_raw: np.ndarray,
) -> tuple[float, np.ndarray]:
    true_app = apparent_earth_from_joint(
        truth_joint_raw
    )
    pred_app = apparent_earth_from_joint(
        pred_joint_raw
    )
    err = pred_app - true_app
    per_horizon = np.sqrt(
        np.mean(
            err[..., 0] ** 2
            + err[..., 1] ** 2,
            axis=0,
        )
    )
    return (
        float(np.mean(per_horizon)),
        np.asarray(per_horizon, dtype=np.float64),
    )


def evaluate_model(
    truth_joint: np.ndarray,
    pred_joint: np.ndarray,
    truth_apparent_ref: np.ndarray,
    horizons: tuple[int, ...],
    split_name: str,
    model_name: str,
    direction_min_speed: float,
    cog_min_speed: float,
    persistence_joint: np.ndarray | None = None,
) -> pd.DataFrame:
    if truth_joint.shape != pred_joint.shape:
        raise ValueError(
            f"truth/pred shape mismatch: "
            f"{truth_joint.shape} vs {pred_joint.shape}"
        )

    pred_app_all = apparent_earth_from_joint(pred_joint)
    persistence_app_all = (
        apparent_earth_from_joint(persistence_joint)
        if persistence_joint is not None
        else None
    )

    rows = []

    for j, horizon in enumerate(horizons):
        truth = truth_joint[:, j, :].astype(
            np.float64, copy=False
        )
        pred = pred_joint[:, j, :].astype(
            np.float64, copy=False
        )

        true_wind = truth[:, 0:2]
        pred_wind = pred[:, 0:2]
        wind_err = pred_wind - true_wind
        du = wind_err[:, 0]
        dv = wind_err[:, 1]

        wind_u_rmse = float(
            np.sqrt(np.mean(du ** 2))
        )
        wind_v_rmse = float(
            np.sqrt(np.mean(dv ** 2))
        )
        wind_vector_rmse = float(
            np.sqrt(np.mean(du ** 2 + dv ** 2))
        )
        wind_vector_mae = float(
            np.mean(np.sqrt(du ** 2 + dv ** 2))
        )

        true_ws = vector_speed(
            true_wind[:, 0],
            true_wind[:, 1],
        )
        pred_ws = vector_speed(
            pred_wind[:, 0],
            pred_wind[:, 1],
        )
        ws_err = pred_ws - true_ws
        wind_speed_rmse = float(
            np.sqrt(np.mean(ws_err ** 2))
        )
        wind_speed_mae = float(
            np.mean(np.abs(ws_err))
        )

        true_wd = direction_from_uv_deg(
            true_wind[:, 0],
            true_wind[:, 1],
        )
        pred_wd = direction_from_uv_deg(
            pred_wind[:, 0],
            pred_wind[:, 1],
        )
        (
            wind_dir_mae,
            wind_dir_rmse,
            n_wind_dir,
        ) = masked_circular_metrics(
            pred_wd,
            true_wd,
            true_ws >= direction_min_speed,
        )

        true_vessel = truth[:, 2:4]
        pred_vessel = pred[:, 2:4]
        vessel_err = pred_vessel - true_vessel
        de = vessel_err[:, 0]
        dn = vessel_err[:, 1]

        vessel_vector_rmse = float(
            np.sqrt(np.mean(de ** 2 + dn ** 2))
        )
        vessel_vector_mae = float(
            np.mean(np.sqrt(de ** 2 + dn ** 2))
        )

        true_sog = vector_speed(
            true_vessel[:, 0],
            true_vessel[:, 1],
        )
        pred_sog = vector_speed(
            pred_vessel[:, 0],
            pred_vessel[:, 1],
        )
        sog_err = pred_sog - true_sog
        sog_rmse = float(
            np.sqrt(np.mean(sog_err ** 2))
        )
        sog_mae = float(
            np.mean(np.abs(sog_err))
        )

        true_cog = cog_from_en_deg(
            true_vessel[:, 0],
            true_vessel[:, 1],
        )
        pred_cog = cog_from_en_deg(
            pred_vessel[:, 0],
            pred_vessel[:, 1],
        )
        cog_mae, cog_rmse, n_cog = (
            masked_circular_metrics(
                pred_cog,
                true_cog,
                true_sog >= cog_min_speed,
            )
        )

        true_hdg_sin = truth[:, 4]
        true_hdg_cos = truth[:, 5]
        pred_hdg_sin = pred[:, 4]
        pred_hdg_cos = pred[:, 5]

        true_hdg = heading_from_sincos_deg(
            true_hdg_sin,
            true_hdg_cos,
        )
        pred_hdg = heading_from_sincos_deg(
            pred_hdg_sin,
            pred_hdg_cos,
        )
        hdg_mae, hdg_rmse, n_hdg = (
            masked_circular_metrics(
                pred_hdg,
                true_hdg,
                np.ones(
                    len(true_hdg),
                    dtype=bool,
                ),
            )
        )
        hdg_unit_norm_mae = unit_norm_error(
            pred_hdg_sin,
            pred_hdg_cos,
        )

        true_app = truth_apparent_ref[
            :, j, :
        ].astype(np.float64, copy=False)
        pred_app = pred_app_all[:, j, :]
        app_err = pred_app - true_app
        app_de = app_err[:, 0]
        app_dn = app_err[:, 1]

        apparent_vector_rmse = float(
            np.sqrt(
                np.mean(
                    app_de ** 2
                    + app_dn ** 2
                )
            )
        )
        apparent_vector_mae = float(
            np.mean(
                np.sqrt(
                    app_de ** 2
                    + app_dn ** 2
                )
            )
        )

        true_aws = vector_speed(
            true_app[:, 0],
            true_app[:, 1],
        )
        pred_aws = vector_speed(
            pred_app[:, 0],
            pred_app[:, 1],
        )
        aws_err = pred_aws - true_aws
        aws_rmse = float(
            np.sqrt(np.mean(aws_err ** 2))
        )
        aws_mae = float(
            np.mean(np.abs(aws_err))
        )

        true_awa = apparent_wind_angle_deg(
            true_app[:, 0],
            true_app[:, 1],
            true_hdg_sin,
            true_hdg_cos,
        )
        pred_awa = apparent_wind_angle_deg(
            pred_app[:, 0],
            pred_app[:, 1],
            pred_hdg_sin,
            pred_hdg_cos,
        )
        awa_mae, awa_rmse, n_awa = (
            masked_circular_metrics(
                pred_awa,
                true_awa,
                true_aws >= direction_min_speed,
            )
        )

        wind_skill = (
            0.0
            if model_name.lower() == "persistence"
            else float("nan")
        )
        vessel_skill = (
            0.0
            if model_name.lower() == "persistence"
            else float("nan")
        )
        app_skill = (
            0.0
            if model_name.lower() == "persistence"
            else float("nan")
        )
        aws_skill = (
            0.0
            if model_name.lower() == "persistence"
            else float("nan")
        )

        if persistence_joint is not None:
            ref = persistence_joint[
                :, j, :
            ].astype(np.float64, copy=False)

            ref_wind_err = (
                ref[:, 0:2] - true_wind
            )
            ref_wind_rmse = float(
                np.sqrt(
                    np.mean(
                        ref_wind_err[:, 0] ** 2
                        + ref_wind_err[:, 1] ** 2
                    )
                )
            )

            ref_vessel_err = (
                ref[:, 2:4] - true_vessel
            )
            ref_vessel_rmse = float(
                np.sqrt(
                    np.mean(
                        ref_vessel_err[:, 0] ** 2
                        + ref_vessel_err[:, 1] ** 2
                    )
                )
            )

            ref_app = persistence_app_all[:, j, :]
            ref_app_err = ref_app - true_app
            ref_app_rmse = float(
                np.sqrt(
                    np.mean(
                        ref_app_err[:, 0] ** 2
                        + ref_app_err[:, 1] ** 2
                    )
                )
            )

            ref_aws = vector_speed(
                ref_app[:, 0],
                ref_app[:, 1],
            )
            ref_aws_rmse = float(
                np.sqrt(
                    np.mean(
                        (ref_aws - true_aws) ** 2
                    )
                )
            )

            if ref_wind_rmse > 0:
                wind_skill = (
                    1.0
                    - wind_vector_rmse
                    / ref_wind_rmse
                )
            if ref_vessel_rmse > 0:
                vessel_skill = (
                    1.0
                    - vessel_vector_rmse
                    / ref_vessel_rmse
                )
            if ref_app_rmse > 0:
                app_skill = (
                    1.0
                    - apparent_vector_rmse
                    / ref_app_rmse
                )
            if ref_aws_rmse > 0:
                aws_skill = (
                    1.0
                    - aws_rmse
                    / ref_aws_rmse
                )

        rows.append({
            "split": split_name,
            "model": model_name,
            "horizon_min": int(horizon),
            "n_samples": int(len(truth)),

            "wind_U_RMSE_mps": wind_u_rmse,
            "wind_V_RMSE_mps": wind_v_rmse,
            "wind_vector_RMSE_mps": wind_vector_rmse,
            "wind_vector_MAE_mps": wind_vector_mae,
            "wind_speed_RMSE_mps": wind_speed_rmse,
            "wind_speed_MAE_mps": wind_speed_mae,
            "wind_direction_MAE_deg": wind_dir_mae,
            "wind_direction_RMSE_deg": wind_dir_rmse,
            "wind_direction_n": n_wind_dir,
            "wind_R2_U": safe_r2(
                true_wind[:, 0],
                pred_wind[:, 0],
            ),
            "wind_R2_V": safe_r2(
                true_wind[:, 1],
                pred_wind[:, 1],
            ),
            "wind_skill_vs_persistence": wind_skill,

            "vessel_vector_RMSE_mps": vessel_vector_rmse,
            "vessel_vector_MAE_mps": vessel_vector_mae,
            "SOG_RMSE_mps": sog_rmse,
            "SOG_MAE_mps": sog_mae,
            "COG_MAE_deg": cog_mae,
            "COG_RMSE_deg": cog_rmse,
            "COG_n": n_cog,
            "vessel_skill_vs_persistence": vessel_skill,

            "HDG_MAE_deg": hdg_mae,
            "HDG_RMSE_deg": hdg_rmse,
            "HDG_n": n_hdg,
            "HDG_pred_unit_norm_MAE": hdg_unit_norm_mae,

            "apparent_vector_RMSE_mps": apparent_vector_rmse,
            "apparent_vector_MAE_mps": apparent_vector_mae,
            "AWS_RMSE_mps": aws_rmse,
            "AWS_MAE_mps": aws_mae,
            "AWA_MAE_deg": awa_mae,
            "AWA_RMSE_deg": awa_rmse,
            "AWA_n": n_awa,
            "apparent_skill_vs_persistence": app_skill,
            "AWS_skill_vs_persistence": aws_skill,
        })

    return pd.DataFrame(rows)


def summarize_metrics(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        "wind_vector_RMSE_mps",
        "wind_speed_RMSE_mps",
        "wind_direction_MAE_deg",
        "vessel_vector_RMSE_mps",
        "SOG_RMSE_mps",
        "COG_MAE_deg",
        "HDG_MAE_deg",
        "apparent_vector_RMSE_mps",
        "AWS_RMSE_mps",
        "AWA_MAE_deg",
        "wind_skill_vs_persistence",
        "vessel_skill_vs_persistence",
        "apparent_skill_vs_persistence",
        "AWS_skill_vs_persistence",
    ]

    rows = []

    for (
        split_name,
        model_name,
    ), grp in metrics.groupby(
        ["split", "model"],
        sort=False,
    ):
        row = {
            "split": split_name,
            "model": model_name,
            "n_horizons": int(len(grp)),
        }

        for col in cols:
            row[f"mean_{col}"] = float(
                grp[col].mean()
            )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# PyTorch
# =============================================================================

def import_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import (
            DataLoader,
            TensorDataset,
        )
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required in the active environment."
        ) from exc

    return torch, nn, DataLoader, TensorDataset


def set_global_seed(
    torch,
    seed: int,
    deterministic: bool,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(
                True,
                warn_only=True,
            )
        except Exception:
            pass

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


def configure_tf32(
    torch,
    enabled: bool,
) -> None:
    if not torch.cuda.is_available():
        return

    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(
            enabled
        )
    except Exception:
        pass

    try:
        torch.backends.cudnn.allow_tf32 = bool(
            enabled
        )
    except Exception:
        pass


def make_grad_scaler(
    torch,
    enabled: bool,
):
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )
    except Exception:
        return torch.cuda.amp.GradScaler(
            enabled=enabled
        )


class AutocastContext:
    def __init__(
        self,
        torch,
        enabled: bool,
        device_type: str,
    ):
        self.torch = torch
        self.enabled = bool(enabled)
        self.device_type = device_type
        self.ctx = None

    def __enter__(self):
        if self.device_type != "cuda":
            self.ctx = self.torch.autocast(
                device_type="cpu",
                enabled=False,
            )
            return self.ctx.__enter__()

        try:
            self.ctx = self.torch.amp.autocast(
                device_type="cuda",
                dtype=self.torch.float16,
                enabled=self.enabled,
            )
        except Exception:
            self.ctx = (
                self.torch.cuda.amp.autocast(
                    dtype=self.torch.float16,
                    enabled=self.enabled,
                )
            )

        return self.ctx.__enter__()

    def __exit__(
        self,
        exc_type,
        exc_value,
        tb,
    ):
        return self.ctx.__exit__(
            exc_type,
            exc_value,
            tb,
        )


def log_device_info(
    torch,
    device,
    amp_enabled: bool,
    tf32_enabled: bool,
) -> dict:
    info = {
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": bool(
            torch.cuda.is_available()
        ),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if hasattr(torch.backends, "cudnn")
            else None
        ),
        "amp_enabled": bool(amp_enabled),
        "tf32_enabled": bool(tf32_enabled),
    }

    log("")
    log("=" * 100)
    log("PYTORCH DEVICE DIAGNOSTICS")
    log("=" * 100)
    log(f"torch version      : {torch.__version__}")
    log(f"selected device    : {device}")
    log(f"CUDA available     : {torch.cuda.is_available()}")
    log(f"torch CUDA version : {torch.version.cuda}")
    log(
        f"cuDNN version      : "
        f"{info['cudnn_version']}"
    )

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(
            device
        )
        info.update({
            "gpu_name": torch.cuda.get_device_name(
                device
            ),
            "gpu_total_vram_gb": float(
                props.total_memory / (1024 ** 3)
            ),
            "gpu_compute_capability": (
                f"{props.major}.{props.minor}"
            ),
        })

        log(f"GPU name           : {info['gpu_name']}")
        log(
            f"GPU VRAM           : "
            f"{info['gpu_total_vram_gb']:.2f} GB"
        )
        log(
            f"compute capability : "
            f"{info['gpu_compute_capability']}"
        )
        log(f"AMP enabled        : {amp_enabled}")
        log(f"TF32 enabled       : {tf32_enabled}")
        log(
            "[OK] Hyperparameter tuning will run on CUDA GPU."
        )
    else:
        log(
            "[WARNING] CUDA unavailable; search will run on CPU."
        )

    log("=" * 100)
    log("")
    return info


# =============================================================================
# Model definitions
# =============================================================================

def build_model_class(
    torch,
    nn,
):
    class CausalConv1d(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
        ):
            super().__init__()
            self.left_padding = (
                int(kernel_size) - 1
            )
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=self.left_padding,
            )

        def forward(self, x):
            y = self.conv(x)
            if self.left_padding > 0:
                y = y[..., :-self.left_padding]
            return y

    class JointSequenceModel(nn.Module):
        def __init__(
            self,
            model_name: str,
            n_features: int,
            n_horizons: int,
            n_targets: int,
            recurrent_units: int,
            recurrent_layers: int,
            dense_units: int,
            dropout: float,
            cnn_filters: int,
            cnn_kernel_size: int,
            cnn_layers: int,
        ):
            super().__init__()

            self.model_name = model_name
            self.n_horizons = n_horizons
            self.n_targets = n_targets

            recurrent_dropout = (
                float(dropout)
                if recurrent_layers > 1
                else 0.0
            )

            if model_name == "LSTM":
                self.conv_stack = None
                self.encoder = nn.LSTM(
                    input_size=n_features,
                    hidden_size=recurrent_units,
                    num_layers=recurrent_layers,
                    batch_first=True,
                    dropout=recurrent_dropout,
                )

            elif model_name == "GRU":
                self.conv_stack = None
                self.encoder = nn.GRU(
                    input_size=n_features,
                    hidden_size=recurrent_units,
                    num_layers=recurrent_layers,
                    batch_first=True,
                    dropout=recurrent_dropout,
                )

            elif model_name == "CNN-LSTM":
                conv_modules = []
                in_ch = n_features

                for _ in range(cnn_layers):
                    conv_modules.extend([
                        CausalConv1d(
                            in_channels=in_ch,
                            out_channels=cnn_filters,
                            kernel_size=cnn_kernel_size,
                        ),
                        nn.ReLU(),
                    ])
                    in_ch = cnn_filters

                self.conv_stack = nn.Sequential(
                    *conv_modules
                )

                self.encoder = nn.LSTM(
                    input_size=cnn_filters,
                    hidden_size=recurrent_units,
                    num_layers=recurrent_layers,
                    batch_first=True,
                    dropout=recurrent_dropout,
                )

            else:
                raise ValueError(
                    f"Unsupported model: {model_name}"
                )

            self.dropout = nn.Dropout(
                float(dropout)
            )
            self.fc1 = nn.Linear(
                recurrent_units,
                dense_units,
            )
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(
                dense_units,
                n_horizons * n_targets,
            )

        def forward(self, x):
            # x: [B, L, F]
            if self.model_name == "CNN-LSTM":
                x = x.transpose(1, 2)
                x = self.conv_stack(x)
                x = x.transpose(1, 2)

            seq, _ = self.encoder(x)
            x = seq[:, -1, :]
            x = self.dropout(x)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)

            return x.reshape(
                x.shape[0],
                self.n_horizons,
                self.n_targets,
            )

    return JointSequenceModel


def count_parameters(model) -> int:
    return int(
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )


# =============================================================================
# Dataset / loaders
# =============================================================================

def make_tensor_dataset(
    torch,
    TensorDataset,
    X: np.ndarray,
    y: np.ndarray,
):
    return TensorDataset(
        torch.from_numpy(
            np.ascontiguousarray(
                X,
                dtype=np.float32,
            )
        ),
        torch.from_numpy(
            np.ascontiguousarray(
                y,
                dtype=np.float32,
            )
        ),
    )


def make_loader(
    DataLoader,
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
):
    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": False,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = bool(
            persistent_workers
        )
        kwargs["prefetch_factor"] = int(
            prefetch_factor
        )

    return DataLoader(**kwargs)


def chronological_inner_split(
    train: dict[str, np.ndarray],
    inner_val_fraction: float,
) -> dict[str, np.ndarray]:
    n = int(train["X"].shape[0])

    n_inner_val = int(
        round(n * inner_val_fraction)
    )
    n_inner_val = max(
        1,
        min(n - 1, n_inner_val),
    )
    n_inner_train = n - n_inner_val

    return {
        "inner_train_X": train["X"][
            :n_inner_train
        ],
        "inner_train_y": train["y"][
            :n_inner_train
        ],
        "inner_val_X": train["X"][
            n_inner_train:
        ],
        "inner_val_y": train["y"][
            n_inner_train:
        ],
        "inner_val_y_raw": train["y_raw"][
            n_inner_train:
        ],
        "n_inner_train": n_inner_train,
        "n_inner_val": n_inner_val,
        "split_index": n_inner_train,
        "inner_train_last_context_ns": int(
            train["context_end_time_ns"][
                n_inner_train - 1
            ]
        ),
        "inner_val_first_context_ns": int(
            train["context_end_time_ns"][
                n_inner_train
            ]
        ),
    }


# =============================================================================
# Training / inference
# =============================================================================

def run_eval(
    torch,
    model,
    loader,
    criterion,
    device,
    amp_enabled: bool,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> tuple[
    float,
    float,
    np.ndarray,
]:
    model.eval()

    total_loss = 0.0
    total_n = 0
    pred_chunks = []
    truth_chunks = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(
                device,
                non_blocking=True,
            )
            y_batch = y_batch.to(
                device,
                non_blocking=True,
            )

            with AutocastContext(
                torch,
                amp_enabled,
                device.type,
            ):
                pred = model(X_batch)
                loss = criterion(
                    pred,
                    y_batch,
                )

            n = int(X_batch.shape[0])
            total_loss += float(
                loss.detach().item()
            ) * n
            total_n += n

            pred_chunks.append(
                pred.detach()
                .float()
                .cpu()
                .numpy()
            )
            truth_chunks.append(
                y_batch.detach()
                .float()
                .cpu()
                .numpy()
            )

    pred_z = np.concatenate(
        pred_chunks,
        axis=0,
    )
    truth_z = np.concatenate(
        truth_chunks,
        axis=0,
    )

    pred_raw = inverse_joint_target(
        pred_z,
        target_mean,
        target_std,
    )
    truth_raw = inverse_joint_target(
        truth_z,
        target_mean,
        target_std,
    )

    aw_score, _ = mean_apparent_vector_rmse(
        truth_raw,
        pred_raw,
    )

    return (
        float(total_loss / max(total_n, 1)),
        float(aw_score),
        pred_z,
    )


def predict_loader(
    torch,
    model,
    loader,
    device,
    amp_enabled: bool,
) -> np.ndarray:
    model.eval()
    chunks = []

    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(
                device,
                non_blocking=True,
            )

            with AutocastContext(
                torch,
                amp_enabled,
                device.type,
            ):
                pred = model(X_batch)

            chunks.append(
                pred.detach()
                .float()
                .cpu()
                .numpy()
            )

    return np.concatenate(
        chunks,
        axis=0,
    )


def train_with_early_stopping(
    torch,
    nn,
    model,
    train_loader,
    val_loader,
    device,
    amp_enabled: bool,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    min_delta: float,
    gradient_clip_norm: float,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    checkpoint_path: Path,
    verbose_epochs: bool,
) -> dict:
    criterion = nn.MSELoss(
        reduction="mean"
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(
                2,
                patience // 3,
            ),
            min_lr=1e-5,
        )
    )
    scaler = make_grad_scaler(
        torch,
        amp_enabled,
    )

    best_score = float("inf")
    best_epoch = 0
    wait = 0
    rows = []
    t0 = time.perf_counter()

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        epoch_t0 = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(
                device,
                non_blocking=True,
            )
            y_batch = y_batch.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with AutocastContext(
                torch,
                amp_enabled,
                device.type,
            ):
                pred = model(X_batch)
                loss = criterion(
                    pred,
                    y_batch,
                )

            if amp_enabled:
                scaler.scale(loss).backward()

                if gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip_norm,
                    )

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip_norm,
                    )

                optimizer.step()

            n = int(X_batch.shape[0])
            train_loss_sum += (
                float(loss.detach().item())
                * n
            )
            train_n += n

        train_loss = (
            train_loss_sum
            / max(train_n, 1)
        )

        (
            val_loss,
            val_aw_rmse,
            _,
        ) = run_eval(
            torch=torch,
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            target_mean=target_mean,
            target_std=target_std,
        )

        lr_before_step = float(
            optimizer.param_groups[0]["lr"]
        )
        scheduler.step(val_loss)

        improved = (
            val_aw_rmse
            < best_score - min_delta
        )

        if improved:
            best_score = float(
                val_aw_rmse
            )
            best_epoch = int(epoch)
            wait = 0

            torch.save(
                {
                    "epoch": best_epoch,
                    "best_score": best_score,
                    "state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
            status = "[BEST]"
        else:
            wait += 1
            status = (
                f"wait={wait}/{patience}"
            )

        epoch_seconds = float(
            time.perf_counter()
            - epoch_t0
        )

        rows.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_apparent_vector_RMSE_mps": float(
                val_aw_rmse
            ),
            "learning_rate": lr_before_step,
            "epoch_seconds": epoch_seconds,
        })

        if verbose_epochs:
            log(
                f"Epoch {epoch:03d}/{max_epochs} | "
                f"loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"val_AW_RMSE={val_aw_rmse:.6f} m/s | "
                f"lr={lr_before_step:.3e} | "
                f"{epoch_seconds:.2f}s | "
                f"{status}"
            )

        if wait >= patience:
            break

    elapsed = float(
        time.perf_counter()
        - t0
    )

    require_file(checkpoint_path)

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(
        ckpt["state_dict"]
    )

    return {
        "model": model,
        "history": pd.DataFrame(rows),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "training_seconds": elapsed,
        "epochs_ran": int(len(rows)),
    }


# =============================================================================
# Search space
# =============================================================================

def make_full_search_space(
    model_name: str,
) -> list[dict]:
    base = {
        "recurrent_units": [32, 64, 128],
        "recurrent_layers": [1, 2],
        "dropout": [0.0, 0.1, 0.2],
        "learning_rate": [3e-4, 1e-3],
    }

    if model_name in ("LSTM", "GRU"):
        keys = list(base.keys())
        values = [base[k] for k in keys]
        configs = [
            dict(zip(keys, combo))
            for combo in itertools.product(*values)
        ]

        for cfg in configs:
            cfg.update({
                "cnn_filters": 32,
                "cnn_kernel_size": 3,
                "cnn_layers": 1,
            })

        return configs

    if model_name == "CNN-LSTM":
        space = {
            **base,
            "cnn_filters": [16, 32, 64],
            "cnn_kernel_size": [3, 5],
            "cnn_layers": [1, 2],
        }
        keys = list(space.keys())
        values = [space[k] for k in keys]

        return [
            dict(zip(keys, combo))
            for combo in itertools.product(*values)
        ]

    raise ValueError(model_name)


def canonical_config_tuple(
    cfg: dict,
) -> tuple:
    return (
        int(cfg["recurrent_units"]),
        int(cfg["recurrent_layers"]),
        float(cfg["dropout"]),
        float(cfg["learning_rate"]),
        int(cfg["cnn_filters"]),
        int(cfg["cnn_kernel_size"]),
        int(cfg["cnn_layers"]),
    )


def select_trial_configs(
    model_name: str,
    n_trials: int,
    seed: int,
) -> list[dict]:
    full = make_full_search_space(
        model_name
    )

    anchor = dict(ANCHOR_CONFIG)
    anchor_tuple = canonical_config_tuple(
        anchor
    )

    by_tuple = {
        canonical_config_tuple(cfg): cfg
        for cfg in full
    }

    if anchor_tuple not in by_tuple:
        raise RuntimeError(
            "Anchor configuration is not in search space."
        )

    anchor_cfg = dict(
        by_tuple[anchor_tuple]
    )

    remaining = [
        dict(cfg)
        for cfg in full
        if canonical_config_tuple(cfg)
        != anchor_tuple
    ]

    rng = random.Random(
        seed
    )
    rng.shuffle(
        remaining
    )

    n_trials = max(
        1,
        min(n_trials, len(full)),
    )

    chosen = [
        anchor_cfg
    ] + remaining[
        : max(0, n_trials - 1)
    ]

    return chosen


# =============================================================================
# Save predictions
# =============================================================================

def save_predictions(
    path: Path,
    model_name: str,
    truth_joint: np.ndarray,
    pred_joint: np.ndarray,
    truth_apparent_ref: np.ndarray,
    context_end_time_ns: np.ndarray,
    target_time_ns: np.ndarray,
    horizons: tuple[int, ...],
) -> None:
    np.savez_compressed(
        path,
        model_name=np.asarray(model_name),
        truth_joint=np.asarray(
            truth_joint,
            dtype=np.float32,
        ),
        pred_joint=np.asarray(
            pred_joint,
            dtype=np.float32,
        ),
        truth_apparent_reference_en=np.asarray(
            truth_apparent_ref,
            dtype=np.float32,
        ),
        context_end_time_ns=np.asarray(
            context_end_time_ns,
            dtype=np.int64,
        ),
        target_time_ns=np.asarray(
            target_time_ns,
            dtype=np.int64,
        ),
        horizons_min=np.asarray(
            horizons,
            dtype=np.int64,
        ),
    )


# =============================================================================
# SCI plotting
# =============================================================================

def load_sci_style(
    script_dir: Path,
):
    path = script_dir / "sci_plot_style.py"

    if not path.exists():
        log(
            f"[PLOT WARNING] "
            f"sci_plot_style.py not found: {path}"
        )
        return None

    spec = importlib.util.spec_from_file_location(
        "sci_plot_style",
        str(path),
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(
        spec
    )
    sys.modules["sci_plot_style"] = module
    spec.loader.exec_module(module)

    if hasattr(module, "apply_sci_style"):
        try:
            selected = module.apply_sci_style(
                base_font_size=8.5
            )
        except TypeError:
            try:
                selected = module.apply_sci_style(
                    font_size=8.5
                )
            except TypeError:
                selected = module.apply_sci_style()

        log(
            f"[PLOT] SCI style applied; "
            f"selected_font={selected}"
        )

    return module


def make_plots(
    trials: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    style = load_sci_style(
        Path(__file__).resolve().parent
    )

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    def finish(ax):
        if (
            style is not None
            and hasattr(style, "clean_axis")
        ):
            style.clean_axis(
                ax,
                grid=True,
            )
        else:
            ax.tick_params(
                which="both",
                direction="in",
                top=True,
                right=True,
            )
            ax.grid(True, alpha=0.25)

    def save(fig, name):
        base = fig_dir / name

        if (
            style is not None
            and hasattr(style, "save_figure")
        ):
            style.save_figure(
                fig,
                base,
            )
        else:
            fig.savefig(
                base.with_suffix(".png"),
                dpi=600,
                bbox_inches="tight",
            )
            fig.savefig(
                base.with_suffix(".pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

    # Search convergence
    fig, ax = plt.subplots(
        figsize=(4.5, 3.1)
    )

    for model_name in [
        "LSTM",
        "GRU",
        "CNN-LSTM",
    ]:
        grp = trials.loc[
            trials["model"] == model_name
        ].sort_values("trial_id")

        if grp.empty:
            continue

        scores = grp[
            "inner_val_best_apparent_vector_RMSE_mps"
        ].to_numpy(
            dtype=float
        )
        best_so_far = np.minimum.accumulate(
            scores
        )

        ax.plot(
            grp["trial_id"],
            best_so_far,
            marker="o",
            label=model_name,
        )

    ax.set_xlabel("Trial")
    ax.set_ylabel(
        r"Best inner-validation AW RMSE (m s$^{-1}$)"
    )
    ax.legend(loc="best")
    finish(ax)
    save(
        fig,
        "01_search_best_score_by_trial",
    )

    def metric_curve(
        split_name: str,
        metric_col: str,
        ylabel: str,
        filename: str,
    ):
        fig, ax = plt.subplots(
            figsize=(4.5, 3.1)
        )

        subset = metrics.loc[
            metrics["split"] == split_name
        ]

        for model_name in MODEL_ORDER:
            grp = subset.loc[
                subset["model"] == model_name
            ].sort_values("horizon_min")

            if grp.empty:
                continue

            ax.plot(
                grp["horizon_min"],
                grp[metric_col],
                marker="o",
                label=model_name,
            )

        ax.set_xlabel(
            "Forecast horizon (min)"
        )
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 2, 3, 5, 10])
        ax.legend(
            loc="best",
            ncol=2,
        )
        finish(ax)
        save(fig, filename)

    metric_curve(
        "validation",
        "apparent_vector_RMSE_mps",
        r"Apparent-wind vector RMSE (m s$^{-1}$)",
        "02_validation_apparent_rmse",
    )
    metric_curve(
        "test",
        "apparent_vector_RMSE_mps",
        r"Apparent-wind vector RMSE (m s$^{-1}$)",
        "03_test_apparent_rmse",
    )
    metric_curve(
        "test",
        "AWS_RMSE_mps",
        r"AWS RMSE (m s$^{-1}$)",
        "04_test_aws_rmse",
    )
    metric_curve(
        "test",
        "AWA_MAE_deg",
        r"AWA MAE ($^\circ$)",
        "05_test_awa_mae",
    )




# =============================================================================
# 06 physics-constrained helpers
# =============================================================================

PHYSICS_VARIANTS = {
    "Direct-GRU": {
        "use_earth": False,
        "use_body": False,
        "use_heading_geo": False,
        "use_heading_norm": False,
    },
    "Physics-GRU-Earth": {
        "use_earth": True,
        "use_body": False,
        "use_heading_geo": False,
        "use_heading_norm": False,
    },
    "Physics-GRU-Full": {
        "use_earth": True,
        "use_body": True,
        "use_heading_geo": True,
        "use_heading_norm": True,
    },
}


def load_tuned_gru_config(
    tuning_dir: Path | None,
) -> tuple[dict, dict]:
    """
    Load the GRU configuration selected by 05C-v2.

    Fallback preserves the frozen tuned result discussed for this project.
    """
    fallback = {
        "recurrent_units": 64,
        "recurrent_layers": 1,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "dense_units": 64,
        "source": "built-in fallback tuned GRU",
    }

    metadata = {
        "loaded_from_file": False,
        "best_hyperparameters_path": None,
        "final_retrain_summary_path": None,
        "matched_seed": None,
    }

    if tuning_dir is None:
        return fallback, metadata

    best_path = (
        tuning_dir
        / "best_hyperparameters.json"
    )

    if best_path.exists():
        obj = load_json(best_path)

        if "GRU" in obj:
            g = obj["GRU"]

            config = {
                "recurrent_units": int(
                    g["recurrent_units"]
                ),
                "recurrent_layers": int(
                    g["recurrent_layers"]
                ),
                "dropout": float(
                    g["dropout"]
                ),
                "learning_rate": float(
                    g["learning_rate"]
                ),
                "dense_units": int(
                    g.get(
                        "dense_units",
                        64,
                    )
                ),
                "source": str(
                    best_path.resolve()
                ),
            }

            metadata[
                "loaded_from_file"
            ] = True

            metadata[
                "best_hyperparameters_path"
            ] = str(
                best_path.resolve()
            )

        else:
            config = fallback
    else:
        config = fallback

    final_path = (
        tuning_dir
        / "final_retrain_summary.csv"
    )

    if final_path.exists():
        try:
            df = pd.read_csv(
                final_path
            )
            row = df.loc[
                df["model"] == "GRU"
            ]

            if not row.empty:
                metadata[
                    "matched_seed"
                ] = int(
                    row.iloc[0]["seed"]
                )

                metadata[
                    "final_retrain_summary_path"
                ] = str(
                    final_path.resolve()
                )

        except Exception:
            pass

    return config, metadata


def torch_inverse_targets(
    y_z,
    target_mean_t,
    target_std_t,
):
    return (
        y_z
        * target_std_t[
            None,
            None,
            :,
        ]
        + target_mean_t[
            None,
            None,
            :,
        ]
    )


def normalize_heading_torch(
    torch,
    hdg_sin,
    hdg_cos,
    eps: float = 1e-8,
):
    norm = torch.sqrt(
        hdg_sin * hdg_sin
        + hdg_cos * hdg_cos
        + eps
    )

    return (
        hdg_sin / norm,
        hdg_cos / norm,
        norm,
    )


def apparent_earth_torch(
    joint_raw,
):
    apparent_e = (
        joint_raw[
            ...,
            0,
        ]
        - joint_raw[
            ...,
            2,
        ]
    )

    apparent_n = (
        joint_raw[
            ...,
            1,
        ]
        - joint_raw[
            ...,
            3,
        ]
    )

    return (
        apparent_e,
        apparent_n,
    )


def apparent_body_torch(
    torch,
    apparent_e,
    apparent_n,
    hdg_sin,
    hdg_cos,
):
    (
        s,
        c,
        _,
    ) = normalize_heading_torch(
        torch,
        hdg_sin,
        hdg_cos,
    )

    apparent_forward = (
        apparent_e
        * s
        + apparent_n
        * c
    )

    apparent_starboard = (
        apparent_e
        * c
        - apparent_n
        * s
    )

    return (
        apparent_forward,
        apparent_starboard,
    )


def compute_train_physics_scales(
    train_y_raw: np.ndarray,
) -> dict:
    """
    Fit physical residual normalization scales ONLY from outer TRAIN targets.
    """
    y = np.asarray(
        train_y_raw,
        dtype=np.float64,
    )

    app = apparent_earth_from_joint(
        y
    )

    true_hdg_sin = y[
        ...,
        4,
    ]
    true_hdg_cos = y[
        ...,
        5,
    ]

    norm = np.sqrt(
        true_hdg_sin ** 2
        + true_hdg_cos ** 2
    )

    safe_norm = np.where(
        norm > 1e-12,
        norm,
        1.0,
    )

    s = (
        true_hdg_sin
        / safe_norm
    )

    c = (
        true_hdg_cos
        / safe_norm
    )

    body_forward = (
        app[
            ...,
            0,
        ]
        * s
        + app[
            ...,
            1,
        ]
        * c
    )

    body_starboard = (
        app[
            ...,
            0,
        ]
        * c
        - app[
            ...,
            1,
        ]
        * s
    )

    def robust_std(
        values: np.ndarray,
    ) -> float:
        value = float(
            np.nanstd(
                values,
                ddof=0,
            )
        )

        if (
            not np.isfinite(
                value
            )
            or value
            < 1e-6
        ):
            value = 1.0

        return value

    scales = {
        "apparent_earth_east_std_mps": robust_std(
            app[
                ...,
                0,
            ]
        ),
        "apparent_earth_north_std_mps": robust_std(
            app[
                ...,
                1,
            ]
        ),
        "apparent_body_forward_std_mps": robust_std(
            body_forward
        ),
        "apparent_body_starboard_std_mps": robust_std(
            body_starboard
        ),
    }

    return scales


def compute_loss_components(
    *,
    torch,
    pred_z,
    true_z,
    target_mean_t,
    target_std_t,
    physics_scale_t: dict,
):
    """
    Return differentiable normalized loss components.

    All losses are scalar means.
    """
    # Same standardized joint MSE as the conventional baseline.
    direct = torch.mean(
        (
            pred_z
            - true_z
        ) ** 2
    )

    pred_raw = torch_inverse_targets(
        pred_z,
        target_mean_t,
        target_std_t,
    )

    true_raw = torch_inverse_targets(
        true_z,
        target_mean_t,
        target_std_t,
    )

    (
        pred_ae,
        pred_an,
    ) = apparent_earth_torch(
        pred_raw
    )

    (
        true_ae,
        true_an,
    ) = apparent_earth_torch(
        true_raw
    )

    earth_loss = 0.5 * torch.mean(
        (
            (
                pred_ae
                - true_ae
            )
            / physics_scale_t[
                "ae"
            ]
        ) ** 2
        + (
            (
                pred_an
                - true_an
            )
            / physics_scale_t[
                "an"
            ]
        ) ** 2
    )

    pred_hdg_sin = pred_raw[
        ...,
        4,
    ]
    pred_hdg_cos = pred_raw[
        ...,
        5,
    ]

    true_hdg_sin = true_raw[
        ...,
        4,
    ]
    true_hdg_cos = true_raw[
        ...,
        5,
    ]

    (
        pred_s,
        pred_c,
        pred_norm,
    ) = normalize_heading_torch(
        torch,
        pred_hdg_sin,
        pred_hdg_cos,
    )

    (
        true_s,
        true_c,
        _,
    ) = normalize_heading_torch(
        torch,
        true_hdg_sin,
        true_hdg_cos,
    )

    pred_body_forward = (
        pred_ae
        * pred_s
        + pred_an
        * pred_c
    )

    pred_body_starboard = (
        pred_ae
        * pred_c
        - pred_an
        * pred_s
    )

    true_body_forward = (
        true_ae
        * true_s
        + true_an
        * true_c
    )

    true_body_starboard = (
        true_ae
        * true_c
        - true_an
        * true_s
    )

    body_loss = 0.5 * torch.mean(
        (
            (
                pred_body_forward
                - true_body_forward
            )
            / physics_scale_t[
                "af"
            ]
        ) ** 2
        + (
            (
                pred_body_starboard
                - true_body_starboard
            )
            / physics_scale_t[
                "as"
            ]
        ) ** 2
    )

    heading_geo = torch.mean(
        1.0
        - (
            pred_s
            * true_s
            + pred_c
            * true_c
        )
    )

    heading_norm = torch.mean(
        (
            pred_norm
            - 1.0
        ) ** 2
    )

    return {
        "direct": direct,
        "apparent_earth": earth_loss,
        "apparent_body": body_loss,
        "heading_geo": heading_geo,
        "heading_norm": heading_norm,
    }


def combine_variant_loss(
    components: dict,
    variant_name: str,
    weights: dict,
):
    cfg = PHYSICS_VARIANTS[
        variant_name
    ]

    total = (
        weights[
            "direct"
        ]
        * components[
            "direct"
        ]
    )

    if cfg[
        "use_earth"
    ]:
        total = (
            total
            + weights[
                "earth"
            ]
            * components[
                "apparent_earth"
            ]
        )

    if cfg[
        "use_body"
    ]:
        total = (
            total
            + weights[
                "body"
            ]
            * components[
                "apparent_body"
            ]
        )

    if cfg[
        "use_heading_geo"
    ]:
        total = (
            total
            + weights[
                "heading_geo"
            ]
            * components[
                "heading_geo"
            ]
        )

    if cfg[
        "use_heading_norm"
    ]:
        total = (
            total
            + weights[
                "heading_norm"
            ]
            * components[
                "heading_norm"
            ]
        )

    return total


def evaluate_physics_validation_pass(
    *,
    torch,
    model,
    loader,
    device,
    amp_enabled: bool,
    target_mean_t,
    target_std_t,
    target_mean_np: np.ndarray,
    target_std_np: np.ndarray,
    physics_scale_t: dict,
    variant_name: str,
    weights: dict,
):
    """
    Single validation pass:
      - total and component losses
      - physical AW vector RMSE used for early stopping

    No duplicate Keras-style second validation pass.
    """
    model.eval()

    sum_dict = {
        "total": 0.0,
        "direct": 0.0,
        "apparent_earth": 0.0,
        "apparent_body": 0.0,
        "heading_geo": 0.0,
        "heading_norm": 0.0,
    }

    total_n = 0
    pred_chunks = []
    true_chunks = []

    with torch.no_grad():
        for (
            X_batch,
            y_batch,
        ) in loader:
            X_batch = X_batch.to(
                device,
                non_blocking=True,
            )

            y_batch = y_batch.to(
                device,
                non_blocking=True,
            )

            with AutocastContext(
                torch,
                amp_enabled,
                device.type,
            ):
                pred = model(
                    X_batch
                )

                components = (
                    compute_loss_components(
                        torch=torch,
                        pred_z=pred,
                        true_z=y_batch,
                        target_mean_t=target_mean_t,
                        target_std_t=target_std_t,
                        physics_scale_t=physics_scale_t,
                    )
                )

                total_loss = (
                    combine_variant_loss(
                        components,
                        variant_name,
                        weights,
                    )
                )

            n = int(
                X_batch.shape[
                    0
                ]
            )

            total_n += n

            sum_dict[
                "total"
            ] += (
                float(
                    total_loss.detach().item()
                )
                * n
            )

            for key in [
                "direct",
                "apparent_earth",
                "apparent_body",
                "heading_geo",
                "heading_norm",
            ]:
                sum_dict[
                    key
                ] += (
                    float(
                        components[
                            key
                        ]
                        .detach()
                        .item()
                    )
                    * n
                )

            pred_chunks.append(
                pred.detach()
                .float()
                .cpu()
                .numpy()
            )

            true_chunks.append(
                y_batch.detach()
                .float()
                .cpu()
                .numpy()
            )

    pred_z = np.concatenate(
        pred_chunks,
        axis=0,
    )

    true_z = np.concatenate(
        true_chunks,
        axis=0,
    )

    pred_raw = inverse_joint_target(
        pred_z,
        target_mean_np,
        target_std_np,
    )

    true_raw = inverse_joint_target(
        true_z,
        target_mean_np,
        target_std_np,
    )

    true_app = apparent_earth_from_joint(
        true_raw
    )

    pred_app = apparent_earth_from_joint(
        pred_raw
    )

    err = (
        pred_app
        - true_app
    )

    per_horizon_rmse = np.sqrt(
        np.mean(
            err[
                ...,
                0,
            ] ** 2
            + err[
                ...,
                1,
            ] ** 2,
            axis=0,
        )
    )

    mean_aw_rmse = float(
        np.mean(
            per_horizon_rmse
        )
    )

    averages = {
        key: (
            value
            / max(
                total_n,
                1,
            )
        )
        for key, value in sum_dict.items()
    }

    return (
        averages,
        mean_aw_rmse,
        per_horizon_rmse,
    )


def train_physics_variant(
    *,
    torch,
    model,
    variant_name: str,
    train_loader,
    val_loader,
    device,
    amp_enabled: bool,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    min_delta: float,
    gradient_clip_norm: float,
    target_mean_t,
    target_std_t,
    target_mean_np: np.ndarray,
    target_std_np: np.ndarray,
    physics_scale_t: dict,
    weights: dict,
    checkpoint_path: Path,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            learning_rate
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(
                2,
                patience
                // 3,
            ),
            min_lr=1e-5,
        )
    )

    scaler = make_grad_scaler(
        torch,
        amp_enabled,
    )

    best_metric = float(
        "inf"
    )

    best_epoch = 0
    wait = 0
    history_rows = []
    start = time.perf_counter()

    for epoch in range(
        1,
        max_epochs
        + 1,
    ):
        epoch_start = (
            time.perf_counter()
        )

        model.train()

        sum_dict = {
            "total": 0.0,
            "direct": 0.0,
            "apparent_earth": 0.0,
            "apparent_body": 0.0,
            "heading_geo": 0.0,
            "heading_norm": 0.0,
        }

        train_n = 0

        for (
            X_batch,
            y_batch,
        ) in train_loader:
            X_batch = X_batch.to(
                device,
                non_blocking=True,
            )

            y_batch = y_batch.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with AutocastContext(
                torch,
                amp_enabled,
                device.type,
            ):
                pred = model(
                    X_batch
                )

                components = (
                    compute_loss_components(
                        torch=torch,
                        pred_z=pred,
                        true_z=y_batch,
                        target_mean_t=target_mean_t,
                        target_std_t=target_std_t,
                        physics_scale_t=physics_scale_t,
                    )
                )

                total_loss = (
                    combine_variant_loss(
                        components,
                        variant_name,
                        weights,
                    )
                )

            if amp_enabled:
                scaler.scale(
                    total_loss
                ).backward()

                if (
                    gradient_clip_norm
                    > 0.0
                ):
                    scaler.unscale_(
                        optimizer
                    )

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip_norm,
                    )

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:
                total_loss.backward()

                if (
                    gradient_clip_norm
                    > 0.0
                ):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip_norm,
                    )

                optimizer.step()

            n = int(
                X_batch.shape[
                    0
                ]
            )

            train_n += n

            sum_dict[
                "total"
            ] += (
                float(
                    total_loss.detach().item()
                )
                * n
            )

            for key in [
                "direct",
                "apparent_earth",
                "apparent_body",
                "heading_geo",
                "heading_norm",
            ]:
                sum_dict[
                    key
                ] += (
                    float(
                        components[
                            key
                        ]
                        .detach()
                        .item()
                    )
                    * n
                )

        train_avg = {
            key: (
                value
                / max(
                    train_n,
                    1,
                )
            )
            for key, value in sum_dict.items()
        }

        (
            val_avg,
            val_aw_rmse,
            val_aw_per_horizon,
        ) = evaluate_physics_validation_pass(
            torch=torch,
            model=model,
            loader=val_loader,
            device=device,
            amp_enabled=amp_enabled,
            target_mean_t=target_mean_t,
            target_std_t=target_std_t,
            target_mean_np=target_mean_np,
            target_std_np=target_std_np,
            physics_scale_t=physics_scale_t,
            variant_name=variant_name,
            weights=weights,
        )

        current_lr = float(
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        # Keep the same LR-scheduler philosophy as the conventional baseline:
        # scheduler follows total validation training objective.
        scheduler.step(
            val_avg[
                "total"
            ]
        )

        improved = (
            val_aw_rmse
            < (
                best_metric
                - min_delta
            )
        )

        if improved:
            best_metric = float(
                val_aw_rmse
            )

            best_epoch = int(
                epoch
            )

            wait = 0

            torch.save(
                {
                    "variant": variant_name,
                    "epoch": best_epoch,
                    "best_validation_AW_RMSE_mps": (
                        best_metric
                    ),
                    "state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

            status = "[BEST]"

        else:
            wait += 1
            status = (
                f"wait={wait}/{patience}"
            )

        epoch_seconds = float(
            time.perf_counter()
            - epoch_start
        )

        row = {
            "epoch": int(
                epoch
            ),
            "learning_rate": current_lr,
            "epoch_seconds": epoch_seconds,
            "train_total_loss": float(
                train_avg[
                    "total"
                ]
            ),
            "train_direct_loss": float(
                train_avg[
                    "direct"
                ]
            ),
            "train_apparent_earth_loss": float(
                train_avg[
                    "apparent_earth"
                ]
            ),
            "train_apparent_body_loss": float(
                train_avg[
                    "apparent_body"
                ]
            ),
            "train_heading_geo_loss": float(
                train_avg[
                    "heading_geo"
                ]
            ),
            "train_heading_norm_loss": float(
                train_avg[
                    "heading_norm"
                ]
            ),
            "val_total_loss": float(
                val_avg[
                    "total"
                ]
            ),
            "val_direct_loss": float(
                val_avg[
                    "direct"
                ]
            ),
            "val_apparent_earth_loss": float(
                val_avg[
                    "apparent_earth"
                ]
            ),
            "val_apparent_body_loss": float(
                val_avg[
                    "apparent_body"
                ]
            ),
            "val_heading_geo_loss": float(
                val_avg[
                    "heading_geo"
                ]
            ),
            "val_heading_norm_loss": float(
                val_avg[
                    "heading_norm"
                ]
            ),
            "val_apparent_vector_RMSE_mps": float(
                val_aw_rmse
            ),
        }

        for i, horizon in enumerate(
            [
                1,
                2,
                3,
                5,
                10,
            ][
                : len(
                    val_aw_per_horizon
                )
            ]
        ):
            row[
                f"val_AW_RMSE_{horizon}min_mps"
            ] = float(
                val_aw_per_horizon[
                    i
                ]
            )

        history_rows.append(
            row
        )

        log(
            f"Epoch {epoch:03d}/{max_epochs} | "
            f"total={train_avg['total']:.6f} | "
            f"direct={train_avg['direct']:.6f} | "
            f"AW_E={train_avg['apparent_earth']:.6f} | "
            f"AW_B={train_avg['apparent_body']:.6f} | "
            f"val_total={val_avg['total']:.6f} | "
            f"val_AW_RMSE={val_aw_rmse:.6f} m/s | "
            f"lr={current_lr:.3e} | "
            f"{epoch_seconds:.2f}s | "
            f"{status}"
        )

        if (
            wait
            >= patience
        ):
            log(
                f"[EARLY STOP] {variant_name}: "
                f"best_epoch={best_epoch}, "
                f"best validation AW RMSE="
                f"{best_metric:.6f} m/s"
            )
            break

    elapsed = float(
        time.perf_counter()
        - start
    )

    require_file(
        checkpoint_path
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "state_dict"
        ]
    )

    return {
        "model": model,
        "history": pd.DataFrame(
            history_rows
        ),
        "best_epoch": int(
            best_epoch
        ),
        "best_validation_AW_RMSE_mps": float(
            best_metric
        ),
        "epochs_ran": int(
            len(
                history_rows
            )
        ),
        "training_seconds": float(
            elapsed
        ),
    }


def make_physics_plots(
    *,
    metrics: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    output_dir: Path,
):
    import matplotlib.pyplot as plt

    style = load_sci_style(
        Path(
            __file__
        ).resolve().parent
    )

    fig_dir = (
        output_dir
        / "figures"
    )

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    def finish(
        ax,
        grid=True,
    ):
        if (
            style is not None
            and hasattr(
                style,
                "clean_axis",
            )
        ):
            style.clean_axis(
                ax,
                grid=grid,
            )
        else:
            ax.tick_params(
                which="both",
                direction="in",
                top=True,
                right=True,
            )

            if grid:
                ax.grid(
                    True,
                    alpha=0.25,
                )

    def save(
        fig,
        name,
    ):
        base = (
            fig_dir
            / name
        )

        if (
            style is not None
            and hasattr(
                style,
                "save_figure",
            )
        ):
            style.save_figure(
                fig,
                base,
            )
        else:
            fig.savefig(
                base.with_suffix(
                    ".png"
                ),
                dpi=600,
                bbox_inches="tight",
            )

            fig.savefig(
                base.with_suffix(
                    ".pdf"
                ),
                bbox_inches="tight",
            )

            plt.close(
                fig
            )

    preferred_order = [
        "Persistence",
        "Ridge",
        "LSTM",
        "GRU",
        "CNN-LSTM",
        "Direct-GRU",
        "Physics-GRU-Earth",
        "Physics-GRU-Full",
    ]

    def metric_curve(
        split_name: str,
        metric_col: str,
        ylabel: str,
        filename: str,
    ):
        fig, ax = plt.subplots(
            figsize=(
                5.2,
                3.3,
            )
        )

        subset = metrics.loc[
            metrics[
                "split"
            ] == split_name
        ]

        for model_name in preferred_order:
            grp = subset.loc[
                subset[
                    "model"
                ] == model_name
            ].sort_values(
                "horizon_min"
            )

            if grp.empty:
                continue

            ax.plot(
                grp[
                    "horizon_min"
                ],
                grp[
                    metric_col
                ],
                marker="o",
                label=model_name,
            )

        ax.set_xlabel(
            "Forecast horizon (min)"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_xticks(
            [1, 2, 3, 5, 10]
        )

        ax.legend(
            loc="best",
            ncol=2,
        )

        finish(
            ax,
            grid=True,
        )

        save(
            fig,
            filename,
        )

    metric_curve(
        "validation",
        "apparent_vector_RMSE_mps",
        r"Apparent-wind vector RMSE (m s$^{-1}$)",
        "01_validation_apparent_rmse",
    )

    metric_curve(
        "test",
        "apparent_vector_RMSE_mps",
        r"Apparent-wind vector RMSE (m s$^{-1}$)",
        "02_test_apparent_rmse",
    )

    metric_curve(
        "test",
        "AWS_RMSE_mps",
        r"AWS RMSE (m s$^{-1}$)",
        "03_test_aws_rmse",
    )

    metric_curve(
        "test",
        "AWA_MAE_deg",
        r"AWA MAE ($^\circ$)",
        "04_test_awa_mae",
    )

    metric_curve(
        "test",
        "wind_vector_RMSE_mps",
        r"Wind vector RMSE (m s$^{-1}$)",
        "05_test_wind_vector_rmse",
    )

    metric_curve(
        "test",
        "vessel_vector_RMSE_mps",
        r"Vessel vector RMSE (m s$^{-1}$)",
        "06_test_vessel_vector_rmse",
    )

    # Validation-AW training history.
    fig, ax = plt.subplots(
        figsize=(
            4.8,
            3.2,
        )
    )

    for (
        model_name,
        history,
    ) in histories.items():
        ax.plot(
            history[
                "epoch"
            ],
            history[
                "val_apparent_vector_RMSE_mps"
            ],
            label=model_name,
        )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        r"Validation AW RMSE (m s$^{-1}$)"
    )

    ax.legend(
        loc="best"
    )

    finish(
        ax,
        grid=True,
    )

    save(
        fig,
        "07_training_validation_aw_rmse",
    )

    # Physics-Full loss components.
    if (
        "Physics-GRU-Full"
        in histories
    ):
        h = histories[
            "Physics-GRU-Full"
        ]

        fig, ax = plt.subplots(
            figsize=(
                4.8,
                3.2,
            )
        )

        ax.plot(
            h[
                "epoch"
            ],
            h[
                "val_direct_loss"
            ],
            label="Direct",
        )

        ax.plot(
            h[
                "epoch"
            ],
            h[
                "val_apparent_earth_loss"
            ],
            label="AW Earth",
        )

        ax.plot(
            h[
                "epoch"
            ],
            h[
                "val_apparent_body_loss"
            ],
            label="AW body",
        )

        ax.plot(
            h[
                "epoch"
            ],
            h[
                "val_heading_geo_loss"
            ],
            label="Heading geometry",
        )

        ax.set_xlabel(
            "Epoch"
        )

        ax.set_ylabel(
            "Validation loss component"
        )

        ax.legend(
            loc="best"
        )

        finish(
            ax,
            grid=True,
        )

        save(
            fig,
            "08_loss_components_physics_full",
        )


# =============================================================================
# Main 06
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        required=True,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--tuning-dir",
        default=None,
        help=(
            "05C-v2 tuning output directory. "
            "Default: <dataset-dir>/deep_baseline_tuning_pytorch_v0_2"
        ),
    )

    parser.add_argument(
        "--variants",
        default=(
            "Direct-GRU,"
            "Physics-GRU-Earth,"
            "Physics-GRU-Full"
        ),
        help=(
            "Comma-separated matched variants."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--direct-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lambda-aw-earth",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lambda-aw-body",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lambda-heading-geo",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--lambda-heading-norm",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Matched seed for all variants. "
            "Default: seed from 05C-v2 final GRU retrain if available."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--no-persistent-workers",
        action="store_true",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    parser.add_argument(
        "--no-tf32",
        action="store_true",
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
    )

    parser.add_argument(
        "--direction-min-speed",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--cog-min-speed",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--skip-baseline-merge",
        action="store_true",
    )

    parser.add_argument(
        "--skip-validation-flag",
        action="store_true",
    )

    parser.add_argument(
        "--plots",
        action="store_true",
    )

    args = parser.parse_args()

    dataset_dir = Path(
        args.dataset_dir
    )

    output_dir = (
        Path(
            args.output
        )
        if args.output
        else (
            dataset_dir
            / "physics_constrained_gru_v0_1"
        )
    )

    tuning_dir = (
        Path(
            args.tuning_dir
        )
        if args.tuning_dir
        else (
            dataset_dir
            / "deep_baseline_tuning_pytorch_v0_2"
        )
    )

    model_dir = (
        output_dir
        / "models"
    )

    history_dir = (
        output_dir
        / "histories"
    )

    prediction_dir = (
        output_dir
        / "predictions"
    )

    for d in [
        output_dir,
        model_dir,
        history_dir,
        prediction_dir,
    ]:
        d.mkdir(
            parents=True,
            exist_ok=True,
        )

    variants = []

    for token in (
        args.variants.split(
            ","
        )
    ):
        token = token.strip()

        if not token:
            continue

        if (
            token
            not in PHYSICS_VARIANTS
        ):
            raise ValueError(
                f"Unsupported variant '{token}'. "
                f"Allowed: {list(PHYSICS_VARIANTS)}"
            )

        if (
            token
            not in variants
        ):
            variants.append(
                token
            )

    if not variants:
        raise ValueError(
            "No variant selected."
        )

    weights = {
        "direct": float(
            args.direct_weight
        ),
        "earth": float(
            args.lambda_aw_earth
        ),
        "body": float(
            args.lambda_aw_body
        ),
        "heading_geo": float(
            args.lambda_heading_geo
        ),
        "heading_norm": float(
            args.lambda_heading_norm
        ),
    }

    if any(
        value < 0
        for value in weights.values()
    ):
        raise ValueError(
            "All loss weights must be non-negative."
        )

    log(
        "=" * 100
    )
    log(
        "06 - PHYSICS-CONSTRAINED JOINT GRU"
    )
    log(
        "=" * 100
    )
    log(
        f"script_version  : {SCRIPT_VERSION}"
    )
    log(
        f"dataset_dir     : {dataset_dir}"
    )
    log(
        f"tuning_dir      : {tuning_dir}"
    )
    log(
        f"output_dir      : {output_dir}"
    )
    log(
        f"variants        : {variants}"
    )
    log(
        f"loss weights    : {weights}"
    )
    log(
        "early stopping   : validation mean apparent-wind vector RMSE"
    )
    log(
        "test policy      : evaluated only after all variants are frozen"
    )
    log("")

    # ---------------------------------------------------------------------
    # Stage 1/8: metadata and tuned GRU configuration
    # ---------------------------------------------------------------------
    log(
        "[STAGE 1/8] Validation gate, metadata, and tuned GRU configuration"
    )

    flags = find_validation_flags(
        dataset_dir
    )

    if (
        not flags
        and not args.skip_validation_flag
    ):
        raise RuntimeError(
            "No VALIDATION_PASSED.flag found. "
            "Run 03B validation first."
        )

    manifest_path = (
        dataset_dir
        / "dataset_manifest.json"
    )

    scalers_path = (
        dataset_dir
        / "scalers.json"
    )

    require_file(
        manifest_path
    )

    require_file(
        scalers_path
    )

    manifest = load_json(
        manifest_path
    )

    scalers = load_json(
        scalers_path
    )

    if (
        manifest.get(
            "task_name"
        )
        != EXPECTED_TASK_NAME
    ):
        raise RuntimeError(
            "Unexpected task_name."
        )

    feature_names = list(
        manifest[
            "feature_names"
        ]
    )

    target_names = list(
        manifest[
            "target_names"
        ]
    )

    horizons = tuple(
        int(
            v
        )
        for v in manifest[
            "forecast_horizons_minutes"
        ]
    )

    lookback = int(
        manifest[
            "lookback_minutes"
        ]
    )

    if (
        feature_names
        != EXPECTED_FEATURE_NAMES
    ):
        raise RuntimeError(
            "Feature schema differs from frozen joint v0.1."
        )

    if (
        target_names
        != EXPECTED_TARGET_NAMES
    ):
        raise RuntimeError(
            "Target schema differs from frozen joint v0.1."
        )

    (
        feature_scaler_names,
        feature_mean,
        feature_std,
    ) = parse_scaler(
        scalers,
        "feature_scaler",
    )

    (
        target_scaler_names,
        target_mean,
        target_std,
    ) = parse_scaler(
        scalers,
        "target_scaler",
    )

    if (
        feature_scaler_names
        != feature_names
    ):
        raise RuntimeError(
            "Feature scaler names mismatch."
        )

    if (
        target_scaler_names
        != target_names
    ):
        raise RuntimeError(
            "Target scaler names mismatch."
        )

    (
        gru_config,
        tuning_metadata,
    ) = load_tuned_gru_config(
        tuning_dir
    )

    matched_seed = (
        int(
            args.seed
        )
        if args.seed
        is not None
        else (
            int(
                tuning_metadata[
                    "matched_seed"
                ]
            )
            if tuning_metadata[
                "matched_seed"
            ]
            is not None
            else 500043
        )
    )

    log(
        "[TUNED GRU] "
        f"units={gru_config['recurrent_units']}, "
        f"layers={gru_config['recurrent_layers']}, "
        f"dropout={gru_config['dropout']}, "
        f"lr={gru_config['learning_rate']}, "
        f"dense={gru_config['dense_units']}"
    )

    log(
        f"[MATCHED SEED] {matched_seed}"
    )

    # ---------------------------------------------------------------------
    # Stage 2/8: data/device/scales
    # ---------------------------------------------------------------------
    log(
        "[STAGE 2/8] Loading data, CUDA, and TRAIN-only physics scales"
    )

    train = load_npz_split(
        dataset_dir,
        "train",
    )

    validation = load_npz_split(
        dataset_dir,
        "validation",
    )

    test = load_npz_split(
        dataset_dir,
        "test",
    )

    (
        torch,
        nn,
        DataLoader,
        TensorDataset,
    ) = import_torch()

    device = (
        torch.device(
            "cpu"
        )
        if (
            args.force_cpu
            or not torch.cuda.is_available()
        )
        else torch.device(
            "cuda:0"
        )
    )

    amp_enabled = bool(
        (
            device.type
            == "cuda"
        )
        and not args.no_amp
    )

    tf32_enabled = bool(
        (
            device.type
            == "cuda"
        )
        and not args.no_tf32
    )

    configure_tf32(
        torch,
        tf32_enabled,
    )

    set_global_seed(
        torch,
        matched_seed,
        args.deterministic,
    )

    device_info = log_device_info(
        torch,
        device,
        amp_enabled,
        tf32_enabled,
    )

    physics_scales = (
        compute_train_physics_scales(
            train[
                "y_raw"
            ]
        )
    )

    log(
        "[TRAIN-ONLY PHYSICS SCALES]"
    )

    for key, value in (
        physics_scales.items()
    ):
        log(
            f"  {key}: {value:.6f}"
        )

    target_mean_t = (
        torch.tensor(
            target_mean,
            dtype=torch.float32,
            device=device,
        )
    )

    target_std_t = (
        torch.tensor(
            target_std,
            dtype=torch.float32,
            device=device,
        )
    )

    physics_scale_t = {
        "ae": torch.tensor(
            physics_scales[
                "apparent_earth_east_std_mps"
            ],
            dtype=torch.float32,
            device=device,
        ),
        "an": torch.tensor(
            physics_scales[
                "apparent_earth_north_std_mps"
            ],
            dtype=torch.float32,
            device=device,
        ),
        "af": torch.tensor(
            physics_scales[
                "apparent_body_forward_std_mps"
            ],
            dtype=torch.float32,
            device=device,
        ),
        "as": torch.tensor(
            physics_scales[
                "apparent_body_starboard_std_mps"
            ],
            dtype=torch.float32,
            device=device,
        ),
    }

    pin_memory = (
        device.type
        == "cuda"
    )

    persistent_workers = (
        args.num_workers
        > 0
        and not args.no_persistent_workers
    )

    train_ds = make_tensor_dataset(
        torch,
        TensorDataset,
        train[
            "X"
        ],
        train[
            "y"
        ],
    )

    validation_ds = (
        make_tensor_dataset(
            torch,
            TensorDataset,
            validation[
                "X"
            ],
            validation[
                "y"
            ],
        )
    )

    test_ds = make_tensor_dataset(
        torch,
        TensorDataset,
        test[
            "X"
        ],
        test[
            "y"
        ],
    )

    train_loader = make_loader(
        DataLoader,
        train_ds,
        args.batch_size,
        True,
        args.num_workers,
        pin_memory,
        persistent_workers,
        args.prefetch_factor,
    )

    validation_loader = make_loader(
        DataLoader,
        validation_ds,
        args.batch_size,
        False,
        args.num_workers,
        pin_memory,
        persistent_workers,
        args.prefetch_factor,
    )

    test_loader = make_loader(
        DataLoader,
        test_ds,
        args.batch_size,
        False,
        args.num_workers,
        pin_memory,
        persistent_workers,
        args.prefetch_factor,
    )

    ModelClass = build_model_class(
        torch,
        nn,
    )

    # ---------------------------------------------------------------------
    # Stage 3/8: train all matched variants, NO TEST evaluation yet
    # ---------------------------------------------------------------------
    log(
        "[STAGE 3/8] Training matched variants on TRAIN; checkpointing on VALIDATION"
    )

    frozen_models = {}
    histories = {}
    validation_prediction_raw = {}
    validation_metric_frames = []
    training_rows = []

    for variant_name in variants:
        log("")
        log(
            "=" * 100
        )
        log(
            f"TRAINING {variant_name}"
        )
        log(
            "=" * 100
        )

        # Same seed for every variant:
        # same initialization + same DataLoader shuffle sequence.
        set_global_seed(
            torch,
            matched_seed,
            args.deterministic,
        )

        model = ModelClass(
            model_name="GRU",
            n_features=len(
                feature_names
            ),
            n_horizons=len(
                horizons
            ),
            n_targets=len(
                target_names
            ),
            recurrent_units=int(
                gru_config[
                    "recurrent_units"
                ]
            ),
            recurrent_layers=int(
                gru_config[
                    "recurrent_layers"
                ]
            ),
            dense_units=int(
                gru_config[
                    "dense_units"
                ]
            ),
            dropout=float(
                gru_config[
                    "dropout"
                ]
            ),
            cnn_filters=32,
            cnn_kernel_size=3,
            cnn_layers=1,
        ).to(
            device
        )

        n_params = count_parameters(
            model
        )

        checkpoint_path = (
            model_dir
            / (
                variant_name.replace(
                    "-",
                    "_",
                )
                + ".pt"
            )
        )

        result = train_physics_variant(
            torch=torch,
            model=model,
            variant_name=variant_name,
            train_loader=train_loader,
            val_loader=validation_loader,
            device=device,
            amp_enabled=amp_enabled,
            learning_rate=float(
                gru_config[
                    "learning_rate"
                ]
            ),
            max_epochs=args.epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            gradient_clip_norm=args.gradient_clip_norm,
            target_mean_t=target_mean_t,
            target_std_t=target_std_t,
            target_mean_np=target_mean,
            target_std_np=target_std,
            physics_scale_t=physics_scale_t,
            weights=weights,
            checkpoint_path=checkpoint_path,
        )

        model = result[
            "model"
        ]

        history = result[
            "history"
        ]

        histories[
            variant_name
        ] = history

        history.to_csv(
            history_dir
            / (
                variant_name.replace(
                    "-",
                    "_",
                )
                + "_history.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

        # Validation prediction is permitted for checkpoint/reporting.
        val_pred_z = predict_loader(
            torch,
            model,
            validation_loader,
            device,
            amp_enabled,
        )

        val_pred_raw = inverse_joint_target(
            val_pred_z,
            target_mean,
            target_std,
        )

        persistence_val = (
            persistence_joint_prediction(
                validation[
                    "X"
                ],
                feature_names,
                feature_mean,
                feature_std,
                len(
                    horizons
                ),
            )
        )

        val_metrics = evaluate_model(
            truth_joint=validation[
                "y_raw"
            ],
            pred_joint=val_pred_raw,
            truth_apparent_ref=validation[
                "apparent_earth_raw"
            ],
            horizons=horizons,
            split_name="validation",
            model_name=variant_name,
            direction_min_speed=args.direction_min_speed,
            cog_min_speed=args.cog_min_speed,
            persistence_joint=persistence_val,
        )

        validation_metric_frames.append(
            val_metrics
        )

        validation_prediction_raw[
            variant_name
        ] = val_pred_raw

        val_summary = summarize_metrics(
            val_metrics
        ).iloc[
            0
        ]

        save_predictions(
            prediction_dir
            / (
                "validation_predictions_"
                + variant_name.replace(
                    "-",
                    "_",
                )
                + ".npz"
            ),
            variant_name,
            validation[
                "y_raw"
            ],
            val_pred_raw,
            validation[
                "apparent_earth_raw"
            ],
            validation[
                "context_end_time_ns"
            ],
            validation[
                "target_time_ns"
            ],
            horizons,
        )

        frozen_models[
            variant_name
        ] = model

        training_rows.append({
            "model": variant_name,
            "seed": int(
                matched_seed
            ),
            "parameters": int(
                n_params
            ),
            "recurrent_units": int(
                gru_config[
                    "recurrent_units"
                ]
            ),
            "recurrent_layers": int(
                gru_config[
                    "recurrent_layers"
                ]
            ),
            "dropout": float(
                gru_config[
                    "dropout"
                ]
            ),
            "learning_rate": float(
                gru_config[
                    "learning_rate"
                ]
            ),
            "best_epoch": int(
                result[
                    "best_epoch"
                ]
            ),
            "epochs_ran": int(
                result[
                    "epochs_ran"
                ]
            ),
            "best_validation_AW_RMSE_mps": float(
                result[
                    "best_validation_AW_RMSE_mps"
                ]
            ),
            "validation_AWS_RMSE_mps": float(
                val_summary[
                    "mean_AWS_RMSE_mps"
                ]
            ),
            "validation_AWA_MAE_deg": float(
                val_summary[
                    "mean_AWA_MAE_deg"
                ]
            ),
            "training_seconds": float(
                result[
                    "training_seconds"
                ]
            ),
            "uses_earth_physics": bool(
                PHYSICS_VARIANTS[
                    variant_name
                ][
                    "use_earth"
                ]
            ),
            "uses_body_physics": bool(
                PHYSICS_VARIANTS[
                    variant_name
                ][
                    "use_body"
                ]
            ),
            "uses_heading_geometry": bool(
                PHYSICS_VARIANTS[
                    variant_name
                ][
                    "use_heading_geo"
                ]
            ),
            "uses_heading_norm": bool(
                PHYSICS_VARIANTS[
                    variant_name
                ][
                    "use_heading_norm"
                ]
            ),
        })

        log(
            f"[FROZEN] {variant_name}: "
            f"parameters={n_params:,}, "
            f"best_epoch={result['best_epoch']}, "
            f"validation AW="
            f"{val_summary['mean_apparent_vector_RMSE_mps']:.4f} m/s"
        )

    training_df = pd.DataFrame(
        training_rows
    )

    training_df.to_csv(
        output_dir
        / "physics_training_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Selection on validation is allowed; TEST still untouched.
    selected_physics_variant = (
        training_df.sort_values(
            "best_validation_AW_RMSE_mps"
        ).iloc[
            0
        ][
            "model"
        ]
    )

    log("")
    log(
        f"[SELECTED ON VALIDATION] "
        f"best matched variant = "
        f"{selected_physics_variant}"
    )

    # ---------------------------------------------------------------------
    # Stage 4/8: TEST evaluation only now, after ALL variants are frozen
    # ---------------------------------------------------------------------
    log(
        "[STAGE 4/8] Evaluating frozen variants on untouched TEST"
    )

    test_metric_frames = []
    test_prediction_raw = {}

    persistence_test = (
        persistence_joint_prediction(
            test[
                "X"
            ],
            feature_names,
            feature_mean,
            feature_std,
            len(
                horizons
            ),
        )
    )

    for variant_name in variants:
        model = frozen_models[
            variant_name
        ]

        test_pred_z = predict_loader(
            torch,
            model,
            test_loader,
            device,
            amp_enabled,
        )

        test_pred_raw = inverse_joint_target(
            test_pred_z,
            target_mean,
            target_std,
        )

        test_prediction_raw[
            variant_name
        ] = test_pred_raw

        test_metrics = evaluate_model(
            truth_joint=test[
                "y_raw"
            ],
            pred_joint=test_pred_raw,
            truth_apparent_ref=test[
                "apparent_earth_raw"
            ],
            horizons=horizons,
            split_name="test",
            model_name=variant_name,
            direction_min_speed=args.direction_min_speed,
            cog_min_speed=args.cog_min_speed,
            persistence_joint=persistence_test,
        )

        test_metric_frames.append(
            test_metrics
        )

        save_predictions(
            prediction_dir
            / (
                "test_predictions_"
                + variant_name.replace(
                    "-",
                    "_",
                )
                + ".npz"
            ),
            variant_name,
            test[
                "y_raw"
            ],
            test_pred_raw,
            test[
                "apparent_earth_raw"
            ],
            test[
                "context_end_time_ns"
            ],
            test[
                "target_time_ns"
            ],
            horizons,
        )

        test_summary_row = (
            summarize_metrics(
                test_metrics
            ).iloc[
                0
            ]
        )

        log(
            f"[TEST] {variant_name}: "
            f"AW={test_summary_row['mean_apparent_vector_RMSE_mps']:.4f} m/s | "
            f"AWS={test_summary_row['mean_AWS_RMSE_mps']:.4f} m/s | "
            f"AWA={test_summary_row['mean_AWA_MAE_deg']:.2f} deg | "
            f"wind={test_summary_row['mean_wind_vector_RMSE_mps']:.4f} m/s | "
            f"vessel={test_summary_row['mean_vessel_vector_RMSE_mps']:.4f} m/s"
        )

    # ---------------------------------------------------------------------
    # Stage 5/8: merge frozen 05C-v2 baselines
    # ---------------------------------------------------------------------
    log(
        "[STAGE 5/8] Merging frozen tuned baselines"
    )

    metric_frames = [
        *validation_metric_frames,
        *test_metric_frames,
    ]

    baseline_merged = False
    baseline_path = (
        tuning_dir
        / "tuned_metrics_per_horizon.csv"
    )

    if (
        not args.skip_baseline_merge
        and baseline_path.exists()
    ):
        baseline_df = pd.read_csv(
            baseline_path
        )

        required_cols = set(
            metric_frames[
                0
            ].columns
        )

        if required_cols.issubset(
            baseline_df.columns
        ):
            baseline_df = baseline_df[
                list(
                    metric_frames[
                        0
                    ].columns
                )
            ]

            metric_frames.append(
                baseline_df
            )

            baseline_merged = True

            log(
                f"[OK] Merged tuned baseline metrics: "
                f"{baseline_path}"
            )

        else:
            log(
                "[WARNING] Tuned baseline metric schema differs; "
                "not merged."
            )

    else:
        log(
            "[INFO] Tuned baseline metrics not merged."
        )

    metrics = pd.concat(
        metric_frames,
        ignore_index=True,
    )

    preferred_order = {
        name: i
        for i, name in enumerate(
            [
                "Persistence",
                "Ridge",
                "LSTM",
                "GRU",
                "CNN-LSTM",
                "Direct-GRU",
                "Physics-GRU-Earth",
                "Physics-GRU-Full",
            ]
        )
    }

    metrics[
        "_order"
    ] = (
        metrics[
            "model"
        ]
        .map(
            preferred_order
        )
        .fillna(
            999
        )
    )

    metrics = (
        metrics.sort_values(
            [
                "split",
                "_order",
                "horizon_min",
            ]
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    metrics.to_csv(
        output_dir
        / "physics_metrics_per_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = summarize_metrics(
        metrics
    )

    summary[
        "_order"
    ] = (
        summary[
            "model"
        ]
        .map(
            preferred_order
        )
        .fillna(
            999
        )
    )

    summary = (
        summary.sort_values(
            [
                "split",
                "_order",
            ]
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    summary.to_csv(
        output_dir
        / "physics_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------------------
    # Stage 6/8: parameter-equivalence and improvement audit
    # ---------------------------------------------------------------------
    log(
        "[STAGE 6/8] Parameter-equivalence and physics-effect audit"
    )

    parameter_counts = (
        training_df[
            "parameters"
        ]
        .astype(
            int
        )
        .unique()
    )

    same_parameter_count = (
        len(
            parameter_counts
        )
        == 1
    )

    if same_parameter_count:
        log(
            f"[OK] All matched variants have exactly "
            f"{parameter_counts[0]:,} trainable parameters."
        )

    test_summary_all = (
        summary.loc[
            summary[
                "split"
            ] == "test"
        ].copy()
    )

    def get_test_row(
        model_name: str,
    ):
        rows = (
            test_summary_all.loc[
                test_summary_all[
                    "model"
                ] == model_name
            ]
        )

        if rows.empty:
            return None

        return rows.iloc[
            0
        ]

    direct_row = get_test_row(
        "Direct-GRU"
    )

    full_row = get_test_row(
        "Physics-GRU-Full"
    )

    ridge_row = get_test_row(
        "Ridge"
    )

    physics_effect = {}

    if (
        direct_row
        is not None
        and full_row
        is not None
    ):
        direct_aw = float(
            direct_row[
                "mean_apparent_vector_RMSE_mps"
            ]
        )

        full_aw = float(
            full_row[
                "mean_apparent_vector_RMSE_mps"
            ]
        )

        physics_effect[
            "AW_RMSE_reduction_vs_Direct_GRU_percent"
        ] = (
            100.0
            * (
                direct_aw
                - full_aw
            )
            / direct_aw
        )

        physics_effect[
            "Direct_GRU_AW_RMSE_mps"
        ] = direct_aw

        physics_effect[
            "Physics_GRU_Full_AW_RMSE_mps"
        ] = full_aw

    if (
        ridge_row
        is not None
        and full_row
        is not None
    ):
        ridge_aw = float(
            ridge_row[
                "mean_apparent_vector_RMSE_mps"
            ]
        )

        full_aw = float(
            full_row[
                "mean_apparent_vector_RMSE_mps"
            ]
        )

        physics_effect[
            "AW_skill_vs_Ridge"
        ] = (
            1.0
            - full_aw
            / ridge_aw
        )

    # ---------------------------------------------------------------------
    # Stage 7/8: manifest/report
    # ---------------------------------------------------------------------
    log(
        "[STAGE 7/8] Writing manifest and scientific report"
    )

    out_manifest = {
        "script_version": SCRIPT_VERSION,
        "backend": "PyTorch",
        "task_name": EXPECTED_TASK_NAME,
        "source_dataset_version": manifest.get(
            "dataset_version"
        ),
        "dataset_dir": str(
            dataset_dir.resolve()
        ),
        "tuning_dir": str(
            tuning_dir.resolve()
        ),
        "tuned_gru_config": gru_config,
        "tuning_metadata": tuning_metadata,
        "matched_seed": int(
            matched_seed
        ),
        "matched_variants": variants,
        "same_trainable_parameter_count_across_variants": bool(
            same_parameter_count
        ),
        "trainable_parameter_count": (
            int(
                parameter_counts[
                    0
                ]
            )
            if same_parameter_count
            else [
                int(
                    v
                )
                for v in parameter_counts
            ]
        ),
        "loss_weights": weights,
        "physics_scales_fitted_on_outer_train_only": (
            physics_scales
        ),
        "physics_definition": {
            "apparent_earth": (
                "A_E = U_w - V_ship,E; "
                "A_N = V_w - V_ship,N"
            ),
            "body_forward": (
                "A_f = A_E sin(HDG) + A_N cos(HDG)"
            ),
            "body_starboard": (
                "A_s = A_E cos(HDG) - A_N sin(HDG)"
            ),
            "AWA_definition": (
                "wind-from angle; 0=headwind, "
                "+90=starboard, -90=port, +/-180=astern"
            ),
        },
        "training_protocol": {
            "train_split": "frozen outer train",
            "checkpoint_split": "frozen outer validation",
            "checkpoint_metric": (
                "mean apparent-wind vector RMSE "
                "over all forecast horizons"
            ),
            "test_evaluated_only_after_all_variants_frozen": True,
            "max_epochs": int(
                args.epochs
            ),
            "patience": int(
                args.patience
            ),
            "batch_size": int(
                args.batch_size
            ),
            "gradient_clip_norm": float(
                args.gradient_clip_norm
            ),
        },
        "selected_matched_variant_on_validation": str(
            selected_physics_variant
        ),
        "physics_effect": physics_effect,
        "baseline_metrics_merged": bool(
            baseline_merged
        ),
        "device_info": device_info,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": (
                torch.backends.cudnn.version()
                if hasattr(
                    torch.backends,
                    "cudnn",
                )
                else None
            ),
        },
    }

    save_json(
        output_dir
        / "physics_model_manifest.json",
        out_manifest,
    )

    with (
        output_dir
        / "physics_model_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "06 Physics-Constrained Joint GRU Report\n"
        )

        f.write(
            "=" * 100
            + "\n\n"
        )

        f.write(
            "Purpose\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )

        f.write(
            "Test whether explicit apparent-wind physics supervision "
            "improves a tuned GRU without increasing trainable model capacity.\n\n"
        )

        f.write(
            "Matched GRU configuration\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )

        f.write(
            json.dumps(
                gru_config,
                ensure_ascii=False,
                indent=2,
            )
        )

        f.write(
            "\n\nLoss weights\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )

        f.write(
            json.dumps(
                weights,
                ensure_ascii=False,
                indent=2,
            )
        )

        f.write(
            "\n\nTRAIN-only physics scales\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )

        f.write(
            json.dumps(
                physics_scales,
                ensure_ascii=False,
                indent=2,
            )
        )

        f.write(
            "\n\nTraining summary\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )

        f.write(
            training_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nValidation metrics\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )

        f.write(
            summary.loc[
                summary[
                    "split"
                ] == "validation"
            ].to_string(
                index=False
            )
        )

        f.write(
            "\n\nTest metrics\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )

        f.write(
            summary.loc[
                summary[
                    "split"
                ] == "test"
            ].to_string(
                index=False
            )
        )

        f.write(
            "\n\nPhysics-effect audit\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )

        f.write(
            json.dumps(
                physics_effect,
                ensure_ascii=False,
                indent=2,
            )
        )

        f.write(
            "\n\nSelected matched variant on validation: "
            f"{selected_physics_variant}\n"
        )

    # ---------------------------------------------------------------------
    # Stage 8/8: figures and console summary
    # ---------------------------------------------------------------------
    log(
        "[STAGE 8/8] Finalization"
    )

    if args.plots:
        make_physics_plots(
            metrics=metrics,
            histories=histories,
            output_dir=output_dir,
        )

        log(
            "[OK] Figures complete"
        )

    log("")
    log(
        "=" * 100
    )
    log(
        "PHYSICS-CONSTRAINED GRU RESULTS"
    )
    log(
        "=" * 100
    )

    for model_name in [
        "Ridge",
        "GRU",
        *variants,
    ]:
        row = get_test_row(
            model_name
        )

        if row is None:
            continue

        log(
            f"{model_name}:"
        )

        log(
            f"  wind vector RMSE     = "
            f"{row['mean_wind_vector_RMSE_mps']:.4f} m/s"
        )

        log(
            f"  vessel vector RMSE   = "
            f"{row['mean_vessel_vector_RMSE_mps']:.4f} m/s"
        )

        log(
            f"  apparent vector RMSE = "
            f"{row['mean_apparent_vector_RMSE_mps']:.4f} m/s"
        )

        log(
            f"  AWS RMSE             = "
            f"{row['mean_AWS_RMSE_mps']:.4f} m/s"
        )

        log(
            f"  AWA MAE              = "
            f"{row['mean_AWA_MAE_deg']:.2f} deg"
        )

        log(
            f"  HDG MAE              = "
            f"{row['mean_HDG_MAE_deg']:.2f} deg"
        )

    log("")

    if physics_effect:
        for key, value in (
            physics_effect.items()
        ):
            log(
                f"[PHYSICS EFFECT] "
                f"{key} = {value:.6f}"
            )

    log(
        f"[SELECTED ON VALIDATION] "
        f"{selected_physics_variant}"
    )

    log(
        f"[DEVICE] {device}"
    )

    if (
        device.type
        == "cuda"
    ):
        log(
            f"[GPU] "
            f"{torch.cuda.get_device_name(device)}"
        )

    log(
        f"[DONE] Physics-GRU outputs: "
        f"{output_dir}"
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

        print(
            "\nPlease send the complete traceback and terminal output.",
            flush=True,
        )

        sys.exit(
            1
        )
