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


SCRIPT_VERSION = "0.1.0"
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
# Main
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
        "--models",
        default="LSTM,GRU,CNN-LSTM",
    )

    parser.add_argument(
        "--trials-per-model",
        type=int,
        default=12,
        help=(
            "Controlled random-search trials per model. "
            "Default: 12."
        ),
    )
    parser.add_argument(
        "--search-epochs",
        type=int,
        default=50,
        help=(
            "Maximum epochs per inner-search trial. "
            "Default: 50."
        ),
    )
    parser.add_argument(
        "--final-epochs",
        type=int,
        default=80,
        help=(
            "Maximum epochs for final refit on full train. "
            "Default: 80."
        ),
    )
    parser.add_argument(
        "--search-patience",
        type=int,
        default=7,
        help=(
            "Early-stopping patience during inner search. "
            "Default: 7."
        ),
    )
    parser.add_argument(
        "--final-patience",
        type=int,
        default=10,
        help=(
            "Early-stopping patience during final refit. "
            "Default: 10."
        ),
    )
    parser.add_argument(
        "--inner-val-fraction",
        type=float,
        default=0.20,
        help=(
            "Last fraction of original train used as inner validation. "
            "Default: 0.20."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--dense-units",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
        "--ridge-baseline-dir",
        default=None,
    )
    parser.add_argument(
        "--skip-ridge-merge",
        action="store_true",
    )
    parser.add_argument(
        "--skip-validation-flag",
        action="store_true",
    )
    parser.add_argument(
        "--verbose-search-epochs",
        action="store_true",
        help=(
            "Print every epoch during hyperparameter search. "
            "Default prints one summary per trial."
        ),
    )
    parser.add_argument(
        "--plots",
        action="store_true",
    )

    args = parser.parse_args()

    if not (0.05 <= args.inner_val_fraction <= 0.40):
        raise ValueError(
            "--inner-val-fraction should be between 0.05 and 0.40."
        )

    requested_models = []

    mapping = {
        "lstm": "LSTM",
        "gru": "GRU",
        "cnn-lstm": "CNN-LSTM",
        "cnn_lstm": "CNN-LSTM",
        "cnnlstm": "CNN-LSTM",
    }

    for token in args.models.split(","):
        token = token.strip()
        if not token:
            continue

        normalized = mapping.get(
            token.lower()
        )
        if normalized is None:
            raise ValueError(
                f"Unsupported model: {token}"
            )

        if normalized not in requested_models:
            requested_models.append(
                normalized
            )

    if not requested_models:
        raise ValueError(
            "No model selected."
        )

    dataset_dir = Path(
        args.dataset_dir
    )

    output_dir = (
        Path(args.output)
        if args.output
        else (
            dataset_dir
            / "deep_baseline_tuning_pytorch_v0_1"
        )
    )

    model_dir = output_dir / "models"
    history_dir = output_dir / "histories"
    prediction_dir = output_dir / "predictions"
    trial_ckpt_dir = output_dir / "_trial_checkpoints"

    for d in [
        output_dir,
        model_dir,
        history_dir,
        prediction_dir,
        trial_ckpt_dir,
    ]:
        d.mkdir(
            parents=True,
            exist_ok=True,
        )

    log("=" * 100)
    log("05C - PYTORCH DEEP-BASELINE HYPERPARAMETER TUNING")
    log("=" * 100)
    log(f"script_version     : {SCRIPT_VERSION}")
    log(f"dataset_dir        : {dataset_dir}")
    log(f"output_dir         : {output_dir}")
    log(f"models             : {requested_models}")
    log(
        f"trials/model       : "
        f"{args.trials_per_model}"
    )
    log(
        f"inner_val_fraction : "
        f"{args.inner_val_fraction:.2f}"
    )
    log(
        "search objective   : "
        "inner-validation mean apparent-wind vector RMSE"
    )
    log(
        "test policy        : "
        "untouched until final selected models are frozen"
    )
    log("")

    # ---------------------------------------------------------------------
    # Stage 1: metadata gate
    # ---------------------------------------------------------------------
    log("[STAGE 1/8] Validation gate and metadata")

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

    require_file(manifest_path)
    require_file(scalers_path)

    manifest = load_json(
        manifest_path
    )
    scalers = load_json(
        scalers_path
    )

    if (
        manifest.get("task_name")
        != EXPECTED_TASK_NAME
    ):
        raise RuntimeError(
            "Unexpected task_name in dataset_manifest.json."
        )

    feature_names = list(
        manifest["feature_names"]
    )
    target_names = list(
        manifest["target_names"]
    )
    horizons = tuple(
        int(v)
        for v in manifest[
            "forecast_horizons_minutes"
        ]
    )
    lookback = int(
        manifest["lookback_minutes"]
    )

    if feature_names != EXPECTED_FEATURE_NAMES:
        raise RuntimeError(
            "Feature schema differs from frozen joint v0.1."
        )

    if target_names != EXPECTED_TARGET_NAMES:
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

    if feature_scaler_names != feature_names:
        raise RuntimeError(
            "Feature scaler names mismatch."
        )

    if target_scaler_names != target_names:
        raise RuntimeError(
            "Target scaler names mismatch."
        )

    # ---------------------------------------------------------------------
    # Stage 2: load data / torch / inner split
    # ---------------------------------------------------------------------
    log("[STAGE 2/8] Loading data, PyTorch, and inner chronological split")

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
        torch.device("cpu")
        if args.force_cpu
        or not torch.cuda.is_available()
        else torch.device("cuda:0")
    )

    amp_enabled = bool(
        device.type == "cuda"
        and not args.no_amp
    )
    tf32_enabled = bool(
        device.type == "cuda"
        and not args.no_tf32
    )

    configure_tf32(
        torch,
        tf32_enabled,
    )
    set_global_seed(
        torch,
        args.seed,
        args.deterministic,
    )

    device_info = log_device_info(
        torch,
        device,
        amp_enabled,
        tf32_enabled,
    )

    inner = chronological_inner_split(
        train,
        args.inner_val_fraction,
    )

    inner_train_last = pd.to_datetime(
        inner["inner_train_last_context_ns"],
        unit="ns",
        utc=True,
    )
    inner_val_first = pd.to_datetime(
        inner["inner_val_first_context_ns"],
        unit="ns",
        utc=True,
    )

    log(
        f"[INNER SPLIT] inner_train={inner['n_inner_train']}, "
        f"inner_val={inner['n_inner_val']}"
    )
    log(
        f"[INNER SPLIT] inner_train last context = "
        f"{inner_train_last}"
    )
    log(
        f"[INNER SPLIT] inner_val first context  = "
        f"{inner_val_first}"
    )

    pin_memory = (
        device.type == "cuda"
    )
    persistent_workers = (
        args.num_workers > 0
        and not args.no_persistent_workers
    )

    inner_train_ds = make_tensor_dataset(
        torch,
        TensorDataset,
        inner["inner_train_X"],
        inner["inner_train_y"],
    )
    inner_val_ds = make_tensor_dataset(
        torch,
        TensorDataset,
        inner["inner_val_X"],
        inner["inner_val_y"],
    )

    inner_train_loader = make_loader(
        DataLoader,
        inner_train_ds,
        args.batch_size,
        True,
        args.num_workers,
        pin_memory,
        persistent_workers,
        args.prefetch_factor,
    )
    inner_val_loader = make_loader(
        DataLoader,
        inner_val_ds,
        args.batch_size,
        False,
        args.num_workers,
        pin_memory,
        persistent_workers,
        args.prefetch_factor,
    )

    # Final loaders
    full_train_ds = make_tensor_dataset(
        torch,
        TensorDataset,
        train["X"],
        train["y"],
    )
    validation_ds = make_tensor_dataset(
        torch,
        TensorDataset,
        validation["X"],
        validation["y"],
    )
    test_ds = make_tensor_dataset(
        torch,
        TensorDataset,
        test["X"],
        test["y"],
    )

    full_train_loader = make_loader(
        DataLoader,
        full_train_ds,
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
    # Stage 3: hyperparameter search
    # ---------------------------------------------------------------------
    log("[STAGE 3/8] Controlled hyperparameter search on original TRAIN only")

    trial_rows = []
    best_by_model = {}

    global_trial_id = 0

    for model_index, model_name in enumerate(
        requested_models
    ):
        log("")
        log("=" * 100)
        log(f"SEARCHING {model_name}")
        log("=" * 100)

        configs = select_trial_configs(
            model_name=model_name,
            n_trials=args.trials_per_model,
            seed=args.seed + 1000 * model_index,
        )

        model_best_score = float("inf")
        model_best_row = None

        for local_trial_id, cfg in enumerate(
            configs,
            start=1,
        ):
            global_trial_id += 1
            trial_seed = (
                args.seed
                + 10000 * model_index
                + local_trial_id
            )

            set_global_seed(
                torch,
                trial_seed,
                args.deterministic,
            )

            model = ModelClass(
                model_name=model_name,
                n_features=len(feature_names),
                n_horizons=len(horizons),
                n_targets=len(target_names),
                recurrent_units=int(
                    cfg["recurrent_units"]
                ),
                recurrent_layers=int(
                    cfg["recurrent_layers"]
                ),
                dense_units=args.dense_units,
                dropout=float(cfg["dropout"]),
                cnn_filters=int(
                    cfg["cnn_filters"]
                ),
                cnn_kernel_size=int(
                    cfg["cnn_kernel_size"]
                ),
                cnn_layers=int(
                    cfg["cnn_layers"]
                ),
            ).to(device)

            n_params = count_parameters(
                model
            )

            ckpt_path = (
                trial_ckpt_dir
                / (
                    f"{model_name.replace('-', '_')}"
                    f"_trial{local_trial_id:03d}.pt"
                )
            )

            log(
                f"[TRIAL {local_trial_id:02d}/{len(configs)}] "
                f"{model_name} | "
                f"units={cfg['recurrent_units']} | "
                f"layers={cfg['recurrent_layers']} | "
                f"dropout={cfg['dropout']} | "
                f"lr={cfg['learning_rate']:.1e}"
                + (
                    f" | filters={cfg['cnn_filters']} | "
                    f"kernel={cfg['cnn_kernel_size']} | "
                    f"conv_layers={cfg['cnn_layers']}"
                    if model_name == "CNN-LSTM"
                    else ""
                )
            )

            result = train_with_early_stopping(
                torch=torch,
                nn=nn,
                model=model,
                train_loader=inner_train_loader,
                val_loader=inner_val_loader,
                device=device,
                amp_enabled=amp_enabled,
                learning_rate=float(
                    cfg["learning_rate"]
                ),
                max_epochs=args.search_epochs,
                patience=args.search_patience,
                min_delta=args.min_delta,
                gradient_clip_norm=args.gradient_clip_norm,
                target_mean=target_mean,
                target_std=target_std,
                checkpoint_path=ckpt_path,
                verbose_epochs=args.verbose_search_epochs,
            )

            score = float(
                result["best_score"]
            )

            row = {
                "global_trial_id": global_trial_id,
                "trial_id": local_trial_id,
                "model": model_name,
                "seed": trial_seed,
                "recurrent_units": int(
                    cfg["recurrent_units"]
                ),
                "recurrent_layers": int(
                    cfg["recurrent_layers"]
                ),
                "dropout": float(
                    cfg["dropout"]
                ),
                "learning_rate": float(
                    cfg["learning_rate"]
                ),
                "cnn_filters": int(
                    cfg["cnn_filters"]
                ),
                "cnn_kernel_size": int(
                    cfg["cnn_kernel_size"]
                ),
                "cnn_layers": int(
                    cfg["cnn_layers"]
                ),
                "dense_units": int(
                    args.dense_units
                ),
                "parameters": int(
                    n_params
                ),
                "epochs_ran": int(
                    result["epochs_ran"]
                ),
                "best_epoch": int(
                    result["best_epoch"]
                ),
                "inner_val_best_apparent_vector_RMSE_mps": score,
                "training_seconds": float(
                    result["training_seconds"]
                ),
                "is_anchor_05B_config": bool(
                    canonical_config_tuple(cfg)
                    == canonical_config_tuple(
                        ANCHOR_CONFIG
                    )
                ),
            }

            trial_rows.append(row)

            log(
                f"        -> best inner-val AW RMSE="
                f"{score:.6f} m/s | "
                f"best_epoch={result['best_epoch']} | "
                f"time={result['training_seconds']:.1f}s"
            )

            if score < model_best_score:
                model_best_score = score
                model_best_row = dict(row)

            del model
            gc.collect()

            if device.type == "cuda":
                torch.cuda.empty_cache()

        if model_best_row is None:
            raise RuntimeError(
                f"No successful trials for {model_name}."
            )

        best_by_model[
            model_name
        ] = model_best_row

        log(
            f"[SEARCH WINNER] {model_name}: "
            f"inner-val AW RMSE="
            f"{model_best_score:.6f} m/s"
        )

    trials_df = pd.DataFrame(
        trial_rows
    )

    trials_df.to_csv(
        output_dir
        / "hyperparameter_trials.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_hparams = {}

    for model_name, row in best_by_model.items():
        best_hparams[model_name] = {
            "recurrent_units": int(
                row["recurrent_units"]
            ),
            "recurrent_layers": int(
                row["recurrent_layers"]
            ),
            "dropout": float(
                row["dropout"]
            ),
            "learning_rate": float(
                row["learning_rate"]
            ),
            "cnn_filters": int(
                row["cnn_filters"]
            ),
            "cnn_kernel_size": int(
                row["cnn_kernel_size"]
            ),
            "cnn_layers": int(
                row["cnn_layers"]
            ),
            "dense_units": int(
                row["dense_units"]
            ),
            "inner_val_best_apparent_vector_RMSE_mps": float(
                row[
                    "inner_val_best_apparent_vector_RMSE_mps"
                ]
            ),
            "search_best_epoch": int(
                row["best_epoch"]
            ),
        }

    save_json(
        output_dir
        / "best_hyperparameters.json",
        best_hparams,
    )

    # ---------------------------------------------------------------------
    # Stage 4: final refit selected configs
    # ---------------------------------------------------------------------
    log("")
    log("[STAGE 4/8] Final refit on FULL train; original validation only for checkpointing")

    final_rows = []
    deep_metric_frames = []

    for model_index, model_name in enumerate(
        requested_models
    ):
        cfg = best_hparams[
            model_name
        ]

        log("")
        log("=" * 100)
        log(f"FINAL REFIT {model_name}")
        log("=" * 100)
        log(
            f"selected config: {cfg}"
        )

        final_seed = (
            args.seed
            + 50000
            + model_index
        )

        set_global_seed(
            torch,
            final_seed,
            args.deterministic,
        )

        model = ModelClass(
            model_name=model_name,
            n_features=len(feature_names),
            n_horizons=len(horizons),
            n_targets=len(target_names),
            recurrent_units=int(
                cfg["recurrent_units"]
            ),
            recurrent_layers=int(
                cfg["recurrent_layers"]
            ),
            dense_units=int(
                cfg["dense_units"]
            ),
            dropout=float(
                cfg["dropout"]
            ),
            cnn_filters=int(
                cfg["cnn_filters"]
            ),
            cnn_kernel_size=int(
                cfg["cnn_kernel_size"]
            ),
            cnn_layers=int(
                cfg["cnn_layers"]
            ),
        ).to(device)

        n_params = count_parameters(
            model
        )

        final_ckpt = (
            model_dir
            / (
                f"best_{model_name.replace('-', '_')}.pt"
            )
        )

        result = train_with_early_stopping(
            torch=torch,
            nn=nn,
            model=model,
            train_loader=full_train_loader,
            val_loader=validation_loader,
            device=device,
            amp_enabled=amp_enabled,
            learning_rate=float(
                cfg["learning_rate"]
            ),
            max_epochs=args.final_epochs,
            patience=args.final_patience,
            min_delta=args.min_delta,
            gradient_clip_norm=args.gradient_clip_norm,
            target_mean=target_mean,
            target_std=target_std,
            checkpoint_path=final_ckpt,
            verbose_epochs=True,
        )

        history = result["history"]
        history.to_csv(
            history_dir
            / (
                f"final_history_"
                f"{model_name.replace('-', '_')}.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

        val_pred_z = predict_loader(
            torch,
            model,
            validation_loader,
            device,
            amp_enabled,
        )
        test_pred_z = predict_loader(
            torch,
            model,
            test_loader,
            device,
            amp_enabled,
        )

        val_pred_raw = inverse_joint_target(
            val_pred_z,
            target_mean,
            target_std,
        )
        test_pred_raw = inverse_joint_target(
            test_pred_z,
            target_mean,
            target_std,
        )

        persistence_val = persistence_joint_prediction(
            validation["X"],
            feature_names,
            feature_mean,
            feature_std,
            len(horizons),
        )
        persistence_test = persistence_joint_prediction(
            test["X"],
            feature_names,
            feature_mean,
            feature_std,
            len(horizons),
        )

        val_metrics = evaluate_model(
            truth_joint=validation["y_raw"],
            pred_joint=val_pred_raw,
            truth_apparent_ref=validation[
                "apparent_earth_raw"
            ],
            horizons=horizons,
            split_name="validation",
            model_name=model_name,
            direction_min_speed=args.direction_min_speed,
            cog_min_speed=args.cog_min_speed,
            persistence_joint=persistence_val,
        )

        test_metrics = evaluate_model(
            truth_joint=test["y_raw"],
            pred_joint=test_pred_raw,
            truth_apparent_ref=test[
                "apparent_earth_raw"
            ],
            horizons=horizons,
            split_name="test",
            model_name=model_name,
            direction_min_speed=args.direction_min_speed,
            cog_min_speed=args.cog_min_speed,
            persistence_joint=persistence_test,
        )

        deep_metric_frames.extend([
            val_metrics,
            test_metrics,
        ])

        save_predictions(
            prediction_dir
            / (
                f"validation_predictions_"
                f"{model_name.replace('-', '_')}.npz"
            ),
            model_name,
            validation["y_raw"],
            val_pred_raw,
            validation["apparent_earth_raw"],
            validation["context_end_time_ns"],
            validation["target_time_ns"],
            horizons,
        )

        save_predictions(
            prediction_dir
            / (
                f"test_predictions_"
                f"{model_name.replace('-', '_')}.npz"
            ),
            model_name,
            test["y_raw"],
            test_pred_raw,
            test["apparent_earth_raw"],
            test["context_end_time_ns"],
            test["target_time_ns"],
            horizons,
        )

        val_summary = summarize_metrics(
            val_metrics
        ).iloc[0]
        test_summary = summarize_metrics(
            test_metrics
        ).iloc[0]

        final_rows.append({
            "model": model_name,
            "seed": final_seed,
            "parameters": n_params,
            "selected_recurrent_units": int(
                cfg["recurrent_units"]
            ),
            "selected_recurrent_layers": int(
                cfg["recurrent_layers"]
            ),
            "selected_dropout": float(
                cfg["dropout"]
            ),
            "selected_learning_rate": float(
                cfg["learning_rate"]
            ),
            "selected_cnn_filters": int(
                cfg["cnn_filters"]
            ),
            "selected_cnn_kernel_size": int(
                cfg["cnn_kernel_size"]
            ),
            "selected_cnn_layers": int(
                cfg["cnn_layers"]
            ),
            "search_inner_val_best_AW_RMSE_mps": float(
                cfg[
                    "inner_val_best_apparent_vector_RMSE_mps"
                ]
            ),
            "final_best_epoch": int(
                result["best_epoch"]
            ),
            "final_epochs_ran": int(
                result["epochs_ran"]
            ),
            "final_validation_AW_RMSE_mps": float(
                val_summary[
                    "mean_apparent_vector_RMSE_mps"
                ]
            ),
            "final_validation_AWS_RMSE_mps": float(
                val_summary[
                    "mean_AWS_RMSE_mps"
                ]
            ),
            "final_validation_AWA_MAE_deg": float(
                val_summary[
                    "mean_AWA_MAE_deg"
                ]
            ),
            "test_AW_RMSE_mps": float(
                test_summary[
                    "mean_apparent_vector_RMSE_mps"
                ]
            ),
            "test_AWS_RMSE_mps": float(
                test_summary[
                    "mean_AWS_RMSE_mps"
                ]
            ),
            "test_AWA_MAE_deg": float(
                test_summary[
                    "mean_AWA_MAE_deg"
                ]
            ),
            "training_seconds": float(
                result["training_seconds"]
            ),
        })

        log(
            f"[FINAL DONE] {model_name}: "
            f"validation AW="
            f"{val_summary['mean_apparent_vector_RMSE_mps']:.4f} m/s | "
            f"test AW="
            f"{test_summary['mean_apparent_vector_RMSE_mps']:.4f} m/s | "
            f"test AWS="
            f"{test_summary['mean_AWS_RMSE_mps']:.4f} m/s | "
            f"test AWA="
            f"{test_summary['mean_AWA_MAE_deg']:.2f} deg"
        )

        del (
            model,
            val_pred_z,
            test_pred_z,
        )
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    final_df = pd.DataFrame(
        final_rows
    )
    final_df.to_csv(
        output_dir
        / "final_retrain_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------------------
    # Stage 5: build baseline comparison tables
    # ---------------------------------------------------------------------
    log("[STAGE 5/8] Building Persistence/Ridge/tuned-deep comparison")

    persistence_val = persistence_joint_prediction(
        validation["X"],
        feature_names,
        feature_mean,
        feature_std,
        len(horizons),
    )
    persistence_test = persistence_joint_prediction(
        test["X"],
        feature_names,
        feature_mean,
        feature_std,
        len(horizons),
    )

    persistence_val_metrics = evaluate_model(
        validation["y_raw"],
        persistence_val,
        validation["apparent_earth_raw"],
        horizons,
        "validation",
        "Persistence",
        args.direction_min_speed,
        args.cog_min_speed,
        None,
    )
    persistence_test_metrics = evaluate_model(
        test["y_raw"],
        persistence_test,
        test["apparent_earth_raw"],
        horizons,
        "test",
        "Persistence",
        args.direction_min_speed,
        args.cog_min_speed,
        None,
    )

    metric_frames = [
        persistence_val_metrics,
        persistence_test_metrics,
        *deep_metric_frames,
    ]

    ridge_merged = False
    ridge_dir = None

    if not args.skip_ridge_merge:
        ridge_dir = (
            Path(args.ridge_baseline_dir)
            if args.ridge_baseline_dir
            else (
                dataset_dir
                / "joint_baseline_v0_1"
            )
        )

        ridge_path = (
            ridge_dir
            / "metrics_per_horizon.csv"
        )

        if ridge_path.exists():
            ridge_df = pd.read_csv(
                ridge_path
            )
            ridge_df = ridge_df.loc[
                ridge_df["model"] == "Ridge"
            ].copy()

            required_cols = set(
                persistence_val_metrics.columns
            )

            if required_cols.issubset(
                ridge_df.columns
            ):
                ridge_df = ridge_df[
                    list(
                        persistence_val_metrics.columns
                    )
                ]
                metric_frames.append(
                    ridge_df
                )
                ridge_merged = True
                log(
                    f"[OK] Merged Ridge: {ridge_path}"
                )
            else:
                log(
                    "[WARNING] Ridge metrics schema mismatch; "
                    "not merged."
                )

    metrics = pd.concat(
        metric_frames,
        ignore_index=True,
    )

    order_map = {
        name: i
        for i, name in enumerate(
            MODEL_ORDER
        )
    }

    metrics["_order"] = (
        metrics["model"]
        .map(order_map)
        .fillna(999)
    )
    metrics = (
        metrics.sort_values(
            ["split", "_order", "horizon_min"]
        )
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    metrics.to_csv(
        output_dir
        / "tuned_metrics_per_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = summarize_metrics(
        metrics
    )
    summary["_order"] = (
        summary["model"]
        .map(order_map)
        .fillna(999)
    )
    summary = (
        summary.sort_values(
            ["split", "_order"]
        )
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    summary.to_csv(
        output_dir
        / "tuned_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------------------
    # Stage 6: model selection on original validation
    # ---------------------------------------------------------------------
    log("[STAGE 6/8] Selecting best tuned deep baseline on original validation")

    deep_val_summary = summary.loc[
        (
            summary["split"] == "validation"
        )
        & (
            summary["model"].isin(
                requested_models
            )
        )
    ].copy()

    best_final_row = (
        deep_val_summary.sort_values(
            "mean_apparent_vector_RMSE_mps"
        )
        .iloc[0]
    )

    best_final_model = str(
        best_final_row["model"]
    )
    best_final_validation_score = float(
        best_final_row[
            "mean_apparent_vector_RMSE_mps"
        ]
    )

    log(
        f"[SELECTED TUNED DEEP BASELINE] "
        f"{best_final_model} | "
        f"validation AW RMSE="
        f"{best_final_validation_score:.6f} m/s"
    )

    # ---------------------------------------------------------------------
    # Stage 7: manifest/report
    # ---------------------------------------------------------------------
    log("[STAGE 7/8] Writing manifest and report")

    tuning_manifest = {
        "script_version": SCRIPT_VERSION,
        "backend": "PyTorch",
        "source_dataset_version": manifest.get(
            "dataset_version"
        ),
        "task_name": EXPECTED_TASK_NAME,
        "dataset_dir": str(
            dataset_dir.resolve()
        ),
        "output_dir": str(
            output_dir.resolve()
        ),
        "models": requested_models,
        "search_protocol": {
            "hyperparameter_search_uses_original_train_only": True,
            "inner_split_type": "chronological holdout",
            "inner_val_fraction": float(
                args.inner_val_fraction
            ),
            "inner_train_samples": int(
                inner["n_inner_train"]
            ),
            "inner_val_samples": int(
                inner["n_inner_val"]
            ),
            "trials_per_model_requested": int(
                args.trials_per_model
            ),
            "search_max_epochs": int(
                args.search_epochs
            ),
            "search_patience": int(
                args.search_patience
            ),
            "search_objective": (
                "mean apparent-wind vector RMSE "
                "over 1/2/3/5/10-min horizons"
            ),
            "anchor_05B_config_always_included": True,
        },
        "final_refit_protocol": {
            "train_data": "full original train split",
            "early_stopping_data": (
                "original validation split"
            ),
            "test_used_during_training_or_tuning": False,
            "final_max_epochs": int(
                args.final_epochs
            ),
            "final_patience": int(
                args.final_patience
            ),
        },
        "search_space": {
            "recurrent_units": [32, 64, 128],
            "recurrent_layers": [1, 2],
            "dropout": [0.0, 0.1, 0.2],
            "learning_rate": [3e-4, 1e-3],
            "cnn_filters": [16, 32, 64],
            "cnn_kernel_size": [3, 5],
            "cnn_layers": [1, 2],
        },
        "best_hyperparameters": best_hparams,
        "best_tuned_deep_model_selected_on_original_validation": (
            best_final_model
        ),
        "best_tuned_deep_validation_AW_RMSE_mps": (
            best_final_validation_score
        ),
        "device_info": device_info,
        "ridge_metrics_merged": bool(
            ridge_merged
        ),
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
        / "tuning_manifest.json",
        tuning_manifest,
    )

    with (
        output_dir
        / "tuning_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "05C tuned joint deep baselines - PyTorch\n"
        )
        f.write("=" * 100 + "\n\n")
        f.write(
            "Hyperparameter search protocol:\n"
        )
        f.write(
            "- Search uses ONLY original training split.\n"
        )
        f.write(
            "- Original train is chronologically divided "
            "into inner_train and inner_val.\n"
        )
        f.write(
            "- Trials are ranked by inner_val mean "
            "apparent-wind vector RMSE.\n"
        )
        f.write(
            "- Selected hyperparameters are then refit on "
            "the full original train split.\n"
        )
        f.write(
            "- Original validation is used only for final "
            "checkpoint/early stopping.\n"
        )
        f.write(
            "- Original test is untouched until final model "
            "freezing.\n\n"
        )

        f.write(
            f"Inner train samples: "
            f"{inner['n_inner_train']}\n"
        )
        f.write(
            f"Inner val samples: "
            f"{inner['n_inner_val']}\n\n"
        )

        f.write(
            "Best hyperparameters\n"
        )
        f.write("-" * 100 + "\n")
        f.write(
            json.dumps(
                best_hparams,
                ensure_ascii=False,
                indent=2,
            )
        )
        f.write("\n\n")

        f.write(
            "Final retrain summary\n"
        )
        f.write("-" * 100 + "\n")
        f.write(
            final_df.to_string(
                index=False
            )
        )
        f.write("\n\n")

        f.write(
            "Validation summary\n"
        )
        f.write("-" * 100 + "\n")
        f.write(
            summary.loc[
                summary["split"] == "validation"
            ].to_string(
                index=False
            )
        )
        f.write("\n\n")

        f.write(
            "Test summary\n"
        )
        f.write("-" * 100 + "\n")
        f.write(
            summary.loc[
                summary["split"] == "test"
            ].to_string(
                index=False
            )
        )
        f.write("\n\n")

        f.write(
            f"Selected tuned deep baseline by original "
            f"validation: {best_final_model}\n"
        )

    # ---------------------------------------------------------------------
    # Stage 8: plots / console summary
    # ---------------------------------------------------------------------
    log("[STAGE 8/8] Finalization")

    if args.plots:
        make_plots(
            trials_df,
            metrics,
            output_dir,
        )
        log("[OK] Figures complete")

    # Remove trial checkpoints after all results are safe.
    for p in trial_ckpt_dir.glob("*.pt"):
        try:
            p.unlink()
        except OSError:
            pass

    try:
        trial_ckpt_dir.rmdir()
    except OSError:
        pass

    log("")
    log("=" * 100)
    log("TUNED DEEP-BASELINE RESULTS")
    log("=" * 100)

    for model_name in requested_models:
        row = final_df.loc[
            final_df["model"] == model_name
        ].iloc[0]

        log(f"{model_name}:")
        log(
            f"  selected units/layers     = "
            f"{int(row['selected_recurrent_units'])}/"
            f"{int(row['selected_recurrent_layers'])}"
        )
        log(
            f"  selected dropout          = "
            f"{row['selected_dropout']:.2f}"
        )
        log(
            f"  selected learning rate    = "
            f"{row['selected_learning_rate']:.1e}"
        )

        if model_name == "CNN-LSTM":
            log(
                f"  selected CNN              = "
                f"filters={int(row['selected_cnn_filters'])}, "
                f"kernel={int(row['selected_cnn_kernel_size'])}, "
                f"layers={int(row['selected_cnn_layers'])}"
            )

        log(
            f"  inner-val search AW RMSE  = "
            f"{row['search_inner_val_best_AW_RMSE_mps']:.4f} m/s"
        )
        log(
            f"  final validation AW RMSE  = "
            f"{row['final_validation_AW_RMSE_mps']:.4f} m/s"
        )
        log(
            f"  test AW RMSE              = "
            f"{row['test_AW_RMSE_mps']:.4f} m/s"
        )
        log(
            f"  test AWS RMSE             = "
            f"{row['test_AWS_RMSE_mps']:.4f} m/s"
        )
        log(
            f"  test AWA MAE              = "
            f"{row['test_AWA_MAE_deg']:.2f} deg"
        )

    log("")
    log(
        f"[SELECTED ON ORIGINAL VALIDATION] "
        f"best tuned deep baseline = "
        f"{best_final_model}"
    )
    log(f"[DEVICE] {device}")

    if device.type == "cuda":
        log(
            f"[GPU] "
            f"{torch.cuda.get_device_name(device)}"
        )

    log(
        f"[DONE] Tuning outputs: {output_dir}"
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
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
        sys.exit(1)
