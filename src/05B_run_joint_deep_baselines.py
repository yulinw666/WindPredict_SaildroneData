# -*- coding: utf-8 -*-
"""
05B_run_joint_deep_baselines.py

Deep-learning baselines for the validated Saildrone joint wind-vessel
forecasting dataset.

Models
------
1. LSTM
2. GRU
3. CNN-LSTM

All three models use the EXACT SAME:
    - chronological train / validation / test split
    - 60-min historical window
    - 11 input features
    - 6 joint future targets
    - 1/2/3/5/10-min horizons
    - TRAIN-only standardization
    - evaluation metrics used by 04B

Deep-baseline training objective
--------------------------------
The deep baselines intentionally use a simple standardized joint MSE:

    L = mean((y_pred_z - y_true_z)^2)

This keeps them conventional and fair. The physics-constrained / multitask
loss should be introduced later as the proposed method, not hidden inside the
baseline models.

Model-selection / early-stopping criterion
------------------------------------------
The primary validation criterion is:

    mean apparent-wind vector RMSE across all forecast horizons

where apparent wind is reconstructed by the physical relation:

    A_E = U_w - V_s,E
    A_N = V_w - V_s,N

A custom Keras callback evaluates this criterion after every epoch, stores the
best weights, and performs early stopping. The TEST split is never used for
training, early stopping, architecture selection, or hyperparameter selection.

Target channels
---------------
[UWND_MEAN,
 VWND_MEAN,
 VESSEL_EAST_MPS,
 VESSEL_NORTH_MPS,
 HDG_sin,
 HDG_cos]

Historical input channels
-------------------------
[UWND_MEAN,
 VWND_MEAN,
 TEMP_AIR_MEAN,
 RH_MEAN,
 SOG,
 COG_sin,
 COG_cos,
 HDG_sin,
 HDG_cos,
 WING_ANGLE_sin,
 WING_ANGLE_cos]

Metrics
-------
The script evaluates:
    Earth-relative wind:
        vector RMSE/MAE, speed RMSE/MAE, direction circular MAE/RMSE
    vessel motion:
        vector RMSE/MAE, SOG RMSE/MAE, COG circular MAE/RMSE
    heading:
        HDG circular MAE/RMSE, predicted sin/cos unit-norm diagnostic
    apparent wind:
        vector RMSE/MAE, AWS RMSE/MAE, AWA circular MAE/RMSE
    skill:
        relative to Persistence

Optional Ridge merge
--------------------
If 04B outputs are found at:
    <dataset-dir>/joint_baseline_v0_1/metrics_per_horizon.csv

they are automatically merged into the comparison tables/plots. This does NOT
retrain or reselect Ridge and therefore preserves the untouched TEST protocol.

Outputs
-------
deep_baseline_manifest.json
deep_baseline_report.txt
metrics_per_horizon.csv
metrics_summary.csv
training_summary.csv
training_history_<model>.csv
validation_predictions_<model>.npz
test_predictions_<model>.npz
models/
    LSTM.keras
    GRU.keras
    CNN-LSTM.keras
figures/
    01_test_apparent_vector_rmse.png/pdf
    02_test_aws_rmse.png/pdf
    03_test_awa_mae.png/pdf
    04_test_wind_vector_rmse.png/pdf
    05_test_vessel_vector_rmse.png/pdf
    06_validation_selection_metric.png/pdf
    07_test_10min_aws_timeseries.png/pdf
    08_test_10min_awa_timeseries.png/pdf
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
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


# ===========================================================================
# Basic I/O
# ===========================================================================

def log(message: str = "") -> None:
    print(message, flush=True)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        dataset_dir.glob(
            "validation*/VALIDATION_PASSED.flag"
        )
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
            key
            for key in required
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

    y_raw = (
        y_z
        * target_std[None, None, :]
        + target_mean[None, None, :]
    )

    return y_raw.astype(np.float32)


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

    index = {
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

    u = last_raw[:, index["UWND_MEAN"]]
    v = last_raw[:, index["VWND_MEAN"]]
    sog = last_raw[:, index["SOG"]]
    cog_sin = last_raw[:, index["COG_sin"]]
    cog_cos = last_raw[:, index["COG_cos"]]
    hdg_sin = last_raw[:, index["HDG_sin"]]
    hdg_cos = last_raw[:, index["HDG_cos"]]

    vessel_east = sog * cog_sin
    vessel_north = sog * cog_cos

    state = np.column_stack(
        [
            u,
            v,
            vessel_east,
            vessel_north,
            hdg_sin,
            hdg_cos,
        ]
    )

    return np.repeat(
        state[:, None, :],
        repeats=n_horizons,
        axis=1,
    ).astype(np.float32)


# ===========================================================================
# Physics / angle helpers
# ===========================================================================

def wrap_deg(angle_deg: np.ndarray) -> np.ndarray:
    return (
        angle_deg
        + 180.0
    ) % 360.0 - 180.0


def circular_error_deg(
    pred_deg: np.ndarray,
    true_deg: np.ndarray,
) -> np.ndarray:
    return wrap_deg(
        pred_deg
        - true_deg
    )


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
        0 deg = from North
        90 deg = from East
    """
    angle = np.degrees(
        np.arctan2(
            -u_east.astype(np.float64),
            -v_north.astype(np.float64),
        )
    )

    return (
        angle
        + 360.0
    ) % 360.0


def cog_from_en_deg(
    east: np.ndarray,
    north: np.ndarray,
) -> np.ndarray:
    angle = np.degrees(
        np.arctan2(
            east.astype(np.float64),
            north.astype(np.float64),
        )
    )

    return (
        angle
        + 360.0
    ) % 360.0


def heading_from_sincos_deg(
    hdg_sin: np.ndarray,
    hdg_cos: np.ndarray,
) -> np.ndarray:
    angle = np.degrees(
        np.arctan2(
            hdg_sin.astype(np.float64),
            hdg_cos.astype(np.float64),
        )
    )

    return (
        angle
        + 360.0
    ) % 360.0


def apparent_earth_from_joint(
    joint_raw: np.ndarray,
) -> np.ndarray:
    app = np.empty(
        joint_raw.shape[:-1]
        + (2,),
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
    Body-frame apparent WIND-FROM angle:
        0 deg = directly ahead
        +90 deg = from starboard
        -90 deg = from port
        +/-180 deg = from astern
    """
    a_e = apparent_east.astype(np.float64)
    a_n = apparent_north.astype(np.float64)

    s = hdg_sin.astype(np.float64)
    c = hdg_cos.astype(np.float64)

    norm = np.sqrt(
        s ** 2
        + c ** 2
    )

    safe = (
        norm
        > 1e-12
    )

    s = np.where(
        safe,
        s / norm,
        0.0,
    )
    c = np.where(
        safe,
        c / norm,
        1.0,
    )

    relative_forward = (
        a_e * s
        + a_n * c
    )

    relative_starboard = (
        a_e * c
        - a_n * s
    )

    wind_from_forward = (
        -relative_forward
    )
    wind_from_starboard = (
        -relative_starboard
    )

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

    ss_res = float(
        np.sum(
            (
                truth
                - pred
            ) ** 2
        )
    )

    centered = (
        truth
        - np.mean(
            truth
        )
    )

    ss_tot = float(
        np.sum(
            centered ** 2
        )
    )

    if ss_tot <= 0.0:
        return float("nan")

    return (
        1.0
        - ss_res
        / ss_tot
    )


def masked_circular_metrics(
    pred_deg: np.ndarray,
    true_deg: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float, int]:
    valid = (
        np.asarray(
            mask,
            dtype=bool,
        )
        & np.isfinite(
            pred_deg
        )
        & np.isfinite(
            true_deg
        )
    )

    n = int(
        valid.sum()
    )

    if n == 0:
        return (
            float("nan"),
            float("nan"),
            0,
        )

    err = circular_error_deg(
        pred_deg[valid],
        true_deg[valid],
    )

    return (
        float(
            np.mean(
                np.abs(
                    err
                )
            )
        ),
        float(
            np.sqrt(
                np.mean(
                    err ** 2
                )
            )
        ),
        n,
    )


def unit_norm_error(
    sin_values: np.ndarray,
    cos_values: np.ndarray,
) -> float:
    norm = np.sqrt(
        sin_values.astype(np.float64) ** 2
        + cos_values.astype(np.float64) ** 2
    )

    return float(
        np.mean(
            np.abs(
                norm
                - 1.0
            )
        )
    )


# ===========================================================================
# Evaluation
# ===========================================================================

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
    if (
        truth_joint.shape
        != pred_joint.shape
    ):
        raise ValueError(
            "truth/pred shape mismatch: "
            f"{truth_joint.shape} vs {pred_joint.shape}"
        )

    pred_apparent = apparent_earth_from_joint(
        pred_joint
    )

    if persistence_joint is not None:
        persistence_apparent = apparent_earth_from_joint(
            persistence_joint
        )
    else:
        persistence_apparent = None

    rows = []

    for j, horizon in enumerate(
        horizons
    ):
        truth = truth_joint[
            :,
            j,
            :,
        ].astype(
            np.float64,
            copy=False,
        )

        pred = pred_joint[
            :,
            j,
            :,
        ].astype(
            np.float64,
            copy=False,
        )

        true_wind = truth[
            :,
            0:2,
        ]
        pred_wind = pred[
            :,
            0:2,
        ]

        true_vessel = truth[
            :,
            2:4,
        ]
        pred_vessel = pred[
            :,
            2:4,
        ]

        # Wind vector
        wind_err = (
            pred_wind
            - true_wind
        )

        du = wind_err[
            :,
            0,
        ]
        dv = wind_err[
            :,
            1,
        ]

        wind_u_rmse = float(
            np.sqrt(
                np.mean(
                    du ** 2
                )
            )
        )
        wind_v_rmse = float(
            np.sqrt(
                np.mean(
                    dv ** 2
                )
            )
        )
        wind_vector_rmse = float(
            np.sqrt(
                np.mean(
                    du ** 2
                    + dv ** 2
                )
            )
        )
        wind_vector_mae = float(
            np.mean(
                np.sqrt(
                    du ** 2
                    + dv ** 2
                )
            )
        )

        true_ws = vector_speed(
            true_wind[
                :,
                0,
            ],
            true_wind[
                :,
                1,
            ],
        )
        pred_ws = vector_speed(
            pred_wind[
                :,
                0,
            ],
            pred_wind[
                :,
                1,
            ],
        )

        ws_err = (
            pred_ws
            - true_ws
        )

        wind_speed_rmse = float(
            np.sqrt(
                np.mean(
                    ws_err ** 2
                )
            )
        )
        wind_speed_mae = float(
            np.mean(
                np.abs(
                    ws_err
                )
            )
        )

        true_wd = direction_from_uv_deg(
            true_wind[
                :,
                0,
            ],
            true_wind[
                :,
                1,
            ],
        )
        pred_wd = direction_from_uv_deg(
            pred_wind[
                :,
                0,
            ],
            pred_wind[
                :,
                1,
            ],
        )

        (
            wind_dir_mae,
            wind_dir_rmse,
            n_wind_dir,
        ) = masked_circular_metrics(
            pred_deg=pred_wd,
            true_deg=true_wd,
            mask=(
                true_ws
                >= direction_min_speed
            ),
        )

        # Vessel
        vessel_err = (
            pred_vessel
            - true_vessel
        )

        de = vessel_err[
            :,
            0,
        ]
        dn = vessel_err[
            :,
            1,
        ]

        vessel_vector_rmse = float(
            np.sqrt(
                np.mean(
                    de ** 2
                    + dn ** 2
                )
            )
        )
        vessel_vector_mae = float(
            np.mean(
                np.sqrt(
                    de ** 2
                    + dn ** 2
                )
            )
        )

        true_sog = vector_speed(
            true_vessel[
                :,
                0,
            ],
            true_vessel[
                :,
                1,
            ],
        )
        pred_sog = vector_speed(
            pred_vessel[
                :,
                0,
            ],
            pred_vessel[
                :,
                1,
            ],
        )

        sog_err = (
            pred_sog
            - true_sog
        )

        sog_rmse = float(
            np.sqrt(
                np.mean(
                    sog_err ** 2
                )
            )
        )
        sog_mae = float(
            np.mean(
                np.abs(
                    sog_err
                )
            )
        )

        true_cog = cog_from_en_deg(
            true_vessel[
                :,
                0,
            ],
            true_vessel[
                :,
                1,
            ],
        )
        pred_cog = cog_from_en_deg(
            pred_vessel[
                :,
                0,
            ],
            pred_vessel[
                :,
                1,
            ],
        )

        (
            cog_mae,
            cog_rmse,
            n_cog,
        ) = masked_circular_metrics(
            pred_deg=pred_cog,
            true_deg=true_cog,
            mask=(
                true_sog
                >= cog_min_speed
            ),
        )

        # Heading
        true_hdg_sin = truth[
            :,
            4,
        ]
        true_hdg_cos = truth[
            :,
            5,
        ]
        pred_hdg_sin = pred[
            :,
            4,
        ]
        pred_hdg_cos = pred[
            :,
            5,
        ]

        true_hdg = heading_from_sincos_deg(
            true_hdg_sin,
            true_hdg_cos,
        )
        pred_hdg = heading_from_sincos_deg(
            pred_hdg_sin,
            pred_hdg_cos,
        )

        (
            hdg_mae,
            hdg_rmse,
            n_hdg,
        ) = masked_circular_metrics(
            pred_deg=pred_hdg,
            true_deg=true_hdg,
            mask=np.ones(
                len(
                    true_hdg
                ),
                dtype=bool,
            ),
        )

        hdg_unit_norm_mae = unit_norm_error(
            pred_hdg_sin,
            pred_hdg_cos,
        )

        # Apparent wind
        true_app = truth_apparent_ref[
            :,
            j,
            :,
        ].astype(
            np.float64,
            copy=False,
        )

        pred_app = pred_apparent[
            :,
            j,
            :,
        ]

        app_err = (
            pred_app
            - true_app
        )

        app_de = app_err[
            :,
            0,
        ]
        app_dn = app_err[
            :,
            1,
        ]

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
            true_app[
                :,
                0,
            ],
            true_app[
                :,
                1,
            ],
        )
        pred_aws = vector_speed(
            pred_app[
                :,
                0,
            ],
            pred_app[
                :,
                1,
            ],
        )

        aws_err = (
            pred_aws
            - true_aws
        )

        aws_rmse = float(
            np.sqrt(
                np.mean(
                    aws_err ** 2
                )
            )
        )
        aws_mae = float(
            np.mean(
                np.abs(
                    aws_err
                )
            )
        )

        true_awa = apparent_wind_angle_deg(
            apparent_east=true_app[
                :,
                0,
            ],
            apparent_north=true_app[
                :,
                1,
            ],
            hdg_sin=true_hdg_sin,
            hdg_cos=true_hdg_cos,
        )

        pred_awa = apparent_wind_angle_deg(
            apparent_east=pred_app[
                :,
                0,
            ],
            apparent_north=pred_app[
                :,
                1,
            ],
            hdg_sin=pred_hdg_sin,
            hdg_cos=pred_hdg_cos,
        )

        (
            awa_mae,
            awa_rmse,
            n_awa,
        ) = masked_circular_metrics(
            pred_deg=pred_awa,
            true_deg=true_awa,
            mask=(
                true_aws
                >= direction_min_speed
            ),
        )

        # Skills
        wind_skill = (
            0.0
            if model_name.lower()
            == "persistence"
            else float("nan")
        )
        vessel_skill = (
            0.0
            if model_name.lower()
            == "persistence"
            else float("nan")
        )
        app_skill = (
            0.0
            if model_name.lower()
            == "persistence"
            else float("nan")
        )
        aws_skill = (
            0.0
            if model_name.lower()
            == "persistence"
            else float("nan")
        )

        if persistence_joint is not None:
            ref = persistence_joint[
                :,
                j,
                :,
            ].astype(
                np.float64,
                copy=False,
            )

            ref_wind_err = (
                ref[
                    :,
                    0:2,
                ]
                - true_wind
            )

            ref_wind_rmse = float(
                np.sqrt(
                    np.mean(
                        ref_wind_err[
                            :,
                            0,
                        ] ** 2
                        + ref_wind_err[
                            :,
                            1,
                        ] ** 2
                    )
                )
            )

            ref_vessel_err = (
                ref[
                    :,
                    2:4,
                ]
                - true_vessel
            )

            ref_vessel_rmse = float(
                np.sqrt(
                    np.mean(
                        ref_vessel_err[
                            :,
                            0,
                        ] ** 2
                        + ref_vessel_err[
                            :,
                            1,
                        ] ** 2
                    )
                )
            )

            ref_app = persistence_apparent[
                :,
                j,
                :,
            ]

            ref_app_err = (
                ref_app
                - true_app
            )

            ref_app_rmse = float(
                np.sqrt(
                    np.mean(
                        ref_app_err[
                            :,
                            0,
                        ] ** 2
                        + ref_app_err[
                            :,
                            1,
                        ] ** 2
                    )
                )
            )

            ref_aws = vector_speed(
                ref_app[
                    :,
                    0,
                ],
                ref_app[
                    :,
                    1,
                ],
            )

            ref_aws_rmse = float(
                np.sqrt(
                    np.mean(
                        (
                            ref_aws
                            - true_aws
                        ) ** 2
                    )
                )
            )

            if ref_wind_rmse > 0.0:
                wind_skill = (
                    1.0
                    - wind_vector_rmse
                    / ref_wind_rmse
                )

            if ref_vessel_rmse > 0.0:
                vessel_skill = (
                    1.0
                    - vessel_vector_rmse
                    / ref_vessel_rmse
                )

            if ref_app_rmse > 0.0:
                app_skill = (
                    1.0
                    - apparent_vector_rmse
                    / ref_app_rmse
                )

            if ref_aws_rmse > 0.0:
                aws_skill = (
                    1.0
                    - aws_rmse
                    / ref_aws_rmse
                )

        rows.append({
            "split": split_name,
            "model": model_name,
            "horizon_min": int(
                horizon
            ),
            "n_samples": int(
                len(
                    truth
                )
            ),

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
                true_wind[
                    :,
                    0,
                ],
                pred_wind[
                    :,
                    0,
                ],
            ),
            "wind_R2_V": safe_r2(
                true_wind[
                    :,
                    1,
                ],
                pred_wind[
                    :,
                    1,
                ],
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

    return pd.DataFrame(
        rows
    )


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
        [
            "split",
            "model",
        ],
        sort=False,
    ):
        row = {
            "split": split_name,
            "model": model_name,
            "n_horizons": int(
                len(
                    grp
                )
            ),
        }

        for col in cols:
            row[
                f"mean_{col}"
            ] = float(
                grp[
                    col
                ].mean()
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ===========================================================================
# TensorFlow models / callback
# ===========================================================================

def import_tensorflow():
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required in the active WindDL environment."
        ) from exc

    return tf


def configure_reproducibility(
    tf,
    seed: int,
    deterministic: bool,
) -> None:
    random.seed(
        seed
    )
    np.random.seed(
        seed
    )
    tf.keras.utils.set_random_seed(
        seed
    )

    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
            log(
                "[OK] TensorFlow op determinism enabled."
            )
        except Exception as exc:
            log(
                "[WARNING] Could not enable full TensorFlow "
                f"determinism: {exc}"
            )


def build_model(
    tf,
    model_name: str,
    lookback: int,
    n_features: int,
    n_horizons: int,
    n_targets: int,
    recurrent_units: int,
    dense_units: int,
    dropout: float,
    cnn_filters: int,
    cnn_kernel_size: int,
    learning_rate: float,
):
    layers = tf.keras.layers

    inputs = layers.Input(
        shape=(
            lookback,
            n_features,
        ),
        name="history",
    )

    if model_name == "LSTM":
        x = layers.LSTM(
            recurrent_units,
            return_sequences=False,
            name="lstm",
        )(
            inputs
        )

    elif model_name == "GRU":
        x = layers.GRU(
            recurrent_units,
            return_sequences=False,
            name="gru",
        )(
            inputs
        )

    elif model_name == "CNN-LSTM":
        x = layers.Conv1D(
            filters=cnn_filters,
            kernel_size=cnn_kernel_size,
            padding="causal",
            activation="relu",
            name="causal_conv1d",
        )(
            inputs
        )

        x = layers.LSTM(
            recurrent_units,
            return_sequences=False,
            name="lstm",
        )(
            x
        )

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    if dropout > 0.0:
        x = layers.Dropout(
            dropout,
            name="dropout",
        )(
            x
        )

    x = layers.Dense(
        dense_units,
        activation="relu",
        name="dense_hidden",
    )(
        x
    )

    outputs = layers.Dense(
        n_horizons
        * n_targets,
        activation="linear",
        name="joint_output_flat",
    )(
        x
    )

    outputs = layers.Reshape(
        (
            n_horizons,
            n_targets,
        ),
        name="joint_output",
    )(
        outputs
    )

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=(
            model_name.replace(
                "-",
                "_",
            )
        ),
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(
                name="mae_z"
            )
        ],
    )

    return model


def make_apparent_callback_class(tf):
    class ApparentValidationEarlyStopping(
        tf.keras.callbacks.Callback
    ):
        """
        Evaluate mean validation apparent-wind vector RMSE after each epoch.

        The best model weights are saved according to this physical validation
        metric, not according to normalized joint MSE.
        """

        def __init__(
            self,
            X_val: np.ndarray,
            y_val_raw: np.ndarray,
            target_mean: np.ndarray,
            target_std: np.ndarray,
            batch_size: int,
            patience: int,
            min_delta: float,
            weights_path: Path,
        ):
            super().__init__()

            self.X_val = X_val
            self.y_val_raw = y_val_raw
            self.target_mean = (
                target_mean.astype(
                    np.float64
                )
            )
            self.target_std = (
                target_std.astype(
                    np.float64
                )
            )
            self.batch_size = int(
                batch_size
            )
            self.patience = int(
                patience
            )
            self.min_delta = float(
                min_delta
            )
            self.weights_path = str(
                weights_path
            )

            self.true_app = (
                apparent_earth_from_joint(
                    y_val_raw
                )
            )

            self.best = float(
                "inf"
            )
            self.best_epoch = 0
            self.wait = 0
            self.values = []

        def on_epoch_end(
            self,
            epoch,
            logs=None,
        ):
            logs = (
                logs
                if logs is not None
                else {}
            )

            pred_z = self.model.predict(
                self.X_val,
                batch_size=self.batch_size,
                verbose=0,
            )

            pred_raw = (
                pred_z.astype(
                    np.float64
                )
                * self.target_std[
                    None,
                    None,
                    :,
                ]
                + self.target_mean[
                    None,
                    None,
                    :,
                ]
            )

            pred_app = (
                apparent_earth_from_joint(
                    pred_raw
                )
            )

            err = (
                pred_app
                - self.true_app
            )

            # Mean of horizon-specific vector RMSE, matching our reporting.
            per_horizon = np.sqrt(
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

            score = float(
                np.mean(
                    per_horizon
                )
            )

            logs[
                "val_apparent_vector_RMSE_mps"
            ] = score

            self.values.append(
                score
            )

            improved = (
                score
                < (
                    self.best
                    - self.min_delta
                )
            )

            if improved:
                self.best = score
                self.best_epoch = (
                    epoch
                    + 1
                )
                self.wait = 0

                self.model.save_weights(
                    self.weights_path
                )

                log(
                    f"      [VAL-AW] epoch={epoch+1:03d} "
                    f"mean apparent RMSE={score:.6f} m/s "
                    f"[BEST]"
                )
            else:
                self.wait += 1

                log(
                    f"      [VAL-AW] epoch={epoch+1:03d} "
                    f"mean apparent RMSE={score:.6f} m/s "
                    f"wait={self.wait}/{self.patience}"
                )

                if (
                    self.wait
                    >= self.patience
                ):
                    self.model.stop_training = True

                    log(
                        "      [EARLY STOP] "
                        f"best epoch={self.best_epoch}, "
                        f"best validation apparent RMSE="
                        f"{self.best:.6f} m/s"
                    )

    return ApparentValidationEarlyStopping


# ===========================================================================
# Prediction export
# ===========================================================================

def derive_export_arrays(
    joint_raw: np.ndarray,
) -> dict[str, np.ndarray]:
    wind = joint_raw[
        ...,
        0:2,
    ].astype(
        np.float64,
        copy=False,
    )

    vessel = joint_raw[
        ...,
        2:4,
    ].astype(
        np.float64,
        copy=False,
    )

    hdg_sin = joint_raw[
        ...,
        4,
    ].astype(
        np.float64,
        copy=False,
    )

    hdg_cos = joint_raw[
        ...,
        5,
    ].astype(
        np.float64,
        copy=False,
    )

    apparent = apparent_earth_from_joint(
        joint_raw
    )

    return {
        "wind_uv": wind.astype(
            np.float32
        ),
        "wind_speed": vector_speed(
            wind[
                ...,
                0,
            ],
            wind[
                ...,
                1,
            ],
        ).astype(
            np.float32
        ),
        "wind_direction_from_deg": (
            direction_from_uv_deg(
                wind[
                    ...,
                    0,
                ],
                wind[
                    ...,
                    1,
                ],
            )
            .astype(
                np.float32
            )
        ),
        "vessel_en": vessel.astype(
            np.float32
        ),
        "SOG": vector_speed(
            vessel[
                ...,
                0,
            ],
            vessel[
                ...,
                1,
            ],
        ).astype(
            np.float32
        ),
        "COG_deg": cog_from_en_deg(
            vessel[
                ...,
                0,
            ],
            vessel[
                ...,
                1,
            ],
        ).astype(
            np.float32
        ),
        "HDG_deg": heading_from_sincos_deg(
            hdg_sin,
            hdg_cos,
        ).astype(
            np.float32
        ),
        "apparent_en": apparent.astype(
            np.float32
        ),
        "AWS": vector_speed(
            apparent[
                ...,
                0,
            ],
            apparent[
                ...,
                1,
            ],
        ).astype(
            np.float32
        ),
        "AWA_deg_starboard_positive": (
            apparent_wind_angle_deg(
                apparent_east=apparent[
                    ...,
                    0,
                ],
                apparent_north=apparent[
                    ...,
                    1,
                ],
                hdg_sin=hdg_sin,
                hdg_cos=hdg_cos,
            )
            .astype(
                np.float32
            )
        ),
    }


def save_model_predictions(
    path: Path,
    model_name: str,
    truth_joint: np.ndarray,
    pred_joint: np.ndarray,
    truth_apparent_ref: np.ndarray,
    context_end_time_ns: np.ndarray,
    target_time_ns: np.ndarray,
    horizons: tuple[int, ...],
) -> None:
    truth_d = derive_export_arrays(
        truth_joint
    )
    pred_d = derive_export_arrays(
        pred_joint
    )

    payload = {
        "model_name": np.asarray(
            model_name
        ),
        "truth_joint": np.asarray(
            truth_joint,
            dtype=np.float32,
        ),
        "pred_joint": np.asarray(
            pred_joint,
            dtype=np.float32,
        ),
        "truth_apparent_reference_en": np.asarray(
            truth_apparent_ref,
            dtype=np.float32,
        ),
        "context_end_time_ns": np.asarray(
            context_end_time_ns,
            dtype=np.int64,
        ),
        "target_time_ns": np.asarray(
            target_time_ns,
            dtype=np.int64,
        ),
        "horizons_min": np.asarray(
            horizons,
            dtype=np.int64,
        ),
    }

    for prefix, data in [
        (
            "truth",
            truth_d,
        ),
        (
            "pred",
            pred_d,
        ),
    ]:
        for key, value in data.items():
            payload[
                f"{prefix}_{key}"
            ] = value

    np.savez_compressed(
        path,
        **payload,
    )


# ===========================================================================
# Plotting
# ===========================================================================

def load_sci_style(script_dir: Path):
    style_path = (
        script_dir
        / "sci_plot_style.py"
    )

    if not style_path.exists():
        log(
            "[PLOT WARNING] sci_plot_style.py not found: "
            f"{style_path}"
        )
        return None

    spec = importlib.util.spec_from_file_location(
        "sci_plot_style",
        str(style_path),
    )

    if (
        spec is None
        or spec.loader is None
    ):
        return None

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        "sci_plot_style"
    ] = module

    spec.loader.exec_module(
        module
    )

    if hasattr(
        module,
        "apply_sci_style",
    ):
        try:
            selected = (
                module
                .apply_sci_style(
                    base_font_size=8.5
                )
            )
        except TypeError:
            try:
                selected = (
                    module
                    .apply_sci_style(
                        font_size=8.5
                    )
                )
            except TypeError:
                selected = (
                    module
                    .apply_sci_style()
                )

        log(
            "[PLOT] SCI style applied; "
            f"selected_font={selected}"
        )

    return module


def make_plots(
    metrics: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    test_truth: np.ndarray,
    test_predictions: dict[str, np.ndarray],
    test_target_time_ns: np.ndarray,
    horizons: tuple[int, ...],
    output_dir: Path,
    example_hours: float,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    style = load_sci_style(
        Path(__file__).resolve().parent
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
        elif (
            style is not None
            and hasattr(
                style,
                "finish_axes",
            )
        ):
            style.finish_axes(
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

    test_metrics = (
        metrics.loc[
            metrics[
                "split"
            ] == "test"
        ]
        .copy()
    )

    available_models = [
        m
        for m in MODEL_ORDER
        if m
        in set(
            test_metrics[
                "model"
            ]
        )
    ]

    def comparison_curve(
        metric_col: str,
        ylabel: str,
        filename: str,
    ):
        fig, ax = plt.subplots(
            figsize=(4.2, 3.0)
        )

        for model_name in available_models:
            grp = (
                test_metrics.loc[
                    test_metrics[
                        "model"
                    ] == model_name
                ]
                .sort_values(
                    "horizon_min"
                )
            )

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
            list(
                horizons
            )
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

    comparison_curve(
        "apparent_vector_RMSE_mps",
        r"Apparent-wind vector RMSE (m s$^{-1}$)",
        "01_test_apparent_vector_rmse",
    )

    comparison_curve(
        "AWS_RMSE_mps",
        r"AWS RMSE (m s$^{-1}$)",
        "02_test_aws_rmse",
    )

    comparison_curve(
        "AWA_MAE_deg",
        r"AWA MAE ($^\circ$)",
        "03_test_awa_mae",
    )

    comparison_curve(
        "wind_vector_RMSE_mps",
        r"Wind vector RMSE (m s$^{-1}$)",
        "04_test_wind_vector_rmse",
    )

    comparison_curve(
        "vessel_vector_RMSE_mps",
        r"Vessel vector RMSE (m s$^{-1}$)",
        "05_test_vessel_vector_rmse",
    )

    # Validation model-selection history
    fig, ax = plt.subplots(
        figsize=(4.2, 3.0)
    )

    for model_name, hist in histories.items():
        if (
            "val_apparent_vector_RMSE_mps"
            not in hist.columns
        ):
            continue

        ax.plot(
            hist[
                "epoch"
            ],
            hist[
                "val_apparent_vector_RMSE_mps"
            ],
            label=model_name,
        )

    ax.set_xlabel(
        "Epoch"
    )
    ax.set_ylabel(
        r"Validation apparent RMSE (m s$^{-1}$)"
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
        "06_validation_selection_metric",
    )

    # Example at longest horizon using best deep model by test table only for
    # visualization labeling. This does NOT feed back into training/selection.
    deep_names = [
        name
        for name in [
            "LSTM",
            "GRU",
            "CNN-LSTM",
        ]
        if name in test_predictions
    ]

    if not deep_names:
        return

    # For the figure, show all deep models plus observed.
    j = (
        len(
            horizons
        )
        - 1
    )

    truth_d = derive_export_arrays(
        test_truth
    )

    target_time = pd.to_datetime(
        test_target_time_ns[
            :,
            j,
        ],
        unit="ns",
        utc=True,
    )

    if len(
        target_time
    ) > 0:
        cutoff = (
            target_time[
                0
            ]
            + pd.Timedelta(
                hours=float(
                    example_hours
                )
            )
        )

        mask = (
            target_time
            <= cutoff
        )
    else:
        mask = np.zeros(
            len(
                target_time
            ),
            dtype=bool,
        )

    # AWS
    fig, ax = plt.subplots(
        figsize=(7.2, 3.0)
    )

    ax.plot(
        target_time[
            mask
        ],
        truth_d[
            "AWS"
        ][
            mask,
            j,
        ],
        label="Observed",
    )

    for model_name in deep_names:
        d = derive_export_arrays(
            test_predictions[
                model_name
            ]
        )

        ax.plot(
            target_time[
                mask
            ],
            d[
                "AWS"
            ][
                mask,
                j,
            ],
            label=model_name,
        )

    ax.set_xlabel(
        "Time"
    )
    ax.set_ylabel(
        r"AWS (m s$^{-1}$)"
    )
    ax.legend(
        loc="best",
        ncol=2,
    )

    locator = mdates.AutoDateLocator(
        minticks=4,
        maxticks=8,
    )
    ax.xaxis.set_major_locator(
        locator
    )
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            locator
        )
    )

    finish(
        ax,
        grid=True,
    )
    save(
        fig,
        (
            "07_test_"
            f"{horizons[j]}min_aws_timeseries"
        ),
    )

    # AWA
    fig, ax = plt.subplots(
        figsize=(7.2, 3.0)
    )

    ax.plot(
        target_time[
            mask
        ],
        truth_d[
            "AWA_deg_starboard_positive"
        ][
            mask,
            j,
        ],
        label="Observed",
    )

    for model_name in deep_names:
        d = derive_export_arrays(
            test_predictions[
                model_name
            ]
        )

        ax.plot(
            target_time[
                mask
            ],
            d[
                "AWA_deg_starboard_positive"
            ][
                mask,
                j,
            ],
            label=model_name,
        )

    ax.set_xlabel(
        "Time"
    )
    ax.set_ylabel(
        r"AWA ($^\circ$)"
    )
    ax.set_ylim(
        -180,
        180,
    )
    ax.legend(
        loc="best",
        ncol=2,
    )

    locator = mdates.AutoDateLocator(
        minticks=4,
        maxticks=8,
    )
    ax.xaxis.set_major_locator(
        locator
    )
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            locator
        )
    )

    finish(
        ax,
        grid=True,
    )
    save(
        fig,
        (
            "08_test_"
            f"{horizons[j]}min_awa_timeseries"
        ),
    )


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        required=True,
        help=(
            "Validated joint forecasting dataset directory."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output directory. Default: "
            "<dataset-dir>/deep_baseline_v0_1"
        ),
    )

    parser.add_argument(
        "--models",
        default="LSTM,GRU,CNN-LSTM",
        help=(
            "Comma-separated deep baselines. "
            "Allowed: LSTM,GRU,CNN-LSTM"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help=(
            "Maximum epochs per model. Default: 80."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help=(
            "Training/prediction batch size. Default: 256."
        ),
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help=(
            "Adam learning rate. Default: 1e-3."
        ),
    )

    parser.add_argument(
        "--recurrent-units",
        type=int,
        default=64,
        help=(
            "LSTM/GRU hidden units. Default: 64."
        ),
    )

    parser.add_argument(
        "--dense-units",
        type=int,
        default=64,
        help=(
            "Final hidden Dense units. Default: 64."
        ),
    )

    parser.add_argument(
        "--cnn-filters",
        type=int,
        default=32,
        help=(
            "CNN-LSTM Conv1D filters. Default: 32."
        ),
    )

    parser.add_argument(
        "--cnn-kernel-size",
        type=int,
        default=3,
        help=(
            "CNN-LSTM causal Conv1D kernel. Default: 3."
        ),
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.10,
        help=(
            "Dropout after recurrent encoder. Default: 0.10."
        ),
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help=(
            "Early-stopping patience on validation apparent RMSE. "
            "Default: 10 epochs."
        ),
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help=(
            "Minimum validation apparent-RMSE improvement. "
            "Default: 1e-4 m/s."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed. Default: 42."
        ),
    )

    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help=(
            "Disable TensorFlow deterministic operations."
        ),
    )

    parser.add_argument(
        "--direction-min-speed",
        type=float,
        default=0.5,
        help=(
            "Minimum true wind/AWS speed for direction metrics. "
            "Default: 0.5 m/s."
        ),
    )

    parser.add_argument(
        "--cog-min-speed",
        type=float,
        default=0.2,
        help=(
            "Minimum true SOG for COG metrics. Default: 0.2 m/s."
        ),
    )

    parser.add_argument(
        "--ridge-baseline-dir",
        default=None,
        help=(
            "Optional 04B output directory. Default: auto-detect "
            "<dataset-dir>/joint_baseline_v0_1"
        ),
    )

    parser.add_argument(
        "--skip-ridge-merge",
        action="store_true",
        help=(
            "Do not merge existing 04B Persistence/Ridge tables."
        ),
    )

    parser.add_argument(
        "--skip-validation-flag",
        action="store_true",
        help=(
            "Bypass VALIDATION_PASSED.flag gate. Not recommended."
        ),
    )

    parser.add_argument(
        "--example-hours",
        type=float,
        default=24.0,
        help=(
            "Time-series example duration. Default: 24 h."
        ),
    )

    parser.add_argument(
        "--plots",
        action="store_true",
        help=(
            "Generate SCI-style PNG/PDF figures."
        ),
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be positive."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    if args.recurrent_units <= 0:
        raise ValueError(
            "--recurrent-units must be positive."
        )

    if args.dense_units <= 0:
        raise ValueError(
            "--dense-units must be positive."
        )

    if not (
        0.0
        <= args.dropout
        < 1.0
    ):
        raise ValueError(
            "--dropout must satisfy 0 <= dropout < 1."
        )

    requested_models = []

    for token in args.models.split(","):
        name = token.strip()

        if not name:
            continue

        normalized = {
            "lstm": "LSTM",
            "gru": "GRU",
            "cnn-lstm": "CNN-LSTM",
            "cnn_lstm": "CNN-LSTM",
            "cnnlstm": "CNN-LSTM",
        }.get(
            name.lower()
        )

        if normalized is None:
            raise ValueError(
                f"Unsupported model '{name}'."
            )

        if normalized not in requested_models:
            requested_models.append(
                normalized
            )

    if not requested_models:
        raise ValueError(
            "No deep model selected."
        )

    dataset_dir = Path(
        args.dataset_dir
    )

    output_dir = (
        Path(
            args.output
        )
        if args.output is not None
        else (
            dataset_dir
            / "deep_baseline_v0_1"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_dir = (
        output_dir
        / "models"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("=" * 100)
    log(
        "SAILDRONE JOINT DEEP BASELINES"
    )
    log("=" * 100)
    log(
        f"script_version     : {SCRIPT_VERSION}"
    )
    log(
        f"dataset_dir        : {dataset_dir}"
    )
    log(
        f"output_dir         : {output_dir}"
    )
    log(
        f"models             : {requested_models}"
    )
    log(
        f"epochs             : {args.epochs}"
    )
    log(
        f"batch_size         : {args.batch_size}"
    )
    log(
        f"learning_rate      : {args.learning_rate}"
    )
    log(
        f"recurrent_units    : {args.recurrent_units}"
    )
    log(
        f"dense_units        : {args.dense_units}"
    )
    log(
        f"dropout            : {args.dropout}"
    )
    log(
        f"patience           : {args.patience}"
    )
    log(
        f"seed               : {args.seed}"
    )
    log(
        "selection metric   : validation mean apparent-wind vector RMSE"
    )
    log("")

    # ----------------------------------------------------------------------
    # Stage 1: gate + metadata
    # ----------------------------------------------------------------------
    log(
        "[STAGE 1/8] Validation gate and metadata"
    )

    validation_flags = find_validation_flags(
        dataset_dir
    )

    if (
        not validation_flags
        and not args.skip_validation_flag
    ):
        raise RuntimeError(
            "No VALIDATION_PASSED.flag found. "
            "Run 03B_validate_joint_forecasting_dataset.py first."
        )

    if validation_flags:
        log(
            "[OK] Validation flag(s):"
        )
        for flag in validation_flags:
            log(
                f"     {flag}"
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
            "Dataset task name is not the expected joint forecasting task."
        )

    lookback = int(
        manifest[
            "lookback_minutes"
        ]
    )

    horizons = tuple(
        int(v)
        for v in manifest[
            "forecast_horizons_minutes"
        ]
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

    if feature_names != EXPECTED_FEATURE_NAMES:
        raise RuntimeError(
            "Feature definition differs from frozen joint v0.1."
        )

    if target_names != EXPECTED_TARGET_NAMES:
        raise RuntimeError(
            "Target definition differs from frozen joint v0.1."
        )

    (
        f_names,
        feature_mean,
        feature_std,
    ) = parse_scaler(
        scalers,
        "feature_scaler",
    )

    (
        y_names,
        target_mean,
        target_std,
    ) = parse_scaler(
        scalers,
        "target_scaler",
    )

    if f_names != feature_names:
        raise RuntimeError(
            "Feature scaler names mismatch."
        )

    if y_names != target_names:
        raise RuntimeError(
            "Target scaler names mismatch."
        )

    # ----------------------------------------------------------------------
    # Stage 2: load data + TensorFlow
    # ----------------------------------------------------------------------
    log(
        "[STAGE 2/8] Loading train/validation/test and TensorFlow"
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

    X_train = train[
        "X"
    ]
    y_train = train[
        "y"
    ]

    X_val = validation[
        "X"
    ]
    y_val = validation[
        "y"
    ]
    y_val_raw = validation[
        "y_raw"
    ]
    app_val_ref = validation[
        "apparent_earth_raw"
    ]

    X_test = test[
        "X"
    ]
    y_test_raw = test[
        "y_raw"
    ]
    app_test_ref = test[
        "apparent_earth_raw"
    ]

    tf = import_tensorflow()

    configure_reproducibility(
        tf=tf,
        seed=args.seed,
        deterministic=(
            not args.no_deterministic
        ),
    )

    log(
        f"[OK] TensorFlow version: {tf.__version__}"
    )

    physical_devices = (
        tf.config.list_physical_devices()
    )

    log(
        "[INFO] TensorFlow devices: "
        + ", ".join(
            (
                f"{d.device_type}:{d.name}"
                for d in physical_devices
            )
        )
    )

    # Persistence reference used for all skill calculations.
    persistence_val = persistence_joint_prediction(
        X=X_val,
        feature_names=feature_names,
        feature_mean=feature_mean,
        feature_std=feature_std,
        n_horizons=len(
            horizons
        ),
    )

    persistence_test = persistence_joint_prediction(
        X=X_test,
        feature_names=feature_names,
        feature_mean=feature_mean,
        feature_std=feature_std,
        n_horizons=len(
            horizons
        ),
    )

    persistence_val_metrics = evaluate_model(
        truth_joint=y_val_raw,
        pred_joint=persistence_val,
        truth_apparent_ref=app_val_ref,
        horizons=horizons,
        split_name="validation",
        model_name="Persistence",
        direction_min_speed=args.direction_min_speed,
        cog_min_speed=args.cog_min_speed,
        persistence_joint=None,
    )

    persistence_test_metrics = evaluate_model(
        truth_joint=y_test_raw,
        pred_joint=persistence_test,
        truth_apparent_ref=app_test_ref,
        horizons=horizons,
        split_name="test",
        model_name="Persistence",
        direction_min_speed=args.direction_min_speed,
        cog_min_speed=args.cog_min_speed,
        persistence_joint=None,
    )

    # ----------------------------------------------------------------------
    # Stage 3: train deep baselines
    # ----------------------------------------------------------------------
    log(
        "[STAGE 3/8] Training LSTM / GRU / CNN-LSTM baselines"
    )

    CallbackClass = make_apparent_callback_class(
        tf
    )

    all_metric_frames = [
        persistence_val_metrics,
        persistence_test_metrics,
    ]

    histories: dict[str, pd.DataFrame] = {}
    test_predictions: dict[str, np.ndarray] = {}
    training_rows = []

    for model_index, model_name in enumerate(
        requested_models
    ):
        log("")
        log(
            "=" * 100
        )
        log(
            f"TRAINING {model_name}"
        )
        log(
            "=" * 100
        )

        # Reset Keras graph between models.
        tf.keras.backend.clear_session()

        # Use a deterministic but distinct seed for each baseline.
        model_seed = (
            args.seed
            + model_index
        )

        configure_reproducibility(
            tf=tf,
            seed=model_seed,
            deterministic=(
                not args.no_deterministic
            ),
        )

        model = build_model(
            tf=tf,
            model_name=model_name,
            lookback=lookback,
            n_features=len(
                feature_names
            ),
            n_horizons=len(
                horizons
            ),
            n_targets=len(
                target_names
            ),
            recurrent_units=args.recurrent_units,
            dense_units=args.dense_units,
            dropout=args.dropout,
            cnn_filters=args.cnn_filters,
            cnn_kernel_size=args.cnn_kernel_size,
            learning_rate=args.learning_rate,
        )

        parameter_count = int(
            model.count_params()
        )

        log(
            f"[MODEL] parameters={parameter_count:,}"
        )

        weights_path = (
            model_dir
            / (
                model_name.replace(
                    "-",
                    "_",
                )
                + "_best.weights.h5"
            )
        )

        selection_callback = CallbackClass(
            X_val=X_val,
            y_val_raw=y_val_raw,
            target_mean=target_mean,
            target_std=target_std,
            batch_size=args.batch_size,
            patience=args.patience,
            min_delta=args.min_delta,
            weights_path=weights_path,
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(
                2,
                args.patience
                // 3,
            ),
            min_lr=1e-5,
            verbose=1,
        )

        training_start = time.perf_counter()

        history_obj = model.fit(
            X_train,
            y_train,
            validation_data=(
                X_val,
                y_val,
            ),
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle=True,
            callbacks=[
                selection_callback,
                reduce_lr,
            ],
            verbose=2,
        )

        training_seconds = float(
            time.perf_counter()
            - training_start
        )

        # Restore weights selected by the primary physical validation metric.
        require_file(
            weights_path
        )

        model.load_weights(
            weights_path
        )

        best_epoch = int(
            selection_callback.best_epoch
        )

        best_val_apparent_rmse = float(
            selection_callback.best
        )

        # Merge Keras history and callback physical metric.
        n_epochs_ran = len(
            history_obj.history[
                "loss"
            ]
        )

        hist_data = {
            "epoch": np.arange(
                1,
                n_epochs_ran
                + 1,
                dtype=np.int64,
            ),
        }

        for key, values in history_obj.history.items():
            hist_data[
                key
            ] = values

        physical_values = (
            selection_callback.values
        )

        if len(
            physical_values
        ) != n_epochs_ran:
            raise RuntimeError(
                "Physical validation callback/history length mismatch."
            )

        hist_data[
            "val_apparent_vector_RMSE_mps"
        ] = physical_values

        history_df = pd.DataFrame(
            hist_data
        )

        histories[
            model_name
        ] = history_df

        history_file = (
            output_dir
            / (
                "training_history_"
                + model_name.replace(
                    "-",
                    "_",
                )
                + ".csv"
            )
        )

        history_df.to_csv(
            history_file,
            index=False,
            encoding="utf-8-sig",
        )

        # Validation prediction from best physical checkpoint.
        val_pred_z = model.predict(
            X_val,
            batch_size=args.batch_size,
            verbose=0,
        )

        val_pred_raw = inverse_joint_target(
            y_z=val_pred_z,
            target_mean=target_mean,
            target_std=target_std,
        )

        val_metrics = evaluate_model(
            truth_joint=y_val_raw,
            pred_joint=val_pred_raw,
            truth_apparent_ref=app_val_ref,
            horizons=horizons,
            split_name="validation",
            model_name=model_name,
            direction_min_speed=args.direction_min_speed,
            cog_min_speed=args.cog_min_speed,
            persistence_joint=persistence_val,
        )

        # Test remains untouched until the model is fully frozen.
        test_pred_z = model.predict(
            X_test,
            batch_size=args.batch_size,
            verbose=0,
        )

        test_pred_raw = inverse_joint_target(
            y_z=test_pred_z,
            target_mean=target_mean,
            target_std=target_std,
        )

        test_metrics = evaluate_model(
            truth_joint=y_test_raw,
            pred_joint=test_pred_raw,
            truth_apparent_ref=app_test_ref,
            horizons=horizons,
            split_name="test",
            model_name=model_name,
            direction_min_speed=args.direction_min_speed,
            cog_min_speed=args.cog_min_speed,
            persistence_joint=persistence_test,
        )

        all_metric_frames.extend(
            [
                val_metrics,
                test_metrics,
            ]
        )

        test_predictions[
            model_name
        ] = test_pred_raw

        save_model_predictions(
            path=(
                output_dir
                / (
                    "validation_predictions_"
                    + model_name.replace(
                        "-",
                        "_",
                    )
                    + ".npz"
                )
            ),
            model_name=model_name,
            truth_joint=y_val_raw,
            pred_joint=val_pred_raw,
            truth_apparent_ref=app_val_ref,
            context_end_time_ns=validation[
                "context_end_time_ns"
            ],
            target_time_ns=validation[
                "target_time_ns"
            ],
            horizons=horizons,
        )

        save_model_predictions(
            path=(
                output_dir
                / (
                    "test_predictions_"
                    + model_name.replace(
                        "-",
                        "_",
                    )
                    + ".npz"
                )
            ),
            model_name=model_name,
            truth_joint=y_test_raw,
            pred_joint=test_pred_raw,
            truth_apparent_ref=app_test_ref,
            context_end_time_ns=test[
                "context_end_time_ns"
            ],
            target_time_ns=test[
                "target_time_ns"
            ],
            horizons=horizons,
        )

        model_path = (
            model_dir
            / (
                model_name.replace(
                    "-",
                    "_",
                )
                + ".keras"
            )
        )

        model.save(
            model_path
        )

        val_summary = summarize_metrics(
            val_metrics
        ).iloc[0]

        test_summary = summarize_metrics(
            test_metrics
        ).iloc[0]

        training_rows.append({
            "model": model_name,
            "seed": model_seed,
            "parameters": parameter_count,
            "epochs_ran": n_epochs_ran,
            "best_epoch": best_epoch,
            "best_validation_apparent_vector_RMSE_mps": (
                best_val_apparent_rmse
            ),
            "training_seconds": training_seconds,
            "validation_mean_AWS_RMSE_mps": float(
                val_summary[
                    "mean_AWS_RMSE_mps"
                ]
            ),
            "validation_mean_AWA_MAE_deg": float(
                val_summary[
                    "mean_AWA_MAE_deg"
                ]
            ),
            "test_mean_apparent_vector_RMSE_mps": float(
                test_summary[
                    "mean_apparent_vector_RMSE_mps"
                ]
            ),
            "test_mean_AWS_RMSE_mps": float(
                test_summary[
                    "mean_AWS_RMSE_mps"
                ]
            ),
            "test_mean_AWA_MAE_deg": float(
                test_summary[
                    "mean_AWA_MAE_deg"
                ]
            ),
        })

        log(
            f"[DONE] {model_name}: "
            f"best_epoch={best_epoch}, "
            f"val apparent RMSE="
            f"{best_val_apparent_rmse:.4f} m/s, "
            f"test apparent RMSE="
            f"{test_summary['mean_apparent_vector_RMSE_mps']:.4f} m/s, "
            f"test AWS RMSE="
            f"{test_summary['mean_AWS_RMSE_mps']:.4f} m/s, "
            f"test AWA MAE="
            f"{test_summary['mean_AWA_MAE_deg']:.2f} deg"
        )

        # Release training graph before next model.
        del (
            model,
            val_pred_z,
            val_pred_raw,
            test_pred_z,
        )

        gc.collect()

    # ----------------------------------------------------------------------
    # Stage 4: optional 04B Ridge merge
    # ----------------------------------------------------------------------
    log("")
    log(
        "[STAGE 4/8] Merging optional 04B Ridge baseline"
    )

    ridge_merged = False
    ridge_dir = None

    if not args.skip_ridge_merge:
        if args.ridge_baseline_dir is not None:
            ridge_dir = Path(
                args.ridge_baseline_dir
            )
        else:
            ridge_dir = (
                dataset_dir
                / "joint_baseline_v0_1"
            )

        ridge_metrics_path = (
            ridge_dir
            / "metrics_per_horizon.csv"
        )

        if ridge_metrics_path.exists():
            ridge_df = pd.read_csv(
                ridge_metrics_path
            )

            # Keep Ridge only; Persistence is recomputed in this script.
            ridge_df = ridge_df.loc[
                ridge_df[
                    "model"
                ] == "Ridge"
            ].copy()

            required_cols = set(
                all_metric_frames[
                    0
                ].columns
            )

            if required_cols.issubset(
                ridge_df.columns
            ):
                ridge_df = ridge_df[
                    list(
                        all_metric_frames[
                            0
                        ].columns
                    )
                ]

                all_metric_frames.append(
                    ridge_df
                )
                ridge_merged = True

                log(
                    f"[OK] Merged Ridge metrics: "
                    f"{ridge_metrics_path}"
                )
            else:
                log(
                    "[WARNING] Existing 04B Ridge metrics use a "
                    "different schema; skipping merge."
                )
        else:
            log(
                "[INFO] No 04B Ridge metrics found; "
                "deep-baseline run continues."
            )

    # ----------------------------------------------------------------------
    # Stage 5: assemble tables
    # ----------------------------------------------------------------------
    log(
        "[STAGE 5/8] Saving metric and training tables"
    )

    metrics = pd.concat(
        all_metric_frames,
        ignore_index=True,
    )

    order_map = {
        name: i
        for i, name in enumerate(
            MODEL_ORDER
        )
    }

    metrics[
        "_model_order"
    ] = metrics[
        "model"
    ].map(
        order_map
    ).fillna(
        999
    )

    metrics = (
        metrics.sort_values(
            [
                "split",
                "_model_order",
                "horizon_min",
            ]
        )
        .drop(
            columns=[
                "_model_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    metrics.to_csv(
        output_dir
        / "metrics_per_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics_summary = summarize_metrics(
        metrics
    )

    metrics_summary[
        "_model_order"
    ] = metrics_summary[
        "model"
    ].map(
        order_map
    ).fillna(
        999
    )

    metrics_summary = (
        metrics_summary.sort_values(
            [
                "split",
                "_model_order",
            ]
        )
        .drop(
            columns=[
                "_model_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    metrics_summary.to_csv(
        output_dir
        / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    training_summary = pd.DataFrame(
        training_rows
    )

    training_summary.to_csv(
        output_dir
        / "training_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ----------------------------------------------------------------------
    # Stage 6: select best deep baseline on VALIDATION only
    # ----------------------------------------------------------------------
    log(
        "[STAGE 6/8] Selecting best deep baseline from validation only"
    )

    validation_deep = (
        metrics_summary.loc[
            (
                metrics_summary[
                    "split"
                ] == "validation"
            )
            & (
                metrics_summary[
                    "model"
                ].isin(
                    requested_models
                )
            )
        ]
        .copy()
    )

    if validation_deep.empty:
        raise RuntimeError(
            "No validation deep-model summaries were produced."
        )

    best_deep_row = (
        validation_deep.sort_values(
            "mean_apparent_vector_RMSE_mps"
        )
        .iloc[0]
    )

    best_deep_model = str(
        best_deep_row[
            "model"
        ]
    )

    best_deep_val_score = float(
        best_deep_row[
            "mean_apparent_vector_RMSE_mps"
        ]
    )

    log(
        f"[SELECTED DEEP BASELINE] {best_deep_model} "
        f"by validation apparent RMSE="
        f"{best_deep_val_score:.6f} m/s"
    )

    # ----------------------------------------------------------------------
    # Stage 7: manifest + report
    # ----------------------------------------------------------------------
    log(
        "[STAGE 7/8] Writing manifest and report"
    )

    tf_device_names = [
        f"{d.device_type}:{d.name}"
        for d in tf.config.list_physical_devices()
    ]

    out_manifest = {
        "script_version": SCRIPT_VERSION,
        "source_dataset_version": manifest.get(
            "dataset_version"
        ),
        "source_dataset_dir": str(
            dataset_dir.resolve()
        ),
        "task_name": EXPECTED_TASK_NAME,
        "validation_flags": [
            str(
                p.resolve()
            )
            for p in validation_flags
        ],
        "lookback_minutes": lookback,
        "forecast_horizons_minutes": list(
            horizons
        ),
        "feature_names": feature_names,
        "target_names": target_names,
        "models_trained": requested_models,
        "best_deep_baseline_selected_on_validation": (
            best_deep_model
        ),
        "primary_validation_metric": (
            "mean apparent-wind vector RMSE across horizons"
        ),
        "training_loss": (
            "standardized joint MSE over all horizons and 6 targets"
        ),
        "training_hyperparameters": {
            "max_epochs": int(
                args.epochs
            ),
            "batch_size": int(
                args.batch_size
            ),
            "learning_rate": float(
                args.learning_rate
            ),
            "recurrent_units": int(
                args.recurrent_units
            ),
            "dense_units": int(
                args.dense_units
            ),
            "cnn_filters": int(
                args.cnn_filters
            ),
            "cnn_kernel_size": int(
                args.cnn_kernel_size
            ),
            "dropout": float(
                args.dropout
            ),
            "patience": int(
                args.patience
            ),
            "min_delta_mps": float(
                args.min_delta
            ),
            "seed": int(
                args.seed
            ),
            "deterministic_ops": bool(
                not args.no_deterministic
            ),
        },
        "direction_metric_thresholds": {
            "wind_and_AWA_min_true_speed_mps": float(
                args.direction_min_speed
            ),
            "COG_min_true_SOG_mps": float(
                args.cog_min_speed
            ),
        },
        "ridge_metrics_merged": bool(
            ridge_merged
        ),
        "ridge_baseline_dir": (
            str(
                ridge_dir.resolve()
            )
            if (
                ridge_merged
                and ridge_dir is not None
            )
            else None
        ),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "tensorflow": tf.__version__,
            "tensorflow_devices": tf_device_names,
        },
    }

    with (
        output_dir
        / "deep_baseline_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            out_manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    test_summary = (
        metrics_summary.loc[
            metrics_summary[
                "split"
            ] == "test"
        ]
        .copy()
    )

    with (
        output_dir
        / "deep_baseline_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "Saildrone joint deep-baseline forecasting report\n"
        )
        f.write(
            "=" * 100
            + "\n\n"
        )
        f.write(
            f"Script version: {SCRIPT_VERSION}\n"
        )
        f.write(
            f"Dataset version: "
            f"{manifest.get('dataset_version')}\n"
        )
        f.write(
            f"Models: {requested_models}\n"
        )
        f.write(
            "Training loss: standardized joint MSE\n"
        )
        f.write(
            "Primary validation selection metric: "
            "mean apparent-wind vector RMSE\n"
        )
        f.write(
            f"Best deep baseline selected from validation: "
            f"{best_deep_model}\n\n"
        )

        f.write(
            "Training summary\n"
        )
        f.write(
            "-" * 100
            + "\n"
        )
        f.write(
            training_summary.to_string(
                index=False
            )
        )

        f.write(
            "\n\nValidation summary\n"
        )
        f.write(
            "-" * 100
            + "\n"
        )
        f.write(
            metrics_summary.loc[
                metrics_summary[
                    "split"
                ] == "validation"
            ].to_string(
                index=False
            )
        )

        f.write(
            "\n\nTest summary\n"
        )
        f.write(
            "-" * 100
            + "\n"
        )
        f.write(
            test_summary.to_string(
                index=False
            )
        )

        f.write(
            "\n\nTest metrics per horizon\n"
        )
        f.write(
            "-" * 100
            + "\n"
        )
        f.write(
            metrics.loc[
                metrics[
                    "split"
                ] == "test"
            ].to_string(
                index=False
            )
        )
        f.write("\n")

    # ----------------------------------------------------------------------
    # Stage 8: plots + concise console summary
    # ----------------------------------------------------------------------
    log(
        "[STAGE 8/8] Finalization"
    )

    if args.plots:
        log(
            "[PLOT] Generating SCI-style deep-baseline figures"
        )

        make_plots(
            metrics=metrics,
            histories=histories,
            test_truth=y_test_raw,
            test_predictions=test_predictions,
            test_target_time_ns=test[
                "target_time_ns"
            ],
            horizons=horizons,
            output_dir=output_dir,
            example_hours=args.example_hours,
        )

        log(
            "[OK] Figures complete"
        )

    log("")
    log("=" * 100)
    log(
        "TEST JOINT DEEP-BASELINE RESULTS"
    )
    log("=" * 100)

    for model_name in MODEL_ORDER:
        matched = test_summary.loc[
            test_summary[
                "model"
            ] == model_name
        ]

        if matched.empty:
            continue

        row = matched.iloc[
            0
        ]

        log(
            f"{model_name}:"
        )
        log(
            "  mean wind vector RMSE     = "
            f"{row['mean_wind_vector_RMSE_mps']:.4f} m/s"
        )
        log(
            "  mean vessel vector RMSE   = "
            f"{row['mean_vessel_vector_RMSE_mps']:.4f} m/s"
        )
        log(
            "  mean apparent vector RMSE = "
            f"{row['mean_apparent_vector_RMSE_mps']:.4f} m/s"
        )
        log(
            "  mean AWS RMSE             = "
            f"{row['mean_AWS_RMSE_mps']:.4f} m/s"
        )
        log(
            "  mean AWA MAE              = "
            f"{row['mean_AWA_MAE_deg']:.2f} deg"
        )
        log(
            "  mean HDG MAE              = "
            f"{row['mean_HDG_MAE_deg']:.2f} deg"
        )
        log(
            "  apparent skill vs Persist = "
            f"{row['mean_apparent_skill_vs_persistence']:.4f}"
        )

    log("")
    log(
        "[SELECTED ON VALIDATION] "
        f"best deep baseline = {best_deep_model}"
    )
    log(
        f"[DONE] Deep-baseline outputs: {output_dir}"
    )

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception:
        print(
            "\n[FATAL ERROR]",
            flush=True,
        )
        traceback.print_exc()
        print(
            "\nPlease send the complete traceback "
            "and terminal output.",
            flush=True,
        )
        sys.exit(1)

    sys.exit(
        exit_code
    )
