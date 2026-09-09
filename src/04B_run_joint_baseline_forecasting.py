# -*- coding: utf-8 -*-
"""
04B_run_joint_baseline_forecasting.py

Baseline experiments for the validated Saildrone joint wind-vessel forecasting
dataset produced by 02B_build_joint_forecasting_dataset.py.

Models
------
1. Persistence
   Future state = last observed state at the end of the historical context.

   Wind:
       U_hat(h) = U_t
       V_hat(h) = V_t

   Vessel motion:
       V_E_hat(h) = SOG_t * sin(COG_t)
       V_N_hat(h) = SOG_t * cos(COG_t)

   Heading:
       sin(HDG_hat(h)) = sin(HDG_t)
       cos(HDG_hat(h)) = cos(HDG_t)

2. Ridge
   The standardized historical sequence [lookback, features] is flattened and
   a multi-output Ridge regression predicts all standardized joint targets for
   all forecast horizons.

Hyperparameter selection
------------------------
Ridge alpha is selected ONLY on the validation split using:

    mean standardized joint MSE

All six target channels were standardized using TRAIN-only statistics in 02B,
so this criterion gives every joint target channel a comparable scale.

The selected model is NOT refit on validation data. It remains trained on TRAIN
only, keeping the protocol aligned with later deep-learning experiments where
validation is used only for model selection / early stopping.

Derived physical quantities
---------------------------
Joint target:
    [U_w, V_w, V_s,E, V_s,N, sin(HDG), cos(HDG)]

Earth-frame relative-air velocity:
    A_E = U_w - V_s,E
    A_N = V_w - V_s,N

Wind speed:
    WS = sqrt(U_w^2 + V_w^2)

Vessel speed over ground:
    SOG = sqrt(V_s,E^2 + V_s,N^2)

Earth-frame meteorological wind-from direction:
    WD_from = atan2(-U_w, -V_w), clockwise from North

COG:
    COG = atan2(V_s,E, V_s,N), clockwise from North

HDG:
    HDG = atan2(sin(HDG), cos(HDG)), clockwise from North

Apparent wind in body frame
---------------------------
Heading psi is clockwise from North.

Forward unit vector:
    f = [sin(psi), cos(psi)]     in [East, North]

Starboard unit vector:
    s = [cos(psi), -sin(psi)]

Relative-air vector components in body frame:
    A_forward   = A_E*sin(psi) + A_N*cos(psi)
    A_starboard = A_E*cos(psi) - A_N*sin(psi)

Wind-from vector is the negative of relative-air velocity:
    W_from_forward   = -A_forward
    W_from_starboard = -A_starboard

Apparent wind angle (AWA):
    AWA = atan2(W_from_starboard, W_from_forward)

Convention:
    AWA = 0 deg      -> wind from directly ahead
    AWA = +90 deg    -> wind from starboard
    AWA = -90 deg    -> wind from port
    AWA = +/-180 deg -> wind from astern

Angle errors are circular and wrapped to [-180, 180).

Metrics reported at 1/2/3/5/10 min
----------------------------------
Earth-relative wind:
    U/V RMSE and MAE
    vector RMSE and MAE
    wind-speed RMSE and MAE
    meteorological wind-direction circular MAE/RMSE
    skill vs Persistence

Vessel motion:
    East/North RMSE and MAE
    vessel-vector RMSE and MAE
    SOG RMSE and MAE
    COG circular MAE/RMSE
    skill vs Persistence

Heading:
    HDG circular MAE/RMSE
    predicted sin/cos unit-norm diagnostic

Apparent wind:
    East/North RMSE
    apparent-vector RMSE and MAE
    AWS RMSE and MAE
    AWA circular MAE/RMSE
    skill vs Persistence

Notes on direction metrics
--------------------------
Direction is physically ill-defined when speed is near zero. Therefore:
    - wind direction is evaluated only when true wind speed >= --direction-min-speed
    - AWA is evaluated only when true AWS >= --direction-min-speed
    - COG is evaluated only when true SOG >= --cog-min-speed

Outputs
-------
joint_baseline_manifest.json
ridge_alpha_search.csv
metrics_per_horizon.csv
metrics_summary.csv
validation_predictions.npz
test_predictions.npz
ridge_model.joblib
joint_baseline_report.txt
figures/*  when --plots is supplied
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE_VERSION = "0.1.0"
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

DEFAULT_ALPHA_GRID = (
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log(message: str = "") -> None:
    print(message, flush=True)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_float_grid(text: str) -> tuple[float, ...]:
    values = []

    for token in text.split(","):
        token = token.strip()

        if not token:
            continue

        value = float(token)

        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "Every Ridge alpha must be finite and >= 0."
            )

        values.append(value)

    if not values:
        raise ValueError(
            "At least one Ridge alpha is required."
        )

    return tuple(
        sorted(
            set(values)
        )
    )


def find_validation_flags(
    dataset_dir: Path,
) -> list[Path]:
    return sorted(
        dataset_dir.glob(
            "validation*/VALIDATION_PASSED.flag"
        )
    )


def parse_scaler(
    scalers: dict,
    key: str,
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
]:
    obj = scalers[key]

    return (
        list(obj["names"]),
        np.asarray(
            obj["mean"],
            dtype=np.float64,
        ),
        np.asarray(
            obj["std"],
            dtype=np.float64,
        ),
    )


def load_npz_split(
    dataset_dir: Path,
    split_name: str,
) -> dict[str, np.ndarray]:
    path = (
        dataset_dir
        / f"{split_name}.npz"
    )

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

    log(
        f"[LOAD] {path}"
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as npz:
        missing = [
            key
            for key in required
            if key not in npz.files
        ]

        if missing:
            raise RuntimeError(
                f"{path.name} missing arrays: "
                f"{missing}"
            )

        return {
            key: npz[key]
            for key in required
        }


def flatten_X(
    X: np.ndarray,
) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(
            f"Expected X [N,L,F], got {X.shape}"
        )

    return X.reshape(
        X.shape[0],
        -1,
    )


def flatten_y(
    y: np.ndarray,
) -> np.ndarray:
    if y.ndim != 3:
        raise ValueError(
            f"Expected y [N,H,T], got {y.shape}"
        )

    return y.reshape(
        y.shape[0],
        -1,
    )


def inverse_joint_target(
    y_z_flat: np.ndarray,
    n_horizons: int,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    n = y_z_flat.shape[0]

    y_z = y_z_flat.reshape(
        n,
        n_horizons,
        len(target_mean),
    ).astype(
        np.float64,
        copy=False,
    )

    y_raw = (
        y_z
        * target_std[
            None,
            None,
            :,
        ]
        + target_mean[
            None,
            None,
            :,
        ]
    )

    return y_raw.astype(
        np.float32
    )


def recover_last_raw_features(
    X: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    last_z = X[
        :,
        -1,
        :,
    ].astype(
        np.float64,
        copy=False,
    )

    return (
        last_z
        * feature_std[
            None,
            :,
        ]
        + feature_mean[
            None,
            :,
        ]
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

    u = last_raw[
        :,
        index["UWND_MEAN"],
    ]
    v = last_raw[
        :,
        index["VWND_MEAN"],
    ]
    sog = last_raw[
        :,
        index["SOG"],
    ]
    cog_sin = last_raw[
        :,
        index["COG_sin"],
    ]
    cog_cos = last_raw[
        :,
        index["COG_cos"],
    ]
    hdg_sin = last_raw[
        :,
        index["HDG_sin"],
    ]
    hdg_cos = last_raw[
        :,
        index["HDG_cos"],
    ]

    vessel_east = (
        sog
        * cog_sin
    )
    vessel_north = (
        sog
        * cog_cos
    )

    one_step_state = np.column_stack(
        [
            u,
            v,
            vessel_east,
            vessel_north,
            hdg_sin,
            hdg_cos,
        ]
    )

    pred = np.repeat(
        one_step_state[
            :,
            None,
            :,
        ],
        repeats=n_horizons,
        axis=1,
    )

    return pred.astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# Angle and physics helpers
# ---------------------------------------------------------------------------

def wrap_deg(
    angle_deg: np.ndarray,
) -> np.ndarray:
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
        east.astype(
            np.float64
        ) ** 2
        + north.astype(
            np.float64
        ) ** 2
    )


def direction_from_uv_deg(
    u_east: np.ndarray,
    v_north: np.ndarray,
) -> np.ndarray:
    """
    Meteorological wind-from direction:
        0 = from North
        90 = from East
        clockwise positive
    """
    angle = np.degrees(
        np.arctan2(
            -u_east.astype(
                np.float64
            ),
            -v_north.astype(
                np.float64
            ),
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
            east.astype(
                np.float64
            ),
            north.astype(
                np.float64
            ),
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
            hdg_sin.astype(
                np.float64
            ),
            hdg_cos.astype(
                np.float64
            ),
        )
    )

    return (
        angle
        + 360.0
    ) % 360.0


def apparent_earth_from_joint(
    joint_raw: np.ndarray,
) -> np.ndarray:
    """
    joint_raw [..., 6]:
        U, V, vessel_E, vessel_N, HDG_sin, HDG_cos
    """
    apparent = np.empty(
        joint_raw.shape[:-1]
        + (2,),
        dtype=np.float64,
    )

    apparent[
        ...,
        0,
    ] = (
        joint_raw[
            ...,
            0,
        ].astype(
            np.float64
        )
        - joint_raw[
            ...,
            2,
        ].astype(
            np.float64
        )
    )

    apparent[
        ...,
        1,
    ] = (
        joint_raw[
            ...,
            1,
        ].astype(
            np.float64
        )
        - joint_raw[
            ...,
            3,
        ].astype(
            np.float64
        )
    )

    return apparent


def apparent_wind_angle_deg(
    apparent_east: np.ndarray,
    apparent_north: np.ndarray,
    hdg_sin: np.ndarray,
    hdg_cos: np.ndarray,
) -> np.ndarray:
    """
    Apparent WIND-FROM angle in body frame, starboard positive.

    Relative-air vector is a velocity-to vector.
    Wind-from vector = negative of relative-air velocity.
    """
    a_e = apparent_east.astype(
        np.float64
    )
    a_n = apparent_north.astype(
        np.float64
    )
    s = hdg_sin.astype(
        np.float64
    )
    c = hdg_cos.astype(
        np.float64
    )

    # Direction only; normalize the predicted heading pair to avoid
    # pathological amplitude effects. atan2 itself is scale-invariant, but
    # the normalized values make the rotation explicitly physical.
    norm = np.sqrt(
        s ** 2
        + c ** 2
    )

    safe = (
        norm
        > 1e-12
    )

    s_norm = np.where(
        safe,
        s / norm,
        0.0,
    )
    c_norm = np.where(
        safe,
        c / norm,
        1.0,
    )

    relative_forward = (
        a_e
        * s_norm
        + a_n
        * c_norm
    )

    relative_starboard = (
        a_e
        * c_norm
        - a_n
        * s_norm
    )

    wind_from_forward = (
        -relative_forward
    )
    wind_from_starboard = (
        -relative_starboard
    )

    awa = np.degrees(
        np.arctan2(
            wind_from_starboard,
            wind_from_forward,
        )
    )

    return wrap_deg(
        awa
    )


def safe_r2(
    truth: np.ndarray,
    pred: np.ndarray,
) -> float:
    truth = truth.astype(
        np.float64,
        copy=False,
    )
    pred = pred.astype(
        np.float64,
        copy=False,
    )

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
) -> tuple[
    float,
    float,
    int,
]:
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
        pred_deg[
            valid
        ],
        true_deg[
            valid
        ],
    )

    mae = float(
        np.mean(
            np.abs(
                err
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                err ** 2
            )
        )
    )

    return (
        mae,
        rmse,
        n,
    )


def unit_norm_error(
    sin_values: np.ndarray,
    cos_values: np.ndarray,
) -> float:
    norm = np.sqrt(
        sin_values.astype(
            np.float64
        ) ** 2
        + cos_values.astype(
            np.float64
        ) ** 2
    )

    return float(
        np.mean(
            np.abs(
                norm
                - 1.0
            )
        )
    )


# ---------------------------------------------------------------------------
# Metric evaluation
# ---------------------------------------------------------------------------

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
            "truth_joint/pred_joint shape mismatch: "
            f"{truth_joint.shape} vs {pred_joint.shape}"
        )

    if (
        truth_joint.shape[1]
        != len(horizons)
    ):
        raise ValueError(
            "Horizon count mismatch."
        )

    pred_apparent = (
        apparent_earth_from_joint(
            pred_joint
        )
    )

    if persistence_joint is not None:
        persistence_apparent = (
            apparent_earth_from_joint(
                persistence_joint
            )
        )
    else:
        persistence_apparent = None

    rows = []

    for j, horizon in enumerate(
        horizons
    ):
        truth = (
            truth_joint[
                :,
                j,
                :,
            ]
            .astype(
                np.float64,
                copy=False,
            )
        )

        pred = (
            pred_joint[
                :,
                j,
                :,
            ]
            .astype(
                np.float64,
                copy=False,
            )
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

        # -------------------------
        # Earth-relative wind
        # -------------------------
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
        wind_u_mae = float(
            np.mean(
                np.abs(
                    du
                )
            )
        )
        wind_v_mae = float(
            np.mean(
                np.abs(
                    dv
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

        # -------------------------
        # Vessel motion
        # -------------------------
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

        vessel_e_rmse = float(
            np.sqrt(
                np.mean(
                    de ** 2
                )
            )
        )
        vessel_n_rmse = float(
            np.sqrt(
                np.mean(
                    dn ** 2
                )
            )
        )
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
        vessel_e_mae = float(
            np.mean(
                np.abs(
                    de
                )
            )
        )
        vessel_n_mae = float(
            np.mean(
                np.abs(
                    dn
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

        # -------------------------
        # Heading
        # -------------------------
        true_hdg = (
            heading_from_sincos_deg(
                true_hdg_sin,
                true_hdg_cos,
            )
        )
        pred_hdg = (
            heading_from_sincos_deg(
                pred_hdg_sin,
                pred_hdg_cos,
            )
        )

        (
            hdg_mae,
            hdg_rmse,
            n_hdg,
        ) = masked_circular_metrics(
            pred_deg=pred_hdg,
            true_deg=true_hdg,
            mask=np.ones(
                len(true_hdg),
                dtype=bool,
            ),
        )

        hdg_unit_norm_mae = (
            unit_norm_error(
                pred_hdg_sin,
                pred_hdg_cos,
            )
        )

        # -------------------------
        # Apparent wind
        # -------------------------
        true_app = (
            truth_apparent_ref[
                :,
                j,
                :,
            ]
            .astype(
                np.float64,
                copy=False,
            )
        )

        pred_app = (
            pred_apparent[
                :,
                j,
                :,
            ]
        )

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

        apparent_e_rmse = float(
            np.sqrt(
                np.mean(
                    app_de ** 2
                )
            )
        )
        apparent_n_rmse = float(
            np.sqrt(
                np.mean(
                    app_dn ** 2
                )
            )
        )
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

        # -------------------------
        # Skill vs persistence
        # -------------------------
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
        apparent_skill = (
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
            ref = (
                persistence_joint[
                    :,
                    j,
                    :,
                ]
                .astype(
                    np.float64,
                    copy=False,
                )
            )

            ref_wind = ref[
                :,
                0:2,
            ]
            ref_vessel = ref[
                :,
                2:4,
            ]

            ref_wind_err = (
                ref_wind
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
                ref_vessel
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

            ref_app = (
                persistence_apparent[
                    :,
                    j,
                    :,
                ]
            )

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
                apparent_skill = (
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
                len(truth)
            ),

            # Wind
            "wind_U_RMSE_mps": wind_u_rmse,
            "wind_V_RMSE_mps": wind_v_rmse,
            "wind_vector_RMSE_mps": wind_vector_rmse,
            "wind_U_MAE_mps": wind_u_mae,
            "wind_V_MAE_mps": wind_v_mae,
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

            # Vessel
            "vessel_E_RMSE_mps": vessel_e_rmse,
            "vessel_N_RMSE_mps": vessel_n_rmse,
            "vessel_vector_RMSE_mps": vessel_vector_rmse,
            "vessel_E_MAE_mps": vessel_e_mae,
            "vessel_N_MAE_mps": vessel_n_mae,
            "vessel_vector_MAE_mps": vessel_vector_mae,
            "SOG_RMSE_mps": sog_rmse,
            "SOG_MAE_mps": sog_mae,
            "COG_MAE_deg": cog_mae,
            "COG_RMSE_deg": cog_rmse,
            "COG_n": n_cog,
            "vessel_skill_vs_persistence": vessel_skill,

            # Heading
            "HDG_MAE_deg": hdg_mae,
            "HDG_RMSE_deg": hdg_rmse,
            "HDG_n": n_hdg,
            "HDG_pred_unit_norm_MAE": hdg_unit_norm_mae,

            # Apparent wind
            "apparent_E_RMSE_mps": apparent_e_rmse,
            "apparent_N_RMSE_mps": apparent_n_rmse,
            "apparent_vector_RMSE_mps": apparent_vector_rmse,
            "apparent_vector_MAE_mps": apparent_vector_mae,
            "AWS_RMSE_mps": aws_rmse,
            "AWS_MAE_mps": aws_mae,
            "AWA_MAE_deg": awa_mae,
            "AWA_RMSE_deg": awa_rmse,
            "AWA_n": n_awa,
            "apparent_skill_vs_persistence": apparent_skill,
            "AWS_skill_vs_persistence": aws_skill,
        })

    return pd.DataFrame(
        rows
    )


def summarize_metrics(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    summary_columns = [
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
                len(grp)
            ),
        }

        for col in summary_columns:
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


# ---------------------------------------------------------------------------
# Derived prediction export
# ---------------------------------------------------------------------------

def derive_export_arrays(
    joint_raw: np.ndarray,
) -> dict[str, np.ndarray]:
    wind = (
        joint_raw[
            ...,
            0:2,
        ]
        .astype(
            np.float64,
            copy=False,
        )
    )

    vessel = (
        joint_raw[
            ...,
            2:4,
        ]
        .astype(
            np.float64,
            copy=False,
        )
    )

    hdg_sin = (
        joint_raw[
            ...,
            4,
        ]
        .astype(
            np.float64,
            copy=False,
        )
    )

    hdg_cos = (
        joint_raw[
            ...,
            5,
        ]
        .astype(
            np.float64,
            copy=False,
        )
    )

    apparent = (
        apparent_earth_from_joint(
            joint_raw
        )
    )

    wind_speed = vector_speed(
        wind[
            ...,
            0,
        ],
        wind[
            ...,
            1,
        ],
    )

    wind_direction = direction_from_uv_deg(
        wind[
            ...,
            0,
        ],
        wind[
            ...,
            1,
        ],
    )

    sog = vector_speed(
        vessel[
            ...,
            0,
        ],
        vessel[
            ...,
            1,
        ],
    )

    cog = cog_from_en_deg(
        vessel[
            ...,
            0,
        ],
        vessel[
            ...,
            1,
        ],
    )

    hdg = heading_from_sincos_deg(
        hdg_sin,
        hdg_cos,
    )

    aws = vector_speed(
        apparent[
            ...,
            0,
        ],
        apparent[
            ...,
            1,
        ],
    )

    awa = apparent_wind_angle_deg(
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

    return {
        "wind_uv": wind.astype(
            np.float32
        ),
        "wind_speed": wind_speed.astype(
            np.float32
        ),
        "wind_direction_from_deg": wind_direction.astype(
            np.float32
        ),
        "vessel_en": vessel.astype(
            np.float32
        ),
        "SOG": sog.astype(
            np.float32
        ),
        "COG_deg": cog.astype(
            np.float32
        ),
        "HDG_deg": hdg.astype(
            np.float32
        ),
        "apparent_en": apparent.astype(
            np.float32
        ),
        "AWS": aws.astype(
            np.float32
        ),
        "AWA_deg_starboard_positive": awa.astype(
            np.float32
        ),
    }


def save_predictions(
    path: Path,
    truth_joint: np.ndarray,
    persistence_joint: np.ndarray,
    ridge_joint: np.ndarray,
    truth_apparent_ref: np.ndarray,
    context_end_time_ns: np.ndarray,
    target_time_ns: np.ndarray,
    horizons: tuple[int, ...],
) -> None:
    truth_derived = derive_export_arrays(
        truth_joint
    )
    persistence_derived = derive_export_arrays(
        persistence_joint
    )
    ridge_derived = derive_export_arrays(
        ridge_joint
    )

    payload = {
        "truth_joint": np.asarray(
            truth_joint,
            dtype=np.float32,
        ),
        "persistence_joint": np.asarray(
            persistence_joint,
            dtype=np.float32,
        ),
        "ridge_joint": np.asarray(
            ridge_joint,
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

    for prefix, derived in [
        (
            "truth",
            truth_derived,
        ),
        (
            "persistence",
            persistence_derived,
        ),
        (
            "ridge",
            ridge_derived,
        ),
    ]:
        for key, value in derived.items():
            payload[
                f"{prefix}_{key}"
            ] = value

    np.savez_compressed(
        path,
        **payload,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def load_sci_style(
    script_dir: Path,
):
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

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
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
    test_truth_joint: np.ndarray,
    test_persistence_joint: np.ndarray,
    test_ridge_joint: np.ndarray,
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

    def two_model_curve(
        metric_col: str,
        ylabel: str,
        filename: str,
    ):
        fig, ax = plt.subplots(
            figsize=(3.6, 2.8)
        )

        for model_name in [
            "Persistence",
            "Ridge",
        ]:
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
            loc="best"
        )

        finish(
            ax,
            grid=True,
        )
        save(
            fig,
            filename,
        )

    two_model_curve(
        metric_col="wind_vector_RMSE_mps",
        ylabel=r"Wind vector RMSE (m s$^{-1}$)",
        filename="01_test_wind_vector_rmse",
    )

    two_model_curve(
        metric_col="vessel_vector_RMSE_mps",
        ylabel=r"Vessel vector RMSE (m s$^{-1}$)",
        filename="02_test_vessel_vector_rmse",
    )

    two_model_curve(
        metric_col="apparent_vector_RMSE_mps",
        ylabel=r"Apparent-wind vector RMSE (m s$^{-1}$)",
        filename="03_test_apparent_vector_rmse",
    )

    two_model_curve(
        metric_col="AWS_RMSE_mps",
        ylabel=r"AWS RMSE (m s$^{-1}$)",
        filename="04_test_aws_rmse",
    )

    two_model_curve(
        metric_col="AWA_MAE_deg",
        ylabel=r"AWA MAE ($^\circ$)",
        filename="05_test_awa_mae",
    )

    # --------------------------------------------------------
    # Example longest-horizon AWS time series
    # --------------------------------------------------------
    j = (
        len(horizons)
        - 1
    )

    truth_d = derive_export_arrays(
        test_truth_joint
    )
    persistence_d = derive_export_arrays(
        test_persistence_joint
    )
    ridge_d = derive_export_arrays(
        test_ridge_joint
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
        end_time = (
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
            <= end_time
        )
    else:
        mask = np.zeros(
            len(
                target_time
            ),
            dtype=bool,
        )

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
    ax.plot(
        target_time[
            mask
        ],
        persistence_d[
            "AWS"
        ][
            mask,
            j,
        ],
        label="Persistence",
    )
    ax.plot(
        target_time[
            mask
        ],
        ridge_d[
            "AWS"
        ][
            mask,
            j,
        ],
        label="Ridge",
    )

    ax.set_xlabel(
        "Time"
    )
    ax.set_ylabel(
        r"AWS (m s$^{-1}$)"
    )
    ax.legend(
        loc="best",
        ncol=3,
    )

    locator = (
        mdates.AutoDateLocator(
            minticks=4,
            maxticks=8,
        )
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
            "06_test_"
            f"{horizons[j]}min_aws_timeseries"
        ),
    )

    # --------------------------------------------------------
    # Example longest-horizon AWA time series
    # --------------------------------------------------------
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
    ax.plot(
        target_time[
            mask
        ],
        persistence_d[
            "AWA_deg_starboard_positive"
        ][
            mask,
            j,
        ],
        label="Persistence",
    )
    ax.plot(
        target_time[
            mask
        ],
        ridge_d[
            "AWA_deg_starboard_positive"
        ][
            mask,
            j,
        ],
        label="Ridge",
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
        ncol=3,
    )

    locator = (
        mdates.AutoDateLocator(
            minticks=4,
            maxticks=8,
        )
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
            f"{horizons[j]}min_awa_timeseries"
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
            "<dataset-dir>/joint_baseline_v0_1"
        ),
    )

    parser.add_argument(
        "--alpha-grid",
        default=",".join(
            str(v)
            for v in DEFAULT_ALPHA_GRID
        ),
        help=(
            "Comma-separated Ridge alpha candidates. "
            "Default: 0.01,0.1,1,10,100"
        ),
    )

    parser.add_argument(
        "--ridge-solver",
        default="lsqr",
        choices=[
            "auto",
            "svd",
            "cholesky",
            "lsqr",
            "sparse_cg",
            "sag",
            "saga",
            "lbfgs",
        ],
        help=(
            "scikit-learn Ridge solver. Default: lsqr."
        ),
    )

    parser.add_argument(
        "--ridge-tol",
        type=float,
        default=1e-4,
        help=(
            "Ridge solver tolerance. Default: 1e-4."
        ),
    )

    parser.add_argument(
        "--ridge-max-iter",
        type=int,
        default=5000,
        help=(
            "Maximum Ridge iterations. Default: 5000."
        ),
    )

    parser.add_argument(
        "--direction-min-speed",
        type=float,
        default=0.5,
        help=(
            "Minimum true wind/AWS speed used for direction/AWA "
            "metrics, m/s. Default: 0.5."
        ),
    )

    parser.add_argument(
        "--cog-min-speed",
        type=float,
        default=0.2,
        help=(
            "Minimum true SOG used for COG metrics, m/s. "
            "Default: 0.2."
        ),
    )

    parser.add_argument(
        "--example-hours",
        type=float,
        default=24.0,
        help=(
            "Example time-series plot duration. Default: 24 h."
        ),
    )

    parser.add_argument(
        "--skip-validation-flag",
        action="store_true",
        help=(
            "Bypass VALIDATION_PASSED.flag gate. "
            "Not recommended."
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

    if args.direction_min_speed < 0.0:
        raise ValueError(
            "--direction-min-speed must be >= 0."
        )

    if args.cog_min_speed < 0.0:
        raise ValueError(
            "--cog-min-speed must be >= 0."
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
            / "joint_baseline_v0_1"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    alpha_grid = parse_float_grid(
        args.alpha_grid
    )

    log("=" * 96)
    log(
        "SAILDRONE JOINT WIND-VESSEL BASELINE FORECASTING"
    )
    log("=" * 96)
    log(
        f"baseline_version    : {BASELINE_VERSION}"
    )
    log(
        f"dataset_dir         : {dataset_dir}"
    )
    log(
        f"output_dir          : {output_dir}"
    )
    log(
        f"alpha_grid          : {list(alpha_grid)}"
    )
    log(
        f"ridge_solver        : {args.ridge_solver}"
    )
    log(
        f"direction_min_speed : {args.direction_min_speed} m/s"
    )
    log(
        f"cog_min_speed       : {args.cog_min_speed} m/s"
    )
    log("")

    # --------------------------------------------------------
    # Stage 1: validation gate and metadata
    # --------------------------------------------------------
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
        for path in validation_flags:
            log(
                f"     {path}"
            )
    else:
        log(
            "[WARNING] Validation gate bypassed."
        )

    manifest_path = (
        dataset_dir
        / "dataset_manifest.json"
    )
    scalers_path = (
        dataset_dir
        / "scalers.json"
    )
    split_summary_path = (
        dataset_dir
        / "split_summary.csv"
    )

    require_file(
        manifest_path
    )
    require_file(
        scalers_path
    )
    require_file(
        split_summary_path
    )

    manifest = load_json(
        manifest_path
    )
    scalers = load_json(
        scalers_path
    )
    split_summary = pd.read_csv(
        split_summary_path
    )

    task_name = manifest.get(
        "task_name"
    )

    if (
        task_name
        != EXPECTED_TASK_NAME
    ):
        raise RuntimeError(
            f"Expected task_name={EXPECTED_TASK_NAME}, "
            f"got {task_name}"
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

    if (
        feature_names
        != EXPECTED_FEATURE_NAMES
    ):
        raise RuntimeError(
            "Unexpected joint feature definition."
        )

    if (
        target_names
        != EXPECTED_TARGET_NAMES
    ):
        raise RuntimeError(
            "Unexpected joint target definition."
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
            "Feature scaler names do not match manifest."
        )

    if (
        target_scaler_names
        != target_names
    ):
        raise RuntimeError(
            "Target scaler names do not match manifest."
        )

    log(
        f"[OK] Lookback : {lookback} min"
    )
    log(
        f"[OK] Horizons : {list(horizons)} min"
    )
    log(
        f"[OK] X        : {len(feature_names)} features"
    )
    log(
        f"[OK] y        : {len(target_names)} targets"
    )

    try:
        import sklearn
        from sklearn.linear_model import Ridge
        import joblib
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn and joblib are required "
            "in the active WindDL environment."
        ) from exc

    # --------------------------------------------------------
    # Stage 2: load train/validation
    # --------------------------------------------------------
    log(
        "[STAGE 2/8] Loading train and validation splits"
    )

    train = load_npz_split(
        dataset_dir,
        "train",
    )

    validation = load_npz_split(
        dataset_dir,
        "validation",
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

    X_train_2d = flatten_X(
        X_train
    )
    y_train_2d = flatten_y(
        y_train
    )
    X_val_2d = flatten_X(
        X_val
    )
    y_val_2d = flatten_y(
        y_val
    )

    log(
        f"[OK] Ridge train X: {X_train_2d.shape}"
    )
    log(
        f"[OK] Ridge train y: {y_train_2d.shape}"
    )
    log(
        f"[OK] Ridge val X  : {X_val_2d.shape}"
    )

    persistence_val = (
        persistence_joint_prediction(
            X=X_val,
            feature_names=feature_names,
            feature_mean=feature_mean,
            feature_std=feature_std,
            n_horizons=len(
                horizons
            ),
        )
    )

    # --------------------------------------------------------
    # Stage 3: alpha search
    # --------------------------------------------------------
    log(
        "[STAGE 3/8] Ridge alpha search on validation only"
    )

    alpha_rows = []

    best_alpha = None
    best_score = float(
        "inf"
    )
    best_model = None
    best_val_joint = None

    for alpha in alpha_grid:
        log(
            f"[RIDGE] alpha={alpha:g}"
        )

        start_time = (
            time.perf_counter()
        )

        model = Ridge(
            alpha=float(
                alpha
            ),
            fit_intercept=True,
            solver=args.ridge_solver,
            tol=float(
                args.ridge_tol
            ),
            max_iter=int(
                args.ridge_max_iter
            ),
        )

        model.fit(
            X_train_2d,
            y_train_2d,
        )

        val_pred_z_flat = (
            model.predict(
                X_val_2d
            )
        )

        # Primary selection criterion in standardized joint target space.
        joint_z_mse = float(
            np.mean(
                (
                    val_pred_z_flat.astype(
                        np.float64
                    )
                    - y_val_2d.astype(
                        np.float64
                    )
                ) ** 2
            )
        )

        val_joint_raw = (
            inverse_joint_target(
                y_z_flat=val_pred_z_flat,
                n_horizons=len(
                    horizons
                ),
                target_mean=target_mean,
                target_std=target_std,
            )
        )

        temp_metrics = evaluate_model(
            truth_joint=y_val_raw,
            pred_joint=val_joint_raw,
            truth_apparent_ref=app_val_ref,
            horizons=horizons,
            split_name="validation",
            model_name="Ridge",
            direction_min_speed=(
                args.direction_min_speed
            ),
            cog_min_speed=(
                args.cog_min_speed
            ),
            persistence_joint=(
                persistence_val
            ),
        )

        mean_wind_rmse = float(
            temp_metrics[
                "wind_vector_RMSE_mps"
            ].mean()
        )

        mean_vessel_rmse = float(
            temp_metrics[
                "vessel_vector_RMSE_mps"
            ].mean()
        )

        mean_app_rmse = float(
            temp_metrics[
                "apparent_vector_RMSE_mps"
            ].mean()
        )

        mean_awa_mae = float(
            temp_metrics[
                "AWA_MAE_deg"
            ].mean()
        )

        elapsed = float(
            time.perf_counter()
            - start_time
        )

        log(
            "        "
            f"joint_z_MSE={joint_z_mse:.6f}, "
            f"wind_RMSE={mean_wind_rmse:.4f}, "
            f"vessel_RMSE={mean_vessel_rmse:.4f}, "
            f"apparent_RMSE={mean_app_rmse:.4f}, "
            f"AWA_MAE={mean_awa_mae:.2f} deg, "
            f"time={elapsed:.1f} s"
        )

        alpha_rows.append({
            "alpha": float(
                alpha
            ),
            "validation_joint_standardized_MSE": joint_z_mse,
            "validation_mean_wind_vector_RMSE_mps": mean_wind_rmse,
            "validation_mean_vessel_vector_RMSE_mps": mean_vessel_rmse,
            "validation_mean_apparent_vector_RMSE_mps": mean_app_rmse,
            "validation_mean_AWA_MAE_deg": mean_awa_mae,
            "fit_and_validation_seconds": elapsed,
        })

        if (
            joint_z_mse
            < best_score
        ):
            best_score = (
                joint_z_mse
            )
            best_alpha = float(
                alpha
            )
            best_model = copy.deepcopy(
                model
            )
            best_val_joint = (
                val_joint_raw.copy()
            )

    if (
        best_model is None
        or best_val_joint is None
    ):
        raise RuntimeError(
            "Ridge alpha search failed."
        )

    alpha_search_df = (
        pd.DataFrame(
            alpha_rows
        )
        .sort_values(
            "alpha"
        )
    )

    alpha_search_df.to_csv(
        output_dir
        / "ridge_alpha_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log(
        f"[SELECTED] alpha={best_alpha:g}, "
        f"validation joint standardized MSE={best_score:.6f}"
    )

    # --------------------------------------------------------
    # Stage 4: validation metrics
    # --------------------------------------------------------
    log(
        "[STAGE 4/8] Final validation metrics"
    )

    val_persistence_metrics = evaluate_model(
        truth_joint=y_val_raw,
        pred_joint=persistence_val,
        truth_apparent_ref=app_val_ref,
        horizons=horizons,
        split_name="validation",
        model_name="Persistence",
        direction_min_speed=(
            args.direction_min_speed
        ),
        cog_min_speed=(
            args.cog_min_speed
        ),
        persistence_joint=None,
    )

    val_ridge_metrics = evaluate_model(
        truth_joint=y_val_raw,
        pred_joint=best_val_joint,
        truth_apparent_ref=app_val_ref,
        horizons=horizons,
        split_name="validation",
        model_name="Ridge",
        direction_min_speed=(
            args.direction_min_speed
        ),
        cog_min_speed=(
            args.cog_min_speed
        ),
        persistence_joint=(
            persistence_val
        ),
    )

    save_predictions(
        path=(
            output_dir
            / "validation_predictions.npz"
        ),
        truth_joint=y_val_raw,
        persistence_joint=persistence_val,
        ridge_joint=best_val_joint,
        truth_apparent_ref=app_val_ref,
        context_end_time_ns=validation[
            "context_end_time_ns"
        ],
        target_time_ns=validation[
            "target_time_ns"
        ],
        horizons=horizons,
    )

    # Free the large training arrays before loading test.
    del (
        train,
        X_train,
        y_train,
        X_train_2d,
        y_train_2d,
    )

    # --------------------------------------------------------
    # Stage 5: untouched test
    # --------------------------------------------------------
    log(
        "[STAGE 5/8] Untouched test evaluation"
    )

    test = load_npz_split(
        dataset_dir,
        "test",
    )

    X_test = test[
        "X"
    ]
    y_test_raw = test[
        "y_raw"
    ]
    app_test_ref = test[
        "apparent_earth_raw"
    ]

    X_test_2d = flatten_X(
        X_test
    )

    persistence_test = (
        persistence_joint_prediction(
            X=X_test,
            feature_names=feature_names,
            feature_mean=feature_mean,
            feature_std=feature_std,
            n_horizons=len(
                horizons
            ),
        )
    )

    test_pred_z_flat = (
        best_model.predict(
            X_test_2d
        )
    )

    ridge_test = (
        inverse_joint_target(
            y_z_flat=test_pred_z_flat,
            n_horizons=len(
                horizons
            ),
            target_mean=target_mean,
            target_std=target_std,
        )
    )

    test_persistence_metrics = (
        evaluate_model(
            truth_joint=y_test_raw,
            pred_joint=persistence_test,
            truth_apparent_ref=app_test_ref,
            horizons=horizons,
            split_name="test",
            model_name="Persistence",
            direction_min_speed=(
                args.direction_min_speed
            ),
            cog_min_speed=(
                args.cog_min_speed
            ),
            persistence_joint=None,
        )
    )

    test_ridge_metrics = (
        evaluate_model(
            truth_joint=y_test_raw,
            pred_joint=ridge_test,
            truth_apparent_ref=app_test_ref,
            horizons=horizons,
            split_name="test",
            model_name="Ridge",
            direction_min_speed=(
                args.direction_min_speed
            ),
            cog_min_speed=(
                args.cog_min_speed
            ),
            persistence_joint=(
                persistence_test
            ),
        )
    )

    save_predictions(
        path=(
            output_dir
            / "test_predictions.npz"
        ),
        truth_joint=y_test_raw,
        persistence_joint=persistence_test,
        ridge_joint=ridge_test,
        truth_apparent_ref=app_test_ref,
        context_end_time_ns=test[
            "context_end_time_ns"
        ],
        target_time_ns=test[
            "target_time_ns"
        ],
        horizons=horizons,
    )

    # --------------------------------------------------------
    # Stage 6: metrics + model
    # --------------------------------------------------------
    log(
        "[STAGE 6/8] Saving metrics and Ridge model"
    )

    metrics = pd.concat(
        [
            val_persistence_metrics,
            val_ridge_metrics,
            test_persistence_metrics,
            test_ridge_metrics,
        ],
        ignore_index=True,
    )

    metrics.to_csv(
        output_dir
        / "metrics_per_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics_summary = (
        summarize_metrics(
            metrics
        )
    )

    metrics_summary.to_csv(
        output_dir
        / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    joblib.dump(
        best_model,
        output_dir
        / "ridge_model.joblib",
    )

    # --------------------------------------------------------
    # Stage 7: manifest + report
    # --------------------------------------------------------
    log(
        "[STAGE 7/8] Writing manifest and report"
    )

    output_manifest = {
        "baseline_version": BASELINE_VERSION,
        "source_dataset_version": (
            manifest.get(
                "dataset_version"
            )
        ),
        "source_task_name": task_name,
        "source_dataset_dir": str(
            dataset_dir.resolve()
        ),
        "validation_flags": [
            str(
                path.resolve()
            )
            for path in validation_flags
        ],
        "lookback_minutes": lookback,
        "forecast_horizons_minutes": list(
            horizons
        ),
        "feature_names": feature_names,
        "target_names": target_names,
        "direction_metric_thresholds": {
            "wind_and_AWA_min_true_speed_mps": float(
                args.direction_min_speed
            ),
            "COG_min_true_SOG_mps": float(
                args.cog_min_speed
            ),
        },
        "angle_conventions": {
            "wind_direction": (
                "meteorological wind-from direction, "
                "degrees clockwise from North"
            ),
            "COG": (
                "degrees clockwise from North"
            ),
            "HDG": (
                "degrees clockwise from North"
            ),
            "AWA": (
                "body-frame apparent wind-from angle, "
                "starboard positive, wrapped to [-180,180)"
            ),
        },
        "models": {
            "Persistence": {
                "definition": (
                    "Future wind, vessel velocity and heading "
                    "equal the last observed state."
                ),
            },
            "Ridge": {
                "flattened_input_count": (
                    lookback
                    * len(
                        feature_names
                    )
                ),
                "flattened_output_count": (
                    len(
                        horizons
                    )
                    * len(
                        target_names
                    )
                ),
                "alpha_candidates": list(
                    alpha_grid
                ),
                "selected_alpha": best_alpha,
                "solver": args.ridge_solver,
                "tol": float(
                    args.ridge_tol
                ),
                "max_iter": int(
                    args.ridge_max_iter
                ),
                "selection_split": "validation",
                "selection_metric": (
                    "mean standardized joint target MSE"
                ),
                "final_fit_scope": (
                    "training split only"
                ),
            },
        },
        "physics_layer": {
            "apparent_E": (
                "UWND_MEAN - VESSEL_EAST_MPS"
            ),
            "apparent_N": (
                "VWND_MEAN - VESSEL_NORTH_MPS"
            ),
            "AWA": (
                "computed from predicted apparent Earth vector "
                "and predicted HDG"
            ),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "files": {
            "ridge_alpha_search": (
                "ridge_alpha_search.csv"
            ),
            "metrics_per_horizon": (
                "metrics_per_horizon.csv"
            ),
            "metrics_summary": (
                "metrics_summary.csv"
            ),
            "validation_predictions": (
                "validation_predictions.npz"
            ),
            "test_predictions": (
                "test_predictions.npz"
            ),
            "ridge_model": (
                "ridge_model.joblib"
            ),
        },
    }

    with (
        output_dir
        / "joint_baseline_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output_manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    test_metrics = (
        metrics.loc[
            metrics[
                "split"
            ] == "test"
        ]
        .copy()
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
        / "joint_baseline_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "Saildrone joint wind-vessel baseline forecasting report\n"
        )
        f.write(
            "=" * 96
            + "\n\n"
        )
        f.write(
            f"Baseline version: {BASELINE_VERSION}\n"
        )
        f.write(
            f"Dataset version: "
            f"{manifest.get('dataset_version')}\n"
        )
        f.write(
            f"Lookback: {lookback} min\n"
        )
        f.write(
            f"Horizons: {list(horizons)} min\n"
        )
        f.write(
            f"Selected Ridge alpha: {best_alpha}\n"
        )
        f.write(
            "Ridge validation selection metric: "
            f"joint standardized MSE = {best_score:.8f}\n"
        )
        f.write(
            "AWA convention: 0=headwind, "
            "+90=starboard, -90=port, +/-180=astern\n\n"
        )

        f.write(
            "Ridge alpha search\n"
        )
        f.write(
            "-" * 96
            + "\n"
        )
        f.write(
            alpha_search_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nTest metrics per horizon\n"
        )
        f.write(
            "-" * 96
            + "\n"
        )
        f.write(
            test_metrics.to_string(
                index=False
            )
        )

        f.write(
            "\n\nTest summary\n"
        )
        f.write(
            "-" * 96
            + "\n"
        )
        f.write(
            test_summary.to_string(
                index=False
            )
        )
        f.write("\n")

    # --------------------------------------------------------
    # Stage 8: figures + console summary
    # --------------------------------------------------------
    log(
        "[STAGE 8/8] Finalization"
    )

    if args.plots:
        log(
            "[PLOT] Generating SCI-style joint baseline figures"
        )

        make_plots(
            metrics=metrics,
            test_truth_joint=y_test_raw,
            test_persistence_joint=persistence_test,
            test_ridge_joint=ridge_test,
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
    log("=" * 96)
    log(
        "TEST JOINT BASELINE RESULTS"
    )
    log("=" * 96)

    for model_name in [
        "Persistence",
        "Ridge",
    ]:
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

        summary_row = (
            test_summary.loc[
                test_summary[
                    "model"
                ] == model_name
            ]
            .iloc[0]
        )

        log(
            f"{model_name}:"
        )
        log(
            "  mean wind vector RMSE     = "
            f"{summary_row['mean_wind_vector_RMSE_mps']:.4f} m/s"
        )
        log(
            "  mean vessel vector RMSE   = "
            f"{summary_row['mean_vessel_vector_RMSE_mps']:.4f} m/s"
        )
        log(
            "  mean apparent vector RMSE = "
            f"{summary_row['mean_apparent_vector_RMSE_mps']:.4f} m/s"
        )
        log(
            "  mean AWS RMSE             = "
            f"{summary_row['mean_AWS_RMSE_mps']:.4f} m/s"
        )
        log(
            "  mean AWA MAE              = "
            f"{summary_row['mean_AWA_MAE_deg']:.2f} deg"
        )
        log(
            "  mean HDG MAE              = "
            f"{summary_row['mean_HDG_MAE_deg']:.2f} deg"
        )

        per_horizon = ", ".join(
            (
                f"{int(row.horizon_min)}min:"
                f"AW={row.apparent_vector_RMSE_mps:.3f},"
                f"AWS={row.AWS_RMSE_mps:.3f},"
                f"AWA={row.AWA_MAE_deg:.2f}deg"
            )
            for row in grp.itertuples()
        )

        log(
            f"  apparent preview: {per_horizon}"
        )

    log("")
    log(
        f"[SELECTED] Ridge alpha = {best_alpha:g}"
    )
    log(
        f"[DONE] Joint baseline outputs: {output_dir}"
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
