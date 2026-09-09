# -*- coding: utf-8 -*-
r"""
13A_BPInput_Attribution_and_ResidualProbe_FinalTropical.py

Stage 13A
=========
BP-owned input attribution for the strict Stage-12G point-sampled benchmark.

PURPOSE
-------
Stage 12Z established a useful architectural fact:

    Base4-Ridge + compact U/V residual correction

can slightly improve U and WS, but Tropical Atlantic V remains close to the
Base4-Ridge value (~0.792 m/s), while published BP-STGNN reports 0.70506 m/s.

The next question is therefore NOT:
    "Should the residual network be deeper?"

It is:
    "Which BP-owned auxiliary observations contain incremental information
     for V / direction beyond [U,V,T,RH]?"

Stage 13A performs TWO complementary attribution experiments.

PART A — DIRECT RIDGE INPUT ATTRIBUTION
---------------------------------------
Fit the SAME standardized multi-output Ridge model under predeclared feature
groups:

    Base4
        U,V,T,RH

    Base4+Motion
        + SOG, COG_sin, COG_cos

    Base4+P
        + P

    Base4+dP
        + dP

    Base4+ThermoTrend
        + dT, dRH

    Base4+PressureDyn
        + P, dP

    Base4+Motion+ThermoTrend
        + SOG, COG_sin, COG_cos, dT, dRH

    Base4+Motion+PressureDyn
        + SOG, COG_sin, COG_cos, P, dP

    BPFullEngineered
        + SOG, COG_sin, COG_cos, P, dT, dP, dRH

COG_sin/COG_cos are deterministic circular reparameterizations of the BP-owned
COG node. They do not add observations.

PART B — BASE4-RIDGE + AUXILIARY RESIDUAL PROBE
------------------------------------------------
Keep Base4-Ridge as the anchor.

For each auxiliary group, fit a small Ridge residual probe to:

    r_U = U_true - U_Base4Ridge_OOF
    r_V = V_true - V_Base4Ridge_OOF

The residual probe sees:
    selected auxiliary sequence
    + Base4-Ridge anchor context [U_R,V_R,WS_R]

This directly tests whether a variable group explains incremental error that
Base4-Ridge leaves behind.

Auxiliary probe groups:
    AnchorOnly
    Motion
    P
    dP
    ThermoTrend
    PressureDyn
    Motion+ThermoTrend
    Motion+PressureDyn
    AllAux

For each group, component-wise analytical shrinkage is calibrated only from
the pooled development out-of-mission predictions:

    alpha_j =
        clip(
            sum(e_j * rhat_j) / sum(rhat_j^2),
            0,
            1
        )

Final residual:
    y_final = y_Base4Ridge + alpha * rhat

If a development residual probe is useless or anti-correlated, alpha can
collapse to 0.

DATA SOURCE / ALIGNMENT
-----------------------
Stage 12G creates two exactly aligned datasets:

    track_J_joint_compatible
        [U,V,T,RH,SOG,COG_sin,COG_cos,WING_sin,WING_cos]

    track_P_paper_informed
        [U,V,SOG,COG,T,RH,P,dT,dP,dRH]

Stage 13A opens BOTH and verifies:
    sample count equal
    context timestamps equal
    target timestamps equal
    U/V target equal
    shared U,V,T,RH,SOG equal
    sin/cos(raw Track-P COG) matches Track-J COG sin/cos

It then builds one canonical 11-feature tensor:

    U
    V
    T
    RH
    SOG
    COG_sin
    COG_cos
    P
    dT
    dP
    dRH

No wing angle is used.

STRICT TEMPORAL PROTOCOL
------------------------
Six point samples every 10 min:
    t-50,t-40,t-30,t-20,t-10,t
predict:
    t+10

No arithmetic 10-min averaging.
No intermediate 1-min points.
No interpolation.

DEVELOPMENT / TROPICAL FIREWALL
-------------------------------
Development:
    Antarctic
    Atlantic
    West Coast

Outer LOMO:
    hold out each development mission once.

Before Tropical is loaded, Stage 13A saves:
    development_direct_ridge_summary.csv
    development_residual_probe_summary.csv
    FROZEN_BEFORE_TROPICAL.json

The frozen JSON contains:
    all predeclared groups
    development rankings
    pooled development analytical alpha_U/V for every residual probe

Only THEN:
    load Tropical_Atlantic_TEST.npz
    fit each predeclared model on all three development missions
    evaluate every predeclared group once on Tropical Atlantic

IMPORTANT:
Tropical Atlantic has already been observed in earlier project stages.
Therefore these final Tropical attribution results are exploratory benchmark
evidence and must not be described as a never-seen pristine test.

SELECTION / REPORTING
---------------------
No group is adaptively created after Tropical results are seen.

Development V-focused score:
    0.20 * U_ratio
  + 0.60 * V_ratio
  + 0.10 * WS_ratio
  + 0.10 * WD_ratio

This is diagnostic ranking only.

OUTPUTS
-------
development_direct_ridge_runs.csv
development_direct_ridge_summary.csv
development_residual_probe_runs.csv
development_residual_probe_summary.csv
development_residual_pooled_calibration.csv
FROZEN_BEFORE_TROPICAL.json

FINAL_TROPICAL_DIRECT_RIDGE_ATTRIBUTION.csv
FINAL_TROPICAL_RESIDUAL_PROBE_ATTRIBUTION.csv
13A_REPORT.txt
13A_REPORT.json

Published BP-STGNN reference:
    U  = 0.748640 m/s
    V  = 0.705060 m/s
    WS = 0.699400 m/s
    WD = 5.584 deg
    params = 242580

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\13A_BPInput_Attribution_and_ResidualProbe_FinalTropical.py" --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\13A_BPInput_Attribution_v0_1"

Smoke:
    add --debug-fast
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import Ridge
except Exception as exc:
    raise RuntimeError(
        "scikit-learn is required. Activate the WindPredict environment."
    ) from exc


SCRIPT_VERSION = "0.1.0-13A-BPInput-Attribution"

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
    / "13A_BPInput_Attribution_v0_1"
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

RIDGE_ALPHA = 1.0
RESIDUAL_RIDGE_ALPHA = 1.0
INNER_CROSSFIT_FOLDS = 5
EPS = 1e-12
ALPHA_MAX = 1.0

TRACK_J_FEATURES = [
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

TRACK_P_FEATURES = [
    "U",
    "V",
    "SOG",
    "COG",
    "T",
    "RH",
    "P",
    "dT",
    "dP",
    "dRH",
]

CANONICAL_FEATURES = [
    "U",
    "V",
    "T",
    "RH",
    "SOG",
    "COG_sin",
    "COG_cos",
    "P",
    "dT",
    "dP",
    "dRH",
]

FEATURE_INDEX = {
    name: i
    for i, name in enumerate(
        CANONICAL_FEATURES
    )
}

DIRECT_GROUPS = {
    "Base4": [
        "U",
        "V",
        "T",
        "RH",
    ],
    "Base4+Motion": [
        "U",
        "V",
        "T",
        "RH",
        "SOG",
        "COG_sin",
        "COG_cos",
    ],
    "Base4+P": [
        "U",
        "V",
        "T",
        "RH",
        "P",
    ],
    "Base4+dP": [
        "U",
        "V",
        "T",
        "RH",
        "dP",
    ],
    "Base4+ThermoTrend": [
        "U",
        "V",
        "T",
        "RH",
        "dT",
        "dRH",
    ],
    "Base4+PressureDyn": [
        "U",
        "V",
        "T",
        "RH",
        "P",
        "dP",
    ],
    "Base4+Motion+ThermoTrend": [
        "U",
        "V",
        "T",
        "RH",
        "SOG",
        "COG_sin",
        "COG_cos",
        "dT",
        "dRH",
    ],
    "Base4+Motion+PressureDyn": [
        "U",
        "V",
        "T",
        "RH",
        "SOG",
        "COG_sin",
        "COG_cos",
        "P",
        "dP",
    ],
    "BPFullEngineered": [
        "U",
        "V",
        "T",
        "RH",
        "SOG",
        "COG_sin",
        "COG_cos",
        "P",
        "dT",
        "dP",
        "dRH",
    ],
}

RESIDUAL_GROUPS = {
    "AnchorOnly": [],
    "Motion": [
        "SOG",
        "COG_sin",
        "COG_cos",
    ],
    "P": [
        "P",
    ],
    "dP": [
        "dP",
    ],
    "ThermoTrend": [
        "dT",
        "dRH",
    ],
    "PressureDyn": [
        "P",
        "dP",
    ],
    "Motion+ThermoTrend": [
        "SOG",
        "COG_sin",
        "COG_cos",
        "dT",
        "dRH",
    ],
    "Motion+PressureDyn": [
        "SOG",
        "COG_sin",
        "COG_cos",
        "P",
        "dP",
    ],
    "AllAux": [
        "SOG",
        "COG_sin",
        "COG_cos",
        "P",
        "dT",
        "dP",
        "dRH",
    ],
}

BP_REFERENCE = {
    "U_RMSE_mps": 0.748640,
    "V_RMSE_mps": 0.705060,
    "WS_RMSE_mps": 0.699400,
    "WD_RMSE_deg": 5.584000,
    "params": 242580,
}


# =============================================================================
# General utilities
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


def rel_improve(old, new):
    return float(
        (
            float(old)
            - float(new)
        )
        / max(
            abs(
                float(old)
            ),
            EPS,
        )
    )


# =============================================================================
# Dataset loading and alignment
# =============================================================================

def resolve_dataset_root(dataset_dir: Path):
    for root in [
        dataset_dir,
        dataset_dir / "dataset",
    ]:
        if (
            root
            / "track_J_joint_compatible"
        ).exists() and (
            root
            / "track_P_paper_informed"
        ).exists():
            return root

    raise FileNotFoundError(
        "Could not find both track_J_joint_compatible and "
        "track_P_paper_informed."
    )


def mission_filename(
    mission: str,
):
    safe = mission.replace(
        " ",
        "_",
    )

    if mission == FINAL_MISSION:
        return (
            f"{safe}_TEST.npz"
        )

    return f"{safe}.npz"


def load_npz_track(
    path: Path,
    expected_features,
):
    if not path.exists():
        raise FileNotFoundError(
            path
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
            z[
                "context_end_time_ns"
            ],
            dtype=np.int64,
        ).reshape(-1)

        target = np.asarray(
            z[
                "target_time_ns"
            ],
            dtype=np.int64,
        ).reshape(-1)

        names = [
            str(x)
            for x
            in z[
                "feature_names"
            ].tolist()
        ]

    if names != expected_features:
        raise RuntimeError(
            f"{path.name}: feature mismatch\n"
            f"expected={expected_features}\n"
            f"actual={names}"
        )

    if (
        X.ndim != 3
        or X.shape[
            1
        ] != LOOKBACK_STEPS
    ):
        raise RuntimeError(
            f"{path.name}: unexpected X shape {X.shape}"
        )

    if y.ndim == 3:
        if y.shape[1:] != (
            1,
            2,
        ):
            raise RuntimeError(
                f"{path.name}: unexpected y shape {y.shape}"
            )
        y = y[
            :,
            0,
            :
        ]

    if (
        y.ndim != 2
        or y.shape[
            1
        ] != 2
    ):
        raise RuntimeError(
            f"{path.name}: unexpected y shape {y.shape}"
        )

    if not np.all(
        target
        - context
        == TEN_MIN_NS
    ):
        raise RuntimeError(
            f"{path.name}: target is not exactly +10 min."
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
            f"{path.name}: nonfinite values."
        )

    return {
        "X": X,
        "y": y,
        "context": context,
        "target": target,
        "path": path,
    }


def load_aligned_mission(
    dataset_root: Path,
    mission: str,
):
    filename = mission_filename(
        mission
    )

    j = load_npz_track(
        dataset_root
        / "track_J_joint_compatible"
        / filename,
        TRACK_J_FEATURES,
    )

    p = load_npz_track(
        dataset_root
        / "track_P_paper_informed"
        / filename,
        TRACK_P_FEATURES,
    )

    if len(
        j[
            "X"
        ]
    ) != len(
        p[
            "X"
        ]
    ):
        raise RuntimeError(
            f"{mission}: Track-J/P sample count mismatch."
        )

    if not np.array_equal(
        j[
            "context"
        ],
        p[
            "context"
        ],
    ):
        raise RuntimeError(
            f"{mission}: context timestamps not aligned."
        )

    if not np.array_equal(
        j[
            "target"
        ],
        p[
            "target"
        ],
    ):
        raise RuntimeError(
            f"{mission}: target timestamps not aligned."
        )

    y_err = float(
        np.max(
            np.abs(
                j[
                    "y"
                ]
                - p[
                    "y"
                ]
            )
        )
    )

    if y_err > 1e-6:
        raise RuntimeError(
            f"{mission}: Track-J/P target mismatch max={y_err:.3e}"
        )

    Xj = j[
        "X"
    ].astype(
        np.float64
    )

    Xp = p[
        "X"
    ].astype(
        np.float64
    )

    # Shared-variable alignment checks.
    shared_pairs = [
        (
            "U",
            Xj[
                :,
                :,
                0
            ],
            Xp[
                :,
                :,
                0
            ],
        ),
        (
            "V",
            Xj[
                :,
                :,
                1
            ],
            Xp[
                :,
                :,
                1
            ],
        ),
        (
            "T",
            Xj[
                :,
                :,
                2
            ],
            Xp[
                :,
                :,
                4
            ],
        ),
        (
            "RH",
            Xj[
                :,
                :,
                3
            ],
            Xp[
                :,
                :,
                5
            ],
        ),
        (
            "SOG",
            Xj[
                :,
                :,
                4
            ],
            Xp[
                :,
                :,
                2
            ],
        ),
    ]

    shared_max = {}

    for name, a, b in shared_pairs:
        err = float(
            np.max(
                np.abs(
                    a - b
                )
            )
        )

        shared_max[
            name
        ] = err

        if err > 2e-5:
            raise RuntimeError(
                f"{mission}: aligned {name} mismatch max={err:.3e}"
            )

    cog_rad = np.deg2rad(
        Xp[
            :,
            :,
            3
        ]
    )

    cog_sin = np.sin(
        cog_rad
    )

    cog_cos = np.cos(
        cog_rad
    )

    cog_sin_err = float(
        np.max(
            np.abs(
                cog_sin
                - Xj[
                    :,
                    :,
                    5
                ]
            )
        )
    )

    cog_cos_err = float(
        np.max(
            np.abs(
                cog_cos
                - Xj[
                    :,
                    :,
                    6
                ]
            )
        )
    )

    if max(
        cog_sin_err,
        cog_cos_err,
    ) > 2e-5:
        raise RuntimeError(
            f"{mission}: COG circular transform mismatch "
            f"sin={cog_sin_err:.3e}, cos={cog_cos_err:.3e}"
        )

    canonical = np.stack(
        [
            Xj[
                :,
                :,
                0
            ],  # U
            Xj[
                :,
                :,
                1
            ],  # V
            Xj[
                :,
                :,
                2
            ],  # T
            Xj[
                :,
                :,
                3
            ],  # RH
            Xj[
                :,
                :,
                4
            ],  # SOG
            Xj[
                :,
                :,
                5
            ],  # COG sin
            Xj[
                :,
                :,
                6
            ],  # COG cos
            Xp[
                :,
                :,
                6
            ],  # P
            Xp[
                :,
                :,
                7
            ],  # dT
            Xp[
                :,
                :,
                8
            ],  # dP
            Xp[
                :,
                :,
                9
            ],  # dRH
        ],
        axis=-1,
    ).astype(
        np.float32
    )

    if canonical.shape[
        2
    ] != len(
        CANONICAL_FEATURES
    ):
        raise RuntimeError(
            "Canonical feature construction failed."
        )

    if not np.isfinite(
        canonical
    ).all():
        raise RuntimeError(
            f"{mission}: canonical tensor nonfinite."
        )

    return {
        "X": canonical,
        "y": j[
            "y"
        ].astype(
            np.float32
        ),
        "mission": mission,
        "context": j[
            "context"
        ],
        "target": j[
            "target"
        ],
        "alignment_audit": {
            "mission": mission,
            "samples": int(
                len(
                    canonical
                )
            ),
            "target_UV_max_abs_error": (
                y_err
            ),
            "COG_sin_max_abs_error": (
                cog_sin_err
            ),
            "COG_cos_max_abs_error": (
                cog_cos_err
            ),
            **{
                f"shared_{k}_max_abs_error": v
                for k, v
                in shared_max.items()
            },
            "TrackJ_path": str(
                j[
                    "path"
                ]
            ),
            "TrackP_path": str(
                p[
                    "path"
                ]
            ),
        },
    }


def concat_missions(
    items,
):
    return {
        "X": np.concatenate(
            [
                x[
                    "X"
                ]
                for x in items
            ],
            axis=0,
        ),
        "y": np.concatenate(
            [
                x[
                    "y"
                ]
                for x in items
            ],
            axis=0,
        ),
        "mission_labels": np.concatenate(
            [
                np.asarray(
                    [
                        x[
                            "mission"
                        ]
                    ]
                    * len(
                        x[
                            "X"
                        ]
                    ),
                    dtype=object,
                )
                for x in items
            ],
            axis=0,
        ),
    }


def select_features(
    X,
    names,
):
    idx = [
        FEATURE_INDEX[
            name
        ]
        for name in names
    ]

    return np.asarray(
        X[
            :,
            :,
            idx
        ],
        dtype=np.float32,
    )


# =============================================================================
# Standardized Ridge
# =============================================================================

def fit_xy_scaler(
    X,
    y,
):
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


def fit_direct_ridge(
    X,
    y,
):
    scaler = fit_xy_scaler(
        X,
        y,
    )

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        transform_X(
            X,
            scaler,
        ).reshape(
            len(X),
            -1,
        ),
        transform_y(
            y,
            scaler,
        ),
    )

    return (
        model,
        scaler,
    )


def predict_direct_ridge(
    model,
    scaler,
    X,
):
    pred_z = np.asarray(
        model.predict(
            transform_X(
                X,
                scaler,
            ).reshape(
                len(X),
                -1,
            )
        ),
        dtype=np.float32,
    )

    return inverse_y(
        pred_z,
        scaler,
    )


def direct_parameter_count(
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
# Metrics
# =============================================================================

def wd_from_uv(
    uv,
):
    uv = np.asarray(
        uv,
        dtype=np.float64,
    )

    return (
        np.degrees(
            np.arctan2(
                -uv[
                    :,
                    0
                ],
                -uv[
                    :,
                    1
                ],
            )
        )
        % 360.0
    )


def circular_diff_deg(
    a,
    b,
):
    return (
        (
            np.asarray(
                a
            )
            - np.asarray(
                b
            )
            + 180.0
        )
        % 360.0
        - 180.0
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

    ws_t = np.linalg.norm(
        yt,
        axis=1,
    )

    ws_p = np.linalg.norm(
        yp,
        axis=1,
    )

    wd_err = circular_diff_deg(
        wd_from_uv(
            yp
        ),
        wd_from_uv(
            yt
        ),
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


def v_focus_score(
    metrics,
    base_metrics,
):
    return float(
        0.20
        * (
            metrics[
                "wind_U_RMSE_mps"
            ]
            / max(
                base_metrics[
                    "wind_U_RMSE_mps"
                ],
                EPS,
            )
        )
        + 0.60
        * (
            metrics[
                "wind_V_RMSE_mps"
            ]
            / max(
                base_metrics[
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
                base_metrics[
                    "wind_speed_RMSE_mps"
                ],
                EPS,
            )
        )
        + 0.10
        * (
            metrics[
                "wind_direction_RMSE_deg"
            ]
            / max(
                base_metrics[
                    "wind_direction_RMSE_deg"
                ],
                EPS,
            )
        )
    )


def corrcoef_safe(
    a,
    b,
):
    a = np.asarray(
        a,
        dtype=np.float64,
    ).reshape(-1)

    b = np.asarray(
        b,
        dtype=np.float64,
    ).reshape(-1)

    if (
        len(
            a
        ) < 3
        or np.std(
            a
        ) < EPS
        or np.std(
            b
        ) < EPS
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


# =============================================================================
# Base4 OOF
# =============================================================================

def blocked_fold_ids(
    mission_labels,
):
    labels = np.asarray(
        mission_labels,
        dtype=object,
    )

    fold_ids = np.full(
        len(
            labels
        ),
        -1,
        dtype=np.int64,
    )

    for mission in np.unique(
        labels
    ):
        idx = np.flatnonzero(
            labels
            == mission
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
        fold_ids
        < 0
    ):
        raise RuntimeError(
            "OOF fold assignment failed."
        )

    return fold_ids


def base4_oof_predictions(
    train_data,
):
    X = select_features(
        train_data[
            "X"
        ],
        DIRECT_GROUPS[
            "Base4"
        ],
    )

    y = train_data[
        "y"
    ]

    fold_ids = blocked_fold_ids(
        train_data[
            "mission_labels"
        ]
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
            fold_ids
            == k
        )

        tr = ~va

        model, scaler = fit_direct_ridge(
            X[
                tr
            ],
            y[
                tr
            ],
        )

        pred[
            va
        ] = predict_direct_ridge(
            model,
            scaler,
            X[
                va
            ],
        )

    if not np.isfinite(
        pred
    ).all():
        raise RuntimeError(
            "Base4 OOF prediction nonfinite."
        )

    return pred


# =============================================================================
# Residual linear probe
# =============================================================================

def build_probe_matrix(
    X_canonical,
    aux_names,
    ridge_anchor,
):
    parts = []

    if aux_names:
        aux = select_features(
            X_canonical,
            aux_names,
        ).reshape(
            len(
                X_canonical
            ),
            -1,
        )

        parts.append(
            aux.astype(
                np.float32
            )
        )

    anchor = np.asarray(
        ridge_anchor,
        dtype=np.float32,
    )

    anchor_context = np.concatenate(
        [
            anchor,
            np.linalg.norm(
                anchor,
                axis=1,
                keepdims=True,
            ),
        ],
        axis=1,
    ).astype(
        np.float32
    )

    parts.append(
        anchor_context
    )

    return np.concatenate(
        parts,
        axis=1,
    ).astype(
        np.float32
    )


def fit_matrix_ridge(
    X2,
    y2,
    alpha,
):
    X64 = np.asarray(
        X2,
        dtype=np.float64,
    )

    y64 = np.asarray(
        y2,
        dtype=np.float64,
    )

    x_mean = np.mean(
        X64,
        axis=0,
    )

    x_std = np.std(
        X64,
        axis=0,
        ddof=0,
    )

    x_std = np.where(
        x_std
        < 1e-8,
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
        y_std
        < 1e-8,
        1.0,
        y_std,
    )

    Xz = (
        X64
        - x_mean[
            None,
            :
        ]
    ) / x_std[
        None,
        :
    ]

    yz = (
        y64
        - y_mean[
            None,
            :
        ]
    ) / y_std[
        None,
        :
    ]

    model = Ridge(
        alpha=float(
            alpha
        )
    )

    model.fit(
        Xz,
        yz,
    )

    scaler = {
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

    return (
        model,
        scaler,
    )


def predict_matrix_ridge(
    model,
    scaler,
    X2,
):
    X64 = np.asarray(
        X2,
        dtype=np.float64,
    )

    Xz = (
        X64
        - scaler[
            "x_mean"
        ][
            None,
            :
        ]
    ) / scaler[
        "x_std"
    ][
        None,
        :
    ]

    yz = np.asarray(
        model.predict(
            Xz
        ),
        dtype=np.float64,
    )

    y = (
        yz
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
    )

    return y.astype(
        np.float32
    )


def analytical_alpha_1d(
    true_res,
    pred_res,
):
    e = np.asarray(
        true_res,
        dtype=np.float64,
    ).reshape(-1)

    r = np.asarray(
        pred_res,
        dtype=np.float64,
    ).reshape(-1)

    denom = float(
        np.dot(
            r,
            r,
        )
    )

    if denom <= EPS:
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
            ALPHA_MAX,
        )
    )


def analytical_alphas(
    y_true,
    ridge_anchor,
    raw_residual,
):
    true_res = (
        np.asarray(
            y_true,
            dtype=np.float64,
        )
        - np.asarray(
            ridge_anchor,
            dtype=np.float64,
        )
    )

    raw = np.asarray(
        raw_residual,
        dtype=np.float64,
    )

    return np.asarray(
        [
            analytical_alpha_1d(
                true_res[
                    :,
                    0
                ],
                raw[
                    :,
                    0
                ],
            ),
            analytical_alpha_1d(
                true_res[
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


def apply_residual(
    ridge_anchor,
    raw_residual,
    alphas,
):
    return (
        np.asarray(
            ridge_anchor,
            dtype=np.float32,
        )
        + np.asarray(
            raw_residual,
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


# =============================================================================
# Development experiments
# =============================================================================

def run_development_direct(
    dev_missions,
):
    rows = []

    for held_out in DEV_MISSIONS:
        train_data = concat_missions(
            [
                dev_missions[
                    m
                ]
                for m in DEV_MISSIONS
                if m
                != held_out
            ]
        )

        val_data = dev_missions[
            held_out
        ]

        base_X_train = select_features(
            train_data[
                "X"
            ],
            DIRECT_GROUPS[
                "Base4"
            ],
        )

        base_X_val = select_features(
            val_data[
                "X"
            ],
            DIRECT_GROUPS[
                "Base4"
            ],
        )

        base_model, base_scaler = fit_direct_ridge(
            base_X_train,
            train_data[
                "y"
            ],
        )

        base_pred = predict_direct_ridge(
            base_model,
            base_scaler,
            base_X_val,
        )

        base_metrics = evaluate_wind(
            val_data[
                "y"
            ],
            base_pred,
        )

        for group_name, names in DIRECT_GROUPS.items():
            Xtr = select_features(
                train_data[
                    "X"
                ],
                names,
            )

            Xva = select_features(
                val_data[
                    "X"
                ],
                names,
            )

            model, scaler = fit_direct_ridge(
                Xtr,
                train_data[
                    "y"
                ],
            )

            pred = predict_direct_ridge(
                model,
                scaler,
                Xva,
            )

            metrics = evaluate_wind(
                val_data[
                    "y"
                ],
                pred,
            )

            rows.append(
                {
                    "held_out_mission": (
                        held_out
                    ),
                    "group": (
                        group_name
                    ),
                    "feature_count_per_step": int(
                        len(
                            names
                        )
                    ),
                    "parameter_count": int(
                        direct_parameter_count(
                            model
                        )
                    ),
                    "v_focus_score": float(
                        v_focus_score(
                            metrics,
                            base_metrics,
                        )
                    ),
                    **metrics,
                    "U_improve_vs_Base4_fraction": (
                        rel_improve(
                            base_metrics[
                                "wind_U_RMSE_mps"
                            ],
                            metrics[
                                "wind_U_RMSE_mps"
                            ],
                        )
                    ),
                    "V_improve_vs_Base4_fraction": (
                        rel_improve(
                            base_metrics[
                                "wind_V_RMSE_mps"
                            ],
                            metrics[
                                "wind_V_RMSE_mps"
                            ],
                        )
                    ),
                    "WS_improve_vs_Base4_fraction": (
                        rel_improve(
                            base_metrics[
                                "wind_speed_RMSE_mps"
                            ],
                            metrics[
                                "wind_speed_RMSE_mps"
                            ],
                        )
                    ),
                    "WD_improve_vs_Base4_fraction": (
                        rel_improve(
                            base_metrics[
                                "wind_direction_RMSE_deg"
                            ],
                            metrics[
                                "wind_direction_RMSE_deg"
                            ],
                        )
                    ),
                }
            )

    runs = pd.DataFrame(
        rows
    )

    summary = (
        runs.groupby(
            "group",
            as_index=False,
        )
        .agg(
            feature_count_per_step=(
                "feature_count_per_step",
                "first",
            ),
            parameter_count_mean=(
                "parameter_count",
                "mean",
            ),
            v_focus_score_mean=(
                "v_focus_score",
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
            vector_RMSE_mean=(
                "wind_vector_RMSE_mps",
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
            U_improve_mean=(
                "U_improve_vs_Base4_fraction",
                "mean",
            ),
            V_improve_mean=(
                "V_improve_vs_Base4_fraction",
                "mean",
            ),
            WS_improve_mean=(
                "WS_improve_vs_Base4_fraction",
                "mean",
            ),
            WD_improve_mean=(
                "WD_improve_vs_Base4_fraction",
                "mean",
            ),
        )
        .sort_values(
            [
                "v_focus_score_mean",
                "V_RMSE_mean",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    wins = (
        runs.assign(
            win=lambda x:
            x[
                "V_improve_vs_Base4_fraction"
            ]
            > 0
        )
        .groupby(
            "group"
        )[
            "win"
        ]
        .sum()
        .astype(int)
        .to_dict()
    )

    summary[
        "mission_V_wins_vs_Base4"
    ] = summary[
        "group"
    ].map(
        wins
    ).fillna(
        0
    ).astype(
        int
    )

    return (
        runs,
        summary,
    )


def run_development_residual(
    dev_missions,
    output_dir,
):
    rows = []

    pred_dir = (
        output_dir
        / "development_residual_predictions"
    )

    pred_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for held_out in DEV_MISSIONS:
        train_data = concat_missions(
            [
                dev_missions[
                    m
                ]
                for m in DEV_MISSIONS
                if m
                != held_out
            ]
        )

        val_data = dev_missions[
            held_out
        ]

        Xtr_base4 = select_features(
            train_data[
                "X"
            ],
            DIRECT_GROUPS[
                "Base4"
            ],
        )

        Xva_base4 = select_features(
            val_data[
                "X"
            ],
            DIRECT_GROUPS[
                "Base4"
            ],
        )

        base_model, base_scaler = fit_direct_ridge(
            Xtr_base4,
            train_data[
                "y"
            ],
        )

        ridge_val = predict_direct_ridge(
            base_model,
            base_scaler,
            Xva_base4,
        )

        base_metrics = evaluate_wind(
            val_data[
                "y"
            ],
            ridge_val,
        )

        ridge_oof = base4_oof_predictions(
            train_data
        )

        residual_target = (
            train_data[
                "y"
            ]
            - ridge_oof
        ).astype(
            np.float32
        )

        true_val_res = (
            val_data[
                "y"
            ]
            - ridge_val
        )

        for group_name, aux_names in RESIDUAL_GROUPS.items():
            Xprobe_train = build_probe_matrix(
                train_data[
                    "X"
                ],
                aux_names,
                ridge_oof,
            )

            Xprobe_val = build_probe_matrix(
                val_data[
                    "X"
                ],
                aux_names,
                ridge_val,
            )

            model, scaler = fit_matrix_ridge(
                Xprobe_train,
                residual_target,
                RESIDUAL_RIDGE_ALPHA,
            )

            raw_val = predict_matrix_ridge(
                model,
                scaler,
                Xprobe_val,
            )

            alphas = analytical_alphas(
                val_data[
                    "y"
                ],
                ridge_val,
                raw_val,
            )

            final_val = apply_residual(
                ridge_val,
                raw_val,
                alphas,
            )

            metrics = evaluate_wind(
                val_data[
                    "y"
                ],
                final_val,
            )

            corr_u = corrcoef_safe(
                true_val_res[
                    :,
                    0
                ],
                raw_val[
                    :,
                    0
                ],
            )

            corr_v = corrcoef_safe(
                true_val_res[
                    :,
                    1
                ],
                raw_val[
                    :,
                    1
                ],
            )

            prediction_path = (
                pred_dir
                / (
                    f"{group_name}"
                    f"__holdout_{held_out.replace(' ', '_')}.npz"
                )
            )

            np.savez_compressed(
                prediction_path,
                y_true=val_data[
                    "y"
                ].astype(
                    np.float32
                ),
                ridge_pred=ridge_val.astype(
                    np.float32
                ),
                raw_residual_pred=raw_val.astype(
                    np.float32
                ),
                fold_oracle_alphas=alphas.astype(
                    np.float32
                ),
            )

            rows.append(
                {
                    "held_out_mission": (
                        held_out
                    ),
                    "group": (
                        group_name
                    ),
                    "aux_feature_count_per_step": int(
                        len(
                            aux_names
                        )
                    ),
                    "probe_input_dim": int(
                        Xprobe_train.shape[
                            1
                        ]
                    ),
                    "probe_parameter_count": int(
                        np.asarray(
                            model.coef_
                        ).size
                        + np.asarray(
                            model.intercept_
                        ).size
                    ),
                    "raw_residual_corr_U": float(
                        corr_u
                    ),
                    "raw_residual_corr_V": float(
                        corr_v
                    ),
                    "fold_oracle_alpha_U": float(
                        alphas[
                            0
                        ]
                    ),
                    "fold_oracle_alpha_V": float(
                        alphas[
                            1
                        ]
                    ),
                    "v_focus_score": float(
                        v_focus_score(
                            metrics,
                            base_metrics,
                        )
                    ),
                    **metrics,
                    "U_improve_vs_Base4_fraction": (
                        rel_improve(
                            base_metrics[
                                "wind_U_RMSE_mps"
                            ],
                            metrics[
                                "wind_U_RMSE_mps"
                            ],
                        )
                    ),
                    "V_improve_vs_Base4_fraction": (
                        rel_improve(
                            base_metrics[
                                "wind_V_RMSE_mps"
                            ],
                            metrics[
                                "wind_V_RMSE_mps"
                            ],
                        )
                    ),
                    "WS_improve_vs_Base4_fraction": (
                        rel_improve(
                            base_metrics[
                                "wind_speed_RMSE_mps"
                            ],
                            metrics[
                                "wind_speed_RMSE_mps"
                            ],
                        )
                    ),
                    "WD_improve_vs_Base4_fraction": (
                        rel_improve(
                            base_metrics[
                                "wind_direction_RMSE_deg"
                            ],
                            metrics[
                                "wind_direction_RMSE_deg"
                            ],
                        )
                    ),
                    "prediction_path": str(
                        prediction_path
                    ),
                }
            )

    runs = pd.DataFrame(
        rows
    )

    summary = (
        runs.groupby(
            "group",
            as_index=False,
        )
        .agg(
            aux_feature_count_per_step=(
                "aux_feature_count_per_step",
                "first",
            ),
            probe_input_dim=(
                "probe_input_dim",
                "first",
            ),
            probe_parameter_count_mean=(
                "probe_parameter_count",
                "mean",
            ),
            v_focus_score_mean=(
                "v_focus_score",
                "mean",
            ),
            raw_residual_corr_U_mean=(
                "raw_residual_corr_U",
                "mean",
            ),
            raw_residual_corr_V_mean=(
                "raw_residual_corr_V",
                "mean",
            ),
            fold_oracle_alpha_U_mean=(
                "fold_oracle_alpha_U",
                "mean",
            ),
            fold_oracle_alpha_V_mean=(
                "fold_oracle_alpha_V",
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
            U_improve_mean=(
                "U_improve_vs_Base4_fraction",
                "mean",
            ),
            V_improve_mean=(
                "V_improve_vs_Base4_fraction",
                "mean",
            ),
            WS_improve_mean=(
                "WS_improve_vs_Base4_fraction",
                "mean",
            ),
            WD_improve_mean=(
                "WD_improve_vs_Base4_fraction",
                "mean",
            ),
        )
        .sort_values(
            [
                "v_focus_score_mean",
                "V_RMSE_mean",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    wins = (
        runs.assign(
            win=lambda x:
            x[
                "V_improve_vs_Base4_fraction"
            ]
            > 0
        )
        .groupby(
            "group"
        )[
            "win"
        ]
        .sum()
        .astype(int)
        .to_dict()
    )

    summary[
        "mission_V_wins_vs_Base4"
    ] = summary[
        "group"
    ].map(
        wins
    ).fillna(
        0
    ).astype(
        int
    )

    return (
        runs,
        summary,
    )


def pooled_residual_calibration(
    residual_runs,
):
    rows = []
    frozen_alphas = {}

    for group_name in RESIDUAL_GROUPS:
        sub = residual_runs.loc[
            residual_runs[
                "group"
            ]
            == group_name
        ].copy()

        pooled_y = []
        pooled_ridge = []
        pooled_raw = []

        for path_string in sub[
            "prediction_path"
        ].tolist():
            path = Path(
                str(
                    path_string
                )
            )

            with np.load(
                path,
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

        y = np.concatenate(
            pooled_y,
            axis=0,
        )

        ridge = np.concatenate(
            pooled_ridge,
            axis=0,
        )

        raw = np.concatenate(
            pooled_raw,
            axis=0,
        )

        alphas = analytical_alphas(
            y,
            ridge,
            raw,
        )

        final = apply_residual(
            ridge,
            raw,
            alphas,
        )

        base_metrics = evaluate_wind(
            y,
            ridge,
        )

        final_metrics = evaluate_wind(
            y,
            final,
        )

        true_res = (
            y
            - ridge
        )

        frozen_alphas[
            group_name
        ] = {
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
        }

        rows.append(
            {
                "group": (
                    group_name
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
                "pooled_raw_corr_U": float(
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
                "pooled_raw_corr_V": float(
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
                "pooled_Base4_U_RMSE": (
                    base_metrics[
                        "wind_U_RMSE_mps"
                    ]
                ),
                "pooled_Base4_V_RMSE": (
                    base_metrics[
                        "wind_V_RMSE_mps"
                    ]
                ),
                "pooled_Final_U_RMSE": (
                    final_metrics[
                        "wind_U_RMSE_mps"
                    ]
                ),
                "pooled_Final_V_RMSE": (
                    final_metrics[
                        "wind_V_RMSE_mps"
                    ]
                ),
                "U_improve_fraction": (
                    rel_improve(
                        base_metrics[
                            "wind_U_RMSE_mps"
                        ],
                        final_metrics[
                            "wind_U_RMSE_mps"
                        ],
                    )
                ),
                "V_improve_fraction": (
                    rel_improve(
                        base_metrics[
                            "wind_V_RMSE_mps"
                        ],
                        final_metrics[
                            "wind_V_RMSE_mps"
                        ],
                    )
                ),
            }
        )

    table = pd.DataFrame(
        rows
    ).sort_values(
        [
            "pooled_Final_V_RMSE",
            "pooled_raw_corr_V",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    return (
        table,
        frozen_alphas,
    )


# =============================================================================
# Final Tropical evaluation
# =============================================================================

def final_direct_attribution(
    dev_all,
    tropical,
):
    rows = []

    base_train = select_features(
        dev_all[
            "X"
        ],
        DIRECT_GROUPS[
            "Base4"
        ],
    )

    base_test = select_features(
        tropical[
            "X"
        ],
        DIRECT_GROUPS[
            "Base4"
        ],
    )

    base_model, base_scaler = fit_direct_ridge(
        base_train,
        dev_all[
            "y"
        ],
    )

    base_pred = predict_direct_ridge(
        base_model,
        base_scaler,
        base_test,
    )

    base_metrics = evaluate_wind(
        tropical[
            "y"
        ],
        base_pred,
    )

    for group_name, names in DIRECT_GROUPS.items():
        Xtr = select_features(
            dev_all[
                "X"
            ],
            names,
        )

        Xte = select_features(
            tropical[
                "X"
            ],
            names,
        )

        model, scaler = fit_direct_ridge(
            Xtr,
            dev_all[
                "y"
            ],
        )

        pred = predict_direct_ridge(
            model,
            scaler,
            Xte,
        )

        metrics = evaluate_wind(
            tropical[
                "y"
            ],
            pred,
        )

        rows.append(
            {
                "group": group_name,
                "feature_count_per_step": int(
                    len(
                        names
                    )
                ),
                "parameter_count": int(
                    direct_parameter_count(
                        model
                    )
                ),
                **metrics,
                "U_improve_vs_Base4_fraction": (
                    rel_improve(
                        base_metrics[
                            "wind_U_RMSE_mps"
                        ],
                        metrics[
                            "wind_U_RMSE_mps"
                        ],
                    )
                ),
                "V_improve_vs_Base4_fraction": (
                    rel_improve(
                        base_metrics[
                            "wind_V_RMSE_mps"
                        ],
                        metrics[
                            "wind_V_RMSE_mps"
                        ],
                    )
                ),
                "WS_improve_vs_Base4_fraction": (
                    rel_improve(
                        base_metrics[
                            "wind_speed_RMSE_mps"
                        ],
                        metrics[
                            "wind_speed_RMSE_mps"
                        ],
                    )
                ),
                "WD_improve_vs_Base4_fraction": (
                    rel_improve(
                        base_metrics[
                            "wind_direction_RMSE_deg"
                        ],
                        metrics[
                            "wind_direction_RMSE_deg"
                        ],
                    )
                ),
                "beats_BP_U": bool(
                    metrics[
                        "wind_U_RMSE_mps"
                    ]
                    < BP_REFERENCE[
                        "U_RMSE_mps"
                    ]
                ),
                "beats_BP_V": bool(
                    metrics[
                        "wind_V_RMSE_mps"
                    ]
                    < BP_REFERENCE[
                        "V_RMSE_mps"
                    ]
                ),
                "beats_BP_WS": bool(
                    metrics[
                        "wind_speed_RMSE_mps"
                    ]
                    < BP_REFERENCE[
                        "WS_RMSE_mps"
                    ]
                ),
                "beats_BP_WD": bool(
                    metrics[
                        "wind_direction_RMSE_deg"
                    ]
                    < BP_REFERENCE[
                        "WD_RMSE_deg"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "wind_V_RMSE_mps",
            "wind_U_RMSE_mps",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )


def final_residual_attribution(
    dev_all,
    tropical,
    frozen_alphas,
):
    rows = []

    Xdev_base4 = select_features(
        dev_all[
            "X"
        ],
        DIRECT_GROUPS[
            "Base4"
        ],
    )

    Xtrop_base4 = select_features(
        tropical[
            "X"
        ],
        DIRECT_GROUPS[
            "Base4"
        ],
    )

    base_model, base_scaler = fit_direct_ridge(
        Xdev_base4,
        dev_all[
            "y"
        ],
    )

    ridge_trop = predict_direct_ridge(
        base_model,
        base_scaler,
        Xtrop_base4,
    )

    base_metrics = evaluate_wind(
        tropical[
            "y"
        ],
        ridge_trop,
    )

    ridge_oof = base4_oof_predictions(
        dev_all
    )

    residual_target = (
        dev_all[
            "y"
        ]
        - ridge_oof
    ).astype(
        np.float32
    )

    true_trop_res = (
        tropical[
            "y"
        ]
        - ridge_trop
    )

    for group_name, aux_names in RESIDUAL_GROUPS.items():
        Xprobe_train = build_probe_matrix(
            dev_all[
                "X"
            ],
            aux_names,
            ridge_oof,
        )

        Xprobe_trop = build_probe_matrix(
            tropical[
                "X"
            ],
            aux_names,
            ridge_trop,
        )

        model, scaler = fit_matrix_ridge(
            Xprobe_train,
            residual_target,
            RESIDUAL_RIDGE_ALPHA,
        )

        raw_trop = predict_matrix_ridge(
            model,
            scaler,
            Xprobe_trop,
        )

        alpha_info = frozen_alphas[
            group_name
        ]

        alphas = np.asarray(
            [
                alpha_info[
                    "alpha_U"
                ],
                alpha_info[
                    "alpha_V"
                ],
            ],
            dtype=np.float32,
        )

        final = apply_residual(
            ridge_trop,
            raw_trop,
            alphas,
        )

        metrics = evaluate_wind(
            tropical[
                "y"
            ],
            final,
        )

        rows.append(
            {
                "group": (
                    group_name
                ),
                "aux_feature_count_per_step": int(
                    len(
                        aux_names
                    )
                ),
                "probe_input_dim": int(
                    Xprobe_train.shape[
                        1
                    ]
                ),
                "probe_parameter_count": int(
                    np.asarray(
                        model.coef_
                    ).size
                    + np.asarray(
                        model.intercept_
                    ).size
                ),
                "frozen_alpha_U": float(
                    alphas[
                        0
                    ]
                ),
                "frozen_alpha_V": float(
                    alphas[
                        1
                    ]
                ),
                "Tropical_raw_corr_U": float(
                    corrcoef_safe(
                        true_trop_res[
                            :,
                            0
                        ],
                        raw_trop[
                            :,
                            0
                        ],
                    )
                ),
                "Tropical_raw_corr_V": float(
                    corrcoef_safe(
                        true_trop_res[
                            :,
                            1
                        ],
                        raw_trop[
                            :,
                            1
                        ],
                    )
                ),
                **metrics,
                "U_improve_vs_Base4_fraction": (
                    rel_improve(
                        base_metrics[
                            "wind_U_RMSE_mps"
                        ],
                        metrics[
                            "wind_U_RMSE_mps"
                        ],
                    )
                ),
                "V_improve_vs_Base4_fraction": (
                    rel_improve(
                        base_metrics[
                            "wind_V_RMSE_mps"
                        ],
                        metrics[
                            "wind_V_RMSE_mps"
                        ],
                    )
                ),
                "WS_improve_vs_Base4_fraction": (
                    rel_improve(
                        base_metrics[
                            "wind_speed_RMSE_mps"
                        ],
                        metrics[
                            "wind_speed_RMSE_mps"
                        ],
                    )
                ),
                "WD_improve_vs_Base4_fraction": (
                    rel_improve(
                        base_metrics[
                            "wind_direction_RMSE_deg"
                        ],
                        metrics[
                            "wind_direction_RMSE_deg"
                        ],
                    )
                ),
                "beats_BP_U": bool(
                    metrics[
                        "wind_U_RMSE_mps"
                    ]
                    < BP_REFERENCE[
                        "U_RMSE_mps"
                    ]
                ),
                "beats_BP_V": bool(
                    metrics[
                        "wind_V_RMSE_mps"
                    ]
                    < BP_REFERENCE[
                        "V_RMSE_mps"
                    ]
                ),
                "beats_BP_WS": bool(
                    metrics[
                        "wind_speed_RMSE_mps"
                    ]
                    < BP_REFERENCE[
                        "WS_RMSE_mps"
                    ]
                ),
                "beats_BP_WD": bool(
                    metrics[
                        "wind_direction_RMSE_deg"
                    ]
                    < BP_REFERENCE[
                        "WD_RMSE_deg"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "wind_V_RMSE_mps",
            "Tropical_raw_corr_V",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )


# =============================================================================
# Interpretation helper
# =============================================================================

def best_group_excluding_base(
    table,
    group_col,
    metric_col,
    excluded,
):
    sub = table.loc[
        ~table[
            group_col
        ].isin(
            excluded
        )
    ].copy()

    if sub.empty:
        return None

    return str(
        sub.sort_values(
            metric_col,
            ascending=True,
        ).iloc[
            0
        ][
            group_col
        ]
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--debug-fast",
        action="store_true",
        help=(
            "Smoke test only: Antarctic holdout and a reduced predeclared "
            "group list. Tropical Atlantic is NOT loaded."
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

    log("=" * 132)
    log(
        "13A — BP-OWNED INPUT ATTRIBUTION + BASE4 RESIDUAL PROBE"
    )
    log("=" * 132)
    log(
        f"dataset : {dataset_root}"
    )
    log(
        f"output  : {output_dir}"
    )
    log(
        "protocol: Stage12G point sampled, 6 history -> +10 min"
    )
    log(
        "goal    : identify which BP-owned auxiliary information can reduce V"
    )

    # ---------------------------------------------------------
    # Development load only
    # ---------------------------------------------------------
    log("")
    log(
        "[STAGE A] Loading development missions only."
    )

    dev_missions = {}

    alignment_rows = []

    for mission in DEV_MISSIONS:
        item = load_aligned_mission(
            dataset_root,
            mission,
        )

        dev_missions[
            mission
        ] = item

        alignment_rows.append(
            item[
                "alignment_audit"
            ]
        )

        log(
            f"  {mission:<12} N={len(item['y']):,} | "
            f"P/J aligned"
        )

    alignment_df = pd.DataFrame(
        alignment_rows
    )

    alignment_path = (
        output_dir
        / "development_trackPJ_alignment_audit.csv"
    )

    alignment_df.to_csv(
        alignment_path,
        index=False,
        encoding="utf-8-sig",
    )

    log(
        "[FIREWALL] Tropical Atlantic NOT loaded."
    )

    if args.debug_fast:
        # Smoke-test the core loaders/models without test access.
        held_out = "Antarctic"

        train_data = concat_missions(
            [
                dev_missions[
                    "Atlantic"
                ],
                dev_missions[
                    "West Coast"
                ],
            ]
        )

        val_data = dev_missions[
            held_out
        ]

        for group_name in [
            "Base4",
            "Base4+Motion",
            "Base4+PressureDyn",
            "BPFullEngineered",
        ]:
            names = DIRECT_GROUPS[
                group_name
            ]

            model, scaler = fit_direct_ridge(
                select_features(
                    train_data[
                        "X"
                    ],
                    names,
                ),
                train_data[
                    "y"
                ],
            )

            pred = predict_direct_ridge(
                model,
                scaler,
                select_features(
                    val_data[
                        "X"
                    ],
                    names,
                ),
            )

            m = evaluate_wind(
                val_data[
                    "y"
                ],
                pred,
            )

            log(
                f"  [DIRECT SMOKE] {group_name:<28} "
                f"U={m['wind_U_RMSE_mps']:.4f} "
                f"V={m['wind_V_RMSE_mps']:.4f}"
            )

        base_oof = base4_oof_predictions(
            train_data
        )

        base_model, base_scaler = fit_direct_ridge(
            select_features(
                train_data[
                    "X"
                ],
                DIRECT_GROUPS[
                    "Base4"
                ],
            ),
            train_data[
                "y"
            ],
        )

        base_val = predict_direct_ridge(
            base_model,
            base_scaler,
            select_features(
                val_data[
                    "X"
                ],
                DIRECT_GROUPS[
                    "Base4"
                ],
            ),
        )

        for group_name in [
            "Motion",
            "PressureDyn",
            "AllAux",
        ]:
            aux_names = RESIDUAL_GROUPS[
                group_name
            ]

            probe_model, probe_scaler = fit_matrix_ridge(
                build_probe_matrix(
                    train_data[
                        "X"
                    ],
                    aux_names,
                    base_oof,
                ),
                train_data[
                    "y"
                ]
                - base_oof,
                RESIDUAL_RIDGE_ALPHA,
            )

            raw = predict_matrix_ridge(
                probe_model,
                probe_scaler,
                build_probe_matrix(
                    val_data[
                        "X"
                    ],
                    aux_names,
                    base_val,
                ),
            )

            corr_v = corrcoef_safe(
                val_data[
                    "y"
                ][
                    :,
                    1
                ]
                - base_val[
                    :,
                    1
                ],
                raw[
                    :,
                    1
                ],
            )

            log(
                f"  [RESIDUAL SMOKE] {group_name:<24} "
                f"corrV={corr_v:+.4f}"
            )

        log("")
        log(
            "[DONE] debug-fast PASS. Tropical Atlantic was NOT loaded."
        )
        return 0

    # ---------------------------------------------------------
    # Development direct attribution
    # ---------------------------------------------------------
    log("")
    log(
        "[STAGE B1] Development LOMO — direct Ridge input attribution."
    )

    direct_runs, direct_summary = run_development_direct(
        dev_missions
    )

    direct_runs_path = (
        output_dir
        / "development_direct_ridge_runs.csv"
    )

    direct_summary_path = (
        output_dir
        / "development_direct_ridge_summary.csv"
    )

    direct_runs.to_csv(
        direct_runs_path,
        index=False,
        encoding="utf-8-sig",
    )

    direct_summary.to_csv(
        direct_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log(
        "DEVELOPMENT DIRECT RIDGE SUMMARY"
    )
    log(
        direct_summary.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Development residual probes
    # ---------------------------------------------------------
    log("")
    log(
        "[STAGE B2] Development LOMO — Base4 residual auxiliary probes."
    )

    residual_runs, residual_summary = run_development_residual(
        dev_missions,
        output_dir,
    )

    residual_runs_path = (
        output_dir
        / "development_residual_probe_runs.csv"
    )

    residual_summary_path = (
        output_dir
        / "development_residual_probe_summary.csv"
    )

    residual_runs.to_csv(
        residual_runs_path,
        index=False,
        encoding="utf-8-sig",
    )

    residual_summary.to_csv(
        residual_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log(
        "DEVELOPMENT RESIDUAL PROBE SUMMARY"
    )
    log(
        residual_summary.to_string(
            index=False
        )
    )

    pooled_calibration, frozen_alphas = (
        pooled_residual_calibration(
            residual_runs
        )
    )

    pooled_path = (
        output_dir
        / "development_residual_pooled_calibration.csv"
    )

    pooled_calibration.to_csv(
        pooled_path,
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log(
        "DEVELOPMENT POOLED RESIDUAL CALIBRATION"
    )
    log(
        pooled_calibration.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Freeze everything before Tropical
    # ---------------------------------------------------------
    frozen = {
        "stage": "13A",
        "script_version": (
            SCRIPT_VERSION
        ),
        "direct_groups_predeclared": (
            DIRECT_GROUPS
        ),
        "residual_groups_predeclared": (
            RESIDUAL_GROUPS
        ),
        "frozen_residual_alphas": (
            frozen_alphas
        ),
        "development_direct_ranking": (
            direct_summary[
                [
                    "group",
                    "v_focus_score_mean",
                    "V_RMSE_mean",
                    "V_improve_mean",
                    "mission_V_wins_vs_Base4",
                ]
            ].to_dict(
                orient="records"
            )
        ),
        "development_residual_ranking": (
            residual_summary[
                [
                    "group",
                    "v_focus_score_mean",
                    "raw_residual_corr_V_mean",
                    "V_RMSE_mean",
                    "V_improve_mean",
                    "mission_V_wins_vs_Base4",
                ]
            ].to_dict(
                orient="records"
            )
        ),
        "Tropical_Atlantic_used_for_group_definition": False,
        "Tropical_Atlantic_used_for_alpha_calibration": False,
        "scientific_role": (
            "predeclared exploratory input-attribution benchmark"
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
    log(
        f"[FROZEN BEFORE TROPICAL] {frozen_path}"
    )

    # ---------------------------------------------------------
    # Tropical final attribution
    # ---------------------------------------------------------
    log("")
    log(
        "[STAGE C] All feature groups and residual alphas frozen. "
        "Loading Tropical Atlantic NOW."
    )

    tropical = load_aligned_mission(
        dataset_root,
        FINAL_MISSION,
    )

    log(
        f"[TROPICAL] N={len(tropical['y']):,} | "
        f"file={mission_filename(FINAL_MISSION)}"
    )

    dev_all = concat_missions(
        [
            dev_missions[
                m
            ]
            for m in DEV_MISSIONS
        ]
    )

    tropical_direct = final_direct_attribution(
        dev_all,
        tropical,
    )

    tropical_residual = final_residual_attribution(
        dev_all,
        tropical,
        frozen_alphas,
    )

    tropical_direct_path = (
        output_dir
        / "FINAL_TROPICAL_DIRECT_RIDGE_ATTRIBUTION.csv"
    )

    tropical_residual_path = (
        output_dir
        / "FINAL_TROPICAL_RESIDUAL_PROBE_ATTRIBUTION.csv"
    )

    tropical_direct.to_csv(
        tropical_direct_path,
        index=False,
        encoding="utf-8-sig",
    )

    tropical_residual.to_csv(
        tropical_residual_path,
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log("=" * 132)
    log(
        "FINAL TROPICAL — DIRECT RIDGE INPUT ATTRIBUTION"
    )
    log("=" * 132)
    log(
        tropical_direct.to_string(
            index=False
        )
    )

    log("")
    log("=" * 132)
    log(
        "FINAL TROPICAL — BASE4 + AUXILIARY RESIDUAL PROBE"
    )
    log("=" * 132)
    log(
        tropical_residual.to_string(
            index=False
        )
    )

    best_direct_v = str(
        tropical_direct.iloc[
            0
        ][
            "group"
        ]
    )

    best_residual_v = str(
        tropical_residual.iloc[
            0
        ][
            "group"
        ]
    )

    best_dev_direct = str(
        direct_summary.iloc[
            0
        ][
            "group"
        ]
    )

    best_dev_residual = str(
        residual_summary.iloc[
            0
        ][
            "group"
        ]
    )

    report = {
        "stage": "13A",
        "script_version": (
            SCRIPT_VERSION
        ),
        "published_BP_STGNN": (
            BP_REFERENCE
        ),
        "best_development_direct_group": (
            best_dev_direct
        ),
        "best_development_residual_group": (
            best_dev_residual
        ),
        "best_Tropical_direct_V_group": (
            best_direct_v
        ),
        "best_Tropical_residual_V_group": (
            best_residual_v
        ),
        "frozen_before_tropical": (
            frozen
        ),
        "interpretation_guardrail": (
            "Use Tropical results to diagnose cross-mission auxiliary "
            "information, not to claim a pristine untouched final test."
        ),
    }

    report_json = (
        output_dir
        / "13A_REPORT.json"
    )

    save_json(
        report_json,
        report,
    )

    report_txt = (
        output_dir
        / "13A_REPORT.txt"
    )

    with report_txt.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "13A BP-OWNED INPUT ATTRIBUTION REPORT\n"
        )
        f.write(
            "=" * 120
            + "\n\n"
        )

        f.write(
            "DEVELOPMENT DIRECT RIDGE SUMMARY\n"
        )
        f.write(
            "-" * 120
            + "\n"
        )
        f.write(
            direct_summary.to_string(
                index=False
            )
        )
        f.write(
            "\n\n"
        )

        f.write(
            "DEVELOPMENT RESIDUAL PROBE SUMMARY\n"
        )
        f.write(
            "-" * 120
            + "\n"
        )
        f.write(
            residual_summary.to_string(
                index=False
            )
        )
        f.write(
            "\n\n"
        )

        f.write(
            "DEVELOPMENT POOLED RESIDUAL CALIBRATION\n"
        )
        f.write(
            "-" * 120
            + "\n"
        )
        f.write(
            pooled_calibration.to_string(
                index=False
            )
        )
        f.write(
            "\n\n"
        )

        f.write(
            "FINAL TROPICAL DIRECT RIDGE ATTRIBUTION\n"
        )
        f.write(
            "-" * 120
            + "\n"
        )
        f.write(
            tropical_direct.to_string(
                index=False
            )
        )
        f.write(
            "\n\n"
        )

        f.write(
            "FINAL TROPICAL RESIDUAL PROBE ATTRIBUTION\n"
        )
        f.write(
            "-" * 120
            + "\n"
        )
        f.write(
            tropical_residual.to_string(
                index=False
            )
        )
        f.write(
            "\n\n"
        )

        f.write(
            f"Best development direct group: {best_dev_direct}\n"
        )
        f.write(
            f"Best development residual group: {best_dev_residual}\n"
        )
        f.write(
            f"Best Tropical direct V group: {best_direct_v}\n"
        )
        f.write(
            f"Best Tropical residual V group: {best_residual_v}\n"
        )

    log("")
    log(
        f"[BEST DEV DIRECT] {best_dev_direct}"
    )
    log(
        f"[BEST DEV RESIDUAL] {best_dev_residual}"
    )
    log(
        f"[BEST TROPICAL DIRECT V] {best_direct_v}"
    )
    log(
        f"[BEST TROPICAL RESIDUAL V] {best_residual_v}"
    )

    log("")
    log(
        f"[SAVED] {tropical_direct_path}"
    )
    log(
        f"[SAVED] {tropical_residual_path}"
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
