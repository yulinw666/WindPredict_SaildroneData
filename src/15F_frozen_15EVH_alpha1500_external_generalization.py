# -*- coding: utf-8 -*-
"""
15F_frozen_15EVH_external_generalization.py

15F — Frozen external generalization of FINAL 15E-VH (joint Ridge alpha=1500)
======================================================

FINAL frozen model
------------------
Atmospheric branch:
    Dedicated TrueWind Ridge:
        [U,V,T,RH] over 60 one-minute observations -> 5 x [U,V]

    Stage15B:
        2-layer GRU, hidden=48
        nonlinear true-wind residual
        horizon/component validation shrinkage alpha in [0,1]

Motion branch:
    Old joint Ridge, alpha=100:
        all 11 historical features -> 5 x
        [U,V,VE,VN,HDG_sin,HDG_cos]

    15E-VH:
        ONE single-layer GRU64 + Dense64
        jointly corrects
        [VE,VN,HDG_sin,HDG_cos]
        using sample-wise gated residuals.

Physics:
    A_earth = W_true - V_vessel
    HDG rotates A_earth to body frame for AWA.
    AWS is the vector magnitude.

Strict external protocol
------------------------
PHASE A — DEVELOPMENT FREEZE:
    * Load SD1090 TRAIN/VALIDATION only.
    * Rebuild deterministic Ridge anchors.
    * Re-train/freeze Stage15B for the five already frozen seeds.
    * Load the already trained 15E-VH motion checkpoints.
    * NO external labels/data are used for model selection.

PHASE B — EXTERNAL INFERENCE:
    * Only after every development component is frozen, load SD1033.
    * No external optimizer step.
    * No external scaler fit.
    * No external alpha fit.
    * No external checkpoint/epoch selection.
    * No external seed selection.
    * No external calibration.

Primary external products
-------------------------
Expected prior products:
    SD1033-2024 matched SD1090 : 111,245 windows
    SD1033-2023 full           : 174,458 windows
    SD1033-2022 full           : 106,584 windows

Optional supplementary:
    SD1033-2024 full           : 115,618 windows

The script can:
    1) accept explicit file/directory paths, OR
    2) auto-discover NPZ products containing "SD1033" under --external-root.

Required external NPZ schema
----------------------------
    X      : (N,60,11)
    y_raw  : (N,5,6)

The X array must be the previously prepared external product using the
frozen SD1090 input-standardization convention. This script never refits
an external input scaler.

Apparent-wind truth is deterministically reconstructed as:
    A_true_earth = W_true - V_true
so no external apparent-wind calibration is required.

Five frozen seeds
-----------------
    500043, 501052, 502061, 503070, 504079

Outputs
-------
15F_external_dataset_summary.csv
15F_external_per_seed_summary.csv
15F_external_per_horizon.csv
15F_external_branch_decomposition.csv
15F_external_vs_old_manuscript.csv
15F_STAGE15B_development_freeze.csv
15F_REPORT.json
15F_REPORT.txt

Reference old-paper Physics-Compact AW vector RMSE
---------------------------------------------------
SD1033-2024 matched : 0.8974 m/s
SD1033-2023 full    : 0.9146 m/s
SD1033-2022 full    : 0.8651 m/s

Reference modern baselines from old paper
-----------------------------------------
TimeMixer:
    2024 matched 0.9340
    2023 full    1.0293
    2022 full    0.9271

DLinear:
    2024 matched 0.9390
    2023 full    1.0792
    2022 full    0.9320

IMPORTANT
---------
The three SD1033 datasets were already examined in the previous manuscript
development. Therefore 15F is a historical external-transfer comparison,
NOT a newly pristine final test.

A genuinely untouched new Saildrone mission is still required for the
strongest independent final-generalization claim.

Recommended PowerShell
----------------------
Auto-discovery:

python "D:\\project\\WindPredict_SaildroneData\\src\\15F_frozen_15EVH_external_generalization.py" --dataset-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\SD1090_TPOS2024_JointForecasting_v0_1" --stage15a-truewind-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15A_R_1min_Dual_Ridge_Anchors_v0_2" --stage15evh-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15E_VH_AlphaSearch_v0_1\\full_retrain\\alpha_1500" --src-dir "D:\\project\\WindPredict_SaildroneData\\src" --external-root "D:\\project\\WindPredict_SaildroneData\\data" --joint-ridge-alpha 1500 --output-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15F_Frozen_15EVH_alpha1500_External_v0_1"

Explicit paths can additionally be supplied:
    --external-2024-matched <path>
    --external-2023 <path>
    --external-2022 <path>
    --external-2024-full <path>
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import math
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Frozen constants
# =============================================================================

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

DEFAULT_STAGE15EVH_DIR = (
    ROOT
    / "data"
    / "forecasting"
    / "15E_VH_JointVesselHDG_5seed_v0_1"
)

DEFAULT_SRC_DIR = (
    ROOT
    / "src"
)

DEFAULT_EXTERNAL_ROOT = (
    ROOT
    / "data"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "data"
    / "forecasting"
    / "15F_Frozen_15EVH_External_v0_1"
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

EXPECTED_EXTERNAL_COUNTS = {
    "SD1033-2024-matched": 111245,
    "SD1033-2023-full": 174458,
    "SD1033-2022-full": 106584,
    "SD1033-2024-full": 115618,
}

PRIMARY_DATASETS = [
    "SD1033-2024-matched",
    "SD1033-2023-full",
    "SD1033-2022-full",
]

SUPPLEMENTARY_DATASET = (
    "SD1033-2024-full"
)

OLD_PC_AW = {
    "SD1033-2024-matched": 0.8974,
    "SD1033-2023-full": 0.9146,
    "SD1033-2022-full": 0.8651,
}

OLD_TIMEMIXER_AW = {
    "SD1033-2024-matched": 0.9340,
    "SD1033-2023-full": 1.0293,
    "SD1033-2022-full": 0.9271,
}

OLD_DLINEAR_AW = {
    "SD1033-2024-matched": 0.9390,
    "SD1033-2023-full": 1.0792,
    "SD1033-2022-full": 0.9320,
}

TRUEWIND_RIDGE_ALPHA = 0.0
OLD_JOINT_RIDGE_ALPHA = 100.0

STAGE15B_PARAMS = 26074
MOTION_PARAMS = 23464
TOTAL_NEURAL_PARAMS = 49538

EPS = 1e-12

# If a deterministic rebuild of the dedicated TrueWind Ridge differs from
# the frozen Stage15A validation anchor by more than this, abort rather than
# silently use an incompatible anchor.
TRUEWIND_RIDGE_VALIDATION_TOL = 2e-4

# Repeated Stage15B training can differ microscopically on CUDA/WDDM.
# This threshold is diagnostic only; it is not used for selection.
STAGE15B_REPRO_WARN_THRESHOLD = 0.01


# =============================================================================
# Utilities
# =============================================================================

def log(msg=""):
    print(
        msg,
        flush=True,
    )


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
        str(
            path
        ),
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


def normalize_heading_pair(
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
        1e-8,
    )

    return (
        q / n
    ).astype(
        np.float32
    )


def mean_and_std(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return (
        float(
            np.mean(
                values
            )
        ),
        float(
            np.std(
                values,
                ddof=1,
            )
        )
        if len(
            values
        ) > 1
        else 0.0,
    )


# =============================================================================
# Dedicated TrueWind Ridge
# =============================================================================

def fit_truewind_ridge(
    emod,
    X_train,
    y_train,
):
    """
    Rebuild the frozen dedicated atmospheric Ridge:
        flattened 60 x [U,V,T,RH] -> 5 x [U,V]
        alpha = 0

    Uses the same centered Gram-matrix / standardized-target formulation
    as the already validated old-joint-Ridge helper.
    """
    Xf = np.asarray(
        X_train[
            :,
            :,
            0:
            4
        ],
        dtype=np.float32,
    ).reshape(
        len(
            X_train
        ),
        -1,
    )

    yf = np.asarray(
        y_train[
            :,
            :,
            0:
            2
        ],
        dtype=np.float32,
    ).reshape(
        len(
            y_train
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
        y_std < 1e-8,
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

    stats = emod.compute_stats(
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

    # Exact alpha=0 first. Tiny numerical jitter is used only if the
    # normal equations are singular.
    try:
        B = np.linalg.solve(
            G,
            C,
        )
        jitter = 0.0

    except np.linalg.LinAlgError:
        jitter = 1e-10

        B = np.linalg.solve(
            G
            + jitter
            * np.eye(
                G.shape[
                    0
                ],
                dtype=np.float64,
            ),
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
        "alpha": TRUEWIND_RIDGE_ALPHA,
        "numerical_jitter": jitter,
    }


def predict_truewind_ridge(
    model,
    X,
):
    Xf = np.asarray(
        X[
            :,
            :,
            0:
            4
        ],
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

    return raw.reshape(
        -1,
        5,
        2,
    ).astype(
        np.float32
    )


# =============================================================================
# External-product discovery / loading
# =============================================================================

def resolve_npz_from_path(
    path: Path,
):
    path = Path(
        path
    )

    if path.is_file():
        if (
            path.suffix.lower()
            != ".npz"
        ):
            raise RuntimeError(
                f"External file must be .npz: {path}"
            )

        return path

    if not path.is_dir():
        raise FileNotFoundError(
            path
        )

    candidates = sorted(
        path.rglob(
            "*.npz"
        )
    )

    valid = []

    for p in candidates:
        try:
            with np.load(
                p,
                allow_pickle=False,
            ) as z:
                if (
                    "X" in z.files
                    and "y_raw" in z.files
                ):
                    xs = z[
                        "X"
                    ].shape

                    ys = z[
                        "y_raw"
                    ].shape

                    if (
                        len(
                            xs
                        ) == 3
                        and xs[
                            1:
                        ] == (
                            60,
                            11,
                        )
                        and len(
                            ys
                        ) == 3
                        and ys[
                            1:
                        ] == (
                            5,
                            6,
                        )
                        and xs[
                            0
                        ] == ys[
                            0
                        ]
                    ):
                        valid.append(
                            p
                        )

        except Exception:
            continue

    if len(
        valid
    ) == 1:
        return valid[
            0
        ]

    if len(
        valid
    ) == 0:
        raise RuntimeError(
            f"No valid external NPZ with X/y_raw found under {path}"
        )

    raise RuntimeError(
        "Ambiguous external directory. Multiple valid NPZ files found:\n"
        + "\n".join(
            str(
                p
            )
            for p in valid
        )
    )


def inspect_external_npz(
    path: Path,
):
    with np.load(
        path,
        allow_pickle=False,
    ) as z:
        if (
            "X" not in z.files
            or "y_raw" not in z.files
        ):
            return None

        X = z[
            "X"
        ]

        y = z[
            "y_raw"
        ]

        if (
            X.ndim != 3
            or X.shape[
                1:
            ] != (
                60,
                11,
            )
            or y.ndim != 3
            or y.shape[
                1:
            ] != (
                5,
                6,
            )
            or len(
                X
            ) != len(
                y
            )
        ):
            return None

        return {
            "path": path,
            "n": int(
                len(
                    X
                )
            ),
        }


def rank_candidate(
    dataset_name,
    path,
):
    s = str(
        path
    ).lower()

    score = 0

    if "sd1033" in s:
        score += 10

    if dataset_name == "SD1033-2024-matched":
        if "2024" in s:
            score += 5
        if "match" in s:
            score += 8
        if "1090" in s:
            score += 3

    elif dataset_name == "SD1033-2023-full":
        if "2023" in s:
            score += 8
        if "full" in s:
            score += 2

    elif dataset_name == "SD1033-2022-full":
        if "2022" in s:
            score += 8
        if "full" in s:
            score += 2

    elif dataset_name == "SD1033-2024-full":
        if "2024" in s:
            score += 6
        if "full" in s:
            score += 5
        if "match" in s:
            score -= 8

    bad_words = [
        "prediction",
        "predictions",
        "result",
        "results",
        "checkpoint",
        "model",
        "output",
    ]

    for w in bad_words:
        if w in s:
            score -= 4

    # Prefer shallower data products when tied.
    score -= 0.01 * len(
        path.parts
    )

    return score


def auto_discover_external(
    external_root: Path,
):
    """
    Discovery happens only after all development models are frozen.

    It identifies prior prepared products from:
      * path contains SD1033
      * X/y_raw schema
      * expected number of windows
    """
    if not external_root.exists():
        raise FileNotFoundError(
            external_root
        )

    all_npz = [
        p
        for p in external_root.rglob(
            "*.npz"
        )
        if "sd1033" in str(
            p
        ).lower()
    ]

    by_count = {}

    for p in all_npz:
        try:
            info = inspect_external_npz(
                p
            )

        except Exception:
            info = None

        if info is None:
            continue

        by_count.setdefault(
            info[
                "n"
            ],
            [],
        ).append(
            p
        )

    resolved = {}

    for name, count in EXPECTED_EXTERNAL_COUNTS.items():
        candidates = by_count.get(
            count,
            [],
        )

        if not candidates:
            continue

        candidates = sorted(
            candidates,
            key=lambda p: rank_candidate(
                name,
                p,
            ),
            reverse=True,
        )

        resolved[
            name
        ] = candidates[
            0
        ]

    return (
        resolved,
        by_count,
    )


def load_external_product(
    path: Path,
):
    with np.load(
        path,
        allow_pickle=False,
    ) as z:
        if "X" not in z.files:
            raise RuntimeError(
                f"{path}: missing X"
            )

        if "y_raw" not in z.files:
            raise RuntimeError(
                f"{path}: missing y_raw. "
                "15F refuses to guess whether a generic y key is standardized."
            )

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
            f"{path}: invalid X shape {X.shape}"
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
            f"{path}: invalid y_raw shape {y.shape}"
        )

    if len(
        X
    ) != len(
        y
    ):
        raise RuntimeError(
            f"{path}: X/y length mismatch."
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
            f"{path}: nonfinite external data."
        )

    # Frozen physical truth; no fitting/calibration.
    app_ref = (
        y[
            :,
            :,
            0:
            2
        ]
        - y[
            :,
            :,
            2:
            4
        ]
    ).astype(
        np.float32
    )

    return {
        "X": X,
        "y": y,
        "app_ref": app_ref,
        "path": path,
    }


# =============================================================================
# Motion-checkpoint loading
# =============================================================================

def load_motion_model(
    *,
    torch,
    nn,
    vmod,
    checkpoint_path,
    device,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    Model = vmod.build_joint_motion_model_class(
        torch,
        nn,
    )

    model = Model().to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()

    n_params = count_parameters(
        model
    )

    if n_params != MOTION_PARAMS:
        raise RuntimeError(
            f"Motion parameter mismatch: "
            f"{n_params} != {MOTION_PARAMS}"
        )

    return (
        model,
        checkpoint,
    )


# =============================================================================
# Metric helpers
# =============================================================================

def mean_truewind_metrics(
    df,
):
    return {
        "TrueWind_U_RMSE_mps": float(
            df[
                "U_RMSE_mps"
            ].mean()
        ),
        "TrueWind_V_RMSE_mps": float(
            df[
                "V_RMSE_mps"
            ].mean()
        ),
        "TrueWind_vector_RMSE_mps": float(
            df[
                "vector_RMSE_mps"
            ].mean()
        ),
        "TrueWind_WS_RMSE_mps": float(
            df[
                "WS_RMSE_mps"
            ].mean()
        ),
        "TrueWind_WD_RMSE_deg": float(
            df[
                "WD_RMSE_deg"
            ].mean()
        ),
    }


def mean_joint_metrics(
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


def add_seed_model_columns(
    df,
    dataset_name,
    seed,
    model_name,
):
    """
    Attach dataset/seed/model metadata safely.

    cmod.metrics_per_horizon(...) already returns a ``model`` column
    (and may also return split/seed/dataset-like metadata in future
    versions).  Therefore do NOT blindly ``insert`` duplicate columns.

    We explicitly overwrite existing metadata columns, then reorder them
    to the front.  This keeps the function compatible with both the
    current metrics helper and older outputs.
    """
    out = df.copy()

    # Overwrite rather than insert to avoid:
    # ValueError: cannot insert model, already exists
    out["dataset"] = dataset_name
    out["seed"] = seed
    out["model"] = model_name

    front = [
        "dataset",
        "seed",
        "model",
    ]

    rest = [
        c
        for c in out.columns
        if c not in front
    ]

    return out[
        front + rest
    ]


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
        "--stage15evh-dir",
        type=Path,
        default=DEFAULT_STAGE15EVH_DIR,
    )

    parser.add_argument(
        "--joint-ridge-alpha",
        type=float,
        default=1500.0,
        help=(
            "Frozen joint Ridge alpha paired with the selected 15E-VH "
            "motion checkpoints. Final selected value: 1500."
        ),
    )

    parser.add_argument(
        "--src-dir",
        type=Path,
        default=DEFAULT_SRC_DIR,
    )

    parser.add_argument(
        "--external-root",
        type=Path,
        default=DEFAULT_EXTERNAL_ROOT,
    )

    parser.add_argument(
        "--external-2024-matched",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--external-2023",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--external-2022",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--external-2024-full",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--include-2024-full",
        action="store_true",
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

    # -----------------------------------------------------------------
    # Dependencies.
    # -----------------------------------------------------------------
    bmod = load_module(
        "stage15b_for_15f",
        args.src_dir
        / "15B_train_1min_TrueWind_Residual_Correction_v2.py",
    )

    emod = load_module(
        "stage15e_helper_for_15f",
        args.src_dir
        / "15E_hybrid_NewWind_OldCompactVessel_5seed_validation.py",
    )

    vmod = load_module(
        "stage15evh_for_15f",
        args.src_dir
        / "15E_VH_joint_VesselHDG_GatedResidual_5seed_validation.py",
    )

    cmod = load_module(
        "stage15c_geometry_for_15f",
        args.src_dir
        / "15C_train_1min_VesselHDG_Residual_ApparentWind.py",
    )

    joint_ridge_alpha = float(args.joint_ridge_alpha)
    if joint_ridge_alpha < 0:
        raise ValueError("--joint-ridge-alpha must be >= 0.")

    # fit_old_joint_ridge() reads OLD_RIDGE_ALPHA from the Stage15E helper.
    # Synchronize it with the alpha used to train the frozen 15E-VH
    # motion checkpoints.
    emod.OLD_RIDGE_ALPHA = joint_ridge_alpha

    log(
        f"[FROZEN MOTION ANCHOR] joint Ridge alpha = "
        f"{joint_ridge_alpha:.12g}"
    )

    # Frozen final Stage15B convention.
    bmod.ALPHA_MAX = 1.0

    torch, nn = emod.import_torch()

    emod.configure_torch(
        torch
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    log("=" * 168)

    log(
        "15F FINAL — FROZEN 15E-VH (JOINT RIDGE ALPHA=1500) EXTERNAL GENERALIZATION"
    )

    log("=" * 168)

    log(
        "PHASE A uses SD1090 TRAIN/VALIDATION only."
    )

    log(
        "External data are loaded ONLY after every model component is frozen."
    )

    log(
        "No external fitting, scaling, alpha selection, checkpointing, "
        "seed selection, or calibration."
    )

    log(
        f"device = {device}"
    )

    log(
        f"seeds = {SEEDS}"
    )

    log(
        f"neural parameters = {TOTAL_NEURAL_PARAMS:,}"
    )

    # =================================================================
    # PHASE A — DEVELOPMENT FREEZE
    # =================================================================
    log("")
    log(
        "=" * 168
    )

    log(
        "PHASE A — DEVELOPMENT FREEZE"
    )

    log(
        "=" * 168
    )

    # -----------------------------------------------------------------
    # A1. Load SD1090 train/validation only.
    # -----------------------------------------------------------------
    log("")
    log(
        "[A1/5] Loading SD1090 TRAIN/VALIDATION only."
    )

    train = emod.load_split(
        args.dataset_dir,
        "train",
    )

    val = emod.load_split(
        args.dataset_dir,
        "validation",
    )

    tw_anchor_train, tw_anchor_val = (
        emod.load_truewind_anchor(
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
            "TRAIN TrueWind anchor alignment mismatch."
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
            "VALIDATION TrueWind anchor alignment mismatch."
        )

    log(
        f"  TRAIN      = {len(train['X']):,}"
    )

    log(
        f"  VALIDATION = {len(val['X']):,}"
    )

    # -----------------------------------------------------------------
    # A2. Rebuild deterministic Ridge anchors.
    # -----------------------------------------------------------------
    log("")
    log(
        "[A2/5] Rebuilding frozen Ridge anchors."
    )
    log(
        f"  Joint motion/heading Ridge alpha = "
        f"{joint_ridge_alpha:.12g}"
    )

    old_joint_ridge = emod.fit_old_joint_ridge(
        train[
            "X"
        ],
        train[
            "y"
        ],
    )

    joint_z_val, joint_raw_val = (
        emod.old_ridge_predict(
            old_joint_ridge,
            val[
                "X"
            ],
        )
    )

    truewind_ridge = fit_truewind_ridge(
        emod,
        train[
            "X"
        ],
        train[
            "y"
        ],
    )

    tw_ridge_val_rebuilt = (
        predict_truewind_ridge(
            truewind_ridge,
            val[
                "X"
            ],
        )
    )

    anchor_diff = float(
        np.sqrt(
            np.mean(
                (
                    tw_ridge_val_rebuilt
                    - tw_anchor_val[
                        "ridge"
                    ]
                )
                ** 2
            )
        )
    )

    log(
        f"  Dedicated TrueWind Ridge rebuild-vs-frozen validation RMSE = "
        f"{anchor_diff:.9f}"
    )

    if (
        anchor_diff
        > TRUEWIND_RIDGE_VALIDATION_TOL
    ):
        raise RuntimeError(
            "Dedicated TrueWind Ridge reconstruction does not match "
            "the frozen Stage15A anchor closely enough. "
            f"RMSE={anchor_diff:.9f} > "
            f"{TRUEWIND_RIDGE_VALIDATION_TOL:.9f}. "
            "Do not continue external inference with an incompatible anchor."
        )

    # -----------------------------------------------------------------
    # A3. Prepare deterministic Stage15B features.
    # -----------------------------------------------------------------
    log("")
    log(
        "[A3/5] Preparing frozen Stage15B development features."
    )

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

    # -----------------------------------------------------------------
    # A4. Freeze five Stage15B models.
    # -----------------------------------------------------------------
    log("")
    log(
        "[A4/5] Training/freezing five Stage15B models."
    )

    frozen_wind_models = {}

    freeze_rows = []

    for i, seed in enumerate(
        SEEDS,
        start=1,
    ):
        log("")
        log(
            f"  [Stage15B seed {i}/5] {seed}"
        )

        bundle = bmod.train_model(
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

        alpha = np.asarray(
            bundle[
                "best_alpha"
            ],
            dtype=np.float32,
        )

        raw_val = bmod.predict_raw(
            torch,
            bundle[
                "model"
            ],
            bmod.transform_features(
                feat_val,
                bundle[
                    "scaler"
                ],
            ),
            bundle[
                "scaler"
            ],
            bundle[
                "device"
            ],
        )

        final_val = bmod.apply_alpha(
            tw_anchor_val[
                "ridge"
            ],
            raw_val,
            alpha,
        )

        val_score = bmod.mean_vector_rmse(
            tw_anchor_val[
                "y"
            ],
            final_val,
        )

        # Compare against the exact earlier 15E-H validation artifact,
        # but DO NOT use this difference for any selection.
        old_artifact_path = (
            args.stage15evh_dir.parent
            / "15E_H_HDG_GatedResidual_5seed_v0_1"
            / f"15E_H_seed_{seed}_validation_predictions.npz"
        )

        stored_score = np.nan
        score_diff = np.nan

        if old_artifact_path.exists():
            with np.load(
                old_artifact_path,
                allow_pickle=False,
            ) as z:
                if (
                    "stage15b_truewind"
                    in z.files
                ):
                    stored_w = np.asarray(
                        z[
                            "stage15b_truewind"
                        ],
                        dtype=np.float32,
                    )

                    stored_score = (
                        bmod.mean_vector_rmse(
                            tw_anchor_val[
                                "y"
                            ],
                            stored_w,
                        )
                    )

                    score_diff = (
                        val_score
                        - stored_score
                    )

        log(
            f"    best epoch = {int(bundle['best_epoch'])}"
        )

        log(
            f"    validation mean vector RMSE = {val_score:.6f}"
        )

        if np.isfinite(
            stored_score
        ):
            log(
                f"    prior 15E-H stored RMSE     = {stored_score:.6f}"
            )

            log(
                f"    repeat-minus-prior          = {score_diff:+.6f}"
            )

            if (
                abs(
                    score_diff
                )
                > STAGE15B_REPRO_WARN_THRESHOLD
            ):
                log(
                    "    [WARNING] repeated Stage15B differs more than "
                    f"{STAGE15B_REPRO_WARN_THRESHOLD:.3f} m/s from "
                    "the prior run. External inference remains protocol-frozen, "
                    "but inspect CUDA determinism before paper reporting."
                )

        freeze_rows.append(
            {
                "seed": seed,
                "best_epoch": int(
                    bundle[
                        "best_epoch"
                    ]
                ),
                "validation_truewind_vector_RMSE": float(
                    val_score
                ),
                "prior_15EH_validation_truewind_vector_RMSE": (
                    float(
                        stored_score
                    )
                    if np.isfinite(
                        stored_score
                    )
                    else np.nan
                ),
                "repeat_minus_prior_RMSE": (
                    float(
                        score_diff
                    )
                    if np.isfinite(
                        score_diff
                    )
                    else np.nan
                ),
                "alpha_U_1m": float(
                    alpha[
                        0,
                        0
                    ]
                ),
                "alpha_V_1m": float(
                    alpha[
                        0,
                        1
                    ]
                ),
                "alpha_U_10m": float(
                    alpha[
                        4,
                        0
                    ]
                ),
                "alpha_V_10m": float(
                    alpha[
                        4,
                        1
                    ]
                ),
            }
        )

        checkpoint_path = (
            out
            / f"15F_seed_{seed}_Stage15B_frozen.pt"
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
            "scaler": {
                k: np.asarray(
                    v
                )
                for k, v in (
                    bundle[
                        "scaler"
                    ].items()
                )
            },
            "alpha": alpha,
            "best_epoch": int(
                bundle[
                    "best_epoch"
                ]
            ),
            "validation_truewind_vector_RMSE": float(
                val_score
            ),
            "parameters": STAGE15B_PARAMS,
        }

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        # Keep frozen model in memory.
        for p in bundle[
            "model"
        ].parameters():
            p.requires_grad_(
                False
            )

        bundle[
            "model"
        ].eval()

        frozen_wind_models[
            seed
        ] = {
            "model": bundle[
                "model"
            ],
            "scaler": bundle[
                "scaler"
            ],
            "alpha": alpha,
            "device": bundle[
                "device"
            ],
            "best_epoch": int(
                bundle[
                    "best_epoch"
                ]
            ),
        }

    freeze_df = pd.DataFrame(
        freeze_rows
    )

    freeze_df.to_csv(
        out
        / "15F_STAGE15B_development_freeze.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # A5. Load/freeze motion checkpoints and reproduce validation.
    # -----------------------------------------------------------------
    log("")
    log(
        "[A5/5] Loading already frozen 15E-VH joint-motion checkpoints."
    )

    frozen_motion_models = {}

    motion_repro_rows = []

    for seed in SEEDS:
        ckpt_path = (
            args.stage15evh_dir
            / f"15E_VH_seed_{seed}_joint_motion_model.pt"
        )

        model, ckpt = load_motion_model(
            torch=torch,
            nn=nn,
            vmod=vmod,
            checkpoint_path=ckpt_path,
            device=device,
        )

        if "ridge_alpha" not in ckpt:
            raise RuntimeError(
                f"{ckpt_path}: checkpoint has no ridge_alpha metadata. "
                "Use the alpha-search checkpoints generated by the "
                "parameterized 15E-VH trainer."
            )

        ckpt_alpha = float(ckpt["ridge_alpha"])
        if not np.isclose(
            ckpt_alpha,
            joint_ridge_alpha,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"{ckpt_path}: checkpoint ridge_alpha={ckpt_alpha:g} "
                f"does not match requested --joint-ridge-alpha="
                f"{joint_ridge_alpha:g}."
            )

        pred = vmod.predict_joint_motion(
            torch=torch,
            model=model,
            X=val[
                "X"
            ],
            ridge_z=joint_z_val,
            old_ridge_model=old_joint_ridge,
            device=device,
        )[
            "pred_raw"
        ]

        saved_pred_path = (
            args.stage15evh_dir
            / f"15E_VH_seed_{seed}_validation_predictions.npz"
        )

        pred_rmse = np.nan

        if saved_pred_path.exists():
            with np.load(
                saved_pred_path,
                allow_pickle=False,
            ) as z:
                if (
                    "joint_vessel_hdg"
                    in z.files
                ):
                    saved = np.asarray(
                        z[
                            "joint_vessel_hdg"
                        ],
                        dtype=np.float32,
                    )

                    pred_rmse = float(
                        np.sqrt(
                            np.mean(
                                (
                                    pred
                                    - saved
                                )
                                ** 2
                            )
                        )
                    )

        log(
            f"  seed {seed}: motion checkpoint loaded "
            f"(ridge alpha={ckpt_alpha:g}); "
            f"validation reproduction RMSE="
            + (
                f"{pred_rmse:.9f}"
                if np.isfinite(
                    pred_rmse
                )
                else "not available"
            )
        )

        motion_repro_rows.append(
            {
                "seed": seed,
                "best_epoch": int(
                    ckpt[
                        "best_epoch"
                    ]
                ),
                "validation_prediction_reproduction_RMSE": (
                    float(
                        pred_rmse
                    )
                    if np.isfinite(
                        pred_rmse
                    )
                    else np.nan
                ),
            }
        )

        for p in model.parameters():
            p.requires_grad_(
                False
            )

        model.eval()

        frozen_motion_models[
            seed
        ] = {
            "model": model,
            "checkpoint": ckpt,
        }

    pd.DataFrame(
        motion_repro_rows
    ).to_csv(
        out
        / "15F_motion_checkpoint_reproduction.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log(
        "[FREEZE COMPLETE]"
    )

    log(
        "All five Stage15B models and all five 15E-VH motion models "
        "are now frozen."
    )

    log(
        "External products may now be opened."
    )

    # =================================================================
    # PHASE B — EXTERNAL INFERENCE
    # =================================================================
    log("")
    log(
        "=" * 168
    )

    log(
        "PHASE B — EXTERNAL INFERENCE"
    )

    log(
        "=" * 168
    )

    # -----------------------------------------------------------------
    # B1. Resolve external products.
    # -----------------------------------------------------------------
    log("")
    log(
        "[B1/4] Resolving external SD1033 products."
    )

    explicit = {
        "SD1033-2024-matched": args.external_2024_matched,
        "SD1033-2023-full": args.external_2023,
        "SD1033-2022-full": args.external_2022,
        "SD1033-2024-full": args.external_2024_full,
    }

    resolved = {}

    for name, path in explicit.items():
        if path is not None:
            resolved[
                name
            ] = resolve_npz_from_path(
                path
            )

    missing_primary = [
        name
        for name in PRIMARY_DATASETS
        if name not in resolved
    ]

    need_supp = (
        args.include_2024_full
        and SUPPLEMENTARY_DATASET
        not in resolved
    )

    if (
        missing_primary
        or need_supp
    ):
        discovered, by_count = (
            auto_discover_external(
                args.external_root
            )
        )

        for name, p in discovered.items():
            if name not in resolved:
                resolved[
                    name
                ] = p

    missing_primary = [
        name
        for name in PRIMARY_DATASETS
        if name not in resolved
    ]

    if missing_primary:
        raise RuntimeError(
            "Could not auto-discover all primary external products: "
            + ", ".join(
                missing_primary
            )
            + "\nProvide explicit paths using "
            "--external-2024-matched, --external-2023, --external-2022."
        )

    datasets_to_run = list(
        PRIMARY_DATASETS
    )

    if args.include_2024_full:
        if (
            SUPPLEMENTARY_DATASET
            not in resolved
        ):
            raise RuntimeError(
                "Requested --include-2024-full but no 2024-full product "
                "was found. Provide --external-2024-full."
            )

        datasets_to_run.append(
            SUPPLEMENTARY_DATASET
        )

    for name in datasets_to_run:
        p = resolved[
            name
        ]

        info = inspect_external_npz(
            p
        )

        if info is None:
            raise RuntimeError(
                f"Invalid external product: {p}"
            )

        expected_n = EXPECTED_EXTERNAL_COUNTS[
            name
        ]

        log(
            f"  {name:24s} -> {p}"
        )

        log(
            f"    windows={info['n']:,} "
            f"(expected {expected_n:,})"
        )

        if (
            info[
                "n"
            ]
            != expected_n
        ):
            raise RuntimeError(
                f"{name}: sample count {info['n']:,} does not match "
                f"the frozen prior product count {expected_n:,}. "
                "Refuse to silently compare incompatible external products."
            )

    # -----------------------------------------------------------------
    # B2/B3. Run external inference.
    # -----------------------------------------------------------------
    log("")
    log(
        "[B2/4] Running frozen external inference."
    )

    all_per_seed_rows = []
    all_horizon_frames = []
    all_branch_rows = []
    dataset_summary_rows = []
    old_compare_rows = []

    for dataset_name in datasets_to_run:
        log("")
        log(
            "-" * 168
        )

        log(
            f"[EXTERNAL] {dataset_name}"
        )

        log(
            "-" * 168
        )

        ext = load_external_product(
            resolved[
                dataset_name
            ]
        )

        X_ext = ext[
            "X"
        ]

        y_ext = ext[
            "y"
        ]

        app_ref_ext = ext[
            "app_ref"
        ]

        y_motion_ext = y_ext[
            :,
            :,
            2:
            6
        ]

        log(
            f"  N = {len(X_ext):,}"
        )

        log(
            f"  X mean(abs channel mean) = "
            f"{np.mean(np.abs(np.mean(X_ext.astype(np.float64),axis=(0,1)))):.4f}"
        )

        log(
            "  No external statistics are fitted."
        )

        # Deterministic external anchors.
        joint_z_ext, joint_raw_ext = (
            emod.old_ridge_predict(
                old_joint_ridge,
                X_ext,
            )
        )

        ridge_motion_ext = joint_raw_ext[
            :,
            :,
            2:
            6
        ].copy()

        ridge_motion_ext[
            :,
            :,
            2:
            4
        ] = normalize_heading_pair(
            ridge_motion_ext[
                :,
                :,
                2:
                4
            ]
        )

        dedicated_wind_ridge_ext = (
            predict_truewind_ridge(
                truewind_ridge,
                X_ext,
            )
        )

        # Old joint-Ridge wind diagnostic.
        old_joint_wind_ext = joint_raw_ext[
            :,
            :,
            0:
            2
        ]

        # Deterministic baseline: Dual Ridge.
        dual_ridge_metrics = (
            cmod.metrics_per_horizon(
                y_motion_ext,
                ridge_motion_ext,
                dedicated_wind_ridge_ext,
                app_ref_ext,
                "external",
                "DualRidge",
            )
        )

        all_horizon_frames.append(
            add_seed_model_columns(
                dual_ridge_metrics,
                dataset_name,
                -1,
                "DualRidge",
            )
        )

        # Old-joint-Ridge compatibility diagnostic.
        old_joint_ridge_metrics = (
            cmod.metrics_per_horizon(
                y_motion_ext,
                ridge_motion_ext,
                old_joint_wind_ext,
                app_ref_ext,
                "external",
                "OldJointRidge",
            )
        )

        all_horizon_frames.append(
            add_seed_model_columns(
                old_joint_ridge_metrics,
                dataset_name,
                -1,
                "OldJointRidge",
            )
        )

        dual_mean = mean_joint_metrics(
            dual_ridge_metrics
        )

        old_joint_mean = mean_joint_metrics(
            old_joint_ridge_metrics
        )

        seed_full_aw = []
        seed_full_vessel = []
        seed_full_hdg = []
        seed_full_aws = []
        seed_full_awa = []
        seed_full_tw = []

        for i, seed in enumerate(
            SEEDS,
            start=1,
        ):
            log(
                f"  seed {i}/5 = {seed}"
            )

            # ---------------------------------------------------------
            # Frozen Stage15B external inference.
            # ---------------------------------------------------------
            wb = frozen_wind_models[
                seed
            ]

            ext_feat = bmod.build_features(
                X_ext
            )

            ext_feat_scaled = bmod.transform_features(
                ext_feat,
                wb[
                    "scaler"
                ],
            )

            raw_w_ext = bmod.predict_raw(
                torch,
                wb[
                    "model"
                ],
                ext_feat_scaled,
                wb[
                    "scaler"
                ],
                wb[
                    "device"
                ],
            )

            wind_ext = bmod.apply_alpha(
                dedicated_wind_ridge_ext,
                raw_w_ext,
                wb[
                    "alpha"
                ],
            )

            tw_metrics = (
                bmod.metrics_per_horizon(
                    y_ext[
                        :,
                        :,
                        0:
                        2
                    ],
                    wind_ext,
                    "15E-VH-TrueWind",
                    "external",
                )
            )

            tw_mean = mean_truewind_metrics(
                tw_metrics
            )

            # ---------------------------------------------------------
            # Frozen 15E-VH joint motion external inference.
            # ---------------------------------------------------------
            mb = frozen_motion_models[
                seed
            ]

            motion_out = vmod.predict_joint_motion(
                torch=torch,
                model=mb[
                    "model"
                ],
                X=X_ext,
                ridge_z=joint_z_ext,
                old_ridge_model=old_joint_ridge,
                device=device,
            )

            motion_ext = motion_out[
                "pred_raw"
            ]

            # ---------------------------------------------------------
            # Branch decomposition.
            # ---------------------------------------------------------

            # A. wind enhanced only
            wind_only_metrics = (
                cmod.metrics_per_horizon(
                    y_motion_ext,
                    ridge_motion_ext,
                    wind_ext,
                    app_ref_ext,
                    "external",
                    "WindEnhanced+RidgeMotion",
                )
            )

            # B. motion enhanced only
            motion_only_metrics = (
                cmod.metrics_per_horizon(
                    y_motion_ext,
                    motion_ext,
                    dedicated_wind_ridge_ext,
                    app_ref_ext,
                    "external",
                    "RidgeWind+MotionEnhanced",
                )
            )

            # C. full proposed
            full_metrics = (
                cmod.metrics_per_horizon(
                    y_motion_ext,
                    motion_ext,
                    wind_ext,
                    app_ref_ext,
                    "external",
                    "15E-VH",
                )
            )

            all_horizon_frames.extend(
                [
                    add_seed_model_columns(
                        wind_only_metrics,
                        dataset_name,
                        seed,
                        "WindEnhanced+RidgeMotion",
                    ),
                    add_seed_model_columns(
                        motion_only_metrics,
                        dataset_name,
                        seed,
                        "RidgeWind+MotionEnhanced",
                    ),
                    add_seed_model_columns(
                        full_metrics,
                        dataset_name,
                        seed,
                        "15E-VH",
                    ),
                ]
            )

            wind_only_mean = mean_joint_metrics(
                wind_only_metrics
            )

            motion_only_mean = mean_joint_metrics(
                motion_only_metrics
            )

            full_mean = mean_joint_metrics(
                full_metrics
            )

            all_per_seed_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    **tw_mean,
                    **full_mean,
                    "DualRidge_AppVector_RMSE_mps": dual_mean[
                        "AppVector_RMSE_mps"
                    ],
                    "WindOnly_AppVector_RMSE_mps": wind_only_mean[
                        "AppVector_RMSE_mps"
                    ],
                    "MotionOnly_AppVector_RMSE_mps": motion_only_mean[
                        "AppVector_RMSE_mps"
                    ],
                    "Full_AppVector_skill_vs_DualRidge_pct": (
                        (
                            dual_mean[
                                "AppVector_RMSE_mps"
                            ]
                            - full_mean[
                                "AppVector_RMSE_mps"
                            ]
                        )
                        / dual_mean[
                            "AppVector_RMSE_mps"
                        ]
                        * 100.0
                    ),
                }
            )

            all_branch_rows.extend(
                [
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "model": "DualRidge",
                        "AppVector_RMSE_mps": dual_mean[
                            "AppVector_RMSE_mps"
                        ],
                        "AWS_RMSE_mps": dual_mean[
                            "AWS_RMSE_mps"
                        ],
                        "AWA_MAE_deg": dual_mean[
                            "AWA_MAE_deg"
                        ],
                        "Vessel_vector_RMSE_mps": dual_mean[
                            "Vessel_vector_RMSE_mps"
                        ],
                        "HDG_MAE_deg": dual_mean[
                            "HDG_MAE_deg"
                        ],
                    },
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "model": "WindEnhanced+RidgeMotion",
                        "AppVector_RMSE_mps": wind_only_mean[
                            "AppVector_RMSE_mps"
                        ],
                        "AWS_RMSE_mps": wind_only_mean[
                            "AWS_RMSE_mps"
                        ],
                        "AWA_MAE_deg": wind_only_mean[
                            "AWA_MAE_deg"
                        ],
                        "Vessel_vector_RMSE_mps": wind_only_mean[
                            "Vessel_vector_RMSE_mps"
                        ],
                        "HDG_MAE_deg": wind_only_mean[
                            "HDG_MAE_deg"
                        ],
                    },
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "model": "RidgeWind+MotionEnhanced",
                        "AppVector_RMSE_mps": motion_only_mean[
                            "AppVector_RMSE_mps"
                        ],
                        "AWS_RMSE_mps": motion_only_mean[
                            "AWS_RMSE_mps"
                        ],
                        "AWA_MAE_deg": motion_only_mean[
                            "AWA_MAE_deg"
                        ],
                        "Vessel_vector_RMSE_mps": motion_only_mean[
                            "Vessel_vector_RMSE_mps"
                        ],
                        "HDG_MAE_deg": motion_only_mean[
                            "HDG_MAE_deg"
                        ],
                    },
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "model": "15E-VH",
                        "AppVector_RMSE_mps": full_mean[
                            "AppVector_RMSE_mps"
                        ],
                        "AWS_RMSE_mps": full_mean[
                            "AWS_RMSE_mps"
                        ],
                        "AWA_MAE_deg": full_mean[
                            "AWA_MAE_deg"
                        ],
                        "Vessel_vector_RMSE_mps": full_mean[
                            "Vessel_vector_RMSE_mps"
                        ],
                        "HDG_MAE_deg": full_mean[
                            "HDG_MAE_deg"
                        ],
                    },
                ]
            )

            seed_full_aw.append(
                full_mean[
                    "AppVector_RMSE_mps"
                ]
            )

            seed_full_vessel.append(
                full_mean[
                    "Vessel_vector_RMSE_mps"
                ]
            )

            seed_full_hdg.append(
                full_mean[
                    "HDG_MAE_deg"
                ]
            )

            seed_full_aws.append(
                full_mean[
                    "AWS_RMSE_mps"
                ]
            )

            seed_full_awa.append(
                full_mean[
                    "AWA_MAE_deg"
                ]
            )

            seed_full_tw.append(
                tw_mean[
                    "TrueWind_vector_RMSE_mps"
                ]
            )

            log(
                f"    AW={full_mean['AppVector_RMSE_mps']:.6f} | "
                f"Vessel={full_mean['Vessel_vector_RMSE_mps']:.6f} | "
                f"HDG={full_mean['HDG_MAE_deg']:.3f} deg | "
                f"AWS={full_mean['AWS_RMSE_mps']:.6f} | "
                f"AWA={full_mean['AWA_MAE_deg']:.3f} deg"
            )

        # -------------------------------------------------------------
        # Dataset aggregate.
        # -------------------------------------------------------------
        aw_mean, aw_std = mean_and_std(
            seed_full_aw
        )

        vessel_mean, vessel_std = (
            mean_and_std(
                seed_full_vessel
            )
        )

        hdg_mean, hdg_std = (
            mean_and_std(
                seed_full_hdg
            )
        )

        aws_mean, aws_std = (
            mean_and_std(
                seed_full_aws
            )
        )

        awa_mean, awa_std = (
            mean_and_std(
                seed_full_awa
            )
        )

        tw_mean, tw_std = (
            mean_and_std(
                seed_full_tw
            )
        )

        dataset_summary_rows.append(
            {
                "dataset": dataset_name,
                "N": len(
                    X_ext
                ),
                "TrueWind_vector_RMSE_mean": tw_mean,
                "TrueWind_vector_RMSE_std": tw_std,
                "Vessel_vector_RMSE_mean": vessel_mean,
                "Vessel_vector_RMSE_std": vessel_std,
                "HDG_MAE_mean_deg": hdg_mean,
                "HDG_MAE_std_deg": hdg_std,
                "AppVector_RMSE_mean": aw_mean,
                "AppVector_RMSE_std": aw_std,
                "AWS_RMSE_mean": aws_mean,
                "AWS_RMSE_std": aws_std,
                "AWA_MAE_mean_deg": awa_mean,
                "AWA_MAE_std_deg": awa_std,
                "DualRidge_AppVector_RMSE": dual_mean[
                    "AppVector_RMSE_mps"
                ],
                "OldJointRidge_AppVector_RMSE": old_joint_mean[
                    "AppVector_RMSE_mps"
                ],
                "Skill_vs_DualRidge_pct": (
                    (
                        dual_mean[
                            "AppVector_RMSE_mps"
                        ]
                        - aw_mean
                    )
                    / dual_mean[
                        "AppVector_RMSE_mps"
                    ]
                    * 100.0
                ),
            }
        )

        if dataset_name in OLD_PC_AW:
            old_pc = OLD_PC_AW[
                dataset_name
            ]

            old_compare_rows.append(
                {
                    "dataset": dataset_name,
                    "Old_PhysicsCompact_AW_RMSE": old_pc,
                    "New_15EVH_AW_RMSE_mean": aw_mean,
                    "New_15EVH_AW_RMSE_std": aw_std,
                    "New_improvement_vs_old_PC_pct": (
                        (
                            old_pc
                            - aw_mean
                        )
                        / old_pc
                        * 100.0
                    ),
                    "Old_TimeMixer_AW_RMSE": OLD_TIMEMIXER_AW[
                        dataset_name
                    ],
                    "New_improvement_vs_old_TimeMixer_pct": (
                        (
                            OLD_TIMEMIXER_AW[
                                dataset_name
                            ]
                            - aw_mean
                        )
                        / OLD_TIMEMIXER_AW[
                            dataset_name
                        ]
                        * 100.0
                    ),
                    "Old_DLinear_AW_RMSE": OLD_DLINEAR_AW[
                        dataset_name
                    ],
                    "New_improvement_vs_old_DLinear_pct": (
                        (
                            OLD_DLINEAR_AW[
                                dataset_name
                            ]
                            - aw_mean
                        )
                        / OLD_DLINEAR_AW[
                            dataset_name
                        ]
                        * 100.0
                    ),
                }
            )

        del ext
        del X_ext
        del y_ext

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # B3. Save.
    # -----------------------------------------------------------------
    log("")
    log(
        "[B3/4] Saving external results."
    )

    per_seed_df = pd.DataFrame(
        all_per_seed_rows
    )

    horizon_df = pd.concat(
        all_horizon_frames,
        ignore_index=True,
    )

    branch_df = pd.DataFrame(
        all_branch_rows
    )

    summary_df = pd.DataFrame(
        dataset_summary_rows
    )

    old_compare_df = pd.DataFrame(
        old_compare_rows
    )

    per_seed_df.to_csv(
        out
        / "15F_external_per_seed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_df.to_csv(
        out
        / "15F_external_per_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    branch_df.to_csv(
        out
        / "15F_external_branch_decomposition.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        out
        / "15F_external_dataset_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    old_compare_df.to_csv(
        out
        / "15F_external_vs_old_manuscript.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # B4. Final report.
    # -----------------------------------------------------------------
    log("")
    log(
        "[B4/4] Final external report."
    )

    log("")
    log("=" * 168)

    log(
        "15F FINAL — FROZEN 15E-VH EXTERNAL GENERALIZATION"
    )

    log("=" * 168)

    for _, row in summary_df.iterrows():
        log("")
        log(
            f"{row['dataset']}  (N={int(row['N']):,})"
        )

        log(
            f"  TrueWind vector RMSE = "
            f"{row['TrueWind_vector_RMSE_mean']:.6f} ± "
            f"{row['TrueWind_vector_RMSE_std']:.6f}"
        )

        log(
            f"  Vessel vector RMSE   = "
            f"{row['Vessel_vector_RMSE_mean']:.6f} ± "
            f"{row['Vessel_vector_RMSE_std']:.6f}"
        )

        log(
            f"  HDG MAE              = "
            f"{row['HDG_MAE_mean_deg']:.6f} ± "
            f"{row['HDG_MAE_std_deg']:.6f} deg"
        )

        log(
            f"  AppVector RMSE       = "
            f"{row['AppVector_RMSE_mean']:.6f} ± "
            f"{row['AppVector_RMSE_std']:.6f}"
        )

        log(
            f"  AWS RMSE             = "
            f"{row['AWS_RMSE_mean']:.6f} ± "
            f"{row['AWS_RMSE_std']:.6f}"
        )

        log(
            f"  AWA MAE              = "
            f"{row['AWA_MAE_mean_deg']:.6f} ± "
            f"{row['AWA_MAE_std_deg']:.6f} deg"
        )

        log(
            f"  DualRidge AW RMSE    = "
            f"{row['DualRidge_AppVector_RMSE']:.6f}"
        )

        log(
            f"  Skill vs DualRidge   = "
            f"{row['Skill_vs_DualRidge_pct']:+.3f}%"
        )

    if len(
        old_compare_df
    ):
        log("")
        log(
            "[NEW 15E-VH vs OLD MANUSCRIPT EXTERNAL RESULTS]"
        )

        log(
            old_compare_df.to_string(
                index=False
            )
        )

    report = {
        "experiment": "15F",
        "model": "frozen 15E-VH",
        "neural_parameters": TOTAL_NEURAL_PARAMS,
        "seeds": SEEDS,
        "external_protocol": {
            "external_finetuning": False,
            "external_scaler_fit": False,
            "external_alpha_selection": False,
            "external_checkpoint_selection": False,
            "external_seed_selection": False,
            "external_calibration": False,
            "external_products_loaded_only_after_development_freeze": True,
        },
        "external_status": (
            "historical external-transfer sets already seen in prior "
            "manuscript development; not pristine final evaluation"
        ),
        "resolved_external_paths": {
            name: str(
                resolved[
                    name
                ]
            )
            for name in datasets_to_run
        },
        "dataset_summary": summary_df.to_dict(
            orient="records"
        ),
        "old_manuscript_comparison": old_compare_df.to_dict(
            orient="records"
        ),
    }

    save_json(
        out
        / "15F_REPORT.json",
        report,
    )

    with (
        out
        / "15F_REPORT.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "15F FINAL — FROZEN 15E-VH (JOINT RIDGE ALPHA=1500) EXTERNAL GENERALIZATION\n"
        )

        f.write(
            "=" * 150
            + "\n\n"
        )

        f.write(
            "No external fitting, rescaling, calibration, "
            "checkpoint selection, or seed selection.\n"
        )

        f.write(
            "These SD1033 products are historical external-transfer "
            "datasets already seen in prior manuscript development; "
            "they are not a new pristine final test.\n\n"
        )

        f.write(
            "DATASET SUMMARY\n"
        )

        f.write(
            "-" * 150
            + "\n"
        )

        f.write(
            summary_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nOLD MANUSCRIPT COMPARISON\n"
        )

        f.write(
            "-" * 150
            + "\n"
        )

        f.write(
            old_compare_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nBRANCH DECOMPOSITION\n"
        )

        f.write(
            "-" * 150
            + "\n"
        )

        branch_agg = (
            branch_df
            .groupby(
                [
                    "dataset",
                    "model",
                ],
                as_index=False,
            )
            .agg(
                AppVector_RMSE_mean=(
                    "AppVector_RMSE_mps",
                    "mean",
                ),
                AppVector_RMSE_std=(
                    "AppVector_RMSE_mps",
                    "std",
                ),
                AWS_RMSE_mean=(
                    "AWS_RMSE_mps",
                    "mean",
                ),
                AWA_MAE_mean=(
                    "AWA_MAE_deg",
                    "mean",
                ),
                Vessel_vector_RMSE_mean=(
                    "Vessel_vector_RMSE_mps",
                    "mean",
                ),
                HDG_MAE_mean=(
                    "HDG_MAE_deg",
                    "mean",
                ),
            )
        )

        f.write(
            branch_agg.to_string(
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
