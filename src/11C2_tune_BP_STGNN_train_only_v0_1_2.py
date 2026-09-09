# -*- coding: utf-8 -*-
"""
11C2_tune_BP_STGNN_train_only.py

Stage 11C2
===========
Predeclared rolling-origin hyperparameter screening for the full-paper-informed,
task-adapted BP-STGNN-F baseline.

SCIENTIFIC ROLE
---------------
This is the FIRST scientific training stage for BP-STGNN in the present study.

Only the frozen Stage-11C0 Track-F SD1090 TRAIN split is accessed:

    X_raw: [N, 60, 10]
    y_wind_raw: [N, 5, 2]
    horizons: [1, 2, 3, 5, 10] min

The script DOES NOT read:
    - SD1090 outer validation
    - SD1090 outer test
    - any SD1033 external dataset

The frozen Physics-Compact-Vessel-Residual v1.0 model is not loaded, modified,
or retuned.

MODEL
-----
The implementation preserves the full-paper-confirmed BP-STGNN mechanisms:

    10 physical graph nodes
      -> Bayesian mask-centered adjacency
      -> graph convolution
      -> GRU
      -> predictive mean + log variance

Paper Eq.(2) mask:
    (S union K) -> T
    E -> S
    E -> E
    self-loops

Bayesian adjacency:
    A_ij ~ N(mu_adj_ij, sigma_adj_ij^2)
    prior implied by paper KL: N(M_ij, 1)
    A_ij = mu_adj_ij + sigma_adj_ij * epsilon

Two-stage loss:
    first 30 zero-based epochs (epoch index 0..29):
        MSE + lambda_KL * KL
        variance head frozen
    from zero-based epoch 30 onward:
        MSE + lambda_NLL * NLL + lambda_KL * KL
        variance head active

Variance head uses detached shared features, as audited in Stage 11C1.

FIXED IMPLEMENTATION ADAPTATIONS
--------------------------------
The paper does not numerically specify several code-level details. Stage 11C2
predeclares and freezes the following choices BEFORE outer-validation or
external evaluation:

    node_embed_dim      = 32
    gcn_layers          = 2
    gru_layers          = 1
    mlp_hidden_dim      = 64
    dropout             = 0.10
    initial_adj_sigma   = 0.10
    sigma parameter     = softplus(rho) + 1e-6
    adjacency norm      = signed A with degree from |A|
    logvar clamp        = [-12, 8]
    optimizer           = Adam
    base learning rate  = 1e-3
    batch size          = 256
    grad clip norm      = 1.0
    max epochs          = 80
    phase-2 starts      = zero-based epoch 30
    early-stop patience = 10 eligible phase-2 epochs
    ReduceLROnPlateau   = factor 0.5, patience 4, min_lr 1e-5
    checkpoint metric   = raw-unit mean wind-vector RMSE across 5 horizons
    training batch order = chronological (shuffle=False)

SEARCH SPACE
------------
Ten PREDECLARED configurations are screened. They vary only dimensions / loss
weights that remain numerically under-specified in the paper.

The architecture-capacity configs bracket the paper's reported 242,580
parameters. In particular, the 64/224 and 96/224 variants are expected to lie
on either side of that count.

    B01: GCN 48, GRU 192, lambda_KL 1e-4, lambda_NLL 0.10
    B02: GCN 64, GRU 192, lambda_KL 1e-4, lambda_NLL 0.10  [11C1 anchor size]
    B03: GCN 96, GRU 192, lambda_KL 1e-4, lambda_NLL 0.10
    B04: GCN 64, GRU 224, lambda_KL 1e-4, lambda_NLL 0.10  [near paper params]
    B05: GCN 96, GRU 224, lambda_KL 1e-4, lambda_NLL 0.10  [near paper params]
    B06: GCN 64, GRU 256, lambda_KL 1e-4, lambda_NLL 0.10
    B07: GCN 64, GRU 224, lambda_KL 1e-5, lambda_NLL 0.10
    B08: GCN 64, GRU 224, lambda_KL 1e-3, lambda_NLL 0.10
    B09: GCN 64, GRU 224, lambda_KL 1e-4, lambda_NLL 0.05
    B10: GCN 64, GRU 224, lambda_KL 1e-4, lambda_NLL 0.25

No search-space edits are permitted after the script begins producing
outer-validation/test/external results. This script freezes a JSON copy and a
SHA-256 digest of the search space.

ROLLING-ORIGIN CV
-----------------
Let N be the accepted Stage-11C0 Track-F SD1090 training samples.

    val_size = floor(N / 6)
    initial_origin = N - 3 * val_size
    purge = max horizon = 10 accepted samples

Fold k:
    effective training = all samples before origin_k - purge
    validation         = next val_size samples
    origin_{k+1}       = origin_k + val_size

Strict timestamp audit:
    max(training target timestamp) < min(validation context-end timestamp)

This mirrors the chronological firewall previously used in the project.

SCALING
-------
Each fold fits its OWN feature and target standardization statistics using only
that fold's effective training samples.

No scaler sees fold validation, outer validation, test, or external data.

SELECTION
---------
For every epoch, deterministic posterior-mean adjacency inference is used for
checkpointing. This is stable and consistent with the paper's reported
deterministic inference mode.

Primary fold score:
    mean over h=[1,2,3,5,10] of
        sqrt(mean((U_hat-U)^2 + (V_hat-V)^2))

Candidate score:
    mean of the three fold scores.

Tie-breakers, in order:
    1) lower mean CV wind-vector RMSE
    2) lower standard deviation across folds
    3) fewer trainable parameters
    4) lexicographically smaller config ID

The selected configuration is frozen for Stage 11C3.

SEEDS
-----
Screening uses one predefined seed per fold, shared across all configurations:

    Fold 1 -> 500043
    Fold 2 -> 501052
    Fold 3 -> 502061

This is screening only. Stage 11C3 will refit the selected configuration using
the full five-seed project schedule:
    500043, 501052, 502061, 503070, 504079

OUTPUT
------
11C2_BP_STGNN_train_only_rolling_cv_v0_1/
    frozen_search_space.json
    rolling_folds.csv
    dataset_audit.json
    fold_results.csv
    candidate_summary.csv
    selected_config.json
    training_manifest.json
    training_report.txt
    histories/
        Bxx_fold1.csv ...
    checkpoints/
        Bxx_fold1.pt ...

RESUME
------
Completed finite fold rows in fold_results.csv are automatically reused when
their config ID and fold ID match. This allows interrupted long searches to be
continued without repeating finished runs.

Recommended:
    conda activate WindPredict
    python "D:\\project\\WindPredict_SaildroneData\\src\\11C2_tune_BP_STGNN_train_only.py"

For a NON-SCIENTIFIC code smoke test only:
    --debug-fast
This reduces the run to B04/fold1 and shortens the phase schedule. A debug run
will never write selected_config.json as a scientific selection.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.2-BP-STGNN-F-train-only-rolling-cv-debug-isolated"

DEFAULT_PROJECT_ROOT = Path(
    r"D:\project\WindPredict_SaildroneData"
)

DEFAULT_FROZEN_DATASET_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "SD1090_TPOS2024_JointForecasting_v0_1"
)

DEFAULT_11C0_DIR = (
    DEFAULT_FROZEN_DATASET_DIR
    / "11C0_BP_STGNN_comparison_datasets_v0_1"
)

DEFAULT_OUTPUT_DIR = (
    DEFAULT_FROZEN_DATASET_DIR
    / "11C2_BP_STGNN_train_only_rolling_cv_v0_1"
)

TRACK_F_TRAIN_FILE = (
    DEFAULT_11C0_DIR
    / "track_F_common_task"
    / "SD1090_train.npz"
)

NODE_NAMES = [
    "U",
    "V",
    "SOG",
    "COG",
    "T",
    "H",
    "P",
    "dT",
    "dP",
    "dH",
]

NODE_GROUPS = {
    "target_T": [
        "U",
        "V",
    ],
    "source_S": [
        "dP",
        "dT",
        "dH",
    ],
    "kinematic_K": [
        "SOG",
        "COG",
    ],
    "environment_E": [
        "P",
        "T",
        "H",
    ],
}

HORIZONS_MIN = [
    1,
    2,
    3,
    5,
    10,
]

LOOKBACK = 60
TARGET_COUNT = 2
SCREENING_FOLD_SEEDS = [
    500043,
    501052,
    502061,
]
FINAL_SEED_SCHEDULE = [
    500043,
    501052,
    502061,
    503070,
    504079,
]


@dataclass(frozen=True)
class FixedTrainConfig:
    node_embed_dim: int = 32
    gcn_layers: int = 2
    gru_layers: int = 1
    mlp_hidden_dim: int = 64
    dropout: float = 0.10
    initial_adj_sigma: float = 0.10
    learning_rate: float = 1e-3
    batch_size: int = 256
    grad_clip_norm: float = 1.0
    max_epochs: int = 80
    phase2_start_epoch_zero_based: int = 30
    early_stop_patience: int = 10
    scheduler_patience: int = 4
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-5
    logvar_min: float = -12.0
    logvar_max: float = 8.0
    use_amp: bool = True
    purge_samples: int = 10
    fold_count: int = 3
    train_batch_shuffle: bool = False


@dataclass(frozen=True)
class SearchConfig:
    config_id: str
    gcn_hidden_dim: int
    gru_hidden_dim: int
    lambda_kl: float
    lambda_nll: float


FIXED = FixedTrainConfig()

SEARCH_CONFIGS = [
    SearchConfig(
        "B01",
        48,
        192,
        1e-4,
        0.10,
    ),
    SearchConfig(
        "B02",
        64,
        192,
        1e-4,
        0.10,
    ),
    SearchConfig(
        "B03",
        96,
        192,
        1e-4,
        0.10,
    ),
    SearchConfig(
        "B04",
        64,
        224,
        1e-4,
        0.10,
    ),
    SearchConfig(
        "B05",
        96,
        224,
        1e-4,
        0.10,
    ),
    SearchConfig(
        "B06",
        64,
        256,
        1e-4,
        0.10,
    ),
    SearchConfig(
        "B07",
        64,
        224,
        1e-5,
        0.10,
    ),
    SearchConfig(
        "B08",
        64,
        224,
        1e-3,
        0.10,
    ),
    SearchConfig(
        "B09",
        64,
        224,
        1e-4,
        0.05,
    ),
    SearchConfig(
        "B10",
        64,
        224,
        1e-4,
        0.25,
    ),
]


def log(
    message="",
):
    print(
        message,
        flush=True,
    )


def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required. Activate the WindPredict environment."
        ) from exc

    return (
        torch,
        nn,
        F,
    )


def seed_everything(
    torch,
    seed: int,
):
    random.seed(
        int(
            seed
        )
    )

    np.random.seed(
        int(
            seed
        )
    )

    torch.manual_seed(
        int(
            seed
        )
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            int(
                seed
            )
        )


def configure_tf32(
    torch,
):
    if not torch.cuda.is_available():
        return

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def stable_json_bytes(
    obj,
):
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )


def sha256_json(
    obj,
):
    return hashlib.sha256(
        stable_json_bytes(
            obj
        )
    ).hexdigest()


def json_convert(
    value,
):
    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        return (
            None
            if np.isnan(
                value
            )
            else float(
                value
            )
        )

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(
            value
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                key
            ): json_convert(
                val
            )
            for key, val
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            json_convert(
                item
            )
            for item in value
        ]

    return value


def save_json(
    path: Path,
    obj,
):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_convert(
                obj
            ),
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def build_physical_mask():
    """
    Nie et al. Eq.(2).

    M_ij = 1 means a directed edge j -> i is allowed by the physics prior.
    """
    idx = {
        name: i
        for i, name
        in enumerate(
            NODE_NAMES
        )
    }

    T = set(
        NODE_GROUPS[
            "target_T"
        ]
    )

    S = set(
        NODE_GROUPS[
            "source_S"
        ]
    )

    K = set(
        NODE_GROUPS[
            "kinematic_K"
        ]
    )

    E = set(
        NODE_GROUPS[
            "environment_E"
        ]
    )

    M = np.zeros(
        (
            len(
                NODE_NAMES
            ),
            len(
                NODE_NAMES
            ),
        ),
        dtype=np.float32,
    )

    for i_name in NODE_NAMES:
        for j_name in NODE_NAMES:
            allowed = False

            if (
                j_name in (
                    S
                    | K
                )
                and i_name in T
            ):
                allowed = True

            if (
                j_name in E
                and i_name in S
            ):
                allowed = True

            if (
                j_name in E
                and i_name in E
            ):
                allowed = True

            if (
                i_name
                == j_name
            ):
                allowed = True

            if allowed:
                M[
                    idx[
                        i_name
                    ],
                    idx[
                        j_name
                    ],
                ] = 1.0

    if (
        M.shape
        != (
            10,
            10,
        )
        or int(
            M.sum()
        )
        != 35
    ):
        raise RuntimeError(
            f"Physical-mask audit failed: shape={M.shape}, ones={int(M.sum())}."
        )

    return M


def load_track_f_train(
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        payload = {
            key: data[
                key
            ]
            for key in data.files
        }

    required = [
        "X_raw",
        "y_wind_raw",
        "context_end_time_ns",
        "target_time_ns",
        "feature_names",
        "target_names",
        "horizons_min",
    ]

    for key in required:
        if key not in payload:
            raise KeyError(
                f"{path}: missing key '{key}', available={list(payload)}"
            )

    X = np.asarray(
        payload[
            "X_raw"
        ],
        dtype=np.float32,
    )

    y = np.asarray(
        payload[
            "y_wind_raw"
        ],
        dtype=np.float32,
    )

    context_end = np.asarray(
        payload[
            "context_end_time_ns"
        ],
        dtype=np.int64,
    ).reshape(
        -1
    )

    target_time = np.asarray(
        payload[
            "target_time_ns"
        ],
        dtype=np.int64,
    )

    feature_names = [
        str(
            x
        )
        for x in np.asarray(
            payload[
                "feature_names"
            ]
        ).tolist()
    ]

    target_names = [
        str(
            x
        )
        for x in np.asarray(
            payload[
                "target_names"
            ]
        ).tolist()
    ]

    horizons = [
        int(
            x
        )
        for x in np.asarray(
            payload[
                "horizons_min"
            ]
        ).tolist()
    ]

    if X.ndim != 3:
        raise RuntimeError(
            f"Expected X [N,60,10], got {X.shape}."
        )

    if y.ndim != 3:
        raise RuntimeError(
            f"Expected y [N,5,2], got {y.shape}."
        )

    if tuple(
        X.shape[
            1:
        ]
    ) != (
        LOOKBACK,
        len(
            NODE_NAMES
        ),
    ):
        raise RuntimeError(
            f"Track-F X schema mismatch: {X.shape}."
        )

    if tuple(
        y.shape[
            1:
        ]
    ) != (
        len(
            HORIZONS_MIN
        ),
        TARGET_COUNT,
    ):
        raise RuntimeError(
            f"Track-F y schema mismatch: {y.shape}."
        )

    if feature_names != NODE_NAMES:
        raise RuntimeError(
            f"Feature schema mismatch: {feature_names}"
        )

    if target_names != [
        "U",
        "V",
    ]:
        raise RuntimeError(
            f"Target schema mismatch: {target_names}"
        )

    if horizons != HORIZONS_MIN:
        raise RuntimeError(
            f"Horizon schema mismatch: {horizons}"
        )

    if not np.isfinite(
        X
    ).all():
        raise RuntimeError(
            "Track-F training X contains non-finite values."
        )

    if not np.isfinite(
        y
    ).all():
        raise RuntimeError(
            "Track-F training y contains non-finite values."
        )

    if len(
        context_end
    ) != len(
        X
    ):
        raise RuntimeError(
            "context_end_time_ns length mismatch."
        )

    if target_time.shape != (
        len(
            X
        ),
        len(
            HORIZONS_MIN
        ),
    ):
        raise RuntimeError(
            f"target_time_ns shape mismatch: {target_time.shape}."
        )

    if np.any(
        np.diff(
            context_end
        )
        <= 0
    ):
        raise RuntimeError(
            "context_end_time_ns is not strictly increasing."
        )

    return {
        "X": X,
        "y": y,
        "context_end_time_ns": context_end,
        "target_time_ns": target_time,
        "feature_names": feature_names,
        "target_names": target_names,
        "horizons_min": horizons,
    }


def build_rolling_folds(
    context_end_time_ns: np.ndarray,
    target_time_ns: np.ndarray,
    purge_samples: int = 10,
):
    n = int(
        len(
            context_end_time_ns
        )
    )

    val_size = n // 6

    if val_size < 1:
        raise RuntimeError(
            "Not enough samples to build rolling folds."
        )

    initial_origin = (
        n
        - 3
        * val_size
    )

    folds = []

    for fold_idx in range(
        3
    ):
        origin = (
            initial_origin
            + fold_idx
            * val_size
        )

        train_end_exclusive = (
            origin
            - int(
                purge_samples
            )
        )

        val_start = origin

        val_end = (
            origin
            + val_size
        )

        if fold_idx == 2:
            val_end = n

        if train_end_exclusive <= 0:
            raise RuntimeError(
                f"Fold {fold_idx+1}: effective training set is empty."
            )

        train_idx = np.arange(
            0,
            train_end_exclusive,
            dtype=np.int64,
        )

        val_idx = np.arange(
            val_start,
            val_end,
            dtype=np.int64,
        )

        max_train_target_ns = int(
            np.max(
                target_time_ns[
                    train_idx
                ]
            )
        )

        min_val_context_end_ns = int(
            np.min(
                context_end_time_ns[
                    val_idx
                ]
            )
        )

        leakage_pass = bool(
            max_train_target_ns
            < min_val_context_end_ns
        )

        if not leakage_pass:
            raise RuntimeError(
                f"Fold {fold_idx+1}: timestamp firewall failed: "
                f"max_train_target={np.datetime64(max_train_target_ns, 'ns')} "
                f">= min_val_context_end="
                f"{np.datetime64(min_val_context_end_ns, 'ns')}."
            )

        folds.append({
            "fold": fold_idx
            + 1,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "train_count": int(
                len(
                    train_idx
                )
            ),
            "val_count": int(
                len(
                    val_idx
                )
            ),
            "purge_count": int(
                purge_samples
            ),
            "origin_index": int(
                origin
            ),
            "train_first_context_end_ns": int(
                context_end_time_ns[
                    train_idx[
                        0
                    ]
                ]
            ),
            "train_last_context_end_ns": int(
                context_end_time_ns[
                    train_idx[
                        -1
                    ]
                ]
            ),
            "train_max_target_ns": max_train_target_ns,
            "val_first_context_end_ns": min_val_context_end_ns,
            "val_last_context_end_ns": int(
                context_end_time_ns[
                    val_idx[
                        -1
                    ]
                ]
            ),
            "val_max_target_ns": int(
                np.max(
                    target_time_ns[
                        val_idx
                    ]
                )
            ),
            "timestamp_firewall_pass": leakage_pass,
        })

    return folds


def ns_iso(
    value: int,
):
    return str(
        np.datetime64(
            int(
                value
            ),
            "ns",
        )
    )


def fit_fold_scalers(
    X_train: np.ndarray,
    y_train: np.ndarray,
):
    flat_X = np.asarray(
        X_train,
        dtype=np.float64,
    ).reshape(
        -1,
        X_train.shape[
            -1
        ],
    )

    flat_y = np.asarray(
        y_train,
        dtype=np.float64,
    ).reshape(
        -1,
        y_train.shape[
            -1
        ],
    )

    feature_mean = flat_X.mean(
        axis=0
    )

    feature_std = flat_X.std(
        axis=0
    )

    target_mean = flat_y.mean(
        axis=0
    )

    target_std = flat_y.std(
        axis=0
    )

    feature_std = np.where(
        feature_std
        < 1e-8,
        1.0,
        feature_std,
    )

    target_std = np.where(
        target_std
        < 1e-8,
        1.0,
        target_std,
    )

    return {
        "feature_mean": feature_mean.astype(
            np.float32
        ),
        "feature_std": feature_std.astype(
            np.float32
        ),
        "target_mean": target_mean.astype(
            np.float32
        ),
        "target_std": target_std.astype(
            np.float32
        ),
    }


def standardize_X(
    X,
    scaler,
):
    return (
        np.asarray(
            X,
            dtype=np.float32,
        )
        - scaler[
            "feature_mean"
        ][
            None,
            None,
            :
        ]
    ) / scaler[
        "feature_std"
    ][
        None,
        None,
        :
    ]


def standardize_y(
    y,
    scaler,
):
    return (
        np.asarray(
            y,
            dtype=np.float32,
        )
        - scaler[
            "target_mean"
        ][
            None,
            None,
            :
        ]
    ) / scaler[
        "target_std"
    ][
        None,
        None,
        :
    ]


def inverse_y(
    y_z,
    scaler,
):
    return (
        np.asarray(
            y_z,
            dtype=np.float64,
        )
        * scaler[
            "target_std"
        ][
            None,
            None,
            :
        ]
        + scaler[
            "target_mean"
        ][
            None,
            None,
            :
        ]
    )


def build_model_classes(
    torch,
    nn,
    F,
):
    class BayesianAdjacency(
        nn.Module
    ):
        def __init__(
            self,
            physical_mask,
            initial_sigma: float,
        ):
            super().__init__()

            mask = torch.as_tensor(
                physical_mask,
                dtype=torch.float32,
            )

            self.register_buffer(
                "physical_mask",
                mask,
            )

            self.mu_adj = nn.Parameter(
                mask.clone()
            )

            rho0 = math.log(
                math.expm1(
                    float(
                        initial_sigma
                    )
                )
            )

            self.rho_adj = nn.Parameter(
                torch.full_like(
                    mask,
                    float(
                        rho0
                    ),
                )
            )

        def sigma(
            self,
        ):
            return (
                F.softplus(
                    self.rho_adj
                )
                + 1e-6
            )

        def sample(
            self,
            stochastic: bool,
        ):
            if stochastic:
                eps = torch.randn_like(
                    self.mu_adj
                )

                return (
                    self.mu_adj
                    + self.sigma()
                    * eps
                )

            return self.mu_adj

        def kl_divergence(
            self,
        ):
            sigma = self.sigma()

            return 0.5 * torch.sum(
                sigma.pow(
                    2
                )
                + (
                    self.mu_adj
                    - self.physical_mask
                ).pow(
                    2
                )
                - 1.0
                - torch.log(
                    sigma.pow(
                        2
                    )
                )
            )

        def normalized(
            self,
            stochastic: bool,
        ):
            A = self.sample(
                stochastic
            )

            eye = torch.eye(
                A.shape[
                    0
                ],
                dtype=A.dtype,
                device=A.device,
            )

            A_safe = (
                A
                + 1e-6
                * eye
            )

            degree = torch.sum(
                torch.abs(
                    A_safe
                ),
                dim=1,
            ).clamp_min(
                1e-6
            )

            inv_sqrt = degree.rsqrt()

            A_norm = (
                inv_sqrt[
                    :,
                    None
                ]
                * A_safe
                * inv_sqrt[
                    None,
                    :
                ]
            )

            return (
                A_norm,
                A,
            )

    class GraphConvBlock(
        nn.Module
    ):
        def __init__(
            self,
            in_dim: int,
            out_dim: int,
            dropout: float,
        ):
            super().__init__()

            self.linear = nn.Linear(
                in_dim,
                out_dim,
            )

            self.norm = nn.LayerNorm(
                out_dim
            )

            self.dropout = nn.Dropout(
                dropout
            )

        def forward(
            self,
            h,
            A_norm,
        ):
            agg = torch.einsum(
                "ij,btjd->btid",
                A_norm,
                h,
            )

            out = self.linear(
                agg
            )

            out = self.norm(
                out
            )

            out = F.gelu(
                out
            )

            return self.dropout(
                out
            )

    class BPSTGNN(
        nn.Module
    ):
        def __init__(
            self,
            physical_mask,
            search_cfg: SearchConfig,
            fixed_cfg: FixedTrainConfig,
        ):
            super().__init__()

            self.search_cfg = search_cfg
            self.fixed_cfg = fixed_cfg

            self.bayes_graph = BayesianAdjacency(
                physical_mask,
                initial_sigma=fixed_cfg.initial_adj_sigma,
            )

            self.scalar_expand = nn.Linear(
                1,
                fixed_cfg.node_embed_dim,
            )

            self.node_embedding = nn.Parameter(
                torch.zeros(
                    len(
                        NODE_NAMES
                    ),
                    fixed_cfg.node_embed_dim,
                )
            )

            nn.init.normal_(
                self.node_embedding,
                mean=0.0,
                std=0.02,
            )

            layers = []

            in_dim = fixed_cfg.node_embed_dim

            for _ in range(
                fixed_cfg.gcn_layers
            ):
                layers.append(
                    GraphConvBlock(
                        in_dim,
                        search_cfg.gcn_hidden_dim,
                        fixed_cfg.dropout,
                    )
                )

                in_dim = search_cfg.gcn_hidden_dim

            self.gcn_layers = nn.ModuleList(
                layers
            )

            self.gru = nn.GRU(
                input_size=search_cfg.gcn_hidden_dim,
                hidden_size=search_cfg.gru_hidden_dim,
                num_layers=fixed_cfg.gru_layers,
                batch_first=True,
                dropout=(
                    fixed_cfg.dropout
                    if fixed_cfg.gru_layers
                    > 1
                    else 0.0
                ),
            )

            self.mean_mlp = nn.Sequential(
                nn.Linear(
                    search_cfg.gru_hidden_dim,
                    fixed_cfg.mlp_hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(
                    fixed_cfg.dropout
                ),
                nn.Linear(
                    fixed_cfg.mlp_hidden_dim,
                    len(
                        HORIZONS_MIN
                    )
                    * TARGET_COUNT,
                ),
            )

            self.var_mlp = nn.Sequential(
                nn.Linear(
                    search_cfg.gru_hidden_dim,
                    fixed_cfg.mlp_hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(
                    fixed_cfg.dropout
                ),
                nn.Linear(
                    fixed_cfg.mlp_hidden_dim,
                    len(
                        HORIZONS_MIN
                    )
                    * TARGET_COUNT,
                ),
            )

            final_var = self.var_mlp[
                -1
            ]

            nn.init.uniform_(
                final_var.weight,
                -1e-4,
                1e-4,
            )

            nn.init.zeros_(
                final_var.bias
            )

        def set_variance_trainable(
            self,
            enabled: bool,
        ):
            for p in self.var_mlp.parameters():
                p.requires_grad = bool(
                    enabled
                )

        def forward(
            self,
            x,
            stochastic_graph: bool,
            detach_variance_features: bool = True,
        ):
            A_norm, A_sample = self.bayes_graph.normalized(
                stochastic=stochastic_graph
            )

            h = self.scalar_expand(
                x.unsqueeze(
                    -1
                )
            )

            h = (
                h
                + self.node_embedding[
                    None,
                    None,
                    :,
                    :,
                ]
            )

            for layer in self.gcn_layers:
                h = layer(
                    h,
                    A_norm,
                )

            graph_sequence = h.mean(
                dim=2
            )

            gru_out, _ = self.gru(
                graph_sequence
            )

            shared = gru_out[
                :,
                -1,
                :
            ]

            mu = self.mean_mlp(
                shared
            ).view(
                x.shape[
                    0
                ],
                len(
                    HORIZONS_MIN
                ),
                TARGET_COUNT,
            )

            var_input = (
                shared.detach()
                if detach_variance_features
                else shared
            )

            logvar = self.var_mlp(
                var_input
            ).view(
                x.shape[
                    0
                ],
                len(
                    HORIZONS_MIN
                ),
                TARGET_COUNT,
            )

            logvar = torch.clamp(
                logvar,
                min=self.fixed_cfg.logvar_min,
                max=self.fixed_cfg.logvar_max,
            )

            return {
                "mu": mu,
                "logvar": logvar,
                "variance": torch.exp(
                    logvar
                ),
                "KL": self.bayes_graph.kl_divergence(),
                "A_sample": A_sample,
                "A_norm": A_norm,
            }

    return BPSTGNN


def count_trainable_parameters(
    model,
):
    return int(
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )


def mse_loss(
    torch,
    mu,
    target,
):
    return torch.mean(
        (
            mu
            - target
        ) ** 2
    )


def gaussian_nll(
    torch,
    mu,
    logvar,
    target,
):
    return torch.mean(
        0.5
        * (
            logvar
            + (
                target
                - mu
            ) ** 2
            * torch.exp(
                -logvar
            )
        )
    )


def compute_loss(
    torch,
    output,
    target,
    cfg: SearchConfig,
    phase: int,
):
    mse = mse_loss(
        torch,
        output[
            "mu"
        ],
        target,
    )

    kl = output[
        "KL"
    ]

    if phase == 1:
        nll = torch.zeros(
            (),
            dtype=mse.dtype,
            device=mse.device,
        )

        total = (
            mse
            + cfg.lambda_kl
            * kl
        )

    else:
        nll = gaussian_nll(
            torch,
            output[
                "mu"
            ],
            output[
                "logvar"
            ],
            target,
        )

        total = (
            mse
            + cfg.lambda_nll
            * nll
            + cfg.lambda_kl
            * kl
        )

    return {
        "total": total,
        "mse": mse,
        "nll": nll,
        "kl": kl,
    }


def create_grad_scaler(
    torch,
    enabled: bool,
):
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


def autocast_context(
    torch,
    enabled: bool,
):
    if not enabled:
        return torch.autocast(
            device_type="cuda",
            enabled=False,
        )

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


def iter_sequential_batches(
    X_z: np.ndarray,
    y_z: np.ndarray,
    batch_size: int,
):
    n = int(
        X_z.shape[
            0
        ]
    )

    for start in range(
        0,
        n,
        int(
            batch_size
        ),
    ):
        end = min(
            n,
            start
            + int(
                batch_size
            ),
        )

        yield (
            X_z[
                start:end
            ],
            y_z[
                start:end
            ],
        )


def calculate_raw_metrics(
    truth_raw: np.ndarray,
    pred_raw: np.ndarray,
):
    err = (
        np.asarray(
            pred_raw,
            dtype=np.float64,
        )
        - np.asarray(
            truth_raw,
            dtype=np.float64,
        )
    )

    rows = []

    for h_idx, h in enumerate(
        HORIZONS_MIN
    ):
        e = err[
            :,
            h_idx,
            :
        ]

        u_rmse = float(
            np.sqrt(
                np.mean(
                    e[
                        :,
                        0
                    ] ** 2
                )
            )
        )

        v_rmse = float(
            np.sqrt(
                np.mean(
                    e[
                        :,
                        1
                    ] ** 2
                )
            )
        )

        vector_rmse = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        e ** 2,
                        axis=1,
                    )
                )
            )
        )

        rows.append({
            "horizon_min": int(
                h
            ),
            "U_RMSE_mps": u_rmse,
            "V_RMSE_mps": v_rmse,
            "wind_vector_RMSE_mps": vector_rmse,
        })

    mean_vector = float(
        np.mean(
            [
                row[
                    "wind_vector_RMSE_mps"
                ]
                for row in rows
            ]
        )
    )

    mean_u = float(
        np.mean(
            [
                row[
                    "U_RMSE_mps"
                ]
                for row in rows
            ]
        )
    )

    mean_v = float(
        np.mean(
            [
                row[
                    "V_RMSE_mps"
                ]
                for row in rows
            ]
        )
    )

    return {
        "per_horizon": rows,
        "mean_vector_RMSE_mps": mean_vector,
        "mean_U_RMSE_mps": mean_u,
        "mean_V_RMSE_mps": mean_v,
    }


def evaluate_model(
    torch,
    model,
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    scaler: Dict[str, np.ndarray],
    device,
    batch_size: int,
    amp_enabled: bool,
):
    model.eval()

    preds_z = []

    val_mse_sum = 0.0

    val_count = 0

    X_z = standardize_X(
        X_raw,
        scaler,
    )

    y_z = standardize_y(
        y_raw,
        scaler,
    )

    with torch.no_grad():
        for Xb, yb in iter_sequential_batches(
            X_z,
            y_z,
            batch_size,
        ):
            xb = torch.from_numpy(
                Xb
            ).to(
                device,
                non_blocking=True,
            )

            ybt = torch.from_numpy(
                yb
            ).to(
                device,
                non_blocking=True,
            )

            with autocast_context(
                torch,
                amp_enabled,
            ):
                out = model(
                    xb,
                    stochastic_graph=False,
                    detach_variance_features=True,
                )

                mse = mse_loss(
                    torch,
                    out[
                        "mu"
                    ],
                    ybt,
                )

            batch_n = int(
                len(
                    Xb
                )
            )

            val_mse_sum += float(
                mse.detach()
                .float()
                .cpu()
                .item()
            ) * batch_n

            val_count += batch_n

            preds_z.append(
                out[
                    "mu"
                ]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            del (
                xb,
                ybt,
                out,
                mse,
            )

    pred_z = np.concatenate(
        preds_z,
        axis=0,
    )

    pred_raw = inverse_y(
        pred_z,
        scaler,
    )

    raw_metrics = calculate_raw_metrics(
        y_raw,
        pred_raw,
    )

    return {
        "standardized_MSE": float(
            val_mse_sum
            / max(
                val_count,
                1,
            )
        ),
        **raw_metrics,
    }


def train_one_epoch(
    torch,
    model,
    optimizer,
    scaler_amp,
    X_z,
    y_z,
    cfg: SearchConfig,
    fixed_cfg: FixedTrainConfig,
    phase: int,
    device,
    amp_enabled: bool,
):
    model.train()

    model.set_variance_trainable(
        phase == 2
    )

    total_sum = 0.0

    mse_sum = 0.0

    nll_sum = 0.0

    kl_sum = 0.0

    sample_count = 0

    for Xb, yb in iter_sequential_batches(
        X_z,
        y_z,
        fixed_cfg.batch_size,
    ):
        xb = torch.from_numpy(
            Xb
        ).to(
            device,
            non_blocking=True,
        )

        ybt = torch.from_numpy(
            yb
        ).to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast_context(
            torch,
            amp_enabled,
        ):
            output = model(
                xb,
                stochastic_graph=True,
                detach_variance_features=True,
            )

            losses = compute_loss(
                torch,
                output,
                ybt,
                cfg,
                phase,
            )

        total = losses[
            "total"
        ]

        if not torch.isfinite(
            total
        ):
            raise FloatingPointError(
                "Non-finite training loss."
            )

        if scaler_amp is not None:
            scaler_amp.scale(
                total
            ).backward()

            scaler_amp.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                fixed_cfg.grad_clip_norm,
            )

            scaler_amp.step(
                optimizer
            )

            scaler_amp.update()

        else:
            total.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                fixed_cfg.grad_clip_norm,
            )

            optimizer.step()

        batch_n = int(
            len(
                Xb
            )
        )

        total_sum += float(
            losses[
                "total"
            ].detach()
            .float()
            .cpu()
            .item()
        ) * batch_n

        mse_sum += float(
            losses[
                "mse"
            ].detach()
            .float()
            .cpu()
            .item()
        ) * batch_n

        nll_sum += float(
            losses[
                "nll"
            ].detach()
            .float()
            .cpu()
            .item()
        ) * batch_n

        kl_sum += float(
            losses[
                "kl"
            ].detach()
            .float()
            .cpu()
            .item()
        ) * batch_n

        sample_count += batch_n

        del (
            xb,
            ybt,
            output,
            losses,
            total,
        )

    denom = max(
        sample_count,
        1,
    )

    return {
        "train_total_loss": total_sum
        / denom,
        "train_MSE": mse_sum
        / denom,
        "train_NLL": nll_sum
        / denom,
        "train_KL": kl_sum
        / denom,
    }


def state_dict_cpu(
    model,
):
    return {
        key: value.detach()
        .cpu()
        .clone()
        for key, value
        in model.state_dict().items()
    }


def train_fold(
    *,
    torch,
    nn,
    F,
    physical_mask: np.ndarray,
    config: SearchConfig,
    fold: Dict,
    X: np.ndarray,
    y: np.ndarray,
    device,
    output_dir: Path,
    debug_fast: bool,
):
    seed = SCREENING_FOLD_SEEDS[
        int(
            fold[
                "fold"
            ]
        )
        - 1
    ]

    seed_everything(
        torch,
        seed,
    )

    train_idx = fold[
        "train_idx"
    ]

    val_idx = fold[
        "val_idx"
    ]

    X_train = X[
        train_idx
    ]

    y_train = y[
        train_idx
    ]

    X_val = X[
        val_idx
    ]

    y_val = y[
        val_idx
    ]

    scaler = fit_fold_scalers(
        X_train,
        y_train,
    )

    X_train_z = standardize_X(
        X_train,
        scaler,
    )

    y_train_z = standardize_y(
        y_train,
        scaler,
    )

    ModelClass = build_model_classes(
        torch,
        nn,
        F,
    )

    model = ModelClass(
        physical_mask,
        config,
        FIXED,
    ).to(
        device
    )

    parameter_count = count_trainable_parameters(
        model
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FIXED.learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=FIXED.scheduler_factor,
        patience=FIXED.scheduler_patience,
        min_lr=FIXED.min_learning_rate,
    )

    amp_enabled = bool(
        FIXED.use_amp
        and device.type
        == "cuda"
    )

    scaler_amp = create_grad_scaler(
        torch,
        amp_enabled,
    )

    if debug_fast:
        max_epochs = 6
        phase2_start = 2
        early_patience = 3
    else:
        max_epochs = FIXED.max_epochs
        phase2_start = (
            FIXED.phase2_start_epoch_zero_based
        )
        early_patience = (
            FIXED.early_stop_patience
        )

    best_metric = math.inf

    best_epoch = None

    best_state = None

    best_val = None

    patience_counter = 0

    history_rows = []

    start_time = time.perf_counter()

    for epoch_zero in range(
        max_epochs
    ):
        phase = (
            1
            if epoch_zero
            < phase2_start
            else 2
        )

        train_metrics = train_one_epoch(
            torch,
            model,
            optimizer,
            scaler_amp,
            X_train_z,
            y_train_z,
            config,
            FIXED,
            phase,
            device,
            amp_enabled,
        )

        val_metrics = evaluate_model(
            torch,
            model,
            X_val,
            y_val,
            scaler,
            device,
            FIXED.batch_size,
            amp_enabled,
        )

        lr = float(
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        eligible = bool(
            epoch_zero
            >= phase2_start
        )

        improved = False

        if eligible:
            current_metric = float(
                val_metrics[
                    "mean_vector_RMSE_mps"
                ]
            )

            if (
                current_metric
                < best_metric
                - 1e-8
            ):
                best_metric = current_metric

                best_epoch = int(
                    epoch_zero
                )

                best_state = state_dict_cpu(
                    model
                )

                best_val = val_metrics

                patience_counter = 0

                improved = True

            else:
                patience_counter += 1

            scheduler.step(
                float(
                    val_metrics[
                        "standardized_MSE"
                    ]
                )
            )

        history_rows.append({
            "config_id": config.config_id,
            "fold": int(
                fold[
                    "fold"
                ]
            ),
            "screening_seed": int(
                seed
            ),
            "epoch_zero_based": int(
                epoch_zero
            ),
            "epoch_one_based": int(
                epoch_zero
                + 1
            ),
            "phase": int(
                phase
            ),
            "checkpoint_eligible": eligible,
            "checkpoint_improved": improved,
            "learning_rate": lr,
            **train_metrics,
            "val_standardized_MSE": float(
                val_metrics[
                    "standardized_MSE"
                ]
            ),
            "val_mean_vector_RMSE_mps": float(
                val_metrics[
                    "mean_vector_RMSE_mps"
                ]
            ),
            "val_mean_U_RMSE_mps": float(
                val_metrics[
                    "mean_U_RMSE_mps"
                ]
            ),
            "val_mean_V_RMSE_mps": float(
                val_metrics[
                    "mean_V_RMSE_mps"
                ]
            ),
            **{
                f"val_vector_RMSE_{row['horizon_min']}min_mps": float(
                    row[
                        "wind_vector_RMSE_mps"
                    ]
                )
                for row
                in val_metrics[
                    "per_horizon"
                ]
            },
        })

        log(
            f"      epoch={epoch_zero+1:02d}/{max_epochs} "
            f"phase={phase} "
            f"trainMSE={train_metrics['train_MSE']:.5f} "
            f"valMSE={val_metrics['standardized_MSE']:.5f} "
            f"valVec={val_metrics['mean_vector_RMSE_mps']:.4f} "
            f"lr={lr:.2e}"
            + (
                " *"
                if improved
                else ""
            )
        )

        if (
            eligible
            and patience_counter
            >= early_patience
        ):
            log(
                f"      early stop after {patience_counter} "
                "eligible epochs without improvement."
            )
            break

    elapsed = time.perf_counter() - start_time

    if (
        best_state
        is None
        or best_epoch
        is None
        or best_val
        is None
    ):
        raise RuntimeError(
            f"{config.config_id}/fold{fold['fold']}: no eligible finite "
            "phase-2 checkpoint was produced."
        )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    # Re-evaluate the frozen best checkpoint for a final consistency audit.
    confirmed = evaluate_model(
        torch,
        model,
        X_val,
        y_val,
        scaler,
        device,
        FIXED.batch_size,
        amp_enabled,
    )

    if abs(
        confirmed[
            "mean_vector_RMSE_mps"
        ]
        - best_val[
            "mean_vector_RMSE_mps"
        ]
    ) > 1e-5:
        raise RuntimeError(
            f"{config.config_id}/fold{fold['fold']}: best checkpoint "
            "re-evaluation mismatch."
        )

    histories_dir = (
        output_dir
        / "histories"
    )

    checkpoints_dir = (
        output_dir
        / "checkpoints"
    )

    histories_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoints_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        histories_dir
        / f"{config.config_id}_fold{fold['fold']}.csv"
    )

    pd.DataFrame(
        history_rows
    ).to_csv(
        history_path,
        index=False,
        encoding="utf-8-sig",
    )

    checkpoint_path = (
        checkpoints_dir
        / f"{config.config_id}_fold{fold['fold']}.pt"
    )

    torch.save({
        "script_version": SCRIPT_VERSION,
        "config": asdict(
            config
        ),
        "fixed_config": asdict(
            FIXED
        ),
        "fold": int(
            fold[
                "fold"
            ]
        ),
        "screening_seed": int(
            seed
        ),
        "best_epoch_zero_based": int(
            best_epoch
        ),
        "best_epoch_one_based": int(
            best_epoch
            + 1
        ),
        "state_dict": best_state,
        "feature_mean": scaler[
            "feature_mean"
        ],
        "feature_std": scaler[
            "feature_std"
        ],
        "target_mean": scaler[
            "target_mean"
        ],
        "target_std": scaler[
            "target_std"
        ],
        "validation_metrics": confirmed,
        "parameter_count": int(
            parameter_count
        ),
        "track": "F",
        "scientific_role": "rolling_cv_screening_checkpoint",
    }, checkpoint_path)

    result = {
        "config_id": config.config_id,
        "fold": int(
            fold[
                "fold"
            ]
        ),
        "screening_seed": int(
            seed
        ),
        "status": "completed_finite",
        "gcn_hidden_dim": int(
            config.gcn_hidden_dim
        ),
        "gru_hidden_dim": int(
            config.gru_hidden_dim
        ),
        "lambda_KL": float(
            config.lambda_kl
        ),
        "lambda_NLL": float(
            config.lambda_nll
        ),
        "parameter_count": int(
            parameter_count
        ),
        "paper_parameter_count": 242580,
        "parameter_count_difference_from_paper": int(
            parameter_count
            - 242580
        ),
        "train_samples": int(
            len(
                train_idx
            )
        ),
        "val_samples": int(
            len(
                val_idx
            )
        ),
        "best_epoch_zero_based": int(
            best_epoch
        ),
        "best_epoch_one_based": int(
            best_epoch
            + 1
        ),
        "val_mean_vector_RMSE_mps": float(
            confirmed[
                "mean_vector_RMSE_mps"
            ]
        ),
        "val_mean_U_RMSE_mps": float(
            confirmed[
                "mean_U_RMSE_mps"
            ]
        ),
        "val_mean_V_RMSE_mps": float(
            confirmed[
                "mean_V_RMSE_mps"
            ]
        ),
        "val_standardized_MSE": float(
            confirmed[
                "standardized_MSE"
            ]
        ),
        **{
            f"val_vector_RMSE_{row['horizon_min']}min_mps": float(
                row[
                    "wind_vector_RMSE_mps"
                ]
            )
            for row
            in confirmed[
                "per_horizon"
            ]
        },
        "elapsed_seconds": float(
            elapsed
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
        "history_path": str(
            history_path
        ),
    }

    del (
        model,
        optimizer,
        scheduler,
        scaler_amp,
        X_train_z,
        y_train_z,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def write_or_verify_frozen_search_space(
    output_dir: Path,
    debug_fast: bool,
):
    search_payload = {
        "stage": "11C2",
        "track": "F_common_task",
        "script_version": SCRIPT_VERSION,
        "scientific_run": bool(
            not debug_fast
        ),
        "fixed_config": asdict(
            FIXED
        ),
        "configs": [
            asdict(
                cfg
            )
            for cfg in SEARCH_CONFIGS
        ],
        "screening_fold_seeds": SCREENING_FOLD_SEEDS,
        "final_seed_schedule_for_11C3": FINAL_SEED_SCHEDULE,
        "selection_metric": (
            "mean raw-unit wind-vector RMSE across horizons "
            "[1,2,3,5,10] min, averaged over 3 rolling folds"
        ),
        "tie_breakers": [
            "lower mean CV vector RMSE",
            "lower fold standard deviation",
            "fewer trainable parameters",
            "lexicographically smaller config_id",
        ],
        "data_firewall": {
            "SD1090_outer_train": True,
            "SD1090_outer_validation": False,
            "SD1090_outer_test": False,
            "SD1033_external": False,
        },
    }

    digest = sha256_json(
        search_payload
    )

    frozen_path = (
        output_dir
        / "frozen_search_space.json"
    )

    if frozen_path.exists():
        existing = json.loads(
            frozen_path.read_text(
                encoding="utf-8"
            )
        )

        existing_digest = existing.get(
            "sha256"
        )

        if existing_digest != digest:
            raise RuntimeError(
                "Existing frozen_search_space.json differs from the current "
                "predeclared search space. Refusing silent search-space edits. "
                "Use a new output/version directory for a changed protocol."
            )

    else:
        payload_to_write = dict(
            search_payload
        )

        payload_to_write[
            "sha256"
        ] = digest

        save_json(
            frozen_path,
            payload_to_write,
        )

    return (
        search_payload,
        digest,
    )


def load_existing_fold_results(
    path: Path,
):
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        path
    )

    return df


def append_fold_result(
    path: Path,
    result: Dict,
):
    row_df = pd.DataFrame(
        [
            result
        ]
    )

    if path.exists():
        existing = pd.read_csv(
            path
        )

        key_mask = (
            (
                existing[
                    "config_id"
                ].astype(
                    str
                )
                == str(
                    result[
                        "config_id"
                    ]
                )
            )
            & (
                existing[
                    "fold"
                ].astype(
                    int
                )
                == int(
                    result[
                        "fold"
                    ]
                )
            )
        )

        existing = existing.loc[
            ~key_mask
        ]

        combined = pd.concat(
            [
                existing,
                row_df,
            ],
            ignore_index=True,
        )

    else:
        combined = row_df

    combined = combined.sort_values(
        [
            "config_id",
            "fold",
        ]
    ).reset_index(
        drop=True
    )

    combined.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def completed_result_exists(
    existing: pd.DataFrame,
    config_id: str,
    fold: int,
):
    if existing.empty:
        return False

    required_cols = {
        "config_id",
        "fold",
        "status",
        "val_mean_vector_RMSE_mps",
    }

    if not required_cols.issubset(
        existing.columns
    ):
        return False

    rows = existing.loc[
        (
            existing[
                "config_id"
            ].astype(
                str
            )
            == str(
                config_id
            )
        )
        & (
            existing[
                "fold"
            ].astype(
                int
            )
            == int(
                fold
            )
        )
    ]

    if rows.empty:
        return False

    row = rows.iloc[
        -1
    ]

    return bool(
        str(
            row[
                "status"
            ]
        )
        == "completed_finite"
        and np.isfinite(
            float(
                row[
                    "val_mean_vector_RMSE_mps"
                ]
            )
        )
    )


def summarize_candidates(
    fold_df: pd.DataFrame,
    expected_config_ids: Sequence[str],
):
    rows = []

    for config_id in expected_config_ids:
        sub = fold_df.loc[
            (
                fold_df[
                    "config_id"
                ].astype(
                    str
                )
                == str(
                    config_id
                )
            )
            & (
                fold_df[
                    "status"
                ]
                == "completed_finite"
            )
        ].copy()

        if len(
            sub
        ) == 0:
            continue

        vector = sub[
            "val_mean_vector_RMSE_mps"
        ].to_numpy(
            dtype=float
        )

        rows.append({
            "config_id": config_id,
            "completed_folds": int(
                len(
                    sub
                )
            ),
            "CV_mean_vector_RMSE_mps": float(
                np.mean(
                    vector
                )
            ),
            "CV_std_vector_RMSE_mps": float(
                np.std(
                    vector,
                    ddof=0,
                )
            ),
            "CV_mean_U_RMSE_mps": float(
                sub[
                    "val_mean_U_RMSE_mps"
                ].mean()
            ),
            "CV_mean_V_RMSE_mps": float(
                sub[
                    "val_mean_V_RMSE_mps"
                ].mean()
            ),
            "parameter_count": int(
                sub[
                    "parameter_count"
                ].iloc[
                    0
                ]
            ),
            "mean_best_epoch_one_based": float(
                sub[
                    "best_epoch_one_based"
                ].mean()
            ),
            "total_elapsed_seconds": float(
                sub[
                    "elapsed_seconds"
                ].sum()
            ),
            **{
                f"CV_mean_vector_RMSE_{h}min_mps": float(
                    sub[
                        f"val_vector_RMSE_{h}min_mps"
                    ].mean()
                )
                for h in HORIZONS_MIN
            },
        })

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(
        rows
    )

    summary[
        "complete_three_fold_candidate"
    ] = (
        summary[
            "completed_folds"
        ]
        == 3
    )

    summary = summary.sort_values(
        [
            "CV_mean_vector_RMSE_mps",
            "CV_std_vector_RMSE_mps",
            "parameter_count",
            "config_id",
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    summary[
        "rank"
    ] = np.arange(
        1,
        len(
            summary
        )
        + 1,
    )

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        default=str(
            DEFAULT_PROJECT_ROOT
        ),
    )

    parser.add_argument(
        "--dataset-dir",
        default=None,
        help=(
            "Stage-11C0 output directory."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    parser.add_argument(
        "--debug-fast",
        action="store_true",
        help=(
            "NON-SCIENTIFIC: B04/fold1 only, shortened phase schedule."
        ),
    )

    parser.add_argument(
        "--config-ids",
        default=None,
        help=(
            "Optional comma-separated subset for debugging/resume. "
            "A scientific selected_config.json is written only if every "
            "predeclared configuration completes all three folds."
        ),
    )

    parser.add_argument(
        "--folds",
        default=None,
        help=(
            "Optional comma-separated fold subset, e.g. 1,2. "
            "Scientific selection still requires all folds."
        ),
    )

    args = parser.parse_args()

    project_root = Path(
        args.project_root
    )

    frozen_dataset_dir = (
        project_root
        / "data"
        / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
    )

    dataset_dir = (
        Path(
            args.dataset_dir
        )
        if args.dataset_dir
        else (
            frozen_dataset_dir
            / "11C0_BP_STGNN_comparison_datasets_v0_1"
        )
    )

    output_dir = (
        Path(
            args.output_dir
        )
        if args.output_dir
        else (
            frozen_dataset_dir
            / "11C2_BP_STGNN_train_only_rolling_cv_v0_1"
        )
    )

    # Debug runs use a physically separate directory so their shortened
    # phase schedule/checkpoints can never be mistaken for or resumed as
    # scientific 11C2 fold results.
    if args.debug_fast:
        output_dir = output_dir.parent / (
            output_dir.name
            + "_DEBUG_FAST"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        dataset_dir
        / "track_F_common_task"
        / "SD1090_train.npz"
    )

    if args.debug_fast:
        active_configs = [
            cfg
            for cfg in SEARCH_CONFIGS
            if cfg.config_id
            == "B04"
        ]

        active_folds = [
            1
        ]

    else:
        if args.config_ids:
            requested = {
                x.strip()
                for x in args.config_ids.split(
                    ","
                )
                if x.strip()
            }

            known = {
                cfg.config_id
                for cfg in SEARCH_CONFIGS
            }

            unknown = requested - known

            if unknown:
                raise ValueError(
                    f"Unknown config IDs: {sorted(unknown)}"
                )

            active_configs = [
                cfg
                for cfg in SEARCH_CONFIGS
                if cfg.config_id
                in requested
            ]

        else:
            active_configs = list(
                SEARCH_CONFIGS
            )

        if args.folds:
            active_folds = [
                int(
                    x.strip()
                )
                for x in args.folds.split(
                    ","
                )
                if x.strip()
            ]

            if not set(
                active_folds
            ).issubset(
                {
                    1,
                    2,
                    3,
                }
            ):
                raise ValueError(
                    "--folds values must be in {1,2,3}."
                )

        else:
            active_folds = [
                1,
                2,
                3,
            ]

    torch, nn, F = import_torch()

    configure_tf32(
        torch
    )

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

    # Runtime override only; frozen scientific config still records AMP=True.
    runtime_amp_enabled = bool(
        FIXED.use_amp
        and not args.no_amp
        and device.type
        == "cuda"
    )

    data = load_track_f_train(
        train_path
    )

    X = data[
        "X"
    ]

    y = data[
        "y"
    ]

    folds = build_rolling_folds(
        data[
            "context_end_time_ns"
        ],
        data[
            "target_time_ns"
        ],
        purge_samples=FIXED.purge_samples,
    )

    physical_mask = build_physical_mask()

    # Cross-check Stage-11C0 frozen mask.
    mask_path = (
        dataset_dir
        / "physical_mask.npy"
    )

    if mask_path.exists():
        frozen_mask = np.load(
            mask_path
        ).astype(
            np.float32
        )

        if not np.array_equal(
            frozen_mask,
            physical_mask,
        ):
            raise RuntimeError(
                "11C2 physical mask differs from frozen 11C0 mask."
            )

    search_payload, search_digest = (
        write_or_verify_frozen_search_space(
            output_dir,
            debug_fast=args.debug_fast,
        )
    )

    fold_table_rows = []

    for fold in folds:
        fold_table_rows.append({
            "fold": int(
                fold[
                    "fold"
                ]
            ),
            "train_samples": int(
                fold[
                    "train_count"
                ]
            ),
            "purge_samples": int(
                fold[
                    "purge_count"
                ]
            ),
            "validation_samples": int(
                fold[
                    "val_count"
                ]
            ),
            "screening_seed": int(
                SCREENING_FOLD_SEEDS[
                    int(
                        fold[
                            "fold"
                        ]
                    )
                    - 1
                ]
            ),
            "train_first_context_end": ns_iso(
                fold[
                    "train_first_context_end_ns"
                ]
            ),
            "train_last_context_end": ns_iso(
                fold[
                    "train_last_context_end_ns"
                ]
            ),
            "train_max_target": ns_iso(
                fold[
                    "train_max_target_ns"
                ]
            ),
            "validation_first_context_end": ns_iso(
                fold[
                    "val_first_context_end_ns"
                ]
            ),
            "validation_last_context_end": ns_iso(
                fold[
                    "val_last_context_end_ns"
                ]
            ),
            "validation_max_target": ns_iso(
                fold[
                    "val_max_target_ns"
                ]
            ),
            "timestamp_firewall_pass": bool(
                fold[
                    "timestamp_firewall_pass"
                ]
            ),
        })

    pd.DataFrame(
        fold_table_rows
    ).to_csv(
        output_dir
        / "rolling_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dataset_audit = {
        "script_version": SCRIPT_VERSION,
        "track": "F_common_task",
        "dataset_path": str(
            train_path
        ),
        "samples": int(
            X.shape[
                0
            ]
        ),
        "X_shape": list(
            X.shape
        ),
        "y_shape": list(
            y.shape
        ),
        "feature_names": data[
            "feature_names"
        ],
        "target_names": data[
            "target_names"
        ],
        "horizons_min": data[
            "horizons_min"
        ],
        "finite_X": bool(
            np.isfinite(
                X
            ).all()
        ),
        "finite_y": bool(
            np.isfinite(
                y
            ).all()
        ),
        "physical_mask_ones": int(
            physical_mask.sum()
        ),
        "search_space_sha256": search_digest,
        "SD1090_outer_validation_accessed": False,
        "SD1090_outer_test_accessed": False,
        "SD1033_accessed": False,
    }

    save_json(
        output_dir
        / "dataset_audit.json",
        dataset_audit,
    )

    log(
        "="
        * 120
    )

    log(
        "11C2 - BP-STGNN-F TRAIN-ONLY ROLLING-CV TUNING"
    )

    log(
        "="
        * 120
    )

    log(
        f"script version        : {SCRIPT_VERSION}"
    )

    log(
        f"train dataset         : {train_path}"
    )

    log(
        f"X / y                 : {X.shape} / {y.shape}"
    )

    log(
        f"device                : {device}"
    )

    if device.type == "cuda":
        log(
            f"GPU                   : {torch.cuda.get_device_name(0)}"
        )

    log(
        f"AMP runtime           : {runtime_amp_enabled}"
    )

    log(
        f"physical mask         : {int(physical_mask.sum())}/100"
    )

    log(
        f"search-space SHA256   : {search_digest}"
    )

    log(
        f"active configs        : {[x.config_id for x in active_configs]}"
    )

    log(
        f"active folds          : {active_folds}"
    )

    log(
        f"debug fast            : {args.debug_fast}"
    )

    log(
        "outer validation      : NOT ACCESSED"
    )

    log(
        "outer test            : NOT ACCESSED"
    )

    log(
        "SD1033 external       : NOT ACCESSED"
    )

    log("")

    for row in fold_table_rows:
        log(
            f"Fold {row['fold']}: train={row['train_samples']} | "
            f"purge={row['purge_samples']} | val={row['validation_samples']} | "
            f"seed={row['screening_seed']} | "
            f"firewall={row['timestamp_firewall_pass']}"
        )

        log(
            f"  train max target : {row['train_max_target']}"
        )

        log(
            f"  val first context: {row['validation_first_context_end']}"
        )

    log("")

    fold_results_path = (
        output_dir
        / "fold_results.csv"
    )

    for config in active_configs:
        log(
            "#"
            * 120
        )

        log(
            f"[CONFIG {config.config_id}] "
            f"GCN={config.gcn_hidden_dim} | "
            f"GRU={config.gru_hidden_dim} | "
            f"lambda_KL={config.lambda_kl:g} | "
            f"lambda_NLL={config.lambda_nll:g}"
        )

        log(
            "#"
            * 120
        )

        for fold in folds:
            fold_id = int(
                fold[
                    "fold"
                ]
            )

            if fold_id not in active_folds:
                continue

            existing = load_existing_fold_results(
                fold_results_path
            )

            if completed_result_exists(
                existing,
                config.config_id,
                fold_id,
            ):
                log(
                    f"  [RESUME] {config.config_id}/fold{fold_id} "
                    "already completed_finite; skipping."
                )

                continue

            log(
                f"  [RUN] {config.config_id}/fold{fold_id} "
                f"train={fold['train_count']} val={fold['val_count']} "
                f"seed={SCREENING_FOLD_SEEDS[fold_id-1]}"
            )

            try:
                result = train_fold(
                    torch=torch,
                    nn=nn,
                    F=F,
                    physical_mask=physical_mask,
                    config=config,
                    fold=fold,
                    X=X,
                    y=y,
                    device=device,
                    output_dir=output_dir,
                    debug_fast=args.debug_fast,
                )

            except Exception as exc:
                failure = {
                    "config_id": config.config_id,
                    "fold": fold_id,
                    "screening_seed": int(
                        SCREENING_FOLD_SEEDS[
                            fold_id
                            - 1
                        ]
                    ),
                    "status": "failed",
                    "gcn_hidden_dim": int(
                        config.gcn_hidden_dim
                    ),
                    "gru_hidden_dim": int(
                        config.gru_hidden_dim
                    ),
                    "lambda_KL": float(
                        config.lambda_kl
                    ),
                    "lambda_NLL": float(
                        config.lambda_nll
                    ),
                    "error_type": type(
                        exc
                    ).__name__,
                    "error_message": str(
                        exc
                    ),
                }

                append_fold_result(
                    fold_results_path,
                    failure,
                )

                raise

            append_fold_result(
                fold_results_path,
                result,
            )

            log(
                f"  [DONE] {config.config_id}/fold{fold_id}: "
                f"best epoch={result['best_epoch_one_based']} | "
                f"CV vector RMSE={result['val_mean_vector_RMSE_mps']:.4f} m/s | "
                f"params={result['parameter_count']:,} | "
                f"time={result['elapsed_seconds']:.1f}s"
            )

            gc.collect()

            if device.type == "cuda":
                torch.cuda.empty_cache()

    fold_df = load_existing_fold_results(
        fold_results_path
    )

    summary = summarize_candidates(
        fold_df,
        [
            cfg.config_id
            for cfg in SEARCH_CONFIGS
        ],
    )

    summary_path = (
        output_dir
        / "candidate_summary.csv"
    )

    if not summary.empty:
        summary.to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig",
        )

    expected_pairs = {
        (
            cfg.config_id,
            fold_id,
        )
        for cfg in SEARCH_CONFIGS
        for fold_id in [
            1,
            2,
            3,
        ]
    }

    completed_pairs = set()

    if not fold_df.empty:
        finite_rows = fold_df.loc[
            fold_df[
                "status"
            ]
            == "completed_finite"
        ]

        for row in finite_rows.itertuples():
            completed_pairs.add(
                (
                    str(
                        row.config_id
                    ),
                    int(
                        row.fold
                    ),
                )
            )

    all_complete = bool(
        expected_pairs.issubset(
            completed_pairs
        )
    )

    selected_path = (
        output_dir
        / "selected_config.json"
    )

    selected = None

    if (
        all_complete
        and not args.debug_fast
    ):
        complete_summary = summary.loc[
            summary[
                "complete_three_fold_candidate"
            ]
        ].copy()

        if len(
            complete_summary
        ) != len(
            SEARCH_CONFIGS
        ):
            raise RuntimeError(
                "All fold pairs appear complete but candidate summary is incomplete."
            )

        winner = complete_summary.sort_values(
            [
                "CV_mean_vector_RMSE_mps",
                "CV_std_vector_RMSE_mps",
                "parameter_count",
                "config_id",
            ],
            ascending=[
                True,
                True,
                True,
                True,
            ],
        ).iloc[
            0
        ]

        config_obj = next(
            cfg
            for cfg in SEARCH_CONFIGS
            if cfg.config_id
            == winner[
                "config_id"
            ]
        )

        selected = {
            "stage": "11C2",
            "selected_config_id": config_obj.config_id,
            "selected_config": asdict(
                config_obj
            ),
            "fixed_training_config": asdict(
                FIXED
            ),
            "selection_metric": (
                "lowest mean 3-fold raw wind-vector RMSE across "
                "1/2/3/5/10-min horizons"
            ),
            "CV_mean_vector_RMSE_mps": float(
                winner[
                    "CV_mean_vector_RMSE_mps"
                ]
            ),
            "CV_std_vector_RMSE_mps": float(
                winner[
                    "CV_std_vector_RMSE_mps"
                ]
            ),
            "CV_mean_U_RMSE_mps": float(
                winner[
                    "CV_mean_U_RMSE_mps"
                ]
            ),
            "CV_mean_V_RMSE_mps": float(
                winner[
                    "CV_mean_V_RMSE_mps"
                ]
            ),
            "parameter_count": int(
                winner[
                    "parameter_count"
                ]
            ),
            "paper_reported_parameter_count": 242580,
            "search_space_sha256": search_digest,
            "outer_validation_used_for_HPO": False,
            "outer_test_used_for_HPO": False,
            "external_data_used_for_HPO": False,
            "next_stage": (
                "11C3 five-seed full-outer-train refit with outer-validation "
                "checkpointing; no test/external access"
            ),
        }

        save_json(
            selected_path,
            selected,
        )

    elif selected_path.exists():
        # Never leave a stale scientific selection after an incomplete/debug run.
        selected_path.unlink()

    manifest = {
        "script_version": SCRIPT_VERSION,
        "stage": "11C2",
        "track": "F_common_task",
        "scientific_run": bool(
            not args.debug_fast
        ),
        "data_firewall": {
            "SD1090_outer_train_accessed": True,
            "SD1090_outer_validation_accessed": False,
            "SD1090_outer_test_accessed": False,
            "SD1033_external_accessed": False,
        },
        "search_space_sha256": search_digest,
        "predeclared_config_count": len(
            SEARCH_CONFIGS
        ),
        "expected_fold_runs": len(
            SEARCH_CONFIGS
        )
        * 3,
        "completed_finite_fold_runs": int(
            len(
                completed_pairs
                & expected_pairs
            )
        ),
        "all_predeclared_runs_complete": all_complete,
        "selected_config_written": bool(
            selected is not None
        ),
        "device": str(
            device
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "runtime_amp_enabled": runtime_amp_enabled,
        "python": sys.version,
        "platform": platform.platform(),
        "proposed_model_modified": False,
    }

    save_json(
        output_dir
        / "training_manifest.json",
        manifest,
    )

    with (
        output_dir
        / "training_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "11C2 BP-STGNN-F Train-Only Rolling-CV Tuning\n"
        )

        f.write(
            "="
            * 120
            + "\n\n"
        )

        f.write(
            "DATA FIREWALL\n"
        )

        f.write(
            "-"
            * 120
            + "\n"
        )

        f.write(
            "SD1090 outer TRAIN accessed: YES\n"
        )

        f.write(
            "SD1090 outer VALIDATION accessed: NO\n"
        )

        f.write(
            "SD1090 outer TEST accessed: NO\n"
        )

        f.write(
            "SD1033 external accessed: NO\n\n"
        )

        f.write(
            f"Search-space SHA256: {search_digest}\n\n"
        )

        f.write(
            "ROLLING FOLDS\n"
        )

        f.write(
            "-"
            * 120
            + "\n"
        )

        f.write(
            pd.DataFrame(
                fold_table_rows
            ).to_string(
                index=False
            )
        )

        f.write(
            "\n\nCANDIDATE SUMMARY\n"
        )

        f.write(
            "-"
            * 120
            + "\n"
        )

        if summary.empty:
            f.write(
                "No completed candidate summaries yet.\n"
            )

        else:
            f.write(
                summary.to_string(
                    index=False
                )
            )

        f.write(
            "\n\nSELECTION\n"
        )

        f.write(
            "-"
            * 120
            + "\n"
        )

        if selected is None:
            f.write(
                "No scientific selection written because the full "
                "predeclared 10-config x 3-fold matrix is not yet complete "
                "or this was a debug run.\n"
            )

        else:
            f.write(
                json.dumps(
                    selected,
                    ensure_ascii=False,
                    indent=2,
                )
            )

            f.write(
                "\n"
            )

    log("")
    log(
        "="
        * 120
    )

    log(
        "11C2 BP-STGNN-F ROLLING-CV RESULTS"
    )

    log(
        "="
        * 120
    )

    if summary.empty:
        log(
            "No completed candidate summary yet."
        )

    else:
        display = summary.loc[
            summary[
                "completed_folds"
            ]
            > 0
        ]

        for row in display.itertuples():
            log(
                f"{row.config_id}: "
                f"CV wind vector RMSE="
                f"{row.CV_mean_vector_RMSE_mps:.4f} "
                f"+/- {row.CV_std_vector_RMSE_mps:.4f} m/s | "
                f"folds={row.completed_folds}/3 | "
                f"params={int(row.parameter_count):,}"
            )

    log("")

    if selected is not None:
        log(
            f"[SELECTED] {selected['selected_config_id']} | "
            f"CV={selected['CV_mean_vector_RMSE_mps']:.4f} "
            f"+/- {selected['CV_std_vector_RMSE_mps']:.4f} m/s | "
            f"params={selected['parameter_count']:,}"
        )

    else:
        missing = sorted(
            expected_pairs
            - completed_pairs
        )

        log(
            f"[INCOMPLETE] scientific selection not written. "
            f"Missing predeclared fold runs: {len(missing)}"
        )

        if (
            len(
                missing
            )
            <= 20
        ):
            log(
                f"  missing={missing}"
            )

    log(
        "[POLICY] Outer validation/test and all SD1033 external data were not accessed."
    )

    log(
        "[POLICY] Physics-Compact-Vessel-Residual v1.0 was not modified."
    )

    log(
        f"[DONE] 11C2 outputs: {output_dir}"
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
