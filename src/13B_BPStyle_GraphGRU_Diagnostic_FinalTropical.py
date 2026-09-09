# -*- coding: utf-8 -*-
r"""
13B_BPStyle_GraphGRU_Diagnostic_FinalTropical.py

Stage 13B
=========
Method-faithful BP-STGNN architecture diagnostic on the EXACT SAME
Stage-12G point-sampled Track-P samples.

QUESTION
--------
Stage 13A showed that simply adding BP-owned auxiliary variables to Ridge does
NOT explain the large Tropical V gap:

    Base4-Ridge V ~ 0.792
    published BP-STGNN V = 0.70506

Therefore Stage 13B asks:

    Can a BP-style graph -> GRU architecture extract materially stronger
    cross-variable / temporal information on OUR exact 12G samples?

If the answer is NO, the remaining discrepancy is unlikely to be explained by
missing inputs alone and instead points toward protocol/preprocessing/split
differences and/or unspecified implementation details in Nie et al.

IMPORTANT SCIENTIFIC LABEL
--------------------------
This script is a METHOD-FAITHFUL / TASK-ADAPTED diagnostic.

It is NOT an exact reproduction of Nie et al. because the paper does not
uniquely specify:
    GCN hidden dimensions
    GRU hidden dimensions/layers
    graph normalization
    optimizer/lr/batch size
    KL/NLL weights
    exact train/validation split
    seeds
    exact resampling implementation

PAPER-FAITHFUL ELEMENTS USED
----------------------------
Track-P ten graph nodes:
    0 U
    1 V
    2 SOG
    3 COG
    4 T
    5 RH
    6 P
    7 dT
    8 dP
    9 dRH

Physical groups:
    target T = {U,V}
    source S = {dP,dT,dRH}
    kinematic K = {SOG,COG}
    environmental E = {P,T,RH}

Physical prior/mask rule:
    (S union K) -> target
    E -> S
    E -> E
    self-loops

This yields exactly 35 / 100 directed entries.

Architecture:
    node scalar embedding
    graph aggregation / graph convolution
    temporal GRU
    output mean U,V

Bayesian variant additionally uses:
    q(A_ij) = N(mu_ij, sigma_ij^2)
    physics-centered prior N(M_ij, 1)
    reparameterization during training
    heteroscedastic mean/log-variance head
    phase 1 first 30 epochs: MSE + KL, variance head frozen
    phase 2: MSE + NLL + KL
    deterministic posterior-mean adjacency for validation/test

FROZEN LOCAL IMPLEMENTATION CHOICES
-----------------------------------
These are Stage-13B diagnostic choices, NOT claims about the original paper:

    node_embed_dim = 16
    graph_hidden_dim = 32
    graph_layers = 2
    graph_to_gru_dim = 96
    gru_hidden_dim = 96
    gru_layers = 1
    head_hidden_dim = 64
    dropout = 0.10

    AdamW lr = 8e-4
    weight_decay = 1e-5
    batch = 256
    max_epochs = 120
    early_stop patience = 18

    Bayesian:
        initial adjacency sigma ~ 0.10
        lambda_KL = 1e-4 after normalization by 100 adjacency entries
        lambda_NLL = 0.10
        phase2 begins after epoch 30

CANDIDATES
----------
1) GRU10
    Full ten-node sequence, flattened as 10 variables per time step.
    No graph.

2) FixedPhysicsGCNGRU
    Fixed 35/100 physics mask -> 2 graph conv layers -> GRU.

3) BayesianPhysicsGCNGRU
    Bayesian adjacency centered on the physical mask -> graph conv -> GRU ->
    heteroscedastic output.

All candidates use exactly the same ten variables and same six time points.

STRICT TEMPORAL PROTOCOL
------------------------
Stage 12G Track-P:
    6 point samples at 10-min intervals
    -> next +10-min point U,V

No:
    arithmetic 10-min averaging
    intermediate 1-min samples
    interpolation
    wing angle
    position

DEVELOPMENT / FIREWALL
----------------------
Development only:
    Antarctic
    Atlantic
    West Coast

LOMO:
    hold out each mission once.

Seeds:
    500043
    501052
    502061

For every candidate:
    select seed by mean LOMO score
    final epoch = median best epoch across three holdouts

Then save:
    FROZEN_BEFORE_TROPICAL.json

Only after that:
    load Tropical_Atlantic_TEST.npz
    refit each candidate on all three development missions
    evaluate each exactly once.

REFERENCE MODELS
----------------
The final Tropical table also reports:
    Base4-Ridge
    Full10-Ridge
    BP-STGNN-published

DIAGNOSTIC INTERPRETATION
-------------------------
This script does NOT automatically assert why results differ.

Useful interpretation:

A) If BP-style graph-GRU materially lowers Tropical V toward ~0.705:
       architecture/nonlinear graph interactions are likely important.
       Next step: distill that signal into a compact Ridge-residual student.

B) If all local BP-style models remain near Ridge (~0.78-0.80 V):
       the published-vs-local discrepancy is unlikely to be explained by
       auxiliary inputs or graph architecture alone.
       Focus next on protocol/preprocessing/sample-selection reproduction.

PUBLISHED REFERENCE
-------------------
BP-STGNN Tropical Atlantic:
    U RMSE  = 0.748640 m/s
    V RMSE  = 0.705060 m/s
    WS RMSE = 0.699400 m/s
    WD RMSE = 5.584 deg
    params  = 242580

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\13B_BPStyle_GraphGRU_Diagnostic_FinalTropical.py" --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\13B_BPStyle_GraphGRU_Diagnostic_v0_1"

Smoke:
    add --debug-fast
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import random
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import Ridge
except Exception as exc:
    raise RuntimeError(
        "scikit-learn is required. Activate the WindPredict environment."
    ) from exc


SCRIPT_VERSION = "0.1.0-13B-BPStyle-GraphGRU-Diagnostic"

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
    / "13B_BPStyle_GraphGRU_Diagnostic_v0_1"
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

NODE_NAMES = list(
    TRACK_P_FEATURES
)

N_NODES = len(
    NODE_NAMES
)

IDX = {
    name: i
    for i, name in enumerate(
        NODE_NAMES
    )
}

BASE4_NAMES = [
    "U",
    "V",
    "T",
    "RH",
]

SEEDS = [
    500043,
    501052,
    502061,
]

RIDGE_ALPHA = 1.0
EPS = 1e-12

BP_REFERENCE = {
    "U_RMSE_mps": 0.748640,
    "V_RMSE_mps": 0.705060,
    "WS_RMSE_mps": 0.699400,
    "WD_RMSE_deg": 5.584000,
    "params": 242580,
}


@dataclass(frozen=True)
class Freeze:
    node_embed_dim: int = 16
    graph_hidden_dim: int = 32
    graph_layers: int = 2
    graph_to_gru_dim: int = 96

    gru_hidden_dim: int = 96
    gru_layers: int = 1
    head_hidden_dim: int = 64
    dropout: float = 0.10

    learning_rate: float = 8e-4
    weight_decay: float = 1e-5
    batch_size: int = 256

    max_epochs: int = 120
    early_stop_patience: int = 18
    scheduler_patience: int = 6
    scheduler_factor: float = 0.5
    min_lr: float = 1e-5
    grad_clip_norm: float = 1.0

    bayes_initial_sigma: float = 0.10
    lambda_kl: float = 1e-4
    lambda_nll: float = 0.10
    phase1_epochs: int = 30

    use_amp: bool = True


FREEZE = Freeze()

CANDIDATES = [
    "GRU10",
    "FixedPhysicsGCNGRU",
    "BayesianPhysicsGCNGRU",
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


def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required. Activate the WindPredict environment."
        ) from exc

    return torch, nn, F


def configure_cuda(torch):
    if not torch.cuda.is_available():
        return

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


def seed_everything(
    torch,
    seed,
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


def amp_context(
    torch,
    enabled,
    device,
):
    if (
        not enabled
        or device.type
        != "cuda"
    ):
        return contextlib.nullcontext()

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


def create_grad_scaler(
    torch,
    enabled,
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


def state_dict_cpu(
    model,
):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def count_parameters(
    model,
):
    return int(
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )


# =============================================================================
# Physical mask
# =============================================================================

def build_physical_mask():
    M = np.zeros(
        (
            N_NODES,
            N_NODES,
        ),
        dtype=np.float32,
    )

    target = [
        IDX[
            "U"
        ],
        IDX[
            "V"
        ],
    ]

    source = [
        IDX[
            "dP"
        ],
        IDX[
            "dT"
        ],
        IDX[
            "dRH"
        ],
    ]

    kinematic = [
        IDX[
            "SOG"
        ],
        IDX[
            "COG"
        ],
    ]

    environmental = [
        IDX[
            "P"
        ],
        IDX[
            "T"
        ],
        IDX[
            "RH"
        ],
    ]

    # src -> dst
    for src in (
        source
        + kinematic
    ):
        for dst in target:
            M[
                src,
                dst
            ] = 1.0

    for src in environmental:
        for dst in source:
            M[
                src,
                dst
            ] = 1.0

    for src in environmental:
        for dst in environmental:
            M[
                src,
                dst
            ] = 1.0

    for i in range(
        N_NODES
    ):
        M[
            i,
            i
        ] = 1.0

    if int(
        np.sum(
            M
        )
    ) != 35:
        raise RuntimeError(
            f"Physical mask must contain 35 ones; got {int(np.sum(M))}."
        )

    return M


PHYSICAL_MASK = build_physical_mask()


# =============================================================================
# Dataset
# =============================================================================

def resolve_dataset_root(
    dataset_dir: Path,
):
    for root in [
        dataset_dir,
        dataset_dir
        / "dataset",
    ]:
        if (
            root
            / "track_P_paper_informed"
        ).exists():
            return root

    raise FileNotFoundError(
        "track_P_paper_informed not found."
    )


def mission_path(
    dataset_root: Path,
    mission: str,
):
    track_dir = (
        dataset_root
        / "track_P_paper_informed"
    )

    safe = mission.replace(
        " ",
        "_",
    )

    if mission == FINAL_MISSION:
        candidates = [
            track_dir
            / f"{safe}_TEST.npz",
            track_dir
            / f"{safe}.npz",
        ]
    else:
        candidates = [
            track_dir
            / f"{safe}.npz",
        ]

    for path in candidates:
        if path.exists():
            return path

    attempted = "\n".join(
        f"  - {p}"
        for p in candidates
    )

    raise FileNotFoundError(
        f"Missing mission {mission}. Tried:\n{attempted}"
    )


def load_mission(
    dataset_root: Path,
    mission: str,
):
    path = mission_path(
        dataset_root,
        mission,
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as z:
        X = np.asarray(
            z[
                "X_raw"
            ],
            dtype=np.float32,
        )

        y = np.asarray(
            z[
                "y_wind_raw"
            ],
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
            for x in z[
                "feature_names"
            ].tolist()
        ]

    if names != TRACK_P_FEATURES:
        raise RuntimeError(
            f"{mission}: Track-P feature mismatch: {names}"
        )

    if (
        X.ndim
        != 3
        or X.shape[
            1:
        ]
        != (
            LOOKBACK_STEPS,
            N_NODES,
        )
    ):
        raise RuntimeError(
            f"{mission}: unexpected X shape {X.shape}"
        )

    if y.ndim == 3:
        y = y[
            :,
            0,
            :
        ]

    if (
        y.ndim
        != 2
        or y.shape[
            1
        ]
        != 2
    ):
        raise RuntimeError(
            f"{mission}: unexpected y shape {y.shape}"
        )

    if not np.all(
        target
        - context
        == TEN_MIN_NS
    ):
        raise RuntimeError(
            f"{mission}: target is not +10 min."
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
            f"{mission}: nonfinite values."
        )

    return {
        "X": X,
        "y": y,
        "mission": mission,
        "path": path,
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
    }


# =============================================================================
# Scaling
# =============================================================================

def fit_scaler(
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


# =============================================================================
# Ridge references
# =============================================================================

def fit_ridge(
    X,
    y,
):
    scaler = fit_scaler(
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
            len(
                X
            ),
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


def predict_ridge(
    model,
    scaler,
    X,
):
    yz = np.asarray(
        model.predict(
            transform_X(
                X,
                scaler,
            ).reshape(
                len(
                    X
                ),
                -1,
            )
        ),
        dtype=np.float32,
    )

    return inverse_y(
        yz,
        scaler,
    )


def base4_from_trackP(
    X,
):
    indices = [
        IDX[
            "U"
        ],
        IDX[
            "V"
        ],
        IDX[
            "T"
        ],
        IDX[
            "RH"
        ],
    ]

    return np.asarray(
        X[
            :,
            :,
            indices
        ],
        dtype=np.float32,
    )


def ridge_param_count(
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


def selection_score(
    metrics,
    base_metrics,
):
    return float(
        0.35
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
        + 0.45
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


# =============================================================================
# Graph normalization
# =============================================================================

def normalize_adjacency_torch(
    torch,
    A,
):
    # Signed adjacency is preserved.
    # Absolute weights are used only for stable degree normalization.
    absA = torch.abs(
        A
    )

    degree = torch.sum(
        absA,
        dim=1,
    ).clamp_min(
        1e-6
    )

    invsqrt = torch.rsqrt(
        degree
    )

    return (
        invsqrt[
            :,
            None
        ]
        * A
        * invsqrt[
            None,
            :
        ]
    )


# =============================================================================
# Models
# =============================================================================

def build_model_class(
    torch,
    nn,
    F,
    candidate,
):
    M_np = PHYSICAL_MASK.copy()

    class GRU10(nn.Module):
        def __init__(self):
            super().__init__()

            self.gru = nn.GRU(
                input_size=N_NODES,
                hidden_size=(
                    FREEZE.gru_hidden_dim
                ),
                num_layers=(
                    FREEZE.gru_layers
                ),
                batch_first=True,
            )

            self.head = nn.Sequential(
                nn.Linear(
                    FREEZE.gru_hidden_dim,
                    FREEZE.head_hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(
                    FREEZE.dropout
                ),
                nn.Linear(
                    FREEZE.head_hidden_dim,
                    2,
                ),
            )

        def forward(
            self,
            x,
            sample_graph=True,
        ):
            z, _ = self.gru(
                x
            )

            mu = self.head(
                z[
                    :,
                    -1,
                    :
                ]
            )

            return {
                "mu": mu,
                "logvar": None,
                "kl": mu.new_tensor(
                    0.0
                ),
            }

    class GraphBackbone(nn.Module):
        def __init__(
            self,
            bayesian,
        ):
            super().__init__()

            self.bayesian = bool(
                bayesian
            )

            self.node_embed = nn.Linear(
                1,
                FREEZE.node_embed_dim,
            )

            self.gcn1 = nn.Linear(
                FREEZE.node_embed_dim,
                FREEZE.graph_hidden_dim,
            )

            self.gcn2 = nn.Linear(
                FREEZE.graph_hidden_dim,
                FREEZE.graph_hidden_dim,
            )

            self.project = nn.Linear(
                N_NODES
                * FREEZE.graph_hidden_dim,
                FREEZE.graph_to_gru_dim,
            )

            self.gru = nn.GRU(
                input_size=(
                    FREEZE.graph_to_gru_dim
                ),
                hidden_size=(
                    FREEZE.gru_hidden_dim
                ),
                num_layers=(
                    FREEZE.gru_layers
                ),
                batch_first=True,
            )

            self.shared_head = nn.Sequential(
                nn.Linear(
                    FREEZE.gru_hidden_dim,
                    FREEZE.head_hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(
                    FREEZE.dropout
                ),
            )

            self.mean_head = nn.Linear(
                FREEZE.head_hidden_dim,
                2,
            )

            if self.bayesian:
                self.logvar_head = nn.Linear(
                    FREEZE.head_hidden_dim,
                    2,
                )

                self.mu_adj = nn.Parameter(
                    torch.from_numpy(
                        M_np
                    ).clone()
                )

                sigma0 = float(
                    FREEZE.bayes_initial_sigma
                )

                rho0 = math.log(
                    math.expm1(
                        sigma0
                    )
                )

                self.rho_adj = nn.Parameter(
                    torch.full(
                        (
                            N_NODES,
                            N_NODES,
                        ),
                        float(
                            rho0
                        ),
                        dtype=torch.float32,
                    )
                )
            else:
                self.register_buffer(
                    "fixed_adj",
                    torch.from_numpy(
                        M_np
                    ),
                )

        def adjacency_and_kl(
            self,
            sample_graph,
        ):
            if not self.bayesian:
                A = self.fixed_adj

                return (
                    normalize_adjacency_torch(
                        torch,
                        A,
                    ),
                    A.new_tensor(
                        0.0
                    ),
                )

            sigma = (
                F.softplus(
                    self.rho_adj
                )
                + 1e-6
            )

            if sample_graph:
                eps = torch.randn_like(
                    self.mu_adj
                )

                A = (
                    self.mu_adj
                    + sigma
                    * eps
                )
            else:
                A = self.mu_adj

            prior_mean = torch.as_tensor(
                M_np,
                dtype=A.dtype,
                device=A.device,
            )

            # KL[q=N(mu,sigma^2) || p=N(M,1)].
            kl_each = 0.5 * (
                sigma ** 2
                + (
                    self.mu_adj
                    - prior_mean
                ) ** 2
                - 1.0
                - torch.log(
                    sigma ** 2
                    + 1e-12
                )
            )

            kl = torch.mean(
                kl_each
            )

            A_norm = normalize_adjacency_torch(
                torch,
                A,
            )

            return (
                A_norm,
                kl,
            )

        def graph_encode(
            self,
            x,
            A_norm,
        ):
            # x: [B,T,N]
            h = self.node_embed(
                x[
                    :,
                    :,
                    :,
                    None
                ]
            )

            # A[src,dst]; aggregate source into destination.
            h = torch.einsum(
                "ij,btif->btjf",
                A_norm,
                h,
            )

            h = F.gelu(
                self.gcn1(
                    h
                )
            )

            h = torch.einsum(
                "ij,btif->btjf",
                A_norm,
                h,
            )

            h = F.gelu(
                self.gcn2(
                    h
                )
            )

            h = h.reshape(
                h.shape[
                    0
                ],
                h.shape[
                    1
                ],
                -1,
            )

            h = F.gelu(
                self.project(
                    h
                )
            )

            return h

        def forward(
            self,
            x,
            sample_graph=True,
        ):
            A_norm, kl = (
                self.adjacency_and_kl(
                    sample_graph
                )
            )

            h = self.graph_encode(
                x,
                A_norm,
            )

            z, _ = self.gru(
                h
            )

            shared = self.shared_head(
                z[
                    :,
                    -1,
                    :
                ]
            )

            mu = self.mean_head(
                shared
            )

            if self.bayesian:
                # Decouple variance head from shared feature gradient.
                logvar = self.logvar_head(
                    shared.detach()
                )

                logvar = torch.clamp(
                    logvar,
                    -8.0,
                    4.0,
                )
            else:
                logvar = None

            return {
                "mu": mu,
                "logvar": logvar,
                "kl": kl,
            }

    if candidate == "GRU10":
        return GRU10

    if candidate == "FixedPhysicsGCNGRU":
        class FixedModel(
            GraphBackbone
        ):
            def __init__(self):
                super().__init__(
                    bayesian=False
                )

        return FixedModel

    if candidate == "BayesianPhysicsGCNGRU":
        class BayesianModel(
            GraphBackbone
        ):
            def __init__(self):
                super().__init__(
                    bayesian=True
                )

        return BayesianModel

    raise ValueError(
        candidate
    )


# =============================================================================
# Loss / prediction
# =============================================================================

def train_loss(
    torch,
    outputs,
    y_z,
    candidate,
    epoch,
):
    mu = outputs[
        "mu"
    ]

    mse = torch.mean(
        (
            mu
            - y_z
        ) ** 2
    )

    if candidate != "BayesianPhysicsGCNGRU":
        return mse

    kl = outputs[
        "kl"
    ]

    if epoch <= FREEZE.phase1_epochs:
        return (
            mse
            + FREEZE.lambda_kl
            * kl
        )

    logvar = outputs[
        "logvar"
    ]

    nll = 0.5 * torch.mean(
        logvar
        + (
            y_z
            - mu
        ) ** 2
        * torch.exp(
            -logvar
        )
    )

    return (
        mse
        + FREEZE.lambda_nll
        * nll
        + FREEZE.lambda_kl
        * kl
    )


def set_variance_trainable(
    model,
    candidate,
    trainable,
):
    if candidate != "BayesianPhysicsGCNGRU":
        return

    for p in model.logvar_head.parameters():
        p.requires_grad = bool(
            trainable
        )


def predict_model(
    *,
    torch,
    model,
    candidate,
    Xz,
    scaler,
    device,
    amp_enabled,
):
    model.eval()

    chunks = []

    with torch.no_grad():
        for start in range(
            0,
            len(
                Xz
            ),
            FREEZE.batch_size,
        ):
            stop = min(
                start
                + FREEZE.batch_size,
                len(
                    Xz
                ),
            )

            x_t = torch.from_numpy(
                Xz[
                    start:stop
                ]
            ).to(
                device,
                non_blocking=True,
            )

            with amp_context(
                torch,
                amp_enabled,
                device,
            ):
                out = model(
                    x_t,
                    sample_graph=False,
                )

            chunks.append(
                out[
                    "mu"
                ].detach()
                .float()
                .cpu()
                .numpy()
            )

    pred_z = np.concatenate(
        chunks,
        axis=0,
    )

    return inverse_y(
        pred_z,
        scaler,
    )


# =============================================================================
# Train one fold
# =============================================================================

def train_one_fold(
    *,
    torch,
    nn,
    F,
    candidate,
    seed,
    train_data,
    val_data,
    held_out,
    output_dir,
    device,
    debug_fast,
):
    scaler = fit_scaler(
        train_data[
            "X"
        ],
        train_data[
            "y"
        ],
    )

    Xtr = transform_X(
        train_data[
            "X"
        ],
        scaler,
    )

    ytr = transform_y(
        train_data[
            "y"
        ],
        scaler,
    )

    Xva = transform_X(
        val_data[
            "X"
        ],
        scaler,
    )

    # Base reference for score = Full10-Ridge on same fold.
    ridge_model, ridge_scaler = fit_ridge(
        train_data[
            "X"
        ],
        train_data[
            "y"
        ],
    )

    ridge_val = predict_ridge(
        ridge_model,
        ridge_scaler,
        val_data[
            "X"
        ],
    )

    base_metrics = evaluate_wind(
        val_data[
            "y"
        ],
        ridge_val,
    )

    seed_everything(
        torch,
        seed,
    )

    ModelClass = build_model_class(
        torch,
        nn,
        F,
        candidate,
    )

    model = ModelClass().to(
        device
    )

    n_params = count_parameters(
        model
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FREEZE.learning_rate,
        weight_decay=(
            FREEZE.weight_decay
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=(
                FREEZE.scheduler_factor
            ),
            patience=(
                FREEZE.scheduler_patience
            ),
            min_lr=(
                FREEZE.min_lr
            ),
        )
    )

    amp_enabled = bool(
        FREEZE.use_amp
        and device.type
        == "cuda"
    )

    grad_scaler = create_grad_scaler(
        torch,
        amp_enabled,
    )

    best_score = float(
        "inf"
    )

    best_epoch = 0

    best_state = state_dict_cpu(
        model
    )

    patience = 0

    max_epochs = (
        8
        if debug_fast
        else FREEZE.max_epochs
    )

    patience_limit = (
        4
        if debug_fast
        else FREEZE.early_stop_patience
    )

    history = []

    n = len(
        Xtr
    )

    for epoch in range(
        1,
        max_epochs
        + 1,
    ):
        if (
            candidate
            == "BayesianPhysicsGCNGRU"
        ):
            set_variance_trainable(
                model,
                candidate,
                epoch
                > FREEZE.phase1_epochs,
            )

        model.train()

        loss_sum = 0.0
        n_seen = 0

        for start in range(
            0,
            n,
            FREEZE.batch_size,
        ):
            stop = min(
                start
                + FREEZE.batch_size,
                n,
            )

            x_t = torch.from_numpy(
                Xtr[
                    start:stop
                ]
            ).to(
                device,
                non_blocking=True,
            )

            y_t = torch.from_numpy(
                ytr[
                    start:stop
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
                amp_enabled,
                device,
            ):
                outputs = model(
                    x_t,
                    sample_graph=True,
                )

                loss = train_loss(
                    torch,
                    outputs,
                    y_t,
                    candidate,
                    epoch,
                )

            if not torch.isfinite(
                loss
            ):
                raise FloatingPointError(
                    f"Nonfinite loss: {candidate}"
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
                    FREEZE.grad_clip_norm,
                )

                grad_scaler.step(
                    optimizer
                )

                grad_scaler.update()
            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    FREEZE.grad_clip_norm,
                )

                optimizer.step()

            bn = stop - start

            loss_sum += float(
                loss.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            n_seen += bn

        pred_val = predict_model(
            torch=torch,
            model=model,
            candidate=candidate,
            Xz=Xva,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
        )

        metrics = evaluate_wind(
            val_data[
                "y"
            ],
            pred_val,
        )

        score = selection_score(
            metrics,
            base_metrics,
        )

        improved = (
            score
            < best_score
            - 1e-8
        )

        if improved:
            best_score = float(
                score
            )

            best_epoch = int(
                epoch
            )

            best_state = state_dict_cpu(
                model
            )

            patience = 0
        else:
            patience += 1

        scheduler.step(
            score
        )

        history.append(
            {
                "epoch": int(
                    epoch
                ),
                "train_loss": (
                    loss_sum
                    / max(
                        n_seen,
                        1,
                    )
                ),
                "selection_score": float(
                    score
                ),
                **metrics,
            }
        )

        if (
            epoch <= 3
            or epoch % 10
            == 0
            or improved
        ):
            log(
                f"      ep={epoch:03d} | "
                f"score={score:.5f} | "
                f"U={metrics['wind_U_RMSE_mps']:.4f} | "
                f"V={metrics['wind_V_RMSE_mps']:.4f} | "
                f"WS={metrics['wind_speed_RMSE_mps']:.4f} | "
                f"WD={metrics['wind_direction_RMSE_deg']:.3f}"
                + (
                    " *"
                    if improved
                    else ""
                )
            )

        if patience >= patience_limit:
            break

    model.load_state_dict(
        best_state,
        strict=True,
    )

    pred_val = predict_model(
        torch=torch,
        model=model,
        candidate=candidate,
        Xz=Xva,
        scaler=scaler,
        device=device,
        amp_enabled=amp_enabled,
    )

    metrics = evaluate_wind(
        val_data[
            "y"
        ],
        pred_val,
    )

    history_dir = (
        output_dir
        / "histories"
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints_dev"
    )

    history_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tag = (
        f"{candidate}"
        f"__holdout_{held_out.replace(' ', '_')}"
        f"__seed{seed}"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_dir
        / f"{tag}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    torch.save(
        {
            "stage": "13B-development",
            "candidate": candidate,
            "held_out": held_out,
            "seed": int(
                seed
            ),
            "best_epoch": int(
                best_epoch
            ),
            "parameter_count": int(
                n_params
            ),
            "state_dict": (
                best_state
            ),
            "freeze": asdict(
                FREEZE
            ),
        },
        checkpoint_dir
        / f"{tag}.pt",
    )

    del (
        model,
        optimizer,
        scheduler,
        grad_scaler,
    )

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "candidate": candidate,
        "held_out_mission": (
            held_out
        ),
        "seed": int(
            seed
        ),
        "best_epoch": int(
            best_epoch
        ),
        "parameter_count": int(
            n_params
        ),
        "selection_score": float(
            selection_score(
                metrics,
                base_metrics,
            )
        ),
        **metrics,
    }


# =============================================================================
# Final fixed-epoch training
# =============================================================================

def train_final_candidate(
    *,
    torch,
    nn,
    F,
    candidate,
    seed,
    epochs,
    dev_all,
    tropical,
    device,
):
    scaler = fit_scaler(
        dev_all[
            "X"
        ],
        dev_all[
            "y"
        ],
    )

    Xtr = transform_X(
        dev_all[
            "X"
        ],
        scaler,
    )

    ytr = transform_y(
        dev_all[
            "y"
        ],
        scaler,
    )

    Xte = transform_X(
        tropical[
            "X"
        ],
        scaler,
    )

    seed_everything(
        torch,
        seed,
    )

    ModelClass = build_model_class(
        torch,
        nn,
        F,
        candidate,
    )

    model = ModelClass().to(
        device
    )

    n_params = count_parameters(
        model
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FREEZE.learning_rate,
        weight_decay=(
            FREEZE.weight_decay
        ),
    )

    amp_enabled = bool(
        FREEZE.use_amp
        and device.type
        == "cuda"
    )

    grad_scaler = create_grad_scaler(
        torch,
        amp_enabled,
    )

    n = len(
        Xtr
    )

    for epoch in range(
        1,
        int(
            epochs
        )
        + 1,
    ):
        if (
            candidate
            == "BayesianPhysicsGCNGRU"
        ):
            set_variance_trainable(
                model,
                candidate,
                epoch
                > FREEZE.phase1_epochs,
            )

        model.train()

        for start in range(
            0,
            n,
            FREEZE.batch_size,
        ):
            stop = min(
                start
                + FREEZE.batch_size,
                n,
            )

            x_t = torch.from_numpy(
                Xtr[
                    start:stop
                ]
            ).to(
                device,
                non_blocking=True,
            )

            y_t = torch.from_numpy(
                ytr[
                    start:stop
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
                amp_enabled,
                device,
            ):
                outputs = model(
                    x_t,
                    sample_graph=True,
                )

                loss = train_loss(
                    torch,
                    outputs,
                    y_t,
                    candidate,
                    epoch,
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
                    FREEZE.grad_clip_norm,
                )

                grad_scaler.step(
                    optimizer
                )

                grad_scaler.update()
            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    FREEZE.grad_clip_norm,
                )

                optimizer.step()

    pred = predict_model(
        torch=torch,
        model=model,
        candidate=candidate,
        Xz=Xte,
        scaler=scaler,
        device=device,
        amp_enabled=amp_enabled,
    )

    metrics = evaluate_wind(
        tropical[
            "y"
        ],
        pred,
    )

    return {
        "model": candidate,
        "parameter_count": int(
            n_params
        ),
        **metrics,
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
    )

    parser.add_argument(
        "--debug-fast",
        action="store_true",
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

    torch, nn, F = import_torch()

    configure_cuda(
        torch
    )

    device = torch.device(
        "cpu"
        if (
            args.force_cpu
            or not torch.cuda.is_available()
        )
        else "cuda"
    )

    log("=" * 132)
    log(
        "13B — BP-STYLE GRAPH→GRU DIAGNOSTIC ON EXACT STAGE-12G SAMPLES"
    )
    log("=" * 132)
    log(
        f"dataset : {dataset_root}"
    )
    log(
        f"output  : {output_dir}"
    )
    log(
        f"physical mask entries: {int(np.sum(PHYSICAL_MASK))}/100"
    )
    log(
        f"device  : {device}"
    )

    if device.type == "cuda":
        log(
            f"GPU     : {torch.cuda.get_device_name(0)}"
        )

    # ---------------------------------------------------------
    # Development only
    # ---------------------------------------------------------
    dev_missions = {
        m: load_mission(
            dataset_root,
            m,
        )
        for m in DEV_MISSIONS
    }

    log(
        "[FIREWALL] Tropical Atlantic NOT loaded."
    )

    active_candidates = (
        [
            "GRU10",
            "FixedPhysicsGCNGRU",
        ]
        if args.debug_fast
        else CANDIDATES
    )

    active_holdouts = (
        [
            "Antarctic"
        ]
        if args.debug_fast
        else DEV_MISSIONS
    )

    active_seeds = (
        [
            SEEDS[
                0
            ]
        ]
        if args.debug_fast
        else SEEDS
    )

    rows = []

    for held_out in active_holdouts:
        train_data = concat_missions(
            [
                dev_missions[
                    m
                ]
                for m in DEV_MISSIONS
                if m != held_out
            ]
        )

        val_data = dev_missions[
            held_out
        ]

        log(
            "\n"
            + "-" * 132
        )

        log(
            f"[HOLDOUT] {held_out} | "
            f"Ntrain={len(train_data['y']):,} | "
            f"Nval={len(val_data['y']):,}"
        )

        for candidate in active_candidates:
            for seed in active_seeds:
                log(
                    f"  [TRAIN] {candidate} | seed={seed}"
                )

                row = train_one_fold(
                    torch=torch,
                    nn=nn,
                    F=F,
                    candidate=candidate,
                    seed=seed,
                    train_data=train_data,
                    val_data=val_data,
                    held_out=held_out,
                    output_dir=output_dir,
                    device=device,
                    debug_fast=(
                        args.debug_fast
                    ),
                )

                rows.append(
                    row
                )

                log(
                    f"  [DONE] score={row['selection_score']:.5f} | "
                    f"U={row['wind_U_RMSE_mps']:.4f} | "
                    f"V={row['wind_V_RMSE_mps']:.4f} | "
                    f"bestEp={row['best_epoch']}"
                )

    runs = pd.DataFrame(
        rows
    )

    runs_path = (
        output_dir
        / "development_runs.csv"
    )

    runs.to_csv(
        runs_path,
        index=False,
        encoding="utf-8-sig",
    )

    if args.debug_fast:
        log(
            "\n[DONE] debug-fast completed. Tropical Atlantic was NOT loaded."
        )
        return 0

    # ---------------------------------------------------------
    # Development freeze
    # ---------------------------------------------------------
    summary_rows = []

    frozen_candidates = {}

    for candidate in CANDIDATES:
        sub = runs.loc[
            runs[
                "candidate"
            ]
            == candidate
        ].copy()

        seed_summary = (
            sub.groupby(
                "seed",
                as_index=False,
            )
            .agg(
                selection_score_mean=(
                    "selection_score",
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
                best_epoch_median=(
                    "best_epoch",
                    "median",
                ),
            )
            .sort_values(
                "selection_score_mean"
            )
            .reset_index(
                drop=True
            )
        )

        selected_seed = int(
            seed_summary.iloc[
                0
            ][
                "seed"
            ]
        )

        selected_seed_runs = sub.loc[
            sub[
                "seed"
            ].astype(
                int
            )
            == selected_seed
        ]

        final_epoch = int(
            max(
                1,
                round(
                    float(
                        np.median(
                            selected_seed_runs[
                                "best_epoch"
                            ].to_numpy(
                                dtype=float
                            )
                        )
                    )
                ),
            )
        )

        frozen_candidates[
            candidate
        ] = {
            "selected_seed": (
                selected_seed
            ),
            "final_epoch": (
                final_epoch
            ),
        }

        summary_rows.append(
            {
                "candidate": candidate,
                "runs": int(
                    len(
                        sub
                    )
                ),
                "selection_score_mean_all_runs": float(
                    sub[
                        "selection_score"
                    ].mean()
                ),
                "U_RMSE_mean_all_runs": float(
                    sub[
                        "wind_U_RMSE_mps"
                    ].mean()
                ),
                "V_RMSE_mean_all_runs": float(
                    sub[
                        "wind_V_RMSE_mps"
                    ].mean()
                ),
                "WS_RMSE_mean_all_runs": float(
                    sub[
                        "wind_speed_RMSE_mps"
                    ].mean()
                ),
                "WD_RMSE_mean_all_runs": float(
                    sub[
                        "wind_direction_RMSE_deg"
                    ].mean()
                ),
                "parameter_count": int(
                    round(
                        sub[
                            "parameter_count"
                        ].mean()
                    )
                ),
                "selected_seed": (
                    selected_seed
                ),
                "final_epoch": (
                    final_epoch
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "selection_score_mean_all_runs",
            "V_RMSE_mean_all_runs",
        ]
    ).reset_index(
        drop=True
    )

    summary_path = (
        output_dir
        / "development_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    frozen = {
        "stage": "13B",
        "script_version": (
            SCRIPT_VERSION
        ),
        "scientific_label": (
            "method-faithful/task-adapted BP-style diagnostic; not exact reproduction"
        ),
        "physical_mask_ones": int(
            np.sum(
                PHYSICAL_MASK
            )
        ),
        "node_order": (
            NODE_NAMES
        ),
        "candidates": (
            frozen_candidates
        ),
        "development_summary": (
            summary.to_dict(
                orient="records"
            )
        ),
        "Tropical_Atlantic_used_for_selection": False,
        "implementation_freeze": asdict(
            FREEZE
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
    log("=" * 132)
    log(
        "13B DEVELOPMENT SUMMARY"
    )
    log("=" * 132)
    log(
        summary.to_string(
            index=False
        )
    )

    log(
        f"\n[FROZEN BEFORE TROPICAL] {frozen_path}"
    )

    # ---------------------------------------------------------
    # Tropical
    # ---------------------------------------------------------
    log("")
    log(
        "[STAGE C] Candidate/seed/epoch frozen. Loading Tropical Atlantic NOW."
    )

    tropical = load_mission(
        dataset_root,
        FINAL_MISSION,
    )

    log(
        f"[TROPICAL DATASET] {tropical['path']}"
    )

    dev_all = concat_missions(
        [
            dev_missions[
                m
            ]
            for m in DEV_MISSIONS
        ]
    )

    final_rows = []

    # Published row
    final_rows.append(
        {
            "model": (
                "BP-STGNN-published"
            ),
            "parameter_count": int(
                BP_REFERENCE[
                    "params"
                ]
            ),
            "wind_U_RMSE_mps": (
                BP_REFERENCE[
                    "U_RMSE_mps"
                ]
            ),
            "wind_V_RMSE_mps": (
                BP_REFERENCE[
                    "V_RMSE_mps"
                ]
            ),
            "wind_vector_RMSE_mps": (
                np.nan
            ),
            "wind_speed_RMSE_mps": (
                BP_REFERENCE[
                    "WS_RMSE_mps"
                ]
            ),
            "wind_direction_RMSE_deg": (
                BP_REFERENCE[
                    "WD_RMSE_deg"
                ]
            ),
        }
    )

    # Base4 Ridge
    base4_model, base4_scaler = fit_ridge(
        base4_from_trackP(
            dev_all[
                "X"
            ]
        ),
        dev_all[
            "y"
        ],
    )

    base4_pred = predict_ridge(
        base4_model,
        base4_scaler,
        base4_from_trackP(
            tropical[
                "X"
            ]
        ),
    )

    base4_metrics = evaluate_wind(
        tropical[
            "y"
        ],
        base4_pred,
    )

    final_rows.append(
        {
            "model": (
                "Base4-Ridge"
            ),
            "parameter_count": int(
                ridge_param_count(
                    base4_model
                )
            ),
            **base4_metrics,
        }
    )

    # Full10 Ridge
    full_model, full_scaler = fit_ridge(
        dev_all[
            "X"
        ],
        dev_all[
            "y"
        ],
    )

    full_pred = predict_ridge(
        full_model,
        full_scaler,
        tropical[
            "X"
        ],
    )

    full_metrics = evaluate_wind(
        tropical[
            "y"
        ],
        full_pred,
    )

    final_rows.append(
        {
            "model": (
                "Full10-Ridge"
            ),
            "parameter_count": int(
                ridge_param_count(
                    full_model
                )
            ),
            **full_metrics,
        }
    )

    # Neural candidates
    for candidate in CANDIDATES:
        info = frozen_candidates[
            candidate
        ]

        log(
            f"[FINAL TRAIN] {candidate} | "
            f"seed={info['selected_seed']} | "
            f"epochs={info['final_epoch']}"
        )

        row = train_final_candidate(
            torch=torch,
            nn=nn,
            F=F,
            candidate=candidate,
            seed=info[
                "selected_seed"
            ],
            epochs=info[
                "final_epoch"
            ],
            dev_all=dev_all,
            tropical=tropical,
            device=device,
        )

        final_rows.append(
            row
        )

    final_df = pd.DataFrame(
        final_rows
    )

    final_path = (
        output_dir
        / "FINAL_TROPICAL_BPSTYLE_DIAGNOSTIC.csv"
    )

    final_df.to_csv(
        final_path,
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log("=" * 132)
    log(
        "FINAL TROPICAL — METHOD-FAITHFUL BP-STYLE ARCHITECTURE DIAGNOSTIC"
    )
    log("=" * 132)
    log(
        final_df.to_string(
            index=False
        )
    )

    local_neural = final_df.loc[
        final_df[
            "model"
        ].isin(
            CANDIDATES
        )
    ].copy()

    best_local_idx = (
        local_neural[
            "wind_V_RMSE_mps"
        ].idxmin()
    )

    best_local = final_df.loc[
        best_local_idx
    ]

    report = {
        "stage": "13B",
        "best_local_neural_by_Tropical_V": {
            "model": str(
                best_local[
                    "model"
                ]
            ),
            "V_RMSE_mps": float(
                best_local[
                    "wind_V_RMSE_mps"
                ]
            ),
            "U_RMSE_mps": float(
                best_local[
                    "wind_U_RMSE_mps"
                ]
            ),
            "WS_RMSE_mps": float(
                best_local[
                    "wind_speed_RMSE_mps"
                ]
            ),
            "WD_RMSE_deg": float(
                best_local[
                    "wind_direction_RMSE_deg"
                ]
            ),
        },
        "published_BP_reference": (
            BP_REFERENCE
        ),
        "frozen_before_tropical": (
            frozen
        ),
        "interpretation": (
            "This is an architecture diagnostic on Stage12G samples, "
            "not an exact reproduction of Nie et al."
        ),
    }

    report_json = (
        output_dir
        / "13B_REPORT.json"
    )

    save_json(
        report_json,
        report,
    )

    report_txt = (
        output_dir
        / "13B_REPORT.txt"
    )

    with report_txt.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "13B BP-STYLE GRAPH-GRU ARCHITECTURE DIAGNOSTIC\n"
        )
        f.write(
            "=" * 120
            + "\n\n"
        )

        f.write(
            "DEVELOPMENT SUMMARY\n"
        )
        f.write(
            "-" * 120
            + "\n"
        )

        f.write(
            summary.to_string(
                index=False
            )
        )

        f.write(
            "\n\nFINAL TROPICAL\n"
        )
        f.write(
            "-" * 120
            + "\n"
        )

        f.write(
            final_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "CAVEAT: method-faithful/task-adapted diagnostic, "
            "not exact official BP-STGNN reproduction.\n"
        )

    log("")
    log(
        f"[BEST LOCAL NEURAL BY V] "
        f"{best_local['model']} | "
        f"V={best_local['wind_V_RMSE_mps']:.6f}"
    )

    log(
        f"[SAVED] {final_path}"
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
