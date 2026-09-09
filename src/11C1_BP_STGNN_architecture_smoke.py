# -*- coding: utf-8 -*-
"""
11C1_BP_STGNN_architecture_smoke.py

Stage 11C1
===========
Architecture + Bayesian-physics graph smoke audit for the full-paper-informed
BP-STGNN reimplementation.

THIS SCRIPT DOES NOT PERFORM SCIENTIFIC MODEL TRAINING OR HYPERPARAMETER
SELECTION.

It uses ONLY:
    SD1090_train.npz
    SD1090_validation.npz

from the frozen Stage-11C0 dataset build. It never reads SD1090_test or any
SD1033 external dataset.

Supported tracks
----------------
Track F:
    1-min common-task adaptation
    X: [B, 60, 10]
    Y: [B, 5, 2], horizons [1,2,3,5,10] min

Track P:
    10-min paper-protocol-oriented adaptation
    X: [B, 6, 10]
    Y: [B, 1, 2], next 10-min resampled U/V

Paper-exact components implemented
----------------------------------
1) Ten graph nodes:
   U, V, SOG, COG, T, H, P, dT, dP, dH

2) Paper Eq.(2) physical mask:
   (S union K) -> T
   E -> S
   E -> E
   self loops
   otherwise zero

3) Bayesian adjacency:
       A_ij ~ N(mu_adj_ij, sigma_adj_ij^2)

4) Physics-centered KL:
       q(A_ij) || N(M_ij, 1)

5) Reparameterization:
       A = mu_adj + sigma_adj * epsilon

6) Spatial graph aggregation -> GRU temporal propagation

7) Heteroscedastic output:
       [mu_hat, log_sigma_hat^2]

8) Two-stage loss logic:
   Phase 1 (epoch < 30):
       MSE + lambda_KL * KL
       variance head frozen
   Phase 2 (epoch >= 30):
       MSE + lambda_NLL * NLL + lambda_KL * KL
       variance head unfrozen

9) Variance-feature gradient decoupling:
   log-variance head receives detached shared features.

Implementation choices NOT specified numerically in Nie et al. (2026)
---------------------------------------------------------------------
The full paper does not uniquely specify GCN widths/depth, GRU hidden size,
MLP size, adjacency normalization, positive-sigma parameterization, exact
lambda_KL/lambda_NLL, or optimizer hyperparameters.

11C1 therefore defines ONE NON-SCIENTIFIC ANCHOR CONFIGURATION only for
tensor/gradient/numerical smoke testing:

    node_embed_dim = 32
    gcn_hidden_dim = 64
    gcn_layers = 2
    gru_hidden_dim = 192
    gru_layers = 1
    mlp_hidden_dim = 64
    dropout = 0.10
    adjacency sigma parameterization = softplus(rho) + 1e-6
    mu_adj initialization = physical mask M
    initial adjacency sigma ~= 0.10
    graph normalization = symmetric normalization of |A| + I safeguard
    lambda_KL = 1e-4       [SMOKE ONLY]
    lambda_NLL = 0.10      [SMOKE ONLY]
    optimizer = Adam, lr=1e-3 [SMOKE ONLY]

NONE of these numerical anchor choices is frozen for Stage 11C2 HPO merely
because this smoke script uses them.

Numerical design notes
----------------------
- Sampled Bayesian edges may be negative. For stable graph normalization,
  the signed sampled adjacency is normalized using row/column degree computed
  from |A| while preserving the sign of A itself.
- The paper's physical mask acts as the prior mean, not a hard zero mask.
  Therefore non-physical edges are NOT forced to zero after sampling.
- COG is retained as the scalar paper node supplied by Stage 11C0.
- Stage 11C1 computes TEMPORARY train-only normalization statistics solely to
  make the numerical smoke test well-conditioned. These statistics are NOT
  saved as the scientific scaler and MUST NOT be reused by Stage 11C2.

Audits
------
For each requested track:
- dataset schema and finite-value checks;
- physical mask == 35/100;
- tensor shape checks;
- Bayesian sigma positivity;
- KL formula cross-check;
- stochastic reparameterization finite check;
- deterministic posterior-mean adjacency path;
- GCN forward;
- GRU forward;
- mean/logvar output shapes;
- predictive variance positivity;
- phase-1 backward;
- phase-2 backward;
- gradient-decoupling test;
- short smoke optimization trajectory;
- trainable parameter count.

Outputs
-------
11C1_BP_STGNN_architecture_smoke_v0_1/
    architecture_audit.csv
    track_schema_audit.csv
    gradient_audit.csv
    smoke_training_results.csv
    anchor_configuration.json
    architecture_manifest.json
    architecture_report.txt

Recommended run
---------------
conda activate WindPredict
python "D:\\project\\WindPredict_SaildroneData\\src\\11C1_BP_STGNN_architecture_smoke.py"

Optional
--------
--track F
--track P
--track both
--force-cpu
--smoke-steps 20
--smoke-train-samples 4096
--smoke-val-samples 2048
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import random
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.1-BP-STGNN-architecture-smoke-nameerror-fix"

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
    / "11C1_BP_STGNN_architecture_smoke_v0_1"
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

TRACK_CONFIG = {
    "F": {
        "dir_name": "track_F_common_task",
        "train_file": "SD1090_train.npz",
        "validation_file": "SD1090_validation.npz",
        "lookback": 60,
        "horizons_min": [
            1,
            2,
            3,
            5,
            10,
        ],
        "horizon_count": 5,
    },
    "P": {
        "dir_name": "track_P_paper_protocol",
        "train_file": "SD1090_train.npz",
        "validation_file": "SD1090_validation.npz",
        "lookback": 6,
        "horizons_min": [
            10,
        ],
        "horizon_count": 1,
    },
}

SMOKE_SEED = 11031


@dataclass
class AnchorConfig:
    node_embed_dim: int = 32
    gcn_hidden_dim: int = 64
    gcn_layers: int = 2
    gru_hidden_dim: int = 192
    gru_layers: int = 1
    mlp_hidden_dim: int = 64
    dropout: float = 0.10
    initial_adj_sigma: float = 0.10
    lambda_kl: float = 1e-4
    lambda_nll: float = 0.10
    learning_rate: float = 1e-3
    grad_clip_norm: float = 1.0
    logvar_min: float = -12.0
    logvar_max: float = 8.0


def log(
    message="",
):
    print(
        message,
        flush=True,
    )


def save_json(
    path: Path,
    obj,
):
    def convert(v):
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
                if np.isnan(
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
                ): convert(
                    x
                )
                for k, x
                in v.items()
            }

        if isinstance(
            v,
            (
                list,
                tuple,
            ),
        ):
            return [
                convert(
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
            convert(
                obj
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )


def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required. Please run in the WindPredict environment."
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


def build_physical_mask():
    """
    Paper Eq.(2).

    M_ij = 1 means directed edge j -> i is physically permitted.
    """
    n = len(
        NODE_NAMES
    )

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
            n,
            n,
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

    return M


def load_track_npz(
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
        result = {
            key: data[
                key
            ]
            for key in data.files
        }

    required = [
        "X_raw",
        "y_wind_raw",
        "feature_names",
        "target_names",
        "horizons_min",
        "context_end_time_ns",
        "target_time_ns",
    ]

    for key in required:
        if key not in result:
            raise KeyError(
                f"{path}: missing '{key}', available={list(result)}"
            )

    return result


def decode_strings(
    arr,
):
    return [
        str(
            x
        )
        for x in np.asarray(
            arr
        ).tolist()
    ]


def audit_track_schema(
    track: str,
    train: Dict[str, np.ndarray],
    val: Dict[str, np.ndarray],
):
    cfg = TRACK_CONFIG[
        track
    ]

    rows = []

    for split_name, data in [
        (
            "train",
            train,
        ),
        (
            "validation",
            val,
        ),
    ]:
        X = np.asarray(
            data[
                "X_raw"
            ]
        )

        y = np.asarray(
            data[
                "y_wind_raw"
            ]
        )

        feature_names = decode_strings(
            data[
                "feature_names"
            ]
        )

        target_names = decode_strings(
            data[
                "target_names"
            ]
        )

        horizons = [
            int(
                x
            )
            for x in np.asarray(
                data[
                    "horizons_min"
                ]
            ).tolist()
        ]

        expected_x = (
            cfg[
                "lookback"
            ],
            len(
                NODE_NAMES
            ),
        )

        expected_y = (
            cfg[
                "horizon_count"
            ],
            2,
        )

        schema_pass = bool(
            X.ndim == 3
            and y.ndim == 3
            and tuple(
                X.shape[
                    1:
                ]
            ) == expected_x
            and tuple(
                y.shape[
                    1:
                ]
            ) == expected_y
            and feature_names == NODE_NAMES
            and target_names
            == [
                "U",
                "V",
            ]
            and horizons
            == cfg[
                "horizons_min"
            ]
        )

        finite_pass = bool(
            np.isfinite(
                X
            ).all()
            and np.isfinite(
                y
            ).all()
        )

        rows.append({
            "track": track,
            "split": split_name,
            "samples": int(
                X.shape[
                    0
                ]
            ),
            "X_shape": str(
                tuple(
                    X.shape
                )
            ),
            "y_shape": str(
                tuple(
                    y.shape
                )
            ),
            "feature_names": ",".join(
                feature_names
            ),
            "horizons_min": ",".join(
                map(
                    str,
                    horizons,
                )
            ),
            "schema_pass": schema_pass,
            "finite_pass": finite_pass,
        })

        if not schema_pass:
            raise RuntimeError(
                f"Track {track}/{split_name}: schema audit failed."
            )

        if not finite_pass:
            raise RuntimeError(
                f"Track {track}/{split_name}: non-finite data found."
            )

    return rows


def temporary_train_scaler(
    train_X: np.ndarray,
    train_y: np.ndarray,
    max_samples: int,
):
    """
    NON-SCIENTIFIC smoke-only normalization.

    The statistics are intentionally NOT saved for Stage 11C2 use.
    """
    n = min(
        int(
            max_samples
        ),
        int(
            train_X.shape[
                0
            ]
        ),
    )

    X = np.asarray(
        train_X[
            :n
        ],
        dtype=np.float64,
    )

    y = np.asarray(
        train_y[
            :n
        ],
        dtype=np.float64,
    )

    feature_mean = X.reshape(
        -1,
        X.shape[
            -1
        ],
    ).mean(
        axis=0
    )

    feature_std = X.reshape(
        -1,
        X.shape[
            -1
        ],
    ).std(
        axis=0
    )

    feature_std = np.where(
        feature_std
        < 1e-8,
        1.0,
        feature_std,
    )

    target_mean = y.reshape(
        -1,
        y.shape[
            -1
        ],
    ).mean(
        axis=0
    )

    target_std = y.reshape(
        -1,
        y.shape[
            -1
        ],
    ).std(
        axis=0
    )

    target_std = np.where(
        target_std
        < 1e-8,
        1.0,
        target_std,
    )

    return (
        feature_mean.astype(
            np.float32
        ),
        feature_std.astype(
            np.float32
        ),
        target_mean.astype(
            np.float32
        ),
        target_std.astype(
            np.float32
        ),
    )


def standardize_X(
    X,
    mean,
    std,
):
    return (
        np.asarray(
            X,
            dtype=np.float32,
        )
        - mean[
            None,
            None,
            :
        ]
    ) / std[
        None,
        None,
        :
    ]


def standardize_y(
    y,
    mean,
    std,
):
    return (
        np.asarray(
            y,
            dtype=np.float32,
        )
        - mean[
            None,
            None,
            :
        ]
    ) / std[
        None,
        None,
        :
    ]


def inverse_y(
    y_z,
    mean,
    std,
):
    return (
        np.asarray(
            y_z,
            dtype=np.float64,
        )
        * std[
            None,
            None,
            :
        ]
        + mean[
            None,
            None,
            :
        ]
    )


def vector_rmse(
    truth,
    pred,
):
    e = (
        np.asarray(
            pred,
            dtype=np.float64,
        )
        - np.asarray(
            truth,
            dtype=np.float64,
        )
    )

    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    e ** 2,
                    axis=-1,
                )
            )
        )
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
            initial_sigma=0.10,
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

            # softplus(rho) ~= initial_sigma
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

        def kl_divergence(
            self,
        ):
            sigma = self.sigma()

            # Eq.(7): KL(q || N(M,1))
            kl_element = 0.5 * (
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

            return kl_element.sum()

        def sample(
            self,
            stochastic=True,
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

        def normalized(
            self,
            stochastic=True,
        ):
            """
            Smoke-test graph normalization.

            Preserve signed edge values, compute degree from absolute weights.
            A tiny identity safeguard prevents zero degree.
            """
            A = self.sample(
                stochastic=stochastic
            )

            n = A.shape[
                0
            ]

            I = torch.eye(
                n,
                device=A.device,
                dtype=A.dtype,
            )

            A_safe = (
                A
                + 1e-6
                * I
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
            in_dim,
            out_dim,
            dropout,
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
            # h: [B,T,N,D], A_norm[i,j] maps j -> i
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

    class BPSTGNNSmoke(
        nn.Module
    ):
        def __init__(
            self,
            physical_mask,
            horizon_count,
            cfg: AnchorConfig,
        ):
            super().__init__()

            self.horizon_count = int(
                horizon_count
            )

            self.target_count = 2

            self.cfg = cfg

            self.bayes_graph = BayesianAdjacency(
                physical_mask,
                initial_sigma=cfg.initial_adj_sigma,
            )

            # Scalar node expansion. Node identity embedding keeps the ten
            # physical variables distinguishable after shared scalar mapping.
            self.scalar_expand = nn.Linear(
                1,
                cfg.node_embed_dim,
            )

            self.node_embedding = nn.Parameter(
                torch.zeros(
                    len(
                        NODE_NAMES
                    ),
                    cfg.node_embed_dim,
                )
            )

            nn.init.normal_(
                self.node_embedding,
                mean=0.0,
                std=0.02,
            )

            gcn_layers = []

            in_dim = cfg.node_embed_dim

            for _ in range(
                cfg.gcn_layers
            ):
                gcn_layers.append(
                    GraphConvBlock(
                        in_dim,
                        cfg.gcn_hidden_dim,
                        cfg.dropout,
                    )
                )

                in_dim = cfg.gcn_hidden_dim

            self.gcn_layers = nn.ModuleList(
                gcn_layers
            )

            # Pool graph nodes after spatial propagation to a graph-level
            # sequence, then apply GRU over time.
            self.gru = nn.GRU(
                input_size=cfg.gcn_hidden_dim,
                hidden_size=cfg.gru_hidden_dim,
                num_layers=cfg.gru_layers,
                batch_first=True,
                dropout=(
                    cfg.dropout
                    if cfg.gru_layers
                    > 1
                    else 0.0
                ),
            )

            self.mean_mlp = nn.Sequential(
                nn.Linear(
                    cfg.gru_hidden_dim,
                    cfg.mlp_hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(
                    cfg.dropout
                ),
                nn.Linear(
                    cfg.mlp_hidden_dim,
                    self.horizon_count
                    * self.target_count,
                ),
            )

            self.var_mlp = nn.Sequential(
                nn.Linear(
                    cfg.gru_hidden_dim,
                    cfg.mlp_hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(
                    cfg.dropout
                ),
                nn.Linear(
                    cfg.mlp_hidden_dim,
                    self.horizon_count
                    * self.target_count,
                ),
            )

            # Paper principle: very small variance-branch initialization and
            # zero output bias. We apply the small initialization to the final
            # variance projection.
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
            trainable: bool,
        ):
            for p in self.var_mlp.parameters():
                p.requires_grad = bool(
                    trainable
                )

        def shared_parameters(
            self,
        ):
            groups = [
                self.scalar_expand,
                self.gcn_layers,
                self.gru,
                self.mean_mlp,
                self.bayes_graph,
            ]

            for group in groups:
                if isinstance(
                    group,
                    nn.Parameter,
                ):
                    yield group

                elif isinstance(
                    group,
                    nn.ParameterList,
                ):
                    for p in group:
                        yield p

                elif isinstance(
                    group,
                    nn.ModuleList,
                ):
                    for module in group:
                        for p in module.parameters():
                            yield p

                else:
                    for p in group.parameters():
                        yield p

            yield self.node_embedding

        def forward(
            self,
            x,
            stochastic_graph=True,
            detach_variance_features=True,
        ):
            # x: [B,T,N]
            if x.ndim != 3:
                raise RuntimeError(
                    f"Expected x [B,T,N], got {tuple(x.shape)}"
                )

            if x.shape[
                -1
            ] != len(
                NODE_NAMES
            ):
                raise RuntimeError(
                    "Node count mismatch."
                )

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

            # [B,T,N,D] -> [B,T,D]
            graph_sequence = h.mean(
                dim=2
            )

            gru_out, hidden = self.gru(
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
                self.horizon_count,
                self.target_count,
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
                self.horizon_count,
                self.target_count,
            )

            logvar = torch.clamp(
                logvar,
                min=self.cfg.logvar_min,
                max=self.cfg.logvar_max,
            )

            return {
                "mu": mu,
                "logvar": logvar,
                "variance": torch.exp(
                    logvar
                ),
                "A_sample": A_sample,
                "A_norm": A_norm,
                "KL": self.bayes_graph.kl_divergence(),
                "shared": shared,
                "graph_sequence": graph_sequence,
            }

    return (
        BayesianAdjacency,
        GraphConvBlock,
        BPSTGNNSmoke,
    )


def mse_loss(
    torch,
    mu,
    y,
):
    return torch.mean(
        (
            mu
            - y
        ) ** 2
    )


def gaussian_nll(
    torch,
    mu,
    logvar,
    y,
):
    return torch.mean(
        0.5
        * (
            logvar
            + (
                y
                - mu
            ) ** 2
            * torch.exp(
                -logvar
            )
        )
    )


def phase_loss(
    torch,
    output,
    y,
    cfg: AnchorConfig,
    phase: int,
):
    mse = mse_loss(
        torch,
        output[
            "mu"
        ],
        y,
    )

    kl = output[
        "KL"
    ]

    if int(
        phase
    ) == 1:
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

    elif int(
        phase
    ) == 2:
        nll = gaussian_nll(
            torch,
            output[
                "mu"
            ],
            output[
                "logvar"
            ],
            y,
        )

        total = (
            mse
            + cfg.lambda_nll
            * nll
            + cfg.lambda_kl
            * kl
        )

    else:
        raise ValueError(
            phase
        )

    return {
        "total": total,
        "mse": mse,
        "nll": nll,
        "kl": kl,
    }


def grad_norm(
    parameters,
):
    total = 0.0

    any_grad = False

    for p in parameters:
        if (
            p.grad
            is not None
        ):
            any_grad = True

            total += float(
                torch_sum_square(
                    p.grad
                )
            )

    return (
        math.sqrt(
            total
        )
        if any_grad
        else 0.0
    )


def torch_sum_square(
    tensor,
):
    return float(
        tensor.detach()
        .float()
        .pow(
            2
        )
        .sum()
        .cpu()
        .item()
    )


def module_grad_norm(
    module,
):
    total = 0.0

    any_grad = False

    for p in module.parameters():
        if p.grad is not None:
            any_grad = True

            total += torch_sum_square(
                p.grad
            )

    return (
        math.sqrt(
            total
        )
        if any_grad
        else 0.0
    )


def count_parameters(
    model,
):
    trainable = sum(
        int(
            p.numel()
        )
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        int(
            p.numel()
        )
        for p in model.parameters()
    )

    return (
        trainable,
        total,
    )


def mini_batches(
    X,
    y,
    batch_size,
    max_samples,
):
    n = min(
        int(
            max_samples
        ),
        int(
            X.shape[
                0
            ]
        ),
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
            X[
                start:end
            ],
            y[
                start:end
            ],
        )


def evaluate_smoke(
    torch,
    model,
    X,
    y,
    feature_mean,
    feature_std,
    target_mean,
    target_std,
    device,
    max_samples,
    batch_size,
):
    model.eval()

    preds = []

    truth = []

    n = min(
        int(
            max_samples
        ),
        int(
            X.shape[
                0
            ]
        ),
    )

    with torch.no_grad():
        for Xb_raw, yb_raw in mini_batches(
            X,
            y,
            batch_size,
            n,
        ):
            Xb = standardize_X(
                Xb_raw,
                feature_mean,
                feature_std,
            )

            xb = torch.from_numpy(
                Xb
            ).to(
                device
            )

            out = model(
                xb,
                stochastic_graph=False,
                detach_variance_features=True,
            )

            pred_z = (
                out[
                    "mu"
                ]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            pred_raw = inverse_y(
                pred_z,
                target_mean,
                target_std,
            )

            preds.append(
                pred_raw
            )

            truth.append(
                np.asarray(
                    yb_raw,
                    dtype=np.float64,
                )
            )

    if not preds:
        return np.nan

    return vector_rmse(
        np.concatenate(
            truth,
            axis=0,
        ),
        np.concatenate(
            preds,
            axis=0,
        ),
    )


def run_track_smoke(
    *,
    track: str,
    train,
    val,
    physical_mask,
    cfg: AnchorConfig,
    torch,
    nn,
    F,
    device,
    smoke_steps,
    smoke_train_samples,
    smoke_val_samples,
    batch_size,
):
    (
        BayesianAdjacency,
        GraphConvBlock,
        BPSTGNNSmoke,
    ) = build_model_classes(
        torch,
        nn,
        F,
    )

    track_cfg = TRACK_CONFIG[
        track
    ]

    X_train = np.asarray(
        train[
            "X_raw"
        ],
        dtype=np.float32,
    )

    y_train = np.asarray(
        train[
            "y_wind_raw"
        ],
        dtype=np.float32,
    )

    X_val = np.asarray(
        val[
            "X_raw"
        ],
        dtype=np.float32,
    )

    y_val = np.asarray(
        val[
            "y_wind_raw"
        ],
        dtype=np.float32,
    )

    (
        feature_mean,
        feature_std,
        target_mean,
        target_std,
    ) = temporary_train_scaler(
        X_train,
        y_train,
        max_samples=min(
            12000,
            len(
                X_train
            ),
        ),
    )

    model = BPSTGNNSmoke(
        physical_mask,
        track_cfg[
            "horizon_count"
        ],
        cfg,
    ).to(
        device
    )

    (
        trainable_params,
        total_params,
    ) = count_parameters(
        model
    )

    # ---------------------------------------------------------------
    # Forward architecture audit
    # ---------------------------------------------------------------
    probe_n = min(
        32,
        len(
            X_train
        ),
    )

    X_probe = standardize_X(
        X_train[
            :probe_n
        ],
        feature_mean,
        feature_std,
    )

    y_probe = standardize_y(
        y_train[
            :probe_n
        ],
        target_mean,
        target_std,
    )

    xb = torch.from_numpy(
        X_probe
    ).to(
        device
    )

    yb = torch.from_numpy(
        y_probe
    ).to(
        device
    )

    model.eval()

    with torch.no_grad():
        out = model(
            xb,
            stochastic_graph=True,
            detach_variance_features=True,
        )

    expected_output_shape = (
        probe_n,
        track_cfg[
            "horizon_count"
        ],
        2,
    )

    tensor_checks = {
        "mu_shape_pass": tuple(
            out[
                "mu"
            ].shape
        )
        == expected_output_shape,
        "logvar_shape_pass": tuple(
            out[
                "logvar"
            ].shape
        )
        == expected_output_shape,
        "variance_positive_pass": bool(
            torch.all(
                out[
                    "variance"
                ]
                > 0
            ).item()
        ),
        "adjacency_shape_pass": tuple(
            out[
                "A_sample"
            ].shape
        )
        == (
            10,
            10,
        ),
        "adjacency_finite_pass": bool(
            torch.isfinite(
                out[
                    "A_sample"
                ]
            ).all().item()
        ),
        "normalized_adjacency_finite_pass": bool(
            torch.isfinite(
                out[
                    "A_norm"
                ]
            ).all().item()
        ),
        "KL_finite_pass": bool(
            torch.isfinite(
                out[
                    "KL"
                ]
            ).item()
        ),
        "sigma_positive_pass": bool(
            torch.all(
                model.bayes_graph.sigma()
                > 0
            ).item()
        ),
    }

    if not all(
        tensor_checks.values()
    ):
        raise RuntimeError(
            f"Track {track}: tensor/finite architecture audit failed: "
            f"{tensor_checks}"
        )

    # ---------------------------------------------------------------
    # KL formula independent cross-check
    # ---------------------------------------------------------------
    with torch.no_grad():
        sigma = model.bayes_graph.sigma()

        mu = model.bayes_graph.mu_adj

        M = model.bayes_graph.physical_mask

        kl_manual = 0.5 * torch.sum(
            sigma ** 2
            + (
                mu
                - M
            ) ** 2
            - 1.0
            - torch.log(
                sigma ** 2
            )
        )

        kl_model = model.bayes_graph.kl_divergence()

        kl_abs_diff = float(
            torch.abs(
                kl_manual
                - kl_model
            )
            .cpu()
            .item()
        )

        kl_formula_pass = bool(
            kl_abs_diff
            <= 1e-6
        )

    if not kl_formula_pass:
        raise RuntimeError(
            f"Track {track}: KL formula cross-check failed "
            f"(abs diff={kl_abs_diff})."
        )

    # ---------------------------------------------------------------
    # Gradient-decoupling audit
    # ---------------------------------------------------------------
    gradient_rows = []

    # Variance-only scalar must update variance head, but shared features
    # should receive no gradient because var_mlp sees shared.detach().
    model.train()

    model.set_variance_trainable(
        True
    )

    model.zero_grad(
        set_to_none=True
    )

    out_var = model(
        xb,
        stochastic_graph=False,
        detach_variance_features=True,
    )

    variance_only = out_var[
        "logvar"
    ].mean()

    variance_only.backward()

    shared_feature_modules = [
        model.scalar_expand,
        model.gcn_layers,
        model.gru,
    ]

    shared_feature_grad_sq = 0.0

    for module in shared_feature_modules:
        if isinstance(
            module,
            nn.ModuleList,
        ):
            for submodule in module:
                g = module_grad_norm(
                    submodule
                )

                shared_feature_grad_sq += (
                    g ** 2
                )

        else:
            g = module_grad_norm(
                module
            )

            shared_feature_grad_sq += (
                g ** 2
            )

    if model.node_embedding.grad is not None:
        shared_feature_grad_sq += torch_sum_square(
            model.node_embedding.grad
        )

    shared_feature_grad = math.sqrt(
        shared_feature_grad_sq
    )

    var_head_grad = module_grad_norm(
        model.var_mlp
    )

    decoupling_pass = bool(
        shared_feature_grad
        <= 1e-12
        and var_head_grad
        > 0.0
    )

    gradient_rows.append({
        "track": track,
        "audit": "variance_only_gradient_decoupling",
        "shared_feature_grad_norm": shared_feature_grad,
        "variance_head_grad_norm": var_head_grad,
        "pass": decoupling_pass,
    })

    if not decoupling_pass:
        raise RuntimeError(
            f"Track {track}: variance-feature gradient decoupling failed. "
            f"shared={shared_feature_grad:.3e}, "
            f"var_head={var_head_grad:.3e}"
        )

    # Mean loss should drive shared features.
    model.zero_grad(
        set_to_none=True
    )

    out_mean = model(
        xb,
        stochastic_graph=False,
        detach_variance_features=True,
    )

    mean_only = mse_loss(
        torch,
        out_mean[
            "mu"
        ],
        yb,
    )

    mean_only.backward()

    shared_feature_grad_sq = 0.0

    for module in shared_feature_modules:
        if isinstance(
            module,
            nn.ModuleList,
        ):
            for submodule in module:
                g = module_grad_norm(
                    submodule
                )

                shared_feature_grad_sq += (
                    g ** 2
                )

        else:
            g = module_grad_norm(
                module
            )

            shared_feature_grad_sq += (
                g ** 2
            )

    if model.node_embedding.grad is not None:
        shared_feature_grad_sq += torch_sum_square(
            model.node_embedding.grad
        )

    mean_shared_grad = math.sqrt(
        shared_feature_grad_sq
    )

    mean_grad_pass = bool(
        mean_shared_grad
        > 0.0
        and np.isfinite(
            mean_shared_grad
        )
    )

    gradient_rows.append({
        "track": track,
        "audit": "mean_loss_drives_shared_features",
        "shared_feature_grad_norm": mean_shared_grad,
        "variance_head_grad_norm": module_grad_norm(
            model.var_mlp
        ),
        "pass": mean_grad_pass,
    })

    if not mean_grad_pass:
        raise RuntimeError(
            f"Track {track}: mean loss did not drive shared features."
        )

    # ---------------------------------------------------------------
    # Phase-1 and phase-2 backward audits
    # ---------------------------------------------------------------
    phase_rows = []

    for phase in [
        1,
        2,
    ]:
        model.zero_grad(
            set_to_none=True
        )

        model.set_variance_trainable(
            phase == 2
        )

        out_phase = model(
            xb,
            stochastic_graph=True,
            detach_variance_features=True,
        )

        losses = phase_loss(
            torch,
            out_phase,
            yb,
            cfg,
            phase,
        )

        losses[
            "total"
        ].backward()

        finite_loss = bool(
            torch.isfinite(
                losses[
                    "total"
                ]
            ).item()
        )

        any_shared_grad = False

        for p in model.parameters():
            if (
                p.requires_grad
                and p.grad
                is not None
                and torch.isfinite(
                    p.grad
                ).all()
            ):
                any_shared_grad = True
                break

        phase_pass = bool(
            finite_loss
            and any_shared_grad
        )

        phase_rows.append({
            "track": track,
            "phase": phase,
            "total_loss": float(
                losses[
                    "total"
                ]
                .detach()
                .cpu()
                .item()
            ),
            "mse": float(
                losses[
                    "mse"
                ]
                .detach()
                .cpu()
                .item()
            ),
            "nll": float(
                losses[
                    "nll"
                ]
                .detach()
                .cpu()
                .item()
            ),
            "kl": float(
                losses[
                    "kl"
                ]
                .detach()
                .cpu()
                .item()
            ),
            "variance_head_trainable": bool(
                phase
                == 2
            ),
            "pass": phase_pass,
        })

        if not phase_pass:
            raise RuntimeError(
                f"Track {track}: phase-{phase} backward audit failed."
            )

    # ---------------------------------------------------------------
    # Short smoke optimization
    # ---------------------------------------------------------------
    model = BPSTGNNSmoke(
        physical_mask,
        track_cfg[
            "horizon_count"
        ],
        cfg,
    ).to(
        device
    )

    model.set_variance_trainable(
        True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
    )

    smoke_n = min(
        int(
            smoke_train_samples
        ),
        len(
            X_train
        ),
    )

    X_smoke = standardize_X(
        X_train[
            :smoke_n
        ],
        feature_mean,
        feature_std,
    )

    y_smoke = standardize_y(
        y_train[
            :smoke_n
        ],
        target_mean,
        target_std,
    )

    if smoke_n < 1:
        raise RuntimeError(
            f"Track {track}: no train samples for smoke."
        )

    initial_val_rmse = evaluate_smoke(
        torch,
        model,
        X_val,
        y_val,
        feature_mean,
        feature_std,
        target_mean,
        target_std,
        device,
        max_samples=smoke_val_samples,
        batch_size=batch_size,
    )

    # IMPORTANT:
    # evaluate_smoke() intentionally switches the model to eval() mode.
    # cuDNN RNN/GRU backward on CUDA requires that the forward pass used for
    # backpropagation be executed in training mode. Explicitly restore train()
    # before the smoke-optimization loop and again at every step for safety.
    model.train()

    losses_trace = []

    for step in range(
        int(
            smoke_steps
        ),
    ):
        start = (
            step
            * int(
                batch_size
            )
        ) % smoke_n

        end = min(
            smoke_n,
            start
            + int(
                batch_size
            ),
        )

        if (
            end
            - start
            < 2
        ):
            start = 0

            end = min(
                smoke_n,
                int(
                    batch_size
                ),
            )

        xb_np = X_smoke[
            start:end
        ]

        yb_np = y_smoke[
            start:end
        ]

        xb_step = torch.from_numpy(
            xb_np
        ).to(
            device
        )

        yb_step = torch.from_numpy(
            yb_np
        ).to(
            device
        )

        # Smoke both phases in one short run. First half phase 1,
        # second half phase 2. This is NOT the paper's 30-epoch schedule;
        # the exact 30-epoch boundary is already audited separately.
        phase = (
            1
            if step
            < max(
                1,
                int(
                    smoke_steps
                )
                // 2,
            )
            else 2
        )

        # Defensive: validation/evaluation helpers may leave the model in
        # eval() mode. Every optimization forward must run in train() mode so
        # cuDNN GRU reserves the tensors required for backward().
        model.train()

        model.set_variance_trainable(
            phase == 2
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        out_step = model(
            xb_step,
            stochastic_graph=True,
            detach_variance_features=True,
        )

        losses = phase_loss(
            torch,
            out_step,
            yb_step,
            cfg,
            phase,
        )

        if not torch.isfinite(
            losses[
                "total"
            ]
        ):
            raise RuntimeError(
                f"Track {track}: non-finite smoke loss at step {step}."
            )

        losses[
            "total"
        ].backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.grad_clip_norm,
        )

        optimizer.step()

        losses_trace.append(
            float(
                losses[
                    "total"
                ]
                .detach()
                .cpu()
                .item()
            )
        )

    final_val_rmse = evaluate_smoke(
        torch,
        model,
        X_val,
        y_val,
        feature_mean,
        feature_std,
        target_mean,
        target_std,
        device,
        max_samples=smoke_val_samples,
        batch_size=batch_size,
    )

    finite_smoke_pass = bool(
        np.isfinite(
            losses_trace
        ).all()
        and np.isfinite(
            initial_val_rmse
        )
        and np.isfinite(
            final_val_rmse
        )
    )

    if not finite_smoke_pass:
        raise RuntimeError(
            f"Track {track}: smoke optimization finite audit failed."
        )

    architecture_row = {
        "track": track,
        "lookback": int(
            track_cfg[
                "lookback"
            ]
        ),
        "horizon_count": int(
            track_cfg[
                "horizon_count"
            ]
        ),
        "horizons_min": ",".join(
            map(
                str,
                track_cfg[
                    "horizons_min"
                ],
            )
        ),
        "node_count": len(
            NODE_NAMES
        ),
        "physical_mask_ones": int(
            np.asarray(
                physical_mask
            ).sum()
        ),
        "trainable_parameters_anchor": int(
            count_parameters(
                model
            )[
                0
            ]
        ),
        "total_parameters_anchor": int(
            count_parameters(
                model
            )[
                1
            ]
        ),
        "paper_reported_parameters": 242580,
        "anchor_minus_paper_parameters": int(
            count_parameters(
                model
            )[
                0
            ]
            - 242580
        ),
        "KL_formula_abs_diff": kl_abs_diff,
        "KL_formula_pass": kl_formula_pass,
        **tensor_checks,
        "all_architecture_pass": bool(
            kl_formula_pass
            and all(
                tensor_checks.values()
            )
            and all(
                x[
                    "pass"
                ]
                for x in gradient_rows
            )
            and all(
                x[
                    "pass"
                ]
                for x in phase_rows
            )
            and finite_smoke_pass
        ),
    }

    smoke_row = {
        "track": track,
        "smoke_train_samples": int(
            smoke_n
        ),
        "smoke_val_samples": int(
            min(
                int(
                    smoke_val_samples
                ),
                len(
                    X_val
                ),
            )
        ),
        "smoke_steps": int(
            smoke_steps
        ),
        "first_loss": float(
            losses_trace[
                0
            ]
        ),
        "last_loss": float(
            losses_trace[
                -1
            ]
        ),
        "min_loss": float(
            np.min(
                losses_trace
            )
        ),
        "initial_val_wind_vector_RMSE_mps": float(
            initial_val_rmse
        ),
        "final_val_wind_vector_RMSE_mps": float(
            final_val_rmse
        ),
        "finite_pass": finite_smoke_pass,
        "scientific_performance_result": False,
    }

    del (
        model,
        xb,
        yb,
        out,
        out_var,
        out_mean,
    )

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (
        architecture_row,
        gradient_rows,
        phase_rows,
        smoke_row,
    )


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
            "Stage-11C0 output directory. Defaults to "
            ".../11C0_BP_STGNN_comparison_datasets_v0_1."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--track",
        choices=[
            "F",
            "P",
            "both",
        ],
        default="both",
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
    )

    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--smoke-train-samples",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--smoke-val-samples",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
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
            / "11C1_BP_STGNN_architecture_smoke_v0_1"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch, nn, F = import_torch()

    seed_everything(
        torch,
        SMOKE_SEED,
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

    tracks = (
        [
            "F",
            "P",
        ]
        if args.track
        == "both"
        else [
            args.track
        ]
    )

    cfg = AnchorConfig()

    physical_mask = build_physical_mask()

    if physical_mask.shape != (
        10,
        10,
    ):
        raise RuntimeError(
            "Physical mask shape must be 10x10."
        )

    if int(
        physical_mask.sum()
    ) != 35:
        raise RuntimeError(
            f"Physical mask should contain 35 ones, "
            f"got {int(physical_mask.sum())}."
        )

    # Cross-check against frozen Stage-11C0 mask if present.
    frozen_mask_path = (
        dataset_dir
        / "physical_mask.npy"
    )

    mask_crosscheck = None

    if frozen_mask_path.exists():
        frozen_mask = np.load(
            frozen_mask_path
        )

        mask_crosscheck = bool(
            np.array_equal(
                frozen_mask.astype(
                    np.float32
                ),
                physical_mask,
            )
        )

        if not mask_crosscheck:
            raise RuntimeError(
                "11C1 physical mask differs from frozen 11C0 mask."
            )

    log(
        "="
        * 116
    )

    log(
        "11C1 - BP-STGNN ARCHITECTURE + BAYESIAN PHYSICS GRAPH SMOKE AUDIT"
    )

    log(
        "="
        * 116
    )

    log(
        f"script version        : {SCRIPT_VERSION}"
    )

    log(
        f"dataset dir           : {dataset_dir}"
    )

    log(
        f"output dir            : {output_dir}"
    )

    log(
        f"tracks                : {tracks}"
    )

    log(
        f"device                : {device}"
    )

    if device.type == "cuda":
        log(
            f"GPU                   : {torch.cuda.get_device_name(0)}"
        )

    log(
        f"physical mask         : {int(physical_mask.sum())}/100 allowed edges"
    )

    log(
        f"11C0 mask cross-check : {mask_crosscheck}"
    )

    log(
        "SD1090 test access    : NONE"
    )

    log(
        "SD1033 access         : NONE"
    )

    log(
        "scientific training   : NONE"
    )

    log(
        "HPO                   : NONE"
    )

    log("")

    schema_rows = []

    architecture_rows = []

    gradient_rows = []

    phase_rows = []

    smoke_rows = []

    for track in tracks:
        cfg_track = TRACK_CONFIG[
            track
        ]

        track_dir = (
            dataset_dir
            / cfg_track[
                "dir_name"
            ]
        )

        train_path = (
            track_dir
            / cfg_track[
                "train_file"
            ]
        )

        val_path = (
            track_dir
            / cfg_track[
                "validation_file"
            ]
        )

        log(
            "-"
            * 116
        )

        log(
            f"[TRACK {track}]"
        )

        log(
            f"train      : {train_path}"
        )

        log(
            f"validation : {val_path}"
        )

        train = load_track_npz(
            train_path
        )

        val = load_track_npz(
            val_path
        )

        schema = audit_track_schema(
            track,
            train,
            val,
        )

        schema_rows.extend(
            schema
        )

        for row in schema:
            log(
                f"  {row['split']}: X={row['X_shape']} | "
                f"y={row['y_shape']} | finite={row['finite_pass']}"
            )

        (
            architecture_row,
            gradient_track,
            phase_track,
            smoke_row,
        ) = run_track_smoke(
            track=track,
            train=train,
            val=val,
            physical_mask=physical_mask,
            cfg=cfg,
            torch=torch,
            nn=nn,
            F=F,
            device=device,
            smoke_steps=args.smoke_steps,
            smoke_train_samples=args.smoke_train_samples,
            smoke_val_samples=args.smoke_val_samples,
            batch_size=args.batch_size,
        )

        architecture_rows.append(
            architecture_row
        )

        gradient_rows.extend(
            gradient_track
        )

        phase_rows.extend(
            phase_track
        )

        smoke_rows.append(
            smoke_row
        )

        log(
            f"  params      : {architecture_row['trainable_parameters_anchor']:,} "
            f"(paper reported 242,580)"
        )

        log(
            f"  KL audit    : PASS | abs diff="
            f"{architecture_row['KL_formula_abs_diff']:.3e}"
        )

        log(
            f"  grad detach : PASS"
        )

        log(
            f"  phase 1/2   : PASS"
        )

        log(
            f"  smoke       : PASS | loss "
            f"{smoke_row['first_loss']:.6f} -> "
            f"{smoke_row['last_loss']:.6f}"
        )

        log(
            f"  smoke val vector RMSE [NOT SCIENTIFIC]: "
            f"{smoke_row['initial_val_wind_vector_RMSE_mps']:.4f} -> "
            f"{smoke_row['final_val_wind_vector_RMSE_mps']:.4f} m/s"
        )

    architecture_df = pd.DataFrame(
        architecture_rows
    )

    schema_df = pd.DataFrame(
        schema_rows
    )

    gradient_df = pd.DataFrame(
        gradient_rows
    )

    phase_df = pd.DataFrame(
        phase_rows
    )

    smoke_df = pd.DataFrame(
        smoke_rows
    )

    architecture_df.to_csv(
        output_dir
        / "architecture_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    schema_df.to_csv(
        output_dir
        / "track_schema_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gradient_df.to_csv(
        output_dir
        / "gradient_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    phase_df.to_csv(
        output_dir
        / "phase_loss_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    smoke_df.to_csv(
        output_dir
        / "smoke_training_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    anchor_json = {
        "status": (
            "SMOKE_ONLY_NOT_FROZEN_FOR_SCIENTIFIC_HPO"
        ),
        "script_version": SCRIPT_VERSION,
        "anchor_configuration": asdict(
            cfg
        ),
        "paper_reported_parameter_count": 242580,
        "explicitly_under_specified_in_paper": [
            "GCN widths/depth",
            "GRU hidden size/layers",
            "MLP dimensions",
            "adjacency normalization",
            "positive-sigma parameterization",
            "lambda_KL",
            "lambda_NLL",
            "optimizer",
            "learning rate",
            "batch size",
            "total epochs / early stopping",
        ],
        "important": (
            "Stage 11C2 must predeclare its own finite search space before "
            "outer validation/test/external results are inspected. This "
            "11C1 anchor does not become the selected configuration merely "
            "because it passes smoke tests."
        ),
    }

    save_json(
        output_dir
        / "anchor_configuration.json",
        anchor_json,
    )

    all_pass = bool(
        architecture_df[
            "all_architecture_pass"
        ].all()
        and gradient_df[
            "pass"
        ].all()
        and phase_df[
            "pass"
        ].all()
        and smoke_df[
            "finite_pass"
        ].all()
        and schema_df[
            "schema_pass"
        ].all()
        and schema_df[
            "finite_pass"
        ].all()
    )

    manifest = {
        "script_version": SCRIPT_VERSION,
        "stage": "11C1",
        "status": (
            "PASSED"
            if all_pass
            else "FAILED"
        ),
        "tracks": tracks,
        "device": str(
            device
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python": sys.version,
        "platform": platform.platform(),
        "physical_mask": {
            "shape": [
                10,
                10,
            ],
            "allowed_entries": int(
                physical_mask.sum()
            ),
            "matches_11C0_mask": mask_crosscheck,
        },
        "data_firewall": {
            "SD1090_train_used": True,
            "SD1090_validation_used": True,
            "SD1090_test_used": False,
            "SD1033_used": False,
        },
        "scientific_policy": {
            "model_selected": False,
            "hyperparameters_selected": False,
            "scientific_scaler_fitted": False,
            "test_performance_reported": False,
            "external_performance_reported": False,
            "proposed_model_changed": False,
            "smoke_metrics_are_scientific_results": False,
        },
        "anchor": asdict(
            cfg
        ),
        "all_pass": all_pass,
    }

    save_json(
        output_dir
        / "architecture_manifest.json",
        manifest,
    )

    with (
        output_dir
        / "architecture_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "11C1 BP-STGNN Architecture Smoke Audit\n"
        )

        f.write(
            "="
            * 116
            + "\n\n"
        )

        f.write(
            f"STATUS: {'PASSED' if all_pass else 'FAILED'}\n\n"
        )

        f.write(
            "Scientific firewall\n"
        )

        f.write(
            "-" * 116
            + "\n"
        )

        f.write(
            "Only SD1090 train and validation were accessed.\n"
        )

        f.write(
            "SD1090 test and all SD1033 external datasets were not accessed.\n"
        )

        f.write(
            "No scientific model selection or HPO was performed.\n"
        )

        f.write(
            "Temporary smoke normalization must not be reused in 11C2.\n\n"
        )

        f.write(
            "Architecture audit\n"
        )

        f.write(
            "-" * 116
            + "\n"
        )

        f.write(
            architecture_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nGradient audit\n"
        )

        f.write(
            "-" * 116
            + "\n"
        )

        f.write(
            gradient_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nPhase loss audit\n"
        )

        f.write(
            "-" * 116
            + "\n"
        )

        f.write(
            phase_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nSmoke results (NOT scientific performance)\n"
        )

        f.write(
            "-" * 116
            + "\n"
        )

        f.write(
            smoke_df.to_string(
                index=False
            )
        )

        f.write(
            "\n"
        )

    log("")
    log(
        "="
        * 116
    )

    log(
        "11C1 BP-STGNN ARCHITECTURE SMOKE RESULTS"
    )

    log(
        "="
        * 116
    )

    for row in architecture_df.itertuples():
        smoke_row = smoke_df.loc[
            smoke_df[
                "track"
            ] == row.track
        ].iloc[
            0
        ]

        log(
            f"Track {row.track}: "
            f"params={row.trainable_parameters_anchor:,} | "
            f"mask={row.physical_mask_ones}/100 | "
            f"KL={'PASS' if row.KL_formula_pass else 'FAIL'} | "
            f"architecture={'PASS' if row.all_architecture_pass else 'FAIL'}"
        )

        log(
            f"  smoke loss: "
            f"{smoke_row['first_loss']:.6f} -> "
            f"{smoke_row['last_loss']:.6f}"
        )

        log(
            f"  smoke val vector RMSE [NOT SCIENTIFIC]: "
            f"{smoke_row['initial_val_wind_vector_RMSE_mps']:.4f} -> "
            f"{smoke_row['final_val_wind_vector_RMSE_mps']:.4f} m/s"
        )

    log("")
    log(
        f"[STATUS] {'PASSED' if all_pass else 'FAILED'}"
    )

    log(
        "[POLICY] No SD1090 test or SD1033 external data were accessed."
    )

    log(
        "[POLICY] Smoke hyperparameters are NOT selected/frozen for Stage 11C2."
    )

    log(
        "[POLICY] Smoke validation metrics are NOT scientific performance results."
    )

    log(
        f"[DONE] 11C1 outputs: {output_dir}"
    )

    if not all_pass:
        return 2

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
