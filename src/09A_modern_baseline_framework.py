# -*- coding: utf-8 -*-
"""
09A_modern_baseline_framework.py

Unified, leakage-safe architecture and smoke-validation framework for four
modern time-series forecasting comparison models:

    1. DLinear
    2. iTransformer
    3. TimeMixer
    4. PatchTST

IMPORTANT SCIENTIFIC ROLE
-------------------------
The proposed Physics-Compact-Vessel-Residual model is FROZEN before Stage 09.
The four models in this script are COMPARISON BASELINES ONLY.

Stage 09 must never use SD1033 external performance to:
    - alter the proposed model,
    - tune any modern baseline,
    - select a modern-baseline checkpoint,
    - change the data preprocessing,
    - change the target definition.

This 09A script does NOT perform the final hyperparameter search. Its purpose
is to establish one unified PyTorch implementation/API, verify architecture
integrity, verify forward/backward compatibility with the frozen joint dataset,
and freeze the search-space contract that will be used by 09B.

All four models receive exactly the same frozen input tensor:
    X: [batch, 60, 11]

and predict exactly the same joint target:
    y: [batch, 5, 6]

Target channels:
    UWND_MEAN
    VWND_MEAN
    VESSEL_EAST_MPS
    VESSEL_NORTH_MPS
    HDG_sin
    HDG_cos

Forecast horizons:
    [1, 2, 3, 5, 10] min

The models are evaluated later using the same deterministic apparent-wind
reconstruction:
    A_E = U - V_ship,E
    A_N = V - V_ship,N

and the same AW/AWS/AWA metrics as the frozen development pipeline.

Why "task-adapted" implementations?
-----------------------------------
The published DLinear, iTransformer, TimeMixer and PatchTST architectures were
designed primarily for conventional multivariate forecasting where input and
output variates usually share the same semantic channels and forecast horizons
are typically contiguous. The present task instead has:
    - 11 historical input features,
    - 6 physically constructed future target channels,
    - irregular horizons [1,2,3,5,10].

Therefore, this script preserves each paper's defining representation/backbone
idea while replacing only the final prediction interface by a common joint
multi-target head. This makes the comparison task-compatible and avoids giving
one model a different prediction target.

Core architectural principles retained
---------------------------------------
DLinear:
    moving-average trend/seasonal decomposition + linear temporal projection.

iTransformer:
    each variate history is embedded as one token; self-attention acts across
    variate tokens rather than temporal tokens.

TimeMixer:
    multiscale temporal views + seasonal/trend decomposition + directional
    cross-scale mixing + multi-predictor fusion.

PatchTST:
    channel-independent patch tokenization + shared Transformer encoder over
    temporal patches.

This script is self-contained except that it imports the existing frozen
05C-v2 utility script for dataset I/O, scaler parsing, inverse transforms and
the established apparent-wind metric definitions. This guarantees that Stage
09 uses the same dataset interpretation as earlier baselines.

Default Windows project paths
-----------------------------
Dataset:
D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\
SD1090_TPOS2024_JointForecasting_v0_1

05C utility:
place 09A in the same src folder as
05C_tune_joint_deep_baselines_pytorch_v2.py

Recommended environment:
    conda activate WindPredict

Outputs
-------
09A_modern_baseline_framework_v0_1/
    architecture_audit.csv
    smoke_training_results.csv
    frozen_search_space.json
    framework_manifest.json
    framework_report.txt

No model from 09A is used as a final paper checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
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


SCRIPT_VERSION = "0.1.0-modern-baseline-framework"
UTILITY_SCRIPT = "05C_tune_joint_deep_baselines_pytorch_v2.py"

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

EXPECTED_HORIZONS = [1, 2, 3, 5, 10]

MODEL_ORDER = [
    "DLinear",
    "iTransformer",
    "TimeMixer",
    "PatchTST",
]

DEFAULT_DATASET_DIR = (
    r"D:\project\WindPredict_SaildroneData\data\forecasting"
    r"\SD1090_TPOS2024_JointForecasting_v0_1"
)

# ---------------------------------------------------------------------------
# Frozen Stage-09B search-space contract.
#
# 09B may search ONLY these values unless a new protocol version is declared
# before any external evaluation. SD1033 must NEVER be used to alter them.
# ---------------------------------------------------------------------------
FROZEN_SEARCH_SPACE = {
    "DLinear": {
        "moving_average_kernel": [15, 25, 35],
        "individual_temporal_heads": [False, True],
        "learning_rate": [3e-4, 1e-3],
        "weight_decay": [0.0, 1e-4],
    },
    "iTransformer": {
        "d_model": [32, 64, 96],
        "n_heads": [4, 8],
        "e_layers": [1, 2],
        "d_ff_multiplier": [2, 4],
        "dropout": [0.0, 0.1, 0.2],
        "learning_rate": [3e-4, 1e-3],
    },
    "TimeMixer": {
        "d_model": [32, 64],
        "n_blocks": [1, 2],
        "downsampling_layers": [2, 3],
        "moving_average_kernel": [5, 9],
        "dropout": [0.0, 0.1],
        "learning_rate": [3e-4, 1e-3],
    },
    "PatchTST": {
        "patch_len": [6, 10, 12, 15],
        "stride": [3, 5, 6],
        "d_model": [32, 64, 96],
        "n_heads": [4, 8],
        "e_layers": [1, 2],
        "d_ff_multiplier": [2, 4],
        "dropout": [0.0, 0.1, 0.2],
        "learning_rate": [3e-4, 1e-3],
    },
}

# Anchor configs are deliberately moderate and are used only by 09A smoke test.
ANCHOR_CONFIGS = {
    "DLinear": {
        "moving_average_kernel": 25,
        "individual_temporal_heads": False,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
    },
    "iTransformer": {
        "d_model": 64,
        "n_heads": 4,
        "e_layers": 1,
        "d_ff_multiplier": 2,
        "dropout": 0.1,
        "learning_rate": 1e-3,
    },
    "TimeMixer": {
        "d_model": 64,
        "n_blocks": 1,
        "downsampling_layers": 2,
        "moving_average_kernel": 5,
        "dropout": 0.1,
        "learning_rate": 1e-3,
    },
    "PatchTST": {
        "patch_len": 10,
        "stride": 5,
        "d_model": 64,
        "n_heads": 4,
        "e_layers": 1,
        "d_ff_multiplier": 2,
        "dropout": 0.1,
        "learning_rate": 1e-3,
    },
}


def log(message=""):
    print(message, flush=True)


def save_json(path: Path, obj):
    def convert(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, np.ndarray):
            return [convert(x) for x in v.tolist()]
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return None if np.isnan(v) else float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, dict):
            return {
                str(k): convert(x)
                for k, x in v.items()
            }
        if isinstance(v, (list, tuple)):
            return [convert(x) for x in v]
        return v

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            convert(obj),
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_utility():
    path = (
        Path(__file__).resolve().parent
        / UTILITY_SCRIPT
    )

    if not path.exists():
        raise FileNotFoundError(
            "Required frozen utility script not found:\n"
            f"{path}\n"
            "Place 09A in the same src directory as "
            f"{UTILITY_SCRIPT}."
        )

    spec = importlib.util.spec_from_file_location(
        "modern_baseline_05c_utility",
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to import utility script: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module, path


def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required. Activate the WindPredict environment."
        ) from exc

    return torch, nn, F, DataLoader, TensorDataset


def set_global_seed(torch, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return int(
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )


def parse_models(text: str):
    requested = []

    lookup = {
        name.lower(): name
        for name in MODEL_ORDER
    }

    for token in text.split(","):
        token = token.strip()

        if not token:
            continue

        key = token.lower()

        if key not in lookup:
            raise ValueError(
                f"Unsupported model '{token}'. "
                f"Allowed: {MODEL_ORDER}"
            )

        canonical = lookup[key]

        if canonical not in requested:
            requested.append(canonical)

    if not requested:
        raise ValueError("No models selected.")

    return requested


def validate_frozen_dataset(
    utility,
    dataset_dir: Path,
):
    manifest_path = (
        dataset_dir
        / "dataset_manifest.json"
    )

    scalers_path = (
        dataset_dir
        / "scalers.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    if not scalers_path.exists():
        raise FileNotFoundError(scalers_path)

    flags = utility.find_validation_flags(
        dataset_dir
    )

    if not flags:
        raise RuntimeError(
            "No VALIDATION_PASSED.flag found. "
            "Stage 09 requires the independently validated frozen dataset."
        )

    manifest = utility.load_json(
        manifest_path
    )

    scalers = utility.load_json(
        scalers_path
    )

    if manifest.get("task_name") != EXPECTED_TASK_NAME:
        raise RuntimeError(
            f"Unexpected task_name: {manifest.get('task_name')}"
        )

    feature_names = list(
        manifest["feature_names"]
    )

    target_names = list(
        manifest["target_names"]
    )

    horizons = [
        int(v)
        for v in manifest[
            "forecast_horizons_minutes"
        ]
    ]

    lookback = int(
        manifest["lookback_minutes"]
    )

    if feature_names != EXPECTED_FEATURE_NAMES:
        raise RuntimeError(
            "Frozen feature schema mismatch."
        )

    if target_names != EXPECTED_TARGET_NAMES:
        raise RuntimeError(
            "Frozen target schema mismatch."
        )

    if horizons != EXPECTED_HORIZONS:
        raise RuntimeError(
            "Frozen horizon schema mismatch."
        )

    if lookback != 60:
        raise RuntimeError(
            f"Expected lookback=60, got {lookback}."
        )

    (
        feature_scaler_names,
        feature_mean,
        feature_std,
    ) = utility.parse_scaler(
        scalers,
        "feature_scaler",
    )

    (
        target_scaler_names,
        target_mean,
        target_std,
    ) = utility.parse_scaler(
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

    train = utility.load_npz_split(
        dataset_dir,
        "train",
    )

    validation = utility.load_npz_split(
        dataset_dir,
        "validation",
    )

    expected_x_shape = (
        60,
        len(feature_names),
    )

    expected_y_shape = (
        len(horizons),
        len(target_names),
    )

    for split_name, split in [
        ("train", train),
        ("validation", validation),
    ]:
        if tuple(
            split["X"].shape[1:]
        ) != expected_x_shape:
            raise RuntimeError(
                f"{split_name} X shape mismatch: "
                f"{split['X'].shape}"
            )

        if tuple(
            split["y"].shape[1:]
        ) != expected_y_shape:
            raise RuntimeError(
                f"{split_name} y shape mismatch: "
                f"{split['y'].shape}"
            )

        if not np.isfinite(
            split["X"]
        ).all():
            raise RuntimeError(
                f"{split_name} X contains NaN/Inf."
            )

        if not np.isfinite(
            split["y"]
        ).all():
            raise RuntimeError(
                f"{split_name} y contains NaN/Inf."
            )

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "scalers_path": scalers_path,
        "feature_names": feature_names,
        "target_names": target_names,
        "horizons": horizons,
        "lookback": lookback,
        "feature_mean": np.asarray(
            feature_mean,
            dtype=np.float64,
        ),
        "feature_std": np.asarray(
            feature_std,
            dtype=np.float64,
        ),
        "target_mean": np.asarray(
            target_mean,
            dtype=np.float64,
        ),
        "target_std": np.asarray(
            target_std,
            dtype=np.float64,
        ),
        "train": train,
        "validation": validation,
        "validation_flags": [
            str(p)
            for p in flags
        ],
    }


def build_model_classes(torch, nn, F):
    class MovingAverage(nn.Module):
        def __init__(
            self,
            kernel_size: int,
        ):
            super().__init__()

            if kernel_size <= 0:
                raise ValueError(
                    "kernel_size must be positive."
                )

            if kernel_size % 2 == 0:
                raise ValueError(
                    "Moving-average kernel must be odd."
                )

            self.kernel_size = int(
                kernel_size
            )

        def forward(self, x):
            # x: [B,L,C]
            pad = (
                self.kernel_size
                - 1
            ) // 2

            xt = x.transpose(
                1,
                2,
            )

            xt = F.pad(
                xt,
                (
                    pad,
                    pad,
                ),
                mode="replicate",
            )

            trend = F.avg_pool1d(
                xt,
                kernel_size=self.kernel_size,
                stride=1,
            )

            return trend.transpose(
                1,
                2,
            )


    class DLinearJoint(nn.Module):
        """
        Task-adapted DLinear.

        Retained principle:
            explicit moving-average decomposition
            + linear temporal forecasting.

        Adaptation:
            because input channels (11) and target channels (6) differ,
            temporal forecasts of all decomposed input channels are mapped
            by a shared linear cross-channel head to the common 6-channel
            joint target.
        """
        def __init__(
            self,
            seq_len,
            n_features,
            n_horizons,
            n_targets,
            moving_average_kernel=25,
            individual_temporal_heads=False,
        ):
            super().__init__()

            self.seq_len = int(seq_len)
            self.n_features = int(
                n_features
            )
            self.n_horizons = int(
                n_horizons
            )
            self.n_targets = int(
                n_targets
            )
            self.individual = bool(
                individual_temporal_heads
            )

            self.decomp = MovingAverage(
                moving_average_kernel
            )

            if self.individual:
                self.seasonal_time = (
                    nn.ModuleList([
                        nn.Linear(
                            self.seq_len,
                            self.n_horizons,
                        )
                        for _ in range(
                            self.n_features
                        )
                    ])
                )

                self.trend_time = (
                    nn.ModuleList([
                        nn.Linear(
                            self.seq_len,
                            self.n_horizons,
                        )
                        for _ in range(
                            self.n_features
                        )
                    ])
                )
            else:
                self.seasonal_time = nn.Linear(
                    self.seq_len,
                    self.n_horizons,
                )

                self.trend_time = nn.Linear(
                    self.seq_len,
                    self.n_horizons,
                )

            self.target_head = nn.Linear(
                2 * self.n_features,
                self.n_targets,
            )

        def _project_time(
            self,
            x,
            layers,
        ):
            # x [B,C,L]
            if not self.individual:
                return layers(x)

            out = []

            for channel, layer in enumerate(
                layers
            ):
                out.append(
                    layer(
                        x[
                            :,
                            channel,
                            :
                        ]
                    ).unsqueeze(
                        1
                    )
                )

            return torch.cat(
                out,
                dim=1,
            )

        def forward(self, x):
            trend = self.decomp(x)
            seasonal = x - trend

            seasonal = seasonal.transpose(
                1,
                2,
            )

            trend = trend.transpose(
                1,
                2,
            )

            seasonal_h = self._project_time(
                seasonal,
                self.seasonal_time,
            )

            trend_h = self._project_time(
                trend,
                self.trend_time,
            )

            # [B,C,H] -> [B,H,C]
            seasonal_h = seasonal_h.transpose(
                1,
                2,
            )

            trend_h = trend_h.transpose(
                1,
                2,
            )

            fused = torch.cat(
                [
                    seasonal_h,
                    trend_h,
                ],
                dim=-1,
            )

            return self.target_head(
                fused
            )


    class iTransformerJoint(nn.Module):
        """
        Task-adapted iTransformer.

        Retained principle:
            each variate history is one token and self-attention operates
            across variates.

        Adaptation:
            encoded variate tokens are fused by a common joint-output head
            because the 6 targets are not identical to the 11 input variates.
        """
        def __init__(
            self,
            seq_len,
            n_features,
            n_horizons,
            n_targets,
            d_model=64,
            n_heads=4,
            e_layers=1,
            d_ff_multiplier=2,
            dropout=0.1,
        ):
            super().__init__()

            if d_model % n_heads != 0:
                raise ValueError(
                    "d_model must be divisible by n_heads."
                )

            self.n_features = int(
                n_features
            )
            self.n_horizons = int(
                n_horizons
            )
            self.n_targets = int(
                n_targets
            )

            self.variate_embedding = (
                nn.Linear(
                    int(seq_len),
                    int(d_model),
                )
            )

            encoder_layer = (
                nn.TransformerEncoderLayer(
                    d_model=int(d_model),
                    nhead=int(n_heads),
                    dim_feedforward=(
                        int(d_model)
                        * int(
                            d_ff_multiplier
                        )
                    ),
                    dropout=float(
                        dropout
                    ),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )

            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(
                    e_layers
                ),
            )

            self.norm = nn.LayerNorm(
                int(d_model)
            )

            hidden = max(
                64,
                int(d_model),
            )

            self.head = nn.Sequential(
                nn.Linear(
                    self.n_features
                    * int(d_model),
                    hidden,
                ),
                nn.GELU(),
                nn.Dropout(
                    float(
                        dropout
                    )
                ),
                nn.Linear(
                    hidden,
                    self.n_horizons
                    * self.n_targets,
                ),
            )

        def forward(self, x):
            # x [B,L,C] -> variate tokens [B,C,L]
            tokens = x.transpose(
                1,
                2,
            )

            tokens = self.variate_embedding(
                tokens
            )

            tokens = self.encoder(
                tokens
            )

            tokens = self.norm(
                tokens
            )

            flat = tokens.flatten(
                start_dim=1
            )

            out = self.head(
                flat
            )

            return out.view(
                x.shape[0],
                self.n_horizons,
                self.n_targets,
            )


    class TimeMixerBlock(nn.Module):
        """
        Lightweight task-compatible Past-Decomposable-Mixing block.

        Seasonal information is mixed fine -> coarse.
        Trend information is mixed coarse -> fine.
        """
        def __init__(
            self,
            d_model,
            n_scales,
            dropout,
        ):
            super().__init__()

            self.n_scales = int(
                n_scales
            )

            def mixer():
                return nn.Sequential(
                    nn.Linear(
                        d_model,
                        d_model,
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        dropout
                    ),
                    nn.Linear(
                        d_model,
                        d_model,
                    ),
                )

            self.seasonal_mixers = (
                nn.ModuleList([
                    mixer()
                    for _ in range(
                        max(
                            0,
                            self.n_scales
                            - 1
                        )
                    )
                ])
            )

            self.trend_mixers = (
                nn.ModuleList([
                    mixer()
                    for _ in range(
                        max(
                            0,
                            self.n_scales
                            - 1
                        )
                    )
                ])
            )

            self.seasonal_norms = (
                nn.ModuleList([
                    nn.LayerNorm(
                        d_model
                    )
                    for _ in range(
                        self.n_scales
                    )
                ])
            )

            self.trend_norms = (
                nn.ModuleList([
                    nn.LayerNorm(
                        d_model
                    )
                    for _ in range(
                        self.n_scales
                    )
                ])
            )

        def forward(
            self,
            seasonal,
            trend,
        ):
            s = list(
                seasonal
            )

            t = list(
                trend
            )

            # Fine -> coarse seasonal mixing.
            for i in range(
                self.n_scales
                - 1
            ):
                s[
                    i + 1
                ] = self.seasonal_norms[
                    i + 1
                ](
                    s[
                        i + 1
                    ]
                    + self.seasonal_mixers[
                        i
                    ](
                        s[
                            i
                        ]
                    )
                )

            # Coarse -> fine trend mixing.
            for i in reversed(
                range(
                    self.n_scales
                    - 1
                )
            ):
                t[
                    i
                ] = self.trend_norms[
                    i
                ](
                    t[
                        i
                    ]
                    + self.trend_mixers[
                        i
                    ](
                        t[
                            i + 1
                        ]
                    )
                )

            return s, t


    class TimeMixerJoint(nn.Module):
        """
        Task-adapted TimeMixer.

        Retained principles:
            multiscale temporal observations,
            seasonal/trend decomposition,
            directional cross-scale mixing,
            future multi-predictor fusion.

        Each scale is represented per input variate. The scale predictors
        produce the same common [H,6] joint task and are fused with learned
        softmax weights.
        """
        def __init__(
            self,
            seq_len,
            n_features,
            n_horizons,
            n_targets,
            d_model=64,
            n_blocks=1,
            downsampling_layers=2,
            moving_average_kernel=5,
            dropout=0.1,
        ):
            super().__init__()

            self.seq_len = int(
                seq_len
            )
            self.n_features = int(
                n_features
            )
            self.n_horizons = int(
                n_horizons
            )
            self.n_targets = int(
                n_targets
            )

            self.n_scales = (
                int(
                    downsampling_layers
                )
                + 1
            )

            lengths = [
                self.seq_len
            ]

            for _ in range(
                self.n_scales
                - 1
            ):
                lengths.append(
                    int(
                        math.ceil(
                            lengths[-1]
                            / 2
                        )
                    )
                )

            self.scale_lengths = lengths

            self.decomp = MovingAverage(
                moving_average_kernel
            )

            self.seasonal_embeddings = (
                nn.ModuleList([
                    nn.Linear(
                        length,
                        int(d_model),
                    )
                    for length in lengths
                ])
            )

            self.trend_embeddings = (
                nn.ModuleList([
                    nn.Linear(
                        length,
                        int(d_model),
                    )
                    for length in lengths
                ])
            )

            self.blocks = (
                nn.ModuleList([
                    TimeMixerBlock(
                        int(d_model),
                        self.n_scales,
                        float(dropout),
                    )
                    for _ in range(
                        int(n_blocks)
                    )
                ])
            )

            predictor_input = (
                2
                * self.n_features
                * int(d_model)
            )

            hidden = max(
                64,
                int(d_model),
            )

            self.scale_predictors = (
                nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(
                            predictor_input,
                            hidden,
                        ),
                        nn.GELU(),
                        nn.Dropout(
                            float(
                                dropout
                            )
                        ),
                        nn.Linear(
                            hidden,
                            self.n_horizons
                            * self.n_targets,
                        ),
                    )
                    for _ in range(
                        self.n_scales
                    )
                ])
            )

            self.scale_logits = nn.Parameter(
                torch.zeros(
                    self.n_scales,
                    dtype=torch.float32,
                )
            )

        def _downsample_views(
            self,
            x,
        ):
            # x [B,L,C]
            views = [
                x
            ]

            current = x.transpose(
                1,
                2,
            )

            for _ in range(
                self.n_scales
                - 1
            ):
                current = F.avg_pool1d(
                    current,
                    kernel_size=2,
                    stride=2,
                    ceil_mode=True,
                )

                views.append(
                    current.transpose(
                        1,
                        2,
                    )
                )

            return views

        def forward(self, x):
            views = self._downsample_views(
                x
            )

            seasonal_latents = []
            trend_latents = []

            for i, view in enumerate(
                views
            ):
                trend = self.decomp(
                    view
                )

                seasonal = view - trend

                # [B,L,C] -> [B,C,L] -> [B,C,D]
                seasonal_latents.append(
                    self.seasonal_embeddings[
                        i
                    ](
                        seasonal.transpose(
                            1,
                            2,
                        )
                    )
                )

                trend_latents.append(
                    self.trend_embeddings[
                        i
                    ](
                        trend.transpose(
                            1,
                            2,
                        )
                    )
                )

            for block in self.blocks:
                (
                    seasonal_latents,
                    trend_latents,
                ) = block(
                    seasonal_latents,
                    trend_latents,
                )

            scale_outputs = []

            for i in range(
                self.n_scales
            ):
                fused = torch.cat(
                    [
                        seasonal_latents[
                            i
                        ],
                        trend_latents[
                            i
                        ],
                    ],
                    dim=-1,
                )

                flat = fused.flatten(
                    start_dim=1
                )

                pred = self.scale_predictors[
                    i
                ](
                    flat
                ).view(
                    x.shape[0],
                    self.n_horizons,
                    self.n_targets,
                )

                scale_outputs.append(
                    pred
                )

            weights = torch.softmax(
                self.scale_logits,
                dim=0,
            )

            out = torch.zeros_like(
                scale_outputs[0]
            )

            for i, pred in enumerate(
                scale_outputs
            ):
                out = (
                    out
                    + weights[
                        i
                    ]
                    * pred
                )

            return out


    class PatchTSTJoint(nn.Module):
        """
        Task-adapted PatchTST.

        Retained principles:
            patch tokenization,
            channel-independent shared Transformer encoder.

        Adaptation:
            channel-level encoded states are fused only in the final common
            joint-output head because the task targets differ semantically
            from the 11 input channels.
        """
        def __init__(
            self,
            seq_len,
            n_features,
            n_horizons,
            n_targets,
            patch_len=10,
            stride=5,
            d_model=64,
            n_heads=4,
            e_layers=1,
            d_ff_multiplier=2,
            dropout=0.1,
        ):
            super().__init__()

            if patch_len > seq_len:
                raise ValueError(
                    "patch_len cannot exceed seq_len."
                )

            if stride <= 0:
                raise ValueError(
                    "stride must be positive."
                )

            if d_model % n_heads != 0:
                raise ValueError(
                    "d_model must be divisible by n_heads."
                )

            self.seq_len = int(
                seq_len
            )
            self.n_features = int(
                n_features
            )
            self.n_horizons = int(
                n_horizons
            )
            self.n_targets = int(
                n_targets
            )
            self.patch_len = int(
                patch_len
            )
            self.stride = int(
                stride
            )

            self.n_patches = (
                1
                + (
                    self.seq_len
                    - self.patch_len
                )
                // self.stride
            )

            if self.n_patches <= 0:
                raise RuntimeError(
                    "No PatchTST patches produced."
                )

            self.patch_embedding = nn.Linear(
                self.patch_len,
                int(d_model),
            )

            self.position_embedding = (
                nn.Parameter(
                    torch.zeros(
                        1,
                        self.n_patches,
                        int(d_model),
                    )
                )
            )

            nn.init.trunc_normal_(
                self.position_embedding,
                std=0.02,
            )

            encoder_layer = (
                nn.TransformerEncoderLayer(
                    d_model=int(d_model),
                    nhead=int(n_heads),
                    dim_feedforward=(
                        int(d_model)
                        * int(
                            d_ff_multiplier
                        )
                    ),
                    dropout=float(
                        dropout
                    ),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )

            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(
                    e_layers
                ),
            )

            self.norm = nn.LayerNorm(
                int(d_model)
            )

            hidden = max(
                64,
                int(d_model),
            )

            self.head = nn.Sequential(
                nn.Linear(
                    self.n_features
                    * int(d_model),
                    hidden,
                ),
                nn.GELU(),
                nn.Dropout(
                    float(
                        dropout
                    )
                ),
                nn.Linear(
                    hidden,
                    self.n_horizons
                    * self.n_targets,
                ),
            )

        def forward(self, x):
            # x [B,L,C] -> [B,C,L]
            xc = x.transpose(
                1,
                2,
            )

            patches = xc.unfold(
                dimension=-1,
                size=self.patch_len,
                step=self.stride,
            )
            # [B,C,P,patch_len]

            B, C, P, K = (
                patches.shape
            )

            if P != self.n_patches:
                raise RuntimeError(
                    f"Unexpected patch count: {P} "
                    f"!= {self.n_patches}"
                )

            tokens = self.patch_embedding(
                patches
            )

            tokens = tokens.view(
                B * C,
                P,
                -1,
            )

            tokens = (
                tokens
                + self.position_embedding
            )

            tokens = self.encoder(
                tokens
            )

            tokens = self.norm(
                tokens
            )

            # Channel-independent temporal representation.
            channel_repr = tokens.mean(
                dim=1
            ).view(
                B,
                C,
                -1,
            )

            flat = channel_repr.flatten(
                start_dim=1
            )

            out = self.head(
                flat
            )

            return out.view(
                B,
                self.n_horizons,
                self.n_targets,
            )


    return {
        "DLinear": DLinearJoint,
        "iTransformer": iTransformerJoint,
        "TimeMixer": TimeMixerJoint,
        "PatchTST": PatchTSTJoint,
    }


def instantiate_model(
    model_classes,
    model_name,
    config,
    *,
    seq_len,
    n_features,
    n_horizons,
    n_targets,
):
    cls = model_classes[
        model_name
    ]

    common = {
        "seq_len": int(
            seq_len
        ),
        "n_features": int(
            n_features
        ),
        "n_horizons": int(
            n_horizons
        ),
        "n_targets": int(
            n_targets
        ),
    }

    if model_name == "DLinear":
        return cls(
            **common,
            moving_average_kernel=int(
                config[
                    "moving_average_kernel"
                ]
            ),
            individual_temporal_heads=bool(
                config[
                    "individual_temporal_heads"
                ]
            ),
        )

    if model_name == "iTransformer":
        return cls(
            **common,
            d_model=int(
                config[
                    "d_model"
                ]
            ),
            n_heads=int(
                config[
                    "n_heads"
                ]
            ),
            e_layers=int(
                config[
                    "e_layers"
                ]
            ),
            d_ff_multiplier=int(
                config[
                    "d_ff_multiplier"
                ]
            ),
            dropout=float(
                config[
                    "dropout"
                ]
            ),
        )

    if model_name == "TimeMixer":
        return cls(
            **common,
            d_model=int(
                config[
                    "d_model"
                ]
            ),
            n_blocks=int(
                config[
                    "n_blocks"
                ]
            ),
            downsampling_layers=int(
                config[
                    "downsampling_layers"
                ]
            ),
            moving_average_kernel=int(
                config[
                    "moving_average_kernel"
                ]
            ),
            dropout=float(
                config[
                    "dropout"
                ]
            ),
        )

    if model_name == "PatchTST":
        return cls(
            **common,
            patch_len=int(
                config[
                    "patch_len"
                ]
            ),
            stride=int(
                config[
                    "stride"
                ]
            ),
            d_model=int(
                config[
                    "d_model"
                ]
            ),
            n_heads=int(
                config[
                    "n_heads"
                ]
            ),
            e_layers=int(
                config[
                    "e_layers"
                ]
            ),
            d_ff_multiplier=int(
                config[
                    "d_ff_multiplier"
                ]
            ),
            dropout=float(
                config[
                    "dropout"
                ]
            ),
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def architecture_audit(
    torch,
    model,
    *,
    model_name,
    device,
    batch_size,
    seq_len,
    n_features,
    n_horizons,
    n_targets,
):
    model.eval()

    X = torch.randn(
        batch_size,
        seq_len,
        n_features,
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        y = model(
            X
        )

    expected_shape = (
        batch_size,
        n_horizons,
        n_targets,
    )

    shape_ok = tuple(
        y.shape
    ) == expected_shape

    finite_ok = bool(
        torch.isfinite(
            y
        ).all().item()
    )

    if not shape_ok:
        raise RuntimeError(
            f"{model_name}: output shape "
            f"{tuple(y.shape)} != {expected_shape}"
        )

    if not finite_ok:
        raise RuntimeError(
            f"{model_name}: non-finite forward output."
        )

    del X, y

    return {
        "model": model_name,
        "parameters": count_parameters(
            model
        ),
        "output_shape_ok": shape_ok,
        "output_all_finite": finite_ok,
        "expected_output_shape": str(
            expected_shape
        ),
    }


def make_smoke_loader(
    torch,
    DataLoader,
    TensorDataset,
    X,
    y,
    *,
    max_samples,
    batch_size,
):
    n = min(
        int(
            max_samples
        ),
        int(
            X.shape[0]
        ),
    )

    dataset = TensorDataset(
        torch.from_numpy(
            np.ascontiguousarray(
                X[
                    :n
                ],
                dtype=np.float32,
            )
        ),
        torch.from_numpy(
            np.ascontiguousarray(
                y[
                    :n
                ],
                dtype=np.float32,
            )
        ),
    )

    return DataLoader(
        dataset,
        batch_size=int(
            batch_size
        ),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def smoke_train(
    utility,
    torch,
    nn,
    *,
    model,
    model_name,
    config,
    train_loader,
    val_loader,
    device,
    target_mean,
    target_std,
    max_steps,
    gradient_clip_norm,
):
    criterion = nn.MSELoss(
        reduction="mean"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            config[
                "learning_rate"
            ]
        ),
        weight_decay=float(
            config.get(
                "weight_decay",
                0.0,
            )
        ),
    )

    model.train()

    step = 0
    loss_start = None
    last_loss = None
    all_finite = True

    t0 = time.perf_counter()

    while step < max_steps:
        progressed = False

        for Xb, yb in train_loader:
            progressed = True

            Xb = Xb.to(
                device
            )

            yb = yb.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(
                Xb
            )

            loss = criterion(
                pred,
                yb,
            )

            if not bool(
                torch.isfinite(
                    loss
                ).item()
            ):
                all_finite = False
                break

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(
                    gradient_clip_norm
                ),
            )

            if not bool(
                torch.isfinite(
                    grad_norm
                ).item()
            ):
                all_finite = False
                break

            optimizer.step()

            value = float(
                loss.detach().item()
            )

            if loss_start is None:
                loss_start = value

            last_loss = value
            step += 1

            if step >= max_steps:
                break

        if not all_finite:
            break

        if not progressed:
            raise RuntimeError(
                "Smoke train loader produced no batches."
            )

    train_seconds = (
        time.perf_counter()
        - t0
    )

    # Validation on the small smoke validation subset.
    model.eval()

    pred_chunks = []
    truth_chunks = []
    val_loss_sum = 0.0
    val_n = 0

    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(
                device
            )

            yb = yb.to(
                device
            )

            pred = model(
                Xb
            )

            loss = criterion(
                pred,
                yb,
            )

            batch_n = int(
                Xb.shape[0]
            )

            val_loss_sum += (
                float(
                    loss.item()
                )
                * batch_n
            )

            val_n += batch_n

            pred_chunks.append(
                pred.detach()
                .float()
                .cpu()
                .numpy()
            )

            truth_chunks.append(
                yb.detach()
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

    pred_raw = utility.inverse_joint_target(
        pred_z,
        target_mean,
        target_std,
    )

    truth_raw = utility.inverse_joint_target(
        truth_z,
        target_mean,
        target_std,
    )

    mean_aw_rmse, per_horizon_aw = (
        utility.mean_apparent_vector_rmse(
            truth_raw,
            pred_raw,
        )
    )

    return {
        "model": model_name,
        "smoke_steps_requested": int(
            max_steps
        ),
        "smoke_steps_completed": int(
            step
        ),
        "loss_all_finite": bool(
            all_finite
        ),
        "first_train_loss": (
            float(
                loss_start
            )
            if loss_start is not None
            else np.nan
        ),
        "last_train_loss": (
            float(
                last_loss
            )
            if last_loss is not None
            else np.nan
        ),
        "smoke_validation_joint_MSE": float(
            val_loss_sum
            / max(
                val_n,
                1,
            )
        ),
        "smoke_validation_mean_AW_RMSE_mps": float(
            mean_aw_rmse
        ),
        "smoke_validation_AW_1m_mps": float(
            per_horizon_aw[
                0
            ]
        ),
        "smoke_validation_AW_2m_mps": float(
            per_horizon_aw[
                1
            ]
        ),
        "smoke_validation_AW_3m_mps": float(
            per_horizon_aw[
                2
            ]
        ),
        "smoke_validation_AW_5m_mps": float(
            per_horizon_aw[
                3
            ]
        ),
        "smoke_validation_AW_10m_mps": float(
            per_horizon_aw[
                4
            ]
        ),
        "training_seconds": float(
            train_seconds
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--models",
        default=",".join(
            MODEL_ORDER
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260815,
    )

    parser.add_argument(
        "--audit-batch-size",
        type=int,
        default=8,
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
        "--smoke-batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--skip-smoke-training",
        action="store_true",
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
    )

    args = parser.parse_args()

    dataset_dir = Path(
        args.dataset_dir
    )

    output_dir = (
        Path(
            args.output_dir
        )
        if args.output_dir
        else (
            dataset_dir
            / "09A_modern_baseline_framework_v0_1"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    models = parse_models(
        args.models
    )

    utility, utility_path = (
        load_utility()
    )

    (
        torch,
        nn,
        F,
        DataLoader,
        TensorDataset,
    ) = import_torch()

    set_global_seed(
        torch,
        args.seed,
    )

    frozen = validate_frozen_dataset(
        utility,
        dataset_dir,
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

    model_classes = build_model_classes(
        torch,
        nn,
        F,
    )

    log(
        "=" * 110
    )

    log(
        "09A - MODERN BASELINE UNIFIED FRAMEWORK"
    )

    log(
        "=" * 110
    )

    log(
        f"script_version         : {SCRIPT_VERSION}"
    )

    log(
        f"dataset_dir            : {dataset_dir}"
    )

    log(
        f"utility                 : {utility_path}"
    )

    log(
        f"models                  : {models}"
    )

    log(
        f"input shape             : "
        f"[B,{frozen['lookback']},{len(frozen['feature_names'])}]"
    )

    log(
        f"target shape            : "
        f"[B,{len(frozen['horizons'])},{len(frozen['target_names'])}]"
    )

    log(
        f"horizons                : {frozen['horizons']} min"
    )

    log(
        f"device                  : {device}"
    )

    if device.type == "cuda":
        log(
            f"GPU                     : "
            f"{torch.cuda.get_device_name(device)}"
        )

    log(
        "SD1033 external data    : NOT ACCESSED"
    )

    log(
        "proposed model          : FROZEN / NOT MODIFIED"
    )

    log("")

    architecture_rows = []
    smoke_rows = []

    train_loader = None
    val_loader = None

    if not args.skip_smoke_training:
        train_loader = make_smoke_loader(
            torch,
            DataLoader,
            TensorDataset,
            frozen[
                "train"
            ][
                "X"
            ],
            frozen[
                "train"
            ][
                "y"
            ],
            max_samples=args.smoke_train_samples,
            batch_size=args.smoke_batch_size,
        )

        val_loader = make_smoke_loader(
            torch,
            DataLoader,
            TensorDataset,
            frozen[
                "validation"
            ][
                "X"
            ],
            frozen[
                "validation"
            ][
                "y"
            ],
            max_samples=args.smoke_val_samples,
            batch_size=args.smoke_batch_size,
        )

    for model_index, model_name in enumerate(
        models,
        start=1,
    ):
        log(
            "-" * 110
        )

        log(
            f"[{model_index}/{len(models)}] {model_name}"
        )

        config = dict(
            ANCHOR_CONFIGS[
                model_name
            ]
        )

        set_global_seed(
            torch,
            args.seed,
        )

        model = instantiate_model(
            model_classes,
            model_name,
            config,
            seq_len=frozen[
                "lookback"
            ],
            n_features=len(
                frozen[
                    "feature_names"
                ]
            ),
            n_horizons=len(
                frozen[
                    "horizons"
                ]
            ),
            n_targets=len(
                frozen[
                    "target_names"
                ]
            ),
        ).to(
            device
        )

        audit = architecture_audit(
            torch,
            model,
            model_name=model_name,
            device=device,
            batch_size=args.audit_batch_size,
            seq_len=frozen[
                "lookback"
            ],
            n_features=len(
                frozen[
                    "feature_names"
                ]
            ),
            n_horizons=len(
                frozen[
                    "horizons"
                ]
            ),
            n_targets=len(
                frozen[
                    "target_names"
                ]
            ),
        )

        audit[
            "anchor_config_json"
        ] = json.dumps(
            config,
            ensure_ascii=False,
            sort_keys=True,
        )

        if model_name == "PatchTST":
            audit[
                "derived_patch_count"
            ] = int(
                model.n_patches
            )

        elif model_name == "TimeMixer":
            audit[
                "derived_scale_lengths"
            ] = str(
                model.scale_lengths
            )

        architecture_rows.append(
            audit
        )

        log(
            f"  parameters             = "
            f"{audit['parameters']:,}"
        )

        log(
            f"  forward shape audit    = PASS"
        )

        if not args.skip_smoke_training:
            smoke = smoke_train(
                utility,
                torch,
                nn,
                model=model,
                model_name=model_name,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                target_mean=frozen[
                    "target_mean"
                ],
                target_std=frozen[
                    "target_std"
                ],
                max_steps=args.smoke_steps,
                gradient_clip_norm=args.gradient_clip_norm,
            )

            smoke_rows.append(
                smoke
            )

            log(
                f"  smoke train steps      = "
                f"{smoke['smoke_steps_completed']}/"
                f"{smoke['smoke_steps_requested']}"
            )

            log(
                f"  first -> last MSE      = "
                f"{smoke['first_train_loss']:.6f} -> "
                f"{smoke['last_train_loss']:.6f}"
            )

            log(
                f"  smoke val AW RMSE      = "
                f"{smoke['smoke_validation_mean_AW_RMSE_mps']:.4f} m/s"
            )

            if not smoke[
                "loss_all_finite"
            ]:
                raise RuntimeError(
                    f"{model_name}: smoke training produced "
                    "non-finite loss/gradient."
                )

        del model
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    architecture_df = pd.DataFrame(
        architecture_rows
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

    smoke_df.to_csv(
        output_dir
        / "smoke_training_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        output_dir
        / "frozen_search_space.json",
        {
            "protocol_version": (
                "09B-search-space-v0.1-frozen-before-search"
            ),
            "search_space": FROZEN_SEARCH_SPACE,
            "anchor_configs": ANCHOR_CONFIGS,
            "selection_rule_planned_for_09B": (
                "rolling-origin CV on frozen SD1090 OUTER TRAIN only; "
                "rank by mean apparent-wind vector RMSE"
            ),
            "external_data_may_modify_search_space": False,
            "SD1090_outer_validation_used_for_search": False,
            "SD1090_test_used_for_search": False,
            "SD1033_used_for_search": False,
        },
    )

    manifest = {
        "script_version": SCRIPT_VERSION,
        "stage": "09A",
        "purpose": (
            "Unified architecture/API and smoke validation for modern "
            "comparison baselines."
        ),
        "scientific_role": (
            "comparison baselines only; proposed Physics-Compact model frozen"
        ),
        "dataset_dir": str(
            dataset_dir
        ),
        "dataset_manifest": str(
            frozen[
                "manifest_path"
            ]
        ),
        "utility_script": str(
            utility_path
        ),
        "models": models,
        "task": {
            "input_shape": [
                frozen[
                    "lookback"
                ],
                len(
                    frozen[
                        "feature_names"
                    ]
                ),
            ],
            "target_shape": [
                len(
                    frozen[
                        "horizons"
                    ]
                ),
                len(
                    frozen[
                        "target_names"
                    ]
                ),
            ],
            "feature_names": frozen[
                "feature_names"
            ],
            "target_names": frozen[
                "target_names"
            ],
            "horizons_min": frozen[
                "horizons"
            ],
            "training_loss_for_09B": (
                "standardized joint MSE"
            ),
            "selection_metric_for_09B": (
                "mean apparent-wind vector RMSE"
            ),
        },
        "fair_comparison_contract": {
            "same_input_dataset": True,
            "same_train_only_scalers": True,
            "same_target_channels": True,
            "same_forecast_horizons": True,
            "same_apparent_wind_reconstruction": True,
            "same_metrics": True,
            "same_outer_chronological_splits": True,
            "SD1033_external_data_accessed": False,
            "proposed_model_modified": False,
        },
        "implementation_notes": {
            "DLinear": (
                "Core trend/seasonal decomposition and linear temporal "
                "projection retained; common 6-target head added."
            ),
            "iTransformer": (
                "Variate-token inversion and cross-variate Transformer "
                "encoder retained; common 6-target head added."
            ),
            "TimeMixer": (
                "Multiscale views, decomposition, directional cross-scale "
                "mixing and multi-predictor fusion retained; common task head."
            ),
            "PatchTST": (
                "Channel-independent patching and shared temporal Transformer "
                "retained; encoded channels fused only in final 6-target head."
            ),
        },
        "architecture_audit": architecture_rows,
        "smoke_training": smoke_rows,
        "frozen_search_space_file": (
            "frozen_search_space.json"
        ),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": bool(
                torch.cuda.is_available()
            ),
            "device": str(
                device
            ),
        },
    }

    save_json(
        output_dir
        / "framework_manifest.json",
        manifest,
    )

    with (
        output_dir
        / "framework_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "09A Modern Forecasting Baseline Framework Report\n"
        )
        f.write(
            "=" * 100
            + "\n\n"
        )
        f.write(
            "Scientific policy\n"
        )
        f.write(
            "-" * 100
            + "\n"
        )
        f.write(
            "The Physics-Compact-Vessel-Residual proposed model is frozen.\n"
        )
        f.write(
            "DLinear, iTransformer, TimeMixer and PatchTST are comparison "
            "baselines only.\n"
        )
        f.write(
            "No SD1033 external data are read or used in Stage 09A/09B "
            "model selection.\n"
        )
        f.write(
            "All four baselines use the same X:[60,11] -> y:[5,6] task.\n\n"
        )
        f.write(
            "Architecture audit\n"
        )
        f.write(
            "-" * 100
            + "\n"
        )
        f.write(
            architecture_df.to_string(
                index=False
            )
        )
        f.write(
            "\n\nSmoke-training results\n"
        )
        f.write(
            "-" * 100
            + "\n"
        )
        if smoke_df.empty:
            f.write(
                "Smoke training skipped.\n"
            )
        else:
            f.write(
                smoke_df.to_string(
                    index=False
                )
            )
            f.write("\n")
        f.write(
            "\nNext stage\n"
        )
        f.write(
            "-" * 100
            + "\n"
        )
        f.write(
            "09B will perform controlled rolling-origin hyperparameter "
            "selection ONLY within the frozen SD1090 outer TRAIN split. "
            "The search space is frozen in frozen_search_space.json.\n"
        )

    log("")
    log(
        "=" * 110
    )

    log(
        "09A MODERN BASELINE FRAMEWORK RESULTS"
    )

    log(
        "=" * 110
    )

    for row in architecture_rows:
        log(
            f"{row['model']}: "
            f"params={row['parameters']:,} | "
            f"forward=PASS | finite=PASS"
        )

    if smoke_rows:
        log("")
        for row in smoke_rows:
            log(
                f"{row['model']}: "
                f"smoke steps={row['smoke_steps_completed']} | "
                f"loss {row['first_train_loss']:.5f}"
                f" -> {row['last_train_loss']:.5f} | "
                f"val AW={row['smoke_validation_mean_AW_RMSE_mps']:.4f} m/s"
            )

    log("")
    log(
        "[FREEZE] Stage-09B search space written before tuning."
    )

    log(
        "[POLICY] Physics-Compact proposed model remains frozen."
    )

    log(
        "[POLICY] SD1033 external data were not accessed."
    )

    log(
        f"[DONE] 09A outputs: {output_dir}"
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
        sys.exit(1)
