# -*- coding: utf-8 -*-
"""
09B_tune_modern_baselines_rolling_cv.py

Leakage-safe rolling-origin hyperparameter screening for:
    DLinear
    iTransformer
    TimeMixer
    PatchTST

This script is Stage 09B of the modern-baseline comparison.

SCIENTIFIC POLICY
-----------------
The proposed Physics-Compact-Vessel-Residual model is already frozen and is
NOT modified here.

09B uses ONLY the frozen SD1090 OUTER TRAIN split. It does NOT access:
    - SD1090 outer validation for hyperparameter ranking,
    - SD1090 development test,
    - any SD1033 external dataset.

The model-selection objective is:
    mean apparent-wind vector RMSE across 3 expanding rolling-origin folds.

Training loss for every modern baseline is exactly the same:
    standardized six-target joint MSE.

No model gets a different target definition or a physics loss.

ROLLING-ORIGIN PROTOCOL
-----------------------
The frozen outer-train set contains accepted forecasting windows in strict
chronological order. The three folds reproduce the previously frozen Stage-05C
logic:

    total outer-train windows = N
    chunk = N / 6

    Fold 1:
        train = first 3*chunk samples, minus 10-sample purge
        validation = next chunk

    Fold 2:
        train expands by one chunk, minus 10-sample purge
        validation = next chunk

    Fold 3:
        train expands again, minus 10-sample purge
        validation = final chunk

For N=77,934 this yields:
    Fold 1 train = 38,957, validation = 12,989
    Fold 2 train = 51,946, validation = 12,989
    Fold 3 train = 64,935, validation = 12,989

The 10-sample purge equals the maximum 10-min forecast horizon.

Strict timestamp audit is performed when the accepted-window NPZ contains
context/target timestamps:
    max(training target time) < min(validation context time)

SEARCH DESIGN
-------------
09A froze the allowed search-space ranges before tuning. 09B uses a compact,
pre-declared screening set of six representative configurations per model.
The configurations are deliberately chosen to cover model size, receptive
structure and regularization without performing an exhaustive Cartesian grid.

Default screening:
    4 models x 6 configs x 3 folds = 72 training runs.

This is only configuration screening. 09C will refit/confirm the selected
configuration for each model on the complete SD1090 outer-train data and use
the frozen outer validation set for final checkpointing/comparison.

EARLY STOPPING
--------------
Per fold:
    optimizer: Adam
    training objective: standardized joint MSE
    LR scheduler monitor: validation joint MSE
    checkpoint monitor: validation mean AW vector RMSE
    max epochs: 60
    patience: 8
    gradient clipping: 1.0

No fold checkpoint is promoted directly to the paper comparison.

RESUMABILITY
------------
Each completed candidate/fold run is appended immediately to:
    screening_fold_results.csv

If the process is interrupted, rerunning with --resume skips already completed
candidate/fold combinations. This is recommended for long GPU runs.

DEPENDENCIES
------------
Place in the same src directory as:
    09A_modern_baseline_framework.py
    05C_tune_joint_deep_baselines_pytorch_v2.py

Recommended environment:
    conda activate WindPredict

OUTPUTS
-------
09B_modern_baseline_rolling_cv_v0_1/
    screening_fold_results.csv
    screening_candidate_summary.csv
    selected_configs.json
    rolling_fold_protocol.csv
    selection_manifest.json
    selection_report.txt

No SD1033 result is read or written.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0-modern-baseline-rolling-cv"
SHARED_09A = "09A_modern_baseline_framework.py"

MODEL_ORDER = [
    "DLinear",
    "iTransformer",
    "TimeMixer",
    "PatchTST",
]

HORIZONS = [1, 2, 3, 5, 10]
PURGE_SAMPLES = 10

# ---------------------------------------------------------------------------
# Six pre-declared screening configurations per model.
#
# Every value is contained in the 09A frozen search-space contract.
# Config IDs are stable and should not be changed after 09B starts.
# ---------------------------------------------------------------------------
SCREENING_CONFIGS = {
    "DLinear": [
        {
            "config_id": "D01",
            "moving_average_kernel": 15,
            "individual_temporal_heads": False,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
        },
        {
            "config_id": "D02",
            "moving_average_kernel": 25,
            "individual_temporal_heads": False,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
        },
        {
            "config_id": "D03",
            "moving_average_kernel": 35,
            "individual_temporal_heads": False,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
        },
        {
            "config_id": "D04",
            "moving_average_kernel": 15,
            "individual_temporal_heads": True,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
        },
        {
            "config_id": "D05",
            "moving_average_kernel": 25,
            "individual_temporal_heads": True,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
        },
        {
            "config_id": "D06",
            "moving_average_kernel": 35,
            "individual_temporal_heads": True,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        },
    ],
    "iTransformer": [
        {
            "config_id": "I01",
            "d_model": 32,
            "n_heads": 4,
            "e_layers": 1,
            "d_ff_multiplier": 2,
            "dropout": 0.1,
            "learning_rate": 1e-3,
        },
        {
            "config_id": "I02",
            "d_model": 64,
            "n_heads": 4,
            "e_layers": 1,
            "d_ff_multiplier": 2,
            "dropout": 0.1,
            "learning_rate": 1e-3,
        },
        {
            "config_id": "I03",
            "d_model": 64,
            "n_heads": 4,
            "e_layers": 2,
            "d_ff_multiplier": 4,
            "dropout": 0.1,
            "learning_rate": 3e-4,
        },
        {
            "config_id": "I04",
            "d_model": 96,
            "n_heads": 8,
            "e_layers": 1,
            "d_ff_multiplier": 2,
            "dropout": 0.1,
            "learning_rate": 3e-4,
        },
        {
            "config_id": "I05",
            "d_model": 96,
            "n_heads": 8,
            "e_layers": 2,
            "d_ff_multiplier": 4,
            "dropout": 0.2,
            "learning_rate": 3e-4,
        },
        {
            "config_id": "I06",
            "d_model": 64,
            "n_heads": 8,
            "e_layers": 2,
            "d_ff_multiplier": 2,
            "dropout": 0.0,
            "learning_rate": 1e-3,
        },
    ],
    "TimeMixer": [
        {
            "config_id": "T01",
            "d_model": 32,
            "n_blocks": 1,
            "downsampling_layers": 2,
            "moving_average_kernel": 5,
            "dropout": 0.0,
            "learning_rate": 1e-3,
        },
        {
            "config_id": "T02",
            "d_model": 32,
            "n_blocks": 2,
            "downsampling_layers": 2,
            "moving_average_kernel": 5,
            "dropout": 0.1,
            "learning_rate": 3e-4,
        },
        {
            "config_id": "T03",
            "d_model": 64,
            "n_blocks": 1,
            "downsampling_layers": 2,
            "moving_average_kernel": 5,
            "dropout": 0.1,
            "learning_rate": 1e-3,
        },
        {
            "config_id": "T04",
            "d_model": 64,
            "n_blocks": 2,
            "downsampling_layers": 2,
            "moving_average_kernel": 9,
            "dropout": 0.1,
            "learning_rate": 3e-4,
        },
        {
            "config_id": "T05",
            "d_model": 32,
            "n_blocks": 1,
            "downsampling_layers": 3,
            "moving_average_kernel": 9,
            "dropout": 0.1,
            "learning_rate": 1e-3,
        },
        {
            "config_id": "T06",
            "d_model": 64,
            "n_blocks": 2,
            "downsampling_layers": 3,
            "moving_average_kernel": 5,
            "dropout": 0.0,
            "learning_rate": 3e-4,
        },
    ],
    "PatchTST": [
        {
            "config_id": "P01",
            "patch_len": 6,
            "stride": 3,
            "d_model": 32,
            "n_heads": 4,
            "e_layers": 1,
            "d_ff_multiplier": 2,
            "dropout": 0.1,
            "learning_rate": 1e-3,
        },
        {
            "config_id": "P02",
            "patch_len": 10,
            "stride": 5,
            "d_model": 64,
            "n_heads": 4,
            "e_layers": 1,
            "d_ff_multiplier": 2,
            "dropout": 0.1,
            "learning_rate": 1e-3,
        },
        {
            "config_id": "P03",
            "patch_len": 12,
            "stride": 6,
            "d_model": 64,
            "n_heads": 8,
            "e_layers": 2,
            "d_ff_multiplier": 4,
            "dropout": 0.1,
            "learning_rate": 3e-4,
        },
        {
            "config_id": "P04",
            "patch_len": 15,
            "stride": 5,
            "d_model": 96,
            "n_heads": 8,
            "e_layers": 1,
            "d_ff_multiplier": 2,
            "dropout": 0.1,
            "learning_rate": 3e-4,
        },
        {
            "config_id": "P05",
            "patch_len": 10,
            "stride": 5,
            "d_model": 96,
            "n_heads": 8,
            "e_layers": 2,
            "d_ff_multiplier": 4,
            "dropout": 0.2,
            "learning_rate": 3e-4,
        },
        {
            "config_id": "P06",
            "patch_len": 6,
            "stride": 3,
            "d_model": 64,
            "n_heads": 4,
            "e_layers": 2,
            "d_ff_multiplier": 2,
            "dropout": 0.0,
            "learning_rate": 1e-3,
        },
    ],
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


def load_09a():
    path = (
        Path(__file__).resolve().parent
        / SHARED_09A
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Required 09A script not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        "modern_baseline_09a_shared",
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module, path


def set_seed(torch, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_models(text):
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
                f"Unsupported model {token}. "
                f"Allowed: {MODEL_ORDER}"
            )

        model = lookup[key]

        if model not in requested:
            requested.append(model)

    if not requested:
        raise ValueError(
            "No model selected."
        )

    return requested


def validate_candidates_against_frozen_search(
    candidates,
    frozen_search,
):
    allowed = frozen_search[
        "search_space"
    ]

    for model_name, configs in candidates.items():
        if model_name not in allowed:
            raise RuntimeError(
                f"{model_name} absent from frozen 09A search space."
            )

        allowed_model = allowed[
            model_name
        ]

        for config in configs:
            for key, value in config.items():
                if key == "config_id":
                    continue

                if key not in allowed_model:
                    # weight_decay is intentionally only applicable to DLinear
                    # and is present in the frozen 09A DLinear space.
                    raise RuntimeError(
                        f"{model_name}/{config['config_id']}: "
                        f"key '{key}' absent from frozen search space."
                    )

                choices = allowed_model[
                    key
                ]

                if value not in choices:
                    raise RuntimeError(
                        f"{model_name}/{config['config_id']}: "
                        f"{key}={value} not in frozen choices {choices}."
                    )


def find_timestamp_key(
    split,
    candidates,
):
    for key in candidates:
        if key in split:
            return key
    return None


def build_rolling_folds(
    train_split,
    *,
    purge_samples=10,
):
    n = int(
        train_split[
            "X"
        ].shape[
            0
        ]
    )

    if n % 6 != 0:
        raise RuntimeError(
            f"Outer-train sample count {n} is not divisible by 6. "
            "The frozen 3-fold protocol cannot be reproduced exactly."
        )

    chunk = n // 6

    folds = []

    for fold_id in range(
        1,
        4,
    ):
        validation_start = (
            (
                3
                + (
                    fold_id
                    - 1
                )
            )
            * chunk
        )

        validation_end = (
            validation_start
            + chunk
        )

        train_end_exclusive = (
            validation_start
            - int(
                purge_samples
            )
        )

        if train_end_exclusive <= 0:
            raise RuntimeError(
                "Invalid rolling fold."
            )

        train_indices = np.arange(
            0,
            train_end_exclusive,
            dtype=np.int64,
        )

        validation_indices = np.arange(
            validation_start,
            validation_end,
            dtype=np.int64,
        )

        folds.append({
            "fold_id": fold_id,
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "purge_start_index": int(
                train_end_exclusive
            ),
            "purge_end_index": int(
                validation_start
                - 1
            ),
            "train_count": int(
                len(
                    train_indices
                )
            ),
            "validation_count": int(
                len(
                    validation_indices
                )
            ),
            "validation_start_index": int(
                validation_start
            ),
            "validation_end_index": int(
                validation_end
                - 1
            ),
        })

    return folds


def timestamp_audit(
    train_split,
    folds,
):
    target_key = find_timestamp_key(
        train_split,
        [
            "target_time_ns",
            "target_times_ns",
            "y_time_ns",
        ],
    )

    context_key = find_timestamp_key(
        train_split,
        [
            "context_start_time_ns",
            "context_end_time_ns",
            "input_end_time_ns",
        ],
    )

    audit_available = (
        target_key is not None
        and context_key is not None
    )

    rows = []

    for fold in folds:
        row = {
            "fold_id": fold[
                "fold_id"
            ],
            "train_count": fold[
                "train_count"
            ],
            "validation_count": fold[
                "validation_count"
            ],
            "purge_samples": (
                fold[
                    "validation_start_index"
                ]
                - fold[
                    "train_count"
                ]
            ),
            "timestamp_audit_available": bool(
                audit_available
            ),
        }

        if audit_available:
            train_idx = fold[
                "train_indices"
            ]

            val_idx = fold[
                "validation_indices"
            ]

            target_time = np.asarray(
                train_split[
                    target_key
                ]
            )

            context_time = np.asarray(
                train_split[
                    context_key
                ]
            )

            if target_time.ndim == 2:
                max_train_target = int(
                    np.max(
                        target_time[
                            train_idx
                        ]
                    )
                )
            else:
                max_train_target = int(
                    np.max(
                        target_time[
                            train_idx
                        ]
                    )
                )

            min_val_context = int(
                np.min(
                    context_time[
                        val_idx
                    ]
                )
            )

            row.update({
                "max_train_target_time_ns": (
                    max_train_target
                ),
                "min_validation_context_time_ns": (
                    min_val_context
                ),
                "strict_temporal_separation": bool(
                    max_train_target
                    < min_val_context
                ),
                "separation_seconds": float(
                    (
                        min_val_context
                        - max_train_target
                    )
                    / 1e9
                ),
            })

            if not row[
                "strict_temporal_separation"
            ]:
                raise RuntimeError(
                    f"Fold {fold['fold_id']} timestamp leakage audit failed."
                )

        else:
            row[
                "strict_temporal_separation"
            ] = np.nan

            row[
                "separation_seconds"
            ] = np.nan

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def build_dataset_class(torch):
    from torch.utils.data import Dataset

    class NumpyIndexDataset(Dataset):
        def __init__(
            self,
            X,
            y,
            indices,
        ):
            self.X = X
            self.y = y
            self.indices = np.asarray(
                indices,
                dtype=np.int64,
            )

        def __len__(self):
            return int(
                len(
                    self.indices
                )
            )

        def __getitem__(self, idx):
            source_index = int(
                self.indices[
                    idx
                ]
            )

            x = torch.from_numpy(
                np.asarray(
                    self.X[
                        source_index
                    ],
                    dtype=np.float32,
                )
            )

            y = torch.from_numpy(
                np.asarray(
                    self.y[
                        source_index
                    ],
                    dtype=np.float32,
                )
            )

            return x, y

    return NumpyIndexDataset


def make_loader(
    torch,
    DataLoader,
    DatasetClass,
    X,
    y,
    indices,
    *,
    batch_size,
    shuffle,
    pin_memory,
):
    dataset = DatasetClass(
        X,
        y,
        indices,
    )

    generator = torch.Generator()

    generator.manual_seed(
        1234567
    )

    return DataLoader(
        dataset,
        batch_size=int(
            batch_size
        ),
        shuffle=bool(
            shuffle
        ),
        num_workers=0,
        pin_memory=bool(
            pin_memory
        ),
        drop_last=False,
        generator=generator,
    )


def inverse_target(
    utility,
    y_z,
    target_mean,
    target_std,
):
    # 05C utility has inverse_joint_target in the user's frozen pipeline.
    return utility.inverse_joint_target(
        y_z,
        target_mean,
        target_std,
    )


def evaluate_validation(
    utility,
    torch,
    model,
    loader,
    *,
    device,
    criterion,
    target_mean,
    target_std,
    use_amp,
):
    model.eval()

    pred_chunks = []
    truth_chunks = []

    loss_sum = 0.0
    sample_count = 0

    amp_context = (
        torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=bool(
                use_amp
                and device.type == "cuda"
            ),
        )
        if hasattr(
            torch,
            "autocast"
        )
        else None
    )

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(
                device,
                non_blocking=True,
            )

            yb = yb.to(
                device,
                non_blocking=True,
            )

            if amp_context is None:
                pred = model(
                    Xb
                )
            else:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=bool(
                        use_amp
                        and device.type == "cuda"
                    ),
                ):
                    pred = model(
                        Xb
                    )

            loss = criterion(
                pred,
                yb,
            )

            batch_n = int(
                Xb.shape[
                    0
                ]
            )

            loss_sum += (
                float(
                    loss.item()
                )
                * batch_n
            )

            sample_count += batch_n

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

    pred_raw = inverse_target(
        utility,
        pred_z,
        target_mean,
        target_std,
    )

    truth_raw = inverse_target(
        utility,
        truth_z,
        target_mean,
        target_std,
    )

    mean_aw, per_horizon_aw = (
        utility.mean_apparent_vector_rmse(
            truth_raw,
            pred_raw,
        )
    )

    return {
        "validation_joint_MSE": float(
            loss_sum
            / max(
                sample_count,
                1
            )
        ),
        "validation_AW_RMSE_mps": float(
            mean_aw
        ),
        "validation_AW_1m_mps": float(
            per_horizon_aw[
                0
            ]
        ),
        "validation_AW_2m_mps": float(
            per_horizon_aw[
                1
            ]
        ),
        "validation_AW_3m_mps": float(
            per_horizon_aw[
                2
            ]
        ),
        "validation_AW_5m_mps": float(
            per_horizon_aw[
                3
            ]
        ),
        "validation_AW_10m_mps": float(
            per_horizon_aw[
                4
            ]
        ),
    }


def train_one_fold(
    shared09a,
    utility,
    torch,
    nn,
    DataLoader,
    DatasetClass,
    model_classes,
    *,
    model_name,
    config,
    fold,
    frozen,
    device,
    batch_size,
    max_epochs,
    patience,
    gradient_clip_norm,
    use_amp,
    seed,
):
    set_seed(
        torch,
        seed,
    )

    model = shared09a.instantiate_model(
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

    n_params = shared09a.count_parameters(
        model
    )

    train_loader = make_loader(
        torch,
        DataLoader,
        DatasetClass,
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
        fold[
            "train_indices"
        ],
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    val_loader = make_loader(
        torch,
        DataLoader,
        DatasetClass,
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
        fold[
            "validation_indices"
        ],
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

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

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )
    )

    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=bool(
                use_amp
                and device.type
                == "cuda"
            )
        )
    )

    best_aw = np.inf
    best_val_mse = np.inf
    best_epoch = -1
    best_per_horizon = None

    epochs_without_improvement = 0

    t0 = time.perf_counter()

    for epoch in range(
        1,
        int(
            max_epochs
        )
        + 1,
    ):
        model.train()

        train_loss_sum = 0.0
        train_count = 0

        for Xb, yb in train_loader:
            Xb = Xb.to(
                device,
                non_blocking=True,
            )

            yb = yb.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=bool(
                    use_amp
                    and device.type
                    == "cuda"
                ),
            ):
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
                raise RuntimeError(
                    f"{model_name}/{config['config_id']}/"
                    f"fold{fold['fold_id']}: non-finite training loss."
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(
                        gradient_clip_norm
                    ),
                )
            )

            if not bool(
                torch.isfinite(
                    grad_norm
                ).item()
            ):
                raise RuntimeError(
                    f"{model_name}/{config['config_id']}/"
                    f"fold{fold['fold_id']}: non-finite gradient."
                )

            scaler.step(
                optimizer
            )

            scaler.update()

            batch_n = int(
                Xb.shape[
                    0
                ]
            )

            train_loss_sum += (
                float(
                    loss.detach().item()
                )
                * batch_n
            )

            train_count += batch_n

        train_mse = (
            train_loss_sum
            / max(
                train_count,
                1
            )
        )

        metrics = evaluate_validation(
            utility,
            torch,
            model,
            val_loader,
            device=device,
            criterion=criterion,
            target_mean=frozen[
                "target_mean"
            ],
            target_std=frozen[
                "target_std"
            ],
            use_amp=use_amp,
        )

        scheduler.step(
            metrics[
                "validation_joint_MSE"
            ]
        )

        current_aw = metrics[
            "validation_AW_RMSE_mps"
        ]

        improved = (
            current_aw
            < best_aw
            - 1e-7
        )

        if improved:
            best_aw = float(
                current_aw
            )

            best_val_mse = float(
                metrics[
                    "validation_joint_MSE"
                ]
            )

            best_epoch = int(
                epoch
            )

            best_per_horizon = {
                key: float(
                    value
                )
                for key, value in metrics.items()
                if key.startswith(
                    "validation_AW_"
                )
                and key
                != "validation_AW_RMSE_mps"
            }

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epoch == 1
            or epoch % 10 == 0
            or improved
        ):
            log(
                f"      epoch={epoch:02d} "
                f"trainMSE={train_mse:.5f} "
                f"valMSE={metrics['validation_joint_MSE']:.5f} "
                f"valAW={current_aw:.4f} "
                f"best={best_aw:.4f}"
            )

        if (
            epochs_without_improvement
            >= int(
                patience
            )
        ):
            break

    elapsed = (
        time.perf_counter()
        - t0
    )

    if best_per_horizon is None:
        raise RuntimeError(
            "No valid best epoch found."
        )

    result = {
        "model": model_name,
        "config_id": config[
            "config_id"
        ],
        "fold_id": int(
            fold[
                "fold_id"
            ]
        ),
        "seed": int(
            seed
        ),
        "parameters": int(
            n_params
        ),
        "train_count": int(
            fold[
                "train_count"
            ]
        ),
        "validation_count": int(
            fold[
                "validation_count"
            ]
        ),
        "purge_samples": int(
            PURGE_SAMPLES
        ),
        "best_epoch": int(
            best_epoch
        ),
        "epochs_completed": int(
            epoch
        ),
        "best_validation_joint_MSE": float(
            best_val_mse
        ),
        "best_validation_AW_RMSE_mps": float(
            best_aw
        ),
        "training_seconds": float(
            elapsed
        ),
        "learning_rate_initial": float(
            config[
                "learning_rate"
            ]
        ),
        "config_json": json.dumps(
            config,
            ensure_ascii=False,
            sort_keys=True,
        ),
        **best_per_horizon,
    }

    del (
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        scaler,
    )

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def append_result_atomic(
    path,
    row,
):
    row_df = pd.DataFrame(
        [
            row
        ]
    )

    if path.exists():
        existing = pd.read_csv(
            path
        )

        combined = pd.concat(
            [
                existing,
                row_df,
            ],
            ignore_index=True,
        )

        combined = combined.drop_duplicates(
            subset=[
                "model",
                "config_id",
                "fold_id",
            ],
            keep="last",
        )
    else:
        combined = row_df

    temp = path.with_suffix(
        ".tmp.csv"
    )

    combined.to_csv(
        temp,
        index=False,
        encoding="utf-8-sig",
    )

    os.replace(
        temp,
        path,
    )


def summarize_candidates(
    fold_df,
):
    rows = []

    for (
        model,
        config_id,
    ), grp in fold_df.groupby(
        [
            "model",
            "config_id",
        ],
        sort=False,
    ):
        if grp[
            "fold_id"
        ].nunique() < 3:
            continue

        grp = grp.sort_values(
            "fold_id"
        )

        row = {
            "model": model,
            "config_id": config_id,
            "parameters": int(
                grp[
                    "parameters"
                ].iloc[
                    0
                ]
            ),
            "fold_count": int(
                len(
                    grp
                )
            ),
            "mean_CV_AW_RMSE_mps": float(
                grp[
                    "best_validation_AW_RMSE_mps"
                ].mean()
            ),
            "std_CV_AW_RMSE_mps": float(
                grp[
                    "best_validation_AW_RMSE_mps"
                ].std(
                    ddof=1
                )
            ),
            "mean_best_epoch": float(
                grp[
                    "best_epoch"
                ].mean()
            ),
            "mean_training_seconds": float(
                grp[
                    "training_seconds"
                ].mean()
            ),
            "total_training_seconds": float(
                grp[
                    "training_seconds"
                ].sum()
            ),
            "config_json": str(
                grp[
                    "config_json"
                ].iloc[
                    0
                ]
            ),
        }

        for horizon in HORIZONS:
            col = (
                f"validation_AW_{horizon}m_mps"
            )

            if col in grp.columns:
                row[
                    f"mean_CV_AW_{horizon}m_mps"
                ] = float(
                    grp[
                        col
                    ].mean()
                )

                row[
                    f"std_CV_AW_{horizon}m_mps"
                ] = float(
                    grp[
                        col
                    ].std(
                        ddof=1
                    )
                )

        rows.append(
            row
        )

    out = pd.DataFrame(
        rows
    )

    if out.empty:
        return out

    out[
        "rank_within_model"
    ] = (
        out.groupby(
            "model"
        )[
            "mean_CV_AW_RMSE_mps"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(
            int
        )
    )

    return out.sort_values(
        [
            "model",
            "rank_within_model",
            "mean_CV_AW_RMSE_mps",
        ]
    ).reset_index(
        drop=True
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        default=(
            r"D:\project\WindPredict_SaildroneData\data\forecasting"
            r"\SD1090_TPOS2024_JointForecasting_v0_1"
        ),
    )

    parser.add_argument(
        "--framework-dir",
        default=None,
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
        "--candidate-limit",
        type=int,
        default=6,
        help=(
            "Use first N predeclared configurations per model. "
            "Default 6 = full 09B screening."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260901,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
    )

    args = parser.parse_args()

    if args.candidate_limit < 1:
        raise ValueError(
            "--candidate-limit must be >=1."
        )

    shared09a, shared09a_path = (
        load_09a()
    )

    utility, utility_path = (
        shared09a.load_utility()
    )

    (
        torch,
        nn,
        F,
        DataLoader,
        _,
    ) = shared09a.import_torch()

    DatasetClass = build_dataset_class(
        torch
    )

    dataset_dir = Path(
        args.dataset_dir
    )

    framework_dir = (
        Path(
            args.framework_dir
        )
        if args.framework_dir
        else (
            dataset_dir
            / "09A_modern_baseline_framework_v0_1"
        )
    )

    frozen_search_path = (
        framework_dir
        / "frozen_search_space.json"
    )

    if not frozen_search_path.exists():
        raise FileNotFoundError(
            "09A frozen_search_space.json not found. "
            "Run 09A successfully before 09B."
        )

    with frozen_search_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        frozen_search = json.load(
            f
        )

    validate_candidates_against_frozen_search(
        SCREENING_CONFIGS,
        frozen_search,
    )

    frozen = (
        shared09a.validate_frozen_dataset(
            utility,
            dataset_dir,
        )
    )

    models = parse_models(
        args.models
    )

    output_dir = (
        Path(
            args.output_dir
        )
        if args.output_dir
        else (
            dataset_dir
            / "09B_modern_baseline_rolling_cv_v0_1"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_result_path = (
        output_dir
        / "screening_fold_results.csv"
    )

    folds = build_rolling_folds(
        frozen[
            "train"
        ],
        purge_samples=PURGE_SAMPLES,
    )

    fold_protocol_df = (
        timestamp_audit(
            frozen[
                "train"
            ],
            folds,
        )
    )

    fold_protocol_df.to_csv(
        output_dir
        / "rolling_fold_protocol.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Exact frozen fold sizes are expected from the current dataset.
    expected_counts = [
        (
            38957,
            12989,
        ),
        (
            51946,
            12989,
        ),
        (
            64935,
            12989,
        ),
    ]

    actual_counts = [
        (
            fold[
                "train_count"
            ],
            fold[
                "validation_count"
            ],
        )
        for fold in folds
    ]

    if actual_counts != expected_counts:
        raise RuntimeError(
            "Rolling fold sizes do not match the previously frozen "
            f"05C protocol. Expected {expected_counts}, got {actual_counts}."
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

    use_amp = bool(
        device.type == "cuda"
        and not args.no_amp
    )

    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    model_classes = (
        shared09a.build_model_classes(
            torch,
            nn,
            F,
        )
    )

    completed = set()

    if (
        args.resume
        and fold_result_path.exists()
    ):
        prior = pd.read_csv(
            fold_result_path
        )

        for row in prior.itertuples():
            completed.add(
                (
                    str(
                        row.model
                    ),
                    str(
                        row.config_id
                    ),
                    int(
                        row.fold_id
                    ),
                )
            )

        log(
            f"[RESUME] Found {len(completed)} completed fold runs."
        )

    candidates = {
        model: SCREENING_CONFIGS[
            model
        ][
            : min(
                args.candidate_limit,
                len(
                    SCREENING_CONFIGS[
                        model
                    ]
                ),
            )
        ]
        for model in models
    }

    total_runs = sum(
        len(
            candidates[
                model
            ]
        )
        * len(
            folds
        )
        for model in models
    )

    log(
        "=" * 110
    )

    log(
        "09B - MODERN BASELINE ROLLING-ORIGIN SCREENING"
    )

    log(
        "=" * 110
    )

    log(
        f"script_version         : {SCRIPT_VERSION}"
    )

    log(
        f"dataset                : {dataset_dir}"
    )

    log(
        f"09A framework          : {framework_dir}"
    )

    log(
        f"models                 : {models}"
    )

    log(
        f"configs/model          : up to {args.candidate_limit}"
    )

    log(
        f"folds                   : 3"
    )

    log(
        f"planned fold runs       : {total_runs}"
    )

    log(
        f"max epochs / patience   : "
        f"{args.max_epochs} / {args.patience}"
    )

    log(
        f"batch size              : {args.batch_size}"
    )

    log(
        f"device                  : {device}"
    )

    log(
        f"AMP                     : {use_amp}"
    )

    log(
        "outer validation        : NOT USED FOR HPO"
    )

    log(
        "SD1090 test             : NOT USED"
    )

    log(
        "SD1033 external         : NOT ACCESSED"
    )

    log(
        "proposed model          : FROZEN"
    )

    log("")

    run_counter = 0

    for model_name in models:
        log(
            "#" * 110
        )

        log(
            f"MODEL: {model_name}"
        )

        log(
            "#" * 110
        )

        for config in candidates[
            model_name
        ]:
            log(
                f"[CONFIG {config['config_id']}] "
                + json.dumps(
                    config,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

            for fold in folds:
                run_counter += 1

                key = (
                    model_name,
                    config[
                        "config_id"
                    ],
                    fold[
                        "fold_id"
                    ],
                )

                if key in completed:
                    log(
                        f"  [SKIP completed] "
                        f"{model_name}/{config['config_id']}/"
                        f"fold{fold['fold_id']}"
                    )
                    continue

                fold_seed = (
                    int(
                        args.seed
                    )
                    + 10000
                    * MODEL_ORDER.index(
                        model_name
                    )
                    + 100
                    * int(
                        config[
                            "config_id"
                        ][
                            1:
                        ]
                    )
                    + int(
                        fold[
                            "fold_id"
                        ]
                    )
                )

                log(
                    f"  [RUN {run_counter}/{total_runs}] "
                    f"fold={fold['fold_id']} | "
                    f"train={fold['train_count']:,} | "
                    f"val={fold['validation_count']:,} | "
                    f"seed={fold_seed}"
                )

                row = train_one_fold(
                    shared09a,
                    utility,
                    torch,
                    nn,
                    DataLoader,
                    DatasetClass,
                    model_classes,
                    model_name=model_name,
                    config=config,
                    fold=fold,
                    frozen=frozen,
                    device=device,
                    batch_size=args.batch_size,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    gradient_clip_norm=args.gradient_clip_norm,
                    use_amp=use_amp,
                    seed=fold_seed,
                )

                append_result_atomic(
                    fold_result_path,
                    row,
                )

                completed.add(
                    key
                )

                log(
                    f"    [DONE] bestAW="
                    f"{row['best_validation_AW_RMSE_mps']:.4f} "
                    f"at epoch {row['best_epoch']} | "
                    f"{row['training_seconds']:.1f}s"
                )

    if not fold_result_path.exists():
        raise RuntimeError(
            "No screening fold results produced."
        )

    fold_df = pd.read_csv(
        fold_result_path
    )

    summary_df = summarize_candidates(
        fold_df
    )

    if summary_df.empty:
        raise RuntimeError(
            "No candidate completed all three folds."
        )

    summary_path = (
        output_dir
        / "screening_candidate_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    selected = {}

    for model_name in models:
        model_summary = (
            summary_df.loc[
                summary_df[
                    "model"
                ] == model_name
            ]
            .sort_values(
                [
                    "mean_CV_AW_RMSE_mps",
                    "std_CV_AW_RMSE_mps",
                    "parameters",
                ]
            )
        )

        if model_summary.empty:
            continue

        winner = model_summary.iloc[
            0
        ]

        selected[
            model_name
        ] = {
            "config_id": str(
                winner[
                    "config_id"
                ]
            ),
            "config": json.loads(
                winner[
                    "config_json"
                ]
            ),
            "mean_CV_AW_RMSE_mps": float(
                winner[
                    "mean_CV_AW_RMSE_mps"
                ]
            ),
            "std_CV_AW_RMSE_mps": float(
                winner[
                    "std_CV_AW_RMSE_mps"
                ]
            ),
            "parameters": int(
                winner[
                    "parameters"
                ]
            ),
            "selection_basis": (
                "lowest mean 3-fold rolling-origin AW vector RMSE "
                "within frozen SD1090 outer TRAIN only"
            ),
        }

    save_json(
        output_dir
        / "selected_configs.json",
        {
            "script_version": SCRIPT_VERSION,
            "selected": selected,
            "external_data_used": False,
            "outer_validation_used_for_hpo": False,
            "test_used_for_hpo": False,
            "next_stage": (
                "09C refit selected configs on full SD1090 outer TRAIN "
                "and evaluate/checkpoint on frozen OUTER VALIDATION."
            ),
        },
    )

    manifest = {
        "script_version": SCRIPT_VERSION,
        "stage": "09B",
        "09A_script": str(
            shared09a_path
        ),
        "05C_utility": str(
            utility_path
        ),
        "dataset_dir": str(
            dataset_dir
        ),
        "framework_dir": str(
            framework_dir
        ),
        "models": models,
        "candidate_limit": int(
            args.candidate_limit
        ),
        "screening_configs": {
            model: candidates[
                model
            ]
            for model in models
        },
        "rolling_protocol": (
            fold_protocol_df.to_dict(
                orient="records"
            )
        ),
        "training": {
            "loss": (
                "standardized six-target joint MSE"
            ),
            "selection_metric": (
                "validation mean apparent-wind vector RMSE"
            ),
            "optimizer": "Adam",
            "max_epochs": int(
                args.max_epochs
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
            "amp": bool(
                use_amp
            ),
        },
        "data_access_policy": {
            "SD1090_outer_train": True,
            "SD1090_outer_validation_for_HPO": False,
            "SD1090_test": False,
            "SD1033_external": False,
            "proposed_model_modified": False,
        },
        "selected": selected,
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(
                device
            ),
        },
    }

    save_json(
        output_dir
        / "selection_manifest.json",
        manifest,
    )

    with (
        output_dir
        / "selection_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "09B Modern-Baseline Rolling-Origin Screening Report\n"
        )
        f.write(
            "=" * 110
            + "\n\n"
        )
        f.write(
            "Protocol\n"
        )
        f.write(
            "-" * 110
            + "\n"
        )
        f.write(
            "HPO data: frozen SD1090 OUTER TRAIN only.\n"
        )
        f.write(
            "Outer validation: not used for HPO.\n"
        )
        f.write(
            "SD1090 test: not used.\n"
        )
        f.write(
            "SD1033: not accessed.\n"
        )
        f.write(
            "Proposed Physics-Compact model: frozen.\n"
        )
        f.write(
            "Selection metric: mean 3-fold rolling-origin AW vector RMSE.\n\n"
        )
        f.write(
            "Rolling folds\n"
        )
        f.write(
            "-" * 110
            + "\n"
        )
        f.write(
            fold_protocol_df.to_string(
                index=False
            )
        )
        f.write(
            "\n\nCandidate ranking\n"
        )
        f.write(
            "-" * 110
            + "\n"
        )
        f.write(
            summary_df.to_string(
                index=False
            )
        )
        f.write(
            "\n\nSelected configurations\n"
        )
        f.write(
            "-" * 110
            + "\n"
        )
        for model_name, item in (
            selected.items()
        ):
            f.write(
                f"{model_name}: "
                f"{item['config_id']} | "
                f"CV AW={item['mean_CV_AW_RMSE_mps']:.6f} "
                f"+/- {item['std_CV_AW_RMSE_mps']:.6f} m/s | "
                f"params={item['parameters']:,}\n"
            )
            f.write(
                "  "
                + json.dumps(
                    item[
                        "config"
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    log("")
    log(
        "=" * 110
    )

    log(
        "09B MODERN BASELINE ROLLING-CV RESULTS"
    )

    log(
        "=" * 110
    )

    for model_name in models:
        if model_name not in selected:
            continue

        item = selected[
            model_name
        ]

        log(
            f"{model_name}:"
        )

        log(
            f"  selected config = "
            f"{item['config_id']}"
        )

        log(
            f"  CV AW RMSE      = "
            f"{item['mean_CV_AW_RMSE_mps']:.4f} "
            f"+/- {item['std_CV_AW_RMSE_mps']:.4f} m/s"
        )

        log(
            f"  parameters      = "
            f"{item['parameters']:,}"
        )

        log(
            f"  config          = "
            + json.dumps(
                item[
                    "config"
                ],
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    log("")
    log(
        "[POLICY] HPO used SD1090 OUTER TRAIN only."
    )

    log(
        "[POLICY] SD1090 outer validation/test and SD1033 were not used."
    )

    log(
        "[POLICY] Physics-Compact proposed model remains frozen."
    )

    log(
        "[NEXT] 09C will refit/confirm the selected modern baselines "
        "on full outer TRAIN and evaluate on frozen outer VALIDATION."
    )

    log(
        f"[DONE] 09B outputs: {output_dir}"
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
