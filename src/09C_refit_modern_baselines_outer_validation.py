# -*- coding: utf-8 -*-
"""
09C_refit_modern_baselines_outer_validation.py

Final SD1090 development refit / validation confirmation for the four modern
comparison baselines selected in Stage 09B:

    DLinear
    iTransformer
    TimeMixer
    PatchTST

SCIENTIFIC ROLE
---------------
The proposed Physics-Compact-Vessel-Residual model remains FROZEN.

Stage 09C does NOT modify any modern-baseline hyperparameter selected by 09B.
It only:

    1. loads the already selected 09B configuration for each model;
    2. trains that configuration on the COMPLETE frozen SD1090 OUTER TRAIN;
    3. uses the frozen SD1090 OUTER VALIDATION only for early stopping /
       checkpointing;
    4. repeats the final refit for five predeclared seeds;
    5. saves all five frozen checkpoints per model;
    6. evaluates all five checkpoints on OUTER VALIDATION;
    7. DOES NOT access SD1090 test;
    8. DOES NOT access SD1033.

All four modern baselines will proceed to Stage 09D external inference. No seed
or model is selected using SD1033.

FAIR-COMPARISON CONTRACT
------------------------
Input:
    X: [B, 60, 11]

Target:
    y: [B, 5, 6]

Target channels:
    U, V,
    VESSEL_EAST_MPS, VESSEL_NORTH_MPS,
    HDG_sin, HDG_cos

Forecast horizons:
    [1, 2, 3, 5, 10] min

Training loss:
    standardized six-target joint MSE

Checkpoint criterion:
    mean apparent-wind vector RMSE on frozen SD1090 OUTER VALIDATION

Apparent-wind reconstruction:
    A_E = U - V_ship,E
    A_N = V - V_ship,N

The metric implementation is imported from the same frozen 05C utility used by
the previous baselines.

MULTI-SEED POLICY
-----------------
The default five seeds are the same schedule previously used for compact-model
confirmation:

    500043
    501052
    502061
    503070
    504079

Every seed uses the SAME 09B-selected hyperparameters. All five checkpoints are
retained. Stage 09D must evaluate all five checkpoints on SD1033 and report
mean +/- std; external data must not be used to choose a seed.

NUMERICAL POLICY
----------------
The architecture-agnostic Stage-09B-v2 numerical policy is reused unchanged:

    - gradient clipping = 1.0
    - AMP on CUDA by default
    - current torch.amp.GradScaler API
    - if an AMP batch produces a non-finite loss/gradient, retry the SAME batch
      once in full FP32 without changing optimizer/loss/learning rate
    - if FP32 recovery also fails after a valid validation checkpoint exists,
      terminate the run and restore the best finite checkpoint
    - if no finite checkpoint exists, the seed run is invalid

No numerical recovery step changes the scientific hyperparameters.

DEFAULT PATHS
-------------
Dataset:
D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\
SD1090_TPOS2024_JointForecasting_v0_1

09B selected configs:
...\\09B_modern_baseline_rolling_cv_v0_2\\selected_configs.json

Output:
...\\09C_modern_baseline_outer_validation_v0_1

DEPENDENCIES
------------
Place in the same src directory as:
    09A_modern_baseline_framework.py
    09B_tune_modern_baselines_rolling_cv_v2.py
    05C_tune_joint_deep_baselines_pytorch_v2.py

Recommended environment:
    conda activate WindPredict

OUTPUTS
-------
09C_modern_baseline_outer_validation_v0_1/
    models/
        DLinear_seed500043.pt
        ...
        PatchTST_seed504079.pt

    histories/
        DLinear_seed500043.csv
        ...

    predictions/                         [optional, enabled by default]
        validation_DLinear_seed500043.npz
        ...

    final_seed_metrics.csv
    final_seed_summary.csv
    final_per_horizon_metrics.csv
    final_per_horizon_summary.csv
    persistence_validation_metrics.csv
    selected_config_audit.csv
    refit_manifest.json
    refit_report.txt

No test or external metric is produced.
"""

from __future__ import annotations

import argparse
import copy
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


SCRIPT_VERSION = "0.1.0-modern-baseline-final-refit"
SHARED_09A = "09A_modern_baseline_framework.py"
SHARED_09B = "09B_tune_modern_baselines_rolling_cv_v2.py"

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

DEFAULT_SEEDS = [
    500043,
    501052,
    502061,
    503070,
    504079,
]

DIRECTION_MIN_SPEED = 0.5
COG_MIN_SPEED = 0.2


def log(message=""):
    print(message, flush=True)


def load_module_from_src(
    filename,
    module_name,
):
    path = (
        Path(__file__).resolve().parent
        / filename
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Required script not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module, path


def save_json(
    path,
    obj,
):
    def convert(v):
        if isinstance(v, Path):
            return str(v)

        if isinstance(v, np.ndarray):
            return [
                convert(x)
                for x in v.tolist()
            ]

        if isinstance(v, np.integer):
            return int(v)

        if isinstance(v, np.floating):
            return (
                None
                if np.isnan(v)
                else float(v)
            )

        if isinstance(v, np.bool_):
            return bool(v)

        if isinstance(v, dict):
            return {
                str(k): convert(x)
                for k, x in v.items()
            }

        if isinstance(v, (list, tuple)):
            return [
                convert(x)
                for x in v
            ]

        return v

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            convert(obj),
            f,
            ensure_ascii=False,
            indent=2,
        )


def parse_models(
    text,
):
    lookup = {
        m.lower(): m
        for m in MODEL_ORDER
    }

    out = []

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

        model = lookup[
            key
        ]

        if model not in out:
            out.append(
                model
            )

    if not out:
        raise ValueError(
            "No models requested."
        )

    return out


def parse_seeds(
    text,
):
    seeds = []

    for token in text.split(","):
        token = token.strip()

        if not token:
            continue

        seed = int(
            token
        )

        if seed not in seeds:
            seeds.append(
                seed
            )

    if not seeds:
        raise ValueError(
            "No seeds supplied."
        )

    return seeds


def set_seed(
    torch,
    seed,
    deterministic,
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

    if deterministic:
        try:
            torch.use_deterministic_algorithms(
                True,
                warn_only=True,
            )
        except Exception:
            pass

        if hasattr(
            torch.backends,
            "cudnn",
        ):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    else:
        if hasattr(
            torch.backends,
            "cudnn",
        ):
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


def load_selected_configs(
    path,
    requested_models,
):
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        obj = json.load(
            f
        )

    selected = obj.get(
        "selected",
        {}
    )

    missing = [
        model
        for model in requested_models
        if model not in selected
    ]

    if missing:
        raise RuntimeError(
            f"09B selected_configs missing models: {missing}"
        )

    # Hard audit: Stage 09B must have selected from outer TRAIN only.
    if bool(
        obj.get(
            "external_data_used",
            False,
        )
    ):
        raise RuntimeError(
            "09B metadata says external data were used. "
            "Refusing 09C."
        )

    if bool(
        obj.get(
            "outer_validation_used_for_hpo",
            False,
        )
    ):
        raise RuntimeError(
            "09B metadata says outer validation was used for HPO. "
            "Refusing 09C."
        )

    if bool(
        obj.get(
            "test_used_for_hpo",
            False,
        )
    ):
        raise RuntimeError(
            "09B metadata says test was used for HPO. "
            "Refusing 09C."
        )

    return obj


def make_loader(
    torch,
    DataLoader,
    TensorDataset,
    X,
    y,
    *,
    batch_size,
    shuffle,
    pin_memory,
    seed,
):
    dataset = TensorDataset(
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

    generator = torch.Generator()

    generator.manual_seed(
        int(
            seed
        )
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


def evaluate_loader(
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

    total_loss = 0.0
    total_n = 0

    pred_chunks = []
    truth_chunks = []

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
                    "Non-finite validation loss."
                )

            n = int(
                Xb.shape[
                    0
                ]
            )

            total_loss += (
                float(
                    loss.item()
                )
                * n
            )

            total_n += n

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

    mean_aw, per_horizon_aw = (
        utility.mean_apparent_vector_rmse(
            truth_raw,
            pred_raw,
        )
    )

    return {
        "joint_MSE": float(
            total_loss
            / max(
                total_n,
                1,
            )
        ),
        "mean_AW_RMSE_mps": float(
            mean_aw
        ),
        "per_horizon_AW_RMSE_mps": np.asarray(
            per_horizon_aw,
            dtype=np.float64,
        ),
        "pred_z": pred_z,
        "truth_z": truth_z,
        "pred_raw": pred_raw,
        "truth_raw": truth_raw,
    }


def copy_state_to_cpu(
    model,
):
    return {
        key: value.detach()
        .cpu()
        .clone()
        for key, value
        in model.state_dict().items()
    }


def train_seed(
    shared09a,
    shared09b,
    utility,
    torch,
    nn,
    DataLoader,
    TensorDataset,
    model_classes,
    *,
    model_name,
    config,
    seed,
    frozen,
    device,
    batch_size,
    max_epochs,
    patience,
    gradient_clip_norm,
    use_amp,
    deterministic,
    checkpoint_path,
    history_path,
):
    set_seed(
        torch,
        seed,
        deterministic,
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

    parameters = (
        shared09a.count_parameters(
            model
        )
    )

    train_loader = make_loader(
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
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(
            device.type
            == "cuda"
        ),
        seed=seed,
    )

    validation_loader = make_loader(
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
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(
            device.type
            == "cuda"
        ),
        seed=seed,
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

    scaler = shared09b.create_grad_scaler(
        torch,
        device=device,
        enabled=use_amp,
    )

    best_aw = np.inf
    best_val_mse = np.inf
    best_epoch = -1
    best_state = None
    best_per_horizon = None

    epochs_without_improvement = 0

    history_rows = []

    nonfinite_event_count = 0
    fp32_retry_count = 0
    fp32_recovery_success_count = 0
    stopped_after_nonfinite = False
    termination_reason = ""
    terminate_run = False

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
        train_n = 0

        epoch_nonfinite_events = 0
        epoch_fp32_recoveries = 0

        for Xb, yb in train_loader:
            Xb = Xb.to(
                device,
                non_blocking=True,
            )

            yb = yb.to(
                device,
                non_blocking=True,
            )

            batch_result = (
                shared09b.run_training_batch_with_recovery(
                    torch,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    criterion=criterion,
                    Xb=Xb,
                    yb=yb,
                    device=device,
                    use_amp=use_amp,
                    gradient_clip_norm=gradient_clip_norm,
                )
            )

            if batch_result[
                "used_fp32_retry"
            ]:
                nonfinite_event_count += 1
                fp32_retry_count += 1
                epoch_nonfinite_events += 1

                log(
                    f"      [RECOVERY] "
                    f"{model_name}/seed{seed}/epoch{epoch} | "
                    f"AMP={batch_result['amp_failure_reason']} | "
                    f"FP32="
                    f"{'PASS' if batch_result['success'] else batch_result['fp32_failure_reason']}"
                )

            if not batch_result[
                "success"
            ]:
                stopped_after_nonfinite = True

                termination_reason = (
                    "FP32 recovery failed: AMP="
                    + str(
                        batch_result[
                            "amp_failure_reason"
                        ]
                    )
                    + "; FP32="
                    + str(
                        batch_result[
                            "fp32_failure_reason"
                        ]
                    )
                )

                terminate_run = True
                break

            if batch_result[
                "used_fp32_retry"
            ]:
                fp32_recovery_success_count += 1
                epoch_fp32_recoveries += 1

            batch_n = int(
                Xb.shape[
                    0
                ]
            )

            train_loss_sum += (
                float(
                    batch_result[
                        "loss_value"
                    ]
                )
                * batch_n
            )

            train_n += batch_n

        if terminate_run:
            break

        train_mse = float(
            train_loss_sum
            / max(
                train_n,
                1,
            )
        )

        val = evaluate_loader(
            utility,
            torch,
            model,
            validation_loader,
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
            val[
                "joint_MSE"
            ]
        )

        current_aw = float(
            val[
                "mean_AW_RMSE_mps"
            ]
        )

        improved = bool(
            current_aw
            < best_aw
            - 1e-7
        )

        if improved:
            best_aw = current_aw
            best_val_mse = float(
                val[
                    "joint_MSE"
                ]
            )

            best_epoch = int(
                epoch
            )

            best_state = copy_state_to_cpu(
                model
            )

            best_per_horizon = np.asarray(
                val[
                    "per_horizon_AW_RMSE_mps"
                ],
                dtype=np.float64,
            ).copy()

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        current_lr = float(
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        history_rows.append({
            "epoch": int(
                epoch
            ),
            "train_joint_MSE": float(
                train_mse
            ),
            "validation_joint_MSE": float(
                val[
                    "joint_MSE"
                ]
            ),
            "validation_AW_RMSE_mps": float(
                current_aw
            ),
            "best_validation_AW_RMSE_mps": float(
                best_aw
            ),
            "learning_rate": current_lr,
            "improved_checkpoint": bool(
                improved
            ),
            "epoch_nonfinite_events": int(
                epoch_nonfinite_events
            ),
            "epoch_fp32_recovery_successes": int(
                epoch_fp32_recoveries
            ),
        })

        if (
            epoch == 1
            or improved
            or epoch % 10 == 0
        ):
            log(
                f"      epoch={epoch:02d} "
                f"trainMSE={train_mse:.5f} "
                f"valMSE={val['joint_MSE']:.5f} "
                f"valAW={current_aw:.4f} "
                f"best={best_aw:.4f} "
                f"lr={current_lr:.2e}"
            )

        if (
            epochs_without_improvement
            >= int(
                patience
            )
        ):
            termination_reason = (
                "early_stopping_patience"
            )
            break

    elapsed = float(
        time.perf_counter()
        - t0
    )

    history_df = pd.DataFrame(
        history_rows
    )

    history_df.to_csv(
        history_path,
        index=False,
        encoding="utf-8-sig",
    )

    if best_state is None:
        raise RuntimeError(
            f"{model_name}/seed{seed}: "
            "no finite outer-validation checkpoint was obtained."
        )

    # Restore the frozen best validation checkpoint.
    model.load_state_dict(
        best_state,
        strict=True,
    )

    final_val = evaluate_loader(
        utility,
        torch,
        model,
        validation_loader,
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

    # Exact reproducibility audit between stored best metric and restored model.
    restore_delta = abs(
        float(
            final_val[
                "mean_AW_RMSE_mps"
            ]
        )
        - float(
            best_aw
        )
    )

    if restore_delta > 1e-5:
        raise RuntimeError(
            f"{model_name}/seed{seed}: checkpoint restore audit failed; "
            f"AW difference={restore_delta:.3e}."
        )

    checkpoint = {
        "script_version": SCRIPT_VERSION,
        "stage": "09C",
        "model_name": model_name,
        "seed": int(
            seed
        ),
        "config": config,
        "parameters": int(
            parameters
        ),
        "state_dict": best_state,
        "best_epoch": int(
            best_epoch
        ),
        "best_validation_AW_RMSE_mps": float(
            best_aw
        ),
        "best_validation_joint_MSE": float(
            best_val_mse
        ),
        "lookback": int(
            frozen[
                "lookback"
            ]
        ),
        "feature_names": list(
            frozen[
                "feature_names"
            ]
        ),
        "target_names": list(
            frozen[
                "target_names"
            ]
        ),
        "horizons_min": list(
            frozen[
                "horizons"
            ]
        ),
        "target_mean": np.asarray(
            frozen[
                "target_mean"
            ],
            dtype=np.float64,
        ),
        "target_std": np.asarray(
            frozen[
                "target_std"
            ],
            dtype=np.float64,
        ),
        "scientific_policy": {
            "09B_selected_config_frozen": True,
            "full_outer_train_used": True,
            "outer_validation_used_for_checkpointing_only": True,
            "SD1090_test_used": False,
            "SD1033_used": False,
            "proposed_model_modified": False,
        },
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    if (
        stopped_after_nonfinite
        and not termination_reason
    ):
        termination_reason = (
            "stopped_after_nonfinite"
        )

    if not termination_reason:
        termination_reason = (
            "max_epochs"
            if (
                history_df.empty
                or int(
                    history_df[
                        "epoch"
                    ].max()
                )
                >= int(
                    max_epochs
                )
            )
            else "completed"
        )

    result = {
        "model": model_name,
        "seed": int(
            seed
        ),
        "parameters": int(
            parameters
        ),
        "config_id": str(
            config[
                "config_id"
            ]
        ),
        "best_epoch": int(
            best_epoch
        ),
        "epochs_with_validation": int(
            len(
                history_df
            )
        ),
        "validation_joint_MSE": float(
            final_val[
                "joint_MSE"
            ]
        ),
        "validation_AW_RMSE_mps": float(
            final_val[
                "mean_AW_RMSE_mps"
            ]
        ),
        "validation_AW_1m_mps": float(
            final_val[
                "per_horizon_AW_RMSE_mps"
            ][
                0
            ]
        ),
        "validation_AW_2m_mps": float(
            final_val[
                "per_horizon_AW_RMSE_mps"
            ][
                1
            ]
        ),
        "validation_AW_3m_mps": float(
            final_val[
                "per_horizon_AW_RMSE_mps"
            ][
                2
            ]
        ),
        "validation_AW_5m_mps": float(
            final_val[
                "per_horizon_AW_RMSE_mps"
            ][
                3
            ]
        ),
        "validation_AW_10m_mps": float(
            final_val[
                "per_horizon_AW_RMSE_mps"
            ][
                4
            ]
        ),
        "training_seconds": elapsed,
        "nonfinite_event_count": int(
            nonfinite_event_count
        ),
        "fp32_retry_count": int(
            fp32_retry_count
        ),
        "fp32_recovery_success_count": int(
            fp32_recovery_success_count
        ),
        "stopped_after_nonfinite": bool(
            stopped_after_nonfinite
        ),
        "termination_reason": str(
            termination_reason
        ),
        "checkpoint_restore_AW_abs_delta": float(
            restore_delta
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
        "history_path": str(
            history_path
        ),
        "config_json": json.dumps(
            config,
            ensure_ascii=False,
            sort_keys=True,
        ),
    }

    predictions = {
        "pred_z": final_val[
            "pred_z"
        ],
        "pred_raw": final_val[
            "pred_raw"
        ],
        "truth_z": final_val[
            "truth_z"
        ],
        "truth_raw": final_val[
            "truth_raw"
        ],
    }

    del (
        model,
        train_loader,
        validation_loader,
        optimizer,
        scheduler,
        scaler,
    )

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (
        result,
        predictions,
    )


def aggregate_seed_summary(
    seed_metric_df,
):
    rows = []

    numeric_metrics = [
        "validation_joint_MSE",
        "validation_AW_RMSE_mps",
        "mean_wind_vector_RMSE_mps",
        "mean_wind_speed_RMSE_mps",
        "mean_wind_direction_MAE_deg",
        "mean_vessel_vector_RMSE_mps",
        "mean_SOG_RMSE_mps",
        "mean_COG_MAE_deg",
        "mean_HDG_MAE_deg",
        "mean_AWS_RMSE_mps",
        "mean_AWA_MAE_deg",
        "mean_apparent_skill_vs_persistence",
    ]

    for model, grp in seed_metric_df.groupby(
        "model",
        sort=False,
    ):
        row = {
            "model": model,
            "seed_count": int(
                len(
                    grp
                )
            ),
            "parameters": int(
                grp[
                    "parameters"
                ].iloc[
                    0
                ]
            ),
            "config_id": str(
                grp[
                    "config_id"
                ].iloc[
                    0
                ]
            ),
            "mean_best_epoch": float(
                grp[
                    "best_epoch"
                ].mean()
            ),
            "std_best_epoch": float(
                grp[
                    "best_epoch"
                ].std(
                    ddof=1
                )
            )
            if len(
                grp
            ) > 1
            else 0.0,
            "mean_training_seconds": float(
                grp[
                    "training_seconds"
                ].mean()
            ),
            "total_nonfinite_events": int(
                grp[
                    "nonfinite_event_count"
                ].sum()
            ),
            "total_fp32_recoveries": int(
                grp[
                    "fp32_recovery_success_count"
                ].sum()
            ),
        }

        for metric in numeric_metrics:
            if metric not in grp.columns:
                continue

            values = grp[
                metric
            ].astype(
                float
            )

            row[
                f"mean_{metric}"
            ] = float(
                values.mean()
            )

            row[
                f"std_{metric}"
            ] = (
                float(
                    values.std(
                        ddof=1
                    )
                )
                if len(
                    values
                ) > 1
                else 0.0
            )

        rows.append(
            row
        )

    out = pd.DataFrame(
        rows
    )

    if not out.empty:
        out[
            "validation_AW_rank"
        ] = (
            out[
                "mean_validation_AW_RMSE_mps"
            ]
            .rank(
                method="min",
                ascending=True,
            )
            .astype(
                int
            )
        )

        out = out.sort_values(
            [
                "validation_AW_rank",
                "mean_validation_AW_RMSE_mps",
                "parameters",
            ]
        ).reset_index(
            drop=True
        )

    return out


def aggregate_per_horizon(
    metrics_df,
):
    metric_cols = [
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
        "apparent_skill_vs_persistence",
    ]

    rows = []

    for (
        model,
        horizon,
    ), grp in metrics_df.groupby(
        [
            "model",
            "horizon_min",
        ],
        sort=False,
    ):
        row = {
            "model": model,
            "horizon_min": int(
                horizon
            ),
            "seed_count": int(
                grp[
                    "seed"
                ].nunique()
            ),
        }

        for col in metric_cols:
            values = grp[
                col
            ].astype(
                float
            )

            row[
                f"mean_{col}"
            ] = float(
                values.mean()
            )

            row[
                f"std_{col}"
            ] = (
                float(
                    values.std(
                        ddof=1
                    )
                )
                if len(
                    values
                ) > 1
                else 0.0
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "model",
            "horizon_min",
        ]
    ).reset_index(
        drop=True
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
    )

    parser.add_argument(
        "--selected-configs",
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
        "--seeds",
        default=",".join(
            str(v)
            for v in DEFAULT_SEEDS
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
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
        "--deterministic",
        action="store_true",
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
    )

    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
    )

    args = parser.parse_args()

    dataset_dir = Path(
        args.dataset_dir
    )

    selected_path = (
        Path(
            args.selected_configs
        )
        if args.selected_configs
        else (
            dataset_dir
            / "09B_modern_baseline_rolling_cv_v0_2"
            / "selected_configs.json"
        )
    )

    output_dir = (
        Path(
            args.output_dir
        )
        if args.output_dir
        else (
            dataset_dir
            / "09C_modern_baseline_outer_validation_v0_1"
        )
    )

    models_dir = (
        output_dir
        / "models"
    )

    histories_dir = (
        output_dir
        / "histories"
    )

    predictions_dir = (
        output_dir
        / "predictions"
    )

    for path in [
        output_dir,
        models_dir,
        histories_dir,
        predictions_dir,
    ]:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    shared09a, path09a = (
        load_module_from_src(
            SHARED_09A,
            "modern_baseline_09a_shared_09c",
        )
    )

    shared09b, path09b = (
        load_module_from_src(
            SHARED_09B,
            "modern_baseline_09b_shared_09c",
        )
    )

    utility, utility_path = (
        shared09a.load_utility()
    )

    (
        torch,
        nn,
        F,
        DataLoader,
        TensorDataset,
    ) = shared09a.import_torch()

    requested_models = parse_models(
        args.models
    )

    seeds = parse_seeds(
        args.seeds
    )

    selected_obj = load_selected_configs(
        selected_path,
        requested_models,
    )

    selected = selected_obj[
        "selected"
    ]

    frozen = (
        shared09a.validate_frozen_dataset(
            utility,
            dataset_dir,
        )
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

    # ------------------------------------------------------------------
    # Exact outer-validation references.
    # ------------------------------------------------------------------
    validation_truth = frozen[
        "validation"
    ][
        "y_raw"
    ].astype(
        np.float32,
        copy=False,
    )

    validation_apparent = frozen[
        "validation"
    ][
        "apparent_earth_raw"
    ].astype(
        np.float32,
        copy=False,
    )

    persistence_joint = (
        utility.persistence_joint_prediction(
            X=frozen[
                "validation"
            ][
                "X"
            ],
            feature_names=frozen[
                "feature_names"
            ],
            feature_mean=frozen[
                "feature_mean"
            ],
            feature_std=frozen[
                "feature_std"
            ],
            n_horizons=len(
                frozen[
                    "horizons"
                ]
            ),
        )
    )

    persistence_metrics = (
        utility.evaluate_model(
            truth_joint=validation_truth,
            pred_joint=persistence_joint,
            truth_apparent_ref=validation_apparent,
            horizons=tuple(
                frozen[
                    "horizons"
                ]
            ),
            split_name="validation",
            model_name="Persistence",
            direction_min_speed=DIRECTION_MIN_SPEED,
            cog_min_speed=COG_MIN_SPEED,
            persistence_joint=persistence_joint,
        )
    )

    persistence_metrics.to_csv(
        output_dir
        / "persistence_validation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    persistence_summary = (
        utility.summarize_metrics(
            persistence_metrics
        )
    )

    log(
        "=" * 110
    )
    log(
        "09C - MODERN BASELINE FULL OUTER-TRAIN REFIT / OUTER-VALIDATION CONFIRMATION"
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
        f"09B selected configs   : {selected_path}"
    )
    log(
        f"models                 : {requested_models}"
    )
    log(
        f"seeds                  : {seeds}"
    )
    log(
        f"outer train samples    : "
        f"{frozen['train']['X'].shape[0]:,}"
    )
    log(
        f"outer validation       : "
        f"{frozen['validation']['X'].shape[0]:,}"
    )
    log(
        f"device                 : {device}"
    )
    log(
        f"AMP                    : {use_amp}"
    )
    log(
        "SD1090 test            : NOT ACCESSED"
    )
    log(
        "SD1033 external        : NOT ACCESSED"
    )
    log(
        "proposed model         : FROZEN"
    )
    log("")

    config_audit_rows = []

    for model_name in requested_models:
        item = selected[
            model_name
        ]

        config = dict(
            item[
                "config"
            ]
        )

        probe = (
            shared09a.instantiate_model(
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
            )
        )

        params = (
            shared09a.count_parameters(
                probe
            )
        )

        if int(
            params
        ) != int(
            item[
                "parameters"
            ]
        ):
            raise RuntimeError(
                f"{model_name}: parameter count mismatch between "
                f"09B selection ({item['parameters']}) and 09C "
                f"reconstruction ({params})."
            )

        config_audit_rows.append({
            "model": model_name,
            "config_id": item[
                "config_id"
            ],
            "09B_CV_AW_RMSE_mps": float(
                item[
                    "mean_CV_AW_RMSE_mps"
                ]
            ),
            "09B_CV_AW_std_mps": float(
                item[
                    "std_CV_AW_RMSE_mps"
                ]
            ),
            "parameters": int(
                params
            ),
            "config_json": json.dumps(
                config,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "parameter_count_audit": True,
        })

        del probe

    config_audit_df = pd.DataFrame(
        config_audit_rows
    )

    config_audit_df.to_csv(
        output_dir
        / "selected_config_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    seed_rows = []
    per_horizon_frames = []

    for model_index, model_name in enumerate(
        requested_models,
        start=1,
    ):
        item = selected[
            model_name
        ]

        config = dict(
            item[
                "config"
            ]
        )

        log(
            "#" * 110
        )
        log(
            f"[{model_index}/{len(requested_models)}] {model_name}"
        )
        log(
            f"config={item['config_id']} | "
            f"09B CV AW="
            f"{item['mean_CV_AW_RMSE_mps']:.4f}"
            f" +/- {item['std_CV_AW_RMSE_mps']:.4f} m/s | "
            f"params={item['parameters']:,}"
        )
        log(
            "#" * 110
        )

        for seed_index, seed in enumerate(
            seeds,
            start=1,
        ):
            log(
                f"  [SEED {seed_index}/{len(seeds)}] "
                f"{seed}"
            )

            safe_model = model_name.replace(
                "-",
                "_",
            )

            checkpoint_path = (
                models_dir
                / f"{safe_model}_seed{seed}.pt"
            )

            history_path = (
                histories_dir
                / f"{safe_model}_seed{seed}.csv"
            )

            result, predictions = train_seed(
                shared09a,
                shared09b,
                utility,
                torch,
                nn,
                DataLoader,
                TensorDataset,
                model_classes,
                model_name=model_name,
                config=config,
                seed=seed,
                frozen=frozen,
                device=device,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                patience=args.patience,
                gradient_clip_norm=args.gradient_clip_norm,
                use_amp=use_amp,
                deterministic=args.deterministic,
                checkpoint_path=checkpoint_path,
                history_path=history_path,
            )

            pred_raw = predictions[
                "pred_raw"
            ]

            metrics = (
                utility.evaluate_model(
                    truth_joint=validation_truth,
                    pred_joint=pred_raw,
                    truth_apparent_ref=validation_apparent,
                    horizons=tuple(
                        frozen[
                            "horizons"
                        ]
                    ),
                    split_name="validation",
                    model_name=model_name,
                    direction_min_speed=DIRECTION_MIN_SPEED,
                    cog_min_speed=COG_MIN_SPEED,
                    persistence_joint=persistence_joint,
                )
            )

            metrics[
                "seed"
            ] = int(
                seed
            )

            metrics[
                "config_id"
            ] = str(
                config[
                    "config_id"
                ]
            )

            per_horizon_frames.append(
                metrics
            )

            summary = (
                utility.summarize_metrics(
                    metrics
                ).iloc[
                    0
                ]
            )

            result.update({
                "mean_wind_vector_RMSE_mps": float(
                    summary[
                        "mean_wind_vector_RMSE_mps"
                    ]
                ),
                "mean_wind_speed_RMSE_mps": float(
                    summary[
                        "mean_wind_speed_RMSE_mps"
                    ]
                ),
                "mean_wind_direction_MAE_deg": float(
                    summary[
                        "mean_wind_direction_MAE_deg"
                    ]
                ),
                "mean_vessel_vector_RMSE_mps": float(
                    summary[
                        "mean_vessel_vector_RMSE_mps"
                    ]
                ),
                "mean_SOG_RMSE_mps": float(
                    summary[
                        "mean_SOG_RMSE_mps"
                    ]
                ),
                "mean_COG_MAE_deg": float(
                    summary[
                        "mean_COG_MAE_deg"
                    ]
                ),
                "mean_HDG_MAE_deg": float(
                    summary[
                        "mean_HDG_MAE_deg"
                    ]
                ),
                "mean_AWS_RMSE_mps": float(
                    summary[
                        "mean_AWS_RMSE_mps"
                    ]
                ),
                "mean_AWA_MAE_deg": float(
                    summary[
                        "mean_AWA_MAE_deg"
                    ]
                ),
                "mean_apparent_skill_vs_persistence": float(
                    summary[
                        "mean_apparent_skill_vs_persistence"
                    ]
                ),
            })

            seed_rows.append(
                result
            )

            if not args.no_save_predictions:
                np.savez_compressed(
                    predictions_dir
                    / (
                        f"validation_{safe_model}"
                        f"_seed{seed}.npz"
                    ),
                    pred_z=predictions[
                        "pred_z"
                    ].astype(
                        np.float32
                    ),
                    pred_raw=predictions[
                        "pred_raw"
                    ].astype(
                        np.float32
                    ),
                    truth_raw=validation_truth,
                    apparent_earth_raw=validation_apparent,
                    context_end_time_ns=frozen[
                        "validation"
                    ][
                        "context_end_time_ns"
                    ],
                    target_time_ns=frozen[
                        "validation"
                    ][
                        "target_time_ns"
                    ],
                    seed=np.asarray(
                        seed,
                        dtype=np.int64,
                    ),
                    model=np.asarray(
                        model_name
                    ),
                )

            log(
                f"    [FROZEN] epoch={result['best_epoch']} | "
                f"AW={result['validation_AW_RMSE_mps']:.4f} | "
                f"wind={result['mean_wind_vector_RMSE_mps']:.4f} | "
                f"vessel={result['mean_vessel_vector_RMSE_mps']:.4f} | "
                f"AWS={result['mean_AWS_RMSE_mps']:.4f} | "
                f"AWA={result['mean_AWA_MAE_deg']:.2f} deg"
            )

    seed_df = pd.DataFrame(
        seed_rows
    )

    per_horizon_df = pd.concat(
        per_horizon_frames,
        ignore_index=True,
    )

    seed_df.to_csv(
        output_dir
        / "final_seed_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_horizon_df.to_csv(
        output_dir
        / "final_per_horizon_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    seed_summary_df = (
        aggregate_seed_summary(
            seed_df
        )
    )

    per_horizon_summary_df = (
        aggregate_per_horizon(
            per_horizon_df
        )
    )

    seed_summary_df.to_csv(
        output_dir
        / "final_seed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_horizon_summary_df.to_csv(
        output_dir
        / "final_per_horizon_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Validation-only ranking is descriptive. It changes no 09B config and
    # all four models still proceed to external evaluation.
    # ------------------------------------------------------------------
    best_row = seed_summary_df.iloc[
        0
    ]

    manifest = {
        "script_version": SCRIPT_VERSION,
        "stage": "09C",
        "dataset_dir": str(
            dataset_dir
        ),
        "09A_script": str(
            path09a
        ),
        "09B_script": str(
            path09b
        ),
        "05C_utility": str(
            utility_path
        ),
        "09B_selected_configs": str(
            selected_path
        ),
        "models": requested_models,
        "seeds": seeds,
        "training": {
            "optimizer": "Adam",
            "loss": (
                "standardized six-target joint MSE"
            ),
            "batch_size": int(
                args.batch_size
            ),
            "max_epochs": int(
                args.max_epochs
            ),
            "patience": int(
                args.patience
            ),
            "checkpoint_metric": (
                "outer-validation mean apparent-wind vector RMSE"
            ),
            "gradient_clip_norm": float(
                args.gradient_clip_norm
            ),
            "amp": bool(
                use_amp
            ),
            "deterministic": bool(
                args.deterministic
            ),
        },
        "scientific_policy": {
            "09B_hyperparameters_changed": False,
            "outer_train_used_for_refit": True,
            "outer_validation_used_for_checkpointing_only": True,
            "outer_validation_used_to_change_hyperparameters": False,
            "SD1090_test_accessed": False,
            "SD1033_accessed": False,
            "all_four_models_proceed_to_external_evaluation": True,
            "all_five_seeds_retained": True,
            "proposed_Physics_Compact_modified": False,
        },
        "validation_descriptive_ranking": (
            seed_summary_df[
                [
                    "model",
                    "validation_AW_rank",
                    "mean_validation_AW_RMSE_mps",
                    "std_validation_AW_RMSE_mps",
                    "parameters",
                ]
            ].to_dict(
                orient="records"
            )
        ),
        "best_modern_baseline_on_outer_validation_descriptive_only": {
            "model": str(
                best_row[
                    "model"
                ]
            ),
            "mean_AW_RMSE_mps": float(
                best_row[
                    "mean_validation_AW_RMSE_mps"
                ]
            ),
            "std_AW_RMSE_mps": float(
                best_row[
                    "std_validation_AW_RMSE_mps"
                ]
            ),
        },
        "persistence_validation_summary": (
            persistence_summary.to_dict(
                orient="records"
            )
        ),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
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
        / "refit_manifest.json",
        manifest,
    )

    with (
        output_dir
        / "refit_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "09C Modern Baseline Full Outer-Train Refit / "
            "Outer-Validation Confirmation\n"
        )
        f.write(
            "=" * 110
            + "\n\n"
        )
        f.write(
            "Scientific protocol\n"
        )
        f.write(
            "-" * 110
            + "\n"
        )
        f.write(
            "09B-selected hyperparameters are frozen.\n"
        )
        f.write(
            "Training data: complete SD1090 outer TRAIN.\n"
        )
        f.write(
            "Checkpointing only: frozen SD1090 outer VALIDATION.\n"
        )
        f.write(
            "SD1090 test: not accessed.\n"
        )
        f.write(
            "SD1033: not accessed.\n"
        )
        f.write(
            "All four models and all five seeds proceed to Stage 09D.\n"
        )
        f.write(
            "Physics-Compact proposed model remains frozen.\n\n"
        )
        f.write(
            "Selected configurations from 09B\n"
        )
        f.write(
            "-" * 110
            + "\n"
        )
        f.write(
            config_audit_df.to_string(
                index=False
            )
        )
        f.write(
            "\n\nFive-seed outer-validation summary\n"
        )
        f.write(
            "-" * 110
            + "\n"
        )
        f.write(
            seed_summary_df.to_string(
                index=False
            )
        )
        f.write(
            "\n\nInterpretation policy\n"
        )
        f.write(
            "-" * 110
            + "\n"
        )
        f.write(
            "The outer-validation ranking is descriptive confirmation only. "
            "It does not alter any 09B-selected hyperparameter. All four "
            "modern baselines will be frozen and evaluated externally in 09D. "
            "No SD1033 result may be used to select a seed or retune a model.\n"
        )

    log("")
    log(
        "=" * 110
    )
    log(
        "09C MODERN BASELINE OUTER-VALIDATION RESULTS"
    )
    log(
        "=" * 110
    )

    for row in seed_summary_df.itertuples():
        log(
            f"{row.model}:"
        )
        log(
            f"  AW RMSE      = "
            f"{row.mean_validation_AW_RMSE_mps:.4f} "
            f"+/- {row.std_validation_AW_RMSE_mps:.4f} m/s"
        )
        log(
            f"  wind RMSE    = "
            f"{row.mean_mean_wind_vector_RMSE_mps:.4f} "
            f"+/- {row.std_mean_wind_vector_RMSE_mps:.4f} m/s"
        )
        log(
            f"  vessel RMSE  = "
            f"{row.mean_mean_vessel_vector_RMSE_mps:.4f} "
            f"+/- {row.std_mean_vessel_vector_RMSE_mps:.4f} m/s"
        )
        log(
            f"  AWS RMSE     = "
            f"{row.mean_mean_AWS_RMSE_mps:.4f} "
            f"+/- {row.std_mean_AWS_RMSE_mps:.4f} m/s"
        )
        log(
            f"  AWA MAE      = "
            f"{row.mean_mean_AWA_MAE_deg:.2f} "
            f"+/- {row.std_mean_AWA_MAE_deg:.2f} deg"
        )
        log(
            f"  parameters   = "
            f"{int(row.parameters):,}"
        )
        log(
            f"  val rank     = "
            f"{int(row.validation_AW_rank)}"
        )

    log("")
    log(
        f"[DESCRIPTIVE VALIDATION WINNER] "
        f"{best_row['model']} | "
        f"AW={best_row['mean_validation_AW_RMSE_mps']:.4f} "
        f"+/- {best_row['std_validation_AW_RMSE_mps']:.4f} m/s"
    )
    log(
        "[POLICY] This validation ranking does NOT change 09B hyperparameters."
    )
    log(
        "[POLICY] All four models and all five seeds are frozen for 09D."
    )
    log(
        "[POLICY] SD1090 test and SD1033 were not accessed."
    )
    log(
        "[POLICY] Physics-Compact proposed model remains frozen."
    )
    log(
        f"[DONE] 09C outputs: {output_dir}"
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
