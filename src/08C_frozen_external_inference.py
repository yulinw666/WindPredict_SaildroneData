# -*- coding: utf-8 -*-
"""
08C_frozen_external_inference.py

Frozen external inference on SD1033 external datasets built by 08B.

Scientific purpose
------------------
Evaluate whether the forecasting system developed ONLY on SD1090-2024
generalizes to unseen SD1033 missions without any external fitting.

Models
------
1. Persistence
2. Frozen-Ridge
3. Frozen-Compact-Vessel-Residual
4. Frozen-Physics-Compact-Vessel-Residual

For the two compact neural models, ALL frozen 07C confirmation checkpoints
are evaluated independently (default: 5 seeds). No checkpoint is selected
using SD1033.

External datasets
-----------------
Default:
    SD1033_2024_full
    SD1033_2024_matched_SD1090
    SD1033_2023_full
    SD1033_2022_full

Strict frozen-inference policy
------------------------------
This script performs:
    forward inference + metric calculation only.

It NEVER:
    - updates model weights;
    - fits/refits scalers;
    - fine-tunes on SD1033;
    - performs early stopping;
    - selects a seed/checkpoint from external performance;
    - changes hyperparameters using SD1033;
    - interpolates external data.

The 08B NPZ files already contain X/y standardized with the frozen
SD1090 TRAIN scalers. This script audits that the 08B manifest references
the same frozen scaler file before inference.

Compact architecture
--------------------
The true compact vessel residual model predicts residual corrections only for:
    [VESSEL_EAST_MPS, VESSEL_NORTH_MPS] x horizons

Wind and heading remain exactly equal to the frozen Ridge forecast:
    W_hat   = W_Ridge
    HDG_hat = HDG_Ridge

Vessel:
    V_hat = V_Ridge + gate * DeltaV

Apparent wind:
    A_E = U_hat - V_E_hat
    A_N = V_hat_wind - V_N_hat

Metrics
-------
Per horizon:
    wind vector RMSE
    wind speed RMSE
    wind direction MAE
    vessel vector RMSE
    SOG RMSE
    COG MAE
    HDG MAE
    apparent-wind vector RMSE
    AWS RMSE
    AWA MAE

Across horizons:
    mean values using the same frozen evaluation utility as the SD1090
    development pipeline.

External skill relative to frozen Ridge:
    Skill_AW = 1 - RMSE_model / RMSE_Ridge

Frozen-seed reporting
---------------------
For Compact and Physics-Compact:
    mean +/- std across all frozen seeds

Paired seed effects:
    Compact vs Ridge
    Physics-Compact vs Ridge
    Physics-Compact vs Compact

No "best external seed" is selected.

Primary outputs
---------------
external_run_metrics.csv
external_seed_summary.csv
external_per_horizon_metrics.csv
external_per_horizon_summary.csv
external_paired_effects.csv
external_inference_audit.csv
external_inference_manifest.json
external_inference_report.txt

figures/ [optional]

Optional:
    --save-predictions
stores each frozen-seed prediction NPZ. This is off by default because the
external datasets are large.

Dependencies
------------
Place this script in the same src directory as:
    07C_confirm_and_compact_vessel_residual.py
    07B_structured_residual_ablation.py
    07A_physics_guided_residual_forecasting.py
    06_run_physics_constrained_joint_gru.py
    sci_plot_style.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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


SCRIPT_VERSION = "0.1.0-frozen-external-inference"
SHARED_07C = "07C_confirm_and_compact_vessel_residual.py"

DEFAULT_EXTERNAL_DIR = (
    r"D:\project\WindPredict_SaildroneData\data\external"
    r"\SD1033_external_forecasting_v0_1"
)

DEFAULT_COMPACT_DIR = (
    r"D:\project\WindPredict_SaildroneData\data\forecasting"
    r"\SD1090_TPOS2024_JointForecasting_v0_1"
    r"\confirm_and_compact_vessel_residual_v0_1"
)

DEFAULT_FROZEN_DATASET_DIR = (
    r"D:\project\WindPredict_SaildroneData\data\forecasting"
    r"\SD1090_TPOS2024_JointForecasting_v0_1"
)

DEFAULT_DATASETS = [
    "SD1033_2024_full",
    "SD1033_2024_matched_SD1090",
    "SD1033_2023_full",
    "SD1033_2022_full",
]

MODEL_PERSISTENCE = "Persistence"
MODEL_RIDGE = "Frozen-Ridge"
MODEL_COMPACT = "Frozen-Compact-Vessel-Residual"
MODEL_PHYSICS_COMPACT = "Frozen-Physics-Compact-Vessel-Residual"

NEURAL_MODELS = [
    MODEL_COMPACT,
    MODEL_PHYSICS_COMPACT,
]

CHECKPOINT_BASENAMES = {
    MODEL_COMPACT: "Compact_Vessel_Residual.pt",
    MODEL_PHYSICS_COMPACT: "Physics_Compact_Vessel_Residual.pt",
}

FEATURE_NAMES = [
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

TARGET_NAMES = [
    "UWND_MEAN",
    "VWND_MEAN",
    "VESSEL_EAST_MPS",
    "VESSEL_NORTH_MPS",
    "HDG_sin",
    "HDG_cos",
]

EXPECTED_HORIZONS = [1, 2, 3, 5, 10]


def log(message=""):
    print(message, flush=True)


def load_neighbor_module(filename: str, module_name: str):
    path = Path(__file__).resolve().parent / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required shared script not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to import {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
            return {str(k): convert(x) for k, x in v.items()}
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


def sha256_file(path: Path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)

    return h.hexdigest()


def parse_dataset_list(text: str):
    values = []

    for token in text.split(","):
        token = token.strip()
        if token and token not in values:
            values.append(token)

    if not values:
        raise ValueError("No external datasets selected.")

    return values


def load_external_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        required = [
            "X",
            "y",
            "y_raw",
            "apparent_earth_raw",
            "context_start_time_ns",
            "context_end_time_ns",
            "target_time_ns",
            "context_start_row",
            "context_end_row",
            "target_row",
            "segment_id",
            "horizons_min",
            "feature_names",
            "target_names",
            "source_file",
            "dataset_id",
            "scaler_source",
        ]

        missing = [
            key
            for key in required
            if key not in data.files
        ]

        if missing:
            raise RuntimeError(
                f"{path.name}: missing NPZ fields: {missing}"
            )

        out = {
            key: np.array(data[key])
            for key in required
        }

    out["feature_names"] = [
        str(x)
        for x in out["feature_names"].tolist()
    ]

    out["target_names"] = [
        str(x)
        for x in out["target_names"].tolist()
    ]

    out["horizons_min"] = [
        int(x)
        for x in out["horizons_min"].tolist()
    ]

    out["dataset_id"] = str(
        out["dataset_id"].item()
    )

    out["source_file"] = str(
        out["source_file"].item()
    )

    out["scaler_source"] = str(
        out["scaler_source"].item()
    )

    return out


def audit_external_npz(
    external,
    expected_dataset_id,
    frozen_scalers_path: Path,
):
    X = external["X"]
    y = external["y"]
    y_raw = external["y_raw"]
    apparent = external["apparent_earth_raw"]

    checks = {
        "dataset_id_matches": (
            external["dataset_id"]
            == expected_dataset_id
        ),
        "feature_names_match": (
            external["feature_names"]
            == FEATURE_NAMES
        ),
        "target_names_match": (
            external["target_names"]
            == TARGET_NAMES
        ),
        "horizons_match": (
            external["horizons_min"]
            == EXPECTED_HORIZONS
        ),
        "X_shape_ok": (
            X.ndim == 3
            and X.shape[1:] == (60, 11)
        ),
        "y_shape_ok": (
            y.ndim == 3
            and y.shape[1:] == (5, 6)
        ),
        "y_raw_shape_ok": (
            y_raw.shape == y.shape
        ),
        "apparent_shape_ok": (
            apparent.shape
            == (
                X.shape[0],
                5,
                2,
            )
        ),
        "sample_counts_match": (
            X.shape[0]
            == y.shape[0]
            == y_raw.shape[0]
            == apparent.shape[0]
        ),
        "X_all_finite": bool(
            np.isfinite(X).all()
        ),
        "y_all_finite": bool(
            np.isfinite(y).all()
        ),
        "y_raw_all_finite": bool(
            np.isfinite(y_raw).all()
        ),
        "apparent_all_finite": bool(
            np.isfinite(apparent).all()
        ),
        "scaler_source_path_matches": (
            Path(
                external["scaler_source"]
            ).resolve()
            == frozen_scalers_path.resolve()
        ),
    }

    reconstructed = np.stack(
        [
            y_raw[:, :, 0]
            - y_raw[:, :, 2],
            y_raw[:, :, 1]
            - y_raw[:, :, 3],
        ],
        axis=-1,
    )

    checks[
        "apparent_reconstruction_max_abs_error"
    ] = float(
        np.max(
            np.abs(
                reconstructed
                - apparent
            )
        )
    )

    checks[
        "audit_passed"
    ] = bool(
        all(
            bool(value)
            for key, value in checks.items()
            if isinstance(value, (bool, np.bool_))
        )
        and checks[
            "apparent_reconstruction_max_abs_error"
        ] < 1e-6
    )

    return checks


def load_frozen_context(
    shared07c,
    shared07a,
    core06,
    compact_dir: Path,
    frozen_dataset_dir: Path,
):
    compact_manifest_path = (
        compact_dir
        / "compact_manifest.json"
    )

    compact_manifest = load_json(
        compact_manifest_path
    )

    frozen_manifest_path = (
        frozen_dataset_dir
        / "dataset_manifest.json"
    )

    scalers_path = (
        frozen_dataset_dir
        / "scalers.json"
    )

    frozen_manifest = core06.load_json(
        frozen_manifest_path
    )

    scalers = core06.load_json(
        scalers_path
    )

    feature_names = list(
        frozen_manifest[
            "feature_names"
        ]
    )

    target_names = list(
        frozen_manifest[
            "target_names"
        ]
    )

    horizons = [
        int(v)
        for v in frozen_manifest[
            "forecast_horizons_minutes"
        ]
    ]

    if feature_names != FEATURE_NAMES:
        raise RuntimeError(
            "Frozen feature schema mismatch."
        )

    if target_names != TARGET_NAMES:
        raise RuntimeError(
            "Frozen target schema mismatch."
        )

    if horizons != EXPECTED_HORIZONS:
        raise RuntimeError(
            "Frozen horizon schema mismatch."
        )

    (
        f_names,
        feature_mean,
        feature_std,
    ) = core06.parse_scaler(
        scalers,
        "feature_scaler",
    )

    (
        t_names,
        target_mean,
        target_std,
    ) = core06.parse_scaler(
        scalers,
        "target_scaler",
    )

    if f_names != FEATURE_NAMES:
        raise RuntimeError(
            "Feature scaler schema mismatch."
        )

    if t_names != TARGET_NAMES:
        raise RuntimeError(
            "Target scaler schema mismatch."
        )

    seed_schedule = [
        int(v)
        for v in compact_manifest[
            "seed_schedule"
        ]
    ]

    gru_config = dict(
        compact_manifest[
            "fixed_gru_config"
        ]
    )

    ridge_model_path = Path(
        compact_manifest[
            "frozen_ridge_model"
        ]
    )

    if not ridge_model_path.exists():
        raise FileNotFoundError(
            ridge_model_path
        )

    ridge_dir = ridge_model_path.parent

    (
        ridge_model,
        loaded_ridge_path,
        ridge_manifest,
    ) = shared07a.load_frozen_ridge(
        ridge_dir
    )

    if (
        loaded_ridge_path.resolve()
        != ridge_model_path.resolve()
    ):
        raise RuntimeError(
            "Loaded Ridge model does not match 07C frozen Ridge path."
        )

    expected_compact_params = (
        compact_manifest.get(
            "audit",
            {}
        ).get(
            "compact_parameters"
        )
    )

    return {
        "compact_manifest": compact_manifest,
        "compact_manifest_path": compact_manifest_path,
        "frozen_manifest": frozen_manifest,
        "frozen_manifest_path": frozen_manifest_path,
        "scalers_path": scalers_path,
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
        "seed_schedule": seed_schedule,
        "gru_config": gru_config,
        "ridge_model": ridge_model,
        "ridge_model_path": ridge_model_path,
        "ridge_manifest": ridge_manifest,
        "expected_compact_params": (
            int(expected_compact_params)
            if expected_compact_params is not None
            else None
        ),
    }


def summarize_metrics(
    core06,
    metrics_df,
):
    row = (
        core06.summarize_metrics(
            metrics_df
        )
        .iloc[0]
    )

    return {
        "wind_RMSE_mps": float(
            row[
                "mean_wind_vector_RMSE_mps"
            ]
        ),
        "wind_speed_RMSE_mps": float(
            row[
                "mean_wind_speed_RMSE_mps"
            ]
        ),
        "wind_direction_MAE_deg": float(
            row[
                "mean_wind_direction_MAE_deg"
            ]
        ),
        "vessel_RMSE_mps": float(
            row[
                "mean_vessel_vector_RMSE_mps"
            ]
        ),
        "SOG_RMSE_mps": float(
            row[
                "mean_SOG_RMSE_mps"
            ]
        ),
        "COG_MAE_deg": float(
            row[
                "mean_COG_MAE_deg"
            ]
        ),
        "HDG_MAE_deg": float(
            row[
                "mean_HDG_MAE_deg"
            ]
        ),
        "AW_RMSE_mps": float(
            row[
                "mean_apparent_vector_RMSE_mps"
            ]
        ),
        "AWS_RMSE_mps": float(
            row[
                "mean_AWS_RMSE_mps"
            ]
        ),
        "AWA_MAE_deg": float(
            row[
                "mean_AWA_MAE_deg"
            ]
        ),
    }


def instantiate_compact_model(
    CompactModelClass,
    *,
    feature_count,
    horizon_count,
    target_count,
    gru_config,
    device,
):
    # gate_bias and residual_init_std only affect initialization.
    # They are irrelevant after strict checkpoint loading.
    model = CompactModelClass(
        n_features=int(
            feature_count
        ),
        n_horizons=int(
            horizon_count
        ),
        n_targets=int(
            target_count
        ),
        recurrent_units=int(
            gru_config[
                "recurrent_units"
            ]
        ),
        recurrent_layers=int(
            gru_config[
                "recurrent_layers"
            ]
        ),
        dense_units=int(
            gru_config[
                "dense_units"
            ]
        ),
        dropout=float(
            gru_config[
                "dropout"
            ]
        ),
        gate_bias=-1.0,
        residual_init_std=1e-3,
    ).to(
        device
    )

    return model


def infer_compact_batches(
    core06,
    torch,
    *,
    model,
    X,
    ridge_z,
    device,
    batch_size,
    amp_enabled,
):
    n = X.shape[
        0
    ]

    pred_z = np.empty(
        ridge_z.shape,
        dtype=np.float32,
    )

    gate_v_sum = np.zeros(
        (
            ridge_z.shape[
                1
            ],
            2,
        ),
        dtype=np.float64,
    )

    gate_v_sq_sum = np.zeros_like(
        gate_v_sum
    )

    corr_v_sq_sum = np.zeros_like(
        gate_v_sum
    )

    count = 0

    model.eval()

    with torch.no_grad():
        for start in range(
            0,
            n,
            batch_size,
        ):
            end = min(
                n,
                start
                + batch_size,
            )

            Xb = torch.from_numpy(
                np.asarray(
                    X[
                        start:end
                    ],
                    dtype=np.float32,
                )
            ).to(
                device,
                non_blocking=True,
            )

            rb = torch.from_numpy(
                np.asarray(
                    ridge_z[
                        start:end
                    ],
                    dtype=np.float32,
                )
            ).to(
                device,
                non_blocking=True,
            )

            with core06.AutocastContext(
                torch,
                amp_enabled,
                device.type,
            ):
                (
                    combined_z,
                    _,
                    gate_full,
                    correction_full,
                    _,
                ) = model(
                    Xb,
                    rb,
                )

            pred_z[
                start:end
            ] = (
                combined_z
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            gate_v = (
                gate_full[
                    :,
                    :,
                    2:4,
                ]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
            )

            corr_v = (
                correction_full[
                    :,
                    :,
                    2:4,
                ]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
            )

            gate_v_sum += np.sum(
                gate_v,
                axis=0,
            )

            gate_v_sq_sum += np.sum(
                gate_v ** 2,
                axis=0,
            )

            corr_v_sq_sum += np.sum(
                corr_v ** 2,
                axis=0,
            )

            count += (
                end
                - start
            )

            del (
                Xb,
                rb,
                combined_z,
                gate_full,
                correction_full,
                gate_v,
                corr_v,
            )

    gate_mean = (
        gate_v_sum
        / max(
            count,
            1,
        )
    )

    gate_var = (
        gate_v_sq_sum
        / max(
            count,
            1,
        )
        - gate_mean ** 2
    )

    gate_std = np.sqrt(
        np.maximum(
            gate_var,
            0.0,
        )
    )

    correction_z_rms = np.sqrt(
        corr_v_sq_sum
        / max(
            count,
            1,
        )
    )

    return {
        "pred_z": pred_z,
        "gate_v_mean": gate_mean,
        "gate_v_std": gate_std,
        "correction_v_z_rms": correction_z_rms,
    }


def aggregate_seed_summary(
    run_df,
):
    metric_cols = [
        "wind_RMSE_mps",
        "wind_speed_RMSE_mps",
        "wind_direction_MAE_deg",
        "vessel_RMSE_mps",
        "SOG_RMSE_mps",
        "COG_MAE_deg",
        "HDG_MAE_deg",
        "AW_RMSE_mps",
        "AWS_RMSE_mps",
        "AWA_MAE_deg",
        "AW_skill_vs_Ridge",
    ]

    rows = []

    for (
        dataset_id,
        model,
    ), grp in run_df.groupby(
        [
            "dataset_id",
            "model",
        ],
        sort=False,
    ):
        row = {
            "dataset_id": dataset_id,
            "model": model,
            "n_runs": int(
                len(
                    grp
                )
            ),
            "parameters": (
                int(
                    grp[
                        "parameters"
                    ].iloc[
                        0
                    ]
                )
                if grp[
                    "parameters"
                ].notna().any()
                else np.nan
            ),
        }

        for col in metric_cols:
            values = grp[
                col
            ].to_numpy(
                dtype=float
            )

            finite = values[
                np.isfinite(
                    values
                )
            ]

            if len(
                finite
            ) == 0:
                row[
                    f"mean_{col}"
                ] = np.nan

                row[
                    f"std_{col}"
                ] = np.nan

            else:
                row[
                    f"mean_{col}"
                ] = float(
                    np.mean(
                        finite
                    )
                )

                row[
                    f"std_{col}"
                ] = float(
                    np.std(
                        finite,
                        ddof=1,
                    )
                    if len(
                        finite
                    ) > 1
                    else 0.0
                )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def aggregate_horizon_summary(
    horizon_df,
):
    numeric_metrics = [
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
        "AW_skill_vs_Ridge",
    ]

    rows = []

    for (
        dataset_id,
        model,
        horizon,
    ), grp in horizon_df.groupby(
        [
            "dataset_id",
            "model",
            "horizon_min",
        ],
        sort=False,
    ):
        row = {
            "dataset_id": dataset_id,
            "model": model,
            "horizon_min": int(
                horizon
            ),
            "n_runs": int(
                len(
                    grp
                )
            ),
        }

        for col in numeric_metrics:
            if col not in grp.columns:
                continue

            values = grp[
                col
            ].to_numpy(
                dtype=float
            )

            finite = values[
                np.isfinite(
                    values
                )
            ]

            if len(
                finite
            ) == 0:
                row[
                    f"mean_{col}"
                ] = np.nan
                row[
                    f"std_{col}"
                ] = np.nan
            else:
                row[
                    f"mean_{col}"
                ] = float(
                    np.mean(
                        finite
                    )
                )
                row[
                    f"std_{col}"
                ] = float(
                    np.std(
                        finite,
                        ddof=1,
                    )
                    if len(
                        finite
                    ) > 1
                    else 0.0
                )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def paired_external_effects(
    run_df,
):
    comparisons = [
        (
            "Compact_vs_Ridge",
            MODEL_RIDGE,
            MODEL_COMPACT,
        ),
        (
            "PhysicsCompact_vs_Ridge",
            MODEL_RIDGE,
            MODEL_PHYSICS_COMPACT,
        ),
        (
            "PhysicsCompact_vs_Compact",
            MODEL_COMPACT,
            MODEL_PHYSICS_COMPACT,
        ),
    ]

    rows = []

    for dataset_id in (
        run_df[
            "dataset_id"
        ].unique()
    ):
        d = run_df.loc[
            run_df[
                "dataset_id"
            ] == dataset_id
        ]

        for label, model_a, model_b in comparisons:
            b = d.loc[
                d[
                    "model"
                ] == model_b
            ][
                [
                    "seed",
                    "AW_RMSE_mps",
                ]
            ].rename(
                columns={
                    "AW_RMSE_mps":
                    "AW_B"
                }
            )

            if b.empty:
                continue

            if model_a == MODEL_RIDGE:
                ridge = d.loc[
                    d[
                        "model"
                    ] == MODEL_RIDGE
                ]

                if ridge.empty:
                    continue

                AW_A = float(
                    ridge.iloc[
                        0
                    ][
                        "AW_RMSE_mps"
                    ]
                )

                paired = b.copy()
                paired[
                    "AW_A"
                ] = AW_A

            else:
                a = d.loc[
                    d[
                        "model"
                    ] == model_a
                ][
                    [
                        "seed",
                        "AW_RMSE_mps",
                    ]
                ].rename(
                    columns={
                        "AW_RMSE_mps":
                        "AW_A"
                    }
                )

                paired = a.merge(
                    b,
                    on="seed",
                    how="inner",
                    validate="one_to_one",
                )

            if paired.empty:
                continue

            diff = (
                paired[
                    "AW_A"
                ].to_numpy(
                    dtype=float
                )
                - paired[
                    "AW_B"
                ].to_numpy(
                    dtype=float
                )
            )

            pct = (
                100.0
                * diff
                / paired[
                    "AW_A"
                ].to_numpy(
                    dtype=float
                )
            )

            n = len(
                diff
            )

            mean_diff = float(
                np.mean(
                    diff
                )
            )

            std_diff = float(
                np.std(
                    diff,
                    ddof=1,
                )
                if n > 1
                else 0.0
            )

            ci_low = np.nan
            ci_high = np.nan

            if n > 1:
                try:
                    from scipy.stats import t

                    critical = float(
                        t.ppf(
                            0.975,
                            df=n - 1,
                        )
                    )

                    half = (
                        critical
                        * std_diff
                        / math.sqrt(
                            n
                        )
                    )

                    ci_low = (
                        mean_diff
                        - half
                    )

                    ci_high = (
                        mean_diff
                        + half
                    )

                except Exception:
                    pass

            rows.append({
                "dataset_id": dataset_id,
                "comparison": label,
                "model_A_reference": model_a,
                "model_B_candidate": model_b,
                "n_paired_runs": int(
                    n
                ),
                "mean_AW_reduction_B_vs_A_mps": mean_diff,
                "std_AW_reduction_mps": std_diff,
                "mean_AW_reduction_percent": float(
                    np.mean(
                        pct
                    )
                ),
                "median_AW_reduction_percent": float(
                    np.median(
                        pct
                    )
                ),
                "B_win_count": int(
                    np.sum(
                        diff > 0
                    )
                ),
                "A_win_count": int(
                    np.sum(
                        diff < 0
                    )
                ),
                "tie_count": int(
                    np.sum(
                        np.isclose(
                            diff,
                            0.0,
                            atol=1e-12,
                        )
                    )
                ),
                "paired_95pct_CI_low_mps": ci_low,
                "paired_95pct_CI_high_mps": ci_high,
            })

    return pd.DataFrame(
        rows
    )


def load_style():
    path = (
        Path(__file__).resolve().parent
        / "sci_plot_style.py"
    )

    if not path.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        "sci_plot_style_08c",
        str(
            path
        ),
    )

    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def make_plots(
    *,
    seed_summary,
    horizon_summary,
    paired_df,
    output_dir,
):
    import matplotlib.pyplot as plt

    style = load_style()

    fig_dir = (
        output_dir
        / "figures"
    )

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    def clean(ax):
        if (
            style is not None
            and hasattr(
                style,
                "clean_axis",
            )
        ):
            style.clean_axis(
                ax,
                grid=True,
            )
        else:
            ax.tick_params(
                which="both",
                direction="in",
                top=True,
                right=True,
            )
            ax.grid(
                True,
                alpha=0.25,
            )

    def save(fig, name):
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

    model_order = [
        MODEL_PERSISTENCE,
        MODEL_RIDGE,
        MODEL_COMPACT,
        MODEL_PHYSICS_COMPACT,
    ]

    dataset_order = [
        d
        for d in DEFAULT_DATASETS
        if d in set(
            seed_summary[
                "dataset_id"
            ]
        )
    ]

    # 01 External AW mean +/- std.
    fig, ax = plt.subplots(
        figsize=(
            7.4,
            3.8,
        )
    )

    x = np.arange(
        len(
            dataset_order
        )
    )

    width = 0.8 / len(
        model_order
    )

    for i, model in enumerate(
        model_order
    ):
        means = []
        stds = []

        for dataset_id in dataset_order:
            row = seed_summary.loc[
                (
                    seed_summary[
                        "dataset_id"
                    ] == dataset_id
                )
                & (
                    seed_summary[
                        "model"
                    ] == model
                )
            ]

            if row.empty:
                means.append(
                    np.nan
                )
                stds.append(
                    np.nan
                )
            else:
                means.append(
                    float(
                        row.iloc[
                            0
                        ][
                            "mean_AW_RMSE_mps"
                        ]
                    )
                )
                stds.append(
                    float(
                        row.iloc[
                            0
                        ][
                            "std_AW_RMSE_mps"
                        ]
                    )
                )

        pos = (
            x
            + (
                i
                - (
                    len(
                        model_order
                    )
                    - 1
                )
                / 2.0
            )
            * width
        )

        ax.bar(
            pos,
            means,
            width=width,
            label=model,
            yerr=stds,
            capsize=2,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        dataset_order,
        rotation=25,
        ha="right",
    )

    ax.set_ylabel(
        r"AW vector RMSE (m s$^{-1}$)"
    )

    ax.legend(
        loc="best",
        ncol=2,
    )

    clean(
        ax
    )

    save(
        fig,
        "01_external_aw_rmse",
    )

    # 02 AW skill vs Ridge for two neural models.
    fig, ax = plt.subplots(
        figsize=(
            6.7,
            3.5,
        )
    )

    x = np.arange(
        len(
            dataset_order
        )
    )

    width = 0.35

    for i, model in enumerate(
        [
            MODEL_COMPACT,
            MODEL_PHYSICS_COMPACT,
        ]
    ):
        values = []

        for dataset_id in dataset_order:
            row = seed_summary.loc[
                (
                    seed_summary[
                        "dataset_id"
                    ] == dataset_id
                )
                & (
                    seed_summary[
                        "model"
                    ] == model
                )
            ]

            values.append(
                100.0
                * float(
                    row.iloc[
                        0
                    ][
                        "mean_AW_skill_vs_Ridge"
                    ]
                )
                if not row.empty
                else np.nan
            )

        ax.bar(
            x
            + (
                i
                - 0.5
            )
            * width,
            values,
            width=width,
            label=model,
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        dataset_order,
        rotation=25,
        ha="right",
    )

    ax.set_ylabel(
        "AW skill vs frozen Ridge (%)"
    )

    ax.legend(
        loc="best"
    )

    clean(
        ax
    )

    save(
        fig,
        "02_external_aw_skill_vs_ridge",
    )

    # 03 Per-horizon Physics-Compact vs Ridge for every dataset.
    for dataset_id in dataset_order:
        fig, ax = plt.subplots(
            figsize=(
                5.2,
                3.3,
            )
        )

        for model in [
            MODEL_RIDGE,
            MODEL_COMPACT,
            MODEL_PHYSICS_COMPACT,
        ]:
            grp = horizon_summary.loc[
                (
                    horizon_summary[
                        "dataset_id"
                    ] == dataset_id
                )
                & (
                    horizon_summary[
                        "model"
                    ] == model
                )
            ].sort_values(
                "horizon_min"
            )

            if grp.empty:
                continue

            ax.errorbar(
                grp[
                    "horizon_min"
                ],
                grp[
                    "mean_apparent_vector_RMSE_mps"
                ],
                yerr=grp[
                    "std_apparent_vector_RMSE_mps"
                ],
                marker="o",
                capsize=2,
                label=model,
            )

        ax.set_xlabel(
            "Forecast horizon (min)"
        )

        ax.set_ylabel(
            r"AW vector RMSE (m s$^{-1}$)"
        )

        ax.set_xticks(
            EXPECTED_HORIZONS
        )

        ax.legend(
            loc="best"
        )

        clean(
            ax
        )

        save(
            fig,
            "03_"
            + dataset_id
            + "_per_horizon_aw",
        )

    # 04 Vessel RMSE.
    fig, ax = plt.subplots(
        figsize=(
            7.0,
            3.6,
        )
    )

    x = np.arange(
        len(
            dataset_order
        )
    )

    width = 0.24

    models = [
        MODEL_RIDGE,
        MODEL_COMPACT,
        MODEL_PHYSICS_COMPACT,
    ]

    for i, model in enumerate(
        models
    ):
        values = []

        for dataset_id in dataset_order:
            row = seed_summary.loc[
                (
                    seed_summary[
                        "dataset_id"
                    ] == dataset_id
                )
                & (
                    seed_summary[
                        "model"
                    ] == model
                )
            ]

            values.append(
                float(
                    row.iloc[
                        0
                    ][
                        "mean_vessel_RMSE_mps"
                    ]
                )
                if not row.empty
                else np.nan
            )

        ax.bar(
            x
            + (
                i
                - 1
            )
            * width,
            values,
            width=width,
            label=model,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        dataset_order,
        rotation=25,
        ha="right",
    )

    ax.set_ylabel(
        r"Vessel vector RMSE (m s$^{-1}$)"
    )

    ax.legend(
        loc="best"
    )

    clean(
        ax
    )

    save(
        fig,
        "04_external_vessel_rmse",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--external-dir",
        default=DEFAULT_EXTERNAL_DIR,
    )

    parser.add_argument(
        "--compact-dir",
        default=DEFAULT_COMPACT_DIR,
    )

    parser.add_argument(
        "--frozen-dataset-dir",
        default=DEFAULT_FROZEN_DATASET_DIR,
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--datasets",
        default=",".join(
            DEFAULT_DATASETS
        ),
        help="Comma-separated external dataset IDs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--ridge-chunk-size",
        type=int,
        default=16384,
    )

    parser.add_argument(
        "--direction-min-speed",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--cog-min-speed",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    parser.add_argument(
        "--no-tf32",
        action="store_true",
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
    )

    parser.add_argument(
        "--plots",
        action="store_true",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be > 0."
        )

    dataset_ids = parse_dataset_list(
        args.datasets
    )

    external_dir = Path(
        args.external_dir
    )

    datasets_dir = (
        external_dir
        / "datasets"
    )

    compact_dir = Path(
        args.compact_dir
    )

    frozen_dataset_dir = Path(
        args.frozen_dataset_dir
    )

    output_dir = (
        Path(
            args.output_dir
        )
        if args.output_dir
        else (
            external_dir
            / "frozen_external_inference_v0_1"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_dir = (
        output_dir
        / "predictions"
    )

    if args.save_predictions:
        prediction_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    shared07c = load_neighbor_module(
        SHARED_07C,
        "shared07c_for_08c",
    )

    shared07b = (
        shared07c.load_shared_07b()
    )

    shared07a = (
        shared07b.load_shared_07a()
    )

    core06 = (
        shared07a.load_core()
    )

    frozen = load_frozen_context(
        shared07c,
        shared07a,
        core06,
        compact_dir,
        frozen_dataset_dir,
    )

    external_manifest_path = (
        external_dir
        / "external_dataset_manifest.json"
    )

    external_manifest = load_json(
        external_manifest_path
    )

    # Strong scaler audit.
    manifest_scaler_path = Path(
        external_manifest[
            "frozen_scalers"
        ]
    )

    if (
        manifest_scaler_path.resolve()
        != frozen[
            "scalers_path"
        ].resolve()
    ):
        raise RuntimeError(
            "08B external datasets were not built with the scaler "
            "currently frozen for 08C."
        )

    expected_hash = (
        external_manifest.get(
            "frozen_scalers_sha256"
        )
    )

    actual_hash = sha256_file(
        frozen[
            "scalers_path"
        ]
    )

    if (
        expected_hash is not None
        and expected_hash
        != actual_hash
    ):
        raise RuntimeError(
            "Frozen scaler SHA256 mismatch between 08B and current file."
        )

    (
        torch,
        nn,
        _,
        _,
    ) = core06.import_torch()

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

    amp_enabled = bool(
        device.type == "cuda"
        and not args.no_amp
    )

    tf32_enabled = bool(
        device.type == "cuda"
        and not args.no_tf32
    )

    core06.configure_tf32(
        torch,
        tf32_enabled,
    )

    device_info = (
        core06.log_device_info(
            torch,
            device,
            amp_enabled,
            tf32_enabled,
        )
    )

    CompactModelClass = (
        shared07c.build_compact_model_class(
            torch,
            nn,
        )
    )

    log(
        "=" * 110
    )

    log(
        "08C - FROZEN EXTERNAL INFERENCE"
    )

    log(
        "=" * 110
    )

    log(
        f"script_version          : {SCRIPT_VERSION}"
    )

    log(
        f"external datasets       : {dataset_ids}"
    )

    log(
        f"frozen seed schedule    : {frozen['seed_schedule']}"
    )

    log(
        f"frozen Ridge            : {frozen['ridge_model_path']}"
    )

    log(
        f"compact model dir       : {compact_dir}"
    )

    log(
        f"external scaler fitting : NEVER"
    )

    log(
        f"external weight updates : NEVER"
    )

    log(
        f"device                  : {device}"
    )

    log("")

    run_rows = []
    horizon_frames = []
    audit_rows = []

    for dataset_index, dataset_id in enumerate(
        dataset_ids,
        start=1,
    ):
        log("")
        log(
            "#" * 110
        )

        log(
            f"DATASET {dataset_index}/{len(dataset_ids)}: {dataset_id}"
        )

        log(
            "#" * 110
        )

        npz_path = (
            datasets_dir
            / (
                dataset_id
                + ".npz"
            )
        )

        load_t0 = time.perf_counter()

        external = load_external_npz(
            npz_path
        )

        audit = audit_external_npz(
            external,
            dataset_id,
            frozen[
                "scalers_path"
            ],
        )

        audit_rows.append({
            "dataset_id": dataset_id,
            **audit,
        })

        if not audit[
            "audit_passed"
        ]:
            failed = [
                key
                for key, value in audit.items()
                if (
                    isinstance(
                        value,
                        (
                            bool,
                            np.bool_,
                        ),
                    )
                    and not bool(
                        value
                    )
                )
            ]

            raise RuntimeError(
                f"{dataset_id}: external NPZ audit failed: {failed}"
            )

        X = np.asarray(
            external[
                "X"
            ],
            dtype=np.float32,
        )

        y_raw = np.asarray(
            external[
                "y_raw"
            ],
            dtype=np.float32,
        )

        apparent_raw = np.asarray(
            external[
                "apparent_earth_raw"
            ],
            dtype=np.float32,
        )

        n = X.shape[
            0
        ]

        log(
            f"[LOAD] {n:,} windows | "
            f"{time.perf_counter() - load_t0:.2f}s"
        )

        # --------------------------------------------------------------
        # Deterministic baselines
        # --------------------------------------------------------------
        log(
            "[BASELINE] Persistence"
        )

        persistence_raw = (
            core06.persistence_joint_prediction(
                X,
                FEATURE_NAMES,
                frozen[
                    "feature_mean"
                ],
                frozen[
                    "feature_std"
                ],
                len(
                    EXPECTED_HORIZONS
                ),
            )
        )

        persistence_metrics = (
            core06.evaluate_model(
                truth_joint=y_raw,
                pred_joint=persistence_raw,
                truth_apparent_ref=apparent_raw,
                horizons=tuple(
                    EXPECTED_HORIZONS
                ),
                split_name=dataset_id,
                model_name=MODEL_PERSISTENCE,
                direction_min_speed=args.direction_min_speed,
                cog_min_speed=args.cog_min_speed,
                persistence_joint=persistence_raw,
            )
        )

        persistence_summary = (
            summarize_metrics(
                core06,
                persistence_metrics,
            )
        )

        log(
            "[BASELINE] Frozen Ridge"
        )

        ridge_z = (
            shared07a.ridge_predict_z(
                frozen[
                    "ridge_model"
                ],
                X,
                len(
                    EXPECTED_HORIZONS
                ),
                len(
                    TARGET_NAMES
                ),
                args.ridge_chunk_size,
            )
        )

        ridge_raw = (
            shared07a.inverse_target(
                ridge_z,
                frozen[
                    "target_mean"
                ],
                frozen[
                    "target_std"
                ],
            )
        )

        ridge_metrics = (
            core06.evaluate_model(
                truth_joint=y_raw,
                pred_joint=ridge_raw,
                truth_apparent_ref=apparent_raw,
                horizons=tuple(
                    EXPECTED_HORIZONS
                ),
                split_name=dataset_id,
                model_name=MODEL_RIDGE,
                direction_min_speed=args.direction_min_speed,
                cog_min_speed=args.cog_min_speed,
                persistence_joint=persistence_raw,
            )
        )

        ridge_summary = (
            summarize_metrics(
                core06,
                ridge_metrics,
            )
        )

        ridge_aw = ridge_summary[
            "AW_RMSE_mps"
        ]

        run_rows.append({
            "dataset_id": dataset_id,
            "model": MODEL_PERSISTENCE,
            "seed": np.nan,
            "parameters": np.nan,
            "checkpoint": None,
            "inference_seconds": np.nan,
            **persistence_summary,
            "AW_skill_vs_Ridge": float(
                1.0
                - persistence_summary[
                    "AW_RMSE_mps"
                ]
                / ridge_aw
            ),
        })

        run_rows.append({
            "dataset_id": dataset_id,
            "model": MODEL_RIDGE,
            "seed": np.nan,
            "parameters": np.nan,
            "checkpoint": str(
                frozen[
                    "ridge_model_path"
                ]
            ),
            "inference_seconds": np.nan,
            **ridge_summary,
            "AW_skill_vs_Ridge": 0.0,
        })

        # Add deterministic per-horizon rows.
        persistence_h = (
            persistence_metrics.copy()
        )

        persistence_h[
            "dataset_id"
        ] = dataset_id

        persistence_h[
            "model"
        ] = MODEL_PERSISTENCE

        persistence_h[
            "seed"
        ] = np.nan

        ridge_per_horizon_aw = (
            ridge_metrics[
                [
                    "horizon_min",
                    "apparent_vector_RMSE_mps",
                ]
            ]
            .set_index(
                "horizon_min"
            )[
                "apparent_vector_RMSE_mps"
            ]
            .to_dict()
        )

        persistence_h[
            "AW_skill_vs_Ridge"
        ] = [
            1.0
            - float(
                row[
                    "apparent_vector_RMSE_mps"
                ]
            )
            / float(
                ridge_per_horizon_aw[
                    int(
                        row[
                            "horizon_min"
                        ]
                    )
                ]
            )
            for _, row in persistence_h.iterrows()
        ]

        horizon_frames.append(
            persistence_h
        )

        ridge_h = (
            ridge_metrics.copy()
        )

        ridge_h[
            "dataset_id"
        ] = dataset_id

        ridge_h[
            "model"
        ] = MODEL_RIDGE

        ridge_h[
            "seed"
        ] = np.nan

        ridge_h[
            "AW_skill_vs_Ridge"
        ] = 0.0

        horizon_frames.append(
            ridge_h
        )

        log(
            f"[RIDGE] AW={ridge_summary['AW_RMSE_mps']:.4f} | "
            f"wind={ridge_summary['wind_RMSE_mps']:.4f} | "
            f"vessel={ridge_summary['vessel_RMSE_mps']:.4f}"
        )

        # --------------------------------------------------------------
        # Frozen 07C compact checkpoints
        # --------------------------------------------------------------
        for model_name in NEURAL_MODELS:
            checkpoint_name = (
                CHECKPOINT_BASENAMES[
                    model_name
                ]
            )

            for seed_index, seed in enumerate(
                frozen[
                    "seed_schedule"
                ],
                start=1,
            ):
                checkpoint_path = (
                    compact_dir
                    / "models"
                    / f"seed_{seed}"
                    / checkpoint_name
                )

                if not checkpoint_path.exists():
                    raise FileNotFoundError(
                        checkpoint_path
                    )

                log(
                    f"[INFER] {model_name} | "
                    f"seed {seed_index}/{len(frozen['seed_schedule'])} "
                    f"({seed})"
                )

                model = instantiate_compact_model(
                    CompactModelClass,
                    feature_count=len(
                        FEATURE_NAMES
                    ),
                    horizon_count=len(
                        EXPECTED_HORIZONS
                    ),
                    target_count=len(
                        TARGET_NAMES
                    ),
                    gru_config=frozen[
                        "gru_config"
                    ],
                    device=device,
                )

                n_params = (
                    shared07a.count_parameters(
                        model
                    )
                )

                if (
                    frozen[
                        "expected_compact_params"
                    ]
                    is not None
                    and n_params
                    != frozen[
                        "expected_compact_params"
                    ]
                ):
                    raise RuntimeError(
                        f"Compact parameter mismatch: "
                        f"{n_params} != "
                        f"{frozen['expected_compact_params']}"
                    )

                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=device,
                    weights_only=False,
                )

                model.load_state_dict(
                    checkpoint[
                        "state_dict"
                    ],
                    strict=True,
                )

                infer_t0 = (
                    time.perf_counter()
                )

                result = infer_compact_batches(
                    core06,
                    torch,
                    model=model,
                    X=X,
                    ridge_z=ridge_z,
                    device=device,
                    batch_size=args.batch_size,
                    amp_enabled=amp_enabled,
                )

                infer_seconds = (
                    time.perf_counter()
                    - infer_t0
                )

                pred_z = result[
                    "pred_z"
                ]

                # Structural audit:
                # compact model is not allowed to change wind or heading.
                fixed_channels = [
                    0,
                    1,
                    4,
                    5,
                ]

                frozen_channel_error = float(
                    np.max(
                        np.abs(
                            pred_z[
                                :,
                                :,
                                fixed_channels
                            ]
                            - ridge_z[
                                :,
                                :,
                                fixed_channels
                            ]
                        )
                    )
                )

                if frozen_channel_error > 1e-7:
                    raise RuntimeError(
                        f"{model_name} seed {seed}: compact structural "
                        f"audit failed; frozen-channel difference="
                        f"{frozen_channel_error}"
                    )

                pred_raw = (
                    shared07a.inverse_target(
                        pred_z,
                        frozen[
                            "target_mean"
                        ],
                        frozen[
                            "target_std"
                        ],
                    )
                )

                metrics = (
                    core06.evaluate_model(
                        truth_joint=y_raw,
                        pred_joint=pred_raw,
                        truth_apparent_ref=apparent_raw,
                        horizons=tuple(
                            EXPECTED_HORIZONS
                        ),
                        split_name=dataset_id,
                        model_name=model_name,
                        direction_min_speed=args.direction_min_speed,
                        cog_min_speed=args.cog_min_speed,
                        persistence_joint=persistence_raw,
                    )
                )

                summary = (
                    summarize_metrics(
                        core06,
                        metrics,
                    )
                )

                aw_skill = float(
                    1.0
                    - summary[
                        "AW_RMSE_mps"
                    ]
                    / ridge_aw
                )

                run_rows.append({
                    "dataset_id": dataset_id,
                    "model": model_name,
                    "seed": int(
                        seed
                    ),
                    "seed_index": int(
                        seed_index
                    ),
                    "parameters": int(
                        n_params
                    ),
                    "checkpoint": str(
                        checkpoint_path
                    ),
                    "checkpoint_epoch": checkpoint.get(
                        "epoch"
                    ),
                    "checkpoint_validation_AW_RMSE_mps": checkpoint.get(
                        "best_validation_AW_RMSE_mps"
                    ),
                    "inference_seconds": float(
                        infer_seconds
                    ),
                    "frozen_wind_heading_max_abs_difference_z": (
                        frozen_channel_error
                    ),
                    "mean_vessel_gate": float(
                        np.mean(
                            result[
                                "gate_v_mean"
                            ]
                        )
                    ),
                    "mean_vessel_correction_z_RMS": float(
                        np.mean(
                            result[
                                "correction_v_z_rms"
                            ]
                        )
                    ),
                    **summary,
                    "AW_skill_vs_Ridge": aw_skill,
                })

                h = metrics.copy()

                h[
                    "dataset_id"
                ] = dataset_id

                h[
                    "model"
                ] = model_name

                h[
                    "seed"
                ] = int(
                    seed
                )

                h[
                    "AW_skill_vs_Ridge"
                ] = [
                    1.0
                    - float(
                        row[
                            "apparent_vector_RMSE_mps"
                        ]
                    )
                    / float(
                        ridge_per_horizon_aw[
                            int(
                                row[
                                    "horizon_min"
                            ]
                            )
                        ]
                    )
                    for _, row in h.iterrows()
                ]

                horizon_frames.append(
                    h
                )

                if args.save_predictions:
                    np.savez_compressed(
                        prediction_dir
                        / (
                            dataset_id
                            + "__"
                            + model_name.replace(
                                "-",
                                "_",
                            )
                            + f"__seed_{seed}.npz"
                        ),
                        pred_raw=pred_raw.astype(
                            np.float32
                        ),
                        target_raw=y_raw,
                        apparent_truth=apparent_raw,
                        context_end_time_ns=external[
                            "context_end_time_ns"
                        ],
                        target_time_ns=external[
                            "target_time_ns"
                        ],
                        segment_id=external[
                            "segment_id"
                        ],
                        seed=np.asarray(
                            seed,
                            dtype=np.int64,
                        ),
                        model=np.asarray(
                            model_name
                        ),
                        dataset_id=np.asarray(
                            dataset_id
                        ),
                    )

                log(
                    f"        AW={summary['AW_RMSE_mps']:.4f} | "
                    f"skill_vs_Ridge={100.0 * aw_skill:+.3f}% | "
                    f"vessel={summary['vessel_RMSE_mps']:.4f} | "
                    f"{infer_seconds:.2f}s"
                )

                del (
                    model,
                    checkpoint,
                    result,
                    pred_z,
                    pred_raw,
                    metrics,
                    h,
                )

                gc.collect()

                if device.type == "cuda":
                    torch.cuda.empty_cache()

        del (
            external,
            X,
            y_raw,
            apparent_raw,
            persistence_raw,
            persistence_metrics,
            ridge_z,
            ridge_raw,
            ridge_metrics,
            persistence_h,
            ridge_h,
        )

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Aggregate external metrics
    # ------------------------------------------------------------------
    run_df = pd.DataFrame(
        run_rows
    )

    horizon_df = pd.concat(
        horizon_frames,
        ignore_index=True,
    )

    seed_summary = aggregate_seed_summary(
        run_df
    )

    horizon_summary = (
        aggregate_horizon_summary(
            horizon_df
        )
    )

    paired_df = paired_external_effects(
        run_df
    )

    audit_df = pd.DataFrame(
        audit_rows
    )

    run_df.to_csv(
        output_dir
        / "external_run_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    seed_summary.to_csv(
        output_dir
        / "external_seed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_df.to_csv(
        output_dir
        / "external_per_horizon_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_summary.to_csv(
        output_dir
        / "external_per_horizon_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired_df.to_csv(
        output_dir
        / "external_paired_effects.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit_df.to_csv(
        output_dir
        / "external_inference_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest_out = {
        "script_version": SCRIPT_VERSION,
        "purpose": (
            "Frozen external inference only; no SD1033 fitting, "
            "training, tuning, checkpoint selection or preprocessing fit."
        ),
        "external_dir": str(
            external_dir
        ),
        "external_manifest": str(
            external_manifest_path
        ),
        "compact_dir": str(
            compact_dir
        ),
        "compact_manifest": str(
            frozen[
                "compact_manifest_path"
            ]
        ),
        "frozen_dataset_dir": str(
            frozen_dataset_dir
        ),
        "frozen_scalers": str(
            frozen[
                "scalers_path"
            ]
        ),
        "frozen_scalers_sha256": actual_hash,
        "frozen_ridge_model": str(
            frozen[
                "ridge_model_path"
            ]
        ),
        "seed_schedule": frozen[
            "seed_schedule"
        ],
        "fixed_gru_config": frozen[
            "gru_config"
        ],
        "expected_compact_parameters": frozen[
            "expected_compact_params"
        ],
        "datasets": dataset_ids,
        "models": [
            MODEL_PERSISTENCE,
            MODEL_RIDGE,
            MODEL_COMPACT,
            MODEL_PHYSICS_COMPACT,
        ],
        "protocol": {
            "external_scaler_fit": False,
            "external_training": False,
            "external_fine_tuning": False,
            "external_early_stopping": False,
            "external_seed_selection": False,
            "all_frozen_seeds_evaluated": True,
            "checkpoint_selection_basis": (
                "07C SD1090 development-stage training/validation only"
            ),
            "external_results_used_to_update_model": False,
        },
        "metrics": {
            "direction_min_speed_mps": float(
                args.direction_min_speed
            ),
            "cog_min_speed_mps": float(
                args.cog_min_speed
            ),
            "AW_skill_vs_Ridge": (
                "1 - model_AW_RMSE / frozen_Ridge_AW_RMSE"
            ),
        },
        "device_info": device_info,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "outputs": {
            "run_metrics": "external_run_metrics.csv",
            "seed_summary": "external_seed_summary.csv",
            "per_horizon_metrics": "external_per_horizon_metrics.csv",
            "per_horizon_summary": "external_per_horizon_summary.csv",
            "paired_effects": "external_paired_effects.csv",
            "audit": "external_inference_audit.csv",
            "report": "external_inference_report.txt",
        },
    }

    save_json(
        output_dir
        / "external_inference_manifest.json",
        manifest_out,
    )

    with (
        output_dir
        / "external_inference_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "08C Frozen External Inference Report\n"
        )

        f.write(
            "=" * 110
            + "\n\n"
        )

        f.write(
            "Policy\n"
        )

        f.write(
            "-" * 110
            + "\n"
        )

        f.write(
            "NO external scaler fitting.\n"
        )

        f.write(
            "NO external training or fine-tuning.\n"
        )

        f.write(
            "NO external early stopping.\n"
        )

        f.write(
            "NO seed/checkpoint selection using external results.\n"
        )

        f.write(
            "All frozen 07C compact-model seeds are evaluated.\n\n"
        )

        f.write(
            "External seed summary\n"
        )

        f.write(
            "-" * 110
            + "\n"
        )

        f.write(
            seed_summary.to_string(
                index=False
            )
        )

        f.write(
            "\n\nPaired external effects\n"
        )

        f.write(
            "-" * 110
            + "\n"
        )

        f.write(
            paired_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nExternal NPZ audit\n"
        )

        f.write(
            "-" * 110
            + "\n"
        )

        f.write(
            audit_df.to_string(
                index=False
            )
        )

        f.write(
            "\n"
        )

    if args.plots:
        make_plots(
            seed_summary=seed_summary,
            horizon_summary=horizon_summary,
            paired_df=paired_df,
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------------
    log("")
    log(
        "=" * 110
    )

    log(
        "08C FROZEN EXTERNAL INFERENCE RESULTS"
    )

    log(
        "=" * 110
    )

    for dataset_id in dataset_ids:
        log("")
        log(
            f"{dataset_id}:"
        )

        d = seed_summary.loc[
            seed_summary[
                "dataset_id"
            ] == dataset_id
        ]

        for model in [
            MODEL_PERSISTENCE,
            MODEL_RIDGE,
            MODEL_COMPACT,
            MODEL_PHYSICS_COMPACT,
        ]:
            row = d.loc[
                d[
                    "model"
                ] == model
            ]

            if row.empty:
                continue

            r = row.iloc[
                0
            ]

            if model in NEURAL_MODELS:
                log(
                    f"  {model}:"
                )

                log(
                    f"    AW RMSE      = "
                    f"{r['mean_AW_RMSE_mps']:.4f} "
                    f"+/- {r['std_AW_RMSE_mps']:.4f} m/s"
                )

                log(
                    f"    vessel RMSE  = "
                    f"{r['mean_vessel_RMSE_mps']:.4f} "
                    f"+/- {r['std_vessel_RMSE_mps']:.4f} m/s"
                )

                log(
                    f"    AWS RMSE     = "
                    f"{r['mean_AWS_RMSE_mps']:.4f} "
                    f"+/- {r['std_AWS_RMSE_mps']:.4f} m/s"
                )

                log(
                    f"    AWA MAE      = "
                    f"{r['mean_AWA_MAE_deg']:.2f} "
                    f"+/- {r['std_AWA_MAE_deg']:.2f} deg"
                )

                log(
                    f"    AW skill/Ridge = "
                    f"{100.0 * r['mean_AW_skill_vs_Ridge']:+.3f}%"
                )

            else:
                log(
                    f"  {model}: "
                    f"AW={r['mean_AW_RMSE_mps']:.4f} | "
                    f"vessel={r['mean_vessel_RMSE_mps']:.4f} | "
                    f"AWS={r['mean_AWS_RMSE_mps']:.4f} | "
                    f"AWA={r['mean_AWA_MAE_deg']:.2f} deg"
                )

        effects = paired_df.loc[
            paired_df[
                "dataset_id"
            ] == dataset_id
        ]

        for _, effect in (
            effects.iterrows()
        ):
            ci_text = ""

            if (
                pd.notna(
                    effect[
                        "paired_95pct_CI_low_mps"
                    ]
                )
                and pd.notna(
                    effect[
                        "paired_95pct_CI_high_mps"
                    ]
                )
            ):
                ci_text = (
                    f" | 95% CI="
                    f"[{effect['paired_95pct_CI_low_mps']:.5f}, "
                    f"{effect['paired_95pct_CI_high_mps']:.5f}]"
                )

            log(
                f"  [PAIRED] {effect['comparison']}: "
                f"AW reduction="
                f"{effect['mean_AW_reduction_B_vs_A_mps']:.5f} m/s "
                f"({effect['mean_AW_reduction_percent']:+.3f}%) | "
                f"wins={int(effect['B_win_count'])}/"
                f"{int(effect['n_paired_runs'])}"
                f"{ci_text}"
            )

    log("")
    log(
        "[FROZEN POLICY] No SD1033 fitting/training/tuning/checkpoint selection was performed."
    )

    log(
        f"[DEVICE] {device}"
    )

    if device.type == "cuda":
        log(
            f"[GPU] {torch.cuda.get_device_name(device)}"
        )

    log(
        f"[DONE] 08C outputs: {output_dir}"
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
