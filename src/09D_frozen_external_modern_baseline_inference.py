# -*- coding: utf-8 -*-
"""
09D_frozen_external_modern_baseline_inference.py

Frozen SD1033 external inference for the four Stage-09 modern comparison
baselines:

    DLinear
    iTransformer
    TimeMixer
    PatchTST

The script evaluates ALL five frozen Stage-09C checkpoints for each model on
the already frozen SD1033 external datasets built in Stage 08B.

SCIENTIFIC POLICY
-----------------
This is inference only. It NEVER:
    - trains or fine-tunes a model;
    - updates any weight;
    - fits/refits a scaler;
    - performs early stopping;
    - selects a seed from SD1033;
    - changes any 09B hyperparameter;
    - changes the proposed Physics-Compact model;
    - interpolates external observations.

The same five seeds are evaluated for every modern baseline. Results are
reported as mean +/- std across all frozen seeds.

External datasets
-----------------
Default:
    SD1033_2024_full
    SD1033_2024_matched_SD1090
    SD1033_2023_full
    SD1033_2022_full

The 2024 matched dataset is the primary same-mission / matched-time /
different-vehicle transfer test. The 2023 and 2022 full datasets are primary
cross-year / cross-mission tests. SD1033_2024_full is supplementary because it
overlaps with the matched-period dataset.

FAIR COMPARISON
---------------
All modern models use exactly:
    X : [B, 60, 11]
    y : [B, 5, 6]

and the same deterministic apparent-wind reconstruction:
    A_E = U - V_ship,E
    A_N = V - V_ship,N

Metrics are computed through the same frozen 05C utility used for the earlier
Ridge/GRU/Physics-Compact pipeline.

The existing Stage-08C external results are read ONLY as frozen comparison
references. They are not used to update any model. The script produces an
integrated external comparison table containing:

    Persistence
    Frozen Ridge
    Frozen Compact Vessel Residual
    Frozen Physics-Compact Vessel Residual
    Frozen DLinear
    Frozen iTransformer
    Frozen TimeMixer
    Frozen PatchTST

DEFAULT PATHS
-------------
External root:
D:\\project\\WindPredict_SaildroneData\\data\\external\\
SD1033_external_forecasting_v0_1

09C checkpoints:
D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\
SD1090_TPOS2024_JointForecasting_v0_1\\
09C_modern_baseline_outer_validation_v0_1

Existing 08C reference inference:
<external-root>\\frozen_external_inference_v0_1

Output:
<external-root>\\09D_modern_baseline_external_inference_v0_1

DEPENDENCIES
------------
Place in the same src directory as:
    09A_modern_baseline_framework.py
    08C_frozen_external_inference.py
    05C_tune_joint_deep_baselines_pytorch_v2.py

OUTPUTS
-------
modern_external_run_metrics.csv
modern_external_seed_summary.csv
modern_external_per_horizon_metrics.csv
modern_external_per_horizon_summary.csv
integrated_external_seed_summary.csv
integrated_external_per_horizon_summary.csv
modern_vs_frozen_references.csv
modern_per_horizon_vs_physics_compact.csv
external_model_ranking.csv
external_inference_audit.csv
external_inference_manifest.json
external_inference_report.txt

Optional:
predictions/
    external_<dataset>_<model>_seed<seed>.npz
"""

from __future__ import annotations

import argparse
import gc
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


SCRIPT_VERSION = "0.1.0-modern-baseline-frozen-external"

SHARED_09A = "09A_modern_baseline_framework.py"
SHARED_08C = "08C_frozen_external_inference.py"

DEFAULT_EXTERNAL_ROOT = (
    r"D:\project\WindPredict_SaildroneData\data\external"
    r"\SD1033_external_forecasting_v0_1"
)

DEFAULT_FROZEN_DATASET_DIR = (
    r"D:\project\WindPredict_SaildroneData\data\forecasting"
    r"\SD1090_TPOS2024_JointForecasting_v0_1"
)

DEFAULT_09C_DIR = (
    r"D:\project\WindPredict_SaildroneData\data\forecasting"
    r"\SD1090_TPOS2024_JointForecasting_v0_1"
    r"\09C_modern_baseline_outer_validation_v0_1"
)

DEFAULT_DATASETS = [
    "SD1033_2024_full",
    "SD1033_2024_matched_SD1090",
    "SD1033_2023_full",
    "SD1033_2022_full",
]

PRIMARY_DATASETS = [
    "SD1033_2024_matched_SD1090",
    "SD1033_2023_full",
    "SD1033_2022_full",
]

MODEL_ORDER = [
    "DLinear",
    "iTransformer",
    "TimeMixer",
    "PatchTST",
]

MODEL_OUTPUT_NAMES = {
    "DLinear": "Frozen-DLinear",
    "iTransformer": "Frozen-iTransformer",
    "TimeMixer": "Frozen-TimeMixer",
    "PatchTST": "Frozen-PatchTST",
}

REFERENCE_PHYSICS_MODEL = (
    "Frozen-Physics-Compact-Vessel-Residual"
)

REFERENCE_RIDGE_MODEL = "Frozen-Ridge"

DEFAULT_SEEDS = [
    500043,
    501052,
    502061,
    503070,
    504079,
]

HORIZONS = [1, 2, 3, 5, 10]

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


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(
            f
        )


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


def parse_csv_list(
    text,
    allowed=None,
):
    values = []

    for token in text.split(","):
        token = token.strip()

        if not token:
            continue

        if (
            allowed is not None
            and token not in allowed
        ):
            raise ValueError(
                f"Unsupported value '{token}'. "
                f"Allowed: {allowed}"
            )

        if token not in values:
            values.append(
                token
            )

    if not values:
        raise ValueError(
            "Empty selection."
        )

    return values


def parse_seeds(text):
    return [
        int(v)
        for v in parse_csv_list(
            text
        )
    ]


def infer_model(
    torch,
    *,
    model,
    X,
    device,
    batch_size,
    use_amp,
):
    model.eval()

    out_chunks = []

    with torch.no_grad():
        for start in range(
            0,
            X.shape[0],
            int(batch_size),
        ):
            end = min(
                X.shape[0],
                start
                + int(batch_size),
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

            if not bool(
                torch.isfinite(
                    pred
                ).all().item()
            ):
                raise RuntimeError(
                    "Non-finite external prediction."
                )

            out_chunks.append(
                pred.detach()
                .float()
                .cpu()
                .numpy()
            )

            del (
                Xb,
                pred,
            )

    return np.concatenate(
        out_chunks,
        axis=0,
    )


def metric_summary_from_utility(
    utility,
    metrics,
):
    row = (
        utility.summarize_metrics(
            metrics
        )
        .iloc[
            0
        ]
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
            "parameters": int(
                grp[
                    "parameters"
                ].iloc[
                    0
                ]
            ),
        }

        for col in metric_cols:
            values = grp[
                col
            ].to_numpy(
                dtype=float
            )

            row[
                f"mean_{col}"
            ] = float(
                np.mean(
                    values
                )
            )

            row[
                f"std_{col}"
            ] = (
                float(
                    np.std(
                        values,
                        ddof=1,
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
    )


def aggregate_horizon_summary(
    horizon_df,
):
    metrics = [
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

        for metric in metrics:
            values = grp[
                metric
            ].to_numpy(
                dtype=float
            )

            row[
                f"mean_{metric}"
            ] = float(
                np.mean(
                    values
                )
            )

            row[
                f"std_{metric}"
            ] = (
                float(
                    np.std(
                        values,
                        ddof=1,
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
            "dataset_id",
            "model",
            "horizon_min",
        ]
    ).reset_index(
        drop=True
    )


def get_exact_row(
    df,
    *,
    dataset_id,
    model,
):
    sub = df.loc[
        (
            df[
                "dataset_id"
            ] == dataset_id
        )
        & (
            df[
                "model"
            ] == model
        )
    ]

    if len(
        sub
    ) != 1:
        raise RuntimeError(
            f"Expected one row for {dataset_id}/{model}, "
            f"found {len(sub)}."
        )

    return sub.iloc[
        0
    ]


def build_reference_effects(
    modern_summary,
    reference_summary,
):
    rows = []

    for _, modern in modern_summary.iterrows():
        dataset_id = str(
            modern[
                "dataset_id"
            ]
        )

        model = str(
            modern[
                "model"
            ]
        )

        ridge = get_exact_row(
            reference_summary,
            dataset_id=dataset_id,
            model=REFERENCE_RIDGE_MODEL,
        )

        physics = get_exact_row(
            reference_summary,
            dataset_id=dataset_id,
            model=REFERENCE_PHYSICS_MODEL,
        )

        modern_aw = float(
            modern[
                "mean_AW_RMSE_mps"
            ]
        )

        ridge_aw = float(
            ridge[
                "mean_AW_RMSE_mps"
            ]
        )

        physics_aw = float(
            physics[
                "mean_AW_RMSE_mps"
            ]
        )

        rows.append({
            "dataset_id": dataset_id,
            "modern_model": model,
            "modern_AW_RMSE_mps": modern_aw,
            "frozen_Ridge_AW_RMSE_mps": ridge_aw,
            "PhysicsCompact_AW_RMSE_mps": physics_aw,
            "modern_improvement_vs_Ridge_percent": (
                100.0
                * (
                    ridge_aw
                    - modern_aw
                )
                / ridge_aw
            ),
            "PhysicsCompact_improvement_vs_modern_percent": (
                100.0
                * (
                    modern_aw
                    - physics_aw
                )
                / modern_aw
            ),
            "PhysicsCompact_minus_modern_AW_mps": (
                physics_aw
                - modern_aw
            ),
            "PhysicsCompact_better_than_modern": bool(
                physics_aw
                < modern_aw
            ),
        })

    return pd.DataFrame(
        rows
    )


def build_per_horizon_vs_physics(
    modern_horizon,
    reference_horizon,
):
    rows = []

    for _, modern in modern_horizon.iterrows():
        dataset_id = str(
            modern[
                "dataset_id"
            ]
        )

        model = str(
            modern[
                "model"
            ]
        )

        horizon = int(
            modern[
                "horizon_min"
            ]
        )

        ref = reference_horizon.loc[
            (
                reference_horizon[
                    "dataset_id"
                ] == dataset_id
            )
            & (
                reference_horizon[
                    "model"
                ] == REFERENCE_PHYSICS_MODEL
            )
            & (
                reference_horizon[
                    "horizon_min"
                ] == horizon
            )
        ]

        if len(
            ref
        ) != 1:
            raise RuntimeError(
                f"Missing Physics-Compact per-horizon reference for "
                f"{dataset_id}/{horizon} min."
            )

        ref = ref.iloc[
            0
        ]

        modern_aw = float(
            modern[
                "mean_apparent_vector_RMSE_mps"
            ]
        )

        physics_aw = float(
            ref[
                "mean_apparent_vector_RMSE_mps"
            ]
        )

        rows.append({
            "dataset_id": dataset_id,
            "model": model,
            "horizon_min": horizon,
            "modern_AW_RMSE_mps": modern_aw,
            "PhysicsCompact_AW_RMSE_mps": physics_aw,
            "PhysicsCompact_improvement_vs_modern_percent": (
                100.0
                * (
                    modern_aw
                    - physics_aw
                )
                / modern_aw
            ),
            "PhysicsCompact_better": bool(
                physics_aw
                < modern_aw
            ),
        })

    return pd.DataFrame(
        rows
    )


def build_external_ranking(
    integrated_summary,
):
    rows = []

    for dataset_id, grp in integrated_summary.groupby(
        "dataset_id",
        sort=False,
    ):
        ranked = grp.sort_values(
            [
                "mean_AW_RMSE_mps",
                "parameters",
            ],
            na_position="last",
        ).copy()

        ranked[
            "AW_rank"
        ] = np.arange(
            1,
            len(
                ranked
            )
            + 1,
        )

        rows.append(
            ranked[
                [
                    "dataset_id",
                    "model",
                    "AW_rank",
                    "mean_AW_RMSE_mps",
                    "std_AW_RMSE_mps",
                    "mean_vessel_RMSE_mps",
                    "mean_AWS_RMSE_mps",
                    "mean_AWA_MAE_deg",
                    "parameters",
                ]
            ]
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--external-root",
        default=DEFAULT_EXTERNAL_ROOT,
    )

    parser.add_argument(
        "--frozen-dataset-dir",
        default=DEFAULT_FROZEN_DATASET_DIR,
    )

    parser.add_argument(
        "--refit-dir",
        default=DEFAULT_09C_DIR,
    )

    parser.add_argument(
        "--reference-08c-dir",
        default=None,
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
        default=1024,
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
        "--save-predictions",
        action="store_true",
    )

    args = parser.parse_args()

    external_root = Path(
        args.external_root
    )

    datasets_dir = (
        external_root
        / "datasets"
    )

    frozen_dataset_dir = Path(
        args.frozen_dataset_dir
    )

    refit_dir = Path(
        args.refit_dir
    )

    reference_08c_dir = (
        Path(
            args.reference_08c_dir
        )
        if args.reference_08c_dir
        else (
            external_root
            / "frozen_external_inference_v0_1"
        )
    )

    output_dir = (
        Path(
            args.output_dir
        )
        if args.output_dir
        else (
            external_root
            / "09D_modern_baseline_external_inference_v0_1"
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

    dataset_ids = parse_csv_list(
        args.datasets,
        allowed=DEFAULT_DATASETS,
    )

    models = parse_csv_list(
        args.models,
        allowed=MODEL_ORDER,
    )

    seeds = parse_seeds(
        args.seeds
    )

    if seeds != DEFAULT_SEEDS:
        log(
            "[WARNING] Non-default seed schedule requested. "
            "Paper comparison should use the frozen five-seed schedule."
        )

    shared09a, path09a = (
        load_module_from_src(
            SHARED_09A,
            "modern_09a_shared_09d",
        )
    )

    shared08c, path08c = (
        load_module_from_src(
            SHARED_08C,
            "external_08c_shared_09d",
        )
    )

    utility, utility_path = (
        shared09a.load_utility()
    )

    (
        torch,
        nn,
        F,
        _,
        _,
    ) = shared09a.import_torch()

    model_classes = (
        shared09a.build_model_classes(
            torch,
            nn,
            F,
        )
    )

    frozen = (
        shared09a.validate_frozen_dataset(
            utility,
            frozen_dataset_dir,
        )
    )

    refit_manifest_path = (
        refit_dir
        / "refit_manifest.json"
    )

    refit_manifest = load_json(
        refit_manifest_path
    )

    policy = refit_manifest.get(
        "scientific_policy",
        {}
    )

    required_false = [
        "SD1090_test_accessed",
        "SD1033_accessed",
        "proposed_Physics_Compact_modified",
    ]

    for key in required_false:
        if bool(
            policy.get(
                key,
                False,
            )
        ):
            raise RuntimeError(
                f"09C policy audit failed: {key}=True"
            )

    if not bool(
        policy.get(
            "all_five_seeds_retained",
            False,
        )
    ):
        raise RuntimeError(
            "09C manifest does not confirm all five seeds retained."
        )

    external_manifest_path = (
        external_root
        / "external_dataset_manifest.json"
    )

    external_manifest = load_json(
        external_manifest_path
    )

    manifest_scaler = Path(
        external_manifest[
            "frozen_scalers"
        ]
    )

    if (
        manifest_scaler.resolve()
        != frozen[
            "scalers_path"
        ].resolve()
    ):
        raise RuntimeError(
            "External datasets and Stage-09C model do not reference "
            "the same frozen SD1090 scaler."
        )

    expected_scaler_hash = (
        external_manifest.get(
            "frozen_scalers_sha256"
        )
    )

    actual_scaler_hash = (
        shared08c.sha256_file(
            frozen[
                "scalers_path"
            ]
        )
    )

    if (
        expected_scaler_hash is not None
        and expected_scaler_hash
        != actual_scaler_hash
    ):
        raise RuntimeError(
            "Frozen scaler SHA256 mismatch."
        )

    reference_seed_path = (
        reference_08c_dir
        / "external_seed_summary.csv"
    )

    reference_horizon_path = (
        reference_08c_dir
        / "external_per_horizon_summary.csv"
    )

    if not reference_seed_path.exists():
        raise FileNotFoundError(
            reference_seed_path
        )

    if not reference_horizon_path.exists():
        raise FileNotFoundError(
            reference_horizon_path
        )

    reference_summary = pd.read_csv(
        reference_seed_path
    )

    reference_horizon = pd.read_csv(
        reference_horizon_path
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

    log(
        "=" * 110
    )
    log(
        "09D - FROZEN EXTERNAL INFERENCE FOR MODERN BASELINES"
    )
    log(
        "=" * 110
    )
    log(
        f"script_version         : {SCRIPT_VERSION}"
    )
    log(
        f"external_root          : {external_root}"
    )
    log(
        f"09C refit dir          : {refit_dir}"
    )
    log(
        f"08C references         : {reference_08c_dir}"
    )
    log(
        f"datasets               : {dataset_ids}"
    )
    log(
        f"models                 : {models}"
    )
    log(
        f"seeds                  : {seeds}"
    )
    log(
        f"device                 : {device}"
    )
    log(
        f"AMP                    : {use_amp}"
    )
    log(
        "external scaler fit    : NEVER"
    )
    log(
        "external weight update : NEVER"
    )
    log(
        "external seed selection: NEVER"
    )
    log(
        "Physics-Compact change : NEVER"
    )
    log("")

    run_rows = []
    horizon_frames = []
    audit_rows = []

    for dataset_index, dataset_id in enumerate(
        dataset_ids,
        start=1,
    ):
        log(
            "#" * 110
        )
        log(
            f"[DATASET {dataset_index}/{len(dataset_ids)}] {dataset_id}"
        )
        log(
            "#" * 110
        )

        external_path = (
            datasets_dir
            / f"{dataset_id}.npz"
        )

        external = (
            shared08c.load_external_npz(
                external_path
            )
        )

        audit = (
            shared08c.audit_external_npz(
                external,
                dataset_id,
                frozen[
                    "scalers_path"
                ],
            )
        )

        audit_rows.append({
            "dataset_id": dataset_id,
            "npz_path": str(
                external_path
            ),
            "sample_count": int(
                external[
                    "X"
                ].shape[
                    0
                ]
            ),
            **audit,
        })

        if not bool(
            audit[
                "audit_passed"
            ]
        ):
            raise RuntimeError(
                f"{dataset_id}: external dataset audit failed."
            )

        X = external[
            "X"
        ].astype(
            np.float32,
            copy=False,
        )

        truth_joint = external[
            "y_raw"
        ].astype(
            np.float32,
            copy=False,
        )

        truth_apparent = external[
            "apparent_earth_raw"
        ].astype(
            np.float32,
            copy=False,
        )

        persistence_joint = (
            utility.persistence_joint_prediction(
                X=X,
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

        ridge_reference = get_exact_row(
            reference_summary,
            dataset_id=dataset_id,
            model=REFERENCE_RIDGE_MODEL,
        )

        ridge_aw = float(
            ridge_reference[
                "mean_AW_RMSE_mps"
            ]
        )

        ridge_horizon_rows = (
            reference_horizon.loc[
                (
                    reference_horizon[
                        "dataset_id"
                    ] == dataset_id
                )
                & (
                    reference_horizon[
                        "model"
                    ] == REFERENCE_RIDGE_MODEL
                )
            ]
            .sort_values(
                "horizon_min"
            )
        )

        if list(
            ridge_horizon_rows[
                "horizon_min"
            ].astype(
                int
            )
        ) != HORIZONS:
            raise RuntimeError(
                f"{dataset_id}: frozen Ridge horizon reference mismatch."
            )

        ridge_aw_by_horizon = {
            int(
                row[
                    "horizon_min"
                ]
            ): float(
                row[
                    "mean_apparent_vector_RMSE_mps"
                ]
            )
            for _, row
            in ridge_horizon_rows.iterrows()
        }

        for model_name in models:
            output_name = (
                MODEL_OUTPUT_NAMES[
                    model_name
                ]
            )

            log(
                f"  [MODEL] {output_name}"
            )

            for seed in seeds:
                checkpoint_path = (
                    refit_dir
                    / "models"
                    / f"{model_name}_seed{seed}.pt"
                )

                if not checkpoint_path.exists():
                    raise FileNotFoundError(
                        checkpoint_path
                    )

                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                )

                if str(
                    checkpoint[
                        "model_name"
                    ]
                ) != model_name:
                    raise RuntimeError(
                        f"Checkpoint model mismatch: {checkpoint_path}"
                    )

                if int(
                    checkpoint[
                        "seed"
                    ]
                ) != int(
                    seed
                ):
                    raise RuntimeError(
                        f"Checkpoint seed mismatch: {checkpoint_path}"
                    )

                if list(
                    checkpoint[
                        "feature_names"
                    ]
                ) != list(
                    frozen[
                        "feature_names"
                    ]
                ):
                    raise RuntimeError(
                        f"{checkpoint_path.name}: feature schema mismatch."
                    )

                if list(
                    checkpoint[
                        "target_names"
                    ]
                ) != list(
                    frozen[
                        "target_names"
                    ]
                ):
                    raise RuntimeError(
                        f"{checkpoint_path.name}: target schema mismatch."
                    )

                if list(
                    checkpoint[
                        "horizons_min"
                    ]
                ) != list(
                    frozen[
                        "horizons"
                    ]
                ):
                    raise RuntimeError(
                        f"{checkpoint_path.name}: horizon mismatch."
                    )

                config = dict(
                    checkpoint[
                        "config"
                    ]
                )

                model = (
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
                        model
                    )
                )

                if int(
                    params
                ) != int(
                    checkpoint[
                        "parameters"
                    ]
                ):
                    raise RuntimeError(
                        f"{checkpoint_path.name}: parameter count mismatch."
                    )

                model.load_state_dict(
                    checkpoint[
                        "state_dict"
                    ],
                    strict=True,
                )

                model = model.to(
                    device
                )

                t0 = time.perf_counter()

                pred_z = infer_model(
                    torch,
                    model=model,
                    X=X,
                    device=device,
                    batch_size=args.batch_size,
                    use_amp=use_amp,
                )

                elapsed = float(
                    time.perf_counter()
                    - t0
                )

                pred_raw = (
                    utility.inverse_joint_target(
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
                    utility.evaluate_model(
                        truth_joint=truth_joint,
                        pred_joint=pred_raw,
                        truth_apparent_ref=truth_apparent,
                        horizons=tuple(
                            frozen[
                                "horizons"
                            ]
                        ),
                        split_name=dataset_id,
                        model_name=output_name,
                        direction_min_speed=DIRECTION_MIN_SPEED,
                        cog_min_speed=COG_MIN_SPEED,
                        persistence_joint=persistence_joint,
                    )
                )

                # Overwrite AW skill so every modern model uses the exact
                # frozen 08C Ridge reference, not another denominator.
                metrics[
                    "AW_skill_vs_Ridge"
                ] = [
                    (
                        1.0
                        - float(
                            row[
                                "apparent_vector_RMSE_mps"
                            ]
                        )
                        / ridge_aw_by_horizon[
                            int(
                                row[
                                    "horizon_min"
                                ]
                            )
                        ]
                    )
                    for _, row
                    in metrics.iterrows()
                ]

                metrics[
                    "dataset_id"
                ] = dataset_id

                metrics[
                    "model"
                ] = output_name

                metrics[
                    "seed"
                ] = int(
                    seed
                )

                metrics[
                    "parameters"
                ] = int(
                    params
                )

                horizon_frames.append(
                    metrics
                )

                summary = metric_summary_from_utility(
                    utility,
                    metrics,
                )

                summary[
                    "AW_skill_vs_Ridge"
                ] = (
                    1.0
                    - summary[
                        "AW_RMSE_mps"
                    ]
                    / ridge_aw
                )

                run_rows.append({
                    "dataset_id": dataset_id,
                    "model": output_name,
                    "base_model_name": model_name,
                    "seed": int(
                        seed
                    ),
                    "parameters": int(
                        params
                    ),
                    "config_id": str(
                        config[
                            "config_id"
                        ]
                    ),
                    "checkpoint_path": str(
                        checkpoint_path
                    ),
                    "inference_seconds": elapsed,
                    **summary,
                })

                if args.save_predictions:
                    np.savez_compressed(
                        prediction_dir
                        / (
                            f"external_{dataset_id}_"
                            f"{model_name}_seed{seed}.npz"
                        ),
                        pred_z=pred_z.astype(
                            np.float32
                        ),
                        pred_raw=pred_raw.astype(
                            np.float32
                        ),
                        target_time_ns=external[
                            "target_time_ns"
                        ],
                        context_end_time_ns=external[
                            "context_end_time_ns"
                        ],
                        dataset_id=np.asarray(
                            dataset_id
                        ),
                        model=np.asarray(
                            output_name
                        ),
                        seed=np.asarray(
                            seed,
                            dtype=np.int64,
                        ),
                    )

                log(
                    f"    seed={seed} | "
                    f"AW={summary['AW_RMSE_mps']:.4f} | "
                    f"wind={summary['wind_RMSE_mps']:.4f} | "
                    f"vessel={summary['vessel_RMSE_mps']:.4f} | "
                    f"AWS={summary['AWS_RMSE_mps']:.4f} | "
                    f"AWA={summary['AWA_MAE_deg']:.2f} deg"
                )

                del (
                    model,
                    pred_z,
                    pred_raw,
                )

                gc.collect()

                if device.type == "cuda":
                    torch.cuda.empty_cache()

    run_df = pd.DataFrame(
        run_rows
    )

    horizon_df = pd.concat(
        horizon_frames,
        ignore_index=True,
    )

    audit_df = pd.DataFrame(
        audit_rows
    )

    modern_summary = (
        aggregate_seed_summary(
            run_df
        )
    )

    modern_horizon_summary = (
        aggregate_horizon_summary(
            horizon_df
        )
    )

    # Keep only requested external datasets in the frozen 08C references.
    reference_summary_subset = (
        reference_summary.loc[
            reference_summary[
                "dataset_id"
            ].isin(
                dataset_ids
            )
        ].copy()
    )

    reference_horizon_subset = (
        reference_horizon.loc[
            reference_horizon[
                "dataset_id"
            ].isin(
                dataset_ids
            )
        ].copy()
    )

    integrated_summary = pd.concat(
        [
            reference_summary_subset,
            modern_summary,
        ],
        ignore_index=True,
        sort=False,
    )

    integrated_horizon = pd.concat(
        [
            reference_horizon_subset,
            modern_horizon_summary,
        ],
        ignore_index=True,
        sort=False,
    )

    reference_effects = (
        build_reference_effects(
            modern_summary,
            reference_summary_subset,
        )
    )

    per_horizon_vs_physics = (
        build_per_horizon_vs_physics(
            modern_horizon_summary,
            reference_horizon_subset,
        )
    )

    ranking = build_external_ranking(
        integrated_summary
    )

    run_df.to_csv(
        output_dir
        / "modern_external_run_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    modern_summary.to_csv(
        output_dir
        / "modern_external_seed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_df.to_csv(
        output_dir
        / "modern_external_per_horizon_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    modern_horizon_summary.to_csv(
        output_dir
        / "modern_external_per_horizon_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    integrated_summary.to_csv(
        output_dir
        / "integrated_external_seed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    integrated_horizon.to_csv(
        output_dir
        / "integrated_external_per_horizon_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    reference_effects.to_csv(
        output_dir
        / "modern_vs_frozen_references.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_horizon_vs_physics.to_csv(
        output_dir
        / "modern_per_horizon_vs_physics_compact.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ranking.to_csv(
        output_dir
        / "external_model_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit_df.to_csv(
        output_dir
        / "external_inference_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    primary_effects = reference_effects.loc[
        reference_effects[
            "dataset_id"
        ].isin(
            PRIMARY_DATASETS
        )
    ].copy()

    primary_physics_win_count = int(
        primary_effects[
            "PhysicsCompact_better_than_modern"
        ].sum()
    )

    primary_pair_count = int(
        len(
            primary_effects
        )
    )

    primary_horizon = (
        per_horizon_vs_physics.loc[
            per_horizon_vs_physics[
                "dataset_id"
            ].isin(
                PRIMARY_DATASETS
            )
        ]
    )

    primary_horizon_physics_win_count = int(
        primary_horizon[
            "PhysicsCompact_better"
        ].sum()
    )

    primary_horizon_pair_count = int(
        len(
            primary_horizon
        )
    )

    manifest = {
        "script_version": SCRIPT_VERSION,
        "stage": "09D",
        "purpose": (
            "Frozen external inference for modern comparison baselines; "
            "no external fitting or selection."
        ),
        "external_root": str(
            external_root
        ),
        "external_manifest": str(
            external_manifest_path
        ),
        "frozen_dataset_dir": str(
            frozen_dataset_dir
        ),
        "09C_refit_dir": str(
            refit_dir
        ),
        "09C_refit_manifest": str(
            refit_manifest_path
        ),
        "08C_reference_dir": str(
            reference_08c_dir
        ),
        "09A_script": str(
            path09a
        ),
        "08C_script": str(
            path08c
        ),
        "05C_utility": str(
            utility_path
        ),
        "datasets": dataset_ids,
        "primary_datasets": PRIMARY_DATASETS,
        "modern_models": models,
        "seed_schedule": seeds,
        "frozen_scalers": str(
            frozen[
                "scalers_path"
            ]
        ),
        "frozen_scalers_sha256": actual_scaler_hash,
        "protocol": {
            "external_scaler_fit": False,
            "external_training": False,
            "external_fine_tuning": False,
            "external_early_stopping": False,
            "external_seed_selection": False,
            "external_hyperparameter_selection": False,
            "all_frozen_seeds_evaluated": True,
            "physics_compact_modified": False,
            "09B_configuration_changed": False,
            "08C_reference_results_used_for_training": False,
            "08C_reference_results_used_only_for_posthoc_comparison": True,
        },
        "descriptive_summary": {
            "PhysicsCompact_better_than_modern_on_primary_dataset_model_pairs": (
                primary_physics_win_count
            ),
            "primary_dataset_model_pair_count": (
                primary_pair_count
            ),
            "PhysicsCompact_better_than_modern_on_primary_dataset_model_horizon_pairs": (
                primary_horizon_physics_win_count
            ),
            "primary_dataset_model_horizon_pair_count": (
                primary_horizon_pair_count
            ),
        },
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
            "amp": bool(
                use_amp
            ),
        },
    }

    save_json(
        output_dir
        / "external_inference_manifest.json",
        manifest,
    )

    with (
        output_dir
        / "external_inference_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "09D Frozen External Inference for Modern Baselines\n"
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
            "No SD1033 training, fine-tuning, scaler fit, early stopping, "
            "seed selection, or hyperparameter selection.\n"
        )
        f.write(
            "All five 09C checkpoints are evaluated for every modern model.\n"
        )
        f.write(
            "The proposed Physics-Compact model remains unchanged.\n\n"
        )
        f.write(
            "Modern external summary\n"
        )
        f.write(
            "-" * 110
            + "\n"
        )
        f.write(
            modern_summary.to_string(
                index=False
            )
        )
        f.write(
            "\n\nModern vs frozen references\n"
        )
        f.write(
            "-" * 110
            + "\n"
        )
        f.write(
            reference_effects.to_string(
                index=False
            )
        )
        f.write(
            "\n\nPrimary consistency\n"
        )
        f.write(
            "-" * 110
            + "\n"
        )
        f.write(
            "Physics-Compact better than modern baselines on "
            f"{primary_physics_win_count}/{primary_pair_count} "
            "primary dataset-model pairs.\n"
        )
        f.write(
            "Physics-Compact better than modern baselines on "
            f"{primary_horizon_physics_win_count}/"
            f"{primary_horizon_pair_count} "
            "primary dataset-model-horizon pairs.\n"
        )

    log("")
    log(
        "=" * 110
    )
    log(
        "09D FROZEN EXTERNAL MODERN-BASELINE RESULTS"
    )
    log(
        "=" * 110
    )

    for dataset_id in dataset_ids:
        log(
            f"{dataset_id}:"
        )

        d = modern_summary.loc[
            modern_summary[
                "dataset_id"
            ] == dataset_id
        ].sort_values(
            "mean_AW_RMSE_mps"
        )

        for _, row in d.iterrows():
            effect = reference_effects.loc[
                (
                    reference_effects[
                        "dataset_id"
                    ] == dataset_id
                )
                & (
                    reference_effects[
                        "modern_model"
                    ] == row[
                        "model"
                    ]
                )
            ].iloc[
                0
            ]

            log(
                f"  {row['model']}: "
                f"AW={row['mean_AW_RMSE_mps']:.4f}"
                f"+/-{row['std_AW_RMSE_mps']:.4f} | "
                f"vessel={row['mean_vessel_RMSE_mps']:.4f} | "
                f"AWS={row['mean_AWS_RMSE_mps']:.4f} | "
                f"PhysicsCompact advantage="
                f"{effect['PhysicsCompact_improvement_vs_modern_percent']:.3f}%"
            )

    log("")
    log(
        "[PRIMARY CONSISTENCY] Physics-Compact better on "
        f"{primary_physics_win_count}/{primary_pair_count} "
        "dataset-model pairs."
    )

    log(
        "[PRIMARY HORIZON CONSISTENCY] Physics-Compact better on "
        f"{primary_horizon_physics_win_count}/"
        f"{primary_horizon_pair_count} dataset-model-horizon pairs."
    )

    log(
        "[POLICY] No external data were used to tune/select any model or seed."
    )

    log(
        "[POLICY] Physics-Compact proposed model remains frozen."
    )

    log(
        f"[DONE] 09D outputs: {output_dir}"
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
