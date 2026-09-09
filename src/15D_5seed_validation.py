# -*- coding: utf-8 -*-
"""
15D_confirm_5seed_NewStagedPhysicsCompact_Validation.py

Five-seed confirmation of the FINAL staged Physics-Compact model
on the SD1090 TRAIN/VALIDATION development protocol.

IMPORTANT
---------
This script intentionally DOES NOT load the internal TEST split.

Purpose:
    Compare the new frozen staged model against the old manuscript's
    SD1090 outer-validation five-seed results using the same development
    validation split.

Frozen new model
----------------
Stage 1A: TrueWind linear anchor
    inputs: U,V,T,RH over 60 x 1-min history
    alpha_Ridge = 0

Stage 1B: Vessel+HDG linear anchor
    inputs: all 11 features over 60 x 1-min history
    alpha_Ridge = 2250

Stage 2: TrueWind residual
    2-layer GRU, hidden=48
    validation-derived horizon/component alpha_U, alpha_V
    alpha cap = 1.0
    TRAIN downstream predictions are 5-fold purged OOF

Stage 3: Vessel/HDG residual + apparent-wind physics
    2-layer GRU, hidden=48
    shared MLP=64
    validation-derived alpha_E, alpha_N, alpha_HDG
    alpha cap = 1.0
    apparent wind = Stage2 true wind - Stage3 vessel velocity

Five frozen seeds
-----------------
500043, 501052, 502061, 503070, 504079

Outputs
-------
15D_per_seed_validation_summary.csv
15D_validation_aggregate.csv
15D_truewind_per_horizon_all_seeds.csv
15D_stage15c_per_horizon_all_seeds.csv
15D_alpha_truewind_all_seeds.csv
15D_alpha_vessel_hdg_all_seeds.csv
15D_old_vs_new_manuscript_table6.csv
15D_REPORT.json
15D_REPORT.txt

Old manuscript Table 6 reference
--------------------------------
Apparent-wind vector RMSE = 1.0785 +/- 0.0032 m/s
Vessel-vector RMSE        = 0.4853 +/- 0.0078 m/s
AWS RMSE                  = 0.8032 m/s
AWA MAE                   = 8.64 deg
reported trainable params = 22,164

The old manuscript numbers are included ONLY as a numerical reference for
the same SD1090 validation split. The new architecture has already been
selected after inspecting development diagnostics, including the previous
internal-test comparisons, so this is a matched development comparison,
not a new untouched generalization claim.

Recommended PowerShell
----------------------
python "D:\\project\\WindPredict_SaildroneData\\src\\15D_confirm_5seed_NewStagedPhysicsCompact_Validation.py" `
  --dataset-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\SD1090_TPOS2024_JointForecasting_v0_1" `
  --stage15a-truewind-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15A_R_1min_Dual_Ridge_Anchors_v0_2" `
  --stage15a-vessel-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15A_F_VesselHDG_alpha2250_v0_3" `
  --src-dir "D:\\project\\WindPredict_SaildroneData\\src" `
  --output-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15D_5seed_NewStagedPhysicsCompact_Validation_v0_1"
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\project\WindPredict_SaildroneData")

DEFAULT_DATASET_DIR = (
    ROOT / "data" / "forecasting" / "SD1090_TPOS2024_JointForecasting_v0_1"
)

DEFAULT_STAGE15A_TRUEWIND_DIR = (
    ROOT / "data" / "forecasting" / "15A_R_1min_Dual_Ridge_Anchors_v0_2"
)

DEFAULT_STAGE15A_VESSEL_DIR = (
    ROOT / "data" / "forecasting" / "15A_F_VesselHDG_alpha2250_v0_3"
)

DEFAULT_SRC_DIR = ROOT / "src"

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "data"
    / "forecasting"
    / "15D_5seed_NewStagedPhysicsCompact_Validation_v0_1"
)

SEEDS = [
    500043,
    501052,
    502061,
    503070,
    504079,
]

HORIZONS = [1, 2, 3, 5, 10]

OLD_TABLE6 = {
    "AppVector_RMSE_mps": 1.0785,
    "AppVector_RMSE_std_mps": 0.0032,
    "Vessel_vector_RMSE_mps": 0.4853,
    "Vessel_vector_RMSE_std_mps": 0.0078,
    "AWS_RMSE_mps": 0.8032,
    "AWA_MAE_deg": 8.64,
    "reported_trainable_parameters": 22164,
}

TRUEWIND_RIDGE_COEFFICIENTS = 2410
VESSEL_HDG_RIDGE_COEFFICIENTS = 13220
STAGE15B_NN_PARAMETERS = 26074
STAGE15C_NN_PARAMETERS = 29252

NEW_NEURAL_TRAINABLE_PARAMETERS = (
    STAGE15B_NN_PARAMETERS
    + STAGE15C_NN_PARAMETERS
)

NEW_TOTAL_FITTED_COEFFICIENTS = (
    TRUEWIND_RIDGE_COEFFICIENTS
    + VESSEL_HDG_RIDGE_COEFFICIENTS
    + NEW_NEURAL_TRAINABLE_PARAMETERS
)


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
            return {str(k): cv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [cv(x) for x in v]
        return v

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            cv(obj),
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
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


def load_truewind_anchors(stage15a_dir: Path):
    train_path = (
        stage15a_dir
        / "15A_train_oof_anchors.npz"
    )

    val_path = (
        stage15a_dir
        / "15A_validation_anchor_predictions.npz"
    )

    fold_candidates = [
        stage15a_dir
        / "15A_R_oof_folds.csv",
        stage15a_dir
        / "15A_oof_folds.csv",
    ]

    if not train_path.exists():
        raise FileNotFoundError(
            train_path
        )

    if not val_path.exists():
        raise FileNotFoundError(
            val_path
        )

    fold_path = None

    for p in fold_candidates:
        if p.exists():
            fold_path = p
            break

    if fold_path is None:
        raise FileNotFoundError(
            "No Stage15A fold file found."
        )

    with np.load(
        train_path,
        allow_pickle=False,
    ) as z:
        train = {
            "y": np.asarray(
                z[
                    "true_wind_y"
                ],
                dtype=np.float32,
            ),
            "ridge": np.asarray(
                z[
                    "true_wind_ridge_oof"
                ],
                dtype=np.float32,
            ),
            "residual": np.asarray(
                z[
                    "true_wind_residual_oof"
                ],
                dtype=np.float32,
            ),
        }

    with np.load(
        val_path,
        allow_pickle=False,
    ) as z:
        val = {
            "y": np.asarray(
                z[
                    "true_wind_y"
                ],
                dtype=np.float32,
            ),
            "ridge": np.asarray(
                z[
                    "true_wind_ridge"
                ],
                dtype=np.float32,
            ),
            "residual": np.asarray(
                z[
                    "true_wind_residual"
                ],
                dtype=np.float32,
            ),
        }

    fold_df = pd.read_csv(
        fold_path
    )

    # Fix the known Stage15A-R mixed-anchor fold file:
    # keep TrueWind rows only when the anchor column exists.
    if "anchor" in fold_df.columns:
        mask = (
            fold_df[
                "anchor"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                "truewind"
            )
        )

        if mask.any():
            fold_df = (
                fold_df.loc[
                    mask
                ]
                .copy()
            )

    dedup_cols = [
        c
        for c in [
            "validation_start_index",
            "validation_end_index_exclusive",
            "purge_start_index",
            "purge_end_index_exclusive",
        ]
        if c in fold_df.columns
    ]

    if dedup_cols:
        fold_df = (
            fold_df
            .drop_duplicates(
                subset=dedup_cols,
                keep="first",
            )
            .reset_index(
                drop=True
            )
        )

    if len(
        fold_df
    ) != 5:
        raise RuntimeError(
            f"Expected 5 unique TrueWind OOF folds, got {len(fold_df)}."
        )

    return (
        train,
        val,
        fold_df,
    )


def load_vessel_anchors(stage15a_dir: Path):
    train_path = (
        stage15a_dir
        / "15A_F_train_vessel_hdg_oof.npz"
    )

    val_path = (
        stage15a_dir
        / "15A_F_validation_vessel_hdg.npz"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            train_path
        )

    if not val_path.exists():
        raise FileNotFoundError(
            val_path
        )

    with np.load(
        train_path,
        allow_pickle=False,
    ) as z:
        train = {
            "y": np.asarray(
                z[
                    "vessel_hdg_y"
                ],
                dtype=np.float32,
            ),
            "ridge": np.asarray(
                z[
                    "vessel_hdg_ridge_oof"
                ],
                dtype=np.float32,
            ),
            "residual": np.asarray(
                z[
                    "vessel_hdg_residual_oof"
                ],
                dtype=np.float32,
            ),
        }

    with np.load(
        val_path,
        allow_pickle=False,
    ) as z:
        val = {
            "y": np.asarray(
                z[
                    "vessel_hdg_y"
                ],
                dtype=np.float32,
            ),
            "ridge": np.asarray(
                z[
                    "vessel_hdg_ridge"
                ],
                dtype=np.float32,
            ),
            "residual": np.asarray(
                z[
                    "vessel_hdg_residual"
                ],
                dtype=np.float32,
            ),
        }

    return (
        train,
        val,
    )


def crossfit_stage15b(
    *,
    bmod,
    torch,
    nn,
    feat_train,
    anchor_train,
    fold_df,
    best_epoch,
    alpha,
    seed,
):
    folds = bmod.make_fold_indices(
        fold_df,
        len(
            feat_train[
                "seq"
            ]
        ),
    )

    if len(
        folds
    ) != 5:
        raise RuntimeError(
            f"Expected 5 Stage15B folds, got {len(folds)}"
        )

    raw_oof = np.full_like(
        anchor_train[
            "ridge"
        ],
        np.nan,
        dtype=np.float32,
    )

    stage2_oof = np.full_like(
        anchor_train[
            "ridge"
        ],
        np.nan,
        dtype=np.float32,
    )

    for fold in folds:
        k = int(
            fold[
                "fold"
            ]
        )

        tr_idx = fold[
            "train_idx"
        ]

        va_idx = fold[
            "val_idx"
        ]

        log(
            f"      Stage15B OOF fold {k}: "
            f"train={len(tr_idx):,}, holdout={len(va_idx):,}"
        )

        tr_feat = {
            "seq": feat_train[
                "seq"
            ][
                tr_idx
            ],
            "summary": feat_train[
                "summary"
            ][
                tr_idx
            ],
        }

        va_feat = {
            "seq": feat_train[
                "seq"
            ][
                va_idx
            ],
            "summary": feat_train[
                "summary"
            ][
                va_idx
            ],
        }

        fold_bundle = bmod.train_model(
            torch=torch,
            nn=nn,
            X_train_features=tr_feat,
            residual_train=anchor_train[
                "residual"
            ][
                tr_idx
            ],
            X_val_features=None,
            y_val=None,
            ridge_val=None,
            seed=seed
            + 1000
            + k,
            max_epochs=best_epoch,
            fixed_epochs=best_epoch,
            verbose=False,
        )

        fold_raw = bmod.predict_raw(
            torch,
            fold_bundle[
                "model"
            ],
            bmod.transform_features(
                va_feat,
                fold_bundle[
                    "scaler"
                ],
            ),
            fold_bundle[
                "scaler"
            ],
            fold_bundle[
                "device"
            ],
        )

        raw_oof[
            va_idx
        ] = fold_raw

        stage2_oof[
            va_idx
        ] = bmod.apply_alpha(
            anchor_train[
                "ridge"
            ][
                va_idx
            ],
            fold_raw,
            alpha,
        )

        del fold_bundle
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not (
        np.isfinite(
            raw_oof
        ).all()
        and np.isfinite(
            stage2_oof
        ).all()
    ):
        raise RuntimeError(
            "Stage15B OOF output incomplete/nonfinite."
        )

    return (
        raw_oof,
        stage2_oof,
    )


def summarize_truewind_metrics(df):
    """
    Mean over the five horizons.
    """
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


def summarize_stage15c_metrics(df):
    return {
        "Vessel_East_RMSE_mps": float(
            df[
                "Vessel_East_RMSE_mps"
            ].mean()
        ),
        "Vessel_North_RMSE_mps": float(
            df[
                "Vessel_North_RMSE_mps"
            ].mean()
        ),
        "Vessel_vector_RMSE_mps": float(
            df[
                "Vessel_vector_RMSE_mps"
            ].mean()
        ),
        "SOG_RMSE_mps": float(
            df[
                "SOG_RMSE_mps"
            ].mean()
        ),
        "HDG_MAE_deg": float(
            df[
                "HDG_MAE_deg"
            ].mean()
        ),
        "HDG_RMSE_deg": float(
            df[
                "HDG_RMSE_deg"
            ].mean()
        ),
        "AppVector_RMSE_mps": float(
            df[
                "AppVector_RMSE_mps"
            ].mean()
        ),
        "AWS_RMSE_mps": float(
            df[
                "AWS_RMSE_mps"
            ].mean()
        ),
        "AWA_MAE_deg": float(
            df[
                "AWA_MAE_deg"
            ].mean()
        ),
        "AWA_RMSE_deg": float(
            df[
                "AWA_RMSE_deg"
            ].mean()
        ),
    }


def aggregate_seed_summary(per_seed_df):
    numeric_cols = [
        c
        for c in per_seed_df.columns
        if c
        not in {
            "seed",
        }
        and np.issubdtype(
            per_seed_df[
                c
            ].dtype,
            np.number,
        )
    ]

    rows = []

    for col in numeric_cols:
        vals = per_seed_df[
            col
        ].to_numpy(
            dtype=np.float64
        )

        rows.append(
            {
                "metric": col,
                "mean": float(
                    np.mean(
                        vals
                    )
                ),
                "std": float(
                    np.std(
                        vals,
                        ddof=1,
                    )
                ),
                "min": float(
                    np.min(
                        vals
                    )
                ),
                "max": float(
                    np.max(
                        vals
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def old_vs_new_table(
    per_seed_df,
):
    specs = [
        (
            "AppVector_RMSE_mps",
            OLD_TABLE6[
                "AppVector_RMSE_mps"
            ],
            OLD_TABLE6[
                "AppVector_RMSE_std_mps"
            ],
        ),
        (
            "Vessel_vector_RMSE_mps",
            OLD_TABLE6[
                "Vessel_vector_RMSE_mps"
            ],
            OLD_TABLE6[
                "Vessel_vector_RMSE_std_mps"
            ],
        ),
        (
            "AWS_RMSE_mps",
            OLD_TABLE6[
                "AWS_RMSE_mps"
            ],
            np.nan,
        ),
        (
            "AWA_MAE_deg",
            OLD_TABLE6[
                "AWA_MAE_deg"
            ],
            np.nan,
        ),
    ]

    rows = []

    for metric, old_mean, old_std in specs:
        vals = per_seed_df[
            metric
        ].to_numpy(
            dtype=np.float64
        )

        new_mean = float(
            np.mean(
                vals
            )
        )

        new_std = float(
            np.std(
                vals,
                ddof=1,
            )
        )

        # For error metrics, positive improvement means NEW is lower.
        improvement = (
            old_mean
            - new_mean
        ) / old_mean * 100.0

        rows.append(
            {
                "metric": metric,
                "old_manuscript_mean": old_mean,
                "old_manuscript_std": old_std,
                "new_5seed_mean": new_mean,
                "new_5seed_std": new_std,
                "new_improvement_vs_old_pct": float(
                    improvement
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


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
        "--stage15a-vessel-dir",
        type=Path,
        default=DEFAULT_STAGE15A_VESSEL_DIR,
    )

    parser.add_argument(
        "--src-dir",
        type=Path,
        default=DEFAULT_SRC_DIR,
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

    b_script = (
        args.src_dir
        / "15B_train_1min_TrueWind_Residual_Correction_v2.py"
    )

    c_script = (
        args.src_dir
        / "15C_train_1min_VesselHDG_Residual_ApparentWind.py"
    )

    bmod = load_module(
        "stage15b_for_15d",
        b_script,
    )

    cmod = load_module(
        "stage15c_for_15d",
        c_script,
    )

    # IMPORTANT: final Stage15B method uses non-binding cap=1.0.
    bmod.ALPHA_MAX = 1.0

    # Ensure frozen Stage15C architecture.
    if int(
        cmod.GRU_LAYERS
    ) != 2:
        raise RuntimeError(
            f"Expected Stage15C GRU_LAYERS=2, got {cmod.GRU_LAYERS}"
        )

    if int(
        cmod.GRU_HIDDEN
    ) != 48:
        raise RuntimeError(
            f"Expected Stage15C GRU_HIDDEN=48, got {cmod.GRU_HIDDEN}"
        )

    torch, nn = bmod.import_torch()

    bmod.configure_torch(
        torch
    )

    cmod.configure_torch(
        torch
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    log("=" * 152)
    log(
        "15D — FIVE-SEED CONFIRMATION OF FINAL STAGED PHYSICS-COMPACT"
    )
    log("=" * 152)
    log(
        "TRAIN/VALIDATION ONLY — INTERNAL TEST IS NOT LOADED."
    )
    log(
        f"device = {device}"
    )
    log(
        f"seeds = {SEEDS}"
    )
    log(
        "Stage15B alpha cap = 1.0"
    )
    log(
        "Stage15C = 2-layer GRU, hidden=48"
    )

    # -----------------------------------------------------------------
    # Data: TRAIN / VALIDATION only.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 1/7] Loading TRAIN/VALIDATION data and frozen anchors."
    )

    ds_train = cmod.load_dataset_split(
        args.dataset_dir,
        "train",
    )

    ds_val = cmod.load_dataset_split(
        args.dataset_dir,
        "validation",
    )

    tw_train, tw_val, fold_df = (
        load_truewind_anchors(
            args.stage15a_truewind_dir
        )
    )

    vh_train, vh_val = (
        load_vessel_anchors(
            args.stage15a_vessel_dir
        )
    )

    n_train = len(
        ds_train[
            "X"
        ]
    )

    n_val = len(
        ds_val[
            "X"
        ]
    )

    if not (
        len(
            tw_train[
                "y"
            ]
        )
        == n_train
        == len(
            vh_train[
                "y"
            ]
        )
    ):
        raise RuntimeError(
            "TRAIN alignment mismatch."
        )

    if not (
        len(
            tw_val[
                "y"
            ]
        )
        == n_val
        == len(
            vh_val[
                "y"
            ]
        )
    ):
        raise RuntimeError(
            "VALIDATION alignment mismatch."
        )

    log(
        f"  TRAIN samples      = {n_train:,}"
    )
    log(
        f"  VALIDATION samples = {n_val:,}"
    )
    log(
        f"  TrueWind OOF folds = {len(fold_df)} unique folds"
    )

    # Build once: same deterministic inputs for all seeds.
    feat_train = bmod.build_features(
        ds_train[
            "X"
        ]
    )

    feat_val = bmod.build_features(
        ds_val[
            "X"
        ]
    )

    seq_train = cmod.build_sequence_features(
        ds_train[
            "X"
        ]
    )

    seq_val = cmod.build_sequence_features(
        ds_val[
            "X"
        ]
    )

    apparent_audit = cmod.audit_apparent_reference(
        ds_train
    )

    app_ref_train = (
        cmod.canonical_apparent_reference(
            ds_train,
            apparent_audit,
        )
    )

    app_ref_val = (
        cmod.canonical_apparent_reference(
            ds_val,
            apparent_audit,
        )
    )

    log("")
    log(
        "[APPARENT REFERENCE]"
    )
    log(
        f"  source     = {apparent_audit['source']}"
    )
    log(
        f"  key        = {apparent_audit['key']}"
    )
    log(
        f"  convention = {apparent_audit['convention']}"
    )
    log(
        f"  TRAIN audit vector RMSE = "
        f"{apparent_audit['train_audit_rmse']:.9f}"
    )

    per_seed_rows = []
    tw_horizon_frames = []
    c_horizon_frames = []
    tw_alpha_rows = []
    vh_alpha_rows = []

    # -----------------------------------------------------------------
    # Five seeds.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 2/7] Running five frozen seeds."
    )

    for seed_idx, seed in enumerate(
        SEEDS,
        start=1,
    ):
        log("")
        log(
            "-" * 152
        )
        log(
            f"[SEED {seed_idx}/5] {seed}"
        )
        log(
            "-" * 152
        )

        # =============================================================
        # Stage15B.
        # =============================================================
        log(
            "  [15B] Training final 2-layer true-wind residual model."
        )

        b_bundle = bmod.train_model(
            torch=torch,
            nn=nn,
            X_train_features=feat_train,
            residual_train=tw_train[
                "residual"
            ],
            X_val_features=feat_val,
            y_val=tw_val[
                "y"
            ],
            ridge_val=tw_val[
                "ridge"
            ],
            seed=seed,
            max_epochs=bmod.MAX_EPOCHS,
            fixed_epochs=None,
            verbose=True,
        )

        b_best_epoch = int(
            b_bundle[
                "best_epoch"
            ]
        )

        b_alpha = np.asarray(
            b_bundle[
                "best_alpha"
            ],
            dtype=np.float32,
        )

        raw_tw_val = bmod.predict_raw(
            torch,
            b_bundle[
                "model"
            ],
            bmod.transform_features(
                feat_val,
                b_bundle[
                    "scaler"
                ],
            ),
            b_bundle[
                "scaler"
            ],
            b_bundle[
                "device"
            ],
        )

        stage2_val = bmod.apply_alpha(
            tw_val[
                "ridge"
            ],
            raw_tw_val,
            b_alpha,
        )

        tw_metrics = (
            bmod.metrics_per_horizon(
                tw_val[
                    "y"
                ],
                stage2_val,
                "Stage15B-Final",
                "validation",
            )
        )

        tw_metrics.insert(
            0,
            "seed",
            seed,
        )

        tw_horizon_frames.append(
            tw_metrics
        )

        for j, h in enumerate(
            HORIZONS
        ):
            tw_alpha_rows.append(
                {
                    "seed": seed,
                    "horizon_min": h,
                    "alpha_U": float(
                        b_alpha[
                            j,
                            0
                        ]
                    ),
                    "alpha_V": float(
                        b_alpha[
                            j,
                            1
                        ]
                    ),
                }
            )

        log(
            f"  [15B] best epoch = {b_best_epoch}"
        )

        for j, h in enumerate(
            HORIZONS
        ):
            log(
                f"         {h:2d} min | "
                f"alpha_U={b_alpha[j,0]:.6f} | "
                f"alpha_V={b_alpha[j,1]:.6f}"
            )

        # Cross-fitted Stage15B outputs for leakage-safe Stage15C training.
        log(
            "  [15B] Rebuilding leakage-safe Stage2 TRAIN OOF."
        )

        _, stage2_train_oof = crossfit_stage15b(
            bmod=bmod,
            torch=torch,
            nn=nn,
            feat_train=feat_train,
            anchor_train=tw_train,
            fold_df=fold_df,
            best_epoch=b_best_epoch,
            alpha=b_alpha,
            seed=seed,
        )

        # Release Stage15B main model before Stage15C.
        del b_bundle
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # =============================================================
        # Stage15C.
        # =============================================================
        log(
            "  [15C] Training frozen 2-layer Vessel/HDG residual model."
        )

        cmod.SEED = int(
            seed
        )

        context_train = cmod.build_future_context(
            stage2_train_oof,
            vh_train[
                "ridge"
            ],
        )

        context_val = cmod.build_future_context(
            stage2_val,
            vh_val[
                "ridge"
            ],
        )

        c_bundle = cmod.train_stage15c(
            torch=torch,
            nn=nn,
            train_seq=seq_train,
            train_context=context_train,
            train_residual=vh_train[
                "residual"
            ],
            train_ridge=vh_train[
                "ridge"
            ],
            train_stage2=stage2_train_oof,
            train_y=vh_train[
                "y"
            ],
            train_app_ref=app_ref_train,
            val_seq=seq_val,
            val_context=context_val,
            val_ridge=vh_val[
                "ridge"
            ],
            val_stage2=stage2_val,
            val_y=vh_val[
                "y"
            ],
            val_app_ref=app_ref_val,
        )

        c_best_epoch = int(
            c_bundle[
                "best_epoch"
            ]
        )

        alpha_motion = np.asarray(
            c_bundle[
                "alpha_motion"
            ],
            dtype=np.float32,
        )

        alpha_heading = np.asarray(
            c_bundle[
                "alpha_heading"
            ],
            dtype=np.float32,
        )

        val_seq_t, val_ctx_t = (
            cmod.transform_inputs(
                seq_val,
                context_val,
                c_bundle[
                    "scaler"
                ],
            )
        )

        raw_vh_val = cmod.predict_raw(
            torch,
            c_bundle[
                "model"
            ],
            val_seq_t,
            val_ctx_t,
            c_bundle[
                "scaler"
            ][
                "residual_std"
            ],
            c_bundle[
                "device"
            ],
        )

        final_vh_val = cmod.apply_alphas_np(
            vh_val[
                "ridge"
            ],
            raw_vh_val,
            alpha_motion,
            alpha_heading,
        )

        c_metrics = (
            cmod.metrics_per_horizon(
                vh_val[
                    "y"
                ],
                final_vh_val,
                stage2_val,
                app_ref_val,
                "validation",
                "Stage15C-Final",
            )
        )

        c_metrics.insert(
            0,
            "seed",
            seed,
        )

        c_horizon_frames.append(
            c_metrics
        )

        for j, h in enumerate(
            HORIZONS
        ):
            vh_alpha_rows.append(
                {
                    "seed": seed,
                    "horizon_min": h,
                    "alpha_E": float(
                        alpha_motion[
                            j,
                            0
                        ]
                    ),
                    "alpha_N": float(
                        alpha_motion[
                            j,
                            1
                        ]
                    ),
                    "alpha_HDG": float(
                        alpha_heading[
                            j
                        ]
                    ),
                }
            )

        log(
            f"  [15C] best epoch = {c_best_epoch}"
        )

        for j, h in enumerate(
            HORIZONS
        ):
            log(
                f"         {h:2d} min | "
                f"alpha_E={alpha_motion[j,0]:.6f} | "
                f"alpha_N={alpha_motion[j,1]:.6f} | "
                f"alpha_HDG={alpha_heading[j]:.6f}"
            )

        # =============================================================
        # Seed summary.
        # =============================================================
        row = {
            "seed": seed,
            "Stage15B_best_epoch": b_best_epoch,
            "Stage15C_best_epoch": c_best_epoch,
            **summarize_truewind_metrics(
                tw_metrics
            ),
            **summarize_stage15c_metrics(
                c_metrics
            ),
        }

        per_seed_rows.append(
            row
        )

        log("")
        log(
            "  [SEED SUMMARY]"
        )
        log(
            f"    TrueWind vector RMSE = "
            f"{row['TrueWind_vector_RMSE_mps']:.6f}"
        )
        log(
            f"    Vessel vector RMSE   = "
            f"{row['Vessel_vector_RMSE_mps']:.6f}"
        )
        log(
            f"    Apparent vector RMSE = "
            f"{row['AppVector_RMSE_mps']:.6f}"
        )
        log(
            f"    AWS RMSE             = "
            f"{row['AWS_RMSE_mps']:.6f}"
        )
        log(
            f"    AWA MAE              = "
            f"{row['AWA_MAE_deg']:.6f} deg"
        )

        del c_bundle
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Aggregate.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 3/7] Aggregating five seeds."
    )

    per_seed_df = pd.DataFrame(
        per_seed_rows
    )

    aggregate_df = aggregate_seed_summary(
        per_seed_df
    )

    tw_horizon_df = pd.concat(
        tw_horizon_frames,
        ignore_index=True,
    )

    c_horizon_df = pd.concat(
        c_horizon_frames,
        ignore_index=True,
    )

    tw_alpha_df = pd.DataFrame(
        tw_alpha_rows
    )

    vh_alpha_df = pd.DataFrame(
        vh_alpha_rows
    )

    old_new_df = old_vs_new_table(
        per_seed_df
    )

    # -----------------------------------------------------------------
    # Save.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 4/7] Saving CSV outputs."
    )

    per_seed_df.to_csv(
        out
        / "15D_per_seed_validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    aggregate_df.to_csv(
        out
        / "15D_validation_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tw_horizon_df.to_csv(
        out
        / "15D_truewind_per_horizon_all_seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    c_horizon_df.to_csv(
        out
        / "15D_stage15c_per_horizon_all_seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tw_alpha_df.to_csv(
        out
        / "15D_alpha_truewind_all_seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    vh_alpha_df.to_csv(
        out
        / "15D_alpha_vessel_hdg_all_seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    old_new_df.to_csv(
        out
        / "15D_old_vs_new_manuscript_table6.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Important headline aggregate.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 5/7] Headline five-seed validation results."
    )

    headline_metrics = [
        "TrueWind_vector_RMSE_mps",
        "Vessel_vector_RMSE_mps",
        "SOG_RMSE_mps",
        "HDG_MAE_deg",
        "AppVector_RMSE_mps",
        "AWS_RMSE_mps",
        "AWA_MAE_deg",
    ]

    headline = {}

    log("")
    log("=" * 152)
    log(
        "15D FINAL — NEW STAGED PHYSICS-COMPACT, SD1090 VALIDATION, FIVE SEEDS"
    )
    log("=" * 152)

    for metric in headline_metrics:
        vals = per_seed_df[
            metric
        ].to_numpy(
            dtype=np.float64
        )

        mean = float(
            np.mean(
                vals
            )
        )

        std = float(
            np.std(
                vals,
                ddof=1,
            )
        )

        headline[
            metric
        ] = {
            "mean": mean,
            "std": std,
        }

        log(
            f"{metric:30s} = "
            f"{mean:.6f} ± {std:.6f}"
        )

    log("")
    log(
        f"Neural trainable parameters = "
        f"{NEW_NEURAL_TRAINABLE_PARAMETERS:,}"
    )

    log(
        f"Total fitted coefficients incl. two Ridge anchors = "
        f"{NEW_TOTAL_FITTED_COEFFICIENTS:,}"
    )

    # -----------------------------------------------------------------
    # Old vs new.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 6/7] Matched numerical comparison with old manuscript Table 6."
    )

    log("")
    log(
        old_new_df.to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Report.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 7/7] Writing report."
    )

    report = {
        "experiment": "15D",
        "purpose": (
            "Five-seed TRAIN/VALIDATION confirmation of final staged Physics-Compact"
        ),
        "internal_test_loaded": False,
        "seeds": SEEDS,
        "dataset_dir": args.dataset_dir,
        "stage15a_truewind_dir": args.stage15a_truewind_dir,
        "stage15a_vessel_dir": args.stage15a_vessel_dir,
        "apparent_reference_audit": apparent_audit,
        "frozen_architecture": {
            "Stage15B": {
                "GRU_layers": 2,
                "hidden": 48,
                "alpha_cap": 1.0,
                "trainable_parameters": STAGE15B_NN_PARAMETERS,
            },
            "Stage15C": {
                "GRU_layers": 2,
                "hidden": 48,
                "shared_hidden": 64,
                "alpha_cap": 1.0,
                "trainable_parameters": STAGE15C_NN_PARAMETERS,
            },
            "TrueWind_Ridge_coefficients": TRUEWIND_RIDGE_COEFFICIENTS,
            "VesselHDG_Ridge_coefficients": VESSEL_HDG_RIDGE_COEFFICIENTS,
            "neural_trainable_parameters": NEW_NEURAL_TRAINABLE_PARAMETERS,
            "total_fitted_coefficients": NEW_TOTAL_FITTED_COEFFICIENTS,
        },
        "headline": headline,
        "old_manuscript_table6": OLD_TABLE6,
        "old_vs_new": old_new_df.to_dict(
            orient="records"
        ),
        "per_seed": per_seed_df.to_dict(
            orient="records"
        ),
    }

    save_json(
        out
        / "15D_REPORT.json",
        report,
    )

    with (
        out
        / "15D_REPORT.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "15D — FIVE-SEED VALIDATION CONFIRMATION\n"
        )
        f.write(
            "="
            * 140
            + "\n\n"
        )
        f.write(
            "INTERNAL TEST WAS NOT LOADED.\n\n"
        )

        f.write(
            "HEADLINE NEW MODEL\n"
        )
        f.write(
            "-"
            * 140
            + "\n"
        )

        for metric in headline_metrics:
            vals = per_seed_df[
                metric
            ].to_numpy(
                dtype=np.float64
            )

            f.write(
                f"{metric:30s} = "
                f"{np.mean(vals):.8f} ± "
                f"{np.std(vals, ddof=1):.8f}\n"
            )

        f.write(
            "\nOLD MANUSCRIPT TABLE 6 vs NEW\n"
        )
        f.write(
            "-"
            * 140
            + "\n"
        )

        f.write(
            old_new_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nPER-SEED\n"
        )
        f.write(
            "-"
            * 140
            + "\n"
        )

        f.write(
            per_seed_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nMODEL SIZE\n"
        )
        f.write(
            "-"
            * 140
            + "\n"
        )

        f.write(
            f"Neural trainable parameters: "
            f"{NEW_NEURAL_TRAINABLE_PARAMETERS}\n"
        )

        f.write(
            f"Total fitted coefficients including two Ridge anchors: "
            f"{NEW_TOTAL_FITTED_COEFFICIENTS}\n"
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
        sys.exit(1)
