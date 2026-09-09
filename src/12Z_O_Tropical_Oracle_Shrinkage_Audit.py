# -*- coding: utf-8 -*-
r"""
12Z_O_Tropical_Oracle_Shrinkage_Audit.py

Stage 12Z-O
===========
Post-hoc Tropical Atlantic ORACLE shrinkage audit for an already-trained
Stage-12Z Ridge-anchored U/V residual model.

THIS SCRIPT DOES NOT RETRAIN THE RESIDUAL NETWORK.

PURPOSE
-------
Stage 12Z uses:

    U_final = U_Ridge + alpha_U(dev) * rhat_U
    V_final = V_Ridge + alpha_V(dev) * rhat_V

where alpha_U/V are calibrated ONLY on the three development missions.

Stage 12Z-O asks a diagnostic question:

    "If we were illegally allowed to use Tropical Atlantic labels only to
     choose alpha_U/V after seeing the test labels, how good could the
     EXISTING residual predictions possibly become?"

This separates two failure modes:

A) residual predictions are informative, but development-calibrated alpha
   does not transfer to Tropical;

B) residual predictions themselves carry too little Tropical information,
   so even an oracle alpha cannot approach BP-STGNN.

CRITICAL SCIENTIFIC STATUS
--------------------------
Every row containing "ORACLE" uses Tropical Atlantic ground truth to choose
one or more coefficients.

Therefore:

    ORACLE RESULTS ARE NOT TEST RESULTS.
    ORACLE RESULTS MUST NOT BE USED AS THE PROPOSED MODEL'S PERFORMANCE.
    ORACLE RESULTS MUST NOT BE CLAIMED AS A FAIR COMPARISON TO BP-STGNN.

They are diagnostic upper-bound / post-hoc sensitivity results only.

NO RETRAINING
-------------
This script:
    1) re-fits the deterministic Base4-Ridge on the same three development
       missions, exactly as Stage 12Z did;
    2) loads the already-trained Stage-12Z residual-network checkpoint;
    3) reconstructs the existing raw Tropical residual predictions;
    4) changes ONLY the scalar residual coefficients.

INPUTS
------
Stage-12G point-sampled dataset.

Stage-12Z output directory containing:
    final_model/12Z_final.pt
    FROZEN_BEFORE_TROPICAL.json   (optional consistency check)
    FINAL_TROPICAL_REPORT.json    (optional consistency check)

Stage-12Z Python script is loaded dynamically so that feature construction,
Ridge scaling, and neural-model definitions remain exactly identical.

AUDITED COEFFICIENT CASES
-------------------------
1) Ridge-only
       alpha_U = 0
       alpha_V = 0

2) Frozen-development-alpha
       alpha_U/V from the Stage-12Z checkpoint
   This should reproduce the previously reported 12Z final result.

3) ORACLE_same_cap_componentwise
       alpha_U, alpha_V optimized independently on Tropical labels
       subject to:
           0 <= alpha_j <= 0.75
   This is the fairest oracle diagnostic relative to the frozen 12Z model
   family because it keeps the original Stage-12Z alpha constraint.

4) ORACLE_unit_cap_componentwise
       0 <= alpha_j <= 1.0

5) ORACLE_unconstrained_componentwise
       alpha_j is the exact unconstrained least-squares scalar:
           alpha_j = sum(e_j*rhat_j) / sum(rhat_j^2)
   Negative values and values > 1 are allowed.

6) ORACLE_same_cap_V_only
       alpha_U = 0
       alpha_V = Tropical-optimal alpha_V in [0,0.75]
   This asks how much V can improve while leaving Ridge U completely intact.

7) ORACLE_same_cap_shared
       one shared alpha for both components in [0,0.75]:
           y_final = y_Ridge + alpha * rhat
       alpha minimizes total U+V squared error.

ANALYTICAL SOLUTION
-------------------
For one component:

    e = y_true - y_Ridge

    alpha* =
        sum(e * rhat)
        ----------------
        sum(rhat^2)

with optional clipping.

Because RMSE and MSE have the same minimizer, this exactly minimizes component
RMSE for the chosen residual predictions.

OUTPUTS
-------
12Z_O_ORACLE_TROPICAL_TABLE.csv
12Z_O_ORACLE_DIAGNOSTICS.json
12Z_O_ORACLE_DIAGNOSTICS.txt
12Z_O_TROPICAL_PREDICTIONS.npz

The report includes:
    raw residual correlation
    frozen alpha
    oracle alpha
    RMSE for U,V,vector,WS,WD
    improvement vs Base4-Ridge
    distance to published BP-STGNN
    whether each oracle row reaches BP per metric

INTERPRETATION
--------------
Especially inspect V:

If even ORACLE_unconstrained_componentwise gives V close to ~0.79:
    -> coefficient calibration is NOT the bottleneck.
    -> current rhat_V lacks useful Tropical signal.
    -> stop tuning alpha/gates for this residual network.

If oracle V falls strongly, e.g. ~0.72 or lower:
    -> residual prediction contains useful information.
    -> development-to-Tropical coefficient transfer is the bottleneck.
    -> adaptive/sample-wise gating may be worth studying.

Published BP-STGNN reference:
    U  = 0.748640 m/s
    V  = 0.705060 m/s
    WS = 0.699400 m/s
    WD = 5.584 deg

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\12Z_O_Tropical_Oracle_Shrinkage_Audit.py" `
  --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" `
  --stage12z-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12Z_RidgeAnchored_UVResidual_Shrinkage_FinalTropical_v0_1" `
  --stage12z-script "D:\project\WindPredict_SaildroneData\src\12Z_train_RidgeAnchored_UVResidual_AnalyticalShrinkage_FinalTropical.py"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


BP_REFERENCE = {
    "U_RMSE_mps": 0.748640,
    "V_RMSE_mps": 0.705060,
    "WS_RMSE_mps": 0.699400,
    "WD_RMSE_deg": 5.584000,
    "params": 242580,
}

DEFAULT_PROJECT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")

DEFAULT_DATASET_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12G_Nie_point_sampled_benchmark_v0_1"
    / "dataset"
)

DEFAULT_STAGE12Z_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12Z_RidgeAnchored_UVResidual_Shrinkage_FinalTropical_v0_1"
)

DEFAULT_STAGE12Z_SCRIPT = (
    DEFAULT_PROJECT_ROOT
    / "src"
    / "12Z_train_RidgeAnchored_UVResidual_AnalyticalShrinkage_FinalTropical.py"
)

EPS = 1e-12


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


def load_stage12z_module(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(
            f"Stage12Z script not found: {script_path}"
        )

    module_name = "stage12z_oracle_dependency"

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(script_path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load Stage12Z module from {script_path}"
        )

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    required = [
        "DEV_MISSIONS",
        "FINAL_MISSION",
        "ALPHA_MAX",
        "resolve_dataset_root",
        "load_mission",
        "concat_missions",
        "build_base4",
        "fit_ridge",
        "ridge_predict",
        "build_residual_features",
        "transform_residual_features",
        "build_model_class",
        "predict_raw_residual",
        "evaluate_wind",
        "import_torch",
        "configure_cuda",
    ]

    missing = [
        name
        for name in required
        if not hasattr(mod, name)
    ]

    if missing:
        raise RuntimeError(
            "The supplied Stage12Z script is incompatible. "
            f"Missing functions/constants: {missing}"
        )

    return mod


def safe_torch_load(torch, checkpoint_path, device):
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location=device,
        )


def corrcoef_safe(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    if (
        len(a) < 3
        or np.std(a) < EPS
        or np.std(b) < EPS
    ):
        return np.nan

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def analytic_alpha_1d(
    true_residual,
    predicted_residual,
    lower=None,
    upper=None,
):
    e = np.asarray(
        true_residual,
        dtype=np.float64,
    ).reshape(-1)

    r = np.asarray(
        predicted_residual,
        dtype=np.float64,
    ).reshape(-1)

    denom = float(
        np.dot(r, r)
    )

    if (
        denom <= EPS
        or not np.isfinite(denom)
    ):
        alpha = 0.0
    else:
        alpha = float(
            np.dot(e, r)
            / denom
        )

    if lower is not None:
        alpha = max(
            float(lower),
            alpha,
        )

    if upper is not None:
        alpha = min(
            float(upper),
            alpha,
        )

    return float(alpha)


def analytic_component_alphas(
    y_true,
    ridge_pred,
    raw_residual,
    lower=None,
    upper=None,
):
    true_res = (
        np.asarray(
            y_true,
            dtype=np.float64,
        )
        - np.asarray(
            ridge_pred,
            dtype=np.float64,
        )
    )

    raw = np.asarray(
        raw_residual,
        dtype=np.float64,
    )

    return np.asarray(
        [
            analytic_alpha_1d(
                true_res[:, 0],
                raw[:, 0],
                lower=lower,
                upper=upper,
            ),
            analytic_alpha_1d(
                true_res[:, 1],
                raw[:, 1],
                lower=lower,
                upper=upper,
            ),
        ],
        dtype=np.float64,
    )


def analytic_shared_alpha(
    y_true,
    ridge_pred,
    raw_residual,
    lower=None,
    upper=None,
):
    true_res = (
        np.asarray(
            y_true,
            dtype=np.float64,
        )
        - np.asarray(
            ridge_pred,
            dtype=np.float64,
        )
    ).reshape(-1)

    raw = np.asarray(
        raw_residual,
        dtype=np.float64,
    ).reshape(-1)

    return analytic_alpha_1d(
        true_res,
        raw,
        lower=lower,
        upper=upper,
    )


def apply_alphas(
    ridge_pred,
    raw_residual,
    alphas,
):
    a = np.asarray(
        alphas,
        dtype=np.float64,
    )

    if a.shape != (2,):
        raise ValueError(
            f"Expected alphas shape (2,), got {a.shape}"
        )

    return (
        np.asarray(
            ridge_pred,
            dtype=np.float64,
        )
        + np.asarray(
            raw_residual,
            dtype=np.float64,
        )
        * a[None, :]
    ).astype(np.float32)


def improvement_fraction(
    baseline,
    value,
):
    return float(
        (
            float(baseline)
            - float(value)
        )
        / max(
            abs(float(baseline)),
            EPS,
        )
    )


def add_row(
    rows,
    *,
    name,
    scientific_status,
    alphas,
    y_true,
    ridge_pred,
    raw_residual,
    evaluate_wind,
    ridge_metrics,
):
    final_pred = apply_alphas(
        ridge_pred,
        raw_residual,
        alphas,
    )

    m = evaluate_wind(
        y_true,
        final_pred,
    )

    row = {
        "model": name,
        "scientific_status": scientific_status,
        "alpha_U": float(alphas[0]),
        "alpha_V": float(alphas[1]),
        "wind_U_RMSE_mps": m[
            "wind_U_RMSE_mps"
        ],
        "wind_V_RMSE_mps": m[
            "wind_V_RMSE_mps"
        ],
        "wind_vector_RMSE_mps": m[
            "wind_vector_RMSE_mps"
        ],
        "wind_speed_RMSE_mps": m[
            "wind_speed_RMSE_mps"
        ],
        "wind_direction_RMSE_deg": m[
            "wind_direction_RMSE_deg"
        ],
        "U_improve_vs_Ridge_fraction": (
            improvement_fraction(
                ridge_metrics[
                    "wind_U_RMSE_mps"
                ],
                m[
                    "wind_U_RMSE_mps"
                ],
            )
        ),
        "V_improve_vs_Ridge_fraction": (
            improvement_fraction(
                ridge_metrics[
                    "wind_V_RMSE_mps"
                ],
                m[
                    "wind_V_RMSE_mps"
                ],
            )
        ),
        "WS_improve_vs_Ridge_fraction": (
            improvement_fraction(
                ridge_metrics[
                    "wind_speed_RMSE_mps"
                ],
                m[
                    "wind_speed_RMSE_mps"
                ],
            )
        ),
        "WD_improve_vs_Ridge_fraction": (
            improvement_fraction(
                ridge_metrics[
                    "wind_direction_RMSE_deg"
                ],
                m[
                    "wind_direction_RMSE_deg"
                ],
            )
        ),
        "beats_BP_U": bool(
            m[
                "wind_U_RMSE_mps"
            ]
            < BP_REFERENCE[
                "U_RMSE_mps"
            ]
        ),
        "beats_BP_V": bool(
            m[
                "wind_V_RMSE_mps"
            ]
            < BP_REFERENCE[
                "V_RMSE_mps"
            ]
        ),
        "beats_BP_WS": bool(
            m[
                "wind_speed_RMSE_mps"
            ]
            < BP_REFERENCE[
                "WS_RMSE_mps"
            ]
        ),
        "beats_BP_WD": bool(
            m[
                "wind_direction_RMSE_deg"
            ]
            < BP_REFERENCE[
                "WD_RMSE_deg"
            ]
        ),
    }

    rows.append(row)

    return final_pred, row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
    )

    parser.add_argument(
        "--stage12z-dir",
        type=Path,
        default=DEFAULT_STAGE12Z_DIR,
    )

    parser.add_argument(
        "--stage12z-script",
        type=Path,
        default=DEFAULT_STAGE12Z_SCRIPT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Default: <stage12z-dir>/12Z_O_Tropical_Oracle_Audit"
        ),
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
    )

    args = parser.parse_args()

    stage12z_dir = args.stage12z_dir

    if args.output_dir is None:
        output_dir = (
            stage12z_dir
            / "12Z_O_Tropical_Oracle_Audit"
        )
    else:
        output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        stage12z_dir
        / "final_model"
        / "12Z_final.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Stage12Z final checkpoint not found: {checkpoint_path}"
        )

    log("=" * 132)
    log(
        "12Z-O — TROPICAL ORACLE SHRINKAGE AUDIT"
    )
    log("=" * 132)
    log(
        "WARNING: ORACLE rows use Tropical labels and are NOT valid test results."
    )
    log(
        f"Stage12Z script : {args.stage12z_script}"
    )
    log(
        f"Stage12Z output : {stage12z_dir}"
    )
    log(
        f"Checkpoint      : {checkpoint_path}"
    )
    log(
        f"Output          : {output_dir}"
    )

    mod = load_stage12z_module(
        args.stage12z_script
    )

    dataset_root = mod.resolve_dataset_root(
        args.dataset_dir
    )

    torch, nn = mod.import_torch()

    mod.configure_cuda(
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

    log(
        f"Device          : {device}"
    )

    if device.type == "cuda":
        log(
            f"GPU             : {torch.cuda.get_device_name(0)}"
        )

    checkpoint = safe_torch_load(
        torch,
        checkpoint_path,
        device,
    )

    if checkpoint.get(
        "stage"
    ) != "12Z-final":
        raise RuntimeError(
            f"Unexpected checkpoint stage: {checkpoint.get('stage')}"
        )

    frozen_alphas = np.asarray(
        checkpoint[
            "frozen_alphas"
        ],
        dtype=np.float64,
    ).reshape(2)

    residual_scalers = checkpoint[
        "scalers"
    ]

    log("")
    log(
        f"Checkpoint frozen alpha_U/V = "
        f"{frozen_alphas[0]:.6f} / {frozen_alphas[1]:.6f}"
    )

    # -----------------------------------------------------------------
    # Reconstruct the exact final Ridge anchor.
    # -----------------------------------------------------------------
    dev_items = [
        mod.load_mission(
            dataset_root,
            mission,
        )
        for mission in mod.DEV_MISSIONS
    ]

    dev_all = mod.concat_missions(
        dev_items
    )

    tropical = mod.load_mission(
        dataset_root,
        mod.FINAL_MISSION,
    )

    Xdev_base = mod.build_base4(
        dev_all[
            "X_full9"
        ]
    )

    Xtrop_base = mod.build_base4(
        tropical[
            "X_full9"
        ]
    )

    ridge_model, ridge_scaler = mod.fit_ridge(
        Xdev_base,
        dev_all[
            "y_uv"
        ],
    )

    ridge_trop = mod.ridge_predict(
        ridge_model,
        ridge_scaler,
        Xtrop_base,
    )

    y_trop = tropical[
        "y_uv"
    ]

    ridge_metrics = mod.evaluate_wind(
        y_trop,
        ridge_trop,
    )

    # -----------------------------------------------------------------
    # Reconstruct existing Stage12Z raw residual prediction from checkpoint.
    # -----------------------------------------------------------------
    features_trop = mod.build_residual_features(
        tropical[
            "X_full9"
        ],
        ridge_trop,
    )

    transformed_trop = mod.transform_residual_features(
        features_trop,
        residual_scalers,
    )

    ModelClass = mod.build_model_class(
        torch,
        nn,
    )

    model = ModelClass().to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "state_dict"
        ],
        strict=True,
    )

    amp_enabled = bool(
        getattr(
            mod.FREEZE,
            "use_amp",
            True,
        )
        and device.type == "cuda"
    )

    raw_trop = mod.predict_raw_residual(
        torch=torch,
        model=model,
        transformed=transformed_trop,
        residual_std=residual_scalers[
            "residual_std"
        ],
        device=device,
        amp_enabled=amp_enabled,
    )

    if not np.isfinite(
        raw_trop
    ).all():
        raise RuntimeError(
            "Nonfinite reconstructed Tropical raw residual prediction."
        )

    true_residual = (
        np.asarray(
            y_trop,
            dtype=np.float64,
        )
        - np.asarray(
            ridge_trop,
            dtype=np.float64,
        )
    )

    raw_corr_U = corrcoef_safe(
        true_residual[
            :,
            0
        ],
        raw_trop[
            :,
            0
        ],
    )

    raw_corr_V = corrcoef_safe(
        true_residual[
            :,
            1
        ],
        raw_trop[
            :,
            1
        ],
    )

    log("")
    log(
        f"Reconstructed raw Tropical residual corr U/V = "
        f"{raw_corr_U:+.6f} / {raw_corr_V:+.6f}"
    )

    # -----------------------------------------------------------------
    # Oracle coefficients.
    # -----------------------------------------------------------------
    original_alpha_max = float(
        getattr(
            mod,
            "ALPHA_MAX",
            0.75,
        )
    )

    oracle_same_cap = analytic_component_alphas(
        y_trop,
        ridge_trop,
        raw_trop,
        lower=0.0,
        upper=original_alpha_max,
    )

    oracle_unit_cap = analytic_component_alphas(
        y_trop,
        ridge_trop,
        raw_trop,
        lower=0.0,
        upper=1.0,
    )

    oracle_unconstrained = analytic_component_alphas(
        y_trop,
        ridge_trop,
        raw_trop,
        lower=None,
        upper=None,
    )

    oracle_v_only = np.asarray(
        [
            0.0,
            oracle_same_cap[
                1
            ],
        ],
        dtype=np.float64,
    )

    oracle_shared_value = analytic_shared_alpha(
        y_trop,
        ridge_trop,
        raw_trop,
        lower=0.0,
        upper=original_alpha_max,
    )

    oracle_shared = np.asarray(
        [
            oracle_shared_value,
            oracle_shared_value,
        ],
        dtype=np.float64,
    )

    # -----------------------------------------------------------------
    # Evaluation table.
    # -----------------------------------------------------------------
    rows = []

    predictions = {}

    ridge_pred_row = {
        "model": "Base4-Ridge",
        "scientific_status": "VALID TEST BASELINE",
        "alpha_U": 0.0,
        "alpha_V": 0.0,
        "wind_U_RMSE_mps": ridge_metrics[
            "wind_U_RMSE_mps"
        ],
        "wind_V_RMSE_mps": ridge_metrics[
            "wind_V_RMSE_mps"
        ],
        "wind_vector_RMSE_mps": ridge_metrics[
            "wind_vector_RMSE_mps"
        ],
        "wind_speed_RMSE_mps": ridge_metrics[
            "wind_speed_RMSE_mps"
        ],
        "wind_direction_RMSE_deg": ridge_metrics[
            "wind_direction_RMSE_deg"
        ],
        "U_improve_vs_Ridge_fraction": 0.0,
        "V_improve_vs_Ridge_fraction": 0.0,
        "WS_improve_vs_Ridge_fraction": 0.0,
        "WD_improve_vs_Ridge_fraction": 0.0,
        "beats_BP_U": bool(
            ridge_metrics[
                "wind_U_RMSE_mps"
            ]
            < BP_REFERENCE[
                "U_RMSE_mps"
            ]
        ),
        "beats_BP_V": bool(
            ridge_metrics[
                "wind_V_RMSE_mps"
            ]
            < BP_REFERENCE[
                "V_RMSE_mps"
            ]
        ),
        "beats_BP_WS": bool(
            ridge_metrics[
                "wind_speed_RMSE_mps"
            ]
            < BP_REFERENCE[
                "WS_RMSE_mps"
            ]
        ),
        "beats_BP_WD": bool(
            ridge_metrics[
                "wind_direction_RMSE_deg"
            ]
            < BP_REFERENCE[
                "WD_RMSE_deg"
            ]
        ),
    }

    rows.append(
        ridge_pred_row
    )

    predictions[
        "ridge"
    ] = ridge_trop.astype(
        np.float32
    )

    frozen_pred, frozen_row = add_row(
        rows,
        name="12Z-FrozenDevelopmentAlpha",
        scientific_status="VALID STAGE12Z TEST RESULT",
        alphas=frozen_alphas,
        y_true=y_trop,
        ridge_pred=ridge_trop,
        raw_residual=raw_trop,
        evaluate_wind=mod.evaluate_wind,
        ridge_metrics=ridge_metrics,
    )

    predictions[
        "frozen"
    ] = frozen_pred

    same_cap_pred, same_cap_row = add_row(
        rows,
        name="ORACLE_same_cap_componentwise",
        scientific_status="POST-HOC ORACLE — NOT TEST PERFORMANCE",
        alphas=oracle_same_cap,
        y_true=y_trop,
        ridge_pred=ridge_trop,
        raw_residual=raw_trop,
        evaluate_wind=mod.evaluate_wind,
        ridge_metrics=ridge_metrics,
    )

    predictions[
        "oracle_same_cap_componentwise"
    ] = same_cap_pred

    unit_cap_pred, unit_cap_row = add_row(
        rows,
        name="ORACLE_unit_cap_componentwise",
        scientific_status="POST-HOC ORACLE — NOT TEST PERFORMANCE",
        alphas=oracle_unit_cap,
        y_true=y_trop,
        ridge_pred=ridge_trop,
        raw_residual=raw_trop,
        evaluate_wind=mod.evaluate_wind,
        ridge_metrics=ridge_metrics,
    )

    predictions[
        "oracle_unit_cap_componentwise"
    ] = unit_cap_pred

    unconstrained_pred, unconstrained_row = add_row(
        rows,
        name="ORACLE_unconstrained_componentwise",
        scientific_status="POST-HOC THEORETICAL ORACLE — NOT TEST PERFORMANCE",
        alphas=oracle_unconstrained,
        y_true=y_trop,
        ridge_pred=ridge_trop,
        raw_residual=raw_trop,
        evaluate_wind=mod.evaluate_wind,
        ridge_metrics=ridge_metrics,
    )

    predictions[
        "oracle_unconstrained_componentwise"
    ] = unconstrained_pred

    v_only_pred, v_only_row = add_row(
        rows,
        name="ORACLE_same_cap_V_only",
        scientific_status="POST-HOC ORACLE — NOT TEST PERFORMANCE",
        alphas=oracle_v_only,
        y_true=y_trop,
        ridge_pred=ridge_trop,
        raw_residual=raw_trop,
        evaluate_wind=mod.evaluate_wind,
        ridge_metrics=ridge_metrics,
    )

    predictions[
        "oracle_same_cap_V_only"
    ] = v_only_pred

    shared_pred, shared_row = add_row(
        rows,
        name="ORACLE_same_cap_shared",
        scientific_status="POST-HOC ORACLE — NOT TEST PERFORMANCE",
        alphas=oracle_shared,
        y_true=y_trop,
        ridge_pred=ridge_trop,
        raw_residual=raw_trop,
        evaluate_wind=mod.evaluate_wind,
        ridge_metrics=ridge_metrics,
    )

    predictions[
        "oracle_same_cap_shared"
    ] = shared_pred

    table = pd.DataFrame(
        rows
    )

    # Published BP reference is shown separately because there is no local
    # prediction vector available for it.
    bp_row = pd.DataFrame(
        [
            {
                "model": "BP-STGNN-published",
                "scientific_status": "PUBLISHED LITERATURE REFERENCE",
                "alpha_U": np.nan,
                "alpha_V": np.nan,
                "wind_U_RMSE_mps": BP_REFERENCE[
                    "U_RMSE_mps"
                ],
                "wind_V_RMSE_mps": BP_REFERENCE[
                    "V_RMSE_mps"
                ],
                "wind_vector_RMSE_mps": np.nan,
                "wind_speed_RMSE_mps": BP_REFERENCE[
                    "WS_RMSE_mps"
                ],
                "wind_direction_RMSE_deg": BP_REFERENCE[
                    "WD_RMSE_deg"
                ],
                "U_improve_vs_Ridge_fraction": np.nan,
                "V_improve_vs_Ridge_fraction": np.nan,
                "WS_improve_vs_Ridge_fraction": np.nan,
                "WD_improve_vs_Ridge_fraction": np.nan,
                "beats_BP_U": np.nan,
                "beats_BP_V": np.nan,
                "beats_BP_WS": np.nan,
                "beats_BP_WD": np.nan,
            }
        ]
    )

    display_table = pd.concat(
        [
            bp_row,
            table,
        ],
        ignore_index=True,
    )

    table_path = (
        output_dir
        / "12Z_O_ORACLE_TROPICAL_TABLE.csv"
    )

    display_table.to_csv(
        table_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Consistency check against existing Stage12Z final report if present.
    # -----------------------------------------------------------------
    consistency = {}

    existing_report_path = (
        stage12z_dir
        / "FINAL_TROPICAL_REPORT.json"
    )

    if existing_report_path.exists():
        with existing_report_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            existing = json.load(
                f
            )

        try:
            old_ours = existing[
                "final_tropical"
            ][
                "residual_model_metrics"
            ]

            consistency = {
                "existing_report_found": True,
                "U_abs_difference": abs(
                    float(
                        old_ours[
                            "wind_U_RMSE_mps"
                        ]
                    )
                    - float(
                        frozen_row[
                            "wind_U_RMSE_mps"
                        ]
                    )
                ),
                "V_abs_difference": abs(
                    float(
                        old_ours[
                            "wind_V_RMSE_mps"
                        ]
                    )
                    - float(
                        frozen_row[
                            "wind_V_RMSE_mps"
                        ]
                    )
                ),
                "WS_abs_difference": abs(
                    float(
                        old_ours[
                            "wind_speed_RMSE_mps"
                        ]
                    )
                    - float(
                        frozen_row[
                            "wind_speed_RMSE_mps"
                        ]
                    )
                ),
                "WD_abs_difference": abs(
                    float(
                        old_ours[
                            "wind_direction_RMSE_deg"
                        ]
                    )
                    - float(
                        frozen_row[
                            "wind_direction_RMSE_deg"
                        ]
                    )
                ),
            }

        except Exception:
            consistency = {
                "existing_report_found": True,
                "existing_report_parse_for_metric_check": False,
            }
    else:
        consistency = {
            "existing_report_found": False,
        }

    # -----------------------------------------------------------------
    # Diagnostics and interpretation thresholds.
    # -----------------------------------------------------------------
    oracle_v = float(
        unconstrained_row[
            "wind_V_RMSE_mps"
        ]
    )

    ridge_v = float(
        ridge_metrics[
            "wind_V_RMSE_mps"
        ]
    )

    bp_v = float(
        BP_REFERENCE[
            "V_RMSE_mps"
        ]
    )

    max_oracle_v_improvement = (
        ridge_v
        - oracle_v
    )

    required_to_bp = (
        ridge_v
        - bp_v
    )

    fraction_of_required_gap_closed = (
        max_oracle_v_improvement
        / required_to_bp
        if required_to_bp
        > EPS
        else np.nan
    )

    if oracle_v >= ridge_v - 0.005:
        interpretation = (
            "Even the unconstrained Tropical oracle coefficient barely "
            "improves V. The current residual prediction itself lacks enough "
            "Tropical V information; alpha/gating is not the main bottleneck."
        )
    elif oracle_v > 0.75:
        interpretation = (
            "Oracle alpha improves V somewhat, but remains far from the "
            "published BP-STGNN V result. Coefficient transfer contributes "
            "some error, but residual information is still insufficient."
        )
    elif oracle_v > bp_v:
        interpretation = (
            "Oracle alpha gives a substantial V improvement but still does "
            "not reach the published BP-STGNN V result. Adaptive gating may "
            "be worth testing, but it cannot fully explain the gap."
        )
    else:
        interpretation = (
            "The existing residual predictions can reach or beat the "
            "published BP V value under a Tropical-label oracle alpha. "
            "This indicates coefficient/gating transfer is a major bottleneck. "
            "A development-only adaptive gating model is worth investigating."
        )

    diagnostics = {
        "scientific_status": (
            "Oracle rows are post-hoc diagnostics and not valid test results."
        ),
        "checkpoint": str(
            checkpoint_path
        ),
        "raw_residual_corr_U": float(
            raw_corr_U
        ),
        "raw_residual_corr_V": float(
            raw_corr_V
        ),
        "frozen_development_alphas": {
            "alpha_U": float(
                frozen_alphas[
                    0
                ]
            ),
            "alpha_V": float(
                frozen_alphas[
                    1
                ]
            ),
        },
        "oracle_same_cap": {
            "alpha_U": float(
                oracle_same_cap[
                    0
                ]
            ),
            "alpha_V": float(
                oracle_same_cap[
                    1
                ]
            ),
            "cap": float(
                original_alpha_max
            ),
        },
        "oracle_unit_cap": {
            "alpha_U": float(
                oracle_unit_cap[
                    0
                ]
            ),
            "alpha_V": float(
                oracle_unit_cap[
                    1
                ]
            ),
        },
        "oracle_unconstrained": {
            "alpha_U": float(
                oracle_unconstrained[
                    0
                ]
            ),
            "alpha_V": float(
                oracle_unconstrained[
                    1
                ]
            ),
        },
        "oracle_shared_same_cap": {
            "alpha": float(
                oracle_shared_value
            ),
        },
        "V_gap_diagnostic": {
            "Ridge_V_RMSE": ridge_v,
            "published_BP_V_RMSE": bp_v,
            "oracle_unconstrained_V_RMSE": (
                oracle_v
            ),
            "required_Ridge_to_BP_reduction_mps": (
                required_to_bp
            ),
            "max_oracle_reduction_mps": (
                max_oracle_v_improvement
            ),
            "fraction_of_BP_gap_closed_by_oracle": (
                fraction_of_required_gap_closed
            ),
        },
        "consistency_with_existing_12Z_report": (
            consistency
        ),
        "interpretation": (
            interpretation
        ),
    }

    diagnostics_path = (
        output_dir
        / "12Z_O_ORACLE_DIAGNOSTICS.json"
    )

    save_json(
        diagnostics_path,
        diagnostics,
    )

    predictions_path = (
        output_dir
        / "12Z_O_TROPICAL_PREDICTIONS.npz"
    )

    np.savez_compressed(
        predictions_path,
        y_true=np.asarray(
            y_trop,
            dtype=np.float32,
        ),
        ridge_pred=np.asarray(
            ridge_trop,
            dtype=np.float32,
        ),
        raw_residual_pred=np.asarray(
            raw_trop,
            dtype=np.float32,
        ),
        frozen_pred=np.asarray(
            frozen_pred,
            dtype=np.float32,
        ),
        oracle_same_cap_componentwise=np.asarray(
            same_cap_pred,
            dtype=np.float32,
        ),
        oracle_unit_cap_componentwise=np.asarray(
            unit_cap_pred,
            dtype=np.float32,
        ),
        oracle_unconstrained_componentwise=np.asarray(
            unconstrained_pred,
            dtype=np.float32,
        ),
        oracle_same_cap_V_only=np.asarray(
            v_only_pred,
            dtype=np.float32,
        ),
        oracle_same_cap_shared=np.asarray(
            shared_pred,
            dtype=np.float32,
        ),
        frozen_alphas=np.asarray(
            frozen_alphas,
            dtype=np.float32,
        ),
        oracle_same_cap_alphas=np.asarray(
            oracle_same_cap,
            dtype=np.float32,
        ),
        oracle_unit_cap_alphas=np.asarray(
            oracle_unit_cap,
            dtype=np.float32,
        ),
        oracle_unconstrained_alphas=np.asarray(
            oracle_unconstrained,
            dtype=np.float32,
        ),
    )

    txt_path = (
        output_dir
        / "12Z_O_ORACLE_DIAGNOSTICS.txt"
    )

    with txt_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "12Z-O TROPICAL ORACLE SHRINKAGE AUDIT\n"
        )
        f.write(
            "=" * 120
            + "\n\n"
        )

        f.write(
            "WARNING\n"
        )
        f.write(
            "-" * 120
            + "\n"
        )
        f.write(
            "ORACLE rows use Tropical Atlantic ground-truth labels to "
            "choose alpha. They are NOT valid test performance and MUST NOT "
            "be used as a fair BP-STGNN comparison.\n\n"
        )

        f.write(
            "RAW RESIDUAL DIAGNOSTICS\n"
        )
        f.write(
            "-" * 120
            + "\n"
        )
        f.write(
            f"corr_U = {raw_corr_U:+.8f}\n"
        )
        f.write(
            f"corr_V = {raw_corr_V:+.8f}\n\n"
        )

        f.write(
            "COEFFICIENTS\n"
        )
        f.write(
            "-" * 120
            + "\n"
        )
        f.write(
            f"Frozen dev alpha U/V      : "
            f"{frozen_alphas[0]:+.8f} / {frozen_alphas[1]:+.8f}\n"
        )
        f.write(
            f"Oracle same-cap U/V       : "
            f"{oracle_same_cap[0]:+.8f} / {oracle_same_cap[1]:+.8f}\n"
        )
        f.write(
            f"Oracle [0,1] U/V          : "
            f"{oracle_unit_cap[0]:+.8f} / {oracle_unit_cap[1]:+.8f}\n"
        )
        f.write(
            f"Oracle unconstrained U/V  : "
            f"{oracle_unconstrained[0]:+.8f} / "
            f"{oracle_unconstrained[1]:+.8f}\n"
        )
        f.write(
            f"Oracle shared same-cap     : "
            f"{oracle_shared_value:+.8f}\n\n"
        )

        f.write(
            "TROPICAL TABLE\n"
        )
        f.write(
            "-" * 120
            + "\n"
        )
        f.write(
            display_table.to_string(
                index=False
            )
        )
        f.write(
            "\n\n"
        )

        f.write(
            "V GAP DIAGNOSTIC\n"
        )
        f.write(
            "-" * 120
            + "\n"
        )
        f.write(
            f"Ridge V                         = {ridge_v:.8f}\n"
        )
        f.write(
            f"Published BP V                  = {bp_v:.8f}\n"
        )
        f.write(
            f"Oracle unconstrained V          = {oracle_v:.8f}\n"
        )
        f.write(
            f"Required Ridge->BP reduction    = {required_to_bp:.8f}\n"
        )
        f.write(
            f"Max oracle reduction            = {max_oracle_v_improvement:.8f}\n"
        )
        f.write(
            f"Fraction of BP gap oracle closes= "
            f"{fraction_of_required_gap_closed:.6f}\n\n"
        )

        f.write(
            "INTERPRETATION\n"
        )
        f.write(
            "-" * 120
            + "\n"
        )
        f.write(
            interpretation
            + "\n"
        )

    log("")
    log("=" * 132)
    log(
        "12Z-O TROPICAL ORACLE TABLE"
    )
    log("=" * 132)
    log(
        display_table.to_string(
            index=False
        )
    )

    log("")
    log(
        "Coefficient audit:"
    )
    log(
        f"  frozen dev alpha U/V     = "
        f"{frozen_alphas[0]:+.6f} / {frozen_alphas[1]:+.6f}"
    )
    log(
        f"  oracle same-cap U/V      = "
        f"{oracle_same_cap[0]:+.6f} / {oracle_same_cap[1]:+.6f}"
    )
    log(
        f"  oracle [0,1] U/V         = "
        f"{oracle_unit_cap[0]:+.6f} / {oracle_unit_cap[1]:+.6f}"
    )
    log(
        f"  oracle unconstrained U/V = "
        f"{oracle_unconstrained[0]:+.6f} / "
        f"{oracle_unconstrained[1]:+.6f}"
    )

    log("")
    log(
        "V gap diagnostic:"
    )
    log(
        f"  Ridge V                 = {ridge_v:.6f}"
    )
    log(
        f"  BP V                    = {bp_v:.6f}"
    )
    log(
        f"  Oracle unconstrained V  = {oracle_v:.6f}"
    )
    log(
        f"  Oracle closes           = "
        f"{100.0*fraction_of_required_gap_closed:.2f}% "
        f"of Ridge->BP V gap"
    )

    log("")
    log(
        f"[INTERPRETATION] {interpretation}"
    )

    log("")
    log(
        f"[SAVED] {table_path}"
    )
    log(
        f"[SAVED] {diagnostics_path}"
    )
    log(
        f"[SAVED] {txt_path}"
    )
    log(
        f"[SAVED] {predictions_path}"
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
