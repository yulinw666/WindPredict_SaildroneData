# -*- coding: utf-8 -*-
r"""
14B_O_Frozen14A_Tropical_PhaseOracle.py

PURPOSE
-------
Find the Tropical Atlantic 10-min point-sampling phase that gives the LOWEST
RMSE for the ALREADY-FROZEN Stage-14A model.

This is NOT a retraining experiment.

FROZEN:
    - final Base4-Ridge fitted on the original phase-0 development dataset
    - Stage-14A residual-network weights
    - Stage-14A residual scalers
    - Stage-14A active-component mask
    - Stage-14A alpha_U / alpha_V

ONLY CHANGED:
    - Tropical Atlantic point-sampling phase p in {0,...,9}

For each phase:
    raw 1-min Tropical
        -> current Stage-12G point-sampling protocol at phase p
        -> frozen phase-0 Ridge
        -> frozen Stage-14A residual NN
        -> frozen component-wise alpha
        -> RMSE metrics

PRIMARY ORACLE:
    phase minimizing final vector RMSE

ALSO REPORTED:
    best phase by U RMSE
    best phase by V RMSE
    best phase by WS RMSE
    best phase by WD RMSE

IMPORTANT SCIENTIFIC STATUS
---------------------------
The phase is selected USING Tropical Atlantic ground-truth labels.

Therefore:
    THIS IS A POST-HOC / ORACLE PHASE AUDIT.
    THE SELECTED-PHASE METRICS ARE NOT AN INDEPENDENT TEST RESULT.

It is appropriate for:
    - diagnosing the attainable performance of the frozen model
    - checking whether phase alignment explains part of the gap
    - designing a later calibration protocol

It is NOT appropriate for:
    - claiming unbiased independent-test performance against BP-STGNN

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\14B_O_Frozen14A_Tropical_PhaseOracle.py" `
  --project-root "D:\project\WindPredict_SaildroneData" `
  --raw-dir "D:\project\WindPredict_SaildroneData\data\raw\RawData" `
  --stage12g-dataset "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" `
  --stage14a-dir "D:\project\WindPredict_SaildroneData\data\forecasting\14A_RidgeAnchored_ComponentSafeResidual_v0_1" `
  --stage14a-script "D:\project\WindPredict_SaildroneData\src\14A_train_RidgeAnchored_ComponentSafeResidual_FinalTropical.py" `
  --stage13c-script "D:\project\WindPredict_SaildroneData\src\13C_Nie_Protocol_Forensic_Ridge_Audit.py" `
  --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\14B_O_Frozen14A_TropicalPhaseOracle_v0_1"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\project\WindPredict_SaildroneData")

DEFAULT_RAW_DIR = ROOT / "data" / "raw" / "RawData"

DEFAULT_STAGE12G = (
    ROOT
    / "data"
    / "forecasting"
    / "12G_Nie_point_sampled_benchmark_v0_1"
    / "dataset"
)

DEFAULT_STAGE14A_DIR = (
    ROOT
    / "data"
    / "forecasting"
    / "14A_RidgeAnchored_ComponentSafeResidual_v0_1"
)

DEFAULT_STAGE14A_SCRIPT = (
    ROOT
    / "src"
    / "14A_train_RidgeAnchored_ComponentSafeResidual_FinalTropical.py"
)

DEFAULT_STAGE13C_SCRIPT = (
    ROOT
    / "src"
    / "13C_Nie_Protocol_Forensic_Ridge_Audit.py"
)

DEFAULT_OUT = (
    ROOT
    / "data"
    / "forecasting"
    / "14B_O_Frozen14A_TropicalPhaseOracle_v0_1"
)

DEV = [
    "Antarctic",
    "Atlantic",
    "West Coast",
]

TEST = "Tropical Atlantic"

PHASES = list(range(10))


def log(msg=""):
    print(msg, flush=True)


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load module: {path}"
        )

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    return mod


def safe_torch_load(torch, path, device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def save_json(path: Path, obj):
    def cv(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return None if not np.isfinite(x) else float(x)
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, dict):
            return {str(k): cv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [cv(v) for v in x]
        return x

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


def load_tropical_raw(
    project_root: Path,
    raw_dir: Path,
    progress_every: int,
):
    stage12b_path = (
        project_root
        / "src"
        / "12B_build_Nie_same_mission_benchmark_dataset.py"
    )

    stage12b = load_module(
        stage12b_path,
        "stage12b_for_14bo",
    )

    xr = stage12b.import_xarray()

    grouped = stage12b.discover_mission_files(
        raw_dir
    )

    paths = grouped[
        TEST
    ]

    log(
        f"[RAW LOAD] {TEST} | files={len(paths)}"
    )

    raw, raw_audit, _ = (
        stage12b.load_mission_raw(
            xr,
            TEST,
            paths,
            progress_every,
        )
    )

    raw = (
        raw.sort_values(
            "time_ns"
        )
        .drop_duplicates(
            "time_ns",
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    return raw, raw_audit


def build_phase_item(
    mod13c,
    raw,
    phase,
):
    """
    Rebuild Tropical with EXACT current 12G sensitivity protocol:
        UTC phase p
        no interpolation
        COMMON8 eligibility
        10-min point-sampling first
        10-min selected-point differentials
    """
    item = mod13c.build_samples(
        raw,
        TEST,
        phase_mode="UTC",
        phase=int(phase),
        diff_mode="10min_after_sampling",
        gap=0,
        eligibility="COMMON8",
    )

    X4 = np.asarray(
        item[
            "X4"
        ],
        dtype=np.float32,
    )

    # Compatibility tensor for frozen 14A functions.
    X9 = np.zeros(
        (
            len(X4),
            X4.shape[1],
            9,
        ),
        dtype=np.float32,
    )

    X9[:, :, :4] = X4

    return {
        "X_full9": X9,
        "y_uv": np.asarray(
            item[
                "y"
            ],
            dtype=np.float32,
        ),
        "target": np.asarray(
            item[
                "target"
            ],
            dtype=np.int64,
        ),
        "mission": TEST,
        "audit": item[
            "audit"
        ],
    }


def reconstruct_frozen_ridge(
    mod14a,
    dataset_dir: Path,
):
    """
    Reconstruct the EXACT final Stage-14A Ridge:
        fit Base4-Ridge on all three development missions,
        using the original phase-0 Stage12G dataset.
    """
    root = mod14a.resolve_dataset_root(
        dataset_dir
    )

    dev_items = [
        mod14a.load_mission(
            root,
            mission,
        )
        for mission in DEV
    ]

    dev_all = mod14a.concat_missions(
        dev_items
    )

    Xbase = mod14a.build_base4(
        dev_all[
            "X_full9"
        ]
    )

    ridge_model, ridge_scaler = (
        mod14a.fit_ridge(
            Xbase,
            dev_all[
                "y_uv"
            ],
        )
    )

    return (
        ridge_model,
        ridge_scaler,
        mod14a.ridge_parameter_count(
            ridge_model
        ),
    )


def load_frozen_residual(
    mod14a,
    torch,
    nn,
    stage14a_dir: Path,
    device,
):
    checkpoint_path = (
        stage14a_dir
        / "final_model"
        / "14A_final.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint = safe_torch_load(
        torch,
        checkpoint_path,
        device,
    )

    ModelClass = mod14a.build_model_class(
        torch,
        nn,
    )

    model = ModelClass().to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "residual_state_dict"
        ],
        strict=True,
    )

    model.eval()

    scalers = checkpoint[
        "residual_scalers"
    ]

    alphas = np.asarray(
        checkpoint[
            "frozen_alphas"
        ],
        dtype=np.float32,
    ).reshape(2)

    active = np.asarray(
        checkpoint[
            "active_components"
        ],
        dtype=bool,
    ).reshape(2)

    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint": checkpoint,
        "model": model,
        "scalers": scalers,
        "alphas": alphas,
        "active": active,
        "nn_params": mod14a.count_parameters(
            model
        ),
    }


def evaluate_phase(
    *,
    mod14a,
    torch,
    frozen_ridge,
    frozen_residual,
    phase_item,
    device,
):
    ridge_model, ridge_scaler, ridge_params = (
        frozen_ridge
    )

    Xbase = mod14a.build_base4(
        phase_item[
            "X_full9"
        ]
    )

    ridge_pred = mod14a.ridge_predict(
        ridge_model,
        ridge_scaler,
        Xbase,
    )

    features = mod14a.build_residual_features(
        phase_item[
            "X_full9"
        ],
        ridge_pred,
    )

    transformed = (
        mod14a.transform_residual_features(
            features,
            frozen_residual[
                "scalers"
            ],
        )
    )

    amp_enabled = bool(
        mod14a.FREEZE.use_amp
        and device.type == "cuda"
    )

    raw_residual = (
        mod14a.predict_raw_residual(
            torch=torch,
            model=frozen_residual[
                "model"
            ],
            transformed=transformed,
            residual_std=frozen_residual[
                "scalers"
            ][
                "residual_std"
            ],
            device=device,
            amp_enabled=amp_enabled,
        )
    )

    final_pred = mod14a.apply_alphas(
        ridge_pred,
        raw_residual,
        frozen_residual[
            "alphas"
        ],
    )

    ridge_metrics = mod14a.evaluate_wind(
        phase_item[
            "y_uv"
        ],
        ridge_pred,
    )

    final_metrics = mod14a.evaluate_wind(
        phase_item[
            "y_uv"
        ],
        final_pred,
    )

    true_res = (
        phase_item[
            "y_uv"
        ]
        - ridge_pred
    )

    corr_u = mod14a.corrcoef_safe(
        true_res[:, 0],
        raw_residual[:, 0],
    )

    corr_v = mod14a.corrcoef_safe(
        true_res[:, 1],
        raw_residual[:, 1],
    )

    return {
        "ridge_pred": ridge_pred,
        "raw_residual": raw_residual,
        "final_pred": final_pred,
        "ridge_metrics": ridge_metrics,
        "final_metrics": final_metrics,
        "corr_u": corr_u,
        "corr_v": corr_v,
        "ridge_params": ridge_params,
    }


def best_row(df, metric):
    idx = df[
        metric
    ].idxmin()

    return df.loc[
        idx
    ].to_dict()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--project-root",
        type=Path,
        default=ROOT,
    )

    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
    )

    ap.add_argument(
        "--stage12g-dataset",
        type=Path,
        default=DEFAULT_STAGE12G,
    )

    ap.add_argument(
        "--stage14a-dir",
        type=Path,
        default=DEFAULT_STAGE14A_DIR,
    )

    ap.add_argument(
        "--stage14a-script",
        type=Path,
        default=DEFAULT_STAGE14A_SCRIPT,
    )

    ap.add_argument(
        "--stage13c-script",
        type=Path,
        default=DEFAULT_STAGE13C_SCRIPT,
    )

    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
    )

    ap.add_argument(
        "--progress-every",
        type=int,
        default=25,
    )

    ap.add_argument(
        "--force-cpu",
        action="store_true",
    )

    a = ap.parse_args()

    out = a.output_dir
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    mod14a = load_module(
        a.stage14a_script,
        "stage14a_for_14bo",
    )

    mod13c = load_module(
        a.stage13c_script,
        "stage13c_for_14bo",
    )

    torch, nn = mod14a.import_torch()

    mod14a.configure_cuda(
        torch
    )

    device = torch.device(
        "cpu"
        if (
            a.force_cpu
            or not torch.cuda.is_available()
        )
        else "cuda"
    )

    log("=" * 140)
    log(
        "14B-O — FROZEN 14A TROPICAL ATLANTIC PHASE ORACLE"
    )
    log("=" * 140)

    log(
        "NO retraining. Frozen Ridge + frozen residual NN + frozen alpha."
    )

    log(
        "Only Tropical UTC phase 0..9 changes."
    )

    log(
        "WARNING: phase is selected using Tropical labels -> POST-HOC ORACLE."
    )

    log(
        f"device = {device}"
    )

    # -------------------------------------------------------------
    # Frozen model.
    # -------------------------------------------------------------
    log("")
    log(
        "[STAGE A] Reconstructing frozen phase-0 final Ridge."
    )

    frozen_ridge = (
        reconstruct_frozen_ridge(
            mod14a,
            a.stage12g_dataset,
        )
    )

    log(
        f"  Ridge params = {frozen_ridge[2]}"
    )

    log(
        "[STAGE B] Loading frozen 14A residual model."
    )

    frozen_residual = (
        load_frozen_residual(
            mod14a,
            torch,
            nn,
            a.stage14a_dir,
            device,
        )
    )

    log(
        f"  checkpoint = {frozen_residual['checkpoint_path']}"
    )

    log(
        f"  active U/V = "
        f"{bool(frozen_residual['active'][0])}/"
        f"{bool(frozen_residual['active'][1])}"
    )

    log(
        f"  alpha U/V  = "
        f"{frozen_residual['alphas'][0]:.6f}/"
        f"{frozen_residual['alphas'][1]:.6f}"
    )

    log(
        f"  NN params   = {frozen_residual['nn_params']}"
    )

    # -------------------------------------------------------------
    # Load Tropical raw.
    # -------------------------------------------------------------
    log("")
    log(
        "[STAGE C] Loading Tropical Atlantic 1-min raw data."
    )

    raw, raw_audit = load_tropical_raw(
        a.project_root,
        a.raw_dir,
        a.progress_every,
    )

    # -------------------------------------------------------------
    # Phase sweep, no model training.
    # -------------------------------------------------------------
    rows = []
    phase_predictions = {}

    for phase in PHASES:
        log("")
        log(
            f"[PHASE {phase}] frozen-model inference"
        )

        item = build_phase_item(
            mod13c,
            raw,
            phase,
        )

        result = evaluate_phase(
            mod14a=mod14a,
            torch=torch,
            frozen_ridge=frozen_ridge,
            frozen_residual=(
                frozen_residual
            ),
            phase_item=item,
            device=device,
        )

        rm = result[
            "ridge_metrics"
        ]

        fm = result[
            "final_metrics"
        ]

        row = {
            "phase": int(
                phase
            ),
            "samples": int(
                len(
                    item[
                        "y_uv"
                    ]
                )
            ),
            "Ridge_U_RMSE_mps": (
                rm[
                    "U_RMSE_mps"
                ]
            ),
            "Ridge_V_RMSE_mps": (
                rm[
                    "V_RMSE_mps"
                ]
            ),
            "Ridge_vector_RMSE_mps": (
                rm[
                    "vector_RMSE_mps"
                ]
            ),
            "Ridge_WS_RMSE_mps": (
                rm[
                    "WS_RMSE_mps"
                ]
            ),
            "Ridge_WD_RMSE_deg": (
                rm[
                    "WD_RMSE_deg"
                ]
            ),
            "Final_U_RMSE_mps": (
                fm[
                    "U_RMSE_mps"
                ]
            ),
            "Final_V_RMSE_mps": (
                fm[
                    "V_RMSE_mps"
                ]
            ),
            "Final_vector_RMSE_mps": (
                fm[
                    "vector_RMSE_mps"
                ]
            ),
            "Final_WS_RMSE_mps": (
                fm[
                    "WS_RMSE_mps"
                ]
            ),
            "Final_WD_RMSE_deg": (
                fm[
                    "WD_RMSE_deg"
                ]
            ),
            "U_improve_vs_phase_Ridge_fraction": (
                mod14a.rel_improve(
                    rm[
                        "U_RMSE_mps"
                    ],
                    fm[
                        "U_RMSE_mps"
                    ],
                )
            ),
            "V_improve_vs_phase_Ridge_fraction": (
                mod14a.rel_improve(
                    rm[
                        "V_RMSE_mps"
                    ],
                    fm[
                        "V_RMSE_mps"
                    ],
                )
            ),
            "vector_improve_vs_phase_Ridge_fraction": (
                mod14a.rel_improve(
                    rm[
                        "vector_RMSE_mps"
                    ],
                    fm[
                        "vector_RMSE_mps"
                    ],
                )
            ),
            "WS_improve_vs_phase_Ridge_fraction": (
                mod14a.rel_improve(
                    rm[
                        "WS_RMSE_mps"
                    ],
                    fm[
                        "WS_RMSE_mps"
                    ],
                )
            ),
            "WD_improve_vs_phase_Ridge_fraction": (
                mod14a.rel_improve(
                    rm[
                        "WD_RMSE_deg"
                    ],
                    fm[
                        "WD_RMSE_deg"
                    ],
                )
            ),
            "raw_residual_corr_U": (
                result[
                    "corr_u"
                ]
            ),
            "raw_residual_corr_V": (
                result[
                    "corr_v"
                ]
            ),
        }

        rows.append(
            row
        )

        phase_predictions[
            phase
        ] = {
            "y_true": item[
                "y_uv"
            ],
            "target": item[
                "target"
            ],
            "ridge_pred": result[
                "ridge_pred"
            ],
            "raw_residual": result[
                "raw_residual"
            ],
            "final_pred": result[
                "final_pred"
            ],
        }

        log(
            f"  Final U/V/vector/WS/WD = "
            f"{fm['U_RMSE_mps']:.6f} / "
            f"{fm['V_RMSE_mps']:.6f} / "
            f"{fm['vector_RMSE_mps']:.6f} / "
            f"{fm['WS_RMSE_mps']:.6f} / "
            f"{fm['WD_RMSE_deg']:.6f}"
        )

    df = pd.DataFrame(
        rows
    ).sort_values(
        "phase"
    ).reset_index(
        drop=True
    )

    csv_path = (
        out
        / "14B_O_FROZEN14A_TROPICAL_PHASE_SWEEP.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # Oracle bests.
    # -------------------------------------------------------------
    best_vector = best_row(
        df,
        "Final_vector_RMSE_mps",
    )

    best_u = best_row(
        df,
        "Final_U_RMSE_mps",
    )

    best_v = best_row(
        df,
        "Final_V_RMSE_mps",
    )

    best_ws = best_row(
        df,
        "Final_WS_RMSE_mps",
    )

    best_wd = best_row(
        df,
        "Final_WD_RMSE_deg",
    )

    selected_phase = int(
        best_vector[
            "phase"
        ]
    )

    pred = phase_predictions[
        selected_phase
    ]

    np.savez_compressed(
        out
        / "14B_O_BEST_VECTOR_PHASE_PREDICTIONS.npz",
        phase=np.asarray(
            [
                selected_phase
            ],
            dtype=np.int64,
        ),
        target_time_ns=pred[
            "target"
        ].astype(
            np.int64
        ),
        y_true=pred[
            "y_true"
        ].astype(
            np.float32
        ),
        ridge_pred=pred[
            "ridge_pred"
        ].astype(
            np.float32
        ),
        raw_residual_pred=pred[
            "raw_residual"
        ].astype(
            np.float32
        ),
        final_pred=pred[
            "final_pred"
        ].astype(
            np.float32
        ),
        frozen_alphas=frozen_residual[
            "alphas"
        ].astype(
            np.float32
        ),
    )

    report = {
        "stage": "14B-O",
        "scientific_status": (
            "POST-HOC TROPICAL PHASE ORACLE; NOT INDEPENDENT TEST PERFORMANCE"
        ),
        "frozen_model": {
            "ridge_training_protocol": (
                "original Stage12G phase-0 development missions"
            ),
            "residual_checkpoint": str(
                frozen_residual[
                    "checkpoint_path"
                ]
            ),
            "active_U": bool(
                frozen_residual[
                    "active"
                ][
                    0
                ]
            ),
            "active_V": bool(
                frozen_residual[
                    "active"
                ][
                    1
                ]
            ),
            "alpha_U": float(
                frozen_residual[
                    "alphas"
                ][
                    0
                ]
            ),
            "alpha_V": float(
                frozen_residual[
                    "alphas"
                ][
                    1
                ]
            ),
            "parameter_count_total": int(
                frozen_ridge[
                    2
                ]
                + frozen_residual[
                    "nn_params"
                ]
            ),
        },
        "primary_oracle": {
            "criterion": (
                "minimum final vector RMSE"
            ),
            "best_phase": (
                selected_phase
            ),
            "row": (
                best_vector
            ),
        },
        "metric_specific_oracles": {
            "U": best_u,
            "V": best_v,
            "WS": best_ws,
            "WD": best_wd,
        },
        "phase_sweep": (
            df.to_dict(
                orient="records"
            )
        ),
    }

    save_json(
        out
        / "14B_O_REPORT.json",
        report,
    )

    with (
        out
        / "14B_O_REPORT.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "14B-O FROZEN 14A TROPICAL PHASE ORACLE\n"
        )
        f.write(
            "=" * 120
            + "\n\n"
        )
        f.write(
            "WARNING: phase selected using Tropical labels; "
            "this is post-hoc/oracle, not independent test performance.\n\n"
        )
        f.write(
            "FROZEN MODEL\n"
        )
        f.write(
            "-" * 120
            + "\n"
        )
        f.write(
            f"active U/V = "
            f"{bool(frozen_residual['active'][0])}/"
            f"{bool(frozen_residual['active'][1])}\n"
        )
        f.write(
            f"alpha U/V = "
            f"{frozen_residual['alphas'][0]:.8f}/"
            f"{frozen_residual['alphas'][1]:.8f}\n\n"
        )
        f.write(
            "PHASE SWEEP\n"
        )
        f.write(
            "-" * 120
            + "\n"
        )
        f.write(
            df.to_string(
                index=False
            )
        )
        f.write(
            "\n\nBEST PHASES\n"
        )
        f.write(
            "-" * 120
            + "\n"
        )
        f.write(
            f"Best vector RMSE phase = {int(best_vector['phase'])} | "
            f"{best_vector['Final_vector_RMSE_mps']:.8f}\n"
        )
        f.write(
            f"Best U RMSE phase      = {int(best_u['phase'])} | "
            f"{best_u['Final_U_RMSE_mps']:.8f}\n"
        )
        f.write(
            f"Best V RMSE phase      = {int(best_v['phase'])} | "
            f"{best_v['Final_V_RMSE_mps']:.8f}\n"
        )
        f.write(
            f"Best WS RMSE phase     = {int(best_ws['phase'])} | "
            f"{best_ws['Final_WS_RMSE_mps']:.8f}\n"
        )
        f.write(
            f"Best WD RMSE phase     = {int(best_wd['phase'])} | "
            f"{best_wd['Final_WD_RMSE_deg']:.8f} deg\n"
        )

    log("")
    log("=" * 140)
    log(
        "14B-O FROZEN 14A TROPICAL PHASE SWEEP"
    )
    log("=" * 140)
    log(
        df.to_string(
            index=False
        )
    )

    log("")
    log(
        "[POST-HOC ORACLE BEST PHASES]"
    )
    log(
        f"  vector : phase {int(best_vector['phase'])} "
        f"-> {best_vector['Final_vector_RMSE_mps']:.6f}"
    )
    log(
        f"  U      : phase {int(best_u['phase'])} "
        f"-> {best_u['Final_U_RMSE_mps']:.6f}"
    )
    log(
        f"  V      : phase {int(best_v['phase'])} "
        f"-> {best_v['Final_V_RMSE_mps']:.6f}"
    )
    log(
        f"  WS     : phase {int(best_ws['phase'])} "
        f"-> {best_ws['Final_WS_RMSE_mps']:.6f}"
    )
    log(
        f"  WD     : phase {int(best_wd['phase'])} "
        f"-> {best_wd['Final_WD_RMSE_deg']:.6f} deg"
    )

    log("")
    log(
        "WARNING: these phase optima use Tropical ground truth and are oracle/post-hoc."
    )

    log(
        f"[SAVED] {csv_path}"
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
