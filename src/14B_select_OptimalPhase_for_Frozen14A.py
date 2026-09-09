# -*- coding: utf-8 -*-
r"""
14B_select_OptimalPhase_for_Frozen14A.py

Stage 14B
=========
Development-only 10-min phase selection for the FROZEN Stage-14A method.

SCIENTIFIC PURPOSE
------------------
Stage 13C showed that the 10-min point-sampling phase materially changes the
Tropical Atlantic Ridge result. The current Stage-12G/14A benchmark uses UTC
phase 0:

    minute = 00,10,20,30,40,50

but the original paper does not uniquely specify the decimation anchor.

Stage 14B treats PHASE as the ONLY new preprocessing hyperparameter.

IMPORTANT:
    We do NOT select phase from Tropical Atlantic labels.

Instead:

    1) Keep the Stage-14A architecture/training recipe frozen.
    2) Keep the Stage-14A selected random seed frozen.
    3) For phase = 0..9, rebuild ONLY Antarctic / Atlantic / West Coast.
    4) Run the same two-stage 14A LOMO procedure:
           Round 1: Ridge
           Round 2: frozen-Ridge residual GRU
    5) Select the phase with the lowest mission-balanced development
       final VECTOR RMSE.
    6) Recompute the 14A component-safe alpha freeze for that phase.
    7) Save FROZEN_PHASE_BEFORE_TROPICAL.json.
    8) Only then load Tropical Atlantic raw data.
    9) Build ONLY the selected phase and evaluate once.

Thus the final selected-phase Tropical row remains a development-selected
benchmark result within this stage.

WHAT IS FROZEN FROM 14A
-----------------------
The following are NOT re-tuned across phases:
    - Base4 raw inputs: U,V,T,RH
    - six history points
    - +10-min target
    - Ridge alpha
    - residual feature engineering
    - residual GRU architecture
    - loss
    - optimizer/lr/weight decay
    - early stopping rule
    - component-safe acceptance rule
    - robust-alpha rule
    - random seed selected by Stage 14A

Only:
    UTC phase in {0,...,9}
is compared.

Each phase still gets its normal development-only early stopping and
component-safe alpha calibration, because these are part of the frozen 14A
training procedure.

PHASE SELECTION METRIC
----------------------
Primary:
    mean of the three LOMO final vector RMSE values

Tie-breaks:
    mean WS RMSE
    mean WD RMSE

This avoids choosing phase from Tropical and avoids an arbitrary weighted
mix of U/V/WS/WD.

RAW PROTOCOL
------------
Exactly the current 12G point-sampling definition:

    UTC phase p
    no interpolation
    COMMON8 finite-node eligibility
    10-min point sampling first
    dT/dP/dRH after selected points

Although 14A uses only U,V,T,RH, COMMON8 is retained so phase-0 sample
eligibility reproduces the current Stage-12G aligned sample set.

The first thing this script does is verify that rebuilt development phase 0
matches the existing Stage-12G Base4 samples and target timestamps.

OUTPUTS
-------
14B_phase_lomo_runs.csv
14B_phase_summary.csv
14B_phase0_reproduction_audit.csv
14B_selected_phase_component_evidence.csv
FROZEN_PHASE_BEFORE_TROPICAL.json
FINAL_TROPICAL_14B_SELECTED_PHASE.csv
FINAL_TROPICAL_14B_PREDICTIONS.npz
14B_REPORT.txt
14B_REPORT.json

INTERPRETATION
--------------
The selected phase is a development-selected preprocessing hyperparameter.
Do NOT report the post-hoc Stage-13C Tropical-best phase as the selected phase
unless development independently selects it.

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\14B_select_OptimalPhase_for_Frozen14A.py" `
  --project-root "D:\project\WindPredict_SaildroneData" `
  --raw-dir "D:\project\WindPredict_SaildroneData\data\raw\RawData" `
  --stage12g-dataset "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" `
  --stage14a-dir "D:\project\WindPredict_SaildroneData\data\forecasting\14A_RidgeAnchored_ComponentSafeResidual_v0_1" `
  --stage14a-script "D:\project\WindPredict_SaildroneData\src\14A_train_RidgeAnchored_ComponentSafeResidual_FinalTropical.py" `
  --stage13c-script "D:\project\WindPredict_SaildroneData\src\13C_Nie_Protocol_Forensic_Ridge_Audit.py" `
  --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\14B_Frozen14A_PhaseSelection_v0_1"

Smoke:
    add --debug-fast
    (checks phase-0 reproduction and trains only phase 0 / Antarctic holdout)
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


SCRIPT_VERSION = "0.1.0-14B-Frozen14A-DevelopmentPhaseSelection"

ROOT = Path(r"D:\project\WindPredict_SaildroneData")
DEFAULT_RAW = ROOT / "data" / "raw" / "RawData"
DEFAULT_12G = ROOT / "data" / "forecasting" / "12G_Nie_point_sampled_benchmark_v0_1" / "dataset"
DEFAULT_14A_DIR = ROOT / "data" / "forecasting" / "14A_RidgeAnchored_ComponentSafeResidual_v0_1"
DEFAULT_14A_SCRIPT = ROOT / "src" / "14A_train_RidgeAnchored_ComponentSafeResidual_FinalTropical.py"
DEFAULT_13C_SCRIPT = ROOT / "src" / "13C_Nie_Protocol_Forensic_Ridge_Audit.py"
DEFAULT_OUT = ROOT / "data" / "forecasting" / "14B_Frozen14A_PhaseSelection_v0_1"

DEV = ["Antarctic", "Atlantic", "West Coast"]
TEST = "Tropical Atlantic"

PHASES = list(range(10))
EPS = 1e-12

BP = {
    "U_RMSE_mps": 0.748640,
    "V_RMSE_mps": 0.705060,
    "WS_RMSE_mps": 0.699400,
    "WD_RMSE_deg": 5.584000,
    "params": 242580,
}


def log(s=""):
    print(s, flush=True)


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import {path}"
        )

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    return mod


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
            return {
                str(k): cv(v)
                for k, v in x.items()
            }
        if isinstance(x, (list, tuple)):
            return [cv(v) for v in x]
        return x

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            cv(obj),
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def load_raw_subset(
    project_root: Path,
    raw_dir: Path,
    missions,
    progress_every: int,
):
    """
    Reuse Stage12B's exact raw loader but load only requested missions.
    This is used to preserve the Tropical firewall during phase selection.
    """
    stage12b_path = (
        project_root
        / "src"
        / "12B_build_Nie_same_mission_benchmark_dataset.py"
    )

    stage12b = load_module(
        stage12b_path,
        "stage12b_for_14b",
    )

    xr = stage12b.import_xarray()

    grouped = stage12b.discover_mission_files(
        raw_dir
    )

    out = {}

    audit = []

    for mission in missions:
        paths = grouped[
            mission
        ]

        log(
            f"  [RAW LOAD] {mission} | files={len(paths)}"
        )

        raw, raw_audit, _ = (
            stage12b.load_mission_raw(
                xr,
                mission,
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

        out[
            mission
        ] = raw

        audit.append(
            {
                "mission": mission,
                "rows": int(
                    len(raw)
                ),
                "source_files": int(
                    len(paths)
                ),
                "loader_audit": str(
                    raw_audit
                ),
            }
        )

    return (
        out,
        pd.DataFrame(
            audit
        ),
    )


def convert_13c_to_14a(
    item,
):
    """
    14A only consumes the first four channels [U,V,T,RH].
    Build a 9-channel compatibility tensor with remaining channels = 0.
    Those zeros are never used by 14A.
    """
    X4 = np.asarray(
        item[
            "X4"
        ],
        dtype=np.float32,
    )

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
        "mission": item[
            "mission"
        ],
        "audit": item[
            "audit"
        ],
    }


def build_phase_mission(
    mod13c,
    raw,
    mission,
    phase,
):
    item = mod13c.build_samples(
        raw,
        mission,
        phase_mode="UTC",
        phase=int(
            phase
        ),
        diff_mode="10min_after_sampling",
        gap=0,
        eligibility="COMMON8",
    )

    return convert_13c_to_14a(
        item
    )


def phase0_reproduction_audit(
    mod14a,
    rebuilt,
    stage12g_dataset,
):
    rows = []

    root = mod14a.resolve_dataset_root(
        stage12g_dataset
    )

    for mission in DEV:
        original = mod14a.load_mission(
            root,
            mission,
        )

        new = rebuilt[
            mission
        ]

        old_x4 = mod14a.build_base4(
            original[
                "X_full9"
            ]
        )

        new_x4 = mod14a.build_base4(
            new[
                "X_full9"
            ]
        )

        same_n = (
            len(old_x4)
            == len(new_x4)
        )

        target_equal = (
            same_n
            and np.array_equal(
                original[
                    "target"
                ],
                new[
                    "target"
                ],
            )
        )

        x_maxerr = (
            float(
                np.max(
                    np.abs(
                        old_x4
                        - new_x4
                    )
                )
            )
            if same_n
            else np.inf
        )

        y_maxerr = (
            float(
                np.max(
                    np.abs(
                        original[
                            "y_uv"
                        ]
                        - new[
                            "y_uv"
                        ]
                    )
                )
            )
            if same_n
            else np.inf
        )

        passed = bool(
            same_n
            and target_equal
            and x_maxerr
            <= 2e-5
            and y_maxerr
            <= 2e-5
        )

        rows.append(
            {
                "mission": mission,
                "original_samples": int(
                    len(
                        old_x4
                    )
                ),
                "rebuilt_samples": int(
                    len(
                        new_x4
                    )
                ),
                "target_equal": bool(
                    target_equal
                ),
                "X4_max_abs_error": (
                    x_maxerr
                ),
                "y_max_abs_error": (
                    y_maxerr
                ),
                "PASS": passed,
            }
        )

    df = pd.DataFrame(
        rows
    )

    if not bool(
        df[
            "PASS"
        ].all()
    ):
        raise RuntimeError(
            "Phase-0 raw rebuild does not exactly reproduce Stage12G. "
            "Do not continue phase selection.\n"
            + df.to_string(
                index=False
            )
        )

    return df


def read_14a_freeze(
    stage14a_dir: Path,
):
    path = (
        stage14a_dir
        / "FROZEN_BEFORE_TROPICAL.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(
            f
        )

    return (
        path,
        data,
    )


def evaluate_prediction_file(
    mod14a,
    path: Path,
):
    with np.load(
        path,
        allow_pickle=False,
    ) as z:
        y = np.asarray(
            z[
                "y_true"
            ],
            dtype=np.float32,
        )

        ridge = np.asarray(
            z[
                "ridge_pred"
            ],
            dtype=np.float32,
        )

        final = np.asarray(
            z[
                "final_pred"
            ],
            dtype=np.float32,
        )

    mr = mod14a.evaluate_wind(
        y,
        ridge,
    )

    mf = mod14a.evaluate_wind(
        y,
        final,
    )

    return (
        mr,
        mf,
    )


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
        default=DEFAULT_RAW,
    )

    ap.add_argument(
        "--stage12g-dataset",
        type=Path,
        default=DEFAULT_12G,
    )

    ap.add_argument(
        "--stage14a-dir",
        type=Path,
        default=DEFAULT_14A_DIR,
    )

    ap.add_argument(
        "--stage14a-script",
        type=Path,
        default=DEFAULT_14A_SCRIPT,
    )

    ap.add_argument(
        "--stage13c-script",
        type=Path,
        default=DEFAULT_13C_SCRIPT,
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
        "--debug-fast",
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
        "stage14a_for_14b",
    )

    mod13c = load_module(
        a.stage13c_script,
        "stage13c_for_14b",
    )

    torch, nn = mod14a.import_torch()

    mod14a.configure_cuda(
        torch
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    freeze_path, freeze14a = (
        read_14a_freeze(
            a.stage14a_dir
        )
    )

    fixed_seed = int(
        freeze14a[
            "selected_seed"
        ]
    )

    log("=" * 140)
    log(
        "14B — DEVELOPMENT-ONLY OPTIMAL PHASE SELECTION FOR FROZEN 14A"
    )
    log("=" * 140)

    log(
        f"14A freeze  : {freeze_path}"
    )

    log(
        f"fixed seed  : {fixed_seed}"
    )

    log(
        f"device      : {device}"
    )

    log(
        "selection   : mean 3-mission LOMO final vector RMSE"
    )

    log(
        "[FIREWALL] Tropical Atlantic is NOT loaded during phase selection."
    )

    # -----------------------------------------------------------------
    # Load development raw only.
    # -----------------------------------------------------------------
    dev_raw, raw_audit = load_raw_subset(
        a.project_root,
        a.raw_dir,
        DEV,
        a.progress_every,
    )

    raw_audit.to_csv(
        out
        / "14B_development_raw_load_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Phase-0 reproduction check before any model sweep.
    # -----------------------------------------------------------------
    rebuilt0 = {
        mission: build_phase_mission(
            mod13c,
            dev_raw[
                mission
            ],
            mission,
            0,
        )
        for mission in DEV
    }

    reproduction = phase0_reproduction_audit(
        mod14a,
        rebuilt0,
        a.stage12g_dataset,
    )

    reproduction_path = (
        out
        / "14B_phase0_reproduction_audit.csv"
    )

    reproduction.to_csv(
        reproduction_path,
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log(
        "[PHASE-0 REPRODUCTION] PASS"
    )

    log(
        reproduction.to_string(
            index=False
        )
    )

    phases = (
        [0]
        if a.debug_fast
        else PHASES
    )

    holdouts = (
        ["Antarctic"]
        if a.debug_fast
        else DEV
    )

    phase_run_rows = []

    phase_phase_data = {}

    # -----------------------------------------------------------------
    # Development-only phase sweep.
    # -----------------------------------------------------------------
    for phase in phases:
        log("")
        log(
            "=" * 140
        )

        log(
            f"[PHASE {phase}] development-only 14A LOMO"
        )

        log(
            "=" * 140
        )

        if phase == 0:
            mission_data = rebuilt0
        else:
            mission_data = {
                mission: build_phase_mission(
                    mod13c,
                    dev_raw[
                        mission
                    ],
                    mission,
                    phase,
                )
                for mission in DEV
            }

        phase_phase_data[
            phase
        ] = mission_data

        phase_out = (
            out
            / f"phase_{phase}"
        )

        phase_out.mkdir(
            parents=True,
            exist_ok=True,
        )

        for held_out in holdouts:
            train_data = mod14a.concat_missions(
                [
                    mission_data[
                        m
                    ]
                    for m in DEV
                    if m
                    != held_out
                ]
            )

            val_data = mission_data[
                held_out
            ]

            log(
                f"  [HOLDOUT] {held_out} | "
                f"Ntrain={len(train_data['y_uv']):,} | "
                f"Nval={len(val_data['y_uv']):,}"
            )

            row = mod14a.train_one_fold(
                torch=torch,
                nn=nn,
                train_data=train_data,
                val_data=val_data,
                held_out=held_out,
                seed=fixed_seed,
                output_dir=phase_out,
                device=device,
                debug_fast=(
                    a.debug_fast
                ),
            )

            mr, mf = evaluate_prediction_file(
                mod14a,
                Path(
                    row[
                        "prediction_path"
                    ]
                ),
            )

            phase_run_rows.append(
                {
                    "phase": int(
                        phase
                    ),
                    **row,
                    "Ridge_vector_RMSE_mps": (
                        mr[
                            "vector_RMSE_mps"
                        ]
                    ),
                    "Ridge_WS_RMSE_mps": (
                        mr[
                            "WS_RMSE_mps"
                        ]
                    ),
                    "Ridge_WD_RMSE_deg": (
                        mr[
                            "WD_RMSE_deg"
                        ]
                    ),
                    "Final_vector_RMSE_mps": (
                        mf[
                            "vector_RMSE_mps"
                        ]
                    ),
                    "Final_WS_RMSE_mps": (
                        mf[
                            "WS_RMSE_mps"
                        ]
                    ),
                    "Final_WD_RMSE_deg": (
                        mf[
                            "WD_RMSE_deg"
                        ]
                    ),
                }
            )

            log(
                f"    final U/V/vector/WS/WD = "
                f"{mf['U_RMSE_mps']:.6f} / "
                f"{mf['V_RMSE_mps']:.6f} / "
                f"{mf['vector_RMSE_mps']:.6f} / "
                f"{mf['WS_RMSE_mps']:.6f} / "
                f"{mf['WD_RMSE_deg']:.6f}"
            )

    runs = pd.DataFrame(
        phase_run_rows
    )

    runs_path = (
        out
        / "14B_phase_lomo_runs.csv"
    )

    runs.to_csv(
        runs_path,
        index=False,
        encoding="utf-8-sig",
    )

    if a.debug_fast:
        log("")
        log(
            "[DEBUG FAST] PASS. Tropical Atlantic was NOT loaded."
        )
        return 0

    # -----------------------------------------------------------------
    # Phase summary and selection.
    # -----------------------------------------------------------------
    summary = (
        runs.groupby(
            "phase",
            as_index=False,
        )
        .agg(
            runs=(
                "held_out_mission",
                "count",
            ),
            U_RMSE_mean=(
                "U_Final",
                "mean",
            ),
            V_RMSE_mean=(
                "V_Final",
                "mean",
            ),
            vector_RMSE_mean=(
                "Final_vector_RMSE_mps",
                "mean",
            ),
            WS_RMSE_mean=(
                "Final_WS_RMSE_mps",
                "mean",
            ),
            WD_RMSE_mean=(
                "Final_WD_RMSE_deg",
                "mean",
            ),
            Ridge_vector_RMSE_mean=(
                "Ridge_vector_RMSE_mps",
                "mean",
            ),
            U_improve_mean=(
                "U_improve_fraction",
                "mean",
            ),
            V_improve_mean=(
                "V_improve_fraction",
                "mean",
            ),
            corr_U_mean=(
                "residual_corr_U",
                "mean",
            ),
            corr_V_mean=(
                "residual_corr_V",
                "mean",
            ),
            best_epoch_median=(
                "best_epoch",
                "median",
            ),
        )
        .sort_values(
            [
                "vector_RMSE_mean",
                "WS_RMSE_mean",
                "WD_RMSE_mean",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    summary[
        "rank"
    ] = np.arange(
        1,
        len(summary) + 1,
    )

    summary_path = (
        out
        / "14B_phase_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log("=" * 140)
    log(
        "14B DEVELOPMENT PHASE SUMMARY"
    )
    log("=" * 140)
    log(
        summary.to_string(
            index=False
        )
    )

    selected_phase = int(
        summary.iloc[
            0
        ][
            "phase"
        ]
    )

    selected_runs = runs.loc[
        runs[
            "phase"
        ]
        == selected_phase
    ].copy()

    final_epoch = int(
        max(
            0,
            round(
                float(
                    np.median(
                        selected_runs[
                            "best_epoch"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                )
            ),
        )
    )

    component_freeze = (
        mod14a.pooled_component_freeze(
            selected_runs
        )
    )

    active = component_freeze[
        "active"
    ]

    frozen_alphas = (
        component_freeze[
            "frozen_alphas"
        ]
    )

    component_evidence_path = (
        out
        / "14B_selected_phase_component_evidence.csv"
    )

    component_freeze[
        "mission_table"
    ].to_csv(
        component_evidence_path,
        index=False,
        encoding="utf-8-sig",
    )

    frozen = {
        "stage": "14B",
        "script_version": (
            SCRIPT_VERSION
        ),
        "base_model": (
            "Frozen Stage14A architecture/training recipe"
        ),
        "phase_selected_from": (
            "Antarctic + Atlantic + West Coast LOMO only"
        ),
        "phase_selection_metric": (
            "lowest mean LOMO final vector RMSE; "
            "tie-break WS then WD"
        ),
        "selected_phase": int(
            selected_phase
        ),
        "fixed_seed_from_14A": int(
            fixed_seed
        ),
        "final_epoch": int(
            final_epoch
        ),
        "active_U": bool(
            active[
                0
            ]
        ),
        "active_V": bool(
            active[
                1
            ]
        ),
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
        "component_info": (
            component_freeze[
                "component_info"
            ]
        ),
        "development_phase_ranking": (
            summary.to_dict(
                orient="records"
            )
        ),
        "Tropical_Atlantic_loaded_before_phase_selection": False,
        "Tropical_Atlantic_used_for_phase_selection": False,
        "Tropical_Atlantic_used_for_alpha": False,
    }

    frozen_path = (
        out
        / "FROZEN_PHASE_BEFORE_TROPICAL.json"
    )

    save_json(
        frozen_path,
        frozen,
    )

    log("")
    log(
        f"[SELECTED PHASE] {selected_phase}"
    )

    log(
        f"[FROZEN BEFORE TROPICAL] "
        f"phase={selected_phase} | "
        f"seed={fixed_seed} | "
        f"epochs={final_epoch} | "
        f"active U/V={bool(active[0])}/{bool(active[1])} | "
        f"alpha U/V={frozen_alphas[0]:.6f}/{frozen_alphas[1]:.6f}"
    )

    log(
        f"[SAVED] {frozen_path}"
    )

    # -----------------------------------------------------------------
    # Final training on all development missions at selected phase.
    # -----------------------------------------------------------------
    selected_dev = (
        phase_phase_data[
            selected_phase
        ]
    )

    dev_all = mod14a.concat_missions(
        [
            selected_dev[
                m
            ]
            for m in DEV
        ]
    )

    final_bundle = mod14a.train_final_network(
        torch=torch,
        nn=nn,
        dev_all=dev_all,
        final_seed=fixed_seed,
        final_epochs=final_epoch,
        device=device,
    )

    model_dir = (
        out
        / "final_model"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        model_dir
        / "14B_selected_phase_final.pt"
    )

    torch.save(
        {
            "stage": (
                "14B-selected-phase-final"
            ),
            "selected_phase": int(
                selected_phase
            ),
            "seed": int(
                fixed_seed
            ),
            "final_epoch": int(
                final_epoch
            ),
            "active_components": (
                active.astype(
                    bool
                )
            ),
            "frozen_alphas": (
                frozen_alphas.astype(
                    np.float32
                )
            ),
            "residual_state_dict": (
                mod14a.state_dict_cpu(
                    final_bundle[
                        "residual_model"
                    ]
                )
            ),
            "residual_scalers": (
                final_bundle[
                    "residual_scalers"
                ]
            ),
        },
        checkpoint_path,
    )

    # -----------------------------------------------------------------
    # NOW load Tropical Atlantic raw.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE C] Phase/model/alpha frozen. Loading Tropical Atlantic NOW."
    )

    trop_raw_dict, trop_audit = (
        load_raw_subset(
            a.project_root,
            a.raw_dir,
            [
                TEST
            ],
            a.progress_every,
        )
    )

    trop_audit.to_csv(
        out
        / "14B_Tropical_raw_load_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tropical = build_phase_mission(
        mod13c,
        trop_raw_dict[
            TEST
        ],
        TEST,
        selected_phase,
    )

    # -----------------------------------------------------------------
    # Final inference.
    # -----------------------------------------------------------------
    Xtrop_base = mod14a.build_base4(
        tropical[
            "X_full9"
        ]
    )

    ridge_trop = mod14a.ridge_predict(
        final_bundle[
            "ridge_model"
        ],
        final_bundle[
            "ridge_scaler"
        ],
        Xtrop_base,
    )

    feat_trop = (
        mod14a.build_residual_features(
            tropical[
                "X_full9"
            ],
            ridge_trop,
        )
    )

    transformed_trop = (
        mod14a.transform_residual_features(
            feat_trop,
            final_bundle[
                "residual_scalers"
            ],
        )
    )

    amp_enabled = bool(
        mod14a.FREEZE.use_amp
        and device.type
        == "cuda"
    )

    raw_trop = (
        mod14a.predict_raw_residual(
            torch=torch,
            model=final_bundle[
                "residual_model"
            ],
            transformed=transformed_trop,
            residual_std=final_bundle[
                "residual_scalers"
            ][
                "residual_std"
            ],
            device=device,
            amp_enabled=amp_enabled,
        )
    )

    final_trop = mod14a.apply_alphas(
        ridge_trop,
        raw_trop,
        frozen_alphas,
    )

    ridge_m = mod14a.evaluate_wind(
        tropical[
            "y_uv"
        ],
        ridge_trop,
    )

    final_m = mod14a.evaluate_wind(
        tropical[
            "y_uv"
        ],
        final_trop,
    )

    true_res = (
        tropical[
            "y_uv"
        ]
        - ridge_trop
    )

    corr_u = mod14a.corrcoef_safe(
        true_res[
            :,
            0
        ],
        raw_trop[
            :,
            0
        ],
    )

    corr_v = mod14a.corrcoef_safe(
        true_res[
            :,
            1
        ],
        raw_trop[
            :,
            1
        ],
    )

    total_params = int(
        final_bundle[
            "parameter_count_nn"
        ]
        + final_bundle[
            "parameter_count_ridge"
        ]
    )

    final_table = pd.DataFrame(
        [
            {
                "model": (
                    "BP-STGNN-published"
                ),
                "phase": np.nan,
                "U_RMSE_mps": (
                    BP[
                        "U_RMSE_mps"
                    ]
                ),
                "V_RMSE_mps": (
                    BP[
                        "V_RMSE_mps"
                    ]
                ),
                "vector_RMSE_mps": np.nan,
                "WS_RMSE_mps": (
                    BP[
                        "WS_RMSE_mps"
                    ]
                ),
                "WD_RMSE_deg": (
                    BP[
                        "WD_RMSE_deg"
                    ]
                ),
                "parameter_count": int(
                    BP[
                        "params"
                    ]
                ),
            },
            {
                "model": (
                    "SelectedPhase-Base4-Ridge"
                ),
                "phase": int(
                    selected_phase
                ),
                **ridge_m,
                "parameter_count": int(
                    final_bundle[
                        "parameter_count_ridge"
                    ]
                ),
            },
            {
                "model": (
                    "SelectedPhase-14A-ComponentSafeResidual"
                ),
                "phase": int(
                    selected_phase
                ),
                **final_m,
                "parameter_count": int(
                    total_params
                ),
            },
        ]
    )

    final_path = (
        out
        / "FINAL_TROPICAL_14B_SELECTED_PHASE.csv"
    )

    final_table.to_csv(
        final_path,
        index=False,
        encoding="utf-8-sig",
    )

    improvement = {
        key: mod14a.rel_improve(
            ridge_m[
                metric
            ],
            final_m[
                metric
            ],
        )
        for key, metric in [
            (
                "U_fraction",
                "U_RMSE_mps",
            ),
            (
                "V_fraction",
                "V_RMSE_mps",
            ),
            (
                "vector_fraction",
                "vector_RMSE_mps",
            ),
            (
                "WS_fraction",
                "WS_RMSE_mps",
            ),
            (
                "WD_fraction",
                "WD_RMSE_deg",
            ),
        ]
    }

    np.savez_compressed(
        out
        / "FINAL_TROPICAL_14B_PREDICTIONS.npz",
        y_true=tropical[
            "y_uv"
        ].astype(
            np.float32
        ),
        ridge_pred=ridge_trop.astype(
            np.float32
        ),
        raw_residual_pred=raw_trop.astype(
            np.float32
        ),
        final_pred=final_trop.astype(
            np.float32
        ),
        selected_phase=np.asarray(
            [
                selected_phase
            ],
            dtype=np.int64,
        ),
        frozen_alphas=frozen_alphas.astype(
            np.float32
        ),
    )

    report = {
        "stage": "14B",
        "selected_phase": int(
            selected_phase
        ),
        "selection_is_development_only": True,
        "fixed_seed": int(
            fixed_seed
        ),
        "final_epoch": int(
            final_epoch
        ),
        "active_U": bool(
            active[
                0
            ]
        ),
        "active_V": bool(
            active[
                1
            ]
        ),
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
        "Tropical_raw_residual_corr_U": float(
            corr_u
        ),
        "Tropical_raw_residual_corr_V": float(
            corr_v
        ),
        "ridge_metrics": ridge_m,
        "final_metrics": final_m,
        "improvement_vs_selected_phase_ridge": (
            improvement
        ),
        "phase_summary": (
            summary.to_dict(
                orient="records"
            )
        ),
    }

    save_json(
        out
        / "14B_REPORT.json",
        report,
    )

    with (
        out
        / "14B_REPORT.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "14B DEVELOPMENT-ONLY PHASE SELECTION FOR FROZEN 14A\n"
        )

        f.write(
            "=" * 120
            + "\n\n"
        )

        f.write(
            "DEVELOPMENT PHASE SUMMARY\n"
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
            "\n\nFROZEN PHASE\n"
        )

        f.write(
            "-" * 120
            + "\n"
        )

        f.write(
            f"selected phase = {selected_phase}\n"
        )

        f.write(
            f"seed = {fixed_seed}\n"
        )

        f.write(
            f"final epoch = {final_epoch}\n"
        )

        f.write(
            f"active U/V = {bool(active[0])}/{bool(active[1])}\n"
        )

        f.write(
            f"alpha U/V = {frozen_alphas[0]:.8f}/{frozen_alphas[1]:.8f}\n"
        )

        f.write(
            "\n\nFINAL TROPICAL\n"
        )

        f.write(
            "-" * 120
            + "\n"
        )

        f.write(
            final_table.to_string(
                index=False
            )
        )

        f.write(
            "\n\nIMPROVEMENT VS SELECTED-PHASE RIDGE\n"
        )

        f.write(
            "-" * 120
            + "\n"
        )

        for k, v in improvement.items():
            f.write(
                f"{k}: {100*v:+.6f}%\n"
            )

        f.write(
            "\n"
            "NOTE: phase was selected using development missions only; "
            "Tropical Atlantic was loaded after the phase/model/alpha freeze.\n"
        )

    log("")
    log("=" * 140)
    log(
        "FINAL TROPICAL — DEVELOPMENT-SELECTED PHASE 14A"
    )
    log("=" * 140)

    log(
        final_table.to_string(
            index=False
        )
    )

    log("")
    log(
        f"Selected phase = {selected_phase}"
    )

    log(
        f"Frozen active U/V = "
        f"{bool(active[0])}/{bool(active[1])}"
    )

    log(
        f"Frozen alpha U/V = "
        f"{frozen_alphas[0]:.6f}/{frozen_alphas[1]:.6f}"
    )

    log(
        f"Tropical residual corr U/V = "
        f"{corr_u:+.4f}/{corr_v:+.4f}"
    )

    log("")
    log(
        "Improvement vs selected-phase Ridge:"
    )

    for k, v in improvement.items():
        log(
            f"  {k:<16}: {100*v:+.3f}%"
        )

    log("")
    log(
        f"[SAVED] {summary_path}"
    )

    log(
        f"[SAVED] {frozen_path}"
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
        sys.exit(1)
