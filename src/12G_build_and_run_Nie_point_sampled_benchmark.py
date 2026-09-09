# -*- coding: utf-8 -*-
r"""
12G_build_and_run_Nie_point_sampled_benchmark.py

Stage 12G
=========
Build and run a FIXED-PHASE, POINT-SAMPLED 10-min literature-aligned benchmark
for the Saildrone missions used by Nie et al., using the already frozen
Physics-Compact-NieData v2 architecture/training code.

Scientific purpose
------------------
Stage 12F showed that the exact definition of "10-min resampling" changes the
difficulty of the Tropical Atlantic task substantially. The existing Stage-12B
benchmark uses arithmetic means over each complete 10-min interval.

Stage 12G therefore constructs a second, explicitly different protocol:

    UTC phase-0 point sampling:
        keep the 1-min record at 00, 10, 20, 30, 40, 50 min of each hour

    historical context:
        six consecutive point samples = 60 min wall-clock context

    target:
        the immediately following point sample = +10 min

Thus the wind task is:

    [x(t-50), x(t-40), ..., x(t)]
        -> x(t+10)

where every x is one selected 1-min Saildrone record, NOT a 10-min block mean.

IMPORTANT:
- This script does NOT claim that Nie et al. definitely used decimation.
- The paper does not specify the exact 10-min aggregation operator sufficiently
  to establish that.
- This is a transparent, paper-informed POINT-SAMPLED sensitivity benchmark.
- Tropical Atlantic has already been used in Stage 12F for protocol audit.
  Therefore Stage 12G must NOT be described as a pristine never-seen test.
  It is a secondary literature-aligned sensitivity benchmark.
- The point-sampling phase is HARD-FROZEN to UTC phase 0 in this script.
  No phase search/selection is performed.

Why reuse Stage 12C2 / 12D?
---------------------------
The Physics-Compact architecture should not be redesigned after seeing the
Tropical Atlantic protocol audit. Stage 12G therefore:

1. builds a new point-sampled dataset with the SAME Track-J schema:
       X: [N,6,9]
       [U,V,T,RH,SOG,COG_sin,COG_cos,WING_ANGLE_sin,WING_ANGLE_cos]

       y: [N,1,4]
       [U,V,VESSEL_E,VESSEL_N]

2. runs the existing Stage-12C2 development-only LOMO script on:
       Antarctic + Atlantic + West Coast

3. runs the existing Stage-12D final-fit/evaluation script on:
       Tropical Atlantic

The following remain unchanged:
- Persistence anchor
- GRU 64 / 1 layer
- Dense 64
- dropout 0.10
- lambda_AW = 2.0
- lambda_res = 0.01
- Adam 1e-3
- batch 256
- checkpoint rule = validation AW vector RMSE
- screening/final seeds
- final epoch-plan derivation

No architecture or hyperparameter is selected from Tropical Atlantic.

Point eligibility
-----------------
A 10-min node is accepted when:
1. its timestamp is exactly UTC phase 0 modulo 10 min;
2. all common raw variables at that selected 1-min record are finite:
       U,V,SOG,COG,WING_ANGLE,T,RH,P

No interpolation is performed.

For aligned Track-P / Track-J samples:
- selected point nodes must be exactly 10 min apart;
- six context nodes + the immediately following target node are required;
- Track-P differential features use the previous contiguous selected point:
      dT  = T[k]-T[k-1]
      dP  = P[k]-P[k-1]
      dRH = RH[k]-RH[k-1]
  so the first context node must also have a contiguous predecessor.

This keeps Track P and Track J sample timestamps exactly aligned.

Vessel state
------------
At each selected point:
    VESSEL_E = SOG * sin(COG)
    VESSEL_N = SOG * cos(COG)

This differs intentionally from Stage 12B, where vessel components were averaged
over all ten 1-min observations within each interval.

Outputs
-------
data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\
    dataset\
        sampled_points\
            Antarctic_point10min_phase0.csv
            Atlantic_point10min_phase0.csv
            West_Coast_point10min_phase0.csv
            Tropical_Atlantic_point10min_phase0.csv

        track_P_paper_informed\
            Antarctic.npz
            Atlantic.npz
            West_Coast.npz
            Tropical_Atlantic_TEST.npz
            development_all.npz

        track_J_joint_compatible\
            Antarctic.npz
            Atlantic.npz
            West_Coast.npz
            Tropical_Atlantic_TEST.npz
            development_all.npz

        mission_build_audit.csv
        sample_alignment_audit.csv
        feature_target_schema.json
        build_manifest.json
        build_report.txt

    lomo\
        (native Stage-12C2 outputs)

    final\
        (native Stage-12D outputs)

    12G_point_sampled_final_comparison.csv
    12G_POINT_FINAL_REPORT.txt
    12G_manifest.json

Recommended execution
---------------------
A. Dataset build only:
    python "...\12G_build_and_run_Nie_point_sampled_benchmark.py" --build-only

B. Full development LOMO + final evaluation:
    python "...\12G_build_and_run_Nie_point_sampled_benchmark.py"

C. Debug LOMO only (NON-SCIENTIFIC):
    python "...\12G_build_and_run_Nie_point_sampled_benchmark.py" --debug-fast --skip-final

Notes
-----
- The full run can take substantial GPU time because it executes the complete
  Stage-12C2 LOMO matrix and then Stage-12D final fitting.
- Existing outputs are reused when possible. Use --rebuild-dataset if the point
  dataset genuinely needs rebuilding.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0-Nie-point-sampled-phase0-orchestrator"

DEFAULT_PROJECT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")
DEFAULT_RAW_DIR = DEFAULT_PROJECT_ROOT / "data" / "raw" / "RawData"
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12G_Nie_point_sampled_benchmark_v0_1"
)

DEVELOPMENT_MISSIONS = ["Antarctic", "Atlantic", "West Coast"]
TEST_MISSION = "Tropical Atlantic"
ALL_MISSIONS = DEVELOPMENT_MISSIONS + [TEST_MISSION]

# HARD-FROZEN. Do not select phase after Tropical Atlantic performance.
POINT_PHASE_MINUTE = 0

LOOKBACK_STEPS = 6
STEP_MINUTES = 10
FORECAST_STEPS = 1

ONE_MIN_NS = 60 * 1_000_000_000
TEN_MIN_NS = 10 * ONE_MIN_NS

COMMON_BASE = [
    "U",
    "V",
    "SOG",
    "COG",
    "WING_ANGLE",
    "T",
    "RH",
    "P",
]

TRACK_P_FEATURES = [
    "U",
    "V",
    "SOG",
    "COG",
    "T",
    "RH",
    "P",
    "dT",
    "dP",
    "dRH",
]

TRACK_J_FEATURES = [
    "U",
    "V",
    "T",
    "RH",
    "SOG",
    "COG_sin",
    "COG_cos",
    "WING_ANGLE_sin",
    "WING_ANGLE_cos",
]

TRACK_P_TARGETS = ["U", "V"]
TRACK_J_TARGETS = ["U", "V", "VESSEL_E", "VESSEL_N"]
APPARENT_TARGETS = ["AW_E", "AW_N"]

NIE_BP_STGNN_REFERENCE = {
    "wind_U_RMSE_mps": 0.74864,
    "wind_V_RMSE_mps": 0.70506,
    "wind_speed_RMSE_mps": 0.69940,
    "wind_direction_RMSE_deg": 5.58400,
}

PHASE0_12F_PERSISTENCE_REFERENCE = {
    "wind_U_RMSE_mps": 0.777875,
    "wind_V_RMSE_mps": 0.874119,
    "wind_speed_RMSE_mps": 0.739705,
    "wind_direction_RMSE_deg": 6.607158,
}


def log(message=""):
    print(message, flush=True)


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, obj):
    def cv(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return None if np.isnan(v) else float(v)
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


def load_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def find_stage12b(project_root: Path):
    path = project_root / "src" / "12B_build_Nie_same_mission_benchmark_dataset.py"
    if not path.exists():
        raise FileNotFoundError(
            "Stage-12B builder is required because Stage 12G reuses its exact "
            f"mission-discovery/raw-variable logic:\n  {path}"
        )
    return path


def find_stage12c2(project_root: Path):
    src = project_root / "src"

    candidates = [
        src / "12C2_train_PhysicsCompact_NieData_v2_PersistenceAnchor_LOMO.py",
        src / "12C2_train_PhysicsCompact_NieData_v2_PersistenceAnchor_LOMO_v0_1_2.py",
    ]

    for path in candidates:
        if path.exists():
            return path

    globbed = sorted(
        src.glob("12C2_train_PhysicsCompact_NieData_v2_PersistenceAnchor_LOMO*.py")
    )

    if globbed:
        return globbed[-1]

    raise FileNotFoundError(
        "Could not locate the Stage-12C2 persistence-anchor LOMO script under:\n"
        f"  {src}"
    )


def find_stage12d(project_root: Path):
    path = project_root / "src" / "12D_final_PhysicsCompact_NieData_v2_TropicalAtlantic.py"

    if not path.exists():
        raise FileNotFoundError(
            "Stage-12D final evaluation script not found:\n"
            f"  {path}"
        )

    return path


def ns_iso(v: int):
    return str(np.datetime64(int(v), "ns"))


def wind_direction_from_uv_deg(u, v):
    return float(np.rad2deg(np.arctan2(-float(u), -float(v))) % 360.0)


def circular_deg_to_sin_cos(deg):
    rad = np.deg2rad(np.asarray(deg, dtype=np.float64))
    return np.sin(rad), np.cos(rad)


def build_point_table(raw: pd.DataFrame, mission: str):
    """
    Keep exactly one selected 1-min record every UTC 10 min.

    phase 0 means timestamps whose absolute minute index modulo 10 is 0.
    """
    raw = raw.sort_values("time_ns").reset_index(drop=True)

    if raw["time_ns"].duplicated().any():
        raise RuntimeError(f"{mission}: duplicate timestamps remain after Stage-12B raw load.")

    time_ns = raw["time_ns"].to_numpy(dtype=np.int64)
    minute_index = time_ns // ONE_MIN_NS

    phase_mask = (minute_index % 10) == POINT_PHASE_MINUTE

    points = raw.loc[phase_mask].copy().reset_index(drop=True)

    if points.empty:
        raise RuntimeError(f"{mission}: no UTC phase-{POINT_PHASE_MINUTE} points.")

    finite_mask = np.isfinite(points[COMMON_BASE].to_numpy(dtype=np.float64)).all(axis=1)

    raw_phase_points = int(len(points))
    rejected_nonfinite = int((~finite_mask).sum())

    points = points.loc[finite_mask].copy().reset_index(drop=True)

    # Derived point-state quantities.
    cog_rad = np.deg2rad(points["COG"].to_numpy(dtype=np.float64))
    wing_rad = np.deg2rad(points["WING_ANGLE"].to_numpy(dtype=np.float64))

    points["COG_sin"] = np.sin(cog_rad)
    points["COG_cos"] = np.cos(cog_rad)
    points["WING_ANGLE_sin"] = np.sin(wing_rad)
    points["WING_ANGLE_cos"] = np.cos(wing_rad)

    points["VESSEL_E"] = points["SOG"].to_numpy(dtype=np.float64) * np.sin(cog_rad)
    points["VESSEL_N"] = points["SOG"].to_numpy(dtype=np.float64) * np.cos(cog_rad)

    points["WIND_SPEED"] = np.hypot(
        points["U"].to_numpy(dtype=np.float64),
        points["V"].to_numpy(dtype=np.float64),
    )

    points["WIND_DIR_FROM_DEG"] = np.rad2deg(
        np.arctan2(
            -points["U"].to_numpy(dtype=np.float64),
            -points["V"].to_numpy(dtype=np.float64),
        )
    ) % 360.0

    # Differential nodes are defined only across exactly contiguous 10-min points.
    t = points["time_ns"].to_numpy(dtype=np.int64)

    contiguous_prev = np.zeros(len(points), dtype=bool)
    if len(points) > 1:
        contiguous_prev[1:] = np.diff(t) == TEN_MIN_NS

    for source, diff_name in [
        ("T", "dT"),
        ("P", "dP"),
        ("RH", "dRH"),
    ]:
        values = points[source].to_numpy(dtype=np.float64)
        diff = np.full(len(points), np.nan, dtype=np.float64)

        if len(points) > 1:
            raw_diff = values[1:] - values[:-1]
            good = contiguous_prev[1:]
            temp = np.full(len(points) - 1, np.nan, dtype=np.float64)
            temp[good] = raw_diff[good]
            diff[1:] = temp

        points[diff_name] = diff

    audit = {
        "mission": mission,
        "raw_rows": int(len(raw)),
        "raw_phase0_points": raw_phase_points,
        "rejected_phase0_nonfinite_common": rejected_nonfinite,
        "accepted_point_nodes": int(len(points)),
        "point_first_time": ns_iso(int(points["time_ns"].iloc[0])),
        "point_last_time": ns_iso(int(points["time_ns"].iloc[-1])),
        "contiguous_point_transitions": int(contiguous_prev.sum()),
        "phase_minute": int(POINT_PHASE_MINUTE),
    }

    keep_columns = [
        "time_ns",
        "U",
        "V",
        "SOG",
        "COG",
        "WING_ANGLE",
        "T",
        "RH",
        "P",
        "COG_sin",
        "COG_cos",
        "WING_ANGLE_sin",
        "WING_ANGLE_cos",
        "VESSEL_E",
        "VESSEL_N",
        "WIND_SPEED",
        "WIND_DIR_FROM_DEG",
        "dT",
        "dP",
        "dRH",
    ]

    return points[keep_columns].copy(), audit


def build_aligned_samples(points: pd.DataFrame, mission: str):
    """
    Track P and J use identical accepted sample timestamps.

    For context indices a...c and target t=c+1:
    - c-a+1 = 6 context points
    - all seven context+target points are exactly 10 min apart
    - Track-P differential values for ALL six context points must be finite,
      which implicitly requires one earlier contiguous point for the first
      context point.
    """
    starts = points["time_ns"].to_numpy(dtype=np.int64)

    Xp = []
    Xj = []
    yp = []
    yj = []
    aw = []
    context_end = []
    target_time = []
    target_speed = []
    target_direction = []

    candidate_count = max(0, len(points) - LOOKBACK_STEPS)

    rejected_noncontiguous = 0
    rejected_diff = 0
    rejected_nonfinite = 0

    for c in range(LOOKBACK_STEPS - 1, len(points) - 1):
        a = c - LOOKBACK_STEPS + 1
        t_idx = c + 1

        expected = (
            starts[a]
            + np.arange(LOOKBACK_STEPS + 1, dtype=np.int64) * TEN_MIN_NS
        )
        actual = starts[a:t_idx + 1]

        if not np.array_equal(actual, expected):
            rejected_noncontiguous += 1
            continue

        context = points.iloc[a:c + 1]
        target = points.iloc[t_idx]

        # To keep Track P aligned, all differential nodes in context must exist.
        diff_values = context[["dT", "dP", "dRH"]].to_numpy(dtype=np.float64)
        if not np.isfinite(diff_values).all():
            rejected_diff += 1
            continue

        x_p = context[TRACK_P_FEATURES].to_numpy(dtype=np.float64)
        x_j = context[TRACK_J_FEATURES].to_numpy(dtype=np.float64)

        y_p = np.asarray(
            [[float(target["U"]), float(target["V"])]],
            dtype=np.float64,
        )

        y_joint = np.asarray(
            [[
                float(target["U"]),
                float(target["V"]),
                float(target["VESSEL_E"]),
                float(target["VESSEL_N"]),
            ]],
            dtype=np.float64,
        )

        aw_ref = np.asarray(
            [[
                float(target["U"] - target["VESSEL_E"]),
                float(target["V"] - target["VESSEL_N"]),
            ]],
            dtype=np.float64,
        )

        all_values = [
            x_p,
            x_j,
            y_p,
            y_joint,
            aw_ref,
        ]

        if not all(np.isfinite(v).all() for v in all_values):
            rejected_nonfinite += 1
            continue

        Xp.append(x_p)
        Xj.append(x_j)
        yp.append(y_p)
        yj.append(y_joint)
        aw.append(aw_ref)
        context_end.append(int(starts[c]))
        target_time.append([int(starts[t_idx])])
        target_speed.append([float(target["WIND_SPEED"])])
        target_direction.append([float(target["WIND_DIR_FROM_DEG"])])

    if not Xj:
        raise RuntimeError(f"{mission}: no point-sampled forecasting samples.")

    result = {
        "X_P": np.asarray(Xp, dtype=np.float32),
        "X_J": np.asarray(Xj, dtype=np.float32),
        "y_P": np.asarray(yp, dtype=np.float32),
        "y_J": np.asarray(yj, dtype=np.float32),
        "apparent_ref": np.asarray(aw, dtype=np.float32),
        "context_end_time_ns": np.asarray(context_end, dtype=np.int64),
        "target_time_ns": np.asarray(target_time, dtype=np.int64),
        "target_wind_speed": np.asarray(target_speed, dtype=np.float32),
        "target_wind_direction_from_deg": np.asarray(
            target_direction,
            dtype=np.float32,
        ),
        "mission": np.asarray([mission] * len(Xj)),
    }

    physics = result["y_J"][:, :, 0:2] - result["y_J"][:, :, 2:4]
    physics_error = float(np.max(np.abs(physics - result["apparent_ref"])))

    if physics_error > 1e-5:
        raise RuntimeError(
            f"{mission}: point AW physics audit failed, max={physics_error:.3e}"
        )

    context_ns = result["context_end_time_ns"]
    target_ns = result["target_time_ns"].reshape(-1)

    if np.any(np.diff(context_ns) <= 0):
        raise RuntimeError(f"{mission}: context timestamps are not strictly increasing.")

    if not np.all(target_ns - context_ns == TEN_MIN_NS):
        raise RuntimeError(f"{mission}: target is not exactly +10 min.")

    audit = {
        "mission": mission,
        "candidate_context_target_windows": int(candidate_count),
        "rejected_noncontiguous": int(rejected_noncontiguous),
        "rejected_missing_trackP_differential": int(rejected_diff),
        "rejected_nonfinite_final": int(rejected_nonfinite),
        "accepted_samples": int(len(Xj)),
        "X_P_shape": str(tuple(result["X_P"].shape)),
        "X_J_shape": str(tuple(result["X_J"].shape)),
        "y_P_shape": str(tuple(result["y_P"].shape)),
        "y_J_shape": str(tuple(result["y_J"].shape)),
        "physics_max_abs_error": physics_error,
        "context_first": ns_iso(int(context_ns[0])),
        "context_last": ns_iso(int(context_ns[-1])),
        "target_first": ns_iso(int(target_ns[0])),
        "target_last": ns_iso(int(target_ns[-1])),
    }

    return result, audit


def save_track_p(path: Path, samples: dict):
    np.savez_compressed(
        path,
        X_raw=samples["X_P"],
        y_wind_raw=samples["y_P"],
        context_end_time_ns=samples["context_end_time_ns"],
        target_time_ns=samples["target_time_ns"],
        target_wind_speed=samples["target_wind_speed"],
        target_wind_direction_from_deg=samples["target_wind_direction_from_deg"],
        mission=samples["mission"],
        feature_names=np.asarray(TRACK_P_FEATURES),
        target_names=np.asarray(TRACK_P_TARGETS),
        lookback_steps=np.asarray([LOOKBACK_STEPS], dtype=np.int64),
        step_minutes=np.asarray([STEP_MINUTES], dtype=np.int64),
        forecast_steps=np.asarray([FORECAST_STEPS], dtype=np.int64),
        point_sampling_phase_minute=np.asarray(
            [POINT_PHASE_MINUTE],
            dtype=np.int64,
        ),
    )


def save_track_j(path: Path, samples: dict):
    np.savez_compressed(
        path,
        X_raw=samples["X_J"],
        y_joint_raw=samples["y_J"],
        y_wind_raw=samples["y_J"][:, :, 0:2],
        y_vessel_raw=samples["y_J"][:, :, 2:4],
        apparent_ref=samples["apparent_ref"],
        context_end_time_ns=samples["context_end_time_ns"],
        target_time_ns=samples["target_time_ns"],
        target_wind_speed=samples["target_wind_speed"],
        target_wind_direction_from_deg=samples["target_wind_direction_from_deg"],
        mission=samples["mission"],
        feature_names=np.asarray(TRACK_J_FEATURES),
        target_names=np.asarray(TRACK_J_TARGETS),
        apparent_names=np.asarray(APPARENT_TARGETS),
        lookback_steps=np.asarray([LOOKBACK_STEPS], dtype=np.int64),
        step_minutes=np.asarray([STEP_MINUTES], dtype=np.int64),
        forecast_steps=np.asarray([FORECAST_STEPS], dtype=np.int64),
        point_sampling_phase_minute=np.asarray(
            [POINT_PHASE_MINUTE],
            dtype=np.int64,
        ),
    )


def concatenate_samples(items):
    keys = [
        "X_P",
        "X_J",
        "y_P",
        "y_J",
        "apparent_ref",
        "context_end_time_ns",
        "target_time_ns",
        "target_wind_speed",
        "target_wind_direction_from_deg",
        "mission",
    ]

    return {
        key: np.concatenate([item[key] for item in items], axis=0)
        for key in keys
    }


def validate_track_j_file(path: Path, expected_mission=None):
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as z:
        required = [
            "X_raw",
            "y_joint_raw",
            "apparent_ref",
            "context_end_time_ns",
            "target_time_ns",
            "feature_names",
            "target_names",
            "apparent_names",
        ]

        missing = [k for k in required if k not in z.files]

        if missing:
            raise RuntimeError(f"{path}: missing keys {missing}")

        X = np.asarray(z["X_raw"], dtype=np.float32)
        y = np.asarray(z["y_joint_raw"], dtype=np.float32)
        aw = np.asarray(z["apparent_ref"], dtype=np.float32)
        context = np.asarray(z["context_end_time_ns"], dtype=np.int64).reshape(-1)
        target = np.asarray(z["target_time_ns"], dtype=np.int64).reshape(-1)

        features = [str(x) for x in z["feature_names"].tolist()]
        targets = [str(x) for x in z["target_names"].tolist()]
        apparent_names = [str(x) for x in z["apparent_names"].tolist()]

    if X.ndim != 3 or X.shape[1:] != (6, 9):
        raise RuntimeError(f"{path.name}: expected X [N,6,9], got {X.shape}")

    if y.ndim != 3 or y.shape[1:] != (1, 4):
        raise RuntimeError(f"{path.name}: expected y [N,1,4], got {y.shape}")

    if aw.ndim != 3 or aw.shape[1:] != (1, 2):
        raise RuntimeError(f"{path.name}: expected AW [N,1,2], got {aw.shape}")

    if features != TRACK_J_FEATURES:
        raise RuntimeError(f"{path.name}: feature schema mismatch {features}")

    if targets != TRACK_J_TARGETS:
        raise RuntimeError(f"{path.name}: target schema mismatch {targets}")

    if apparent_names != APPARENT_TARGETS:
        raise RuntimeError(f"{path.name}: apparent schema mismatch {apparent_names}")

    if not (
        np.isfinite(X).all()
        and np.isfinite(y).all()
        and np.isfinite(aw).all()
    ):
        raise RuntimeError(f"{path.name}: non-finite values.")

    if not np.all(target - context == TEN_MIN_NS):
        raise RuntimeError(f"{path.name}: target is not +10 min.")

    physics = y[:, :, 0:2] - y[:, :, 2:4]
    physics_max = float(np.max(np.abs(physics - aw)))

    return {
        "path": str(path),
        "samples": int(len(X)),
        "X_shape": tuple(int(v) for v in X.shape),
        "y_shape": tuple(int(v) for v in y.shape),
        "physics_max_abs_error": physics_max,
        "context_first": ns_iso(int(context[0])),
        "context_last": ns_iso(int(context[-1])),
        "target_first": ns_iso(int(target[0])),
        "target_last": ns_iso(int(target[-1])),
    }


def dataset_complete(dataset_dir: Path):
    required = [
        dataset_dir / "track_J_joint_compatible" / "Antarctic.npz",
        dataset_dir / "track_J_joint_compatible" / "Atlantic.npz",
        dataset_dir / "track_J_joint_compatible" / "West_Coast.npz",
        dataset_dir / "track_J_joint_compatible" / "Tropical_Atlantic_TEST.npz",
        dataset_dir / "track_J_joint_compatible" / "development_all.npz",
        dataset_dir / "feature_target_schema.json",
        dataset_dir / "build_manifest.json",
    ]

    return all(p.exists() for p in required)


def build_dataset(
    project_root: Path,
    raw_dir: Path,
    dataset_dir: Path,
    progress_every: int,
):
    stage12b_path = find_stage12b(project_root)
    stage12b = load_module(stage12b_path, "stage12b_point_builder")

    xr = stage12b.import_xarray()
    grouped = stage12b.discover_mission_files(raw_dir)

    sampled_dir = dataset_dir / "sampled_points"
    p_dir = dataset_dir / "track_P_paper_informed"
    j_dir = dataset_dir / "track_J_joint_compatible"

    for d in [sampled_dir, p_dir, j_dir]:
        d.mkdir(parents=True, exist_ok=True)

    point_audits = []
    sample_audits = []
    samples_by_mission = {}

    log("=" * 120)
    log("12G DATASET BUILD — FIXED UTC PHASE-0 POINT SAMPLING")
    log("=" * 120)
    log(f"raw dir       : {raw_dir}")
    log(f"dataset dir   : {dataset_dir}")
    log(f"sampling      : one 1-min record every 10 min, UTC phase={POINT_PHASE_MINUTE}")
    log("interpolation : NONE")
    log("")

    for mission in ALL_MISSIONS:
        paths = grouped[mission]

        log(f"[BUILD] {mission} | files={len(paths)}")

        raw, raw_audit, file_rows = stage12b.load_mission_raw(
            xr,
            mission,
            paths,
            progress_every,
        )

        points, point_audit = build_point_table(raw, mission)
        samples, sample_audit = build_aligned_samples(points, mission)

        samples_by_mission[mission] = samples

        safe = mission.replace(" ", "_")

        points.to_csv(
            sampled_dir / f"{safe}_point10min_phase0.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if mission == TEST_MISSION:
            p_path = p_dir / "Tropical_Atlantic_TEST.npz"
            j_path = j_dir / "Tropical_Atlantic_TEST.npz"
        else:
            p_path = p_dir / f"{safe}.npz"
            j_path = j_dir / f"{safe}.npz"

        save_track_p(p_path, samples)
        save_track_j(j_path, samples)

        combined_audit = {
            **point_audit,
            **sample_audit,
            "source_file_count": int(len(paths)),
            "raw_rows_after_dedup": int(raw_audit["rows_after_dedup"]),
        }

        point_audits.append(point_audit)
        sample_audits.append(combined_audit)

        log(
            f"  raw={len(raw):,} | selected finite phase0 nodes={len(points):,} | "
            f"samples={len(samples['X_J']):,} | physics={sample_audit['physics_max_abs_error']:.2e}"
        )

    # Development-all concatenation: explicit first-three missions only.
    dev = concatenate_samples(
        [samples_by_mission[m] for m in DEVELOPMENT_MISSIONS]
    )

    save_track_p(p_dir / "development_all.npz", dev)
    save_track_j(j_dir / "development_all.npz", dev)

    # Audit P/J alignment.
    alignment_rows = []

    for mission in ALL_MISSIONS:
        safe = mission.replace(" ", "_")

        if mission == TEST_MISSION:
            p_path = p_dir / "Tropical_Atlantic_TEST.npz"
            j_path = j_dir / "Tropical_Atlantic_TEST.npz"
        else:
            p_path = p_dir / f"{safe}.npz"
            j_path = j_dir / f"{safe}.npz"

        with np.load(p_path, allow_pickle=False) as p, np.load(
            j_path,
            allow_pickle=False,
        ) as j:
            p_c = np.asarray(p["context_end_time_ns"])
            j_c = np.asarray(j["context_end_time_ns"])

            p_t = np.asarray(p["target_time_ns"])
            j_t = np.asarray(j["target_time_ns"])

            alignment_rows.append({
                "mission": mission,
                "samples": int(len(j_c)),
                "context_aligned": bool(np.array_equal(p_c, j_c)),
                "target_aligned": bool(np.array_equal(p_t, j_t)),
                "TrackP_X_shape": str(tuple(p["X_raw"].shape)),
                "TrackJ_X_shape": str(tuple(j["X_raw"].shape)),
            })

    mission_audit_df = pd.DataFrame(sample_audits)
    alignment_df = pd.DataFrame(alignment_rows)

    mission_audit_df.to_csv(
        dataset_dir / "mission_build_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    alignment_df.to_csv(
        dataset_dir / "sample_alignment_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    schema = {
        "protocol_name": "fixed_UTC_phase0_point_sampling",
        "scientific_role": (
            "secondary literature-aligned point-sampled sensitivity benchmark; "
            "NOT exact BP-STGNN reproduction"
        ),
        "point_definition": (
            "one selected 1-min record at UTC minute 00/10/20/30/40/50"
        ),
        "phase_minute": POINT_PHASE_MINUTE,
        "lookback_steps": LOOKBACK_STEPS,
        "step_minutes": STEP_MINUTES,
        "forecast_steps": FORECAST_STEPS,
        "Track_P_features": TRACK_P_FEATURES,
        "Track_P_targets": TRACK_P_TARGETS,
        "Track_J_features": TRACK_J_FEATURES,
        "Track_J_targets": TRACK_J_TARGETS,
        "apparent_targets": APPARENT_TARGETS,
        "development_missions": DEVELOPMENT_MISSIONS,
        "literature_test_mission": TEST_MISSION,
        "interpolation": False,
        "test_status_caveat": (
            "Tropical Atlantic was previously used in Stage 12F for protocol "
            "resampling audit; Stage 12G is not a pristine never-seen final test."
        ),
    }

    save_json(dataset_dir / "feature_target_schema.json", schema)

    build_manifest = {
        "stage": "12G_dataset_build",
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "project_root": str(project_root),
        "raw_dir": str(raw_dir),
        "dataset_dir": str(dataset_dir),
        "stage12b_raw_loader": str(stage12b_path),
        "phase_minute": POINT_PHASE_MINUTE,
        "no_phase_search": True,
        "no_interpolation": True,
        "development_samples": int(len(dev["X_J"])),
        "mission_samples": {
            m: int(len(samples_by_mission[m]["X_J"]))
            for m in ALL_MISSIONS
        },
        "published_BP_STGNN_reference": NIE_BP_STGNN_REFERENCE,
        "stage12F_phase0_persistence_reference": PHASE0_12F_PERSISTENCE_REFERENCE,
        "scientific_caveat": (
            "Point sampling is a paper-informed sensitivity protocol, not proof "
            "of Nie et al.'s exact resampling implementation."
        ),
    }

    save_json(dataset_dir / "build_manifest.json", build_manifest)

    # Validate exactly the schema that 12C2/12D require.
    validation_rows = []

    for mission in DEVELOPMENT_MISSIONS:
        path = j_dir / f"{mission.replace(' ', '_')}.npz"
        row = validate_track_j_file(path, mission)
        row["mission"] = mission
        validation_rows.append(row)

    row = validate_track_j_file(
        j_dir / "Tropical_Atlantic_TEST.npz",
        TEST_MISSION,
    )
    row["mission"] = TEST_MISSION
    validation_rows.append(row)

    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(
        dataset_dir / "trackJ_validation_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (dataset_dir / "build_report.txt").open("w", encoding="utf-8") as f:
        f.write("12G Fixed-Phase Point-Sampled Benchmark Build\n")
        f.write("=" * 118 + "\n\n")
        f.write(
            "Protocol: keep one finite 1-min record every UTC 10 min "
            "(phase 0), six context points -> next +10-min point.\n"
        )
        f.write("No interpolation. No phase search.\n\n")
        f.write(
            "IMPORTANT: This is a literature-aligned sensitivity protocol, "
            "not an exact reproduction of BP-STGNN preprocessing.\n"
        )
        f.write(
            "Tropical Atlantic was previously accessed in Stage 12F for "
            "resampling audit, so this should not be called a pristine "
            "never-seen final test.\n\n"
        )
        f.write("MISSION BUILD AUDIT\n")
        f.write("-" * 118 + "\n")
        f.write(mission_audit_df.to_string(index=False))
        f.write("\n\nTRACK P/J ALIGNMENT\n")
        f.write("-" * 118 + "\n")
        f.write(alignment_df.to_string(index=False))
        f.write("\n\nTRACK J VALIDATION\n")
        f.write("-" * 118 + "\n")
        f.write(validation_df.to_string(index=False))

    log("")
    log("[DATASET READY]")
    log(f"  development samples : {len(dev['X_J']):,}")
    log(
        f"  Tropical samples    : "
        f"{len(samples_by_mission[TEST_MISSION]['X_J']):,}"
    )
    log(f"  output              : {dataset_dir}")
    log("")

    return build_manifest


def run_subprocess(command, cwd: Path, label: str):
    log("")
    log("=" * 120)
    log(label)
    log("=" * 120)
    log("COMMAND:")
    log(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in command))
    log("")

    completed = subprocess.run(
        [str(x) for x in command],
        cwd=str(cwd),
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with return code {completed.returncode}."
        )


def summarize_final(final_dir: Path, output_root: Path):
    summary_path = final_dir / "tropical_atlantic_method_summary.csv"
    published_path = final_dir / "published_BP_STGNN_comparison.csv"

    if not summary_path.exists():
        log(
            "[SUMMARY] Native Stage-12D method summary not found yet. "
            "Final run may have been skipped."
        )
        return None

    summary = pd.read_csv(summary_path)

    rows = []

    for method in [
        "Persistence",
        "Ridge",
        "PhysicsCompactNieDataV2",
    ]:
        q = summary.loc[summary["method"] == method]

        if q.empty:
            continue

        r = q.iloc[0]

        rows.append({
            "method": method,
            "source": "12G point-sampled common protocol",
            "U_RMSE_mps": float(r["wind_U_RMSE_mps_mean"]),
            "V_RMSE_mps": float(r["wind_V_RMSE_mps_mean"]),
            "WS_RMSE_mps": float(r["wind_speed_RMSE_mps_mean"]),
            "WD_RMSE_deg": float(r["wind_direction_RMSE_deg_mean"]),
            "wind_vector_RMSE_mps": float(r["wind_vector_RMSE_mps_mean"]),
            "vessel_vector_RMSE_mps": float(r["vessel_vector_RMSE_mps_mean"]),
            "AW_vector_RMSE_mps": float(r["AW_vector_RMSE_mps_mean"]),
            "AWS_RMSE_mps": float(r["AWS_RMSE_mps_mean"]),
        })

    rows.append({
        "method": "BP-STGNN (published)",
        "source": "Nie et al. published Tropical Atlantic result",
        "U_RMSE_mps": NIE_BP_STGNN_REFERENCE["wind_U_RMSE_mps"],
        "V_RMSE_mps": NIE_BP_STGNN_REFERENCE["wind_V_RMSE_mps"],
        "WS_RMSE_mps": NIE_BP_STGNN_REFERENCE["wind_speed_RMSE_mps"],
        "WD_RMSE_deg": NIE_BP_STGNN_REFERENCE["wind_direction_RMSE_deg"],
        "wind_vector_RMSE_mps": np.nan,
        "vessel_vector_RMSE_mps": np.nan,
        "AW_vector_RMSE_mps": np.nan,
        "AWS_RMSE_mps": np.nan,
    })

    comparison = pd.DataFrame(rows)

    comparison.to_csv(
        output_root / "12G_point_sampled_final_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    proposed = comparison.loc[
        comparison["method"] == "PhysicsCompactNieDataV2"
    ]

    with (output_root / "12G_POINT_FINAL_REPORT.txt").open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("12G Point-Sampled Tropical Atlantic Literature-Aligned Benchmark\n")
        f.write("=" * 118 + "\n\n")
        f.write(
            "Protocol: fixed UTC phase-0 decimation of the 1-min Saildrone "
            "records; six 10-min-spaced context points predict the next "
            "10-min-spaced point.\n"
        )
        f.write(
            "This is NOT claimed to be the exact unpublished BP-STGNN "
            "resampling implementation.\n"
        )
        f.write(
            "Tropical Atlantic was previously used in Stage 12F for protocol "
            "audit; therefore this is a secondary sensitivity benchmark, not "
            "a pristine untouched test.\n\n"
        )
        f.write("FINAL POINT-SAMPLED RESULTS\n")
        f.write("-" * 118 + "\n")
        f.write(comparison.to_string(index=False))
        f.write("\n\n")

        if not proposed.empty:
            p = proposed.iloc[0]
            f.write("PHYSICS-COMPACT v2 POINT-SAMPLED HEADLINE\n")
            f.write("-" * 118 + "\n")
            f.write(
                f"U RMSE  = {p['U_RMSE_mps']:.5f} m/s\n"
                f"V RMSE  = {p['V_RMSE_mps']:.5f} m/s\n"
                f"WS RMSE = {p['WS_RMSE_mps']:.5f} m/s\n"
                f"WD RMSE = {p['WD_RMSE_deg']:.3f} deg\n"
                f"Vessel vector RMSE = {p['vessel_vector_RMSE_mps']:.5f} m/s\n"
                f"AW vector RMSE     = {p['AW_vector_RMSE_mps']:.5f} m/s\n"
            )

        f.write("\nINTERPRETATION POLICY\n")
        f.write("-" * 118 + "\n")
        f.write(
            "Do not report an exact percentage improvement over BP-STGNN "
            "unless the published resampling/operator details can be shown to "
            "match this phase-0 protocol. Numerical side-by-side context is "
            "acceptable with the protocol caveat.\n"
        )

    log("")
    log("=" * 120)
    log("12G FINAL POINT-SAMPLED COMPARISON")
    log("=" * 120)
    log(comparison.to_string(index=False))
    log("")
    log(f"[SAVED] {output_root / '12G_point_sampled_final_comparison.csv'}")
    log(f"[SAVED] {output_root / '12G_POINT_FINAL_REPORT.txt'}")

    return comparison


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build/validate the point-sampled dataset and stop.",
    )

    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse an already built/validated Stage-12G dataset.",
    )

    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Rebuild the point dataset even if required outputs already exist.",
    )

    parser.add_argument(
        "--skip-lomo",
        action="store_true",
        help="Skip Stage-12C2 development-only LOMO.",
    )

    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Skip Stage-12D final point-sampled Tropical evaluation.",
    )

    parser.add_argument(
        "--preflight-final-only",
        action="store_true",
        help=(
            "After LOMO, run only Stage-12D --preflight-only and stop before "
            "final fitting/test evaluation."
        ),
    )

    parser.add_argument(
        "--debug-fast",
        action="store_true",
        help=(
            "NON-SCIENTIFIC: pass --debug-fast to Stage-12C2. "
            "Use with --skip-final."
        ),
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
    )

    parser.add_argument(
        "--allow-repeat-final-access",
        action="store_true",
        help=(
            "Pass Stage-12D --allow-repeat-test-access. Use only for a genuine "
            "technical rerun after this specific 12G final output was created."
        ),
    )

    args = parser.parse_args()

    project_root = args.project_root

    raw_dir = (
        args.raw_dir
        if args.raw_dir is not None
        else project_root / "data" / "raw" / "RawData"
    )

    output_root = (
        args.output_root
        if args.output_root is not None
        else project_root
        / "data"
        / "forecasting"
        / "12G_Nie_point_sampled_benchmark_v0_1"
    )

    dataset_dir = output_root / "dataset"
    lomo_dir = output_root / "lomo"
    final_dir = output_root / "final"

    for d in [output_root, dataset_dir, lomo_dir, final_dir]:
        d.mkdir(parents=True, exist_ok=True)

    stage12b_path = find_stage12b(project_root)
    stage12c2_path = find_stage12c2(project_root)
    stage12d_path = find_stage12d(project_root)

    manifest = {
        "stage": "12G",
        "script_version": SCRIPT_VERSION,
        "started_utc": utc_now_iso(),
        "project_root": str(project_root),
        "raw_dir": str(raw_dir),
        "output_root": str(output_root),
        "dataset_dir": str(dataset_dir),
        "lomo_dir": str(lomo_dir),
        "final_dir": str(final_dir),
        "stage12B_script": str(stage12b_path),
        "stage12C2_script": str(stage12c2_path),
        "stage12D_script": str(stage12d_path),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "phase_minute": POINT_PHASE_MINUTE,
        "phase_selection": "hard-frozen phase 0; no phase optimization",
        "scientific_role": (
            "secondary literature-aligned point-sampled sensitivity benchmark"
        ),
        "tropical_access_status": (
            "Tropical Atlantic was already accessed in Stage 12F for protocol audit."
        ),
        "published_BP_STGNN_reference": NIE_BP_STGNN_REFERENCE,
    }

    save_json(output_root / "12G_manifest.json", manifest)

    log("=" * 124)
    log("12G — BUILD AND RUN NIE POINT-SAMPLED BENCHMARK")
    log("=" * 124)
    log(f"script version  : {SCRIPT_VERSION}")
    log(f"project root    : {project_root}")
    log(f"raw dir         : {raw_dir}")
    log(f"output root     : {output_root}")
    log(f"12C2 script     : {stage12c2_path.name}")
    log(f"12D script      : {stage12d_path.name}")
    log(f"point protocol  : fixed UTC phase {POINT_PHASE_MINUTE}")
    log("phase tuning    : NONE")
    log("interpolation   : NONE")
    log(
        "[CAVEAT] Tropical Atlantic was already used in 12F for resampling "
        "forensics. 12G is a secondary point-protocol benchmark."
    )
    log("")

    if args.debug_fast and not args.skip_final:
        raise RuntimeError(
            "--debug-fast is NON-SCIENTIFIC and must be used with --skip-final."
        )

    if args.skip_build:
        if not dataset_complete(dataset_dir):
            raise RuntimeError(
                "--skip-build requested, but the required Stage-12G dataset "
                "files are incomplete."
            )
        log("[DATASET] Reusing existing Stage-12G point dataset.")
    else:
        need_build = args.rebuild_dataset or not dataset_complete(dataset_dir)

        if need_build:
            build_dataset(
                project_root=project_root,
                raw_dir=raw_dir,
                dataset_dir=dataset_dir,
                progress_every=args.progress_every,
            )
        else:
            log(
                "[DATASET] Required point dataset files already exist. "
                "Reusing them. Use --rebuild-dataset only if genuinely needed."
            )

    # Always validate the development/test Track-J files before model code.
    validation_rows = []

    for mission in DEVELOPMENT_MISSIONS:
        validation_rows.append(
            {
                "mission": mission,
                **validate_track_j_file(
                    dataset_dir
                    / "track_J_joint_compatible"
                    / f"{mission.replace(' ', '_')}.npz"
                ),
            }
        )

    validation_rows.append(
        {
            "mission": TEST_MISSION,
            **validate_track_j_file(
                dataset_dir
                / "track_J_joint_compatible"
                / "Tropical_Atlantic_TEST.npz"
            ),
        }
    )

    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(
        output_root / "12G_dataset_pre_model_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log("[DATASET VALIDATION]")
    log(
        validation_df[
            ["mission", "samples", "X_shape", "y_shape", "physics_max_abs_error"]
        ].to_string(index=False)
    )
    log("")

    if args.build_only:
        log("[DONE] --build-only requested. No model training was run.")
        return 0

    if not args.skip_lomo:
        command = [
            sys.executable,
            stage12c2_path,
            "--project-root",
            project_root,
            "--dataset-dir",
            dataset_dir,
            "--output-dir",
            lomo_dir,
        ]

        if args.force_cpu:
            command.append("--force-cpu")

        if args.debug_fast:
            command.append("--debug-fast")

        run_subprocess(
            command,
            cwd=project_root,
            label="12G STAGE B — DEVELOPMENT-ONLY POINT-SAMPLED LOMO via 12C2",
        )
    else:
        log("[LOMO] Skipped by user request.")

    if args.debug_fast:
        log(
            "[DONE] Debug-fast LOMO completed/skipped. Final evaluation is "
            "intentionally disabled."
        )
        return 0

    # Final preflight requires complete scientific 12C2 outputs.
    preflight_cmd = [
        sys.executable,
        stage12d_path,
        "--project-root",
        project_root,
        "--dataset-dir",
        dataset_dir,
        "--c2-dir",
        lomo_dir,
        "--output-dir",
        final_dir,
        "--preflight-only",
    ]

    if args.force_cpu:
        preflight_cmd.append("--force-cpu")

    run_subprocess(
        preflight_cmd,
        cwd=project_root,
        label="12G STAGE C — FINAL-TRAINING PREFLIGHT via 12D",
    )

    if args.preflight_final_only or args.skip_final:
        log(
            "[DONE] Final preflight passed; final point-sampled Tropical "
            "evaluation was not run."
        )
        return 0

    final_cmd = [
        sys.executable,
        stage12d_path,
        "--project-root",
        project_root,
        "--dataset-dir",
        dataset_dir,
        "--c2-dir",
        lomo_dir,
        "--output-dir",
        final_dir,
    ]

    if args.force_cpu:
        final_cmd.append("--force-cpu")

    if args.allow_repeat_final_access:
        final_cmd.append("--allow-repeat-test-access")

    run_subprocess(
        final_cmd,
        cwd=project_root,
        label="12G STAGE D — FINAL POINT-SAMPLED TROPICAL EVALUATION via 12D",
    )

    summarize_final(
        final_dir=final_dir,
        output_root=output_root,
    )

    manifest["completed_utc"] = utc_now_iso()
    manifest["status"] = "completed"
    save_json(output_root / "12G_manifest.json", manifest)

    log("")
    log("=" * 124)
    log("12G COMPLETE")
    log("=" * 124)
    log(f"Outputs: {output_root}")
    log("")
    log("Please send back:")
    log("  1) dataset mission sample counts;")
    log("  2) the 12C2 LOMO method summary;")
    log("  3) the final 12G comparison table / 12G_POINT_FINAL_REPORT.txt.")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n[FATAL ERROR]", flush=True)
        traceback.print_exc()
        print(
            "\nPlease send the complete traceback and terminal output.",
            flush=True,
        )
        sys.exit(1)
