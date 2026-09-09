# -*- coding: utf-8 -*-
r"""
12B_build_Nie_same_mission_benchmark_dataset.py

Stage 12B
=========
Build a SAME-MISSION, PAPER-INFORMED 10-min benchmark dataset from the raw
Saildrone missions used by Nie et al. for BP-STGNN.

NO MODEL TRAINING IS PERFORMED.

Default raw directory
---------------------
D:\project\WindPredict_SaildroneData\data\raw\RawData

Mission roles
-------------
Development missions:
    Antarctic
    Atlantic
    West Coast

Held-out test mission:
    Tropical Atlantic

The held-out test DATASET is built here using a preprocessing protocol frozen
before model training. Stage 12B does not evaluate any model on it.

Important compatibility decision
--------------------------------
Atlantic does not provide a true vessel HDG variable; HDG_WING is wing heading,
not platform heading. Therefore the Nie-data compatibility version of our joint
forecasting model does NOT use HDG.

Two exactly aligned tracks are produced:

Track P -- paper-informed wind benchmark
    10-min input nodes:
        U, V, SOG, COG, T, RH, P, dT, dP, dRH
    target:
        next 10-min U, V

Track J -- Physics-Compact NieData compatibility task
    9 input features:
        U, V, T, RH, SOG,
        COG_sin, COG_cos,
        WING_ANGLE_sin, WING_ANGLE_cos
    target:
        next 10-min U, V, VESSEL_E, VESSEL_N

    apparent-wind reference:
        AW_E = U - VESSEL_E
        AW_N = V - VESSEL_N

No body-frame AWA is constructed because Atlantic lacks true platform HDG.

10-min resampling
-----------------
A block is accepted only when:
    1. it contains exactly 10 timestamps;
    2. timestamps are exactly one minute apart and aligned to a UTC 10-min bin;
    3. all COMMON base variables are finite for all 10 raw minutes:
           U, V, SOG, COG, WING_ANGLE, T, RH, P

No interpolation is performed.

Block aggregation:
    U, V, SOG, T, RH, P : arithmetic mean
    COG                  : circular mean, degrees [0,360)
    WING_ANGLE           : circular mean, degrees [-180,180)
    VESSEL_E             : mean(SOG * sin(COG))
    VESSEL_N             : mean(SOG * cos(COG))

The vessel vector is averaged component-wise from the raw 1-min observations;
this is more physically coherent than reconstructing it from mean SOG/COG.

Paper-differential adaptation
-----------------------------
The full paper confirms dT, dP, dH/RH as nodes but does not provide their exact
numerical implementation. We predeclare:
    dT  = T[k]  - T[k-1]
    dP  = P[k]  - P[k-1]
    dRH = RH[k] - RH[k-1]
after 10-min resampling, and only across contiguous 10-min blocks.

Historical context
------------------
The paper does not report its exact history length. For this same-mission
benchmark we predeclare:
    lookback = 6 x 10-min steps = 60 min
    forecast = immediately following 10-min block

Thus this is PAPER-INFORMED / SAME-MISSION, not an exact official reproduction.

Sample alignment
----------------
Track P and Track J are constructed from the SAME accepted samples and therefore
have identical:
    context_end_time_ns
    target_time_ns
    mission labels

Data partition policy
---------------------
No contexts cross mission boundaries.

Development pool:
    Antarctic + Atlantic + West Coast

Held-out test:
    Tropical Atlantic

No train/validation split and no scaler are fitted in Stage 12B.
Stage 12C will perform development-only model selection / scaler fitting.
Tropical Atlantic must not be used for HPO.

Exact raw variable names frozen from Stage 12A v0.3
---------------------------------------------------
    U           = UWND_MEAN
    V           = VWND_MEAN
    SOG         = SOG
    COG         = COG
    WING_ANGLE  = WING_ANGLE
    T           = TEMP_AIR_MEAN
    RH          = RH_MEAN
    P           = BARO_PRES_MEAN

West Coast
----------
All .nc files under a path component "1026" are treated as the West Coast /
SD1026 segmented mission, concatenated chronologically, and duplicate timestamps
are consolidated by first non-null value per variable.

Outputs
-------
12B_Nie_same_mission_benchmark_v0_1/
    resampled_blocks/
        Antarctic_10min.csv
        Atlantic_10min.csv
        West_Coast_10min.csv
        Tropical_Atlantic_10min.csv
    track_P_paper_informed/
        Antarctic.npz
        Atlantic.npz
        West_Coast.npz
        Tropical_Atlantic_TEST.npz
        development_all.npz
    track_J_joint_compatible/
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

Run
---
conda activate WindPredict
python "D:\project\WindPredict_SaildroneData\src\12B_build_Nie_same_mission_benchmark_dataset.py"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0-Nie-same-mission-benchmark-builder"

DEFAULT_PROJECT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")
DEFAULT_RAW_DIR = DEFAULT_PROJECT_ROOT / "data" / "raw" / "RawData"
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12B_Nie_same_mission_benchmark_v0_1"
)

DEVELOPMENT_MISSIONS = ["Antarctic", "Atlantic", "West Coast"]
TEST_MISSION = "Tropical Atlantic"
ALL_MISSIONS = DEVELOPMENT_MISSIONS + [TEST_MISSION]

LOOKBACK_STEPS = 6
STEP_MINUTES = 10
FORECAST_STEPS = 1

RAW_NAME = {
    "U": "UWND_MEAN",
    "V": "VWND_MEAN",
    "SOG": "SOG",
    "COG": "COG",
    "WING_ANGLE": "WING_ANGLE",
    "T": "TEMP_AIR_MEAN",
    "RH": "RH_MEAN",
    "P": "BARO_PRES_MEAN",
}

COMMON_BASE = ["U", "V", "SOG", "COG", "WING_ANGLE", "T", "RH", "P"]

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

PAPER_TABLE1_COUNTS = {
    "Antarctic": 66126,
    "Atlantic": 60656,
    "West Coast": 88616,
    "Tropical Atlantic": 66240,
}

TEN_MIN_NS = 10 * 60 * 1_000_000_000
ONE_MIN_NS = 60 * 1_000_000_000


def log(message=""):
    print(message, flush=True)


def import_xarray():
    try:
        import xarray as xr
    except Exception as exc:
        raise RuntimeError(
            "xarray is required. Activate the WindPredict environment."
        ) from exc
    return xr


def save_json(path: Path, obj):
    def convert(v):
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
            return {str(k): convert(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [convert(x) for x in v]
        return v

    with path.open("w", encoding="utf-8") as f:
        json.dump(convert(obj), f, ensure_ascii=False, indent=2, sort_keys=True)


def normalize_time(values):
    flat = np.asarray(values).reshape(-1)
    dt = pd.to_datetime(flat, utc=True, errors="coerce")
    return (
        pd.DatetimeIndex(dt)
        .tz_convert(None)
        .to_numpy(dtype="datetime64[ns]")
    )


def ns_iso(v: int):
    return str(np.datetime64(int(v), "ns"))


def infer_platform(path: Path):
    text = str(path).lower()
    m = re.search(r"sd(\d{3,5})", text)

    if m:
        return f"SD{m.group(1)}"

    if any(part.lower() == "1026" for part in path.parts):
        return "SD1026"

    return ""


def infer_mission(path: Path):
    full = str(path).lower()
    parts = {x.lower() for x in path.parts}
    platform = infer_platform(path)

    if "1026" in parts or platform == "SD1026":
        return "West Coast"

    if platform == "SD1020" or "antarctica" in full or "antarctic" in full:
        return "Antarctic"

    if platform == "SD1061" or "eurec4a" in full or "atomic" in full:
        return "Tropical Atlantic"

    if platform == "SD1021":
        return "Atlantic"

    return "Unknown"


def discover_mission_files(raw_dir: Path):
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)

    files = sorted(p for p in raw_dir.rglob("*.nc") if p.is_file())

    if not files:
        raise FileNotFoundError(f"No .nc files found under {raw_dir}")

    grouped = defaultdict(list)

    for path in files:
        mission = infer_mission(path)

        if mission in ALL_MISSIONS:
            grouped[mission].append(path)

    missing = [m for m in ALL_MISSIONS if not grouped.get(m)]

    if missing:
        raise RuntimeError(
            f"Missing required mission file groups: {missing}. "
            f"Discovered groups={dict((k, len(v)) for k, v in grouped.items())}"
        )

    return grouped


def numeric_time_aligned(da, n_expected):
    raw = np.asarray(da.values)

    if raw.size != int(n_expected):
        return None

    try:
        arr = raw.reshape(-1).astype(np.float64)
    except Exception:
        return None

    # Defensive fill-value handling.
    for key in ["_FillValue", "missing_value"]:
        if key in da.attrs:
            try:
                sentinel = float(da.attrs[key])
                arr[arr == sentinel] = np.nan
            except Exception:
                pass

    arr[np.abs(arr) > 1e30] = np.nan

    return arr


def load_mission_raw(xr, mission: str, paths: List[Path], progress_every: int):
    frames = []
    file_rows = []

    for i, path in enumerate(sorted(paths), start=1):
        if (
            i == 1
            or i == len(paths)
            or i % max(1, progress_every) == 0
        ):
            log(f"    loading {i}/{len(paths)}: {path.name}")

        with xr.open_dataset(path, decode_times=True) as ds:
            if "time" not in ds.variables:
                raise RuntimeError(f"{path}: missing time variable.")

            raw_time = normalize_time(ds["time"].values)
            valid_time = ~pd.isna(pd.to_datetime(raw_time))

            time_ns = (
                raw_time[valid_time]
                .astype("datetime64[ns]")
                .astype(np.int64)
            )

            if len(time_ns) == 0:
                continue

            order = np.argsort(time_ns)
            time_ns = time_ns[order]

            frame = pd.DataFrame({"time_ns": time_ns})

            for canonical, raw_name in RAW_NAME.items():
                if raw_name not in ds.variables:
                    # Segment-level absence is represented as NaN; the later
                    # complete-block rule will reject affected intervals.
                    frame[canonical] = np.nan
                    continue

                arr = numeric_time_aligned(
                    ds[raw_name],
                    np.asarray(ds["time"].values).size,
                )

                if arr is None:
                    raise RuntimeError(
                        f"{path}: {raw_name} cannot be aligned with flattened time."
                    )

                frame[canonical] = arr[valid_time][order]

            frames.append(frame)

            file_rows.append({
                "mission": mission,
                "file": path.name,
                "rows": int(len(time_ns)),
                "start": ns_iso(time_ns[0]),
                "end": ns_iso(time_ns[-1]),
            })

    if not frames:
        raise RuntimeError(f"{mission}: no readable raw rows.")

    raw = pd.concat(frames, ignore_index=True, copy=False)
    raw = raw.sort_values("time_ns", kind="mergesort").reset_index(drop=True)

    rows_before = int(len(raw))
    duplicate_rows = int(raw["time_ns"].duplicated(keep=False).sum())

    if duplicate_rows:
        raw = (
            raw.groupby("time_ns", as_index=False, sort=True)
            .first()
        )

    raw = raw.sort_values("time_ns").reset_index(drop=True)

    return raw, {
        "mission": mission,
        "segment_files": int(len(paths)),
        "rows_before_dedup": rows_before,
        "duplicate_rows_detected": duplicate_rows,
        "rows_after_dedup": int(len(raw)),
        "start": ns_iso(raw["time_ns"].iloc[0]),
        "end": ns_iso(raw["time_ns"].iloc[-1]),
    }, file_rows


def circular_mean_deg(values, wrap_360: bool):
    values = np.asarray(values, dtype=np.float64)

    if not np.isfinite(values).all():
        return np.nan

    rad = np.deg2rad(values)
    s = float(np.mean(np.sin(rad)))
    c = float(np.mean(np.cos(rad)))

    # Degenerate circular mean.
    if abs(s) < 1e-12 and abs(c) < 1e-12:
        return np.nan

    angle = float(np.rad2deg(np.arctan2(s, c)))

    if wrap_360:
        angle = angle % 360.0
    else:
        angle = ((angle + 180.0) % 360.0) - 180.0

    return angle


def build_complete_10min_blocks(raw: pd.DataFrame):
    """
    Return only blocks that satisfy the frozen COMMON completeness protocol.
    """
    if raw.empty:
        return pd.DataFrame()

    time_ns = raw["time_ns"].to_numpy(dtype=np.int64)
    block_start = (time_ns // TEN_MIN_NS) * TEN_MIN_NS

    unique, first, counts = np.unique(
        block_start,
        return_index=True,
        return_counts=True,
    )

    rows = []
    rejected_bad_count = 0
    rejected_bad_time = 0
    rejected_nonfinite = 0

    for b, s, c in zip(unique, first, counts):
        if int(c) != 10:
            rejected_bad_count += 1
            continue

        s = int(s)
        e = s + 10

        actual_time = time_ns[s:e]
        expected_time = int(b) + np.arange(10, dtype=np.int64) * ONE_MIN_NS

        if not np.array_equal(actual_time, expected_time):
            rejected_bad_time += 1
            continue

        block = raw.iloc[s:e]

        common_matrix = block[COMMON_BASE].to_numpy(dtype=np.float64)

        if not np.isfinite(common_matrix).all():
            rejected_nonfinite += 1
            continue

        U = float(block["U"].mean())
        V = float(block["V"].mean())
        SOG = float(block["SOG"].mean())
        COG = circular_mean_deg(block["COG"].to_numpy(), wrap_360=True)
        WING = circular_mean_deg(
            block["WING_ANGLE"].to_numpy(),
            wrap_360=False,
        )
        T = float(block["T"].mean())
        RH = float(block["RH"].mean())
        P = float(block["P"].mean())

        cog_rad_raw = np.deg2rad(block["COG"].to_numpy(dtype=np.float64))
        sog_raw = block["SOG"].to_numpy(dtype=np.float64)

        vessel_e = float(np.mean(sog_raw * np.sin(cog_rad_raw)))
        vessel_n = float(np.mean(sog_raw * np.cos(cog_rad_raw)))

        aw_e = U - vessel_e
        aw_n = V - vessel_n

        wind_speed = float(np.hypot(U, V))
        wind_dir_from = float(
            np.rad2deg(np.arctan2(-U, -V)) % 360.0
        )

        rows.append({
            "block_start_ns": int(b),
            "block_end_ns": int(b + 9 * ONE_MIN_NS),
            "U": U,
            "V": V,
            "SOG": SOG,
            "COG": COG,
            "WING_ANGLE": WING,
            "T": T,
            "RH": RH,
            "P": P,
            "VESSEL_E": vessel_e,
            "VESSEL_N": vessel_n,
            "AW_E": aw_e,
            "AW_N": aw_n,
            "WIND_SPEED": wind_speed,
            "WIND_DIR_FROM_DEG": wind_dir_from,
        })

    blocks = pd.DataFrame(rows)

    if blocks.empty:
        return blocks, {
            "candidate_10min_bins": int(len(unique)),
            "accepted_complete_blocks": 0,
            "rejected_bad_raw_count": rejected_bad_count,
            "rejected_bad_timestamp_grid": rejected_bad_time,
            "rejected_nonfinite_common_base": rejected_nonfinite,
        }

    blocks = blocks.sort_values("block_start_ns").reset_index(drop=True)

    # Paper differentials: only across exactly contiguous 10-min blocks.
    prev_time = blocks["block_start_ns"].shift(1)
    contiguous_prev = (
        blocks["block_start_ns"] - prev_time
    ) == TEN_MIN_NS

    blocks["dT"] = blocks["T"].diff()
    blocks["dP"] = blocks["P"].diff()
    blocks["dRH"] = blocks["RH"].diff()

    for col in ["dT", "dP", "dRH"]:
        blocks.loc[~contiguous_prev, col] = np.nan

    cog_rad = np.deg2rad(blocks["COG"].to_numpy(dtype=np.float64))
    wing_rad = np.deg2rad(blocks["WING_ANGLE"].to_numpy(dtype=np.float64))

    blocks["COG_sin"] = np.sin(cog_rad)
    blocks["COG_cos"] = np.cos(cog_rad)
    blocks["WING_ANGLE_sin"] = np.sin(wing_rad)
    blocks["WING_ANGLE_cos"] = np.cos(wing_rad)

    return blocks, {
        "candidate_10min_bins": int(len(unique)),
        "accepted_complete_blocks": int(len(blocks)),
        "rejected_bad_raw_count": int(rejected_bad_count),
        "rejected_bad_timestamp_grid": int(rejected_bad_time),
        "rejected_nonfinite_common_base": int(rejected_nonfinite),
    }


def build_aligned_samples(blocks: pd.DataFrame, mission: str):
    """
    Build exactly aligned Track-P and Track-J samples.

    A valid sample needs:
        previous block for dT/dP/dRH
        6 contiguous context blocks
        1 contiguous target block
    """
    if blocks.empty:
        raise RuntimeError(f"{mission}: no accepted 10-min blocks.")

    Xp = []
    Xj = []
    yp = []
    yj = []
    aw = []
    context_end = []
    target_time = []
    target_speed = []
    target_direction = []

    starts = blocks["block_start_ns"].to_numpy(dtype=np.int64)

    # End context index c; context is [c-5, ..., c], target = c+1.
    for c in range(LOOKBACK_STEPS - 1, len(blocks) - 1):
        a = c - LOOKBACK_STEPS + 1
        t = c + 1

        context_times = starts[a:c + 1]
        expected_context = (
            starts[a]
            + np.arange(LOOKBACK_STEPS, dtype=np.int64) * TEN_MIN_NS
        )

        if not np.array_equal(context_times, expected_context):
            continue

        if starts[t] - starts[c] != TEN_MIN_NS:
            continue

        p_context = blocks.iloc[a:c + 1][TRACK_P_FEATURES].to_numpy(
            dtype=np.float32
        )

        j_context = blocks.iloc[a:c + 1][TRACK_J_FEATURES].to_numpy(
            dtype=np.float32
        )

        if not np.isfinite(p_context).all():
            # This typically removes the first context after a gap because
            # its dT/dP/dRH are undefined.
            continue

        if not np.isfinite(j_context).all():
            continue

        target = blocks.iloc[t]

        p_target = np.array(
            [[target["U"], target["V"]]],
            dtype=np.float32,
        )

        j_target = np.array(
            [[
                target["U"],
                target["V"],
                target["VESSEL_E"],
                target["VESSEL_N"],
            ]],
            dtype=np.float32,
        )

        aw_target = np.array(
            [[target["AW_E"], target["AW_N"]]],
            dtype=np.float32,
        )

        if (
            not np.isfinite(p_target).all()
            or not np.isfinite(j_target).all()
            or not np.isfinite(aw_target).all()
        ):
            continue

        Xp.append(p_context)
        Xj.append(j_context)
        yp.append(p_target)
        yj.append(j_target)
        aw.append(aw_target)

        context_end.append(int(starts[c]))
        target_time.append([int(starts[t])])
        target_speed.append([float(target["WIND_SPEED"])])
        target_direction.append([float(target["WIND_DIR_FROM_DEG"])])

    if not Xp:
        raise RuntimeError(f"{mission}: no aligned forecasting samples.")

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
        "mission": np.asarray([mission] * len(Xp)),
    }

    # Alignment and physics audits.
    if not np.array_equal(
        result["context_end_time_ns"],
        result["context_end_time_ns"],
    ):
        raise RuntimeError("Internal alignment audit failed.")

    physics = (
        result["y_J"][:, :, 0:2]
        - result["y_J"][:, :, 2:4]
    )

    max_physics_error = float(
        np.max(np.abs(physics - result["apparent_ref"]))
    )

    if max_physics_error > 1e-5:
        raise RuntimeError(
            f"{mission}: apparent-wind physics audit failed, "
            f"max abs error={max_physics_error:.3e}"
        )

    return result, max_physics_error


def save_track_p(path: Path, samples: Dict):
    np.savez_compressed(
        path,
        X_raw=samples["X_P"],
        y_wind_raw=samples["y_P"],
        context_end_time_ns=samples["context_end_time_ns"],
        target_time_ns=samples["target_time_ns"],
        target_wind_speed=samples["target_wind_speed"],
        target_wind_direction_from_deg=samples[
            "target_wind_direction_from_deg"
        ],
        mission=samples["mission"],
        feature_names=np.asarray(TRACK_P_FEATURES),
        target_names=np.asarray(TRACK_P_TARGETS),
        lookback_steps=np.asarray([LOOKBACK_STEPS], dtype=np.int64),
        step_minutes=np.asarray([STEP_MINUTES], dtype=np.int64),
        forecast_steps=np.asarray([FORECAST_STEPS], dtype=np.int64),
    )


def save_track_j(path: Path, samples: Dict):
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
        target_wind_direction_from_deg=samples[
            "target_wind_direction_from_deg"
        ],
        mission=samples["mission"],
        feature_names=np.asarray(TRACK_J_FEATURES),
        target_names=np.asarray(TRACK_J_TARGETS),
        apparent_names=np.asarray(APPARENT_TARGETS),
        lookback_steps=np.asarray([LOOKBACK_STEPS], dtype=np.int64),
        step_minutes=np.asarray([STEP_MINUTES], dtype=np.int64),
        forecast_steps=np.asarray([FORECAST_STEPS], dtype=np.int64),
    )


def concatenate_samples(items: List[Dict]):
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


def validate_npz_pair(track_p_path: Path, track_j_path: Path):
    with np.load(track_p_path, allow_pickle=False) as p, np.load(
        track_j_path,
        allow_pickle=False,
    ) as j:
        p_context = p["context_end_time_ns"]
        j_context = j["context_end_time_ns"]

        p_target = p["target_time_ns"]
        j_target = j["target_time_ns"]

        aligned = (
            np.array_equal(p_context, j_context)
            and np.array_equal(p_target, j_target)
            and len(p_context) == len(j_context)
        )

        finite = (
            np.isfinite(p["X_raw"]).all()
            and np.isfinite(p["y_wind_raw"]).all()
            and np.isfinite(j["X_raw"]).all()
            and np.isfinite(j["y_joint_raw"]).all()
            and np.isfinite(j["apparent_ref"]).all()
        )

        return {
            "aligned": bool(aligned),
            "finite": bool(finite),
            "samples": int(len(p_context)),
            "P_X_shape": tuple(int(x) for x in p["X_raw"].shape),
            "P_y_shape": tuple(int(x) for x in p["y_wind_raw"].shape),
            "J_X_shape": tuple(int(x) for x in j["X_raw"].shape),
            "J_y_shape": tuple(int(x) for x in j["y_joint_raw"].shape),
        }


def safe_mission_filename(mission: str):
    return mission.replace(" ", "_")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        default=str(DEFAULT_PROJECT_ROOT),
    )

    parser.add_argument(
        "--raw-dir",
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    project_root = Path(args.project_root)

    raw_dir = (
        Path(args.raw_dir)
        if args.raw_dir
        else project_root / "data" / "raw" / "RawData"
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root
        / "data"
        / "forecasting"
        / "12B_Nie_same_mission_benchmark_v0_1"
    )

    block_dir = output_dir / "resampled_blocks"
    p_dir = output_dir / "track_P_paper_informed"
    j_dir = output_dir / "track_J_joint_compatible"

    for directory in [output_dir, block_dir, p_dir, j_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    xr = import_xarray()
    grouped = discover_mission_files(raw_dir)

    log("=" * 122)
    log("12B - BUILD NIE SAME-MISSION 10-MIN BENCHMARK DATASET")
    log("=" * 122)
    log(f"script version        : {SCRIPT_VERSION}")
    log(f"raw dir               : {raw_dir}")
    log(f"output dir            : {output_dir}")
    log(f"lookback              : {LOOKBACK_STEPS} x 10 min = 60 min")
    log("forecast              : next 10-min block")
    log(f"development missions  : {DEVELOPMENT_MISSIONS}")
    log(f"held-out test mission : {TEST_MISSION}")
    log("HDG used              : NO")
    log("interpolation         : NONE")
    log("model training        : NONE")
    log("")

    mission_build_rows = []
    alignment_rows = []
    per_mission_samples = {}

    for mission in ALL_MISSIONS:
        paths = grouped[mission]

        log("-" * 122)
        log(f"[MISSION] {mission} | files={len(paths)}")

        raw, raw_audit, _ = load_mission_raw(
            xr,
            mission,
            paths,
            args.progress_every,
        )

        blocks, block_audit = build_complete_10min_blocks(raw)

        if blocks.empty:
            raise RuntimeError(
                f"{mission}: no complete common 10-min blocks."
            )

        samples, physics_error = build_aligned_samples(
            blocks,
            mission,
        )

        per_mission_samples[mission] = samples

        block_csv = block_dir / f"{safe_mission_filename(mission)}_10min.csv"

        blocks_to_save = blocks.copy()
        blocks_to_save["block_start_time"] = pd.to_datetime(
            blocks_to_save["block_start_ns"],
            unit="ns",
        )
        blocks_to_save["block_end_time"] = pd.to_datetime(
            blocks_to_save["block_end_ns"],
            unit="ns",
        )
        blocks_to_save.to_csv(
            block_csv,
            index=False,
            encoding="utf-8-sig",
        )

        suffix = "_TEST" if mission == TEST_MISSION else ""

        p_path = p_dir / f"{safe_mission_filename(mission)}{suffix}.npz"
        j_path = j_dir / f"{safe_mission_filename(mission)}{suffix}.npz"

        save_track_p(p_path, samples)
        save_track_j(j_path, samples)

        pair_audit = validate_npz_pair(p_path, j_path)

        if not pair_audit["aligned"] or not pair_audit["finite"]:
            raise RuntimeError(
                f"{mission}: saved Track-P/Track-J validation failed: "
                f"{pair_audit}"
            )

        paper_count = PAPER_TABLE1_COUNTS.get(mission)

        mission_build_rows.append({
            "mission": mission,
            "role": (
                "HELD_OUT_TEST"
                if mission == TEST_MISSION
                else "DEVELOPMENT"
            ),
            "segment_files": raw_audit["segment_files"],
            "raw_rows_after_dedup": raw_audit["rows_after_dedup"],
            "paper_Table1_reference_rows": paper_count,
            "candidate_10min_bins": block_audit["candidate_10min_bins"],
            "accepted_complete_10min_blocks": block_audit[
                "accepted_complete_blocks"
            ],
            "rejected_bad_raw_count": block_audit[
                "rejected_bad_raw_count"
            ],
            "rejected_bad_timestamp_grid": block_audit[
                "rejected_bad_timestamp_grid"
            ],
            "rejected_nonfinite_common_base": block_audit[
                "rejected_nonfinite_common_base"
            ],
            "forecast_samples": int(len(samples["X_P"])),
            "first_context_end": ns_iso(
                samples["context_end_time_ns"][0]
            ),
            "last_context_end": ns_iso(
                samples["context_end_time_ns"][-1]
            ),
            "first_target": ns_iso(
                samples["target_time_ns"][0, 0]
            ),
            "last_target": ns_iso(
                samples["target_time_ns"][-1, 0]
            ),
            "physics_max_abs_error": physics_error,
            "Track_P_X_shape": str(samples["X_P"].shape),
            "Track_J_X_shape": str(samples["X_J"].shape),
        })

        alignment_rows.append({
            "mission": mission,
            "samples": pair_audit["samples"],
            "track_P_J_time_alignment": pair_audit["aligned"],
            "all_saved_arrays_finite": pair_audit["finite"],
            "P_X_shape": str(pair_audit["P_X_shape"]),
            "P_y_shape": str(pair_audit["P_y_shape"]),
            "J_X_shape": str(pair_audit["J_X_shape"]),
            "J_y_shape": str(pair_audit["J_y_shape"]),
        })

        log(
            f"  raw rows             : {raw_audit['rows_after_dedup']:,}"
        )
        log(
            f"  complete 10-min      : {len(blocks):,}"
        )
        log(
            f"  forecast samples     : {len(samples['X_P']):,}"
        )
        log(
            f"  Track P              : X={samples['X_P'].shape} | "
            f"y={samples['y_P'].shape}"
        )
        log(
            f"  Track J              : X={samples['X_J'].shape} | "
            f"y={samples['y_J'].shape}"
        )
        log(
            f"  physics AW audit     : max abs error={physics_error:.3e}"
        )

        del raw, blocks

    # Development pool is concatenated mission-wise. No context crosses missions.
    dev = concatenate_samples(
        [per_mission_samples[m] for m in DEVELOPMENT_MISSIONS]
    )

    p_dev_path = p_dir / "development_all.npz"
    j_dev_path = j_dir / "development_all.npz"

    save_track_p(p_dev_path, dev)
    save_track_j(j_dev_path, dev)

    dev_audit = validate_npz_pair(p_dev_path, j_dev_path)

    if not dev_audit["aligned"] or not dev_audit["finite"]:
        raise RuntimeError(
            f"Development-pool validation failed: {dev_audit}"
        )

    schema = {
        "stage": "12B",
        "script_version": SCRIPT_VERSION,
        "sampling": {
            "raw_resolution_minutes": 1,
            "resampled_resolution_minutes": 10,
            "lookback_steps": LOOKBACK_STEPS,
            "lookback_minutes": LOOKBACK_STEPS * STEP_MINUTES,
            "forecast_steps": FORECAST_STEPS,
            "forecast_minutes": FORECAST_STEPS * STEP_MINUTES,
        },
        "raw_variable_names": RAW_NAME,
        "common_complete_block_variables": COMMON_BASE,
        "Track_P": {
            "role": "paper-informed wind benchmark",
            "feature_names": TRACK_P_FEATURES,
            "target_names": TRACK_P_TARGETS,
            "differential_definition": {
                "dT": "T[k]-T[k-1] after 10-min resampling",
                "dP": "P[k]-P[k-1] after 10-min resampling",
                "dRH": "RH[k]-RH[k-1] after 10-min resampling",
            },
        },
        "Track_J": {
            "role": "Physics-Compact NieData compatibility task",
            "feature_names": TRACK_J_FEATURES,
            "target_names": TRACK_J_TARGETS,
            "apparent_reference_names": APPARENT_TARGETS,
            "vessel_components": {
                "VESSEL_E": "mean over raw 1-min block of SOG*sin(COG)",
                "VESSEL_N": "mean over raw 1-min block of SOG*cos(COG)",
            },
            "apparent_wind": {
                "AW_E": "U - VESSEL_E",
                "AW_N": "V - VESSEL_N",
            },
            "HDG_policy": (
                "HDG excluded because Atlantic has HDG_WING but no true "
                "platform HDG; HDG_WING is not substituted for vessel heading."
            ),
        },
    }

    save_json(output_dir / "feature_target_schema.json", schema)

    mission_df = pd.DataFrame(mission_build_rows)
    alignment_df = pd.DataFrame(alignment_rows)

    mission_df.to_csv(
        output_dir / "mission_build_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    alignment_df.to_csv(
        output_dir / "sample_alignment_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_samples = per_mission_samples[TEST_MISSION]

    manifest = {
        "stage": "12B",
        "script_version": SCRIPT_VERSION,
        "dataset_name": "Nie_same_mission_benchmark_v0_1",
        "scientific_status": "FROZEN_DATASET_CANDIDATE",
        "protocol_label": (
            "paper-informed same-mission benchmark; not exact official "
            "BP-STGNN reproduction"
        ),
        "development_missions": DEVELOPMENT_MISSIONS,
        "held_out_test_mission": TEST_MISSION,
        "development_samples": int(len(dev["X_P"])),
        "held_out_test_samples": int(len(test_samples["X_P"])),
        "Track_P_feature_count": len(TRACK_P_FEATURES),
        "Track_J_feature_count": len(TRACK_J_FEATURES),
        "Track_P_target_count": len(TRACK_P_TARGETS),
        "Track_J_target_count": len(TRACK_J_TARGETS),
        "Track_P_J_exact_time_alignment": bool(
            alignment_df["track_P_J_time_alignment"].all()
            and dev_audit["aligned"]
        ),
        "no_interpolation": True,
        "no_scaler_fit": True,
        "no_model_training": True,
        "no_model_performance_evaluated": True,
        "held_out_test_policy": (
            "Tropical Atlantic dataset is constructed under the frozen common "
            "preprocessing protocol, but must not be used for HPO or model "
            "selection in Stage 12C."
        ),
        "comparison_caveat": (
            "The missions and 10-min resolution match the paper setting, but "
            "the paper does not report exact history length, exact 10-min "
            "aggregation operator, exact differential formula, or exact "
            "training-interval selection for all development missions. "
            "No interpolation is used here, whereas the paper states that "
            "short gaps were spline-interpolated. Therefore comparisons to "
            "published BP-STGNN values are same-mission/paper-informed, not "
            "sample-for-sample official reproduction."
        ),
    }

    save_json(output_dir / "build_manifest.json", manifest)

    with (output_dir / "build_report.txt").open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("12B Nie Same-Mission Benchmark Dataset Build\n")
        f.write("=" * 122 + "\n\n")
        f.write(
            "This is a PAPER-INFORMED SAME-MISSION benchmark, not an exact "
            "official BP-STGNN reproduction.\n\n"
        )
        f.write("MISSION BUILD AUDIT\n")
        f.write("-" * 122 + "\n")
        f.write(mission_df.to_string(index=False))
        f.write("\n\nALIGNMENT AUDIT\n")
        f.write("-" * 122 + "\n")
        f.write(alignment_df.to_string(index=False))
        f.write("\n\nDEVELOPMENT POOL\n")
        f.write("-" * 122 + "\n")
        f.write(
            f"missions={DEVELOPMENT_MISSIONS}\n"
            f"samples={len(dev['X_P'])}\n"
            f"Track-P X={dev['X_P'].shape}, y={dev['y_P'].shape}\n"
            f"Track-J X={dev['X_J'].shape}, y={dev['y_J'].shape}\n"
        )
        f.write("\nHELD-OUT TEST\n")
        f.write("-" * 122 + "\n")
        f.write(
            f"mission={TEST_MISSION}\n"
            f"samples={len(test_samples['X_P'])}\n"
            "No model performance is evaluated in Stage 12B.\n"
        )
        f.write("\nHDG POLICY\n")
        f.write("-" * 122 + "\n")
        f.write(
            "HDG is excluded from Track J. Atlantic HDG_WING is wing heading "
            "and is not treated as vessel/platform heading.\n"
        )

    log("")
    log("=" * 122)
    log("12B BUILD SUMMARY")
    log("=" * 122)

    for row in mission_build_rows:
        log(
            f"{row['mission']}: role={row['role']} | "
            f"raw={row['raw_rows_after_dedup']:,} | "
            f"10min={row['accepted_complete_10min_blocks']:,} | "
            f"samples={row['forecast_samples']:,}"
        )

    log("")
    log(
        f"Development pool: samples={len(dev['X_P']):,} | "
        f"Track-P X={dev['X_P'].shape} | Track-J X={dev['X_J'].shape}"
    )
    log(
        f"Held-out Tropical Atlantic: samples={len(test_samples['X_P']):,}"
    )
    log(
        "[PASS] Track-P and Track-J use exactly aligned context/target timestamps."
    )
    log(
        "[PASS] HDG_WING is not used as vessel heading."
    )
    log(
        "[POLICY] No interpolation, scaler fitting, model training, HPO, or "
        "test performance evaluation was performed."
    )
    log(
        "[NOTE] Published BP-STGNN comparisons must be described as "
        "same-mission/paper-informed rather than exact sample-for-sample "
        "reproduction."
    )
    log(f"[DONE] 12B outputs: {output_dir}")

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
