# -*- coding: utf-8 -*-
r"""
12A_audit_Nie_BPSTGNN_raw_dataset_v0_2.py

Stage 12A v0.2
==============
Mission-level audit of the raw Saildrone data used for the Nie/BP-STGNN
same-data benchmark.

This revision specifically handles the West Coast mission stored as MANY
NetCDF segment files under:

    D:\project\WindPredict_SaildroneData\data\raw\RawData\1026

Key changes from v0.1
---------------------
1. ALWAYS discovers NetCDF files recursively with rglob("*.nc").
2. Any .nc under a directory component named "1026", or whose filename
   contains sd1026, is assigned to the "West Coast" mission candidate.
3. The 933 West-Coast segment files are audited as ONE mission after
   concatenation, chronological sorting, and duplicate-timestamp consolidation.
4. 10-min completeness is computed at mission level, so blocks crossing
   individual file boundaries are not lost.
5. Paper Table-1 sample-count comparisons are done at MISSION level, not
   separately for each one of the 933 files.
6. Progress is printed while scanning large segmented missions.

NO MODEL TRAINING.
NO INTERPOLATION.
NO PERFORMANCE COMPARISON.

Default raw folder
------------------
D:\project\WindPredict_SaildroneData\data\raw\RawData

Known single-file missions
--------------------------
- Atlantic / SD1021
- Tropical Atlantic / SD1061 (EUREC4A/ATOMIC)
- Antarctic / SD1020

Segmented mission
-----------------
- West Coast / SD1026 / RawData\1026\*.nc

Outputs
-------
12A_Nie_BPSTGNN_raw_dataset_audit_v0_2/
    segment_inventory.csv
    mission_inventory.csv
    variable_mapping_audit.csv
    mission_variable_coverage.csv
    metadata_wind_semantics_audit.csv
    mission_temporal_audit.csv
    mission_ten_min_block_audit.csv
    paper_region_mapping_audit.csv
    transferability_audit.csv
    audit_manifest.json
    audit_report.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.3.0-Nie-raw-data-dimension-fix"

DEFAULT_PROJECT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")
DEFAULT_RAW_DIR = DEFAULT_PROJECT_ROOT / "data" / "raw" / "RawData"
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12A_Nie_BPSTGNN_raw_dataset_audit_v0_3"
)

KNOWN_SINGLE_FILES = {
    "Atlantic": (
        "saildrone-gen_5-1021_atlantic-sd1021-20190525T000000-"
        "20191021T235959-1_minutes-v1.1571806429446.nc"
    ),
    "Tropical Atlantic": (
        "saildrone-gen_5-atomic_eurec4a_2020-sd1061-20200117T000000-"
        "20200302T235959-1_minutes-v1.1595707641693.nc"
    ),
    "Antarctic": (
        "surface-saildrone-gen_5-antarctica_circumnavigation_2019-sd1020-"
        "20190119T040000-20190803T043000-1_minutes-v1.1620360815446.nc"
    ),
}

PAPER_TABLE1_COUNTS = {
    "Antarctic": 66126,
    "Atlantic": 60656,
    "West Coast": 88616,
    "Tropical Atlantic": 66240,
}

ALIASES = {
    "U": [
        "UWND_MEAN", "UWND", "U_WIND", "wind_u",
        "eastward_wind", "U",
    ],
    "V": [
        "VWND_MEAN", "VWND", "V_WIND", "wind_v",
        "northward_wind", "V",
    ],
    "SOG": [
        "SOG", "platform_speed_wrt_ground", "speed_over_ground",
        "VEHICLE_SPEED", "SPEED_OVER_GROUND",
    ],
    "COG": [
        "COG", "platform_course", "course_over_ground",
        "VEHICLE_COURSE", "COURSE_OVER_GROUND",
    ],
    "HDG": [
        "HDG", "heading", "platform_yaw_angle",
        "vehicle_heading", "HEADING",
    ],
    "WING_ANGLE": [
        "WING_ANGLE", "wing_angle", "SAIL_ANGLE",
        "sail_angle", "WING_ANGLE_MEAN",
    ],
    "T": [
        "TEMP_AIR_MEAN", "TEMP_AIR", "air_temperature",
        "AIR_TEMP", "TA",
    ],
    "RH": [
        "RH_MEAN", "RH", "relative_humidity",
        "RELATIVE_HUMIDITY",
    ],
    "P": [
        "BARO_PRES_MEAN", "BARO_PRES", "AIR_PRESSURE_MEAN",
        "PRESSURE_AIR_MEAN", "PRES_MEAN", "air_pressure",
        "ATM_PRESSURE", "P",
    ],
    "AWS": [
        "WIND_SPEED_MEAN", "WIND_SPEED", "AWS",
        "apparent_wind_speed", "relative_wind_speed",
    ],
    "AWA": [
        "WIND_DIRECTION_MEAN", "WIND_DIRECTION", "AWA",
        "apparent_wind_direction", "relative_wind_direction",
    ],
}

PAPER_BASE = ["U", "V", "SOG", "COG", "T", "RH", "P"]
OUR_PREFERRED = [
    "U", "V", "SOG", "COG", "HDG", "WING_ANGLE", "T", "RH"
]
AUDIT_VARIABLES = list(ALIASES)


def log(msg=""):
    print(msg, flush=True)


def import_xarray():
    try:
        import xarray as xr
    except Exception as exc:
        raise RuntimeError(
            "xarray is required. Activate the WindPredict environment."
        ) from exc
    return xr


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
        json.dump(cv(obj), f, ensure_ascii=False, indent=2)


def normalize_time(values):
    """
    Saildrone legacy NetCDF files may store trajectory data as (1, N) or
    (N, 1), not strictly (N,). Always flatten the time variable first.
    """
    flat = np.asarray(values).reshape(-1)
    dt = pd.to_datetime(flat, utc=True, errors="coerce")
    return (
        pd.DatetimeIndex(dt)
        .tz_convert(None)
        .to_numpy(dtype="datetime64[ns]")
    )


def ns_iso(v):
    return str(np.datetime64(int(v), "ns"))


def infer_platform(path: Path):
    # Search full path so folder names can also help.
    text = str(path).lower()
    m = re.search(r"sd(\d{3,5})", text)
    if m:
        return f"SD{m.group(1)}"

    # Explicit segmented-folder fallback.
    if any(part.lower() == "1026" for part in path.parts):
        return "SD1026"

    return ""


def infer_region(path: Path):
    """
    Region inference uses the FULL PATH, not just filename.

    This is essential for the RawData/1026 segmented mission where filenames may not
    contain the phrase "West Coast".
    """
    full = str(path).lower()
    parts = {x.lower() for x in path.parts}
    platform = infer_platform(path)

    if "1026" in parts or platform == "SD1026":
        return "West Coast"

    if "antarctica" in full or "antarctic" in full or platform == "SD1020":
        return "Antarctic"

    if (
        "eurec4a" in full
        or "atomic" in full
        or platform == "SD1061"
    ):
        return "Tropical Atlantic"

    if platform == "SD1021" or "atlantic" in full:
        return "Atlantic"

    return "Unknown"


def mission_key(path: Path):
    region = infer_region(path)
    platform = infer_platform(path)
    if region != "Unknown":
        return region
    if platform:
        return f"Unknown-{platform}"
    return f"Unknown-{path.parent.name}"


def metadata_text(da):
    pieces = [str(da.name)]
    for key in [
        "standard_name",
        "long_name",
        "description",
        "comment",
        "units",
        "coordinates",
        "reference",
    ]:
        if key in da.attrs:
            pieces.append(str(da.attrs.get(key, "")))
    return " | ".join(pieces)


def resolve_variable(ds, canonical):
    aliases = ALIASES[canonical]
    names = [str(x) for x in ds.variables]
    lower = {x.lower(): x for x in names}

    for alias in aliases:
        if alias in ds.variables:
            return alias, "exact_alias", 100

        if alias.lower() in lower:
            return lower[alias.lower()], "case_insensitive_alias", 95

    token_map = {
        "U": ["eastward wind", "eastward_wind", "uwnd"],
        "V": ["northward wind", "northward_wind", "vwnd"],
        "SOG": ["speed over ground", "speed_wrt_ground", "sog"],
        "COG": ["course over ground", "platform_course", "cog"],
        "HDG": ["heading", "yaw", "hdg"],
        "WING_ANGLE": ["wing angle", "sail angle", "wing_angle"],
        "T": ["air temperature", "temp_air"],
        "RH": ["relative humidity", "rh_mean"],
        "P": ["air pressure", "barometric pressure", "baro_pres"],
        "AWS": ["apparent wind speed", "relative wind speed", "wind speed"],
        "AWA": [
            "apparent wind direction",
            "relative wind direction",
            "wind direction",
        ],
    }

    best_name = None
    best_score = -999

    for name in names:
        da = ds[name]
        text = metadata_text(da).lower()

        score = sum(
            20
            for token in token_map[canonical]
            if token in text
        )

        if any(
            bad in name.lower()
            for bad in ["qc", "quality", "flag", "status"]
        ):
            score -= 60

        # Legacy Saildrone trajectory files often store otherwise 1-D time
        # series as (trajectory, obs), e.g. (1, N). Do NOT penalize ndim>1.
        # Alignment is checked later against the flattened time-variable size.

        if score > best_score:
            best_name = name
            best_score = score

    if best_score >= 20:
        return best_name, "metadata_scored", int(best_score)

    return None, "not_found", 0


def extract_time_aligned_numeric(da, n_expected):
    """
    Extract a numeric variable aligned with the flattened time axis.

    Handles common legacy Saildrone shapes:
        (N,)
        (1, N)
        (N, 1)

    The decisive criterion is TOTAL SIZE == flattened time size, not da.ndim.
    """
    raw = np.asarray(da.values)

    if raw.size != int(n_expected):
        return None

    try:
        arr = raw.reshape(-1).astype(np.float64)
    except Exception:
        return None

    # xarray normally decodes _FillValue/missing_value to NaN, but keep a
    # defensive pass because some legacy files expose raw sentinels.
    sentinels = []
    for key in ["_FillValue", "missing_value"]:
        if key in da.attrs:
            try:
                sentinels.append(float(da.attrs[key]))
            except Exception:
                pass

    for sentinel in sentinels:
        if np.isfinite(sentinel):
            arr[np.isclose(arr, sentinel, rtol=0.0, atol=0.0)] = np.nan

    # Common Saildrone fill values are around +/-1e34.
    arr[np.abs(arr) > 1e30] = np.nan

    return arr


def classify_wind_semantics(da, canonical):
    text = metadata_text(da)
    low = text.lower()

    earth_keywords = [
        "earth relative",
        "earth-relative",
        "true wind",
        "motion corrected",
        "motion-corrected",
        "eastward_wind",
        "northward_wind",
    ]

    apparent_keywords = [
        "apparent wind",
        "relative wind",
        "relative to platform",
        "ship-relative",
        "body-relative",
    ]

    earth_hits = [x for x in earth_keywords if x in low]
    apparent_hits = [x for x in apparent_keywords if x in low]

    standard_name = str(
        da.attrs.get("standard_name", "")
    ).lower()

    if canonical == "U" and standard_name == "eastward_wind":
        earth_hits.append("standard_name=eastward_wind")

    if canonical == "V" and standard_name == "northward_wind":
        earth_hits.append("standard_name=northward_wind")

    if earth_hits and not apparent_hits:
        cls = "earth_relative_or_geographic_component_supported_by_metadata"
    elif apparent_hits and not earth_hits:
        cls = "apparent_or_platform_relative_supported_by_metadata"
    elif earth_hits and apparent_hits:
        cls = "mixed_or_ambiguous_metadata"
    else:
        cls = "undetermined_from_metadata"

    return {
        "classification": cls,
        "earth_relative_evidence": "; ".join(sorted(set(earth_hits))),
        "apparent_relative_evidence": "; ".join(
            sorted(set(apparent_hits))
        ),
        "metadata_text": text,
    }


def discover_files(raw_dir: Path):
    """
    ALWAYS recursive.

    v0.1 only used rglob when no top-level .nc existed, which would miss the
    RawData\1026 segmented West-Coast mission. v0.2 fixes that.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)

    files = sorted(
        p
        for p in raw_dir.rglob("*.nc")
        if p.is_file()
    )

    if not files:
        raise FileNotFoundError(
            f"No .nc files found recursively under {raw_dir}"
        )

    return files


def consolidate_mission_frames(frames: List[pd.DataFrame]):
    """
    Concatenate segment rows and consolidate duplicate timestamps.

    pandas groupby.first() returns the first non-null value per variable,
    which is useful if overlapping segment files carry complementary values.
    """
    if not frames:
        return pd.DataFrame(
            columns=["time_ns"] + AUDIT_VARIABLES
        )

    df = pd.concat(
        frames,
        ignore_index=True,
        copy=False,
    )

    df = df.sort_values(
        "time_ns",
        kind="mergesort",
    ).reset_index(drop=True)

    duplicate_rows = int(
        df["time_ns"].duplicated(keep=False).sum()
    )

    unique_before = int(
        df["time_ns"].nunique()
    )

    if duplicate_rows:
        # Numeric NaNs are naturally skipped by first().
        df = (
            df.groupby(
                "time_ns",
                as_index=False,
                sort=True,
            )
            .first()
        )

    df = df.sort_values(
        "time_ns"
    ).reset_index(drop=True)

    return df, duplicate_rows, unique_before


def temporal_audit(time_ns):
    time_ns = np.asarray(time_ns, dtype=np.int64)

    if len(time_ns) < 2:
        return {
            "rows": int(len(time_ns)),
            "median_dt_seconds": np.nan,
            "mode_dt_seconds": np.nan,
            "one_minute_fraction": np.nan,
            "strictly_increasing": True,
            "gap_gt_1min_count": 0,
            "max_gap_minutes": np.nan,
        }

    diff_s = np.diff(time_ns) / 1e9
    positive = diff_s[diff_s > 0]
    vals, counts = np.unique(diff_s, return_counts=True)

    return {
        "rows": int(len(time_ns)),
        "median_dt_seconds": (
            float(np.median(positive))
            if len(positive)
            else np.nan
        ),
        "mode_dt_seconds": (
            float(vals[np.argmax(counts)])
            if len(vals)
            else np.nan
        ),
        "one_minute_fraction": float(
            np.mean(diff_s == 60.0)
        ),
        "strictly_increasing": bool(
            np.all(diff_s > 0)
        ),
        "gap_gt_1min_count": int(
            np.sum(diff_s > 60.0)
        ),
        "max_gap_minutes": float(
            np.max(diff_s) / 60.0
        ),
    }


def longest_missing_run(valid, time_ns):
    best_count = 0
    best_start = None
    best_end = None
    start = None

    for i, ok in enumerate(valid):
        if not ok:
            if start is None:
                start = i
        elif start is not None:
            count = i - start
            if count > best_count:
                best_count = count
                best_start = time_ns[start]
                best_end = time_ns[i - 1]
            start = None

    if start is not None:
        count = len(valid) - start
        if count > best_count:
            best_count = count
            best_start = time_ns[start]
            best_end = time_ns[-1]

    return (
        best_count,
        best_start,
        best_end,
    )


def count_ten_min_blocks(df: pd.DataFrame):
    if df.empty:
        return {
            "total_exact_time_blocks": 0,
            "uv_complete_blocks": 0,
            "paper7_complete_blocks": 0,
            "our_preferred_complete_blocks": 0,
            "our_preferred_schema_available": False,
        }

    time_ns = df["time_ns"].to_numpy(dtype=np.int64)

    block_ns = 10 * 60 * 1_000_000_000
    minute_ns = 60 * 1_000_000_000

    starts = (time_ns // block_ns) * block_ns

    unique, first, counts = np.unique(
        starts,
        return_index=True,
        return_counts=True,
    )

    exact = 0
    uv = 0
    paper7 = 0
    ours = 0

    ours_schema = all(
        k in df.columns
        and bool(df[k].notna().any())
        for k in OUR_PREFERRED
    )

    for b, s, c in zip(
        unique,
        first,
        counts,
    ):
        if int(c) != 10:
            continue

        s = int(s)

        actual = time_ns[s:s + 10]

        expected = (
            int(b)
            + np.arange(
                10,
                dtype=np.int64,
            )
            * minute_ns
        )

        if not np.array_equal(
            actual,
            expected,
        ):
            continue

        exact += 1

        def complete(keys):
            for key in keys:
                if key not in df.columns:
                    return False

                arr = df[key].iloc[
                    s:s + 10
                ].to_numpy(
                    dtype=np.float64
                )

                if not np.isfinite(arr).all():
                    return False

            return True

        uv += int(
            complete(["U", "V"])
        )

        paper7 += int(
            complete(PAPER_BASE)
        )

        if ours_schema:
            ours += int(
                complete(OUR_PREFERRED)
            )

    return {
        "total_exact_time_blocks": int(exact),
        "uv_complete_blocks": int(uv),
        "paper7_complete_blocks": int(paper7),
        "our_preferred_complete_blocks": int(ours),
        "our_preferred_schema_available": bool(
            ours_schema
        ),
    }


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
        help="Print progress every N NetCDF files.",
    )

    args = parser.parse_args()

    project_root = Path(
        args.project_root
    )

    raw_dir = (
        Path(args.raw_dir)
        if args.raw_dir
        else project_root
        / "data"
        / "raw"
        / "RawData"
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root
        / "data"
        / "forecasting"
        / "12A_Nie_BPSTGNN_raw_dataset_audit_v0_3"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    xr = import_xarray()

    files = discover_files(
        raw_dir
    )

    # Pre-group before opening files.
    grouped_paths = defaultdict(list)

    for path in files:
        grouped_paths[
            mission_key(path)
        ].append(path)

    log("=" * 122)
    log("12A v0.3 - NIE/BP-STGNN RAW DATASET DIMENSION-ROBUST MISSION AUDIT")
    log("=" * 122)
    log(f"script version        : {SCRIPT_VERSION}")
    log(f"raw dir               : {raw_dir}")
    log(f"recursive .nc count   : {len(files)}")
    log(f"mission groups        : {dict((k, len(v)) for k, v in grouped_paths.items())}")
    log("model training        : NONE")
    log("interpolation         : NONE")
    log("")

    segment_rows = []
    mapping_rows = []
    semantics_rows = []
    raw_schema_rows = []

    mission_frames = defaultdict(list)

    # Track whether a canonical variable was ever found in each mission.
    mission_mapping_names = defaultdict(
        lambda: defaultdict(set)
    )

    for mission, mission_paths in sorted(
        grouped_paths.items()
    ):
        log("-" * 122)
        log(
            f"[MISSION] {mission} | "
            f"NetCDF segments={len(mission_paths)}"
        )

        for local_idx, path in enumerate(
            sorted(mission_paths),
            start=1,
        ):
            if (
                local_idx == 1
                or local_idx % max(
                    1,
                    args.progress_every,
                )
                == 0
                or local_idx == len(mission_paths)
            ):
                log(
                    f"  scanning {local_idx}/{len(mission_paths)}: "
                    f"{path.name}"
                )

            platform = infer_platform(path)
            region = infer_region(path)

            try:
                with xr.open_dataset(
                    path,
                    decode_times=True,
                ) as ds:
                    if "time" not in ds.variables:
                        segment_rows.append({
                            "mission": mission,
                            "region_guess": region,
                            "platform_id": platform,
                            "file": path.name,
                            "relative_path": str(
                                path.relative_to(raw_dir)
                            ),
                            "time_found": False,
                            "rows": 0,
                            "start": "",
                            "end": "",
                            "error": "no_time_variable",
                        })
                        continue

                    # Raw schema probe: capture all variables for the first
                    # segment of each mission. This makes future resolver
                    # problems directly diagnosable from CSV.
                    if local_idx == 1:
                        for variable_name in ds.variables:
                            da_probe = ds[variable_name]
                            raw_schema_rows.append({
                                "mission": mission,
                                "file": path.name,
                                "variable": str(variable_name),
                                "dims": ",".join(str(x) for x in da_probe.dims),
                                "shape": str(tuple(int(x) for x in da_probe.shape)),
                                "size": int(da_probe.size),
                                "dtype": str(da_probe.dtype),
                                "standard_name": str(
                                    da_probe.attrs.get("standard_name", "")
                                ),
                                "long_name": str(
                                    da_probe.attrs.get("long_name", "")
                                ),
                                "units": str(
                                    da_probe.attrs.get("units", "")
                                ),
                            })

                    raw_time = normalize_time(
                        ds["time"].values
                    )

                    valid_time = ~pd.isna(
                        pd.to_datetime(
                            raw_time
                        )
                    )

                    time_ns = (
                        raw_time[
                            valid_time
                        ]
                        .astype(
                            "datetime64[ns]"
                        )
                        .astype(
                            np.int64
                        )
                    )

                    if len(time_ns) == 0:
                        segment_rows.append({
                            "mission": mission,
                            "region_guess": region,
                            "platform_id": platform,
                            "file": path.name,
                            "relative_path": str(
                                path.relative_to(raw_dir)
                            ),
                            "time_found": True,
                            "rows": 0,
                            "start": "",
                            "end": "",
                            "error": "no_valid_timestamp",
                        })
                        continue

                    order = np.argsort(
                        time_ns
                    )

                    time_ns = time_ns[
                        order
                    ]

                    frame = pd.DataFrame({
                        "time_ns": time_ns,
                    })

                    variable_count = 0

                    for canonical in AUDIT_VARIABLES:
                        raw_name, method, score = resolve_variable(
                            ds,
                            canonical,
                        )

                        if raw_name is None:
                            frame[
                                canonical
                            ] = np.nan

                            mapping_rows.append({
                                "mission": mission,
                                "file": path.name,
                                "canonical": canonical,
                                "found": False,
                                "raw_variable": "",
                                "resolution_method": method,
                                "resolution_score": score,
                                "units": "",
                                "standard_name": "",
                                "long_name": "",
                            })

                            continue

                        da = ds[
                            raw_name
                        ]

                        arr = extract_time_aligned_numeric(
                            da,
                            np.asarray(ds["time"].values).size,
                        )

                        if arr is None:
                            frame[
                                canonical
                            ] = np.nan

                            mapping_rows.append({
                                "mission": mission,
                                "file": path.name,
                                "canonical": canonical,
                                "found": False,
                                "raw_variable": raw_name,
                                "resolution_method": (
                                    method
                                    + "_time_size_mismatch"
                                ),
                                "resolution_score": score,
                                "units": str(
                                    da.attrs.get(
                                        "units",
                                        "",
                                    )
                                ),
                                "standard_name": str(
                                    da.attrs.get(
                                        "standard_name",
                                        "",
                                    )
                                ),
                                "long_name": str(
                                    da.attrs.get(
                                        "long_name",
                                        "",
                                    )
                                ),
                            })

                            continue

                        arr = arr[
                            valid_time
                        ][
                            order
                        ]

                        frame[
                            canonical
                        ] = arr

                        variable_count += 1

                        mission_mapping_names[
                            mission
                        ][
                            canonical
                        ].add(
                            raw_name
                        )

                        mapping_rows.append({
                            "mission": mission,
                            "file": path.name,
                            "canonical": canonical,
                            "found": True,
                            "raw_variable": raw_name,
                            "resolution_method": method,
                            "resolution_score": score,
                            "units": str(
                                da.attrs.get(
                                    "units",
                                    "",
                                )
                            ),
                            "standard_name": str(
                                da.attrs.get(
                                    "standard_name",
                                    "",
                                )
                            ),
                            "long_name": str(
                                da.attrs.get(
                                    "long_name",
                                    "",
                                )
                            ),
                        })

                        if canonical in [
                            "U",
                            "V",
                        ]:
                            semantics_rows.append({
                                "mission": mission,
                                "file": path.name,
                                "canonical": canonical,
                                "raw_variable": raw_name,
                                **classify_wind_semantics(
                                    da,
                                    canonical,
                                ),
                            })

                    mission_frames[
                        mission
                    ].append(
                        frame
                    )

                    segment_rows.append({
                        "mission": mission,
                        "region_guess": region,
                        "platform_id": platform,
                        "file": path.name,
                        "relative_path": str(
                            path.relative_to(raw_dir)
                        ),
                        "time_found": True,
                        "rows": int(
                            len(
                                time_ns
                            )
                        ),
                        "start": ns_iso(
                            time_ns[
                                0
                            ]
                        ),
                        "end": ns_iso(
                            time_ns[
                                -1
                            ]
                        ),
                        "resolved_variable_count": int(
                            variable_count
                        ),
                        "error": "",
                    })

            except Exception as exc:
                segment_rows.append({
                    "mission": mission,
                    "region_guess": region,
                    "platform_id": platform,
                    "file": path.name,
                    "relative_path": str(
                        path.relative_to(raw_dir)
                    ),
                    "time_found": False,
                    "rows": 0,
                    "start": "",
                    "end": "",
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                })

                log(
                    f"    [WARN] failed segment: "
                    f"{type(exc).__name__}: {exc}"
                )

    log("")
    log("=" * 122)
    log("MISSION-LEVEL CONSOLIDATION")
    log("=" * 122)

    mission_inventory_rows = []
    mission_variable_rows = []
    mission_temporal_rows = []
    block_rows = []
    paper_rows = []
    transfer_rows = []
    mission_summary_rows = []

    for mission, frames in sorted(
        mission_frames.items()
    ):
        log(
            f"[CONSOLIDATE] {mission}: frames={len(frames)}"
        )

        consolidated, duplicate_rows, unique_before = (
            consolidate_mission_frames(
                frames
            )
        )

        if consolidated.empty:
            continue

        time_ns = consolidated[
            "time_ns"
        ].to_numpy(
            dtype=np.int64
        )

        ta = temporal_audit(
            time_ns
        )

        blocks = count_ten_min_blocks(
            consolidated
        )

        region = (
            mission
            if mission in PAPER_TABLE1_COUNTS
            else "Unknown"
        )

        platforms = sorted({
            infer_platform(p)
            for p in grouped_paths[
                mission
            ]
            if infer_platform(p)
        })

        mission_inventory_rows.append({
            "mission": mission,
            "region": region,
            "platform_ids": ",".join(
                platforms
            ),
            "segment_file_count": int(
                len(
                    grouped_paths[
                        mission
                    ]
                )
            ),
            "concatenated_rows_before_dedup": int(
                sum(
                    len(
                        frame
                    )
                    for frame in frames
                )
            ),
            "duplicate_rows_detected": int(
                duplicate_rows
            ),
            "unique_timestamps_before_consolidation": int(
                unique_before
            ),
            "mission_rows_after_consolidation": int(
                len(
                    consolidated
                )
            ),
            "start": ns_iso(
                time_ns[
                    0
                ]
            ),
            "end": ns_iso(
                time_ns[
                    -1
                ]
            ),
        })

        mission_temporal_rows.append({
            "mission": mission,
            "region": region,
            "start": ns_iso(
                time_ns[
                    0
                ]
            ),
            "end": ns_iso(
                time_ns[
                    -1
                ]
            ),
            **ta,
        })

        for canonical in AUDIT_VARIABLES:
            arr = consolidated[
                canonical
            ].to_numpy(
                dtype=np.float64
            )

            finite = np.isfinite(
                arr
            )

            nrun, g0, g1 = longest_missing_run(
                finite,
                time_ns,
            )

            mission_variable_rows.append({
                "mission": mission,
                "region": region,
                "canonical": canonical,
                "raw_variable_names_seen": ",".join(
                    sorted(
                        mission_mapping_names[
                            mission
                        ][
                            canonical
                        ]
                    )
                ),
                "rows": int(
                    len(
                        arr
                    )
                ),
                "finite_rows": int(
                    finite.sum()
                ),
                "missing_rows": int(
                    (
                        ~finite
                    ).sum()
                ),
                "valid_fraction": float(
                    np.mean(
                        finite
                    )
                ),
                "longest_missing_run_rows": int(
                    nrun
                ),
                "longest_missing_start": (
                    ns_iso(
                        g0
                    )
                    if g0 is not None
                    else ""
                ),
                "longest_missing_end": (
                    ns_iso(
                        g1
                    )
                    if g1 is not None
                    else ""
                ),
            })

        block_rows.append({
            "mission": mission,
            "region": region,
            "segment_file_count": int(
                len(
                    grouped_paths[
                        mission
                    ]
                )
            ),
            **blocks,
        })

        expected = PAPER_TABLE1_COUNTS.get(
            region
        )

        paper_rows.append({
            "mission": mission,
            "region": region,
            "paper_expected_sample_count": expected,
            "mission_rows_after_consolidation": int(
                len(
                    consolidated
                )
            ),
            "raw_minus_paper_count": (
                int(
                    len(
                        consolidated
                    )
                    - expected
                )
                if expected is not None
                else np.nan
            ),
            "raw_to_paper_ratio": (
                float(
                    len(
                        consolidated
                    )
                    / expected
                )
                if expected not in [
                    None,
                    0,
                ]
                else np.nan
            ),
            "likely_paper_role": (
                "held_out_test_candidate"
                if region
                == "Tropical Atlantic"
                else "train_or_validation_candidate"
                if region in [
                    "Antarctic",
                    "Atlantic",
                    "West Coast",
                ]
                else "unknown"
            ),
            "important_note": (
                "Raw mission rows are not expected to equal Table-1 rows "
                "exactly because the paper may apply QC, interval selection, "
                "missing-data treatment, and 10-min resampling."
            ),
        })

        present = {
            canonical: bool(
                consolidated[
                    canonical
                ].notna().any()
            )
            for canonical in [
                "U",
                "V",
                "SOG",
                "COG",
                "HDG",
                "WING_ANGLE",
                "T",
                "RH",
                "P",
            ]
        }

        paper7_ok = all(
            present.get(
                k,
                False,
            )
            for k in PAPER_BASE
        )

        ours_ok = all(
            present.get(
                k,
                False,
            )
            for k in OUR_PREFERRED
        )

        missing_ours = [
            k
            for k in OUR_PREFERRED
            if not present.get(
                k,
                False,
            )
        ]

        transfer_rows.append({
            "mission": mission,
            "region": region,
            **{
                f"{k}_available": v
                for k, v in present.items()
            },
            "paper7_all_present": bool(
                paper7_ok
            ),
            "our_preferred_all_present": bool(
                ours_ok
            ),
            "missing_from_our_preferred": ",".join(
                missing_ours
            ),
            "transfer_status": (
                "CURRENT_INPUT_SCHEMA_POTENTIALLY_TRANSFERABLE"
                if ours_ok
                else "CURRENT_INPUT_SCHEMA_NOT_DIRECTLY_TRANSFERABLE"
            ),
        })

        mission_summary_rows.append({
            "mission": mission,
            "region": region,
            "segments": int(
                len(
                    grouped_paths[
                        mission
                    ]
                )
            ),
            "rows": int(
                len(
                    consolidated
                )
            ),
            "paper7": bool(
                paper7_ok
            ),
            "our_schema": bool(
                ours_ok
            ),
            "tenmin_UV": int(
                blocks[
                    "uv_complete_blocks"
                ]
            ),
            "tenmin_paper7": int(
                blocks[
                    "paper7_complete_blocks"
                ]
            ),
            "tenmin_ours": int(
                blocks[
                    "our_preferred_complete_blocks"
                ]
            ),
        })

        log(
            f"  {mission}: rows={len(consolidated):,} | "
            f"segments={len(grouped_paths[mission])} | "
            f"1-min fraction={100*ta['one_minute_fraction']:.2f}% | "
            f"10min UV={blocks['uv_complete_blocks']:,} | "
            f"paper7={blocks['paper7_complete_blocks']:,} | "
            f"ours={blocks['our_preferred_complete_blocks']:,}"
        )

    # Write files.
    outputs = {
        "raw_schema_probe.csv": pd.DataFrame(
            raw_schema_rows
        ),
        "segment_inventory.csv": pd.DataFrame(
            segment_rows
        ),
        "mission_inventory.csv": pd.DataFrame(
            mission_inventory_rows
        ),
        "variable_mapping_audit.csv": pd.DataFrame(
            mapping_rows
        ),
        "mission_variable_coverage.csv": pd.DataFrame(
            mission_variable_rows
        ),
        "metadata_wind_semantics_audit.csv": pd.DataFrame(
            semantics_rows
        ),
        "mission_temporal_audit.csv": pd.DataFrame(
            mission_temporal_rows
        ),
        "mission_ten_min_block_audit.csv": pd.DataFrame(
            block_rows
        ),
        "paper_region_mapping_audit.csv": pd.DataFrame(
            paper_rows
        ),
        "transferability_audit.csv": pd.DataFrame(
            transfer_rows
        ),
    }

    for filename, df in outputs.items():
        df.to_csv(
            output_dir
            / filename,
            index=False,
            encoding="utf-8-sig",
        )

    semantics_df = outputs[
        "metadata_wind_semantics_audit.csv"
    ]

    if semantics_df.empty:
        semantics_conclusion = "undetermined"
    else:
        # Use unique mission/canonical classifications.
        classes = set(
            semantics_df[
                "classification"
            ].astype(
                str
            )
        )

        if classes == {
            "earth_relative_or_geographic_component_supported_by_metadata"
        }:
            semantics_conclusion = (
                "all_resolved_UV_metadata_support_geographic_or_"
                "earth_relative_components"
            )

        elif (
            "apparent_or_platform_relative_supported_by_metadata"
            in classes
        ):
            semantics_conclusion = (
                "at_least_one_resolved_UV_variable_looks_"
                "apparent_or_platform_relative"
            )

        else:
            semantics_conclusion = (
                "mixed_or_incomplete_metadata_evidence"
            )

    regions_found = sorted({
        row[
            "region"
        ]
        for row in mission_summary_rows
        if row[
            "region"
        ]
        != "Unknown"
    })

    missing_regions = sorted(
        set(
            PAPER_TABLE1_COUNTS
        )
        - set(
            regions_found
        )
    )

    known_single_presence = {
        region: bool(
            any(
                p.name
                == filename
                for p in files
            )
        )
        for region, filename
        in KNOWN_SINGLE_FILES.items()
    }

    west_coast_file_count = sum(
        1
        for p in files
        if infer_region(
            p
        )
        == "West Coast"
    )

    manifest = {
        "stage": "12A",
        "script_version": SCRIPT_VERSION,
        "raw_dir": str(
            raw_dir
        ),
        "recursive_netcdf_file_count": int(
            len(
                files
            )
        ),
        "mission_file_counts": {
            key: len(
                value
            )
            for key, value
            in grouped_paths.items()
        },
        "known_single_file_presence": known_single_presence,
        "west_coast_segment_count": int(
            west_coast_file_count
        ),
        "paper_table1_counts_reference": PAPER_TABLE1_COUNTS,
        "regions_found": regions_found,
        "paper_regions_missing": missing_regions,
        "wind_semantics_metadata_conclusion": semantics_conclusion,
        "policy": {
            "model_training": False,
            "interpolation": False,
            "mission_level_consolidation": True,
            "duplicate_timestamp_policy": (
                "groupby timestamp and take first non-null value per variable"
            ),
            "semantic_classification_uses_metadata_only": True,
        },
    }

    save_json(
        output_dir
        / "audit_manifest.json",
        manifest,
    )

    with (
        output_dir
        / "audit_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "12A v0.3 Nie/BP-STGNN Raw Dataset Mission Audit\n"
        )
        f.write(
            "="
            * 122
            + "\n\n"
        )
        f.write(
            "NO MODEL TRAINING WAS PERFORMED.\n"
        )
        f.write(
            "NO INTERPOLATION WAS PERFORMED.\n\n"
        )

        f.write(
            "MISSION SUMMARY\n"
        )
        f.write(
            "-"
            * 122
            + "\n"
        )
        f.write(
            pd.DataFrame(
                mission_summary_rows
            ).to_string(
                index=False
            )
        )

        f.write(
            "\n\nPAPER REGION MAPPING\n"
        )
        f.write(
            "-"
            * 122
            + "\n"
        )
        f.write(
            outputs[
                "paper_region_mapping_audit.csv"
            ].to_string(
                index=False
            )
        )

        f.write(
            "\n\nTRANSFERABILITY\n"
        )
        f.write(
            "-"
            * 122
            + "\n"
        )
        f.write(
            outputs[
                "transferability_audit.csv"
            ].to_string(
                index=False
            )
        )

        f.write(
            "\n\nWIND SEMANTICS FROM METADATA\n"
        )
        f.write(
            "-"
            * 122
            + "\n"
        )
        f.write(
            f"Overall conclusion: {semantics_conclusion}\n"
        )

        if not semantics_df.empty:
            dedup = semantics_df.drop_duplicates(
                subset=[
                    "mission",
                    "canonical",
                    "raw_variable",
                    "classification",
                ]
            )

            f.write(
                dedup[
                    [
                        "mission",
                        "canonical",
                        "raw_variable",
                        "classification",
                        "earth_relative_evidence",
                        "apparent_relative_evidence",
                    ]
                ].to_string(
                    index=False
                )
            )

        f.write(
            "\n"
        )

    log("")
    log("=" * 122)
    log("12A v0.3 AUDIT SUMMARY")
    log("=" * 122)

    for row in mission_summary_rows:
        log(
            f"{row['mission']}: "
            f"segments={row['segments']} | "
            f"rows={row['rows']:,} | "
            f"paper7={'YES' if row['paper7'] else 'NO'} | "
            f"our-schema={'YES' if row['our_schema'] else 'NO'} | "
            f"10min UV={row['tenmin_UV']:,} | "
            f"paper7={row['tenmin_paper7']:,} | "
            f"ours={row['tenmin_ours']:,}"
        )

    log("")
    log(
        f"West Coast recursive segments found: "
        f"{west_coast_file_count}"
    )
    log(
        f"Wind semantics from metadata: "
        f"{semantics_conclusion}"
    )

    core_resolved_any = any(
        row.get("canonical") in ["U", "V"]
        and bool(row.get("found"))
        for row in mapping_rows
    )

    if not core_resolved_any:
        log(
            "[ERROR-AUDIT] U/V still unresolved after dimension fix. "
            "Inspect raw_schema_probe.csv before any dataset construction."
        )
    else:
        log(
            "[PASS] At least one mission resolved U/V after dimension-robust extraction."
        )

    if missing_regions:
        log(
            f"[WARN] Paper regions still missing after recursive scan: "
            f"{missing_regions}"
        )
    else:
        log(
            "[PASS] All four paper regional groups have mission candidates."
        )

    if west_coast_file_count == 0:
        log(
            "[WARN] No West Coast / SD1026 files were recognized."
        )
    else:
        log(
            "[PASS] RawData\\1026 was included as West Coast / SD1026."
        )

    log(
        "[POLICY] No model training and no interpolation were performed."
    )
    log(
        f"[DONE] 12A v0.3 outputs: {output_dir}"
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
