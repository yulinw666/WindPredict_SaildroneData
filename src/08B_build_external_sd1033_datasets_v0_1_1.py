# -*- coding: utf-8 -*-
"""
08B_build_external_sd1033_datasets.py

Build frozen external-generalization datasets from SD1033 TPOS 2022/2023/2024.

IMPORTANT: this is an inference-dataset builder, NOT a training stage.

It performs:
    raw NetCDF -> frozen 11-feature schema -> frozen SD1090 TRAIN scaling
    -> external forecasting windows.

It NEVER:
    - fits or refits a scaler on SD1033;
    - interpolates missing values;
    - trains a model;
    - selects hyperparameters;
    - creates train/validation/test splits on SD1033.

External products
-----------------
1. SD1033_2024_full.npz
2. SD1033_2024_matched_SD1090.npz
3. SD1033_2023_full.npz
4. SD1033_2022_full.npz

The "full" products use ALL continuous model-ready segments with length >= 70
samples. Windows are built strictly inside each segment and never cross a gap.

The "matched" 2024 product uses ALL continuous timestamps that are model-ready
for BOTH SD1033-2024 and SD1090-2024, again requiring >= 70 continuous 1-min
samples. This is the same-mission / same-calendar-period / different-vehicle
external test.

Frozen temporal protocol
------------------------
lookback = 60 min
horizons = [1, 2, 3, 5, 10] min

Frozen engineered feature schema
---------------------------------
[
    UWND_MEAN,
    VWND_MEAN,
    TEMP_AIR_MEAN,
    RH_MEAN,
    SOG,
    COG_sin,
    COG_cos,
    HDG_sin,
    HDG_cos,
    WING_ANGLE_sin,
    WING_ANGLE_cos,
]

Frozen target schema
--------------------
[
    UWND_MEAN,
    VWND_MEAN,
    VESSEL_EAST_MPS,
    VESSEL_NORTH_MPS,
    HDG_sin,
    HDG_cos,
]

with:
    VESSEL_EAST_MPS  = SOG * sin(COG)
    VESSEL_NORTH_MPS = SOG * cos(COG)

and external apparent wind:
    APPARENT_EAST  = UWND_MEAN - VESSEL_EAST_MPS
    APPARENT_NORTH = VWND_MEAN - VESSEL_NORTH_MPS

Frozen scaling
--------------
The feature and target scalers are loaded ONLY from the frozen SD1090
development dataset:

    <frozen-dataset-dir>/scalers.json

No SD1033 statistics are used to fit preprocessing.

External distribution-shift audit
---------------------------------
For each engineered feature under the frozen SD1090 scaler, the script reports:
    mean(z), std(z), median(z), p01(z), p99(z),
    mean(|z|), fraction(|z| > 2), fraction(|z| > 3)

computed on unique source rows that participate in eligible external segments.

NPZ fields
----------
X                       float32 [N,60,11] standardized by SD1090 TRAIN scaler
y                       float32 [N,5,6] standardized by SD1090 TRAIN scaler
y_raw                   float32 [N,5,6]
apparent_earth_raw      float32 [N,5,2]
context_start_time_ns   int64   [N]
context_end_time_ns     int64   [N]
target_time_ns          int64   [N,5]
context_start_row       int64   [N]
context_end_row         int64   [N]
target_row              int64   [N,5]
segment_id              int32   [N]
horizons_min            int64   [5]
feature_names           unicode [11]
target_names            unicode [6]
source_file              unicode scalar
dataset_id               unicode scalar
scaler_source            unicode scalar

Dependencies
------------
Place this file in the same src directory as:
    08A_audit_external_sd1033.py
    06_run_physics_constrained_joint_gru.py

The 08A script supplies the already-tested NetCDF variable resolution/audit
logic. The 06 script supplies the frozen scaler parser so the exact scaler
schema stays consistent with the existing pipeline.

v0.1.1 fix
----------
Corrects the target-time validation check. v0.1.0 used np.array_equal on
[N, H] and [1, H] arrays; np.array_equal does not broadcast, so a valid
dataset was incorrectly rejected. The corrected check uses broadcasted
elementwise comparison with np.all().

Outputs
-------
external_dataset_summary.csv
external_segment_summary.csv
feature_shift_summary.csv
external_dataset_validation.csv
external_dataset_manifest.json
external_dataset_report.txt

datasets/
    SD1033_2024_full.npz
    SD1033_2024_matched_SD1090.npz
    SD1033_2023_full.npz
    SD1033_2022_full.npz

figures/ [optional]
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.1-external-dataset-builder"
SHARED_08A = "08A_audit_external_sd1033.py"
SHARED_06 = "06_run_physics_constrained_joint_gru.py"

DEFAULT_RAW_FILES = {
    "SD1033_2024_full": (
        r"D:\project\WindPredict_SaildroneData\data\raw"
        r"\TPOS-2024_SD1033_1min.nc"
    ),
    "SD1033_2023_full": (
        r"D:\project\WindPredict_SaildroneData\data\raw"
        r"\TPOS-2023_SD1033_1min.nc"
    ),
    "SD1033_2022_full": (
        r"D:\project\WindPredict_SaildroneData\data\raw"
        r"\TPOS-2022_SD1033_1min.nc"
    ),
}

DEFAULT_REFERENCE_SD1090 = (
    r"D:\project\WindPredict_SaildroneData\data\raw"
    r"\TPOS-2024_SD1090_1min.nc"
)

DEFAULT_FROZEN_DATASET_DIR = (
    r"D:\project\WindPredict_SaildroneData\data\forecasting"
    r"\SD1090_TPOS2024_JointForecasting_v0_1"
)

LOOKBACK = 60
HORIZONS = np.asarray(
    [1, 2, 3, 5, 10],
    dtype=np.int64,
)
MAX_HORIZON = int(
    HORIZONS.max()
)
MIN_SEGMENT_ROWS = (
    LOOKBACK
    + MAX_HORIZON
)

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

REQUIRED_RAW = [
    "UWND_MEAN",
    "VWND_MEAN",
    "TEMP_AIR_MEAN",
    "RH_MEAN",
    "SOG",
    "COG",
    "HDG",
    "WING_ANGLE",
]


def log(message=""):
    print(message, flush=True)


def import_module_from_neighbor(
    filename: str,
    module_name: str,
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
            f"Cannot import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        spec.name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
        while True:
            block = f.read(
                chunk_size
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def save_json(
    path: Path,
    obj,
):
    def convert(
        value
    ):
        if isinstance(
            value,
            Path,
        ):
            return str(
                value
            )

        if isinstance(
            value,
            np.ndarray,
        ):
            return [
                convert(
                    v
                )
                for v in value.tolist()
            ]

        if isinstance(
            value,
            (
                np.integer,
            ),
        ):
            return int(
                value
            )

        if isinstance(
            value,
            (
                np.floating,
            ),
        ):
            if np.isnan(
                value
            ):
                return None

            return float(
                value
            )

        if isinstance(
            value,
            (
                np.bool_,
            ),
        ):
            return bool(
                value
            )

        if isinstance(
            value,
            dict,
        ):
            return {
                str(
                    k
                ): convert(
                    v
                )
                for k, v in (
                    value.items()
                )
            }

        if isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            return [
                convert(
                    v
                )
                for v in value
            ]

        return value

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            convert(
                obj
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )


def default_output_dir(
    frozen_dataset_dir: Path,
) -> Path:
    # .../data/forecasting/frozen -> .../data/external/...
    data_dir = (
        frozen_dataset_dir
        .parent
        .parent
    )

    return (
        data_dir
        / "external"
        / "SD1033_external_forecasting_v0_1"
    )


def load_frozen_scalers(
    core06,
    frozen_dataset_dir: Path,
):
    manifest_path = (
        frozen_dataset_dir
        / "dataset_manifest.json"
    )

    scalers_path = (
        frozen_dataset_dir
        / "scalers.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            manifest_path
        )

    if not scalers_path.exists():
        raise FileNotFoundError(
            scalers_path
        )

    manifest = core06.load_json(
        manifest_path
    )

    scalers = core06.load_json(
        scalers_path
    )

    manifest_features = list(
        manifest[
            "feature_names"
        ]
    )

    manifest_targets = list(
        manifest[
            "target_names"
        ]
    )

    manifest_horizons = [
        int(
            x
        )
        for x in manifest[
            "forecast_horizons_minutes"
        ]
    ]

    if (
        manifest_features
        != FEATURE_NAMES
    ):
        raise RuntimeError(
            "Frozen feature schema mismatch.\n"
            f"Expected: {FEATURE_NAMES}\n"
            f"Found:    {manifest_features}"
        )

    if (
        manifest_targets
        != TARGET_NAMES
    ):
        raise RuntimeError(
            "Frozen target schema mismatch.\n"
            f"Expected: {TARGET_NAMES}\n"
            f"Found:    {manifest_targets}"
        )

    if (
        manifest_horizons
        != HORIZONS.tolist()
    ):
        raise RuntimeError(
            "Frozen horizon schema mismatch."
        )

    (
        feature_scaler_names,
        feature_mean,
        feature_std,
    ) = core06.parse_scaler(
        scalers,
        "feature_scaler",
    )

    (
        target_scaler_names,
        target_mean,
        target_std,
    ) = core06.parse_scaler(
        scalers,
        "target_scaler",
    )

    if (
        feature_scaler_names
        != FEATURE_NAMES
    ):
        raise RuntimeError(
            "Frozen feature scaler name mismatch."
        )

    if (
        target_scaler_names
        != TARGET_NAMES
    ):
        raise RuntimeError(
            "Frozen target scaler name mismatch."
        )

    feature_mean = np.asarray(
        feature_mean,
        dtype=np.float64,
    )

    feature_std = np.asarray(
        feature_std,
        dtype=np.float64,
    )

    target_mean = np.asarray(
        target_mean,
        dtype=np.float64,
    )

    target_std = np.asarray(
        target_std,
        dtype=np.float64,
    )

    if np.any(
        ~np.isfinite(
            feature_mean
        )
    ) or np.any(
        ~np.isfinite(
            feature_std
        )
    ):
        raise RuntimeError(
            "Non-finite frozen feature scaler."
        )

    if np.any(
        feature_std
        <= 0
    ):
        raise RuntimeError(
            "Non-positive frozen feature std."
        )

    if np.any(
        target_std
        <= 0
    ):
        raise RuntimeError(
            "Non-positive frozen target std."
        )

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "scalers_path": scalers_path,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }


def build_engineered_rows(
    audit,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Return:
        feature_raw [rows, 11]
        target_raw  [rows,  6]

    NaNs are preserved outside model-ready segments. No interpolation.
    """
    arrays = audit[
        "arrays"
    ]

    for key in REQUIRED_RAW:
        if key not in arrays:
            raise RuntimeError(
                f"{audit['label']}: missing required raw array {key}"
            )

    u = np.asarray(
        arrays[
            "UWND_MEAN"
        ],
        dtype=np.float64,
    )

    v = np.asarray(
        arrays[
            "VWND_MEAN"
        ],
        dtype=np.float64,
    )

    temp = np.asarray(
        arrays[
            "TEMP_AIR_MEAN"
        ],
        dtype=np.float64,
    )

    rh = np.asarray(
        arrays[
            "RH_MEAN"
        ],
        dtype=np.float64,
    )

    sog = np.asarray(
        arrays[
            "SOG"
        ],
        dtype=np.float64,
    )

    cog = np.asarray(
        arrays[
            "COG"
        ],
        dtype=np.float64,
    )

    hdg = np.asarray(
        arrays[
            "HDG"
        ],
        dtype=np.float64,
    )

    wing = np.asarray(
        arrays[
            "WING_ANGLE"
        ],
        dtype=np.float64,
    )

    cog_rad = np.deg2rad(
        cog
    )

    hdg_rad = np.deg2rad(
        hdg
    )

    wing_rad = np.deg2rad(
        wing
    )

    vessel_e = (
        sog
        * np.sin(
            cog_rad
        )
    )

    vessel_n = (
        sog
        * np.cos(
            cog_rad
        )
    )

    feature_raw = np.column_stack(
        [
            u,
            v,
            temp,
            rh,
            sog,
            np.sin(
                cog_rad
            ),
            np.cos(
                cog_rad
            ),
            np.sin(
                hdg_rad
            ),
            np.cos(
                hdg_rad
            ),
            np.sin(
                wing_rad
            ),
            np.cos(
                wing_rad
            ),
        ]
    ).astype(
        np.float64,
        copy=False,
    )

    target_raw = np.column_stack(
        [
            u,
            v,
            vessel_e,
            vessel_n,
            np.sin(
                hdg_rad
            ),
            np.cos(
                hdg_rad
            ),
        ]
    ).astype(
        np.float64,
        copy=False,
    )

    return (
        feature_raw,
        target_raw,
    )


def eligible_full_segments(
    audit,
    audit08a,
    step_tolerance_seconds: float,
):
    segments = (
        audit08a.contiguous_segments(
            audit[
                "ready"
            ],
            audit[
                "time"
            ],
            step_tolerance_seconds,
        )
    )

    segments = [
        (
            int(
                a
            ),
            int(
                b
            ),
        )
        for a, b in segments
        if (
            b
            - a
            + 1
        )
        >= MIN_SEGMENT_ROWS
    ]

    return segments


def eligible_matched_segments(
    sd1033,
    sd1090,
    audit08a,
    step_tolerance_seconds: float,
):
    """
    Exact intersection of model-ready timestamps in both vehicles.
    Return SD1033 source-row index intervals for every common continuous segment.
    """
    t1033_ready = pd.DatetimeIndex(
        sd1033[
            "time"
        ][
            sd1033[
                "ready"
            ]
        ]
    )

    t1090_ready = pd.DatetimeIndex(
        sd1090[
            "time"
        ][
            sd1090[
                "ready"
            ]
        ]
    )

    common = t1033_ready.intersection(
        t1090_ready
    ).sort_values()

    if len(
        common
    ) == 0:
        return []

    common_segments = (
        audit08a.contiguous_segments(
            np.ones(
                len(
                    common
                ),
                dtype=bool,
            ),
            common,
            step_tolerance_seconds,
        )
    )

    # Map every common timestamp back to the SD1033 raw row.
    sd1033_index = pd.Index(
        sd1033[
            "time"
        ]
    )

    segments = []

    for ca, cb in (
        common_segments
    ):
        length = (
            cb
            - ca
            + 1
        )

        if length < MIN_SEGMENT_ROWS:
            continue

        start_time = common[
            ca
        ]

        end_time = common[
            cb
        ]

        start_row = int(
            sd1033_index.get_loc(
                start_time
            )
        )

        end_row = int(
            sd1033_index.get_loc(
                end_time
            )
        )

        # Continuous common timestamps guarantee the raw row interval should
        # also be dense in SD1033 because 08A already verified 1-min time steps.
        if (
            end_row
            - start_row
            + 1
        ) != length:
            raise RuntimeError(
                "Matched common segment does not map to a dense "
                "SD1033 raw-row interval."
            )

        segments.append(
            (
                start_row,
                end_row,
            )
        )

    return segments


def count_windows(
    segments,
) -> int:
    return int(
        sum(
            (
                b
                - a
                + 1
            )
            - MIN_SEGMENT_ROWS
            + 1
            for a, b in segments
        )
    )


def segment_source_row_mask(
    n_rows: int,
    segments,
):
    mask = np.zeros(
        n_rows,
        dtype=bool,
    )

    for a, b in segments:
        mask[
            a:b + 1
        ] = True

    return mask


def feature_shift_rows(
    *,
    dataset_id: str,
    feature_raw: np.ndarray,
    source_row_mask: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
):
    if not np.any(
        source_row_mask
    ):
        return []

    x = feature_raw[
        source_row_mask
    ]

    z = (
        x
        - feature_mean[
            None,
            :,
        ]
    ) / feature_std[
        None,
        :,
    ]

    rows = []

    for j, name in enumerate(
        FEATURE_NAMES
    ):
        values = z[
            :,
            j,
        ]

        finite = np.isfinite(
            values
        )

        values = values[
            finite
        ]

        if len(
            values
        ) == 0:
            continue

        rows.append({
            "dataset_id": dataset_id,
            "feature": name,
            "source_rows": int(
                len(
                    values
                )
            ),
            "mean_z": float(
                np.mean(
                    values
                )
            ),
            "std_z": float(
                np.std(
                    values
                )
            ),
            "median_z": float(
                np.median(
                    values
                )
            ),
            "p01_z": float(
                np.quantile(
                    values,
                    0.01,
                )
            ),
            "p99_z": float(
                np.quantile(
                    values,
                    0.99,
                )
            ),
            "min_z": float(
                np.min(
                    values
                )
            ),
            "max_z": float(
                np.max(
                    values
                )
            ),
            "mean_abs_z": float(
                np.mean(
                    np.abs(
                        values
                    )
                )
            ),
            "fraction_abs_z_gt_2": float(
                np.mean(
                    np.abs(
                        values
                    )
                    > 2.0
                )
            ),
            "fraction_abs_z_gt_3": float(
                np.mean(
                    np.abs(
                        values
                    )
                    > 3.0
                )
            ),
        })

    return rows


def allocate_dataset_arrays(
    n_windows: int,
):
    return {
        "X": np.empty(
            (
                n_windows,
                LOOKBACK,
                len(
                    FEATURE_NAMES
                ),
            ),
            dtype=np.float32,
        ),
        "y": np.empty(
            (
                n_windows,
                len(
                    HORIZONS
                ),
                len(
                    TARGET_NAMES
                ),
            ),
            dtype=np.float32,
        ),
        "y_raw": np.empty(
            (
                n_windows,
                len(
                    HORIZONS
                ),
                len(
                    TARGET_NAMES
                ),
            ),
            dtype=np.float32,
        ),
        "apparent_earth_raw": np.empty(
            (
                n_windows,
                len(
                    HORIZONS
                ),
                2,
            ),
            dtype=np.float32,
        ),
        "context_start_time_ns": np.empty(
            n_windows,
            dtype=np.int64,
        ),
        "context_end_time_ns": np.empty(
            n_windows,
            dtype=np.int64,
        ),
        "target_time_ns": np.empty(
            (
                n_windows,
                len(
                    HORIZONS
                ),
            ),
            dtype=np.int64,
        ),
        "context_start_row": np.empty(
            n_windows,
            dtype=np.int64,
        ),
        "context_end_row": np.empty(
            n_windows,
            dtype=np.int64,
        ),
        "target_row": np.empty(
            (
                n_windows,
                len(
                    HORIZONS
                ),
            ),
            dtype=np.int64,
        ),
        "segment_id": np.empty(
            n_windows,
            dtype=np.int32,
        ),
    }


def build_one_dataset(
    *,
    dataset_id: str,
    source_file: Path,
    audit,
    segments,
    feature_raw: np.ndarray,
    target_raw: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
):
    n_windows = count_windows(
        segments
    )

    if n_windows <= 0:
        raise RuntimeError(
            f"{dataset_id}: zero eligible external windows."
        )

    log(
        f"[BUILD] {dataset_id}: "
        f"{len(segments)} eligible segments, "
        f"{n_windows:,} windows"
    )

    arrays = allocate_dataset_arrays(
        n_windows
    )

    feature_z = (
        (
            feature_raw
            - feature_mean[
                None,
                :,
            ]
        )
        / feature_std[
            None,
            :,
        ]
    ).astype(
        np.float32
    )

    target_z = (
        (
            target_raw
            - target_mean[
                None,
                :,
            ]
        )
        / target_std[
            None,
            :,
        ]
    ).astype(
        np.float32
    )

    time_ns = audit[
        "time"
    ].view(
        "int64"
    )

    segment_rows = []

    write = 0

    for seg_id, (
        seg_start,
        seg_end,
    ) in enumerate(
        segments
    ):
        seg_len = (
            seg_end
            - seg_start
            + 1
        )

        n_seg_windows = (
            seg_len
            - MIN_SEGMENT_ROWS
            + 1
        )

        start_rows = (
            seg_start
            + np.arange(
                n_seg_windows,
                dtype=np.int64,
            )
        )

        context_end_rows = (
            start_rows
            + LOOKBACK
            - 1
        )

        target_rows = (
            context_end_rows[
                :,
                None,
            ]
            + HORIZONS[
                None,
                :,
            ]
        )

        context_index = (
            start_rows[
                :,
                None,
            ]
            + np.arange(
                LOOKBACK,
                dtype=np.int64,
            )[
                None,
                :,
            ]
        )

        lo = write
        hi = (
            write
            + n_seg_windows
        )

        arrays[
            "X"
        ][
            lo:hi
        ] = feature_z[
            context_index
        ]

        arrays[
            "y"
        ][
            lo:hi
        ] = target_z[
            target_rows
        ]

        y_raw = target_raw[
            target_rows
        ].astype(
            np.float32
        )

        arrays[
            "y_raw"
        ][
            lo:hi
        ] = y_raw

        arrays[
            "apparent_earth_raw"
        ][
            lo:hi,
            :,
            0,
        ] = (
            y_raw[
                :,
                :,
                0
            ]
            - y_raw[
                :,
                :,
                2
            ]
        )

        arrays[
            "apparent_earth_raw"
        ][
            lo:hi,
            :,
            1,
        ] = (
            y_raw[
                :,
                :,
                1
            ]
            - y_raw[
                :,
                :,
                3
            ]
        )

        arrays[
            "context_start_time_ns"
        ][
            lo:hi
        ] = time_ns[
            start_rows
        ]

        arrays[
            "context_end_time_ns"
        ][
            lo:hi
        ] = time_ns[
            context_end_rows
        ]

        arrays[
            "target_time_ns"
        ][
            lo:hi
        ] = time_ns[
            target_rows
        ]

        arrays[
            "context_start_row"
        ][
            lo:hi
        ] = start_rows

        arrays[
            "context_end_row"
        ][
            lo:hi
        ] = context_end_rows

        arrays[
            "target_row"
        ][
            lo:hi
        ] = target_rows

        arrays[
            "segment_id"
        ][
            lo:hi
        ] = int(
            seg_id
        )

        segment_rows.append({
            "dataset_id": dataset_id,
            "segment_id": int(
                seg_id
            ),
            "source_start_row": int(
                seg_start
            ),
            "source_end_row": int(
                seg_end
            ),
            "source_sample_count": int(
                seg_len
            ),
            "start_time_utc": audit[
                "time"
            ][
                seg_start
            ].isoformat(),
            "end_time_utc": audit[
                "time"
            ][
                seg_end
            ].isoformat(),
            "span_days": float(
                (
                    audit[
                        "time"
                    ][
                        seg_end
                    ]
                    - audit[
                        "time"
                    ][
                        seg_start
                    ]
                ).total_seconds()
                / 86400.0
            ),
            "window_count": int(
                n_seg_windows
            ),
        })

        write = hi

        del (
            start_rows,
            context_end_rows,
            target_rows,
            context_index,
            y_raw,
        )

    if write != n_windows:
        raise RuntimeError(
            f"{dataset_id}: write count {write} != expected {n_windows}"
        )

    return (
        arrays,
        segment_rows,
    )


def validate_built_dataset(
    *,
    dataset_id: str,
    arrays,
    feature_mean,
    feature_std,
    target_mean,
    target_std,
):
    X = arrays[
        "X"
    ]

    y = arrays[
        "y"
    ]

    y_raw = arrays[
        "y_raw"
    ]

    apparent = arrays[
        "apparent_earth_raw"
    ]

    cstart = arrays[
        "context_start_time_ns"
    ]

    cend = arrays[
        "context_end_time_ns"
    ]

    ttime = arrays[
        "target_time_ns"
    ]

    target_row = arrays[
        "target_row"
    ]

    segment_id = arrays[
        "segment_id"
    ]

    n = X.shape[
        0
    ]

    checks = {}

    checks[
        "X_shape_ok"
    ] = bool(
        X.shape[
            1:
        ]
        == (
            LOOKBACK,
            len(
                FEATURE_NAMES
            ),
        )
    )

    checks[
        "y_shape_ok"
    ] = bool(
        y.shape
        == (
            n,
            len(
                HORIZONS
            ),
            len(
                TARGET_NAMES
            ),
        )
    )

    checks[
        "all_X_finite"
    ] = bool(
        np.isfinite(
            X
        ).all()
    )

    checks[
        "all_y_finite"
    ] = bool(
        np.isfinite(
            y
        ).all()
    )

    checks[
        "all_y_raw_finite"
    ] = bool(
        np.isfinite(
            y_raw
        ).all()
    )

    checks[
        "all_apparent_finite"
    ] = bool(
        np.isfinite(
            apparent
        ).all()
    )

    expected_target = (
        (
            y_raw.astype(
                np.float64
            )
            - target_mean[
                None,
                None,
                :,
            ]
        )
        / target_std[
            None,
            None,
            :,
        ]
    )

    checks[
        "target_scaling_max_abs_error"
    ] = float(
        np.max(
            np.abs(
                expected_target
                - y.astype(
                    np.float64
                )
            )
        )
    )

    reconstructed_apparent = np.stack(
        [
            y_raw[
                :,
                :,
                0
            ]
            - y_raw[
                :,
                :,
                2
            ],
            y_raw[
                :,
                :,
                1
            ]
            - y_raw[
                :,
                :,
                3
            ],
        ],
        axis=-1,
    )

    checks[
        "apparent_reconstruction_max_abs_error"
    ] = float(
        np.max(
            np.abs(
                reconstructed_apparent
                - apparent
            )
        )
    )

    horizon_ns = (
        HORIZONS[
            None,
            :,
        ]
        * 60
        * 1_000_000_000
    )

    # IMPORTANT:
    # np.array_equal() requires identical array shapes and does NOT broadcast.
    # Here the left side is [N, H] while horizon_ns is [1, H], so the v0.1.0
    # check incorrectly returned False for every non-empty dataset even when
    # all timestamps were exactly correct. Use an elementwise broadcasted
    # comparison instead.
    checks[
        "target_time_horizon_exact"
    ] = bool(
        np.all(
            (
                ttime
                - cend[
                    :,
                    None,
                ]
            )
            == horizon_ns
        )
    )

    checks[
        "context_duration_exact_59min"
    ] = bool(
        np.all(
            (
                cend
                - cstart
            )
            == (
                LOOKBACK
                - 1
            )
            * 60
            * 1_000_000_000
        )
    )

    checks[
        "target_after_context"
    ] = bool(
        np.all(
            ttime
            > cend[
                :,
                None,
            ]
        )
    )

    checks[
        "target_rows_strictly_increasing_by_horizon"
    ] = bool(
        np.all(
            np.diff(
                target_row,
                axis=1,
            )
            > 0
        )
    )

    checks[
        "segment_ids_nonnegative"
    ] = bool(
        np.all(
            segment_id
            >= 0
        )
    )

    pass_flags = [
        checks[
            "X_shape_ok"
        ],
        checks[
            "y_shape_ok"
        ],
        checks[
            "all_X_finite"
        ],
        checks[
            "all_y_finite"
        ],
        checks[
            "all_y_raw_finite"
        ],
        checks[
            "all_apparent_finite"
        ],
        checks[
            "target_time_horizon_exact"
        ],
        checks[
            "context_duration_exact_59min"
        ],
        checks[
            "target_after_context"
        ],
        checks[
            "target_rows_strictly_increasing_by_horizon"
        ],
        checks[
            "segment_ids_nonnegative"
        ],
        (
            checks[
                "target_scaling_max_abs_error"
            ]
            < 1e-5
        ),
        (
            checks[
                "apparent_reconstruction_max_abs_error"
            ]
            < 1e-6
        ),
    ]

    checks[
        "validation_passed"
    ] = bool(
        all(
            pass_flags
        )
    )

    checks[
        "dataset_id"
    ] = dataset_id

    checks[
        "window_count"
    ] = int(
        n
    )

    return checks


def save_npz(
    *,
    path: Path,
    arrays,
    dataset_id: str,
    source_file: Path,
    scaler_source: Path,
    compressed: bool,
):
    payload = {
        **arrays,
        "horizons_min": HORIZONS,
        "feature_names": np.asarray(
            FEATURE_NAMES
        ),
        "target_names": np.asarray(
            TARGET_NAMES
        ),
        "source_file": np.asarray(
            str(
                source_file
            )
        ),
        "dataset_id": np.asarray(
            dataset_id
        ),
        "scaler_source": np.asarray(
            str(
                scaler_source
            )
        ),
    }

    if compressed:
        np.savez_compressed(
            path,
            **payload,
        )
    else:
        np.savez(
            path,
            **payload,
        )


def load_sci_style():
    path = (
        Path(__file__).resolve().parent
        / "sci_plot_style.py"
    )

    if not path.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        "sci_plot_style_08b",
        str(
            path
        ),
    )

    if (
        spec is None
        or spec.loader is None
    ):
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
    summary_df,
    shift_df,
    segment_df,
    output_dir,
):
    import matplotlib.pyplot as plt

    style = load_sci_style()

    fig_dir = (
        output_dir
        / "figures"
    )

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    def clean(
        ax
    ):
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

    def save(
        fig,
        name
    ):
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

    # Window counts.
    fig, ax = plt.subplots(
        figsize=(
            6.0,
            3.3,
        )
    )

    x = np.arange(
        len(
            summary_df
        )
    )

    ax.bar(
        x,
        summary_df[
            "window_count"
        ],
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        summary_df[
            "dataset_id"
        ],
        rotation=30,
        ha="right",
    )

    ax.set_ylabel(
        "External forecasting windows"
    )

    clean(
        ax
    )

    save(
        fig,
        "01_external_window_counts",
    )

    # Feature shift heatmap: mean absolute z.
    if not shift_df.empty:
        pivot = (
            shift_df
            .pivot(
                index="dataset_id",
                columns="feature",
                values="mean_abs_z",
            )
            .reindex(
                columns=FEATURE_NAMES
            )
        )

        fig, ax = plt.subplots(
            figsize=(
                8.2,
                3.5,
            )
        )

        im = ax.imshow(
            pivot.to_numpy(),
            aspect="auto",
        )

        ax.set_yticks(
            np.arange(
                len(
                    pivot.index
                )
            )
        )

        ax.set_yticklabels(
            pivot.index
        )

        ax.set_xticks(
            np.arange(
                len(
                    FEATURE_NAMES
                )
            )
        )

        ax.set_xticklabels(
            FEATURE_NAMES,
            rotation=40,
            ha="right",
        )

        ax.set_ylabel(
            "External dataset"
        )

        cbar = fig.colorbar(
            im,
            ax=ax,
        )

        cbar.set_label(
            r"Mean $|z|$ under SD1090 TRAIN scaler"
        )

        clean(
            ax
        )

        save(
            fig,
            "02_feature_shift_mean_abs_z",
        )

    # Segment windows.
    if not segment_df.empty:
        fig, ax = plt.subplots(
            figsize=(
                6.2,
                3.5,
            )
        )

        for dataset_id in (
            summary_df[
                "dataset_id"
            ]
        ):
            grp = (
                segment_df.loc[
                    segment_df[
                        "dataset_id"
                    ] == dataset_id
                ]
                .sort_values(
                    "window_count",
                    ascending=False,
                )
                .reset_index(
                    drop=True
                )
            )

            if grp.empty:
                continue

            ax.plot(
                np.arange(
                    1,
                    len(
                        grp
                    )
                    + 1
                ),
                grp[
                    "window_count"
                ],
                marker="o",
                label=dataset_id,
            )

        ax.set_xlabel(
            "Eligible segment rank"
        )

        ax.set_ylabel(
            "Windows per segment"
        )

        ax.set_yscale(
            "log"
        )

        ax.legend(
            loc="best"
        )

        clean(
            ax
        )

        save(
            fig,
            "03_segment_window_distribution",
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sd1033-2024",
        default=DEFAULT_RAW_FILES[
            "SD1033_2024_full"
        ],
    )

    parser.add_argument(
        "--sd1033-2023",
        default=DEFAULT_RAW_FILES[
            "SD1033_2023_full"
        ],
    )

    parser.add_argument(
        "--sd1033-2022",
        default=DEFAULT_RAW_FILES[
            "SD1033_2022_full"
        ],
    )

    parser.add_argument(
        "--reference-sd1090",
        default=DEFAULT_REFERENCE_SD1090,
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
        "--step-tolerance-seconds",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--no-matched-2024",
        action="store_true",
    )

    parser.add_argument(
        "--no-compression",
        action="store_true",
    )

    parser.add_argument(
        "--plots",
        action="store_true",
    )

    args = parser.parse_args()

    audit08a = import_module_from_neighbor(
        SHARED_08A,
        "audit08a_for_08b",
    )

    core06 = import_module_from_neighbor(
        SHARED_06,
        "core06_for_08b",
    )

    xr = audit08a.import_xarray()

    frozen_dataset_dir = Path(
        args.frozen_dataset_dir
    )

    frozen = load_frozen_scalers(
        core06,
        frozen_dataset_dir,
    )

    output_dir = (
        Path(
            args.output_dir
        )
        if args.output_dir
        else default_output_dir(
            frozen_dataset_dir
        )
    )

    dataset_dir = (
        output_dir
        / "datasets"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_paths = {
        "SD1033_2024_full": Path(
            args.sd1033_2024
        ),
        "SD1033_2023_full": Path(
            args.sd1033_2023
        ),
        "SD1033_2022_full": Path(
            args.sd1033_2022
        ),
    }

    reference_path = Path(
        args.reference_sd1090
    )

    log(
        "=" * 110
    )

    log(
        "08B - BUILD FROZEN EXTERNAL SD1033 DATASETS"
    )

    log(
        "=" * 110
    )

    log(
        f"script_version          : {SCRIPT_VERSION}"
    )

    log(
        f"frozen dataset          : {frozen_dataset_dir}"
    )

    log(
        f"frozen scalers           : {frozen['scalers_path']}"
    )

    log(
        f"external scaler fitting : NEVER"
    )

    log(
        f"lookback / horizons     : "
        f"{LOOKBACK} / {HORIZONS.tolist()} min"
    )

    log(
        f"output_dir              : {output_dir}"
    )

    log("")

    audits = {}

    for dataset_id, path in (
        raw_paths.items()
    ):
        audits[
            dataset_id
        ] = audit08a.audit_one(
            xr,
            path,
            args.step_tolerance_seconds,
            top_segments=20,
            top_missing=10,
        )

    reference_audit = None

    if (
        not args.no_matched_2024
    ):
        if not reference_path.exists():
            raise FileNotFoundError(
                "Matched-2024 dataset requested but SD1090 reference "
                f"does not exist: {reference_path}"
            )

        reference_audit = (
            audit08a.audit_one(
                xr,
                reference_path,
                args.step_tolerance_seconds,
                top_segments=20,
                top_missing=10,
            )
        )

    dataset_specs = []

    for dataset_id in [
        "SD1033_2024_full",
        "SD1033_2023_full",
        "SD1033_2022_full",
    ]:
        audit = audits[
            dataset_id
        ]

        segments = eligible_full_segments(
            audit,
            audit08a,
            args.step_tolerance_seconds,
        )

        dataset_specs.append({
            "dataset_id": dataset_id,
            "source_file": raw_paths[
                dataset_id
            ],
            "audit": audit,
            "segments": segments,
            "mode": "all_model_ready_segments",
        })

    if (
        not args.no_matched_2024
        and reference_audit is not None
    ):
        matched_segments = (
            eligible_matched_segments(
                audits[
                    "SD1033_2024_full"
                ],
                reference_audit,
                audit08a,
                args.step_tolerance_seconds,
            )
        )

        dataset_specs.insert(
            1,
            {
                "dataset_id": (
                    "SD1033_2024_matched_SD1090"
                ),
                "source_file": raw_paths[
                    "SD1033_2024_full"
                ],
                "audit": audits[
                    "SD1033_2024_full"
                ],
                "segments": matched_segments,
                "mode": (
                    "exact_common_model_ready_timestamps_with_SD1090_2024"
                ),
            },
        )

    summary_rows = []
    segment_rows_all = []
    shift_rows_all = []
    validation_rows = []
    dataset_manifest_entries = {}

    compressed = (
        not args.no_compression
    )

    for spec in dataset_specs:
        dataset_id = spec[
            "dataset_id"
        ]

        source_file = spec[
            "source_file"
        ]

        audit = spec[
            "audit"
        ]

        segments = spec[
            "segments"
        ]

        if not segments:
            raise RuntimeError(
                f"{dataset_id}: no eligible segments >= {MIN_SEGMENT_ROWS} rows"
            )

        feature_raw, target_raw = (
            build_engineered_rows(
                audit
            )
        )

        source_mask = segment_source_row_mask(
            len(
                audit[
                    "time"
                ]
            ),
            segments,
        )

        shift_rows_all.extend(
            feature_shift_rows(
                dataset_id=dataset_id,
                feature_raw=feature_raw,
                source_row_mask=source_mask,
                feature_mean=frozen[
                    "feature_mean"
                ],
                feature_std=frozen[
                    "feature_std"
                ],
            )
        )

        build_t0 = time.perf_counter()

        arrays, segment_rows = (
            build_one_dataset(
                dataset_id=dataset_id,
                source_file=source_file,
                audit=audit,
                segments=segments,
                feature_raw=feature_raw,
                target_raw=target_raw,
                feature_mean=frozen[
                    "feature_mean"
                ],
                feature_std=frozen[
                    "feature_std"
                ],
                target_mean=frozen[
                    "target_mean"
                ],
                target_std=frozen[
                    "target_std"
                ],
            )
        )

        validation = (
            validate_built_dataset(
                dataset_id=dataset_id,
                arrays=arrays,
                feature_mean=frozen[
                    "feature_mean"
                ],
                feature_std=frozen[
                    "feature_std"
                ],
                target_mean=frozen[
                    "target_mean"
                ],
                target_std=frozen[
                    "target_std"
                ],
            )
        )

        validation_rows.append(
            validation
        )

        if not validation[
            "validation_passed"
        ]:
            failed_checks = []

            for key, value in validation.items():
                if key in {
                    "dataset_id",
                    "window_count",
                    "validation_passed",
                }:
                    continue

                if isinstance(
                    value,
                    (
                        bool,
                        np.bool_,
                    ),
                ) and not bool(
                    value
                ):
                    failed_checks.append(
                        f"{key}=False"
                    )

            if (
                validation[
                    "target_scaling_max_abs_error"
                ]
                >= 1e-5
            ):
                failed_checks.append(
                    "target_scaling_max_abs_error="
                    f"{validation['target_scaling_max_abs_error']:.8g}"
                )

            if (
                validation[
                    "apparent_reconstruction_max_abs_error"
                ]
                >= 1e-6
            ):
                failed_checks.append(
                    "apparent_reconstruction_max_abs_error="
                    f"{validation['apparent_reconstruction_max_abs_error']:.8g}"
                )

            log(
                "[VALIDATION FAILED] "
                + dataset_id
                + ": "
                + ", ".join(
                    failed_checks
                )
            )

            raise RuntimeError(
                f"{dataset_id}: internal validation failed. "
                f"Failed checks: {', '.join(failed_checks)}"
            )

        out_path = (
            dataset_dir
            / (
                dataset_id
                + ".npz"
            )
        )

        save_t0 = time.perf_counter()

        save_npz(
            path=out_path,
            arrays=arrays,
            dataset_id=dataset_id,
            source_file=source_file,
            scaler_source=frozen[
                "scalers_path"
            ],
            compressed=compressed,
        )

        save_seconds = (
            time.perf_counter()
            - save_t0
        )

        file_size_mb = (
            out_path.stat().st_size
            / 1024.0**2
        )

        segment_rows_all.extend(
            segment_rows
        )

        n_windows = int(
            arrays[
                "X"
            ].shape[
                0
            ]
        )

        n_source_rows = int(
            source_mask.sum()
        )

        summary_rows.append({
            "dataset_id": dataset_id,
            "mode": spec[
                "mode"
            ],
            "source_file": str(
                source_file
            ),
            "npz_file": str(
                out_path
            ),
            "eligible_segment_count": int(
                len(
                    segments
                )
            ),
            "eligible_source_row_count": n_source_rows,
            "window_count": n_windows,
            "lookback_min": LOOKBACK,
            "max_horizon_min": MAX_HORIZON,
            "first_eligible_time_utc": audit[
                "time"
            ][
                segments[
                    0
                ][
                    0
                ]
            ].isoformat(),
            "last_eligible_time_utc": audit[
                "time"
            ][
                segments[
                    -1
                ][
                    1
                ]
            ].isoformat(),
            "build_seconds": float(
                time.perf_counter()
                - build_t0
            ),
            "save_seconds": float(
                save_seconds
            ),
            "npz_size_mb": float(
                file_size_mb
            ),
            "compressed": bool(
                compressed
            ),
            "validation_passed": bool(
                validation[
                    "validation_passed"
                ]
            ),
        })

        dataset_manifest_entries[
            dataset_id
        ] = {
            "source_file": str(
                source_file
            ),
            "mode": spec[
                "mode"
            ],
            "npz_file": str(
                out_path
            ),
            "eligible_segments": int(
                len(
                    segments
                )
            ),
            "eligible_source_rows": n_source_rows,
            "windows": n_windows,
            "validation": validation,
        }

        log(
            f"[OK] {dataset_id}: "
            f"{n_windows:,} windows | "
            f"{len(segments)} segments | "
            f"{file_size_mb:.1f} MB | "
            f"validation PASS"
        )

        del (
            arrays,
            feature_raw,
            target_raw,
            source_mask,
        )

        gc.collect()

    summary_df = pd.DataFrame(
        summary_rows
    )

    segment_df = pd.DataFrame(
        segment_rows_all
    )

    shift_df = pd.DataFrame(
        shift_rows_all
    )

    validation_df = pd.DataFrame(
        validation_rows
    )

    summary_df.to_csv(
        output_dir
        / "external_dataset_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    segment_df.to_csv(
        output_dir
        / "external_segment_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    shift_df.to_csv(
        output_dir
        / "feature_shift_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    validation_df.to_csv(
        output_dir
        / "external_dataset_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest_out = {
        "script_version": SCRIPT_VERSION,
        "purpose": (
            "Frozen external inference dataset construction; "
            "no external scaler fit, interpolation, training or tuning."
        ),
        "frozen_source_dataset": str(
            frozen_dataset_dir
        ),
        "frozen_dataset_manifest": str(
            frozen[
                "manifest_path"
            ]
        ),
        "frozen_scalers": str(
            frozen[
                "scalers_path"
            ]
        ),
        "frozen_scalers_sha256": sha256_file(
            frozen[
                "scalers_path"
            ]
        ),
        "frozen_dataset_manifest_sha256": sha256_file(
            frozen[
                "manifest_path"
            ]
        ),
        "frozen_protocol": {
            "lookback_minutes": LOOKBACK,
            "forecast_horizons_minutes": HORIZONS,
            "feature_names": FEATURE_NAMES,
            "target_names": TARGET_NAMES,
            "feature_scaler_fit_on_external_data": False,
            "target_scaler_fit_on_external_data": False,
            "external_interpolation": False,
            "cross_gap_windows_allowed": False,
        },
        "datasets": dataset_manifest_entries,
        "reference_sd1090": (
            str(
                reference_path
            )
            if (
                not args.no_matched_2024
            )
            else None
        ),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xarray": xr.__version__,
        },
        "outputs": {
            "summary": "external_dataset_summary.csv",
            "segments": "external_segment_summary.csv",
            "feature_shift": "feature_shift_summary.csv",
            "validation": "external_dataset_validation.csv",
            "report": "external_dataset_report.txt",
            "dataset_directory": "datasets",
        },
    }

    save_json(
        output_dir
        / "external_dataset_manifest.json",
        manifest_out,
    )

    with (
        output_dir
        / "external_dataset_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "08B Frozen External SD1033 Dataset Builder Report\n"
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
            "NO interpolation.\n"
        )

        f.write(
            "NO training / fine-tuning / hyperparameter selection.\n"
        )

        f.write(
            "NO window may cross a missing-data or time-gap boundary.\n"
        )

        f.write(
            f"Frozen lookback={LOOKBACK} min, "
            f"horizons={HORIZONS.tolist()} min.\n\n"
        )

        f.write(
            "Dataset summary\n"
        )

        f.write(
            "-" * 110
            + "\n"
        )

        f.write(
            summary_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nFeature distribution shift under SD1090 TRAIN scaler\n"
        )

        f.write(
            "-" * 110
            + "\n"
        )

        f.write(
            shift_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nInternal validation\n"
        )

        f.write(
            "-" * 110
            + "\n"
        )

        f.write(
            validation_df.to_string(
                index=False
            )
        )

        f.write(
            "\n"
        )

    if args.plots:
        make_plots(
            summary_df=summary_df,
            shift_df=shift_df,
            segment_df=segment_df,
            output_dir=output_dir,
        )

    log("")
    log(
        "=" * 110
    )

    log(
        "08B EXTERNAL DATASET BUILD RESULTS"
    )

    log(
        "=" * 110
    )

    for row in (
        summary_df.itertuples()
    ):
        log(
            f"{row.dataset_id}:"
        )

        log(
            f"  eligible segments      = "
            f"{row.eligible_segment_count:,}"
        )

        log(
            f"  eligible source rows   = "
            f"{row.eligible_source_row_count:,}"
        )

        log(
            f"  forecasting windows    = "
            f"{row.window_count:,}"
        )

        log(
            f"  NPZ size               = "
            f"{row.npz_size_mb:.1f} MB"
        )

        log(
            f"  validation             = "
            f"{'PASS' if row.validation_passed else 'FAIL'}"
        )

    log("")
    log(
        "[SCALER POLICY] "
        "All external X/y standardized using frozen SD1090 TRAIN scalers only."
    )

    log(
        "[NO EXTERNAL FIT] "
        "No SD1033 mean/std was used for preprocessing."
    )

    log(
        "[NO GAP CROSSING] "
        "Every window is contained within one continuous model-ready segment."
    )

    log(
        f"[DONE] 08B outputs: "
        f"{output_dir}"
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
