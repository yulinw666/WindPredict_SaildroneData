# -*- coding: utf-8 -*-
"""
11C0_build_BP_STGNN_comparison_datasets.py

Builds and audits the datasets required for a full-paper-informed BP-STGNN
comparison against the frozen Physics-Compact-Vessel-Residual v1.0 model.

This script is DATA CONSTRUCTION ONLY. It never trains a neural network.

Two complementary tracks are produced.

Track P -- paper-protocol-oriented BP-STGNN
-------------------------------------------
- raw Saildrone observations;
- 10-min resampling;
- paper-exact 10 graph nodes:
    U, V, SOG, COG, T, H, P, dT, dP, dH
- paper-exact physical mask;
- default 60-min historical span represented by 6 x 10-min samples;
- next 10-min resampled U/V as the target.

Important:
The BP-STGNN paper does NOT report the exact 10-min aggregation operator,
historical window length, or exact differential-node formula. Therefore the
following choices are explicit PROJECT ADAPTATIONS, not claims about the
authors' exact implementation:

    linear variables: arithmetic mean in each 10-min block
    COG: circular mean
    dT/dP/dH: first difference of the 10-min resampled sequence
    default paper-track lookback: 6 resampled steps = 60 min
    target: the immediately following 10-min resampled U/V pair

Each resampled block must contain a complete set of raw 1-min samples for all
required variables. No long pressure gap is interpolated.

Track F -- common-task BP-STGNN adaptation
------------------------------------------
- 1-min temporal grid;
- same 60-min history as the frozen proposed model;
- same target horizons [1, 2, 3, 5, 10] min;
- same paper-exact 10 graph nodes;
- same context-end and target timestamps as the frozen proposed-model NPZ;
- pressure/differential completeness required.

Therefore Track F can be compared on EXACT COMMON TARGET TIMESTAMPS with the
frozen proposed model. The output stores the proposed-model sample index for
every retained BP-STGNN sample.

SCIENTIFIC FIREWALL
-------------------
- Proposed Physics-Compact v1.0 is never modified.
- No model is trained here.
- No scalers are fit here.
- No hyperparameters are selected here.
- No missing multi-day pressure gap is interpolated.
- No external performance metric is computed.
- External datasets are processed with exactly the same frozen preprocessing
  rules as development datasets.

The Track-F sample loss caused by pressure is reported explicitly.

Recommended environment:
    conda activate WindPredict

Default project paths:
    D:\\project\\WindPredict_SaildroneData

Expected frozen proposed-model dataset:
    data\\forecasting\\SD1090_TPOS2024_JointForecasting_v0_1

Expected external proposed-model datasets:
    data\\external\\SD1033_external_forecasting_v0_1\\datasets

Raw NetCDF files are auto-discovered recursively under:
    data\\raw
    data\\external

If discovery is ambiguous, provide --raw-map-json with a JSON mapping, e.g.

{
  "SD1090": "D:/.../TPOS-2024_SD1090_1min.nc",
  "SD1033_2024": "D:/.../SD1033_2024_1min.nc",
  "SD1033_2023": "D:/.../SD1033_2023_1min.nc",
  "SD1033_2022": "D:/.../SD1033_2022_1min.nc"
}

OUTPUT
------
11C0_BP_STGNN_comparison_datasets_v0_1/
    physical_mask.npy
    physical_mask.csv
    paper_node_schema.csv
    raw_file_audit.csv
    variable_mapping_audit.csv
    pressure_coverage_audit.csv
    track_F_common_task/
        SD1090_train.npz
        SD1090_validation.npz
        SD1090_test.npz
        SD1033_2024_matched_SD1090.npz
        SD1033_2023_full.npz
        SD1033_2022_full.npz
        [SD1033_2024_full.npz if available]
    track_P_paper_protocol/
        SD1090_train.npz
        SD1090_validation.npz
        SD1090_test.npz
        SD1033_2024_matched_SD1090.npz
        SD1033_2023_full.npz
        SD1033_2022_full.npz
        [SD1033_2024_full.npz if available]
    track_F_sample_audit.csv
    track_P_sample_audit.csv
    comparison_coverage_summary.csv
    preprocessing_manifest.json
    dataset_report.txt

The script intentionally stores RAW physical-value tensors. Train-only scaling
must be fitted later by Stage 11C2.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0-BP-STGNN-dataset-build"

DEFAULT_PROJECT_ROOT = Path(
    r"D:\project\WindPredict_SaildroneData"
)

DEFAULT_FROZEN_DATASET_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "SD1090_TPOS2024_JointForecasting_v0_1"
)

DEFAULT_EXTERNAL_ROOT = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "external"
    / "SD1033_external_forecasting_v0_1"
)

DEFAULT_RAW_ROOTS = [
    DEFAULT_PROJECT_ROOT
    / "data"
    / "raw",
    DEFAULT_PROJECT_ROOT
    / "data"
    / "external",
]

DEFAULT_OUTPUT_DIR = (
    DEFAULT_FROZEN_DATASET_DIR
    / "11C0_BP_STGNN_comparison_datasets_v0_1"
)

HORIZONS_MIN = [
    1,
    2,
    3,
    5,
    10,
]

TRACK_F_LOOKBACK_MIN = 60

TRACK_P_RESAMPLE_MIN = 10
TRACK_P_LOOKBACK_STEPS = 6

NODE_NAMES = [
    "U",
    "V",
    "SOG",
    "COG",
    "T",
    "H",
    "P",
    "dT",
    "dP",
    "dH",
]

NODE_GROUPS = {
    "target_T": [
        "U",
        "V",
    ],
    "source_S": [
        "dP",
        "dT",
        "dH",
    ],
    "kinematic_K": [
        "SOG",
        "COG",
    ],
    "environment_E": [
        "P",
        "T",
        "H",
    ],
}

RAW_ALIASES = {
    "U": [
        "UWND_MEAN",
        "UWND",
        "U_WIND",
        "U",
        "eastward_wind",
    ],
    "V": [
        "VWND_MEAN",
        "VWND",
        "V_WIND",
        "V",
        "northward_wind",
    ],
    "SOG": [
        "SOG",
        "platform_speed_wrt_ground",
        "SPEED_OVER_GROUND",
    ],
    "COG": [
        "COG",
        "platform_course",
        "COURSE_OVER_GROUND",
    ],
    "T": [
        "TEMP_AIR_MEAN",
        "TEMP_AIR",
        "air_temperature",
        "AIR_TEMP",
        "TA",
    ],
    "H": [
        "RH_MEAN",
        "RH",
        "relative_humidity",
        "RELATIVE_HUMIDITY",
    ],
    "P": [
        "BARO_PRES_MEAN",
        "BARO_PRES",
        "AIR_PRESSURE_MEAN",
        "PRESSURE_AIR_MEAN",
        "PRES_MEAN",
        "air_pressure",
        "ATM_PRESSURE",
        "P",
    ],
}

PROPOSED_EXTERNAL_IDS = [
    "SD1033_2024_full",
    "SD1033_2024_matched_SD1090",
    "SD1033_2023_full",
    "SD1033_2022_full",
]


def log(
    message="",
):
    print(
        message,
        flush=True,
    )


def save_json(
    path: Path,
    obj,
):
    def cv(v):
        if isinstance(
            v,
            Path,
        ):
            return str(
                v
            )

        if isinstance(
            v,
            np.ndarray,
        ):
            return v.tolist()

        if isinstance(
            v,
            np.integer,
        ):
            return int(
                v
            )

        if isinstance(
            v,
            np.floating,
        ):
            return (
                None
                if np.isnan(
                    v
                )
                else float(
                    v
                )
            )

        if isinstance(
            v,
            np.bool_,
        ):
            return bool(
                v
            )

        if isinstance(
            v,
            dict,
        ):
            return {
                str(
                    k
                ): cv(
                    x
                )
                for k, x
                in v.items()
            }

        if isinstance(
            v,
            (
                list,
                tuple,
            ),
        ):
            return [
                cv(
                    x
                )
                for x in v
            ]

        return v

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            cv(
                obj
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(
            f
        )


def import_xarray():
    try:
        import xarray as xr
    except Exception as exc:
        raise RuntimeError(
            "xarray is required. Activate the WindPredict/WindDL environment "
            "that contains xarray/netCDF4."
        ) from exc

    return xr


def normalize_datetime_ns(
    values,
):
    dt = pd.to_datetime(
        values,
        utc=True,
        errors="coerce",
    )

    if isinstance(
        dt,
        pd.DatetimeIndex,
    ):
        return dt.tz_convert(
            None
        ).to_numpy(
            dtype="datetime64[ns]"
        )

    return (
        pd.DatetimeIndex(
            dt
        )
        .tz_convert(
            None
        )
        .to_numpy(
            dtype="datetime64[ns]"
        )
    )


def time_to_ns(
    values,
):
    return normalize_datetime_ns(
        values
    ).astype(
        "datetime64[ns]"
    ).astype(
        np.int64
    )


def ns_to_iso(
    value,
):
    return str(
        np.datetime64(
            int(
                value
            ),
            "ns",
        )
    )


def circular_mean_deg(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if (
        values.size == 0
        or not np.isfinite(
            values
        ).all()
    ):
        return np.nan

    rad = np.deg2rad(
        values
    )

    s = float(
        np.mean(
            np.sin(
                rad
            )
        )
    )

    c = float(
        np.mean(
            np.cos(
                rad
            )
        )
    )

    if (
        abs(
            s
        ) < 1e-15
        and abs(
            c
        ) < 1e-15
    ):
        return np.nan

    return float(
        np.degrees(
            np.arctan2(
                s,
                c,
            )
        )
        % 360.0
    )


def build_physical_mask():
    """
    Paper Eq. (2).

    M_ij = 1 when directed edge j -> i is permitted.
    """
    n = len(
        NODE_NAMES
    )

    idx = {
        name: i
        for i, name
        in enumerate(
            NODE_NAMES
        )
    }

    T = set(
        NODE_GROUPS[
            "target_T"
        ]
    )

    S = set(
        NODE_GROUPS[
            "source_S"
        ]
    )

    K = set(
        NODE_GROUPS[
            "kinematic_K"
        ]
    )

    E = set(
        NODE_GROUPS[
            "environment_E"
        ]
    )

    M = np.zeros(
        (
            n,
            n,
        ),
        dtype=np.float32,
    )

    for i_name in NODE_NAMES:
        for j_name in NODE_NAMES:
            allowed = False

            if (
                j_name in (
                    S
                    | K
                )
                and i_name in T
            ):
                allowed = True

            if (
                j_name in E
                and i_name in S
            ):
                allowed = True

            if (
                j_name in E
                and i_name in E
            ):
                allowed = True

            if (
                i_name
                == j_name
            ):
                allowed = True

            if allowed:
                M[
                    idx[
                        i_name
                    ],
                    idx[
                        j_name
                    ],
                ] = 1.0

    if int(
        M.sum()
    ) != 35:
        raise RuntimeError(
            "Physical-mask audit failed: expected 35 allowed entries, "
            f"got {int(M.sum())}."
        )

    return M


def save_mask_and_schema(
    output_dir: Path,
):
    M = build_physical_mask()

    np.save(
        output_dir
        / "physical_mask.npy",
        M,
    )

    pd.DataFrame(
        M,
        index=NODE_NAMES,
        columns=NODE_NAMES,
    ).to_csv(
        output_dir
        / "physical_mask.csv",
        encoding="utf-8-sig",
    )

    schema_rows = []

    for i, name in enumerate(
        NODE_NAMES
    ):
        group = None

        for group_name, values in NODE_GROUPS.items():
            if name in values:
                group = group_name
                break

        schema_rows.append({
            "node_index": i,
            "node_name": name,
            "functional_group": group,
            "paper_exact_node": True,
        })

    pd.DataFrame(
        schema_rows
    ).to_csv(
        output_dir
        / "paper_node_schema.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return M


def recursive_nc_files(
    roots: List[Path],
):
    files = []

    for root in roots:
        if not root.exists():
            continue

        for pattern in [
            "*.nc",
            "*.nc4",
            "*.cdf",
        ]:
            files.extend(
                root.rglob(
                    pattern
                )
            )

    unique = []

    seen = set()

    for p in files:
        rp = str(
            p.resolve()
        ).lower()

        if rp not in seen:
            seen.add(
                rp
            )

            unique.append(
                p
            )

    return unique


def score_file_name(
    path: Path,
    platform: str,
    year: int,
):
    name = path.name.lower()

    score = 0

    if platform.lower() in name:
        score += 10

    if str(
        year
    ) in name:
        score += 5

    if (
        "1min" in name
        or "1_min" in name
        or "1-min" in name
    ):
        score += 2

    return score


def inspect_time_coverage(
    path: Path,
):
    xr = import_xarray()

    try:
        with xr.open_dataset(
            path,
            decode_times=True,
        ) as ds:
            if (
                "time"
                not in ds.variables
            ):
                return None

            t = normalize_datetime_ns(
                ds[
                    "time"
                ].values
            )

            if len(
                t
            ) == 0:
                return None

            return {
                "start": t[
                    0
                ],
                "end": t[
                    -1
                ],
                "count": int(
                    len(
                        t
                    )
                ),
            }

    except Exception:
        return None


def choose_raw_file(
    files: List[Path],
    platform: str,
    year: int,
):
    candidates = []

    for path in files:
        name_score = score_file_name(
            path,
            platform,
            year,
        )

        if name_score <= 0:
            continue

        coverage = inspect_time_coverage(
            path
        )

        if coverage is None:
            continue

        years = pd.DatetimeIndex(
            [
                coverage[
                    "start"
                ],
                coverage[
                    "end"
                ],
            ]
        ).year

        year_overlap = (
            year
            in range(
                int(
                    min(
                        years
                    )
                ),
                int(
                    max(
                        years
                    )
                )
                + 1,
            )
        )

        if not year_overlap:
            continue

        candidates.append(
            (
                name_score,
                coverage[
                    "count"
                ],
                path,
                coverage,
            )
        )

    if not candidates:
        return (
            None,
            [],
        )

    candidates.sort(
        key=lambda x: (
            x[
                0
            ],
            x[
                1
            ],
        ),
        reverse=True,
    )

    return (
        candidates[
            0
        ][
            2
        ],
        candidates,
    )


def resolve_raw_map(
    raw_roots: List[Path],
    raw_map_json: Optional[Path],
):
    if (
        raw_map_json
        is not None
    ):
        mapping = load_json(
            raw_map_json
        )

        return {
            key: Path(
                value
            )
            for key, value
            in mapping.items()
        }

    files = recursive_nc_files(
        raw_roots
    )

    requested = {
        "SD1090": (
            "SD1090",
            2024,
        ),
        "SD1033_2024": (
            "SD1033",
            2024,
        ),
        "SD1033_2023": (
            "SD1033",
            2023,
        ),
        "SD1033_2022": (
            "SD1033",
            2022,
        ),
    }

    result = {}

    for key, (
        platform,
        year,
    ) in requested.items():
        best, _ = choose_raw_file(
            files,
            platform,
            year,
        )

        if (
            best
            is not None
        ):
            result[
                key
            ] = best

    return result


def resolve_variable(
    ds,
    canonical_name: str,
):
    aliases = RAW_ALIASES[
        canonical_name
    ]

    variables_lower = {
        str(
            name
        ).lower(): str(
            name
        )
        for name in ds.variables
    }

    for alias in aliases:
        if alias in ds.variables:
            return alias

        key = alias.lower()

        if key in variables_lower:
            return variables_lower[
                key
            ]

    # Fuzzy last resort.
    for alias in aliases:
        key = alias.lower()

        for name in ds.variables:
            lname = str(
                name
            ).lower()

            if (
                key in lname
                or lname in key
            ):
                return str(
                    name
                )

    raise KeyError(
        f"Could not resolve raw variable '{canonical_name}'. "
        f"Tried aliases={aliases}. Available variables include "
        f"{list(ds.variables)[:80]}"
    )


@dataclass
class RawSeries:
    source_path: Path
    time_ns: np.ndarray
    values: Dict[str, np.ndarray]
    variable_map: Dict[str, str]
    units: Dict[str, str]


def read_raw_series(
    path: Path,
):
    xr = import_xarray()

    with xr.open_dataset(
        path,
        decode_times=True,
    ) as ds:
        if (
            "time"
            not in ds.variables
        ):
            raise KeyError(
                f"{path}: no 'time' variable."
            )

        time_ns = time_to_ns(
            ds[
                "time"
            ].values
        )

        if (
            len(
                time_ns
            ) == 0
        ):
            raise RuntimeError(
                f"{path}: empty time array."
            )

        values = {}

        mapping = {}

        units = {}

        for canonical in [
            "U",
            "V",
            "SOG",
            "COG",
            "T",
            "H",
            "P",
        ]:
            raw_name = resolve_variable(
                ds,
                canonical,
            )

            arr = np.asarray(
                ds[
                    raw_name
                ].values
            ).reshape(
                -1
            )

            if len(
                arr
            ) != len(
                time_ns
            ):
                raise RuntimeError(
                    f"{path}: {raw_name} length={len(arr)} does not match "
                    f"time length={len(time_ns)}."
                )

            values[
                canonical
            ] = arr.astype(
                np.float64
            )

            mapping[
                canonical
            ] = raw_name

            units[
                canonical
            ] = str(
                ds[
                    raw_name
                ].attrs.get(
                    "units",
                    "",
                )
            )

    order = np.argsort(
        time_ns
    )

    time_ns = time_ns[
        order
    ]

    for key in values:
        values[
            key
        ] = values[
            key
        ][
            order
        ]

    # Keep the first occurrence of duplicate timestamps.
    _, unique_idx = np.unique(
        time_ns,
        return_index=True,
    )

    unique_idx = np.sort(
        unique_idx
    )

    time_ns = time_ns[
        unique_idx
    ]

    for key in values:
        values[
            key
        ] = values[
            key
        ][
            unique_idx
        ]

    return RawSeries(
        source_path=path,
        time_ns=time_ns,
        values=values,
        variable_map=mapping,
        units=units,
    )


def build_native_node_matrix(
    raw: RawSeries,
):
    """
    1-min Track-F node construction.

    dT/dP/dH are first differences between consecutive 1-min observations.
    If the timestamp difference is not exactly 1 min, the differential nodes
    at the later sample are NaN.
    """
    n = len(
        raw.time_ns
    )

    nodes = np.full(
        (
            n,
            len(
                NODE_NAMES
            ),
        ),
        np.nan,
        dtype=np.float64,
    )

    idx = {
        name: i
        for i, name
        in enumerate(
            NODE_NAMES
        )
    }

    for name in [
        "U",
        "V",
        "SOG",
        "COG",
        "T",
        "H",
        "P",
    ]:
        nodes[
            :,
            idx[
                name
            ],
        ] = raw.values[
            name
        ]

    one_min_ns = 60_000_000_000

    contiguous = np.zeros(
        n,
        dtype=bool,
    )

    contiguous[
        1:
    ] = (
        np.diff(
            raw.time_ns
        )
        == one_min_ns
    )

    for base, delta in [
        (
            "T",
            "dT",
        ),
        (
            "P",
            "dP",
        ),
        (
            "H",
            "dH",
        ),
    ]:
        diff = np.full(
            n,
            np.nan,
            dtype=np.float64,
        )

        x = raw.values[
            base
        ]

        diff[
            1:
        ] = (
            x[
                1:
            ]
            - x[
                :-1
            ]
        )

        diff[
            ~contiguous
        ] = np.nan

        diff[
            ~np.isfinite(
                diff
            )
        ] = np.nan

        nodes[
            :,
            idx[
                delta
            ],
        ] = diff

    return nodes


def build_time_index(
    time_ns,
):
    return {
        int(
            t
        ): i
        for i, t
        in enumerate(
            time_ns
        )
    }


def load_proposed_npz(
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        result = {
            key: data[
                key
            ]
            for key in data.files
        }

    required = [
        "context_end_time_ns",
        "target_time_ns",
    ]

    for key in required:
        if key not in result:
            raise KeyError(
                f"{path}: missing required key '{key}'. "
                f"Available={list(result)}"
            )

    return result


def proposed_split_path(
    frozen_dataset_dir: Path,
    split: str,
):
    candidates = [
        frozen_dataset_dir
        / f"{split}.npz",
        frozen_dataset_dir
        / f"{split}_dataset.npz",
    ]

    for p in candidates:
        if p.exists():
            return p

    matches = list(
        frozen_dataset_dir.glob(
            f"*{split}*.npz"
        )
    )

    matches = [
        p
        for p in matches
        if "prediction" not in p.name.lower()
    ]

    if len(
        matches
    ) == 1:
        return matches[
            0
        ]

    raise FileNotFoundError(
        f"Could not uniquely locate proposed-model split NPZ for '{split}' "
        f"in {frozen_dataset_dir}. Matches={matches}"
    )


def external_proposed_path(
    external_root: Path,
    dataset_id: str,
):
    path = (
        external_root
        / "datasets"
        / f"{dataset_id}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    return path


def build_track_f_common_windows(
    raw: RawSeries,
    nodes_1min: np.ndarray,
    proposed: Dict[str, np.ndarray],
    *,
    dataset_id: str,
):
    """
    Construct BP-STGNN-F samples ONLY at context/target timestamps already
    present in the frozen proposed-model dataset.

    This guarantees exact temporal alignment for later head-to-head inference.
    """
    time_index = build_time_index(
        raw.time_ns
    )

    context_end = np.asarray(
        proposed[
            "context_end_time_ns"
        ],
        dtype=np.int64,
    ).reshape(
        -1
    )

    target_times = np.asarray(
        proposed[
            "target_time_ns"
        ],
        dtype=np.int64,
    )

    if target_times.ndim != 2:
        raise RuntimeError(
            f"{dataset_id}: expected target_time_ns [N,H], "
            f"got {target_times.shape}."
        )

    if target_times.shape[
        1
    ] != len(
        HORIZONS_MIN
    ):
        raise RuntimeError(
            f"{dataset_id}: target horizon count mismatch."
        )

    one_min_ns = 60_000_000_000

    accepted_X = []

    accepted_y = []

    accepted_context = []

    accepted_targets = []

    accepted_proposed_index = []

    rejection = {
        "missing_context_end_in_raw": 0,
        "context_not_contiguous_60min": 0,
        "missing_node_in_context": 0,
        "target_timestamp_mismatch": 0,
        "target_missing_in_raw": 0,
        "target_wind_nonfinite": 0,
    }

    for sample_idx in range(
        len(
            context_end
        )
    ):
        ce = int(
            context_end[
                sample_idx
            ]
        )

        if ce not in time_index:
            rejection[
                "missing_context_end_in_raw"
            ] += 1
            continue

        end_idx = time_index[
            ce
        ]

        start_idx = (
            end_idx
            - TRACK_F_LOOKBACK_MIN
            + 1
        )

        if start_idx < 0:
            rejection[
                "context_not_contiguous_60min"
            ] += 1
            continue

        expected_context = (
            ce
            - np.arange(
                TRACK_F_LOOKBACK_MIN
                - 1,
                -1,
                -1,
                dtype=np.int64,
            )
            * one_min_ns
        )

        actual_context = raw.time_ns[
            start_idx:
            end_idx
            + 1
        ]

        if (
            len(
                actual_context
            )
            != TRACK_F_LOOKBACK_MIN
            or not np.array_equal(
                actual_context,
                expected_context,
            )
        ):
            rejection[
                "context_not_contiguous_60min"
            ] += 1
            continue

        X = nodes_1min[
            start_idx:
            end_idx
            + 1,
            :
        ]

        if not np.isfinite(
            X
        ).all():
            rejection[
                "missing_node_in_context"
            ] += 1
            continue

        expected_targets = np.asarray(
            [
                ce
                + int(
                    h
                )
                * one_min_ns
                for h in HORIZONS_MIN
            ],
            dtype=np.int64,
        )

        supplied_targets = target_times[
            sample_idx
        ]

        if not np.array_equal(
            supplied_targets,
            expected_targets,
        ):
            rejection[
                "target_timestamp_mismatch"
            ] += 1
            continue

        target_indices = []

        missing_target = False

        for t in expected_targets:
            ti = time_index.get(
                int(
                    t
                )
            )

            if ti is None:
                missing_target = True
                break

            target_indices.append(
                ti
            )

        if missing_target:
            rejection[
                "target_missing_in_raw"
            ] += 1
            continue

        y = nodes_1min[
            np.asarray(
                target_indices,
                dtype=np.int64,
            ),
            0:2,
        ]

        if not np.isfinite(
            y
        ).all():
            rejection[
                "target_wind_nonfinite"
            ] += 1
            continue

        accepted_X.append(
            X.astype(
                np.float32
            )
        )

        accepted_y.append(
            y.astype(
                np.float32
            )
        )

        accepted_context.append(
            ce
        )

        accepted_targets.append(
            expected_targets
        )

        accepted_proposed_index.append(
            sample_idx
        )

    if accepted_X:
        X_out = np.stack(
            accepted_X,
            axis=0,
        )

        y_out = np.stack(
            accepted_y,
            axis=0,
        )

        context_out = np.asarray(
            accepted_context,
            dtype=np.int64,
        )

        target_out = np.stack(
            accepted_targets,
            axis=0,
        ).astype(
            np.int64
        )

        proposed_idx_out = np.asarray(
            accepted_proposed_index,
            dtype=np.int64,
        )

    else:
        X_out = np.empty(
            (
                0,
                TRACK_F_LOOKBACK_MIN,
                len(
                    NODE_NAMES
                ),
            ),
            dtype=np.float32,
        )

        y_out = np.empty(
            (
                0,
                len(
                    HORIZONS_MIN
                ),
                2,
            ),
            dtype=np.float32,
        )

        context_out = np.empty(
            0,
            dtype=np.int64,
        )

        target_out = np.empty(
            (
                0,
                len(
                    HORIZONS_MIN
                ),
            ),
            dtype=np.int64,
        )

        proposed_idx_out = np.empty(
            0,
            dtype=np.int64,
        )

    audit = {
        "dataset_id": dataset_id,
        "track": "F_common_task_1min",
        "proposed_samples": int(
            len(
                context_end
            )
        ),
        "accepted_samples": int(
            len(
                X_out
            )
        ),
        "accepted_fraction": (
            float(
                len(
                    X_out
                )
                / len(
                    context_end
                )
            )
            if len(
                context_end
            )
            > 0
            else np.nan
        ),
        **{
            key: int(
                value
            )
            for key, value
            in rejection.items()
        },
    }

    return {
        "X_raw": X_out,
        "y_wind_raw": y_out,
        "context_end_time_ns": context_out,
        "target_time_ns": target_out,
        "proposed_sample_index": proposed_idx_out,
        "feature_names": np.asarray(
            NODE_NAMES,
            dtype="U16",
        ),
        "target_names": np.asarray(
            [
                "U",
                "V",
            ],
            dtype="U8",
        ),
        "horizons_min": np.asarray(
            HORIZONS_MIN,
            dtype=np.int64,
        ),
    }, audit


def floor_to_block_start_ns(
    time_ns: np.ndarray,
    block_minutes: int,
):
    block_ns = (
        int(
            block_minutes
        )
        * 60_000_000_000
    )

    return (
        time_ns
        // block_ns
    ) * block_ns


def resample_10min_complete_blocks(
    raw: RawSeries,
):
    """
    Explicit Stage-11C project adaptation.

    - Floor timestamps to 10-min block start.
    - Require exactly 10 one-minute timestamps spaced by 1 min.
    - Require all seven base variables finite in the block.
    - Arithmetic mean for U,V,SOG,T,H,P.
    - Circular mean for COG.
    - dT/dP/dH computed AFTER resampling as first differences between
      contiguous 10-min blocks.
    """
    block_start = floor_to_block_start_ns(
        raw.time_ns,
        TRACK_P_RESAMPLE_MIN,
    )

    unique_blocks, first_idx, counts = np.unique(
        block_start,
        return_index=True,
        return_counts=True,
    )

    rows = []

    expected_count = TRACK_P_RESAMPLE_MIN

    one_min_ns = 60_000_000_000

    for b, start, count in zip(
        unique_blocks,
        first_idx,
        counts,
    ):
        if int(
            count
        ) != expected_count:
            continue

        end = (
            int(
                start
            )
            + int(
                count
            )
        )

        times = raw.time_ns[
            int(
                start
            ):
            end
        ]

        expected = (
            int(
                b
            )
            + np.arange(
                expected_count,
                dtype=np.int64,
            )
            * one_min_ns
        )

        if not np.array_equal(
            times,
            expected,
        ):
            continue

        base = {}

        valid = True

        for name in [
            "U",
            "V",
            "SOG",
            "COG",
            "T",
            "H",
            "P",
        ]:
            values = raw.values[
                name
            ][
                int(
                    start
                ):
                end
            ]

            if not np.isfinite(
                values
            ).all():
                valid = False
                break

            base[
                name
            ] = values

        if not valid:
            continue

        row = {
            "block_start_ns": int(
                b
            ),
            "U": float(
                np.mean(
                    base[
                        "U"
                    ]
                )
            ),
            "V": float(
                np.mean(
                    base[
                        "V"
                    ]
                )
            ),
            "SOG": float(
                np.mean(
                    base[
                        "SOG"
                    ]
                )
            ),
            "COG": circular_mean_deg(
                base[
                    "COG"
                ]
            ),
            "T": float(
                np.mean(
                    base[
                        "T"
                    ]
                )
            ),
            "H": float(
                np.mean(
                    base[
                        "H"
                    ]
                )
            ),
            "P": float(
                np.mean(
                    base[
                        "P"
                    ]
                )
            ),
        }

        rows.append(
            row
        )

    df = pd.DataFrame(
        rows
    )

    if df.empty:
        return df

    df = df.sort_values(
        "block_start_ns"
    ).reset_index(
        drop=True
    )

    block_ns = (
        TRACK_P_RESAMPLE_MIN
        * 60_000_000_000
    )

    contiguous = np.zeros(
        len(
            df
        ),
        dtype=bool,
    )

    contiguous[
        1:
    ] = (
        np.diff(
            df[
                "block_start_ns"
            ].to_numpy(
                dtype=np.int64
            )
        )
        == block_ns
    )

    for base, delta in [
        (
            "T",
            "dT",
        ),
        (
            "P",
            "dP",
        ),
        (
            "H",
            "dH",
        ),
    ]:
        diff = np.full(
            len(
                df
            ),
            np.nan,
            dtype=np.float64,
        )

        x = df[
            base
        ].to_numpy(
            dtype=np.float64
        )

        diff[
            1:
        ] = (
            x[
                1:
            ]
            - x[
                :-1
            ]
        )

        diff[
            ~contiguous
        ] = np.nan

        df[
            delta
        ] = diff

    return df


def split_bounds_from_proposed(
    proposed: Dict[str, np.ndarray],
):
    context = np.asarray(
        proposed[
            "context_end_time_ns"
        ],
        dtype=np.int64,
    ).reshape(
        -1
    )

    targets = np.asarray(
        proposed[
            "target_time_ns"
        ],
        dtype=np.int64,
    )

    return {
        "min_context_end_ns": int(
            context.min()
        ),
        "max_context_end_ns": int(
            context.max()
        ),
        "min_target_ns": int(
            targets.min()
        ),
        "max_target_ns": int(
            targets.max()
        ),
    }


def build_track_p_windows(
    resampled: pd.DataFrame,
    *,
    dataset_id: str,
    split_bounds: Optional[Dict[str, int]],
):
    if resampled.empty:
        return {
            "X_raw": np.empty(
                (
                    0,
                    TRACK_P_LOOKBACK_STEPS,
                    len(
                        NODE_NAMES
                    ),
                ),
                dtype=np.float32,
            ),
            "y_wind_raw": np.empty(
                (
                    0,
                    1,
                    2,
                ),
                dtype=np.float32,
            ),
            "context_end_time_ns": np.empty(
                0,
                dtype=np.int64,
            ),
            "target_time_ns": np.empty(
                (
                    0,
                    1,
                ),
                dtype=np.int64,
            ),
            "feature_names": np.asarray(
                NODE_NAMES,
                dtype="U16",
            ),
            "target_names": np.asarray(
                [
                    "U",
                    "V",
                ],
                dtype="U8",
            ),
            "horizons_min": np.asarray(
                [
                    10
                ],
                dtype=np.int64,
            ),
        }, {
            "dataset_id": dataset_id,
            "track": "P_paper_protocol_10min",
            "candidate_windows": 0,
            "accepted_samples": 0,
            "rejected_noncontiguous": 0,
            "rejected_nonfinite": 0,
            "rejected_split_boundary": 0,
        }

    values = resampled[
        NODE_NAMES
    ].to_numpy(
        dtype=np.float64
    )

    times = resampled[
        "block_start_ns"
    ].to_numpy(
        dtype=np.int64
    )

    block_ns = (
        TRACK_P_RESAMPLE_MIN
        * 60_000_000_000
    )

    Xs = []

    ys = []

    ces = []

    tts = []

    rejection = {
        "rejected_noncontiguous": 0,
        "rejected_nonfinite": 0,
        "rejected_split_boundary": 0,
    }

    candidate_windows = max(
        0,
        len(
            resampled
        )
        - TRACK_P_LOOKBACK_STEPS,
    )

    for target_idx in range(
        TRACK_P_LOOKBACK_STEPS,
        len(
            resampled
        ),
    ):
        start_idx = (
            target_idx
            - TRACK_P_LOOKBACK_STEPS
        )

        context_idx = np.arange(
            start_idx,
            target_idx,
        )

        expected = (
            times[
                start_idx
            ]
            + np.arange(
                TRACK_P_LOOKBACK_STEPS
                + 1,
                dtype=np.int64,
            )
            * block_ns
        )

        actual = times[
            start_idx:
            target_idx
            + 1
        ]

        if not np.array_equal(
            actual,
            expected,
        ):
            rejection[
                "rejected_noncontiguous"
            ] += 1
            continue

        X = values[
            context_idx
        ]

        y = values[
            target_idx,
            0:2,
        ]

        if (
            not np.isfinite(
                X
            ).all()
            or not np.isfinite(
                y
            ).all()
        ):
            rejection[
                "rejected_nonfinite"
            ] += 1
            continue

        ce = int(
            times[
                target_idx
                - 1
            ]
        )

        tt = int(
            times[
                target_idx
            ]
        )

        if (
            split_bounds
            is not None
        ):
            # Keep the complete historical and target timeline within the
            # original proposed-model split's temporal domain.
            context_start = int(
                times[
                    start_idx
                ]
            )

            if (
                context_start
                < (
                    split_bounds[
                        "min_context_end_ns"
                    ]
                    - (
                        TRACK_F_LOOKBACK_MIN
                        - 1
                    )
                    * 60_000_000_000
                )
                or tt
                > split_bounds[
                    "max_target_ns"
                ]
            ):
                rejection[
                    "rejected_split_boundary"
                ] += 1
                continue

        Xs.append(
            X.astype(
                np.float32
            )
        )

        ys.append(
            y[
                None,
                :
            ].astype(
                np.float32
            )
        )

        ces.append(
            ce
        )

        tts.append(
            [
                tt
            ]
        )

    if Xs:
        X_out = np.stack(
            Xs
        )

        y_out = np.stack(
            ys
        )

        ce_out = np.asarray(
            ces,
            dtype=np.int64,
        )

        tt_out = np.asarray(
            tts,
            dtype=np.int64,
        )

    else:
        X_out = np.empty(
            (
                0,
                TRACK_P_LOOKBACK_STEPS,
                len(
                    NODE_NAMES
                ),
            ),
            dtype=np.float32,
        )

        y_out = np.empty(
            (
                0,
                1,
                2,
            ),
            dtype=np.float32,
        )

        ce_out = np.empty(
            0,
            dtype=np.int64,
        )

        tt_out = np.empty(
            (
                0,
                1,
            ),
            dtype=np.int64,
        )

    audit = {
        "dataset_id": dataset_id,
        "track": "P_paper_protocol_10min",
        "candidate_windows": int(
            candidate_windows
        ),
        "accepted_samples": int(
            len(
                X_out
            )
        ),
        "accepted_fraction": (
            float(
                len(
                    X_out
                )
                / candidate_windows
            )
            if candidate_windows
            > 0
            else np.nan
        ),
        **{
            key: int(
                value
            )
            for key, value
            in rejection.items()
        },
    }

    return {
        "X_raw": X_out,
        "y_wind_raw": y_out,
        "context_end_time_ns": ce_out,
        "target_time_ns": tt_out,
        "feature_names": np.asarray(
            NODE_NAMES,
            dtype="U16",
        ),
        "target_names": np.asarray(
            [
                "U",
                "V",
            ],
            dtype="U8",
        ),
        "horizons_min": np.asarray(
            [
                10
            ],
            dtype=np.int64,
        ),
    }, audit


def save_npz(
    path: Path,
    payload: Dict[str, np.ndarray],
):
    np.savez_compressed(
        path,
        **payload,
    )


def pressure_gap_audit(
    raw: RawSeries,
    dataset_key: str,
):
    P = np.asarray(
        raw.values[
            "P"
        ],
        dtype=np.float64,
    )

    valid = np.isfinite(
        P
    )

    time_ns = raw.time_ns

    rows = []

    if len(
        P
    ) == 0:
        return {
            "raw_dataset_key": dataset_key,
            "rows": 0,
            "pressure_valid_fraction": np.nan,
            "pressure_missing_rows": 0,
            "longest_missing_gap_minutes": np.nan,
            "longest_missing_gap_start": "",
            "longest_missing_gap_end": "",
        }

    longest_count = 0

    longest_start = None

    longest_end = None

    start = None

    for i, is_valid in enumerate(
        valid
    ):
        if not is_valid:
            if (
                start
                is None
            ):
                start = i

        elif (
            start
            is not None
        ):
            count = (
                i
                - start
            )

            if count > longest_count:
                longest_count = count
                longest_start = time_ns[
                    start
                ]
                longest_end = time_ns[
                    i
                    - 1
                ]

            start = None

    if (
        start
        is not None
    ):
        count = (
            len(
                valid
            )
            - start
        )

        if count > longest_count:
            longest_count = count
            longest_start = time_ns[
                start
            ]
            longest_end = time_ns[
                -1
            ]

    return {
        "raw_dataset_key": dataset_key,
        "rows": int(
            len(
                P
            )
        ),
        "pressure_valid_fraction": float(
            np.mean(
                valid
            )
        ),
        "pressure_missing_rows": int(
            (
                ~valid
            ).sum()
        ),
        "longest_missing_gap_minutes": int(
            longest_count
        ),
        "longest_missing_gap_start": (
            ns_to_iso(
                longest_start
            )
            if longest_start
            is not None
            else ""
        ),
        "longest_missing_gap_end": (
            ns_to_iso(
                longest_end
            )
            if longest_end
            is not None
            else ""
        ),
    }


def raw_key_for_dataset_id(
    dataset_id: str,
):
    if dataset_id.startswith(
        "SD1090"
    ):
        return "SD1090"

    if "2024" in dataset_id:
        return "SD1033_2024"

    if "2023" in dataset_id:
        return "SD1033_2023"

    if "2022" in dataset_id:
        return "SD1033_2022"

    raise ValueError(
        dataset_id
    )


def subset_track_p_to_time_domain(
    resampled: pd.DataFrame,
    proposed: Dict[str, np.ndarray],
):
    """
    External Track-P dataset should cover the same broad time domain as the
    corresponding proposed-model external dataset.

    Since Track-P targets are 10-min aggregated blocks, exact target timestamps
    need not coincide with Track-F/proposed point forecasts.
    """
    target_times = np.asarray(
        proposed[
            "target_time_ns"
        ],
        dtype=np.int64,
    )

    context_times = np.asarray(
        proposed[
            "context_end_time_ns"
        ],
        dtype=np.int64,
    ).reshape(
        -1
    )

    lower = int(
        min(
            context_times.min(),
            target_times.min(),
        )
        - TRACK_F_LOOKBACK_MIN
        * 60_000_000_000
    )

    upper = int(
        max(
            context_times.max(),
            target_times.max(),
        )
        + TRACK_P_RESAMPLE_MIN
        * 60_000_000_000
    )

    return resampled.loc[
        (
            resampled[
                "block_start_ns"
            ]
            >= lower
        )
        & (
            resampled[
                "block_start_ns"
            ]
            <= upper
        )
    ].reset_index(
        drop=True
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        default=str(
            DEFAULT_PROJECT_ROOT
        ),
    )

    parser.add_argument(
        "--frozen-dataset-dir",
        default=None,
    )

    parser.add_argument(
        "--external-root",
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--raw-map-json",
        default=None,
    )

    parser.add_argument(
        "--raw-root",
        action="append",
        default=None,
        help=(
            "Additional raw-data search root. Can be supplied multiple times."
        ),
    )

    parser.add_argument(
        "--skip-sd1090-test",
        action="store_true",
        help=(
            "Do not build SD1090 test files at this stage. "
            "Recommended during model-development setup."
        ),
    )

    parser.add_argument(
        "--skip-external",
        action="store_true",
        help=(
            "Only build SD1090 development data."
        ),
    )

    args = parser.parse_args()

    project_root = Path(
        args.project_root
    )

    frozen_dataset_dir = (
        Path(
            args.frozen_dataset_dir
        )
        if args.frozen_dataset_dir
        else (
            project_root
            / "data"
            / "forecasting"
            / "SD1090_TPOS2024_JointForecasting_v0_1"
        )
    )

    external_root = (
        Path(
            args.external_root
        )
        if args.external_root
        else (
            project_root
            / "data"
            / "external"
            / "SD1033_external_forecasting_v0_1"
        )
    )

    output_dir = (
        Path(
            args.output_dir
        )
        if args.output_dir
        else (
            frozen_dataset_dir
            / "11C0_BP_STGNN_comparison_datasets_v0_1"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    track_f_dir = (
        output_dir
        / "track_F_common_task"
    )

    track_p_dir = (
        output_dir
        / "track_P_paper_protocol"
    )

    track_f_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    track_p_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_roots = [
        project_root
        / "data"
        / "raw",
        project_root
        / "data"
        / "external",
    ]

    if args.raw_root:
        raw_roots.extend(
            Path(
                x
            )
            for x in args.raw_root
        )

    raw_map = resolve_raw_map(
        raw_roots,
        (
            Path(
                args.raw_map_json
            )
            if args.raw_map_json
            else None
        ),
    )

    required_sd1090 = "SD1090"

    if required_sd1090 not in raw_map:
        raise FileNotFoundError(
            "Could not auto-discover SD1090 raw NetCDF. "
            "Provide --raw-map-json."
        )

    M = save_mask_and_schema(
        output_dir
    )

    log(
        "="
        * 112
    )

    log(
        "11C0 - BUILD BP-STGNN COMPARISON DATASETS"
    )

    log(
        "="
        * 112
    )

    log(
        f"script_version       : {SCRIPT_VERSION}"
    )

    log(
        f"frozen dataset dir   : {frozen_dataset_dir}"
    )

    log(
        f"external root        : {external_root}"
    )

    log(
        f"output dir           : {output_dir}"
    )

    log(
        f"physical-mask ones   : {int(M.sum())}/100"
    )

    log(
        "Track F              : 1-min, lookback=60, horizons=1/2/3/5/10"
    )

    log(
        "Track P              : 10-min resampled, lookback=6, next-step U/V"
    )

    log(
        "pressure interpolation: NONE"
    )

    log(
        "model training       : NONE"
    )

    log("")

    raw_file_rows = []

    mapping_rows = []

    pressure_rows = []

    raw_cache: Dict[str, RawSeries] = {}

    native_node_cache: Dict[str, np.ndarray] = {}

    resampled_cache: Dict[str, pd.DataFrame] = {}

    for key, path in raw_map.items():
        log(
            f"[RAW] {key}: {path}"
        )

        raw = read_raw_series(
            path
        )

        raw_cache[
            key
        ] = raw

        native_node_cache[
            key
        ] = build_native_node_matrix(
            raw
        )

        resampled_cache[
            key
        ] = resample_10min_complete_blocks(
            raw
        )

        raw_file_rows.append({
            "raw_dataset_key": key,
            "path": str(
                path
            ),
            "rows": int(
                len(
                    raw.time_ns
                )
            ),
            "start": ns_to_iso(
                raw.time_ns[
                    0
                ]
            ),
            "end": ns_to_iso(
                raw.time_ns[
                    -1
                ]
            ),
            "complete_10min_blocks": int(
                len(
                    resampled_cache[
                        key
                    ]
                )
            ),
        })

        for canonical, raw_name in raw.variable_map.items():
            mapping_rows.append({
                "raw_dataset_key": key,
                "canonical": canonical,
                "raw_variable": raw_name,
                "units": raw.units.get(
                    canonical,
                    "",
                ),
            })

        pressure_rows.append(
            pressure_gap_audit(
                raw,
                key,
            )
        )

    pd.DataFrame(
        raw_file_rows
    ).to_csv(
        output_dir
        / "raw_file_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        mapping_rows
    ).to_csv(
        output_dir
        / "variable_mapping_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        pressure_rows
    ).to_csv(
        output_dir
        / "pressure_coverage_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    track_f_audits = []

    track_p_audits = []

    coverage_rows = []

    # ------------------------------------------------------------------
    # SD1090 frozen splits
    # ------------------------------------------------------------------
    sd1090_splits = [
        "train",
        "validation",
    ]

    if not args.skip_sd1090_test:
        sd1090_splits.append(
            "test"
        )

    raw1090 = raw_cache[
        "SD1090"
    ]

    nodes1090 = native_node_cache[
        "SD1090"
    ]

    resampled1090 = resampled_cache[
        "SD1090"
    ]

    for split in sd1090_splits:
        proposed_path = proposed_split_path(
            frozen_dataset_dir,
            split,
        )

        proposed = load_proposed_npz(
            proposed_path
        )

        dataset_id = (
            f"SD1090_{split}"
        )

        log(
            f"\n[BUILD] {dataset_id}"
        )

        track_f, audit_f = build_track_f_common_windows(
            raw1090,
            nodes1090,
            proposed,
            dataset_id=dataset_id,
        )

        save_npz(
            track_f_dir
            / f"{dataset_id}.npz",
            track_f,
        )

        track_f_audits.append(
            audit_f
        )

        split_bounds = split_bounds_from_proposed(
            proposed
        )

        resampled_subset = subset_track_p_to_time_domain(
            resampled1090,
            proposed,
        )

        track_p, audit_p = build_track_p_windows(
            resampled_subset,
            dataset_id=dataset_id,
            split_bounds=split_bounds,
        )

        save_npz(
            track_p_dir
            / f"{dataset_id}.npz",
            track_p,
        )

        track_p_audits.append(
            audit_p
        )

        coverage_rows.append({
            "dataset_id": dataset_id,
            "raw_key": "SD1090",
            "proposed_samples": int(
                audit_f[
                    "proposed_samples"
                ]
            ),
            "track_F_common_samples": int(
                audit_f[
                    "accepted_samples"
                ]
            ),
            "track_F_coverage_fraction": float(
                audit_f[
                    "accepted_fraction"
                ]
            ),
            "track_P_samples": int(
                audit_p[
                    "accepted_samples"
                ]
            ),
        })

        log(
            f"  Track F: {audit_f['accepted_samples']}/"
            f"{audit_f['proposed_samples']} "
            f"({100.0*audit_f['accepted_fraction']:.2f}%) common samples"
        )

        log(
            f"  Track P: {audit_p['accepted_samples']} paper-protocol windows"
        )

    # ------------------------------------------------------------------
    # External frozen datasets
    # ------------------------------------------------------------------
    if not args.skip_external:
        for dataset_id in PROPOSED_EXTERNAL_IDS:
            proposed_path = (
                external_root
                / "datasets"
                / f"{dataset_id}.npz"
            )

            if not proposed_path.exists():
                log(
                    f"[SKIP] external proposed NPZ not found: {proposed_path}"
                )
                continue

            raw_key = raw_key_for_dataset_id(
                dataset_id
            )

            if raw_key not in raw_cache:
                log(
                    f"[SKIP] raw NetCDF not resolved for {dataset_id} "
                    f"(expected raw key {raw_key})."
                )
                continue

            proposed = load_proposed_npz(
                proposed_path
            )

            raw = raw_cache[
                raw_key
            ]

            nodes = native_node_cache[
                raw_key
            ]

            resampled = resampled_cache[
                raw_key
            ]

            log(
                f"\n[BUILD] {dataset_id}"
            )

            track_f, audit_f = build_track_f_common_windows(
                raw,
                nodes,
                proposed,
                dataset_id=dataset_id,
            )

            save_npz(
                track_f_dir
                / f"{dataset_id}.npz",
                track_f,
            )

            track_f_audits.append(
                audit_f
            )

            resampled_subset = subset_track_p_to_time_domain(
                resampled,
                proposed,
            )

            track_p, audit_p = build_track_p_windows(
                resampled_subset,
                dataset_id=dataset_id,
                split_bounds=None,
            )

            save_npz(
                track_p_dir
                / f"{dataset_id}.npz",
                track_p,
            )

            track_p_audits.append(
                audit_p
            )

            coverage_rows.append({
                "dataset_id": dataset_id,
                "raw_key": raw_key,
                "proposed_samples": int(
                    audit_f[
                        "proposed_samples"
                    ]
                ),
                "track_F_common_samples": int(
                    audit_f[
                        "accepted_samples"
                    ]
                ),
                "track_F_coverage_fraction": float(
                    audit_f[
                        "accepted_fraction"
                    ]
                ),
                "track_P_samples": int(
                    audit_p[
                        "accepted_samples"
                    ]
                ),
            })

            log(
                f"  Track F: {audit_f['accepted_samples']}/"
                f"{audit_f['proposed_samples']} "
                f"({100.0*audit_f['accepted_fraction']:.2f}%) common samples"
            )

            log(
                f"  Track P: {audit_p['accepted_samples']} paper-protocol windows"
            )

    track_f_audit_df = pd.DataFrame(
        track_f_audits
    )

    track_p_audit_df = pd.DataFrame(
        track_p_audits
    )

    coverage_df = pd.DataFrame(
        coverage_rows
    )

    track_f_audit_df.to_csv(
        output_dir
        / "track_F_sample_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    track_p_audit_df.to_csv(
        output_dir
        / "track_P_sample_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    coverage_df.to_csv(
        output_dir
        / "comparison_coverage_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "script_version": SCRIPT_VERSION,
        "stage": "11C0",
        "paper": {
            "method": "BP-STGNN",
            "paper_exact_nodes": NODE_NAMES,
            "node_groups": NODE_GROUPS,
            "physical_mask_allowed_entries": int(
                M.sum()
            ),
            "physical_mask_shape": list(
                M.shape
            ),
            "paper_raw_resolution_min": 1,
            "paper_model_resolution_min": 10,
        },
        "track_F_common_task": {
            "resolution_min": 1,
            "lookback_min": TRACK_F_LOOKBACK_MIN,
            "horizons_min": HORIZONS_MIN,
            "feature_names": NODE_NAMES,
            "target_names": [
                "U",
                "V",
            ],
            "timestamp_alignment": (
                "Exact context_end_time_ns and target_time_ns inherited from "
                "frozen proposed-model datasets."
            ),
            "differential_rule": (
                "First difference between contiguous 1-min samples."
            ),
            "pressure_rule": (
                "No interpolation. Window rejected if any required node is "
                "non-finite in the 60-min context."
            ),
        },
        "track_P_paper_protocol": {
            "resolution_min": TRACK_P_RESAMPLE_MIN,
            "lookback_steps": TRACK_P_LOOKBACK_STEPS,
            "lookback_wallclock_min": (
                TRACK_P_LOOKBACK_STEPS
                * TRACK_P_RESAMPLE_MIN
            ),
            "horizon_min": 10,
            "feature_names": NODE_NAMES,
            "target_names": [
                "U",
                "V",
            ],
            "project_adaptations_due_to_paper_underspecification": {
                "linear_resampling": (
                    "Arithmetic mean over each complete 10-min block."
                ),
                "COG_resampling": (
                    "Circular mean over each complete 10-min block."
                ),
                "differential_nodes": (
                    "First differences AFTER 10-min resampling."
                ),
                "lookback": (
                    "6 resampled steps = 60 min to match the proposed model's "
                    "historical wall-clock span."
                ),
                "target": (
                    "Immediately following 10-min resampled U/V block."
                ),
            },
            "pressure_rule": (
                "No interpolation. A 10-min block is retained only if all ten "
                "raw 1-min observations are present and all seven required "
                "base variables are finite."
            ),
        },
        "scientific_firewall": {
            "model_training": False,
            "scaler_fitting": False,
            "hyperparameter_selection": False,
            "proposed_model_modified": False,
            "long_pressure_gap_interpolated": False,
            "external_performance_metrics_computed": False,
        },
        "raw_map": {
            key: str(
                value
            )
            for key, value
            in raw_map.items()
        },
        "output_dir": str(
            output_dir
        ),
    }

    save_json(
        output_dir
        / "preprocessing_manifest.json",
        manifest,
    )

    with (
        output_dir
        / "dataset_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "11C0 BP-STGNN Comparison Dataset Report\n"
        )

        f.write(
            "="
            * 112
            + "\n\n"
        )

        f.write(
            "NO MODEL TRAINING WAS PERFORMED.\n"
        )

        f.write(
            "Physics-Compact-Vessel-Residual v1.0 remains frozen.\n"
        )

        f.write(
            "No long pressure gap was interpolated.\n\n"
        )

        f.write(
            "Track F -- common-task 1-min alignment\n"
        )

        f.write(
            "-"
            * 112
            + "\n"
        )

        f.write(
            track_f_audit_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nTrack P -- paper-protocol 10-min adaptation\n"
        )

        f.write(
            "-"
            * 112
            + "\n"
        )

        f.write(
            track_p_audit_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nPressure audit\n"
        )

        f.write(
            "-"
            * 112
            + "\n"
        )

        f.write(
            pd.DataFrame(
                pressure_rows
            ).to_string(
                index=False
            )
        )

        f.write(
            "\n\nComparison coverage\n"
        )

        f.write(
            "-"
            * 112
            + "\n"
        )

        f.write(
            coverage_df.to_string(
                index=False
            )
        )

        f.write(
            "\n"
        )

    log("")
    log(
        "="
        * 112
    )

    log(
        "11C0 DATASET BUILD SUMMARY"
    )

    log(
        "="
        * 112
    )

    if not coverage_df.empty:
        for row in coverage_df.itertuples():
            log(
                f"{row.dataset_id}: "
                f"Track-F common={row.track_F_common_samples}/"
                f"{row.proposed_samples} "
                f"({100.0*row.track_F_coverage_fraction:.2f}%) | "
                f"Track-P={row.track_P_samples}"
            )

    log("")
    log(
        "Pressure coverage:"
    )

    for row in pd.DataFrame(
        pressure_rows
    ).itertuples():
        log(
            f"  {row.raw_dataset_key}: valid="
            f"{100.0*row.pressure_valid_fraction:.2f}% | "
            f"longest missing gap={row.longest_missing_gap_minutes} min | "
            f"{row.longest_missing_gap_start} -> "
            f"{row.longest_missing_gap_end}"
        )

    log("")
    log(
        "[POLICY] No model training/scaler fitting/hyperparameter selection."
    )

    log(
        "[POLICY] No multi-day pressure gap was interpolated."
    )

    log(
        "[POLICY] Track-F samples are exact timestamp intersections with the "
        "frozen proposed-model datasets."
    )

    log(
        "[POLICY] Track-P preprocessing adaptations are recorded explicitly "
        "in preprocessing_manifest.json."
    )

    log(
        f"[DONE] 11C0 outputs: {output_dir}"
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
