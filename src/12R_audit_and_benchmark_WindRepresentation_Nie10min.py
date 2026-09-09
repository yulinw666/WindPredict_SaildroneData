# -*- coding: utf-8 -*-
r"""
12R_audit_and_benchmark_WindRepresentation_Nie10min.py

Stage 12R
=========
Wind-representation audit and development-only representation benchmark for the
Nie/BP-STGNN-aligned Saildrone task.

Frozen forecasting protocol
----------------------------
Data resolution:
    10 min

Historical input:
    6 time steps = 60 min context

Forecast target:
    immediately following block = +10 min

Stage-12B already constructed exactly aligned samples with:
    X [N,6,...]
    target at context_end + 10 min

This stage REUSES those frozen timestamps and never creates a different sample
set for different feature representations.

Why Stage 12R?
--------------
Before adding Kalman filtering, a circular Transformer, VMD, or separate
wind-speed/wind-direction neural branches, answer three simpler questions:

1) RAW DATA PRECISION / CONSISTENCY
   Are Saildrone WIND_SPEED_MEAN and WIND_FROM_MEAN numerically equivalent to
   the wind magnitude and meteorological FROM direction reconstructed from
   UWND_MEAN / VWND_MEAN?

   WS_uv = sqrt(U^2 + V^2)
   WD_uv = atan2(-U, -V) mod 360 deg

   Also inspect:
       dtype
       scale_factor
       add_offset
       least_significant_digit
       _FillValue
       empirical decimal quantization grid

2) FEATURE ABLATION
   For true-wind-only prediction, do SOG/COG/wing variables help?

3) REPRESENTATION ABLATION
   Is a polar representation easier for a simple linear model than direct
   Cartesian U/V prediction?

Development-only missions
-------------------------
    Antarctic
    Atlantic
    West Coast

Tropical Atlantic is NEVER opened in this script.

Exactly aligned forecasting candidates
---------------------------------------
A) Full9-UV-Ridge
   input:
       U,V,T,RH,SOG,COG_sin,COG_cos,WING_sin,WING_cos
   target:
       U,V
   This is the old 9-feature Cartesian Ridge reference.

B) Met4-UV-Ridge
   input:
       U,V,T,RH
   target:
       U,V

C) Wind2-UV-Ridge
   input:
       U,V
   target:
       U,V

D) Polar5-Ridge
   input:
       WS, sin(WD), cos(WD), T, RH
   target heads:
       speed head     -> WS
       direction head -> sin(WD), cos(WD)

   The predicted direction vector is projected back to the unit circle before
   reconstructing:
       U = -WS * sin(WD)
       V = -WS * cos(WD)

IMPORTANT:
-------
Polar5 uses WS/WD DERIVED FROM U/V, not raw WIND_SPEED_MEAN/WIND_FROM_MEAN.
Therefore the Cartesian-vs-polar benchmark changes only representation, not
sensor information. Raw WS/WD are audited separately.

LOMO protocol
-------------
Holdout Antarctic  <- train Atlantic + West Coast
Holdout Atlantic   <- train Antarctic + West Coast
Holdout West Coast <- train Antarctic + Atlantic

All scalers are fold-train-only.

Ridge
-----
alpha = 1.0

Representation score
--------------------
For each fold, Met4-UV-Ridge is the representation reference:

    Score =
        0.25 * U_RMSE  / Met4_U
      + 0.35 * V_RMSE  / Met4_V
      + 0.25 * WS_RMSE / Met4_WS
      + 0.15 * WD_RMSE / Met4_WD

Predeclared "Polar is promising" rule
-------------------------------------
Polar5-Ridge is considered promising enough to justify a dedicated
speed/direction model (Kalman + circular attention) only if ALL are true:

    mean score vs Met4 < 0.99
    Polar wins >= 2/3 held-out missions
    mean vector RMSE does not worsen by > 0.5%
    and at least one of:
        V RMSE improves > 1%
        WD RMSE improves > 1%

This rule is development-only.

Raw audit outputs
-----------------
raw_variable_metadata.csv
raw_quantization_summary.csv
raw_uv_vs_wswd_consistency.csv

Forecast benchmark outputs
--------------------------
representation_fold_results.csv
representation_summary.csv
selected_representation.json
12R_REPORT.txt

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\12R_audit_and_benchmark_WindRepresentation_Nie10min.py" --raw-dir "D:\project\WindPredict_SaildroneData\data\raw\RawData" --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12R_WindRepresentationAudit_Nie10min_v0_1"

If your Stage-12G folder is elsewhere, only --dataset-dir needs changing.

NOTE ON DATASET PATH
--------------------
The script first tries the supplied dataset directory for:
    track_J_joint_compatible/Antarctic.npz
If absent, it also accepts a Stage-12B dataset directory containing the same
track_J_joint_compatible subfolder.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import Ridge
except Exception as exc:
    raise RuntimeError(
        "scikit-learn is required. Activate the WindPredict environment."
    ) from exc


SCRIPT_VERSION = "0.1.0-12R-wind-representation-audit"

DEFAULT_PROJECT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")
DEFAULT_RAW_DIR = DEFAULT_PROJECT_ROOT / "data" / "raw" / "RawData"

DEFAULT_DATASET_CANDIDATES = [
    (
        DEFAULT_PROJECT_ROOT
        / "data"
        / "forecasting"
        / "12G_Nie_point_sampled_benchmark_v0_1"
        / "dataset"
    ),
    (
        DEFAULT_PROJECT_ROOT
        / "data"
        / "forecasting"
        / "12B_Nie_same_mission_benchmark_v0_1"
    ),
]

DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12R_WindRepresentationAudit_Nie10min_v0_1"
)

DEVELOPMENT_MISSIONS = [
    "Antarctic",
    "Atlantic",
    "West Coast",
]

LOOKBACK_STEPS = 6
STEP_MINUTES = 10
FORECAST_MINUTES = 10
TEN_MIN_NS = 10 * 60 * 1_000_000_000

RIDGE_ALPHA = 1.0
EPS = 1e-12

FULL9_FEATURE_NAMES = [
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

MET4_INDICES = [0, 1, 2, 3]
WIND2_INDICES = [0, 1]

MET4_NAMES = ["U", "V", "T", "RH"]
WIND2_NAMES = ["U", "V"]
POLAR5_NAMES = ["WS", "WD_sin", "WD_cos", "T", "RH"]

RAW_ALIASES = {
    "U": [
        "UWND_MEAN",
        "uwnd_mean",
        "eastward_wind",
        "eastward_wind_mean",
        "wind_u_mean",
    ],
    "V": [
        "VWND_MEAN",
        "vwnd_mean",
        "northward_wind",
        "northward_wind_mean",
        "wind_v_mean",
    ],
    "WS": [
        "WIND_SPEED_MEAN",
        "wind_speed_mean",
        "wind_speed",
        "WIND_SPEED",
    ],
    "WD": [
        "WIND_FROM_MEAN",
        "wind_from_mean",
        "WIND_FROM",
        "wind_from_direction",
        "WIND_DIRECTION_MEAN",
        "wind_direction_mean",
    ],
}

SCORE_WEIGHTS = {
    "U": 0.25,
    "V": 0.35,
    "WS": 0.25,
    "WD": 0.15,
}

POLAR_SCORE_THRESHOLD = 0.99
POLAR_MIN_MISSION_WINS = 2
POLAR_MAX_VECTOR_WORSENING = 0.005
POLAR_MIN_V_OR_WD_IMPROVEMENT = 0.01

QUANTIZATION_STEPS = [
    1.0,
    0.1,
    0.01,
    0.001,
    0.0001,
    0.00001,
    0.000001,
]
QUANTIZATION_SAMPLE_LIMIT = 500_000

BP_STGNN_PUBLISHED_REFERENCE = {
    "U_RMSE_mps": 0.748640,
    "V_RMSE_mps": 0.705060,
    "WS_RMSE_mps": 0.699400,
    "WD_RMSE_deg": 5.584000,
}


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


# =============================================================================
# Stage-12B helper loading: mission discovery only
# =============================================================================

def load_stage12b(project_root: Path):
    path = (
        project_root
        / "src"
        / "12B_build_Nie_same_mission_benchmark_dataset.py"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Required Stage-12B source not found:\n"
            f"  {path}"
        )

    name = "stage12b_for_12r"

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot create import spec for {path}"
        )

    mod = importlib.util.module_from_spec(
        spec
    )

    # Robust Python 3.11+ dynamic import.
    sys.modules[name] = mod

    try:
        spec.loader.exec_module(
            mod
        )
    except Exception:
        sys.modules.pop(
            name,
            None,
        )
        raise

    for required in [
        "import_xarray",
        "discover_mission_files",
    ]:
        if not hasattr(
            mod,
            required,
        ):
            raise RuntimeError(
                f"Stage-12B missing helper: {required}"
            )

    return mod, path


def resolve_dataset_dir(
    requested: Path | None,
):
    candidates = []

    if requested is not None:
        candidates.append(
            requested
        )

    candidates.extend(
        DEFAULT_DATASET_CANDIDATES
    )

    seen = set()

    for base in candidates:
        base = Path(base)

        if str(base).lower() in seen:
            continue

        seen.add(
            str(base).lower()
        )

        # User may point directly to the dataset root.
        j_dir = (
            base
            / "track_J_joint_compatible"
        )

        if (
            j_dir
            / "Antarctic.npz"
        ).exists():
            return base

        # Or to a parent containing "dataset".
        j_dir2 = (
            base
            / "dataset"
            / "track_J_joint_compatible"
        )

        if (
            j_dir2
            / "Antarctic.npz"
        ).exists():
            return (
                base
                / "dataset"
            )

    raise FileNotFoundError(
        "Cannot locate track_J_joint_compatible/Antarctic.npz. "
        "Pass --dataset-dir explicitly."
    )


# =============================================================================
# RAW NetCDF precision / U-V vs WS-WD audit
# =============================================================================

def all_dataset_names(ds):
    names = []

    for source in [
        ds.variables,
        ds.coords,
        ds.data_vars,
    ]:
        for name in source:
            name = str(name)
            if name not in names:
                names.append(
                    name
                )

    return names


def resolve_alias(
    ds,
    aliases,
):
    names = all_dataset_names(
        ds
    )

    exact = set(
        names
    )

    lower = {
        n.lower(): n
        for n in names
    }

    for alias in aliases:
        if alias in exact:
            return alias

    for alias in aliases:
        hit = lower.get(
            alias.lower()
        )
        if hit is not None:
            return hit

    return None


def first_existing(
    mapping,
    keys,
    default=None,
):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def decoded_numeric_from_unscaled_da(
    da,
):
    """
    Decode a variable opened with mask_and_scale=False.
    """
    raw = np.asarray(
        da.values
    )

    try:
        x = raw.reshape(
            -1
        ).astype(
            np.float64
        )
    except Exception:
        return None

    attrs = dict(
        da.attrs
    )
    encoding = dict(
        da.encoding
    )

    fill_candidates = []

    for key in [
        "_FillValue",
        "missing_value",
    ]:
        v = first_existing(
            attrs,
            [key],
            None,
        )

        if v is None:
            v = first_existing(
                encoding,
                [key],
                None,
            )

        if v is not None:
            try:
                fill_candidates.extend(
                    np.asarray(v)
                    .reshape(-1)
                    .astype(float)
                    .tolist()
                )
            except Exception:
                pass

    for sentinel in fill_candidates:
        if np.isfinite(
            sentinel
        ):
            x[
                x == sentinel
            ] = np.nan

    x[
        np.abs(x)
        > 1e30
    ] = np.nan

    scale = attrs.get(
        "scale_factor",
        encoding.get(
            "scale_factor",
            1.0,
        ),
    )

    offset = attrs.get(
        "add_offset",
        encoding.get(
            "add_offset",
            0.0,
        ),
    )

    try:
        scale = float(
            scale
        )
    except Exception:
        scale = 1.0

    try:
        offset = float(
            offset
        )
    except Exception:
        offset = 0.0

    x = (
        x * scale
        + offset
    )

    return x


def scalar_attr(
    da,
    name,
):
    value = da.attrs.get(
        name,
        da.encoding.get(
            name,
            None,
        ),
    )

    if value is None:
        return None

    try:
        arr = np.asarray(
            value
        ).reshape(-1)

        if len(arr) == 1:
            v = arr[0]
            if isinstance(
                v,
                np.generic,
            ):
                v = v.item()
            return v

        return str(
            value
        )
    except Exception:
        return str(
            value
        )


def quantization_alignment_fraction(
    values,
    step,
):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return np.nan, np.nan

    nearest = (
        np.round(
            x / step
        )
        * step
    )

    err = np.abs(
        x - nearest
    )

    tolerance = max(
        1e-7,
        abs(step)
        * 1e-4,
    )

    return (
        float(
            np.mean(
                err <= tolerance
            )
        ),
        float(
            np.max(
                err
            )
        ),
    )


def infer_empirical_quantization(
    values,
):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return {
            "empirical_quantization_step": np.nan,
            "empirical_quantization_alignment_fraction": np.nan,
            "sample_count": 0,
        }

    # Deterministic thinning for memory/runtime stability.
    if len(x) > QUANTIZATION_SAMPLE_LIMIT:
        idx = np.linspace(
            0,
            len(x) - 1,
            QUANTIZATION_SAMPLE_LIMIT,
            dtype=np.int64,
        )
        x = x[
            idx
        ]

    best_step = np.nan
    best_fraction = np.nan

    for step in QUANTIZATION_STEPS:
        fraction, _ = (
            quantization_alignment_fraction(
                x,
                step,
            )
        )

        # Choose COARSEST grid that explains >=99.99% of sampled values.
        if (
            np.isfinite(
                fraction
            )
            and fraction >= 0.9999
        ):
            best_step = float(
                step
            )
            best_fraction = float(
                fraction
            )
            break

    if not np.isfinite(
        best_step
    ):
        # Report the finest tested alignment if no clean grid exists.
        best_step = float(
            QUANTIZATION_STEPS[-1]
        )
        best_fraction, _ = (
            quantization_alignment_fraction(
                x,
                best_step,
            )
        )

    decimals = max(
        0,
        int(
            round(
                -math.log10(
                    best_step
                )
            )
        ),
    )

    return {
        "empirical_quantization_step": (
            best_step
        ),
        "empirical_quantization_decimals": (
            decimals
        ),
        "empirical_quantization_alignment_fraction": (
            float(
                best_fraction
            )
        ),
        "sample_count": int(
            len(x)
        ),
    }


class RunningErrorStats:
    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.max_abs = 0.0
        self.abs_samples = []

    def update(
        self,
        errors,
    ):
        e = np.asarray(
            errors,
            dtype=np.float64,
        )

        e = e[
            np.isfinite(e)
        ]

        if len(e) == 0:
            return

        self.n += int(
            len(e)
        )

        self.sum += float(
            np.sum(
                e
            )
        )

        ae = np.abs(
            e
        )

        self.sum_abs += float(
            np.sum(
                ae
            )
        )

        self.sum_sq += float(
            np.sum(
                e ** 2
            )
        )

        self.max_abs = max(
            self.max_abs,
            float(
                np.max(
                    ae
                )
            ),
        )

        remaining = (
            QUANTIZATION_SAMPLE_LIMIT
            - sum(
                len(x)
                for x
                in self.abs_samples
            )
        )

        if remaining > 0:
            if len(ae) <= remaining:
                self.abs_samples.append(
                    ae.astype(
                        np.float32
                    )
                )
            else:
                idx = np.linspace(
                    0,
                    len(ae) - 1,
                    remaining,
                    dtype=np.int64,
                )
                self.abs_samples.append(
                    ae[
                        idx
                    ].astype(
                        np.float32
                    )
                )

    def summary(self):
        if self.n == 0:
            return {
                "n": 0,
                "bias": np.nan,
                "MAE": np.nan,
                "RMSE": np.nan,
                "max_abs": np.nan,
                "p95_abs": np.nan,
                "p99_abs": np.nan,
            }

        if self.abs_samples:
            a = np.concatenate(
                self.abs_samples
            )
            p95 = float(
                np.quantile(
                    a,
                    0.95,
                )
            )
            p99 = float(
                np.quantile(
                    a,
                    0.99,
                )
            )
        else:
            p95 = np.nan
            p99 = np.nan

        return {
            "n": int(
                self.n
            ),
            "bias": float(
                self.sum
                / self.n
            ),
            "MAE": float(
                self.sum_abs
                / self.n
            ),
            "RMSE": float(
                math.sqrt(
                    self.sum_sq
                    / self.n
                )
            ),
            "max_abs": float(
                self.max_abs
            ),
            "p95_abs": p95,
            "p99_abs": p99,
        }


def meteorological_wd_from_uv(
    U,
    V,
):
    U = np.asarray(
        U,
        dtype=np.float64,
    )
    V = np.asarray(
        V,
        dtype=np.float64,
    )

    return (
        np.degrees(
            np.arctan2(
                -U,
                -V,
            )
        )
        % 360.0
    )


def mathematical_to_direction_from_uv(
    U,
    V,
):
    return (
        np.degrees(
            np.arctan2(
                V,
                U,
            )
        )
        % 360.0
    )


def circular_diff_deg(
    a,
    b,
):
    return (
        (
            np.asarray(
                a,
                dtype=np.float64,
            )
            - np.asarray(
                b,
                dtype=np.float64,
            )
            + 180.0
        )
        % 360.0
        - 180.0
    )


def run_raw_audit(
    *,
    stage12b,
    raw_dir: Path,
    output_dir: Path,
):
    xr = stage12b.import_xarray()

    grouped = stage12b.discover_mission_files(
        raw_dir
    )

    metadata_rows = []
    quant_collect = {
        mission: {
            key: []
            for key in RAW_ALIASES
        }
        for mission in DEVELOPMENT_MISSIONS
    }

    mission_stats = {}

    log("=" * 132)
    log(
        "12R PART A — RAW PRECISION / U-V vs WS-WD CONSISTENCY AUDIT"
    )
    log("=" * 132)
    log(
        "[FIREWALL] Opening development mission files only."
    )
    log("")

    for mission in DEVELOPMENT_MISSIONS:
        ws_stats = RunningErrorStats()
        wd_from_stats = RunningErrorStats()
        wd_to_stats = RunningErrorStats()

        files_all4 = 0
        rows_all4 = 0

        paths = grouped[
            mission
        ]

        log(
            f"[RAW AUDIT] {mission} | files={len(paths)}"
        )

        for file_index, path in enumerate(
            sorted(paths),
            start=1,
        ):
            if (
                file_index == 1
                or file_index == len(paths)
                or file_index % 25 == 0
            ):
                log(
                    f"  file {file_index}/{len(paths)}: {path.name}"
                )

            # IMPORTANT: mask_and_scale=False so storage metadata is visible.
            with xr.open_dataset(
                path,
                decode_times=False,
                mask_and_scale=False,
            ) as ds:
                resolved = {
                    key: resolve_alias(
                        ds,
                        aliases,
                    )
                    for key, aliases
                    in RAW_ALIASES.items()
                }

                arrays = {}

                for key in [
                    "U",
                    "V",
                    "WS",
                    "WD",
                ]:
                    raw_name = resolved[
                        key
                    ]

                    if raw_name is None:
                        metadata_rows.append({
                            "mission": mission,
                            "file": path.name,
                            "canonical": key,
                            "resolved_name": "",
                            "present": False,
                        })
                        continue

                    da = ds[
                        raw_name
                    ]

                    decoded = (
                        decoded_numeric_from_unscaled_da(
                            da
                        )
                    )

                    arrays[
                        key
                    ] = decoded

                    storage_dtype = str(
                        da.dtype
                    )

                    metadata_rows.append({
                        "mission": mission,
                        "file": path.name,
                        "canonical": key,
                        "resolved_name": raw_name,
                        "present": True,
                        "storage_dtype": storage_dtype,
                        "units": scalar_attr(
                            da,
                            "units",
                        ),
                        "long_name": scalar_attr(
                            da,
                            "long_name",
                        ),
                        "standard_name": scalar_attr(
                            da,
                            "standard_name",
                        ),
                        "scale_factor": scalar_attr(
                            da,
                            "scale_factor",
                        ),
                        "add_offset": scalar_attr(
                            da,
                            "add_offset",
                        ),
                        "least_significant_digit": scalar_attr(
                            da,
                            "least_significant_digit",
                        ),
                        "_FillValue": scalar_attr(
                            da,
                            "_FillValue",
                        ),
                        "valid_min": scalar_attr(
                            da,
                            "valid_min",
                        ),
                        "valid_max": scalar_attr(
                            da,
                            "valid_max",
                        ),
                        "decoded_finite_count": (
                            int(
                                np.isfinite(
                                    decoded
                                ).sum()
                            )
                            if decoded is not None
                            else 0
                        ),
                    })

                    if decoded is not None:
                        finite = decoded[
                            np.isfinite(
                                decoded
                            )
                        ]

                        current = sum(
                            len(x)
                            for x in quant_collect[
                                mission
                            ][
                                key
                            ]
                        )

                        remaining = (
                            QUANTIZATION_SAMPLE_LIMIT
                            - current
                        )

                        if (
                            remaining > 0
                            and len(finite) > 0
                        ):
                            if len(finite) <= remaining:
                                take = finite
                            else:
                                idx = np.linspace(
                                    0,
                                    len(finite) - 1,
                                    remaining,
                                    dtype=np.int64,
                                )
                                take = finite[
                                    idx
                                ]

                            quant_collect[
                                mission
                            ][
                                key
                            ].append(
                                np.asarray(
                                    take,
                                    dtype=np.float32,
                                )
                            )

                if all(
                    key in arrays
                    and arrays[key] is not None
                    for key in [
                        "U",
                        "V",
                        "WS",
                        "WD",
                    ]
                ):
                    lengths = {
                        len(
                            arrays[key]
                        )
                        for key in [
                            "U",
                            "V",
                            "WS",
                            "WD",
                        ]
                    }

                    if len(
                        lengths
                    ) != 1:
                        log(
                            "    WARNING: U/V/WS/WD flattened lengths differ; "
                            "skipping consistency audit for this file."
                        )
                        continue

                    U = arrays[
                        "U"
                    ]
                    V = arrays[
                        "V"
                    ]
                    WS = arrays[
                        "WS"
                    ]
                    WD = arrays[
                        "WD"
                    ]

                    mask = (
                        np.isfinite(U)
                        & np.isfinite(V)
                        & np.isfinite(WS)
                        & np.isfinite(WD)
                    )

                    if not np.any(
                        mask
                    ):
                        continue

                    U = U[
                        mask
                    ]
                    V = V[
                        mask
                    ]
                    WS = WS[
                        mask
                    ]
                    WD = WD[
                        mask
                    ]

                    ws_uv = np.hypot(
                        U,
                        V,
                    )

                    wd_from_uv = (
                        meteorological_wd_from_uv(
                            U,
                            V,
                        )
                    )

                    wd_to_uv = (
                        mathematical_to_direction_from_uv(
                            U,
                            V,
                        )
                    )

                    ws_stats.update(
                        WS
                        - ws_uv
                    )

                    wd_from_stats.update(
                        circular_diff_deg(
                            WD,
                            wd_from_uv,
                        )
                    )

                    wd_to_stats.update(
                        circular_diff_deg(
                            WD,
                            wd_to_uv,
                        )
                    )

                    files_all4 += 1
                    rows_all4 += int(
                        len(U)
                    )

        ws_s = ws_stats.summary()
        wd_from_s = wd_from_stats.summary()
        wd_to_s = wd_to_stats.summary()

        direction_interpretation = (
            "meteorological_FROM"
            if (
                np.isfinite(
                    wd_from_s[
                        "RMSE"
                    ]
                )
                and (
                    not np.isfinite(
                        wd_to_s[
                            "RMSE"
                        ]
                    )
                    or wd_from_s[
                        "RMSE"
                    ]
                    <= wd_to_s[
                        "RMSE"
                    ]
                )
            )
            else "mathematical_TO_or_other"
        )

        mission_stats[
            mission
        ] = {
            "mission": mission,
            "files_with_all_U_V_WS_WD": int(
                files_all4
            ),
            "rows_with_all_U_V_WS_WD": int(
                rows_all4
            ),
            "WS_minus_hypotUV_bias_mps": (
                ws_s["bias"]
            ),
            "WS_minus_hypotUV_MAE_mps": (
                ws_s["MAE"]
            ),
            "WS_minus_hypotUV_RMSE_mps": (
                ws_s["RMSE"]
            ),
            "WS_minus_hypotUV_p95_abs_mps": (
                ws_s["p95_abs"]
            ),
            "WS_minus_hypotUV_p99_abs_mps": (
                ws_s["p99_abs"]
            ),
            "WS_minus_hypotUV_max_abs_mps": (
                ws_s["max_abs"]
            ),
            "WD_raw_vs_UV_FROM_MAE_deg": (
                wd_from_s["MAE"]
            ),
            "WD_raw_vs_UV_FROM_RMSE_deg": (
                wd_from_s["RMSE"]
            ),
            "WD_raw_vs_UV_FROM_p95_abs_deg": (
                wd_from_s["p95_abs"]
            ),
            "WD_raw_vs_UV_FROM_p99_abs_deg": (
                wd_from_s["p99_abs"]
            ),
            "WD_raw_vs_UV_FROM_max_abs_deg": (
                wd_from_s["max_abs"]
            ),
            "WD_raw_vs_UV_TO_RMSE_deg": (
                wd_to_s["RMSE"]
            ),
            "inferred_raw_direction_convention": (
                direction_interpretation
            ),
        }

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    metadata_path = (
        output_dir
        / "raw_variable_metadata.csv"
    )

    metadata_df.to_csv(
        metadata_path,
        index=False,
        encoding="utf-8-sig",
    )

    quant_rows = []

    for mission in DEVELOPMENT_MISSIONS:
        for key in [
            "U",
            "V",
            "WS",
            "WD",
        ]:
            pieces = quant_collect[
                mission
            ][
                key
            ]

            if pieces:
                values = np.concatenate(
                    pieces
                )
            else:
                values = np.empty(
                    0,
                    dtype=np.float32,
                )

            q = infer_empirical_quantization(
                values
            )

            quant_rows.append({
                "mission": mission,
                "canonical": key,
                **q,
                "sample_min": (
                    float(
                        np.min(
                            values
                        )
                    )
                    if len(values)
                    else np.nan
                ),
                "sample_max": (
                    float(
                        np.max(
                            values
                        )
                    )
                    if len(values)
                    else np.nan
                ),
            })

    quant_df = pd.DataFrame(
        quant_rows
    )

    quant_path = (
        output_dir
        / "raw_quantization_summary.csv"
    )

    quant_df.to_csv(
        quant_path,
        index=False,
        encoding="utf-8-sig",
    )

    consistency_df = pd.DataFrame(
        [
            mission_stats[
                m
            ]
            for m in DEVELOPMENT_MISSIONS
        ]
    )

    consistency_path = (
        output_dir
        / "raw_uv_vs_wswd_consistency.csv"
    )

    consistency_df.to_csv(
        consistency_path,
        index=False,
        encoding="utf-8-sig",
    )

    return (
        metadata_df,
        quant_df,
        consistency_df,
        metadata_path,
        quant_path,
        consistency_path,
    )


# =============================================================================
# Frozen 6 x 10-min forecasting dataset
# =============================================================================

def track_j_dir(
    dataset_dir: Path,
):
    return (
        dataset_dir
        / "track_J_joint_compatible"
    )


def load_track_j(
    dataset_dir: Path,
    mission: str,
):
    safe = mission.replace(
        " ",
        "_",
    )

    path = (
        track_j_dir(
            dataset_dir
        )
        / f"{safe}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as z:
        required = [
            "X_raw",
            "y_wind_raw",
            "context_end_time_ns",
            "target_time_ns",
            "feature_names",
        ]

        missing = [
            x
            for x in required
            if x not in z.files
        ]

        if missing:
            raise KeyError(
                f"{path}: missing {missing}; available={z.files}"
            )

        X = np.asarray(
            z[
                "X_raw"
            ],
            dtype=np.float32,
        )

        y = np.asarray(
            z[
                "y_wind_raw"
            ],
            dtype=np.float32,
        )

        context = np.asarray(
            z[
                "context_end_time_ns"
            ],
            dtype=np.int64,
        ).reshape(
            -1
        )

        target = np.asarray(
            z[
                "target_time_ns"
            ],
            dtype=np.int64,
        ).reshape(
            -1
        )

        names = [
            str(x)
            for x in z[
                "feature_names"
            ].tolist()
        ]

        lookback = (
            int(
                np.asarray(
                    z[
                        "lookback_steps"
                    ]
                ).reshape(
                    -1
                )[0]
            )
            if "lookback_steps" in z.files
            else X.shape[
                1
            ]
        )

        step_minutes = (
            int(
                np.asarray(
                    z[
                        "step_minutes"
                    ]
                ).reshape(
                    -1
                )[0]
            )
            if "step_minutes" in z.files
            else STEP_MINUTES
        )

    if X.ndim != 3:
        raise RuntimeError(
            f"{mission}: expected 3D X, got {X.shape}"
        )

    if X.shape[
        1
    ] != LOOKBACK_STEPS:
        raise RuntimeError(
            f"{mission}: expected {LOOKBACK_STEPS} history steps, got {X.shape}"
        )

    if names != FULL9_FEATURE_NAMES:
        raise RuntimeError(
            f"{mission}: unexpected feature names {names}"
        )

    if y.ndim == 3:
        if y.shape[
            1:
        ] != (
            1,
            2,
        ):
            raise RuntimeError(
                f"{mission}: expected y [N,1,2], got {y.shape}"
            )

        y = y[
            :,
            0,
            :,
        ]

    if y.ndim != 2 or y.shape[
        1
    ] != 2:
        raise RuntimeError(
            f"{mission}: bad y shape {y.shape}"
        )

    if lookback != LOOKBACK_STEPS:
        raise RuntimeError(
            f"{mission}: lookback metadata={lookback}"
        )

    if step_minutes != STEP_MINUTES:
        raise RuntimeError(
            f"{mission}: step_minutes={step_minutes}, expected {STEP_MINUTES}"
        )

    if not np.all(
        target
        - context
        == TEN_MIN_NS
    ):
        raise RuntimeError(
            f"{mission}: target is not exactly +10 min."
        )

    if not (
        np.isfinite(
            X
        ).all()
        and np.isfinite(
            y
        ).all()
    ):
        raise RuntimeError(
            f"{mission}: non-finite values."
        )

    return {
        "X_full9": X,
        "y_uv": y,
        "context_end_time_ns": context,
        "target_time_ns": target,
        "mission": mission,
    }


def concat_missions(
    items,
):
    return {
        "X_full9": np.concatenate(
            [
                x[
                    "X_full9"
                ]
                for x
                in items
            ],
            axis=0,
        ),
        "y_uv": np.concatenate(
            [
                x[
                    "y_uv"
                ]
                for x
                in items
            ],
            axis=0,
        ),
    }


# =============================================================================
# Representations / metrics
# =============================================================================

def uv_to_polar_features(
    U,
    V,
):
    U = np.asarray(
        U,
        dtype=np.float64,
    )

    V = np.asarray(
        V,
        dtype=np.float64,
    )

    WS = np.hypot(
        U,
        V,
    )

    WD_rad = np.arctan2(
        -U,
        -V,
    )

    return (
        WS.astype(
            np.float32
        ),
        np.sin(
            WD_rad
        ).astype(
            np.float32
        ),
        np.cos(
            WD_rad
        ).astype(
            np.float32
        ),
    )


def make_met4(
    X_full9,
):
    return np.asarray(
        X_full9[
            :,
            :,
            MET4_INDICES
        ],
        dtype=np.float32,
    )


def make_wind2(
    X_full9,
):
    return np.asarray(
        X_full9[
            :,
            :,
            WIND2_INDICES
        ],
        dtype=np.float32,
    )


def make_polar5(
    X_full9,
):
    X = np.asarray(
        X_full9,
        dtype=np.float32,
    )

    U = X[
        :,
        :,
        0
    ]
    V = X[
        :,
        :,
        1
    ]

    WS, s, c = uv_to_polar_features(
        U,
        V,
    )

    T = X[
        :,
        :,
        2
    ]
    RH = X[
        :,
        :,
        3
    ]

    return np.stack(
        [
            WS,
            s,
            c,
            T,
            RH,
        ],
        axis=-1,
    ).astype(
        np.float32
    )


def target_polar(
    y_uv,
):
    y = np.asarray(
        y_uv,
        dtype=np.float32,
    )

    WS, s, c = uv_to_polar_features(
        y[
            :,
            0
        ],
        y[
            :,
            1
        ],
    )

    return (
        WS.reshape(
            -1,
            1
        ),
        np.stack(
            [
                s,
                c,
            ],
            axis=1,
        ),
    )


def fit_x_scaler(
    X,
):
    X64 = np.asarray(
        X,
        dtype=np.float64,
    )

    mean = np.mean(
        X64,
        axis=(
            0,
            1,
        ),
    )

    std = np.std(
        X64,
        axis=(
            0,
            1,
        ),
        ddof=0,
    )

    std = np.where(
        std < 1e-8,
        1.0,
        std,
    )

    return {
        "mean": mean.astype(
            np.float32
        ),
        "std": std.astype(
            np.float32
        ),
    }


def transform_X(
    X,
    scaler,
):
    return (
        (
            np.asarray(
                X,
                dtype=np.float32,
            )
            - scaler[
                "mean"
            ][
                None,
                None,
                :,
            ]
        )
        / scaler[
            "std"
        ][
            None,
            None,
            :,
        ]
    ).astype(
        np.float32
    )


def fit_y_scaler(
    y,
):
    y64 = np.asarray(
        y,
        dtype=np.float64,
    )

    mean = np.mean(
        y64,
        axis=0,
    )

    std = np.std(
        y64,
        axis=0,
        ddof=0,
    )

    std = np.where(
        std < 1e-8,
        1.0,
        std,
    )

    return {
        "mean": mean.astype(
            np.float32
        ),
        "std": std.astype(
            np.float32
        ),
    }


def transform_y(
    y,
    scaler,
):
    return (
        (
            np.asarray(
                y,
                dtype=np.float32,
            )
            - scaler[
                "mean"
            ][
                None,
                :,
            ]
        )
        / scaler[
            "std"
        ][
            None,
            :,
        ]
    ).astype(
        np.float32
    )


def inverse_y(
    yz,
    scaler,
):
    return (
        np.asarray(
            yz,
            dtype=np.float32,
        )
        * scaler[
            "std"
        ][
            None,
            :,
        ]
        + scaler[
            "mean"
        ][
            None,
            :,
        ]
    ).astype(
        np.float32
    )


def fit_predict_cartesian_ridge(
    X_train,
    y_train,
    X_val,
):
    xs = fit_x_scaler(
        X_train
    )

    ys = fit_y_scaler(
        y_train
    )

    Xtr = transform_X(
        X_train,
        xs,
    ).reshape(
        len(
            X_train
        ),
        -1,
    )

    Xva = transform_X(
        X_val,
        xs,
    ).reshape(
        len(
            X_val
        ),
        -1,
    )

    ytr = transform_y(
        y_train,
        ys,
    )

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        Xtr,
        ytr,
    )

    pred_z = model.predict(
        Xva
    )

    pred = inverse_y(
        pred_z,
        ys,
    )

    params = int(
        np.asarray(
            model.coef_
        ).size
        + np.asarray(
            model.intercept_
        ).size
    )

    return (
        pred.astype(
            np.float32
        ),
        params,
    )


def fit_predict_polar_ridge(
    X_train_polar,
    y_train_uv,
    X_val_polar,
):
    """
    Separate speed and direction Ridge heads.

    Speed:
        standardized WS target

    Direction:
        raw unit-circle sin/cos targets
        post-prediction unit normalization
    """
    xs = fit_x_scaler(
        X_train_polar
    )

    Xtr = transform_X(
        X_train_polar,
        xs,
    ).reshape(
        len(
            X_train_polar
        ),
        -1,
    )

    Xva = transform_X(
        X_val_polar,
        xs,
    ).reshape(
        len(
            X_val_polar
        ),
        -1,
    )

    ws_train, dir_train = target_polar(
        y_train_uv
    )

    ws_scaler = fit_y_scaler(
        ws_train
    )

    ws_train_z = transform_y(
        ws_train,
        ws_scaler,
    )

    speed_model = Ridge(
        alpha=RIDGE_ALPHA
    )

    direction_model = Ridge(
        alpha=RIDGE_ALPHA
    )

    speed_model.fit(
        Xtr,
        ws_train_z,
    )

    direction_model.fit(
        Xtr,
        dir_train,
    )

    ws_pred_z = np.asarray(
        speed_model.predict(
            Xva
        ),
        dtype=np.float32,
    ).reshape(
        -1,
        1,
    )

    ws_pred_raw = inverse_y(
        ws_pred_z,
        ws_scaler,
    ).reshape(
        -1
    )

    negative_speed_fraction = float(
        np.mean(
            ws_pred_raw
            < 0.0
        )
    )

    ws_pred = np.maximum(
        ws_pred_raw,
        0.0,
    )

    dir_raw = np.asarray(
        direction_model.predict(
            Xva
        ),
        dtype=np.float64,
    )

    dir_norm = np.linalg.norm(
        dir_raw,
        axis=1,
    )

    fallback = (
        dir_norm
        < 1e-8
    )

    safe_norm = np.maximum(
        dir_norm,
        1e-8,
    )

    direction = (
        dir_raw
        / safe_norm[
            :,
            None,
        ]
    )

    if np.any(
        fallback
    ):
        # Causal fallback: last observed polar direction.
        last_s = np.asarray(
            X_val_polar[
                :,
                -1,
                1
            ],
            dtype=np.float64,
        )

        last_c = np.asarray(
            X_val_polar[
                :,
                -1,
                2
            ],
            dtype=np.float64,
        )

        fallback_norm = np.maximum(
            np.sqrt(
                last_s ** 2
                + last_c ** 2
            ),
            1e-8,
        )

        direction[
            fallback,
            0,
        ] = (
            last_s[
                fallback
            ]
            / fallback_norm[
                fallback
            ]
        )

        direction[
            fallback,
            1,
        ] = (
            last_c[
                fallback
            ]
            / fallback_norm[
                fallback
            ]
        )

    sin_wd = direction[
        :,
        0
    ]

    cos_wd = direction[
        :,
        1
    ]

    U = (
        -ws_pred
        * sin_wd
    )

    V = (
        -ws_pred
        * cos_wd
    )

    pred_uv = np.stack(
        [
            U,
            V,
        ],
        axis=1,
    ).astype(
        np.float32
    )

    params = int(
        np.asarray(
            speed_model.coef_
        ).size
        + np.asarray(
            speed_model.intercept_
        ).size
        + np.asarray(
            direction_model.coef_
        ).size
        + np.asarray(
            direction_model.intercept_
        ).size
    )

    diagnostics = {
        "polar_negative_speed_fraction_preclip": (
            negative_speed_fraction
        ),
        "polar_direction_norm_fallback_fraction": float(
            np.mean(
                fallback
            )
        ),
        "polar_direction_raw_norm_mean": float(
            np.mean(
                dir_norm
            )
        ),
    }

    return (
        pred_uv,
        params,
        diagnostics,
    )


def evaluate_wind(
    y_true,
    y_pred,
):
    yt = np.asarray(
        y_true,
        dtype=np.float64,
    )

    yp = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    err = (
        yp
        - yt
    )

    u_rmse = float(
        np.sqrt(
            np.mean(
                err[
                    :,
                    0
                ] ** 2
            )
        )
    )

    v_rmse = float(
        np.sqrt(
            np.mean(
                err[
                    :,
                    1
                ] ** 2
            )
        )
    )

    vector_rmse = float(
        np.sqrt(
            np.mean(
                np.sum(
                    err ** 2,
                    axis=1,
                )
            )
        )
    )

    ws_true = np.linalg.norm(
        yt,
        axis=1,
    )

    ws_pred = np.linalg.norm(
        yp,
        axis=1,
    )

    ws_rmse = float(
        np.sqrt(
            np.mean(
                (
                    ws_pred
                    - ws_true
                ) ** 2
            )
        )
    )

    wd_true = meteorological_wd_from_uv(
        yt[
            :,
            0
        ],
        yt[
            :,
            1
        ],
    )

    wd_pred = meteorological_wd_from_uv(
        yp[
            :,
            0
        ],
        yp[
            :,
            1
        ],
    )

    wd_err = circular_diff_deg(
        wd_pred,
        wd_true,
    )

    return {
        "wind_U_RMSE_mps": (
            u_rmse
        ),
        "wind_V_RMSE_mps": (
            v_rmse
        ),
        "wind_vector_RMSE_mps": (
            vector_rmse
        ),
        "wind_speed_RMSE_mps": (
            ws_rmse
        ),
        "wind_direction_MAE_deg": float(
            np.mean(
                np.abs(
                    wd_err
                )
            )
        ),
        "wind_direction_RMSE_deg": float(
            np.sqrt(
                np.mean(
                    wd_err ** 2
                )
            )
        ),
    }


def score_vs_reference(
    metrics,
    reference,
):
    ratios = {
        "U_ratio_vs_Met4": (
            metrics[
                "wind_U_RMSE_mps"
            ]
            / max(
                reference[
                    "wind_U_RMSE_mps"
                ],
                EPS,
            )
        ),
        "V_ratio_vs_Met4": (
            metrics[
                "wind_V_RMSE_mps"
            ]
            / max(
                reference[
                    "wind_V_RMSE_mps"
                ],
                EPS,
            )
        ),
        "WS_ratio_vs_Met4": (
            metrics[
                "wind_speed_RMSE_mps"
            ]
            / max(
                reference[
                    "wind_speed_RMSE_mps"
                ],
                EPS,
            )
        ),
        "WD_ratio_vs_Met4": (
            metrics[
                "wind_direction_RMSE_deg"
            ]
            / max(
                reference[
                    "wind_direction_RMSE_deg"
                ],
                EPS,
            )
        ),
    }

    score = (
        SCORE_WEIGHTS[
            "U"
        ]
        * ratios[
            "U_ratio_vs_Met4"
        ]
        + SCORE_WEIGHTS[
            "V"
        ]
        * ratios[
            "V_ratio_vs_Met4"
        ]
        + SCORE_WEIGHTS[
            "WS"
        ]
        * ratios[
            "WS_ratio_vs_Met4"
        ]
        + SCORE_WEIGHTS[
            "WD"
        ]
        * ratios[
            "WD_ratio_vs_Met4"
        ]
    )

    return (
        float(
            score
        ),
        ratios,
    )


# =============================================================================
# Development-only representation benchmark
# =============================================================================

def run_representation_lomo(
    *,
    dataset_dir: Path,
    output_dir: Path,
):
    missions = {
        m: load_track_j(
            dataset_dir,
            m,
        )
        for m in DEVELOPMENT_MISSIONS
    }

    rows = []

    log("")
    log("=" * 132)
    log(
        "12R PART B — 6 x 10-MIN -> +10-MIN REPRESENTATION LOMO"
    )
    log("=" * 132)
    log(
        "[FIREWALL] Tropical Atlantic is not loaded."
    )
    log("")

    for held_out in DEVELOPMENT_MISSIONS:
        train_names = [
            m
            for m in DEVELOPMENT_MISSIONS
            if m != held_out
        ]

        train = concat_missions(
            [
                missions[
                    m
                ]
                for m in train_names
            ]
        )

        val = missions[
            held_out
        ]

        Xtr_full = train[
            "X_full9"
        ]

        Xva_full = val[
            "X_full9"
        ]

        ytr = train[
            "y_uv"
        ]

        yva = val[
            "y_uv"
        ]

        candidate_predictions = {}
        candidate_params = {}
        candidate_extra = {}

        # A) Full 9-feature Cartesian.
        pred, params = (
            fit_predict_cartesian_ridge(
                Xtr_full,
                ytr,
                Xva_full,
            )
        )

        candidate_predictions[
            "Full9-UV-Ridge"
        ] = pred

        candidate_params[
            "Full9-UV-Ridge"
        ] = params

        candidate_extra[
            "Full9-UV-Ridge"
        ] = {}

        # B) Meteorological 4-feature Cartesian.
        Xtr_met4 = make_met4(
            Xtr_full
        )
        Xva_met4 = make_met4(
            Xva_full
        )

        pred, params = (
            fit_predict_cartesian_ridge(
                Xtr_met4,
                ytr,
                Xva_met4,
            )
        )

        candidate_predictions[
            "Met4-UV-Ridge"
        ] = pred

        candidate_params[
            "Met4-UV-Ridge"
        ] = params

        candidate_extra[
            "Met4-UV-Ridge"
        ] = {}

        # C) Wind-only Cartesian.
        Xtr_wind2 = make_wind2(
            Xtr_full
        )
        Xva_wind2 = make_wind2(
            Xva_full
        )

        pred, params = (
            fit_predict_cartesian_ridge(
                Xtr_wind2,
                ytr,
                Xva_wind2,
            )
        )

        candidate_predictions[
            "Wind2-UV-Ridge"
        ] = pred

        candidate_params[
            "Wind2-UV-Ridge"
        ] = params

        candidate_extra[
            "Wind2-UV-Ridge"
        ] = {}

        # D) Polar meteorological representation.
        Xtr_polar = make_polar5(
            Xtr_full
        )

        Xva_polar = make_polar5(
            Xva_full
        )

        (
            pred,
            params,
            extra,
        ) = fit_predict_polar_ridge(
            Xtr_polar,
            ytr,
            Xva_polar,
        )

        candidate_predictions[
            "Polar5-Ridge"
        ] = pred

        candidate_params[
            "Polar5-Ridge"
        ] = params

        candidate_extra[
            "Polar5-Ridge"
        ] = extra

        metrics_by_candidate = {
            candidate: evaluate_wind(
                yva,
                prediction,
            )
            for candidate, prediction
            in candidate_predictions.items()
        }

        met4_metrics = (
            metrics_by_candidate[
                "Met4-UV-Ridge"
            ]
        )

        log("-" * 132)
        log(
            f"[{held_out}] "
            f"Ntrain={len(Xtr_full):,} | "
            f"Nval={len(Xva_full):,}"
        )

        for candidate in [
            "Full9-UV-Ridge",
            "Met4-UV-Ridge",
            "Wind2-UV-Ridge",
            "Polar5-Ridge",
        ]:
            metrics = (
                metrics_by_candidate[
                    candidate
                ]
            )

            score, ratios = (
                score_vs_reference(
                    metrics,
                    met4_metrics,
                )
            )

            row = {
                "candidate": candidate,
                "held_out_mission": held_out,
                "n_train": int(
                    len(
                        Xtr_full
                    )
                ),
                "n_val": int(
                    len(
                        Xva_full
                    )
                ),
                "lookback_steps": (
                    LOOKBACK_STEPS
                ),
                "step_minutes": (
                    STEP_MINUTES
                ),
                "forecast_minutes": (
                    FORECAST_MINUTES
                ),
                "ridge_alpha": (
                    RIDGE_ALPHA
                ),
                "parameter_count": int(
                    candidate_params[
                        candidate
                    ]
                ),
                "selection_score_vs_Met4": (
                    score
                ),
                **metrics,
                **ratios,
                **candidate_extra[
                    candidate
                ],
            }

            rows.append(
                row
            )

            log(
                f"  {candidate:20s} | "
                f"score={score:.5f} | "
                f"U={metrics['wind_U_RMSE_mps']:.4f} | "
                f"V={metrics['wind_V_RMSE_mps']:.4f} | "
                f"WS={metrics['wind_speed_RMSE_mps']:.4f} | "
                f"WD={metrics['wind_direction_RMSE_deg']:.3f}"
            )

    fold_df = pd.DataFrame(
        rows
    )

    fold_path = (
        output_dir
        / "representation_fold_results.csv"
    )

    fold_df.to_csv(
        fold_path,
        index=False,
        encoding="utf-8-sig",
    )

    return (
        fold_df,
        fold_path,
    )


def summarize_representation(
    fold_df,
    output_dir,
):
    candidates = [
        "Full9-UV-Ridge",
        "Met4-UV-Ridge",
        "Wind2-UV-Ridge",
        "Polar5-Ridge",
    ]

    rows = []

    met4 = fold_df.loc[
        fold_df[
            "candidate"
        ]
        == "Met4-UV-Ridge"
    ].copy()

    met4_means = {
        k: float(
            met4[
                k
            ].mean()
        )
        for k in [
            "wind_U_RMSE_mps",
            "wind_V_RMSE_mps",
            "wind_vector_RMSE_mps",
            "wind_speed_RMSE_mps",
            "wind_direction_RMSE_deg",
        ]
    }

    full9 = fold_df.loc[
        fold_df[
            "candidate"
        ]
        == "Full9-UV-Ridge"
    ].copy()

    full9_means = {
        k: float(
            full9[
                k
            ].mean()
        )
        for k in met4_means
    }

    for candidate in candidates:
        sub = fold_df.loc[
            fold_df[
                "candidate"
            ]
            == candidate
        ].copy()

        if len(
            sub
        ) != 3:
            raise RuntimeError(
                f"{candidate}: expected 3 fold rows, found {len(sub)}"
            )

        row = {
            "candidate": candidate,
            "folds": int(
                len(
                    sub
                )
            ),
            "selection_score_vs_Met4_mean": float(
                sub[
                    "selection_score_vs_Met4"
                ].mean()
            ),
            "selection_score_vs_Met4_std": float(
                sub[
                    "selection_score_vs_Met4"
                ].std(
                    ddof=0
                )
            ),
            "parameter_count": int(
                round(
                    sub[
                        "parameter_count"
                    ].mean()
                )
            ),
            "mission_wins_vs_Met4": int(
                np.sum(
                    sub[
                        "selection_score_vs_Met4"
                    ].to_numpy(
                        dtype=float
                    )
                    < 1.0
                )
            ),
        }

        for metric in met4_means:
            mean = float(
                sub[
                    metric
                ].mean()
            )

            std = float(
                sub[
                    metric
                ].std(
                    ddof=0
                )
            )

            row[
                f"{metric}_mean"
            ] = mean

            row[
                f"{metric}_std"
            ] = std

            row[
                f"{metric}_improvement_vs_Met4_fraction"
            ] = (
                met4_means[
                    metric
                ]
                - mean
            ) / max(
                met4_means[
                    metric
                ],
                EPS,
            )

            row[
                f"{metric}_improvement_vs_Full9_fraction"
            ] = (
                full9_means[
                    metric
                ]
                - mean
            ) / max(
                full9_means[
                    metric
                ],
                EPS,
            )

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    ).sort_values(
        [
            "selection_score_vs_Met4_mean",
            "parameter_count",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    summary_path = (
        output_dir
        / "representation_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    polar = summary.loc[
        summary[
            "candidate"
        ]
        == "Polar5-Ridge"
    ].iloc[
        0
    ]

    polar_score = float(
        polar[
            "selection_score_vs_Met4_mean"
        ]
    )

    polar_wins = int(
        polar[
            "mission_wins_vs_Met4"
        ]
    )

    polar_vector_imp = float(
        polar[
            "wind_vector_RMSE_mps_improvement_vs_Met4_fraction"
        ]
    )

    polar_v_imp = float(
        polar[
            "wind_V_RMSE_mps_improvement_vs_Met4_fraction"
        ]
    )

    polar_wd_imp = float(
        polar[
            "wind_direction_RMSE_deg_improvement_vs_Met4_fraction"
        ]
    )

    vector_worsening = (
        -polar_vector_imp
    )

    polar_promising = bool(
        polar_score
        < POLAR_SCORE_THRESHOLD
        and polar_wins
        >= POLAR_MIN_MISSION_WINS
        and vector_worsening
        <= POLAR_MAX_VECTOR_WORSENING
        and (
            polar_v_imp
            > POLAR_MIN_V_OR_WD_IMPROVEMENT
            or polar_wd_imp
            > POLAR_MIN_V_OR_WD_IMPROVEMENT
        )
    )

    met4_summary = summary.loc[
        summary[
            "candidate"
        ]
        == "Met4-UV-Ridge"
    ].iloc[
        0
    ]

    full9_summary = summary.loc[
        summary[
            "candidate"
        ]
        == "Full9-UV-Ridge"
    ].iloc[
        0
    ]

    met4_score_vs_full9 = (
        0.25
        * (
            met4_summary[
                "wind_U_RMSE_mps_mean"
            ]
            / full9_summary[
                "wind_U_RMSE_mps_mean"
            ]
        )
        + 0.35
        * (
            met4_summary[
                "wind_V_RMSE_mps_mean"
            ]
            / full9_summary[
                "wind_V_RMSE_mps_mean"
            ]
        )
        + 0.25
        * (
            met4_summary[
                "wind_speed_RMSE_mps_mean"
            ]
            / full9_summary[
                "wind_speed_RMSE_mps_mean"
            ]
        )
        + 0.15
        * (
            met4_summary[
                "wind_direction_RMSE_deg_mean"
            ]
            / full9_summary[
                "wind_direction_RMSE_deg_mean"
            ]
        )
    )

    decision = (
        "POLAR_PROMISING_NEXT_KALMAN_CIRCULAR_MODEL"
        if polar_promising
        else "POLAR_NOT_YET_SUPERIOR_KEEP_CARTESIAN_AS_REFERENCE"
    )

    payload = {
        "stage": "12R",
        "script_version": SCRIPT_VERSION,
        "task": {
            "resolution_minutes": STEP_MINUTES,
            "lookback_steps": LOOKBACK_STEPS,
            "lookback_minutes": LOOKBACK_STEPS * STEP_MINUTES,
            "forecast_minutes": FORECAST_MINUTES,
        },
        "development_only": True,
        "development_missions": DEVELOPMENT_MISSIONS,
        "Tropical_Atlantic_used": False,
        "score_weights": SCORE_WEIGHTS,
        "feature_ablation": {
            "Met4_score_vs_Full9_mean_metric_ratio": float(
                met4_score_vs_full9
            ),
            "interpretation": (
                "below 1 means the meteorological-only U,V,T,RH representation "
                "is better on average than the old 9-feature input"
            ),
        },
        "polar_rule": {
            "score_lt": POLAR_SCORE_THRESHOLD,
            "mission_wins_ge": POLAR_MIN_MISSION_WINS,
            "vector_worsening_le": POLAR_MAX_VECTOR_WORSENING,
            "V_or_WD_improvement_gt": (
                POLAR_MIN_V_OR_WD_IMPROVEMENT
            ),
        },
        "polar_score_mean": polar_score,
        "polar_mission_wins": polar_wins,
        "polar_vector_improvement_fraction": (
            polar_vector_imp
        ),
        "polar_V_improvement_fraction": (
            polar_v_imp
        ),
        "polar_WD_improvement_fraction": (
            polar_wd_imp
        ),
        "polar_promising": polar_promising,
        "decision": decision,
        "published_BP_STGNN_reference_context_only": (
            BP_STGNN_PUBLISHED_REFERENCE
        ),
        "scientific_note": (
            "Polar benchmark uses WS/sinWD/cosWD derived exactly from U/V so "
            "that it is a representation-only comparison. Raw WS/WD fields "
            "are audited separately and are not added as new sensor inputs."
        ),
    }

    decision_path = (
        output_dir
        / "selected_representation.json"
    )

    save_json(
        decision_path,
        payload,
    )

    return (
        summary,
        summary_path,
        payload,
        decision_path,
    )


# =============================================================================
# Report
# =============================================================================

def write_report(
    *,
    output_dir,
    dataset_dir,
    raw_dir,
    quant_df,
    consistency_df,
    summary,
    decision,
):
    report_path = (
        output_dir
        / "12R_REPORT.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "12R Wind Representation Audit + Nie 6x10-min -> +10-min Ridge Benchmark\n"
        )
        f.write(
            "=" * 132
            + "\n\n"
        )

        f.write(
            "FROZEN TASK\n"
        )
        f.write(
            "-" * 132
            + "\n"
        )
        f.write(
            "resolution: 10 min\n"
        )
        f.write(
            "history: 6 points = 60 min\n"
        )
        f.write(
            "target: next +10-min block\n"
        )
        f.write(
            f"dataset: {dataset_dir}\n"
        )
        f.write(
            f"raw source: {raw_dir}\n"
        )
        f.write(
            "development missions: Antarctic / Atlantic / West Coast\n"
        )
        f.write(
            "Tropical Atlantic accessed: NO\n\n"
        )

        f.write(
            "RAW EMPIRICAL QUANTIZATION\n"
        )
        f.write(
            "-" * 132
            + "\n"
        )
        f.write(
            quant_df.to_string(
                index=False
            )
        )
        f.write(
            "\n\n"
        )

        f.write(
            "RAW U/V vs WS/WD CONSISTENCY\n"
        )
        f.write(
            "-" * 132
            + "\n"
        )
        f.write(
            consistency_df.to_string(
                index=False
            )
        )
        f.write(
            "\n\n"
        )

        f.write(
            "REPRESENTATION SUMMARY\n"
        )
        f.write(
            "-" * 132
            + "\n"
        )
        f.write(
            summary.to_string(
                index=False
            )
        )
        f.write(
            "\n\n"
        )

        f.write(
            "DECISION\n"
        )
        f.write(
            "-" * 132
            + "\n"
        )
        f.write(
            f"polar promising: {decision['polar_promising']}\n"
        )
        f.write(
            f"decision: {decision['decision']}\n"
        )
        f.write(
            f"Polar mean score vs Met4: {decision['polar_score_mean']:.6f}\n"
        )
        f.write(
            f"Polar mission wins: {decision['polar_mission_wins']}/3\n"
        )
        f.write(
            f"Polar V improvement: "
            f"{100.0*decision['polar_V_improvement_fraction']:+.3f}%\n"
        )
        f.write(
            f"Polar WD improvement: "
            f"{100.0*decision['polar_WD_improvement_fraction']:+.3f}%\n"
        )
        f.write(
            f"Polar vector improvement: "
            f"{100.0*decision['polar_vector_improvement_fraction']:+.3f}%\n"
        )
        f.write(
            "\nPublished BP-STGNN values are context only and are not used "
            "for development selection.\n"
        )

    return report_path


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
        default=DEFAULT_RAW_DIR,
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Dataset root containing track_J_joint_compatible. "
            "If omitted, common Stage-12G/12B paths are searched."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--skip-raw-audit",
        action="store_true",
        help=(
            "Skip NetCDF precision/consistency audit and run only "
            "the representation benchmark."
        ),
    )

    args = parser.parse_args()

    project_root = args.project_root
    raw_dir = args.raw_dir
    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_dir = resolve_dataset_dir(
        args.dataset_dir
    )

    stage12b, stage12b_path = (
        load_stage12b(
            project_root
        )
    )

    log("=" * 132)
    log(
        "12R — WIND REPRESENTATION AUDIT"
    )
    log("=" * 132)
    log(
        f"script version : {SCRIPT_VERSION}"
    )
    log(
        f"Stage-12B      : {stage12b_path}"
    )
    log(
        f"raw dir        : {raw_dir}"
    )
    log(
        f"dataset dir    : {dataset_dir}"
    )
    log(
        f"output dir     : {output_dir}"
    )
    log(
        "forecast task  : 6 x 10-min history -> next +10-min wind"
    )
    log(
        "[FIREWALL] Tropical Atlantic will NOT be loaded."
    )
    log("")

    if args.skip_raw_audit:
        metadata_df = pd.DataFrame()
        quant_df = pd.DataFrame()
        consistency_df = pd.DataFrame()

        metadata_path = None
        quant_path = None
        consistency_path = None

        log(
            "[RAW AUDIT] skipped by command-line flag."
        )
    else:
        (
            metadata_df,
            quant_df,
            consistency_df,
            metadata_path,
            quant_path,
            consistency_path,
        ) = run_raw_audit(
            stage12b=stage12b,
            raw_dir=raw_dir,
            output_dir=output_dir,
        )

    fold_df, fold_path = (
        run_representation_lomo(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
        )
    )

    (
        summary,
        summary_path,
        decision,
        decision_path,
    ) = summarize_representation(
        fold_df,
        output_dir,
    )

    report_path = write_report(
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        quant_df=quant_df,
        consistency_df=consistency_df,
        summary=summary,
        decision=decision,
    )

    log("")
    log("=" * 132)
    log(
        "12R REPRESENTATION SUMMARY"
    )
    log("=" * 132)
    log(
        summary.to_string(
            index=False
        )
    )

    if not consistency_df.empty:
        log("")
        log(
            "12R RAW U/V vs WS/WD CONSISTENCY"
        )
        log(
            "=" * 132
        )

        cols = [
            "mission",
            "WS_minus_hypotUV_MAE_mps",
            "WS_minus_hypotUV_RMSE_mps",
            "WD_raw_vs_UV_FROM_MAE_deg",
            "WD_raw_vs_UV_FROM_RMSE_deg",
            "inferred_raw_direction_convention",
        ]

        log(
            consistency_df[
                cols
            ].to_string(
                index=False
            )
        )

    if not quant_df.empty:
        log("")
        log(
            "12R RAW EMPIRICAL QUANTIZATION"
        )
        log(
            "=" * 132
        )

        log(
            quant_df[
                [
                    "mission",
                    "canonical",
                    "empirical_quantization_step",
                    "empirical_quantization_decimals",
                    "empirical_quantization_alignment_fraction",
                ]
            ].to_string(
                index=False
            )
        )

    log("")
    log(
        f"[POLAR PROMISING] "
        f"{decision['polar_promising']}"
    )
    log(
        f"[DECISION] "
        f"{decision['decision']}"
    )
    log(
        "Polar vs Met4: "
        f"score={decision['polar_score_mean']:.6f} | "
        f"V={100.0*decision['polar_V_improvement_fraction']:+.3f}% | "
        f"WD={100.0*decision['polar_WD_improvement_fraction']:+.3f}% | "
        f"vector={100.0*decision['polar_vector_improvement_fraction']:+.3f}% | "
        f"wins={decision['polar_mission_wins']}/3"
    )
    log("")
    log(
        "Published BP-STGNN reference (context only): "
        "U=0.74864 | V=0.70506 | WS=0.69940 | WD=5.584 deg"
    )
    log("")
    log(
        f"[SAVED] {fold_path}"
    )
    log(
        f"[SAVED] {summary_path}"
    )
    log(
        f"[SAVED] {decision_path}"
    )

    if metadata_path is not None:
        log(
            f"[SAVED] {metadata_path}"
        )
        log(
            f"[SAVED] {quant_path}"
        )
        log(
            f"[SAVED] {consistency_path}"
        )

    log(
        f"[SAVED] {report_path}"
    )

    if decision[
        "polar_promising"
    ]:
        log("")
        log(
            "NEXT RECOMMENDATION: build the dedicated dual-branch model: "
            "Kalman/local-trend wind-speed anchor + circular sin/cos "
            "direction model, with U/V reconstruction consistency."
        )
    else:
        log("")
        log(
            "NEXT RECOMMENDATION: do not assume polar coordinates alone solve "
            "the problem. Use the 12R diagnostics to choose the next mechanism."
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
