# -*- coding: utf-8 -*-
r"""
12P_A_screen_1min_lookback_10min_horizon_Ridge_LOMO.py

Stage 12P-A
===========
Development-only information-screening experiment for the Nie/Saildrone
same-mission data.

Question
--------
Does a longer ONE-MINUTE historical context improve prediction of the wind
exactly +10 minutes after the end of the context?

Frozen task
-----------
Raw resolution:
    1 minute

Input features at each historical minute:
    [U, V, T, RH, SOG,
     COG_sin, COG_cos,
     WING_ANGLE_sin, WING_ANGLE_cos]

Candidate lookbacks:
    6 min
    15 min
    30 min
    60 min

For a context ending at time t:
    input = x[t-L+1], ..., x[t]
    target = wind at exactly t + 10 min

Thus, for L=60:
    [x(t-59), ..., x(t)] -> W(t+10)

IMPORTANT COMPARABILITY NOTE
----------------------------
Nie et al. / BP-STGNN first resample the original 1-min data to a 10-min
modeling interval. Therefore this 12P-A experiment has the SAME 10-min
wall-clock forecast horizon, but it is NOT preprocessing-identical to the
published BP-STGNN protocol.

This stage is an ENHANCED-CONTEXT experiment intended to test whether the
previous 6x9 input representation was information-limited.

Fair lookback comparison
------------------------
All lookbacks use EXACTLY THE SAME samples.

We first build a common dataset requiring:
    - 60 consecutive one-minute context rows ending at t;
    - all 9 input features finite for all 60 context rows;
    - a finite target row at exactly t+10 min.

Then each lookback is a suffix of the SAME 60-min context:
    L=6  -> last 6 rows
    L=15 -> last 15 rows
    L=30 -> last 30 rows
    L=60 -> all 60 rows

Therefore target timestamps do not change with lookback length.

Development-only LOMO
---------------------
Development missions:
    Antarctic
    Atlantic
    West Coast

Folds:
    holdout Antarctic  <- train Atlantic + West Coast
    holdout Atlantic   <- train Antarctic + West Coast
    holdout West Coast <- train Antarctic + Atlantic

Tropical Atlantic is NEVER loaded by this script.

Model
-----
Only Ridge(alpha=1.0) is used in 12P-A.

Reason:
    isolate the information value of longer context before adding engineered
    temporal features or nonlinear models.

Fold-train-only standardization:
    X: feature-wise mean/std over all training samples and historical steps
    y: target U/V mean/std over training samples

Selection score
---------------
For each held-out mission, L=6 Ridge is the within-fold reference:

    Score(L) =
        0.25 * U_RMSE(L)  / U_RMSE(L6)
      + 0.35 * V_RMSE(L)  / V_RMSE(L6)
      + 0.25 * WS_RMSE(L) / WS_RMSE(L6)
      + 0.15 * WD_RMSE(L) / WD_RMSE(L6)

L=6 therefore has score exactly 1.0 in every fold.

Predeclared complexity rule
---------------------------
A longer lookback replaces L=6 only if:
    - it has the lowest mean LOMO score;
    - its mean score improves by at least 0.5% versus L=6;
    - it beats L=6 in at least 2 of 3 held-out missions.

Otherwise retain L=6.

This prevents selecting a much larger input solely for numerical noise.

Scientific firewall
-------------------
This script uses Stage-12B only as the frozen RAW MISSION DISCOVERY / VARIABLE
LOADER. It loads ONLY:
    Antarctic
    Atlantic
    West Coast

It does NOT load:
    Tropical Atlantic
    Stage-12K outputs
    Stage-12K metrics
    Stage-12G Tropical test data

Outputs
-------
<output-dir>/
    dataset_1min_common60/
        Antarctic.npz
        Atlantic.npz
        West_Coast.npz
        dataset_build_audit.csv
        schema.json
    ridge_lomo_fold_results.csv
    lookback_summary.csv
    selected_lookback.json
    12P_A_REPORT.txt

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\12P_A_screen_1min_lookback_10min_horizon_Ridge_LOMO.py" --raw-dir "D:\project\WindPredict_SaildroneData\data\raw\RawData" --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12P_A_1minLookback_10minHorizon_Ridge_LOMO_v0_1"

Use --rebuild-dataset only when the raw-source dataset genuinely changes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import Ridge
except Exception as exc:
    raise RuntimeError(
        "scikit-learn is required. Activate the WindPredict environment."
    ) from exc


SCRIPT_VERSION = "0.1.0-12P-A-1min-lookback-screen-10min-horizon"

DEFAULT_PROJECT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")
DEFAULT_RAW_DIR = DEFAULT_PROJECT_ROOT / "data" / "raw" / "RawData"
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12P_A_1minLookback_10minHorizon_Ridge_LOMO_v0_1"
)

DEVELOPMENT_MISSIONS = ["Antarctic", "Atlantic", "West Coast"]

ONE_MIN_NS = 60 * 1_000_000_000
TEN_MIN_NS = 10 * ONE_MIN_NS

MAX_LOOKBACK_MIN = 60
LOOKBACKS_MIN = [6, 15, 30, 60]
FORECAST_HORIZON_MIN = 10

RIDGE_ALPHA = 1.0
EPS = 1e-12

FEATURE_NAMES = [
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
TARGET_NAMES = ["U", "V", "VESSEL_E", "VESSEL_N"]

RAW_REQUIRED_CONTEXT = [
    "U",
    "V",
    "T",
    "RH",
    "SOG",
    "COG",
    "WING_ANGLE",
]
RAW_REQUIRED_TARGET = [
    "U",
    "V",
    "SOG",
    "COG",
]

SCORE_WEIGHTS = {
    "U": 0.25,
    "V": 0.35,
    "WS": 0.25,
    "WD": 0.15,
}

MIN_LONGER_LOOKBACK_SCORE_IMPROVEMENT = 0.005  # 0.5%
MIN_MISSION_WINS = 2

# Published literature values are CONTEXT ONLY. Never used for selection.
BP_STGNN_PUBLISHED_REFERENCE = {
    "U_RMSE_mps": 0.74864,
    "V_RMSE_mps": 0.70506,
    "WS_RMSE_mps": 0.69940,
    "WD_RMSE_deg": 5.58400,
}


def log(msg=""):
    print(msg, flush=True)


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


def load_stage12b(project_root: Path):
    path = project_root / "src" / "12B_build_Nie_same_mission_benchmark_dataset.py"

    if not path.exists():
        raise FileNotFoundError(
            "Stage-12B raw loader not found:\n"
            f"  {path}"
        )

    name = "stage12b_raw_loader_for_12pa"

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot create import spec for: {path}"
        )

    mod = importlib.util.module_from_spec(spec)

    # Robust for Python 3.11 introspection/dataclass behavior.
    sys.modules[name] = mod

    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise

    required = [
        "import_xarray",
        "discover_mission_files",
        "load_mission_raw",
    ]
    missing = [
        x for x in required
        if not hasattr(mod, x)
    ]
    if missing:
        raise RuntimeError(
            f"Stage-12B raw loader missing functions: {missing}"
        )

    return mod, path


def ns_iso(v: int):
    return str(np.datetime64(int(v), "ns"))


def rolling_contiguous_run_length(time_ns: np.ndarray):
    """
    run[i] = number of consecutive 1-min rows ending at i.
    """
    t = np.asarray(time_ns, dtype=np.int64)

    if len(t) == 0:
        return np.empty(0, dtype=np.int32)

    run = np.ones(len(t), dtype=np.int32)

    for i in range(1, len(t)):
        if t[i] - t[i - 1] == ONE_MIN_NS:
            run[i] = run[i - 1] + 1
        else:
            run[i] = 1

    return run


def rolling_bad_count_ending_at(
    bad: np.ndarray,
    window: int,
):
    """
    Number of True values in [i-window+1, i] for each i.
    """
    bad = np.asarray(bad, dtype=np.int64)
    c = np.concatenate(
        [
            np.zeros(1, dtype=np.int64),
            np.cumsum(bad, dtype=np.int64),
        ]
    )

    out = np.full(
        len(bad),
        -1,
        dtype=np.int64,
    )

    if len(bad) >= window:
        ends = np.arange(
            window - 1,
            len(bad),
            dtype=np.int64,
        )
        starts = ends - window + 1

        out[ends] = (
            c[ends + 1]
            - c[starts]
        )

    return out


def build_feature_matrix(raw: pd.DataFrame):
    U = raw["U"].to_numpy(dtype=np.float64)
    V = raw["V"].to_numpy(dtype=np.float64)
    T = raw["T"].to_numpy(dtype=np.float64)
    RH = raw["RH"].to_numpy(dtype=np.float64)
    SOG = raw["SOG"].to_numpy(dtype=np.float64)
    COG = raw["COG"].to_numpy(dtype=np.float64)
    WING = raw["WING_ANGLE"].to_numpy(dtype=np.float64)

    cog_rad = np.deg2rad(COG)
    wing_rad = np.deg2rad(WING)

    X = np.stack(
        [
            U,
            V,
            T,
            RH,
            SOG,
            np.sin(cog_rad),
            np.cos(cog_rad),
            np.sin(wing_rad),
            np.cos(wing_rad),
        ],
        axis=1,
    )

    return X.astype(np.float32)


def build_target_matrix(raw: pd.DataFrame):
    U = raw["U"].to_numpy(dtype=np.float64)
    V = raw["V"].to_numpy(dtype=np.float64)
    SOG = raw["SOG"].to_numpy(dtype=np.float64)
    COG = raw["COG"].to_numpy(dtype=np.float64)

    cog_rad = np.deg2rad(COG)

    vessel_e = SOG * np.sin(cog_rad)
    vessel_n = SOG * np.cos(cog_rad)

    y = np.stack(
        [
            U,
            V,
            vessel_e,
            vessel_n,
        ],
        axis=1,
    )

    return y.astype(np.float32)


def build_common60_samples(
    raw: pd.DataFrame,
    mission: str,
):
    """
    Build the strict common sample set used by ALL lookbacks.

    A sample at context end index i is accepted iff:
      1. rows i-59 ... i are exactly consecutive at 1-min spacing;
      2. all 9 model features in those 60 rows are finite;
      3. there is a raw row at exactly time[i] + 10 min;
      4. target U,V,SOG,COG are finite.

    No interpolation.
    No requirement is imposed on future rows t+1 ... t+9; only the exact
    t+10 target must exist and be finite.
    """
    raw = (
        raw.sort_values(
            "time_ns",
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )

    if raw["time_ns"].duplicated().any():
        raise RuntimeError(
            f"{mission}: duplicate timestamps remain after Stage-12B raw load."
        )

    time_ns = raw["time_ns"].to_numpy(dtype=np.int64)
    X_all = build_feature_matrix(raw)
    y_all = build_target_matrix(raw)

    feature_row_bad = ~np.isfinite(X_all).all(axis=1)
    target_row_bad = ~np.isfinite(y_all).all(axis=1)

    run = rolling_contiguous_run_length(time_ns)
    bad60 = rolling_bad_count_ending_at(
        feature_row_bad,
        MAX_LOOKBACK_MIN,
    )

    # Exact timestamp lookup.
    time_to_index = {
        int(t): int(i)
        for i, t in enumerate(time_ns)
    }

    accepted_end = []
    target_idx = []

    rejected_short_or_gap = 0
    rejected_context_nonfinite = 0
    rejected_missing_tplus10 = 0
    rejected_target_nonfinite = 0

    for i in range(
        MAX_LOOKBACK_MIN - 1,
        len(raw),
    ):
        if run[i] < MAX_LOOKBACK_MIN:
            rejected_short_or_gap += 1
            continue

        if bad60[i] != 0:
            rejected_context_nonfinite += 1
            continue

        t_target = int(
            time_ns[i]
            + FORECAST_HORIZON_MIN * ONE_MIN_NS
        )

        j = time_to_index.get(
            t_target,
            None,
        )
        if j is None:
            rejected_missing_tplus10 += 1
            continue

        if target_row_bad[j]:
            rejected_target_nonfinite += 1
            continue

        accepted_end.append(i)
        target_idx.append(j)

    if not accepted_end:
        raise RuntimeError(
            f"{mission}: no valid common-60 samples."
        )

    accepted_end = np.asarray(
        accepted_end,
        dtype=np.int64,
    )
    target_idx = np.asarray(
        target_idx,
        dtype=np.int64,
    )
    starts = (
        accepted_end
        - MAX_LOOKBACK_MIN
        + 1
    )

    # View all 60-row windows, then copy only accepted windows.
    window_view = np.lib.stride_tricks.sliding_window_view(
        X_all,
        window_shape=MAX_LOOKBACK_MIN,
        axis=0,
    )
    # NumPy shape: [N-59, F, 60] -> [N-59, 60, F]
    window_view = np.transpose(
        window_view,
        (0, 2, 1),
    )

    X60 = np.asarray(
        window_view[starts],
        dtype=np.float32,
    ).copy()

    y = y_all[target_idx].astype(
        np.float32,
        copy=True,
    )[:, None, :]

    apparent_ref = (
        y[:, :, 0:2]
        - y[:, :, 2:4]
    ).astype(np.float32)

    context_end_time_ns = (
        time_ns[accepted_end]
        .astype(np.int64)
    )
    target_time_ns = (
        time_ns[target_idx]
        .astype(np.int64)
    )[:, None]

    if X60.shape[1:] != (
        MAX_LOOKBACK_MIN,
        len(FEATURE_NAMES),
    ):
        raise RuntimeError(
            f"{mission}: unexpected X60 shape {X60.shape}"
        )

    if not np.all(
        target_time_ns.reshape(-1)
        - context_end_time_ns
        == FORECAST_HORIZON_MIN * ONE_MIN_NS
    ):
        raise RuntimeError(
            f"{mission}: target horizon audit failed."
        )

    if not (
        np.isfinite(X60).all()
        and np.isfinite(y).all()
        and np.isfinite(apparent_ref).all()
    ):
        raise RuntimeError(
            f"{mission}: non-finite values in built dataset."
        )

    # Exact context timestamp audit on a deterministic subset + endpoints.
    audit_indices = np.unique(
        np.concatenate(
            [
                np.asarray([0, len(accepted_end) - 1]),
                np.linspace(
                    0,
                    len(accepted_end) - 1,
                    num=min(100, len(accepted_end)),
                    dtype=np.int64,
                ),
            ]
        )
    )

    max_context_timing_error = 0

    for k in audit_indices:
        i = int(accepted_end[k])
        a = i - MAX_LOOKBACK_MIN + 1

        expected = (
            time_ns[a]
            + np.arange(
                MAX_LOOKBACK_MIN,
                dtype=np.int64,
            ) * ONE_MIN_NS
        )
        actual = time_ns[a:i + 1]

        if not np.array_equal(
            actual,
            expected,
        ):
            raise RuntimeError(
                f"{mission}: 60-min context timing audit failed."
            )

        err = int(
            np.max(
                np.abs(actual - expected)
            )
        )
        max_context_timing_error = max(
            max_context_timing_error,
            err,
        )

    result = {
        "X60": X60,
        "y_joint": y,
        "apparent_ref": apparent_ref,
        "context_end_time_ns": context_end_time_ns,
        "target_time_ns": target_time_ns,
        "mission": np.asarray(
            [mission] * len(X60)
        ),
    }

    audit = {
        "mission": mission,
        "raw_rows": int(len(raw)),
        "accepted_common60_samples": int(
            len(X60)
        ),
        "rejected_short_context_or_gap": int(
            rejected_short_or_gap
        ),
        "rejected_context_nonfinite": int(
            rejected_context_nonfinite
        ),
        "rejected_missing_exact_tplus10": int(
            rejected_missing_tplus10
        ),
        "rejected_target_nonfinite": int(
            rejected_target_nonfinite
        ),
        "X60_shape": str(
            tuple(X60.shape)
        ),
        "y_shape": str(
            tuple(y.shape)
        ),
        "context_first": ns_iso(
            int(context_end_time_ns[0])
        ),
        "context_last": ns_iso(
            int(context_end_time_ns[-1])
        ),
        "target_first": ns_iso(
            int(target_time_ns[0, 0])
        ),
        "target_last": ns_iso(
            int(target_time_ns[-1, 0])
        ),
        "forecast_horizon_minutes": int(
            FORECAST_HORIZON_MIN
        ),
        "max_context_timing_error_ns": int(
            max_context_timing_error
        ),
        "physics_max_abs_error": float(
            np.max(
                np.abs(
                    (
                        y[:, :, 0:2]
                        - y[:, :, 2:4]
                    )
                    - apparent_ref
                )
            )
        ),
    }

    return result, audit


def save_mission_npz(
    path: Path,
    samples: dict,
):
    np.savez_compressed(
        path,
        X60_raw=samples["X60"],
        y_joint_raw=samples["y_joint"],
        y_wind_raw=samples["y_joint"][:, :, 0:2],
        y_vessel_raw=samples["y_joint"][:, :, 2:4],
        apparent_ref=samples["apparent_ref"],
        context_end_time_ns=samples["context_end_time_ns"],
        target_time_ns=samples["target_time_ns"],
        mission=samples["mission"],
        feature_names=np.asarray(FEATURE_NAMES),
        target_names=np.asarray(TARGET_NAMES),
        max_lookback_minutes=np.asarray(
            [MAX_LOOKBACK_MIN],
            dtype=np.int64,
        ),
        raw_step_minutes=np.asarray(
            [1],
            dtype=np.int64,
        ),
        forecast_horizon_minutes=np.asarray(
            [FORECAST_HORIZON_MIN],
            dtype=np.int64,
        ),
        candidate_lookbacks_minutes=np.asarray(
            LOOKBACKS_MIN,
            dtype=np.int64,
        ),
    )


def dataset_complete(dataset_dir: Path):
    required = [
        dataset_dir / "Antarctic.npz",
        dataset_dir / "Atlantic.npz",
        dataset_dir / "West_Coast.npz",
        dataset_dir / "dataset_build_audit.csv",
        dataset_dir / "schema.json",
    ]
    return all(p.exists() for p in required)


def build_or_reuse_dataset(
    *,
    project_root: Path,
    raw_dir: Path,
    dataset_dir: Path,
    progress_every: int,
    rebuild: bool,
):
    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        dataset_complete(dataset_dir)
        and not rebuild
    ):
        log(
            "[DATASET] Reusing existing common-60 1-min development dataset."
        )
        return

    stage12b, stage12b_path = (
        load_stage12b(
            project_root
        )
    )

    xr = stage12b.import_xarray()
    grouped = stage12b.discover_mission_files(
        raw_dir
    )

    # Hard firewall: only development missions are loaded below.
    audits = []

    log("=" * 126)
    log(
        "12P-A DATASET BUILD — 1-MIN INPUT / EXACT +10-MIN TARGET"
    )
    log("=" * 126)
    log(
        f"raw directory : {raw_dir}"
    )
    log(
        f"dataset dir   : {dataset_dir}"
    )
    log(
        f"lookbacks     : {LOOKBACKS_MIN} min"
    )
    log(
        f"common base   : {MAX_LOOKBACK_MIN} consecutive 1-min rows"
    )
    log(
        f"target        : exactly context_end + {FORECAST_HORIZON_MIN} min"
    )
    log(
        "[FIREWALL] Tropical Atlantic is not loaded."
    )
    log("")

    for mission in DEVELOPMENT_MISSIONS:
        paths = grouped[
            mission
        ]

        log(
            f"[BUILD] {mission} | files={len(paths)}"
        )

        raw, raw_audit, _ = (
            stage12b.load_mission_raw(
                xr,
                mission,
                paths,
                progress_every,
            )
        )

        samples, audit = (
            build_common60_samples(
                raw,
                mission,
            )
        )

        safe = mission.replace(
            " ",
            "_",
        )

        save_mission_npz(
            dataset_dir / f"{safe}.npz",
            samples,
        )

        audit.update({
            "source_file_count": int(
                len(paths)
            ),
            "raw_rows_after_stage12b_dedup": int(
                raw_audit["rows_after_dedup"]
            ),
            "stage12b_loader": str(
                stage12b_path
            ),
        })

        audits.append(
            audit
        )

        log(
            f"  raw={len(raw):,} | "
            f"common60 samples={len(samples['X60']):,} | "
            f"X={samples['X60'].shape}"
        )

    pd.DataFrame(
        audits
    ).to_csv(
        dataset_dir
        / "dataset_build_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    schema = {
        "stage": "12P-A",
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "scientific_role": (
            "development-only enhanced-context information screening"
        ),
        "raw_resolution_minutes": 1,
        "forecast_horizon_minutes": FORECAST_HORIZON_MIN,
        "candidate_lookbacks_minutes": LOOKBACKS_MIN,
        "common_sample_policy": (
            "all lookbacks are suffixes of the same valid 60-min context; "
            "therefore every candidate has identical target timestamps"
        ),
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "development_missions": DEVELOPMENT_MISSIONS,
        "Tropical_Atlantic_loaded": False,
        "no_interpolation": True,
        "published_BP_STGNN_reference_context_only": (
            BP_STGNN_PUBLISHED_REFERENCE
        ),
        "BP_STGNN_comparability_caveat": (
            "same +10-min wall-clock forecast horizon, but NOT preprocessing-"
            "identical because BP-STGNN uses a 10-min-resampled modeling dataset"
        ),
    }

    save_json(
        dataset_dir
        / "schema.json",
        schema,
    )


def load_mission_dataset(
    dataset_dir: Path,
    mission: str,
):
    path = (
        dataset_dir
        / f"{mission.replace(' ', '_')}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as z:
        X60 = np.asarray(
            z["X60_raw"],
            dtype=np.float32,
        )
        y = np.asarray(
            z["y_joint_raw"],
            dtype=np.float32,
        )
        aw = np.asarray(
            z["apparent_ref"],
            dtype=np.float32,
        )
        context = np.asarray(
            z["context_end_time_ns"],
            dtype=np.int64,
        ).reshape(-1)
        target = np.asarray(
            z["target_time_ns"],
            dtype=np.int64,
        ).reshape(-1)
        features = [
            str(x)
            for x in z["feature_names"].tolist()
        ]

    if (
        X60.ndim != 3
        or X60.shape[1:] != (
            MAX_LOOKBACK_MIN,
            len(FEATURE_NAMES),
        )
    ):
        raise RuntimeError(
            f"{path.name}: bad X60 shape {X60.shape}"
        )

    if (
        y.ndim != 3
        or y.shape[1:] != (1, 4)
    ):
        raise RuntimeError(
            f"{path.name}: bad y shape {y.shape}"
        )

    if features != FEATURE_NAMES:
        raise RuntimeError(
            f"{path.name}: feature schema mismatch."
        )

    if not np.all(
        target - context
        == FORECAST_HORIZON_MIN
        * ONE_MIN_NS
    ):
        raise RuntimeError(
            f"{path.name}: +10-min target audit failed."
        )

    if not (
        np.isfinite(X60).all()
        and np.isfinite(y).all()
        and np.isfinite(aw).all()
    ):
        raise RuntimeError(
            f"{path.name}: non-finite data."
        )

    return {
        "X60": X60,
        "y": y,
        "aw": aw,
        "context_end_time_ns": context,
        "target_time_ns": target,
        "mission": mission,
    }


def concat_missions(items):
    return {
        "X60": np.concatenate(
            [x["X60"] for x in items],
            axis=0,
        ),
        "y": np.concatenate(
            [x["y"] for x in items],
            axis=0,
        ),
        "aw": np.concatenate(
            [x["aw"] for x in items],
            axis=0,
        ),
    }


def fit_scalers(
    X_train: np.ndarray,
    y_train_wind: np.ndarray,
):
    X = np.asarray(
        X_train,
        dtype=np.float64,
    )
    y = np.asarray(
        y_train_wind,
        dtype=np.float64,
    )

    x_mean = np.mean(
        X,
        axis=(0, 1),
    )
    x_std = np.std(
        X,
        axis=(0, 1),
        ddof=0,
    )
    x_std = np.where(
        x_std < 1e-8,
        1.0,
        x_std,
    )

    y_mean = np.mean(
        y,
        axis=0,
    )
    y_std = np.std(
        y,
        axis=0,
        ddof=0,
    )
    y_std = np.where(
        y_std < 1e-8,
        1.0,
        y_std,
    )

    return {
        "x_mean": x_mean.astype(
            np.float32
        ),
        "x_std": x_std.astype(
            np.float32
        ),
        "y_mean": y_mean.astype(
            np.float32
        ),
        "y_std": y_std.astype(
            np.float32
        ),
    }


def standardize_X(
    X: np.ndarray,
    scaler: dict,
):
    return (
        (
            np.asarray(
                X,
                dtype=np.float32,
            )
            - scaler["x_mean"][None, None, :]
        )
        / scaler["x_std"][None, None, :]
    ).astype(np.float32)


def standardize_y(
    y: np.ndarray,
    scaler: dict,
):
    return (
        (
            np.asarray(
                y,
                dtype=np.float32,
            )
            - scaler["y_mean"][None, :]
        )
        / scaler["y_std"][None, :]
    ).astype(np.float32)


def inverse_y(
    yz: np.ndarray,
    scaler: dict,
):
    return (
        np.asarray(
            yz,
            dtype=np.float32,
        )
        * scaler["y_std"][None, :]
        + scaler["y_mean"][None, :]
    ).astype(np.float32)


def circular_diff_deg(
    pred_deg,
    true_deg,
):
    return (
        (
            np.asarray(
                pred_deg,
                dtype=np.float64,
            )
            - np.asarray(
                true_deg,
                dtype=np.float64,
            )
            + 180.0
        )
        % 360.0
        - 180.0
    )


def meteorological_wind_direction_from_uv(
    uv: np.ndarray,
):
    uv = np.asarray(
        uv,
        dtype=np.float64,
    )

    U = uv[:, 0]
    V = uv[:, 1]

    # Meteorological FROM direction, matching the benchmark convention.
    return (
        np.degrees(
            np.arctan2(
                -U,
                -V,
            )
        )
        % 360.0
    )


def evaluate_wind(
    y_true_uv: np.ndarray,
    y_pred_uv: np.ndarray,
):
    yt = np.asarray(
        y_true_uv,
        dtype=np.float64,
    )
    yp = np.asarray(
        y_pred_uv,
        dtype=np.float64,
    )

    err = yp - yt

    u_rmse = float(
        np.sqrt(
            np.mean(
                err[:, 0] ** 2
            )
        )
    )
    v_rmse = float(
        np.sqrt(
            np.mean(
                err[:, 1] ** 2
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

    wd_true = (
        meteorological_wind_direction_from_uv(
            yt
        )
    )
    wd_pred = (
        meteorological_wind_direction_from_uv(
            yp
        )
    )
    wd_err = circular_diff_deg(
        wd_pred,
        wd_true,
    )

    wd_mae = float(
        np.mean(
            np.abs(
                wd_err
            )
        )
    )
    wd_rmse = float(
        np.sqrt(
            np.mean(
                wd_err ** 2
            )
        )
    )

    return {
        "wind_U_RMSE_mps": u_rmse,
        "wind_V_RMSE_mps": v_rmse,
        "wind_vector_RMSE_mps": vector_rmse,
        "wind_speed_RMSE_mps": ws_rmse,
        "wind_direction_MAE_deg": wd_mae,
        "wind_direction_RMSE_deg": wd_rmse,
    }


def fit_predict_ridge(
    X_train,
    y_train_uv,
    X_val,
):
    scaler = fit_scalers(
        X_train,
        y_train_uv,
    )

    Xtr_z = standardize_X(
        X_train,
        scaler,
    )
    Xva_z = standardize_X(
        X_val,
        scaler,
    )
    ytr_z = standardize_y(
        y_train_uv,
        scaler,
    )

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        Xtr_z.reshape(
            len(Xtr_z),
            -1,
        ),
        ytr_z,
    )

    pred_z = model.predict(
        Xva_z.reshape(
            len(Xva_z),
            -1,
        )
    )

    pred_raw = inverse_y(
        pred_z,
        scaler,
    )

    parameter_count = int(
        np.asarray(
            model.coef_
        ).size
        + np.asarray(
            model.intercept_
        ).size
    )

    return (
        pred_raw,
        parameter_count,
    )


def fold_score(
    row: dict,
    l6_row: dict,
):
    ratios = {
        "U_ratio": (
            row["wind_U_RMSE_mps"]
            / max(
                l6_row["wind_U_RMSE_mps"],
                EPS,
            )
        ),
        "V_ratio": (
            row["wind_V_RMSE_mps"]
            / max(
                l6_row["wind_V_RMSE_mps"],
                EPS,
            )
        ),
        "WS_ratio": (
            row["wind_speed_RMSE_mps"]
            / max(
                l6_row["wind_speed_RMSE_mps"],
                EPS,
            )
        ),
        "WD_ratio": (
            row["wind_direction_RMSE_deg"]
            / max(
                l6_row["wind_direction_RMSE_deg"],
                EPS,
            )
        ),
    }

    score = (
        SCORE_WEIGHTS["U"]
        * ratios["U_ratio"]
        + SCORE_WEIGHTS["V"]
        * ratios["V_ratio"]
        + SCORE_WEIGHTS["WS"]
        * ratios["WS_ratio"]
        + SCORE_WEIGHTS["WD"]
        * ratios["WD_ratio"]
    )

    return float(score), ratios


def run_lomo_screen(
    *,
    dataset_dir: Path,
    output_dir: Path,
):
    missions = {
        m: load_mission_dataset(
            dataset_dir,
            m,
        )
        for m in DEVELOPMENT_MISSIONS
    }

    rows = []

    log("")
    log("=" * 126)
    log(
        "12P-A RIDGE LOMO — SAME SAMPLES, VARIABLE 1-MIN LOOKBACK"
    )
    log("=" * 126)
    log(
        f"lookbacks : {LOOKBACKS_MIN} min"
    )
    log(
        f"horizon   : +{FORECAST_HORIZON_MIN} min"
    )
    log(
        "[FIREWALL] Tropical Atlantic is not accessed."
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
                missions[m]
                for m in train_names
            ]
        )
        val = missions[
            held_out
        ]

        log("-" * 126)
        log(
            f"[{held_out}] "
            f"Ntrain={len(train['X60']):,} | "
            f"Nval={len(val['X60']):,}"
        )

        fold_temp = []

        for lookback in LOOKBACKS_MIN:
            Xtr = train[
                "X60"
            ][:, -lookback:, :]
            Xva = val[
                "X60"
            ][:, -lookback:, :]

            ytr_uv = train[
                "y"
            ][:, 0, 0:2]
            yva_uv = val[
                "y"
            ][:, 0, 0:2]

            pred_uv, param_count = (
                fit_predict_ridge(
                    Xtr,
                    ytr_uv,
                    Xva,
                )
            )

            metrics = evaluate_wind(
                yva_uv,
                pred_uv,
            )

            persistence_uv = (
                Xva[:, -1, 0:2]
                .astype(
                    np.float32
                )
            )
            persistence_metrics = (
                evaluate_wind(
                    yva_uv,
                    persistence_uv,
                )
            )

            row = {
                "held_out_mission": held_out,
                "lookback_minutes": int(
                    lookback
                ),
                "raw_resolution_minutes": 1,
                "forecast_horizon_minutes": int(
                    FORECAST_HORIZON_MIN
                ),
                "n_train": int(
                    len(Xtr)
                ),
                "n_val": int(
                    len(Xva)
                ),
                "ridge_alpha": float(
                    RIDGE_ALPHA
                ),
                "ridge_parameter_count": int(
                    param_count
                ),
                **metrics,
                **{
                    f"persistence_{k}": v
                    for k, v
                    in persistence_metrics.items()
                },
            }

            fold_temp.append(
                row
            )

        l6 = next(
            x
            for x in fold_temp
            if int(
                x["lookback_minutes"]
            ) == 6
        )

        for row in fold_temp:
            score, ratios = fold_score(
                row,
                l6,
            )
            row[
                "selection_score_vs_L6"
            ] = score
            row.update(
                ratios
            )
            rows.append(
                row
            )

            log(
                f"  L={row['lookback_minutes']:2d} min | "
                f"score={score:.5f} | "
                f"U={row['wind_U_RMSE_mps']:.4f} | "
                f"V={row['wind_V_RMSE_mps']:.4f} | "
                f"WS={row['wind_speed_RMSE_mps']:.4f} | "
                f"WD={row['wind_direction_RMSE_deg']:.3f}"
            )

    fold_df = pd.DataFrame(
        rows
    )

    fold_path = (
        output_dir
        / "ridge_lomo_fold_results.csv"
    )
    fold_df.to_csv(
        fold_path,
        index=False,
        encoding="utf-8-sig",
    )

    return fold_df


def summarize_and_select(
    fold_df: pd.DataFrame,
    output_dir: Path,
):
    rows = []

    l6 = fold_df.loc[
        fold_df["lookback_minutes"]
        == 6
    ].copy()

    l6_metric_means = {
        col: float(
            l6[col].mean()
        )
        for col in [
            "wind_U_RMSE_mps",
            "wind_V_RMSE_mps",
            "wind_vector_RMSE_mps",
            "wind_speed_RMSE_mps",
            "wind_direction_RMSE_deg",
        ]
    }

    for lookback in LOOKBACKS_MIN:
        sub = fold_df.loc[
            fold_df[
                "lookback_minutes"
            ]
            == lookback
        ].copy()

        if len(sub) != len(
            DEVELOPMENT_MISSIONS
        ):
            raise RuntimeError(
                f"L={lookback}: expected "
                f"{len(DEVELOPMENT_MISSIONS)} fold rows, "
                f"found {len(sub)}."
            )

        mission_wins = int(
            np.sum(
                sub[
                    "selection_score_vs_L6"
                ].to_numpy(
                    dtype=float
                )
                < 1.0 - 1e-12
            )
        )

        row = {
            "lookback_minutes": int(
                lookback
            ),
            "folds": int(
                len(sub)
            ),
            "selection_score_mean": float(
                sub[
                    "selection_score_vs_L6"
                ].mean()
            ),
            "selection_score_std": float(
                sub[
                    "selection_score_vs_L6"
                ].std(
                    ddof=0
                )
            ),
            "mission_wins_vs_L6": int(
                mission_wins
            ),
            "ridge_parameter_count": int(
                round(
                    sub[
                        "ridge_parameter_count"
                    ].mean()
                )
            ),
        }

        for col in [
            "wind_U_RMSE_mps",
            "wind_V_RMSE_mps",
            "wind_vector_RMSE_mps",
            "wind_speed_RMSE_mps",
            "wind_direction_RMSE_deg",
        ]:
            vals = sub[
                col
            ].to_numpy(
                dtype=float
            )

            row[
                f"{col}_mean"
            ] = float(
                np.mean(
                    vals
                )
            )
            row[
                f"{col}_std"
            ] = float(
                np.std(
                    vals,
                    ddof=0,
                )
            )

            row[
                f"{col}_improvement_vs_L6_fraction"
            ] = float(
                (
                    l6_metric_means[col]
                    - row[
                        f"{col}_mean"
                    ]
                )
                / max(
                    l6_metric_means[col],
                    EPS,
                )
            )

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    ).sort_values(
        [
            "selection_score_mean",
            "lookback_minutes",
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
        / "lookback_summary.csv"
    )
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    raw_best = summary.iloc[
        0
    ]

    raw_best_L = int(
        raw_best[
            "lookback_minutes"
        ]
    )
    raw_best_score = float(
        raw_best[
            "selection_score_mean"
        ]
    )
    raw_best_wins = int(
        raw_best[
            "mission_wins_vs_L6"
        ]
    )

    score_improvement = (
        1.0
        - raw_best_score
    )

    if raw_best_L == 6:
        selected_L = 6
        reason = (
            "L=6 has the lowest mean development-only LOMO score."
        )
    elif (
        score_improvement
        >= MIN_LONGER_LOOKBACK_SCORE_IMPROVEMENT
        and raw_best_wins
        >= MIN_MISSION_WINS
    ):
        selected_L = raw_best_L
        reason = (
            f"L={raw_best_L} is the lowest-score lookback, improves the "
            f"mean score by {100.0*score_improvement:.3f}% versus L=6, "
            f"and wins {raw_best_wins}/3 held-out missions."
        )
    else:
        selected_L = 6
        reason = (
            f"Raw best L={raw_best_L} does not clear the predeclared "
            "complexity threshold: at least 0.5% mean score improvement "
            "and >=2/3 mission wins are required. Retain L=6."
        )

    selected_row = summary.loc[
        summary[
            "lookback_minutes"
        ]
        == selected_L
    ].iloc[0]

    decision = {
        "stage": "12P-A",
        "script_version": SCRIPT_VERSION,
        "selected_lookback_minutes": int(
            selected_L
        ),
        "raw_best_lookback_minutes": int(
            raw_best_L
        ),
        "raw_best_score_mean": float(
            raw_best_score
        ),
        "selected_score_mean": float(
            selected_row[
                "selection_score_mean"
            ]
        ),
        "selection_reason": reason,
        "predeclared_rule": {
            "candidate_lookbacks_minutes": LOOKBACKS_MIN,
            "reference_lookback_minutes": 6,
            "minimum_mean_score_improvement_for_longer_context": (
                MIN_LONGER_LOOKBACK_SCORE_IMPROVEMENT
            ),
            "minimum_mission_wins": (
                MIN_MISSION_WINS
            ),
            "score_weights": (
                SCORE_WEIGHTS
            ),
        },
        "task": {
            "raw_resolution_minutes": 1,
            "forecast_horizon_minutes": (
                FORECAST_HORIZON_MIN
            ),
            "target_definition": (
                "wind U,V at exactly context_end + 10 min"
            ),
            "common_sample_max_lookback_minutes": (
                MAX_LOOKBACK_MIN
            ),
        },
        "development_only": True,
        "development_missions": DEVELOPMENT_MISSIONS,
        "Tropical_Atlantic_loaded": False,
        "published_BP_STGNN_reference_context_only": (
            BP_STGNN_PUBLISHED_REFERENCE
        ),
        "BP_STGNN_comparability_caveat": (
            "same 10-min wall-clock horizon; not preprocessing-identical "
            "because BP-STGNN first uses a 10-min-resampled modeling dataset"
        ),
        "next_stage": (
            "12P-B: with the selected lookback frozen, add only physically "
            "interpretable temporal tendency/variability features, then compare "
            "Enhanced-Ridge / compact MLP / compact GRU on development-only LOMO."
        ),
    }

    selected_path = (
        output_dir
        / "selected_lookback.json"
    )
    save_json(
        selected_path,
        decision,
    )

    report_path = (
        output_dir
        / "12P_A_REPORT.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "12P-A One-Minute Lookback Screening, Exact +10-Minute Horizon\n"
        )
        f.write(
            "=" * 126
            + "\n\n"
        )

        f.write(
            "SCIENTIFIC TASK\n"
        )
        f.write(
            "-" * 126
            + "\n"
        )
        f.write(
            "Raw input resolution: 1 min\n"
        )
        f.write(
            f"Forecast target: exactly t+{FORECAST_HORIZON_MIN} min\n"
        )
        f.write(
            f"Candidate lookbacks: {LOOKBACKS_MIN} min\n"
        )
        f.write(
            "All candidates use identical target timestamps via the common "
            "valid 60-min context sample set.\n"
        )
        f.write(
            "Tropical Atlantic accessed: NO\n\n"
        )

        f.write(
            "BP-STGNN COMPARABILITY NOTE\n"
        )
        f.write(
            "-" * 126
            + "\n"
        )
        f.write(
            "This experiment shares the 10-min wall-clock forecast horizon "
            "with BP-STGNN, but is NOT preprocessing-identical because the "
            "published method first resamples the original 1-min data to a "
            "10-min modeling interval.\n\n"
        )

        f.write(
            "LOOKBACK SUMMARY\n"
        )
        f.write(
            "-" * 126
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
            "SELECTION\n"
        )
        f.write(
            "-" * 126
            + "\n"
        )
        f.write(
            f"Raw best lookback: {raw_best_L} min\n"
        )
        f.write(
            f"Selected lookback: {selected_L} min\n"
        )
        f.write(
            f"Reason: {reason}\n\n"
        )

        f.write(
            "PUBLISHED BP-STGNN VALUES — CONTEXT ONLY, NOT USED FOR SELECTION\n"
        )
        f.write(
            "-" * 126
            + "\n"
        )
        for k, v in BP_STGNN_PUBLISHED_REFERENCE.items():
            f.write(
                f"{k}: {v}\n"
            )

    return (
        summary,
        decision,
        report_path,
    )


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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help=(
            "Rebuild the development-only common-60 1-min dataset even "
            "when complete outputs already exist."
        ),
    )

    args = parser.parse_args()

    project_root = (
        args.project_root
    )
    raw_dir = (
        args.raw_dir
    )
    output_dir = (
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_dir = (
        output_dir
        / "dataset_1min_common60"
    )

    log("=" * 126)
    log(
        "12P-A — 1-MIN LOOKBACK SCREEN / EXACT +10-MIN WIND TARGET"
    )
    log("=" * 126)
    log(
        f"script version : {SCRIPT_VERSION}"
    )
    log(
        f"project root   : {project_root}"
    )
    log(
        f"raw dir        : {raw_dir}"
    )
    log(
        f"output dir     : {output_dir}"
    )
    log(
        f"lookbacks      : {LOOKBACKS_MIN} min"
    )
    log(
        f"forecast       : +{FORECAST_HORIZON_MIN} min exactly"
    )
    log(
        "[FIREWALL] Antarctic / Atlantic / West Coast only."
    )
    log(
        "[CAVEAT] Same 10-min wall-clock horizon as BP-STGNN; "
        "not preprocessing-identical."
    )
    log("")

    build_or_reuse_dataset(
        project_root=project_root,
        raw_dir=raw_dir,
        dataset_dir=dataset_dir,
        progress_every=args.progress_every,
        rebuild=args.rebuild_dataset,
    )

    fold_df = run_lomo_screen(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
    )

    (
        summary,
        decision,
        report_path,
    ) = summarize_and_select(
        fold_df,
        output_dir,
    )

    log("")
    log("=" * 126)
    log(
        "12P-A LOOKBACK SUMMARY"
    )
    log("=" * 126)
    log(
        summary.to_string(
            index=False
        )
    )
    log("")
    log(
        f"[RAW BEST] L="
        f"{decision['raw_best_lookback_minutes']} min"
    )
    log(
        f"[SELECTED] L="
        f"{decision['selected_lookback_minutes']} min"
    )
    log(
        f"[REASON] "
        f"{decision['selection_reason']}"
    )
    log("")
    log(
        "Published BP-STGNN reference (context only; "
        "NOT used for development selection):"
    )
    log(
        "  U=0.74864 | V=0.70506 | "
        "WS=0.69940 | WD=5.584 deg"
    )
    log(
        "[CAVEAT] Do not claim protocol-identical superiority from 12P-A. "
        "This is a 1-min enhanced-context task with the same +10-min horizon."
    )
    log("")
    log(
        f"[SAVED] "
        f"{output_dir / 'ridge_lomo_fold_results.csv'}"
    )
    log(
        f"[SAVED] "
        f"{output_dir / 'lookback_summary.csv'}"
    )
    log(
        f"[SAVED] "
        f"{output_dir / 'selected_lookback.json'}"
    )
    log(
        f"[SAVED] "
        f"{report_path}"
    )
    log("")
    log(
        "NEXT: freeze the selected lookback and build 12P-B temporal "
        "tendency/variability features without changing the +10-min target."
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
