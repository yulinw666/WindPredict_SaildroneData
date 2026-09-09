# -*- coding: utf-8 -*-
r"""
12U_train_MultiscaleTurbulenceAware_UV_Nie10min_LOMO.py

Stage 12U
=========
Development-only multiscale / turbulence-aware true-wind forecasting under a
TRUE 10-minute block-mean protocol.

This stage intentionally moves away from Stage-12G point sampling.  It reuses
the exact DEVELOPMENT sample timestamps from Stage 12B, whose 10-min nodes are
formed from complete 10 x 1-min blocks.  This is closer to the published
BP-STGNN statement that original 1-min Saildrone data were converted to a
10-min modeling interval.

Frozen task
-----------
Input:
    6 consecutive 10-min blocks = 60 min history

Output:
    immediately following 10-min block mean U,V

Development missions only:
    Antarctic
    Atlantic
    West Coast

Tropical Atlantic:
    NEVER loaded / opened / built by this script.

Core scientific idea
--------------------
A 10-min mean state discards potentially predictive sub-window information.
Stage 12T found that the magnitude of the speed prediction error is related to
historical wind variability.  Stage 12U therefore preserves the 10-min
forecasting scale while adding CAUSAL descriptors computed only from the ten
already-observed 1-min records inside each historical 10-min block.

Each block starts from:
    U_mean, V_mean, T_mean, RH_mean

and may add:
    U_std
    V_std
    WS_std
    TI_WS
    DIR_dispersion = 1 - directional resultant length
    scalar_vector_gap = mean(|W_1min|) - |mean(W_vector)|

Additional full descriptors:
    WS_gust = max(WS_1min) - mean(WS_1min)
    WS_range
    WS_slope within the 10-min block
    U_slope within the block
    V_slope within the block
    UV_covariance

All target quantities remain the NEXT block's 10-min mean U,V.

Feature groups
--------------
Base4:
    [U_mean, V_mean, T_mean, RH_mean]

TurbCore10:
    Base4 +
    [U_std, V_std, WS_std, TI_WS, DIR_dispersion, scalar_vector_gap]

TurbFull16:
    TurbCore10 +
    [WS_gust, WS_range, WS_slope, U_slope, V_slope, UV_cov]

Candidate models
----------------
1) Base4-Ridge
2) TurbCore10-Ridge
3) TurbFull16-Ridge

4) TurbFull16-RidgeResidualMLP
5) TurbFull16-RidgeResidualGRU

6) TurbFull16-HGBR-Direct
   Separate sklearn HistGradientBoosting regressors for U and V.

7) TurbFull16-RidgeResidual-HGBR
   HistGradientBoosting predicts only the U/V residual around the full Ridge
   anchor.

Why both direct and residual HGBR?
---------------------------------
The residual experiment tests whether nonlinear structure remains after a
strong multiscale linear anchor.  The direct HGBR is a non-neural nonlinear
alternative and prevents the experiment from assuming residual learning is the
only useful mechanism.

Leakage control for residual targets
------------------------------------
Residual models DO NOT train on in-sample Ridge residuals.

For each outer LOMO training pool, Stage 12U creates 5-fold BLOCKED,
mission-preserving cross-fitted Ridge predictions:
    - each training mission is split into five chronological chunks;
    - inner fold k fits Ridge on all other chunks from both training missions;
    - it predicts chunk k;
    - every training row receives an out-of-fold Ridge prediction.

Residual targets are:
    r_OOF = y_true - y_Ridge_OOF

For the held-out outer mission:
    Ridge is fit on the full two-mission outer training pool.

This avoids making the residual target artificially easy through in-sample
Ridge fitting.

Neural residual safety
----------------------
The final residual head is initialized to zero:
    epoch 0 = exact TurbFull16-Ridge on the held-out mission.

If the neural correction is harmful, the best checkpoint can remain epoch 0.

Metrics
-------
Primary:
    U RMSE
    V RMSE
    vector RMSE
    WS RMSE
    WD RMSE

Development score relative to Base4-Ridge:
    0.25 U ratio
  + 0.35 V ratio
  + 0.25 WS ratio
  + 0.15 WD ratio

A candidate is considered a meaningful development improvement only if:
    mean score < 0.99
    mission wins >= 2/3
    vector RMSE improves > 0.5%
    and U or V improves > 1%

Published BP-STGNN reference values are printed for NUMERICAL CONTEXT ONLY:
    U  0.748640 m/s
    V  0.705060 m/s
    WS 0.699400 m/s
    WD 5.584 deg

They are NOT used for model selection because this enhanced input contains
additional 1-min intra-block statistics.

Output
------
<output-dir>/
    enhanced_dataset/
        Antarctic.npz
        Atlantic.npz
        West_Coast.npz
        build_audit.csv
        feature_schema.json
    candidate_run_results.csv
    candidate_summary.csv
    ridge_feature_group_summary.csv
    residual_diagnostics.csv
    selected_config.json
    selected_candidate_lomo_runs.csv
    histories/
    checkpoints/
    12U_REPORT.txt

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\12U_train_MultiscaleTurbulenceAware_UV_Nie10min_LOMO.py" --project-root "D:\project\WindPredict_SaildroneData" --raw-dir "D:\project\WindPredict_SaildroneData\data\raw\RawData" --stage12b-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12B_Nie_same_mission_benchmark_v0_1" --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12U_MultiscaleTurbulenceAware_UV_Nie10min_LOMO_v0_1"

Smoke test:
    add --debug-fast

Rebuild enhanced dataset:
    add --rebuild-dataset
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import math
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
except Exception as exc:
    raise RuntimeError(
        "scikit-learn is required. Activate the WindPredict environment."
    ) from exc


SCRIPT_VERSION = "0.1.0-12U-MultiscaleTurbulenceAware-UV"

DEFAULT_PROJECT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")
DEFAULT_RAW_DIR = DEFAULT_PROJECT_ROOT / "data" / "raw" / "RawData"
DEFAULT_STAGE12B_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12B_Nie_same_mission_benchmark_v0_1"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12U_MultiscaleTurbulenceAware_UV_Nie10min_LOMO_v0_1"
)

DEVELOPMENT_MISSIONS = [
    "Antarctic",
    "Atlantic",
    "West Coast",
]

LOOKBACK_STEPS = 6
STEP_MINUTES = 10
FORECAST_MINUTES = 10

ONE_MIN_NS = 60 * 1_000_000_000
TEN_MIN_NS = 10 * ONE_MIN_NS

EPS = 1e-12
RIDGE_ALPHA = 1.0
INNER_CROSSFIT_FOLDS = 5

SCREENING_SEEDS = [
    500043,
    501052,
    502061,
]

BP_STGNN_REFERENCE = {
    "U_RMSE_mps": 0.748640,
    "V_RMSE_mps": 0.705060,
    "WS_RMSE_mps": 0.699400,
    "WD_RMSE_deg": 5.584000,
    "params": 242580,
}

SCORE_WEIGHTS = {
    "U": 0.25,
    "V": 0.35,
    "WS": 0.25,
    "WD": 0.15,
}

CONTINUE_SCORE_THRESHOLD = 0.99
MIN_MISSION_WINS = 2
MIN_VECTOR_IMPROVEMENT = 0.005
MIN_U_OR_V_IMPROVEMENT = 0.01

BASE4_FEATURES = [
    "U_mean",
    "V_mean",
    "T_mean",
    "RH_mean",
]

TURB_CORE_EXTRA = [
    "U_std",
    "V_std",
    "WS_std",
    "TI_WS",
    "DIR_dispersion",
    "scalar_vector_gap",
]

TURB_FULL_EXTRA = [
    "WS_gust",
    "WS_range",
    "WS_slope",
    "U_slope",
    "V_slope",
    "UV_cov",
]

TURB_CORE_FEATURES = (
    BASE4_FEATURES
    + TURB_CORE_EXTRA
)

TURB_FULL_FEATURES = (
    TURB_CORE_FEATURES
    + TURB_FULL_EXTRA
)

ALL_FEATURES = TURB_FULL_FEATURES

FEATURE_GROUPS = {
    "Base4": BASE4_FEATURES,
    "TurbCore10": TURB_CORE_FEATURES,
    "TurbFull16": TURB_FULL_FEATURES,
}

NEURAL_CANDIDATES = {
    "TurbFull16-RidgeResidualMLP": "mlp",
    "TurbFull16-RidgeResidualGRU": "gru",
}


@dataclass(frozen=True)
class NeuralFreeze:
    mlp_hidden_1: int = 64
    mlp_hidden_2: int = 32

    gru_hidden: int = 48
    dense_hidden: int = 48

    dropout: float = 0.10
    learning_rate: float = 8e-4
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 100
    early_stop_patience: int = 15
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-5
    grad_clip_norm: float = 1.0

    correction_penalty: float = 1e-4
    use_amp: bool = True


@dataclass(frozen=True)
class HGBFreeze:
    learning_rate: float = 0.05
    max_iter: int = 250
    max_leaf_nodes: int = 15
    max_depth: int = 4
    min_samples_leaf: int = 50
    l2_regularization: float = 1.0


NEURAL_FREEZE = NeuralFreeze()
HGB_FREEZE = HGBFreeze()


# =============================================================================
# General utilities
# =============================================================================

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


def import_stage12b(project_root: Path):
    path = (
        project_root
        / "src"
        / "12B_build_Nie_same_mission_benchmark_dataset.py"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    name = "stage12b_for_stage12u"

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import {path}"
        )

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod

    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise

    for fn in [
        "import_xarray",
        "discover_mission_files",
        "load_mission_raw",
        "build_complete_10min_blocks",
    ]:
        if not hasattr(mod, fn):
            raise RuntimeError(
                f"Stage12B helper missing: {fn}"
            )

    return mod, path


# =============================================================================
# Torch utilities
# =============================================================================

def import_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required. Activate the WindPredict environment."
        ) from exc

    return torch, nn


def configure_cuda(torch):
    if not torch.cuda.is_available():
        return

    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def seed_everything(torch, seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def amp_context(torch, enabled: bool, device):
    if (
        not enabled
        or device.type != "cuda"
    ):
        return contextlib.nullcontext()

    try:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        )
    except Exception:
        return torch.cuda.amp.autocast(
            enabled=True
        )


def create_grad_scaler(torch, enabled: bool):
    if not enabled:
        return None

    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )
    except Exception:
        try:
            return torch.cuda.amp.GradScaler(
                enabled=True
            )
        except Exception:
            return None


def state_dict_cpu(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def count_parameters(model):
    return int(
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )


def sequential_batches(*arrays, batch_size: int):
    if not arrays:
        return

    n = len(arrays[0])

    if any(len(a) != n for a in arrays):
        raise ValueError(
            "Batch arrays have inconsistent first dimension."
        )

    for start in range(0, n, int(batch_size)):
        stop = min(start + int(batch_size), n)

        yield tuple(
            a[start:stop]
            for a in arrays
        )


# =============================================================================
# Stage-12B exact sample timestamps
# =============================================================================

def stage12b_npz_path(
    stage12b_dir: Path,
    mission: str,
):
    safe = mission.replace(" ", "_")

    return (
        stage12b_dir
        / "track_J_joint_compatible"
        / f"{safe}.npz"
    )


def load_stage12b_reference_sample_times(
    stage12b_dir: Path,
    mission: str,
):
    path = stage12b_npz_path(
        stage12b_dir,
        mission,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Stage12B reference dataset missing: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as z:
        context = np.asarray(
            z["context_end_time_ns"],
            dtype=np.int64,
        ).reshape(-1)

        target = np.asarray(
            z["target_time_ns"],
            dtype=np.int64,
        )

        y = np.asarray(
            z["y_wind_raw"],
            dtype=np.float32,
        )

        lookback = int(
            np.asarray(
                z["lookback_steps"]
            ).reshape(-1)[0]
        )

        step = int(
            np.asarray(
                z["step_minutes"]
            ).reshape(-1)[0]
        )

    if target.ndim == 2:
        target = target[:, 0]

    target = target.reshape(-1)

    if y.ndim == 3:
        y = y[:, 0, :]

    if (
        lookback != LOOKBACK_STEPS
        or step != STEP_MINUTES
    ):
        raise RuntimeError(
            f"{mission}: Stage12B protocol mismatch."
        )

    if not np.all(
        target - context == TEN_MIN_NS
    ):
        raise RuntimeError(
            f"{mission}: Stage12B target not exactly next 10-min block."
        )

    return {
        "context_end_time_ns": context,
        "target_time_ns": target,
        "y_uv": y.astype(np.float32),
        "reference_path": path,
    }


# =============================================================================
# Intra-block feature construction
# =============================================================================

def slope_1min(values):
    x = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    t = np.arange(
        len(x),
        dtype=np.float64,
    )

    t0 = t - np.mean(t)
    x0 = x - np.mean(x)

    denom = float(
        np.sum(t0 ** 2)
    )

    return float(
        np.sum(
            t0 * x0
        )
        / max(
            denom,
            EPS,
        )
    )


def compute_block_descriptors(
    raw: pd.DataFrame,
    accepted_blocks: pd.DataFrame,
):
    """
    Use Stage-12B accepted complete block starts, but compute additional
    descriptors from the 10 already-observed raw 1-min records in each block.
    """
    raw = raw.copy()

    raw["block_start_ns"] = (
        raw["time_ns"].to_numpy(
            dtype=np.int64
        )
        // TEN_MIN_NS
    ) * TEN_MIN_NS

    accepted_starts = set(
        accepted_blocks[
            "block_start_ns"
        ].to_numpy(
            dtype=np.int64
        ).tolist()
    )

    rows = []

    grouped = raw.groupby(
        "block_start_ns",
        sort=True,
    )

    accepted_lookup = (
        accepted_blocks
        .set_index(
            "block_start_ns"
        )
    )

    for block_start, block in grouped:
        block_start = int(block_start)

        if block_start not in accepted_starts:
            continue

        if len(block) != 10:
            raise RuntimeError(
                f"Accepted Stage12B block {block_start} does not have 10 raw rows."
            )

        U = block["U"].to_numpy(
            dtype=np.float64
        )
        V = block["V"].to_numpy(
            dtype=np.float64
        )
        T = block["T"].to_numpy(
            dtype=np.float64
        )
        RH = block["RH"].to_numpy(
            dtype=np.float64
        )

        if not (
            np.isfinite(U).all()
            and np.isfinite(V).all()
            and np.isfinite(T).all()
            and np.isfinite(RH).all()
        ):
            raise RuntimeError(
                "Accepted Stage12B block contains nonfinite meteorological data."
            )

        ref = accepted_lookup.loc[
            block_start
        ]

        U_mean = float(ref["U"])
        V_mean = float(ref["V"])
        T_mean = float(ref["T"])
        RH_mean = float(ref["RH"])

        ws_1m = np.hypot(
            U,
            V,
        )

        ws_scalar_mean = float(
            np.mean(
                ws_1m
            )
        )

        ws_vector_mean = float(
            np.hypot(
                U_mean,
                V_mean,
            )
        )

        u_std = float(
            np.std(
                U,
                ddof=0,
            )
        )

        v_std = float(
            np.std(
                V,
                ddof=0,
            )
        )

        ws_std = float(
            np.std(
                ws_1m,
                ddof=0,
            )
        )

        ti_ws = float(
            ws_std
            / max(
                ws_scalar_mean,
                0.5,
            )
        )

        theta = np.arctan2(
            -U,
            -V,
        )

        resultant = float(
            np.hypot(
                np.mean(
                    np.sin(theta)
                ),
                np.mean(
                    np.cos(theta)
                ),
            )
        )

        resultant = min(
            max(
                resultant,
                0.0,
            ),
            1.0,
        )

        dir_dispersion = float(
            1.0
            - resultant
        )

        scalar_vector_gap = float(
            ws_scalar_mean
            - ws_vector_mean
        )

        ws_gust = float(
            np.max(
                ws_1m
            )
            - ws_scalar_mean
        )

        ws_range = float(
            np.max(
                ws_1m
            )
            - np.min(
                ws_1m
            )
        )

        uv_cov = float(
            np.mean(
                (
                    U
                    - np.mean(U)
                )
                * (
                    V
                    - np.mean(V)
                )
            )
        )

        row = {
            "block_start_ns": block_start,
            "U_mean": U_mean,
            "V_mean": V_mean,
            "T_mean": T_mean,
            "RH_mean": RH_mean,
            "U_std": u_std,
            "V_std": v_std,
            "WS_std": ws_std,
            "TI_WS": ti_ws,
            "DIR_dispersion": dir_dispersion,
            "scalar_vector_gap": scalar_vector_gap,
            "WS_gust": ws_gust,
            "WS_range": ws_range,
            "WS_slope": slope_1min(
                ws_1m
            ),
            "U_slope": slope_1min(
                U
            ),
            "V_slope": slope_1min(
                V
            ),
            "UV_cov": uv_cov,
        }

        if not np.isfinite(
            np.asarray(
                [
                    row[f]
                    for f in ALL_FEATURES
                ],
                dtype=np.float64,
            )
        ).all():
            raise RuntimeError(
                f"Nonfinite enhanced descriptor in block {block_start}."
            )

        rows.append(row)

    result = pd.DataFrame(
        rows
    ).sort_values(
        "block_start_ns"
    ).reset_index(
        drop=True
    )

    if len(result) != len(
        accepted_blocks
    ):
        raise RuntimeError(
            "Enhanced block count does not match Stage12B accepted block count: "
            f"{len(result)} vs {len(accepted_blocks)}"
        )

    return result


def build_enhanced_mission_dataset(
    *,
    stage12b,
    xr,
    grouped,
    stage12b_dir: Path,
    mission: str,
    progress_every: int,
):
    ref = load_stage12b_reference_sample_times(
        stage12b_dir,
        mission,
    )

    raw, raw_audit, _ = stage12b.load_mission_raw(
        xr,
        mission,
        grouped[mission],
        progress_every,
    )

    blocks, block_audit = (
        stage12b.build_complete_10min_blocks(
            raw
        )
    )

    if blocks.empty:
        raise RuntimeError(
            f"{mission}: no complete Stage12B blocks."
        )

    enhanced_blocks = compute_block_descriptors(
        raw,
        blocks,
    )

    lookup = {
        int(row["block_start_ns"]): row
        for row in enhanced_blocks.to_dict(
            orient="records"
        )
    }

    X = []
    y = []
    context_out = []
    target_out = []

    for i, (
        context_end,
        target_time,
    ) in enumerate(
        zip(
            ref[
                "context_end_time_ns"
            ],
            ref[
                "target_time_ns"
            ],
        )
    ):
        context_starts = (
            int(context_end)
            - np.arange(
                LOOKBACK_STEPS - 1,
                -1,
                -1,
                dtype=np.int64,
            )
            * TEN_MIN_NS
        )

        rows = []

        for t in context_starts:
            key = int(t)

            if key not in lookup:
                raise RuntimeError(
                    f"{mission}: Stage12B reference context block {key} "
                    "missing from enhanced block lookup."
                )

            rows.append(
                [
                    lookup[key][f]
                    for f in ALL_FEATURES
                ]
            )

        X.append(
            rows
        )

        y.append(
            ref["y_uv"][
                i
            ]
        )

        context_out.append(
            int(
                context_end
            )
        )

        target_out.append(
            int(
                target_time
            )
        )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.float32,
    )

    context_out = np.asarray(
        context_out,
        dtype=np.int64,
    )

    target_out = np.asarray(
        target_out,
        dtype=np.int64,
    )

    if (
        X.shape
        != (
            len(
                ref[
                    "context_end_time_ns"
                ]
            ),
            LOOKBACK_STEPS,
            len(
                ALL_FEATURES
            ),
        )
    ):
        raise RuntimeError(
            f"{mission}: bad enhanced X shape {X.shape}"
        )

    if not (
        np.isfinite(X).all()
        and np.isfinite(y).all()
    ):
        raise RuntimeError(
            f"{mission}: nonfinite enhanced dataset."
        )

    if not np.array_equal(
        context_out,
        ref[
            "context_end_time_ns"
        ],
    ):
        raise RuntimeError(
            f"{mission}: context-time alignment failure."
        )

    if not np.array_equal(
        target_out,
        ref[
            "target_time_ns"
        ],
    ):
        raise RuntimeError(
            f"{mission}: target-time alignment failure."
        )

    audit = {
        "mission": mission,
        "source_segment_files": int(
            raw_audit[
                "segment_files"
            ]
        ),
        "raw_rows_after_dedup": int(
            raw_audit[
                "rows_after_dedup"
            ]
        ),
        "accepted_stage12b_blocks": int(
            len(
                blocks
            )
        ),
        "enhanced_blocks": int(
            len(
                enhanced_blocks
            )
        ),
        "stage12b_reference_samples": int(
            len(
                ref[
                    "context_end_time_ns"
                ]
            )
        ),
        "enhanced_samples": int(
            len(X)
        ),
        "alignment_exact": True,
        "Stage12B_rejected_bad_raw_count": int(
            block_audit[
                "rejected_bad_raw_count"
            ]
        ),
        "Stage12B_rejected_bad_timestamp_grid": int(
            block_audit[
                "rejected_bad_timestamp_grid"
            ]
        ),
        "Stage12B_rejected_nonfinite_common_base": int(
            block_audit[
                "rejected_nonfinite_common_base"
            ]
        ),
    }

    return {
        "X": X,
        "y_uv": y,
        "context_end_time_ns": context_out,
        "target_time_ns": target_out,
        "audit": audit,
    }


def enhanced_dataset_complete(
    dataset_dir: Path,
):
    required = [
        dataset_dir
        / f"{m.replace(' ', '_')}.npz"
        for m in DEVELOPMENT_MISSIONS
    ]

    required.extend(
        [
            dataset_dir
            / "build_audit.csv",
            dataset_dir
            / "feature_schema.json",
        ]
    )

    return all(
        p.exists()
        for p in required
    )


def build_enhanced_dataset(
    *,
    project_root: Path,
    raw_dir: Path,
    stage12b_dir: Path,
    dataset_dir: Path,
    progress_every: int,
):
    stage12b, stage12b_path = (
        import_stage12b(
            project_root
        )
    )

    xr = stage12b.import_xarray()
    grouped = stage12b.discover_mission_files(
        raw_dir
    )

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audits = []

    log("=" * 132)
    log(
        "12U DATASET BUILD — TRUE 10-MIN MEANS + 1-MIN INTRA-BLOCK DESCRIPTORS"
    )
    log("=" * 132)
    log(
        f"Stage12B source : {stage12b_path}"
    )
    log(
        "[FIREWALL] Building development missions only; Tropical Atlantic is not opened."
    )

    for mission in DEVELOPMENT_MISSIONS:
        log(
            f"[BUILD] {mission} | files={len(grouped[mission])}"
        )

        data = build_enhanced_mission_dataset(
            stage12b=stage12b,
            xr=xr,
            grouped=grouped,
            stage12b_dir=stage12b_dir,
            mission=mission,
            progress_every=progress_every,
        )

        safe = mission.replace(
            " ",
            "_",
        )

        np.savez_compressed(
            dataset_dir
            / f"{safe}.npz",
            X_raw=data["X"],
            y_uv=data["y_uv"],
            context_end_time_ns=(
                data[
                    "context_end_time_ns"
                ]
            ),
            target_time_ns=(
                data[
                    "target_time_ns"
                ]
            ),
            feature_names=np.asarray(
                ALL_FEATURES
            ),
            lookback_steps=np.asarray(
                [
                    LOOKBACK_STEPS
                ],
                dtype=np.int64,
            ),
            step_minutes=np.asarray(
                [
                    STEP_MINUTES
                ],
                dtype=np.int64,
            ),
            forecast_minutes=np.asarray(
                [
                    FORECAST_MINUTES
                ],
                dtype=np.int64,
            ),
        )

        audits.append(
            data[
                "audit"
            ]
        )

        log(
            f"  blocks={data['audit']['enhanced_blocks']:,} | "
            f"samples={data['audit']['enhanced_samples']:,} | "
            "alignment=EXACT"
        )

        gc.collect()

    audit_df = pd.DataFrame(
        audits
    )

    audit_df.to_csv(
        dataset_dir
        / "build_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    schema = {
        "stage": "12U",
        "script_version": SCRIPT_VERSION,
        "source_protocol": (
            "Stage12B exact development sample timestamps; "
            "complete 10 x 1-min UTC-aligned blocks"
        ),
        "task": {
            "raw_resolution_minutes": 1,
            "model_resolution_minutes": 10,
            "lookback_steps": LOOKBACK_STEPS,
            "lookback_minutes": (
                LOOKBACK_STEPS
                * STEP_MINUTES
            ),
            "forecast_minutes": (
                FORECAST_MINUTES
            ),
        },
        "feature_groups": (
            FEATURE_GROUPS
        ),
        "all_feature_names": (
            ALL_FEATURES
        ),
        "target": [
            "U_next_10min_mean",
            "V_next_10min_mean",
        ],
        "development_only": True,
        "development_missions": (
            DEVELOPMENT_MISSIONS
        ),
        "Tropical_Atlantic_used": False,
    }

    save_json(
        dataset_dir
        / "feature_schema.json",
        schema,
    )


def load_enhanced_mission(
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
        X = np.asarray(
            z["X_raw"],
            dtype=np.float32,
        )

        y = np.asarray(
            z["y_uv"],
            dtype=np.float32,
        )

        context = np.asarray(
            z[
                "context_end_time_ns"
            ],
            dtype=np.int64,
        ).reshape(-1)

        target = np.asarray(
            z[
                "target_time_ns"
            ],
            dtype=np.int64,
        ).reshape(-1)

        names = [
            str(x)
            for x
            in z[
                "feature_names"
            ].tolist()
        ]

    if names != ALL_FEATURES:
        raise RuntimeError(
            f"{mission}: feature schema mismatch."
        )

    if X.shape[1:] != (
        LOOKBACK_STEPS,
        len(
            ALL_FEATURES
        ),
    ):
        raise RuntimeError(
            f"{mission}: bad X shape {X.shape}"
        )

    if not np.all(
        target - context
        == TEN_MIN_NS
    ):
        raise RuntimeError(
            f"{mission}: target not next 10-min block."
        )

    return {
        "X": X,
        "y": y,
        "mission": mission,
    }


def concat_missions(items):
    X = np.concatenate(
        [
            x["X"]
            for x in items
        ],
        axis=0,
    )

    y = np.concatenate(
        [
            x["y"]
            for x in items
        ],
        axis=0,
    )

    labels = np.concatenate(
        [
            np.asarray(
                [
                    x["mission"]
                ]
                * len(
                    x["X"]
                ),
                dtype=object,
            )
            for x in items
        ],
        axis=0,
    )

    return {
        "X": X,
        "y": y,
        "mission_labels": labels,
    }


# =============================================================================
# Feature-group helpers
# =============================================================================

def feature_indices(
    group_name: str,
):
    names = FEATURE_GROUPS[
        group_name
    ]

    lookup = {
        name: i
        for i, name
        in enumerate(
            ALL_FEATURES
        )
    }

    return [
        lookup[name]
        for name in names
    ]


def select_group(
    X,
    group_name: str,
):
    return np.asarray(
        X[
            :,
            :,
            feature_indices(
                group_name
            ),
        ],
        dtype=np.float32,
    )


# =============================================================================
# Scalers and Ridge
# =============================================================================

def fit_scaler(
    X,
    y,
):
    X64 = np.asarray(
        X,
        dtype=np.float64,
    )

    y64 = np.asarray(
        y,
        dtype=np.float64,
    )

    x_mean = np.mean(
        X64,
        axis=(0, 1),
    )

    x_std = np.std(
        X64,
        axis=(0, 1),
        ddof=0,
    )

    x_std = np.where(
        x_std < 1e-8,
        1.0,
        x_std,
    )

    y_mean = np.mean(
        y64,
        axis=0,
    )

    y_std = np.std(
        y64,
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
                "x_mean"
            ][
                None,
                None,
                :,
            ]
        )
        / scaler[
            "x_std"
        ][
            None,
            None,
            :,
        ]
    ).astype(
        np.float32
    )


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
                "y_mean"
            ][
                None,
                :,
            ]
        )
        / scaler[
            "y_std"
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
            "y_std"
        ][
            None,
            :,
        ]
        + scaler[
            "y_mean"
        ][
            None,
            :,
        ]
    ).astype(
        np.float32
    )


def fit_ridge(
    X_train,
    y_train,
):
    scaler = fit_scaler(
        X_train,
        y_train,
    )

    Xz = transform_X(
        X_train,
        scaler,
    ).reshape(
        len(X_train),
        -1,
    )

    yz = transform_y(
        y_train,
        scaler,
    )

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        Xz,
        yz,
    )

    return (
        model,
        scaler,
    )


def ridge_predict(
    model,
    scaler,
    X,
):
    Xz = transform_X(
        X,
        scaler,
    ).reshape(
        len(X),
        -1,
    )

    pred_z = np.asarray(
        model.predict(
            Xz
        ),
        dtype=np.float32,
    )

    return inverse_y(
        pred_z,
        scaler,
    )


def ridge_param_count(
    model,
):
    return int(
        np.asarray(
            model.coef_
        ).size
        + np.asarray(
            model.intercept_
        ).size
    )


# =============================================================================
# Inner blocked cross-fitting for Ridge residual targets
# =============================================================================

def blocked_fold_ids(
    mission_labels,
    n_folds: int,
):
    labels = np.asarray(
        mission_labels,
        dtype=object,
    )

    fold_ids = np.full(
        len(labels),
        -1,
        dtype=np.int64,
    )

    for mission in np.unique(
        labels
    ):
        idx = np.flatnonzero(
            labels == mission
        )

        chunks = np.array_split(
            idx,
            n_folds,
        )

        for k, chunk in enumerate(
            chunks
        ):
            fold_ids[
                chunk
            ] = k

    if np.any(
        fold_ids < 0
    ):
        raise RuntimeError(
            "Failed to assign all cross-fit fold IDs."
        )

    return fold_ids


def crossfit_ridge_predictions(
    X_train,
    y_train,
    mission_labels,
    n_folds=INNER_CROSSFIT_FOLDS,
):
    fold_ids = blocked_fold_ids(
        mission_labels,
        n_folds,
    )

    pred = np.full_like(
        y_train,
        np.nan,
        dtype=np.float32,
    )

    for k in range(
        n_folds
    ):
        val_mask = (
            fold_ids == k
        )

        train_mask = (
            ~val_mask
        )

        if not np.any(
            val_mask
        ):
            continue

        model, scaler = fit_ridge(
            X_train[
                train_mask
            ],
            y_train[
                train_mask
            ],
        )

        pred[
            val_mask
        ] = ridge_predict(
            model,
            scaler,
            X_train[
                val_mask
            ],
        )

    if not np.isfinite(
        pred
    ).all():
        raise RuntimeError(
            "Nonfinite Ridge OOF predictions."
        )

    return pred


# =============================================================================
# Metrics
# =============================================================================

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


def wd_from_uv(
    uv,
):
    uv = np.asarray(
        uv,
        dtype=np.float64,
    )

    return (
        np.degrees(
            np.arctan2(
                -uv[:, 0],
                -uv[:, 1],
            )
        )
        % 360.0
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

    ws_t = np.linalg.norm(
        yt,
        axis=1,
    )

    ws_p = np.linalg.norm(
        yp,
        axis=1,
    )

    wd_t = wd_from_uv(
        yt
    )

    wd_p = wd_from_uv(
        yp
    )

    wd_err = circular_diff_deg(
        wd_p,
        wd_t,
    )

    return {
        "wind_U_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    err[:, 0] ** 2
                )
            )
        ),
        "wind_V_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    err[:, 1] ** 2
                )
            )
        ),
        "wind_vector_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        err ** 2,
                        axis=1,
                    )
                )
            )
        ),
        "wind_speed_RMSE_mps": float(
            np.sqrt(
                np.mean(
                    (
                        ws_p
                        - ws_t
                    ) ** 2
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


def selection_score(
    metrics,
    base_metrics,
):
    ratios = {
        "U_ratio": (
            metrics[
                "wind_U_RMSE_mps"
            ]
            / max(
                base_metrics[
                    "wind_U_RMSE_mps"
                ],
                EPS,
            )
        ),
        "V_ratio": (
            metrics[
                "wind_V_RMSE_mps"
            ]
            / max(
                base_metrics[
                    "wind_V_RMSE_mps"
                ],
                EPS,
            )
        ),
        "WS_ratio": (
            metrics[
                "wind_speed_RMSE_mps"
            ]
            / max(
                base_metrics[
                    "wind_speed_RMSE_mps"
                ],
                EPS,
            )
        ),
        "WD_ratio": (
            metrics[
                "wind_direction_RMSE_deg"
            ]
            / max(
                base_metrics[
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
            "U_ratio"
        ]
        + SCORE_WEIGHTS[
            "V"
        ]
        * ratios[
            "V_ratio"
        ]
        + SCORE_WEIGHTS[
            "WS"
        ]
        * ratios[
            "WS_ratio"
        ]
        + SCORE_WEIGHTS[
            "WD"
        ]
        * ratios[
            "WD_ratio"
        ]
    )

    return (
        float(score),
        ratios,
    )


def corrcoef_safe(
    a,
    b,
):
    a = np.asarray(
        a,
        dtype=np.float64,
    ).reshape(-1)

    b = np.asarray(
        b,
        dtype=np.float64,
    ).reshape(-1)

    if (
        len(a) < 3
        or np.std(a) < EPS
        or np.std(b) < EPS
    ):
        return np.nan

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def residual_diagnostics(
    y_true,
    ridge_pred,
    final_pred,
):
    r_true = (
        np.asarray(
            y_true,
            dtype=np.float64,
        )
        - np.asarray(
            ridge_pred,
            dtype=np.float64,
        )
    )

    r_pred = (
        np.asarray(
            final_pred,
            dtype=np.float64,
        )
        - np.asarray(
            ridge_pred,
            dtype=np.float64,
        )
    )

    return {
        "residual_corr_U": (
            corrcoef_safe(
                r_true[:, 0],
                r_pred[:, 0],
            )
        ),
        "residual_corr_V": (
            corrcoef_safe(
                r_true[:, 1],
                r_pred[:, 1],
            )
        ),
        "residual_sign_accuracy_U": float(
            np.mean(
                np.sign(
                    r_true[:, 0]
                )
                == np.sign(
                    r_pred[:, 0]
                )
            )
        ),
        "residual_sign_accuracy_V": float(
            np.mean(
                np.sign(
                    r_true[:, 1]
                )
                == np.sign(
                    r_pred[:, 1]
                )
            )
        ),
        "true_residual_RMS_U_mps": float(
            np.sqrt(
                np.mean(
                    r_true[:, 0] ** 2
                )
            )
        ),
        "true_residual_RMS_V_mps": float(
            np.sqrt(
                np.mean(
                    r_true[:, 1] ** 2
                )
            )
        ),
        "pred_correction_RMS_U_mps": float(
            np.sqrt(
                np.mean(
                    r_pred[:, 0] ** 2
                )
            )
        ),
        "pred_correction_RMS_V_mps": float(
            np.sqrt(
                np.mean(
                    r_pred[:, 1] ** 2
                )
            )
        ),
    }


# =============================================================================
# Residual neural data / models
# =============================================================================

def prepare_residual_data(
    X_train,
    X_val,
    y_train,
    ridge_train_oof,
    ridge_val,
):
    scaler = fit_scaler(
        X_train,
        y_train,
    )

    Xtr_z = transform_X(
        X_train,
        scaler,
    )

    Xva_z = transform_X(
        X_val,
        scaler,
    )

    anchor_mean = np.mean(
        ridge_train_oof,
        axis=0,
    )

    anchor_std = np.std(
        ridge_train_oof,
        axis=0,
        ddof=0,
    )

    anchor_std = np.where(
        anchor_std < 1e-8,
        1.0,
        anchor_std,
    )

    anchor_train_z = (
        (
            ridge_train_oof
            - anchor_mean[
                None,
                :,
            ]
        )
        / anchor_std[
            None,
            :,
        ]
    ).astype(
        np.float32
    )

    anchor_val_z = (
        (
            ridge_val
            - anchor_mean[
                None,
                :,
            ]
        )
        / anchor_std[
            None,
            :,
        ]
    ).astype(
        np.float32
    )

    residual = (
        y_train
        - ridge_train_oof
    ).astype(
        np.float32
    )

    residual_std = np.std(
        residual.astype(
            np.float64
        ),
        axis=0,
        ddof=0,
    )

    residual_std = np.where(
        residual_std < 1e-4,
        1.0,
        residual_std,
    ).astype(
        np.float32
    )

    target_norm = (
        residual
        / residual_std[
            None,
            :,
        ]
    ).astype(
        np.float32
    )

    return {
        "X_train_z": Xtr_z,
        "X_val_z": Xva_z,
        "anchor_train_z": (
            anchor_train_z
        ),
        "anchor_val_z": (
            anchor_val_z
        ),
        "target_norm": (
            target_norm
        ),
        "residual_std": (
            residual_std
        ),
    }


def build_residual_model_class(
    torch,
    nn,
    mode: str,
):
    input_dim = len(
        TURB_FULL_FEATURES
    )

    if mode == "mlp":
        class ResidualMLP(nn.Module):
            def __init__(self):
                super().__init__()

                self.backbone = nn.Sequential(
                    nn.Linear(
                        LOOKBACK_STEPS
                        * input_dim
                        + 2,
                        NEURAL_FREEZE.mlp_hidden_1,
                    ),
                    nn.LayerNorm(
                        NEURAL_FREEZE.mlp_hidden_1
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        NEURAL_FREEZE.dropout
                    ),
                    nn.Linear(
                        NEURAL_FREEZE.mlp_hidden_1,
                        NEURAL_FREEZE.mlp_hidden_2,
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        NEURAL_FREEZE.dropout
                    ),
                )

                self.out = nn.Linear(
                    NEURAL_FREEZE.mlp_hidden_2,
                    2,
                )

                nn.init.zeros_(
                    self.out.weight
                )
                nn.init.zeros_(
                    self.out.bias
                )

            def forward(
                self,
                x,
                anchor,
            ):
                z = torch.cat(
                    [
                        x.flatten(
                            start_dim=1
                        ),
                        anchor,
                    ],
                    dim=1,
                )

                return self.out(
                    self.backbone(
                        z
                    )
                )

        return ResidualMLP

    if mode == "gru":
        class ResidualGRU(nn.Module):
            def __init__(self):
                super().__init__()

                self.gru = nn.GRU(
                    input_size=input_dim,
                    hidden_size=(
                        NEURAL_FREEZE.gru_hidden
                    ),
                    num_layers=1,
                    batch_first=True,
                )

                self.backbone = nn.Sequential(
                    nn.Linear(
                        NEURAL_FREEZE.gru_hidden
                        + 2,
                        NEURAL_FREEZE.dense_hidden,
                    ),
                    nn.LayerNorm(
                        NEURAL_FREEZE.dense_hidden
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        NEURAL_FREEZE.dropout
                    ),
                )

                self.out = nn.Linear(
                    NEURAL_FREEZE.dense_hidden,
                    2,
                )

                nn.init.zeros_(
                    self.out.weight
                )
                nn.init.zeros_(
                    self.out.bias
                )

            def forward(
                self,
                x,
                anchor,
            ):
                seq, _ = self.gru(
                    x
                )

                h = seq[
                    :,
                    -1,
                    :,
                ]

                z = torch.cat(
                    [
                        h,
                        anchor,
                    ],
                    dim=1,
                )

                return self.out(
                    self.backbone(
                        z
                    )
                )

        return ResidualGRU

    raise ValueError(
        f"Unknown residual model mode: {mode}"
    )


def evaluate_neural_residual(
    *,
    torch,
    model,
    residual_data,
    y_val,
    ridge_val,
    device,
    amp_enabled,
):
    model.eval()

    chunks = []

    with torch.no_grad():
        for xb_np, ab_np in sequential_batches(
            residual_data[
                "X_val_z"
            ],
            residual_data[
                "anchor_val_z"
            ],
            batch_size=(
                NEURAL_FREEZE.batch_size
            ),
        ):
            xb = torch.from_numpy(
                xb_np
            ).to(
                device,
                non_blocking=True,
            )

            ab = torch.from_numpy(
                ab_np
            ).to(
                device,
                non_blocking=True,
            )

            with amp_context(
                torch,
                amp_enabled,
                device,
            ):
                pred_norm = model(
                    xb,
                    ab,
                )

            chunks.append(
                pred_norm.detach()
                .float()
                .cpu()
                .numpy()
            )

    pred_norm = np.concatenate(
        chunks,
        axis=0,
    )

    correction = (
        pred_norm
        * residual_data[
            "residual_std"
        ][
            None,
            :,
        ]
    )

    pred = (
        ridge_val
        + correction
    ).astype(
        np.float32
    )

    metrics = evaluate_wind(
        y_val,
        pred,
    )

    metrics.update(
        residual_diagnostics(
            y_val,
            ridge_val,
            pred,
        )
    )

    return (
        metrics,
        pred,
    )


def train_neural_residual_run(
    *,
    torch,
    nn,
    candidate,
    mode,
    seed,
    held_out,
    residual_data,
    y_val,
    ridge_val,
    full_ridge_metrics,
    base4_metrics,
    output_dir,
    device,
    debug_fast,
):
    seed_everything(
        torch,
        seed,
    )

    ModelClass = (
        build_residual_model_class(
            torch,
            nn,
            mode,
        )
    )

    model = ModelClass().to(
        device
    )

    param_count = count_parameters(
        model
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=(
            NEURAL_FREEZE.learning_rate
        ),
        weight_decay=(
            NEURAL_FREEZE.weight_decay
        ),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=(
            NEURAL_FREEZE.scheduler_factor
        ),
        patience=(
            NEURAL_FREEZE.scheduler_patience
        ),
        min_lr=(
            NEURAL_FREEZE.min_learning_rate
        ),
    )

    amp_enabled = bool(
        NEURAL_FREEZE.use_amp
        and device.type == "cuda"
    )

    grad_scaler = create_grad_scaler(
        torch,
        amp_enabled,
    )

    # Epoch 0 = exact full Ridge anchor.
    initial_metrics, _ = (
        evaluate_neural_residual(
            torch=torch,
            model=model,
            residual_data=(
                residual_data
            ),
            y_val=y_val,
            ridge_val=(
                ridge_val
            ),
            device=device,
            amp_enabled=(
                amp_enabled
            ),
        )
    )

    best_anchor_score, _ = selection_score(
        initial_metrics,
        full_ridge_metrics,
    )

    best_epoch = 0
    best_state = state_dict_cpu(
        model
    )
    best_metrics = dict(
        initial_metrics
    )

    history = [
        {
            "candidate": candidate,
            "held_out_mission": (
                held_out
            ),
            "seed": int(seed),
            "epoch_one_based": 0,
            "train_total_loss": np.nan,
            "anchor_score": (
                best_anchor_score
            ),
            "checkpoint_improved": True,
            **{
                f"val_{k}": v
                for k, v
                in initial_metrics.items()
            },
        }
    ]

    patience = 0

    max_epochs = (
        8
        if debug_fast
        else NEURAL_FREEZE.max_epochs
    )

    patience_limit = (
        4
        if debug_fast
        else NEURAL_FREEZE.early_stop_patience
    )

    started = time.perf_counter()

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        model.train()

        loss_sum = 0.0
        n_seen = 0

        for (
            xb_np,
            ab_np,
            tb_np,
        ) in sequential_batches(
            residual_data[
                "X_train_z"
            ],
            residual_data[
                "anchor_train_z"
            ],
            residual_data[
                "target_norm"
            ],
            batch_size=(
                NEURAL_FREEZE.batch_size
            ),
        ):
            xb = torch.from_numpy(
                xb_np
            ).to(
                device,
                non_blocking=True,
            )

            ab = torch.from_numpy(
                ab_np
            ).to(
                device,
                non_blocking=True,
            )

            tb = torch.from_numpy(
                tb_np
            ).to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with amp_context(
                torch,
                amp_enabled,
                device,
            ):
                pred_norm = model(
                    xb,
                    ab,
                )

                mse = torch.mean(
                    (
                        pred_norm
                        - tb
                    ) ** 2
                )

                penalty = torch.mean(
                    pred_norm ** 2
                )

                loss = (
                    mse
                    + NEURAL_FREEZE.correction_penalty
                    * penalty
                )

            if not torch.isfinite(
                loss
            ):
                raise FloatingPointError(
                    "Nonfinite Stage12U neural loss."
                )

            if grad_scaler is not None:
                grad_scaler.scale(
                    loss
                ).backward()

                grad_scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    NEURAL_FREEZE.grad_clip_norm,
                )

                grad_scaler.step(
                    optimizer
                )

                grad_scaler.update()
            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    NEURAL_FREEZE.grad_clip_norm,
                )

                optimizer.step()

            bn = len(
                xb_np
            )

            loss_sum += float(
                loss.detach()
                .float()
                .cpu()
                .item()
            ) * bn

            n_seen += bn

        val_metrics, _ = (
            evaluate_neural_residual(
                torch=torch,
                model=model,
                residual_data=(
                    residual_data
                ),
                y_val=y_val,
                ridge_val=(
                    ridge_val
                ),
                device=device,
                amp_enabled=(
                    amp_enabled
                ),
            )
        )

        anchor_score, _ = selection_score(
            val_metrics,
            full_ridge_metrics,
        )

        improved = (
            anchor_score
            < best_anchor_score
            - 1e-8
        )

        if improved:
            best_anchor_score = float(
                anchor_score
            )
            best_epoch = int(
                epoch
            )
            best_state = state_dict_cpu(
                model
            )
            best_metrics = dict(
                val_metrics
            )
            patience = 0
        else:
            patience += 1

        scheduler.step(
            anchor_score
        )

        history.append(
            {
                "candidate": candidate,
                "held_out_mission": (
                    held_out
                ),
                "seed": int(seed),
                "epoch_one_based": int(
                    epoch
                ),
                "train_total_loss": (
                    loss_sum
                    / max(
                        n_seen,
                        1,
                    )
                ),
                "anchor_score": float(
                    anchor_score
                ),
                "checkpoint_improved": bool(
                    improved
                ),
                **{
                    f"val_{k}": v
                    for k, v
                    in val_metrics.items()
                },
            }
        )

        log(
            f"      ep={epoch:03d} | "
            f"anchorScore={anchor_score:.5f} | "
            f"U={val_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={val_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={val_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={val_metrics['wind_direction_RMSE_deg']:.3f} | "
            f"corr=({val_metrics['residual_corr_U']:+.3f},"
            f"{val_metrics['residual_corr_V']:+.3f})"
            + (
                " *"
                if improved
                else ""
            )
        )

        if patience >= patience_limit:
            break

    elapsed = (
        time.perf_counter()
        - started
    )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    confirmed_metrics, _ = (
        evaluate_neural_residual(
            torch=torch,
            model=model,
            residual_data=(
                residual_data
            ),
            y_val=y_val,
            ridge_val=(
                ridge_val
            ),
            device=device,
            amp_enabled=(
                amp_enabled
            ),
        )
    )

    final_score, score_parts = selection_score(
        confirmed_metrics,
        base4_metrics,
    )

    history_dir = (
        output_dir
        / "histories"
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints"
    )

    history_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tag = (
        f"{candidate}"
        f"__holdout_{held_out.replace(' ', '_')}"
        f"__seed_{seed}"
    )

    history_path = (
        history_dir
        / f"{tag}.csv"
    )

    checkpoint_path = (
        checkpoint_dir
        / f"{tag}.pt"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False,
        encoding="utf-8-sig",
    )

    torch.save(
        {
            "stage": "12U",
            "candidate": candidate,
            "held_out_mission": (
                held_out
            ),
            "seed": int(seed),
            "best_epoch": int(
                best_epoch
            ),
            "state_dict": (
                best_state
            ),
            "neural_parameter_count": int(
                param_count
            ),
            "score_vs_Base4": float(
                final_score
            ),
            "metrics": (
                confirmed_metrics
            ),
            "feature_names": (
                TURB_FULL_FEATURES
            ),
            "development_only": True,
            "Tropical_Atlantic_used": False,
        },
        checkpoint_path,
    )

    return {
        "candidate": candidate,
        "held_out_mission": (
            held_out
        ),
        "seed": int(seed),
        "status": (
            "completed_finite"
        ),
        "parameter_count": int(
            param_count
        ),
        "best_epoch": int(
            best_epoch
        ),
        "selection_score": float(
            final_score
        ),
        "elapsed_seconds": float(
            elapsed
        ),
        **confirmed_metrics,
        **score_parts,
        "history_path": str(
            history_path
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
    }


# =============================================================================
# HistGradientBoosting candidates
# =============================================================================

def make_hgb():
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=(
            HGB_FREEZE.learning_rate
        ),
        max_iter=(
            HGB_FREEZE.max_iter
        ),
        max_leaf_nodes=(
            HGB_FREEZE.max_leaf_nodes
        ),
        max_depth=(
            HGB_FREEZE.max_depth
        ),
        min_samples_leaf=(
            HGB_FREEZE.min_samples_leaf
        ),
        l2_regularization=(
            HGB_FREEZE.l2_regularization
        ),
        early_stopping=False,
        random_state=42,
    )


def hgb_leaf_count(
    model,
):
    total = 0

    predictors = getattr(
        model,
        "_predictors",
        None,
    )

    if predictors is None:
        return -1

    for stage in predictors:
        for predictor in stage:
            nodes = getattr(
                predictor,
                "nodes",
                None,
            )

            if nodes is None:
                continue

            names = getattr(
                nodes.dtype,
                "names",
                None,
            )

            if names is not None and "is_leaf" in names:
                total += int(
                    np.sum(
                        nodes[
                            "is_leaf"
                        ]
                    )
                )

    return int(total)


def fit_hgb_direct(
    X_train,
    y_train,
    X_val,
):
    Xtr = np.asarray(
        X_train,
        dtype=np.float32,
    ).reshape(
        len(X_train),
        -1,
    )

    Xva = np.asarray(
        X_val,
        dtype=np.float32,
    ).reshape(
        len(X_val),
        -1,
    )

    models = []

    pred = np.zeros(
        (
            len(X_val),
            2,
        ),
        dtype=np.float32,
    )

    for j in range(2):
        model = make_hgb()

        model.fit(
            Xtr,
            y_train[
                :,
                j
            ],
        )

        pred[
            :,
            j
        ] = model.predict(
            Xva
        ).astype(
            np.float32
        )

        models.append(
            model
        )

    leaf_count = sum(
        hgb_leaf_count(
            m
        )
        for m in models
    )

    return (
        pred,
        leaf_count,
    )


def fit_hgb_residual(
    X_train,
    ridge_train_oof,
    y_train,
    X_val,
    ridge_val,
):
    anchor_mean = np.mean(
        ridge_train_oof,
        axis=0,
    )

    anchor_std = np.std(
        ridge_train_oof,
        axis=0,
        ddof=0,
    )

    anchor_std = np.where(
        anchor_std < 1e-8,
        1.0,
        anchor_std,
    )

    Xtr_flat = np.asarray(
        X_train,
        dtype=np.float32,
    ).reshape(
        len(X_train),
        -1,
    )

    Xva_flat = np.asarray(
        X_val,
        dtype=np.float32,
    ).reshape(
        len(X_val),
        -1,
    )

    A_tr = (
        (
            ridge_train_oof
            - anchor_mean[
                None,
                :,
            ]
        )
        / anchor_std[
            None,
            :,
        ]
    ).astype(
        np.float32
    )

    A_va = (
        (
            ridge_val
            - anchor_mean[
                None,
                :,
            ]
        )
        / anchor_std[
            None,
            :,
        ]
    ).astype(
        np.float32
    )

    Xtr = np.concatenate(
        [
            Xtr_flat,
            A_tr,
        ],
        axis=1,
    )

    Xva = np.concatenate(
        [
            Xva_flat,
            A_va,
        ],
        axis=1,
    )

    residual = (
        y_train
        - ridge_train_oof
    )

    correction = np.zeros(
        (
            len(X_val),
            2,
        ),
        dtype=np.float32,
    )

    models = []

    for j in range(2):
        model = make_hgb()

        model.fit(
            Xtr,
            residual[
                :,
                j
            ],
        )

        correction[
            :,
            j
        ] = model.predict(
            Xva
        ).astype(
            np.float32
        )

        models.append(
            model
        )

    pred = (
        ridge_val
        + correction
    ).astype(
        np.float32
    )

    leaf_count = sum(
        hgb_leaf_count(
            m
        )
        for m in models
    )

    return (
        pred,
        leaf_count,
    )


# =============================================================================
# Result helpers
# =============================================================================

def append_csv(
    path: Path,
    row: dict,
):
    new = pd.DataFrame(
        [row]
    )

    if path.exists():
        old = pd.read_csv(
            path
        )

        keys = [
            "candidate",
            "held_out_mission",
            "seed",
        ]

        if all(
            k in old.columns
            for k in keys
        ):
            mask = (
                (
                    old[
                        "candidate"
                    ].astype(
                        str
                    )
                    == str(
                        row[
                            "candidate"
                        ]
                    )
                )
                & (
                    old[
                        "held_out_mission"
                    ].astype(
                        str
                    )
                    == str(
                        row[
                            "held_out_mission"
                        ]
                    )
                )
                & (
                    old[
                        "seed"
                    ].astype(
                        int
                    )
                    == int(
                        row[
                            "seed"
                        ]
                    )
                )
            )

            old = old.loc[
                ~mask
            ].copy()

        new = pd.concat(
            [
                old,
                new,
            ],
            ignore_index=True,
        )

    new.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def summarize_candidate(
    runs,
    candidate,
):
    sub = runs.loc[
        (
            runs[
                "candidate"
            ].astype(
                str
            )
            == candidate
        )
        & (
            runs[
                "status"
            ].astype(
                str
            )
            == "completed_finite"
        )
    ].copy()

    if sub.empty:
        raise RuntimeError(
            f"No rows for {candidate}"
        )

    row = {
        "candidate": candidate,
        "runs": int(
            len(sub)
        ),
        "mission_count": int(
            sub[
                "held_out_mission"
            ].nunique()
        ),
        "selection_score_mean": float(
            sub[
                "selection_score"
            ].mean()
        ),
        "selection_score_std": float(
            sub[
                "selection_score"
            ].std(
                ddof=0
            )
        ),
        "parameter_count_mean": float(
            sub[
                "parameter_count"
            ].mean()
        ),
    }

    for col in [
        "wind_U_RMSE_mps",
        "wind_V_RMSE_mps",
        "wind_vector_RMSE_mps",
        "wind_speed_RMSE_mps",
        "wind_direction_RMSE_deg",
        "residual_corr_U",
        "residual_corr_V",
    ]:
        if col not in sub.columns:
            continue

        vals = sub[
            col
        ].to_numpy(
            dtype=float
        )

        row[
            f"{col}_mean"
        ] = float(
            np.nanmean(
                vals
            )
        )

        row[
            f"{col}_std"
        ] = float(
            np.nanstd(
                vals,
                ddof=0,
            )
        )

    mission_scores = (
        sub.groupby(
            "held_out_mission"
        )[
            "selection_score"
        ]
        .mean()
    )

    row[
        "mission_wins_vs_Base4"
    ] = int(
        np.sum(
            mission_scores.to_numpy()
            < 1.0
        )
    )

    return row


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        type=Path,
        default=(
            DEFAULT_PROJECT_ROOT
        ),
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=(
            DEFAULT_RAW_DIR
        ),
    )

    parser.add_argument(
        "--stage12b-dir",
        type=Path,
        default=(
            DEFAULT_STAGE12B_DIR
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
    )

    parser.add_argument(
        "--debug-fast",
        action="store_true",
        help=(
            "Smoke test: build/reuse dataset, Antarctic holdout only, "
            "one residual-GRU seed, HGB skipped."
        ),
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    dataset_dir = (
        output_dir
        / "enhanced_dataset"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        args.rebuild_dataset
        or not enhanced_dataset_complete(
            dataset_dir
        )
    ):
        build_enhanced_dataset(
            project_root=(
                args.project_root
            ),
            raw_dir=(
                args.raw_dir
            ),
            stage12b_dir=(
                args.stage12b_dir
            ),
            dataset_dir=(
                dataset_dir
            ),
            progress_every=(
                args.progress_every
            ),
        )
    else:
        log(
            f"[DATASET] Reusing cached enhanced dataset: {dataset_dir}"
        )

    missions = {
        m: load_enhanced_mission(
            dataset_dir,
            m,
        )
        for m in DEVELOPMENT_MISSIONS
    }

    torch, nn = import_torch()
    configure_cuda(
        torch
    )

    if (
        args.force_cpu
        or not torch.cuda.is_available()
    ):
        device = torch.device(
            "cpu"
        )
    else:
        device = torch.device(
            "cuda"
        )

    active_holdouts = (
        ["Antarctic"]
        if args.debug_fast
        else DEVELOPMENT_MISSIONS
    )

    active_seeds = (
        [
            SCREENING_SEEDS[
                0
            ]
        ]
        if args.debug_fast
        else SCREENING_SEEDS
    )

    run_path = (
        output_dir
        / "candidate_run_results.csv"
    )

    residual_diag_rows = []

    log("")
    log("=" * 132)
    log(
        "12U — MULTISCALE TURBULENCE-AWARE DIRECT U/V FORECASTING"
    )
    log("=" * 132)
    log(
        "protocol : TRUE Stage12B 10-min block means, 6 blocks -> next block"
    )
    log(
        "extra    : causal 1-min intra-block turbulence descriptors"
    )
    log(
        f"device   : {device}"
    )
    log(
        "[FIREWALL] Development missions only. Tropical Atlantic is not loaded."
    )
    log("")

    for held_out in active_holdouts:
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

        ytr = train[
            "y"
        ]

        yva = val[
            "y"
        ]

        log("-" * 132)
        log(
            f"[{held_out}] "
            f"Ntrain={len(ytr):,} | "
            f"Nval={len(yva):,}"
        )

        ridge_results = {}

        for group_name in [
            "Base4",
            "TurbCore10",
            "TurbFull16",
        ]:
            Xtr = select_group(
                train[
                    "X"
                ],
                group_name,
            )

            Xva = select_group(
                val[
                    "X"
                ],
                group_name,
            )

            model, scaler = fit_ridge(
                Xtr,
                ytr,
            )

            pred = ridge_predict(
                model,
                scaler,
                Xva,
            )

            metrics = evaluate_wind(
                yva,
                pred,
            )

            ridge_results[
                group_name
            ] = {
                "model": model,
                "scaler": scaler,
                "pred": pred,
                "metrics": metrics,
                "Xtr": Xtr,
                "Xva": Xva,
            }

        base_metrics = ridge_results[
            "Base4"
        ][
            "metrics"
        ]

        for group_name in [
            "Base4",
            "TurbCore10",
            "TurbFull16",
        ]:
            info = ridge_results[
                group_name
            ]

            score, parts = selection_score(
                info[
                    "metrics"
                ],
                base_metrics,
            )

            candidate = (
                f"{group_name}-Ridge"
            )

            row = {
                "candidate": candidate,
                "held_out_mission": (
                    held_out
                ),
                "seed": -1,
                "status": (
                    "completed_finite"
                ),
                "parameter_count": int(
                    ridge_param_count(
                        info[
                            "model"
                        ]
                    )
                ),
                "selection_score": float(
                    score
                ),
                "best_epoch": 0,
                "elapsed_seconds": 0.0,
                **info[
                    "metrics"
                ],
                **parts,
                "residual_corr_U": np.nan,
                "residual_corr_V": np.nan,
            }

            append_csv(
                run_path,
                row,
            )

            log(
                f"  {candidate:31s} | "
                f"score={score:.5f} | "
                f"U={info['metrics']['wind_U_RMSE_mps']:.4f} | "
                f"V={info['metrics']['wind_V_RMSE_mps']:.4f} | "
                f"WS={info['metrics']['wind_speed_RMSE_mps']:.4f} | "
                f"WD={info['metrics']['wind_direction_RMSE_deg']:.3f}"
            )

        # -------------------------------------------------------------
        # Full-feature cross-fitted Ridge residual targets
        # -------------------------------------------------------------
        Xtr_full = ridge_results[
            "TurbFull16"
        ][
            "Xtr"
        ]

        Xva_full = ridge_results[
            "TurbFull16"
        ][
            "Xva"
        ]

        ridge_full_val = ridge_results[
            "TurbFull16"
        ][
            "pred"
        ]

        ridge_full_metrics = ridge_results[
            "TurbFull16"
        ][
            "metrics"
        ]

        ridge_train_oof = (
            crossfit_ridge_predictions(
                Xtr_full,
                ytr,
                train[
                    "mission_labels"
                ],
            )
        )

        oof_residual = (
            ytr
            - ridge_train_oof
        )

        residual_diag_rows.append(
            {
                "held_out_mission": (
                    held_out
                ),
                "OOF_residual_RMS_U_mps": float(
                    np.sqrt(
                        np.mean(
                            oof_residual[
                                :,
                                0
                            ] ** 2
                        )
                    )
                ),
                "OOF_residual_RMS_V_mps": float(
                    np.sqrt(
                        np.mean(
                            oof_residual[
                                :,
                                1
                            ] ** 2
                        )
                    )
                ),
                "val_fullRidge_U_RMSE_mps": (
                    ridge_full_metrics[
                        "wind_U_RMSE_mps"
                    ]
                ),
                "val_fullRidge_V_RMSE_mps": (
                    ridge_full_metrics[
                        "wind_V_RMSE_mps"
                    ]
                ),
            }
        )

        residual_data = prepare_residual_data(
            Xtr_full,
            Xva_full,
            ytr,
            ridge_train_oof,
            ridge_full_val,
        )

        # -------------------------------------------------------------
        # Neural safe residual models
        # -------------------------------------------------------------
        neural_specs = (
            {
                "TurbFull16-RidgeResidualGRU": "gru"
            }
            if args.debug_fast
            else NEURAL_CANDIDATES
        )

        for candidate, mode in neural_specs.items():
            for seed in active_seeds:
                log(
                    f"  [TRAIN] {candidate} | seed={seed}"
                )

                result = train_neural_residual_run(
                    torch=torch,
                    nn=nn,
                    candidate=(
                        candidate
                    ),
                    mode=mode,
                    seed=seed,
                    held_out=(
                        held_out
                    ),
                    residual_data=(
                        residual_data
                    ),
                    y_val=yva,
                    ridge_val=(
                        ridge_full_val
                    ),
                    full_ridge_metrics=(
                        ridge_full_metrics
                    ),
                    base4_metrics=(
                        base_metrics
                    ),
                    output_dir=(
                        output_dir
                    ),
                    device=device,
                    debug_fast=(
                        args.debug_fast
                    ),
                )

                result[
                    "parameter_count"
                ] += int(
                    ridge_param_count(
                        ridge_results[
                            "TurbFull16"
                        ][
                            "model"
                        ]
                    )
                )

                append_csv(
                    run_path,
                    result,
                )

                log(
                    f"  [DONE] "
                    f"score={result['selection_score']:.5f} | "
                    f"U={result['wind_U_RMSE_mps']:.4f} | "
                    f"V={result['wind_V_RMSE_mps']:.4f} | "
                    f"WS={result['wind_speed_RMSE_mps']:.4f} | "
                    f"WD={result['wind_direction_RMSE_deg']:.3f}"
                )

        # -------------------------------------------------------------
        # HistGradientBoosting candidates
        # -------------------------------------------------------------
        if not args.debug_fast:
            started = (
                time.perf_counter()
            )

            direct_pred, direct_leaves = (
                fit_hgb_direct(
                    Xtr_full,
                    ytr,
                    Xva_full,
                )
            )

            direct_metrics = evaluate_wind(
                yva,
                direct_pred,
            )

            direct_score, direct_parts = (
                selection_score(
                    direct_metrics,
                    base_metrics,
                )
            )

            direct_row = {
                "candidate": (
                    "TurbFull16-HGBR-Direct"
                ),
                "held_out_mission": (
                    held_out
                ),
                "seed": -1,
                "status": (
                    "completed_finite"
                ),
                "parameter_count": int(
                    direct_leaves
                ),
                "selection_score": float(
                    direct_score
                ),
                "best_epoch": int(
                    HGB_FREEZE.max_iter
                ),
                "elapsed_seconds": float(
                    time.perf_counter()
                    - started
                ),
                **direct_metrics,
                **direct_parts,
                "residual_corr_U": np.nan,
                "residual_corr_V": np.nan,
            }

            append_csv(
                run_path,
                direct_row,
            )

            log(
                "  TurbFull16-HGBR-Direct        | "
                f"score={direct_score:.5f} | "
                f"U={direct_metrics['wind_U_RMSE_mps']:.4f} | "
                f"V={direct_metrics['wind_V_RMSE_mps']:.4f} | "
                f"WS={direct_metrics['wind_speed_RMSE_mps']:.4f} | "
                f"WD={direct_metrics['wind_direction_RMSE_deg']:.3f}"
            )

            started = (
                time.perf_counter()
            )

            residual_pred, residual_leaves = (
                fit_hgb_residual(
                    Xtr_full,
                    ridge_train_oof,
                    ytr,
                    Xva_full,
                    ridge_full_val,
                )
            )

            residual_metrics = evaluate_wind(
                yva,
                residual_pred,
            )

            residual_metrics.update(
                residual_diagnostics(
                    yva,
                    ridge_full_val,
                    residual_pred,
                )
            )

            residual_score, residual_parts = (
                selection_score(
                    residual_metrics,
                    base_metrics,
                )
            )

            residual_row = {
                "candidate": (
                    "TurbFull16-RidgeResidual-HGBR"
                ),
                "held_out_mission": (
                    held_out
                ),
                "seed": -1,
                "status": (
                    "completed_finite"
                ),
                "parameter_count": int(
                    residual_leaves
                    + ridge_param_count(
                        ridge_results[
                            "TurbFull16"
                        ][
                            "model"
                        ]
                    )
                ),
                "selection_score": float(
                    residual_score
                ),
                "best_epoch": int(
                    HGB_FREEZE.max_iter
                ),
                "elapsed_seconds": float(
                    time.perf_counter()
                    - started
                ),
                **residual_metrics,
                **residual_parts,
            }

            append_csv(
                run_path,
                residual_row,
            )

            log(
                "  TurbFull16-RidgeResidual-HGBR | "
                f"score={residual_score:.5f} | "
                f"U={residual_metrics['wind_U_RMSE_mps']:.4f} | "
                f"V={residual_metrics['wind_V_RMSE_mps']:.4f} | "
                f"WS={residual_metrics['wind_speed_RMSE_mps']:.4f} | "
                f"WD={residual_metrics['wind_direction_RMSE_deg']:.3f} | "
                f"corr=({residual_metrics['residual_corr_U']:+.3f},"
                f"{residual_metrics['residual_corr_V']:+.3f})"
            )

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    residual_diag_df = pd.DataFrame(
        residual_diag_rows
    )

    residual_diag_path = (
        output_dir
        / "residual_diagnostics.csv"
    )

    residual_diag_df.to_csv(
        residual_diag_path,
        index=False,
        encoding="utf-8-sig",
    )

    if args.debug_fast:
        log("")
        log(
            "[DONE] debug-fast completed. No scientific final selection written."
        )
        return 0

    # =================================================================
    # Summary
    # =================================================================
    runs = pd.read_csv(
        run_path
    )

    candidate_order = [
        "Base4-Ridge",
        "TurbCore10-Ridge",
        "TurbFull16-Ridge",
        "TurbFull16-RidgeResidualMLP",
        "TurbFull16-RidgeResidualGRU",
        "TurbFull16-HGBR-Direct",
        "TurbFull16-RidgeResidual-HGBR",
    ]

    expected = {
        "Base4-Ridge": 3,
        "TurbCore10-Ridge": 3,
        "TurbFull16-Ridge": 3,
        "TurbFull16-RidgeResidualMLP": 9,
        "TurbFull16-RidgeResidualGRU": 9,
        "TurbFull16-HGBR-Direct": 3,
        "TurbFull16-RidgeResidual-HGBR": 3,
    }

    for candidate, n_expected in expected.items():
        got = int(
            np.sum(
                (
                    runs[
                        "candidate"
                    ].astype(
                        str
                    )
                    == candidate
                )
                & (
                    runs[
                        "status"
                    ].astype(
                        str
                    )
                    == "completed_finite"
                )
            )
        )

        if got != n_expected:
            raise RuntimeError(
                f"{candidate}: expected {n_expected} completed rows, got {got}."
            )

    summary = pd.DataFrame(
        [
            summarize_candidate(
                runs,
                candidate,
            )
            for candidate in candidate_order
        ]
    )

    base_runs = runs.loc[
        runs[
            "candidate"
        ].astype(str)
        == "Base4-Ridge"
    ].copy()

    base_means = {
        metric: float(
            base_runs[
                metric
            ].mean()
        )
        for metric in [
            "wind_U_RMSE_mps",
            "wind_V_RMSE_mps",
            "wind_vector_RMSE_mps",
            "wind_speed_RMSE_mps",
            "wind_direction_RMSE_deg",
        ]
    }

    for idx in summary.index:
        candidate = summary.loc[
            idx,
            "candidate",
        ]

        sub = runs.loc[
            runs[
                "candidate"
            ].astype(str)
            == candidate
        ]

        for metric, base_value in base_means.items():
            candidate_value = float(
                sub[
                    metric
                ].mean()
            )

            summary.loc[
                idx,
                f"{metric}_improvement_vs_Base4_fraction",
            ] = (
                base_value
                - candidate_value
            ) / max(
                base_value,
                EPS,
            )

    summary = summary.sort_values(
        [
            "selection_score_mean",
            "parameter_count_mean",
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
        / "candidate_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    ridge_summary = summary.loc[
        summary[
            "candidate"
        ].isin(
            [
                "Base4-Ridge",
                "TurbCore10-Ridge",
                "TurbFull16-Ridge",
            ]
        )
    ].copy()

    ridge_summary_path = (
        output_dir
        / "ridge_feature_group_summary.csv"
    )

    ridge_summary.to_csv(
        ridge_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    raw_best = summary.iloc[
        0
    ]

    raw_best_name = str(
        raw_best[
            "candidate"
        ]
    )

    raw_best_score = float(
        raw_best[
            "selection_score_mean"
        ]
    )

    raw_best_wins = int(
        raw_best[
            "mission_wins_vs_Base4"
        ]
    )

    vector_imp = float(
        raw_best[
            "wind_vector_RMSE_mps_improvement_vs_Base4_fraction"
        ]
    )

    u_imp = float(
        raw_best[
            "wind_U_RMSE_mps_improvement_vs_Base4_fraction"
        ]
    )

    v_imp = float(
        raw_best[
            "wind_V_RMSE_mps_improvement_vs_Base4_fraction"
        ]
    )

    meaningful = bool(
        raw_best_name
        != "Base4-Ridge"
        and raw_best_score
        < CONTINUE_SCORE_THRESHOLD
        and raw_best_wins
        >= MIN_MISSION_WINS
        and vector_imp
        > MIN_VECTOR_IMPROVEMENT
        and (
            u_imp
            > MIN_U_OR_V_IMPROVEMENT
            or v_imp
            > MIN_U_OR_V_IMPROVEMENT
        )
    )

    if meaningful:
        selected = raw_best_name
        decision = (
            "CONTINUE_MULTISCALE_TURBULENCE_AWARE_ROUTE"
        )
    else:
        selected = (
            "Base4-Ridge"
        )
        decision = (
            "MULTISCALE_ROUTE_NOT_YET_STRONG_ENOUGH"
        )

    selected_runs = runs.loc[
        runs[
            "candidate"
        ].astype(str)
        == selected
    ].copy()

    selected_runs_path = (
        output_dir
        / "selected_candidate_lomo_runs.csv"
    )

    selected_runs.to_csv(
        selected_runs_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Numeric comparison with published BP reference, clearly not used in selection.
    numeric_bp = {
        "best_U_minus_BP": float(
            raw_best[
                "wind_U_RMSE_mps_mean"
            ]
            - BP_STGNN_REFERENCE[
                "U_RMSE_mps"
            ]
        ),
        "best_V_minus_BP": float(
            raw_best[
                "wind_V_RMSE_mps_mean"
            ]
            - BP_STGNN_REFERENCE[
                "V_RMSE_mps"
            ]
        ),
        "best_WS_minus_BP": float(
            raw_best[
                "wind_speed_RMSE_mps_mean"
            ]
            - BP_STGNN_REFERENCE[
                "WS_RMSE_mps"
            ]
        ),
        "best_WD_minus_BP": float(
            raw_best[
                "wind_direction_RMSE_deg_mean"
            ]
            - BP_STGNN_REFERENCE[
                "WD_RMSE_deg"
            ]
        ),
    }

    payload = {
        "stage": "12U",
        "script_version": SCRIPT_VERSION,
        "task": {
            "protocol": (
                "Stage12B exact development sample timestamps"
            ),
            "10min_node_definition": (
                "complete 10 x 1-min block arithmetic mean for U,V,T,RH"
            ),
            "lookback_steps": LOOKBACK_STEPS,
            "forecast_minutes": FORECAST_MINUTES,
        },
        "feature_groups": (
            FEATURE_GROUPS
        ),
        "models": (
            candidate_order
        ),
        "neural_freeze": asdict(
            NEURAL_FREEZE
        ),
        "hgb_freeze": asdict(
            HGB_FREEZE
        ),
        "residual_training": (
            "5-fold blocked mission-preserving OOF Ridge residual targets"
        ),
        "raw_best_candidate": (
            raw_best_name
        ),
        "selected_candidate": (
            selected
        ),
        "decision": (
            decision
        ),
        "meaningful_improvement": (
            meaningful
        ),
        "numeric_BP_reference_difference_context_only": (
            numeric_bp
        ),
        "development_only": True,
        "Tropical_Atlantic_used": False,
    }

    selected_json_path = (
        output_dir
        / "selected_config.json"
    )

    save_json(
        selected_json_path,
        payload,
    )

    report_path = (
        output_dir
        / "12U_REPORT.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "12U Multiscale Turbulence-Aware Direct U/V Forecasting\n"
        )
        f.write(
            "=" * 132
            + "\n\n"
        )

        f.write(
            "PROTOCOL\n"
        )
        f.write(
            "-" * 132
            + "\n"
        )
        f.write(
            "TRUE Stage12B complete 10-min block means\n"
        )
        f.write(
            "6 historical blocks -> next 10-min block\n"
        )
        f.write(
            "Additional descriptors use only the 10 already-observed 1-min "
            "records inside each historical block.\n"
        )
        f.write(
            "Tropical Atlantic accessed: NO\n\n"
        )

        f.write(
            "CANDIDATE SUMMARY\n"
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
            "RESIDUAL DIAGNOSTICS\n"
        )
        f.write(
            "-" * 132
            + "\n"
        )
        f.write(
            residual_diag_df.to_string(
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
            f"raw best: {raw_best_name}\n"
        )
        f.write(
            f"selected: {selected}\n"
        )
        f.write(
            f"decision: {decision}\n\n"
        )

        f.write(
            "PUBLISHED BP-STGNN NUMERICAL REFERENCE — CONTEXT ONLY\n"
        )
        f.write(
            "-" * 132
            + "\n"
        )
        f.write(
            f"BP U={BP_STGNN_REFERENCE['U_RMSE_mps']:.6f}\n"
        )
        f.write(
            f"BP V={BP_STGNN_REFERENCE['V_RMSE_mps']:.6f}\n"
        )
        f.write(
            f"BP WS={BP_STGNN_REFERENCE['WS_RMSE_mps']:.6f}\n"
        )
        f.write(
            f"BP WD={BP_STGNN_REFERENCE['WD_RMSE_deg']:.3f}\n"
        )
        f.write(
            "These values are not used for development selection.\n"
        )

    log("")
    log("=" * 132)
    log(
        "12U CANDIDATE SUMMARY"
    )
    log("=" * 132)
    log(
        summary.to_string(
            index=False
        )
    )
    log("")
    log(
        f"[RAW BEST] {raw_best_name}"
    )
    log(
        f"[SELECTED] {selected}"
    )
    log(
        f"[DECISION] {decision}"
    )
    log("")
    log(
        "Best numerical difference vs published BP reference "
        "(context only; not preprocessing-identical):"
    )
    log(
        f"  U  : {numeric_bp['best_U_minus_BP']:+.6f} m/s"
    )
    log(
        f"  V  : {numeric_bp['best_V_minus_BP']:+.6f} m/s"
    )
    log(
        f"  WS : {numeric_bp['best_WS_minus_BP']:+.6f} m/s"
    )
    log(
        f"  WD : {numeric_bp['best_WD_minus_BP']:+.3f} deg"
    )
    log("")
    log(
        f"[SAVED] {summary_path}"
    )
    log(
        f"[SAVED] {ridge_summary_path}"
    )
    log(
        f"[SAVED] {residual_diag_path}"
    )
    log(
        f"[SAVED] {selected_json_path}"
    )
    log(
        f"[SAVED] {report_path}"
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
