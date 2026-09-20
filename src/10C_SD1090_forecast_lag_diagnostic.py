# -*- coding: utf-8 -*-
r"""
10C_SD1090_forecast_lag_diagnostic.py

Lag / phase diagnostic for the frozen SD1090 internal-evaluation results.

What this script answers
------------------------
The formal forecast at horizon h is already correctly aligned to its future
target timestamp t+h.  This script DOES NOT change that evaluation.

Instead, it performs a post-hoc diagnostic:

    E_h(L) = RMSE( A_true(t), A_pred,h(t + L) )

with integer diagnostic lag L in minutes.

Sign convention
---------------
    L > 0  => the predicted curve is dynamically L minutes LATE
              (prediction lags the truth).

Why?
If a predicted event occurs 3 min after the observed event, then
A_pred(t+3) resembles A_true(t), so the minimum diagnostic RMSE occurs
at L=+3 min.

IMPORTANT
---------
- L=0 is the ONLY formal forecast evaluation.
- Lag-adjusted RMSE is diagnostic only.
- Never replace the paper's target-time-aligned RMSE with lag-adjusted RMSE.
- No model is retrained, recalibrated, shifted, or re-selected.

Default models
--------------
Persistence
Dual Ridge
DLinear
TimeMixer
iTransformer
PatchTST
Physics-Compact

Default horizons
----------------
1, 2, 3, 5, 10 min

Diagnostics
-----------
For every model/seed/horizon/lag:
1) Earth-frame apparent-wind vector RMSE
2) U_app RMSE
3) V_app RMSE
4) level correlation of U_app / V_app
5) first-difference correlation of U_app / V_app

The first-difference correlation is included because ordinary level
correlation can be dominated by slow trends and strong autocorrelation.

Hard audits
-----------
1) Verify target_time - context_end_time = [1,2,3,5,10] min.
2) Verify zero-lag apparent-wind RMSE reproduces 10A's saved formal RMSE.
3) Use exact target timestamps for every lag alignment, not index shifting.

Outputs
-------
<10A eval dir>/lag_diagnostic/
    lag_curve_per_seed.csv
    lag_curve_seed_aggregate.csv
    lag_summary_by_horizon.csv
    lag_best_by_seed.csv
    lag_zero_shift_audit.csv
    lag_table_wide.csv
    lag_diagnostic_report.txt

    Fig_optimal_lag_vs_horizon.png/.pdf
    Fig_lag_rmse_reduction_vs_horizon.png/.pdf
    Fig_lagRMSE_h01min.png/.pdf
    Fig_lagRMSE_h02min.png/.pdf
    Fig_lagRMSE_h03min.png/.pdf
    Fig_lagRMSE_h05min.png/.pdf
    Fig_lagRMSE_h10min.png/.pdf
    Fig_deltaCorr_h01min.png/.pdf
    ...
    Fig_deltaCorr_h10min.png/.pdf

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\10C_SD1090_forecast_lag_diagnostic.py" `
  --root "D:\project\WindPredict_SaildroneData"

Default lag scan:
    -15 ... +15 min

A wider scan if desired:
python "D:\project\WindPredict_SaildroneData\src\10C_SD1090_forecast_lag_diagnostic.py" `
  --root "D:\project\WindPredict_SaildroneData" `
  --max-lag-min 20
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


HORIZONS = np.asarray([1, 2, 3, 5, 10], dtype=int)
SEEDS = [500043, 501052, 502061, 503070, 504079]

DEFAULT_MODELS = [
    "Persistence",
    "Dual Ridge",
    "DLinear",
    "TimeMixer",
    "iTransformer",
    "PatchTST",
    "Physics-Compact",
]

STOCHASTIC_MODELS = {
    "DLinear",
    "TimeMixer",
    "iTransformer",
    "PatchTST",
    "Physics-Compact",
}

COLORS = {
    "Persistence": "#7f7f7f",
    "Dual Ridge": "#8c564b",
    "Physics-Compact": "#1f77b4",
    "DLinear": "#ff7f0e",
    "TimeMixer": "#2ca02c",
    "iTransformer": "#d62728",
    "PatchTST": "#9467bd",
}

LINESTYLES = {
    "Persistence": "--",
    "Dual Ridge": "-.",
    "Physics-Compact": "-",
    "DLinear": "-",
    "TimeMixer": "-",
    "iTransformer": "-",
    "PatchTST": "-",
}

ONE_MIN_NS = int(60 * 1e9)


# =============================================================================
# Generic utilities
# =============================================================================

def log(msg=""):
    print(msg, flush=True)


def norm_name(s: str) -> str:
    return (
        str(s)
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import module from {path}"
        )

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    return mod


def load_sci_style(root: Path) -> None:
    style_path = root / "src" / "sci_plot_style.py"

    if style_path.exists():
        try:
            mod = load_module(
                style_path,
                "sci_plot_style_10c",
            )

            if hasattr(mod, "apply_sci_style"):
                try:
                    mod.apply_sci_style(
                        base_font_size=8.5
                    )
                except TypeError:
                    try:
                        mod.apply_sci_style(
                            font_size=8.5
                        )
                    except TypeError:
                        mod.apply_sci_style()

                return

        except Exception as e:
            log(
                f"[WARNING] sci_plot_style.py could not be applied: {e}"
            )

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman No9 L",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.1,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "axes.unicode_minus": False,
        }
    )


def clean_axis(ax):
    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
    )
    ax.minorticks_on()
    ax.grid(
        True,
        which="major",
        linewidth=0.40,
        alpha=0.18,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save_both(fig, base: Path):
    base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        base.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        base.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


# =============================================================================
# Resolve 10A output / import exact deterministic baselines
# =============================================================================

def resolve_eval_dir(
    dataset_dir: Path,
    explicit: Optional[Path],
) -> Path:
    if explicit is not None:
        p = explicit.resolve()

        if not p.exists():
            raise FileNotFoundError(p)

        return p

    candidates = [
        dataset_dir
        / "10A_SD1090_internal_evaluation_v0_4",
        dataset_dir
        / "10A_SD1090_internal_evaluation_v0_3",
        dataset_dir
        / "10A_SD1090_internal_evaluation_v0_1",
    ]

    for p in candidates:
        if (
            p.exists()
            and (p / "predictions").exists()
            and (p / "tables").exists()
        ):
            return p

    raise FileNotFoundError(
        "Could not find a completed 10A internal-evaluation directory. "
        "Run 10A-v4 first, or pass --eval-dir explicitly."
    )


def resolve_10a_script(root: Path) -> Path:
    candidates = [
        root
        / "src"
        / "10A_run_SD1090_internal_evaluation_v4.py",
        root
        / "src"
        / "10A_run_SD1090_internal_evaluation_v3.py",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Need 10A-v4 or 10A-v3 in the project's src directory "
        "to reconstruct Persistence and Dual Ridge exactly."
    )


# =============================================================================
# Data and target-time audit
# =============================================================================

def load_raw_test_npz(
    dataset_dir: Path,
):
    p = dataset_dir / "test.npz"

    if not p.exists():
        raise FileNotFoundError(p)

    with np.load(
        p,
        allow_pickle=False,
    ) as z:
        required = [
            "context_end_time_ns",
            "target_time_ns",
        ]

        missing = [
            k
            for k in required
            if k not in z.files
        ]

        if missing:
            raise KeyError(
                f"{p}: missing {missing}"
            )

        context_end = np.asarray(
            z["context_end_time_ns"],
            dtype=np.int64,
        )

        target_time = np.asarray(
            z["target_time_ns"],
            dtype=np.int64,
        )

        y_raw = (
            np.asarray(
                z["y_raw"],
                dtype=np.float32,
            )
            if "y_raw" in z.files
            else None
        )

    return {
        "context_end_time_ns": context_end,
        "target_time_ns": target_time,
        "y_raw": y_raw,
    }


def audit_target_times(
    raw_test,
):
    context_end = raw_test[
        "context_end_time_ns"
    ]

    target_time = raw_test[
        "target_time_ns"
    ]

    if target_time.shape != (
        len(context_end),
        len(HORIZONS),
    ):
        raise RuntimeError(
            "target_time_ns shape mismatch: "
            f"{target_time.shape}"
        )

    delta_min = (
        target_time
        - context_end[:, None]
    ) / float(ONE_MIN_NS)

    expected = HORIZONS.astype(
        np.float64
    )

    passed = np.allclose(
        delta_min,
        expected[None, :],
        atol=1e-12,
        rtol=0.0,
    )

    log("")
    log("=" * 104)
    log("TARGET-TIME AUDIT")
    log("=" * 104)
    log(
        f"shape = {delta_min.shape}"
    )
    log(
        "min   = "
        + np.array2string(
            delta_min.min(axis=0),
            precision=6,
        )
    )
    log(
        "max   = "
        + np.array2string(
            delta_min.max(axis=0),
            precision=6,
        )
    )
    log(
        "mean  = "
        + np.array2string(
            delta_min.mean(axis=0),
            precision=6,
        )
    )
    log(
        f"expected = {HORIZONS.tolist()}"
    )
    log(
        f"PASS = {passed}"
    )

    if not passed:
        raise RuntimeError(
            "Target-time audit FAILED. "
            "Do not run a lag diagnostic on a misaligned dataset."
        )

    # Every horizon time series should be strictly increasing.
    for j, h in enumerate(HORIZONS):
        if not np.all(
            np.diff(
                target_time[:, j]
            )
            > 0
        ):
            raise RuntimeError(
                f"h={h}: target timestamps are not strictly increasing."
            )

    return delta_min


# =============================================================================
# Exact prediction loading
# =============================================================================

def apparent_from_joint(y):
    y = np.asarray(
        y,
        dtype=np.float64,
    )

    return (
        y[..., 0:2]
        - y[..., 2:4]
    )


def load_modern_predictions(
    pred_dir: Path,
    model: str,
    truth_shape,
):
    safe = model.replace(
        "-",
        "_",
    )

    out = {}

    for seed in SEEDS:
        p = (
            pred_dir
            / f"test_{safe}_seed{seed}.npz"
        )

        if not p.exists():
            raise FileNotFoundError(
                f"Missing {model} TEST prediction: {p}"
            )

        with np.load(
            p,
            allow_pickle=False,
        ) as z:
            if "pred_raw" not in z.files:
                raise KeyError(
                    f"{p}: missing pred_raw"
                )

            pred = np.asarray(
                z["pred_raw"],
                dtype=np.float32,
            )

        if pred.shape != truth_shape:
            raise RuntimeError(
                f"{p}: shape={pred.shape}, "
                f"expected={truth_shape}"
            )

        out[seed] = pred

    return out


def load_pc_predictions(
    pred_dir: Path,
    truth_shape,
):
    out = {}

    for seed in SEEDS:
        p = (
            pred_dir
            / f"15E_VH_seed_{seed}_test_predictions.npz"
        )

        if not p.exists():
            raise FileNotFoundError(
                f"Missing Physics-Compact TEST prediction: {p}"
            )

        with np.load(
            p,
            allow_pickle=False,
        ) as z:
            required = [
                "frozen_truewind",
                "joint_vessel_hdg",
            ]

            missing = [
                k
                for k in required
                if k not in z.files
            ]

            if missing:
                raise KeyError(
                    f"{p}: missing {missing}"
                )

            wind = np.asarray(
                z["frozen_truewind"],
                dtype=np.float32,
            )

            motion = np.asarray(
                z["joint_vessel_hdg"],
                dtype=np.float32,
            )

        pred = np.empty(
            truth_shape,
            dtype=np.float32,
        )

        pred[..., 0:2] = wind
        pred[..., 2:6] = motion

        out[seed] = pred

    return out


def build_prediction_sets(
    root: Path,
    dataset_dir: Path,
    eval_dir: Path,
):
    """
    Load the exact formal TEST predictions and reconstruct deterministic
    Persistence / Dual Ridge using 10A's own functions.
    """
    ten_a_path = resolve_10a_script(
        root
    )

    ten_a = load_module(
        ten_a_path,
        "ten_a_for_lag_diagnostic",
    )

    (
        a,
        c,
        utility,
        utility_path,
        torch,
        nn,
        F,
        DataLoader,
        TensorDataset,
    ) = ten_a.load_infrastructure(
        root
    )

    frozen = ten_a.load_frozen_with_test(
        a,
        utility,
        dataset_dir,
    )

    truth = np.asarray(
        frozen["test_y_raw"],
        dtype=np.float32,
    )

    pred_dir = eval_dir / "predictions"

    result: Dict[
        str,
        Dict[int, np.ndarray],
    ] = {}

    persistence = ten_a.build_persistence(
        frozen
    )

    ridge_wind, ridge_motion = (
        ten_a.fit_final_ridges(
            frozen
        )
    )

    dual_ridge = ten_a.make_joint(
        ridge_wind,
        ridge_motion,
    )

    result["Persistence"] = {
        -1: persistence
    }

    result["Dual Ridge"] = {
        -1: dual_ridge
    }

    for model in [
        "DLinear",
        "TimeMixer",
        "iTransformer",
        "PatchTST",
    ]:
        result[model] = (
            load_modern_predictions(
                pred_dir,
                model,
                truth.shape,
            )
        )

    result["Physics-Compact"] = (
        load_pc_predictions(
            pred_dir,
            truth.shape,
        )
    )

    return truth, frozen, result


# =============================================================================
# Lag alignment
# =============================================================================

def aligned_indices(
    times_ns: np.ndarray,
    lag_min: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Positive lag means the prediction is late.

    Compare:
        truth(t)
        with
        prediction(t + lag)

    using exact target timestamps.

    Returns
    -------
    truth_idx, pred_idx
    """
    times = np.asarray(
        times_ns,
        dtype=np.int64,
    )

    desired_pred_time = (
        times
        + int(lag_min)
        * ONE_MIN_NS
    )

    pred_idx = np.searchsorted(
        times,
        desired_pred_time,
        side="left",
    )

    valid = (
        pred_idx
        < len(times)
    )

    safe_idx = np.minimum(
        pred_idx,
        len(times) - 1,
    )

    valid &= (
        times[
            safe_idx
        ]
        == desired_pred_time
    )

    truth_idx = np.nonzero(
        valid
    )[0]

    return (
        truth_idx,
        pred_idx[
            valid
        ],
    )


def scalar_rmse(
    obs,
    pred,
):
    e = (
        np.asarray(
            pred,
            dtype=np.float64,
        )
        - np.asarray(
            obs,
            dtype=np.float64,
        )
    )

    return float(
        np.sqrt(
            np.mean(
                e * e
            )
        )
    )


def vector_rmse(
    obs,
    pred,
):
    e = (
        np.asarray(
            pred,
            dtype=np.float64,
        )
        - np.asarray(
            obs,
            dtype=np.float64,
        )
    )

    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    e * e,
                    axis=-1,
                )
            )
        )
    )


def safe_corr(
    a,
    b,
):
    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    good = (
        np.isfinite(a)
        & np.isfinite(b)
    )

    a = a[good]
    b = b[good]

    if len(a) < 3:
        return np.nan

    sa = float(
        np.std(
            a,
            ddof=0,
        )
    )

    sb = float(
        np.std(
            b,
            ddof=0,
        )
    )

    if (
        sa < 1e-12
        or sb < 1e-12
    ):
        return np.nan

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def make_one_minute_differences(
    times_ns,
    uv,
):
    """
    Attach each first difference to its second timestamp and retain only
    genuine consecutive 1-min pairs.
    """
    times = np.asarray(
        times_ns,
        dtype=np.int64,
    )

    uv = np.asarray(
        uv,
        dtype=np.float64,
    )

    if len(times) != len(uv):
        raise RuntimeError(
            "time/value length mismatch"
        )

    consecutive = (
        np.diff(
            times
        )
        == ONE_MIN_NS
    )

    diff_time = times[
        1:
    ][
        consecutive
    ]

    diff_uv = (
        uv[
            1:
        ]
        - uv[
            :-1
        ]
    )[
        consecutive
    ]

    return (
        diff_time,
        diff_uv,
    )


def lag_metrics(
    times_ns,
    truth_uv,
    pred_uv,
    lag_min: int,
):
    truth_idx, pred_idx = (
        aligned_indices(
            times_ns,
            lag_min,
        )
    )

    if len(truth_idx) < 10:
        raise RuntimeError(
            f"Too few aligned level samples at lag={lag_min}: "
            f"{len(truth_idx)}"
        )

    t = np.asarray(
        truth_uv,
        dtype=np.float64,
    )[
        truth_idx
    ]

    p = np.asarray(
        pred_uv,
        dtype=np.float64,
    )[
        pred_idx
    ]

    # First differences are separately timestamp-aligned.
    dt_t, dtruth = (
        make_one_minute_differences(
            times_ns,
            truth_uv,
        )
    )

    dt_p, dpred = (
        make_one_minute_differences(
            times_ns,
            pred_uv,
        )
    )

    desired_pred_diff_time = (
        dt_t
        + int(lag_min)
        * ONE_MIN_NS
    )

    pidx = np.searchsorted(
        dt_p,
        desired_pred_diff_time,
        side="left",
    )

    valid = pidx < len(dt_p)

    safe = np.minimum(
        pidx,
        len(dt_p) - 1,
    )

    valid &= (
        dt_p[
            safe
        ]
        == desired_pred_diff_time
    )

    dt = dtruth[
        valid
    ]

    dp = dpred[
        pidx[
            valid
        ]
    ]

    level_corr_u = safe_corr(
        t[:, 0],
        p[:, 0],
    )

    level_corr_v = safe_corr(
        t[:, 1],
        p[:, 1],
    )

    delta_corr_u = safe_corr(
        dt[:, 0],
        dp[:, 0],
    )

    delta_corr_v = safe_corr(
        dt[:, 1],
        dp[:, 1],
    )

    corr_level_mean = float(
        np.nanmean(
            [
                level_corr_u,
                level_corr_v,
            ]
        )
    )

    corr_delta_mean = float(
        np.nanmean(
            [
                delta_corr_u,
                delta_corr_v,
            ]
        )
    )

    return {
        "n_level": int(
            len(t)
        ),
        "n_delta": int(
            len(dt)
        ),
        "Uapp_RMSE_mps": scalar_rmse(
            t[:, 0],
            p[:, 0],
        ),
        "Vapp_RMSE_mps": scalar_rmse(
            t[:, 1],
            p[:, 1],
        ),
        "AppVector_RMSE_mps": vector_rmse(
            t,
            p,
        ),
        "level_corr_U": level_corr_u,
        "level_corr_V": level_corr_v,
        "level_corr_mean": corr_level_mean,
        "delta_corr_U": delta_corr_u,
        "delta_corr_V": delta_corr_v,
        "delta_corr_mean": corr_delta_mean,
    }


# =============================================================================
# Full diagnostic
# =============================================================================

def run_lag_scan(
    truth_joint,
    predictions,
    target_time_ns,
    max_lag_min: int,
):
    rows = []

    lags = np.arange(
        -max_lag_min,
        max_lag_min + 1,
        dtype=int,
    )

    truth_app = apparent_from_joint(
        truth_joint
    )

    for model in DEFAULT_MODELS:
        if model not in predictions:
            continue

        log("")
        log(
            f"[LAG SCAN] {model}"
        )

        for seed, pred_joint in predictions[
            model
        ].items():
            pred_app = apparent_from_joint(
                pred_joint
            )

            for h_idx, h in enumerate(
                HORIZONS
            ):
                times = target_time_ns[
                    :,
                    h_idx
                ]

                true_uv = truth_app[
                    :,
                    h_idx,
                    :
                ]

                pred_uv = pred_app[
                    :,
                    h_idx,
                    :
                ]

                for lag in lags:
                    met = lag_metrics(
                        times,
                        true_uv,
                        pred_uv,
                        int(lag),
                    )

                    rows.append(
                        {
                            "model": model,
                            "seed": int(seed),
                            "horizon_min": int(h),
                            "lag_min": int(lag),
                            **met,
                        }
                    )

            log(
                f"  seed={seed} complete"
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Aggregate and optimal-lag extraction
# =============================================================================

def aggregate_lag_curves(
    per_seed: pd.DataFrame,
):
    metrics = [
        "Uapp_RMSE_mps",
        "Vapp_RMSE_mps",
        "AppVector_RMSE_mps",
        "level_corr_U",
        "level_corr_V",
        "level_corr_mean",
        "delta_corr_U",
        "delta_corr_V",
        "delta_corr_mean",
    ]

    rows = []

    for (
        model,
        horizon,
        lag,
    ), g in per_seed.groupby(
        [
            "model",
            "horizon_min",
            "lag_min",
        ],
        sort=False,
    ):
        row = {
            "model": model,
            "horizon_min": int(
                horizon
            ),
            "lag_min": int(
                lag
            ),
            "n_seeds": int(
                g["seed"].nunique()
            ),
            "n_level_min": int(
                g[
                    "n_level"
                ].min()
            ),
            "n_delta_min": int(
                g[
                    "n_delta"
                ].min()
            ),
        }

        for metric in metrics:
            vals = pd.to_numeric(
                g[
                    metric
                ],
                errors="coerce",
            )

            row[
                f"{metric}_mean"
            ] = float(
                vals.mean()
            )

            row[
                f"{metric}_std"
            ] = (
                float(
                    vals.std(
                        ddof=1
                    )
                )
                if len(
                    vals
                )
                > 1
                else 0.0
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def best_lag_per_seed(
    per_seed: pd.DataFrame,
):
    rows = []

    for (
        model,
        seed,
        horizon,
    ), g in per_seed.groupby(
        [
            "model",
            "seed",
            "horizon_min",
        ],
        sort=False,
    ):
        g = g.sort_values(
            "lag_min"
        ).reset_index(
            drop=True
        )

        zero = g[
            g[
                "lag_min"
            ]
            == 0
        ]

        if len(zero) != 1:
            raise RuntimeError(
                f"{model}/{seed}/h={horizon}: "
                "zero-lag row missing."
            )

        # RMSE optimum; ties resolved by smallest absolute lag, then
        # smallest signed lag.
        min_rmse = float(
            g[
                "AppVector_RMSE_mps"
            ].min()
        )

        candidates = g[
            np.isclose(
                g[
                    "AppVector_RMSE_mps"
                ],
                min_rmse,
                atol=1e-12,
                rtol=0.0,
            )
        ].copy()

        candidates[
            "_abs_lag"
        ] = np.abs(
            candidates[
                "lag_min"
            ]
        )

        best = (
            candidates
            .sort_values(
                [
                    "_abs_lag",
                    "lag_min",
                ]
            )
            .iloc[
                0
            ]
        )

        # Independent first-difference-correlation optimum.
        max_dc = float(
            g[
                "delta_corr_mean"
            ].max()
        )

        c2 = g[
            np.isclose(
                g[
                    "delta_corr_mean"
                ],
                max_dc,
                atol=1e-12,
                rtol=0.0,
            )
        ].copy()

        c2[
            "_abs_lag"
        ] = np.abs(
            c2[
                "lag_min"
            ]
        )

        best_dc = (
            c2
            .sort_values(
                [
                    "_abs_lag",
                    "lag_min",
                ]
            )
            .iloc[
                0
            ]
        )

        zero_rmse = float(
            zero.iloc[
                0
            ][
                "AppVector_RMSE_mps"
            ]
        )

        best_rmse = float(
            best[
                "AppVector_RMSE_mps"
            ]
        )

        reduction = (
            (
                zero_rmse
                - best_rmse
            )
            / max(
                zero_rmse,
                1e-12,
            )
            * 100.0
        )

        rows.append(
            {
                "model": model,
                "seed": int(
                    seed
                ),
                "horizon_min": int(
                    horizon
                ),
                "formal_zero_lag_RMSE_mps": zero_rmse,
                "diagnostic_best_lag_min": int(
                    best[
                        "lag_min"
                    ]
                ),
                "diagnostic_best_lag_RMSE_mps": best_rmse,
                "diagnostic_RMSE_reduction_pct": float(
                    reduction
                ),
                "zero_lag_delta_corr_mean": float(
                    zero.iloc[
                        0
                    ][
                        "delta_corr_mean"
                    ]
                ),
                "best_delta_corr_lag_min": int(
                    best_dc[
                        "lag_min"
                    ]
                ),
                "best_delta_corr_mean": float(
                    best_dc[
                        "delta_corr_mean"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def horizon_summary_from_aggregate(
    agg_curve: pd.DataFrame,
):
    """
    Paper-level diagnostic:
    choose one common best lag for each model/horizon from the
    five-seed MEAN lag-RMSE curve.
    """
    rows = []

    for (
        model,
        horizon,
    ), g in agg_curve.groupby(
        [
            "model",
            "horizon_min",
        ],
        sort=False,
    ):
        g = g.sort_values(
            "lag_min"
        ).reset_index(
            drop=True
        )

        zero = g[
            g[
                "lag_min"
            ]
            == 0
        ]

        if len(zero) != 1:
            raise RuntimeError(
                f"{model}/h={horizon}: "
                "zero-lag aggregate row missing."
            )

        min_rmse = float(
            g[
                "AppVector_RMSE_mps_mean"
            ].min()
        )

        c = g[
            np.isclose(
                g[
                    "AppVector_RMSE_mps_mean"
                ],
                min_rmse,
                atol=1e-12,
                rtol=0.0,
            )
        ].copy()

        c[
            "_abs_lag"
        ] = np.abs(
            c[
                "lag_min"
            ]
        )

        best = (
            c
            .sort_values(
                [
                    "_abs_lag",
                    "lag_min",
                ]
            )
            .iloc[
                0
            ]
        )

        max_dc = float(
            g[
                "delta_corr_mean_mean"
            ].max()
        )

        c2 = g[
            np.isclose(
                g[
                    "delta_corr_mean_mean"
                ],
                max_dc,
                atol=1e-12,
                rtol=0.0,
            )
        ].copy()

        c2[
            "_abs_lag"
        ] = np.abs(
            c2[
                "lag_min"
            ]
        )

        best_dc = (
            c2
            .sort_values(
                [
                    "_abs_lag",
                    "lag_min",
                ]
            )
            .iloc[
                0
            ]
        )

        zero_rmse = float(
            zero.iloc[
                0
            ][
                "AppVector_RMSE_mps_mean"
            ]
        )

        best_rmse = float(
            best[
                "AppVector_RMSE_mps_mean"
            ]
        )

        reduction = (
            (
                zero_rmse
                - best_rmse
            )
            / max(
                zero_rmse,
                1e-12,
            )
            * 100.0
        )

        rows.append(
            {
                "model": model,
                "horizon_min": int(
                    horizon
                ),
                "n_seeds": int(
                    zero.iloc[
                        0
                    ][
                        "n_seeds"
                    ]
                ),
                "formal_zero_lag_RMSE_mean_mps": zero_rmse,
                "formal_zero_lag_RMSE_std_mps": float(
                    zero.iloc[
                        0
                    ][
                        "AppVector_RMSE_mps_std"
                    ]
                ),
                "diagnostic_best_lag_min": int(
                    best[
                        "lag_min"
                    ]
                ),
                "diagnostic_best_lag_RMSE_mean_mps": best_rmse,
                "diagnostic_best_lag_RMSE_std_mps": float(
                    best[
                        "AppVector_RMSE_mps_std"
                    ]
                ),
                "diagnostic_RMSE_reduction_pct": float(
                    reduction
                ),
                "formal_zero_lag_delta_corr_mean": float(
                    zero.iloc[
                        0
                    ][
                        "delta_corr_mean_mean"
                    ]
                ),
                "best_delta_corr_lag_min": int(
                    best_dc[
                        "lag_min"
                    ]
                ),
                "best_delta_corr_mean": float(
                    best_dc[
                        "delta_corr_mean_mean"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Zero-lag audit against 10A formal metrics
# =============================================================================

def audit_zero_lag_against_10a(
    eval_dir: Path,
    per_seed_curve: pd.DataFrame,
    tolerance: float = 5e-6,
):
    p = (
        eval_dir
        / "tables"
        / "internal_per_horizon_metrics.csv"
    )

    if not p.exists():
        raise FileNotFoundError(
            f"10A formal per-horizon metrics not found: {p}"
        )

    formal = pd.read_csv(
        p
    )

    required = [
        "model",
        "seed",
        "horizon_min",
        "apparent_vector_RMSE_mps",
    ]

    missing = [
        c
        for c in required
        if c not in formal.columns
    ]

    if missing:
        raise RuntimeError(
            f"{p}: missing columns {missing}"
        )

    zero = per_seed_curve[
        per_seed_curve[
            "lag_min"
        ]
        == 0
    ][
        [
            "model",
            "seed",
            "horizon_min",
            "AppVector_RMSE_mps",
        ]
    ].copy()

    zero = zero.rename(
        columns={
            "AppVector_RMSE_mps":
            "lag_script_zero_RMSE_mps"
        }
    )

    formal2 = formal[
        formal[
            "model"
        ].isin(
            DEFAULT_MODELS
        )
    ][
        required
    ].copy()

    merged = zero.merge(
        formal2,
        on=[
            "model",
            "seed",
            "horizon_min",
        ],
        how="left",
        validate="one_to_one",
    )

    merged[
        "absolute_difference_mps"
    ] = np.abs(
        merged[
            "lag_script_zero_RMSE_mps"
        ]
        - merged[
            "apparent_vector_RMSE_mps"
        ]
    )

    merged[
        "PASS"
    ] = (
        merged[
            "absolute_difference_mps"
        ]
        <= tolerance
    )

    missing_formal = merged[
        "apparent_vector_RMSE_mps"
    ].isna()

    if missing_formal.any():
        bad = merged[
            missing_formal
        ][
            [
                "model",
                "seed",
                "horizon_min",
            ]
        ]

        raise RuntimeError(
            "Could not match some zero-lag rows to 10A formal metrics:\n"
            + bad.to_string(
                index=False
            )
        )

    max_diff = float(
        merged[
            "absolute_difference_mps"
        ].max()
    )

    log("")
    log("=" * 104)
    log("ZERO-LAG RMSE AUDIT AGAINST 10A")
    log("=" * 104)
    log(
        f"rows      = {len(merged)}"
    )
    log(
        f"max |diff| = {max_diff:.9e} m/s"
    )
    log(
        f"tolerance  = {tolerance:.9e} m/s"
    )
    log(
        f"PASS       = {bool(merged['PASS'].all())}"
    )

    if not bool(
        merged[
            "PASS"
        ].all()
    ):
        bad = merged[
            ~merged[
                "PASS"
            ]
        ].sort_values(
            "absolute_difference_mps",
            ascending=False,
        )

        raise RuntimeError(
            "Lag diagnostic zero-lag RMSE does not reproduce 10A.\n"
            + bad.head(
                20
            ).to_string(
                index=False
            )
        )

    return merged


# =============================================================================
# Wide table for manuscript inspection
# =============================================================================

def make_wide_table(
    summary: pd.DataFrame,
):
    rows = []

    for model in DEFAULT_MODELS:
        g = summary[
            summary[
                "model"
            ]
            == model
        ]

        if g.empty:
            continue

        row = {
            "Model": model
        }

        for h in HORIZONS:
            q = g[
                g[
                    "horizon_min"
                ]
                == int(h)
            ]

            if q.empty:
                continue

            r = q.iloc[
                0
            ]

            row[
                f"{h}min zero-lag RMSE"
            ] = float(
                r[
                    "formal_zero_lag_RMSE_mean_mps"
                ]
            )

            row[
                f"{h}min best lag"
            ] = int(
                r[
                    "diagnostic_best_lag_min"
                ]
            )

            row[
                f"{h}min lag-adjusted RMSE"
            ] = float(
                r[
                    "diagnostic_best_lag_RMSE_mean_mps"
                ]
            )

            row[
                f"{h}min reduction %"
            ] = float(
                r[
                    "diagnostic_RMSE_reduction_pct"
                ]
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Plotting
# =============================================================================

def plot_optimal_lag(
    summary: pd.DataFrame,
    out_dir: Path,
):
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(7.15, 3.65),
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.17,
        top=0.84,
    )

    for model in DEFAULT_MODELS:
        g = (
            summary[
                summary[
                    "model"
                ]
                == model
            ]
            .sort_values(
                "horizon_min"
            )
        )

        if g.empty:
            continue

        ax.plot(
            g[
                "horizon_min"
            ],
            g[
                "diagnostic_best_lag_min"
            ],
            marker="o",
            markersize=4.0,
            linewidth=1.05,
            linestyle=LINESTYLES[
                model
            ],
            color=COLORS[
                model
            ],
            label=model,
        )

    ax.axhline(
        0.0,
        linewidth=0.7,
        color="black",
        alpha=0.55,
    )

    ax.set_xticks(
        HORIZONS
    )

    ax.set_xlabel(
        "Forecast horizon (min)"
    )

    ax.set_ylabel(
        "Diagnostic optimal lag (min)"
    )

    ax.text(
        0.995,
        0.02,
        "Positive lag = prediction is late",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
    )

    clean_axis(
        ax
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.20,
        ),
        ncol=4,
        frameon=False,
        fontsize=6.9,
        columnspacing=0.9,
        handlelength=2.0,
    )

    save_both(
        fig,
        out_dir
        / "Fig_optimal_lag_vs_horizon",
    )


def plot_reduction(
    summary: pd.DataFrame,
    out_dir: Path,
):
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(7.15, 3.65),
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.17,
        top=0.84,
    )

    for model in DEFAULT_MODELS:
        g = (
            summary[
                summary[
                    "model"
                ]
                == model
            ]
            .sort_values(
                "horizon_min"
            )
        )

        if g.empty:
            continue

        ax.plot(
            g[
                "horizon_min"
            ],
            g[
                "diagnostic_RMSE_reduction_pct"
            ],
            marker="o",
            markersize=4.0,
            linewidth=1.05,
            linestyle=LINESTYLES[
                model
            ],
            color=COLORS[
                model
            ],
            label=model,
        )

    ax.set_xticks(
        HORIZONS
    )

    ax.set_xlabel(
        "Forecast horizon (min)"
    )

    ax.set_ylabel(
        "RMSE removable by lag alignment (%)"
    )

    clean_axis(
        ax
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.20,
        ),
        ncol=4,
        frameon=False,
        fontsize=6.9,
        columnspacing=0.9,
        handlelength=2.0,
    )

    save_both(
        fig,
        out_dir
        / "Fig_lag_rmse_reduction_vs_horizon",
    )


def plot_lag_rmse_curves(
    agg_curve: pd.DataFrame,
    out_dir: Path,
):
    for h in HORIZONS:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(7.15, 3.75),
            constrained_layout=False,
        )

        fig.subplots_adjust(
            left=0.105,
            right=0.985,
            bottom=0.17,
            top=0.84,
        )

        for model in DEFAULT_MODELS:
            g = (
                agg_curve[
                    (
                        agg_curve[
                            "model"
                        ]
                        == model
                    )
                    & (
                        agg_curve[
                            "horizon_min"
                        ]
                        == int(h)
                    )
                ]
                .sort_values(
                    "lag_min"
                )
            )

            if g.empty:
                continue

            zero_q = g[
                g[
                    "lag_min"
                ]
                == 0
            ]

            if zero_q.empty:
                continue

            zero = float(
                zero_q.iloc[
                    0
                ][
                    "AppVector_RMSE_mps_mean"
                ]
            )

            ratio = (
                g[
                    "AppVector_RMSE_mps_mean"
                ].to_numpy(
                    dtype=float
                )
                / max(
                    zero,
                    1e-12,
                )
            )

            ax.plot(
                g[
                    "lag_min"
                ],
                ratio,
                linewidth=1.05,
                linestyle=LINESTYLES[
                    model
                ],
                color=COLORS[
                    model
                ],
                label=model,
            )

        ax.axvline(
            0,
            linewidth=0.7,
            color="black",
            alpha=0.55,
        )

        ax.axhline(
            1.0,
            linewidth=0.7,
            color="black",
            alpha=0.35,
        )

        ax.set_xlabel(
            "Diagnostic lag (min)"
        )

        ax.set_ylabel(
            r"RMSE$(L)$/RMSE$(0)$"
        )

        ax.set_title(
            f"{int(h)}-min forecast horizon",
            pad=4,
        )

        ax.text(
            0.995,
            0.02,
            "Positive lag = prediction is late",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.2,
        )

        clean_axis(
            ax
        )

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(
                0.5,
                1.20,
            ),
            ncol=4,
            frameon=False,
            fontsize=6.9,
            columnspacing=0.9,
            handlelength=2.0,
        )

        save_both(
            fig,
            out_dir
            / f"Fig_lagRMSE_h{int(h):02d}min",
        )


def plot_delta_corr_curves(
    agg_curve: pd.DataFrame,
    out_dir: Path,
):
    for h in HORIZONS:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(7.15, 3.75),
            constrained_layout=False,
        )

        fig.subplots_adjust(
            left=0.105,
            right=0.985,
            bottom=0.17,
            top=0.84,
        )

        for model in DEFAULT_MODELS:
            g = (
                agg_curve[
                    (
                        agg_curve[
                            "model"
                        ]
                        == model
                    )
                    & (
                        agg_curve[
                            "horizon_min"
                        ]
                        == int(h)
                    )
                ]
                .sort_values(
                    "lag_min"
                )
            )

            if g.empty:
                continue

            ax.plot(
                g[
                    "lag_min"
                ],
                g[
                    "delta_corr_mean_mean"
                ],
                linewidth=1.05,
                linestyle=LINESTYLES[
                    model
                ],
                color=COLORS[
                    model
                ],
                label=model,
            )

        ax.axvline(
            0,
            linewidth=0.7,
            color="black",
            alpha=0.55,
        )

        ax.set_xlabel(
            "Diagnostic lag (min)"
        )

        ax.set_ylabel(
            r"Mean correlation of $\Delta U_{\rm app},\Delta V_{\rm app}$"
        )

        ax.set_title(
            f"{int(h)}-min forecast horizon",
            pad=4,
        )

        ax.text(
            0.995,
            0.02,
            "Positive lag = prediction is late",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.2,
        )

        clean_axis(
            ax
        )

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(
                0.5,
                1.20,
            ),
            ncol=4,
            frameon=False,
            fontsize=6.9,
            columnspacing=0.9,
            handlelength=2.0,
        )

        save_both(
            fig,
            out_dir
            / f"Fig_deltaCorr_h{int(h):02d}min",
        )


# =============================================================================
# Text report
# =============================================================================

def write_report(
    out_dir: Path,
    eval_dir: Path,
    summary: pd.DataFrame,
    zero_audit: pd.DataFrame,
    max_lag_min: int,
):
    lines = []

    lines.append(
        "10C — SD1090 FORECAST-LAG DIAGNOSTIC"
    )

    lines.append(
        "=" * 110
    )

    lines.append("")
    lines.append(
        "Interpretation:"
    )

    lines.append(
        "  Formal evaluation is always lag=0."
    )

    lines.append(
        "  Positive diagnostic lag means the predicted curve is dynamically late."
    )

    lines.append(
        "  Lag-adjusted RMSE is diagnostic only and must not replace formal forecast RMSE."
    )

    lines.append("")
    lines.append(
        f"10A source: {eval_dir}"
    )

    lines.append(
        f"Lag search: {-max_lag_min} ... +{max_lag_min} min"
    )

    lines.append("")
    lines.append(
        "ZERO-LAG AUDIT"
    )

    lines.append(
        "-" * 110
    )

    lines.append(
        f"Rows: {len(zero_audit)}"
    )

    lines.append(
        "Maximum absolute difference versus 10A formal AW RMSE: "
        f"{zero_audit['absolute_difference_mps'].max():.9e} m/s"
    )

    lines.append(
        f"PASS: {bool(zero_audit['PASS'].all())}"
    )

    lines.append("")
    lines.append(
        "HORIZON SUMMARY"
    )

    lines.append(
        "-" * 110
    )

    display_cols = [
        "model",
        "horizon_min",
        "formal_zero_lag_RMSE_mean_mps",
        "diagnostic_best_lag_min",
        "diagnostic_best_lag_RMSE_mean_mps",
        "diagnostic_RMSE_reduction_pct",
        "formal_zero_lag_delta_corr_mean",
        "best_delta_corr_lag_min",
        "best_delta_corr_mean",
    ]

    lines.append(
        summary[
            display_cols
        ].to_string(
            index=False
        )
    )

    lines.append("")
    lines.append(
        "CAUTION"
    )

    lines.append(
        "-" * 110
    )

    lines.append(
        "Do not shift predictions before reporting RMSE in the manuscript."
    )

    lines.append(
        "Use the lag analysis only as post-hoc dynamic-error diagnosis."
    )

    (
        out_dir
        / "lag_diagnostic_report.txt"
    ).write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        type=Path,
        default=Path(
            r"D:\project\WindPredict_SaildroneData"
        ),
    )

    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--max-lag-min",
        type=int,
        default=15,
    )

    args = ap.parse_args()

    if args.max_lag_min < 1:
        raise ValueError(
            "--max-lag-min must be >= 1"
        )

    root = args.root.resolve()

    dataset_dir = (
        args.dataset_dir.resolve()
        if args.dataset_dir is not None
        else root
        / "data"
        / "forecasting"
        / "SD1090_TPOS2024_JointForecasting_v0_1"
    )

    eval_dir = resolve_eval_dir(
        dataset_dir,
        args.eval_dir,
    )

    out_dir = (
        eval_dir
        / "lag_diagnostic"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    load_sci_style(
        root
    )

    log(
        "=" * 104
    )

    log(
        "10C — SD1090 FORECAST-LAG DIAGNOSTIC"
    )

    log(
        "=" * 104
    )

    log(
        f"root       : {root}"
    )

    log(
        f"dataset    : {dataset_dir}"
    )

    log(
        f"10A output : {eval_dir}"
    )

    log(
        f"output     : {out_dir}"
    )

    log(
        f"lag scan   : {-args.max_lag_min} ... +{args.max_lag_min} min"
    )

    log(
        "sign       : positive = prediction is dynamically late"
    )

    log(
        "formal RMSE: lag=0 only"
    )

    # -----------------------------------------------------------------
    # Target-time audit.
    # -----------------------------------------------------------------
    raw_test = load_raw_test_npz(
        dataset_dir
    )

    audit_target_times(
        raw_test
    )

    target_time_ns = raw_test[
        "target_time_ns"
    ]

    # -----------------------------------------------------------------
    # Exact formal predictions.
    # -----------------------------------------------------------------
    truth, frozen, predictions = (
        build_prediction_sets(
            root,
            dataset_dir,
            eval_dir,
        )
    )

    if (
        raw_test[
            "y_raw"
        ]
        is not None
    ):
        if not np.allclose(
            truth,
            raw_test[
                "y_raw"
            ],
            atol=2e-5,
            rtol=0.0,
        ):
            raise RuntimeError(
                "10A truth does not align with raw test.npz y_raw."
            )

    log("")
    log(
        "Models:"
    )

    for model in DEFAULT_MODELS:
        sm = predictions[
            model
        ]

        log(
            f"  {model:17s} | "
            f"prediction runs={len(sm)}"
        )

    # -----------------------------------------------------------------
    # Lag scan.
    # -----------------------------------------------------------------
    per_seed_curve = run_lag_scan(
        truth,
        predictions,
        target_time_ns,
        args.max_lag_min,
    )

    # -----------------------------------------------------------------
    # Hard zero-lag reproduction audit.
    # -----------------------------------------------------------------
    zero_audit = audit_zero_lag_against_10a(
        eval_dir,
        per_seed_curve,
    )

    # -----------------------------------------------------------------
    # Aggregate and choose diagnostic optimums.
    # -----------------------------------------------------------------
    agg_curve = aggregate_lag_curves(
        per_seed_curve
    )

    best_seed = best_lag_per_seed(
        per_seed_curve
    )

    summary = horizon_summary_from_aggregate(
        agg_curve
    )

    wide = make_wide_table(
        summary
    )

    # -----------------------------------------------------------------
    # Save numeric products.
    # -----------------------------------------------------------------
    per_seed_curve.to_csv(
        out_dir
        / "lag_curve_per_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    agg_curve.to_csv(
        out_dir
        / "lag_curve_seed_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_seed.to_csv(
        out_dir
        / "lag_best_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        out_dir
        / "lag_summary_by_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    zero_audit.to_csv(
        out_dir
        / "lag_zero_shift_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    wide.to_csv(
        out_dir
        / "lag_table_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Figures.
    # -----------------------------------------------------------------
    plot_optimal_lag(
        summary,
        out_dir,
    )

    plot_reduction(
        summary,
        out_dir,
    )

    plot_lag_rmse_curves(
        agg_curve,
        out_dir,
    )

    plot_delta_corr_curves(
        agg_curve,
        out_dir,
    )

    # -----------------------------------------------------------------
    # Text report.
    # -----------------------------------------------------------------
    write_report(
        out_dir,
        eval_dir,
        summary,
        zero_audit,
        args.max_lag_min,
    )

    # -----------------------------------------------------------------
    # Console key results.
    # -----------------------------------------------------------------
    log("")
    log(
        "=" * 104
    )

    log(
        "KEY LAG-DIAGNOSTIC RESULTS"
    )

    log(
        "=" * 104
    )

    cols = [
        "model",
        "horizon_min",
        "formal_zero_lag_RMSE_mean_mps",
        "diagnostic_best_lag_min",
        "diagnostic_best_lag_RMSE_mean_mps",
        "diagnostic_RMSE_reduction_pct",
        "formal_zero_lag_delta_corr_mean",
        "best_delta_corr_lag_min",
        "best_delta_corr_mean",
    ]

    pd.set_option(
        "display.width",
        220,
    )

    pd.set_option(
        "display.max_columns",
        None,
    )

    log(
        summary[
            cols
        ].to_string(
            index=False
        )
    )

    log("")
    log(
        "[INTERPRETATION]"
    )

    log(
        "  Positive best lag -> prediction is dynamically late."
    )

    log(
        "  Near-zero best lag -> little systematic phase lag."
    )

    log(
        "  Larger RMSE reduction after lag alignment -> a larger share "
        "of the formal error is associated with timing/phase."
    )

    log(
        "  First-difference correlation diagnoses rapid-change tracking "
        "rather than only slow-trend agreement."
    )

    log("")
    log(
        f"[SAVED] {out_dir}"
    )


if __name__ == "__main__":
    main()
