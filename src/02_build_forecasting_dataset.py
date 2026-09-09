# -*- coding: utf-8 -*-
"""
02_build_forecasting_dataset.py

Build a leakage-safe forecasting dataset from the audited Saildrone SD1090 data.

Frozen v0.1 design
------------------
Source:
    selected_variables_raw.csv
    joint_wind_valid_segments.csv
produced by 01_audit_saildrone_wind_v5_sci.py

Automatically:
1. Selects the longest continuous joint U/V-valid segment.
2. Uses only that segment; no long-gap interpolation.
3. Uses chronological 70/15/15 raw-time splits by default.
4. Splits raw rows FIRST, then builds windows independently inside each split.
   Therefore no input window crosses train/validation/test boundaries.
5. Encodes circular COG/HDG as sin/cos.
6. Uses the frozen v0.1 inputs:
       UWND_MEAN
       VWND_MEAN
       TEMP_AIR_MEAN
       RH_MEAN
       SOG
       sin(COG), cos(COG)
       sin(HDG), cos(HDG)
7. Predicts future horizontal wind vector:
       UWND_MEAN, VWND_MEAN
   at horizons:
       1, 2, 3, 5, 10 min
8. Does NOT interpolate missing values.
   Any candidate sample whose input window or requested targets contain NaN/Inf
   is discarded.
9. Fits feature and target standardization using TRAIN data only.
10. Saves train/validation/test datasets and full reproducibility metadata.

Default look-back:
    60 min

Default outputs:
    train.npz
    validation.npz
    test.npz
    scalers.json
    dataset_manifest.json
    split_summary.csv
    feature_definition.csv
    rejected_windows.csv
    dataset_build_report.txt
    figures/  (only when --plots is supplied)

NPZ contents
------------
X:
    standardized float32 input array
    shape = [samples, lookback, features]

y:
    standardized float32 target array
    shape = [samples, horizons, 2]

y_raw:
    original m/s target values
    shape = [samples, horizons, 2]

context_end_time_ns:
    int64 Unix nanoseconds for the last observation in each input window

target_time_ns:
    int64 Unix nanoseconds
    shape = [samples, horizons]

anchor_row_index:
    row index in the longest continuous segment for each sample

Notes
-----
- Standardization statistics are fitted ONLY on valid TRAIN rows.
- U/V targets are standardized with their own train-only mean/std.
- Validation and test never influence scalers.
- The script verifies that each split has strict 1-min time continuity.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_VERSION = "0.1.2"

DEFAULT_LOOKBACK = 60
DEFAULT_HORIZONS = (1, 2, 3, 5, 10)
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VALIDATION_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.15
EXPECTED_DT_MIN = 1.0

RAW_REQUIRED_COLUMNS = [
    "time",
    "UWND_MEAN",
    "VWND_MEAN",
    "TEMP_AIR_MEAN",
    "RH_MEAN",
    "SOG",
    "COG",
    "HDG",
]

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
]

TARGET_NAMES = [
    "UWND_MEAN",
    "VWND_MEAN",
]


def log(message: str = "") -> None:
    print(message, flush=True)


def parse_horizons(text: str) -> tuple[int, ...]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("All forecast horizons must be positive integers.")
        values.append(value)

    if not values:
        raise ValueError("At least one forecast horizon is required.")

    values = sorted(set(values))
    return tuple(values)


def check_split_fractions(train: float, val: float, test: float) -> None:
    values = [train, val, test]
    if any(v <= 0.0 for v in values):
        raise ValueError("All split fractions must be > 0.")

    total = sum(values)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"train + validation + test fractions must equal 1.0, got {total:.12f}"
        )


def read_audit_inputs(audit_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_csv = audit_dir / "selected_variables_raw.csv"
    segment_csv = audit_dir / "joint_wind_valid_segments.csv"

    if not raw_csv.exists():
        raise FileNotFoundError(
            f"Missing required audit output:\n{raw_csv}"
        )

    if not segment_csv.exists():
        raise FileNotFoundError(
            f"Missing required audit output:\n{segment_csv}"
        )

    log(f"[READ] {raw_csv}")
    raw = pd.read_csv(raw_csv, low_memory=False)

    log(f"[READ] {segment_csv}")
    segments = pd.read_csv(segment_csv)

    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise KeyError(
            "selected_variables_raw.csv is missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    if segments.empty:
        raise RuntimeError("joint_wind_valid_segments.csv contains no valid segments.")

    for c in ["start_time", "end_time", "sample_count"]:
        if c not in segments.columns:
            raise KeyError(
                f"joint_wind_valid_segments.csv is missing required column: {c}"
            )

    raw["time"] = pd.to_datetime(raw["time"], errors="coerce")

    if raw["time"].isna().any():
        bad = int(raw["time"].isna().sum())
        raise RuntimeError(f"Raw table contains {bad} unparsable timestamps.")

    return raw, segments


def choose_longest_segment(
    raw: pd.DataFrame,
    segments: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    seg = segments.copy()
    seg["sample_count"] = pd.to_numeric(seg["sample_count"], errors="coerce")
    seg["start_time"] = pd.to_datetime(seg["start_time"], errors="coerce")
    seg["end_time"] = pd.to_datetime(seg["end_time"], errors="coerce")

    seg = seg.dropna(subset=["sample_count", "start_time", "end_time"])

    if seg.empty:
        raise RuntimeError("No parseable valid segment was found.")

    longest = seg.sort_values(
        ["sample_count", "start_time"],
        ascending=[False, True],
    ).iloc[0]

    start = longest["start_time"]
    end = longest["end_time"]

    core = raw.loc[
        (raw["time"] >= start) & (raw["time"] <= end)
    ].copy()

    core = core.sort_values("time").reset_index(drop=True)

    expected_count = int(longest["sample_count"])

    if len(core) != expected_count:
        raise RuntimeError(
            "Longest-segment row count does not match the audit table.\n"
            f"Audit sample_count = {expected_count}\n"
            f"Rows selected       = {len(core)}\n"
            "Do not continue until the audit/source mismatch is resolved."
        )

    dt = core["time"].diff().dt.total_seconds().div(60.0)
    bad_transition = (
        dt.iloc[1:].isna()
        | (np.abs(dt.iloc[1:] - EXPECTED_DT_MIN) > 1e-9)
    )

    if bad_transition.any():
        raise RuntimeError(
            f"Longest segment contains {int(bad_transition.sum())} non-1-min transitions."
        )

    meta = {
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "sample_count": expected_count,
        "duration_days": float(expected_count / 1440.0),
    }

    return core, meta


def encode_features(core: pd.DataFrame) -> pd.DataFrame:
    """
    Create the frozen v0.1 feature table.

    Circular variables are encoded before any standardization:
        theta -> sin(theta), cos(theta)
    """
    out = pd.DataFrame(index=core.index)
    out["time"] = core["time"]

    numeric_direct = [
        "UWND_MEAN",
        "VWND_MEAN",
        "TEMP_AIR_MEAN",
        "RH_MEAN",
        "SOG",
    ]

    for name in numeric_direct:
        out[name] = pd.to_numeric(core[name], errors="coerce")

    cog_deg = pd.to_numeric(core["COG"], errors="coerce")
    hdg_deg = pd.to_numeric(core["HDG"], errors="coerce")

    cog_rad = np.deg2rad(cog_deg.to_numpy(dtype=np.float64))
    hdg_rad = np.deg2rad(hdg_deg.to_numpy(dtype=np.float64))

    out["COG_sin"] = np.sin(cog_rad)
    out["COG_cos"] = np.cos(cog_rad)
    out["HDG_sin"] = np.sin(hdg_rad)
    out["HDG_cos"] = np.cos(hdg_rad)

    return out


def chronological_split_indices(
    n_rows: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, tuple[int, int]]:
    """
    Return half-open raw-row index intervals [start, end).

    Rounding:
    - train uses floor(n * train_fraction)
    - validation uses floor(n * val_fraction)
    - test receives the remainder

    This keeps all rows while preserving chronology.
    """
    n_train = int(math.floor(n_rows * train_fraction))
    n_val = int(math.floor(n_rows * val_fraction))
    n_test = n_rows - n_train - n_val

    if min(n_train, n_val, n_test) <= 0:
        raise RuntimeError(
            f"Invalid split sizes: train={n_train}, val={n_val}, test={n_test}"
        )

    return {
        "train": (0, n_train),
        "validation": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_rows),
    }


def compute_scaler(
    values: np.ndarray,
    valid_rows: np.ndarray,
    names: list[str],
) -> dict:
    """
    Fit mean/std using valid TRAIN rows only.

    Parameters
    ----------
    values : [rows, variables]
    valid_rows : [rows]
    """
    if values.ndim != 2:
        raise ValueError("Scaler input must be 2D.")

    valid_values = values[valid_rows]

    if len(valid_values) == 0:
        raise RuntimeError("No valid training rows available for scaler fitting.")

    mean = np.mean(valid_values, axis=0, dtype=np.float64)
    std = np.std(valid_values, axis=0, dtype=np.float64, ddof=0)

    zero_std = ~np.isfinite(std) | (std <= 0.0)
    if zero_std.any():
        bad_names = [names[i] for i in np.flatnonzero(zero_std)]
        raise RuntimeError(
            "Cannot standardize zero/non-finite variance variables: "
            + ", ".join(bad_names)
        )

    return {
        "names": list(names),
        "mean": mean.astype(np.float64),
        "std": std.astype(np.float64),
        "n_fit_rows": int(valid_rows.sum()),
    }


def scaler_to_jsonable(scaler: dict) -> dict:
    return {
        "names": scaler["names"],
        "mean": [float(v) for v in scaler["mean"]],
        "std": [float(v) for v in scaler["std"]],
        "n_fit_rows": int(scaler["n_fit_rows"]),
        "definition": "z = (x - mean) / std",
        "fit_scope": "training split only",
    }


def valid_feature_row_mask(feature_values: np.ndarray) -> np.ndarray:
    return np.isfinite(feature_values).all(axis=1)


def valid_target_row_mask(target_values: np.ndarray) -> np.ndarray:
    return np.isfinite(target_values).all(axis=1)


def build_split_dataset(
    split_name: str,
    feature_values_raw: np.ndarray,
    target_values_raw: np.ndarray,
    times_ns: np.ndarray,
    lookback: int,
    horizons: tuple[int, ...],
    feature_scaler: dict,
    target_scaler: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    """
    Build windows INSIDE one raw split only.

    No window can cross a split boundary because this function receives only
    the rows belonging to that split.
    """
    n = len(feature_values_raw)
    max_h = max(horizons)

    minimum_rows = lookback + max_h
    if n < minimum_rows:
        raise RuntimeError(
            f"{split_name}: only {n} raw rows, but at least {minimum_rows} are required."
        )

    f_valid = valid_feature_row_mask(feature_values_raw)
    y_valid = valid_target_row_mask(target_values_raw)

    # Prefix sums allow O(1) validity checks for every input window.
    invalid_f = (~f_valid).astype(np.int64)
    prefix = np.r_[0, np.cumsum(invalid_f)]

    # Anchor is the last row of the input window.
    anchors_all = np.arange(
        lookback - 1,
        n - max_h,
        dtype=np.int64,
    )

    window_start = anchors_all - lookback + 1
    window_end_exclusive = anchors_all + 1

    input_invalid_count = (
        prefix[window_end_exclusive] - prefix[window_start]
    )
    input_ok = input_invalid_count == 0

    horizons_arr = np.asarray(horizons, dtype=np.int64)
    target_idx_all = anchors_all[:, None] + horizons_arr[None, :]
    target_ok = y_valid[target_idx_all].all(axis=1)

    accepted_mask = input_ok & target_ok
    anchors = anchors_all[accepted_mask]
    target_idx = target_idx_all[accepted_mask]

    rejected_input = int((~input_ok).sum())
    rejected_target = int((input_ok & ~target_ok).sum())
    accepted = int(accepted_mask.sum())

    if accepted == 0:
        raise RuntimeError(f"{split_name}: zero valid forecasting windows were created.")

    # Standardize rows before gathering windows.
    f_mean = feature_scaler["mean"].astype(np.float64)
    f_std = feature_scaler["std"].astype(np.float64)
    y_mean = target_scaler["mean"].astype(np.float64)
    y_std = target_scaler["std"].astype(np.float64)

    feature_z = (
        (feature_values_raw.astype(np.float64) - f_mean) / f_std
    ).astype(np.float32)

    target_z_rows = (
        (target_values_raw.astype(np.float64) - y_mean) / y_std
    ).astype(np.float32)

    # Gather input windows. Shape: [samples, lookback, features].
    offsets = np.arange(lookback - 1, -1, -1, dtype=np.int64)
    input_indices = anchors[:, None] - offsets[None, :]

    X = feature_z[input_indices]
    y = target_z_rows[target_idx]
    y_raw = target_values_raw[target_idx].astype(np.float32)

    context_end_time_ns = times_ns[anchors].astype(np.int64)
    target_time_ns = times_ns[target_idx].astype(np.int64)

    result = {
        "X": np.ascontiguousarray(X, dtype=np.float32),
        "y": np.ascontiguousarray(y, dtype=np.float32),
        "y_raw": np.ascontiguousarray(y_raw, dtype=np.float32),
        "context_end_time_ns": np.ascontiguousarray(
            context_end_time_ns,
            dtype=np.int64,
        ),
        "target_time_ns": np.ascontiguousarray(
            target_time_ns,
            dtype=np.int64,
        ),
        "anchor_row_index": np.ascontiguousarray(
            anchors,
            dtype=np.int64,
        ),
    }

    report = {
        "split": split_name,
        "raw_rows": int(n),
        "candidate_windows": int(len(anchors_all)),
        "accepted_windows": accepted,
        "rejected_windows_total": int(len(anchors_all) - accepted),
        "rejected_due_to_input_missing": rejected_input,
        "rejected_due_to_target_missing_after_valid_input": rejected_target,
        "raw_valid_feature_rows": int(f_valid.sum()),
        "raw_invalid_feature_rows": int((~f_valid).sum()),
        "raw_valid_target_rows": int(y_valid.sum()),
        "raw_invalid_target_rows": int((~y_valid).sum()),
    }

    return result, report


def save_npz(path: Path, arrays: dict[str, np.ndarray], compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def write_feature_definition(output_file: Path) -> None:
    rows = [
        {
            "order": 1,
            "feature": "UWND_MEAN",
            "source": "UWND_MEAN",
            "transform": "none",
            "role": "wind history",
        },
        {
            "order": 2,
            "feature": "VWND_MEAN",
            "source": "VWND_MEAN",
            "transform": "none",
            "role": "wind history",
        },
        {
            "order": 3,
            "feature": "TEMP_AIR_MEAN",
            "source": "TEMP_AIR_MEAN",
            "transform": "none",
            "role": "meteorological context",
        },
        {
            "order": 4,
            "feature": "RH_MEAN",
            "source": "RH_MEAN",
            "transform": "none",
            "role": "meteorological context",
        },
        {
            "order": 5,
            "feature": "SOG",
            "source": "SOG",
            "transform": "none",
            "role": "platform state",
        },
        {
            "order": 6,
            "feature": "COG_sin",
            "source": "COG",
            "transform": "sin(deg2rad(COG))",
            "role": "platform course",
        },
        {
            "order": 7,
            "feature": "COG_cos",
            "source": "COG",
            "transform": "cos(deg2rad(COG))",
            "role": "platform course",
        },
        {
            "order": 8,
            "feature": "HDG_sin",
            "source": "HDG",
            "transform": "sin(deg2rad(HDG))",
            "role": "platform heading",
        },
        {
            "order": 9,
            "feature": "HDG_cos",
            "source": "HDG",
            "transform": "cos(deg2rad(HDG))",
            "role": "platform heading",
        },
    ]

    pd.DataFrame(rows).to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )


def load_sci_style(script_dir: Path):
    style_path = script_dir / "sci_plot_style.py"

    if not style_path.exists():
        log(
            f"[PLOT WARNING] sci_plot_style.py not found at {style_path}; "
            "dataset construction is unaffected."
        )
        return None

    spec = importlib.util.spec_from_file_location(
        "sci_plot_style",
        str(style_path),
    )

    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules["sci_plot_style"] = module
    spec.loader.exec_module(module)

    if hasattr(module, "apply_sci_style"):
        try:
            selected = module.apply_sci_style(base_font_size=8.5)
        except TypeError:
            try:
                selected = module.apply_sci_style(font_size=8.5)
            except TypeError:
                selected = module.apply_sci_style()

        log(f"[PLOT] SCI style applied; selected_font={selected}")

    return module


def make_plots(
    core: pd.DataFrame,
    split_intervals: dict[str, tuple[int, int]],
    split_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    style = load_sci_style(Path(__file__).resolve().parent)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def finish(ax, grid=True):
        if style is not None and hasattr(style, "clean_axis"):
            style.clean_axis(ax, grid=grid)
        elif style is not None and hasattr(style, "finish_axes"):
            style.finish_axes(ax, grid=grid)
        else:
            ax.tick_params(which="both", direction="in", top=True, right=True)
            if grid:
                ax.grid(True, alpha=0.25)

    def save(fig, name):
        base = fig_dir / name
        if style is not None and hasattr(style, "save_figure"):
            style.save_figure(fig, base)
        else:
            fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
            fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)

    # Figure 1: time split overview over U.
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(
        core["time"],
        core["UWND_MEAN"],
        linewidth=0.65,
        label=r"$U$",
    )

    span_alpha = 0.10
    split_labels = {
        "train": "Train",
        "validation": "Validation",
        "test": "Test",
    }

    for split_name, (a, b) in split_intervals.items():
        start = core["time"].iloc[a]
        end = core["time"].iloc[b - 1]
        ax.axvspan(
            start,
            end,
            alpha=span_alpha,
            label=split_labels[split_name],
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$U$ (m s$^{-1}$)")
    ax.legend(loc="best", ncol=4)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    finish(ax, grid=True)
    save(fig, "01_chronological_split")

    # Figure 2: accepted windows by split.
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    names = ["train", "validation", "test"]
    values = [
        int(
            split_summary.loc[
                split_summary["split"] == name,
                "accepted_windows",
            ].iloc[0]
        )
        for name in names
    ]
    ax.bar(["Train", "Validation", "Test"], values)
    ax.set_ylabel("Accepted windows")
    finish(ax, grid=True)
    save(fig, "02_window_counts")


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audit-dir",
        required=True,
        help=(
            "Directory produced by 01_audit_saildrone_wind_v5_sci.py, "
            "containing selected_variables_raw.csv and joint_wind_valid_segments.csv."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output forecasting-dataset directory.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK,
        help=f"Input history length in minutes. Default: {DEFAULT_LOOKBACK}.",
    )
    parser.add_argument(
        "--horizons",
        default=",".join(map(str, DEFAULT_HORIZONS)),
        help="Comma-separated future horizons in minutes. Default: 1,2,3,5,10.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Use uncompressed NPZ for faster writing at the cost of larger files.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate SCI-style dataset split/QC figures.",
    )

    args = parser.parse_args()

    if args.lookback <= 0:
        raise ValueError("--lookback must be a positive integer.")

    horizons = parse_horizons(args.horizons)

    check_split_fractions(
        args.train_fraction,
        args.validation_fraction,
        args.test_fraction,
    )

    audit_dir = Path(args.audit_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("SAILDRONE FORECASTING DATASET BUILDER")
    log("=" * 80)
    log(f"dataset_version : {DATASET_VERSION}")
    log(f"audit_dir       : {audit_dir}")
    log(f"output_dir      : {output_dir}")
    log(f"lookback        : {args.lookback} min")
    log(f"horizons        : {list(horizons)} min")
    log(
        "split fractions : "
        f"{args.train_fraction:.2f}/"
        f"{args.validation_fraction:.2f}/"
        f"{args.test_fraction:.2f}"
    )
    log("missing policy  : no interpolation; reject affected windows")
    log("split policy    : split raw time first, build windows second")
    log("")

    # ------------------------------------------------------------------
    # Stage 1: read audit outputs and select longest valid U/V segment.
    # ------------------------------------------------------------------
    log("[STAGE 1/8] Reading audited data")
    raw, segments = read_audit_inputs(audit_dir)
    core, core_meta = choose_longest_segment(raw, segments)

    log(
        "[OK] Longest continuous U/V segment: "
        f"{core_meta['start_time']} -> {core_meta['end_time']}"
    )
    log(f"[OK] Rows: {core_meta['sample_count']}")
    log(f"[OK] Duration: {core_meta['duration_days']:.3f} days")

    # ------------------------------------------------------------------
    # Stage 2: feature engineering.
    # ------------------------------------------------------------------
    log("[STAGE 2/8] Creating frozen v0.1 features")
    feature_df = encode_features(core)

    feature_values = feature_df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    target_values = core[TARGET_NAMES].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    times = pd.to_datetime(core["time"])

    # IMPORTANT:
    # pandas may internally use datetime64[us] on newer versions. Calling
    # astype("int64") directly would then return microseconds while the array
    # name says "_ns". Force NumPy datetime64[ns] first so stored integer
    # timestamps are always true Unix nanoseconds.
    times_ns = (
        times.to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )

    log(f"[OK] Feature count: {len(FEATURE_NAMES)}")
    for i, name in enumerate(FEATURE_NAMES, start=1):
        log(f"     {i:02d}. {name}")

    # ------------------------------------------------------------------
    # Stage 3: strict chronological raw-row split.
    # ------------------------------------------------------------------
    log("[STAGE 3/8] Chronological raw-row split")

    intervals = chronological_split_indices(
        len(core),
        args.train_fraction,
        args.validation_fraction,
        args.test_fraction,
    )

    split_rows = []

    for name, (a, b) in intervals.items():
        split_rows.append({
            "split": name,
            "raw_start_index": a,
            "raw_end_index_exclusive": b,
            "raw_rows": b - a,
            "start_time": times.iloc[a],
            "end_time": times.iloc[b - 1],
        })

        log(
            f"[SPLIT] {name:10s} rows={b-a:6d}  "
            f"{times.iloc[a]} -> {times.iloc[b-1]}"
        )

    # ------------------------------------------------------------------
    # Stage 4: train-only scalers.
    # ------------------------------------------------------------------
    log("[STAGE 4/8] Fitting train-only scalers")

    train_a, train_b = intervals["train"]
    train_feature_values = feature_values[train_a:train_b]
    train_target_values = target_values[train_a:train_b]

    train_feature_valid = np.isfinite(train_feature_values).all(axis=1)
    train_target_valid = np.isfinite(train_target_values).all(axis=1)

    feature_scaler = compute_scaler(
        train_feature_values,
        train_feature_valid,
        FEATURE_NAMES,
    )

    target_scaler = compute_scaler(
        train_target_values,
        train_target_valid,
        TARGET_NAMES,
    )

    log(
        f"[OK] Feature scaler rows: {feature_scaler['n_fit_rows']}"
    )
    log(
        f"[OK] Target scaler rows : {target_scaler['n_fit_rows']}"
    )

    # ------------------------------------------------------------------
    # Stage 5: construct each split independently.
    # ------------------------------------------------------------------
    log("[STAGE 5/8] Building leakage-safe forecasting windows")

    dataset_reports = []
    saved_paths = {}

    compress = not args.no_compress

    for split_name in ["train", "validation", "test"]:
        a, b = intervals[split_name]

        split_feature = feature_values[a:b]
        split_target = target_values[a:b]
        split_time_ns = times_ns[a:b]

        arrays, report = build_split_dataset(
            split_name=split_name,
            feature_values_raw=split_feature,
            target_values_raw=split_target,
            times_ns=split_time_ns,
            lookback=args.lookback,
            horizons=horizons,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
        )

        # anchor_row_index is currently local to the split; convert to core index.
        arrays["anchor_row_index"] = (
            arrays["anchor_row_index"] + a
        ).astype(np.int64)

        out_file = output_dir / f"{split_name}.npz"

        log(
            f"[SAVE] {split_name:10s} "
            f"X={arrays['X'].shape} "
            f"y={arrays['y'].shape} "
            f"-> {out_file.name}"
        )

        save_npz(
            out_file,
            arrays,
            compressed=compress,
        )

        saved_paths[split_name] = str(out_file)
        dataset_reports.append(report)

    # ------------------------------------------------------------------
    # Stage 6: metadata tables.
    # ------------------------------------------------------------------
    log("[STAGE 6/8] Writing metadata")

    reports_df = pd.DataFrame(dataset_reports)
    split_meta_df = pd.DataFrame(split_rows)
    split_summary = split_meta_df.merge(
        reports_df,
        on=["split", "raw_rows"],
        how="left",
    )

    split_summary.to_csv(
        output_dir / "split_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rejected_df = reports_df[
        [
            "split",
            "candidate_windows",
            "accepted_windows",
            "rejected_windows_total",
            "rejected_due_to_input_missing",
            "rejected_due_to_target_missing_after_valid_input",
        ]
    ].copy()

    rejected_df.to_csv(
        output_dir / "rejected_windows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_feature_definition(
        output_dir / "feature_definition.csv"
    )

    scaler_payload = {
        "dataset_version": DATASET_VERSION,
        "feature_scaler": scaler_to_jsonable(feature_scaler),
        "target_scaler": scaler_to_jsonable(target_scaler),
    }

    with (output_dir / "scalers.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            scaler_payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Stage 7: manifest and human-readable report.
    # ------------------------------------------------------------------
    log("[STAGE 7/8] Writing manifest and report")

    manifest = {
        "dataset_version": DATASET_VERSION,
        "source_audit_dir": str(audit_dir.resolve()),
        "source_selected_variables_raw": str(
            (audit_dir / "selected_variables_raw.csv").resolve()
        ),
        "source_joint_valid_segments": str(
            (audit_dir / "joint_wind_valid_segments.csv").resolve()
        ),
        "longest_segment": core_meta,
        "sample_interval_minutes": 1,
        "timestamp_integer_unit": "ns",
        "timestamp_definition": "Unix nanoseconds since 1970-01-01T00:00:00Z",
        "lookback_minutes": int(args.lookback),
        "forecast_horizons_minutes": list(horizons),
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "target_definition": "future horizontal wind vector components U and V",
        "angle_encoding": {
            "COG": ["COG_sin", "COG_cos"],
            "HDG": ["HDG_sin", "HDG_cos"],
            "formula": "sin/cos of angle in radians",
        },
        "missing_value_policy": (
            "No interpolation. Candidate windows are rejected if any input "
            "feature row or any requested target contains NaN/Inf."
        ),
        "split_policy": (
            "Chronological raw-row split first; windows are then built independently "
            "inside train/validation/test so no window crosses a split boundary."
        ),
        "split_fractions": {
            "train": float(args.train_fraction),
            "validation": float(args.validation_fraction),
            "test": float(args.test_fraction),
        },
        "standardization_policy": (
            "Feature and target mean/std are fitted using training split only."
        ),
        "npz_compressed": bool(compress),
        "array_shapes": {
            row["split"]: {
                "accepted_windows": int(row["accepted_windows"]),
                "X": [
                    int(row["accepted_windows"]),
                    int(args.lookback),
                    len(FEATURE_NAMES),
                ],
                "y": [
                    int(row["accepted_windows"]),
                    len(horizons),
                    len(TARGET_NAMES),
                ],
            }
            for row in dataset_reports
        },
        "files": {
            "train": "train.npz",
            "validation": "validation.npz",
            "test": "test.npz",
            "scalers": "scalers.json",
            "split_summary": "split_summary.csv",
            "feature_definition": "feature_definition.csv",
            "rejected_windows": "rejected_windows.csv",
        },
    }

    with (output_dir / "dataset_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with (output_dir / "dataset_build_report.txt").open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("Saildrone forecasting dataset build report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset version: {DATASET_VERSION}\n")
        f.write(
            f"Longest valid segment: "
            f"{core_meta['start_time']} -> {core_meta['end_time']}\n"
        )
        f.write(f"Rows in segment: {core_meta['sample_count']}\n")
        f.write(f"Duration days: {core_meta['duration_days']:.6f}\n")
        f.write(f"Lookback: {args.lookback} min\n")
        f.write(f"Horizons: {list(horizons)} min\n\n")

        f.write("Features\n")
        f.write("-" * 80 + "\n")
        for i, name in enumerate(FEATURE_NAMES, start=1):
            f.write(f"{i:02d}. {name}\n")

        f.write("\nTargets\n")
        f.write("-" * 80 + "\n")
        for i, name in enumerate(TARGET_NAMES, start=1):
            f.write(f"{i:02d}. {name}\n")

        f.write("\nData leakage controls\n")
        f.write("-" * 80 + "\n")
        f.write(
            "1. The original continuous time series is split chronologically "
            "before window construction.\n"
        )
        f.write(
            "2. Train/validation/test windows are built independently and "
            "cannot cross split boundaries.\n"
        )
        f.write(
            "3. Feature and target scalers use training data only.\n"
        )
        f.write(
            "4. Missing values are not interpolated; affected windows are rejected.\n"
        )

        f.write("\nSplit summary\n")
        f.write("-" * 80 + "\n")
        f.write(split_summary.to_string(index=False))
        f.write("\n")

    # ------------------------------------------------------------------
    # Stage 8: optional SCI figures + final report.
    # ------------------------------------------------------------------
    log("[STAGE 8/8] Finalization")

    if args.plots:
        log("[PLOT] Building SCI-style split/QC figures")
        make_plots(
            core=core,
            split_intervals=intervals,
            split_summary=split_summary,
            output_dir=output_dir,
        )
        log("[OK] Figures complete")

    log("")
    log("=" * 80)
    log("FORECASTING DATASET COMPLETE")
    log("=" * 80)

    for _, row in split_summary.iterrows():
        log(
            f"{row['split']:10s}: "
            f"raw_rows={int(row['raw_rows']):6d}, "
            f"candidate={int(row['candidate_windows']):6d}, "
            f"accepted={int(row['accepted_windows']):6d}, "
            f"rejected={int(row['rejected_windows_total']):4d}"
        )

    log("")
    log(f"Feature shape per sample : ({args.lookback}, {len(FEATURE_NAMES)})")
    log(
        f"Target shape per sample  : ({len(horizons)}, {len(TARGET_NAMES)})"
    )
    log(f"Output directory         : {output_dir}")
    log("[DONE] No-leak forecasting dataset built successfully.")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception:
        print("\n[FATAL ERROR]", flush=True)
        traceback.print_exc()
        print(
            "\nPlease send the complete traceback and terminal output.",
            flush=True,
        )
        sys.exit(1)

    sys.exit(exit_code)
