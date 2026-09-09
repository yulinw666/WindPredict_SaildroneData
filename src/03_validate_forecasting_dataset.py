# -*- coding: utf-8 -*-
"""
03_validate_forecasting_dataset.py

Independent integrity validator for the Saildrone forecasting dataset produced by
02_build_forecasting_dataset.py.

Validation goals
----------------
A. File / schema integrity
   - required files exist
   - train/validation/test NPZ arrays exist
   - shapes, dtypes, sample counts, feature count and horizon count are consistent

B. Numerical integrity
   - X, y, y_raw contain no NaN/Inf
   - standardized y matches y_raw through scalers.json
   - train-only scaler statistics are finite and non-degenerate

C. Temporal integrity
   - context times are strictly increasing
   - target times match context_end + [1,2,3,5,10] min exactly
   - anchor_row_index matches source timestamps
   - no window crosses train/validation/test raw-time boundaries
   - train/validation/test use disjoint chronological raw-row ranges

D. Source reconstruction
   When the source audit CSV referenced by dataset_manifest.json is available:
   - reconstruct the frozen v0.1 features from selected_variables_raw.csv
   - recompute train-only feature/target scalers and compare against scalers.json
   - verify ALL y_raw values against the audited source
   - verify ALL target timestamps against the audited source
   - verify a deterministic sample of X windows against reconstructed,
     train-standardized raw inputs

E. Circular encoding
   - verify raw COG_sin/cos and HDG_sin/cos obey sin^2 + cos^2 = 1
     wherever the source angle is valid

Outputs
-------
validation_report.txt
validation_checks.csv
split_integrity.csv
standardization_check.csv
source_reconstruction_check.csv
validation_summary.json
VALIDATION_PASSED.flag       (only when all critical checks pass)
figures/*.png + figures/*.pdf  (when --plots is supplied)

Usage
-----
python 03_validate_forecasting_dataset.py ^
    --dataset-dir "D:\\...\\SD1090_TPOS2024_Forecasting_v0_1" ^
    --output "D:\\...\\SD1090_TPOS2024_Forecasting_v0_1\\validation_v0_1" ^
    --plots
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


VALIDATOR_VERSION = "0.1.1"
EXPECTED_DT_MIN = 1
REQUIRED_SPLITS = ("train", "validation", "test")

REQUIRED_NPZ_KEYS = (
    "X",
    "y",
    "y_raw",
    "context_end_time_ns",
    "target_time_ns",
    "anchor_row_index",
)

FROZEN_FEATURE_NAMES = [
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

FROZEN_TARGET_NAMES = [
    "UWND_MEAN",
    "VWND_MEAN",
]


def log(message: str = "") -> None:
    print(message, flush=True)


@dataclass
class Check:
    category: str
    check: str
    status: str
    severity: str
    value: str
    expected: str
    details: str = ""


class Recorder:
    def __init__(self) -> None:
        self.rows: list[Check] = []

    def add(
        self,
        category: str,
        check: str,
        passed: bool,
        value,
        expected,
        details: str = "",
        severity: str = "critical",
    ) -> None:
        row = Check(
            category=category,
            check=check,
            status="PASS" if passed else "FAIL",
            severity=severity,
            value=str(value),
            expected=str(expected),
            details=details,
        )
        self.rows.append(row)

        prefix = "[PASS]" if passed else "[FAIL]"
        log(f"{prefix} {category}: {check} | value={value} | expected={expected}")

    def warn(
        self,
        category: str,
        check: str,
        value,
        expected,
        details: str = "",
    ) -> None:
        row = Check(
            category=category,
            check=check,
            status="WARN",
            severity="warning",
            value=str(value),
            expected=str(expected),
            details=details,
        )
        self.rows.append(row)
        log(f"[WARN] {category}: {check} | value={value} | expected={expected}")

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.rows])

    def critical_failures(self) -> int:
        return sum(
            r.status == "FAIL" and r.severity == "critical"
            for r in self.rows
        )

    def warnings(self) -> int:
        return sum(r.status == "WARN" for r in self.rows)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def parse_manifest(dataset_dir: Path) -> tuple[dict, dict, pd.DataFrame]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    scalers_path = dataset_dir / "scalers.json"
    split_summary_path = dataset_dir / "split_summary.csv"

    require_file(manifest_path)
    require_file(scalers_path)
    require_file(split_summary_path)

    manifest = load_json(manifest_path)
    scalers = load_json(scalers_path)
    split_summary = pd.read_csv(split_summary_path)

    return manifest, scalers, split_summary


def parse_scaler(scalers: dict, key: str) -> tuple[list[str], np.ndarray, np.ndarray, int]:
    obj = scalers[key]
    names = list(obj["names"])
    mean = np.asarray(obj["mean"], dtype=np.float64)
    std = np.asarray(obj["std"], dtype=np.float64)
    n_fit_rows = int(obj["n_fit_rows"])
    return names, mean, std, n_fit_rows


def ensure_expected_split_summary(split_summary: pd.DataFrame) -> None:
    needed = [
        "split",
        "raw_start_index",
        "raw_end_index_exclusive",
        "raw_rows",
        "start_time",
        "end_time",
        "candidate_windows",
        "accepted_windows",
        "rejected_windows_total",
    ]
    missing = [c for c in needed if c not in split_summary.columns]
    if missing:
        raise KeyError(
            "split_summary.csv is missing columns: " + ", ".join(missing)
        )


def candidate_window_count(raw_rows: int, lookback: int, max_h: int) -> int:
    return raw_rows - lookback - max_h + 1


def build_core_source(manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_path = Path(manifest["source_selected_variables_raw"])
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    raw = pd.read_csv(source_path, low_memory=False)
    raw["time"] = pd.to_datetime(raw["time"], errors="coerce")

    segment = manifest["longest_segment"]
    start = pd.Timestamp(segment["start_time"])
    end = pd.Timestamp(segment["end_time"])

    core = raw.loc[
        (raw["time"] >= start) & (raw["time"] <= end)
    ].copy()

    core = core.sort_values("time").reset_index(drop=True)

    required = [
        "time",
        "UWND_MEAN",
        "VWND_MEAN",
        "TEMP_AIR_MEAN",
        "RH_MEAN",
        "SOG",
        "COG",
        "HDG",
    ]
    missing = [c for c in required if c not in core.columns]
    if missing:
        raise KeyError(
            "Source audit table is missing frozen v0.1 columns: "
            + ", ".join(missing)
        )

    features = pd.DataFrame(index=core.index)
    features["UWND_MEAN"] = pd.to_numeric(core["UWND_MEAN"], errors="coerce")
    features["VWND_MEAN"] = pd.to_numeric(core["VWND_MEAN"], errors="coerce")
    features["TEMP_AIR_MEAN"] = pd.to_numeric(
        core["TEMP_AIR_MEAN"], errors="coerce"
    )
    features["RH_MEAN"] = pd.to_numeric(core["RH_MEAN"], errors="coerce")
    features["SOG"] = pd.to_numeric(core["SOG"], errors="coerce")

    cog = np.deg2rad(pd.to_numeric(core["COG"], errors="coerce").to_numpy(float))
    hdg = np.deg2rad(pd.to_numeric(core["HDG"], errors="coerce").to_numpy(float))

    features["COG_sin"] = np.sin(cog)
    features["COG_cos"] = np.cos(cog)
    features["HDG_sin"] = np.sin(hdg)
    features["HDG_cos"] = np.cos(hdg)

    return core, features


def compute_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    valid = np.isfinite(values).all(axis=1)
    selected = values[valid]

    if len(selected) == 0:
        raise RuntimeError("No valid rows available to recompute scaler.")

    mean = np.mean(selected, axis=0, dtype=np.float64)
    std = np.std(selected, axis=0, dtype=np.float64, ddof=0)
    return mean, std, int(valid.sum())


def load_sci_style(script_dir: Path):
    style_path = script_dir / "sci_plot_style.py"

    if not style_path.exists():
        log(f"[PLOT WARNING] sci_plot_style.py not found: {style_path}")
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


def make_validation_plots(
    output_dir: Path,
    split_integrity: pd.DataFrame,
    train_feature_stats: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

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

    # Accepted sample counts.
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    order = ["train", "validation", "test"]
    values = [
        int(
            split_integrity.loc[
                split_integrity["split"] == s,
                "samples",
            ].iloc[0]
        )
        for s in order
    ]
    ax.bar(["Train", "Validation", "Test"], values)
    ax.set_ylabel("Forecasting samples")
    finish(ax, grid=True)
    save(fig, "01_validated_sample_counts")

    # Standardized train-X mean by feature.
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.bar(
        np.arange(len(train_feature_stats)),
        train_feature_stats["window_weighted_mean"].to_numpy(),
    )
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xticks(np.arange(len(train_feature_stats)))
    ax.set_xticklabels(
        train_feature_stats["feature"].tolist(),
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("Mean of standardized train $X$")
    finish(ax, grid=True)
    save(fig, "02_train_standardized_feature_means")

    # Standardized train-X std by feature.
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.bar(
        np.arange(len(train_feature_stats)),
        train_feature_stats["window_weighted_std"].to_numpy(),
    )
    ax.axhline(1.0, linewidth=0.8)
    ax.set_xticks(np.arange(len(train_feature_stats)))
    ax.set_xticklabels(
        train_feature_stats["feature"].tolist(),
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("Std. of standardized train $X$")
    finish(ax, grid=True)
    save(fig, "03_train_standardized_feature_stds")


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory created by 02_build_forecasting_dataset.py",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Validation output directory. Default: "
            "<dataset-dir>/validation_v0_1"
        ),
    )
    parser.add_argument(
        "--x-check-samples",
        type=int,
        default=256,
        help="Number of deterministic X windows checked per split against source.",
    )
    parser.add_argument(
        "--require-source",
        action="store_true",
        help="Fail if the audited source CSV referenced by the manifest is unavailable.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate SCI-style validation plots.",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = (
        Path(args.output)
        if args.output is not None
        else dataset_dir / "validation_v0_1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    recorder = Recorder()

    log("=" * 88)
    log("SAILDRONE FORECASTING DATASET VALIDATOR")
    log("=" * 88)
    log(f"validator_version : {VALIDATOR_VERSION}")
    log(f"dataset_dir       : {dataset_dir}")
    log(f"output_dir        : {output_dir}")
    log("")

    # ------------------------------------------------------------------
    # Stage 1: metadata and configuration.
    # ------------------------------------------------------------------
    log("[STAGE 1/8] Reading manifest, scalers, and split metadata")
    manifest, scalers, split_summary = parse_manifest(dataset_dir)
    ensure_expected_split_summary(split_summary)

    lookback = int(manifest["lookback_minutes"])
    horizons = tuple(int(v) for v in manifest["forecast_horizons_minutes"])
    max_h = max(horizons)
    feature_names = list(manifest["feature_names"])
    target_names = list(manifest["target_names"])
    interval_min = int(manifest["sample_interval_minutes"])

    f_names, f_mean, f_std, f_n_fit = parse_scaler(
        scalers,
        "feature_scaler",
    )
    y_names, y_mean, y_std, y_n_fit = parse_scaler(
        scalers,
        "target_scaler",
    )

    recorder.add(
        "metadata",
        "sample interval",
        interval_min == EXPECTED_DT_MIN,
        interval_min,
        EXPECTED_DT_MIN,
    )

    timestamp_unit = manifest.get("timestamp_integer_unit", None)
    recorder.add(
        "metadata",
        "timestamp integer unit",
        timestamp_unit == "ns",
        timestamp_unit,
        "ns",
        details="context_end_time_ns and target_time_ns must store Unix nanoseconds",
    )
    recorder.add(
        "metadata",
        "feature names match frozen v0.1",
        feature_names == FROZEN_FEATURE_NAMES,
        feature_names,
        FROZEN_FEATURE_NAMES,
    )
    recorder.add(
        "metadata",
        "target names match frozen v0.1",
        target_names == FROZEN_TARGET_NAMES,
        target_names,
        FROZEN_TARGET_NAMES,
    )
    recorder.add(
        "metadata",
        "feature scaler names match manifest",
        f_names == feature_names,
        f_names,
        feature_names,
    )
    recorder.add(
        "metadata",
        "target scaler names match manifest",
        y_names == target_names,
        y_names,
        target_names,
    )
    recorder.add(
        "metadata",
        "feature scaler finite",
        bool(np.isfinite(f_mean).all() and np.isfinite(f_std).all()),
        "finite" if np.isfinite(f_mean).all() and np.isfinite(f_std).all() else "non-finite",
        "all finite",
    )
    recorder.add(
        "metadata",
        "target scaler finite",
        bool(np.isfinite(y_mean).all() and np.isfinite(y_std).all()),
        "finite" if np.isfinite(y_mean).all() and np.isfinite(y_std).all() else "non-finite",
        "all finite",
    )
    recorder.add(
        "metadata",
        "feature scaler nonzero std",
        bool((f_std > 0.0).all()),
        float(np.min(f_std)),
        "> 0",
    )
    recorder.add(
        "metadata",
        "target scaler nonzero std",
        bool((y_std > 0.0).all()),
        float(np.min(y_std)),
        "> 0",
    )

    # ------------------------------------------------------------------
    # Stage 2: source reconstruction and train-only scaler verification.
    # ------------------------------------------------------------------
    log("[STAGE 2/8] Reconstructing audited source and train-only scalers")

    source_available = False
    core = None
    source_features = None
    source_targets = None
    source_times_ns = None

    source_check_rows = []

    try:
        core, source_features = build_core_source(manifest)
        source_available = True

        source_feature_values = source_features[
            feature_names
        ].to_numpy(dtype=np.float64)

        source_targets = core[
            target_names
        ].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        # Force the source time vector to true Unix nanoseconds.
        # Do not rely on pandas' internal datetime resolution (which may be us).
        source_times_ns = (
            pd.to_datetime(core["time"])
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )

        expected_core_rows = int(manifest["longest_segment"]["sample_count"])

        recorder.add(
            "source",
            "core row count matches manifest",
            len(core) == expected_core_rows,
            len(core),
            expected_core_rows,
        )

        dt = core["time"].diff().dt.total_seconds().div(60.0)
        non_1min = int(
            (
                dt.iloc[1:].isna()
                | (np.abs(dt.iloc[1:] - EXPECTED_DT_MIN) > 1e-9)
            ).sum()
        )
        recorder.add(
            "source",
            "core time continuity",
            non_1min == 0,
            non_1min,
            0,
            "number of non-1-min transitions",
        )

        train_row = split_summary.loc[
            split_summary["split"] == "train"
        ].iloc[0]
        train_a = int(train_row["raw_start_index"])
        train_b = int(train_row["raw_end_index_exclusive"])

        recompute_f_mean, recompute_f_std, recompute_f_n = compute_scaler(
            source_feature_values[train_a:train_b]
        )
        recompute_y_mean, recompute_y_std, recompute_y_n = compute_scaler(
            source_targets[train_a:train_b]
        )

        f_mean_err = float(np.max(np.abs(recompute_f_mean - f_mean)))
        f_std_err = float(np.max(np.abs(recompute_f_std - f_std)))
        y_mean_err = float(np.max(np.abs(recompute_y_mean - y_mean)))
        y_std_err = float(np.max(np.abs(recompute_y_std - y_std)))

        scaler_tol = 1e-10

        recorder.add(
            "standardization",
            "feature scaler mean is train-only reconstruction",
            f_mean_err <= scaler_tol,
            f"{f_mean_err:.3e}",
            f"<= {scaler_tol:.1e}",
        )
        recorder.add(
            "standardization",
            "feature scaler std is train-only reconstruction",
            f_std_err <= scaler_tol,
            f"{f_std_err:.3e}",
            f"<= {scaler_tol:.1e}",
        )
        recorder.add(
            "standardization",
            "target scaler mean is train-only reconstruction",
            y_mean_err <= scaler_tol,
            f"{y_mean_err:.3e}",
            f"<= {scaler_tol:.1e}",
        )
        recorder.add(
            "standardization",
            "target scaler std is train-only reconstruction",
            y_std_err <= scaler_tol,
            f"{y_std_err:.3e}",
            f"<= {scaler_tol:.1e}",
        )
        recorder.add(
            "standardization",
            "feature scaler fit-row count",
            recompute_f_n == f_n_fit,
            recompute_f_n,
            f_n_fit,
        )
        recorder.add(
            "standardization",
            "target scaler fit-row count",
            recompute_y_n == y_n_fit,
            recompute_y_n,
            y_n_fit,
        )

        # Circular encoding checks on raw engineered rows.
        cog_valid = np.isfinite(
            source_features[["COG_sin", "COG_cos"]].to_numpy(float)
        ).all(axis=1)
        hdg_valid = np.isfinite(
            source_features[["HDG_sin", "HDG_cos"]].to_numpy(float)
        ).all(axis=1)

        cog_norm = (
            source_features.loc[cog_valid, "COG_sin"].to_numpy(float) ** 2
            + source_features.loc[cog_valid, "COG_cos"].to_numpy(float) ** 2
        )
        hdg_norm = (
            source_features.loc[hdg_valid, "HDG_sin"].to_numpy(float) ** 2
            + source_features.loc[hdg_valid, "HDG_cos"].to_numpy(float) ** 2
        )

        cog_err = float(np.max(np.abs(cog_norm - 1.0)))
        hdg_err = float(np.max(np.abs(hdg_norm - 1.0)))

        recorder.add(
            "circular encoding",
            "COG sin/cos unit-circle identity",
            cog_err <= 1e-12,
            f"{cog_err:.3e}",
            "<= 1e-12",
        )
        recorder.add(
            "circular encoding",
            "HDG sin/cos unit-circle identity",
            hdg_err <= 1e-12,
            f"{hdg_err:.3e}",
            "<= 1e-12",
        )

        source_check_rows.extend([
            {
                "check": "feature_scaler_mean_max_abs_error",
                "value": f_mean_err,
            },
            {
                "check": "feature_scaler_std_max_abs_error",
                "value": f_std_err,
            },
            {
                "check": "target_scaler_mean_max_abs_error",
                "value": y_mean_err,
            },
            {
                "check": "target_scaler_std_max_abs_error",
                "value": y_std_err,
            },
            {
                "check": "COG_unit_circle_max_abs_error",
                "value": cog_err,
            },
            {
                "check": "HDG_unit_circle_max_abs_error",
                "value": hdg_err,
            },
        ])

    except FileNotFoundError as exc:
        if args.require_source:
            raise
        recorder.warn(
            "source",
            "source reconstruction available",
            False,
            True,
            details=f"Source path unavailable: {exc}",
        )
        log(
            "[WARN] Source reconstruction skipped. "
            "Structural NPZ validation will continue."
        )

    # ------------------------------------------------------------------
    # Stage 3: raw split chronology and no-overlap.
    # ------------------------------------------------------------------
    log("[STAGE 3/8] Validating chronological raw split boundaries")

    split_order = split_summary.set_index("split").loc[
        list(REQUIRED_SPLITS)
    ].reset_index()

    chronology_ok = True
    for i in range(len(split_order) - 1):
        left = split_order.iloc[i]
        right = split_order.iloc[i + 1]

        left_end = int(left["raw_end_index_exclusive"])
        right_start = int(right["raw_start_index"])

        if left_end != right_start:
            chronology_ok = False

    recorder.add(
        "split chronology",
        "raw split indices are contiguous and disjoint",
        chronology_ok,
        chronology_ok,
        True,
    )

    all_split_rows = int(split_order["raw_rows"].sum())
    expected_total = int(manifest["longest_segment"]["sample_count"])

    recorder.add(
        "split chronology",
        "raw split rows sum to core rows",
        all_split_rows == expected_total,
        all_split_rows,
        expected_total,
    )

    for _, row in split_order.iterrows():
        split_name = row["split"]
        raw_rows = int(row["raw_rows"])
        expected_candidates = candidate_window_count(
            raw_rows,
            lookback,
            max_h,
        )
        actual_candidates = int(row["candidate_windows"])

        recorder.add(
            "split chronology",
            f"{split_name} candidate window formula",
            actual_candidates == expected_candidates,
            actual_candidates,
            expected_candidates,
        )

    # ------------------------------------------------------------------
    # Stage 4-6: validate each NPZ independently.
    # ------------------------------------------------------------------
    log("[STAGE 4/8] Validating NPZ structure, time geometry, and values")

    split_integrity_rows = []
    standardization_rows = []
    train_feature_stats = None

    horizons_arr = np.asarray(horizons, dtype=np.int64)
    horizon_ns = horizons_arr * 60 * 1_000_000_000

    previous_latest_used_index = None
    previous_split_name = None

    for split_name in REQUIRED_SPLITS:
        log("")
        log(f"--- {split_name.upper()} ---")

        npz_path = dataset_dir / f"{split_name}.npz"
        require_file(npz_path)

        summary_row = split_summary.loc[
            split_summary["split"] == split_name
        ].iloc[0]

        raw_a = int(summary_row["raw_start_index"])
        raw_b = int(summary_row["raw_end_index_exclusive"])
        accepted_expected = int(summary_row["accepted_windows"])

        with np.load(npz_path, allow_pickle=False) as npz:
            missing_keys = [
                key for key in REQUIRED_NPZ_KEYS
                if key not in npz.files
            ]
            recorder.add(
                "npz schema",
                f"{split_name} required array keys",
                len(missing_keys) == 0,
                missing_keys if missing_keys else "all present",
                "all present",
            )

            if missing_keys:
                raise RuntimeError(
                    f"{split_name}.npz is missing keys: {missing_keys}"
                )

            X = npz["X"]
            y = npz["y"]
            y_raw = npz["y_raw"]
            context_ns = npz["context_end_time_ns"]
            target_ns = npz["target_time_ns"]
            anchors = npz["anchor_row_index"]

            n = X.shape[0]

            # Shapes.
            expected_X_shape = (
                n,
                lookback,
                len(feature_names),
            )
            expected_y_shape = (
                n,
                len(horizons),
                len(target_names),
            )

            recorder.add(
                "shape",
                f"{split_name} sample count matches split_summary",
                n == accepted_expected,
                n,
                accepted_expected,
            )
            recorder.add(
                "shape",
                f"{split_name} X shape",
                tuple(X.shape) == expected_X_shape,
                tuple(X.shape),
                expected_X_shape,
            )
            recorder.add(
                "shape",
                f"{split_name} y shape",
                tuple(y.shape) == expected_y_shape,
                tuple(y.shape),
                expected_y_shape,
            )
            recorder.add(
                "shape",
                f"{split_name} y_raw shape",
                tuple(y_raw.shape) == expected_y_shape,
                tuple(y_raw.shape),
                expected_y_shape,
            )
            recorder.add(
                "shape",
                f"{split_name} target_time_ns shape",
                tuple(target_ns.shape) == (n, len(horizons)),
                tuple(target_ns.shape),
                (n, len(horizons)),
            )
            recorder.add(
                "shape",
                f"{split_name} context_end_time_ns shape",
                tuple(context_ns.shape) == (n,),
                tuple(context_ns.shape),
                (n,),
            )
            recorder.add(
                "shape",
                f"{split_name} anchor_row_index shape",
                tuple(anchors.shape) == (n,),
                tuple(anchors.shape),
                (n,),
            )

            # Dtypes.
            recorder.add(
                "dtype",
                f"{split_name} X float32",
                X.dtype == np.float32,
                X.dtype,
                np.float32,
                severity="warning",
            )
            recorder.add(
                "dtype",
                f"{split_name} y float32",
                y.dtype == np.float32,
                y.dtype,
                np.float32,
                severity="warning",
            )
            recorder.add(
                "dtype",
                f"{split_name} y_raw float32",
                y_raw.dtype == np.float32,
                y_raw.dtype,
                np.float32,
                severity="warning",
            )

            # Finite.
            X_finite = bool(np.isfinite(X).all())
            y_finite = bool(np.isfinite(y).all())
            y_raw_finite = bool(np.isfinite(y_raw).all())

            recorder.add(
                "numerical",
                f"{split_name} X finite",
                X_finite,
                X_finite,
                True,
            )
            recorder.add(
                "numerical",
                f"{split_name} y finite",
                y_finite,
                y_finite,
                True,
            )
            recorder.add(
                "numerical",
                f"{split_name} y_raw finite",
                y_raw_finite,
                y_raw_finite,
                True,
            )

            # Anchor monotonicity/uniqueness.
            anchor_diff = np.diff(anchors)
            recorder.add(
                "temporal",
                f"{split_name} anchors strictly increasing",
                bool((anchor_diff > 0).all()) if len(anchor_diff) else True,
                int(np.min(anchor_diff)) if len(anchor_diff) else "n/a",
                "> 0",
            )
            recorder.add(
                "temporal",
                f"{split_name} anchors unique",
                len(np.unique(anchors)) == len(anchors),
                len(np.unique(anchors)),
                len(anchors),
            )

            # Every input+target span stays within this raw split.
            earliest_input_idx = anchors - lookback + 1
            latest_target_idx = anchors + max_h

            within_raw_split = bool(
                (earliest_input_idx >= raw_a).all()
                and (latest_target_idx < raw_b).all()
            )
            recorder.add(
                "leakage",
                f"{split_name} windows stay inside raw split",
                within_raw_split,
                (
                    f"used index range "
                    f"{int(np.min(earliest_input_idx))}.."
                    f"{int(np.max(latest_target_idx))}"
                ),
                f"{raw_a}..{raw_b - 1}",
            )

            # Cross-split raw index separation.
            current_earliest = int(np.min(earliest_input_idx))
            current_latest = int(np.max(latest_target_idx))

            if previous_latest_used_index is not None:
                disjoint = current_earliest > previous_latest_used_index
                recorder.add(
                    "leakage",
                    f"{previous_split_name}->{split_name} used raw indices disjoint",
                    disjoint,
                    f"{previous_latest_used_index} < {current_earliest}",
                    "strictly separated",
                )

            previous_latest_used_index = current_latest
            previous_split_name = split_name

            # Context and target time geometry.
            context_diff_ns = np.diff(context_ns)
            context_increasing = (
                bool((context_diff_ns > 0).all())
                if len(context_diff_ns)
                else True
            )
            recorder.add(
                "temporal",
                f"{split_name} context times strictly increasing",
                context_increasing,
                context_increasing,
                True,
            )

            # Decode the stored integers explicitly as nanoseconds and verify
            # that they fall within the actual raw split time range.
            context_dt = pd.to_datetime(context_ns, unit="ns", utc=True)
            split_start_expected = pd.Timestamp(summary_row["start_time"])
            split_end_expected = pd.Timestamp(summary_row["end_time"])
            if split_start_expected.tzinfo is None:
                split_start_expected = split_start_expected.tz_localize("UTC")
            else:
                split_start_expected = split_start_expected.tz_convert("UTC")
            if split_end_expected.tzinfo is None:
                split_end_expected = split_end_expected.tz_localize("UTC")
            else:
                split_end_expected = split_end_expected.tz_convert("UTC")

            absolute_time_ok = bool(
                context_dt[0] >= split_start_expected
                and context_dt[-1] <= split_end_expected
            )
            recorder.add(
                "temporal",
                f"{split_name} decoded context timestamps lie inside split dates",
                absolute_time_ok,
                f"{context_dt[0]} -> {context_dt[-1]}",
                f"inside {split_start_expected} -> {split_end_expected}",
            )

            expected_target_ns = (
                context_ns[:, None] + horizon_ns[None, :]
            )
            target_time_exact = bool(
                np.array_equal(target_ns, expected_target_ns)
            )
            recorder.add(
                "temporal",
                f"{split_name} target times equal context+horizon",
                target_time_exact,
                target_time_exact,
                True,
            )

            # y standardization identity.
            expected_y = (
                (y_raw.astype(np.float64) - y_mean[None, None, :])
                / y_std[None, None, :]
            )
            y_std_error = float(
                np.max(np.abs(expected_y - y.astype(np.float64)))
            )
            recorder.add(
                "standardization",
                f"{split_name} y equals standardized y_raw",
                y_std_error <= 2e-5,
                f"{y_std_error:.3e}",
                "<= 2e-5",
            )

            # Source-referenced exact checks.
            x_reconstruction_error = np.nan
            y_raw_source_error = np.nan
            context_source_exact = np.nan
            target_source_exact = np.nan

            if source_available:
                # Anchor time must exactly match source timestamp.
                expected_context_source = source_times_ns[anchors]
                context_source_exact = bool(
                    np.array_equal(context_ns, expected_context_source)
                )
                recorder.add(
                    "source reconstruction",
                    f"{split_name} context timestamps match source anchors",
                    context_source_exact,
                    context_source_exact,
                    True,
                )

                # All target raw values and times against source.
                target_indices = anchors[:, None] + horizons_arr[None, :]
                expected_y_raw_source = source_targets[target_indices]
                y_raw_source_error = float(
                    np.max(
                        np.abs(
                            expected_y_raw_source.astype(np.float64)
                            - y_raw.astype(np.float64)
                        )
                    )
                )
                recorder.add(
                    "source reconstruction",
                    f"{split_name} all y_raw values match source",
                    y_raw_source_error <= 2e-6,
                    f"{y_raw_source_error:.3e}",
                    "<= 2e-6",
                )

                expected_target_source = source_times_ns[target_indices]
                target_source_exact = bool(
                    np.array_equal(target_ns, expected_target_source)
                )
                recorder.add(
                    "source reconstruction",
                    f"{split_name} all target timestamps match source",
                    target_source_exact,
                    target_source_exact,
                    True,
                )

                # Deterministic sample of X windows reconstructed from source.
                sample_count = min(args.x_check_samples, n)
                if sample_count > 0:
                    # Evenly spaced indices give deterministic coverage over whole split.
                    chosen = np.unique(
                        np.linspace(
                            0,
                            n - 1,
                            num=sample_count,
                            dtype=np.int64,
                        )
                    )

                    max_error = 0.0
                    f_mean64 = f_mean.astype(np.float64)
                    f_std64 = f_std.astype(np.float64)

                    for sample_i in chosen:
                        anchor = int(anchors[sample_i])
                        idx = np.arange(
                            anchor - lookback + 1,
                            anchor + 1,
                            dtype=np.int64,
                        )
                        expected_X = (
                            (
                                source_feature_values[idx]
                                - f_mean64[None, :]
                            )
                            / f_std64[None, :]
                        )
                        actual_X = X[sample_i].astype(np.float64)
                        err = float(
                            np.max(np.abs(expected_X - actual_X))
                        )
                        if err > max_error:
                            max_error = err

                    x_reconstruction_error = max_error
                    recorder.add(
                        "source reconstruction",
                        f"{split_name} sampled X windows match source+train scaler",
                        x_reconstruction_error <= 2e-5,
                        f"{x_reconstruction_error:.3e}",
                        "<= 2e-5",
                        details=f"deterministic windows checked={len(chosen)}",
                    )

            # Window-weighted X statistics for diagnostics.
            feature_mean = np.mean(
                X.astype(np.float64),
                axis=(0, 1),
            )
            feature_std = np.std(
                X.astype(np.float64),
                axis=(0, 1),
                ddof=0,
            )

            for i, name in enumerate(feature_names):
                standardization_rows.append({
                    "split": split_name,
                    "feature": name,
                    "window_weighted_mean": float(feature_mean[i]),
                    "window_weighted_std": float(feature_std[i]),
                })

            if split_name == "train":
                train_feature_stats = pd.DataFrame([
                    {
                        "feature": name,
                        "window_weighted_mean": float(feature_mean[i]),
                        "window_weighted_std": float(feature_std[i]),
                    }
                    for i, name in enumerate(feature_names)
                ])

            context_gap_minutes = (
                context_diff_ns.astype(np.float64)
                / (60 * 1_000_000_000)
                if len(context_diff_ns)
                else np.array([], dtype=np.float64)
            )

            split_integrity_rows.append({
                "split": split_name,
                "samples": n,
                "X_shape": str(tuple(X.shape)),
                "y_shape": str(tuple(y.shape)),
                "raw_split_start_index": raw_a,
                "raw_split_end_index_exclusive": raw_b,
                "used_earliest_input_index": current_earliest,
                "used_latest_target_index": current_latest,
                "context_start_utc": pd.to_datetime(
                    int(context_ns[0]),
                    unit="ns",
                    utc=True,
                ),
                "context_end_utc": pd.to_datetime(
                    int(context_ns[-1]),
                    unit="ns",
                    utc=True,
                ),
                "min_context_gap_min": (
                    float(np.min(context_gap_minutes))
                    if len(context_gap_minutes)
                    else np.nan
                ),
                "max_context_gap_min": (
                    float(np.max(context_gap_minutes))
                    if len(context_gap_minutes)
                    else np.nan
                ),
                "y_standardization_max_abs_error": y_std_error,
                "X_source_sample_max_abs_error": x_reconstruction_error,
                "y_raw_source_max_abs_error": y_raw_source_error,
                "context_matches_source": context_source_exact,
                "target_times_match_source": target_source_exact,
            })

            del X, y, y_raw, context_ns, target_ns, anchors
            gc.collect()

    # ------------------------------------------------------------------
    # Stage 7: write reports.
    # ------------------------------------------------------------------
    log("")
    log("[STAGE 7/8] Writing validation reports")

    checks_df = recorder.dataframe()
    split_integrity_df = pd.DataFrame(split_integrity_rows)
    standardization_df = pd.DataFrame(standardization_rows)
    source_reconstruction_df = pd.DataFrame(source_check_rows)

    checks_df.to_csv(
        output_dir / "validation_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    split_integrity_df.to_csv(
        output_dir / "split_integrity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    standardization_df.to_csv(
        output_dir / "standardization_check.csv",
        index=False,
        encoding="utf-8-sig",
    )
    source_reconstruction_df.to_csv(
        output_dir / "source_reconstruction_check.csv",
        index=False,
        encoding="utf-8-sig",
    )

    critical_failures = recorder.critical_failures()
    warning_count = recorder.warnings()
    passed = critical_failures == 0

    summary_payload = {
        "validator_version": VALIDATOR_VERSION,
        "dataset_version": manifest.get("dataset_version"),
        "dataset_dir": str(dataset_dir.resolve()),
        "validation_passed": passed,
        "critical_failures": critical_failures,
        "warnings": warning_count,
        "source_reconstruction_performed": source_available,
        "lookback_minutes": lookback,
        "forecast_horizons_minutes": list(horizons),
        "feature_names": feature_names,
        "target_names": target_names,
        "split_samples": {
            row["split"]: int(row["samples"])
            for row in split_integrity_rows
        },
    }

    with (output_dir / "validation_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary_payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with (output_dir / "validation_report.txt").open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("Saildrone forecasting dataset validation report\n")
        f.write("=" * 88 + "\n\n")
        f.write(f"Validator version: {VALIDATOR_VERSION}\n")
        f.write(f"Dataset version: {manifest.get('dataset_version')}\n")
        f.write(f"Dataset directory: {dataset_dir}\n")
        f.write(f"Validation passed: {passed}\n")
        f.write(f"Critical failures: {critical_failures}\n")
        f.write(f"Warnings: {warning_count}\n")
        f.write(f"Source reconstruction performed: {source_available}\n")
        f.write(f"Lookback: {lookback} min\n")
        f.write(f"Horizons: {list(horizons)} min\n")
        f.write(f"Features: {feature_names}\n")
        f.write(f"Targets: {target_names}\n\n")

        f.write("Split integrity\n")
        f.write("-" * 88 + "\n")
        f.write(split_integrity_df.to_string(index=False))
        f.write("\n\n")

        f.write("Checks\n")
        f.write("-" * 88 + "\n")
        f.write(checks_df.to_string(index=False))
        f.write("\n")

    flag_path = output_dir / "VALIDATION_PASSED.flag"
    if passed:
        flag_path.write_text(
            "All critical forecasting-dataset validation checks passed.\n",
            encoding="utf-8",
        )
    elif flag_path.exists():
        flag_path.unlink()

    # ------------------------------------------------------------------
    # Stage 8: optional SCI figures and final verdict.
    # ------------------------------------------------------------------
    log("[STAGE 8/8] Finalization")

    if args.plots and train_feature_stats is not None:
        log("[PLOT] Generating SCI-style validation figures")
        make_validation_plots(
            output_dir=output_dir,
            split_integrity=split_integrity_df,
            train_feature_stats=train_feature_stats,
        )
        log("[OK] Figures complete")

    log("")
    log("=" * 88)
    log("VALIDATION VERDICT")
    log("=" * 88)
    log(f"Critical failures : {critical_failures}")
    log(f"Warnings          : {warning_count}")
    log(f"Source checked    : {source_available}")

    for row in split_integrity_rows:
        log(
            f"{row['split']:10s}: "
            f"samples={row['samples']:6d}, "
            f"X={row['X_shape']}, "
            f"y={row['y_shape']}"
        )

    log("")
    if passed:
        log("[PASS] Forecasting dataset passed all critical integrity checks.")
        log(f"[PASS] Flag written: {flag_path}")
        log(f"[DONE] Validation output: {output_dir}")
        return 0

    log("[FAIL] Forecasting dataset has critical integrity failures.")
    log(f"[FAIL] Inspect: {output_dir / 'validation_checks.csv'}")
    return 2


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
