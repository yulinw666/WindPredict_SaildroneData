# -*- coding: utf-8 -*-
"""
03B_validate_joint_forecasting_dataset.py

Independent validator for the joint Saildrone wind-vessel forecasting dataset
produced by 02B_build_joint_forecasting_dataset.py.

Joint-task v0.1 expected inputs (11)
------------------------------------
UWND_MEAN, VWND_MEAN, TEMP_AIR_MEAN, RH_MEAN, SOG,
COG_sin, COG_cos, HDG_sin, HDG_cos,
WING_ANGLE_sin, WING_ANGLE_cos

Expected joint targets (6)
--------------------------
UWND_MEAN, VWND_MEAN,
VESSEL_EAST_MPS, VESSEL_NORTH_MPS,
HDG_sin, HDG_cos

Expected physics-derived reference
----------------------------------
APPARENT_EAST_MPS  = UWND_MEAN - VESSEL_EAST_MPS
APPARENT_NORTH_MPS = VWND_MEAN - VESSEL_NORTH_MPS

Core validation groups
----------------------
A. Metadata / schema integrity
B. Numerical integrity (NaN/Inf, dtype, shape)
C. Chronology and data-leakage integrity
D. Train-only scaler reconstruction from audited source
E. Full y_raw reconstruction from audited source
F. Deterministic X-window reconstruction from audited source
G. Circular encoding identities:
       sin^2 + cos^2 = 1
   for COG, HDG and WING_ANGLE
H. Vessel-velocity physics:
       V_E = SOG * sin(COG)
       V_N = SOG * cos(COG)
I. Apparent-air physics:
       A_E = U - V_E
       A_N = V - V_N
J. Heading target physical consistency

Outputs
-------
validation_report.txt
validation_checks.csv
split_integrity.csv
standardization_check.csv
source_reconstruction_check.csv
physics_consistency_check.csv
validation_summary.json
VALIDATION_PASSED.flag
figures/*  when --plots is supplied
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


VALIDATOR_VERSION = "0.1.0"

EXPECTED_FEATURE_NAMES = [
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

EXPECTED_TARGET_NAMES = [
    "UWND_MEAN",
    "VWND_MEAN",
    "VESSEL_EAST_MPS",
    "VESSEL_NORTH_MPS",
    "HDG_sin",
    "HDG_cos",
]

EXPECTED_APPARENT_NAMES = [
    "APPARENT_EAST_MPS",
    "APPARENT_NORTH_MPS",
]

REQUIRED_NPZ_KEYS = [
    "X",
    "y",
    "y_raw",
    "apparent_earth_raw",
    "context_end_time_ns",
    "target_time_ns",
    "anchor_row_index",
]

SPLIT_ORDER = ["train", "validation", "test"]
EXPECTED_DT_MIN = 1.0


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
        log(
            f"{prefix} {category}: {check} | "
            f"value={value} | expected={expected}"
        )

    def warn(
        self,
        category: str,
        check: str,
        value,
        expected,
        details: str = "",
    ) -> None:
        self.rows.append(
            Check(
                category=category,
                check=check,
                status="WARN",
                severity="warning",
                value=str(value),
                expected=str(expected),
                details=details,
            )
        )
        log(
            f"[WARN] {category}: {check} | "
            f"value={value} | expected={expected}"
        )

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.rows])

    def critical_failures(self) -> int:
        return sum(
            r.status == "FAIL" and r.severity == "critical"
            for r in self.rows
        )

    def warnings(self) -> int:
        return sum(r.status == "WARN" for r in self.rows)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_scaler(
    scalers: dict,
    key: str,
) -> tuple[list[str], np.ndarray, np.ndarray, int]:
    obj = scalers[key]
    return (
        list(obj["names"]),
        np.asarray(obj["mean"], dtype=np.float64),
        np.asarray(obj["std"], dtype=np.float64),
        int(obj["n_fit_rows"]),
    )


def compute_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    valid = np.isfinite(values).all(axis=1)
    selected = values[valid]

    if len(selected) == 0:
        raise RuntimeError("No valid rows available for scaler reconstruction.")

    mean = np.mean(selected, axis=0, dtype=np.float64)
    std = np.std(selected, axis=0, dtype=np.float64, ddof=0)

    return mean, std, int(valid.sum())


def reconstruct_source(
    manifest: dict,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    source_path = Path(
        manifest["source_selected_variables_raw"]
    )
    require_file(source_path)

    raw = pd.read_csv(
        source_path,
        low_memory=False,
    )
    raw["time"] = pd.to_datetime(
        raw["time"],
        errors="coerce",
    )

    segment = manifest["longest_segment"]
    start = pd.Timestamp(segment["start_time"])
    end = pd.Timestamp(segment["end_time"])

    core = raw.loc[
        (raw["time"] >= start)
        & (raw["time"] <= end)
    ].copy()

    core = (
        core.sort_values("time")
        .reset_index(drop=True)
    )

    required = [
        "time",
        "UWND_MEAN",
        "VWND_MEAN",
        "TEMP_AIR_MEAN",
        "RH_MEAN",
        "SOG",
        "COG",
        "HDG",
        "WING_ANGLE",
    ]

    missing = [
        name
        for name in required
        if name not in core.columns
    ]

    if missing:
        raise KeyError(
            "Audited source missing required joint-task columns: "
            + ", ".join(missing)
        )

    u = pd.to_numeric(
        core["UWND_MEAN"],
        errors="coerce",
    )
    v = pd.to_numeric(
        core["VWND_MEAN"],
        errors="coerce",
    )
    temp = pd.to_numeric(
        core["TEMP_AIR_MEAN"],
        errors="coerce",
    )
    rh = pd.to_numeric(
        core["RH_MEAN"],
        errors="coerce",
    )
    sog = pd.to_numeric(
        core["SOG"],
        errors="coerce",
    )
    cog = pd.to_numeric(
        core["COG"],
        errors="coerce",
    )
    hdg = pd.to_numeric(
        core["HDG"],
        errors="coerce",
    )
    wing = pd.to_numeric(
        core["WING_ANGLE"],
        errors="coerce",
    )

    cog_rad = np.deg2rad(
        cog.to_numpy(dtype=np.float64)
    )
    hdg_rad = np.deg2rad(
        hdg.to_numpy(dtype=np.float64)
    )
    wing_rad = np.deg2rad(
        wing.to_numpy(dtype=np.float64)
    )

    cog_sin = np.sin(cog_rad)
    cog_cos = np.cos(cog_rad)
    hdg_sin = np.sin(hdg_rad)
    hdg_cos = np.cos(hdg_rad)
    wing_sin = np.sin(wing_rad)
    wing_cos = np.cos(wing_rad)

    sog_np = sog.to_numpy(dtype=np.float64)

    vessel_east = (
        sog_np
        * cog_sin
    )
    vessel_north = (
        sog_np
        * cog_cos
    )

    features = pd.DataFrame({
        "UWND_MEAN": u,
        "VWND_MEAN": v,
        "TEMP_AIR_MEAN": temp,
        "RH_MEAN": rh,
        "SOG": sog,
        "COG_sin": cog_sin,
        "COG_cos": cog_cos,
        "HDG_sin": hdg_sin,
        "HDG_cos": hdg_cos,
        "WING_ANGLE_sin": wing_sin,
        "WING_ANGLE_cos": wing_cos,
    })

    targets = pd.DataFrame({
        "UWND_MEAN": u,
        "VWND_MEAN": v,
        "VESSEL_EAST_MPS": vessel_east,
        "VESSEL_NORTH_MPS": vessel_north,
        "HDG_sin": hdg_sin,
        "HDG_cos": hdg_cos,
    })

    apparent = pd.DataFrame({
        "APPARENT_EAST_MPS": (
            u.to_numpy(dtype=np.float64)
            - vessel_east
        ),
        "APPARENT_NORTH_MPS": (
            v.to_numpy(dtype=np.float64)
            - vessel_north
        ),
    })

    return core, features, targets, apparent


def safe_max_abs_error(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    if a.shape != b.shape:
        return float("inf")

    finite = (
        np.isfinite(a)
        & np.isfinite(b)
    )

    if not finite.any():
        return float("nan")

    return float(
        np.max(
            np.abs(
                a[finite].astype(np.float64)
                - b[finite].astype(np.float64)
            )
        )
    )


def load_sci_style(script_dir: Path):
    style_path = (
        script_dir
        / "sci_plot_style.py"
    )

    if not style_path.exists():
        log(
            "[PLOT WARNING] sci_plot_style.py not found: "
            f"{style_path}"
        )
        return None

    spec = importlib.util.spec_from_file_location(
        "sci_plot_style",
        str(style_path),
    )

    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules["sci_plot_style"] = module
    spec.loader.exec_module(module)

    if hasattr(
        module,
        "apply_sci_style",
    ):
        try:
            selected = module.apply_sci_style(
                base_font_size=8.5
            )
        except TypeError:
            try:
                selected = module.apply_sci_style(
                    font_size=8.5
                )
            except TypeError:
                selected = module.apply_sci_style()

        log(
            "[PLOT] SCI style applied; "
            f"selected_font={selected}"
        )

    return module


def make_plots(
    output_dir: Path,
    split_integrity: pd.DataFrame,
    physics_df: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    style = load_sci_style(
        Path(__file__).resolve().parent
    )

    fig_dir = (
        output_dir
        / "figures"
    )
    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    def finish(ax, grid=True):
        if (
            style is not None
            and hasattr(style, "clean_axis")
        ):
            style.clean_axis(
                ax,
                grid=grid,
            )
        elif (
            style is not None
            and hasattr(style, "finish_axes")
        ):
            style.finish_axes(
                ax,
                grid=grid,
            )
        else:
            ax.tick_params(
                which="both",
                direction="in",
                top=True,
                right=True,
            )
            if grid:
                ax.grid(
                    True,
                    alpha=0.25,
                )

    def save(fig, name):
        base = fig_dir / name

        if (
            style is not None
            and hasattr(style, "save_figure")
        ):
            style.save_figure(
                fig,
                base,
            )
        else:
            fig.savefig(
                base.with_suffix(".png"),
                dpi=600,
                bbox_inches="tight",
            )
            fig.savefig(
                base.with_suffix(".pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

    # 1. Sample counts
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

    fig, ax = plt.subplots(
        figsize=(3.6, 2.8)
    )
    ax.bar(
        ["Train", "Validation", "Test"],
        values,
    )
    ax.set_ylabel(
        "Forecasting samples"
    )
    finish(
        ax,
        grid=True,
    )
    save(
        fig,
        "01_joint_validated_sample_counts",
    )

    # 2. Physics reconstruction errors
    plot_df = physics_df.loc[
        np.isfinite(
            physics_df["max_abs_error"].to_numpy(
                dtype=np.float64
            )
        )
    ].copy()

    if not plot_df.empty:
        fig, ax = plt.subplots(
            figsize=(7.2, 3.0)
        )
        x = np.arange(
            len(plot_df)
        )
        ax.bar(
            x,
            plot_df[
                "max_abs_error"
            ].to_numpy(
                dtype=np.float64
            ),
        )
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(
            plot_df["check"].tolist(),
            rotation=35,
            ha="right",
        )
        ax.set_ylabel(
            "Maximum absolute error"
        )
        finish(
            ax,
            grid=True,
        )
        save(
            fig,
            "02_joint_physics_consistency_errors",
        )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        required=True,
        help=(
            "Directory produced by "
            "02B_build_joint_forecasting_dataset.py"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Validation output directory. "
            "Default: <dataset-dir>/validation_v0_1"
        ),
    )

    parser.add_argument(
        "--x-check-samples",
        type=int,
        default=256,
        help=(
            "Deterministic source-reconstruction "
            "X windows checked per split."
        ),
    )

    parser.add_argument(
        "--plots",
        action="store_true",
        help=(
            "Generate SCI-style validation figures."
        ),
    )

    args = parser.parse_args()

    dataset_dir = Path(
        args.dataset_dir
    )

    output_dir = (
        Path(args.output)
        if args.output is not None
        else dataset_dir / "validation_v0_1"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    recorder = Recorder()

    log("=" * 92)
    log(
        "SAILDRONE JOINT WIND-VESSEL DATASET VALIDATOR"
    )
    log("=" * 92)
    log(
        f"validator_version : {VALIDATOR_VERSION}"
    )
    log(
        f"dataset_dir       : {dataset_dir}"
    )
    log(
        f"output_dir        : {output_dir}"
    )
    log("")

    # --------------------------------------------------------
    # Stage 1: metadata
    # --------------------------------------------------------
    log(
        "[STAGE 1/8] Reading manifest, scalers, and split metadata"
    )

    manifest_path = (
        dataset_dir
        / "dataset_manifest.json"
    )
    scalers_path = (
        dataset_dir
        / "scalers.json"
    )
    split_summary_path = (
        dataset_dir
        / "split_summary.csv"
    )

    require_file(
        manifest_path
    )
    require_file(
        scalers_path
    )
    require_file(
        split_summary_path
    )

    manifest = load_json(
        manifest_path
    )
    scalers = load_json(
        scalers_path
    )
    split_summary = pd.read_csv(
        split_summary_path
    )

    lookback = int(
        manifest[
            "lookback_minutes"
        ]
    )
    horizons = tuple(
        int(v)
        for v in manifest[
            "forecast_horizons_minutes"
        ]
    )
    max_h = max(
        horizons
    )

    feature_names = list(
        manifest[
            "feature_names"
        ]
    )
    target_names = list(
        manifest[
            "target_names"
        ]
    )
    apparent_names = list(
        manifest.get(
            "apparent_reference_names",
            [],
        )
    )

    (
        f_names,
        f_mean,
        f_std,
        f_fit_rows,
    ) = parse_scaler(
        scalers,
        "feature_scaler",
    )

    (
        y_names,
        y_mean,
        y_std,
        y_fit_rows,
    ) = parse_scaler(
        scalers,
        "target_scaler",
    )

    recorder.add(
        "metadata",
        "task name",
        manifest.get("task_name")
        == "joint_wind_vessel_forecasting",
        manifest.get("task_name"),
        "joint_wind_vessel_forecasting",
    )

    recorder.add(
        "metadata",
        "feature names",
        feature_names
        == EXPECTED_FEATURE_NAMES,
        feature_names,
        EXPECTED_FEATURE_NAMES,
    )

    recorder.add(
        "metadata",
        "target names",
        target_names
        == EXPECTED_TARGET_NAMES,
        target_names,
        EXPECTED_TARGET_NAMES,
    )

    recorder.add(
        "metadata",
        "apparent reference names",
        apparent_names
        == EXPECTED_APPARENT_NAMES,
        apparent_names,
        EXPECTED_APPARENT_NAMES,
    )

    recorder.add(
        "metadata",
        "timestamp unit",
        manifest.get(
            "timestamp_integer_unit"
        ) == "ns",
        manifest.get(
            "timestamp_integer_unit"
        ),
        "ns",
    )

    recorder.add(
        "metadata",
        "feature scaler names match",
        f_names == feature_names,
        f_names,
        feature_names,
    )

    recorder.add(
        "metadata",
        "target scaler names match",
        y_names == target_names,
        y_names,
        target_names,
    )

    recorder.add(
        "metadata",
        "feature scaler finite/nonzero",
        bool(
            np.isfinite(f_mean).all()
            and np.isfinite(f_std).all()
            and (f_std > 0).all()
        ),
        "valid",
        "valid",
    )

    recorder.add(
        "metadata",
        "target scaler finite/nonzero",
        bool(
            np.isfinite(y_mean).all()
            and np.isfinite(y_std).all()
            and (y_std > 0).all()
        ),
        "valid",
        "valid",
    )

    # --------------------------------------------------------
    # Stage 2: reconstruct source and scalers
    # --------------------------------------------------------
    log(
        "[STAGE 2/8] Reconstructing audited source and physics variables"
    )

    (
        core,
        source_features_df,
        source_targets_df,
        source_apparent_df,
    ) = reconstruct_source(
        manifest
    )

    expected_core_rows = int(
        manifest[
            "longest_segment"
        ][
            "sample_count"
        ]
    )

    recorder.add(
        "source",
        "core row count",
        len(core)
        == expected_core_rows,
        len(core),
        expected_core_rows,
    )

    dt = (
        core["time"]
        .diff()
        .dt.total_seconds()
        .div(60.0)
    )

    bad_dt = int(
        (
            dt.iloc[1:].isna()
            | (
                np.abs(
                    dt.iloc[1:]
                    - EXPECTED_DT_MIN
                )
                > 1e-9
            )
        ).sum()
    )

    recorder.add(
        "source",
        "core 1-min time continuity",
        bad_dt == 0,
        bad_dt,
        0,
    )

    source_features = (
        source_features_df[
            feature_names
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    source_targets = (
        source_targets_df[
            target_names
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    source_apparent = (
        source_apparent_df[
            apparent_names
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    source_times_ns = (
        pd.to_datetime(
            core["time"]
        )
        .to_numpy(
            dtype="datetime64[ns]"
        )
        .astype(np.int64)
    )

    train_row = (
        split_summary.loc[
            split_summary["split"]
            == "train"
        ]
        .iloc[0]
    )

    train_a = int(
        train_row[
            "raw_start_index"
        ]
    )
    train_b = int(
        train_row[
            "raw_end_index_exclusive"
        ]
    )

    (
        f_mean_re,
        f_std_re,
        f_fit_re,
    ) = compute_scaler(
        source_features[
            train_a:train_b
        ]
    )

    (
        y_mean_re,
        y_std_re,
        y_fit_re,
    ) = compute_scaler(
        source_targets[
            train_a:train_b
        ]
    )

    tol_scaler = 1e-10

    recorder.add(
        "standardization",
        "feature train-only mean reconstruction",
        safe_max_abs_error(
            f_mean_re,
            f_mean,
        ) <= tol_scaler,
        f"{safe_max_abs_error(f_mean_re, f_mean):.3e}",
        f"<= {tol_scaler:.1e}",
    )

    recorder.add(
        "standardization",
        "feature train-only std reconstruction",
        safe_max_abs_error(
            f_std_re,
            f_std,
        ) <= tol_scaler,
        f"{safe_max_abs_error(f_std_re, f_std):.3e}",
        f"<= {tol_scaler:.1e}",
    )

    recorder.add(
        "standardization",
        "target train-only mean reconstruction",
        safe_max_abs_error(
            y_mean_re,
            y_mean,
        ) <= tol_scaler,
        f"{safe_max_abs_error(y_mean_re, y_mean):.3e}",
        f"<= {tol_scaler:.1e}",
    )

    recorder.add(
        "standardization",
        "target train-only std reconstruction",
        safe_max_abs_error(
            y_std_re,
            y_std,
        ) <= tol_scaler,
        f"{safe_max_abs_error(y_std_re, y_std):.3e}",
        f"<= {tol_scaler:.1e}",
    )

    recorder.add(
        "standardization",
        "feature scaler fit rows",
        f_fit_re
        == f_fit_rows,
        f_fit_re,
        f_fit_rows,
    )

    recorder.add(
        "standardization",
        "target scaler fit rows",
        y_fit_re
        == y_fit_rows,
        y_fit_re,
        y_fit_rows,
    )

    # --------------------------------------------------------
    # Stage 3: physics/circular identities on source
    # --------------------------------------------------------
    log(
        "[STAGE 3/8] Checking circular and physical consistency"
    )

    physics_rows = []

    def record_physics(
        check_name: str,
        error: float,
        tolerance: float,
        details: str = "",
    ) -> None:
        recorder.add(
            "physics",
            check_name,
            error <= tolerance,
            f"{error:.3e}",
            f"<= {tolerance:.1e}",
            details=details,
        )
        physics_rows.append({
            "check": check_name,
            "max_abs_error": error,
            "tolerance": tolerance,
            "details": details,
        })

    for prefix in [
        "COG",
        "HDG",
        "WING_ANGLE",
    ]:
        sin_col = f"{prefix}_sin"
        cos_col = f"{prefix}_cos"

        arr = source_features_df[
            [sin_col, cos_col]
        ].to_numpy(
            dtype=np.float64
        )

        valid = np.isfinite(
            arr
        ).all(axis=1)

        norm_error = float(
            np.max(
                np.abs(
                    arr[
                        valid,
                        0,
                    ] ** 2
                    + arr[
                        valid,
                        1,
                    ] ** 2
                    - 1.0
                )
            )
        )

        record_physics(
            f"{prefix} sin/cos unit-circle identity",
            norm_error,
            1e-12,
        )

    # Vessel velocity identity using raw SOG/COG
    sog = pd.to_numeric(
        core["SOG"],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )
    cog = pd.to_numeric(
        core["COG"],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )
    cog_rad = np.deg2rad(
        cog
    )

    vessel_e_expected = (
        sog
        * np.sin(
            cog_rad
        )
    )
    vessel_n_expected = (
        sog
        * np.cos(
            cog_rad
        )
    )

    vessel_e_actual = (
        source_targets_df[
            "VESSEL_EAST_MPS"
        ].to_numpy(
            dtype=np.float64
        )
    )
    vessel_n_actual = (
        source_targets_df[
            "VESSEL_NORTH_MPS"
        ].to_numpy(
            dtype=np.float64
        )
    )

    record_physics(
        "VESSEL_EAST_MPS = SOG*sin(COG)",
        safe_max_abs_error(
            vessel_e_actual,
            vessel_e_expected,
        ),
        1e-12,
    )

    record_physics(
        "VESSEL_NORTH_MPS = SOG*cos(COG)",
        safe_max_abs_error(
            vessel_n_actual,
            vessel_n_expected,
        ),
        1e-12,
    )

    app_e_expected = (
        source_targets_df[
            "UWND_MEAN"
        ].to_numpy(
            dtype=np.float64
        )
        - vessel_e_actual
    )
    app_n_expected = (
        source_targets_df[
            "VWND_MEAN"
        ].to_numpy(
            dtype=np.float64
        )
        - vessel_n_actual
    )

    record_physics(
        "APPARENT_EAST = U - VESSEL_EAST",
        safe_max_abs_error(
            source_apparent_df[
                "APPARENT_EAST_MPS"
            ].to_numpy(
                dtype=np.float64
            ),
            app_e_expected,
        ),
        1e-12,
    )

    record_physics(
        "APPARENT_NORTH = V - VESSEL_NORTH",
        safe_max_abs_error(
            source_apparent_df[
                "APPARENT_NORTH_MPS"
            ].to_numpy(
                dtype=np.float64
            ),
            app_n_expected,
        ),
        1e-12,
    )

    # --------------------------------------------------------
    # Stage 4: split chronology / raw boundaries
    # --------------------------------------------------------
    log(
        "[STAGE 4/8] Checking chronological split boundaries"
    )

    split_order_df = (
        split_summary
        .set_index("split")
        .loc[SPLIT_ORDER]
        .reset_index()
    )

    contiguous = True

    for i in range(
        len(split_order_df) - 1
    ):
        left = split_order_df.iloc[i]
        right = split_order_df.iloc[i + 1]

        if int(
            left[
                "raw_end_index_exclusive"
            ]
        ) != int(
            right[
                "raw_start_index"
            ]
        ):
            contiguous = False

    recorder.add(
        "leakage",
        "raw splits contiguous and disjoint",
        contiguous,
        contiguous,
        True,
    )

    total_raw_rows = int(
        split_order_df[
            "raw_rows"
        ].sum()
    )

    recorder.add(
        "leakage",
        "split raw rows sum to core",
        total_raw_rows
        == expected_core_rows,
        total_raw_rows,
        expected_core_rows,
    )

    for _, row in (
        split_order_df.iterrows()
    ):
        name = row["split"]
        raw_rows = int(
            row["raw_rows"]
        )

        expected_candidate = (
            raw_rows
            - lookback
            - max_h
            + 1
        )

        actual_candidate = int(
            row[
                "candidate_windows"
            ]
        )

        recorder.add(
            "window geometry",
            f"{name} candidate-window formula",
            actual_candidate
            == expected_candidate,
            actual_candidate,
            expected_candidate,
        )

    # --------------------------------------------------------
    # Stage 5: validate each NPZ
    # --------------------------------------------------------
    log(
        "[STAGE 5/8] Validating split NPZ arrays"
    )

    horizons_arr = np.asarray(
        horizons,
        dtype=np.int64,
    )

    horizon_ns = (
        horizons_arr
        * 60
        * 1_000_000_000
    )

    split_integrity_rows = []
    standardization_rows = []
    source_reconstruction_rows = []

    previous_latest_used = None
    previous_split_name = None

    for split_name in SPLIT_ORDER:
        log("")
        log(
            f"--- {split_name.upper()} ---"
        )

        npz_path = (
            dataset_dir
            / f"{split_name}.npz"
        )
        require_file(
            npz_path
        )

        summary_row = (
            split_summary.loc[
                split_summary["split"]
                == split_name
            ]
            .iloc[0]
        )

        raw_a = int(
            summary_row[
                "raw_start_index"
            ]
        )
        raw_b = int(
            summary_row[
                "raw_end_index_exclusive"
            ]
        )

        expected_samples = int(
            summary_row[
                "accepted_windows"
            ]
        )

        with np.load(
            npz_path,
            allow_pickle=False,
        ) as npz:
            missing_keys = [
                key
                for key in REQUIRED_NPZ_KEYS
                if key not in npz.files
            ]

            recorder.add(
                "npz schema",
                f"{split_name} required keys",
                len(missing_keys) == 0,
                missing_keys
                if missing_keys
                else "all present",
                "all present",
            )

            if missing_keys:
                raise RuntimeError(
                    f"{npz_path.name} missing keys: "
                    f"{missing_keys}"
                )

            X = npz["X"]
            y = npz["y"]
            y_raw = npz["y_raw"]
            apparent_raw = (
                npz[
                    "apparent_earth_raw"
                ]
            )
            context_ns = (
                npz[
                    "context_end_time_ns"
                ]
            )
            target_ns = (
                npz[
                    "target_time_ns"
                ]
            )
            anchors = (
                npz[
                    "anchor_row_index"
                ]
            )

            n = X.shape[0]

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
            expected_app_shape = (
                n,
                len(horizons),
                2,
            )

            recorder.add(
                "shape",
                f"{split_name} sample count",
                n == expected_samples,
                n,
                expected_samples,
            )

            recorder.add(
                "shape",
                f"{split_name} X shape",
                tuple(X.shape)
                == expected_X_shape,
                tuple(X.shape),
                expected_X_shape,
            )

            recorder.add(
                "shape",
                f"{split_name} y shape",
                tuple(y.shape)
                == expected_y_shape,
                tuple(y.shape),
                expected_y_shape,
            )

            recorder.add(
                "shape",
                f"{split_name} y_raw shape",
                tuple(y_raw.shape)
                == expected_y_shape,
                tuple(y_raw.shape),
                expected_y_shape,
            )

            recorder.add(
                "shape",
                f"{split_name} apparent shape",
                tuple(apparent_raw.shape)
                == expected_app_shape,
                tuple(apparent_raw.shape),
                expected_app_shape,
            )

            recorder.add(
                "numerical",
                f"{split_name} X finite",
                bool(
                    np.isfinite(X).all()
                ),
                bool(
                    np.isfinite(X).all()
                ),
                True,
            )

            recorder.add(
                "numerical",
                f"{split_name} y finite",
                bool(
                    np.isfinite(y).all()
                ),
                bool(
                    np.isfinite(y).all()
                ),
                True,
            )

            recorder.add(
                "numerical",
                f"{split_name} y_raw finite",
                bool(
                    np.isfinite(y_raw).all()
                ),
                bool(
                    np.isfinite(y_raw).all()
                ),
                True,
            )

            recorder.add(
                "numerical",
                f"{split_name} apparent finite",
                bool(
                    np.isfinite(
                        apparent_raw
                    ).all()
                ),
                bool(
                    np.isfinite(
                        apparent_raw
                    ).all()
                ),
                True,
            )

            # Window boundaries.
            earliest_input = (
                anchors
                - lookback
                + 1
            )
            latest_target = (
                anchors
                + max_h
            )

            within_split = bool(
                (
                    earliest_input
                    >= raw_a
                ).all()
                and (
                    latest_target
                    < raw_b
                ).all()
            )

            recorder.add(
                "leakage",
                f"{split_name} windows stay inside split",
                within_split,
                (
                    f"{int(np.min(earliest_input))}.."
                    f"{int(np.max(latest_target))}"
                ),
                f"{raw_a}..{raw_b - 1}",
            )

            current_earliest = int(
                np.min(
                    earliest_input
                )
            )
            current_latest = int(
                np.max(
                    latest_target
                )
            )

            if previous_latest_used is not None:
                recorder.add(
                    "leakage",
                    (
                        f"{previous_split_name}->"
                        f"{split_name} used indices disjoint"
                    ),
                    current_earliest
                    > previous_latest_used,
                    (
                        f"{previous_latest_used} < "
                        f"{current_earliest}"
                    ),
                    "strictly separated",
                )

            previous_latest_used = (
                current_latest
            )
            previous_split_name = (
                split_name
            )

            # Time geometry.
            context_diff = np.diff(
                context_ns
            )

            recorder.add(
                "temporal",
                f"{split_name} context times increasing",
                bool(
                    (
                        context_diff
                        > 0
                    ).all()
                )
                if len(
                    context_diff
                )
                else True,
                (
                    int(
                        np.min(
                            context_diff
                        )
                    )
                    if len(
                        context_diff
                    )
                    else "n/a"
                ),
                "> 0",
            )

            expected_target_ns = (
                context_ns[:, None]
                + horizon_ns[None, :]
            )

            recorder.add(
                "temporal",
                (
                    f"{split_name} target time = "
                    "context+horizon"
                ),
                bool(
                    np.array_equal(
                        target_ns,
                        expected_target_ns,
                    )
                ),
                bool(
                    np.array_equal(
                        target_ns,
                        expected_target_ns,
                    )
                ),
                True,
            )

            source_context_exact = (
                np.array_equal(
                    context_ns,
                    source_times_ns[
                        anchors
                    ],
                )
            )

            recorder.add(
                "source reconstruction",
                (
                    f"{split_name} context timestamps "
                    "match source"
                ),
                bool(
                    source_context_exact
                ),
                bool(
                    source_context_exact
                ),
                True,
            )

            target_indices = (
                anchors[:, None]
                + horizons_arr[None, :]
            )

            source_target_times_exact = (
                np.array_equal(
                    target_ns,
                    source_times_ns[
                        target_indices
                    ],
                )
            )

            recorder.add(
                "source reconstruction",
                (
                    f"{split_name} target timestamps "
                    "match source"
                ),
                bool(
                    source_target_times_exact
                ),
                bool(
                    source_target_times_exact
                ),
                True,
            )

            # Full target reconstruction.
            source_y_expected = (
                source_targets[
                    target_indices
                ]
            )

            y_raw_err = safe_max_abs_error(
                y_raw,
                source_y_expected,
            )

            recorder.add(
                "source reconstruction",
                (
                    f"{split_name} all y_raw "
                    "match source"
                ),
                y_raw_err <= 2e-6,
                f"{y_raw_err:.3e}",
                "<= 2e-6",
            )

            source_app_expected = (
                source_apparent[
                    target_indices
                ]
            )

            app_err = safe_max_abs_error(
                apparent_raw,
                source_app_expected,
            )

            recorder.add(
                "physics",
                (
                    f"{split_name} apparent reference "
                    "matches source physics"
                ),
                app_err <= 2e-6,
                f"{app_err:.3e}",
                "<= 2e-6",
            )

            # Recalculate apparent directly from stored y_raw.
            # y_raw indices:
            # 0 U, 1 V, 2 vessel_E, 3 vessel_N
            app_from_y = np.empty_like(
                apparent_raw,
                dtype=np.float64,
            )

            app_from_y[
                ...,
                0,
            ] = (
                y_raw[
                    ...,
                    0,
                ]
                - y_raw[
                    ...,
                    2,
                ]
            )

            app_from_y[
                ...,
                1,
            ] = (
                y_raw[
                    ...,
                    1,
                ]
                - y_raw[
                    ...,
                    3,
                ]
            )

            app_internal_err = (
                safe_max_abs_error(
                    apparent_raw,
                    app_from_y,
                )
            )

            recorder.add(
                "physics",
                (
                    f"{split_name} apparent = "
                    "wind-vessel from stored y_raw"
                ),
                app_internal_err
                <= 2e-6,
                f"{app_internal_err:.3e}",
                "<= 2e-6",
            )

            # Heading output on unit circle.
            hdg_norm = (
                y_raw[
                    ...,
                    4,
                ].astype(
                    np.float64
                ) ** 2
                + y_raw[
                    ...,
                    5,
                ].astype(
                    np.float64
                ) ** 2
            )

            hdg_norm_err = float(
                np.max(
                    np.abs(
                        hdg_norm
                        - 1.0
                    )
                )
            )

            recorder.add(
                "physics",
                (
                    f"{split_name} target HDG "
                    "sin/cos unit circle"
                ),
                hdg_norm_err <= 2e-6,
                f"{hdg_norm_err:.3e}",
                "<= 2e-6",
            )

            # y z-standardization identity.
            y_expected_z = (
                (
                    y_raw.astype(
                        np.float64
                    )
                    - y_mean[
                        None,
                        None,
                        :,
                    ]
                )
                / y_std[
                    None,
                    None,
                    :,
                ]
            )

            y_z_err = safe_max_abs_error(
                y,
                y_expected_z,
            )

            recorder.add(
                "standardization",
                (
                    f"{split_name} y equals "
                    "standardized y_raw"
                ),
                y_z_err <= 2e-5,
                f"{y_z_err:.3e}",
                "<= 2e-5",
            )

            # Deterministic X reconstruction.
            sample_count = min(
                args.x_check_samples,
                n,
            )

            chosen = np.unique(
                np.linspace(
                    0,
                    n - 1,
                    num=sample_count,
                    dtype=np.int64,
                )
            )

            x_max_err = 0.0

            for sample_i in chosen:
                anchor = int(
                    anchors[
                        sample_i
                    ]
                )

                idx = np.arange(
                    anchor
                    - lookback
                    + 1,
                    anchor
                    + 1,
                    dtype=np.int64,
                )

                x_expected = (
                    (
                        source_features[
                            idx
                        ]
                        - f_mean[
                            None,
                            :,
                        ]
                    )
                    / f_std[
                        None,
                        :,
                    ]
                )

                err = safe_max_abs_error(
                    X[
                        sample_i
                    ],
                    x_expected,
                )

                if err > x_max_err:
                    x_max_err = err

            recorder.add(
                "source reconstruction",
                (
                    f"{split_name} sampled X "
                    "match source+train scaler"
                ),
                x_max_err <= 2e-5,
                f"{x_max_err:.3e}",
                "<= 2e-5",
                details=(
                    f"deterministic windows checked="
                    f"{len(chosen)}"
                ),
            )

            # Diagnostic window-weighted stats.
            x_mean = np.mean(
                X.astype(
                    np.float64
                ),
                axis=(0, 1),
            )
            x_std = np.std(
                X.astype(
                    np.float64
                ),
                axis=(0, 1),
                ddof=0,
            )

            for i, name in enumerate(
                feature_names
            ):
                standardization_rows.append({
                    "split": split_name,
                    "feature": name,
                    "window_weighted_mean": float(
                        x_mean[i]
                    ),
                    "window_weighted_std": float(
                        x_std[i]
                    ),
                })

            context_gap_min = (
                context_diff.astype(
                    np.float64
                )
                / (
                    60
                    * 1_000_000_000
                )
            )

            split_integrity_rows.append({
                "split": split_name,
                "samples": n,
                "X_shape": str(
                    tuple(
                        X.shape
                    )
                ),
                "y_shape": str(
                    tuple(
                        y.shape
                    )
                ),
                "apparent_shape": str(
                    tuple(
                        apparent_raw.shape
                    )
                ),
                "raw_split_start_index": raw_a,
                "raw_split_end_index_exclusive": raw_b,
                "used_earliest_input_index": current_earliest,
                "used_latest_target_index": current_latest,
                "context_start_utc": pd.to_datetime(
                    int(
                        context_ns[0]
                    ),
                    unit="ns",
                    utc=True,
                ),
                "context_end_utc": pd.to_datetime(
                    int(
                        context_ns[-1]
                    ),
                    unit="ns",
                    utc=True,
                ),
                "min_context_gap_min": (
                    float(
                        np.min(
                            context_gap_min
                        )
                    )
                    if len(
                        context_gap_min
                    )
                    else np.nan
                ),
                "max_context_gap_min": (
                    float(
                        np.max(
                            context_gap_min
                        )
                    )
                    if len(
                        context_gap_min
                    )
                    else np.nan
                ),
                "y_raw_source_max_abs_error": y_raw_err,
                "apparent_source_max_abs_error": app_err,
                "apparent_internal_max_abs_error": app_internal_err,
                "HDG_unit_circle_max_abs_error": hdg_norm_err,
                "X_source_sample_max_abs_error": x_max_err,
                "y_standardization_max_abs_error": y_z_err,
                "context_matches_source": source_context_exact,
                "target_times_match_source": source_target_times_exact,
            })

            source_reconstruction_rows.extend([
                {
                    "split": split_name,
                    "check": "y_raw_source",
                    "max_abs_error": y_raw_err,
                },
                {
                    "split": split_name,
                    "check": "apparent_source",
                    "max_abs_error": app_err,
                },
                {
                    "split": split_name,
                    "check": "X_sample_source",
                    "max_abs_error": x_max_err,
                },
            ])

            del (
                X,
                y,
                y_raw,
                apparent_raw,
                context_ns,
                target_ns,
                anchors,
            )
            gc.collect()

    # --------------------------------------------------------
    # Stage 6: diagnose rejected-window counts
    # --------------------------------------------------------
    log("")
    log(
        "[STAGE 6/8] Diagnosing rejected-window counts"
    )

    rejected_path = (
        dataset_dir
        / "rejected_windows.csv"
    )

    rejected_df = (
        pd.read_csv(
            rejected_path
        )
        if rejected_path.exists()
        else pd.DataFrame()
    )

    if not rejected_df.empty:
        for _, row in (
            rejected_df.iterrows()
        ):
            split_name = (
                row["split"]
            )

            log(
                f"[REJECT] {split_name:10s} "
                f"total={int(row['rejected_windows_total'])}, "
                f"input_missing="
                f"{int(row['rejected_due_to_input_missing'])}, "
                f"joint_target_missing="
                f"{int(row['rejected_due_to_joint_target_missing_after_valid_input'])}"
            )

    # --------------------------------------------------------
    # Stage 7: write reports
    # --------------------------------------------------------
    log(
        "[STAGE 7/8] Writing validation outputs"
    )

    checks_df = (
        recorder.dataframe()
    )

    split_integrity_df = pd.DataFrame(
        split_integrity_rows
    )

    standardization_df = pd.DataFrame(
        standardization_rows
    )

    source_reconstruction_df = pd.DataFrame(
        source_reconstruction_rows
    )

    physics_df = pd.DataFrame(
        physics_rows
    )

    checks_df.to_csv(
        output_dir
        / "validation_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_integrity_df.to_csv(
        output_dir
        / "split_integrity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    standardization_df.to_csv(
        output_dir
        / "standardization_check.csv",
        index=False,
        encoding="utf-8-sig",
    )

    source_reconstruction_df.to_csv(
        output_dir
        / "source_reconstruction_check.csv",
        index=False,
        encoding="utf-8-sig",
    )

    physics_df.to_csv(
        output_dir
        / "physics_consistency_check.csv",
        index=False,
        encoding="utf-8-sig",
    )

    critical_failures = (
        recorder.critical_failures()
    )

    warning_count = (
        recorder.warnings()
    )

    passed = (
        critical_failures
        == 0
    )

    summary = {
        "validator_version": VALIDATOR_VERSION,
        "dataset_version": manifest.get(
            "dataset_version"
        ),
        "task_name": manifest.get(
            "task_name"
        ),
        "validation_passed": passed,
        "critical_failures": critical_failures,
        "warnings": warning_count,
        "lookback_minutes": lookback,
        "forecast_horizons_minutes": list(
            horizons
        ),
        "feature_names": feature_names,
        "target_names": target_names,
        "apparent_reference_names": apparent_names,
        "split_samples": {
            row["split"]: int(
                row["samples"]
            )
            for row in split_integrity_rows
        },
    }

    with (
        output_dir
        / "validation_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with (
        output_dir
        / "validation_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "Saildrone joint wind-vessel "
            "forecasting dataset validation report\n"
        )
        f.write(
            "=" * 92
            + "\n\n"
        )
        f.write(
            f"Validator version: "
            f"{VALIDATOR_VERSION}\n"
        )
        f.write(
            f"Dataset version: "
            f"{manifest.get('dataset_version')}\n"
        )
        f.write(
            f"Task: "
            f"{manifest.get('task_name')}\n"
        )
        f.write(
            f"Validation passed: "
            f"{passed}\n"
        )
        f.write(
            f"Critical failures: "
            f"{critical_failures}\n"
        )
        f.write(
            f"Warnings: "
            f"{warning_count}\n"
        )
        f.write(
            f"Lookback: "
            f"{lookback} min\n"
        )
        f.write(
            f"Horizons: "
            f"{list(horizons)} min\n"
        )
        f.write(
            f"Features: "
            f"{feature_names}\n"
        )
        f.write(
            f"Targets: "
            f"{target_names}\n\n"
        )

        f.write(
            "Split integrity\n"
        )
        f.write(
            "-" * 92
            + "\n"
        )
        f.write(
            split_integrity_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nPhysics consistency\n"
        )
        f.write(
            "-" * 92
            + "\n"
        )
        f.write(
            physics_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nAll checks\n"
        )
        f.write(
            "-" * 92
            + "\n"
        )
        f.write(
            checks_df.to_string(
                index=False
            )
        )
        f.write("\n")

    flag_path = (
        output_dir
        / "VALIDATION_PASSED.flag"
    )

    if passed:
        flag_path.write_text(
            "All critical joint wind-vessel "
            "dataset validation checks passed.\n",
            encoding="utf-8",
        )
    elif flag_path.exists():
        flag_path.unlink()

    # --------------------------------------------------------
    # Stage 8: plots and verdict
    # --------------------------------------------------------
    log(
        "[STAGE 8/8] Finalization"
    )

    if args.plots:
        log(
            "[PLOT] Generating SCI-style validation figures"
        )
        make_plots(
            output_dir=output_dir,
            split_integrity=split_integrity_df,
            physics_df=physics_df,
        )
        log(
            "[OK] Figures complete"
        )

    log("")
    log("=" * 92)
    log(
        "JOINT DATASET VALIDATION VERDICT"
    )
    log("=" * 92)
    log(
        f"Critical failures : "
        f"{critical_failures}"
    )
    log(
        f"Warnings          : "
        f"{warning_count}"
    )

    for row in split_integrity_rows:
        log(
            f"{row['split']:10s}: "
            f"samples={row['samples']:6d}, "
            f"X={row['X_shape']}, "
            f"y={row['y_shape']}, "
            f"apparent={row['apparent_shape']}"
        )

    log("")

    if passed:
        log(
            "[PASS] Joint wind-vessel dataset "
            "passed all critical checks."
        )
        log(
            f"[PASS] Flag written: "
            f"{flag_path}"
        )
        log(
            f"[DONE] Validation output: "
            f"{output_dir}"
        )
        return 0

    log(
        "[FAIL] Joint dataset has critical "
        "integrity failures."
    )
    log(
        f"[FAIL] Inspect: "
        f"{output_dir / 'validation_checks.csv'}"
    )

    return 2


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception:
        print(
            "\n[FATAL ERROR]",
            flush=True,
        )
        traceback.print_exc()
        print(
            "\nPlease send the complete traceback "
            "and terminal output.",
            flush=True,
        )
        sys.exit(1)

    sys.exit(
        exit_code
    )
