# -*- coding: utf-8 -*-
"""
08A_audit_external_sd1033.py

Audit-only external generalization preparation for:
- TPOS-2024_SD1033_1min.nc
- TPOS-2023_SD1033_1min.nc
- TPOS-2022_SD1033_1min.nc

No interpolation, no scaler fitting, no training, no tuning.

The script:
1) audits schema/metadata/missingness/time regularity;
2) finds continuous model-ready intervals for the frozen 60-min lookback and
   [1,2,3,5,10]-min horizons;
3) checks compatibility with the frozen SD1090 raw-variable schema;
4) optionally audits the common SD1033-2024 / SD1090-2024 ready period;
5) exports CSV/JSON/report and optional SCI-style figures.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_VERSION = "0.1.0-external-sd1033-audit"

DEFAULT_FILES = [
    r"D:\project\WindPredict_SaildroneData\data\raw\TPOS-2024_SD1033_1min.nc",
    r"D:\project\WindPredict_SaildroneData\data\raw\TPOS-2023_SD1033_1min.nc",
    r"D:\project\WindPredict_SaildroneData\data\raw\TPOS-2022_SD1033_1min.nc",
]
DEFAULT_REFERENCE = (
    r"D:\project\WindPredict_SaildroneData\data\raw\TPOS-2024_SD1090_1min.nc"
)

LOOKBACK_MIN = 60
HORIZONS_MIN = (1, 2, 3, 5, 10)
MIN_READY_ROWS = LOOKBACK_MIN + max(HORIZONS_MIN)
EXPECTED_STEP_S = 60.0

REQUIRED = [
    "UWND_MEAN",
    "VWND_MEAN",
    "TEMP_AIR_MEAN",
    "RH_MEAN",
    "SOG",
    "COG",
    "HDG",
    "WING_ANGLE",
]

ALIASES = {
    "TIME": ["time", "TIME", "timestamp", "time_utc"],
    "UWND_MEAN": [
        "UWND_MEAN", "uwnd_mean", "eastward_wind",
        "eastward_wind_mean", "wind_u_mean"
    ],
    "VWND_MEAN": [
        "VWND_MEAN", "vwnd_mean", "northward_wind",
        "northward_wind_mean", "wind_v_mean"
    ],
    "TEMP_AIR_MEAN": [
        "TEMP_AIR_MEAN", "temp_air_mean",
        "air_temperature", "air_temperature_mean", "TEMP_AIR"
    ],
    "RH_MEAN": [
        "RH_MEAN", "rh_mean", "relative_humidity",
        "relative_humidity_mean", "RH"
    ],
    "SOG": [
        "SOG", "sog", "platform_speed_wrt_ground", "speed_over_ground"
    ],
    "COG": [
        "COG", "cog", "platform_course", "course_over_ground"
    ],
    "HDG": [
        "HDG", "hdg", "platform_yaw_angle", "heading"
    ],
    "WING_ANGLE": [
        "WING_ANGLE", "wing_angle", "WINGANGLE"
    ],
    "LATITUDE": ["latitude", "LATITUDE", "lat", "LAT"],
    "LONGITUDE": ["longitude", "LONGITUDE", "lon", "LON"],
    "WIND_SPEED_MEAN": ["WIND_SPEED_MEAN", "wind_speed_mean", "wind_speed"],
    "WIND_FROM_MEAN": [
        "WIND_FROM_MEAN", "wind_from_mean", "WIND_FROM",
        "wind_from_direction"
    ],
}
OPTIONAL = ["LATITUDE", "LONGITUDE", "WIND_SPEED_MEAN", "WIND_FROM_MEAN"]

QC_RANGES = {
    "UWND_MEAN": (-80.0, 80.0),
    "VWND_MEAN": (-80.0, 80.0),
    "RH_MEAN": (-5.0, 105.0),
    "SOG": (0.0, 25.0),
    "COG": (-360.0, 720.0),
    "HDG": (-360.0, 720.0),
    "WING_ANGLE": (-720.0, 720.0),
}


def log(msg=""):
    print(msg, flush=True)


def import_xarray():
    try:
        import xarray as xr
    except Exception as exc:
        raise RuntimeError(
            "xarray is required; install xarray/netCDF4 in the active env."
        ) from exc
    return xr


def open_nc(xr, path):
    errors = []
    for kwargs in ({}, {"engine": "netcdf4"}, {"engine": "h5netcdf"}):
        try:
            return xr.open_dataset(
                path, decode_times=True, mask_and_scale=True, **kwargs
            )
        except Exception as exc:
            errors.append(f"{kwargs}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"Cannot open {path}\n" + "\n".join(errors)
    )


def dataset_label(path: Path):
    y = re.search(r"(20\d{2})", path.stem)
    s = re.search(r"(SD\d+)", path.stem, flags=re.I)
    year = y.group(1) if y else "unknown"
    saildrone = s.group(1).upper() if s else "unknown"
    return f"{saildrone}_{year}"


def all_names(ds):
    out = []
    for obj in (ds.variables, ds.coords, ds.data_vars):
        for name in obj:
            if name not in out:
                out.append(str(name))
    return out


def resolve(ds, key):
    names = all_names(ds)
    exact = set(names)
    lower = {n.lower(): n for n in names}

    for alias in ALIASES.get(key, [key]):
        if alias in exact:
            return alias
    for alias in ALIASES.get(key, [key]):
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def decode_time(ds, time_name):
    raw = np.squeeze(np.asarray(ds[time_name].values))
    if raw.ndim != 1:
        raise ValueError(
            f"time variable {time_name} is not 1-D: {raw.shape}"
        )
    try:
        idx = pd.to_datetime(raw, utc=True, errors="coerce")
    except Exception:
        idx = pd.to_datetime(
            [str(v) for v in raw], utc=True, errors="coerce"
        )
    idx = pd.DatetimeIndex(idx)
    if idx.isna().any():
        raise ValueError(
            f"{idx.isna().sum()} undecodable timestamps in {time_name}"
        )
    return idx


def to_numeric_1d(var, n, name):
    x = np.squeeze(np.asarray(var.values))
    if x.ndim == 0:
        return np.full(n, float(x), dtype=np.float64)
    if x.ndim == 1 and len(x) == n:
        return pd.to_numeric(
            pd.Series(x), errors="coerce"
        ).to_numpy(dtype=np.float64)

    axes = [i for i, size in enumerate(x.shape) if size == n]
    if len(axes) == 1:
        x = np.moveaxis(x, axes[0], 0)
        if int(np.prod(x.shape[1:])) == 1:
            x = x.reshape(n)
            return pd.to_numeric(
                pd.Series(x), errors="coerce"
            ).to_numpy(dtype=np.float64)

    raise ValueError(
        f"{name}: cannot safely align shape {x.shape} to time length {n}"
    )


def height_metadata(var):
    items = [
        f"{k}={v}"
        for k, v in var.attrs.items()
        if "height" in str(k).lower()
    ]
    return " | ".join(items) if items else None


def variable_stats(key, values):
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    n = len(x)
    nv = int(finite.sum())

    row = {
        "sample_count": n,
        "valid_count": nv,
        "missing_count": n - nv,
        "valid_fraction": nv / n if n else np.nan,
        "missing_fraction": 1.0 - nv / n if n else np.nan,
        "min": np.nan,
        "p01": np.nan,
        "median": np.nan,
        "mean": np.nan,
        "p99": np.nan,
        "max": np.nan,
        "std": np.nan,
        "qc_outside_count": np.nan,
        "qc_outside_fraction": np.nan,
    }

    if nv == 0:
        return row

    z = x[finite]
    row.update({
        "min": float(np.min(z)),
        "p01": float(np.quantile(z, 0.01)),
        "median": float(np.median(z)),
        "mean": float(np.mean(z)),
        "p99": float(np.quantile(z, 0.99)),
        "max": float(np.max(z)),
        "std": float(np.std(z)),
    })

    if key == "TEMP_AIR_MEAN":
        lo, hi = (
            (180.0, 340.0)
            if row["median"] > 150
            else (-60.0, 60.0)
        )
        bad = (z < lo) | (z > hi)
        row["qc_outside_count"] = int(bad.sum())
        row["qc_outside_fraction"] = float(bad.mean())
    elif key in QC_RANGES:
        lo, hi = QC_RANGES[key]
        bad = (z < lo) | (z > hi)
        row["qc_outside_count"] = int(bad.sum())
        row["qc_outside_fraction"] = float(bad.mean())

    return row


def contiguous_segments(mask, time, step_tol):
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != len(time):
        raise ValueError("mask/time length mismatch")
    if len(mask) == 0:
        return []

    dt = np.full(len(time), np.nan)
    if len(time) > 1:
        dt[1:] = np.diff(time.view("int64")) / 1e9

    step_ok = np.ones(len(time), dtype=bool)
    if len(time) > 1:
        step_ok[1:] = (
            np.isfinite(dt[1:])
            & (np.abs(dt[1:] - EXPECTED_STEP_S) <= step_tol)
        )

    out, start = [], None
    for i in range(len(mask)):
        if not mask[i]:
            if start is not None:
                out.append((start, i - 1))
                start = None
            continue

        if start is None:
            start = i
        elif not step_ok[i]:
            out.append((start, i - 1))
            start = i

    if start is not None:
        out.append((start, len(mask) - 1))

    return out


def missing_segments(mask):
    invalid = ~np.asarray(mask, dtype=bool)
    out, start = [], None
    for i, bad in enumerate(invalid):
        if bad and start is None:
            start = i
        elif not bad and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(invalid) - 1))
    return out


def ranked(segments):
    return sorted(
        segments,
        key=lambda p: p[1] - p[0] + 1,
        reverse=True,
    )


def segment_dict(label, kind, rank, start, end, time):
    rows = end - start + 1
    span_s = (time[end] - time[start]).total_seconds()
    return {
        "dataset": label,
        "segment_type": kind,
        "rank_by_length": rank,
        "start_index": start,
        "end_index": end,
        "sample_count": rows,
        "start_time_utc": time[start].isoformat(),
        "end_time_utc": time[end].isoformat(),
        "span_hours": span_s / 3600.0,
        "span_days": span_s / 86400.0,
        "usable_for_60min_lookback_plus_10min_horizon": (
            rows >= MIN_READY_ROWS
        ),
        "possible_dense_windows": max(0, rows - MIN_READY_ROWS + 1),
    }


def audit_one(
    xr,
    path,
    step_tol,
    top_segments,
    top_missing,
):
    if not path.exists():
        raise FileNotFoundError(path)

    label = dataset_label(path)
    log(f"[OPEN] {label}: {path}")

    ds = open_nc(xr, path)
    try:
        resolved = {
            key: resolve(ds, key)
            for key in ["TIME", *REQUIRED, *OPTIONAL]
        }

        if resolved["TIME"] is None:
            raise RuntimeError(f"{label}: no time variable found")

        time = decode_time(ds, resolved["TIME"])
        n = len(time)

        arrays = {}
        inventory = []
        schema = []

        for key in [*REQUIRED, *OPTIONAL]:
            actual = resolved[key]
            required = key in REQUIRED

            base = {
                "dataset": label,
                "logical_key": key,
                "required_for_frozen_model": required,
                "resolved_variable": actual,
                "present": actual is not None,
                "compatible_length": False,
                "units": None,
                "long_name": None,
                "standard_name": None,
                "height_metadata": None,
                "valid_fraction": np.nan,
            }

            if actual is None:
                schema.append(base)
                continue

            var = ds[actual]
            try:
                values = to_numeric_1d(var, n, actual)
                compatible = True
            except Exception as exc:
                values = np.full(n, np.nan, dtype=np.float64)
                compatible = False
                base["conversion_error"] = str(exc)

            arrays[key] = values
            stats = variable_stats(key, values)

            base.update({
                "compatible_length": compatible,
                "units": var.attrs.get("units"),
                "long_name": var.attrs.get("long_name"),
                "standard_name": var.attrs.get("standard_name"),
                "height_metadata": height_metadata(var),
                "valid_fraction": stats["valid_fraction"],
            })
            schema.append(base)

            inventory.append({
                "dataset": label,
                "logical_key": key,
                "actual_variable": actual,
                "required_for_frozen_model": required,
                "dimensions": ",".join(map(str, var.dims)),
                "shape": str(tuple(var.shape)),
                "dtype": str(var.dtype),
                "units": var.attrs.get("units"),
                "long_name": var.attrs.get("long_name"),
                "standard_name": var.attrs.get("standard_name"),
                "height_metadata": height_metadata(var),
                **stats,
            })

        missing_req = [
            key for key in REQUIRED
            if resolved[key] is None
        ]
        incompatible_req = [
            row["logical_key"]
            for row in schema
            if (
                row["required_for_frozen_model"]
                and row["present"]
                and not row["compatible_length"]
            )
        ]

        ready = np.ones(n, dtype=bool)
        for key in REQUIRED:
            if key not in arrays:
                ready[:] = False
                break
            ready &= np.isfinite(arrays[key])

        wind_ready = np.ones(n, dtype=bool)
        for key in ["UWND_MEAN", "VWND_MEAN"]:
            if key not in arrays:
                wind_ready[:] = False
                break
            wind_ready &= np.isfinite(arrays[key])

        ready_seg = contiguous_segments(
            ready, time, step_tol
        )
        wind_seg = contiguous_segments(
            wind_ready, time, step_tol
        )

        segment_rows = []
        for kind, segs in [
            ("model_ready", ready_seg),
            ("wind_uv_valid", wind_seg),
        ]:
            for rank_i, (a, b) in enumerate(
                ranked(segs)[:top_segments],
                start=1,
            ):
                segment_rows.append(
                    segment_dict(
                        label, kind, rank_i, a, b, time
                    )
                )

        missing_rows = []
        for key in REQUIRED:
            if key not in arrays:
                continue
            finite = np.isfinite(arrays[key])
            for rank_i, (a, b) in enumerate(
                ranked(missing_segments(finite))[:top_missing],
                start=1,
            ):
                span_s = (time[b] - time[a]).total_seconds()
                missing_rows.append({
                    "dataset": label,
                    "logical_key": key,
                    "rank_by_length": rank_i,
                    "start_index": a,
                    "end_index": b,
                    "sample_count": b - a + 1,
                    "start_time_utc": time[a].isoformat(),
                    "end_time_utc": time[b].isoformat(),
                    "span_hours": span_s / 3600.0,
                    "span_days": span_s / 86400.0,
                })

        if len(time) > 1:
            dt = np.diff(time.view("int64")) / 1e9
            regular = (
                np.abs(dt - EXPECTED_STEP_S) <= step_tol
            )
            bad_idx = np.where(~regular)[0]

            time_gap_rows = [{
                "dataset": label,
                "left_index": int(i),
                "right_index": int(i + 1),
                "left_time_utc": time[i].isoformat(),
                "right_time_utc": time[i + 1].isoformat(),
                "step_seconds": float(dt[i]),
                "is_duplicate": bool(dt[i] == 0),
                "is_backward": bool(dt[i] < 0),
            } for i in bad_idx]

            time_summary = {
                "strictly_increasing": bool(np.all(dt > 0)),
                "duplicate_timestamp_count": int(np.sum(dt == 0)),
                "negative_time_step_count": int(np.sum(dt < 0)),
                "non_60s_step_count": int(np.sum(~regular)),
                "median_step_seconds": float(np.median(dt)),
                "min_step_seconds": float(np.min(dt)),
                "max_step_seconds": float(np.max(dt)),
            }
        else:
            time_gap_rows = []
            time_summary = {
                "strictly_increasing": True,
                "duplicate_timestamp_count": 0,
                "negative_time_step_count": 0,
                "non_60s_step_count": 0,
                "median_step_seconds": np.nan,
                "min_step_seconds": np.nan,
                "max_step_seconds": np.nan,
            }

        longest = ranked(ready_seg)[0] if ready_seg else None
        if longest is not None:
            a, b = longest
            longest_rows = b - a + 1
            longest_days = (
                (time[b] - time[a]).total_seconds()
                / 86400.0
            )
            longest_start = time[a].isoformat()
            longest_end = time[b].isoformat()
        else:
            longest_rows = 0
            longest_days = 0.0
            longest_start = None
            longest_end = None

        attrs = {
            str(k): str(v)
            for k, v in ds.attrs.items()
        }

        summary = {
            "dataset": label,
            "file": str(path),
            "file_size_mb": path.stat().st_size / 1024.0**2,
            "sample_count": n,
            "time_variable": resolved["TIME"],
            "time_start_utc": (
                time[0].isoformat() if n else None
            ),
            "time_end_utc": (
                time[-1].isoformat() if n else None
            ),
            "duration_days": (
                (time[-1] - time[0]).total_seconds()
                / 86400.0 if n > 1 else 0.0
            ),
            "all_required_variables_present": (
                len(missing_req) == 0
            ),
            "all_required_variables_length_compatible": (
                len(incompatible_req) == 0
            ),
            "missing_required_variables": ",".join(missing_req),
            "incompatible_required_variables": ",".join(
                incompatible_req
            ),
            "wind_uv_valid_fraction": (
                float(wind_ready.mean()) if n else np.nan
            ),
            "model_ready_fraction": (
                float(ready.mean()) if n else np.nan
            ),
            "model_ready_segment_count": len(ready_seg),
            "model_ready_segments_ge_70_rows": sum(
                (b - a + 1) >= MIN_READY_ROWS
                for a, b in ready_seg
            ),
            "longest_model_ready_rows": longest_rows,
            "longest_model_ready_days": longest_days,
            "longest_model_ready_start_utc": longest_start,
            "longest_model_ready_end_utc": longest_end,
            "longest_model_ready_possible_windows": max(
                0, longest_rows - MIN_READY_ROWS + 1
            ),
            "global_title": attrs.get("title"),
            "platform_attr": (
                attrs.get("platform")
                or attrs.get("platform_name")
                or attrs.get("vehicle")
            ),
            "time_coverage_start_attr": attrs.get(
                "time_coverage_start"
            ),
            "time_coverage_end_attr": attrs.get(
                "time_coverage_end"
            ),
            "time_coverage_duration_attr": attrs.get(
                "time_coverage_duration"
            ),
            **time_summary,
        }

        return {
            "label": label,
            "path": path,
            "time": time,
            "ready": ready,
            "wind_ready": wind_ready,
            "arrays": arrays,
            "resolved": resolved,
            "summary": summary,
            "inventory": inventory,
            "schema": schema,
            "segments": segment_rows,
            "missing": missing_rows,
            "time_gaps": time_gap_rows,
        }

    finally:
        ds.close()


def common_ready_segments(
    a,
    b,
    step_tol,
    top_segments,
):
    ta = pd.DatetimeIndex(
        a["time"][a["ready"]]
    )
    tb = pd.DatetimeIndex(
        b["time"][b["ready"]]
    )

    common = ta.intersection(tb).sort_values()

    if len(common) == 0:
        return []

    segs = contiguous_segments(
        np.ones(len(common), dtype=bool),
        common,
        step_tol,
    )

    rows = []
    for rank_i, (i, j) in enumerate(
        ranked(segs)[:top_segments],
        start=1,
    ):
        row = segment_dict(
            f"{a['label']}__X__{b['label']}",
            "matched_model_ready",
            rank_i,
            i,
            j,
            common,
        )
        row["dataset_A"] = a["label"]
        row["dataset_B"] = b["label"]
        rows.append(row)

    return rows


def infer_output_dir(files):
    first = files[0]
    if first.parent.name.lower() == "raw":
        return (
            first.parent.parent
            / "audit"
            / "SD1033_external_audit_v0_1"
        )
    return first.parent / "SD1033_external_audit_v0_1"


def load_style():
    path = (
        Path(__file__).resolve().parent
        / "sci_plot_style.py"
    )
    if not path.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        "sci_plot_style_08a", str(path)
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_figure(fig, base, style):
    if (
        style is not None
        and hasattr(style, "save_figure")
    ):
        style.save_figure(fig, base)
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
        import matplotlib.pyplot as plt
        plt.close(fig)


def make_plots(externals, matched_rows, output_dir):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    style = load_style()
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def clean(ax):
        if (
            style is not None
            and hasattr(style, "clean_axis")
        ):
            style.clean_axis(ax, grid=True)
        else:
            ax.tick_params(
                which="both",
                direction="in",
                top=True,
                right=True,
            )
            ax.grid(True, alpha=0.25)

    # 01 ready coverage
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for y, audit in enumerate(externals):
        t = audit["time"][audit["ready"]]
        ax.scatter(
            t,
            np.full(len(t), y),
            s=1.0,
        )
    ax.set_yticks(np.arange(len(externals)))
    ax.set_yticklabels(
        [a["label"] for a in externals]
    )
    ax.set_xlabel("UTC time")
    ax.set_ylabel("Dataset")
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m")
    )
    clean(ax)
    save_figure(
        fig,
        fig_dir / "01_model_ready_coverage",
        style,
    )

    # 02 valid fractions
    rows = []
    for audit in externals:
        for key in REQUIRED:
            frac = (
                np.isfinite(audit["arrays"][key]).mean()
                if key in audit["arrays"]
                else 0.0
            )
            rows.append({
                "dataset": audit["label"],
                "variable": key,
                "valid_fraction": frac,
            })

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    x = np.arange(len(REQUIRED))
    width = 0.8 / max(len(externals), 1)

    for k, audit in enumerate(externals):
        g = (
            df.loc[df["dataset"] == audit["label"]]
            .set_index("variable")
            .reindex(REQUIRED)
        )
        ax.bar(
            x + (k - (len(externals)-1)/2) * width,
            g["valid_fraction"],
            width=width,
            label=audit["label"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        REQUIRED,
        rotation=35,
        ha="right",
    )
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Valid fraction")
    ax.legend(loc="best")
    clean(ax)
    save_figure(
        fig,
        fig_dir / "02_required_variable_valid_fraction",
        style,
    )

    # 03 longest segment
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    labels = [a["label"] for a in externals]
    vals = [
        a["summary"]["longest_model_ready_days"]
        for a in externals
    ]
    x = np.arange(len(labels))
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Longest model-ready segment (days)")
    clean(ax)
    save_figure(
        fig,
        fig_dir / "03_longest_ready_segment_days",
        style,
    )

    if matched_rows:
        md = pd.DataFrame(matched_rows)
        fig, ax = plt.subplots(figsize=(5.0, 3.1))
        x = np.arange(len(md))
        ax.bar(x, md["span_days"])
        ax.set_xticks(x)
        ax.set_xticklabels(
            md["rank_by_length"].astype(str)
        )
        ax.set_xlabel("Matched segment rank")
        ax.set_ylabel("Common model-ready span (days)")
        clean(ax)
        save_figure(
            fig,
            fig_dir / "04_matched_2024_ready_coverage",
            style,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
    )
    parser.add_argument(
        "--reference-sd1090",
        default=DEFAULT_REFERENCE,
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
        "--top-segments",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--top-missing-segments",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--plots",
        action="store_true",
    )

    args = parser.parse_args()

    files = [Path(p) for p in args.files]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else infer_output_dir(files)
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference = (
        Path(args.reference_sd1090)
        if args.reference_sd1090
        else None
    )

    xr = import_xarray()

    log("=" * 100)
    log("08A - EXTERNAL SD1033 AUDIT")
    log("=" * 100)
    log(f"script_version      : {SCRIPT_VERSION}")
    log(f"lookback            : {LOOKBACK_MIN} min")
    log(f"horizons            : {list(HORIZONS_MIN)} min")
    log(f"minimum ready rows  : {MIN_READY_ROWS}")
    log(f"output_dir          : {output_dir}")
    log("")

    externals = []
    errors = []

    for path in files:
        try:
            externals.append(
                audit_one(
                    xr,
                    path,
                    args.step_tolerance_seconds,
                    args.top_segments,
                    args.top_missing_segments,
                )
            )
        except Exception as exc:
            errors.append({
                "file": str(path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            log(
                f"[ERROR] {path}: "
                f"{type(exc).__name__}: {exc}"
            )

    if not externals:
        raise RuntimeError(
            "None of the SD1033 files could be audited."
        )

    reference_audit = None
    if reference is not None and reference.exists():
        try:
            reference_audit = audit_one(
                xr,
                reference,
                args.step_tolerance_seconds,
                args.top_segments,
                args.top_missing_segments,
            )
            log(
                f"[REFERENCE OK] {reference_audit['label']}"
            )
        except Exception as exc:
            errors.append({
                "file": str(reference),
                "role": "reference",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            log(
                f"[REFERENCE WARNING] {type(exc).__name__}: {exc}"
            )
    elif reference is not None:
        log(f"[REFERENCE SKIPPED] Not found: {reference}")

    all_audits = list(externals)
    if reference_audit is not None:
        all_audits.append(reference_audit)

    summary_df = pd.DataFrame(
        [a["summary"] for a in externals]
    )
    inventory_df = pd.DataFrame(
        [
            row
            for a in all_audits
            for row in a["inventory"]
        ]
    )
    schema_df = pd.DataFrame(
        [
            row
            for a in all_audits
            for row in a["schema"]
        ]
    )
    segments_df = pd.DataFrame(
        [
            row
            for a in all_audits
            for row in a["segments"]
        ]
    )
    missing_df = pd.DataFrame(
        [
            row
            for a in all_audits
            for row in a["missing"]
        ]
    )
    gaps_df = pd.DataFrame(
        [
            row
            for a in all_audits
            for row in a["time_gaps"]
        ]
    )

    matched_rows = []
    if reference_audit is not None:
        sd1033_2024 = next(
            (
                a for a in externals
                if (
                    "SD1033" in a["label"]
                    and "2024" in a["label"]
                )
            ),
            None,
        )
        if sd1033_2024 is not None:
            matched_rows = common_ready_segments(
                sd1033_2024,
                reference_audit,
                args.step_tolerance_seconds,
                args.top_segments,
            )

    matched_df = pd.DataFrame(matched_rows)

    summary_df.to_csv(
        output_dir / "external_file_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    inventory_df.to_csv(
        output_dir / "variable_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    schema_df.to_csv(
        output_dir / "schema_compatibility.csv",
        index=False,
        encoding="utf-8-sig",
    )
    segments_df.to_csv(
        output_dir / "valid_segments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    missing_df.to_csv(
        output_dir / "missing_segments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gaps_df.to_csv(
        output_dir / "time_gap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    matched_df.to_csv(
        output_dir / "matched_period_segments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    compatibility = {}
    for a in externals:
        s = a["summary"]
        recommendation = bool(
            s["all_required_variables_present"]
            and s["all_required_variables_length_compatible"]
            and s["longest_model_ready_rows"] >= MIN_READY_ROWS
        )
        compatibility[a["label"]] = {
            "recommended_for_08B_external_builder": recommendation,
            "longest_model_ready_rows": int(
                s["longest_model_ready_rows"]
            ),
            "longest_model_ready_start_utc": (
                s["longest_model_ready_start_utc"]
            ),
            "longest_model_ready_end_utc": (
                s["longest_model_ready_end_utc"]
            ),
            "possible_windows": int(
                s["longest_model_ready_possible_windows"]
            ),
        }

    manifest = {
        "script_version": SCRIPT_VERSION,
        "purpose": (
            "External audit only; no interpolation, scaler fit, "
            "model training or tuning."
        ),
        "external_files": [str(p) for p in files],
        "reference_sd1090": (
            str(reference)
            if reference is not None
            else None
        ),
        "frozen_protocol": {
            "lookback_minutes": LOOKBACK_MIN,
            "forecast_horizons_minutes": list(HORIZONS_MIN),
            "minimum_continuous_ready_rows": MIN_READY_ROWS,
            "required_raw_variables": REQUIRED,
            "expected_time_step_seconds": EXPECTED_STEP_S,
            "time_step_tolerance_seconds": (
                args.step_tolerance_seconds
            ),
        },
        "compatibility": compatibility,
        "matched_period_top_segment": (
            matched_rows[0]
            if matched_rows
            else None
        ),
        "errors": errors,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xarray": xr.__version__,
        },
    }

    with (
        output_dir / "external_audit_manifest.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with (
        output_dir / "external_audit_report.txt"
    ).open("w", encoding="utf-8") as f:
        f.write(
            "08A External SD1033 Audit Report\n"
        )
        f.write("=" * 100 + "\n\n")
        f.write(
            "No interpolation, no scaler fitting, no model training.\n"
        )
        f.write(
            f"lookback={LOOKBACK_MIN} min; "
            f"horizons={list(HORIZONS_MIN)} min; "
            f"minimum ready rows={MIN_READY_ROWS}\n\n"
        )

        f.write("External summary\n")
        f.write("-" * 100 + "\n")
        f.write(summary_df.to_string(index=False))

        f.write("\n\n08B compatibility\n")
        f.write("-" * 100 + "\n")
        for label, info in compatibility.items():
            f.write(
                f"{label}: recommended="
                f"{info['recommended_for_08B_external_builder']}, "
                f"longest_ready_rows="
                f"{info['longest_model_ready_rows']}, "
                f"possible_windows="
                f"{info['possible_windows']}, "
                f"interval="
                f"{info['longest_model_ready_start_utc']} -> "
                f"{info['longest_model_ready_end_utc']}\n"
            )

        f.write("\nMatched 2024 cross-vehicle segments\n")
        f.write("-" * 100 + "\n")
        if not matched_df.empty:
            f.write(matched_df.to_string(index=False))
        else:
            f.write("No matched model-ready segment computed.\n")

        if errors:
            f.write("\n\nWarnings/errors\n")
            f.write("-" * 100 + "\n")
            f.write(
                json.dumps(
                    errors,
                    ensure_ascii=False,
                    indent=2,
                )
            )

    if args.plots:
        make_plots(
            externals,
            matched_rows,
            output_dir,
        )

    log("")
    log("=" * 100)
    log("08A EXTERNAL SD1033 AUDIT RESULTS")
    log("=" * 100)

    for a in externals:
        s = a["summary"]
        log(f"{a['label']}:")
        log(
            f"  samples                   = "
            f"{s['sample_count']:,}"
        )
        log(
            f"  time                      = "
            f"{s['time_start_utc']} -> "
            f"{s['time_end_utc']}"
        )
        log(
            f"  model-ready fraction      = "
            f"{s['model_ready_fraction']:.6f}"
        )
        log(
            f"  longest ready segment     = "
            f"{s['longest_model_ready_rows']:,} rows "
            f"({s['longest_model_ready_days']:.3f} d)"
        )
        log(
            f"  possible 60->10m windows = "
            f"{s['longest_model_ready_possible_windows']:,}"
        )
        log(
            f"  non-60s time steps        = "
            f"{s['non_60s_step_count']:,}"
        )
        log(
            f"  recommended for 08B       = "
            f"{compatibility[a['label']]['recommended_for_08B_external_builder']}"
        )

    if matched_rows:
        top = matched_rows[0]
        log("")
        log("[MATCHED SD1033-2024 x SD1090-2024]")
        log(
            f"  longest common ready      = "
            f"{top['sample_count']:,} rows "
            f"({top['span_days']:.3f} d)"
        )
        log(
            f"  interval                  = "
            f"{top['start_time_utc']} -> "
            f"{top['end_time_utc']}"
        )
        log(
            f"  possible common windows   = "
            f"{top['possible_dense_windows']:,}"
        )

    if errors:
        log("")
        log(
            f"[WARNINGS] {len(errors)} item(s); "
            "see external_audit_manifest.json."
        )

    log("")
    log(f"[DONE] 08A outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n[FATAL ERROR]", flush=True)
        traceback.print_exc()
        print(
            "\nPlease send the complete traceback and terminal output.",
            flush=True,
        )
        sys.exit(1)
