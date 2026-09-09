# -*- coding: utf-8 -*-
"""
01_audit_saildrone_wind_v5_sci.py

Robust audit for TPOS-2024 SD1090 1-min Saildrone NetCDF data.

v5 explicitly calls sci_plot_style.apply_sci_style() and routes axis cleanup
and PNG/PDF export through sci_plot_style.py. The style file is now the
single source of truth for publication figure formatting.

Key changes from v1/v2:
0. Workaround for Windows/Matplotlib native crash during tight_layout:
   - force non-interactive Agg backend
   - disable external TeX rendering for this QC script
   - do not call tight_layout()
   - do not use bbox_inches='tight'
1. Exact loading of sci_plot_style.py from the same src directory.
2. faulthandler + full traceback for diagnosing abnormal exits.
3. Stage-by-stage progress messages with flush=True.
4. NetCDF is closed immediately after selected variables are loaded.
5. Missing-run detection is vectorized instead of looping over every row.
6. All later processing uses pandas/numpy only.
"""

from __future__ import annotations

import argparse
import faulthandler
import importlib.util
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

faulthandler.enable()

EXPECTED_DT_MIN = 1.0

TARGET_VARS = ["UWND_MEAN", "VWND_MEAN"]

WIND_DIAGNOSTIC_VARS = [
    "WIND_SPEED_MEAN",
    "WIND_FROM_MEAN",
    "GUST_WND_MEAN",
    "WWND_MEAN",
    "UWND_STDDEV",
    "VWND_STDDEV",
    "WWND_STDDEV",
    "WIND_MEASUREMENT_HEIGHT_MEAN",
    "WIND_SPEED_STDDEV",
]

CORE_INPUT_VARS = [
    "UWND_MEAN",
    "VWND_MEAN",
    "TEMP_AIR_MEAN",
    "RH_MEAN",
    "BARO_PRES_MEAN",
    "SOG",
    "COG",
    "HDG",
]

OPTIONAL_INPUT_VARS = [
    "latitude",
    "longitude",
    "ROLL_FILTERED_MEAN",
    "PITCH_FILTERED_MEAN",
    "WING_ANGLE",
    "WING_HDG_FILTERED_MEAN",
    "WAVE_SIGNIFICANT_HEIGHT",
    "WAVE_DOMINANT_PERIOD",
    "WATER_CURRENT_SPEED_MEAN",
    "WATER_CURRENT_DIRECTION_MEAN",
]

AUDIT_VARS = list(dict.fromkeys(
    TARGET_VARS
    + WIND_DIAGNOSTIC_VARS
    + CORE_INPUT_VARS
    + OPTIONAL_INPUT_VARS
))


def log(msg: str) -> None:
    print(msg, flush=True)


def load_plot_style():
    """
    Strictly load and APPLY sci_plot_style.py from the same src directory.

    Returns
    -------
    module
        Loaded sci_plot_style module.

    Notes
    -----
    Previous versions only imported sci_plot_style.py, but did not call
    apply_sci_style(). Therefore the figures did not actually inherit the
    publication rcParams. This version explicitly applies the style and uses
    the module's clean_axis()/finish_axes() and save_figure() helpers.
    """
    style_path = Path(__file__).resolve().parent / "sci_plot_style.py"

    if not style_path.exists():
        raise FileNotFoundError(
            "Required SCI style file was not found:\n"
            f"{style_path}\n"
            "Place sci_plot_style.py in the same src directory as this script."
        )

    spec = importlib.util.spec_from_file_location(
        "sci_plot_style",
        str(style_path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not create import spec for {style_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules["sci_plot_style"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "apply_sci_style"):
        raise AttributeError(
            "sci_plot_style.py does not define apply_sci_style()."
        )

    # Support both versions used in this project:
    #   apply_sci_style(base_font_size=8.5)
    #   apply_sci_style(font_size=9.0)
    selected_font = None
    try:
        selected_font = module.apply_sci_style(base_font_size=8.5)
    except TypeError:
        try:
            selected_font = module.apply_sci_style(font_size=8.5)
        except TypeError:
            selected_font = module.apply_sci_style()

    # Do not override rcParams here. The style file is the single source of truth.
    log(f"[OK] Loaded and APPLIED SCI style: {style_path}")
    if selected_font is not None:
        log(f"[OK] Selected font: {selected_font}")
    else:
        log(
            "[OK] SCI style applied. "
            f"font.family={plt.rcParams.get('font.family')}; "
            f"font.serif={plt.rcParams.get('font.serif')}"
        )
    log(
        "[OK] Export settings: "
        f"savefig.dpi={plt.rcParams.get('savefig.dpi')}, "
        f"pdf.fonttype={plt.rcParams.get('pdf.fonttype')}"
    )

    return module


def style_axis(style, ax, *, grid: bool = True) -> None:
    """Use the axis-finishing helper defined by sci_plot_style.py."""
    if hasattr(style, "clean_axis"):
        style.clean_axis(ax, grid=grid)
    elif hasattr(style, "finish_axes"):
        style.finish_axes(ax, grid=grid)
    else:
        # Only a compatibility fallback; rcParams are still from sci_plot_style.
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.minorticks_on()
        if grid:
            ax.grid(True, which="major")


def style_save_figure(style, fig, base: Path) -> None:
    """
    Save through sci_plot_style.save_figure() whenever available.

    This preserves the project's required 600 dpi PNG + vector PDF behavior.
    """
    base.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(style, "save_figure"):
        log(f"[PLOT] SCI save -> {base.name}.png / {base.name}.pdf")
        style.save_figure(fig, base)
        return

    # Strict compatible fallback if an older style file lacks save_figure().
    log(f"[PLOT] fallback save -> {base.name}.png / {base.name}.pdf")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def palette_value(style, candidates, default):
    """Take colors from sci_plot_style.PALETTE without redefining a project palette."""
    palette = getattr(style, "PALETTE", {})
    for key in candidates:
        if key in palette:
            return palette[key]
    return default

def find_time_variable(ds: xr.Dataset) -> str:
    for name in ["time", "TIME", "Time", "timestamp", "datetime"]:
        if name in ds.variables:
            return name

    for name in ds.variables:
        da = ds[name]
        if str(da.attrs.get("standard_name", "")).lower() == "time":
            return name
        if str(da.attrs.get("axis", "")).upper() == "T":
            return name
        try:
            if np.issubdtype(da.dtype, np.datetime64):
                return name
        except TypeError:
            pass

    raise RuntimeError("No time variable could be identified.")


def to_1d_array(
    ds: xr.Dataset,
    name: str,
    record_dim: str,
    n: int,
):
    if name not in ds.variables:
        return None

    da = ds[name]

    try:
        da = da.squeeze(drop=True)
    except Exception:
        pass

    if da.ndim != 1:
        return None

    if record_dim not in da.dims:
        return None

    if int(da.sizes[record_dim]) != n:
        return None

    return np.asarray(da.values).squeeze()


def build_dataframe(
    ds: xr.Dataset,
    time_name: str,
) -> tuple[pd.DataFrame, str]:
    time_da = ds[time_name].squeeze(drop=True)

    if time_da.ndim != 1:
        raise RuntimeError(
            f"Time variable {time_name!r} is not one-dimensional."
        )

    record_dim = time_da.dims[0]
    time = pd.to_datetime(time_da.values, errors="coerce")

    df = pd.DataFrame({"time": time})
    n = len(df)

    for k, name in enumerate(AUDIT_VARS, start=1):
        arr = to_1d_array(ds, name, record_dim, n)

        if arr is not None:
            df[name] = arr

        if k % 5 == 0 or k == len(AUDIT_VARS):
            log(
                f"[LOAD] checked {k:02d}/{len(AUDIT_VARS):02d} variables; "
                f"loaded={len(df.columns)-1}"
            )

    return df, record_dim


def numeric_series(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df[name], errors="coerce")


def finite_mask(df: pd.DataFrame, name: str) -> np.ndarray:
    x = numeric_series(df, name).to_numpy(dtype=np.float64)
    return np.isfinite(x)


def build_validity_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    names = [name for name in AUDIT_VARS if name in df.columns]

    for k, name in enumerate(names, start=1):
        s = numeric_series(df, name)
        values = s.to_numpy(dtype=np.float64)
        valid = np.isfinite(values)

        if valid.any():
            finite_values = values[valid]
            minimum = float(np.min(finite_values))
            maximum = float(np.max(finite_values))
            mean = float(np.mean(finite_values))
            std = float(np.std(finite_values, ddof=1)) if valid.sum() > 1 else np.nan
        else:
            minimum = maximum = mean = std = np.nan

        rows.append({
            "variable": name,
            "sample_count": len(df),
            "valid_count": int(valid.sum()),
            "missing_count": int((~valid).sum()),
            "valid_fraction": float(valid.mean()),
            "missing_fraction": float((~valid).mean()),
            "minimum": minimum,
            "maximum": maximum,
            "mean": mean,
            "std": std,
        })

        if k % 5 == 0 or k == len(names):
            log(f"[QC] validity {k:02d}/{len(names):02d}: {name}")

    return pd.DataFrame(rows)


def run_table(
    time: pd.Series,
    state_mask: np.ndarray,
    wanted_value: bool,
    variable: str,
    run_type: str,
) -> pd.DataFrame:
    """
    Vectorized consecutive-run detector.

    A run is broken by:
    - state change
    - non-1-min time step
    """
    t = pd.to_datetime(time, errors="coerce").reset_index(drop=True)
    mask = np.asarray(state_mask, dtype=bool)
    n = len(mask)

    columns = [
        "variable",
        "run_type",
        "start_time",
        "end_time",
        "sample_count",
        "duration_minutes",
        "duration_hours",
        "duration_days",
    ]

    if n == 0:
        return pd.DataFrame(columns=columns)

    dt = t.diff().dt.total_seconds().div(60.0).to_numpy()

    time_break = np.zeros(n, dtype=bool)
    if n > 1:
        time_break[1:] = (
            ~np.isfinite(dt[1:])
            | (np.abs(dt[1:] - EXPECTED_DT_MIN) > 1e-9)
        )

    state_change = np.zeros(n, dtype=bool)
    if n > 1:
        state_change[1:] = mask[1:] != mask[:-1]

    run_start = np.zeros(n, dtype=bool)
    run_start[0] = True
    run_start[1:] = state_change[1:] | time_break[1:]

    starts = np.flatnonzero(run_start)
    ends = np.r_[starts[1:] - 1, n - 1]

    rows = []

    for start, end in zip(starts, ends):
        if bool(mask[start]) != wanted_value:
            continue

        start_time = t.iloc[start]
        end_time = t.iloc[end]
        sample_count = int(end - start + 1)

        if pd.isna(start_time) or pd.isna(end_time):
            duration_minutes = np.nan
        else:
            duration_minutes = (
                (end_time - start_time).total_seconds() / 60.0
                + EXPECTED_DT_MIN
            )

        rows.append({
            "variable": variable,
            "run_type": run_type,
            "start_time": start_time,
            "end_time": end_time,
            "sample_count": sample_count,
            "duration_minutes": duration_minutes,
            "duration_hours": (
                duration_minutes / 60.0
                if np.isfinite(duration_minutes)
                else np.nan
            ),
            "duration_days": (
                duration_minutes / 1440.0
                if np.isfinite(duration_minutes)
                else np.nan
            ),
        })

    return pd.DataFrame(rows, columns=columns)


def joint_uv_valid_mask(df: pd.DataFrame) -> np.ndarray:
    for name in TARGET_VARS:
        if name not in df.columns:
            raise RuntimeError(f"Required variable not loaded: {name}")

    valid = (
        finite_mask(df, "UWND_MEAN")
        & finite_mask(df, "VWND_MEAN")
        & pd.to_datetime(
            df["time"],
            errors="coerce",
        ).notna().to_numpy()
    )

    return valid


def set_time_axis(ax) -> None:
    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator)
    )


def plot_outputs(
    df: pd.DataFrame,
    uv_valid: np.ndarray,
    fig_dir: Path,
    style,
) -> None:
    """
    Publication-style QC figures using sci_plot_style.py as the single style source.

    Figure convention
    -----------------
    - Times New Roman (or style-file fallback with warning)
    - inward ticks on all four sides
    - compact SCI font sizes
    - style-file line widths / grid
    - 600 dpi PNG
    - vector PDF
    - no large in-figure titles; information is carried by axis labels/legend
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Pull colors from the shared SCI palette rather than defining a new palette.
    color_u = palette_value(
        style,
        ["streamwise", "stage3", "proposed", "observed"],
        "#0072B2",
    )
    color_v = palette_value(
        style,
        ["crosswind", "proposed", "stable", "fitted"],
        "#D55E00",
    )
    color_main = palette_value(
        style,
        ["observed", "ohats", "near_neutral", "stage3"],
        "#000000",
    )
    color_missing = palette_value(
        style,
        ["background", "fill", "baseline"],
        "#7F7F7F",
    )

    # --------------------------------------------------------
    # 1. Full-mission U/V wind components
    # Double-column width, compact height.
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 3.0))

    ax.plot(
        df["time"],
        df["UWND_MEAN"],
        color=color_u,
        label=r"$U$",
        rasterized=False,
    )
    ax.plot(
        df["time"],
        df["VWND_MEAN"],
        color=color_v,
        label=r"$V$",
        rasterized=False,
    )

    ax.set_xlabel("Time")
    ax.set_ylabel(r"Wind component (m s$^{-1}$)")
    ax.legend(loc="best", ncol=2)

    set_time_axis(ax)
    style_axis(style, ax, grid=True)

    style_save_figure(
        style,
        fig,
        fig_dir / "01_UV_wind_full_mission",
    )

    # --------------------------------------------------------
    # 2. Mean wind speed
    # --------------------------------------------------------
    if "WIND_SPEED_MEAN" in df.columns:
        fig, ax = plt.subplots(figsize=(7.2, 2.8))

        ax.plot(
            df["time"],
            df["WIND_SPEED_MEAN"],
            color=color_main,
        )

        ax.set_xlabel("Time")
        ax.set_ylabel(r"Wind speed (m s$^{-1}$)")

        set_time_axis(ax)
        style_axis(style, ax, grid=True)

        style_save_figure(
            style,
            fig,
            fig_dir / "02_WIND_SPEED_MEAN",
        )

    # --------------------------------------------------------
    # 3. Wind-from direction
    # --------------------------------------------------------
    if "WIND_FROM_MEAN" in df.columns:
        fig, ax = plt.subplots(figsize=(7.2, 2.8))

        ax.plot(
            df["time"],
            df["WIND_FROM_MEAN"],
            color=color_main,
        )

        ax.set_xlabel("Time")
        ax.set_ylabel(r"Wind direction ($^\circ$)")
        ax.set_ylim(0.0, 360.0)
        ax.set_yticks([0, 90, 180, 270, 360])

        set_time_axis(ax)
        style_axis(style, ax, grid=True)

        style_save_figure(
            style,
            fig,
            fig_dir / "03_WIND_FROM_MEAN",
        )

    # --------------------------------------------------------
    # 4. Joint U/V data validity
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 2.2))

    ax.fill_between(
        df["time"],
        0.0,
        uv_valid.astype(float),
        step="post",
        color=color_u,
        alpha=0.55,
        linewidth=0.0,
    )
    ax.step(
        df["time"],
        uv_valid.astype(float),
        where="post",
        color=color_main,
        linewidth=0.8,
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("U/V availability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Missing", "Valid"])

    set_time_axis(ax)
    style_axis(style, ax, grid=False)

    style_save_figure(
        style,
        fig,
        fig_dir / "04_joint_UV_validity",
    )

    # --------------------------------------------------------
    # 5. Missing-data map
    # --------------------------------------------------------
    variables = [
        v
        for v in CORE_INPUT_VARS + OPTIONAL_INPUT_VARS
        if v in df.columns
    ]

    if variables:
        max_columns = 3000
        stride = max(
            1,
            int(np.ceil(len(df) / max_columns)),
        )

        matrix = np.vstack(
            [
                (~finite_mask(df, name))[::stride].astype(np.uint8)
                for name in variables
            ]
        )

        # Use a simple two-level grayscale-compatible colormap assembled
        # from the shared style palette.
        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(["white", color_missing])

        fig_height = max(4.0, 0.23 * len(variables) + 1.0)
        fig, ax = plt.subplots(figsize=(7.2, fig_height))

        ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )

        ax.set_yticks(np.arange(len(variables)))
        ax.set_yticklabels(variables)
        ax.set_xlabel(
            f"Mission sample index (display stride = {stride})"
        )
        ax.set_ylabel("Variable")

        # imshow should not use minor ticks/grid.
        ax.tick_params(which="both", direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(plt.rcParams["axes.linewidth"])

        style_save_figure(
            style,
            fig,
            fig_dir / "05_missingness_map",
        )

def write_summary(
    output_file: Path,
    attrs: dict,
    df: pd.DataFrame,
    record_dim: str,
    uv_valid: np.ndarray,
    segments: pd.DataFrame,
    missing_uv: pd.DataFrame,
) -> None:
    t = pd.to_datetime(df["time"], errors="coerce")
    dt = t.diff().dt.total_seconds().div(60.0)

    irregular = int(
        (
            dt.iloc[1:].notna()
            & (np.abs(dt.iloc[1:] - EXPECTED_DT_MIN) > 1e-9)
        ).sum()
    )

    longest = None
    if not segments.empty:
        longest = segments.sort_values(
            "sample_count",
            ascending=False,
        ).iloc[0]

    sustained = (
        missing_uv.loc[
            missing_uv["duration_hours"] >= 6.0
        ].sort_values("start_time")
        if not missing_uv.empty
        else pd.DataFrame()
    )

    with output_file.open("w", encoding="utf-8") as f:
        f.write("Saildrone TPOS-2024 SD1090 data audit\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset title: {attrs.get('title', '')}\n")
        f.write(f"Mission: {attrs.get('mission', '')}\n")
        f.write(f"Drone ID: {attrs.get('drone_id', '')}\n")
        f.write(f"Record dimension: {record_dim}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Start: {t.min()}\n")
        f.write(f"End: {t.max()}\n")
        f.write(f"Non-1-min transitions: {irregular}\n\n")

        f.write("Joint U/V target availability\n")
        f.write("-" * 80 + "\n")
        f.write(f"Valid rows: {int(uv_valid.sum())}\n")
        f.write(f"Missing rows: {int((~uv_valid).sum())}\n")
        f.write(f"Valid fraction: {uv_valid.mean():.8f}\n")
        f.write(f"Continuous valid segments: {len(segments)}\n")

        if longest is not None:
            f.write("\nLongest continuous valid U/V segment\n")
            f.write(f"Start: {longest['start_time']}\n")
            f.write(f"End: {longest['end_time']}\n")
            f.write(f"Samples: {int(longest['sample_count'])}\n")
            f.write(f"Duration days: {longest['duration_days']:.6f}\n")

        if not sustained.empty:
            first = sustained.iloc[0]
            f.write("\nFirst sustained U/V outage >= 6 h\n")
            f.write(f"Start: {first['start_time']}\n")
            f.write(f"End: {first['end_time']}\n")
            f.write(f"Samples: {int(first['sample_count'])}\n")
            f.write(f"Duration hours: {first['duration_hours']:.6f}\n")
        else:
            f.write("\nNo sustained U/V outage >= 6 h detected.\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nc",
        required=True,
        help="Input Saildrone NetCDF file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output audit directory",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate SCI-style figures using sci_plot_style.py.",
    )
    args = parser.parse_args()

    nc_file = Path(args.nc)
    out_dir = Path(args.output)
    fig_dir = out_dir / "figures"

    if not nc_file.exists():
        raise FileNotFoundError(nc_file)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log("[STAGE 0/8] Loading and applying plotting style")
    style = load_plot_style()

    log("[STAGE 1/8] Opening NetCDF")
    log(f"[INFO] {nc_file}")

    ds = xr.open_dataset(
        nc_file,
        decode_times=True,
    )

    attrs = dict(ds.attrs)

    try:
        time_name = find_time_variable(ds)
        df, record_dim = build_dataframe(
            ds,
            time_name,
        )
    finally:
        # From here onward, everything is in memory.
        ds.close()
        log("[OK] NetCDF closed after variable extraction")

    log(f"[INFO] Time variable: {time_name}")
    log(f"[INFO] Record dimension: {record_dim}")
    log(f"[INFO] Rows: {len(df)}")
    log(
        f"[INFO] Selected columns found: "
        f"{len(df.columns) - 1}"
    )

    log("[STAGE 2/8] Computing variable validity")
    validity = build_validity_table(df)
    validity.to_csv(
        out_dir / "variable_validity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    log("[OK] variable_validity.csv")

    log("[STAGE 3/8] Detecting missing runs")
    missing_frames = []

    names = [
        name for name in AUDIT_VARS
        if name in df.columns
    ]

    for k, name in enumerate(names, start=1):
        valid = finite_mask(df, name)

        runs = run_table(
            time=df["time"],
            state_mask=valid,
            wanted_value=False,
            variable=name,
            run_type="missing",
        )

        if not runs.empty:
            missing_frames.append(runs)

        if k % 5 == 0 or k == len(names):
            log(
                f"[RUNS] {k:02d}/{len(names):02d}: "
                f"{name}"
            )

    if missing_frames:
        missing_runs = pd.concat(
            missing_frames,
            ignore_index=True,
        ).sort_values(
            ["variable", "start_time"]
        )
    else:
        missing_runs = pd.DataFrame(
            columns=[
                "variable",
                "run_type",
                "start_time",
                "end_time",
                "sample_count",
                "duration_minutes",
                "duration_hours",
                "duration_days",
            ]
        )

    missing_runs.to_csv(
        out_dir / "missing_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    log("[OK] missing_runs.csv")

    log("[STAGE 4/8] Detecting continuous joint U/V segments")
    uv_valid = joint_uv_valid_mask(df)

    segments = run_table(
        time=df["time"],
        state_mask=uv_valid,
        wanted_value=True,
        variable="JOINT_UV",
        run_type="valid",
    )

    missing_uv = run_table(
        time=df["time"],
        state_mask=uv_valid,
        wanted_value=False,
        variable="JOINT_UV",
        run_type="missing",
    )

    segments.to_csv(
        out_dir / "joint_wind_valid_segments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missing_uv.to_csv(
        out_dir / "joint_wind_missing_segments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("[OK] joint U/V segment tables")

    log("[STAGE 5/8] Saving selected raw variables")
    df.to_csv(
        out_dir / "selected_variables_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    log("[OK] selected_variables_raw.csv")

    log("[STAGE 6/8] Writing summary")
    write_summary(
        output_file=out_dir / "qc_summary.txt",
        attrs=attrs,
        df=df,
        record_dim=record_dim,
        uv_valid=uv_valid,
        segments=segments,
        missing_uv=missing_uv,
    )
    log("[OK] qc_summary.txt")

    log("[STAGE 7/8] Figure output")
    if args.plots:
        log("[INFO] --plots enabled; attempting Matplotlib output.")
        plot_outputs(
            df=df,
            uv_valid=uv_valid,
            fig_dir=fig_dir,
            style=style,
        )
        log("[OK] figures complete")
    else:
        log("[SKIP] Figures not requested. Re-run with --plots to export SCI-style figures.")

    log("[STAGE 8/8] Final report")
    log("=" * 80)
    log("JOINT U/V WIND AUDIT")
    log("=" * 80)
    log(f"Valid rows    : {int(uv_valid.sum())}")
    log(f"Missing rows  : {int((~uv_valid).sum())}")
    log(f"Valid ratio   : {uv_valid.mean():.4%}")
    log(f"Valid segments: {len(segments)}")

    if not segments.empty:
        longest = segments.sort_values(
            "sample_count",
            ascending=False,
        ).iloc[0]

        log("")
        log("Longest continuous valid segment:")
        log(f"  start   : {longest['start_time']}")
        log(f"  end     : {longest['end_time']}")
        log(f"  samples : {int(longest['sample_count'])}")
        log(f"  days    : {longest['duration_days']:.3f}")

    sustained = (
        missing_uv.loc[
            missing_uv["duration_hours"] >= 6.0
        ].sort_values("start_time")
        if not missing_uv.empty
        else pd.DataFrame()
    )

    if not sustained.empty:
        first = sustained.iloc[0]
        log("")
        log("First sustained U/V outage >= 6 h:")
        log(f"  start : {first['start_time']}")
        log(f"  end   : {first['end_time']}")
        log(f"  hours : {first['duration_hours']:.3f}")

    log("")
    log(f"Output directory: {out_dir}")
    log("[DONE] Audit completed successfully.")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        print("\n[FATAL ERROR]", flush=True)
        traceback.print_exc()
        print(
            "\nPlease copy everything from [FATAL ERROR] "
            "to the end and send it back.",
            flush=True,
        )
        sys.exit(1)

    sys.exit(exit_code)
