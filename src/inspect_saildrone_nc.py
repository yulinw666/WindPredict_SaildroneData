# -*- coding: utf-8 -*-

"""
inspect_saildrone_nc.py

功能：
1. 读取 Saildrone NetCDF (.nc) 文件
2. 查看 dataset 的 dimensions / coordinates / variables
3. 自动识别时间坐标
4. 导出变量清单 variable_inventory.csv
5. 将所有一维时间序列变量整理成 CSV
6. 自动绘制变量时序曲线
7. 保存 600 dpi PNG + vector PDF

依赖：
    xarray
    netCDF4
    numpy
    pandas
    matplotlib

安装：
    pip install xarray netCDF4 numpy pandas matplotlib
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# 1. 用户设置
# ============================================================

# 修改成你的 .nc 文件路径
NC_FILE = Path(
    r"D:\project\WindPredict_SaildroneData\data\raw\TPOS-2024_SD1090_1min.nc"
)

# 输出文件夹
OUTPUT_DIR = Path(
    r"D:\project\WindPredict_SaildroneData\data\processed"
)

# ------------------------------------------------------------
# 需要绘制哪些变量
#
# None：
#     自动绘制所有一维数值型时间序列
#
# 例如：
# PLOT_VARIABLES = [
#     "TEMP_AIR_MEAN",
#     "RH_MEAN",
#     "BARO_PRES_MEAN",
#     "UWND_MEAN",
#     "VWND_MEAN",
# ]
# ------------------------------------------------------------

PLOT_VARIABLES = None


# 自动绘图时，最多绘制多少个变量
# 如果你想全部画，可以设置 None
MAX_AUTO_PLOTS = None


# 时序表格预览行数
PREVIEW_ROWS = 2000


# ============================================================
# 2. 绘图风格
# ============================================================

# 如果你的 sci_plot_style.py 在同目录或 Python path 中，
# 会尝试自动导入。
try:
    import sci_plot_style  # noqa: F401
    print("[INFO] sci_plot_style.py loaded.")
except Exception as exc:
    warnings.warn(
        f"sci_plot_style.py could not be loaded: {exc}\n"
        "Using fallback Matplotlib settings."
    )

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "Times New Roman",
        "DejaVu Serif"
    ]
    plt.rcParams["mathtext.fontset"] = "stix"

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9


# ============================================================
# 3. 工具函数
# ============================================================

def clean_filename(name):
    """
    将变量名转换成安全文件名。
    """
    invalid_chars = '<>:"/\\|?*'

    result = str(name)

    for c in invalid_chars:
        result = result.replace(c, "_")

    return result


def find_time_coordinate(ds):
    """
    自动寻找时间坐标。

    优先寻找：
        time
        TIME
        Time
        timestamp
        datetime

    如果找不到，再根据 dtype 判断。
    """

    candidates = [
        "time",
        "TIME",
        "Time",
        "timestamp",
        "Timestamp",
        "datetime",
        "date_time",
    ]

    # --------------------------------------------------------
    # 方法 1：按变量名称寻找
    # --------------------------------------------------------

    for name in candidates:
        if name in ds.coords:
            return name

        if name in ds.variables:
            return name

    # --------------------------------------------------------
    # 方法 2：查看变量属性
    # --------------------------------------------------------

    for name in ds.variables:

        var = ds[name]

        standard_name = str(
            var.attrs.get("standard_name", "")
        ).lower()

        long_name = str(
            var.attrs.get("long_name", "")
        ).lower()

        axis = str(
            var.attrs.get("axis", "")
        ).upper()

        if (
            standard_name == "time"
            or axis == "T"
            or "time" in long_name
        ):
            return name

    # --------------------------------------------------------
    # 方法 3：datetime64 dtype
    # --------------------------------------------------------

    for name in ds.variables:

        var = ds[name]

        try:
            if np.issubdtype(var.dtype, np.datetime64):
                return name
        except TypeError:
            pass

    return None


def convert_time_values(time_values):
    """
    尝试将 NetCDF 时间转换为 pandas datetime。
    如果失败，则保留原值。
    """

    try:
        time_index = pd.to_datetime(time_values)

        if time_index.notna().sum() > 0:
            return time_index

    except Exception:
        pass

    # 对 cftime 或特殊时间类型
    try:
        return pd.Index(
            [str(t) for t in time_values],
            name="time"
        )

    except Exception:
        return pd.Index(time_values, name="time")


def is_numeric_array(var):
    """
    判断变量是否为数值变量。
    """

    try:
        return np.issubdtype(var.dtype, np.number)

    except TypeError:
        return False


def extract_1d_time_series(ds, variable_name, time_name):
    """
    提取与时间记录维度对应的一维变量。

    兼容这种 Saildrone NetCDF 结构：

        Dimensions:
            row: 144000

        Variables:
            time       (row)
            UWND_MEAN  (row)
            VWND_MEAN  (row)

    注意：
    time 是时间变量名称，
    row 才是真正的时间记录维度名称。
    """

    da = ds[variable_name]

    # --------------------------------------------------------
    # 1. 找到 time 所对应的真正维度
    # --------------------------------------------------------

    time_da = ds[time_name]

    try:
        time_da = time_da.squeeze(drop=True)
    except Exception:
        pass

    if time_da.ndim != 1:
        return None

    # 对你的 Saildrone 数据，这里就是 "row"
    record_dim = time_da.dims[0]

    # --------------------------------------------------------
    # 2. 去掉长度为 1 的无用维度
    # --------------------------------------------------------

    try:
        da = da.squeeze(drop=True)
    except Exception:
        pass

    # --------------------------------------------------------
    # 3. 判断变量是否沿着记录维度变化
    # --------------------------------------------------------

    if record_dim not in da.dims:
        return None

    # --------------------------------------------------------
    # 4. 目前只处理一维时间序列
    # --------------------------------------------------------

    if da.ndim != 1:
        return None

    # --------------------------------------------------------
    # 5. 保证长度和时间一致
    # --------------------------------------------------------

    if da.sizes[record_dim] != time_da.sizes[record_dim]:
        return None

    return da


def variable_description(var):
    """
    从 NetCDF attributes 获取变量说明。
    """

    candidates = [
        "long_name",
        "standard_name",
        "description",
        "comment",
    ]

    for key in candidates:

        value = var.attrs.get(key)

        if value is not None:
            return str(value)

    return ""


def get_units(var):
    """
    获取变量单位。
    """

    value = var.attrs.get("units", "")

    return str(value)


# ============================================================
# 4. 输出 NetCDF 总体信息
# ============================================================

def save_dataset_overview(ds, output_file):
    """
    将 xarray Dataset 信息保存到 txt。
    """

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("=" * 80 + "\n")
        f.write("XARRAY DATASET OVERVIEW\n")
        f.write("=" * 80 + "\n\n")

        f.write(str(ds))

        f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("GLOBAL ATTRIBUTES\n")
        f.write("=" * 80 + "\n\n")

        for key, value in ds.attrs.items():
            f.write(f"{key}: {value}\n")


# ============================================================
# 5. 创建变量清单
# ============================================================

def build_variable_inventory(ds, time_name):
    """
    创建变量信息表。
    """

    rows = []

    all_names = list(ds.variables)

    for name in all_names:

        var = ds[name]

        if name in ds.coords:
            role = "coordinate"
        else:
            role = "data_variable"

        dims = ", ".join(var.dims)

        shape = " x ".join(
            str(v) for v in var.shape
        )

        units = get_units(var)

        description = variable_description(var)

        numeric = is_numeric_array(var)

        is_time_series = False

        if time_name is not None:

            temp = extract_1d_time_series(
                ds,
                name,
                time_name
            )

            if temp is not None:
                is_time_series = True

        row = {
            "variable": name,
            "role": role,
            "dtype": str(var.dtype),
            "dimensions": dims,
            "shape": shape,
            "units": units,
            "description": description,
            "numeric": numeric,
            "one_dimensional_time_series": is_time_series,
        }

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 6. 将时间序列转换成 DataFrame
# ============================================================

def build_timeseries_dataframe(ds, time_name):
    """
    将所有和时间记录维度对应的一维变量转换为 DataFrame。
    """

    if time_name is None:
        raise RuntimeError(
            "No time variable was detected."
        )

    # ========================================================
    # 获取时间
    # ========================================================

    time_da = ds[time_name]

    try:
        time_da = time_da.squeeze(drop=True)
    except Exception:
        pass

    if time_da.ndim != 1:
        raise RuntimeError(
            f"Time variable '{time_name}' is not one-dimensional."
        )

    # 真正的数据记录维度
    record_dim = time_da.dims[0]

    print(
        f"[INFO] Time variable : {time_name}"
    )

    print(
        f"[INFO] Record dimension : {record_dim}"
    )

    print(
        f"[INFO] Number of records : "
        f"{time_da.sizes[record_dim]}"
    )

    time_values = time_da.values

    time_index = convert_time_values(
        time_values
    )

    data = {
        "time": time_index
    }

    included_variables = []

    skipped_variables = []

    # ========================================================
    # 遍历全部变量
    # ========================================================

    for name in ds.variables:

        # time 已经添加
        if name == time_name:
            continue

        da = ds[name]

        # 去掉 singleton dimensions
        try:
            da = da.squeeze(drop=True)
        except Exception:
            pass

        # ----------------------------------------------
        # 必须包含 row 维度
        # ----------------------------------------------

        if record_dim not in da.dims:

            skipped_variables.append(
                name
            )

            continue

        # ----------------------------------------------
        # 这里只处理一维变量
        # ----------------------------------------------

        if da.ndim != 1:

            skipped_variables.append(
                name
            )

            continue

        # ----------------------------------------------
        # 长度必须一致
        # ----------------------------------------------

        if (
            da.sizes[record_dim]
            !=
            len(time_index)
        ):

            skipped_variables.append(
                name
            )

            continue

        # ----------------------------------------------
        # 提取
        # ----------------------------------------------

        try:

            values = np.asarray(
                da.values
            ).squeeze()

            data[name] = values

            included_variables.append(
                name
            )

        except Exception as exc:

            print(
                f"[WARNING] Failed to read "
                f"{name}: {exc}"
            )

            skipped_variables.append(
                name
            )

    # ========================================================
    # 转 DataFrame
    # ========================================================

    df = pd.DataFrame(
        data
    )

    return (
        df,
        included_variables,
        skipped_variables
    )


# ============================================================
# 7. 绘图
# ============================================================

def plot_time_series(
    df,
    ds,
    variable_name,
    output_dir
):
    """
    绘制单个变量时序。
    """

    if variable_name not in df.columns:
        return

    y = df[variable_name]

    # 只画数值变量
    if not pd.api.types.is_numeric_dtype(y):
        return

    x = df["time"]

    units = ""

    description = ""

    if variable_name in ds.variables:

        units = get_units(
            ds[variable_name]
        )

        description = variable_description(
            ds[variable_name]
        )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7.2, 3.6)
    )

    ax.plot(
        x,
        y,
        linewidth=0.8
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel("Time")

    if units:
        ax.set_ylabel(
            f"{variable_name} ({units})"
        )
    else:
        ax.set_ylabel(
            variable_name
        )

    if description:

        ax.set_title(
            f"{variable_name}: {description}"
        )

    else:

        ax.set_title(
            variable_name
        )

    # --------------------------------------------------------
    # 时间坐标优化
    # --------------------------------------------------------

    if pd.api.types.is_datetime64_any_dtype(x):

        locator = mdates.AutoDateLocator(
            minticks=4,
            maxticks=8
        )

        formatter = mdates.ConciseDateFormatter(
            locator
        )

        ax.xaxis.set_major_locator(
            locator
        )

        ax.xaxis.set_major_formatter(
            formatter
        )

    ax.grid(
        True,
        linewidth=0.4,
        alpha=0.25
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    safe_name = clean_filename(
        variable_name
    )

    png_file = (
        output_dir /
        f"{safe_name}.png"
    )

    pdf_file = (
        output_dir /
        f"{safe_name}.pdf"
    )

    fig.savefig(
        png_file,
        dpi=600,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_file,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[PLOT] {variable_name}"
    )


# ============================================================
# 8. 主程序
# ============================================================

def main():

    # --------------------------------------------------------
    # 路径检查
    # --------------------------------------------------------

    if not NC_FILE.exists():

        raise FileNotFoundError(
            f"\nNetCDF file does not exist:\n"
            f"{NC_FILE}\n"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_dir = (
        OUTPUT_DIR /
        "timeseries_plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 打开 NetCDF
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("OPENING NETCDF")
    print("=" * 80)

    print(
        f"File:\n{NC_FILE}"
    )

    ds = xr.open_dataset(
        NC_FILE,
        decode_times=True
    )

    # --------------------------------------------------------
    # Dataset 信息
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("DATASET")
    print("=" * 80)

    print(ds)

    # 保存 Dataset 信息
    overview_file = (
        OUTPUT_DIR /
        "dataset_overview.txt"
    )

    save_dataset_overview(
        ds,
        overview_file
    )

    # --------------------------------------------------------
    # 时间变量
    # --------------------------------------------------------

    time_name = find_time_coordinate(
        ds
    )

    print()
    print("=" * 80)
    print("TIME COORDINATE")
    print("=" * 80)

    if time_name is None:

        print(
            "[WARNING] No time coordinate detected."
        )

    else:

        print(
            f"Detected time variable: {time_name}"
        )

        print(
            ds[time_name]
        )

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("DIMENSIONS")
    print("=" * 80)

    for dim, size in ds.sizes.items():

        print(
            f"{dim:30s} : {size}"
        )

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("VARIABLES")
    print("=" * 80)

    for i, name in enumerate(
        ds.variables,
        start=1
    ):

        var = ds[name]

        units = get_units(var)

        desc = variable_description(var)

        print(
            f"{i:3d}. "
            f"{name:35s} "
            f"dims={str(var.dims):30s} "
            f"shape={str(var.shape):20s} "
            f"units={units}"
        )

        if desc:

            print(
                f"     {desc}"
            )

    # --------------------------------------------------------
    # 变量清单
    # --------------------------------------------------------

    inventory = build_variable_inventory(
        ds,
        time_name
    )

    inventory_file = (
        OUTPUT_DIR /
        "variable_inventory.csv"
    )

    inventory.to_csv(
        inventory_file,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        f"[SAVED] {inventory_file}"
    )

    # --------------------------------------------------------
    # 构建时序 DataFrame
    # --------------------------------------------------------

    if time_name is None:

        print(
            "\nNo time coordinate detected. "
            "Timeseries extraction skipped."
        )

        ds.close()

        return

    (
        df,
        included_variables,
        skipped_variables
    ) = build_timeseries_dataframe(
        ds,
        time_name
    )

    # --------------------------------------------------------
    # 打印表格基本信息
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("TIME-SERIES TABLE")
    print("=" * 80)

    print(
        f"Rows    : {len(df)}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    print()
    print("Included variables:")

    for name in included_variables:

        print(
            f"  - {name}"
        )

    if skipped_variables:

        print()
        print(
            "Skipped multi-dimensional variables:"
        )

        for name in skipped_variables:

            print(
                f"  - {name}"
            )

    # --------------------------------------------------------
    # 打印前 20 行
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FIRST 20 ROWS")
    print("=" * 80)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        250
    ):

        print(
            df.head(20)
        )

    # --------------------------------------------------------
    # 保存完整 CSV
    # --------------------------------------------------------

    full_csv = (
        OUTPUT_DIR /
        "timeseries_all.csv"
    )

    df.to_csv(
        full_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        f"[SAVED] {full_csv}"
    )

    # --------------------------------------------------------
    # 保存预览 CSV
    # --------------------------------------------------------

    preview_csv = (
        OUTPUT_DIR /
        "timeseries_preview.csv"
    )

    df.head(
        PREVIEW_ROWS
    ).to_csv(
        preview_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"[SAVED] {preview_csv}"
    )

    # --------------------------------------------------------
    # 缺失率统计
    # --------------------------------------------------------

    missing_rows = []

    for column in df.columns:

        if column == "time":
            continue

        series = df[column]

        missing_count = (
            series.isna().sum()
        )

        missing_fraction = (
            missing_count /
            len(series)
            if len(series) > 0
            else np.nan
        )

        row = {
            "variable": column,
            "sample_count": len(series),
            "missing_count": missing_count,
            "missing_fraction": missing_fraction,
        }

        if pd.api.types.is_numeric_dtype(
            series
        ):

            row["minimum"] = (
                series.min(skipna=True)
            )

            row["maximum"] = (
                series.max(skipna=True)
            )

            row["mean"] = (
                series.mean(skipna=True)
            )

            row["std"] = (
                series.std(skipna=True)
            )

        missing_rows.append(row)

    stats_df = pd.DataFrame(
        missing_rows
    )

    stats_file = (
        OUTPUT_DIR /
        "timeseries_statistics.csv"
    )

    stats_df.to_csv(
        stats_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"[SAVED] {stats_file}"
    )

    # --------------------------------------------------------
    # 绘制变量
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("PLOTTING")
    print("=" * 80)

    if PLOT_VARIABLES is None:

        plot_variables = []

        for column in df.columns:

            if column == "time":
                continue

            if pd.api.types.is_numeric_dtype(
                df[column]
            ):

                plot_variables.append(
                    column
                )

        if MAX_AUTO_PLOTS is not None:

            plot_variables = (
                plot_variables[
                    :MAX_AUTO_PLOTS
                ]
            )

    else:

        plot_variables = (
            PLOT_VARIABLES
        )

    for variable_name in plot_variables:

        if variable_name not in df.columns:

            print(
                f"[WARNING] Variable not found: "
                f"{variable_name}"
            )

            continue

        plot_time_series(
            df=df,
            ds=ds,
            variable_name=variable_name,
            output_dir=plot_dir
        )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    ds.close()

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )

    print()
    print("Main files:")

    print(
        "  dataset_overview.txt"
    )

    print(
        "  variable_inventory.csv"
    )

    print(
        "  timeseries_all.csv"
    )

    print(
        "  timeseries_preview.csv"
    )

    print(
        "  timeseries_statistics.csv"
    )

    print(
        "  timeseries_plots/"
    )


if __name__ == "__main__":
    main()