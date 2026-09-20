# -*- coding: utf-8 -*-
"""
report_07BC_ablation_for_paper_v2.py

Reporting-only helper for manuscript Section 5.4.
NO TRAINING. NO MODEL SELECTION.

Main improvement over v1
------------------------
You do NOT need to manually type the 07B/07C output directories.

By default the script recursively searches:
    D:\project\WindPredict_SaildroneData\data\forecasting

for the frozen result files:

07C:
    seed_summary.csv
    paired_seed_effects.csv
    compact_parameter_efficiency.csv
    seed_per_horizon_summary.csv

07B:
    structured_ablation_metrics_summary.csv
    structured_ablation_effects.csv

If one valid 07B and one valid 07C directory are found, they are used
automatically.

If multiple candidates are found, the script prints all candidates and selects
the directory containing the largest number of expected files, preferring path
names containing "07B" or "07C".

You can still explicitly override the directories with --dir07b and --dir07c.

Run:
    conda activate WindPredict
    python "D:\project\WindPredict_SaildroneData\src\report_07BC_ablation_for_paper_v2.py"

Optional:
    python "...report_07BC_ablation_for_paper_v2.py" --project-root "D:\project\WindPredict_SaildroneData"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


DEFAULT_PROJECT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")

FILES_07C = [
    "seed_summary.csv",
    "paired_seed_effects.csv",
    "compact_parameter_efficiency.csv",
    "seed_per_horizon_summary.csv",
]

FILES_07B = [
    "structured_ablation_metrics_summary.csv",
    "structured_ablation_effects.csv",
]


def count_expected(directory: Path, expected: list[str]) -> int:
    return sum((directory / name).exists() for name in expected)


def collect_candidate_dirs(search_root: Path, expected: list[str]) -> list[Path]:
    dirs = set()
    for name in expected:
        for path in search_root.rglob(name):
            if path.is_file():
                dirs.add(path.parent)
    return sorted(dirs)


def rank_candidate(directory: Path, expected: list[str], stage_token: str):
    score_files = count_expected(directory, expected)
    score_stage = 1 if stage_token.lower() in str(directory).lower() else 0
    # More complete first; then stage-token preference; then shorter path.
    return (score_files, score_stage, -len(str(directory)))


def choose_directory(
    *,
    explicit: Path | None,
    search_root: Path,
    expected: list[str],
    stage_token: str,
    required_anchor: str,
) -> Path:
    if explicit is not None:
        directory = explicit
        if not directory.exists():
            raise FileNotFoundError(f"Explicit {stage_token} directory does not exist: {directory}")
        if not (directory / required_anchor).exists():
            raise FileNotFoundError(
                f"Explicit {stage_token} directory does not contain "
                f"{required_anchor}: {directory}"
            )
        return directory

    candidates = collect_candidate_dirs(search_root, expected)

    if not candidates:
        raise FileNotFoundError(
            f"No {stage_token} candidate directory was found under:\n"
            f"  {search_root}\n\n"
            f"Searched for:\n  " + "\n  ".join(expected)
        )

    candidates = sorted(
        candidates,
        key=lambda p: rank_candidate(p, expected, stage_token),
        reverse=True,
    )

    print(f"\n[{stage_token}] candidate directories found:")
    for p in candidates:
        n = count_expected(p, expected)
        print(f"  files={n}/{len(expected)} | {p}")

    chosen = candidates[0]

    if not (chosen / required_anchor).exists():
        raise FileNotFoundError(
            f"Best {stage_token} candidate lacks required file "
            f"{required_anchor}: {chosen}"
        )

    print(f"[{stage_token}] selected automatically:")
    print(f"  {chosen}")

    return chosen


def read_csv_if_exists(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def print_df(title: str, df: pd.DataFrame | None):
    print("\n" + "=" * 130)
    print(title)
    print("=" * 130)

    if df is None:
        print("[NOT FOUND]")
        return

    if df.empty:
        print("[EMPTY CSV]")
        return

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 260,
        "display.max_colwidth", 80,
    ):
        print(df.to_string(index=False))


def filter_validation(df: pd.DataFrame) -> pd.DataFrame:
    if "split" in df.columns:
        mask = df["split"].astype(str).str.lower().eq("validation")
        if mask.any():
            return df.loc[mask].copy()

    if "dataset" in df.columns:
        # Do not filter unless clearly labelled validation.
        vals = df["dataset"].astype(str).str.lower()
        mask = vals.str.contains("validation", na=False)
        if mask.any():
            return df.loc[mask].copy()

    return df.copy()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )

    parser.add_argument(
        "--search-root",
        type=Path,
        default=None,
        help="Optional search root. Default: <project-root>/data/forecasting",
    )

    parser.add_argument(
        "--dir07b",
        type=Path,
        default=None,
        help="Optional explicit 07B output directory.",
    )

    parser.add_argument(
        "--dir07c",
        type=Path,
        default=None,
        help="Optional explicit 07C output directory.",
    )

    args = parser.parse_args()

    search_root = (
        args.search_root
        if args.search_root is not None
        else args.project_root / "data" / "forecasting"
    )

    if not search_root.exists():
        raise FileNotFoundError(
            f"Search root does not exist:\n  {search_root}"
        )

    print("=" * 130)
    print("07B / 07C ABLATION REPORT — AUTO DISCOVERY")
    print("=" * 130)
    print(f"project root : {args.project_root}")
    print(f"search root  : {search_root}")
    print("[POLICY] Reporting only. No training, tuning, or checkpoint changes.")

    dir07c = choose_directory(
        explicit=args.dir07c,
        search_root=search_root,
        expected=FILES_07C,
        stage_token="07C",
        required_anchor="seed_summary.csv",
    )

    dir07b = choose_directory(
        explicit=args.dir07b,
        search_root=search_root,
        expected=FILES_07B,
        stage_token="07B",
        required_anchor="structured_ablation_metrics_summary.csv",
    )

    print("\n" + "=" * 130)
    print("SELECTED DIRECTORIES")
    print("=" * 130)
    print(f"07B: {dir07b}")
    print(f"07C: {dir07c}")

    seed_summary = read_csv_if_exists(dir07c / "seed_summary.csv")
    paired = read_csv_if_exists(dir07c / "paired_seed_effects.csv")
    params = read_csv_if_exists(dir07c / "compact_parameter_efficiency.csv")
    per_h = read_csv_if_exists(dir07c / "seed_per_horizon_summary.csv")

    ablation07b = read_csv_if_exists(
        dir07b / "structured_ablation_metrics_summary.csv"
    )
    effects07b = read_csv_if_exists(
        dir07b / "structured_ablation_effects.csv"
    )

    if seed_summary is not None:
        seed_summary_val = filter_validation(seed_summary)
    else:
        seed_summary_val = None

    print_df("07C VALIDATION SEED SUMMARY", seed_summary_val)
    print_df("07C PAIRED SEED EFFECTS", paired)
    print_df("07C PARAMETER EFFICIENCY", params)
    print_df("07C SEED / PER-HORIZON SUMMARY", per_h)
    print_df("07B STRUCTURED ABLATION SUMMARY", ablation07b)
    print_df("07B STRUCTURED ABLATION EFFECTS", effects07b)

    # Save one consolidated Excel workbook for manuscript drafting.
    out_path = args.project_root / "data" / "forecasting" / "07BC_ablation_paper_report.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        if seed_summary_val is not None:
            seed_summary_val.to_excel(
                writer, sheet_name="07C_seed_summary", index=False
            )
        if paired is not None:
            paired.to_excel(
                writer, sheet_name="07C_paired_effects", index=False
            )
        if params is not None:
            params.to_excel(
                writer, sheet_name="07C_parameter_eff", index=False
            )
        if per_h is not None:
            per_h.to_excel(
                writer, sheet_name="07C_per_horizon", index=False
            )
        if ablation07b is not None:
            ablation07b.to_excel(
                writer, sheet_name="07B_ablation_summary", index=False
            )
        if effects07b is not None:
            effects07b.to_excel(
                writer, sheet_name="07B_ablation_effects", index=False
            )

    print("\n" + "=" * 130)
    print("DONE")
    print("=" * 130)
    print(f"Consolidated workbook:")
    print(f"  {out_path}")
    print("\nPlease send me the terminal output from:")
    print("  07C VALIDATION SEED SUMMARY")
    print("  07C PAIRED SEED EFFECTS")
    print("  07C PARAMETER EFFICIENCY")
    print("  07B STRUCTURED ABLATION SUMMARY")


if __name__ == "__main__":
    main()
