# -*- coding: utf-8 -*-
"""
report_07BC_ablation_for_paper.py

Reporting-only helper for Section 5.4.
NO TRAINING. NO MODEL SELECTION.

It searches the user-specified 07B and 07C output directories for the frozen
CSV summaries and prints manuscript-ready tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir07b", type=Path, required=True)
    parser.add_argument("--dir07c", type=Path, required=True)
    args = parser.parse_args()

    seed_summary = read_required(args.dir07c / "seed_summary.csv")
    paired = read_required(args.dir07c / "paired_seed_effects.csv")
    params = read_required(args.dir07c / "compact_parameter_efficiency.csv")

    print("=" * 120)
    print("07C VALIDATION SEED SUMMARY")
    print("=" * 120)
    if "split" in seed_summary.columns:
        x = seed_summary.loc[
            seed_summary["split"].astype(str).str.lower() == "validation"
        ].copy()
    else:
        x = seed_summary.copy()
    print(x.to_string(index=False))

    print("\n" + "=" * 120)
    print("07C PAIRED SEED EFFECTS")
    print("=" * 120)
    print(paired.to_string(index=False))

    print("\n" + "=" * 120)
    print("07C PARAMETER EFFICIENCY")
    print("=" * 120)
    print(params.to_string(index=False))

    p07b = args.dir07b / "structured_ablation_metrics_summary.csv"
    if p07b.exists():
        df07b = pd.read_csv(p07b)
        print("\n" + "=" * 120)
        print("07B STRUCTURED ABLATION SUMMARY")
        print("=" * 120)
        print(df07b.to_string(index=False))

    effects07b = args.dir07b / "structured_ablation_effects.csv"
    if effects07b.exists():
        df = pd.read_csv(effects07b)
        print("\n" + "=" * 120)
        print("07B STRUCTURED ABLATION EFFECTS")
        print("=" * 120)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
