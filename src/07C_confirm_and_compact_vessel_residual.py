# -*- coding: utf-8 -*-
"""
07C_confirm_and_compact_vessel_residual.py

Multi-seed confirmation + compact vessel-only residual decoder.

Scientific objective
--------------------
07B showed that vessel-only residual correction captured essentially all of the
Full-Residual apparent-wind improvement, while Wind-Only correction contributed
almost nothing. 07C performs two controlled tests:

Stage A -- multi-seed confirmation
    Re-train and compare, with matched seeds:
        Full-Residual
        Vessel-Only-Residual
        Physics-Full-Residual
        Physics-Vessel-Only-Residual

Stage B -- compact vessel decoder
    Replace the masked 5x6 residual heads with a true 5x2 vessel-only decoder:
        Compact-Vessel-Residual
        Physics-Compact-Vessel-Residual

The compact model still receives the full historical sequence and the full
frozen Ridge forecast as context, but it outputs residual corrections only for:

        [VESSEL_EAST_MPS, VESSEL_NORTH_MPS] x 5 horizons

Wind and heading remain exactly equal to the frozen Ridge prediction.

Expected parameter reduction
----------------------------
Masked 07B model:
    residual head : 64 -> 30
    gate head     : 64 -> 30

Compact model:
    residual head : 64 -> 10
    gate head     : 64 -> 10

The GRU encoder and fusion layer remain unchanged so the experiment isolates
the benefit of removing unused output heads.

Physics-compact loss
--------------------
For the compact vessel-only model, the only trainable physics term from the
frozen 06B solution is the Earth-frame apparent-wind loss:

    L = L_direct
        + lambda_E * L_AW,E
        + lambda_R * L_res

because:
    - lambda_B was selected as 0 by 06B;
    - heading is frozen to Ridge in the compact model, so heading losses do
      not provide gradients to the vessel residual branch.

The script therefore records both:
    original 06B weights
and:
    effective compact physics weights

Multi-seed protocol
-------------------
Default:
    5 matched seeds

Seed schedule:
    base_seed + seed_stride * seed_index

Default:
    base_seed = frozen 05C-v2 GRU final seed (expected 500043)
    seed_stride = 1009

For each seed all six neural variants use:
    - identical frozen Ridge predictions;
    - identical TRAIN / VALIDATION data;
    - identical GRU/fusion architecture where applicable;
    - matched random seed;
    - the same early-stopping metric:
          mean validation apparent-wind vector RMSE.

The already-inspected SD1090 test period is NOT evaluated by default.

Use:
    --evaluate-development-test

only for an explicit development diagnostic. It is labelled
"development_test".

Primary outputs
---------------
seed_run_metrics.csv
seed_summary.csv
seed_per_horizon_metrics.csv
seed_per_horizon_summary.csv
paired_seed_effects.csv
compact_parameter_efficiency.csv
compact_manifest.json
compact_report.txt

models/
histories/
figures/

Dependencies
------------
Place this script in the same src directory as:
    07B_structured_residual_ablation.py
    07A_physics_guided_residual_forecasting.py
    06_run_physics_constrained_joint_gru.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0"
SHARED_07B = "07B_structured_residual_ablation.py"

CONFIRM_VARIANTS = [
    "Full-Residual",
    "Vessel-Only-Residual",
    "Physics-Full-Residual",
    "Physics-Vessel-Only-Residual",
]

COMPACT_VARIANTS = [
    "Compact-Vessel-Residual",
    "Physics-Compact-Vessel-Residual",
]

ALL_VARIANTS = CONFIRM_VARIANTS + COMPACT_VARIANTS


def load_shared_07b():
    path = Path(__file__).resolve().parent / SHARED_07B

    if not path.exists():
        raise FileNotFoundError(
            f"Required shared script not found: {path}\n"
            f"Place {SHARED_07B} in the same src directory as 07C."
        )

    spec = importlib.util.spec_from_file_location(
        "shared_07b_for_07c",
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def save_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def build_compact_model_class(torch, nn):
    class CompactVesselResidualGRU(nn.Module):
        """
        True compact vessel-only residual decoder.

        Input:
            X       [B, L, F]
            ridge_z [B, H, 6]

        Output residual channels:
            only target indices 2:4 (vessel East/North)

        Full prediction:
            wind    = frozen Ridge
            vessel  = Ridge + gated residual
            heading = frozen Ridge
        """

        def __init__(
            self,
            *,
            n_features: int,
            n_horizons: int,
            n_targets: int,
            recurrent_units: int,
            recurrent_layers: int,
            dense_units: int,
            dropout: float,
            gate_bias: float,
            residual_init_std: float,
        ):
            super().__init__()

            if n_targets != 6:
                raise ValueError(
                    "Compact model expects the frozen six-target schema."
                )

            self.n_horizons = int(n_horizons)
            self.n_targets = int(n_targets)
            self.n_vessel_targets = 2

            rnn_dropout = (
                float(dropout)
                if recurrent_layers > 1
                else 0.0
            )

            self.encoder = nn.GRU(
                input_size=int(n_features),
                hidden_size=int(recurrent_units),
                num_layers=int(recurrent_layers),
                batch_first=True,
                dropout=rnn_dropout,
            )

            ridge_dim = (
                int(n_horizons)
                * int(n_targets)
            )

            self.fusion = nn.Linear(
                int(recurrent_units)
                + ridge_dim,
                int(dense_units),
            )

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(
                float(dropout)
            )

            compact_dim = (
                int(n_horizons)
                * self.n_vessel_targets
            )

            self.delta_head = nn.Linear(
                int(dense_units),
                compact_dim,
            )

            self.gate_head = nn.Linear(
                int(dense_units),
                compact_dim,
            )

            nn.init.normal_(
                self.delta_head.weight,
                mean=0.0,
                std=float(residual_init_std),
            )
            nn.init.zeros_(
                self.delta_head.bias
            )
            nn.init.xavier_uniform_(
                self.gate_head.weight
            )
            nn.init.constant_(
                self.gate_head.bias,
                float(gate_bias),
            )

        def forward(
            self,
            X,
            ridge_z,
        ):
            seq, _ = self.encoder(X)
            h = seq[:, -1, :]

            ridge_flat = ridge_z.reshape(
                ridge_z.shape[0],
                -1,
            )

            z = torch.cat(
                [h, ridge_flat],
                dim=1,
            )

            z = self.dropout(
                self.relu(
                    self.fusion(z)
                )
            )

            delta_v = (
                self.delta_head(z)
                .reshape(
                    -1,
                    self.n_horizons,
                    self.n_vessel_targets,
                )
            )

            gate_v = (
                torch.sigmoid(
                    self.gate_head(z)
                )
                .reshape(
                    -1,
                    self.n_horizons,
                    self.n_vessel_targets,
                )
            )

            correction_v = (
                gate_v
                * delta_v
            )

            zeros_wind = torch.zeros_like(
                ridge_z[..., 0:2]
            )

            zeros_heading = torch.zeros_like(
                ridge_z[..., 4:6]
            )

            # Full-shape tensors preserve the shared 07B evaluation API.
            delta_full = torch.cat(
                [
                    zeros_wind,
                    delta_v,
                    zeros_heading,
                ],
                dim=-1,
            )

            gate_full = torch.cat(
                [
                    zeros_wind,
                    gate_v,
                    zeros_heading,
                ],
                dim=-1,
            )

            correction_full = torch.cat(
                [
                    zeros_wind,
                    correction_v,
                    zeros_heading,
                ],
                dim=-1,
            )

            combined_z = (
                ridge_z
                + correction_full
            )

            raw_correction_full = correction_full

            return (
                combined_z,
                delta_full,
                gate_full,
                correction_full,
                raw_correction_full,
            )

    return CompactVesselResidualGRU


def summarize_one_model(
    core06,
    metrics_df: pd.DataFrame,
) -> dict:
    row = core06.summarize_metrics(
        metrics_df
    ).iloc[0]

    return {
        "AW_RMSE_mps": float(
            row[
                "mean_apparent_vector_RMSE_mps"
            ]
        ),
        "AWS_RMSE_mps": float(
            row[
                "mean_AWS_RMSE_mps"
            ]
        ),
        "AWA_MAE_deg": float(
            row[
                "mean_AWA_MAE_deg"
            ]
        ),
        "wind_RMSE_mps": float(
            row[
                "mean_wind_vector_RMSE_mps"
            ]
        ),
        "vessel_RMSE_mps": float(
            row[
                "mean_vessel_vector_RMSE_mps"
            ]
        ),
        "HDG_MAE_deg": float(
            row[
                "mean_HDG_MAE_deg"
            ]
        ),
        "SOG_RMSE_mps": float(
            row[
                "mean_SOG_RMSE_mps"
            ]
        ),
        "COG_MAE_deg": float(
            row[
                "mean_COG_MAE_deg"
            ]
        ),
    }


def aggregate_seed_summary(
    run_df: pd.DataFrame,
) -> pd.DataFrame:
    metric_cols = [
        "AW_RMSE_mps",
        "AWS_RMSE_mps",
        "AWA_MAE_deg",
        "wind_RMSE_mps",
        "vessel_RMSE_mps",
        "HDG_MAE_deg",
        "SOG_RMSE_mps",
        "COG_MAE_deg",
    ]

    rows = []

    for (
        split_name,
        model_name,
    ), grp in run_df.groupby(
        [
            "split",
            "model",
        ],
        sort=False,
    ):
        row = {
            "split": split_name,
            "model": model_name,
            "n_seeds": int(
                len(grp)
            ),
            "parameters": int(
                grp[
                    "parameters"
                ].iloc[
                    0
                ]
            ),
        }

        for col in metric_cols:
            values = grp[
                col
            ].to_numpy(
                dtype=float
            )

            row[
                f"mean_{col}"
            ] = float(
                np.mean(
                    values
                )
            )

            row[
                f"std_{col}"
            ] = float(
                np.std(
                    values,
                    ddof=1,
                )
                if len(
                    values
                ) > 1
                else 0.0
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def aggregate_horizon_summary(
    horizon_df: pd.DataFrame,
) -> pd.DataFrame:
    metric_cols = [
        "wind_vector_RMSE_mps",
        "vessel_vector_RMSE_mps",
        "apparent_vector_RMSE_mps",
        "AWS_RMSE_mps",
        "AWA_MAE_deg",
        "HDG_MAE_deg",
    ]

    rows = []

    for (
        split_name,
        model_name,
        horizon,
    ), grp in horizon_df.groupby(
        [
            "split",
            "model",
            "horizon_min",
        ],
        sort=False,
    ):
        row = {
            "split": split_name,
            "model": model_name,
            "horizon_min": int(
                horizon
            ),
            "n_seeds": int(
                len(grp)
            ),
        }

        for col in metric_cols:
            values = grp[
                col
            ].to_numpy(
                dtype=float
            )

            row[
                f"mean_{col}"
            ] = float(
                np.mean(
                    values
                )
            )

            row[
                f"std_{col}"
            ] = float(
                np.std(
                    values,
                    ddof=1,
                )
                if len(
                    values
                ) > 1
                else 0.0
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def paired_effects(
    run_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Positive AW_reduction means the right-hand/model_B model is better.
    All comparisons are seed-matched.
    """
    comparisons = [
        (
            "VesselOnly_vs_Full",
            "Full-Residual",
            "Vessel-Only-Residual",
        ),
        (
            "PhysicsVesselOnly_vs_PhysicsFull",
            "Physics-Full-Residual",
            "Physics-Vessel-Only-Residual",
        ),
        (
            "PhysicsFull_vs_Full",
            "Full-Residual",
            "Physics-Full-Residual",
        ),
        (
            "PhysicsVesselOnly_vs_VesselOnly",
            "Vessel-Only-Residual",
            "Physics-Vessel-Only-Residual",
        ),
        (
            "Compact_vs_MaskedVesselOnly",
            "Vessel-Only-Residual",
            "Compact-Vessel-Residual",
        ),
        (
            "PhysicsCompact_vs_PhysicsMaskedVesselOnly",
            "Physics-Vessel-Only-Residual",
            "Physics-Compact-Vessel-Residual",
        ),
        (
            "PhysicsCompact_vs_Compact",
            "Compact-Vessel-Residual",
            "Physics-Compact-Vessel-Residual",
        ),
    ]

    val = run_df.loc[
        run_df[
            "split"
        ] == "validation"
    ].copy()

    rows = []

    for label, model_a, model_b in comparisons:
        a = val.loc[
            val[
                "model"
            ] == model_a
        ][
            [
                "seed",
                "AW_RMSE_mps",
            ]
        ].rename(
            columns={
                "AW_RMSE_mps":
                "AW_A"
            }
        )

        b = val.loc[
            val[
                "model"
            ] == model_b
        ][
            [
                "seed",
                "AW_RMSE_mps",
            ]
        ].rename(
            columns={
                "AW_RMSE_mps":
                "AW_B"
            }
        )

        paired = a.merge(
            b,
            on="seed",
            how="inner",
            validate="one_to_one",
        )

        if paired.empty:
            continue

        diff = (
            paired[
                "AW_A"
            ].to_numpy(
                dtype=float
            )
            - paired[
                "AW_B"
            ].to_numpy(
                dtype=float
            )
        )

        pct = (
            100.0
            * diff
            / paired[
                "AW_A"
            ].to_numpy(
                dtype=float
            )
        )

        n = len(
            diff
        )

        mean_diff = float(
            np.mean(
                diff
            )
        )

        std_diff = float(
            np.std(
                diff,
                ddof=1,
            )
            if n > 1
            else 0.0
        )

        ci_low = None
        ci_high = None

        if n > 1:
            try:
                from scipy.stats import t

                crit = float(
                    t.ppf(
                        0.975,
                        df=n - 1,
                    )
                )

                half = (
                    crit
                    * std_diff
                    / math.sqrt(
                        n
                    )
                )

                ci_low = (
                    mean_diff
                    - half
                )

                ci_high = (
                    mean_diff
                    + half
                )

            except Exception:
                pass

        rows.append({
            "comparison": label,
            "model_A_reference": model_a,
            "model_B_candidate": model_b,
            "n_paired_seeds": int(
                n
            ),
            "mean_AW_reduction_B_vs_A_mps": mean_diff,
            "std_AW_reduction_mps": std_diff,
            "mean_AW_reduction_percent": float(
                np.mean(
                    pct
                )
            ),
            "median_AW_reduction_percent": float(
                np.median(
                    pct
                )
            ),
            "B_win_count": int(
                np.sum(
                    diff > 0
                )
            ),
            "tie_count": int(
                np.sum(
                    np.isclose(
                        diff,
                        0.0,
                        atol=1e-12,
                    )
                )
            ),
            "A_win_count": int(
                np.sum(
                    diff < 0
                )
            ),
            "paired_95pct_CI_low_mps": ci_low,
            "paired_95pct_CI_high_mps": ci_high,
        })

    return pd.DataFrame(
        rows
    )


def parameter_efficiency(
    seed_summary: pd.DataFrame,
) -> pd.DataFrame:
    val = seed_summary.loc[
        seed_summary[
            "split"
        ] == "validation"
    ].copy()

    lookup = {
        row[
            "model"
        ]: row
        for _, row in (
            val.iterrows()
        )
    }

    pairs = [
        (
            "Compact_vs_MaskedVesselOnly",
            "Vessel-Only-Residual",
            "Compact-Vessel-Residual",
        ),
        (
            "PhysicsCompact_vs_PhysicsMaskedVesselOnly",
            "Physics-Vessel-Only-Residual",
            "Physics-Compact-Vessel-Residual",
        ),
    ]

    rows = []

    for label, masked, compact in pairs:
        if (
            masked not in lookup
            or compact not in lookup
        ):
            continue

        a = lookup[
            masked
        ]

        b = lookup[
            compact
        ]

        p_a = int(
            a[
                "parameters"
            ]
        )

        p_b = int(
            b[
                "parameters"
            ]
        )

        aw_a = float(
            a[
                "mean_AW_RMSE_mps"
            ]
        )

        aw_b = float(
            b[
                "mean_AW_RMSE_mps"
            ]
        )

        rows.append({
            "comparison": label,
            "masked_model": masked,
            "compact_model": compact,
            "masked_parameters": p_a,
            "compact_parameters": p_b,
            "parameter_reduction": int(
                p_a
                - p_b
            ),
            "parameter_reduction_percent": float(
                100.0
                * (
                    p_a
                    - p_b
                )
                / p_a
            ),
            "masked_mean_validation_AW_RMSE_mps": aw_a,
            "compact_mean_validation_AW_RMSE_mps": aw_b,
            "compact_minus_masked_AW_mps": float(
                aw_b
                - aw_a
            ),
            "compact_relative_AW_change_percent": float(
                100.0
                * (
                    aw_b
                    - aw_a
                )
                / aw_a
            ),
        })

    return pd.DataFrame(
        rows
    )


def make_plots(
    core06,
    *,
    seed_summary,
    horizon_summary,
    paired_df,
    param_df,
    output_dir,
    has_development_test,
):
    import matplotlib.pyplot as plt

    style = core06.load_sci_style(
        Path(
            __file__
        ).resolve().parent
    )

    fig_dir = (
        output_dir
        / "figures"
    )

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    def finish(
        ax,
    ):
        if (
            style is not None
            and hasattr(
                style,
                "clean_axis",
            )
        ):
            style.clean_axis(
                ax,
                grid=True,
            )

        else:
            ax.tick_params(
                which="both",
                direction="in",
                top=True,
                right=True,
            )

            ax.grid(
                True,
                alpha=0.25,
            )

    def save(
        fig,
        name,
    ):
        base = (
            fig_dir
            / name
        )

        if (
            style is not None
            and hasattr(
                style,
                "save_figure",
            )
        ):
            style.save_figure(
                fig,
                base,
            )

        else:
            fig.savefig(
                base.with_suffix(
                    ".png"
                ),
                dpi=600,
                bbox_inches="tight",
            )

            fig.savefig(
                base.with_suffix(
                    ".pdf"
                ),
                bbox_inches="tight",
            )

            plt.close(
                fig
            )

    # Mean +/- std validation AW.
    val = seed_summary.loc[
        seed_summary[
            "split"
        ] == "validation"
    ].copy()

    order = [
        name
        for name in ALL_VARIANTS
        if name in set(
            val[
                "model"
            ]
        )
    ]

    val[
        "_order"
    ] = val[
        "model"
    ].map({
        name: i
        for i, name in enumerate(
            order
        )
    })

    val = val.sort_values(
        "_order"
    )

    fig, ax = plt.subplots(
        figsize=(
            6.5,
            3.7,
        )
    )

    x = np.arange(
        len(
            val
        )
    )

    ax.errorbar(
        x,
        val[
            "mean_AW_RMSE_mps"
        ],
        yerr=val[
            "std_AW_RMSE_mps"
        ],
        marker="o",
        linestyle="none",
        capsize=3,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        val[
            "model"
        ],
        rotation=35,
        ha="right",
    )

    ax.set_ylabel(
        r"Validation AW RMSE (m s$^{-1}$)"
    )

    finish(
        ax
    )

    save(
        fig,
        "01_validation_aw_mean_std",
    )

    # Per-horizon mean +/- std for selected variants.
    hv = horizon_summary.loc[
        horizon_summary[
            "split"
        ] == "validation"
    ]

    fig, ax = plt.subplots(
        figsize=(
            5.7,
            3.5,
        )
    )

    for name in ALL_VARIANTS:
        grp = hv.loc[
            hv[
                "model"
            ] == name
        ].sort_values(
            "horizon_min"
        )

        if grp.empty:
            continue

        ax.errorbar(
            grp[
                "horizon_min"
            ],
            grp[
                "mean_apparent_vector_RMSE_mps"
            ],
            yerr=grp[
                "std_apparent_vector_RMSE_mps"
            ],
            marker="o",
            capsize=2,
            label=name,
        )

    ax.set_xlabel(
        "Forecast horizon (min)"
    )

    ax.set_ylabel(
        r"AW vector RMSE (m s$^{-1}$)"
    )

    ax.set_xticks(
        [1, 2, 3, 5, 10]
    )

    ax.legend(
        loc="best",
        ncol=2,
    )

    finish(
        ax
    )

    save(
        fig,
        "02_validation_aw_per_horizon_mean_std",
    )

    # Paired effect bars.
    if not paired_df.empty:
        fig, ax = plt.subplots(
            figsize=(
                6.4,
                3.6,
            )
        )

        x = np.arange(
            len(
                paired_df
            )
        )

        ax.bar(
            x,
            paired_df[
                "mean_AW_reduction_percent"
            ],
        )

        ax.axhline(
            0.0,
            linewidth=1.0,
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            paired_df[
                "comparison"
            ],
            rotation=35,
            ha="right",
        )

        ax.set_ylabel(
            "Paired AW reduction of B vs A (%)"
        )

        finish(
            ax
        )

        save(
            fig,
            "03_paired_seed_effects",
        )

    # Parameter-performance tradeoff.
    if not val.empty:
        fig, ax = plt.subplots(
            figsize=(
                5.1,
                3.4,
            )
        )

        for _, row in (
            val.iterrows()
        ):
            ax.scatter(
                row[
                    "parameters"
                ],
                row[
                    "mean_AW_RMSE_mps"
                ],
            )

            ax.annotate(
                row[
                    "model"
                ],
                (
                    row[
                        "parameters"
                    ],
                    row[
                        "mean_AW_RMSE_mps"
                    ],
                ),
                xytext=(
                    4,
                    4,
                ),
                textcoords="offset points",
                fontsize=8,
            )

        ax.set_xlabel(
            "Trainable parameters"
        )

        ax.set_ylabel(
            r"Mean validation AW RMSE (m s$^{-1}$)"
        )

        finish(
            ax
        )

        save(
            fig,
            "04_parameter_performance_tradeoff",
        )

    if has_development_test:
        dev = seed_summary.loc[
            seed_summary[
                "split"
            ] == "development_test"
        ]

        if not dev.empty:
            fig, ax = plt.subplots(
                figsize=(
                    6.5,
                    3.7,
                )
            )

            x = np.arange(
                len(
                    dev
                )
            )

            ax.errorbar(
                x,
                dev[
                    "mean_AW_RMSE_mps"
                ],
                yerr=dev[
                    "std_AW_RMSE_mps"
                ],
                marker="o",
                linestyle="none",
                capsize=3,
            )

            ax.set_xticks(
                x
            )

            ax.set_xticklabels(
                dev[
                    "model"
                ],
                rotation=35,
                ha="right",
            )

            ax.set_ylabel(
                r"Development-test AW RMSE (m s$^{-1}$)"
            )

            finish(
                ax
            )

            save(
                fig,
                "05_development_test_aw_mean_std",
            )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        required=True,
    )

    parser.add_argument(
        "--ridge-dir",
        default=None,
    )

    parser.add_argument(
        "--tuning-dir",
        default=None,
    )

    parser.add_argument(
        "--physics-weight-dir",
        default=None,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--confirmation-seeds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed-stride",
        type=int,
        default=1009,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lambda-residual",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--gate-bias",
        type=float,
        default=-1.0,
    )

    parser.add_argument(
        "--residual-init-std",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--ridge-chunk-size",
        type=int,
        default=16384,
    )

    parser.add_argument(
        "--ridge-reproduction-tolerance",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--skip-ridge-reproduction-check",
        action="store_true",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--no-persistent-workers",
        action="store_true",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    parser.add_argument(
        "--no-tf32",
        action="store_true",
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
    )

    parser.add_argument(
        "--direction-min-speed",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--cog-min-speed",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--evaluate-development-test",
        action="store_true",
    )

    parser.add_argument(
        "--skip-validation-flag",
        action="store_true",
    )

    parser.add_argument(
        "--plots",
        action="store_true",
    )

    args = parser.parse_args()

    if args.confirmation_seeds < 2:
        raise ValueError(
            "--confirmation-seeds must be >= 2."
        )

    if args.lambda_residual < 0:
        raise ValueError(
            "--lambda-residual must be >= 0."
        )

    shared07b = load_shared_07b()
    shared07a = shared07b.load_shared_07a()
    core06 = shared07a.load_core()

    dataset_dir = Path(
        args.dataset_dir
    )

    ridge_dir = (
        Path(
            args.ridge_dir
        )
        if args.ridge_dir
        else (
            dataset_dir
            / "joint_baseline_v0_1"
        )
    )

    tuning_dir = (
        Path(
            args.tuning_dir
        )
        if args.tuning_dir
        else (
            dataset_dir
            / "deep_baseline_tuning_pytorch_v0_2"
        )
    )

    physics_weight_dir = (
        Path(
            args.physics_weight_dir
        )
        if args.physics_weight_dir
        else (
            dataset_dir
            / "physics_loss_weight_tuning_v0_1"
        )
    )

    output_dir = (
        Path(
            args.output
        )
        if args.output
        else (
            dataset_dir
            / "confirm_and_compact_vessel_residual_v0_1"
        )
    )

    model_dir = (
        output_dir
        / "models"
    )

    history_dir = (
        output_dir
        / "histories"
    )

    for d in [
        output_dir,
        model_dir,
        history_dir,
    ]:
        d.mkdir(
            parents=True,
            exist_ok=True,
        )

    core06.log(
        "=" * 100
    )

    core06.log(
        "07C - MULTI-SEED CONFIRMATION + COMPACT VESSEL RESIDUAL"
    )

    core06.log(
        "=" * 100
    )

    core06.log(
        f"script_version       : {SCRIPT_VERSION}"
    )

    core06.log(
        f"confirmation_seeds   : {args.confirmation_seeds}"
    )

    core06.log(
        f"seed_stride          : {args.seed_stride}"
    )

    core06.log(
        f"development test     : {args.evaluate_development_test}"
    )

    core06.log("")

    # ------------------------------------------------------------------
    # Stage 1/8: frozen upstream configuration
    # ------------------------------------------------------------------
    core06.log(
        "[STAGE 1/8] Frozen dataset / Ridge / GRU / 06B physics configuration"
    )

    flags = core06.find_validation_flags(
        dataset_dir
    )

    if (
        not flags
        and not args.skip_validation_flag
    ):
        raise RuntimeError(
            "No VALIDATION_PASSED.flag found."
        )

    manifest_path = (
        dataset_dir
        / "dataset_manifest.json"
    )

    scalers_path = (
        dataset_dir
        / "scalers.json"
    )

    core06.require_file(
        manifest_path
    )

    core06.require_file(
        scalers_path
    )

    manifest = core06.load_json(
        manifest_path
    )

    scalers = core06.load_json(
        scalers_path
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

    horizons = tuple(
        int(
            v
        )
        for v in manifest[
            "forecast_horizons_minutes"
        ]
    )

    (
        _,
        feature_mean,
        feature_std,
    ) = core06.parse_scaler(
        scalers,
        "feature_scaler",
    )

    (
        _,
        target_mean,
        target_std,
    ) = core06.parse_scaler(
        scalers,
        "target_scaler",
    )

    (
        gru_config,
        tuning_meta,
    ) = shared07a.load_tuned_gru_config(
        tuning_dir
    )

    (
        physics_weights,
        physics_meta,
    ) = shared07a.load_physics_weights(
        physics_weight_dir
    )

    (
        ridge_model,
        ridge_model_path,
        ridge_manifest,
    ) = shared07a.load_frozen_ridge(
        ridge_dir
    )

    base_seed = (
        int(
            args.base_seed
        )
        if args.base_seed is not None
        else (
            int(
                tuning_meta[
                    "matched_seed"
                ]
            )
            if tuning_meta[
                "matched_seed"
            ] is not None
            else 500043
        )
    )

    seed_schedule = [
        int(
            base_seed
            + i
            * args.seed_stride
        )
        for i in range(
            args.confirmation_seeds
        )
    ]

    compact_physics_weights = {
        "earth": float(
            physics_weights[
                "earth"
            ]
        ),
        "body": 0.0,
        "heading_geo": 0.0,
        "heading_norm": 0.0,
    }

    core06.log(
        f"[SEEDS] {seed_schedule}"
    )

    core06.log(
        "[ORIGINAL 06B PHYSICS] "
        f"E={physics_weights['earth']}, "
        f"B={physics_weights['body']}, "
        f"H={physics_weights['heading_geo']}, "
        f"N={physics_weights['heading_norm']}"
    )

    core06.log(
        "[EFFECTIVE COMPACT PHYSICS] "
        f"E={compact_physics_weights['earth']}, "
        "B=0, H=0, N=0"
    )

    # ------------------------------------------------------------------
    # Stage 2/8: data, device, Ridge predictions
    # ------------------------------------------------------------------
    core06.log(
        "[STAGE 2/8] Loading data / CUDA / frozen Ridge predictions"
    )

    train = core06.load_npz_split(
        dataset_dir,
        "train",
    )

    validation = core06.load_npz_split(
        dataset_dir,
        "validation",
    )

    test = core06.load_npz_split(
        dataset_dir,
        "test",
    )

    (
        torch,
        nn,
        DataLoader,
        TensorDataset,
    ) = core06.import_torch()

    device = (
        torch.device(
            "cpu"
        )
        if (
            args.force_cpu
            or not torch.cuda.is_available()
        )
        else torch.device(
            "cuda:0"
        )
    )

    amp_enabled = bool(
        device.type == "cuda"
        and not args.no_amp
    )

    tf32_enabled = bool(
        device.type == "cuda"
        and not args.no_tf32
    )

    core06.configure_tf32(
        torch,
        tf32_enabled,
    )

    device_info = core06.log_device_info(
        torch,
        device,
        amp_enabled,
        tf32_enabled,
    )

    ridge_train_z = (
        shared07a.ridge_predict_z(
            ridge_model,
            train[
                "X"
            ],
            len(
                horizons
            ),
            len(
                target_names
            ),
            args.ridge_chunk_size,
        )
    )

    ridge_val_z = (
        shared07a.ridge_predict_z(
            ridge_model,
            validation[
                "X"
            ],
            len(
                horizons
            ),
            len(
                target_names
            ),
            args.ridge_chunk_size,
        )
    )

    ridge_val_raw = (
        shared07a.inverse_target(
            ridge_val_z,
            target_mean,
            target_std,
        )
    )

    persistence_val = (
        core06.persistence_joint_prediction(
            validation[
                "X"
            ],
            feature_names,
            feature_mean,
            feature_std,
            len(
                horizons
            ),
        )
    )

    ridge_val_metrics = (
        core06.evaluate_model(
            truth_joint=validation[
                "y_raw"
            ],
            pred_joint=ridge_val_raw,
            truth_apparent_ref=validation[
                "apparent_earth_raw"
            ],
            horizons=horizons,
            split_name="validation",
            model_name="Ridge",
            direction_min_speed=args.direction_min_speed,
            cog_min_speed=args.cog_min_speed,
            persistence_joint=persistence_val,
        )
    )

    ridge_audit = (
        shared07a.ridge_reproduction_audit(
            ridge_dir,
            ridge_val_metrics,
            args.ridge_reproduction_tolerance,
            args.skip_ridge_reproduction_check,
        )
    )

    save_json(
        output_dir
        / "ridge_reproduction_audit.json",
        ridge_audit,
    )

    physics_scales = (
        core06.compute_train_physics_scales(
            train[
                "y_raw"
            ]
        )
    )

    target_mean_t = torch.tensor(
        target_mean,
        dtype=torch.float32,
        device=device,
    )

    target_std_t = torch.tensor(
        target_std,
        dtype=torch.float32,
        device=device,
    )

    physics_scale_t = {
        "ae": torch.tensor(
            physics_scales[
                "apparent_earth_east_std_mps"
            ],
            dtype=torch.float32,
            device=device,
        ),
        "an": torch.tensor(
            physics_scales[
                "apparent_earth_north_std_mps"
            ],
            dtype=torch.float32,
            device=device,
        ),
        "af": torch.tensor(
            physics_scales[
                "apparent_body_forward_std_mps"
            ],
            dtype=torch.float32,
            device=device,
        ),
        "as": torch.tensor(
            physics_scales[
                "apparent_body_starboard_std_mps"
            ],
            dtype=torch.float32,
            device=device,
        ),
    }

    pin_memory = (
        device.type
        == "cuda"
    )

    persistent_workers = (
        args.num_workers > 0
        and not args.no_persistent_workers
    )

    train_ds = shared07a.make_dataset(
        torch,
        TensorDataset,
        train[
            "X"
        ],
        train[
            "y"
        ],
        ridge_train_z,
    )

    val_ds = shared07a.make_dataset(
        torch,
        TensorDataset,
        validation[
            "X"
        ],
        validation[
            "y"
        ],
        ridge_val_z,
    )

    MaskedModelClass = (
        shared07b.build_structured_model_class(
            torch,
            nn,
        )
    )

    CompactModelClass = (
        build_compact_model_class(
            torch,
            nn,
        )
    )

    # ------------------------------------------------------------------
    # Stage 3/8: multi-seed train + validation
    # ------------------------------------------------------------------
    core06.log(
        "[STAGE 3/8] Multi-seed confirmation and compact-model training"
    )

    run_rows = []
    horizon_rows = []

    validation_models_by_seed = {}

    for seed_index, seed in enumerate(
        seed_schedule,
        start=1,
    ):
        core06.log("")
        core06.log(
            "#" * 100
        )

        core06.log(
            f"SEED {seed_index}/{len(seed_schedule)} : {seed}"
        )

        core06.log(
            "#" * 100
        )

        validation_models_by_seed[
            seed
        ] = {}

        for model_name in ALL_VARIANTS:
            is_compact = (
                model_name
                in COMPACT_VARIANTS
            )

            use_physics = (
                "Physics"
                in model_name
            )

            core06.log("")
            core06.log(
                f"[TRAIN] {model_name}"
            )

            core06.set_global_seed(
                torch,
                seed,
                args.deterministic,
            )

            train_loader = shared07a.make_loader(
                DataLoader,
                train_ds,
                args.batch_size,
                True,
                args.num_workers,
                pin_memory,
                persistent_workers,
                args.prefetch_factor,
            )

            val_loader = shared07a.make_loader(
                DataLoader,
                val_ds,
                args.batch_size,
                False,
                args.num_workers,
                pin_memory,
                persistent_workers,
                args.prefetch_factor,
            )

            if is_compact:
                model = CompactModelClass(
                    n_features=len(
                        feature_names
                    ),
                    n_horizons=len(
                        horizons
                    ),
                    n_targets=len(
                        target_names
                    ),
                    recurrent_units=int(
                        gru_config[
                            "recurrent_units"
                        ]
                    ),
                    recurrent_layers=int(
                        gru_config[
                            "recurrent_layers"
                        ]
                    ),
                    dense_units=int(
                        gru_config[
                            "dense_units"
                        ]
                    ),
                    dropout=float(
                        gru_config[
                            "dropout"
                        ]
                    ),
                    gate_bias=float(
                        args.gate_bias
                    ),
                    residual_init_std=float(
                        args.residual_init_std
                    ),
                ).to(
                    device
                )

                effective_weights = (
                    compact_physics_weights
                    if use_physics
                    else {
                        "earth": 0.0,
                        "body": 0.0,
                        "heading_geo": 0.0,
                        "heading_norm": 0.0,
                    }
                )

            else:
                if (
                    "Full"
                    in model_name
                ):
                    active_groups = (
                        "wind",
                        "vessel",
                        "heading",
                    )

                else:
                    active_groups = (
                        "vessel",
                    )

                mask = shared07b.build_mask(
                    active_groups,
                    len(
                        horizons
                    ),
                    len(
                        target_names
                    ),
                )

                model = MaskedModelClass(
                    n_features=len(
                        feature_names
                    ),
                    n_horizons=len(
                        horizons
                    ),
                    n_targets=len(
                        target_names
                    ),
                    recurrent_units=int(
                        gru_config[
                            "recurrent_units"
                        ]
                    ),
                    recurrent_layers=int(
                        gru_config[
                            "recurrent_layers"
                        ]
                    ),
                    dense_units=int(
                        gru_config[
                            "dense_units"
                        ]
                    ),
                    dropout=float(
                        gru_config[
                            "dropout"
                        ]
                    ),
                    gate_bias=float(
                        args.gate_bias
                    ),
                    residual_init_std=float(
                        args.residual_init_std
                    ),
                    correction_mask=mask,
                ).to(
                    device
                )

                effective_weights = (
                    physics_weights
                    if use_physics
                    else {
                        "earth": 0.0,
                        "body": 0.0,
                        "heading_geo": 0.0,
                        "heading_norm": 0.0,
                    }
                )

            n_params = (
                shared07a.count_parameters(
                    model
                )
            )

            seed_model_dir = (
                model_dir
                / f"seed_{seed}"
            )

            seed_history_dir = (
                history_dir
                / f"seed_{seed}"
            )

            seed_model_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            seed_history_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            checkpoint = (
                seed_model_dir
                / (
                    model_name.replace(
                        "-",
                        "_",
                    )
                    + ".pt"
                )
            )

            result = shared07b.train_variant(
                shared07a,
                core06,
                torch,
                model=model,
                variant_name=model_name,
                use_physics=use_physics,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                amp_enabled=amp_enabled,
                learning_rate=float(
                    gru_config[
                        "learning_rate"
                    ]
                ),
                max_epochs=args.epochs,
                patience=args.patience,
                min_delta=args.min_delta,
                gradient_clip_norm=args.gradient_clip_norm,
                target_mean_t=target_mean_t,
                target_std_t=target_std_t,
                target_mean_np=target_mean,
                target_std_np=target_std,
                physics_scale_t=physics_scale_t,
                physics_weights=effective_weights,
                lambda_residual=args.lambda_residual,
                checkpoint_path=checkpoint,
                horizons=horizons,
            )

            result[
                "history"
            ].to_csv(
                seed_history_dir
                / (
                    model_name.replace(
                        "-",
                        "_",
                    )
                    + "_history.csv"
                ),
                index=False,
                encoding="utf-8-sig",
            )

            val_result = (
                result[
                    "validation"
                ]
            )

            val_metrics = (
                core06.evaluate_model(
                    truth_joint=validation[
                        "y_raw"
                    ],
                    pred_joint=val_result[
                        "pred_raw"
                    ],
                    truth_apparent_ref=validation[
                        "apparent_earth_raw"
                    ],
                    horizons=horizons,
                    split_name="validation",
                    model_name=model_name,
                    direction_min_speed=args.direction_min_speed,
                    cog_min_speed=args.cog_min_speed,
                    persistence_joint=persistence_val,
                )
            )

            summary_metrics = (
                summarize_one_model(
                    core06,
                    val_metrics,
                )
            )

            run_rows.append({
                "split": "validation",
                "model": model_name,
                "seed_index": int(
                    seed_index
                ),
                "seed": int(
                    seed
                ),
                "parameters": int(
                    n_params
                ),
                "is_compact": bool(
                    is_compact
                ),
                "use_physics": bool(
                    use_physics
                ),
                "best_epoch": int(
                    result[
                        "best_epoch"
                    ]
                ),
                "epochs_ran": int(
                    result[
                        "epochs_ran"
                    ]
                ),
                "training_seconds": float(
                    result[
                        "training_seconds"
                    ]
                ),
                **summary_metrics,
            })

            h = val_metrics.copy()

            h[
                "seed_index"
            ] = int(
                seed_index
            )

            h[
                "seed"
            ] = int(
                seed
            )

            h[
                "parameters"
            ] = int(
                n_params
            )

            horizon_rows.extend(
                h.to_dict(
                    orient="records"
                )
            )

            validation_models_by_seed[
                seed
            ][
                model_name
            ] = model

            core06.log(
                f"[VALIDATION] {model_name}: "
                f"AW={summary_metrics['AW_RMSE_mps']:.4f} | "
                f"wind={summary_metrics['wind_RMSE_mps']:.4f} | "
                f"vessel={summary_metrics['vessel_RMSE_mps']:.4f} | "
                f"params={n_params:,}"
            )

    # ------------------------------------------------------------------
    # Stage 4/8: optional development test
    # ------------------------------------------------------------------
    if args.evaluate_development_test:
        core06.log("")
        core06.log(
            "[STAGE 4/8] Optional already-inspected development-test evaluation"
        )

        ridge_test_z = (
            shared07a.ridge_predict_z(
                ridge_model,
                test[
                    "X"
                ],
                len(
                    horizons
                ),
                len(
                    target_names
                ),
                args.ridge_chunk_size,
            )
        )

        ridge_test_raw = (
            shared07a.inverse_target(
                ridge_test_z,
                target_mean,
                target_std,
            )
        )

        persistence_test = (
            core06.persistence_joint_prediction(
                test[
                    "X"
                ],
                feature_names,
                feature_mean,
                feature_std,
                len(
                    horizons
                ),
            )
        )

        test_ds = shared07a.make_dataset(
            torch,
            TensorDataset,
            test[
                "X"
            ],
            test[
                "y"
            ],
            ridge_test_z,
        )

        test_loader = shared07a.make_loader(
            DataLoader,
            test_ds,
            args.batch_size,
            False,
            args.num_workers,
            pin_memory,
            persistent_workers,
            args.prefetch_factor,
        )

        for seed_index, seed in enumerate(
            seed_schedule,
            start=1,
        ):
            for model_name in ALL_VARIANTS:
                model = (
                    validation_models_by_seed[
                        seed
                    ][
                        model_name
                    ]
                )

                use_physics = (
                    "Physics"
                    in model_name
                )

                is_compact = (
                    model_name
                    in COMPACT_VARIANTS
                )

                effective_weights = (
                    (
                        compact_physics_weights
                        if is_compact
                        else physics_weights
                    )
                    if use_physics
                    else {
                        "earth": 0.0,
                        "body": 0.0,
                        "heading_geo": 0.0,
                        "heading_norm": 0.0,
                    }
                )

                test_result = (
                    shared07b.evaluate_structured_loader(
                        shared07a,
                        core06,
                        torch,
                        model=model,
                        loader=test_loader,
                        device=device,
                        amp_enabled=amp_enabled,
                        target_mean_t=target_mean_t,
                        target_std_t=target_std_t,
                        target_mean_np=target_mean,
                        target_std_np=target_std,
                        physics_scale_t=physics_scale_t,
                        physics_weights=effective_weights,
                        lambda_residual=args.lambda_residual,
                        use_physics=use_physics,
                    )
                )

                test_metrics = (
                    core06.evaluate_model(
                        truth_joint=test[
                            "y_raw"
                        ],
                        pred_joint=test_result[
                            "pred_raw"
                        ],
                        truth_apparent_ref=test[
                            "apparent_earth_raw"
                        ],
                        horizons=horizons,
                        split_name="development_test",
                        model_name=model_name,
                        direction_min_speed=args.direction_min_speed,
                        cog_min_speed=args.cog_min_speed,
                        persistence_joint=persistence_test,
                    )
                )

                summary_metrics = (
                    summarize_one_model(
                        core06,
                        test_metrics,
                    )
                )

                n_params = (
                    shared07a.count_parameters(
                        model
                    )
                )

                run_rows.append({
                    "split": "development_test",
                    "model": model_name,
                    "seed_index": int(
                        seed_index
                    ),
                    "seed": int(
                        seed
                    ),
                    "parameters": int(
                        n_params
                    ),
                    "is_compact": bool(
                        is_compact
                    ),
                    "use_physics": bool(
                        use_physics
                    ),
                    "best_epoch": np.nan,
                    "epochs_ran": np.nan,
                    "training_seconds": np.nan,
                    **summary_metrics,
                })

                h = test_metrics.copy()

                h[
                    "seed_index"
                ] = int(
                    seed_index
                )

                h[
                    "seed"
                ] = int(
                    seed
                )

                h[
                    "parameters"
                ] = int(
                    n_params
                )

                horizon_rows.extend(
                    h.to_dict(
                        orient="records"
                    )
                )

    else:
        core06.log("")
        core06.log(
            "[STAGE 4/8] SD1090 development test intentionally NOT evaluated"
        )

    # ------------------------------------------------------------------
    # Stage 5/8: aggregate statistics
    # ------------------------------------------------------------------
    core06.log(
        "[STAGE 5/8] Aggregating seed robustness and per-horizon statistics"
    )

    run_df = pd.DataFrame(
        run_rows
    )

    horizon_df = pd.DataFrame(
        horizon_rows
    )

    run_df.to_csv(
        output_dir
        / "seed_run_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_df.to_csv(
        output_dir
        / "seed_per_horizon_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    seed_summary = aggregate_seed_summary(
        run_df
    )

    seed_summary.to_csv(
        output_dir
        / "seed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_summary = (
        aggregate_horizon_summary(
            horizon_df
        )
    )

    horizon_summary.to_csv(
        output_dir
        / "seed_per_horizon_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired_df = paired_effects(
        run_df
    )

    paired_df.to_csv(
        output_dir
        / "paired_seed_effects.csv",
        index=False,
        encoding="utf-8-sig",
    )

    param_df = parameter_efficiency(
        seed_summary
    )

    param_df.to_csv(
        output_dir
        / "compact_parameter_efficiency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Stage 6/8: scientific audit
    # ------------------------------------------------------------------
    core06.log(
        "[STAGE 6/8] Confirmation and compact-model audit"
    )

    val_summary = seed_summary.loc[
        seed_summary[
            "split"
        ] == "validation"
    ].copy()

    best_row = val_summary.sort_values(
        "mean_AW_RMSE_mps"
    ).iloc[
        0
    ]

    masked_params = None
    compact_params = None
    parameter_reduction_pct = None

    if not param_df.empty:
        p = param_df.iloc[
            0
        ]

        masked_params = int(
            p[
                "masked_parameters"
            ]
        )

        compact_params = int(
            p[
                "compact_parameters"
            ]
        )

        parameter_reduction_pct = float(
            p[
                "parameter_reduction_percent"
            ]
        )

    audit = {
        "seed_schedule": seed_schedule,
        "n_confirmation_seeds": int(
            len(
                seed_schedule
            )
        ),
        "best_mean_validation_model": str(
            best_row[
                "model"
            ]
        ),
        "best_mean_validation_AW_RMSE_mps": float(
            best_row[
                "mean_AW_RMSE_mps"
            ]
        ),
        "best_mean_validation_AW_std_mps": float(
            best_row[
                "std_AW_RMSE_mps"
            ]
        ),
        "masked_parameters": masked_params,
        "compact_parameters": compact_params,
        "parameter_reduction_percent": parameter_reduction_pct,
        "development_test_evaluated": bool(
            args.evaluate_development_test
        ),
    }

    # ------------------------------------------------------------------
    # Stage 7/8: manifest/report
    # ------------------------------------------------------------------
    core06.log(
        "[STAGE 7/8] Writing manifest and report"
    )

    manifest_out = {
        "script_version": SCRIPT_VERSION,
        "task_name": manifest.get(
            "task_name"
        ),
        "source_dataset_version": manifest.get(
            "dataset_version"
        ),
        "dataset_dir": str(
            dataset_dir.resolve()
        ),
        "frozen_ridge_model": str(
            ridge_model_path.resolve()
        ),
        "ridge_manifest": ridge_manifest,
        "ridge_reproduction_audit": ridge_audit,
        "fixed_gru_config": gru_config,
        "fixed_06B_physics_weights": physics_weights,
        "effective_compact_physics_weights": compact_physics_weights,
        "physics_weight_metadata": physics_meta,
        "seed_schedule": seed_schedule,
        "protocol": {
            "confirmation_seed_count": int(
                args.confirmation_seeds
            ),
            "same_seed_matched_across_variants": True,
            "train_split": "frozen outer train",
            "checkpoint_split": "frozen outer validation",
            "ridge_frozen": True,
            "compact_decoder_outputs": (
                "[VESSEL_EAST_MPS, VESSEL_NORTH_MPS] x horizons"
            ),
            "full_ridge_forecast_used_as_decoder_context": True,
            "development_test_evaluated": bool(
                args.evaluate_development_test
            ),
            "development_test_used_for_selection": False,
        },
        "loss": {
            "lambda_residual": float(
                args.lambda_residual
            ),
            "original_physics_weights": physics_weights,
            "compact_effective_physics_weights": compact_physics_weights,
        },
        "audit": audit,
        "device_info": device_info,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    }

    save_json(
        output_dir
        / "compact_manifest.json",
        manifest_out,
    )

    with (
        output_dir
        / "compact_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "07C Multi-Seed Confirmation + Compact Vessel Residual Report\n"
        )

        f.write(
            "=" * 100
            + "\n\n"
        )

        f.write(
            "Seed schedule\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )

        f.write(
            str(
                seed_schedule
            )
        )

        f.write(
            "\n\nValidation seed summary\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )

        f.write(
            val_summary.to_string(
                index=False
            )
        )

        f.write(
            "\n\nPaired seed effects\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )

        f.write(
            paired_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nCompact parameter efficiency\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )

        f.write(
            param_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nAudit\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )

        f.write(
            json.dumps(
                audit,
                ensure_ascii=False,
                indent=2,
            )
        )

        f.write(
            "\n"
        )

    # ------------------------------------------------------------------
    # Stage 8/8: plots and terminal summary
    # ------------------------------------------------------------------
    core06.log(
        "[STAGE 8/8] Finalization"
    )

    if args.plots:
        make_plots(
            core06,
            seed_summary=seed_summary,
            horizon_summary=horizon_summary,
            paired_df=paired_df,
            param_df=param_df,
            output_dir=output_dir,
            has_development_test=bool(
                args.evaluate_development_test
            ),
        )

        core06.log(
            "[OK] Figures complete"
        )

    core06.log("")
    core06.log(
        "=" * 100
    )

    core06.log(
        "07C MULTI-SEED + COMPACT RESULTS"
    )

    core06.log(
        "=" * 100
    )

    ordered = val_summary.sort_values(
        "mean_AW_RMSE_mps"
    )

    for _, row in (
        ordered.iterrows()
    ):
        core06.log(
            f"{row['model']}:"
        )

        core06.log(
            f"  parameters       = "
            f"{int(row['parameters']):,}"
        )

        core06.log(
            f"  validation AW    = "
            f"{row['mean_AW_RMSE_mps']:.4f} "
            f"+/- {row['std_AW_RMSE_mps']:.4f} m/s"
        )

        core06.log(
            f"  vessel RMSE      = "
            f"{row['mean_vessel_RMSE_mps']:.4f} "
            f"+/- {row['std_vessel_RMSE_mps']:.4f} m/s"
        )

        core06.log(
            f"  AWS RMSE         = "
            f"{row['mean_AWS_RMSE_mps']:.4f} "
            f"+/- {row['std_AWS_RMSE_mps']:.4f} m/s"
        )

    core06.log("")

    for _, row in (
        paired_df.iterrows()
    ):
        ci_text = ""

        if pd.notna(
            row[
                "paired_95pct_CI_low_mps"
            ]
        ):
            ci_text = (
                f" | 95% CI="
                f"[{row['paired_95pct_CI_low_mps']:.5f}, "
                f"{row['paired_95pct_CI_high_mps']:.5f}]"
            )

        core06.log(
            f"[PAIRED] {row['comparison']}: "
            f"B-A AW reduction="
            f"{row['mean_AW_reduction_B_vs_A_mps']:.5f} m/s "
            f"({row['mean_AW_reduction_percent']:.3f}%) | "
            f"wins={int(row['B_win_count'])}/"
            f"{int(row['n_paired_seeds'])}"
            f"{ci_text}"
        )

    if not param_df.empty:
        for _, row in (
            param_df.iterrows()
        ):
            core06.log(
                f"[COMPACT] {row['comparison']}: "
                f"params {int(row['masked_parameters']):,} -> "
                f"{int(row['compact_parameters']):,} "
                f"({row['parameter_reduction_percent']:.2f}% fewer) | "
                f"AW change="
                f"{row['compact_relative_AW_change_percent']:.3f}%"
            )

    core06.log(
        f"[BEST MEAN VALIDATION] "
        f"{audit['best_mean_validation_model']} | "
        f"AW={audit['best_mean_validation_AW_RMSE_mps']:.4f} "
        f"+/- {audit['best_mean_validation_AW_std_mps']:.4f} m/s"
    )

    if not args.evaluate_development_test:
        core06.log(
            "[TEST POLICY] SD1090 test not evaluated."
        )

    core06.log(
        f"[DEVICE] {device}"
    )

    if device.type == "cuda":
        core06.log(
            f"[GPU] "
            f"{torch.cuda.get_device_name(device)}"
        )

    core06.log(
        f"[DONE] 07C outputs: "
        f"{output_dir}"
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

        print(
            "\nPlease send the complete traceback and terminal output.",
            flush=True,
        )

        sys.exit(
            1
        )
