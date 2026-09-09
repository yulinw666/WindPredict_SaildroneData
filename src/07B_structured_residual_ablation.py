# -*- coding: utf-8 -*-
"""
07B_structured_residual_ablation.py

Structured residual ablation for the frozen Saildrone wind-vessel forecasting
pipeline.

Purpose
-------
07A showed that the learned residual gate strongly preferred vessel-motion
correction while leaving the Ridge wind forecast almost unchanged.

07B tests that observation explicitly with hard physical correction masks.

All neural variants use:
    - the same frozen 04B Ridge baseline;
    - the same 05C-v2 GRU backbone;
    - the same decoder architecture;
    - the same nominal trainable parameter count;
    - the same random seed;
    - the same TRAIN / VALIDATION protocol.

Only the ACTIVE residual output channels and optional 06B physics loss differ.

Supported variants
------------------
Full-Residual
    Correct all six targets:
    [U, V, vessel_E, vessel_N, HDG_sin, HDG_cos]

Wind-Only-Residual
    Correct only [U, V].

Vessel-Only-Residual
    Correct only [vessel_E, vessel_N].

Heading-Only-Residual
    Correct only [HDG_sin, HDG_cos].

Vessel-Heading-Residual
    Correct vessel motion + heading, while Ridge wind remains frozen.

Physics-Full-Residual
    Same as Full-Residual + fixed 06B physics loss.

Physics-Vessel-Only-Residual
    Same as Vessel-Only-Residual + fixed 06B physics loss.

Physics-Vessel-Heading-Residual
    Same as Vessel-Heading-Residual + fixed 06B physics loss.

Default ablation
----------------
The default run trains six variants:

    Full-Residual
    Wind-Only-Residual
    Vessel-Only-Residual
    Vessel-Heading-Residual
    Physics-Full-Residual
    Physics-Vessel-Only-Residual

This directly tests the key hypothesis:

    Most nonlinear residual benefit comes from vessel-motion correction,
    while the Earth-relative wind forecast can remain primarily linear/Ridge.

Residual formulation
--------------------
For target mask M in {0,1}^{H x T}:

    correction_z = M * gate * delta_z
    y_hat_z      = y_Ridge,z + correction_z

The mask is fixed and non-trainable.

Residual penalty:

    L_res = mean(correction_z^2)

Physics variants use the frozen 06B weights:

    L = L_direct
        + lambda_E L_AW,E
        + lambda_B L_AW,B
        + lambda_H L_heading
        + lambda_N L_norm
        + lambda_R L_res

Non-physics variants use:

    L = L_direct + lambda_R L_res

Test policy
-----------
The already-inspected SD1090 test split is NOT evaluated by default.

Use:
    --evaluate-development-test

only for an explicit development diagnostic. It will be labelled
"development_test".

Dependencies
------------
Place this script in the same src directory as:

    07A_physics_guided_residual_forecasting.py
    06_run_physics_constrained_joint_gru.py

Main outputs
------------
structured_ablation_training_summary.csv
structured_ablation_metrics_per_horizon.csv
structured_ablation_metrics_summary.csv
structured_ablation_effects.csv
structured_ablation_gate_statistics.csv
structured_ablation_correction_statistics.csv
structured_ablation_manifest.json
structured_ablation_report.txt

models/
histories/
predictions/
figures/
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0"
SHARED_07A = "07A_physics_guided_residual_forecasting.py"

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


VARIANT_SPECS = {
    "Full-Residual": {
        "active_groups": ("wind", "vessel", "heading"),
        "use_physics": False,
    },
    "Wind-Only-Residual": {
        "active_groups": ("wind",),
        "use_physics": False,
    },
    "Vessel-Only-Residual": {
        "active_groups": ("vessel",),
        "use_physics": False,
    },
    "Heading-Only-Residual": {
        "active_groups": ("heading",),
        "use_physics": False,
    },
    "Vessel-Heading-Residual": {
        "active_groups": ("vessel", "heading"),
        "use_physics": False,
    },
    "Physics-Full-Residual": {
        "active_groups": ("wind", "vessel", "heading"),
        "use_physics": True,
    },
    "Physics-Vessel-Only-Residual": {
        "active_groups": ("vessel",),
        "use_physics": True,
    },
    "Physics-Vessel-Heading-Residual": {
        "active_groups": ("vessel", "heading"),
        "use_physics": True,
    },
}


DEFAULT_VARIANTS = ",".join([
    "Full-Residual",
    "Wind-Only-Residual",
    "Vessel-Only-Residual",
    "Vessel-Heading-Residual",
    "Physics-Full-Residual",
    "Physics-Vessel-Only-Residual",
])


def load_shared_07a():
    path = Path(__file__).resolve().parent / SHARED_07A

    if not path.exists():
        raise FileNotFoundError(
            f"Required shared script not found: {path}\n"
            f"Place {SHARED_07A} in the same src directory as 07B."
        )

    spec = importlib.util.spec_from_file_location(
        "shared_07a_for_07b",
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to import {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def save_json(path: Path, obj: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def parse_variants(text: str) -> list[str]:
    out = []

    for token in text.split(","):
        name = token.strip()

        if not name:
            continue

        if name not in VARIANT_SPECS:
            raise ValueError(
                f"Unsupported variant: {name}\n"
                f"Allowed: {list(VARIANT_SPECS)}"
            )

        if name not in out:
            out.append(name)

    if not out:
        raise ValueError("No variants selected.")

    return out


def group_target_indices(
    active_groups: tuple[str, ...],
) -> list[int]:
    group_map = {
        "wind": [0, 1],
        "vessel": [2, 3],
        "heading": [4, 5],
    }

    indices = []

    for group in active_groups:
        indices.extend(
            group_map[group]
        )

    return sorted(set(indices))


def build_mask(
    active_groups: tuple[str, ...],
    n_horizons: int,
    n_targets: int,
) -> np.ndarray:
    mask = np.zeros(
        (
            n_horizons,
            n_targets,
        ),
        dtype=np.float32,
    )

    idx = group_target_indices(
        active_groups
    )

    mask[
        :,
        idx,
    ] = 1.0

    return mask


def build_structured_model_class(
    torch,
    nn,
):
    class StructuredGatedResidualGRU(nn.Module):
        """
        Same decoder structure as 07A, with a fixed non-trainable
        correction mask M[horizon, target].
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
            correction_mask: np.ndarray,
        ):
            super().__init__()

            self.n_horizons = int(
                n_horizons
            )

            self.n_targets = int(
                n_targets
            )

            rnn_dropout = (
                float(
                    dropout
                )
                if recurrent_layers > 1
                else 0.0
            )

            self.encoder = nn.GRU(
                input_size=int(
                    n_features
                ),
                hidden_size=int(
                    recurrent_units
                ),
                num_layers=int(
                    recurrent_layers
                ),
                batch_first=True,
                dropout=rnn_dropout,
            )

            output_dim = (
                int(
                    n_horizons
                )
                * int(
                    n_targets
                )
            )

            self.fusion = nn.Linear(
                int(
                    recurrent_units
                )
                + output_dim,
                int(
                    dense_units
                ),
            )

            self.relu = nn.ReLU()

            self.dropout = nn.Dropout(
                float(
                    dropout
                )
            )

            self.delta_head = nn.Linear(
                int(
                    dense_units
                ),
                output_dim,
            )

            self.gate_head = nn.Linear(
                int(
                    dense_units
                ),
                output_dim,
            )

            nn.init.normal_(
                self.delta_head.weight,
                mean=0.0,
                std=float(
                    residual_init_std
                ),
            )

            nn.init.zeros_(
                self.delta_head.bias
            )

            nn.init.xavier_uniform_(
                self.gate_head.weight
            )

            nn.init.constant_(
                self.gate_head.bias,
                float(
                    gate_bias
                ),
            )

            mask = np.asarray(
                correction_mask,
                dtype=np.float32,
            )

            if mask.shape != (
                n_horizons,
                n_targets,
            ):
                raise ValueError(
                    f"Mask shape {mask.shape} != "
                    f"({n_horizons},{n_targets})"
                )

            self.register_buffer(
                "correction_mask",
                torch.from_numpy(
                    mask
                ).reshape(
                    1,
                    n_horizons,
                    n_targets,
                ),
                persistent=True,
            )

        def forward(
            self,
            X,
            ridge_z,
        ):
            seq, _ = self.encoder(
                X
            )

            h = seq[
                :,
                -1,
                :,
            ]

            ridge_flat = ridge_z.reshape(
                ridge_z.shape[
                    0
                ],
                -1,
            )

            z = torch.cat(
                [
                    h,
                    ridge_flat,
                ],
                dim=1,
            )

            z = self.dropout(
                self.relu(
                    self.fusion(
                        z
                    )
                )
            )

            delta_z = (
                self.delta_head(
                    z
                )
                .reshape(
                    -1,
                    self.n_horizons,
                    self.n_targets,
                )
            )

            gate = (
                torch.sigmoid(
                    self.gate_head(
                        z
                    )
                )
                .reshape(
                    -1,
                    self.n_horizons,
                    self.n_targets,
                )
            )

            raw_correction_z = (
                gate
                * delta_z
            )

            correction_z = (
                raw_correction_z
                * self.correction_mask
            )

            combined_z = (
                ridge_z
                + correction_z
            )

            return (
                combined_z,
                delta_z,
                gate,
                correction_z,
                raw_correction_z,
            )

    return StructuredGatedResidualGRU


def compute_losses(
    core06,
    torch,
    *,
    combined_z,
    true_z,
    correction_z,
    target_mean_t,
    target_std_t,
    physics_scale_t,
    physics_weights,
    lambda_residual: float,
    use_physics: bool,
):
    components = (
        core06.compute_loss_components(
            torch=torch,
            pred_z=combined_z,
            true_z=true_z,
            target_mean_t=target_mean_t,
            target_std_t=target_std_t,
            physics_scale_t=physics_scale_t,
        )
    )

    residual_penalty = torch.mean(
        correction_z ** 2
    )

    total = (
        components[
            "direct"
        ]
        + float(
            lambda_residual
        )
        * residual_penalty
    )

    if use_physics:
        total = (
            total
            + float(
                physics_weights[
                    "earth"
                ]
            )
            * components[
                "apparent_earth"
            ]
            + float(
                physics_weights[
                    "body"
                ]
            )
            * components[
                "apparent_body"
            ]
            + float(
                physics_weights[
                    "heading_geo"
                ]
            )
            * components[
                "heading_geo"
            ]
            + float(
                physics_weights[
                    "heading_norm"
                ]
            )
            * components[
                "heading_norm"
            ]
        )

    return (
        total,
        components,
        residual_penalty,
    )


def evaluate_structured_loader(
    shared07a,
    core06,
    torch,
    *,
    model,
    loader,
    device,
    amp_enabled,
    target_mean_t,
    target_std_t,
    target_mean_np,
    target_std_np,
    physics_scale_t,
    physics_weights,
    lambda_residual,
    use_physics,
):
    model.eval()

    sums = {
        "total": 0.0,
        "direct": 0.0,
        "earth": 0.0,
        "body": 0.0,
        "heading_geo": 0.0,
        "heading_norm": 0.0,
        "residual_penalty": 0.0,
    }

    total_n = 0

    pred_chunks = []
    truth_chunks = []
    gate_chunks = []
    correction_chunks = []
    raw_correction_chunks = []

    with torch.no_grad():
        for (
            Xb,
            yb,
            ridge_b,
        ) in loader:
            Xb = Xb.to(
                device,
                non_blocking=True,
            )

            yb = yb.to(
                device,
                non_blocking=True,
            )

            ridge_b = ridge_b.to(
                device,
                non_blocking=True,
            )

            with core06.AutocastContext(
                torch,
                amp_enabled,
                device.type,
            ):
                (
                    combined_z,
                    _,
                    gate,
                    correction_z,
                    raw_correction_z,
                ) = model(
                    Xb,
                    ridge_b,
                )

                (
                    total_loss,
                    components,
                    residual_penalty,
                ) = compute_losses(
                    core06,
                    torch,
                    combined_z=combined_z,
                    true_z=yb,
                    correction_z=correction_z,
                    target_mean_t=target_mean_t,
                    target_std_t=target_std_t,
                    physics_scale_t=physics_scale_t,
                    physics_weights=physics_weights,
                    lambda_residual=lambda_residual,
                    use_physics=use_physics,
                )

            n = int(
                Xb.shape[
                    0
                ]
            )

            total_n += n

            sums[
                "total"
            ] += (
                float(
                    total_loss
                    .detach()
                    .item()
                )
                * n
            )

            sums[
                "direct"
            ] += (
                float(
                    components[
                        "direct"
                    ]
                    .detach()
                    .item()
                )
                * n
            )

            sums[
                "earth"
            ] += (
                float(
                    components[
                        "apparent_earth"
                    ]
                    .detach()
                    .item()
                )
                * n
            )

            sums[
                "body"
            ] += (
                float(
                    components[
                        "apparent_body"
                    ]
                    .detach()
                    .item()
                )
                * n
            )

            sums[
                "heading_geo"
            ] += (
                float(
                    components[
                        "heading_geo"
                    ]
                    .detach()
                    .item()
                )
                * n
            )

            sums[
                "heading_norm"
            ] += (
                float(
                    components[
                        "heading_norm"
                    ]
                    .detach()
                    .item()
                )
                * n
            )

            sums[
                "residual_penalty"
            ] += (
                float(
                    residual_penalty
                    .detach()
                    .item()
                )
                * n
            )

            pred_chunks.append(
                combined_z
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            truth_chunks.append(
                yb
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            gate_chunks.append(
                gate
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            correction_chunks.append(
                correction_z
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            raw_correction_chunks.append(
                raw_correction_z
                .detach()
                .float()
                .cpu()
                .numpy()
            )

    pred_z = np.concatenate(
        pred_chunks,
        axis=0,
    )

    truth_z = np.concatenate(
        truth_chunks,
        axis=0,
    )

    gate = np.concatenate(
        gate_chunks,
        axis=0,
    )

    correction_z = np.concatenate(
        correction_chunks,
        axis=0,
    )

    raw_correction_z = np.concatenate(
        raw_correction_chunks,
        axis=0,
    )

    pred_raw = shared07a.inverse_target(
        pred_z,
        target_mean_np,
        target_std_np,
    )

    truth_raw = shared07a.inverse_target(
        truth_z,
        target_mean_np,
        target_std_np,
    )

    (
        mean_aw,
        per_horizon_aw,
    ) = core06.mean_apparent_vector_rmse(
        truth_raw,
        pred_raw,
    )

    return {
        "losses": {
            key: float(
                value
                / max(
                    total_n,
                    1,
                )
            )
            for key, value in (
                sums.items()
            )
        },
        "mean_aw_rmse": float(
            mean_aw
        ),
        "per_horizon_aw_rmse": np.asarray(
            per_horizon_aw,
            dtype=np.float64,
        ),
        "pred_z": pred_z,
        "pred_raw": pred_raw,
        "gate": gate,
        "correction_z": correction_z,
        "raw_correction_z": raw_correction_z,
    }


def train_variant(
    shared07a,
    core06,
    torch,
    *,
    model,
    variant_name,
    use_physics,
    train_loader,
    val_loader,
    device,
    amp_enabled,
    learning_rate,
    max_epochs,
    patience,
    min_delta,
    gradient_clip_norm,
    target_mean_t,
    target_std_t,
    target_mean_np,
    target_std_np,
    physics_scale_t,
    physics_weights,
    lambda_residual,
    checkpoint_path,
    horizons,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            learning_rate
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(
                2,
                int(
                    patience
                )
                // 3,
            ),
            min_lr=1e-5,
        )
    )

    scaler = core06.make_grad_scaler(
        torch,
        amp_enabled,
    )

    best_aw = float(
        "inf"
    )

    best_epoch = 0
    wait = 0
    history = []

    t0 = time.perf_counter()

    for epoch in range(
        1,
        int(
            max_epochs
        )
        + 1,
    ):
        epoch_t0 = time.perf_counter()

        model.train()

        sums = {
            "total": 0.0,
            "direct": 0.0,
            "earth": 0.0,
            "body": 0.0,
            "heading_geo": 0.0,
            "heading_norm": 0.0,
            "residual_penalty": 0.0,
        }

        total_n = 0

        for (
            Xb,
            yb,
            ridge_b,
        ) in train_loader:
            Xb = Xb.to(
                device,
                non_blocking=True,
            )

            yb = yb.to(
                device,
                non_blocking=True,
            )

            ridge_b = ridge_b.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with core06.AutocastContext(
                torch,
                amp_enabled,
                device.type,
            ):
                (
                    combined_z,
                    _,
                    _,
                    correction_z,
                    _,
                ) = model(
                    Xb,
                    ridge_b,
                )

                (
                    total_loss,
                    components,
                    residual_penalty,
                ) = compute_losses(
                    core06,
                    torch,
                    combined_z=combined_z,
                    true_z=yb,
                    correction_z=correction_z,
                    target_mean_t=target_mean_t,
                    target_std_t=target_std_t,
                    physics_scale_t=physics_scale_t,
                    physics_weights=physics_weights,
                    lambda_residual=lambda_residual,
                    use_physics=use_physics,
                )

            if amp_enabled:
                scaler.scale(
                    total_loss
                ).backward()

                if gradient_clip_norm > 0:
                    scaler.unscale_(
                        optimizer
                    )

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=float(
                            gradient_clip_norm
                        ),
                    )

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:
                total_loss.backward()

                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=float(
                            gradient_clip_norm
                        ),
                    )

                optimizer.step()

            n = int(
                Xb.shape[
                    0
                ]
            )

            total_n += n

            sums[
                "total"
            ] += (
                float(
                    total_loss
                    .detach()
                    .item()
                )
                * n
            )

            sums[
                "direct"
            ] += (
                float(
                    components[
                        "direct"
                    ]
                    .detach()
                    .item()
                )
                * n
            )

            sums[
                "earth"
            ] += (
                float(
                    components[
                        "apparent_earth"
                    ]
                    .detach()
                    .item()
                )
                * n
            )

            sums[
                "body"
            ] += (
                float(
                    components[
                        "apparent_body"
                    ]
                    .detach()
                    .item()
                )
                * n
            )

            sums[
                "heading_geo"
            ] += (
                float(
                    components[
                        "heading_geo"
                    ]
                    .detach()
                    .item()
                )
                * n
            )

            sums[
                "heading_norm"
            ] += (
                float(
                    components[
                        "heading_norm"
                    ]
                    .detach()
                    .item()
                )
                * n
            )

            sums[
                "residual_penalty"
            ] += (
                float(
                    residual_penalty
                    .detach()
                    .item()
                )
                * n
            )

        train_avg = {
            k: float(
                v
                / max(
                    total_n,
                    1,
                )
            )
            for k, v in (
                sums.items()
            )
        }

        val_result = (
            evaluate_structured_loader(
                shared07a,
                core06,
                torch,
                model=model,
                loader=val_loader,
                device=device,
                amp_enabled=amp_enabled,
                target_mean_t=target_mean_t,
                target_std_t=target_std_t,
                target_mean_np=target_mean_np,
                target_std_np=target_std_np,
                physics_scale_t=physics_scale_t,
                physics_weights=physics_weights,
                lambda_residual=lambda_residual,
                use_physics=use_physics,
            )
        )

        val_aw = float(
            val_result[
                "mean_aw_rmse"
            ]
        )

        val_losses = val_result[
            "losses"
        ]

        lr = float(
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        scheduler.step(
            val_losses[
                "total"
            ]
        )

        improved = (
            val_aw
            < (
                best_aw
                - float(
                    min_delta
                )
            )
        )

        if improved:
            best_aw = val_aw
            best_epoch = int(
                epoch
            )
            wait = 0

            torch.save(
                {
                    "variant": variant_name,
                    "epoch": best_epoch,
                    "best_validation_AW_RMSE_mps": best_aw,
                    "state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

            status = "[BEST]"

        else:
            wait += 1
            status = (
                f"wait={wait}/{patience}"
            )

        row = {
            "epoch": int(
                epoch
            ),
            "learning_rate": lr,
            "epoch_seconds": float(
                time.perf_counter()
                - epoch_t0
            ),
            "train_total_loss": train_avg[
                "total"
            ],
            "train_direct_loss": train_avg[
                "direct"
            ],
            "train_apparent_earth_loss": train_avg[
                "earth"
            ],
            "train_apparent_body_loss": train_avg[
                "body"
            ],
            "train_heading_geo_loss": train_avg[
                "heading_geo"
            ],
            "train_heading_norm_loss": train_avg[
                "heading_norm"
            ],
            "train_residual_penalty": train_avg[
                "residual_penalty"
            ],
            "val_total_loss": val_losses[
                "total"
            ],
            "val_direct_loss": val_losses[
                "direct"
            ],
            "val_apparent_earth_loss": val_losses[
                "earth"
            ],
            "val_apparent_body_loss": val_losses[
                "body"
            ],
            "val_heading_geo_loss": val_losses[
                "heading_geo"
            ],
            "val_heading_norm_loss": val_losses[
                "heading_norm"
            ],
            "val_residual_penalty": val_losses[
                "residual_penalty"
            ],
            "val_apparent_vector_RMSE_mps": val_aw,
            "val_mean_active_correction_z_RMS": float(
                np.sqrt(
                    np.mean(
                        val_result[
                            "correction_z"
                        ] ** 2
                    )
                )
            ),
        }

        for j, horizon in enumerate(
            horizons
        ):
            row[
                f"val_AW_RMSE_{horizon}min_mps"
            ] = float(
                val_result[
                    "per_horizon_aw_rmse"
                ][
                    j
                ]
            )

        history.append(
            row
        )

        core06.log(
            f"Epoch {epoch:03d}/{max_epochs} | "
            f"train_total={train_avg['total']:.6f} | "
            f"val_total={val_losses['total']:.6f} | "
            f"val_AW={val_aw:.6f} m/s | "
            f"corr_z={row['val_mean_active_correction_z_RMS']:.3f} | "
            f"lr={lr:.3e} | "
            f"{row['epoch_seconds']:.2f}s | "
            f"{status}"
        )

        if wait >= patience:
            core06.log(
                f"[EARLY STOP] {variant_name}: "
                f"best_epoch={best_epoch}, "
                f"best val AW={best_aw:.6f} m/s"
            )

            break

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        ckpt[
            "state_dict"
        ]
    )

    final_val = (
        evaluate_structured_loader(
            shared07a,
            core06,
            torch,
            model=model,
            loader=val_loader,
            device=device,
            amp_enabled=amp_enabled,
            target_mean_t=target_mean_t,
            target_std_t=target_std_t,
            target_mean_np=target_mean_np,
            target_std_np=target_std_np,
            physics_scale_t=physics_scale_t,
            physics_weights=physics_weights,
            lambda_residual=lambda_residual,
            use_physics=use_physics,
        )
    )

    return {
        "model": model,
        "history": pd.DataFrame(
            history
        ),
        "best_epoch": int(
            best_epoch
        ),
        "best_validation_AW_RMSE_mps": float(
            best_aw
        ),
        "epochs_ran": int(
            len(
                history
            )
        ),
        "training_seconds": float(
            time.perf_counter()
            - t0
        ),
        "validation": final_val,
    }


def active_gate_statistics(
    *,
    model_name,
    split_name,
    gate,
    mask,
    horizons,
    target_names,
):
    rows = []

    group_map = {
        "wind": [0, 1],
        "vessel": [2, 3],
        "heading": [4, 5],
    }

    for j, horizon in enumerate(
        horizons
    ):
        for group, idx in (
            group_map.items()
        ):
            active_idx = [
                t
                for t in idx
                if mask[
                    j,
                    t
                ] > 0.5
            ]

            if not active_idx:
                continue

            values = gate[
                :,
                j,
                active_idx,
            ].astype(
                np.float64
            )

            rows.append({
                "split": split_name,
                "model": model_name,
                "horizon_min": int(
                    horizon
                ),
                "group": group,
                "mean_gate": float(
                    np.mean(
                        values
                    )
                ),
                "std_gate": float(
                    np.std(
                        values
                    )
                ),
                "median_gate": float(
                    np.median(
                        values
                    )
                ),
            })

        active_all = np.where(
            mask[
                j
            ] > 0.5
        )[0]

        if len(
            active_all
        ) > 0:
            values = gate[
                :,
                j,
                active_all,
            ].astype(
                np.float64
            )

            rows.append({
                "split": split_name,
                "model": model_name,
                "horizon_min": int(
                    horizon
                ),
                "group": "all_active",
                "mean_gate": float(
                    np.mean(
                        values
                    )
                ),
                "std_gate": float(
                    np.std(
                        values
                    )
                ),
                "median_gate": float(
                    np.median(
                        values
                    )
                ),
            })

    return pd.DataFrame(
        rows
    )


def compute_effect_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    val = summary.loc[
        summary[
            "split"
        ] == "validation"
    ].copy()

    if val.empty:
        return pd.DataFrame()

    lookup = {
        row[
            "model"
        ]: row
        for _, row in (
            val.iterrows()
        )
    }

    ridge_row = lookup.get(
        "Ridge"
    )

    full_row = lookup.get(
        "Full-Residual"
    )

    rows = []

    if ridge_row is None:
        return pd.DataFrame()

    ridge_aw = float(
        ridge_row[
            "mean_apparent_vector_RMSE_mps"
        ]
    )

    full_gain = None

    if full_row is not None:
        full_aw = float(
            full_row[
                "mean_apparent_vector_RMSE_mps"
            ]
        )

        full_gain = (
            ridge_aw
            - full_aw
        )

    for model_name, row in (
        lookup.items()
    ):
        if model_name == "Ridge":
            continue

        aw = float(
            row[
                "mean_apparent_vector_RMSE_mps"
            ]
        )

        gain = (
            ridge_aw
            - aw
        )

        gain_pct = (
            100.0
            * gain
            / ridge_aw
        )

        fraction_full = None

        if (
            full_gain is not None
            and abs(
                full_gain
            ) > 1e-12
        ):
            fraction_full = (
                gain
                / full_gain
            )

        rows.append({
            "model": model_name,
            "validation_AW_RMSE_mps": aw,
            "AW_improvement_vs_Ridge_mps": gain,
            "AW_improvement_vs_Ridge_percent": gain_pct,
            "fraction_of_Full_Residual_AW_gain": fraction_full,
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
            "HDG_MAE_deg": float(
                row[
                    "mean_HDG_MAE_deg"
                ]
            ),
        })

    return pd.DataFrame(
        rows
    ).sort_values(
        "validation_AW_RMSE_mps"
    ).reset_index(
        drop=True
    )


def make_plots(
    core06,
    *,
    metrics,
    effects,
    correction_stats,
    histories,
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

    order = [
        "Ridge",
        "Full-Residual",
        "Wind-Only-Residual",
        "Vessel-Only-Residual",
        "Heading-Only-Residual",
        "Vessel-Heading-Residual",
        "Physics-Full-Residual",
        "Physics-Vessel-Only-Residual",
        "Physics-Vessel-Heading-Residual",
    ]

    def curve(
        split,
        metric,
        ylabel,
        filename,
    ):
        subset = metrics.loc[
            metrics[
                "split"
            ] == split
        ]

        if subset.empty:
            return

        fig, ax = plt.subplots(
            figsize=(
                5.4,
                3.4,
            )
        )

        for name in order:
            grp = subset.loc[
                subset[
                    "model"
                ] == name
            ].sort_values(
                "horizon_min"
            )

            if grp.empty:
                continue

            ax.plot(
                grp[
                    "horizon_min"
                ],
                grp[
                    metric
                ],
                marker="o",
                label=name,
            )

        ax.set_xlabel(
            "Forecast horizon (min)"
        )

        ax.set_ylabel(
            ylabel
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
            filename,
        )

    curve(
        "validation",
        "apparent_vector_RMSE_mps",
        r"Apparent-wind vector RMSE (m s$^{-1}$)",
        "01_validation_apparent_rmse",
    )

    curve(
        "validation",
        "vessel_vector_RMSE_mps",
        r"Vessel vector RMSE (m s$^{-1}$)",
        "02_validation_vessel_rmse",
    )

    curve(
        "validation",
        "wind_vector_RMSE_mps",
        r"Wind vector RMSE (m s$^{-1}$)",
        "03_validation_wind_rmse",
    )

    if not effects.empty:
        fig, ax = plt.subplots(
            figsize=(
                6.2,
                3.6,
            )
        )

        ordered = effects.sort_values(
            "AW_improvement_vs_Ridge_percent",
            ascending=False,
        )

        x = np.arange(
            len(
                ordered
            )
        )

        ax.bar(
            x,
            ordered[
                "AW_improvement_vs_Ridge_percent"
            ],
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            ordered[
                "model"
            ],
            rotation=35,
            ha="right",
        )

        ax.set_ylabel(
            "Validation AW improvement vs Ridge (%)"
        )

        finish(
            ax
        )

        save(
            fig,
            "04_ablation_effect_vs_ridge",
        )

    # Correction magnitude of wind/vessel branches.
    subset = correction_stats.loc[
        (
            correction_stats[
                "split"
            ] == "validation"
        )
        & (
            correction_stats[
                "target"
            ].isin(
                [
                    "wind_vector",
                    "vessel_vector",
                ]
            )
        )
    ]

    if not subset.empty:
        fig, ax = plt.subplots(
            figsize=(
                5.6,
                3.4,
            )
        )

        for name in order:
            grp = subset.loc[
                (
                    subset[
                        "model"
                    ] == name
                )
                & (
                    subset[
                        "target"
                    ] == "vessel_vector"
                )
            ].sort_values(
                "horizon_min"
            )

            if grp.empty:
                continue

            ax.plot(
                grp[
                    "horizon_min"
                ],
                grp[
                    "correction_raw_RMS"
                ],
                marker="o",
                label=name,
            )

        ax.set_xlabel(
            "Forecast horizon (min)"
        )

        ax.set_ylabel(
            r"Vessel correction RMS (m s$^{-1}$)"
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
            "05_vessel_correction_by_horizon",
        )

    # Training validation AW.
    fig, ax = plt.subplots(
        figsize=(
            5.4,
            3.4,
        )
    )

    for name, hist in (
        histories.items()
    ):
        ax.plot(
            hist[
                "epoch"
            ],
            hist[
                "val_apparent_vector_RMSE_mps"
            ],
            label=name,
        )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        r"Validation AW RMSE (m s$^{-1}$)"
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
        "06_training_validation_aw",
    )

    if has_development_test:
        curve(
            "development_test",
            "apparent_vector_RMSE_mps",
            r"Development-test AW RMSE (m s$^{-1}$)",
            "07_development_test_apparent_rmse",
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
        "--variants",
        default=DEFAULT_VARIANTS,
        help=(
            "Comma-separated structured variants. "
            f"Default: {DEFAULT_VARIANTS}"
        ),
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
        "--seed",
        type=int,
        default=None,
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

    if args.lambda_residual < 0:
        raise ValueError(
            "--lambda-residual must be >= 0."
        )

    variants = parse_variants(
        args.variants
    )

    shared07a = load_shared_07a()
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
            / "structured_residual_ablation_v0_1"
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

    prediction_dir = (
        output_dir
        / "predictions"
    )

    for d in [
        output_dir,
        model_dir,
        history_dir,
        prediction_dir,
    ]:
        d.mkdir(
            parents=True,
            exist_ok=True,
        )

    core06.log(
        "=" * 100
    )

    core06.log(
        "07B - STRUCTURED RESIDUAL ABLATION"
    )

    core06.log(
        "=" * 100
    )

    core06.log(
        f"script_version      : {SCRIPT_VERSION}"
    )

    core06.log(
        f"variants            : {variants}"
    )

    core06.log(
        f"dataset_dir         : {dataset_dir}"
    )

    core06.log(
        f"output_dir          : {output_dir}"
    )

    core06.log(
        f"lambda_residual     : {args.lambda_residual}"
    )

    core06.log(
        f"development test    : {args.evaluate_development_test}"
    )

    core06.log("")

    # ------------------------------------------------------------------
    # Stage 1/8: frozen configs
    # ------------------------------------------------------------------
    core06.log(
        "[STAGE 1/8] Frozen dataset / Ridge / GRU / physics configuration"
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

    if feature_names != EXPECTED_FEATURE_NAMES:
        raise RuntimeError(
            "Feature schema mismatch."
        )

    if target_names != EXPECTED_TARGET_NAMES:
        raise RuntimeError(
            "Target schema mismatch."
        )

    (
        f_names,
        feature_mean,
        feature_std,
    ) = core06.parse_scaler(
        scalers,
        "feature_scaler",
    )

    (
        t_names,
        target_mean,
        target_std,
    ) = core06.parse_scaler(
        scalers,
        "target_scaler",
    )

    if f_names != feature_names:
        raise RuntimeError(
            "Feature scaler schema mismatch."
        )

    if t_names != target_names:
        raise RuntimeError(
            "Target scaler schema mismatch."
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

    matched_seed = (
        int(
            args.seed
        )
        if args.seed is not None
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

    core06.log(
        "[FIXED GRU] "
        f"units={gru_config['recurrent_units']}, "
        f"layers={gru_config['recurrent_layers']}, "
        f"dropout={gru_config['dropout']}, "
        f"lr={gru_config['learning_rate']}"
    )

    core06.log(
        "[FIXED PHYSICS] "
        f"E={physics_weights['earth']}, "
        f"B={physics_weights['body']}, "
        f"H={physics_weights['heading_geo']}, "
        f"N={physics_weights['heading_norm']}"
    )

    core06.log(
        f"[MATCHED SEED] {matched_seed}"
    )

    # ------------------------------------------------------------------
    # Stage 2/8: data/device/Ridge
    # ------------------------------------------------------------------
    core06.log(
        "[STAGE 2/8] Loading data, CUDA, frozen Ridge predictions"
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

    core06.set_global_seed(
        torch,
        matched_seed,
        args.deterministic,
    )

    device_info = (
        core06.log_device_info(
            torch,
            device,
            amp_enabled,
            tf32_enabled,
        )
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

    ModelClass = build_structured_model_class(
        torch,
        nn,
    )

    # ------------------------------------------------------------------
    # Stage 3/8: matched ablation training
    # ------------------------------------------------------------------
    core06.log(
        "[STAGE 3/8] Training structured residual variants"
    )

    metric_frames = [
        ridge_val_metrics
    ]

    training_rows = []
    gate_frames = []
    correction_frames = []
    histories = {}
    trained = {}

    for variant_name in variants:
        spec = VARIANT_SPECS[
            variant_name
        ]

        mask = build_mask(
            spec[
                "active_groups"
            ],
            len(
                horizons
            ),
            len(
                target_names
            ),
        )

        active_target_indices = (
            group_target_indices(
                spec[
                    "active_groups"
                ]
            )
        )

        core06.log("")
        core06.log(
            "=" * 100
        )

        core06.log(
            f"TRAINING {variant_name}"
        )

        core06.log(
            "=" * 100
        )

        core06.log(
            f"active_groups={spec['active_groups']} | "
            f"use_physics={spec['use_physics']} | "
            f"active_targets="
            f"{[target_names[i] for i in active_target_indices]}"
        )

        # Same seed and freshly recreated loader for matched ablation.
        core06.set_global_seed(
            torch,
            matched_seed,
            args.deterministic,
        )

        train_loader = (
            shared07a.make_loader(
                DataLoader,
                train_ds,
                args.batch_size,
                True,
                args.num_workers,
                pin_memory,
                persistent_workers,
                args.prefetch_factor,
            )
        )

        val_loader = (
            shared07a.make_loader(
                DataLoader,
                val_ds,
                args.batch_size,
                False,
                args.num_workers,
                pin_memory,
                persistent_workers,
                args.prefetch_factor,
            )
        )

        model = ModelClass(
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

        n_params = (
            shared07a.count_parameters(
                model
            )
        )

        checkpoint = (
            model_dir
            / (
                variant_name.replace(
                    "-",
                    "_",
                )
                + ".pt"
            )
        )

        result = train_variant(
            shared07a,
            core06,
            torch,
            model=model,
            variant_name=variant_name,
            use_physics=bool(
                spec[
                    "use_physics"
                ]
            ),
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
            physics_weights=physics_weights,
            lambda_residual=args.lambda_residual,
            checkpoint_path=checkpoint,
            horizons=horizons,
        )

        history = result[
            "history"
        ]

        histories[
            variant_name
        ] = history

        history.to_csv(
            history_dir
            / (
                variant_name.replace(
                    "-",
                    "_",
                )
                + "_history.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

        val_result = result[
            "validation"
        ]

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
                model_name=variant_name,
                direction_min_speed=args.direction_min_speed,
                cog_min_speed=args.cog_min_speed,
                persistence_joint=persistence_val,
            )
        )

        metric_frames.append(
            val_metrics
        )

        gate_frames.append(
            active_gate_statistics(
                model_name=variant_name,
                split_name="validation",
                gate=val_result[
                    "gate"
                ],
                mask=mask,
                horizons=horizons,
                target_names=target_names,
            )
        )

        correction_frames.append(
            shared07a.correction_statistics(
                variant_name,
                "validation",
                val_result[
                    "correction_z"
                ],
                horizons,
                target_names,
                target_std,
            )
        )

        shared07a.save_prediction_bundle(
            prediction_dir
            / (
                "validation_predictions_"
                + variant_name.replace(
                    "-",
                    "_",
                )
                + ".npz"
            ),
            variant_name,
            validation[
                "y_raw"
            ],
            val_result[
                "pred_raw"
            ],
            ridge_val_raw,
            validation[
                "context_end_time_ns"
            ],
            validation[
                "target_time_ns"
            ],
            horizons,
            gate=val_result[
                "gate"
            ],
            correction_z=val_result[
                "correction_z"
            ],
        )

        s = (
            core06.summarize_metrics(
                val_metrics
            ).iloc[
                0
            ]
        )

        training_rows.append({
            "model": variant_name,
            "seed": int(
                matched_seed
            ),
            "parameters": int(
                n_params
            ),
            "active_groups": "+".join(
                spec[
                    "active_groups"
                ]
            ),
            "active_target_count": int(
                len(
                    active_target_indices
                )
            ),
            "use_physics": bool(
                spec[
                    "use_physics"
                ]
            ),
            "lambda_residual": float(
                args.lambda_residual
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
            "validation_AW_RMSE_mps": float(
                s[
                    "mean_apparent_vector_RMSE_mps"
                ]
            ),
            "validation_wind_RMSE_mps": float(
                s[
                    "mean_wind_vector_RMSE_mps"
                ]
            ),
            "validation_vessel_RMSE_mps": float(
                s[
                    "mean_vessel_vector_RMSE_mps"
                ]
            ),
            "validation_AWS_RMSE_mps": float(
                s[
                    "mean_AWS_RMSE_mps"
                ]
            ),
            "validation_AWA_MAE_deg": float(
                s[
                    "mean_AWA_MAE_deg"
                ]
            ),
            "validation_HDG_MAE_deg": float(
                s[
                    "mean_HDG_MAE_deg"
                ]
            ),
            "validation_active_correction_z_RMS": float(
                np.sqrt(
                    np.mean(
                        val_result[
                            "correction_z"
                        ] ** 2
                    )
                )
            ),
        })

        trained[
            variant_name
        ] = {
            "model": model,
            "mask": mask,
        }

        core06.log(
            f"[VALIDATION] {variant_name}: "
            f"AW={s['mean_apparent_vector_RMSE_mps']:.4f} | "
            f"wind={s['mean_wind_vector_RMSE_mps']:.4f} | "
            f"vessel={s['mean_vessel_vector_RMSE_mps']:.4f} | "
            f"AWS={s['mean_AWS_RMSE_mps']:.4f} | "
            f"AWA={s['mean_AWA_MAE_deg']:.2f} deg"
        )

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    training_df = pd.DataFrame(
        training_rows
    )

    training_df.to_csv(
        output_dir
        / "structured_ablation_training_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Stage 4/8: optional development test
    # ------------------------------------------------------------------
    if args.evaluate_development_test:
        core06.log("")
        core06.log(
            "[STAGE 4/8] Evaluating already-inspected SD1090 development test"
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

        ridge_test_metrics = (
            core06.evaluate_model(
                truth_joint=test[
                    "y_raw"
                ],
                pred_joint=ridge_test_raw,
                truth_apparent_ref=test[
                    "apparent_earth_raw"
                ],
                horizons=horizons,
                split_name="development_test",
                model_name="Ridge",
                direction_min_speed=args.direction_min_speed,
                cog_min_speed=args.cog_min_speed,
                persistence_joint=persistence_test,
            )
        )

        metric_frames.append(
            ridge_test_metrics
        )

        test_ds = (
            shared07a.make_dataset(
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
        )

        test_loader = (
            shared07a.make_loader(
                DataLoader,
                test_ds,
                args.batch_size,
                False,
                args.num_workers,
                pin_memory,
                persistent_workers,
                args.prefetch_factor,
            )
        )

        for variant_name in variants:
            spec = VARIANT_SPECS[
                variant_name
            ]

            result = (
                evaluate_structured_loader(
                    shared07a,
                    core06,
                    torch,
                    model=trained[
                        variant_name
                    ][
                        "model"
                    ],
                    loader=test_loader,
                    device=device,
                    amp_enabled=amp_enabled,
                    target_mean_t=target_mean_t,
                    target_std_t=target_std_t,
                    target_mean_np=target_mean,
                    target_std_np=target_std,
                    physics_scale_t=physics_scale_t,
                    physics_weights=physics_weights,
                    lambda_residual=args.lambda_residual,
                    use_physics=bool(
                        spec[
                            "use_physics"
                        ]
                    ),
                )
            )

            m = (
                core06.evaluate_model(
                    truth_joint=test[
                        "y_raw"
                    ],
                    pred_joint=result[
                        "pred_raw"
                    ],
                    truth_apparent_ref=test[
                        "apparent_earth_raw"
                    ],
                    horizons=horizons,
                    split_name="development_test",
                    model_name=variant_name,
                    direction_min_speed=args.direction_min_speed,
                    cog_min_speed=args.cog_min_speed,
                    persistence_joint=persistence_test,
                )
            )

            metric_frames.append(
                m
            )

            gate_frames.append(
                active_gate_statistics(
                    model_name=variant_name,
                    split_name="development_test",
                    gate=result[
                        "gate"
                    ],
                    mask=trained[
                        variant_name
                    ][
                        "mask"
                    ],
                    horizons=horizons,
                    target_names=target_names,
                )
            )

            correction_frames.append(
                shared07a.correction_statistics(
                    variant_name,
                    "development_test",
                    result[
                        "correction_z"
                    ],
                    horizons,
                    target_names,
                    target_std,
                )
            )

            shared07a.save_prediction_bundle(
                prediction_dir
                / (
                    "development_test_predictions_"
                    + variant_name.replace(
                        "-",
                        "_",
                    )
                    + ".npz"
                ),
                variant_name,
                test[
                    "y_raw"
                ],
                result[
                    "pred_raw"
                ],
                ridge_test_raw,
                test[
                    "context_end_time_ns"
                ],
                test[
                    "target_time_ns"
                ],
                horizons,
                gate=result[
                    "gate"
                ],
                correction_z=result[
                    "correction_z"
                ],
            )

    else:
        core06.log("")
        core06.log(
            "[STAGE 4/8] SD1090 development test intentionally NOT evaluated"
        )

    # ------------------------------------------------------------------
    # Stage 5/8: metrics and effects
    # ------------------------------------------------------------------
    core06.log(
        "[STAGE 5/8] Building ablation metrics"
    )

    metrics = pd.concat(
        metric_frames,
        ignore_index=True,
    )

    metrics.to_csv(
        output_dir
        / "structured_ablation_metrics_per_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = core06.summarize_metrics(
        metrics
    )

    summary.to_csv(
        output_dir
        / "structured_ablation_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    effects = compute_effect_table(
        summary
    )

    effects.to_csv(
        output_dir
        / "structured_ablation_effects.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gate_stats = pd.concat(
        gate_frames,
        ignore_index=True,
    )

    correction_stats = pd.concat(
        correction_frames,
        ignore_index=True,
    )

    gate_stats.to_csv(
        output_dir
        / "structured_ablation_gate_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    correction_stats.to_csv(
        output_dir
        / "structured_ablation_correction_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Stage 6/8: scientific audit
    # ------------------------------------------------------------------
    core06.log(
        "[STAGE 6/8] Scientific interpretation audit"
    )

    parameter_counts = (
        training_df[
            "parameters"
        ]
        .astype(
            int
        )
        .unique()
    )

    nominal_parameter_match = (
        len(
            parameter_counts
        )
        == 1
    )

    best_validation_model = None
    best_validation_aw = None

    if not effects.empty:
        best_validation_model = str(
            effects.iloc[
                0
            ][
                "model"
            ]
        )

        best_validation_aw = float(
            effects.iloc[
                0
            ][
                "validation_AW_RMSE_mps"
            ]
        )

    vessel_only_fraction = None

    if not effects.empty:
        row = effects.loc[
            effects[
                "model"
            ] == "Vessel-Only-Residual"
        ]

        if (
            not row.empty
            and pd.notna(
                row.iloc[
                    0
                ][
                    "fraction_of_Full_Residual_AW_gain"
                ]
            )
        ):
            vessel_only_fraction = float(
                row.iloc[
                    0
                ][
                    "fraction_of_Full_Residual_AW_gain"
                ]
            )

    audit = {
        "nominal_parameter_count_matched_across_neural_variants": bool(
            nominal_parameter_match
        ),
        "nominal_trainable_parameters": (
            int(
                parameter_counts[
                    0
                ]
            )
            if nominal_parameter_match
            else [
                int(
                    v
                )
                for v in parameter_counts
            ]
        ),
        "hard_mask_is_nontrainable": True,
        "best_validation_model": best_validation_model,
        "best_validation_AW_RMSE_mps": best_validation_aw,
        "vessel_only_fraction_of_full_residual_AW_gain": vessel_only_fraction,
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
        "variants": {
            name: {
                "active_groups": list(
                    VARIANT_SPECS[
                        name
                    ][
                        "active_groups"
                    ]
                ),
                "use_physics": bool(
                    VARIANT_SPECS[
                        name
                    ][
                        "use_physics"
                    ]
                ),
            }
            for name in variants
        },
        "fixed_ridge_model": str(
            ridge_model_path.resolve()
        ),
        "ridge_manifest": ridge_manifest,
        "ridge_reproduction_audit": ridge_audit,
        "fixed_gru_config": gru_config,
        "fixed_06B_physics_weights": physics_weights,
        "physics_weight_metadata": physics_meta,
        "matched_seed": int(
            matched_seed
        ),
        "lambda_residual": float(
            args.lambda_residual
        ),
        "gate_bias": float(
            args.gate_bias
        ),
        "residual_init_std": float(
            args.residual_init_std
        ),
        "protocol": {
            "train_split": "frozen outer train",
            "checkpoint_split": "frozen outer validation",
            "ridge_frozen": True,
            "same_neural_architecture_across_variants": True,
            "hard_branch_mask_only_structural_difference": True,
            "same_seed_across_variants": True,
            "development_test_evaluated": bool(
                args.evaluate_development_test
            ),
            "development_test_used_for_selection": False,
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
        / "structured_ablation_manifest.json",
        manifest_out,
    )

    with (
        output_dir
        / "structured_ablation_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "07B Structured Residual Ablation Report\n"
        )

        f.write(
            "=" * 100
            + "\n\n"
        )

        f.write(
            "Variants\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )

        for name in variants:
            f.write(
                f"{name}: "
                f"active={VARIANT_SPECS[name]['active_groups']}, "
                f"physics={VARIANT_SPECS[name]['use_physics']}\n"
            )

        f.write(
            "\nTraining summary\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )

        f.write(
            training_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nValidation metrics summary\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )

        f.write(
            summary.loc[
                summary[
                    "split"
                ] == "validation"
            ].to_string(
                index=False
            )
        )

        f.write(
            "\n\nAblation effects\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )

        f.write(
            effects.to_string(
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
    # Stage 8/8: plots / console
    # ------------------------------------------------------------------
    core06.log(
        "[STAGE 8/8] Finalization"
    )

    if args.plots:
        make_plots(
            core06,
            metrics=metrics,
            effects=effects,
            correction_stats=correction_stats,
            histories=histories,
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
        "07B STRUCTURED RESIDUAL ABLATION RESULTS"
    )
    core06.log(
        "=" * 100
    )

    val_summary = summary.loc[
        summary[
            "split"
        ] == "validation"
    ]

    ridge_row = val_summary.loc[
        val_summary[
            "model"
        ] == "Ridge"
    ]

    if not ridge_row.empty:
        r = ridge_row.iloc[
            0
        ]

        core06.log(
            "Ridge:"
        )

        core06.log(
            f"  AW RMSE      = "
            f"{r['mean_apparent_vector_RMSE_mps']:.4f} m/s"
        )

        core06.log(
            f"  wind RMSE    = "
            f"{r['mean_wind_vector_RMSE_mps']:.4f} m/s"
        )

        core06.log(
            f"  vessel RMSE  = "
            f"{r['mean_vessel_vector_RMSE_mps']:.4f} m/s"
        )

    for variant_name in variants:
        row = val_summary.loc[
            val_summary[
                "model"
            ] == variant_name
        ]

        if row.empty:
            continue

        r = row.iloc[
            0
        ]

        core06.log(
            f"{variant_name}:"
        )

        core06.log(
            f"  AW RMSE      = "
            f"{r['mean_apparent_vector_RMSE_mps']:.4f} m/s"
        )

        core06.log(
            f"  wind RMSE    = "
            f"{r['mean_wind_vector_RMSE_mps']:.4f} m/s"
        )

        core06.log(
            f"  vessel RMSE  = "
            f"{r['mean_vessel_vector_RMSE_mps']:.4f} m/s"
        )

        core06.log(
            f"  AWS RMSE     = "
            f"{r['mean_AWS_RMSE_mps']:.4f} m/s"
        )

        core06.log(
            f"  AWA MAE      = "
            f"{r['mean_AWA_MAE_deg']:.2f} deg"
        )

    core06.log("")

    if not effects.empty:
        for _, row in (
            effects.iterrows()
        ):
            fraction = (
                row[
                    "fraction_of_Full_Residual_AW_gain"
                ]
            )

            fraction_text = (
                "NA"
                if pd.isna(
                    fraction
                )
                else f"{100.0 * fraction:.1f}%"
            )

            core06.log(
                f"[ABLATION] {row['model']}: "
                f"AW improvement vs Ridge="
                f"{row['AW_improvement_vs_Ridge_percent']:.3f}% | "
                f"fraction of Full gain="
                f"{fraction_text}"
            )

    core06.log(
        f"[BEST VALIDATION VARIANT] "
        f"{best_validation_model} | "
        f"AW={best_validation_aw}"
    )

    if (
        vessel_only_fraction
        is not None
    ):
        core06.log(
            "[VESSEL-ONLY CONTRIBUTION] "
            f"captures "
            f"{100.0 * vessel_only_fraction:.1f}% "
            f"of Full-Residual AW gain"
        )

    core06.log(
        f"[PARAMETER MATCH] "
        f"{nominal_parameter_match}"
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
        f"[DONE] 07B outputs: "
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
