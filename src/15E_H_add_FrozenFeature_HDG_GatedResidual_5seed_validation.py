# -*- coding: utf-8 -*-
"""
15E_H_add_FrozenFeature_HDG_GatedResidual_5seed_validation.py

15E-H
=====
Start from the frozen 15E idea:

    NEW Stage15B true wind
    +
    OLD manuscript compact vessel residual
    +
    OLD joint-Ridge heading

and add ONLY a heading residual correction.

Key scientific constraint
-------------------------
Wind and vessel predictions are unchanged by the HDG experiment.

Therefore:
    Earth-frame apparent-wind vector = W - V
is unchanged.

Consequently:
    AppVector RMSE must be identical between 15E and 15E-H.
    AWS RMSE must be identical between 15E and 15E-H.

Only:
    HDG metrics
and
    body-frame AWA metrics
may change.

HDG residual design
-------------------
The documented old vessel network is first trained exactly as in 15E:
    single-layer GRU64
    + 64-unit fusion layer
    + vessel residual head
    + vessel sigmoid gate head

After that model is frozen, its 64-dimensional fusion representation z
is extracted in eval mode.

A tiny heading-only gated residual head is trained on the frozen z:

    delta_q = f_delta(z)     -> 5 x 2
    gate_q  = sigmoid(f_g(z))-> 5 x 2

where q = [sin(HDG), cos(HDG)].

The initial residual output is zero and gate bias is -1, so the initial
prediction is exactly the Ridge heading.

A horizon-wise global safety shrinkage alpha_HDG(h) in [0,1] is then
selected using VALIDATION HDG MAE only:

    q_tilde(h) = q_Ridge(h)
               + alpha_HDG(h) * gate(h) * delta_q(h)

The corrected raw sine/cosine pair is normalized to the unit circle
before conversion to heading.

The alpha search DOES NOT optimize AWA. Therefore any AWA improvement is
a consequence of better heading, not direct tuning to apparent-wind angle.

Five seeds
----------
500043, 501052, 502061, 503070, 504079

TRAIN/VALIDATION only.
Internal TEST is NOT loaded.

Dependency
----------
This script reuses the validated helper functions and documented old
vessel architecture from:

    15E_hybrid_NewWind_OldCompactVessel_5seed_validation.py

That 15E script should already exist in --src-dir.

Recommended PowerShell
----------------------
python "D:\\project\\WindPredict_SaildroneData\\src\\15E_H_add_FrozenFeature_HDG_GatedResidual_5seed_validation.py" --dataset-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\SD1090_TPOS2024_JointForecasting_v0_1" --stage15a-truewind-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15A_R_1min_Dual_Ridge_Anchors_v0_2" --src-dir "D:\\project\\WindPredict_SaildroneData\\src" --output-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15E_H_HDG_GatedResidual_5seed_v0_1"
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import math
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\project\WindPredict_SaildroneData")

DEFAULT_DATASET_DIR = (
    ROOT / "data" / "forecasting"
    / "SD1090_TPOS2024_JointForecasting_v0_1"
)

DEFAULT_STAGE15A_TRUEWIND_DIR = (
    ROOT / "data" / "forecasting"
    / "15A_R_1min_Dual_Ridge_Anchors_v0_2"
)

DEFAULT_SRC_DIR = ROOT / "src"

DEFAULT_OUTPUT_DIR = (
    ROOT / "data" / "forecasting"
    / "15E_H_HDG_GatedResidual_5seed_v0_1"
)

SEEDS = [500043, 501052, 502061, 503070, 504079]
HORIZONS = [1, 2, 3, 5, 10]

HDG_LR = 1e-3
HDG_WEIGHT_DECAY = 1e-5
HDG_BATCH_SIZE = 512
HDG_MAX_EPOCHS = 80
HDG_PATIENCE = 12
HDG_GRAD_CLIP = 1.0

ALPHA_GRID_SIZE = 1001
ALPHA_MIN = 0.0
ALPHA_MAX = 1.0

EPS = 1e-8

# The new HDG head reuses the frozen 64-d old-vessel fusion feature.
HDG_FEATURE_DIM = 64
HDG_OUTPUT_DIM = 10  # 5 horizons x [sin, cos]

# Two Linear(64 -> 10) heads.
EXPECTED_HDG_HEAD_PARAMETERS = (
    (HDG_FEATURE_DIM * HDG_OUTPUT_DIM + HDG_OUTPUT_DIM)
    + (HDG_FEATURE_DIM * HDG_OUTPUT_DIM + HDG_OUTPUT_DIM)
)  # 1300


def log(msg=""):
    print(msg, flush=True)


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save_json(path: Path, obj):
    def cv(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return None if not np.isfinite(v) else float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, dict):
            return {str(k): cv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [cv(x) for x in v]
        return v

    with path.open("w", encoding="utf-8") as f:
        json.dump(cv(obj), f, ensure_ascii=False, indent=2, sort_keys=True)


def seed_everything(torch, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_dict_cpu(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def count_parameters(model):
    return int(sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    ))


def wrap180_deg(x):
    return ((x + 180.0) % 360.0) - 180.0


def heading_deg_from_q(q):
    """
    q[..., 0] = sin(psi), q[..., 1] = cos(psi)
    navigation heading: clockwise from North.
    """
    q = np.asarray(q, dtype=np.float64)

    n = np.sqrt(
        q[..., 0] ** 2
        + q[..., 1] ** 2
    )

    n = np.maximum(n, EPS)

    s = q[..., 0] / n
    c = q[..., 1] / n

    return (
        np.degrees(
            np.arctan2(s, c)
        ) % 360.0
    )


def hdg_mae_per_horizon(q_true, q_pred):
    true_deg = heading_deg_from_q(q_true)
    pred_deg = heading_deg_from_q(q_pred)

    err = wrap180_deg(
        pred_deg - true_deg
    )

    return np.mean(
        np.abs(err),
        axis=0,
    )


def hdg_rmse_per_horizon(q_true, q_pred):
    true_deg = heading_deg_from_q(q_true)
    pred_deg = heading_deg_from_q(q_pred)

    err = wrap180_deg(
        pred_deg - true_deg
    )

    return np.sqrt(
        np.mean(
            err * err,
            axis=0,
        )
    )


def normalize_q(q):
    q = np.asarray(q, dtype=np.float32)

    norm = np.sqrt(
        np.sum(
            q * q,
            axis=-1,
            keepdims=True,
        )
    )

    norm = np.maximum(
        norm,
        EPS,
    )

    return (
        q / norm
    ).astype(np.float32)


def old_heading_scaler_arrays(old_ridge_model):
    ym = np.asarray(
        old_ridge_model["y_mean_raw"],
        dtype=np.float64,
    ).reshape(5, 6)

    ys = np.asarray(
        old_ridge_model["y_std_raw"],
        dtype=np.float64,
    ).reshape(5, 6)

    return (
        ym[:, 4:6].astype(np.float32),
        ys[:, 4:6].astype(np.float32),
    )


def q_raw_to_z(q_raw, old_ridge_model):
    mean_q, std_q = old_heading_scaler_arrays(
        old_ridge_model
    )

    return (
        (
            np.asarray(q_raw, dtype=np.float32)
            - mean_q[None, :, :]
        )
        / std_q[None, :, :]
    ).astype(np.float32)


def q_z_to_raw(q_z, old_ridge_model):
    mean_q, std_q = old_heading_scaler_arrays(
        old_ridge_model
    )

    return (
        np.asarray(q_z, dtype=np.float32)
        * std_q[None, :, :]
        + mean_q[None, :, :]
    ).astype(np.float32)


# =============================================================================
# Extract frozen old-vessel fusion representation
# =============================================================================

def extract_old_fusion_features(
    *,
    torch,
    model,
    X,
    ridge_z,
    device,
    batch_size=512,
):
    """
    Reproduce the frozen old network up to the 64-d fusion feature.

    IMPORTANT:
    model.eval() disables dropout, making the feature deterministic.
    No old vessel weight is modified.
    """
    model.eval()

    feats = []

    with torch.no_grad():
        for a in range(
            0,
            len(X),
            batch_size,
        ):
            b = min(
                a + batch_size,
                len(X),
            )

            seq_b = (
                torch.from_numpy(
                    X[a:b]
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            rz_b = (
                torch.from_numpy(
                    ridge_z[a:b].reshape(
                        b - a,
                        -1,
                    )
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            # Same internal path as 15E OldCompactVesselResidual.
            out, _ = model.gru(seq_b)

            h = out[:, -1, :]

            h = model.history_dropout(h)

            z = model.fusion(
                torch.cat(
                    [h, rz_b],
                    dim=1,
                )
            )

            feats.append(
                z.detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    Z = np.concatenate(
        feats,
        axis=0,
    )

    if Z.shape != (
        len(X),
        HDG_FEATURE_DIM,
    ):
        raise RuntimeError(
            f"Unexpected frozen feature shape: {Z.shape}"
        )

    if not np.isfinite(Z).all():
        raise RuntimeError(
            "Frozen HDG features contain nonfinite values."
        )

    return Z


# =============================================================================
# Tiny heading residual head
# =============================================================================

def build_hdg_head_class(torch, nn):
    class HDGGatedResidualHead(nn.Module):
        def __init__(self):
            super().__init__()

            self.residual_head = nn.Linear(
                HDG_FEATURE_DIM,
                HDG_OUTPUT_DIM,
            )

            self.gate_head = nn.Linear(
                HDG_FEATURE_DIM,
                HDG_OUTPUT_DIM,
            )

            # Start exactly from Ridge.
            nn.init.zeros_(
                self.residual_head.weight
            )
            nn.init.zeros_(
                self.residual_head.bias
            )

            nn.init.zeros_(
                self.gate_head.weight
            )
            nn.init.constant_(
                self.gate_head.bias,
                -1.0,
            )

        def forward(self, z):
            residual = (
                self.residual_head(z)
                .reshape(-1, 5, 2)
            )

            gate = (
                torch.sigmoid(
                    self.gate_head(z)
                )
                .reshape(-1, 5, 2)
            )

            correction = gate * residual

            return (
                residual,
                gate,
                correction,
            )

    return HDGGatedResidualHead


def predict_hdg_raw_correction(
    *,
    torch,
    model,
    Z,
    device,
):
    model.eval()

    corr = []
    gate = []
    residual = []

    with torch.no_grad():
        for a in range(
            0,
            len(Z),
            HDG_BATCH_SIZE,
        ):
            b = min(
                a + HDG_BATCH_SIZE,
                len(Z),
            )

            zb = (
                torch.from_numpy(
                    Z[a:b]
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            r, g, c = model(zb)

            residual.append(
                r.detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            gate.append(
                g.detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            corr.append(
                c.detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    return (
        np.concatenate(
            residual,
            axis=0,
        ),
        np.concatenate(
            gate,
            axis=0,
        ),
        np.concatenate(
            corr,
            axis=0,
        ),
    )


def apply_hdg_alpha(
    ridge_q_z,
    raw_corr_z,
    alpha_h,
    old_ridge_model,
):
    alpha_h = np.asarray(
        alpha_h,
        dtype=np.float32,
    ).reshape(1, 5, 1)

    q_z = (
        np.asarray(
            ridge_q_z,
            dtype=np.float32,
        )
        + alpha_h
        * np.asarray(
            raw_corr_z,
            dtype=np.float32,
        )
    )

    q_raw = q_z_to_raw(
        q_z,
        old_ridge_model,
    )

    return normalize_q(q_raw)


def select_alpha_per_horizon(
    *,
    y_q_raw_val,
    ridge_q_z_val,
    raw_corr_z_val,
    old_ridge_model,
):
    """
    Validation-only alpha selection using HDG MAE, horizon by horizon.

    AWA is NOT used.
    """
    grid = np.linspace(
        ALPHA_MIN,
        ALPHA_MAX,
        ALPHA_GRID_SIZE,
        dtype=np.float64,
    )

    alpha = np.zeros(
        5,
        dtype=np.float32,
    )

    score = np.zeros(
        5,
        dtype=np.float64,
    )

    true_deg = heading_deg_from_q(
        y_q_raw_val
    )

    for h in range(5):
        best_a = 0.0

        base_q = q_z_to_raw(
            ridge_q_z_val[:, h:h+1, :],
            old_ridge_model={
                "y_mean_raw": np.asarray(
                    old_ridge_model["y_mean_raw"]
                ).reshape(5, 6)[h:h+1]
                .reshape(-1),
                "y_std_raw": np.asarray(
                    old_ridge_model["y_std_raw"]
                ).reshape(5, 6)[h:h+1]
                .reshape(-1),
            }
        )
        # The helper above expects 5x6, so use direct conversion here.
        mean_q, std_q = old_heading_scaler_arrays(
            old_ridge_model
        )

        base_z = ridge_q_z_val[:, h, :]
        corr_z = raw_corr_z_val[:, h, :]

        best_score = float(
            hdg_mae_per_horizon(
                y_q_raw_val[:, h:h+1, :],
                normalize_q(
                    (
                        base_z[:, None, :]
                        * std_q[h:h+1][None, :, :]
                        + mean_q[h:h+1][None, :, :]
                    )
                ),
            )[0]
        )

        for a in grid[1:]:
            q_z = (
                base_z
                + float(a) * corr_z
            )

            q_raw = (
                q_z
                * std_q[h][None, :]
                + mean_q[h][None, :]
            )

            q_raw = normalize_q(
                q_raw[:, None, :]
            )[:, 0, :]

            pred_deg = heading_deg_from_q(
                q_raw
            )

            err = wrap180_deg(
                pred_deg
                - true_deg[:, h]
            )

            mae = float(
                np.mean(
                    np.abs(err)
                )
            )

            if mae < best_score - 1e-12:
                best_score = mae
                best_a = float(a)

        alpha[h] = best_a
        score[h] = best_score

    return (
        alpha,
        score,
    )


# =============================================================================
# Fix for helper above: a small direct alpha selector
# =============================================================================

def select_alpha_per_horizon(
    *,
    y_q_raw_val,
    ridge_q_z_val,
    raw_corr_z_val,
    old_ridge_model,
):
    grid = np.linspace(
        ALPHA_MIN,
        ALPHA_MAX,
        ALPHA_GRID_SIZE,
        dtype=np.float64,
    )

    mean_q, std_q = old_heading_scaler_arrays(
        old_ridge_model
    )

    true_deg = heading_deg_from_q(
        y_q_raw_val
    )

    alpha = np.zeros(
        5,
        dtype=np.float32,
    )

    best_scores = np.zeros(
        5,
        dtype=np.float64,
    )

    for h in range(5):
        best_a = 0.0
        best_mae = np.inf

        base_z = np.asarray(
            ridge_q_z_val[:, h, :],
            dtype=np.float64,
        )

        corr_z = np.asarray(
            raw_corr_z_val[:, h, :],
            dtype=np.float64,
        )

        for a in grid:
            q_z = (
                base_z
                + float(a) * corr_z
            )

            q_raw = (
                q_z
                * std_q[h][None, :]
                + mean_q[h][None, :]
            )

            norm = np.sqrt(
                np.sum(
                    q_raw * q_raw,
                    axis=1,
                    keepdims=True,
                )
            )

            norm = np.maximum(
                norm,
                EPS,
            )

            q_raw = q_raw / norm

            pred_deg = (
                np.degrees(
                    np.arctan2(
                        q_raw[:, 0],
                        q_raw[:, 1],
                    )
                )
                % 360.0
            )

            err = wrap180_deg(
                pred_deg
                - true_deg[:, h]
            )

            mae = float(
                np.mean(
                    np.abs(err)
                )
            )

            if mae < best_mae - 1e-12:
                best_mae = mae
                best_a = float(a)

        alpha[h] = best_a
        best_scores[h] = best_mae

    return (
        alpha,
        best_scores,
    )


def train_hdg_head(
    *,
    torch,
    nn,
    seed,
    Z_train,
    ridge_q_z_train,
    y_q_z_train,
    Z_val,
    ridge_q_z_val,
    y_q_raw_val,
    old_ridge_model,
):
    seed_everything(
        torch,
        seed + 700000,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    Head = build_hdg_head_class(
        torch,
        nn,
    )

    model = Head().to(
        device
    )

    params = count_parameters(
        model
    )

    if (
        params
        != EXPECTED_HDG_HEAD_PARAMETERS
    ):
        raise RuntimeError(
            f"HDG head parameters = {params}, "
            f"expected {EXPECTED_HDG_HEAD_PARAMETERS}."
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HDG_LR,
        weight_decay=HDG_WEIGHT_DECAY,
    )

    best_state = state_dict_cpu(
        model
    )

    # Epoch 0 exactly Ridge.
    ridge_q_raw_val = q_z_to_raw(
        ridge_q_z_val,
        old_ridge_model,
    )

    ridge_q_raw_val = normalize_q(
        ridge_q_raw_val
    )

    best_score = float(
        np.mean(
            hdg_mae_per_horizon(
                y_q_raw_val,
                ridge_q_raw_val,
            )
        )
    )

    best_epoch = 0
    patience = 0
    history = []

    n = len(
        Z_train
    )

    for epoch in range(
        1,
        HDG_MAX_EPOCHS + 1,
    ):
        model.train()

        rng = np.random.default_rng(
            seed
            + 17041
            * epoch
        )

        order = np.arange(
            n
        )

        rng.shuffle(
            order
        )

        total_loss = 0.0
        seen = 0

        for a in range(
            0,
            n,
            HDG_BATCH_SIZE,
        ):
            idx = order[
                a:
                a + HDG_BATCH_SIZE
            ]

            z_b = (
                torch.from_numpy(
                    Z_train[idx]
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            ridge_b = (
                torch.from_numpy(
                    ridge_q_z_train[idx]
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            target_b = (
                torch.from_numpy(
                    y_q_z_train[idx]
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            _, _, correction = model(
                z_b
            )

            pred = (
                ridge_b
                + correction
            )

            loss = torch.mean(
                (
                    pred
                    - target_b
                )
                ** 2
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                HDG_GRAD_CLIP,
            )

            optimizer.step()

            bn = len(
                idx
            )

            total_loss += float(
                loss.detach()
                .cpu()
                .item()
            ) * bn

            seen += bn

        # Validation raw head with alpha=1 for checkpoint selection.
        _, _, corr_val = predict_hdg_raw_correction(
            torch=torch,
            model=model,
            Z=Z_val,
            device=device,
        )

        q_val = apply_hdg_alpha(
            ridge_q_z_val,
            corr_val,
            np.ones(
                5,
                dtype=np.float32,
            ),
            old_ridge_model,
        )

        score = float(
            np.mean(
                hdg_mae_per_horizon(
                    y_q_raw_val,
                    q_val,
                )
            )
        )

        improved = (
            score
            < best_score
            - 1e-8
        )

        if improved:
            best_score = score
            best_epoch = epoch
            best_state = state_dict_cpu(
                model
            )
            patience = 0
        else:
            patience += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": (
                    total_loss
                    / max(
                        seen,
                        1,
                    )
                ),
                "validation_HDG_MAE_alpha1_deg": score,
                "best_validation_HDG_MAE_alpha1_deg": best_score,
            }
        )

        if (
            epoch <= 3
            or epoch % 10 == 0
            or improved
        ):
            log(
                f"      HDG epoch={epoch:03d} | "
                f"loss={total_loss/max(seen,1):.6f} | "
                f"HDG_MAE(alpha=1)={score:.6f}"
                + (
                    " *"
                    if improved
                    else ""
                )
            )

        if (
            patience
            >= HDG_PATIENCE
        ):
            break

    model.load_state_dict(
        best_state,
        strict=True,
    )

    _, gate_val, corr_val = predict_hdg_raw_correction(
        torch=torch,
        model=model,
        Z=Z_val,
        device=device,
    )

    alpha, alpha_score = select_alpha_per_horizon(
        y_q_raw_val=y_q_raw_val,
        ridge_q_z_val=ridge_q_z_val,
        raw_corr_z_val=corr_val,
        old_ridge_model=old_ridge_model,
    )

    final_q_val = apply_hdg_alpha(
        ridge_q_z_val,
        corr_val,
        alpha,
        old_ridge_model,
    )

    return {
        "model": model,
        "device": device,
        "best_epoch": best_epoch,
        "best_alpha1_HDG_MAE": best_score,
        "alpha": alpha,
        "alpha_hdg_mae": alpha_score,
        "final_q_val": final_q_val,
        "gate_val": gate_val,
        "corr_val": corr_val,
        "history": pd.DataFrame(
            history
        ),
        "parameters": params,
    }


def mean_stage_metrics(df):
    cols = [
        "Vessel_vector_RMSE_mps",
        "SOG_RMSE_mps",
        "HDG_MAE_deg",
        "HDG_RMSE_deg",
        "AppVector_RMSE_mps",
        "AWS_RMSE_mps",
        "AWA_MAE_deg",
        "AWA_RMSE_deg",
    ]

    return {
        c: float(
            df[c].mean()
        )
        for c in cols
    }


def aggregate_rows(df):
    rows = []

    numeric_cols = [
        c
        for c in df.columns
        if (
            c not in {
                "seed",
                "model",
            }
            and np.issubdtype(
                df[c].dtype,
                np.number,
            )
        )
    ]

    for c in numeric_cols:
        vals = df[c].to_numpy(
            dtype=np.float64
        )

        rows.append(
            {
                "metric": c,
                "mean": float(
                    np.mean(vals)
                ),
                "std": float(
                    np.std(
                        vals,
                        ddof=1,
                    )
                ),
                "min": float(
                    np.min(vals)
                ),
                "max": float(
                    np.max(vals)
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
    )

    parser.add_argument(
        "--stage15a-truewind-dir",
        type=Path,
        default=DEFAULT_STAGE15A_TRUEWIND_DIR,
    )

    parser.add_argument(
        "--src-dir",
        type=Path,
        default=DEFAULT_SRC_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    out = args.output_dir

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Reuse validated 15E implementation.
    emod = load_module(
        "stage15e_base_for_15eh",
        args.src_dir
        / "15E_hybrid_NewWind_OldCompactVessel_5seed_validation.py",
    )

    bmod = load_module(
        "stage15b_for_15eh",
        args.src_dir
        / "15B_train_1min_TrueWind_Residual_Correction_v2.py",
    )

    cmod = load_module(
        "stage15c_geometry_for_15eh",
        args.src_dir
        / "15C_train_1min_VesselHDG_Residual_ApparentWind.py",
    )

    bmod.ALPHA_MAX = 1.0

    torch, nn = emod.import_torch()

    emod.configure_torch(
        torch
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    log("=" * 158)
    log(
        "15E-H — NEW WIND + OLD COMPACT VESSEL + FROZEN-FEATURE HDG GATED RESIDUAL"
    )
    log("=" * 158)
    log(
        "TRAIN/VALIDATION ONLY — INTERNAL TEST IS NOT LOADED."
    )
    log(
        f"device = {device}"
    )
    log(
        f"seeds = {SEEDS}"
    )
    log(
        f"additional HDG-head parameters = "
        f"{EXPECTED_HDG_HEAD_PARAMETERS:,}"
    )

    # -----------------------------------------------------------------
    # Data.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 1/8] Loading TRAIN/VALIDATION."
    )

    train = emod.load_split(
        args.dataset_dir,
        "train",
    )

    val = emod.load_split(
        args.dataset_dir,
        "validation",
    )

    tw_train, tw_val = (
        emod.load_truewind_anchor(
            args.stage15a_truewind_dir
        )
    )

    if (
        len(train["X"])
        != len(tw_train["y"])
    ):
        raise RuntimeError(
            "TRAIN alignment mismatch."
        )

    if (
        len(val["X"])
        != len(tw_val["y"])
    ):
        raise RuntimeError(
            "VALIDATION alignment mismatch."
        )

    log(
        f"  TRAIN      = {len(train['X']):,}"
    )
    log(
        f"  VALIDATION = {len(val['X']):,}"
    )

    # -----------------------------------------------------------------
    # Apparent reference.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 2/8] Apparent-wind convention audit."
    )

    train_for_c = {
        "X": train["X"],
        "y": train["y"],
        "apparent_candidate": train["apparent"],
        "apparent_key": train["apparent_key"],
    }

    val_for_c = {
        "X": val["X"],
        "y": val["y"],
        "apparent_candidate": val["apparent"],
        "apparent_key": val["apparent_key"],
    }

    audit = cmod.audit_apparent_reference(
        train_for_c
    )

    app_ref_train = (
        cmod.canonical_apparent_reference(
            train_for_c,
            audit,
        )
    )

    app_ref_val = (
        cmod.canonical_apparent_reference(
            val_for_c,
            audit,
        )
    )

    log(
        f"  source     = {audit['source']}"
    )
    log(
        f"  key        = {audit['key']}"
    )
    log(
        f"  convention = {audit['convention']}"
    )
    log(
        f"  TRAIN audit RMSE = "
        f"{audit['train_audit_rmse']:.9f}"
    )

    # -----------------------------------------------------------------
    # Old joint Ridge.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 3/8] Rebuilding old joint Ridge alpha=100."
    )

    old_ridge = emod.fit_old_joint_ridge(
        train["X"],
        train["y"],
    )

    ridge_z_train, ridge_raw_train = (
        emod.old_ridge_predict(
            old_ridge,
            train["X"],
        )
    )

    ridge_z_val, ridge_raw_val = (
        emod.old_ridge_predict(
            old_ridge,
            val["X"],
        )
    )

    y_z_train = emod.target_to_old_z(
        old_ridge,
        train["y"],
    )

    # Heading targets / anchors.
    y_q_raw_train = train["y"][:, :, 4:6]
    y_q_raw_val = val["y"][:, :, 4:6]

    y_q_z_train = y_z_train[:, :, 4:6]
    ridge_q_z_train = ridge_z_train[:, :, 4:6]
    ridge_q_z_val = ridge_z_val[:, :, 4:6]

    # Stage15B features once.
    feat_train = bmod.build_features(
        train["X"]
    )

    feat_val = bmod.build_features(
        val["X"]
    )

    y_vh_val = val["y"][:, :, 2:6]

    per_seed_rows = []
    horizon_frames = []
    alpha_rows = []
    head_param_check = None

    # -----------------------------------------------------------------
    # Five seeds.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 4/8] Five-seed 15E-H training."
    )

    for seed_idx, seed in enumerate(
        SEEDS,
        start=1,
    ):
        log("")
        log(
            "-" * 158
        )
        log(
            f"[SEED {seed_idx}/5] {seed}"
        )
        log(
            "-" * 158
        )

        # =============================================================
        # NEW Stage15B wind.
        # =============================================================
        log(
            "  [15B WIND] Training frozen 2-layer true-wind residual."
        )

        b_bundle = bmod.train_model(
            torch=torch,
            nn=nn,
            X_train_features=feat_train,
            residual_train=tw_train["residual"],
            X_val_features=feat_val,
            y_val=tw_val["y"],
            ridge_val=tw_val["ridge"],
            seed=seed,
            max_epochs=bmod.MAX_EPOCHS,
            fixed_epochs=None,
            verbose=True,
        )

        b_alpha = np.asarray(
            b_bundle["best_alpha"],
            dtype=np.float32,
        )

        raw_tw_val = bmod.predict_raw(
            torch,
            b_bundle["model"],
            bmod.transform_features(
                feat_val,
                b_bundle["scaler"],
            ),
            b_bundle["scaler"],
            b_bundle["device"],
        )

        stage2_wind_val = (
            bmod.apply_alpha(
                tw_val["ridge"],
                raw_tw_val,
                b_alpha,
            )
        )

        # =============================================================
        # OLD vessel branch.
        # =============================================================
        log(
            "  [OLD VESSEL] Training documented single-GRU64 gated residual."
        )

        old_bundle = emod.train_old_vessel_model(
            torch=torch,
            nn=nn,
            cmod=cmod,
            seed=seed,
            X_train=train["X"],
            y_train_raw=train["y"],
            y_train_z=y_z_train,
            ridge_z_train=ridge_z_train,
            ridge_raw_train=ridge_raw_train,
            app_ref_train=app_ref_train,
            X_val=val["X"],
            y_val_raw=val["y"],
            ridge_z_val=ridge_z_val,
            ridge_raw_val=ridge_raw_val,
            app_ref_val=app_ref_val,
            old_ridge_model=old_ridge,
        )

        old_vessel_full_val, corr_v_val, gate_v_val = (
            emod.predict_old_vessel(
                torch=torch,
                model=old_bundle["model"],
                X=val["X"],
                ridge_z=ridge_z_val,
                ridge_raw=ridge_raw_val,
                old_ridge_model=old_ridge,
                device=old_bundle["device"],
            )
        )

        # =============================================================
        # Freeze old model and extract features.
        # =============================================================
        log(
            "  [HDG] Freezing old vessel network and extracting 64-d context."
        )

        for p in old_bundle["model"].parameters():
            p.requires_grad_(False)

        Z_train = extract_old_fusion_features(
            torch=torch,
            model=old_bundle["model"],
            X=train["X"],
            ridge_z=ridge_z_train,
            device=old_bundle["device"],
        )

        Z_val = extract_old_fusion_features(
            torch=torch,
            model=old_bundle["model"],
            X=val["X"],
            ridge_z=ridge_z_val,
            device=old_bundle["device"],
        )

        # =============================================================
        # Heading-only head.
        # =============================================================
        log(
            "  [HDG] Training tiny gated residual head; vessel weights frozen."
        )

        h_bundle = train_hdg_head(
            torch=torch,
            nn=nn,
            seed=seed,
            Z_train=Z_train,
            ridge_q_z_train=ridge_q_z_train,
            y_q_z_train=y_q_z_train,
            Z_val=Z_val,
            ridge_q_z_val=ridge_q_z_val,
            y_q_raw_val=y_q_raw_val,
            old_ridge_model=old_ridge,
        )

        head_param_check = int(
            h_bundle["parameters"]
        )

        alpha_h = np.asarray(
            h_bundle["alpha"],
            dtype=np.float32,
        )

        final_q_val = np.asarray(
            h_bundle["final_q_val"],
            dtype=np.float32,
        )

        # =============================================================
        # 15E baseline: same wind/vessel, Ridge heading.
        # =============================================================
        vh_15e = old_vessel_full_val[
            :,
            :,
            2:6
        ].copy()

        # Old vessel output has Ridge heading unchanged.
        metrics_15e = (
            cmod.metrics_per_horizon(
                y_vh_val,
                vh_15e,
                stage2_wind_val,
                app_ref_val,
                "validation",
                "15E",
            )
        )

        metrics_15e.insert(
            0,
            "seed",
            seed,
        )

        horizon_frames.append(
            metrics_15e
        )

        # =============================================================
        # 15E-H: ONLY replace heading pair.
        # =============================================================
        vh_15eh = vh_15e.copy()

        vh_15eh[
            :,
            :,
            2:4
        ] = final_q_val

        metrics_15eh = (
            cmod.metrics_per_horizon(
                y_vh_val,
                vh_15eh,
                stage2_wind_val,
                app_ref_val,
                "validation",
                "15E-H",
            )
        )

        metrics_15eh.insert(
            0,
            "seed",
            seed,
        )

        horizon_frames.append(
            metrics_15eh
        )

        mean_15e = mean_stage_metrics(
            metrics_15e
        )

        mean_15eh = mean_stage_metrics(
            metrics_15eh
        )

        # Hard scientific invariance checks.
        invariant_cols = [
            "Vessel_vector_RMSE_mps",
            "SOG_RMSE_mps",
            "AppVector_RMSE_mps",
            "AWS_RMSE_mps",
        ]

        for col in invariant_cols:
            if not np.isclose(
                mean_15e[col],
                mean_15eh[col],
                rtol=0.0,
                atol=1e-9,
            ):
                raise RuntimeError(
                    f"Invariance failure for {col}: "
                    f"15E={mean_15e[col]}, "
                    f"15E-H={mean_15eh[col]}"
                )

        for j, h in enumerate(
            HORIZONS
        ):
            alpha_rows.append(
                {
                    "seed": seed,
                    "horizon_min": h,
                    "alpha_HDG": float(
                        alpha_h[j]
                    ),
                    "validation_HDG_MAE_after_alpha_deg": float(
                        h_bundle[
                            "alpha_hdg_mae"
                        ][j]
                    ),
                    "mean_HDG_gate": float(
                        np.mean(
                            h_bundle[
                                "gate_val"
                            ][:, j, :]
                        )
                    ),
                }
            )

        row_15e = {
            "seed": seed,
            "model": "15E",
            "Stage15B_best_epoch": int(
                b_bundle["best_epoch"]
            ),
            "OldVessel_best_epoch": int(
                old_bundle["best_epoch"]
            ),
            "HDG_best_epoch": 0,
            "HDG_head_parameters": 0,
            **mean_15e,
        }

        row_15eh = {
            "seed": seed,
            "model": "15E-H",
            "Stage15B_best_epoch": int(
                b_bundle["best_epoch"]
            ),
            "OldVessel_best_epoch": int(
                old_bundle["best_epoch"]
            ),
            "HDG_best_epoch": int(
                h_bundle["best_epoch"]
            ),
            "HDG_head_parameters": int(
                h_bundle["parameters"]
            ),
            **mean_15eh,
        }

        per_seed_rows.extend(
            [
                row_15e,
                row_15eh,
            ]
        )

        log("")
        log(
            "  [FROZEN HDG ALPHA]"
        )

        for j, h in enumerate(
            HORIZONS
        ):
            log(
                f"      {h:2d} min | "
                f"alpha_HDG={alpha_h[j]:.6f} | "
                f"mean_gate={np.mean(h_bundle['gate_val'][:,j,:]):.6f}"
            )

        log("")
        log(
            "  [SEED SUMMARY]"
        )

        log(
            f"      15E   HDG MAE = "
            f"{mean_15e['HDG_MAE_deg']:.6f} deg"
        )

        log(
            f"      15E-H HDG MAE = "
            f"{mean_15eh['HDG_MAE_deg']:.6f} deg"
        )

        log(
            f"      15E   AWA MAE = "
            f"{mean_15e['AWA_MAE_deg']:.6f} deg"
        )

        log(
            f"      15E-H AWA MAE = "
            f"{mean_15eh['AWA_MAE_deg']:.6f} deg"
        )

        log(
            f"      AW vector unchanged = "
            f"{mean_15eh['AppVector_RMSE_mps']:.6f}"
        )

        log(
            f"      AWS unchanged       = "
            f"{mean_15eh['AWS_RMSE_mps']:.6f}"
        )

        # Save seed diagnostics.
        np.savez_compressed(
            out
            / f"15E_H_seed_{seed}_validation_predictions.npz",
            y_raw=val["y"].astype(
                np.float32
            ),
            stage15b_truewind=stage2_wind_val.astype(
                np.float32
            ),
            old_vessel_hdg=vh_15e.astype(
                np.float32
            ),
            final_hdg_q=final_q_val.astype(
                np.float32
            ),
            alpha_hdg=alpha_h.astype(
                np.float32
            ),
            hdg_gate=h_bundle[
                "gate_val"
            ].astype(
                np.float32
            ),
            hdg_correction_z=h_bundle[
                "corr_val"
            ].astype(
                np.float32
            ),
            horizons_min=np.asarray(
                HORIZONS,
                dtype=np.int64,
            ),
        )

        h_bundle["history"].to_csv(
            out
            / f"15E_H_seed_{seed}_HDG_training_history.csv",
            index=False,
            encoding="utf-8-sig",
        )

        del Z_train
        del Z_val
        del h_bundle
        del old_bundle
        del b_bundle

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Aggregate.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 5/8] Aggregating five seeds."
    )

    per_seed_df = pd.DataFrame(
        per_seed_rows
    )

    horizon_df = pd.concat(
        horizon_frames,
        ignore_index=True,
    )

    alpha_df = pd.DataFrame(
        alpha_rows
    )

    df_15e = (
        per_seed_df.loc[
            per_seed_df["model"]
            == "15E"
        ]
        .reset_index(
            drop=True
        )
    )

    df_15eh = (
        per_seed_df.loc[
            per_seed_df["model"]
            == "15E-H"
        ]
        .reset_index(
            drop=True
        )
    )

    agg_15e = aggregate_rows(
        df_15e
    )

    agg_15e.insert(
        0,
        "model",
        "15E",
    )

    agg_15eh = aggregate_rows(
        df_15eh
    )

    agg_15eh.insert(
        0,
        "model",
        "15E-H",
    )

    aggregate_df = pd.concat(
        [
            agg_15e,
            agg_15eh,
        ],
        ignore_index=True,
    )

    # Paired 15E -> 15E-H effects.
    paired_rows = []

    for metric in [
        "HDG_MAE_deg",
        "HDG_RMSE_deg",
        "AWA_MAE_deg",
        "AWA_RMSE_deg",
        "AppVector_RMSE_mps",
        "AWS_RMSE_mps",
        "Vessel_vector_RMSE_mps",
    ]:
        a = df_15e[
            metric
        ].to_numpy(
            dtype=np.float64
        )

        b = df_15eh[
            metric
        ].to_numpy(
            dtype=np.float64
        )

        diff = a - b

        paired_rows.append(
            {
                "metric": metric,
                "15E_mean": float(
                    np.mean(a)
                ),
                "15E_H_mean": float(
                    np.mean(b)
                ),
                "mean_reduction_15E_minus_15E_H": float(
                    np.mean(diff)
                ),
                "improvement_pct": float(
                    (
                        np.mean(a)
                        - np.mean(b)
                    )
                    / np.mean(a)
                    * 100.0
                )
                if abs(
                    np.mean(a)
                ) > EPS
                else 0.0,
                "paired_diff_std": float(
                    np.std(
                        diff,
                        ddof=1,
                    )
                ),
            }
        )

    paired_df = pd.DataFrame(
        paired_rows
    )

    # -----------------------------------------------------------------
    # Save.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 6/8] Saving outputs."
    )

    per_seed_df.to_csv(
        out
        / "15E_H_per_seed_validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    aggregate_df.to_csv(
        out
        / "15E_H_validation_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired_df.to_csv(
        out
        / "15E_H_paired_effect.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_df.to_csv(
        out
        / "15E_H_per_horizon_all_seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    alpha_df.to_csv(
        out
        / "15E_H_alpha_gate_all_seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Print final.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 7/8] Final five-seed results."
    )

    def mstd(df, col):
        vals = df[
            col
        ].to_numpy(
            dtype=np.float64
        )

        return (
            float(
                np.mean(vals)
            ),
            float(
                np.std(
                    vals,
                    ddof=1,
                )
            ),
        )

    metrics_to_print = [
        "HDG_MAE_deg",
        "HDG_RMSE_deg",
        "AWA_MAE_deg",
        "AWA_RMSE_deg",
        "AppVector_RMSE_mps",
        "AWS_RMSE_mps",
        "Vessel_vector_RMSE_mps",
        "SOG_RMSE_mps",
    ]

    log("")
    log("=" * 158)
    log(
        "15E-H FINAL — HEADING RESIDUAL CONFIRMATION, SD1090 VALIDATION, FIVE SEEDS"
    )
    log("=" * 158)

    for metric in metrics_to_print:
        a = mstd(
            df_15e,
            metric,
        )

        b = mstd(
            df_15eh,
            metric,
        )

        log(
            f"{metric:28s} | "
            f"15E={a[0]:.6f} ± {a[1]:.6f} | "
            f"15E-H={b[0]:.6f} ± {b[1]:.6f}"
        )

    log("")
    log(
        "[PAIRED 15E -> 15E-H EFFECT]"
    )

    log(
        paired_df.to_string(
            index=False
        )
    )

    log("")
    log(
        f"Additional HDG head parameters = "
        f"{head_param_check:,}"
    )

    log(
        f"15E neural parameters          = "
        f"{emod.HYBRID_NEURAL_PARAMETERS:,}"
    )

    log(
        f"15E-H neural parameters        = "
        f"{emod.HYBRID_NEURAL_PARAMETERS + head_param_check:,}"
    )

    # -----------------------------------------------------------------
    # Report.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 8/8] Writing report."
    )

    report = {
        "experiment": "15E-H",
        "internal_test_loaded": False,
        "purpose": (
            "Add a heading-only gated residual head to 15E while "
            "freezing true wind and vessel motion."
        ),
        "scientific_invariance": {
            "Earth_frame_apparent_wind_depends_on_HDG": False,
            "AWS_depends_on_HDG": False,
            "AWA_depends_on_HDG": True,
            "expected_unchanged_metrics": [
                "Vessel_vector_RMSE_mps",
                "SOG_RMSE_mps",
                "AppVector_RMSE_mps",
                "AWS_RMSE_mps",
            ],
        },
        "HDG_head": {
            "input_feature": (
                "frozen old-vessel 64-d fusion representation"
            ),
            "residual_outputs": 10,
            "gate_outputs": 10,
            "parameters": head_param_check,
            "global_alpha_range": [
                ALPHA_MIN,
                ALPHA_MAX,
            ],
            "alpha_grid_size": ALPHA_GRID_SIZE,
            "alpha_selection_metric": (
                "validation HDG MAE only"
            ),
            "AWA_used_for_alpha_selection": False,
        },
        "seeds": SEEDS,
        "paired_effect": paired_df.to_dict(
            orient="records"
        ),
        "alpha_gate": alpha_df.to_dict(
            orient="records"
        ),
    }

    save_json(
        out
        / "15E_H_REPORT.json",
        report,
    )

    with (
        out
        / "15E_H_REPORT.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "15E-H — HEADING-ONLY GATED RESIDUAL\n"
        )
        f.write(
            "=" * 140
            + "\n\n"
        )
        f.write(
            "Earth-frame apparent wind and AWS do not depend on heading.\n"
        )
        f.write(
            "Only HDG and AWA are allowed to change.\n\n"
        )

        for metric in metrics_to_print:
            a = mstd(
                df_15e,
                metric,
            )

            b = mstd(
                df_15eh,
                metric,
            )

            f.write(
                f"{metric:28s} | "
                f"15E={a[0]:.8f} ± {a[1]:.8f} | "
                f"15E-H={b[0]:.8f} ± {b[1]:.8f}\n"
            )

        f.write(
            "\nPAIRED EFFECT\n"
        )

        f.write(
            "-" * 140
            + "\n"
        )

        f.write(
            paired_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nALPHA / GATE\n"
        )

        f.write(
            "-" * 140
            + "\n"
        )

        f.write(
            alpha_df.to_string(
                index=False
            )
        )

    log("")
    log(
        f"[SAVED] {out}"
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
        sys.exit(1)
