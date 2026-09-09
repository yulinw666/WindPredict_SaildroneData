# -*- coding: utf-8 -*-
"""
15B_train_1min_TrueWind_Residual_Correction.py

STEP 2 of the staged 1-min apparent-wind forecasting method.\n\nv2 compatibility update:\n    - defaults to frozen 15A-R v0.2\n    - accepts 15A_R_oof_folds.csv\n    - still accepts legacy 15A_oof_folds.csv

Prerequisite:
    Stage 15A dual Ridge anchors are complete.

This stage ONLY trains the TRUE-WIND residual branch.

Frozen from Stage 15A:
    TrueWind Ridge:
        [U,V,T,RH] x 60 -> future [U,V] at 1/2/3/5/10 min

Training target:
    r_wind^OOF = y_true_wind - y_ridge^OOF

Residual network input:
    ONLY information derived from [U,V,T,RH] history.

Per-step residual features:
    U, V, T, RH,
    dU, dV,
    ddU, ddV,
    WS, dWS

Residual network:
    2-layer GRU
    hidden = 48
    shared MLP = 48
    output = 5 horizons x 2 components

The final output layer is ZERO initialized, therefore epoch 0 is exactly Ridge.

FINAL TRUE-WIND:
    W_stage2(h,c)
        = W_ridge(h,c)
        + alpha(h,c) * DeltaW_NN(h,c)

where alpha is validation-calibrated analytically and independently for:
    horizon x component
    = 5 x 2 = 10 shrinkage coefficients

alpha range:
    [0, 0.75]

If a residual direction is not useful on validation:
    alpha = 0
and that horizon/component falls back exactly to Ridge.

MODEL SELECTION:
    TRAIN residual network only on the frozen outer TRAIN split.
    VALIDATION is used for:
        - early stopping
        - analytical alpha calibration
    TEST is not used until everything is frozen.

TRAIN OOF STAGE-2 PREDICTIONS:
    Stage 15C must not use an in-sample Stage-2 wind prediction as upstream
    training input.

Therefore after epoch/alpha freeze, this script also trains 5 cross-fitted
residual models using the SAME Stage15A purged folds and produces:

    train_stage2_truewind_oof

For each held-out block:
    - residual NN does not see that block
    - Ridge anchor is Stage15A true_wind_ridge_oof
    - frozen alpha matrix is applied

This is the leakage-safe true-wind prediction that Stage15C must use when
training vessel/HDG residual + apparent-wind coupling.

OUTPUTS
-------
15B_development_history.csv
15B_frozen_truewind_residual.json
15B_truewind_residual.pt

15B_validation_predictions.npz
15B_test_predictions.npz
15B_train_stage2_oof_predictions.npz

15B_metrics_per_horizon.csv
15B_metrics_summary.csv
15B_REPORT.txt
15B_REPORT.json

Recommended PowerShell
----------------------
python "D:\\project\\WindPredict_SaildroneData\\src\\15B_train_1min_TrueWind_Residual_Correction.py" ^
  --dataset-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\SD1090_TPOS2024_JointForecasting_v0_1" ^
  --stage15a-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15A_R_1min_Dual_Ridge_Anchors_v0_2" ^
  --output-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15B_1min_TrueWind_Residual_v0_1"

PowerShell one-line:
python "D:\\project\\WindPredict_SaildroneData\\src\\15B_train_1min_TrueWind_Residual_Correction.py" --dataset-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\SD1090_TPOS2024_JointForecasting_v0_1" --stage15a-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15A_R_1min_Dual_Ridge_Anchors_v0_2" --output-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15B_1min_TrueWind_Residual_v0_1"
"""

from __future__ import annotations

import argparse
import contextlib
import gc
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
    ROOT
    / "data"
    / "forecasting"
    / "SD1090_TPOS2024_JointForecasting_v0_1"
)

DEFAULT_STAGE15A_DIR = (
    ROOT
    / "data"
    / "forecasting"
    / "15A_R_1min_Dual_Ridge_Anchors_v0_2"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "data"
    / "forecasting"
    / "15B_1min_TrueWind_Residual_v0_1"
)

HORIZONS = [1, 2, 3, 5, 10]

SEED = 500043

GRU_HIDDEN = 48
GRU_LAYERS = 2
SHARED_HIDDEN = 48
DROPOUT = 0.10

LR = 8e-4
WEIGHT_DECAY = 2e-5
BATCH_SIZE = 512
MAX_EPOCHS = 120
EARLY_STOP_PATIENCE = 18
GRAD_CLIP = 1.0

ALPHA_MAX = 0.75
EPS = 1e-12

OOF_FOLDS = 5
OOF_PURGE = 10


# =============================================================================
# General
# =============================================================================

def log(msg=""):
    print(msg, flush=True)


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


def import_torch():
    import torch
    import torch.nn as nn
    return torch, nn


def seed_everything(torch, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch(torch):
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def amp_context(torch, device):
    if device.type != "cuda":
        return contextlib.nullcontext()

    try:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        )
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def create_scaler(torch, device):
    if device.type != "cuda":
        return None

    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        try:
            return torch.cuda.amp.GradScaler(enabled=True)
        except Exception:
            return None


def state_dict_cpu(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def count_parameters(model):
    return int(
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )


# =============================================================================
# Dataset / Stage15A
# =============================================================================

def load_dataset_split(dataset_dir: Path, split: str):
    path = dataset_dir / f"{split}.npz"

    with np.load(path, allow_pickle=False) as z:
        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y_raw"], dtype=np.float32)

    if X.ndim != 3 or X.shape[1:] != (60, 11):
        raise RuntimeError(
            f"{split}: expected X=(N,60,11), got {X.shape}"
        )

    if y.ndim != 3 or y.shape[1:] != (5, 6):
        raise RuntimeError(
            f"{split}: expected y=(N,5,6), got {y.shape}"
        )

    return {
        "X": X,
        "y": y,
    }


def load_stage15a(stage15a_dir: Path):
    train_path = stage15a_dir / "15A_train_oof_anchors.npz"
    val_path = stage15a_dir / "15A_validation_anchor_predictions.npz"
    test_path = stage15a_dir / "15A_test_anchor_predictions.npz"
    fold_candidates = [
        stage15a_dir / "15A_R_oof_folds.csv",
        stage15a_dir / "15A_oof_folds.csv",
    ]

    folds_path = None
    for candidate in fold_candidates:
        if candidate.exists():
            folds_path = candidate
            break

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Stage15A prerequisite missing: {p}"
            )

    if folds_path is None:
        raise FileNotFoundError(
            "No compatible OOF fold file found. Tried: "
            + ", ".join(str(p) for p in fold_candidates)
        )

    with np.load(train_path, allow_pickle=False) as z:
        train = {
            "y": np.asarray(z["true_wind_y"], dtype=np.float32),
            "ridge": np.asarray(z["true_wind_ridge_oof"], dtype=np.float32),
            "residual": np.asarray(z["true_wind_residual_oof"], dtype=np.float32),
        }

    with np.load(val_path, allow_pickle=False) as z:
        val = {
            "y": np.asarray(z["true_wind_y"], dtype=np.float32),
            "ridge": np.asarray(z["true_wind_ridge"], dtype=np.float32),
            "residual": np.asarray(z["true_wind_residual"], dtype=np.float32),
        }

    with np.load(test_path, allow_pickle=False) as z:
        test = {
            "y": np.asarray(z["true_wind_y"], dtype=np.float32),
            "ridge": np.asarray(z["true_wind_ridge"], dtype=np.float32),
            "residual": np.asarray(z["true_wind_residual"], dtype=np.float32),
        }

    fold_df = pd.read_csv(folds_path)

    return train, val, test, fold_df


# =============================================================================
# Residual features
# =============================================================================

def build_features(X11):
    """
    Input to Stage15B uses ONLY U,V,T,RH-derived information.

    Dataset X is already train-standardized. Since deterministic differences of
    standardized U/V are still valid temporal features, this is safe.

    seq channels:
        U, V, T, RH,
        dU, dV,
        ddU, ddV,
        WS_proxy, dWS_proxy

    Note:
        WS_proxy is computed in dataset-standardized U/V space.
        It is a deterministic nonlinear feature for the NN, not a reported
        physical wind speed.
    """
    X4 = np.asarray(
        X11[:, :, 0:4],
        dtype=np.float32,
    )

    U = X4[:, :, 0]
    V = X4[:, :, 1]
    T = X4[:, :, 2]
    RH = X4[:, :, 3]

    dU = np.zeros_like(U)
    dV = np.zeros_like(V)

    dU[:, 1:] = U[:, 1:] - U[:, :-1]
    dV[:, 1:] = V[:, 1:] - V[:, :-1]

    ddU = np.zeros_like(U)
    ddV = np.zeros_like(V)

    ddU[:, 2:] = dU[:, 2:] - dU[:, 1:-1]
    ddV[:, 2:] = dV[:, 2:] - dV[:, 1:-1]

    wsp = np.sqrt(U ** 2 + V ** 2)

    dwsp = np.zeros_like(wsp)
    dwsp[:, 1:] = wsp[:, 1:] - wsp[:, :-1]

    seq = np.stack(
        [
            U,
            V,
            T,
            RH,
            dU,
            dV,
            ddU,
            ddV,
            wsp,
            dwsp,
        ],
        axis=-1,
    ).astype(np.float32)

    summary = np.stack(
        [
            U[:, -1],
            V[:, -1],
            dU[:, -1],
            dV[:, -1],
            np.std(U, axis=1),
            np.std(V, axis=1),
            np.std(wsp, axis=1),
            U[:, -1] - U[:, 0],
            V[:, -1] - V[:, 0],
            wsp[:, -1] - wsp[:, 0],
        ],
        axis=-1,
    ).astype(np.float32)

    if not (
        np.isfinite(seq).all()
        and np.isfinite(summary).all()
    ):
        raise RuntimeError("Nonfinite residual features.")

    return {
        "seq": seq,
        "summary": summary,
    }


def fit_feature_scaler(features, residual):
    seq = np.asarray(features["seq"], dtype=np.float64)
    summary = np.asarray(features["summary"], dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)

    seq_mean = np.mean(seq, axis=(0, 1))
    seq_std = np.std(seq, axis=(0, 1), ddof=0)
    seq_std = np.where(seq_std < 1e-8, 1.0, seq_std)

    sum_mean = np.mean(summary, axis=0)
    sum_std = np.std(summary, axis=0, ddof=0)
    sum_std = np.where(sum_std < 1e-8, 1.0, sum_std)

    residual_std = np.std(residual, axis=0, ddof=0)
    residual_std = np.where(residual_std < 1e-6, 1.0, residual_std)

    return {
        "seq_mean": seq_mean.astype(np.float32),
        "seq_std": seq_std.astype(np.float32),
        "summary_mean": sum_mean.astype(np.float32),
        "summary_std": sum_std.astype(np.float32),
        "residual_std": residual_std.astype(np.float32),
    }


def transform_features(features, scaler):
    seq = (
        (
            features["seq"]
            - scaler["seq_mean"][None, None, :]
        )
        / scaler["seq_std"][None, None, :]
    ).astype(np.float32)

    summary = (
        (
            features["summary"]
            - scaler["summary_mean"][None, :]
        )
        / scaler["summary_std"][None, :]
    ).astype(np.float32)

    return {
        "seq": seq,
        "summary": summary,
    }


def normalize_residual_target(residual, scaler):
    return (
        np.asarray(residual, dtype=np.float32)
        / scaler["residual_std"][None, :, :]
    ).astype(np.float32)


# =============================================================================
# Model
# =============================================================================

def build_model_class(torch, nn):
    class TrueWindResidualGRU(nn.Module):
        def __init__(self):
            super().__init__()

            self.gru = nn.GRU(
                input_size=10,
                hidden_size=GRU_HIDDEN,
                num_layers=GRU_LAYERS,
                batch_first=True,
                dropout=DROPOUT if GRU_LAYERS > 1 else 0.0,
            )

            self.shared = nn.Sequential(
                nn.Linear(
                    GRU_HIDDEN + 10,
                    SHARED_HIDDEN,
                ),
                nn.GELU(),
                nn.Dropout(DROPOUT),
            )

            self.out = nn.Linear(
                SHARED_HIDDEN,
                len(HORIZONS) * 2,
            )

            # Epoch 0 = zero residual = exact Ridge.
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, seq, summary):
            z, _ = self.gru(seq)
            last = z[:, -1, :]

            h = self.shared(
                torch.cat(
                    [last, summary],
                    dim=1,
                )
            )

            out = self.out(h)

            # bounded normalized residual
            out = torch.tanh(out)

            return out.reshape(
                -1,
                len(HORIZONS),
                2,
            )

    return TrueWindResidualGRU


# =============================================================================
# Alpha / metrics
# =============================================================================

def analytic_alpha_matrix(y_true, ridge, raw_residual):
    """
    Independent analytical shrinkage for each horizon/component.
    """
    e = (
        np.asarray(y_true, dtype=np.float64)
        - np.asarray(ridge, dtype=np.float64)
    )

    r = np.asarray(
        raw_residual,
        dtype=np.float64,
    )

    alpha = np.zeros(
        (
            len(HORIZONS),
            2,
        ),
        dtype=np.float32,
    )

    for h in range(len(HORIZONS)):
        for c in range(2):
            ee = e[:, h, c]
            rr = r[:, h, c]

            denom = float(
                np.dot(rr, rr)
            )

            if denom <= EPS:
                a = 0.0
            else:
                a = float(
                    np.dot(ee, rr)
                    / denom
                )

            alpha[h, c] = np.clip(
                a,
                0.0,
                ALPHA_MAX,
            )

    return alpha


def apply_alpha(ridge, raw_residual, alpha):
    return (
        np.asarray(ridge, dtype=np.float32)
        + np.asarray(raw_residual, dtype=np.float32)
        * np.asarray(alpha, dtype=np.float32)[None, :, :]
    ).astype(np.float32)


def wd_from_uv(uv):
    return (
        np.degrees(
            np.arctan2(
                -uv[..., 0],
                -uv[..., 1],
            )
        )
        % 360.0
    )


def circular_diff_deg(a, b):
    return (
        (
            np.asarray(a)
            - np.asarray(b)
            + 180.0
        )
        % 360.0
        - 180.0
    )


def metrics_per_horizon(y_true, y_pred, model_name, split):
    rows = []

    for j, horizon in enumerate(HORIZONS):
        t = np.asarray(y_true[:, j, :], dtype=np.float64)
        p = np.asarray(y_pred[:, j, :], dtype=np.float64)

        e = p - t

        ws_t = np.linalg.norm(t, axis=1)
        ws_p = np.linalg.norm(p, axis=1)

        wd_err = circular_diff_deg(
            wd_from_uv(p),
            wd_from_uv(t),
        )

        rows.append(
            {
                "split": split,
                "model": model_name,
                "horizon_min": horizon,
                "U_RMSE_mps": float(
                    np.sqrt(np.mean(e[:, 0] ** 2))
                ),
                "V_RMSE_mps": float(
                    np.sqrt(np.mean(e[:, 1] ** 2))
                ),
                "vector_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            np.sum(e ** 2, axis=1)
                        )
                    )
                ),
                "WS_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            (ws_p - ws_t) ** 2
                        )
                    )
                ),
                "WD_RMSE_deg": float(
                    np.sqrt(
                        np.mean(
                            wd_err ** 2
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def mean_vector_rmse(y_true, pred):
    m = metrics_per_horizon(
        y_true,
        pred,
        "tmp",
        "tmp",
    )

    return float(
        m["vector_RMSE_mps"].mean()
    )


def residual_corr(y_true, ridge, raw):
    true_res = (
        np.asarray(y_true, dtype=np.float64)
        - np.asarray(ridge, dtype=np.float64)
    )

    raw = np.asarray(raw, dtype=np.float64)

    corr = np.full(
        (
            len(HORIZONS),
            2,
        ),
        np.nan,
        dtype=np.float64,
    )

    for h in range(len(HORIZONS)):
        for c in range(2):
            a = true_res[:, h, c]
            b = raw[:, h, c]

            if (
                np.std(a) > EPS
                and np.std(b) > EPS
            ):
                corr[h, c] = np.corrcoef(
                    a,
                    b,
                )[0, 1]

    return corr


# =============================================================================
# Prediction
# =============================================================================

def predict_raw(
    torch,
    model,
    transformed,
    scaler,
    device,
):
    model.eval()

    outputs = []

    with torch.no_grad():
        for start in range(
            0,
            len(transformed["seq"]),
            BATCH_SIZE,
        ):
            stop = min(
                start + BATCH_SIZE,
                len(transformed["seq"]),
            )

            seq_t = torch.from_numpy(
                transformed["seq"][start:stop]
            ).to(
                device,
                non_blocking=True,
            )

            sum_t = torch.from_numpy(
                transformed["summary"][start:stop]
            ).to(
                device,
                non_blocking=True,
            )

            with amp_context(
                torch,
                device,
            ):
                pred_norm = model(
                    seq_t,
                    sum_t,
                )

            outputs.append(
                pred_norm.detach()
                .float()
                .cpu()
                .numpy()
            )

    pred_norm = np.concatenate(
        outputs,
        axis=0,
    )

    return (
        pred_norm
        * scaler["residual_std"][None, :, :]
    ).astype(np.float32)


# =============================================================================
# Training
# =============================================================================

def train_model(
    *,
    torch,
    nn,
    X_train_features,
    residual_train,
    X_val_features,
    y_val,
    ridge_val,
    seed,
    max_epochs,
    fixed_epochs=None,
    verbose=True,
):
    seed_everything(
        torch,
        seed,
    )

    scaler = fit_feature_scaler(
        X_train_features,
        residual_train,
    )

    tr = transform_features(
        X_train_features,
        scaler,
    )

    va = (
        transform_features(
            X_val_features,
            scaler,
        )
        if X_val_features
        is not None
        else None
    )

    target_norm = normalize_residual_target(
        residual_train,
        scaler,
    )

    Model = build_model_class(
        torch,
        nn,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = Model().to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    grad_scaler = create_scaler(
        torch,
        device,
    )

    n = len(
        target_norm
    )

    if fixed_epochs is not None:
        epochs = int(
            fixed_epochs
        )
        early_stop = False
    else:
        epochs = int(
            max_epochs
        )
        early_stop = True

    best_state = state_dict_cpu(
        model
    )

    best_epoch = 0

    if va is not None:
        zero_raw = np.zeros_like(
            ridge_val,
            dtype=np.float32,
        )

        alpha0 = analytic_alpha_matrix(
            y_val,
            ridge_val,
            zero_raw,
        )

        best_score = mean_vector_rmse(
            y_val,
            apply_alpha(
                ridge_val,
                zero_raw,
                alpha0,
            ),
        )

        best_alpha = alpha0
        best_raw_val = zero_raw
    else:
        best_score = np.inf
        best_alpha = None
        best_raw_val = None

    patience = 0
    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        rng = np.random.default_rng(
            seed
            + epoch
            * 1009
        )

        order = np.arange(
            n
        )

        rng.shuffle(
            order
        )

        running = 0.0
        seen = 0

        for start in range(
            0,
            n,
            BATCH_SIZE,
        ):
            idx = order[
                start:
                start
                + BATCH_SIZE
            ]

            seq_t = torch.from_numpy(
                tr["seq"][idx]
            ).to(
                device,
                non_blocking=True,
            )

            sum_t = torch.from_numpy(
                tr["summary"][idx]
            ).to(
                device,
                non_blocking=True,
            )

            target_t = torch.from_numpy(
                target_norm[idx]
            ).to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with amp_context(
                torch,
                device,
            ):
                pred = model(
                    seq_t,
                    sum_t,
                )

                loss = torch.mean(
                    (
                        pred
                        - target_t
                    )
                    ** 2
                )

            if grad_scaler is not None:
                grad_scaler.scale(
                    loss
                ).backward()

                grad_scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                grad_scaler.step(
                    optimizer
                )

                grad_scaler.update()
            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                optimizer.step()

            bn = len(
                idx
            )

            running += (
                float(
                    loss.detach()
                    .float()
                    .cpu()
                    .item()
                )
                * bn
            )

            seen += bn

        train_loss = (
            running
            / max(
                seen,
                1,
            )
        )

        if va is not None:
            raw_val = predict_raw(
                torch,
                model,
                va,
                scaler,
                device,
            )

            alpha = analytic_alpha_matrix(
                y_val,
                ridge_val,
                raw_val,
            )

            final_val = apply_alpha(
                ridge_val,
                raw_val,
                alpha,
            )

            score = mean_vector_rmse(
                y_val,
                final_val,
            )

            improved = (
                score
                < best_score
                - 1e-7
            )

            if improved:
                best_score = score
                best_epoch = epoch
                best_state = state_dict_cpu(
                    model
                )
                best_alpha = alpha.copy()
                best_raw_val = raw_val.copy()
                patience = 0
            else:
                patience += 1

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_mean_vector_RMSE": score,
                    "best_validation_mean_vector_RMSE": best_score,
                }
            )

            if (
                verbose
                and (
                    epoch <= 3
                    or epoch % 10 == 0
                    or improved
                )
            ):
                log(
                    f"  epoch={epoch:03d} | "
                    f"loss={train_loss:.6f} | "
                    f"val vector={score:.6f}"
                    + (
                        " *"
                        if improved
                        else ""
                    )
                )

            if (
                early_stop
                and patience
                >= EARLY_STOP_PATIENCE
            ):
                break
        else:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                }
            )

            if (
                verbose
                and (
                    epoch <= 3
                    or epoch % 10 == 0
                    or epoch == epochs
                )
            ):
                log(
                    f"  epoch={epoch:03d}/{epochs:03d} | "
                    f"loss={train_loss:.6f}"
                )

    if va is not None:
        model.load_state_dict(
            best_state,
            strict=True,
        )

    return {
        "model": model,
        "scaler": scaler,
        "device": device,
        "best_epoch": (
            best_epoch
            if va is not None
            else epochs
        ),
        "best_alpha": best_alpha,
        "best_score": best_score,
        "best_raw_val": best_raw_val,
        "history": pd.DataFrame(
            history
        ),
    }


# =============================================================================
# Cross-fitted Stage2 train predictions
# =============================================================================

def make_fold_indices(fold_df, n):
    folds = []

    all_idx = np.arange(
        n,
        dtype=np.int64,
    )

    for _, row in fold_df.iterrows():
        a = int(
            row[
                "validation_start_index"
            ]
        )

        b = int(
            row[
                "validation_end_index_exclusive"
            ]
        )

        pa = int(
            row[
                "purge_start_index"
            ]
        )

        pb = int(
            row[
                "purge_end_index_exclusive"
            ]
        )

        va = np.arange(
            a,
            b,
            dtype=np.int64,
        )

        mask = np.ones(
            n,
            dtype=bool,
        )

        mask[
            pa:
            pb
        ] = False

        tr = all_idx[
            mask
        ]

        folds.append(
            {
                "fold": int(
                    row[
                        "fold"
                    ]
                ),
                "train_idx": tr,
                "val_idx": va,
            }
        )

    return folds


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
    )

    parser.add_argument(
        "--stage15a-dir",
        type=Path,
        default=DEFAULT_STAGE15A_DIR,
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

    torch, nn = import_torch()

    configure_torch(
        torch
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    log("=" * 140)
    log(
        "15B — STEP 2: 1-MIN TRUE-WIND RESIDUAL CORRECTION"
    )
    log("=" * 140)
    log(
        "Only [U,V,T,RH] information is used."
    )
    log(
        "Stage15A TrueWind Ridge is frozen."
    )
    log(
        f"device = {device}"
    )

    # -----------------------------------------------------------------
    # Data.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 1/7] Loading dataset + Stage15A anchors."
    )

    train_ds = load_dataset_split(
        args.dataset_dir,
        "train",
    )

    val_ds = load_dataset_split(
        args.dataset_dir,
        "validation",
    )

    test_ds = load_dataset_split(
        args.dataset_dir,
        "test",
    )

    a_train, a_val, a_test, fold_df = (
        load_stage15a(
            args.stage15a_dir
        )
    )

    if not (
        len(train_ds["X"])
        == len(a_train["y"])
        and len(val_ds["X"])
        == len(a_val["y"])
        and len(test_ds["X"])
        == len(a_test["y"])
    ):
        raise RuntimeError(
            "Stage15A prediction arrays do not align with dataset splits."
        )

    feat_train = build_features(
        train_ds["X"]
    )

    feat_val = build_features(
        val_ds["X"]
    )

    feat_test = build_features(
        test_ds["X"]
    )

    # -----------------------------------------------------------------
    # Development training.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 2/7] Training Stage15B residual network on OOF Ridge residuals."
    )

    bundle = train_model(
        torch=torch,
        nn=nn,
        X_train_features=feat_train,
        residual_train=a_train["residual"],
        X_val_features=feat_val,
        y_val=a_val["y"],
        ridge_val=a_val["ridge"],
        seed=SEED,
        max_epochs=MAX_EPOCHS,
        fixed_epochs=None,
        verbose=True,
    )

    best_epoch = int(
        bundle[
            "best_epoch"
        ]
    )

    alpha = np.asarray(
        bundle[
            "best_alpha"
        ],
        dtype=np.float32,
    )

    bundle[
        "history"
    ].to_csv(
        out
        / "15B_development_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log(
        f"[FROZEN] best epoch = {best_epoch}"
    )

    log(
        "[FROZEN] alpha matrix [horizon x U/V]:"
    )

    for j, h in enumerate(
        HORIZONS
    ):
        log(
            f"  {h:2d} min: "
            f"alpha_U={alpha[j,0]:.6f} | "
            f"alpha_V={alpha[j,1]:.6f}"
        )

    # -----------------------------------------------------------------
    # Frozen validation/test predictions.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 3/7] Frozen validation/test predictions."
    )

    raw_val = predict_raw(
        torch,
        bundle[
            "model"
        ],
        transform_features(
            feat_val,
            bundle[
                "scaler"
            ],
        ),
        bundle[
            "scaler"
        ],
        device,
    )

    raw_test = predict_raw(
        torch,
        bundle[
            "model"
        ],
        transform_features(
            feat_test,
            bundle[
                "scaler"
            ],
        ),
        bundle[
            "scaler"
        ],
        device,
    )

    final_val = apply_alpha(
        a_val[
            "ridge"
        ],
        raw_val,
        alpha,
    )

    final_test = apply_alpha(
        a_test[
            "ridge"
        ],
        raw_test,
        alpha,
    )

    val_corr = residual_corr(
        a_val[
            "y"
        ],
        a_val[
            "ridge"
        ],
        raw_val,
    )

    test_corr = residual_corr(
        a_test[
            "y"
        ],
        a_test[
            "ridge"
        ],
        raw_test,
    )

    # -----------------------------------------------------------------
    # Cross-fitted Stage2 train output.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 4/7] Cross-fitting Stage2 TRUE-WIND predictions for Stage15C."
    )

    folds = make_fold_indices(
        fold_df,
        len(
            train_ds[
                "X"
            ]
        ),
    )

    train_raw_oof = np.full_like(
        a_train[
            "ridge"
        ],
        np.nan,
        dtype=np.float32,
    )

    train_stage2_oof = np.full_like(
        a_train[
            "ridge"
        ],
        np.nan,
        dtype=np.float32,
    )

    for fold in folds:
        k = fold[
            "fold"
        ]

        tr_idx = fold[
            "train_idx"
        ]

        va_idx = fold[
            "val_idx"
        ]

        log(
            f"  [FOLD {k}] train={len(tr_idx):,} | holdout={len(va_idx):,}"
        )

        tr_feat = {
            "seq": feat_train[
                "seq"
            ][
                tr_idx
            ],
            "summary": feat_train[
                "summary"
            ][
                tr_idx
            ],
        }

        va_feat = {
            "seq": feat_train[
                "seq"
            ][
                va_idx
            ],
            "summary": feat_train[
                "summary"
            ][
                va_idx
            ],
        }

        fold_bundle = train_model(
            torch=torch,
            nn=nn,
            X_train_features=tr_feat,
            residual_train=a_train[
                "residual"
            ][
                tr_idx
            ],
            X_val_features=None,
            y_val=None,
            ridge_val=None,
            seed=SEED + 1000 + k,
            max_epochs=best_epoch,
            fixed_epochs=best_epoch,
            verbose=False,
        )

        fold_raw = predict_raw(
            torch,
            fold_bundle[
                "model"
            ],
            transform_features(
                va_feat,
                fold_bundle[
                    "scaler"
                ],
            ),
            fold_bundle[
                "scaler"
            ],
            fold_bundle[
                "device"
            ],
        )

        train_raw_oof[
            va_idx
        ] = fold_raw

        train_stage2_oof[
            va_idx
        ] = apply_alpha(
            a_train[
                "ridge"
            ][
                va_idx
            ],
            fold_raw,
            alpha,
        )

        del fold_bundle
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not (
        np.isfinite(
            train_raw_oof
        ).all()
        and np.isfinite(
            train_stage2_oof
        ).all()
    ):
        raise RuntimeError(
            "Stage15B cross-fitted train predictions incomplete."
        )

    # -----------------------------------------------------------------
    # Metrics.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 5/7] Metrics."
    )

    metric_frames = []

    for split, yy, ridge, final in [
        (
            "train_OOF",
            a_train[
                "y"
            ],
            a_train[
                "ridge"
            ],
            train_stage2_oof,
        ),
        (
            "validation",
            a_val[
                "y"
            ],
            a_val[
                "ridge"
            ],
            final_val,
        ),
        (
            "test",
            a_test[
                "y"
            ],
            a_test[
                "ridge"
            ],
            final_test,
        ),
    ]:
        metric_frames.append(
            metrics_per_horizon(
                yy,
                ridge,
                "TrueWind-Ridge",
                split,
            )
        )

        metric_frames.append(
            metrics_per_horizon(
                yy,
                final,
                "Stage15B-RidgeResidual",
                split,
            )
        )

    metrics = pd.concat(
        metric_frames,
        ignore_index=True,
    )

    metrics.to_csv(
        out
        / "15B_metrics_per_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = []

    for (
        split,
        model_name,
    ), sub in metrics.groupby(
        [
            "split",
            "model",
        ]
    ):
        summary_rows.append(
            {
                "split": split,
                "model": model_name,
                "mean_U_RMSE_mps": float(
                    sub[
                        "U_RMSE_mps"
                    ].mean()
                ),
                "mean_V_RMSE_mps": float(
                    sub[
                        "V_RMSE_mps"
                    ].mean()
                ),
                "mean_vector_RMSE_mps": float(
                    sub[
                        "vector_RMSE_mps"
                    ].mean()
                ),
                "mean_WS_RMSE_mps": float(
                    sub[
                        "WS_RMSE_mps"
                    ].mean()
                ),
                "mean_WD_RMSE_deg": float(
                    sub[
                        "WD_RMSE_deg"
                    ].mean()
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        out
        / "15B_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Save outputs.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 6/7] Saving Stage15B final true-wind predictions."
    )

    np.savez_compressed(
        out
        / "15B_train_stage2_oof_predictions.npz",
        y_true=a_train[
            "y"
        ].astype(
            np.float32
        ),
        ridge_oof=a_train[
            "ridge"
        ].astype(
            np.float32
        ),
        raw_residual_oof=train_raw_oof.astype(
            np.float32
        ),
        stage2_truewind_oof=train_stage2_oof.astype(
            np.float32
        ),
        alpha=alpha.astype(
            np.float32
        ),
        horizons_min=np.asarray(
            HORIZONS,
            dtype=np.int64,
        ),
    )

    np.savez_compressed(
        out
        / "15B_validation_predictions.npz",
        y_true=a_val[
            "y"
        ].astype(
            np.float32
        ),
        ridge=a_val[
            "ridge"
        ].astype(
            np.float32
        ),
        raw_residual=raw_val.astype(
            np.float32
        ),
        stage2_truewind=final_val.astype(
            np.float32
        ),
        alpha=alpha.astype(
            np.float32
        ),
        horizons_min=np.asarray(
            HORIZONS,
            dtype=np.int64,
        ),
    )

    np.savez_compressed(
        out
        / "15B_test_predictions.npz",
        y_true=a_test[
            "y"
        ].astype(
            np.float32
        ),
        ridge=a_test[
            "ridge"
        ].astype(
            np.float32
        ),
        raw_residual=raw_test.astype(
            np.float32
        ),
        stage2_truewind=final_test.astype(
            np.float32
        ),
        alpha=alpha.astype(
            np.float32
        ),
        horizons_min=np.asarray(
            HORIZONS,
            dtype=np.int64,
        ),
    )

    checkpoint_path = (
        out
        / "15B_truewind_residual.pt"
    )

    torch.save(
        {
            "state_dict": state_dict_cpu(
                bundle[
                    "model"
                ]
            ),
            "feature_scaler": bundle[
                "scaler"
            ],
            "alpha": alpha,
            "best_epoch": best_epoch,
            "seed": SEED,
            "horizons_min": HORIZONS,
            "network": {
                "gru_hidden": GRU_HIDDEN,
                "gru_layers": GRU_LAYERS,
                "shared_hidden": SHARED_HIDDEN,
                "dropout": DROPOUT,
            },
        },
        checkpoint_path,
    )

    freeze = {
        "stage": "15B",
        "training": (
            "Stage15A TrueWind Ridge frozen; "
            "residual GRU trained on OOF Ridge residual"
        ),
        "input_information": (
            "only U,V,T,RH and deterministic temporal transforms"
        ),
        "best_epoch": best_epoch,
        "alpha_max": ALPHA_MAX,
        "alpha_matrix": alpha,
        "validation_residual_corr": val_corr,
        "test_residual_corr_diagnostic": test_corr,
        "parameter_count_residual_nn": count_parameters(
            bundle[
                "model"
            ]
        ),
        "test_used_for_training": False,
        "test_used_for_alpha": False,
        "next_stage": (
            "Stage15C must use train_stage2_truewind_oof for training and "
            "15B validation/test stage2_truewind for validation/test apparent-wind physics."
        ),
    }

    save_json(
        out
        / "15B_frozen_truewind_residual.json",
        freeze,
    )

    # -----------------------------------------------------------------
    # Report.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 7/7] Final report."
    )

    test_table = metrics.loc[
        metrics[
            "split"
        ]
        == "test"
    ].copy()

    val_table = metrics.loc[
        metrics[
            "split"
        ]
        == "validation"
    ].copy()

    with (
        out
        / "15B_REPORT.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "15B — STEP 2: TRUE-WIND RIDGE RESIDUAL CORRECTION\n"
        )

        f.write(
            "="
            * 120
            + "\n\n"
        )

        f.write(
            f"Best epoch: {best_epoch}\n"
        )

        f.write(
            f"Residual NN params: {count_parameters(bundle['model'])}\n\n"
        )

        f.write(
            "FROZEN ALPHA [horizon x U/V]\n"
        )

        f.write(
            "-"
            * 120
            + "\n"
        )

        for j, h in enumerate(
            HORIZONS
        ):
            f.write(
                f"{h:2d} min: "
                f"U={alpha[j,0]:.8f}, "
                f"V={alpha[j,1]:.8f}\n"
            )

        f.write(
            "\nVALIDATION\n"
        )

        f.write(
            "-"
            * 120
            + "\n"
        )

        f.write(
            val_table.to_string(
                index=False
            )
        )

        f.write(
            "\n\nTEST\n"
        )

        f.write(
            "-"
            * 120
            + "\n"
        )

        f.write(
            test_table.to_string(
                index=False
            )
        )

        f.write(
            "\n\nNEXT\n"
        )

        f.write(
            "-"
            * 120
            + "\n"
        )

        f.write(
            "Stage15C must use 15B_train_stage2_oof_predictions.npz "
            "as the TRAIN true-wind input to apparent-wind physics. "
            "Do not replace it with the original Ridge wind prediction.\n"
        )

    report = {
        "best_epoch": best_epoch,
        "alpha": alpha,
        "residual_nn_params": count_parameters(
            bundle[
                "model"
            ]
        ),
        "validation_corr": val_corr,
        "test_corr_diagnostic": test_corr,
        "metrics_summary": summary.to_dict(
            orient="records"
        ),
    }

    save_json(
        out
        / "15B_REPORT.json",
        report,
    )

    log("")
    log("=" * 140)
    log(
        "15B FINAL TEST — TRUE WIND"
    )
    log("=" * 140)

    log(
        test_table.to_string(
            index=False
        )
    )

    log("")
    log(
        "Frozen alpha:"
    )

    for j, h in enumerate(
        HORIZONS
    ):
        log(
            f"  {h:2d} min | "
            f"U={alpha[j,0]:.4f} | "
            f"V={alpha[j,1]:.4f} | "
            f"test corr U/V="
            f"{test_corr[j,0]:+.4f}/"
            f"{test_corr[j,1]:+.4f}"
        )

    log("")
    log(
        "[NEXT] Stage15C = Vessel/HDG residual branch."
    )

    log(
        "       Apparent-wind physics must use Stage15B final true wind."
    )

    log("")
    log(
        f"[SAVED] {checkpoint_path}"
    )

    log(
        f"[SAVED] {out / '15B_train_stage2_oof_predictions.npz'}"
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
