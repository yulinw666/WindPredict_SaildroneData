# -*- coding: utf-8 -*-
"""
15E_WG_TrueWind_GatedResidual_5seed_validation.py

15E-WG
======
Strict wind-only mechanism ablation built on the already frozen 15E-H.

Baseline 15E-H true-wind branch:
    W_hat = W_Ridge + alpha(h,c) * DeltaW

15E-WG true-wind branch:
    W_hat = W_Ridge
          + alpha(h,c)
          * gate(i,h,c)
          * DeltaW

where:
    alpha(h,c) : validation-derived GLOBAL horizon/component shrinkage
    gate(i,h,c): learned SAMPLE-WISE sigmoid trust
    DeltaW     : nonlinear true-wind residual

Scientific isolation
--------------------
The OLD compact vessel prediction and the 15E-H corrected heading are
loaded from the previously completed 15E-H validation artifacts.

They are NOT retrained in this script.

Therefore 15E-H -> 15E-WG changes ONLY the true-wind residual mechanism.

Expected invariants:
    Vessel-vector RMSE : identical
    SOG RMSE           : identical
    HDG MAE/RMSE       : identical

Metrics allowed to change:
    U/V/vector/WS/WD
    apparent-wind vector
    AWS
    AWA

Gated TrueWind architecture
---------------------------
The encoder is kept matched to Stage15B:
    input features : U,V,T,RH-derived 10-channel sequence
    history        : 60 one-minute observations
    GRU            : 2 layers, hidden=48
    shared MLP     : 48
    dropout        : same as Stage15B

Only one mechanism is added:
    residual head : 10 outputs
    gate head     : 10 outputs

Residual head:
    tanh-bounded normalized residual, zero initialized

Gate head:
    sigmoid output
    zero weights
    bias = -1 -> initial gate = 0.26894

Because residual output is initialized to zero, epoch 0 remains exactly
the Ridge anchor regardless of the initial gate.

Training:
    loss = MSE(gate * residual_norm, residual_target_norm)

Validation checkpointing:
    global alpha(h,c) is analytically selected on validation
    after the sample-wise gate is applied.

Alpha:
    clipped to [0,1], same final safety convention as Stage15B-F.

Five seeds
----------
500043, 501052, 502061, 503070, 504079

TRAIN/VALIDATION only.
Internal TEST is NOT loaded.

Dependencies
------------
The following scripts should exist in --src-dir:
    15B_train_1min_TrueWind_Residual_Correction_v2.py
    15E_hybrid_NewWind_OldCompactVessel_5seed_validation.py
    15C_train_1min_VesselHDG_Residual_ApparentWind.py

The already completed 15E-H output directory must contain:
    15E_H_seed_<seed>_validation_predictions.npz

Recommended PowerShell
----------------------
python "D:\\project\\WindPredict_SaildroneData\\src\\15E_WG_TrueWind_GatedResidual_5seed_validation.py" --dataset-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\SD1090_TPOS2024_JointForecasting_v0_1" --stage15a-truewind-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15A_R_1min_Dual_Ridge_Anchors_v0_2" --stage15eh-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15E_H_HDG_GatedResidual_5seed_v0_1" --src-dir "D:\\project\\WindPredict_SaildroneData\\src" --output-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15E_WG_TrueWind_GatedResidual_5seed_v0_1"
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
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

DEFAULT_STAGE15EH_DIR = (
    ROOT / "data" / "forecasting"
    / "15E_H_HDG_GatedResidual_5seed_v0_1"
)

DEFAULT_SRC_DIR = ROOT / "src"

DEFAULT_OUTPUT_DIR = (
    ROOT / "data" / "forecasting"
    / "15E_WG_TrueWind_GatedResidual_5seed_v0_1"
)

SEEDS = [
    500043,
    501052,
    502061,
    503070,
    504079,
]

HORIZONS = [1, 2, 3, 5, 10]

ALPHA_CAP = 1.0
GATE_BIAS_INIT = -1.0
EPS = 1e-12

# Current Stage15B neural params = 26,074.
# WG adds one Linear(48 -> 10) gate head = 490 params.
EXPECTED_WG_TRUEWIND_PARAMS = 26564

# Frozen 15E-H whole neural count:
# Stage15B 26074 + old vessel 22164 + HDG head 1300 = 49538.
FROZEN_15EH_NEURAL_PARAMS = 49538
WG_TOTAL_NEURAL_PARAMS = (
    FROZEN_15EH_NEURAL_PARAMS
    - 26074
    + EXPECTED_WG_TRUEWIND_PARAMS
)  # 50028


# =============================================================================
# Generic utilities
# =============================================================================

def log(msg=""):
    print(msg, flush=True)


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import module from {path}"
        )

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
        json.dump(
            cv(obj),
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


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
    return int(
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )


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
        return torch.cuda.amp.autocast(
            enabled=True
        )


def create_grad_scaler(torch, device):
    if device.type != "cuda":
        return None

    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )
    except Exception:
        try:
            return torch.cuda.amp.GradScaler(
                enabled=True
            )
        except Exception:
            return None


# =============================================================================
# Frozen 15E-H validation artifacts
# =============================================================================

def load_15eh_seed(
    stage15eh_dir: Path,
    seed: int,
):
    path = (
        stage15eh_dir
        / f"15E_H_seed_{seed}_validation_predictions.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen 15E-H artifact: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as z:
        required = [
            "y_raw",
            "stage15b_truewind",
            "old_vessel_hdg",
            "final_hdg_q",
        ]

        missing = [
            k
            for k in required
            if k not in z.files
        ]

        if missing:
            raise RuntimeError(
                f"{path.name}: missing keys {missing}"
            )

        out = {
            "y_raw": np.asarray(
                z["y_raw"],
                dtype=np.float32,
            ),
            "baseline_truewind": np.asarray(
                z["stage15b_truewind"],
                dtype=np.float32,
            ),
            "old_vessel_hdg": np.asarray(
                z["old_vessel_hdg"],
                dtype=np.float32,
            ),
            "final_hdg_q": np.asarray(
                z["final_hdg_q"],
                dtype=np.float32,
            ),
        }

        if "alpha_hdg" in z.files:
            out["alpha_hdg"] = np.asarray(
                z["alpha_hdg"],
                dtype=np.float32,
            )

    if (
        out["baseline_truewind"].ndim != 3
        or out["baseline_truewind"].shape[1:] != (5, 2)
    ):
        raise RuntimeError(
            f"{path.name}: bad truewind shape "
            f"{out['baseline_truewind'].shape}"
        )

    if (
        out["old_vessel_hdg"].ndim != 3
        or out["old_vessel_hdg"].shape[1:] != (5, 4)
    ):
        raise RuntimeError(
            f"{path.name}: bad vessel/HDG shape "
            f"{out['old_vessel_hdg'].shape}"
        )

    if (
        out["final_hdg_q"].ndim != 3
        or out["final_hdg_q"].shape[1:] != (5, 2)
    ):
        raise RuntimeError(
            f"{path.name}: bad final HDG shape "
            f"{out['final_hdg_q'].shape}"
        )

    # Rebuild the exact frozen 15E-H vessel + heading target tensor.
    vh = out["old_vessel_hdg"].copy()
    vh[:, :, 2:4] = out["final_hdg_q"]

    out["final_vessel_hdg"] = vh

    return out


# =============================================================================
# WG TrueWind model
# =============================================================================

def build_wg_model_class(torch, nn, bmod):
    """
    Same Stage15B encoder/shared trunk.
    Only addition: a sample-wise gate head.
    """

    class GatedTrueWindResidualGRU(nn.Module):
        def __init__(self):
            super().__init__()

            self.gru = nn.GRU(
                input_size=10,
                hidden_size=bmod.GRU_HIDDEN,
                num_layers=bmod.GRU_LAYERS,
                batch_first=True,
                dropout=(
                    bmod.DROPOUT
                    if bmod.GRU_LAYERS > 1
                    else 0.0
                ),
            )

            self.shared = nn.Sequential(
                nn.Linear(
                    bmod.GRU_HIDDEN + 10,
                    bmod.SHARED_HIDDEN,
                ),
                nn.GELU(),
                nn.Dropout(
                    bmod.DROPOUT
                ),
            )

            self.residual_head = nn.Linear(
                bmod.SHARED_HIDDEN,
                len(HORIZONS) * 2,
            )

            self.gate_head = nn.Linear(
                bmod.SHARED_HIDDEN,
                len(HORIZONS) * 2,
            )

            # Exact-Ridge initialization.
            nn.init.zeros_(
                self.residual_head.weight
            )
            nn.init.zeros_(
                self.residual_head.bias
            )

            # Gate initially conservative, but correction is still exactly zero
            # because residual head is zero.
            nn.init.zeros_(
                self.gate_head.weight
            )
            nn.init.constant_(
                self.gate_head.bias,
                GATE_BIAS_INIT,
            )

        def forward(self, seq, summary):
            z, _ = self.gru(seq)
            last = z[:, -1, :]

            h = self.shared(
                torch.cat(
                    [last, summary],
                    dim=1,
                )
            )

            residual = torch.tanh(
                self.residual_head(h)
            ).reshape(
                -1,
                len(HORIZONS),
                2,
            )

            gate = torch.sigmoid(
                self.gate_head(h)
            ).reshape(
                -1,
                len(HORIZONS),
                2,
            )

            correction = gate * residual

            return (
                residual,
                gate,
                correction,
            )

    return GatedTrueWindResidualGRU


def predict_wg(
    *,
    torch,
    model,
    features_scaled,
    scaler,
    device,
    batch_size,
):
    model.eval()

    residual_all = []
    gate_all = []
    correction_all = []

    with torch.no_grad():
        for a in range(
            0,
            len(features_scaled["seq"]),
            batch_size,
        ):
            b = min(
                a + batch_size,
                len(features_scaled["seq"]),
            )

            seq_t = (
                torch.from_numpy(
                    features_scaled["seq"][a:b]
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            sum_t = (
                torch.from_numpy(
                    features_scaled["summary"][a:b]
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            with amp_context(
                torch,
                device,
            ):
                residual_norm, gate, correction_norm = model(
                    seq_t,
                    sum_t,
                )

            residual_norm = (
                residual_norm.detach()
                .float()
                .cpu()
                .numpy()
            )

            gate_np = (
                gate.detach()
                .float()
                .cpu()
                .numpy()
            )

            correction_norm = (
                correction_norm.detach()
                .float()
                .cpu()
                .numpy()
            )

            residual_raw = (
                residual_norm
                * scaler["residual_std"][None, :, :]
            )

            correction_raw = (
                correction_norm
                * scaler["residual_std"][None, :, :]
            )

            residual_all.append(
                residual_raw.astype(np.float32)
            )

            gate_all.append(
                gate_np.astype(np.float32)
            )

            correction_all.append(
                correction_raw.astype(np.float32)
            )

    return (
        np.concatenate(
            residual_all,
            axis=0,
        ),
        np.concatenate(
            gate_all,
            axis=0,
        ),
        np.concatenate(
            correction_all,
            axis=0,
        ),
    )


def train_wg_model(
    *,
    torch,
    nn,
    bmod,
    train_features,
    residual_train,
    val_features,
    y_val,
    ridge_val,
    seed,
):
    seed_everything(
        torch,
        seed,
    )

    scaler = bmod.fit_feature_scaler(
        train_features,
        residual_train,
    )

    tr = bmod.transform_features(
        train_features,
        scaler,
    )

    va = bmod.transform_features(
        val_features,
        scaler,
    )

    target_norm = bmod.normalize_residual_target(
        residual_train,
        scaler,
    )

    Model = build_wg_model_class(
        torch,
        nn,
        bmod,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = Model().to(
        device
    )

    params = count_parameters(
        model
    )

    if params != EXPECTED_WG_TRUEWIND_PARAMS:
        raise RuntimeError(
            f"WG parameter mismatch: "
            f"{params} != {EXPECTED_WG_TRUEWIND_PARAMS}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=bmod.LR,
        weight_decay=bmod.WEIGHT_DECAY,
    )

    grad_scaler = create_grad_scaler(
        torch,
        device,
    )

    # Force final frozen alpha convention.
    bmod.ALPHA_MAX = ALPHA_CAP

    # Epoch 0 = exact Ridge.
    zero = np.zeros_like(
        ridge_val,
        dtype=np.float32,
    )

    alpha0 = bmod.analytic_alpha_matrix(
        y_val,
        ridge_val,
        zero,
    )

    best_score = bmod.mean_vector_rmse(
        y_val,
        bmod.apply_alpha(
            ridge_val,
            zero,
            alpha0,
        ),
    )

    best_epoch = 0
    best_state = state_dict_cpu(
        model
    )
    best_alpha = alpha0.copy()
    best_gate_val = np.full(
        (
            len(y_val),
            len(HORIZONS),
            2,
        ),
        1.0 / (1.0 + np.exp(-GATE_BIAS_INIT)),
        dtype=np.float32,
    )
    best_correction_val = zero.copy()
    best_residual_val = zero.copy()

    patience = 0
    history = []

    n = len(
        target_norm
    )

    for epoch in range(
        1,
        int(bmod.MAX_EPOCHS) + 1,
    ):
        model.train()

        rng = np.random.default_rng(
            seed + epoch * 1009
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
            int(bmod.BATCH_SIZE),
        ):
            idx = order[
                start:
                start + int(bmod.BATCH_SIZE)
            ]

            seq_t = (
                torch.from_numpy(
                    tr["seq"][idx]
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            sum_t = (
                torch.from_numpy(
                    tr["summary"][idx]
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            target_t = (
                torch.from_numpy(
                    target_norm[idx]
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with amp_context(
                torch,
                device,
            ):
                _, _, correction_norm = model(
                    seq_t,
                    sum_t,
                )

                # Gate is part of the learned correction itself.
                loss = torch.mean(
                    (
                        correction_norm
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
                    bmod.GRAD_CLIP,
                )

                grad_scaler.step(
                    optimizer
                )

                grad_scaler.update()

            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    bmod.GRAD_CLIP,
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

        residual_val, gate_val, correction_val = predict_wg(
            torch=torch,
            model=model,
            features_scaled=va,
            scaler=scaler,
            device=device,
            batch_size=int(
                bmod.BATCH_SIZE
            ),
        )

        alpha = bmod.analytic_alpha_matrix(
            y_val,
            ridge_val,
            correction_val,
        )

        final_val = bmod.apply_alpha(
            ridge_val,
            correction_val,
            alpha,
        )

        score = bmod.mean_vector_rmse(
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
            best_gate_val = gate_val.copy()
            best_correction_val = correction_val.copy()
            best_residual_val = residual_val.copy()
            patience = 0
        else:
            patience += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_mean_vector_RMSE": score,
                "best_validation_mean_vector_RMSE": best_score,
                "mean_gate": float(
                    np.mean(gate_val)
                ),
            }
        )

        if (
            epoch <= 3
            or epoch % 10 == 0
            or improved
        ):
            log(
                f"      WG epoch={epoch:03d} | "
                f"loss={train_loss:.6f} | "
                f"val vector={score:.6f} | "
                f"mean gate={np.mean(gate_val):.4f}"
                + (
                    " *"
                    if improved
                    else ""
                )
            )

        if (
            patience
            >= int(bmod.EARLY_STOP_PATIENCE)
        ):
            break

    model.load_state_dict(
        best_state,
        strict=True,
    )

    # Recompute from the frozen best checkpoint to guarantee consistency.
    residual_val, gate_val, correction_val = predict_wg(
        torch=torch,
        model=model,
        features_scaled=va,
        scaler=scaler,
        device=device,
        batch_size=int(
            bmod.BATCH_SIZE
        ),
    )

    alpha = bmod.analytic_alpha_matrix(
        y_val,
        ridge_val,
        correction_val,
    )

    final_val = bmod.apply_alpha(
        ridge_val,
        correction_val,
        alpha,
    )

    final_score = bmod.mean_vector_rmse(
        y_val,
        final_val,
    )

    if not np.isclose(
        final_score,
        best_score,
        rtol=0.0,
        atol=2e-6,
    ):
        raise RuntimeError(
            f"Frozen WG checkpoint mismatch: "
            f"{final_score} vs {best_score}"
        )

    return {
        "model": model,
        "device": device,
        "scaler": scaler,
        "best_epoch": int(
            best_epoch
        ),
        "best_score": float(
            final_score
        ),
        "alpha": alpha.astype(
            np.float32
        ),
        "residual_val": residual_val.astype(
            np.float32
        ),
        "gate_val": gate_val.astype(
            np.float32
        ),
        "correction_val": correction_val.astype(
            np.float32
        ),
        "final_val": final_val.astype(
            np.float32
        ),
        "history": pd.DataFrame(
            history
        ),
        "parameters": int(
            params
        ),
    }


# =============================================================================
# Reporting helpers
# =============================================================================

def mean_truewind_metrics(df):
    cols = [
        "U_RMSE_mps",
        "V_RMSE_mps",
        "vector_RMSE_mps",
        "WS_RMSE_mps",
        "WD_RMSE_deg",
    ]

    return {
        c: float(
            df[c].mean()
        )
        for c in cols
    }


def mean_apparent_metrics(df):
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


def aggregate_df(df):
    rows = []

    for col in df.columns:
        if col in {
            "seed",
            "model",
        }:
            continue

        if not np.issubdtype(
            df[col].dtype,
            np.number,
        ):
            continue

        vals = df[col].to_numpy(
            dtype=np.float64
        )

        rows.append(
            {
                "metric": col,
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

    return pd.DataFrame(
        rows
    )


def paired_effect(
    baseline_df,
    candidate_df,
    metrics,
):
    rows = []

    for metric in metrics:
        a = baseline_df[
            metric
        ].to_numpy(
            dtype=np.float64
        )

        b = candidate_df[
            metric
        ].to_numpy(
            dtype=np.float64
        )

        d = a - b

        rows.append(
            {
                "metric": metric,
                "15E_H_mean": float(
                    np.mean(a)
                ),
                "15E_WG_mean": float(
                    np.mean(b)
                ),
                "mean_reduction_15E_H_minus_15E_WG": float(
                    np.mean(d)
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
                        d,
                        ddof=1,
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


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
        "--stage15a-truewind-dir",
        type=Path,
        default=DEFAULT_STAGE15A_TRUEWIND_DIR,
    )

    parser.add_argument(
        "--stage15eh-dir",
        type=Path,
        default=DEFAULT_STAGE15EH_DIR,
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

    # Dependencies.
    bmod = load_module(
        "stage15b_for_15ewg",
        args.src_dir
        / "15B_train_1min_TrueWind_Residual_Correction_v2.py",
    )

    emod = load_module(
        "stage15e_helper_for_15ewg",
        args.src_dir
        / "15E_hybrid_NewWind_OldCompactVessel_5seed_validation.py",
    )

    cmod = load_module(
        "stage15c_geometry_for_15ewg",
        args.src_dir
        / "15C_train_1min_VesselHDG_Residual_ApparentWind.py",
    )

    bmod.ALPHA_MAX = ALPHA_CAP

    torch, nn = emod.import_torch()
    emod.configure_torch(
        torch
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    log("=" * 164)
    log(
        "15E-WG — TRUE-WIND SAMPLE-WISE GATED RESIDUAL, FIVE-SEED VALIDATION"
    )
    log("=" * 164)

    log(
        "Baseline = frozen 15E-H."
    )

    log(
        "Only the TrueWind residual mechanism is changed."
    )

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
        f"WG Stage15B parameters = "
        f"{EXPECTED_WG_TRUEWIND_PARAMS:,}"
    )

    log(
        f"Whole 15E-WG neural parameters = "
        f"{WG_TOTAL_NEURAL_PARAMS:,}"
    )

    # -----------------------------------------------------------------
    # Load train / validation only.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 1/7] Loading TRAIN/VALIDATION and TrueWind anchors."
    )

    train = emod.load_split(
        args.dataset_dir,
        "train",
    )

    val = emod.load_split(
        args.dataset_dir,
        "validation",
    )

    tw_train, tw_val = emod.load_truewind_anchor(
        args.stage15a_truewind_dir
    )

    if len(train["X"]) != len(tw_train["y"]):
        raise RuntimeError(
            "TRAIN TrueWind alignment mismatch."
        )

    if len(val["X"]) != len(tw_val["y"]):
        raise RuntimeError(
            "VALIDATION TrueWind alignment mismatch."
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
        "[STAGE 2/7] Apparent-wind reference audit."
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

    app_ref_val = cmod.canonical_apparent_reference(
        val_for_c,
        audit,
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

    # Build same deterministic Stage15B feature set once.
    feat_train = bmod.build_features(
        train["X"]
    )

    feat_val = bmod.build_features(
        val["X"]
    )

    y_vh_val = val["y"][:, :, 2:6]

    per_seed_rows = []
    wind_horizon_frames = []
    apparent_horizon_frames = []
    alpha_gate_rows = []

    # -----------------------------------------------------------------
    # Five seeds.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 3/7] Five-seed gated TrueWind training."
    )

    for seed_idx, seed in enumerate(
        SEEDS,
        start=1,
    ):
        log("")
        log(
            "-" * 164
        )

        log(
            f"[SEED {seed_idx}/5] {seed}"
        )

        log(
            "-" * 164
        )

        # =============================================================
        # Load exact frozen vessel+HDG + baseline Stage15B for this seed.
        # =============================================================
        frozen = load_15eh_seed(
            args.stage15eh_dir,
            seed,
        )

        if not np.allclose(
            frozen["y_raw"],
            val["y"],
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"Seed {seed}: 15E-H y_raw does not align "
                f"with current validation data."
            )

        vh_frozen = frozen[
            "final_vessel_hdg"
        ]

        baseline_wind = frozen[
            "baseline_truewind"
        ]

        # Baseline 15E-H metrics.
        base_tw_metrics = bmod.metrics_per_horizon(
            tw_val["y"],
            baseline_wind,
            "15E-H-TrueWind",
            "validation",
        )

        base_app_metrics = cmod.metrics_per_horizon(
            y_vh_val,
            vh_frozen,
            baseline_wind,
            app_ref_val,
            "validation",
            "15E-H",
        )

        # =============================================================
        # Train only WG TrueWind.
        # =============================================================
        log(
            "  [WG WIND] Training 2-layer GRU48 "
            "+ sample-wise gate."
        )

        wg = train_wg_model(
            torch=torch,
            nn=nn,
            bmod=bmod,
            train_features=feat_train,
            residual_train=tw_train["residual"],
            val_features=feat_val,
            y_val=tw_val["y"],
            ridge_val=tw_val["ridge"],
            seed=seed,
        )

        wg_wind = wg[
            "final_val"
        ]

        wg_tw_metrics = bmod.metrics_per_horizon(
            tw_val["y"],
            wg_wind,
            "15E-WG-TrueWind",
            "validation",
        )

        wg_app_metrics = cmod.metrics_per_horizon(
            y_vh_val,
            vh_frozen,
            wg_wind,
            app_ref_val,
            "validation",
            "15E-WG",
        )

        # =============================================================
        # Invariance checks: vessel & heading cannot change.
        # =============================================================
        invariant_cols = [
            "Vessel_vector_RMSE_mps",
            "SOG_RMSE_mps",
            "HDG_MAE_deg",
            "HDG_RMSE_deg",
        ]

        base_app_mean = mean_apparent_metrics(
            base_app_metrics
        )

        wg_app_mean = mean_apparent_metrics(
            wg_app_metrics
        )

        for col in invariant_cols:
            if not np.isclose(
                base_app_mean[col],
                wg_app_mean[col],
                rtol=0.0,
                atol=1e-10,
            ):
                raise RuntimeError(
                    f"Wind-only invariance failure: {col}: "
                    f"baseline={base_app_mean[col]}, "
                    f"WG={wg_app_mean[col]}"
                )

        base_tw_mean = mean_truewind_metrics(
            base_tw_metrics
        )

        wg_tw_mean = mean_truewind_metrics(
            wg_tw_metrics
        )

        # Append horizon outputs.
        btf = base_tw_metrics.copy()
        btf.insert(
            0,
            "seed",
            seed,
        )
        wind_horizon_frames.append(
            btf
        )

        wtf = wg_tw_metrics.copy()
        wtf.insert(
            0,
            "seed",
            seed,
        )
        wind_horizon_frames.append(
            wtf
        )

        baf = base_app_metrics.copy()
        baf.insert(
            0,
            "seed",
            seed,
        )
        apparent_horizon_frames.append(
            baf
        )

        waf = wg_app_metrics.copy()
        waf.insert(
            0,
            "seed",
            seed,
        )
        apparent_horizon_frames.append(
            waf
        )

        # Gate / alpha diagnostics.
        alpha = np.asarray(
            wg["alpha"],
            dtype=np.float32,
        )

        gate = np.asarray(
            wg["gate_val"],
            dtype=np.float32,
        )

        for j, h in enumerate(
            HORIZONS
        ):
            for c, name in enumerate(
                ["U", "V"]
            ):
                g = gate[:, j, c]

                alpha_gate_rows.append(
                    {
                        "seed": seed,
                        "horizon_min": h,
                        "component": name,
                        "alpha": float(
                            alpha[j, c]
                        ),
                        "mean_gate": float(
                            np.mean(g)
                        ),
                        "std_gate": float(
                            np.std(
                                g,
                                ddof=0,
                            )
                        ),
                        "p10_gate": float(
                            np.percentile(
                                g,
                                10,
                            )
                        ),
                        "p50_gate": float(
                            np.percentile(
                                g,
                                50,
                            )
                        ),
                        "p90_gate": float(
                            np.percentile(
                                g,
                                90,
                            )
                        ),
                        "effective_mean_scale_alpha_times_gate": float(
                            alpha[j, c]
                            * np.mean(g)
                        ),
                    }
                )

        # Seed summary.
        row_base = {
            "seed": seed,
            "model": "15E-H",
            "WG_best_epoch": np.nan,
            "TrueWind_U_RMSE_mps": base_tw_mean[
                "U_RMSE_mps"
            ],
            "TrueWind_V_RMSE_mps": base_tw_mean[
                "V_RMSE_mps"
            ],
            "TrueWind_vector_RMSE_mps": base_tw_mean[
                "vector_RMSE_mps"
            ],
            "TrueWind_WS_RMSE_mps": base_tw_mean[
                "WS_RMSE_mps"
            ],
            "TrueWind_WD_RMSE_deg": base_tw_mean[
                "WD_RMSE_deg"
            ],
            **base_app_mean,
        }

        row_wg = {
            "seed": seed,
            "model": "15E-WG",
            "WG_best_epoch": int(
                wg["best_epoch"]
            ),
            "TrueWind_U_RMSE_mps": wg_tw_mean[
                "U_RMSE_mps"
            ],
            "TrueWind_V_RMSE_mps": wg_tw_mean[
                "V_RMSE_mps"
            ],
            "TrueWind_vector_RMSE_mps": wg_tw_mean[
                "vector_RMSE_mps"
            ],
            "TrueWind_WS_RMSE_mps": wg_tw_mean[
                "WS_RMSE_mps"
            ],
            "TrueWind_WD_RMSE_deg": wg_tw_mean[
                "WD_RMSE_deg"
            ],
            **wg_app_mean,
        }

        per_seed_rows.extend(
            [
                row_base,
                row_wg,
            ]
        )

        log("")
        log(
            "  [FROZEN WG ALPHA / GATE]"
        )

        for j, h in enumerate(
            HORIZONS
        ):
            log(
                f"      {h:2d} min | "
                f"alpha_U={alpha[j,0]:.6f}, "
                f"gate_U={np.mean(gate[:,j,0]):.4f} | "
                f"alpha_V={alpha[j,1]:.6f}, "
                f"gate_V={np.mean(gate[:,j,1]):.4f}"
            )

        log("")
        log(
            "  [SEED SUMMARY]"
        )

        log(
            f"      TrueWind vector: "
            f"15E-H={base_tw_mean['vector_RMSE_mps']:.6f} | "
            f"WG={wg_tw_mean['vector_RMSE_mps']:.6f}"
        )

        log(
            f"      AW vector:       "
            f"15E-H={base_app_mean['AppVector_RMSE_mps']:.6f} | "
            f"WG={wg_app_mean['AppVector_RMSE_mps']:.6f}"
        )

        log(
            f"      AWS:             "
            f"15E-H={base_app_mean['AWS_RMSE_mps']:.6f} | "
            f"WG={wg_app_mean['AWS_RMSE_mps']:.6f}"
        )

        log(
            f"      AWA MAE:         "
            f"15E-H={base_app_mean['AWA_MAE_deg']:.6f} | "
            f"WG={wg_app_mean['AWA_MAE_deg']:.6f}"
        )

        # Save validation prediction artifact.
        np.savez_compressed(
            out
            / f"15E_WG_seed_{seed}_validation_predictions.npz",
            y_raw=val["y"].astype(
                np.float32
            ),
            ridge_truewind=tw_val[
                "ridge"
            ].astype(
                np.float32
            ),
            baseline_15EH_truewind=baseline_wind.astype(
                np.float32
            ),
            gated_truewind=wg_wind.astype(
                np.float32
            ),
            frozen_vessel_hdg=vh_frozen.astype(
                np.float32
            ),
            alpha_truewind=alpha.astype(
                np.float32
            ),
            gate_truewind=gate.astype(
                np.float32
            ),
            residual_head_raw=wg[
                "residual_val"
            ].astype(
                np.float32
            ),
            gated_correction_raw=wg[
                "correction_val"
            ].astype(
                np.float32
            ),
            horizons_min=np.asarray(
                HORIZONS,
                dtype=np.int64,
            ),
        )

        # Save the frozen WG model for later external inference.
        checkpoint = {
            "seed": int(
                seed
            ),
            "model_state_dict": state_dict_cpu(
                wg["model"]
            ),
            "scaler": {
                k: np.asarray(
                    v
                )
                for k, v in wg[
                    "scaler"
                ].items()
            },
            "alpha": np.asarray(
                alpha,
                dtype=np.float32,
            ),
            "best_epoch": int(
                wg["best_epoch"]
            ),
            "best_validation_vector_RMSE": float(
                wg["best_score"]
            ),
            "GRU_hidden": int(
                bmod.GRU_HIDDEN
            ),
            "GRU_layers": int(
                bmod.GRU_LAYERS
            ),
            "shared_hidden": int(
                bmod.SHARED_HIDDEN
            ),
            "dropout": float(
                bmod.DROPOUT
            ),
            "alpha_cap": float(
                ALPHA_CAP
            ),
            "gate_bias_init": float(
                GATE_BIAS_INIT
            ),
            "parameters": int(
                wg["parameters"]
            ),
        }

        torch.save(
            checkpoint,
            out
            / f"15E_WG_seed_{seed}_truewind_model.pt",
        )

        wg["history"].to_csv(
            out
            / f"15E_WG_seed_{seed}_training_history.csv",
            index=False,
            encoding="utf-8-sig",
        )

        del wg

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Aggregate.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 4/7] Aggregating five seeds."
    )

    per_seed_df = pd.DataFrame(
        per_seed_rows
    )

    wind_horizon_df = pd.concat(
        wind_horizon_frames,
        ignore_index=True,
    )

    apparent_horizon_df = pd.concat(
        apparent_horizon_frames,
        ignore_index=True,
    )

    alpha_gate_df = pd.DataFrame(
        alpha_gate_rows
    )

    df_base = (
        per_seed_df.loc[
            per_seed_df["model"]
            == "15E-H"
        ]
        .reset_index(
            drop=True
        )
    )

    df_wg = (
        per_seed_df.loc[
            per_seed_df["model"]
            == "15E-WG"
        ]
        .reset_index(
            drop=True
        )
    )

    agg_base = aggregate_df(
        df_base
    )

    agg_base.insert(
        0,
        "model",
        "15E-H",
    )

    agg_wg = aggregate_df(
        df_wg
    )

    agg_wg.insert(
        0,
        "model",
        "15E-WG",
    )

    aggregate = pd.concat(
        [
            agg_base,
            agg_wg,
        ],
        ignore_index=True,
    )

    paired_metrics = [
        "TrueWind_U_RMSE_mps",
        "TrueWind_V_RMSE_mps",
        "TrueWind_vector_RMSE_mps",
        "TrueWind_WS_RMSE_mps",
        "TrueWind_WD_RMSE_deg",
        "AppVector_RMSE_mps",
        "AWS_RMSE_mps",
        "AWA_MAE_deg",
        "AWA_RMSE_deg",
        "Vessel_vector_RMSE_mps",
        "HDG_MAE_deg",
    ]

    paired = paired_effect(
        df_base,
        df_wg,
        paired_metrics,
    )

    # -----------------------------------------------------------------
    # Save tabular results.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 5/7] Saving CSV outputs."
    )

    per_seed_df.to_csv(
        out
        / "15E_WG_per_seed_validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    aggregate.to_csv(
        out
        / "15E_WG_validation_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired.to_csv(
        out
        / "15E_WG_paired_effect.csv",
        index=False,
        encoding="utf-8-sig",
    )

    wind_horizon_df.to_csv(
        out
        / "15E_WG_truewind_per_horizon_all_seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    apparent_horizon_df.to_csv(
        out
        / "15E_WG_apparent_per_horizon_all_seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    alpha_gate_df.to_csv(
        out
        / "15E_WG_alpha_gate_all_seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Final output.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 6/7] Final five-seed comparison."
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

    final_metrics = [
        "TrueWind_U_RMSE_mps",
        "TrueWind_V_RMSE_mps",
        "TrueWind_vector_RMSE_mps",
        "TrueWind_WS_RMSE_mps",
        "TrueWind_WD_RMSE_deg",
        "AppVector_RMSE_mps",
        "AWS_RMSE_mps",
        "AWA_MAE_deg",
        "AWA_RMSE_deg",
        "Vessel_vector_RMSE_mps",
        "HDG_MAE_deg",
    ]

    log("")
    log("=" * 164)

    log(
        "15E-WG FINAL — TRUE-WIND GATED RESIDUAL, "
        "SD1090 VALIDATION, FIVE SEEDS"
    )

    log("=" * 164)

    for metric in final_metrics:
        a = mstd(
            df_base,
            metric,
        )

        b = mstd(
            df_wg,
            metric,
        )

        log(
            f"{metric:30s} | "
            f"15E-H={a[0]:.6f} ± {a[1]:.6f} | "
            f"15E-WG={b[0]:.6f} ± {b[1]:.6f}"
        )

    log("")
    log(
        "[PAIRED 15E-H -> 15E-WG EFFECT]"
    )

    log(
        paired.to_string(
            index=False
        )
    )

    log("")
    log(
        f"Stage15B original neural params = 26,074"
    )

    log(
        f"WG TrueWind neural params       = "
        f"{EXPECTED_WG_TRUEWIND_PARAMS:,}"
    )

    log(
        f"Additional gate parameters      = "
        f"{EXPECTED_WG_TRUEWIND_PARAMS - 26074:,}"
    )

    log(
        f"Whole 15E-H neural params       = "
        f"{FROZEN_15EH_NEURAL_PARAMS:,}"
    )

    log(
        f"Whole 15E-WG neural params      = "
        f"{WG_TOTAL_NEURAL_PARAMS:,}"
    )

    # -----------------------------------------------------------------
    # Report.
    # -----------------------------------------------------------------
    log("")
    log(
        "[STAGE 7/7] Writing report."
    )

    report = {
        "experiment": "15E-WG",
        "internal_test_loaded": False,
        "baseline": "frozen 15E-H",
        "changed_component": (
            "TrueWind residual mechanism only"
        ),
        "formula": (
            "W = W_Ridge + alpha(h,c) * gate(i,h,c) * DeltaW"
        ),
        "architecture": {
            "GRU_layers": int(
                bmod.GRU_LAYERS
            ),
            "GRU_hidden": int(
                bmod.GRU_HIDDEN
            ),
            "shared_hidden": int(
                bmod.SHARED_HIDDEN
            ),
            "dropout": float(
                bmod.DROPOUT
            ),
            "gate_bias_init": float(
                GATE_BIAS_INIT
            ),
            "alpha_cap": float(
                ALPHA_CAP
            ),
            "WG_truewind_parameters": int(
                EXPECTED_WG_TRUEWIND_PARAMS
            ),
            "whole_15EWG_neural_parameters": int(
                WG_TOTAL_NEURAL_PARAMS
            ),
        },
        "scientific_invariants": {
            "vessel_prediction_frozen": True,
            "heading_prediction_frozen": True,
            "expected_identical_metrics": [
                "Vessel_vector_RMSE_mps",
                "SOG_RMSE_mps",
                "HDG_MAE_deg",
                "HDG_RMSE_deg",
            ],
        },
        "seeds": SEEDS,
        "paired_effect": paired.to_dict(
            orient="records"
        ),
        "alpha_gate_summary": alpha_gate_df.to_dict(
            orient="records"
        ),
    }

    save_json(
        out
        / "15E_WG_REPORT.json",
        report,
    )

    with (
        out
        / "15E_WG_REPORT.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "15E-WG — TRUE-WIND SAMPLE-WISE GATED RESIDUAL\n"
        )

        f.write(
            "=" * 150
            + "\n\n"
        )

        f.write(
            "Only the TrueWind residual mechanism changes.\n"
        )

        f.write(
            "Frozen vessel and heading are loaded from 15E-H.\n\n"
        )

        for metric in final_metrics:
            a = mstd(
                df_base,
                metric,
            )

            b = mstd(
                df_wg,
                metric,
            )

            f.write(
                f"{metric:30s} | "
                f"15E-H={a[0]:.8f} ± {a[1]:.8f} | "
                f"15E-WG={b[0]:.8f} ± {b[1]:.8f}\n"
            )

        f.write(
            "\nPAIRED EFFECT\n"
        )

        f.write(
            "-" * 150
            + "\n"
        )

        f.write(
            paired.to_string(
                index=False
            )
        )

        f.write(
            "\n\nALPHA / GATE\n"
        )

        f.write(
            "-" * 150
            + "\n"
        )

        f.write(
            alpha_gate_df.to_string(
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
