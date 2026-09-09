# -*- coding: utf-8 -*-
r"""
12M_train_WindResidualSpecialist_NieData_v5_LOMO.py

Development-only LOMO screening for a dedicated wind residual specialist.

Why this stage exists
---------------------
Stage 12L showed that the selected 12J/v4 joint model had nearly zero pooled
cross-mission correlation between its final wind correction and the true Ridge
wind residual:
    U corr ~ -0.02
    V corr ~ +0.03

At the same time, v4 strongly improved vessel/apparent-wind prediction. This
suggests multi-task gradient interference and poor conditioning of the small
wind residual inside the full-target standardized loss.

12M therefore isolates the scientific question:
    Can a compact model predict the *remaining Ridge U/V residual*
    across missions when that residual is trained directly and normalized
    by its own fold-training standard deviation?

SCIENTIFIC FIREWALL
-------------------
Reads ONLY:
    Antarctic.npz
    Atlantic.npz
    West_Coast.npz

NEVER reads:
    Tropical_Atlantic_TEST.npz
    Stage-12K outputs
    any held-out test metric

LOMO
----
Holdout Antarctic: train Atlantic + West Coast
Holdout Atlantic : train Antarctic + West Coast
Holdout West Coast: train Antarctic + Atlantic

Ridge anchor
------------
The exact Stage-12C Ridge protocol is retained:
    alpha = 1.0
    fold-train scalers only
    input = flattened 6 x 9 standardized context
    output = 4 standardized joint targets

Only the wind components are corrected in this stage:
    W_hat_z = W_ridge_z + DeltaW_z
    VESSEL_hat_z = VESSEL_ridge_z

Residual conditioning
---------------------
On each fold TRAIN only:

    R_train_z = Y_wind,z - Ridge_wind,z
    s_R = std(R_train_z)

The specialist is trained on:

    R_norm = R_train_z / s_R

and predicts correction:

    DeltaW_z = s_R * R_hat_norm

The output head is initialized to zero, so epoch 0 equals Ridge exactly.

Predeclared compact candidates
------------------------------
1) V5_ResidualMLP
   flattened standardized history + Ridge wind anchor -> compact MLP

2) V5_ResidualGRU
   standardized sequence -> GRU -> Ridge anchor context

3) V5_ResidualGRUTrend
   standardized sequence concatenated with first differences -> GRU
   -> Ridge anchor context

All candidates use the same residual-normalized MSE objective and wind-only
development selection score.

Selection score
---------------
    WindScore =
        0.25 * U_RMSE / Ridge_U_RMSE
      + 0.35 * V_RMSE / Ridge_V_RMSE
      + 0.25 * WS_RMSE / Ridge_WS_RMSE
      + 0.15 * WD_RMSE / Ridge_WD_RMSE

Lower is better. Ridge is exactly 1.0.

Each run includes epoch 0 (exact Ridge) as a valid checkpoint. Therefore a
candidate cannot be forced to keep a harmful neural correction.

Usage
-----
python "D:\project\WindPredict_SaildroneData\src\12M_train_WindResidualSpecialist_NieData_v5_LOMO.py" --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12M_WindResidualSpecialist_NieData_v5_LOMO_point_v0_1"

Smoke test
----------
Add:
    --debug-fast
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
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0-WindResidualSpecialist-NieData-v5-LOMO"

DEFAULT_PROJECT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")
DEFAULT_DATASET_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12G_Nie_point_sampled_benchmark_v0_1"
    / "dataset"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12M_WindResidualSpecialist_NieData_v5_LOMO_point_v0_1"
)

DEVELOPMENT_MISSIONS = ["Antarctic", "Atlantic", "West Coast"]
SCREENING_SEEDS = [500043, 501052, 502061]

RIDGE_ALPHA = 1.0
EPS = 1e-12


@dataclass(frozen=True)
class TrainFreeze:
    mlp_hidden_1: int = 64
    mlp_hidden_2: int = 32

    gru_hidden: int = 48
    dense_hidden: int = 48
    dropout: float = 0.10

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 100
    early_stop_patience: int = 15
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-5
    grad_clip_norm: float = 1.0

    correction_penalty: float = 1e-4
    residual_std_floor: float = 1e-4

    use_amp: bool = True
    train_shuffle: bool = False


FREEZE = TrainFreeze()

# Predeclared architecture screen.
CANDIDATES = {
    "V5_ResidualMLP": {
        "mode": "mlp",
        "description": "Flattened standardized history + Ridge wind anchor",
    },
    "V5_ResidualGRU": {
        "mode": "gru",
        "description": "GRU sequence encoder + Ridge wind anchor",
    },
    "V5_ResidualGRUTrend": {
        "mode": "gru_trend",
        "description": "GRU on [X_z, first-difference(X_z)] + Ridge wind anchor",
    },
}

WIND_SELECTION_WEIGHTS = {
    "U": 0.25,
    "V": 0.35,
    "WS": 0.25,
    "WD": 0.15,
}


def log(msg=""):
    print(msg, flush=True)


def load_base_module(project_root: Path):
    path = project_root / "src" / "12C_train_PhysicsCompact_NieData_LOMO.py"
    if not path.exists():
        raise FileNotFoundError(
            "Required Stage-12C base script not found:\n"
            f"  {path}"
        )

    name = "stage12c_base_for_12m"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for: {path}")

    mod = importlib.util.module_from_spec(spec)

    # Required for dataclass/type introspection under Python 3.11.
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise

    return mod, path


def save_json(path: Path, obj):
    def cv(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return None if not np.isfinite(x) else float(x)
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, dict):
            return {str(k): cv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [cv(v) for v in x]
        return x

    with path.open("w", encoding="utf-8") as f:
        json.dump(cv(obj), f, ensure_ascii=False, indent=2, sort_keys=True)


def seed_everything(torch, seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def amp_context(torch, enabled: bool, device):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    try:
        return torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=True
        )
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def create_grad_scaler(torch, enabled: bool):
    if not enabled:
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
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )


def make_trend_sequence_np(Xz: np.ndarray) -> np.ndarray:
    """
    Xz: [N,T,F]
    Return [N,T,2F] = [Xz, dXz].
    First difference at t=0 is zero.
    """
    Xz = np.asarray(Xz, dtype=np.float32)
    dx = np.zeros_like(Xz, dtype=np.float32)
    dx[:, 1:, :] = Xz[:, 1:, :] - Xz[:, :-1, :]
    return np.concatenate([Xz, dx], axis=-1).astype(
        np.float32, copy=False
    )


def build_model_class(torch, nn, feature_count: int, mode: str):
    if mode == "mlp":
        class ResidualMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Linear(
                        6 * feature_count + 2,
                        FREEZE.mlp_hidden_1,
                    ),
                    nn.LayerNorm(FREEZE.mlp_hidden_1),
                    nn.GELU(),
                    nn.Dropout(FREEZE.dropout),
                    nn.Linear(
                        FREEZE.mlp_hidden_1,
                        FREEZE.mlp_hidden_2,
                    ),
                    nn.GELU(),
                    nn.Dropout(FREEZE.dropout),
                )
                self.out = nn.Linear(FREEZE.mlp_hidden_2, 2)

                # Exact Ridge start.
                nn.init.zeros_(self.out.weight)
                nn.init.zeros_(self.out.bias)

            def forward(self, x, ridge_w_z):
                h = torch.cat(
                    [x.flatten(start_dim=1), ridge_w_z], dim=1
                )
                return self.out(self.backbone(h))

        return ResidualMLP

    if mode in {"gru", "gru_trend"}:
        in_features = (
            feature_count if mode == "gru" else 2 * feature_count
        )

        class ResidualGRU(nn.Module):
            def __init__(self):
                super().__init__()
                self.mode = mode
                self.gru = nn.GRU(
                    input_size=in_features,
                    hidden_size=FREEZE.gru_hidden,
                    num_layers=1,
                    batch_first=True,
                )
                self.backbone = nn.Sequential(
                    nn.Linear(
                        FREEZE.gru_hidden + 2,
                        FREEZE.dense_hidden,
                    ),
                    nn.LayerNorm(FREEZE.dense_hidden),
                    nn.GELU(),
                    nn.Dropout(FREEZE.dropout),
                )
                self.out = nn.Linear(FREEZE.dense_hidden, 2)

                # Exact Ridge start.
                nn.init.zeros_(self.out.weight)
                nn.init.zeros_(self.out.bias)

            def forward(self, x, ridge_w_z):
                if self.mode == "gru_trend":
                    dx = torch.zeros_like(x)
                    dx[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
                    x_in = torch.cat([x, dx], dim=-1)
                else:
                    x_in = x

                seq, _ = self.gru(x_in)
                h = seq[:, -1, :]
                z = self.backbone(
                    torch.cat([h, ridge_w_z], dim=1)
                )
                return self.out(z)

        return ResidualGRU

    raise ValueError(f"Unknown candidate mode: {mode}")


def wind_selection_score(metrics, ridge_metrics):
    eps = 1e-12

    u = metrics["wind_U_RMSE_mps"] / max(
        ridge_metrics["wind_U_RMSE_mps"], eps
    )
    v = metrics["wind_V_RMSE_mps"] / max(
        ridge_metrics["wind_V_RMSE_mps"], eps
    )
    ws = metrics["wind_speed_RMSE_mps"] / max(
        ridge_metrics["wind_speed_RMSE_mps"], eps
    )
    wd = metrics["wind_direction_RMSE_deg"] / max(
        ridge_metrics["wind_direction_RMSE_deg"], eps
    )

    score = (
        WIND_SELECTION_WEIGHTS["U"] * u
        + WIND_SELECTION_WEIGHTS["V"] * v
        + WIND_SELECTION_WEIGHTS["WS"] * ws
        + WIND_SELECTION_WEIGHTS["WD"] * wd
    )

    return float(score), {
        "U_ratio": float(u),
        "V_ratio": float(v),
        "WS_ratio": float(ws),
        "WD_ratio": float(wd),
    }


def corrcoef_safe(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 3 or np.std(a) < EPS or np.std(b) < EPS:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def residual_diagnostics(
    y_true,
    pred_raw,
    ridge_raw,
):
    true_w = np.asarray(
        y_true[:, 0, 0:2], dtype=np.float64
    )
    pred_w = np.asarray(
        pred_raw[:, 0, 0:2], dtype=np.float64
    )
    ridge_w = np.asarray(
        ridge_raw[:, 0, 0:2], dtype=np.float64
    )

    r = true_w - ridge_w
    c = pred_w - ridge_w

    return {
        "residual_corr_U": corrcoef_safe(r[:, 0], c[:, 0]),
        "residual_corr_V": corrcoef_safe(r[:, 1], c[:, 1]),
        "true_residual_RMS_U_mps": float(
            np.sqrt(np.mean(r[:, 0] ** 2))
        ),
        "true_residual_RMS_V_mps": float(
            np.sqrt(np.mean(r[:, 1] ** 2))
        ),
        "pred_correction_RMS_U_mps": float(
            np.sqrt(np.mean(c[:, 0] ** 2))
        ),
        "pred_correction_RMS_V_mps": float(
            np.sqrt(np.mean(c[:, 1] ** 2))
        ),
    }


def evaluate_model(
    *,
    base,
    torch,
    model,
    X,
    y,
    aw,
    ridge_pred_z,
    scaler,
    residual_std_z,
    device,
    amp_enabled,
):
    model.eval()
    Xz = base.standardize_X(X, scaler)

    residual_std_z = np.asarray(
        residual_std_z, dtype=np.float32
    ).reshape(1, 2)

    chunks = []
    norm_chunks = []

    with torch.no_grad():
        for xb_np, rb_np in base.sequential_batches(
            Xz,
            ridge_pred_z,
            batch_size=FREEZE.batch_size,
        ):
            xb = torch.from_numpy(xb_np).to(
                device, non_blocking=True
            )
            rb = torch.from_numpy(
                rb_np[:, 0, 0:2]
            ).to(device, non_blocking=True)

            with amp_context(torch, amp_enabled, device):
                pred_norm = model(xb, rb)

            norm_chunks.append(
                pred_norm.detach().float().cpu().numpy()
            )

    pred_norm = np.concatenate(norm_chunks, axis=0)
    correction_z = pred_norm * residual_std_z

    pred_z = np.asarray(
        ridge_pred_z, dtype=np.float32
    ).copy()
    pred_z[:, 0, 0:2] += correction_z

    pred_raw = base.inverse_y(pred_z, scaler)
    ridge_raw = base.inverse_y(ridge_pred_z, scaler)

    metrics = base.evaluate_raw(y, pred_raw, aw)
    metrics.update(
        residual_diagnostics(y, pred_raw, ridge_raw)
    )
    metrics["pred_residual_norm_RMS"] = float(
        np.sqrt(np.mean(pred_norm ** 2))
    )

    return metrics, pred_raw


def train_one_run(
    *,
    base,
    torch,
    nn,
    candidate_name,
    candidate_cfg,
    seed,
    held_out_mission,
    X_train,
    y_train,
    aw_train,
    X_val,
    y_val,
    aw_val,
    ridge_train_z,
    ridge_val_z,
    ridge_val_metrics,
    scaler,
    device,
    output_dir,
    debug_fast,
):
    seed_everything(torch, seed)

    mode = str(candidate_cfg["mode"])
    ModelClass = build_model_class(
        torch, nn, len(base.FEATURE_NAMES), mode
    )
    model = ModelClass().to(device)
    parameter_count = count_parameters(model)

    X_train_z = base.standardize_X(X_train, scaler)
    y_train_z = base.standardize_y(y_train, scaler)

    train_residual_z = (
        y_train_z[:, 0, 0:2]
        - ridge_train_z[:, 0, 0:2]
    ).astype(np.float32)

    residual_std_z = np.std(
        train_residual_z.astype(np.float64),
        axis=0,
        ddof=0,
    )
    residual_std_z = np.where(
        residual_std_z < FREEZE.residual_std_floor,
        1.0,
        residual_std_z,
    ).astype(np.float32)

    target_residual_norm = (
        train_residual_z / residual_std_z[None, :]
    ).astype(np.float32)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FREEZE.learning_rate,
        weight_decay=FREEZE.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=FREEZE.scheduler_factor,
        patience=FREEZE.scheduler_patience,
        min_lr=FREEZE.min_learning_rate,
    )

    amp_enabled = bool(
        FREEZE.use_amp and device.type == "cuda"
    )
    grad_scaler = create_grad_scaler(
        torch, amp_enabled
    )

    # Epoch 0 = exact Ridge.
    initial_metrics, _ = evaluate_model(
        base=base,
        torch=torch,
        model=model,
        X=X_val,
        y=y_val,
        aw=aw_val,
        ridge_pred_z=ridge_val_z,
        scaler=scaler,
        residual_std_z=residual_std_z,
        device=device,
        amp_enabled=amp_enabled,
    )
    best_score, best_parts = wind_selection_score(
        initial_metrics, ridge_val_metrics
    )
    best_epoch = 0
    best_state = state_dict_cpu(model)
    best_metrics = dict(initial_metrics)
    best_parts_saved = dict(best_parts)

    history = [{
        "candidate": candidate_name,
        "held_out_mission": held_out_mission,
        "seed": int(seed),
        "epoch_one_based": 0,
        "learning_rate": float(FREEZE.learning_rate),
        "selection_score": float(best_score),
        "checkpoint_improved": True,
        "train_residual_loss": np.nan,
        **{f"val_{k}": v for k, v in initial_metrics.items()},
        **{f"score_{k}": v for k, v in best_parts.items()},
    }]

    max_epochs = 8 if debug_fast else FREEZE.max_epochs
    patience_limit = 4 if debug_fast else FREEZE.early_stop_patience
    patience = 0
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        sum_total = 0.0
        sum_mse = 0.0
        sum_penalty = 0.0
        n_seen = 0

        for xb_np, tb_np, rb_np in base.sequential_batches(
            X_train_z,
            target_residual_norm,
            ridge_train_z,
            batch_size=FREEZE.batch_size,
        ):
            xb = torch.from_numpy(xb_np).to(
                device, non_blocking=True
            )
            tb = torch.from_numpy(tb_np).to(
                device, non_blocking=True
            )
            rb = torch.from_numpy(
                rb_np[:, 0, 0:2]
            ).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp_context(torch, amp_enabled, device):
                pred_norm = model(xb, rb)
                mse = torch.mean((pred_norm - tb) ** 2)
                penalty = torch.mean(pred_norm ** 2)
                total = (
                    mse
                    + FREEZE.correction_penalty * penalty
                )

            if not torch.isfinite(total):
                raise FloatingPointError(
                    "Non-finite residual-specialist loss."
                )

            if grad_scaler is not None:
                grad_scaler.scale(total).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    FREEZE.grad_clip_norm,
                )
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    FREEZE.grad_clip_norm,
                )
                optimizer.step()

            bn = len(xb_np)
            sum_total += float(
                total.detach().float().cpu().item()
            ) * bn
            sum_mse += float(
                mse.detach().float().cpu().item()
            ) * bn
            sum_penalty += float(
                penalty.detach().float().cpu().item()
            ) * bn
            n_seen += bn

        val_metrics, _ = evaluate_model(
            base=base,
            torch=torch,
            model=model,
            X=X_val,
            y=y_val,
            aw=aw_val,
            ridge_pred_z=ridge_val_z,
            scaler=scaler,
            residual_std_z=residual_std_z,
            device=device,
            amp_enabled=amp_enabled,
        )
        score, parts = wind_selection_score(
            val_metrics, ridge_val_metrics
        )

        improved = score < best_score - 1e-8
        if improved:
            best_score = float(score)
            best_epoch = int(epoch)
            best_state = state_dict_cpu(model)
            best_metrics = dict(val_metrics)
            best_parts_saved = dict(parts)
            patience = 0
        else:
            patience += 1

        lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(score)

        history.append({
            "candidate": candidate_name,
            "held_out_mission": held_out_mission,
            "seed": int(seed),
            "epoch_one_based": int(epoch),
            "learning_rate": lr,
            "selection_score": float(score),
            "checkpoint_improved": bool(improved),
            "train_total": sum_total / max(n_seen, 1),
            "train_residual_loss": sum_mse / max(n_seen, 1),
            "train_correction_penalty": (
                sum_penalty / max(n_seen, 1)
            ),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"score_{k}": v for k, v in parts.items()},
        })

        log(
            f"      ep={epoch:03d} | score={score:.5f} | "
            f"U={val_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={val_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={val_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={val_metrics['wind_direction_RMSE_deg']:.2f} | "
            f"corr=({val_metrics['residual_corr_U']:+.3f},"
            f"{val_metrics['residual_corr_V']:+.3f})"
            + (" *" if improved else "")
        )

        if patience >= patience_limit:
            log(
                f"      early stop after {patience} epochs "
                "without wind-score improvement."
            )
            break

    elapsed = time.perf_counter() - started

    model.load_state_dict(best_state, strict=True)
    confirmed, _ = evaluate_model(
        base=base,
        torch=torch,
        model=model,
        X=X_val,
        y=y_val,
        aw=aw_val,
        ridge_pred_z=ridge_val_z,
        scaler=scaler,
        residual_std_z=residual_std_z,
        device=device,
        amp_enabled=amp_enabled,
    )
    confirmed_score, confirmed_parts = wind_selection_score(
        confirmed, ridge_val_metrics
    )

    if abs(confirmed_score - best_score) > 1e-6:
        raise RuntimeError(
            "Best-checkpoint score re-evaluation mismatch."
        )

    histories_dir = output_dir / "histories"
    checkpoints_dir = output_dir / "checkpoints"
    histories_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    tag = (
        f"{candidate_name}"
        f"__holdout_{held_out_mission.replace(' ', '_')}"
        f"__seed_{seed}"
    )
    history_path = histories_dir / f"{tag}.csv"
    checkpoint_path = checkpoints_dir / f"{tag}.pt"

    pd.DataFrame(history).to_csv(
        history_path, index=False, encoding="utf-8-sig"
    )

    torch.save({
        "stage": "12M",
        "script_version": SCRIPT_VERSION,
        "model_name": "WindResidualSpecialist-NieData-v5",
        "candidate": candidate_name,
        "candidate_config": dict(candidate_cfg),
        "wind_selection_weights": dict(
            WIND_SELECTION_WEIGHTS
        ),
        "held_out_mission": held_out_mission,
        "seed": int(seed),
        "parameter_count": int(parameter_count),
        "best_epoch_one_based": int(best_epoch),
        "selection_score": float(confirmed_score),
        "validation_metrics": confirmed,
        "score_parts": confirmed_parts,
        "state_dict": best_state,
        "scaler": scaler,
        "residual_std_z": residual_std_z,
        "ridge_alpha": RIDGE_ALPHA,
        "feature_names": list(base.FEATURE_NAMES),
        "target_names": list(base.TARGET_NAMES),
        "Tropical_Atlantic_used": False,
        "Stage_12K_used": False,
    }, checkpoint_path)

    result = {
        "method": candidate_name,
        "held_out_mission": held_out_mission,
        "seed": int(seed),
        "status": "completed_finite",
        "parameter_count": int(parameter_count),
        "best_epoch_one_based": int(best_epoch),
        "selection_score": float(confirmed_score),
        "elapsed_seconds": float(elapsed),
        "residual_std_z_U": float(residual_std_z[0]),
        "residual_std_z_V": float(residual_std_z[1]),
        **confirmed,
        **{
            f"score_{k}": v
            for k, v in confirmed_parts.items()
        },
        "history_path": str(history_path),
        "checkpoint_path": str(checkpoint_path),
    }

    del model, optimizer, scheduler, grad_scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def append_csv(path: Path, row: dict):
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        keys = {"method", "held_out_mission", "seed"}
        if keys.issubset(old.columns):
            mask = (
                (old["method"].astype(str) == str(row["method"]))
                & (
                    old["held_out_mission"].astype(str)
                    == str(row["held_out_mission"])
                )
                & (
                    old["seed"].astype(int)
                    == int(row["seed"])
                )
            )
            old = old.loc[~mask].copy()
        new = pd.concat([old, new], ignore_index=True)

    new.to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
    )
    parser.add_argument(
        "--debug-fast",
        action="store_true",
        help=(
            "NON-SCIENTIFIC smoke: Antarctic holdout, "
            "one seed, one candidate, <=8 epochs."
        ),
    )
    args = parser.parse_args()

    project_root = args.project_root
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base, base_path = load_base_module(project_root)
    torch, nn, _ = base.import_torch()
    base.configure_cuda(torch)

    if args.force_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    paths = base.explicit_development_paths(dataset_dir)

    # Hard firewall audit.
    for mission, p in paths.items():
        if "tropical" in str(p).lower():
            raise RuntimeError(
                "FIREWALL VIOLATION: development path contains "
                f"'tropical': {mission} -> {p}"
            )

    missions = {
        m: base.load_development_mission(paths[m], m)
        for m in DEVELOPMENT_MISSIONS
    }

    active_holdouts = (
        ["Antarctic"]
        if args.debug_fast
        else DEVELOPMENT_MISSIONS
    )
    active_seeds = (
        [SCREENING_SEEDS[0]]
        if args.debug_fast
        else SCREENING_SEEDS
    )
    active_candidates = (
        {
            "V5_ResidualGRUTrend":
            CANDIDATES["V5_ResidualGRUTrend"]
        }
        if args.debug_fast
        else CANDIDATES
    )

    log("=" * 128)
    log(
        "12M — WindResidualSpecialist-NieData v5 "
        "DEVELOPMENT-ONLY LOMO"
    )
    log("=" * 128)
    log(f"dataset      : {dataset_dir}")
    log(f"output       : {output_dir}")
    log(f"base script  : {base_path}")
    log(f"device       : {device}")
    if device.type == "cuda":
        log(f"GPU          : {torch.cuda.get_device_name(0)}")
    log(f"candidates   : {list(active_candidates)}")
    log(f"seeds        : {active_seeds}")
    log(
        "[FIREWALL] Tropical Atlantic and Stage 12K "
        "are NOT read."
    )
    log("")

    run_path = output_dir / "candidate_run_results.csv"
    baseline_rows = []

    for held_out in active_holdouts:
        train_names = [
            m for m in DEVELOPMENT_MISSIONS
            if m != held_out
        ]
        train = base.concat_missions(
            [missions[m] for m in train_names]
        )
        val = missions[held_out]

        scaler = base.fit_scalers(
            train["X"], train["y"], train["aw"]
        )
        ridge = base.fit_ridge(
            train["X"], train["y"], scaler
        )
        ridge_train_z = base.ridge_predict_z(
            ridge, train["X"], scaler
        )
        ridge_val_z = base.ridge_predict_z(
            ridge, val["X"], scaler
        )
        ridge_val_raw = base.inverse_y(
            ridge_val_z, scaler
        )
        ridge_metrics = base.evaluate_raw(
            val["y"], ridge_val_raw, val["aw"]
        )

        baseline_rows.append({
            "method": "Ridge",
            "held_out_mission": held_out,
            **ridge_metrics,
        })

        log("-" * 128)
        log(
            f"[{held_out}] Ntrain={len(train['X']):,} "
            f"Nval={len(val['X']):,}"
        )
        log(
            f"  Ridge: U={ridge_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={ridge_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={ridge_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={ridge_metrics['wind_direction_RMSE_deg']:.2f}"
        )

        for candidate_name, cfg in active_candidates.items():
            for seed in active_seeds:
                log(
                    f"  [TRAIN] {candidate_name} | seed={seed}"
                )
                result = train_one_run(
                    base=base,
                    torch=torch,
                    nn=nn,
                    candidate_name=candidate_name,
                    candidate_cfg=cfg,
                    seed=seed,
                    held_out_mission=held_out,
                    X_train=train["X"],
                    y_train=train["y"],
                    aw_train=train["aw"],
                    X_val=val["X"],
                    y_val=val["y"],
                    aw_val=val["aw"],
                    ridge_train_z=ridge_train_z,
                    ridge_val_z=ridge_val_z,
                    ridge_val_metrics=ridge_metrics,
                    scaler=scaler,
                    device=device,
                    output_dir=output_dir,
                    debug_fast=args.debug_fast,
                )
                append_csv(run_path, result)
                log(
                    f"  [DONE] score={result['selection_score']:.5f} | "
                    f"best_ep={result['best_epoch_one_based']} | "
                    f"U={result['wind_U_RMSE_mps']:.4f} | "
                    f"V={result['wind_V_RMSE_mps']:.4f} | "
                    f"corr=({result['residual_corr_U']:+.3f},"
                    f"{result['residual_corr_V']:+.3f})"
                )

        del (
            train, val, scaler, ridge,
            ridge_train_z, ridge_val_z,
            ridge_val_raw,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(
        output_dir / "baseline_fold_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if args.debug_fast:
        log("")
        log(
            "[DONE] debug-fast completed. "
            "No scientific candidate freeze was written."
        )
        return 0

    runs = pd.read_csv(run_path)

    expected_per_candidate = (
        len(DEVELOPMENT_MISSIONS)
        * len(SCREENING_SEEDS)
    )

    summary_rows = []
    for candidate_name, cfg in CANDIDATES.items():
        sub = runs.loc[
            (runs["method"].astype(str) == candidate_name)
            & (
                runs["status"].astype(str)
                == "completed_finite"
            )
        ].copy()

        if len(sub) != expected_per_candidate:
            raise RuntimeError(
                f"{candidate_name}: expected "
                f"{expected_per_candidate} completed runs, "
                f"found {len(sub)}."
            )

        row = {
            "candidate": candidate_name,
            "runs": int(len(sub)),
            "selection_score_mean": float(
                sub["selection_score"].mean()
            ),
            "selection_score_std": float(
                sub["selection_score"].std(ddof=0)
            ),
            "parameter_count": int(
                round(sub["parameter_count"].mean())
            ),
            "best_epoch_median": float(
                np.median(
                    sub["best_epoch_one_based"].to_numpy()
                )
            ),
        }

        for col in [
            "wind_U_RMSE_mps",
            "wind_V_RMSE_mps",
            "wind_vector_RMSE_mps",
            "wind_speed_RMSE_mps",
            "wind_direction_RMSE_deg",
            "AW_vector_RMSE_mps",
            "residual_corr_U",
            "residual_corr_V",
            "pred_correction_RMS_U_mps",
            "pred_correction_RMS_V_mps",
        ]:
            vals = sub[col].to_numpy(dtype=float)
            row[f"{col}_mean"] = float(
                np.nanmean(vals)
            )
            row[f"{col}_std"] = float(
                np.nanstd(vals, ddof=0)
            )

        row["mode"] = str(cfg["mode"])
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["selection_score_mean", "parameter_count"],
        ascending=[True, True],
    ).reset_index(drop=True)

    summary_path = output_dir / "candidate_summary.csv"
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    selected_name = str(
        summary.iloc[0]["candidate"]
    )
    selected_cfg = dict(CANDIDATES[selected_name])

    selected_runs = runs.loc[
        runs["method"].astype(str) == selected_name
    ].copy()

    selected_runs_path = (
        output_dir / "selected_candidate_lomo_runs.csv"
    )
    selected_runs.to_csv(
        selected_runs_path,
        index=False,
        encoding="utf-8-sig",
    )

    ridge_means = {
        k: float(baseline_df[k].mean())
        for k in [
            "wind_U_RMSE_mps",
            "wind_V_RMSE_mps",
            "wind_vector_RMSE_mps",
            "wind_speed_RMSE_mps",
            "wind_direction_RMSE_deg",
        ]
    }

    selected_means = {
        k: float(selected_runs[k].mean())
        for k in ridge_means
    }

    relative_improvement = {
        k: float(
            (ridge_means[k] - selected_means[k])
            / max(ridge_means[k], EPS)
        )
        for k in ridge_means
    }

    selected_payload = {
        "stage": "12M",
        "script_version": SCRIPT_VERSION,
        "model_name": "WindResidualSpecialist-NieData-v5",
        "selected_candidate": selected_name,
        "selected_config": selected_cfg,
        "train_freeze": asdict(FREEZE),
        "wind_selection_weights": dict(
            WIND_SELECTION_WEIGHTS
        ),
        "development_missions": DEVELOPMENT_MISSIONS,
        "screening_seeds": SCREENING_SEEDS,
        "ridge_alpha": RIDGE_ALPHA,
        "ridge_development_means": ridge_means,
        "selected_development_means": selected_means,
        "relative_improvement_vs_ridge": relative_improvement,
        "Tropical_Atlantic_used": False,
        "Stage_12K_used": False,
        "next_stage_policy": (
            "Only if the selected wind residual specialist shows "
            "stable development-only improvement over Ridge and "
            "positive residual correlations should it be integrated "
            "with the strong v4 vessel/apparent-wind branch. "
            "Do not tune using Tropical Atlantic."
        ),
    }
    selected_json_path = (
        output_dir / "selected_config.json"
    )
    save_json(
        selected_json_path, selected_payload
    )

    report_path = output_dir / "12M_REPORT.txt"
    with report_path.open(
        "w", encoding="utf-8"
    ) as f:
        f.write(
            "12M WindResidualSpecialist-NieData v5 "
            "Development-Only LOMO\n"
        )
        f.write("=" * 128 + "\n\n")
        f.write("FIREWALL\n")
        f.write("-" * 128 + "\n")
        f.write(
            "Development missions: Antarctic, Atlantic, West Coast\n"
        )
        f.write("Tropical Atlantic accessed: NO\n")
        f.write("Stage 12K accessed: NO\n\n")

        f.write("CANDIDATE SUMMARY\n")
        f.write("-" * 128 + "\n")
        f.write(summary.to_string(index=False))
        f.write("\n\n")

        f.write("SELECTED\n")
        f.write("-" * 128 + "\n")
        f.write(f"candidate: {selected_name}\n")
        f.write(
            f"configuration: {json.dumps(selected_cfg, ensure_ascii=False)}\n"
        )
        f.write(
            f"selection score mean: "
            f"{float(summary.iloc[0]['selection_score_mean']):.6f}\n"
        )
        f.write("\nDevelopment relative improvement vs Ridge:\n")
        for k, v in relative_improvement.items():
            f.write(f"  {k}: {100.0*v:+.4f}%\n")

        f.write("\nResidual correlations at selected checkpoints:\n")
        f.write(
            f"  U mean: "
            f"{selected_runs['residual_corr_U'].mean():+.6f}\n"
        )
        f.write(
            f"  V mean: "
            f"{selected_runs['residual_corr_V'].mean():+.6f}\n"
        )

    log("")
    log("=" * 128)
    log("12M CANDIDATE SUMMARY")
    log("=" * 128)
    log(summary.to_string(index=False))
    log("")
    log(f"[SELECTED] {selected_name}")
    log(
        "Development relative improvement vs Ridge: "
        + " | ".join(
            f"{k}={100.0*v:+.3f}%"
            for k, v in relative_improvement.items()
        )
    )
    log(
        "Selected residual corr mean: "
        f"U={selected_runs['residual_corr_U'].mean():+.4f} | "
        f"V={selected_runs['residual_corr_V'].mean():+.4f}"
    )
    log("")
    log(f"[SAVED] {summary_path}")
    log(f"[SAVED] {selected_runs_path}")
    log(f"[SAVED] {selected_json_path}")
    log(f"[SAVED] {report_path}")
    log("")
    log(
        "NEXT DECISION: integrate with the v4 vessel/AW branch only "
        "if the selected specialist improves development wind metrics "
        "and its residual correlations are consistently positive."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n[FATAL ERROR]", flush=True)
        traceback.print_exc()
        sys.exit(1)
