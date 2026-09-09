# -*- coding: utf-8 -*-
r"""
12J_train_PhysicsCompact_NieData_v4_RidgeResidual_LOMO.py

Development-only model selection for a compact Ridge-residual wind-vessel model
on the Stage-12G point-sampled dataset.

Core model
----------
    W_hat_z = W_ridge_z + g_W * DeltaW_z
    V_hat_z = V_ridge_z + g_V * DeltaV_z
    A_hat   = W_hat - V_hat

The GRU encodes the six 10-min-spaced historical observations.  Unlike the old
v1 implementation, the decoder also receives the fold-specific Ridge anchor
forecast explicitly:
    c = [h_GRU ; vec(Y_ridge)]

The model remains small (~20k trainable parameters, depending only on the
fixed 64-unit architecture).

Scientific selection policy
---------------------------
This script reads ONLY:
    Antarctic.npz
    Atlantic.npz
    West_Coast.npz

It does not read Tropical_Atlantic_TEST.npz.

Four predeclared loss configurations are screened with the same three LOMO
missions and three seeds.  Checkpoints and the final candidate are selected by
a development-only composite score dominated by wind metrics but retaining
vessel/apparent-wind performance:

    WindScore =
        0.25 * U_RMSE / Ridge_U_RMSE
      + 0.35 * V_RMSE / Ridge_V_RMSE
      + 0.25 * WS_RMSE / Ridge_WS_RMSE
      + 0.15 * WD_RMSE / Ridge_WD_RMSE

    SelectionScore =
        0.75 * WindScore
      + 0.15 * AW_RMSE / Ridge_AW_RMSE
      + 0.10 * Vessel_RMSE / Ridge_Vessel_RMSE

Lower is better.

The untrained model is evaluated before epoch 1.  Because residual heads are
initialized at zero, this initial checkpoint equals Ridge exactly.  Therefore,
a LOMO run can retain the Ridge-equivalent state when neural correction does
not improve the development-only selection score.

Usage
-----
python "D:\project\WindPredict_SaildroneData\src\12J_train_PhysicsCompact_NieData_v4_RidgeResidual_LOMO.py" ^
  --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" ^
  --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12J_PhysicsCompact_NieData_v4_RidgeResidual_LOMO_point_v0_1"

Optional smoke test:
    --debug-fast
"""

from __future__ import annotations

import argparse
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


SCRIPT_VERSION = "0.1.0-PhysicsCompact-NieData-v4-RidgeResidual-LOMO"

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
    / "12J_PhysicsCompact_NieData_v4_RidgeResidual_LOMO_point_v0_1"
)

DEVELOPMENT_MISSIONS = ["Antarctic", "Atlantic", "West Coast"]
SCREENING_SEEDS = [500043, 501052, 502061]
FINAL_SEEDS = [500043, 501052, 502061, 503070, 504079]

RIDGE_ALPHA = 1.0


@dataclass(frozen=True)
class TrainFreeze:
    gru_hidden: int = 64
    gru_layers: int = 1
    dense_hidden: int = 64
    dropout: float = 0.10

    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 100
    early_stop_patience: int = 15
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-5
    grad_clip_norm: float = 1.0

    use_amp: bool = True
    train_shuffle: bool = False

    wind_gate_bias: float = -1.5
    vessel_gate_bias: float = -1.0


FREEZE = TrainFreeze()


# Predeclared before development LOMO execution.
CANDIDATES = {
    "V4_JointBalanced": {
        "lambda_wind": 1.00,
        "lambda_vessel": 1.00,
        "lambda_aw": 1.00,
        "lambda_ws": 0.25,
        "lambda_dir": 0.10,
        "lambda_residual": 0.005,
        "v_component_weight": 1.00,
    },
    "V4_WindBalanced": {
        "lambda_wind": 1.50,
        "lambda_vessel": 0.75,
        "lambda_aw": 0.75,
        "lambda_ws": 0.50,
        "lambda_dir": 0.20,
        "lambda_residual": 0.005,
        "v_component_weight": 1.00,
    },
    "V4_WindDirection": {
        "lambda_wind": 1.50,
        "lambda_vessel": 0.75,
        "lambda_aw": 0.50,
        "lambda_ws": 0.50,
        "lambda_dir": 0.35,
        "lambda_residual": 0.003,
        "v_component_weight": 1.00,
    },
    "V4_VComponent": {
        "lambda_wind": 1.50,
        "lambda_vessel": 0.75,
        "lambda_aw": 0.75,
        "lambda_ws": 0.50,
        "lambda_dir": 0.25,
        "lambda_residual": 0.003,
        "v_component_weight": 1.25,
    },
}

SELECTION_WEIGHTS = {
    "wind_U": 0.25,
    "wind_V": 0.35,
    "wind_speed": 0.25,
    "wind_direction": 0.15,
    "wind_total": 0.75,
    "apparent": 0.15,
    "vessel": 0.10,
}


def log(msg=""):
    print(msg, flush=True)


def load_base_module(project_root):
    import importlib.util
    import sys
    from pathlib import Path

    project_root = Path(project_root)
    base_path = project_root / "src" / "12C_train_PhysicsCompact_NieData_LOMO.py"

    if not base_path.exists():
        raise FileNotFoundError(
            f"Base PhysicsCompact module not found:\n{base_path}"
        )

    module_name = "physicscompact_niedata_lomo_base"

    # Reuse if already imported.
    if module_name in sys.modules:
        return sys.modules[module_name], base_path

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(base_path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Failed to create import specification for:\n{base_path}"
        )

    mod = importlib.util.module_from_spec(spec)

    # Required for @dataclass and some type/introspection utilities.
    sys.modules[module_name] = mod

    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return mod, base_path


def save_json(path: Path, obj):
    def cv(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return None if np.isnan(x) else float(x)
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


def build_model_class(torch, nn, feature_count: int):
    class PhysicsCompactNieDataV4(nn.Module):
        def __init__(self):
            super().__init__()

            self.gru = nn.GRU(
                input_size=feature_count,
                hidden_size=FREEZE.gru_hidden,
                num_layers=FREEZE.gru_layers,
                batch_first=True,
                dropout=FREEZE.dropout if FREEZE.gru_layers > 1 else 0.0,
            )

            # Explicit Ridge-anchor context: 4 standardized target values.
            self.shared = nn.Sequential(
                nn.Linear(FREEZE.gru_hidden + 4, FREEZE.dense_hidden),
                nn.GELU(),
                nn.Dropout(FREEZE.dropout),
            )

            self.wind_delta = nn.Linear(FREEZE.dense_hidden, 2)
            self.wind_gate = nn.Linear(FREEZE.dense_hidden, 2)
            self.vessel_delta = nn.Linear(FREEZE.dense_hidden, 2)
            self.vessel_gate = nn.Linear(FREEZE.dense_hidden, 2)

            # Exact Ridge start.
            nn.init.zeros_(self.wind_delta.weight)
            nn.init.zeros_(self.wind_delta.bias)
            nn.init.zeros_(self.vessel_delta.weight)
            nn.init.zeros_(self.vessel_delta.bias)

            nn.init.constant_(self.wind_gate.bias, FREEZE.wind_gate_bias)
            nn.init.constant_(self.vessel_gate.bias, FREEZE.vessel_gate_bias)

        def forward(self, x, ridge_pred_z):
            seq, _ = self.gru(x)
            h = seq[:, -1, :]
            ridge_context = ridge_pred_z[:, 0, :]
            z = self.shared(torch.cat([h, ridge_context], dim=1))

            dw = self.wind_delta(z)
            gw = torch.sigmoid(self.wind_gate(z))
            dv = self.vessel_delta(z)
            gv = torch.sigmoid(self.vessel_gate(z))

            pred_w = ridge_pred_z[:, 0, 0:2] + gw * dw
            pred_v = ridge_pred_z[:, 0, 2:4] + gv * dv
            pred = torch.cat([pred_w, pred_v], dim=1)[:, None, :]

            return {
                "pred_z": pred,
                "wind_delta_z": dw,
                "wind_gate": gw,
                "vessel_delta_z": dv,
                "vessel_gate": gv,
                "wind_correction_z": gw * dw,
                "vessel_correction_z": gv * dv,
            }

    return PhysicsCompactNieDataV4


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def make_constants(torch, scaler, y_train, device):
    y_mean = torch.as_tensor(
        scaler["y_mean"], dtype=torch.float32, device=device
    )
    y_std = torch.as_tensor(
        scaler["y_std"], dtype=torch.float32, device=device
    )
    aw_std = torch.as_tensor(
        scaler["aw_std"], dtype=torch.float32, device=device
    )

    w = np.asarray(y_train[:, 0, 0:2], dtype=np.float64)
    ws = np.linalg.norm(w, axis=1)
    ws_std = float(np.std(ws))
    if not np.isfinite(ws_std) or ws_std < 1e-6:
        ws_std = 1.0

    return {
        "y_mean": y_mean,
        "y_std": y_std,
        "aw_std": aw_std,
        "ws_std": torch.as_tensor(ws_std, dtype=torch.float32, device=device),
    }


def compute_loss(torch, out, y_true_z, aw_true_raw, const, cfg):
    pred_z = out["pred_z"]

    # Component loss, with optional extra emphasis on the northward V component.
    wind_err = pred_z[:, :, 0:2] - y_true_z[:, :, 0:2]
    u_mse = torch.mean(wind_err[:, :, 0] ** 2)
    v_mse = torch.mean(wind_err[:, :, 1] ** 2)
    wind_mse = 0.5 * (
        u_mse + float(cfg["v_component_weight"]) * v_mse
    )

    vessel_mse = torch.mean(
        (pred_z[:, :, 2:4] - y_true_z[:, :, 2:4]) ** 2
    )

    y_mean = const["y_mean"][None, None, :]
    y_std = const["y_std"][None, None, :]

    pred_raw = pred_z * y_std + y_mean
    true_raw = y_true_z * y_std + y_mean

    pred_w = pred_raw[:, :, 0:2]
    true_w = true_raw[:, :, 0:2]
    pred_v = pred_raw[:, :, 2:4]

    aw_pred = pred_w - pred_v
    aw_scale = const["aw_std"][None, None, :].clamp_min(1e-6)
    aw_loss = torch.mean(((aw_pred - aw_true_raw) / aw_scale) ** 2)

    pred_ws = torch.linalg.vector_norm(pred_w, dim=-1)
    true_ws = torch.linalg.vector_norm(true_w, dim=-1)
    ws_loss = torch.mean(
        ((pred_ws - true_ws) / const["ws_std"].clamp_min(1e-6)) ** 2
    )

    # Differentiable directional loss: 1 - cos(delta angle).
    # Use only non-calm true winds, matching the manuscript direction-mask idea.
    pred_norm = torch.linalg.vector_norm(pred_w, dim=-1).clamp_min(1e-6)
    true_norm = torch.linalg.vector_norm(true_w, dim=-1).clamp_min(1e-6)
    cosine = torch.sum(pred_w * true_w, dim=-1) / (pred_norm * true_norm)
    cosine = torch.clamp(cosine, -1.0, 1.0)
    valid_dir = true_norm >= 0.5
    if torch.any(valid_dir):
        dir_loss = torch.mean(1.0 - cosine[valid_dir])
    else:
        dir_loss = torch.zeros((), dtype=pred_z.dtype, device=pred_z.device)

    residual = (
        torch.mean(out["wind_correction_z"] ** 2)
        + torch.mean(out["vessel_correction_z"] ** 2)
    )

    total = (
        float(cfg["lambda_wind"]) * wind_mse
        + float(cfg["lambda_vessel"]) * vessel_mse
        + float(cfg["lambda_aw"]) * aw_loss
        + float(cfg["lambda_ws"]) * ws_loss
        + float(cfg["lambda_dir"]) * dir_loss
        + float(cfg["lambda_residual"]) * residual
    )

    return {
        "total": total,
        "wind_mse_z": wind_mse,
        "wind_U_mse_z": u_mse,
        "wind_V_mse_z": v_mse,
        "vessel_mse_z": vessel_mse,
        "aw_loss_norm": aw_loss,
        "ws_loss_norm": ws_loss,
        "direction_cosine_loss": dir_loss,
        "residual_penalty": residual,
    }


def selection_score(metrics, ridge_metrics):
    eps = 1e-12

    u_ratio = metrics["wind_U_RMSE_mps"] / max(
        ridge_metrics["wind_U_RMSE_mps"], eps
    )
    v_ratio = metrics["wind_V_RMSE_mps"] / max(
        ridge_metrics["wind_V_RMSE_mps"], eps
    )
    ws_ratio = metrics["wind_speed_RMSE_mps"] / max(
        ridge_metrics["wind_speed_RMSE_mps"], eps
    )
    wd_ratio = metrics["wind_direction_RMSE_deg"] / max(
        ridge_metrics["wind_direction_RMSE_deg"], eps
    )

    wind_score = (
        SELECTION_WEIGHTS["wind_U"] * u_ratio
        + SELECTION_WEIGHTS["wind_V"] * v_ratio
        + SELECTION_WEIGHTS["wind_speed"] * ws_ratio
        + SELECTION_WEIGHTS["wind_direction"] * wd_ratio
    )

    aw_ratio = metrics["AW_vector_RMSE_mps"] / max(
        ridge_metrics["AW_vector_RMSE_mps"], eps
    )
    vessel_ratio = metrics["vessel_vector_RMSE_mps"] / max(
        ridge_metrics["vessel_vector_RMSE_mps"], eps
    )

    score = (
        SELECTION_WEIGHTS["wind_total"] * wind_score
        + SELECTION_WEIGHTS["apparent"] * aw_ratio
        + SELECTION_WEIGHTS["vessel"] * vessel_ratio
    )

    return float(score), {
        "wind_score": float(wind_score),
        "U_ratio": float(u_ratio),
        "V_ratio": float(v_ratio),
        "WS_ratio": float(ws_ratio),
        "WD_ratio": float(wd_ratio),
        "AW_ratio": float(aw_ratio),
        "vessel_ratio": float(vessel_ratio),
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
    device,
    amp_enabled,
):
    model.eval()
    Xz = base.standardize_X(X, scaler)

    pred_chunks = []
    wg, vg, wc, vc = [], [], [], []

    with torch.no_grad():
        for xb_np, rb_np in base.sequential_batches(
            Xz,
            ridge_pred_z,
            batch_size=FREEZE.batch_size,
        ):
            xb = torch.from_numpy(xb_np).to(device, non_blocking=True)
            rb = torch.from_numpy(rb_np).to(device, non_blocking=True)

            with base.autocast_context(torch, amp_enabled):
                out = model(xb, rb)

            pred_chunks.append(
                out["pred_z"].detach().float().cpu().numpy()
            )
            wg.append(out["wind_gate"].detach().float().cpu().numpy())
            vg.append(out["vessel_gate"].detach().float().cpu().numpy())
            wc.append(
                out["wind_correction_z"].detach().float().cpu().numpy()
            )
            vc.append(
                out["vessel_correction_z"].detach().float().cpu().numpy()
            )

    pred_z = np.concatenate(pred_chunks, axis=0)
    pred_raw = base.inverse_y(pred_z, scaler)
    metrics = base.evaluate_raw(y, pred_raw, aw)

    metrics.update({
        "wind_gate_mean": float(np.mean(np.concatenate(wg, axis=0))),
        "vessel_gate_mean": float(np.mean(np.concatenate(vg, axis=0))),
        "wind_correction_z_RMS": float(
            np.sqrt(np.mean(np.concatenate(wc, axis=0) ** 2))
        ),
        "vessel_correction_z_RMS": float(
            np.sqrt(np.mean(np.concatenate(vc, axis=0) ** 2))
        ),
    })

    return metrics, pred_raw


def train_one_run(
    *,
    base,
    torch,
    nn,
    candidate_name,
    cfg,
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

    ModelClass = build_model_class(torch, nn, len(base.FEATURE_NAMES))
    model = ModelClass().to(device)
    parameter_count = count_parameters(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=FREEZE.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=FREEZE.scheduler_factor,
        patience=FREEZE.scheduler_patience,
        min_lr=FREEZE.min_learning_rate,
    )

    amp_enabled = bool(FREEZE.use_amp and device.type == "cuda")
    grad_scaler = base.create_grad_scaler(torch, amp_enabled)

    X_train_z = base.standardize_X(X_train, scaler)
    y_train_z = base.standardize_y(y_train, scaler)

    const = make_constants(torch, scaler, y_train, device)

    max_epochs = 8 if debug_fast else FREEZE.max_epochs
    patience_limit = 4 if debug_fast else FREEZE.early_stop_patience

    # Epoch 0 = exact Ridge. This is a real candidate checkpoint.
    initial_metrics, _ = evaluate_model(
        base=base,
        torch=torch,
        model=model,
        X=X_val,
        y=y_val,
        aw=aw_val,
        ridge_pred_z=ridge_val_z,
        scaler=scaler,
        device=device,
        amp_enabled=amp_enabled,
    )
    best_score, best_parts = selection_score(
        initial_metrics, ridge_val_metrics
    )
    best_epoch = 0
    best_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }
    best_metrics = initial_metrics
    best_score_parts = best_parts

    history_rows = [{
        "candidate": candidate_name,
        "held_out_mission": held_out_mission,
        "seed": int(seed),
        "epoch_one_based": 0,
        "learning_rate": float(FREEZE.learning_rate),
        "selection_score": best_score,
        "checkpoint_improved": True,
        **{f"val_{k}": v for k, v in initial_metrics.items()},
        **{f"score_{k}": v for k, v in best_parts.items()},
    }]

    patience = 0
    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        sums = {}
        n_seen = 0

        for xb_np, yb_np, awb_np, rb_np in base.sequential_batches(
            X_train_z,
            y_train_z,
            aw_train,
            ridge_train_z,
            batch_size=FREEZE.batch_size,
        ):
            xb = torch.from_numpy(xb_np).to(device, non_blocking=True)
            yb = torch.from_numpy(yb_np).to(device, non_blocking=True)
            awb = torch.from_numpy(awb_np).to(device, non_blocking=True)
            rb = torch.from_numpy(rb_np).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with base.autocast_context(torch, amp_enabled):
                out = model(xb, rb)
                losses = compute_loss(
                    torch, out, yb, awb, const, cfg
                )

            total = losses["total"]
            if not torch.isfinite(total):
                raise FloatingPointError("Non-finite v4 training loss.")

            if grad_scaler is not None:
                grad_scaler.scale(total).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), FREEZE.grad_clip_norm
                )
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), FREEZE.grad_clip_norm
                )
                optimizer.step()

            bn = len(xb_np)
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(
                    value.detach().float().cpu().item()
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
            device=device,
            amp_enabled=amp_enabled,
        )
        score, score_parts = selection_score(
            val_metrics, ridge_val_metrics
        )

        improved = score < best_score - 1e-8
        if improved:
            best_score = score
            best_epoch = int(epoch)
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_metrics = val_metrics
            best_score_parts = score_parts
            patience = 0
        else:
            patience += 1

        lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(score)

        row = {
            "candidate": candidate_name,
            "held_out_mission": held_out_mission,
            "seed": int(seed),
            "epoch_one_based": int(epoch),
            "learning_rate": lr,
            "selection_score": float(score),
            "checkpoint_improved": bool(improved),
        }
        row.update({
            f"train_{k}": v / max(n_seen, 1)
            for k, v in sums.items()
        })
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row.update({f"score_{k}": v for k, v in score_parts.items()})
        history_rows.append(row)

        log(
            f"      ep={epoch:03d} | score={score:.4f} | "
            f"U={val_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={val_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={val_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={val_metrics['wind_direction_RMSE_deg']:.2f} | "
            f"AW={val_metrics['AW_vector_RMSE_mps']:.4f}"
            + (" *" if improved else "")
        )

        if patience >= patience_limit:
            log(
                f"      early stop after {patience} epochs "
                "without composite-score improvement."
            )
            break

    elapsed = time.perf_counter() - start_time

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
        device=device,
        amp_enabled=amp_enabled,
    )
    confirmed_score, confirmed_parts = selection_score(
        confirmed, ridge_val_metrics
    )

    if abs(confirmed_score - best_score) > 1e-6:
        raise RuntimeError("Best-checkpoint score re-evaluation mismatch.")

    histories_dir = output_dir / "histories"
    checkpoints_dir = output_dir / "checkpoints"
    histories_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    tag = (
        f"{candidate_name}__holdout_{held_out_mission.replace(' ', '_')}"
        f"__seed_{seed}"
    )
    history_path = histories_dir / f"{tag}.csv"
    checkpoint_path = checkpoints_dir / f"{tag}.pt"

    pd.DataFrame(history_rows).to_csv(
        history_path, index=False, encoding="utf-8-sig"
    )

    torch.save({
        "stage": "12J",
        "script_version": SCRIPT_VERSION,
        "model_name": "Physics-Compact-NieData v4",
        "candidate": candidate_name,
        "candidate_config": dict(cfg),
        "selection_weights": dict(SELECTION_WEIGHTS),
        "held_out_mission": held_out_mission,
        "seed": int(seed),
        "parameter_count": int(parameter_count),
        "best_epoch_one_based": int(best_epoch),
        "selection_score": float(confirmed_score),
        "validation_metrics": confirmed,
        "score_parts": confirmed_parts,
        "state_dict": best_state,
        "scaler": scaler,
        "ridge_alpha": RIDGE_ALPHA,
        "feature_names": list(base.FEATURE_NAMES),
        "target_names": list(base.TARGET_NAMES),
        "Tropical_Atlantic_used": False,
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
        **confirmed,
        **{f"score_{k}": v for k, v in confirmed_parts.items()},
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
        if {"method", "held_out_mission", "seed"}.issubset(old.columns):
            mask = (
                (old["method"].astype(str) == str(row["method"]))
                & (
                    old["held_out_mission"].astype(str)
                    == str(row["held_out_mission"])
                )
                & (old["seed"].astype(int) == int(row["seed"]))
            )
            old = old.loc[~mask]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=DEFAULT_PROJECT_ROOT
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=None
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None
    )
    parser.add_argument(
        "--force-cpu", action="store_true"
    )
    parser.add_argument(
        "--debug-fast", action="store_true"
    )
    args = parser.parse_args()

    project_root = args.project_root
    dataset_dir = args.dataset_dir or (
        project_root
        / "data"
        / "forecasting"
        / "12G_Nie_point_sampled_benchmark_v0_1"
        / "dataset"
    )
    output_dir = args.output_dir or (
        project_root
        / "data"
        / "forecasting"
        / "12J_PhysicsCompact_NieData_v4_RidgeResidual_LOMO_point_v0_1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    base, base_path = load_base_module(project_root)
    torch, nn, _ = base.import_torch()
    base.configure_cuda(torch)

    if args.force_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    paths = base.explicit_development_paths(dataset_dir)
    missions = {
        m: base.load_development_mission(paths[m], m)
        for m in DEVELOPMENT_MISSIONS
    }

    log("=" * 124)
    log("12J — Physics-Compact-NieData v4 DEVELOPMENT-ONLY LOMO")
    log("=" * 124)
    log(f"dataset      : {dataset_dir}")
    log(f"output       : {output_dir}")
    log(f"base script  : {base_path}")
    log(f"device       : {device}")
    log(f"candidates   : {list(CANDIDATES)}")
    log("[FIREWALL] Tropical Atlantic is not read by this script.")
    log("")

    baseline_rows = []
    run_path = output_dir / "candidate_run_results.csv"

    active_candidates = (
        {"V4_WindBalanced": CANDIDATES["V4_WindBalanced"]}
        if args.debug_fast
        else CANDIDATES
    )
    active_seeds = [SCREENING_SEEDS[0]] if args.debug_fast else SCREENING_SEEDS

    for held_out in DEVELOPMENT_MISSIONS:
        train_names = [m for m in DEVELOPMENT_MISSIONS if m != held_out]
        train = base.concat_missions([missions[m] for m in train_names])
        val = missions[held_out]

        scaler = base.fit_scalers(
            train["X"], train["y"], train["aw"]
        )
        ridge = base.fit_ridge(train["X"], train["y"], scaler)
        ridge_train_z = base.ridge_predict_z(
            ridge, train["X"], scaler
        )
        ridge_val_z = base.ridge_predict_z(
            ridge, val["X"], scaler
        )
        ridge_val_raw = base.inverse_y(ridge_val_z, scaler)

        p_y, _ = base.persistence_predict(val["X"])
        p_metrics = base.evaluate_raw(val["y"], p_y, val["aw"])
        r_metrics = base.evaluate_raw(
            val["y"], ridge_val_raw, val["aw"]
        )

        for name, metrics in [
            ("Persistence", p_metrics),
            ("Ridge", r_metrics),
        ]:
            baseline_rows.append({
                "method": name,
                "held_out_mission": held_out,
                **metrics,
            })

        log("-" * 124)
        log(
            f"[{held_out}] Ntrain={len(train['X']):,} "
            f"Nval={len(val['X']):,}"
        )
        log(
            f"  Ridge: U={r_metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={r_metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={r_metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={r_metrics['wind_direction_RMSE_deg']:.2f} | "
            f"AW={r_metrics['AW_vector_RMSE_mps']:.4f}"
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
                    cfg=cfg,
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
                    ridge_val_metrics=r_metrics,
                    scaler=scaler,
                    device=device,
                    output_dir=output_dir,
                    debug_fast=args.debug_fast,
                )
                append_csv(run_path, result)
                log(
                    f"  [DONE] score={result['selection_score']:.4f} | "
                    f"best_ep={result['best_epoch_one_based']} | "
                    f"W={result['wind_vector_RMSE_mps']:.4f} | "
                    f"Vship={result['vessel_vector_RMSE_mps']:.4f} | "
                    f"AW={result['AW_vector_RMSE_mps']:.4f}"
                )

        del train, val, scaler, ridge, ridge_train_z, ridge_val_z
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
        log("[DONE] Debug-fast complete; no scientific candidate freeze written.")
        return 0

    runs = pd.read_csv(run_path)

    summary_rows = []
    for candidate_name, cfg in CANDIDATES.items():
        sub = runs.loc[
            (runs["method"] == candidate_name)
            & (runs["status"] == "completed_finite")
        ].copy()

        expected = len(DEVELOPMENT_MISSIONS) * len(SCREENING_SEEDS)
        if len(sub) != expected:
            raise RuntimeError(
                f"{candidate_name}: expected {expected} completed LOMO runs, "
                f"found {len(sub)}."
            )

        row = {
            "candidate": candidate_name,
            "runs": int(len(sub)),
            "selection_score_mean": float(sub["selection_score"].mean()),
            "selection_score_std": float(sub["selection_score"].std(ddof=0)),
            "parameter_count": int(round(sub["parameter_count"].mean())),
            "best_epoch_median": float(np.median(sub["best_epoch_one_based"])),
        }

        metric_cols = [
            "wind_U_RMSE_mps",
            "wind_V_RMSE_mps",
            "wind_vector_RMSE_mps",
            "wind_speed_RMSE_mps",
            "wind_direction_RMSE_deg",
            "vessel_vector_RMSE_mps",
            "AW_vector_RMSE_mps",
            "AWS_RMSE_mps",
        ]
        for col in metric_cols:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_std"] = float(sub[col].std(ddof=0))

        row.update({f"cfg_{k}": v for k, v in cfg.items()})
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["selection_score_mean", "parameter_count"]
    ).reset_index(drop=True)
    summary.to_csv(
        output_dir / "candidate_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_name = str(summary.iloc[0]["candidate"])
    selected_cfg = dict(CANDIDATES[selected_name])

    selected_runs = runs.loc[
        runs["method"] == selected_name
    ].copy()

    selected = {
        "stage": "12J",
        "script_version": SCRIPT_VERSION,
        "model_name": "Physics-Compact-NieData v4",
        "selected_candidate": selected_name,
        "selected_config": selected_cfg,
        "selection_weights": dict(SELECTION_WEIGHTS),
        "parameter_count": int(summary.iloc[0]["parameter_count"]),
        "selection_score_LOMO_mean": float(
            summary.iloc[0]["selection_score_mean"]
        ),
        "screening_seeds": SCREENING_SEEDS,
        "final_seeds": FINAL_SEEDS,
        "ridge_alpha": RIDGE_ALPHA,
        "architecture": {
            "gru_hidden": FREEZE.gru_hidden,
            "gru_layers": FREEZE.gru_layers,
            "dense_hidden": FREEZE.dense_hidden,
            "dropout": FREEZE.dropout,
            "anchor_context_dimension": 4,
            "equations": [
                "W_hat_z = W_ridge_z + sigmoid(gW) * DeltaW_z",
                "V_hat_z = V_ridge_z + sigmoid(gV) * DeltaV_z",
                "A_hat = W_hat - V_hat",
            ],
            "wind_gate_bias": FREEZE.wind_gate_bias,
            "vessel_gate_bias": FREEZE.vessel_gate_bias,
            "delta_initialization": "zero; epoch 0 equals Ridge",
        },
        "development_only_selection": True,
        "Tropical_Atlantic_read_by_12J": False,
    }

    save_json(output_dir / "selected_config.json", selected)

    selected_runs.to_csv(
        output_dir / "selected_candidate_lomo_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (output_dir / "12J_REPORT.txt").open("w", encoding="utf-8") as f:
        f.write("12J Physics-Compact-NieData v4 Development-Only LOMO\n")
        f.write("=" * 124 + "\n\n")
        f.write("CANDIDATE SUMMARY\n")
        f.write("-" * 124 + "\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nSELECTED\n")
        f.write("-" * 124 + "\n")
        f.write(json.dumps(selected, indent=2))
        f.write("\n")

    log("")
    log("=" * 124)
    log("12J DEVELOPMENT SELECTION COMPLETE")
    log("=" * 124)
    log(summary.to_string(index=False))
    log("")
    log(f"SELECTED: {selected_name}")
    log(f"parameters: {selected['parameter_count']:,}")
    log(f"score     : {selected['selection_score_LOMO_mean']:.5f}")
    log(f"[SAVED] {output_dir / 'selected_config.json'}")
    log(f"[SAVED] {output_dir / '12J_REPORT.txt'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n[FATAL ERROR]", flush=True)
        traceback.print_exc()
        sys.exit(1)
