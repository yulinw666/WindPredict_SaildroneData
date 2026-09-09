# -*- coding: utf-8 -*-
r"""
16A_run_conventional_neural_baselines_final.py

Final five-seed conventional neural baselines for the SD1090 1-min
joint forecasting task.

Models
------
1) LSTM
2) GRU
3) CNN-LSTM

The configurations are FROZEN from the previously completed tuning:
- LSTM:
    recurrent_units=64
    recurrent_layers=1
    dense_units=64
    dropout=0.2
    learning_rate=1e-3
- GRU:
    recurrent_units=64
    recurrent_layers=1
    dense_units=64
    dropout=0.1
    learning_rate=1e-3
- CNN-LSTM:
    cnn_filters=64
    cnn_layers=2
    cnn_kernel_size=3
    recurrent_units=32
    recurrent_layers=1
    dense_units=64
    dropout=0.1
    learning_rate=3e-4

Five frozen seeds
-----------------
500043, 501052, 502061, 503070, 504079

Protocol
--------
- Dataset: SD1090_TPOS2024_JointForecasting_v0_1
- Input: 60 x 11
- Target: 5 x 6
- Horizons: 1, 2, 3, 5, 10 min
- Training loss: standardized joint MSE
- Checkpoint selection: validation mean apparent-wind vector RMSE
- Batch size: 256
- Max epochs: 80
- Patience: 10
- No hyperparameter retuning is performed.

Outputs
-------
- per_seed_summary.csv
- per_horizon_metrics.csv
- aggregate_summary.csv
- frozen_hyperparameters.json
- training_history_<model>_<seed>.csv
- checkpoints/<model>_seed_<seed>.pt
- optional comparison_with_physicscompact_validation.csv
- optional conventional_neural_baselines_table.tex

Recommended PowerShell
----------------------
python "D:\project\WindPredict_SaildroneData\src\16A_run_conventional_neural_baselines_final.py" `
  --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\SD1090_TPOS2024_JointForecasting_v0_1" `
  --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\16A_ConventionalDeepBaselines_Final_v0_1"

Notes
-----
- The script fits normalization statistics from the SD1090 TRAIN split only.
- Validation and test are transformed with the frozen TRAIN statistics.
- Apparent wind is reconstructed by A_earth = W_true - V_ship.
- AWA is derived in the vessel frame using predicted HDG.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Frozen protocol
# =============================================================================

HORIZONS = np.asarray([1, 2, 3, 5, 10], dtype=int)
SEEDS = [500043, 501052, 502061, 503070, 504079]


@dataclass(frozen=True)
class ModelConfig:
    recurrent_units: int
    recurrent_layers: int
    dense_units: int
    dropout: float
    learning_rate: float
    cnn_filters: int = 32
    cnn_layers: int = 1
    cnn_kernel_size: int = 3


FROZEN_CONFIGS: Dict[str, ModelConfig] = {
    "LSTM": ModelConfig(
        recurrent_units=64,
        recurrent_layers=1,
        dense_units=64,
        dropout=0.2,
        learning_rate=1.0e-3,
        cnn_filters=32,
        cnn_layers=1,
        cnn_kernel_size=3,
    ),
    "GRU": ModelConfig(
        recurrent_units=64,
        recurrent_layers=1,
        dense_units=64,
        dropout=0.1,
        learning_rate=1.0e-3,
        cnn_filters=32,
        cnn_layers=1,
        cnn_kernel_size=3,
    ),
    "CNN-LSTM": ModelConfig(
        recurrent_units=32,
        recurrent_layers=1,
        dense_units=64,
        dropout=0.1,
        learning_rate=3.0e-4,
        cnn_filters=64,
        cnn_layers=2,
        cnn_kernel_size=3,
    ),
}


# =============================================================================
# Reproducibility and device
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproducible cuDNN behavior.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # TF32 can slightly change floating-point results. Keep it off here for
    # the final five-seed paper benchmark.
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


def select_device(force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    print("=" * 100)
    print("DEVICE")
    print("=" * 100)
    print(f"torch version      : {torch.__version__}")
    print(f"selected device    : {device}")
    print(f"CUDA available     : {torch.cuda.is_available()}")

    if device.type == "cuda":
        print(f"torch CUDA version : {torch.version.cuda}")
        print(f"GPU name           : {torch.cuda.get_device_name(0)}")
        prop = torch.cuda.get_device_properties(0)
        print(f"GPU VRAM           : {prop.total_memory / 1024**3:.2f} GB")

    print("=" * 100)
    return device


# =============================================================================
# Data
# =============================================================================

def load_npz_split(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as z:
        if "X" not in z:
            raise KeyError(f"{path} does not contain key 'X'.")
        if "y_raw" not in z:
            raise KeyError(f"{path} does not contain key 'y_raw'.")

        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y_raw"], dtype=np.float32)

    if X.ndim != 3 or X.shape[1:] != (60, 11):
        raise RuntimeError(
            f"Expected X shape (N,60,11), got {X.shape} from {path}"
        )

    if y.ndim != 3 or y.shape[1:] != (5, 6):
        raise RuntimeError(
            f"Expected y_raw shape (N,5,6), got {y.shape} from {path}"
        )

    if not np.isfinite(X).all():
        raise RuntimeError(f"Non-finite values found in X: {path}")

    if not np.isfinite(y).all():
        raise RuntimeError(f"Non-finite values found in y_raw: {path}")

    return X, y


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit_inputs(cls, X_train: np.ndarray) -> "Standardizer":
        # Per input channel, fitted using all TRAIN samples and history times.
        mean = X_train.astype(np.float64).mean(axis=(0, 1))
        std = X_train.astype(np.float64).std(axis=(0, 1), ddof=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    @classmethod
    def fit_targets(cls, y_train: np.ndarray) -> "Standardizer":
        # Per target channel, shared across forecast horizons.
        mean = y_train.astype(np.float64).mean(axis=(0, 1))
        std = y_train.astype(np.float64).std(axis=(0, 1), ddof=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform_inputs(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean[None, None, :]) /
                self.std[None, None, :]).astype(np.float32)

    def transform_targets(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.mean[None, None, :]) /
                self.std[None, None, :]).astype(np.float32)

    def inverse_targets(self, yz: np.ndarray) -> np.ndarray:
        return (yz * self.std[None, None, :] +
                self.mean[None, None, :]).astype(np.float32)


class ForecastDataset(Dataset):
    def __init__(self, Xz: np.ndarray, yz: np.ndarray):
        self.X = torch.from_numpy(np.ascontiguousarray(Xz))
        self.y = torch.from_numpy(np.ascontiguousarray(yz))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# =============================================================================
# Models
# =============================================================================

class DirectRecurrentModel(nn.Module):
    def __init__(
        self,
        cell_type: str,
        input_dim: int,
        output_dim: int,
        cfg: ModelConfig,
    ):
        super().__init__()

        if cell_type == "LSTM":
            recurrent_cls = nn.LSTM
        elif cell_type == "GRU":
            recurrent_cls = nn.GRU
        else:
            raise ValueError(cell_type)

        # PyTorch recurrent dropout is only active when num_layers > 1,
        # so the frozen single-layer models use an explicit dropout block.
        self.recurrent = recurrent_cls(
            input_size=input_dim,
            hidden_size=cfg.recurrent_units,
            num_layers=cfg.recurrent_layers,
            batch_first=True,
            dropout=(
                cfg.dropout
                if cfg.recurrent_layers > 1
                else 0.0
            ),
        )

        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.recurrent_units, cfg.dense_units),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dense_units, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, state = self.recurrent(x)

        if isinstance(state, tuple):  # LSTM -> (h_n, c_n)
            h = state[0][-1]
        else:  # GRU -> h_n
            h = state[-1]

        y = self.head(h)
        return y.view(-1, 5, 6)


class CNNLSTMModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cfg: ModelConfig,
    ):
        super().__init__()

        convs: List[nn.Module] = []
        in_channels = input_dim

        for _ in range(cfg.cnn_layers):
            convs.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=cfg.cnn_filters,
                        kernel_size=cfg.cnn_kernel_size,
                        padding=cfg.cnn_kernel_size // 2,
                    ),
                    nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                ]
            )
            in_channels = cfg.cnn_filters

        self.cnn = nn.Sequential(*convs)

        self.lstm = nn.LSTM(
            input_size=cfg.cnn_filters,
            hidden_size=cfg.recurrent_units,
            num_layers=cfg.recurrent_layers,
            batch_first=True,
            dropout=(
                cfg.dropout
                if cfg.recurrent_layers > 1
                else 0.0
            ),
        )

        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.recurrent_units, cfg.dense_units),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dense_units, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> [B, C, T]
        z = x.transpose(1, 2)
        z = self.cnn(z)

        # Back to recurrent sequence: [B, T, C_cnn]
        z = z.transpose(1, 2)

        _, (h_n, _) = self.lstm(z)
        h = h_n[-1]

        y = self.head(h)
        return y.view(-1, 5, 6)


def build_model(model_name: str, cfg: ModelConfig) -> nn.Module:
    if model_name in {"LSTM", "GRU"}:
        return DirectRecurrentModel(
            cell_type=model_name,
            input_dim=11,
            output_dim=30,
            cfg=cfg,
        )

    if model_name == "CNN-LSTM":
        return CNNLSTMModel(
            input_dim=11,
            output_dim=30,
            cfg=cfg,
        )

    raise ValueError(model_name)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Metrics
# =============================================================================

def vector_rmse(true_xy: np.ndarray, pred_xy: np.ndarray) -> np.ndarray:
    err = pred_xy.astype(np.float64) - true_xy.astype(np.float64)
    return np.sqrt(np.mean(np.sum(err * err, axis=-1), axis=0))


def scalar_rmse(true_x: np.ndarray, pred_x: np.ndarray) -> np.ndarray:
    err = pred_x.astype(np.float64) - true_x.astype(np.float64)
    return np.sqrt(np.mean(err * err, axis=0))


def wrap_deg(delta_deg: np.ndarray) -> np.ndarray:
    return (delta_deg + 180.0) % 360.0 - 180.0


def heading_deg_from_sincos(q: np.ndarray) -> np.ndarray:
    sin_h = q[..., 0]
    cos_h = q[..., 1]
    return np.degrees(np.arctan2(sin_h, cos_h))


def normalize_heading_pair(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64)
    n = np.sqrt(np.sum(q * q, axis=-1, keepdims=True))
    n = np.maximum(n, 1e-12)
    return (q / n).astype(np.float32)


def body_apparent_from_earth(
    app_earth: np.ndarray,
    heading_pair: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    q = normalize_heading_pair(heading_pair)

    sin_h = q[..., 0]
    cos_h = q[..., 1]

    ae = app_earth[..., 0]
    an = app_earth[..., 1]

    # Forward and starboard body axes.
    af = ae * sin_h + an * cos_h
    a_star = ae * cos_h - an * sin_h

    return af, a_star


def awa_deg(
    app_earth: np.ndarray,
    heading_pair: np.ndarray,
) -> np.ndarray:
    af, a_star = body_apparent_from_earth(app_earth, heading_pair)
    return np.degrees(np.arctan2(-a_star, -af))


def metric_dict_per_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, np.ndarray]:
    # Target layout:
    # 0 U
    # 1 V
    # 2 vessel east
    # 3 vessel north
    # 4 HDG sin
    # 5 HDG cos

    wind_true = y_true[..., 0:2]
    wind_pred = y_pred[..., 0:2]

    vessel_true = y_true[..., 2:4]
    vessel_pred = y_pred[..., 2:4]

    hdg_true_pair = normalize_heading_pair(y_true[..., 4:6])
    hdg_pred_pair = normalize_heading_pair(y_pred[..., 4:6])

    app_true = wind_true - vessel_true
    app_pred = wind_pred - vessel_pred

    ws_true = np.linalg.norm(wind_true, axis=-1)
    ws_pred = np.linalg.norm(wind_pred, axis=-1)

    sog_true = np.linalg.norm(vessel_true, axis=-1)
    sog_pred = np.linalg.norm(vessel_pred, axis=-1)

    aws_true = np.linalg.norm(app_true, axis=-1)
    aws_pred = np.linalg.norm(app_pred, axis=-1)

    hdg_true = heading_deg_from_sincos(hdg_true_pair)
    hdg_pred = heading_deg_from_sincos(hdg_pred_pair)
    hdg_err = np.abs(wrap_deg(hdg_pred - hdg_true))

    awa_true = awa_deg(app_true, hdg_true_pair)
    awa_pred = awa_deg(app_pred, hdg_pred_pair)
    awa_err = np.abs(wrap_deg(awa_pred - awa_true))

    return {
        "WindVector_RMSE_mps": vector_rmse(wind_true, wind_pred),
        "VesselVector_RMSE_mps": vector_rmse(vessel_true, vessel_pred),
        "AppVector_RMSE_mps": vector_rmse(app_true, app_pred),
        "WS_RMSE_mps": scalar_rmse(ws_true, ws_pred),
        "SOG_RMSE_mps": scalar_rmse(sog_true, sog_pred),
        "AWS_RMSE_mps": scalar_rmse(aws_true, aws_pred),
        "HDG_MAE_deg": np.mean(hdg_err, axis=0),
        "AWA_MAE_deg": np.mean(awa_err, axis=0),
    }


def mean_over_horizons(metrics: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {
        key: float(np.mean(value))
        for key, value in metrics.items()
    }


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def predict_standardized(
    model: nn.Module,
    Xz: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp_enabled: bool,
) -> np.ndarray:
    model.eval()

    dummy_y = np.zeros((len(Xz), 5, 6), dtype=np.float32)
    loader = DataLoader(
        ForecastDataset(Xz, dummy_y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    out = []

    for xb, _ in loader:
        xb = xb.to(
            device,
            non_blocking=(device.type == "cuda"),
        )

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        ):
            pred = model(xb)

        out.append(pred.float().cpu().numpy())

    return np.concatenate(out, axis=0)


def validation_aw_rmse(
    model: nn.Module,
    Xval_z: np.ndarray,
    yval_raw: np.ndarray,
    y_scaler: Standardizer,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp_enabled: bool,
) -> float:
    pred_z = predict_standardized(
        model=model,
        Xz=Xval_z,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        amp_enabled=amp_enabled,
    )
    pred_raw = y_scaler.inverse_targets(pred_z)

    metrics = metric_dict_per_horizon(yval_raw, pred_raw)
    return float(np.mean(metrics["AppVector_RMSE_mps"]))


# =============================================================================
# Training
# =============================================================================

def train_one_seed(
    model_name: str,
    cfg: ModelConfig,
    seed: int,
    Xtrain_z: np.ndarray,
    ytrain_z: np.ndarray,
    Xval_z: np.ndarray,
    yval_raw: np.ndarray,
    y_scaler: Standardizer,
    device: torch.device,
    output_dir: Path,
    batch_size: int,
    max_epochs: int,
    patience: int,
    num_workers: int,
    amp_enabled: bool,
) -> Tuple[nn.Module, pd.DataFrame, int, float]:
    set_seed(seed)

    model = build_model(model_name, cfg).to(device)
    n_params = count_parameters(model)

    print("")
    print("=" * 100)
    print(f"TRAINING {model_name} | seed={seed}")
    print("=" * 100)
    print(f"config     : {asdict(cfg)}")
    print(f"parameters : {n_params:,}")
    print(f"device     : {device}")

    train_loader = DataLoader(
        ForecastDataset(Xtrain_z, ytrain_z),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
    )

    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=(amp_enabled and device.type == "cuda"),
    )

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.replace("-", "_")
    ckpt_path = ckpt_dir / f"{safe_name}_seed_{seed}.pt"

    best_score = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()

        running_loss = 0.0
        n_seen = 0

        t0 = time.time()

        for xb, yb in train_loader:
            xb = xb.to(
                device,
                non_blocking=(device.type == "cuda"),
            )
            yb = yb.to(
                device,
                non_blocking=(device.type == "cuda"),
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            ):
                pred = model(xb)
                loss = criterion(pred, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = xb.shape[0]
            running_loss += float(loss.detach().cpu()) * bs
            n_seen += bs

        train_loss = running_loss / max(n_seen, 1)

        val_aw = validation_aw_rmse(
            model=model,
            Xval_z=Xval_z,
            yval_raw=yval_raw,
            y_scaler=y_scaler,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            amp_enabled=amp_enabled,
        )

        epoch_time = time.time() - t0

        improved = val_aw < best_score - 1e-8

        if improved:
            best_score = val_aw
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "model_name": model_name,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_AppVector_RMSE_mps": best_score,
                    "config": asdict(cfg),
                    "state_dict": model.state_dict(),
                    "n_parameters": n_params,
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1

        history.append(
            {
                "model": model_name,
                "seed": seed,
                "epoch": epoch,
                "train_joint_MSE_z": train_loss,
                "validation_AppVector_RMSE_mps": val_aw,
                "is_best": int(improved),
                "epoch_time_s": epoch_time,
            }
        )

        print(
            f"Epoch {epoch:03d}/{max_epochs} | "
            f"train_MSE_z={train_loss:.6f} | "
            f"val_AppVector_RMSE={val_aw:.6f} | "
            f"best={best_score:.6f} @ {best_epoch:03d} | "
            f"patience={bad_epochs}/{patience} | "
            f"{epoch_time:.1f}s"
        )

        if bad_epochs >= patience:
            print(
                f"[EARLY STOP] {model_name} seed={seed} "
                f"at epoch {epoch}; best epoch={best_epoch}."
            )
            break

    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint was not saved: {ckpt_path}")

    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])

    history_df = pd.DataFrame(history)
    history_path = (
        output_dir
        / f"training_history_{safe_name}_seed_{seed}.csv"
    )
    history_df.to_csv(history_path, index=False)

    return model, history_df, best_epoch, best_score


# =============================================================================
# Evaluation and reporting
# =============================================================================

def evaluate_split(
    model_name: str,
    seed: int,
    split_name: str,
    model: nn.Module,
    Xz: np.ndarray,
    y_raw: np.ndarray,
    y_scaler: Standardizer,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp_enabled: bool,
) -> Tuple[Dict[str, float], List[Dict[str, float]], np.ndarray]:
    pred_z = predict_standardized(
        model=model,
        Xz=Xz,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        amp_enabled=amp_enabled,
    )
    pred_raw = y_scaler.inverse_targets(pred_z)

    metrics_h = metric_dict_per_horizon(y_raw, pred_raw)
    summary = mean_over_horizons(metrics_h)

    summary_row = {
        "model": model_name,
        "seed": seed,
        "split": split_name,
        **summary,
    }

    horizon_rows = []
    for h_idx, h in enumerate(HORIZONS):
        row = {
            "model": model_name,
            "seed": seed,
            "split": split_name,
            "horizon_min": int(h),
        }
        for key, values in metrics_h.items():
            row[key] = float(values[h_idx])
        horizon_rows.append(row)

    return summary_row, horizon_rows, pred_raw


def aggregate_seed_results(per_seed: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "WindVector_RMSE_mps",
        "VesselVector_RMSE_mps",
        "AppVector_RMSE_mps",
        "WS_RMSE_mps",
        "SOG_RMSE_mps",
        "AWS_RMSE_mps",
        "HDG_MAE_deg",
        "AWA_MAE_deg",
    ]

    rows = []

    for (model, split), g in per_seed.groupby(["model", "split"]):
        row = {
            "model": model,
            "split": split,
            "n_seeds": int(len(g)),
        }

        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=1))

        if "parameters" in g:
            row["parameters"] = int(g["parameters"].iloc[0])

        rows.append(row)

    return pd.DataFrame(rows)


def load_physicscompact_validation(
    physics_dir: Path,
) -> pd.DataFrame:
    """
    Optional: calculate the FINAL Physics-Compact validation metrics from
    existing alpha_M=1500 five-seed prediction files so the conventional
    baseline table can include the proposed model without retraining it.
    """
    rows = []

    for seed in SEEDS:
        p = physics_dir / f"15E_VH_seed_{seed}_validation_predictions.npz"

        if not p.exists():
            raise FileNotFoundError(p)

        with np.load(p, allow_pickle=False) as z:
            required = ["y_raw", "frozen_truewind", "joint_vessel_hdg"]
            missing = [k for k in required if k not in z]
            if missing:
                raise KeyError(f"{p}: missing keys {missing}")

            y_true = np.asarray(z["y_raw"], dtype=np.float32)
            wind = np.asarray(z["frozen_truewind"], dtype=np.float32)
            motion = np.asarray(z["joint_vessel_hdg"], dtype=np.float32)

        y_pred = np.empty_like(y_true)
        y_pred[..., 0:2] = wind
        y_pred[..., 2:6] = motion

        metrics = mean_over_horizons(
            metric_dict_per_horizon(y_true, y_pred)
        )

        rows.append(
            {
                "model": "Physics-Compact",
                "seed": seed,
                "split": "validation",
                **metrics,
                "parameters": 49538,
            }
        )

    return pd.DataFrame(rows)


def make_latex_table(
    aggregate_validation: pd.DataFrame,
    output_path: Path,
) -> None:
    order = [
        "LSTM",
        "GRU",
        "CNN-LSTM",
        "Physics-Compact",
    ]

    g = aggregate_validation.copy()
    g["order"] = g["model"].map(
        {name: i for i, name in enumerate(order)}
    )
    g = g.sort_values("order")

    def fmt(mean, std, digits=4):
        if pd.isna(std):
            return f"{mean:.{digits}f}"
        return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Five-seed comparison with conventional neural forecasting baselines on the frozen SD1090 validation interval.}",
        r"\label{tab:conventional_neural_baselines}",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Model & Wind vector & Vessel vector & App. vector & AWS & AWA MAE & Parameters \\",
        r"\hline",
    ]

    for _, r in g.iterrows():
        model = str(r["model"])
        if model == "Physics-Compact":
            model = r"\textbf{Physics-Compact}"

        app = fmt(
            r["AppVector_RMSE_mps_mean"],
            r["AppVector_RMSE_mps_std"],
        )
        wind = fmt(
            r["WindVector_RMSE_mps_mean"],
            r["WindVector_RMSE_mps_std"],
        )
        vessel = fmt(
            r["VesselVector_RMSE_mps_mean"],
            r["VesselVector_RMSE_mps_std"],
        )
        aws = fmt(
            r["AWS_RMSE_mps_mean"],
            r["AWS_RMSE_mps_std"],
        )
        awa = fmt(
            r["AWA_MAE_deg_mean"],
            r["AWA_MAE_deg_std"],
            digits=2,
        )
        params = int(r["parameters"])

        lines.append(
            f"{model} & {wind} & {vessel} & {app} & {aws} & {awa} & {params:,} \\\\"
        )

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--physics-dir",
        type=Path,
        default=None,
        help=(
            "Optional FINAL Physics-Compact alpha_M=1500 validation "
            "prediction directory for automatic comparison-table output."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["LSTM", "GRU", "CNN-LSTM"],
        default=["LSTM", "GRU", "CNN-LSTM"],
    )

    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.physics_dir is None:
        project_root = dataset_dir
        for p in [dataset_dir, *dataset_dir.parents]:
            if p.name.lower() == "windpredict_saildronedata":
                project_root = p
                break

        default_physics = (
            project_root
            / "data"
            / "forecasting"
            / "15E_VH_AlphaSearch_v0_1"
            / "full_retrain"
            / "alpha_1500"
        )

        physics_dir = (
            default_physics
            if default_physics.exists()
            else None
        )
    else:
        physics_dir = args.physics_dir.resolve()

    device = select_device(force_cpu=args.cpu)
    amp_enabled = (
        device.type == "cuda"
        and not args.no_amp
    )

    print("")
    print("=" * 100)
    print("16A FINAL CONVENTIONAL NEURAL BASELINES")
    print("=" * 100)
    print(f"dataset_dir       : {dataset_dir}")
    print(f"output_dir        : {output_dir}")
    print(f"models            : {args.models}")
    print(f"seeds             : {SEEDS}")
    print(f"batch_size        : {args.batch_size}")
    print(f"max_epochs        : {args.epochs}")
    print(f"patience          : {args.patience}")
    print(f"AMP               : {amp_enabled}")
    print("selection metric  : validation mean apparent-wind vector RMSE")
    print("hyperparameter tuning: DISABLED / FROZEN")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Load frozen splits
    # -------------------------------------------------------------------------
    Xtr_raw, ytr_raw = load_npz_split(dataset_dir / "train.npz")
    Xva_raw, yva_raw = load_npz_split(dataset_dir / "validation.npz")
    Xte_raw, yte_raw = load_npz_split(dataset_dir / "test.npz")

    print(f"train      : X={Xtr_raw.shape}, y={ytr_raw.shape}")
    print(f"validation : X={Xva_raw.shape}, y={yva_raw.shape}")
    print(f"test       : X={Xte_raw.shape}, y={yte_raw.shape}")

    # -------------------------------------------------------------------------
    # Training-only standardization
    # -------------------------------------------------------------------------
    x_scaler = Standardizer.fit_inputs(Xtr_raw)
    y_scaler = Standardizer.fit_targets(ytr_raw)

    Xtr_z = x_scaler.transform_inputs(Xtr_raw)
    Xva_z = x_scaler.transform_inputs(Xva_raw)
    Xte_z = x_scaler.transform_inputs(Xte_raw)

    ytr_z = y_scaler.transform_targets(ytr_raw)

    np.savez_compressed(
        output_dir / "training_scalers.npz",
        x_mean=x_scaler.mean,
        x_std=x_scaler.std,
        y_mean=y_scaler.mean,
        y_std=y_scaler.std,
    )

    frozen_manifest = {
        "script": "16A_run_conventional_neural_baselines_final.py",
        "task": "SD1090 1-min joint wind-vessel multi-horizon forecasting",
        "horizons_min": HORIZONS.tolist(),
        "seeds": SEEDS,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "selection_metric": "validation mean apparent-wind vector RMSE",
        "training_loss": "standardized joint MSE",
        "hyperparameter_retuning": False,
        "models": {
            k: asdict(v)
            for k, v in FROZEN_CONFIGS.items()
        },
    }

    with open(
        output_dir / "frozen_hyperparameters.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(frozen_manifest, f, indent=2)

    # -------------------------------------------------------------------------
    # Final five-seed train / validation / test
    # -------------------------------------------------------------------------
    per_seed_rows: List[Dict[str, float]] = []
    per_horizon_rows: List[Dict[str, float]] = []

    for model_name in args.models:
        cfg = FROZEN_CONFIGS[model_name]

        for seed in SEEDS:
            model, history, best_epoch, best_val = train_one_seed(
                model_name=model_name,
                cfg=cfg,
                seed=seed,
                Xtrain_z=Xtr_z,
                ytrain_z=ytr_z,
                Xval_z=Xva_z,
                yval_raw=yva_raw,
                y_scaler=y_scaler,
                device=device,
                output_dir=output_dir,
                batch_size=args.batch_size,
                max_epochs=args.epochs,
                patience=args.patience,
                num_workers=args.num_workers,
                amp_enabled=amp_enabled,
            )

            n_params = count_parameters(model)

            # Validation
            val_row, val_h, val_pred = evaluate_split(
                model_name=model_name,
                seed=seed,
                split_name="validation",
                model=model,
                Xz=Xva_z,
                y_raw=yva_raw,
                y_scaler=y_scaler,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                amp_enabled=amp_enabled,
            )
            val_row["best_epoch"] = best_epoch
            val_row["best_validation_selection_RMSE_mps"] = best_val
            val_row["parameters"] = n_params

            # Test
            test_row, test_h, test_pred = evaluate_split(
                model_name=model_name,
                seed=seed,
                split_name="test",
                model=model,
                Xz=Xte_z,
                y_raw=yte_raw,
                y_scaler=y_scaler,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                amp_enabled=amp_enabled,
            )
            test_row["best_epoch"] = best_epoch
            test_row["best_validation_selection_RMSE_mps"] = best_val
            test_row["parameters"] = n_params

            per_seed_rows.extend([val_row, test_row])
            per_horizon_rows.extend(val_h + test_h)

            # Save final validation/test predictions for audit/replotting.
            pred_dir = output_dir / "predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)

            safe_name = model_name.replace("-", "_")

            np.savez_compressed(
                pred_dir
                / f"{safe_name}_seed_{seed}_validation_predictions.npz",
                y_raw=yva_raw,
                y_pred=val_pred,
                horizons_min=HORIZONS,
            )

            np.savez_compressed(
                pred_dir
                / f"{safe_name}_seed_{seed}_test_predictions.npz",
                y_raw=yte_raw,
                y_pred=test_pred,
                horizons_min=HORIZONS,
            )

            print("")
            print(
                f"[RESULT] {model_name} seed={seed} | "
                f"VAL AppVector={val_row['AppVector_RMSE_mps']:.6f} | "
                f"TEST AppVector={test_row['AppVector_RMSE_mps']:.6f} | "
                f"params={n_params:,}"
            )

    per_seed = pd.DataFrame(per_seed_rows)
    per_horizon = pd.DataFrame(per_horizon_rows)

    per_seed.to_csv(
        output_dir / "per_seed_summary.csv",
        index=False,
    )
    per_horizon.to_csv(
        output_dir / "per_horizon_metrics.csv",
        index=False,
    )

    aggregate = aggregate_seed_results(per_seed)
    aggregate.to_csv(
        output_dir / "aggregate_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Optional final Physics-Compact comparison
    # -------------------------------------------------------------------------
    comparison = aggregate[
        aggregate["split"] == "validation"
    ].copy()

    if physics_dir is not None and physics_dir.exists():
        print("")
        print(f"[LOAD] FINAL Physics-Compact validation: {physics_dir}")

        physics_seed = load_physicscompact_validation(physics_dir)
        physics_agg = aggregate_seed_results(physics_seed)

        comparison = pd.concat(
            [
                comparison,
                physics_agg[
                    physics_agg["split"] == "validation"
                ],
            ],
            ignore_index=True,
        )

        physics_seed.to_csv(
            output_dir
            / "physicscompact_validation_per_seed.csv",
            index=False,
        )

    comparison.to_csv(
        output_dir
        / "comparison_with_physicscompact_validation.csv",
        index=False,
    )

    if "Physics-Compact" in set(comparison["model"]):
        make_latex_table(
            comparison,
            output_dir
            / "conventional_neural_baselines_table.tex",
        )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    cols = [
        "model",
        "split",
        "n_seeds",
        "WindVector_RMSE_mps_mean",
        "VesselVector_RMSE_mps_mean",
        "AppVector_RMSE_mps_mean",
        "AWS_RMSE_mps_mean",
        "AWA_MAE_deg_mean",
        "parameters",
    ]

    print("")
    print("=" * 100)
    print("FINAL FIVE-SEED AGGREGATE")
    print("=" * 100)
    print(aggregate[cols].to_string(index=False))

    if "Physics-Compact" in set(comparison["model"]):
        print("")
        print("=" * 100)
        print("VALIDATION COMPARISON INCLUDING FINAL PHYSICS-COMPACT")
        print("=" * 100)
        print(comparison[cols].to_string(index=False))

    print("")
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
