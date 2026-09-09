# -*- coding: utf-8 -*-
"""
07A_physics_guided_residual_forecasting.py

Physics-guided gated residual forecasting for the frozen Saildrone
joint wind-vessel forecasting dataset.

Model chain
-----------
Frozen Ridge
    -> Residual-GRU
    -> Physics-Residual-GRU

The residual models do NOT replace Ridge. They learn only a gated nonlinear
correction:

    y_hat_z = y_ridge_z + gate * delta_z

where gate is sample-, horizon-, and target-specific.

The Physics-Residual-GRU uses the physics-loss weights selected by 06B.
By default the already-inspected SD1090 test split is NOT evaluated. Use
--evaluate-development-test only for an explicit development diagnostic.

Important dependency
--------------------
Place this file in the same src directory as:
    06_run_physics_constrained_joint_gru.py

07A dynamically imports shared validated I/O, metrics, physics, plotting and
PyTorch helpers from that file to keep all definitions consistent with 06.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import platform
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0"
CORE_SCRIPT = "06_run_physics_constrained_joint_gru.py"

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


def load_core():
    path = Path(__file__).resolve().parent / CORE_SCRIPT
    if not path.exists():
        raise FileNotFoundError(
            f"Required shared core script not found: {path}\n"
            f"Place {CORE_SCRIPT} in the same src directory as 07A."
        )

    spec = importlib.util.spec_from_file_location(
        "physics_gru_core_07a",
        str(path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import shared core: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def save_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_tuned_gru_config(tuning_dir: Path) -> tuple[dict, dict]:
    best_path = tuning_dir / "best_hyperparameters.json"
    if not best_path.exists():
        raise FileNotFoundError(best_path)

    obj = json.loads(best_path.read_text(encoding="utf-8"))
    if "GRU" not in obj:
        raise RuntimeError("GRU not found in 05C-v2 best_hyperparameters.json.")

    g = obj["GRU"]
    cfg = {
        "recurrent_units": int(g["recurrent_units"]),
        "recurrent_layers": int(g["recurrent_layers"]),
        "dropout": float(g["dropout"]),
        "learning_rate": float(g["learning_rate"]),
        "dense_units": int(g.get("dense_units", 64)),
    }

    meta = {
        "best_hyperparameters_path": str(best_path.resolve()),
        "matched_seed": None,
    }

    final_path = tuning_dir / "final_retrain_summary.csv"
    if final_path.exists():
        df = pd.read_csv(final_path)
        row = df.loc[df["model"] == "GRU"]
        if not row.empty and "seed" in row.columns:
            meta["matched_seed"] = int(row.iloc[0]["seed"])
        meta["final_retrain_summary_path"] = str(final_path.resolve())

    return cfg, meta


def load_physics_weights(physics_weight_dir: Path) -> tuple[dict, dict]:
    path = physics_weight_dir / "best_physics_weights.json"
    if not path.exists():
        raise FileNotFoundError(path)

    obj = json.loads(path.read_text(encoding="utf-8"))
    weights = {
        "earth": float(obj["earth"]),
        "body": float(obj["body"]),
        "heading_geo": float(obj["heading_geo"]),
        "heading_norm": float(obj["heading_norm"]),
    }
    return weights, {"path": str(path.resolve()), "raw": obj}


def load_frozen_ridge(ridge_dir: Path):
    try:
        import joblib
    except Exception as exc:
        raise RuntimeError("joblib is required to load Ridge.") from exc

    path = ridge_dir / "ridge_model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen Ridge model not found: {path}\n"
            "Run 04B first or point --ridge-dir to its output directory."
        )

    model = joblib.load(path)

    manifest_path = ridge_dir / "joint_baseline_manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return model, path, manifest


def ridge_predict_z(
    model,
    X: np.ndarray,
    n_horizons: int,
    n_targets: int,
    chunk_size: int,
) -> np.ndarray:
    n = X.shape[0]
    out = np.empty((n, n_horizons, n_targets), dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        x2 = X[start:end].reshape(end - start, -1)
        pred = np.asarray(model.predict(x2), dtype=np.float32)
        out[start:end] = pred.reshape(end - start, n_horizons, n_targets)

    return out


def inverse_target(
    y_z: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    return (
        y_z.astype(np.float64)
        * target_std[None, None, :]
        + target_mean[None, None, :]
    ).astype(np.float32)


def build_residual_model_class(torch, nn):
    class GatedResidualGRU(nn.Module):
        """
        Ridge-conditioned gated residual network.

        X -> GRU -> concat(flat Ridge forecast)
          -> Dense -> ReLU -> Dropout
          -> delta head
          -> gate head (sigmoid)

        combined_z = ridge_z + gate * delta_z
        """

        def __init__(
            self,
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
            self.n_horizons = n_horizons
            self.n_targets = n_targets

            rnn_dropout = dropout if recurrent_layers > 1 else 0.0
            self.encoder = nn.GRU(
                input_size=n_features,
                hidden_size=recurrent_units,
                num_layers=recurrent_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )

            ridge_dim = n_horizons * n_targets
            self.fusion = nn.Linear(
                recurrent_units + ridge_dim,
                dense_units,
            )
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            self.delta_head = nn.Linear(dense_units, ridge_dim)
            self.gate_head = nn.Linear(dense_units, ridge_dim)

            # Begin very close to the frozen Ridge solution.
            nn.init.normal_(
                self.delta_head.weight,
                mean=0.0,
                std=residual_init_std,
            )
            nn.init.zeros_(self.delta_head.bias)
            nn.init.xavier_uniform_(self.gate_head.weight)
            nn.init.constant_(self.gate_head.bias, gate_bias)

        def forward(self, X, ridge_z):
            seq, _ = self.encoder(X)
            h = seq[:, -1, :]

            ridge_flat = ridge_z.reshape(ridge_z.shape[0], -1)
            z = torch.cat([h, ridge_flat], dim=1)
            z = self.dropout(self.relu(self.fusion(z)))

            delta_z = self.delta_head(z).reshape(
                -1, self.n_horizons, self.n_targets
            )
            gate = torch.sigmoid(self.gate_head(z)).reshape(
                -1, self.n_horizons, self.n_targets
            )

            correction_z = gate * delta_z
            combined_z = ridge_z + correction_z
            return combined_z, delta_z, gate, correction_z

    return GatedResidualGRU


def make_dataset(torch, TensorDataset, X, y, ridge_z):
    return TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(ridge_z, dtype=np.float32)),
    )


def make_loader(
    DataLoader,
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def count_parameters(model) -> int:
    return int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )


def compute_losses(
    core,
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
    components = core.compute_loss_components(
        torch=torch,
        pred_z=combined_z,
        true_z=true_z,
        target_mean_t=target_mean_t,
        target_std_t=target_std_t,
        physics_scale_t=physics_scale_t,
    )

    residual_penalty = torch.mean(correction_z ** 2)

    total = (
        components["direct"]
        + lambda_residual * residual_penalty
    )

    if use_physics:
        total = (
            total
            + physics_weights["earth"] * components["apparent_earth"]
            + physics_weights["body"] * components["apparent_body"]
            + physics_weights["heading_geo"] * components["heading_geo"]
            + physics_weights["heading_norm"] * components["heading_norm"]
        )

    return total, components, residual_penalty


def evaluate_loader(
    core,
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

    with torch.no_grad():
        for Xb, yb, rb in loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            rb = rb.to(device, non_blocking=True)

            with core.AutocastContext(torch, amp_enabled, device.type):
                combined_z, _, gate, correction_z = model(Xb, rb)

                total_loss, components, residual_penalty = compute_losses(
                    core,
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

            n = int(Xb.shape[0])
            total_n += n

            sums["total"] += float(total_loss.detach().item()) * n
            sums["direct"] += float(components["direct"].detach().item()) * n
            sums["earth"] += float(
                components["apparent_earth"].detach().item()
            ) * n
            sums["body"] += float(
                components["apparent_body"].detach().item()
            ) * n
            sums["heading_geo"] += float(
                components["heading_geo"].detach().item()
            ) * n
            sums["heading_norm"] += float(
                components["heading_norm"].detach().item()
            ) * n
            sums["residual_penalty"] += float(
                residual_penalty.detach().item()
            ) * n

            pred_chunks.append(
                combined_z.detach().float().cpu().numpy()
            )
            truth_chunks.append(
                yb.detach().float().cpu().numpy()
            )
            gate_chunks.append(
                gate.detach().float().cpu().numpy()
            )
            correction_chunks.append(
                correction_z.detach().float().cpu().numpy()
            )

    pred_z = np.concatenate(pred_chunks, axis=0)
    truth_z = np.concatenate(truth_chunks, axis=0)
    gate = np.concatenate(gate_chunks, axis=0)
    correction_z = np.concatenate(correction_chunks, axis=0)

    pred_raw = inverse_target(
        pred_z,
        target_mean_np,
        target_std_np,
    )
    truth_raw = inverse_target(
        truth_z,
        target_mean_np,
        target_std_np,
    )

    mean_aw, per_horizon_aw = core.mean_apparent_vector_rmse(
        truth_raw,
        pred_raw,
    )

    return {
        "losses": {
            k: float(v / max(total_n, 1))
            for k, v in sums.items()
        },
        "mean_aw_rmse": float(mean_aw),
        "per_horizon_aw_rmse": np.asarray(
            per_horizon_aw,
            dtype=np.float64,
        ),
        "pred_z": pred_z,
        "pred_raw": pred_raw,
        "gate": gate,
        "correction_z": correction_z,
    }


def train_variant(
    core,
    torch,
    *,
    model,
    variant_name,
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
    use_physics = variant_name == "Physics-Residual-GRU"

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, patience // 3),
        min_lr=1e-5,
    )
    scaler = core.make_grad_scaler(torch, amp_enabled)

    best_aw = float("inf")
    best_epoch = 0
    wait = 0
    history = []
    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
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

        for Xb, yb, rb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            rb = rb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with core.AutocastContext(torch, amp_enabled, device.type):
                combined_z, _, gate, correction_z = model(Xb, rb)

                total_loss, components, residual_penalty = compute_losses(
                    core,
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
                scaler.scale(total_loss).backward()
                if gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip_norm,
                    )
                optimizer.step()

            n = int(Xb.shape[0])
            total_n += n
            sums["total"] += float(total_loss.detach().item()) * n
            sums["direct"] += float(components["direct"].detach().item()) * n
            sums["earth"] += float(
                components["apparent_earth"].detach().item()
            ) * n
            sums["body"] += float(
                components["apparent_body"].detach().item()
            ) * n
            sums["heading_geo"] += float(
                components["heading_geo"].detach().item()
            ) * n
            sums["heading_norm"] += float(
                components["heading_norm"].detach().item()
            ) * n
            sums["residual_penalty"] += float(
                residual_penalty.detach().item()
            ) * n

        train_avg = {
            k: float(v / max(total_n, 1))
            for k, v in sums.items()
        }

        val_result = evaluate_loader(
            core,
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

        val_aw = val_result["mean_aw_rmse"]
        val_losses = val_result["losses"]
        lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_losses["total"])

        improved = val_aw < best_aw - min_delta

        if improved:
            best_aw = float(val_aw)
            best_epoch = int(epoch)
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
            status = f"wait={wait}/{patience}"

        row = {
            "epoch": epoch,
            "learning_rate": lr,
            "epoch_seconds": float(time.perf_counter() - epoch_t0),
            "train_total_loss": train_avg["total"],
            "train_direct_loss": train_avg["direct"],
            "train_apparent_earth_loss": train_avg["earth"],
            "train_apparent_body_loss": train_avg["body"],
            "train_heading_geo_loss": train_avg["heading_geo"],
            "train_heading_norm_loss": train_avg["heading_norm"],
            "train_residual_penalty": train_avg["residual_penalty"],
            "val_total_loss": val_losses["total"],
            "val_direct_loss": val_losses["direct"],
            "val_apparent_earth_loss": val_losses["earth"],
            "val_apparent_body_loss": val_losses["body"],
            "val_heading_geo_loss": val_losses["heading_geo"],
            "val_heading_norm_loss": val_losses["heading_norm"],
            "val_residual_penalty": val_losses["residual_penalty"],
            "val_apparent_vector_RMSE_mps": val_aw,
            "val_mean_gate": float(np.mean(val_result["gate"])),
            "val_correction_z_RMS": float(
                np.sqrt(np.mean(val_result["correction_z"] ** 2))
            ),
        }

        for j, horizon in enumerate(horizons):
            row[f"val_AW_RMSE_{horizon}min_mps"] = float(
                val_result["per_horizon_aw_rmse"][j]
            )

        history.append(row)

        core.log(
            f"Epoch {epoch:03d}/{max_epochs} | "
            f"train_total={train_avg['total']:.6f} | "
            f"val_total={val_losses['total']:.6f} | "
            f"val_AW={val_aw:.6f} m/s | "
            f"gate={row['val_mean_gate']:.3f} | "
            f"corr_z={row['val_correction_z_RMS']:.3f} | "
            f"lr={lr:.3e} | "
            f"{row['epoch_seconds']:.2f}s | {status}"
        )

        if wait >= patience:
            core.log(
                f"[EARLY STOP] {variant_name}: "
                f"best_epoch={best_epoch}, "
                f"best val AW={best_aw:.6f} m/s"
            )
            break

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["state_dict"])

    final_val = evaluate_loader(
        core,
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

    return {
        "model": model,
        "history": pd.DataFrame(history),
        "best_epoch": best_epoch,
        "best_validation_AW_RMSE_mps": best_aw,
        "epochs_ran": len(history),
        "training_seconds": float(time.perf_counter() - t0),
        "validation": final_val,
    }


def gate_statistics(
    model_name,
    split_name,
    gate,
    horizons,
    target_names,
):
    groups = {
        "wind": [0, 1],
        "vessel": [2, 3],
        "heading": [4, 5],
        "all": list(range(len(target_names))),
    }

    rows = []

    for j, h in enumerate(horizons):
        for group, idx in groups.items():
            v = gate[:, j, idx].astype(np.float64)
            rows.append({
                "split": split_name,
                "model": model_name,
                "horizon_min": int(h),
                "group": group,
                "mean_gate": float(np.mean(v)),
                "std_gate": float(np.std(v)),
                "p10_gate": float(np.quantile(v, 0.10)),
                "median_gate": float(np.median(v)),
                "p90_gate": float(np.quantile(v, 0.90)),
            })

        for t, name in enumerate(target_names):
            v = gate[:, j, t].astype(np.float64)
            rows.append({
                "split": split_name,
                "model": model_name,
                "horizon_min": int(h),
                "group": name,
                "mean_gate": float(np.mean(v)),
                "std_gate": float(np.std(v)),
                "p10_gate": float(np.quantile(v, 0.10)),
                "median_gate": float(np.median(v)),
                "p90_gate": float(np.quantile(v, 0.90)),
            })

    return pd.DataFrame(rows)


def correction_statistics(
    model_name,
    split_name,
    correction_z,
    horizons,
    target_names,
    target_std,
):
    correction_raw = (
        correction_z.astype(np.float64)
        * target_std[None, None, :]
    )

    rows = []

    for j, h in enumerate(horizons):
        for t, name in enumerate(target_names):
            z = correction_z[:, j, t].astype(np.float64)
            raw = correction_raw[:, j, t]
            rows.append({
                "split": split_name,
                "model": model_name,
                "horizon_min": int(h),
                "target": name,
                "correction_z_RMS": float(np.sqrt(np.mean(z ** 2))),
                "correction_z_MAE": float(np.mean(np.abs(z))),
                "correction_raw_RMS": float(np.sqrt(np.mean(raw ** 2))),
                "correction_raw_MAE": float(np.mean(np.abs(raw))),
            })

        for label, sl in [
            ("wind_vector", slice(0, 2)),
            ("vessel_vector", slice(2, 4)),
        ]:
            raw = correction_raw[:, j, sl]
            z = correction_z[:, j, sl].astype(np.float64)

            rows.append({
                "split": split_name,
                "model": model_name,
                "horizon_min": int(h),
                "target": label,
                "correction_z_RMS": float(np.sqrt(np.mean(z ** 2))),
                "correction_z_MAE": float(np.mean(np.abs(z))),
                "correction_raw_RMS": float(
                    np.sqrt(np.mean(raw[:, 0] ** 2 + raw[:, 1] ** 2))
                ),
                "correction_raw_MAE": float(
                    np.mean(np.sqrt(raw[:, 0] ** 2 + raw[:, 1] ** 2))
                ),
            })

    return pd.DataFrame(rows)


def save_prediction_bundle(
    path,
    model_name,
    truth_raw,
    pred_raw,
    ridge_raw,
    context_end_time_ns,
    target_time_ns,
    horizons,
    gate=None,
    correction_z=None,
):
    payload = {
        "model_name": np.asarray(model_name),
        "truth_joint": np.asarray(truth_raw, dtype=np.float32),
        "pred_joint": np.asarray(pred_raw, dtype=np.float32),
        "ridge_joint": np.asarray(ridge_raw, dtype=np.float32),
        "context_end_time_ns": np.asarray(context_end_time_ns, dtype=np.int64),
        "target_time_ns": np.asarray(target_time_ns, dtype=np.int64),
        "horizons_min": np.asarray(horizons, dtype=np.int64),
    }
    if gate is not None:
        payload["gate"] = np.asarray(gate, dtype=np.float32)
    if correction_z is not None:
        payload["correction_z"] = np.asarray(correction_z, dtype=np.float32)

    np.savez_compressed(path, **payload)


def ridge_reproduction_audit(
    ridge_dir,
    generated_metrics,
    tolerance,
    skip_check,
):
    path = ridge_dir / "metrics_per_horizon.csv"

    audit = {
        "checked": False,
        "passed": None,
        "tolerance": float(tolerance),
        "frozen_metrics_path": str(path.resolve()) if path.exists() else None,
    }

    if not path.exists():
        audit["reason"] = "Frozen 04B metrics_per_horizon.csv not found."
        return audit

    frozen = pd.read_csv(path)
    frozen = frozen.loc[
        (frozen["model"] == "Ridge")
        & (frozen["split"] == "validation")
    ].sort_values("horizon_min")

    generated = generated_metrics.sort_values("horizon_min")

    if len(frozen) != len(generated):
        audit["reason"] = "Horizon count mismatch."
        return audit

    diff = np.abs(
        frozen["apparent_vector_RMSE_mps"].to_numpy(dtype=float)
        - generated["apparent_vector_RMSE_mps"].to_numpy(dtype=float)
    )

    max_diff = float(np.max(diff))
    passed = max_diff <= tolerance

    audit.update({
        "checked": True,
        "passed": bool(passed),
        "max_abs_AW_RMSE_difference_mps": max_diff,
    })

    if not passed and not skip_check:
        raise RuntimeError(
            "Frozen Ridge reproduction check failed: "
            f"max AW-RMSE difference={max_diff:.8g} m/s "
            f"> tolerance={tolerance:.8g}."
        )

    return audit


def make_plots(
    core,
    metrics,
    gate_stats,
    correction_stats,
    histories,
    output_dir,
    has_development_test,
):
    import matplotlib.pyplot as plt

    style = core.load_sci_style(
        Path(__file__).resolve().parent
    )

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def finish(ax):
        if style is not None and hasattr(style, "clean_axis"):
            style.clean_axis(ax, grid=True)
        else:
            ax.tick_params(
                which="both",
                direction="in",
                top=True,
                right=True,
            )
            ax.grid(True, alpha=0.25)

    def save(fig, name):
        base = fig_dir / name
        if style is not None and hasattr(style, "save_figure"):
            style.save_figure(fig, base)
        else:
            fig.savefig(
                base.with_suffix(".png"),
                dpi=600,
                bbox_inches="tight",
            )
            fig.savefig(
                base.with_suffix(".pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

    order = [
        "Persistence",
        "Ridge",
        "GRU",
        "Tuned-Physics-GRU",
        "Residual-GRU",
        "Physics-Residual-GRU",
    ]

    def curve(split, metric, ylabel, name):
        subset = metrics.loc[metrics["split"] == split]
        if subset.empty:
            return

        fig, ax = plt.subplots(figsize=(5.0, 3.2))

        for model_name in order:
            grp = subset.loc[
                subset["model"] == model_name
            ].sort_values("horizon_min")

            if grp.empty:
                continue

            ax.plot(
                grp["horizon_min"],
                grp[metric],
                marker="o",
                label=model_name,
            )

        ax.set_xlabel("Forecast horizon (min)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 2, 3, 5, 10])
        ax.legend(loc="best", ncol=2)
        finish(ax)
        save(fig, name)

    curve(
        "validation",
        "apparent_vector_RMSE_mps",
        r"Apparent-wind vector RMSE (m s$^{-1}$)",
        "01_validation_apparent_rmse",
    )
    curve(
        "validation",
        "AWS_RMSE_mps",
        r"AWS RMSE (m s$^{-1}$)",
        "02_validation_aws_rmse",
    )
    curve(
        "validation",
        "AWA_MAE_deg",
        r"AWA MAE ($^\circ$)",
        "03_validation_awa_mae",
    )
    curve(
        "validation",
        "wind_vector_RMSE_mps",
        r"Wind vector RMSE (m s$^{-1}$)",
        "04_validation_wind_rmse",
    )
    curve(
        "validation",
        "vessel_vector_RMSE_mps",
        r"Vessel vector RMSE (m s$^{-1}$)",
        "05_validation_vessel_rmse",
    )

    # Gate by horizon for the physics residual model.
    fig, ax = plt.subplots(figsize=(4.8, 3.2))

    subset = gate_stats.loc[
        (gate_stats["split"] == "validation")
        & (gate_stats["model"] == "Physics-Residual-GRU")
        & (gate_stats["group"].isin(["wind", "vessel", "heading"]))
    ]

    for group in ["wind", "vessel", "heading"]:
        grp = subset.loc[
            subset["group"] == group
        ].sort_values("horizon_min")

        if not grp.empty:
            ax.plot(
                grp["horizon_min"],
                grp["mean_gate"],
                marker="o",
                label=group.capitalize(),
            )

    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel("Mean residual gate")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([1, 2, 3, 5, 10])
    ax.legend(loc="best")
    finish(ax)
    save(fig, "06_gate_by_horizon")

    # Effective correction magnitude.
    fig, ax = plt.subplots(figsize=(4.8, 3.2))

    subset = correction_stats.loc[
        (correction_stats["split"] == "validation")
        & (correction_stats["model"] == "Physics-Residual-GRU")
        & (
            correction_stats["target"].isin(
                ["wind_vector", "vessel_vector"]
            )
        )
    ]

    for target, label in [
        ("wind_vector", "Wind correction"),
        ("vessel_vector", "Vessel correction"),
    ]:
        grp = subset.loc[
            subset["target"] == target
        ].sort_values("horizon_min")

        if not grp.empty:
            ax.plot(
                grp["horizon_min"],
                grp["correction_raw_RMS"],
                marker="o",
                label=label,
            )

    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel(r"RMS effective correction (m s$^{-1}$)")
    ax.set_xticks([1, 2, 3, 5, 10])
    ax.legend(loc="best")
    finish(ax)
    save(fig, "07_correction_by_horizon")

    # Training curves.
    fig, ax = plt.subplots(figsize=(4.8, 3.2))

    for model_name, hist in histories.items():
        ax.plot(
            hist["epoch"],
            hist["val_apparent_vector_RMSE_mps"],
            label=model_name,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Validation AW RMSE (m s$^{-1}$)")
    ax.legend(loc="best")
    finish(ax)
    save(fig, "08_training_validation_aw")

    if has_development_test:
        curve(
            "development_test",
            "apparent_vector_RMSE_mps",
            r"Development-test AW RMSE (m s$^{-1}$)",
            "09_development_test_apparent_rmse",
        )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--ridge-dir", default=None)
    parser.add_argument("--tuning-dir", default=None)
    parser.add_argument("--physics-weight-dir", default=None)
    parser.add_argument("--output", default=None)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)

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

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ridge-chunk-size", type=int, default=16384)
    parser.add_argument(
        "--ridge-reproduction-tolerance",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--skip-ridge-reproduction-check",
        action="store_true",
    )

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--direction-min-speed", type=float, default=0.5)
    parser.add_argument("--cog-min-speed", type=float, default=0.2)

    parser.add_argument(
        "--evaluate-development-test",
        action="store_true",
    )
    parser.add_argument(
        "--skip-validation-flag",
        action="store_true",
    )
    parser.add_argument(
        "--skip-upstream-metric-merge",
        action="store_true",
    )
    parser.add_argument("--plots", action="store_true")

    args = parser.parse_args()

    if args.lambda_residual < 0:
        raise ValueError("--lambda-residual must be >= 0.")
    if args.residual_init_std < 0:
        raise ValueError("--residual-init-std must be >= 0.")

    core = load_core()

    dataset_dir = Path(args.dataset_dir)
    ridge_dir = (
        Path(args.ridge_dir)
        if args.ridge_dir
        else dataset_dir / "joint_baseline_v0_1"
    )
    tuning_dir = (
        Path(args.tuning_dir)
        if args.tuning_dir
        else dataset_dir / "deep_baseline_tuning_pytorch_v0_2"
    )
    physics_weight_dir = (
        Path(args.physics_weight_dir)
        if args.physics_weight_dir
        else dataset_dir / "physics_loss_weight_tuning_v0_1"
    )
    output_dir = (
        Path(args.output)
        if args.output
        else dataset_dir / "physics_guided_residual_forecasting_v0_1"
    )

    model_dir = output_dir / "models"
    history_dir = output_dir / "histories"
    prediction_dir = output_dir / "predictions"

    for d in [output_dir, model_dir, history_dir, prediction_dir]:
        d.mkdir(parents=True, exist_ok=True)

    core.log("=" * 100)
    core.log("07A - PHYSICS-GUIDED GATED RESIDUAL FORECASTING")
    core.log("=" * 100)
    core.log(f"script_version       : {SCRIPT_VERSION}")
    core.log(f"dataset_dir          : {dataset_dir}")
    core.log(f"ridge_dir            : {ridge_dir}")
    core.log(f"tuning_dir           : {tuning_dir}")
    core.log(f"physics_weight_dir   : {physics_weight_dir}")
    core.log(f"output_dir           : {output_dir}")
    core.log(f"lambda_residual      : {args.lambda_residual}")
    core.log(f"gate_bias            : {args.gate_bias}")
    core.log(f"development test     : {args.evaluate_development_test}")
    core.log("")

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------
    core.log("[STAGE 1/8] Frozen dataset, Ridge, GRU and physics configuration")

    flags = core.find_validation_flags(dataset_dir)
    if not flags and not args.skip_validation_flag:
        raise RuntimeError(
            "No VALIDATION_PASSED.flag found. Run 03B validation first."
        )

    manifest_path = dataset_dir / "dataset_manifest.json"
    scalers_path = dataset_dir / "scalers.json"
    core.require_file(manifest_path)
    core.require_file(scalers_path)

    manifest = core.load_json(manifest_path)
    scalers = core.load_json(scalers_path)

    feature_names = list(manifest["feature_names"])
    target_names = list(manifest["target_names"])
    horizons = tuple(
        int(v)
        for v in manifest["forecast_horizons_minutes"]
    )

    if feature_names != EXPECTED_FEATURE_NAMES:
        raise RuntimeError("Feature schema mismatch.")
    if target_names != EXPECTED_TARGET_NAMES:
        raise RuntimeError("Target schema mismatch.")

    f_names, feature_mean, feature_std = core.parse_scaler(
        scalers,
        "feature_scaler",
    )
    t_names, target_mean, target_std = core.parse_scaler(
        scalers,
        "target_scaler",
    )

    if f_names != feature_names:
        raise RuntimeError("Feature scaler schema mismatch.")
    if t_names != target_names:
        raise RuntimeError("Target scaler schema mismatch.")

    gru_config, tuning_meta = load_tuned_gru_config(tuning_dir)
    physics_weights, physics_meta = load_physics_weights(
        physics_weight_dir
    )
    ridge_model, ridge_model_path, ridge_manifest = load_frozen_ridge(
        ridge_dir
    )

    matched_seed = (
        int(args.seed)
        if args.seed is not None
        else (
            int(tuning_meta["matched_seed"])
            if tuning_meta["matched_seed"] is not None
            else 500043
        )
    )

    core.log(
        "[FIXED GRU] "
        f"units={gru_config['recurrent_units']}, "
        f"layers={gru_config['recurrent_layers']}, "
        f"dropout={gru_config['dropout']}, "
        f"lr={gru_config['learning_rate']}, "
        f"dense={gru_config['dense_units']}"
    )
    core.log(
        "[FIXED PHYSICS] "
        f"E={physics_weights['earth']}, "
        f"B={physics_weights['body']}, "
        f"H={physics_weights['heading_geo']}, "
        f"N={physics_weights['heading_norm']}"
    )
    core.log(f"[FROZEN RIDGE] {ridge_model_path}")
    core.log(f"[MATCHED SEED] {matched_seed}")

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------
    core.log("[STAGE 2/8] Loading data, CUDA and frozen Ridge predictions")

    train = core.load_npz_split(dataset_dir, "train")
    validation = core.load_npz_split(dataset_dir, "validation")
    test = core.load_npz_split(dataset_dir, "test")

    torch, nn, DataLoader, TensorDataset = core.import_torch()

    device = (
        torch.device("cpu")
        if args.force_cpu or not torch.cuda.is_available()
        else torch.device("cuda:0")
    )

    amp_enabled = bool(device.type == "cuda" and not args.no_amp)
    tf32_enabled = bool(device.type == "cuda" and not args.no_tf32)

    core.configure_tf32(torch, tf32_enabled)
    core.set_global_seed(torch, matched_seed, args.deterministic)

    device_info = core.log_device_info(
        torch,
        device,
        amp_enabled,
        tf32_enabled,
    )

    ridge_train_z = ridge_predict_z(
        ridge_model,
        train["X"],
        len(horizons),
        len(target_names),
        args.ridge_chunk_size,
    )
    ridge_val_z = ridge_predict_z(
        ridge_model,
        validation["X"],
        len(horizons),
        len(target_names),
        args.ridge_chunk_size,
    )

    ridge_train_raw = inverse_target(
        ridge_train_z,
        target_mean,
        target_std,
    )
    ridge_val_raw = inverse_target(
        ridge_val_z,
        target_mean,
        target_std,
    )

    persistence_val = core.persistence_joint_prediction(
        validation["X"],
        feature_names,
        feature_mean,
        feature_std,
        len(horizons),
    )

    ridge_val_metrics = core.evaluate_model(
        truth_joint=validation["y_raw"],
        pred_joint=ridge_val_raw,
        truth_apparent_ref=validation["apparent_earth_raw"],
        horizons=horizons,
        split_name="validation",
        model_name="Ridge",
        direction_min_speed=args.direction_min_speed,
        cog_min_speed=args.cog_min_speed,
        persistence_joint=persistence_val,
    )

    ridge_audit = ridge_reproduction_audit(
        ridge_dir,
        ridge_val_metrics,
        args.ridge_reproduction_tolerance,
        args.skip_ridge_reproduction_check,
    )
    save_json(
        output_dir / "ridge_reproduction_audit.json",
        ridge_audit,
    )

    if ridge_audit.get("checked"):
        core.log(
            "[RIDGE AUDIT] "
            f"passed={ridge_audit['passed']}, "
            f"max AW difference="
            f"{ridge_audit['max_abs_AW_RMSE_difference_mps']:.3e} m/s"
        )

    physics_scales = core.compute_train_physics_scales(
        train["y_raw"]
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
            physics_scales["apparent_earth_east_std_mps"],
            dtype=torch.float32,
            device=device,
        ),
        "an": torch.tensor(
            physics_scales["apparent_earth_north_std_mps"],
            dtype=torch.float32,
            device=device,
        ),
        "af": torch.tensor(
            physics_scales["apparent_body_forward_std_mps"],
            dtype=torch.float32,
            device=device,
        ),
        "as": torch.tensor(
            physics_scales["apparent_body_starboard_std_mps"],
            dtype=torch.float32,
            device=device,
        ),
    }

    pin_memory = device.type == "cuda"
    persistent_workers = (
        args.num_workers > 0
        and not args.no_persistent_workers
    )

    train_ds = make_dataset(
        torch,
        TensorDataset,
        train["X"],
        train["y"],
        ridge_train_z,
    )
    val_ds = make_dataset(
        torch,
        TensorDataset,
        validation["X"],
        validation["y"],
        ridge_val_z,
    )

    ResidualModel = build_residual_model_class(torch, nn)

    # ------------------------------------------------------------------
    # Stage 3
    # ------------------------------------------------------------------
    core.log(
        "[STAGE 3/8] Training matched Residual-GRU and Physics-Residual-GRU"
    )

    variants = ["Residual-GRU", "Physics-Residual-GRU"]
    trained = {}
    histories = {}
    training_rows = []
    metric_frames = [ridge_val_metrics]
    gate_frames = []
    correction_frames = []

    for variant_name in variants:
        core.log("")
        core.log("=" * 100)
        core.log(f"TRAINING {variant_name}")
        core.log("=" * 100)

        # Identical seed and loader initialization for matched ablation.
        core.set_global_seed(
            torch,
            matched_seed,
            args.deterministic,
        )

        train_loader = make_loader(
            DataLoader,
            train_ds,
            args.batch_size,
            True,
            args.num_workers,
            pin_memory,
            persistent_workers,
            args.prefetch_factor,
        )
        val_loader = make_loader(
            DataLoader,
            val_ds,
            args.batch_size,
            False,
            args.num_workers,
            pin_memory,
            persistent_workers,
            args.prefetch_factor,
        )

        model = ResidualModel(
            n_features=len(feature_names),
            n_horizons=len(horizons),
            n_targets=len(target_names),
            recurrent_units=gru_config["recurrent_units"],
            recurrent_layers=gru_config["recurrent_layers"],
            dense_units=gru_config["dense_units"],
            dropout=gru_config["dropout"],
            gate_bias=args.gate_bias,
            residual_init_std=args.residual_init_std,
        ).to(device)

        n_params = count_parameters(model)
        model_path = (
            model_dir
            / f"{variant_name.replace('-', '_')}.pt"
        )

        result = train_variant(
            core,
            torch,
            model=model,
            variant_name=variant_name,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            amp_enabled=amp_enabled,
            learning_rate=gru_config["learning_rate"],
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
            checkpoint_path=model_path,
            horizons=horizons,
        )

        hist = result["history"]
        histories[variant_name] = hist
        hist.to_csv(
            history_dir
            / f"{variant_name.replace('-', '_')}_history.csv",
            index=False,
            encoding="utf-8-sig",
        )

        val_result = result["validation"]

        val_metrics = core.evaluate_model(
            truth_joint=validation["y_raw"],
            pred_joint=val_result["pred_raw"],
            truth_apparent_ref=validation["apparent_earth_raw"],
            horizons=horizons,
            split_name="validation",
            model_name=variant_name,
            direction_min_speed=args.direction_min_speed,
            cog_min_speed=args.cog_min_speed,
            persistence_joint=persistence_val,
        )
        metric_frames.append(val_metrics)

        gate_frames.append(
            gate_statistics(
                variant_name,
                "validation",
                val_result["gate"],
                horizons,
                target_names,
            )
        )
        correction_frames.append(
            correction_statistics(
                variant_name,
                "validation",
                val_result["correction_z"],
                horizons,
                target_names,
                target_std,
            )
        )

        save_prediction_bundle(
            prediction_dir
            / (
                f"validation_predictions_"
                f"{variant_name.replace('-', '_')}.npz"
            ),
            variant_name,
            validation["y_raw"],
            val_result["pred_raw"],
            ridge_val_raw,
            validation["context_end_time_ns"],
            validation["target_time_ns"],
            horizons,
            gate=val_result["gate"],
            correction_z=val_result["correction_z"],
        )

        s = core.summarize_metrics(val_metrics).iloc[0]

        training_rows.append({
            "model": variant_name,
            "seed": matched_seed,
            "parameters": n_params,
            "recurrent_units": gru_config["recurrent_units"],
            "recurrent_layers": gru_config["recurrent_layers"],
            "dropout": gru_config["dropout"],
            "dense_units": gru_config["dense_units"],
            "learning_rate": gru_config["learning_rate"],
            "lambda_residual": args.lambda_residual,
            "lambda_aw_earth": (
                physics_weights["earth"]
                if variant_name == "Physics-Residual-GRU"
                else 0.0
            ),
            "lambda_aw_body": (
                physics_weights["body"]
                if variant_name == "Physics-Residual-GRU"
                else 0.0
            ),
            "lambda_heading_geo": (
                physics_weights["heading_geo"]
                if variant_name == "Physics-Residual-GRU"
                else 0.0
            ),
            "lambda_heading_norm": (
                physics_weights["heading_norm"]
                if variant_name == "Physics-Residual-GRU"
                else 0.0
            ),
            "best_epoch": result["best_epoch"],
            "epochs_ran": result["epochs_ran"],
            "training_seconds": result["training_seconds"],
            "validation_AW_RMSE_mps": float(
                s["mean_apparent_vector_RMSE_mps"]
            ),
            "validation_AWS_RMSE_mps": float(
                s["mean_AWS_RMSE_mps"]
            ),
            "validation_AWA_MAE_deg": float(
                s["mean_AWA_MAE_deg"]
            ),
            "validation_wind_RMSE_mps": float(
                s["mean_wind_vector_RMSE_mps"]
            ),
            "validation_vessel_RMSE_mps": float(
                s["mean_vessel_vector_RMSE_mps"]
            ),
            "validation_HDG_MAE_deg": float(
                s["mean_HDG_MAE_deg"]
            ),
            "validation_mean_gate": float(
                np.mean(val_result["gate"])
            ),
            "validation_correction_z_RMS": float(
                np.sqrt(
                    np.mean(val_result["correction_z"] ** 2)
                )
            ),
        })

        trained[variant_name] = {
            "model": model,
            "validation": val_result,
        }

        core.log(
            f"[VALIDATION] {variant_name}: "
            f"AW={s['mean_apparent_vector_RMSE_mps']:.4f} m/s | "
            f"AWS={s['mean_AWS_RMSE_mps']:.4f} m/s | "
            f"AWA={s['mean_AWA_MAE_deg']:.2f} deg | "
            f"wind={s['mean_wind_vector_RMSE_mps']:.4f} m/s | "
            f"vessel={s['mean_vessel_vector_RMSE_mps']:.4f} m/s | "
            f"gate={np.mean(val_result['gate']):.3f}"
        )

    training_df = pd.DataFrame(training_rows)
    training_df.to_csv(
        output_dir / "residual_training_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_prediction_bundle(
        prediction_dir / "validation_predictions_Ridge.npz",
        "Ridge",
        validation["y_raw"],
        ridge_val_raw,
        ridge_val_raw,
        validation["context_end_time_ns"],
        validation["target_time_ns"],
        horizons,
    )

    # ------------------------------------------------------------------
    # Stage 4 - optional development test
    # ------------------------------------------------------------------
    if args.evaluate_development_test:
        core.log("")
        core.log(
            "[STAGE 4/8] Evaluating already-inspected SD1090 development test"
        )

        ridge_test_z = ridge_predict_z(
            ridge_model,
            test["X"],
            len(horizons),
            len(target_names),
            args.ridge_chunk_size,
        )
        ridge_test_raw = inverse_target(
            ridge_test_z,
            target_mean,
            target_std,
        )

        persistence_test = core.persistence_joint_prediction(
            test["X"],
            feature_names,
            feature_mean,
            feature_std,
            len(horizons),
        )

        metric_frames.append(
            core.evaluate_model(
                truth_joint=test["y_raw"],
                pred_joint=ridge_test_raw,
                truth_apparent_ref=test["apparent_earth_raw"],
                horizons=horizons,
                split_name="development_test",
                model_name="Ridge",
                direction_min_speed=args.direction_min_speed,
                cog_min_speed=args.cog_min_speed,
                persistence_joint=persistence_test,
            )
        )

        test_ds = make_dataset(
            torch,
            TensorDataset,
            test["X"],
            test["y"],
            ridge_test_z,
        )
        test_loader = make_loader(
            DataLoader,
            test_ds,
            args.batch_size,
            False,
            args.num_workers,
            pin_memory,
            persistent_workers,
            args.prefetch_factor,
        )

        for variant_name in variants:
            result = evaluate_loader(
                core,
                torch,
                model=trained[variant_name]["model"],
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
                use_physics=(
                    variant_name == "Physics-Residual-GRU"
                ),
            )

            m = core.evaluate_model(
                truth_joint=test["y_raw"],
                pred_joint=result["pred_raw"],
                truth_apparent_ref=test["apparent_earth_raw"],
                horizons=horizons,
                split_name="development_test",
                model_name=variant_name,
                direction_min_speed=args.direction_min_speed,
                cog_min_speed=args.cog_min_speed,
                persistence_joint=persistence_test,
            )
            metric_frames.append(m)

            gate_frames.append(
                gate_statistics(
                    variant_name,
                    "development_test",
                    result["gate"],
                    horizons,
                    target_names,
                )
            )
            correction_frames.append(
                correction_statistics(
                    variant_name,
                    "development_test",
                    result["correction_z"],
                    horizons,
                    target_names,
                    target_std,
                )
            )

            save_prediction_bundle(
                prediction_dir
                / (
                    f"development_test_predictions_"
                    f"{variant_name.replace('-', '_')}.npz"
                ),
                variant_name,
                test["y_raw"],
                result["pred_raw"],
                ridge_test_raw,
                test["context_end_time_ns"],
                test["target_time_ns"],
                horizons,
                gate=result["gate"],
                correction_z=result["correction_z"],
            )

            s = core.summarize_metrics(m).iloc[0]
            core.log(
                f"[DEVELOPMENT TEST] {variant_name}: "
                f"AW={s['mean_apparent_vector_RMSE_mps']:.4f} m/s | "
                f"AWS={s['mean_AWS_RMSE_mps']:.4f} m/s | "
                f"AWA={s['mean_AWA_MAE_deg']:.2f} deg"
            )
    else:
        core.log("")
        core.log(
            "[STAGE 4/8] SD1090 development test intentionally NOT evaluated"
        )

    # ------------------------------------------------------------------
    # Stage 5 - merge upstream validation metrics
    # ------------------------------------------------------------------
    core.log("[STAGE 5/8] Building comparison metrics")

    if not args.skip_upstream_metric_merge:
        upstream_path = (
            physics_weight_dir
            / "physics_weight_metrics_per_horizon.csv"
        )

        if upstream_path.exists():
            upstream = pd.read_csv(upstream_path)
            upstream = upstream.loc[
                upstream["split"] == "validation"
            ].copy()

            # Ridge is regenerated locally from the actual frozen model.
            upstream = upstream.loc[
                upstream["model"] != "Ridge"
            ]

            required = set(ridge_val_metrics.columns)
            if required.issubset(upstream.columns):
                upstream = upstream[
                    list(ridge_val_metrics.columns)
                ]
                metric_frames.append(upstream)
                core.log(
                    f"[OK] Merged upstream validation metrics: {upstream_path}"
                )
            else:
                core.log(
                    "[WARNING] Upstream metric schema mismatch; not merged."
                )

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics = metrics.drop_duplicates(
        subset=["split", "model", "horizon_min"],
        keep="first",
    )

    order = {
        name: i
        for i, name in enumerate([
            "Persistence",
            "Ridge",
            "LSTM",
            "GRU",
            "CNN-LSTM",
            "Direct-GRU-Control",
            "Tuned-Physics-GRU",
            "Residual-GRU",
            "Physics-Residual-GRU",
        ])
    }

    metrics["_order"] = (
        metrics["model"].map(order).fillna(999)
    )
    metrics = (
        metrics.sort_values(
            ["split", "_order", "horizon_min"]
        )
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    metrics.to_csv(
        output_dir / "residual_metrics_per_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = core.summarize_metrics(metrics)
    summary["_order"] = (
        summary["model"].map(order).fillna(999)
    )
    summary = (
        summary.sort_values(["split", "_order"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    summary.to_csv(
        output_dir / "residual_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gate_stats = pd.concat(gate_frames, ignore_index=True)
    correction_stats = pd.concat(
        correction_frames,
        ignore_index=True,
    )

    gate_stats.to_csv(
        output_dir / "gate_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    correction_stats.to_csv(
        output_dir / "correction_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Stage 6 - scientific effect audit
    # ------------------------------------------------------------------
    core.log("[STAGE 6/8] Residual and physics-effect audit")

    val_summary = summary.loc[
        summary["split"] == "validation"
    ]

    def get_row(model_name):
        row = val_summary.loc[
            val_summary["model"] == model_name
        ]
        return None if row.empty else row.iloc[0]

    ridge_row = get_row("Ridge")
    residual_row = get_row("Residual-GRU")
    physics_row = get_row("Physics-Residual-GRU")

    effect = {}

    def reduction(a, b):
        # positive = b improves over a
        return 100.0 * (a - b) / a

    if ridge_row is not None and residual_row is not None:
        effect["Residual_GRU_AW_reduction_vs_Ridge_percent"] = reduction(
            float(ridge_row["mean_apparent_vector_RMSE_mps"]),
            float(residual_row["mean_apparent_vector_RMSE_mps"]),
        )

    if residual_row is not None and physics_row is not None:
        effect[
            "Physics_Residual_AW_reduction_vs_Residual_GRU_percent"
        ] = reduction(
            float(residual_row["mean_apparent_vector_RMSE_mps"]),
            float(physics_row["mean_apparent_vector_RMSE_mps"]),
        )

    if ridge_row is not None and physics_row is not None:
        effect["Physics_Residual_AW_reduction_vs_Ridge_percent"] = reduction(
            float(ridge_row["mean_apparent_vector_RMSE_mps"]),
            float(physics_row["mean_apparent_vector_RMSE_mps"]),
        )

    params = training_df["parameters"].astype(int).unique()
    effect["Residual_variants_same_parameter_count"] = bool(
        len(params) == 1
    )
    effect["Residual_variant_trainable_parameters"] = (
        int(params[0])
        if len(params) == 1
        else [int(v) for v in params]
    )

    # ------------------------------------------------------------------
    # Stage 7 - manifest/report
    # ------------------------------------------------------------------
    core.log("[STAGE 7/8] Writing manifest and report")

    out_manifest = {
        "script_version": SCRIPT_VERSION,
        "backend": "PyTorch",
        "task_name": manifest.get("task_name"),
        "source_dataset_version": manifest.get("dataset_version"),
        "dataset_dir": str(dataset_dir.resolve()),
        "frozen_ridge": {
            "model_path": str(ridge_model_path.resolve()),
            "manifest": ridge_manifest,
            "reproduction_audit": ridge_audit,
        },
        "fixed_gru_config": gru_config,
        "tuning_metadata": tuning_meta,
        "fixed_06B_physics_weights": physics_weights,
        "physics_weight_metadata": physics_meta,
        "matched_seed": matched_seed,
        "residual_architecture": {
            "ridge_conditioned_decoder": True,
            "sample_horizon_target_specific_sigmoid_gate": True,
            "gate_bias_initial": args.gate_bias,
            "residual_head_weight_init_std": args.residual_init_std,
        },
        "loss": {
            "lambda_residual": args.lambda_residual,
            "residual_penalty": "mean((gate * delta_z)^2)",
            "physics_weights_only_on_Physics_Residual_GRU": physics_weights,
        },
        "train_only_physics_scales": physics_scales,
        "protocol": {
            "ridge_frozen": True,
            "ridge_not_refit_or_reselected": True,
            "train_split": "frozen outer train",
            "checkpoint_split": "frozen outer validation",
            "same_seed_and_architecture_for_residual_ablation": True,
            "development_test_evaluated": bool(
                args.evaluate_development_test
            ),
            "development_test_used_for_selection": False,
        },
        "effect_audit": effect,
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
        output_dir / "residual_model_manifest.json",
        out_manifest,
    )

    with (
        output_dir / "residual_model_report.txt"
    ).open("w", encoding="utf-8") as f:
        f.write("07A Physics-Guided Residual Forecasting Report\n")
        f.write("=" * 100 + "\n\n")

        f.write("Fixed GRU configuration\n")
        f.write("-" * 100 + "\n")
        f.write(json.dumps(gru_config, ensure_ascii=False, indent=2))

        f.write("\n\nFixed 06B physics weights\n")
        f.write("-" * 100 + "\n")
        f.write(
            json.dumps(
                physics_weights,
                ensure_ascii=False,
                indent=2,
            )
        )

        f.write("\n\nTraining summary\n")
        f.write("-" * 100 + "\n")
        f.write(training_df.to_string(index=False))

        f.write("\n\nMetrics summary\n")
        f.write("-" * 100 + "\n")
        f.write(summary.to_string(index=False))

        f.write("\n\nEffect audit\n")
        f.write("-" * 100 + "\n")
        f.write(json.dumps(effect, ensure_ascii=False, indent=2))

        f.write("\n\nRidge reproduction audit\n")
        f.write("-" * 100 + "\n")
        f.write(
            json.dumps(
                ridge_audit,
                ensure_ascii=False,
                indent=2,
            )
        )
        f.write("\n")

    # ------------------------------------------------------------------
    # Stage 8 - plots/console
    # ------------------------------------------------------------------
    core.log("[STAGE 8/8] Finalization")

    if args.plots:
        make_plots(
            core,
            metrics,
            gate_stats,
            correction_stats,
            histories,
            output_dir,
            args.evaluate_development_test,
        )
        core.log("[OK] Figures complete")

    core.log("")
    core.log("=" * 100)
    core.log("07A PHYSICS-GUIDED RESIDUAL RESULTS")
    core.log("=" * 100)

    for model_name in [
        "Ridge",
        "Residual-GRU",
        "Physics-Residual-GRU",
    ]:
        row = get_row(model_name)
        if row is None:
            continue

        core.log(f"{model_name}:")
        core.log(
            f"  validation wind RMSE    = "
            f"{row['mean_wind_vector_RMSE_mps']:.4f} m/s"
        )
        core.log(
            f"  validation vessel RMSE  = "
            f"{row['mean_vessel_vector_RMSE_mps']:.4f} m/s"
        )
        core.log(
            f"  validation AW RMSE      = "
            f"{row['mean_apparent_vector_RMSE_mps']:.4f} m/s"
        )
        core.log(
            f"  validation AWS RMSE     = "
            f"{row['mean_AWS_RMSE_mps']:.4f} m/s"
        )
        core.log(
            f"  validation AWA MAE      = "
            f"{row['mean_AWA_MAE_deg']:.2f} deg"
        )
        core.log(
            f"  validation HDG MAE      = "
            f"{row['mean_HDG_MAE_deg']:.2f} deg"
        )

    core.log("")
    for key, value in effect.items():
        core.log(f"[EFFECT] {key} = {value}")

    physics_gate = gate_stats.loc[
        (gate_stats["split"] == "validation")
        & (gate_stats["model"] == "Physics-Residual-GRU")
        & (gate_stats["group"] == "all")
    ].sort_values("horizon_min")

    if not physics_gate.empty:
        text = ", ".join(
            f"{int(r.horizon_min)}min={r.mean_gate:.3f}"
            for r in physics_gate.itertuples()
        )
        core.log(f"[PHYSICS GATE BY HORIZON] {text}")

    if not args.evaluate_development_test:
        core.log(
            "[TEST POLICY] SD1090 test not evaluated. "
            "Use --evaluate-development-test only for an explicit "
            "development diagnostic."
        )

    core.log(f"[DEVICE] {device}")
    if device.type == "cuda":
        core.log(f"[GPU] {torch.cuda.get_device_name(device)}")

    core.log(f"[DONE] 07A outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n[FATAL ERROR]", flush=True)
        traceback.print_exc()
        print(
            "\nPlease send the complete traceback and terminal output.",
            flush=True,
        )
        sys.exit(1)
