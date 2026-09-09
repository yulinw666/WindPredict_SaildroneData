# -*- coding: utf-8 -*-
r"""
12L_diagnose_PhysicsCompact_NieData_v4_OOFResidualGate.py

Development-only residual diagnosis and channel-wise post-calibration for
Physics-Compact-NieData v4.

Purpose
-------
Stage 12J already uses sample-dependent learned gates:

    W_hat_z = W_ridge_z + sigmoid(gW) * DeltaW_z

Therefore, this script does NOT add another neural gate. Instead, it asks a
narrower diagnostic question:

    After the learned sample-wise gate, is the FINAL wind correction
    (W_hat - W_ridge) systematically over- or under-scaled in U or V?

For each selected 12J LOMO checkpoint:
    true residual      r = W_true - W_ridge
    model correction  c = W_v4   - W_ridge

The least-squares scalar calibration for each wind component is:

    alpha* = sum(c * r) / sum(c^2)

and the constrained shrinkage coefficient is:

    alpha = clip(alpha*, 0, 1)

The script reports:
    - residual correlation
    - residual/correction RMS
    - sign agreement
    - unconstrained and clipped alpha_U / alpha_V
    - per-run Ridge and v4 metrics
    - pooled development-OOF alpha_U / alpha_V
    - v4 metrics after applying the pooled OOF channel calibration

SCIENTIFIC FIREWALL
-------------------
This script reads ONLY the three Stage-12G development missions:
    Antarctic.npz
    Atlantic.npz
    West_Coast.npz

It does NOT read Tropical_Atlantic_TEST.npz.
It does NOT read Stage-12K outputs.
No test metric is used to estimate alpha.

The output is intended to decide whether a v5 "OOF-calibrated residual"
mechanism is justified before changing the final model.

Recommended usage (PowerShell, one line)
----------------------------------------
python "D:\project\WindPredict_SaildroneData\src\12L_diagnose_PhysicsCompact_NieData_v4_OOFResidualGate.py" --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" --j-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12J_PhysicsCompact_NieData_v4_RidgeResidual_LOMO_point_v0_1" --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12L_PhysicsCompact_NieData_v4_OOFResidualGate_v0_1"
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0-PhysicsCompact-NieData-v4-OOFResidualGate-Diagnostic"

DEFAULT_PROJECT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")
DEFAULT_DATASET_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12G_Nie_point_sampled_benchmark_v0_1"
    / "dataset"
)
DEFAULT_J_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12J_PhysicsCompact_NieData_v4_RidgeResidual_LOMO_point_v0_1"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "forecasting"
    / "12L_PhysicsCompact_NieData_v4_OOFResidualGate_v0_1"
)

DEVELOPMENT_MISSIONS = ["Antarctic", "Atlantic", "West Coast"]
EPS = 1e-12


def log(msg=""):
    print(msg, flush=True)


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
        if isinstance(x, dict):
            return {str(k): cv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [cv(v) for v in x]
        return x

    with path.open("w", encoding="utf-8") as f:
        json.dump(cv(obj), f, ensure_ascii=False, indent=2, sort_keys=True)


def load_module(path: Path, name: str):
    """Robust dynamic import; sys.modules registration is required by dataclasses."""
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for: {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def torch_load_full(torch, path: Path, device):
    """Compatible with PyTorch versions that default weights_only differently."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_checkpoint_path(j_dir: Path, row) -> Path:
    raw = str(getattr(row, "checkpoint_path", "") or "").strip()
    candidates = []

    if raw:
        candidates.append(Path(raw))
        candidates.append(j_dir / "checkpoints" / Path(raw).name)

    method = str(row.method)
    mission = str(row.held_out_mission)
    seed = int(row.seed)
    tag = (
        f"{method}__holdout_{mission.replace(' ', '_')}"
        f"__seed_{seed}.pt"
    )
    candidates.append(j_dir / "checkpoints" / tag)

    for p in candidates:
        if p.exists():
            return p

    lines = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not resolve 12J checkpoint for "
        f"method={method}, mission={mission}, seed={seed}.\nTried:\n{lines}"
    )


def corrcoef_safe(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 3:
        return np.nan
    if np.std(a) < EPS or np.std(b) < EPS:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


def rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def optimal_alpha(correction, true_residual):
    c = np.asarray(correction, dtype=np.float64).reshape(-1)
    r = np.asarray(true_residual, dtype=np.float64).reshape(-1)
    mask = np.isfinite(c) & np.isfinite(r)
    c = c[mask]
    r = r[mask]

    denom = float(np.dot(c, c))
    if denom <= EPS:
        return np.nan, np.nan

    alpha_raw = float(np.dot(c, r) / denom)
    alpha_clip = float(np.clip(alpha_raw, 0.0, 1.0))
    return alpha_raw, alpha_clip


def sign_accuracy(correction, true_residual):
    c = np.asarray(correction, dtype=np.float64).reshape(-1)
    r = np.asarray(true_residual, dtype=np.float64).reshape(-1)
    mask = np.isfinite(c) & np.isfinite(r) & (np.abs(r) > 1e-8)
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.sign(c[mask]) == np.sign(r[mask])))


def infer_model(
    *,
    base,
    jmod,
    torch,
    model,
    X,
    ridge_pred_z,
    scaler,
    device,
    amp_enabled,
):
    model.eval()
    Xz = base.standardize_X(X, scaler)

    pred_chunks = []
    gate_chunks = []
    delta_chunks = []
    correction_chunks = []

    with torch.no_grad():
        for xb_np, rb_np in base.sequential_batches(
            Xz,
            ridge_pred_z,
            batch_size=jmod.FREEZE.batch_size,
        ):
            xb = torch.from_numpy(xb_np).to(device, non_blocking=True)
            rb = torch.from_numpy(rb_np).to(device, non_blocking=True)

            with base.autocast_context(torch, amp_enabled):
                out = model(xb, rb)

            pred_chunks.append(
                out["pred_z"].detach().float().cpu().numpy()
            )
            gate_chunks.append(
                out["wind_gate"].detach().float().cpu().numpy()
            )
            delta_chunks.append(
                out["wind_delta_z"].detach().float().cpu().numpy()
            )
            correction_chunks.append(
                out["wind_correction_z"].detach().float().cpu().numpy()
            )

    pred_z = np.concatenate(pred_chunks, axis=0)
    pred_raw = base.inverse_y(pred_z, scaler)

    gate = np.concatenate(gate_chunks, axis=0)
    delta_z = np.concatenate(delta_chunks, axis=0)
    correction_z = np.concatenate(correction_chunks, axis=0)

    return {
        "pred_raw": pred_raw,
        "pred_z": pred_z,
        "wind_gate": gate,
        "wind_delta_z": delta_z,
        "wind_correction_z": correction_z,
    }


def apply_wind_alpha(pred_raw, ridge_raw, alpha_u, alpha_v):
    out = np.asarray(pred_raw, dtype=np.float64).copy()
    ridge = np.asarray(ridge_raw, dtype=np.float64)

    out[:, 0, 0] = (
        ridge[:, 0, 0]
        + float(alpha_u) * (out[:, 0, 0] - ridge[:, 0, 0])
    )
    out[:, 0, 1] = (
        ridge[:, 0, 1]
        + float(alpha_v) * (out[:, 0, 1] - ridge[:, 0, 1])
    )
    return out.astype(np.float32, copy=False)


def metric_subset(metrics):
    keys = [
        "wind_U_RMSE_mps",
        "wind_V_RMSE_mps",
        "wind_vector_RMSE_mps",
        "wind_speed_RMSE_mps",
        "wind_direction_MAE_deg",
        "wind_direction_RMSE_deg",
        "vessel_vector_RMSE_mps",
        "AW_vector_RMSE_mps",
        "AWS_RMSE_mps",
    ]
    return {k: float(metrics[k]) for k in keys if k in metrics}


def aggregate_metric_rows(rows: pd.DataFrame, method_name: str):
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
    out = {"method": method_name, "runs": int(len(rows))}
    for col in metric_cols:
        if col in rows.columns:
            vals = rows[col].to_numpy(dtype=float)
            out[f"{col}_mean"] = float(np.mean(vals))
            out[f"{col}_std"] = float(np.std(vals, ddof=0))
    return out


def recommendation_text(alpha_u, alpha_v, corr_u, corr_v):
    def one(comp, alpha, corr):
        if not np.isfinite(alpha):
            return f"{comp}: correction is numerically degenerate; do not calibrate yet."
        if not np.isfinite(corr):
            return f"{comp}: residual correlation is undefined/unstable."

        if alpha <= 0.15:
            return (
                f"{comp}: strong suppression recommended "
                f"(alpha={alpha:.3f}, corr={corr:.3f}); "
                "the v4 correction is weak or poorly aligned."
            )
        if alpha < 0.75:
            return (
                f"{comp}: shrinkage is justified "
                f"(alpha={alpha:.3f}, corr={corr:.3f}); "
                "the v4 correction appears systematically too large."
            )
        if alpha <= 1.0:
            return (
                f"{comp}: little/moderate shrinkage "
                f"(alpha={alpha:.3f}, corr={corr:.3f})."
            )
        return (
            f"{comp}: unconstrained expansion would be suggested, but the "
            "scientific v5 gate is constrained to [0,1]."
        )

    return [one("U", alpha_u, corr_u), one("V", alpha_v, corr_v)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=DEFAULT_PROJECT_ROOT
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR
    )
    parser.add_argument(
        "--j-dir", type=Path, default=DEFAULT_J_DIR
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--force-cpu", action="store_true"
    )
    parser.add_argument(
        "--save-oof-npz", action="store_true",
        help="Save pooled development-OOF residual arrays for later plotting."
    )
    args = parser.parse_args()

    project_root = args.project_root
    dataset_dir = args.dataset_dir
    j_dir = args.j_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Explicitly load only 12C + 12J source modules.
    base_path = (
        project_root / "src" / "12C_train_PhysicsCompact_NieData_LOMO.py"
    )
    j_script_path = (
        project_root
        / "src"
        / "12J_train_PhysicsCompact_NieData_v4_RidgeResidual_LOMO.py"
    )

    base = load_module(base_path, "stage12c_base_for_12l")
    jmod = load_module(j_script_path, "stage12j_model_for_12l")

    torch, nn, _ = base.import_torch()
    base.configure_cuda(torch)

    if args.force_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    selected_path = j_dir / "selected_config.json"
    selected_runs_path = j_dir / "selected_candidate_lomo_runs.csv"

    if not selected_path.exists():
        raise FileNotFoundError(selected_path)
    if not selected_runs_path.exists():
        raise FileNotFoundError(selected_runs_path)

    with selected_path.open("r", encoding="utf-8") as f:
        selected = json.load(f)

    selected_name = str(selected["selected_candidate"])
    selected_runs = pd.read_csv(selected_runs_path)

    selected_runs = selected_runs.loc[
        (selected_runs["method"].astype(str) == selected_name)
        & (selected_runs["status"].astype(str) == "completed_finite")
    ].copy()

    expected = len(DEVELOPMENT_MISSIONS) * len(
        selected.get("screening_seeds", jmod.SCREENING_SEEDS)
    )
    if len(selected_runs) != expected:
        raise RuntimeError(
            f"Expected {expected} completed selected-candidate LOMO rows, "
            f"found {len(selected_runs)}."
        )

    # This call is development-only by design in Stage 12C.
    dev_paths = base.explicit_development_paths(dataset_dir)

    # Hard firewall audit: refuse any path whose filename contains Tropical.
    for mission, p in dev_paths.items():
        if "tropical" in str(p).lower():
            raise RuntimeError(
                "FIREWALL VIOLATION: a development path contains 'tropical': "
                f"{mission} -> {p}"
            )

    missions = {
        m: base.load_development_mission(dev_paths[m], m)
        for m in DEVELOPMENT_MISSIONS
    }

    log("=" * 126)
    log("12L — Physics-Compact-NieData v4 DEVELOPMENT-ONLY OOF RESIDUAL DIAGNOSIS")
    log("=" * 126)
    log(f"selected candidate : {selected_name}")
    log(f"dataset            : {dataset_dir}")
    log(f"12J directory      : {j_dir}")
    log(f"output             : {output_dir}")
    log(f"device             : {device}")
    log("[FIREWALL] Only Antarctic / Atlantic / West Coast are loaded.")
    log("[FIREWALL] Stage 12K and Tropical Atlantic are not read.")
    log("")

    ModelClass = jmod.build_model_class(
        torch, nn, len(base.FEATURE_NAMES)
    )
    amp_enabled = bool(jmod.FREEZE.use_amp and device.type == "cuda")

    diagnostic_rows = []
    cache = []

    pooled_true_u = []
    pooled_true_v = []
    pooled_corr_u = []
    pooled_corr_v = []

    for row in selected_runs.itertuples(index=False):
        held_out = str(row.held_out_mission)
        seed = int(row.seed)

        if held_out not in DEVELOPMENT_MISSIONS:
            raise RuntimeError(
                f"Unexpected held-out mission in selected runs: {held_out}"
            )

        train_names = [
            m for m in DEVELOPMENT_MISSIONS if m != held_out
        ]
        train = base.concat_missions(
            [missions[m] for m in train_names]
        )
        val = missions[held_out]

        ckpt_path = resolve_checkpoint_path(j_dir, row)
        ckpt = torch_load_full(torch, ckpt_path, device)
        scaler = ckpt["scaler"]

        # Refit the exact fold-specific Ridge anchor used by 12J.
        ridge = base.fit_ridge(
            train["X"], train["y"], scaler
        )
        ridge_val_z = base.ridge_predict_z(
            ridge, val["X"], scaler
        )
        ridge_raw = base.inverse_y(
            ridge_val_z, scaler
        )

        model = ModelClass().to(device)
        model.load_state_dict(ckpt["state_dict"], strict=True)

        inf = infer_model(
            base=base,
            jmod=jmod,
            torch=torch,
            model=model,
            X=val["X"],
            ridge_pred_z=ridge_val_z,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
        )
        pred_raw = inf["pred_raw"]

        ridge_metrics = base.evaluate_raw(
            val["y"], ridge_raw, val["aw"]
        )
        model_metrics = base.evaluate_raw(
            val["y"], pred_raw, val["aw"]
        )

        true_w = np.asarray(
            val["y"][:, 0, 0:2], dtype=np.float64
        )
        ridge_w = np.asarray(
            ridge_raw[:, 0, 0:2], dtype=np.float64
        )
        pred_w = np.asarray(
            pred_raw[:, 0, 0:2], dtype=np.float64
        )

        true_resid = true_w - ridge_w
        correction = pred_w - ridge_w

        alpha_u_raw, alpha_u = optimal_alpha(
            correction[:, 0], true_resid[:, 0]
        )
        alpha_v_raw, alpha_v = optimal_alpha(
            correction[:, 1], true_resid[:, 1]
        )

        corr_u = corrcoef_safe(
            correction[:, 0], true_resid[:, 0]
        )
        corr_v = corrcoef_safe(
            correction[:, 1], true_resid[:, 1]
        )

        calibrated_local = apply_wind_alpha(
            pred_raw, ridge_raw, alpha_u, alpha_v
        )
        calibrated_local_metrics = base.evaluate_raw(
            val["y"], calibrated_local, val["aw"]
        )

        gates = np.asarray(inf["wind_gate"], dtype=np.float64)

        d = {
            "method": selected_name,
            "held_out_mission": held_out,
            "seed": seed,
            "n_val": int(len(val["X"])),
            "checkpoint": str(ckpt_path),

            "residual_corr_U": corr_u,
            "residual_corr_V": corr_v,
            "sign_accuracy_U": sign_accuracy(
                correction[:, 0], true_resid[:, 0]
            ),
            "sign_accuracy_V": sign_accuracy(
                correction[:, 1], true_resid[:, 1]
            ),

            "true_residual_RMS_U_mps": rms(true_resid[:, 0]),
            "true_residual_RMS_V_mps": rms(true_resid[:, 1]),
            "model_correction_RMS_U_mps": rms(correction[:, 0]),
            "model_correction_RMS_V_mps": rms(correction[:, 1]),

            "alpha_U_unconstrained": alpha_u_raw,
            "alpha_V_unconstrained": alpha_v_raw,
            "alpha_U_clipped_0_1": alpha_u,
            "alpha_V_clipped_0_1": alpha_v,

            "learned_gate_U_mean": float(np.mean(gates[:, 0])),
            "learned_gate_V_mean": float(np.mean(gates[:, 1])),
            "learned_gate_U_std": float(np.std(gates[:, 0], ddof=0)),
            "learned_gate_V_std": float(np.std(gates[:, 1], ddof=0)),

            "ridge_U_RMSE_mps": float(
                ridge_metrics["wind_U_RMSE_mps"]
            ),
            "ridge_V_RMSE_mps": float(
                ridge_metrics["wind_V_RMSE_mps"]
            ),
            "v4_U_RMSE_mps": float(
                model_metrics["wind_U_RMSE_mps"]
            ),
            "v4_V_RMSE_mps": float(
                model_metrics["wind_V_RMSE_mps"]
            ),
            "v4_WS_RMSE_mps": float(
                model_metrics["wind_speed_RMSE_mps"]
            ),
            "v4_WD_RMSE_deg": float(
                model_metrics["wind_direction_RMSE_deg"]
            ),
            "local_alpha_U_RMSE_mps": float(
                calibrated_local_metrics["wind_U_RMSE_mps"]
            ),
            "local_alpha_V_RMSE_mps": float(
                calibrated_local_metrics["wind_V_RMSE_mps"]
            ),
            "local_alpha_WS_RMSE_mps": float(
                calibrated_local_metrics["wind_speed_RMSE_mps"]
            ),
            "local_alpha_WD_RMSE_deg": float(
                calibrated_local_metrics["wind_direction_RMSE_deg"]
            ),
        }
        diagnostic_rows.append(d)

        pooled_true_u.append(true_resid[:, 0])
        pooled_true_v.append(true_resid[:, 1])
        pooled_corr_u.append(correction[:, 0])
        pooled_corr_v.append(correction[:, 1])

        cache.append({
            "held_out_mission": held_out,
            "seed": seed,
            "y": val["y"],
            "aw": val["aw"],
            "ridge_raw": ridge_raw,
            "pred_raw": pred_raw,
            "ridge_metrics": ridge_metrics,
            "model_metrics": model_metrics,
        })

        log(
            f"[{held_out:10s} seed={seed}] "
            f"corr(U,V)=({corr_u:+.3f},{corr_v:+.3f}) | "
            f"alpha(U,V)=({alpha_u:.3f},{alpha_v:.3f}) | "
            f"RMSE V: Ridge={ridge_metrics['wind_V_RMSE_mps']:.4f} "
            f"v4={model_metrics['wind_V_RMSE_mps']:.4f}"
        )

        del model, ridge, ckpt, inf, train, val
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    diag_df = pd.DataFrame(diagnostic_rows)
    diag_path = output_dir / "12L_run_diagnostics.csv"
    diag_df.to_csv(diag_path, index=False, encoding="utf-8-sig")

    true_u = np.concatenate(pooled_true_u)
    true_v = np.concatenate(pooled_true_v)
    corr_u_arr = np.concatenate(pooled_corr_u)
    corr_v_arr = np.concatenate(pooled_corr_v)

    alpha_u_raw, alpha_u = optimal_alpha(corr_u_arr, true_u)
    alpha_v_raw, alpha_v = optimal_alpha(corr_v_arr, true_v)
    pooled_corrcoef_u = corrcoef_safe(corr_u_arr, true_u)
    pooled_corrcoef_v = corrcoef_safe(corr_v_arr, true_v)

    # Robust descriptive alternative: median across the 9 run-wise clipped alphas.
    median_alpha_u = float(
        np.nanmedian(diag_df["alpha_U_clipped_0_1"].to_numpy(dtype=float))
    )
    median_alpha_v = float(
        np.nanmedian(diag_df["alpha_V_clipped_0_1"].to_numpy(dtype=float))
    )

    comparison_rows = []
    ridge_rows = []
    raw_rows = []
    calibrated_rows = []

    for item in cache:
        ridge_metrics = item["ridge_metrics"]
        model_metrics = item["model_metrics"]
        cal_pred = apply_wind_alpha(
            item["pred_raw"],
            item["ridge_raw"],
            alpha_u,
            alpha_v,
        )
        cal_metrics = base.evaluate_raw(
            item["y"], cal_pred, item["aw"]
        )

        common = {
            "held_out_mission": item["held_out_mission"],
            "seed": item["seed"],
        }

        rr = {**common, **metric_subset(ridge_metrics)}
        vr = {**common, **metric_subset(model_metrics)}
        cr = {**common, **metric_subset(cal_metrics)}

        ridge_rows.append(rr)
        raw_rows.append(vr)
        calibrated_rows.append(cr)

        comparison_rows.extend([
            {"method": "Ridge", **rr},
            {"method": "PhysicsCompactNieDataV4", **vr},
            {
                "method": "PhysicsCompactNieDataV4+OOFChannelCalibration",
                **cr,
            },
        ])

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = output_dir / "12L_per_run_metric_comparison.csv"
    comparison_df.to_csv(
        comparison_path, index=False, encoding="utf-8-sig"
    )

    aggregate = pd.DataFrame([
        aggregate_metric_rows(pd.DataFrame(ridge_rows), "Ridge"),
        aggregate_metric_rows(
            pd.DataFrame(raw_rows), "PhysicsCompactNieDataV4"
        ),
        aggregate_metric_rows(
            pd.DataFrame(calibrated_rows),
            "PhysicsCompactNieDataV4+OOFChannelCalibration",
        ),
    ])
    aggregate_path = output_dir / "12L_aggregate_metric_comparison.csv"
    aggregate.to_csv(
        aggregate_path, index=False, encoding="utf-8-sig"
    )

    per_mission = (
        diag_df.groupby("held_out_mission", as_index=False)
        .agg({
            "residual_corr_U": "mean",
            "residual_corr_V": "mean",
            "alpha_U_clipped_0_1": "mean",
            "alpha_V_clipped_0_1": "mean",
            "learned_gate_U_mean": "mean",
            "learned_gate_V_mean": "mean",
            "ridge_U_RMSE_mps": "mean",
            "ridge_V_RMSE_mps": "mean",
            "v4_U_RMSE_mps": "mean",
            "v4_V_RMSE_mps": "mean",
        })
    )
    per_mission.to_csv(
        output_dir / "12L_mission_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    recommendations = recommendation_text(
        alpha_u, alpha_v, pooled_corrcoef_u, pooled_corrcoef_v
    )

    gate_info = {
        "stage": "12L",
        "script_version": SCRIPT_VERSION,
        "source_stage": "12J",
        "selected_candidate": selected_name,
        "development_missions": DEVELOPMENT_MISSIONS,
        "development_only": True,
        "Tropical_Atlantic_read": False,
        "Stage_12K_read": False,

        "pooled_OOF": {
            "alpha_U_unconstrained": alpha_u_raw,
            "alpha_V_unconstrained": alpha_v_raw,
            "alpha_U_clipped_0_1": alpha_u,
            "alpha_V_clipped_0_1": alpha_v,
            "residual_corr_U": pooled_corrcoef_u,
            "residual_corr_V": pooled_corrcoef_v,
            "sign_accuracy_U": sign_accuracy(corr_u_arr, true_u),
            "sign_accuracy_V": sign_accuracy(corr_v_arr, true_v),
            "true_residual_RMS_U_mps": rms(true_u),
            "true_residual_RMS_V_mps": rms(true_v),
            "model_correction_RMS_U_mps": rms(corr_u_arr),
            "model_correction_RMS_V_mps": rms(corr_v_arr),
        },
        "runwise_median_clipped_alpha": {
            "alpha_U": median_alpha_u,
            "alpha_V": median_alpha_v,
        },
        "recommended_v5_channel_calibration": {
            "alpha_U": alpha_u,
            "alpha_V": alpha_v,
            "equation_U": "U_v5 = U_ridge + alpha_U * (U_v4 - U_ridge)",
            "equation_V": "V_v5 = V_ridge + alpha_V * (V_v4 - V_ridge)",
            "constraint": "alpha_U, alpha_V in [0, 1]",
            "frozen_from": "12J selected-candidate development OOF predictions only",
        },
        "interpretation": recommendations,
    }
    gate_json_path = output_dir / "12L_gate_recommendation.json"
    save_json(gate_json_path, gate_info)

    if args.save_oof_npz:
        np.savez_compressed(
            output_dir / "12L_pooled_oof_residuals.npz",
            true_residual_U=true_u.astype(np.float32),
            true_residual_V=true_v.astype(np.float32),
            model_correction_U=corr_u_arr.astype(np.float32),
            model_correction_V=corr_v_arr.astype(np.float32),
        )

    # Human-readable report.
    report_path = output_dir / "12L_REPORT.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("12L Physics-Compact-NieData v4 Development-Only OOF Residual Diagnosis\n")
        f.write("=" * 126 + "\n\n")
        f.write(f"Selected candidate: {selected_name}\n")
        f.write("FIREWALL: Antarctic / Atlantic / West Coast only.\n")
        f.write("Tropical Atlantic read: False\n")
        f.write("Stage 12K read: False\n\n")

        f.write("POOLED DEVELOPMENT-OOF DIAGNOSIS\n")
        f.write("-" * 126 + "\n")
        f.write(
            f"U residual correlation       : {pooled_corrcoef_u:+.6f}\n"
            f"V residual correlation       : {pooled_corrcoef_v:+.6f}\n"
            f"U alpha unconstrained        : {alpha_u_raw:.6f}\n"
            f"V alpha unconstrained        : {alpha_v_raw:.6f}\n"
            f"U alpha clipped [0,1]        : {alpha_u:.6f}\n"
            f"V alpha clipped [0,1]        : {alpha_v:.6f}\n"
            f"U runwise median alpha       : {median_alpha_u:.6f}\n"
            f"V runwise median alpha       : {median_alpha_v:.6f}\n"
            f"U sign accuracy              : {sign_accuracy(corr_u_arr, true_u):.6f}\n"
            f"V sign accuracy              : {sign_accuracy(corr_v_arr, true_v):.6f}\n"
        )

        f.write("\nINTERPRETATION\n")
        f.write("-" * 126 + "\n")
        for line in recommendations:
            f.write(line + "\n")

        f.write("\nAGGREGATE DEVELOPMENT METRICS (mean across LOMO x seed runs)\n")
        f.write("-" * 126 + "\n")
        f.write(aggregate.to_string(index=False))
        f.write("\n\nPER-MISSION DIAGNOSIS\n")
        f.write("-" * 126 + "\n")
        f.write(per_mission.to_string(index=False))
        f.write("\n")

    log("")
    log("=" * 126)
    log("12L POOLED DEVELOPMENT-OOF RESULT")
    log("=" * 126)
    log(
        f"U: corr={pooled_corrcoef_u:+.4f} | "
        f"alpha_raw={alpha_u_raw:+.4f} | alpha_clip={alpha_u:.4f}"
    )
    log(
        f"V: corr={pooled_corrcoef_v:+.4f} | "
        f"alpha_raw={alpha_v_raw:+.4f} | alpha_clip={alpha_v:.4f}"
    )
    log("")
    log("Aggregate development metrics:")
    log(aggregate.to_string(index=False))
    log("")
    for line in recommendations:
        log("[INTERPRET] " + line)
    log("")
    log(f"[SAVED] {diag_path}")
    log(f"[SAVED] {aggregate_path}")
    log(f"[SAVED] {gate_json_path}")
    log(f"[SAVED] {report_path}")
    log("")
    log(
        "NEXT: If alpha_U/alpha_V are stable and the calibrated development "
        "metrics improve, freeze them into a v5 script. Do NOT change them "
        "using Tropical Atlantic."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n[FATAL ERROR]", flush=True)
        traceback.print_exc()
        sys.exit(1)
