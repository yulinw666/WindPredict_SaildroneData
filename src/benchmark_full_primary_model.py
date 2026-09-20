# -*- coding: utf-8 -*-
r"""
benchmark_full_primary_model.py

Local, same-machine benchmark for the complete 1-min Physics-Compact
15E-VH model used in the paper.

The timed path contains all deployed components:

    standardized X (60, 11)
      -> dedicated TrueWind Ridge + Stage15B residual NN (26,074)
      -> joint Ridge anchor + joint vessel/HDG residual NN (23,464)
      -> true-wind / vessel subtraction (apparent wind)

This is deliberately different from benchmark_current_hardware_14A.py,
which measures only the 10-min 14A compatibility model.

The script does not retrain anything.  It rebuilds the deterministic Ridge
anchors from the training split and loads frozen Stage15B and 15E-VH
checkpoints.  The input is the already standardized (N,60,11) dataset array,
which is the same boundary used by the model scripts.

Recommended PowerShell (from the project src directory)
--------------------------------------------------------
python .\benchmark_full_primary_model.py `
  --src-dir "D:\project\WindPredict_SaildroneData\src" `
  --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\SD1090_TPOS2024_JointForecasting_v0_1" `
  --stage15b-dir "D:\project\WindPredict_SaildroneData\data\forecasting\15F_Frozen_15EVH_alpha1500_External_v0_1" `
  --motion-dir "D:\project\WindPredict_SaildroneData\data\forecasting\15E_VH_AlphaSearch_v0_1\full_retrain\alpha_1500" `
  --seed 500043 `
  --warmup 100 `
  --repeats 300

If your frozen Stage15B checkpoints are in another directory, only change
--stage15b-dir.  The script accepts either the per-seed 15F checkpoint
(`15F_seed_<seed>_Stage15B_frozen.pt`) or the single 15B checkpoint
(`15B_truewind_residual.pt`).

Outputs
-------
full_primary_hardware_efficiency.csv
full_primary_hardware_efficiency.tex
full_primary_hardware_report.txt
hardware_metadata.json

Interpretation
--------------
The Physics-Compact timing is measured locally on the current machine.  The
BP-STGNN row contains its published parameter count only; its local timing is
left blank because that model is not executed here.  Report the measured
latency as per frozen seed, batch=1, and state separately if a five-seed
ensemble is deployed (approximately five sequential model evaluations).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0-local-full-primary-15E-VH"
SEEDS = [500043, 501052, 502061, 503070, 504079]

STAGE15B_PARAMS = 26_074
MOTION_PARAMS = 23_464
TOTAL_NEURAL_PARAMS = STAGE15B_PARAMS + MOTION_PARAMS
TRUEWIND_RIDGE_PARAMS = 2_410       # 240 inputs x 10 outputs + 10 intercepts
JOINT_RIDGE_PARAMS = 19_830         # 660 inputs x 30 outputs + 30 intercepts
FROZEN_RIDGE_PARAMS = TRUEWIND_RIDGE_PARAMS + JOINT_RIDGE_PARAMS
TOTAL_FITTED_COEFFICIENTS = TOTAL_NEURAL_PARAMS + FROZEN_RIDGE_PARAMS
BP_STGNN_PARAMS = 242_580
RIDGE_STATS_CHUNK = 8192
EPS = 1.0e-8

PROJECT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")
DEFAULT_SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT / "data" / "forecasting"
    / "SD1090_TPOS2024_JointForecasting_v0_1"
)
DEFAULT_STAGE15B_DIR = (
    PROJECT_ROOT / "data" / "forecasting"
    / "15F_Frozen_15EVH_alpha1500_External_v0_1"
)
DEFAULT_MOTION_DIR = (
    PROJECT_ROOT / "data" / "forecasting"
    / "15E_VH_AlphaSearch_v0_1" / "full_retrain" / "alpha_1500"
)


def log(message: str = "") -> None:
    print(message, flush=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)


def import_script(path: Path, tag: str):
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Required model script not found: {path}")
    spec = importlib.util.spec_from_file_location(f"{tag}_{abs(hash(str(path)))}", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def import_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required; activate the WindPredict environment first.") from exc
    return torch, nn


def select_device(torch, requested: str):
    requested = requested.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
        return torch.device("cuda:0")
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown --device value: {requested}")


def synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(torch, device, enabled: bool):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def count_parameters(model) -> int:
    return int(sum(int(p.numel()) for p in model.parameters() if p.requires_grad))


def checkpoint_size(path: Path) -> int:
    return int(path.stat().st_size) if path.exists() else 0


def load_split(dataset_dir: Path, split: str) -> Dict[str, np.ndarray]:
    path = dataset_dir / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    with np.load(path, allow_pickle=False) as z:
        if "X" not in z.files or "y_raw" not in z.files:
            raise KeyError(f"{path}: expected X and y_raw; keys={list(z.files)}")
        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y_raw"], dtype=np.float32)
    if X.ndim != 3 or X.shape[1:] != (60, 11):
        raise RuntimeError(f"{path}: X must have shape (N,60,11), got {X.shape}")
    if y.ndim != 3 or y.shape[1:] != (5, 6):
        raise RuntimeError(f"{path}: y_raw must have shape (N,5,6), got {y.shape}")
    if len(X) != len(y) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise RuntimeError(f"{path}: invalid length or non-finite values")
    return {"X": X, "y": y, "path": path}


def torch_load(path: Path, torch):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location="cpu")


def resolve_checkpoint(
    explicit: Optional[Path], directory: Path, seed: int, kind: str
) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Checkpoint not found: {explicit}")
        return explicit

    if kind == "wind":
        candidates = [
            directory / f"15F_seed_{seed}_Stage15B_frozen.pt",
            directory / f"15B_seed_{seed}_truewind_residual.pt",
            directory / "15B_truewind_residual.pt",
            directory.parent / "15B_1min_TrueWind_Residual_v0_1" / "15B_truewind_residual.pt",
        ]
    elif kind == "motion":
        candidates = [
            directory / f"15E_VH_seed_{seed}_joint_motion_model.pt",
            directory / f"15E_VH_seed_{seed}_motion_model.pt",
        ]
    else:
        raise ValueError(kind)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = "\n".join(f"  {p}" for p in candidates)
    raise FileNotFoundError(
        f"Could not find the frozen {kind} checkpoint for seed {seed}. Tried:\n{tried}"
    )


def _fit_ridge_from_standardized_targets(
    X: np.ndarray,
    y: np.ndarray,
    channels: Iterable[int],
    alpha: float,
) -> Dict[str, Any]:
    """Match the centered, standardized-target Ridge used by the model scripts."""
    channels = list(channels)
    Xf = np.asarray(X[:, :, channels], dtype=np.float64).reshape(len(X), -1)
    yf = np.asarray(y, dtype=np.float64).reshape(len(y), -1)

    y_mean_raw = yf.mean(axis=0)
    y_std_raw = yf.std(axis=0, ddof=0)
    y_std_raw = np.where(y_std_raw < EPS, 1.0, y_std_raw)
    yz = (yf - y_mean_raw[None, :]) / y_std_raw[None, :]

    p, q = Xf.shape[1], yz.shape[1]
    sum_x = np.zeros(p, dtype=np.float64)
    sum_x2 = np.zeros((p, p), dtype=np.float64)
    sum_y = np.zeros(q, dtype=np.float64)
    sum_xy = np.zeros((p, q), dtype=np.float64)
    n = 0
    for start in range(0, len(Xf), RIDGE_STATS_CHUNK):
        stop = min(start + RIDGE_STATS_CHUNK, len(Xf))
        xb = Xf[start:stop]
        yb = yz[start:stop]
        sum_x += xb.sum(axis=0)
        sum_x2 += xb.T @ xb
        sum_y += yb.sum(axis=0)
        sum_xy += xb.T @ yb
        n += stop - start

    n_float = float(n)
    x_mean = sum_x / n_float
    y_mean_z = sum_y / n_float
    G = sum_x2 - np.outer(sum_x, sum_x) / n_float
    G = 0.5 * (G + G.T)
    C = sum_xy - np.outer(sum_x, y_mean_z)
    regularizer = float(alpha) * np.eye(p, dtype=np.float64)
    try:
        coef_z = np.linalg.solve(G + regularizer, C)
        jitter = 0.0
    except np.linalg.LinAlgError:
        jitter = 1.0e-10
        coef_z = np.linalg.solve(G + regularizer + jitter * np.eye(p), C)
    intercept_z = y_mean_z - x_mean @ coef_z
    return {
        "coef_z": coef_z,
        "intercept_z": intercept_z,
        "y_mean_raw": y_mean_raw,
        "y_std_raw": y_std_raw,
        "alpha": float(alpha),
        "channels": channels,
        "jitter": jitter,
    }


def fit_anchors(train: Dict[str, np.ndarray], joint_alpha: float):
    X, y = train["X"], train["y"]
    truewind = _fit_ridge_from_standardized_targets(
        X, y[:, :, 0:2], channels=[0, 1, 2, 3], alpha=0.0
    )
    joint = _fit_ridge_from_standardized_targets(
        X, y, channels=list(range(11)), alpha=joint_alpha
    )
    return truewind, joint


def ridge_predict(model: Dict[str, Any], X: np.ndarray):
    channels = model["channels"]
    Xf = np.asarray(X[:, :, channels], dtype=np.float64).reshape(len(X), -1)
    z = Xf @ model["coef_z"] + model["intercept_z"][None, :]
    raw = z * model["y_std_raw"][None, :] + model["y_mean_raw"][None, :]
    if raw.shape[1] == 10:
        shape = (-1, 5, 2)
    elif raw.shape[1] == 30:
        shape = (-1, 5, 6)
    else:
        raise RuntimeError(f"Unexpected Ridge output width: {raw.shape[1]}")
    return z.astype(np.float32), raw.reshape(shape).astype(np.float32)


def normalize_heading_pair(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = np.sqrt(np.sum(q * q, axis=-1, keepdims=True))
    return (q / np.maximum(norm, EPS)).astype(np.float32)


def load_wind_bundle(torch, nn, bmod, checkpoint_path: Path, device):
    checkpoint = torch_load(checkpoint_path, torch)
    model = bmod.build_model_class(torch, nn)()
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state is None:
        raise KeyError(f"{checkpoint_path}: no model_state_dict/state_dict")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    params = count_parameters(model)
    if params != STAGE15B_PARAMS:
        raise RuntimeError(f"Stage15B parameter count {params} != {STAGE15B_PARAMS}")
    scaler = checkpoint.get("scaler", checkpoint.get("feature_scaler"))
    if scaler is None:
        raise KeyError(f"{checkpoint_path}: no scaler/feature_scaler")
    scaler = {k: np.asarray(v, dtype=np.float32) for k, v in scaler.items()}
    alpha = np.asarray(checkpoint.get("alpha"), dtype=np.float32)
    if alpha.shape != (5, 2):
        raise RuntimeError(f"{checkpoint_path}: alpha must have shape (5,2), got {alpha.shape}")
    return {"model": model, "scaler": scaler, "alpha": alpha, "checkpoint": checkpoint}


def load_motion_bundle(torch, nn, vmod, checkpoint_path: Path, device):
    checkpoint = torch_load(checkpoint_path, torch)
    model = vmod.build_joint_motion_model_class(torch, nn)()
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state is None:
        raise KeyError(f"{checkpoint_path}: no model_state_dict/state_dict")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    params = count_parameters(model)
    if params != MOTION_PARAMS:
        raise RuntimeError(f"15E-VH motion parameter count {params} != {MOTION_PARAMS}")
    if "ridge_alpha" not in checkpoint:
        raise KeyError(f"{checkpoint_path}: missing ridge_alpha metadata")
    return {"model": model, "checkpoint": checkpoint, "ridge_alpha": float(checkpoint["ridge_alpha"])}


def prepare_sample(torch, bmod, x_one, truewind_ridge, joint_ridge, wind_bundle, device):
    tw_z, tw_ridge = ridge_predict(truewind_ridge, x_one)
    joint_z, joint_raw = ridge_predict(joint_ridge, x_one)
    features = bmod.build_features(x_one)
    transformed = bmod.transform_features(features, wind_bundle["scaler"])
    wind_inputs = (
        torch.from_numpy(transformed["seq"]).to(device, non_blocking=True),
        torch.from_numpy(transformed["summary"]).to(device, non_blocking=True),
    )
    motion_inputs = (
        torch.from_numpy(x_one).to(device, non_blocking=True),
        torch.from_numpy(joint_z.reshape(len(x_one), -1)).to(device, non_blocking=True),
    )
    return {
        "tw_ridge": tw_ridge,
        "joint_z": joint_z.reshape(len(x_one), 5, 6),
        "joint_raw": joint_raw,
        "transformed": transformed,
        "wind_inputs": wind_inputs,
        "motion_inputs": motion_inputs,
    }


def motion_z_to_raw(joint_ridge: Dict[str, Any], motion_z: np.ndarray) -> np.ndarray:
    mean = joint_ridge["y_mean_raw"].reshape(5, 6)[:, 2:6]
    std = joint_ridge["y_std_raw"].reshape(5, 6)[:, 2:6]
    raw = motion_z * std[None, :, :] + mean[None, :, :]
    raw = raw.astype(np.float32)
    raw[:, :, 2:4] = normalize_heading_pair(raw[:, :, 2:4])
    return raw


def forward_preloaded(torch, wind_bundle, motion_bundle, prepared, device, amp: bool):
    with torch.inference_mode(), autocast_context(torch, device, amp):
        wind_norm = wind_bundle["model"](*prepared["wind_inputs"])
        motion_out = motion_bundle["model"](*prepared["motion_inputs"])
        correction = motion_out[2] if isinstance(motion_out, (tuple, list)) else motion_out
    return wind_norm, correction


def full_inference_once(
    torch, bmod, x_one, truewind_ridge, joint_ridge, wind_bundle, motion_bundle, device, amp: bool
):
    prepared = prepare_sample(torch, bmod, x_one, truewind_ridge, joint_ridge, wind_bundle, device)
    wind_norm, correction = forward_preloaded(
        torch, wind_bundle, motion_bundle, prepared, device, amp
    )
    raw_residual = (
        wind_norm.detach().float().cpu().numpy()
        * wind_bundle["scaler"]["residual_std"][None, :, :]
    ).astype(np.float32)
    wind = bmod.apply_alpha(prepared["tw_ridge"], raw_residual, wind_bundle["alpha"])
    correction_np = correction.detach().float().cpu().numpy().astype(np.float32)
    motion_z = prepared["joint_z"][:, :, 2:6] + correction_np
    motion = motion_z_to_raw(joint_ridge, motion_z)
    apparent = (wind - motion[:, :, 0:2]).astype(np.float32)
    return {"truewind": wind, "motion": motion, "apparent": apparent}


def summary_stats(values: List[float], prefix: str) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean_ms": float(np.mean(arr)),
        f"{prefix}_median_ms": float(np.median(arr)),
        f"{prefix}_p95_ms": float(np.percentile(arr, 95)),
    }


def benchmark_neural(
    torch, wind_bundle, motion_bundle, prepared, device, warmup: int, repeats: int, amp: bool
):
    for _ in range(max(0, warmup)):
        forward_preloaded(torch, wind_bundle, motion_bundle, prepared, device, amp)
    synchronize(torch, device)
    values = []
    for _ in range(repeats):
        synchronize(torch, device)
        t0 = time.perf_counter()
        forward_preloaded(torch, wind_bundle, motion_bundle, prepared, device, amp)
        synchronize(torch, device)
        values.append((time.perf_counter() - t0) * 1000.0)
    return summary_stats(values, "neural_forward")


def benchmark_end_to_end(
    torch, bmod, x_one, truewind_ridge, joint_ridge, wind_bundle, motion_bundle,
    device, warmup: int, repeats: int, amp: bool,
):
    for _ in range(max(0, warmup)):
        full_inference_once(
            torch, bmod, x_one, truewind_ridge, joint_ridge, wind_bundle, motion_bundle, device, amp
        )
    synchronize(torch, device)
    values = []
    for _ in range(repeats):
        synchronize(torch, device)
        t0 = time.perf_counter()
        full_inference_once(
            torch, bmod, x_one, truewind_ridge, joint_ridge, wind_bundle, motion_bundle, device, amp
        )
        synchronize(torch, device)
        values.append((time.perf_counter() - t0) * 1000.0)
    return summary_stats(values, "end_to_end")


def benchmark_ridge(
    x_one, truewind_ridge, joint_ridge, warmup: int, repeats: int
):
    def run():
        _, tw = ridge_predict(truewind_ridge, x_one)
        _, joint = ridge_predict(joint_ridge, x_one)
        _ = tw - joint[:, :, 2:4]
    for _ in range(max(0, warmup)):
        run()
    values = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run()
        values.append((time.perf_counter() - t0) * 1000.0)
    return summary_stats(values, "ridge_and_reconstruction")


def hardware_metadata(torch, device, args, source_paths: Dict[str, Path]) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "device": str(device),
        "amp": bool(args.amp),
        "batch_size": 1,
        "split": args.split,
        "sample_index": int(args.sample_index),
        "source_paths": {k: str(v.resolve()) for k, v in source_paths.items()},
    }
    try:
        data["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        data["cudnn_version"] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        data.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": f"{props.major}.{props.minor}",
            "gpu_total_memory_bytes": int(props.total_memory),
            "gpu_total_memory_gib": float(props.total_memory / 2**30),
        })
    else:
        data.update({"gpu_name": None, "gpu_capability": None,
                     "gpu_total_memory_bytes": None, "gpu_total_memory_gib": None})
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the complete 1-min 15E-VH model.")
    parser.add_argument("--src-dir", type=Path, default=DEFAULT_SRC_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--stage15b-dir", type=Path, default=DEFAULT_STAGE15B_DIR)
    parser.add_argument("--motion-dir", type=Path, default=DEFAULT_MOTION_DIR)
    parser.add_argument("--stage15b-checkpoint", type=Path, default=None)
    parser.add_argument("--motion-checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=500043)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--joint-ridge-alpha", type=float, default=None,
                        help="Override checkpoint ridge_alpha; default reads it from the motion checkpoint.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA FP16 autocast; default is FP32.")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "forecasting" / "hardware_benchmark_full_primary")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("--warmup must be >= 0 and --repeats must be >= 1")

    torch, nn = import_torch()
    device = select_device(torch, args.device)
    bmod = import_script(args.src_dir / "15B_train_1min_TrueWind_Residual_Correction_v2.py", "stage15b_benchmark")
    vmod = import_script(args.src_dir / "15E_VH_joint_VesselHDG_GatedResidual_5seed_validation.py", "stage15evh_benchmark")

    train = load_split(args.dataset_dir, "train")
    evaluated = load_split(args.dataset_dir, args.split)
    if not (0 <= args.sample_index < len(evaluated["X"])):
        raise IndexError(f"--sample-index {args.sample_index} is outside {args.split} (N={len(evaluated['X'])})")
    x_one = evaluated["X"][args.sample_index:args.sample_index + 1]

    seeds = SEEDS if args.all_seeds else [args.seed]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    metadata = hardware_metadata(torch, device, args, {
        "source_15B": args.src_dir / "15B_train_1min_TrueWind_Residual_Correction_v2.py",
        "source_15E_VH": args.src_dir / "15E_VH_joint_VesselHDG_GatedResidual_5seed_validation.py",
        "dataset_train": train["path"],
        "dataset_evaluated": evaluated["path"],
    })
    metadata.update({"warmup": int(args.warmup), "repeats": int(args.repeats)})

    # The dedicated TrueWind Ridge is shared by all seeds.
    log("[PREPARE] Rebuilding dedicated TrueWind Ridge (alpha=0).")
    truewind_ridge, _ = fit_anchors(train, joint_alpha=0.0)

    for seed in seeds:
        wind_path = resolve_checkpoint(args.stage15b_checkpoint, args.stage15b_dir, seed, "wind")
        motion_path = resolve_checkpoint(args.motion_checkpoint, args.motion_dir, seed, "motion")
        wind_bundle = load_wind_bundle(torch, nn, bmod, wind_path, device)
        motion_bundle = load_motion_bundle(torch, nn, vmod, motion_path, device)
        checkpoint_alpha = motion_bundle["ridge_alpha"]
        joint_alpha = checkpoint_alpha if args.joint_ridge_alpha is None else float(args.joint_ridge_alpha)
        if not np.isclose(joint_alpha, checkpoint_alpha, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(
                f"{motion_path}: checkpoint ridge_alpha={checkpoint_alpha:g} "
                f"does not match --joint-ridge-alpha={joint_alpha:g}"
            )
        joint_ridge = _fit_ridge_from_standardized_targets(
            train["X"], train["y"], channels=list(range(11)), alpha=joint_alpha
        )
        prepared = prepare_sample(torch, bmod, x_one, truewind_ridge, joint_ridge, wind_bundle, device)

        log("")
        log(f"[SEED {seed}] {motion_path.name}")
        log(f"  device={device} | amp={args.amp} | alpha={joint_alpha:g}")
        log(f"  neural parameters={TOTAL_NEURAL_PARAMS:,} | frozen Ridge={FROZEN_RIDGE_PARAMS:,}")

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        neural = benchmark_neural(torch, wind_bundle, motion_bundle, prepared, device, args.warmup, args.repeats, args.amp)
        ridge = benchmark_ridge(x_one, truewind_ridge, joint_ridge, args.warmup, args.repeats)
        end_to_end = benchmark_end_to_end(
            torch, bmod, x_one, truewind_ridge, joint_ridge, wind_bundle, motion_bundle,
            device, args.warmup, args.repeats, args.amp,
        )
        peak_alloc = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        output = full_inference_once(
            torch, bmod, x_one, truewind_ridge, joint_ridge, wind_bundle, motion_bundle, device, args.amp
        )
        if output["truewind"].shape != (1, 5, 2) or output["motion"].shape != (1, 5, 4):
            raise RuntimeError("Full-model output shape audit failed")

        row = {
            "model": f"Physics-Compact full (seed {seed})",
            "comparison_role": "same-current-hardware measurement",
            "seed": int(seed),
            "input_shape": "(1,60,11)",
            "horizons_min": "1,2,3,5,10",
            "trainable_neural_parameters": TOTAL_NEURAL_PARAMS,
            "truewind_neural_parameters": STAGE15B_PARAMS,
            "motion_neural_parameters": MOTION_PARAMS,
            "frozen_ridge_parameters": FROZEN_RIDGE_PARAMS,
            "total_fitted_coefficients": TOTAL_FITTED_COEFFICIENTS,
            "stage15b_checkpoint_bytes": checkpoint_size(wind_path),
            "motion_checkpoint_bytes": checkpoint_size(motion_path),
            "neural_forward_mean_ms": neural["neural_forward_mean_ms"],
            "neural_forward_median_ms": neural["neural_forward_median_ms"],
            "neural_forward_p95_ms": neural["neural_forward_p95_ms"],
            "ridge_and_reconstruction_mean_ms": ridge["ridge_and_reconstruction_mean_ms"],
            "ridge_and_reconstruction_median_ms": ridge["ridge_and_reconstruction_median_ms"],
            "ridge_and_reconstruction_p95_ms": ridge["ridge_and_reconstruction_p95_ms"],
            "end_to_end_mean_ms": end_to_end["end_to_end_mean_ms"],
            "end_to_end_median_ms": end_to_end["end_to_end_median_ms"],
            "end_to_end_p95_ms": end_to_end["end_to_end_p95_ms"],
            "gpu_peak_allocated_bytes": peak_alloc,
            "gpu_peak_reserved_bytes": peak_reserved,
            "precision": "cuda-fp16-autocast" if args.amp and device.type == "cuda" else "fp32",
            "timing_status": "measured locally",
            "timing_definition": "standardized raw X -> two Ridge anchors -> both NNs -> D2H -> apparent wind reconstruction",
        }
        rows.append(row)

        metadata.setdefault("checkpoints", {})[str(seed)] = {
            "stage15b": str(wind_path.resolve()),
            "motion": str(motion_path.resolve()),
            "joint_ridge_alpha": joint_alpha,
        }

    # Literature context only; do not fill timing values with the published hardware result.
    rows.append({
        "model": "BP-STGNN (published)",
        "comparison_role": "literature context; not executed locally",
        "seed": None,
        "input_shape": None,
        "horizons_min": None,
        "trainable_neural_parameters": BP_STGNN_PARAMS,
        "truewind_neural_parameters": None,
        "motion_neural_parameters": None,
        "frozen_ridge_parameters": None,
        "total_fitted_coefficients": None,
        "stage15b_checkpoint_bytes": None,
        "motion_checkpoint_bytes": None,
        "neural_forward_mean_ms": None,
        "neural_forward_median_ms": None,
        "neural_forward_p95_ms": None,
        "ridge_and_reconstruction_mean_ms": None,
        "ridge_and_reconstruction_median_ms": None,
        "ridge_and_reconstruction_p95_ms": None,
        "end_to_end_mean_ms": None,
        "end_to_end_median_ms": None,
        "end_to_end_p95_ms": None,
        "gpu_peak_allocated_bytes": None,
        "gpu_peak_reserved_bytes": None,
        "precision": None,
        "timing_status": "not measured on current hardware",
        "timing_definition": None,
    })

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "full_primary_hardware_efficiency.csv"
    tex_path = args.output_dir / "full_primary_hardware_efficiency.tex"
    report_path = args.output_dir / "full_primary_hardware_report.txt"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_latex(tex_path, index=False, escape=True, na_rep="--", float_format=lambda x: f"{x:.6f}")
    metadata.update({
        "model_definition": {
            "stage15b_neural_parameters": STAGE15B_PARAMS,
            "motion_neural_parameters": MOTION_PARAMS,
            "total_neural_parameters": TOTAL_NEURAL_PARAMS,
            "frozen_ridge_parameters": FROZEN_RIDGE_PARAMS,
            "total_fitted_coefficients": TOTAL_FITTED_COEFFICIENTS,
        },
        "outputs": {"csv": str(csv_path), "tex": str(tex_path), "report": str(report_path)},
    })
    save_json(args.output_dir / "hardware_metadata.json", metadata)

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("FULL PRIMARY MODEL — LOCAL HARDWARE BENCHMARK\n")
        handle.write("=" * 100 + "\n\n")
        handle.write("Complete 1-min 15E-VH path: Stage15B TrueWind + joint vessel/HDG + Ridge anchors + apparent-wind reconstruction.\n")
        handle.write("Timing is batch=1 and per frozen seed; BP-STGNN is context only and was not executed locally.\n\n")
        handle.write(df.to_string(index=False))
        handle.write("\n\nParameter accounting\n" + "-" * 100 + "\n")
        handle.write(f"Stage15B neural: {STAGE15B_PARAMS:,}\n")
        handle.write(f"Joint vessel/HDG neural: {MOTION_PARAMS:,}\n")
        handle.write(f"Total neural: {TOTAL_NEURAL_PARAMS:,}\n")
        handle.write(f"Frozen Ridge coefficients: {FROZEN_RIDGE_PARAMS:,}\n")
        handle.write(f"Total fitted coefficients: {TOTAL_FITTED_COEFFICIENTS:,}\n")

    log("")
    log("RESULT")
    log("=" * 100)
    show_cols = ["model", "trainable_neural_parameters", "end_to_end_median_ms", "end_to_end_p95_ms", "timing_status"]
    log(df[show_cols].to_string(index=False))
    log("")
    log(f"[SAVED] {csv_path}")
    log(f"[SAVED] {tex_path}")
    log(f"[SAVED] {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("\n[FATAL ERROR]", flush=True)
        traceback.print_exc()
        raise SystemExit(1)
