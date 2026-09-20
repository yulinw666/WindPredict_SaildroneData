# -*- coding: utf-8 -*-
"""
benchmark_current_hardware_14A.py

Local, same-machine efficiency benchmark for the 10-min Physics-Compact
Ridge-anchored residual model.

Purpose
-------
Nie et al.'s BP-STGNN timing was obtained on hardware that is not available
here.  This script therefore does *not* manufacture a hardware-to-hardware
comparison.  It measures the local Physics-Compact implementation on the
current machine and writes the published BP-STGNN parameter count as a
literature-context row with no local timing value.

Measured locally
----------------
* trainable neural parameters;
* frozen Ridge coefficient count;
* checkpoint size;
* batch-1 residual-network forward latency;
* batch-1 Ridge-only latency;
* batch-1 end-to-end latency, including Ridge prediction, feature
  construction, CPU-to-device transfer, residual forward pass, and
  synchronization;
* peak CUDA memory, when CUDA is available;
* optional short training-step timing on the current machine.

The script reuses the data/model functions in the supplied 14A training
script.  It refits the final development Ridge anchor from the three
development missions, loads the frozen residual checkpoint, and evaluates the
held-out Tropical Atlantic data only for preparing a representative inference
sample.  No model selection is performed and no test result is changed.

Example (PowerShell)
--------------------
python .\\benchmark_current_hardware_14A.py `
  --model-script .\\14A_train_RidgeAnchored_ComponentSafeResidual_FinalTropical.py `
  --dataset-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\12G_Nie_point_sampled_benchmark_v0_1\\dataset" `
  --checkpoint "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\14A_RidgeAnchored_ComponentSafeResidual_v0_1\\final_model\\14A_final.pt" `
  --output-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\14A_RidgeAnchored_ComponentSafeResidual_v0_1\\hardware_benchmark" `
  --warmup 100 `
  --repeats 300

Optional short local training timing (not a full-training claim):
  add --train-steps 100

Outputs
-------
* hardware_metadata.json
* current_hardware_efficiency.csv
* current_hardware_efficiency.tex
* current_hardware_report.txt

Important interpretation rule
------------------------------
The BP-STGNN row has ``local_forward_ms = NaN``.  Do not copy Nie et al.'s
published latency into that column and do not claim that Physics-Compact is
faster than BP-STGNN unless BP-STGNN is also executed on this same machine
with the same input, precision, batch size, and timing boundary.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0-local-hardware-benchmark-14A"
BP_STGNN_PARAMS = 242_580


def log(message: str = "") -> None:
    print(message, flush=True)


def json_safe(value: Any) -> Any:
    """Convert common NumPy/PyTorch/path values into JSON-safe objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            json_safe(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def import_script(path: Path):
    """Import a user script whose filename may contain spaces/parentheses."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model script not found: {path}")

    module_name = f"benchmark_model_script_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model script: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def import_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyTorch is required. Activate the WindPredict environment first."
        ) from exc
    return torch, nn


def select_device(torch, requested: str):
    requested = requested.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
        return torch.device("cuda:0")
    if requested != "auto":
        raise ValueError(f"Unknown device: {requested}")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hardware_metadata(torch, device, checkpoint_path: Path) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "torch_version": getattr(torch, "__version__", None),
        "torch_device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": None,
        "checkpoint": str(checkpoint_path.resolve()),
    }

    try:
        metadata["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        pass

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        metadata.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_capability": f"{props.major}.{props.minor}",
                "gpu_total_memory_bytes": int(props.total_memory),
                "gpu_total_memory_gib": float(props.total_memory / 2**30),
            }
        )
    else:
        metadata.update(
            {
                "gpu_name": None,
                "gpu_capability": None,
                "gpu_total_memory_bytes": None,
                "gpu_total_memory_gib": None,
            }
        )

    try:
        import psutil

        vm = psutil.virtual_memory()
        metadata.update(
            {
                "ram_total_bytes": int(vm.total),
                "ram_total_gib": float(vm.total / 2**30),
            }
        )
    except Exception:
        metadata["ram_total_bytes"] = None
        metadata["ram_total_gib"] = None

    return metadata


def synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(torch, device, enabled: bool):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def count_trainable_parameters(model) -> int:
    return int(
        sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
    )


def checkpoint_file_size(path: Path) -> int:
    return int(path.stat().st_size) if path.exists() else 0


def make_dev_bundle(module, dataset_root: Path):
    dev_missions = [
        module.load_mission(dataset_root, mission)
        for mission in module.DEV_MISSIONS
    ]
    dev_all = module.concat_missions(dev_missions)
    final_mission = getattr(module, "FINAL_MISSION", "Tropical Atlantic")
    test_data = module.load_mission(dataset_root, final_mission)
    return dev_all, test_data


def prepare_14a(module, torch, nn, dataset_root: Path, checkpoint_path: Path, device):
    """Reconstruct the frozen 14A inference path without retraining the NN."""
    dev_all, test_data = make_dev_bundle(module, dataset_root)

    x_dev_base = module.build_base4(dev_all["X_full9"])
    ridge_model, ridge_scaler = module.fit_ridge(x_dev_base, dev_all["y_uv"])

    x_test_base = module.build_base4(test_data["X_full9"])
    ridge_test = module.ridge_predict(ridge_model, ridge_scaler, x_test_base)

    residual_features = module.build_residual_features(
        test_data["X_full9"],
        ridge_test,
    )

    # The checkpoint stores NumPy scalers in addition to tensors.  PyTorch
    # 2.6+ defaults to weights_only=True, which rejects those trusted local
    # objects; older PyTorch versions do not accept the keyword, hence the
    # compatibility fallback.
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "residual_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain residual_state_dict. "
            "Use the final 14A checkpoint (14A_final.pt)."
        )
    if "residual_scalers" not in checkpoint:
        raise KeyError("Checkpoint does not contain residual_scalers.")

    transformed = module.transform_residual_features(
        residual_features,
        checkpoint["residual_scalers"],
    )

    model_class = module.build_model_class(torch, nn)
    residual_model = model_class()
    residual_model.load_state_dict(checkpoint["residual_state_dict"], strict=True)
    residual_model.to(device)
    residual_model.eval()

    active_components = np.asarray(
        checkpoint.get("active_components", [True, True]),
        dtype=bool,
    ).reshape(-1)
    frozen_alphas = np.asarray(
        checkpoint.get("frozen_alphas", [1.0, 1.0]),
        dtype=np.float32,
    ).reshape(-1)

    if active_components.size != 2 or frozen_alphas.size != 2:
        raise RuntimeError("Unexpected active_components/frozen_alphas shape in checkpoint.")

    return {
        "dev_all": dev_all,
        "test_data": test_data,
        "ridge_model": ridge_model,
        "ridge_scaler": ridge_scaler,
        "ridge_test": ridge_test,
        "transformed_test": transformed,
        "residual_model": residual_model,
        "checkpoint": checkpoint,
        "active_components": active_components,
        "frozen_alphas": frozen_alphas,
    }


def one_model_forward(
    torch,
    model,
    transformed: Dict[str, np.ndarray],
    index: int,
    device,
    amp: bool,
):
    seq = torch.from_numpy(transformed["seq"][index:index + 1]).to(
        device,
        non_blocking=True,
    )
    summary = torch.from_numpy(transformed["summary"][index:index + 1]).to(
        device,
        non_blocking=True,
    )
    anchor = torch.from_numpy(transformed["anchor"][index:index + 1]).to(
        device,
        non_blocking=True,
    )
    with torch.inference_mode(), autocast_context(torch, device, amp):
        output = model(seq, summary, anchor)
    return output


def preloaded_model_inputs(torch, transformed, index: int, device):
    """Move one prepared sample to the selected device once."""
    return (
        torch.from_numpy(transformed["seq"][index:index + 1]).to(
            device,
            non_blocking=True,
        ),
        torch.from_numpy(transformed["summary"][index:index + 1]).to(
            device,
            non_blocking=True,
        ),
        torch.from_numpy(transformed["anchor"][index:index + 1]).to(
            device,
            non_blocking=True,
        ),
    )


def model_forward_preloaded(torch, model, inputs, device, amp: bool):
    with torch.inference_mode(), autocast_context(torch, device, amp):
        return model(*inputs)


def benchmark_model_forward(
    torch,
    model,
    transformed,
    device,
    warmup: int,
    repeats: int,
    amp: bool,
) -> Dict[str, float]:
    model.eval()
    n = len(transformed["seq"])
    if n < 1:
        raise ValueError("No test samples available for forward benchmark.")

    # Keep the input tensors on the target device for this measurement.  This
    # isolates the residual-network forward pass; the end-to-end benchmark
    # below includes feature construction and host/device transfer.
    inputs = preloaded_model_inputs(torch, transformed, 0, device)

    for step in range(max(0, warmup)):
        model_forward_preloaded(torch, model, inputs, device, amp)
    synchronize(torch, device)

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for step in range(repeats):
            model_forward_preloaded(torch, model, inputs, device, amp)
        end_event.record()
        torch.cuda.synchronize(device)
        total_ms = float(start_event.elapsed_time(end_event))
        return {
            "residual_forward_mean_ms": total_ms / max(repeats, 1),
            "residual_forward_median_ms": total_ms / max(repeats, 1),
            "residual_forward_p95_ms": total_ms / max(repeats, 1),
            "timing_note": "CUDA event; GPU kernel time, batch=1",
        }

    values_ms: List[float] = []
    for step in range(repeats):
        t0 = time.perf_counter()
        model_forward_preloaded(torch, model, inputs, device, amp)
        values_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "residual_forward_mean_ms": float(np.mean(values_ms)),
        "residual_forward_median_ms": float(np.median(values_ms)),
        "residual_forward_p95_ms": float(np.percentile(values_ms, 95)),
        "timing_note": "perf_counter; batch=1",
    }


def benchmark_ridge_only(module, bundle, repeats: int, warmup: int) -> Dict[str, float]:
    x_test = module.build_base4(bundle["test_data"]["X_full9"])
    ridge_model = bundle["ridge_model"]
    ridge_scaler = bundle["ridge_scaler"]
    n = len(x_test)

    for step in range(max(0, warmup)):
        module.ridge_predict(ridge_model, ridge_scaler, x_test[step % n:step % n + 1])

    values_ms: List[float] = []
    for step in range(repeats):
        index = step % n
        t0 = time.perf_counter()
        module.ridge_predict(ridge_model, ridge_scaler, x_test[index:index + 1])
        values_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "ridge_only_mean_ms": float(np.mean(values_ms)),
        "ridge_only_median_ms": float(np.median(values_ms)),
        "ridge_only_p95_ms": float(np.percentile(values_ms, 95)),
    }


def benchmark_end_to_end(
    module,
    torch,
    bundle,
    device,
    warmup: int,
    repeats: int,
    amp: bool,
) -> Dict[str, float]:
    """Time one complete batch-1 forecast from raw six-step input."""
    raw_x = bundle["test_data"]["X_full9"]
    ridge_model = bundle["ridge_model"]
    ridge_scaler = bundle["ridge_scaler"]
    residual_model = bundle["residual_model"]
    n = len(raw_x)

    def run_once(index: int):
        x_one = raw_x[index:index + 1]
        x_base = module.build_base4(x_one)
        ridge_one = module.ridge_predict(ridge_model, ridge_scaler, x_base)
        features_one = module.build_residual_features(x_one, ridge_one)
        transformed_one = module.transform_residual_features(
            features_one,
            bundle["checkpoint"]["residual_scalers"],
        )
        _ = one_model_forward(
            torch,
            residual_model,
            transformed_one,
            0,
            device,
            amp,
        )

    for step in range(max(0, warmup)):
        run_once(step % n)
    synchronize(torch, device)

    values_ms: List[float] = []
    for step in range(repeats):
        synchronize(torch, device)
        t0 = time.perf_counter()
        run_once(step % n)
        synchronize(torch, device)
        values_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "end_to_end_mean_ms": float(np.mean(values_ms)),
        "end_to_end_median_ms": float(np.median(values_ms)),
        "end_to_end_p95_ms": float(np.percentile(values_ms, 95)),
        "end_to_end_definition": (
            "raw six-step sample -> Base4 Ridge -> residual features -> "
            "host/device transfer -> residual forward -> synchronization"
        ),
    }


def build_training_batch(module, bundle):
    """Prepare OOF residual targets for optional short timing only."""
    dev_all = bundle["dev_all"]
    x_base = module.build_base4(dev_all["X_full9"])
    ridge_oof = module.ridge_oof_predict(
        x_base,
        dev_all["y_uv"],
        dev_all["mission_labels"],
    )
    residual_target = (dev_all["y_uv"] - ridge_oof).astype(np.float32)
    features = module.build_residual_features(dev_all["X_full9"], ridge_oof)
    scalers = module.fit_residual_scalers(features, residual_target)
    transformed = module.transform_residual_features(features, scalers)
    target_norm = module.normalized_residual_target(
        residual_target,
        scalers["residual_std"],
    )
    return transformed, target_norm


def benchmark_training_steps(
    module,
    torch,
    nn,
    bundle,
    device,
    steps: int,
    batch_size: int,
    amp: bool,
) -> Dict[str, float]:
    if steps <= 0:
        return {
            "training_steps": 0,
            "training_total_s": None,
            "training_step_mean_ms": None,
            "training_note": "not requested",
        }

    transformed, target_norm = build_training_batch(module, bundle)
    model_class = module.build_model_class(torch, nn)
    model = model_class().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    n = len(target_norm)
    step_times: List[float] = []
    start_total = time.perf_counter()

    for step in range(steps):
        start = (step * batch_size) % n
        indices = (np.arange(batch_size) + start) % n
        seq = torch.from_numpy(transformed["seq"][indices]).to(device, non_blocking=True)
        summary = torch.from_numpy(transformed["summary"][indices]).to(device, non_blocking=True)
        anchor = torch.from_numpy(transformed["anchor"][indices]).to(device, non_blocking=True)
        target = torch.from_numpy(target_norm[indices]).to(device, non_blocking=True)

        synchronize(torch, device)
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(torch, device, amp):
            prediction = model(seq, summary, anchor)
            loss = 0.5 * (
                (prediction[:, 0] - target[:, 0]).pow(2).mean()
                + (prediction[:, 1] - target[:, 1]).pow(2).mean()
            )
        loss.backward()
        optimizer.step()
        synchronize(torch, device)
        step_times.append((time.perf_counter() - t0) * 1000.0)

    total_s = time.perf_counter() - start_total
    return {
        "training_steps": int(steps),
        "training_total_s": float(total_s),
        "training_step_mean_ms": float(np.mean(step_times)),
        "training_step_median_ms": float(np.median(step_times)),
        "training_step_p95_ms": float(np.percentile(step_times, 95)),
        "training_batch_size": int(batch_size),
        "training_note": "short local timing only; not a full-fit duration",
    }


def make_rows(module, torch, bundle, device, benchmark, ridge_benchmark, train_benchmark, metadata):
    model = bundle["residual_model"]
    neural_params = count_trainable_parameters(model)
    ridge_params = int(module.ridge_parameter_count(bundle["ridge_model"]))
    total_params = neural_params + ridge_params
    ckpt_size = checkpoint_file_size(Path(metadata["checkpoint"]))

    local_row = {
        "model": "Physics-Compact (local)",
        "comparison_role": "same-current-hardware measurement",
        "trainable_neural_parameters": neural_params,
        "frozen_ridge_parameters": ridge_params,
        "total_local_parameters": total_params,
        "checkpoint_bytes": ckpt_size,
        "residual_forward_mean_ms": benchmark["residual_forward_mean_ms"],
        "residual_forward_median_ms": benchmark["residual_forward_median_ms"],
        "residual_forward_p95_ms": benchmark["residual_forward_p95_ms"],
        "ridge_only_mean_ms": ridge_benchmark["ridge_only_mean_ms"],
        "ridge_only_median_ms": ridge_benchmark["ridge_only_median_ms"],
        "ridge_only_p95_ms": ridge_benchmark["ridge_only_p95_ms"],
        "end_to_end_mean_ms": benchmark["end_to_end_mean_ms"],
        "end_to_end_median_ms": benchmark["end_to_end_median_ms"],
        "end_to_end_p95_ms": benchmark["end_to_end_p95_ms"],
        "gpu_peak_allocated_bytes": metadata.get("gpu_peak_allocated_bytes"),
        "gpu_peak_reserved_bytes": metadata.get("gpu_peak_reserved_bytes"),
        "timing_status": "measured locally",
        "timing_note": benchmark["timing_note"],
        **train_benchmark,
    }

    bp_row = {
        "model": "BP-STGNN (published)",
        "comparison_role": "literature context only",
        "trainable_neural_parameters": BP_STGNN_PARAMS,
        "frozen_ridge_parameters": np.nan,
        "total_local_parameters": np.nan,
        "checkpoint_bytes": np.nan,
        "residual_forward_mean_ms": np.nan,
        "residual_forward_median_ms": np.nan,
        "residual_forward_p95_ms": np.nan,
        "ridge_only_mean_ms": np.nan,
        "ridge_only_median_ms": np.nan,
        "ridge_only_p95_ms": np.nan,
        "end_to_end_mean_ms": np.nan,
        "end_to_end_median_ms": np.nan,
        "end_to_end_p95_ms": np.nan,
        "gpu_peak_allocated_bytes": np.nan,
        "gpu_peak_reserved_bytes": np.nan,
        "training_steps": np.nan,
        "training_total_s": np.nan,
        "training_step_mean_ms": np.nan,
        "training_step_median_ms": np.nan,
        "training_step_p95_ms": np.nan,
        "training_batch_size": np.nan,
        "timing_status": "not measured on current hardware",
        "timing_note": "published result uses different hardware/protocol",
    }
    return [local_row, bp_row]


def make_latex_table(df: pd.DataFrame, path: Path) -> None:
    columns = [
        "model",
        "trainable_neural_parameters",
        "frozen_ridge_parameters",
        "end_to_end_median_ms",
        "end_to_end_p95_ms",
        "gpu_peak_allocated_bytes",
        "timing_status",
    ]
    out = df[columns].copy()
    out.columns = [
        "Model",
        "Neural parameters",
        "Ridge parameters",
        "E2E median (ms)",
        "E2E P95 (ms)",
        "Peak GPU bytes",
        "Timing basis",
    ]
    out.to_latex(
        path,
        index=False,
        escape=True,
        na_rep="--",
        float_format=lambda value: f"{value:.3f}",
    )


def write_report(
    path: Path,
    metadata: Dict[str, Any],
    df: pd.DataFrame,
    bundle: Dict[str, Any],
    args,
) -> None:
    local = df[df["model"] == "Physics-Compact (local)"].iloc[0]
    lines = [
        "LOCAL CURRENT-HARDWARE EFFICIENCY BENCHMARK",
        "=" * 100,
        f"script_version: {SCRIPT_VERSION}",
        f"device: {metadata.get('torch_device')}",
        f"gpu: {metadata.get('gpu_name')}",
        f"torch: {metadata.get('torch_version')}",
        f"cuda: {metadata.get('torch_cuda_version')}",
        "",
        "LOCAL PHYSICS-COMPACT",
        "-" * 100,
        f"trainable neural parameters: {int(local['trainable_neural_parameters']):,}",
        f"frozen Ridge parameters: {int(local['frozen_ridge_parameters']):,}",
        f"total local parameters (neural + Ridge): {int(local['total_local_parameters']):,}",
        f"Ridge-only median latency (ms): {local['ridge_only_median_ms']:.6f}",
        f"residual-forward median latency (ms): {local['residual_forward_median_ms']:.6f}",
        f"end-to-end median latency (ms): {local['end_to_end_median_ms']:.6f}",
        f"end-to-end P95 latency (ms): {local['end_to_end_p95_ms']:.6f}",
        f"peak allocated GPU memory (bytes): {local['gpu_peak_allocated_bytes']}",
        "",
        "INTERPRETATION",
        "-" * 100,
        "The BP-STGNN row contains the published parameter count only.",
        "Its published latency is not inserted into the local latency columns.",
        "Therefore this report supports local reproducibility and local model",
        "profiling, but does not support a claim that Physics-Compact is faster",
        "than BP-STGNN across different hardware.",
        "",
        "COMMAND SETTINGS",
        "-" * 100,
        f"warmup={args.warmup}, repeats={args.repeats}, amp={args.amp}, train_steps={args.train_steps}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure Physics-Compact efficiency on the current hardware only."
    )
    parser.add_argument("--model-script", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--amp", action="store_true", help="Use CUDA FP16 autocast for the NN forward.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("--warmup must be >= 0 and --repeats must be >= 1.")
    if args.train_steps < 0 or args.train_batch_size < 1:
        raise ValueError("Training timing arguments must be non-negative/positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    module = import_script(args.model_script)
    torch, nn = import_torch()
    device = select_device(torch, args.device)

    if device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    metadata = hardware_metadata(torch, device, args.checkpoint)
    log("=" * 100)
    log("LOCAL CURRENT-HARDWARE BENCHMARK")
    log("=" * 100)
    log(f"device : {device}")
    log(f"gpu    : {metadata.get('gpu_name')}")
    log(f"torch  : {metadata.get('torch_version')}")
    log(f"cuda   : {metadata.get('torch_cuda_version')}")

    dataset_root = module.resolve_dataset_root(args.dataset_dir)
    bundle = prepare_14a(
        module,
        torch,
        nn,
        dataset_root,
        args.checkpoint,
        device,
    )

    model = bundle["residual_model"]
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    benchmark = benchmark_model_forward(
        torch,
        model,
        bundle["transformed_test"],
        device,
        args.warmup,
        args.repeats,
        args.amp,
    )
    ridge_benchmark = benchmark_ridge_only(
        module,
        bundle,
        args.repeats,
        args.warmup,
    )
    e2e_benchmark = benchmark_end_to_end(
        module,
        torch,
        bundle,
        device,
        args.warmup,
        args.repeats,
        args.amp,
    )
    benchmark.update(e2e_benchmark)

    if device.type == "cuda":
        synchronize(torch, device)
        metadata["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        metadata["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    else:
        metadata["gpu_peak_allocated_bytes"] = None
        metadata["gpu_peak_reserved_bytes"] = None

    train_benchmark = benchmark_training_steps(
        module,
        torch,
        nn,
        bundle,
        device,
        args.train_steps,
        args.train_batch_size,
        args.amp,
    )

    rows = make_rows(
        module,
        torch,
        bundle,
        device,
        benchmark,
        ridge_benchmark,
        train_benchmark,
        metadata,
    )
    result_df = pd.DataFrame(rows)
    result_df.to_csv(
        args.output_dir / "current_hardware_efficiency.csv",
        index=False,
        encoding="utf-8-sig",
    )
    make_latex_table(
        result_df,
        args.output_dir / "current_hardware_efficiency.tex",
    )

    metadata.update(
        {
            "dataset_root": str(dataset_root),
            "n_tropical_test_samples": int(len(bundle["test_data"]["y_uv"])),
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
            "amp": bool(args.amp),
            "train_steps": int(args.train_steps),
            "train_batch_size": int(args.train_batch_size),
            "benchmark_definition": benchmark["end_to_end_definition"],
        }
    )
    save_json(args.output_dir / "hardware_metadata.json", metadata)
    write_report(
        args.output_dir / "current_hardware_report.txt",
        metadata,
        result_df,
        bundle,
        args,
    )

    log("")
    log("RESULT")
    log("=" * 100)
    shown = result_df[
        [
            "model",
            "trainable_neural_parameters",
            "end_to_end_median_ms",
            "end_to_end_p95_ms",
            "timing_status",
        ]
    ]
    log(shown.to_string(index=False))
    log("")
    log(f"[SAVED] {args.output_dir / 'current_hardware_efficiency.csv'}")
    log(f"[SAVED] {args.output_dir / 'current_hardware_efficiency.tex'}")
    log(f"[SAVED] {args.output_dir / 'hardware_metadata.json'}")
    log(f"[SAVED] {args.output_dir / 'current_hardware_report.txt'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        log("\n[FATAL ERROR]")
        traceback.print_exc()
        raise
