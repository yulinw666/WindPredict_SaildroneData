# -*- coding: utf-8 -*-
r"""
12K_evaluate_PhysicsCompact_NieData_v4_PointBenchmark.py

Final all-development fitting and Tropical Atlantic re-evaluation for the
Stage-12J selected Physics-Compact-NieData v4 candidate.

This script:
1. reads the selected v4 configuration from Stage 12J;
2. derives fixed training epochs from development-only LOMO best epochs;
3. fits scaler + Ridge on Antarctic + Atlantic + West Coast;
4. trains five v4 seeds for fixed epochs with no Tropical-based early stopping;
5. evaluates Persistence, Ridge and v4 on the Stage-12G point-sampled
   Tropical Atlantic samples;
6. reports the published BP-STGNN values as literature context.

The Tropical Atlantic point-sampled dataset has already been examined in earlier
protocol analyses, so this script labels the result as a re-evaluation.  Model
weights and fixed epochs are nevertheless determined only from development
missions in this script.

Usage
-----
python "D:\project\WindPredict_SaildroneData\src\12K_evaluate_PhysicsCompact_NieData_v4_PointBenchmark.py" ^
  --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" ^
  --j-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12J_PhysicsCompact_NieData_v4_RidgeResidual_LOMO_point_v0_1" ^
  --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12K_PhysicsCompact_NieData_v4_PointBenchmark_v0_1"
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0-PhysicsCompact-NieData-v4-PointBenchmark"

DEFAULT_PROJECT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")
DEVELOPMENT_MISSIONS = ["Antarctic", "Atlantic", "West Coast"]
TEST_MISSION = "Tropical Atlantic"
SCREENING_SEEDS = [500043, 501052, 502061]
FINAL_SEEDS = [500043, 501052, 502061, 503070, 504079]
RIDGE_ALPHA = 1.0

NIE_BP_STGNN_REFERENCE = {
    "wind_U_RMSE_mps": 0.74864,
    "wind_V_RMSE_mps": 0.70506,
    "wind_speed_RMSE_mps": 0.69940,
    "wind_direction_RMSE_deg": 5.58400,
}


def log(msg=""):
    print(msg, flush=True)


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, str(path))

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {path}")

    mod = importlib.util.module_from_spec(spec)

    # Required for dataclass/type introspection during dynamic import
    sys.modules[name] = mod

    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise

    return mod


def seed_everything(torch, seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def derive_epoch_plan(selected_runs: pd.DataFrame):
    required = selected_runs.loc[
        selected_runs["seed"].isin(SCREENING_SEEDS)
    ].copy()

    expected = len(SCREENING_SEEDS) * len(DEVELOPMENT_MISSIONS)
    if len(required) != expected:
        raise RuntimeError(
            f"Expected {expected} selected LOMO rows, found {len(required)}."
        )

    epochs = required["best_epoch_one_based"].astype(int)
    global_median = int(np.median(epochs.to_numpy()))

    plan = {}
    for seed in SCREENING_SEEDS:
        vals = required.loc[
            required["seed"].astype(int) == int(seed),
            "best_epoch_one_based",
        ].astype(int).to_numpy()
        if len(vals) != len(DEVELOPMENT_MISSIONS):
            raise RuntimeError(
                f"Seed {seed}: expected 3 LOMO epochs, got {len(vals)}."
            )
        plan[int(seed)] = int(np.median(vals))

    for seed in FINAL_SEEDS:
        if seed not in plan:
            plan[int(seed)] = global_median

    return plan, global_median


def summarize_metrics(rows: pd.DataFrame):
    metric_cols = [
        "wind_U_RMSE_mps",
        "wind_V_RMSE_mps",
        "wind_vector_RMSE_mps",
        "wind_speed_RMSE_mps",
        "wind_direction_MAE_deg",
        "wind_direction_RMSE_deg",
        "vessel_E_RMSE_mps",
        "vessel_N_RMSE_mps",
        "vessel_vector_RMSE_mps",
        "AW_E_RMSE_mps",
        "AW_N_RMSE_mps",
        "AW_vector_RMSE_mps",
        "AWS_RMSE_mps",
        "wind_gate_mean",
        "vessel_gate_mean",
        "wind_correction_z_RMS",
        "vessel_correction_z_RMS",
    ]
    out = {"method": "PhysicsCompactNieDataV4", "seed_count": len(rows)}
    for col in metric_cols:
        vals = rows[col].to_numpy(dtype=float)
        out[f"{col}_mean"] = float(np.mean(vals))
        out[f"{col}_std"] = float(np.std(vals, ddof=0))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=DEFAULT_PROJECT_ROOT
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--j-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root
    dataset_dir = args.dataset_dir
    j_dir = args.j_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_path = project_root / "src" / "12C_train_PhysicsCompact_NieData_LOMO.py"
    d_path = project_root / "src" / "12D_final_PhysicsCompact_NieData_v2_TropicalAtlantic.py"
    j_script_path = project_root / "src" / "12J_train_PhysicsCompact_NieData_v4_RidgeResidual_LOMO.py"

    base = load_module(base_path, "stage12c_base_for_12k")
    dbase = load_module(d_path, "stage12d_base_for_12k")
    jmod = load_module(j_script_path, "stage12j_model_for_12k")

    torch, nn, _ = base.import_torch()
    base.configure_cuda(torch)

    if args.force_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    selected_path = j_dir / "selected_config.json"
    selected_runs_path = j_dir / "selected_candidate_lomo_runs.csv"

    if not selected_path.exists() or not selected_runs_path.exists():
        raise FileNotFoundError(
            "Stage 12J selected_config.json / selected_candidate_lomo_runs.csv "
            "not found. Complete full 12J first."
        )

    with selected_path.open("r", encoding="utf-8") as f:
        selected = json.load(f)

    cfg = selected["selected_config"]
    selected_name = selected["selected_candidate"]
    selected_runs = pd.read_csv(selected_runs_path)
    epoch_plan, global_median = derive_epoch_plan(selected_runs)

    dev_paths = base.explicit_development_paths(dataset_dir)
    dev_missions = {
        m: base.load_development_mission(dev_paths[m], m)
        for m in DEVELOPMENT_MISSIONS
    }
    train = base.concat_missions(
        [dev_missions[m] for m in DEVELOPMENT_MISSIONS]
    )

    scaler = base.fit_scalers(train["X"], train["y"], train["aw"])
    ridge = base.fit_ridge(train["X"], train["y"], scaler)
    ridge_train_z = base.ridge_predict_z(ridge, train["X"], scaler)

    ModelClass = jmod.build_model_class(
        torch, nn, len(base.FEATURE_NAMES)
    )
    amp_enabled = bool(jmod.FREEZE.use_amp and device.type == "cuda")
    const = jmod.make_constants(torch, scaler, train["y"], device)

    ckpt_dir = output_dir / "final_checkpoints"
    hist_dir = output_dir / "final_histories"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 124)
    log("12K — Physics-Compact-NieData v4 POINT BENCHMARK")
    log("=" * 124)
    log(f"candidate      : {selected_name}")
    log(f"config         : {cfg}")
    log(f"parameter ref  : {selected.get('parameter_count')}")
    log(f"epoch plan     : {epoch_plan}")
    log(f"device         : {device}")
    log("")

    # Final all-development training. No test is loaded before all checkpoints exist.
    final_ckpts = {}

    Xz = base.standardize_X(train["X"], scaler)
    yz = base.standardize_y(train["y"], scaler)

    for seed in FINAL_SEEDS:
        epochs = int(epoch_plan[int(seed)])
        seed_everything(torch, seed)
        model = ModelClass().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=jmod.FREEZE.learning_rate
        )
        grad_scaler = base.create_grad_scaler(torch, amp_enabled)

        history = []
        start = time.perf_counter()

        log(f"[TRAIN] seed={seed} | fixed_epochs={epochs}")

        for epoch in range(1, epochs + 1):
            model.train()
            sums = {}
            count = 0

            for xb_np, yb_np, awb_np, rb_np in base.sequential_batches(
                Xz,
                yz,
                train["aw"],
                ridge_train_z,
                batch_size=jmod.FREEZE.batch_size,
            ):
                xb = torch.from_numpy(xb_np).to(device, non_blocking=True)
                yb = torch.from_numpy(yb_np).to(device, non_blocking=True)
                awb = torch.from_numpy(awb_np).to(device, non_blocking=True)
                rb = torch.from_numpy(rb_np).to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with base.autocast_context(torch, amp_enabled):
                    out = model(xb, rb)
                    losses = jmod.compute_loss(
                        torch, out, yb, awb, const, cfg
                    )

                total = losses["total"]
                if not torch.isfinite(total):
                    raise FloatingPointError("Non-finite 12K training loss.")

                if grad_scaler is not None:
                    grad_scaler.scale(total).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), jmod.FREEZE.grad_clip_norm
                    )
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), jmod.FREEZE.grad_clip_norm
                    )
                    optimizer.step()

                bn = len(xb_np)
                for key, value in losses.items():
                    sums[key] = sums.get(key, 0.0) + float(
                        value.detach().float().cpu().item()
                    ) * bn
                count += bn

            history.append({
                "seed": int(seed),
                "epoch_one_based": int(epoch),
                **{
                    f"train_{k}": v / max(count, 1)
                    for k, v in sums.items()
                },
            })

        elapsed = time.perf_counter() - start
        ckpt_path = ckpt_dir / f"PhysicsCompactNieDataV4__seed_{seed}.pt"
        torch.save({
            "stage": "12K",
            "script_version": SCRIPT_VERSION,
            "model_name": "Physics-Compact-NieData v4",
            "candidate": selected_name,
            "candidate_config": cfg,
            "seed": int(seed),
            "fixed_train_epochs": int(epochs),
            "parameter_count": jmod.count_parameters(model),
            "state_dict": {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            },
            "scaler": scaler,
            "ridge_alpha": RIDGE_ALPHA,
            "Tropical_Atlantic_used_for_training": False,
            "Tropical_Atlantic_used_for_epoch_selection": False,
        }, ckpt_path)

        pd.DataFrame(history).to_csv(
            hist_dir / f"seed_{seed}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        final_ckpts[int(seed)] = ckpt_path

        log(
            f"  saved seed={seed}, epochs={epochs}, "
            f"params={jmod.count_parameters(model):,}, "
            f"time={elapsed:.1f}s"
        )

        del model, optimizer, grad_scaler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Only now open Tropical Atlantic.
    test_path = (
        dataset_dir
        / "track_J_joint_compatible"
        / "Tropical_Atlantic_TEST.npz"
    )
    test = dbase.load_track_j(test_path, TEST_MISSION)

    ridge_test_z = base.ridge_predict_z(ridge, test["X"], scaler)
    ridge_test_raw = base.inverse_y(ridge_test_z, scaler)
    ridge_metrics = base.evaluate_raw(
        test["y"], ridge_test_raw, test["aw"]
    )

    p_y, _ = base.persistence_predict(test["X"])
    persistence_metrics = base.evaluate_raw(
        test["y"], p_y, test["aw"]
    )

    seed_rows = []

    for seed in FINAL_SEEDS:
        ckpt = torch.load(
            final_ckpts[int(seed)],
            map_location="cpu",
            weights_only=False,
        )
        model = ModelClass().to(device)
        model.load_state_dict(ckpt["state_dict"], strict=True)

        metrics, pred_raw = jmod.evaluate_model(
            base=base,
            torch=torch,
            model=model,
            X=test["X"],
            y=test["y"],
            aw=test["aw"],
            ridge_pred_z=ridge_test_z,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
        )

        row = {
            "method": "PhysicsCompactNieDataV4",
            "seed": int(seed),
            "fixed_train_epochs": int(epoch_plan[int(seed)]),
            **metrics,
        }
        seed_rows.append(row)

        np.savez_compressed(
            output_dir / f"prediction_seed_{seed}.npz",
            pred_raw=np.asarray(pred_raw, dtype=np.float32),
            true_raw=np.asarray(test["y"], dtype=np.float32),
            apparent_ref=np.asarray(test["aw"], dtype=np.float32),
        )

        log(
            f"[TEST seed={seed}] "
            f"U={metrics['wind_U_RMSE_mps']:.4f} | "
            f"V={metrics['wind_V_RMSE_mps']:.4f} | "
            f"WS={metrics['wind_speed_RMSE_mps']:.4f} | "
            f"WD={metrics['wind_direction_RMSE_deg']:.2f} | "
            f"Vship={metrics['vessel_vector_RMSE_mps']:.4f} | "
            f"AW={metrics['AW_vector_RMSE_mps']:.4f}"
        )

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(
        output_dir / "tropical_atlantic_seed_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    proposed = summarize_metrics(seed_df)

    rows = []
    for method, metrics in [
        ("Persistence", persistence_metrics),
        ("Ridge", ridge_metrics),
    ]:
        row = {"method": method, "seed_count": 0}
        for key, value in metrics.items():
            row[f"{key}_mean"] = float(value)
            row[f"{key}_std"] = 0.0
        rows.append(row)

    rows.append(proposed)
    summary = pd.DataFrame(rows)
    summary.to_csv(
        output_dir / "tropical_atlantic_method_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    p = proposed
    comparison = pd.DataFrame([
        {
            "method": "PhysicsCompactNieDataV4",
            "source": "12K point-sampled protocol",
            "U_RMSE_mps": p["wind_U_RMSE_mps_mean"],
            "V_RMSE_mps": p["wind_V_RMSE_mps_mean"],
            "WS_RMSE_mps": p["wind_speed_RMSE_mps_mean"],
            "WD_RMSE_deg": p["wind_direction_RMSE_deg_mean"],
            "Vessel_RMSE_mps": p["vessel_vector_RMSE_mps_mean"],
            "AW_RMSE_mps": p["AW_vector_RMSE_mps_mean"],
            "parameters": int(selected["parameter_count"]),
        },
        {
            "method": "Ridge",
            "source": "12K point-sampled protocol",
            "U_RMSE_mps": ridge_metrics["wind_U_RMSE_mps"],
            "V_RMSE_mps": ridge_metrics["wind_V_RMSE_mps"],
            "WS_RMSE_mps": ridge_metrics["wind_speed_RMSE_mps"],
            "WD_RMSE_deg": ridge_metrics["wind_direction_RMSE_deg"],
            "Vessel_RMSE_mps": ridge_metrics["vessel_vector_RMSE_mps"],
            "AW_RMSE_mps": ridge_metrics["AW_vector_RMSE_mps"],
            "parameters": np.nan,
        },
        {
            "method": "BP-STGNN (published)",
            "source": "Nie et al. published literature value",
            "U_RMSE_mps": NIE_BP_STGNN_REFERENCE["wind_U_RMSE_mps"],
            "V_RMSE_mps": NIE_BP_STGNN_REFERENCE["wind_V_RMSE_mps"],
            "WS_RMSE_mps": NIE_BP_STGNN_REFERENCE["wind_speed_RMSE_mps"],
            "WD_RMSE_deg": NIE_BP_STGNN_REFERENCE["wind_direction_RMSE_deg"],
            "Vessel_RMSE_mps": np.nan,
            "AW_RMSE_mps": np.nan,
            "parameters": 242580,
        },
    ])

    comparison.to_csv(
        output_dir / "12K_final_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (output_dir / "12K_FINAL_REPORT.txt").open(
        "w", encoding="utf-8"
    ) as f:
        f.write("12K Physics-Compact-NieData v4 Point Benchmark\n")
        f.write("=" * 124 + "\n\n")
        f.write(f"Selected candidate: {selected_name}\n")
        f.write(f"Config: {cfg}\n")
        f.write(f"Epoch plan: {epoch_plan}\n")
        f.write(
            "Tropical Atlantic status: re-evaluation; no test samples used "
            "for training or epoch selection in 12K.\n\n"
        )
        f.write("FINAL COMPARISON\n")
        f.write("-" * 124 + "\n")
        f.write(comparison.to_string(index=False))
        f.write("\n")

    log("")
    log("=" * 124)
    log("12K FINAL COMPARISON")
    log("=" * 124)
    log(comparison.to_string(index=False))
    log("")
    log(f"[SAVED] {output_dir / '12K_FINAL_REPORT.txt'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n[FATAL ERROR]", flush=True)
        traceback.print_exc()
        sys.exit(1)
