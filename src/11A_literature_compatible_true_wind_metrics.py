# -*- coding: utf-8 -*-
"""
11A_literature_compatible_true_wind_metrics.py

Post-hoc metric extraction for literature-level comparison with Saildrone /
unmanned-sailboat wind-forecasting papers (especially BP-STGNN and HD-Meta).

No model is trained, tuned, or modified.

Primary 10-min metrics:
    U RMSE
    V RMSE
    wind-speed RMSE
    meteorological wind-direction RMSE

Physics-Compact does not modify the Earth-relative wind branch, so its U/V
prediction is exactly the frozen Ridge wind prediction. The script verifies
this numerically for all frozen 07C seeds.

Published cross-paper values are CONTEXTUAL ONLY because datasets/splits and
protocols differ. Do not calculate a percentage improvement over BP-STGNN or
HD-Meta unless the same data/protocol are reproduced.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import platform
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "0.1.0-literature-compatible-true-wind"
SHARED_08C = "08C_frozen_external_inference.py"
SHARED_07C = "07C_confirm_and_compact_vessel_residual.py"

DEFAULT_DATASET_DIR = (
    r"D:\project\WindPredict_SaildroneData\data\forecasting"
    r"\SD1090_TPOS2024_JointForecasting_v0_1"
)
DEFAULT_COMPACT_DIR = (
    DEFAULT_DATASET_DIR + r"\confirm_and_compact_vessel_residual_v0_1"
)
DEFAULT_EXTERNAL_ROOT = (
    r"D:\project\WindPredict_SaildroneData\data\external"
    r"\SD1033_external_forecasting_v0_1"
)
DEFAULT_DATASETS = [
    "SD1090_validation",
    "SD1033_2024_matched_SD1090",
    "SD1033_2023_full",
    "SD1033_2022_full",
]
HORIZONS = [1, 2, 3, 5, 10]
FEATURES = [
    "UWND_MEAN", "VWND_MEAN", "TEMP_AIR_MEAN", "RH_MEAN", "SOG",
    "COG_sin", "COG_cos", "HDG_sin", "HDG_cos",
    "WING_ANGLE_sin", "WING_ANGLE_cos",
]
TARGETS = [
    "UWND_MEAN", "VWND_MEAN", "VESSEL_EAST_MPS", "VESSEL_NORTH_MPS",
    "HDG_sin", "HDG_cos",
]
PHYSICS_CKPT = "Physics_Compact_Vessel_Residual.pt"
DIRECTION_SPEED_MASK = 0.5


def log(msg=""):
    print(msg, flush=True)


def load_module(filename, module_name):
    path = Path(__file__).resolve().parent / filename
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod, path


def save_json(path, obj):
    def cv(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return None if np.isnan(x) else float(x)
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, dict):
            return {str(k): cv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [cv(v) for v in x]
        return x
    path.write_text(json.dumps(cv(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def wrap_deg(x):
    return (np.asarray(x, dtype=np.float64) + 180.0) % 360.0 - 180.0


def wind_from_deg(u, v):
    # CF / meteorological wind-from direction: atan2(-U, -V)
    return np.degrees(np.arctan2(-np.asarray(u), -np.asarray(v))) % 360.0


def rmse(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return np.nan if len(x) == 0 else float(np.sqrt(np.mean(x**2)))


def mae(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return np.nan if len(x) == 0 else float(np.mean(np.abs(x)))


def wind_metrics(truth, pred, dataset_id, model, h_idx, horizon, speed_mask):
    ut = truth[:, h_idx, 0].astype(np.float64)
    vt = truth[:, h_idx, 1].astype(np.float64)
    up = pred[:, h_idx, 0].astype(np.float64)
    vp = pred[:, h_idx, 1].astype(np.float64)
    good = np.isfinite(ut) & np.isfinite(vt) & np.isfinite(up) & np.isfinite(vp)
    ut, vt, up, vp = ut[good], vt[good], up[good], vp[good]

    du, dv = up-ut, vp-vt
    st = np.sqrt(ut**2 + vt**2)
    sp = np.sqrt(up**2 + vp**2)
    wd_err = wrap_deg(wind_from_deg(up, vp) - wind_from_deg(ut, vt))
    dmask = st >= float(speed_mask)

    u_rmse = rmse(du)
    v_rmse = rmse(dv)
    vector_rmse = float(np.sqrt(np.mean(du**2 + dv**2)))

    return {
        "dataset_id": dataset_id,
        "model": model,
        "horizon_min": int(horizon),
        "n_samples": int(len(ut)),
        "U_RMSE_mps": u_rmse,
        "V_RMSE_mps": v_rmse,
        "wind_vector_RMSE_mps": vector_rmse,
        "vector_identity_abs_error": abs(vector_rmse**2-u_rmse**2-v_rmse**2),
        "wind_speed_RMSE_mps": rmse(sp-st),
        "wind_speed_MAE_mps": mae(sp-st),
        "wind_direction_RMSE_all_deg": rmse(wd_err),
        "wind_direction_MAE_all_deg": mae(wd_err),
        "direction_speed_mask_mps": float(speed_mask),
        "n_direction_speedmask": int(dmask.sum()),
        "wind_direction_RMSE_speedmask_deg": rmse(wd_err[dmask]),
        "wind_direction_MAE_speedmask_deg": mae(wd_err[dmask]),
        "U_bias_mps": float(np.mean(du)),
        "V_bias_mps": float(np.mean(dv)),
        "wind_speed_bias_mps": float(np.mean(sp-st)),
    }


def load_dataset(dataset_id, core06, shared08c, dataset_dir, external_root):
    if dataset_id == "SD1090_validation":
        return core06.load_npz_split(dataset_dir, "validation")
    return shared08c.load_external_npz(external_root / "datasets" / f"{dataset_id}.npz")


def audit_physics_identity(core06, shared08c, torch, CompactClass, X, ridge_z,
                           ckpt_path, frozen, device, batch_size, amp):
    model = shared08c.instantiate_compact_model(
        CompactClass,
        feature_count=len(FEATURES),
        horizon_count=len(HORIZONS),
        target_count=len(TARGETS),
        gru_config=frozen["gru_config"],
        device=device,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    max_wind = 0.0
    max_fixed = 0.0
    fixed = [0, 1, 4, 5]
    with torch.no_grad():
        for a in range(0, X.shape[0], batch_size):
            b = min(X.shape[0], a+batch_size)
            xb = torch.from_numpy(np.asarray(X[a:b], dtype=np.float32)).to(device)
            rb = torch.from_numpy(np.asarray(ridge_z[a:b], dtype=np.float32)).to(device)
            with core06.AutocastContext(torch, amp, device.type):
                out = model(xb, rb)[0]
            p = out.detach().float().cpu().numpy()
            r = np.asarray(ridge_z[a:b], dtype=np.float32)
            max_wind = max(max_wind, float(np.max(np.abs(p[:, :, 0:2]-r[:, :, 0:2]))))
            max_fixed = max(max_fixed, float(np.max(np.abs(p[:, :, fixed]-r[:, :, fixed]))))
    del model, ckpt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return max_wind, max_fixed, bool(max_fixed <= 1e-7)


def literature_table():
    return pd.DataFrame([
        {
            "method": "BP-STGNN",
            "paper": "Nie et al., Ocean Engineering 362 (2026), 126026",
            "reported_horizon": "10-min-ahead",
            "U_RMSE_mps": 0.7486,
            "V_RMSE_mps": 0.7051,
            "wind_speed_RMSE_mps": 0.6994,
            "wind_direction_RMSE_deg": 5.58,
            "strict_head_to_head_allowed": False,
            "comparison_role": "contextual_literature_reference",
            "note": "Different Saildrone voyages/data split/protocol from our study.",
        },
        {
            "method": "HD-Meta",
            "paper": "Ning et al., Ocean Engineering 341 (2025), 122771",
            "reported_horizon": "paper protocol; 1-min forecasting time step",
            "U_RMSE_mps": np.nan,
            "V_RMSE_mps": np.nan,
            "wind_speed_RMSE_mps": 0.0735,
            "wind_direction_RMSE_deg": np.nan,
            "strict_head_to_head_allowed": False,
            "comparison_role": "contextual_only_fulltext_protocol_audit_required",
            "note": (
                "Abstract reports wind-speed RMSE=0.0735 on new-sea-area validation. "
                "Do not treat as directly comparable m/s performance until target, "
                "scaling/inverse-transform, voyage split and exact horizon are audited."
            ),
        },
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    ap.add_argument("--compact-dir", default=DEFAULT_COMPACT_DIR)
    ap.add_argument("--external-root", default=DEFAULT_EXTERNAL_ROOT)
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--ridge-chunk-size", type=int, default=16384)
    ap.add_argument("--direction-speed-mask", type=float, default=DIRECTION_SPEED_MASK)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--skip-physics-identity-audit", action="store_true")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    compact_dir = Path(args.compact_dir)
    external_root = Path(args.external_root)
    outdir = Path(args.output_dir) if args.output_dir else dataset_dir / "11A_literature_compatible_true_wind_metrics_v0_1"
    outdir.mkdir(parents=True, exist_ok=True)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    shared08c, p08c = load_module(SHARED_08C, "shared08c_11a")
    shared07c, p07c = load_module(SHARED_07C, "shared07c_11a")
    shared07b = shared07c.load_shared_07b()
    shared07a = shared07b.load_shared_07a()
    core06 = shared07a.load_core()
    torch, nn, _, _ = core06.import_torch()

    frozen = shared08c.load_frozen_context(
        shared07c, shared07a, core06, compact_dir, dataset_dir
    )
    if list(frozen["frozen_manifest"]["feature_names"]) != FEATURES:
        raise RuntimeError("Frozen feature schema mismatch.")
    if list(frozen["frozen_manifest"]["target_names"]) != TARGETS:
        raise RuntimeError("Frozen target schema mismatch.")

    seeds = [int(x) for x in frozen["seed_schedule"]]
    device = torch.device("cpu") if args.force_cpu or not torch.cuda.is_available() else torch.device("cuda:0")
    amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        core06.configure_tf32(torch, True)
    CompactClass = shared07c.build_compact_model_class(torch, nn)

    log("="*110)
    log("11A - LITERATURE-COMPATIBLE TRUE-WIND METRICS")
    log("="*110)
    log(f"datasets               : {datasets}")
    log("primary literature h   : 10 min")
    log(f"direction speed mask   : {args.direction_speed_mask:.3f} m/s")
    log(f"frozen seeds           : {seeds}")
    log(f"device / AMP           : {device} / {amp}")
    log("training/fine-tuning   : NONE")
    log("cross-paper % gain     : NOT COMPUTED")
    log("")

    rows = []
    audit_rows = []

    for di, dataset_id in enumerate(datasets, 1):
        log("#"*110)
        log(f"[DATASET {di}/{len(datasets)}] {dataset_id}")
        log("#"*110)
        d = load_dataset(dataset_id, core06, shared08c, dataset_dir, external_root)
        X = d["X"].astype(np.float32, copy=False)
        y = d["y_raw"].astype(np.float32, copy=False)

        ridge_z = shared07a.ridge_predict_z(
            frozen["ridge_model"], X, len(HORIZONS), len(TARGETS), args.ridge_chunk_size
        )
        ridge_raw = shared07a.inverse_target(
            ridge_z, frozen["target_mean"], frozen["target_std"]
        )
        persistence_raw = core06.persistence_joint_prediction(
            X, FEATURES, frozen["feature_mean"], frozen["feature_std"], len(HORIZONS)
        )

        for j, h in enumerate(HORIZONS):
            rows.append(wind_metrics(y, persistence_raw, dataset_id, "Persistence", j, h, args.direction_speed_mask))
            r = wind_metrics(y, ridge_raw, dataset_id, "Frozen-Ridge", j, h, args.direction_speed_mask)
            rows.append(r)
            p = dict(r)
            p["model"] = "Frozen-Physics-Compact-Vessel-Residual"
            p["wind_metric_source"] = "identical frozen Ridge wind branch by architecture"
            rows.append(p)

        if not args.skip_physics_identity_audit:
            log("  Physics-Compact all-seed wind identity audit:")
            for seed in seeds:
                ckpt = compact_dir / "models" / f"seed_{seed}" / PHYSICS_CKPT
                if not ckpt.exists():
                    raise FileNotFoundError(ckpt)
                max_wind, max_fixed, passed = audit_physics_identity(
                    core06, shared08c, torch, CompactClass, X, ridge_z, ckpt,
                    frozen, device, args.batch_size, amp
                )
                audit_rows.append({
                    "dataset_id": dataset_id,
                    "seed": seed,
                    "max_abs_wind_difference_standardized": max_wind,
                    "max_abs_all_frozen_channels_difference_standardized": max_fixed,
                    "identity_passed": passed,
                    "checkpoint_path": str(ckpt),
                })
                log(f"    seed={seed} | max wind diff={max_wind:.3e} | PASS={passed}")
                if not passed:
                    raise RuntimeError(f"Physics-Compact/Ridge identity failed for {dataset_id}/seed{seed}")

        m = wind_metrics(y, ridge_raw, dataset_id, "Frozen-Ridge / Physics-Compact wind", 4, 10, args.direction_speed_mask)
        log(
            f"  10-min | U={m['U_RMSE_mps']:.4f} | V={m['V_RMSE_mps']:.4f} | "
            f"WS={m['wind_speed_RMSE_mps']:.4f} | "
            f"WD(all)={m['wind_direction_RMSE_all_deg']:.3f} deg | "
            f"WD(mask)={m['wind_direction_RMSE_speedmask_deg']:.3f} deg"
        )

        del d, X, y, ridge_z, ridge_raw, persistence_raw
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = pd.DataFrame(rows)
    if "wind_metric_source" not in metrics.columns:
        metrics["wind_metric_source"] = ""
    metrics["wind_metric_source"] = metrics["wind_metric_source"].fillna("")
    metrics.to_csv(outdir/"our_true_wind_metrics_per_horizon.csv", index=False, encoding="utf-8-sig")

    metrics10 = metrics[metrics.horizon_min == 10].copy().reset_index(drop=True)
    metrics10.to_csv(outdir/"our_true_wind_metrics_10min.csv", index=False, encoding="utf-8-sig")

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(outdir/"physicscompact_wind_identity_audit.csv", index=False, encoding="utf-8-sig")

    literature = literature_table()
    literature.to_csv(outdir/"literature_context_table.csv", index=False, encoding="utf-8-sig")

    combined_rows = []
    for _, r in metrics10[metrics10.model.isin(["Frozen-Ridge", "Frozen-Physics-Compact-Vessel-Residual"])].iterrows():
        combined_rows.append({
            "method": r.model,
            "paper_or_dataset": r.dataset_id,
            "horizon": "10-min-ahead",
            "U_RMSE_mps": r.U_RMSE_mps,
            "V_RMSE_mps": r.V_RMSE_mps,
            "wind_speed_RMSE_mps": r.wind_speed_RMSE_mps,
            "wind_direction_RMSE_deg": r.wind_direction_RMSE_all_deg,
            "wind_direction_RMSE_speedmask_deg": r.wind_direction_RMSE_speedmask_deg,
            "comparison_role": "our_frozen_evaluation",
            "strict_head_to_head_allowed": False,
            "note": "Different dataset/protocol from literature rows.",
        })
    for _, r in literature.iterrows():
        combined_rows.append({
            "method": r.method,
            "paper_or_dataset": r.paper,
            "horizon": r.reported_horizon,
            "U_RMSE_mps": r.U_RMSE_mps,
            "V_RMSE_mps": r.V_RMSE_mps,
            "wind_speed_RMSE_mps": r.wind_speed_RMSE_mps,
            "wind_direction_RMSE_deg": r.wind_direction_RMSE_deg,
            "wind_direction_RMSE_speedmask_deg": np.nan,
            "comparison_role": r.comparison_role,
            "strict_head_to_head_allowed": r.strict_head_to_head_allowed,
            "note": r.note,
        })
    combined = pd.DataFrame(combined_rows)
    combined.to_csv(outdir/"combined_contextual_comparison_table.csv", index=False, encoding="utf-8-sig")

    # Exact metric identity between Ridge and Physics-Compact.
    r10 = metrics10[metrics10.model == "Frozen-Ridge"].sort_values("dataset_id")
    p10 = metrics10[metrics10.model == "Frozen-Physics-Compact-Vessel-Residual"].sort_values("dataset_id")
    cols = [
        "U_RMSE_mps", "V_RMSE_mps", "wind_vector_RMSE_mps",
        "wind_speed_RMSE_mps", "wind_direction_RMSE_all_deg",
        "wind_direction_RMSE_speedmask_deg",
    ]
    max_metric_diff = max(
        float(np.max(np.abs(r10[c].to_numpy(float)-p10[c].to_numpy(float))))
        for c in cols
    )
    if max_metric_diff > 1e-12:
        raise RuntimeError("Ridge and Physics-Compact true-wind metrics are not identical.")

    manifest = {
        "version": VERSION,
        "datasets": datasets,
        "primary_horizon_min": 10,
        "direction_definition": "meteorological wind-from = atan2(-U,-V); circular error in [-180,180)",
        "direction_speed_mask_mps": args.direction_speed_mask,
        "physicscompact_wind_equals_ridge": True,
        "physicscompact_metric_identity_max_abs_error": max_metric_diff,
        "all_seed_identity_audit_performed": not args.skip_physics_identity_audit,
        "literature": {
            "BP-STGNN": {"U_RMSE_mps":0.7486, "V_RMSE_mps":0.7051,
                           "WS_RMSE_mps":0.6994, "WD_RMSE_deg":5.58,
                           "horizon":"10-min-ahead", "contextual_only":True},
            "HD-Meta": {"reported_WS_RMSE":0.0735,
                         "contextual_only":True,
                         "fulltext_protocol_audit_required":True},
        },
        "policy": {
            "training": False,
            "fine_tuning": False,
            "model_change": False,
            "seed_selection": False,
            "external_scaler_fit": False,
            "cross_paper_percentage_improvement": False,
        },
        "shared_scripts": {"08C":str(p08c), "07C":str(p07c)},
        "software": {
            "python":sys.version, "platform":platform.platform(),
            "numpy":np.__version__, "pandas":pd.__version__,
            "torch":torch.__version__, "torch_cuda":torch.version.cuda,
            "device":str(device), "amp":amp,
        },
    }
    save_json(outdir/"literature_metric_manifest.json", manifest)

    with (outdir/"literature_metric_report.txt").open("w", encoding="utf-8") as f:
        f.write("11A Literature-Compatible True-Wind Metrics\n" + "="*100 + "\n\n")
        f.write("No training/fine-tuning/model change/seed selection.\n")
        f.write("Physics-Compact true-wind channels are exactly frozen Ridge channels.\n")
        f.write("Cross-paper values are contextual only; no percentage gain is valid across different datasets/protocols.\n\n")
        f.write("Our 10-min results\n" + "-"*100 + "\n")
        f.write(metrics10.to_string(index=False))
        f.write("\n\nLiterature context\n" + "-"*100 + "\n")
        f.write(literature.to_string(index=False))
        f.write("\n\nDefinitions\n" + "-"*100 + "\n")
        f.write("U_RMSE=sqrt(mean((Uhat-U)^2)); V analogous.\n")
        f.write("WS=sqrt(U^2+V^2).\n")
        f.write("WD=meteorological wind-from atan2(-U,-V), circular error.\n")
        f.write("Our vector RMSE obeys vector_RMSE^2=U_RMSE^2+V_RMSE^2.\n")

    log("")
    log("="*110)
    log("11A LITERATURE-COMPATIBLE 10-MIN TRUE-WIND RESULTS")
    log("="*110)
    for dataset_id in datasets:
        q = metrics10[(metrics10.dataset_id==dataset_id)&(metrics10.model=="Frozen-Ridge")]
        if len(q):
            r = q.iloc[0]
            log(f"{dataset_id}:")
            log(f"  PhysicsCompact/Ridge wind: U={r.U_RMSE_mps:.4f} m/s | V={r.V_RMSE_mps:.4f} m/s")
            log(f"  WS={r.wind_speed_RMSE_mps:.4f} m/s | WD(all)={r.wind_direction_RMSE_all_deg:.3f} deg | WD(mask)={r.wind_direction_RMSE_speedmask_deg:.3f} deg")
    log("")
    log("BP-STGNN published context: U=0.7486 | V=0.7051 | WS=0.6994 m/s | WD=5.58 deg (10 min)")
    log("HD-Meta published context: WS RMSE=0.0735 under its own new-sea-area protocol; full protocol audit required.")
    log(f"[AUDIT] PhysicsCompact/Ridge true-wind metric max difference = {max_metric_diff:.3e}")
    log("[POLICY] Literature values are contextual only; no cross-paper percentage gain is computed.")
    log(f"[DONE] 11A outputs: {outdir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n[FATAL ERROR]", flush=True)
        traceback.print_exc()
        print("\nPlease send the complete traceback and terminal output.", flush=True)
        sys.exit(1)
