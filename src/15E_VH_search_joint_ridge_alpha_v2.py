# -*- coding: utf-8 -*-
"""
Coarse-to-fine optimization of the joint Ridge alpha used by 15E-VH.

Uses SD1090 TRAIN/VALIDATION only.
Never opens internal TEST or SD1033 external products.

Stage A:
  Cheap broad Ridge-anchor sweep.

Stage B (--full-search):
  Retrain the complete five-seed 15E-VH motion branch for the best
  alpha candidates, then select alpha using the same direct-motion
  philosophy used by the final motion branch:
      0.5 * Vessel-vector ratio + 0.5 * HDG-MAE ratio
  relative to the alpha=100 fully retrained model.
  AppVector RMSE is only a tie-breaker.

After alpha selection, freeze it and rerun downstream inference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\project\WindPredict_SaildroneData")

DEFAULT_DATASET_DIR = ROOT / "data" / "forecasting" / "SD1090_TPOS2024_JointForecasting_v0_1"
DEFAULT_STAGE15EH_DIR = ROOT / "data" / "forecasting" / "15E_H_HDG_GatedResidual_5seed_v0_1"
DEFAULT_SRC_DIR = ROOT / "src"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "forecasting" / "15E_VH_AlphaSearch_v0_1"

EPS = 1e-12

DEFAULT_ALPHAS = [
    0.0, 0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 300.0,
    600.0, 1000.0, 1500.0, 2250.0, 3000.0, 5000.0,
    10000.0, 30000.0,
]


def log(msg=""):
    print(msg, flush=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_train_val(dataset_dir: Path):
    def one(split):
        p = dataset_dir / f"{split}.npz"
        if not p.exists():
            raise FileNotFoundError(p)
        with np.load(p, allow_pickle=False) as z:
            X = np.asarray(z["X"], dtype=np.float32)
            y = np.asarray(z["y_raw"], dtype=np.float32)

        if X.shape[1:] != (60, 11):
            raise RuntimeError(f"{split}: expected X (*,60,11), got {X.shape}")
        if y.shape[1:] != (5, 6):
            raise RuntimeError(f"{split}: expected y_raw (*,5,6), got {y.shape}")
        return X, y

    Xtr, ytr = one("train")
    Xva, yva = one("validation")
    return Xtr, ytr, Xva, yva


def normalize_heading(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.sqrt(np.sum(q * q, axis=-1, keepdims=True))
    return q / np.maximum(n, EPS)


def heading_abs_error_deg(true_pair, pred_pair):
    tq = normalize_heading(true_pair)
    pq = normalize_heading(pred_pair)
    a_true = np.arctan2(tq[..., 0], tq[..., 1])
    a_pred = np.arctan2(pq[..., 0], pq[..., 1])
    d = np.arctan2(np.sin(a_pred - a_true), np.cos(a_pred - a_true))
    return np.abs(np.rad2deg(d))


def mean_motion_metrics(y_true_raw, y_pred_raw):
    yt = np.asarray(y_true_raw, dtype=np.float64)[:, :, 2:6]
    yp = np.asarray(y_pred_raw, dtype=np.float64)[:, :, 2:6].copy()
    yp[:, :, 2:4] = normalize_heading(yp[:, :, 2:4])

    vessel = []
    sog = []
    hdg_mae = []
    hdg_rmse = []

    for h in range(5):
        dv = yp[:, h, 0:2] - yt[:, h, 0:2]
        vessel.append(float(np.sqrt(np.mean(np.sum(dv * dv, axis=1)))))

        st = np.sqrt(np.sum(yt[:, h, 0:2] ** 2, axis=1))
        sp = np.sqrt(np.sum(yp[:, h, 0:2] ** 2, axis=1))
        sog.append(float(np.sqrt(np.mean((sp - st) ** 2))))

        ae = heading_abs_error_deg(yt[:, h, 2:4], yp[:, h, 2:4])
        hdg_mae.append(float(np.mean(ae)))
        hdg_rmse.append(float(np.sqrt(np.mean(ae * ae))))

    return {
        "Vessel_vector_RMSE_mps": float(np.mean(vessel)),
        "SOG_RMSE_mps": float(np.mean(sog)),
        "HDG_MAE_deg": float(np.mean(hdg_mae)),
        "HDG_RMSE_deg": float(np.mean(hdg_rmse)),
    }


def build_ridge_path(emod, X_train, y_train_raw):
    Xf = np.asarray(X_train, dtype=np.float32).reshape(len(X_train), -1)
    yf = np.asarray(y_train_raw, dtype=np.float32).reshape(len(y_train_raw), -1)

    y_mean = np.mean(yf.astype(np.float64), axis=0)
    y_std = np.std(yf.astype(np.float64), axis=0, ddof=0)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)

    yz = ((yf - y_mean[None, :]) / y_std[None, :]).astype(np.float32)
    stats = emod.compute_stats(Xf, yz)

    n = float(stats["n"])
    x_mean = stats["sum_x"] / n
    y_mean_z = stats["sum_y"] / n

    G = stats["sum_x2"] - np.outer(stats["sum_x"], stats["sum_x"]) / n
    G = 0.5 * (G + G.T)

    C = stats["sum_xy"] - np.outer(stats["sum_x"], y_mean_z)

    evals, Q = np.linalg.eigh(G)
    evals = np.maximum(evals, 0.0)
    QtC = Q.T @ C

    return {
        "x_mean": x_mean,
        "y_mean_z": y_mean_z,
        "y_mean_raw": y_mean,
        "y_std_raw": y_std,
        "evals": evals,
        "Q": Q,
        "QtC": QtC,
    }


def predict_ridge(path, X, alpha):
    denom = np.maximum(path["evals"] + float(alpha), 1e-10)
    B = path["Q"] @ (path["QtC"] / denom[:, None])
    intercept = path["y_mean_z"] - path["x_mean"] @ B

    Xf = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
    pred_z = np.empty((len(Xf), 30), dtype=np.float32)

    for a in range(0, len(Xf), 4096):
        b = min(a + 4096, len(Xf))
        pred_z[a:b] = (
            np.asarray(Xf[a:b], dtype=np.float64) @ B
            + intercept[None, :]
        ).astype(np.float32)

    pred_raw = (
        pred_z.astype(np.float64) * path["y_std_raw"][None, :]
        + path["y_mean_raw"][None, :]
    ).reshape(-1, 5, 6)

    return pred_raw.astype(np.float32)


def coarse_sweep(emod, Xtr, ytr, Xva, yva, alphas):
    log("[A1/3] Building sufficient statistics and eigendecomposition.")
    path = build_ridge_path(emod, Xtr, ytr)

    rows = []
    for alpha in alphas:
        log(f"  alpha={alpha:g}")
        pred = predict_ridge(path, Xva, alpha)
        rows.append({
            "alpha": float(alpha),
            **mean_motion_metrics(yva, pred),
        })

    df = pd.DataFrame(rows)
    ref = df.loc[np.isclose(df["alpha"], 100.0)]
    if len(ref) != 1:
        raise RuntimeError("alpha=100 must be present exactly once.")
    ref = ref.iloc[0]

    df["Vessel_ratio_vs_a100"] = (
        df["Vessel_vector_RMSE_mps"] /
        float(ref["Vessel_vector_RMSE_mps"])
    )
    df["HDG_ratio_vs_a100"] = (
        df["HDG_MAE_deg"] /
        float(ref["HDG_MAE_deg"])
    )
    df["coarse_motion_score"] = (
        0.5 * df["Vessel_ratio_vs_a100"]
        + 0.5 * df["HDG_ratio_vs_a100"]
    )

    return df.sort_values(
        ["coarse_motion_score", "Vessel_vector_RMSE_mps", "HDG_MAE_deg"]
    ).reset_index(drop=True)


def select_fine_candidates(coarse, top_k):
    ranked = coarse["alpha"].tolist()
    grid = sorted(coarse["alpha"].tolist())

    selected = set(ranked[:max(1, top_k)])
    selected.add(100.0)

    # Add neighbors of the two strongest coarse points.
    for alpha in ranked[:2]:
        j = grid.index(alpha)
        if j > 0:
            selected.add(grid[j - 1])
        if j + 1 < len(grid):
            selected.add(grid[j + 1])

    return sorted(selected)


def read_full_result(run_dir: Path, alpha: float):
    p = run_dir / "15E_VH_per_seed_validation_summary.csv"
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)
    cand = df.loc[df["model"] == "15E-VH"].copy()
    if len(cand) != 5:
        raise RuntimeError(f"{run_dir}: expected 5 candidate seed rows, got {len(cand)}")

    row = {"alpha": float(alpha), "n_seeds": int(len(cand))}
    metrics = [
        "Vessel_vector_RMSE_mps",
        "SOG_RMSE_mps",
        "HDG_MAE_deg",
        "HDG_RMSE_deg",
        "AppVector_RMSE_mps",
        "AWS_RMSE_mps",
        "AWA_MAE_deg",
        "AWA_RMSE_deg",
    ]

    for m in metrics:
        row[m + "_mean"] = float(cand[m].mean())
        row[m + "_std"] = float(cand[m].std(ddof=1))

    return row


def run_full_search(args, candidates):
    trainer = args.src_dir / "15E_VH_joint_VesselHDG_GatedResidual_alpha_v2.py"
    if not trainer.exists():
        raise FileNotFoundError(
            f"Put the parameterized trainer in src first: {trainer}"
        )

    full_root = args.output_dir / "full_retrain"
    full_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for alpha in candidates:
        tag = f"{alpha:g}".replace(".", "p").replace("-", "m")
        run_dir = full_root / f"alpha_{tag}"

        log("")
        log("=" * 100)
        log(f"[FULL RETRAIN] alpha={alpha:g}")
        log("=" * 100)

        cmd = [
            sys.executable,
            str(trainer),
            "--dataset-dir", str(args.dataset_dir),
            "--stage15eh-dir", str(args.stage15eh_dir),
            "--src-dir", str(args.src_dir),
            "--output-dir", str(run_dir),
            "--ridge-alpha", str(alpha),
        ]
        subprocess.run(cmd, check=True)
        rows.append(read_full_result(run_dir, alpha))

    df = pd.DataFrame(rows)

    ref = df.loc[np.isclose(df["alpha"], 100.0)]
    if len(ref) != 1:
        raise RuntimeError("Full search must contain alpha=100.")
    ref = ref.iloc[0]

    df["Vessel_ratio_vs_a100"] = (
        df["Vessel_vector_RMSE_mps_mean"] /
        float(ref["Vessel_vector_RMSE_mps_mean"])
    )
    df["HDG_ratio_vs_a100"] = (
        df["HDG_MAE_deg_mean"] /
        float(ref["HDG_MAE_deg_mean"])
    )

    df["full_motion_score"] = (
        0.5 * df["Vessel_ratio_vs_a100"]
        + 0.5 * df["HDG_ratio_vs_a100"]
    )

    return df.sort_values(
        [
            "full_motion_score",
            "AppVector_RMSE_mps_mean",
            "Vessel_vector_RMSE_mps_mean",
            "HDG_MAE_deg_mean",
        ]
    ).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--stage15eh-dir", type=Path, default=DEFAULT_STAGE15EH_DIR)
    parser.add_argument("--src-dir", type=Path, default=DEFAULT_SRC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alphas", type=float, nargs="*", default=DEFAULT_ALPHAS)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--full-search", action="store_true")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    alphas = sorted(set(float(x) for x in args.alphas))
    if any(x < 0 for x in alphas):
        raise ValueError("All alphas must be >= 0.")
    if 100.0 not in alphas:
        alphas.append(100.0)
        alphas = sorted(set(alphas))

    emod = load_module(
        "stage15e_helper_alpha_search",
        args.src_dir / "15E_hybrid_NewWind_OldCompactVessel_5seed_validation.py",
    )

    log("=" * 100)
    log("15E-VH JOINT RIDGE ALPHA SEARCH")
    log("SD1090 TRAIN/VALIDATION ONLY")
    log("TEST AND SD1033 ARE NOT OPENED")
    log("=" * 100)

    log("[A0/3] Loading train/validation.")
    Xtr, ytr, Xva, yva = load_train_val(args.dataset_dir)
    log(f"  train      = {len(Xtr):,}")
    log(f"  validation = {len(Xva):,}")

    coarse = coarse_sweep(emod, Xtr, ytr, Xva, yva, alphas)
    coarse.to_csv(
        args.output_dir / "15E_VH_joint_ridge_alpha_coarse.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log("[A2/3] COARSE RIDGE-ANCHOR RANKING")
    log(
        coarse[
            [
                "alpha",
                "Vessel_vector_RMSE_mps",
                "HDG_MAE_deg",
                "SOG_RMSE_mps",
                "coarse_motion_score",
            ]
        ].head(12).to_string(index=False)
    )

    fine = select_fine_candidates(coarse, int(args.top_k))
    pd.DataFrame({"alpha": fine}).to_csv(
        args.output_dir / "15E_VH_joint_ridge_alpha_fine_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("")
    log("[A3/3] Fine candidates:")
    log("  " + ", ".join(f"{x:g}" for x in fine))

    report = {
        "coarse_best_alpha": float(coarse.iloc[0]["alpha"]),
        "fine_candidates": [float(x) for x in fine],
        "selection_rule": (
            "0.5*VesselVector_ratio_to_alpha100 + "
            "0.5*HDG_MAE_ratio_to_alpha100; "
            "AppVector used as full-search tie-breaker"
        ),
        "test_loaded": False,
        "external_loaded": False,
    }

    if args.full_search:
        full = run_full_search(args, fine)
        full.to_csv(
            args.output_dir / "15E_VH_joint_ridge_alpha_full_5seed.csv",
            index=False,
            encoding="utf-8-sig",
        )

        best = full.iloc[0]
        report.update({
            "selected_alpha": float(best["alpha"]),
            "selected_full_motion_score": float(best["full_motion_score"]),
            "selected_vessel_vector_rmse_mean": float(
                best["Vessel_vector_RMSE_mps_mean"]
            ),
            "selected_hdg_mae_mean": float(best["HDG_MAE_deg_mean"]),
            "selected_appvector_rmse_mean": float(
                best["AppVector_RMSE_mps_mean"]
            ),
            "selected_aws_rmse_mean": float(best["AWS_RMSE_mps_mean"]),
            "selected_awa_mae_mean": float(best["AWA_MAE_deg_mean"]),
        })

        log("")
        log("=" * 100)
        log("FULL FIVE-SEED ALPHA RANKING")
        log("=" * 100)
        log(
            full[
                [
                    "alpha",
                    "Vessel_vector_RMSE_mps_mean",
                    "HDG_MAE_deg_mean",
                    "AppVector_RMSE_mps_mean",
                    "AWS_RMSE_mps_mean",
                    "AWA_MAE_deg_mean",
                    "full_motion_score",
                ]
            ].to_string(index=False)
        )

        log("")
        log(f"[SELECTED ALPHA] {float(best['alpha']):g}")
        log("Freeze this alpha before downstream test/external reruns.")
    else:
        log("")
        log("Coarse search complete. Add --full-search for final five-seed selection.")

    with (
        args.output_dir / "15E_VH_joint_ridge_alpha_search_report.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log("")
    log(f"[SAVED] {args.output_dir}")


if __name__ == "__main__":
    main()
