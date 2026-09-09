# -*- coding: utf-8 -*-
"""
15B_F_finalize_TrueWind_alpha1.py

Finalize Stage15B after validation-only alpha-cap audit.

What this script does
---------------------
NO neural retraining.

It:
1. Reads Stage15B validation raw residual predictions.
2. Recomputes analytical alpha from VALIDATION only.
3. Uses alpha_cap = 1.0.
4. Freezes the resulting 5x2 alpha matrix.
5. Re-applies frozen alpha to:
       - validation
       - test
       - train cross-fitted OOF raw residual predictions
6. Rewrites clean final Stage2 true-wind prediction files for Stage15C.
7. Recomputes final Ridge vs Stage15B metrics.

Important
---------
TEST is NEVER used to determine alpha.

The train OOF file already contains one prediction per sample:
    ridge_oof
    raw_residual_oof

Even though an earlier Stage15B run redundantly trained duplicate fold rows,
the saved array has one final prediction per sample. This script does not
retrain those networks; it only re-applies the newly frozen validation alpha.

Final alpha:
    alpha = clip(alpha_unconstrained_validation, 0, 1.0)

Expected Stage15B directory:
    15B_1min_TrueWind_Residual_v0_2

Recommended PowerShell
----------------------
python "D:\\project\\WindPredict_SaildroneData\\src\\15B_F_finalize_TrueWind_alpha1.py" --stage15b-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15B_1min_TrueWind_Residual_v0_2" --output-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15B_F_1min_TrueWind_Residual_v0_3"
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\project\WindPredict_SaildroneData")

DEFAULT_STAGE15B_DIR = (
    ROOT
    / "data"
    / "forecasting"
    / "15B_1min_TrueWind_Residual_v0_2"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "data"
    / "forecasting"
    / "15B_F_1min_TrueWind_Residual_v0_3"
)

HORIZONS = [1, 2, 3, 5, 10]
ALPHA_CAP = 1.0
EPS = 1e-12


def log(msg=""):
    print(msg, flush=True)


def save_json(path: Path, obj):
    def cv(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return None if not np.isfinite(v) else float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, dict):
            return {str(k): cv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [cv(x) for x in v]
        return v

    with path.open("w", encoding="utf-8") as f:
        json.dump(cv(obj), f, ensure_ascii=False, indent=2, sort_keys=True)


def alpha_unconstrained(y, ridge, raw):
    e = (
        np.asarray(y, dtype=np.float64)
        - np.asarray(ridge, dtype=np.float64)
    )

    r = np.asarray(raw, dtype=np.float64)

    alpha = np.zeros((len(HORIZONS), 2), dtype=np.float64)

    for h in range(len(HORIZONS)):
        for c in range(2):
            rr = r[:, h, c]
            ee = e[:, h, c]

            denom = float(np.dot(rr, rr))

            if denom <= EPS:
                a = 0.0
            else:
                a = float(np.dot(ee, rr) / denom)

            alpha[h, c] = a

    return alpha


def apply_alpha(ridge, raw, alpha):
    return (
        np.asarray(ridge, dtype=np.float64)
        + np.asarray(raw, dtype=np.float64)
        * np.asarray(alpha, dtype=np.float64)[None, :, :]
    ).astype(np.float32)


def wd_from_uv(uv):
    uv = np.asarray(uv, dtype=np.float64)

    return (
        np.degrees(
            np.arctan2(
                -uv[..., 0],
                -uv[..., 1],
            )
        )
        % 360.0
    )


def circular_diff_deg(a, b):
    return (
        (
            np.asarray(a)
            - np.asarray(b)
            + 180.0
        )
        % 360.0
        - 180.0
    )


def metrics_per_horizon(y_true, y_pred, split, model):
    rows = []

    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    for j, h in enumerate(HORIZONS):
        t = yt[:, j, :]
        p = yp[:, j, :]

        e = p - t

        ws_t = np.linalg.norm(t, axis=1)
        ws_p = np.linalg.norm(p, axis=1)

        wd_err = circular_diff_deg(
            wd_from_uv(p),
            wd_from_uv(t),
        )

        rows.append(
            {
                "split": split,
                "model": model,
                "horizon_min": h,
                "U_RMSE_mps": float(
                    np.sqrt(np.mean(e[:, 0] ** 2))
                ),
                "V_RMSE_mps": float(
                    np.sqrt(np.mean(e[:, 1] ** 2))
                ),
                "vector_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            np.sum(e ** 2, axis=1)
                        )
                    )
                ),
                "WS_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            (ws_p - ws_t) ** 2
                        )
                    )
                ),
                "WD_RMSE_deg": float(
                    np.sqrt(
                        np.mean(
                            wd_err ** 2
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def residual_corr(y, ridge, raw):
    true_res = (
        np.asarray(y, dtype=np.float64)
        - np.asarray(ridge, dtype=np.float64)
    )

    raw = np.asarray(raw, dtype=np.float64)

    corr = np.full((len(HORIZONS), 2), np.nan, dtype=np.float64)

    for h in range(len(HORIZONS)):
        for c in range(2):
            a = true_res[:, h, c]
            b = raw[:, h, c]

            if (
                np.std(a) > EPS
                and np.std(b) > EPS
            ):
                corr[h, c] = np.corrcoef(a, b)[0, 1]

    return corr


def load_pred_file(path: Path, keys):
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as z:
        out = {}

        for key in keys:
            if key not in z.files:
                raise RuntimeError(
                    f"{path}: key '{key}' missing. "
                    f"Available={z.files}"
                )

            out[key] = np.asarray(z[key])

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage15b-dir",
        type=Path,
        default=DEFAULT_STAGE15B_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    log("=" * 140)
    log("15B-F — FINALIZE TRUE-WIND RESIDUAL WITH VALIDATION ALPHA CAP=1.0")
    log("=" * 140)
    log("NO neural retraining.")
    log("Alpha is determined from VALIDATION only.")

    val_path = (
        args.stage15b_dir
        / "15B_validation_predictions.npz"
    )

    test_path = (
        args.stage15b_dir
        / "15B_test_predictions.npz"
    )

    train_path = (
        args.stage15b_dir
        / "15B_train_stage2_oof_predictions.npz"
    )

    log("")
    log("[STAGE 1/5] Loading validation predictions for alpha freeze.")

    val = load_pred_file(
        val_path,
        [
            "y_true",
            "ridge",
            "raw_residual",
        ],
    )

    unconstrained = alpha_unconstrained(
        val["y_true"],
        val["ridge"],
        val["raw_residual"],
    )

    alpha = np.clip(
        unconstrained,
        0.0,
        ALPHA_CAP,
    ).astype(np.float32)

    log("")
    log("[FROZEN ALPHA]")

    for j, h in enumerate(HORIZONS):
        log(
            f"  {h:2d} min | "
            f"U={alpha[j,0]:.8f} | "
            f"V={alpha[j,1]:.8f}"
        )

    if np.max(alpha) >= ALPHA_CAP - 1e-7:
        log(
            "[WARNING] At least one alpha still touches cap=1.0."
        )
    else:
        log(
            "[OK] No alpha touches cap=1.0; cap is non-binding."
        )

    log("")
    log("[STAGE 2/5] Loading test and train-OOF raw residual outputs.")

    test = load_pred_file(
        test_path,
        [
            "y_true",
            "ridge",
            "raw_residual",
        ],
    )

    train = load_pred_file(
        train_path,
        [
            "y_true",
            "ridge_oof",
            "raw_residual_oof",
        ],
    )

    final_val = apply_alpha(
        val["ridge"],
        val["raw_residual"],
        alpha,
    )

    final_test = apply_alpha(
        test["ridge"],
        test["raw_residual"],
        alpha,
    )

    final_train_oof = apply_alpha(
        train["ridge_oof"],
        train["raw_residual_oof"],
        alpha,
    )

    log("")
    log("[STAGE 3/5] Recomputing final metrics.")

    metric_frames = []

    for split, y, ridge, final in [
        (
            "train_OOF",
            train["y_true"],
            train["ridge_oof"],
            final_train_oof,
        ),
        (
            "validation",
            val["y_true"],
            val["ridge"],
            final_val,
        ),
        (
            "test",
            test["y_true"],
            test["ridge"],
            final_test,
        ),
    ]:
        metric_frames.append(
            metrics_per_horizon(
                y,
                ridge,
                split,
                "TrueWind-Ridge",
            )
        )

        metric_frames.append(
            metrics_per_horizon(
                y,
                final,
                split,
                "Stage15B-Final",
            )
        )

    metrics = pd.concat(
        metric_frames,
        ignore_index=True,
    )

    metrics.to_csv(
        out
        / "15B_F_metrics_per_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = []

    for (split, model), sub in metrics.groupby(
        ["split", "model"]
    ):
        summary_rows.append(
            {
                "split": split,
                "model": model,
                "mean_U_RMSE_mps": float(
                    sub["U_RMSE_mps"].mean()
                ),
                "mean_V_RMSE_mps": float(
                    sub["V_RMSE_mps"].mean()
                ),
                "mean_vector_RMSE_mps": float(
                    sub["vector_RMSE_mps"].mean()
                ),
                "mean_WS_RMSE_mps": float(
                    sub["WS_RMSE_mps"].mean()
                ),
                "mean_WD_RMSE_deg": float(
                    sub["WD_RMSE_deg"].mean()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        out
        / "15B_F_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    val_corr = residual_corr(
        val["y_true"],
        val["ridge"],
        val["raw_residual"],
    )

    test_corr = residual_corr(
        test["y_true"],
        test["ridge"],
        test["raw_residual"],
    )

    log("")
    log("[STAGE 4/5] Saving Stage15C-ready final true-wind files.")

    np.savez_compressed(
        out
        / "15B_F_train_stage2_oof_predictions.npz",
        y_true=np.asarray(
            train["y_true"],
            dtype=np.float32,
        ),
        ridge_oof=np.asarray(
            train["ridge_oof"],
            dtype=np.float32,
        ),
        raw_residual_oof=np.asarray(
            train["raw_residual_oof"],
            dtype=np.float32,
        ),
        stage2_truewind_oof=final_train_oof.astype(
            np.float32
        ),
        alpha=alpha.astype(
            np.float32
        ),
        horizons_min=np.asarray(
            HORIZONS,
            dtype=np.int64,
        ),
    )

    np.savez_compressed(
        out
        / "15B_F_validation_predictions.npz",
        y_true=np.asarray(
            val["y_true"],
            dtype=np.float32,
        ),
        ridge=np.asarray(
            val["ridge"],
            dtype=np.float32,
        ),
        raw_residual=np.asarray(
            val["raw_residual"],
            dtype=np.float32,
        ),
        stage2_truewind=final_val.astype(
            np.float32
        ),
        alpha=alpha.astype(
            np.float32
        ),
        horizons_min=np.asarray(
            HORIZONS,
            dtype=np.int64,
        ),
    )

    np.savez_compressed(
        out
        / "15B_F_test_predictions.npz",
        y_true=np.asarray(
            test["y_true"],
            dtype=np.float32,
        ),
        ridge=np.asarray(
            test["ridge"],
            dtype=np.float32,
        ),
        raw_residual=np.asarray(
            test["raw_residual"],
            dtype=np.float32,
        ),
        stage2_truewind=final_test.astype(
            np.float32
        ),
        alpha=alpha.astype(
            np.float32
        ),
        horizons_min=np.asarray(
            HORIZONS,
            dtype=np.int64,
        ),
    )

    report = {
        "stage": "15B-F",
        "neural_retraining": False,
        "alpha_selection_split": "validation only",
        "alpha_cap": ALPHA_CAP,
        "alpha_unconstrained_validation": unconstrained,
        "alpha_frozen": alpha,
        "alpha_cap_nonbinding": bool(
            np.max(alpha) < ALPHA_CAP - 1e-7
        ),
        "validation_residual_corr": val_corr,
        "test_residual_corr_diagnostic": test_corr,
        "metrics_summary": summary.to_dict(
            orient="records"
        ),
        "stage15c_train_truewind_file": (
            out
            / "15B_F_train_stage2_oof_predictions.npz"
        ),
    }

    save_json(
        out
        / "15B_F_REPORT.json",
        report,
    )

    log("")
    log("[STAGE 5/5] Final report.")

    test_table = metrics.loc[
        metrics["split"] == "test"
    ].copy()

    log("")
    log("=" * 140)
    log("15B-F FINAL TEST — TRUE WIND")
    log("=" * 140)
    log(
        test_table.to_string(
            index=False
        )
    )

    log("")
    log("Frozen alpha + TEST residual correlation:")

    for j, h in enumerate(HORIZONS):
        log(
            f"  {h:2d} min | "
            f"alpha U/V="
            f"{alpha[j,0]:.6f}/"
            f"{alpha[j,1]:.6f} | "
            f"corr U/V="
            f"{test_corr[j,0]:+.4f}/"
            f"{test_corr[j,1]:+.4f}"
        )

    log("")
    log(
        "[NEXT] Stage15C must use:"
    )

    log(
        f"       {out / '15B_F_train_stage2_oof_predictions.npz'}"
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n[FATAL ERROR]", flush=True)
        traceback.print_exc()
        sys.exit(1)
