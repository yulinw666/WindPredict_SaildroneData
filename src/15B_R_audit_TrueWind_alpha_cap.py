# -*- coding: utf-8 -*-
"""
15B_R_audit_TrueWind_alpha_cap.py

Validation-only audit for Stage15B residual shrinkage coefficients.

Purpose
-------
Stage15B used alpha_max = 0.75 and several alpha values hit that boundary.
This script checks whether the boundary is active.

It:
    - loads ONLY Stage15B validation predictions
    - does NOT load test
    - does NOT retrain the neural network
    - computes unconstrained analytical alpha for each horizon/component
    - evaluates several alpha caps on validation

Files expected in Stage15B directory:
    15B_validation_predictions.npz

Required keys:
    y_true
    ridge
    raw_residual
    stage2_truewind
    alpha

Recommended:
python "D:\\project\\WindPredict_SaildroneData\\src\\15B_R_audit_TrueWind_alpha_cap.py" --stage15b-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15B_1min_TrueWind_Residual_v0_2" --output-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\15B_R_TrueWind_AlphaCapAudit_v0_1"
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
    / "15B_R_TrueWind_AlphaCapAudit_v0_1"
)

HORIZONS = [1, 2, 3, 5, 10]

CAPS = [
    0.75,
    0.85,
    1.00,
    1.25,
    1.50,
    2.00,
    5.00,
]

EPS = 1e-12


def log(msg=""):
    print(msg, flush=True)


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


def circ_diff(a, b):
    return (
        (
            np.asarray(a)
            - np.asarray(b)
            + 180.0
        )
        % 360.0
        - 180.0
    )


def alpha_unconstrained(y, ridge, raw):
    e = (
        np.asarray(y, dtype=np.float64)
        - np.asarray(ridge, dtype=np.float64)
    )
    r = np.asarray(raw, dtype=np.float64)

    a = np.zeros(
        (len(HORIZONS), 2),
        dtype=np.float64,
    )

    for h in range(len(HORIZONS)):
        for c in range(2):
            rr = r[:, h, c]
            ee = e[:, h, c]

            denom = float(
                np.dot(rr, rr)
            )

            if denom <= EPS:
                a[h, c] = 0.0
            else:
                a[h, c] = float(
                    np.dot(ee, rr)
                    / denom
                )

    return a


def apply_alpha(ridge, raw, alpha):
    return (
        np.asarray(ridge, dtype=np.float64)
        + np.asarray(raw, dtype=np.float64)
        * np.asarray(alpha, dtype=np.float64)[None, :, :]
    )


def metrics(y, p):
    yt = np.asarray(y, dtype=np.float64)
    yp = np.asarray(p, dtype=np.float64)

    rows = []

    for j, h in enumerate(HORIZONS):
        t = yt[:, j, :]
        q = yp[:, j, :]
        e = q - t

        ws_t = np.linalg.norm(t, axis=1)
        ws_q = np.linalg.norm(q, axis=1)

        wd_err = circ_diff(
            wd_from_uv(q),
            wd_from_uv(t),
        )

        rows.append(
            {
                "horizon_min": h,
                "U_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            e[:, 0] ** 2
                        )
                    )
                ),
                "V_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            e[:, 1] ** 2
                        )
                    )
                ),
                "vector_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                e ** 2,
                                axis=1,
                            )
                        )
                    )
                ),
                "WS_RMSE_mps": float(
                    np.sqrt(
                        np.mean(
                            (ws_q - ws_t) ** 2
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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--stage15b-dir",
        type=Path,
        default=DEFAULT_STAGE15B_DIR,
    )

    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = ap.parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        args.stage15b_dir
        / "15B_validation_predictions.npz"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    # IMPORTANT: validation only. Test is never opened.
    with np.load(
        path,
        allow_pickle=False,
    ) as z:
        y = np.asarray(
            z["y_true"],
            dtype=np.float64,
        )
        ridge = np.asarray(
            z["ridge"],
            dtype=np.float64,
        )
        raw = np.asarray(
            z["raw_residual"],
            dtype=np.float64,
        )
        frozen_alpha = np.asarray(
            z["alpha"],
            dtype=np.float64,
        )

    log("=" * 120)
    log("15B-R — VALIDATION-ONLY TRUE-WIND ALPHA CAP AUDIT")
    log("=" * 120)
    log("TEST is NOT loaded.")
    log(f"Validation samples = {len(y):,}")

    unconstrained = alpha_unconstrained(
        y,
        ridge,
        raw,
    )

    alpha_rows = []

    log("")
    log("[UNCONSTRAINED ANALYTICAL ALPHA]")

    for j, h in enumerate(HORIZONS):
        log(
            f"  {h:2d} min | "
            f"U={unconstrained[j,0]:.8f} | "
            f"V={unconstrained[j,1]:.8f} | "
            f"old capped U/V="
            f"{frozen_alpha[j,0]:.8f}/"
            f"{frozen_alpha[j,1]:.8f}"
        )

        alpha_rows.append(
            {
                "horizon_min": h,
                "component": "U",
                "alpha_unconstrained": unconstrained[j, 0],
                "alpha_old": frozen_alpha[j, 0],
            }
        )

        alpha_rows.append(
            {
                "horizon_min": h,
                "component": "V",
                "alpha_unconstrained": unconstrained[j, 1],
                "alpha_old": frozen_alpha[j, 1],
            }
        )

    pd.DataFrame(alpha_rows).to_csv(
        args.output_dir
        / "15B_R_unconstrained_alpha.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Ridge baseline.
    ridge_m = metrics(
        y,
        ridge,
    )

    summary_rows = []

    summary_rows.append(
        {
            "mode": "Ridge",
            "cap": 0.0,
            "mean_U_RMSE_mps": float(
                ridge_m["U_RMSE_mps"].mean()
            ),
            "mean_V_RMSE_mps": float(
                ridge_m["V_RMSE_mps"].mean()
            ),
            "mean_vector_RMSE_mps": float(
                ridge_m["vector_RMSE_mps"].mean()
            ),
            "mean_WS_RMSE_mps": float(
                ridge_m["WS_RMSE_mps"].mean()
            ),
            "mean_WD_RMSE_deg": float(
                ridge_m["WD_RMSE_deg"].mean()
            ),
        }
    )

    log("")
    log("[CAP SWEEP — VALIDATION ONLY]")

    for cap in CAPS:
        a = np.clip(
            unconstrained,
            0.0,
            float(cap),
        )

        pred = apply_alpha(
            ridge,
            raw,
            a,
        )

        m = metrics(
            y,
            pred,
        )

        row = {
            "mode": "capped_alpha",
            "cap": float(cap),
            "mean_U_RMSE_mps": float(
                m["U_RMSE_mps"].mean()
            ),
            "mean_V_RMSE_mps": float(
                m["V_RMSE_mps"].mean()
            ),
            "mean_vector_RMSE_mps": float(
                m["vector_RMSE_mps"].mean()
            ),
            "mean_WS_RMSE_mps": float(
                m["WS_RMSE_mps"].mean()
            ),
            "mean_WD_RMSE_deg": float(
                m["WD_RMSE_deg"].mean()
            ),
        }

        summary_rows.append(row)

        log(
            f"  cap={cap:4.2f} | "
            f"U={row['mean_U_RMSE_mps']:.6f} | "
            f"V={row['mean_V_RMSE_mps']:.6f} | "
            f"vector={row['mean_vector_RMSE_mps']:.6f} | "
            f"WS={row['mean_WS_RMSE_mps']:.6f} | "
            f"WD={row['mean_WD_RMSE_deg']:.6f}"
        )

    # Fully unconstrained positive analytical alpha.
    a_pos = np.maximum(
        unconstrained,
        0.0,
    )

    pred = apply_alpha(
        ridge,
        raw,
        a_pos,
    )

    m = metrics(
        y,
        pred,
    )

    uncon_row = {
        "mode": "positive_unconstrained",
        "cap": np.nan,
        "mean_U_RMSE_mps": float(
            m["U_RMSE_mps"].mean()
        ),
        "mean_V_RMSE_mps": float(
            m["V_RMSE_mps"].mean()
        ),
        "mean_vector_RMSE_mps": float(
            m["vector_RMSE_mps"].mean()
        ),
        "mean_WS_RMSE_mps": float(
            m["WS_RMSE_mps"].mean()
        ),
        "mean_WD_RMSE_deg": float(
            m["WD_RMSE_deg"].mean()
        ),
    }

    summary_rows.append(
        uncon_row
    )

    log(
        "  unconstrained | "
        f"U={uncon_row['mean_U_RMSE_mps']:.6f} | "
        f"V={uncon_row['mean_V_RMSE_mps']:.6f} | "
        f"vector={uncon_row['mean_vector_RMSE_mps']:.6f} | "
        f"WS={uncon_row['mean_WS_RMSE_mps']:.6f} | "
        f"WD={uncon_row['mean_WD_RMSE_deg']:.6f}"
    )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        args.output_dir
        / "15B_R_alpha_cap_validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Best among finite caps by validation mean vector RMSE.
    finite = summary.loc[
        summary["mode"]
        == "capped_alpha"
    ].copy()

    best = finite.loc[
        finite["mean_vector_RMSE_mps"].idxmin()
    ]

    report = {
        "test_loaded": False,
        "best_cap_by_validation_mean_vector_RMSE": float(
            best["cap"]
        ),
        "best_validation_mean_vector_RMSE": float(
            best["mean_vector_RMSE_mps"]
        ),
        "unconstrained_alpha": unconstrained.tolist(),
        "old_alpha": frozen_alpha.tolist(),
    }

    with (
        args.output_dir
        / "15B_R_REPORT.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    log("")
    log("=" * 120)
    log(
        f"BEST FINITE CAP BY VALIDATION VECTOR RMSE = "
        f"{float(best['cap']):.2f}"
    )
    log("=" * 120)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n[FATAL ERROR]", flush=True)
        traceback.print_exc()
        sys.exit(1)
