# -*- coding: utf-8 -*-
r"""
12H_make_PhysicsCompact_NieData_v3_RidgeWind_VesselResidual.py

Purpose
-------
Generate a scientifically auditable Physics-Compact-NieData v3 code pair from
the project's already tested Stage-12C (Ridge-anchor) and Stage-12D v2 scripts.

v3 architecture
---------------
Wind branch:
    W_hat_z = W_ridge_z
    (frozen; no trainable wind residual)

Vessel branch:
    V_hat_z = V_ridge_z + sigmoid(g_V) * DeltaV_z

Apparent wind:
    A_hat = W_hat - V_hat

Loss:
    L = 0.5*MSE_z(W) + 0.5*MSE_z(V)
        + 2.0*L_AW
        + 0.01*mean((g_V*DeltaV_z)^2)

Because W_hat is frozen to Ridge, the wind-MSE term is constant with respect
to the neural parameters and contributes no wind-branch gradient.

Why generate by patching?
-------------------------
The local 12C/12D scripts already contain the project's tested:
- data loaders
- scalers
- Ridge implementation
- LOMO folds
- metrics
- five-seed reporting
- fixed-epoch final fitting
- output/audit machinery

This generator changes only the architecture/anchor logic needed for v3 and
keeps the rest of that tested pipeline intact.

Scientific provenance
---------------------
This v3 architecture is being defined AFTER prior Tropical Atlantic analyses
(Stage 12F/12G). Therefore the generated Stage-12I script explicitly records
that Tropical Atlantic is a RE-EVALUATION, not a pristine first test.

The generated Stage-12H LOMO script itself reads only:
    Antarctic, Atlantic, West Coast

and can be used to assess whether the v3 structural change is supported by
development-only evidence.

This script intentionally DOES NOT provide an option to relabel prior Tropical
Atlantic access as a first test.

Expected local source files
---------------------------
src\12C_train_PhysicsCompact_NieData_LOMO.py
src\12D_final_PhysicsCompact_NieData_v2_TropicalAtlantic.py

Generated files
---------------
src\12H_train_PhysicsCompact_NieData_v3_RidgeWind_VesselResidual_LOMO.py
src\12I_reevaluate_PhysicsCompact_NieData_v3_TropicalAtlantic.py

Recommended usage for the point-sampled Stage-12G dataset
----------------------------------------------------------
1) Generate v3 scripts:
python "D:\project\WindPredict_SaildroneData\src\12H_make_PhysicsCompact_NieData_v3_RidgeWind_VesselResidual.py"

2) Development-only LOMO on the point-sampled dataset:
python "D:\project\WindPredict_SaildroneData\src\12H_train_PhysicsCompact_NieData_v3_RidgeWind_VesselResidual_LOMO.py" ^
  --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" ^
  --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12H_PhysicsCompact_NieData_v3_RidgeWind_VesselResidual_LOMO_point_v0_1"

3) Final/re-evaluation using the epoch plan derived from Stage-12H:
python "D:\project\WindPredict_SaildroneData\src\12I_reevaluate_PhysicsCompact_NieData_v3_TropicalAtlantic.py" ^
  --dataset-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12G_Nie_point_sampled_benchmark_v0_1\dataset" ^
  --c2-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12H_PhysicsCompact_NieData_v3_RidgeWind_VesselResidual_LOMO_point_v0_1" ^
  --output-dir "D:\project\WindPredict_SaildroneData\data\forecasting\12I_PhysicsCompact_NieData_v3_TropicalAtlantic_REEVAL_point_v0_1"

Notes
-----
- The generated v3 point-wind prediction equals Ridge exactly by design.
- Therefore v3 cannot have a lower U/V/WS/WD error than Ridge unless a
  trainable wind branch is reintroduced.
- The purpose of v3 is to test the primary-paper structural principle:
  preserve a strong linear wind forecast and allocate nonlinear capacity to
  vessel motion.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_ROOT = Path(r"D:\project\WindPredict_SaildroneData")


C_MODEL_FUNCTION = r