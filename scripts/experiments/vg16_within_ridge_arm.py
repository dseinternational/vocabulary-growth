#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Test VG16's joint within-child lag estimator on simulated data.

The two-step experiment in ``vg16_within_lag_bias.py`` does not reproduce joint
estimation of the child comprehension effect and lag coefficient. This arm fits
the PyMC graph to data generated at a stated coefficient. Compare its recovery
with the two-step results while checking sampling quality. Poor recovery can
support a joint-estimation concern but does not isolate a particular cause.
Good recovery at selected truths does not rule out problems elsewhere.

Usage::

    python scripts/experiments/vg16_within_ridge_arm.py truth-zero --output-dir /scratch/vg16-ridge
    python scripts/experiments/vg16_within_ridge_arm.py truth-plus --output-dir /scratch/vg16-ridge

``--config`` defaults to ``test``. Use a separate output root and check the
convergence diagnostics before interpreting recovery.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import os
from multiprocessing import freeze_support

import numpy as np
import pandas as pd

#: ``real`` fits the observed data; the other arms simulate at stated truths.
ARMS = {"truth-zero": 0.0, "truth-plus": 0.203, "real": None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument(
        "--fit-baseline",
        default="within",
        choices=["within", "population"],
        help=(
            "Baseline the FITTED model uses (default: within). Use 'population' with "
            "truth-zero to null-calibrate the headline estimator on the real machinery."
        ),
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--config", default="test")
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument(
        "--generate-under",
        default="within",
        choices=["within", "population"],
        help="Baseline the data are generated under (default: within, matching the fitted one).",
    )
    args = ap.parse_args()

    import dse_research_utils.environment.setup as setup

    from vocab_growth import environment as env
    from vocab_growth.models import definitions as D
    from vocab_growth.models.common import run_fit_pipeline
    from vocab_growth.models.common_bivariate_re import bivariate_re_stages
    from vocab_growth.recovery.simulate import build_model_data

    # Loaded by path: scripts/ is not a package.
    spec = importlib.util.spec_from_file_location(
        "vg16_bias", os.path.join(os.path.dirname(__file__), "vg16_within_lag_bias.py")
    )
    sim_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim_mod)

    beta_true = ARMS[args.arm]
    real_data = args.arm == "real"

    if not real_data:
        # --- simulate on the real design, from the fitted posterior ----------
        # The truth is read from the *model of record's* output root, not the arm's.
        truth, design = sim_mod.load_truth(env.output_root())
        rng = np.random.default_rng(args.seed)
        sim = sim_mod.simulate(rng, truth, design, beta_true, args.generate_under)

    # --- assemble the analysis frame the engine expects ----------------------
    # Only the outcome columns are simulated; age, child, study and the observed
    # /missing pattern are the real ones, so the wave structure under test is the
    # actual one.
    frame = None if real_data else pd.DataFrame(
        {
            "age": design["age"],
            "subject_code": design["subj"],
            "study_code": design["study"],
            "understood": np.where(design["umask"], sim["y_u"], np.nan),
            "spoken": np.where(design["smask"], sim["y_s"], np.nan),
        }
    )
    if frame is not None:
        frame["subject_id"] = frame["subject_code"].astype(str)
        frame["study"] = frame["study_code"].astype(str)
        frame["subject_key"] = frame["subject_id"]

    # --- VG16 with the within-child baseline ---------------------------------
    defn = dataclasses.replace(
        D.VG16,
        lag_baseline=args.fit_baseline,
        config_name=f"vg16-{args.fit_baseline}-ridge-{args.arm}",
        model_id="VG16",
    )

    env.set_output_root(args.output_dir)
    setup.init_script()
    print(
        f"[ridge_arm] arm={args.arm} beta_true={beta_true} "
        f"generated_under={'real data' if real_data else args.generate_under} "
        f"fit_baseline={defn.lag_baseline} config={args.config} "
        f"rows={'real' if real_data else len(frame)}"
    )

    stages = bivariate_re_stages(defn)
    assert stages[0][0] == "Prepare data", stages[0][0]

    def inject(ctx):
        ctx.set_model_data(build_model_data(frame, defn), frame.copy())
        print(
            f"[ridge_arm] injected simulated frame: {len(frame)} rows, "
            f"{int(design['umask'].sum())} understood, {int(design['smask'].sum())} spoken, "
            f"beta_true={beta_true}"
        )

    # The real arm keeps the engine's own data-preparation stage.
    if not real_data:
        stages[0] = ("Prepare data", inject)
    run_fit_pipeline(args.config, defn, stages=stages)
    print(f"[ridge_arm] done. beta_true was {beta_true}; read beta_lag from diagnostics.csv")
    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
