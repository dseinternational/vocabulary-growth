#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Is VG16's within-child cross-lag anomaly a joint-estimation ridge? Fit and see.

`vg16_within_lag_bias.py` established that the within-child baseline's strongly
negative `dev`-tier estimate (`beta` ~ -0.60) is **not** the short-T /
errors-in-variables artefact the VG16 report attributed it to: a two-step
estimator of the same quantity, on simulated data with a known `beta_lag`, is
unbiased at every truth tested, and an oracle-intercept variant rules out
estimation error in the child's understood intercept specifically.

That two-step design severs one thing the real model has: in VG16 the child
understood intercept `delta_subj_u` is estimated **jointly** with `beta_lag`, so
the spoken likelihood feeds back onto the intercept through `x_lag`. A ridge
between the two is the leading remaining explanation, and this arm is its direct
test — it fits the *actual* PyMC model, with `lag_baseline="within"`, to data
simulated at a known `beta_lag`.

Reading the result:

* `beta_lag` recovered near truth -> the ridge hypothesis is refuted too, and the
  -0.60 belongs to the real data or to `dev`-tier non-convergence.
* `beta_lag` strongly negative where the two-step estimator was unbiased -> the
  ridge is confirmed, and a decoupled estimator is the remedy.

Run the `truth-zero` arm first: a large negative estimate when the truth is
exactly zero is the cleanest possible demonstration.

Usage::

    python scripts/experiments/vg16_within_ridge_arm.py truth-zero --output-dir /scratch/vg16-ridge
    python scripts/experiments/vg16_within_ridge_arm.py truth-plus --output-dir /scratch/vg16-ridge

`--config` defaults to ``test``, not ``dev``: the -0.60 is a `dev` figure and the
project holds that `dev` under-converges the hierarchical models, so a `dev` arm
could not separate the ridge from non-convergence. Writes to its own output root
so no model of record can be touched.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import os
from multiprocessing import freeze_support

import numpy as np
import pandas as pd

#: Truth values the arms simulate at.
ARMS = {"truth-zero": 0.0, "truth-plus": 0.203}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=sorted(ARMS))
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

    # --- simulate on the real design, from the fitted posterior --------------
    # The truth is read from the *model of record's* output root, not the arm's.
    truth, design = sim_mod.load_truth(env.output_root())
    rng = np.random.default_rng(args.seed)
    sim = sim_mod.simulate(rng, truth, design, beta_true, args.generate_under)

    # --- assemble the analysis frame the engine expects ----------------------
    # Only the outcome columns are simulated; age, child, study and the observed
    # /missing pattern are the real ones, so the wave structure under test is the
    # actual one.
    frame = pd.DataFrame(
        {
            "age": design["age"],
            "subject_code": design["subj"],
            "study_code": design["study"],
            "understood": np.where(design["umask"], sim["y_u"], np.nan),
            "spoken": np.where(design["smask"], sim["y_s"], np.nan),
        }
    )
    frame["subject_id"] = frame["subject_code"].astype(str)
    frame["study"] = frame["study_code"].astype(str)
    frame["subject_key"] = frame["subject_id"]

    # --- VG16 with the within-child baseline ---------------------------------
    defn = dataclasses.replace(
        D.VG16,
        lag_baseline="within",
        config_name=f"vg16-within-ridge-{args.arm}",
        model_id="VG16",
    )

    env.set_output_root(args.output_dir)
    setup.init_script()
    print(
        f"[ridge_arm] arm={args.arm} beta_true={beta_true} "
        f"generated_under={args.generate_under} lag_baseline={defn.lag_baseline} "
        f"config={args.config} rows={len(frame)}"
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

    stages[0] = ("Prepare data", inject)
    run_fit_pipeline(args.config, defn, stages=stages)
    print(f"[ridge_arm] done. beta_true was {beta_true}; read beta_lag from diagnostics.csv")
    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
