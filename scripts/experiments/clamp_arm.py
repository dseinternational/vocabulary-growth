# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fit one VG01 arm at `test` config to test the clamp hypothesis for `eta`.

`eta` presses its prior in VG01, VG03, VG11 and VG12 — exactly the univariate
models, which are exactly the ones the 2026-08-04 `clamp_mean_above_hi_anchor`
fix never reached, because the field lives on the joint definition classes only.

The hypothesis: `eta` presses because the parametric mean extrapolates past its
high anchor and the GP spends its amplitude correcting that rather than
describing developmental curvature. Precedent: applying the clamp to the joint
models lowered both GP amplitudes with their contraction rising.

VG01 is the test case — its extrapolation region is the largest of the four
(84-115 months, ~29% of the GP domain against ~18% for the TD models).

Adding the field to `UnivariateModelDefinition` would invalidate every VG01-VG04,
VG11 and VG12 fit, so the arm uses a throwaway subclass instead. The engines read
the flag through `getattr`, so this needs no production change to run.

Usage: clamp_arm.py {baseline,clamped} --output-dir DIR
"""
import argparse
import dataclasses
import importlib
from multiprocessing import freeze_support

import dse_research_utils.environment.setup as setup

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=["baseline", "clamped"])
    p.add_argument("--output-dir", required=True)
    freeze_support()
    a = p.parse_args()

    from vocab_growth import environment as env
    from vocab_growth.models import definitions as D

    @dataclasses.dataclass
    class ClampableUnivariate(D.UnivariateModelDefinition):
        clamp_mean_above_hi_anchor: bool = False

    base_fields = {f.name: getattr(D.VG01, f.name) for f in dataclasses.fields(D.VG01)}
    defn = ClampableUnivariate(
        **base_fields,
        clamp_mean_above_hi_anchor=(a.arm == "clamped"),
    )
    defn = dataclasses.replace(defn, config_name=f"clamp-{a.arm}")

    D.VG01 = defn
    D.MODEL_REGISTRY["VG01"] = defn

    env.set_output_root(a.output_dir)
    setup.init_script()
    print(
        f"[clamp_arm] VG01 arm={a.arm} "
        f"clamp={defn.clamp_mean_above_hi_anchor} "
        f"slope_anchors={defn.slope_anchors} gp_domain={defn.gp_domain_months}"
    )
    m = importlib.import_module("vocab_growth.models.model_vg01")
    m.VG01 = defn
    m.fit("test")
