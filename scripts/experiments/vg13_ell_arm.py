# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fit one VG13 arm at `test` config to test whether its window hides curvature.

VG13's GP is unidentifiable by construction: `ell_months_range = (6, 18)` puts
the median length-scale at 12 months over a **10-month** data window (8-18), and
the per-draw anchor orthogonalises the GP against [1, z] — the very shape a
length-scale longer than the window can produce. So it can express almost
nothing, and the amplitude of nothing is unidentifiable.

That says the model *cannot answer* whether there is curvature in 8-18 months. It
does not say there is none. A GP that cannot express curvature looks inert either
way, so the inert-GP diagnostics are uninformative about the question.

The `rescaled` arm shortens the length-scale prior to (2, 8) months, short enough
to express structure inside the window. Comparing the two arms by ELPD answers
it: if the flexible GP earns predictive improvement there is curve the current
model is blind to; if it does not, there is not.

This matters beyond VG13's own fit — VG13 supplies the typically-developing side
of the matched-comprehension contrast, and 8-18 months is where typically
developing vocabulary acceleration begins.

Usage: vg13_ell_arm.py {baseline,rescaled} --output-dir DIR
"""
import argparse
import dataclasses
import importlib
from multiprocessing import freeze_support

import dse_research_utils.environment.setup as setup

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=["baseline", "rescaled"])
    p.add_argument("--output-dir", required=True)
    freeze_support()
    a = p.parse_args()

    from vocab_growth import environment as env
    from vocab_growth.models import definitions as D

    ell = (6, 18) if a.arm == "baseline" else (2, 8)
    defn = dataclasses.replace(
        D.VG13, ell_months_range=ell, config_name=f"ell-{a.arm}"
    )
    D.VG13 = defn
    D.MODEL_REGISTRY["VG13"] = defn

    env.set_output_root(a.output_dir)
    setup.init_script()
    print(
        f"[vg13_ell_arm] arm={a.arm} ell_months_range={defn.ell_months_range} "
        f"gp_domain={defn.gp_domain_months} (window "
        f"{defn.gp_domain_months[1] - defn.gp_domain_months[0]} months)"
    )
    m = importlib.import_module("vocab_growth.models.model_vg13")
    m.VG13 = defn
    m.fit("test")
