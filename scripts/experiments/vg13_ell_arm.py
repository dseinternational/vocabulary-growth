# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fit VG13 with its baseline or a shorter GP length-scale prior.

Compare the registered 6-18 month length-scale range with 2-8 months over the
8-18 month data window. A long length scale, after removal of constant and linear
components, can leave little curvature inside a short window. It does not make
all curvature mathematically impossible or establish non-identification by itself.

Predictive comparisons assess whether the shorter-range arm helps on these data.
No improvement would not prove that the underlying vocabulary trajectory has no
curvature. This is an experiment, separate from the models of record.

Usage::

    python scripts/experiments/vg13_ell_arm.py {baseline,rescaled} --output-dir DIR
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
