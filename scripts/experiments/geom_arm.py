# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fit one VG12 geometry arm at `test` config, into a throwaway output root.

Each arm patches `definitions.VG12` *before* the model module is imported (the
module binds the definition at import time) and gives it a distinct config_name
so the arms land in separate output directories and cannot collide with any model
of record.

The `eta` widening is held at its current value (0.5) in every arm except the
`eta` arm, so the two geometry changes are not confounded with the calibration
change. See notes/202608050900-td-hierarchical-geometry.md §7.

Usage: geom_arm.py {baseline,eta,centred,partition} --output-dir DIR
"""
import argparse
import dataclasses
import importlib
from multiprocessing import freeze_support

import dse_research_utils.environment.setup as setup

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=["baseline", "eta", "centred", "partition"])
    p.add_argument("--output-dir", required=True)
    freeze_support()
    a = p.parse_args()

    from vocab_growth import environment as env
    from vocab_growth.models import definitions as D

    # Start from the current registered model with the eta widening reverted, so
    # `baseline` really is the model that produced the rep fit under study.
    base = dataclasses.replace(D.VG12, eta_sigma=0.5)

    if a.arm == "baseline":
        defn = base
    elif a.arm == "eta":
        defn = dataclasses.replace(base, eta_sigma=1.0)
    elif a.arm == "centred":
        defn = dataclasses.replace(base, centred_study_re=True)
    else:
        defn = dataclasses.replace(
            base, subject_variance_partition=D._TD_UNDERSTOOD_VARIANCE_PARTITION
        )

    defn = dataclasses.replace(defn, config_name=f"geom-{a.arm}")
    D.VG12 = defn
    D.MODEL_REGISTRY["VG12"] = defn

    env.set_output_root(a.output_dir)
    setup.init_script()
    print(f"[geom_arm] VG12 arm={a.arm} config_name={defn.config_name}")
    print(
        f"[geom_arm]   eta_sigma={defn.eta_sigma} "
        f"centred_study_re={defn.centred_study_re} "
        f"partition={'yes' if defn.subject_variance_partition else 'no'}"
    )

    m = importlib.import_module("vocab_growth.models.model_vg12")
    # The module does `from ...definitions import VG12` at import time. Patching
    # D.VG12 first covers a fresh import; rebinding covers the case where
    # something already imported it.
    m.VG12 = defn
    m.fit("test")
