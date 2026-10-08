# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Refit VG12 with ``eta_sigma=0.5`` in a separate output root.

Compare diagnostics with an otherwise matching fit to assess sensitivity to this
prior scale. A difference in divergence counts alone does not identify its cause.
"""
import dataclasses
import importlib
from multiprocessing import freeze_support

import dse_research_utils.environment.setup as setup

if __name__ == "__main__":
    freeze_support()
    from vocab_growth import environment as env
    from vocab_growth.models import definitions as D

    defn = dataclasses.replace(D.VG12, eta_sigma=0.5, config_name="eta050-isolate")
    D.VG12 = defn
    D.MODEL_REGISTRY["VG12"] = defn
    env.set_output_root("/scratch/vg-geom-output")
    setup.init_script()
    print(f"[isolate] VG12 eta_sigma={defn.eta_sigma} "
          f"centred={defn.centred_study_re} "
          f"partition={'yes' if defn.subject_variance_partition else 'no'}")
    m = importlib.import_module("vocab_growth.models.model_vg12")
    m.VG12 = defn
    m.fit("rep")
