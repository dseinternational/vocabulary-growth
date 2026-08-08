"""Refit VG12 at rep with eta_sigma reverted to 0.5, keeping centring + partition.

Isolates whether the 2026-08-05 eta widening is responsible for VG12's
divergences rising from 2 to 29. Writes to a throwaway output root so the model
of record is untouched.
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
