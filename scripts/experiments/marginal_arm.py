# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fit one VG12 (or VG11) singleton-marginalisation arm into a throwaway root.

The fifth arm of the geometry table in
notes/202608050900-td-hierarchical-geometry.md §9, and the equivalence and
node-sensitivity checks that
notes/202608231410-td-geometry-remaining-levers.md §3 makes obligations:

    marginal_arm.py explicit               -- the model of record's graph
    marginal_arm.py marginal               -- singleton child effects integrated out
    marginal_arm.py marginal --nodes 40    -- the node-count sensitivity of that arm

`explicit` and `marginal` differ in the sampled space alone: the marginalisation
is exact, so `tau_subject`, `kappa` and the trajectory must agree within Monte
Carlo error. They will not agree bit for bit -- the sampled space has different
dimensions -- which is why the comparison is a posterior comparison and not a
diff.

Each arm patches the definition *before* the model module is imported (the
module binds it at import time) and gives it a distinct config_name, so the arms
land in separate output directories and cannot collide with a model of record.

Usage: marginal_arm.py {explicit,marginal} [--model vg12] [--nodes 20]
                       [--config test] --output-dir DIR
"""
import argparse
import dataclasses
import importlib
from multiprocessing import freeze_support

import dse_research_utils.environment.setup as setup

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["explicit", "marginal"])
    parser.add_argument("--model", default="vg12", choices=["vg11", "vg12"])
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument("--config", default="test")
    parser.add_argument("--output-dir", required=True)
    freeze_support()
    args = parser.parse_args()

    from vocab_growth import environment as env
    from vocab_growth.models import definitions as D

    model_id = args.model.upper()
    base = getattr(D, model_id)

    if args.arm == "explicit":
        definition = dataclasses.replace(base, config_name=f"marg-explicit-{args.model}")
    else:
        definition = D._as_definition_subclass(
            base,
            D.UnivariateMarginalisedREModelDefinition,
            singleton_marginalisation=D.SingletonMarginalisationParams(
                n_nodes=args.nodes
            ),
            config_name=f"marg-{args.nodes}-{args.model}",
        )

    D.validate_model_definition(definition)
    setattr(D, model_id, definition)
    D.MODEL_REGISTRY[model_id.lower()] = definition

    env.set_output_root(args.output_dir)
    setup.init_script()
    print(f"[marginal_arm] {model_id} arm={args.arm} config_name={definition.config_name}")
    marginalisation = getattr(definition, "singleton_marginalisation", None)
    print(
        f"[marginal_arm]   singleton_marginalisation="
        f"{'none' if marginalisation is None else marginalisation.n_nodes}"
        f" sampling_config={args.config}"
    )

    module = importlib.import_module(f"vocab_growth.models.model_{args.model}")
    # The module does `from ...definitions import VGnn` at import time. Patching
    # D first covers a fresh import; rebinding covers an earlier one.
    setattr(module, model_id, definition)
    module.fit(args.config)
