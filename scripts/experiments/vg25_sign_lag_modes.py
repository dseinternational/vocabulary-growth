#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Probe VG25 sign-lag designs for multimodality with many short chains.

VG25's first rep fit (2026-09-15) had the lag in the cross-tab compositions under
the within-child baseline and was bimodal: four chains at ``beta_sign_lag`` +0.69,
two at -0.50, never mixing. This builds the joint engine's real graph on the real
frame for one of four designs -- the two baselines crossed with the two scopes --
and samples it with nutpie using many chains (12 by default, where a rep fit has
6), keeping only scalar parameters, then writes each chain's mean for the
parameters that split the rep fit. Two clusters of chain means is two modes.

It does not validate a stored trace, because it reads none: every number comes
from the fit it runs, against the checkout it runs in. It writes only under
``--out``, which should not be the project's output root.

Cited by ``notes/202609151930-vg25-lag-out-of-the-cells.md``: two modes for the
within-child baseline in the compositions, one for each other design.

Run: ``python scripts/experiments/vg25_sign_lag_modes.py --design within-cells --out <scratch>``
(designs: within-cells, population-cells, within-marginal, population-marginal).
The four took 400-470 s of sampling each at 12 chains, 1,500 tune and 500 draws on
the fitting workstation, two at a time.
"""

import argparse
import dataclasses
import json
import os
import sys
import time
import types

import dse_research_utils.statistics.models.pymc_utils as pymc_utils
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import pymc as pm

from vocab_growth.models.common import ModelFitContext
from vocab_growth.models.common_joint_modality import joint_stages
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.sensitivity.registry import build_variant

WATCH = ("beta_sign_lag", "rho_u_sign", "rho_sign_q", "rho_uq", "tau_subj_sign", "tau_subj_q", "tau_subj_u", "psi", "kappa_young_s")


def definition_for(design: str):
    base = MODEL_REGISTRY["vg25"]
    if design == "within-cells":
        return base
    if design == "population-cells":
        (d,) = build_variant("vg25", "sign-lag-population")
        return d
    if design == "within-marginal":
        (d,) = build_variant("vg25", "sign-lag-marginal-only")
        return d
    if design == "population-marginal":
        (d,) = build_variant("vg25", "sign-lag-population")
        return dataclasses.replace(d, sign_lag_in_cells=False)
    raise SystemExit(f"unknown design {design}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--chains", type=int, default=12)
    ap.add_argument("--tune", type=int, default=1500)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pymc_utils.model_to_graphviz = lambda model: types.SimpleNamespace(render=lambda *a, **k: None)
    definition = definition_for(args.design)
    print("design", args.design, "baseline", definition.sign_lag_baseline, "in_cells", definition.sign_lag_in_cells, flush=True)

    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name=f"VG25-probe-{args.design}",
            config_name="probe",
            output_root_dir=args.out,
            ci_prob=0.89,
            interval_kind="eti",
        ),
        sampling=sampling.SamplingConfiguration(
            draws=args.draws, tune=args.tune, chains=args.chains, cores=args.chains,
            target_accept=0.95, random_seed=args.seed,
        ),
    )
    os.makedirs(context.reporting.output_dir, exist_ok=True)
    stages = joint_stages(definition)
    for name, fn in stages[:3]:
        t0 = time.time()
        fn(context)
        print(f"stage {name}: {time.time() - t0:.0f}s", flush=True)

    model = context.model
    scalars = [rv.name for rv in model.free_RVs if rv.ndim == 0]
    scalars += [d.name for d in model.deterministics if d.name in WATCH and d.ndim == 0 and d.name not in scalars]
    t0 = time.time()
    with model:
        trace = pm.sample(
            args.draws, tune=args.tune, chains=args.chains, cores=args.chains,
            target_accept=0.95, nuts_sampler="nutpie", progressbar=False,
            random_seed=args.seed, var_names=scalars, return_inferencedata=True,
        )
    print(f"sampling: {time.time() - t0:.0f}s", flush=True)

    post = trace.posterior
    summary = {"design": args.design, "chains": args.chains, "tune": args.tune, "draws": args.draws, "per_chain": {}}
    for name in WATCH:
        if name in post:
            v = post[name].values
            v = v.reshape(v.shape[0], v.shape[1], -1)[:, :, 0]
            summary["per_chain"][name] = [round(float(x), 4) for x in v.mean(axis=1)]
    stats = trace.sample_stats
    summary["sample_stats_keys"] = sorted(stats.data_vars)
    for key in ("logp", "lp"):
        if key in stats:
            summary["per_chain"]["logp"] = [round(float(x), 2) for x in stats[key].values.mean(axis=1)]
    if "diverging" in stats:
        summary["divergences"] = [int(x) for x in stats["diverging"].values.sum(axis=1)]
    print(json.dumps(summary, indent=1), flush=True)
    with open(os.path.join(args.out, f"summary-{args.design}.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    trace.to_netcdf(os.path.join(args.out, f"trace-{args.design}.nc"))


if __name__ == "__main__":
    sys.exit(main())
