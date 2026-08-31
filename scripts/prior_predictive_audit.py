# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regenerate prior-predictive checks for the §6 prior-predictive audit (issue #89).

Builds each model and runs its ``prior_predictive_checks`` stage **only** — i.e.
``sample_prior_predictive`` with no posterior sampling — so the prior-predictive
plots (``prior_samples_*.png``, ``prior_predictive_checks.png``,
``prior_predictions.png`` and the analytic prior-distribution PNGs) are
regenerated cheaply into each model's output dir for review.

The engine and the calling convention come from
:mod:`vocab_growth.models.catalogue`, not from a table maintained here. Until
issue #273 they were maintained here, and the table had gone stale: VG16 and
VG19-VG23 all fit on ``common_bivariate_re`` while a hard-coded set routed them
through the plain ``common_bivariate``, so an audit of those six models built a
graph without the cross-lag, child-slope, correlated-effect or factor structure
that distinguishes them — and still produced plots, which is how a mismatch
comes to be mistaken for a valid prior check. The same table dropped the
definition argument for every random-effect model, discarding
:mod:`vocab_growth.models.prior_child_checks`'s unseen-child figures, which are
the ones a child-effect model's prior audit exists to look at (issue #233).

Usage:
    python scripts/prior_predictive_audit.py [models...]   # default: family reps
"""

import argparse
import os
import sys
from multiprocessing import freeze_support

import dse_research_utils.environment.setup as setup
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling

from vocab_growth import environment as env
from vocab_growth.models.catalogue import get as catalogue_get
from vocab_growth.models.common import ModelFitContext
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.reporting import console, heading

# §6 regeneration set (family representatives + the reporting models).
_DEFAULT = ["vg10", "vg11", "vg12", "vg13", "vg14", "vg15"]


def _context(definition) -> ModelFitContext:
    ctx = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name=definition.config_name,
            output_root_dir=env.output_root(),
            ci_prob=0.89,
            interval_kind="eti",
        ),
        sampling=sampling.get_sampling_configuration("dev"),
    )
    os.makedirs(ctx.reporting.output_dir, exist_ok=True)
    return ctx


def audit(model_key: str) -> str:
    """Build the model and run only its prior-predictive stage; return its dir."""
    model = catalogue_get(model_key)
    definition = model.definition
    engine = model.engine
    ctx = _context(definition)

    engine.resolve("prepare")(ctx, definition)
    engine.resolve("priors")(ctx, definition)
    engine.resolve("build")(ctx, definition)

    prior_checks = engine.resolve("prior_checks")
    call = engine.prior_checks_call
    if call == "outcome":
        prior_checks(ctx, definition.outcome.value, definition.outcome_label)
    elif call == "definition":
        prior_checks(ctx, definition)
    elif call == "context":
        prior_checks(ctx)
    else:
        raise ValueError(
            f"engine {engine.name!r} declares unknown prior_checks_call {call!r}"
        )
    return ctx.reporting.output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=None, help="Model keys (default: family reps).")
    freeze_support()
    setup.init_script()
    args = parser.parse_args()
    models = [m.lower() for m in (args.models or _DEFAULT)]

    for key in models:
        if key not in MODEL_REGISTRY:
            console.print(f"[bold red]Unknown model: {key}[/bold red]")
            sys.exit(1)
        heading(f"Prior-predictive audit: {key.upper()}")
        out = audit(key)
        console.print(f"[dim]Prior-predictive plots regenerated in {out}[/dim]")
