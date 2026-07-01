# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regenerate prior-predictive checks for the §6 prior-predictive audit (issue #89).

Builds each model and runs its ``prior_predictive_checks`` stage **only** — i.e.
``sample_prior_predictive`` with no posterior sampling — so the prior-predictive
plots (``prior_samples_*.png``, ``prior_predictive_checks.png``,
``prior_predictions.png`` and the analytic prior-distribution PNGs) are
regenerated cheaply into each model's output dir for review.

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
from vocab_growth.models import common
from vocab_growth.models import common_bivariate as cb
from vocab_growth.models import common_bivariate_re as cbr
from vocab_growth.models import common_joint_modality as cj
from vocab_growth.models import common_trivariate as ct
from vocab_growth.models import common_univariate_re as cur
from vocab_growth.models.common import ModelFitContext
from vocab_growth.models.definitions import MODEL_REGISTRY, ModelType
from vocab_growth.reporting import console, heading

# §6 regeneration set (family representatives + the reporting models).
_DEFAULT = ["vg10", "vg11", "vg12", "vg13", "vg14", "vg15"]
_UNIVARIATE_RE = {"vg11", "vg12"}
_BIVARIATE_RE = {"vg07", "vg08", "vg09", "vg10", "vg13"}


def _context(definition) -> ModelFitContext:
    ctx = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name=definition.config_name,
            output_root_dir=env.OUTPUT_DIR,
            hdi=0.90,
        ),
        sampling=sampling.get_sampling_configuration("dev"),
    )
    os.makedirs(ctx.reporting.output_dir, exist_ok=True)
    return ctx


def audit(model_key: str) -> str:
    """Build the model and run only its prior-predictive stage; return its dir."""
    d = MODEL_REGISTRY[model_key]
    ctx = _context(d)
    mt = d.model_type
    if mt == ModelType.UNIVARIATE and model_key in _UNIVARIATE_RE:
        cur.prepare_univariate_re_data(ctx, d)
        common.configure_univariate_priors(ctx, d)
        cur.build_univariate_re_model(ctx, d)
        common.prior_predictive_checks(ctx, d.outcome.value, d.outcome_label)
    elif mt == ModelType.UNIVARIATE:
        common.prepare_univariate_data(ctx, d)
        common.configure_univariate_priors(ctx, d)
        common.build_model(ctx)
        common.prior_predictive_checks(ctx, d.outcome.value, d.outcome_label)
    elif mt == ModelType.BIVARIATE:
        if model_key in _BIVARIATE_RE:
            cbr.prepare_bivariate_re_data(ctx, d)
            cb.configure_bivariate_priors(ctx, d)
            cbr.build_model_re(ctx, d)
        else:
            cb.prepare_bivariate_data(ctx, d)
            cb.configure_bivariate_priors(ctx, d)
            cb.build_model(ctx)
        cb.prior_predictive_checks(ctx)
    elif mt == ModelType.TRIVARIATE:
        ct.prepare_trivariate_data(ctx, d)
        ct.configure_trivariate_priors(ctx, d)
        ct.build_model(ctx)
        ct.prior_predictive_checks(ctx)
    elif mt == ModelType.JOINT:
        cj.prepare_joint_data(ctx, d)
        cj.configure_joint_priors(ctx, d)
        cj.build_model(ctx, d)
        cj.prior_predictive_checks(ctx)
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
