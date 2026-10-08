# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regenerate prior-predictive figures without posterior sampling.

Resolve the engine and its calling convention through the model catalogue.
The default selects one model per engine and per distinct child-effect structure.
Other registered models may be requested explicitly.

Usage::

    python scripts/prior_predictive_audit.py [models...]
"""

import argparse
import os
import sys
from multiprocessing import freeze_support

import dse_research_utils.environment.setup as setup
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling

from vocab_growth import environment as env
from vocab_growth.models import subject_effects
from vocab_growth.models.catalogue import get as catalogue_get
from vocab_growth.models.common import ModelFitContext
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.reporting import console, heading


def _default_models() -> list[str]:
    """Select engine and child-effect coverage in stable registry order.

    One example per engine exercises each graph builder. Distinct child structures
    also need examples so the default reaches their unseen-child prior checks.
    This is structural coverage, not an audit of every registered definition.
    """
    chosen: list[str] = []
    seen_engines: set[str] = set()
    seen_structures: set[str] = set()
    for key in MODEL_REGISTRY:
        record = catalogue_get(key)
        engine = record.engine.name
        structure = _child_effect_signature(MODEL_REGISTRY[key])
        if engine not in seen_engines or structure not in seen_structures:
            chosen.append(key)
        seen_engines.add(engine)
        seen_structures.add(structure)
    return chosen


def _child_effect_signature(definition) -> str:
    """A comparable label for a definition's child-effect structure.

    Read through :func:`vocab_growth.models.subject_effects.resolve`, the one
    resolver, rather than by sniffing fields here.
    """
    try:
        plan = subject_effects.resolve(definition)
    except Exception:
        # Not a multi-outcome definition, or a combination the resolver refuses:
        # either way it is not a child-effect model and groups with the others.
        return "none"
    kinds = "+".join(sorted({effect.kind.value for effect in plan.effects}))
    return f"{kinds}|corr={plan.correlation_eta is not None}|factor={plan.factor is not None}"


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
    parser.add_argument(
        "models",
        nargs="*",
        default=None,
        help="Model keys (default: one per engine and per child-effect structure).",
    )
    freeze_support()
    setup.init_script()
    args = parser.parse_args()
    using_default = not args.models
    models = [m.lower() for m in (args.models or _default_models())]
    if using_default:
        # Say what a default run does NOT cover. A partial default that reads as
        # complete is what let VG19-VG23 go unaudited; naming the gap is the
        # cheapest guard against the same thing happening to the next model.
        skipped = [key for key in MODEL_REGISTRY if key not in models]
        console.print(
            f"[dim]Default set ({len(models)} of {len(MODEL_REGISTRY)}): one model per "
            f"engine and per child-effect structure. Not audited: "
            f"{', '.join(k.upper() for k in skipped) or 'none'} — each shares both an "
            f"engine and a child-effect structure with a model above. "
            f"Pass model keys explicitly to override.[/dim]"
        )

    for key in models:
        if key not in MODEL_REGISTRY:
            console.print(f"[bold red]Unknown model: {key}[/bold red]")
            sys.exit(1)
        heading(f"Prior-predictive audit: {key.upper()}")
        out = audit(key)
        console.print(f"[dim]Prior-predictive plots regenerated in {out}[/dim]")
