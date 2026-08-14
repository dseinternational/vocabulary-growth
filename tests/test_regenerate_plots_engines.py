# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``regenerate_plots.py``'s engine table.

The script drives each fitted model's plot stage by name, from a declarative
table, so nothing here fails at import time -- a renamed pipeline function or a
model added without an engine surfaces only when someone tries to redraw that
model, which is exactly when they are least able to fix it. The script's own
docstring makes the point about silent skips reading as passes; these tests are
the standing check behind it.

``plots_call`` is the part most likely to rot. The plot stages genuinely differ
in signature -- the single-outcome stage is shared across models plotting
different outcomes and so takes an ``outcome_label`` keyword -- and passing the
wrong convention raises only once a redraw is attempted.
"""

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

from vocab_growth.models.definitions import MODEL_REGISTRY

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "regenerate_plots.py"
_SPEC = importlib.util.spec_from_file_location("regenerate_plots_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

ENGINES = _MODULE.ENGINES
ENGINE_BY_MODEL = _MODULE.ENGINE_BY_MODEL

# Models with no regeneration path, and why. VG11 and VG12 are single-outcome
# models *with* dataset and child random intercepts ("VG03/VG04 + random
# effects"), so they share neither the plain univariate engine's preparation and
# build nor the bivariate RE engine's. They need a ``univariate_re`` entry that
# does not exist yet; until it does they can only be redrawn by refitting.
EXEMPT = {"vg11", "vg12"}


def test_every_registered_model_has_an_engine_or_a_recorded_exemption():
    missing = {
        m for m in MODEL_REGISTRY if m not in ENGINE_BY_MODEL and m not in EXEMPT
    }
    assert not missing, (
        f"models with no plot-regeneration path and no recorded exemption: "
        f"{sorted(missing)}. Add an ENGINES entry, or add the model to EXEMPT "
        "here with the reason, so a cosmetic plot fix does not silently require "
        "a refit."
    )


def test_exemptions_are_still_registered_models():
    """A stale exemption would hide a model that has since gained an engine."""
    for model_id in EXEMPT:
        assert model_id in MODEL_REGISTRY, f"{model_id} is exempt but not registered"
        assert model_id not in ENGINE_BY_MODEL, (
            f"{model_id} now has an engine; remove it from EXEMPT."
        )


@pytest.mark.parametrize("engine_name", sorted(ENGINES))
def test_engine_functions_resolve(engine_name):
    """Every name in the table must exist on the module it names."""
    engine = ENGINES[engine_name]
    module = importlib.import_module(engine["module"])
    for key in ("prepare", "priors", "build", "plots"):
        attr = engine[key]
        assert getattr(module, attr, None) is not None, (
            f"{engine['module']} has no {attr!r} (engine {engine_name!r}, key {key!r})"
        )


@pytest.mark.parametrize("engine_name", sorted(ENGINES))
def test_plots_call_matches_the_plot_stage_signature(engine_name):
    """The declared calling convention must match what the function accepts.

    This is the check that would have caught routing VG01-VG04 through the
    ``definition`` convention: ``run_standard_plots`` takes the context plus a
    keyword-only ``outcome_label``, and would have raised ``TypeError`` on a
    second positional argument.
    """
    engine = ENGINES[engine_name]
    module = importlib.import_module(engine["module"])
    params = inspect.signature(getattr(module, engine["plots"])).parameters
    positional = [
        p
        for p in params.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    call = engine["plots_call"]

    if call == "definition":
        assert len(positional) >= 2, (
            f"{engine['plots']} declares plots_call='definition' but takes "
            f"{len(positional)} positional parameter(s)"
        )
    elif call == "outcome_label":
        assert "outcome_label" in params, (
            f"{engine['plots']} declares plots_call='outcome_label' but has no "
            "such parameter"
        )
        assert len(positional) == 1, (
            f"{engine['plots']} declares plots_call='outcome_label' but takes "
            f"{len(positional)} positional parameter(s); expected just the context"
        )
    elif call == "context":
        assert len(positional) == 1, (
            f"{engine['plots']} declares plots_call='context' but takes "
            f"{len(positional)} positional parameter(s)"
        )
    else:
        pytest.fail(f"engine {engine_name!r} has unknown plots_call {call!r}")


@pytest.mark.parametrize("model_id", sorted(ENGINE_BY_MODEL))
def test_every_model_engine_is_defined(model_id):
    assert ENGINE_BY_MODEL[model_id] in ENGINES


@pytest.mark.parametrize("model_id", sorted(ENGINE_BY_MODEL))
def test_mapped_models_are_registered(model_id):
    assert model_id in MODEL_REGISTRY


def test_outcome_label_is_available_for_the_univariate_models():
    """``plots_call='outcome_label'`` reads it off the definition, so it must exist."""
    for model_id, engine_name in ENGINE_BY_MODEL.items():
        if ENGINES[engine_name]["plots_call"] != "outcome_label":
            continue
        label = MODEL_REGISTRY[model_id].outcome_label
        assert isinstance(label, str) and label, (
            f"{model_id} has no usable outcome_label ({label!r})"
        )
