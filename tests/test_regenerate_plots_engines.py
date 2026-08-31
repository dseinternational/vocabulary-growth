# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``regenerate_plots.py``'s engine table.

The script drives each fitted model's plot stage by name, from a declarative
table, so nothing here fails at import time -- a renamed pipeline function or a
model added without an engine surfaces only when someone tries to redraw that
model, which is exactly when they are least able to fix it. The script's own
docstring makes the point about silent skips reading as passes; these tests are
the standing check behind it.

Since issue #273 the table is derived from
:mod:`vocab_growth.models.catalogue` rather than restated in the script, so what
is checked here is that the *derivation* still produces a usable dispatch: the
names resolve, the calling conventions match the signatures, and every
registered model is either drivable or carries a recorded reason why not.
``tests/test_model_catalogue.py`` checks the catalogue's own claims.

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

from vocab_growth.models.catalogue import CATALOGUE
from vocab_growth.models.definitions import MODEL_REGISTRY

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "regenerate_plots.py"
_SPEC = importlib.util.spec_from_file_location("regenerate_plots_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

ENGINES = _MODULE.ENGINES
ENGINE_BY_MODEL = _MODULE.ENGINE_BY_MODEL

# Models with no regeneration path, derived from the catalogue rather than
# listed here. VG11 and VG12 are single-outcome models *with* dataset and child
# random intercepts ("VG03/VG04 + random effects"), so they share neither the
# plain univariate engine's preparation and build nor the bivariate RE engine's;
# their engine records that it has no exercised replot path, and that record is
# what this set reads.
EXEMPT = {key for key, model in CATALOGUE.items() if not model.engine.supports_replot}


def test_every_registered_model_has_an_engine_or_a_recorded_exemption():
    missing = {
        m for m in MODEL_REGISTRY if m not in ENGINE_BY_MODEL and m not in EXEMPT
    }
    assert not missing, (
        f"models with no plot-regeneration path and no recorded exemption: "
        f"{sorted(missing)}. Give the model's engine a plot hook in the "
        "catalogue, or a replot_note saying why it has none, so a cosmetic plot "
        "fix does not silently require a refit."
    )


def test_every_exemption_carries_a_reason():
    """A silent skip in a diagnostic reads as a pass; a reason is what stops that."""
    assert EXEMPT == {"vg11", "vg12"}, (
        f"the set of models that cannot be redrawn changed: {sorted(EXEMPT)}. "
        "That is a real change in what a cosmetic fix costs, so it belongs in a "
        "commit message rather than passing silently."
    )
    for model_id in EXEMPT:
        assert model_id in MODEL_REGISTRY, f"{model_id} is exempt but not registered"
        assert model_id not in ENGINE_BY_MODEL, (
            f"{model_id} now has an engine; the catalogue and the derived table "
            "disagree."
        )
        note = CATALOGUE[model_id].engine.replot_note
        assert note, f"{model_id}'s engine has no plot hook and no replot_note"
        assert _MODULE._no_replot_reason(model_id) == note


@pytest.mark.parametrize("engine_name", sorted(ENGINES))
def test_engine_functions_resolve(engine_name):
    """Every name in the table must exist on the module it names."""
    engine = ENGINES[engine_name]
    for key in ("prepare", "priors", "build", "plots", "prior_checks", "fit"):
        # `resolve` raises AttributeError naming the engine and the field.
        assert callable(engine.resolve(key)), f"{engine_name}.{key}"


@pytest.mark.parametrize("engine_name", sorted(ENGINES))
def test_plots_call_matches_the_plot_stage_signature(engine_name):
    """The declared calling convention must match what the function accepts.

    This is the check that would have caught routing VG01-VG04 through the
    ``definition`` convention: ``run_standard_plots`` takes the context plus a
    keyword-only ``outcome_label``, and would have raised ``TypeError`` on a
    second positional argument.
    """
    engine = ENGINES[engine_name]
    params = inspect.signature(engine.resolve("plots")).parameters
    positional = [
        p
        for p in params.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    call = engine.plots_call

    if call == "definition":
        assert len(positional) >= 2, (
            f"{engine.plots} declares plots_call='definition' but takes "
            f"{len(positional)} positional parameter(s)"
        )
    elif call == "outcome_label":
        assert "outcome_label" in params, (
            f"{engine.plots} declares plots_call='outcome_label' but has no "
            "such parameter"
        )
        assert len(positional) == 1, (
            f"{engine.plots} declares plots_call='outcome_label' but takes "
            f"{len(positional)} positional parameter(s); expected just the context"
        )
    elif call == "context":
        assert len(positional) == 1, (
            f"{engine.plots} declares plots_call='context' but takes "
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
        if ENGINES[engine_name].plots_call != "outcome_label":
            continue
        label = MODEL_REGISTRY[model_id].outcome_label
        assert isinstance(label, str) and label, (
            f"{model_id} has no usable outcome_label ({label!r})"
        )
