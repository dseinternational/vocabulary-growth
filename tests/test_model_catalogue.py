# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The model catalogue is the single source for every per-model dispatch.

Before issue #273 a model's engine was declared in five places: the
``model_vgNN`` wrapper's import, ``analysis_frames.FRAME_BUILDERS``,
``scripts/regenerate_plots.py``, ``scripts/prior_predictive_audit.py`` and
``scripts/fit_sensitivity.py`` (with a sixth copy in ``refit_hightune.py`` and a
seventh in ``recovery/spec.py``). Drift guards existed over two of them, which
is why two of the others were wrong at once: the prior audit routed VG16 and
VG19-VG23 through the plain bivariate engine, and the sensitivity scripts could
not reach VG16's, VG21's or VG23's registered variants at all.

Every one of those tables is now derived from
:mod:`vocab_growth.models.catalogue`. These tests pin the catalogue's own
claims against the code they describe -- which is the only place a check is
worth putting once everything else follows from one record.

Nothing here samples. The heaviest thing is importing the engine modules.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

from vocab_growth.analysis_frames import FRAME_BUILDERS
from vocab_growth.models.catalogue import (
    CATALOGUE,
    ENGINES,
    EngineAdapter,
    engine_for,
    get,
)
from vocab_growth.models.definitions import MODEL_REGISTRY

_REPO_ROOT = Path(__file__).parents[1]
_MODEL_KEYS = sorted(MODEL_REGISTRY)
_ENGINE_NAMES = sorted(ENGINES)

#: Adapter fields naming a callable on the engine module. ``plots`` and
#: ``stages`` are optional and checked separately.
_REQUIRED_HOOKS = ("prepare", "priors", "build", "prior_checks", "frame_builder", "fit")


# --- the catalogue covers the registry, exactly ---------------------------------


def test_the_catalogue_and_the_registry_hold_the_same_models():
    assert sorted(CATALOGUE) == _MODEL_KEYS
    # Registration order matters for `fit_model.py all` and every summary table.
    assert list(CATALOGUE) == list(MODEL_REGISTRY)


def test_an_unregistered_model_is_refused_rather_than_guessed():
    """Routing an unknown model through a plausible engine is the failure mode."""
    with pytest.raises(KeyError, match="not in the model catalogue"):
        get("vg99")


def test_lookup_is_case_insensitive():
    assert get("VG01") is get("vg01")


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_the_definition_is_the_registered_one(model_key):
    """Held by reference, never copied: a copy could go stale against the registry."""
    assert get(model_key).definition is MODEL_REGISTRY[model_key]


def test_the_exploratory_modules_are_deliberately_absent():
    """VG17/VG18 bypass the shared manifest, staging and convergence gate.

    A catalogue entry would assert a supported lifecycle they do not have, and
    would make ``fit_model.py`` and the validators offer to treat their output as
    publishable. Their lifecycle is an open model decision (#273 finding 4).
    """
    assert "vg17" not in CATALOGUE
    assert "vg18" not in CATALOGUE
    for key in ("vg17", "vg18"):
        assert (_REPO_ROOT / "src" / "vocab_growth" / "models" / f"model_{key}.py").is_file()


# --- every declared hook exists, and is called the way it is declared ------------


@pytest.mark.parametrize("engine_name", _ENGINE_NAMES)
@pytest.mark.parametrize("hook", _REQUIRED_HOOKS)
def test_every_engine_hook_resolves(engine_name, hook):
    assert callable(ENGINES[engine_name].resolve(hook))


@pytest.mark.parametrize("engine_name", _ENGINE_NAMES)
def test_the_optional_hooks_are_all_or_nothing(engine_name):
    engine = ENGINES[engine_name]
    if engine.plots is None:
        assert engine.plots_call is None
        assert engine.replot_note, (
            f"{engine_name} declares no plot stage and no reason; a silent skip "
            "in a diagnostic reads as a pass"
        )
        assert not engine.supports_replot
    else:
        assert engine.plots_call in {"definition", "context", "outcome_label"}
        assert engine.supports_replot
        assert callable(engine.resolve("plots"))
    if engine.stages is not None:
        assert callable(engine.resolve("stages"))


@pytest.mark.parametrize("engine_name", _ENGINE_NAMES)
def test_resolve_names_the_engine_and_the_field_when_a_hook_is_missing(engine_name):
    broken = EngineAdapter(**{**ENGINES[engine_name].__dict__, "build": "no_such_stage"})
    with pytest.raises(AttributeError, match="no_such_stage"):
        broken.resolve("build")


@pytest.mark.parametrize("engine_name", _ENGINE_NAMES)
def test_the_prior_check_convention_matches_the_signature(engine_name):
    """Passing the wrong convention raises only once that stage is reached.

    Which, for a script that runs a single stage, is at the point of use -- and
    the prior audit is exactly such a script. ``"definition"`` is what carries
    :mod:`vocab_growth.models.prior_child_checks`'s unseen-child figures, so
    getting it wrong drops the figures a child-effect model's prior audit exists
    to look at rather than raising.
    """
    engine = ENGINES[engine_name]
    params = inspect.signature(engine.resolve("prior_checks")).parameters
    positional = [
        p for p in params.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    call = engine.prior_checks_call
    if call == "outcome":
        assert [p.name for p in positional[1:3]] == ["outcome_col", "outcome_label"]
    elif call == "definition":
        assert len(positional) >= 2, f"{engine.prior_checks} takes no definition"
    elif call == "context":
        assert positional[0].name in {"context", "ctx"}
    else:
        pytest.fail(f"engine {engine_name!r} has unknown prior_checks_call {call!r}")


@pytest.mark.parametrize("engine_name", _ENGINE_NAMES)
def test_the_prepare_priors_and_build_stages_take_a_definition(engine_name):
    engine = ENGINES[engine_name]
    for hook in ("prepare", "priors", "build"):
        params = inspect.signature(engine.resolve(hook)).parameters
        positional = [
            p for p in params.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        assert len(positional) >= 2, f"{engine_name}.{hook} takes no definition"


@pytest.mark.parametrize("engine_name", _ENGINE_NAMES)
def test_the_frame_builder_is_pure_in_the_definition(engine_name):
    """It runs outside a fit, so it must take the definition and nothing else."""
    params = inspect.signature(ENGINES[engine_name].resolve("frame_builder")).parameters
    required = [p for p in params.values() if p.default is p.empty and p.kind != p.VAR_KEYWORD]
    assert len(required) == 1, f"{engine_name}'s frame builder needs a fit context"


# --- the declared engine is the one the model actually uses ---------------------


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_the_declared_engine_is_the_one_the_wrapper_imports(model_key):
    """The catalogue's claim, checked against the module that does the fitting.

    Engine identity cannot be inferred from the definition class -- VG05 and VG07
    share ``BivariateModelDefinition`` on different engines -- so it is declared,
    and this is what stops a declaration drifting from the code.
    """
    module = importlib.import_module(get(model_key).wrapper_module)
    imported = {
        value.__module__
        for value in vars(module).values()
        if callable(value) and getattr(value, "__module__", None)
    }
    engine = engine_for(model_key)
    assert engine.module in imported, (
        f"{model_key} is catalogued on {engine.module}, but "
        f"model_{model_key}.py imports from "
        f"{sorted(m for m in imported if 'common' in m)}."
    )


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_fit_dispatches_to_the_declared_engine_with_the_registered_definition(
    model_key, monkeypatch
):
    """``model_vgNN.fit(config)`` must call *this* engine with *this* definition.

    Exercised rather than read: the wrapper binds the engine's fit function at
    import time, so substituting it here is what the call actually reaches. No
    sampling happens -- the substitute records its arguments and returns.
    """
    model = get(model_key)
    module = importlib.import_module(model.wrapper_module)
    calls = []
    monkeypatch.setattr(
        module, model.engine.fit, lambda config, definition: calls.append((config, definition))
    )
    module.fit("test")
    assert calls == [("test", MODEL_REGISTRY[model_key])]


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_the_frame_builder_map_agrees_with_the_catalogue(model_key):
    engine = engine_for(model_key)
    assert FRAME_BUILDERS[model_key] == f"{engine.module}:{engine.frame_builder}"


# --- the derived dispatch tables -----------------------------------------------


def test_the_prior_audit_routes_every_model_through_its_own_engine():
    """The defect this catalogue was built for.

    ``scripts/prior_predictive_audit.py`` held its own bivariate-RE set listing
    only VG07-VG10 and VG13, so VG16 and VG19-VG23 -- the cross-lag,
    child-slope, correlated-effect and factor models -- were audited on a graph
    without the structure that distinguishes them, and the script still produced
    plots.
    """
    import importlib.util
    import sys

    path = _REPO_ROOT / "scripts" / "prior_predictive_audit.py"
    spec = importlib.util.spec_from_file_location("prior_predictive_audit_script", path)
    assert spec is not None and spec.loader is not None
    audit = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = audit
    spec.loader.exec_module(audit)

    source = inspect.getsource(audit.audit)
    # The script must not name an engine module: everything comes from the entry.
    for engine in ENGINES.values():
        assert engine.module not in source, (
            f"the audit hard-codes {engine.module}; it must resolve the engine "
            "from the catalogue"
        )
    for hook in ("prepare", "priors", "build", "prior_checks"):
        assert f'resolve("{hook}")' in source, f"the audit does not run the {hook} stage"

    # And the six models the stale table mis-routed are on the RE engine.
    for model_key in ("vg16", "vg19", "vg20", "vg21", "vg22", "vg23"):
        assert engine_for(model_key).name == "bivariate_re"
        assert engine_for(model_key).prior_checks_call == "definition"


def test_the_recovery_harness_uses_the_catalogued_engine():
    """A recovery fit must run the pipeline of the engine that fits the model.

    The harness substitutes stage 0 for a simulated-frame loader and runs the
    rest unchanged, so pairing a model with another engine's stage factory would
    refit synthetic data through a graph the truth was never drawn from. The
    spec's own ``engine`` field is a second record of the same fact and must
    agree.
    """
    from vocab_growth.recovery.spec import recovery_target, supported_models

    for model_key in supported_models():
        engine = engine_for(model_key)
        target = recovery_target(model_key)
        assert engine.stages is not None, f"{model_key}'s engine has no stage factory"
        assert target.stages_factory == f"{engine.module}:{engine.stages}"
        assert target.spec.engine == engine.module.rpartition(".")[2]


def test_recovery_refuses_a_model_whose_engine_has_no_stage_factory():
    """The univariate, bivariate and trivariate engines build their stages inline.

    A recovery target on one of them cannot have stage 0 swapped, so it must be
    refused by name rather than fail somewhere inside the pipeline.
    """
    from vocab_growth.recovery.spec import supported_models

    for model_key in supported_models():
        assert engine_for(model_key).stages is not None
    inline = [name for name, engine in ENGINES.items() if engine.stages is None]
    assert set(inline) == {"univariate", "bivariate", "trivariate"}


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_every_model_has_a_report_template(model_key):
    """The fit's report stage copies this file and raises when it is absent."""
    template = _REPO_ROOT / get(model_key).report_template
    assert template.is_file(), (
        f"{model_key} has no report template at {get(model_key).report_template}; "
        "the fit pipeline's report stage would raise FileNotFoundError after a "
        "completed fit."
    )


def test_no_orphan_report_templates():
    """A template for an unregistered model is a model someone forgot to remove."""
    templates = {
        path.parent.name
        for path in (_REPO_ROOT / "docs" / "models").glob("*/index.qmd")
    }
    assert templates == set(CATALOGUE), (
        f"report templates without a catalogue entry: {sorted(templates - set(CATALOGUE))}; "
        f"catalogue entries without a template: {sorted(set(CATALOGUE) - templates)}"
    )


# --- documentation that would otherwise be a hand-copied count ------------------


def test_the_inventory_covers_every_catalogued_model():
    """``docs/models/README.md`` is named the single source of truth for the set.

    A model registered without an inventory row leaves the document that claims
    to be canonical describing a smaller family than the code fits.
    """
    inventory = (_REPO_ROOT / "docs" / "models" / "README.md").read_text(encoding="utf-8")
    missing = [
        key for key in CATALOGUE if f"[{key.upper()}]({key}/index.qmd)" not in inventory
    ]
    assert not missing, (
        f"registered models with no docs/models/README.md inventory row: {missing}"
    )


@pytest.mark.parametrize(
    "path", ["AGENTS.md", "CLAUDE.md", ".github/copilot-instructions.md"]
)
def test_the_agent_instructions_state_the_right_model_count(path):
    """The three copies carry the count in prose, so it can drift silently.

    It has: ``models/__init__.py`` said "VG01-VG16" for as long as there had
    been twenty registered models. Pinning the word against the catalogue is
    what turns the next registration into a failing test rather than a document
    that quietly stops being true.
    """
    words = {
        18: "eighteen", 19: "nineteen", 20: "twenty", 21: "twenty-one",
        22: "twenty-two", 23: "twenty-three", 24: "twenty-four",
    }
    count = len(CATALOGUE)
    expected = words.get(count)
    assert expected, f"add the number word for {count} to this test"
    text = (_REPO_ROOT / path).read_text(encoding="utf-8")
    assert f"{expected} registered models" in text, (
        f"{path} does not say '{expected} registered models'; the catalogue holds "
        f"{count}."
    )
    # And the ranges beside it must name every one of them.
    for key in CATALOGUE:
        model_id = key.upper()
        assert model_id in text or _in_a_range(model_id, text), (
            f"{path}'s model-set sentence does not cover {model_id}"
        )


def _in_a_range(model_id: str, text: str) -> bool:
    """Whether ``text`` names a ``VGnn`-`VGmm`` range covering ``model_id``."""
    import re

    number = int(model_id[2:])
    return any(
        int(low) <= number <= int(high)
        for low, high in re.findall(r"`VG(\d{2})`-`VG(\d{2})`", text)
    )


def test_the_package_docstring_points_at_the_source_rather_than_restating_it():
    """``models/__init__.py`` carried its own model range, and it rotted.

    It said "VG01-VG16" for as long as there had been twenty registered models.
    The fix is not a corrected range but no range: the docstring names where the
    set lives.
    """
    import vocab_growth.models as package

    docstring = package.__doc__ or ""
    assert "catalogue" in docstring and "MODEL_REGISTRY" in docstring, (
        "the package docstring should point at the catalogue and the registry "
        "rather than restate what they hold"
    )
