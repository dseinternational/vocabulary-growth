# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check catalogue hooks, engine dispatch and documented reporting roles.

These tests compare declarations with wrappers, stage factories and guides.
They import engines but do not sample.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

import pytest

from vocab_growth.analysis_frames import FRAME_BUILDERS
from vocab_growth.models.catalogue import (
    CATALOGUE,
    ENGINES,
    EngineAdapter,
    ModelRole,
    engine_for,
    get,
    models_with_role,
    publication_models,
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
    """Keep VG17 and VG18 outside registered fitting and publication paths."""
    assert "vg17" not in CATALOGUE
    assert "vg18" not in CATALOGUE
    exploratory = _REPO_ROOT / "src" / "vocab_growth" / "models" / "exploratory"
    for key in ("vg17", "vg18"):
        assert (exploratory / f"{key}.py").is_file()
        assert not (
            _REPO_ROOT / "src" / "vocab_growth" / "models" / f"model_{key}.py"
        ).exists(), (
            f"model_{key}.py is back beside the registered wrappers, where "
            "fit_model.py's name resolution can reach it"
        )


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
def test_a_replot_engine_declares_exactly_one_way_to_obtain_samples(engine_name):
    """Require one sample source: extraction or posterior predictive sampling."""
    engine = ENGINES[engine_name]
    declared = [
        h for h in ("samples_extractor", "posterior_predictive")
        if getattr(engine, h) is not None
    ]
    if not engine.supports_replot:
        return
    assert len(declared) == 1, (
        f"{engine_name} declares {declared or 'neither'}; a replot uses exactly one"
    )
    assert callable(engine.resolve(declared[0]))


@pytest.mark.parametrize("engine_name", _ENGINE_NAMES)
def test_a_declared_extractor_really_is_importable_from_the_engine_module(engine_name):
    """Check that the declared sample source matches the engine exports."""
    engine = ENGINES[engine_name]
    module = importlib.import_module(engine.module)
    exports_extractor = hasattr(module, "extract_model_samples")
    if engine.samples_extractor is not None:
        assert exports_extractor, (
            f"{engine_name} declares samples_extractor but {engine.module} has no "
            "extract_model_samples"
        )
    elif engine.supports_replot:
        assert not exports_extractor, (
            f"{engine.module} exports extract_model_samples but {engine_name} "
            "declares posterior_predictive. Decide which the replot should use and "
            "declare it -- do not leave the two disagreeing."
        )


@pytest.mark.parametrize("engine_name", _ENGINE_NAMES)
def test_resolve_names_the_engine_and_the_field_when_a_hook_is_missing(engine_name):
    broken = EngineAdapter(**{**ENGINES[engine_name].__dict__, "build": "no_such_stage"})
    with pytest.raises(AttributeError, match="no_such_stage"):
        broken.resolve("build")


@pytest.mark.parametrize("engine_name", _ENGINE_NAMES)
def test_the_prior_check_convention_matches_the_signature(engine_name):
    """Check the prior-check signature against its declared calling convention."""
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
    """Check the wrapper import because the definition class does not identify its engine."""
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
    """Record wrapper dispatch without running the engine or sampling."""
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
    """Check that each prior audit uses the catalogued engine."""
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

    # These models use the bivariate random-effects engine.
    for model_key in ("vg16", "vg19", "vg20", "vg21", "vg22", "vg23"):
        assert engine_for(model_key).name == "bivariate_re"
        assert engine_for(model_key).prior_checks_call == "definition"


def test_the_recovery_harness_uses_the_catalogued_engine():
    """Keep recovery stage factories and engine declarations consistent."""
    from vocab_growth.recovery.spec import recovery_target, supported_models

    for model_key in supported_models():
        engine = engine_for(model_key)
        target = recovery_target(model_key)
        assert engine.stages is not None, f"{model_key}'s engine has no stage factory"
        assert target.stages_factory == f"{engine.module}:{engine.stages}"
        assert target.spec.engine == engine.module.rpartition(".")[2]


def test_recovery_refuses_a_model_whose_engine_has_no_stage_factory():
    """Reject engines whose inline stages cannot accept a simulated-frame loader."""
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
    """Require every report template to belong to a registered model."""
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
    """Keep the documented model inventory consistent with the executable catalogue."""
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
    """Check the model count and ranges in all three instruction copies."""
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
    """Point readers to the registry instead of duplicating its model range."""
    import vocab_growth.models as package

    docstring = package.__doc__ or ""
    assert "catalogue" in docstring and "MODEL_REGISTRY" in docstring, (
        "the package docstring should point at the catalogue and the registry "
        "rather than restate what they hold"
    )


# --- roles: declared here, pinned against the documented table ------------------
#
# Keep the catalogue's publication roles consistent with the documented table.



def _roles_table_section() -> str:
    """The ``### Model roles`` section of the inventory, which is the record."""
    text = (_REPO_ROOT / "docs" / "models" / "README.md").read_text(encoding="utf-8")
    start = text.index("### Model roles")
    section = text[start:]
    end = section.find(chr(10) + "### ", 1)
    return section if end == -1 else section[:end]


def _documented_development_steps() -> set[str]:
    """Model keys the roles table calls a development step or model."""
    section = _roles_table_section()
    found: set[str] = set()
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2 or "Development" not in cells[1]:
            continue
        # The first cell may name several models ("VG05, VG07, VG08").
        for token in cells[0].replace(",", " ").split():
            token = token.strip("*` ")
            if token.lower() in MODEL_REGISTRY:
                found.add(token.lower())
    return found


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_every_registered_model_declares_a_role(model_key):
    """A model with no role would silently pick up whatever the default is."""
    assert isinstance(get(model_key).role, ModelRole)


def test_the_documented_development_steps_are_declared_as_such():
    """The roles table is the record; the catalogue must agree with it.

    Only checked in the direction the table can support. The table names the
    development steps explicitly, so every one of them must be declared. It does
    not name a role for every model, and the ones it omits are deliberately
    ``UNCLASSIFIED`` rather than guessed.
    """
    documented = _documented_development_steps()
    assert documented, "the roles table named no development steps; has it moved?"
    declared = {
        key for key, model in CATALOGUE.items() if model.role is ModelRole.DEVELOPMENT_STEP
    }
    assert documented <= declared, (
        "documented as a development step but not declared one: "
        f"{sorted(documented - declared)}"
    )


#: The roles table's own labels, matched by prefix on the first bold span of
#: the Role cell, so ``**Development step**``, ``**Development steps.**`` and
#: ``**Development model, single-purpose; …**`` all read as one role.
_ROLE_LABELS = (
    ("Model of record", ModelRole.MODEL_OF_RECORD),
    ("Development", ModelRole.DEVELOPMENT_STEP),
    ("TD reference", ModelRole.TD_REFERENCE),
    ("Superseded", ModelRole.SUPERSEDED),
)


def _documented_roles() -> dict[str, ModelRole]:
    """Return documented roles, rejecting labels that match no known role."""
    found: dict[str, ModelRole] = {}
    for line in _roles_table_section().splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        keys = [
            token.strip("*` ").lower()
            for token in cells[0].replace(",", " ").split()
        ]
        keys = [key for key in keys if key in MODEL_REGISTRY]
        if not keys:
            continue
        match = re.match(r"\*\*(.+?)\*\*", cells[1])
        assert match, f"roles-table row for {keys} has no bold role label: {cells[1][:60]!r}"
        label = match.group(1)
        role = next((r for prefix, r in _ROLE_LABELS if label.startswith(prefix)), None)
        assert role is not None, f"unrecognised role label {label!r} for {keys}"
        for key in keys:
            assert key not in found, f"{key} appears in two roles-table rows"
            found[key] = role
    return found


def test_every_role_the_table_states_is_declared_exactly():
    """The table is the record; the catalogue must say the same, model for model."""
    documented = _documented_roles()
    assert documented, "the roles table names no models; has it moved?"
    for key, role in documented.items():
        assert CATALOGUE[key].role is role, (
            f"{key}: the roles table says {role.value!r}, the catalogue declares "
            f"{CATALOGUE[key].role.value!r}"
        )


def test_every_declared_role_has_a_roles_table_row():
    """A classified model with no row would leave the code free of the record."""
    documented = _documented_roles()
    undocumented = sorted(
        key
        for key, model in CATALOGUE.items()
        if model.role is not ModelRole.UNCLASSIFIED and key not in documented
    )
    assert not undocumented, f"declared a role but absent from the roles table: {undocumented}"


def test_an_unclassified_model_still_requires_publication_validation():
    """Require publication validation when a model has no assigned role."""
    assert ModelRole.UNCLASSIFIED.publication_required


@pytest.mark.parametrize(
    "role", [ModelRole.MODEL_OF_RECORD, ModelRole.TD_REFERENCE, ModelRole.UNCLASSIFIED]
)
def test_the_roles_that_supply_numbers_are_publication_required(role):
    assert role.publication_required


@pytest.mark.parametrize("role", [ModelRole.DEVELOPMENT_STEP, ModelRole.SUPERSEDED])
def test_the_roles_that_supply_no_number_are_not_publication_required(role):
    """Exclude development and superseded roles from required publication fits."""
    assert not role.publication_required


def test_at_least_one_model_of_record_is_declared():
    """A catalogue with no model of record would publish nothing, silently."""
    assert any(m.role is ModelRole.MODEL_OF_RECORD for m in CATALOGUE.values())


def test_the_publication_scope_is_exactly_the_publication_required_models():
    """What the refit driver defaults to, and what the sync refuses without."""
    assert set(publication_models()) == {
        key for key, model in CATALOGUE.items() if model.role.publication_required
    }


def test_the_publication_scope_is_a_strict_subset_of_the_registry():
    """If it were the whole registry the role wiring would be doing nothing."""
    scope = set(publication_models())
    assert scope < set(MODEL_REGISTRY)


def test_an_unclassified_model_is_still_in_the_publication_scope():
    """Keep unclassified models in the required refit scope."""
    unclassified = models_with_role(ModelRole.UNCLASSIFIED)
    assert set(unclassified) <= set(publication_models())


def test_no_development_step_is_in_the_publication_scope():
    steps = set(models_with_role(ModelRole.DEVELOPMENT_STEP))
    assert steps and not (steps & set(publication_models()))


def test_the_scope_helpers_return_registry_keys():
    for key in publication_models() + models_with_role(ModelRole.DEVELOPMENT_STEP):
        assert key in MODEL_REGISTRY
