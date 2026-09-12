# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The shared cross-validation fold fit, and the bug that made it shared.

``kfold_loso.py`` and ``wave_forward_score.py`` hold different things out and
score different units, but the fit between those two decisions is identical.
The second script had it copied by hand, and the copy read the energy verdict
from ``gate["bfmi_ok"]`` where the payload carries it under
``gate["checks"]["bfmi"]`` -- so every fold reported a passing BFMI whatever the
sampler found, and nothing in the output would have shown it.

So the two claims pinned here are the reading itself, and that there is only one
of it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from vocab_growth.fold_fits import fold_gate_fields

SCRIPTS = Path(__file__).parents[1] / "scripts"


def _gate(**overrides) -> dict:
    payload = {
        "passed": True,
        "max_rhat": 1.004,
        "min_ess": 1500.0,
        "divergences": 0,
        "checks": {
            "rhat": True,
            "ess": True,
            "divergences": True,
            "bfmi": True,
            "diagnostics_assessable": True,
        },
    }
    payload.update(overrides)
    return payload


def test_the_energy_verdict_is_read_from_the_nested_checks():
    """The exact mistake the copy made.

    A gate that passed everything but the energy check must come back with
    ``bfmi_ok`` False. Reading a top-level ``bfmi_ok`` returns the default and
    silently reports every fold as fine.
    """
    checks = _gate()["checks"] | {"bfmi": False}
    fields = fold_gate_fields(_gate(passed=False, checks=checks))
    assert fields["bfmi_ok"] is False
    assert fields["passed"] is False


def test_an_absent_energy_check_is_unknown_rather_than_passing():
    """``None``, not ``True``.

    A gate payload with no energy check has not told us the fit was fine; a
    default of True would put that claim in the fits table.
    """
    fields = fold_gate_fields(_gate(checks={}))
    assert fields["bfmi_ok"] is None


@pytest.mark.parametrize(
    "field,value",
    [("max_rhat", 1.03), ("min_ess", 55.0), ("divergences", 7)],
)
def test_the_scalar_verdicts_pass_through(field, value):
    assert fold_gate_fields(_gate(**{field: value}))[field] == value


def test_no_script_defines_its_own_copy_of_the_gate_reading():
    """One reading of the payload, or the next copy repeats the bug.

    Checked on the source rather than by import, so a script that is not
    importable in a test environment still counts.
    """
    offenders = []
    for path in sorted(SCRIPTS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in {
                "fold_gate_fields",
                "fit_holdout_fold",
            }:
                offenders.append(f"{path.name}:{node.name}")
    assert not offenders, (
        "These scripts define their own copy of a fold-fit helper instead of "
        f"importing vocab_growth.fold_fits: {', '.join(offenders)}."
    )


# ==========================================================================
# The engine a fold is built with, and what its trace must expose
# ==========================================================================


#: What a forward score reads off a fold trace, per engine. Both entries are the
#: quantities `scripts/wave_forward_score.py` evaluates a held-out row's density
#: from; the joint entry is longer because that engine carries a third marginal
#: and two composition likelihoods.
_ROW_QUANTITIES = {
    "vg16": (
        "p_u_obs",
        "p_s_obs",
        "q_obs",
        "kappa_u_obs",
        "kappa_s_obs",
    ),
    "vg25": (
        "p_u_obs",
        "q_obs",
        "r_obs",
        "kappa_u_obs",
        "kappa_s_obs",
        "kappa_sign_obs",
        "pi_cells_obs",
    ),
}


def test_a_fold_is_built_by_its_own_models_engine():
    """It was hard-wired to the bivariate random-effect builder.

    Nothing said so, and nothing would have: a joint definition passed in would
    have been built by `build_model_re` and produced a graph that is not the
    model, then fitted and scored without complaint.
    """
    from vocab_growth.models.catalogue import engine_for_definition
    from vocab_growth.models.definitions import MODEL_REGISTRY

    assert engine_for_definition(MODEL_REGISTRY["vg16"]).name == "bivariate_re"
    assert engine_for_definition(MODEL_REGISTRY["vg25"]).name == "joint"


def test_the_engine_is_found_for_a_control_arm_too():
    """A fold arm is a `dataclasses.replace` with a different `config_name`.

    The engine is a property of the model, so it has to be found by something a
    variant copy does not change -- which the config name is not.
    """
    import dataclasses

    from vocab_growth.models.catalogue import engine_for_definition
    from vocab_growth.models.definitions import MODEL_REGISTRY

    definition = MODEL_REGISTRY["vg25"]
    control = dataclasses.replace(
        definition,
        use_sign_cross_lag=False,
        config_name=f"{definition.config_name}-nolag",
    )
    assert engine_for_definition(control).name == "joint"


def test_an_unregistered_definition_has_no_engine_rather_than_a_default():
    import dataclasses

    from vocab_growth.models.catalogue import engine_for_definition
    from vocab_growth.models.definitions import MODEL_REGISTRY

    stranger = dataclasses.replace(MODEL_REGISTRY["vg25"], model_id="VG99")
    with pytest.raises(KeyError, match="VG99"):
        engine_for_definition(stranger)


@pytest.mark.slow
@pytest.mark.parametrize("model_key", sorted(_ROW_QUANTITIES))
def test_the_quantities_a_forward_score_reads_are_named_and_never_stored(
    model_key, tmp_path, monkeypatch
):
    """Two halves, and both matter.

    **Named**, or the forward score cannot evaluate a held-out row's density at
    all -- which is where the joint engine was: it named only `kappa_sign_obs`
    and `z_obs`, so VG25's report could point the reader at a script that could
    not run.

    **Never stored**, or naming them would put an ``n_obs x draws`` array per
    quantity into every trace, which is exactly what
    `fit_artifacts.sampled_variable_names` exists to keep out. They carry
    ``obs_id``, so the sampler skips them and only a caller that asks --
    `fit_holdout_fold`, with ``store_observation_deterministics`` -- pays for
    them.
    """
    from _pytest.monkeypatch import MonkeyPatch
    from support.synthetic_graphs import build_registered_model

    from vocab_growth.fit_artifacts import (
        sampled_variable_names,
        unsampled_deterministic_names,
    )

    patcher = MonkeyPatch()
    try:
        context = build_registered_model(
            model_key, output_dir=str(tmp_path), monkeypatch=patcher
        )
    finally:
        patcher.undo()

    named = {d.name for d in context.model.deterministics}
    unstored = set(unsampled_deterministic_names(context.model))
    stored = set(sampled_variable_names(context.model))

    for quantity in _ROW_QUANTITIES[model_key]:
        assert quantity in named, f"{model_key} does not name {quantity}"
        assert quantity in unstored, f"{model_key} would store {quantity} in every fit"
        assert quantity not in stored
