# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A cross-lag model must not report the leave-one-out scores that leak.

The lag predictor holds an earlier wave's observed counts as fixed covariates of
every later row it feeds, so leaving one of those likelihood terms out does not
remove that count from the model. The "held-out" score still conditions on the
held-out outcome, and Pareto-k checks the importance-sampling approximation
rather than this leakage, so nothing in the number itself shows it (#242).

Both engines that carry a cross-lag suppress the scores that leak and keep the
one that does not. Neither suppression had a test until VG25's was written, and
the joint engine had no suppression at all: it computed all three marginals plus
the administration-level score, with the leak flagged only in the report's prose.
"""

from __future__ import annotations

import types

import pytest

from vocab_growth.models import common_bivariate, common_joint_modality
from vocab_growth.models.definitions import MODEL_REGISTRY


def _capture(monkeypatch, module):
    """Run the engine's diagnostics stage and return the kwargs it passed on."""
    captured: dict = {}

    def _fake(context, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(module, "_shared_diagnostics", _fake)
    return captured


def _stub_context(variables):
    """The little of a fit context the diagnostics stage reads before scoring."""
    posterior = types.SimpleNamespace(data_vars={name: None for name in variables})
    return types.SimpleNamespace(trace=types.SimpleNamespace(posterior=posterior))


# ==========================================================================
# The joint engine (VG25)
# ==========================================================================


def test_the_sign_lag_drops_the_two_marginals_its_predictor_reads(monkeypatch):
    """VG25 reads an earlier wave's `signed` and `understood`, so both leak."""
    captured = _capture(monkeypatch, common_joint_modality)
    common_joint_modality.diagnostics(
        _stub_context(("psi", "conc", "beta_sign_lag")), MODEL_REGISTRY["vg25"]
    )
    scored = [name for name, _label in captured["loo_var_names"]]
    assert scored == ["y_s_obs"]


def test_the_sign_lag_keeps_the_spoken_score_and_says_what_it_conditions_on(
    monkeypatch,
):
    """A retained score with an unqualified label is the failure mode.

    The spoken count is read by no predictor, so leaving its term out is clean
    with respect to the lag -- but the estimate is prediction conditional on the
    child's observed sign history, not unconditional new-observation
    prediction, and the label is where a reader learns that.
    """
    captured = _capture(monkeypatch, common_joint_modality)
    common_joint_modality.diagnostics(
        _stub_context(("psi", "conc")), MODEL_REGISTRY["vg25"]
    )
    ((_name, label),) = captured["loo_var_names"]
    assert "conditional" in label and "sign" in label


def test_the_sign_lag_suppresses_the_administration_score_too(monkeypatch):
    """It bundles the two leaking factors with the two the lag also enters.

    With `sign_lag_in_cells` the coefficient reaches the composition terms as
    well, so no pointwise hold-out of a whole administration is clean either.
    Reporting one would tell the reader the estimate is
    leave-one-administration-out while it conditions on the administration it
    claims to have left out.
    """
    captured = _capture(monkeypatch, common_joint_modality)
    common_joint_modality.diagnostics(
        _stub_context(("psi", "conc")), MODEL_REGISTRY["vg25"]
    )
    assert captured.get("administration_factors") is None


@pytest.mark.parametrize("model_key", ["vg15", "vg24"])
def test_a_joint_model_without_the_lag_scores_everything_it_always_did(
    monkeypatch, model_key
):
    """The suppression is the lag's, not the engine's."""
    captured = _capture(monkeypatch, common_joint_modality)
    common_joint_modality.diagnostics(
        _stub_context(("psi", "conc")), MODEL_REGISTRY[model_key]
    )
    scored = [name for name, _label in captured["loo_var_names"]]
    assert scored == ["y_u_obs", "y_s_obs", "y_sign_obs"]
    factors = captured["administration_factors"]
    assert [factor.variable for factor in factors] == [
        "y_u_obs",
        "y_s_obs",
        "y_sign_obs",
        "cells_obs",
        "nz_prod_cells_obs",
    ]


# ==========================================================================
# The bivariate random-effect engine (VG16), which had no test either
# ==========================================================================


def test_the_count_lag_drops_the_understood_score(monkeypatch):
    captured = _capture(monkeypatch, common_bivariate)
    common_bivariate.diagnostics(
        _stub_context(("beta_lag", "tau_subj_u")), MODEL_REGISTRY["vg16"]
    )
    scored = [name for name, _label in captured["loo_var_names"]]
    assert scored == ["y_s_obs"]
    assert captured.get("administration_factors") is None


def test_a_bivariate_model_without_the_lag_scores_both_outcomes(monkeypatch):
    captured = _capture(monkeypatch, common_bivariate)
    common_bivariate.diagnostics(
        _stub_context(("tau_subj_u",)), MODEL_REGISTRY["vg10"]
    )
    scored = [name for name, _label in captured["loo_var_names"]]
    assert scored == ["y_s_obs", "y_u_obs"]
    assert captured["administration_factors"] is not None


# ==========================================================================
# Every cross-lag model, on whichever engine
# ==========================================================================


def test_no_registered_cross_lag_model_reports_a_leaking_score(monkeypatch):
    """Fail-closed across the registry rather than model by model.

    A third cross-lag, on either engine, has to arrive with its suppression or
    this fails and names it.
    """
    engines = {
        "bivariate_re": common_bivariate,
        "joint": common_joint_modality,
    }
    from vocab_growth.models.catalogue import engine_for

    lagged = [
        key
        for key, definition in MODEL_REGISTRY.items()
        if getattr(definition, "use_cross_lag", False)
        or getattr(definition, "use_sign_cross_lag", False)
    ]
    assert lagged, "no cross-lag model is registered; this test has lost its subject"

    for key in lagged:
        engine_name = engine_for(key).name
        module = engines.get(engine_name)
        assert module is not None, (
            f"{key} runs on the {engine_name!r} engine, which has no cross-lag "
            "LOO suppression -- add one before registering a lag on it."
        )
        captured = _capture(monkeypatch, module)
        module.diagnostics(_stub_context(("psi", "conc")), MODEL_REGISTRY[key])
        scored = [name for name, _label in captured["loo_var_names"]]
        assert scored == ["y_s_obs"], (
            f"{key} reports {scored}; a cross-lag model may keep only the "
            "spoken score, which no predictor reads."
        )
        assert captured.get("administration_factors") is None, key


# ==========================================================================
# The backfill path, which writes the same table from a stored trace
# ==========================================================================


def _backfill_module():
    """`scripts/emit_loo_summaries.py`, loaded by path as the other tests do."""
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / "emit_loo_summaries.py"
    spec = importlib.util.spec_from_file_location("emit_loo_summaries_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_backfill_refuses_exactly_what_the_engine_refuses(monkeypatch):
    """Two paths write `loo_summary.csv`, and they must not disagree.

    `emit_loo_summaries.py` recomputes the table from a stored trace for fits
    made before the fit pipeline wrote one. It scored every log-likelihood
    variable it found, so a backfill of a cross-lag model would have written the
    row its own engine refuses to compute -- and a published table would then
    hold the leaking number for exactly the models the suppression exists for.
    """
    backfill = _backfill_module()
    engines = {
        "bivariate_re": common_bivariate,
        "joint": common_joint_modality,
    }
    import dataclasses

    from vocab_growth.models.catalogue import engine_for

    for key, definition in MODEL_REGISTRY.items():
        module = engines.get(engine_for(key).name)
        if module is None:
            continue
        lag_fields = [
            field
            for field in ("use_cross_lag", "use_sign_cross_lag")
            if getattr(definition, field, False)
        ]
        refused_by_backfill = backfill.suppressed_outcomes(definition.model_id)
        if not lag_fields:
            assert refused_by_backfill == set(), key
            continue

        def _scored(candidate, engine_module=module):
            captured = _capture(monkeypatch, engine_module)
            engine_module.diagnostics(_stub_context(("psi", "conc")), candidate)
            return {name for name, _label in captured["loo_var_names"]}

        # The universe is what this same model scores with its lag switched off,
        # so the comparison is against the outcomes this engine has rather than
        # against every outcome any engine has -- VG16 carries no `y_sign_obs`
        # to refuse.
        control = dataclasses.replace(definition, **{lag_fields[0]: False})
        refused_by_engine = _scored(control) - _scored(definition)
        assert refused_by_backfill == refused_by_engine, (
            f"{key}: the engine refuses {sorted(refused_by_engine)} but the "
            f"backfill refuses {sorted(refused_by_backfill)}"
        )


def test_a_model_without_a_lag_has_nothing_suppressed_in_a_backfill():
    backfill = _backfill_module()
    for model_id in ("VG10", "VG15", "VG24"):
        assert backfill.suppressed_outcomes(model_id) == set()
