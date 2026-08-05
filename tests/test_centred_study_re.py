# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guard the optional centred parameterisation of the study random intercepts.

``UnivariateModelDefinition.centred_study_re`` switches the study block from the
non-centred form of issue #65 (``delta = tau * delta_raw``) to sampling ``delta``
directly as ``ZeroSumNormal(sigma=tau * sqrt(K/(K-1)))``. Three things need
pinning:

1. the flag defaults to off, and with it off the graph is exactly the one every
   existing fit was produced under -- this is what keeps the registered models'
   manifests valid;
2. with it on, ``delta`` becomes a free RV and ``delta_raw`` disappears; and
3. the two branches induce the **same prior** on ``delta``, so the switch is a
   change of sampling coordinates rather than of the model.

(3) is the load-bearing claim: it is why enabling the flag cannot move a
posterior except through the sampler's efficiency. See
``notes/202608050900-td-hierarchical-geometry.md`` §§2-3 for why it is expected
to help ``tau``'s ESS and *not* the energy BFMI.
"""

import dataclasses
import os

import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pymc as pm
import pytest

import vocab_growth.data_utils as vocab_data_utils
from vocab_growth.models import common_univariate_re as cur
from vocab_growth.models.common import ModelFitContext
from vocab_growth.models.definitions import VG12


def _build(definition, tmp_path, monkeypatch):
    monkeypatch.setattr(cur, "render_model_graph", lambda *a, **k: None)
    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name=definition.config_name,
            output_root_dir=str(tmp_path),
            ci_prob=0.90,
            interval_kind="hdi",
        ),
        sampling=sampling.get_sampling_configuration("dev"),
    )
    os.makedirs(context.reporting.output_dir, exist_ok=True)
    cur.prepare_univariate_re_data(context, definition)
    cur.configure_univariate_priors(context, definition)
    cur.build_univariate_re_model(context, definition)
    return context.model


# A cheap stand-in for VG12: same engine and study block, a tenth of the rows.
# Only the graph's *structure* is under test, so the subsample is immaterial.
SMALL = dataclasses.replace(VG12, sample_fraction=0.1, min_study_observations=20)


@pytest.fixture(scope="module")
def _require_data():
    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")


def test_flag_defaults_off_on_every_registered_univariate_model():
    """No registered model may silently acquire the centred branch."""
    from vocab_growth.models.definitions import MODEL_REGISTRY

    for definition in MODEL_REGISTRY.values():
        if hasattr(definition, "centred_study_re"):
            assert definition.centred_study_re is False, (
                f"{definition.model_id} enables centred_study_re; enabling it is a "
                "graph change that invalidates that model's existing fits."
            )


def test_non_centred_branch_is_unchanged(_require_data, tmp_path, monkeypatch):
    model = _build(SMALL, tmp_path, monkeypatch)
    free = {v.name for v in model.free_RVs}
    deterministics = {d.name for d in model.deterministics}
    assert "delta_raw" in free
    assert "delta" in deterministics
    assert "delta" not in free
    assert "tau" in free


def test_centred_branch_samples_delta_directly(_require_data, tmp_path, monkeypatch):
    centred = dataclasses.replace(SMALL, centred_study_re=True)
    model = _build(centred, tmp_path, monkeypatch)
    free = {v.name for v in model.free_RVs}
    deterministics = {d.name for d in model.deterministics}
    assert "delta" in free
    assert "delta_raw" not in free and "delta_raw" not in deterministics
    assert "tau" in free
    # Still the zero-sum axis, so the intercept/study-mean ridge stays removed.
    assert "ZeroSum" in type(model["delta"].owner.op).__name__


def test_centred_branch_still_sums_to_zero(_require_data, tmp_path, monkeypatch):
    centred = dataclasses.replace(SMALL, centred_study_re=True)
    model = _build(centred, tmp_path, monkeypatch)
    with model:
        draws = pm.draw(model["delta"], draws=64, random_seed=0)
    assert np.allclose(draws.sum(axis=-1), 0.0, atol=1e-6)


@pytest.mark.parametrize("n_studies", [4, 12])
def test_the_two_branches_induce_the_same_prior(n_studies):
    """``tau * ZSN(s)`` and ``ZSN(tau * s)`` are the same distribution.

    Checked on the marginal SD and on the full sorted draw distribution, at a
    fixed ``tau`` prior, rather than by trusting the algebra alone.
    """
    zsn_sigma = float(np.sqrt(n_studies / (n_studies - 1)))
    draws = 40_000

    with pm.Model():
        tau = pm.HalfNormal("tau", sigma=0.5)
        non_centred = pm.Deterministic(
            "d", tau * pm.ZeroSumNormal("raw", sigma=zsn_sigma, shape=n_studies)
        )
        a = pm.draw(non_centred, draws=draws, random_seed=11)

    with pm.Model():
        tau = pm.HalfNormal("tau", sigma=0.5)
        centred = pm.ZeroSumNormal("d", sigma=tau * zsn_sigma, shape=n_studies)
        b = pm.draw(centred, draws=draws, random_seed=11)

    # Marginal per-study SD matches, and equals tau's own scale (the sqrt(K/(K-1))
    # rescaling is exactly what preserves it).
    assert a.std() == pytest.approx(b.std(), rel=0.05)
    # Whole distribution matches, not just its second moment.
    qs = np.linspace(0.01, 0.99, 99)
    assert np.allclose(
        np.quantile(a.ravel(), qs), np.quantile(b.ravel(), qs), atol=0.02
    )
