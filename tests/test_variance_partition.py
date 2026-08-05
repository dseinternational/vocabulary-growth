# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guard the shared-scatter-budget reparameterisation of the subject/dispersion pair.

``SubjectVariancePartitionParams`` replaces independent priors on ``tau_subject``
and the young ``kappa`` anchor with one prior on their total and one on the share
between them, to break the ridge that drives VG12's and VG13's energy BFMI
failure (``notes/202608050900-td-hierarchical-geometry.md`` §§2, 4, 7.1).

What has to hold:

1. no registered model uses it yet -- it is calibrated but unvalidated, and
   attaching it is a graph change that would invalidate that model's fits;
2. with it off, the graph is exactly what every existing fit was produced under;
3. with it on, ``tau_subject`` and ``kappa_excess_young`` are *still present* under
   their usual names, because the DS/TD heterogeneity contrast and every summary
   read them by name -- this is the property that makes the change a
   reparameterisation rather than a different model;
4. the algebra round-trips: the budget and share recover the two scales exactly;
5. the misuse combinations are rejected rather than silently ignored.
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
from vocab_growth.models.definitions import (
    MODEL_REGISTRY,
    _TD_UNDERSTOOD_VARIANCE_PARTITION,
    VG12,
)

SMALL = dataclasses.replace(VG12, sample_fraction=0.1, min_study_observations=20)
PARTITIONED = dataclasses.replace(
    SMALL, subject_variance_partition=_TD_UNDERSTOOD_VARIANCE_PARTITION
)


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


@pytest.fixture(autouse=True)
def _require_data():
    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")


def test_no_registered_model_uses_the_partition_yet():
    for definition in MODEL_REGISTRY.values():
        assert getattr(definition, "subject_variance_partition", None) is None, (
            f"{definition.model_id} enables subject_variance_partition; it is "
            "calibrated but not yet validated, and attaching it invalidates fits."
        )


def test_without_the_partition_the_scales_are_free_rvs(tmp_path, monkeypatch):
    model = _build(SMALL, tmp_path, monkeypatch)
    free = {v.name for v in model.free_RVs}
    assert {"tau_subject", "kappa_excess_young"}.issubset(free)
    assert "v_total" not in free
    assert "subject_variance_share" not in free


def test_with_the_partition_the_budget_is_sampled_instead(tmp_path, monkeypatch):
    model = _build(PARTITIONED, tmp_path, monkeypatch)
    free = {v.name for v in model.free_RVs}
    deterministics = {d.name for d in model.deterministics}
    # The budget and the split are what the sampler now explores...
    assert {"v_total", "subject_variance_share"}.issubset(free)
    # ...and the two original scales are neither sampled nor lost.
    assert {"tau_subject", "kappa_excess_young"}.issubset(deterministics)
    assert {"tau_subject", "kappa_excess_young"}.isdisjoint(free)
    # Downstream names that summaries and comparators read must survive.
    assert {"kappa_young", "kappa_old", "a_kappa", "b_kappa"}.issubset(deterministics)


def test_the_partition_algebra_round_trips(tmp_path, monkeypatch):
    """Draws of the budget and share reproduce the two scales they encode."""
    model = _build(PARTITIONED, tmp_path, monkeypatch)
    p0 = _TD_UNDERSTOOD_VARIANCE_PARTITION.reference_proportion
    c = 1.0 / (p0 * (1.0 - p0))
    with model:
        v, s, tau, exc = pm.draw(
            [
                model["v_total"],
                model["subject_variance_share"],
                model["tau_subject"],
                model["kappa_excess_young"],
            ],
            draws=256,
            random_seed=3,
        )
    assert np.allclose(tau, np.sqrt(s * v), rtol=1e-10)
    assert np.allclose(exc, c / ((1 - s) * v), rtol=1e-10)
    # Both stay strictly positive, which is what makes kappa_young well-defined.
    assert np.all(tau > 0) and np.all(exc > 0)


def test_partition_without_subject_re_is_rejected(tmp_path, monkeypatch):
    broken = dataclasses.replace(PARTITIONED, use_subject_re=False)
    with pytest.raises(ValueError, match="use_subject_re is False"):
        _build(broken, tmp_path, monkeypatch)


def test_partition_requires_the_anchored_kappa_form():
    """The budget allocates the two-anchor form's young anchor; the legacy
    parameterisation has no such quantity, so the pairing is rejected at
    configuration time rather than producing a misleading graph."""
    from vocab_growth.models.common import ModelConfiguration
    import preliz as pz

    with pytest.raises(ValueError, match="kappa_anchored must be configured"):
        ModelConfiguration(
            slope_anchors=(12, 26),
            ell_months_range=(6, 18),
            p_slope_low_dist=pz.Beta(alpha=1.2, beta=8.0),
            p_slope_hi_dist=pz.Beta(alpha=1.3, beta=1.3),
            ell_unit_dist=pz.Beta(alpha=3.0, beta=3.0),
            eta_dist=pz.HalfNormal(sigma=1.0),
            n_plot=10,
            ages_query=[12],
            kappa_min_dist=pz.LogNormal(mu=0.0, sigma=1.0),
            a_kappa_dist=pz.Normal(mu=0.0, sigma=1.0),
            b_kappa_mag_dist=pz.HalfNormal(sigma=1.0),
            variance_partition_total_dist=pz.LogNormal(mu=0.0, sigma=0.8),
            variance_partition_share_dist=pz.Beta(alpha=3.9, beta=2.1),
            variance_partition_reference_proportion=0.1041,
        )


def test_partition_fields_must_be_set_together():
    from vocab_growth.models.common import ModelConfiguration
    import preliz as pz

    with pytest.raises(ValueError, match="must be set together"):
        ModelConfiguration(
            slope_anchors=(12, 26),
            ell_months_range=(6, 18),
            p_slope_low_dist=pz.Beta(alpha=1.2, beta=8.0),
            p_slope_hi_dist=pz.Beta(alpha=1.3, beta=1.3),
            ell_unit_dist=pz.Beta(alpha=3.0, beta=3.0),
            eta_dist=pz.HalfNormal(sigma=1.0),
            n_plot=10,
            ages_query=[12],
            kappa_min_dist=pz.LogNormal(mu=0.0, sigma=1.0),
            a_kappa_dist=pz.Normal(mu=0.0, sigma=1.0),
            b_kappa_mag_dist=pz.HalfNormal(sigma=1.0),
            variance_partition_total_dist=pz.LogNormal(mu=0.0, sigma=0.8),
        )


@pytest.mark.parametrize("p0", [0.0, 1.0, -0.1, 1.5])
def test_reference_proportion_must_be_a_proportion(p0):
    from vocab_growth.models.gp_utils import build_variance_partition
    import preliz as pz

    with pm.Model():
        with pytest.raises(ValueError, match="strictly in"):
            build_variance_partition(
                pz.LogNormal(mu=0.0, sigma=0.8),
                pz.Beta(alpha=3.9, beta=2.1),
                reference_proportion=p0,
                subject_scale_name="tau_subject",
            )


def test_induced_marginals_stay_near_the_priors_they_replace():
    """The calibration claim in the definition comment, checked rather than trusted.

    The dispersion marginal should stay close to ``LogNormal(log 40, 0.9)``; the
    subject marginal is expected to be *tighter* than ``HalfNormal(1.5)``, because
    a shared budget cannot let both range over 30x independently.
    """
    vp = _TD_UNDERSTOOD_VARIANCE_PARTITION
    c = 1.0 / (vp.reference_proportion * (1 - vp.reference_proportion))
    rng = np.random.default_rng(20260805)
    n = 200_000
    v = rng.lognormal(vp.total_mu, vp.total_sigma, n)
    s = rng.beta(vp.share_alpha, vp.share_beta, n)
    tau = np.sqrt(s * v)
    exc = c / ((1 - s) * v)

    exc_q = np.quantile(exc, [0.05, 0.5, 0.95])
    # Against LogNormal(log 40, 0.9): 9.10 / 40.0 / 175.8.
    assert 5.0 < exc_q[0] < 12.0
    assert 28.0 < exc_q[1] < 48.0
    assert 150.0 < exc_q[2] < 300.0

    tau_q = np.quantile(tau, [0.05, 0.5, 0.95])
    assert 0.3 < tau_q[0] < 0.5
    assert 0.65 < tau_q[1] < 0.95
    assert 1.3 < tau_q[2] < 1.9

    # The share prior must not assert the split: VG12's posterior implies 0.598,
    # and it should sit in the prior's central mass, not its tail.
    from scipy import stats

    cdf = stats.beta.cdf(0.598, vp.share_alpha, vp.share_beta)
    assert 0.2 < cdf < 0.6, f"share prior puts the fitted split at CDF {cdf:.3f}"
