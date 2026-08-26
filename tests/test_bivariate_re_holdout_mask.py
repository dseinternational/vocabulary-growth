# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for the bivariate-RE holdout-mask alignment (issue #67).

When a ``holdout`` column is present, ``build_model_re`` fits the likelihood on
the training rows only, so the stored ``obs_u_mask`` / ``obs_s_mask`` must be the
*training* masks (full observed mask minus holdout rows) to stay aligned with
``observed_data`` when ``extract_model_samples`` scatters it back to full length.

These tests build the real VG07 (study-RE) model on synthetic data with a holdout
column. To exercise the genuine extraction path without a full MCMC fit, the
prior-predictive draws stand in as the posterior and are fed through the real
``sample_posterior_predictive`` pipeline step (which calls ``extract_model_samples``).
No graphviz / ``dot`` binary is required.
"""

import os
from dataclasses import fields as dc_fields
from dataclasses import replace

import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.pymc_utils as pymc_utils
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from vocab_growth.models.common import ModelFitContext
from vocab_growth.models.common_bivariate import (
    configure_bivariate_priors,
    extract_model_samples,
    sample_posterior_predictive,
)
from vocab_growth.models.common_bivariate_re import build_model_re
from vocab_growth.models.definitions import (
    VG07,
    BivariateChildSlopeModelDefinition,
    SubjectSlopePriorParams,
)

# Every test here builds a VG07 graph and runs prior- and posterior-predictive
# sampling through it. Minutes, not seconds, so deselected unless `-m ""`
# asks for it.
pytestmark = pytest.mark.slow


class _NoopDigraph:
    """Stand-in for a graphviz Digraph so the build needs no ``dot`` binary."""

    def render(self, *args, **kwargs):
        return None


def _as_dataset(node):
    """Return the xarray ``Dataset`` for an InferenceData/DataTree group node."""
    return node if isinstance(node, xr.Dataset) else node.to_dataset()


def _build_holdout_model(tmp_path, monkeypatch, definition=VG07):
    """Build the real VG07 model on synthetic data carrying a ``holdout`` column.

    Returns the populated context plus the synthetic ``understood`` / ``spoken``
    arrays (integer-valued, as the model stores them) and the ``holdout`` mask.
    """
    # Keep the build hermetic and fast: skip the per-prior diagnostic plots and,
    # if this build variant renders a model graph, the graphviz ``dot`` call.
    monkeypatch.setattr(
        pymc_utils, "model_to_graphviz", lambda model: _NoopDigraph(), raising=False
    )

    n = 24
    ages = np.linspace(10.0, 90.0, n)
    # Integer counts so the model's ``astype(int)`` round-trips exactly.
    understood = np.round(ages * 5.0)
    spoken = np.round(ages * 3.0)

    # Partial observations so has_u, has_s and holdout are all distinct masks.
    understood[[2, 3]] = np.nan  # spoken-only rows
    spoken[[0, 1]] = np.nan  # understood-only rows

    holdout = np.zeros(n, dtype=bool)
    holdout[[0, 2, 20, 21]] = True  # mix of partial + both-observed rows

    analysis_df = pd.DataFrame(
        {
            "age": ages,
            "understood": understood,
            "spoken": spoken,
            "study": ["study_a"] * (n // 2) + ["study_b"] * (n - n // 2),
            "study_code": [0] * (n // 2) + [1] * (n - n // 2),
            "subject_code": np.repeat(np.arange(n // 2), 2),
            "holdout": holdout,
        }
    )

    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name="TEST_VG07_HOLDOUT",
            config_name="test",
            output_root_dir=str(tmp_path),
            ci_prob=0.90,
            interval_kind="hdi",
        ),
        sampling=sampling.get_sampling_configuration("test"),
    )
    os.makedirs(context.reporting.output_dir, exist_ok=True)

    bmd = model_data.BinomialModelData(
        X_obs=ages.reshape(-1, 1),
        y_obs=np.zeros(n, dtype=int),
        n_trials=definition.n_trials,
    )
    context.set_model_data(bmd, analysis_df)
    configure_bivariate_priors(context, definition)
    build_model_re(context, definition)

    return context, understood, spoken, holdout


def _prior_as_posterior_trace(context):
    """Sample the prior predictive and reshape it into a fit-like trace (prior
    draws standing in as the posterior) for the extraction pipeline."""
    with context.model:
        prior = pm.sample_prior_predictive(draws=4, random_seed=0)
    return xr.DataTree.from_dict(
        {
            "posterior": _as_dataset(prior["prior"]),
            "constant_data": _as_dataset(prior["constant_data"]),
            "observed_data": _as_dataset(prior["observed_data"]),
        }
    )


def test_holdout_masks_round_trip_through_extract_model_samples(tmp_path, monkeypatch):
    context, understood, spoken, holdout = _build_holdout_model(tmp_path, monkeypatch)

    has_u = ~np.isnan(understood)
    has_s = ~np.isnan(spoken)
    has_u_train = has_u & ~holdout
    has_s_train = has_s & ~holdout

    context.set_trace(_prior_as_posterior_trace(context))
    # Builds the y_*_plot / y_*_query predictive nodes and calls the real
    # extract_model_samples; pre-fix this raises on the misaligned scatter.
    sample_posterior_predictive(context, VG07)
    samples = context.model_samples

    # Stored masks are the *training* masks (holdout rows removed).
    np.testing.assert_array_equal(samples.obs_u_mask, has_u_train)
    np.testing.assert_array_equal(samples.obs_s_mask, has_s_train)
    assert int(samples.obs_u_mask.sum()) < int(has_u.sum())
    assert int(samples.obs_s_mask.sum()) < int(has_s.sum())

    # Observed counts land at exactly the training rows; everything else is NaN.
    np.testing.assert_array_equal(np.isnan(samples.y_u_obs), ~has_u_train)
    np.testing.assert_array_equal(np.isnan(samples.y_s_obs), ~has_s_train)
    np.testing.assert_array_equal(samples.y_u_obs[has_u_train], understood[has_u_train])
    np.testing.assert_array_equal(samples.y_s_obs[has_s_train], spoken[has_s_train])


def test_paired_only_masks_mark_the_likelihood_rows(tmp_path, monkeypatch):
    """Under paired-only the marginal spoken rows leave the likelihood, so the
    stored ``obs_s_mask`` must shrink with them (issue #266 finding 3).

    Calibration and extraction both read ``obs_s_mask`` as "the rows
    ``y_s_obs`` covers"; storing the unfiltered mask made every paired-only
    fit fail inside ``sample_posterior_predictive`` — after sampling, before
    the trace was saved. This runs the same real pipeline step the fits do.
    """
    from vocab_growth.models.likelihood_utils import SPOKEN_FALLBACK_PAIRED_ONLY

    definition = replace(VG07, spoken_fallback=SPOKEN_FALLBACK_PAIRED_ONLY)
    context, understood, spoken, holdout = _build_holdout_model(
        tmp_path, monkeypatch, definition
    )

    has_u = ~np.isnan(understood)
    has_s = ~np.isnan(spoken)
    # Paired-only keeps only the conditional rows: spoken observed, understood
    # observed, and not held out.
    has_s_likelihood = has_s & has_u & ~holdout

    context.set_trace(_prior_as_posterior_trace(context))
    sample_posterior_predictive(context, definition)
    samples = context.model_samples

    np.testing.assert_array_equal(samples.obs_s_mask, has_s_likelihood)
    # The synthetic frame carries a spoken-only row outside the holdout, so the
    # treatment genuinely dropped something here.
    assert int(samples.obs_s_mask.sum()) < int((has_s & ~holdout).sum())
    np.testing.assert_array_equal(np.isnan(samples.y_s_obs), ~has_s_likelihood)
    np.testing.assert_array_equal(
        samples.y_s_obs[has_s_likelihood], spoken[has_s_likelihood]
    )


def test_subject_marginal_predictive_uses_one_new_subject_per_draw(tmp_path, monkeypatch):
    definition = replace(VG07, use_subject_re_u=True, use_subject_re_q=True)
    context, *_ = _build_holdout_model(tmp_path, monkeypatch, definition)

    context.set_trace(_prior_as_posterior_trace(context))
    sample_posterior_predictive(context, definition)

    assert "_delta_subj_u_plot_marg" not in context.model.named_vars
    assert "_delta_subj_u_query_marg" not in context.model.named_vars
    assert "_delta_subj_q_plot_marg" not in context.model.named_vars
    assert "_delta_subj_q_query_marg" not in context.model.named_vars
    assert context.model.named_vars["_delta_subj_u_marg"].ndim == 0
    assert context.model.named_vars["_delta_subj_q_marg"].ndim == 0


def test_child_slope_predictive_draws_an_intercept_and_a_slope(tmp_path, monkeypatch):
    """VG19's unseen child is a (b0, b1) pair, not one deviate scaled by a curve.

    The distinction is the whole point of the structure. A1 scales a single
    deviate by tau(age), which makes children's ranks identical at every age --
    they never cross. Two deviates per outcome let the trajectory fan and let one
    child overtake another, which is what a random slope means.
    """
    values = {f.name: getattr(VG07, f.name) for f in dc_fields(VG07)}
    values.update(
        use_subject_re_u=True,
        use_subject_re_q=True,
        subject_slope_ref_age_months=36.0,
        tau_subj_u_sigma=SubjectSlopePriorParams(tau0_sigma=0.9, tau1_sigma=0.5),
        tau_subj_q_sigma=SubjectSlopePriorParams(tau0_sigma=1.2, tau1_sigma=0.5),
    )
    definition = BivariateChildSlopeModelDefinition(**values)
    context, *_ = _build_holdout_model(tmp_path, monkeypatch, definition)

    context.set_trace(_prior_as_posterior_trace(context))
    sample_posterior_predictive(context, definition)

    named = context.model.named_vars
    # Two standard deviates per outcome, both scalar: one child per draw.
    for tag in ("subj_u", "subj_q"):
        for k in ("0", "1"):
            assert named[f"_z{k}_{tag}_marg"].ndim == 0
    # And NOT the constant-offset or A1 nodes, which would be the wrong child.
    assert "_delta_subj_u_marg" not in named
    assert "_delta_subj_q_marg" not in named
    assert "_z_subj_u_marg" not in named
    assert "_z_subj_q_marg" not in named


def test_extract_model_samples_guards_against_misaligned_mask(tmp_path, monkeypatch):
    context, *_ = _build_holdout_model(tmp_path, monkeypatch)
    trace = _prior_as_posterior_trace(context)

    # Reintroduce the pre-fix condition: mark every row as understood-observed so
    # the stored mask count exceeds the training-length observed data. The guard
    # fires before the (here absent) posterior-predictive nodes are read.
    bad_const = _as_dataset(trace["constant_data"]).copy(deep=True)
    bad_const["obs_u_mask"].values[:] = 1
    bad_trace = xr.DataTree.from_dict(
        {
            "posterior": _as_dataset(trace["posterior"]),
            "constant_data": bad_const,
            "observed_data": _as_dataset(trace["observed_data"]),
        }
    )

    with pytest.raises(ValueError, match="#67"):
        extract_model_samples(bad_trace)
