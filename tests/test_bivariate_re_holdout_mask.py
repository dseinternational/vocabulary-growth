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

import vocab_growth.models.common_bivariate as common_bivariate
from vocab_growth.models.common import ModelFitContext
from vocab_growth.models.common_bivariate import (
    configure_bivariate_priors,
    extract_model_samples,
    sample_posterior_predictive,
)
from vocab_growth.models.common_bivariate_re import build_model_re
from vocab_growth.models.definitions import VG07


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
    monkeypatch.setattr(common_bivariate, "_plot_and_print_dist", lambda *a, **k: None)
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
