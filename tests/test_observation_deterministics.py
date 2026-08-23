# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The sampler does not store the observation-sized deterministics.

Since 2026-08-23 the engines' ``sample`` stage passes ``var_names`` from
``fit_artifacts.sampled_variable_names`` to ``pm.sample``, so nutpie never
evaluates or stores ``f_obs``/``p_obs``/``kappa_obs``, their per-outcome
counterparts or the concatenated ``*_all`` grids — the variables that made fit
memory scale as ``n_obs x draws``. These tests pin what that must and must not
change: which names are excluded (a pure rule on dimensions), that the graph and
therefore the draws are untouched, that the trace and the manifest record what
was left out, and that ``posterior_recompute.with_deterministics`` rebuilds an
excluded variable exactly from the stored free parameters.

The end-to-end tests build the real VG07 model on synthetic data and sample it
with nutpie for a handful of draws; the point is exactness, not convergence.
"""

import json
import os
from dataclasses import fields as dc_fields

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
from vocab_growth.fit_artifacts import (
    NOT_SAMPLED_ATTR,
    plan_trace_persistence,
    read_not_sampled_attr,
    sampled_variable_names,
    save_trace,
    unsampled_deterministic_names,
)
from vocab_growth.models.common import ModelFitContext, ModelSamples, sample
from vocab_growth.models.common_bivariate import (
    BivariateModelSamples,
    configure_bivariate_priors,
)
from vocab_growth.models.common_bivariate_re import build_model_re
from vocab_growth.models.common_trivariate import TrivariateModelSamples
from vocab_growth.models.definitions import VG07
from vocab_growth.posterior_recompute import (
    missing_deterministics,
    with_deterministics,
)


class _NoopDigraph:
    def render(self, *args, **kwargs):
        return None


def _as_dataset(node):
    return node if isinstance(node, xr.Dataset) else node.to_dataset()


# --- the rule ------------------------------------------------------------------


def _toy_model():
    coords = {
        "obs_id": range(5),
        "all_id": range(8),
        "plot_id": range(3),
        "study_id": ["a", "b"],
    }
    with pm.Model(coords=coords) as m:
        mu = pm.Normal("mu", 0.0, 1.0)
        tau = pm.HalfNormal("tau", 1.0)
        delta_raw = pm.Normal("delta_raw", 0.0, 1.0, dims="study_id")
        pm.Deterministic("delta", tau * delta_raw, dims="study_id")
        pm.Deterministic("f_all", mu + pm.math.zeros(8), dims="all_id")
        pm.Deterministic("f_obs", mu + pm.math.zeros(5), dims="obs_id")
        pm.Deterministic("f_plot", mu + pm.math.zeros(3), dims="plot_id")
        pm.Deterministic("scale", tau * 2.0)
        pm.Deterministic("anon", mu * 3.0)  # no dims at all
    return m


def test_unsampled_names_are_exactly_the_observation_and_all_grid_deterministics():
    assert unsampled_deterministic_names(_toy_model()) == ["f_all", "f_obs"]


def test_sampled_names_keep_every_free_rv_and_the_other_deterministics():
    names = sampled_variable_names(_toy_model())
    assert names[:3] == ["mu", "tau", "delta_raw"]  # free RVs, model order
    assert set(names[3:]) == {"delta", "f_plot", "scale", "anon"}
    assert not {"f_obs", "f_all"} & set(names)


@pytest.mark.parametrize("struct", [ModelSamples, BivariateModelSamples, TrivariateModelSamples])
def test_sample_structs_carry_no_observation_level_posterior(struct):
    # Nothing read these fields; the extractors no longer populate them, so the
    # structs must not declare them either (a declared field would make the
    # extractor's omission a construction error).
    names = {f.name for f in dc_fields(struct)}
    forbidden = {
        "X_obs_z", "f_obs", "p_obs", "f_u_obs", "p_u_obs", "h_obs", "q_obs",
        "f_s_obs", "p_s_obs", "f_sign_obs", "p_sign_obs",
    }
    assert not names & forbidden
    # The observed counts and the constant age grid are data, not posterior, and stay.
    assert "X_obs" in names


# --- end to end on the real VG07 graph ------------------------------------------


def _vg07_context(tmp_path, monkeypatch, *, seed=11):
    monkeypatch.setattr(common_bivariate, "_plot_and_print_dist", lambda *a, **k: None)
    monkeypatch.setattr(
        pymc_utils, "model_to_graphviz", lambda model: _NoopDigraph(), raising=False
    )
    n = 24
    ages = np.linspace(10.0, 90.0, n)
    understood = np.round(ages * 5.0)
    spoken = np.round(ages * 3.0)
    analysis_df = pd.DataFrame(
        {
            "age": ages,
            "understood": understood,
            "spoken": spoken,
            "study": ["study_a"] * (n // 2) + ["study_b"] * (n - n // 2),
            "study_code": [0] * (n // 2) + [1] * (n - n // 2),
            "subject_code": np.repeat(np.arange(n // 2), 2),
        }
    )
    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name="TEST_VG07_OBSDET",
            config_name="test",
            output_root_dir=str(tmp_path),
            ci_prob=0.89,
            interval_kind="eti",
        ),
        # Tiny: exactness is the claim under test, not convergence.
        sampling=sampling.SamplingConfiguration(
            draws=12, tune=12, chains=2, cores=1, target_accept=0.8, random_seed=seed
        ),
    )
    os.makedirs(context.reporting.output_dir, exist_ok=True)
    bmd = model_data.BinomialModelData(
        X_obs=ages.reshape(-1, 1),
        y_obs=np.zeros(n, dtype=int),
        n_trials=VG07.n_trials,
    )
    context.set_model_data(bmd, analysis_df)
    configure_bivariate_priors(context, VG07)
    build_model_re(context, VG07)
    return context


@pytest.fixture(scope="module")
def two_fits(tmp_path_factory):
    """The same VG07 fit twice: once storing everything, once not."""
    # A module-scoped monkeypatch, since the fixture is shared by several tests.
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    try:
        root = tmp_path_factory.mktemp("obsdet")
        lean = _vg07_context(root / "lean", mp)
        sample(lean)
        full = _vg07_context(root / "full", mp)
        sample(full, store_observation_deterministics=True)
        yield lean, full
    finally:
        mp.undo()


def test_lean_fit_stores_no_observation_deterministics(two_fits):
    lean, full = two_fits
    excluded = unsampled_deterministic_names(lean.model)
    assert excluded, "VG07 defines observation-sized deterministics"
    lean_post = _as_dataset(lean.trace.posterior)
    full_post = _as_dataset(full.trace.posterior)
    assert not set(excluded) & set(lean_post.data_vars)
    assert set(excluded) <= set(full_post.data_vars)
    # Everything else is stored in both: free RVs and the grid deterministics.
    kept = set(sampled_variable_names(lean.model))
    assert kept <= set(lean_post.data_vars)
    assert kept <= set(full_post.data_vars)


def test_not_storing_them_does_not_change_the_draws(two_fits):
    lean, full = two_fits
    lean_post = _as_dataset(lean.trace.posterior)
    full_post = _as_dataset(full.trace.posterior)
    # The posterior itself is bit-identical: the sampler's trajectory depends
    # only on the logp graph, which var_names does not touch.
    for rv in lean.model.free_RVs:
        np.testing.assert_array_equal(
            lean_post[rv.name].values, full_post[rv.name].values, err_msg=rv.name
        )
    # The stored grid deterministics agree to floating-point rounding, not to
    # the bit: nutpie compiles a different "expand" function when fewer outputs
    # are requested, and PyTensor's loop fusion then rounds the last ulp
    # differently (measured at 4.4e-16 absolute, 1.1e-14 relative on this
    # graph). That is the precision reported tables are compared at, not the
    # precision they are printed at, and it is why the equivalence claim is
    # "identical draws, derived quantities to rounding".
    for name in ("f_u_plot", "p_u_query", "h_plot", "kappa_u_query"):
        np.testing.assert_allclose(
            lean_post[name].values, full_post[name].values, rtol=1e-12, atol=1e-12,
            err_msg=name,
        )


def test_trace_records_what_was_not_sampled(two_fits):
    lean, full = two_fits
    excluded = unsampled_deterministic_names(lean.model)
    assert json.loads(_as_dataset(lean.trace.posterior).attrs[NOT_SAMPLED_ATTR]) == excluded
    assert read_not_sampled_attr(lean.trace) == excluded
    assert read_not_sampled_attr(full.trace) == []


def test_recomputed_deterministic_matches_the_stored_one(two_fits):
    lean, full = two_fits
    lean_post = _as_dataset(lean.trace.posterior)
    full_post = _as_dataset(full.trace.posterior)
    names = ["f_u_obs", "h_obs", "kappa_u_obs", "kappa_s_obs"]
    assert missing_deterministics(lean_post, names) == names
    rebuilt = with_deterministics(lean_post, lean.model, names)
    for name in names:
        np.testing.assert_allclose(
            rebuilt[name].values, full_post[name].values, rtol=1e-10, atol=1e-12,
            err_msg=name,
        )
    # Present names are left alone, and the input is not modified.
    assert with_deterministics(full_post, full.model, names) is full_post
    assert missing_deterministics(lean_post, names) == names


def test_recompute_refuses_names_the_model_does_not_define(two_fits):
    lean, _ = two_fits
    with pytest.raises(KeyError, match="no_such_deterministic"):
        with_deterministics(_as_dataset(lean.trace.posterior), lean.model, ["no_such_deterministic"])


def test_save_trace_records_not_sampled_and_compact_finds_nothing_observation_sized(
    two_fits, tmp_path
):
    lean, _ = two_fits
    excluded = unsampled_deterministic_names(lean.model)
    record = save_trace(lean.trace, str(tmp_path), persistence="full")
    assert record["not_sampled"] == excluded
    assert record["dropped"] == {}
    # Compact has only the duplicated scaled effects left to drop: the
    # observation-sized variables are already absent.
    plan = plan_trace_persistence(lean.trace, "compact")
    assert not set(excluded) & set(plan.get("posterior", []))


def test_recompute_works_on_a_thinned_posterior(two_fits):
    # loso_compare thins before recomputing; compute_deterministics relabels the
    # draw axis 0..n-1, so without carrying the input's labels over the exact
    # merge refuses (the real-fit check that found this: draw labels 0, 36, ...).
    lean, full = two_fits
    lean_post = _as_dataset(lean.trace.posterior).isel(draw=slice(0, None, 5))
    full_post = _as_dataset(full.trace.posterior).isel(draw=slice(0, None, 5))
    rebuilt = with_deterministics(lean_post, lean.model, ["f_u_obs", "kappa_s_obs"])
    np.testing.assert_array_equal(rebuilt["draw"].values, lean_post["draw"].values)
    np.testing.assert_allclose(
        rebuilt["f_u_obs"].values, full_post["f_u_obs"].values, rtol=1e-10, atol=1e-12
    )
