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

import ast
import importlib
import inspect
import json
import os
from dataclasses import fields as dc_fields
from pathlib import Path

import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.pymc_utils as pymc_utils
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from vocab_growth.fit_artifacts import (
    NOT_SAMPLED_ATTR,
    plan_trace_persistence,
    read_not_sampled_attr,
    sampled_variable_names,
    save_trace,
    unsampled_deterministic_names,
)
from vocab_growth.models import catalogue
from vocab_growth.models.common import ModelFitContext, ModelSamples, sample
from vocab_growth.models.common_bivariate import (
    BivariateModelSamples,
    configure_bivariate_priors,
)
from vocab_growth.models.common_bivariate_re import build_model_re
from vocab_growth.models.common_joint_modality import JointModelSamples
from vocab_growth.models.common_trivariate import TrivariateModelSamples
from vocab_growth.models.definitions import VG07
from vocab_growth.posterior_recompute import (
    missing_deterministics,
    with_deterministics,
)

# The `two_fits` fixture is two real nutpie fits of the same model, which is the
# only way to show the draws are unchanged. Minutes, not seconds, so every test
# that draws it is marked `slow` and deselected unless `-m "slow or not slow"`
# asks for it.
#
# The rule tests above it are pure -- a toy graph, the sample structs' declared
# fields, and an AST walk over the engine sources -- and are deliberately *not*
# marked, so they run in the fast job. Until issue #273 a module-level
# `pytestmark` made the whole file slow, which is why the struct contract that
# should have caught VG14's `g_sign_obs` and `r_obs` was not running on any
# pull request even after it was extended.
_slow = pytest.mark.slow


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


@pytest.mark.parametrize(
    "struct",
    [ModelSamples, BivariateModelSamples, TrivariateModelSamples, JointModelSamples],
)
def test_sample_structs_carry_no_observation_level_posterior(struct):
    # Nothing read these fields; the extractors no longer populate them, so the
    # structs must not declare them either (a declared field would make the
    # extractor's omission a construction error).
    #
    # `g_sign_obs` and `r_obs` were the two that survived the 2026-08-23 sweep:
    # VG14 alone carries them, they were not in this list, and a fresh default
    # VG14 fit therefore raised KeyError in `extract_model_samples` after
    # posterior-predictive sampling -- hours into a fit, on a model no CI job
    # samples (issue #273). The general rule is pinned by
    # `test_no_extractor_reads_an_observation_dimensioned_posterior` below;
    # this list is the readable statement of it.
    names = {f.name for f in dc_fields(struct)}
    forbidden = {
        "X_obs_z", "f_obs", "p_obs", "f_u_obs", "p_u_obs", "h_obs", "q_obs",
        "f_s_obs", "p_s_obs", "f_sign_obs", "p_sign_obs",
        "g_sign_obs", "r_obs",
    }
    assert not names & forbidden
    # The observed counts and the constant age grid are data, not posterior, and
    # stay. The joint engine carries no observation grid at all.
    if struct is not JointModelSamples:
        assert "X_obs" in names


#: Trace dimensions the sampler is told not to store, so nothing may read a
#: posterior variable indexed by one. Mirrors `fit_artifacts._is_recomputable_dim`.
_UNSTORED_DIMS = {"obs_id", "all_id"}

_ENGINE_MODULES = sorted(
    {engine.module for engine in catalogue.ENGINES.values()}
)


@pytest.mark.parametrize("module_name", _ENGINE_MODULES)
def test_no_extractor_reads_an_observation_dimensioned_posterior(module_name):
    """No engine may read back a deterministic the sampler was told to skip.

    The rule is on the *dimension*, so this is checked as a rule rather than as
    a list of names: any ``extract_posterior(trace, <name>, "obs_id")`` reads
    ``trace.posterior[<name>]`` for a variable ``pm.sample`` never stored, and
    raises ``KeyError`` -- after sampling and after posterior prediction, which
    on a reporting fit is hours in. Read from the source rather than from a fit,
    so it costs nothing and covers every engine including the ones no CI job
    samples.

    ``posterior_predictive`` is a different group and is stored in full, so
    ``extract_posterior_predictive`` on ``obs_id`` is fine and is not matched
    here.
    """
    tree = ast.parse(
        (Path(inspect.getfile(importlib.import_module(module_name)))).read_text(
            encoding="utf-8"
        )
    )
    offences = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in {"extract_posterior", "_extract_posterior"}:
            continue
        dim = node.args[-1]
        if isinstance(dim, ast.Constant) and dim.value in _UNSTORED_DIMS:
            variable = node.args[1]
            offences.append(
                (getattr(variable, "value", "<expr>"), dim.value, node.lineno)
            )
    assert not offences, (
        f"{module_name} reads posterior variables the sampler does not store: "
        + ", ".join(f"{n!r} on {d} (line {ln})" for n, d, ln in offences)
        + ". Drop the read and the struct field, or recompute it with "
        "vocab_growth.posterior_recompute."
    )


# --- end to end on the real VG07 graph ------------------------------------------


def _vg07_context(tmp_path, monkeypatch, *, seed=11):
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


@_slow
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


@_slow
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


@_slow
def test_trace_records_what_was_not_sampled(two_fits):
    lean, full = two_fits
    excluded = unsampled_deterministic_names(lean.model)
    assert json.loads(_as_dataset(lean.trace.posterior).attrs[NOT_SAMPLED_ATTR]) == excluded
    assert read_not_sampled_attr(lean.trace) == excluded
    assert read_not_sampled_attr(full.trace) == []


@_slow
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


@_slow
def test_recompute_refuses_names_the_model_does_not_define(two_fits):
    lean, _ = two_fits
    with pytest.raises(KeyError, match="no_such_deterministic"):
        with_deterministics(_as_dataset(lean.trace.posterior), lean.model, ["no_such_deterministic"])


@_slow
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


@_slow
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


@_slow
def test_trace_records_the_sampled_parameters(two_fits):
    # What loo_reff pins PSIS-LOO's relative efficiency to, readable from the
    # stored trace without the model.
    from vocab_growth.fit_artifacts import read_sampled_parameters_attr

    lean, full = two_fits
    expected = [rv.name for rv in lean.model.free_RVs]
    assert read_sampled_parameters_attr(lean.trace) == expected
    assert read_sampled_parameters_attr(full.trace) == expected
