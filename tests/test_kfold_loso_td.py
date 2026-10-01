# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Held-out validation of the typically developing models in ``kfold_loso`` (#240).

The typically developing path refits a fold with whole children or a whole
study out of the likelihood and scores them with their effects integrated per
posterior draw. What is pinned here, without sampling:

- each model's frame is the one its engine fits, and the fold frames leak
  nothing: no held-out child or study contributes a likelihood row, and a
  held-out study leaves the zero-sum study block;
- the NumPy Beta-Binomial the scorer evaluates is the PyMC model's own
  pointwise log density at a fixed parameter point, on both engines, and the
  fixed part of each logit is decomposed exactly as the graph composes it;
- the adaptive quadrature integrates what it claims to;
- what the scorer does not implement is refused before any fit.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.special import expit, logsumexp
from scipy.stats import betabinom, norm

from vocab_growth.fold_fits import build_holdout_fold_context
from vocab_growth.models.definitions import MODEL_REGISTRY, VG10, VG12, VG21, VG23
from vocab_growth.sensitivity.registry import build_variant

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "kfold_loso.py"
_SPEC = importlib.util.spec_from_file_location("kfold_loso_td_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
KL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = KL
_SPEC.loader.exec_module(KL)

TD_KEYS = ("VG11", "VG12", "VG21", "VG23", "VG26")


# ---------------------------------------------------------------------------
# Small stand-ins: the same engines and graph features on a fraction of the
# children, so a build costs seconds.
# ---------------------------------------------------------------------------

SMALL_VG12 = dataclasses.replace(VG12, sample_fraction=0.04, min_study_observations=20)
SMALL_VG21 = dataclasses.replace(VG21, sample_fraction=0.04, min_study_observations=20)
SMALL_VG23 = dataclasses.replace(VG23, sample_fraction=0.04, min_study_observations=20)


def _small_frame(definition):
    from vocab_growth.models.common_bivariate_re import (
        build_bivariate_re_analysis_frame,
    )
    from vocab_growth.models.common_univariate_re import (
        build_univariate_re_analysis_frame,
    )

    if KL.td_engine(definition) == "univariate_re":
        frame, _ = build_univariate_re_analysis_frame(definition)
    else:
        frame, _ = build_bivariate_re_analysis_frame(definition)
    return frame


def _build(definition, fold_frame, tmp_path):
    return build_holdout_fold_context(
        definition,
        fold_frame,
        sampling.get_sampling_configuration("dev"),
        label="test",
        tmp_root=str(tmp_path),
        name_prefix="KFOLD-TEST",
    )


def _jittered_point(model, seed=3):
    rng = np.random.default_rng(seed)
    return {
        name: value + rng.normal(0.0, 0.25, size=np.shape(value))
        for name, value in model.initial_point().items()
    }


def _evaluate(model, point, names):
    names = [name for name in names if name in model.named_vars]
    outs = model.replace_rvs_by_values([model[name] for name in names])
    fn = model.compile_fn(outs, inputs=model.value_vars, on_unused_input="ignore")
    return dict(zip(names, fn(point), strict=True))


def _observed_values(model, name):
    return np.asarray(model.rvs_to_values[model[name]].data)


def _pointwise_logp(model, point, name):
    fn = model.compile_logp(vars=[model[name]], sum=False)
    return np.asarray(fn(point)[0], dtype=float)


def _one_draw_trace(model, values):
    """A posterior of one draw holding ``values`` under the model's own dims."""
    data = {}
    for name, value in values.items():
        dims = model.named_vars_to_dims.get(name) or ()
        data[name] = (("chain", "draw", *dims), np.asarray(value, dtype=float)[None, None])
    return xr.DataTree.from_dict({"posterior": xr.Dataset(data)})


_UNIVARIATE_NAMES = (
    "f_obs", "p_obs", "kappa_obs", "delta_subject", "delta", "tau_subject", "tau",
    "beta_sex",
)
_BIVARIATE_NAMES = (
    "f_u_obs", "h_obs", "p_u_obs", "q_obs", "kappa_u_obs", "kappa_s_obs",
    "delta_subj_u", "delta_subj_q", "delta_u", "delta_q", "tau_subj_u",
    "tau_subj_q", "tau_u", "tau_q", "beta_sex_u", "beta_sex_q",
)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def test_the_td_models_are_the_five_reference_models():
    assert set(KL.TD_AVAILABLE) == set(TD_KEYS)
    assert all(KL.is_td_model(d) for d in KL.TD_AVAILABLE.values())
    assert not any(KL.is_td_model(d) for d in KL.DS_AVAILABLE.values())
    for key in TD_KEYS:
        assert KL.td_scoring_refusal(KL.AVAILABLE[key], "subject") is None
        assert KL.td_scoring_refusal(KL.AVAILABLE[key], "study") is None


@pytest.mark.parametrize("key", TD_KEYS)
def test_each_td_frame_is_the_one_its_engine_fits(require_prepared_data, key):
    from vocab_growth.analysis_frames import (
        analysis_frame_hash,
        expected_analysis_frame_hash,
    )

    frame = KL.load_td_frame(key)
    assert analysis_frame_hash(frame) == expected_analysis_frame_hash(
        key.lower(), MODEL_REGISTRY[key.lower()]
    )
    assert frame["study"].nunique() >= 6
    assert (frame["subject_code"].to_numpy() >= 0).all()


def test_models_on_different_frames_cannot_share_a_run(require_prepared_data):
    KL.check_td_run(("VG21", "VG26"), "subject")
    with pytest.raises(SystemExit, match="different prepared frames"):
        KL.check_td_run(("VG21", "VG23"), "subject")


# ---------------------------------------------------------------------------
# Folds: nothing held out reaches the likelihood
# ---------------------------------------------------------------------------


def test_child_folds_partition_the_children_and_leak_nothing(require_prepared_data):
    frame = KL.load_td_frame("VG12")
    folds, _ = KL.stratified_subject_folds(frame, K=5)
    children = np.concatenate(folds)
    assert len(children) == len(set(children)) == frame["subject_code"].nunique()
    for fold in folds:
        fold_frame = KL.build_td_fold_frame(frame, "subject", fold)
        held = fold_frame["holdout"].to_numpy()
        np.testing.assert_array_equal(held, frame["subject_code"].isin(fold).to_numpy())
        training_children = set(fold_frame.loc[~held, "subject_code"])
        assert training_children.isdisjoint(set(fold))
        # The child unit leaves the study coding alone.
        np.testing.assert_array_equal(fold_frame["study_code"], frame["study_code"])


def test_a_study_fold_removes_the_study_from_the_study_block(require_prepared_data):
    frame = KL.load_td_frame("VG21")
    studies = KL.study_folds(frame)
    assert studies == sorted(frame["study"].unique())
    for study in studies:
        fold = KL.build_td_fold_frame(frame, "study", study)
        held = fold["holdout"].to_numpy()
        np.testing.assert_array_equal(held, (frame["study"] == study).to_numpy())
        training = fold.loc[~held]
        assert study not in set(training["study"])
        # The training studies are recoded densely, so the refitted zero-sum
        # block has exactly K - 1 members and none of them is the held-out one.
        assert sorted(training["study_code"].unique()) == list(range(len(studies) - 1))
        assert training.groupby("study")["study_code"].nunique().eq(1).all()
        assert (fold.loc[held, "study_code"] == 0).all()
        # Names, children and the original code are untouched.
        pd.testing.assert_series_equal(fold["study"], frame["study"])
        np.testing.assert_array_equal(fold["study_code_full"], frame["study_code"])
        assert set(training["subject_code"]).isdisjoint(set(fold.loc[held, "subject_code"]))


def test_a_univariate_fold_likelihood_carries_only_training_rows(
    require_prepared_data, tmp_path
):
    frame = _small_frame(SMALL_VG12)
    study = KL.study_folds(frame)[0]
    fold = KL.build_td_fold_frame(frame, "study", study)
    model = _build(SMALL_VG12, fold, tmp_path).model
    train = ~fold["holdout"].to_numpy()
    np.testing.assert_array_equal(
        _observed_values(model, "y_obs"), fold.loc[train, "understood"].to_numpy(dtype=int)
    )
    # Every row stays in observation space, so the held-out rows' curve and
    # dispersion are still evaluated at their ages ...
    assert len(model.coords["obs_id"]) == len(fold)
    # ... but the held-out study is not in the refitted zero-sum block.
    np.testing.assert_array_equal(
        np.asarray(model.coords["study_id"]), np.arange(frame["study"].nunique() - 1)
    )


def test_a_bivariate_fold_likelihood_carries_only_training_rows(
    require_prepared_data, tmp_path
):
    frame = _small_frame(SMALL_VG21)
    folds, _ = KL.stratified_subject_folds(frame, K=3)
    fold = KL.build_td_fold_frame(frame, "subject", folds[0])
    model = _build(SMALL_VG21, fold, tmp_path).model
    train = ~fold["holdout"].to_numpy()
    np.testing.assert_array_equal(
        _observed_values(model, "y_u_obs"),
        fold.loc[train & fold["understood"].notna().to_numpy(), "understood"].to_numpy(dtype=int),
    )
    assert len(_observed_values(model, "y_s_obs")) == int(
        (train & fold["spoken"].notna().to_numpy()).sum()
    )


# ---------------------------------------------------------------------------
# The scorer's density is the model's own
# ---------------------------------------------------------------------------


def test_the_univariate_density_is_the_pymc_models_own(require_prepared_data, tmp_path):
    """At a fixed parameter point, row by row, against ``y_obs``'s own logp.

    The child effect is set to the value the point gives it, so this checks the
    NumPy Beta-Binomial, the age-varying dispersion at each row, the sex
    contrast and the decomposition of the logit together. VG12 carries the
    variance partition, centred study effects and sex.
    """
    frame = _small_frame(SMALL_VG12)
    model = _build(SMALL_VG12, frame, tmp_path).model
    point = _jittered_point(model)
    values = _evaluate(model, point, _UNIVARIATE_NAMES)
    expected = _pointwise_logp(model, point, "y_obs")
    trace = _one_draw_trace(model, values)

    inputs = KL.td_score_inputs(frame.assign(holdout=True), trace, SMALL_VG12, "subject")
    child = values["delta_subject"][frame["subject_code"].to_numpy()[inputs.rows]]
    actual = KL._row_logliks(inputs, child[None, :, None, None], slice(None))[0, :, 0]
    np.testing.assert_allclose(actual, expected[inputs.rows], rtol=1e-9, atol=1e-9)
    # The integrated scale is the child scale alone under the child unit.
    np.testing.assert_allclose(inputs.chol[0, 0, 0], values["tau_subject"])


@pytest.mark.parametrize("definition", [SMALL_VG21, SMALL_VG23], ids=["VG21", "VG23"])
def test_the_bivariate_density_is_the_pymc_models_own(
    require_prepared_data, tmp_path, definition
):
    """Comprehension and nested speech, each against the model's own logp.

    VG23 adds the correlated child block, whose ``q`` effect mixes both raw
    deviates; the Cholesky factor the scorer integrates through must reproduce
    the model's covariance.
    """
    from vocab_growth.models.likelihood_utils import nested_outcome_spec

    frame = _small_frame(definition)
    model = _build(definition, frame, tmp_path).model
    point = _jittered_point(model)
    values = _evaluate(model, point, (*_BIVARIATE_NAMES, "rho_uq"))
    trace = _one_draw_trace(model, values)
    expected_u = np.full(len(frame), np.nan)
    expected_u[np.flatnonzero(frame["understood"].notna())] = _pointwise_logp(
        model, point, "y_u_obs"
    )
    spec = nested_outcome_spec(
        frame, parent_col="understood", outcome_col="spoken", n_trials=definition.n_trials
    )
    expected_s = np.full(len(frame), np.nan)
    expected_s[spec.indices] = _pointwise_logp(model, point, "y_s_obs")

    inputs = KL.td_score_inputs(frame.assign(holdout=True), trace, definition, "subject")
    codes = frame["subject_code"].to_numpy()[inputs.rows]
    effects = np.stack(
        [values["delta_subj_u"][codes], values["delta_subj_q"][codes]], axis=-1
    )[None, :, None, :]
    total = KL._row_logliks(inputs, effects, slice(None))[0, :, 0]
    understood = KL._row_logliks(inputs, effects, slice(None), understood_only=True)[0, :, 0]
    rows = inputs.rows
    has_u = ~np.isnan(expected_u[rows])
    has_s = ~np.isnan(expected_s[rows])
    assert has_u.sum() > 0 and has_s.sum() > 0
    np.testing.assert_allclose(
        understood[has_u], expected_u[rows][has_u], rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        (total - understood)[has_s], expected_s[rows][has_s], rtol=1e-9, atol=1e-9
    )
    # The scorer's covariance is the model's: L L^T against tau and rho.
    rho = float(values.get("rho_uq", 0.0))
    tau_u, tau_q = float(values["tau_subj_u"]), float(values["tau_subj_q"])
    cov = inputs.chol[0] @ inputs.chol[0].T
    np.testing.assert_allclose(
        cov,
        [[tau_u**2, rho * tau_u * tau_q], [rho * tau_u * tau_q, tau_q**2]],
        rtol=1e-10,
    )


@pytest.mark.parametrize("definition", [SMALL_VG12, SMALL_VG21], ids=["VG12", "VG21"])
def test_a_study_fold_scores_the_population_curve_plus_a_fresh_study(
    require_prepared_data, tmp_path, definition
):
    """The held-out study's placeholder offset is removed, and its scale integrated."""
    frame = _small_frame(definition)
    fold = KL.build_td_fold_frame(frame, "study", KL.study_folds(frame)[-1])
    model = _build(definition, fold, tmp_path).model
    point = _jittered_point(model)
    univariate = KL.td_engine(definition) == "univariate_re"
    values = _evaluate(model, point, _UNIVARIATE_NAMES if univariate else _BIVARIATE_NAMES)
    inputs = KL.td_score_inputs(fold, _one_draw_trace(model, values), definition, "study")
    held = fold["holdout"].to_numpy()
    assert set(inputs.rows) == set(np.flatnonzero(held))
    if univariate:
        codes = fold["subject_code"].to_numpy()[inputs.rows]
        np.testing.assert_allclose(
            inputs.arrays["m"][0],
            values["f_obs"][inputs.rows] - values["delta_subject"][codes] - values["delta"][0],
        )
        np.testing.assert_allclose(
            inputs.chol[0, 0, 0] ** 2, values["tau_subject"] ** 2 + values["tau"] ** 2
        )
    else:
        np.testing.assert_allclose(
            inputs.arrays["m_u"][0],
            values["f_u_obs"][inputs.rows]
            + values["beta_sex_u"] * KL._sex_shift(fold, definition)[inputs.rows],
        )
        cov = inputs.chol[0] @ inputs.chol[0].T
        np.testing.assert_allclose(
            np.diag(cov),
            [
                values["tau_subj_u"] ** 2 + values["tau_u"] ** 2,
                values["tau_subj_q"] ** 2 + values["tau_q"] ** 2,
            ],
        )


def test_an_unmodelled_term_in_the_fold_graph_fails_loudly(require_prepared_data, tmp_path):
    frame = _small_frame(SMALL_VG21)
    model = _build(SMALL_VG21, frame, tmp_path).model
    point = _jittered_point(model)
    values = _evaluate(model, point, _BIVARIATE_NAMES)
    p_u = values["p_u_obs"]
    values["p_u_obs"] = expit(np.log(p_u / (1 - p_u)) + 0.01)
    with pytest.raises(RuntimeError, match="does not model"):
        KL.td_score_inputs(
            frame.assign(holdout=True), _one_draw_trace(model, values), SMALL_VG21, "subject"
        )


# ---------------------------------------------------------------------------
# The integration
# ---------------------------------------------------------------------------


def _brute_force_1d(loglik_of_effect, sigma):
    z = np.linspace(-12, 12, 240_001)
    log_f = loglik_of_effect(sigma * z) + norm.logpdf(z)
    return logsumexp(log_f) + np.log(z[1] - z[0])


@pytest.mark.parametrize(
    ("kappa", "n_rows", "sigma"), [(20.0, 1, 0.8), (400.0, 8, 1.5), (60.0, 3, 0.05)]
)
def test_one_dimensional_quadrature_matches_brute_force(kappa, n_rows, sigma):
    rng = np.random.default_rng(int(kappa) + n_rows)
    m = rng.normal(-1.5, 0.5, n_rows)
    y = rng.integers(20, 300, n_rows).astype(float)

    def loglik_of_effect(e):
        p = expit(m + np.asarray(e)[..., None])
        return KL.betabinomial_logpmf(y, 810.0, p, kappa).sum(axis=-1)

    def loglik(z):  # z: (1, P, 1)
        return loglik_of_effect(sigma * z[..., 0])

    actual = KL.log_normal_expectation(loglik, (1,), 1, KL.DEFAULT_QUADRATURE_NODES)[0]
    assert actual == pytest.approx(_brute_force_1d(loglik_of_effect, sigma), abs=1e-4)


def test_node_placement_does_not_cycle_on_a_skewed_single_row():
    """A case where an undamped Newton step cycled between 0 and 2 forever.

    The log-integrand's curvature at the origin is positive here, so the first
    step falls back to the prior curvature, overshoots to the clamp, and the
    step back overshoots again. Backtracking is what stops it; without it the
    nodes sat a full prior SD off the mode and the result was off by 0.23 nats.
    """
    y, m, kappa, sigma = np.array([355.0]), np.array([-2.02]), 20.0, 1.6

    def loglik_of_effect(e):
        p = KL._clip(expit(m + np.asarray(e)[..., None]))
        return KL.betabinomial_logpmf(y, 810.0, p, kappa).sum(axis=-1)

    actual = KL.log_normal_expectation(
        lambda z: loglik_of_effect(sigma * z[..., 0]), (1,), 1, KL.DEFAULT_QUADRATURE_NODES
    )[0]
    assert actual == pytest.approx(_brute_force_1d(loglik_of_effect, sigma), abs=1e-4)


def test_two_dimensional_quadrature_matches_brute_force():
    """Correlated effects on two logits, comprehension sharpened by its own rows."""
    rng = np.random.default_rng(11)
    cov = np.array([[0.7**2, 0.3 * 0.7 * 1.1], [0.3 * 0.7 * 1.1, 1.1**2]])
    chol = np.linalg.cholesky(cov)
    m_u, m_q = rng.normal(-1.0, 0.3, 3), rng.normal(-0.5, 0.3, 3)
    y_u = np.array([150.0, 210.0, 260.0])
    y_s = np.array([40.0, 70.0, 120.0])

    def loglik(z):  # (..., 2) -> (...)
        e = np.einsum("ij,...j->...i", chol, z)
        p_u = expit(m_u + e[..., :1])
        q = expit(m_q + e[..., 1:])
        return (
            KL.betabinomial_logpmf(y_u, 810.0, p_u, 40.0)
            + KL.betabinomial_logpmf(y_s, y_u, q, 25.0)
        ).sum(axis=-1)

    actual = KL.log_normal_expectation(loglik, (1,), 2, KL.DEFAULT_QUADRATURE_NODES)[0]
    g = np.linspace(-9, 9, 1201)
    z = np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)
    log_f = loglik(z) + norm.logpdf(z[:, 0]) + norm.logpdf(z[:, 1])
    expected = logsumexp(log_f) + 2 * np.log(g[1] - g[0])
    assert actual == pytest.approx(expected, abs=1e-4)


def test_the_child_elpd_averages_integrated_densities_over_draws():
    """End to end on a hand-built univariate input: three draws, two children."""
    m = np.array(
        [[-1.0, -0.8, -0.6, -0.4], [-1.2, -1.0, -0.7, -0.5], [-0.9, -0.9, -0.5, -0.3]]
    )
    kappa = np.array(
        [[30.0, 30.0, 32.0, 35.0], [25.0, 25.0, 28.0, 30.0], [40.0, 40.0, 41.0, 45.0]]
    )
    sigma = np.array([0.6, 0.9, 0.75])
    y = np.array([120.0, 140.0, 200.0, 260.0])
    inputs = KL.TDScoreInputs(
        engine="univariate_re",
        rows=np.arange(4),
        starts=np.array([0, 1]),
        child_codes=np.array([7, 9]),
        child_of_row=np.array([0, 1, 1, 1]),
        chol=sigma[:, None, None],
        arrays={"m": m, "kappa": kappa, "y": y, "n": 810.0},
    )
    scores = KL.td_child_elpds(inputs).set_index("subject_code")
    for code, rows in ((7, [0]), (9, [1, 2, 3])):
        per_draw = [
            _brute_force_1d(
                lambda e, s=s, rows=rows: KL.betabinomial_logpmf(
                    y[rows], 810.0, expit(m[s, rows] + np.asarray(e)[..., None]), kappa[s, rows]
                ).sum(axis=-1),
                sigma[s],
            )
            for s in range(3)
        ]
        expected = logsumexp(per_draw) - np.log(3)
        assert scores.loc[code, "elpd"] == pytest.approx(expected, abs=1e-4)
        assert scores.loc[code, "n_rows"] == len(rows)


def test_numpy_betabinomial_is_scipys():
    y, n, p, kappa = 37.0, 810.0, 0.07, 18.0
    assert KL.betabinomial_logpmf(y, n, p, kappa) == pytest.approx(
        betabinom.logpmf(y, n, p * kappa, (1 - p) * kappa), abs=1e-10
    )


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["vg12", "vg21"])
def test_an_age_varying_child_scale_is_refused(key):
    variant = build_variant(key, "a1-tau-age-varying")[0]
    assert "age_varying" in KL.td_scoring_refusal(variant, "subject")


def test_a_fallback_the_scorer_does_not_implement_is_refused():
    variant = dataclasses.replace(VG21, spoken_fallback="moment_matched")
    assert "spoken_fallback" in KL.td_scoring_refusal(variant, "subject")


def test_study_slopes_are_refused_only_by_the_study_unit():
    variant = build_variant("vg21", "study-age-slopes")[0]
    assert KL.td_scoring_refusal(variant, "subject") is None
    assert "slopes" in KL.td_scoring_refusal(variant, "study")


def test_singleton_marginalisation_is_refused():
    from vocab_growth.models.definitions import (
        SingletonMarginalisationParams,
        UnivariateMarginalisedREModelDefinition,
        _as_definition_subclass,
    )

    marginal = _as_definition_subclass(
        VG12,
        UnivariateMarginalisedREModelDefinition,
        singleton_marginalisation=SingletonMarginalisationParams(n_nodes=12),
        config_name="marg-test",
    )
    assert "marginalisation" in KL.td_scoring_refusal(marginal, "subject")


def test_the_later_waves_unit_is_refused_for_td():
    assert "holdout units" in KL.td_scoring_refusal(VG12, "later-waves")


def test_a_run_takes_one_population_and_study_holdout_is_td_only():
    with pytest.raises(SystemExit, match="one population"):
        KL.main(models=("VG10", "VG12"))
    with pytest.raises(SystemExit, match="typically developing models"):
        KL.main(models=("VG10",), holdout_unit="study")
    assert KL.AVAILABLE["VG10"] is VG10


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _scores():
    return pd.DataFrame(
        {
            "model": ["A"] * 4 + ["B"] * 4,
            "fold": 0,
            "study": ["s1", "s1", "s2", "s2"] * 2,
            "subject_key": ["c1", "c2", "c3", "c4"] * 2,
            "subject_code": [1, 2, 3, 4] * 2,
            "n_rows": [1, 2, 1, 1] * 2,
            "elpd": [-10.0, -12.0, -9.0, -11.0, -9.5, -12.5, -8.0, -10.0],
        }
    )


def test_study_unit_standard_errors_are_over_study_totals():
    flags = {"A": True, "B": False}
    summary = KL.summarise_td(_scores(), "study", flags).set_index("model")
    totals = np.array([-22.0, -20.0])
    assert summary.loc["A", "elpd"] == pytest.approx(-42.0)
    assert summary.loc["A", "se"] == pytest.approx(np.sqrt(2) * np.std(totals, ddof=1))
    assert summary.loc["A", "n_clusters"] == 2
    assert summary.loc["A", "n_children"] == 4
    assert not summary.loc["B", "all_folds_converged"]

    pairs = KL.pairwise_td(_scores(), "study", flags)
    assert len(pairs) == 1
    diffs = np.array([(-9.5 - 12.5) - (-22.0), (-8.0 - 10.0) - (-20.0)])
    assert pairs.loc[0, "elpd_diff_b_minus_a"] == pytest.approx(diffs.sum())
    assert pairs.loc[0, "se_paired"] == pytest.approx(np.sqrt(2) * np.std(diffs, ddof=1))
    assert not pairs.loc[0, "all_folds_converged"]


def test_child_unit_standard_errors_are_over_children():
    summary = KL.summarise_td(_scores(), "subject_code", {"A": True, "B": True})
    a = summary.set_index("model").loc["A"]
    child = np.array([-10.0, -12.0, -9.0, -11.0])
    assert a["se"] == pytest.approx(np.sqrt(4) * np.std(child, ddof=1))
    assert a["n_clusters"] == 4
