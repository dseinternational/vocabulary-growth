# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The singleton child effects are integrated out, exactly and only.

``SingletonMarginalisationParams`` replaces the explicit ``delta_subject`` of a
child seen once with a quadrature integral over its prior. The claims that make
that admissible, and which these tests pin:

1. it is **exact** -- the marginal log density matches a fine-grid numerical
   integral of the same integrand, at the dispersion and scale both models of
   record actually occupy, and it does not move when the node count doubles;
2. it changes **nothing** for a child seen repeatedly -- those rows keep the
   conditional Beta-Binomial density the models have always used, to the last
   bit;
3. it removes exactly the singleton dimensions from the sampled space, and
   leaves an exact zero in the linear predictor where the effect used to be;
4. with the flag off the graph is the one every existing fit was produced
   under, and neither VG11 nor VG12 carries the flag, so their fits stay valid.

See ``notes/202608231410-td-geometry-remaining-levers.md`` §3.
"""

import dataclasses
import os

import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
from scipy import special

import vocab_growth.data_utils as vocab_data_utils
from vocab_growth.models import common_univariate_re as cur
from vocab_growth.models.common import ModelFitContext
from vocab_growth.models.definitions import (
    MODEL_REGISTRY,
    VG11,
    VG12,
    SingletonMarginalisationParams,
    UnivariateMarginalisedREModelDefinition,
    UnivariateREModelDefinition,
    _as_definition_subclass,
    validate_model_definition,
)
from vocab_growth.models.subject_marginal import (
    DEFAULT_QUADRATURE_NODES,
    _marginal_logp,
    betabinomial_logp,
    partition_subject_rows,
    singleton_first_order,
    standard_normal_quadrature,
    subject_marginal_betabinomial,
    zero_padded_subject_shift,
)

N_TRIALS = 810


# ============================================================
# The quadrature rule
# ============================================================


def test_weights_are_a_normalised_expectation():
    nodes, log_weights = standard_normal_quadrature(20)
    assert np.exp(log_weights).sum() == pytest.approx(1.0, abs=1e-15)
    # Symmetric about zero, so the rule integrates odd functions to zero.
    assert np.allclose(nodes, -nodes[::-1])
    assert (nodes * np.exp(log_weights)).sum() == pytest.approx(0.0, abs=1e-12)
    # And it reproduces the standard normal's second moment.
    assert (nodes**2 * np.exp(log_weights)).sum() == pytest.approx(1.0, abs=1e-12)


def test_two_nodes_is_the_floor():
    with pytest.raises(ValueError, match="at least 2 nodes"):
        standard_normal_quadrature(1)


def test_the_written_out_density_is_pymcs():
    """The hoisted form is an optimisation, not a second opinion.

    Both sides are given ``float64`` inputs deliberately. PyTensor types a bare
    Python scalar as the narrowest dtype that holds it, and a single-precision
    ``gammaln`` of an argument in the hundreds is wrong in the seventh decimal
    -- which is what this comparison would otherwise be measuring. The library
    function casts defensively for that reason; the test's own inputs have to be
    explicit so the comparison is between two double-precision computations.
    """
    value = np.array([0, 1, 37, 405, 809, 810], dtype=np.float64)
    for kappa_value in np.float64([5.0, 37.0, 288.0, 714.0]):
        for p_value in np.float64([0.002, 0.05, 0.4, 0.95]):
            ours = betabinomial_logp(
                pt.as_tensor_variable(value),
                pt.as_tensor_variable(p_value),
                pt.as_tensor_variable(kappa_value),
                n_trials=N_TRIALS,
            ).eval()
            theirs = pm.logp(
                pm.BetaBinomial.dist(
                    n=N_TRIALS,
                    alpha=p_value * kappa_value,
                    beta=(1 - p_value) * kappa_value,
                ),
                value,
            ).eval()
            assert np.allclose(ours, theirs, rtol=0, atol=1e-9)


def _reference_marginal(mu, kappa, y, tau, n_grid=200_001):
    """log E[BetaBinom(y | sigmoid(mu + tau u), kappa)] by a fine grid at the mode."""
    def log_terms(u):
        p = np.clip(special.expit(mu + tau * u), 1e-12, 1 - 1e-12)
        a, b = p * kappa, (1 - p) * kappa
        return (
            -0.5 * u * u
            - 0.5 * np.log(2 * np.pi)
            + special.gammaln(N_TRIALS + 1)
            - special.gammaln(y + 1)
            - special.gammaln(N_TRIALS - y + 1)
            + special.betaln(y + a, N_TRIALS - y + b)
            - special.betaln(a, b)
        )

    coarse = np.linspace(-30.0, 30.0, 6001)
    centre = coarse[np.argmax(log_terms(coarse))]
    fine = np.linspace(centre - 8.0, centre + 8.0, n_grid)
    terms = log_terms(fine)
    peak = terms.max()
    return peak + np.log(np.trapezoid(np.exp(terms - peak), fine))


def _graph_marginal(mu, kappa, y, tau, n_nodes=DEFAULT_QUADRATURE_NODES):
    nodes, log_weights = standard_normal_quadrature(n_nodes)
    return _marginal_logp(
        pt.as_tensor_variable(np.asarray(y, dtype=float)),
        pt.as_tensor_variable(np.asarray(mu, dtype=float)),
        pt.as_tensor_variable(np.asarray(kappa, dtype=float)),
        pt.as_tensor_variable(float(tau)),
        n_trials=N_TRIALS,
        nodes=nodes,
        log_weights=log_weights,
        epsilon=1e-12,
    ).eval()


# The two regimes the models of record occupy, plus rows whose child is far from
# the population -- the case that defeats prior-centred nodes (see the module
# docstring of vocab_growth.models.subject_marginal).
REGIMES = [
    pytest.param(0.687, 37.0, id="vg12-young"),
    pytest.param(0.687, 98.0, id="vg12-old"),
    pytest.param(1.038, 288.0, id="vg11-anchor"),
    pytest.param(1.038, 714.0, id="vg11-youngest"),
]


@pytest.mark.parametrize(("tau", "kappa"), REGIMES)
def test_marginal_matches_numerical_integration(tau, kappa):
    p_values = np.array([0.01, 0.05, 0.2, 0.5])
    mu = special.logit(p_values)
    for scale in (0.2, 1.0, 3.0):
        y = np.clip(np.round(p_values * scale * N_TRIALS), 0, N_TRIALS)
        got = _graph_marginal(mu, np.full(mu.shape, kappa), y, tau)
        want = np.array(
            [_reference_marginal(m, kappa, v, tau) for m, v in zip(mu, y, strict=True)]
        )
        assert np.allclose(got, want, rtol=0, atol=1e-4), (
            f"tau={tau} kappa={kappa} scale={scale}: {got - want}"
        )


@pytest.mark.parametrize(("tau", "kappa"), REGIMES)
def test_doubling_the_nodes_does_not_move_it(tau, kappa):
    """The node-count sensitivity check, at suite scale.

    A fit-level version of the same check is an obligation on any model that
    adopts the flag: the definition's ``n_nodes`` is what it doubles.
    """
    p_values = np.array([0.002, 0.05, 0.3, 0.8])
    mu = special.logit(p_values)
    y = np.clip(np.round(p_values * 2.0 * N_TRIALS), 0, N_TRIALS)
    kappas = np.full(mu.shape, kappa)
    twenty = _graph_marginal(mu, kappas, y, tau, n_nodes=20)
    forty = _graph_marginal(mu, kappas, y, tau, n_nodes=40)
    assert np.allclose(twenty, forty, rtol=0, atol=1e-4)


def test_a_vanishing_child_scale_is_the_conditional_density():
    """With no between-child variation left there is nothing to integrate."""
    mu = special.logit(np.array([0.01, 0.1, 0.5, 0.9]))
    kappa = np.array([37.0, 98.0, 288.0, 714.0])
    y = np.array([5.0, 90.0, 400.0, 730.0])
    got = _graph_marginal(mu, kappa, y, tau=1e-9)
    want = betabinomial_logp(
        pt.as_tensor_variable(y),
        pt.as_tensor_variable(special.expit(mu)),
        pt.as_tensor_variable(kappa),
        n_trials=N_TRIALS,
    ).eval()
    assert np.allclose(got, want, rtol=0, atol=1e-8)


def test_the_marginal_is_never_a_probability_above_one():
    """A quadrature that overestimated could invent an attractor in warmup.

    Includes rows a fit only reaches in early warmup: a child at the inventory
    ceiling while the population prediction is a fraction of a percent.
    """
    mu = special.logit(np.array([0.002, 0.002, 0.5, 0.98]))
    kappa = np.array([1000.0, 5.0, 1000.0, 50.0])
    y = np.array([810.0, 810.0, 0.0, 1.0])
    for tau in (0.3, 1.0, 2.5):
        assert np.all(_graph_marginal(mu, kappa, y, tau) < 0.0)


# ============================================================
# The row partition
# ============================================================


def test_partition_splits_rows_by_their_childs_administrations():
    # Children 0 and 3 are seen twice; 1, 2 and 4 once.
    codes = np.array([0, 1, 0, 2, 3, 3, 4])
    partition = partition_subject_rows(codes)
    assert partition.singleton_rows.tolist() == [1, 3, 6]
    assert partition.repeat_rows.tolist() == [0, 2, 4, 5]
    assert partition.repeat_labels.tolist() == [0, 3]
    assert partition.n_subjects == 5
    assert partition.n_repeat_subjects == 2
    assert partition.n_singleton_subjects == 3
    assert partition.n_singleton_rows == 3
    assert partition.n_repeat_rows == 4


def test_padded_codes_send_marginalised_rows_to_the_trailing_zero():
    codes = np.array([0, 1, 0, 2, 3, 3, 4])
    partition = partition_subject_rows(codes)
    # Two repeat-measured children, so the sentinel is position 2.
    assert partition.padded_codes.tolist() == [0, 2, 0, 2, 1, 1, 2]
    effects = pt.as_tensor_variable(np.array([-0.5, 1.25]))
    shift = zero_padded_subject_shift(effects, partition).eval()
    assert shift.tolist() == [-0.5, 0.0, -0.5, 0.0, 1.25, 1.25, 0.0]
    # The zeros are structural: they do not move with the effects.
    assert np.all(shift[partition.singleton_rows] == 0.0)


def test_singleton_first_order_is_stable():
    """Marginalised rows first, each block otherwise in the order it arrived."""
    codes = np.array([0, 1, 0, 2, 3, 3, 4])
    assert singleton_first_order(codes).tolist() == [1, 3, 6, 0, 2, 4, 5]
    reordered = partition_subject_rows(codes[singleton_first_order(codes)])
    assert reordered.is_singleton_first


def test_the_likelihood_refuses_rows_it_cannot_slice():
    """It reads its two blocks as slices, so an unordered frame is an error.

    Not a silent fallback to indexing them out: the gather that would need --
    and its scatter counterpart -- were where a non-finite gradient and a
    thread race came from. See ``singleton_first_order``.
    """
    partition = partition_subject_rows(np.array([0, 1, 0, 2]))
    assert not partition.is_singleton_first
    with pytest.raises(ValueError, match="marginalised row first"), pm.Model():
        subject_marginal_betabinomial(
            "y",
            mu=pt.zeros(4),
            kappa=pt.ones(4),
            tau_subject=pt.as_tensor_variable(np.float64(1.0)),
            observed=np.zeros(4, dtype=int),
            n_trials=10,
            partition=partition,
        )


def test_partition_rejects_malformed_codes():
    with pytest.raises(ValueError, match="one-dimensional"):
        partition_subject_rows(np.zeros((3, 2), dtype=int))
    with pytest.raises(ValueError, match="non-negative"):
        partition_subject_rows(np.array([0, -1, 2]))


# ============================================================
# The definition flag
# ============================================================


def test_the_models_of_record_do_not_carry_the_flag():
    """The invariant that keeps VG11's and VG12's published fits valid.

    A fit is validated by comparing ``dataclasses.asdict`` of its definition
    against the registered one, so a definition class that gains a field
    invalidates every fit of that class. The flag therefore lives on a subclass
    that no registered model uses yet. When one adopts it -- after the VG12
    bench -- that model needs a refit, and this test is the place to say so.
    """
    for definition in MODEL_REGISTRY.values():
        assert not isinstance(definition, UnivariateMarginalisedREModelDefinition), (
            f"{definition.model_id} adopted singleton marginalisation; its fit of "
            "record is stale and this guard needs updating with the refit."
        )
    assert isinstance(VG11, UnivariateREModelDefinition)
    assert isinstance(VG12, UnivariateREModelDefinition)
    assert not hasattr(VG12, "singleton_marginalisation")


def test_marginalisation_requires_a_subject_effect_to_marginalise():
    from vocab_growth.models.definitions import VG03

    definition = _as_definition_subclass(
        dataclasses.replace(VG03, config_name="marg-invalid"),
        UnivariateMarginalisedREModelDefinition,
        singleton_marginalisation=SingletonMarginalisationParams(n_nodes=20),
    )
    assert definition.use_subject_re is False
    with pytest.raises(ValueError, match="no subject random effect"):
        validate_model_definition(definition)


def test_the_node_count_must_be_a_usable_integer():
    with pytest.raises(ValueError, match="n_nodes"):
        SingletonMarginalisationParams(n_nodes=1)


# ============================================================
# The engine
# ============================================================

# A cheap stand-in for VG12: same engine, a twentieth of the children.
SMALL = dataclasses.replace(VG12, sample_fraction=0.05, min_study_observations=20)
SMALL_MARGINAL = _as_definition_subclass(
    SMALL,
    UnivariateMarginalisedREModelDefinition,
    singleton_marginalisation=SingletonMarginalisationParams(n_nodes=12),
    config_name="marg-test",
)


@pytest.fixture(scope="module")
def _require_data():
    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")


def _build(definition, tmp_path_factory, monkeypatch_module):
    monkeypatch_module.setattr(cur, "render_model_graph", lambda *a, **k: None)
    root = str(tmp_path_factory.mktemp(definition.config_name))
    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name=definition.config_name,
            output_root_dir=root,
            ci_prob=0.90,
            interval_kind="hdi",
        ),
        sampling=sampling.get_sampling_configuration("dev"),
    )
    os.makedirs(context.reporting.output_dir, exist_ok=True)
    cur.prepare_univariate_re_data(context, definition)
    cur.configure_univariate_priors(context, definition)
    cur.build_univariate_re_model(context, definition)
    return context


@pytest.fixture(scope="module")
def explicit_context(_require_data, tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield _build(SMALL, tmp_path_factory, monkeypatch)


@pytest.fixture(scope="module")
def marginal_context(_require_data, tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield _build(SMALL_MARGINAL, tmp_path_factory, monkeypatch)


def _dimensions(model):
    return {name: np.shape(value) for name, value in model.initial_point().items()}


def test_the_flag_off_graph_is_unchanged(explicit_context):
    model = explicit_context.model
    assert "subject_id" in model.coords and "repeat_subject_id" not in model.coords
    assert model["delta_subject"].type.shape == model["delta_subject_raw"].type.shape
    assert type(model["y_obs"].owner.op).__name__.startswith("BetaBinomial")


def test_only_repeat_measured_children_keep_an_effect(marginal_context):
    model = marginal_context.model
    analysis = marginal_context.analysis_df
    codes = np.asarray(analysis["subject_code"], dtype=int)
    partition = partition_subject_rows(codes)
    assert partition.n_singleton_subjects > 0, "the subsample has no singleton child"

    assert "repeat_subject_id" in model.coords
    assert len(model.coords["repeat_subject_id"]) == partition.n_repeat_subjects
    point = _dimensions(model)
    assert point["delta_subject_raw"] == (partition.n_repeat_subjects,)
    # tau_subject survives: the mixing is integrated, not deleted. (VG12 builds
    # it as a deterministic of the variance partition rather than sampling it.)
    names = {rv.name for rv in model.free_RVs} | {d.name for d in model.deterministics}
    assert "tau_subject" in names


def test_data_preparation_orders_marginalised_rows_first(marginal_context):
    """The engine's data preparation is what makes the slices legal."""
    codes = np.asarray(marginal_context.analysis_df["subject_code"], dtype=int)
    partition = partition_subject_rows(codes)
    assert partition.is_singleton_first
    assert partition.n_singleton_rows > 0
    assert partition.n_repeat_rows > 0


def test_the_flag_off_leaves_the_row_order_alone(explicit_context, marginal_context):
    """Same rows either way; only the marginalised build reorders them."""
    explicit_ages = list(explicit_context.analysis_df["age"])
    marginal_ages = list(marginal_context.analysis_df["age"])
    assert sorted(explicit_ages) == sorted(marginal_ages)
    assert explicit_ages != marginal_ages


def test_the_marginalised_rows_carry_no_child_effect(marginal_context):
    """f_obs on those rows is the population-and-study prediction, exactly."""
    model = marginal_context.model
    codes = np.asarray(marginal_context.analysis_df["subject_code"], dtype=int)
    partition = partition_subject_rows(codes)
    import pytensor

    point = model.initial_point()
    value_vars = model.value_vars
    outputs = model.replace_rvs_by_values([model["f_obs"]])
    evaluate = pytensor.function(value_vars, outputs, on_unused_input="ignore")

    baseline = evaluate(*[point[v.name] for v in value_vars])[0]
    moved = dict(point)
    moved["delta_subject_raw"] = point["delta_subject_raw"] + 3.0
    shifted = evaluate(*[moved[v.name] for v in value_vars])[0]

    assert np.array_equal(
        baseline[partition.singleton_rows], shifted[partition.singleton_rows]
    )
    assert np.all(
        baseline[partition.repeat_rows] != shifted[partition.repeat_rows]
    )


def test_repeat_rows_keep_the_conditional_density(marginal_context):
    """Bit for bit: those rows are not approximated, they are untouched."""
    import pytensor

    model = marginal_context.model
    codes = np.asarray(marginal_context.analysis_df["subject_code"], dtype=int)
    partition = partition_subject_rows(codes)

    point = model.initial_point()
    rng = np.random.default_rng(4)
    point = {
        name: value + rng.normal(scale=0.2, size=np.shape(value))
        for name, value in point.items()
    }
    value_vars = model.value_vars
    arguments = [point[v.name] for v in value_vars]

    pointwise = pytensor.function(
        value_vars, model.logp(sum=False), on_unused_input="ignore"
    )(*arguments)
    observed = [
        term for term, rv in zip(pointwise, model.basic_RVs, strict=True) if rv.name == "y_obs"
    ][0]

    f_obs, kappa_obs = pytensor.function(
        value_vars,
        model.replace_rvs_by_values([model["f_obs"], model["kappa_obs"]]),
        on_unused_input="ignore",
    )(*arguments)
    y = np.asarray(marginal_context.model_data.y_obs, dtype=float)

    rows = partition.repeat_rows
    expected = pm.logp(
        pm.BetaBinomial.dist(
            n=marginal_context.model_data.n_trials,
            alpha=special.expit(f_obs[rows]) * kappa_obs[rows],
            beta=(1 - special.expit(f_obs[rows])) * kappa_obs[rows],
        ),
        y[rows],
    ).eval()
    assert np.allclose(np.asarray(observed)[rows], expected, rtol=0, atol=1e-9)


def test_the_marginalised_rows_integrate_their_child_effect(marginal_context):
    """And the singleton rows match an independent numerical integration."""
    import pytensor

    model = marginal_context.model
    codes = np.asarray(marginal_context.analysis_df["subject_code"], dtype=int)
    partition = partition_subject_rows(codes)
    point = model.initial_point()
    value_vars = model.value_vars
    arguments = [point[v.name] for v in value_vars]

    pointwise = pytensor.function(
        value_vars, model.logp(sum=False), on_unused_input="ignore"
    )(*arguments)
    observed = np.asarray(
        [term for term, rv in zip(pointwise, model.basic_RVs, strict=True) if rv.name == "y_obs"][0]
    )
    f_obs, kappa_obs, tau_subject = pytensor.function(
        value_vars,
        model.replace_rvs_by_values(
            [model["f_obs"], model["kappa_obs"], model["tau_subject"]]
        ),
        on_unused_input="ignore",
    )(*arguments)
    y = np.asarray(marginal_context.model_data.y_obs, dtype=float)

    rows = partition.singleton_rows[:25]
    expected = np.array(
        [
            _reference_marginal(f_obs[i], kappa_obs[i], y[i], float(tau_subject))
            for i in rows
        ]
    )
    assert np.allclose(observed[rows], expected, rtol=0, atol=1e-4)


def test_the_marginalised_model_samples_and_predicts(marginal_context):
    """The whole path a fit needs: NUTS, log-likelihood, posterior predictive."""
    model = marginal_context.model
    with model:
        trace = pm.sample(
            draws=8,
            tune=8,
            chains=1,
            cores=1,
            progressbar=False,
            random_seed=17,
            compute_convergence_checks=False,
        )
        pm.compute_log_likelihood(trace, progressbar=False)
        pm.sample_posterior_predictive(
            trace, var_names=["y_obs"], progressbar=False, random_seed=17,
            extend_inferencedata=True,
        )

    n_rows = len(marginal_context.analysis_df)
    assert trace.log_likelihood["y_obs"].shape == (1, 8, n_rows)
    assert trace.posterior_predictive["y_obs"].shape == (1, 8, n_rows)
    assert np.isfinite(trace.log_likelihood["y_obs"].values).all()
    codes = np.asarray(marginal_context.analysis_df["subject_code"], dtype=int)
    partition = partition_subject_rows(codes)
    assert trace.posterior["delta_subject_raw"].shape[-1] == partition.n_repeat_subjects
