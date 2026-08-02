# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the conditional (GLMM) dispersion calibration.

The estimator behind ``notes/202608020829-kappa-and-eta-q-prior-recalibration.md``
§19 moved VG11's dispersion prior by a factor of ten, so what it claims has to be
checkable without a full pool: these run on small synthetic designs where the
truth is known by construction.

The load-bearing property is that the estimator can tell a subject random effect
from observation-level dispersion. The two are only weakly separated for a child
measured once -- both add variance to the same single number, and what
distinguishes them is the shape of the resulting count distribution rather than
its spread -- so repeated administrations are what make the separation precise.
If it failed silently the calibration would read the whole of the between-child
spread as dispersion and set a prior an order of magnitude out, which is exactly
the error it exists to correct.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "kappa_conditional_calibration.py"
)
_SPEC = importlib.util.spec_from_file_location("kappa_conditional_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

Design = _MODULE.Design
fit = _MODULE.fit
simulate = _MODULE.simulate

ANCHORS = (12.0, 20.0)
# enough to identify tau, small enough to fit in a couple of seconds
N_SUBJECTS = 900
N_NODES = 48


def _synthetic_design(*, repeats=True, seed=0, **basis):
    """A design with the shape of a real pool: several ages, studies, repeats."""
    rng = np.random.default_rng(seed)
    ages, subject, study = [], [], []
    for i in range(N_SUBJECTS):
        n_obs = 2 if (repeats and i % 3 == 0) else 1
        first = rng.integers(10, 22)
        for j in range(n_obs):
            ages.append(float(min(first + 3 * j, 24)))
            subject.append(i)
            study.append(i % 3)
    ages = np.array(ages, float)
    # placeholder counts; every test overwrites these with a simulated draw
    y = np.full(ages.size, 100.0)
    return Design(
        age=ages, y=y, n_trials=np.full(ages.size, 810.0),
        subject=np.array(subject), study=np.array(study), min_cell=1, **basis,
    )


def _refit(design, truth, *, seed, nodes=N_NODES):
    y = simulate(design, anchor_ages=ANCHORS, seed=seed, **truth)
    sim = Design(
        age=design.age, y=y, n_trials=design.n_trials,
        subject=design.subject_idx, study=design.study_idx, min_cell=1,
    )
    return fit(sim, ANCHORS, n_nodes=nodes)


_SUBJECT_HEAVY = dict(tau=1.0, kappa_min=5.0, excess_young=295.0, excess_old=45.0)
# tau here is 0.3 rather than the ~0.15 the real marginal regime implies: a tau
# that small is not identifiable from 1,200 rows (on the full VG11 frame, 13x
# larger, its standard error is still 0.25 on the log scale). The contrast being
# tested is the regime, and 0.3 against 1.0 is a regime apart.
_DISPERSION_HEAVY = dict(tau=0.3, kappa_min=3.0, excess_young=27.0, excess_old=3.6)

# Tolerances are measured, not guessed: the worst error over seeds 7-11 on this
# design, rounded up. A small tau is intrinsically harder to pin down than a
# large one, and asserting otherwise is what broke this file on CI.
_TAU_TOLERANCE = {"subject-heavy": 0.10, "dispersion-heavy": 0.35}


# --- the property everything else rests on ------------------------------------


@pytest.mark.parametrize(
    "truth,name", [(_SUBJECT_HEAVY, "subject-heavy"),
                   (_DISPERSION_HEAVY, "dispersion-heavy")]
)
@pytest.mark.parametrize("seed", [7, 8, 9])
def test_recovers_tau_in_both_regimes(truth, name, seed):
    """`tau` is the load-bearing quantity, and it recovers tightly.

    Whether the estimator can separate the subject effect from the dispersion is
    a question about `tau`: a design that cannot tell them apart returns a `tau`
    pulled toward whichever the simulation did not use. Measured over seeds
    7-11, recovery is within 6% for a large tau and 28% for a small one.

    Several seeds, because a single one hides how much of a pass is luck — the
    earlier version of this file asserted a `kappa` this design cannot pin down,
    passed locally, and failed on CI at a different point on the same flat ridge.
    """
    res = _refit(_synthetic_design(), truth, seed=seed)

    assert res.tau == pytest.approx(truth["tau"], rel=_TAU_TOLERANCE[name])


def test_kappa_recovers_to_within_the_design_s_resolution():
    """`kappa` recovers only loosely here, and the tolerance says how loosely.

    1,200 observations with a saturated mean spending 15 degrees of freedom do
    not pin a dispersion parameter down: across seeds this design returns 26-33
    against a truth of 30, and at the ~300 of VG11's real posterior it returns
    anywhere from 158 to 298. The tight check belongs on the real 16,235-row
    frame, where `scripts/kappa_conditional_calibration.py --recover` puts it
    within 7% (see section 19 of the note); what is checkable here is that the
    estimate lands in the right region rather than at the prior or at a bound.
    """
    res = _refit(_synthetic_design(), _DISPERSION_HEAVY, seed=7)
    truth = _DISPERSION_HEAVY["kappa_min"] + _DISPERSION_HEAVY["excess_young"]

    assert res.kappa_young == pytest.approx(truth, rel=0.4)


def test_the_two_regimes_are_told_apart():
    """The sharper claim: the fits do not land in the same place.

    A recovery tolerance can be met while the estimator returns much the same
    answer either way. This asserts the contrast directly, on `tau`, which is
    where the identification question actually lives.
    """
    design = _synthetic_design()
    heavy = _refit(design, _SUBJECT_HEAVY, seed=7)
    light = _refit(design, _DISPERSION_HEAVY, seed=7)

    assert heavy.tau > 2 * light.tau


def test_a_small_tau_needs_the_repeats():
    """The converse, so the test above is known to be testing something.

    It is tempting to say the subject effect and the dispersion are *confounded*
    for a child measured once — both add variance to one number. That is too
    strong, and this design shows why: a logit-normal random effect and a
    Beta-Binomial leave differently shaped count distributions, so a large tau is
    still recovered from 900 singletons (0.91-1.03 against a truth of 1.0).

    What the repeats buy is resolution at the *small* end, and there the
    difference is stark. Strip them out and a truth of tau = 0.3 comes back
    anywhere in 0.001-0.48 across seeds — the estimator can no longer tell
    whether there is a subject effect at all. That is the regime that matters,
    since it is what separates a model needing a conditional prior from one that
    does not.
    """
    truth = _DISPERSION_HEAVY["tau"]
    without = [
        _refit(_synthetic_design(repeats=False), _DISPERSION_HEAVY, seed=s).tau
        for s in (10, 11)
    ]

    assert not all(
        t == pytest.approx(truth, rel=_TAU_TOLERANCE["dispersion-heavy"])
        for t in without
    )


def test_too_few_quadrature_nodes_biases_kappa_down():
    """Why DEFAULT_NODES is 160 and not the 24 a first pass would reach for.

    Under-integrating a wide subject distribution understates the spread the
    random effect accounts for, so the dispersion has to absorb it and `kappa`
    comes out too low. On the real VG11 frame that was a 17% error.
    """
    design = _synthetic_design()
    coarse = _refit(design, _SUBJECT_HEAVY, seed=7, nodes=8)
    fine = _refit(design, _SUBJECT_HEAVY, seed=7, nodes=160)

    # A wide margin, not a hair: what matters is the direction and that the error
    # is large enough to matter, not its exact size on a design this small.
    assert coarse.kappa_young < 0.8 * fine.kappa_young


# --- the age-varying subject loading (a diagnostic, not a calibration path) -----
#
# Section 21 of the note traces the 16-18 month typically-developing understood
# `kappa` spike to a subject scale that falls with age while the model holds it
# constant. `--loading` is the check for that, and it is only worth anything if
# it is quiet when the scale really is constant and loud when it is not. Both
# truths below are simulated on the same design; tolerances are the worst over
# seeds 7-11, rounded out.

_CONSTANT_LOADING = dict(tau=1.0, kappa_min=5.0, excess_young=295.0, excess_old=45.0)
_FALLING_LOADING = dict(tau=1.3, lam_old=0.55, kappa_min=5.0,
                        excess_young=95.0, excess_old=45.0)


def _refit_loading(truth, *, seed, nodes=N_NODES):
    """Fit the same simulated draw both ways; return (constant tau, loading)."""
    design = _synthetic_design()
    y = simulate(design, anchor_ages=ANCHORS, seed=seed, **truth)
    sim = Design(
        age=design.age, y=y, n_trials=design.n_trials,
        subject=design.subject_idx, study=design.study_idx, min_cell=1,
    )
    return (fit(sim, ANCHORS, n_nodes=nodes),
            fit(sim, ANCHORS, n_nodes=nodes, loading=True))


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_a_constant_subject_scale_reads_as_constant(seed):
    """The null. A one-parameter extension must not pay for itself on noise.

    Simulating a genuinely constant loading and fitting the age-varying form
    returns a flat one (ratio 0.94-1.11 over seeds 7-11) for 0.4-1.9 log-likelihood
    units. Without this the diagnostic would flag every pool it was pointed at.
    """
    const, varying = _refit_loading(_CONSTANT_LOADING, seed=seed)

    assert const.nll - varying.nll < 5.0
    assert varying.lam_old / varying.tau == pytest.approx(1.0, abs=0.15)


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_a_falling_subject_scale_is_found_and_distorts_kappa_if_it_is_not(seed):
    """The alternative, and why it matters that the diagnostic exists.

    A loading falling 1.3 -> 0.55 across the anchors is recovered as 0.43-0.50 of
    its young value for 63-82 log-likelihood units. Fitting the same draw with
    the constant scale every registered model carries does not merely lose those
    units: its `kappa` at the young anchor comes back at 13-26 against a truth of
    100, so a calibration read off it would be out by nearly an order of
    magnitude in a way nothing in the fit itself announces.
    """
    truth_ratio = _FALLING_LOADING["lam_old"] / _FALLING_LOADING["tau"]
    truth_kappa_young = _FALLING_LOADING["kappa_min"] + _FALLING_LOADING["excess_young"]
    const, varying = _refit_loading(_FALLING_LOADING, seed=seed)

    assert const.nll - varying.nll > 30.0
    assert varying.lam_old / varying.tau == pytest.approx(truth_ratio, rel=0.35)
    assert const.kappa_young < 0.5 * truth_kappa_young


def test_the_loading_fit_leaves_the_constant_form_untouched():
    """`loading=False` must be exactly the code path the calibration already uses.

    The loading parameter is appended after the existing four, so every index
    into the parameter vector is unchanged and a default fit cannot drift.
    """
    design = _synthetic_design()
    plain = _MODULE._layout(design)
    extended = _MODULE._layout(design, loading=True)

    assert extended["n_params"] == plain["n_params"] + 1
    assert extended["log_lam_old"] == plain["n_params"]
    for key in ("log_tau", "log_kmin", "log_ey", "log_eo"):
        assert extended[key] == plain[key]
    assert "log_lam_old" not in plain


def test_loading_and_a_pinned_tau_are_rejected_together():
    """A fixed scale and a varying one are contradictory, not silently merged."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _MODULE.make_objective(
            _synthetic_design(), ANCHORS, n_nodes=8, tau_fixed=1e-6, loading=True
        )


# --- the anchored curve --------------------------------------------------------


def test_kappa_at_hits_the_anchors():
    design = _synthetic_design()
    res = _refit(design, _SUBJECT_HEAVY, seed=3)

    assert res.kappa_at(ANCHORS[0]) == pytest.approx(res.kappa_young)
    assert res.kappa_at(ANCHORS[1]) == pytest.approx(res.kappa_old)


def test_kappa_at_is_log_linear_above_the_floor_in_months():
    """The interpolation is in months, matching the model-side builder."""
    design = _synthetic_design()
    res = _refit(design, _SUBJECT_HEAVY, seed=3)

    midpoint = 0.5 * (ANCHORS[0] + ANCHORS[1])
    above_floor = res.kappa_at(midpoint) - res.kappa_min
    assert above_floor == pytest.approx(
        np.sqrt(res.excess_young * res.excess_old), rel=1e-9
    )


def test_unordered_anchors_are_rejected():
    design = _synthetic_design()
    with pytest.raises(ValueError, match="ordered"):
        fit(design, (20.0, 12.0), n_nodes=8)


# --- the marginal fit is the thing being contrasted against --------------------


def test_pinning_tau_at_zero_recovers_the_marginal_estimate():
    """`tau_fixed` is what produces the marginal column of the results table.

    With the subject effect switched off, the between-child spread has nowhere
    to go but the dispersion, so kappa must come out *lower* -- the direction of
    the factor of ten the calibration corrects.
    """
    design = _synthetic_design()
    y = simulate(design, anchor_ages=ANCHORS, seed=5, **_SUBJECT_HEAVY)
    sim = Design(
        age=design.age, y=y, n_trials=design.n_trials,
        subject=design.subject_idx, study=design.study_idx, min_cell=1,
    )

    conditional = fit(sim, ANCHORS, n_nodes=N_NODES)
    marginal = fit(sim, ANCHORS, n_nodes=N_NODES, tau_fixed=1e-6)

    assert marginal.tau == pytest.approx(0.0, abs=1e-5)
    assert marginal.kappa_young < conditional.kappa_young
    # and the conditional fit must actually be the better explanation
    assert conditional.nll < marginal.nll


# --- numerical robustness ---------------------------------------------------------


@pytest.mark.parametrize("log_kappa", [np.log(300.0), np.log(5e4), np.log(1e7)])
def test_gradient_stays_finite_at_extreme_dispersion(log_kappa):
    """The failure that only showed up on CI.

    With a wide `tau` the outermost quadrature nodes push p to the edge, and once
    `kappa` is large the smaller Beta parameter underflows to zero. `betaln` then
    returns inf; logsumexp gives such a node a softmax weight of about e^-80, but
    reverse-mode AD still computes 0 * inf = NaN and the entire gradient is lost.
    L-BFGS receives NaN, stops, and reports a nonsense optimum — which is exactly
    what happened when the line search happened to wander far enough, hence the
    platform dependence. The estimator converged locally and returned 158 against
    a truth of 300 in CI.
    """
    design = _synthetic_design()
    nll, layout = _MODULE.make_objective(design, ANCHORS, n_nodes=48)

    theta = np.zeros(layout["n_params"])
    theta[layout["log_tau"]] = np.log(2.5)  # wide, so the far nodes really are far
    theta[layout["log_kmin"]] = log_kappa
    theta[layout["log_ey"]] = theta[layout["log_eo"]] = log_kappa

    import jax

    value = float(nll(np.asarray(theta)))
    grad = np.asarray(jax.grad(nll)(np.asarray(theta)), float)

    assert np.isfinite(value)
    assert np.all(np.isfinite(grad))


def test_the_optimiser_stays_inside_its_box():
    """The bounds exist to stop a runaway line search, not to shape the answer.

    Both boxes sit orders of magnitude outside any real fit, so a converged
    optimum must be strictly interior — an estimate sitting *on* a bound would
    mean the bounds had become part of the model.
    """
    res = _refit(_synthetic_design(), _SUBJECT_HEAVY, seed=7)

    assert 1e-3 < res.tau < 10.0
    for value in (res.kappa_min, res.excess_young, res.excess_old):
        assert 1e-4 < value < 1e6


# --- the likelihood is the likelihood --------------------------------------------


def test_nll_matches_scipy_on_a_design_with_no_random_effect():
    """Pin the absolute nll, not just differences of it.

    The normalising constant cancels from every optimum and every likelihood
    ratio, so an error in it is invisible to all the other tests here — it was in
    fact wrong by ``2 log(n + 1)`` per row in the first version of this estimator.
    With ``tau`` pinned at zero, one study and one age cell, the marginal
    likelihood collapses to a plain Beta-Binomial sum that scipy can evaluate.
    """
    from scipy.stats import betabinom

    n_trials, kappa_min, excess = 810.0, 5.0, 25.0
    y = np.array([120.0, 200.0, 75.0, 310.0])
    design = Design(
        age=np.full(4, 12.0), y=y, n_trials=np.full(4, n_trials),
        subject=np.arange(4), study=np.zeros(4), min_cell=1,
    )
    nll, layout = _MODULE.make_objective(design, ANCHORS, n_nodes=8, tau_fixed=0.0)

    logit_p = 0.3
    theta = np.zeros(layout["n_params"])
    theta[layout["m"]] = logit_p
    theta[layout["log_kmin"]] = np.log(kappa_min)
    # both anchors equal => kappa is flat at kappa_min + excess for every age
    theta[layout["log_ey"]] = theta[layout["log_eo"]] = np.log(excess)

    p = 1.0 / (1.0 + np.exp(-logit_p))
    kappa = kappa_min + excess
    expected = -betabinom.logpmf(y, int(n_trials), p * kappa, (1 - p) * kappa).sum()

    # rel=1e-7, not tighter: JAX's betaln differs from scipy's by ~7e-8 absolute
    # at these arguments, which is a special-function precision difference and not
    # a discrepancy in the likelihood. That still leaves eight orders of magnitude
    # between this tolerance and the constant this test exists to catch.
    assert float(nll(np.asarray(theta))) == pytest.approx(expected, rel=1e-7)


# --- design construction --------------------------------------------------------


def test_sparse_age_cells_are_dropped_by_min_cell():
    design = Design(
        age=np.array([12.0] * 20 + [30.0] * 3),
        y=np.full(23, 100.0),
        n_trials=np.full(23, 810.0),
        subject=np.arange(23),
        study=np.zeros(23),
        min_cell=15,
    )

    assert design.cells.tolist() == [12]
    assert design.n_dropped == 3


def test_the_spline_mean_is_an_alternative_to_the_saturated_one():
    saturated = _synthetic_design()
    spline = _synthetic_design(mean="spline", n_knots=5)

    assert saturated.B.shape[1] == len(saturated.cells)
    assert spline.B.shape[1] < saturated.B.shape[1]
    # a basis, not an indicator: rows are smooth partitions of unity
    assert np.allclose(spline.B.sum(axis=1), 1.0)


def test_repeated_subjects_are_counted():
    design = _synthetic_design()

    assert design.n_subjects == N_SUBJECTS
    assert design.n_repeat == len(range(0, N_SUBJECTS, 3))


def test_every_registered_pool_names_a_real_model_and_ordered_anchors():
    for key, pool in _MODULE.POOLS.items():
        assert hasattr(_MODULE.defs, pool.model_id), key
        assert pool.anchors[1] > pool.anchors[0], key
        assert pool.part in (None, "u", "s"), key
