# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the conditional (GLMM) dispersion calibration.

The estimator behind ``notes/202608020829-kappa-and-eta-q-prior-recalibration.md``
§19 moved VG11's dispersion prior by a factor of ten, so what it claims has to be
checkable without a full pool: these run on small synthetic designs where the
truth is known by construction.

The load-bearing property is that the estimator can tell a subject random effect
from observation-level dispersion. For a child measured once the two are
confounded -- they add variance to the same single number -- and only children
with a repeat separate them. If that failed silently the calibration would read
the whole of the between-child spread as dispersion and set a prior an order of
magnitude out, which is exactly the error it exists to correct.
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


# --- the property everything else rests on ------------------------------------


@pytest.mark.parametrize(
    "truth,name", [(_SUBJECT_HEAVY, "subject-heavy"),
                   (_DISPERSION_HEAVY, "dispersion-heavy")]
)
def test_recovers_tau_and_kappa_in_both_regimes(truth, name):
    """A large tau with small kappa must not be confusable with the reverse."""
    design = _synthetic_design()
    res = _refit(design, truth, seed=7)

    assert res.tau == pytest.approx(truth["tau"], rel=0.35)
    # the anchors are the quantities the prior is stated on
    assert res.kappa_young == pytest.approx(
        truth["kappa_min"] + truth["excess_young"], rel=0.45
    )
    assert res.kappa_old == pytest.approx(
        truth["kappa_min"] + truth["excess_old"], rel=0.45
    )


def test_the_two_regimes_are_told_apart():
    """The weaker but sharper claim: the fits do not land in the same place.

    Recovery tolerances are loose enough that both could in principle pass while
    the estimator was returning much the same answer either way. They do not.
    """
    design = _synthetic_design()
    heavy = _refit(design, _SUBJECT_HEAVY, seed=7)
    light = _refit(design, _DISPERSION_HEAVY, seed=7)

    assert heavy.tau > 2 * light.tau
    assert heavy.kappa_young > 5 * light.kappa_young


def test_without_repeats_tau_and_kappa_are_not_separable():
    """The converse, so the test above is known to be testing something.

    With one observation per child the subject effect and the dispersion are
    formally confounded, and the fit should not reproduce the truth. This is why
    the DS frame's 1.73 observations per child matter more than its row count.
    """
    design = _synthetic_design(repeats=False)
    res = _refit(design, _SUBJECT_HEAVY, seed=7)

    recovered_the_truth = (
        res.tau == pytest.approx(_SUBJECT_HEAVY["tau"], rel=0.35)
        and res.kappa_young == pytest.approx(
            _SUBJECT_HEAVY["kappa_min"] + _SUBJECT_HEAVY["excess_young"], rel=0.45
        )
    )
    assert not recovered_the_truth


def test_too_few_quadrature_nodes_biases_kappa_down():
    """Why DEFAULT_NODES is 160 and not the 24 a first pass would reach for.

    Under-integrating a wide subject distribution understates the spread the
    random effect accounts for, so the dispersion has to absorb it and `kappa`
    comes out too low. On the real VG11 frame that was a 17% error.
    """
    design = _synthetic_design()
    coarse = _refit(design, _SUBJECT_HEAVY, seed=7, nodes=8)
    fine = _refit(design, _SUBJECT_HEAVY, seed=7, nodes=160)

    assert coarse.kappa_young < fine.kappa_young


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
