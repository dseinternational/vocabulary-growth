# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

import numpy as np
import pytest

from vocab_growth import comparison
from vocab_growth import environment as env


# ---- registry resolution ----
def test_registry_resolution():
    # The output root is resolved at call time (an explicit override, then
    # DSE_VOCAB_GROWTH_OUTPUT_DIR, then the repo-local default), so the expected
    # directory is derived from the same resolution rather than assuming the
    # default; the test then holds wherever the fits have been redirected.
    expected = os.path.join(env.models_output_dir(), "VG11-age-spoken-td-re")
    assert os.path.normcase(comparison.model_dir("vg11")) == os.path.normcase(expected)
    assert comparison.model_dir("vg11").replace("\\", "/").endswith(
        "/models/VG11-age-spoken-td-re"
    )
    assert comparison.n_trials("vg11") == 810
    assert comparison.n_trials("vg10") == 810
    assert comparison.population("vg11") == "td"
    assert comparison.population("vg10") == "ds"
    assert comparison.model_label("vg11") == "VG11 (TD)"
    assert comparison.model_label("vg10") == "VG10 (DS)"
    assert comparison.trace_path("vg10").replace("\\", "/").endswith("trace.nc")


# ---- first_crossing (scalar, CSV curves) ----
def test_first_crossing_linear_interpolation():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 10.0, 20.0, 30.0])
    assert comparison.first_crossing(x, y, 15.0) == 1.5


def test_first_crossing_never_reached_is_none():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    assert comparison.first_crossing(x, y, 99.0) is None


def test_first_crossing_already_above_is_none():
    # Curve already exceeds the threshold at the youngest grid point: the true
    # crossing lies below the observed range, so the milestone is unidentified
    # rather than clamped to x[0].
    x = np.array([5.0, 6.0, 7.0])
    y = np.array([50.0, 60.0, 70.0])
    assert comparison.first_crossing(x, y, 10.0) is None


def test_first_crossing_exactly_at_first_point_returns_first_x():
    # Equals the threshold exactly at the youngest grid point: a genuine crossing.
    x = np.array([5.0, 6.0, 7.0])
    y = np.array([10.0, 60.0, 70.0])
    assert comparison.first_crossing(x, y, 10.0) == 5.0


# ---- per-draw crossing / interpolation ----
def test_first_crossing_age_recovers_shift():
    ages = np.linspace(0.0, 100.0, 1001)
    U = np.stack([ages, ages + 0.5, ages - 0.3])
    S = U - 5.0
    for n in (10.0, 50.0, 90.0):
        da = comparison.first_crossing_age(S, ages, n) - comparison.first_crossing_age(U, ages, n)
        assert np.allclose(da, 5.0, atol=1e-2)


def test_evaluate_at_ages_interp_and_out_of_range():
    ages = np.array([0.0, 10.0, 20.0])
    Y = np.array([[0.0, 100.0, 200.0]])
    assert np.isclose(comparison.evaluate_at_ages(Y, ages, np.array([5.0]))[0], 50.0)
    assert np.isnan(comparison.evaluate_at_ages(Y, ages, np.array([25.0]))[0])


def test_interp_draws_out_of_range_returns_nan():
    ages = np.array([0.0, 10.0, 20.0])
    Y = np.array([[0.0, 100.0, 200.0]])
    out = comparison.interp_draws(ages, Y, np.array([-1.0, 5.0, 21.0]))
    assert np.isnan(out[0, 0])
    assert np.isclose(out[0, 1], 50.0)
    assert np.isnan(out[0, 2])


# ---- HDI / summary ----
def test_hdi_from_samples_uniform_grid():
    x = np.linspace(0.0, 1.0, 1001)
    lo, hi = comparison.hdi_from_samples(x, 0.5)
    assert np.isclose(hi - lo, 0.5, atol=1e-2)
    assert lo >= 0.0 and hi <= 1.0


def test_summarise_per_N_coverage_and_columns():
    samples = np.tile(np.array([1.0, 2.0, 3.0]), (100, 1))
    samples[:50, 1] = np.nan  # half-missing middle column
    df = comparison.summarise_per_N(samples, np.array([10.0, 20.0, 30.0]))
    assert list(df["coverage"]) == [1.0, 0.5, 1.0]
    assert {"median", "ci50_lo", "ci_hi"}.issubset(df.columns)


# ---- analyses ----
def test_compute_latency_shifted_linear():
    ages = np.linspace(0.0, 100.0, 1001)
    U = np.stack([ages, ages + 0.2])
    S = U - 5.0
    da_df, extra_df = comparison.compute_latency(ages, U, S, np.array([10.0, 50.0]))
    assert np.allclose(da_df["median"], 5.0, atol=1e-2)
    assert np.allclose(extra_df["median"], 5.0, atol=1e-2)


def test_compute_q_at_U_constant_ratio():
    ages = np.linspace(0.0, 100.0, 1001)
    U = np.stack([ages * 8.0, ages * 8.0])  # 0..800
    S = 0.3 * U
    q = comparison.compute_q_at_U(ages, U, S, np.array([100.0, 400.0]))
    assert np.allclose(q[~np.isnan(q)], 0.3, atol=1e-6)


def test_comprehension_equivalent_age_uses_first_crossing_for_nonmonotone_reference():
    ages_ds = np.array([0.0, 1.0])
    U_ds = np.array([[0.0, 7.0]])
    S_ds = U_ds.copy()
    ages_td = np.array([0.0, 1.0, 2.0, 3.0])
    # This reference draw first crosses 7 words between ages 0 and 1, then dips.
    # Plain np.interp(target, W, ages) is undefined here because W is not sorted.
    U_td = np.array([[0.0, 10.0, 5.0, 15.0]])
    S_td = U_td.copy()

    out = comparison.comprehension_equivalent_age(
        ages_ds, U_ds, S_ds, ages_td, U_td, S_td, np.array([1.0])
    )

    assert np.isclose(out["cea_U"][0, 0], 0.7)
    assert np.isclose(out["cea_S"][0, 0], 0.7)


def test_milestone_table_is_median_of_crossings():
    # Three draws with different linear slopes; each reaches 100 words at a known
    # age. milestone_table must report the median-of-crossings (age 10 here), NOT
    # the crossing of the median count curve.
    ages = np.array([0.0, 10.0, 20.0, 30.0])
    W = np.array([
        [0.0, 200.0, 400.0, 600.0],  # crosses 100 at age 5
        [0.0, 100.0, 200.0, 300.0],  # crosses 100 at age 10
        [0.0, 50.0, 100.0, 150.0],   # crosses 100 at age 20
    ])
    tbl = comparison.milestone_table(W, ages, targets=[100], ci_prob=0.89)
    row = tbl.iloc[0]
    assert row["target_words"] == 100
    assert np.isclose(row["age_median"], 10.0)   # median of {5, 10, 20}
    assert row["prop_reaching"] == 1.0
    assert row["age_ci_lo"] <= 10.0 <= row["age_ci_hi"]


def test_milestone_table_flags_unreached_and_below_support():
    ages = np.array([8.0, 12.0, 16.0])
    # Draw 0 never reaches 400 on the grid; draw 1 is already above 400 at the
    # youngest age (crossing below support) — both are excluded from the age
    # summary and counted in prop_reaching.
    W = np.array([
        [10.0, 50.0, 120.0],   # never reaches 400
        [500.0, 600.0, 700.0],  # already > 400 at age 8 → unidentified
    ])
    tbl = comparison.milestone_table(W, ages, targets=[400])
    row = tbl.iloc[0]
    assert row["prop_reaching"] == 0.0
    assert row["age_median"] is None


# ---- dq_contrast_facts (matched-comprehension prose derivation) ----
def _dq_frame(rows):
    """Build a summarise_draws-shaped Δq frame from (words, median, lo, hi, cov)."""
    import pandas as pd

    return pd.DataFrame(
        rows,
        columns=["words", "dq_median", "dq_ci_lo", "dq_ci_hi", "dq_coverage"],
    )


def test_dq_contrast_facts_missing_or_empty_is_none():
    import pandas as pd

    assert comparison.dq_contrast_facts(None) is None
    assert comparison.dq_contrast_facts(pd.DataFrame()) is None
    # Present but every point below the coverage floor: conditional summaries are
    # dropped rather than reported, so there is nothing to say.
    thin = _dq_frame([(50, 0.05, 0.01, 0.09, 0.10)])
    assert comparison.dq_contrast_facts(thin) is None


def test_dq_contrast_facts_extents_peak_and_direction():
    # Rises from a non-credible negative point, through zero, to a credible
    # positive plateau — the shape the current DS/TD fits produce.
    facts = comparison.dq_contrast_facts(_dq_frame([
        (30, -0.014, -0.033, +0.001, 1.0),
        (50, -0.001, -0.018, +0.014, 1.0),
        (100, +0.034, +0.013, +0.051, 1.0),
        (175, +0.063, +0.021, +0.095, 1.0),
        (200, +0.049, -0.019, +0.097, 1.0),
    ]))
    assert facts["covered"] == (30.0, 200.0)
    assert facts["positive"] == (100.0, 175.0)   # interval excludes zero
    assert facts["negative"] is None             # never credibly negative
    assert facts["peak"]["words"] == 175         # largest magnitude
    assert facts["rises"] is True                # advantage grows with vocabulary


def test_dq_contrast_facts_peak_is_signed_not_largest_positive():
    # A contrast that is mostly negative must report the negative peak, not the
    # largest positive point, or the prose would invert the finding.
    facts = comparison.dq_contrast_facts(_dq_frame([
        (30, -0.20, -0.30, -0.10, 1.0),
        (60, -0.05, -0.12, +0.02, 1.0),
        (90, +0.02, -0.04, +0.08, 1.0),
    ]))
    assert facts["peak"]["dq_median"] == -0.20
    assert facts["negative"] == (30.0, 30.0)
    assert facts["positive"] is None
    assert facts["rises"] is True


def test_dq_contrast_facts_applies_coverage_floor():
    facts = comparison.dq_contrast_facts(_dq_frame([
        (25, -0.017, -0.036, -0.001, 0.44),   # below the floor: excluded
        (30, -0.014, -0.033, +0.001, 1.00),
        (100, +0.034, +0.013, +0.051, 1.00),
        (150, +0.059, +0.029, +0.084, 1.00),
    ]))
    assert facts["covered"][0] == 30.0        # not 25
    assert facts["negative"] is None          # the credible-negative point was filtered
    assert len(facts["table"]) == 3


def test_dq_contrast_facts_direction_undefined_for_two_points():
    facts = comparison.dq_contrast_facts(_dq_frame([
        (100, +0.03, +0.01, +0.05, 1.0),
        (150, +0.06, +0.03, +0.08, 1.0),
    ]))
    assert facts["rises"] is None              # no monotone claim from two points
    assert facts["positive"] == (100.0, 150.0)


# ---- between-child heterogeneity (subject random-effect scale) ----
def test_child_spread_single_returns_tau_exactly():
    # One Normal intercept on the outcome's own logit: the SD of the child's logit
    # p *is* tau, at every age and whatever the population trajectory does.
    f = np.array([[-6.0, -2.0, 0.0, 3.0], [-1.0, -1.0, -1.0, -1.0]])
    tau = np.array([0.4, 1.7])
    tau_logit, _ = comparison.child_spread_single(f, tau, 810)
    assert tau_logit.shape == f.shape
    assert np.allclose(tau_logit[0], 0.4)
    assert np.allclose(tau_logit[1], 1.7)


def test_child_spread_single_word_sd_matches_monte_carlo():
    rng = np.random.default_rng(20260808)
    f = np.array([[-4.0, -1.0, 0.5]])
    tau, n = 0.9, 810
    _, sd_words = comparison.child_spread_single(f, np.array([tau]), n)
    z = rng.standard_normal((400_000, 1))
    p = 1.0 / (1.0 + np.exp(-(f + tau * z)))
    assert np.allclose(sd_words[0], n * p.std(axis=0), rtol=0.02)


def test_child_spread_product_matches_monte_carlo():
    # The joint model's induced spoken scale is the quantity with no parameter to
    # read off, so it is the one that has to be checked against brute force. Kept
    # to moderate proportions: out in the lower tail the *Monte Carlo* estimate of
    # a word SD is the noisy one, and the comparison stops testing the quadrature.
    rng = np.random.default_rng(20260808)
    f_u = np.array([[-2.0, 0.0, 1.0]])
    h = np.array([[-1.5, 0.0, 0.5]])
    tau_u, tau_q, n = 0.8, 1.4, 810
    tau_logit, sd_words = comparison.child_spread_product(
        f_u, h, np.array([tau_u]), np.array([tau_q]), n
    )
    z1 = rng.standard_normal((400_000, 1))
    z2 = rng.standard_normal((400_000, 1))
    p = (1.0 / (1.0 + np.exp(-(f_u + tau_u * z1)))) * (
        1.0 / (1.0 + np.exp(-(h + tau_q * z2)))
    )
    logit_p = np.log(p) - np.log1p(-p)
    assert np.allclose(tau_logit[0], logit_p.std(axis=0), rtol=0.02)
    assert np.allclose(sd_words[0], n * p.std(axis=0), rtol=0.02)


def test_child_spread_product_correlated_matches_monte_carlo():
    # Same brute-force check as above, but with the two child deviations drawn
    # from the joint distribution VG20 actually fits. If the Cholesky applied to
    # the quadrature nodes is wrong, this is where it shows: the independent case
    # would still pass every other test in this file.
    rng = np.random.default_rng(20260819)
    f_u = np.array([[-2.0, 0.0, 1.0]])
    h = np.array([[-1.5, 0.0, 0.5]])
    tau_u, tau_q, rho, n = 0.8, 1.4, 0.368, 810
    tau_logit, sd_words = comparison.child_spread_product(
        f_u, h, np.array([tau_u]), np.array([tau_q]), n, rho=np.array([rho])
    )
    z1 = rng.standard_normal((400_000, 1))
    z2 = rho * z1 + np.sqrt(1.0 - rho**2) * rng.standard_normal((400_000, 1))
    p = (1.0 / (1.0 + np.exp(-(f_u + tau_u * z1)))) * (
        1.0 / (1.0 + np.exp(-(h + tau_q * z2)))
    )
    logit_p = np.log(p) - np.log1p(-p)
    assert np.allclose(tau_logit[0], logit_p.std(axis=0), rtol=0.02)
    assert np.allclose(sd_words[0], n * p.std(axis=0), rtol=0.02)


def test_child_spread_product_rho_zero_is_the_independent_case_exactly():
    # `rho=None` is not a separate approximation, it is the rho = 0 branch. Every
    # model before VG20 goes down the None path, so this equality is what lets the
    # correlation be added without restating any published uncorrelated number.
    f_u = np.array([[-3.0, -1.0, 0.5], [-2.5, -0.5, 1.0]])
    h = np.array([[-2.0, -0.5, 0.5], [-1.5, 0.0, 0.8]])
    tau_u, tau_q = np.array([0.8, 0.9]), np.array([1.3, 1.2])
    base = comparison.child_spread_product(f_u, h, tau_u, tau_q, 810)
    zero = comparison.child_spread_product(
        f_u, h, tau_u, tau_q, 810, rho=np.zeros(2)
    )
    assert np.array_equal(base[0], zero[0])
    assert np.array_equal(base[1], zero[1])


def test_child_spread_product_positive_rho_widens_and_negative_narrows():
    # The direction is the whole reason the parameter has to be carried:
    # log p_S = log p_U + log q gains 2 Cov, so the independent-draw derivation
    # understates the spoken spread whenever the correlation is positive. VG20
    # puts it at +0.368, so the published DS spoken tau was biased low.
    f_u = np.array([[-3.0, -1.0, 0.5]])
    h = np.array([[-2.0, -0.5, 0.5]])
    tau_u, tau_q = np.array([0.79]), np.array([1.28])
    base_tau, base_sd = comparison.child_spread_product(f_u, h, tau_u, tau_q, 810)
    up_tau, up_sd = comparison.child_spread_product(
        f_u, h, tau_u, tau_q, 810, rho=np.array([0.368])
    )
    down_tau, down_sd = comparison.child_spread_product(
        f_u, h, tau_u, tau_q, 810, rho=np.array([-0.368])
    )
    assert np.all(up_tau > base_tau) and np.all(up_sd > base_sd)
    assert np.all(down_tau < base_tau) and np.all(down_sd < base_sd)


def test_child_spread_product_stays_finite_at_the_correlation_bounds():
    # 1 - rho^2 is clipped at zero rather than trusted: a rho arriving as exactly
    # +/-1 from a summary, or a hair outside it from floating point, must not put
    # a NaN into a published between-child table.
    f_u = np.array([[-3.0, -1.0, 0.5]])
    h = np.array([[-2.0, -0.5, 0.5]])
    tau_u, tau_q = np.array([0.79]), np.array([1.28])
    for rho in (-1.0, 1.0):
        tau_logit, sd_words = comparison.child_spread_product(
            f_u, h, tau_u, tau_q, 810, rho=np.array([rho])
        )
        assert np.all(np.isfinite(tau_logit))
        assert np.all(np.isfinite(sd_words))
        assert np.all(sd_words >= 0.0)


def test_child_spread_product_tau_varies_with_age_at_constant_scales():
    # Both subject scales are constants, yet the induced spoken tau is age-varying
    # because the product passes through two nonlinearities. This is why the joint
    # model has no single spoken subject scale to contrast directly.
    f_u = np.array([[-6.0, -3.0, 0.0]])
    h = np.array([[-5.0, -2.0, 0.5]])
    tau_logit, _ = comparison.child_spread_product(
        f_u, h, np.array([1.0]), np.array([1.0]), 810
    )
    assert tau_logit[0, 0] > tau_logit[0, -1]
    assert not np.allclose(tau_logit[0, 0], tau_logit[0, -1])


def test_child_spread_product_quadrature_has_converged_at_the_default_node_count():
    # Monte Carlo cannot referee the far lower tail (see above), so the tail is
    # checked for node convergence instead: the shipped default must already agree
    # with a far finer grid on the most extreme scales the DS fits could produce.
    f_u = np.array([[-12.0, -6.0, -2.0]])
    h = np.array([[-10.0, -4.0, -1.0]])
    tau_u, tau_q = np.array([3.0]), np.array([3.0])
    coarse = comparison.child_spread_product(f_u, h, tau_u, tau_q, 810)
    fine = comparison.child_spread_product(f_u, h, tau_u, tau_q, 810, n_nodes=81)
    assert np.allclose(coarse[0], fine[0], rtol=1e-4)     # tau
    assert np.allclose(coarse[1], fine[1], rtol=5e-3)     # sd in words


def test_child_spread_product_is_stable_in_the_far_lower_tail():
    # DS spoken proportions at the young end are small enough that a naive
    # log(p) - log(1-p) on a clipped p would report the clip. Nothing may be
    # non-finite here.
    f_u = np.array([[-30.0, -20.0, -12.0]])
    h = np.array([[-25.0, -18.0, -10.0]])
    tau_logit, sd_words = comparison.child_spread_product(
        f_u, h, np.array([1.5]), np.array([2.0]), 810
    )
    assert np.all(np.isfinite(tau_logit))
    assert np.all(np.isfinite(sd_words))
    assert np.all(sd_words >= 0.0)


# ---------------------------------------------------------------- signing


def _sign_speech_trace(tmp_path, ages, p_u, q, r, pi):
    """A minimal VG15-shaped trace: one chain, two draws, three ages."""
    import xarray as xr

    n_draw = 2
    dims = ("chain", "draw", "plot_id")

    def da(v):
        return xr.DataArray(
            np.broadcast_to(np.asarray(v, dtype=float), (1, n_draw, len(ages))).copy(),
            dims=dims,
        )

    sign_only, both, speak_only = pi
    post = xr.Dataset({
        "p_u_plot": da(p_u), "q_plot": da(q), "r_plot": da(r),
        "p_any_plot": da(np.asarray(sign_only) + np.asarray(both) + np.asarray(speak_only)),
        "p_any_indep_plot": da(np.asarray(sign_only) + np.asarray(both) + np.asarray(speak_only)),
        "pi_sign_only_plot": da(sign_only), "pi_both_plot": da(both),
        "pi_speak_only_plot": da(speak_only),
    })
    # p_any_plot / p_any_indep_plot are unconditional in VG15; scale them here so
    # the fixture matches the engine rather than the cells.
    post["p_any_plot"] = post["p_any_plot"] * da(p_u)
    post["p_any_indep_plot"] = post["p_any_indep_plot"] * da(p_u)
    const = xr.Dataset({"X_plot": xr.DataArray(np.asarray(ages, dtype=float), dims=("plot_id",))})
    path = tmp_path / "trace.nc"
    xr.DataTree.from_dict({"posterior": post, "constant_data": const}).to_netcdf(str(path))
    return str(path)


def test_sign_speech_cells_are_scaled_by_comprehension(tmp_path):
    """The pi_* cells are conditional on understood, so they scale by p_u.

    Treating them as unconditional inflates every cell by 1/p_u, which at the
    youngest modelled ages is a factor of fifty — the difference between "a
    2-year-old has 28 sign-only words" and "197".
    """
    ages = [12.0, 24.0, 36.0]
    p_u = [0.02, 0.10, 0.30]
    cells = ([0.5, 0.4, 0.2], [0.1, 0.2, 0.3], [0.1, 0.2, 0.4])  # sign, both, speak
    path = _sign_speech_trace(tmp_path, ages, p_u, [0.3, 0.4, 0.5], [0.6, 0.6, 0.5], cells)

    got_ages, s = comparison.load_sign_speech_trajectory(path, 810)
    np.testing.assert_allclose(got_ages, ages)
    # sign_only words = p_u * pi_sign_only * n_trials
    np.testing.assert_allclose(s["sign_only"][0], np.array(p_u) * np.array(cells[0]) * 810)
    # and the three cells sum to total expressive vocabulary
    np.testing.assert_allclose(
        s["sign_only"] + s["both"] + s["speak_only"], s["any"], rtol=1e-9
    )


def test_sign_speech_spoken_is_reconstructed_from_p_u_and_q(tmp_path):
    """VG15 emits no p_s_plot — spoken is a ratio of understood in that engine."""
    ages = [12.0, 24.0, 36.0]
    p_u, q = [0.02, 0.10, 0.30], [0.3, 0.4, 0.5]
    path = _sign_speech_trace(
        tmp_path, ages, p_u, q, [0.6, 0.6, 0.5],
        ([0.5, 0.4, 0.2], [0.1, 0.2, 0.3], [0.1, 0.2, 0.4]),
    )
    _, s = comparison.load_sign_speech_trajectory(path, 810)
    np.testing.assert_allclose(s["spoken"][0], np.array(p_u) * np.array(q) * 810)
    np.testing.assert_allclose(s["understood"][0], np.array(p_u) * 810)


def test_sign_speech_r_stays_a_fraction(tmp_path):
    """r is a ratio by construction; a word count of it would be meaningless."""
    r = [0.6, 0.6, 0.5]
    path = _sign_speech_trace(
        tmp_path, [12.0, 24.0, 36.0], [0.02, 0.10, 0.30], [0.3, 0.4, 0.5], r,
        ([0.5, 0.4, 0.2], [0.1, 0.2, 0.3], [0.1, 0.2, 0.4]),
    )
    _, s = comparison.load_sign_speech_trajectory(path, 810)
    np.testing.assert_allclose(s["r"][0], r)
    assert s["r"].max() <= 1.0


# ------------------------------------------- subject-effect correlation


def _subject_effect_trace(tmp_path, u, q):
    """A trace carrying only the two per-child deviation vectors.

    ``u`` and ``q`` are ``(n_draw, n_child)``; one chain keeps the fixture small.
    """
    import xarray as xr

    dims = ("chain", "draw", "subject_id")

    def da(v):
        return xr.DataArray(np.asarray(v, dtype=float)[None, ...], dims=dims)

    post = xr.Dataset({"delta_subj_u": da(u), "delta_subj_q": da(q)})
    path = tmp_path / "trace.nc"
    xr.DataTree.from_dict({"posterior": post}).to_netcdf(str(path))
    return str(path)


def _patch_trace(monkeypatch, path):
    monkeypatch.setattr(comparison, "trace_path", lambda key: path)


def test_subject_effect_correlation_matches_numpy(tmp_path, monkeypatch):
    """The vectorised per-draw correlation is the ordinary one, draw by draw."""
    rng = np.random.default_rng(0)
    u = rng.normal(size=(4, 50))
    q = 0.6 * u + rng.normal(size=(4, 50))
    _patch_trace(monkeypatch, _subject_effect_trace(tmp_path, u, q))

    got, n_children = comparison.subject_effect_correlation("vg10", thin=1)

    assert n_children == 50
    expected = [np.corrcoef(u[i], q[i])[0, 1] for i in range(4)]
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_subject_effect_correlation_recovers_independence(tmp_path, monkeypatch):
    """Independent deviations give a correlation centred on zero, not a bias."""
    rng = np.random.default_rng(1)
    u = rng.normal(size=(200, 400))
    q = rng.normal(size=(200, 400))
    _patch_trace(monkeypatch, _subject_effect_trace(tmp_path, u, q))

    got, _ = comparison.subject_effect_correlation("vg10", thin=1)

    assert abs(float(np.median(got))) < 0.02


def test_subject_effect_correlation_needs_both_vectors(tmp_path, monkeypatch):
    """A trace missing a deviation vector must say so, not return something."""
    import xarray as xr

    post = xr.Dataset({
        "delta_subj_u": xr.DataArray(
            np.zeros((1, 2, 3)), dims=("chain", "draw", "subject_id")
        )
    })
    path = tmp_path / "trace.nc"
    xr.DataTree.from_dict({"posterior": post}).to_netcdf(str(path))
    _patch_trace(monkeypatch, str(path))

    with pytest.raises(ValueError, match="delta_subj_q"):
        comparison.subject_effect_correlation("vg10")


# ============================================================
# VG19: the age-varying child scale
# ============================================================


def test_child_scale_of_age_is_tau0_at_the_reference_age():
    """`tau0` is defined as the spread AT the reference age, so D = 0 must return it."""
    tau0 = np.array([0.9, 1.4, 0.3])
    tau1 = np.array([0.5, 0.2, 0.8])
    rho = np.array([0.43, -0.6, 0.0])
    ages = np.array([24.0, 36.0, 60.0])
    sd = comparison.child_scale_of_age(tau0, tau1, rho, ages, ref_age_months=36.0)
    assert sd.shape == (3, 3)
    np.testing.assert_allclose(sd[:, 1], tau0, rtol=1e-12)


def test_child_scale_of_age_reduces_to_the_constant_model_at_tau1_zero():
    """`tau1 = 0` is the nesting: the model of record's constant offset, at every age."""
    tau0 = np.array([0.9, 1.4])
    ages = np.linspace(8.0, 115.0, 25)
    sd = comparison.child_scale_of_age(
        tau0, np.zeros(2), np.array([0.43, -0.2]), ages, ref_age_months=36.0
    )
    np.testing.assert_allclose(sd, np.broadcast_to(tau0[:, None], sd.shape), rtol=1e-12)


def test_child_scale_of_age_matches_monte_carlo():
    """The closed form is Var(b0 + b1 D); check it against draws from the same block."""
    rng = np.random.default_rng(20260821)
    tau0, tau1, rho = 0.9, 0.35, 0.43
    ages = np.array([12.0, 36.0, 84.0])
    z = rng.standard_normal((400_000, 2))
    b0 = tau0 * z[:, 0]
    b1 = tau1 * (rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1])
    want = np.array([(b0 + b1 * (a - 36.0) / 12.0).std(ddof=0) for a in ages])
    got = comparison.child_scale_of_age(
        np.array([tau0]), np.array([tau1]), np.array([rho]), ages, ref_age_months=36.0
    )[0]
    np.testing.assert_allclose(got, want, rtol=3e-3)


def test_child_spread_single_accepts_an_age_varying_scale():
    """A constant scale passed as (n_draw, n_age) must reproduce the (n_draw,) result."""
    rng = np.random.default_rng(11)
    f = rng.normal(-1.0, 0.5, size=(7, 5))
    tau = np.array([0.8, 1.1, 0.5, 0.9, 1.3, 0.7, 1.0])
    a_logit, a_words = comparison.child_spread_single(f, tau, 810)
    b_logit, b_words = comparison.child_spread_single(
        f, np.broadcast_to(tau[:, None], f.shape).copy(), 810
    )
    np.testing.assert_allclose(a_logit, b_logit, rtol=1e-12)
    np.testing.assert_allclose(a_words, b_words, rtol=1e-12)


def test_child_spread_product_accepts_an_age_varying_scale():
    rng = np.random.default_rng(12)
    f_u = rng.normal(-0.5, 0.4, size=(6, 4))
    h = rng.normal(-1.5, 0.4, size=(6, 4))
    tu = np.array([0.9, 1.0, 0.8, 1.2, 0.7, 1.1])
    tq = np.array([1.2, 1.1, 1.3, 0.9, 1.0, 1.4])
    a = comparison.child_spread_product(f_u, h, tu, tq, 810)
    b = comparison.child_spread_product(
        f_u,
        h,
        np.broadcast_to(tu[:, None], f_u.shape).copy(),
        np.broadcast_to(tq[:, None], h.shape).copy(),
        810,
    )
    for x, y in zip(a, b, strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-12)


def test_child_spread_product_refuses_an_age_varying_scale_with_a_cross_outcome_rho():
    """VG19 and VG20's rho_uq together are a 4x4 this quadrature does not model."""
    f_u = np.zeros((3, 4))
    h = np.zeros((3, 4))
    with pytest.raises(ValueError, match="4x4"):
        comparison.child_spread_product(
            f_u, h, np.ones((3, 4)), np.ones(3), 810, rho=np.full(3, 0.3)
        )


def test_tau_to_draw_age_rejects_a_mismatched_grid():
    f = np.zeros((3, 5))
    with pytest.raises(ValueError, match="expected"):
        comparison.child_spread_single(f, np.ones((3, 4)), 810)


def _slope_trace(tmp_path, ages, f_u, h, tu, tq):
    """A VG19-shaped trace: both outcomes carry (tau0, tau1, rho01), not a scalar."""
    import xarray as xr

    n_draw = f_u.shape[0]

    def grid(v):
        return xr.DataArray(v.reshape(1, n_draw, len(ages)).copy(), dims=("chain", "draw", "plot_id"))

    def scalar(v):
        return xr.DataArray(np.asarray(v, dtype=float).reshape(1, n_draw).copy(), dims=("chain", "draw"))

    post = xr.Dataset({
        "f_u_plot": grid(f_u),
        "h_plot": grid(h),
        "tau_subj_u_0": scalar(tu[0]), "tau_subj_u_1": scalar(tu[1]), "tau_subj_u_rho": scalar(tu[2]),
        "tau_subj_q_0": scalar(tq[0]), "tau_subj_q_1": scalar(tq[1]), "tau_subj_q_rho": scalar(tq[2]),
    })
    const = xr.Dataset({"X_plot": xr.DataArray(np.asarray(ages, dtype=float), dims=("plot_id",))})
    path = tmp_path / "slope_trace.nc"
    xr.DataTree.from_dict({"posterior": post, "constant_data": const}).to_netcdf(str(path))
    return str(path)


def _slope_fixture(tmp_path, monkeypatch):
    ages = np.array([12.0, 24.0, 36.0, 48.0, 60.0])
    rng = np.random.default_rng(21)
    f_u = rng.normal(-1.0, 0.3, size=(4, len(ages)))
    h = rng.normal(-1.5, 0.3, size=(4, len(ages)))
    tu = (np.array([0.8, 0.9, 1.0, 1.1]), np.array([0.20, 0.30, 0.10, 0.25]),
          np.array([0.4, -0.3, 0.0, 0.6]))
    tq = (np.array([1.2, 1.3, 1.1, 1.4]), np.array([0.15, 0.05, 0.30, 0.20]),
          np.array([-0.2, 0.5, 0.3, 0.1]))
    _patch_trace(monkeypatch, _slope_trace(tmp_path, ages, f_u, h, tu, tq))
    return ages, f_u, h, tu, tq


def test_subject_heterogeneity_reads_the_rate_not_just_the_reference_age_spread(
    tmp_path, monkeypatch
):
    """VG19's between-child spread is a curve; reading `tau_subj_u` gives a line.

    A slope model still emits `tau_subj_u` as a Deterministic equal to `tau0`, so
    a loader that reads it succeeds and silently reports the 36-month spread at
    every age -- discarding `tau1` and `rho01`. This is the #224 defect class: a
    fitted parameter thrown away by the derived quantity that exists to use it.
    """
    ages, f_u, _h, tu, _tq = _slope_fixture(tmp_path, monkeypatch)
    _, tau_logit, _, _ = comparison.subject_heterogeneity("vg19", "understood")

    # `child_spread_single` returns the scale itself, so this is exact.
    want = comparison.child_scale_of_age(tu[0], tu[1], tu[2], ages, ref_age_months=36.0)
    np.testing.assert_allclose(tau_logit, want, rtol=1e-12)

    # And it is genuinely age-varying: the flat reading would be wrong everywhere
    # except at the reference age, where the two must agree exactly.
    ref = list(ages).index(36.0)
    np.testing.assert_allclose(tau_logit[:, ref], tu[0], rtol=1e-12)
    assert not np.allclose(tau_logit, tu[0][:, None]), "scale collapsed to tau0"


def test_subject_heterogeneity_slope_scale_follows_the_caller_grid(tmp_path, monkeypatch):
    """The scale is a function of age, so it is built on the grid actually reported.

    Interpolating it from the model's native grid would be a different quantity:
    the curve is a square root of a quadratic, not a smooth the interpolator can
    stand in for.
    """
    _ages, _f_u, _h, tu, _tq = _slope_fixture(tmp_path, monkeypatch)
    want_at = np.array([18.0, 30.0, 42.0])
    grid, tau_logit, _, _ = comparison.subject_heterogeneity(
        "vg19", "understood", ages=want_at
    )
    np.testing.assert_allclose(grid, want_at)
    want = comparison.child_scale_of_age(tu[0], tu[1], tu[2], want_at, ref_age_months=36.0)
    np.testing.assert_allclose(tau_logit, want, rtol=1e-12)


def test_subject_heterogeneity_spoken_carries_both_rates(tmp_path, monkeypatch):
    """The induced spoken scale must differ from the constant-offset reading."""
    ages, f_u, h, tu, tq = _slope_fixture(tmp_path, monkeypatch)
    _, tau_logit, sd_words, n = comparison.subject_heterogeneity("vg19", "spoken")

    want, want_sd = comparison.child_spread_product(
        f_u, h,
        comparison.child_scale_of_age(tu[0], tu[1], tu[2], ages, ref_age_months=36.0),
        comparison.child_scale_of_age(tq[0], tq[1], tq[2], ages, ref_age_months=36.0),
        n,
    )
    np.testing.assert_allclose(tau_logit, want, rtol=1e-12)
    np.testing.assert_allclose(sd_words, want_sd, rtol=1e-12)

    flat, _ = comparison.child_spread_product(f_u, h, tu[0], tq[0], n)
    assert not np.allclose(tau_logit, flat), "spoken scale ignored the rates"


def test_product_marginal_kappa_matches_the_graph_form_and_its_limits():
    """The NumPy port must agree with ``likelihood_utils.product_marginal_concentration``.

    That is the form the graph's ``product_marginal`` spoken fallback uses, and
    the DS side of the spoken dispersion contrast now depends on this port
    reproducing it. Also the documented limit: at kappa_U -> inf and p_U = 1 the
    marginal concentration is kappa_S itself.
    """
    import pytensor

    from vocab_growth.models.likelihood_utils import product_marginal_concentration

    p_u = np.array([0.05, 0.30, 0.60, 0.95])
    k_u = np.array([8.0, 40.0, 120.0, 30.0])
    q = np.array([0.10, 0.40, 0.70, 0.99])
    k_s = np.array([5.0, 20.0, 60.0, 15.0])
    graph = product_marginal_concentration(p_u, k_u, q, k_s, epsilon=1e-9)
    want = np.asarray(pytensor.function([], graph)())
    got = comparison.product_marginal_kappa(p_u, k_u, q, k_s)
    assert np.allclose(got, want, rtol=1e-9), (got, want)

    limit = comparison.product_marginal_kappa(
        np.array([1.0]), np.array([1e12]), np.array([0.4]), np.array([20.0])
    )
    assert np.allclose(limit, 20.0, rtol=1e-4), limit


def test_product_marginal_kappa_matches_monte_carlo_moments():
    """kappa_eff is the concentration of the Beta with the product's mean and variance."""
    rng = np.random.default_rng(3)
    p_u, k_u, q, k_s = 0.30, 25.0, 0.45, 12.0
    theta_u = rng.beta(p_u * k_u, (1 - p_u) * k_u, 2_000_000)
    theta_s = rng.beta(q * k_s, (1 - q) * k_s, 2_000_000)
    prod = theta_u * theta_s
    m, v = prod.mean(), prod.var()
    want = m * (1 - m) / v - 1
    got = float(comparison.product_marginal_kappa(p_u, k_u, q, k_s))
    assert abs(got - want) / want < 0.01, (got, want)


def test_load_marginal_spoken_trajectory_rejects_a_univariate_model():
    with pytest.raises(ValueError, match="not a bivariate model"):
        comparison.load_marginal_spoken_trajectory("vg11")
