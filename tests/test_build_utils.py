# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the pure-NumPy build helpers in ``models.build_utils``.

These pin the exact formulas hoisted out of the engines so that any transcription
drift (e.g. a ``ddof=0`` regression in the age z-scores) is caught immediately.
"""

import numpy as np
import pytest

from vocab_growth.models.build_utils import (
    CLAMP_SOFTNESS,
    construct_age_grids,
    require_integral_counts,
    require_valid_counts,
    standardize_ages,
    standardize_ages_to_z,
    standardize_anchor_ages,
    validate_ell_bounds,
)

# --- standardize_ages -------------------------------------------------------


def test_standardize_ages_known_values():
    X = np.array([[10.0], [20.0], [30.0]])
    mean, std, X_z = standardize_ages(X)
    assert np.isclose(mean, 20.0)
    assert np.isclose(std, 10.0)  # ddof=1 sample std
    np.testing.assert_allclose(X_z, [[-1.0], [0.0], [1.0]])


def test_standardize_ages_uses_ddof1():
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    _, std, _ = standardize_ages(X)
    assert np.isclose(std, np.std(X, ddof=1))
    assert not np.isclose(std, np.std(X, ddof=0))


def test_standardize_ages_affine_inverse():
    rng = np.random.default_rng(0)
    X = rng.normal(30.0, 8.0, size=(50, 1))
    mean, std, X_z = standardize_ages(X)
    np.testing.assert_allclose(X_z * std + mean, X)
    assert np.isclose(X_z.mean(), 0.0, atol=1e-12)


def test_standardize_ages_degenerate_raises():
    with pytest.raises(ValueError):
        standardize_ages(np.array([[5.0], [5.0], [5.0]]))


def test_standardize_ages_nan_raises():
    with pytest.raises(ValueError):
        standardize_ages(np.array([[5.0], [np.nan], [7.0]]))


# --- validate_ell_bounds ----------------------------------------------------


def test_validate_ell_bounds_ok():
    assert validate_ell_bounds((6, 18)) == (6.0, 18.0)


def test_validate_ell_bounds_nonpositive_raises():
    with pytest.raises(ValueError):
        validate_ell_bounds((0, 18))
    with pytest.raises(ValueError):
        validate_ell_bounds((-1, 18))


def test_validate_ell_bounds_non_increasing_raises():
    with pytest.raises(ValueError):
        validate_ell_bounds((18, 6))
    with pytest.raises(ValueError):
        validate_ell_bounds((6, 6))


# --- standardize_anchor_ages ------------------------------------------------


def test_standardize_anchor_ages_known_values():
    y_z, o_z = standardize_anchor_ages((12, 20), X_obs_mean=19.6, X_obs_std=5.9)
    assert np.isclose(y_z, (12 - 19.6) / 5.9)
    assert np.isclose(o_z, (20 - 19.6) / 5.9)


def test_slope_anchors_and_kappa_anchors_share_one_conversion():
    # One conversion, so the mean trajectory's anchors and the dispersion
    # curve's cannot drift apart. `slope_anchor_logit_coeffs` was a body-less
    # alias for this and the engines now call it directly.
    a_z, b_z = standardize_anchor_ages((24, 84), X_obs_mean=30.0, X_obs_std=12.0)
    assert np.isclose(a_z, (24 - 30) / 12)
    assert np.isclose(b_z, (84 - 30) / 12)


def test_standardize_ages_to_z_agrees_with_the_pair_form():
    # The n-ary form is the same arithmetic, so the three-anchor signed hump and
    # the two-anchor trajectories cannot diverge either.
    kwargs = {"X_obs_mean": 30.0, "X_obs_std": 12.0}

    assert standardize_ages_to_z((24, 84), **kwargs) == standardize_anchor_ages(
        (24, 84), **kwargs
    )
    assert standardize_ages_to_z((15.0, 36.0, 96.0), **kwargs) == (
        (15.0 - 30) / 12,
        (36.0 - 30) / 12,
        (96.0 - 30) / 12,
    )


def test_standardize_anchor_ages_preserves_the_relative_position_of_an_age():
    # The property the two-anchor kappa prior relies on: the interpolation
    # weight of any age between the anchors is the same under any
    # standardisation, because z is affine in age.
    def weight(mean, sd):
        y_z, o_z = standardize_anchor_ages((12, 20), X_obs_mean=mean, X_obs_std=sd)
        return ((18.0 - mean) / sd - y_z) / (o_z - y_z)

    assert np.isclose(weight(19.6, 5.9), weight(33.0, 15.0))
    assert np.isclose(weight(19.6, 5.9), (18 - 12) / (20 - 12))


# --- construct_age_grids ----------------------------------------------------

N_PLOT = 100
AGES_QUERY = [12, 24, 36]
SLOPE_ANCHORS = (24, 84)


def _grids(
    *,
    use_gp_anchor=False,
    gp_anchor_age_months=None,
    ages_query=AGES_QUERY,
    gp_domain_months=None,
):
    X = np.linspace(8.0, 60.0, 40).reshape(-1, 1)
    mean, std, X_z = standardize_ages(X)
    grids = construct_age_grids(
        X,
        X_z,
        X_obs_mean=mean,
        X_obs_std=std,
        n_plot=N_PLOT,
        ages_query=ages_query,
        slope_anchors=SLOPE_ANCHORS,
        use_gp_anchor=use_gp_anchor,
        gp_anchor_age_months=gp_anchor_age_months,
        gp_domain_months=gp_domain_months,
    )
    return X, mean, std, grids


def test_construct_age_grids_shapes_no_anchor():
    X, _, _, g = _grids()
    n = X.shape[0]
    assert g.X_plot.shape == (N_PLOT, 1)
    assert g.n_plot == N_PLOT
    assert g.n_query == len(AGES_QUERY)
    assert g.X_all_z.shape == (n + N_PLOT + len(AGES_QUERY), 1)
    assert g.n_all == n + N_PLOT + len(AGES_QUERY)
    assert g.i_anchor is None
    assert g.anchor_age_months is None


def test_construct_age_grids_plot_endpoints():
    X, _, _, g = _grids()
    assert np.isclose(g.X_plot[0, 0], X.min())
    assert np.isclose(g.X_plot[-1, 0], X.max())


def test_construct_age_grids_index_tuples():
    X, _, _, g = _grids()
    n = X.shape[0]
    assert g.i_obs == (0, n)
    assert g.i_plot == (n, n + N_PLOT)
    assert g.i_query == (n + N_PLOT, n + N_PLOT + len(AGES_QUERY))


def test_construct_age_grids_blocks_are_zscored():
    X, mean, std, g = _grids()
    n = X.shape[0]
    np.testing.assert_allclose(g.X_all_z[n : n + N_PLOT], (g.X_plot - mean) / std)
    np.testing.assert_allclose(
        g.X_all_z[n + N_PLOT : n + N_PLOT + len(AGES_QUERY)],
        (g.X_query - mean) / std,
    )


def test_construct_age_grids_separates_hsgp_domain_from_query_grid():
    _, _, _, first = _grids(ages_query=[12, 24])
    _, _, _, second = _grids(ages_query=[12, 24, 48])

    np.testing.assert_allclose(first.X_gp_domain_z, second.X_gp_domain_z)


def test_construct_age_grids_uses_explicit_hsgp_domain():
    _, mean, std, grids = _grids(gp_domain_months=(6, 90))

    np.testing.assert_allclose(
        grids.X_gp_domain_z,
        (np.array([[6.0], [90.0]]) - mean) / std,
    )


def test_construct_age_grids_rejects_query_outside_hsgp_domain():
    with pytest.raises(ValueError, match="must lie inside gp_domain_months"):
        _grids(ages_query=[6, 24], gp_domain_months=(8, 60))


def test_construct_age_grids_anchor_default_midpoint():
    X, mean, std, g = _grids(use_gp_anchor=True)
    n = X.shape[0]
    assert g.anchor_age_months == 54.0  # midpoint of (24, 84)
    assert g.i_anchor == n + N_PLOT + len(AGES_QUERY)
    assert g.X_all_z.shape == (n + N_PLOT + len(AGES_QUERY) + 1, 1)
    np.testing.assert_allclose(g.X_all_z[g.i_anchor, 0], (54.0 - mean) / std)


def test_construct_age_grids_anchor_explicit_age():
    X, mean, std, g = _grids(use_gp_anchor=True, gp_anchor_age_months=19.0)
    assert g.anchor_age_months == 19.0
    np.testing.assert_allclose(g.X_all_z[g.i_anchor, 0], (19.0 - mean) / std)


# --- require_integral_counts ----------------------------------------------------


def test_require_integral_counts_accepts_whole_numbers():
    require_integral_counts(np.array([0.0, 3.0, 810.0]), "spoken")


def test_require_integral_counts_rejects_fractions_with_examples():
    with pytest.raises(ValueError, match="spoken contains 2 non-integral"):
        require_integral_counts(np.array([1.0, 2.5, 3.0, 4.25]), "spoken")


def test_require_integral_counts_rejects_infinities():
    # np.floor(inf) == inf, so an infinity slips past the integrality check and
    # would cast to an arbitrary integer; it must be rejected explicitly (#236).
    with pytest.raises(ValueError, match="understood contains 1 non-finite"):
        require_integral_counts(np.array([1.0, np.inf, 3.0]), "understood")


def test_require_integral_counts_reports_nan_as_non_finite():
    with pytest.raises(ValueError, match="spoken contains 1 non-finite"):
        require_integral_counts(np.array([1.0, np.nan]), "spoken")


# --- require_valid_counts ----------------------------------------------------
#
# The contract the nested spoken likelihood already gets from
# nested_outcome_spec, applied to a count column an engine casts itself. VG13
# cast `understood` to int before any check, so a fractional value would have
# been silently truncated (#240).


def test_require_valid_counts_accepts_integral_in_range():
    require_valid_counts(np.array([0.0, 3.0, 810.0]), "understood", 810)


def test_require_valid_counts_rejects_fractions():
    with pytest.raises(ValueError, match="understood contains 1 non-integral"):
        require_valid_counts(np.array([1.0, 2.5]), "understood", 810)


def test_require_valid_counts_rejects_non_finite():
    with pytest.raises(ValueError, match="non-finite"):
        require_valid_counts(np.array([1.0, np.inf]), "understood", 810)


def test_require_valid_counts_rejects_out_of_range():
    with pytest.raises(ValueError, match="between 0 and n_trials"):
        require_valid_counts(np.array([1.0, 811.0]), "understood", 810)
    with pytest.raises(ValueError, match="between 0 and n_trials"):
        require_valid_counts(np.array([-1.0, 3.0]), "understood", 810)


def test_require_valid_counts_delegates_the_non_finite_message():
    # require_valid_counts defers finiteness and integrality to
    # require_integral_counts, so the failure names the offending values rather
    # than reporting a bare "non-finite observed count(s)" (#236).
    with pytest.raises(ValueError, match="understood contains 1 non-finite"):
        require_valid_counts(np.array([1.0, np.inf]), "understood", 810)

# --- CLAMP_SOFTNESS ----------------------------------------------------------


def test_the_months_and_standardised_soft_clamps_are_exactly_equal():
    """One constant, two implementations, and they must not merely be close.

    `gp_utils._soft_clamp_z` works in standardised age for the graph;
    `vocab_growth.report_illustrations` works in months for the methods chapter's
    figures, which `sync_report_figures.py` does not validate. Because
    CLAMP_SOFTNESS is expressed per unit of anchor span, the standard deviation
    cancels: `beta_z * (hi_z - z)` reduces to `CLAMP_SOFTNESS * (hi - age) / span`.
    So the two agree exactly in algebra, and the assertion below is a tolerance
    tight enough that no changed constant can hide inside it: 1e-9 absolute with
    `rtol=0`, against clamp arguments of order 1. Not literal float identity --
    the two expressions reach the same value by different operation orders, so the
    last bit is not guaranteed to match and asserting it would make this test
    brittle rather than strict.
    """
    import numpy as np

    lo, hi = 24.0, 84.0
    mean, std = 30.0, 12.0
    ages = np.array([8.0, 24.0, 48.0, 84.0, 96.0, 115.0])

    # Months form, as the figure module computes it.
    beta_months = CLAMP_SOFTNESS / (hi - lo)
    months = hi - np.logaddexp(0.0, beta_months * (hi - ages)) / beta_months

    # Standardised form, as the graph computes it, mapped back to months.
    z = (ages - mean) / std
    lo_z, hi_z = (lo - mean) / std, (hi - mean) / std
    beta_z = CLAMP_SOFTNESS / (hi_z - lo_z)
    clamped_z = hi_z - np.logaddexp(0.0, beta_z * (hi_z - z)) / beta_z
    from_z = clamped_z * std + mean

    assert np.allclose(months, from_z, rtol=0, atol=1e-9), (months, from_z)
    # Below the anchor the clamp is near-identity; above it, flat.
    assert months[3] < hi and hi - months[3] < 0.9
    assert abs(months[-1] - hi) < 1e-6


def test_the_soft_clamp_constant_has_one_home():
    """The figure module and the graph builder must read the same object."""
    from vocab_growth import report_illustrations
    from vocab_growth.models import gp_utils

    assert gp_utils.CLAMP_SOFTNESS is CLAMP_SOFTNESS
    assert report_illustrations.CLAMP_SOFTNESS is CLAMP_SOFTNESS

