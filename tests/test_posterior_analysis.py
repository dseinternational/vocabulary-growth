# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import types

import numpy as np
import pytest
import xarray as xr

from vocab_growth.posterior_analysis import (
    COUNT_BUCKET_THRESHOLDS,
    MAX_MONTH_SNAP_OFFSET,
    add_probability_estimand_columns,
    expand_observed_to_obs_id,
    extract_posterior,
    extract_posterior_predictive,
    monthly_summary_table,
    posterior_summary_table,
)


def _fake_trace(group: str, name: str, values: np.ndarray):
    """Minimal stand-in for an InferenceData with one (chain, draw, plot_id) var."""
    n_chain, n_draw, n_plot = values.shape
    ds = xr.Dataset(
        {name: (("chain", "draw", "plot_id"), values)},
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            "plot_id": np.arange(n_plot),
        },
    )
    return types.SimpleNamespace(**{group: ds})


def test_extract_posterior_shape_and_order():
    values = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    trace = _fake_trace("posterior", "f", values)
    out = extract_posterior(trace, "f", "plot_id")
    # (n_plot_id, n_chain * n_draw); sample axis iterates chain-outer, draw-inner
    assert out.shape == (4, 6)
    np.testing.assert_array_equal(out, values.transpose(2, 0, 1).reshape(4, 6))


def test_extract_posterior_predictive_is_integer():
    values = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4) + 0.5
    trace = _fake_trace("posterior_predictive", "y", values)
    out = extract_posterior_predictive(trace, "y", "plot_id")
    assert out.shape == (4, 6)
    assert np.issubdtype(out.dtype, np.integer)
    # float 0.5+ values are truncated toward zero by the int cast
    np.testing.assert_array_equal(
        out, values.transpose(2, 0, 1).reshape(4, 6).astype(int)
    )

N_SAMPLES = 2000
EXPECTED_COLUMNS = {
    "age_months",
    "p_median", "p_ci50_lo", "p_ci50_hi", "p_ci_lo", "p_ci_hi",
    "Ey_median", "Ey_ci50_lo", "Ey_ci50_hi", "Ey_ci_lo", "Ey_ci_hi",
    "Y_median", "Y_ci50_lo", "Y_ci50_hi", "Y_ci_lo", "Y_ci_hi",
    "P(Y=0)", "P(Y<=5)", "P(Y<=400)", "P(Y>400)",
}


def _summary():
    # Row order is (age 24, age 12); the table should come back age-sorted.
    X_query = np.array([24, 12])
    p_query = np.vstack([np.full(N_SAMPLES, 0.5), np.full(N_SAMPLES, 0.25)])
    y_query = np.vstack([np.full(N_SAMPLES, 400), np.zeros(N_SAMPLES, dtype=int)])
    return posterior_summary_table(X_query, p_query, y_query, n_trials=800, ci_prob=0.89)


def test_posterior_summary_columns_and_sorting():
    df = _summary()
    assert EXPECTED_COLUMNS.issubset(df.columns)
    # Sorted ascending by age regardless of input order.
    assert list(df["age_months"]) == [12.0, 24.0]


def test_posterior_summary_values_age_12():
    df = _summary()
    row = df[df["age_months"] == 12.0].iloc[0]
    assert np.isclose(row["p_median"], 0.25)
    assert np.isclose(row["Ey_median"], 0.25 * 800)  # p * n_trials
    assert row["Y_median"] == 0.0
    assert np.isclose(row["P(Y=0)"], 1.0)  # every draw is zero
    assert np.isclose(row["P(Y>400)"], 0.0)


def test_posterior_summary_values_age_24():
    df = _summary()
    row = df[df["age_months"] == 24.0].iloc[0]
    assert np.isclose(row["p_median"], 0.5)
    assert row["Y_median"] == 400.0
    assert np.isclose(row["P(Y=0)"], 0.0)
    assert np.isclose(row["P(Y<=400)"], 1.0)  # all draws == 400
    assert np.isclose(row["P(Y>400)"], 0.0)


def test_add_probability_estimand_columns_makes_population_and_subject_explicit():
    summary = _summary()
    p_population = np.vstack([
        np.full(N_SAMPLES, 0.25),
        np.full(N_SAMPLES, 0.50),
    ])
    p_subject = np.vstack([
        np.full(N_SAMPLES, 0.20),
        np.full(N_SAMPLES, 0.60),
    ])

    out = add_probability_estimand_columns(
        summary,
        p_population,
        p_subject,
        n_trials=800,
        ci_prob=0.89,
    )

    row = out[out["age_months"] == 12.0].iloc[0]
    assert np.isclose(row["p_population_median"], 0.25)
    assert np.isclose(row["Ey_population_median"], 200.0)
    assert np.isclose(row["p_subject_marginal_median"], 0.20)
    assert np.isclose(row["Ey_subject_marginal_median"], 160.0)


# ---------------------------------------------------------------------------
# Whole-month summary table
# ---------------------------------------------------------------------------


def _plot_grid_draws(lo: float = 8.0, hi: float = 90.0, n_plot: int = 500, seed: int = 7):
    """A plot grid with a smooth rising trajectory and predictive count draws."""
    rng = np.random.default_rng(seed)
    X_plot = np.linspace(lo, hi, n_plot)
    p = 1.0 / (1.0 + np.exp(-(X_plot - 45.0) / 12.0))
    p_plot = np.repeat(p[:, None], N_SAMPLES, axis=1)
    y_plot = rng.binomial(800, np.clip(p_plot, 1e-6, 1 - 1e-6))
    return X_plot, p_plot, y_plot


def test_monthly_summary_covers_every_whole_month_in_range():
    X_plot, p_plot, y_plot = _plot_grid_draws()
    monthly = monthly_summary_table(X_plot, p_plot, y_plot, n_trials=800)

    ages = monthly["age_months"].to_numpy()
    np.testing.assert_array_equal(ages, np.arange(8, 91))
    # One row per month, no gaps or repeats, ascending.
    assert len(monthly) == 83
    assert monthly["age_months"].is_monotonic_increasing


def test_monthly_summary_records_its_snap_provenance():
    X_plot, p_plot, y_plot = _plot_grid_draws()
    monthly = monthly_summary_table(X_plot, p_plot, y_plot, n_trials=800)

    offsets = monthly["grid_offset_months"].to_numpy()
    grid_ages = monthly["grid_age_months"].to_numpy()
    # The recorded grid age is the stated month plus the recorded offset, and the
    # offset stays inside the documented bound.
    np.testing.assert_allclose(grid_ages, monthly["age_months"].to_numpy() + offsets)
    assert np.abs(offsets).max() <= MAX_MONTH_SNAP_OFFSET
    # A 500-point grid over 82 months snaps to well under a week.
    assert np.abs(offsets).max() < 0.1


def test_monthly_summary_rejects_a_grid_too_coarse_for_months():
    # 20 points over 82 months is a ~4-month step: snapping would silently
    # mislabel ages, so the table must refuse rather than emit wrong ages.
    X_plot, p_plot, y_plot = _plot_grid_draws(n_plot=20)
    with pytest.raises(ValueError, match="too coarse for whole-month"):
        monthly_summary_table(X_plot, p_plot, y_plot, n_trials=800)


def test_monthly_summary_agrees_with_the_canonical_table_at_shared_ages():
    """The monthly table must be the canonical table at finer resolution.

    Both read the same posterior draws through the same row builder, so at a
    canonical query age the monthly row is the query row up to the sub-month
    difference in age. This is the check that the plot-grid derivation is a
    refinement of the reported table rather than a different quantity.
    """
    X_plot, p_plot, y_plot = _plot_grid_draws()
    monthly = monthly_summary_table(X_plot, p_plot, y_plot, n_trials=800)

    canonical_ages = [12, 24, 36, 48, 60, 72, 84]
    nearest = [int(np.abs(X_plot - a).argmin()) for a in canonical_ages]
    canonical = posterior_summary_table(
        np.array(canonical_ages, dtype=float),
        p_plot[nearest, :],
        y_plot[nearest, :],
        n_trials=800,
    )

    shared = [c for c in canonical.columns if c != "age_months"]
    monthly_rows = monthly.set_index("age_months").loc[canonical_ages, shared]
    np.testing.assert_allclose(
        monthly_rows.to_numpy(dtype=float),
        canonical.set_index("age_months")[shared].to_numpy(dtype=float),
    )


def test_monthly_summary_counts_observed_administrations_per_month():
    X_plot, p_plot, y_plot = _plot_grid_draws()
    # Three administrations at 24 months (one recorded at 23.7, which rounds to
    # 24), one at 25, and one outside the grid entirely.
    X_obs = np.array([24.0, 24.0, 23.7, 25.0, 200.0])
    monthly = monthly_summary_table(
        X_plot, p_plot, y_plot, n_trials=800, X_obs=X_obs
    )
    by_month = monthly.set_index("age_months")["n_obs"]
    assert by_month.loc[24] == 3
    assert by_month.loc[25] == 1
    assert by_month.loc[26] == 0
    # n_obs is present for every month, so an unsupported row is visibly zero
    # rather than missing.
    assert by_month.notna().all()


def test_monthly_summary_bucket_probabilities_are_cumulative():
    X_plot, p_plot, y_plot = _plot_grid_draws()
    monthly = monthly_summary_table(X_plot, p_plot, y_plot, n_trials=800)

    bucket_columns = [f"P(Y<={k})" for k in COUNT_BUCKET_THRESHOLDS]
    buckets = monthly[bucket_columns].to_numpy()
    # Non-decreasing across thresholds within every month.
    assert np.all(np.diff(buckets, axis=1) >= -1e-12)
    # The complement column closes the set.
    np.testing.assert_allclose(
        monthly[f"P(Y<={COUNT_BUCKET_THRESHOLDS[-1]})"].to_numpy()
        + monthly[f"P(Y>{COUNT_BUCKET_THRESHOLDS[-1]})"].to_numpy(),
        1.0,
    )
    # Cumulative probability falls as the trajectory rises.
    assert monthly["P(Y<=100)"].iloc[0] > monthly["P(Y<=100)"].iloc[-1]


def test_monthly_summary_without_predictive_draws_omits_the_predictive_columns():
    """A model with no plot-grid predictive counts still gets expected counts.

    The joint sign/speech engine draws no ``y_*_plot``, so its monthly table
    carries ``p_*`` and ``Ey_*`` only. The predictive columns must be absent
    rather than zero-filled, so a reader cannot mistake a missing estimand for a
    computed one.
    """
    X_plot, p_plot, _ = _plot_grid_draws()
    monthly = monthly_summary_table(X_plot, p_plot, None, n_trials=800)

    assert {"age_months", "Ey_median", "Ey_ci_lo", "p_median"} <= set(monthly.columns)
    assert not [c for c in monthly.columns if c.startswith("Y_") or c.startswith("P(Y")]
    assert len(monthly) == 83


def test_monthly_summary_excludes_months_outside_the_grid_span():
    """A boundary month that would snap from outside the span is dropped.

    Raised in review on PR #187: with a grid starting at 8.1, month 8 sits 0.1
    months away — inside MAX_MONTH_SNAP_OFFSET — so a "nearest point within
    bound" reading would include it. It is excluded deliberately. Month 8 is
    below every observed age (the plot grid spans exactly the observed range), so
    reporting it would extrapolate, and its value would be the trajectory at 8.1
    labelled as month 8.

    Pinned so the boundary rule cannot be widened into extrapolation by a later
    change that reads the snapping bound as the coverage rule.
    """
    X_plot, p_plot, y_plot = _plot_grid_draws(lo=8.1, hi=89.9)
    monthly = monthly_summary_table(X_plot, p_plot, y_plot, n_trials=800)

    ages = monthly["age_months"].to_numpy()
    assert ages.min() == 9, "month 8 lies below the grid and must not be reported"
    assert ages.max() == 89, "month 90 lies above the grid and must not be reported"
    # Every reported month is bracketed by real grid points, so no row is an
    # extrapolation of the fitted trajectory.
    assert X_plot.min() <= monthly["grid_age_months"].min()
    assert monthly["grid_age_months"].max() <= X_plot.max()


def test_monthly_summary_keeps_boundary_months_when_the_grid_is_integral():
    """With whole-month observed ages — the real case — no month is lost."""
    X_plot, p_plot, y_plot = _plot_grid_draws(lo=8.0, hi=115.0)
    monthly = monthly_summary_table(X_plot, p_plot, y_plot, n_trials=800)

    assert monthly["age_months"].min() == 8
    assert monthly["age_months"].max() == 115
    # The endpoints land exactly on grid points, so they carry no snap error.
    offsets = monthly.set_index("age_months")["grid_offset_months"]
    assert offsets.loc[8] == 0.0
    assert offsets.loc[115] == 0.0


# --------------------------------------------------------------------------
# expand_observed_to_obs_id (issue #67)
# --------------------------------------------------------------------------


def _masked_trace(mask, observed):
    """Just the two groups the expansion reads."""
    return xr.DataTree.from_dict(
        {
            "constant_data": xr.Dataset(
                {"obs_u_mask": (("obs_id",), np.asarray(mask, dtype=int))}
            ),
            "observed_data": xr.Dataset(
                {"y_u_obs": (("row",), np.asarray(observed, dtype=float))}
            ),
        }
    )


def test_expansion_scatters_the_observed_rows_through_the_mask():
    """The five copies of this in two engines asserted nothing between them."""
    trace = _masked_trace([1, 0, 1, 1, 0], [10.0, 30.0, 40.0])

    out = expand_observed_to_obs_id(trace, "y_u_obs", "obs_u_mask")

    assert out.shape == (5,)
    assert out[0] == 10.0 and out[2] == 30.0 and out[3] == 40.0
    assert np.isnan(out[1]) and np.isnan(out[4])


def test_expansion_of_an_all_false_mask_is_all_nan():
    trace = _masked_trace([0, 0, 0], [])

    out = expand_observed_to_obs_id(trace, "y_u_obs", "obs_u_mask")

    assert out.shape == (3,) and np.isnan(out).all()


@pytest.mark.parametrize(
    ("mask", "observed"),
    [([1, 1, 0], [5.0]), ([1, 0, 0], [5.0, 6.0])],
)
def test_a_mask_that_disagrees_with_the_likelihood_rows_is_refused(mask, observed):
    """Issue #67: silently scattering a misaligned vector produces plausible figures.

    Both directions, because only one of them would ever raise on its own: too few
    observed rows leaves trailing NaN, too many raises from NumPy with a shape
    message that says nothing about masks.
    """
    trace = _masked_trace(mask, observed)

    with pytest.raises(ValueError, match=r"issue #67"):
        expand_observed_to_obs_id(trace, "y_u_obs", "obs_u_mask")
