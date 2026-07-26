# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np

from vocab_growth import comparison


# ---- registry resolution ----
def test_registry_resolution():
    assert comparison.model_dir("vg11").replace("\\", "/").endswith(
        "output/models/VG11-age-spoken-td-re"
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
