# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd

from vocab_growth import comparison


# ---- registry resolution ----
def test_registry_resolution():
    assert comparison.model_dir("vg11").replace("\\", "/").endswith(
        "output/models/VG11-age-spoken-td-re"
    )
    assert comparison.n_trials("vg11") == 800
    assert comparison.n_trials("vg10") == 800
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
    assert {"median", "hdi50_lo", "hdi90_hi"}.issubset(df.columns)


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


def test_invert_curve_linear():
    df = pd.DataFrame(
        {
            "age_months": [0.0, 10.0, 20.0, 30.0],
            "Y_hdi_lo": [0.0, 50.0, 100.0, 150.0],
            "Y_median": [0.0, 100.0, 200.0, 300.0],
            "Y_hdi_hi": [0.0, 200.0, 400.0, 600.0],
        }
    )
    inv = comparison.invert_curve(df, targets=[100])
    row = inv.iloc[0]
    assert row["target_words"] == 100
    assert np.isclose(row["age_typical_child_p50"], 10.0)  # median hits 100 at age 10
    assert np.isclose(row["age_fast_child_p95"], 5.0)      # fast hits 100 at age 5
    assert np.isclose(row["age_slow_child_p5"], 20.0)      # slow hits 100 at age 20
