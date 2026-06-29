# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import types

import numpy as np
import xarray as xr

from vocab_growth.posterior_analysis import (
    add_probability_estimand_columns,
    extract_posterior,
    extract_posterior_predictive,
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
    "age_months", "p_median", "p_hdi_lo", "p_hdi_hi",
    "Ey_median", "Ey_hdi_lo", "Ey_hdi_hi",
    "Y_median", "Y_hdi_lo", "Y_hdi_hi",
    "P(Y=0)", "P(Y<=5)", "P(Y<=400)", "P(Y>400)",
}


def _summary():
    # Row order is (age 24, age 12); the table should come back age-sorted.
    X_query = np.array([24, 12])
    p_query = np.vstack([np.full(N_SAMPLES, 0.5), np.full(N_SAMPLES, 0.25)])
    y_query = np.vstack([np.full(N_SAMPLES, 400), np.zeros(N_SAMPLES, dtype=int)])
    return posterior_summary_table(X_query, p_query, y_query, n_trials=800, hdi_prob=0.90)


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
        hdi_prob=0.90,
    )

    row = out[out["age_months"] == 12.0].iloc[0]
    assert np.isclose(row["p_population_median"], 0.25)
    assert np.isclose(row["Ey_population_median"], 200.0)
    assert np.isclose(row["p_subject_marginal_median"], 0.20)
    assert np.isclose(row["Ey_subject_marginal_median"], 160.0)
