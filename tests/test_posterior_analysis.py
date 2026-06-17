# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np

from vocab_growth.posterior_analysis import posterior_summary_table

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
