# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for quantitative posterior-predictive calibration summaries."""

import numpy as np
import pytest

from vocab_growth.models.calibration import predictive_calibration_table


def test_predictive_calibration_reports_overall_and_age_bands():
    observed = np.array([0, 2, 4, 6])
    predictive = np.array(
        [
            [0, 0, 1, 1],
            [1, 2, 2, 3],
            [3, 4, 4, 5],
            [5, 6, 6, 7],
        ]
    )
    ages = np.array([8, 10, 14, 18])

    table = predictive_calibration_table(observed, predictive, ages)

    assert set(table["age_band_months"]) == {"all", "[0, 12)", "[12, 24)"}
    assert set(table["interval_probability"]) == {0.5, 0.8, 0.9}
    overall = table[table["age_band_months"] == "all"]
    assert (overall["empirical_coverage"] == 1.0).all()
    assert (overall["n_observations"] == 4).all()


def test_predictive_calibration_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="row-aligned"):
        predictive_calibration_table(
            observed=np.array([1]),
            predictive=np.ones((2, 4)),
            ages=np.array([10]),
        )
