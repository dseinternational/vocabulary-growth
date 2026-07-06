# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the VG16 within-child cross-lag helpers (issue #113)."""

import numpy as np
import pandas as pd
import pytest

from vocab_growth.models.common_bivariate_re import (
    _compute_prev_wave_lag,
    _validate_cross_lag,
)

N_TRIALS = 800


def _synthetic_df():
    """Rows cover the cases the lag source must handle, in a known row order.

    - subject 0: three waves, all with understood -> waves 2/3 have a prior source
    - subject 1: a single wave -> no prior source
    - subject 2: first wave lacks understood -> its later wave has no valid source
    - subject 3: middle wave lacks understood -> the source skips it to an earlier wave
    """
    return pd.DataFrame(
        {
            "subject_code": [0, 0, 0, 1, 2, 2, 3, 3, 3],
            "age": [12, 24, 36, 18, 12, 24, 12, 24, 36],
            "understood": [100, 200, 300, 150, np.nan, 180, 120, np.nan, 360],
        }
    )


def test_compute_prev_wave_lag_identifies_prior_understood_wave():
    prev_idx, has_lag_f, _ = _compute_prev_wave_lag(_synthetic_df(), N_TRIALS)

    # Only rows with a valid earlier understood wave carry a lag.
    np.testing.assert_array_equal(has_lag_f, [0, 1, 1, 0, 0, 0, 0, 1, 1])
    # prev_idx points at that earlier wave (row 8's source skips the NaN row 7 -> row 6).
    assert prev_idx[1] == 0
    assert prev_idx[2] == 1
    assert prev_idx[7] == 6
    assert prev_idx[8] == 6


def test_compute_prev_wave_lag_logit_of_prior_understood():
    _, has_lag_f, y_u_prev_logit = _compute_prev_wave_lag(_synthetic_df(), N_TRIALS)

    def logit(p):
        return np.log(p) - np.log(1 - p)

    # Non-lagged rows are 0; lagged rows carry the logit of the prior wave's proportion.
    np.testing.assert_allclose(
        y_u_prev_logit[[1, 2, 7, 8]],
        [logit(100 / 800), logit(200 / 800), logit(120 / 800), logit(120 / 800)],
        rtol=1e-6,
    )
    np.testing.assert_array_equal(y_u_prev_logit[[0, 3, 4, 5, 6]], 0.0)


def test_compute_prev_wave_lag_ignores_same_age_duplicates():
    df = pd.DataFrame(
        {
            "subject_code": [0, 0, 0],
            "age": [12, 12, 24],
            "understood": [100, 150, 300],
        }
    )

    prev_idx, has_lag_f, y_u_prev_logit = _compute_prev_wave_lag(df, N_TRIALS)

    np.testing.assert_array_equal(has_lag_f, [0, 0, 1])
    assert prev_idx[2] == 1
    assert y_u_prev_logit[1] == 0.0
    np.testing.assert_allclose(
        y_u_prev_logit[2],
        np.log(150 / 800) - np.log(1 - 150 / 800),
        rtol=1e-6,
    )


@pytest.mark.parametrize("baseline", ["population", "within"])
def test_validate_cross_lag_accepts_valid_config(baseline):
    _validate_cross_lag(baseline, use_subject_re_u=True)  # no raise


def test_validate_cross_lag_rejects_unknown_baseline():
    with pytest.raises(ValueError, match="lag_baseline"):
        _validate_cross_lag("bogus", use_subject_re_u=True)


@pytest.mark.parametrize("baseline", ["population", "within"])
def test_validate_cross_lag_requires_subject_re_u(baseline):
    with pytest.raises(ValueError, match="use_subject_re_u"):
        _validate_cross_lag(baseline, use_subject_re_u=False)
