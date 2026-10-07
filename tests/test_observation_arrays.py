# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check bivariate observation arrays without constructing a model.

Likelihood masks must reflect branch selection. Validate counts before integer
conversion, and retain held-out rows in observation space while excluding their
likelihood contributions.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from vocab_growth.models.definitions import VG10
from vocab_growth.models.likelihood_utils import (
    SPOKEN_FALLBACK_PAIRED_ONLY,
    SPOKEN_FALLBACK_PRODUCT,
)
from vocab_growth.models.observation_arrays import prepare_bivariate_observations


def _frame(n=12, *, understood=None, spoken=None, holdout=None):
    """A tiny frame with both outcomes, two studies and repeat-measured children."""
    ages = np.linspace(18.0, 72.0, n)
    frame = pd.DataFrame(
        {
            "age": ages,
            "understood": np.round(ages * 5.0) if understood is None else understood,
            "spoken": np.round(ages * 2.0) if spoken is None else spoken,
            "study_code": [i % 2 for i in range(n)],
            "subject_code": np.repeat(np.arange(n // 2), 2),
        }
    )
    if holdout is not None:
        frame["holdout"] = holdout
    return frame


def _prepare(frame, definition=VG10, **kwargs):
    return prepare_bivariate_observations(
        frame,
        definition,
        n_trials=definition.n_trials,
        use_subject_codes=kwargs.pop("use_subject_codes", True),
        **kwargs,
    )


def test_the_shapes_and_counts_line_up():
    observations = _prepare(_frame())
    assert observations.n == 12
    assert observations.X_obs.shape == (12, 1)
    assert observations.n_u == 12
    assert observations.n_s == 12
    assert observations.n_studies == 2
    assert observations.n_subjects == 6


def test_a_missing_outcome_leaves_its_row_out_of_that_likelihood_only():
    spoken = np.round(np.linspace(18.0, 72.0, 12) * 2.0)
    spoken[3] = np.nan
    observations = _prepare(_frame(spoken=spoken))
    assert observations.n_u == 12
    assert observations.n_s == 11
    assert 3 not in observations.idx_s
    assert 3 in observations.idx_u
    # The row is still in observation space, so its deterministics exist.
    assert observations.n == 12


# --- the spoken likelihood mask (#266 finding 3) --------------------------------


def test_the_spoken_mask_marks_the_likelihood_rows_not_every_recorded_row():
    """Exclude marginal fallback rows from the paired-only likelihood mask."""
    understood = np.round(np.linspace(18.0, 72.0, 12) * 5.0)
    # Four rows record speech but no comprehension to condition it on.
    understood[[1, 4, 7, 10]] = np.nan
    frame = _frame(understood=understood)

    product = _prepare(frame, dataclasses.replace(VG10, spoken_fallback=SPOKEN_FALLBACK_PRODUCT))
    paired = _prepare(frame, dataclasses.replace(VG10, spoken_fallback=SPOKEN_FALLBACK_PAIRED_ONLY))

    assert product.n_fallback_dropped == 0
    assert paired.n_fallback_dropped == 4
    # The mask follows the likelihood, not the recorded data.
    assert int(product.has_s_likelihood.sum()) == product.n_s == 12
    assert int(paired.has_s_likelihood.sum()) == paired.n_s == 8
    assert not paired.has_s_likelihood[[1, 4, 7, 10]].any()
    # And it is exactly the rows the likelihood indexes.
    np.testing.assert_array_equal(
        np.flatnonzero(paired.has_s_likelihood), paired.idx_s
    )


def test_the_two_masks_have_the_same_length_as_the_observation_dimension():
    """They are stored with ``dims=("obs_id",)``, so a short mask misaligns silently."""
    observations = _prepare(_frame())
    assert observations.has_u_likelihood.shape == (observations.n,)
    assert observations.has_s_likelihood.shape == (observations.n,)


# --- validation before the integer cast (#236, #240) ----------------------------


@pytest.mark.parametrize("bad", [810.9, -0.1, 1_000.0, float("inf")])
def test_a_count_the_integer_cast_would_hide_is_refused(bad):
    """NumPy truncates silently: 810.9 and -0.1 land inside a post-cast check."""
    understood = np.round(np.linspace(18.0, 72.0, 12) * 5.0)
    understood[5] = bad
    with pytest.raises(ValueError):
        _prepare(_frame(understood=understood))


def test_a_missing_count_is_not_an_invalid_one():
    """Omit missing counts from their likelihood instead of rejecting the row."""
    understood = np.round(np.linspace(18.0, 72.0, 12) * 5.0)
    understood[5] = np.nan
    observations = _prepare(_frame(understood=understood))
    assert observations.n_u == 11
    assert 5 not in observations.idx_u
    assert not observations.has_u_likelihood[5]


def test_a_valid_count_at_the_boundary_is_accepted():
    understood = np.full(12, float(VG10.n_trials))
    spoken = np.full(12, float(VG10.n_trials))
    observations = _prepare(_frame(understood=understood, spoken=spoken))
    assert observations.y_u_observed.max() == VG10.n_trials


# --- the held-out mask ----------------------------------------------------------


def test_held_out_rows_stay_in_observation_space_and_leave_the_likelihood():
    holdout = np.zeros(12, dtype=bool)
    holdout[[2, 6]] = True
    observations = _prepare(_frame(holdout=holdout))

    assert observations.n == 12  # still in obs space
    assert observations.n_u == 10
    assert observations.n_s == 10
    for row in (2, 6):
        assert row not in observations.idx_u
        assert row not in observations.idx_s
        assert not observations.has_u_likelihood[row]
        assert not observations.has_s_likelihood[row]


def test_a_frame_with_no_holdout_column_holds_nothing_out():
    observations = _prepare(_frame())
    assert not observations.holdout.any()
    assert observations.n_u == observations.n


# --- subject codes come from the plan, not from a second derivation -------------


def test_subject_codes_are_absent_when_no_outcome_carries_a_child_effect():
    """Follow the subject-effect plan when constructing child codes."""
    observations = _prepare(_frame(), use_subject_codes=False)
    assert observations.subject_codes is None
    assert observations.n_subjects == 0


def test_the_record_is_immutable():
    observations = _prepare(_frame())
    with pytest.raises(dataclasses.FrozenInstanceError):
        observations.n = 0
