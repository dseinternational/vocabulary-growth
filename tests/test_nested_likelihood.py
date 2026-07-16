# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for nested outcome likelihood classification."""

import numpy as np
import pandas as pd
import pytest

from vocab_growth.models.likelihood_utils import nested_outcome_spec


def test_nested_outcome_spec_uses_parent_count_when_valid():
    df = pd.DataFrame(
        {
            "understood": [100, np.nan, 40, 30, 810],
            "spoken": [25, 10, 50, np.nan, 810],
        }
    )

    spec = nested_outcome_spec(
        df,
        parent_col="understood",
        outcome_col="spoken",
        n_trials=810,
    )

    np.testing.assert_array_equal(spec.indices, [0, 1, 2, 4])
    np.testing.assert_array_equal(spec.observed, [25, 10, 50, 810])
    np.testing.assert_array_equal(spec.trials, [100, 810, 810, 810])
    np.testing.assert_array_equal(spec.is_conditional, [True, False, False, True])
    assert spec.n_conditional == 2
    assert spec.n_marginal == 2
    assert spec.n_parent_violations == 1


def test_nested_outcome_spec_respects_eligibility_mask():
    df = pd.DataFrame({"understood": [100, 200], "spoken": [25, 50]})

    spec = nested_outcome_spec(
        df,
        parent_col="understood",
        outcome_col="spoken",
        n_trials=810,
        eligible_mask=np.array([False, True]),
    )

    np.testing.assert_array_equal(spec.indices, [1])
    np.testing.assert_array_equal(spec.trials, [200])


@pytest.mark.parametrize("spoken", [-1, 811, 1.5])
def test_nested_outcome_spec_rejects_invalid_child_counts(spoken):
    df = pd.DataFrame({"understood": [100], "spoken": [spoken]})

    with pytest.raises(ValueError):
        nested_outcome_spec(
            df,
            parent_col="understood",
            outcome_col="spoken",
            n_trials=810,
        )
