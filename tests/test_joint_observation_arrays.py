# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Likelihood membership and count validation without constructing a model."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from vocab_growth.models.definitions import VG15
from vocab_growth.models.observation_arrays import prepare_joint_observations


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "age": [18, 24, 30, 36, 42, 48],
            "understood": [20, 10, np.nan, np.nan, 20, 12],
            "spoken": [5, 4, 3, np.nan, 5, 3],
            "signed": [4, 2, 1, np.nan, 4, 2],
            "understood_only": [14, np.nan, np.nan, np.nan, 14, np.nan],
            "signed_only": [1, np.nan, np.nan, np.nan, 1, np.nan],
            "spoken_only": [2, np.nan, np.nan, np.nan, 2, np.nan],
            "signed_spoken": [3, np.nan, np.nan, np.nan, 3, np.nan],
            "cell_total": [20, np.nan, np.nan, np.nan, 20, np.nan],
            "prod_signed_only": [np.nan, np.nan, np.nan, 1, np.nan, np.nan],
            "prod_spoken_only": [np.nan, np.nan, np.nan, 2, np.nan, np.nan],
            "prod_signed_spoken": [np.nan, np.nan, np.nan, 1, np.nan, np.nan],
            "prod_total": [np.nan, np.nan, np.nan, 4, np.nan, np.nan],
            "study_code": [0, 1, 1, 0, 0, 1],
            "subject_code": [0, 1, 1, 2, 0, 1],
            "holdout": [False, False, False, False, True, True],
        }
    )


def prepare(frame, definition=VG15):
    return prepare_joint_observations(
        frame,
        definition,
        n_trials=definition.n_trials,
        use_subject_codes=True,
    )


@pytest.mark.parametrize(
    "fallback,indices,dropped",
    [
        ("product_marginal", [1, 2], 0),
        ("paired_only", [1], 2),
    ],
)
def test_likelihoods_use_filtered_rows_and_retain_prediction_rows(
    frame,
    fallback,
    indices,
    dropped,
    capsys,
):
    original = frame.copy(deep=True)
    observations = prepare(frame, replace(VG15, spoken_fallback=fallback))
    assert observations.n == 6
    np.testing.assert_array_equal(observations.idx_u, [0, 1])
    np.testing.assert_array_equal(observations.idx_cells, [0])
    np.testing.assert_array_equal(observations.idx_prod, [3])
    np.testing.assert_array_equal(observations.idx_s, indices)
    np.testing.assert_array_equal(observations.idx_sign, indices)
    for suffix in ("s", "sign"):
        np.testing.assert_array_equal(
            np.flatnonzero(getattr(observations, f"has_{suffix}_likelihood")),
            indices,
        )
    np.testing.assert_array_equal(observations.cell_counts, [[14, 1, 2, 3]])
    np.testing.assert_array_equal(observations.prod_counts, [[1, 2, 1]])
    assert observations.n_fallback_dropped == dropped
    pd.testing.assert_frame_equal(frame, original)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "row,column,value,match",
    [
        (1, "understood", 10.4, "non-integral"),
        (0, "understood_only", 14.4, "non-integral"),
        (3, "prod_signed_only", -1, "negative produced-cell"),
        (3, "prod_total", 5, "do not sum"),
    ],
)
def test_invalid_counts_are_rejected_before_graph_construction(
    frame, row, column, value, match
):
    frame[column] = frame[column].astype(float)
    frame.loc[row, column] = value
    with pytest.raises(ValueError, match=match):
        prepare(frame)


def test_all_held_out_rows_leave_every_likelihood(frame):
    frame["holdout"] = True
    observations = prepare(frame)
    assert observations.n == len(frame)
    for name in ("idx_u", "idx_s", "idx_sign", "idx_cells", "idx_prod"):
        assert getattr(observations, name).size == 0
    assert observations.cell_counts.shape == (0, 4)
    assert observations.prod_counts.shape == (0, 3)
