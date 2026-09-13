# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A form restriction changes lag membership, with explicit source provenance."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from vocab_growth import sign_lag_forms
from vocab_growth.models.cross_lag import (
    prev_wave_sign_share_lag,
    prev_wave_sign_share_lag_for_frame,
)
from vocab_growth.models.definitions import VG25
from vocab_growth.sensitivity.registry import build_variant


def test_same_form_drops_the_lag_without_choosing_an_older_wave():
    previous, has_lag, logits = prev_wave_sign_share_lag(
        [0, 0, 0],
        [12, 24, 36],
        [5, 10, 12],
        [20, 30, 40],
        same_form_only=True,
        form_ceiling=[810, 416, 810],
    )
    np.testing.assert_array_equal(has_lag, [0, 0, 0])
    np.testing.assert_array_equal(previous, [0, 0, 0])
    np.testing.assert_array_equal(logits, [0, 0, 0])


@pytest.mark.parametrize("sizes", [[810, np.nan], [np.nan, 810]])
def test_unknown_form_cannot_certify_a_matching_lag(sizes):
    _, has_lag, _ = prev_wave_sign_share_lag(
        [0, 0],
        [12, 24],
        [5, 10],
        [20, 30],
        same_form_only=True,
        form_ceiling=sizes,
    )
    np.testing.assert_array_equal(has_lag, [0, 0])


@pytest.mark.parametrize("order", [[0, 1, 2], [1, 0, 2]])
def test_source_form_breaks_a_count_tie_independently_of_row_order(order):
    ages = np.array([12, 12, 24])[order]
    sizes = np.array([416, 810, 810])[order]
    previous, has_lag, _ = prev_wave_sign_share_lag(
        [0, 0, 0],
        ages,
        [5, 5, 10],
        [20, 20, 30],
        same_form_only=True,
        form_ceiling=sizes,
    )
    assert has_lag[2] == 1
    assert sizes[previous[2]] == 810


def test_frame_entry_point_requires_metadata_for_the_requested_restriction():
    definition = replace(VG25, sign_lag_same_form_only=True)
    with pytest.raises(ValueError, match="survey_vocab_max"):
        prev_wave_sign_share_lag_for_frame(pd.DataFrame(), definition)
    (variant,) = build_variant("vg25", "sign-lag-same-form")
    assert variant.sign_lag_same_form_only is True
    assert VG25.sign_lag_same_form_only is False


def test_uk02_counts_disambiguate_two_forms_at_the_same_child_age(monkeypatch):
    four = pd.DataFrame(
        {
            "subject_id": ["child"],
            "age": [24],
            "cell_total": [20],
            "comprehension": [21],
            "spoken": [5],
            "signed": [4],
            "understood_only": [14],
            "signed_only": [1],
            "spoken_only": [2],
            "signed_spoken": [3],
            "form": ["DSE"],
        }
    )
    marginal = pd.DataFrame(
        {
            "subject_id": ["child"],
            "age": [24],
            "comprehension": [20],
            "spoken": [5],
            "signed": [np.nan],
            "form": ["Oxford_CDI"],
        }
    )
    prepared_four = four.assign(understood=20, spoken=np.nan, signed=np.nan)
    prepared_marginal = marginal.assign(understood=20)
    frame = pd.concat([prepared_marginal, prepared_four], ignore_index=True).assign(
        study="uk_02"
    )
    metadata = pd.DataFrame(
        {
            "study": ["uk_02", "uk_02"],
            "subject_id": ["child", "child"],
            "age": [24, 24],
            "survey_vocab_max": [810, 416],
        }
    )
    monkeypatch.setattr(
        sign_lag_forms.data_utils, "load_data", lambda **kwargs: metadata.copy()
    )
    monkeypatch.setattr(
        sign_lag_forms, "load_uk02_four_cell", lambda: (four.copy(), marginal.copy())
    )
    original = frame.copy(deep=True)
    np.testing.assert_array_equal(
        sign_lag_forms.joint_inventory_sizes(frame, VG25), [416, 810]
    )
    pd.testing.assert_frame_equal(frame, original)
