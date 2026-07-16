# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for repeated-measures subject identifier validation."""

import numpy as np
import pandas as pd
import pytest

from vocab_growth.models import common_bivariate_re, common_univariate_re
from vocab_growth.models.definitions import VG11, VG13


def test_univariate_re_fails_before_subject_key_coercion(monkeypatch):
    frame = pd.DataFrame(
        {
            "age": [12.0, 13.0],
            "spoken": [5, 8],
            "study": ["study_a", "study_a"],
            "subject_id": [np.nan, np.nan],
        }
    )
    monkeypatch.setattr(
        common_univariate_re.vocab_data_utils,
        "load_data",
        lambda *args, **kwargs: frame.copy(),
    )
    monkeypatch.setattr(
        common_univariate_re.vocab_data_utils,
        "filter_studies_by_min_obs",
        lambda analysis_df, min_obs: (analysis_df, []),
    )

    with pytest.raises(ValueError, match="found 2 invalid row"):
        common_univariate_re.prepare_univariate_re_data(None, VG11)


def test_bivariate_re_fails_before_subject_key_coercion(monkeypatch):
    frame = pd.DataFrame(
        {
            "age": [12.0, 13.0],
            "understood": [20, 30],
            "spoken": [5, 8],
            "study": ["study_a", "study_a"],
            "subject_id": [np.nan, np.nan],
        }
    )
    monkeypatch.setattr(
        common_bivariate_re.vocab_data_utils,
        "load_data",
        lambda *args, **kwargs: frame.copy(),
    )
    monkeypatch.setattr(
        common_bivariate_re.vocab_data_utils,
        "filter_studies_by_min_obs",
        lambda analysis_df, min_obs: (analysis_df, []),
    )

    with pytest.raises(ValueError, match="found 2 invalid row"):
        common_bivariate_re.prepare_bivariate_re_data(None, VG13)
