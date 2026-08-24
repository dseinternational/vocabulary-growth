# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests: understood counts are validated *before* the int cast.

Both bivariate engines cast the understood column to ``int`` for the
Beta-Binomial likelihood. NumPy's cast truncates toward zero silently, so a
fractional value such as ``810.9`` or ``-0.1`` would become ``810`` or ``0``
and then pass the post-cast bounds checks (#236). These tests pin that the
guard fires on the raw values, on both engines, before any model is built.
Spoken counts were already validated pre-cast by ``nested_outcome_spec``.

Both engines share ``require_valid_counts`` (finite, integral, in range), which
arrived on the RE engine with the VG11-VG13 review (#240) and is applied here to
the non-RE engine and to both engines' preparation paths as well.
"""

import os

import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd
import pytest

from vocab_growth.models.common import ModelFitContext
from vocab_growth.models.common_bivariate import build_model, configure_bivariate_priors
from vocab_growth.models.common_bivariate_re import build_model_re
from vocab_growth.models.definitions import VG05, VG07


def _context_with_frame(tmp_path, monkeypatch, definition, understood_values):
    n = len(understood_values)
    ages = np.linspace(12.0, 60.0, n)
    analysis_df = pd.DataFrame(
        {
            "age": ages,
            "understood": understood_values,
            "spoken": np.round(ages * 2.0),
            "study": ["study_a"] * (n // 2) + ["study_b"] * (n - n // 2),
            "study_code": [0] * (n // 2) + [1] * (n - n // 2),
            "subject_code": np.arange(n),
        }
    )
    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name="TEST_COUNT_VALIDATION",
            config_name="test",
            output_root_dir=str(tmp_path),
            ci_prob=0.90,
            interval_kind="hdi",
        ),
        sampling=sampling.get_sampling_configuration("test"),
    )
    bmd = model_data.BinomialModelData(
        X_obs=ages.reshape(-1, 1),
        y_obs=np.zeros(n, dtype=int),
        n_trials=definition.n_trials,
    )
    context.set_model_data(bmd, analysis_df)
    # Populate the model configuration the build reads first.
    os.makedirs(context.reporting.output_dir, exist_ok=True)
    configure_bivariate_priors(context, definition)
    return context


def test_build_model_re_rejects_fractional_understood(tmp_path, monkeypatch):
    understood = np.round(np.linspace(50.0, 400.0, 8))
    understood[3] = 100.5  # would silently truncate to 100 pre-#236
    context = _context_with_frame(tmp_path, monkeypatch, VG07, understood)
    with pytest.raises(ValueError, match="understood contains 1 non-integral"):
        build_model_re(context, VG07)


def test_build_model_re_rejects_just_outside_bound_fraction(tmp_path, monkeypatch):
    understood = np.round(np.linspace(50.0, 400.0, 8))
    understood[7] = 810.9  # would truncate to exactly n_trials and pass bounds
    context = _context_with_frame(tmp_path, monkeypatch, VG07, understood)
    with pytest.raises(ValueError, match="understood contains 1 non-integral"):
        build_model_re(context, VG07)


def test_build_model_rejects_fractional_understood(tmp_path, monkeypatch):
    understood = np.round(np.linspace(50.0, 400.0, 8))
    understood[2] = -0.1  # would truncate to 0 and pass the >= 0 bounds check
    context = _context_with_frame(tmp_path, monkeypatch, VG05, understood)
    with pytest.raises(ValueError, match="understood contains 1 non-integral"):
        build_model(context, VG05)


def test_both_engines_reject_out_of_range_understood(tmp_path, monkeypatch):
    """An integral but out-of-range count fails the shared pre-cast contract.

    This one was already caught, but only by the post-cast bounds check further
    down each build; `require_valid_counts` now rejects it before the cast, so
    both engines report it the same way.
    """
    for definition, build in ((VG07, build_model_re), (VG05, build_model)):
        understood = np.round(np.linspace(50.0, 400.0, 8))
        understood[5] = 811.0  # one past the 810-item reference inventory
        context = _context_with_frame(tmp_path, monkeypatch, definition, understood)
        with pytest.raises(ValueError, match="between 0 and n_trials"):
            build(context, definition)
