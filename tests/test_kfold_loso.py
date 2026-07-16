# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for nested held-out scoring in ``kfold_loso``."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import betabinom

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "kfold_loso.py"
_SPEC = importlib.util.spec_from_file_location("kfold_loso_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

N_TRIALS = _MODULE.N_TRIALS
holdout_subject_elpds = _MODULE.holdout_subject_elpds


def test_non_integer_parent_uses_marginal_spoken_likelihood():
    analysis_df = pd.DataFrame(
        {
            "subject_code": [0],
            "understood": [100.5],
            "spoken": [25],
        },
        index=[42],
    )
    posterior = xr.Dataset(
        {
            "p_u_obs": (("chain", "draw", "obs_id"), [[[0.4]]]),
            "p_s_obs": (("chain", "draw", "obs_id"), [[[0.2]]]),
            "q_obs": (("chain", "draw", "obs_id"), [[[0.9]]]),
            "kappa_u_obs": (("chain", "draw", "obs_id"), [[[20.0]]]),
            "kappa_s_obs": (("chain", "draw", "obs_id"), [[[15.0]]]),
        }
    )
    trace = xr.DataTree.from_dict({"posterior": posterior})

    actual = holdout_subject_elpds(analysis_df, trace, np.array([0]))[0]
    expected = betabinom.logpmf(100, N_TRIALS, 0.4 * 20.0, 0.6 * 20.0)
    expected += betabinom.logpmf(25, N_TRIALS, 0.2 * 15.0, 0.8 * 15.0)

    assert actual == pytest.approx(expected)
