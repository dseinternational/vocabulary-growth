# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for the per-administration joint log-likelihood in
``scripts/loo_compare.py`` (#236).

The coherent pointwise unit of the joint bivariate likelihood is the
administration: log p(U_i) + log p(S_i | U_i) for a paired row, and the single
observed factor for an understood-only or spoken-only row. The pre-#236
implementation concatenated the two factor vectors instead, giving every paired
administration two PSIS weights and leaking the conditioned-on understood count
whenever its own factor was held out.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "loo_compare.py"
_SPEC = importlib.util.spec_from_file_location("loo_compare_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_attach_joint_log_likelihood = _MODULE._attach_joint_log_likelihood


def _make_trace(u_mask, s_mask, u_ll, s_ll, *, constant_data=True):
    """Build a minimal trace: log-likelihood factor vectors plus the stored
    masks that map them back to administration rows.

    ``u_ll`` / ``s_ll`` are (chain, draw, factor) arrays whose factor lengths
    must match the mask counts (unless a test deliberately breaks that).
    """
    log_likelihood = xr.Dataset(
        {
            "y_u_obs": (("chain", "draw", "obs_u_id"), np.asarray(u_ll, dtype=float)),
            "y_s_obs": (("chain", "draw", "obs_s_id"), np.asarray(s_ll, dtype=float)),
        }
    )
    groups = {"log_likelihood": log_likelihood}
    if constant_data:
        groups["constant_data"] = xr.Dataset(
            {
                "obs_u_mask": ("obs_id", np.asarray(u_mask, dtype=int)),
                "obs_s_mask": ("obs_id", np.asarray(s_mask, dtype=int)),
            }
        )
    return xr.DataTree.from_dict(groups)


def test_paired_factors_are_summed_per_administration():
    # Four administrations: paired, understood-only, spoken-only, paired.
    u_mask = [1, 1, 0, 1]
    s_mask = [1, 0, 1, 1]
    u_ll = [[[-1.0, -2.0, -3.0]]]  # factors for rows 0, 1, 3
    s_ll = [[[-10.0, -20.0, -30.0]]]  # factors for rows 0, 2, 3

    idata = _make_trace(u_mask, s_mask, u_ll, s_ll)
    _attach_joint_log_likelihood(idata)

    joint = idata.log_likelihood["y_joint"]
    # One entry per administration with any observation, not one per factor.
    assert joint.sizes["obs_joint"] == 4
    np.testing.assert_array_equal(joint["obs_joint"].values, [0, 1, 2, 3])
    np.testing.assert_allclose(
        joint.values[0, 0], [-11.0, -2.0, -20.0, -33.0]
    )


def test_rows_without_any_observation_are_dropped():
    u_mask = [1, 0, 0]
    s_mask = [0, 0, 1]
    idata = _make_trace(u_mask, s_mask, [[[-1.0]]], [[[-2.0]]])
    _attach_joint_log_likelihood(idata)

    joint = idata.log_likelihood["y_joint"]
    assert joint.sizes["obs_joint"] == 2
    np.testing.assert_array_equal(joint["obs_joint"].values, [0, 2])
    np.testing.assert_allclose(joint.values[0, 0], [-1.0, -2.0])


def test_joint_preserves_chain_and_draw_structure():
    u_mask = [1, 1]
    s_mask = [1, 0]
    u_ll = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2) * -1.0
    s_ll = np.arange(2 * 3 * 1, dtype=float).reshape(2, 3, 1) * -10.0

    idata = _make_trace(u_mask, s_mask, u_ll, s_ll)
    _attach_joint_log_likelihood(idata)

    joint = idata.log_likelihood["y_joint"]
    assert joint.sizes == {"chain": 2, "draw": 3, "obs_joint": 2}
    np.testing.assert_allclose(
        joint.values[:, :, 0], u_ll[:, :, 0] + s_ll[:, :, 0]
    )
    np.testing.assert_allclose(joint.values[:, :, 1], u_ll[:, :, 1])


def test_mask_factor_mismatch_raises():
    # Three understood factors against a mask claiming two observed rows.
    idata = _make_trace([1, 1, 0], [1, 0, 0], [[[-1.0, -2.0, -3.0]]], [[[-4.0]]])
    with pytest.raises(ValueError, match="do not match the likelihood rows"):
        _attach_joint_log_likelihood(idata)


def test_missing_constant_data_raises():
    idata = _make_trace([1], [1], [[[-1.0]]], [[[-2.0]]], constant_data=False)
    with pytest.raises(ValueError, match="constant_data"):
        _attach_joint_log_likelihood(idata)


def test_existing_y_joint_is_left_alone():
    idata = _make_trace([1, 1], [1, 0], [[[-1.0, -2.0]]], [[[-3.0]]])
    _attach_joint_log_likelihood(idata)
    before = idata.log_likelihood["y_joint"].values.copy()
    _attach_joint_log_likelihood(idata)
    np.testing.assert_array_equal(idata.log_likelihood["y_joint"].values, before)
