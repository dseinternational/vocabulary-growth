# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for ``loo_dropping_degenerate`` in ``models.common``.

The nested joint likelihood models a paired outcome conditionally on the
observed *understood* count, so an ``understood == 0`` row gives that outcome a
structurally constant (``n = 0``) pointwise log-likelihood across every draw.
``arviz_stats`` cannot run such a degenerate point through PSIS-LOO — ``az.loo``
raises ``ValueError: All tail values are the same``. The helper drops those
observations before calling ``az.loo``.

These tests pin that the helper (a) reproduces — and then prevents — the bare
``az.loo`` crash, (b) counts the dropped observations, (c) is a no-op when no
observation is degenerate, and (d) leaves the caller's ``InferenceData``
unmutated so repeated per-outcome calls stay independent. A synthetic
``InferenceData`` reproduces the degeneracy exactly, so no fitted joint trace
is required.
"""

import arviz as az
import numpy as np
import pytest
from arviz import ELPDData

from vocab_growth.models.common import loo_dropping_degenerate

_NOBS = 25
_VAR = "y_s_obs"


def _synthetic_idata(constant_obs, *, var_name=_VAR, nobs=_NOBS, seed=1):
    """Build an ``InferenceData`` whose ``constant_obs`` log-lik rows are
    exactly constant across draws (the ``n = 0`` degeneracy)."""
    rng = np.random.default_rng(seed)
    chains, draws = 4, 800
    ll = rng.normal(-1.5, 0.5, size=(chains, draws, nobs))
    for i in constant_obs:
        ll[:, :, i] = 0.0
    data = {
        "posterior": {"theta": rng.normal(size=(chains, draws))},
        "log_likelihood": {var_name: ll},
    }
    return az.from_dict(data, dims={var_name: ["obs"]})


def test_bare_loo_crashes_on_constant_observation():
    """Regression guard: without the drop, ``az.loo`` fails on ``n = 0`` rows."""
    idata = _synthetic_idata([5, 12])
    with pytest.raises(ValueError, match="tail values are the same"):
        az.loo(idata, var_name=_VAR)


def test_drops_degenerate_and_computes_loo():
    idata = _synthetic_idata([5, 12])
    loo, n_dropped = loo_dropping_degenerate(idata, var_name=_VAR)
    assert n_dropped == 2
    assert isinstance(loo, ELPDData)


def test_no_degenerate_observations_is_a_no_op():
    idata = _synthetic_idata([])
    loo, n_dropped = loo_dropping_degenerate(idata, var_name=_VAR)
    assert n_dropped == 0
    assert isinstance(loo, ELPDData)


def test_original_idata_not_mutated():
    idata = _synthetic_idata([5, 12])
    before = int(idata.log_likelihood.sizes["obs"])
    loo_dropping_degenerate(idata, var_name=_VAR)
    after = int(idata.log_likelihood.sizes["obs"])
    assert before == after == _NOBS


def test_resolves_single_var_when_var_name_is_none():
    """The ``loo_var_names is None`` engine path passes no ``var_name``; the
    helper must resolve the sole log-likelihood array and still drop/compute."""
    idata = _synthetic_idata([5, 12])
    loo, n_dropped = loo_dropping_degenerate(idata)
    assert n_dropped == 2
    assert isinstance(loo, ELPDData)
