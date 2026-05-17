# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import arviz as az
import numpy as np
import xarray as xr

from vocab_growth.models.diagnostics_utils import (
    pair_plot_rc_params,
    pair_plot_required_subplots,
)


def test_pair_plot_required_subplots_counts_non_sample_dimensions():
    trace = _trace_with_scalar_and_vector_parameters()

    assert pair_plot_required_subplots(trace, ["alpha", "beta"]) == 9


def test_pair_plot_rc_params_uses_expanded_pair_plot_count():
    trace = _trace_with_scalar_and_vector_parameters()

    with az.rc_context({"plot.max_subplots": 8}):
        assert pair_plot_rc_params(trace, ["alpha", "beta"]) == {
            "plot.max_subplots": 9
        }


def test_pair_plot_rc_params_keeps_existing_limit_when_sufficient():
    trace = _trace_with_scalar_and_vector_parameters()

    with az.rc_context({"plot.max_subplots": 9}):
        assert pair_plot_rc_params(trace, ["alpha", "beta"]) == {}


def _trace_with_scalar_and_vector_parameters():
    posterior = xr.Dataset(
        data_vars={
            "alpha": (("chain", "draw"), np.ones((1, 2))),
            "beta": (("chain", "draw", "coef"), np.ones((1, 2, 2))),
        },
        coords={
            "chain": [0],
            "draw": [0, 1],
            "coef": ["intercept", "slope"],
        },
        attrs={"sample_dims": ["chain", "draw"]},
    )
    return xr.DataTree.from_dict({"posterior": posterior})
