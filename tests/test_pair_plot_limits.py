# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import arviz as az
import numpy as np
import xarray as xr

from vocab_growth.models.diagnostics_utils import (
    capped_plot_var_names,
    plot_required_subplots,
    plot_variable_count,
)


def test_plot_required_subplots_counts_non_sample_dimensions():
    trace = _trace_with_scalar_and_vector_parameters()

    assert plot_required_subplots(trace, ["alpha", "beta"], squared=True) == 9


def test_capped_plot_var_names_limits_pair_plot_grid():
    trace = _trace_with_scalar_and_vector_parameters()

    with az.rc_context({"plot.max_subplots": 8}):
        var_names = capped_plot_var_names(
            trace,
            ["alpha", "beta"],
            squared=True,
        )

    assert var_names == ["alpha"]
    assert plot_required_subplots(trace, var_names, squared=True) <= 8


def test_capped_plot_var_names_keeps_pair_plot_vars_when_limit_is_sufficient():
    trace = _trace_with_scalar_and_vector_parameters()

    with az.rc_context({"plot.max_subplots": 9}):
        assert capped_plot_var_names(
            trace,
            ["alpha", "beta"],
            squared=True,
        ) == ["alpha", "beta"]


def test_capped_plot_var_names_skips_large_observed_diagnostic():
    trace = _trace_with_large_observed_diagnostic()

    with az.rc_context({"plot.max_subplots": 40}):
        var_names = capped_plot_var_names(trace, ["alpha", "kappa_obs"])

    assert var_names == ["alpha"]
    assert plot_variable_count(trace, "kappa_obs") == 100


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


def _trace_with_large_observed_diagnostic():
    posterior = xr.Dataset(
        data_vars={
            "alpha": (("chain", "draw"), np.ones((1, 2))),
            "kappa_obs": (("chain", "draw", "obs"), np.ones((1, 2, 100))),
        },
        coords={
            "chain": [0],
            "draw": [0, 1],
            "obs": range(100),
        },
        attrs={"sample_dims": ["chain", "draw"]},
    )
    return xr.DataTree.from_dict({"posterior": posterior})
