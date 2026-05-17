# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Helpers for model diagnostics."""

import arviz as az
import numpy as np


def pair_plot_rc_params(trace, var_names: list[str]) -> dict[str, int]:
    """Return temporary rcParams needed for plotting all pair-plot panels."""
    max_subplots = az.rcParams.get("plot.max_subplots")
    if not isinstance(max_subplots, int):
        return {}

    required_subplots = pair_plot_required_subplots(trace, var_names)
    if required_subplots <= max_subplots:
        return {}

    return {"plot.max_subplots": required_subplots}


def pair_plot_required_subplots(trace, var_names: list[str]) -> int:
    posterior = trace.posterior
    if hasattr(posterior, "dataset"):
        posterior = posterior.dataset
    posterior = posterior[var_names]
    sample_dims = posterior.attrs.get("sample_dims", az.rcParams["data.sample_dims"])
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]

    sample_dims = set(sample_dims)
    n_pairs = 0
    for var_name in var_names:
        variable = posterior[var_name]
        plot_dims = [dim for dim in variable.dims if dim not in sample_dims]
        n_pairs += int(np.prod([variable.sizes[dim] for dim in plot_dims]))

    return n_pairs**2
