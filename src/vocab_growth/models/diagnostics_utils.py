# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Helpers for model diagnostics."""

import arviz as az
import numpy as np


def capped_plot_var_names(
    trace,
    var_names: list[str],
    *,
    squared: bool = False,
) -> list[str]:
    """Return the first variables that fit within ArviZ's subplot limit."""
    max_subplots = az.rcParams.get("plot.max_subplots")
    if not isinstance(max_subplots, int):
        return list(var_names)

    max_plot_items = max_subplots
    if squared:
        max_plot_items = int(np.floor(np.sqrt(max_subplots)))

    selected_var_names: list[str] = []
    selected_plot_items = 0
    for var_name in var_names:
        plot_items = plot_variable_count(trace, var_name)
        if selected_plot_items + plot_items <= max_plot_items:
            selected_var_names.append(var_name)
            selected_plot_items += plot_items

    return selected_var_names


def plot_required_subplots(
    trace,
    var_names: list[str],
    *,
    squared: bool = False,
) -> int:
    n_plot_items = sum(plot_variable_count(trace, var_name) for var_name in var_names)
    if squared:
        return n_plot_items**2
    return n_plot_items


def plot_variable_count(trace, var_name: str) -> int:
    posterior = trace.posterior
    if hasattr(posterior, "dataset"):
        posterior = posterior.dataset

    sample_dims = posterior.attrs.get("sample_dims", az.rcParams["data.sample_dims"])
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]

    sample_dims = set(sample_dims)
    variable = posterior[var_name]
    plot_dims = [dim for dim in variable.dims if dim not in sample_dims]
    return int(np.prod([variable.sizes[dim] for dim in plot_dims]))
