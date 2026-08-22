# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Helpers for model diagnostics."""

import arviz as az
import numpy as np

from vocab_growth.fit_artifacts import ACCEPTED_EXCEPTION_KEY, read_convergence_caveats


def render_convergence_caveats(directory: str = ".") -> None:
    """Print this fit's convergence caveats for a report cell with ``#| output: asis``.

    Every model report used to carry its own copy of this logic, and every copy
    read only two of the gate's four checks — divergences and the energy BFMI.
    A fit published under a recorded R-hat exception therefore printed the
    sentence "this fit cleared the hard convergence tier", which is the opposite
    of what the exception means, while the exception itself went unmentioned.
    VG11 is exactly that case, so the defect was live rather than theoretical.

    Delegating to :func:`vocab_growth.fit_artifacts.read_convergence_caveats`
    puts the reader's page on the same implementation as the fit-time gate,
    ``validate_fit_output`` and Appendix B, which is the invariant
    :func:`vocab_growth.fit_artifacts.convergence_caveats` was written to hold:
    a disclosure that can drift between its producer and its consumer is a
    disclosure that will.

    Silent when the fit is clean, so a clean report gains nothing from the call
    and a caveated one cannot lose it.
    """
    import json
    import os

    caveats = read_convergence_caveats(directory)
    if not caveats:
        return

    # The hard tier is fail-closed, so a fit that reaches a report either cleared
    # it or holds a recorded exception. Which of the two decides the framing: the
    # first is "reportable with noted sampling caveats", the second is "did not
    # clear the gate and is published anyway, on the record".
    accepted = None
    summary_path = os.path.join(directory, "diagnostics_summary.json")
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, encoding="utf-8") as handle:
                accepted = json.load(handle).get(ACCEPTED_EXCEPTION_KEY)
        except (OSError, ValueError):
            accepted = None

    if accepted:
        title = "Published under a recorded convergence exception"
        opening = (
            "This fit **did not clear** the hard convergence tier (R-hat and "
            "effective sample size). It is published under an exception recorded "
            "against the model, reproduced below with any other sampling caveats:"
        )
    else:
        title = "Soft-tier convergence caveats"
        opening = (
            "This fit cleared the hard convergence tier (R-hat and effective "
            "sample size) but not the soft tier:"
        )

    print(f'::: {{.callout-warning title="{title}"}}')
    print()
    print(opening)
    print()
    for caveat in caveats:
        print(f"- {caveat}")
    print()
    if accepted and accepted.get("reason"):
        print(f"**Why the exception was accepted:** {accepted['reason']}")
        print()
    print(
        "The fit remains reportable and is published carrying this mark; see "
        "Appendix B of the technical report."
    )
    print()
    print(":::")


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
