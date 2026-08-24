# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Helpers for model diagnostics."""

import dse_research_utils.plot.diagnostics_mcmc as shared_plot_diagnostics

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


# The subplot-budget helpers now live in the shared library
# (dse_research_utils.plot.diagnostics_mcmc, v0.12.0); re-exported here so the
# existing import paths in models/common.py and the tests keep working.
capped_plot_var_names = shared_plot_diagnostics.capped_plot_var_names
plot_required_subplots = shared_plot_diagnostics.plot_required_subplots
plot_variable_count = shared_plot_diagnostics.plot_variable_count
