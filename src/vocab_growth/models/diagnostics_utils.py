# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Helpers for model diagnostics."""

import dse_research_utils.plot.diagnostics_mcmc as shared_plot_diagnostics

from vocab_growth.fit_artifacts import (
    ACCEPTED_EXCEPTION_KEY,
    HardConvergenceStatus,
    hard_tier_status,
    read_convergence_caveats,
)


def pair_plot_priority(definition) -> tuple[str, ...]:
    """The variables the pair plot must show for ``definition``, most important first.

    ArviZ caps a pair plot at ``floor(sqrt(plot.max_subplots))`` variables, so a
    grid built in model order fits about six -- and model order is the build
    order, which puts the mean-function and GP parameters first. Every parameter
    a child-effect model was *added for* therefore fell off the end: VG19's
    slope block, VG20's ``rho_uq``, VG22's factor scales. The captions in those
    reports tell the reader to inspect exactly those ridges, so the plot
    contradicted the text it was captioned with (#233).

    Ordering rather than filtering, so nothing is hidden -- the cap simply
    consumes the list from a different end. An empty tuple means "model order",
    which is what every model without a distinguishing child structure gets, and
    those pair plots are byte-identical to before.

    The names are read from the definition rather than the trace so the intent
    is declared by the model, not inferred from what happened to be sampled.

    **One function for both engines that order a pair plot.** It began as the
    bivariate engine's, with the joint engine leading unconditionally with
    ``psi`` and ``conc`` and taking build order after them -- which reproduced
    #233 on that side, for VG24 as well as VG25.
    Consolidating is not a matter of applying the bivariate rules to joint
    models: the two engines' child blocks differ, and a naive merge would have
    handed VG24 the *bivariate* headline ``rho_uq`` and dropped the sign-speech
    correlation the model exists to estimate. The branches below are therefore
    modality-aware where the structure differs and shared where it does not.

    The engines that do **not** install this ordering -- univariate, univariate
    with random effects, and trivariate -- must have nothing to order, and
    ``tests/test_pair_plot_limits.py`` asserts exactly that, so registering a
    model with a distinguishing structure on one of them fails rather than
    silently ignoring its priority.
    """
    joint = _is_joint(definition)
    head = ["psi", "conc"] if joint else []

    # The definition-driven part, kept separate from the head above: a joint
    # model with no distinguishing child structure (VG15) must come out as the
    # bare head and keep the plot it has, not acquire the scale block below
    # because `psi` happened to make the list non-empty.
    priority: list[str] = []

    # Exploratory sex-shift variant of VG20 (issue #295): the two coefficients
    # it exists to estimate, ahead of the correlation it inherits.
    if getattr(definition, "sex_effect_sigma", None) is not None:
        priority += ["beta_sex_u", "beta_sex_q"]

    if getattr(definition, "use_cross_lag", False):
        priority.append("beta_lag")

    # VG25: the coefficient, then the two parameters its report asks the reader
    # to read it against -- the persistent association it must be
    # distinguishable from, and the scale of the signing child effect its
    # within-child baseline is built from.
    if getattr(definition, "use_sign_cross_lag", False):
        priority += ["beta_sign_lag", "rho_sign_q", "tau_subj_sign"]

    # VG20 and VG24: the correlated child block. The bivariate engine's block
    # holds one correlation; the joint engine's is 3x3, and `rho_sign_q` -- not
    # `rho_uq` -- is the one VG24 was registered to estimate.
    if getattr(definition, "subject_re_correlation_eta", None) is not None:
        priority += (
            ["rho_sign_q", "rho_u_sign", "rho_uq"] if joint else ["rho_uq"]
        )

    # VG22: the factor form emits rho_uq as a deterministic and carries a rate
    # scale per outcome. `subject_factor_corr` is deliberately absent -- a 4x4
    # matrix is 16 plot items and would consume the whole grid on its own.
    if getattr(definition, "subject_factor", None) is not None:
        priority += ["rho_uq", "tau_subj_u_1", "tau_subj_q_1"]

    # VG19: the intercept-and-rate block, whose two correlations are the part a
    # reader can actually test from an interval.
    for name in ("tau_subj_u", "tau_subj_q"):
        spec = getattr(definition, f"{name}_sigma", None)
        if getattr(spec, "tau1_sigma", None) is not None:
            priority += [f"{name}_1", f"{name}_rho", f"{name}_0"]

    if priority:
        priority += ["tau_subj_u", "tau_subj_q"]
        if joint:
            priority.append("tau_subj_sign")
        priority += ["tau_u", "tau_q"]
        if joint:
            priority.append("tau_sign")
    return tuple(dict.fromkeys([*head, *priority]))


def _is_joint(definition) -> bool:
    """Whether ``definition`` is fitted by the joint engine.

    Read off the composition association's own prior, which only the joint
    definitions carry -- the trivariate models have three marginal outcomes and
    no ``psi`` at all. Checked as a field rather than with ``isinstance`` to
    keep this module free of a definitions import, and because every other
    branch here reads the definition the same way.
    """
    return getattr(definition, "log_psi_mu", None) is not None


def pair_plot_var_names_fn(definition, posterior_vars):
    """A ``var_names_fn`` that applies :func:`pair_plot_priority`, or ``None``.

    ``None`` means model order, which is what the shared engine takes when no
    reordering is wanted -- so a model with nothing to prioritise keeps the plot
    it had. Names absent from ``posterior_vars`` are dropped, which is what lets
    the priority list name a parameter a sibling model emits and this one does
    not.
    """
    priority = pair_plot_priority(definition)
    if not priority:
        return None

    def _prioritise(names: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for name in (*priority, *names):
            if name in posterior_vars and name not in seen:
                ordered.append(name)
                seen.add(name)
        return ordered

    return _prioritise


def render_convergence_caveats(directory: str = ".") -> None:
    """Render a fit's convergence evidence and caveats as Quarto text.

    Use the shared assessment to distinguish a confirmed pass, a failure and
    incomplete evidence. A recorded publication exception is disclosed separately.
    Missing evidence cannot justify a pass. A clean assessed fit adds no callout.
    See test_report_cells.py for agreement with the diagnostic verdict.
    """
    import json
    import os

    caveats = read_convergence_caveats(directory)

    summary = None
    summary_path = os.path.join(directory, "diagnostics_summary.json")
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, encoding="utf-8") as handle:
                loaded = json.load(handle)
            summary = loaded if isinstance(loaded, dict) else None
        except OSError, ValueError:
            summary = None

    accepted = (summary or {}).get(ACCEPTED_EXCEPTION_KEY)
    status = hard_tier_status(summary)
    failed = status is HardConvergenceStatus.FAILED and not accepted
    unknown = status is HardConvergenceStatus.UNKNOWN and not accepted
    if not caveats and (summary is None or status is HardConvergenceStatus.PASSED):
        return

    if accepted:
        title = "Published under a recorded convergence exception"
        opening = (
            "This fit **did not clear** the hard convergence tier (R-hat and "
            "effective sample size). It is published under an exception recorded "
            "against the model, reproduced below with any other sampling caveats:"
        )
    elif failed:
        # Reachable only below reporting quality, where the gate records a
        # failing fit instead of stopping it. The closing sentence matches
        # `render_diagnostic_verdict`'s for the same case word for word, so the
        # two cells on one page cannot be read as disagreeing.
        title = "The hard convergence tier was not cleared"
        opening = (
            "This fit **did not clear** the hard convergence tier (R-hat and "
            "effective sample size), and no exception is recorded for it. The "
            "pipeline allows that only below reporting quality, where it records a "
            "fit that has not mixed rather than stopping it."
            + (" It also carries these sampling caveats:" if caveats else "")
        )
    elif unknown:
        title = "Convergence evidence is incomplete"
        opening = (
            "The hard convergence tier **cannot be assessed** from the recorded "
            "R-hat and effective sample size. Missing evidence does not establish "
            "that this fit cleared the tier."
            + (" The recorded sampling caveats are:" if caveats else "")
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
    if caveats:
        for caveat in caveats:
            print(f"- {caveat}")
        print()
    if accepted and accepted.get("reason"):
        print(f"**Why the exception was accepted:** {accepted['reason']}")
        print()
    if failed or unknown:
        print(
            "These results are provisional and must not be published as a completed fit."
        )
    else:
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
