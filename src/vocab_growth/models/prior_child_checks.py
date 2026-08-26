# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prior predictive checks that exercise the child random effects (issue #233).

``prior_predictive_checks`` draws from the prior and plots ``p_u_plot``,
``q_plot`` and ``p_s_plot`` — all three evaluated at **zero** study and child
effects. That is the right check for the mean function and it is the only check
these reports have ever had, which leaves a gap that #233 named precisely: the
figures a child-effect model puts in front of a reader cannot test the prior
that model was added for. VG19's cannot say whether ``tau1_sigma = 0.5`` per
year implies plausible individual trajectories, and VG20's cannot reveal a
defect in the correlated block, because neither figure contains a child.

This module fills the gap without touching the graph. Everything here is
computed in NumPy from prior draws the model already emits — the zero-effect
logit curves ``f_u_plot`` and ``h_plot``, the dispersion curves
``kappa_u_plot`` and ``kappa_s_plot``, and whichever child-effect scales the
definition carries — so no node is added, no fit is invalidated, and the check
can run at build time where the unseen-child block does not yet exist (it is
created in ``sample_posterior_predictive``).

Two properties make the output worth reading:

* **One child per prior draw, reused across the whole grid.** A child effect
  drawn independently at each age would give a scatter, not a trajectory, and
  would hide exactly the thing a slope prior needs testing for.
* **Actual nested Beta-Binomial counts.** Words understood are drawn from the
  child's own ``p_u`` at the grid's dispersion, and words spoken are then drawn
  **conditional on that draw** — the same nesting the likelihood uses, so a
  prior that implies impossible children shows up as impossible counts rather
  than as an implausible mean.

The child-effect construction mirrors ``common_bivariate``'s predictive path
branch for branch. It is a second implementation, which is a drift risk, so
``tests/test_prior_child_checks.py`` pins the correlated branch against the
graph's own ``unseen_child_correlated_delta_q`` at shared standard normals.
"""

from __future__ import annotations

import os

import dse_research_utils.plot.styles as plot_styles
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import betabinom

N_CHILD_CURVES = 300
"""Unseen children drawn for the trajectory figures.

One per prior draw, so this is also the number of prior draws consumed. Enough
to show the spread the prior implies without the fan becoming a solid block.
"""

ASSOCIATION_AGES = (12, 24, 36, 60)
"""Ages at which the induced joint (understood, spoken) association is drawn."""


def _flat(prior, name):
    """A prior variable as ``(draw, ...)``, chain and draw collapsed."""
    values = prior[name].values
    return values.reshape((-1,) + values.shape[2:])


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def child_effect_structure(prior, definition) -> str:
    """Which unseen-child construction this fit's prior draws support.

    Dispatch order matches ``common_bivariate.sample_posterior_predictive``:
    the factor form is checked before the slope form, because a factor model
    emits the slope form's parameter names too.
    """
    if "subject_factor_loadings" in prior:
        return "factor"
    if "tau_subj_u_1" in prior or "tau_subj_q_1" in prior:
        return "slope"
    if "rho_uq" in prior:
        return "correlated"
    if "tau_subj_u" in prior or "tau_subj_q" in prior:
        return "independent"
    return "none"


def unseen_child_deltas(prior, definition, ages_months, rng):
    """Logit-scale child offsets on ``ages_months`` for one child per draw.

    Returns ``(delta_u, delta_q)``, each ``(draw, n_ages)``, or ``(None, None)``
    when the fit carries no child effects. A model with a rate gives offsets
    that vary along the grid; a constant-offset model gives the same value at
    every age, which is the point of the comparison.
    """
    structure = child_effect_structure(prior, definition)
    if structure == "none":
        return None, None

    ages = np.asarray(ages_months, dtype=float)
    n_draws = _flat(prior, "f_u_plot").shape[0]
    ref = float(getattr(definition, "subject_slope_ref_age_months", 36.0) or 36.0)
    years = (ages - ref) / 12.0

    if structure == "factor":
        # VG22: b = z @ L.T over (b0u, b1u, b0q, b1q).
        loadings = _flat(prior, "subject_factor_loadings")  # (draw, 4, k)
        z = rng.standard_normal((n_draws, loadings.shape[2]))
        b = np.einsum("dik,dk->di", loadings, z)
        delta_u = b[:, 0][:, None] + b[:, 1][:, None] * years[None, :]
        delta_q = b[:, 2][:, None] + b[:, 3][:, None] * years[None, :]
        return delta_u, delta_q

    if structure == "slope":
        # VG19: an intercept and a rate per outcome, correlated within outcome.
        def block(name):
            tau0 = _flat(prior, f"{name}_0")
            tau1 = _flat(prior, f"{name}_1")
            rho = _flat(prior, f"{name}_rho")
            z0, z1 = rng.standard_normal((2, n_draws))
            b0 = tau0 * z0
            b1 = tau1 * (rho * z0 + np.sqrt(1.0 - rho**2) * z1)
            return b0[:, None] + b1[:, None] * years[None, :]

        return block("tau_subj_u"), block("tau_subj_q")

    tau_u = _flat(prior, "tau_subj_u")
    tau_q = _flat(prior, "tau_subj_q")
    z_u, z_q = rng.standard_normal((2, n_draws))
    delta_u = tau_u * z_u

    if structure == "correlated":
        # VG20, and the same Cholesky construction the graph uses.
        rho = _flat(prior, "rho_uq")
        delta_q = tau_q * (rho * z_u + np.sqrt(1.0 - rho**2) * z_q)
    else:
        delta_q = tau_q * z_q

    return (
        np.broadcast_to(delta_u[:, None], (n_draws, ages.size)).copy(),
        np.broadcast_to(delta_q[:, None], (n_draws, ages.size)).copy(),
    )


def unseen_child_curves(prior, definition, rng, *, n_children=N_CHILD_CURVES):
    """One unseen child per draw: their ``p_u`` and ``q`` curves, and counts.

    Returns a dict with the plot ages, the child-level probability curves, the
    nested Beta-Binomial count draws, and the zero-effect population curves for
    comparison. ``None`` when the fit carries no child effects.
    """
    # Checked before anything is read, so a model without child effects returns
    # cleanly rather than failing on a variable it has no reason to carry.
    if child_effect_structure(prior, definition) == "none":
        return None

    ages = np.asarray(prior["X_plot"].values)
    f_u = _flat(prior, "f_u_plot")
    h = _flat(prior, "h_plot")
    kappa_u = _flat(prior, "kappa_u_plot")
    kappa_s = _flat(prior, "kappa_s_plot")

    take = min(n_children, f_u.shape[0])
    f_u, h, kappa_u, kappa_s = f_u[:take], h[:take], kappa_u[:take], kappa_s[:take]

    delta_u, delta_q = unseen_child_deltas(prior, definition, ages, rng)
    delta_u, delta_q = delta_u[:take], delta_q[:take]

    p_u_child = _sigmoid(f_u + delta_u)
    q_child = _sigmoid(h + delta_q)

    n_trials = int(definition.n_trials)
    understood = betabinom.rvs(
        n_trials,
        np.clip(p_u_child * kappa_u, 1e-6, None),
        np.clip((1.0 - p_u_child) * kappa_u, 1e-6, None),
        random_state=rng,
    )
    # Spoken is drawn CONDITIONAL on the understood draw, which is the nesting
    # the likelihood uses. Drawing it against the reference inventory instead
    # would let a child say more words than they understand.
    spoken = betabinom.rvs(
        np.maximum(understood, 1),
        np.clip(q_child * kappa_s, 1e-6, None),
        np.clip((1.0 - q_child) * kappa_s, 1e-6, None),
        random_state=rng,
    )
    spoken = np.where(understood > 0, spoken, 0)

    return {
        "ages": ages,
        "p_u_child": p_u_child,
        "q_child": q_child,
        "understood": understood,
        "spoken": spoken,
        "p_u_population": _sigmoid(f_u),
        "q_population": _sigmoid(h),
        "delta_u": delta_u,
        "delta_q": delta_q,
        "structure": child_effect_structure(prior, definition),
    }


def _save(fig, output_dir, filename):
    if output_dir is None or filename is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, f"{filename}.svg"))


def plot_unseen_child_trajectories(curves, definition, *, output_dir=None):
    """Individual child trajectories around **one fixed** population curve.

    The figure #233 asks for on VG19, and the reason it holds the mean function
    fixed is the whole point of it. Every other figure here varies the mean
    function draw by draw, which is correct for "where would one more child
    fall?" but useless for "is this child-effect prior plausible?": with a
    different trend under every child, the curves cross whatever the child
    block does, and a rate prior cannot be distinguished from a constant one.

    Fixing the trend at the zero-effect median and varying only the child effect
    separates them. Under a **constant offset** the curves are parallel on the
    logit scale and can never cross, however wide ``tau_subject`` is. Under a
    **rate** they fan, converge and cross, and an implausible ``tau1`` shows as
    children reaching the ceiling or the floor well inside the reported range.
    """
    ages = curves["ages"]
    n_trials = int(definition.n_trials)
    fig, axes = plt.subplots(1, 2, figsize=plot_styles.FIGSIZE_XL)

    for ax, pop_key, delta_key, label in (
        (axes[0], "p_u_population", "delta_u", "Expected words understood"),
        (axes[1], "q_population", "delta_q", "Production ratio $q$"),
    ):
        scale = n_trials if delta_key == "delta_u" else 1.0
        base = _logit(np.median(curves[pop_key], axis=0))
        deltas = curves[delta_key]
        values = _sigmoid(base[None, :] + deltas) * scale
        for row in values[: min(200, values.shape[0])]:
            ax.plot(ages, row, color=plot_styles.COLOUR_ORANGE, alpha=0.12, lw=1.0)
        ax.plot(ages, _sigmoid(base) * scale, lw=3, color="black", label="Zero effect")
        ax.set_xlabel("Age (months)")
        ax.set_ylabel(label)
        ax.legend(loc="upper left", frameon=True)
        ax.set_ylim(0, 1 if delta_key == "delta_q" else n_trials)

    fig.suptitle(
        "Prior child trajectories around a fixed population curve "
        "(child effects only)"
    )
    fig.tight_layout()
    _save(fig, output_dir, "prior_unseen_children")
    return fig


def plot_unseen_child_counts(curves, definition, *, output_dir=None):
    """Nested Beta-Binomial count draws for the same unseen children.

    The mean-function figures cannot show these: a prior can imply a perfectly
    reasonable expected trajectory and still put a real administration at an
    impossible count once the child effect and the dispersion are both in play.
    """
    ages = curves["ages"]
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    step = max(1, ages.size // 60)
    idx = np.arange(0, ages.size, step)

    for counts, colour, label in (
        (curves["understood"], plot_styles.COLOUR_BLUE, "Words understood"),
        (curves["spoken"], plot_styles.COLOUR_ORANGE, "Words spoken"),
    ):
        lo, mid, hi = np.percentile(counts[:, idx], [5.5, 50, 94.5], axis=0)
        ax.fill_between(ages[idx], lo, hi, alpha=0.20, color=colour)
        ax.plot(ages[idx], mid, lw=2.5, color=colour, label=f"{label} (median, 89%)")

    ax.set_xlabel("Age (months)")
    ax.set_ylabel(f"Words (of {int(definition.n_trials):,})")
    ax.set_ylim(0, int(definition.n_trials))
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("Prior nested Beta-Binomial counts for one unseen child per draw")
    fig.tight_layout()
    _save(fig, output_dir, "prior_unseen_child_counts")
    return fig


def plot_prior_joint_association(curves, definition, *, output_dir=None):
    """The joint (understood, spoken) association the prior induces.

    The figure #233 asks for on VG20. The correlated block's whole purpose is
    the alignment between a child's comprehension standing and their conversion
    standing, and no zero-effect figure can show it. Each panel is one age; the
    printed correlation is of the *drawn counts*, which is what a defect in the
    correlated block would move.
    """
    ages = curves["ages"]
    fig, axes = plt.subplots(
        1, len(ASSOCIATION_AGES), figsize=plot_styles.FIGSIZE_XL, sharex=True, sharey=True
    )
    n_trials = int(definition.n_trials)

    for ax, age in zip(np.atleast_1d(axes), ASSOCIATION_AGES, strict=False):
        j = int(np.argmin(np.abs(ages - age)))
        u = curves["understood"][:, j]
        s = curves["spoken"][:, j]
        ax.scatter(u, s, s=8, alpha=0.35, color=plot_styles.COLOUR_BLUE)
        ax.plot([0, n_trials], [0, n_trials], ls="--", lw=1, color="grey")
        r = float(np.corrcoef(u, s)[0, 1]) if np.std(u) > 0 and np.std(s) > 0 else np.nan
        ax.set_title(f"{ages[j]:.0f} mo (r = {r:+.2f})")
        ax.set_xlabel("Words understood")
        ax.set_xlim(0, n_trials)
        ax.set_ylim(0, n_trials)
    np.atleast_1d(axes)[0].set_ylabel("Words spoken")

    # The count correlation in each panel is NOT the correlated block's doing:
    # it is dominated by the shared age trend, which moves both outcomes
    # together draw by draw, and it would be positive under independent child
    # effects too. The deviate correlation is the one the block controls, so it
    # is reported beside it -- if the two are confused, this figure looks like
    # confirmation of a correlation the model may not have estimated.
    deviate_r = float(
        np.corrcoef(curves["delta_u"][:, 0], curves["delta_q"][:, 0])[0, 1]
    )
    fig.suptitle(
        "Prior joint association across unseen children — "
        f"child-deviate correlation {deviate_r:+.2f}; "
        "the per-panel r also carries the shared age trend"
    )
    fig.tight_layout()
    _save(fig, output_dir, "prior_joint_association")
    return fig


def run(context, definition, *, seed=None):
    """Draw the child-level prior checks for ``definition``, if it has any.

    Called from the engine's prior-predictive stage, after the population
    figures. Returns the figure names written, so a caller can report them; an
    empty list means the fit carries no child effects and nothing was drawn.
    """
    prior = context.prior_samples.prior
    if "X_plot" not in prior and hasattr(context.prior_samples, "constant_data"):
        prior = prior.assign(X_plot=context.prior_samples.constant_data["X_plot"])

    rng = np.random.default_rng(
        context.sampling.random_seed if seed is None else seed
    )
    curves = unseen_child_curves(prior, definition, rng)
    if curves is None:
        return []

    output_dir = context.reporting.output_dir
    written = []

    fig = plot_unseen_child_trajectories(curves, definition, output_dir=output_dir)
    context.plots["prior_unseen_children"] = fig
    written.append("prior_unseen_children")
    plt.close(fig)

    fig = plot_unseen_child_counts(curves, definition, output_dir=output_dir)
    context.plots["prior_unseen_child_counts"] = fig
    written.append("prior_unseen_child_counts")
    plt.close(fig)

    # The joint association is only interesting where the two child effects can
    # actually be aligned -- a model drawing them independently induces the
    # association its mean function implies and nothing more.
    if curves["structure"] in {"correlated", "factor"}:
        fig = plot_prior_joint_association(curves, definition, output_dir=output_dir)
        context.plots["prior_joint_association"] = fig
        written.append("prior_joint_association")
        plt.close(fig)

    return written
