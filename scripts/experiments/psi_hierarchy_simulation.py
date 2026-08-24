#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Is VG15's four-group ``psi`` hierarchy recoverable, and is its prior the cause?

The question
------------
[#226](https://github.com/dseinternational/vocabulary-growth/issues/226) reports
``psi`` and ``psi_study`` recovering biased low, through a ``tau_psi`` that is
itself underestimated, and proposes the standard few-groups pathology: with four
informing sources the between-study spread is barely identified, and
``tau_psi ~ HalfNormal(1.0)`` -- prior median 0.67 against truths of 0.9-1.3 --
pulls it down, which over-shrinks the group estimates. The issue's evidence is
three ``test``-tier VG15 replicates, one of which cleared the convergence gate.

Refitting VG15 more times would cost days and still confound the hierarchy with
everything else in a joint three-outcome model. This isolates the hierarchy.

The reduction
-------------
VG15's association block is, exactly::

    log_psi     ~ Normal(log_psi_mu, log_psi_sigma)
    tau_psi     ~ HalfNormal(tau_psi_sigma)
    z_psi       ~ ZeroSumNormal(sigma=sqrt(J/(J-1)), shape=J)
    log_psi_j    = log_psi + tau_psi * z_psi[j]

and each informing study's cross-tab enters through a Dirichlet-Multinomial over
a Plackett composition. Replace only that last line by its Laplace
approximation -- ``y_j ~ Normal(log_psi_j, s_j^2)``, with ``s_j`` computed from
that study's *actual* cells -- and the whole model becomes Gaussian, with a
closed-form posterior. No MCMC, no convergence gate, no sampling noise, and
thousands of replicates in seconds instead of three in a day.

What that buys, and what it costs:

* **Buys** a clean answer to items 2 and 3 of #226. The prior on ``tau_psi`` can
  be swept against a *fixed* truth, which is the comparison the fitting harness
  could not make until the cross-definition seam landed alongside this, and
  which registered-variant robustness runs cannot make at all.
* **Costs** each study's likelihood departing from a normal one, the joint
  estimation of ``r``, ``q`` and ``conc`` alongside ``psi``, and every
  correlation between ``psi`` and the rest of VG15. So a bias found here is a
  *lower bound* on VG15's: it is the part attributable to the hierarchy alone.
  A bias NOT found here would have been the interesting outcome, because it
  would have located the problem elsewhere.

The ``s_j`` are not invented. They come from the observed information of the
model's own Dirichlet-Multinomial likelihood, evaluated on the real cross-tab
cells with each row's own observed margins -- so the *relative* precision of the
four sources is the data's, not an assumption.

The design effect
-----------------
#226's sharpest observation is that the pattern tracks **children**, not
administrations: ``nz_01`` has the second-largest number of administrations and
the fewest children, and is shrunk hardest. Independent administrations would
make ``s_j`` scale with administrations; perfectly redundant repeats within a
child would make it scale with children. ``--design-rho`` sweeps between
(``0`` = administrations, ``1`` = children), so the study can say whether that
observation is reproduced by the hierarchy or needs another explanation.

Usage
-----
    python scripts/experiments/psi_hierarchy_simulation.py --replicates 2000
    python scripts/experiments/psi_hierarchy_simulation.py --verify

``--verify`` fits one replicate with PyMC, using PyMC's own ``ZeroSumNormal``,
and compares it against the closed form. Two implementations of one model is a
drift risk, and that is the guard on it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from vocab_growth import environment as env  # noqa: E402
from vocab_growth.models.common_joint_modality import (  # noqa: E402
    PROD_CELL_COLUMNS,
)
from vocab_growth.models.definitions import MODEL_REGISTRY  # noqa: E402
from vocab_growth.reporting import (  # noqa: E402
    console,
    dataframe_table,
    key_value_table,
)

FOUR_CELL_COLUMNS = [
    "understood_only",
    "signed_only",
    "spoken_only",
    "signed_spoken",
]


# ==========================================================================
# The Plackett composition, in NumPy
# ==========================================================================


def plackett_pi_both(r, q, psi):
    """P(both | understood) under a Plackett copula with odds ratio ``psi``.

    The engine's own rationalised form (``common_joint_modality._plackett_pi_both``),
    in NumPy: ``2 psi r q / (S + disc)`` rather than ``(S - disc)/(2(psi-1))``,
    which is continuous at ``psi = 1`` instead of 0/0.
    """
    S = 1.0 + (r + q) * (psi - 1.0)
    disc = np.sqrt(np.maximum(S * S - 4.0 * psi * (psi - 1.0) * r * q, 0.0))
    pi_both = 2.0 * psi * r * q / np.maximum(S + disc, 1e-12)
    return np.clip(pi_both, np.maximum(0.0, r + q - 1.0), np.minimum(r, q))


def four_cell_pi(r, q, psi, epsilon=1e-12):
    """The four-way within-understood composition, in the engine's cell order."""
    both = plackett_pi_both(r, q, psi)
    stack = np.stack(
        [
            np.maximum(1 - r - q + both, epsilon),
            np.maximum(r - both, epsilon),
            np.maximum(q - both, epsilon),
            np.maximum(both, epsilon),
        ],
        axis=-1,
    )
    return stack / stack.sum(axis=-1, keepdims=True)


def produced_pi(r, q, psi, epsilon=1e-12):
    """nz_01's three-cell within-produced composition (no comprehension total)."""
    both = plackett_pi_both(r, q, psi)
    stack = np.stack(
        [
            np.maximum(r - both, epsilon),
            np.maximum(q - both, epsilon),
            np.maximum(both, epsilon),
        ],
        axis=-1,
    )
    return stack / stack.sum(axis=-1, keepdims=True)


def dirichlet_multinomial_logpmf(counts, alpha):
    """Row-wise Dirichlet-Multinomial log-density, dropping the count constant.

    The multinomial coefficient does not depend on ``psi``, so it cancels out of
    every derivative taken below and is omitted rather than computed.
    """
    a0 = alpha.sum(axis=-1)
    n = counts.sum(axis=-1)
    return (
        gammaln(a0)
        - gammaln(a0 + n)
        + (gammaln(counts + alpha) - gammaln(alpha)).sum(axis=-1)
    )


# ==========================================================================
# What each study's cross-tab actually carries about its own log psi
# ==========================================================================


def _observed_information(log_likelihood, at, step=1e-3):
    """Negative curvature of a scalar log-likelihood, by central differences."""
    plus = log_likelihood(at + step)
    minus = log_likelihood(at - step)
    centre = log_likelihood(at)
    return -(plus - 2.0 * centre + minus) / (step * step)


def study_information(analysis_df, *, conc, log_psi_truth):
    """Per-study Fisher information for ``log psi_j``, from the real cells.

    Each row contributes its own Dirichlet-Multinomial term at its own observed
    margins, so the four sources' *relative* precision is the data's. Rows are
    added because the model treats them as conditionally independent given the
    study's ``psi_j``; whether that overstates the information is exactly what
    ``--design-rho`` puts a number on.
    """
    rows = []

    four = analysis_df[analysis_df["signed_spoken"].notna()]
    for study, block in four.groupby("study"):
        counts = block[FOUR_CELL_COLUMNS].to_numpy(dtype=float)
        # Cell order in the frame is (understood_only, signed_only, spoken_only,
        # signed_spoken); the composition's is (neither, sign_only, speak_only,
        # both) -- the same order, under the two naming conventions.
        total = counts.sum(axis=1)
        r = (counts[:, 1] + counts[:, 3]) / total
        q = (counts[:, 2] + counts[:, 3]) / total

        def log_likelihood(value, counts=counts, r=r, q=q, total=total):
            pi = four_cell_pi(r, q, float(np.exp(value)))
            return float(dirichlet_multinomial_logpmf(counts, conc * pi).sum())

        rows.append(
            {
                "study": study,
                "source": "four-cell",
                "administrations": len(block),
                "children": int(block["subject_id"].nunique()),
                "information": _observed_information(log_likelihood, log_psi_truth),
            }
        )

    if "prod_signed_spoken" in analysis_df.columns:
        prod = analysis_df[analysis_df["prod_signed_spoken"].notna()]
        for study, block in prod.groupby("study"):
            counts = block[PROD_CELL_COLUMNS].to_numpy(dtype=float)
            total = counts.sum(axis=1)
            # Within-produced margins: the renormalised composition's own r and
            # q are the produced shares, which is what nz_01 observes.
            r = (counts[:, 0] + counts[:, 2]) / total
            q = (counts[:, 1] + counts[:, 2]) / total

            def log_likelihood(value, counts=counts, r=r, q=q):
                pi = produced_pi(r, q, float(np.exp(value)))
                return float(dirichlet_multinomial_logpmf(counts, conc * pi).sum())

            rows.append(
                {
                    "study": study,
                    "source": "produced",
                    "administrations": len(block),
                    "children": int(block["subject_id"].nunique()),
                    "information": _observed_information(log_likelihood, log_psi_truth),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No psi-informing cross-tab rows found.")
    # A study can inform psi through both a four-cell and a produced table; the
    # information adds, as the likelihood terms do.
    frame = (
        frame.groupby("study")
        .agg(
            sources=("source", lambda values: "+".join(sorted(values))),
            administrations=("administrations", "sum"),
            children=("children", "max"),
            information=("information", "sum"),
        )
        .reset_index()
    )
    return frame.sort_values("study").reset_index(drop=True)


def standard_errors(frame, design_rho):
    """Per-study ``s_j``, discounted for repeated administrations of one child.

    ``design_rho = 0`` treats every administration as an independent draw;
    ``design_rho = 1`` treats a child's repeats as carrying no more information
    than one of them, so ``s_j`` tracks children. The usual design effect,
    ``1 + (mbar - 1) rho``, interpolates.
    """
    mbar = frame["administrations"].to_numpy(float) / frame["children"].to_numpy(float)
    design_effect = 1.0 + (mbar - 1.0) * design_rho
    return np.sqrt(design_effect / frame["information"].to_numpy(float))


# ==========================================================================
# The hierarchy, in closed form
# ==========================================================================


def zero_sum_basis(n_groups):
    """``L`` with ``L L' = (J/(J-1)) (I - 11'/J)``: the ZeroSumNormal covariance.

    Any basis of the zero-sum subspace gives the same *distribution*, which is
    all the posterior depends on; ``--verify`` checks that against PyMC's own
    construction rather than against this comment.
    """
    projection = np.eye(n_groups) - np.ones((n_groups, n_groups)) / n_groups
    # Columns of an orthonormal basis for the zero-sum subspace.
    eigenvalues, eigenvectors = np.linalg.eigh(projection)
    basis = eigenvectors[:, eigenvalues > 0.5]
    return basis * np.sqrt(n_groups / (n_groups - 1.0))


def posterior_grid(
    y,
    s,
    *,
    basis,
    log_psi_mu,
    log_psi_sigma,
    tau_sigma,
    tau_grid,
    log_psi_grid,
):
    """The exact joint posterior of ``(log_psi, tau)`` on a grid, plus per-group
    conditional moments.

    ``y | log_psi, tau ~ Normal(log_psi * 1, tau^2 L L' + diag(s^2))``, which is
    the marginal likelihood with the group deviations integrated out in closed
    form. The per-group posterior is then a finite Gaussian mixture over the
    grid, so its moments and its CDF are exact up to the grid.

    Vectorised over ``log_psi`` rather than looped, which is what makes
    thousands of replicates practical: at fixed ``tau`` the quadratic form is a
    quadratic in the centre, the conditional group mean is affine in it, and the
    conditional group variance does not depend on it at all.
    """
    n_groups = y.size
    ones = np.ones(n_groups)
    noise = np.diag(s**2)
    centres = log_psi_grid

    n_tau = tau_grid.size
    log_weights = np.empty((n_tau, centres.size))
    group_mean = np.empty((n_tau, centres.size, n_groups))
    group_var = np.empty((n_tau, centres.size, n_groups))

    for i, tau in enumerate(tau_grid):
        covariance = tau**2 * (basis @ basis.T) + noise
        precision = np.linalg.inv(covariance)
        sign, logdet = np.linalg.slogdet(covariance)
        assert sign > 0

        # residual' P residual, expanded in the centre.
        a = y @ precision @ y
        b = ones @ precision @ y
        c = ones @ precision @ ones
        quadratic = a - 2.0 * centres * b + centres**2 * c
        log_weights[i] = -0.5 * (
            logdet + quadratic + n_groups * np.log(2 * np.pi)
        )

        # Conditional moments of the group vector given (log_psi, tau). The
        # deviations live in the (J-1)-dimensional zero-sum space, so the
        # conjugate update is taken there and mapped back.
        scaled = tau * basis
        posterior_covariance = np.linalg.inv(
            np.eye(n_groups - 1) + scaled.T @ np.linalg.solve(noise, scaled)
        )
        deviation_covariance = scaled @ posterior_covariance @ scaled.T
        gain = deviation_covariance @ np.linalg.inv(noise)
        intercept = gain @ y
        slope = ones - gain @ ones
        group_mean[i] = centres[:, None] * slope[None, :] + intercept[None, :]
        group_var[i] = np.diag(deviation_covariance)[None, :]

    log_prior = (
        norm.logpdf(centres, log_psi_mu, log_psi_sigma)[None, :]
        + (np.log(2.0) + norm.logpdf(tau_grid, 0.0, tau_sigma))[:, None]
    )
    log_posterior = log_weights + log_prior
    log_posterior -= log_posterior.max()
    weights = np.exp(log_posterior)
    weights /= weights.sum()
    return weights, group_mean, group_var


def mixture_moments(weights, means, variances):
    """Mean and SD of a weighted Gaussian mixture, per group."""
    flat_w = weights.ravel()
    flat_m = means.reshape(flat_w.size, -1)
    flat_v = variances.reshape(flat_w.size, -1)
    mean = flat_w @ flat_m
    second = flat_w @ (flat_v + flat_m**2)
    return mean, np.sqrt(np.maximum(second - mean**2, 0.0))


def mixture_cdf(weights, means, variances, at):
    """``P(X <= at)`` under the mixture, per group.

    Coverage is read from the CDF at the truth rather than by inverting for the
    interval endpoints. The two are the same statement -- the truth is inside
    the equal-tailed interval exactly when its posterior CDF is between the
    tails -- and this one costs one evaluation instead of a root-find per group
    per replicate, which is the difference between a study that runs in minutes
    and one that runs overnight.
    """
    flat_w = weights.ravel()
    flat_m = means.reshape(flat_w.size, -1)
    flat_s = np.sqrt(variances.reshape(flat_w.size, -1))
    return flat_w @ norm.cdf((np.asarray(at)[None, :] - flat_m) / flat_s)


def scalar_posterior(weights, grid, axis):
    """Marginal mean, median and CDF of one grid axis."""
    marginal = weights.sum(axis=1 - axis)
    order = np.argsort(grid)
    values, mass = grid[order], marginal[order]
    cumulative = np.cumsum(mass)
    return {
        "mean": float(values @ mass),
        "median": float(np.interp(0.5, cumulative, values)),
        "cdf": lambda x: float(np.interp(x, values, cumulative)),
    }


def covered(cdf_value, probability=0.89):
    """Is the truth inside the equal-tailed interval at this level?"""
    tail = (1.0 - probability) / 2.0
    return bool(tail <= cdf_value <= 1.0 - tail)


# ==========================================================================
# The study
# ==========================================================================


def simulate_and_score(
    *,
    s,
    basis,
    tau_truth,
    log_psi_truth,
    tau_sigma,
    log_psi_mu,
    log_psi_sigma,
    replicates,
    rng,
    tau_grid,
    log_psi_grid,
):
    """One cell of the study: many replicates at one truth and one prior."""
    n_groups = s.size
    records = []
    for _ in range(replicates):
        deviations = tau_truth * (basis @ rng.standard_normal(n_groups - 1))
        group_truth = log_psi_truth + deviations
        y = group_truth + s * rng.standard_normal(n_groups)

        weights, group_mean, group_var = posterior_grid(
            y,
            s,
            basis=basis,
            log_psi_mu=log_psi_mu,
            log_psi_sigma=log_psi_sigma,
            tau_sigma=tau_sigma,
            tau_grid=tau_grid,
            log_psi_grid=log_psi_grid,
        )
        tau_post = scalar_posterior(weights, tau_grid, axis=0)
        centre_post = scalar_posterior(weights, log_psi_grid, axis=1)
        mean, sd = mixture_moments(weights, group_mean, group_var)
        group_cdf = mixture_cdf(weights, group_mean, group_var, group_truth)

        record = {
            "tau_median": tau_post["median"],
            "tau_covered": covered(tau_post["cdf"](tau_truth)),
            "log_psi_mean": centre_post["mean"],
            "log_psi_covered": covered(centre_post["cdf"](log_psi_truth)),
        }
        for j in range(n_groups):
            # Deviations, not levels, and each measured from its own centre: the
            # true one for the truth and the estimated one for the estimate.
            # Measuring the truth from the *estimated* centre would fold that
            # centre's replicate-level error into the regressor and attenuate
            # every slope below -- which would look exactly like shrinkage.
            record[f"group{j}_truth_deviation"] = deviations[j]
            record[f"group{j}_mean_deviation"] = mean[j] - centre_post["mean"]
            record[f"group{j}_sd"] = sd[j]
            record[f"group{j}_covered"] = covered(group_cdf[j])
        records.append(record)
    return pd.DataFrame(records)


def _slope(frame, groups):
    """Regression slope of recovered on true group deviation.

    1.0 means the group estimates track their truths; 0 means they are pulled
    entirely to the centre. This is the quantity #226 reads off its table by eye
    ("posterior means cluster at 1.9-2.4 largely independently of the truth").
    """
    truth = np.concatenate([frame[f"group{j}_truth_deviation"] for j in groups])
    estimate = np.concatenate([frame[f"group{j}_mean_deviation"] for j in groups])
    return float(np.polyfit(truth, estimate, 1)[0])


def oracle_slope(tau, s):
    """The same slope for a model that KNEW tau and the centre.

    Two attenuations compose, and only one of them is the hierarchy's. The
    posterior deviation shrinks the *observed* deviation by tau^2/(tau^2+s^2),
    and the observed deviation is itself the true one plus noise, which
    attenuates the regression by the same factor again -- so even a correctly
    specified, oracle-tau model has a slope of ``(tau^2/(tau^2+s^2))^2``, not 1.

    Reporting the ratio of the realised slope to this is what separates "shrunk
    because the hierarchy's spread was underestimated", which is #226's proposed
    mechanism, from "shrunk because the source is noisy", which no prior can fix.
    """
    factor = tau**2 / (tau**2 + np.asarray(s) ** 2)
    return factor**2


def verify_against_pymc(y, s, basis, tau_sigma, log_psi_mu, log_psi_sigma, seed):
    """Fit one replicate with PyMC and compare against the closed form.

    Two implementations of one model is a drift risk. This is the guard: PyMC
    builds the hierarchy with its own ``ZeroSumNormal``, exactly as VG15 does,
    and the two posteriors must agree.
    """
    import arviz as az
    import pymc as pm

    n_groups = y.size
    with pm.Model() as model:
        log_psi = pm.Normal("log_psi", log_psi_mu, log_psi_sigma)
        tau = pm.HalfNormal("tau_psi", sigma=tau_sigma)
        z = pm.ZeroSumNormal(
            "z_psi", sigma=float(np.sqrt(n_groups / (n_groups - 1))), shape=n_groups
        )
        group = pm.Deterministic("log_psi_study", log_psi + tau * z)
        pm.Normal("y", mu=group, sigma=s, observed=y)
        trace = pm.sample(
            2000, tune=2000, chains=4, random_seed=seed, progressbar=False,
            target_accept=0.95,
        )

    summary = az.summary(trace, var_names=["log_psi", "tau_psi", "log_psi_study"])
    return model, trace, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument(
        "--tau-truth",
        type=float,
        nargs="+",
        default=[0.45, 0.70, 0.95, 1.30],
        help=(
            "True between-study SDs to test. The default spans VG15's three "
            "recovery truths (0.446, 0.923, 1.279)."
        ),
    )
    parser.add_argument(
        "--tau-prior",
        type=float,
        nargs="+",
        default=[0.3, 1.0, 2.0],
        help="HalfNormal scales: the registered tau-psi-narrow / record / -wide.",
    )
    parser.add_argument(
        "--design-rho",
        type=float,
        default=1.0,
        help=(
            "0 = a study's information tracks administrations; 1 = it tracks "
            "children. #226 argues for the latter (default: 1)."
        ),
    )
    parser.add_argument(
        "--se-scale",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0, 8.0],
        help=(
            "Multiply every s_j by each of these, keeping their relative sizes "
            "(which are the data's). The Laplace s_j hold r, q and conc FIXED at "
            "their observed/prior values, so they are a floor on what VG15's "
            "per-study likelihood actually costs; sweeping the scale asks how "
            "much less informative the sources would have to be for the "
            "hierarchy alone to produce the shrinkage #226 reports."
        ),
    )
    parser.add_argument("--conc", type=float, default=None, help="Dirichlet-Multinomial concentration (default: VG15's prior median).")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--verify", action="store_true", help="Cross-check one replicate against PyMC and stop.")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    env.set_output_root(args.output_dir)

    definition = MODEL_REGISTRY["vg15"]
    conc = float(np.exp(definition.log_conc_mu)) if args.conc is None else args.conc
    log_psi_truth = float(np.log(2.34))  # VG15's reported population association.

    console.print("[dim]Preparing VG15's analysis frame ...[/dim]")
    analysis_df = _prepared_frame(definition)
    information = study_information(
        analysis_df, conc=conc, log_psi_truth=log_psi_truth
    )
    s_admin = standard_errors(information, 0.0)
    s = standard_errors(information, args.design_rho)
    information = information.assign(
        se_administrations=s_admin, se_effective=s
    )

    key_value_table(
        "Reduction",
        [
            ("Informing studies", len(information)),
            ("Dirichlet-Multinomial concentration", round(conc, 2)),
            ("Population log psi (truth)", round(log_psi_truth, 4)),
            ("Design rho", args.design_rho),
            ("Standard-error scales", ", ".join(str(v) for v in args.se_scale)),
            ("Replicates per cell", args.replicates),
        ],
    )
    dataframe_table(
        information.round(4), title="What each source carries about its own log psi"
    )

    n_groups = len(information)
    basis = zero_sum_basis(n_groups)
    tau_grid = np.linspace(1e-3, 4.0, 240)
    log_psi_grid = np.linspace(log_psi_truth - 3.0, log_psi_truth + 3.0, 181)

    if args.verify:
        rng = np.random.default_rng(args.seed)
        deviations = 0.95 * (basis @ rng.standard_normal(n_groups - 1))
        y = log_psi_truth + deviations + s * rng.standard_normal(n_groups)
        weights, group_mean, group_var = posterior_grid(
            y, s, basis=basis, log_psi_mu=definition.log_psi_mu,
            log_psi_sigma=definition.log_psi_sigma, tau_sigma=1.0,
            tau_grid=tau_grid, log_psi_grid=log_psi_grid,
        )
        mean, sd = mixture_moments(weights, group_mean, group_var)
        closed = {
            "log_psi": scalar_posterior(weights, log_psi_grid, axis=1)["mean"],
            "tau_psi": scalar_posterior(weights, tau_grid, axis=0)["mean"],
            **{f"log_psi_study[{j}]": mean[j] for j in range(n_groups)},
        }
        _model, _trace, summary = verify_against_pymc(
            y, s, basis, 1.0, definition.log_psi_mu, definition.log_psi_sigma,
            args.seed,
        )
        comparison = pd.DataFrame(
            {
                "closed_form": pd.Series(closed),
                "pymc": summary["mean"],
                "mcse": summary["mcse_mean"],
            }
        )
        comparison["z"] = (
            comparison["closed_form"] - comparison["pymc"]
        ) / comparison["mcse"]
        dataframe_table(
            comparison.reset_index().round(4),
            title="Closed form against PyMC (|z| < 3 is agreement within MCMC error)",
        )
        console.print(f"[dim]Group posterior SDs (closed form): {np.round(sd, 4)}[/dim]")
        return 0

    rng = np.random.default_rng(args.seed)
    cells = []
    per_group = []
    for se_scale in args.se_scale:
        s_cell = s * se_scale
        for tau_sigma in args.tau_prior:
            for tau_truth in args.tau_truth:
                frame = simulate_and_score(
                    s=s_cell, basis=basis, tau_truth=tau_truth,
                    log_psi_truth=log_psi_truth, tau_sigma=tau_sigma,
                    log_psi_mu=definition.log_psi_mu,
                    log_psi_sigma=definition.log_psi_sigma,
                    replicates=args.replicates, rng=rng,
                    tau_grid=tau_grid, log_psi_grid=log_psi_grid,
                )
                cells.append(
                    {
                        "se_scale": se_scale,
                        "tau_prior_sigma": tau_sigma,
                        "tau_truth": tau_truth,
                        "tau_median": frame["tau_median"].mean(),
                        "tau_bias": frame["tau_median"].mean() - tau_truth,
                        "tau_coverage89": frame["tau_covered"].mean(),
                        "log_psi_bias": frame["log_psi_mean"].mean() - log_psi_truth,
                        "log_psi_coverage89": frame["log_psi_covered"].mean(),
                        "group_slope": _slope(frame, range(n_groups)),
                        "oracle_slope": float(
                            np.mean(oracle_slope(tau_truth, s_cell))
                        ),
                        "group_coverage89": np.mean(
                            [frame[f"group{j}_covered"].mean() for j in range(n_groups)]
                        ),
                    }
                )
                for j, study in enumerate(information["study"]):
                    per_group.append(
                        {
                            "se_scale": se_scale,
                            "tau_prior_sigma": tau_sigma,
                            "tau_truth": tau_truth,
                            "study": study,
                            "children": int(information["children"].iloc[j]),
                            "se": float(s_cell[j]),
                            "slope": _slope(frame, [j]),
                            "oracle_slope": float(
                                oracle_slope(tau_truth, s_cell[j])
                            ),
                            "coverage89": float(frame[f"group{j}_covered"].mean()),
                        }
                    )

    matrix = pd.DataFrame(cells)
    groups = pd.DataFrame(per_group)
    dataframe_table(
        matrix[matrix["se_scale"] == 1.0].drop(columns="se_scale").round(4),
        title="tau_psi recovery by prior and truth, at the cross-tabs' own precision",
    )
    registered = (matrix["tau_prior_sigma"] == 1.0) & (matrix["tau_truth"] == 0.95)
    dataframe_table(
        matrix[registered].round(4),
        title="How much less informative the sources would have to be (HalfNormal(1.0), tau = 0.95)",
    )
    dataframe_table(
        groups[
            (groups["tau_prior_sigma"] == 1.0) & (groups["se_scale"] == 1.0)
        ].drop(columns=["tau_prior_sigma", "se_scale"]).round(4),
        title="Per-study slope against the oracle-tau benchmark, at HalfNormal(1.0)",
    )

    out_dir = os.path.join(env.output_root(), "comparisons", "psi-hierarchy")
    os.makedirs(out_dir, exist_ok=True)
    matrix.to_csv(os.path.join(out_dir, "psi_hierarchy_matrix.csv"), index=False)
    groups.to_csv(os.path.join(out_dir, "psi_hierarchy_by_study.csv"), index=False)
    information.to_csv(os.path.join(out_dir, "psi_study_information.csv"), index=False)
    with open(os.path.join(out_dir, "settings.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args) | {"conc": conc, "log_psi_truth": log_psi_truth}, handle, indent=2)
    console.print(f"[dim]Written to {out_dir}[/dim]")
    return 0


def _prepared_frame(definition):
    """VG15's own analysis frame, through the engine's own preparation stage."""
    import dse_research_utils.statistics.models.reporting as model_reporting
    import dse_research_utils.statistics.models.sampling as sampling

    from vocab_growth.models.common_joint_modality import (
        JointContext,
        prepare_joint_data,
    )

    reporting = model_reporting.ReportingConfiguration(
        model_name=definition.model_id,
        config_name=f"{definition.config_name}-psi-hierarchy",
        output_root_dir=env.output_root(),
        ci_prob=0.89,
        interval_kind="eti",
    )
    os.makedirs(reporting.output_dir, exist_ok=True)
    context = JointContext(
        reporting=reporting,
        sampling=sampling.get_sampling_configuration("dev"),
        sampling_config_name="dev",
    )
    prepare_joint_data(context, definition)
    return context.analysis_df


if __name__ == "__main__":
    raise SystemExit(main())
