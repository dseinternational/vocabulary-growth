# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Simulated figures for the report's introduction and model structure.

``write_intro_illustrations`` shows Bayesian updating and count dispersion.
``write_prior_illustrations`` shows VG01's anchor-defined trend and VG10's
per-draw GP constraint. ``scripts/prepare_report_figures.py`` writes these
illustrations into the report cache; they do not use fitted output.

Anchor priors, GP amplitudes and the VG10 clamp setting come from registered
model definitions. GP length-scale settings and plotting domains are local
constants. Ages are handled in months, and random effects are set to zero.
These illustrations draw from the GP kernel directly; fitted models use its
finite HSGP approximation, so the simulated curves are not exact draws from
those fitted model graphs.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import beta as beta_dist
from scipy.stats import binom

from vocab_growth.data_utils import load_combined_data
from vocab_growth.intervals import DEFAULT_CI_PROB, INNER_CI_PROB
from vocab_growth.models.build_utils import CLAMP_SOFTNESS
from vocab_growth.models.definitions import MODEL_REGISTRY

RANDOM_SEED = 47

N_TRIALS = 810
GP_DOMAIN_MONTHS = (8.0, 115.0)
ELL_MONTHS_RANGE = (6.0, 18.0)
ELL_UNIT = (3.0, 3.0)

C_DRAWS = "#E8863B"
C_HIGHLIGHT = "#B4450E"
C_ANCHOR = "#0F447A"
C_TREND = "#111111"


def _save(fig, out_dir: str, filename: str) -> str:
    """Write ``filename`` as PNG and SVG into ``out_dir`` and return the stem."""
    os.makedirs(out_dir, exist_ok=True)
    for ext, kw in (("png", {"dpi": 300}), ("svg", {})):
        fig.savefig(os.path.join(out_dir, f"{filename}.{ext}"),
                    bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"Wrote {filename}.png/.svg to {out_dir}")
    return filename


# --------------------------------------------------------------------------- #
# Introduction: Bayesian updating and count dispersion
# --------------------------------------------------------------------------- #
def plot_bayes_update(out_dir, n, rng, true_p, alpha0, beta0, filename):
    """A Beta(alpha0, beta0) prior updated by ``n`` simulated Bernoulli trials."""
    data = rng.binomial(n=1, p=true_p, size=n)
    k = int(data.sum())

    print(f"Simulated data: {k} successes out of {n} trials (true p = {true_p:.2f})")

    alpha_post = alpha0 + k
    beta_post = beta0 + (n - k)

    p_grid = np.linspace(0.0, 1.0, 1000)

    prior_pdf = beta_dist.pdf(p_grid, alpha0, beta0)
    post_pdf = beta_dist.pdf(p_grid, alpha_post, beta_post)

    likelihood = binom.pmf(k, n, p_grid)
    likelihood_norm = likelihood / np.trapezoid(likelihood, p_grid)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(p_grid, prior_pdf, label="Prior", c="#0080e0")
    ax.plot(
        p_grid, likelihood_norm, label=f"Observed ($n = {n}$, $k = {k}$)", c="#009900"
    )
    ax.plot(p_grid, post_pdf, label="Posterior", c="#ff7800")

    ax.axvline(
        k / n,
        linestyle="--",
        linewidth=1,
        c="#009900",
        label=f"Observed mean = {k/n:.2f}",
    )
    ax.axvline(
        true_p,
        linestyle=":",
        linewidth=1,
        c="#ff0033",
        label=f"True $p$ (simulated) = {true_p:.2f}",
    )

    prior_mean = alpha0 / (alpha0 + beta0)
    ax.axvline(
        prior_mean,
        linestyle="--",
        linewidth=1,
        c="#0080e0",
        label=f"Prior mean = {prior_mean:.2f}",
    )

    posterior_mean = alpha_post / (alpha_post + beta_post)
    ax.axvline(
        posterior_mean,
        linestyle="-.",
        linewidth=1,
        c="#ff7800",
        label=f"Posterior mean = {posterior_mean:.2f}",
    )

    ax.set_xlabel("$p$")
    ax.set_ylabel("Density / scaled likelihood")

    ax.set_ylim(0, 9)

    ax.legend()

    pd.DataFrame({
        "p": p_grid,
        "prior_pdf": prior_pdf,
        "likelihood_norm": likelihood_norm,
        "posterior_pdf": post_pdf,
    }).to_csv(os.path.join(out_dir, f"{filename}.csv"), index=False)
    return _save(fig, out_dir, filename)


def plot_count_dispersion(out_dir, rng, n, p, kappa, n_draws, filename):
    """Simulated counts under a Binomial and a mean-matched Beta-Binomial.

    Both panels share the mean np; the Beta-Binomial splits the concentration
    kappa as alpha = p * kappa, beta = (1 - p) * kappa, the parameterisation the
    models use, so the only difference on display is the dispersion.
    """
    alpha_bb = p * kappa
    beta_bb = (1.0 - p) * kappa

    y_binom = rng.binomial(n=n, p=p, size=n_draws)
    p_obs = rng.beta(alpha_bb, beta_bb, size=n_draws)
    y_betabinom = rng.binomial(n=n, p=p_obs)

    print(
        f"Binomial(n={n}, p={p}): mean {y_binom.mean():.1f}, sd {y_binom.std():.1f}; "
        f"Beta-Binomial(alpha={alpha_bb:g}, beta={beta_bb:g}): "
        f"mean {y_betabinom.mean():.1f}, sd {y_betabinom.std():.1f}"
    )

    bins = np.arange(0, n + 16, 15)
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    for ax, draws, label, colour in (
        (axes[0], y_binom, f"Binomial($n = {n}$, $p = {p}$)", "#0080e0"),
        (
            axes[1],
            y_betabinom,
            f"Beta-Binomial($n = {n}$, $p = {p}$, $\\kappa = {kappa:g}$)",
            "#ff7800",
        ),
    ):
        ax.hist(draws, bins=bins, color=colour, label=label)
        ax.axvline(
            n * p,
            linestyle="--",
            linewidth=1,
            c="#555555",
            label=f"Mean $np = {n * p:.0f}$",
        )
        ax.set_ylabel(f"Draws (of {n_draws})")
        ax.legend()

    axes[1].set_xlabel(f"Count of words checked (out of $n = {n}$)")
    axes[1].set_xlim(0, n)

    pd.DataFrame({
        "binomial": y_binom,
        "beta_binomial": y_betabinom,
    }).to_csv(os.path.join(out_dir, f"{filename}.csv"), index=False)
    return _save(fig, out_dir, filename)


def write_intro_illustrations(out_dir: str, seed: int = RANDOM_SEED) -> list[str]:
    """Write the introduction's four simulated figures; returns their stems.

    One generator is threaded through all four in this order, so the figures
    are reproducible as a set from ``seed``.
    """
    rng = np.random.default_rng(seed)
    true_p = 0.65
    alpha0, beta0 = 2.0, 2.0
    return [
        plot_bayes_update(out_dir, 20, rng, true_p, alpha0, beta0, "bayes_update"),
        plot_bayes_update(out_dir, 100, rng, true_p, alpha0, beta0, "bayes_update_2"),
        plot_bayes_update(out_dir, 250, rng, true_p, alpha0, beta0, "bayes_update_3"),
        plot_count_dispersion(
            out_dir, rng, n=810, p=0.3, kappa=4.0, n_draws=1000,
            filename="binomial_betabinomial_draws",
        ),
    ]


# --------------------------------------------------------------------------- #
# Model structure: shared helpers, in months
# --------------------------------------------------------------------------- #
def _definition(model_id):
    """Read anchor priors and GP amplitudes from the registered definition."""
    return MODEL_REGISTRY[model_id.lower()]


def _soft_clamp(ages, anchors):
    """``gp_utils._soft_clamp_z`` in months: linear below the high anchor, flat above."""
    a_lo, a_hi = anchors
    beta = CLAMP_SOFTNESS / (a_hi - a_lo)
    return a_hi - np.logaddexp(0.0, beta * (a_hi - ages)) / beta


def _interval(dist, prob):
    lo = (1.0 - prob) / 2.0
    return dist.ppf(lo), dist.ppf(1.0 - lo)


def _gp_draw(rng, ages, ell):
    d = ages[:, None] - ages[None, :]
    k = np.exp(-(d**2) / (2.0 * ell**2)) + 1e-8 * np.eye(len(ages))
    return np.linalg.cholesky(k) @ rng.standard_normal(len(ages))


def _draw_ell_eta(rng, eta_sigma):
    ell = ELL_MONTHS_RANGE[0] + np.ptp(ELL_MONTHS_RANGE) * rng.beta(*ELL_UNIT)
    return ell, abs(rng.normal(0.0, eta_sigma))


def _trend_logit(ages, p_lo, p_hi, anchors):
    a_lo, a_hi = anchors
    slope = (logit(p_hi) - logit(p_lo)) / (a_hi - a_lo)
    return logit(p_lo) + slope * (ages - a_lo)


def _draw_anchor_priors(ax, anchors, dists, callouts=None):
    """Median marker with nested 50% / 89% interval bars at each anchor age."""
    for age, dist in zip(anchors, dists, strict=True):
        o_lo, o_hi = _interval(dist, DEFAULT_CI_PROB)
        i_lo, i_hi = _interval(dist, INNER_CI_PROB)
        ax.plot([age, age], [N_TRIALS * o_lo, N_TRIALS * o_hi],
                color=C_ANCHOR, lw=2.2, solid_capstyle="butt", zorder=4)
        ax.plot([age, age], [N_TRIALS * i_lo, N_TRIALS * i_hi],
                color=C_ANCHOR, lw=6.0, solid_capstyle="butt", zorder=4)
        ax.plot([age], [N_TRIALS * dist.ppf(0.5)], "o", color="white", markersize=8,
                markeredgecolor=C_ANCHOR, markeredgewidth=2.0, zorder=5)
    if not callouts:
        return
    for age, dist, dx, dy, ha in callouts:
        o_lo, o_hi = _interval(dist, DEFAULT_CI_PROB)
        med = N_TRIALS * dist.ppf(0.5)
        ax.annotate(
            f"{age:.0f} mo anchor\nmedian {med:.0f} words\n"
            f"{DEFAULT_CI_PROB:.0%}: {N_TRIALS * o_lo:.0f}–{N_TRIALS * o_hi:.0f}",
            xy=(age, med), xytext=(age + dx, med + dy), fontsize=8.5,
            color=C_ANCHOR, ha=ha, va="center", zorder=6,
            arrowprops=dict(arrowstyle="-", color=C_ANCHOR, lw=0.9, alpha=0.6),
        )


# --------------------------------------------------------------------------- #
# Figure 1: VG01 prior structure
# --------------------------------------------------------------------------- #
def build_prior_structure(out_dir, filename="model_structure_prior_vg01",
                          n_draws=250, n_grid=260, seed=RANDOM_SEED):
    """Prior trajectories, anchors and one draw's GP departure, for VG01.

    No title: the report's figure caption carries it.
    """
    d = _definition("vg01")
    anchors = tuple(float(a) for a in d.slope_anchors)
    d_lo = beta_dist(d.p_slope_low_alpha, d.p_slope_low_beta)
    d_hi = beta_dist(d.p_slope_hi_alpha, d.p_slope_hi_beta)

    rng = np.random.default_rng(seed)
    ages = np.linspace(*GP_DOMAIN_MONTHS, n_grid)
    counts = np.empty((n_draws, len(ages)))
    trends = np.empty((n_draws, len(ages)))
    for i in range(n_draws):
        trend = _trend_logit(ages, rng.beta(d.p_slope_low_alpha, d.p_slope_low_beta),
                             rng.beta(d.p_slope_hi_alpha, d.p_slope_hi_beta), anchors)
        ell, eta = _draw_ell_eta(rng, d.eta_sigma)
        counts[i] = N_TRIALS * expit(trend + eta * _gp_draw(rng, ages, ell))
        trends[i] = N_TRIALS * expit(trend)

    median_trend = N_TRIALS * expit(
        _trend_logit(ages, d_lo.ppf(0.5), d_hi.ppf(0.5), anchors)
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    for a in anchors:
        ax.axvline(a, color="0.85", lw=1.0, zorder=0)
    ax.plot(ages, counts.T, color=C_DRAWS, alpha=0.16, lw=0.8, zorder=1)
    ax.plot([], [], color=C_DRAWS, alpha=0.6, lw=1.2,
            label=f"Prior trajectories ({n_draws} draws)")

    # Highlight a draw near the median trend at the high anchor. Prefer monotone
    # curves, then choose the largest GP departure among eligible draws. This is
    # a selected illustration, not a representative random draw.
    k = int(np.argmin(np.abs(ages - anchors[1])))
    near = np.abs(trends[:, k] - median_trend[k]) < 0.12 * median_trend[k]
    mono = np.all(np.diff(counts, axis=1) >= -1e-9, axis=1)
    eligible = (near & mono) if (near & mono).any() else near
    j = int(np.argmax(np.where(eligible, np.abs(counts - trends).max(axis=1), -np.inf)))

    ax.fill_between(ages, trends[j], counts[j], color=C_HIGHLIGHT, alpha=0.16,
                    lw=0, zorder=2)
    ax.plot(ages, trends[j], color=C_HIGHLIGHT, lw=1.6, ls=(0, (5, 2)), zorder=3,
            label="One draw: its logit-linear trend")
    ax.plot(ages, counts[j], color=C_HIGHLIGHT, lw=2.0, zorder=3,
            label="The same draw, after the GP")

    _draw_anchor_priors(
        ax, anchors, (d_lo, d_hi),
        callouts=[(anchors[0], d_lo, 5, 150, "left"),
                  (anchors[1], d_hi, 6, -200, "left")],
    )
    ax.plot([], [], color=C_ANCHOR, lw=2.2,
            label=f"Anchor priors: median, {INNER_CI_PROB:.0%} and {DEFAULT_CI_PROB:.0%} intervals")
    ax.plot(ages, median_trend, color=C_TREND, lw=2.0, ls=(0, (1.5, 2.5)), zorder=6,
            label="Median trend joining the anchors")

    ax.set_xlim(*GP_DOMAIN_MONTHS)
    ax.set_ylim(-15, N_TRIALS + 15)
    ax.set_xlabel("Age (months)")
    ax.set_ylabel(f"Words spoken (of {N_TRIALS})")
    ax.legend(loc="upper left", frameon=True, fontsize="small", framealpha=0.95)
    ax.grid(True, color="0.92", lw=0.8)
    ax.set_axisbelow(True)
    return _save(fig, out_dir, filename)


# --------------------------------------------------------------------------- #
# Figure 2: VG10 per-draw GP anchoring
# --------------------------------------------------------------------------- #
def _observed_ages():
    """Ages of the VG10 analysis frame, with multiplicity, for the projection."""
    df = load_combined_data()
    frame = df[df["understood"].notna() | df["spoken"].notna()]
    return frame["age"].to_numpy(dtype=float)


def build_gp_anchoring(out_dir, filename="gp_anchoring_vg10",
                       n_draws=250, n_grid=240, seed=RANDOM_SEED):
    """The same prior draws with and without VG10's per-draw GP anchor."""
    d = _definition("vg10")
    anchors = tuple(float(a) for a in d.slope_anchors)
    ref = float(d.gp_anchor_age_months)
    d_lo = beta_dist(d.p_slope_low_u_alpha, d.p_slope_low_u_beta)
    d_hi = beta_dist(d.p_slope_hi_u_alpha, d.p_slope_hi_u_beta)

    rng = np.random.default_rng(seed)
    # Include the reference age exactly to show the per-draw zero constraint.
    plot_ages = np.unique(
        np.concatenate([np.linspace(*GP_DOMAIN_MONTHS, n_grid), [ref]])
    )
    obs_ages = _observed_ages()

    # One GP realisation per draw over the union of plot, observed and anchor
    # ages, as in the model's stacked X_all grid.
    grid = np.unique(np.concatenate([plot_ages, obs_ages, [ref]]))
    i_plot = np.searchsorted(grid, plot_ages)
    i_obs = np.searchsorted(grid, obs_ages)
    i_ref = int(np.searchsorted(grid, ref))

    # Match the definition's choice of whether to flatten the trend.
    if d.clamp_mean_above_hi_anchor:
        eff_plot, eff_grid = _soft_clamp(plot_ages, anchors), _soft_clamp(grid, anchors)
    else:
        eff_plot, eff_grid = plot_ages, grid
    # Projection basis [1, a_eff]; coefficients are fitted on the observed rows
    # only, so repeated ages carry their true weight, exactly as the engine does.
    B_obs = np.column_stack([np.ones(len(i_obs)), eff_grid[i_obs]])
    B_grid = np.column_stack([np.ones(len(grid)), eff_grid])
    gram = B_obs.T @ B_obs + 1e-6 * np.eye(2)

    trend = np.empty((n_draws, len(plot_ages)))
    g_free = np.empty((n_draws, len(plot_ages)))
    g_anch = np.empty((n_draws, len(plot_ages)))
    for i in range(n_draws):
        p_lo = rng.beta(d.p_slope_low_u_alpha, d.p_slope_low_u_beta)
        p_hi = rng.beta(d.p_slope_hi_u_alpha, d.p_slope_hi_u_beta)
        trend[i] = _trend_logit(eff_plot, p_lo, p_hi, anchors)
        ell, eta = _draw_ell_eta(rng, d.eta_u_sigma)
        g_unit = _gp_draw(rng, grid, ell)
        g_free[i] = eta * g_unit[i_plot]
        resid = g_unit - B_grid @ np.linalg.solve(gram, B_obs.T @ g_unit[i_obs])
        g_anch[i] = eta * (resid[i_plot] - resid[i_ref])

    median_trend = N_TRIALS * expit(
        _trend_logit(eff_plot, d_lo.ppf(0.5), d_hi.ppf(0.5), anchors)
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1]})
    titles = ("GP free (VG09)", f"GP anchored at {ref:.0f} months (VG10)")
    for col, (g, title) in enumerate(zip((g_free, g_anch), titles, strict=True)):
        ax = axes[0, col]
        ax.plot(plot_ages, (N_TRIALS * expit(trend + g)).T,
                color=C_DRAWS, alpha=0.14, lw=0.8)
        ax.plot(plot_ages, median_trend, color=C_TREND, lw=1.8, ls=(0, (1.5, 2.5)),
                label="Median trend")
        _draw_anchor_priors(ax, anchors, (d_lo, d_hi))
        ax.set_ylim(-15, N_TRIALS + 15)
        ax.set_title(title, fontsize=12)
        if col == 0:
            ax.set_ylabel(f"Words understood (of {N_TRIALS})")
            ax.legend(loc="upper left", fontsize="small", frameon=True)

        ax = axes[1, col]
        ax.axhline(0.0, color="0.45", lw=1.0)
        ax.plot(plot_ages, g.T, color=C_DRAWS, alpha=0.14, lw=0.8)
        ax.set_ylim(-4.2, 4.2)
        ax.set_xlabel("Age (months)")
        if col == 0:
            ax.set_ylabel("GP contribution (logits)")

        for a in axes[:, col]:
            a.axvline(ref, color=C_HIGHLIGHT, lw=1.2, ls=(0, (4, 3)), zorder=0)
            a.grid(True, color="0.93", lw=0.7)
            a.set_axisbelow(True)
            a.set_xlim(*GP_DOMAIN_MONTHS)

    axes[1, 1].annotate("every draw passes\nthrough zero here",
                        xy=(ref, 0.0), xytext=(ref + 12, 2.9), fontsize=9,
                        color=C_HIGHLIGHT,
                        arrowprops=dict(arrowstyle="->", color=C_HIGHLIGHT, lw=1.1))
    fig.suptitle("Per-draw GP anchoring: identical prior draws, with and without the constraint",
                 fontsize=13)
    fig.tight_layout()

    k = int(np.argmin(np.abs(plot_ages - ref)))
    print(f"  GP contribution at {ref:.0f} mo — free sd {g_free[:, k].std():.3f} logits; "
          f"anchored max |g| {np.abs(g_anch[:, k]).max():.2e}")
    return _save(fig, out_dir, filename)


PRIOR_FIGURES = {"structure": build_prior_structure, "anchoring": build_gp_anchoring}


def write_prior_illustrations(
    out_dir: str,
    figures: list[str] | None = None,
    n_draws: int = 250,
    seed: int = RANDOM_SEED,
) -> list[str]:
    """Write the model-structure figures (all of :data:`PRIOR_FIGURES` by default)."""
    names = list(PRIOR_FIGURES) if figures is None else list(figures)
    return [PRIOR_FIGURES[name](out_dir, n_draws=n_draws, seed=seed) for name in names]
