# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Per-model prior-vs-posterior overlay plots for the headline parameters.

For each model, this rebuilds the prior distributions analytically from
the model definition (using `preliz` for analytic PDF evaluation) and
overlays them with a kernel-density estimate of the posterior samples
drawn from the saved trace. This is the standard Bayesian-workflow
figure showing how much each prior was updated by the data.

Outputs (under each `output/models/<MODEL>/`):

- `prior_vs_posterior.png` / `.svg` — multi-panel figure (one panel per
  headline parameter).
"""

from __future__ import annotations

import os

import arviz as az
import dse_research_utils.plot.styles as plot_styles
import matplotlib.pyplot as plt
import numpy as np
import preliz as pz
from scipy import stats

from vocab_growth import environment as env
from vocab_growth.models.definitions import (
    MODEL_REGISTRY,
    BivariateModelDefinition,
    KappaAnchorPriorParams,
    UnivariateModelDefinition,
)

MODELS_DIR = env.models_output_dir()

# Registry-derived: covers every univariate/bivariate model, matching the two
# prior-dispatch functions below. Trivariate (VG14) and joint (VG15)
# definitions carry extra priors (signed ratio, psi, conc) this script does
# not build, so they are intentionally excluded rather than silently
# under-plotted.
MODEL_LABELS = {
    d.model_id: (f"{d.model_id}-{d.config_name}", d)
    for d in MODEL_REGISTRY.values()
    if isinstance(d, (UnivariateModelDefinition, BivariateModelDefinition))
}


def kappa_priors(kp, suffix: str = "") -> dict[str, pz.distributions.distributions.Continuous]:
    """Priors for whichever kappa parameterisation `kp` is.

    Only the *free* parameters appear. Under the two-anchor form ``a_kappa`` and
    ``b_kappa`` are derived, so they are in the trace but have no prior to plot
    against — including them would invite a comparison against a distribution
    that was never specified.
    """
    priors = {
        f"kappa_min{suffix}": pz.LogNormal(mu=kp.kappa_min_mu, sigma=kp.kappa_min_sigma)
    }
    if isinstance(kp, KappaAnchorPriorParams):
        priors[f"kappa_excess_young{suffix}"] = pz.LogNormal(
            mu=kp.excess_young_mu, sigma=kp.excess_young_sigma
        )
        priors[f"kappa_excess_old{suffix}"] = pz.LogNormal(
            mu=kp.excess_old_mu, sigma=kp.excess_old_sigma
        )
    else:
        priors[f"a_kappa{suffix}"] = pz.Normal(mu=kp.a_kappa_mu, sigma=kp.a_kappa_sigma)
        priors[f"b_kappa_mag{suffix}"] = pz.HalfNormal(sigma=kp.b_kappa_mag_sigma)
    return priors


def univariate_priors(d: UnivariateModelDefinition) -> dict[str, pz.distributions.distributions.Continuous]:
    return {
        "ell_unit": pz.Beta(alpha=d.ell_unit_alpha, beta=d.ell_unit_beta),
        "eta": pz.HalfNormal(sigma=d.eta_sigma),
        "p_slope_low": pz.Beta(alpha=d.p_slope_low_alpha, beta=d.p_slope_low_beta),
        "p_slope_hi": pz.Beta(alpha=d.p_slope_hi_alpha, beta=d.p_slope_hi_beta),
        **kappa_priors(d.kappa),
    }


def bivariate_priors(d: BivariateModelDefinition) -> dict[str, pz.distributions.distributions.Continuous]:
    priors = {
        "ell_unit_u": pz.Beta(alpha=d.ell_unit_u_alpha, beta=d.ell_unit_u_beta),
        "eta_u": pz.HalfNormal(sigma=d.eta_u_sigma),
        "ell_unit_q": pz.Beta(alpha=d.ell_unit_q_alpha, beta=d.ell_unit_q_beta),
        "eta_q": pz.HalfNormal(sigma=d.eta_q_sigma),
        "p_slope_low_u": pz.Beta(alpha=d.p_slope_low_u_alpha, beta=d.p_slope_low_u_beta),
        "p_slope_hi_u": pz.Beta(alpha=d.p_slope_hi_u_alpha, beta=d.p_slope_hi_u_beta),
        "p_slope_low_q": pz.Beta(alpha=d.p_slope_low_q_alpha, beta=d.p_slope_low_q_beta),
        "p_slope_hi_q": pz.Beta(alpha=d.p_slope_hi_q_alpha, beta=d.p_slope_hi_q_beta),
        "kappa_min_u": pz.LogNormal(mu=d.kappa_u.kappa_min_mu, sigma=d.kappa_u.kappa_min_sigma),
        "kappa_min_s": pz.LogNormal(mu=d.kappa_s.kappa_min_mu, sigma=d.kappa_s.kappa_min_sigma),
    }
    # The bivariate models have not migrated to the two-anchor kappa form; if one
    # does, its a_kappa / b_kappa_mag rows here would silently become wrong.
    for name, kp in (("kappa_u", d.kappa_u), ("kappa_s", d.kappa_s)):
        if isinstance(kp, KappaAnchorPriorParams):
            raise NotImplementedError(
                f"{d.model_id}.{name} uses the two-anchor kappa form; extend "
                "bivariate_priors before plotting it."
            )
    if getattr(d, "tau_u_sigma", None) is not None:
        priors["tau_u"] = pz.HalfNormal(sigma=d.tau_u_sigma)
        priors["tau_q"] = pz.HalfNormal(sigma=d.tau_q_sigma)
    if getattr(d, "use_subject_re_u", False):
        priors["tau_subj_u"] = pz.HalfNormal(sigma=d.tau_subj_u_sigma)
    if getattr(d, "use_subject_re_q", False):
        priors["tau_subj_q"] = pz.HalfNormal(sigma=d.tau_subj_q_sigma)
    return priors


def _support_grid(prior, samples: np.ndarray) -> np.ndarray:
    """Reasonable x range covering both prior support and posterior mass."""
    posterior_lo = float(np.quantile(samples, 0.001))
    posterior_hi = float(np.quantile(samples, 0.999))
    # Prior support summary
    try:
        prior_lo = float(prior.ppf(1e-4))
        prior_hi = float(prior.ppf(1 - 1e-4))
    except Exception:
        prior_lo, prior_hi = posterior_lo, posterior_hi
    lo = min(prior_lo, posterior_lo)
    hi = max(prior_hi, posterior_hi)
    if not np.isfinite(lo):
        lo = posterior_lo
    if not np.isfinite(hi):
        hi = posterior_hi
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    return np.linspace(lo - pad, hi + pad, 400)


def _plot_panel(ax, name, prior, post_samples):
    grid = _support_grid(prior, post_samples)
    prior_pdf = prior.pdf(grid)
    kde = stats.gaussian_kde(post_samples)
    post_pdf = kde(grid)

    ax.fill_between(grid, prior_pdf, alpha=0.2,
                    color=plot_styles.COLOUR_BLUE)
    ax.plot(grid, prior_pdf, color=plot_styles.COLOUR_BLUE, lw=1.5, label="Prior")
    ax.fill_between(grid, post_pdf, alpha=0.3,
                    color=plot_styles.COLOUR_ORANGE)
    ax.plot(grid, post_pdf, color=plot_styles.COLOUR_ORANGE, lw=1.5,
            label="Posterior")
    ax.set_title(name)
    ax.set_yticks([])


def overlay_model(short: str, label: str,
                  definition) -> None:
    trace_path = os.path.join(MODELS_DIR, label, "trace.nc")
    if not os.path.exists(trace_path):
        print(f"  {short}: trace not found — skipped")
        return
    print(f"  {short}: loading trace …", flush=True)
    idata = az.from_netcdf(trace_path)
    post = idata.posterior

    if isinstance(definition, UnivariateModelDefinition):
        priors = univariate_priors(definition)
    else:
        priors = bivariate_priors(definition)

    n = len(priors)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(plot_styles.FIGSIZE_XL[0] * 1.2,
                 plot_styles.FIGSIZE_XL[1] * (nrows / 3.0 + 0.1)),
        constrained_layout=True,
    )
    axes_flat = list(axes.flat) if nrows * ncols > 1 else [axes]

    for ax, (name, prior) in zip(axes_flat, priors.items(), strict=False):
        if name not in post.data_vars:
            ax.set_visible(False)
            continue
        samples = post[name].values.reshape(-1)
        if not np.all(np.isfinite(samples)):
            samples = samples[np.isfinite(samples)]
        _plot_panel(ax, name, prior, samples)

    for ax in axes_flat[len(priors):]:
        ax.set_visible(False)

    axes_flat[0].legend(loc="upper right", frameon=True, fontsize=8)
    fig.suptitle(f"{short} — prior vs posterior", fontweight="bold")

    out_dir = os.path.join(MODELS_DIR, label)
    fig.savefig(os.path.join(out_dir, "prior_vs_posterior.png"))
    fig.savefig(os.path.join(out_dir, "prior_vs_posterior.svg"))
    plt.close(fig)
    print(f"  {short}: wrote prior_vs_posterior.{{png,svg}}")


def main() -> None:
    plot_styles.set_matplotlib_default_style()
    for short, (label, definition) in MODEL_LABELS.items():
        overlay_model(short, label, definition)


if __name__ == "__main__":
    main()
