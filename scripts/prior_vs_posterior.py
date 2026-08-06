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
from vocab_growth.fit_artifacts import (
    FIT_MANIFEST_FILENAME,
    FitValidationError,
    normalise_for_json,
    read_json,
)
from vocab_growth.models.definitions import (
    MODEL_REGISTRY,
    BivariateModelDefinition,
    JointModelDefinition,
    KappaAnchorPriorParams,
    TrivariateModelDefinition,
    UnivariateModelDefinition,
)

MODELS_DIR = env.models_output_dir()

# Registry-derived: every registered model. Trivariate (VG14) and joint (VG15)
# were excluded until 2026-08-06 because their signed-ratio, psi and
# concentration priors were not reconstructed here. That exclusion hid a real
# defect -- VG14's `b_kappa_mag_s` sits 4 sigma beyond its prior, on a parameter
# the sweep could already build. `signing_priors` closes the gap; see
# `model_priors` for the dispatch.
MODEL_LABELS = {
    d.model_id: (f"{d.model_id}-{d.config_name}", d)
    for d in MODEL_REGISTRY.values()
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
    priors = {
        "ell_unit": pz.Beta(alpha=d.ell_unit_alpha, beta=d.ell_unit_beta),
        "eta": pz.HalfNormal(sigma=d.eta_sigma),
        "p_slope_low": pz.Beta(alpha=d.p_slope_low_alpha, beta=d.p_slope_low_beta),
        "p_slope_hi": pz.Beta(alpha=d.p_slope_hi_alpha, beta=d.p_slope_hi_beta),
        **kappa_priors(d.kappa),
    }
    # The random-effect models carry scale priors the base univariate models do
    # not. Omitting them left `tau` — VG12's worst-mixing parameter before the
    # study block was centred — with no prior check at all.
    if getattr(d, "tau_study_sigma", None) is not None and getattr(
        d, "min_study_observations", None
    ) is not None:
        priors["tau"] = pz.HalfNormal(sigma=d.tau_study_sigma)
    partition = getattr(d, "subject_variance_partition", None)
    if partition is not None:
        # Under the variance partition `tau_subject` is a Deterministic with no
        # prior of its own; the budget and the split are what carry one. Checking
        # the derived quantity against the prior it no longer has would be wrong.
        priors["v_total"] = pz.LogNormal(mu=partition.total_mu, sigma=partition.total_sigma)
        priors["subject_variance_share"] = pz.Beta(
            alpha=partition.share_alpha, beta=partition.share_beta
        )
    elif getattr(d, "use_subject_re", False):
        priors["tau_subject"] = pz.HalfNormal(sigma=d.tau_subject_sigma)
    return priors


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
        # Dispatched per outcome, because a definition may carry a different
        # dispersion form on each: `validate_kappa_fields` checks the two blocks
        # independently and permits a mixed pair. Every registered bivariate
        # model happens to anchor both, so calling `kappa_priors` once would
        # currently give the same answer — but it would be right by coincidence,
        # and would start mis-plotting the moment one outcome migrated alone.
        **kappa_priors(d.kappa_u, "_u"),
        **kappa_priors(d.kappa_s, "_s"),
    }
    if getattr(d, "tau_u_sigma", None) is not None:
        priors["tau_u"] = pz.HalfNormal(sigma=d.tau_u_sigma)
        priors["tau_q"] = pz.HalfNormal(sigma=d.tau_q_sigma)
    if getattr(d, "use_subject_re_u", False):
        priors["tau_subj_u"] = pz.HalfNormal(sigma=d.tau_subj_u_sigma)
    if getattr(d, "use_subject_re_q", False):
        priors["tau_subj_q"] = pz.HalfNormal(sigma=d.tau_subj_q_sigma)
    return priors


def signing_priors(d) -> dict[str, pz.distributions.distributions.Continuous]:
    """The extra priors the signing models carry beyond the bivariate set.

    VG14 and VG15 add a third outcome with its own GP and a **three-anchor** mean
    (low / mid / high) rather than the two-anchor form the other outcomes use, and
    VG15 adds the sign-speech association and the Dirichlet-Multinomial
    concentration. These were the reason the two models were excluded from the
    sweep entirely — which is how VG14's 4-sigma `b_kappa_mag_s` conflict went
    unseen, since that parameter is part of the *shared* bivariate set the sweep
    could already reconstruct.
    """
    priors: dict = {
        "ell_unit_sign": pz.Beta(alpha=d.ell_unit_sign_alpha, beta=d.ell_unit_sign_beta),
        "eta_sign": pz.HalfNormal(sigma=d.eta_sign_sigma),
        "p_slope_low_sign": pz.Beta(
            alpha=d.p_slope_low_sign_alpha, beta=d.p_slope_low_sign_beta
        ),
        "p_slope_mid_sign": pz.Beta(
            alpha=d.p_slope_mid_sign_alpha, beta=d.p_slope_mid_sign_beta
        ),
        "p_slope_hi_sign": pz.Beta(
            alpha=d.p_slope_hi_sign_alpha, beta=d.p_slope_hi_sign_beta
        ),
        **kappa_priors(d.kappa_sign, "_sign"),
    }
    # VG15 only: the association and the concentration are given on the log scale,
    # and the trace stores `log_psi` / `log_conc` under those names.
    if getattr(d, "log_psi_sigma", None) is not None:
        priors["log_psi"] = pz.Normal(mu=d.log_psi_mu, sigma=d.log_psi_sigma)
    if getattr(d, "log_conc_sigma", None) is not None:
        priors["log_conc"] = pz.Normal(mu=d.log_conc_mu, sigma=d.log_conc_sigma)
    if getattr(d, "tau_sign_sigma", None) is not None:
        priors["tau_sign"] = pz.HalfNormal(sigma=d.tau_sign_sigma)
    if getattr(d, "use_subject_re_sign", False):
        priors["tau_subj_sign"] = pz.HalfNormal(sigma=d.tau_subj_sign_sigma)
    return priors


def model_priors(d) -> dict[str, pz.distributions.distributions.Continuous]:
    """Every prior this definition specifies, whichever family it belongs to."""
    if isinstance(d, (TrivariateModelDefinition, JointModelDefinition)):
        # Both carry the full bivariate understood/spoken block plus a sign block.
        return {**bivariate_priors(d), **signing_priors(d)}
    if isinstance(d, BivariateModelDefinition):
        return bivariate_priors(d)
    return univariate_priors(d)


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


CONFLICT_CDF = 0.95
"""Prior CDF at the posterior mean above which a parameter counts as pressing."""
CONTRACTION_FLOOR = 0.05
"""Contraction below which the posterior is essentially reporting the prior back."""


def conflict_table(short: str, label: str, definition) -> list[dict]:
    """Prior-data conflict diagnostics for one model — review R3.

    Two numbers per parameter. **Prior CDF** at the posterior mean says where the
    data landed inside the prior: near 1 means the prior is a ceiling the
    likelihood is pushing against. **Contraction**, ``1 - posterior sd / prior
    sd``, says how much the data actually informed it: at or below zero the
    posterior is no narrower than the prior, so the reported value is the prior
    restated rather than an estimate.

    Both matter and neither is sufficient. VG12's `eta` was flagged on the pair
    (CDF 0.913, contraction 0.106) — pressing *and* uninformed. VG13's `eta_q`
    sits mid-prior but with contraction below zero, which the CDF alone would
    have passed. See notes/202608050900-td-hierarchical-geometry.md §5.
    """
    trace_path = os.path.join(MODELS_DIR, label, "trace.nc")
    if not os.path.isfile(trace_path):
        return []
    # A trace fitted under a different definition must not be scored against the
    # current priors: the result looks like a prior-data conflict but is only a
    # mismatch. This bit the sweep's own first run — VG12's eta showed prior CDF
    # 0.991 with contraction -0.670, which was an eta=1.0 posterior being read
    # against the eta=0.5 prior it had just been reverted to.
    manifest_path = os.path.join(MODELS_DIR, label, FIT_MANIFEST_FILENAME)
    if os.path.isfile(manifest_path):
        try:
            stored = read_json(manifest_path).get("model", {}).get("definition")
        except FitValidationError:
            stored = None
        if stored is not None and stored != normalise_for_json(definition):
            print(f"  {short}: SKIPPED — trace predates the current definition (refit needed)")
            return []
    idata = az.from_netcdf(trace_path)
    priors = model_priors(definition)
    rows = []
    for name, prior in priors.items():
        if name not in idata.posterior.data_vars:
            continue
        x = np.asarray(idata.posterior[name].values).reshape(-1)
        try:
            prior_sd = float(np.sqrt(prior.var()))
            cdf = float(prior.cdf(float(x.mean())))
        except Exception:
            continue
        contraction = 1.0 - float(x.std()) / prior_sd if prior_sd > 0 else float("nan")
        flags = []
        if cdf >= CONFLICT_CDF:
            flags.append("pressing")
        if contraction <= CONTRACTION_FLOOR:
            flags.append("uninformed")
        rows.append(
            dict(
                model=short,
                parameter=name,
                posterior_mean=round(float(x.mean()), 4),
                posterior_sd=round(float(x.std()), 4),
                prior_cdf=round(cdf, 4),
                contraction=round(contraction, 4),
                flags="+".join(flags),
            )
        )
    return rows


def write_conflict_table() -> None:
    import csv

    rows = []
    for short, (label, definition) in MODEL_LABELS.items():
        rows.extend(conflict_table(short, label, definition))
    if not rows:
        print("No fitted traces found.")
        return
    out_dir = env.comparisons_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "prior_posterior_conflict.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    flagged = [r for r in rows if r["flags"]]
    print(f"{len(rows)} parameters checked across {len(MODEL_LABELS)} models -> {path}")
    print(f"\n{len(flagged)} FLAGGED:\n")
    print(f"  {'model':7s} {'parameter':24s} {'post mean':>10s} {'priorCDF':>9s} {'contract':>9s}  flags")
    for r in sorted(flagged, key=lambda r: (-r["prior_cdf"], r["contraction"])):
        print(
            f"  {r['model']:7s} {r['parameter']:24s} {r['posterior_mean']:10.4f} "
            f"{r['prior_cdf']:9.3f} {r['contraction']:9.3f}  {r['flags']}"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        action="store_true",
        help="Emit the prior-data conflict table (review R3) instead of the plots.",
    )
    args = parser.parse_args()
    if args.table:
        write_conflict_table()
        return
    plot_styles.set_matplotlib_default_style()
    for short, (label, definition) in MODEL_LABELS.items():
        overlay_model(short, label, definition)


if __name__ == "__main__":
    main()
