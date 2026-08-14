# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generate the methods chapter's prior-structure illustrations.

Two figures, both simulated from the registered model definitions' own priors —
no fitted output is read, so they can be regenerated at any time:

``model_structure_prior_vg01``
    VG01's prior trajectories with the two slope anchors, their 50% and 89%
    intervals, and the median trend joining them. One draw is shown against its
    own trend so the Gaussian-process contribution is a visible gap rather than
    something inferred from the spread of the bundle. Supports @sec-mean.

``gp_anchoring_vg10``
    VG10's per-draw GP anchor at 54 months, as a like-for-like comparison: both
    columns share one set of draws and differ only in whether the GP is
    orthogonalised and pinned. Supports @sec-gpanchor.

Both are written into the report figure cache
(``docs/report/figures/methods/``), which is gitignored and rebuilt on demand,
alongside the descriptives written by ``generate_descriptive_report.py``. They
are not produced by ``sync_report_figures.py`` because they are not fit
artefacts and have no model output to validate against.

    python scripts/generate_prior_figures.py            # both
    python scripts/generate_prior_figures.py structure  # just VG01
    python scripts/generate_prior_figures.py anchoring  # just VG10

Ages are handled in months rather than standardised units throughout. The models
standardise age before building the kernel, the trend and the soft clamp, and
the standardisation cancels in all three, so the simulated prior is identical
and the plotted axis stays readable. Population trajectories set every random
effect to zero, following @sec-randomeffects.
"""

import argparse
import os

import dse_research_utils.environment.setup as setup
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, logit
from scipy.stats import beta as beta_dist

import vocab_growth.environment as local_env
from vocab_growth.data_utils import load_combined_data
from vocab_growth.intervals import DEFAULT_CI_PROB, INNER_CI_PROB
from vocab_growth.models.definitions import MODEL_REGISTRY

N_TRIALS = 810
GP_DOMAIN_MONTHS = (8.0, 115.0)
ELL_MONTHS_RANGE = (6.0, 18.0)
ELL_UNIT = (3.0, 3.0)
CLAMP_SOFTNESS = 50.0  # gp_utils._CLAMP_SOFTNESS

C_DRAWS = "#E8863B"
C_HIGHLIGHT = "#B4450E"
C_ANCHOR = "#0F447A"
C_TREND = "#111111"

RANDOM_SEED = 47


def _definition(model_id):
    """The registered definition, so the priors here cannot drift from the models."""
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


def _save(fig, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    for ext, kw in (("png", {"dpi": 300}), ("svg", {})):
        fig.savefig(os.path.join(out_dir, f"{filename}.{ext}"),
                    bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"Wrote {filename}.png/.svg to {out_dir}")


# --------------------------------------------------------------------------- #
# Figure 1 — VG01 prior structure
# --------------------------------------------------------------------------- #
def build_structure(out_dir, filename="model_structure_prior_vg01",
                    n_draws=250, n_grid=260, seed=RANDOM_SEED):
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

    # One draw against its own trend. Restricted to draws whose trend sits near
    # the median trend at the high anchor (so the gap is the GP, not a different
    # pair of anchors) and whose result is monotone -- an example chosen to be
    # legible, not the prior's worst case -- then the largest remaining excursion.
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
    ax.set_title("VG01 prior: anchored logit-linear trend with Gaussian-process departures")
    ax.legend(loc="upper left", frameon=True, fontsize="small", framealpha=0.95)
    ax.grid(True, color="0.92", lw=0.8)
    ax.set_axisbelow(True)
    _save(fig, out_dir, filename)


# --------------------------------------------------------------------------- #
# Figure 2 — VG10 per-draw GP anchoring
# --------------------------------------------------------------------------- #
def _observed_ages():
    """Ages of the VG10 analysis frame, with multiplicity, for the projection."""
    df = load_combined_data()
    frame = df[df["understood"].notna() | df["spoken"].notna()]
    return frame["age"].to_numpy(dtype=float)


def build_anchoring(out_dir, filename="gp_anchoring_vg10",
                    n_draws=250, n_grid=240, seed=RANDOM_SEED):
    d = _definition("vg10")
    anchors = tuple(float(a) for a in d.slope_anchors)
    ref = float(d.gp_anchor_age_months)
    d_lo = beta_dist(d.p_slope_low_u_alpha, d.p_slope_low_u_beta)
    d_hi = beta_dist(d.p_slope_hi_u_alpha, d.p_slope_hi_u_beta)

    rng = np.random.default_rng(seed)
    # The reference age goes on the plot grid explicitly, or the nearest point is
    # up to ~0.22 months away and the pinch reads as approximate rather than exact.
    plot_ages = np.unique(
        np.concatenate([np.linspace(*GP_DOMAIN_MONTHS, n_grid), [ref]])
    )
    obs_ages = _observed_ages()

    # One GP realisation per draw over the union of plot, observed and anchor
    # ages -- the rows the model stacks into X_all.
    grid = np.unique(np.concatenate([plot_ages, obs_ages, [ref]]))
    i_plot = np.searchsorted(grid, plot_ages)
    i_obs = np.searchsorted(grid, obs_ages)
    i_ref = int(np.searchsorted(grid, ref))

    # Whether the mean levels off above the high anchor is read from the
    # definition, not assumed: a model with the clamp turned off would otherwise
    # be drawn with a flattening its own specification does not have.
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
    _save(fig, out_dir, filename)


FIGURES = {"structure": build_structure, "anchoring": build_anchoring}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("figure", nargs="?", default="all",
                    choices=["all", *FIGURES], help="Which figure to build.")
    ap.add_argument("--output-dir", default=None,
                    help="Override the report figure cache destination.")
    ap.add_argument("--draws", type=int, default=250)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    setup.init_script()
    out_dir = args.output_dir or os.path.join(local_env.REPORT_FIGS_DIR, "methods")
    for name in (FIGURES if args.figure == "all" else [args.figure]):
        FIGURES[name](out_dir, n_draws=args.draws, seed=args.seed)


if __name__ == "__main__":
    main()
