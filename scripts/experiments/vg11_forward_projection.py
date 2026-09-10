#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What might VG11's typically-developing spoken trajectory look like at 5 years?

The typically-developing companion to ``vg20_forward_projection.py``, asked for
by the study owner on 2026-09-10 and recorded in
``notes/202609101115-vg11-forward-projection-to-5-years.md``. It is **not** a
model output, and it is a *weaker* extrapolation than the Down syndrome one --
weak enough that the note's conclusion is about what the projection cannot say.

VG11's frame stops at 30 months, where the median child is at 0.53 of the
810-word reference form and still gaining 24 words a month, six months past its
peak rate of 39: the deceleration has begun but has barely got anywhere, so
nothing in the data locates where the curve levels off. The same three-family
construction the Down syndrome projection uses (which extends from 84 months,
where the spoken curve is gaining 3 words a month and slowing) therefore
disagrees with itself by 173 words at 5 years, and the model's own mean function
-- a logit-linear age trend, recoverable here exactly from ``intercept`` and
``slope`` -- runs to the checklist ceiling by 4 years. The two bracket the
answer rather than locate it, which is the finding.

Method, following the Down syndrome harness:

1. Read the model of record's stored population curve on the 8-30 month plot
   grid (``p_plot``, at zero study and child effects, i.e. the median child),
   the between-child scale ``tau_subject``, and dispersion ``kappa_plot``.
2. For each thinned posterior draw, fit three saturating growth families to the
   model's own curve over ``--fit-lo`` to ``--fit-hi`` (default 12-30 months):
   logistic, Gompertz and Hill (log-logistic), each with a free asymptote below
   the checklist ceiling.
3. Above the window, continue each family from the model's own value at the
   window's top -- a constant shift on the logit scale keeps it continuous and
   bounded. Families are pooled with equal weight.
4. Three levels are reported: the population median child (curve uncertainty
   only); individual children (adds ``tau_subject`` on the logit scale, the
   persistent between-child differences); and the full predictive count (adds
   the Beta-Binomial occasion-level dispersion, which is what VG11's reported
   ``Y_*`` columns carry). Above the window ``kappa`` is continued by the
   model's own exact functional form, ``kappa_min + exp(a + b z)``, recovered
   per draw from the stored grid -- it decays toward ``kappa_min``, so the
   continuation is bounded (``--kappa-mode hold`` pins it at the window's top
   value instead).
5. VG11's own logit-linear mean function is reported alongside, as the Down
   syndrome note reports VG20's own climb: the reference the data stop
   correcting. Two readings of it -- ``vg11_trend_continued``, the mean function
   itself, which is where the curve goes if the HSGP deviation reverts to zero
   above the window, and ``vg11_trend_spliced``, its slope continued from the
   model's own value at the window's top. Both reach the ceiling well inside
   5 years, and they are the projection's upper bracket.

Inside the window the predictive level reproduces VG11's reported ``Y_*``
intervals, which is what checks the construction.

Writes, under ``<output-root>/experiments/vg11-forward-projection/``:

* ``projection_intervals.csv`` -- monthly medians with 50%, 75% and 89%
  equal-tailed intervals at each level.
* ``projection_family_medians.csv`` -- the per-family medians.
* ``projection_fit_diagnostics.csv`` -- per draw and family: RMSE over the
  window, fitted asymptote, residual at the window's top, and the family's
  value at ``--max-age``.
* ``vg11_projection_to_5y.png`` -- the figure.

Usage::

    python scripts/experiments/vg11_forward_projection.py
    python scripts/experiments/vg11_forward_projection.py --fit-hi 27 --tag _fit27

The fit is validated through ``vocab_growth.fit_consumers`` before it is read;
``--allow-stale-fit`` is the documented override. The one-off harness
conventions of ``scripts/experiments/README.md`` apply.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import least_squares
from scipy.special import expit, logit

from vocab_growth import environment as env
from vocab_growth import fit_consumers
from vocab_growth.models.definitions import MODEL_REGISTRY

MODEL_KEY = "vg11"
CONSUMER = "vg11_forward_projection.py"
LEVELS = {"ci50": (0.25, 0.75), "ci75": (0.125, 0.875), "ci89": (0.055, 0.945)}
EPS = 1e-9
# The typically-developing pool's spoken counts come from 680-item Words &
# Sentences forms scored against the 810-word reference, so 680/810 is the
# largest proportion any observation in the frame can express. Drawn on the
# figure and quoted in the note; nothing in the projection is clipped to it.
WS_FORM_ITEMS = 680
# The levels the age-at-count table is read off: the pooled families and the
# model's own logit-linear continuation, the two ends of the bracket.
AGE_AT_COUNT_LEVELS = ("population_median_child", "vg11_trend_spliced")


# ------------------------------------------------------------------ families
def _logistic(t, a, lk, t0, amax):
    return amax * expit(a) / (1.0 + np.exp(-np.exp(lk) * (t - t0)))


def _gompertz(t, a, lk, t0, amax):
    return amax * expit(a) * np.exp(-np.exp(-np.exp(lk) * (t - t0)))


def _hill(t, a, lb, lt0, amax):
    return amax * expit(a) / (1.0 + np.exp(-np.exp(lb) * (np.log(t) - lt0)))


def _start_asymptote(y):
    return logit(min(0.97, max(y.max() * 1.3, 0.05)))


FAMILIES = {
    "logistic": (_logistic, lambda y: [_start_asymptote(y), np.log(0.20), 26.0]),
    "gompertz": (_gompertz, lambda y: [_start_asymptote(y), np.log(0.12), 24.0]),
    "hill": (_hill, lambda y: [_start_asymptote(y), np.log(6.0), np.log(26.0)]),
}
FAMILY_LABEL = {"logistic": "logistic", "gompertz": "Gompertz", "hill": "Hill (log-logistic)"}


def fit_family(name, t, y, amax):
    fn, init = FAMILIES[name]
    with np.errstate(over="ignore"):
        res = least_squares(
            lambda th: fn(t, *th, amax) - y, init(y), method="lm", max_nfev=2000
        )
    return res.x, res.success, float(np.sqrt(np.mean(res.fun**2)))


def spliced(name, params, t_out, amax, t_top, y_top):
    """The family on ``t_out``, shifted on the logit scale to pass through (t_top, y_top)."""
    fn = FAMILIES[name][0]
    with np.errstate(over="ignore"):
        y = np.clip(fn(t_out, *params, amax) / amax, EPS, 1 - EPS)
        y_at_top = np.clip(fn(np.array([t_top]), *params, amax)[0] / amax, EPS, 1 - EPS)
    shift = logit(np.clip(y_top / amax, EPS, 1 - EPS)) - logit(y_at_top)
    return expit(logit(y) + shift) * amax


def quantile_table(samples, ages, label):
    out = {"age_months": ages, "age_years": ages / 12.0, "outcome": label}
    out["median"] = np.median(samples, axis=0)
    for key, (lo, hi) in LEVELS.items():
        out[f"{key}_lo"] = np.quantile(samples, lo, axis=0)
        out[f"{key}_hi"] = np.quantile(samples, hi, axis=0)
    return pd.DataFrame(out)


def age_to_z(t_grid, z_all, n_obs):
    """Recover the exact affine age standardisation the fit used."""
    z_plot = np.asarray(z_all).reshape(len(z_all), -1)[n_obs : n_obs + len(t_grid), 0]
    slope, intercept = np.polyfit(t_grid, z_plot, 1)
    resid = float(np.abs(np.polyval((slope, intercept), t_grid) - z_plot).max())
    if resid > 1e-8:
        raise RuntimeError(f"age standardisation is not affine on the plot grid ({resid:.2e})")
    return lambda a: slope * np.asarray(a, dtype=float) + intercept


def kappa_continuation(kappa_grid, kappa_min, z_grid, z_out, mode):
    """Continue ``kappa`` above the window.

    ``kappa(z) = kappa_min + exp(a + b z)`` by construction (`gp_utils`'s
    two-anchor form), so ``a`` and ``b`` are recovered exactly from two grid
    points and the continuation is the model's own function, not a new
    assumption. It decays toward ``kappa_min`` here, so it stays bounded.
    """
    if mode == "hold":
        return np.repeat(kappa_grid[:, -1:], len(z_out), axis=1)
    excess = np.clip(kappa_grid - kappa_min[:, None], EPS, None)
    log_excess = np.log(excess)
    b = (log_excess[:, -1] - log_excess[:, 0]) / (z_grid[-1] - z_grid[0])
    a = log_excess[:, 0] - b * z_grid[0]
    return kappa_min[:, None] + np.exp(a[:, None] + b[:, None] * z_out[None, :])


# ------------------------------------------------------------------ projection
def project(args, trace_path, n_trials, out_dir):
    rng = np.random.default_rng(args.seed)
    cd = xr.open_dataset(trace_path, group="constant_data")
    post = xr.open_dataset(trace_path, group="posterior")
    t_grid = cd["X_plot"].values.astype(float).ravel()
    z_of = age_to_z(t_grid, cd["X_all_z"].values, len(cd["X_obs"].values))

    def flat(name):
        return post[name].stack(sample=("chain", "draw")).transpose("sample", ...).values

    n_total = post.sizes["chain"] * post.sizes["draw"]
    step = max(1, n_total // args.draws)
    idx = np.arange(0, n_total, step)[: args.draws]
    p = flat("p_plot")[idx]
    kappa_grid = flat("kappa_plot")[idx]
    kappa_min = flat("kappa_min")[idx]
    tau_subject = flat("tau_subject")[idx]
    intercept, slope = flat("intercept")[idx], flat("slope")[idx]
    n = len(idx)
    print(
        f"draws used: {n} of {n_total} (every {step}th); "
        f"grid {t_grid.min():.0f}-{t_grid.max():.0f} mo; ceiling {n_trials} words"
    )

    win = (t_grid >= args.fit_lo) & (t_grid <= args.fit_hi)
    t_fit = t_grid[win]
    t_out = np.arange(8.0, args.max_age + 1e-9, 1.0)
    i_top = int(np.argmin(np.abs(t_grid - args.fit_hi)))
    t_top = t_grid[i_top]
    inside = t_out <= t_top
    z_out = z_of(t_out)

    fams = list(FAMILIES)
    p_fam = {f: np.empty((n, len(t_out))) for f in fams}
    diag = []
    fails = 0
    for d in range(n):
        y = p[d]
        for f in fams:
            params, ok, rmse = fit_family(f, t_fit, y[win], 1.0)
            fails += not ok
            curve = spliced(f, params, t_out, 1.0, t_top, y[i_top])
            # inside the window the model's own curve; the family only above it
            curve[inside] = np.interp(t_out[inside], t_grid, y)
            p_fam[f][d] = curve
            fn = FAMILIES[f][0]
            with np.errstate(over="ignore"):
                at_top = fn(np.array([t_top]), *params, 1.0)[0]
            diag.append(
                dict(
                    draw=d, family=f, outcome="spoken", rmse=rmse, ok=ok,
                    asymptote=expit(params[0]),
                    resid_at_top=y[i_top] - at_top,
                    at_max_age=curve[-1],
                )
            )
    print(f"non-converged fits: {fails} of {n * len(fams)}")
    diag = pd.DataFrame(diag)
    diag.to_csv(os.path.join(out_dir, f"projection_fit_diagnostics{args.tag}.csv"), index=False)
    print(
        diag.groupby("family")[["rmse", "asymptote", "resid_at_top", "at_max_age"]]
        .median().to_string()
    )

    # kappa above the window, on the model's own functional form
    z_grid = z_of(t_grid)
    kappa_out = kappa_continuation(kappa_grid, kappa_min, z_grid, z_out, args.kappa_mode)
    on_grid_k = np.stack(
        [np.interp(t_out, t_grid, row, left=np.nan, right=np.nan) for row in kappa_grid]
    )
    kappa_out = np.where(np.isnan(on_grid_k), kappa_out, on_grid_k)

    # level 1: population median child, pooled across families
    pop_p = np.concatenate([p_fam[f] for f in fams])
    pop = quantile_table(n_trials * pop_p, t_out, "spoken")
    pop["level"] = "population_median_child"

    fam_df = pd.concat([
        pd.DataFrame({
            "age_months": t_out, "outcome": "spoken", "family": f,
            "median": np.median(n_trials * p_fam[f], axis=0),
        })
        for f in fams
    ])
    fam_df.to_csv(os.path.join(out_dir, f"projection_family_medians{args.tag}.csv"), index=False)

    # levels 2 and 3: one new child per (draw, replicate), then a count for it
    m = args.children_per_draw
    child_p, child_kappa = [], []
    for f in fams:
        shift = tau_subject[:, None] * rng.standard_normal((n, m))
        pc = expit(logit(np.clip(p_fam[f], EPS, 1 - EPS))[:, None, :] + shift[:, :, None])
        child_p.append(pc.reshape(-1, len(t_out)))
        child_kappa.append(np.repeat(kappa_out, m, axis=0))
    child_p = np.concatenate(child_p)
    child_kappa = np.concatenate(child_kappa)

    child = quantile_table(n_trials * child_p, t_out, "spoken")
    child["level"] = "individual_children"

    pc = np.clip(child_p, EPS, 1 - EPS)
    theta = rng.beta(pc * child_kappa, (1 - pc) * child_kappa)
    counts = rng.binomial(n_trials, np.clip(theta, 0.0, 1.0)).astype(float)
    pred = quantile_table(counts, t_out, "spoken")
    pred["level"] = "individual_children_predictive"

    # VG11's own mean function, continued: the reference the data stop
    # correcting. Two readings, both the model's own logit-linear age trend.
    # `continued` is the mean function itself, which is where the curve goes if
    # the HSGP deviation reverts to zero above the window as it does past the
    # data in the Down syndrome models; it is discontinuous with the model's
    # curve at the window's top, because that deviation is large and negative
    # there. `spliced` continues the trend's slope from the model's own value
    # instead, the same splice the families get: growth carrying on at the
    # fitted logit-linear rate from where the data leave off.
    trend = expit(intercept[:, None] + slope[:, None] * z_out[None, :])
    tr = quantile_table(n_trials * trend, t_out, "spoken")
    tr["level"] = "vg11_trend_continued"

    logit_top = logit(np.clip(p[:, i_top], EPS, 1 - EPS))
    trend_spliced = expit(logit_top[:, None] + slope[:, None] * (z_out - z_of(t_top))[None, :])
    trs = quantile_table(n_trials * trend_spliced, t_out, "spoken")
    trs["level"] = "vg11_trend_spliced"

    allq = pd.concat([pop, child, pred, tr, trs], ignore_index=True)
    allq.to_csv(os.path.join(out_dir, f"projection_intervals{args.tag}.csv"), index=False)

    ages_show = [30, 36, 42, 48, 54, 60]
    cols = ["age_months", "median", "ci50_lo", "ci50_hi", "ci75_lo", "ci75_hi", "ci89_lo", "ci89_hi"]
    for level in allq.level.unique():
        sub = allq[(allq.level == level) & allq.age_months.isin(ages_show)]
        print(f"\n== {level}")
        print(sub[cols].round(0).to_string(index=False))
    print("\n== inside the window (checks against VG11's reported Y_*)")
    sub = allq[(allq.level == "individual_children_predictive") & allq.age_months.isin([24, 27, 30])]
    print(sub[cols].round(0).to_string(index=False))
    print("\n== per-family medians")
    print(
        fam_df[fam_df.age_months.isin([36, 48, 60])]
        .pivot_table(index="age_months", columns="family", values="median")
        .round(0).to_string()
    )
    print("\n== median kappa")
    for a in (30, 36, 48, 60):
        i = int(np.argmin(np.abs(t_out - a)))
        print(f"  {a:3d} mo  {np.median(kappa_out[:, i]):7.2f}")

    # The age the median child reaches a given count. Interesting because it is
    # far better supported than a count at 5 years: the count grows slowly above
    # the window under every continuation, so an age read off it moves little.
    print("\n== age (months) the population median child reaches a count")
    print(f"  {'words':>6s}  " + "  ".join(f"{lv:>23s}" for lv in AGE_AT_COUNT_LEVELS))
    for target in args.counts:
        cells = []
        for lv in AGE_AT_COUNT_LEVELS:
            d = allq[allq.level == lv].sort_values("age_months")
            if lv != "population_median_child":
                # only a projection above the window: below its top this level
                # is a backward extension of the trend, not the model's curve
                d = d[d.age_months >= t_top]
            med = d["median"].to_numpy()
            if med.min() > target:
                cells.append(f"{'in window':>23s}")
            elif med.max() < target:
                cells.append(f"{'>' + str(int(args.max_age)):>23s}")
            else:
                cells.append(f"{np.interp(target, med, d.age_months.to_numpy()):23.1f}")
        print(f"  {target:6.0f}  " + "  ".join(cells))

    od = xr.open_dataset(trace_path, group="observed_data")
    obs = (cd["X_obs"].values.astype(float), od[next(iter(od.data_vars))].values.astype(float))
    return allq, fam_df, obs


# ------------------------------------------------------------------ figure
def make_figure(iv, fam, obs, out_png, fit_hi, fit_lo, n_trials):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ink, ink2, muted, grid = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
    line, bands = "#1c5cab", ["#cde2fb", "#9ec5f4", "#5598e7"]
    levels = [("ci89", "89%"), ("ci75", "75%"), ("ci50", "50%")]
    fit_hi_y = fit_hi / 12.0
    x_max = iv.age_months.max() / 12.0
    panels = [
        ("population_median_child", "Population median child\n(posterior + extrapolation uncertainty)"),
        ("individual_children", "Individual children\n(adds persistent between-child differences)"),
        ("individual_children_predictive", "One administration for one child\n(adds occasion-level dispersion)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), sharex=True, sharey=True, dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (level, title) in zip(axes, panels, strict=True):
        ax.set_facecolor("#fcfcfb")
        d = iv[iv.level == level].sort_values("age_months")
        yrs = d.age_months / 12.0
        for (key, _), colour in zip(levels, bands, strict=True):
            ax.fill_between(yrs, d[f"{key}_lo"], d[f"{key}_hi"], color=colour, lw=0, zorder=2)
        if level == "individual_children_predictive":
            ox, oy = obs
            ax.scatter(ox / 12.0, oy, s=4, color=muted, alpha=0.12, lw=0, zorder=3)
        ax.plot(yrs, d["median"], color=line, lw=2, zorder=5)
        ref = iv[(iv.level == "vg11_trend_spliced") & (iv.age_months >= fit_hi)].sort_values("age_months")
        ax.plot(ref.age_months / 12.0, ref["median"], color=ink2, lw=1.4, ls=(0, (4, 2)), zorder=4)
        if level == "population_median_child":
            for f in FAMILIES:
                g = fam[(fam.family == f) & (fam.age_months > fit_hi)].sort_values("age_months")
                ax.plot(g.age_months / 12.0, g["median"], color=line, lw=0.8, alpha=0.75, zorder=4)
                ax.annotate(
                    FAMILY_LABEL[f], (g.age_months.iloc[-1] / 12.0 + 0.04, g["median"].iloc[-1]),
                    fontsize=7.5, color=ink2, va="center", ha="left",
                )
            ax.annotate(
                "VG11's own logit-linear\nrate continued",
                (3.6, 760), fontsize=7.5, color=ink2, va="top", ha="left",
            )
            ax.text(0.68, n_trials + 10, f"reference form: {n_trials} words", fontsize=7.5, color=ink2, va="bottom")
            ax.text(0.68, WS_FORM_ITEMS + 10, f"Words & Sentences form: {WS_FORM_ITEMS}", fontsize=7.5, color=ink2, va="bottom")
        ax.axvline(fit_hi_y, color=muted, lw=0.8, ls=":", zorder=1)
        ax.axhline(n_trials, color=muted, lw=0.8, ls=":", zorder=1)
        ax.axhline(WS_FORM_ITEMS, color=muted, lw=0.8, ls=(0, (1, 3)), zorder=1)
        ax.set_title(title, fontsize=10, color=ink, loc="left", pad=8)
        ax.text(fit_hi_y - 0.08, 30, f"data stop ({fit_hi_y:g} y)", fontsize=7.5, color=ink2, ha="right")
        ax.set_xlim(0.6, x_max + 0.75)
        ax.set_ylim(0, n_trials + 55)
        ax.set_xticks(range(1, int(x_max) + 1))
        ax.set_yticks(range(0, n_trials + 1, 100))
        ax.grid(True, axis="y", color=grid, lw=0.7, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8.5, length=0)
        ax.set_xlabel("Age (years)", fontsize=9.5, color=ink)
    axes[0].set_ylabel(f"Words spoken (of {n_trials})", fontsize=9.5, color=ink)

    handles = [Patch(color=bands[i], label=f"{lbl} interval") for i, (_, lbl) in enumerate(levels)]
    handles += [
        Line2D([0], [0], color=line, lw=2, label="median (families pooled)"),
        Line2D([0], [0], color=line, lw=0.8, alpha=0.75, label="one growth family"),
        Line2D([0], [0], color=ink2, lw=1.4, ls=(0, (4, 2)), label="VG11's own logit-linear rate continued"),
        Line2D([0], [0], marker="o", color=muted, lw=0, markersize=3, alpha=0.6, label="observed administrations"),
    ]
    fig.legend(handles=handles, loc="lower center", fontsize=8, frameon=False, ncol=7, labelcolor=ink2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"Typically-developing words spoken to {x_max:g} years: VG11 posterior to {fit_hi_y:g} years, parametric projection beyond",
        fontsize=12.5, color=ink, x=0.01, ha="left", y=0.995,
    )
    fig.text(
        0.01, 0.945,
        "Three saturating growth families (logistic, Gompertz, Hill) fitted per posterior draw to the model's own curve over "
        f"{fit_lo:g}-{fit_hi:g} months and continued upward, pooled with equal weight. The window ends six months past the "
        "peak growth rate and nowhere near a plateau, so the families disagree by 173 words at 5 years and the model's own "
        "mean function runs to the ceiling: the projection brackets the answer rather than locating it. Exploratory "
        "extrapolation, not a model output.",
        fontsize=8.5, color=ink2, ha="left", va="top", wrap=True,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", out_png)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=None, help="output root (default: the environment's)")
    ap.add_argument("--fit-lo", type=float, default=12.0, help="bottom of the fit window, months")
    ap.add_argument("--fit-hi", type=float, default=30.0, help="top of the fit window, months")
    ap.add_argument("--max-age", type=float, default=60.0, help="last projected age, months")
    ap.add_argument("--draws", type=int, default=2000, help="posterior draws after thinning")
    ap.add_argument("--children-per-draw", type=int, default=3)
    ap.add_argument(
        "--kappa-mode", choices=("trend", "hold"), default="trend",
        help="continue the model's own kappa function above the window, or hold it at the window's top",
    )
    ap.add_argument(
        "--counts", type=lambda v: [float(x) for x in v.split(",")],
        default=[300.0, 430.0, 525.0, 600.0, 680.0],
        help="comma-separated word counts for the age-at-count table",
    )
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. _fit27")
    ap.add_argument("--no-figure", action="store_true")
    fit_consumers.add_allow_stale_argument(ap)
    args = ap.parse_args()

    env.set_output_root(args.output_dir)
    fit_dir = fit_consumers.model_fit_dir(MODEL_KEY)
    fit_consumers.require_current_fit(
        MODEL_KEY, fit_dir, consumer=CONSUMER, allow_stale=args.allow_stale_fit
    )
    n_trials = MODEL_REGISTRY[MODEL_KEY].n_trials
    out_dir = os.path.join(env.output_root(), "experiments", "vg11-forward-projection")
    os.makedirs(out_dir, exist_ok=True)

    allq, fam_df, obs = project(args, os.path.join(fit_dir, "trace.nc"), n_trials, out_dir)
    if not args.no_figure:
        make_figure(
            allq, fam_df, obs, os.path.join(out_dir, f"vg11_projection_to_5y{args.tag}.png"),
            args.fit_hi, args.fit_lo, n_trials,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
