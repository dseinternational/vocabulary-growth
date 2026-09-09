#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What might VG20's understood and spoken trajectories look like at 11 years?

An exploratory forward projection, asked for by the study owner on 2026-09-09
and recorded in ``notes/202609091900-vg20-forward-projection-to-11-years.md``.
It is **not** a model output: VG20's own curve exists to 115 months, but above
about 8 years it is the logit-linear age trend running with no data to correct
it -- 12 comprehension and 49 spoken rows lie above 7 years, 4 and 12 above 8 --
so the projection is built from the part of the curve the data support and
extended by assumption, with the assumption's spread carried into the bands.

Method:

1. Read the model of record's stored population curves on the 8-115 month plot
   grid -- ``p_u_plot`` (understood proportion) and ``q_plot`` (production
   ratio), both at zero study and child effects, i.e. the median child -- plus
   the between-child scales ``tau_subj_u`` / ``tau_subj_q`` and their correlation
   ``rho_uq``.
2. For each thinned posterior draw, fit three saturating growth families to the
   model's own curve over the supported window (``--fit-lo`` to ``--fit-hi``,
   default 12-84 months, the high slope anchor): logistic, Gompertz and Hill
   (log-logistic), each with a free asymptote below the checklist ceiling. The
   target is a smooth curve, so each fit is a projection of that draw onto the
   family; the spread comes from the posterior and from the family, not from
   fit noise.
3. Above the window, continue each family from the model's own value at the
   window's top -- a constant shift on the logit scale keeps it continuous and
   bounded. Spoken is ``810 * p_u * q``, crossing every understood family with
   every ``q`` family. Families are pooled with equal weight.
4. Child-level bands add the model's own correlated child effects on the logit
   scale; inside the window they reproduce the report's subject-marginal
   intervals.

Writes, under ``<output-root>/experiments/vg20-forward-projection/``:

* ``projection_intervals.csv`` -- monthly medians with 50%, 75% and 89%
  equal-tailed intervals for the population median child, for individual
  children, and for VG20's own curve where it exists.
* ``projection_family_medians.csv`` -- the per-family medians.
* ``projection_fit_diagnostics.csv`` -- per draw and family: RMSE over the
  window, fitted asymptote, residual at the window's top, and the family's
  value at 115 months against the model's own.
* ``vg20_projection_to_11y.png`` -- the figure.

Usage::

    python scripts/experiments/vg20_forward_projection.py
    python scripts/experiments/vg20_forward_projection.py --fit-hi 72 --tag _fit72

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

MODEL_KEY = "vg20"
CONSUMER = "vg20_forward_projection.py"
LEVELS = {"ci50": (0.25, 0.75), "ci75": (0.125, 0.875), "ci89": (0.055, 0.945)}
EPS = 1e-9


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
    "logistic": (_logistic, lambda y: [_start_asymptote(y), np.log(0.05), 60.0]),
    "gompertz": (_gompertz, lambda y: [_start_asymptote(y), np.log(0.03), 40.0]),
    "hill": (_hill, lambda y: [_start_asymptote(y), np.log(2.5), np.log(60.0)]),
}
FAMILY_LABEL = {"logistic": "logistic", "gompertz": "Gompertz", "hill": "Hill (log-logistic)"}


def fit_family(name, t, y, amax):
    fn, init = FAMILIES[name]
    res = least_squares(lambda th: fn(t, *th, amax) - y, init(y), method="lm", max_nfev=2000)
    return res.x, res.success, float(np.sqrt(np.mean(res.fun**2)))


def spliced(name, params, t_out, amax, t_top, y_top):
    """The family on ``t_out``, shifted on the logit scale to pass through (t_top, y_top)."""
    fn = FAMILIES[name][0]
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


# ------------------------------------------------------------------ projection
def project(args, trace_path, n_trials, out_dir):
    rng = np.random.default_rng(args.seed)
    cd = xr.open_dataset(trace_path, group="constant_data")
    post = xr.open_dataset(trace_path, group="posterior")
    t_grid = cd["X_plot"].values.astype(float)

    def flat(name):
        return post[name].stack(sample=("chain", "draw")).transpose("sample", ...).values

    n_total = post.sizes["chain"] * post.sizes["draw"]
    step = max(1, n_total // args.draws)
    idx = np.arange(0, n_total, step)[: args.draws]
    p_u, q = flat("p_u_plot")[idx], flat("q_plot")[idx]
    tau_u, tau_q, rho = flat("tau_subj_u")[idx], flat("tau_subj_q")[idx], flat("rho_uq")[idx]
    n = len(idx)
    print(f"draws used: {n} of {n_total} (every {step}th); grid {t_grid.min():.0f}-{t_grid.max():.0f} mo")

    win = (t_grid >= args.fit_lo) & (t_grid <= args.fit_hi)
    t_fit = t_grid[win]
    t_out = np.arange(8.0, args.max_age + 1e-9, 1.0)
    i_top = int(np.argmin(np.abs(t_grid - args.fit_hi)))
    t_top = t_grid[i_top]
    inside = t_out <= t_top

    def on_grid(mat):
        return np.stack([np.interp(t_out, t_grid, row, left=np.nan, right=np.nan) for row in mat])

    fams = list(FAMILIES)
    pu_fam = {f: np.empty((n, len(t_out))) for f in fams}
    q_fam = {f: np.empty((n, len(t_out))) for f in fams}
    diag = []
    fails = 0
    for d in range(n):
        for f in fams:
            for label, y, store in (("p_u", p_u[d], pu_fam), ("q", q[d], q_fam)):
                params, ok, rmse = fit_family(f, t_fit, y[win], 1.0)
                fails += not ok
                curve = spliced(f, params, t_out, 1.0, t_top, y[i_top])
                # inside the window the model's own curve; the family only above it
                curve[inside] = np.interp(t_out[inside], t_grid, y)
                store[f][d] = curve
                fn = FAMILIES[f][0]
                diag.append(
                    dict(
                        draw=d, family=f, outcome=label, rmse=rmse, ok=ok,
                        asymptote=expit(params[0]),
                        resid_at_top=y[i_top] - fn(np.array([t_top]), *params, 1.0)[0],
                        model_at_115=y[-1],
                        family_at_115=spliced(f, params, np.array([t_grid.max()]), 1.0, t_top, y[i_top])[0],
                    )
                )
    print(f"non-converged fits: {fails} of {n * len(fams) * 2}")
    diag = pd.DataFrame(diag)
    diag.to_csv(os.path.join(out_dir, f"projection_fit_diagnostics{args.tag}.csv"), index=False)
    print(
        diag.groupby(["outcome", "family"])[
            ["rmse", "asymptote", "resid_at_top", "model_at_115", "family_at_115"]
        ].median().to_string()
    )

    # population (median child), pooled across families
    u_pop = np.concatenate([n_trials * pu_fam[f] for f in fams])
    s_pop = np.concatenate([n_trials * pu_fam[fu] * q_fam[fq] for fu in fams for fq in fams])
    q_pop = np.concatenate([q_fam[f] for f in fams])
    pop = pd.concat([
        quantile_table(u_pop, t_out, "understood"),
        quantile_table(s_pop, t_out, "spoken"),
        quantile_table(q_pop, t_out, "production_ratio"),
    ])
    pop["level"] = "population_median_child"

    fam_rows = []
    for f in fams:
        fam_rows.append(pd.DataFrame({
            "age_months": t_out, "outcome": "understood", "family": f,
            "median": np.median(n_trials * pu_fam[f], axis=0),
        }))
        fam_rows.append(pd.DataFrame({
            "age_months": t_out, "outcome": "spoken", "family": f,
            "median": np.median(n_trials * pu_fam[f] * q_fam[f], axis=0),
        }))
    fam_df = pd.concat(fam_rows)
    fam_df.to_csv(os.path.join(out_dir, f"projection_family_medians{args.tag}.csv"), index=False)

    # the model's own curve, where it exists
    pu_model, q_model = on_grid(p_u), on_grid(q)
    ref = pd.concat([
        quantile_table(n_trials * pu_model, t_out, "understood"),
        quantile_table(n_trials * pu_model * q_model, t_out, "spoken"),
    ])
    ref["level"] = "vg20_model_curve"
    ref = ref.dropna(subset=["median"])

    # individual children: the model's correlated child effects on the logit scale
    m = args.children_per_draw

    def child_samples(pu_mat, q_mat):
        z1 = rng.standard_normal((pu_mat.shape[0], m))
        z2 = rng.standard_normal((pu_mat.shape[0], m))
        du = tau_u[:, None] * z1
        dq = tau_q[:, None] * (rho[:, None] * z1 + np.sqrt(1 - rho[:, None] ** 2) * z2)
        pu_c = expit(logit(np.clip(pu_mat, EPS, 1 - EPS))[:, None, :] + du[:, :, None])
        q_c = expit(logit(np.clip(q_mat, EPS, 1 - EPS))[:, None, :] + dq[:, :, None])
        width = pu_mat.shape[1]
        return (n_trials * pu_c).reshape(-1, width), (n_trials * pu_c * q_c).reshape(-1, width)

    u_child, s_child = [], []
    for fu in fams:
        for fq in fams:
            u_c, s_c = child_samples(pu_fam[fu], q_fam[fq])
            if fq == fams[0]:
                u_child.append(u_c)
            s_child.append(s_c)
    child = pd.concat([
        quantile_table(np.concatenate(u_child), t_out, "understood"),
        quantile_table(np.concatenate(s_child), t_out, "spoken"),
    ])
    child["level"] = "individual_children"

    allq = pd.concat([pop, ref, child], ignore_index=True)
    allq.to_csv(os.path.join(out_dir, f"projection_intervals{args.tag}.csv"), index=False)

    ages_show = [72, 84, 96, 108, 120, 132]
    cols = ["age_months", "median", "ci50_lo", "ci50_hi", "ci75_lo", "ci75_hi", "ci89_lo", "ci89_hi"]
    for level in ("population_median_child", "individual_children"):
        for outcome in ("understood", "spoken"):
            sub = allq[(allq.level == level) & (allq.outcome == outcome) & allq.age_months.isin(ages_show)]
            print(f"\n== {level} / {outcome}")
            print(sub[cols].round(0).to_string(index=False))
    sub = allq[(allq.level == "vg20_model_curve") & allq.age_months.isin([72, 84, 96, 108, 115])]
    print("\n== VG20's own curve (unsupported above ~8 years)")
    print(sub[["outcome", "age_months", "median", "ci89_lo", "ci89_hi"]].round(0).to_string(index=False))
    print("\n== per-family medians")
    print(
        fam_df[fam_df.age_months.isin([96, 108, 120, 132])]
        .pivot_table(index=["outcome", "age_months"], columns="family", values="median")
        .round(0).to_string()
    )

    obs = {}
    od = xr.open_dataset(trace_path, group="observed_data")
    x = cd["X_obs"].values
    um = cd["obs_u_mask"].values.astype(bool)
    sm = cd["obs_s_mask"].values.astype(bool)
    obs["understood"] = (x[um], od["y_u_obs"].values)
    obs["spoken"] = (x[sm], od["y_s_obs"].values)
    return allq, fam_df, obs


# ------------------------------------------------------------------ figure
def make_figure(iv, fam, obs, out_png, fit_hi, fit_lo, n_trials):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ink, ink2, muted, grid = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
    hues = {
        "understood": {"line": "#1c5cab", "bands": ["#cde2fb", "#9ec5f4", "#5598e7"]},
        "spoken": {"line": "#c44a1e", "bands": ["#fbe0d5", "#f5b89f", "#ee8a63"]},
    }
    levels = [("ci89", "89%"), ("ci75", "75%"), ("ci50", "50%")]
    fit_hi_y = fit_hi / 12.0
    x_max = iv.age_months.max() / 12.0

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), sharex=True, sharey=True, dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    cols = [
        ("population_median_child", "Population median child\n(posterior + extrapolation uncertainty)"),
        ("individual_children", "Individual children\n(adds the model's between-child spread)"),
    ]
    for r, outcome in enumerate(("understood", "spoken")):
        hue = hues[outcome]
        for c, (level, col_title) in enumerate(cols):
            ax = axes[r, c]
            ax.set_facecolor("#fcfcfb")
            d = iv[(iv.level == level) & (iv.outcome == outcome)].sort_values("age_months")
            yrs = d.age_months / 12.0
            for (key, _), colour in zip(levels, hue["bands"], strict=True):
                ax.fill_between(yrs, d[f"{key}_lo"], d[f"{key}_hi"], color=colour, lw=0, zorder=2)
            if level == "individual_children":
                ox, oy = obs[outcome]
                ax.scatter(ox / 12.0, oy, s=7, color=muted, alpha=0.35, lw=0, zorder=3)
            ax.plot(yrs, d["median"], color=hue["line"], lw=2, zorder=5)
            ref = iv[(iv.level == "vg20_model_curve") & (iv.outcome == outcome)].sort_values("age_months")
            ref = ref[ref.age_months > fit_hi]
            ax.plot(ref.age_months / 12.0, ref["median"], color=ink2, lw=1.4, ls=(0, (4, 2)), zorder=4)
            if level == "population_median_child":
                for f in FAMILIES:
                    g = fam[(fam.outcome == outcome) & (fam.family == f) & (fam.age_months > fit_hi)]
                    g = g.sort_values("age_months")
                    ax.plot(g.age_months / 12.0, g["median"], color=hue["line"], lw=0.8, alpha=0.75, zorder=4)
                    ax.annotate(
                        FAMILY_LABEL[f], (g.age_months.iloc[-1] / 12.0 + 0.08, g["median"].iloc[-1]),
                        fontsize=7.5, color=ink2, va="center", ha="left",
                    )
                ax.annotate(
                    "VG20's own curve\n(to 115 mo; unsupported\nabove ~8 y)",
                    (ref.age_months.iloc[-1] / 12.0 - 0.05, ref["median"].iloc[-1] + 4),
                    fontsize=7.5, color=ink2, va="bottom", ha="right",
                )
            ax.axvline(fit_hi_y, color=muted, lw=0.8, ls=":", zorder=1)
            ax.axhline(n_trials, color=muted, lw=0.8, ls=":", zorder=1)
            if r == 0:
                ax.set_title(col_title, fontsize=10.5, color=ink, loc="left", pad=8)
                ax.text(fit_hi_y + 0.08, 20, f"projection begins ({fit_hi_y:g} y)", fontsize=7.5, color=ink2)
            if r == 0 and c == 1:
                ax.text(1.05, n_trials + 8, f"checklist ceiling: {n_trials} words", fontsize=7.5, color=ink2, va="bottom")
            ax.set_xlim(0.9, x_max + 1.3)
            ax.set_ylim(0, n_trials + 50)
            ax.set_xticks(range(1, int(x_max) + 1))
            ax.set_yticks(range(0, n_trials, 100))
            ax.grid(True, axis="y", color=grid, lw=0.7, zorder=0)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(grid)
            ax.tick_params(colors=ink2, labelsize=8.5, length=0)
            if c == 0:
                ax.set_ylabel(f"Words {outcome} (of {n_trials})", fontsize=9.5, color=ink)
            if r == 1:
                ax.set_xlabel("Age (years)", fontsize=9.5, color=ink)

    handles = [Patch(color=hues["understood"]["bands"][i], label=f"{lbl} interval") for i, (_, lbl) in enumerate(levels)]
    handles += [
        Line2D([0], [0], color=hues["understood"]["line"], lw=2, label="median (families pooled)"),
        Line2D([0], [0], color=hues["understood"]["line"], lw=0.8, alpha=0.75, label="one growth family"),
        Line2D([0], [0], color=ink2, lw=1.4, ls=(0, (4, 2)), label="VG20's own fitted curve"),
        Line2D([0], [0], marker="o", color=muted, lw=0, markersize=3, alpha=0.6, label="observed administrations"),
    ]
    fig.legend(handles=handles, loc="lower center", fontsize=8, frameon=False, ncol=7, labelcolor=ink2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"Down syndrome vocabulary to {x_max:g} years: VG20 posterior to {fit_hi_y:g} years, parametric projection beyond",
        fontsize=12.5, color=ink, x=0.01, ha="left", y=0.995,
    )
    fig.text(
        0.01, 0.955,
        "Three saturating growth families (logistic, Gompertz, Hill) fitted per posterior draw to the model's own "
        f"curve over {fit_lo:g}-{fit_hi:g} months and continued upward, pooled with equal weight. Blue: understood; "
        "orange: spoken. Exploratory extrapolation, not a model output.",
        fontsize=8.5, color=ink2, ha="left", va="top", wrap=True,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", out_png)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=None, help="output root (default: the environment's)")
    ap.add_argument("--fit-lo", type=float, default=12.0, help="bottom of the fit window, months")
    ap.add_argument("--fit-hi", type=float, default=84.0, help="top of the fit window, months")
    ap.add_argument("--max-age", type=float, default=132.0, help="last projected age, months")
    ap.add_argument("--draws", type=int, default=2000, help="posterior draws after thinning")
    ap.add_argument("--children-per-draw", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. _fit72")
    ap.add_argument("--no-figure", action="store_true")
    fit_consumers.add_allow_stale_argument(ap)
    args = ap.parse_args()

    env.set_output_root(args.output_dir)
    fit_dir = fit_consumers.model_fit_dir(MODEL_KEY)
    fit_consumers.require_current_fit(
        MODEL_KEY, fit_dir, consumer=CONSUMER, allow_stale=args.allow_stale_fit
    )
    n_trials = MODEL_REGISTRY[MODEL_KEY].n_trials
    out_dir = os.path.join(env.output_root(), "experiments", "vg20-forward-projection")
    os.makedirs(out_dir, exist_ok=True)

    allq, fam_df, obs = project(args, os.path.join(fit_dir, "trace.nc"), n_trials, out_dir)
    if not args.no_figure:
        make_figure(
            allq, fam_df, obs, os.path.join(out_dir, f"vg20_projection_to_11y{args.tag}.png"),
            args.fit_hi, args.fit_lo, n_trials,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
