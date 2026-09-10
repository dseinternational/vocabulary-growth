#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How do our fitted vocabulary trajectories compare with Gompertz growth curves?

Day, Borovsky, Thal & Elison (2025), "Modeling longitudinal trajectories of word
production with the CDI", Developmental Science 28(4):e70036
(doi:10.1111/desc.70036), propose Gompertz curves as the functional form for CDI
inventory totals, with the asymptote fixed at the form's item count and two free
parameters:

    W(t) = A * exp(-exp(-k_g * (t - Ti)))

``Ti`` is the age of maximum growth and ``k_g`` the dimensionless growth rate;
the maximum rate in words per month is ``k_U = k_g * A / e``, reached exactly
when the child is at ``A / e`` = 36.79% of the asymptote. The study owner asked
on 2026-09-10 whether our fits can be evaluated against that form. Recorded in
``notes/202609101230-gompertz-comparison.md``.

One of the form's implications is **parameter-free**, so it tests the shape of
our fitted curves without fitting anything: at maximum growth the curve is at
exactly ``1/e`` of its asymptote, whatever ``k_g`` and ``Ti`` are (a logistic
would say 1/2). Where our curves put that point is therefore a direct reading of
the asymptote the form would need, and it is reported alongside the fits.

The form as published is reconstructed rather than copied: the PMC rendering of
their equation loses the exponents, so the expression above is the
two-parameter member of the family that satisfies both properties they state and
use -- the peak at ``A/e`` and a maximum rate of ``k_g A / e``. It reproduces
their published numbers: ``k_g = 0.161`` with ``A = 680`` gives 40.3 words per
month at the peak, which is their headline rate, and puts 250 words -- the
threshold whose age they report as 23.3 months -- at exactly ``Ti``, because
``680 / e = 250.2``.

What this harness does, for each of six curves (TD spoken VG11, TD understood
VG12, DS understood and spoken from VG20, and the single-level DS baselines VG01
and VG02):

1. Reads the stored population curve on the plot grid, trimmed to the model's own
   reporting age cap (``vocab_growth.reporting_ages``), at zero study and child
   effects -- the same curve the report quotes.
2. Per posterior draw, measures the age of maximum growth, the rate there in
   words per month, and **the fraction of the asymptote the curve has reached at
   that age**, which is the ``1/e`` test.
3. Per posterior draw, fits the paper's two-parameter Gompertz to our own curve
   with the asymptote fixed, and a three-parameter version with it free, and
   records ``k_g``, ``Ti`` and the RMSE in words.
4. Fits the paper's form to the **observed rows** by nonlinear least squares, as
   the paper does, with a child-clustered bootstrap for intervals -- so our
   posterior ``k_g`` and ``Ti`` can be read against a Gompertz fitted to the same
   data by their method, and against their published values.

Two denominators are used throughout, because ``k_g`` and the ``1/e`` test are
both defined relative to an assumed asymptote and our models score every count
against 810 whatever form produced it. ``n_trials`` is the modelled scale (810);
``observed_max`` is the largest count in the model's own frame, an empirical
proxy for the reachable ceiling of the instruments that frame is made of (680 for
the typically-developing spoken pool, which is the Words & Sentences form the
paper used, so that column is the directly comparable one). ``Ti`` is
scale-free and needs no such care.

This is a descriptive shape comparison and **not** a model comparison: it says
whether our fitted mean functions have Gompertz shape, not whether a Gompertz
mean would predict the data as well. That question needs the mean function
swapped inside the same likelihood and random-effect structure and scored by LOO,
which is a refit; see the note's §6.

Writes, under ``<output-root>/experiments/gompertz-comparison/``:

* ``shape_diagnostics.csv`` -- per curve: the peak age, peak rate, fraction of
  each asymptote at the peak, and the implied ``k_g``, with 89% intervals.
* ``curve_fits.csv`` -- per curve and Gompertz variant: ``k_g``, ``Ti``, the
  fitted asymptote and the RMSE against our own curve, with 89% intervals.
* ``data_fits.csv`` -- the paper's method on our rows: ``k_g``, ``Ti``, RMSE,
  with bootstrap intervals.
* ``gompertz_comparison.png`` -- the figure.

Usage::

    python scripts/experiments/gompertz_comparison.py
    python scripts/experiments/gompertz_comparison.py --draws 200 --bootstrap 50

Each fit is validated through ``vocab_growth.fit_consumers`` before it is read;
``--allow-stale-fit`` is the documented override. The one-off harness
conventions of ``scripts/experiments/README.md`` apply.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import least_squares

from vocab_growth import environment as env
from vocab_growth import fit_consumers, reporting_ages
from vocab_growth.models.definitions import MODEL_REGISTRY

CONSUMER = "gompertz_comparison.py"
CI = (0.055, 0.945)
#: 1/e -- the fraction of its asymptote a Gompertz has reached at maximum growth.
GOMPERTZ_PEAK_FRACTION = float(np.exp(-1.0))
#: Offset from ``Ti`` to the age at 0.9 A, in units of ``1/k_g``: 2.2504. Used to
#: report where each fitted curve would reach 0.9 of its asymptote -- a forward
#: statement, comparable with
#: ``notes/202609101115-vg11-forward-projection-to-5-years.md``.
GOMPERTZ_FALL = float(-np.log(-np.log(0.9)))
#: Published values, Day et al. (2025) doi:10.1111/desc.70036. Their Part 1 fits
#: one Gompertz to the whole Wordbank pool and reports ``k_g`` with the age at
#: 250 words, which is that curve's inflection because 250 is 680/e; Part 2's
#: group-level fits give a slightly higher ``k_g`` for the same pool, so read a
#: comparison against the band rather than against one value. `ku_*` and `ti_*`
#: are means over their individual-level fits, by cohort: EIRLI's three
#: diagnostic groups (no language disorder, status unknown, and a speech,
#: language, learning or reading diagnosis at 4-7 years, latest of the three) and
#: BCP, whose sufficient-data group sits inside the same range.
PAPER = {
    "kg_wordbank_part1": 0.161,
    "kg_wordbank_part2": 0.17,
    "ti_wordbank": 23.3,
    "kg_bcp": 0.15,
    "kg_eirli": 0.17,
    "ku_range": (34.3, 39.3),
    "ti_range": (22.1, 25.0),
    "ti_bcp": 24.3,
}


@dataclass(frozen=True)
class CurveSpec:
    """One population curve to evaluate."""

    model_key: str
    outcome: str
    population: str
    label: str
    #: ``None`` for the single-outcome engines' ``p_plot``; for the joint engine,
    #: the understood proportion and (for spoken) the production ratio to
    #: multiply it by.
    understood_var: str | None = None
    ratio_var: str | None = None
    role: str = ""


CURVES = (
    CurveSpec("vg12", "understood", "TD", "VG12 TD understood", role="TD reference"),
    CurveSpec("vg11", "spoken", "TD", "VG11 TD spoken", role="TD reference"),
    CurveSpec(
        "vg20", "understood", "DS", "VG20 DS understood",
        understood_var="p_u_plot", role="model of record",
    ),
    CurveSpec(
        "vg20", "spoken", "DS", "VG20 DS spoken",
        understood_var="p_u_plot", ratio_var="q_plot", role="model of record",
    ),
    CurveSpec("vg02", "understood", "DS", "VG02 DS understood", role="development step"),
    CurveSpec("vg01", "spoken", "DS", "VG01 DS spoken", role="development step"),
)


# ------------------------------------------------------------------ the form
def gompertz(t, k_g, t_i, amax):
    """Day et al.'s equation: ``A exp(-exp(-k (t - Ti)))``."""
    with np.errstate(over="ignore"):
        return amax * np.exp(-np.exp(-k_g * (np.asarray(t, dtype=float) - t_i)))


#: Bounds on (k_g, Ti) and, when the asymptote is free, on its logit share of the
#: checklist. A fit that lands on one is reported as unidentified rather than as a
#: number: with the asymptote fixed at a value the data never approach, the least
#: squares problem is genuinely degenerate (k_g -> 0 with Ti -> -inf), which is
#: itself one of this comparison's findings.
K_BOUNDS = (1e-4, 2.0)
TI_BOUNDS = (0.0, 240.0)
SHARE_BOUNDS = (-8.0, 8.0)
_STARTS = ((0.15, 24.0), (0.05, 40.0), (0.30, 18.0), (0.02, 70.0))


def fit_gompertz(t, y, amax, *, free_amax=False, init=None):
    """Least-squares fit of the form to ``(t, y)``; ``y`` on the same scale as ``amax``.

    Multi-start and bounded, because the fixed-asymptote problem is ill-posed
    whenever the data stop well below the asymptote. Returns
    ``(k_g, Ti, asymptote, rmse, ok)``, with ``ok`` false when the best solution
    sits on a bound.
    """
    starts = list(_STARTS) if init is None else [init, *_STARTS]
    if free_amax:
        share0 = float(np.log(0.9 / 0.1))

        def resid(th):
            return gompertz(t, th[0], th[1], amax / (1.0 + np.exp(-th[2]))) - y

        lo = [K_BOUNDS[0], TI_BOUNDS[0], SHARE_BOUNDS[0]]
        hi = [K_BOUNDS[1], TI_BOUNDS[1], SHARE_BOUNDS[1]]
        candidates = [[*s, share0] for s in starts]
    else:
        def resid(th):
            return gompertz(t, th[0], th[1], amax) - y

        lo = [K_BOUNDS[0], TI_BOUNDS[0]]
        hi = [K_BOUNDS[1], TI_BOUNDS[1]]
        candidates = [list(s) for s in starts]

    best = None
    for x0 in candidates:
        x0 = np.clip(x0, lo, hi)
        try:
            res = least_squares(resid, x0, bounds=(lo, hi), method="trf", max_nfev=4000)
        except Exception:  # noqa: BLE001 - a failed start is skipped
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        return np.nan, np.nan, np.nan, np.nan, False
    k_g, t_i = float(best.x[0]), float(best.x[1])
    fitted = amax / (1.0 + np.exp(-float(best.x[2]))) if free_amax else float(amax)
    ok = (
        k_g > K_BOUNDS[0] * 1.01
        and TI_BOUNDS[0] + 0.01 < t_i < TI_BOUNDS[1] - 1.0
    )
    return k_g, t_i, fitted, _rmse(best), ok


def _rmse(res):
    return float(np.sqrt(np.mean(res.fun**2)))


def interval(values):
    v = np.asarray(values, dtype=float)
    return float(np.median(v)), float(np.quantile(v, CI[0])), float(np.quantile(v, CI[1]))


def _row(prefix, values):
    med, lo, hi = interval(values)
    return {f"{prefix}": med, f"{prefix}_lo": lo, f"{prefix}_hi": hi}


# ------------------------------------------------------------------ curves
def read_curve(spec, args):
    """The population proportion curve per draw, its ages, and the observed rows."""
    fit_dir = fit_consumers.model_fit_dir(spec.model_key)
    fit_consumers.require_current_fit(
        spec.model_key, fit_dir, consumer=CONSUMER, allow_stale=args.allow_stale_fit
    )
    definition = MODEL_REGISTRY[spec.model_key]
    trace_path = os.path.join(fit_dir, "trace.nc")
    cd = xr.open_dataset(trace_path, group="constant_data")
    post = xr.open_dataset(trace_path, group="posterior")
    od = xr.open_dataset(trace_path, group="observed_data")

    def flat(name):
        return post[name].stack(sample=("chain", "draw")).transpose("sample", ...).values

    n_total = post.sizes["chain"] * post.sizes["draw"]
    step = max(1, n_total // args.draws)
    idx = np.arange(0, n_total, step)[: args.draws]

    if spec.understood_var is None:
        prop = flat("p_plot")[idx]
    else:
        prop = flat(spec.understood_var)[idx]
        if spec.ratio_var is not None:
            prop = prop * flat(spec.ratio_var)[idx]

    t_grid = cd["X_plot"].values.astype(float).ravel()
    cap = reporting_ages.max_age_for(
        definition, reporting_ages.ReportedQuantity(spec.outcome)
    )
    keep = t_grid <= (cap if cap is not None else t_grid.max())
    keep &= t_grid >= args.fit_lo

    x_obs = cd["X_obs"].values.astype(float).ravel()
    # The single-level baselines carry no subject index, having no child
    # effects; their pools are the joint model's, so the data fit is taken
    # there (see `main`) and this is None only as a safety net.
    subj_obs = cd["subject_obs"].values.ravel() if "subject_obs" in cd else None
    if spec.understood_var is None:
        y_obs = od[next(iter(od.data_vars))].values.astype(float).ravel()
    else:
        short = "u" if spec.outcome == "understood" else "s"
        mask = cd[f"obs_{short}_mask"].values.astype(bool)
        y_obs = od[f"y_{short}_obs"].values.astype(float).ravel()
        x_obs = x_obs[mask]
        subj_obs = subj_obs[mask]
    keep_obs = x_obs <= (cap if cap is not None else np.inf)

    return dict(
        prop=prop[:, keep],
        ages=t_grid[keep],
        n_trials=float(definition.n_trials),
        cap=float(cap) if cap is not None else float(t_grid.max()),
        x_obs=x_obs[keep_obs],
        y_obs=y_obs[keep_obs],
        subj_obs=None if subj_obs is None else subj_obs[keep_obs],
        n_draws=len(idx),
        n_total=n_total,
    )


def shape_diagnostics(spec, curve):
    """The parameter-free tests, per draw, plus the implied ``k_g``."""
    ages, prop, n_trials = curve["ages"], curve["prop"], curve["n_trials"]
    observed_max = float(curve["y_obs"].max())
    rate = np.gradient(prop, ages, axis=1)  # per month, proportion units
    i_peak = np.argmax(rate, axis=1)
    at_edge = (i_peak == 0) | (i_peak == len(ages) - 1)
    peak_age = ages[i_peak]
    peak_rate = rate[np.arange(len(i_peak)), i_peak]
    peak_prop = prop[np.arange(len(i_peak)), i_peak]

    out = {
        "curve": spec.label, "population": spec.population, "outcome": spec.outcome,
        "model": spec.model_key.upper(), "role": spec.role,
        "n_draws": curve["n_draws"], "age_lo": float(ages.min()), "age_hi": float(ages.max()),
        "n_trials": n_trials, "observed_max": observed_max,
        "peak_at_grid_edge_share": float(at_edge.mean()),
    }
    # our own mean function against the same rows, for the like-for-like
    # comparison the deduplicated data-fit table cannot carry per model. It is
    # the median-child curve at zero study and child effects, so it sits below
    # the row mean by construction; see the note.
    ours = np.interp(curve["x_obs"], ages, np.median(prop, axis=0) * n_trials)
    out["our_curve_rmse_vs_rows_words"] = float(
        np.sqrt(np.mean((ours - curve["y_obs"]) ** 2))
    )
    out.update(_row("peak_age_months", peak_age))
    out.update(_row("peak_rate_words_per_month", peak_rate * n_trials))
    # the 1/e test, on each denominator
    out.update(_row("peak_fraction_of_n_trials", peak_prop))
    out.update(_row("peak_fraction_of_observed_max", peak_prop * n_trials / observed_max))
    # a Gompertz with our peak rate would have this k_g, on each denominator
    out.update(_row("implied_kg_n_trials", np.e * peak_rate))
    out.update(_row("implied_kg_observed_max", np.e * peak_rate * n_trials / observed_max))
    out["gompertz_peak_fraction"] = GOMPERTZ_PEAK_FRACTION
    return out


def curve_fits(spec, curve):
    """Fit the form to our own curve, per draw, three ways."""
    ages, prop, n_trials = curve["ages"], curve["prop"], curve["n_trials"]
    observed_max = float(curve["y_obs"].max())
    variants = {
        "fixed_A_n_trials": dict(amax=n_trials, free_amax=False),
        "fixed_A_observed_max": dict(amax=observed_max, free_amax=False),
        "free_A": dict(amax=n_trials, free_amax=True),
    }
    rows = []
    for name, kwargs in variants.items():
        acc = {"k_g": [], "T_i": [], "amax": [], "rmse": []}
        n_ok = 0
        for d in range(prop.shape[0]):
            k_g, t_i, amax, rmse, ok = fit_gompertz(ages, prop[d] * n_trials, **kwargs)
            n_ok += ok
            if not ok:
                continue
            acc["k_g"].append(k_g)
            acc["T_i"].append(t_i)
            acc["amax"].append(amax)
            acc["rmse"].append(rmse)
        row = {
            "curve": spec.label, "model": spec.model_key.upper(),
            "population": spec.population, "outcome": spec.outcome, "variant": name,
            "identified_share": n_ok / prop.shape[0],
        }
        for key, values in acc.items():
            row.update(_row(key, values or [np.nan]))
        row["amax_share_of_n_trials"] = row["amax"] / n_trials
        row.update(_row("age_at_0.9_asymptote_months", [
            t + GOMPERTZ_FALL / k for t, k in zip(acc["T_i"], acc["k_g"], strict=True)
        ] or [np.nan]))
        row["k_U_words_per_month"] = row["k_g"] * row["amax"] / np.e
        rows.append(row)
    return rows


def _child_resample(subj, rng):
    """Row indices for one bootstrap resample of **children**, not rows.

    The Down syndrome pools average several administrations per child, so a row
    bootstrap would treat repeated measurements as independent and return
    intervals that are too narrow.
    """
    if subj is None:
        raise RuntimeError(
            "no subject index in this trace: fit the rows on a model that has one "
            "(the pools are shared, see `main`)"
        )
    by_child = {}
    for i, s in enumerate(subj):
        by_child.setdefault(s, []).append(i)
    keys = list(by_child)
    picked = rng.integers(0, len(keys), len(keys))
    return np.concatenate([by_child[keys[k]] for k in picked])


def data_fits(spec, curve, args, rng):
    """The paper's method: nonlinear least squares on the observed rows."""
    x, y = curve["x_obs"], curve["y_obs"]
    n_trials, observed_max = curve["n_trials"], float(y.max())
    # Our own mean function evaluated at the same row ages, for a like-for-like
    # RMSE. It is the *median*-child curve at zero study and child effects, so it
    # sits below the row mean by construction and this comparison is not fair to
    # it; see the note. Reported so the Gompertz's two parameters can be scored
    # against a flexible mean on the same rows at all.
    ours = np.interp(x, curve["ages"], np.median(curve["prop"], axis=0) * n_trials)
    our_rmse = float(np.sqrt(np.mean((ours - y) ** 2)))
    rows = []
    for name, amax in (("fixed_A_n_trials", n_trials), ("fixed_A_observed_max", observed_max)):
        k_g, t_i, _, rmse, ok = fit_gompertz(x, y, amax)
        boot = {"k_g": [], "T_i": []}
        for _ in range(args.bootstrap):
            take = _child_resample(curve["subj_obs"], rng)
            bk, bt, _, _, bok = fit_gompertz(x[take], y[take], amax, init=(k_g, t_i))
            if not bok:
                continue
            boot["k_g"].append(bk)
            boot["T_i"].append(bt)
        row = {
            "curve": spec.label, "model": spec.model_key.upper(),
            "population": spec.population, "outcome": spec.outcome, "variant": name,
            "n_rows": int(len(x)), "amax": float(amax), "identified": bool(ok),
            "k_g": k_g if ok else np.nan, "T_i": t_i if ok else np.nan,
            "rmse_words": rmse if ok else np.nan,
            "k_U_words_per_month": (k_g * amax / np.e) if ok else np.nan,
            "our_curve_rmse_words": our_rmse,
        }
        if boot["k_g"]:
            for key, values in boot.items():
                _, lo, hi = interval(values)
                row[f"{key}_boot_lo"], row[f"{key}_boot_hi"] = lo, hi
        rows.append(row)
    return rows


# ------------------------------------------------------------------ figure
def make_figure(shape, fits, curves, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ink, ink2, muted, grid = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
    hues = {"understood": "#1c5cab", "spoken": "#c44a1e"}
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, spec in zip(axes.ravel(), CURVES, strict=True):
        curve = curves[spec.label]
        sh = shape[shape.curve == spec.label].iloc[0]
        ft = fits[(fits.curve == spec.label) & (fits.variant == "fixed_A_observed_max")].iloc[0]
        ages, prop, n_trials = curve["ages"], curve["prop"], curve["n_trials"]
        colour = hues[spec.outcome]
        ax.set_facecolor("#fcfcfb")
        ax.scatter(curve["x_obs"], curve["y_obs"], s=5, color=muted,
                   alpha=0.10 if len(curve["x_obs"]) > 3000 else 0.35, lw=0, zorder=2)
        lo, hi = np.quantile(prop * n_trials, CI, axis=0)
        ax.fill_between(ages, lo, hi, color=colour, alpha=0.18, lw=0, zorder=3)
        ax.plot(ages, np.median(prop, axis=0) * n_trials, color=colour, lw=2, zorder=5,
                label="our fitted curve")
        ax.plot(ages, gompertz(ages, ft["k_g"], ft["T_i"], ft["amax"]), color=ink2, lw=1.5,
                ls=(0, (4, 2)), zorder=6, label="best-fit Gompertz")
        ax.axhline(sh["observed_max"] * GOMPERTZ_PEAK_FRACTION, color=muted, lw=0.8,
                   ls=(0, (1, 3)), zorder=1)
        ax.scatter([sh["peak_age_months"]],
                   [sh["peak_fraction_of_n_trials"] * n_trials], s=34, color=colour,
                   edgecolor="#fcfcfb", lw=1.0, zorder=7)
        ax.set_title(f"{spec.label}\n{spec.role}", fontsize=9.5, color=ink, loc="left", pad=6)
        ax.text(
            0.03, 0.97,
            f"peak {sh['peak_age_months']:.1f} mo, {sh['peak_rate_words_per_month']:.0f} words/mo\n"
            f"at {sh['peak_fraction_of_observed_max']:.3f} of ceiling (Gompertz: 0.368)\n"
            f"$k_g$ {ft['k_g']:.3f}, $T_i$ {ft['T_i']:.1f} mo, RMSE {ft['rmse']:.0f} words",
            transform=ax.transAxes, fontsize=7.5, color=ink2, ha="left", va="top",
        )
        ax.set_xlim(6, ages.max() + 2)
        ax.set_ylim(0, max(sh["observed_max"], (prop * n_trials).max()) * 1.12)
        ax.grid(True, axis="y", color=grid, lw=0.7, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8.5, length=0)
        ax.set_xlabel("Age (months)", fontsize=9, color=ink)
        ax.set_ylabel(f"Words {spec.outcome} (of 810)", fontsize=9, color=ink)

    handles = [
        Line2D([0], [0], color=hues["understood"], lw=2, label="our fitted population curve (89% band)"),
        Line2D([0], [0], color=ink2, lw=1.5, ls=(0, (4, 2)), label="best-fit Gompertz, asymptote at the frame's observed maximum"),
        Line2D([0], [0], marker="o", color=hues["understood"], lw=0, markersize=5, label="our curve's maximum-growth point"),
        Line2D([0], [0], color=muted, lw=0.8, ls=(0, (1, 3)), label="1/e of that asymptote, where a Gompertz peaks"),
        Line2D([0], [0], marker="o", color=muted, lw=0, markersize=3, alpha=0.6, label="observed administrations"),
    ]
    fig.legend(handles=handles, loc="lower center", fontsize=8, frameon=False, ncol=3,
               labelcolor=ink2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        "Our fitted vocabulary trajectories against Gompertz growth curves (Day et al. 2025)",
        fontsize=12.5, color=ink, x=0.01, ha="left", y=0.995,
    )
    fig.text(
        0.01, 0.955,
        "Each panel: our model's population curve for the median child over its own reporting age range, the best two-parameter Gompertz "
        "fitted to it, and the parameter-free test — a Gompertz reaches its maximum growth at exactly 1/e of its asymptote. "
        "Descriptive shape comparison, not a model comparison.",
        fontsize=8.5, color=ink2, ha="left", va="top", wrap=True,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", out_png)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=None, help="output root (default: the environment's)")
    ap.add_argument("--fit-lo", type=float, default=8.0, help="bottom of the comparison window, months")
    ap.add_argument("--draws", type=int, default=500, help="posterior draws after thinning")
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples for the data fits")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--no-figure", action="store_true")
    fit_consumers.add_allow_stale_argument(ap)
    args = ap.parse_args()

    env.set_output_root(args.output_dir)
    out_dir = os.path.join(env.output_root(), "experiments", "gompertz-comparison")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    curves, shape_rows, fit_rows, data_rows = {}, [], [], []
    seen_pairs = set()
    for spec in CURVES:
        curve = read_curve(spec, args)
        curves[spec.label] = curve
        print(
            f"{spec.label:22s} {curve['n_draws']:4d} draws of {curve['n_total']}, "
            f"ages {curve['ages'].min():.0f}-{curve['ages'].max():.0f} mo "
            f"(cap {curve['cap']:.0f}), {len(curve['x_obs'])} rows, "
            f"observed max {curve['y_obs'].max():.0f} of {curve['n_trials']:.0f}"
        )
        shape_rows.append(shape_diagnostics(spec, curve))
        fit_rows.extend(curve_fits(spec, curve))
        # The Gompertz fitted to the rows is a property of the pool, not of the
        # model: VG01/VG02 consume exactly the rows VG20 does (1,225 understood
        # and 1,394 spoken under the same masking rules), so it is fitted once
        # per population and outcome, on the first spec that claims the pair --
        # which CURVES orders to be the TD reference or the model of record.
        pair = (spec.population, spec.outcome)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            data_rows.extend(data_fits(spec, curve, args, rng))

    shape = pd.DataFrame(shape_rows)
    fits = pd.DataFrame(fit_rows)
    data = pd.DataFrame(data_rows)
    shape.to_csv(os.path.join(out_dir, f"shape_diagnostics{args.tag}.csv"), index=False)
    fits.to_csv(os.path.join(out_dir, f"curve_fits{args.tag}.csv"), index=False)
    data.to_csv(os.path.join(out_dir, f"data_fits{args.tag}.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\n== the parameter-free shape test: where does maximum growth happen?")
    print(
        shape[[
            "curve", "peak_age_months", "peak_age_months_lo", "peak_age_months_hi",
            "peak_rate_words_per_month", "peak_fraction_of_n_trials",
            "peak_fraction_of_observed_max", "peak_at_grid_edge_share",
            "our_curve_rmse_vs_rows_words",
        ]].round(3).to_string(index=False)
    )
    print(f"   (a Gompertz peaks at {GOMPERTZ_PEAK_FRACTION:.3f} of its asymptote; a logistic at 0.500)")

    print("\n== the Gompertz fitted to our own curve")
    print(
        fits[[
            "curve", "variant", "k_g", "k_g_lo", "k_g_hi", "T_i", "T_i_lo", "T_i_hi",
            "amax_share_of_n_trials", "k_U_words_per_month", "rmse", "rmse_lo", "rmse_hi",
            "age_at_0.9_asymptote_months", "identified_share",
        ]].round(3).to_string(index=False)
    )

    print("\n== the paper's method on our rows (nls, child-clustered bootstrap)")
    cols = [c for c in (
        "curve", "variant", "n_rows", "amax", "identified", "k_g", "k_g_boot_lo", "k_g_boot_hi",
        "T_i", "T_i_boot_lo", "T_i_boot_hi", "k_U_words_per_month", "rmse_words",
        "our_curve_rmse_words",
    ) if c in data.columns]
    print(data[cols].round(3).to_string(index=False))

    print("\n== published values, Day et al. (2025) doi:10.1111/desc.70036")
    print(
        f"   pooled Gompertz on English Wordbank (A = 680): k_g {PAPER['kg_wordbank_part1']} "
        f"(Part 1) to {PAPER['kg_wordbank_part2']} (Part 2), T_i {PAPER['ti_wordbank']} mo\n"
        f"   their other cohorts' pooled k_g: BCP {PAPER['kg_bcp']}, EIRLI {PAPER['kg_eirli']}\n"
        f"   individual-level group means: k_U {PAPER['ku_range'][0]}-{PAPER['ku_range'][1]} words/mo, "
        f"T_i {PAPER['ti_range'][0]}-{PAPER['ti_range'][1]} mo (diagnosed group latest), "
        f"BCP {PAPER['ti_bcp']} mo"
    )

    if not args.no_figure:
        make_figure(shape, fits, curves, os.path.join(out_dir, f"gompertz_comparison{args.tag}.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
