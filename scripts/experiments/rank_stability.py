#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Do children keep their relative vocabulary standing over time?

Developmental **tracking**: if a child is ahead of others their age at one
assessment, are they still ahead later? The repeated-measures structure answers
this directly, and no fitted model can — VG08-VG10 give each child a *constant*
random intercept, which assumes perfect tracking by construction and therefore
cannot test it.

Method
------
Raw counts cannot be compared across ages or instruments, so each observation is
scored as a residual:

1. Take the count as a proportion of that administration's own form ceiling
   (``survey_vocab_max``), clipped away from 0 and 1 by half an item.
2. Take its logit, which is the scale the models themselves work on and which
   linearises the trajectory's middle.
3. Regress on a cubic in standardised age plus study fixed effects, and keep the
   residual.

The residual is therefore "how this child compares with others of the same age
measured by the same study". Study fixed effects matter: instrument, country and
recruitment differ enough between sources that an unadjusted comparison largely
recovers the study, not the child.

Three summaries are reported.

* **ICC** — the between-child share of residual variance. This is the tracking
  coefficient in variance terms: 1.0 means a child's standing never changes,
  0.0 means each assessment is independent of the last.
* **Lag-binned correlation** — Spearman between pairs of observations on the same
  child, binned by the gap between them. A stable trait gives a flat curve; a
  trait that drifts gives decay.
* **Quartile transitions** — the same thing in a form a practitioner can read.

All intervals are **cluster bootstrap** over children (children are the
independent unit, not observations), which is why they are wider than a naive
bootstrap over rows would give.

Attenuation
-----------
Observed correlations are attenuated by measurement error: a child's count is one
noisy realisation, so two assessments correlate less than their underlying
standings do. A binomial lower bound on the error variance is reported, and with
it a disattenuated correlation. Because that bound ignores the extra-binomial
dispersion the models fit (``kappa``), it *understates* error and therefore
*understates* the correction — the disattenuated figure is a lower bound on true
trait stability, not an estimate of it.

Usage::

    python scripts/experiments/rank_stability.py            # DS and TD, both outcomes
    python scripts/experiments/rank_stability.py --boot 1000
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from vocab_growth import data_utils as du
from vocab_growth.models.definitions import (
    ENGLISH_AND_ROMANCE_LANGUAGES,
    Population,
)

LAG_BINS = [(1, 6), (6, 12), (12, 24), (24, 60)]
RNG = np.random.default_rng(47)


# ----------------------------------------------------------------- scoring


def _design(age: np.ndarray, study: pd.Series) -> np.ndarray:
    a = (age - age.mean()) / age.std()
    dummies = pd.get_dummies(study, drop_first=True).values.astype(float)
    return np.column_stack([np.ones(len(a)), a, a**2, a**3, dummies])


def adjusted_scores(population: str, outcome: str) -> pd.DataFrame:
    """Age- and study-adjusted logit scores, one row per administration."""
    if population == "ds":
        d = du.load_combined_data()
        d = d[d[outcome].notna() & d.survey_vocab_max.notna()].copy()
        ceiling = d.survey_vocab_max.values.astype(float)
    else:
        d = du.load_data(
            Population.TYPICALLY_DEVELOPING,
            ["age", outcome, "study", "subject_id", "form"],
            languages=ENGLISH_AND_ROMANCE_LANGUAGES,
        )
        d = d[d[outcome].notna()].copy()
        # TD rows carry no per-row ceiling; the form does. Use the observed
        # maximum per form, which is the ceiling the instrument admits.
        ceiling = d.groupby("form")[outcome].transform("max").values.astype(float)
        ceiling = np.maximum(ceiling, d[outcome].values + 1)

    p = np.clip(d[outcome].values / ceiling, 0.5 / ceiling, 1 - 0.5 / ceiling)
    d["logit"] = np.log(p / (1 - p))
    d["p_hat"] = p
    d["n_items"] = ceiling

    X = _design(d.age.values.astype(float), d.study)
    beta, *_ = np.linalg.lstsq(X, d["logit"].values, rcond=None)
    d["resid"] = d["logit"].values - X @ beta
    # A child key that cannot collide across studies.
    d["child"] = d.study.astype(str) + "|" + d.subject_id.astype(str)
    return d.sort_values(["child", "age"])


# ----------------------------------------------------------------- measures


def icc(d: pd.DataFrame, key: str = "child") -> float:
    """Between-child share of residual variance (observations, not pairs)."""
    m = d.groupby(key)["resid"].transform("mean")
    vb = np.var(m, ddof=1)
    vw = np.var(d["resid"].values - m.values, ddof=1)
    return float(vb / (vb + vw)) if (vb + vw) > 0 else np.nan


def icc_ci(d: pd.DataFrame, n_boot: int) -> tuple[float, float]:
    """Cluster bootstrap CI for the ICC.

    A child drawn twice must count twice, so each draw gets a fresh key. An
    earlier version subset the frame with ``isin(unique())``, which silently
    dropped the duplicates and sampled WITHOUT replacement -- it returned
    intervals that did not contain their own point estimate, which is how the
    error was caught.
    """
    rep = d.groupby("child").filter(lambda g: len(g) >= 2)
    if rep.empty:
        return (np.nan, np.nan)
    by_child = {c: g for c, g in rep.groupby("child", sort=False)}
    children = list(by_child)
    vals = []
    for _ in range(n_boot):
        pick = RNG.choice(children, size=len(children), replace=True)
        frames = []
        for k, c in enumerate(pick):
            g = by_child[c].copy()
            g["boot_key"] = k          # duplicates become distinct clusters
            frames.append(g)
        v = icc(pd.concat(frames, ignore_index=True), key="boot_key")
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 5.5)), float(np.percentile(vals, 94.5)))


def within_child_pairs(d: pd.DataFrame) -> pd.DataFrame:
    """Every ordered pair of observations on the same child, with its lag."""
    out = []
    for child, g in d.groupby("child", sort=False):
        if len(g) < 2:
            continue
        a = g["age"].values
        r = g["resid"].values
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                out.append((child, a[j] - a[i], r[i], r[j]))
    return pd.DataFrame(out, columns=["child", "lag", "r0", "r1"])


def _spearman(x, y) -> float:
    if len(x) < 5:
        return np.nan
    return float(stats.spearmanr(x, y).statistic)


def cluster_boot(pairs: pd.DataFrame, stat, n_boot: int) -> tuple[float, float]:
    """Percentile CI, resampling CHILDREN (the independent unit)."""
    children = pairs["child"].unique()
    by_child = {c: g for c, g in pairs.groupby("child", sort=False)}
    vals = []
    for _ in range(n_boot):
        pick = RNG.choice(children, size=len(children), replace=True)
        b = pd.concat([by_child[c] for c in pick], ignore_index=True)
        v = stat(b)
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 5.5)), float(np.percentile(vals, 94.5)))


def reliability_bound(d: pd.DataFrame) -> float:
    """Binomial lower bound on measurement error, as a reliability.

    On the logit scale the sampling variance of an observed proportion is
    approximately ``1 / (n p (1 - p))``. Ignoring the extra-binomial dispersion
    the models fit, this UNDERSTATES error, so the reliability it gives is an
    upper bound and the resulting disattenuation is conservative.
    """
    err = 1.0 / (d["n_items"].values * d["p_hat"].values * (1 - d["p_hat"].values))
    total = np.var(d["resid"].values, ddof=1)
    return float(max(0.0, (total - np.mean(err)) / total))


AGE_BANDS = [(8, 18), (18, 30), (30, 42), (42, 60), (60, 84), (84, 120)]


def spread_by_age_band(d: pd.DataFrame) -> pd.DataFrame:
    """Non-measurement spread of the adjusted score, by age band.

    The quantity Proposal A1 exists to estimate: does the spread between children
    widen with age? Each band's observed residual SD is decomposed by subtracting
    the mean binomial sampling variance from :func:`reliability_bound`, leaving
    between-child *plus* occasion variation — not between-child alone, because a
    single cross-sectional band cannot separate the two.

    Read on the logit scale, which is the scale the models work on and the one
    the residuals are defined on. It compresses near 0 and 1, so a band whose
    counts sit against the floor or the ceiling understates the spread; the
    ``measurement SD`` column is what says whether that is happening.
    """
    rows = []
    for lo, hi in AGE_BANDS:
        band = d[(d.age >= lo) & (d.age < hi)]
        if len(band) < 30:
            continue
        observed = float(np.std(band["resid"].values, ddof=1))
        err = 1.0 / (
            band["n_items"].values * band["p_hat"].values * (1 - band["p_hat"].values)
        )
        measurement = float(np.sqrt(np.mean(err)))
        trait = float(np.sqrt(max(0.0, observed**2 - np.mean(err))))
        rows.append(
            {
                "band": f"{lo}-{hi}",
                "n": len(band),
                "children": int(band.child.nunique()),
                "sd_observed": observed,
                "sd_measurement": measurement,
                "sd_non_measurement": trait,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------- within-child structure (ML)


def _child_blocks(d: pd.DataFrame, min_obs: int = 1) -> list[tuple]:
    """Per-child ``(ages, residuals, known sampling variances)``, ages ascending."""
    out = []
    for _, g in d.groupby("child"):
        if len(g) < min_obs:
            continue
        g = g.sort_values("age")
        v = 1.0 / (
            g["n_items"].values * g["p_hat"].values * (1 - g["p_hat"].values)
        )
        out.append((g.age.values.astype(float), g["resid"].values, v))
    return out


def _neg_loglik(blocks, centre, tau0, tau1, rho01, sigma_occ, ell=None, tau_tran=None):
    """Gaussian log-likelihood of the adjusted scores under one child structure.

    Each child is one multivariate normal. The diagonal carries the **known**
    binomial sampling variance per observation plus a free occasion term, so the
    child structure is estimated against a measurement model rather than
    absorbing it. Passing ``ell``/``tau_tran`` adds a mean-reverting
    (Ornstein-Uhlenbeck) component instead of, or alongside, the slope.
    """
    total = 0.0
    for a, r, v in blocks:
        c = a - centre
        S = (
            tau0**2
            + rho01 * tau0 * tau1 * (c[:, None] + c[None, :])
            + tau1**2 * np.outer(c, c)
        )
        if ell is not None:
            S = S + tau_tran**2 * np.exp(-np.abs(a[:, None] - a[None, :]) / ell)
        S = S + np.diag(v + sigma_occ**2)
        try:
            chol = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            return 1e10
        z = np.linalg.solve(chol, r)
        total += 2 * np.log(np.diag(chol)).sum() + z @ z
    return 0.5 * total


def fit_child_structure(blocks, centre, *, slope=True, fix_rho=None):
    """Maximum-likelihood fit of one child structure. Returns ``(params, negll)``.

    ``slope=False`` is the constant-intercept baseline the models of record
    carry; ``fix_rho=1.0`` is Proposal A1 (one deviate scaled by an age function
    is a rank-one covariance, i.e. perfect rank correlation). Several starting
    values are tried because the slope scale is small and the surface is flat
    near ``tau1 = 0``.
    """
    from scipy.optimize import minimize

    def objective(theta):
        tau0 = np.exp(theta[0])
        if not slope:
            return _neg_loglik(blocks, centre, tau0, 0.0, 0.0, np.exp(theta[1]))
        tau1, sigma_occ = np.exp(theta[1]), np.exp(theta[2])
        rho = fix_rho if fix_rho is not None else np.tanh(theta[3])
        return _neg_loglik(blocks, centre, tau0, tau1, rho, sigma_occ)

    if not slope:
        starts = [[np.log(1.2), np.log(0.6)]]
    else:
        tail = [] if fix_rho is not None else [0.0]
        starts = [
            [np.log(1.2), np.log(s), np.log(0.6), *tail]
            for s in (0.008, 0.02, 0.05)
        ]
    best = None
    for start in starts:
        r = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 10000},
        )
        if best is None or r.fun < best.fun:
            best = r
    params = {"tau0": float(np.exp(best.x[0]))}
    if slope:
        params["tau1"] = float(np.exp(best.x[1]))
        params["sigma_occ"] = float(np.exp(best.x[2]))
        params["rho01"] = float(fix_rho if fix_rho is not None else np.tanh(best.x[3]))
    else:
        params["tau1"] = 0.0
        params["rho01"] = 0.0
        params["sigma_occ"] = float(np.exp(best.x[1]))
    return params, float(best.fun)


def _block_covariance(a, v, centre, params):
    """One child's model covariance under a fitted structure."""
    c = a - centre
    tau0, tau1, rho01 = params["tau0"], params["tau1"], params["rho01"]
    return (
        tau0**2
        + rho01 * tau0 * tau1 * (c[:, None] + c[None, :])
        + tau1**2 * np.outer(c, c)
        + np.diag(v + params["sigma_occ"] ** 2)
    )


def _simulate_under(blocks, centre, params, rng):
    """Regenerate every child's adjusted scores from a fitted structure.

    The design is kept exactly -- the same children, the same ages, the same
    known binomial sampling variances -- and only the residuals are redrawn, so
    the simulated data have the study's own singleton/repeater mix.
    """
    simulated = []
    for a, _r, v in blocks:
        chol = np.linalg.cholesky(_block_covariance(a, v, centre, params))
        simulated.append((a, chol @ rng.standard_normal(len(a)), v))
    return simulated


def bootstrap_null(blocks, centre, *, null_kwargs, alt_kwargs, draws, seed):
    """Parametric-bootstrap null distribution of ``2 * delta logL``.

    **Why this and not a chi-square (issue #233).** Both comparisons this script
    makes sit on a boundary of the parameter space, where Wilks' theorem does not
    hold and an ordinary chi-square reference distribution is invalid:

    - ``tau1 = 0`` is a variance component at zero. The usual consequence is a
      mixture of chi-squares rather than a single one, and here it is worse than
      that -- at ``tau1 = 0`` the correlation ``rho01`` is not identified at all,
      so one of the two nominal degrees of freedom does not exist under the null.
    - ``rho01 = 1`` is a correlation at its own boundary.

    Reporting "2dlogL = 36.05 (2 df)" invited a p-value that could not be
    computed from it. The bootstrap replaces the reference distribution rather
    than the statistic: fit the null, simulate from it on the study's own design,
    refit both structures to each simulation, and read the statistic's null
    distribution off the replicates. Returns ``(observed, p_value, replicates)``.
    """
    null_params, ll_null = fit_child_structure(blocks, centre, **null_kwargs)
    _, ll_alt = fit_child_structure(blocks, centre, **alt_kwargs)
    observed = 2.0 * (ll_null - ll_alt)

    rng = np.random.default_rng(seed)
    replicates = []
    for _ in range(draws):
        simulated = _simulate_under(blocks, centre, null_params, rng)
        _, sim_null = fit_child_structure(simulated, centre, **null_kwargs)
        _, sim_alt = fit_child_structure(simulated, centre, **alt_kwargs)
        replicates.append(2.0 * (sim_null - sim_alt))
    replicates = np.asarray(replicates)
    # The +1 convention keeps the p-value strictly positive and unbiased for a
    # finite number of replicates.
    p_value = float((1 + (replicates >= observed).sum()) / (len(replicates) + 1))
    return observed, p_value, replicates


def report_child_structure(population: str, outcome: str, bootstrap: int = 0) -> None:
    """Which within-child structure the repeated measures actually support.

    Reports ``2 * delta logL`` against the constant-intercept baseline the models
    of record carry, for all children and for the repeated-measures children
    alone. The two columns matter separately: singletons inform the marginal
    spread and its age dependence, but say nothing about whether a child drifts
    or whether children cross, so a slope that appears only in the first column
    is cross-sectional widening rather than drift.

    ``bootstrap`` replicates give each statistic a **valid** reference
    distribution; without it the statistics are printed with no p-value at all,
    because neither comparison admits a chi-square one. See
    :func:`bootstrap_null`. Start around 200-500 replicates; each is two Gaussian
    ML fits on a few hundred children, so this is seconds to minutes rather than
    a sampling job.
    """
    d = adjusted_scores(population, outcome)
    centre = float(np.median(d.age.values.astype(float)))
    print(f"\n{'=' * 72}\n{population.upper()} / {outcome} — within-child structure "
          f"(ages centred at {centre:.0f} mo)\n{'=' * 72}")
    comparisons = (
        ("slope vs constant intercept", {"slope": False}, {}),
        ("free rho vs rho=1 (rank one)", {"fix_rho": 1.0}, {}),
    )
    for label, min_obs in (("all children", 1), ("repeats only", 2)):
        blocks = _child_blocks(d, min_obs)
        params, ll_slope = fit_child_structure(blocks, centre)
        print(f"  {label:14s} n={len(blocks):4d}   tau0={params['tau0']:.3f}  "
              f"tau1={params['tau1']:.4f}/mo  rho01={params['rho01']:+.3f}  "
              f"sigma_occ={params['sigma_occ']:.3f}")
        for name, null_kwargs, alt_kwargs in comparisons:
            _, ll_null = fit_child_structure(blocks, centre, **null_kwargs)
            statistic = 2.0 * (ll_null - ll_slope)
            if not bootstrap:
                # No degrees of freedom are printed: both nulls are on a
                # boundary, so there is no chi-square this could be referred to.
                print(f"     {name:28s} 2dlogL = {statistic:7.2f}  "
                      "(no reference distribution — pass --bootstrap)")
                continue
            statistic, p_value, replicates = bootstrap_null(
                blocks,
                centre,
                null_kwargs=null_kwargs,
                alt_kwargs=alt_kwargs,
                draws=bootstrap,
                seed=20260824,
            )
            print(f"     {name:28s} 2dlogL = {statistic:7.2f}  "
                  f"p = {p_value:.4f} ({bootstrap} parametric-bootstrap replicates; "
                  f"null 95th pct {np.quantile(replicates, 0.95):.2f})")


# ----------------------------------------------------------------- reporting


def quartile_table(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    g = d.groupby("child")
    size = g.size()
    keep = size[size >= 2].index
    first = g.first().loc[keep]
    last = g.last().loc[keep]
    fq = pd.qcut(first["resid"], 4, labels=[1, 2, 3, 4]).astype(int)
    lq = pd.qcut(last["resid"], 4, labels=[1, 2, 3, 4]).astype(int)
    ct = pd.crosstab(fq, lq, normalize="index") * 100
    facts = {
        "n": len(keep),
        "same": float((fq == lq).mean() * 100),
        "within_one": float(((fq - lq).abs() <= 1).mean() * 100),
        "bottom_to_top": int(((fq == 1) & (lq == 4)).sum()),
        "top_to_bottom": int(((fq == 4) & (lq == 1)).sum()),
    }
    return ct, facts


def report(population: str, outcome: str, n_boot: int, max_age: float | None = None) -> None:
    d = adjusted_scores(population, outcome)
    if max_age is not None:
        d = d[d.age <= max_age].copy()
    pairs = within_child_pairs(d)
    n_children = d.groupby("child").size()
    n_rep = int((n_children >= 2).sum())

    label = population.upper() + (f" (age<={max_age:.0f})" if max_age else "")
    print(f"\n{'=' * 72}\n{label} / {outcome}  "
          f"— {len(d)} obs, {n_children.size} children, {n_rep} with >=2\n{'=' * 72}")
    if n_rep < 30:
        print("  too few repeated-measures children; skipping")
        return

    # ICC is defined on children with repeated measures; singletons contribute
    # no within-child variance and would inflate it.
    rep_only = d.groupby("child").filter(lambda g: len(g) >= 2)
    i = icc(rep_only)
    lo, hi = icc_ci(d, max(100, n_boot // 3))
    print(f"  ICC (between-child share of adjusted variance) = {i:.3f}  [{lo:.3f}, {hi:.3f}]")

    # Reliability must be measured on the SAME rows the pairs come from -- the
    # repeated-measures children -- or rho/rel mixes two populations. Using the
    # full set (singletons included) gave 0.865 against 0.853 here; small, but
    # the decomposition below is only coherent on one of them.
    rel = reliability_bound(rep_only)
    print(f"  reliability (binomial upper bound, repeats)    = {rel:.3f}")
    print(f"  variance: between-child {i:5.1%}  within {1 - i:5.1%} "
          f"(measurement {1 - rel:5.1%}, occasion {(1 - i) - (1 - rel):+5.1%})")

    print(f"\n  {'lag (mo)':>10s} {'pairs':>7s} {'children':>9s} {'rho':>7s} "
          f"{'89% CI':>18s} {'rho/rel':>8s}")
    for a, b in LAG_BINS:
        sub = pairs[(pairs.lag >= a) & (pairs.lag < b)]
        if len(sub) < 20:
            continue
        rho = _spearman(sub.r0, sub.r1)
        clo, chi = cluster_boot(sub, lambda x: _spearman(x.r0, x.r1), n_boot)
        dis = min(1.0, rho / rel) if rel > 0 else np.nan
        print(f"  {a:4d}-{b:<5d} {len(sub):7d} {sub.child.nunique():9d} "
              f"{rho:7.3f} [{clo:7.3f},{chi:7.3f}] {dis:8.3f}")

    bands = spread_by_age_band(d)
    if len(bands):
        print("\n  Spread of the adjusted score by age band (logit scale):")
        print(f"  {'band':>8s} {'n':>6s} {'SD obs':>8s} {'SD meas':>8s} {'SD non-meas':>12s}")
        for _, r in bands.iterrows():
            print(f"  {r.band:>8s} {r.n:6d} {r.sd_observed:8.3f} "
                  f"{r.sd_measurement:8.3f} {r.sd_non_measurement:12.3f}")

    ct, f = quartile_table(d)
    print(f"\n  Quartile at first vs last observation (n={f['n']}):")
    print("    first\\last     Q1     Q2     Q3     Q4")
    for q in [1, 2, 3, 4]:
        print(f"         Q{q}      " + " ".join(f"{ct.loc[q, c]:5.0f}%" for c in [1, 2, 3, 4]))
    print(f"    same quartile {f['same']:.0f}%   within one {f['within_one']:.0f}%   "
          f"bottom->top {f['bottom_to_top']}   top->bottom {f['top_to_bottom']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", type=int, default=400, help="cluster bootstrap replicates")
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help=(
            "Parametric-bootstrap replicates for the within-child structure "
            "comparisons (0 prints the statistics without a p-value, because "
            "neither null admits a chi-square reference; see bootstrap_null)."
        ),
    )
    args = ap.parse_args()
    for population in ("ds", "td"):
        for outcome in ("spoken", "understood"):
            report(population, outcome, args.boot)
    # The TD pool spans 8-30 months only, and that window is the vocabulary
    # explosion. Comparing it with a DS pool spanning 8-115 confounds tracking
    # with developmental stage, so repeat DS restricted to the same ages.
    for outcome in ("spoken", "understood"):
        report("ds", outcome, args.boot, max_age=30)
    # Which within-child structure the repeated measures support (section 10).
    for outcome in ("spoken", "understood"):
        report_child_structure("ds", outcome, args.bootstrap)


if __name__ == "__main__":
    main()
