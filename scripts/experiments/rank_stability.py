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

    rel = reliability_bound(d)
    print(f"  reliability (binomial upper bound)            = {rel:.3f}")

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
    args = ap.parse_args()
    for population in ("ds", "td"):
        for outcome in ("spoken", "understood"):
            report(population, outcome, args.boot)
    # The TD pool spans 8-30 months only, and that window is the vocabulary
    # explosion. Comparing it with a DS pool spanning 8-115 confounds tracking
    # with developmental stage, so repeat DS restricted to the same ages.
    for outcome in ("spoken", "understood"):
        report("ds", outcome, args.boot, max_age=30)


if __name__ == "__main__":
    main()
