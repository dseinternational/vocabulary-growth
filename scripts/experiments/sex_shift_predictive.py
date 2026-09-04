#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What a sex effect would do to VG20's posterior predictive for a new child.

Reconstructs the new-child predictive from the stored posterior medians of a
VG20 fit -- population proportions at the canonical ages, the child scales
``tau_subj_u`` and ``tau_subj_q``, ``rho_uq`` and the age-varying ``kappa`` --
with the model's paired structure (understood from the form ceiling, spoken
from the child's own understood count). It first checks that reconstruction
against the fit's own stored predictive quantiles and threshold probabilities,
then applies a girl/boy shift on the logit scale, girls up by half the
difference and boys down by half, and reports what moves.

Parameter uncertainty is dropped (posterior medians are used), which is small
next to the child spread and count noise that dominate these intervals. The
shift sizes are the descriptive estimates from
``scripts/experiments/sex_effect_by_study.py``, not fitted quantities.

Writes ``<output-root>/comparisons/sex-effect/sex_shift_predictive.csv``,
``sex_shift_thresholds.csv`` and ``sex_shift_words_months.csv`` -- the last
reads the base-case constant logit shift on the count scale and as months of
lead along the population curve, which is how a constant shift comes to look
age-varying. Cited by
``notes/202609041206-sex-differences-in-vocabulary.md``.

Run: ``python scripts/experiments/sex_shift_predictive.py [--model-dir DIR] [--output-dir DIR]``
"""

import argparse
import os

import numpy as np
import pandas as pd

from vocab_growth import environment as env

VG20_DIR = "VG20-age-understood-spoken-ds-re-subj-uq-anchored-corr"
N_TRIALS = 810
N_DRAWS = 400_000
AGES = (24, 36, 48, 60, 72)
QUANTILES = (0.055, 0.5, 0.945)
# (label, delta on the understood logit, delta on the production-ratio logit)
CASES = [
    ("base", 0.20, 0.15),  # what most studies and both TD references agree on
    ("large", 0.33, 0.22),  # the pooled DS estimate once ie_02 is included
]
THRESHOLDS = [
    (36, 10, "<="),
    (36, 50, ">="),
    (48, 50, "<="),
    (48, 100, ">="),
    (60, 100, "<="),
    (60, 300, ">="),
]


def logit(p):
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


class Predictive:
    """New-child predictive at a canonical age from a fit's stored summaries."""

    def __init__(self, model_dir, seed=20260904):
        self.rng = np.random.default_rng(seed)
        self.su = pd.read_csv(os.path.join(model_dir, "posterior_summary_u.csv")).set_index("age_months")
        self.ss = pd.read_csv(os.path.join(model_dir, "posterior_summary_s.csv")).set_index("age_months")
        self.sq = pd.read_csv(os.path.join(model_dir, "posterior_summary_q.csv")).set_index("age_months")
        self.ku = pd.read_csv(os.path.join(model_dir, "posterior_kappa_u.csv"))
        self.ks = pd.read_csv(os.path.join(model_dir, "posterior_kappa_s.csv"))
        diag = pd.read_csv(os.path.join(model_dir, "diagnostics.csv"), index_col=0)
        self.tau_u = float(diag.loc["tau_subj_u", "mean"])
        self.tau_q = float(diag.loc["tau_subj_q", "mean"])
        self.rho = float(diag.loc["rho_uq", "mean"])

    @staticmethod
    def _kappa(table, age):
        return float(table.iloc[(table["age_months"] - age).abs().argmin()]["kappa_median"])

    def draw(self, age, delta_u=0.0, delta_q=0.0):
        z1, z2 = self.rng.standard_normal(N_DRAWS), self.rng.standard_normal(N_DRAWS)
        p_u = expit(logit(self.su.loc[age, "p_population_median"]) + delta_u + self.tau_u * z1)
        q_logit = logit(self.sq.loc[age, "q_median"]) + delta_q
        q = expit(q_logit + self.tau_q * (self.rho * z1 + np.sqrt(1 - self.rho**2) * z2))
        k_u, k_s = self._kappa(self.ku, age), self._kappa(self.ks, age)
        y_u = self.rng.binomial(N_TRIALS, self.rng.beta(p_u * k_u, (1 - p_u) * k_u))
        y_s = self.rng.binomial(y_u, self.rng.beta(q * k_s, (1 - q) * k_s))
        return y_u, y_s


def constant_shift_in_words_and_months(pred, d_u, d_q):
    """A constant logit shift read on the count scale and as months of lead."""
    rows = []
    for outcome, table, delta in (("understood", pred.su, d_u), ("spoken", pred.ss, d_u + d_q)):
        ages = np.array([a for a in range(24, 73, 6) if a in table.index], dtype=float)
        lg = np.array([logit(table.loc[a, "p_population_median"]) for a in ages])
        slope = np.gradient(lg, ages / 12)  # population logit gain per year
        for age, l, s in zip(ages, lg, slope, strict=True):
            rows.append(
                dict(
                    outcome=outcome,
                    logit_shift=delta,
                    age=int(age),
                    median_words=N_TRIALS * expit(l),
                    gap_words=N_TRIALS * (expit(l + delta / 2) - expit(l - delta / 2)),
                    logit_slope_per_year=s,
                    gap_months=12 * delta / s,
                )
            )
    return pd.DataFrame(rows)


def check(pred):
    print("==== reconstruction against the fit's stored predictive (median [89% ETI]) ====")
    for age in AGES:
        y_u, y_s = pred.draw(age)
        qu, qs = np.quantile(y_u, QUANTILES), np.quantile(y_s, QUANTILES)
        r, t = pred.su.loc[age], pred.ss.loc[age]
        print(
            f"{age:>2}m understood {qu[1]:4.0f} [{qu[0]:4.0f}, {qu[2]:4.0f}]"
            f" vs stored {r['Y_median']:4.0f} [{r['Y_ci_lo']:4.0f}, {r['Y_ci_hi']:4.0f}]   "
            f"spoken {qs[1]:4.0f} [{qs[0]:4.0f}, {qs[2]:4.0f}]"
            f" vs stored {t['Y_median']:4.0f} [{t['Y_ci_lo']:4.0f}, {t['Y_ci_hi']:4.0f}]   "
            f"P(spoken<=10) {np.mean(y_s <= 10):.3f} vs {t['P(Y<=10)']:.3f}"
        )


def shifts(pred):
    rows, thresholds = [], []
    for case, d_u, d_q in CASES:
        for age in AGES:
            pooled = pred.draw(age)
            girls = pred.draw(age, d_u / 2, d_q / 2)
            boys = pred.draw(age, -d_u / 2, -d_q / 2)
            for i, outcome in enumerate(("understood", "spoken")):
                qp, qg, qb = (np.quantile(s[i], QUANTILES) for s in (pooled, girls, boys))
                rows.append(
                    dict(
                        case=case,
                        age=age,
                        outcome=outcome,
                        pooled_lo=qp[0],
                        pooled_median=qp[1],
                        pooled_hi=qp[2],
                        girls_lo=qg[0],
                        girls_median=qg[1],
                        girls_hi=qg[2],
                        boys_lo=qb[0],
                        boys_median=qb[1],
                        boys_hi=qb[2],
                        median_gap=qg[1] - qb[1],
                        gap_over_width=(qg[1] - qb[1]) / (qp[2] - qp[0]),
                        boys_median_percentile_in_girls=100 * np.mean(girls[i] <= qb[1]),
                    )
                )
            for t_age, k, side in THRESHOLDS:
                if t_age != age:
                    continue
                if side == "<=":
                    prob = lambda y, k=k: np.mean(y <= k)  # noqa: E731
                else:
                    prob = lambda y, k=k: np.mean(y >= k)  # noqa: E731
                thresholds.append(
                    dict(
                        case=case,
                        age=age,
                        question=f"P(spoken {side} {k})",
                        pooled=prob(pooled[1]),
                        girls=prob(girls[1]),
                        boys=prob(boys[1]),
                    )
                )
    return pd.DataFrame(rows), pd.DataFrame(thresholds)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", default=None, help="A VG20 fit directory (defaults to the model of record under the output root).")
    parser.add_argument("--output-dir", default=None, help="Output root (defaults to the repository's resolved output root).")
    args = parser.parse_args()
    if args.output_dir:
        env.set_output_root(args.output_dir)
    model_dir = args.model_dir or os.path.join(env.output_root(), "models", VG20_DIR)
    out_dir = os.path.join(env.output_root(), "comparisons", "sex-effect")
    os.makedirs(out_dir, exist_ok=True)

    pred = Predictive(model_dir)
    print(f"model: {model_dir}\ntau_subj_u={pred.tau_u:.3f} tau_subj_q={pred.tau_q:.3f} rho_uq={pred.rho:.2f}\n")
    check(pred)
    table, thresholds = shifts(pred)
    table.to_csv(os.path.join(out_dir, "sex_shift_predictive.csv"), index=False)
    thresholds.to_csv(os.path.join(out_dir, "sex_shift_thresholds.csv"), index=False)
    base_u, base_q = CASES[0][1], CASES[0][2]
    words_months = constant_shift_in_words_and_months(pred, base_u, base_q)
    words_months.to_csv(os.path.join(out_dir, "sex_shift_words_months.csv"), index=False)

    for case, d_u, d_q in CASES:
        print(f"\n==== {case} case: delta_u {d_u:.2f}, delta_q {d_q:.2f}; girls +delta/2, boys -delta/2 ====")
        for _, r in table[table["case"] == case].iterrows():
            print(
                f"{r['age']:>3}m {r['outcome']:10s}"
                f" pooled {r['pooled_median']:4.0f} [{r['pooled_lo']:3.0f}, {r['pooled_hi']:4.0f}]"
                f"  girls {r['girls_median']:4.0f} [{r['girls_lo']:3.0f}, {r['girls_hi']:4.0f}]"
                f"  boys {r['boys_median']:4.0f} [{r['boys_lo']:3.0f}, {r['boys_hi']:4.0f}]"
                f"  gap {r['median_gap']:+4.0f}  gap/width {r['gap_over_width']:.2f}"
                f"  boys' median at girls' {r['boys_median_percentile_in_girls']:.0f}th percentile"
            )
        for _, r in thresholds[thresholds["case"] == case].iterrows():
            print(f"    {r['age']}m {r['question']}: pooled {r['pooled']:.2f}  girls {r['girls']:.2f}  boys {r['boys']:.2f}")
    print()
    print("==== the base-case constant logit shift, read in words and in months along the population curve ====")
    for outcome, sub in words_months.groupby("outcome", sort=False):
        print(f"  {outcome} (shift {sub['logit_shift'].iloc[0]:.2f}):")
        print("    age (m)       : " + "  ".join(f"{a:5d}" for a in sub["age"]))
        print("    median words  : " + "  ".join(f"{v:5.0f}" for v in sub["median_words"]))
        print("    gap in words  : " + "  ".join(f"{v:+5.0f}" for v in sub["gap_words"]))
        print("    gap in months : " + "  ".join(f"{v:+5.1f}" for v in sub["gap_months"]))
    print()
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
