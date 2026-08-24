# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Age-adjusted DSE(810)<->Oxford(416) checklist crosswalk from the uk_02 dual-form children.

Reproduces the equating analysis in ``notes/202607121200-statistical-model-review.md``
§3(A). Every uk_02 child assessed on both the 810-item DSE checklist and the 416-item
Oxford CDI is used to fit a small Bayesian measurement model — a shared per-child
latent DSE-frame trajectory (population age trend + child random intercept), each form
observed at its own age through the correct Beta-Binomial denominator, and a
logit-scale form offset as the crosswalk — and the age-adjusted count ratio
R = DSE-count / Oxford-count is reported. Reference points: fixed n = 810 (the models'
current choice) implies R = 1; a per-form ``n_trials`` implies R = 810/416 = 1.95.

Usage:
    python scripts/crosswalk_dse_oxford.py [--outcome understood|spoken|both]
"""

import argparse

import arviz as az
import dse_research_utils.statistics.intervals as stats_intervals
import duckdb
import numpy as np
import pandas as pd
import pymc as pm

from vocab_growth.data_utils import VOCABULARY_DATA_PATH

LEN_DSE, LEN_OXF = 810, 416
AGE_CENTRE, AGE_SCALE = 36.0, 12.0  # z(age) = (age - 36) / 12
REPORT_AGES = (25, 31, 37, 43, 49)


def z_age(age):
    return (np.asarray(age, dtype=float) - AGE_CENTRE) / AGE_SCALE


def load_dualform() -> pd.DataFrame:
    """Return uk_02 rows for children assessed on both the DSE and Oxford forms."""
    with duckdb.connect(VOCABULARY_DATA_PATH, read_only=True) as con:
        df = con.execute(
            """
            WITH dualform AS (
                SELECT subject_id
                FROM vocab_combined
                WHERE study = 'uk_02'
                GROUP BY subject_id
                HAVING COUNT(DISTINCT survey_vocab_max) > 1
            )
            SELECT subject_id, age, survey_vocab_max AS n_form, understood, spoken
            FROM vocab_combined
            WHERE study = 'uk_02'
              AND subject_id IN (SELECT subject_id FROM dualform)
            ORDER BY subject_id, age, n_form
            """
        ).df()
    df["form"] = np.where(df.n_form == LEN_DSE, "DSE", "OXF")
    return df


def fit(df: pd.DataFrame, outcome: str, age_varying: bool, seed: int, draws: int, tune: int):
    """Fit the crosswalk model for one outcome; return (inference data, analysis frame)."""
    d = df[df[outcome].notna() & (df[outcome] >= 0)].reset_index(drop=True)
    child_idx, children = pd.factorize(d.subject_id)
    is_dse = (d.form.values == "DSE").astype(float)
    n_form = np.where(d.form.values == "DSE", LEN_DSE, LEN_OXF).astype(float)
    z = z_age(d.age.values)
    y = d[outcome].values.astype(int)

    with pm.Model():
        b0 = pm.Normal("b0", -0.5, 1.5)
        b1 = pm.Normal("b1", 0.4, 0.5)
        sigma_u = pm.HalfNormal("sigma_u", 1.0)
        u = pm.Deterministic("u", sigma_u * pm.Normal("u_raw", 0, 1, shape=len(children)))
        delta = pm.Normal("delta", 0.0, 1.0)  # crosswalk: logit offset of Oxford vs DSE
        delta1 = pm.Normal("delta1", 0.0, 0.5) if age_varying else None
        kappa = pm.Deterministic("kappa", pm.math.exp(pm.Normal("log_kappa", 3.0, 1.0)))

        eta_dse = b0 + b1 * z + u[child_idx]
        offset = delta + (delta1 * z if age_varying else 0.0)
        p = pm.math.sigmoid(eta_dse + (1.0 - is_dse) * offset)
        pm.BetaBinomial("y", n=n_form, alpha=p * kappa, beta=(1 - p) * kappa, observed=y)

        idata = pm.sample(
            draws, tune=tune, chains=4, target_accept=0.95,
            nuts_sampler="nutpie", random_seed=seed, progressbar=False,
        )
    return idata, d


def count_ratio(idata, age, age_varying: bool):
    """Posterior draws of R = DSE-count / Oxford-count at ``age`` (population level, u = 0)."""
    post = idata.posterior
    b0 = post["b0"].values.ravel()
    b1 = post["b1"].values.ravel()
    delta = post["delta"].values.ravel()
    z = z_age(age)
    eta = b0 + b1 * z
    offset = delta + (post["delta1"].values.ravel() * z if age_varying else 0.0)
    p_dse = 1.0 / (1.0 + np.exp(-eta))
    p_oxf = 1.0 / (1.0 + np.exp(-(eta + offset)))
    return (LEN_DSE * p_dse) / (LEN_OXF * p_oxf)


def _eti_90(x):
    # A 90% equal-tailed interval; this was previously (mis)printed as an HDI.
    lo, hi = stats_intervals.eti_1d(x, eti_prob=0.90)
    return lo, float(np.median(np.asarray(x)[np.isfinite(x)])), hi


def report(df: pd.DataFrame, outcome: str, seed: int, draws: int, tune: int) -> None:
    n_obs = int(df[outcome].notna().sum())
    print(f"\n===  {outcome.upper()}  (obs={n_obs}, children={df.subject_id.nunique()})  ===")

    idata, d = fit(df, outcome, age_varying=False, seed=seed, draws=draws, tune=tune)
    summ = az.summary(idata, var_names=["b0", "b1", "sigma_u", "delta", "log_kappa"])
    print(f"  max r-hat {float(summ['r_hat'].max()):.3f}   "
          f"min ess {float(summ['ess_bulk'].min()):.0f}")

    lo, md, hi = _eti_90(idata.posterior["delta"].values.ravel())
    print(f"  delta (logit offset): median {md:.3f}  90% ETI [{lo:.3f}, {hi:.3f}]  "
          f"(0 = per-form; {np.log(LEN_DSE / LEN_OXF):.3f} = length-only)")

    print("  R = DSE/Oxford count ratio by age (population level):")
    for age in REPORT_AGES:
        lo, md, hi = _eti_90(count_ratio(idata, age, age_varying=False))
        print(f"    age {age:2d}: median {md:.3f}  90% ETI [{lo:.3f}, {hi:.3f}]")
    r_mid = count_ratio(idata, float(np.median(d.age)), age_varying=False)
    print(f"  P(R > 1) = {np.mean(r_mid > 1):.3f}   P(R < 1.95) = {np.mean(r_mid < 1.95):.3f}  "
          f"(at median age {np.median(d.age):.0f})")

    idata_av, _ = fit(df, outcome, age_varying=True, seed=seed, draws=draws, tune=tune)
    lo, md, hi = _eti_90(idata_av.posterior["delta1"].values.ravel())
    excludes = "yes" if (lo > 0 or hi < 0) else "no"
    print(f"  age-varying delta1: median {md:+.3f}  90% ETI [{lo:+.3f}, {hi:+.3f}]  "
          f"(CI excludes 0? {excludes})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outcome", choices=["understood", "spoken", "both"], default="both")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    args = parser.parse_args()

    df = load_dualform()
    outcomes = ["understood", "spoken"] if args.outcome == "both" else [args.outcome]
    for outcome in outcomes:
        report(df, outcome, seed=args.seed, draws=args.draws, tune=args.tune)


if __name__ == "__main__":
    main()
