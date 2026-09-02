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

The two forms were mostly not completed together. The Oxford CDI was given at the
study's first assessment, and for 24 of the 34 children the DSE checklist is recorded
at a different age — up to four months apart, and usually later. Placing each form at
its own age is what makes the crosswalk an estimate at matched age rather than a raw
ratio inflated by growth over the gap; the raw ratio does rise with the gap. Because
the DSE completion dates are not recorded, ``--variant`` re-fits under alternative
treatments of that gap (see ``notes/202609021236-crosswalk-timing-sensitivity.md``):

``base``
    Every dual-form child, every row at its recorded age — the analysis of record.
``concurrent``
    Only children whose Oxford age equals one of their DSE ages, keeping all their rows.
``strict``
    Only the concurrent DSE/Oxford row pairs themselves.
``realigned``
    Every dual-form child, but the DSE row nearest each Oxford administration is given
    the Oxford age — the bound in which the DSE checklist reflects the child at the
    Oxford test date despite its later recorded age.
``all``
    Each of the four in turn.

Usage:
    python scripts/crosswalk_dse_oxford.py [--outcome understood|spoken|both] [--variant base|concurrent|strict|realigned|all]
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
VARIANTS = ("base", "concurrent", "strict", "realigned")


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


def pair_forms(df: pd.DataFrame) -> pd.DataFrame:
    """Pair each Oxford administration with the same child's nearest-in-age DSE row.

    One row per Oxford administration: the index labels of both rows and the gap
    ``dse_age - oxford_age`` in months (positive when the DSE checklist is recorded later).
    """
    pairs = []
    for oxf_idx, oxf in df[df.form == "OXF"].iterrows():
        dse = df[(df.subject_id == oxf.subject_id) & (df.form == "DSE")]
        dse_idx = (dse.age - oxf.age).abs().idxmin()
        pairs.append(
            {
                "subject_id": oxf.subject_id,
                "oxf_idx": oxf_idx,
                "dse_idx": dse_idx,
                "gap": int(df.at[dse_idx, "age"] - oxf.age),
            }
        )
    return pd.DataFrame(pairs)


def describe_gaps(df: pd.DataFrame) -> None:
    """Print how far apart in age each child's two forms are recorded."""
    pairs = pair_forms(df)
    counts = pairs.gap.value_counts().sort_index()
    print(f"dual-form children: {pairs.subject_id.nunique()}; Oxford administrations: {len(pairs)}")
    print("gap (nearest DSE age - Oxford age, months): "
          + ", ".join(f"{int(g):+d}: {int(n)}" for g, n in counts.items()))


def apply_variant(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Return the analysis frame for one timing variant (see the module docstring)."""
    if variant == "base":
        return df.copy()
    pairs = pair_forms(df)
    concurrent = pairs[pairs.gap == 0]
    if variant == "concurrent":
        return df[df.subject_id.isin(concurrent.subject_id)].copy()
    if variant == "strict":
        keep = sorted(set(concurrent.dse_idx) | set(concurrent.oxf_idx))
        return df.loc[keep].copy()
    if variant == "realigned":
        out = df.copy()
        for p in pairs.itertuples():
            out.at[p.dse_idx, "age"] = df.at[p.oxf_idx, "age"]
        return out
    raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")


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


def report(df: pd.DataFrame, outcome: str, variant: str, seed: int, draws: int, tune: int) -> None:
    n_obs = int(df[outcome].notna().sum())
    print(f"\n===  {outcome.upper()}  variant={variant}  "
          f"(obs={n_obs}, children={df.subject_id.nunique()})  ===")

    idata, d = fit(df, outcome, age_varying=False, seed=seed, draws=draws, tune=tune)
    summ = az.summary(idata, var_names=["b0", "b1", "sigma_u", "delta", "log_kappa"])
    print(f"  max r-hat {float(summ['r_hat'].max()):.3f}   "
          f"min ess {float(summ['ess_bulk'].min()):.0f}")

    lo, md, hi = _eti_90(idata.posterior["b1"].values.ravel())
    print(f"  b1 (logit per 12 months): median {md:.3f}  90% ETI [{lo:.3f}, {hi:.3f}]")
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
    parser.add_argument(
        "--variant", choices=[*VARIANTS, "all"], default="base",
        help="timing-sensitivity variant (see the module docstring); 'all' runs each in turn",
    )
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    args = parser.parse_args()

    df = load_dualform()
    describe_gaps(df)
    outcomes = ["understood", "spoken"] if args.outcome == "both" else [args.outcome]
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    for variant in variants:
        frame = apply_variant(df, variant)
        for outcome in outcomes:
            report(frame, outcome, variant, seed=args.seed, draws=args.draws, tune=args.tune)


if __name__ == "__main__":
    main()
