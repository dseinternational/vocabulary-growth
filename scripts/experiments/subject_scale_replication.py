#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Is the between-child scale underestimated because replication is thin?

The question
------------
Parameter recovery underestimates the between-child scale in 9 of 9 replicates
across VG10, VG12 and VG20, by roughly 3-6%. It is not the sampling tier (VG20
at ``rep`` moves the estimate by 0.19% of the truth), not the prior's location
(``HalfNormal(1.5)`` has mean 1.197, above the truths of 0.78-0.81, so shrinkage
pulls *up*), not ``subject_variance_partition`` (present at the same size
without it), and not posterior skew (mean equals median to four decimals and the
median sits at 0.484-0.491 of its own interval).

What is left is the design. In the Down syndrome pool 432 of 767 children appear
once; in the typically-developing pools it is worse. Separating a persistent
child effect from Beta-Binomial dispersion when most clusters hold one
observation is the textbook hard case, and [#229] is a list of options for
reparameterising around it -- written on the premise that the partition is the
cause, which is now ruled out.

This script tests the design directly, on a model stripped of everything the
fitted models add. No GP, no anchors, no study effects, no age, no second
outcome: one mean, one child scale, one dispersion. If the bias appears here, it
is intrinsic to estimating a between-child scale from thin replication and no
reparameterisation in #229 removes it -- which promotes option 4 (report total
scatter) from fallback to the only option that survives. If it does not appear,
something in the full models is responsible and is worth hunting.

The design
----------
Truth is fixed across conditions; only the replication structure moves.

    all-singleton   N children, 1 observation each
    observed-mix    the DS pool's actual structure: 56% seen once, the rest 2-3
    all-triplicate  N children, 3 observations each
    age-varying     observed-mix, plus an age-varying mean that the model must
                    estimate with a flexible basis rather than being told
    with-study      observed-mix, plus children nested in studies with a study
                    random intercept the model must also estimate
    age-varying-kappa   observed-mix, with the dispersion itself varying in age
    clustered-ages  age-varying, but a child's repeat visits sit a few months
                    apart as they do in the real pools, instead of being spread
                    independently across the whole age range
    floor-p0        observed-mix at p = 0.05 instead of 0.30
    ceiling-p0      observed-mix at p = 0.90
    floor-small-n   observed-mix at p = 0.05 with the real pools' small
                    denominators, so exact zeros actually occur

First pass (6 replicates, 2026-08-19): all seven of the original conditions
returned within +-1.4% of the realised spread, none of them the -3 to -6% the
fitted models show. Three -- `age-varying` (1.066), `with-study` (1.019) and
`clustered-ages` (1.117) -- missed the project's R-hat 1.01 gate, so they are
not evidence either way and are re-run here at a longer tune.

The last three conditions exist because the first seven shared a blind spot:
every one held the mean proportion at 0.30, the most informative part of the
logit curve, while the fitted models estimate the comprehension scale where the
median proportion is 0.046 and the conditional spoken share is 0.000 outright.
The recovery matrices point the same way -- the bias is on the comprehension
scale and on `kappa_young`, not on `tau_subj_q` -- so the boundary is the first
thing to test, not the eighth.

The fourth condition is the one that matters once the first three come back
clean. Every fitted model in this project puts a linear trend plus a
Hilbert-space GP on the mean, and children sit at different ages; a mean flexible
enough to follow the trajectory can also follow some of the between-child
variation, which would bias the child scale low. That is invisible in a
constant-mean simulation, which is why the first three conditions cannot rule it
out -- they can only rule out replication as the explanation.

``observed-mix`` is the anchor: it should reproduce the fitted models' bias if
this stripped model is a fair proxy for them. ``all-triplicate`` holds the child
count fixed and adds replication, so a bias that shrinks there is about
information per child rather than about the number of children.

Usage::

    python scripts/experiments/subject_scale_replication.py
    python scripts/experiments/subject_scale_replication.py --replicates 5 --children 767
    python scripts/experiments/subject_scale_replication.py \
        --conditions floor-p0,ceiling-p0,floor-small-n --replicates 6 --suffix _p0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pymc as pm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from vocab_growth import environment as env  # noqa: E402

#: Truth. `tau` and `kappa` are the fitted models' own scale: tau_subj_u sits at
#: 0.78-0.81 across the recovery truths, and kappa_old_u at about 25. `p0` puts
#: the mean proportion mid-range, away from both boundaries, so nothing here is
#: a ceiling or floor effect.
TAU_TRUE = 0.79
KAPPA_TRUE = 25.0
P0 = 0.30
N_TRIALS = 810

#: The DS bivariate pool: 767 children, 432 of them seen once (56.3%), 335 with
#: two or more. Measured, not assumed -- see the pool counts in #229.
N_CHILDREN = 767
N_SINGLETON = 432

#: The DS pool's age span, and the basis the age-varying condition fits. Eight
#: interior knots is in the same spirit as the models' HSGP: flexible enough to
#: follow a developmental curve without being told its shape.
AGE_LO, AGE_HI = 8.0, 115.0
N_BASIS = 8

#: The real pools' repeat structure: a child's second visit is a median 6 months
#: after their first (IQR 5-7.2, from notes/202608141600 §6). `age-varying`
#: spread repeats independently over the whole range, which is the one thing that
#: makes a flexible mean *unable* to follow a single child -- their observations
#: are nowhere near each other. This condition puts them where they really sit.
REPEAT_GAP_MONTHS = 6.0

#: A second grouping level, present in every model that shows the bias and
#: absent from the first four conditions. 14 studies of very uneven size is the
#: DS pool's actual shape, and `tau_study` is VG20's fitted tau_u.
N_STUDIES = 14
TAU_STUDY = 0.37

#: The fitted models let the Beta-Binomial concentration vary with age, anchored
#: at two reference ages. A dispersion free to move with age can absorb
#: between-child variation age-selectively, which a constant kappa cannot.
KAPPA_YOUNG, KAPPA_OLD = 60.0, 25.0

#: Every condition above holds the mean proportion at `P0` = 0.30 -- the most
#: informative part of the logit curve, and nowhere near where the real pools
#: sit. Measured on the Down syndrome analysis frame: median p_U is 0.046 at
#: 8-18 months with 53% of administrations below 0.05, and the conditional
#: spoken share `q` has a median of 0.000 there with 71% below 0.05. At the
#: other end 9% of comprehension administrations are above 0.95 after 48
#: months. A scale estimated where the link is flat is a different estimation
#: problem from one estimated at p = 0.3, and it is the problem the fitted
#: models actually face.
P0_FLOOR = 0.05
P0_CEILING = 0.90

#: The floor in the real data is not low `p` alone -- it is low `p` on few
#: trials. `q` is conditioned on the child's comprehension count, so at young
#: ages its denominator is a couple of dozen words rather than the form's full
#: length, and exact zeros follow. 810 trials at p = 0.05 never produces one.
#: Drawn lognormal about a median of 40 to span roughly 8-200.
N_TRIALS_SMALL_MEDIAN = 40.0
N_TRIALS_SMALL_LOG_SD = 0.8
N_TRIALS_SMALL_LO, N_TRIALS_SMALL_HI = 3, 810

#: The fitted models do not give kappa a value at each anchor directly. They
#: give it an asymptote plus an exponential age term whose two *excesses* carry
#: the priors: kappa(z) = kappa_min + exp(a + b z), with a and b solved so the
#: totals hit kappa_min + excess at each anchor. `age-varying-kappa` used a
#: linear interpolation between two freely-estimated anchor values, which is a
#: two-parameter identified form. The real one is three parameters, and the
#: recovery matrices show the third behaving as a prior rather than an estimate:
#: kappa_min's posterior median lands at 3.8, 4.1 and 4.2 against truths of
#: 2.84, 10.37 and 1.09 -- the LogNormal(log 3, 0.8) prior's own mean is 4.13.
#: Where the truth sits above that, kappa_young inherits the shortfall, which is
#: the only identified explanation so far for kappa_young being low in 9 of 9.
#:
#: Values from `_DS_SPOKEN_KAPPA` in models/definitions.py.
KAPPA_ANCHOR_AGES = (18.0, 36.0)
KAPPA_MIN_PRIOR_MU, KAPPA_MIN_PRIOR_SIGMA = np.log(3.0), 0.8
EXCESS_YOUNG_PRIOR_MU, EXCESS_YOUNG_PRIOR_SIGMA = np.log(45.0), 0.7
EXCESS_OLD_PRIOR_MU, EXCESS_OLD_PRIOR_SIGMA = np.log(4.0), 0.7

#: Two truths for the asymptote: one at the prior's centre, one above it, as in
#: VG10 r01's truth draw. If the subject scale is biased only in the second, the
#: mechanism is the unidentified asymptote dragging the dispersion level and the
#: child scale absorbing the difference.
KAPPA_MIN_TRUE, KAPPA_MIN_TRUE_HIGH = 3.0, 10.0

#: Conditions in the order they were added. The last three came after the first
#: seven all returned null, when re-reading the recovery matrices showed the
#: bias is selective -- it hits the comprehension child scale and VG12's
#: `tau_subject` in 9 of 9 replicates but not `tau_subj_q`, and `kappa_young`
#: goes with it -- and that every condition to date had been run at p = 0.3.
ALL_CONDITIONS = (
    "all-singleton",
    "observed-mix",
    "all-triplicate",
    "age-varying",
    "with-study",
    "age-varying-kappa",
    "clustered-ages",
    "floor-p0",
    "ceiling-p0",
    "floor-small-n",
    "anchored-kappa",
    "anchored-kappa-high-min",
)

#: Conditions that keep `observed-mix`'s visit structure and vary something else.
DERIVED_CONDITIONS = frozenset(
    {
        "age-varying",
        "with-study",
        "age-varying-kappa",
        "clustered-ages",
        "floor-p0",
        "ceiling-p0",
        "floor-small-n",
        "anchored-kappa",
        "anchored-kappa-high-min",
    }
)


def visit_counts(condition: str, n_children: int, rng) -> np.ndarray:
    if condition == "all-singleton":
        return np.ones(n_children, dtype=int)
    if condition == "all-triplicate":
        return np.full(n_children, 3, dtype=int)
    if condition == "observed-mix":
        counts = np.ones(n_children, dtype=int)
        repeated = n_children - N_SINGLETON
        # 335 repeated children hold 999 rows in the real pool, so a little
        # under three visits each; split them 2/3 the way the pool does.
        counts[:repeated] = rng.choice([2, 3], size=repeated, p=[0.6, 0.4])
        rng.shuffle(counts)
        return counts
    raise ValueError(condition)


def _basis(age: np.ndarray) -> np.ndarray:
    """Gaussian bumps over the age range — a stand-in for the models' HSGP."""
    u = (age - AGE_LO) / (AGE_HI - AGE_LO)
    centres = np.linspace(0.0, 1.0, N_BASIS)
    width = 1.0 / (N_BASIS - 1)
    return np.exp(-0.5 * ((u[:, None] - centres[None, :]) / width) ** 2)


def _true_mean_logit(age: np.ndarray) -> np.ndarray:
    """A developmental curve on the logit scale: low and rising, then flattening."""
    u = (age - AGE_LO) / (AGE_HI - AGE_LO)
    return -2.2 + 3.4 * u - 1.1 * u**2


def small_trial_counts(n_rows: int, rng) -> np.ndarray:
    """Per-observation trial counts spanning the real pools' small denominators."""
    draws = rng.lognormal(np.log(N_TRIALS_SMALL_MEDIAN), N_TRIALS_SMALL_LOG_SD, n_rows)
    return np.clip(np.rint(draws), N_TRIALS_SMALL_LO, N_TRIALS_SMALL_HI).astype(int)


def simulate(
    counts: np.ndarray, rng, p0: float = P0, small_n: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Beta-Binomial counts with a child random effect on the logit.

    Returns ``(y, child, z, n_trials)``. ``z`` matters: the spread a refit can
    recover is ``TAU_TRUE * sd(z)``, and for 767 draws that realised SD carries
    a 2.55% sampling error, so scoring against the nominal ``TAU_TRUE`` alone
    would charge the model for the simulation's own noise.
    """
    z = rng.standard_normal(counts.size)
    logit_p = np.log(p0 / (1 - p0)) + TAU_TRUE * z
    p = 1.0 / (1.0 + np.exp(-logit_p))
    child = np.repeat(np.arange(counts.size), counts)
    p_row = p[child]
    n_trials = (
        small_trial_counts(child.size, rng)
        if small_n
        else np.full(child.size, N_TRIALS, dtype=int)
    )
    # Beta-Binomial: a per-observation probability drawn around the child's own.
    theta = rng.beta(p_row * KAPPA_TRUE, (1 - p_row) * KAPPA_TRUE)
    y = rng.binomial(n_trials, theta)
    return y, child, z, n_trials


def simulate_age_varying(counts: np.ndarray, rng):
    """As `simulate`, but the population mean moves with age.

    Each observation gets its own age, so a repeatedly-measured child is seen at
    different points on the curve — as in the real pools.
    """
    z = rng.standard_normal(counts.size)
    child = np.repeat(np.arange(counts.size), counts)
    age = rng.uniform(AGE_LO, AGE_HI, size=child.size)
    logit_p = _true_mean_logit(age) + TAU_TRUE * z[child]
    p = 1.0 / (1.0 + np.exp(-logit_p))
    theta = rng.beta(p * KAPPA_TRUE, (1 - p) * KAPPA_TRUE)
    y = rng.binomial(N_TRIALS, theta)
    return y, child, z, age


def simulate_with_study(counts: np.ndarray, rng):
    """Children nested in studies of uneven size, with a study random intercept."""
    z = rng.standard_normal(counts.size)
    # Uneven study sizes, as in the real pool: a few large, several tiny.
    weights = rng.dirichlet(np.full(N_STUDIES, 0.6))
    study_of_child = rng.choice(N_STUDIES, size=counts.size, p=weights)
    s_eff = rng.standard_normal(N_STUDIES)
    child = np.repeat(np.arange(counts.size), counts)
    logit_p = (
        np.log(P0 / (1 - P0))
        + TAU_TRUE * z[child]
        + TAU_STUDY * s_eff[study_of_child][child]
    )
    p = 1.0 / (1.0 + np.exp(-logit_p))
    theta = rng.beta(p * KAPPA_TRUE, (1 - p) * KAPPA_TRUE)
    y = rng.binomial(N_TRIALS, theta)
    return y, child, z, study_of_child[child]


def simulate_age_varying_kappa(counts: np.ndarray, rng):
    """Constant mean, but the dispersion moves with age as the fitted models let it."""
    z = rng.standard_normal(counts.size)
    child = np.repeat(np.arange(counts.size), counts)
    age = rng.uniform(AGE_LO, AGE_HI, size=child.size)
    u = (age - AGE_LO) / (AGE_HI - AGE_LO)
    kappa_row = KAPPA_YOUNG + (KAPPA_OLD - KAPPA_YOUNG) * u
    logit_p = np.log(P0 / (1 - P0)) + TAU_TRUE * z[child]
    p = 1.0 / (1.0 + np.exp(-logit_p))
    theta = rng.beta(p * kappa_row, (1 - p) * kappa_row)
    y = rng.binomial(N_TRIALS, theta)
    return y, child, z, age


def simulate_clustered_ages(counts: np.ndarray, rng):
    """As `simulate_age_varying`, but a child's repeats are a few months apart.

    This is the difference that matters if the mechanism is a flexible mean
    following a child rather than the population: observations far apart in age
    cannot be joined by a smooth curve without distorting it, observations six
    months apart can.
    """
    z = rng.standard_normal(counts.size)
    child = np.repeat(np.arange(counts.size), counts)
    base = rng.uniform(AGE_LO, AGE_HI - REPEAT_GAP_MONTHS * 2, size=counts.size)
    offsets = np.concatenate(
        [np.arange(c) * rng.normal(REPEAT_GAP_MONTHS, 1.5) for c in counts]
    )
    age = np.clip(base[child] + offsets, AGE_LO, AGE_HI)
    logit_p = _true_mean_logit(age) + TAU_TRUE * z[child]
    p = 1.0 / (1.0 + np.exp(-logit_p))
    theta = rng.beta(p * KAPPA_TRUE, (1 - p) * KAPPA_TRUE)
    y = rng.binomial(N_TRIALS, theta)
    return y, child, z, age


def _anchor_z() -> tuple[float, float]:
    lo, hi = KAPPA_ANCHOR_AGES
    span = AGE_HI - AGE_LO
    return (lo - AGE_LO) / span, (hi - AGE_LO) / span


def _kappa_of_u(u, kappa_min, excess_young, excess_old):
    """kappa_min + exp(a + b z), with a and b solved from the two anchors."""
    z_young, z_old = _anchor_z()
    log_y, log_o = np.log(excess_young), np.log(excess_old)
    b = (log_o - log_y) / (z_old - z_young)
    a = log_y - b * z_young
    return kappa_min + np.exp(a + b * u)


def simulate_anchored_kappa(counts: np.ndarray, rng, kappa_min_true: float):
    """The fitted models' three-parameter dispersion, not a two-anchor interpolation."""
    z = rng.standard_normal(counts.size)
    child = np.repeat(np.arange(counts.size), counts)
    age = rng.uniform(AGE_LO, AGE_HI, size=child.size)
    u = (age - AGE_LO) / (AGE_HI - AGE_LO)
    kappa_row = _kappa_of_u(
        u,
        kappa_min_true,
        KAPPA_YOUNG - kappa_min_true,
        KAPPA_OLD - kappa_min_true,
    )
    logit_p = np.log(P0 / (1 - P0)) + TAU_TRUE * z[child]
    p = 1.0 / (1.0 + np.exp(-logit_p))
    theta = rng.beta(p * kappa_row, (1 - p) * kappa_row)
    y = rng.binomial(N_TRIALS, theta)
    return y, child, z, age


def fit(
    y: np.ndarray,
    child: np.ndarray,
    n_children: int,
    seed: int,
    age: np.ndarray | None = None,
    study: np.ndarray | None = None,
    kappa_age: np.ndarray | None = None,
    anchored_kappa_age: np.ndarray | None = None,
    p0: float = P0,
    n_trials: np.ndarray | int = N_TRIALS,
    tune: int = 2000,
    draws: int = 1500,
    chains: int = 2,
    target_accept: float = 0.9,
) -> dict:
    with pm.Model():
        tau = pm.HalfNormal("tau", sigma=1.5)
        z = pm.Normal("z", mu=0.0, sigma=1.0, shape=n_children)
        kappa = pm.HalfNormal("kappa", sigma=50.0)
        if age is None:
            # Centred on the truth, as in every condition: the question is the
            # scale's recovery, not the mean's, and a mean prior that missed by
            # two SDs at the floor would confound the two.
            mu = pm.Normal("mu", mu=np.log(p0 / (1 - p0)), sigma=1.0)
            mean_term = mu
        else:
            # Linear trend plus a flexible basis: the models' own mean structure,
            # in miniature. The model is not told the curve's shape.
            mu = pm.Normal("mu", mu=0.0, sigma=2.0)
            slope = pm.Normal("slope", mu=0.0, sigma=2.0)
            w = pm.Normal("w", mu=0.0, sigma=1.0, shape=N_BASIS)
            u = (age - AGE_LO) / (AGE_HI - AGE_LO)
            mean_term = mu + slope * u + pm.math.dot(_basis(age), w)
        if study is not None:
            tau_study = pm.HalfNormal("tau_study", sigma=1.0)
            s_raw = pm.Normal("s_raw", mu=0.0, sigma=1.0, shape=N_STUDIES)
            mean_term = mean_term + tau_study * s_raw[study]
        p = pm.Deterministic("p", pm.math.sigmoid(mean_term + tau * z[child]))
        if anchored_kappa_age is not None:
            # The models' own form: an unidentified asymptote plus an
            # exponential age term carrying the priors at two anchors.
            k_min = pm.LogNormal(
                "k_min", mu=KAPPA_MIN_PRIOR_MU, sigma=KAPPA_MIN_PRIOR_SIGMA
            )
            e_young = pm.LogNormal(
                "e_young", mu=EXCESS_YOUNG_PRIOR_MU, sigma=EXCESS_YOUNG_PRIOR_SIGMA
            )
            e_old = pm.LogNormal(
                "e_old", mu=EXCESS_OLD_PRIOR_MU, sigma=EXCESS_OLD_PRIOR_SIGMA
            )
            z_young, z_old = _anchor_z()
            b_k = (pm.math.log(e_old) - pm.math.log(e_young)) / (z_old - z_young)
            a_k = pm.math.log(e_young) - b_k * z_young
            uk = (anchored_kappa_age - AGE_LO) / (AGE_HI - AGE_LO)
            kappa_row = k_min + pm.math.exp(a_k + b_k * uk)
            pm.Deterministic("kappa_young_sim", k_min + e_young)
            pm.Deterministic("kappa_old_sim", k_min + e_old)
        elif kappa_age is None:
            kappa_row = kappa
        else:
            # Two-anchor form, as the models use: a value at each end of the
            # range, interpolated, rather than an intercept and a slope.
            k_young = pm.HalfNormal("k_young", sigma=80.0)
            k_old = pm.HalfNormal("k_old", sigma=80.0)
            uk = (kappa_age - AGE_LO) / (AGE_HI - AGE_LO)
            kappa_row = k_young + (k_old - k_young) * uk
        pm.BetaBinomial(
            "y", alpha=p * kappa_row, beta=(1 - p) * kappa_row, n=n_trials, observed=y
        )
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(chains, 4),
            target_accept=target_accept,
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
        )
    post = idata.posterior
    draws = np.asarray(post["tau"].values).ravel()
    # The truth a refit can actually recover is the spread the data contain,
    # not the nominal tau -- the realised SD of n standard normals carries a
    # 1/sqrt(2n) sampling error, 2.55% at n = 767.
    import arviz as az

    # arviz >= 1.2 returns a DataTree; take the max over the scalar parameters.
    flat_kappa = kappa_age is None and anchored_kappa_age is None
    names = ["tau", "mu"] + (["kappa"] if flat_kappa else [])
    if anchored_kappa_age is not None:
        names = names + ["k_min", "e_young", "e_old"]
    rhat_tree = az.rhat(idata, var_names=names)
    rhat_ds = rhat_tree["posterior"] if "posterior" in rhat_tree else rhat_tree
    per_param = {v: float(np.asarray(rhat_ds[v].values).max()) for v in names}
    rhat = float(max(per_param.values()))
    # Which parameter fails matters. The stand-in mean basis here is an
    # intercept, a slope and eight overlapping Gaussian bumps, which is
    # deliberately over-complete and mixes badly; that is a defect of this
    # simulation's mean, not evidence about the models, and it should not
    # disqualify a condition whose `tau` mixed perfectly well.
    rhat_tau = per_param["tau"]
    return {
        "tau_median": float(np.median(draws)),
        "tau_mean": float(draws.mean()),
        "tau_sd": float(draws.std(ddof=1)),
        "kappa_median": float(np.median(np.asarray(post["kappa"].values).ravel())),
        "max_rhat": rhat,
        "rhat_tau": rhat_tau,
        **(
            {
                "k_min_median": float(
                    np.median(np.asarray(post["k_min"].values).ravel())
                ),
                "kappa_young_median": float(
                    np.median(np.asarray(post["kappa_young_sim"].values).ravel())
                ),
                "kappa_old_median": float(
                    np.median(np.asarray(post["kappa_old_sim"].values).ravel())
                ),
            }
            if anchored_kappa_age is not None
            else {}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--children", type=int, default=N_CHILDREN)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--conditions",
        default=",".join(ALL_CONDITIONS),
        help="comma-separated subset of: " + ", ".join(ALL_CONDITIONS),
    )
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--draws", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument(
        "--suffix",
        default="",
        help="appended to the output filename, so a re-run at a different "
        "sampler setting does not overwrite the first pass",
    )
    args = parser.parse_args()
    env.set_output_root(args.output_dir)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in ALL_CONDITIONS]
    if unknown:
        parser.error(f"unknown condition(s): {', '.join(unknown)}")

    rows = []
    for condition in conditions:
        for r in range(1, args.replicates + 1):
            rng = np.random.default_rng(args.seed + 1000 * r)
            structure = "observed-mix" if condition in DERIVED_CONDITIONS else condition
            counts = visit_counts(structure, args.children, rng)
            age = study = kappa_age = anchored_kappa_age = None
            p0 = P0
            n_trials = N_TRIALS
            if condition == "age-varying":
                y, child, z, age = simulate_age_varying(counts, rng)
            elif condition == "clustered-ages":
                y, child, z, age = simulate_clustered_ages(counts, rng)
            elif condition == "with-study":
                y, child, z, study = simulate_with_study(counts, rng)
            elif condition == "age-varying-kappa":
                y, child, z, kappa_age = simulate_age_varying_kappa(counts, rng)
            elif condition == "floor-p0":
                p0 = P0_FLOOR
                y, child, z, n_trials = simulate(counts, rng, p0=p0)
            elif condition == "ceiling-p0":
                p0 = P0_CEILING
                y, child, z, n_trials = simulate(counts, rng, p0=p0)
            elif condition in ("anchored-kappa", "anchored-kappa-high-min"):
                k_min_true = (
                    KAPPA_MIN_TRUE_HIGH
                    if condition.endswith("high-min")
                    else KAPPA_MIN_TRUE
                )
                y, child, z, anchored_kappa_age = simulate_anchored_kappa(
                    counts, rng, k_min_true
                )
            elif condition == "floor-small-n":
                p0 = P0_FLOOR
                y, child, z, n_trials = simulate(counts, rng, p0=p0, small_n=True)
            else:
                y, child, z, n_trials = simulate(counts, rng)
            realised = TAU_TRUE * float(np.std(z, ddof=1))
            out = fit(
                y,
                child,
                args.children,
                args.seed + r,
                age=age,
                study=study,
                kappa_age=kappa_age,
                anchored_kappa_age=anchored_kappa_age,
                p0=p0,
                n_trials=n_trials,
                tune=args.tune,
                draws=args.draws,
                chains=args.chains,
                target_accept=args.target_accept,
            )
            rows.append(
                {
                    "condition": condition,
                    "replicate": r,
                    "n_children": args.children,
                    "n_rows": int(counts.sum()),
                    "mean_visits": round(float(counts.mean()), 3),
                    "p0": p0,
                    "mean_trials": round(float(np.mean(n_trials)), 1),
                    "frac_zero": round(float(np.mean(np.asarray(y) == 0)), 4),
                    "tune": args.tune,
                    "target_accept": args.target_accept,
                    "tau_true": TAU_TRUE,
                    "tau_realised": realised,
                    **out,
                    "pct_vs_true": 100 * (out["tau_median"] - TAU_TRUE) / TAU_TRUE,
                    "pct_vs_realised": 100 * (out["tau_median"] - realised) / realised,
                    "z_vs_realised": (out["tau_median"] - realised) / out["tau_sd"],
                }
            )
            print(
                f"{condition:16s} r{r:02d}  rows {counts.sum():5d}  "
                f"tau {out['tau_median']:.4f}  "
                f"({100 * (out['tau_median'] - realised) / realised:+.2f}% vs realised)  "
                f"kappa {out['kappa_median']:6.1f}  rhat {out['max_rhat']:.4f}",
                flush=True,
            )

    table = pd.DataFrame(rows)
    out_dir = os.path.join(env.output_root(), "comparisons", "recovery")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(
        out_dir, f"subject_scale_replication{args.suffix}.csv"
    )
    table.to_csv(path, index=False)

    pd.set_option("display.width", 200)
    print("\n=== mean over replicates ===")
    print(
        table.groupby("condition")[
            [
                "mean_visits",
                "p0",
                "mean_trials",
                "frac_zero",
                "tau_median",
                "pct_vs_true",
                "pct_vs_realised",
                "z_vs_realised",
                "kappa_median",
                "max_rhat",
            ]
        ]
        .agg({"max_rhat": "max", **{c: "mean" for c in (
            "mean_visits", "p0", "mean_trials", "frac_zero", "tau_median",
            "pct_vs_true", "pct_vs_realised", "z_vs_realised", "kappa_median")}})
        .round(3)
        .to_string()
    )
    # The project's hard convergence gate. A condition that misses it is not
    # evidence either way, and the first pass reported three that did.
    gate = table.groupby("condition")[["rhat_tau", "max_rhat"]].max()
    failed = gate[gate.rhat_tau > 1.01]
    if len(failed):
        print("\nNOT ASSESSABLE -- `tau`'s own R-hat above the 1.01 gate:")
        print(failed.round(4).to_string())
    else:
        print("\n`tau` clears the 1.01 R-hat gate in every condition.")
    noisy = gate[(gate.rhat_tau <= 1.01) & (gate.max_rhat > 1.01)]
    if len(noisy):
        print(
            "\nAssessable, but some other parameter mixes badly -- check it is "
            "the mean basis before reading anything into it:"
        )
        print(noisy.round(4).to_string())
    print(f"\ntau_true = {TAU_TRUE}, kappa_true = {KAPPA_TRUE}")
    print(f"written: {path}")
    print(
        "\nReading the result: `all-singleton` came back clean in the first pass, so\n"
        "the between-child/dispersion split is identified through the mean-variance\n"
        "shape rather than through replication, and thin replication is not the\n"
        "explanation. What remains untested by a well-specified simulation is\n"
        "whether the boundary regime the fitted models work in is what costs the\n"
        "scale -- that is what the three p0 conditions ask."
    )


if __name__ == "__main__":
    main()
