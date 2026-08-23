# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ages by which a given share of children with Down syndrome reach a word count.

This is a **between-child** quantity, not the population-milestone quantity that
``scripts/time_to_milestone.py`` reports. That script gives the age at which the
*population* trajectory reaches a target, with an interval expressing posterior
uncertainty about that single age; its docstring is explicit that the interval
"is not a spread across individual children (that would need new-child
posterior-predictive draws)". This harness computes the spread across children.

Model: VG20, the model of record for the DS joint understood + spoken estimands.
Each child carries a constant offset per outcome -- ``z_u`` on the logit of
understood and ``z_q`` on the logit of the production ratio -- drawn from a
bivariate normal with scales ``tau_subj_u`` / ``tau_subj_q`` and correlation
``rho_uq``. Counts are on the 810-word reference scale.

Understood is exact. A child's curve is ``expit(f_u(a) + z_u)``, so it reaches N
words exactly where ``f_u(a) = logit(N/810) - z_u``. Crossing age decreases
monotonically in ``z_u``, so the child at the X-th percentile of *crossing age*
is the child at the (1-X)-th percentile of ``z_u``: no simulation is needed, and
the answer for each posterior draw is one threshold crossing.

Spoken is simulated. ``S = 810 * expit(f_u + z_u) * expit(h + z_q)`` depends on
both effects at once and is not a monotone function of a single index, so for
each posterior draw a synthetic cohort is drawn from the fitted covariance and
the empirical percentile of crossing age is taken within that draw. Posterior
uncertainty is then the spread of that percentile across draws.

Censoring is reported, never extrapolated. Crossing ages beyond a quantity's
reporting cap (understood 72 months per the model definition, spoken 90) are
reported as beyond the window rather than as a number, and where more than
(1-X) of a draw's cohort never reach the target inside the grid the percentile
itself is censored.

The latent proportion is the estimand, not a simulated questionnaire score: the
Beta-Binomial observation noise (``kappa``) is measurement scatter around a
child's own trajectory, so including it would answer "what would this child
score on one administration", not "has this child learned N words".
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

from vocab_growth import comparison as C
from vocab_growth import environment as env
from vocab_growth.models.definitions import MODEL_REGISTRY

MODEL = "vg20"
TARGETS = [50, 150, 300, 500, 750]
SHARES = [0.50, 0.75, 0.90]
N_CHILDREN = 2000          # synthetic cohort per posterior draw, spoken only
DRAW_SUBSAMPLE = 2000      # posterior draws used for the spoken simulation
HDI_PROB = 0.89
RNG = np.random.default_rng(20260823)

CAP_UNDERSTOOD = MODEL_REGISTRY[MODEL].report_max_age_understood  # 72
CAP_SPOKEN = 90.0  # per-quantity reporting policy; the plot grid's own maximum


def load() -> dict:
    """Latent-scale population curves plus the child-effect scales, per draw.

    Goes through the shared loader rather than reading the trace directly: it
    age-sorts the plot grid, and ``X_plot`` is not guaranteed to be stored in
    ascending order. Every crossing routine here assumes it is.
    """
    ages, (f_u, h), scal = C._load_reshaped_draws(
        C.trace_path(MODEL),
        ("f_u_plot", "h_plot"),
        ("tau_subj_u", "tau_subj_q", "rho_uq"),
    )
    return {
        "ages": ages,
        "f_u": f_u,
        "h": h,
        "tau_u": scal["tau_subj_u"],
        "tau_q": scal["tau_subj_q"],
        "rho": scal["rho_uq"],
        "n_trials": C.n_trials(MODEL),
    }


def crossing_ages(Y: np.ndarray, ages: np.ndarray, level: float) -> np.ndarray:
    """First age each row of Y reaches ``level``, with the two NaN cases split.

    ``C.first_crossing_age`` returns NaN both where a series never reaches the
    level and where it is *already above* it at the youngest supported age -- it
    deliberately refuses to invent a crossing below its grid. Those are opposite
    censorings and collapsing them would bias every summary here, so map "never
    reaches" to +inf and "reached before the grid starts" to -inf.
    """
    out = C.first_crossing_age(Y, ages, level)
    below_support = Y[:, 0] > level
    out = np.where(np.isnan(out) & below_support, -np.inf, out)
    return np.where(np.isnan(out), np.inf, out)


def summarise(values: np.ndarray, cap: float, floor: float) -> dict:
    """Median and HDI of a crossing-age sample, with censoring made explicit.

    Censored values are kept as +/-inf rather than dropped: discarding the draws
    that never reach the target would pull the median toward the ages of the
    draws that happened to reach it, which is exactly the bias that makes a
    milestone look earlier than the model says.
    """
    beyond = float(np.mean(values > cap))
    before = float(np.mean(values < floor))
    median = float(np.median(values))
    finite = np.isfinite(values)
    lo, hi = (
        C.hdi_from_samples(values[finite], HDI_PROB)
        if finite.sum() > 1
        else (np.nan, np.nan)
    )
    return {
        "median": median if np.isfinite(median) and median <= cap else np.nan,
        "lo": float(lo) if np.isfinite(lo) else np.nan,
        "hi": float(hi) if np.isfinite(hi) else np.nan,
        "frac_beyond_window": beyond,
        "frac_before_window": before,
        "censored": bool(beyond > 0.0 or before > 0.0),
        "beyond_cap": bool(not (np.isfinite(median) and median <= cap)),
    }


def understood_rows(d: dict) -> list[dict]:
    """Exact inversion: the X-th percentile child is a shifted logit threshold."""
    rows = []
    f_u, tau_u, ages, n = d["f_u"], d["tau_u"], d["ages"], d["n_trials"]
    for target in TARGETS:
        base = logit(target / n)
        for share in SHARES:
            # Crossing age falls as z_u rises, so share X of children corresponds
            # to the (1 - X) quantile of z_u.
            z = tau_u * norm.ppf(1.0 - share)
            # Per draw, find where f_u(a) reaches the child's own threshold. The
            # threshold varies by draw, so shift the curve instead of the level.
            shifted = f_u - (base - z)[:, None]
            crossing = crossing_ages(shifted, ages, 0.0)
            s = summarise(crossing, CAP_UNDERSTOOD, ages[0])
            rows.append({"outcome": "understood", "target": target, "share": share, **s})
    return rows


def spoken_rows(d: dict) -> list[dict]:
    """Simulated: spoken depends on both child effects and has no closed form."""
    rows = []
    f_u, h, ages, n = d["f_u"], d["h"], d["ages"], d["n_trials"]
    tau_u, tau_q, rho = d["tau_u"], d["tau_q"], d["rho"]
    n_draw = f_u.shape[0]
    idx = RNG.choice(n_draw, size=min(DRAW_SUBSAMPLE, n_draw), replace=False)

    per_draw = {(t, s): np.full(len(idx), np.nan) for t in TARGETS for s in SHARES}
    for k, i in enumerate(idx):
        zu = RNG.standard_normal(N_CHILDREN)
        zq = rho[i] * zu + np.sqrt(max(1.0 - rho[i] ** 2, 0.0)) * RNG.standard_normal(N_CHILDREN)
        zu = tau_u[i] * zu
        zq = tau_q[i] * zq
        # (children, ages) for this one draw's population curves.
        S = n * expit(f_u[i][None, :] + zu[:, None]) * expit(h[i][None, :] + zq[:, None])
        for target in TARGETS:
            crossing = crossing_ages(S, ages, target)
            for share in SHARES:
                # The cohort percentile is well defined with censoring as long as
                # the share sits inside the reached portion: np.quantile on the
                # sorted array handles +/-inf correctly because it only needs the
                # order statistic, not the arithmetic, at that position.
                per_draw[(target, share)][k] = np.quantile(
                    crossing, share, method="inverted_cdf"
                )
    for target in TARGETS:
        for share in SHARES:
            rows.append(
                {
                    "outcome": "spoken",
                    "target": target,
                    "share": share,
                    **summarise(per_draw[(target, share)], CAP_SPOKEN, ages[0]),
                }
            )
    return rows


def main() -> None:
    d = load()
    print(f"VG20  draws={d['f_u'].shape[0]}  ages={d['ages'].min():.0f}-{d['ages'].max():.0f} mo  scale={d['n_trials']}")
    print(f"tau_subj_u={d['tau_u'].mean():.3f}  tau_subj_q={d['tau_q'].mean():.3f}  rho_uq={d['rho'].mean():+.3f}")
    df = pd.DataFrame(understood_rows(d) + spoken_rows(d))
    os.makedirs(env.comparisons_output_dir(), exist_ok=True)
    out = os.path.join(env.comparisons_output_dir(), "age_at_word_count_vg20.csv")
    df.to_csv(out, index=False)

    for outcome, cap in (("understood", CAP_UNDERSTOOD), ("spoken", CAP_SPOKEN)):
        print(f"\n=== Words {outcome} — age (months) by which each share of children reaches the count ===")
        print(f"    (reporting cap {cap:.0f} months; '>cap' = not reached inside the window)")
        sub = df[df.outcome == outcome]
        header = f"{'words':>6} | " + " | ".join(f"{int(s*100)}% of children".center(22) for s in SHARES)
        print(header)
        print("-" * len(header))
        for target in TARGETS:
            cells = []
            for share in SHARES:
                r = sub[(sub.target == target) & (sub.share == share)].iloc[0]
                if r["beyond_cap"]:
                    cells.append(f"not by {cap:.0f} mo".center(22))
                else:
                    hi = f"{r['hi']:.1f}" if np.isfinite(r["hi"]) and r["hi"] <= cap else f">{cap:.0f}"
                    cells.append(f"{r['median']:5.1f}  [{r['lo']:.1f}, {hi}]".center(22))
            print(f"{target:>6} | " + " | ".join(cells))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
