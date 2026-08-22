#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gate 1 for the 4x4 successor to VG19 and VG20.

VG19 gives each child a rate as well as an offset on each outcome but forces the
two outcomes' child effects to be independent. VG20 correlates the two outcomes'
offsets but gives no rates. Their union is a 4x4 over ``(b0u, b1u, b0q, b1q)``
with six correlations, and the element **neither model estimates** is
``corr(b1u, b1q)``: do children who gain comprehension faster than their peers
also convert what they understand into speech faster?

This decides whether that model is worth registering, by the same method
202608141600 §10.3 used for VG19's own Gate 1 -- maximum likelihood on
age- and study-adjusted residuals against a known per-observation binomial
sampling variance, no PyMC and no registered-model fit involved. Reported as
``2 * delta logL`` against nested restrictions:

* ``block`` -- the two outcomes independent. This is VG19's structure.
* ``no-slope-corr`` -- everything free except ``corr(b1u, b1q)``. The 1 df test
  against the full model is the question above, isolated.
* ``intercepts-only`` -- both rates zero, one cross-outcome correlation. This is
  VG20's structure.

The residual scale mirrors the models' own nested likelihood: ``u`` is scored
against the administration's form ceiling and ``q`` against that same
administration's **understood count**, so the two sampling variances are the
ones the Beta-Binomial layer would use.

Cited by ``notes/`` -- see the index. Run: ``python four_by_four_gate1.py``.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from vocab_growth import data_utils as du

PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
NAMES = ["b0u", "b1u", "b0q", "b1q"]


def paired_residuals():
    """One row per administration carrying both a ``u`` and a ``q`` residual."""
    d = du.load_combined_data()
    d = d[d.understood.notna() & d.spoken.notna() & d.survey_vocab_max.notna()].copy()
    # q is only defined where the child understands something to convert.
    d = d[d.understood > 0].copy()

    ceiling = d.survey_vocab_max.to_numpy(float)
    und = d.understood.to_numpy(float)
    spo = np.minimum(d.spoken.to_numpy(float), und)

    pu = np.clip(und / ceiling, 0.5 / ceiling, 1 - 0.5 / ceiling)
    pq = np.clip(spo / und, 0.5 / und, 1 - 0.5 / und)

    d["child"] = d.study.astype(str) + "|" + d.subject_id.astype(str)
    age = d.age.to_numpy(float)
    a = (age - age.mean()) / age.std()
    X = np.column_stack(
        [np.ones(len(a)), a, a**2, a**3,
         pd.get_dummies(d.study, drop_first=True).to_numpy(float)]
    )
    for key, p, n in (("u", pu, ceiling), ("q", pq, und)):
        y = np.log(p / (1 - p))
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        d[f"resid_{key}"] = y - X @ beta
        d[f"var_{key}"] = 1.0 / (n * p * (1 - p))
    return d.sort_values(["child", "age"]), age


def blocks_of(d, centre, min_obs=1):
    """Per child: stacked (u, q) residuals, their design rows and known variances.

    ``load`` is the (2m x 4) matrix mapping ``(b0u, b1u, b0q, b1q)`` onto the
    child's 2m stacked observations, so the child covariance is
    ``load @ Sigma @ load.T`` plus the diagonal.
    """
    out = []
    for _, g in d.groupby("child"):
        if len(g) < min_obs:
            continue
        g = g.sort_values("age")
        c = (g.age.to_numpy(float) - centre) / 12.0
        m = len(g)
        load = np.zeros((2 * m, 4))
        load[:m, 0], load[:m, 1] = 1.0, c
        load[m:, 2], load[m:, 3] = 1.0, c
        r = np.concatenate([g.resid_u.to_numpy(), g.resid_q.to_numpy()])
        v = np.concatenate([g.var_u.to_numpy(), g.var_q.to_numpy()])
        out.append((load, r, v, m))
    return out


def sigma_from(theta, free_pairs):
    """Build Sigma from 4 log-SDs and the free correlations; None if not PD."""
    sd = np.exp(theta[:4])
    R = np.eye(4)
    for val, (i, j) in zip(theta[4 : 4 + len(free_pairs)], free_pairs, strict=True):
        R[i, j] = R[j, i] = np.tanh(val)
    S = R * np.outer(sd, sd)
    try:
        np.linalg.cholesky(S + 1e-10 * np.eye(4))
    except np.linalg.LinAlgError:
        return None
    return S


def negll(blocks, S, occ_u, occ_q):
    total = 0.0
    for load, r, v, m in blocks:
        C = load @ S @ load.T
        C.flat[:: C.shape[0] + 1] += v + np.concatenate(
            [np.full(m, occ_u**2), np.full(m, occ_q**2)]
        )
        try:
            chol = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            return 1e10
        z = np.linalg.solve(chol, r)
        total += 2 * np.log(np.diag(chol)).sum() + z @ z
    return 0.5 * total


def fit(blocks, free_pairs, *, zero_slopes=(), start=None):
    """ML fit. ``zero_slopes`` pins those component SDs to (near) zero."""
    npar = 4 + len(free_pairs)

    def objective(theta):
        t = theta.copy()
        for k in zero_slopes:
            t[k] = np.log(1e-6)
        S = sigma_from(t, free_pairs)
        if S is None:
            return 1e10
        return negll(blocks, S, np.exp(t[npar]), np.exp(t[npar + 1]))

    if start is None:
        start = np.concatenate([np.log([0.7, 0.2, 1.1, 0.5]),
                                np.zeros(len(free_pairs)), np.log([0.4, 0.5])])
    best = None
    for jitter in (0.0, 0.25, -0.25):
        s = start + jitter * np.concatenate(
            [np.ones(4) * 0.2, np.ones(len(free_pairs)) * 0.3, np.ones(2) * 0.2])
        r = minimize(objective, s, method="Powell",
                     options={"xtol": 1e-4, "ftol": 1e-5, "maxiter": 60000})
        r = minimize(objective, r.x, method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 60000})
        if best is None or r.fun < best.fun:
            best = r
    return best


def main():
    d, _ = paired_residuals()
    centre = 36.0
    for label, min_obs in (("all children", 1), ("repeats only", 2)):
        blocks = blocks_of(d, centre, min_obs)
        n_obs = sum(b[3] for b in blocks)
        print(f"\n{'=' * 78}\nDS / (understood, q) — 4x4 child covariance, {label}: "
              f"{len(blocks)} children, {n_obs} paired administrations "
              f"(ages centred at {centre:.0f} mo, rates per year)\n{'=' * 78}")

        full = fit(blocks, PAIRS)
        S = sigma_from(full.x, PAIRS)
        sd = np.sqrt(np.diag(S))
        R = S / np.outer(sd, sd)
        print("  SDs:  " + "  ".join(f"{n}={s:.3f}" for n, s in zip(NAMES, sd, strict=True)))
        print("  correlations")
        for i, j in PAIRS:
            print(f"    {NAMES[i]:>3s} , {NAMES[j]:<3s}  {R[i, j]:+.3f}")
        print(f"  occasion SD: u={np.exp(full.x[10]):.3f}  q={np.exp(full.x[11]):.3f}")

        tests = [
            ("block (= VG19: outcomes independent)", [(0, 1), (2, 3)], (), 4),
            ("intercepts only (= VG20)", [(0, 2)], (1, 3), 5),
            ("no cross SLOPE-SLOPE  b1u,b1q", [p for p in PAIRS if p != (1, 3)], (), 1),
            ("no cross INTERCEPT-INTERCEPT  b0u,b0q", [p for p in PAIRS if p != (0, 2)], (), 1),
            ("no cross  b0u,b1q  (u level -> q rate)", [p for p in PAIRS if p != (0, 3)], (), 1),
            ("no cross  b1u,b0q  (u rate -> q level)", [p for p in PAIRS if p != (1, 2)], (), 1),
            ("VG19 + rho_uq only (one cross term)", [(0, 1), (2, 3), (0, 2)], (), 3),
        ]
        print("\n  2*dlogL of the full 4x4 over each restriction (positive favours the 4x4)")
        for name, fp, zs, df in tests:
            sub = fit(blocks, fp, zero_slopes=zs)
            print(f"    vs {name:<40s} {2 * (sub.fun - full.fun):8.2f}  ({df} df)")


def rank_analysis():
    """How many dimensions does the child covariance actually have?

    The tanh-per-entry parameterisation in :func:`sigma_from` rejects
    non-positive-definite proposals, so the optimiser can and does walk to the
    boundary of the feasible set. That is not a bug in the search -- it is where
    the likelihood is maximised -- but it means the correlation matrix at the
    optimum can be singular, which invalidates the chi-square reference for the
    tests in :func:`main` and matters far more for the design than any single
    correlation does.

    Refits Sigma = L L' with L of shape (4, rank), which is unconstrained,
    always positive semi-definite, and exactly rank ``rank``. A rank-3 fit
    reaching the same likelihood as rank-4 confirms the deficiency is real
    rather than an artefact of the other parameterisation, because the two
    searches share no coordinates. Free covariance parameters are
    ``4 * rank - rank * (rank - 1) / 2``: 4, 7, 9, 10 for ranks 1 to 4.
    """
    d, _ = paired_residuals()
    for label, min_obs in (("all children", 1), ("repeats only", 2)):
        blocks = blocks_of(d, 36.0, min_obs)
        full = fit(blocks, PAIRS)
        S = sigma_from(full.x, PAIRS)
        sd = np.sqrt(np.diag(S))
        ev = np.linalg.eigvalsh(S / np.outer(sd, sd))
        print(f"\n######## {label}: dimension of the child covariance")
        print(f"  correlation eigenvalues at the 4x4 optimum: "
              f"{np.array2string(ev, precision=6)}")
        for rank in (1, 2, 3, 4):
            nll, L, occ = _fit_low_rank(blocks, rank)
            Sr = L @ L.T
            sdr = np.sqrt(np.diag(Sr))
            for k in range(rank):
                if L[np.argmax(np.abs(L[:, k])), k] < 0:
                    L[:, k] *= -1
            npar = 4 * rank - rank * (rank - 1) // 2
            print(f"  rank {rank} ({npar:2d} params)  negll {nll:9.4f}   SDs "
                  + " ".join(f"{n}={x:.3f}" for n, x in zip(NAMES, sdr, strict=True)))
            for k in range(rank):
                print(f"      factor {k + 1} loadings: "
                      + "  ".join(f"{n}={x:+.3f}"
                                  for n, x in zip(NAMES, L[:, k] / sdr, strict=True)))


def _fit_low_rank(blocks, rank, tries=6, seed=7):
    rng = np.random.default_rng(seed)
    n = 4 * rank

    def obj(th):
        L = th[:n].reshape(4, rank)
        return negll(blocks, L @ L.T, np.exp(th[n]), np.exp(th[n + 1]))

    best = None
    for _ in range(tries):
        s = np.concatenate([rng.normal(0, 0.6, n), np.log([0.5, 0.6])])
        r = minimize(obj, s, method="Powell",
                     options={"xtol": 1e-5, "ftol": 1e-7, "maxiter": 90000})
        r = minimize(obj, r.x, method="Nelder-Mead",
                     options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 90000})
        if best is None or r.fun < best.fun:
            best = r
    return best.fun, best.x[:n].reshape(4, rank), np.exp(best.x[n:n + 2])


if __name__ == "__main__":
    main()
    rank_analysis()
