#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What does VG19's child random slope say about individual trajectories?

VG08-VG10 and VG20 give each child a single constant offset, so every child's
curve is the population curve shifted vertically and the model cannot express a
child who changes standing. VG19 gives each child an offset *and* a rate, drawn
from a per-outcome 2x2 covariance. This harness reads those three scalars per
outcome out of the fitted trace and turns them into the quantities a reader
actually asks about: how stable a child's standing is, how much of later
standing is predictable from earlier standing, and how wide the between-child
band is on the natural scale.

Everything here is a deterministic function of ``tau_subj_{u,q}_{0,1,rho}``
propagated over posterior draws -- no refitting, no simulation. The child effect
at age ``a`` is ``b0 + b1 * D(a)`` with ``D(a) = (a - ref) / 12``, so

    Cov(a, b) = tau0^2 + rho * tau0 * tau1 * (D(a) + D(b)) + tau1^2 * D(a) * D(b)

and the reported correlations, regressions and rank-swap probabilities all
follow from that one expression. ``P(swap)`` is the bivariate-normal orthant
probability ``arccos(corr) / pi`` for the difference between two independently
drawn children.

Hard-coded to the output root of the machine the 2026-08-22 run happened on.
Cited by ``notes/202608220748-vg19-individual-trajectories.md``.
"""

import dse_research_utils.statistics.intervals as stats_intervals
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import expit

BASE = "/scratch2/vg-output/models/"
VG19 = BASE + "VG19-age-understood-spoken-ds-re-subj-uq-anchored-slope"
VG20 = BASE + "VG20-age-understood-spoken-ds-re-subj-uq-anchored-corr"
REF_AGE = 36.0
AGES = [18, 24, 36, 48, 60, 72, 84]


def posterior(path, names):
    ds = xr.open_dataset(path + "/trace.nc", group="posterior", engine="h5netcdf")
    out = {n: ds[n].values.ravel() for n in names}
    ds.close()
    return out


def D(a):
    return (a - REF_AGE) / 12.0


def cov(t0, t1, r, a, b):
    return t0**2 + r * t0 * t1 * (D(a) + D(b)) + t1**2 * D(a) * D(b)


def sd(t0, t1, r, a):
    return np.sqrt(cov(t0, t1, r, a, a))


def eti89(x):
    lo, hi = stats_intervals.eti_1d(x, eti_prob=stats_intervals.DEFAULT_CI_PROB)
    return np.mean(x), lo, hi


def main():
    g = posterior(VG19, [f"tau_subj_{k}_{p}" for k in "uq" for p in ("0", "1", "rho")])
    flat = posterior(VG20, ["tau_subj_u", "tau_subj_q"])

    for label, k in (("understood (logit u)", "u"), ("production ratio (logit q)", "q")):
        t0, t1, r = (g[f"tau_subj_{k}_0"], g[f"tau_subj_{k}_1"], g[f"tau_subj_{k}_rho"])
        print(f"\n######## {label}")
        m1, l1, u1 = eti89(t1)
        mr, lr, ur = eti89(r)
        print(f"  tau0 {t0.mean():.3f}  tau1 {m1:.3f} [{l1:.3f}, {u1:.3f}]  rho01 {mr:+.3f} [{lr:+.3f}, {ur:+.3f}]")
        print(f"  tau1/tau0 = {(t1 / t0).mean():.3f}    VG20's flat tau = {flat[f'tau_subj_{k}'].mean():.3f}")

        print("\n  between-child SD by age (VG19 | VG20 flat)")
        for a in AGES:
            ms, ls, us = eti89(sd(t0, t1, r, a))
            print(f"    {a:2d} mo  {ms:.3f} [{ls:.3f}, {us:.3f}] | {flat[f'tau_subj_{k}'].mean():.3f}")
        dmin = -r * t0 / t1
        mn, ln, un = eti89(REF_AGE + 12 * dmin)
        print(f"    narrowest at {mn:.1f} months [{ln:.1f}, {un:.1f}]")

        print("\n  corr(child effect at age A, at age B)")
        print("        " + "".join(f"{a:8d}" for a in AGES))
        for a in AGES:
            row = f"   {a:4d}"
            for b in AGES:
                c = cov(t0, t1, r, a, b) / (sd(t0, t1, r, a) * sd(t0, t1, r, b))
                row += f"{c.mean():8.2f}"
            print(row)

        for a0, a1 in ((24, 60), (36, 84)):
            beta = cov(t0, t1, r, a0, a1) / cov(t0, t1, r, a0, a0)
            s0, s1 = sd(t0, t1, r, a0), sd(t0, t1, r, a1)
            resid = np.sqrt(np.maximum(cov(t0, t1, r, a1, a1) - beta * cov(t0, t1, r, a0, a1), 0))
            rho_ab = cov(t0, t1, r, a0, a1) / (s0 * s1)
            print(f"\n  a child +1 SD at {a0} months sits at {(beta * s0 / s1).mean():+.2f} SD at {a1} months")
            print(f"    unpredictable residual at {a1} mo = {100 * (resid / s1).mean():.0f}% of that age's between-child SD")
            print(f"    P(two random children swap rank between {a0} and {a1} mo) = {100 * (np.arccos(rho_ab) / np.pi).mean():.1f}%")

    print("\n######## q on the natural scale, at the population median trajectory")
    t0, t1, r = g["tau_subj_q_0"], g["tau_subj_q_1"], g["tau_subj_q_rho"]
    pop = pd.read_csv(VG19 + "/posterior_summary_q.csv")
    age_col = next(c for c in pop.columns if "age" in c.lower())
    med_col = next(c for c in pop.columns if "median" in c.lower())
    for a in (24, 36, 48, 60, 72, 84):
        i = int(np.abs(pop[age_col].to_numpy() - a).argmin())
        qm = float(pop[med_col].iloc[i])
        lo = np.log(qm / (1 - qm))
        s = sd(t0, t1, r, a).mean()
        print(f"  {a:2d} mo: median q = {qm:.2f} -> 10th pct child {expit(lo - 1.2816 * s):.2f}, "
              f"90th pct child {expit(lo + 1.2816 * s):.2f}  (SD {s:.2f} logit)")


def spoken_spread():
    """Between-child spread in the SPOKEN outcome, where the two models cross over.

    Spoken is ``u * q``, so a child's spoken position depends on both random
    effects together. VG20 correlates them (``rho_uq`` = +0.368) but holds each
    constant in age; VG19 lets each grow with age but forces the correlation to
    zero. Neither dominates: the compounding wins at young ages and the growth
    wins at old ones. Simulated because ``log(expit(x) * expit(y))`` has no
    closed-form variance -- 400 posterior draws x 4000 synthetic children, at
    each model's own population median curve.
    """
    g = posterior(VG19, [f"tau_subj_{k}_{s}" for k in "uq" for s in ("0", "1", "rho")])
    f = posterior(VG20, ["tau_subj_u", "tau_subj_q", "rho_uq"])

    def curve(path, which, age):
        d = pd.read_csv(f"{path}/posterior_summary_{which}.csv")
        ac = next(c for c in d.columns if "age" in c.lower())
        mc = next(c for c in d.columns if "median" in c.lower())
        m = d[mc].to_numpy()[np.abs(d[ac].to_numpy() - age).argmin()]
        return np.log(m / (1 - m))

    rng = np.random.default_rng(11)
    ndraw, nkid = 400, 4000
    i19 = rng.choice(len(g["tau_subj_u_0"]), ndraw, replace=False)
    i20 = rng.choice(len(f["tau_subj_u"]), ndraw, replace=False)

    print("\n######## between-child spread in spoken words (u * q)")
    print(f"{'age':>4} | {'VG19 sd(log p)':>15} {'p10-p90 words':>15}"
          f" | {'VG20 sd(log p)':>15} {'p10-p90 words':>15}")
    for age in (24, 36, 48, 60, 72, 84):
        d_ = (age - REF_AGE) / 12.0
        out = {}
        for lab, path in (("VG19", VG19), ("VG20", VG20)):
            fu, fq = curve(path, "u", age), curve(path, "q", age)
            sds, p10, p90 = [], [], []
            for j in range(ndraw):
                if lab == "VG19":
                    i = i19[j]
                    su = sd(g["tau_subj_u_0"], g["tau_subj_u_1"], g["tau_subj_u_rho"], age)[i]
                    sq = sd(g["tau_subj_q_0"], g["tau_subj_q_1"], g["tau_subj_q_rho"], age)[i]
                    r = 0.0
                else:
                    i = i20[j]
                    su, sq, r = f["tau_subj_u"][i], f["tau_subj_q"][i], f["rho_uq"][i]
                z = rng.standard_normal((nkid, 2))
                ps = expit(fu + su * z[:, 0]) * expit(fq + sq * (r * z[:, 0] + np.sqrt(1 - r**2) * z[:, 1]))
                sds.append(np.std(np.log(ps)))
                p10.append(np.quantile(ps, 0.10))
                p90.append(np.quantile(ps, 0.90))
            out[lab] = (np.mean(sds), np.mean(p10) * 810, np.mean(p90) * 810)
        print(f"{age:4d} | {out['VG19'][0]:15.2f} {out['VG19'][1]:6.0f}-{out['VG19'][2]:<8.0f}"
              f" | {out['VG20'][0]:15.2f} {out['VG20'][1]:6.0f}-{out['VG20'][2]:<8.0f}")
        _ = d_


if __name__ == "__main__":
    main()
    spoken_spread()
