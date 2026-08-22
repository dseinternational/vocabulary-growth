#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What does VG16's cross-lag actually say, in interpretable units?

VG16 adds one term to VG09/VG10: the child's prior-wave understood count,
relative to the population + study expectation at that age, shifts the logit of
their current production ratio ``q`` by ``beta_lag * x_lag``. The fitted
coefficient is a number on the logit scale; this script translates it into
units a reader can weigh, and measures how much (and how little) the term does.

Sections printed, all from the fitted ``rep`` traces of VG16 and VG10:

1. **Identification base** — how many observations and children carry a
   prior-wave comprehension source, and over what gaps.
2. **Posterior summaries and scalings** — ``beta_lag`` per between-child SD of
   receptive standing (``tau_subj_u``), per observed SD of the predictor
   ``x_lag``, and as months of the population ``q`` trajectory.
3. **Two-channel decomposition** — a child ahead receptively says more words
   both because more are available (direct channel) and because a higher
   fraction of them is produced (cross-lag channel). The split is computed for
   several contrast sizes; the *share* of the spoken gap carried by the
   cross-lag channel is the framing-robust quantity.
4. **What the term does not do** — its variance share in the ``q`` logit;
   ``tau_subj_q`` against VG10's; the residual correlation between the
   understood and ``q`` subject intercepts, which the independent-prior
   structure cannot express; and the errors-in-variables attenuation implied
   by treating ``x_lag`` as a proxy for persistent standing.
5. **Trajectory comparison** — VG16 minus VG10 at the reporting ages. The two
   definitions differ only in ``use_cross_lag`` (``lag_baseline`` is inert when
   the lag is off), so this isolates the term's effect on everything the
   models of record report.

Usage::

    python scripts/experiments/vg16_crosslag_quantification.py [--output-dir <dir>] [--step 8]

``--step`` thins the posterior for the per-age interpolation loops; the default
of 8 keeps 4,500 of the 36,000 draws and reproduces the note's figures.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import xarray as xr

from vocab_growth import environment as env

VG10_DIR = "VG10-age-understood-spoken-ds-re-subj-uq-anchored"
VG16_DIR = "VG16-age-understood-spoken-ds-re-subj-uq-crosslag"
N_TRIALS = 810
AGES = [24, 30, 36, 42, 48, 54, 60, 72]

#: Quartiles of a normal: Q3 sits ``PHI_75`` SDs above the median. Hard-coded
#: rather than imported so the script has no scipy dependency.
PHI_75 = 0.6744897501960817


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def eti(v, label, fmt="{: .4f}"):
    lo, med, hi = np.percentile(v, [5.5, 50, 94.5])
    print(f"  {label:58s} {fmt.format(med)}  [{fmt.format(lo).strip()}, {fmt.format(hi).strip()}]")
    return med, lo, hi


def flat(post, name):
    """Posterior variable flattened to (draw, ...) with chain and draw merged."""
    v = post[name].values
    return v.reshape(-1, *v.shape[2:])


def prev_wave_lag(age, subj, und):
    """Replicate ``_compute_prev_wave_lag`` from the model engine on trace data."""
    n = len(age)
    prev_idx = np.zeros(n, dtype=int)
    has = np.zeros(n)
    row_order = np.arange(n)
    prev_subj, last, last_age = -1, -1, np.nan
    for pos in np.lexsort((row_order, age, subj)):
        if subj[pos] != prev_subj:
            prev_subj, last, last_age = subj[pos], -1, np.nan
        if last >= 0 and age[pos] > last_age:
            prev_idx[pos] = last
            has[pos] = 1.0
        if not np.isnan(und[pos]):
            last, last_age = pos, age[pos]
    p_prev = np.clip(np.where(has > 0, und[prev_idx], N_TRIALS * 0.5) / N_TRIALS, 1e-4, 1 - 1e-4)
    y_logit = np.where(has > 0, np.log(p_prev) - np.log(1 - p_prev), 0.0)
    return prev_idx, has, y_logit


def age_equivalent(target_logit, grid_logit, grid_age):
    """Age at which a monotone trajectory reaches ``target_logit`` (nan past its max)."""
    if target_logit > grid_logit.max():
        return np.nan
    return np.interp(target_logit, grid_logit, grid_age)


def decomposition(fu, h, tu, bl, Xp, k_lo, k_hi, title):
    """Two-channel split of the spoken gap between children at ``k_lo`` and ``k_hi`` SD.

    The direct channel pairs the upper child's comprehension with the *lower*
    child's conversion rate — a counterfactual decomposition step, not a
    prediction for any child.
    """
    print(f"\n--- {title} (delta_u = {k_lo:+.3f} -> {k_hi:+.3f} x tau_subj_u) ---")
    print(
        f"{'age':>4} {'und lo':>7} {'und hi':>7} {'gap mo':>7} | "
        f"{'spk lo':>7} {'direct':>7} {'spk hi':>7} {'x-lag adds':>16} {'share of gap':>18}"
    )
    for a in AGES:
        fa = np.array([np.interp(a, Xp, fu[d]) for d in range(fu.shape[0])])
        ha = np.array([np.interp(a, Xp, h[d]) for d in range(h.shape[0])])
        d_lo, d_hi = k_lo * tu, k_hi * tu
        u_lo, u_hi = sig(fa + d_lo) * N_TRIALS, sig(fa + d_hi) * N_TRIALS
        gap = np.array(
            [
                age_equivalent(fa[j] + d_hi[j], fu[j], Xp)
                - age_equivalent(fa[j] + d_lo[j], fu[j], Xp)
                for j in range(len(fa))
            ]
        )
        gap = gap[~np.isnan(gap)]
        s_lo = sig(fa + d_lo) * sig(ha + bl * d_lo) * N_TRIALS
        direct = sig(fa + d_hi) * sig(ha + bl * d_lo) * N_TRIALS
        s_hi = sig(fa + d_hi) * sig(ha + bl * d_hi) * N_TRIALS
        xl = s_hi - direct
        share = xl / (s_hi - s_lo)
        mx, lx, hx = np.percentile(xl, [50, 5.5, 94.5])
        mf, lf, hf = np.percentile(share, [50, 5.5, 94.5])
        print(
            f"{a:>4} {np.median(u_lo):>7.0f} {np.median(u_hi):>7.0f} {np.median(gap):>7.1f} | "
            f"{np.median(s_lo):>7.0f} {np.median(direct):>7.0f} {np.median(s_hi):>7.0f} "
            f"{mx:>6.0f} [{lx:.0f}, {hx:.0f}]  {100 * mf:>6.1f}% [{100 * lf:.0f}, {100 * hf:.0f}]"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--step", type=int, default=8, help="posterior thinning for per-age loops")
    args = ap.parse_args()
    env.set_output_root(args.output_dir)
    models = os.path.join(env.output_root(), "models")

    t16 = xr.open_datatree(os.path.join(models, VG16_DIR, "trace.nc"))
    t10 = xr.open_datatree(os.path.join(models, VG10_DIR, "trace.nc"))
    post, cd, od = (
        t16["posterior"].to_dataset(),
        t16["constant_data"].to_dataset(),
        t16["observed_data"].to_dataset(),
    )
    post10 = t10["posterior"].to_dataset()

    # ---- 1. identification base ------------------------------------------
    age = np.asarray(cd["X_obs"]).ravel()
    subj = np.asarray(cd["subject_obs"]).astype(int)
    study = np.asarray(cd["study_obs"]).astype(int)
    umask = np.asarray(cd["obs_u_mask"]).astype(bool)
    smask = np.asarray(cd["obs_s_mask"]).astype(bool)
    n = len(age)
    und = np.full(n, np.nan)
    und[umask] = np.asarray(od["y_u_obs"]).astype(float)
    prev_idx, has, y_logit = prev_wave_lag(age, subj, und)
    lagged = has > 0
    gaps = age[lagged] - age[prev_idx[lagged]]
    print("=== 1. identification base ===")
    print(f"  observations: {n}; with a prior-wave understood source: {int(has.sum())}")
    print(f"  children: {len(np.unique(subj))}; contributing a lagged observation: {len(np.unique(subj[lagged]))}")
    print(f"  of the lagged observations, with a spoken outcome: {int((lagged & smask).sum())}")
    print(
        f"  gap to lag source (months): median {np.median(gaps):.1f}, "
        f"IQR {np.percentile(gaps, 25):.1f}-{np.percentile(gaps, 75):.1f}, "
        f"range {gaps.min():.0f}-{gaps.max():.0f}"
    )
    print(
        f"  age at current wave: median {np.median(age[lagged]):.1f}, "
        f"range {age[lagged].min():.0f}-{age[lagged].max():.0f}"
    )

    # ---- 2. posterior summaries and scalings -----------------------------
    bl = flat(post, "beta_lag")
    tu = flat(post, "tau_subj_u")
    tq = flat(post, "tau_subj_q")
    Xp = np.asarray(cd["X_plot"]).ravel()
    fu = flat(post, "f_u_plot")
    h = flat(post, "h_plot")
    s = slice(None, None, args.step)
    print("\n=== 2. posterior summaries and scalings ===")
    eti(bl, "beta_lag")
    print(f"  {'P(beta_lag > 0)':58s} {np.mean(bl > 0): .4f}")
    eti(tu, "tau_subj_u")
    eti(tq, "tau_subj_q")
    eti(flat(post10, "tau_subj_u"), "tau_subj_u (VG10, no cross-lag)")
    eti(flat(post10, "tau_subj_q"), "tau_subj_q (VG10, no cross-lag)")

    # SD of x_lag under the population baseline, per (thinned) draw
    du = flat(post, "delta_u_raw") * flat(post, "tau_u")[:, None]
    ap_, sp_ = age[prev_idx], study[prev_idx]
    sds = np.array(
        [(y_logit - (np.interp(ap_, Xp, fu[d]) + du[d][sp_]))[lagged].std() for d in range(0, fu.shape[0], args.step)]
    )
    eti(sds, "SD of x_lag over the lagged observations")
    eti(bl * tu, "beta_lag * tau_subj_u  (logits on q, per SD of standing)")
    eti(bl * tu / tq, "  ... as a fraction of tau_subj_q")
    eti(bl[s][: len(sds)] * sds, "beta_lag * SD(x_lag)   (logits on q, per SD of predictor)")

    print("\n  cross-lag shift beta_lag * tau_subj_u as months of the population q trajectory:")
    for a in [24, 30, 36, 42, 48]:
        out = []
        for d in range(0, h.shape[0], args.step):
            ha = np.interp(a, Xp, h[d])
            ae = age_equivalent(ha + bl[d] * tu[d], h[d], Xp)
            if not np.isnan(ae):
                out.append(ae - a)
        o = np.array(out)
        print(f"    age {a}: +{np.median(o):.1f} months [{np.percentile(o, 5.5):.1f}, {np.percentile(o, 94.5):.1f}]")
    print("  (beyond ~54 months q saturates and the age-equivalence is unstable; not quoted)")

    # ---- 3. two-channel decomposition ------------------------------------
    print("\n=== 3. two-channel decomposition of the spoken gap ===")
    fut, ht = fu[s], h[s]
    tut, blt = tu[s][: fut.shape[0]], bl[s][: fut.shape[0]]
    decomposition(fut, ht, tut, blt, Xp, 0.0, 0.5, "+0.5 SD vs population")
    decomposition(fut, ht, tut, blt, Xp, 0.0, PHI_75, "Q3 vs median")
    decomposition(fut, ht, tut, blt, Xp, 0.0, 1.0, "+1 SD vs population")
    decomposition(fut, ht, tut, blt, Xp, -PHI_75, PHI_75, "Q3 vs Q1")

    # ---- 4. what the term does not do ------------------------------------
    print("\n=== 4. what the cross-lag does not do ===")
    v_lag = (bl[s][: len(sds)] * sds) ** 2
    v_q = tq[s][: len(sds)] ** 2
    eti(100 * v_lag / (v_lag + v_q), "variance share of beta_lag*x_lag in the q logit (%)", "{: .2f}")
    du_s = flat(post, "delta_subj_u_raw") * tu[:, None]
    dq_s = flat(post, "delta_subj_q_raw") * tq[:, None]
    cors = np.array(
        [np.corrcoef(du_s[d], dq_s[d])[0, 1] for d in range(0, du_s.shape[0], max(args.step, 20))]
    )
    eti(cors, "residual corr(delta_subj_u, delta_subj_q) across children")
    rel = tu[s][: len(sds)] ** 2 / sds**2
    eti(rel, "reliability of x_lag for persistent standing")
    eti(bl[s][: len(sds)] / rel, "disattenuated beta_lag")

    # ---- 5. trajectory comparison ----------------------------------------
    print("\n=== 5. reported trajectories: VG16 minus VG10 (population level) ===")
    Xq = np.asarray(t10["constant_data"].to_dataset()["X_query"]).ravel()
    print(f"{'age':>5} {'d understood (words)':>21} {'d q (pp)':>10} {'d spoken (words)':>17}")
    for i, a in enumerate(Xq):
        if a > 84:
            continue
        d_u = np.median(flat(post, "p_u_query")[:, i]) * N_TRIALS - np.median(flat(post10, "p_u_query")[:, i]) * N_TRIALS
        d_q = 100 * (np.median(flat(post, "q_query")[:, i]) - np.median(flat(post10, "q_query")[:, i]))
        d_s = np.median(flat(post, "p_s_query")[:, i]) * N_TRIALS - np.median(flat(post10, "p_s_query")[:, i]) * N_TRIALS
        print(f"{a:>5.0f} {d_u:>21.2f} {d_q:>10.2f} {d_s:>17.2f}")

    t16.close()
    t10.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
