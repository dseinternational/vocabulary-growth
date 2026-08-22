#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does VG16's within-child cross-lag baseline carry a short-T bias? Simulate and see.

VG16 reports the **population-relative** cross-lag baseline. Its **within-child**
(RI-CLPM) alternative — which additionally subtracts the child's *own* estimated
understood intercept — came out strongly negative when fitted as a diagnostic
(`beta` ~ -0.60, 89% [-0.85, -0.35] at `dev`), and the report attributes that to a
short-T (Nickell-type) / errors-in-variables artefact. **That attribution has
only ever been reasoned, never demonstrated**, and the remedy for the bias
depends on which mechanism is real.

This is Stage 1 of `notes/202608151500-within-child-crosslag-feasibility.md`: it
simulates outcomes from VG16's own posterior on the **real observed wave
structure** at a *known* `beta_lag` — including `beta_lag = 0` — and then applies
both baselines to the same simulated data. That last point is what makes the
result interpretable: the two estimators differ only in whether the child's own
intercept is subtracted, so any divergence between them is the mechanism under
test and not a simulation artefact.

Why a bespoke simulator rather than `scripts/fit_recovery.py`: VG16 is
deliberately excluded from that harness because its cross-lag predictor is a
function of the outcome. Simulation therefore has to walk each child's waves in
age order, deriving `x_lag` at wave t from the *already simulated* understood
count at wave t-1. That sequential dependence is the whole point here, so it is
built explicitly below.

**What this does and does not establish.** The estimators are marginal
likelihood, with the child effects integrated out by Gauss-Hermite quadrature
and the population trajectory, study effects and dispersion held at their fitted
values. They reproduce the *structure* that generates the bias — a child
intercept estimated from the same waves that supply the lag — and so speak to
its sign, its rough size and its dependence on wave count. They are not the full
Bayesian joint fit VG16 runs, so a magnitude here should not be quoted as if it
were VG16's own.

**A first attempt used the obvious shortcut and got the sign wrong.** Regressing
`logit(y_s / y_u)` on the lag returned a *positive* bias under every truth,
including zero. The cause is that 159 of 973 conditional rows (16%) have zero
spoken words, and `logit(0)` clips to -9.21 — a large negative outlier arising
for exactly the small-vocabulary children who also have a low lag value, which
manufactures positive correlation. Anything that reduces the outcome to a ratio
inherits this. The beta-binomial likelihood below handles a zero count as a
zero count, which is why it is used.

Usage::

    python scripts/experiments/vg16_within_lag_bias.py [--replicates 200] [--seed 20260815]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import xarray as xr
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from vocab_growth import environment as env

VG16_DIR = "VG16-age-understood-spoken-ds-re-subj-uq-crosslag"
N_TRIALS = 810
EPS = 1e-4

#: Truths to simulate at. 0.0 is the honest null — if the within estimator
#: returns a large negative number here, the bias is established outright.
TRUTHS = (0.0, 0.203, 0.400)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log(1 - p)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def rbetabinom(rng, n, p, kappa):
    """Beta-binomial draw, vectorised over observations."""
    p = np.clip(p, EPS, 1 - EPS)
    theta = rng.beta(p * kappa, (1 - p) * kappa)
    return rng.binomial(np.maximum(n.astype(int), 0), theta)


def load_truth(root):
    """Population trajectory, study effects and dispersion at VG16's posterior mean."""
    t = xr.open_datatree(os.path.join(root, "models", VG16_DIR, "trace.nc"))
    post, cd = t["posterior"].to_dataset(), t["constant_data"].to_dataset()

    def m(name):
        return np.asarray(post[name].mean(dim=("chain", "draw")).values)

    truth = {
        "Xp": np.asarray(cd["X_plot"]).ravel(),
        "f_u_plot": m("f_u_plot"),
        "h_plot": m("h_plot"),
        "kappa_u_plot": m("kappa_u_plot"),
        "kappa_s_plot": m("kappa_s_plot"),
        "delta_u": m("delta_u_raw") * float(m("tau_u")),
        "delta_q": m("delta_q_raw") * float(m("tau_q")),
        "tau_subj_u": float(m("tau_subj_u")),
        "tau_subj_q": float(m("tau_subj_q")),
    }
    design = {
        "age": np.asarray(cd["X_obs"]).ravel(),
        "subj": np.asarray(cd["subject_obs"]).astype(int),
        "study": np.asarray(cd["study_obs"]).astype(int),
        "umask": np.asarray(cd["obs_u_mask"]).astype(bool),
        "smask": np.asarray(cd["obs_s_mask"]).astype(bool),
    }
    # s_is_conditional is stored on the spoken-only index; expand to all rows.
    cond_s = np.asarray(cd["s_is_conditional"]).astype(bool)
    cond = np.zeros(len(design["age"]), dtype=bool)
    cond[np.where(design["smask"])[0]] = cond_s
    design["cond"] = cond
    # Per-observation population + study values, used by both the simulator and
    # the estimators. The estimators hold these at their fitted values: the
    # question is the bias from the lag construction, not from re-estimating the
    # trajectory.
    age, study = design["age"], design["study"]
    design["f_u"] = np.interp(age, truth["Xp"], truth["f_u_plot"]) + truth["delta_u"][study]
    design["h"] = np.interp(age, truth["Xp"], truth["h_plot"]) + truth["delta_q"][study]
    design["k_u"] = np.interp(age, truth["Xp"], truth["kappa_u_plot"])
    design["k_s"] = np.interp(age, truth["Xp"], truth["kappa_s_plot"])
    t.close()
    return truth, design


def child_order(age, subj):
    """Row order walking each child's waves in age order, and the wave index."""
    order = np.lexsort((np.arange(len(age)), age, subj))
    return order


def simulate(rng, truth, design, beta_lag, baseline):
    """Simulate understood and spoken counts at a known ``beta_lag``.

    Walks each child in age order so ``x_lag`` at wave t is built from the
    understood count simulated at wave t-1, exactly as the engine builds it from
    the observed one. ``baseline`` selects the data-generating baseline; the
    estimators below are applied to whatever is generated here.
    """
    age, subj = design["age"], design["subj"]
    umask, cond = design["umask"], design["cond"]
    n = len(age)
    f_u, h = design["f_u"], design["h"]
    k_u, k_s = design["k_u"], design["k_s"]

    n_child = int(subj.max()) + 1
    d_u = rng.normal(0.0, truth["tau_subj_u"], n_child)
    d_q = rng.normal(0.0, truth["tau_subj_q"], n_child)

    y_u = np.zeros(n, dtype=int)
    y_s = np.zeros(n, dtype=int)
    x_lag = np.zeros(n)
    has = np.zeros(n, dtype=bool)

    prev_subj, last, last_age = -1, -1, np.nan
    for pos in child_order(age, subj):
        i = subj[pos]
        if i != prev_subj:
            prev_subj, last, last_age = i, -1, np.nan

        p_u = sig(f_u[pos] + d_u[i])
        y_u[pos] = rbetabinom(rng, np.array([N_TRIALS]), np.array([p_u]), k_u[pos])[0]

        if last >= 0 and age[pos] > last_age:
            has[pos] = True
            base = f_u[last]  # population + study at the prior wave
            if baseline == "within":
                base = base + d_u[i]
            x_lag[pos] = logit(y_u[last] / N_TRIALS) - base

        q = sig(h[pos] + d_q[i] + beta_lag * x_lag[pos])
        if cond[pos]:
            y_s[pos] = rbetabinom(rng, np.array([y_u[pos]]), np.array([q]), k_s[pos])[0]
        else:
            y_s[pos] = rbetabinom(
                rng, np.array([N_TRIALS]), np.array([p_u * q]), k_s[pos]
            )[0]

        if umask[pos]:
            last, last_age = pos, age[pos]

    return {"y_u": y_u, "y_s": y_s, "x_lag_true": x_lag, "has_true": has,
            "d_u": d_u, "d_q": d_q}


#: Gauss-Hermite nodes for integrating out the child random effects. 24 is
#: comfortably enough for a smooth one-dimensional normal integral.
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(24)
_GH_WEIGHTS = _GH_WEIGHTS / _GH_WEIGHTS.sum()


def _bb_logpmf(k, n, p, kappa):
    """Beta-binomial log pmf, vectorised, with the k-independent terms dropped.

    Only differences in ``p`` matter for the estimates below, so the binomial
    coefficient is omitted.
    """
    p = np.clip(p, EPS, 1 - EPS)
    a, b = p * kappa, (1 - p) * kappa
    return (
        gammaln(k + a) + gammaln(n - k + b) - gammaln(n + a + b)
        - (gammaln(a) + gammaln(b) - gammaln(a + b))
    )


def _posterior_mean_d_u(truth, design, sim):
    """Posterior mean of each child's understood intercept, from its own waves.

    This is the step the short-T argument holds responsible, so it is computed
    on the actual beta-binomial likelihood rather than a normal approximation
    to a logit residual.
    """
    subj, umask = design["subj"], design["umask"]
    f_u = design["f_u"]
    idx = np.where(umask)[0]
    tau = truth["tau_subj_u"]
    nodes = _GH_NODES * tau
    # (n_obs_u, n_nodes) log-likelihood contributions
    ll = _bb_logpmf(
        sim["y_u"][idx][:, None],
        N_TRIALS,
        sig(f_u[idx][:, None] + nodes[None, :]),
        design["k_u"][idx][:, None],
    )
    n_child = int(subj.max()) + 1
    acc = np.zeros((n_child, nodes.size))
    np.add.at(acc, subj[idx], ll)
    acc -= acc.max(axis=1, keepdims=True)
    w = np.exp(acc) * _GH_WEIGHTS[None, :]
    w /= w.sum(axis=1, keepdims=True)
    return w @ nodes


def _fit_beta(truth, design, sim, x, use):
    """Marginal-likelihood estimate of ``beta_lag`` with the child `q` effect integrated out.

    The outcome is modelled on its own beta-binomial likelihood — spoken given
    understood — so the 16% of conditional rows with zero spoken words are
    handled correctly instead of becoming clipped logit outliers.
    """
    subj = design["subj"][use]
    y_s, n_s = sim["y_s"][use], sim["y_u"][use]
    h, k_s, xu = design["h"][use], design["k_s"][use], x[use]
    nodes = _GH_NODES * truth["tau_subj_q"]
    uniq, gi = np.unique(subj, return_inverse=True)

    def nll(beta):
        p = sig((h + beta * xu)[:, None] + nodes[None, :])
        ll = _bb_logpmf(y_s[:, None], n_s[:, None], p, k_s[:, None])
        acc = np.zeros((uniq.size, nodes.size))
        np.add.at(acc, gi, ll)
        mx = acc.max(axis=1, keepdims=True)
        return -float(np.sum(mx.ravel() + np.log(np.exp(acc - mx) @ _GH_WEIGHTS)))

    res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x) if res.success else np.nan


def estimate(truth, design, sim, baseline):
    """Estimate ``beta_lag`` under the chosen baseline, by marginal likelihood."""
    age, subj = design["age"], design["subj"]
    umask, smask, cond = design["umask"], design["smask"], design["cond"]
    n = len(age)
    y_u = sim["y_u"]

    if baseline == "within":
        d_u_hat = _posterior_mean_d_u(truth, design, sim)
    elif baseline == "within-oracle":
        # The child's TRUE intercept. If the plug-in and oracle variants agree,
        # estimation error in that intercept — the errors-in-variables half of
        # the attributed mechanism — is not producing the bias.
        d_u_hat = sim["d_u"]
    else:
        d_u_hat = None

    prev_idx = np.zeros(n, dtype=int)
    has = np.zeros(n, dtype=bool)
    prev_subj, last, last_age = -1, -1, np.nan
    for pos in child_order(age, subj):
        if subj[pos] != prev_subj:
            prev_subj, last, last_age = subj[pos], -1, np.nan
        if last >= 0 and age[pos] > last_age:
            prev_idx[pos], has[pos] = last, True
        if umask[pos]:
            last, last_age = pos, age[pos]
    base = design["f_u"][prev_idx]
    if d_u_hat is not None:
        base = base + d_u_hat[subj]
    x = np.where(has, logit(y_u[prev_idx] / N_TRIALS) - base, 0.0)

    use = has & smask & cond & (y_u > 0)
    if use.sum() < 20:
        return np.nan
    return _fit_beta(truth, design, sim, x, use)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    env.set_output_root(args.output_dir)
    truth, design = load_truth(env.output_root())

    print(f"design: {len(design['age'])} observations, "
          f"{int(design['subj'].max()) + 1} children, "
          f"{int(design['umask'].sum())} understood, {int(design['smask'].sum())} spoken\n")
    cols = ("population", "within", "within-oracle")
    print(f"{'truth':>7} {'generated':>12} |" + "".join(f"{c:>28} |" for c in cols))
    print(f"{'':>7} {'':>12} |" + "".join(f"{'mean':>9}{'bias':>10}{'sd':>8} |" for _ in cols))

    rng = np.random.default_rng(args.seed)
    for gen in ("population", "within"):
        for b in TRUTHS:
            acc = {c: [] for c in cols}
            for _ in range(args.replicates):
                sim = simulate(rng, truth, design, b, gen)
                for c in cols:
                    acc[c].append(estimate(truth, design, sim, c))
            line = f"{b:>7.3f} {gen:>12} |"
            for c in cols:
                v = np.array(acc[c])
                v = v[~np.isnan(v)]
                line += f"{v.mean():>9.3f}{v.mean() - b:>+10.3f}{v.std():>8.3f} |"
            print(line)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
