# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Conditional (GLMM) calibration of the Beta-Binomial dispersion prior.

`scripts/`'s companion marginal calibration estimates the *total* spread of
counts at an age: a free proportion per integer-age cell, one shared dispersion
curve, nothing else. That is the right target for a model with no grouping
structure (VG01, VG03), and it is what
``notes/202608020829-kappa-and-eta-q-prior-recalibration.md`` §§1-2 and §18 used.

It is the wrong target for a model that carries study and subject random
intercepts. Those effects absorb between-child spread before the likelihood sees
it, so the model's `kappa` governs a much smaller residual and the marginal
estimate is a lower bound -- by a factor of ten on VG11. This module puts the
random effects into the calibration instead:

    logit p_ij = m_c(ij) + s_k(i) + b_i,     b_i ~ N(0, tau^2)
    y_ij       ~ BetaBinomial(N_ij, p_ij, kappa(a_ij))

`m_c` is a *saturated* mean -- one free level per integer-age cell -- so that
dispersion is estimated given whatever the mean does, rather than inheriting a
mean-model choice. Study effects are fixed and sum-to-zero, matching the
engines' near-unshrunk ``ZeroSumNormal`` intercepts. The subject effect is
integrated out by Gauss-Hermite quadrature, the whole log-likelihood is
differentiated with JAX, and the maximum is found with L-BFGS.

`kappa` is parameterised directly by the three quantities the two-anchor prior
needs -- the floor and the age term at each of two anchor ages -- so the Hessian
returns standard errors on those and not on some reparameterisation of them.

**The design has to be able to tell `tau` from `kappa` before any of this means
anything.** For a child measured once they add variance to the same single
number; only children with a repeat separate them. ``--recover`` simulates from
a known truth on a real frame and refits, which is what established that they
*are* separable here (and that 24 quadrature nodes are not enough -- see
``--nodes``). Run it before trusting a new pool.

Usage:
    python scripts/kappa_conditional_calibration.py                 # every pool
    python scripts/kappa_conditional_calibration.py vg11-spoken
    python scripts/kappa_conditional_calibration.py --recover vg11-spoken
    python scripts/kappa_conditional_calibration.py --mean-sweep vg12-understood
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import betaln, logsumexp
from scipy.optimize import minimize

import vocab_growth.data_utils as du
from vocab_growth.models import definitions as defs

# Counts near 810 with kappa in the hundreds put the Beta-Binomial's betaln
# differences well inside float32's resolution; the optimiser's gradients are
# meaningless without this.
jax.config.update("jax_enable_x64", True)

N_TRIALS = 810

#: Gauss-Hermite nodes. 24 is *not* enough where tau is near 1 -- it biased
#: VG11's young anchor down by 17% against a simulated truth. Everything
#: converges by 96-160; 160 is used so a new pool does not silently repeat that.
DEFAULT_NODES = 160

#: Minimum observations in an integer-age cell for the saturated mean to use it.
MIN_CELL = 15


# --------------------------------------------------------------------------
# design
# --------------------------------------------------------------------------


class Design:
    """One outcome's rows, grouped by subject for the quadrature."""

    def __init__(self, age, y, n_trials, subject, study, *, min_cell=MIN_CELL,
                 mean="saturated", n_knots=8):
        age = np.asarray(age, float)
        y = np.asarray(y, float)
        n_trials = np.asarray(n_trials, float)

        keep = np.isfinite(age) & np.isfinite(y) & (n_trials > 0)
        age, y, n_trials = age[keep], y[keep], n_trials[keep]
        subject = np.asarray(subject)[keep]
        study = np.asarray(study)[keep]

        cell_age = np.round(age).astype(int)
        counts = dict(zip(*np.unique(cell_age, return_counts=True), strict=True))
        keep = np.array([counts[a] >= min_cell for a in cell_age])
        self.n_dropped = int((~keep).sum())
        age, y, n_trials = age[keep], y[keep], n_trials[keep]
        subject, study, cell_age = subject[keep], study[keep], cell_age[keep]

        order = np.argsort(subject, kind="stable")
        age, y, n_trials = age[order], y[order], n_trials[order]
        subject, study, cell_age = subject[order], study[order], cell_age[order]

        self.cells, self.cell_idx = np.unique(cell_age, return_inverse=True)
        self.studies, self.study_idx = np.unique(study, return_inverse=True)
        self.subjects, self.subject_idx = np.unique(subject, return_inverse=True)

        self.age = age
        self.y = y
        self.n_trials = n_trials
        self.n_obs = len(y)
        self.n_subjects = len(self.subjects)
        self.n_repeat = int((np.bincount(self.subject_idx) > 1).sum())
        self.cell_counts = np.bincount(self.cell_idx)
        self.n_excluded_scale_rows = 0
        self.mean = mean
        self.B = self._mean_basis(mean, n_knots)

    def _mean_basis(self, mean, n_knots):
        """Design matrix for the age mean.

        ``saturated`` spends one parameter per age cell, which the DS joint
        frame (671 rows over a 12-46 month span) cannot afford; ``spline`` is
        the cheaper alternative for those. The two agree to ~3% on every frame
        dense enough to carry both -- ``--mean-sweep`` is what checks that, and
        it is also what showed the DS frame is *not* stable enough to calibrate
        from.
        """
        if mean == "saturated":
            B = np.zeros((self.n_obs, len(self.cells)))
            B[np.arange(self.n_obs), self.cell_idx] = 1.0
            return B
        if mean == "spline":
            from scipy.interpolate import BSpline

            lo, hi = float(self.age.min()), float(self.age.max())
            inner = np.unique(np.quantile(self.age, np.linspace(0, 1, n_knots)[1:-1]))
            inner = inner[(inner > lo) & (inner < hi)]
            knots = np.concatenate([[lo] * 4, inner, [hi] * 4])
            n_basis = len(knots) - 4
            eye = np.eye(n_basis)
            return np.column_stack([
                BSpline(knots, eye[j], 3, extrapolate=True)(self.age)
                for j in range(n_basis)
            ])
        raise ValueError(f"unknown mean basis {mean!r}")

    def describe(self):
        return (
            f"n={self.n_obs:,}  subjects={self.n_subjects:,} "
            f"({self.n_repeat:,} with a repeat, "
            f"{self.n_obs / self.n_subjects:.2f} obs/subject)  "
            f"studies={len(self.studies)}  cells={len(self.cells)} "
            f"({self.cells.min()}-{self.cells.max()} mo)  "
            f"dropped={self.n_dropped}  mean={self.mean}[{self.B.shape[1]}]"
        )


# --------------------------------------------------------------------------
# likelihood
# --------------------------------------------------------------------------


def _layout(design):
    n_mean = design.B.shape[1]
    n_study_free = max(len(design.studies) - 1, 0)
    return {
        "m": slice(0, n_mean),
        "s": slice(n_mean, n_mean + n_study_free),
        "log_tau": n_mean + n_study_free,
        "log_kmin": n_mean + n_study_free + 1,
        "log_ey": n_mean + n_study_free + 2,
        "log_eo": n_mean + n_study_free + 3,
        "n_params": n_mean + n_study_free + 4,
    }


def make_objective(design, anchor_ages, *, n_nodes=DEFAULT_NODES, tau_fixed=None):
    """Build the JAX negative log-likelihood for one design."""
    layout = _layout(design)
    young, old = float(anchor_ages[0]), float(anchor_ages[1])
    if not old > young:
        raise ValueError(f"anchor_ages must be ordered (young, old); got {anchor_ages!r}")

    # The interpolation weight is in *months*, so the curve is the same function
    # of age whatever standardisation a pool happens to have -- the property the
    # two-anchor form exists for (see gp_utils.build_kappa_of_z_anchored).
    w = jnp.asarray((design.age - young) / (old - young))

    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    nodes = jnp.asarray(nodes)
    log_w = jnp.asarray(np.log(weights / np.sqrt(np.pi)))

    y = jnp.asarray(design.y)
    n_trials = jnp.asarray(design.n_trials)
    B = jnp.asarray(design.B)
    study_idx = jnp.asarray(design.study_idx)
    subject_idx = jnp.asarray(design.subject_idx)
    n_studies = len(design.studies)
    n_subjects = design.n_subjects

    # log C(n, y), via C(n, y) = 1 / ((n + 1) * B(n - y + 1, y + 1)). Constant in
    # the parameters, so it changes neither the optimum nor any likelihood ratio;
    # carried so the reported nll is the actual negative log-likelihood and can be
    # checked against scipy's betabinom.logpmf.
    log_binom = -jnp.log(n_trials + 1) - betaln(n_trials - y + 1.0, y + 1.0)

    def nll(theta):
        m = theta[layout["m"]]
        s_free = theta[layout["s"]]
        s = (jnp.concatenate([s_free, -jnp.sum(s_free)[None]])
             if n_studies > 1 else jnp.zeros(1))
        tau = (jnp.exp(theta[layout["log_tau"]]) if tau_fixed is None
               else jnp.asarray(tau_fixed))
        log_ey, log_eo = theta[layout["log_ey"]], theta[layout["log_eo"]]

        kappa = jnp.exp(theta[layout["log_kmin"]]) + jnp.exp(
            log_ey + w * (log_eo - log_ey)
        )

        eta = (B @ m + s[study_idx])[:, None] + (jnp.sqrt(2.0) * tau * nodes)[None, :]
        p = jnp.clip(jax.nn.sigmoid(eta), 1e-12, 1 - 1e-12)
        a = p * kappa[:, None]
        b = (1 - p) * kappa[:, None]
        logpmf = (
            log_binom[:, None]
            + betaln(y[:, None] + a, n_trials[:, None] - y[:, None] + b)
            - betaln(a, b)
        )

        per_subject = jnp.zeros((n_subjects, n_nodes)).at[subject_idx].add(logpmf)
        return -jnp.sum(logsumexp(per_subject + log_w[None, :], axis=1))

    return jax.jit(nll), layout


@dataclass
class Result:
    design: Design
    anchor_ages: tuple
    nll: float
    tau: float
    kappa_min: float
    excess_young: float
    excess_old: float
    theta: np.ndarray
    converged: bool
    se: np.ndarray | None = None
    corr: np.ndarray | None = None

    @property
    def kappa_young(self):
        return self.kappa_min + self.excess_young

    @property
    def kappa_old(self):
        return self.kappa_min + self.excess_old

    def kappa_at(self, age):
        young, old = self.anchor_ages
        w = (np.asarray(age, float) - young) / (old - young)
        ly, lo = np.log(self.excess_young), np.log(self.excess_old)
        return self.kappa_min + np.exp(ly + w * (lo - ly))

    def summary(self, label=""):
        ya, oa = self.anchor_ages
        lines = [
            f"  {label}nll={self.nll:,.2f}  tau={self.tau:.3f}  "
            f"kappa_min={self.kappa_min:.2f}"
            + ("" if self.converged else "   [DID NOT CONVERGE]"),
            f"    kappa({ya:.0f} mo)={self.kappa_young:8.1f} "
            f"(excess {self.excess_young:.1f})    "
            f"kappa({oa:.0f} mo)={self.kappa_old:8.1f} "
            f"(excess {self.excess_old:.1f})",
        ]
        if self.se is not None:
            lines.append(
                f"    log-scale SE: tau {self.se[0]:.3f}  "
                f"kappa_min {self.se[1]:.3f}  excess_young {self.se[2]:.3f}  "
                f"excess_old {self.se[3]:.3f}"
            )
        if self.corr is not None:
            lines.append(
                f"    corr(log tau, .): kappa_min {self.corr[0, 1]:+.3f}  "
                f"excess_young {self.corr[0, 2]:+.3f}  "
                f"excess_old {self.corr[0, 3]:+.3f}"
            )
        return "\n".join(lines)


def fit(design, anchor_ages, *, n_nodes=DEFAULT_NODES, tau_fixed=None, x0=None,
        standard_errors=False):
    """Maximise the marginal likelihood; return the anchored dispersion at its optimum."""
    nll, layout = make_objective(
        design, anchor_ages, n_nodes=n_nodes, tau_fixed=tau_fixed
    )
    grad = jax.jit(jax.grad(nll))

    if x0 is None:
        theta0 = np.zeros(layout["n_params"])
        prop = np.clip(design.y / design.n_trials, 1e-4, 1 - 1e-4)
        theta0[layout["m"]] = np.linalg.lstsq(
            design.B, np.log(prop / (1 - prop)), rcond=None
        )[0]
        theta0[layout["log_tau"]] = np.log(0.5)
        theta0[layout["log_kmin"]] = np.log(3.0)
        theta0[layout["log_ey"]] = np.log(30.0)
        theta0[layout["log_eo"]] = np.log(3.0)
    else:
        theta0 = np.asarray(x0, float).copy()

    def f(t):
        t = jnp.asarray(t)
        return float(nll(t)), np.asarray(grad(t), float)

    opts = {"maxiter": 4000, "maxfun": 6000, "ftol": 1e-14, "gtol": 1e-9}
    res = minimize(f, theta0, jac=True, method="L-BFGS-B", options=opts)
    # a second pass from the optimum guards against an early L-BFGS stop
    res = minimize(f, res.x, jac=True, method="L-BFGS-B", options=opts)

    se = corr = None
    if standard_errors:
        idx = [layout["log_tau"], layout["log_kmin"],
               layout["log_ey"], layout["log_eo"]]
        try:
            cov = np.linalg.inv(np.asarray(jax.hessian(nll)(jnp.asarray(res.x)), float))
            sub = cov[np.ix_(idx, idx)]
            se = np.sqrt(np.abs(np.diag(sub)))
            corr = sub / np.outer(se, se)
        except np.linalg.LinAlgError:
            pass

    return Result(
        design=design,
        anchor_ages=(float(anchor_ages[0]), float(anchor_ages[1])),
        nll=float(res.fun),
        tau=(float(tau_fixed) if tau_fixed is not None
             else float(np.exp(res.x[layout["log_tau"]]))),
        kappa_min=float(np.exp(res.x[layout["log_kmin"]])),
        excess_young=float(np.exp(res.x[layout["log_ey"]])),
        excess_old=float(np.exp(res.x[layout["log_eo"]])),
        theta=np.asarray(res.x, float),
        converged=bool(res.success),
        se=se,
        corr=corr,
    )


def simulate(design, *, tau, kappa_min, excess_young, excess_old, anchor_ages, seed):
    """Draw counts from the GLMM at a known truth, on a real design."""
    rng = np.random.default_rng(seed)
    prop = np.bincount(design.cell_idx, weights=design.y / design.n_trials)
    prop = np.clip(prop / design.cell_counts, 1e-4, 1 - 1e-4)
    m = np.log(prop / (1 - prop))

    s = rng.normal(0.0, 0.25, len(design.studies))
    s -= s.mean()
    b = rng.normal(0.0, tau, design.n_subjects)

    eta = m[design.cell_idx] + s[design.study_idx] + b[design.subject_idx]
    p = 1.0 / (1.0 + np.exp(-eta))

    young, old = anchor_ages
    w = (design.age - young) / (old - young)
    ly, lo = np.log(excess_young), np.log(excess_old)
    kappa = kappa_min + np.exp(ly + w * (lo - ly))

    p_draw = rng.beta(p * kappa, (1 - p) * kappa)
    return rng.binomial(design.n_trials.astype(int), p_draw).astype(float)


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------


def _subject_key(df):
    return (df["study"].astype(str) + "::" + df["subject_id"].astype(str)).to_numpy()


def univariate_frame(definition, **basis):
    """VG11 / VG12: one outcome out of 810, study + subject random intercepts."""
    y_col = definition.outcome.value
    columns = ["age", y_col, "study", "subject_id"]
    df = du.load_data(
        population=definition.population,
        columns=columns,
        sample_fraction=definition.sample_fraction,
        random_seed=definition.random_seed,
    )
    df = df[columns].dropna(subset=["age", y_col]).reset_index(drop=True)
    df, _ = du.filter_studies_by_min_obs(df, definition.min_study_observations)
    return Design(
        age=df["age"].to_numpy(float),
        y=df[y_col].to_numpy(float),
        n_trials=np.full(len(df), float(N_TRIALS)),
        subject=_subject_key(df),
        study=df["study"].to_numpy(),
        **basis,
    )


def bivariate_frames(definition, **basis):
    """Joint models: understood out of 810, and spoken out of understood (q).

    Mirrors ``prepare_bivariate_re_data`` -- same columns, age bound, us_01
    ceiling exclusion and study minimum. The spoken design is the *nested* scale
    of ``nested_outcome_spec``: spoken successes out of that child's observed
    understood count, with mean q. Rows the engine falls back to the marginal
    (spoken-out-of-810) likelihood for are excluded and counted in
    ``n_excluded_scale_rows``; on the DS frame that is a large minority, which is
    one reason its `kappa_s` is not calibrated from here.
    """
    columns = ["age", "understood", "spoken", "study", "subject_id"]
    load_columns = list(columns)
    if definition.exclude_us01_spoken_ceiling:
        load_columns.append("survey_vocab_max")
    df = du.load_data(
        population=definition.population,
        columns=load_columns,
        sample_fraction=definition.sample_fraction,
        random_seed=definition.random_seed,
        max_age_months=definition.max_age_months,
        include_implausible_production=definition.include_implausible_production,
    )
    if definition.exclude_us01_spoken_ceiling:
        df, _ = du.exclude_us01_spoken_ceiling_rows(df)
    df = df[columns].copy().dropna(subset=["age"])
    df = df[df["understood"].notna() | df["spoken"].notna()].reset_index(drop=True)
    df, _ = du.filter_studies_by_min_obs(df, definition.min_study_observations)

    u = df.dropna(subset=["understood"])
    design_u = Design(
        age=u["age"].to_numpy(float),
        y=u["understood"].to_numpy(float),
        n_trials=np.full(len(u), float(N_TRIALS)),
        subject=_subject_key(u),
        study=u["study"].to_numpy(),
        **basis,
    )

    both = df.dropna(subset=["understood", "spoken"])
    nested = both[
        (both["understood"] > 0)  # n = 0 contributes nothing to the likelihood
        & (both["understood"] <= N_TRIALS)
        & (both["spoken"] <= both["understood"])
    ]
    design_s = Design(
        age=nested["age"].to_numpy(float),
        y=nested["spoken"].to_numpy(float),
        n_trials=nested["understood"].to_numpy(float),
        subject=_subject_key(nested),
        study=nested["study"].to_numpy(),
        **basis,
    )
    design_s.n_excluded_scale_rows = (
        int(df["spoken"].notna().sum()) - len(nested)
    )
    return {"u": design_u, "s": design_s}


@dataclass(frozen=True)
class Pool:
    """One (model, outcome) pair, with the anchors the calibration reports at.

    ``study_effects`` and ``subject_effects`` record what the *model* carries,
    and the estimator mirrors them. That is the whole lesson of the VG11 failure:
    a dispersion prior is a prior about the residual left after the model's
    grouping structure, so a calibration that includes more effects than the
    model does will understate `kappa`, and one that includes fewer will
    overstate it. VG01-VG04 carry neither, so for them the fit here reduces to
    the marginal per-age estimate and the two columns coincide.
    """

    label: str
    model_id: str
    part: str | None
    anchors: tuple
    study_effects: bool = True
    subject_effects: bool = True
    note: str = ""

    def design(self, **basis):
        definition = getattr(defs, self.model_id)
        design = (univariate_frame(definition, **basis) if self.part is None
                  else bivariate_frames(definition, **basis)[self.part])
        if not self.study_effects:
            # one group => the sum-to-zero contrast is empty and no between-study
            # spread is removed, so kappa keeps it, as the model's must
            design.studies = np.zeros(1)
            design.study_idx = np.zeros(design.n_obs, dtype=int)
        return design


POOLS = {
    # No grouping structure: kappa carries every source of spread, so the fit
    # here is the marginal per-age estimate and there is no contrast to draw.
    "vg02-understood": Pool(
        "VG02 understood (DS)", "VG02", None, (18.0, 36.0),
        study_effects=False, subject_effects=False,
    ),
    "vg04-understood": Pool(
        "VG04 understood (TD, 25% subsample)", "VG04", None, (12.0, 18.0),
        study_effects=False, subject_effects=False,
    ),
    # Study and subject random intercepts: kappa is what is left after them.
    "vg11-spoken": Pool("VG11 spoken (TD)", "VG11", None, (12.0, 20.0)),
    "vg12-understood": Pool("VG12 understood (TD)", "VG12", None, (12.0, 20.0)),
    "vg13-understood": Pool("VG13 understood (TD 8-18)", "VG13", "u", (12.0, 17.0)),
    "vg13-q": Pool("VG13 q = spoken|understood (TD 8-18)", "VG13", "s", (12.0, 17.0)),
    "vg09-understood": Pool(
        "VG09/VG10/VG16 understood (DS)", "VG09", "u", (24.0, 48.0),
        note="frame too thin for a stable estimate -- see --mean-sweep",
    ),
    "vg09-q": Pool(
        "VG09/VG10/VG16 q (DS)", "VG09", "s", (24.0, 48.0),
        note="frame too thin for a stable estimate -- see --mean-sweep",
    ),
}


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def run_pool(key, *, nodes=DEFAULT_NODES, basis=None):
    pool = POOLS[key]
    design = pool.design(**(basis or {}))
    print(f"\n{'=' * 78}\n{pool.label}   anchors={pool.anchors}")
    print(" ", design.describe(), flush=True)
    if design.n_excluded_scale_rows:
        print(f"   rows on the marginal (out-of-810) spoken scale, excluded: "
              f"{design.n_excluded_scale_rows:,}")
    if pool.note:
        print(f"   NOTE: {pool.note}")

    marginal = fit(design, pool.anchors, n_nodes=nodes, tau_fixed=1e-6,
                   standard_errors=not pool.subject_effects)
    print(marginal.summary(label="marginal (tau=0): "), flush=True)

    if not pool.subject_effects:
        # The model has no subject effect, so the marginal fit *is* the matching
        # specification. Fitting a conditional one too would calibrate against a
        # residual this model never forms.
        print("    (model carries no random effects — this is the fit to use)")
        return marginal, marginal

    conditional = fit(design, pool.anchors, n_nodes=nodes, standard_errors=True)
    print(conditional.summary(label="conditional:      "), flush=True)

    print(f"    LR against tau=0: {2 * (marginal.nll - conditional.nll):,.1f} on 1 df")
    print(f"    conditional / marginal: "
          f"kappa({pool.anchors[0]:.0f}) "
          f"{conditional.kappa_young / marginal.kappa_young:.2f}x   "
          f"kappa({pool.anchors[1]:.0f}) "
          f"{conditional.kappa_old / marginal.kappa_old:.2f}x", flush=True)
    return conditional, marginal


def run_recovery(key, *, nodes=DEFAULT_NODES, basis=None):
    """Simulate from two opposite truths on the real design and refit both.

    Run this with the same ``basis`` the pool would be calibrated under —
    a design that recovers a known truth with one mean model says nothing about
    another.
    """
    pool = POOLS[key]
    base = pool.design(**(basis or {}))
    print(f"\n{'=' * 78}\nrecovery on {pool.label}\n  {base.describe()}", flush=True)

    truths = [
        ("subject-heavy", dict(tau=1.05, kappa_min=6.0,
                               excess_young=311.0, excess_old=44.0)),
        ("dispersion-heavy", dict(tau=0.15, kappa_min=3.0,
                                  excess_young=27.0, excess_old=3.6)),
    ]
    for name, truth in truths:
        print(f"\n  truth ({name}): tau={truth['tau']}  "
              f"kappa({pool.anchors[0]:.0f})="
              f"{truth['kappa_min'] + truth['excess_young']:.1f}  "
              f"kappa({pool.anchors[1]:.0f})="
              f"{truth['kappa_min'] + truth['excess_old']:.1f}", flush=True)
        for seed in (11, 12):
            y = simulate(base, anchor_ages=pool.anchors, seed=seed, **truth)
            sim = Design(
                age=base.age, y=y, n_trials=base.n_trials,
                subject=base.subject_idx, study=base.study_idx, min_cell=1,
            )
            print(fit(sim, pool.anchors, n_nodes=nodes).summary(
                label=f"seed {seed}: "), flush=True)


def run_mean_sweep(key, *, nodes=DEFAULT_NODES):
    """How much does the answer depend on how flexible the mean is?

    A pool whose kappa moves with the mean model cannot be calibrated from.
    """
    pool = POOLS[key]
    ya, oa = int(pool.anchors[0]), int(pool.anchors[1])
    print(f"\n{'=' * 78}\nmean sweep -- {pool.label}   anchors={pool.anchors}")
    print(f"  {'mean':>16} {'tau':>7} {'kappa_min':>10} "
          f"{f'kappa({ya})':>11} {f'kappa({oa})':>11}", flush=True)
    # Sweep the specification the pool actually calibrates under, not always the
    # conditional one — otherwise a no-random-effects model is tested for the
    # stability of a fit it never uses.
    tau_fixed = None if pool.subject_effects else 1e-6

    def _row(design):
        res = fit(design, pool.anchors, n_nodes=nodes, tau_fixed=tau_fixed)
        name = f"{design.mean}[{design.B.shape[1]}]"
        print(f"  {name:>16} {res.tau:>7.3f} {res.kappa_min:>10.2f} "
              f"{res.kappa_young:>11.1f} {res.kappa_old:>11.1f}", flush=True)

    for knots in (4, 6, 8, 12):
        _row(pool.design(mean="spline", min_cell=1, n_knots=knots))
    _row(pool.design())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pools", nargs="*", default=None,
                        help=f"any of: {', '.join(POOLS)} (default: all)")
    parser.add_argument("--nodes", type=int, default=DEFAULT_NODES,
                        help="Gauss-Hermite nodes (default %(default)s)")
    parser.add_argument("--recover", action="store_true",
                        help="simulate from a known truth and refit, instead of fitting")
    parser.add_argument("--mean-sweep", action="store_true",
                        help="refit across mean models to test whether kappa is stable")
    parser.add_argument("--mean", choices=("saturated", "spline"), default="saturated",
                        help="mean model for --recover and the default fit "
                             "(default %(default)s; the DS joint frame needs spline, "
                             "since the cell rule drops half its rows)")
    parser.add_argument("--knots", type=int, default=8,
                        help="spline knots when --mean spline (default %(default)s)")
    args = parser.parse_args(argv)

    keys = args.pools or list(POOLS)
    unknown = [k for k in keys if k not in POOLS]
    if unknown:
        parser.error(f"unknown pool(s): {', '.join(unknown)}")

    basis = ({"mean": "spline", "min_cell": 1, "n_knots": args.knots}
             if args.mean == "spline" else {})

    for key in keys:
        if args.recover:
            run_recovery(key, nodes=args.nodes, basis=basis)
        elif args.mean_sweep:
            run_mean_sweep(key, nodes=args.nodes)
        else:
            run_pool(key, nodes=args.nodes, basis=basis)
    return 0


if __name__ == "__main__":
    sys.exit(main())
