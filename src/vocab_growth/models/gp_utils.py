# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""PyMC-graph build helpers shared across the model engines.

This module holds helpers that emit PyMC ops (as opposed to the pure-NumPy
helpers in :mod:`vocab_growth.models.build_utils`). It is deliberately kept
separate so the pure module stays ``pymc``-free.

It provides :func:`make_kappa_of_z`, the age-varying dispersion
closure factory. The factory takes random variables the caller has already
created (in the caller's own order) and returns a closure whose body is
byte-identical to the ``kappa_of_z`` closures previously inlined in every engine,
so it moves no random-variable creation and cannot change the model graph.

Two builders wrap it, differing only in how the same curve is parameterised:
:func:`build_kappa_of_z` (a free intercept and a sign-constrained slope) and
:func:`build_kappa_of_z_anchored` (priors on the age term at two reference ages,
from which the intercept and slope are derived). Models migrate one at a time;
see :class:`~vocab_growth.models.definitions.KappaAnchorPriorParams`.

It also provides the mean + HSGP constructions shared by every engine. They differ
only in the mean term and share the HSGP tail through :func:`_gp_from_mean`:
:func:`trend_and_gp` (a logit-linear age trend, anchored at two reference ages —
every understood and production-ratio trajectory in the family) and
:func:`tent_and_gp` (three anchors interpolated as a tent meeting at a peak, used
for the signed ratio by VG14 and VG15, where a free age slope would extrapolate
the ratio below the data floor; it replaced an intercept-only builder in #154).
Because these carry the sole RNG-bearing call (``hsgp.prior()``), they are
parameterised (``suffix``, ``store_deterministic``, ``latent_name``,
``anchor_idx``, ``grid``) so each engine reproduces its previous PyMC graph
byte-for-byte (same free RVs, in the same order, with the same names and
``logp``); the named-``Deterministic`` differences between engines change only
what is stored in the trace, not the sampled distribution.

Child-effect and variance builders live here too, and are tabulated by structure
in :mod:`vocab_growth.models.subject_effects`, which resolves *which* structure a
definition selects but holds none of the builders: :func:`build_child_factor`
(VG22's low-rank factor), :func:`build_child_slope` (VG19's per-child slope),
:func:`build_subject_scale_of_z` (Proposal A1's age-varying scale) and
:func:`build_variance_partition` (the shared subject/dispersion budget).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from dse_research_utils.statistics.models.pymc_utils import logit
from pytensor.tensor.linalg import solve as pt_solve

from vocab_growth.models.build_utils import CLAMP_SOFTNESS


def make_kappa_of_z(kappa_min, a_kappa, b_kappa):
    """Return the age-varying dispersion closure ``z -> kappa_min + exp(a + b z)``.

    ``kappa_min``, ``a_kappa`` and ``b_kappa`` are PyMC variables the caller has
    already created (including any ``b_kappa = -b_kappa_mag`` deterministic). The
    returned closure is evaluated later at the standardised ages, emitting the
    same ops the inlined closures did.
    """

    def kappa_of_z(z):
        return kappa_min + pm.math.exp(a_kappa + b_kappa * z)

    return kappa_of_z


def build_kappa_of_z(kappa_min_dist, a_kappa_dist, b_kappa_mag_dist, suffix=""):
    """Create the kappa RVs and return the age-varying dispersion closure.

    Mechanical extraction of the identical four-line block every engine
    repeats once per outcome: ``kappa_min{suffix}``, ``a_kappa{suffix}`` and
    ``b_kappa_mag{suffix}`` are created via ``to_pymc`` (in that order), then
    ``b_kappa{suffix} = -b_kappa_mag{suffix}`` is stored as a named
    ``Deterministic``, and the two feed :func:`make_kappa_of_z`. Because the
    op order and names are unchanged from the inlined form, this moves no
    random-variable creation and cannot change the model graph (same
    contract as :func:`make_kappa_of_z`, which it wraps).
    """
    kappa_min = kappa_min_dist.to_pymc(f"kappa_min{suffix}")
    a_kappa = a_kappa_dist.to_pymc(f"a_kappa{suffix}")
    b_kappa_mag = b_kappa_mag_dist.to_pymc(f"b_kappa_mag{suffix}")
    b_kappa = pm.Deterministic(f"b_kappa{suffix}", -b_kappa_mag)
    return make_kappa_of_z(kappa_min, a_kappa, b_kappa)


def build_subject_scale_of_z(spec, *, anchor_z, name):
    """Create the A1 age-varying subject-effect scale and return its closure.

    ``spec`` is an
    :class:`~vocab_growth.models.definitions.AgeVaryingSubjectScale`; ``anchor_z``
    is its two reference ages on the standardised scale; ``name`` is the scalar
    parameter this replaces (``"tau_subj_u"``, ``"tau_subj_q"``,
    ``"tau_subject"``), which fixes every emitted variable name.

    The graph is::

        {name}_young  ~ HalfNormal(spec.young_sigma)
        log_{name}_ratio ~ Normal(0, spec.log_ratio_sigma)
        tau(z) = {name}_young * exp(log_ratio * (z - z_young) / (z_old - z_young))

    so ``log_{name}_ratio = 0`` is the constant-scale model of record and its
    posterior interval *is* the answer to "does the between-child spread widen".
    The ratio is multiplicative rather than log-linear between two independent
    anchors: no logarithm is taken of a ``HalfNormal`` that can approach zero,
    and the young anchor keeps the record's own prior unmodified.

    ``{name}`` itself is emitted as a scalar ``Deterministic`` equal to the scale
    **at the young anchor**, so every consumer that reads the constant-``tau``
    name — the posterior summaries, the heterogeneity comparators, the recovery
    scorer — keeps working and reads a quantity with a stated age attached.
    ``{name}_old`` is emitted for symmetry with the kappa anchors.

    Returns ``(tau_of_z, tau_young)`` — the closure, and the young-anchor scalar
    itself so the caller can reuse it without going back through the model's
    variable table (which is not populated until the build returns).
    """
    z_young, z_old = (float(anchor_z[0]), float(anchor_z[1]))
    if not z_old > z_young:
        raise ValueError(
            f"subject-scale anchor_z must be ordered (young, old); got {anchor_z!r}."
        )
    span = z_old - z_young
    tau_young = pm.HalfNormal(f"{name}_young", sigma=spec.young_sigma)
    log_ratio = pm.Normal(
        f"log_{name}_ratio", mu=0.0, sigma=spec.log_ratio_sigma
    )
    _ = pm.Deterministic(name, tau_young)
    _ = pm.Deterministic(f"{name}_old", tau_young * pm.math.exp(log_ratio))

    def tau_of_z(z):
        return tau_young * pm.math.exp(log_ratio * (z - z_young) / span)

    return tau_of_z, tau_young


#: Anchor order for VG22's child-factor gauge, as indices into the effect order
#: ``(b0u, b1u, b0q, b1q)``: the two levels first, then the production-ratio
#: rate, with the comprehension rate last so it carries a diagonal at no
#: registered rank. See :func:`build_child_factor` for why.
CHILD_FACTOR_ANCHOR_ORDER = (0, 2, 3, 1)


def build_child_factor(
    spec,
    *,
    tau0_u_sigma,
    tau0_q_sigma,
    age_obs_months,
    subject_obs,
):
    """Create VG22's low-rank factor over the four child effects.

    ``spec`` is a
    :class:`~vocab_growth.models.definitions.SubjectFactorPriorParams`;
    ``tau0_u_sigma`` / ``tau0_q_sigma`` are the definition's own scalar
    ``tau_subj_*_sigma`` priors, re-used for the two LEVEL scales so the parent's
    priors are inherited rather than restated; ``age_obs_months`` is
    unstandardised age in months, because the rate is per year of real age.

    The four effects, in this order throughout, are
    ``(b0u, b1u, b0q, b1q)`` -- comprehension level and rate, then production
    ratio level and rate. The graph::

        tau        = [tau_subj_u_0, tau_subj_u_1, tau_subj_q_0, tau_subj_q_1]
        W          triangular with a positive diagonal when its rows are taken
                   in the anchor order (b0u, b0q, b1q, b1u) -- see below
        L[i, :]    = tau[i] * W[i, :] / ||W[i, :]||          # unit directions
        z          ~ Normal(0, 1), dims (subject_id, factor)
        b          = z @ L.T                                  # (subject, 4)
        shift_u(obs) = b[subject, 0] + b[subject, 1] * (age - ref) / 12
        shift_q(obs) = b[subject, 2] + b[subject, 3] * (age - ref) / 12

    so ``Sigma = L L'`` is positive semi-definite by construction with no
    constraint to enforce, and ``Sigma_ii = tau[i] ** 2`` exactly -- which is
    what keeps ``tau_subj_u_0`` meaning the same between-child spread it means in
    VG10, VG19 and VG20 rather than something rank-dependent.

    **Why the triangular constraint, and why the anchor order.** ``L`` and
    ``L Q`` give the same covariance for any orthogonal ``Q``, so without it the
    loadings sit on a rotational ridge -- a sampling problem, not merely an
    interpretive one. Taking ``k`` anchor rows and making their ``k x k`` block
    lower-triangular with a positive diagonal removes exactly the
    ``k (k - 1) / 2`` rotational degrees of freedom and the reflections, leaving
    ``4 + (k - 1) * (k / 2 + 4 - k)`` free covariance parameters: 4, 7 and 9 at
    ranks 1, 2 and 3, reproducing the rank table in
    ``notes/202608221000-four-by-four-gate1.md`` §4. Which rows anchor is a
    gauge choice -- it changes neither ``Sigma`` nor the counts -- but it is not
    a free one: a diagonal only pins its column's sign if the row it sits on has
    real between-child variance, because a row whose ``tau`` is ~0 contributes
    ~0 to ``L`` whatever its direction, and its constraint then pins nothing.
    The anchors are therefore :data:`CHILD_FACTOR_ANCHOR_ORDER`,
    ``(b0u, b0q, b1q, b1u)``: the two levels, then the production-ratio rate,
    with the comprehension rate last so it carries a diagonal at no registered
    rank. ``b1u`` is the one effect every fit of this family puts at ~0 (Gate 1:
    0.079; the dev and first ``rep`` fits: 0.04), and anchoring the second
    factor on it is exactly what split the 2026-08-23 ``rep`` fit into mirror
    modes -- see ``notes/202608231420-vg22-factor-anchor-bimodality.md``. At
    ``k = 1`` the anchor order changes nothing: every row has one entry and only
    ``b0u``'s is positive.

    **Name preservation.** ``tau_subj_u`` and ``tau_subj_q`` are emitted as
    scalar deterministics equal to the two level scales, and
    ``delta_subj_u`` / ``delta_subj_q`` as the per-child offsets at the reference
    age, so every consumer written against the constant-offset models keeps
    working and reads a quantity with a stated age attached -- the same contract
    :func:`build_child_slope` and :func:`build_subject_scale_of_z` keep.
    ``rho_uq`` is emitted as the implied level-level correlation so that VG20's
    comparator reads the quantity VG20 estimates, and the full 4x4 correlation is
    emitted as ``subject_factor_corr`` because with a factor form the individual
    correlations are derived rather than sampled.

    Returns ``(shift_u_obs, shift_q_obs, tau0_u, tau0_q)``.
    """
    k = int(spec.rank)
    rho_uq_eta = float(getattr(spec, "rho_uq_eta", 2.0))
    if not rho_uq_eta > 0:
        raise ValueError(f"rho_uq_eta must be positive; got {rho_uq_eta!r}.")
    if not tau0_u_sigma > 0 or not tau0_q_sigma > 0:
        raise ValueError(
            "child-factor level scales must be positive; got "
            f"tau0_u_sigma={tau0_u_sigma!r}, tau0_q_sigma={tau0_q_sigma!r}."
        )

    tau0_u = pm.HalfNormal("tau_subj_u_0", sigma=tau0_u_sigma)
    tau1_u = pm.HalfNormal("tau_subj_u_1", sigma=spec.tau1_u_sigma)
    tau0_q = pm.HalfNormal("tau_subj_q_0", sigma=tau0_q_sigma)
    tau1_q = pm.HalfNormal("tau_subj_q_1", sigma=spec.tau1_q_sigma)
    tau = pm.math.stack([tau0_u, tau1_u, tau0_q, tau1_q])

    # Rows of the loading matrix as UNIT directions, emitted in effect order but
    # constrained in anchor order: the row at anchor position p spans the first
    # min(p + 1, k) columns and, for p < k, has a positive entry at column p, so
    # the k anchor rows form a lower-triangular block with a positive diagonal
    # and the rotation (and the sign of each factor) is pinned on rows that have
    # variance to pin it with. Rows are built individually rather than as a
    # masked matrix because the mask would put structural zeros in the trace and
    # make the free-parameter count unreadable.
    #
    # Sigma depends on the rows only through their DIRECTIONS -- each row's
    # radial magnitude cancels in the normalisation -- so a row sampled as m
    # entries and then normalised spends m parameters on m - 1 identified
    # quantities. Issue #266 finding 5 asked for those prior-only magnitudes to
    # be removed where possible. Two of the four can be, and are; the other two
    # cannot without a chart on the sphere, whose azimuth wraps at 0 = 2*pi. That
    # was measured rather than assumed: on a direction with a real posterior the
    # (z, phi) chart lost up to 17x the effective sample size against normalised
    # normals and reached R-hat 1.053, which this project's convergence gate
    # fails. An inert parameter costs a row in the gate; a wrapped coordinate
    # costs the fit.
    width_of = {
        effect: min(position + 1, k)
        for position, effect in enumerate(CHILD_FACTOR_ANCHOR_ORDER)
    }
    diagonal_of = {
        effect: position
        for position, effect in enumerate(CHILD_FACTOR_ANCHOR_ORDER)
        if position < k
    }
    first_anchor = CHILD_FACTOR_ANCHOR_ORDER[0]
    second_anchor = CHILD_FACTOR_ANCHOR_ORDER[1]

    def _pad(entries):
        if len(entries) < k:
            entries = entries + [pt.constant(0.0)] * (k - len(entries))
        return pm.math.stack(entries)

    rows = []
    for i in range(4):
        width = width_of[i]
        diagonal = diagonal_of.get(i)

        if i == first_anchor and width == 1 and diagonal == 0:
            # Its direction is e_0 for ANY positive entry, so the entry carried
            # no information at all -- not even a sign. Sampling it also left the
            # documented `Sigma_ii = tau_i ** 2` false in the tail: a one-entry
            # row's norm IS its magnitude, so a near-zero HalfNormal draw met the
            # numerical floor and shrank the row below unit length. Measured at
            # 55 draws in two million more than 0.1% short, worst case 37%.
            rows.append(_pad([pt.constant(1.0)]))
            continue

        if i == second_anchor and width == 2 and diagonal == 1:
            # `rho_uq` IS this row's first coordinate, because the first anchor
            # row is exactly e_0 -- so a prior placed here is a prior on the
            # correlation, with no approximation. Written as VG20 writes it, so
            # `rho_uq_raw` means the same thing in both models.
            rho_raw = pm.Beta("rho_uq_raw", alpha=rho_uq_eta, beta=rho_uq_eta)
            rho = pm.Deterministic("rho_uq", 2.0 * rho_raw - 1.0)
            rows.append(
                _pad([rho, pm.math.sqrt(pm.math.maximum(1.0 - rho**2, 1e-12))])
            )
            continue

        # Everything else keeps the normalise-a-Normal construction: it has no
        # boundary and no wrap, at the cost of one inert magnitude per row.
        entries = []
        for j in range(width):
            if diagonal == j:
                entries.append(pm.HalfNormal(f"subject_factor_w_{i}{j}", sigma=1.0))
            else:
                entries.append(pm.Normal(f"subject_factor_w_{i}{j}", mu=0.0, sigma=1.0))
        raw = pm.math.stack(entries)
        norm = pm.math.sqrt(pm.math.sum(raw**2) + 1e-12)
        rows.append(_pad([entry / norm for entry in entries]))

    U = pm.math.stack(rows)  # (4, k), unit rows

    L = pm.Deterministic(
        "subject_factor_loadings",
        tau[:, None] * U,
        dims=("child_effect4", "factor"),
    )

    sigma_mat = pm.math.dot(L, L.T)
    sd = pm.math.sqrt(pm.math.diag(sigma_mat))
    corr = pm.Deterministic(
        "subject_factor_corr",
        sigma_mat / (sd[:, None] * sd[None, :]),
        dims=("child_effect4", "child_effect4_b"),
    )
    # The element VG20 estimates, so its comparator and the recovery scorer read
    # the same named quantity here as there. Emitted by the second-anchor branch
    # above wherever that branch exists -- where it does, `corr[0, 2]` equals the
    # sampled value exactly, since the first anchor row is e_0. At rank 1 there
    # is no such branch: every effect is one deviate scaled four ways, so
    # `rho_uq` is +/-1 by construction and is read off the matrix.
    if "rho_uq" not in pm.modelcontext(None).named_vars:
        _ = pm.Deterministic("rho_uq", corr[0, 2])

    z = pm.Normal(
        "subject_factor_z", mu=0.0, sigma=1.0, dims=("subject_id", "factor")
    )
    b = pm.math.dot(z, L.T)  # (subject, 4)

    b0_u = pm.Deterministic("b0_tau_subj_u", b[:, 0], dims="subject_id")
    b1_u = pm.Deterministic("b1_tau_subj_u", b[:, 1], dims="subject_id")
    b0_q = pm.Deterministic("b0_tau_subj_q", b[:, 2], dims="subject_id")
    b1_q = pm.Deterministic("b1_tau_subj_q", b[:, 3], dims="subject_id")

    # Constant-offset names, kept for every downstream reader.
    _ = pm.Deterministic("tau_subj_u", tau0_u)
    _ = pm.Deterministic("tau_subj_q", tau0_q)
    _ = pm.Deterministic("delta_subj_u", b0_u, dims="subject_id")
    _ = pm.Deterministic("delta_subj_q", b0_q, dims="subject_id")

    d_obs = (pt.as_tensor_variable(age_obs_months) - spec.ref_age_months) / 12.0
    shift_u = b0_u[subject_obs] + b1_u[subject_obs] * d_obs
    shift_q = b0_q[subject_obs] + b1_q[subject_obs] * d_obs
    return shift_u, shift_q, tau0_u, tau0_q


def build_child_slope(spec, *, age_obs_months, subject_obs, ref_age_months, name):
    """Create the VG19 child intercept-and-slope block and return its closure.

    ``spec`` is a
    :class:`~vocab_growth.models.definitions.SubjectSlopePriorParams`;
    ``age_obs_months`` is the observation ages in **months** (unstandardised —
    the slope is per year of real age, not per standard deviation, so that
    ``tau1`` stays readable and comparable across pools); ``subject_obs`` indexes
    each observation's child; ``ref_age_months`` is the age at which ``tau0`` is
    the between-child spread; ``name`` is the scalar parameter this replaces
    (``"tau_subj_u"`` / ``"tau_subj_q"``), which fixes every emitted name.

    The graph, non-centred, with the 2x2 Cholesky written out::

        {name}_0    ~ HalfNormal(spec.tau0_sigma)
        {name}_1    ~ HalfNormal(spec.tau1_sigma)       # per year
        {name}_rho_raw ~ Beta(eta, eta)
        rho01       = 2 * {name}_rho_raw - 1
        z           ~ Normal(0, 1), dims (subject_id, "child_effect")
        b0 = tau0 * z[:, 0]
        b1 = tau1 * (rho01 * z[:, 0] + sqrt(1 - rho01**2) * z[:, 1])
        shift(obs) = b0[subject] + b1[subject] * (age - ref) / 12

    which is ``z @ L.T`` for ``L = [[tau0, 0], [rho01*tau1, tau1*sqrt(1-rho01^2)]]``
    written elementwise, so the two columns keep their names in the trace.

    **Name preservation.** ``{name}`` is emitted as a scalar ``Deterministic``
    equal to ``tau0`` — the spread at the reference age — so every consumer that
    reads the constant-tau name keeps working and reads a quantity with a stated
    age attached, exactly as :func:`build_subject_scale_of_z` does.
    ``delta_{name-without-tau_}`` keeps its per-child meaning as the offset at
    the reference age. ``{name}_rho`` is emitted so the correlation is a named
    variable rather than an element of a packed vector.

    Returns ``(shift_obs, tau0)`` — the per-observation shift and the reference-age
    scale, the latter so the caller can reuse it without going back through the
    model's variable table.
    """
    if not spec.tau0_sigma > 0 or not spec.tau1_sigma > 0:
        raise ValueError(
            f"child-slope scales must be positive; got tau0_sigma="
            f"{spec.tau0_sigma!r}, tau1_sigma={spec.tau1_sigma!r}."
        )
    if not spec.rho_eta > 0:
        raise ValueError(f"child-slope rho_eta must be positive; got {spec.rho_eta!r}.")

    tau0 = pm.HalfNormal(f"{name}_0", sigma=spec.tau0_sigma)
    tau1 = pm.HalfNormal(f"{name}_1", sigma=spec.tau1_sigma)
    rho_raw = pm.Beta(f"{name}_rho_raw", alpha=spec.rho_eta, beta=spec.rho_eta)
    rho01 = pm.Deterministic(f"{name}_rho", 2.0 * rho_raw - 1.0)

    z = pm.Normal(f"{name}_z", mu=0.0, sigma=1.0, dims=("subject_id", "child_effect"))
    b0 = pm.Deterministic(f"b0_{name}", tau0 * z[:, 0], dims="subject_id")
    b1 = pm.Deterministic(
        f"b1_{name}",
        tau1 * (rho01 * z[:, 0] + pm.math.sqrt(1.0 - rho01**2) * z[:, 1]),
        dims="subject_id",
    )

    # The record's own name, and its per-child companion, both read at ref_age.
    _ = pm.Deterministic(name, tau0)
    _ = pm.Deterministic(f"delta_{name.replace('tau_', '')}", b0, dims="subject_id")

    years = (age_obs_months - float(ref_age_months)) / 12.0
    shift_obs = b0[subject_obs] + b1[subject_obs] * years
    return shift_obs, tau0


def build_kappa_of_z_anchored(
    kappa_min_dist,
    excess_young_dist,
    excess_old_dist,
    *,
    anchor_z,
    suffix="",
    excess_young_value=None,
    hold_constant=False,
):
    """Create the two-anchor kappa RVs and return the dispersion closure.

    The two-anchor counterpart of :func:`build_kappa_of_z`. Instead of free
    ``a_kappa`` / ``b_kappa_mag``, the *age term* ``exp(a_kappa + b_kappa z)`` is
    given priors at two standardised reference ages ``anchor_z = (z_young,
    z_old)``, and the intercept and slope are solved for:

        b_kappa = (log e_old - log e_young) / (z_old - z_young)
        a_kappa = log e_young - b_kappa * z_young

    so that ``kappa(z_young) = kappa_min + e_young`` and ``kappa(z_old) =
    kappa_min + e_old`` exactly. See
    :class:`~vocab_growth.models.definitions.KappaAnchorPriorParams` for why the
    reparameterisation is worth making.

    **Only the anchor totals are estimands.** The split of each total into
    ``kappa_min`` plus an excess is not identified: parameter recovery scores
    ``kappa_min`` at -60.1%, -55.7%, -53.9% (VG10) and ``kappa_excess_old_s`` at
    +263%, +155%, +112% (VG20), while ``kappa_old_u`` and ``kappa_old_s`` --
    the sums containing them -- come back within a few percent on the same
    fits. The floor's *share* of reported kappa carries an 89% interval of
    [13.2%, 78.3%] on comprehension at 84 months and [21.0%, 99.8%] on the
    nested spoken scale. So ``kappa_min``, ``kappa_excess_young`` and
    ``kappa_excess_old`` are sampling coordinates; report and interpret
    ``kappa`` at an age, never a component, and never a trend in one.

    The totals themselves are data-driven. Re-centring the ``kappa_min`` prior
    from a median of 3.0 to the conditionally calibrated 7.8 -- a 160% move --
    shifts reported kappa by 14.2% at 84 months and 6.6% at 72 on
    comprehension, an elasticity of 0.09 and 0.04, and the 89% intervals
    overlap throughout. See ``notes/202608191800-kappa-components-not-estimands.md``.

    ``a_kappa{suffix}`` and ``b_kappa{suffix}`` are still stored as named
    ``Deterministic``\\ s, under the same names the legacy form gives them, so a
    migrated model's dispersion posterior stays directly comparable with the
    fits that preceded it — ``a_kappa`` is a derived quantity here and a free RV
    there, but it is the same quantity in both. ``kappa_young{suffix}`` and
    ``kappa_old{suffix}`` carry *total* kappa at the two anchors, which is what a
    per-age empirical estimate can be checked against.

    The three free RVs are the asymptote and the two anchors. No prior is placed
    on the slope at all: its sign is whatever the two anchors imply, so a rising
    dispersion trajectory is representable (the legacy form's ``b_kappa =
    -b_kappa_mag <= 0`` is not). When it does rise, ``kappa_min`` is the
    *young*-age asymptote rather than an old-age floor — the exponential term
    vanishes at whichever end the slope points away from.

    ``hold_constant`` pins ``b_kappa`` to exactly zero, so dispersion is flat in
    age with its *level* still free (``kappa_min + excess_young``). It exists for
    Proposal A1, which *moves* the age variation onto the between-child scale
    rather than adding it there, and it drops ``excess_old_dist`` — there is no
    old anchor left to place a prior on. Every name the anchored form emits is
    still emitted, with ``kappa_old`` equal to ``kappa_young`` by construction,
    so a flat-kappa fit stays directly comparable with the fits around it.
    """
    z_young, z_old = (float(anchor_z[0]), float(anchor_z[1]))
    if not z_old > z_young:
        raise ValueError(
            f"kappa anchor_z must be ordered (young, old); got {anchor_z!r}."
        )
    if hold_constant:
        kappa_min = kappa_min_dist.to_pymc(f"kappa_min{suffix}")
        if excess_young_value is None:
            excess_young = excess_young_dist.to_pymc(f"kappa_excess_young{suffix}")
        else:
            excess_young = pm.Deterministic(
                f"kappa_excess_young{suffix}", excess_young_value
            )
        _ = pm.Deterministic(f"kappa_excess_old{suffix}", excess_young)
        b_kappa = pm.Deterministic(f"b_kappa{suffix}", pt.zeros(()))
        a_kappa = pm.Deterministic(f"a_kappa{suffix}", pm.math.log(excess_young))
        _ = pm.Deterministic(f"kappa_young{suffix}", kappa_min + excess_young)
        _ = pm.Deterministic(f"kappa_old{suffix}", kappa_min + excess_young)
        return make_kappa_of_z(kappa_min, a_kappa, b_kappa)
    kappa_min = kappa_min_dist.to_pymc(f"kappa_min{suffix}")
    if excess_young_value is None:
        excess_young = excess_young_dist.to_pymc(f"kappa_excess_young{suffix}")
    else:
        # The young anchor is being supplied by the variance-partition
        # reparameterisation (see `build_variance_partition`), which allocates it
        # and the subject-effect scale from one shared budget. It keeps its usual
        # name as a Deterministic so every downstream consumer -- the comparators,
        # the posterior summaries, the recovery harness -- still finds it, and
        # `excess_young_dist` goes unused because the prior now sits on the budget
        # and the split rather than on this quantity directly.
        excess_young = pm.Deterministic(
            f"kappa_excess_young{suffix}", excess_young_value
        )
    excess_old = excess_old_dist.to_pymc(f"kappa_excess_old{suffix}")
    log_young = pm.math.log(excess_young)
    log_old = pm.math.log(excess_old)
    b_kappa = pm.Deterministic(
        f"b_kappa{suffix}", (log_old - log_young) / (z_old - z_young)
    )
    a_kappa = pm.Deterministic(f"a_kappa{suffix}", log_young - b_kappa * z_young)
    _ = pm.Deterministic(f"kappa_young{suffix}", kappa_min + excess_young)
    _ = pm.Deterministic(f"kappa_old{suffix}", kappa_min + excess_old)
    return make_kappa_of_z(kappa_min, a_kappa, b_kappa)


def build_variance_partition(
    total_dist,
    share_dist,
    *,
    reference_proportion,
    subject_scale_name,
    suffix="",
):
    """Split one scatter budget between subject effects and dispersion.

    The subject random-effect scale and the Beta-Binomial dispersion both describe
    how far observations at a given age fall from the population trajectory —
    ``tau_subject`` attributing that scatter to persistent between-child
    differences, ``kappa`` to within-child noise. Sampled as two free scales they
    compete for the same variance, and in the typically-developing hierarchical
    models the resulting ridge is the dominant sampling pathology: VG12 records
    ``corr(tau_subject, kappa_young) = +0.755`` with both parameters at the top of
    the marginal-energy correlations (-0.812 and -0.783), which is what its energy
    BFMI failure is made of. Only children measured more than once carry the
    within-child replication that identifies the split, and the TD pool averages
    1.21 observations per child.

    This reparameterises the pair into the quantity the data *do* identify and the
    one they do not:

        v_total  — total logit-scale scatter at the young kappa anchor
        share    — the fraction of it attributable to persistent child differences

        tau_subject         = sqrt(share * v_total)
        kappa_excess_young  = c / ((1 - share) * v_total)

    where ``c = 1 / (p0 * (1 - p0))`` converts a Beta-Binomial concentration into
    an approximate logit-scale variance by the delta method, at a **fixed**
    reference proportion ``p0``. Fixing ``p0`` rather than reading it off the
    fitted trajectory is deliberate: it keeps this a pure change of coordinates on
    the two scale parameters, with no dependence on the mean function, so the
    priors below mean the same thing regardless of what the trend does.

    The young *excess* is allocated rather than total ``kappa`` so that positivity
    is automatic — ``kappa_min`` remains a free asymptote and ``kappa_young =
    kappa_min + excess_young`` is positive by construction.

    Both original parameters are returned to the graph under their usual names, so
    this changes what the sampler explores and not which quantities the model
    reports. The prior does move, necessarily and by design — it now sits on the
    budget and the split, which is where a prior on this pair can actually be
    reasoned about.

    **It does not follow that the reported values are unaffected, and this
    docstring used to claim they were.** Parameter recovery on VG12 returns
    ``tau_subject`` below its truth in three replicates of three, by about 5.8%,
    with the truth outside the 89% interval every time. ``v_total`` recovers
    cleanly (z = +1.05, +0.63, −0.10) and ``subject_variance_share`` is biased low
    (−20.6%, −12.7%, −8.7%), so on VG12 it is the split that is mis-estimated
    rather than the budget, and ``tau_subject`` inherits the bias amplified
    because the square root concentrates its posterior.

    **This docstring used to add that VG10, carrying two free scales and no
    partition, showed no such consistent direction. That is false.** VG10 returns
    −5.28%, −8.23%, −3.20% and VG20 −7.08%, −4.42%, −5.95%, against VG12's
    −6.59%, −5.32%, −5.66%: the same size and the same sign in 9 replicates of
    9, with the partition and without it. So the partition is not the cause — but
    the two families fail differently. VG12 keeps its budget and mis-splits it,
    whereas in VG10 and VG20 the dispersion concentration at the young anchor
    comes back low alongside the subject scale (9 of 9), so there the budget
    itself is under-recovered rather than merely misallocated.

    Since ``tau_subject`` is the typically-developing side of the DS/TD
    between-child contrast, a low bias there **overstates** the reported
    difference — but the Down syndrome side now carries a bias of the same sign
    and similar size, so the contrast is far less affected than either side
    alone. See ``notes/202608161700-recovery-baseline-215.md`` and issues #225
    and #229; treat the contrast as carrying this caveat until one reports.

    See ``notes/202608050900-td-hierarchical-geometry.md`` §§2, 4 and 7.1.
    """
    if not 0.0 < float(reference_proportion) < 1.0:
        raise ValueError(
            "reference_proportion must lie strictly in (0, 1); got "
            f"{reference_proportion!r}."
        )
    p0 = float(reference_proportion)
    c = 1.0 / (p0 * (1.0 - p0))

    v_total = total_dist.to_pymc(f"v_total{suffix}")
    share = share_dist.to_pymc(f"subject_variance_share{suffix}")
    subject_scale = pm.Deterministic(
        subject_scale_name, pm.math.sqrt(share * v_total)
    )
    excess_young_value = c / ((1.0 - share) * v_total)
    return subject_scale, excess_young_value


@dataclass(frozen=True)
class GPGrid:
    """The standardised-grid + HSGP scalars the trend/GP helpers need.

    These are values the engines already compute: the slope-anchor standardised
    reference ages (``sa_z``, ``sb_z``), the length-scale bounds on the z scale
    (``ell_low_z``, ``ell_high_z``), and the per-dimension HSGP basis counts ``M``
    and boundaries ``L`` (each a length-one list for the 1-D age kernel, passed
    straight to ``pm.gp.HSGP``). Bundling them keeps the helper signatures small and
    identical across engines.

    ``x_center_z`` optionally pins the HSGP basis centre (on the standardised age
    scale). Left ``None``, PyMC centres the basis on the midpoint of the min/max
    of whatever ``X`` reaches ``hsgp.prior`` — for these engines the stacked
    ``[obs, plot, query]`` grid, so a reporting query that extends past the
    observed range silently moves the approximation's accuracy region. Passing
    the declared GP domain's midpoint here decouples the basis from the
    reporting grid (#234). For every current model of record the two midpoints
    coincide, so pinning is a numerical no-op that removes latent regression
    debt rather than changing any fitted graph.
    """

    sa_z: float
    sb_z: float
    ell_low_z: float
    ell_high_z: float
    M: list[int]
    L: list[float]
    x_center_z: float | None = None


def _soft_clamp_z(z, grid):
    """Soft minimum of ``z`` and ``grid.sb_z`` — linear below, flat above.

    Smooth everywhere, unlike ``pt.minimum``, so the mean has no derivative jump
    at the anchor and the fitted curve inherits no elbow. Asymptotically exact in
    both directions: the departure from ``min(z, sb_z)`` decays exponentially away
    from the anchor and is at most ``log(2) / beta`` there.
    """
    beta = CLAMP_SOFTNESS / (grid.sb_z - grid.sa_z)
    return grid.sb_z - pt.softplus(beta * (grid.sb_z - z)) / beta


def trend_and_gp(
    *,
    cfg_low,
    cfg_hi,
    cfg_ell,
    cfg_eta,
    suffix,
    X_all_z_data,
    grid,
    store_deterministic,
    latent_name=None,
    anchor_idx=None,
    n_obs=None,
    clamp_above_hi=False,
):
    """Logit-linear trend + HSGP deviation; return the full-grid latent.

    ``suffix`` carries its own leading underscore (``""`` for single-outcome
    engines; ``"_u"`` / ``"_q"`` / ``"_sign"`` otherwise). When
    ``store_deterministic`` is true the GP value ``g{suffix}`` and the latent
    ``latent_name`` are stored as named ``Deterministic``\\ s (``dims=("all_id",)``);
    otherwise a plain tensor is returned (the trace-memory discipline used by the
    trivariate / joint engines). When ``anchor_idx`` is set the GP is
    orthogonalised against this mean's identifiable basis (coefficients fitted on
    the first ``n_obs`` observed rows only) and pinned to zero at the reference-age
    anchor row — so it carries no linear component over the observed rows and
    cannot alias with ``slope``. The pinning shift restores a constant component
    (fixed to zero at the reference age), so the level is identified by the point
    anchor itself rather than by orthogonality to ``[1]`` — see
    :func:`_orthogonalise_and_anchor` for exactly what the composition guarantees.

    ``clamp_above_hi`` levels the mean off above the high anchor instead of
    extrapolating the line. The Down syndrome GP domain runs to 115 months while
    the anchors sit at 24 and 84, so a quarter of the domain is extrapolation that
    no prior constrains, and on the logit scale a line that has to climb several
    logits between the anchors saturates there: VG10's fitted ``q`` mean alone
    reaches 0.993 at 115 months (P(mean > 0.99) = 0.90 across the posterior)
    against a realised 0.842, forcing the GP to spend −3.3 logits hauling it back
    while it is idle (+0.08) at 48 months where the data are. Levelling off leaves
    the GP free to carry departures rather than correct the mean's asymptote. It is
    deliberately **one-sided**: below the low anchor the line extrapolates
    accurately (VG10 ``q`` at 12 months is 0.019 by extrapolation against a fitted
    0.022), and clamping there would instead pin young-age values at the 24-month
    level, which is much worse. See notes/202608042030-q-mean-extrapolation.md.

    The transition uses a **soft** minimum,
    ``sb_z - softplus(beta * (sb_z - z)) / beta``, rather than ``min(z, sb_z)``.
    A hard minimum is continuous but its derivative jumps at the anchor, and the
    fitted curve inherits a visible elbow there — in the first VG10 refit it made
    the spoken trajectory briefly *non-monotone* (428.6 words at 84.3 months
    dipping to 426.6 at 85.6), which is not defensible in a growth-curve figure.
    ``beta`` is set from the anchor span so the rounding is scale-free across
    models; the mean's largest departure from the hard-clamped form is
    ``slope * log(2) / beta``, at the anchor itself, decaying exponentially away
    from it in both directions.

    The cost is that ``p_slope_hi`` is no longer *exactly* the mean at the high
    anchor age — it is short by ``slope * log(2) / beta``, which at
    ``build_utils.CLAMP_SOFTNESS`` = 50 is 1.4% of the anchor span (about 0.8 months of age
    for the Down syndrome models). Between the anchors the mean is otherwise
    untouched, so both anchor priors carry over unchanged.
    """
    p_lo = cfg_low.to_pymc(f"p_slope_low{suffix}")
    p_hi = cfg_hi.to_pymc(f"p_slope_hi{suffix}")
    slope = pm.Deterministic(
        f"slope{suffix}", (logit(p_hi) - logit(p_lo)) / (grid.sb_z - grid.sa_z)
    )
    intercept = pm.Deterministic(
        f"intercept{suffix}", logit(p_lo) - slope * grid.sa_z
    )
    z = X_all_z_data[:, 0]
    # The GP must be orthogonalised against whatever the mean can actually
    # express, so the basis uses the same coordinate as the mean itself — with the
    # clamp on, the direction the mean can move in is z_eff, not z.
    z_eff = _soft_clamp_z(z, grid) if clamp_above_hi else z
    mean_trend = intercept + slope * z_eff
    nuisance_basis = (
        pt.stack([pt.ones_like(z), z_eff], axis=1) if anchor_idx is not None else None
    )
    return _gp_from_mean(
        mean_trend,
        cfg_ell=cfg_ell,
        cfg_eta=cfg_eta,
        suffix=suffix,
        X_all_z_data=X_all_z_data,
        grid=grid,
        store_deterministic=store_deterministic,
        latent_name=latent_name,
        anchor_idx=anchor_idx,
        n_obs=n_obs,
        nuisance_basis=nuisance_basis,
    )


def tent_and_gp(
    *,
    cfg_low,
    cfg_mid,
    cfg_hi,
    z_low,
    z_mid,
    z_hi,
    cfg_peak=None,
    cfg_ell,
    cfg_eta,
    suffix,
    X_all_z_data,
    grid,
    store_deterministic,
    latent_name=None,
    anchor_idx=None,
    n_obs=None,
):
    """Three-anchor "tent" mean (rise to a peak anchor, then decline) + HSGP.

    Used for the signed ratio ``r(a) = P(sign | understood)``, whose developmental
    trajectory is a hump — near zero at young ages, peaking in the preschool years,
    then receding as words move into speech — rather than the monotone trend of
    ``U``/``q``. Three Beta anchors give ``r`` at a young, a peak and an old
    reference age (``z_low < z_mid < z_hi`` on the standardised-age scale); the mean
    is two logit-linear segments meeting at the peak anchor, **clamped flat beyond
    the outer anchors** so it does not extrapolate to implausible values. The peak
    therefore sits at the middle anchor age by construction, and the GP carries
    smooth departures. When anchored the GP is orthogonalised against this mean's
    full basis — the three fixed tent hats spanning ``{p_low, p_mid, p_hi}``, a
    larger space than ``[1, z]`` — so it cannot mimic a *relative* shift of the
    anchors, then pinned to zero at the reference-age anchor row. The pinning
    restores a common constant, so orthogonality to the hats holds up to that
    constant rather than exactly (see :func:`_orthogonalise_and_anchor`).
    """
    p_low = cfg_low.to_pymc(f"p_slope_low{suffix}")
    p_mid = cfg_mid.to_pymc(f"p_slope_mid{suffix}")
    p_hi = cfg_hi.to_pymc(f"p_slope_hi{suffix}")
    if cfg_peak is not None:
        # Estimate WHERE the peak is, instead of asserting it. `peak_unit` places
        # the middle anchor between the outer two; the ordering z_low < z_mid <
        # z_hi therefore holds by construction, which a prior directly on the age
        # could not guarantee. Standardisation is affine, so a unit position in z
        # is the same unit position in months.
        #
        # The fixed anchor is not harmless. With the peak pinned at 36 months,
        # VG15 under-predicts the signed ratio at every band above it -- mean
        # residual +0.059 against -0.006 below, worst at 48-54 months where
        # observed 0.365 against fitted 0.242 -- and the residual sign flips
        # exactly at the knot, which random-effect marginalisation cannot produce.
        # The observed ratio is a plateau from roughly 30 to 54 months, not a peak
        # at 36. See notes/202608060900-three-prior-conflicts.md.
        peak_unit = cfg_peak.to_pymc(f"peak_unit{suffix}")
        z_mid = pm.Deterministic(f"z_peak{suffix}", z_low + peak_unit * (z_hi - z_low))
    slope_up = pm.Deterministic(
        f"slope_up{suffix}", (logit(p_mid) - logit(p_low)) / (z_mid - z_low)
    )
    slope_dn = pm.Deterministic(
        f"slope_dn{suffix}", (logit(p_hi) - logit(p_mid)) / (z_hi - z_mid)
    )
    zc = X_all_z_data[:, 0]
    mean_tent = pm.math.switch(
        zc <= z_low,
        logit(p_low),
        pm.math.switch(
            zc <= z_mid,
            logit(p_low) + slope_up * (zc - z_low),
            pm.math.switch(
                zc <= z_hi,
                logit(p_mid) + slope_dn * (zc - z_mid),
                logit(p_hi),
            ),
        ),
    )
    if anchor_idx is not None:
        # Fixed partition-of-unity tent hats: mean_tent == logit(p_low)*phi_low +
        # logit(p_mid)*phi_mid + logit(p_hi)*phi_hi. Projecting the GP out of their
        # span removes exactly the directions that alias with the three anchors
        # (a strictly larger nuisance space than [1, z]).
        phi_low = pt.clip((z_mid - zc) / (z_mid - z_low), 0.0, 1.0)
        phi_hi = pt.clip((zc - z_mid) / (z_hi - z_mid), 0.0, 1.0)
        phi_mid = pt.clip(
            pt.minimum((zc - z_low) / (z_mid - z_low), (z_hi - zc) / (z_hi - z_mid)),
            0.0,
            1.0,
        )
        nuisance_basis = pt.stack([phi_low, phi_mid, phi_hi], axis=1)
    else:
        nuisance_basis = None
    return _gp_from_mean(
        mean_tent,
        cfg_ell=cfg_ell,
        cfg_eta=cfg_eta,
        suffix=suffix,
        X_all_z_data=X_all_z_data,
        grid=grid,
        store_deterministic=store_deterministic,
        latent_name=latent_name,
        anchor_idx=anchor_idx,
        n_obs=n_obs,
        nuisance_basis=nuisance_basis,
    )


def _orthogonalise_and_anchor(g_unit, nuisance_basis, n_obs, anchor_idx, *, ridge=1e-6):
    """Project the GP out of the mean's identifiable basis, then point-anchor it.

    ``nuisance_basis`` is the ``(n_all, k)`` design whose columns span the mean term
    the GP would otherwise alias with (``[1, z]`` for the logit-linear trend, ``[1]``
    for the free-intercept mean, the three tent hats for the peak mean). Two
    properties matter, and both are enforced here:

    * **Inference must not depend on the reporting grid.** ``X_all_z`` stacks the
      observations with plot points, query ages and the anchor row; the projection
      coefficients are therefore fitted on the first ``n_obs`` *observed* rows only
      (a fixed model design), then applied to every row. Changing ``n_plot`` /
      ``ages_query`` cannot move the observed-row latent, so it cannot move the
      likelihood or posterior.
    * **The reference-age anchor contract is preserved.** After removing the
      identifiable directions, the residual is shifted so it is exactly zero at
      ``anchor_idx`` — every posterior draw still passes through zero at the
      reference age, fixing the GP level against the mean (the reference age is a
      deliberate model choice, unlike the plot/query grids). The linear/tent
      directions removed above are the additional decoupling that stops the GP
      aliasing with ``slope`` / the anchors.

    The two steps do **not** compose into full-basis orthogonality, and this
    docstring must not claim they do (#240): subtracting ``g[anchor_idx]``
    restores a constant component that is generically nonzero over the observed
    rows, so the result is not orthogonal to the constant direction — nor, for
    the tent basis, to any individual hat except up to that shared constant.
    What survives exactly is the centred orthogonality (``z`` is standardised
    over the observed rows, so orthogonality to it is constant-invariant and the
    GP still carries no linear component there) and the point anchor, which is
    what fixes the level. No graph-identification failure follows: the constant
    direction is pinned by the anchor rather than projected away.

    A tiny ridge stabilises the normal-equations solve if a basis column is empty
    over the observed rows (e.g. a tent hat with no observations in its support).
    """
    B = nuisance_basis
    B_obs = B[:n_obs]
    g_obs = g_unit[:n_obs]
    gram = pt.dot(B_obs.T, B_obs) + ridge * pt.eye(B_obs.shape[1])
    coef = pt_solve(gram, pt.dot(B_obs.T, g_obs), assume_a="pos")
    g_unit = g_unit - pt.dot(B, coef)
    return g_unit - g_unit[anchor_idx]


def _gp_from_mean(
    mean_trend,
    *,
    cfg_ell,
    cfg_eta,
    suffix,
    X_all_z_data,
    grid,
    store_deterministic,
    latent_name,
    anchor_idx,
    n_obs=None,
    nuisance_basis=None,
):
    """Shared HSGP tail: build ell/eta/HSGP, sample ``g_unit``, combine with the mean.

    Factored out of :func:`trend_and_gp` / :func:`tent_and_gp` so the mean builders
    differ only in their mean term. ``ell_unit`` and ``eta`` are created after the
    mean term and before the single RNG-bearing ``hsgp.prior`` call, so the free-RV
    stream is identical across engines. When ``anchor_idx`` is set the GP is
    orthogonalised against ``nuisance_basis`` and pinned to zero at the reference
    row by :func:`_orthogonalise_and_anchor` (deterministic ops only — no new RVs).
    """
    if cfg_eta is None:
        # No GP at all: the mean carries the whole latent. For an outcome whose GP
        # hyperparameters are unidentifiable, sampling them adds prior-driven
        # spread to the reported band without adding information -- VG15's signed
        # GP contributes a posterior-median curve of at most 0.11 logits (7% of the
        # tent's range) while injecting a per-age posterior sd of 0.269, six times
        # larger. Dropping it makes a trajectory that is already parametric in
        # substance parametric in form, and says so.
        if store_deterministic:
            return pm.Deterministic(latent_name, mean_trend, dims=("all_id",))
        return mean_trend

    if isinstance(cfg_ell, (int, float)):
        # Fixed length-scale on the unit scale. Keeps the GP's flexibility while
        # removing a hyperparameter the data cannot inform (VG15's `ell_unit_sign`
        # reaches contraction 0.033). Still stored under its usual name so
        # downstream readers do not need to know which branch produced it.
        ell_unit = pm.Deterministic(f"ell_unit{suffix}", pt.as_tensor_variable(float(cfg_ell)))
    else:
        ell_unit = cfg_ell.to_pymc(f"ell_unit{suffix}")
    ell = pm.Deterministic(
        f"ell{suffix}", grid.ell_low_z + (grid.ell_high_z - grid.ell_low_z) * ell_unit
    )
    eta = cfg_eta.to_pymc(f"eta{suffix}")
    cov = pm.gp.cov.ExpQuad(1, ls=ell)
    hsgp = pm.gp.HSGP(cov_func=cov, m=grid.M, L=grid.L)
    if grid.x_center_z is not None:
        # PyMC (6.3.1) exposes no constructor argument for the basis centre; it
        # sets `_X_center` lazily from min/max of the X passed to `prior`, guarded
        # by a None check (pymc/gp/hsgp_approx.py). Pre-setting it here pins the
        # centre to the declared GP domain's midpoint so the reporting grid
        # cannot move the approximation. Covered by a regression test against the
        # locked PyMC version.
        hsgp._X_center = np.array([float(grid.x_center_z)])
    g_unit = hsgp.prior(f"g_unit{suffix}", X=X_all_z_data, dims="all_id")
    if anchor_idx is not None:
        if n_obs is None or nuisance_basis is None:
            raise ValueError(
                "anchored GP requires n_obs and nuisance_basis "
                f"(suffix={suffix!r}, anchor_idx={anchor_idx})"
            )
        g_unit = _orthogonalise_and_anchor(g_unit, nuisance_basis, n_obs, anchor_idx)
    if store_deterministic:
        g = pm.Deterministic(f"g{suffix}", eta * g_unit, dims=("all_id",))
        return pm.Deterministic(latent_name, mean_trend + g, dims=("all_id",))
    return mean_trend + eta * g_unit
