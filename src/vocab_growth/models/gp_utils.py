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

It also provides the trend + HSGP construction (``trend_and_gp`` /
``intercept_and_gp``) shared by every engine. Because these carry the sole
RNG-bearing call (``hsgp.prior()``), they are parameterised (``suffix``,
``store_deterministic``, ``latent_name``, ``anchor_idx``, ``grid``) so each engine
reproduces its previous PyMC graph byte-for-byte (same free RVs, in the same order,
with the same names and ``logp``); the named-``Deterministic`` differences between
engines change only what is stored in the trace, not the sampled distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import pymc as pm
import pytensor.tensor as pt
from dse_research_utils.statistics.models.pymc_utils import logit
from pytensor.tensor.linalg import solve as pt_solve


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
    """

    sa_z: float
    sb_z: float
    ell_low_z: float
    ell_high_z: float
    M: list[int]
    L: list[float]


#: Sharpness of the soft clamp above the high anchor, in units of the anchor span
#: (``beta = _CLAMP_SOFTNESS / (sb_z - sa_z)``). Stating it relative to the span
#: makes the rounding scale-free: whatever a model's age standardisation, the
#: mean's largest departure from a hard ``min(z, sb_z)`` is
#: ``slope * log(2) / beta`` = ``slope * (sb_z - sa_z) * log(2) / 50``, i.e. 1.4%
#: of the anchor span, which for the Down syndrome 24-84 month anchors is about
#: 0.8 months of age. Raising it sharpens the corner toward the hard clamp (and
#: its elbow); lowering it rounds the corner further below the anchor, which eats
#: into the region where ``p_slope_hi`` is meant to be interpretable.
_CLAMP_SOFTNESS = 50.0


def _soft_clamp_z(z, grid):
    """Soft minimum of ``z`` and ``grid.sb_z`` — linear below, flat above.

    Smooth everywhere, unlike ``pt.minimum``, so the mean has no derivative jump
    at the anchor and the fitted curve inherits no elbow. Asymptotically exact in
    both directions: the departure from ``min(z, sb_z)`` decays exponentially away
    from the anchor and is at most ``log(2) / beta`` there.
    """
    beta = _CLAMP_SOFTNESS / (grid.sb_z - grid.sa_z)
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
    anchor row — so it carries only nonlinear curvature and its level is fixed
    against ``intercept``/``slope`` (see :func:`_gp_from_mean`).

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
    ``_CLAMP_SOFTNESS`` = 50 is 1.4% of the anchor span (about 0.8 months of age
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


def intercept_and_gp(
    *,
    cfg_intercept,
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
    """Intercept-only mean (no age slope) + HSGP deviation; full-grid latent.

    Used for the signed ratio, where a free age slope would extrapolate the ratio
    below the data floor: the mean is the free RV ``intercept{suffix}`` (created via
    ``to_pymc``, *not* a ``Deterministic``) and the GP carries the age-varying
    shape. Otherwise identical to :func:`trend_and_gp` (see it for the parameters).
    When anchored the nuisance basis is ``[1]`` only — the mean carries no age slope,
    so a linear GP direction is genuine signal here and must not be projected out;
    only the level (co-identified with the free ``intercept``) is removed, then the
    GP is pinned to zero at the reference-age anchor row.
    """
    intercept = cfg_intercept.to_pymc(f"intercept{suffix}")
    nuisance_basis = (
        pt.stack([pt.ones_like(X_all_z_data[:, 0])], axis=1)
        if anchor_idx is not None
        else None
    )
    return _gp_from_mean(
        intercept,
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
    larger space than ``[1, z]`` — so it cannot mimic a shift of any anchor, then
    pinned to zero at the reference-age anchor row (see :func:`_gp_from_mean`).
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

    Factored out of :func:`trend_and_gp` / :func:`intercept_and_gp` so the two
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
