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


def build_kappa_of_z_anchored(
    kappa_min_dist,
    excess_young_dist,
    excess_old_dist,
    *,
    anchor_z,
    suffix="",
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
    """
    z_young, z_old = (float(anchor_z[0]), float(anchor_z[1]))
    if not z_old > z_young:
        raise ValueError(
            f"kappa anchor_z must be ordered (young, old); got {anchor_z!r}."
        )
    kappa_min = kappa_min_dist.to_pymc(f"kappa_min{suffix}")
    excess_young = excess_young_dist.to_pymc(f"kappa_excess_young{suffix}")
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
):
    """Logit-linear trend + HSGP deviation; return the full-grid latent.

    ``suffix`` carries its own leading underscore (``""`` for single-outcome
    engines; ``"_u"`` / ``"_q"`` / ``"_sign"`` otherwise). When
    ``store_deterministic`` is true the GP value ``g{suffix}`` and the latent
    ``latent_name`` are stored as named ``Deterministic``\\ s (``dims=("all_id",)``);
    otherwise a plain tensor is returned (the trace-memory discipline used by the
    trivariate / joint engines). When ``anchor_idx`` is set the GP is
    orthogonalised against this mean's identifiable basis ``[1, z]`` (coefficients
    fitted on the first ``n_obs`` observed rows only) and pinned to zero at the
    reference-age anchor row — so it carries only nonlinear curvature and its level
    is fixed against ``intercept``/``slope`` (see :func:`_gp_from_mean`).
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
    mean_trend = intercept + slope * z
    nuisance_basis = pt.stack([pt.ones_like(z), z], axis=1) if anchor_idx is not None else None
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
