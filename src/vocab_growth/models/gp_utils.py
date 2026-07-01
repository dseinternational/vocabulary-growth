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
from dse_research_utils.statistics.models.pymc_utils import logit


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
):
    """Logit-linear trend + HSGP deviation; return the full-grid latent.

    Emits exactly the ops previously inlined in every engine, so it moves no
    random-variable creation and cannot change the model graph. ``suffix`` carries
    its own leading underscore (``""`` for single-outcome engines; ``"_u"`` /
    ``"_q"`` / ``"_sign"`` otherwise). When ``store_deterministic`` is true the GP
    value ``g{suffix}`` and the latent ``latent_name`` are stored as named
    ``Deterministic``\\ s (``dims=("all_id",)``); otherwise a plain tensor is
    returned (the trace-memory discipline used by the trivariate / joint engines).
    ``anchor_idx`` (Option D) centres the GP to pass through zero at that grid row
    for every draw.
    """
    p_lo = cfg_low.to_pymc(f"p_slope_low{suffix}")
    p_hi = cfg_hi.to_pymc(f"p_slope_hi{suffix}")
    slope = pm.Deterministic(
        f"slope{suffix}", (logit(p_hi) - logit(p_lo)) / (grid.sb_z - grid.sa_z)
    )
    intercept = pm.Deterministic(
        f"intercept{suffix}", logit(p_lo) - slope * grid.sa_z
    )
    mean_trend = intercept + slope * X_all_z_data[:, 0]
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
):
    """Intercept-only mean (no age slope) + HSGP deviation; full-grid latent.

    Used for the signed ratio, where a free age slope would extrapolate the ratio
    below the data floor: the mean is the free RV ``intercept{suffix}`` (created via
    ``to_pymc``, *not* a ``Deterministic``) and the GP carries the age-varying
    shape. Otherwise identical to :func:`trend_and_gp` (see it for the parameters).
    """
    intercept = cfg_intercept.to_pymc(f"intercept{suffix}")
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
    )


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
):
    """Shared HSGP tail: build ell/eta/HSGP, sample ``g_unit``, combine with the mean.

    Factored out of :func:`trend_and_gp` / :func:`intercept_and_gp` so the two
    differ only in their mean term. The op order is unchanged from the inlined
    engines: ``ell_unit`` and ``eta`` are created after the mean term and before the
    single RNG-bearing ``hsgp.prior`` call, so the free-RV stream is identical.
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
        g_unit = g_unit - g_unit[anchor_idx]
    if store_deterministic:
        g = pm.Deterministic(f"g{suffix}", eta * g_unit, dims=("all_id",))
        return pm.Deterministic(latent_name, mean_trend + g, dims=("all_id",))
    return mean_trend + eta * g_unit
