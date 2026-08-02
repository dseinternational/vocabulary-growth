# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the shared HSGP/trend build helpers.

Covers the kappa dispersion-closure factory (``make_kappa_of_z`` — pinning the
closed form ``z -> kappa_min + exp(a_kappa + b_kappa * z)``), the two-anchor
builder over the same curve (``build_kappa_of_z_anchored``), the trend + HSGP
graph builders (``trend_and_gp`` / ``intercept_and_gp`` — checking which RVs and
deterministics they emit), and ``get_hsgp_hyperparams`` (the boundary/basis
sizing). Evaluating with constant inputs is sufficient (and matches the
per-engine usage, where the same expression is built from random variables the
caller has already created).
"""

import numpy as np
import preliz as pz
import pymc as pm
import pytest

from vocab_growth.models.common import get_hsgp_hyperparams
from vocab_growth.models.gp_utils import (
    GPGrid,
    build_kappa_of_z_anchored,
    intercept_and_gp,
    make_kappa_of_z,
    tent_and_gp,
    trend_and_gp,
)


def test_make_kappa_of_z_at_zero():
    f = make_kappa_of_z(2.0, 0.5, -0.3)
    # at z = 0: kappa_min + exp(a_kappa)
    assert np.isclose(float(f(0.0).eval()), 2.0 + np.exp(0.5))


def test_make_kappa_of_z_closed_form():
    kappa_min, a_kappa, b_kappa, z = 1.5, 0.2, -0.4, 0.7
    f = make_kappa_of_z(kappa_min, a_kappa, b_kappa)
    expected = kappa_min + np.exp(a_kappa + b_kappa * z)
    assert np.isclose(float(f(z).eval()), expected)


def test_make_kappa_of_z_monotone_for_negative_b():
    f = make_kappa_of_z(1.0, 0.0, -0.5)
    lo = float(f(-1.0).eval())
    hi = float(f(1.0).eval())
    # negative b_kappa => kappa decreases as standardised age increases
    assert lo > hi


# --- build_kappa_of_z_anchored -----------------------------------------------


def _anchored(anchor_z, young=30.0, old=3.0, floor=4.0, suffix=""):
    """Build the anchored kappa closure from point-mass priors, in a model.

    Each prior is a LogNormal with sigma driven to ~0 so the single prior draw is
    its median. That makes the anchor identity checkable exactly rather than
    distributionally: the point of the parameterisation is that ``kappa`` at the
    anchor ages *is* the floor plus the drawn excess, for every draw.
    """
    with pm.Model() as model:
        f = build_kappa_of_z_anchored(
            pz.LogNormal(mu=np.log(floor), sigma=1e-9),
            pz.LogNormal(mu=np.log(young), sigma=1e-9),
            pz.LogNormal(mu=np.log(old), sigma=1e-9),
            anchor_z=anchor_z,
            suffix=suffix,
        )
    return model, f


def test_anchored_kappa_hits_both_anchors():
    zy, zo = -1.3, 0.4
    _, f = _anchored((zy, zo))
    assert np.isclose(float(f(zy).eval()), 4.0 + 30.0)
    assert np.isclose(float(f(zo).eval()), 4.0 + 3.0)


def test_anchored_kappa_is_log_linear_above_the_floor():
    zy, zo = -1.0, 1.0
    _, f = _anchored((zy, zo))
    # midway between the anchors the excess is the geometric mean of the two
    mid = float(f(0.0).eval())
    assert np.isclose(mid, 4.0 + np.sqrt(30.0 * 3.0))


def test_anchored_kappa_admits_a_rising_trajectory():
    # excess_old > excess_young: the legacy b_kappa_mag >= 0 cannot express this
    _, f = _anchored((-1.0, 1.0), young=3.0, old=30.0)
    assert float(f(1.0).eval()) > float(f(-1.0).eval())


def test_anchored_kappa_emits_the_expected_variables():
    model, _ = _anchored((-1.0, 1.0), suffix="_u")
    assert {v.name for v in model.free_RVs} == {
        "kappa_min_u", "kappa_excess_young_u", "kappa_excess_old_u"
    }
    # a_kappa/b_kappa keep the legacy names so the two forms stay comparable
    assert {"a_kappa_u", "b_kappa_u", "kappa_young_u", "kappa_old_u"} == {
        d.name for d in model.deterministics
    }


def test_anchored_kappa_derives_the_legacy_coefficients():
    zy, zo = -1.2, 0.6
    model, f = _anchored((zy, zo), young=30.0, old=3.0, floor=4.0)
    b = float(model["b_kappa"].eval())
    a = float(model["a_kappa"].eval())
    assert np.isclose(b, (np.log(3.0) - np.log(30.0)) / (zo - zy))
    # a_kappa is the log-excess at z = 0, so the closure agrees with the legacy form
    assert np.isclose(float(f(0.37).eval()), 4.0 + np.exp(a + b * 0.37))


def test_anchored_kappa_rejects_unordered_anchors():
    with pytest.raises(ValueError, match="ordered"):
        _anchored((0.5, -0.5))


# --- trend_and_gp / intercept_and_gp -----------------------------------------

_GRID = GPGrid(sa_z=-1.0, sb_z=1.0, ell_low_z=0.2, ell_high_z=2.0, M=[10], L=[2.0])


def _model_with(call):
    """Build a throwaway model, invoke ``call(X_all_z_data)`` in it, return it."""
    x = np.linspace(-2.0, 2.0, 8).reshape(-1, 1)
    with pm.Model(coords={"all_id": range(8), "x_dim": range(1)}) as m:
        X = pm.Data("X_all_z", x, dims=("all_id", "x_dim"))
        call(X)
    return m


def test_trend_and_gp_stores_named_deterministics():
    def call(X):
        trend_and_gp(
            cfg_low=pz.Beta(alpha=1, beta=15),
            cfg_hi=pz.Beta(alpha=1.1, beta=1.1),
            cfg_ell=pz.Beta(alpha=2, beta=2),
            cfg_eta=pz.HalfNormal(sigma=1.0),
            suffix="_u",
            X_all_z_data=X,
            grid=_GRID,
            store_deterministic=True,
            latent_name="f_u_all",
        )

    m = _model_with(call)
    free = {v.name for v in m.free_RVs}
    det = {d.name for d in m.deterministics}
    assert {"p_slope_low_u", "p_slope_hi_u", "ell_unit_u", "eta_u"} <= free
    assert {"slope_u", "intercept_u", "ell_u", "g_u", "f_u_all"} <= det


def test_trend_and_gp_plain_tensor_keeps_no_latent_deterministic():
    def call(X):
        trend_and_gp(
            cfg_low=pz.Beta(alpha=1, beta=15),
            cfg_hi=pz.Beta(alpha=1.1, beta=1.1),
            cfg_ell=pz.Beta(alpha=2, beta=2),
            cfg_eta=pz.HalfNormal(sigma=1.0),
            suffix="_q",
            X_all_z_data=X,
            grid=_GRID,
            store_deterministic=False,
        )

    det = {d.name for d in _model_with(call).deterministics}
    assert {"slope_q", "intercept_q", "ell_q"} <= det  # scalars always stored
    assert {"g_q", "h_all"}.isdisjoint(det)  # latent kept as a plain tensor


def test_trend_and_gp_anchor_adds_no_free_rv():
    cfg = dict(
        cfg_low=pz.Beta(alpha=1, beta=15),
        cfg_hi=pz.Beta(alpha=1.1, beta=1.1),
        cfg_ell=pz.Beta(alpha=2, beta=2),
        cfg_eta=pz.HalfNormal(sigma=1.0),
        suffix="",
        grid=_GRID,
        store_deterministic=True,
        latent_name="f_all",
    )
    plain = _model_with(lambda X: trend_and_gp(X_all_z_data=X, anchor_idx=None, **cfg))
    anchored = _model_with(
        lambda X: trend_and_gp(X_all_z_data=X, anchor_idx=3, n_obs=6, **cfg)
    )
    # The anchor + orthogonalisation is a deterministic transform: same free RVs,
    # same order (no new sampled quantities).
    assert [v.name for v in plain.free_RVs] == [v.name for v in anchored.free_RVs]


def test_intercept_and_gp_intercept_is_free_no_slope():
    def call(X):
        intercept_and_gp(
            cfg_intercept=pz.Normal(mu=0, sigma=1),
            cfg_ell=pz.Beta(alpha=2, beta=2),
            cfg_eta=pz.HalfNormal(sigma=1.0),
            suffix="_sign",
            X_all_z_data=X,
            grid=_GRID,
            store_deterministic=False,
        )

    m = _model_with(call)
    free = {v.name for v in m.free_RVs}
    det = {d.name for d in m.deterministics}
    assert "intercept_sign" in free  # intercept-only mean is a free RV
    assert "slope_sign" not in det  # no slope for the signed trajectory
    assert "ell_sign" in det


def test_tent_and_gp_three_anchors_no_intercept():
    def call(X):
        tent_and_gp(
            cfg_low=pz.Beta(alpha=2, beta=20),
            cfg_mid=pz.Beta(alpha=3, beta=4),
            cfg_hi=pz.Beta(alpha=2, beta=16),
            z_low=-1.0,
            z_mid=0.0,
            z_hi=1.5,
            cfg_ell=pz.Beta(alpha=2, beta=2),
            cfg_eta=pz.HalfNormal(sigma=1.0),
            suffix="_sign",
            X_all_z_data=X,
            grid=_GRID,
            store_deterministic=False,
        )

    m = _model_with(call)
    free = {v.name for v in m.free_RVs}
    det = {d.name for d in m.deterministics}
    # Three anchor RVs (young / peak / old); no intercept-only intercept_sign.
    assert {"p_slope_low_sign", "p_slope_mid_sign", "p_slope_hi_sign"} <= free
    assert "intercept_sign" not in free
    # Two segment slopes are stored Deterministics; no single slope_sign.
    assert {"slope_up_sign", "slope_dn_sign"} <= det
    assert "slope_sign" not in det


# --- get_hsgp_hyperparams -----------------------------------------------------


def test_get_hsgp_hyperparams_L_is_c_times_half_range():
    # Grid deliberately skewed about zero: max|z| = 3.5 exceeds the half-range
    # 2.5. HSGP.prior_linearized centres inputs at the grid midpoint, so the
    # boundary must be c times the half-range — the same S that
    # approx_hsgp_hyperparams sizes (m, c) for — not c times max|z|.
    x_min, x_max = -1.5, 3.5
    X_all_z = np.array([x_min, -0.5, 0.0, 1.0, x_max]).reshape(-1, 1)
    ell_range_z = (0.3, 1.2)

    L, M = get_hsgp_hyperparams(X_all_z, ell_range_z)

    m, c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
        x_range=[x_min, x_max],
        lengthscale_range=list(ell_range_z),
        cov_func="expquad",
    )
    half_range = (x_max - x_min) / 2.0
    assert np.isclose(L[0], c * half_range)
    assert M == [m]
    # regression guard: the pre-fix boundary was c * max|z|, strictly larger here
    assert L[0] < c * max(abs(x_min), abs(x_max))


def test_get_hsgp_hyperparams_boundary_covers_centred_grid():
    x = np.array([-2.0, 0.0, 4.0])
    L, _ = get_hsgp_hyperparams(x.reshape(-1, 1), (0.5, 1.5))
    centred = x - (x.max() + x.min()) / 2.0
    # c >= 1.2, so the boundary always covers the midpoint-centred grid
    assert L[0] >= np.max(np.abs(centred))


