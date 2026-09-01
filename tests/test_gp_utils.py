# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the shared HSGP/trend build helpers.

Covers the kappa dispersion-closure factory (``make_kappa_of_z`` — pinning the
closed form ``z -> kappa_min + exp(a_kappa + b_kappa * z)``), the two-anchor
builder over the same curve (``build_kappa_of_z_anchored``), the trend + HSGP
graph builders (``trend_and_gp`` / ``tent_and_gp`` — checking which RVs and
deterministics they emit), ``get_hsgp_hyperparams`` (the boundary/basis
sizing), and VG22's child-factor builder (``build_child_factor`` — pinning which
raw loading entries exist, which are positive, and that the gauge is anchored on
effects with between-child variance). Evaluating with constant inputs is
sufficient (and matches the per-engine usage, where the same expression is built
from random variables the caller has already created).
"""

import numpy as np
import preliz as pz
import pymc as pm
import pytensor.tensor as pt
import pytest

from vocab_growth.models.build_utils import CLAMP_SOFTNESS
from vocab_growth.models.common import get_hsgp_hyperparams
from vocab_growth.models.definitions import SubjectFactorPriorParams
from vocab_growth.models.gp_utils import (
    CHILD_FACTOR_ANCHOR_ORDER,
    GPGrid,
    _soft_clamp_z,
    build_child_factor,
    build_kappa_of_z_anchored,
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


# --- trend_and_gp / tent_and_gp ----------------------------------------------

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


# --- clamp_above_hi ----------------------------------------------------------
#
# The mean levels off above the high anchor instead of extrapolating the line. The
# transition is a soft minimum, so the mean stays differentiable and the fitted
# curve inherits no elbow -- a hard min(z, sb_z) is continuous but kinks at the
# anchor, which made VG10's spoken trajectory briefly non-monotone. Each test
# draws the mean and the model's own intercept/slope jointly from one seeded draw,
# so identities are checked against the same realisation.

_SOFT_BETA = CLAMP_SOFTNESS / (_GRID.sb_z - _GRID.sa_z)
#: Largest departure of the soft form from a hard clamp, in z units, at the anchor.
_SOFT_MAX_DZ = np.log(2.0) / _SOFT_BETA


def _draw_mean(clamp, z_values, *, seed=17):
    """Draw (f_all, intercept, slope) jointly from one model with eta driven to ~0.

    A near-zero GP amplitude isolates the mean term, which is what the clamp
    changes; ``g_unit`` is still built so the graph matches production use.
    """
    x = np.asarray(z_values, dtype=float).reshape(-1, 1)
    n = x.shape[0]
    with pm.Model(coords={"all_id": range(n), "x_dim": range(1)}) as m:
        X = pm.Data("X_all_z", x, dims=("all_id", "x_dim"))
        trend_and_gp(
            cfg_low=pz.Beta(alpha=1, beta=15),
            cfg_hi=pz.Beta(alpha=1.1, beta=1.1),
            cfg_ell=pz.Beta(alpha=2, beta=2),
            cfg_eta=pz.HalfNormal(sigma=1e-9),
            suffix="",
            X_all_z_data=X,
            grid=_GRID,
            store_deterministic=True,
            latent_name="f_all",
            clamp_above_hi=clamp,
        )
        f, icpt, slope = pm.draw(
            [m["f_all"], m["intercept"], m["slope"]], random_seed=seed
        )
    return np.asarray(f), float(icpt), float(slope)


def test_clamp_above_hi_levels_the_mean_off_past_the_high_anchor():
    # _GRID anchors at sa_z = -1, sb_z = +1; the soft form is exact well above.
    f, icpt, slope = _draw_mean(True, [2.0, 3.0, 4.0])
    assert np.allclose(f, icpt + slope * _GRID.sb_z, atol=1e-6)


def test_clamp_above_hi_leaves_the_mean_linear_well_below_the_high_anchor():
    z = np.array([-2.0, -1.0, 0.0, 0.5])
    f, icpt, slope = _draw_mean(True, z)
    assert np.allclose(f, icpt + slope * z, atol=1e-4)


def test_clamp_above_hi_still_extrapolates_below_the_low_anchor():
    # One-sided by design: young-age extrapolation is accurate and must remain.
    z = np.array([-3.0, -2.0, -1.0])
    f, icpt, slope = _draw_mean(True, z)
    assert np.allclose(f, icpt + slope * z, atol=1e-4)
    assert f[0] < f[1] < f[2]  # still sloping, not pinned at the low anchor


def test_clamp_above_hi_costs_a_bounded_offset_at_the_anchor_itself():
    """The documented price of smoothness: p_slope_hi is no longer exact there."""
    f, icpt, slope = _draw_mean(True, [_GRID.sb_z])
    shortfall = (icpt + slope * _GRID.sb_z) - f[0]
    assert 0.0 < shortfall
    assert np.isclose(shortfall, slope * _SOFT_MAX_DZ, rtol=1e-3)


def test_clamp_above_hi_keeps_the_mean_monotone_through_the_anchor():
    """The property a hard min(z, sb_z) fails, and the reason for the soft form."""
    z = np.linspace(_GRID.sb_z - 1.0, _GRID.sb_z + 1.0, 400)
    f, _, _ = _draw_mean(True, z)
    assert np.all(np.diff(f) > 0)  # monotone: no dip at the anchor


def test_clamp_above_hi_has_no_derivative_jump_at_the_anchor():
    """No single step may carry an appreciable share of the slope's fall to zero.

    The slope necessarily *does* fall from its full value to zero across the
    anchor; what distinguishes the soft form is that it does so gradually. With a
    hard clamp the entire fall happens in one grid step, so this ratio is 1.
    """
    z = np.linspace(_GRID.sb_z - 1.0, _GRID.sb_z + 1.0, 4000)
    f, _, _ = _draw_mean(True, z)
    d1 = np.diff(f)
    biggest_single_step = np.abs(np.diff(d1)).max()
    total_fall = d1.max() - d1.min()
    assert biggest_single_step / total_fall < 0.05


def test_hard_clamp_would_jump_confirming_the_previous_test_can_fail():
    """Control: on a hard clamp the whole fall happens at the kink.

    The kink generally lands between two grid points rather than on one, so the
    fall is split across at most two steps; the ratio is therefore near 1 but not
    exactly 1. It still separates from the soft form's < 0.05 by an order of
    magnitude, which is what makes the preceding test meaningful.
    """
    z = np.linspace(_GRID.sb_z - 1.0, _GRID.sb_z + 1.0, 4000)
    hard = 1.7 * np.minimum(z, _GRID.sb_z)
    d1 = np.diff(hard)
    assert np.abs(np.diff(d1)).max() / (d1.max() - d1.min()) > 0.4


def test_without_the_clamp_the_mean_keeps_extrapolating_above_the_high_anchor():
    z = np.array([1.0, 2.0, 3.0])
    f, icpt, slope = _draw_mean(False, z)
    assert np.allclose(f, icpt + slope * z, atol=1e-6)
    assert f[2] > f[0]  # the behaviour the clamp exists to remove


def test_clamp_above_hi_adds_no_free_rv():
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
    plain = _model_with(lambda X: trend_and_gp(X_all_z_data=X, **cfg))
    clamped = _model_with(
        lambda X: trend_and_gp(X_all_z_data=X, clamp_above_hi=True, **cfg)
    )
    # Same sampled quantities in the same order, so the RNG stream is unchanged.
    assert [v.name for v in plain.free_RVs] == [v.name for v in clamped.free_RVs]


def _anchored_gp(clamp, seed=5):
    """Draw the anchored GP over a grid spanning both sides of the high anchor."""
    x = np.linspace(-2.0, 3.0, 12).reshape(-1, 1)
    with pm.Model(coords={"all_id": range(12), "x_dim": range(1)}) as m:
        X = pm.Data("X_all_z", x, dims=("all_id", "x_dim"))
        trend_and_gp(
            cfg_low=pz.Beta(alpha=1, beta=15),
            cfg_hi=pz.Beta(alpha=1.1, beta=1.1),
            cfg_ell=pz.Beta(alpha=2, beta=2),
            cfg_eta=pz.HalfNormal(sigma=1.0),
            suffix="",
            X_all_z_data=X,
            grid=_GRID,
            store_deterministic=True,
            latent_name="f_all",
            clamp_above_hi=clamp,
            anchor_idx=0,
            n_obs=12,
        )
        g = np.asarray(pm.draw(m["g"], random_seed=seed))
    return g, x[:, 0]


def test_clamp_above_hi_orthogonalises_against_the_clamped_coordinate():
    """The GP must be projected out of what the mean can express, not out of z.

    With the clamp on, the mean's identifiable directions are [1, z_eff];
    orthogonalising against [1, z] instead would leave the GP able to mimic a
    slope change above the high anchor. The projection removes both directions and
    the result is then shifted to zero at the anchor row, which re-introduces a
    constant — so the recoverable invariant is that the *centred* GP is orthogonal
    to the slope column it was projected against.
    """
    g, z = _anchored_gp(clamp=True)
    z_eff = np.asarray(
        _soft_clamp_z(pt.as_tensor_variable(z), _GRID).eval()
    )
    centred = g - g.mean()
    assert abs(float(centred @ z_eff)) < 1e-6
    # ... and demonstrably not against the unclamped coordinate, which differs
    # over the grid points above the high anchor.
    assert abs(float(centred @ z)) > 1e-3
    # The anchor contract still holds exactly.
    assert abs(float(g[0])) < 1e-9


def test_without_the_clamp_orthogonalisation_uses_the_raw_coordinate():
    g, z = _anchored_gp(clamp=False)
    centred = g - g.mean()
    assert abs(float(centred @ z)) < 1e-6
    assert abs(float(g[0])) < 1e-9


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


# ---------------------------------------------------------------------------
# build_child_factor: the gauge is anchored on effects that have variance
# ---------------------------------------------------------------------------
#
# The first `rep` fit of VG22 (2026-08-23) split its chains into mirror modes
# because the triangular identification anchored on `b1u`, the comprehension
# rate, whose between-child scale the data put at ~0.04: a row with no scale
# pins nothing, so the second factor's reflection was free. These tests pin the
# anchor order (b0u, b0q, b1q, b1u) and the structure it implies, which no test
# constrained before. See notes/202608231420-vg22-factor-anchor-bimodality.md.

_EFFECTS = ("b0u", "b1u", "b0q", "b1q")


def _build_child_factor(rank):
    spec = SubjectFactorPriorParams(rank=rank, tau1_u_sigma=0.5, tau1_q_sigma=0.5)
    n_subjects, n_obs = 5, 8
    rng = np.random.default_rng(0)
    with pm.Model(
        coords={
            "subject_id": range(n_subjects),
            "factor": range(rank),
            "child_effect4": _EFFECTS,
            "child_effect4_b": _EFFECTS,
        }
    ) as m:
        build_child_factor(
            spec,
            tau0_u_sigma=1.0,
            tau0_q_sigma=1.0,
            age_obs_months=rng.uniform(12.0, 60.0, size=n_obs),
            subject_obs=rng.integers(0, n_subjects, size=n_obs),
        )
    return m


def _raw_loading_entries(model):
    """``{(row, col): distribution name}`` for every raw loading entry."""
    out = {}
    for rv in model.free_RVs:
        if rv.name.startswith("subject_factor_w_"):
            out[(int(rv.name[-2]), int(rv.name[-1]))] = rv.owner.op.name
    return out


def test_child_factor_anchor_order_is_levels_then_q_rate_then_u_rate():
    assert CHILD_FACTOR_ANCHOR_ORDER == (0, 2, 3, 1)
    assert [_EFFECTS[i] for i in CHILD_FACTOR_ANCHOR_ORDER] == ["b0u", "b0q", "b1q", "b1u"]


# Since issue #266 finding 5 the two leading anchor rows are no longer sampled
# as raw entries: b0u's direction is the constant e_0, and b0q's is set by the
# designed `rho_uq`. Both changes remove a magnitude that cancelled before
# reaching Sigma. The rows that remain keep the normalise-a-Normal construction,
# because the alternative -- a chart on the sphere -- wraps at its azimuth and
# was measured losing up to 17x the effective sample size and reaching R-hat
# 1.053, which this project's gate fails.
@pytest.mark.parametrize(
    "rank, expected_entries, expected_positive",
    [
        # One column: b0u is the constant e_0 and b0q has no second column to
        # place a correlation in, so the three remaining rows carry a sign each.
        (1, {(1, 0), (2, 0), (3, 0)}, set()),
        # Two columns: b0q's row is `rho_uq`; b1q and b1u are free directions.
        (2, {(1, 0), (1, 1), (3, 0), (3, 1)}, set()),
        # Three columns: b1q pins column 3; b1u carries all three, none positive.
        (3, {(1, 0), (1, 1), (1, 2), (3, 0), (3, 1), (3, 2)}, {(3, 2)}),
    ],
)
def test_child_factor_anchors_on_live_effects(rank, expected_entries, expected_positive):
    entries = _raw_loading_entries(_build_child_factor(rank))
    assert set(entries) == expected_entries
    positive = {key for key, dist in entries.items() if dist == "halfnormal"}
    assert positive == expected_positive
    assert all(dist in {"halfnormal", "normal"} for dist in entries.values())
    # The comprehension rate (row 1) never carries a diagonal at any registered
    # rank: it is the one effect every fit of the family puts at ~0.
    assert all(row != 1 for row, _col in positive)


@pytest.mark.parametrize("rank, expected", [(1, 4), (2, 7), (3, 9)])
def test_child_factor_free_covariance_parameter_count(rank, expected):
    """Gate 1's rank table (4, 7, 9) is unchanged by the reparameterisation.

    This is the check that the design did not change what Gate 1 analysed: the
    identified count is the same, only the number of parameters spent reaching
    it has fallen. `rho_uq_raw` is one identified direction parameter; the raw
    entries contribute one fewer than they number, per row, to the unit-row
    normalisation.
    """
    m = _build_child_factor(rank)
    names = {rv.name for rv in m.free_RVs}
    rows = {int(name.split("_w_")[1][0]) for name in names if "_w_" in name}
    n_w = sum(1 for name in names if name.startswith("subject_factor_w_"))
    identified = n_w - len(rows) + int("rho_uq_raw" in names)
    assert 4 + identified == expected


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_child_factor_rows_carry_tau_and_sigma_has_rank_k(rank):
    m = _build_child_factor(rank)
    L, *taus = pm.draw(
        [
            m["subject_factor_loadings"],
            m["tau_subj_u_0"],
            m["tau_subj_u_1"],
            m["tau_subj_q_0"],
            m["tau_subj_q_1"],
        ],
        random_seed=7,
    )
    assert L.shape == (4, rank)
    assert np.allclose(np.linalg.norm(L, axis=1), taus)
    assert np.linalg.matrix_rank(L @ L.T) == rank


# --- HSGP basis centre pinning (GPGrid.x_center_z) -----------------------------


def test_hsgp_prior_respects_a_preset_basis_centre():
    """The locked-PyMC mechanism the pin relies on: a preset centre wins.

    ``pm.gp.HSGP.prior_linearized`` computes ``_X_center`` from the min/max of
    the X it receives only when the attribute is still None. If an upgrade ever
    removes that guard, ``GPGrid.x_center_z`` would silently stop pinning and
    the basis centre would follow the reporting grid again.
    """
    with pm.Model():
        cov = pm.gp.cov.ExpQuad(1, ls=1.0)
        hsgp = pm.gp.HSGP(cov_func=cov, m=[8], L=[5.0])
        hsgp._X_center = np.array([2.0])
        # Lazy midpoint of this X would be 1.0, not the preset 2.0.
        hsgp.prior("g", X=np.linspace(0.0, 2.0, 7).reshape(-1, 1))
        np.testing.assert_allclose(hsgp._X_center, [2.0])


def _trend_gp_model(X, x_center_z, n_rows):
    with pm.Model(coords={"all_id": np.arange(n_rows)}) as m:
        trend_and_gp(
            cfg_low=pz.Beta(alpha=2.0, beta=8.0),
            cfg_hi=pz.Beta(alpha=8.0, beta=2.0),
            cfg_ell=pz.Beta(alpha=2.0, beta=2.0),
            cfg_eta=pz.HalfNormal(sigma=1.0),
            suffix="",
            X_all_z_data=pt.as_tensor_variable(X),
            grid=GPGrid(
                sa_z=-1.0,
                sb_z=1.0,
                ell_low_z=0.3,
                ell_high_z=1.5,
                M=[8],
                L=[5.0],
                x_center_z=x_center_z,
            ),
            store_deterministic=True,
            latent_name="f_all",
        )
    return m


def test_pinning_the_centre_at_the_lazy_value_is_a_noop():
    """Models of record: pinning where the lazy centre already sits changes nothing."""
    X = np.linspace(-2.0, 2.0, 9).reshape(-1, 1)  # lazy midpoint = 0.0
    draw_lazy = pm.draw(_trend_gp_model(X, None, 9)["f_all"], random_seed=7)
    draw_pinned = pm.draw(_trend_gp_model(X, 0.0, 9)["f_all"], random_seed=7)
    np.testing.assert_allclose(draw_lazy, draw_pinned)
    # Sanity that the comparison can fail: a genuinely different centre moves it.
    draw_moved = pm.draw(_trend_gp_model(X, 0.5, 9)["f_all"], random_seed=7)
    assert not np.allclose(draw_lazy, draw_moved)


def test_pinned_centre_decouples_the_basis_from_appended_query_rows():
    """Extending the grid past the observed range must not move the shared rows.

    This is the #234 regression: without the pin, appending a reporting query
    beyond the data (VG04's 27- and 30-month queries) shifts the lazily computed
    basis centre and with it every basis feature, so the same free-RV draws give
    a different latent at the *same* ages.
    """
    X_base = np.linspace(-2.0, 2.0, 9).reshape(-1, 1)
    X_extended = np.vstack([X_base, [[3.0]]])  # lazy midpoint would move to 0.5
    draw_base = pm.draw(_trend_gp_model(X_base, 0.0, 9)["f_all"], random_seed=11)
    draw_extended = pm.draw(
        _trend_gp_model(X_extended, 0.0, 10)["f_all"], random_seed=11
    )
    np.testing.assert_allclose(draw_base, draw_extended[:9])
    # Unpinned, the same extension moves the shared rows — the defect the pin fixes.
    lazy_base = pm.draw(_trend_gp_model(X_base, None, 9)["f_all"], random_seed=11)
    lazy_extended = pm.draw(
        _trend_gp_model(X_extended, None, 10)["f_all"], random_seed=11
    )
    assert not np.allclose(lazy_base, lazy_extended[:9])

