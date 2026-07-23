# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Synthetic guards for the anchored-GP orthogonalisation and the sum-to-zero
study random-intercept rescale.

These tests are deliberately data-free (no DuckDB, no MCMC): they exercise the
graph helpers on small hand-built designs so they run in CI, where ``pytest``
executes before ``scripts/prepare_data.py`` and the model-construction fixtures
that need the prepared database skip. They pin the statistical contracts of the
#176 conditioning fixes directly:

* the anchored GP is orthogonalised against its mean's basis using coefficients
  fitted on the observed rows only (so the plot/query reporting grid cannot leak
  into the observed-row latent), and it is pinned to zero at the reference-age
  anchor row (the ``anchor_g*_at_ref`` contract); and
* ``ZeroSumNormal`` rescaled by ``sqrt(K / (K - 1))`` restores each group's
  marginal prior variance to that of the original independent ``Normal(0, 1)``
  offsets, so only the group-mean degree of freedom is removed.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from vocab_growth.models.gp_utils import _orthogonalise_and_anchor


def _apply(g_np, basis_np, n_obs, anchor_idx):
    """Evaluate ``_orthogonalise_and_anchor`` on numeric inputs."""
    out = _orthogonalise_and_anchor(
        pt.as_tensor_variable(np.asarray(g_np, dtype="float64")),
        pt.as_tensor_variable(np.asarray(basis_np, dtype="float64")),
        n_obs,
        anchor_idx,
    )
    return np.asarray(out.eval())


def _linear_basis(z):
    return np.column_stack([np.ones_like(z), z])


def _grid(n_obs=60, n_plot=80, anchor_z=0.7, seed=0):
    """Observed rows, then plot rows, then a single appended anchor row."""
    rng = np.random.default_rng(seed)
    z_obs = np.linspace(-2.0, 2.0, n_obs)
    z_plot = np.linspace(-3.0, 3.0, n_plot)
    z = np.concatenate([z_obs, z_plot, [anchor_z]])
    anchor_idx = z.size - 1
    return z, n_obs, anchor_idx, rng


def _centered_cov(a, b):
    return float(((a - a.mean()) * (b - b.mean())).mean())


def test_point_anchor_is_exactly_zero():
    z, n_obs, anchor_idx, rng = _grid()
    g = rng.standard_normal(z.size)
    out = _apply(g, _linear_basis(z), n_obs, anchor_idx)
    assert abs(out[anchor_idx]) < 1e-9


def test_orthogonal_to_linear_trend_over_observed_rows():
    z, n_obs, anchor_idx, rng = _grid()
    g = 1.3 + 0.8 * z + rng.standard_normal(z.size)  # deliberate level + slope
    out = _apply(g, _linear_basis(z), n_obs, anchor_idx)
    z_obs, out_obs = z[:n_obs], out[:n_obs]
    # After projecting out [1, z] on the observed rows, the residual has no linear
    # component there (constant-invariant, so unaffected by the anchor shift). The
    # bound is set by the 1e-6 stabilising ridge, not the float epsilon.
    assert abs(_centered_cov(out_obs, z_obs)) < 1e-6


def test_observed_latent_is_invariant_to_the_reporting_grid():
    # Same observed rows and same observed g; different plot rows and plot g.
    z_obs = np.linspace(-2.0, 2.0, 60)
    anchor_z = 0.7
    rng = np.random.default_rng(1)
    g_obs = rng.standard_normal(z_obs.size)
    g_anchor = 0.42  # the anchor row is part of the model — hold it fixed too

    def build(n_plot, plot_seed):
        pr = np.random.default_rng(plot_seed)
        z_plot = np.linspace(-3.0, 3.0, n_plot)
        z = np.concatenate([z_obs, z_plot, [anchor_z]])
        g = np.concatenate([g_obs, pr.standard_normal(n_plot), [g_anchor]])
        out = _apply(g, _linear_basis(z), z_obs.size, z.size - 1)
        return out[: z_obs.size]

    out_a = build(n_plot=80, plot_seed=2)
    out_b = build(n_plot=200, plot_seed=3)
    # Changing only the plot grid (count and values) leaves the observed-row latent
    # unchanged: the projection coefficients come from the observed rows alone.
    assert np.allclose(out_a, out_b, atol=1e-10)


def test_intercept_only_basis_preserves_a_linear_trend():
    # For the free-intercept mean the nuisance basis is [1] only: a linear GP
    # direction is genuine signal and must NOT be projected out.
    z, n_obs, anchor_idx, rng = _grid()
    g = 0.5 + 0.9 * z + 0.1 * rng.standard_normal(z.size)
    out = _apply(g, np.ones((z.size, 1)), n_obs, anchor_idx)
    assert abs(out[anchor_idx]) < 1e-9
    # The slope over the observed rows survives (it was ~0.9, well clear of 0).
    z_obs, out_obs = z[:n_obs], out[:n_obs]
    slope = _centered_cov(out_obs, z_obs) / _centered_cov(z_obs, z_obs)
    assert abs(slope - 0.9) < 0.15


def test_tent_basis_removes_all_three_anchor_directions():
    # The peak ("tent") mean spans three hats, a larger space than [1, z]; the GP
    # must be orthogonal to every column over the observed rows.
    z, n_obs, anchor_idx, rng = _grid(anchor_z=1.0)
    z_low, z_mid, z_hi = -1.0, 0.0, 1.5
    phi_low = np.clip((z_mid - z) / (z_mid - z_low), 0.0, 1.0)
    phi_hi = np.clip((z - z_mid) / (z_hi - z_mid), 0.0, 1.0)
    phi_mid = np.clip(
        np.minimum((z - z_low) / (z_mid - z_low), (z_hi - z) / (z_hi - z_mid)), 0.0, 1.0
    )
    basis = np.column_stack([phi_low, phi_mid, phi_hi])
    g = 2.0 * phi_low - 1.0 * phi_mid + 0.5 * phi_hi + rng.standard_normal(z.size)
    out = _apply(g, basis, n_obs, anchor_idx)
    assert abs(out[anchor_idx]) < 1e-9
    out_obs = out[:n_obs]
    for j in range(basis.shape[1]):
        assert abs(_centered_cov(out_obs, basis[:n_obs, j])) < 1e-6


def test_zero_sum_rescale_preserves_marginal_variance():
    # ZeroSumNormal(sigma=1) has marginal variance (K-1)/K; rescaling sigma by
    # sqrt(K/(K-1)) restores it to 1 (the independent-Normal(0,1) marginal), so
    # only the group-mean DOF is removed, not the per-study prior scale.
    for K in (3, 4, 6):
        sigma = float(np.sqrt(K / (K - 1)))
        with pm.Model():
            x = pm.ZeroSumNormal("x", sigma=sigma, shape=K)
            draws = pm.draw(x, draws=40000, random_seed=0)
        assert np.allclose(draws.sum(axis=1), 0.0, atol=1e-6)
        assert abs(draws.var(axis=0).mean() - 1.0) < 0.05
