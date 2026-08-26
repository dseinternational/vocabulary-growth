# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Child-level prior predictive checks (issue #233).

The population prior figures set every random effect to zero, so a child-effect
model's prior figures contained no child and could not test the prior the model
was added for. These checks fill that gap in NumPy, from draws the model already
emits — which buys a second implementation of the unseen-child construction, and
therefore a drift risk. The first test here is the guard on that: it pins the
correlated branch against the graph's own ``unseen_child_correlated_delta_q`` at
shared standard normals.
"""

import numpy as np
import pytest
import xarray as xr

from vocab_growth.models import prior_child_checks as pcc


def _prior(**arrays):
    """A prior group in the ``(chain, draw, ...)`` shape the module reads."""
    data = {}
    for name, values in arrays.items():
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            data[name] = (("chain", "draw"), values[None, :])
        else:
            data[name] = (("chain", "draw", "dim"), values[None, ...])
    return xr.Dataset(data)


class _Definition:
    n_trials = 810
    subject_slope_ref_age_months = 36.0


# --------------------------------------------------------------------------
# The drift guard
# --------------------------------------------------------------------------


def test_the_correlated_branch_matches_the_graph_construction():
    """Two implementations of one construction must not diverge.

    `unseen_child_correlated_delta_q` is what the predictive path runs; this
    module reimplements it in NumPy because the prior check happens before that
    block exists in the graph. Evaluated at the same standard normals the two
    must agree exactly, or the prior figures describe a model the fit does not
    use. The graph expression is evaluated with its own `_z_subj_q_marg`
    replaced by a constant, so this is arithmetic rather than sampling.
    """
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.graph.replace import graph_replace

    from vocab_growth.models.common_bivariate import unseen_child_correlated_delta_q

    tau_u, tau_q, rho = 0.786, 1.285, 0.368
    z_u, z_q = 0.7431, -1.2049

    with pm.Model() as model:
        delta_q = unseen_child_correlated_delta_q(
            pt.constant(tau_u * z_u),
            tau_subj_u=tau_u,
            tau_subj_q=tau_q,
            rho=rho,
        )
    from_graph = float(
        graph_replace(delta_q, {model["_z_subj_q_marg"]: pt.constant(z_q)}).eval()
    )

    prior = _prior(
        f_u_plot=np.zeros(1),
        h_plot=np.zeros(1),
        tau_subj_u=np.array([tau_u]),
        tau_subj_q=np.array([tau_q]),
        rho_uq=np.array([rho]),
    )

    class _FixedNormals:
        """Returns the chosen normals, so the module's own code path runs.

        Restating the arithmetic here instead would test this test, not the
        module: the point is that `unseen_child_deltas` produces the graph's
        value, not that two copies of one formula agree.
        """

        @staticmethod
        def standard_normal(shape):
            return np.array([[z_u], [z_q]]).reshape(shape)

    _, delta_q_numpy = pcc.unseen_child_deltas(
        prior, _Definition(), [36.0], _FixedNormals()
    )

    assert float(delta_q_numpy[0, 0]) == pytest.approx(from_graph, rel=1e-12)


def test_the_correlated_branch_reproduces_rho_across_many_draws():
    """The realised correlation of the module's own deviates must be rho."""
    prior = _prior(
        f_u_plot=np.zeros(4000),
        h_plot=np.zeros(4000),
        tau_subj_u=np.full(4000, 0.786),
        tau_subj_q=np.full(4000, 1.285),
        rho_uq=np.full(4000, 0.368),
    )
    rng = np.random.default_rng(20260824)
    delta_u, delta_q = pcc.unseen_child_deltas(prior, _Definition(), [36.0], rng)

    assert np.corrcoef(delta_u[:, 0], delta_q[:, 0])[0, 1] == pytest.approx(
        0.368, abs=0.03
    )
    assert delta_u[:, 0].std() == pytest.approx(0.786, rel=0.05)
    assert delta_q[:, 0].std() == pytest.approx(1.285, rel=0.05)


# --------------------------------------------------------------------------
# Structure dispatch
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        (("tau_subj_u", "tau_subj_q"), "independent"),
        (("tau_subj_u", "tau_subj_q", "rho_uq"), "correlated"),
        (("tau_subj_u", "tau_subj_u_1"), "slope"),
        (("tau_subj_u", "subject_factor_loadings"), "factor"),
        ((), "none"),
    ],
)
def test_structure_dispatch(variables, expected):
    """The factor form emits the slope form's names, so order matters."""
    prior = _prior(**{name: np.zeros(3) for name in variables})
    assert pcc.child_effect_structure(prior, _Definition()) == expected


def test_a_constant_offset_gives_the_same_deviate_at_every_age():
    """The property that makes the trajectory figure read: no crossings."""
    prior = _prior(
        f_u_plot=np.zeros(500),
        tau_subj_u=np.full(500, 0.8),
        tau_subj_q=np.full(500, 1.2),
    )
    rng = np.random.default_rng(1)
    delta_u, _ = pcc.unseen_child_deltas(prior, _Definition(), [12, 36, 84], rng)

    assert np.allclose(delta_u[:, 0], delta_u[:, 1])
    assert np.allclose(delta_u[:, 1], delta_u[:, 2])


def test_a_rate_gives_a_deviate_that_moves_with_age():
    """And the property that distinguishes VG19 from it."""
    prior = _prior(
        f_u_plot=np.zeros(2000),
        tau_subj_u_0=np.full(2000, 0.751),
        tau_subj_u_1=np.full(2000, 0.176),
        tau_subj_u_rho=np.full(2000, -0.219),
        tau_subj_q_0=np.full(2000, 1.207),
        tau_subj_q_1=np.full(2000, 0.640),
        tau_subj_q_rho=np.full(2000, 0.469),
    )
    rng = np.random.default_rng(2)
    ages = [12.0, 36.0, 84.0]
    delta_u, _ = pcc.unseen_child_deltas(prior, _Definition(), ages, rng)

    assert not np.allclose(delta_u[:, 0], delta_u[:, 2])
    # tau0 is the spread AT the reference age, which is `ages[1]`.
    assert delta_u[:, 1].std() == pytest.approx(0.751, rel=0.06)
    # And the spread is a parabola in age, so it is wider away from that age.
    assert delta_u[:, 0].std() > delta_u[:, 1].std()

    # Children must be able to cross, which a constant offset cannot represent.
    first, last = delta_u[:, 0], delta_u[:, 2]
    order_changed = np.mean(
        (first[:-1] < first[1:]) != (last[:-1] < last[1:])
    )
    assert order_changed > 0.05, "a rate that never reorders children is not a rate"


def test_the_slope_reference_age_comes_from_the_definition():
    """`tau0` means the spread at a stated age; reading the wrong one moves it."""
    prior = _prior(
        f_u_plot=np.zeros(3000),
        tau_subj_u_0=np.full(3000, 0.75),
        tau_subj_u_1=np.full(3000, 0.5),
        tau_subj_u_rho=np.zeros(3000),
        tau_subj_q_0=np.full(3000, 1.0),
        tau_subj_q_1=np.full(3000, 0.5),
        tau_subj_q_rho=np.zeros(3000),
    )

    class _Ref24(_Definition):
        subject_slope_ref_age_months = 24.0

    rng = np.random.default_rng(3)
    delta, _ = pcc.unseen_child_deltas(prior, _Ref24(), [24.0, 36.0], rng)
    assert delta[:, 0].std() == pytest.approx(0.75, rel=0.06)
    assert delta[:, 1].std() > delta[:, 0].std()


# --------------------------------------------------------------------------
# Counts
# --------------------------------------------------------------------------


def test_spoken_is_drawn_conditional_on_the_understood_draw():
    """The nesting the likelihood uses. Drawing spoken against the reference
    inventory instead would let a child say more words than they understand,
    which is the defect this figure exists to be able to show."""
    n = 400
    prior = _prior(
        f_u_plot=np.zeros(n),
        h_plot=np.zeros(n),
        kappa_u_plot=np.full(n, 20.0),
        kappa_s_plot=np.full(n, 20.0),
        tau_subj_u=np.full(n, 0.8),
        tau_subj_q=np.full(n, 1.2),
    )
    prior = prior.assign(X_plot=xr.DataArray(np.array([12.0]), dims="plot_dim"))
    rng = np.random.default_rng(4)
    curves = pcc.unseen_child_curves(prior, _Definition(), rng, n_children=n)

    assert curves is not None
    assert np.all(curves["spoken"] <= curves["understood"]), (
        "a child cannot say more distinct words than they understand"
    )
    assert np.all(curves["understood"] <= _Definition.n_trials)
    assert np.all(curves["understood"] >= 0)


def test_no_child_effects_means_no_child_figures():
    """VG05 and VG07 carry no child effects, so there is nothing to draw."""
    prior = _prior(f_u_plot=np.zeros(5), h_plot=np.zeros(5))
    prior = prior.assign(X_plot=xr.DataArray(np.array([12.0]), dims="plot_dim"))
    rng = np.random.default_rng(5)
    assert pcc.unseen_child_curves(prior, _Definition(), rng) is None


@pytest.mark.parametrize(
    ("carried", "missing"),
    [("tau_subj_u", "tau_subj_q"), ("tau_subj_q", "tau_subj_u")],
)
def test_a_one_sided_child_effect_gives_a_zero_offset_on_the_other_outcome(
    carried, missing
):
    """VG08 has a child effect on U and none on q (issue #266 finding 2).

    The graph sets the missing outcome's shift to exactly zero, so the check
    must produce a zero offset at the expected shape rather than a KeyError
    before posterior sampling ever starts.
    """
    n = 300
    prior = _prior(
        f_u_plot=np.zeros((n, 2)),
        h_plot=np.zeros((n, 2)),
        kappa_u_plot=np.full((n, 2), 20.0),
        kappa_s_plot=np.full((n, 2), 20.0),
        **{carried: np.full(n, 0.8)},
    )
    prior = prior.assign(X_plot=xr.DataArray(np.array([12.0, 36.0]), dims="plot_dim"))
    rng = np.random.default_rng(6)

    deltas = dict(
        zip(
            ("tau_subj_u", "tau_subj_q"),
            pcc.unseen_child_deltas(prior, _Definition(), [12.0, 36.0], rng),
            strict=True,
        )
    )
    assert deltas[carried].shape == (n, 2)
    assert deltas[missing].shape == (n, 2)
    assert np.allclose(deltas[missing], 0.0)
    assert deltas[carried][:, 0].std() == pytest.approx(0.8, rel=0.15)

    curves = pcc.unseen_child_curves(prior, _Definition(), rng, n_children=n)
    assert curves is not None
    assert curves["structure"] == "independent"


@pytest.mark.parametrize(
    ("carried_prefix", "missing_prefix"),
    [("tau_subj_u", "tau_subj_q"), ("tau_subj_q", "tau_subj_u")],
)
def test_a_one_sided_slope_gives_a_zero_offset_on_the_other_outcome(
    carried_prefix, missing_prefix
):
    """The slope branch reads both blocks too; no registered model is one-sided
    with a slope today, but the same defect was latent there (issue #266)."""
    n = 500
    prior = _prior(
        f_u_plot=np.zeros(n),
        **{
            f"{carried_prefix}_0": np.full(n, 0.75),
            f"{carried_prefix}_1": np.full(n, 0.4),
            f"{carried_prefix}_rho": np.zeros(n),
        },
    )
    rng = np.random.default_rng(7)
    deltas = dict(
        zip(
            ("tau_subj_u", "tau_subj_q"),
            pcc.unseen_child_deltas(prior, _Definition(), [12.0, 36.0, 84.0], rng),
            strict=True,
        )
    )
    assert deltas[missing_prefix].shape == (n, 3)
    assert np.allclose(deltas[missing_prefix], 0.0)
    assert not np.allclose(deltas[carried_prefix], 0.0)
