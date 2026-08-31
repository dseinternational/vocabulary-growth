# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG22's level–level correlation has a designed prior, not an induced one.

Issue #266 finding 5. The loading rows were sampled independently and
normalised, which left ``rho_uq`` with whatever prior the geometry happened to
give — **exactly the arcsine**, whose density piles at the extremes.
``P(|rho_uq| > 0.8) = 0.410`` and an 89% interval of ``[-0.985, +0.985]``,
against ``0.056`` and ``[-0.715, +0.715]`` under the ``LKJ(2)`` VG20 places on
the same quantity. The two models' posteriors on that quantity were therefore
not prior-comparable, and the induced prior depended on the anchor order, which
had been documented as a pure gauge choice.

The fix is exact rather than approximate, and the geometry that caused the
problem is what makes it so: the first anchor row is the constant ``e_0``, so
``rho_uq`` **is** the second row's first coordinate. A prior placed there is a
prior on the correlation.

These tests are arithmetic on the construction, not on a fit. What they pin:
the induced prior really was arcsine (so the finding is reproduced, not taken on
trust), the designed one really is VG20's, the five other correlations are
unmoved, and the parameter counts Gate 1 analysed are unchanged.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from vocab_growth.models.definitions import VG20, VG22

pytestmark = pytest.mark.slow

#: Effect order throughout the factor block.
EFFECTS = ("b0u", "b1u", "b0q", "b1q")
N_DRAWS = 400_000


def _prior_draws(definition, names, seed=3, draws=N_DRAWS):
    """Prior draws of ``names`` from the real built graph."""
    import contextlib
    import io

    import pymc as pm
    from support.synthetic_graphs import build_synthetic_model

    from vocab_growth.models.catalogue import get

    class _Patcher:
        def setattr(self, obj, name, value, raising=True):
            setattr(obj, name, value)

    engine = get("vg22").engine
    with contextlib.redirect_stdout(io.StringIO()):
        context = build_synthetic_model(
            definition, engine, output_dir="/tmp", monkeypatch=_Patcher()
        )
        with context.model:
            prior = pm.sample_prior_predictive(
                draws=draws, random_seed=seed, var_names=list(names)
            )
    return {name: prior.prior[name].values for name in names}


def _summary(rho):
    rho = np.asarray(rho).ravel()
    lo, hi = np.quantile(rho, [0.055, 0.945])
    return float(np.mean(np.abs(rho) > 0.8)), float(lo), float(hi)


def test_the_designed_prior_is_vg20s_on_the_same_quantity(tmp_path):
    """The whole point of the finding: the two must be prior-comparable.

    VG23's registration gives the reason for matching VG20's eta there, and it
    is the same reason here — a difference between two posteriors on the same
    quantity should be a difference between models, not between priors.
    """
    drawn = _prior_draws(VG22, ["rho_uq"])
    tail, lo, hi = _summary(drawn["rho_uq"])

    # VG20's LKJ(2) marginal, in closed form: (rho + 1) / 2 ~ Beta(2, 2).
    from scipy import stats

    eta = VG20.subject_re_correlation_eta
    assert VG22.subject_factor.rho_uq_eta == eta, (
        "VG22's correlation concentration no longer matches VG20's, so the two "
        "models' posteriors on rho_uq stop being prior-comparable"
    )
    expected_tail = 2 * stats.beta.sf((0.8 + 1) / 2, eta, eta)
    expected_lo = 2 * stats.beta.ppf(0.055, eta, eta) - 1
    expected_hi = 2 * stats.beta.ppf(0.945, eta, eta) - 1

    assert tail == pytest.approx(expected_tail, abs=0.005)
    assert lo == pytest.approx(expected_lo, abs=0.01)
    assert hi == pytest.approx(expected_hi, abs=0.01)
    # And far from what it replaced.
    assert tail < 0.10, f"P(|rho_uq| > 0.8) = {tail:.3f}; the arcsine gave 0.410"


def test_the_induced_prior_really_was_arcsine():
    """Reproduce the finding rather than take it on trust.

    The old construction is `X / sqrt(X^2 + Y^2)` with X Normal and Y
    HalfNormal, whose angle is uniform on (0, pi) — so the cosine is arcsine on
    (-1, 1), with closed-form tail `1 - (2/pi) arcsin(0.8)`.
    """
    rng = np.random.default_rng(11)
    x = rng.standard_normal(N_DRAWS)
    y = np.abs(rng.standard_normal(N_DRAWS))
    rho = x / np.sqrt(x**2 + y**2)
    tail, lo, hi = _summary(rho)
    assert tail == pytest.approx(1 - 2 / np.pi * np.arcsin(0.8), abs=0.005)
    assert tail == pytest.approx(0.410, abs=0.005)
    assert lo == pytest.approx(-0.985, abs=0.005)
    assert hi == pytest.approx(+0.985, abs=0.005)


def test_the_other_five_correlations_are_unchanged_and_uniform(tmp_path):
    """The reparameterisation moved rho_uq and nothing else.

    The remaining rows keep the direction distributions they had, so their
    induced marginals are the LKJ(1) they always were — flat, and stated on the
    model page rather than left implicit.
    """
    drawn = _prior_draws(VG22, ["subject_factor_corr"])
    corr = drawn["subject_factor_corr"].reshape(-1, 4, 4)
    for a, b in ((0, 1), (0, 3), (1, 2), (1, 3), (2, 3)):
        tail, lo, hi = _summary(corr[:, a, b])
        assert tail == pytest.approx(0.200, abs=0.01), f"rho({EFFECTS[a]},{EFFECTS[b]})"
        assert lo == pytest.approx(-0.890, abs=0.01)
        assert hi == pytest.approx(+0.890, abs=0.01)


def test_the_unit_rows_are_exactly_unit(tmp_path):
    """`Sigma_ii = tau_i ** 2` is documented as exact, and now is.

    It was not. The first anchor row had a single entry, so its norm *was* its
    magnitude, and a near-zero draw met the numerical floor and left the row
    short — 55 draws in two million more than 0.1% below, worst case 37%.
    Constructing that row as a constant removes the failure mode with it.
    """
    drawn = _prior_draws(VG22, ["subject_factor_corr"], draws=200_000)
    corr = drawn["subject_factor_corr"].reshape(-1, 4, 4)
    diagonal = np.array([corr[:, i, i] for i in range(4)])
    assert np.abs(diagonal - 1.0).max() < 1e-9


@pytest.mark.parametrize("rank, expected", [(1, 4), (2, 7), (3, 9)])
def test_gate_ones_rank_table_is_unchanged(rank, expected, tmp_path):
    """The design must not change what Gate 1 analysed.

    The identified free-covariance count is the same at every registered rank;
    what fell is the number of sampled parameters spent reaching it.
    """
    import contextlib
    import io

    from support.synthetic_graphs import build_synthetic_model

    from vocab_growth.models.catalogue import get

    class _Patcher:
        def setattr(self, obj, name, value, raising=True):
            setattr(obj, name, value)

    spec = dataclasses.replace(VG22.subject_factor, rank=rank)
    definition = dataclasses.replace(VG22, subject_factor=spec)
    with contextlib.redirect_stdout(io.StringIO()):
        context = build_synthetic_model(
            definition, get("vg22").engine, output_dir=str(tmp_path),
            monkeypatch=_Patcher(),
        )
    names = {rv.name for rv in context.model.free_RVs}
    entries = [n for n in names if n.startswith("subject_factor_w_")]
    rows = {n.split("_w_")[1][0] for n in entries}
    identified = len(entries) - len(rows) + int("rho_uq_raw" in names)
    assert 4 + identified == expected


def test_rank_one_has_no_correlation_to_design(tmp_path):
    """One latent dimension forces |rho| = 1, so no prior over (-1, 1) applies."""
    import contextlib
    import io

    from support.synthetic_graphs import build_synthetic_model

    from vocab_growth.models.catalogue import get

    class _Patcher:
        def setattr(self, obj, name, value, raising=True):
            setattr(obj, name, value)

    spec = dataclasses.replace(VG22.subject_factor, rank=1)
    definition = dataclasses.replace(VG22, subject_factor=spec)
    with contextlib.redirect_stdout(io.StringIO()):
        context = build_synthetic_model(
            definition, get("vg22").engine, output_dir=str(tmp_path),
            monkeypatch=_Patcher(),
        )
    names = {rv.name for rv in context.model.free_RVs}
    assert "rho_uq_raw" not in names
    # And `rho_uq` is still emitted, read off the correlation matrix.
    assert "rho_uq" in {d.name for d in context.model.deterministics}


def test_a_non_positive_concentration_is_refused():
    with pytest.raises(ValueError, match="rho_uq_eta must be positive"):
        dataclasses.replace(VG22.subject_factor, rho_uq_eta=0.0)
