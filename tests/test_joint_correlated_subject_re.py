# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check VG24's correlated child effects against the independent VG15 structure.

At the identity correlation, the Cholesky transformation must reproduce the
independent block. Tests also check the named correlation entries, the LKJ
prior and unchanged marginal scale priors. The correlation field belongs on
a subclass so VG15's serialised definition stays unchanged.
"""

import math
from dataclasses import fields
from itertools import permutations

import numpy as np
import pymc as pm
import pytensor
import pytest

from vocab_growth.models.common_joint_modality import (
    SUBJECT_RE_CORRELATIONS,
    SUBJECT_RE_OUTCOMES,
)
from vocab_growth.models.definitions import (
    VG15,
    VG24,
    JointCorrelatedSubjectREModelDefinition,
    JointModelDefinition,
    _as_definition_subclass,
    validate_model_definition,
)

#: VG24 uses a three-dimensional correlation matrix with an LKJ(2) prior.
LKJ_ETA = 2.0
BLOCK_SIZE = 3


# ============================================================
# The definition
# ============================================================


def test_vg24_differs_from_vg15_only_in_naming_and_the_correlation():
    """The two models must differ in one substantive field and nothing else."""
    v15 = {f.name: getattr(VG15, f.name) for f in fields(VG15)}
    v24 = {f.name: getattr(VG24, f.name) for f in fields(VG24)}
    changed = {
        k for k in set(v15) | set(v24) if v15.get(k, "<absent>") != v24.get(k, "<absent>")
    }
    assert changed == {
        "model_id",
        "config_name",
        "banner",
        "subject_re_correlation_eta",
    }


def test_vg15_does_not_gain_the_field():
    """The subclass must not leak onto the parent class.

    A fit is validated by comparing the serialised definition field for field, so
    if this ever fails, every fitted VG15 output on disk becomes invalid at the
    same moment -- including the sensitivity arms fitted against it.
    """
    assert "subject_re_correlation_eta" not in {f.name for f in fields(VG15)}
    assert type(VG15) is JointModelDefinition


def test_vg24_is_the_subclass():
    assert isinstance(VG24, JointCorrelatedSubjectREModelDefinition)
    assert VG24.subject_re_correlation_eta == LKJ_ETA


def test_vg24_matches_vg20_and_vg23_on_eta():
    """Use the same LKJ eta across the compared models.

    Matrix dimension still changes each correlation's marginal prior. Equal
    eta alone does not establish equal regularisation.
    """
    from vocab_growth.models.definitions import VG20, VG23

    assert (
        VG24.subject_re_correlation_eta
        == VG20.subject_re_correlation_eta
        == VG23.subject_re_correlation_eta
    )


@pytest.mark.parametrize("drop", SUBJECT_RE_OUTCOMES)
def test_a_partial_block_is_rejected(drop):
    """Two of three flags must fail, not silently build a degenerate block."""
    definition = _as_definition_subclass(
        VG24,
        JointCorrelatedSubjectREModelDefinition,
        model_id="VG99",
        **{f"use_subject_re_{drop}": False},
    )
    with pytest.raises(ValueError, match=f"use_subject_re_{drop}=True"):
        validate_model_definition(definition)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), True])
def test_a_non_positive_eta_is_rejected(bad):
    """`eta = 0` must reach the check that rejects it, not read as "off"."""
    definition = _as_definition_subclass(
        VG24,
        JointCorrelatedSubjectREModelDefinition,
        model_id="VG99",
        subject_re_correlation_eta=bad,
    )
    with pytest.raises(ValueError, match="positive finite LKJ concentration"):
        validate_model_definition(definition)


def test_vg15_still_validates():
    """The new checks must be inert for a definition that sets no correlation."""
    validate_model_definition(VG15)
    validate_model_definition(VG24)


# ============================================================
# The arithmetic the engine block relies on
# ============================================================


def test_the_identity_correlation_reproduces_the_independent_block_exactly():
    """At the identity, `z @ chol.T` is `tau * z` -- VG15's expression, op for op.

    ``LKJCholeskyCov`` returns the Cholesky factor of the COVARIANCE, so it
    carries the scales; at the identity correlation that factor is ``diag(tau)``.
    Exact equality, not a tolerance: this is what makes VG15 nested rather than
    merely close.
    """
    rng = np.random.default_rng(24)
    tau = np.array([0.7, 1.1, 1.3])
    z = rng.standard_normal((9, len(tau)))

    correlated = z @ np.diag(tau).T
    independent = z * tau

    np.testing.assert_array_equal(correlated, independent)


@pytest.mark.parametrize("rho", [-0.8, -0.25, 0.0, 0.3, 0.75])
def test_the_block_delivers_the_stated_correlation_and_leaves_the_scales_alone(rho):
    """A correlated draw must have the correlation asked for and the SD of tau.

    If the Cholesky were applied on the wrong side, or the scales multiplied in
    twice, the marginal spread of the child effects would move with the
    correlation -- silently rescaling a reported quantity.
    """
    rng = np.random.default_rng(240)
    tau = np.array([0.7, 1.1, 1.3])
    correlation = np.array(
        [[1.0, rho, 0.0], [rho, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    covariance_chol = np.linalg.cholesky(np.outer(tau, tau) * correlation)

    deviations = rng.standard_normal((400_000, 3)) @ covariance_chol.T

    assert np.corrcoef(deviations[:, 0], deviations[:, 1])[0, 1] == pytest.approx(
        rho, abs=0.01
    )
    for position, scale in enumerate(tau):
        assert deviations[:, position].std() == pytest.approx(scale, rel=0.01)


def test_the_correlation_names_index_the_matrix_they_claim_to():
    """`rho_sign_q` must be the sign/q entry, not whichever the ordering gives.

    Spelt out rather than derived from the names, because deriving it from the
    names is what the code under test does: a rule that reproduces the bug
    cannot detect it. If ``SUBJECT_RE_OUTCOMES`` is ever reordered, this fails
    and the reordering has to be deliberate.
    """
    assert SUBJECT_RE_OUTCOMES == ("u", "q", "sign")
    assert SUBJECT_RE_CORRELATIONS == (
        ("rho_uq", 0, 1),
        ("rho_u_sign", 0, 2),
        ("rho_sign_q", 1, 2),
    )
    # Every off-diagonal named exactly once, none twice, none on the diagonal.
    positions = {(row, column) for _, row, column in SUBJECT_RE_CORRELATIONS}
    assert len(positions) == BLOCK_SIZE * (BLOCK_SIZE - 1) // 2
    assert all(row < column for row, column in positions)


# ============================================================
# The primitive: why LKJCholeskyCov and not LKJCorr
# ============================================================


def test_lkjcorr_is_not_a_correlation_matrix():
    """`pm.LKJCorr` returns a Cholesky FACTOR, so `corr[i, j]` reads a zero.

    This is the trap VG24 was first written into: its own docstring example
    indexes the result as a correlation matrix, and doing so gives every
    correlation exactly 0.000 in every draw -- a model that looks like it fitted.

    Pinned as the reason the engine uses ``LKJCholeskyCov``. If a future PyMC
    makes ``LKJCorr`` return the matrix, this fails and the engine comment should
    be re-read rather than the test simply deleted: the exchangeability result
    below is the property that actually decides the choice.
    """
    with pm.Model():
        corr = pm.LKJCorr("corr", eta=LKJ_ETA, n=BLOCK_SIZE)
    drawn = pm.draw(corr, draws=64, random_seed=1)

    assert drawn.shape == (64, BLOCK_SIZE, BLOCK_SIZE)
    # Structurally zero above the diagonal, and rows of unit norm: a Cholesky
    # factor of a correlation matrix, not the matrix.
    assert np.all(drawn[:, 0, 1:] == 0.0)
    np.testing.assert_allclose(
        np.linalg.norm(drawn, axis=2), 1.0, atol=1e-9
    )
    np.testing.assert_allclose(
        np.diagonal(drawn @ np.swapaxes(drawn, 1, 2), axis1=1, axis2=2), 1.0, atol=1e-9
    )


@pytest.mark.slow
def test_the_correlation_prior_is_exchangeable_and_is_lkj():
    """Check the fitted density against the normalised three-dimensional LKJ(2).

    For a correlation matrix R, p(R) = 16 * det(R) / (3 * pi**2).
    Evaluate PyMC's density on packed covariance Cholesky factors L, then
    convert it to the density on scales s and correlations R. The absolute
    Jacobian determinant of (s, R) -> L is

        s[1] * s[2]**2 / sqrt((1 - R[0, 1]**2) * det(R)).

    Dividing out the independent HalfNormal scale densities must leave p(R).
    All six variable orderings must agree, including with unequal scales.
    These fixed evaluations check density shape and normalisation without
    depending on the sampler's random trajectories near zero scales.
    """
    with pm.Model() as model:
        pm.LKJCholeskyCov(
            "subject_re",
            eta=LKJ_ETA,
            n=BLOCK_SIZE,
            sd_dist=pm.HalfNormal.dist(sigma=[1.5] * BLOCK_SIZE, shape=BLOCK_SIZE),
            compute_corr=True,
            store_in_trace=False,
        )

    covariance_chol = model["subject_re"]
    packed = covariance_chol.type("packed")
    evaluate = pytensor.function([packed], pm.logp(covariance_chol, packed))
    correlations = (
        np.eye(3),
        np.array([[1.0, 0.6, 0.2], [0.6, 1.0, -0.3], [0.2, -0.3, 1.0]]),
        np.full((3, 3), 0.9) + 0.1 * np.eye(3),
        np.array([[1.0, 0.7, -0.7], [0.7, 1.0, -0.1], [-0.7, -0.1, 1.0]]),
    )
    scale_vectors = (
        np.ones(3),
        np.array([0.1, 1.5, 3.0]),
        np.array([3.0, 0.5, 0.25]),
    )
    for original in correlations:
        for order in permutations(range(BLOCK_SIZE)):
            correlation = original[np.ix_(order, order)]
            sign, logdet = np.linalg.slogdet(correlation)
            assert sign == 1
            expected = -math.log(3 * math.pi**2 / 16) + logdet
            for scales in scale_vectors:
                chol = np.linalg.cholesky(np.outer(scales, scales) * correlation)
                logp_chol = float(evaluate(chol[np.tril_indices(BLOCK_SIZE)]))
                logp_scales = (
                    3 * (0.5 * math.log(2 / math.pi) - math.log(1.5))
                    - 0.5 * np.sum((scales / 1.5) ** 2)
                )
                log_jacobian = (
                    math.log(scales[1])
                    + 2 * math.log(scales[2])
                    - 0.5 * math.log1p(-correlation[0, 1] ** 2)
                    - 0.5 * logdet
                )
                actual = logp_chol + log_jacobian - logp_scales
                assert actual == pytest.approx(expected, rel=0, abs=1e-9), (
                    order, scales, correlation, actual, expected
                )


@pytest.mark.slow
def test_the_scales_keep_vg15s_halfnormal_priors():
    """`sd_dist` must reproduce three independent HalfNormal(1.5), or the child
    scales are not comparable with VG15's."""
    sigma = VG24.tau_subj_u_sigma
    assert sigma == VG24.tau_subj_q_sigma == VG24.tau_subj_sign_sigma == 1.5

    with pm.Model() as model:
        _, _, stds = pm.LKJCholeskyCov(
            "subject_re",
            eta=LKJ_ETA,
            n=BLOCK_SIZE,
            sd_dist=pm.HalfNormal.dist(sigma=[sigma] * BLOCK_SIZE, shape=BLOCK_SIZE),
            compute_corr=True,
            store_in_trace=False,
        )
        for position, suffix in enumerate(SUBJECT_RE_OUTCOMES):
            pm.Deterministic(f"tau_subj_{suffix}", stds[position])

    idata = pm.sample(
        draws=4000,
        tune=1000,
        chains=4,
        model=model,
        random_seed=297,
        progressbar=False,
        compute_convergence_checks=False,
    )

    expected_mean = sigma * math.sqrt(2.0 / math.pi)
    expected_sd = sigma * math.sqrt(1.0 - 2.0 / math.pi)
    for suffix in SUBJECT_RE_OUTCOMES:
        values = idata.posterior[f"tau_subj_{suffix}"].values.ravel()
        assert values.mean() == pytest.approx(expected_mean, rel=0.08), suffix
        assert values.std() == pytest.approx(expected_sd, rel=0.10), suffix
