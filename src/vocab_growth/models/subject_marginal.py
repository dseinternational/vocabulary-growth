# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Numerically integrate child effects for children observed once (experimental).

For one row, integrate the Beta-Binomial likelihood over a standard-normal
child deviate, with latent logit eta + tau*u. This retains tau and the child
variation in the model while removing that child's sampled coordinate.
Repeated children retain explicit effects shared across their rows.

Exact integration would preserve the posterior of retained parameters. Finite
quadrature can change it. The stored pointwise likelihood also changes from
conditional to marginal, so importance-sampling diagnostics can differ.
Fresh-child predictions differ from replicates conditional on fitted effects;
neither has guaranteed training-data coverage.

Adaptive Gauss-Hermite quadrature centres nodes near the integrand's mode.
The mode uses damped Newton steps on finite differences. Node placement is
held out of the gradient; see _marginal_logp for the resulting approximation.
Validation must cover values and gradients over each proposed parameter range.
No registered model enables this path.

Keep marginalised rows first, use the local density in both likelihood blocks,
and use double precision. Earlier graph variants produced non-finite gradients
under concurrent sampling. The rationale and measurements are recorded in
notes/202608231745-singleton-marginalisation.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import dse_research_utils.math.constants as math_constants
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.gradient import disconnected_grad

EPSILON = math_constants.EPSILON

#: Default node count. The definition can request more for a convergence check.
# Historical probes do not bound the error globally. For n=810, y=400, mu=8,
# kappa=1000 and sigma=0.5, 20 nodes overestimate log probability by about 0.751.
DEFAULT_QUADRATURE_NODES = 20

#: Fixed number of damped Newton updates in the mode approximation.
_NEWTON_STEPS = 3

#: Finite-difference step, in standard-normal child-deviate units.
_NEWTON_FD = 1e-3

#: Limits on the Newton update and mode, in prior standard deviations.
#: These numerical safeguards do not bound quadrature error or its direction.
_NEWTON_MAX_STEP = 2.0
_MODE_CLAMP = 8.0


def standard_normal_quadrature(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and log weights for ``E[f(u)]`` under ``u ~ Normal(0, 1)``.

    ``numpy.polynomial.hermite_e`` is the probabilists' Hermite family, whose
    weight function is the standard normal kernel, so its nodes need no
    ``sqrt(2)`` rescaling. The weights are normalised to sum to one: they sum to
    ``sqrt(2 pi)`` as returned, and normalising here is what makes a degenerate
    ``sigma = 0`` row reproduce the plain Beta-Binomial log density to a rounding
    error rather than to the weight sum's error.
    """
    if n_nodes < 2:
        raise ValueError(f"Quadrature needs at least 2 nodes; got {n_nodes}.")
    nodes, weights = np.polynomial.hermite_e.hermegauss(int(n_nodes))
    log_weights = np.log(weights) - np.log(weights.sum())
    return nodes, log_weights


@dataclass(frozen=True)
class SubjectPartition:
    """Observation rows split by whether their child is seen once or repeatedly.

    ``padded_codes`` indexes a ``delta_subject`` vector extended by one trailing
    zero: a repeat-measured row gets its child's position in ``repeat_labels``,
    and a marginalised row gets the sentinel position, so the single gather that
    used to read ``delta_subject[subject_obs]`` still reads one vector and
    returns an exact zero where the effect has been integrated out.
    """

    singleton_rows: np.ndarray
    repeat_rows: np.ndarray
    repeat_labels: np.ndarray
    padded_codes: np.ndarray
    n_subjects: int

    @property
    def is_singleton_first(self) -> bool:
        """Whether the rows are ordered so both blocks are contiguous slices."""
        return bool(
            np.array_equal(self.singleton_rows, np.arange(self.n_singleton_rows))
            and np.array_equal(
                self.repeat_rows,
                np.arange(self.n_singleton_rows, self.n_singleton_rows + self.n_repeat_rows),
            )
        )

    @property
    def n_repeat_subjects(self) -> int:
        return int(self.repeat_labels.size)

    @property
    def n_singleton_subjects(self) -> int:
        return int(self.n_subjects - self.n_repeat_subjects)

    @property
    def n_singleton_rows(self) -> int:
        return int(self.singleton_rows.size)

    @property
    def n_repeat_rows(self) -> int:
        return int(self.repeat_rows.size)

    def summary_rows(self) -> list[tuple[str, object]]:
        """Rows describing the split for the build-configuration table."""
        return [
            ("Marginalised children (seen once)", self.n_singleton_subjects),
            ("Sampled child effects (seen repeatedly)", self.n_repeat_subjects),
            ("Marginalised rows", self.n_singleton_rows),
            ("Conditional rows", self.n_repeat_rows),
        ]


def singleton_first_order(subject_codes: np.ndarray) -> np.ndarray:
    """Return a stable row order with marginalised children first.

    Contiguous slices avoid the gather/scatter graph that produced non-finite
    gradients in earlier concurrent-sampling probes. Keep the input order within
    each block.
    """
    codes = np.asarray(subject_codes, dtype=int)
    counts = np.bincount(codes, minlength=int(codes.max()) + 1 if codes.size else 0)
    return np.argsort(counts[codes] > 1, kind="stable")


def partition_subject_rows(subject_codes: np.ndarray) -> SubjectPartition:
    """Split rows by their child's number of administrations.

    ``subject_codes`` is the per-row integer child code the engine already
    builds. Children are counted over the rows given, so a subsample that leaves
    a child with a single row correctly marginalises it.
    """
    codes = np.asarray(subject_codes, dtype=int)
    if codes.ndim != 1:
        raise ValueError("subject_codes must be a one-dimensional array of rows.")
    if codes.size and codes.min() < 0:
        raise ValueError("subject_codes must be non-negative row codes.")
    n_subjects = int(codes.max()) + 1 if codes.size else 0
    counts = np.bincount(codes, minlength=n_subjects)
    is_repeat_row = counts[codes] > 1

    repeat_labels = np.flatnonzero(counts > 1)
    position_of = np.full(n_subjects, repeat_labels.size, dtype=int)
    position_of[repeat_labels] = np.arange(repeat_labels.size)

    return SubjectPartition(
        singleton_rows=np.flatnonzero(~is_repeat_row),
        repeat_rows=np.flatnonzero(is_repeat_row),
        repeat_labels=repeat_labels,
        padded_codes=position_of[codes],
        n_subjects=n_subjects,
    )


def zero_padded_subject_shift(delta_subject, partition: SubjectPartition):
    """Return each repeated child's shift and zero on marginalised rows.

    The likelihood integrates the latter rows' child effects. Their zero-effect
    linear predictor is not the integrated mean count.
    """
    padded = pt.concatenate([delta_subject, pt.zeros(1)])
    return padded[partition.padded_codes]


def _row_terms(value, kappa, *, n_trials):
    """The part of the Beta-Binomial log density that does not move with ``p``.

    The binomial coefficient and the ``kappa`` normaliser: constant across the
    quadrature nodes of a row, so they are added once after the node sum rather
    than at every node, and they cancel outright in the finite differences the
    mode search takes. ``alpha + beta`` is ``kappa`` exactly, so the normaliser
    needs no addition of its own.

    Everything is forced to double first. PyTensor types a Python scalar
    constant as the narrowest dtype that holds it -- ``gammaln(811)`` and
    ``gammaln(5.0)`` both come back ``float32`` -- and single-precision
    ``gammaln`` of an argument this large is wrong in the fourth decimal, which
    would swamp the accuracy the adaptive placement above exists to buy. The
    casts are free when the caller already passes doubles, which the engines do.
    """
    n = np.float64(n_trials)
    value = pt.cast(value, "float64")
    kappa = pt.cast(kappa, "float64")
    return (
        pt.gammaln(n + 1.0)
        - pt.gammaln(value + 1.0)
        - pt.gammaln(n - value + 1.0)
        - pt.gammaln(n + kappa)
        + pt.gammaln(kappa)
    )


def _node_terms(value, p, kappa, *, n_trials, epsilon):
    """The part that moves with ``p`` -- the four gammaln calls per node.

    Cast to double for the reason :func:`_row_terms` gives.
    """
    value = pt.cast(value, "float64")
    kappa = pt.cast(kappa, "float64")
    p = pt.clip(pt.cast(p, "float64"), epsilon, 1 - epsilon)
    alpha = p * kappa
    beta = kappa - alpha
    return (
        pt.gammaln(value + alpha)
        + pt.gammaln(np.float64(n_trials) - value + beta)
        - pt.gammaln(alpha)
        - pt.gammaln(beta)
    )


def betabinomial_logp(value, p, kappa, *, n_trials, epsilon=EPSILON):
    """Beta-Binomial log density, split so the node loop stays cheap.

    Identical to ``pm.logp(pm.BetaBinomial.dist(...), value)`` on the support
    (``tests/test_subject_marginal.py`` pins that), but written in ``gammaln``
    terms directly and in two halves: the quadrature evaluates :func:`_node_terms`
    at every node of every marginalised row, while :func:`_row_terms` -- which the
    generic path would recompute at each node -- is added once per row.
    """
    return _row_terms(value, kappa, n_trials=n_trials) + _node_terms(
        value, p, kappa, n_trials=n_trials, epsilon=epsilon
    )


def _log_integrand(u, value, mu, kappa, sigma, *, n_trials, epsilon):
    """``log phi(u) + log L(u)``, up to an additive constant in ``u``.

    The mode search takes first and second differences in ``u``, and both the
    ``-0.5 log(2 pi)`` of the prior and :func:`_row_terms` cancel in those, so
    neither is computed here.
    """
    return -0.5 * u * u + _node_terms(
        value,
        pm.math.sigmoid(mu + sigma * u),
        kappa,
        n_trials=n_trials,
        epsilon=epsilon,
    )


def _node_placement(value, mu, kappa, sigma, *, n_trials, epsilon):
    """Where to put the quadrature nodes for each row: mode and Laplace scale.

    Starts from a closed-form Gaussian approximation -- the child effect implied
    by the row's own count, shrunk towards the prior by the Beta-Binomial's
    information -- and refines it with damped Newton steps on finite differences
    of :func:`_log_integrand`. Capping the proposal scale at the prior scale is
    a numerical heuristic. The Beta-Binomial likelihood is not globally
    log-concave in this parameterisation, and the approximation error can have
    either sign. Check larger node counts against independent integration over
    the parameter range of each proposed use.
    """
    p_hat = (value + 0.5) / (np.float64(n_trials) + 1.0)
    eta_hat = pt.log(p_hat) - pt.log1p(-p_hat)
    # Beta-Binomial information about logit(p) from one row: n p (1 - p) in the
    # binomial limit, about kappa p (1 - p) when the Beta mixing dominates.
    info = n_trials * kappa * p_hat * (1.0 - p_hat) / (n_trials + kappa)
    precision = 1.0 + sigma * sigma * info
    u = pt.clip(sigma * info * (eta_hat - mu) / precision, -_MODE_CLAMP, _MODE_CLAMP)

    def integrand(x):
        return _log_integrand(
            x, value, mu, kappa, sigma, n_trials=n_trials, epsilon=epsilon
        )

    for _ in range(_NEWTON_STEPS):
        up, mid, down = integrand(u + _NEWTON_FD), integrand(u), integrand(u - _NEWTON_FD)
        first = (up - down) / (2.0 * _NEWTON_FD)
        second = pt.minimum((up - 2.0 * mid + down) / (_NEWTON_FD * _NEWTON_FD), -1e-6)
        step = pt.clip(-first / second, -_NEWTON_MAX_STEP, _NEWTON_MAX_STEP)
        u = pt.clip(u + step, -_MODE_CLAMP, _MODE_CLAMP)

    up, mid, down = integrand(u + _NEWTON_FD), integrand(u), integrand(u - _NEWTON_FD)
    second = pt.minimum((up - 2.0 * mid + down) / (_NEWTON_FD * _NEWTON_FD), -1e-6)
    return u, pt.minimum(1.0 / pt.sqrt(-second), 1.0)


def _conditional_logp(value, mu, kappa, *, n_trials, epsilon):
    """Evaluate the conditional density through the local Beta-Binomial helper.

    Earlier CustomDist graphs using pm.logp returned non-finite gradients during
    initialisation. The local density is tested against PyMC on the support.
    """
    p = pt.clip(pm.math.sigmoid(mu), epsilon, 1 - epsilon)
    return betabinomial_logp(value, p, kappa, n_trials=n_trials, epsilon=epsilon)


def _marginal_logp(value, mu, kappa, sigma, *, n_trials, nodes, log_weights, epsilon):
    """The same density with the row's child effect integrated out.

    Adaptive Gauss-Hermite: with nodes ``z = c + s x`` the change of variables
    contributes ``log s + x^2 / 2 - z^2 / 2`` per node, and the ``2 pi`` factors
    of the two Gaussians cancel exactly. Setting ``c = 0`` and ``s = 1`` recovers
    the plain prior-node rule, which is what makes the two comparable.
    """
    sigma_t = pt.as_tensor_variable(sigma)
    # The engines pass a scalar tau_subject; a per-row vector is tolerated, and
    # must already be restricted to the same rows as ``value`` -- see the
    # caller in :func:`subject_marginal_betabinomial`.
    sigma_col = sigma_t if sigma_t.ndim == 0 else sigma_t[:, None]
    centre, scale = _node_placement(
        value, mu, kappa, sigma_t, n_trials=n_trials, epsilon=epsilon
    )
    # Hold node placement out of the gradient. Differentiating the finite-
    # difference mode search produced non-finite gradients in earlier probes.
    # This drops the finite rule's derivative through node placement. Validate
    # both value and gradient errors against independent integration.
    centre = disconnected_grad(centre)
    scale = disconnected_grad(scale)
    z = centre[:, None] + scale[:, None] * nodes[None, :]
    log_terms = (
        log_weights[None, :]
        + 0.5 * nodes[None, :] ** 2
        + pt.log(scale)[:, None]
        - 0.5 * z * z
        + _node_terms(
            value[:, None],
            pm.math.sigmoid(mu[:, None] + sigma_col * z),
            kappa[:, None],
            n_trials=n_trials,
            epsilon=epsilon,
        )
    )
    # The row constants factor straight out of the node sum.
    return _row_terms(value, kappa, n_trials=n_trials) + pt.logsumexp(log_terms, axis=1)


def subject_marginal_betabinomial(
    name: str,
    *,
    mu,
    kappa,
    tau_subject,
    observed,
    n_trials: int,
    partition: SubjectPartition,
    n_nodes: int = DEFAULT_QUADRATURE_NODES,
    dims=None,
    epsilon: float = EPSILON,
):
    """The outcome likelihood with singleton child effects integrated out.

    One observed variable over every row, so the pointwise ``log_likelihood``,
    the posterior predictive, and every consumer that reads them keep the name
    and shape they have always had. Rows whose child is seen repeatedly take the
    unchanged conditional Beta-Binomial density, on the ``mu`` the engine built
    with that child's explicit effect in it; rows whose child is seen once take
    the quadrature marginal, on a ``mu`` that carries no child effect. The two
    blocks are computed separately -- as contiguous slices, which is why the
    data preparation orders marginalised rows first -- rather than running the
    quadrature everywhere with a zero spread on the repeat rows, which would
    cost about 40% more likelihood evaluations for the same answer.
    """
    if not partition.is_singleton_first:
        raise ValueError(
            "The marginalised likelihood needs its rows ordered with every "
            "marginalised row first; see singleton_first_order, which the data "
            "preparation applies when the definition asks for marginalisation."
        )
    nodes, log_weights = standard_normal_quadrature(n_nodes)
    n_marginal = partition.n_singleton_rows

    def logp(value, mu_, kappa_, sigma_):
        if n_marginal == 0:
            return _conditional_logp(
                value, mu_, kappa_, n_trials=n_trials, epsilon=epsilon
            )
        sigma_t = pt.as_tensor_variable(sigma_)
        marginal = _marginal_logp(
            value[:n_marginal],
            mu_[:n_marginal],
            kappa_[:n_marginal],
            sigma_t if sigma_t.ndim == 0 else sigma_t[:n_marginal],
            n_trials=n_trials,
            nodes=nodes,
            log_weights=log_weights,
            epsilon=epsilon,
        )
        if partition.n_repeat_rows == 0:
            return marginal
        conditional = _conditional_logp(
            value[n_marginal:],
            mu_[n_marginal:],
            kappa_[n_marginal:],
            n_trials=n_trials,
            epsilon=epsilon,
        )
        return pt.concatenate([marginal, conditional])

    def random(mu_, kappa_, sigma_, rng=None, size=None):
        mu_ = np.asarray(mu_, dtype=float)
        kappa_ = np.asarray(kappa_, dtype=float)
        sigma_ = np.asarray(sigma_, dtype=float)
        shape = (
            tuple(size)
            if size is not None
            else np.broadcast_shapes(mu_.shape, kappa_.shape)
        )
        mu_ = np.broadcast_to(mu_, shape)
        kappa_ = np.broadcast_to(kappa_, shape)
        # A marginalised row predicts a fresh child, which is exactly what its
        # likelihood integrates over; a repeat-measured row keeps the fitted
        # child effect already inside mu.
        shift = np.zeros(shape)
        if n_marginal:
            unit = rng.normal(size=shape[:-1] + (n_marginal,))
            shift[..., :n_marginal] = unit * sigma_[..., None]
        p = np.clip(1.0 / (1.0 + np.exp(-(mu_ + shift))), epsilon, 1 - epsilon)
        return rng.binomial(n_trials, rng.beta(p * kappa_, (1 - p) * kappa_))

    return pm.CustomDist(
        name,
        mu,
        kappa,
        tau_subject,
        logp=logp,
        random=random,
        observed=observed,
        dims=dims,
    )
