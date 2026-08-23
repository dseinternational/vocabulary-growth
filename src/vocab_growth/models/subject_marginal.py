# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Exact marginalisation of the child effects only one observation ever sees.

A child assessed once contributes a single likelihood term, and its random
effect ``delta_subject`` appears in that term alone. Nothing else in the model
sees it, so it can be integrated out in closed form up to quadrature error:

    p(y | eta, kappa, tau) = INT phi(u) BetaBinom(y | sigmoid(eta + tau u), kappa) du

Children with repeated administrations keep an explicit ``delta_subject``: their
rows are coupled through it, and integrating a shared latent out of several
terms is not a one-dimensional integral.

**This is not the removal of the subject random effect.** The marginal
likelihood still contains ``tau_subject`` -- the mixing is integrated, not
deleted -- so the joint model, ``kappa``'s meaning as within-child dispersion,
and the posterior of every retained quantity are unchanged. What changes is the
sampled space: the thousands of prior-dominated singleton dimensions whose
conditional scale tracks ``tau_subject`` (the funnel mass that
``notes/202608050900-td-hierarchical-geometry.md`` section 4 measured as the
energy-BFMI driver) leave it. See
``notes/202608231410-td-geometry-remaining-levers.md`` section 3.

Two consequences are real rather than cosmetic, and are why this is a definition
flag and not an implementation detail:

* the pointwise ``log_likelihood`` of a marginalised row is the **marginal**
  predictive density, not the conditional one, so its ``elpd`` is not comparable
  with a fit made without marginalisation -- which is also why it is the right
  quantity for a leave-one-subject-out reading of a singleton row; and
* posterior predictive draws for a marginalised row draw a fresh child effect
  rather than reusing the fitted one, which widens the predictive interval for
  those rows to its honest marginal width.

Why the quadrature is adaptive
------------------------------

Gauss-Hermite nodes on the ``Normal(0, 1)`` prior -- the rule the lever was
proposed with -- are **not** accurate enough for these models. One
administration of an 810-word inventory is informative about that child's own
level, so the integrand is a narrow spike inside a unit-width prior, and prior
nodes sit roughly 0.6 apart where the spike's width is 0.2. Measured against a
mode-centred fine grid on the fitted rows and draws of the two models of record:

| rule                     | VG12 worst row | VG11 worst row |
| ------------------------ | -------------: | -------------: |
| prior nodes, 20          |        6.6e-02 |        1.5e+00 |
| prior nodes, 40          |        2.1e-03 |        3.9e-01 |
| prior nodes, 80          |        6.7e-06 |        6.1e-02 |
| **adaptive, 20**         |    **3.4e-06** |    **4.0e-05** |
| adaptive, 30             |              - |        4.5e-06 |

(VG11 is the harder model: its dispersion reaches ``kappa`` 714 at the youngest
ages against VG12's 98, and a larger ``kappa`` is a sharper spike.) The nodes
are therefore placed at the integrand's own mode and scaled by its curvature --
adaptive Gauss-Hermite, the rule ``lme4`` and ``glmmTMB`` use for the same
reason. The mode comes from a short damped Newton search on **finite
differences** of the log-integrand rather than on its analytic derivatives: that
keeps the whole expression a composition of ordinary Beta-Binomial evaluations,
so PyTensor differentiates it exactly without needing a third derivative of
``gammaln``, which it does not implement.
"""

from __future__ import annotations

from dataclasses import dataclass

import dse_research_utils.math.constants as math_constants
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.gradient import disconnected_grad

EPSILON = math_constants.EPSILON

#: Quadrature nodes used unless a definition asks for more. With the adaptive
#: placement below, twenty nodes hold the worst per-row error to 3.4e-06 on
#: VG12's fitted rows and 4.0e-05 on VG11's; thirty takes VG11 to 4.5e-06. The
#: node count is a definition field so the doubling sensitivity check is a
#: definition change, not a code change.
DEFAULT_QUADRATURE_NODES = 20

#: Damped Newton steps used to find each row's integrand mode. Measured on
#: VG11's rows, the search converges in two steps: three steps and five give the
#: same node placement to the last digit, so three is one step of headroom
#: rather than a tuning parameter.
_NEWTON_STEPS = 3

#: Step for the central differences the Newton search uses, in prior standard
#: deviations of the child effect. Large enough that cancellation in the second
#: difference stays below 1e-8 absolute, small enough that its own truncation
#: bias is far below the node-placement tolerance.
_NEWTON_FD = 1e-3

#: Largest Newton step, and the range the mode is confined to, both in prior
#: standard deviations. The clamps only bind on rows whose child effect would
#: have to be many prior standard deviations out -- rows contributing a
#: log-density of a few hundred negative nats, which a fit visits only in early
#: warmup if at all. There the rule degrades to an underestimate of a
#: numerically absent term rather than to an overestimate, which would be an
#: attractor.
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
    """A stable row order that puts every marginalised row before every other.

    The likelihood then reads its two blocks as **slices** rather than by
    indexing them out and permuting the results back. That is not a
    micro-optimisation: an advanced-index gather over a concatenation returned a
    non-finite gradient on this graph, and its in-place scatter counterpart
    raced across nutpie's threads. Slices and one concatenation have neither
    problem. Stable, so within each block the rows keep the order the data
    preparation gave them.
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
    """The per-row child shift: the child's effect, or an exact zero.

    The zero is structural, not estimated: a marginalised row's child effect
    lives inside the likelihood's quadrature rather than in the linear
    predictor, so ``f_obs`` and ``p_obs`` on those rows are the
    population-and-study prediction for the row, which is what they mean once
    the child effect has been integrated out.
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
    of :func:`_log_integrand`. The scale is capped at the prior's, which the
    exact conditional posterior of a log-concave likelihood cannot exceed, so a
    failed search underestimates a tail row rather than inventing mass.
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
    """The Beta-Binomial log density these models have always used.

    Written out through :func:`betabinomial_logp` rather than by calling
    ``pm.logp(pm.BetaBinomial.dist(...), value)``, which is what this block did
    first. The two agree on the value to 1e-9 -- a test pins that -- but PyMC's
    version carries its parameter checks into the graph, and inside a
    ``CustomDist`` those return a **non-finite gradient**: measured on the small
    VG12 build, 132 of 150 jittered points against 0 of 150 for the form below.
    A sampler cannot start from a point whose gradient is not finite, and nutpie
    rejected every initial point until this changed.
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
    # Where the nodes sit is held out of the gradient. Two reasons, one
    # necessary and one welcome. Necessary: the mode search divides second
    # differences by the square of a 1e-3 step, and differentiating through that
    # chain returns a non-finite gradient at a large fraction of the points a
    # sampler's initialisation jitter visits -- measured at 319 of 400 on the
    # small VG12 build -- and NUTS cannot start where the gradient is not
    # finite. Welcome: it prunes the backward pass through the mode search's
    # twelve extra density evaluations. It is legitimate because the value is
    # unchanged and the rule is, to quadrature accuracy, invariant to where its
    # nodes sit: the term this drops is the rule's sensitivity to the placement
    # times the placement's sensitivity to the parameters, and the first factor
    # is of the order of the quadrature error itself. What the sampler is given
    # is the exact gradient of the same rule with the nodes held fixed, which is
    # the same quadrature applied to the derivative of the integrand.
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
