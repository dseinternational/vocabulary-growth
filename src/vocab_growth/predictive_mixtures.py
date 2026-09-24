# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Condition a predictive mixture on an observed parent count.

Each array entry represents a parameter and child-effect draw. Observing the
parent changes their weights. Substituting its count into the child likelihood
alone does not condition the mixture.
"""

import warnings

import numpy as np
from scipy.special import logsumexp


def conditional_log_predictive(log_parent, log_child, *, axis=None):
    """log p(child | parent) from equally weighted joint mixture components."""
    return logsumexp(log_parent + log_child, axis=axis) - logsumexp(log_parent, axis=axis)


def importance_weights(log_weights, *, axis=0, minimum_ess=100, label="conditional prediction"):
    """Normalised weights and Kish ESS; warn about weight concentration.

    ESS is 1/sum(w**2). It diagnoses weight concentration, not dependence
    between posterior samples or accuracy of the fitted model. The warning
    threshold is a numerical screening rule, not a coverage guarantee.
    """
    weights = np.exp(log_weights - logsumexp(log_weights, axis=axis, keepdims=True))
    if not np.all(np.isfinite(weights)):
        raise ValueError("Conditioning event has no finite predictive probability.")
    ess = 1 / np.sum(weights**2, axis=axis)
    if np.any(ess < minimum_ess):
        warnings.warn(
            f"{label}: importance-weight ESS as low as {np.min(ess):.1f} "
            f"(screening threshold {minimum_ess}); increase candidate draws and "
            "check numerical stability before interpreting intervals or scores.",
            RuntimeWarning, stacklevel=2,
        )
    return weights, ess


def conditioned_resample(values, log_weights, rng, *, return_ess=False):
    """Resample each column over all parameter and effect draws on axis zero."""
    values = np.asarray(values)
    log_weights = np.broadcast_to(log_weights, values.shape)
    weights, ess = importance_weights(log_weights)
    out = np.empty_like(values)
    for column in range(values.shape[1]):
        pick = rng.choice(len(values), size=len(values), p=weights[:, column])
        out[:, column] = values[pick, column]
    return (out, ess) if return_ess else out
