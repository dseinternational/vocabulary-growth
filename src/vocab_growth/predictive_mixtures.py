# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Condition a predictive mixture on an observed parent count.

Each array entry represents a parameter and child-effect draw. Observing the
parent changes their weights. Substituting its count into the child likelihood
alone does not condition the mixture.
"""

import numpy as np
from scipy.special import logsumexp


def conditional_log_predictive(log_parent, log_child, *, axis=None):
    """log p(child | parent) from equally weighted joint mixture components."""
    return logsumexp(log_parent + log_child, axis=axis) - logsumexp(log_parent, axis=axis)


def conditioned_resample(values, log_weights, rng):
    """Resample each column over all parameter and effect draws on axis zero."""
    values = np.asarray(values)
    log_weights = np.broadcast_to(log_weights, values.shape)
    weights = np.exp(log_weights - logsumexp(log_weights, axis=0, keepdims=True))
    if not np.all(np.isfinite(weights)):
        raise ValueError("Conditioning event has no finite predictive probability.")
    out = np.empty_like(values)
    for column in range(values.shape[1]):
        pick = rng.choice(len(values), size=len(values), p=weights[:, column])
        out[:, column] = values[pick, column]
    return out
