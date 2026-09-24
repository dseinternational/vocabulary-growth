# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pointwise likelihood factors that leak a held-out value through a lag."""

from vocab_growth.models.definitions import MODEL_REGISTRY


def suppressed_outcomes(model_id: str) -> set[str]:
    definition = MODEL_REGISTRY.get(model_id.lower())
    if definition is None:
        raise ValueError(f"Unknown model {model_id!r}; cannot establish LOO policy.")
    terms: set[str] = set()
    if getattr(definition, "use_cross_lag", False):
        terms.add("y_u_obs")
    if getattr(definition, "use_sign_cross_lag", False):
        terms.update(("y_u_obs", "y_sign_obs"))
    return terms
