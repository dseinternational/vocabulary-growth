# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG14: Trivariate model of words understood, spoken and signed
(A -> U, A -> S, A -> Sign; U -> S, U -> Sign) - children with Down syndrome.

Adds signing as a third production modality on top of the bivariate
(understood + spoken) structure, via a signed ratio r(a) and the derived
total-expressive quantity p_any(a). See ``common_trivariate`` for the engine.
"""

from vocab_growth.models.common_trivariate import (
    TrivariateContext,
    fit_trivariate_model,
)
from vocab_growth.models.definitions import VG14


def fit(config: str) -> TrivariateContext:
    return fit_trivariate_model(config, VG14)
