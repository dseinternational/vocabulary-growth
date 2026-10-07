# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG14 models understood, spoken and signed vocabulary in Down syndrome.

Spoken and signed vocabulary are shares of understood vocabulary. The derived
total expressive proportion ``p_any`` assumes that speech and signing are
independent within understood words. VG15 estimates their association instead.
"""

from vocab_growth.models.common_trivariate import (
    TrivariateContext,
    fit_trivariate_model,
)
from vocab_growth.models.definitions import VG14


def fit(config: str) -> TrivariateContext:
    return fit_trivariate_model(config, VG14)
