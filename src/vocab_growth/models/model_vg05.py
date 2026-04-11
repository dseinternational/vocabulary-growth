# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG05: Joint model of words understood and spoken (A -> U, A -> S, U -> S)
- children with Down syndrome
"""

from vocab_growth.models.common_bivariate import BivariateContext, fit_bivariate_model
from vocab_growth.models.definitions import VG05


def fit(config: str) -> BivariateContext:
    return fit_bivariate_model(config, VG05)
