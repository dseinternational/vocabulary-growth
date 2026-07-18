# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG07: Joint model of words understood and spoken with study random intercepts
- children with Down syndrome
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG07


def fit(config: str, *, render: bool = False) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG07, render=render)
