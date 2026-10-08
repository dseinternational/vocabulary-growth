# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
VG08 jointly models understood and spoken vocabulary in Down syndrome.

It adds child random intercepts on comprehension to VG07's study effects.
The child intercepts describe persistent differences between children and
induce dependence among repeated observations from the same child.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG08


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG08)
