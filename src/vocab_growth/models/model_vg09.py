# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
VG09 jointly models understood and spoken vocabulary in Down syndrome.

It adds child random intercepts on the spoken share ``q`` to VG08's child
intercepts on comprehension. It also uses two-anchor dispersion priors.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG09


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG09)
