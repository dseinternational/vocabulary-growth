# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG09: Joint model of words understood and spoken with study + subject
random intercepts on BOTH the understood trajectory and the production ratio
- children with Down syndrome.

Extends VG08 by adding non-centered subject-level random intercepts on the
production ratio q in addition to the existing subject REs on understood.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG09


def fit(config: str, *, render: bool = False) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG09, render=render)
