# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG13 jointly models understood and spoken vocabulary at 8-18 months in TD.

It includes study and child random intercepts. VG21 widens the age window to
8-22 months and supplies the matched-comprehension reference. VG13 remains
available as the narrower-window comparison for VG23.
See ``docs/models/README.md`` for current reporting roles.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG13


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG13)
