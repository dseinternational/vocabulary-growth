# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG19 adds child slopes to VG10's comprehension and spoken-share intercepts.

Each outcome has its own correlated intercept-and-slope block. The two blocks
are independent. The child-effect distribution reduces to VG10's when both
slope scales are zero. VG20 instead correlates the two outcomes' intercepts;
VG19 does not estimate that correlation or the correlation between their rates.

A positive-scale interval above zero does not by itself establish non-zero
rate variation. See ``docs/models/README.md`` for current reporting limits and
``notes/202608211500-vg19-registration.md`` for the comparison design.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG19


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG19)
