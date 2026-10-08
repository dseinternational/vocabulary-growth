# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG10 adds a per-draw GP anchor to VG09.

The Gaussian process correction is zero at the reference age of 54 months in
each draw. This separates the correction from the age trend and child
intercepts at that age. The other priors match VG09.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG10


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG10)
