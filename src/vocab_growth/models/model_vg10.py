# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG10: VG09 variant with tighter q-anchor priors (Option A) and the
HSGP correction terms anchored to zero at a reference age (Option D).

Variant of VG09 introduced to test whether the marginal r_hat / ess_tail
issues observed on the q-trajectory hyperparameters under VG09 reflect a
structural redundancy between the linear mean trend, the HSGP correction,
and the subject random intercepts. See notes/202605131500-vg09-structural-options.md.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG10


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG10)
