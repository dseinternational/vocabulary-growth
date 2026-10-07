# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG26 correlates VG21's TD child comprehension and spoken-share intercepts.

It retains the 8-22-month age window and adds ``rho_uq``. At zero correlation,
the child-effect distribution reduces to VG21's independent pair.

Shared questionnaire reporting may contribute to the fitted correlation.
Changes in comprehension measurement at 19-22 months also limit comparisons
with VG23's 8-18-month estimate. Neither contribution is separately identified.
See ``docs/models/README.md`` for current reporting roles.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG26


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG26)
