# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG26: VG21 + correlated subject random effects on (understood, q) (issue #240).

VG21 is the typically-developing reference for the matched-comprehension contrast
over 8-22 months, and like VG13 it draws each child's understood and
production-ratio deviations independently. VG23 showed that independence is
mis-specified on VG13's 8-18-month window (``rho_uq`` 0.128 [0.096, 0.160]), and
the Down syndrome side of the same contrast, VG20, already carries the
correlation. VG26 adds that one parameter to VG21, which is nested exactly at
``rho_uq = 0``, so one typically-developing model can serve both the trajectory
contrast and the correlation contrast.

Two things travel with any estimate from it. Both counts come from one parent's
questionnaire, so shared reporting tendency biases ``rho_uq`` upward, as it does
for VG20 and VG23. And 19-22 months is where comprehension measurement changes
in this pool, so ``rho_uq`` should be read beside VG23's 8-18-month value rather
than as a replacement for it.

See the registration comment above ``VG26`` in ``definitions.py`` and #240.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG26


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG26)
