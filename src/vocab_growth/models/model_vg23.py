# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG23 correlates child comprehension and spoken-share intercepts in TD.

It extends VG13's 8-18-month model with ``rho_uq``. At zero correlation, the
child-effect distribution reduces to VG13's independent pair. The registered
VG23 definition also includes sex as a covariate.

Paired counts can inform the child correlation, but both counts come from the
same questionnaire. Shared reporting tendencies may contribute to the fitted
association. The model cannot separate that contribution from vocabulary
differences, and it does not bound the true correlation from either side.
See ``docs/models/README.md`` for interpretation and reporting roles.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG23


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG23)
