# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG21 extends VG13's TD joint model to ages 8-22 months.

The wider window also changes the anchors, GP domain and amplitude prior. It
extends support for comparisons at matched comprehension. The upper age limit
reflects concern about checklist ceiling effects at older ages, as described in
``notes/202608211100-window-22-adopted.md``.

Some anchor priors use the fitted data, and the dispersion priors were not
recalibrated for the wider window. See ``definitions.py`` for their values and
``docs/models/README.md`` for the model's reporting role.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG21


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG21)
