# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG16: VG09 + a within-child cross-lag (issue #113).

Extends the bivariate + subject-random-intercept foundation with one population
lead-lag coefficient: a child's prior-wave understood residual (relative to their
own comprehension trajectory) predicts their current production ratio q, i.e.
earlier receptive vocabulary -> later expressive vocabulary, within child. With
the subject intercepts acting as random intercepts this is a random-intercept
cross-lagged panel (RI-CLPM). See
notes/202607031200-vg16-within-child-scoping.md.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG16


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG16)
