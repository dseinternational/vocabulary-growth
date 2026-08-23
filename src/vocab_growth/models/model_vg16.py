# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG16: VG09 + a prior-understood cross-lag (issue #113).

Extends the bivariate + subject-random-intercept foundation with one lead-lag
coefficient: a child's prior-wave understood residual predicts their current
production ratio q, i.e. earlier receptive vocabulary -> later expressive
vocabulary. The lag source is assigned per complete (subject, age)
administration wave (issue #242). The model definition chooses whether that
residual is population-relative or within-child; the headline VG16 definition
uses the population-relative baseline, whose coefficient reads as a
history-dependent mixture of between- and within-child association. See
notes/202607031200-vg16-within-child-scoping.md and
notes/202608231714-vg16-statistical-model-review.md.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG16


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG16)
