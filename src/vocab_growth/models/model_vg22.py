# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG22 links four child effects through a low-rank factor model.

The effects are comprehension level and rate and spoken-share level and rate.
A shared factor vector ``z`` gives ``b = L z`` with a ``(4, rank)`` loading
matrix. The resulting covariance is positive semidefinite. The definition
sets the rank; sensitivities compare ranks one, two and three.

Unlike VG19's independent outcome blocks, this structure permits associations
across outcomes and between levels and rates. Higher-rank estimates may depend
strongly on the prior when the data provide little information.
See ``docs/models/README.md`` for the limits on reporting rate magnitudes and
``notes/202608221000-four-by-four-gate1.md`` for the exploratory rationale.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG22


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG22)
