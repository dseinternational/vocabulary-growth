# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG11 models spoken vocabulary in typically developing children.

Study intercepts sum to zero. Child intercepts use the registered variance
partition. The Gaussian process correction is anchored at 19 months.
See ``definitions.py`` for the priors and language scope.
"""

from vocab_growth.models.common_univariate_re import (
    UnivariateREContext,
    fit_univariate_re_model,
)
from vocab_growth.models.definitions import VG11


def fit(config: str) -> UnivariateREContext:
    return fit_univariate_re_model(config, VG11)
