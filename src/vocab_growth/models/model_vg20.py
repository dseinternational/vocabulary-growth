# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG20 adds correlated child offsets to VG10's comprehension and spoken ratio.

``rho_uq`` describes whether children with higher comprehension for their age
also tend to speak a larger fraction of what they understand. At zero
correlation, the child-offset distribution reduces to VG10's independent pair.

The correlation changes the distribution of spoken vocabulary, whose mean
fraction is ``p_U * q`` at fixed parameter values. Its direction and size need
to be estimated. Correlations calculated from shrunken fitted child offsets
are descriptive estimates, not guaranteed lower bounds for ``rho_uq``.

Earlier estimates used overlapping data. Their historical values and limits
are discussed in notes/202608151120-vg16-cross-lag-quantified.md and
notes/202609131044-model-review-implementation.md.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG20


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG20)
