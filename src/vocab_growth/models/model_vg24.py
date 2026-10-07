# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG24 correlates VG15's three child random intercepts in Down syndrome.

The joint prior adds correlations among comprehension, spoken share ``q`` and
signed share. ``rho_sign_q`` describes the association between children's
persistent signed and spoken shares. The child-effect distribution reduces
to VG15's when the correlation matrix is the identity.

The child scale priors retain VG15's HalfNormal(1.5) distributions. Child
shifts enter marginal counts, not cross-tabulation likelihoods. Thus paired
signed and spoken marginal observations inform ``rho_sign_q``; the cell
counts directly inform the separate word-level association ``psi``.
See ``docs/models/vg24/index.qmd`` for the available paired observations.
"""

from vocab_growth.models.common_joint_modality import (
    JointContext,
    fit_joint_model,
)
from vocab_growth.models.definitions import VG24


def fit(config: str) -> JointContext:
    return fit_joint_model(config, VG24)
