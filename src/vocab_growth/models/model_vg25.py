# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG25 adds an earlier-signing predictor to VG24's spoken marginal.

``beta_sign_lag`` multiplies a child's prior-wave signed share of comprehension,
relative to the child's fitted persistent signing level. Sources are assigned
per complete child-age wave. The coefficient is a prospective association;
it does not establish that signing causes later speech. VG24's correlated
child block describes persistent between-child associations separately.

The lag enters the spoken marginal only. The ``sign-lag-in-cells`` sensitivity
also includes it in cell likelihoods under a population baseline. The design
rationale is in ``notes/202609151930-vg25-lag-out-of-the-cells.md``.
See ``docs/models/vg25/index.qmd`` for the available repeated observations.
"""

from vocab_growth.models.common_joint_modality import (
    JointContext,
    fit_joint_model,
)
from vocab_growth.models.definitions import VG25


def fit(config: str) -> JointContext:
    return fit_joint_model(config, VG25)
