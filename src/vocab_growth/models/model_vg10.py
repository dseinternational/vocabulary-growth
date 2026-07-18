# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG10: VG09 variant with the HSGP correction terms anchored to zero at a
reference age (Option D).

Variant of VG09 introduced to test whether the marginal r_hat / ess_tail
issues observed on the q-trajectory hyperparameters under VG09 reflect a
structural redundancy between the linear mean trend, the HSGP correction,
and the subject random intercepts. (VG10 was originally also given tighter,
VG07-posterior-derived q anchors -- "Option A" -- but #155 broadened those back
across the DS-joint family to remove the prior-data double-dipping, so the q
anchors now match VG09 and the GP anchor is the sole structural difference.)
See notes/202605131500-vg09-structural-options.md.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG10


def fit(config: str, *, render: bool = False) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG10, render=render)
