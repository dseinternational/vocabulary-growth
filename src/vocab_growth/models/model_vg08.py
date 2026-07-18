# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG08: Joint model of words understood and spoken with study + subject
random intercepts (on understood) - children with Down syndrome.

Extends VG07 by adding non-centered subject-level random intercepts on the
understood trajectory, partitioning between-child variability from within-child
repeated-measures correlation.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG08


def fit(config: str, *, render: bool = False) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG08, render=render)
