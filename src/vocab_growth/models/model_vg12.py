# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG12: Words understood (TD) with dataset-level study random intercepts

Extends VG04 with study-level random intercepts (non-centred parameterisation)
and a GP anchor constraint at 19 months to remove the GP–intercept ridge.
"""

from vocab_growth.models.common_univariate_re import (
    UnivariateREContext,
    fit_univariate_re_model,
)
from vocab_growth.models.definitions import VG12


def fit(config: str, *, render: bool = False) -> UnivariateREContext:
    return fit_univariate_re_model(config, VG12, render=render)
