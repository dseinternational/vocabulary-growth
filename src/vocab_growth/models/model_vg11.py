# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG11: Words spoken (TD) with dataset-level study random intercepts

Extends VG03 with dataset-level study random intercepts (sampled in centred,
sum-to-zero form since 2026-08-05), a child random intercept whose scale is set
by the shared variance partition, and a GP anchor constraint at 19 months to
remove the GP–intercept ridge. See the VG11 definition in ``definitions.py``
for the registered graph; this docstring must track it (#240).
"""

from vocab_growth.models.common_univariate_re import (
    UnivariateREContext,
    fit_univariate_re_model,
)
from vocab_growth.models.definitions import VG11


def fit(config: str) -> UnivariateREContext:
    return fit_univariate_re_model(config, VG11)
