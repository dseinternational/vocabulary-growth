# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG12: Words understood (TD) with dataset-level study random intercepts

Extends VG04 with dataset-level study random intercepts (sampled in centred,
sum-to-zero form since 2026-08-05), a child random intercept whose scale is set
by the shared variance partition, and a GP anchor constraint at 19 months to
remove the GP–intercept ridge. See the VG12 definition in ``definitions.py``
for the registered graph; this docstring must track it (#240).
"""

from vocab_growth.models.common_univariate_re import (
    UnivariateREContext,
    fit_univariate_re_model,
)
from vocab_growth.models.definitions import VG12


def fit(config: str) -> UnivariateREContext:
    return fit_univariate_re_model(config, VG12)
