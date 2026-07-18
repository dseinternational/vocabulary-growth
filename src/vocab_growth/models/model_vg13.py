# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG13: Joint words understood + spoken (TD, 8–18 months) with dataset-level
study random intercepts

Extends VG06 with study-level random intercepts and restricts to ages 8–18 months
where WG and Oxford CDI data are dense. Above 18 months, only a single dataset
(Floccia/Oxford CDI) contributes bivariate observations, making inference unreliable.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG13


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG13)
