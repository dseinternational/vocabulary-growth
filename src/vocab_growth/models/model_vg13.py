# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG13: Joint words understood + spoken (TD, 8–18 months) with dataset-level
study random intercepts

Succeeds the retired VG06 with study-level and child-level random intercepts,
restricted to ages 8–18 months where WG and Oxford CDI data are dense. The
original rationale for the cap — a single bivariate dataset above 18 months —
is obsolete: the Romance-language extension admits paired Italian WG rows to
24 months, and 694 admissible administrations from two studies now sit above
it. What remains of the cap is density (thin above 18 months) and the Oxford
CDI's 418-item ceiling binding at 23–25 months. The study owner adopted the
8–22-month `window-22` specification as this model's successor
(notes/202608211100-window-22-adopted.md), and it landed as **VG21**, registered
and fitted at `rep` on 2026-09-02 (#240). VG13 is retained as the historical
8–18-month model.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG13


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG13)
