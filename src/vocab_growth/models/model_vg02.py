# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG02: Influence of age on words understood (A -> U) - children with Down syndrome
"""

from vocab_growth.models.common import ModelFitContext, fit_single_outcome_model
from vocab_growth.models.definitions import VG02


def fit(config: str) -> ModelFitContext:
    return fit_single_outcome_model(config, VG02)
