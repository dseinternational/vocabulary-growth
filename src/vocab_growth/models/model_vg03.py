# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG03: Influence of age on words spoken (A -> S) - typically developing children
"""

from vocab_growth.models.common import ModelFitContext, fit_single_outcome_model
from vocab_growth.models.definitions import VG03


def fit(config: str, *, render: bool = False) -> ModelFitContext:
    return fit_single_outcome_model(config, VG03, render=render)
