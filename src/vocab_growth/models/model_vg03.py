# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG03 models words spoken by age in typically developing children."""

from vocab_growth.models.common import ModelFitContext, fit_single_outcome_model
from vocab_growth.models.definitions import VG03


def fit(config: str) -> ModelFitContext:
    return fit_single_outcome_model(config, VG03)
