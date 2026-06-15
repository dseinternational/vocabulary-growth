# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG15: Joint sign/speech model (issue #49 Option 3) - children with Down
syndrome.

Estimates the within-understood sign-speech association (a scalar Plackett odds
ratio, identified from the uk_02 four-cell cross-tab) and adds study random
intercepts, replacing VG14's independence-based p_any upper bound with a
data-identified total expressive vocabulary. See ``common_joint_modality`` for
the engine.
"""

from vocab_growth.models.common_joint_modality import (
    JointContext,
    fit_joint_model,
)
from vocab_growth.models.definitions import VG15


def fit(config: str) -> JointContext:
    return fit_joint_model(config, VG15)
