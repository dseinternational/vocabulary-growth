# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG15: Joint sign/speech model (issue #49 Option 3) - children with Down
syndrome.

Estimates the within-understood sign-speech association: a Plackett odds ratio
``psi`` with a study-level random intercept, identified from four
cross-tabulation sources — the uk_02, uk_07 and es_01 four-cell
within-understood cross-tabs plus nz_01's three-cell within-produced cross-tab.
The population ``psi`` is a shrunk centre over sources that disagree; the
per-study values are the primary read. Study *and* subject random intercepts
sit on all three latent trajectories (understood, speak ratio q, sign ratio r),
each zero-summed over the studies its likelihood actually informs, together
with VG10's stabilisation package (tighter q-GP amplitude + a per-draw GP
anchor at the reference age) and an estimated (not fixed) signed-tent peak
position. The q age anchors remain the shared weakly informative DS-joint
priors. It replaces VG14's independence-based p_any upper bound with a
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
