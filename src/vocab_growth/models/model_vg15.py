# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG15 jointly models signing and speech in children with Down syndrome.

The Plackett odds ratio ``psi`` describes their association within understood
words. It uses four-cell counts from uk_02, uk_07 and es_01 and three-cell
counts conditional on production from nz_01. Study-specific estimates matter
because these sources differ in measurement.

All three latent trajectories have study and child intercepts. Child
intercepts enter the marginal likelihoods, not the cell likelihoods. Study
offsets sum to zero over the studies contributing to each likelihood.
The signing trend has an estimated middle reference age.

Total expressive vocabulary uses the fitted overlap. VG14 assumes independence.
See ``common_joint_modality`` for the engine and ``definitions.py`` for priors.
"""

from vocab_growth.models.common_joint_modality import (
    JointContext,
    fit_joint_model,
)
from vocab_growth.models.definitions import VG15


def fit(config: str) -> JointContext:
    return fit_joint_model(config, VG15)
