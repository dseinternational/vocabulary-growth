# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG24: VG15 + correlated subject random effects on understood, ``q`` and
the signed ratio - children with Down syndrome.

VG15 with its three independent subject random-intercept blocks replaced by a
joint prior with a free 3x3 correlation matrix. Three added parameters; VG15 is
nested exactly at the identity. The deliverable is ``rho_sign_q``: do children
who persistently sign a larger share of what they understand also persistently
say a larger share of it? ``rho_uq`` -- the quantity VG20 estimates for VG10,
here on the trivariate frame -- and ``rho_u_sign`` come with it.

The scales are unchanged: ``tau_subj_u``, ``tau_subj_q`` and ``tau_subj_sign``
keep VG15's HalfNormal(1.5) priors and their per-child meaning, so only the
correlation is new. Because the subject shifts enter the marginal likelihoods
and not the four-cell or produced-cell Dirichlet-Multinomials (deliberately, so
the subject block cannot pull ``psi``), ``rho_sign_q`` is identified by the
children carrying both a signed and a spoken marginal rather than by every child
in the frame -- see ``docs/models/vg24/index.qmd`` for the support.

Issue #296. See ``common_joint_modality`` for the engine.
"""

from vocab_growth.models.common_joint_modality import (
    JointContext,
    fit_joint_model,
)
from vocab_growth.models.definitions import VG24


def fit(config: str) -> JointContext:
    return fit_joint_model(config, VG24)
