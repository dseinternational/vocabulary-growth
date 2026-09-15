# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG25: VG24 + a sign -> speech within-child cross-lag - children with
Down syndrome.

One added coefficient, ``beta_sign_lag``: a child's prior-wave **signed share of
comprehension**, relative to their own persistent signing standing, shifts the
logit of their current production ratio ``q``. VG24 is nested exactly at
``beta_sign_lag = 0``. The lag source is assigned per complete ``(subject, age)``
administration wave, as VG16's is (issue #242).

It extends VG24 rather than VG15 because the coefficient is only interpretable
with the correlated child block present: ``rho_sign_q`` carries the *persistent*
sign-speech association between children, which leaves the lag measuring the
prospective, occasion-level quantity it is named for. Reported beside VG24's
``rho_sign_q`` and the two-wave residual regression of
``notes/202608160930-early-signing-and-later-speech.md``, those are three
different quantities and the report's job is to say which is which.

The term enters the spoken marginal only, which is where VG15's subject shifts
are confined. It was registered in the cross-tab compositions as well, and the
first ``rep`` fit was bimodal because the within-child baseline carries each
child's estimated signing intercept into them; see
``notes/202609151930-vg25-lag-out-of-the-cells.md``. The ``sign-lag-in-cells``
sensitivity puts the term in the compositions under the population baseline.
See ``docs/models/vg25/index.qmd`` for the support and ``common_joint_modality``
for the engine.

Issue #297.
"""

from vocab_growth.models.common_joint_modality import (
    JointContext,
    fit_joint_model,
)
from vocab_growth.models.definitions import VG25


def fit(config: str) -> JointContext:
    return fit_joint_model(config, VG25)
