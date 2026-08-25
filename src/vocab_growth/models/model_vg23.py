# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG23: VG13 + correlated subject random effects on (understood, q) (issue #229).

VG13 gives each typically-developing child two deviations — one on the understood
trajectory, one on the production ratio q — and draws them independently. VG23
replaces that with a joint Normal carrying one free correlation, ``rho_uq``. VG13
is nested exactly at ``rho_uq = 0``, so the pair is a one-factor contrast, and it
is the same change VG20 already makes to VG10 on the Down syndrome side.

It exists for an identification reason rather than for the correlation itself.
The typically-developing pool averages 1.16 administrations per child and only
15.1% of VG13's children have a repeat visit, so the between-child / within-child
split rests on the Beta-Binomial's functional form rather than on replication —
and the between-child scale is recovered low in 9 of 9 replicates across three
models (#225). Every VG13 administration nonetheless yields *two* counts from one
child on one day. A child's persistent ability moves both while the observation
noise is assumed independent across them, so their agreement identifies the child
effect on 100% of rows rather than on the 15% with a second visit.

The confound is real and runs the other way from the one it is meant to help
with: both counts come from one questionnaire completed by one parent, so shared
reporter tendency is indistinguishable from shared child ability and biases
``rho_uq`` — and through it the child scales — upward, where the variance
partition biases them downward. The two bracket the truth; neither settles it.
Read any estimate from this model with that stated.

See #229 option 3, #225, and notes/202608050900-td-hierarchical-geometry.md.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG23


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG23)
