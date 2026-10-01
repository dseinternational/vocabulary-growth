# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG19: VG10 + a child random slope on understood and on q.

VG08-VG10 give each child a **constant** offset from the population trajectory,
so three distinct quantities have to live in two parameters: persistent
between-child differences, occasion-to-occasion movement, and drift — a child
systematically pulling away from, or toward, the population curve as they grow.
VG19 separates the third by giving each child a rate as well as an offset. The
model of record is nested exactly at ``tau1 = 0``.

Historical fits to adjusted residuals motivated a child slope. Their numerical
contrasts depend on the observation-error benchmark and residual adjustment.
They do not establish a unique mechanism for changing child ranks. Correlation
one gives an affine scale in age, whereas A1 uses an exponential scale and a
different dispersion restriction. A1 is therefore not this model's rank-one
special case. Posterior rate scales, predictive checks and sensitivity are
needed to assess the registered model; a positive-scale equal-tailed interval
above zero is not itself evidence against zero variation.

**Gated against VG10, not VG20**, on the study owner's decision of 2026-08-21.
VG19 and VG20 are parallel refinements of the same parent and are not composable
as written: VG20 estimates one correlation between the two outcomes' constant
offsets, VG19 estimates two different ones between each outcome's own intercept
and slope, and their union is a 4x4 covariance whose most interesting element —
whether children who gain comprehension faster also convert faster — is
estimated by neither. Gating against VG10 keeps the comparison one-factor.

See notes/202608141900-child-slope-implementation-plan.md and
notes/202608211500-vg19-registration.md.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG19


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG19)
