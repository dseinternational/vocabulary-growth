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

The structure was chosen before any of it was written. Gate 1 fitted three
candidates to the fitted residuals by maximum likelihood, against a known
per-observation binomial sampling variance, and a random slope beat a constant
intercept by ``2 x delta logL = 36.05`` on spoken — surviving restriction to the
334 children with repeated spoken measures at 20.81, so it is genuine
within-child drift rather than cross-sectional widening. An AR(1) transient
collapsed to zero persistence on both outcomes, so the missing structure is drift
and not an autocorrelated child process. Proposal A1 is the same model with
``rho01`` pinned to 1 — one deviate scaled by an age function is a rank-one
covariance — and freeing it costs 6.28 on 1 df on the repeats-only production
fit, so it is free here.

Comprehension is expected to behave differently from production, and the model
is built to let it: the slope is worth 27.09 across all 610 children but 0.82
across the 253 with repeats, so that widening is cross-sectional. ``tau1_u``'s
posterior interval is the answer to "does comprehension drift within a child?"
rather than something the specification decides in advance.

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
