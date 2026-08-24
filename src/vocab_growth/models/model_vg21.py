# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG21: Joint words understood + spoken (TD, 8-22 months) with dataset-level
study random intercepts.

VG13 with its age window widened from 18 to 22 months. That is the whole change,
and it exists because VG13 runs out of matched comprehension at about 221
understood words, which is short of where the Down-syndrome-versus-typically-
developing production-ratio contrast becomes interesting. This window reaches
328.

**Why 22 and not 25.** Both windows were fitted at VG13's own sampling settings
on 2026-08-20, so the window is the only thing separating them from VG13 and
from each other. Convergence does not discriminate — all three clear the hard
gate with zero divergences, and both extensions are better on energy BFMI than
VG13 is. The Oxford CDI's 418-item ceiling does. The share of administrations
sitting within 10% of their form's ceiling runs at 1-5% below 19 months and 7-8%
at 19-22, then jumps to 20.2%, 27.7% and 36.1% at 23, 24 and 25. Ceiling
compression holds *understood* down while spoken keeps rising, which inflates
``q = S/U``, so a contaminated window reads high — and ``window-25`` does, by
0.098 at 328 understood words. The contamination is confined to the months where
the instrument runs out of items, which is what makes it diagnostic rather than
merely a difference.

**What the wider window changes substantively.** Under this window the DS/TD gap
in ``q`` closes by 300 understood words (Δq = -0.00, P(TD>DS) = 0.49) where
``window-25`` keeps it open at +0.09. That closure was gated against its own
priors under a decision rule fixed in writing before the fit: a variant with both
high slope anchors widened in the direction that would reopen the gap moved
P(TD>DS) by at most 0.0085 anywhere on the grid.

Two weaknesses are inherited rather than resolved. The high-anchor priors come
from in-sample medians rather than published norms, because no CDI comprehension
norm exists above 18 months; and the ``kappa`` magnitudes are VG12's, moved to
anchor ages inside the window but not recalibrated for it.

See notes/202608211100-window-22-adopted.md and
notes/202608211545-window-22-prior-gate-passed.md.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG21


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG21)
