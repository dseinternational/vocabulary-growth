# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG22: VG10 + a low-rank factor over the four child effects (Down syndrome).

VG19 and VG20 each hold half of a structure neither can express alone. VG19 gives
each child an intercept *and* a rate on each outcome, but in two independent
blocks, so a child's comprehension standing says nothing about their production
ratio. VG20 correlates the two intercepts, but holds both rates at zero, so the
between-child spread is frozen at one width for every age. Their union is a 4x4
covariance over ``(b0u, b1u, b0q, b1q)``.

**The 4x4 was gated before this model was written, and it came back twice
negative.** The element that motivated it — do children who gain comprehension
faster also convert faster, ``corr(b1u, b1q)`` — is the *weakest* of the six
cross terms, and weakest of all on the children with repeated measures, where a
within-child rate coupling would have to show if it existed. And the 4x4 is not
identified by these data: its maximum-likelihood correlation matrix is singular,
a rank-3 fit reaches the identical likelihood, and rank 2 costs 2.60 on 2 df.

So this is not a 4x4. It is a factor form, ``b = L z`` with ``z`` of dimension
``k`` and ``L`` a free ``(4, k)`` loading matrix, which is positive semi-definite
by construction and needs no positive-definiteness constraint written into the
graph. ``rank`` is a definition field because the data cannot choose it: 2 and 3
are both defensible and 4 is excluded, so the honest settlement is the
sensitivity family at 1, 2 and 3, which differ by one column of ``L``.

**What the gate did find**, and no current model estimates, is an asymmetric
coupling nobody proposed: a child's comprehension **level** predicts their
production-ratio **rate**, at nearly twice the strength of the correlation VG20
fits. A factor form carries that term without being asked to.

One caveat belongs on the model rather than only in the note. Residual maximum
likelihood sits on the singularity boundary and a Bayesian fit will not, because
the prior regularises away from it. The finding is not that the fourth dimension
is zero but that **the data carry almost no information about it**, so at higher
ranks the prior supplies most of what is reported.

See notes/202608221000-four-by-four-gate1.md.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG22


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG22)
