# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG20: VG10 + correlated subject random effects on (understood, q) (issue #224).

VG10 gives each child two deviations — one on the understood trajectory, one on
the production ratio q — and draws them independently. VG20 replaces that with a
joint Normal carrying one free correlation, ``rho_uq``. The model of record is
nested exactly at ``rho_uq = 0``.

The parameter answers a question the study asks directly: do children who
understand more words than expected for their age also say a larger fraction of
what they understand? It is also a correction rather than only an addition.
Because a child's spoken vocabulary is the product of what they understand and
the share they say, ``log p_S = log p_U + log q`` gains ``2 Cov`` — so assuming
independence understates how much children with Down syndrome differ from one
another in speech, and that assumption is the one asymmetry in the DS-vs-TD
between-child spoken contrast that biases the number rather than merely
complicating it (VG11's single spoken intercept absorbs the covariance).

Two independent measurements already say the correlation is positive and small:
VG16's realised subject intercepts still correlate at +0.135 [0.087, 0.180] with
its cross-lag fitted, and VG10's own fitted deviations at +0.152 [0.105, 0.195]
with no cross-lag at all. Both are shrunk toward zero by the independence prior
they are measured under, so both are lower bounds. VG20 estimates the quantity
instead of inferring it from a residual, on all 767 children rather than the 250
with a prior-wave comprehension source, and free of the attenuation the cross-lag
proxy carries (reliability 0.53).

See notes/202608151120-vg16-crosslag-quantified.md and
notes/202608151140-cross-lag-not-for-models-of-record.md.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG20


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG20)
