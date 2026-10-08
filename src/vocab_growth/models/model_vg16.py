# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG16 adds an earlier-comprehension predictor to a joint vocabulary model.

It uses child intercepts and the GP anchors used by VG10. The lag source is
assigned per complete child-age wave. The definition selects
a population or within-child baseline. The registered default uses a
population baseline, so its coefficient mixes persistent differences between
children with change within a child. It describes an association and does not
establish that earlier comprehension causes later speech.

See ``docs/models/README.md`` for the limits on reporting this model.
"""

from vocab_growth.models.common_bivariate_re import (
    BivariateREContext,
    fit_bivariate_re_model,
)
from vocab_growth.models.definitions import VG16


def fit(config: str) -> BivariateREContext:
    return fit_bivariate_re_model(config, VG16)
