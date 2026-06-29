# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""PyMC-graph build helpers shared across the model engines.

This module holds helpers that emit PyMC ops (as opposed to the pure-NumPy
helpers in :mod:`vocab_growth.models.build_utils`). It is deliberately kept
separate so the pure module stays ``pymc``-free.

Currently it provides only :func:`make_kappa_of_z`, the age-varying dispersion
closure factory. The factory takes random variables the caller has already
created (in the caller's own order) and returns a closure whose body is
byte-identical to the ``kappa_of_z`` closures previously inlined in every engine,
so it moves no random-variable creation and cannot change the model graph.

The trend + HSGP construction (``trend_and_gp`` / ``intercept_and_gp``), which is
still duplicated across the engines, is intended to be consolidated here in a
follow-up; it is excluded from the current change because it moves the sole
RNG-bearing call (``hsgp.prior()``) and requires empirical fit-equivalence proof.
"""

from __future__ import annotations

import pymc as pm


def make_kappa_of_z(kappa_min, a_kappa, b_kappa):
    """Return the age-varying dispersion closure ``z -> kappa_min + exp(a + b z)``.

    ``kappa_min``, ``a_kappa`` and ``b_kappa`` are PyMC variables the caller has
    already created (including any ``b_kappa = -b_kappa_mag`` deterministic). The
    returned closure is evaluated later at the standardised ages, emitting the
    same ops the inlined closures did.
    """

    def kappa_of_z(z):
        return kappa_min + pm.math.exp(a_kappa + b_kappa * z)

    return kappa_of_z
