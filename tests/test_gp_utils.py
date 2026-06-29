# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the kappa dispersion-closure factory in ``models.gp_utils``.

The factory returns the closure ``z -> kappa_min + exp(a_kappa + b_kappa * z)``;
these tests pin that closed form. Evaluating with constant inputs is sufficient
(and matches the per-engine usage, where the same expression is built from random
variables the caller has already created).
"""

import numpy as np

from vocab_growth.models.gp_utils import make_kappa_of_z


def test_make_kappa_of_z_at_zero():
    f = make_kappa_of_z(2.0, 0.5, -0.3)
    # at z = 0: kappa_min + exp(a_kappa)
    assert np.isclose(float(f(0.0).eval()), 2.0 + np.exp(0.5))


def test_make_kappa_of_z_closed_form():
    kappa_min, a_kappa, b_kappa, z = 1.5, 0.2, -0.4, 0.7
    f = make_kappa_of_z(kappa_min, a_kappa, b_kappa)
    expected = kappa_min + np.exp(a_kappa + b_kappa * z)
    assert np.isclose(float(f(z).eval()), expected)


def test_make_kappa_of_z_monotone_for_negative_b():
    f = make_kappa_of_z(1.0, 0.0, -0.5)
    lo = float(f(-1.0).eval())
    hi = float(f(1.0).eval())
    # negative b_kappa => kappa decreases as standardised age increases
    assert lo > hi
