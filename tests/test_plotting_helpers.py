# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest

from vocab_growth.plotting import _maybe_savgol, _resolve_savgol_window_length


@pytest.mark.parametrize("n", [5, 7, 15, 21, 100])
def test_resolve_window_is_valid(n):
    polyorder = 3
    wl = _resolve_savgol_window_length(n, window_length=None, polyorder=polyorder)
    assert wl % 2 == 1          # odd
    assert wl <= n              # fits the data
    assert wl > polyorder       # valid for savgol


def test_resolve_window_raises_when_polyorder_too_high_for_n():
    # n=4 cannot accommodate a cubic (needs an odd window > 3, i.e. >= 5 > n).
    with pytest.raises(ValueError):
        _resolve_savgol_window_length(4, window_length=None, polyorder=3)


def test_resolve_window_even_is_made_odd():
    wl = _resolve_savgol_window_length(50, window_length=20, polyorder=3)
    assert wl % 2 == 1
    assert wl <= 20


def test_resolve_window_below_polyorder_is_bumped():
    # Requesting a window <= polyorder must be raised to a valid odd value.
    wl = _resolve_savgol_window_length(50, window_length=2, polyorder=3)
    assert wl > 3
    assert wl % 2 == 1


def test_resolve_window_too_few_points_raises():
    with pytest.raises(ValueError):
        _resolve_savgol_window_length(2, window_length=None, polyorder=3)


def test_maybe_savgol_passthrough_when_disabled():
    y = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    out = _maybe_savgol(y, smooth=False, window_length=None, polyorder=2)
    np.testing.assert_array_equal(out, y)
    # The returned array is always float (the smoothing path returns floats too).
    assert out.dtype == float
