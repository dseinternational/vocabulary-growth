# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Apply the coverage floor to expressive-comparison panels by default.

The local band wrapper must retain the shared 0.80 floor. A narrow interval
from very few covered draws must not appear as a supported contrast.
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "compare_ds_td_expressive.py"
_SPEC = importlib.util.spec_from_file_location("compare_ds_td_expressive_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _frame() -> pd.DataFrame:
    """Five grid points, the last two below the coverage floor."""
    return pd.DataFrame({
        "age_months": [10.0, 20.0, 30.0, 40.0, 50.0],
        "coverage": [1.0, 1.0, 0.9, 0.03, 0.0000278],
        "median": [1.0, 2.0, 3.0, 4.0, 5.0],
        "ci50_lo": [0.5, 1.5, 2.5, 4.0, 5.0],
        "ci50_hi": [1.5, 2.5, 3.5, 4.0, 5.0],
        "ci_lo": [0.0, 1.0, 2.0, 4.0, 5.0],
        "ci_hi": [2.0, 3.0, 4.0, 4.0, 5.0],
    })


def test_band_defaults_to_the_module_coverage_floor():
    # The bug was a default, not a call site, so the default is what to pin.
    assert _MODULE._band.__kwdefaults__["cov"] == _MODULE.MIN_COVERAGE
    assert _MODULE.MIN_COVERAGE == 0.80


def test_band_drops_uncovered_points_by_default():
    fig, ax = plt.subplots()
    try:
        _MODULE._band(ax, _frame(), "age_months", "test", "C0")
        drawn = [line.get_xdata() for line in ax.lines if len(line.get_xdata()) > 1]
        assert drawn, "expected a median line"
        assert max(drawn[0]) == 30.0, "the 0.03 and 2.8e-05 coverage points must not be plotted"
    finally:
        plt.close(fig)


def test_band_can_still_opt_out_explicitly():
    fig, ax = plt.subplots()
    try:
        _MODULE._band(ax, _frame(), "age_months", "test", "C0", cov=0.0)
        drawn = [line.get_xdata() for line in ax.lines if len(line.get_xdata()) > 1]
        assert max(drawn[0]) == 50.0
    finally:
        plt.close(fig)
