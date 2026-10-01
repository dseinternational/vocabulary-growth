# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Crossing-age summaries that keep draws outside the supported age window."""

import numpy as np


def summarise_crossing_ages(values, cap, floor, probability=0.89):
    """Equal-tailed order-statistic interval, retaining both censoring states.

    A negative infinity means the target was reached before the grid. Positive
    infinity means it was not reached by the cap. Neither is discarded when
    computing quantiles. NaNs do not encode either state and are refused.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.isnan(values).any():
        raise ValueError("Crossing draws must be non-empty and use infinities for censoring.")
    values = np.where(values > cap, np.inf, np.where(values < floor, -np.inf, values))
    tail = (1 - probability) / 2
    lo, median, hi = np.quantile(values, [tail, 0.5, 1 - tail], method="inverted_cdf")
    before, beyond = np.mean(values < floor), np.mean(values > cap)
    return {
        "median": float(median), "lo": float(lo), "hi": float(hi),
        "frac_beyond_window": float(beyond), "frac_before_window": float(before),
        "censored": bool(before > 0 or beyond > 0),
        "before_floor": bool(median < floor), "beyond_cap": bool(median > cap),
        "interval_type": "equal-tailed", "interval_probability": probability,
    }
