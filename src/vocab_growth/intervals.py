# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Project-wide credible-interval reporting policy — the single source of truth.

Every ``vocab_growth`` report, plot, and summary table routes its interval
computation through this module so the convention cannot drift between the
tables, plots, and diagnostics.

Convention
----------
Report the posterior **median** with a **50%** (inner) and an **89%** (outer)
credible interval, alongside the full posterior. Intervals are **equal-tailed
(ETI, percentile-based) by default**; a named short-list of strongly skewed or
boundary-censored estimands (:data:`HDI_ESTIMANDS`) is reported with
**highest-density intervals (HDI)** instead, where an equal-tailed interval
would misrepresent the credible region.

Why 89% (not 95%)? 89% is a deliberately non-special width: it carries no more
authority than any other, which is the point — it discourages reading an
interval as a hypothesis test. Its 5.5th/94.5th-percentile limits are also more
stable to estimate from a finite MCMC sample than the 2.5th/97.5th limits of a
95% interval at the same effective sample size. See McElreath, *Statistical
Rethinking* (2020) and Kruschke, *Bayesian Analysis Reporting Guidelines*
(Nat. Hum. Behav. 2021), and ``docs/models/README.md`` (Interval reporting
convention).

The outer mass is carried by the shared
:class:`dse_research_utils.statistics.models.reporting.ReportingConfiguration`
``ci_prob`` (0.89) and its ``interval_kind`` (``"eti"``); the inner mass is
:data:`INNER_CI_PROB` (0.50), which the shared config does not yet model.
"""

from typing import Literal

import dse_research_utils.statistics.intervals as stats_intervals
import numpy as np
import pandas as pd

IntervalKind = Literal["eti", "hdi"]

# Outer interval mass. Kept in sync with ReportingConfiguration.ci_prob and the
# shared library default; used as the fallback for free functions that do not
# carry a reporting context.
DEFAULT_CI_PROB: float = 0.89

# Inner interval mass. The shared ReportingConfiguration carries only a single
# ``ci_prob``, so the inner band lives here until a second mass is first-class
# upstream.
INNER_CI_PROB: float = 0.50

# Estimands reported with HDI rather than the ETI default, because their
# posteriors are strongly skewed or boundary-censored so an equal-tailed
# interval would misrepresent the credible region:
#   psi            - sign-speech association (ratio-like, positive, right-skewed)
#   conc / kappa   - Beta-Binomial concentration / dispersion (positive, right-skewed)
#   peak_age       - trajectory peak age (piles up against the modelled age-grid edge)
#   milestone_age  - age a target count is first reached (boundary-censored)
#   attainment_age - as milestone_age, for the cross-model contrasts
HDI_ESTIMANDS: frozenset[str] = frozenset(
    {"psi", "conc", "kappa", "peak_age", "milestone_age", "attainment_age"}
)


def interval_kind_for(name: str | None, default_kind: IntervalKind = "eti") -> IntervalKind:
    """Return the interval kind for a named estimand.

    ``"hdi"`` for the skewed short-list (:data:`HDI_ESTIMANDS`), otherwise
    ``default_kind`` (the ETI house standard, or the reporting config's kind).
    """
    return "hdi" if name in HDI_ESTIMANDS else default_kind


def interval_1d(
    x: np.ndarray | list[float],
    prob: float = DEFAULT_CI_PROB,
    kind: IntervalKind = "eti",
) -> tuple[float, float]:
    """Credible interval of a 1-D sample array, NaN-aware.

    ``kind="eti"`` returns the equal-tailed (percentile) interval; ``"hdi"``
    the highest-density interval. Both delegate to the shared
    :mod:`dse_research_utils.statistics.intervals` primitives.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    if kind == "hdi":
        return stats_intervals.hdi_1d(x, hdi_prob=prob)
    return stats_intervals.eti_1d(x, eti_prob=prob)


def bands(
    samples: np.ndarray,
    prob: float = DEFAULT_CI_PROB,
    kind: IntervalKind = "eti",
    *,
    sample_axis: int = 1,
) -> np.ndarray:
    """Per-grid credible interval of a 2-D sample array.

    ``samples`` has one axis of draws (``sample_axis``) and one grid axis; the
    return is ``(n_grid, 2)`` of ``(lo, hi)`` along the grid, NaN-aware. Use for
    trajectory / query-age bands where the interval is computed independently at
    each grid point.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"bands expects a 2-D array, got shape {arr.shape}")
    grid_axis = 1 - sample_axis
    n_grid = arr.shape[grid_axis]
    out = np.empty((n_grid, 2), dtype=float)
    for i in range(n_grid):
        draws = arr[:, i] if sample_axis == 0 else arr[i, :]
        out[i, 0], out[i, 1] = interval_1d(draws, prob, kind)
    return out


def summarise(
    samples: np.ndarray,
    grid: np.ndarray,
    *,
    name: str | None = None,
    kind: IntervalKind | None = None,
    outer: float = DEFAULT_CI_PROB,
    inner: float = INNER_CI_PROB,
    sample_axis: int = 1,
    grid_name: str = "age_months",
) -> pd.DataFrame:
    """Median + inner + outer credible interval per grid point, as a tidy frame.

    Columns: ``grid_name``, ``median``, ``ci50_lo``/``ci50_hi`` (inner),
    ``ci_lo``/``ci_hi`` (outer), ``interval_kind``. The kind defaults to
    :func:`interval_kind_for` applied to ``name`` (so callers can just pass the
    estimand name and get ETI, or HDI for the skewed short-list); pass ``kind``
    to override.
    """
    resolved_kind = kind if kind is not None else interval_kind_for(name)
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None] if sample_axis == 0 else arr[None, :]
    inner_band = bands(arr, inner, resolved_kind, sample_axis=sample_axis)
    outer_band = bands(arr, outer, resolved_kind, sample_axis=sample_axis)
    median = np.nanmedian(arr, axis=sample_axis)
    return pd.DataFrame(
        {
            grid_name: np.asarray(grid, dtype=float),
            "median": median,
            "ci50_lo": inner_band[:, 0],
            "ci50_hi": inner_band[:, 1],
            "ci_lo": outer_band[:, 0],
            "ci_hi": outer_band[:, 1],
            "interval_kind": resolved_kind,
        }
    )
