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

The interval *mechanics* — the kind dispatcher, the per-grid bands, and the
tidy two-band summary — live in :mod:`dse_research_utils.statistics.intervals`
(v0.12.0) together with the shared masses (:data:`DEFAULT_CI_PROB` 0.89,
:data:`INNER_CI_PROB` 0.50), re-exported here. What stays local is the
*policy*: which named estimands report with HDI (:data:`HDI_ESTIMANDS` /
:func:`interval_kind_for`) and the ``age_months`` grid naming. This project
passes ``interval_kind="eti"`` explicitly on its
:class:`~dse_research_utils.statistics.models.reporting.ReportingConfiguration`
(the shared default is ``"hdi"``).
"""

import numpy as np
import pandas as pd
from dse_research_utils.statistics.intervals import (
    DEFAULT_CI_PROB,  # noqa: F401 — re-exported: the outer 89% house mass
    INNER_CI_PROB,  # noqa: F401 — re-exported: the inner 50% house mass
    IntervalKind,
)
from dse_research_utils.statistics.intervals import bands as _shared_bands
from dse_research_utils.statistics.intervals import interval_1d as _shared_interval_1d
from dse_research_utils.statistics.intervals import (
    summarise_bands as _shared_summarise_bands,
)

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

    Delegates to :func:`dse_research_utils.statistics.intervals.interval_1d`.
    """
    return _shared_interval_1d(x, prob, kind)


def bands(
    samples: np.ndarray,
    prob: float = DEFAULT_CI_PROB,
    kind: IntervalKind = "eti",
    *,
    sample_axis: int = 1,
) -> np.ndarray:
    """Per-grid credible interval of a 2-D sample array, ``(n_grid, 2)``.

    Delegates to :func:`dse_research_utils.statistics.intervals.bands`.
    """
    return _shared_bands(samples, prob, kind, sample_axis=sample_axis)


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
    to override. Delegates to
    :func:`dse_research_utils.statistics.intervals.summarise_bands`.
    """
    resolved_kind = kind if kind is not None else interval_kind_for(name)
    return _shared_summarise_bands(
        samples,
        grid,
        kind=resolved_kind,
        outer=outer,
        inner=inner,
        sample_axis=sample_axis,
        grid_name=grid_name,
    )
