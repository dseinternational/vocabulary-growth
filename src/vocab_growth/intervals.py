# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Default credible-interval policy for reports, plots and summary tables.

Report posterior medians with 50% inner and 89% outer intervals. Equal-tailed
intervals (ETIs) are the default; named quantities in ``HDI_ESTIMANDS`` use
highest-density intervals (HDIs). Either interval type is a probability summary
under the model, not a decision threshold.

The shared ``dse_research_utils.statistics.intervals`` module computes intervals.
This module supplies the named-quantity policy and ``age_months`` grid label.
Reporting configurations explicitly select ``"eti"`` because the shared default
is ``"hdi"``. Crossing-age outputs have separate censoring rules documented in
``docs/models/README.md``.
"""

import numpy as np
import pandas as pd
from dse_research_utils.statistics.intervals import (
    DEFAULT_CI_PROB,  # noqa: F401 - re-exported outer interval probability (89%)
    INNER_CI_PROB,  # noqa: F401 - re-exported inner interval probability (50%)
    IntervalKind,
)
from dse_research_utils.statistics.intervals import bands as _shared_bands
from dse_research_utils.statistics.intervals import interval_1d as _shared_interval_1d
from dse_research_utils.statistics.intervals import (
    summarise_bands as _shared_summarise_bands,
)

# Named quantities using HDIs because skew or boundary effects matter:
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
