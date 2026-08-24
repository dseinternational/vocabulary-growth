# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pure-NumPy build helpers shared across the model engines.

These functions factor out blocks that were previously copy-and-pasted, byte for
byte, into the ``build`` step of every engine (``common.py`` and its
copy-and-extend siblings): age standardisation, plot/query grid construction,
length-scale validation, and slope-anchor z-scoring.

They are deliberately free of any ``pymc`` import. Every function performs only
deterministic NumPy arithmetic and returns plain Python scalars / NumPy arrays
that are subsequently fed into ``pm.Data(...)`` and the model trend. Because the
operations (and their order) are identical to the inlined code they replace, the
values handed to the PyMC graph are bit-identical, so the graph and the sampling
RNG are unaffected. Graph-building helpers that create PyMC random variables live
separately in :mod:`vocab_growth.models.gp_utils`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def standardize_ages(X_obs: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Return ``(mean, std, X_obs_z)`` for the observed ages.

    ``std`` uses ``ddof=1`` (sample standard deviation), matching the convention
    every engine relies on for its z-scores. Raises ``ValueError`` if the
    standard deviation is non-finite or non-positive (degenerate ages).

    The age *median* is intentionally not computed here: it is reporting-only and
    used by a single engine's build table, so it stays in the caller.
    """
    X_obs_mean = float(np.mean(X_obs))
    X_obs_std = float(np.std(X_obs, ddof=1))

    if not np.isfinite(X_obs_std) or X_obs_std <= 0:
        raise ValueError("Age standard deviation must be positive.")

    X_obs_z = (X_obs - X_obs_mean) / X_obs_std
    return X_obs_mean, X_obs_std, X_obs_z


@dataclass(frozen=True)
class AgeGrids:
    """Standardised age grids for observed, plot, and query points.

    ``X_all_z`` is the vertical stack ``[obs, plot, query]`` (plus a single anchor
    row when ``use_gp_anchor`` is requested). The ``i_*`` fields are
    ``(start, stop)`` slice bounds into ``X_all_z``; ``i_anchor`` is the integer
    index of the anchor row (or ``None`` when not anchored).
    """

    X_plot: np.ndarray
    X_plot_z: np.ndarray
    X_query: np.ndarray
    X_query_z: np.ndarray
    X_gp_domain_z: np.ndarray
    """Two endpoints used only to size the HSGP approximation domain."""
    X_all_z: np.ndarray
    n_plot: int
    n_query: int
    n_all: int
    i_obs: tuple[int, int]
    i_plot: tuple[int, int]
    i_query: tuple[int, int]
    i_anchor: int | None
    anchor_age_months: float | None


def construct_age_grids(
    X_obs: np.ndarray,
    X_obs_z: np.ndarray,
    *,
    X_obs_mean: float,
    X_obs_std: float,
    n_plot: int,
    ages_query,
    slope_anchors,
    use_gp_anchor: bool = False,
    gp_anchor_age_months: float | None = None,
    gp_domain_months: tuple[float, float] | None = None,
) -> AgeGrids:
    """Build the plot/query grids and the stacked standardised grid ``X_all_z``.

    ``X_obs`` is in original months (used for the ``linspace`` plot grid bounds);
    ``X_obs_z`` is the already-standardised observed grid (used in the stack).

    When ``use_gp_anchor`` is True a single anchor row is appended to ``X_all_z``
    at ``gp_anchor_age_months`` (defaulting to the midpoint of ``slope_anchors``
    when not given), matching the reference-age anchoring used by the random-
    effects and joint engines.

    HSGP basis sizing is deliberately separated from the evaluation grid. If
    ``gp_domain_months`` is omitted, the observed age range is the domain; an
    explicit pair can instead fix it in the model definition. Plot, query, and
    anchor points must lie inside that domain. Therefore changing a reporting
    query cannot silently change the HSGP approximation itself.
    """
    X_plot = np.linspace(X_obs.min(), X_obs.max(), n_plot).reshape(-1, 1)
    X_plot_z = (X_plot - X_obs_mean) / X_obs_std

    X_query = np.array(ages_query).reshape(-1, 1)
    X_query_z = (X_query - X_obs_mean) / X_obs_std

    n = X_obs_z.shape[0]
    n_plot_actual = X_plot_z.shape[0]
    n_query = X_query_z.shape[0]

    if use_gp_anchor:
        if gp_anchor_age_months is not None:
            anchor_age_months: float | None = float(gp_anchor_age_months)
        else:
            anchor_age_months = (
                float(slope_anchors[0]) + float(slope_anchors[1])
            ) / 2.0
        X_anchor_z = (
            np.array([[anchor_age_months]], dtype=float) - X_obs_mean
        ) / X_obs_std
        X_all_z = np.vstack([X_obs_z, X_plot_z, X_query_z, X_anchor_z])
        i_anchor: int | None = n + n_plot_actual + n_query
    else:
        anchor_age_months = None
        X_all_z = np.vstack([X_obs_z, X_plot_z, X_query_z])
        i_anchor = None

    if gp_domain_months is None:
        domain_low = float(np.min(X_obs))
        domain_high = float(np.max(X_obs))
    else:
        if len(gp_domain_months) != 2:
            raise ValueError("gp_domain_months must contain exactly two ages.")
        domain_low = float(gp_domain_months[0])
        domain_high = float(gp_domain_months[1])
        if not np.isfinite(domain_low) or not np.isfinite(domain_high):
            raise ValueError("gp_domain_months must contain finite ages.")
        if domain_high <= domain_low:
            raise ValueError("gp_domain_months must be ordered (low, high).")

    evaluation_ages = [float(np.min(X_obs)), float(np.max(X_obs))]
    evaluation_ages.extend(float(age) for age in np.asarray(X_query).ravel())
    if anchor_age_months is not None:
        evaluation_ages.append(anchor_age_months)
    if min(evaluation_ages) < domain_low or max(evaluation_ages) > domain_high:
        raise ValueError(
            "Observed, query, and GP-anchor ages must lie inside gp_domain_months "
            f"[{domain_low}, {domain_high}]."
        )
    X_gp_domain_z = (
        np.array([[domain_low], [domain_high]], dtype=float) - X_obs_mean
    ) / X_obs_std

    n_all = X_all_z.shape[0]

    return AgeGrids(
        X_plot=X_plot,
        X_plot_z=X_plot_z,
        X_query=X_query,
        X_query_z=X_query_z,
        X_gp_domain_z=X_gp_domain_z,
        X_all_z=X_all_z,
        n_plot=n_plot_actual,
        n_query=n_query,
        n_all=n_all,
        i_obs=(0, n),
        i_plot=(n, n + n_plot_actual),
        i_query=(n + n_plot_actual, n + n_plot_actual + n_query),
        i_anchor=i_anchor,
        anchor_age_months=anchor_age_months,
    )


def require_valid_counts(values: np.ndarray, name: str, n_trials: int) -> None:
    """Fail loudly on a non-finite, fractional or out-of-range count column.

    The nested spoken likelihood gets these three checks from
    :func:`vocab_growth.models.likelihood_utils.nested_outcome_spec`; this is
    the same contract for a count column an engine casts and bounds itself.
    VG13 used to cast ``understood`` to ``int`` *before* any check, so a
    fractional value would have been silently truncated and an out-of-range
    one would have surfaced only as a likelihood failure (#240).

    ``values`` must already be free of NaN (callers drop or mask missing
    counts before casting).

    The non-finite and integrality checks are :func:`require_integral_counts`'s,
    which names the offending values; this adds the range check on top.
    """
    values = np.asarray(values, dtype=float)
    require_integral_counts(values, name)
    if not np.all((values >= 0) & (values <= n_trials)):
        raise ValueError(f"{name} must lie between 0 and n_trials.")


def require_integral_counts(values: np.ndarray, name: str) -> None:
    """Fail loudly if a count column carries non-finite or fractional values.

    Every engine casts its outcome columns to ``int`` for the Beta-Binomial
    likelihood, and NumPy's cast truncates toward zero silently — a fractional
    count (an averaged or hand-edited source cell, a bad merge) would be floored
    without a trace, and an infinity would cast to an arbitrary integer that a
    later bounds check could only misdiagnose. All current source counts are
    finite and integral, so this guard costs nothing until the day it fires
    (#234, #236).

    ``values`` must already be free of NaN (callers drop or mask missing counts
    before casting); a NaN that does reach this guard is reported as non-finite
    rather than truncated.
    """
    values = np.asarray(values, dtype=float)
    non_finite = ~np.isfinite(values)
    if non_finite.any():
        bad = np.flatnonzero(non_finite)
        examples = ", ".join(f"{values[i]:g}" for i in bad[:5])
        raise ValueError(
            f"{name} contains {bad.size} non-finite count(s) "
            f"(e.g. {examples}); counts must be finite whole numbers."
        )
    fractional = values != np.floor(values)
    if fractional.any():
        bad = np.flatnonzero(fractional)
        examples = ", ".join(f"{values[i]:g}" for i in bad[:5])
        raise ValueError(
            f"{name} contains {bad.size} non-integral count(s) "
            f"(e.g. {examples}); counts must be whole numbers — a silent cast "
            "would truncate them."
        )


def validate_ell_bounds(ell_months_range) -> tuple[float, float]:
    """Return ``(ell_low_months, ell_high_months)`` as floats after validation.

    Raises ``ValueError`` if either bound is non-positive or if the range is not
    strictly increasing — the two checks previously inlined in every engine.
    """
    ell_low_months = float(ell_months_range[0])
    ell_high_months = float(ell_months_range[1])

    if ell_low_months <= 0 or ell_high_months <= 0:
        raise ValueError("Length-scale bounds must be positive (in months).")
    if ell_high_months <= ell_low_months:
        raise ValueError("ell_months_range must be (low, high) with high > low.")

    return ell_low_months, ell_high_months


def standardize_anchor_ages(
    anchor_ages,
    *,
    X_obs_mean: float,
    X_obs_std: float,
) -> tuple[float, float]:
    """Z-score a pair of reference ages against the observed-age standardisation.

    Shared by every anchored parameterisation — the mean trajectory's slope
    anchors and the dispersion curve's kappa anchors — so an age stated in a model
    definition always reaches the graph through the same conversion.
    """
    age_a, age_b = float(anchor_ages[0]), float(anchor_ages[1])
    return (age_a - X_obs_mean) / X_obs_std, (age_b - X_obs_mean) / X_obs_std


def slope_anchor_logit_coeffs(
    slope_anchors,
    *,
    X_obs_mean: float,
    X_obs_std: float,
) -> tuple[float, float]:
    """Return the z-scored slope-anchor ages ``(slope_age_a_z, slope_age_b_z)``.

    Only the pure z-scoring is shared. The logit-difference computation of the
    slope/intercept coefficients operates on PyMC random variables and stays
    inline in each engine's model context.
    """
    return standardize_anchor_ages(
        slope_anchors, X_obs_mean=X_obs_mean, X_obs_std=X_obs_std
    )
