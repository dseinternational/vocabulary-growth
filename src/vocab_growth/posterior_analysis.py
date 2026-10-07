# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Post-processing for posterior count, proportion and signing-milestone summaries.

Interval columns follow the project convention (see :mod:`vocab_growth.intervals`
and ``docs/models/README.md``): the posterior median with an inner 50%
(``*_ci50_lo``/``*_ci50_hi``) and an outer 89% (``*_ci_lo``/``*_ci_hi``)
credible interval. Counts and probabilities are summarised with equal-tailed
intervals (ETI); the ``interval_kind`` argument threads the reporting config's
kind through so the tables stay consistent with the plots and diagnostics.
"""

import numpy as np
import pandas as pd
from dse_research_utils.statistics.samples import sample_matrix

from vocab_growth import intervals


def _extract_samples(array, dim, *, dtype=None):
    """Copy labelled samples, retaining positional extraction for old traces.

    Unlabelled archived arrays retain their prior behaviour. This fallback
    does not establish observation identity and does not invent coordinates.
    """
    if any(name not in array.indexes for name in ("chain", "draw", dim)):
        return np.array(
            array.stack(sample=("chain", "draw")).transpose(dim, "sample").values,
            dtype=dtype,
        )
    matrix = sample_matrix(array, sample_dims=("chain", "draw"), observation_dims=(dim,))
    return np.array(matrix.values, dtype=dtype, copy=True)


def extract_posterior(trace, name, dim):
    """Copy posterior samples into observation-by-sample order."""
    return _extract_samples(trace.posterior[name], dim)


def extract_posterior_predictive(trace, name, dim):
    """Copy posterior-predictive word counts with the existing integer cast."""
    return _extract_samples(trace.posterior_predictive[name], dim, dtype=int)


def extract_posterior_predictive_float(trace, name, dim):
    """Copy posterior-predictive non-count values without an integer cast."""
    return _extract_samples(trace.posterior_predictive[name], dim)


def expand_observed_to_obs_id(trace, observed_name: str, mask_name: str):
    """One likelihood's observed counts, expanded to obs_id length with NaN gaps.

    A multi-outcome likelihood is registered only on the rows that carry its
    outcome, so ``trace.observed_data[observed_name]`` is as long as that mask's
    count, not as long as the frame. Plots and summaries index by obs_id, so the
    vector is scattered back through the stored mask and left NaN elsewhere.

    The marked-row count must match the observed array length before scattering.
    This checks observed counts; predictive-array alignment is checked separately.
    """
    mask = np.array(trace.constant_data[mask_name].values, dtype=bool)
    observed = np.array(trace.observed_data[observed_name].values, dtype=float)
    if int(mask.sum()) != observed.shape[0]:
        raise ValueError(
            f"{mask_name} count ({int(mask.sum())}) does not match observed "
            f"{observed_name} length ({observed.shape[0]}); stored mask and "
            "likelihood rows are misaligned (issue #67)."
        )
    expanded = np.full(len(mask), np.nan)
    expanded[mask] = observed
    return expanded


def add_probability_estimand_columns(
    summary: pd.DataFrame,
    p_population: np.ndarray,
    p_subject_marginal: np.ndarray,
    *,
    n_trials: int,
    ci_prob: float = intervals.DEFAULT_CI_PROB,
    inner_ci_prob: float = intervals.INNER_CI_PROB,
    interval_kind: intervals.IntervalKind = "eti",
) -> pd.DataFrame:
    """Add explicit population and new-child p/Ey columns to a summary table.

    ``posterior_summary_table`` keeps the historical ``p_*`` / ``Ey_*`` columns
    as the latent population probability and expected count. For models with
    subject random effects, the posterior-predictive ``Y_*`` columns can instead
    target a new-child distribution that integrates over subject-level
    variability. This helper makes both estimands visible without changing the
    existing column contract.

    Each block emits the median with an inner (``*_ci50_*``) and outer
    (``*_ci_*``) interval at the project convention (:mod:`vocab_growth.intervals`).
    """
    out = summary.copy()

    def add_block(prefix: str, draws: np.ndarray) -> None:
        _interval_columns(
            out,
            f"p_{prefix}",
            draws,
            ci_prob=ci_prob,
            inner_ci_prob=inner_ci_prob,
            interval_kind=interval_kind,
        )
        _interval_columns(
            out,
            f"Ey_{prefix}",
            draws * n_trials,
            ci_prob=ci_prob,
            inner_ci_prob=inner_ci_prob,
            interval_kind=interval_kind,
        )

    add_block("population", p_population)
    add_block("subject_marginal", p_subject_marginal)
    return out


def _interval_columns(
    out: pd.DataFrame,
    prefix: str,
    draws: np.ndarray,
    *,
    ci_prob: float,
    inner_ci_prob: float,
    interval_kind: intervals.IntervalKind,
) -> None:
    """One estimand's median with its inner and outer interval, in place."""
    outer = intervals.bands(draws, ci_prob, interval_kind, sample_axis=1)
    inner = intervals.bands(draws, inner_ci_prob, interval_kind, sample_axis=1)
    out[f"{prefix}_median"] = np.median(draws, axis=1)
    out[f"{prefix}_ci50_lo"] = inner[:, 0]
    out[f"{prefix}_ci50_hi"] = inner[:, 1]
    out[f"{prefix}_ci_lo"] = outer[:, 0]
    out[f"{prefix}_ci_hi"] = outer[:, 1]


def add_rate_estimand_columns(
    summary: pd.DataFrame,
    population: np.ndarray,
    subject_marginal: np.ndarray,
    *,
    name: str = "q",
    ci_prob: float = intervals.DEFAULT_CI_PROB,
    inner_ci_prob: float = intervals.INNER_CI_PROB,
    interval_kind: intervals.IntervalKind = "eti",
) -> pd.DataFrame:
    """Add explicit population and new-child columns for a bounded rate.

    ``q`` is the spoken share of understood words. This helper emits no ``Ey_*``
    block because ``q * n_trials`` represents a spoken count only for a child
    who understands the whole inventory.

    The existing ``{name}_*`` columns describe the trajectory with random effects
    set to zero. The subject-marginal share needs its own draws: a ratio of
    marginal medians is not the median of the ratio.
    """
    out = summary.copy()
    for prefix, draws in (
        ("population", population),
        ("subject_marginal", subject_marginal),
    ):
        _interval_columns(
            out,
            f"{name}_{prefix}",
            draws,
            ci_prob=ci_prob,
            inner_ci_prob=inner_ci_prob,
            interval_kind=interval_kind,
        )
    return out


def trim_reported_ages(
    df: pd.DataFrame,
    max_age_months: float | None,
    *,
    age_column: str = "age_months",
) -> pd.DataFrame:
    """Drop summary rows above a quantity's reporting cap.

    Trimming changes neither the model graph nor the posterior. Caps can differ
    between outcomes and do not guarantee adequate evidence at all retained ages.
    ``--render-only`` reads saved CSVs, so it cannot apply a new cap to them.
    Definition compatibility and regeneration of summaries must be handled by
    the caller. ``None`` returns the frame unchanged.
    """
    if max_age_months is None:
        return df
    return df[df[age_column] <= max_age_months].reset_index(drop=True)


def posterior_summary_table(
    X_query: np.ndarray,
    p_query: np.ndarray,
    y_query: np.ndarray,
    n_trials: int,
    ci_prob: float = intervals.DEFAULT_CI_PROB,
    inner_ci_prob: float = intervals.INNER_CI_PROB,
    interval_kind: intervals.IntervalKind = "eti",
) -> pd.DataFrame:
    """
    Build the posterior summary DataFrame at query ages.

    Each estimand (latent proportion ``p``, expected count ``Ey``, new-child
    predictive count ``Y``) is summarised by its median with an inner
    (``*_ci50_*``) and outer (``*_ci_*``) credible interval at the project
    convention (:mod:`vocab_growth.intervals`).

    Parameters
    ----------
    X_query : np.ndarray
        Query ages.
    p_query : np.ndarray
        Posterior draws of the latent mean probability/proportion at each query age.
    y_query : np.ndarray
        Posterior predictive word counts for each query age.
    n_trials : int
        Maximum score.
    ci_prob : float
        Outer interval probability mass (default 0.89).
    inner_ci_prob : float
        Inner interval probability mass (default 0.50).
    interval_kind : {"eti", "hdi"}
        Interval convention; counts/proportions default to equal-tailed.

    Returns
    -------
    pd.DataFrame
    """
    rows = [
        summary_row(
            float(a),
            p_query[j, :],
            y_query[j, :],
            n_trials=n_trials,
            ci_prob=ci_prob,
            inner_ci_prob=inner_ci_prob,
            interval_kind=interval_kind,
        )
        for j, a in enumerate(X_query)
    ]

    return pd.DataFrame(rows).sort_values("age_months").reset_index(drop=True)


COUNT_BUCKET_THRESHOLDS: tuple[int, ...] = (5, 10, 25, 50, 100, 200, 400)
"""Cumulative predictive-count thresholds reported as ``P(Y<=k)`` columns.

Shared by :func:`posterior_summary_table` and :func:`monthly_summary_table` so
the canonical query-age table and the whole-month table cannot drift apart in
either the thresholds or the column names.
"""


def summary_row(
    age_months: float,
    p: np.ndarray,
    y: np.ndarray | None,
    *,
    n_trials: int,
    ci_prob: float = intervals.DEFAULT_CI_PROB,
    inner_ci_prob: float = intervals.INNER_CI_PROB,
    interval_kind: intervals.IntervalKind = "eti",
) -> dict:
    """Summarise one age's posterior draws into the standard summary columns.

    The input draws determine the prediction target. Column families distinguish
    latent proportions, expected counts and predictive counts:

    ``p_*``
        the latent proportion supplied in ``p``;
    ``Ey_*``
        the expected count, ``p * n_trials``. Its interval excludes the count
        variation introduced by the likelihood;
    ``Y_*`` and ``P(Y<=k)``
        the predictive count supplied in ``y``, including the random effects
        and count variation used by the caller.

    ``y`` may be ``None`` for a model that carries no predictive count draws at
    this grid, as in the joint sign/speech engine. The
    ``Y_*`` and ``P(Y<=k)`` columns are then absent rather than zero-filled, so a
    reader cannot mistake a missing estimand for a computed one.
    """
    p = np.asarray(p, dtype=float)
    Ey = p * n_trials

    p_lo, p_hi = intervals.interval_1d(p, ci_prob, interval_kind)
    p_lo50, p_hi50 = intervals.interval_1d(p, inner_ci_prob, interval_kind)
    Ey_lo, Ey_hi = intervals.interval_1d(Ey, ci_prob, interval_kind)
    Ey_lo50, Ey_hi50 = intervals.interval_1d(Ey, inner_ci_prob, interval_kind)

    row = {
        "age_months": float(age_months),
        "p_median": float(np.median(p)),
        "p_ci50_lo": p_lo50,
        "p_ci50_hi": p_hi50,
        "p_ci_lo": p_lo,
        "p_ci_hi": p_hi,
        "Ey_median": float(np.median(Ey)),
        "Ey_ci50_lo": Ey_lo50,
        "Ey_ci50_hi": Ey_hi50,
        "Ey_ci_lo": Ey_lo,
        "Ey_ci_hi": Ey_hi,
    }

    if y is None:
        return row

    y = np.asarray(y, dtype=float)
    y_lo, y_hi = intervals.interval_1d(y, ci_prob, interval_kind)
    y_lo50, y_hi50 = intervals.interval_1d(y, inner_ci_prob, interval_kind)
    row.update({
        "Y_median": float(np.median(y)),
        "Y_ci50_lo": y_lo50,
        "Y_ci50_hi": y_hi50,
        "Y_ci_lo": y_lo,
        "Y_ci_hi": y_hi,
        "P(Y=0)": float((y == 0).mean()),
    })
    for k in COUNT_BUCKET_THRESHOLDS:
        row[f"P(Y<={k})"] = float((y <= k).mean())
    row[f"P(Y>{COUNT_BUCKET_THRESHOLDS[-1]})"] = float(
        (y > COUNT_BUCKET_THRESHOLDS[-1]).mean()
    )
    return row


# A whole-month row is read off the nearest plot-grid age. With the default
# n_plot = 500 the grid step is about 0.2 months, so the snap is a few days;
# this bound rejects a grid too coarse to carry monthly reporting rather than
# emitting rows whose stated age is wrong.
MAX_MONTH_SNAP_OFFSET: float = 0.25


def monthly_summary_table(
    X_plot: np.ndarray,
    p_plot: np.ndarray,
    y_plot: np.ndarray | None,
    n_trials: int,
    *,
    X_obs: np.ndarray | pd.Series | None = None,
    ci_prob: float = intervals.DEFAULT_CI_PROB,
    inner_ci_prob: float = intervals.INNER_CI_PROB,
    interval_kind: intervals.IntervalKind = "eti",
) -> pd.DataFrame:
    """Build the summary table at every whole month, from the plot grid.

    The companion uses the canonical table's columns and count thresholds.
    Canonical ``ages_query`` steps are generally six months for DS models and
    three months for TD models.

    It reads the *plot* grid rather than adding query ages to the model, so it is
    pure post-processing of a fitted trace: no change to the model graph, the
    HSGP domain, or the ``query_id`` dimension the report and comparisons
    consume. Each whole month takes the nearest plot-grid point, and the two
    provenance columns record which:

    ``grid_age_months``
        the plot-grid age actually summarised;
    ``grid_offset_months``
        ``grid_age_months - age_months``, bounded by
        :data:`MAX_MONTH_SNAP_OFFSET`.

    Coverage is every whole month inside the supplied plot grid. Months outside
    that span are excluded even if their nearest point would meet the offset
    limit. These companions can extend beyond canonical reporting caps and
    include months with no observations; they do not extend the published range.

    ``X_obs`` adds ``n_obs`` counts by rounding recorded ages to the nearest
    whole month. Read these counts alongside the two grid-provenance columns.

    Raises
    ------
    ValueError
        If the plot grid is too coarse for whole-month resolution, naming the
        offending offset, so a reduced ``n_plot`` cannot silently mislabel ages.
    """
    X_plot = np.asarray(X_plot, dtype=float).reshape(-1)
    p_plot = np.asarray(p_plot, dtype=float)
    y_plot = None if y_plot is None else np.asarray(y_plot, dtype=float)

    if p_plot.shape[0] != X_plot.shape[0]:
        raise ValueError(
            "p_plot must have one row per plot age "
            f"(X_plot {X_plot.shape[0]}, p_plot {p_plot.shape[0]})."
        )
    if y_plot is not None and y_plot.shape[0] != X_plot.shape[0]:
        raise ValueError(
            "y_plot must have one row per plot age "
            f"(X_plot {X_plot.shape[0]}, y_plot {y_plot.shape[0]})."
        )

    # Include grid endpoints when they are whole months; exclude months outside
    # the span even if a nearest-point snap would meet the offset limit.
    months = np.arange(
        int(np.ceil(X_plot.min())), int(np.floor(X_plot.max())) + 1, dtype=int
    )
    if months.size == 0:
        raise ValueError(
            f"The plot grid spans no whole month (ages {X_plot.min():.2f}-{X_plot.max():.2f})."
        )

    nearest = np.abs(months[:, None] - X_plot[None, :]).argmin(axis=1)
    offsets = X_plot[nearest] - months
    worst = float(np.max(np.abs(offsets)))
    if worst > MAX_MONTH_SNAP_OFFSET:
        raise ValueError(
            "The plot grid is too coarse for whole-month reporting: nearest-point "
            f"offset reaches {worst:.3f} months against a {MAX_MONTH_SNAP_OFFSET} "
            f"limit (grid step {np.diff(X_plot).max():.3f} months over "
            f"{X_plot.min():.1f}-{X_plot.max():.1f}). Raise n_plot."
        )

    if X_obs is not None:
        observed = np.asarray(X_obs, dtype=float).reshape(-1)
        observed = observed[np.isfinite(observed)]
        n_obs = [int(np.sum(np.rint(observed) == month)) for month in months]
    else:
        n_obs = None

    rows = []
    for position, (month, index) in enumerate(zip(months, nearest, strict=True)):
        row = summary_row(
            float(month),
            p_plot[index, :],
            None if y_plot is None else y_plot[index, :],
            n_trials=n_trials,
            ci_prob=ci_prob,
            inner_ci_prob=inner_ci_prob,
            interval_kind=interval_kind,
        )
        row["age_months"] = int(month)
        if n_obs is not None:
            row["n_obs"] = n_obs[position]
        row["grid_age_months"] = float(X_plot[index])
        row["grid_offset_months"] = float(offsets[position])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("age_months").reset_index(drop=True)


def signing_milestone_table(
    ages: np.ndarray,
    sign_only: np.ndarray,
    both: np.ndarray,
    speak_only: np.ndarray,
    *,
    ci_prob: float = intervals.DEFAULT_CI_PROB,
    min_words: float = 1.0,
) -> pd.DataFrame:
    """Summarise per-draw signing peaks and sign-speech crossings.

    Arrays contain counts with shape ``(n_age, n_draw)``. Find each milestone
    within each draw before summarising; a median curve's crossing or peak need
    not equal the median of the draw-specific ages. These are features of the
    supplied trajectories, not observed developmental events for a child.

    Crossings require a false-to-true transition whose endpoints both have at
    least ``min_words`` of total expressive vocabulary. A condition already
    true at the first eligible age is censored unless it later becomes false
    and crosses again. Peak calculations use the full grid; a maximum at either
    edge is censored because its age is not resolved within the grid.

    Medians and highest-density intervals use only finite milestone values.
    ``draws_reaching`` and ``draws_censored`` report their respective fractions
    among all draws. The remainder has no qualifying transition. The same
    finite-peak rule applies to the peak word-count row.
    """
    ages = np.asarray(ages, dtype=float)
    sign_only = np.asarray(sign_only, dtype=float)
    both = np.asarray(both, dtype=float)
    speak_only = np.asarray(speak_only, dtype=float)
    total = np.maximum(sign_only + both + speak_only, 1e-9)
    established = total >= min_words

    def _transition_ages(condition: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(per-draw first false->true transition age or NaN, initially-true mask)."""
        n_age, n_draw = condition.shape
        est = established
        # A transition needs both its endpoints established, and the condition
        # false at the earlier one.
        trans = np.zeros_like(condition, dtype=bool)
        trans[1:] = condition[1:] & ~condition[:-1] & est[1:] & est[:-1]
        has_trans = trans.any(axis=0)
        idx = trans.argmax(axis=0)
        out = np.full(n_draw, np.nan)
        out[has_trans] = ages[idx[has_trans]]
        # Initially true: the condition already holds at the draw's first
        # established age. Only meaningful for draws with an established region.
        any_est = est.any(axis=0)
        first_est = est.argmax(axis=0)
        cols = np.arange(n_draw)
        initially = np.zeros(n_draw, dtype=bool)
        initially[any_est] = condition[first_est[any_est], cols[any_est]]
        # A draw with a genuine later transition is reported as reaching it even
        # if the state also held initially (true -> false -> true again).
        return out, initially & ~has_trans

    def _peak_ages(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(per-draw interior peak age or NaN, peak words or NaN, censored mask)."""
        idx = np.argmax(values, axis=0)
        censored = (idx == 0) | (idx == len(ages) - 1)
        peak_age = np.where(censored, np.nan, ages[idx])
        cols = np.arange(values.shape[1])
        peak_words = np.where(censored, np.nan, values[idx, cols])
        return peak_age, peak_words, censored

    peak_age, peak_words, peak_censored = _peak_ages(sign_only)
    below_half, below_half_initial = _transition_ages((sign_only / total) < 0.5)
    overtake, overtake_initial = _transition_ages(speak_only >= sign_only)

    rows = [
        ("sign_only_peak_age", peak_age, peak_censored),
        ("sign_only_peak_words", peak_words, peak_censored),
        ("sign_only_share_below_half_age", below_half, below_half_initial),
        ("speech_only_overtakes_sign_only_age", overtake, overtake_initial),
    ]
    out = []
    for name, draws, censored in rows:
        ok = np.isfinite(draws)
        vals = draws[ok]
        lo, hi = intervals.interval_1d(vals, ci_prob, "hdi")
        out.append({
            "quantity": name,
            "median": float(np.median(vals)) if vals.size else np.nan,
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "draws_reaching": float(ok.mean()),
            "draws_censored": float(np.asarray(censored, dtype=bool).mean()),
        })
    return pd.DataFrame(out)
