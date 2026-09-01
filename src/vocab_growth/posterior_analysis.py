# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Post-processing: posterior summaries, learning rate, kappa summaries.

Interval columns follow the project convention (see :mod:`vocab_growth.intervals`
and ``docs/models/README.md``): the posterior median with an inner 50%
(``*_ci50_lo``/``*_ci50_hi``) and an outer 89% (``*_ci_lo``/``*_ci_hi``)
credible interval. Counts and probabilities are summarised with equal-tailed
intervals (ETI); the ``interval_kind`` argument threads the reporting config's
kind through so the tables stay consistent with the plots and diagnostics.
"""

import numpy as np
import pandas as pd

from vocab_growth import intervals


def extract_posterior(trace, name, dim):
    """Extract posterior samples for ``name``, stacking chains and draws.

    Returns an array shaped ``(len(dim), n_chain * n_draw)``. Shared by the
    multivariate engines, which previously each defined an identical private copy.
    """
    return np.array(
        trace.posterior[name]
        .stack(sample=("chain", "draw"))
        .transpose(dim, "sample")
        .values
    )


def extract_posterior_predictive(trace, name, dim):
    """Extract posterior-predictive samples for ``name`` as integer counts.

    As :func:`extract_posterior`, but reads from ``posterior_predictive`` and
    casts to ``int`` (the predictive draws are word counts).
    """
    return np.array(
        trace.posterior_predictive[name]
        .stack(sample=("chain", "draw"))
        .transpose(dim, "sample")
        .values,
        dtype=int,
    )


def extract_posterior_predictive_float(trace, name, dim):
    """Extract posterior-predictive samples for non-count deterministic values.

    This mirrors :func:`extract_posterior_predictive`, but preserves floating
    point values. It is used for predictive probabilities saved alongside
    posterior-predictive count draws.
    """
    return np.array(
        trace.posterior_predictive[name]
        .stack(sample=("chain", "draw"))
        .transpose(dim, "sample")
        .values
    )


def expand_observed_to_obs_id(trace, observed_name: str, mask_name: str):
    """One likelihood's observed counts, expanded to obs_id length with NaN gaps.

    A multi-outcome likelihood is registered only on the rows that carry its
    outcome, so ``trace.observed_data[observed_name]`` is as long as that mask's
    count, not as long as the frame. Plots and summaries index by obs_id, so the
    vector is scattered back through the stored mask and left NaN elsewhere.

    The length comparison is issue #67: if the mask stored in ``constant_data`` and
    the rows the likelihood actually saw disagree, the scatter silently misaligns
    every observation, and the figures look plausible. It was five verbatim copies
    across the bivariate and trivariate engines before it lived here.

    Not for the joint engine's two similarly-worded checks: those compare a
    *posterior-predictive* array's leading dimension against a mask count, which is
    a different pairing and a different failure.
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
        ey = draws * n_trials
        p_outer = intervals.bands(draws, ci_prob, interval_kind, sample_axis=1)
        p_inner = intervals.bands(draws, inner_ci_prob, interval_kind, sample_axis=1)
        ey_outer = intervals.bands(ey, ci_prob, interval_kind, sample_axis=1)
        ey_inner = intervals.bands(ey, inner_ci_prob, interval_kind, sample_axis=1)
        out[f"p_{prefix}_median"] = np.median(draws, axis=1)
        out[f"p_{prefix}_ci50_lo"] = p_inner[:, 0]
        out[f"p_{prefix}_ci50_hi"] = p_inner[:, 1]
        out[f"p_{prefix}_ci_lo"] = p_outer[:, 0]
        out[f"p_{prefix}_ci_hi"] = p_outer[:, 1]
        out[f"Ey_{prefix}_median"] = np.median(ey, axis=1)
        out[f"Ey_{prefix}_ci50_lo"] = ey_inner[:, 0]
        out[f"Ey_{prefix}_ci50_hi"] = ey_inner[:, 1]
        out[f"Ey_{prefix}_ci_lo"] = ey_outer[:, 0]
        out[f"Ey_{prefix}_ci_hi"] = ey_outer[:, 1]

    add_block("population", p_population)
    add_block("subject_marginal", p_subject_marginal)
    return out


def trim_reported_ages(
    df: pd.DataFrame,
    max_age_months: float | None,
    *,
    age_column: str = "age_months",
) -> pd.DataFrame:
    """Drop summary rows above the age at which a quantity's evidence stops.

    A model's ``ages_query`` grid is shared by every outcome it reports, but the
    outcomes are not observed over the same age range. In the Down syndrome pool
    the two diverge sharply: comprehension is observed on 905 rows with a 95th
    percentile of 64 months and only 15 rows at or above 72, whereas production
    is observed on 1346 rows with a 95th percentile of 78 and 51 rows at or above
    84. Reporting understood and ``q`` on the same grid as spoken therefore
    quotes a median and an interval at ages where almost nothing was measured,
    and — above the high slope anchor — where the mean is a levelled-off
    extrapolation rather than an estimate (see
    :func:`vocab_growth.models.gp_utils.trend_and_gp`).

    This is post-processing of a fitted trace, deliberately not a change to the
    query grid: the model graph and the ``query_id`` dimension are untouched, so
    trimming what is reported cannot move a posterior. That was checked directly
    — refitting VG10 across the change at a fixed seed reproduced its diagnostics
    bit-for-bit. The dropped ages remain in the trace for anyone who wants them.

    It does not follow that changing the cap is free. These tables are written
    during the fit pipeline and ``--render-only`` re-renders Quarto against the
    CSVs already on disk rather than rebuilding them, so a new cap only takes
    effect on a refit — and the cap is part of the recorded model definition, so
    output produced under a different one is correctly reported as stale.

    ``max_age_months`` of ``None`` returns the frame unchanged, which is the
    default for every model whose outcomes share one evidential range.
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

    Three distinct estimands, deliberately kept in separate column families
    because they answer different questions and behave oppositely as data
    accumulate:

    ``p_*``
        the latent population proportion;
    ``Ey_*``
        the **expected** count, ``p * n_trials`` — a credible interval on the
        mean trajectory, carrying parameter uncertainty only, which narrows
        toward zero width as more children are observed;
    ``Y_*`` and ``P(Y<=k)``
        the posterior **predictive** count for a child — parameter uncertainty
        plus between-child and occasion-level dispersion, which converges on the
        real population spread rather than on zero.

    Quoting an ``Ey_*`` interval where a ``Y_*`` interval belongs understates the
    range of individual children substantially, so the naming is load-bearing.

    ``y`` may be ``None`` for a model that carries no predictive count draws at
    this grid — the joint sign/speech engine is the case in this project. The
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

    The canonical reporting ages (:func:`posterior_summary_table` at the model
    definition's ``ages_query``) stay 6-monthly for the report; this is the
    finer-grained companion, one row per whole month of age, with the same
    columns and the same bucket thresholds.

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

    Coverage is every whole month lying **inside** the plot grid's span, which is
    the observed age range. In practice that is wider than the canonical query
    ages, not narrower: the Down syndrome pool spans 8-115 months and the
    typically-developing pool 8-25, so every canonical age has a monthly
    counterpart and the extra months run out to the tails of the data. Many of
    those tail months hold no observation at all, which is what ``n_obs`` is for.

    A month **outside** the span is excluded even where it would snap within
    :data:`MAX_MONTH_SNAP_OFFSET` — with a grid starting at 8.1, month 8 is
    dropped rather than reported from the 8.1 point. Both halves of that matter:
    the month lies below every observed age, so reporting it would extrapolate,
    and its value would be the trajectory at 8.1 wearing an "8" label. Recorded
    ages are whole months throughout this project, so no month is currently lost
    this way; the rule is what keeps a future fractional-age source from
    acquiring a silently extrapolated boundary row.

    ``X_obs``, when given, adds an ``n_obs`` column counting the observed
    administrations falling in each whole month — the check on whether a row is
    data-supported or interpolated between sparse ages.

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

    # Whole months strictly inside the grid span. ceil/floor deliberately exclude
    # a boundary month that would snap from outside — a month below X_plot.min()
    # is below every observed age, so reporting it would extrapolate and would
    # label the trajectory at (say) 8.1 months as month 8. Widening this to
    # "nearest point within MAX_MONTH_SNAP_OFFSET" would reintroduce both.
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
    """Per-draw ages for the sign-to-speech hand-over, with honest censoring.

    Arrays are ``(n_age, n_draw)`` word counts. Each milestone is found **in
    every draw and then summarised**, never read off the median curve: the
    median of crossings is not the crossing of the median, and for a peak the
    difference is not subtle — averaging curves whose peaks sit at different
    ages flattens the peak and drags it towards the middle of the grid.

    This is the single implementation the fit pipeline
    (:mod:`vocab_growth.models.common_joint_modality`) and the DS/TD comparison
    script (``scripts/compare_ds_td_expressive.py``) both use. They previously
    carried duplicate copies with two defects the VG14/VG15 statistical review
    (#238) confirmed:

    * **Crossings must be transitions.** The old rule reported the *first age at
      which a condition held*, so a draw in which speech-only exceeded sign-only
      from the first eligible age was labelled an "overtake", and a draw that
      was never majority sign-only was labelled as "falling below half". A
      crossing here now requires a genuine false-to-true transition inside the
      established region; a state already true at the youngest established age
      is counted in ``draws_censored`` instead (left-censored: the transition,
      if there was one, happened before the grid or before the child had a
      vocabulary to divide up).
    * **A grid-boundary maximum is censored, not reached.** The old rule
      reported the grid ``argmax`` as a peak even when it sat on the last
      reported age, where the true peak may lie beyond the grid. A draw whose
      maximum falls on either end of the grid now counts toward
      ``draws_censored`` and contributes no age.

    A milestone is only read once the draw's child has at least ``min_words``
    of expressive vocabulary: below that the three cells are fractions of a
    word and their ordering is arithmetic noise. The gate is on a word count
    rather than a grid-point count so it cannot silently depend on the grid
    step.

    Intervals are highest-density (:data:`vocab_growth.intervals.HDI_ESTIMANDS`
    lists ``milestone_age``/``peak_age``): milestone ages are typically skewed,
    and this is the same policy the DS/TD peak-growth ages already follow.

    Columns: ``quantity``, ``median``, ``ci_lo``, ``ci_hi`` (over the draws
    that genuinely reach the milestone), ``draws_reaching`` (fraction reaching
    it), and ``draws_censored`` (fraction where it is censored rather than
    absent — already true at the youngest established age for a crossing, or a
    maximum on a grid edge for a peak). The remainder to 1 never satisfies the
    condition at all.
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
