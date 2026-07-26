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

    Coverage is the plot grid's span, which is the **observed** age range — so
    query ages outside it (for example the Down syndrome models' 90-month row,
    beyond the oldest observation) have no monthly counterpart by construction.
    That is deliberate: those are extrapolations, and this table does not
    manufacture them.

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
