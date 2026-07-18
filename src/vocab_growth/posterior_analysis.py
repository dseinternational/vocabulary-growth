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
    rows = []
    for j, a in enumerate(X_query):
        p = p_query[j, :]
        y = y_query[j, :]
        Ey = p * n_trials

        p_lo, p_hi = intervals.interval_1d(p, ci_prob, interval_kind)
        p_lo50, p_hi50 = intervals.interval_1d(p, inner_ci_prob, interval_kind)
        Ey_lo, Ey_hi = intervals.interval_1d(Ey, ci_prob, interval_kind)
        Ey_lo50, Ey_hi50 = intervals.interval_1d(Ey, inner_ci_prob, interval_kind)
        y_lo, y_hi = intervals.interval_1d(y, ci_prob, interval_kind)
        y_lo50, y_hi50 = intervals.interval_1d(y, inner_ci_prob, interval_kind)

        rows.append({
            "age_months": float(a),
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
            "Y_median": float(np.median(y)),
            "Y_ci50_lo": y_lo50,
            "Y_ci50_hi": y_hi50,
            "Y_ci_lo": y_lo,
            "Y_ci_hi": y_hi,
            "P(Y=0)": float((y == 0).mean()),
            "P(Y<=5)": float((y <= 5).mean()),
            "P(Y<=10)": float((y <= 10).mean()),
            "P(Y<=25)": float((y <= 25).mean()),
            "P(Y<=50)": float((y <= 50).mean()),
            "P(Y<=100)": float((y <= 100).mean()),
            "P(Y<=200)": float((y <= 200).mean()),
            "P(Y<=400)": float((y <= 400).mean()),
            "P(Y>400)": float((y > 400).mean()),
        })

    return pd.DataFrame(rows).sort_values("age_months").reset_index(drop=True)
