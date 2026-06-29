# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Post-processing: posterior summaries, learning rate, kappa summaries."""

import dse_research_utils.statistics.intervals as stats_intervals
import numpy as np
import pandas as pd


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
    hdi_prob: float = 0.90,
) -> pd.DataFrame:
    """Add explicit population and new-child p/Ey columns to a summary table.

    ``posterior_summary_table`` keeps the historical ``p_*`` / ``Ey_*`` columns
    as the latent population probability and expected count. For models with
    subject random effects, the posterior-predictive ``Y_*`` columns can instead
    target a new-child distribution that integrates over subject-level
    variability. This helper makes both estimands visible without changing the
    existing column contract.
    """
    out = summary.copy()

    def add_block(prefix: str, draws: np.ndarray) -> None:
        ey = draws * n_trials
        p_hdi = np.array(
            [stats_intervals.hdi_1d(row, hdi_prob=hdi_prob) for row in draws]
        )
        ey_hdi = np.array(
            [stats_intervals.hdi_1d(row, hdi_prob=hdi_prob) for row in ey]
        )
        out[f"p_{prefix}_median"] = np.median(draws, axis=1)
        out[f"p_{prefix}_hdi_lo"] = p_hdi[:, 0]
        out[f"p_{prefix}_hdi_hi"] = p_hdi[:, 1]
        out[f"Ey_{prefix}_median"] = np.median(ey, axis=1)
        out[f"Ey_{prefix}_hdi_lo"] = ey_hdi[:, 0]
        out[f"Ey_{prefix}_hdi_hi"] = ey_hdi[:, 1]

    add_block("population", p_population)
    add_block("subject_marginal", p_subject_marginal)
    return out


def posterior_summary_table(
    X_query: np.ndarray,
    p_query: np.ndarray,
    y_query: np.ndarray,
    n_trials: int,
    hdi_prob: float = 0.90
) -> pd.DataFrame:
    """
    Build the posterior summary DataFrame at query ages.

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
    hdi_prob : float
        HDI probability mass.

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for j, a in enumerate(X_query):
        p = p_query[j, :]
        y = y_query[j, :]
        Ey = p * n_trials

        p_lo, p_hi = stats_intervals.hdi_1d(p, hdi_prob=hdi_prob)
        Ey_lo, Ey_hi = stats_intervals.hdi_1d(Ey, hdi_prob=hdi_prob)
        y_lo, y_hi = stats_intervals.hdi_1d(y, hdi_prob=hdi_prob)

        rows.append({
            "age_months": float(a),
            "p_median": float(np.median(p)),
            "p_hdi_lo": p_lo,
            "p_hdi_hi": p_hi,
            "Ey_median": float(np.median(Ey)),
            "Ey_hdi_lo": Ey_lo,
            "Ey_hdi_hi": Ey_hi,
            "Y_median": float(np.median(y)),
            "Y_hdi_lo": y_lo,
            "Y_hdi_hi": y_hi,
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
