# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Post-processing: posterior summaries, learning rate, kappa summaries."""

import dse_research_utils.statistics.intervals as stats_intervals
import numpy as np
import pandas as pd


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
