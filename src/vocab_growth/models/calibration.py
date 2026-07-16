# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Quantitative posterior-predictive calibration summaries."""

import os

import numpy as np
import pandas as pd
import xarray as xr


def predictive_calibration_table(
    observed: np.ndarray,
    predictive: np.ndarray,
    ages: np.ndarray,
    *,
    interval_probs: tuple[float, ...] = (0.50, 0.80, 0.90),
    age_band_months: int = 12,
) -> pd.DataFrame:
    """Summarise predictive coverage and discrete probability-integral transforms.

    ``predictive`` must have shape ``(observation, posterior_sample)``. The
    mid-PIT uses half the replicated probability mass equal to the observation,
    which is a deterministic diagnostic suitable for a discrete count outcome.
    """
    observed = np.asarray(observed, dtype=float)
    predictive = np.asarray(predictive, dtype=float)
    ages = np.asarray(ages, dtype=float)
    if predictive.ndim != 2:
        raise ValueError("predictive must have shape (observation, sample).")
    if observed.shape != (predictive.shape[0],) or ages.shape != observed.shape:
        raise ValueError("observed, predictive, and ages are not row-aligned.")
    if age_band_months <= 0:
        raise ValueError("age_band_months must be positive.")
    if not interval_probs or any(not 0 < prob < 1 for prob in interval_probs):
        raise ValueError("interval_probs must contain probabilities between 0 and 1.")

    age_starts = np.floor(ages / age_band_months).astype(int) * age_band_months
    groups: list[tuple[str, np.ndarray]] = [("all", np.ones(observed.size, dtype=bool))]
    for start in sorted(np.unique(age_starts)):
        groups.append(
            (
                f"[{start}, {start + age_band_months})",
                age_starts == start,
            )
        )

    rows: list[dict[str, float | int | str]] = []
    for age_band, mask in groups:
        y = observed[mask]
        y_rep = predictive[mask]
        pit = np.mean(y_rep < y[:, None], axis=1) + 0.5 * np.mean(
            y_rep == y[:, None], axis=1
        )
        shared = {
            "age_band_months": age_band,
            "n_observations": int(mask.sum()),
            "observed_mean": float(y.mean()),
            "predictive_mean": float(y_rep.mean()),
            "mean_error": float(y_rep.mean() - y.mean()),
            "observed_zero_rate": float(np.mean(y == 0)),
            "predictive_zero_rate": float(np.mean(y_rep == 0)),
            "mid_pit_mean": float(pit.mean()),
            "mid_pit_variance": float(pit.var()),
            "mid_pit_extreme_rate": float(np.mean((pit < 0.05) | (pit > 0.95))),
        }
        for prob in interval_probs:
            tail = (1 - prob) / 2
            lower = np.quantile(y_rep, tail, axis=1)
            upper = np.quantile(y_rep, 1 - tail, axis=1)
            rows.append(
                {
                    **shared,
                    "interval_probability": prob,
                    "empirical_coverage": float(np.mean((y >= lower) & (y <= upper))),
                    "mean_interval_width": float(np.mean(upper - lower)),
                }
            )
    return pd.DataFrame(rows)


def write_trace_calibration(
    trace: xr.DataTree,
    analysis_df: pd.DataFrame,
    output_dir: str,
    outcomes: tuple[tuple[str, str, str | None], ...],
) -> pd.DataFrame:
    """Write calibration rows for posterior-predictive variables in a trace.

    Each outcome is ``(label, posterior_predictive_variable, mask_variable)``.
    A ``None`` mask means every prepared analysis row is represented.
    """
    tables: list[pd.DataFrame] = []
    for label, variable, mask_variable in outcomes:
        if variable not in trace.posterior_predictive:
            continue
        if mask_variable is None:
            mask = np.ones(len(analysis_df), dtype=bool)
        else:
            mask = np.asarray(trace.constant_data[mask_variable].values, dtype=bool)

        replicated = trace.posterior_predictive[variable]
        observation_dims = [
            dim for dim in replicated.dims if dim not in {"chain", "draw"}
        ]
        if len(observation_dims) != 1:
            raise ValueError(
                f"{variable} must have exactly one observation dimension; "
                f"found {observation_dims}."
            )
        observation_dim = observation_dims[0]
        predictive = np.asarray(
            replicated.stack(sample=("chain", "draw"))
            .transpose(observation_dim, "sample")
            .values
        )
        observed = np.asarray(trace.observed_data[variable].values)
        ages = np.asarray(analysis_df.loc[mask, "age"], dtype=float)
        table = predictive_calibration_table(observed, predictive, ages)
        table.insert(0, "outcome", label)
        tables.append(table)

    result = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    result.to_csv(
        os.path.join(output_dir, "posterior_predictive_calibration.csv"),
        index=False,
    )
    return result
