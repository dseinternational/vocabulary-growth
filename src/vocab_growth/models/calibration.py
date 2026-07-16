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
    observation_chunk_size: int = 256,
) -> pd.DataFrame:
    """Summarise predictive coverage and discrete probability-integral transforms.

    ``predictive`` must have shape ``(observation, posterior_sample)``. The
    mid-PIT uses half the replicated probability mass equal to the observation,
    which is a deterministic diagnostic suitable for a discrete count outcome.
    """
    observed = np.asarray(observed, dtype=float)
    predictive = np.asarray(predictive)
    ages = np.asarray(ages, dtype=float)
    if predictive.ndim != 2:
        raise ValueError("predictive must have shape (observation, sample).")
    if observed.shape != (predictive.shape[0],) or ages.shape != observed.shape:
        raise ValueError("observed, predictive, and ages are not row-aligned.")
    if age_band_months <= 0:
        raise ValueError("age_band_months must be positive.")
    if not interval_probs or any(not 0 < prob < 1 for prob in interval_probs):
        raise ValueError("interval_probs must contain probabilities between 0 and 1.")
    if observation_chunk_size <= 0:
        raise ValueError("observation_chunk_size must be positive.")
    if observed.size == 0 or predictive.shape[1] == 0:
        raise ValueError("calibration requires observations and predictive samples.")

    n_observations = observed.size
    predictive_mean = np.empty(n_observations, dtype=float)
    predictive_zero_rate = np.empty(n_observations, dtype=float)
    pit = np.empty(n_observations, dtype=float)
    coverage_by_prob = {
        prob: np.empty(n_observations, dtype=bool) for prob in interval_probs
    }
    width_by_prob = {
        prob: np.empty(n_observations, dtype=float) for prob in interval_probs
    }

    for start in range(0, n_observations, observation_chunk_size):
        stop = min(start + observation_chunk_size, n_observations)
        y = observed[start:stop]
        y_rep = predictive[start:stop]
        predictive_mean[start:stop] = y_rep.mean(axis=1)
        predictive_zero_rate[start:stop] = np.mean(y_rep == 0, axis=1)
        pit[start:stop] = np.mean(y_rep < y[:, None], axis=1) + 0.5 * np.mean(
            y_rep == y[:, None], axis=1
        )
        for prob in interval_probs:
            tail = (1 - prob) / 2
            lower = np.quantile(y_rep, tail, axis=1)
            upper = np.quantile(y_rep, 1 - tail, axis=1)
            coverage_by_prob[prob][start:stop] = (y >= lower) & (y <= upper)
            width_by_prob[prob][start:stop] = upper - lower

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
        pit_group = pit[mask]
        shared = {
            "age_band_months": age_band,
            "n_observations": int(mask.sum()),
            "observed_mean": float(y.mean()),
            "predictive_mean": float(predictive_mean[mask].mean()),
            "mean_error": float(predictive_mean[mask].mean() - y.mean()),
            "observed_zero_rate": float(np.mean(y == 0)),
            "predictive_zero_rate": float(predictive_zero_rate[mask].mean()),
            "mid_pit_mean": float(pit_group.mean()),
            "mid_pit_variance": float(pit_group.var()),
            "mid_pit_extreme_rate": float(
                np.mean((pit_group < 0.05) | (pit_group > 0.95))
            ),
        }
        for prob in interval_probs:
            rows.append(
                {
                    **shared,
                    "interval_probability": prob,
                    "empirical_coverage": float(coverage_by_prob[prob][mask].mean()),
                    "mean_interval_width": float(width_by_prob[prob][mask].mean()),
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
        predictive = (
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
