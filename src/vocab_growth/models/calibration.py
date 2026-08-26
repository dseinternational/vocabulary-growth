# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Quantitative posterior-predictive calibration summaries.

Every fit writes one of these tables. It answers a different question from the
convergence gate: the gate establishes that the sampler characterised the
specified posterior, while this establishes whether the fitted model's replicated
outcomes look like the observed ones — how often a predictive interval contains
the observation it is predicting, and whether the observations sit uniformly
within their predictive distributions.

The reporting helpers at the foot of this module turn a written table into the
form the report and the per-model dashboards present, so the interpretation
cannot drift between them.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from vocab_growth import intervals

# Nominal levels tabulated by default. The inner and outer levels are the
# project's reporting convention (vocab_growth.intervals), so predictive coverage
# is reported at the same widths as every credible interval in the report rather
# than at an unrelated round number. The middle level is an intermediate
# reference point.
DEFAULT_INTERVAL_PROBS: tuple[float, ...] = (
    intervals.INNER_CI_PROB,
    0.80,
    intervals.DEFAULT_CI_PROB,
)

# Variance of a standard uniform. For a *continuous* outcome this is the value a
# perfectly calibrated probability integral transform has. For a discrete count
# the deterministic mid-PIT is NOT uniform even under perfect calibration: its
# variance is (1 - E[sum_k p_k^3]) / 12, strictly below 1/12, with the shortfall
# largest where the predictive mass concentrates on few values (young-age bands
# full of zero counts). The tables therefore carry a model-implied
# ``expected_mid_pit_variance`` column computed from the predictive draws
# themselves, and the reporting compares against that, not against 1/12 (#234).
# The constant is kept as the continuous limit both references approach.
UNIFORM_PIT_VARIANCE: float = 1.0 / 12.0


def predictive_calibration_table(
    observed: np.ndarray,
    predictive: np.ndarray,
    ages: np.ndarray,
    *,
    interval_probs: tuple[float, ...] = DEFAULT_INTERVAL_PROBS,
    age_band_months: int = 12,
    observation_chunk_size: int = 256,
) -> pd.DataFrame:
    """Summarise predictive coverage and discrete probability-integral transforms.

    ``predictive`` must have shape ``(observation, posterior_sample)``. The
    mid-PIT uses half the replicated probability mass equal to the observation,
    which is a deterministic diagnostic suitable for a discrete count outcome —
    but its calibrated reference is *not* the standard uniform. Under perfect
    calibration a discrete mid-PIT has mean 1/2 and variance
    ``(1 - sum_k p_k^3) / 12`` per observation (Bernoulli(0.5): values 0.25/0.75,
    variance 0.0625, not 1/12), and an equal-tailed discrete interval covers at
    least its nominal mass because quantiles land on atoms. Each row therefore
    also carries the model-implied calibrated references, estimated from the
    predictive draws: ``expected_mid_pit_variance`` (the mean over observations
    of the per-observation calibrated variance) and ``expected_coverage`` (the
    mean over observations of the predictive mass inside the tabulated
    interval). Compare the empirical columns against these, not against 1/12 or
    the nominal level (#234).
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
    expected_pit_variance = np.empty(n_observations, dtype=float)
    coverage_by_prob = {
        prob: np.empty(n_observations, dtype=bool) for prob in interval_probs
    }
    expected_coverage_by_prob = {
        prob: np.empty(n_observations, dtype=float) for prob in interval_probs
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
        for offset, row in enumerate(y_rep):
            _, counts = np.unique(row, return_counts=True)
            mass = counts / row.size
            # Var(mid-PIT | calibrated) = (1 - sum p_k^3) / 12 for this
            # observation's predictive distribution.
            expected_pit_variance[start + offset] = (
                1.0 - float(np.sum(mass**3))
            ) / 12.0
        for prob in interval_probs:
            tail = (1 - prob) / 2
            lower = np.quantile(y_rep, tail, axis=1)
            upper = np.quantile(y_rep, 1 - tail, axis=1)
            coverage_by_prob[prob][start:stop] = (y >= lower) & (y <= upper)
            # Predictive mass inside the tabulated interval: what coverage a
            # perfectly calibrated model would show, discreteness included.
            expected_coverage_by_prob[prob][start:stop] = np.mean(
                (y_rep >= lower[:, None]) & (y_rep <= upper[:, None]), axis=1
            )
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
            # Pooled calibrated reference: every per-observation mid-PIT has
            # mean 1/2 under calibration, so the pooled variance is the mean of
            # the per-observation variances.
            "expected_mid_pit_variance": float(
                expected_pit_variance[mask].mean()
            ),
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
                    "expected_coverage": float(
                        expected_coverage_by_prob[prob][mask].mean()
                    ),
                    "mean_interval_width": float(width_by_prob[prob][mask].mean()),
                }
            )
    return pd.DataFrame(rows)


def write_trace_calibration(
    trace: xr.DataTree,
    analysis_df: pd.DataFrame,
    output_dir: str,
    outcomes: tuple[tuple[str, str, str | None], ...],
    strata: dict[str, tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """Write calibration rows for posterior-predictive variables in a trace.

    Each outcome is ``(label, posterior_predictive_variable, mask_variable)``.
    A ``None`` mask means every prepared analysis row is represented.

    ``strata`` splits an outcome into two additional labelled sub-tables, keyed
    by outcome label and giving ``(constant_data variable, true label, false
    label)``. It exists for the spoken outcome's two likelihood branches (issue
    #236): the conditional and marginal-fallback rows are fitted by different
    distributions and differ systematically in age and study, so a pooled
    calibration row can be well behaved while one branch is not. The split is
    read from the fitted trace's own ``constant_data``, so it costs no refit and
    applies to fits made before it existed. A stratum with no rows is skipped
    rather than written empty -- which is what the ``paired_only`` treatment
    leaves behind.
    """
    strata = strata or {}
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

        stratum = strata.get(label)
        if stratum is None:
            continue
        stratum_variable, true_label, false_label = stratum
        if stratum_variable not in trace.constant_data:
            continue
        selector = np.asarray(
            trace.constant_data[stratum_variable].values, dtype=bool
        )
        if selector.shape != observed.shape:
            raise ValueError(
                f"{stratum_variable} has {selector.shape} rows but {variable} has "
                f"{observed.shape}; a stratum must be defined over the outcome's "
                "own observed rows."
            )
        for rows, sub_label in ((selector, true_label), (~selector, false_label)):
            if not rows.any():
                continue
            sub = predictive_calibration_table(
                observed[rows], predictive[rows], ages[rows]
            )
            sub.insert(0, "outcome", sub_label)
            tables.append(sub)

    result = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    result.to_csv(
        os.path.join(output_dir, "posterior_predictive_calibration.csv"),
        index=False,
    )
    return result


# ==========================================================================
# Reporting helpers
# ==========================================================================
#
# A written table carries every tabulated nominal level and every age band. The
# report and the per-model dashboards present one level at a time, so these
# helpers resolve the level and label the columns in one place — the numbers in
# the report and the numbers on a model's page are then the same numbers,
# selected the same way.

# Display labels, in presentation order.
_DISPLAY_COLUMNS: dict[str, str] = {
    "outcome": "Outcome",
    "age_band_months": "Age band (mo)",
    "n_observations": "n",
    "observed_mean": "Observed mean",
    "predictive_mean": "Predicted mean",
    "mean_error": "Mean error",
    "empirical_coverage": "Coverage",
    "expected_coverage": "Calibrated coverage",
    "mid_pit_mean": "PIT mean",
    "mid_pit_variance": "PIT variance",
    "expected_mid_pit_variance": "Calibrated PIT variance",
    "mid_pit_extreme_rate": "PIT extreme rate",
}

OVERALL_BAND = "all"


def nominal_level(table: pd.DataFrame, target: float | None = None) -> float:
    """The tabulated nominal level closest to the reporting convention.

    Resolved from the table rather than assumed, because a table written before
    the tabulated levels were tied to :mod:`vocab_growth.intervals` carries a 0.90
    outer level where a current one carries 0.89. Reporting the level actually
    found — and labelling it — keeps an older fit's table readable and honest
    instead of silently mislabelling it.
    """
    if "interval_probability" not in table.columns or table.empty:
        raise ValueError("Not a calibration table: no interval_probability column.")
    wanted = intervals.DEFAULT_CI_PROB if target is None else target
    levels = np.asarray(sorted(table["interval_probability"].dropna().unique()), dtype=float)
    if levels.size == 0:
        raise ValueError("Calibration table has no tabulated interval levels.")
    return float(levels[int(np.argmin(np.abs(levels - wanted)))])


def _at_level(table: pd.DataFrame, level: float | None) -> tuple[pd.DataFrame, float]:
    resolved = nominal_level(table) if level is None else level
    rows = table[np.isclose(table["interval_probability"], resolved)]
    return rows, resolved


def overall_calibration(
    table: pd.DataFrame, level: float | None = None
) -> tuple[pd.DataFrame, float]:
    """Per-outcome calibration pooled over ages, with the level it is reported at.

    Returns ``(frame, level)``. The frame has one row per outcome.
    """
    rows, resolved = _at_level(table, level)
    rows = rows[rows["age_band_months"] == OVERALL_BAND]
    return rows.reset_index(drop=True), resolved


def calibration_by_age(
    table: pd.DataFrame, level: float | None = None
) -> tuple[pd.DataFrame, float]:
    """Per-outcome, per-age-band calibration, with the level it is reported at.

    Age bands sort by their lower bound rather than lexically, so ``[108, 120)``
    follows ``[96, 108)`` instead of ``[12, 24)``.
    """
    rows, resolved = _at_level(table, level)
    rows = rows[rows["age_band_months"] != OVERALL_BAND].copy()
    if rows.empty:
        return rows.reset_index(drop=True), resolved
    rows["_lower"] = (
        rows["age_band_months"].str.extract(r"\[(-?\d+)", expand=False).astype(float)
    )
    sort_columns = ["outcome", "_lower"] if "outcome" in rows.columns else ["_lower"]
    rows = rows.sort_values(sort_columns).drop(columns="_lower")
    return rows.reset_index(drop=True), resolved


def format_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    """Select and label the presentation columns of a calibration selection."""
    columns = [column for column in _DISPLAY_COLUMNS if column in frame.columns]
    out = frame[columns].rename(columns=_DISPLAY_COLUMNS)
    for label, places in (
        ("Observed mean", 1),
        ("Predicted mean", 1),
        ("Mean error", 1),
        ("Coverage", 3),
        ("Calibrated coverage", 3),
        ("PIT mean", 3),
        ("PIT variance", 3),
        ("Calibrated PIT variance", 3),
        ("PIT extreme rate", 3),
    ):
        if label in out.columns:
            out[label] = out[label].astype(float).round(places)
    if "n" in out.columns:
        out["n"] = out["n"].astype(int)
    return out


def render_calibration_section(directory: str = ".") -> None:
    """Print the calibration evidence for a per-model report cell.

    Intended for a report cell with ``#| output: asis``, mirroring
    :func:`vocab_growth.plotting.ppc_count_distribution_gallery`. Prints a note
    recording the nominal level actually tabulated, the per-outcome table pooled
    over ages, and the per-age-band breakdown. Prints an explanatory line rather
    than failing when the fit predates the calibration table, so the section is
    never silently empty.
    """
    path = os.path.join(directory, "posterior_predictive_calibration.csv")
    if not os.path.exists(path):
        print(
            "_No calibration table for this fit "
            "(`posterior_predictive_calibration.csv` absent — refit to generate it)._"
        )
        return
    table = pd.read_csv(path)
    if table.empty:
        print("_The calibration table for this fit is empty._")
        return

    overall, level = overall_calibration(table)
    has_discrete_reference = "expected_mid_pit_variance" in table.columns
    if has_discrete_reference:
        print(
            f"Predictive coverage is reported at the **{level:.0%}** nominal level, "
            "the outer interval this fit tabulated. The outcome is a discrete "
            "count, so a perfectly calibrated model does *not* show coverage "
            f"exactly {level:.0%} or a PIT variance of "
            f"{UNIFORM_PIT_VARIANCE:.3f} (the continuous-uniform value): discrete "
            "intervals cover at least their nominal mass, and the deterministic "
            "mid-PIT is under-dispersed wherever the predictive mass concentrates "
            "on few counts (the young-age bands full of zeros). The **calibrated "
            "coverage** and **calibrated PIT variance** columns give the values a "
            "perfectly calibrated model would show for these observations, "
            "computed from this fit's own predictive draws — read the empirical "
            "columns against those, not against the nominal level or "
            f"{UNIFORM_PIT_VARIANCE:.3f}. A PIT mean near 0.5 still indicates "
            "little systematic bias; coverage above its calibrated reference with "
            "PIT variance below its calibrated reference means the predictive "
            "distribution is wider than the data warrant.\n"
        )
    else:
        # A table written before the discrete reference existed: state the
        # continuous benchmark, with the discreteness caveat it needs.
        print(
            f"Predictive coverage is reported at the **{level:.0%}** nominal "
            "level, the outer interval this fit tabulated. A well-calibrated "
            f"model has coverage near {level:.0%}, a PIT mean near 0.5, and a "
            f"PIT variance near {UNIFORM_PIT_VARIANCE:.3f} (the variance of a "
            "standard uniform) — but the outcome is a discrete count, so even a "
            "perfectly calibrated model shows some over-coverage and a PIT "
            "variance below that value, most visibly in bands where the "
            "predictive mass concentrates on few counts. This fit predates the "
            "calibrated discrete reference columns (refit to produce them), so "
            "small departures in that direction are not evidence of "
            "miscalibration.\n"
        )
    # The caveat travels with the numbers. Without it a reader of this page alone
    # would take over-coverage as evidence that the reported intervals are
    # conservative for a new child, which these in-sample checks cannot support.
    print(
        "These are **in-sample** checks: the replications condition on parameters "
        "that these same observations informed, so some over-coverage and an "
        "under-dispersed PIT are expected even for a well-specified model. That "
        "direction is therefore *not* evidence that the reported intervals are "
        "conservative for a new child — read these numbers for comparison across "
        "outcomes and ages, and for gross misfit. The out-of-sample counterpart is "
        "the LOO/ELPD comparison.\n"
    )
    print(format_calibration(overall).to_markdown(index=False))
    print("\n: Predictive calibration pooled over ages {#tbl-calibration-overall}\n")

    by_age, _ = calibration_by_age(table)
    if not by_age.empty:
        print("\n### By age band\n")
        print(
            "_Bands with few observations are uninformative: a single-observation "
            "band reports a coverage of 0 or 1 and a PIT variance of 0 by "
            "construction._\n"
        )
        print(format_calibration(by_age).to_markdown(index=False))
        print("\n: Predictive calibration by age band {#tbl-calibration-by-age}\n")
