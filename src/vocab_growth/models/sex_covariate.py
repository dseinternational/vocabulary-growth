# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Sex as a covariate in the reporting models (issue #324).

The study owner decided on 2026-09-08 that the models whose numbers are reported
should carry sex, so that predictions can be made by age **and** sex, in both
populations. The design is a centred covariate estimated only where sex is
recorded: girls ``+1/2``, boys ``-1/2``, and a child of unrecorded sex at zero,
the logit midpoint between them. No row is dropped. In the Down syndrome pool
missingness is exactly study-level, and every model carrying the term also
carries study intercepts, which absorb the unrecorded studies' sex mix; the
coefficients are identified from the studies that record sex, on the assumption
that the effect is common across studies.

This module holds what the three engines share:

* the two readers :func:`sex_effect_sigma` and :func:`sex_known_only`, which are
  the call sites ``fit_identity.BACKFILL_DEFAULTS`` makes its claim about -- a
  definition that predates the fields resolves to "no sex term, no restriction";
* :data:`SEX_LEVELS`, the contrast each reported level is evaluated at;
* the writers for the by-sex summary tables, so the three engines emit the same
  columns under the same names.

Every population trajectory the reporting models write is at contrast zero, the
sex-balanced midpoint, in both populations. The by-sex tables put the girls' and
boys' trajectories half a coefficient either side of it, from the same posterior
draws, so a girl-minus-boy difference is a paired quantity rather than a
difference of two independent summaries.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import vocab_growth.intervals as intervals
import vocab_growth.posterior_analysis as posterior_analysis

#: The two reported levels and the contrast each is evaluated at. Matches
#: ``observation_arrays.SEX_CONTRAST``, which codes the observed rows.
SEX_LEVELS: tuple[tuple[str, float], ...] = (("girls", 0.5), ("boys", -0.5))

#: Column the by-sex tables carry the level in.
SEX_COLUMN = "sex"


def sex_effect_sigma(definition) -> float | None:
    """The definition's sex-coefficient prior SD, or ``None`` for no sex term."""
    return getattr(definition, "sex_effect_sigma", None)


def sex_known_only(definition) -> bool:
    """Whether the definition restricts its frame to children of recorded sex."""
    return bool(getattr(definition, "sex_known_only", False))


def needs_sex_column(definition) -> bool:
    """Whether the frame builder must carry the ``sex`` column for this definition."""
    return sex_effect_sigma(definition) is not None or sex_known_only(definition)


def coefficient_draws(trace, name: str) -> np.ndarray | None:
    """A scalar coefficient's posterior draws, stacked as the grid extractors stack
    them (``chain`` then ``draw``), or ``None`` when the fit has no such variable."""
    if name not in trace.posterior:
        return None
    return np.asarray(
        trace.posterior[name].stack(sample=("chain", "draw")).values, dtype=float
    )


def shifted(logit_draws: np.ndarray, beta_draws: np.ndarray, contrast: float) -> np.ndarray:
    """Probability draws at one sex level, from population logits and a coefficient.

    ``logit_draws`` is ``(n_grid, n_samples)`` and ``beta_draws`` is
    ``(n_samples,)``, aligned draw for draw.
    """
    return 1.0 / (1.0 + np.exp(-(logit_draws + contrast * beta_draws[None, :])))


def logit(probability_draws: np.ndarray) -> np.ndarray:
    """Logit of stored probability draws, clipped away from the boundary."""
    p = np.clip(probability_draws, 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def _probability_block(
    X_query: np.ndarray,
    p_population: np.ndarray,
    *,
    n_trials: int,
    ci_prob: float,
    interval_kind: intervals.IntervalKind,
    y_predictive: np.ndarray | None,
    p_subject_marginal: np.ndarray | None,
) -> pd.DataFrame:
    if y_predictive is not None:
        table = posterior_analysis.posterior_summary_table(
            X_query,
            p_population,
            y_predictive,
            n_trials=n_trials,
            ci_prob=ci_prob,
            interval_kind=interval_kind,
        )
    else:
        # The joint engine draws no predictive counts at query ages, so its by-sex
        # table carries the latent proportion and expected count only, as its
        # pooled tables do.
        table = pd.DataFrame({"age_months": np.asarray(X_query, dtype=float)})
        for prefix, draws in (("p", p_population), ("Ey", p_population * n_trials)):
            outer = intervals.bands(draws, ci_prob, interval_kind, sample_axis=1)
            inner = intervals.bands(
                draws, intervals.INNER_CI_PROB, interval_kind, sample_axis=1
            )
            table[f"{prefix}_median"] = np.median(draws, axis=1)
            table[f"{prefix}_ci50_lo"] = inner[:, 0]
            table[f"{prefix}_ci50_hi"] = inner[:, 1]
            table[f"{prefix}_ci_lo"] = outer[:, 0]
            table[f"{prefix}_ci_hi"] = outer[:, 1]
    if p_subject_marginal is not None:
        table = posterior_analysis.add_probability_estimand_columns(
            table,
            p_population,
            p_subject_marginal,
            n_trials=n_trials,
            ci_prob=ci_prob,
            interval_kind=interval_kind,
        )
    return table


def write_probability_by_sex(
    output_dir: str,
    suffix: str,
    X_query: np.ndarray,
    population: dict[str, np.ndarray],
    *,
    n_trials: int,
    ci_prob: float,
    interval_kind: intervals.IntervalKind = "eti",
    predictive: dict[str, np.ndarray] | None = None,
    subject_marginal: dict[str, np.ndarray] | None = None,
    max_age_months: float | None = None,
) -> pd.DataFrame:
    """Write ``posterior_summary_{suffix}_by_sex.csv`` and return the table.

    One block per level in :data:`SEX_LEVELS`, each with the columns the pooled
    ``posterior_summary_{suffix}.csv`` carries for this engine, and a leading
    ``sex`` column. ``population`` maps each level to its zero-effect probability
    draws; ``predictive`` and ``subject_marginal``, where the engine draws them,
    to a new child's counts and probabilities at that level. An empty ``suffix``
    names ``posterior_summary_by_sex.csv``, beside a single-outcome model's
    ``posterior_summary.csv``.
    """
    blocks = []
    for level, _ in SEX_LEVELS:
        block = _probability_block(
            X_query,
            population[level],
            n_trials=n_trials,
            ci_prob=ci_prob,
            interval_kind=interval_kind,
            y_predictive=None if predictive is None else predictive[level],
            p_subject_marginal=None
            if subject_marginal is None
            else subject_marginal[level],
        )
        block.insert(0, SEX_COLUMN, level)
        blocks.append(block)
    table = posterior_analysis.trim_reported_ages(
        pd.concat(blocks, ignore_index=True), max_age_months
    )
    stem = f"posterior_summary_{suffix}" if suffix else "posterior_summary"
    table.to_csv(os.path.join(output_dir, f"{stem}_by_sex.csv"), index=False)
    return table


def write_rate_by_sex(
    output_dir: str,
    name: str,
    X_query: np.ndarray,
    population: dict[str, np.ndarray],
    *,
    ci_prob: float,
    subject_marginal: dict[str, np.ndarray] | None = None,
    max_age_months: float | None = None,
) -> pd.DataFrame:
    """Write ``posterior_summary_{name}_by_sex.csv`` for a bounded rate (``q``, ``r``).

    No expected-count columns, for the reason
    :func:`vocab_growth.posterior_analysis.add_rate_estimand_columns` gives.
    """
    blocks = []
    for level, _ in SEX_LEVELS:
        block = intervals.summarise(
            population[level],
            X_query,
            name=f"{name}_query",
            outer=ci_prob,
            sample_axis=1,
        ).rename(
            columns={
                "median": f"{name}_median",
                "ci50_lo": f"{name}_ci50_lo",
                "ci50_hi": f"{name}_ci50_hi",
                "ci_lo": f"{name}_ci_lo",
                "ci_hi": f"{name}_ci_hi",
            }
        )
        if subject_marginal is not None:
            block = posterior_analysis.add_rate_estimand_columns(
                block,
                population[level],
                subject_marginal[level],
                name=name,
                ci_prob=ci_prob,
            )
        block.insert(0, SEX_COLUMN, level)
        blocks.append(block)
    table = posterior_analysis.trim_reported_ages(
        pd.concat(blocks, ignore_index=True), max_age_months
    )
    table.to_csv(
        os.path.join(output_dir, f"posterior_summary_{name}_by_sex.csv"), index=False
    )
    return table


def difference_rows(
    X_query: np.ndarray,
    outcome: str,
    girls: np.ndarray,
    boys: np.ndarray,
    *,
    n_trials: int,
    ci_prob: float,
    max_age_months: float | None = None,
) -> pd.DataFrame:
    """Girl-minus-boy expected counts at each query age, paired draw for draw."""
    difference = (girls - boys) * n_trials
    outer = intervals.bands(difference, ci_prob, "eti", sample_axis=1)
    inner = intervals.bands(difference, intervals.INNER_CI_PROB, "eti", sample_axis=1)
    table = pd.DataFrame(
        {
            "outcome": outcome,
            "age_months": np.asarray(X_query, dtype=float),
            "Ey_girls_median": np.median(girls * n_trials, axis=1),
            "Ey_boys_median": np.median(boys * n_trials, axis=1),
            "Ey_difference_median": np.median(difference, axis=1),
            "Ey_difference_ci50_lo": inner[:, 0],
            "Ey_difference_ci50_hi": inner[:, 1],
            "Ey_difference_ci_lo": outer[:, 0],
            "Ey_difference_ci_hi": outer[:, 1],
            "P_girls_gt_boys": np.mean(difference > 0, axis=1),
        }
    )
    return posterior_analysis.trim_reported_ages(table, max_age_months)


def write_differences(output_dir: str, tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Write ``posterior_summary_sex_difference.csv`` from per-outcome tables."""
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(
        os.path.join(output_dir, "posterior_summary_sex_difference.csv"), index=False
    )
    return table


def write_coefficients(
    output_dir: str,
    draws: dict[str, np.ndarray],
    *,
    ci_prob: float,
) -> pd.DataFrame:
    """Write ``posterior_summary_sex_effect.csv``: each coefficient on the logit scale.

    ``draws`` maps a parameter name to its flattened posterior draws. The odds
    ratio column is ``exp`` of the coefficient: the girls-to-boys odds ratio of a
    word being understood (or, for a ratio, said or signed given understood).
    """
    rows = []
    for name, values in draws.items():
        values = np.asarray(values, dtype=float).ravel()
        lo, hi = intervals.interval_1d(values, ci_prob, "eti")
        rows.append(
            {
                "parameter": name,
                "median": float(np.median(values)),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "odds_ratio_median": float(np.exp(np.median(values))),
                "P_positive": float(np.mean(values > 0)),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(output_dir, "posterior_summary_sex_effect.csv"), index=False)
    return table
