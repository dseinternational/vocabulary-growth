# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Wave-forward sequential validation for the cross-lag models (issue #242 item 5).

VG16's understood PSIS-LOO is suppressed, and correctly so: the lag predictor for
a child's later wave embeds that child's *earlier* understood count, so leaving
one administration out of the likelihood does not leave it out of the model, and
Pareto-k cannot see the leak. The registered replacement is a grouped
forward-chaining score, and this is it.

**Both cross-lag models, on both engines.** VG25's sign -> speech lag has the
same property for the same reason -- its predictor reads an earlier wave's
`signed` and `understood` counts -- so its understood and signed LOO are
suppressed too, and this is equally the replacement. Its report page says so and
sends the reader here, which for a while it could not do: the script took only
models carrying `use_cross_lag`, a `BivariateModelDefinition` field that VG25
does not have.

What is held out, and why
-------------------------

Children are split into K folds. For fold ``k``, every row of a fold-``k`` child
**after their first administration wave** is removed from the likelihood; their
first wave stays in training, as does everything belonging to every other child.
The held-out later waves are then scored.

That conditioning is the point. The lag predictor for a held-out wave is built
from the child's earlier wave, which is training data, so the score is an honest
one-step-ahead prediction: *given what this child was at their first visit and
what other children did, how well is their next visit predicted?* Holding out the
whole child instead (``--holdout-unit child``) makes the lag source a row the
model never saw, which is a different and stricter question -- the child's random
effect is then drawn from its prior -- and it is offered because #289 task 3.8
asks for the choice to be made explicitly rather than assumed.

**First waves are never scored, under either unit.** A first wave carries
``has_lag = 0``, so ``x_lag`` is zero and the cross-lag term vanishes from its
likelihood: the two models being compared are numerically identical there, and
including those rows would dilute the comparison with rows that cannot inform it.
Excluding them is therefore not only leak avoidance, it is the only informative
scored set.

What is compared
----------------

The model against **itself with the coefficient removed** -- ``use_cross_lag`` for
VG16, ``use_sign_cross_lag`` for VG25 -- on the identical prepared frame and the
identical folds. Comparing VG16 against VG10, or VG25 against VG24, would
confound the coefficient with every other field the two definitions do not
share; comparing against a definition that differs in one boolean does not. The
control is fitted in a scratch directory and is never a model of record.

The headline is the **spoken** elpd difference on both models, because on both
the coefficient enters the production-ratio logit. The other outcomes are
written beside it and are not the headline:

- VG16: understood is a pure control. The lag enters nothing else, so a held-out
  row's understood density is the same under both arms up to sampling noise.
- VG25: understood and signed are the controls, and the **four-cell composition
  is not** -- with ``sign_lag_in_cells`` the lag enters the population marginals
  the Dirichlet-Multinomials are built on, which is the scope decision VG25's
  registration turned on. Scoring the marginals alone would score the model on
  the evidence that decision chose against, so `elpd_cells` is computed and
  reported as a second place the coefficient can pay for itself. nz_01's
  three-cell produced composition is scored into the same column: its rows are
  disjoint from the four-cell rows, so no row is counted twice.

Every fold fit is screened by the canonical diagnostics scan, and a model with
any failed fold is flagged rather than dropped. The fit itself is
``vocab_growth.fold_fits.fit_holdout_fold``, shared with ``kfold_loso.py``: the
two scripts hold different things out and score different units, but the fit
between those decisions is the same one, and the hand copy this script started
with read the energy verdict from the wrong key.

Outputs
-------

- ``output/comparisons/wave_forward_row_elpds.csv`` -- per scored row, per model
- ``output/comparisons/wave_forward_summary.csv`` -- per-model totals
- ``output/comparisons/wave_forward_compare.csv`` -- the paired difference with
  its standard error
- ``output/comparisons/wave_forward_fits.csv`` -- per (model, fold) convergence

Usage
-----

    uv run python scripts/wave_forward_score.py [--model vg16|vg25] [--folds 5]
        [--config test] [--holdout-unit later-waves|child] [--suffix ...]

Like ``kfold_loso.py`` this fits its own folds rather than reading a model of
record, so its comparison-manifest entry records the raw-data fingerprint rather
than a contributing fit.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import time
from dataclasses import dataclass

import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import gammaln, logsumexp
from scipy.stats import betabinom

from vocab_growth import environment as env
from vocab_growth.analysis_frames import build_analysis_frame
from vocab_growth.comparisons_provenance import (
    ComparisonOutputs,
    write_comparison_manifest,
)
from vocab_growth.fit_artifacts import source_data_hash
from vocab_growth.fold_fits import fit_holdout_fold, fold_gate_fields
from vocab_growth.models.catalogue import engine_for_definition
from vocab_growth.models.common_joint_modality import (
    CELL_COLUMNS,
    PROD_CELL_COLUMNS,
)
from vocab_growth.models.cross_lag import (
    prev_wave_lag_for_frame,
    prev_wave_sign_share_lag_for_frame,
)
from vocab_growth.models.cross_lag import (
    wave_index as subject_wave_index,
)
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.models.likelihood_utils import (
    SPOKEN_FALLBACK_PRODUCT,
    nested_outcome_spec,
)

OUT_DIR = env.comparisons_output_dir()
TMP_DIR = os.path.join(env.output_root(), "wave_forward_tmp")

#: The definition fields that turn a cross-lag on, one per engine. Derived
#: rather than listed so a third lag cannot be added without appearing here:
#: `lag_field` raises for a definition carrying none of them, and for one
#: carrying two, which would make "remove the coefficient" ambiguous.
LAG_FIELDS = ("use_cross_lag", "use_sign_cross_lag")

#: Models this script can score. The requirement is a cross-lag: without one
#: there is no coefficient to remove and the two arms are the same model.
CROSS_LAG_MODELS = tuple(
    key
    for key, d in MODEL_REGISTRY.items()
    if any(getattr(d, field, False) for field in LAG_FIELDS)
)

#: Which elpd columns each engine produces, headline first. The headline is the
#: spoken difference on both engines, because that is where the coefficient
#: enters the logit; the rest are written beside it.
OUTCOME_COLUMNS = {
    "bivariate_re": ("elpd_spoken", "elpd_understood"),
    "joint": ("elpd_spoken", "elpd_cells", "elpd_understood", "elpd_signed"),
}

CONTROL_SUFFIX = "-nolag"


def lag_field(definition) -> str:
    """The boolean that turns this definition's cross-lag on.

    One and exactly one, checked rather than assumed: a definition with none has
    no coefficient to remove, and one with two would leave "the control" naming
    two different models.
    """
    present = [field for field in LAG_FIELDS if getattr(definition, field, False)]
    if len(present) == 1:
        return present[0]
    raise ValueError(
        f"{definition.model_id} carries {len(present)} cross-lag field(s) "
        f"{present}; exactly one of {list(LAG_FIELDS)} is required for the "
        "control arm to be the same model with one coefficient removed."
    )


# ============================================================
# Waves and folds
# ============================================================


def wave_index(analysis_df: pd.DataFrame) -> np.ndarray:
    """0 for each child's first administration wave, 1 for the next, and so on.

    The definition lives beside ``iter_subject_age_waves`` in
    ``vocab_growth.models.cross_lag`` -- the same grouping the lag itself uses,
    so "a later wave" here and "a row with a prior-wave source" there cannot
    drift apart. This is the frame-taking wrapper.
    """
    return subject_wave_index(
        analysis_df["subject_code"].to_numpy(), analysis_df["age"].to_numpy()
    )


def stratified_subject_folds(
    analysis_df: pd.DataFrame, K: int, seed: int = 47
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Assign each subject to a fold, stratified by study and wave count.

    Stratifying on the number of *waves* rather than the number of rows is what
    this script needs and ``kfold_loso.py`` does not: a child with one wave
    contributes nothing to score however many forms they carry at it, so folds
    balanced on row count could still put most of the scorable children in one
    fold.
    """
    waves = wave_index(analysis_df)
    subj = (
        analysis_df.assign(_wave=waves)
        .groupby("subject_code")
        .agg(study_code=("study_code", "first"), n_waves=("_wave", "max"))
        .reset_index()
    )
    subj["n_waves"] = subj["n_waves"] + 1

    def _bin(n: int) -> str:
        return str(n) if n < 4 else "4+"

    subj["stratum"] = (
        subj["study_code"].astype(str) + "_" + subj["n_waves"].apply(_bin)
    )
    rng = np.random.default_rng(seed)
    fold_of = np.zeros(len(subj), dtype=int)
    for _stratum, group in subj.groupby("stratum"):
        idxs = np.asarray(group.index.to_numpy(), dtype=np.int64).copy()
        rng.shuffle(idxs)
        for k, i in enumerate(idxs):
            fold_of[i] = k % K
    subj["fold"] = fold_of
    folds = [subj.loc[subj["fold"] == k, "subject_code"].to_numpy() for k in range(K)]
    return folds, subj


def holdout_mask(
    analysis_df: pd.DataFrame,
    fold_subjects: np.ndarray,
    *,
    unit: str,
) -> np.ndarray:
    """Which rows leave the likelihood for this fold.

    ``later-waves`` keeps each fold child's first wave in training, so the lag
    source for every scored row is data the model saw. ``child`` removes the
    whole child, which makes the lag source a row the model never saw and the
    child's random effect a prior draw.
    """
    in_fold = analysis_df["subject_code"].isin(fold_subjects).to_numpy()
    if unit == "child":
        return in_fold
    if unit != "later-waves":
        raise ValueError(f"unknown holdout unit {unit!r}")
    return in_fold & (wave_index(analysis_df) > 0)


def scored_rows(analysis_df: pd.DataFrame, fold_subjects: np.ndarray) -> np.ndarray:
    """The rows this fold contributes to the comparison, as positional indices.

    A fold child's later waves, and only those. First waves are excluded under
    both holdout units because ``x_lag`` is zero there, so the two arms assign
    them the identical density and scoring them adds noise without signal.
    """
    in_fold = analysis_df["subject_code"].isin(fold_subjects).to_numpy()
    return np.flatnonzero(in_fold & (wave_index(analysis_df) > 0))


def lag_source(analysis_df: pd.DataFrame, definition) -> tuple[np.ndarray, np.ndarray]:
    """``(has_lag, source_row)`` from the model's own lag rule.

    A later wave is not automatically a lagged wave: if every earlier wave of
    that child lacks a comprehension count -- and 391 of the pool's 448
    comprehension-less rows sit on three forms that never record one -- then
    ``has_lag`` is 0 and the cross-lag term drops out of that row's likelihood.
    The two arms then assign the row the identical density, so it contributes
    exactly nothing to the elpd difference while still changing the paired
    standard error.

    Read from ``prev_wave_lag_for_frame`` rather than reconstructed, so a variant
    that moves the gap ceiling, the zero handling or the same-form restriction
    moves this too -- and so ``source_row`` is the row the model actually reads,
    which is what decides whether the predictor stands on training data.
    """
    if lag_field(definition) == "use_sign_cross_lag":
        # The ratio lag takes no `n_trials`: it divides one count by another
        # scored on the same form in the same administration, so the inventory
        # size cancels out of the predictor.
        prev_idx, has_lag_f, _logit = prev_wave_sign_share_lag_for_frame(
            analysis_df, definition
        )
    else:
        prev_idx, has_lag_f, _logit = prev_wave_lag_for_frame(
            analysis_df, definition.n_trials, definition
        )
    return np.asarray(has_lag_f, dtype=float) > 0, np.asarray(prev_idx, dtype=int)


def source_in_training(
    has_lag_flags: np.ndarray, source_row: np.ndarray, holdout: np.ndarray
) -> np.ndarray:
    """Whether each row's lag source is a row the likelihood saw.

    Under ``later-waves`` a second wave's source is the child's first, which
    stays in training; a *third* wave's source is the second, which does not. So
    the scored set splits in two, and the split matters for what may be claimed.
    A row whose source is in training is predicted from data the model was fitted
    to, which is the ordinary out-of-sample statement. A row whose source is held
    out is still an honest forward-chaining prediction -- its own outcome appears
    nowhere in its own predictor -- but it conditions on a row the likelihood
    never saw, so it is not "trained on everything except the scored rows".

    Reported as a column rather than settled in prose, and the headline
    restriction takes the strict side.
    """
    return has_lag_flags & ~np.asarray(holdout, dtype=bool)[source_row]


# ============================================================
# Per-fold fit
# ============================================================


@dataclass
class FoldFitRecord:
    arm: str
    fold: int
    n_holdout_rows: int
    n_scored_rows: int
    wall_seconds: float
    passed: bool
    max_rhat: float
    min_ess: float
    divergences: int
    bfmi_ok: bool


# ============================================================
# Row-level held-out predictive density
# ============================================================


def _betabinom_elpd(y, n, p_draws, k_draws, log_NK: float) -> float:
    """Log mean predictive density of one count over the posterior draws."""
    p = np.clip(p_draws, 1e-12, 1 - 1e-12)
    ll = betabinom.logpmf(y, n, p * k_draws, (1 - p) * k_draws)
    return float(logsumexp(ll.ravel()) - log_NK)


def _dirichlet_multinomial_elpd(counts, total, alpha_draws, log_NK: float) -> float:
    """Log mean predictive density of one composition over the posterior draws.

    ``alpha_draws`` is ``(chain, draw, K)`` and ``counts`` is ``(K,)``. Written
    out rather than taken from a library because the engine's own likelihood is
    a ``pm.DirichletMultinomial`` on exactly these parameters, and the point of
    the score is to evaluate that density at a held-out row.
    """
    counts = np.asarray(counts, dtype=float)
    alpha_sum = alpha_draws.sum(axis=-1)
    ll = (
        gammaln(total + 1.0)
        + gammaln(alpha_sum)
        - gammaln(total + alpha_sum)
        + (
            gammaln(counts + alpha_draws)
            - gammaln(alpha_draws)
            - gammaln(counts + 1.0)
        ).sum(axis=-1)
    )
    return float(logsumexp(ll.ravel()) - log_NK)


def _nested_rows(
    frame: pd.DataFrame, outcome_col: str, n_trials: int, *, eligible=None
):
    """``(observed, trials, is_conditional)`` per frame row for a nested outcome.

    The engines decide per row whether an outcome enters conditionally on the
    same administration's comprehension or marginally over the inventory, and
    the density differs between the two. Read from the shared
    :func:`nested_outcome_spec` so this cannot drift from the likelihood.

    ``eligible`` is the engine's own eligibility mask minus its holdout term:
    the joint engine gives a cross-tab row no marginal outcome at all, and a
    scored row must be classified the way the model would classify it. Absent
    for the bivariate engine, which has no such exclusion.
    """
    spec = nested_outcome_spec(
        frame,
        parent_col="understood",
        outcome_col=outcome_col,
        n_trials=n_trials,
        eligible_mask=eligible,
    )
    observed = np.full(len(frame), -1, dtype=int)
    trials = np.full(len(frame), n_trials, dtype=int)
    is_conditional = np.zeros(len(frame), dtype=bool)
    observed[spec.indices] = spec.observed
    trials[spec.indices] = spec.trials
    is_conditional[spec.indices] = spec.is_conditional
    return observed, trials, is_conditional


def row_elpds(
    frame: pd.DataFrame,
    trace: xr.DataTree,
    rows: np.ndarray,
    definition,
    lagged: np.ndarray,
    clean: np.ndarray,
    *,
    engine: str,
) -> pd.DataFrame:
    """Marginal predictive log-density of each scored row, by outcome.

    One row at a time rather than one child at a time (``kfold_loso``'s unit),
    because the scored set here is a subset of a child's rows and the paired
    comparison needs the two arms aligned on the same rows.

    The outcomes are kept apart rather than summed, because on both engines some
    of them are controls the coefficient cannot reach and summing would bury the
    signal in them. Which outcomes there are is the engine's business, and
    :data:`OUTCOME_COLUMNS` is the same table the driver reports on.
    """
    if engine == "joint":
        return _joint_row_elpds(frame, trace, rows, definition, lagged, clean)
    if engine == "bivariate_re":
        return _bivariate_row_elpds(frame, trace, rows, definition, lagged, clean)
    raise ValueError(
        f"No row scoring for the {engine!r} engine. A cross-lag registered on a "
        "new engine needs its likelihood evaluated here before it can be scored."
    )


def _bivariate_row_elpds(
    frame: pd.DataFrame,
    trace: xr.DataTree,
    rows: np.ndarray,
    definition,
    lagged: np.ndarray,
    clean: np.ndarray,
) -> pd.DataFrame:
    """VG16: understood is a pure control, spoken carries the coefficient."""
    n_trials = definition.n_trials
    p_u_obs = trace.posterior["p_u_obs"].values
    p_s_obs = trace.posterior["p_s_obs"].values
    q_obs = trace.posterior["q_obs"].values
    kappa_u_obs = trace.posterior["kappa_u_obs"].values
    kappa_s_obs = trace.posterior["kappa_s_obs"].values
    n_chain, n_draw, _ = p_u_obs.shape
    log_NK = math.log(n_chain * n_draw)

    spoken_observed, spoken_trials, spoken_is_conditional = _nested_rows(
        frame, "spoken", n_trials
    )

    records = []
    for idx in rows:
        row = frame.iloc[idx]
        elpd_u = float("nan")
        elpd_s = float("nan")
        if pd.notna(row["understood"]):
            elpd_u = _betabinom_elpd(
                int(row["understood"]),
                n_trials,
                p_u_obs[:, :, idx],
                kappa_u_obs[:, :, idx],
                log_NK,
            )
        if pd.notna(row["spoken"]):
            p_draws = (
                q_obs[:, :, idx]
                if spoken_is_conditional[idx]
                else p_s_obs[:, :, idx]
            )
            elpd_s = _betabinom_elpd(
                spoken_observed[idx],
                spoken_trials[idx],
                p_draws,
                kappa_s_obs[:, :, idx],
                log_NK,
            )
        records.append(
            {
                "row": int(idx),
                "subject_code": int(row["subject_code"]),
                "age_months": float(row["age"]),
                "has_lag": bool(lagged[idx]),
                "source_in_training": bool(clean[idx]),
                "elpd_understood": elpd_u,
                "elpd_spoken": elpd_s,
                "spoken_branch": (
                    ""
                    if pd.isna(row["spoken"])
                    else ("conditional" if spoken_is_conditional[idx] else "marginal")
                ),
            }
        )
    return pd.DataFrame(records)


def _joint_row_elpds(
    frame: pd.DataFrame,
    trace: xr.DataTree,
    rows: np.ndarray,
    definition,
    lagged: np.ndarray,
    clean: np.ndarray,
) -> pd.DataFrame:
    """VG25: spoken carries the coefficient, and so does the composition.

    Understood and signed are controls -- the sign lag enters neither. The cell
    compositions are **not** a control: with ``sign_lag_in_cells`` the lag is
    added to the population production logit the Dirichlet-Multinomials are
    built on, which is the scope decision the registration took, so the
    composition is a second place the coefficient can pay for itself.

    The four-cell and produced-cell rows are disjoint -- different studies
    contribute them -- so both land in one ``elpd_cells`` column without a row
    being counted twice.

    **Which rows those are is read from the frame, not from the model's masks.**
    ``obs_cells_mask`` and ``obs_prod_mask`` mark the rows in the *likelihood*,
    and a fold's held-out rows are excluded from it by construction -- so every
    row this function scores is absent from both, and using them scored no
    composition at all. The frame's own columns are the criterion, and no
    inclusion flag has to be reconstructed to read them: ``include_uk07_cells``
    and ``include_es01_cells`` act at data preparation, moving a study's rows
    between the cross-tab and marginal branches of the *frame*, so a row that is
    not a cross-tab row arrives with ``signed_spoken`` missing.

    A four-cell row also carries **no spoken or signed marginal**: the engine's
    ``marginal_outcome_eligible`` excludes it, because its production
    information is in the composition. Scoring a marginal density there would
    score a density the model does not hold, so the same exclusion is applied
    here through ``nested_outcome_spec``'s own ``eligible_mask``. nz_01's
    produced rows are not excluded and carry both.
    """
    n_trials = definition.n_trials
    posterior = trace.posterior
    p_u_obs = posterior["p_u_obs"].values
    q_obs = posterior["q_obs"].values
    r_obs = posterior["r_obs"].values
    kappa_u_obs = posterior["kappa_u_obs"].values
    kappa_s_obs = posterior["kappa_s_obs"].values
    kappa_sign_obs = posterior["kappa_sign_obs"].values
    pi_cells = posterior["pi_cells_obs"].values
    conc = posterior["conc"].values
    n_chain, n_draw, _ = p_u_obs.shape
    log_NK = math.log(n_chain * n_draw)

    is_cell_row = frame["signed_spoken"].notna().to_numpy()
    is_prod_row = (
        frame["prod_signed_spoken"].notna().to_numpy()
        if "prod_signed_spoken" in frame.columns
        else np.zeros(len(frame), dtype=bool)
    )

    spoken_observed, spoken_trials, spoken_is_conditional = _nested_rows(
        frame, "spoken", n_trials, eligible=~is_cell_row
    )
    signed_observed, signed_trials, signed_is_conditional = _nested_rows(
        frame, "signed", n_trials, eligible=~is_cell_row
    )

    def _counts(columns):
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            return None
        return frame[columns].to_numpy(dtype=float)

    cell_counts = _counts(CELL_COLUMNS)
    prod_counts = _counts(PROD_CELL_COLUMNS)
    cell_total = pd.to_numeric(
        frame.get("cell_total", pd.Series(np.nan, index=frame.index)), errors="coerce"
    ).to_numpy(dtype=float)
    prod_total = pd.to_numeric(
        frame.get("prod_total", pd.Series(np.nan, index=frame.index)), errors="coerce"
    ).to_numpy(dtype=float)

    records = []
    for idx in rows:
        row = frame.iloc[idx]
        elpd_u = float("nan")
        elpd_s = float("nan")
        elpd_sign = float("nan")
        elpd_cells = float("nan")

        if pd.notna(row["understood"]):
            elpd_u = _betabinom_elpd(
                int(row["understood"]),
                n_trials,
                p_u_obs[:, :, idx],
                kappa_u_obs[:, :, idx],
                log_NK,
            )
        if pd.notna(row["spoken"]) and spoken_observed[idx] >= 0:
            p_draws = (
                q_obs[:, :, idx]
                if spoken_is_conditional[idx]
                # Indexed before the product, not after: `p_u_obs * q_obs`
                # builds a (chain, draw, n_obs) array, and doing that inside the
                # row loop repeats it once per scored row.
                else p_u_obs[:, :, idx] * q_obs[:, :, idx]
            )
            elpd_s = _betabinom_elpd(
                spoken_observed[idx],
                spoken_trials[idx],
                p_draws,
                kappa_s_obs[:, :, idx],
                log_NK,
            )
        if pd.notna(row["signed"]) and signed_observed[idx] >= 0:
            p_draws = (
                r_obs[:, :, idx]
                if signed_is_conditional[idx]
                # Indexed before the product, not after: `p_u_obs * r_obs`
                # builds a (chain, draw, n_obs) array, and doing that inside the
                # row loop repeats it once per scored row.
                else p_u_obs[:, :, idx] * r_obs[:, :, idx]
            )
            elpd_sign = _betabinom_elpd(
                signed_observed[idx],
                signed_trials[idx],
                p_draws,
                kappa_sign_obs[:, :, idx],
                log_NK,
            )
        if is_cell_row[idx] and cell_counts is not None:
            elpd_cells = _dirichlet_multinomial_elpd(
                cell_counts[idx],
                cell_total[idx],
                conc[:, :, None] * pi_cells[:, :, idx, :],
                log_NK,
            )
        elif is_prod_row[idx] and prod_counts is not None:
            # The produced composition drops the "neither" cell and keeps the
            # other three Dirichlet parameters unrenormalised, exactly as the
            # likelihood does -- renormalising would define a different model.
            elpd_cells = _dirichlet_multinomial_elpd(
                prod_counts[idx],
                prod_total[idx],
                conc[:, :, None] * pi_cells[:, :, idx, 1:],
                log_NK,
            )

        records.append(
            {
                "row": int(idx),
                "subject_code": int(row["subject_code"]),
                "age_months": float(row["age"]),
                "has_lag": bool(lagged[idx]),
                "source_in_training": bool(clean[idx]),
                "elpd_understood": elpd_u,
                "elpd_spoken": elpd_s,
                "elpd_signed": elpd_sign,
                "elpd_cells": elpd_cells,
                "spoken_branch": (
                    ""
                    if pd.isna(row["spoken"])
                    else ("conditional" if spoken_is_conditional[idx] else "marginal")
                ),
            }
        )
    return pd.DataFrame(records)


# ============================================================
# Comparison
# ============================================================


#: The three row sets the comparison is reported on, strictest first.
#:
#: ``lagged-from-training`` is the headline: the coefficient can move these rows,
#: and their predictor stands entirely on data the likelihood saw. ``lagged``
#: adds the rows whose lag source was itself held out -- still forward chaining,
#: but conditioning on an unseen row. ``all-later-waves`` adds the rows with no
#: lag at all, which enter both arms identically: they leave the total untouched
#: and shrink the per-row spread the standard error is built from, so quoting
#: that standard error would make the comparison look more precise than its
#: evidence.
RESTRICTIONS = ("lagged-from-training", "lagged", "all-later-waves")


def _restriction_mask(wide: pd.DataFrame, restriction: str) -> np.ndarray:
    if restriction == "lagged-from-training":
        return wide["source_in_training"].to_numpy(dtype=bool)
    if restriction == "lagged":
        return wide["has_lag"].to_numpy(dtype=bool)
    if restriction == "all-later-waves":
        return np.ones(len(wide), dtype=bool)
    raise ValueError(f"unknown restriction {restriction!r}")


#: The row identity a scored row is paired on across the two arms. Both arms
#: score the identical rows, so these are the columns that must match.
PAIR_KEYS = (
    "fold",
    "row",
    "subject_code",
    "age_months",
    "has_lag",
    "source_in_training",
    "spoken_branch",
)


def wide_table(long: pd.DataFrame, columns, arms) -> pd.DataFrame:
    """One row per scored row, with each outcome's elpd under each arm.

    ``pivot_table`` drops a value column that is entirely missing, and an
    outcome can legitimately score nothing in a run -- a fold with no cross-tab
    row among its later waves scores no composition. The absent columns are
    named here instead, so :func:`paired_difference` reports them as zero rows
    rather than raising a ``KeyError`` several minutes of fold fitting later.

    ``dropna=False`` is **not** the way to do that. On a multi-column index it
    expands the result to the cartesian product of the index levels: on a
    two-row example whose levels multiply out to eight, it returned eight rows,
    six of them combinations that never existed.
    """
    wide = long.pivot_table(
        index=list(PAIR_KEYS), columns="arm", values=list(columns)
    )
    wide.columns = [f"{outcome}_{arm}" for outcome, arm in wide.columns]
    wide = wide.reset_index()
    for column in columns:
        for arm in arms:
            if f"{column}_{arm}" not in wide.columns:
                wide[f"{column}_{arm}"] = np.nan
    return wide


def paired_difference(wide: pd.DataFrame, column: str, *, restriction: str) -> dict:
    """The cross-lag arm minus the control, paired on the scored rows.

    Paired because both arms score the identical rows: differencing two
    independent totals throws away the correlation between them and inflates the
    standard error, which is the defect #289 task 3.2 records for the VG20/VG22
    comparison. Rows either arm could not score are dropped from both.

    See :data:`RESTRICTIONS` for what each row set contains and why the strictest
    is the one to quote.
    """
    keep = _restriction_mask(wide, restriction)
    lag = wide[f"{column}_lag"].to_numpy()
    ctl = wide[f"{column}_control"].to_numpy()
    usable = keep & np.isfinite(lag) & np.isfinite(ctl)
    diff = lag[usable] - ctl[usable]
    n = int(usable.sum())
    row = {"outcome": column, "rows_scored": restriction, "n_rows": n}
    if n < 2:
        return {**row, "elpd_diff": float("nan"), "se": float("nan"),
                "mean_per_row": float("nan")}
    return {
        **row,
        "elpd_diff": float(diff.sum()),
        "se": float(np.sqrt(n) * np.std(diff, ddof=1)),
        "mean_per_row": float(diff.mean()),
    }


# ============================================================
# Driver
# ============================================================


def control_definition(definition):
    """``definition`` with the cross-lag removed and nothing else changed."""
    field = lag_field(definition)
    control = dataclasses.replace(
        definition,
        **{field: False},
        config_name=f"{definition.config_name}{CONTROL_SUFFIX}",
    )
    changed = {
        f.name
        for f in dataclasses.fields(definition)
        if getattr(definition, f.name) != getattr(control, f.name)
    }
    # The control exists to isolate one coefficient. If `replace` ever starts
    # moving anything else -- a derived field, a renamed flag -- the comparison
    # stops being about the cross-lag and this is where it should stop.
    assert changed == {field, "config_name"}, changed
    return control


def main(
    model_key: str,
    K: int,
    sampling_config_name: str,
    unit: str,
    suffix: str,
) -> None:
    definition = MODEL_REGISTRY[model_key]
    try:
        lag_field(definition)
    except ValueError as exc:
        raise SystemExit(
            f"{exc} Models this script can score: {', '.join(CROSS_LAG_MODELS)}."
        ) from None
    engine = engine_for_definition(definition).name
    if engine not in OUTCOME_COLUMNS:
        raise SystemExit(
            f"{model_key} runs on the {engine!r} engine, which this script has "
            "no row scoring for."
        )
    columns = OUTCOME_COLUMNS[engine]
    if definition.spoken_fallback != SPOKEN_FALLBACK_PRODUCT:
        raise NotImplementedError(
            "row_elpds implements only the "
            f"{SPOKEN_FALLBACK_PRODUCT!r} spoken fallback; {model_key} uses "
            f"{definition.spoken_fallback!r} and would be scored under the "
            "wrong likelihood."
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    written = ComparisonOutputs(OUT_DIR)

    frame, _meta = build_analysis_frame(model_key, definition)
    waves = wave_index(frame)
    n_scorable = int((waves > 0).sum())
    print(
        f"{model_key.upper()}: {len(frame)} rows / "
        f"{frame['subject_code'].nunique()} children; "
        f"{n_scorable} rows in a later wave, "
        f"{frame.loc[waves > 0, 'subject_code'].nunique()} children contribute one"
    )
    if n_scorable == 0:
        raise SystemExit("No child has a second wave; there is nothing to score.")

    lagged, source_row = lag_source(frame, definition)
    print(
        f"  of the later-wave rows, {int((lagged & (waves > 0)).sum())} carry a "
        "non-zero lag; the rest enter both arms identically"
    )

    arms = {"lag": definition, "control": control_definition(definition)}
    folds, _subj = stratified_subject_folds(frame, K=K)
    sampling_cfg = sampling.get_sampling_configuration(sampling_config_name)

    per_arm: dict[str, list[pd.DataFrame]] = {name: [] for name in arms}
    fit_records: list[FoldFitRecord] = []

    for k, fold_subjects in enumerate(folds):
        rows = scored_rows(frame, fold_subjects)
        mask = holdout_mask(frame, fold_subjects, unit=unit)
        print(
            f"\n=== Fold {k}/{K} — {len(fold_subjects)} children, "
            f"{int(mask.sum())} rows out of the likelihood, "
            f"{len(rows)} scored ==="
        )
        if len(rows) == 0:
            print("  no later waves in this fold; skipped")
            continue
        for arm, arm_definition in arms.items():
            label = f"{model_key}_{arm}_fold{k}"
            started = time.perf_counter()
            marked = frame.copy()
            marked["holdout"] = mask
            clean = source_in_training(lagged, source_row, mask)
            trace, gate = fit_holdout_fold(
                arm_definition,
                marked,
                sampling_cfg,
                label=label,
                tmp_root=TMP_DIR,
                name_prefix="WAVEFWD",
            )
            scores = row_elpds(
                marked, trace, rows, definition, lagged, clean, engine=engine
            )
            scores.insert(0, "fold", k)
            per_arm[arm].append(scores)
            elapsed = time.perf_counter() - started
            record = FoldFitRecord(
                arm=arm,
                fold=k,
                n_holdout_rows=int(mask.sum()),
                n_scored_rows=len(rows),
                wall_seconds=elapsed,
                **fold_gate_fields(gate),
            )
            fit_records.append(record)
            print(
                f"  {arm:8s} {elapsed:6.1f}s  gate "
                f"{'PASS' if record.passed else 'FAIL'}  "
                f"max R-hat {record.max_rhat:.4f}  div {record.divergences}"
            )

    if not fit_records:
        raise SystemExit("No fold produced a scorable row.")

    long = pd.concat(
        [df.assign(arm=arm) for arm, frames in per_arm.items() for df in frames],
        ignore_index=True,
    )
    wide = wide_table(long, columns, tuple(arms))

    fits = pd.DataFrame([dataclasses.asdict(r) for r in fit_records])
    converged = bool(fits["passed"].all())
    if not converged:
        print("\n" + "!" * 74)
        print("!!! CONVERGENCE WARNING: at least one fold fit failed the gate.")
        print("!!! The elpd difference below must not be interpreted.")
        print("!" * 74)

    comparison = pd.DataFrame(
        [
            paired_difference(wide, column, restriction=restriction)
            for restriction in RESTRICTIONS
            for column in columns
        ]
    )
    comparison["all_folds_converged"] = converged
    comparison["holdout_unit"] = unit
    comparison["config"] = sampling_config_name
    comparison["model"] = model_key
    comparison["engine"] = engine

    summary = (
        long.groupby("arm")[list(columns)]
        .agg(["sum", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in col if part).strip("_") for col in summary.columns
    ]
    summary["all_folds_converged"] = converged

    paths = {
        f"wave_forward_row_elpds{suffix}.csv": wide,
        f"wave_forward_summary{suffix}.csv": summary,
        f"wave_forward_compare{suffix}.csv": comparison,
        f"wave_forward_fits{suffix}.csv": fits,
    }
    for name, table in paths.items():
        table.to_csv(os.path.join(OUT_DIR, name), index=False)

    spoken = comparison.iloc[0]
    print(
        f"\nSpoken elpd on {spoken['rows_scored']} rows, cross-lag minus "
        f"control: {spoken['elpd_diff']:+.2f} (SE {spoken['se']:.2f}) "
        f"over {int(spoken['n_rows'])} rows"
    )
    print(f"Written to {OUT_DIR}")

    # This script fits its own folds and reads no model of record, so its
    # provenance is the raw data it prepared them from -- the same claim
    # `kfold_loso.py` records.
    write_comparison_manifest(
        OUT_DIR,
        script="wave_forward_score.py",
        contributing={},
        outputs=written.written(),
        source_data_hash=source_data_hash(env.DATA_DIR),
        arguments=[
            model_key,
            f"K={K}",
            f"config={sampling_config_name}",
            f"holdout_unit={unit}",
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="vg16",
        choices=sorted(CROSS_LAG_MODELS),
        help=(
            "Which cross-lag model to score. Derived from the registry: a model "
            "carrying a cross-lag field appears here without a list to update."
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--config", default="test")
    parser.add_argument(
        "--holdout-unit",
        default="later-waves",
        choices=("later-waves", "child"),
        help=(
            "later-waves keeps each fold child's first administration in "
            "training, so the lag source for every scored row is data the model "
            "saw; child removes the whole child, making the lag source unseen "
            "and the child effect a prior draw."
        ),
    )
    parser.add_argument("--suffix", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        model_key=args.model,
        K=args.folds,
        sampling_config_name=args.config,
        unit=args.holdout_unit,
        suffix=args.suffix,
    )
