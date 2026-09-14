# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare observation arrays for the bivariate and joint likelihoods.

An observation row, a child and a study are different units. The records below
keep their indices separate and document each array's shape. Preparation keeps
held-out rows available for prediction, while excluding them from likelihoods.

Validate counts before converting them to integers. Build likelihood masks
after applying the missing-parent treatment. For composition rows, use the
cell likelihood in place of separate signed and spoken likelihoods.

These functions construct no PyMC variables and write no reports. Earlier
masking defects are recorded in notes/202609131044-model-review-implementation.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vocab_growth.models.build_utils import (
    require_integral_counts,
    require_valid_counts,
)
from vocab_growth.models.likelihood_utils import (
    SPOKEN_FALLBACK_PAIRED_ONLY,
    NestedOutcomeSpec,
    nested_outcome_spec,
    resolve_fallback_treatment,
)


@dataclass(frozen=True)
class BivariateObservations:
    """Everything the bivariate likelihood is assembled from, and nothing else."""

    X_obs: np.ndarray
    """Ages in months, shaped ``(n, 1)`` as the GP grids expect."""

    n: int
    """Rows in the analysis frame — the length of the ``obs_id`` dimension."""

    n_trials: int

    has_u: np.ndarray
    has_s: np.ndarray
    """Rows with a recorded count, before the held-out mask is applied."""

    holdout: np.ndarray
    """Rows kept in observation space but excluded from every likelihood."""

    y_u_observed: np.ndarray
    idx_u: np.ndarray
    """Comprehension counts, and the rows of the frame they come from."""

    y_s_observed: np.ndarray
    idx_s: np.ndarray
    """Speech counts and rows, **after** the fallback treatment has been applied."""

    has_u_likelihood: np.ndarray
    """Which rows the comprehension likelihood covers, over all ``n`` rows.

    ``has_u & ~holdout``. Stored in the trace as ``obs_u_mask``. Unlike the
    spoken side there is no fallback treatment that drops rows, so it has no
    second definition -- named as a likelihood mask anyway, so the two sides
    read the same way and a reader does not have to know that asymmetry."""

    has_s_likelihood: np.ndarray
    """Which rows the spoken likelihood covers, as a mask over all ``n`` rows.

    Stored in the trace and read by calibration, extraction, LOO and the
    recovery harness. Equal to ``has_s & ~holdout`` under every treatment except
    paired-only, which drops the marginal fallback rows."""

    spoken_spec: NestedOutcomeSpec
    """The resolved nested parent/child coupling (``likelihood_utils``)."""

    spoken_fallback: str
    n_fallback_dropped: int
    """How the rows with no usable comprehension count are treated, and how many
    the paired-only treatment removed from the spoken likelihood."""

    study_codes: np.ndarray
    n_studies: int

    subject_codes: np.ndarray | None
    n_subjects: int
    """``None`` and ``0`` when no outcome carries a child effect."""

    @property
    def n_u(self) -> int:
        return len(self.y_u_observed)

    @property
    def n_s(self) -> int:
        return int(self.spoken_spec.n_observed)


def prepare_bivariate_observations(
    analysis_df: pd.DataFrame,
    definition,
    *,
    n_trials: int,
    use_subject_codes: bool,
) -> BivariateObservations:
    """Derive the bivariate study-effect likelihood's arrays from a prepared frame.

    The frame needs age, understood, spoken and study_code columns. Child effects
    also need subject_code. An optional holdout column excludes rows from the
    likelihood without removing their predictions. ``use_subject_codes`` comes
    from the resolved child-effect plan.

    The plain VG05 frame has no study codes and uses its own preparation. Adding
    columns merely to reuse this helper would change that frame's stored hash.
    """
    has_u = analysis_df["understood"].notna().values
    has_s = analysis_df["spoken"].notna().values

    # Held-out rows stay in observation space -- `f_u_obs`, `h_obs` and the rest
    # are still computed at their ages -- but leave every likelihood, so a
    # held-out subject's random effect is a draw from the prior. That is exactly
    # what K-fold LOSO needs.
    if "holdout" in analysis_df.columns:
        holdout = analysis_df["holdout"].fillna(False).astype(bool).values
    else:
        holdout = np.zeros(len(analysis_df), dtype=bool)
    has_u_train = has_u & ~holdout
    has_s_train = has_s & ~holdout

    X_obs = np.asarray(analysis_df["age"], dtype=float).reshape(-1, 1)
    y_u_values = np.asarray(analysis_df.loc[has_u_train, "understood"], dtype=float)
    # Validate BEFORE the integer cast: NumPy's cast truncates silently, so a
    # fractional or out-of-range understood count would corrupt the likelihood
    # without a trace -- a post-cast bound cannot catch 810.9 or -0.1,
    # which truncate into range. The spoken side gets the same
    # finite/integral/range checks from `nested_outcome_spec` (#240, #236).
    require_valid_counts(y_u_values, "understood", n_trials)
    y_u_observed = y_u_values.astype(int)

    spoken_spec = nested_outcome_spec(
        analysis_df,
        parent_col="understood",
        outcome_col="spoken",
        n_trials=n_trials,
        eligible_mask=~holdout,
    )
    if not np.array_equal(spoken_spec.indices, np.flatnonzero(has_s_train)):
        raise ValueError("Spoken likelihood rows do not match the training-data mask.")
    # The mask check above runs against the unfiltered spec, so it still tests
    # what it was written to test under every treatment.
    spoken_fallback = resolve_fallback_treatment(definition)
    n_fallback_dropped = 0
    if spoken_fallback == SPOKEN_FALLBACK_PAIRED_ONLY:
        n_fallback_dropped = spoken_spec.n_marginal
        spoken_spec = spoken_spec.conditional_only()

    n = len(X_obs)
    # The stored spoken mask must mark the LIKELIHOOD rows, not every row with a
    # spoken observation: the paired-only treatment drops the marginal fallback
    # rows from the spoken likelihood, and every downstream consumer of
    # `obs_s_mask` -- calibration's age alignment, extraction's scatter, LOO's
    # per-administration alignment, the recovery harness's row masks -- needs the
    # rows the likelihood actually carries. Storing the unfiltered mask made
    # every paired-only fit fail at calibration, after sampling and before the
    # trace was saved (issue #266 finding 3). Under the other treatments no rows
    # are dropped, so this equals `has_s_train` exactly.
    has_s_likelihood = np.zeros(n, dtype=bool)
    has_s_likelihood[spoken_spec.indices] = True

    study_codes = np.asarray(analysis_df["study_code"], dtype=int)
    if use_subject_codes:
        subject_codes = np.asarray(analysis_df["subject_code"], dtype=int)
        n_subjects = int(subject_codes.max()) + 1
    else:
        subject_codes = None
        n_subjects = 0

    return BivariateObservations(
        X_obs=X_obs,
        n=n,
        n_trials=n_trials,
        has_u=has_u,
        has_s=has_s,
        holdout=holdout,
        y_u_observed=y_u_observed,
        idx_u=np.where(has_u_train)[0],
        has_u_likelihood=has_u_train,
        y_s_observed=spoken_spec.observed,
        idx_s=spoken_spec.indices,
        has_s_likelihood=has_s_likelihood,
        spoken_spec=spoken_spec,
        spoken_fallback=spoken_fallback,
        n_fallback_dropped=n_fallback_dropped,
        study_codes=study_codes,
        n_studies=int(study_codes.max()) + 1,
        subject_codes=subject_codes,
        n_subjects=n_subjects,
    )


#: Contrast coding for the sex covariate (issues #295, #324): girls ``+1/2``,
#: boys ``-1/2``, and a child of unrecorded sex ``0``. Zero is the midpoint on
#: the logit scale, not generally the average probability. Each coefficient is a
#: girl-minus-boy logit difference.
SEX_CONTRAST: dict[str, float] = {"F": 0.5, "M": -0.5}


def sex_contrast_codes(
    analysis_df: pd.DataFrame, *, allow_unknown: bool = False
) -> np.ndarray:
    """The per-row sex contrast the sex coefficients multiply.

    Sex is a child-level covariate recorded per administration, so it is resolved
    per child first: a child whose rows carry one recorded value takes it on every
    row, including a row where it was left blank, and a child carrying **two**
    values is refused rather than coded either way. Any value other than the
    loaders' ``'F'``/``'M'`` is refused too.

    With ``allow_unknown`` (the covariate design, #324) a child with no recorded
    value on any row is coded ``0``. Without it (the ``sex_known_only`` control
    arm) such a row is refused, because that restriction is what guarantees there
    are none, and a covariate silently zero for rows the arm meant to exclude would
    fit without complaint.
    """
    if "sex" not in analysis_df.columns:
        raise KeyError(
            "The frame carries no `sex` column; a sex term needs `sex_effect_sigma` "
            "or `sex_known_only` on its definition so the frame builder loads it."
        )
    sex = analysis_df["sex"]
    unexpected = sorted(set(sex.dropna().unique()) - set(SEX_CONTRAST))
    if unexpected:
        raise ValueError(f"Unexpected sex codes {unexpected}; expected {sorted(SEX_CONTRAST)}.")
    if {"study", "subject_id"}.issubset(analysis_df.columns):
        keys = [analysis_df["study"].astype(str), analysis_df["subject_id"].astype(str)]
        per_child = sex.groupby(keys, sort=False).nunique()
        inconsistent = int((per_child > 1).sum())
        if inconsistent:
            raise ValueError(
                f"{inconsistent} children carry more than one sex value across their "
                "administrations; sex is a child-level covariate."
            )
        sex = sex.groupby(keys, sort=False).transform("first")
    missing = int(sex.isna().sum())
    if missing and not allow_unknown:
        raise ValueError(f"{missing} rows have no recorded sex; the contrast is undefined for them.")
    return sex.map(SEX_CONTRAST).fillna(0.0).to_numpy(dtype=float)


# Counts and probabilities use these orders throughout fitting and scoring.
CELL_NAMES = ["neither", "sign_only", "speak_only", "both"]
CELL_COLUMNS = ["understood_only", "signed_only", "spoken_only", "signed_spoken"]
PROD_CELL_NAMES = ["sign_only", "speak_only", "both"]
PROD_CELL_COLUMNS = ["prod_signed_only", "prod_spoken_only", "prod_signed_spoken"]


@dataclass(frozen=True)
class JointObservations:
    """Validated data for marginal outcomes and observed sign/speech partitions.

    ``X_obs`` contains ages in months with shape ``(n, 1)``. Each likelihood
    mask has length ``n`` and each index array selects rows of the full frame.
    Four-cell counts have shape ``(len(idx_cells), 4)`` and sum to understood.
    Three-cell counts have shape ``(len(idx_prod), 3)`` and sum to produced.
    Study and child codes identify groups; they are not observation indices.
    """

    X_obs: np.ndarray
    n: int
    n_trials: int
    idx_u: np.ndarray
    y_u: np.ndarray
    spoken_spec: NestedOutcomeSpec
    signed_spec: NestedOutcomeSpec
    spoken_fallback: str
    n_fallback_dropped: int
    has_u_likelihood: np.ndarray
    has_s_likelihood: np.ndarray
    has_sign_likelihood: np.ndarray
    has_cells_likelihood: np.ndarray
    has_prod_likelihood: np.ndarray
    idx_cells: np.ndarray
    cell_counts: np.ndarray
    cell_total: np.ndarray
    idx_prod: np.ndarray
    prod_counts: np.ndarray
    prod_total: np.ndarray
    study_codes: np.ndarray
    n_studies: int
    subject_codes: np.ndarray | None
    n_subjects: int

    @property
    def idx_s(self) -> np.ndarray:
        return self.spoken_spec.indices

    @property
    def idx_sign(self) -> np.ndarray:
        return self.signed_spec.indices

    @property
    def y_s(self) -> np.ndarray:
        return self.spoken_spec.observed

    @property
    def y_sign(self) -> np.ndarray:
        return self.signed_spec.observed


def prepare_joint_observations(
    df: pd.DataFrame,
    definition,
    *,
    n_trials: int,
    use_subject_codes: bool,
) -> JointObservations:
    """Prepare the joint likelihood without building a graph or writing reports.

    Four-cell rows use a composition likelihood in place of separate signed
    and spoken counts. Missing or unusable parent counts follow the definition's
    fallback treatment. No likelihood includes a held-out row.
    """
    has_u = df["understood"].notna().values
    has_cells = df["signed_spoken"].notna().values

    # Held-out rows retain predictions at their ages but supply no likelihood.
    if "holdout" in df.columns:
        holdout = df["holdout"].fillna(False).astype(bool).values
    else:
        holdout = np.zeros(len(df), dtype=bool)
    has_u_t = has_u & ~holdout
    has_cells_t = has_cells & ~holdout

    idx_u = np.where(has_u_t)[0]
    idx_cells = np.where(has_cells_t)[0]

    y_u_values = np.asarray(df.loc[has_u_t, "understood"], dtype=float)
    # Validate before casting: converting to integer silently truncates fractions.
    require_valid_counts(y_u_values, "understood", n_trials)
    y_u = y_u_values.astype(int)
    marginal_outcome_eligible = ~holdout & ~has_cells
    spoken_fallback = resolve_fallback_treatment(definition)
    spoken_spec = nested_outcome_spec(
        df,
        parent_col="understood",
        outcome_col="spoken",
        n_trials=n_trials,
        eligible_mask=marginal_outcome_eligible,
    )
    signed_spec = nested_outcome_spec(
        df,
        parent_col="understood",
        outcome_col="signed",
        n_trials=n_trials,
        eligible_mask=marginal_outcome_eligible,
    )

    expected_spoken = marginal_outcome_eligible & df["spoken"].notna().to_numpy()
    expected_signed = marginal_outcome_eligible & df["signed"].notna().to_numpy()
    if not np.array_equal(spoken_spec.indices, np.flatnonzero(expected_spoken)):
        raise ValueError("Spoken likelihood rows do not match the marginal-data mask.")
    if not np.array_equal(signed_spec.indices, np.flatnonzero(expected_signed)):
        raise ValueError("Signed likelihood rows do not match the marginal-data mask.")
    n_fallback_dropped = 0
    if spoken_fallback == SPOKEN_FALLBACK_PAIRED_ONLY:
        n_fallback_dropped = spoken_spec.n_marginal + signed_spec.n_marginal
        spoken_spec = spoken_spec.conditional_only()
        signed_spec = signed_spec.conditional_only()

    # Counts, coordinates and masks must all use the filtered likelihood rows.
    idx_s = spoken_spec.indices
    idx_sign = signed_spec.indices

    has_s_likelihood = np.zeros(len(df), dtype=bool)
    has_sign_likelihood = np.zeros(len(df), dtype=bool)
    has_s_likelihood[idx_s] = True
    has_sign_likelihood[idx_sign] = True

    cell_values = np.asarray(
        df.loc[has_cells_t, CELL_COLUMNS],
        dtype=float,
    )
    require_integral_counts(cell_values.ravel(), "four-cell counts")
    cell_counts = cell_values.astype(int)
    cell_total_values = np.asarray(df.loc[has_cells_t, "cell_total"], dtype=float)
    require_integral_counts(cell_total_values, "cell_total")
    cell_total = cell_total_values.astype(int)

    if cell_counts.size:
        if cell_counts.min() < 0:
            raise ValueError("negative four-cell count.")
        if not np.array_equal(cell_counts.sum(axis=1), cell_total):
            raise ValueError("four-cell counts do not sum to cell_total.")
        if cell_total.max() > n_trials:
            raise ValueError(f"four-cell total exceeds n_trials ({n_trials}).")

    has_prod = (
        df["prod_signed_spoken"].notna().values
        if "prod_signed_spoken" in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    has_prod_t = has_prod & ~holdout
    idx_prod = np.where(has_prod_t)[0]
    if idx_prod.size:
        prod_values = np.asarray(df.loc[has_prod_t, PROD_CELL_COLUMNS], dtype=float)
        require_integral_counts(prod_values.ravel(), "produced-cell counts")
        prod_counts = prod_values.astype(int)
        prod_total_values = np.asarray(df.loc[has_prod_t, "prod_total"], dtype=float)
        require_integral_counts(prod_total_values, "prod_total")
        prod_total = prod_total_values.astype(int)
        if prod_counts.min() < 0:
            raise ValueError("negative produced-cell count.")
        if not np.array_equal(prod_counts.sum(axis=1), prod_total):
            raise ValueError("produced-cell counts do not sum to prod_total.")
    else:
        prod_counts = np.zeros((0, 3), dtype=int)
        prod_total = np.zeros(0, dtype=int)

    study_codes = np.asarray(df["study_code"], dtype=int)
    n_studies = int(study_codes.max()) + 1

    if use_subject_codes:
        subject_codes = np.asarray(df["subject_code"], dtype=int)
        n_subjects = int(subject_codes.max()) + 1
    else:
        subject_codes = None
        n_subjects = 0
    X_obs = np.asarray(df["age"], dtype=float).reshape(-1, 1)

    return JointObservations(
        X_obs=X_obs,
        n=len(X_obs),
        n_trials=n_trials,
        idx_u=idx_u,
        y_u=y_u,
        spoken_spec=spoken_spec,
        signed_spec=signed_spec,
        spoken_fallback=spoken_fallback,
        n_fallback_dropped=n_fallback_dropped,
        has_u_likelihood=has_u_t,
        has_s_likelihood=has_s_likelihood,
        has_sign_likelihood=has_sign_likelihood,
        has_cells_likelihood=has_cells_t,
        has_prod_likelihood=has_prod_t,
        idx_cells=idx_cells,
        cell_counts=cell_counts,
        cell_total=cell_total,
        idx_prod=idx_prod,
        prod_counts=prod_counts,
        prod_total=prod_total,
        study_codes=study_codes,
        n_studies=n_studies,
        subject_codes=subject_codes,
        n_subjects=n_subjects,
    )
