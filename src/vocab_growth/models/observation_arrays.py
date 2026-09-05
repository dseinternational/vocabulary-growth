# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The observation arrays and masks a bivariate fit's likelihood is built from.

The first of issue #273's "pure and stable units": everything the bivariate
random-effect builder derives from the analysis frame before any PyMC object
exists -- the ages, the two outcomes' observed counts, the row indices each
likelihood factor covers, the training and held-out masks, and the nested
parent/child coupling between comprehension and speech.

It was seventy lines at the top of a 743-line function, mixed with the graph it
feeds. Separated, it is a pure function of ``(analysis_df, definition,
n_trials)`` that returns a frozen record, so the parts most likely to be got
wrong can be tested directly:

* the **spoken likelihood mask** must mark the rows the likelihood actually
  carries, not every row with a spoken count. The paired-only fallback drops the
  marginal rows, and storing the unfiltered mask made every paired-only fit fail
  at calibration -- after sampling, before the trace was written (issue #266
  finding 3);
* the **count validation** must run before the integer cast, because NumPy
  truncates silently and 810.9 or -0.1 would land inside the post-cast bounds
  check (issues #236, #240);
* the **held-out mask** keeps its rows in observation space, so their
  deterministics are still computed at their own ages, while excluding them from
  the likelihood -- which is what makes a K-fold LOSO subject's random effect a
  draw from the prior rather than a fitted value.

Numpy and pandas only. No PyMC, no context, no printing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vocab_growth.models.build_utils import require_valid_counts
from vocab_growth.models.likelihood_utils import (
    SPOKEN_FALLBACK_PAIRED_ONLY,
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

    spoken_spec: object
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
    """Derive the likelihood's arrays and masks from the prepared frame.

    ``use_subject_codes`` comes from the resolved
    :class:`~vocab_growth.models.subject_effects.SubjectEffectPlan` rather than
    being re-derived here: which outcomes carry a child effect is that plan's
    question, and asking it twice is how the two answers come apart.

    **Frame preconditions.** This is the *bivariate-RE* engine's derivation, not a
    general one, and it reads two columns beyond the two outcomes:

    * ``study_code`` -- so a frame without study codes raises ``KeyError``. VG05's
      frame has none and VG05 has no study effect, which is why the plain bivariate
      engine does not call this. Do not add the column to VG05's frame to make it
      fit: ``analysis_frames.analysis_frame_hash`` hashes the schema, so a new
      column stales VG05's fitted output.
    * ``holdout`` is optional and defaults to all-False.

    Routing another engine through this function therefore needs a parameter for the
    study codes, not a one-line call; and the trivariate and joint cases want
    *sibling* functions here rather than a generalisation of this one. Its only
    caller is `common_bivariate_re.build_model_re`, which eleven registered models
    run: VG07-VG10, VG13, VG16 and VG19-VG23.
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


#: Contrast coding for the exploratory sex-shift variant of VG20 (issue #295):
#: girls ``+1/2``, boys ``-1/2``. Centred so the population curve is the
#: sex-balanced average and each coefficient is the girl-minus-boy difference.
SEX_CONTRAST: dict[str, float] = {"F": 0.5, "M": -0.5}


def sex_contrast_codes(analysis_df: pd.DataFrame) -> np.ndarray:
    """The per-row sex contrast the sex-shift variant multiplies its coefficients by.

    Refuses a frame with any row lacking a recorded sex, or carrying a value other
    than the loader's ``'F'``/``'M'``, and a frame in which a retained child
    carries two values: the frame builder's ``sex_known_only`` restriction is what
    guarantees the first two, and the loader's provenance rule (which drops the one
    batch of us_01 rows whose sex disagreed within a child) the third. Each would
    otherwise fit silently with a covariate that is zero, or wrong, for some rows.
    """
    if "sex" not in analysis_df.columns:
        raise KeyError(
            "The frame carries no `sex` column; the sex-shift variant needs "
            "`sex_known_only` on its definition so the frame builder loads it."
        )
    sex = analysis_df["sex"]
    missing = int(sex.isna().sum())
    if missing:
        raise ValueError(f"{missing} rows have no recorded sex; the contrast is undefined for them.")
    unexpected = sorted(set(sex.unique()) - set(SEX_CONTRAST))
    if unexpected:
        raise ValueError(f"Unexpected sex codes {unexpected}; expected {sorted(SEX_CONTRAST)}.")
    if {"study", "subject_id"}.issubset(analysis_df.columns):
        per_child = analysis_df.groupby(["study", "subject_id"], sort=False)["sex"].nunique()
        inconsistent = int((per_child > 1).sum())
        if inconsistent:
            raise ValueError(
                f"{inconsistent} children carry more than one sex value across their "
                "administrations; sex is a child-level covariate."
            )
    return sex.map(SEX_CONTRAST).to_numpy(dtype=float)
