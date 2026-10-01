# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
K-fold leave-one-subject-out (LOSO) gold-standard comparison of DS bivariate models.

Defaults to VG07/VG08/VG09, the set it was written for; ``--models`` selects any
subset of VG07-VG10, VG19, VG20 and VG22, and ``--suffix`` keeps a non-default
run's output from overwriting the default one's.

Two holdout units, because two different questions are asked of the same fits.
``--holdout-unit subject`` (the default, and everything this script did before
2026-09-09) removes every row of a fold's children: their effects are drawn from
the prior, and the score is "predict a child you have never seen". ``--holdout-unit
later-waves`` removes only a fold child's rows *after their first administration
wave*, keeping that first wave and every other child's later waves in training,
so the score is "given what this child was at their first visit, how well is their
next visit predicted". The second is `wave_forward_score.py`'s unit applied to two
*different models* rather than to one model with a coefficient removed, and it
exists for criterion 4 of the VG20/VG22 promotion decision
(``notes/202609031930-vg20-vg22-decision.md``), whose original ``us_03``
out-of-sample vehicle was invalidated when ``us_03`` was ingested on 2026-09-06 and
became 284 rows of both models' own training frames.

``--visit1-conditioning understood-only`` additionally masks the fold children's
first-wave ``spoken`` count to missing, so the child's effects are informed by
their first-visit comprehension alone. That is the conditioning criterion 4 names
-- it is where VG20 leads VG22 by 6.36 nats -- and it is the one the child-effect
correlation exists to serve, because the correlation's job is to carry a child's
comprehension standing into a prediction about their production. It is only
meaningful with ``--holdout-unit later-waves``. The frame already carries
comprehension-only administrations (``us_03`` is 284 of them), so masking is a
shape the loader and the likelihood already handle rather than a new code path.

Under ``later-waves`` the per-subject elpds are written twice: the total over both
held-out outcomes, and **comprehension only**, which is criterion 4's own quantity
(``*_understood_*`` in the file names).

Splits the unique DS subjects into K folds (stratified by study and
observation count -- 943 of them since the ``us_03`` ingestion, 510 before it),
then for each (model, fold) pair refits the model with
the fold's subjects' observations excluded from the likelihood (but kept
in `obs_id` space so f_u_obs, h_obs and so on are computed at their ages).
Subject REs for the held-out subjects are then drawn from their priors
during MCMC, which means the trace's `p_u_obs` and `p_s_obs` at held-out
observations are already the marginal posterior predictive probabilities
— no extra Monte Carlo integration is needed.

For each held-out subject the predictive log-density of their observed
counts is

    log p(y_subject | training_data, model)
        = logsumexp_{c, d} log p(y_subject | params_{c,d}, RE_{c,d})
          - log(n_chain * n_draw)

where the inner log p is the Beta-Binomial log-pmf summed over the
subject's held-out outcomes. Spoken counts use the nested S | U likelihood
when the observed counts are logically nested, and the marginal spoken
likelihood otherwise. The outer logsumexp marginalises over the joint posterior
over hyperparameters and held-out REs.

Every fold fit is screened by the canonical diagnostics scan
(``write_diagnostics_summary`` over every free RV, element-wise — the same
construction the fit pipeline's diagnostics stage uses), and the per-fold
verdict travels with the outputs: `kfold_loso_fits.csv` carries ``passed``,
``max_rhat``, ``min_ess``, ``divergences`` and ``bfmi_ok`` per (model, fold),
and the summary and pairwise tables carry ``all_folds_converged``. A model with
any failed fold is flagged rather than dropped, and its elpd must not be
interpreted. The hard-stop ``enforce_convergence_gate`` is deliberately not
called here: it is a no-op below the reporting tier, and a comparison wants the
verdict recorded, not the run aborted.

Outputs:

- `output/comparisons/kfold_loso_subject_elpds.csv`
- `output/comparisons/kfold_loso_summary.csv`
- `output/comparisons/kfold_loso_compare.csv`
- `output/comparisons/kfold_loso_fits.csv`

Typically developing reference models (#240)
--------------------------------------------

``--models`` also accepts the five typically developing reference models: VG11
and VG12 (univariate) and VG21, VG23 and VG26 (bivariate). Their PSIS-LOO is
unusable -- 37% to 59% of rows at Pareto k >= 0.7, because most of their
children are seen once (``notes/202609131214-held-out-validation-for-the-td-
models.md``) -- and this is the replacement. A run takes either population, never
both, and every model in one run must share one prepared frame, so a pairwise
difference is always on identical children.

Their frame comes from the definition (``analysis_frames.build_analysis_frame``),
not from the Down syndrome pool. Two holdout units are offered:

- ``--holdout-unit subject``: grouped K-fold by child, stratified as above.
- ``--holdout-unit study``: leave one study out, one fold per retained study. The
  held-out study is removed from the zero-sum study block, which is refitted on
  the remaining studies; its rows stay in observation space so that the
  population curve and dispersion are evaluated at their ages, and its effect is
  *integrated* over ``Normal(0, tau)`` -- the fitted between-study scale, as
  ``predict_new_study.py`` uses -- rather than read off a fitted value. Leaving
  the study in the zero-sum block would have fixed its offset at minus the sum
  of the others.

Scoring does not use the one child-effect draw per posterior draw that MCMC
leaves at a held-out row, which the Down syndrome path relies on. Each held-out
child's effects -- and, under ``study``, the study offset, which is Gaussian and
independent of them, so the two add into one Gaussian -- are integrated out per
posterior draw by adaptive Gauss-Hermite quadrature (nodes placed at the mode
and curvature of the child's own likelihood, as the engine's singleton
marginalisation does), and the posterior draws are then averaged on the
probability scale (log-mean-exp). The likelihood is the model's own:
Beta-Binomial with the age-varying ``kappa`` at the row, the sex contrast, and
for the bivariate models comprehension on the 810-item scale with speech
nested in it on the engine's paired-count rule. The fixed part of each row's
logit is read from the fold trace and checked against the trace's own
probabilities, so a graph term the scorer does not model fails loudly.

The study unit scores each child of the held-out study as a new child in a new
study. Children of one study share its offset, so their scores are dependent:
the standard error is computed over study totals, and with six to ten studies
it is itself rough. It is the per-child predictive, not the joint density of the
whole study.

Bivariate models write three quantities per child: the joint density of all
their counts, comprehension alone, and speech **conditional on the child's
observed comprehension** (the difference of the first two, which is the
conditional predictive with the mixture reweighted by comprehension, #39).

Definitions or fields the scorer does not implement are refused before any fit:
see :func:`td_scoring_refusal`.

TD outputs are written as ``kfold_loso_td_<unit>_*{suffix}.csv`` and recorded
under their own manifest label, so they never overwrite the Down syndrome tables.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd
import xarray as xr
from dse_research_utils.math.constants import EPSILON
from scipy.special import expit, gammaln, logsumexp
from scipy.stats import betabinom

import vocab_growth.data_utils as data_utils
from vocab_growth import environment as env
from vocab_growth.analysis_frames import analysis_frame_hash, build_analysis_frame
from vocab_growth.comparisons_provenance import (
    ComparisonOutputs,
    write_comparison_manifest,
)
from vocab_growth.fit_artifacts import source_data_hash
from vocab_growth.fold_fits import fit_holdout_fold, fold_gate_fields
from vocab_growth.models import subject_effects
from vocab_growth.models.catalogue import engine_for_definition
from vocab_growth.models.cross_lag import wave_index as subject_wave_index
from vocab_growth.models.definitions import (
    VG07,
    VG08,
    VG09,
    VG10,
    VG11,
    VG12,
    VG19,
    VG20,
    VG21,
    VG22,
    VG23,
    VG26,
    BivariateModelDefinition,
    Population,
)
from vocab_growth.models.likelihood_utils import (
    SPOKEN_FALLBACK_PRODUCT,
    nested_outcome_spec,
    resolve_fallback_treatment,
)
from vocab_growth.models.observation_arrays import sex_contrast_codes
from vocab_growth.models.sex_covariate import sex_effect_sigma
from vocab_growth.models.subject_effects import UNIVARIATE_OUTCOME, SubjectEffectKind
from vocab_growth.models.subject_marginal import standard_normal_quadrature

# Every model this script can compare. `build_model_re` dispatches on the
# definition's own fields — through `subject_effects.resolve` — so the
# child-slope (VG19), correlated-intercept (VG20) and low-rank-factor (VG22)
# structures need no separate builder here, which is the whole reason the
# comparison is one line of configuration rather than a second script.
#
# VG22 was added on 2026-09-09 for the VG20/VG22 promotion decision
# (`notes/202609031930-vg20-vg22-decision.md` criterion 3). That criterion asks
# for a paired `loo_compare` at more than four standard errors, and PSIS-LOO
# degenerates on every model in this family — 36% of VG22's joint observations
# and 24% of its comprehension observations exceed Pareto-k 0.7 or are
# non-finite on the 2026-09-08 `rep` fits, which `loo_compare.py` itself marks
# `[unusable]`. This script is the registered remedy for exactly that: it refits
# per fold rather than importance-weighting, so no Pareto diagnostic is
# involved. A held-out subject's factor scores stay at their `Normal(0, 1)`
# prior during MCMC exactly as the other structures' subject REs do, because
# `subject_factor_z` is indexed by `subject_id` and only the likelihood drops
# the fold's rows.
DS_AVAILABLE = {
    "VG07": VG07, "VG08": VG08, "VG09": VG09,
    "VG10": VG10, "VG19": VG19, "VG20": VG20, "VG22": VG22,
}
# The typically developing reference models (#240), scored by the integrated
# scorer below rather than by `holdout_subject_elpds`. VG13 is not here: it was
# superseded as the comparator by VG21 and remains only as VG23's nested null.
TD_AVAILABLE = {
    "VG11": VG11, "VG12": VG12, "VG21": VG21, "VG23": VG23, "VG26": VG26,
}
AVAILABLE = {**DS_AVAILABLE, **TD_AVAILABLE}
DEFAULT_MODELS = ("VG07", "VG08", "VG09")

# Derive the Beta-Binomial trial count from the model definitions rather than a
# literal (issue #131). Every model shares the common 810-item reference scale;
# assert they agree so a future divergence surfaces here.
_n_trials_set = {d.n_trials for d in AVAILABLE.values()}
assert len(_n_trials_set) == 1, (
    f"compared models disagree on n_trials ({sorted(_n_trials_set)}); "
    "kfold_loso assumes a single common trial count."
)
N_TRIALS = _n_trials_set.pop()

# `holdout_subject_elpds` reimplements the spoken likelihood in NumPy, and it
# implements one treatment of the fallback branch: the historical
# `product_marginal`. Scoring a model whose graph uses a different one would
# silently compare a held-out density the model never fitted, so refuse it here
# rather than in a code comment (#233, #236).
_unsupported_fallback = {
    name: resolve_fallback_treatment(definition)
    for name, definition in AVAILABLE.items()
    if resolve_fallback_treatment(definition) != SPOKEN_FALLBACK_PRODUCT
}
if _unsupported_fallback:
    raise NotImplementedError(
        "kfold_loso implements only the "
        f"{SPOKEN_FALLBACK_PRODUCT!r} spoken fallback; "
        f"{_unsupported_fallback} would be scored under the wrong likelihood."
    )
OUT_DIR = env.comparisons_output_dir()
KFOLD_TMP_DIR = os.path.join(env.output_root(), "kfold_tmp")

HOLDOUT_UNITS = ("subject", "later-waves", "study")
#: The units the typically developing path implements; `later-waves` is the Down
#: syndrome criterion-4 unit and `study` is typically developing only.
TD_HOLDOUT_UNITS = ("subject", "study")
VISIT1_CONDITIONINGS = ("all-outcomes", "understood-only")

#: Posterior draws the integrated scorer averages over, evenly thinned from the
#: fold's chains. The held-out density is a mean over draws, so thinning a long
#: chain costs Monte Carlo precision, not bias; 2,000 is the cap the 2026-09-23
#: corrections use for their new-child predictives.
DEFAULT_SCORE_DRAWS = 2000
#: Adaptive Gauss-Hermite nodes per effect dimension (11, or 11 x 11 = 121 for the
#: bivariate models' two child effects).
DEFAULT_QUADRATURE_NODES = 11


def wave_index(analysis_df: pd.DataFrame) -> np.ndarray:
    """0 for each child's first administration wave, 1 for the next, and so on.

    The definition lives beside ``iter_subject_age_waves`` in
    ``vocab_growth.models.cross_lag`` since the 2026-09-09 refit window; this is
    the frame-taking wrapper the fold code calls.
    """
    return subject_wave_index(
        analysis_df["subject_code"].to_numpy(), analysis_df["age"].to_numpy()
    )


def build_fold_frame(
    analysis_df: pd.DataFrame,
    fold_subjects: np.ndarray,
    waves: np.ndarray,
    holdout_unit: str,
    visit1_conditioning: str,
) -> pd.DataFrame:
    """The fold's frame, carrying the ``holdout`` column its unit implies.

    ``holdout`` marks rows that leave the likelihood but stay in ``obs_id`` space.
    Under ``later-waves`` that is a fold child's rows after their first wave;
    under ``subject`` it is all of their rows, which is what this script did
    before the unit existed.

    ``understood-only`` conditioning is expressed by masking the fold children's
    first-wave ``spoken`` to missing rather than by a second holdout flag, because
    the holdout is per row and this is per outcome -- and because a row with a
    comprehension count and no production count is a shape the frame already
    carries in quantity, so nothing downstream meets a new case.
    """
    frame = analysis_df.copy()
    in_fold = frame["subject_code"].isin(fold_subjects).to_numpy()
    if holdout_unit == "subject":
        frame["holdout"] = in_fold
        return frame

    frame["holdout"] = in_fold & (waves > 0)
    if visit1_conditioning == "understood-only":
        frame.loc[in_fold & (waves == 0), "spoken"] = np.nan
    return frame


@dataclass(frozen=True)
class FoldFitRecord:
    model_short: str
    fold: int
    n_holdout_subjects: int
    n_holdout_obs_u: int
    n_holdout_obs_s: int
    wall_seconds: float
    # Convergence verdict for this fold's fit, from the canonical diagnostics
    # scan. A fold that failed the gate poisons its model's pooled elpd, so the
    # verdict must travel with the timing record into kfold_loso_fits.csv.
    passed: bool
    max_rhat: float | None
    min_ess: float | None
    divergences: int | None
    bfmi_ok: bool | None


def model_convergence_flags(fit_records: list[FoldFitRecord]) -> dict[str, bool]:
    """Per-model AND over every fold's convergence gate.

    A model's pooled elpd sums over all folds, so one non-converged fold makes
    the total uninterpretable — the flag is per model, not per fold.
    """
    flags: dict[str, bool] = {}
    for record in fit_records:
        flags[record.model_short] = flags.get(record.model_short, True) and bool(
            record.passed
        )
    return flags


# ============================================================
# Data + folds
# ============================================================


def load_analysis_frame() -> pd.DataFrame:
    df = data_utils.load_combined_data()
    # `sex` travels with the frame because VG20 carries the sex covariate from
    # 2026-09-13 (#324) and its build stage refuses a frame without the column;
    # a model without the covariate never reads it.
    analysis_df = df[["age", "understood", "spoken", "study", "subject_id", "sex"]].copy()
    analysis_df = analysis_df.dropna(subset=["age"])
    has_u = analysis_df["understood"].notna()
    has_s = analysis_df["spoken"].notna()
    analysis_df = analysis_df[has_u | has_s].reset_index(drop=True)
    data_utils.validate_subject_ids(analysis_df)

    unique_studies = sorted(analysis_df["study"].unique())
    study_map = {s: i for i, s in enumerate(unique_studies)}
    analysis_df["study_code"] = analysis_df["study"].map(study_map).astype(int)

    subj_keys = (
        analysis_df["study"].astype(str) + "::" + analysis_df["subject_id"].astype(str)
    )
    unique_subjects = sorted(subj_keys.unique())
    subject_map = {s: i for i, s in enumerate(unique_subjects)}
    analysis_df["subject_code"] = subj_keys.map(subject_map).astype(int)
    return analysis_df


def sex_covariate_mismatch(specs) -> str | None:
    """Say so when the compared models disagree about the sex covariate.

    VG20 carries it from 2026-09-13 and the development steps it is compared
    with -- VG22 for the VG20/VG22 decision's criterion 3 among them -- do not, so
    a paired difference between them measures the child structure and the
    covariate together. Returns ``None`` when every model agrees.
    """
    from vocab_growth.models.sex_covariate import sex_effect_sigma

    settings = {name: sex_effect_sigma(definition) for name, definition in specs}
    if len(set(settings.values())) <= 1:
        return None
    carrying = sorted(name for name, sigma in settings.items() if sigma is not None)
    lacking = sorted(name for name, sigma in settings.items() if sigma is None)
    return (
        f"the compared models differ in the sex covariate ({', '.join(carrying)} carry it; "
        f"{', '.join(lacking)} do not), so a paired difference between them is not a "
        "one-factor contrast of their child structures"
    )


def stratified_subject_folds(
    analysis_df: pd.DataFrame, K: int = 5, seed: int = 47
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Assign each subject to one of K folds, stratified by (study, n_obs_bin)."""
    subj = analysis_df.groupby("subject_code").agg(
        study_code=("study_code", "first"),
        n_obs=("age", "size"),
    ).reset_index()
    # Bin observation counts.
    def _bin(n):
        if n == 1:
            return "1"
        if n == 2:
            return "2"
        if n == 3:
            return "3"
        return "4+"

    subj["n_obs_bin"] = subj["n_obs"].apply(_bin)
    subj["stratum"] = (
        subj["study_code"].astype(str) + "_" + subj["n_obs_bin"]
    )

    rng = np.random.default_rng(seed)
    fold_of = np.zeros(len(subj), dtype=int)
    for _stratum, group in subj.groupby("stratum"):
        idxs = np.asarray(group.index.to_numpy(), dtype=np.int64).copy()
        rng.shuffle(idxs)
        for k, i in enumerate(idxs):
            fold_of[i] = k % K
    subj["fold"] = fold_of
    folds = [
        subj.loc[subj["fold"] == k, "subject_code"].to_numpy() for k in range(K)
    ]
    return folds, subj


# ============================================================
# Per-fold fit (minimal pipeline)
# ============================================================


def fit_fold(
    definition: BivariateModelDefinition,
    analysis_df_with_holdout: pd.DataFrame,
    sampling_cfg: sampling.SamplingConfiguration,
    label: str,
) -> tuple[xr.DataTree, int, dict]:
    """:func:`vocab_growth.fold_fits.fit_holdout_fold`, plus this script's row count.

    The mechanics moved to :mod:`vocab_growth.fold_fits` when
    ``wave_forward_score.py`` needed the same fold fit for a different holdout
    rule; the hand copy made for it read the energy verdict from the wrong key
    and reported every fold's BFMI as passing.
    """
    trace, gate = fit_holdout_fold(
        definition,
        analysis_df_with_holdout,
        sampling_cfg,
        label=label,
        tmp_root=KFOLD_TMP_DIR,
        name_prefix="KFOLD",
    )
    return trace, len(analysis_df_with_holdout), gate


# ============================================================
# Held-out predictive log-density
# ============================================================


def holdout_subject_elpds(
    analysis_df: pd.DataFrame,
    trace: xr.DataTree,
    holdout_subject_codes: np.ndarray,
    row_mask: np.ndarray | None = None,
    outcomes: tuple[str, ...] = ("understood", "spoken"),
) -> dict[int, float]:
    """Marginal predictive log-density per held-out subject.

    ``row_mask`` restricts the sum to particular rows, which is what the
    ``later-waves`` unit needs: a fold child keeps their first wave in the
    likelihood, so scoring all of their rows would be scoring training data.
    ``None`` sums every row of the subject, which is the ``subject`` unit's
    behaviour and this function's original one.

    ``outcomes`` restricts which of the two counts contribute. Criterion 4 of the
    VG20/VG22 decision is stated on second-visit **comprehension**, so it reads
    the ``("understood",)`` call; the default sums both, unchanged.
    """
    p_u_obs = trace.posterior["p_u_obs"].values
    p_s_obs = trace.posterior["p_s_obs"].values
    q_obs = trace.posterior["q_obs"].values
    kappa_u_obs = trace.posterior["kappa_u_obs"].values
    kappa_s_obs = trace.posterior["kappa_s_obs"].values

    n_chain, n_draw, _ = p_u_obs.shape
    log_NK = math.log(n_chain * n_draw)
    elpd: dict[int, float] = {}

    spoken_spec = nested_outcome_spec(
        analysis_df,
        parent_col="understood",
        outcome_col="spoken",
        n_trials=N_TRIALS,
    )
    spoken_observed = np.full(len(analysis_df), -1, dtype=int)
    spoken_trials = np.full(len(analysis_df), N_TRIALS, dtype=int)
    spoken_is_conditional = np.zeros(len(analysis_df), dtype=bool)
    spoken_observed[spoken_spec.indices] = spoken_spec.observed
    spoken_trials[spoken_spec.indices] = spoken_spec.trials
    spoken_is_conditional[spoken_spec.indices] = spoken_spec.is_conditional

    holdout_set = set(int(s) for s in holdout_subject_codes)
    subject_codes = analysis_df["subject_code"].to_numpy(dtype=int)
    scored = (
        np.ones(len(analysis_df), dtype=bool) if row_mask is None
        else np.asarray(row_mask, dtype=bool)
    )
    score_u = "understood" in outcomes
    score_s = "spoken" in outcomes
    for s_code in holdout_set:
        log_lik = np.zeros((n_chain, n_draw), dtype=np.float64)
        rows_scored = 0
        for idx in np.flatnonzero((subject_codes == s_code) & scored):
            row = analysis_df.iloc[idx]
            rows_scored += 1
            if score_u and pd.notna(row["understood"]):
                y = int(row["understood"])
                p = np.clip(p_u_obs[:, :, idx], 1e-12, 1 - 1e-12)
                k = kappa_u_obs[:, :, idx]
                log_lik += betabinom.logpmf(y, N_TRIALS, p * k, (1 - p) * k)
            if score_s and pd.notna(row["spoken"]):
                y = spoken_observed[idx]
                if spoken_is_conditional[idx]:
                    p = np.clip(q_obs[:, :, idx], 1e-12, 1 - 1e-12)
                else:
                    p = np.clip(p_s_obs[:, :, idx], 1e-12, 1 - 1e-12)
                k = kappa_s_obs[:, :, idx]
                log_lik += betabinom.logpmf(
                    y, spoken_trials[idx], p * k, (1 - p) * k
                )
        # A fold child with no scored row contributes nothing and must not enter
        # the table as a 0.0, which would read as a perfect prediction. Under the
        # later-waves unit this is every child seen once -- the majority of the
        # pool -- so the distinction is not an edge case.
        if rows_scored:
            elpd[s_code] = float(logsumexp(log_lik.ravel()) - log_NK)
    return elpd


# ============================================================
# Typically developing models: frames, folds and refusals (#240)
# ============================================================


def is_td_model(definition) -> bool:
    return definition.population is Population.TYPICALLY_DEVELOPING


def td_engine(definition) -> str:
    """``"univariate_re"`` or ``"bivariate_re"``: the engines the scorer implements."""
    return engine_for_definition(definition).name


def td_scoring_refusal(definition, holdout_unit: str) -> str | None:
    """Why the integrated scorer cannot score ``definition``, or ``None``.

    The scorer evaluates the model's likelihood in NumPy, so every graph feature
    it does not reproduce has to be refused here rather than scored under a
    density the model never fitted -- the same rule this script already applies
    to the spoken fallback. What it implements: the univariate and bivariate
    random-effect engines; one time-constant child effect per outcome (the
    variance-partition scale included), optionally correlated through
    ``rho_uq``; zero-sum study intercepts; the sex contrast; and the
    ``product_marginal`` spoken fallback.
    """
    name = definition.model_id
    if holdout_unit not in TD_HOLDOUT_UNITS:
        return (
            f"{name}: the typically developing scorer implements the holdout units "
            f"{TD_HOLDOUT_UNITS}, not {holdout_unit!r}."
        )
    engine = td_engine(definition)
    if engine not in ("univariate_re", "bivariate_re"):
        return f"{name}: no integrated scorer for the {engine!r} engine."
    plan = subject_effects.resolve(definition)
    allowed = {SubjectEffectKind.CONSTANT, SubjectEffectKind.VARIANCE_PARTITION}
    outcomes = (UNIVARIATE_OUTCOME,) if engine == "univariate_re" else ("u", "q")
    for outcome in outcomes:
        kind = plan[outcome].kind
        if kind not in allowed:
            return (
                f"{name}: the {plan[outcome].scale_name} child effect is "
                f"{kind.value!r}; the scorer integrates a time-constant child effect "
                "per outcome only (no age-varying scale, child slope or factor)."
            )
    if plan.factor is not None:
        return f"{name}: the low-rank child factor is not implemented by the scorer."
    if getattr(definition, "singleton_marginalisation", None) is not None:
        return (
            f"{name}: singleton marginalisation leaves no explicit child effect to "
            "remove from the held-out rows' logits."
        )
    if getattr(definition, "one_observation_per_subject", False):
        return f"{name}: the single-administration frame has no child effects to integrate."
    if any(
        getattr(definition, field, False)
        for field in ("use_cross_lag", "use_sign_cross_lag")
    ):
        return f"{name}: a cross-lag reads earlier counts; use wave_forward_score.py."
    if engine == "bivariate_re":
        treatment = resolve_fallback_treatment(definition)
        if treatment != SPOKEN_FALLBACK_PRODUCT:
            return (
                f"{name}: spoken_fallback {treatment!r}; the scorer implements only "
                f"{SPOKEN_FALLBACK_PRODUCT!r}."
            )
    if (
        holdout_unit == "study"
        and getattr(definition, "study_age_slope_sigma", None) is not None
    ):
        return (
            f"{name}: per-study age slopes would need a held-out study's slope "
            "integrated too, which the study unit does not implement."
        )
    return None


def load_td_frame(model_key: str) -> pd.DataFrame:
    """The prepared frame ``model_key``'s engine fits, rebuilt from its definition."""
    frame, _ = build_analysis_frame(model_key.lower(), AVAILABLE[model_key.upper()])
    required = {"age", "study", "study_code", "subject_code", "subject_key"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{model_key}'s frame lacks {sorted(missing)}.")
    return frame


def study_folds(frame: pd.DataFrame) -> list[str]:
    """One fold per retained study, in the frame's own (sorted) study order."""
    return sorted(frame["study"].unique())


def build_td_fold_frame(
    frame: pd.DataFrame, holdout_unit: str, held: np.ndarray | str
) -> pd.DataFrame:
    """The fold's frame: ``holdout`` marks the rows that leave the likelihood.

    ``subject``: every row of the fold's children (``held`` is their codes).

    ``study``: every row of study ``held``. The remaining studies are recoded
    ``0 .. K-2`` in their original order, so the refitted zero-sum block is over
    the training studies only; the held-out rows carry the placeholder code 0,
    which the scorer removes from their logits before integrating a fresh study
    offset. ``study_code_full`` keeps the frame's original code either way, and
    the ``study`` name column is left untouched because the sex contrast is
    resolved per (study, child).
    """
    fold = frame.copy()
    fold["study_code_full"] = fold["study_code"]
    if holdout_unit == "subject":
        fold["holdout"] = fold["subject_code"].isin(np.asarray(held)).to_numpy()
        return fold
    if holdout_unit != "study":
        raise ValueError(f"unknown typically developing holdout unit {holdout_unit!r}")
    in_study = (fold["study"] == held).to_numpy()
    if not in_study.any():
        raise ValueError(f"study {held!r} has no rows in the frame")
    training = sorted(fold.loc[~in_study, "study"].unique())
    if len(training) < 2:
        raise ValueError("a study fold needs at least two training studies")
    recode = {study: code for code, study in enumerate(training)}
    fold["holdout"] = in_study
    fold["study_code"] = fold["study"].map(recode).fillna(0).astype(int)
    return fold


# ============================================================
# Typically developing models: the integrated scorer
# ============================================================

#: Finite-difference step, Newton iterations and clamps for placing the
#: quadrature nodes, in standard-normal units of the effect. They mirror the
#: engine's own adaptive rule (`subject_marginal`).
_FD_STEP = 1e-3
_NEWTON_STEPS = 20
_BACKTRACK_HALVINGS = 12
_NEWTON_MAX_STEP = 2.0
_MODE_CLAMP = 8.0
#: Elements per working array: (draws x rows x nodes) in one batch.
_BATCH_ELEMENTS = 6_000_000


def betabinomial_logpmf(y, n, p, kappa):
    """Beta-Binomial log pmf with mean ``p`` and concentration ``kappa``.

    ``alpha = p kappa`` and ``beta = (1 - p) kappa``, as every engine builds them.
    """
    a = p * kappa
    b = (1.0 - p) * kappa
    return (
        gammaln(n + 1.0) - gammaln(y + 1.0) - gammaln(n - y + 1.0)
        + gammaln(y + a) + gammaln(n - y + b) - gammaln(n + a + b)
        - gammaln(a) - gammaln(b) + gammaln(a + b)
    )


def _clip(p):
    return np.clip(p, EPSILON, 1.0 - EPSILON)


def _hermite_rule(n_nodes: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Product Gauss-Hermite rule for ``N(0, I_dim)``: nodes ``(M, dim)``, log weights."""
    x, log_w = standard_normal_quadrature(n_nodes)
    if dim == 1:
        return x[:, None], log_w
    grid = np.stack(np.meshgrid(x, x, indexing="ij"), axis=-1).reshape(-1, 2)
    return grid, (log_w[:, None] + log_w[None, :]).reshape(-1)


def _stencil(dim: int) -> np.ndarray:
    h = _FD_STEP
    if dim == 1:
        return np.array([[0.0], [h], [-h]])
    return np.array(
        [[0, 0], [h, 0], [-h, 0], [0, h], [0, -h], [h, h], [h, -h], [-h, h], [-h, -h]],
        dtype=float,
    )


def _gradient_precision(values: np.ndarray, dim: int):
    """Gradient and negative Hessian of a log-integrand from its stencil values."""
    h = _FD_STEP
    if dim == 1:
        f0, fp, fm = values[..., 0], values[..., 1], values[..., 2]
        grad = ((fp - fm) / (2 * h))[..., None]
        prec = (-(fp - 2 * f0 + fm) / h**2)[..., None, None]
        return grad, prec
    f0 = values[..., 0]
    g1 = (values[..., 1] - values[..., 2]) / (2 * h)
    g2 = (values[..., 3] - values[..., 4]) / (2 * h)
    h11 = (values[..., 1] - 2 * f0 + values[..., 2]) / h**2
    h22 = (values[..., 3] - 2 * f0 + values[..., 4]) / h**2
    h12 = (values[..., 5] - values[..., 6] - values[..., 7] + values[..., 8]) / (4 * h**2)
    grad = np.stack([g1, g2], axis=-1)
    prec = -np.stack([np.stack([h11, h12], -1), np.stack([h12, h22], -1)], axis=-2)
    return grad, prec


def _safe_precision(prec: np.ndarray) -> np.ndarray:
    """Replace a non-positive-definite (or non-finite) precision by the prior's, I.

    The log-integrand is the log-likelihood plus the standard-normal log prior, so
    its curvature is at least the prior's wherever the likelihood is log-concave;
    a failed finite difference only moves where the nodes sit, never the value
    the rule converges to.
    """
    dim = prec.shape[-1]
    if dim == 1:
        ok = np.isfinite(prec[..., 0, 0]) & (prec[..., 0, 0] > 1e-8)
    else:
        a, b, c = prec[..., 0, 0], prec[..., 0, 1], prec[..., 1, 1]
        ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & (a > 1e-8) & (a * c - b * b > 1e-12)
    return np.where(ok[..., None, None], prec, np.eye(dim))


def _covariance_cholesky(prec: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor of ``prec^-1`` for ``dim`` 1 or 2, batched."""
    dim = prec.shape[-1]
    if dim == 1:
        return 1.0 / np.sqrt(prec)
    a, b, c = prec[..., 0, 0], prec[..., 0, 1], prec[..., 1, 1]
    det = a * c - b * b
    s11, s12, s22 = c / det, -b / det, a / det
    l11 = np.sqrt(s11)
    l21 = s12 / l11
    l22 = np.sqrt(np.maximum(s22 - l21**2, 1e-300))
    out = np.zeros(prec.shape)
    out[..., 0, 0] = l11
    out[..., 1, 0] = l21
    out[..., 1, 1] = l22
    return out


def log_normal_expectation(loglik, batch_shape: tuple[int, ...], dim: int, n_nodes: int):
    """``log E[exp(loglik(z))]`` for ``z ~ N(0, I_dim)``, by adaptive Gauss-Hermite.

    ``loglik`` maps ``z`` of shape ``(*batch_shape, P, dim)`` to log-likelihoods
    of shape ``(*batch_shape, P)``. The nodes are placed at the mode of
    ``loglik(z) - |z|^2 / 2`` and scaled by its curvature, found by a few Newton
    steps on finite differences; with nodes ``z = c + A x`` the change of
    variables contributes ``log|A| + |x|^2 / 2 - |z|^2 / 2``, the rule
    :mod:`vocab_growth.models.subject_marginal` uses for the engine's own
    marginalised children. Placing the nodes changes only the rule's accuracy,
    never the quantity it estimates.
    """
    stencil = _stencil(dim)

    def log_integrand(z):
        return loglik(z) - 0.5 * np.sum(z * z, axis=-1)

    mode = np.zeros((*batch_shape, dim))
    for _ in range(_NEWTON_STEPS):
        values = log_integrand(mode[..., None, :] + stencil)
        grad, prec = _gradient_precision(values, dim)
        prec = _safe_precision(prec)
        step = np.linalg.solve(prec, grad[..., None])[..., 0]
        step = np.where(np.isfinite(step), step, 0.0)
        norm = np.linalg.norm(step, axis=-1, keepdims=True)
        step = step * np.minimum(1.0, _NEWTON_MAX_STEP / np.maximum(norm, 1e-300))
        # Backtracking: a step is taken only where it raises the log-integrand,
        # halving it otherwise. Where the curvature at the start is not negative
        # -- far from the mode of a skewed Beta-Binomial -- a full Newton step
        # can overshoot and cycle between two points indefinitely.
        current = values[..., 0]
        for _halving in range(_BACKTRACK_HALVINGS):
            candidate = np.clip(mode + step, -_MODE_CLAMP, _MODE_CLAMP)
            better = log_integrand(candidate[..., None, :])[..., 0] >= current
            if better.all():
                break
            step = np.where(better[..., None], step, 0.5 * step)
        else:
            candidate = np.clip(mode + step, -_MODE_CLAMP, _MODE_CLAMP)
            better = log_integrand(candidate[..., None, :])[..., 0] >= current
            step = np.where(better[..., None], step, 0.0)
        mode = np.clip(mode + step, -_MODE_CLAMP, _MODE_CLAMP)
        if np.max(np.abs(step)) < 1e-7:
            break
    _, prec = _gradient_precision(log_integrand(mode[..., None, :] + stencil), dim)
    chol = _covariance_cholesky(_safe_precision(prec))
    x, log_w = _hermite_rule(n_nodes, dim)
    z = mode[..., None, :] + np.einsum("...ij,mj->...mi", chol, x)
    log_det = np.sum(np.log(np.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
    log_terms = (
        log_w + 0.5 * np.sum(x * x, axis=-1) + log_det[..., None] + log_integrand(z)
    )
    return logsumexp(log_terms, axis=-1)


def _flat_draws(posterior, name: str, draws: np.ndarray, **isel) -> np.ndarray:
    """``(n_draws, ...)`` values of ``name`` at the selected flattened draws."""
    da = posterior[name]
    if isel:
        da = da.isel(**isel)
    values = np.asarray(da.values, dtype=float)
    return values.reshape(-1, *values.shape[2:])[draws]


def thinned_draws(n_total: int, max_draws: int) -> np.ndarray:
    """Evenly spaced indices into the flattened (chain, draw) axis."""
    if max_draws <= 0 or max_draws >= n_total:
        return np.arange(n_total)
    return np.unique(np.linspace(0, n_total - 1, max_draws).round().astype(int))


@dataclass(frozen=True)
class TDScoreInputs:
    """Everything the integrated scorer reads, for the scored rows only.

    Rows are ordered so each child's rows are contiguous; ``starts`` indexes the
    first row of each child and ``child_codes`` names them. Draw-indexed arrays
    are ``(S, R)``; ``chol`` maps a standard-normal vector to the integrated
    effects, ``(S, dim, dim)``, and ``chol_u`` does so for comprehension alone in
    a bivariate model.
    """

    engine: str
    rows: np.ndarray
    starts: np.ndarray
    child_codes: np.ndarray
    child_of_row: np.ndarray
    chol: np.ndarray
    arrays: dict
    chol_u: np.ndarray | None = None


def _sex_shift(frame: pd.DataFrame, definition) -> np.ndarray:
    if sex_effect_sigma(definition) is None:
        return np.zeros(len(frame))
    return sex_contrast_codes(frame, allow_unknown=True)


def td_score_inputs(
    frame: pd.DataFrame,
    trace,
    definition,
    holdout_unit: str,
    *,
    max_draws: int = DEFAULT_SCORE_DRAWS,
    check_tolerance: float = 1e-8,
) -> TDScoreInputs:
    """Read the held-out rows' fixed logits, dispersions and effect scales.

    ``frame`` is the fold frame (with ``holdout``). The logit of each held-out
    row is decomposed as the trace computed it -- population curve, study offset,
    sex contrast, child effect -- and the child effect (plus, under ``study``,
    the placeholder study offset) is removed, leaving the part the integration
    adds a fresh draw to. That decomposition is then checked against the trace's
    own per-row probabilities: a graph term the scorer does not know about makes
    the check fail rather than silently changing the density being scored.
    """
    engine = td_engine(definition)
    post = trace.posterior
    holdout = frame["holdout"].to_numpy(dtype=bool)
    subject_codes = frame["subject_code"].to_numpy(dtype=int)
    order = np.lexsort((np.arange(len(frame)), subject_codes))
    rows = order[holdout[order]]
    if rows.size == 0:
        raise ValueError("the fold holds out no rows")
    row_children = subject_codes[rows]
    starts = np.flatnonzero(np.r_[True, row_children[1:] != row_children[:-1]])
    child_codes = row_children[starts]
    child_of_row = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, rows.size]))

    n_total = int(post.sizes["chain"] * post.sizes["draw"])
    draws = thinned_draws(n_total, max_draws)
    study_codes = frame["study_code"].to_numpy(dtype=int)[rows]
    x_sex = _sex_shift(frame, definition)[rows]
    study_unit = holdout_unit == "study"

    def scalar(name):
        return _flat_draws(post, name, draws)

    def at_rows(name):
        return _flat_draws(post, name, draws, obs_id=rows)

    def at_children(name):
        return _flat_draws(post, name, draws, subject_id=row_children)

    def at_studies(name):
        return _flat_draws(post, name, draws, study_id=study_codes)

    if engine == "univariate_re":
        y_col = definition.outcome.value
        f_obs = at_rows("f_obs")
        m = f_obs - at_children("delta_subject")
        if study_unit:
            m = m - at_studies("delta")
        tau_child = scalar("tau_subject")
        variance = tau_child**2 + (scalar("tau") ** 2 if study_unit else 0.0)
        chol = np.sqrt(variance)[:, None, None]
        arrays = {
            "m": m,
            "kappa": at_rows("kappa_obs"),
            "y": frame[y_col].to_numpy(dtype=float)[rows],
            "n": float(definition.n_trials),
        }
        # Nothing to check against beyond `f_obs` itself, which the fixed part
        # was taken from; the study unit's population part is checked below.
        population = m - (
            scalar("beta_sex")[:, None] * x_sex if "beta_sex" in post.data_vars else 0.0
        )
        if not study_unit:
            population = population - at_studies("delta")
        _assert_function_of_age(population, frame["age"].to_numpy()[rows], check_tolerance)
        return TDScoreInputs(
            engine, rows, starts, child_codes, child_of_row, chol, arrays
        )

    # Bivariate: comprehension logit and the production-ratio logit.
    has_sex = "beta_sex_u" in post.data_vars
    shift_u = scalar("beta_sex_u")[:, None] * x_sex if has_sex else 0.0
    shift_q = scalar("beta_sex_q")[:, None] * x_sex if has_sex else 0.0
    pop_u = at_rows("f_u_obs")
    pop_q = at_rows("h_obs")
    delta_u = at_studies("delta_u")
    delta_q = at_studies("delta_q")
    child_u = at_children("delta_subj_u")
    child_q = at_children("delta_subj_q")
    # The decomposition, checked against the trace's own probabilities at
    # these rows and draws.
    for name, total in (
        ("p_u_obs", pop_u + delta_u + shift_u + child_u),
        ("q_obs", pop_q + delta_q + shift_q + child_q),
    ):
        gap = float(np.max(np.abs(expit(total) - at_rows(name))))
        if not gap <= check_tolerance:
            raise RuntimeError(
                f"{definition.model_id}: the scorer's decomposition of {name} is off by "
                f"{gap:.3g} at the held-out rows; the fold graph carries a term the "
                "integrated scorer does not model."
            )
    m_u = pop_u + shift_u + (0.0 if study_unit else delta_u)
    m_q = pop_q + shift_q + (0.0 if study_unit else delta_q)
    _assert_function_of_age(pop_u, frame["age"].to_numpy()[rows], check_tolerance)

    tau_u = scalar("tau_subj_u")
    tau_q = scalar("tau_subj_q")
    rho = (
        np.clip(scalar("rho_uq"), -0.999999, 0.999999)
        if "rho_uq" in post.data_vars
        else np.zeros_like(tau_u)
    )
    s11 = tau_u**2
    s22 = tau_q**2
    s12 = rho * tau_u * tau_q
    if study_unit:
        s11 = s11 + scalar("tau_u") ** 2
        s22 = s22 + scalar("tau_q") ** 2
    l11 = np.sqrt(s11)
    l21 = s12 / l11
    l22 = np.sqrt(np.maximum(s22 - l21**2, 1e-300))
    chol = np.zeros((len(draws), 2, 2))
    chol[:, 0, 0] = l11
    chol[:, 1, 0] = l21
    chol[:, 1, 1] = l22

    n_trials = definition.n_trials
    understood = frame["understood"].to_numpy(dtype=float)
    spoken_observed = np.full(len(frame), -1, dtype=int)
    spoken_trials = np.full(len(frame), n_trials, dtype=int)
    spoken_conditional = np.zeros(len(frame), dtype=bool)
    # The engine's paired-count rule, applied to the whole fold frame without
    # the holdout mask, so a held-out row is classified exactly as the model
    # classifies the same row when it trains on it (the 2026-09-23 correction).
    spec = nested_outcome_spec(
        frame, parent_col="understood", outcome_col="spoken", n_trials=n_trials
    )
    spoken_observed[spec.indices] = spec.observed
    spoken_trials[spec.indices] = spec.trials
    spoken_conditional[spec.indices] = spec.is_conditional
    has_u = np.isfinite(understood[rows])
    has_s = spoken_observed[rows] >= 0
    arrays = {
        "m_u": m_u,
        "m_q": m_q,
        "kappa_u": at_rows("kappa_u_obs"),
        "kappa_s": at_rows("kappa_s_obs"),
        "y_u": np.where(has_u, understood[rows], 0.0),
        "has_u": has_u,
        "y_s": np.where(has_s, spoken_observed[rows], 0).astype(float),
        "s_trials": spoken_trials[rows].astype(float),
        "s_conditional": spoken_conditional[rows],
        "has_s": has_s,
        "n": float(n_trials),
    }
    return TDScoreInputs(
        engine,
        rows,
        starts,
        child_codes,
        child_of_row,
        chol,
        arrays,
        chol_u=l11[:, None, None],
    )


def _assert_function_of_age(population: np.ndarray, ages: np.ndarray, tolerance: float):
    """The population part of a logit must depend on age alone.

    A cheap invariant that catches an unmodelled study- or child-level term left
    in the fixed part: rows of equal age must agree draw by draw.
    """
    population = np.broadcast_to(population, (population.shape[0], ages.size))
    for age in np.unique(ages):
        block = population[:, ages == age]
        spread = float(np.max(block.max(axis=1) - block.min(axis=1)))
        if not spread <= tolerance * max(1.0, float(np.max(np.abs(block)))):
            raise RuntimeError(
                f"the fixed part of the held-out logits varies by {spread:.3g} at age "
                f"{age:g} after removing every term the scorer models; the fold graph "
                "carries a term the integrated scorer does not implement."
            )


def _row_logliks(inputs: TDScoreInputs, effects: np.ndarray, sl: slice, *, understood_only=False):
    """Per-row log-likelihood at the given row effects ``(S, R, P, dim)``."""
    a = inputs.arrays
    n = a["n"]
    if inputs.engine == "univariate_re":
        p = _clip(expit(a["m"][:, sl, None] + effects[..., 0]))
        return betabinomial_logpmf(a["y"][None, sl, None], n, p, a["kappa"][:, sl, None])
    p_u = expit(a["m_u"][:, sl, None] + effects[..., 0])
    ll_u = betabinomial_logpmf(
        a["y_u"][None, sl, None], n, _clip(p_u), a["kappa_u"][:, sl, None]
    )
    ll = np.where(a["has_u"][None, sl, None], ll_u, 0.0)
    if understood_only:
        return ll
    q = expit(a["m_q"][:, sl, None] + effects[..., 1])
    p_s = np.where(a["s_conditional"][None, sl, None], q, p_u * q)
    ll_s = betabinomial_logpmf(
        a["y_s"][None, sl, None],
        a["s_trials"][None, sl, None],
        _clip(p_s),
        a["kappa_s"][:, sl, None],
    )
    return ll + np.where(a["has_s"][None, sl, None], ll_s, 0.0)


def _child_batches(inputs: TDScoreInputs, n_draws: int, n_points: int):
    """Contiguous child ranges whose rows fit the working-array budget."""
    row_starts = np.r_[inputs.starts, inputs.rows.size]
    budget = max(1, _BATCH_ELEMENTS // max(1, n_draws * n_points))
    first = 0
    n_children = inputs.starts.size
    while first < n_children:
        last = first + 1
        while (
            last < n_children
            and row_starts[last + 1] - row_starts[first] <= budget
        ):
            last += 1
        yield first, last
        first = last


def integrated_log_densities(
    inputs: TDScoreInputs, *, n_nodes: int = DEFAULT_QUADRATURE_NODES, understood_only=False
) -> np.ndarray:
    """``(S, n_children)`` log densities of each held-out child's counts per draw.

    Each child's effects are integrated out per posterior draw; the caller then
    averages over draws.
    """
    chol = inputs.chol_u if understood_only else inputs.chol
    dim = chol.shape[-1]
    n_draws = chol.shape[0]
    n_points = max(_stencil(dim).shape[0], n_nodes**dim)
    out = np.empty((n_draws, inputs.starts.size))
    row_starts = np.r_[inputs.starts, inputs.rows.size]
    for first, last in _child_batches(inputs, n_draws, n_points):
        sl = slice(row_starts[first], row_starts[last])
        local_child = inputs.child_of_row[sl] - first
        local_starts = inputs.starts[first:last] - row_starts[first]

        def loglik(z, sl=sl, local_child=local_child, local_starts=local_starts):
            effects = np.einsum("sij,scpj->scpi", chol, z)[:, local_child]
            ll = _row_logliks(inputs, effects, sl, understood_only=understood_only)
            return np.add.reduceat(ll, local_starts, axis=1)

        out[:, first:last] = log_normal_expectation(
            loglik, (n_draws, last - first), dim, n_nodes
        )
    return out


def td_child_elpds(
    inputs: TDScoreInputs, *, n_nodes: int = DEFAULT_QUADRATURE_NODES
) -> pd.DataFrame:
    """Per held-out child: elpd of all their counts, and by outcome where bivariate.

    ``elpd`` is ``log mean_s p(y_child | draw s)`` with the child's effects
    integrated out inside each draw. A bivariate model also gets
    ``elpd_understood`` (comprehension alone, integrated over the comprehension
    effect's marginal) and ``elpd_spoken_given_understood``, their difference,
    which is the speech density conditional on the child's observed
    comprehension: the mixture over draws and effects is reweighted by the
    comprehension counts, not merely given them as a denominator.
    """
    log_s = math.log(inputs.chol.shape[0])
    joint = integrated_log_densities(inputs, n_nodes=n_nodes)
    row_counts = np.diff(np.r_[inputs.starts, inputs.rows.size])
    out = pd.DataFrame(
        {
            "subject_code": inputs.child_codes,
            "n_rows": row_counts,
            "elpd": logsumexp(joint, axis=0) - log_s,
        }
    )
    if inputs.engine == "bivariate_re":
        a = inputs.arrays
        n_u = np.add.reduceat(a["has_u"].astype(int), inputs.starts)
        n_s = np.add.reduceat(a["has_s"].astype(int), inputs.starts)
        marginal_u = integrated_log_densities(inputs, n_nodes=n_nodes, understood_only=True)
        elpd_u = logsumexp(marginal_u, axis=0) - log_s
        # A child with no count of an outcome contributes no score for it, and
        # must not enter the table as 0.0, which would read as a perfect one.
        out["n_understood"] = n_u
        out["n_spoken"] = n_s
        out["elpd_understood"] = np.where(n_u > 0, elpd_u, np.nan)
        out["elpd_spoken_given_understood"] = np.where(
            n_s > 0, out["elpd"].to_numpy() - np.where(n_u > 0, elpd_u, 0.0), np.nan
        )
    return out


#: The quantities each engine reports, headline first.
TD_QUANTITIES = {
    "univariate_re": ("elpd",),
    "bivariate_re": ("elpd", "elpd_understood", "elpd_spoken_given_understood"),
}


def summarise_td(
    child_df: pd.DataFrame, cluster: str, flags: dict[str, bool]
) -> pd.DataFrame:
    """Per model and quantity: total, clustered SE, counts, convergence flag.

    The standard error is computed over cluster totals -- children for the child
    unit, studies for the study unit -- because the scores inside a cluster share
    a child's or a study's effect (the 2026-09-23 correction for paired
    forward scores, applied to the totals too).
    """
    records = []
    for (model, quantity), group in _long_scores(child_df).groupby(["model", "quantity"], sort=False):
        totals = group.groupby(cluster)["value"].sum()
        n_clusters = len(totals)
        records.append(
            {
                "model": model,
                "quantity": quantity,
                "elpd": float(totals.sum()),
                "se": float(np.sqrt(n_clusters) * np.std(totals, ddof=1)) if n_clusters > 1 else float("nan"),
                "cluster": cluster,
                "n_clusters": n_clusters,
                "n_children": int(group["subject_code"].nunique()),
                "n_rows": int(group["n_rows"].sum()),
                "mean_elpd_per_child": float(group["value"].mean()),
                "all_folds_converged": bool(flags.get(model, False)),
            }
        )
    return pd.DataFrame(records)


def pairwise_td(child_df: pd.DataFrame, cluster: str, flags: dict[str, bool]) -> pd.DataFrame:
    """Paired differences between models on the same children, SE over clusters."""
    long = _long_scores(child_df)
    models = list(dict.fromkeys(long["model"]))
    records = []
    for quantity, group in long.groupby("quantity", sort=False):
        index = ["subject_code"] if cluster == "subject_code" else ["subject_code", cluster]
        wide = group.pivot_table(index=index, columns="model", values="value")
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                if a not in wide or b not in wide:
                    continue
                common = wide[[a, b]].dropna()
                diff = (common[b] - common[a]).groupby(level=cluster).sum()
                n = len(diff)
                se = float(np.sqrt(n) * np.std(diff, ddof=1)) if n > 1 else float("nan")
                total = float(diff.sum())
                records.append(
                    {
                        "quantity": quantity,
                        "model_a": a,
                        "model_b": b,
                        "elpd_diff_b_minus_a": total,
                        "se_paired": se,
                        "diff_over_se": total / se if se > 0 else float("nan"),
                        "cluster": cluster,
                        "n_clusters": n,
                        "n_children": len(common),
                        "all_folds_converged": bool(flags.get(a, False) and flags.get(b, False)),
                    }
                )
    return pd.DataFrame(records)


def _long_scores(child_df: pd.DataFrame) -> pd.DataFrame:
    quantities = [q for q in ("elpd", "elpd_understood", "elpd_spoken_given_understood") if q in child_df]
    long = child_df.melt(
        id_vars=[c for c in child_df.columns if c not in quantities],
        value_vars=quantities,
        var_name="quantity",
        value_name="value",
    )
    return long.dropna(subset=["value"])


# ============================================================
# Driver
# ============================================================


def summarise_models(
    elpd_df: pd.DataFrame, models: tuple[str, ...], flags: dict[str, bool]
) -> pd.DataFrame:
    """Per-model elpd totals, with the fold-convergence flag attached."""
    summary_rows = []
    for short in models:
        e = elpd_df[short].dropna().to_numpy()
        n = len(e)
        # SE of total elpd = sqrt(n * var of per-subject elpd).
        se = float(np.sqrt(n) * np.std(e, ddof=1))
        summary_rows.append(
            {
                "model": short,
                "elpd_loso": float(e.sum()),
                "se": se,
                "n_subjects": n,
                "mean_elpd_per_subject": float(e.mean()),
                "all_folds_converged": bool(flags.get(short, False)),
            }
        )
    return pd.DataFrame(summary_rows)


def pairwise_compare(
    elpd_df: pd.DataFrame, models: tuple[str, ...], flags: dict[str, bool]
) -> pd.DataFrame:
    """Pairwise elpd differences with paired SE, flagged for convergence.

    A row involving any model with a non-converged fold carries
    ``all_folds_converged`` False. It is flagged rather than dropped so the row
    stays visible for what it is, but its elpd difference must not be
    interpreted.
    """
    pair_rows = []
    for i, sa in enumerate(models):
        for j, sb in enumerate(models):
            if i >= j:
                continue
            common = elpd_df[[sa, sb]].dropna()
            diff = common[sb] - common[sa]
            elpd_diff = float(diff.sum())
            dse = float(np.sqrt(len(diff)) * np.std(diff, ddof=1))
            pair_rows.append(
                {
                    "model_a": sa,
                    "model_b": sb,
                    "elpd_diff_b_minus_a": elpd_diff,
                    "dse_paired": dse,
                    "diff_over_dse": elpd_diff / dse if dse > 0 else float("nan"),
                    "n_subjects": len(diff),
                    "all_folds_converged": bool(
                        flags.get(sa, False) and flags.get(sb, False)
                    ),
                }
            )
    return pd.DataFrame(pair_rows)


def check_td_run(models: tuple[str, ...], holdout_unit: str) -> None:
    """Refuse, before any fit, a typically developing run the scorer cannot score.

    Every model must pass :func:`td_scoring_refusal`, and all of them must share
    one prepared frame: the folds are built on it and a paired difference is
    only a difference between models when both scored the same children.
    """
    reasons = [
        reason
        for m in models
        if (reason := td_scoring_refusal(AVAILABLE[m], holdout_unit)) is not None
    ]
    if reasons:
        raise SystemExit("Cannot score:\n  " + "\n  ".join(reasons))
    hashes = {m: analysis_frame_hash(load_td_frame(m)) for m in models}
    if len(set(hashes.values())) > 1:
        raise SystemExit(
            "These models fit different prepared frames, so their folds and scores "
            f"cannot be paired: {hashes}. Run them separately, each with its own "
            "--suffix."
        )


def main_td(
    K: int,
    sampling_config_name: str,
    models: tuple[str, ...],
    suffix: str,
    holdout_unit: str,
    *,
    max_folds: int | None = None,
    score_draws: int = DEFAULT_SCORE_DRAWS,
    quadrature_nodes: int = DEFAULT_QUADRATURE_NODES,
) -> None:
    """Grouped K-fold by child, or leave-one-study-out, for the TD reference models."""
    check_td_run(models, holdout_unit)
    specs = [(m, AVAILABLE[m]) for m in models]
    frame = load_td_frame(models[0])
    frame_hash = analysis_frame_hash(frame)
    os.makedirs(OUT_DIR, exist_ok=True)
    written = ComparisonOutputs(OUT_DIR)
    stem = f"kfold_loso_td_{holdout_unit}"
    print(f"models: {', '.join(models)}   unit={holdout_unit}   config={sampling_config_name}")
    print(
        f"  frame {frame_hash[:19]}: {len(frame)} rows / "
        f"{frame['subject_code'].nunique()} children / {frame['study'].nunique()} studies"
    )
    print(
        f"  scoring: up to {score_draws} posterior draws, adaptive Gauss-Hermite with "
        f"{quadrature_nodes} nodes per effect dimension"
    )

    if holdout_unit == "subject":
        folds, _ = stratified_subject_folds(frame, K=K)
        fold_items: list = list(folds)
        labels = [f"fold{k}" for k in range(len(folds))]
    else:
        fold_items = study_folds(frame)
        labels = [f"study-{s}" for s in fold_items]
    if max_folds is not None:
        fold_items = fold_items[:max_folds]
        labels = labels[:max_folds]
        print(f"  --max-folds {max_folds}: running {len(fold_items)} fold(s) only")

    sampling_cfg = sampling.get_sampling_configuration(sampling_config_name)
    scores: list[pd.DataFrame] = []
    fit_records: list[FoldFitRecord] = []
    for k, (held, label) in enumerate(zip(fold_items, labels, strict=True)):
        fold_frame = build_td_fold_frame(frame, holdout_unit, held)
        holdout = fold_frame["holdout"].to_numpy(dtype=bool)
        n_children = int(fold_frame.loc[holdout, "subject_code"].nunique())
        print(f"\n=== {label}: {int(holdout.sum())} held-out rows, {n_children} children ===")
        for short, definition in specs:
            started = time.perf_counter()
            print(f"  fitting {short} …", flush=True)
            trace, _, gate = fit_fold(definition, fold_frame, sampling_cfg, f"{short}_{label}")
            scoring_started = time.perf_counter()
            inputs = td_score_inputs(
                fold_frame, trace, definition, holdout_unit, max_draws=score_draws
            )
            child_df = td_child_elpds(inputs, n_nodes=quadrature_nodes)
            print(
                f"    scored {len(child_df)} children over {inputs.chol.shape[0]} draws "
                f"in {time.perf_counter() - scoring_started:.1f}s"
            )
            child_frame = fold_frame.drop_duplicates("subject_code").set_index("subject_code")
            child_df.insert(0, "model", short)
            child_df.insert(1, "fold", k)
            child_df.insert(2, "study", child_frame.loc[child_df["subject_code"], "study"].to_numpy())
            child_df.insert(3, "subject_key", child_frame.loc[child_df["subject_code"], "subject_key"].to_numpy())
            scores.append(child_df)
            n_u = int(holdout.sum()) if td_engine(definition) == "univariate_re" else int(
                (fold_frame["understood"].notna().to_numpy() & holdout).sum()
            )
            n_s = 0 if td_engine(definition) == "univariate_re" else int(
                (fold_frame["spoken"].notna().to_numpy() & holdout).sum()
            )
            elapsed = time.perf_counter() - started
            record = FoldFitRecord(
                model_short=short,
                fold=k,
                n_holdout_subjects=n_children,
                n_holdout_obs_u=n_u,
                n_holdout_obs_s=n_s,
                wall_seconds=elapsed,
                **fold_gate_fields(gate),
            )
            fit_records.append(record)
            del trace
            print(
                f"    {short} {label} done in {elapsed:.1f}s; {len(child_df)} children "
                f"scored, elpd {child_df['elpd'].sum():.1f}; convergence gate "
                f"{'PASS' if record.passed else 'FAIL'}"
            )

    child_scores = pd.concat(scores, ignore_index=True)
    child_scores.to_csv(os.path.join(OUT_DIR, f"{stem}_child_elpds{suffix}.csv"), index=False)
    quantities = [q for q in TD_QUANTITIES[td_engine(specs[0][1])] if q in child_scores]
    by_study = (
        child_scores.groupby(["model", "study"], sort=False)
        .agg(n_children=("subject_code", "nunique"), n_rows=("n_rows", "sum"),
             **{q: (q, "sum") for q in quantities})
        .reset_index()
    )
    by_study.to_csv(os.path.join(OUT_DIR, f"{stem}_by_study{suffix}.csv"), index=False)

    flags = model_convergence_flags(fit_records)
    unconverged = sorted(m for m, ok in flags.items() if not ok)
    if unconverged:
        print("\n" + "!" * 74)
        print("!!! CONVERGENCE WARNING: at least one fold fit failed the diagnostics gate")
        print(f"!!! for: {', '.join(unconverged)}. Their scores must not be interpreted")
        print("!!! (all_folds_converged=False in the CSVs).")
        print("!" * 74)

    cluster = "subject_code" if holdout_unit == "subject" else "study"
    summary = summarise_td(child_scores, cluster, flags)
    summary.to_csv(os.path.join(OUT_DIR, f"{stem}_summary{suffix}.csv"), index=False)
    print("\n=== Summary (SE over " + ("children" if cluster == "subject_code" else "studies") + ") ===")
    print(summary.to_string(index=False))
    if len(models) > 1:
        pairs = pairwise_td(child_scores, cluster, flags)
        pairs.to_csv(os.path.join(OUT_DIR, f"{stem}_compare{suffix}.csv"), index=False)
        print("\n=== Pairwise (paired over the same children) ===")
        print(pairs.to_string(index=False))

    fit_df = pd.DataFrame([r.__dict__ for r in fit_records])
    fit_df.to_csv(os.path.join(OUT_DIR, f"{stem}_fits{suffix}.csv"), index=False)
    print("\n=== Fit timings and convergence ===")
    print(fit_df.to_string(index=False))

    write_comparison_manifest(
        OUT_DIR,
        script=f"kfold_loso.py (td {holdout_unit}: {'+'.join(models)}{suffix})",
        contributing={},
        outputs=written.written(),
        source_data_hash=source_data_hash(env.DATA_DIR),
        arguments=[
            *models,
            f"holdout-unit={holdout_unit}",
            f"K={K}" if holdout_unit == "subject" else "K=studies",
            f"max-folds={max_folds}",
            f"config={sampling_config_name}",
            f"score-draws={score_draws}",
            f"quadrature-nodes={quadrature_nodes}",
            f"analysis-frame={frame_hash}",
        ],
    )
    total_wall = fit_df["wall_seconds"].sum()
    print(f"\nTotal fit + scoring wall time: {total_wall:.1f}s ({total_wall/60:.1f} min)")


def main(
    K: int = 5,
    sampling_config_name: str = "test",
    models: tuple[str, ...] = DEFAULT_MODELS,
    suffix: str = "",
    holdout_unit: str = "subject",
    visit1_conditioning: str = "all-outcomes",
    *,
    max_folds: int | None = None,
    score_draws: int = DEFAULT_SCORE_DRAWS,
    quadrature_nodes: int = DEFAULT_QUADRATURE_NODES,
) -> None:
    td = [m for m in models if m in TD_AVAILABLE]
    if td:
        if len(td) != len(models):
            raise SystemExit(
                "A run scores one population: the Down syndrome models are scored on "
                "the pooled Down syndrome frame and the typically developing ones on "
                f"their own. Got {', '.join(models)}."
            )
        if visit1_conditioning != "all-outcomes":
            raise SystemExit("--visit1-conditioning applies to the Down syndrome later-waves unit only.")
        main_td(
            K, sampling_config_name, models, suffix, holdout_unit,
            max_folds=max_folds, score_draws=score_draws,
            quadrature_nodes=quadrature_nodes,
        )
        return
    if holdout_unit == "study":
        raise SystemExit(
            "--holdout-unit study is implemented for the typically developing models "
            f"{sorted(TD_AVAILABLE)} only."
        )
    if max_folds is not None:
        raise SystemExit("--max-folds applies to the typically developing path only.")
    if holdout_unit not in HOLDOUT_UNITS:
        raise SystemExit(f"--holdout-unit must be one of {HOLDOUT_UNITS}")
    if visit1_conditioning not in VISIT1_CONDITIONINGS:
        raise SystemExit(f"--visit1-conditioning must be one of {VISIT1_CONDITIONINGS}")
    if holdout_unit == "subject" and visit1_conditioning != "all-outcomes":
        # Under whole-subject holdout there is no retained first visit to
        # condition on, so the flag would silently do nothing.
        raise SystemExit(
            "--visit1-conditioning applies only to --holdout-unit later-waves: "
            "the subject unit holds the first wave out too, so there is no "
            "first-visit outcome left to condition on."
        )
    SPECS = [(m, AVAILABLE[m]) for m in models]
    os.makedirs(OUT_DIR, exist_ok=True)
    written = ComparisonOutputs(OUT_DIR)
    print(f"models: {', '.join(models)}   K={K}   config={sampling_config_name}")
    print(f"holdout unit: {holdout_unit}   visit-1 conditioning: {visit1_conditioning}")
    mismatch = sex_covariate_mismatch(SPECS)
    if mismatch:
        print(f"!!! {mismatch}")

    print("Reloading DS analysis frame …", flush=True)
    analysis_df = load_analysis_frame()
    print(
        f"  {len(analysis_df)} observations / "
        f"{analysis_df['subject_code'].nunique()} subjects"
    )

    waves = wave_index(analysis_df)
    if holdout_unit == "later-waves":
        n_later = int((waves > 0).sum())
        n_repeat_children = int(
            analysis_df.loc[waves > 0, "subject_code"].nunique()
        )
        print(
            f"  {n_later} rows in a later wave, across {n_repeat_children} "
            f"children with more than one visit"
        )
        if n_later == 0:
            raise SystemExit(
                "No row is in a later wave, so the later-waves unit would hold "
                "nothing out and score nothing."
            )

    print(f"\nBuilding {K} stratified folds …", flush=True)
    folds, _subj_table = stratified_subject_folds(analysis_df, K=K)
    for k, fold in enumerate(folds):
        n_obs_in = analysis_df["subject_code"].isin(fold).sum()
        print(f"  fold {k}: {len(fold)} subjects, {n_obs_in} observations")

    sampling_cfg = sampling.get_sampling_configuration(sampling_config_name)

    # Per-model, per-subject elpd accumulator.
    elpd_per_model: dict[str, dict[int, float]] = {
        short: {} for short, _ in SPECS
    }
    # Criterion 4 is stated on comprehension alone, so the later-waves unit
    # accumulates that column beside the two-outcome total.
    elpd_u_per_model: dict[str, dict[int, float]] = {
        short: {} for short, _ in SPECS
    }
    fit_records: list[FoldFitRecord] = []

    for k, fold_subjects in enumerate(folds):
        print(f"\n=== Fold {k}/{K} — {len(fold_subjects)} held-out subjects ===")
        for short, definition in SPECS:
            label = f"{short}_fold{k}"
            print(f"  fitting {short} …", flush=True)
            started = time.perf_counter()

            df_with_holdout = build_fold_frame(
                analysis_df, fold_subjects, waves, holdout_unit, visit1_conditioning
            )
            trace, _, gate = fit_fold(definition, df_with_holdout, sampling_cfg, label)
            holdout_mask = df_with_holdout["holdout"].to_numpy()
            # Score exactly what left the likelihood. Under the subject unit that
            # is every row of the fold's children, which is what the unmasked call
            # summed before; under later-waves the retained first wave is training
            # data and scoring it would be scoring the fit to itself.
            scored_rows = None if holdout_unit == "subject" else holdout_mask
            elpds = holdout_subject_elpds(
                df_with_holdout, trace, fold_subjects, row_mask=scored_rows
            )
            elpd_per_model[short].update(elpds)
            if holdout_unit == "later-waves":
                elpd_u_per_model[short].update(
                    holdout_subject_elpds(
                        df_with_holdout,
                        trace,
                        fold_subjects,
                        row_mask=scored_rows,
                        outcomes=("understood",),
                    )
                )
            n_u_holdout = int(
                ((df_with_holdout["understood"].notna()) & holdout_mask).sum()
            )
            n_s_holdout = int(
                ((df_with_holdout["spoken"].notna()) & holdout_mask).sum()
            )
            elapsed = time.perf_counter() - started
            record = FoldFitRecord(
                model_short=short,
                fold=k,
                n_holdout_subjects=len(fold_subjects),
                n_holdout_obs_u=n_u_holdout,
                n_holdout_obs_s=n_s_holdout,
                wall_seconds=elapsed,
                **fold_gate_fields(gate),
            )
            fit_records.append(record)
            print(
                f"    fold {k} / {short} done in {elapsed:.1f}s — "
                f"{len(elpds)} holdout subjects evaluated; convergence gate "
                f"{'PASS' if record.passed else 'FAIL'}"
            )

    # Build a per-subject elpd table.
    rows = []
    for short in elpd_per_model:
        for s_code, e in elpd_per_model[short].items():
            rows.append(
                {"model": short, "subject_code": s_code, "elpd": e}
            )
    elpd_df = pd.DataFrame(rows).pivot_table(
        index="subject_code", columns="model", values="elpd"
    )
    elpd_df.to_csv(os.path.join(OUT_DIR, f"kfold_loso_subject_elpds{suffix}.csv"))

    # Convergence flags across folds. A model with any failed fold has its rows
    # flagged, never silently dropped: the numbers stay visible for what they
    # are, but they must not be read as a model comparison.
    flags = model_convergence_flags(fit_records)
    unconverged = sorted(m for m, ok in flags.items() if not ok)
    if unconverged:
        print("\n" + "!" * 74)
        print(
            "!!! CONVERGENCE WARNING: at least one fold fit failed the "
            "diagnostics gate"
        )
        print(f"!!! for: {', '.join(unconverged)}.")
        print(
            "!!! Their elpd totals and every pairwise comparison involving them"
        )
        print(
            "!!! must not be interpreted (all_folds_converged=False in the CSVs;"
        )
        print("!!! per-fold detail in kfold_loso_fits.csv).")
        print("!" * 74)

    # Per-model totals + paired SE.
    print("\n=== K-fold LOSO summary ===")
    summary_df = summarise_models(elpd_df, models, flags)
    summary_df.to_csv(
        os.path.join(OUT_DIR, f"kfold_loso_summary{suffix}.csv"), index=False
    )
    print(summary_df.to_string(index=False))

    # Pairwise comparison with paired SE.
    print("\n=== Pairwise comparisons (paired-difference SE) ===")
    pair_df = pairwise_compare(elpd_df, models, flags)
    pair_df.to_csv(
        os.path.join(OUT_DIR, f"kfold_loso_compare{suffix}.csv"), index=False
    )
    print(pair_df.to_string(index=False))

    # Criterion 4's own quantity: held-out later-wave comprehension, scored on
    # its own. Written as a separate set rather than an extra column so the
    # summary and comparison tables keep one meaning per file.
    if holdout_unit == "later-waves":
        u_rows = [
            {"model": short, "subject_code": s_code, "elpd": e}
            for short in elpd_u_per_model
            for s_code, e in elpd_u_per_model[short].items()
        ]
        elpd_u_df = pd.DataFrame(u_rows).pivot_table(
            index="subject_code", columns="model", values="elpd"
        )
        elpd_u_df.to_csv(
            os.path.join(OUT_DIR, f"kfold_loso_understood_elpds{suffix}.csv")
        )
        print("\n=== Held-out later-wave COMPREHENSION only (criterion 4) ===")
        summary_u = summarise_models(elpd_u_df, models, flags)
        summary_u.to_csv(
            os.path.join(OUT_DIR, f"kfold_loso_understood_summary{suffix}.csv"),
            index=False,
        )
        print(summary_u.to_string(index=False))
        pair_u = pairwise_compare(elpd_u_df, models, flags)
        pair_u.to_csv(
            os.path.join(OUT_DIR, f"kfold_loso_understood_compare{suffix}.csv"),
            index=False,
        )
        print(pair_u.to_string(index=False))

    fit_df = pd.DataFrame([r.__dict__ for r in fit_records])
    fit_df.to_csv(os.path.join(OUT_DIR, f"kfold_loso_fits{suffix}.csv"), index=False)
    print("\n=== Fit timings and convergence ===")
    print(fit_df.to_string(index=False))

    # This script fits its own folds rather than reading a model of record, so
    # there is no contributing fit to fingerprint. What its tables can outlive
    # is a data change, and that is what the manifest records for them
    # (issue #266 finding 1).
    write_comparison_manifest(
        OUT_DIR,
        script="kfold_loso.py",
        contributing={},
        outputs=written.written(),
        source_data_hash=source_data_hash(env.DATA_DIR),
        arguments=[
            *models,
            f"K={K}",
            f"config={sampling_config_name}",
            f"holdout-unit={holdout_unit}",
            f"visit1-conditioning={visit1_conditioning}",
        ],
    )

    total_wall = fit_df["wall_seconds"].sum()
    print(f"\nTotal fit wall time: {total_wall:.1f}s ({total_wall/60:.1f} min)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--config", default="test")
    ap.add_argument(
        "--holdout-unit",
        default="subject",
        choices=HOLDOUT_UNITS,
        help=(
            "what a fold removes from the likelihood: every row of its children "
            "(subject, the default), only their rows after the first "
            "administration wave (later-waves, Down syndrome only), or one whole "
            "study per fold (study, typically developing only)"
        ),
    )
    ap.add_argument(
        "--visit1-conditioning",
        default="all-outcomes",
        choices=VISIT1_CONDITIONINGS,
        help=(
            "with --holdout-unit later-waves, whether a fold child's retained "
            "first wave keeps both counts or only its comprehension count "
            "(understood-only is criterion 4's conditioning)"
        ),
    )
    ap.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=f"comma-separated subset of {sorted(AVAILABLE)}",
    )
    ap.add_argument(
        "--suffix",
        default="",
        help="appended to the output filenames so a non-default model set "
        "does not overwrite the VG07/VG08/VG09 results",
    )
    ap.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="typically developing path only: run the first N folds and stop "
        "(a smoke run; the totals then cover only those folds' children)",
    )
    ap.add_argument(
        "--score-draws",
        type=int,
        default=DEFAULT_SCORE_DRAWS,
        help="typically developing path: posterior draws the integrated scorer "
        "averages over, evenly thinned (0 = all)",
    )
    ap.add_argument(
        "--quadrature-nodes",
        type=int,
        default=DEFAULT_QUADRATURE_NODES,
        help="typically developing path: adaptive Gauss-Hermite nodes per effect "
        "dimension",
    )
    a = ap.parse_args()
    chosen = tuple(m.strip().upper() for m in a.models.split(",") if m.strip())
    unknown = [m for m in chosen if m not in AVAILABLE]
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}; available {sorted(AVAILABLE)}")
    main(
        K=a.folds,
        sampling_config_name=a.config,
        models=chosen,
        suffix=a.suffix,
        holdout_unit=a.holdout_unit,
        visit1_conditioning=a.visit1_conditioning,
        max_folds=a.max_folds,
        score_draws=a.score_draws,
        quadrature_nodes=a.quadrature_nodes,
    )
