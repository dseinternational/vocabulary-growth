# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model definitions for the vocabulary growth model family.

Each model is fully determined by its definition: population, outcome(s),
prior parameters, and data configuration. Procedural code (model building,
sampling, plotting, reporting) lives in the six engines — common.py,
common_univariate_re.py, common_bivariate.py, common_bivariate_re.py,
common_trivariate.py and common_joint_modality.py — with sampling,
diagnostics and fit orchestration shared from common.py across all of them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum


class Population(Enum):
    """Study population."""

    DOWN_SYNDROME = "ds"
    TYPICALLY_DEVELOPING = "td"


class Outcome(Enum):
    """Vocabulary outcome variable."""

    SPOKEN = "spoken"
    UNDERSTOOD = "understood"


class ModelType(Enum):
    """Model structure type."""

    UNIVARIATE = "univariate"
    BIVARIATE = "bivariate"
    TRIVARIATE = "trivariate"
    JOINT = "joint"


# ============================================================
# Shared prior defaults (same default kappa shape reused by every model)
# ============================================================


@dataclass
class KappaPriorParams:
    """Parameters for the dispersion (kappa) prior distributions."""

    kappa_min_mu: float = math.log(5.0)
    """LogNormal mu for kappa_min."""
    kappa_min_sigma: float = 0.6
    """LogNormal sigma for kappa_min."""
    a_kappa_mu: float = math.log(8.0)
    """Normal mu for a_kappa."""
    a_kappa_sigma: float = 1.0
    """Normal sigma for a_kappa."""
    b_kappa_mag_sigma: float = 0.3
    """HalfNormal sigma for b_kappa_mag."""


# ============================================================
# Univariate model definition
# ============================================================


@dataclass
class UnivariateModelDefinition:
    """Complete definition for a single-outcome model (VG01-VG04, VG11-VG12)."""

    model_id: str
    """Model identifier, e.g. 'VG01'."""
    config_name: str
    """Configuration name, e.g. 'age-spoken-ds'."""
    banner: str
    """Banner text printed at fit start."""
    population: Population
    outcome: Outcome
    n_trials: int
    """Number of words on the vocabulary checklist."""
    slope_anchors: tuple[float, float]
    """Reference ages (months) for the slope parameterisation."""
    ages_query: list[int]
    """Ages (months) at which to query the posterior."""

    # -- Slope priors (the values that vary across models) --
    p_slope_low_alpha: float
    p_slope_low_beta: float
    p_slope_hi_alpha: float
    p_slope_hi_beta: float

    gp_domain_months: tuple[float, float] | None = None
    """Fixed HSGP age domain. ``None`` uses the observed age range; reporting
    query ages never determine the approximation domain."""

    # -- TD-specific data parameters --
    sample_fraction: float = 1.0
    """Fraction of TD data to subsample (1.0 = no subsampling)."""
    random_seed: int = 47
    """Random seed for TD subsampling."""

    # -- Shared priors (same across all univariate models) --
    ell_unit_alpha: float = 3.0
    ell_unit_beta: float = 3.0
    eta_sigma: float = 0.4
    ell_months_range: tuple[int, int] = (6, 18)
    n_plot: int = 500
    kappa: KappaPriorParams = field(default_factory=KappaPriorParams)

    # -- Study-level random intercepts --
    tau_study_sigma: float = 0.5
    """HalfNormal scale for study intercept SD (logit scale)."""
    min_study_observations: int | None = None
    """Drop studies with fewer than this many observations before fitting study
    intercepts (None = keep all). Trims tiny, near-unidentified study intercepts
    that add parameters without informing the estimates."""

    # -- Subject-level clustering --
    use_subject_re: bool = False
    """If True, add a subject-level random intercept to account for repeated
    assessments of the same child."""
    tau_subject_sigma: float = 0.5
    """HalfNormal scale for the subject intercept SD (logit scale)."""
    one_observation_per_subject: bool = False
    """If True, retain one reproducibly sampled administration per subject. This
    is a clustering sensitivity analysis, not the default estimand."""

    # -- GP anchor constraint (per-draw zero at reference age) --
    anchor_g_at_ref: bool = False
    """If True, constrain the GP to equal zero at the reference age for every draw."""
    gp_anchor_age_months: float | None = None
    """Reference age (months) for the GP anchor. If None, defaults to the midpoint of
    slope_anchors."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.UNIVARIATE

    @property
    def outcome_label(self) -> str:
        return f"Words {self.outcome.value}"


# ============================================================
# Bivariate model definition
# ============================================================


@dataclass
class BivariateModelDefinition:
    """Complete definition for a joint understood+spoken model (e.g. VG05, VG07-VG10, VG13)."""

    model_id: str
    """Model identifier, e.g. 'VG05'."""
    config_name: str
    """Configuration name, e.g. 'age-understood-spoken-ds'."""
    banner: str
    """Banner text printed at fit start."""
    population: Population
    n_trials: int
    """Number of words on the vocabulary checklist."""
    slope_anchors: tuple[float, float]
    """Reference ages (months) for the slope parameterisation."""
    ages_query: list[int]
    """Ages (months) at which to query the posterior."""

    # -- Understood (U) trajectory slope priors --
    p_slope_low_u_alpha: float
    p_slope_low_u_beta: float
    p_slope_hi_u_alpha: float
    p_slope_hi_u_beta: float

    gp_domain_months: tuple[float, float] | None = None
    """Fixed HSGP age domain. ``None`` uses the observed age range; reporting
    query ages never determine the approximation domain."""

    # -- Production ratio (q) slope priors --
    p_slope_low_q_alpha: float = 1.0
    p_slope_low_q_beta: float = 1.5
    p_slope_hi_q_alpha: float = 2.0
    p_slope_hi_q_beta: float = 1.2

    # -- TD-specific data parameters --
    sample_fraction: float = 1.0
    """Fraction of TD data to subsample (1.0 = no subsampling)."""
    random_seed: int = 47
    """Random seed for TD subsampling."""

    # -- Shared priors (same across all bivariate models) --
    ell_unit_u_alpha: float = 3.0
    ell_unit_u_beta: float = 3.0
    eta_u_sigma: float = 0.4
    ell_unit_q_alpha: float = 3.0
    ell_unit_q_beta: float = 3.0
    eta_q_sigma: float = 0.20  # tightened from 0.4 to curb the q-GP<->slope_q/intercept_q competition that the broadened q anchors surface (smoothness prior, not double-dipping; VG09-note Option B)
    ell_months_range: tuple[int, int] = (6, 18)
    n_plot: int = 500
    kappa_u: KappaPriorParams = field(default_factory=KappaPriorParams)
    kappa_s: KappaPriorParams = field(default_factory=KappaPriorParams)

    # -- Study-level random intercepts --
    tau_u_sigma: float = 0.5
    """HalfNormal scale for study intercept SD on understood (logit scale)."""
    tau_q_sigma: float = 0.5
    """HalfNormal scale for study intercept SD on production ratio (logit scale)."""
    min_study_observations: int | None = None
    """Drop studies with fewer than this many observations before fitting study
    intercepts (None = keep all). Trims tiny, near-unidentified study intercepts
    that add parameters without informing the estimates."""

    # -- Subject-level random intercepts --
    use_subject_re_u: bool = False
    """If True, add subject-level random intercepts on the understood trajectory."""
    tau_subj_u_sigma: float = 0.5
    """HalfNormal scale for subject intercept SD on understood (logit scale)."""
    use_subject_re_q: bool = False
    """If True, add subject-level random intercepts on the production ratio q."""
    tau_subj_q_sigma: float = 0.5
    """HalfNormal scale for subject intercept SD on q (logit scale)."""
    one_observation_per_subject: bool = False
    """If True, retain one reproducibly sampled administration per subject. This
    provides a cheap sensitivity analysis for repeated-measures dependence."""

    # -- Within-child cross-lag (VG16, issue #113) --
    use_cross_lag: bool = False
    """If True, add a within-child cross-lag: the child's prior-wave understood
    residual predicts their current production ratio q (earlier receptive ->
    later expressive). Uses the subject understood intercept, so requires
    use_subject_re_u=True for the 'within' baseline."""
    lag_baseline: str = "within"
    """Baseline for the lag residual. 'within' subtracts the child's own understood
    intercept (RI-CLPM within-child effect); 'population' subtracts only the
    population+study level (robustness companion; blends within/between)."""
    beta_lag_mu: float = 0.0
    """Normal mean for the cross-lag coefficient beta_lag (0 = no direction imposed)."""
    beta_lag_sigma: float = 0.5
    """Normal SD for beta_lag (logit scale, weakly-informative)."""

    # -- GP anchor constraint (per-draw zero at reference age) --
    anchor_g_u_at_ref: bool = False
    """If True, constrain g_u to equal zero at the reference age for every draw."""
    anchor_g_q_at_ref: bool = False
    """If True, constrain g_q to equal zero at the reference age for every draw."""
    gp_anchor_age_months: float | None = None
    """Reference age (months) for the GP anchor constraint. If None, defaults to the
    midpoint of slope_anchors."""

    # -- Data age filtering --
    max_age_months: int | None = None
    """Upper bound on age (inclusive, months) for data loading. None = no limit."""
    exclude_us01_spoken_ceiling: bool = False
    """Exclude us_01 WS spoken counts at the 680-word ceiling. This is a
    source-censoring sensitivity only; primary models retain these valid rows."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.BIVARIATE


# ============================================================
# Trivariate model definition
# ============================================================


@dataclass
class TrivariateModelDefinition:
    """Complete definition for a joint understood + spoken + signed model (VG14).

    Extends the bivariate (understood + spoken) structure with a third
    production-ratio curve for signing:

        p_U(a)    = sigmoid(f_U(a))
        q(a)      = sigmoid(h(a))        # fraction of understood words spoken
        r(a)      = sigmoid(g_sign(a))   # fraction of understood words signed
        p_S(a)    = p_U(a) * q(a)
        p_Sign(a) = p_U(a) * r(a)

    Signing is only present in the Down syndrome datasets, so this model is
    DS-only and carries no typically-developing data parameters. It is a
    self-contained copy-and-extend of ``BivariateModelDefinition`` (kept
    isolated; the random-intercept / GP-anchor options are intentionally
    omitted, mirroring the plain VG05 specification).
    """

    model_id: str
    """Model identifier, e.g. 'VG14'."""
    config_name: str
    """Configuration name, e.g. 'age-understood-spoken-signed-ds'."""
    banner: str
    """Banner text printed at fit start."""
    population: Population
    n_trials: int
    """Number of words on the vocabulary checklist."""
    slope_anchors: tuple[float, float]
    """Reference ages (months) for the slope parameterisation."""
    ages_query: list[int]
    """Ages (months) at which to query the posterior."""

    # -- Understood (U) trajectory slope priors --
    p_slope_low_u_alpha: float
    p_slope_low_u_beta: float
    p_slope_hi_u_alpha: float
    p_slope_hi_u_beta: float

    gp_domain_months: tuple[float, float] | None = None
    """Fixed HSGP age domain. ``None`` uses the observed age range; reporting
    query ages never determine the approximation domain."""

    # -- Production ratio (q) slope priors --
    p_slope_low_q_alpha: float = 1.0
    p_slope_low_q_beta: float = 1.5
    p_slope_hi_q_alpha: float = 2.0
    p_slope_hi_q_beta: float = 1.2

    # -- Signed ratio (r) mean prior: THREE-ANCHOR HUMP --
    # r(a) = P(sign | understood) is a developmental HUMP: near zero at young ages
    # (signing just emerging), peaking in the preschool years, then receding as words
    # move into speech. It is anchored at THREE reference ages (sign_anchor_ages) —
    # young / peak / old — with the mean built as two logit-linear segments meeting
    # at the peak anchor, clamped flat outside (see gp_utils.tent_and_gp). This makes
    # the prior MEDIAN a hill — unlike the intercept-only mean (flat median, so words
    # signed = understood x r rose monotonically) and unlike a free monotone slope
    # (which extrapolated to a spurious ~58% signed at 12 mo). The GP now only models
    # smooth departures, so eta_sign reverts toward standard (below).
    #
    # Anchor ages/levels come from the INDEPENDENT DS sign literature, not the fitted
    # data: signing peaks ~mental age 17 mo (Miller 1992 via Clibbens: signed = 2x
    # spoken there, declining by MA ~26 mo) which at a DS DQ ~0.5 is chronological
    # ~34 mo; the inverted-U shape is confirmed by Zampini (parabolic gesture
    # trajectory); DS retain signs longer than TD (Te Kaat-van den Os review), so the
    # old anchor stays modest (not near-zero) and uk_06 has real 60-115 mo signers.
    # The peak LEVEL is kept broad because the peak AGE is only weakly identifiable.
    sign_anchor_ages: tuple[float, float, float] = (15.0, 36.0, 96.0)
    """Young / peak / old reference ages (months) for the signed-ratio hump."""
    p_slope_low_sign_alpha: float = 2.0
    p_slope_low_sign_beta: float = 20.0
    """Young anchor r(~15 mo): Beta(2, 20), median ~0.08 (signing just emerging)."""
    p_slope_mid_sign_alpha: float = 3.0
    p_slope_mid_sign_beta: float = 4.0
    """Peak anchor r(~36 mo): Beta(3, 4), median ~0.42, broad 5-95% ~[0.15, 0.72]."""
    p_slope_hi_sign_alpha: float = 2.0
    p_slope_hi_sign_beta: float = 16.0
    """Old anchor r(~96 mo): Beta(2, 16), median ~0.11 (declined, but not to zero)."""

    # -- Shared GP / amplitude priors --
    ell_unit_u_alpha: float = 3.0
    ell_unit_u_beta: float = 3.0
    eta_u_sigma: float = 0.4
    ell_unit_q_alpha: float = 3.0
    ell_unit_q_beta: float = 3.0
    eta_q_sigma: float = 0.20  # tightened from 0.4 to curb the q-GP<->slope_q/intercept_q competition that the broadened q anchors surface (smoothness prior, not double-dipping; VG09-note Option B)
    # Signed GP favours a shorter lengthscale (~9 mo) than U/q so the signing
    # peak can stand apart from the post-60 mo collapse to near-zero, rather than
    # being smoothed into a monotone decline. (Shorter still only adds wiggle
    # without moving the population peak past ~30 mo: the late-preschool data
    # spike is too sparse/overdispersed to pull the population ratio there.)
    ell_unit_sign_alpha: float = 2.0
    ell_unit_sign_beta: float = 5.0
    eta_sign_sigma: float = 0.4
    """HalfNormal scale for the signed-ratio GP amplitude. Reverted to the standard
    ~0.4 now that the three-anchor mean carries the rise-then-fall hump: the GP only
    needs to model smooth departures. (It was inflated to ~1.0 only to force a hump
    out of a flat intercept-only mean; that hack is no longer needed.)"""
    ell_months_range: tuple[int, int] = (6, 18)
    n_plot: int = 500
    kappa_u: KappaPriorParams = field(default_factory=KappaPriorParams)
    kappa_s: KappaPriorParams = field(default_factory=KappaPriorParams)
    kappa_sign: KappaPriorParams = field(default_factory=KappaPriorParams)

    # -- Signed data inclusion --
    include_uk01_signed: bool = False
    """Re-include uk_01's signed-only count as if it were total sign use.

    False by default because uk_01 excludes words that are also spoken, whereas
    the model estimand and the other sources use total signed vocabulary.  This
    switch exists only for a source-sensitivity comparison.
    """
    include_uk06: bool = False
    """Re-include uk_06's unverified signing field.

    False by default until its field dictionary confirms comparability with the
    total-sign construct.  Understood and spoken uk_06 observations remain in the
    fit regardless of this flag.
    """

    # -- Data age filtering --
    max_age_months: int | None = None
    """Upper bound on age (inclusive, months) for data loading. None = no limit."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.TRIVARIATE


# ============================================================
# Joint sign/speech model definition (VG15, issue #49 Option 3)
# ============================================================


@dataclass
class JointModelDefinition:
    """Complete definition for the joint sign/speech model (VG15).

    Extends the trivariate structure (understood + within-understood sign/speak
    ratios r, q) with a scalar Plackett association `psi` (identified from the
    uk_02 four-cell cross-tab) and study random intercepts on each latent
    trajectory. The r/q/p_U prior specs are seeded from the (uk_06-included)
    VG14 fit (same hump-capable signed-ratio spec).

    Optionally (flag-gated, defaults off) also carries subject-level random
    intercepts on each trajectory (`use_subject_re_u/q/sign`) and VG10's
    per-draw GP anchor at a reference age (`anchor_g_u/q/sign_at_ref` +
    `gp_anchor_age_months`), which together remove the GP<->intercept
    redundancy once subject REs add another level-carrying term.
    """

    model_id: str
    config_name: str
    banner: str
    population: Population
    n_trials: int
    slope_anchors: tuple[float, float]
    ages_query: list[int]
    gp_domain_months: tuple[float, float] | None = None
    """Fixed HSGP age domain. ``None`` uses the observed age range; reporting
    query ages never determine the approximation domain."""

    # -- Understood (U) slope priors (aligned with the recalibrated VG02 / VG05) --
    p_slope_low_u_alpha: float = 1.0
    p_slope_low_u_beta: float = 7.0
    p_slope_hi_u_alpha: float = 2.0
    p_slope_hi_u_beta: float = 1.5

    # -- Speak-given-understood (q) slope priors (bivariate defaults) --
    p_slope_low_q_alpha: float = 1.0
    p_slope_low_q_beta: float = 1.5
    p_slope_hi_q_alpha: float = 2.0
    p_slope_hi_q_beta: float = 1.2

    # -- Sign-given-understood (r) mean prior: THREE-ANCHOR HUMP (matching VG14) --
    # r(a) = P(sign | understood) is a developmental hump (near zero young, peaking
    # in the preschool years, receding as words move into speech), anchored at three
    # reference ages (sign_anchor_ages) and built as a tent meeting at the peak
    # anchor (gp_utils.tent_and_gp) so the prior median is a hill. Anchor ages/levels
    # come from the independent DS sign literature (peak ~MA 17 mo ~= chronological
    # ~34 mo, Miller/Clibbens; inverted-U shape, Zampini; DS retain signs longer,
    # Te Kaat) — see the VG14 (TrivariateModelDefinition) comment for the full
    # rationale. Study REs carry between-study level; the GP (anchored at 54 mo,
    # below) carries smooth departures.
    sign_anchor_ages: tuple[float, float, float] = (15.0, 36.0, 96.0)
    """Young / peak / old reference ages (months) for the signed-ratio hump."""
    p_slope_low_sign_alpha: float = 2.0
    p_slope_low_sign_beta: float = 20.0
    """Young anchor r(~15 mo): Beta(2, 20), median ~0.08 (signing just emerging)."""
    p_slope_mid_sign_alpha: float = 3.0
    p_slope_mid_sign_beta: float = 4.0
    """Peak anchor r(~36 mo): Beta(3, 4), median ~0.42, broad 5-95% ~[0.15, 0.72]."""
    p_slope_hi_sign_alpha: float = 2.0
    p_slope_hi_sign_beta: float = 16.0
    """Old anchor r(~96 mo): Beta(2, 16), median ~0.11 (declined, but not to zero)."""

    # -- Shared GP / amplitude priors (sign GP looser + shorter, per VG14) --
    ell_unit_u_alpha: float = 3.0
    ell_unit_u_beta: float = 3.0
    eta_u_sigma: float = 0.6  # aligned with the recalibrated VG02 understood trajectory
    ell_unit_q_alpha: float = 3.0
    ell_unit_q_beta: float = 3.0
    eta_q_sigma: float = 0.20  # tightened from 0.4 to curb the q-GP<->slope_q/intercept_q competition that the broadened q anchors surface (smoothness prior, not double-dipping; VG09-note Option B)
    ell_unit_sign_alpha: float = 2.0
    ell_unit_sign_beta: float = 5.0
    eta_sign_sigma: float = 0.4  # reverted to standard (matches VG14): the three-anchor mean now carries the hump, so the GP only models smooth departures
    ell_months_range: tuple[int, int] = (6, 18)
    n_plot: int = 500
    kappa_u: KappaPriorParams = field(default_factory=KappaPriorParams)
    kappa_s: KappaPriorParams = field(default_factory=KappaPriorParams)
    kappa_sign: KappaPriorParams = field(default_factory=KappaPriorParams)

    # -- Association (Plackett log odds-ratio) --
    log_psi_mu: float = 0.3
    """Normal mu for log psi. Weakly positive (uk_02 shows both > r·q) but spans
    independence (psi = 1)."""
    log_psi_sigma: float = 0.5

    # -- Dirichlet-Multinomial concentration (log scale) --
    log_conc_mu: float = 3.0
    log_conc_sigma: float = 1.0

    # -- Study random-intercept scales (VG07-VG10 pattern) --
    tau_u_sigma: float = 0.5
    tau_q_sigma: float = 0.5
    tau_sign_sigma: float = 0.5

    # -- Subject-level random intercepts (VG08-VG10 pattern, issue #59) --
    use_subject_re_u: bool = False
    """If True, add subject-level random intercepts on the understood trajectory."""
    tau_subj_u_sigma: float = 0.5
    """HalfNormal scale for the subject intercept SD on understood (logit scale)."""
    use_subject_re_q: bool = False
    """If True, add subject-level random intercepts on the speak ratio q."""
    tau_subj_q_sigma: float = 0.5
    """HalfNormal scale for the subject intercept SD on q (logit scale)."""
    use_subject_re_sign: bool = False
    """If True, add subject-level random intercepts on the sign ratio r. Signing is
    the sparsest modality, so this is gated: inspect tau_subj_sign and fall back to
    study-RE-only on r (set False) if it pins near its prior with poor diagnostics."""
    tau_subj_sign_sigma: float = 0.5
    """HalfNormal scale for the subject intercept SD on r (logit scale)."""

    # -- GP anchor constraint (Option D: per-draw zero at reference age) --
    anchor_g_u_at_ref: bool = False
    """If True, constrain g_u to equal zero at the reference age for every draw."""
    anchor_g_q_at_ref: bool = False
    """If True, constrain g_q to equal zero at the reference age for every draw."""
    anchor_g_sign_at_ref: bool = False
    """If True, constrain g_sign to equal zero at the reference age for every draw."""
    gp_anchor_age_months: float | None = None
    """Reference age (months) for the GP anchor. If None, defaults to the midpoint
    of slope_anchors."""

    # -- Signed data inclusion (inherits VG14's decision) --
    include_uk01_signed: bool = False
    """Re-include uk_01's signed-only field for a source-sensitivity fit."""
    include_uk06: bool = False
    """Re-include uk_06's unverified signing field for a source-sensitivity fit."""
    exclude_us01_spoken_ceiling: bool = False
    """Exclude us_01 WS spoken counts at the 680-word ceiling. This is a
    source-censoring sensitivity only; primary models retain these valid rows."""

    # -- nz_01 (Foster-Cohen) produced cross-tab inclusion --
    include_nz01_cells: bool = True
    """If True (default), nz_01's produced modality cross-tab (word-only / sign-only
    / both) enters via a within-produced Dirichlet-Multinomial that informs psi/q/r
    (see common_joint_modality). If False, nz_01 is excluded from VG15 entirely
    (its production-only, 675-item marginals are not comparable to the 810-item
    marginal likelihoods); the flag is kept for reversibility and for isolating
    nz_01's pull on psi."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.JOINT


# ============================================================
# Model instances
# ============================================================

_DS_GP_DOMAIN_MONTHS = (8, 115)
_TD_GP_DOMAIN_MONTHS = (8, 30)
_YOUNG_TD_GP_DOMAIN_MONTHS = (8, 18)

VG01 = UnivariateModelDefinition(
    model_id="VG01",
    config_name="age-spoken-ds",
    banner="Fitting Model VG01: Influence of age on words spoken (A -> S)",
    population=Population.DOWN_SYNDROME,
    outcome=Outcome.SPOKEN,
    n_trials=810,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    gp_domain_months=_DS_GP_DOMAIN_MONTHS,
    p_slope_low_alpha=1.0,
    # Independent anchor — Berglund et al. (2001, Table 3; 330 DS children,
    # confirmed non-overlapping with the training data): DS spoken vocabulary is
    # ~10 median words at 24 mo. Beta(1, 25) places the 24 mo centre at ~22 words
    # (median), deliberately a little above the cohort median so the near-zero
    # early-speech floor is respected without excluding early talkers; the
    # in-sample mean (~15) corroborates. See docs/models/PRIORS.md, "DS anchor
    # priors vs independent cohorts".
    p_slope_low_beta=25.0,
    # 84 mo high anchor: beyond the range of every independent DS CDI cohort
    # (Berglund tops out at 60 mo), so this is deliberately broad regularisation,
    # NOT an externally anchored value. Nudged off the near-uniform Beta(1.1,1.1)
    # only on plausibility grounds — to rule out a priori implausible
    # flat-near-zero spoken curves at age 7. Beta(2, 1.5) lifts the 7-year level
    # and curbs both tails while staying broad.
    p_slope_hi_alpha=2.0,
    p_slope_hi_beta=1.5,
    # Raised from 0.4 to offset the p_slope_low pull-down: lets the HSGP add
    # mid-range curvature so the steep 36-60 mo rise stays covered.
    eta_sigma=0.5,
)

VG02 = UnivariateModelDefinition(
    model_id="VG02",
    config_name="age-understood-ds",
    banner="Fitting Model VG02: Influence of age on words understood (A -> U)",
    population=Population.DOWN_SYNDROME,
    outcome=Outcome.UNDERSTOOD,
    n_trials=810,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    gp_domain_months=_DS_GP_DOMAIN_MONTHS,
    p_slope_low_alpha=1.0,
    # Data-informed regularisation — NOT independently anchored. DS comprehension
    # at chronological age has no independent source in the current library
    # (Berglund is production-only), so this young-age understood anchor rests on
    # the project's own DS data: the previous Beta(1,10) band under-covered the
    # 30-48 mo centre, and Beta(1,7) widens the upper tail toward it. Flagged as a
    # sensitivity target in docs/models/PRIORS.md (no independent comprehension
    # norm). Paired with eta_sigma=0.6 below, which widens the range of early
    # growth rates the HSGP can express.
    p_slope_low_beta=7.0,
    # 84 mo understood high anchor: likewise no independent DS comprehension norm.
    # Nudged off the near-uniform Beta(1.1, 1.1) only on plausibility grounds —
    # the old anchor placed ~10% of prior mass below 80 words at age 7 and ~10%
    # above 720. Beta(2, 1.5) is mildly informative (mean ~0.57, spanning
    # ~0.1-0.95), curbing the flat-near-zero and rocket-to-810 tails without
    # over-committing the level.
    p_slope_hi_alpha=2.0,
    p_slope_hi_beta=1.5,
    eta_sigma=0.6,
)

VG03 = UnivariateModelDefinition(
    model_id="VG03",
    config_name="age-spoken-td",
    banner="Fitting Model VG03: Influence of age on words spoken (A -> S)",
    population=Population.TYPICALLY_DEVELOPING,
    outcome=Outcome.SPOKEN,
    # Common 810-item reference inventory for TD/DS comparisons. The TD
    # loader uses WG and Oxford CDI production plus WS production-only rows.
    n_trials=810,
    slope_anchors=(12, 26),
    ages_query=[9, 12, 15, 18, 21, 24, 27, 30],
    # Pin the established 8-30 month TD reporting domain explicitly so
    # query-grid edits cannot resize the approximation.
    gp_domain_months=_TD_GP_DOMAIN_MONTHS,
    # Independent anchor — Wordbank US-English TD normative deciles (published
    # percentiles, not the training rows): spoken median ~11 words/810 at 12 mo,
    # ~349 at 26 mo (docs/models/PRIORS.md, "TD anchor priors vs Wordbank norms").
    # Lower the 12 mo anchor toward the near-zero norm floor (Beta(1,15) ->
    # Beta(1,30), median ~18 words), soften the near-uniform 26 mo anchor
    # (Beta(1.5,1.1) -> Beta(1.3,1.3), median ~400, broad enough to cover the ~349
    # norm), and widen eta (0.4 -> 0.5). The in-sample mean (~10 words at 12 mo)
    # corroborates.
    p_slope_low_alpha=1.0,
    p_slope_low_beta=30.0,
    p_slope_hi_alpha=1.3,
    p_slope_hi_beta=1.3,
    eta_sigma=0.5,
    # Bumped from 0.1: total TD pool shrank from 16,552 to 6,134
    # after the WS exclusion; this keeps the effective training set
    # (~1,500 rows) close to the previous VG03 fit.
    sample_fraction=0.25,
)

VG04 = UnivariateModelDefinition(
    model_id="VG04",
    config_name="age-understood-td",
    banner="Fitting Model VG04: Influence of age on words understood (A -> U)",
    population=Population.TYPICALLY_DEVELOPING,
    outcome=Outcome.UNDERSTOOD,
    # Common 810-item reference inventory for TD/DS comparisons. The TD
    # loader excludes WS comprehension because it is a production proxy.
    n_trials=810,
    slope_anchors=(12, 26),
    ages_query=[9, 12, 15, 18, 21, 24, 27, 30],
    # Comprehension observations end at 25 months, but the declared reporting
    # range reaches 30. This preserves the existing 8-30 month HSGP domain.
    gp_domain_months=_TD_GP_DOMAIN_MONTHS,
    # 12 mo understood low anchor — independent Wordbank TD norm: comprehension
    # median ~84 words/810 at 12 mo. Beta(1.2, 8) matches at median ~84 (the
    # in-sample mean ~82 corroborates); the old Beta(1,20) centred it at ~28, well
    # below the norm. See docs/models/PRIORS.md, "TD anchor priors vs Wordbank
    # norms".
    p_slope_low_alpha=1.2,
    p_slope_low_beta=8.0,
    # 26 mo understood high anchor — NO independent CDI comprehension norm (WS is
    # production-only), so Beta(1.3, 1.3) is broad regularisation and a named
    # sensitivity target in PRIORS.md, not an externally anchored value.
    p_slope_hi_alpha=1.3,
    p_slope_hi_beta=1.3,
    eta_sigma=0.5,
    # Bumped from 0.1: total comprehension pool shrank from 16,552 to 6,134
    # after the WS exclusion; this keeps the effective training set
    # (~1,500 rows) close to the previous VG04 fit.
    sample_fraction=0.25,
)

VG05 = BivariateModelDefinition(
    model_id="VG05",
    config_name="age-understood-spoken-ds",
    banner=(
        "Fitting Model VG05: Joint model of words understood and spoken"
        " (A -> U, A -> S, U -> S)"
    ),
    population=Population.DOWN_SYNDROME,
    n_trials=810,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    gp_domain_months=_DS_GP_DOMAIN_MONTHS,
    # Understood anchors aligned with the recalibrated VG02 (#135): raise the
    # 24 mo anchor (Beta(1,10) -> Beta(1,7)) and soften the near-uniform 84 mo
    # anchor (Beta(1.1,1.1) -> Beta(2,1.5)); widen eta_u (0.4 -> 0.6). The old
    # joint U band sat ~2.5-3x below the empirical mean at 24-48 mo.
    p_slope_low_u_alpha=1.0,
    p_slope_low_u_beta=7.0,
    p_slope_hi_u_alpha=2.0,
    p_slope_hi_u_beta=1.5,
    eta_u_sigma=0.6,
    # q anchors (weakly-informative, non-double-dipping): broadened from the
    # VG07-posterior-derived Beta(3,22)/Beta(20,4). q_low ~ Beta(2,12) keeps the
    # independent TD q(~12mo) ~= 0.12 centre; q_high ~ Beta(3,2) is broad (5-95%
    # ~0.25-0.90, no independent DS source), so the data set the 84 mo level.
    p_slope_low_q_alpha=2.0,
    p_slope_low_q_beta=12.0,
    p_slope_hi_q_alpha=3.0,
    p_slope_hi_q_beta=2.0,
)

VG07 = BivariateModelDefinition(
    model_id="VG07",
    config_name="age-understood-spoken-ds-re",
    banner=(
        "Fitting Model VG07: Joint model with study random intercepts"
        " (A -> U, A -> S, U -> S) - Down syndrome"
    ),
    population=Population.DOWN_SYNDROME,
    n_trials=810,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    gp_domain_months=_DS_GP_DOMAIN_MONTHS,
    # Understood anchors aligned with the recalibrated VG02 (#135): raise the
    # 24 mo anchor (Beta(1,10) -> Beta(1,7)) and soften the near-uniform 84 mo
    # anchor (Beta(1.1,1.1) -> Beta(2,1.5)); widen eta_u (0.4 -> 0.6). The old
    # joint U band sat ~2.5-3x below the empirical mean at 24-48 mo.
    p_slope_low_u_alpha=1.0,
    p_slope_low_u_beta=7.0,
    p_slope_hi_u_alpha=2.0,
    p_slope_hi_u_beta=1.5,
    eta_u_sigma=0.6,
    # q anchors (weakly-informative, non-double-dipping): broadened from the
    # VG07-posterior-derived Beta(3,22)/Beta(20,4). q_low ~ Beta(2,12) keeps the
    # independent TD q(~12mo) ~= 0.12 centre; q_high ~ Beta(3,2) is broad (5-95%
    # ~0.25-0.90, no independent DS source), so the data set the 84 mo level.
    p_slope_low_q_alpha=2.0,
    p_slope_low_q_beta=12.0,
    p_slope_hi_q_alpha=3.0,
    p_slope_hi_q_beta=2.0,
    tau_u_sigma=0.5,
    tau_q_sigma=0.5,
)

VG08 = BivariateModelDefinition(
    model_id="VG08",
    config_name="age-understood-spoken-ds-re-subj",
    banner=(
        "Fitting Model VG08: Joint model with study + subject random intercepts on U"
        " (A -> U, A -> S, U -> S) - Down syndrome"
    ),
    population=Population.DOWN_SYNDROME,
    n_trials=810,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    gp_domain_months=_DS_GP_DOMAIN_MONTHS,
    # Understood anchors aligned with the recalibrated VG02 (#135): raise the
    # 24 mo anchor (Beta(1,10) -> Beta(1,7)) and soften the near-uniform 84 mo
    # anchor (Beta(1.1,1.1) -> Beta(2,1.5)); widen eta_u (0.4 -> 0.6). The old
    # joint U band sat ~2.5-3x below the empirical mean at 24-48 mo.
    p_slope_low_u_alpha=1.0,
    p_slope_low_u_beta=7.0,
    p_slope_hi_u_alpha=2.0,
    p_slope_hi_u_beta=1.5,
    eta_u_sigma=0.6,
    # q anchors (weakly-informative, non-double-dipping): broadened from the
    # VG07-posterior-derived Beta(3,22)/Beta(20,4). q_low ~ Beta(2,12) keeps the
    # independent TD q(~12mo) ~= 0.12 centre; q_high ~ Beta(3,2) is broad (5-95%
    # ~0.25-0.90, no independent DS source), so the data set the 84 mo level.
    p_slope_low_q_alpha=2.0,
    p_slope_low_q_beta=12.0,
    p_slope_hi_q_alpha=3.0,
    p_slope_hi_q_beta=2.0,
    tau_u_sigma=0.5,
    tau_q_sigma=0.5,
    use_subject_re_u=True,
    tau_subj_u_sigma=0.5,
)

VG09 = BivariateModelDefinition(
    model_id="VG09",
    config_name="age-understood-spoken-ds-re-subj-uq",
    banner=(
        "Fitting Model VG09: Joint model with study + subject random intercepts on U and q"
        " (A -> U, A -> S, U -> S) - Down syndrome"
    ),
    population=Population.DOWN_SYNDROME,
    n_trials=810,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    gp_domain_months=_DS_GP_DOMAIN_MONTHS,
    # Understood anchors aligned with the recalibrated VG02 (#135): raise the
    # 24 mo anchor (Beta(1,10) -> Beta(1,7)) and soften the near-uniform 84 mo
    # anchor (Beta(1.1,1.1) -> Beta(2,1.5)); widen eta_u (0.4 -> 0.6). The old
    # joint U band sat ~2.5-3x below the empirical mean at 24-48 mo.
    p_slope_low_u_alpha=1.0,
    p_slope_low_u_beta=7.0,
    p_slope_hi_u_alpha=2.0,
    p_slope_hi_u_beta=1.5,
    eta_u_sigma=0.6,
    # q anchors (weakly-informative, non-double-dipping): broadened from the
    # VG07-posterior-derived Beta(3,22)/Beta(20,4). q_low ~ Beta(2,12) keeps the
    # independent TD q(~12mo) ~= 0.12 centre; q_high ~ Beta(3,2) is broad (5-95%
    # ~0.25-0.90, no independent DS source), so the data set the 84 mo level.
    p_slope_low_q_alpha=2.0,
    p_slope_low_q_beta=12.0,
    p_slope_hi_q_alpha=3.0,
    p_slope_hi_q_beta=2.0,
    tau_u_sigma=0.5,
    tau_q_sigma=0.5,
    use_subject_re_u=True,
    tau_subj_u_sigma=0.5,
    use_subject_re_q=True,
    tau_subj_q_sigma=0.5,
)

VG10 = BivariateModelDefinition(
    model_id="VG10",
    config_name="age-understood-spoken-ds-re-subj-uq-anchored",
    banner=(
        "Fitting Model VG10: VG09 + GP anchored at reference age"
        " (A -> U, A -> S, U -> S) - Down syndrome"
    ),
    population=Population.DOWN_SYNDROME,
    n_trials=810,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    gp_domain_months=_DS_GP_DOMAIN_MONTHS,
    # Understood anchors aligned with the recalibrated VG02 (#135): raise the
    # 24 mo anchor (Beta(1,10) -> Beta(1,7)) and soften the near-uniform 84 mo
    # anchor (Beta(1.1,1.1) -> Beta(2,1.5)); widen eta_u (0.4 -> 0.6). The old
    # joint U band sat ~2.5-3x below the empirical mean at 24-48 mo.
    p_slope_low_u_alpha=1.0,
    p_slope_low_u_beta=7.0,
    p_slope_hi_u_alpha=2.0,
    p_slope_hi_u_beta=1.5,
    eta_u_sigma=0.6,
    # q anchors (weakly-informative, non-double-dipping): broadened from the
    # VG07-posterior-derived Beta(3,22)/Beta(20,4). q_low ~ Beta(2,12) keeps the
    # independent TD q(~12mo) ~= 0.12 centre; q_high ~ Beta(3,2) is broad (5-95%
    # ~0.25-0.90, no independent DS source), so the data set the 84 mo level.
    p_slope_low_q_alpha=2.0,
    p_slope_low_q_beta=12.0,
    p_slope_hi_q_alpha=3.0,
    p_slope_hi_q_beta=2.0,
    tau_u_sigma=0.5,
    tau_q_sigma=0.5,
    use_subject_re_u=True,
    tau_subj_u_sigma=0.5,
    use_subject_re_q=True,
    tau_subj_q_sigma=0.5,
    # GP anchor constraint (Option D) — applied symmetrically to both trajectories
    anchor_g_u_at_ref=True,
    anchor_g_q_at_ref=True,
    gp_anchor_age_months=54.0,
)

VG11 = UnivariateModelDefinition(
    model_id="VG11",
    config_name="age-spoken-td-re",
    banner=(
        "Fitting Model VG11: Words spoken (TD) with dataset-level study random intercepts"
    ),
    population=Population.TYPICALLY_DEVELOPING,
    outcome=Outcome.SPOKEN,
    # Common 810-item reference inventory for TD/DS comparisons.
    n_trials=810,
    slope_anchors=(12, 26),
    ages_query=[9, 12, 15, 18, 21, 24, 27, 30],
    gp_domain_months=_TD_GP_DOMAIN_MONTHS,
    # Spoken trajectory priors shared with VG03 (see the note there): lower the
    # 12 mo anchor for delayed TD production, soften the 26 mo anchor, widen eta.
    p_slope_low_alpha=1.0,
    p_slope_low_beta=30.0,
    p_slope_hi_alpha=1.3,
    p_slope_hi_beta=1.3,
    eta_sigma=0.5,
    # Use all bivariate-capable rows (WG + Oxford CDI) plus WS production rows.
    # Study REs absorb between-lab variation, so subsampling is not needed.
    sample_fraction=1.0,
    # Study-level random intercepts on the spoken trajectory
    tau_study_sigma=0.5,
    # Drop datasets with <200 observations (issue #55): roughly halves the study
    # count while retaining >97% of observations.
    min_study_observations=200,
    use_subject_re=True,
    tau_subject_sigma=0.5,
    # Anchor the GP at the midpoint of slope_anchors (19 months) to remove the
    # GP–intercept ridge that arises when study REs are present.
    anchor_g_at_ref=True,
    gp_anchor_age_months=19.0,
)

VG12 = UnivariateModelDefinition(
    model_id="VG12",
    config_name="age-understood-td-re",
    banner=(
        "Fitting Model VG12: Words understood (TD) with dataset-level study random intercepts"
    ),
    population=Population.TYPICALLY_DEVELOPING,
    outcome=Outcome.UNDERSTOOD,
    # Common 810-item reference inventory for TD/DS comparisons.
    n_trials=810,
    slope_anchors=(12, 26),
    ages_query=[9, 12, 15, 18, 21, 24, 27, 30],
    # As in VG04, observed comprehension ends at 25 months while reporting
    # reaches 30; this preserves the established 8-30 month HSGP domain.
    gp_domain_months=_TD_GP_DOMAIN_MONTHS,
    # Understood trajectory priors shared with VG04 (see the note there): the
    # 12 mo low anchor is anchored to the independent Wordbank comprehension norm
    # (~83 words), while the 26 mo high anchor has no independent CDI norm (WS is
    # production-only) and remains broad regularisation / a sensitivity target.
    p_slope_low_alpha=1.2,
    p_slope_low_beta=8.0,
    p_slope_hi_alpha=1.3,
    p_slope_hi_beta=1.3,
    eta_sigma=0.5,
    # WG + Oxford CDI only (WS comprehension is a production proxy).
    # Study REs absorb between-lab variation, so subsampling is not needed.
    sample_fraction=1.0,
    # Study-level random intercepts on the understood trajectory
    tau_study_sigma=0.5,
    # Drop datasets with <200 observations (issue #55): roughly halves the study
    # count while retaining >97% of observations.
    min_study_observations=200,
    use_subject_re=True,
    tau_subject_sigma=0.5,
    # Anchor the GP at the midpoint of slope_anchors (19 months).
    anchor_g_at_ref=True,
    gp_anchor_age_months=19.0,
)

VG13 = BivariateModelDefinition(
    model_id="VG13",
    config_name="age-understood-spoken-td-re-young",
    banner=(
        "Fitting Model VG13: Joint words understood + spoken (TD, 8–18 months) "
        "with dataset-level study random intercepts"
    ),
    population=Population.TYPICALLY_DEVELOPING,
    # Common 810-item reference inventory. Counts from WG (ceiling 396) and
    # Oxford CDI (ceiling 418) are interpreted on this shared reference scale;
    # source-form ceilings remain an interpretation caveat.
    n_trials=810,
    # Restrict to 8–18 months where WG/Oxford CDI data are dense and the WS
    # bias (production proxy comprehension) is avoided entirely.
    max_age_months=18,
    slope_anchors=(10, 16),
    ages_query=[8, 10, 12, 14, 16, 18],
    gp_domain_months=_YOUNG_TD_GP_DOMAIN_MONTHS,
    # Understood trajectory — Wordbank TD normative medians (published deciles):
    # ~50 words/810 at 10 mo, ~180 at 16 mo. Beta(1,15) (10 mo, median ~36) sits a
    # touch below the norm floor by design (Fenson: percentiles are unstable where
    # a skill is just emerging — re-centre toward norms, do not tighten). The old
    # 16 mo Beta(2,2) (~400 words) overshot the ~177 norm ~2x AND sat against the
    # WG comprehension ceiling (396/810 = 0.489); Beta(2,6) (median 0.228, ~185
    # words) matches the norm and stays clear of the ceiling. In-sample means
    # (~51, ~178) corroborate. See PRIORS.md, "TD anchor priors vs Wordbank norms".
    p_slope_low_u_alpha=1.0,
    p_slope_low_u_beta=15.0,
    p_slope_hi_u_alpha=2.0,
    p_slope_hi_u_beta=6.0,
    # Production ratio q = P(speak | understood). Independent norm-derived TD q(a)
    # (ratio of Wordbank median production to median comprehension): ~0.12 at
    # 10 mo rising to ~0.19 at 16 mo (PRIORS.md, "Production ratio q(a) from
    # norms"). The shared bivariate defaults (lo Beta(1,1.5)~0.4, hi Beta(2,1.2)
    # ~0.62) are tuned for the DS 24/84 mo window and sit ~3x above this young-TD
    # curve, compounding with U to overshoot spoken ~5x. Set window-appropriate
    # anchors at/just below the norm floor: lo Beta(1,10) (median ~0.067), hi
    # Beta(2,7) (median ~0.201). The in-sample q (~0.09 at 10 mo, ~0.23 at 16 mo)
    # corroborates.
    p_slope_low_q_alpha=1.0,
    p_slope_low_q_beta=10.0,
    p_slope_hi_q_alpha=2.0,
    p_slope_hi_q_beta=7.0,
    # Use all available bivariate rows in the 8–18 month window; study REs
    # absorb between-lab variation so no subsampling is required.
    sample_fraction=1.0,
    # Dataset-level study random intercepts on both trajectories
    tau_u_sigma=0.5,
    tau_q_sigma=0.5,
    # Drop datasets with <200 observations (issue #55): roughly halves the study
    # count while retaining >97% of observations.
    min_study_observations=200,
    use_subject_re_u=True,
    tau_subj_u_sigma=0.5,
    use_subject_re_q=True,
    tau_subj_q_sigma=0.5,
    # Anchor GPs at the midpoint of slope_anchors (13 months)
    anchor_g_u_at_ref=True,
    anchor_g_q_at_ref=True,
    gp_anchor_age_months=13.0,
)

VG14 = TrivariateModelDefinition(
    model_id="VG14",
    config_name="age-understood-spoken-signed-ds",
    banner=(
        "Fitting Model VG14: Trivariate model of words understood, spoken and"
        " signed (A -> U, A -> S, A -> Sign; U -> S, U -> Sign)"
    ),
    population=Population.DOWN_SYNDROME,
    n_trials=810,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    gp_domain_months=_DS_GP_DOMAIN_MONTHS,
    # Understood trajectory: matches VG05 — anchors aligned with the recalibrated
    # VG02 (#135): raise the 24 mo anchor (Beta(1,10) -> Beta(1,7)), soften the
    # near-uniform 84 mo anchor (Beta(1.1,1.1) -> Beta(2,1.5)), widen eta_u to 0.6.
    p_slope_low_u_alpha=1.0,
    p_slope_low_u_beta=7.0,
    p_slope_hi_u_alpha=2.0,
    p_slope_hi_u_beta=1.5,
    eta_u_sigma=0.6,
    # Spoken ratio q (weakly-informative, non-double-dipping): broadened from the
    # VG07-posterior-derived Beta(3,22)/Beta(20,4). q_low ~ Beta(2,12) keeps the
    # independent TD q(~12mo) ~= 0.12 centre; q_high ~ Beta(3,2) is broad (5-95%
    # ~0.25-0.90, no independent DS source), so the data set the 84 mo level.
    p_slope_low_q_alpha=2.0,
    p_slope_low_q_beta=12.0,
    p_slope_hi_q_alpha=3.0,
    p_slope_hi_q_beta=2.0,
    # Signed ratio r uses the three-anchor tent + GP defined above.  uk_01's
    # signed-only field and uk_06's unverified field are excluded from the signed
    # likelihood by default; their understood/spoken observations remain.
)

VG15 = JointModelDefinition(
    model_id="VG15",
    config_name="age-joint-signspeech-ds",
    banner=(
        "Fitting Model VG15: Joint sign/speech model with within-understood"
        " association (psi), study + subject random intercepts, and GP anchoring"
        " - Down syndrome"
    ),
    population=Population.DOWN_SYNDROME,
    n_trials=810,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    gp_domain_months=_DS_GP_DOMAIN_MONTHS,
    # r/q/p_U priors seeded from the (uk_06-included) VG14 fit (see dataclass
    # defaults); psi ~ logNormal(0.3, 0.5) (weakly positive, spans independence);
    # study random intercepts on f_U/g/h (tau_*=0.5).
    #
    # Issue #59 — subject random intercepts throughout + VG10 stabilisation:
    # Option A (ported from VG10), now broadened: the q anchors are
    # weakly-informative (q_low ~ Beta(2,12) at the independent TD q ~= 0.12
    # centre; q_high ~ Beta(3,2) broad, no independent DS source), replacing the
    # VG07-posterior-derived Beta(3,22)/Beta(20,4). The u anchors are left
    # unchanged, matching VG10. The signed mean is a three-anchor hump (tent),
    # inherited from the JointModelDefinition dataclass defaults (young/peak/old
    # sign anchors + GP), so there is no monotone signed slope to tighten; the
    # anchors set the level and the GP carries smooth departures. Option D (below)
    # removes the GP<->intercept ridge.
    p_slope_low_q_alpha=2.0,
    p_slope_low_q_beta=12.0,
    p_slope_hi_q_alpha=3.0,
    p_slope_hi_q_beta=2.0,
    # Subject random intercepts on all three trajectories. Signed data has more
    # repeated-subject structure than first feared (substantial repeats across
    # uk_01/02/04/05), so the sign-subject RE is strongly data-identified — its
    # scale sits *well above* the HalfNormal(0.5) prior (dev posterior tau_subj_sign
    # ~1.4, tight), reflecting large between-child variation in signing, and it
    # improves out-of-sample fit. Kept at 0.5 (porting VG10); use_subject_re_sign
    # gates a one-line fallback to study-RE-only if a future fit misbehaves. Note:
    # the four-cell DM is fed population+study marginals only, so this RE does not
    # pull the headline association psi (see the engine comment + the #59 note).
    use_subject_re_u=True,
    tau_subj_u_sigma=0.5,
    use_subject_re_q=True,
    tau_subj_q_sigma=0.5,
    use_subject_re_sign=True,
    tau_subj_sign_sigma=0.5,
    # Option D (ported from VG10): per-draw GP anchor at the reference age
    # (54 mo = midpoint of the (24, 84) anchors), applied to all three GPs to
    # remove the GP<->intercept redundancy that worsens once subject REs add
    # another level-carrying term to each predictor.
    anchor_g_u_at_ref=True,
    anchor_g_q_at_ref=True,
    anchor_g_sign_at_ref=True,
    gp_anchor_age_months=54.0,
)

# ============================================================
# VG16 — within-child cross-lag (issue #113): VG09 + prior understood -> current q
# ============================================================

VG16 = BivariateModelDefinition(
    model_id="VG16",
    config_name="age-understood-spoken-ds-re-subj-uq-crosslag",
    banner=(
        "Fitting Model VG16: VG09 + cross-lag (prior understood -> current q;"
        " bias-robust population-relative baseline) - Down syndrome"
    ),
    population=Population.DOWN_SYNDROME,
    n_trials=810,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    gp_domain_months=_DS_GP_DOMAIN_MONTHS,
    # Understood anchors aligned with the recalibrated VG02 (#135): raise the
    # 24 mo anchor (Beta(1,10) -> Beta(1,7)) and soften the near-uniform 84 mo
    # anchor (Beta(1.1,1.1) -> Beta(2,1.5)); widen eta_u (0.4 -> 0.6). The old
    # joint U band sat ~2.5-3x below the empirical mean at 24-48 mo.
    p_slope_low_u_alpha=1.0,
    p_slope_low_u_beta=7.0,
    p_slope_hi_u_alpha=2.0,
    p_slope_hi_u_beta=1.5,
    eta_u_sigma=0.6,
    # q anchors (weakly-informative, non-double-dipping): broadened from the
    # VG07-posterior-derived Beta(3,22)/Beta(20,4). q_low ~ Beta(2,12) keeps the
    # independent TD q(~12mo) ~= 0.12 centre; q_high ~ Beta(3,2) is broad (5-95%
    # ~0.25-0.90, no independent DS source), so the data set the 84 mo level.
    p_slope_low_q_alpha=2.0,
    p_slope_low_q_beta=12.0,
    p_slope_hi_q_alpha=3.0,
    p_slope_hi_q_beta=2.0,
    tau_u_sigma=0.5,
    tau_q_sigma=0.5,
    use_subject_re_u=True,
    tau_subj_u_sigma=0.5,
    use_subject_re_q=True,
    tau_subj_q_sigma=0.5,
    use_cross_lag=True,
    # Headline uses the population-relative baseline: with 2-wave-dominated data the
    # pure within-child (own-intercept) baseline is biased by the short-T / Nickell
    # / errors-in-variables mechanics (dev: beta -0.60 [-0.85,-0.35], an artifact),
    # while the population-relative estimate is null (dev: +0.05 [-0.07,0.17]). The
    # within-child variant is reported as a cautionary contrast. See the scoping note.
    lag_baseline="population",
    beta_lag_mu=0.0,
    beta_lag_sigma=0.5,
)

MODEL_REGISTRY: dict[
    str,
    UnivariateModelDefinition
    | BivariateModelDefinition
    | TrivariateModelDefinition
    | JointModelDefinition,
] = {
    "vg01": VG01,
    "vg02": VG02,
    "vg03": VG03,
    "vg04": VG04,
    "vg05": VG05,
    "vg07": VG07,
    "vg08": VG08,
    "vg09": VG09,
    "vg10": VG10,
    "vg11": VG11,
    "vg12": VG12,
    "vg13": VG13,
    "vg14": VG14,
    "vg15": VG15,
    "vg16": VG16,
}


def _validate_positive_scale_fields(value, *, path: str) -> None:
    """Recursively validate distribution shape and scale parameters."""
    if not is_dataclass(value) or isinstance(value, type):
        return
    for item in fields(value):
        field_value = getattr(value, item.name)
        field_path = f"{path}.{item.name}"
        if is_dataclass(field_value):
            _validate_positive_scale_fields(field_value, path=field_path)
        elif item.name.endswith(("_alpha", "_beta", "_sigma")):
            if (
                not isinstance(field_value, (int, float))
                or not math.isfinite(field_value)
                or field_value <= 0
            ):
                raise ValueError(f"{field_path} must be positive; got {field_value!r}.")


def validate_model_definition(definition) -> None:
    """Fail early when a declarative model specification is internally invalid."""
    prefix = getattr(definition, "model_id", type(definition).__name__)
    if not re.fullmatch(r"VG\d{2}", definition.model_id):
        raise ValueError(f"{prefix}.model_id must have the form VG01.")
    if not re.fullmatch(r"[A-Za-z0-9]+(?:[A-Za-z0-9_-]*[A-Za-z0-9])?", definition.config_name):
        raise ValueError(f"{prefix}.config_name must be a non-empty path-safe label.")
    if definition.n_trials <= 0:
        raise ValueError(f"{prefix}.n_trials must be positive.")
    if (
        len(definition.slope_anchors) != 2
        or not all(math.isfinite(age) for age in definition.slope_anchors)
        or not (definition.slope_anchors[0] < definition.slope_anchors[1])
    ):
        raise ValueError(f"{prefix}.slope_anchors must be ordered (low, high).")
    if not definition.ages_query:
        raise ValueError(f"{prefix}.ages_query must not be empty.")
    if not all(
        isinstance(age, (int, float)) and math.isfinite(age)
        for age in definition.ages_query
    ) or list(definition.ages_query) != sorted(set(definition.ages_query)):
        raise ValueError(f"{prefix}.ages_query must be sorted with no duplicates.")
    if definition.n_plot <= 0:
        raise ValueError(f"{prefix}.n_plot must be positive.")
    if len(definition.ell_months_range) != 2 or not (
        0 < definition.ell_months_range[0] < definition.ell_months_range[1]
    ):
        raise ValueError(
            f"{prefix}.ell_months_range must be positive and ordered (low, high)."
        )

    domain = definition.gp_domain_months
    if domain is not None:
        if (
            len(domain) != 2
            or not all(math.isfinite(age) for age in domain)
            or not (domain[0] < domain[1])
        ):
            raise ValueError(f"{prefix}.gp_domain_months must be ordered (low, high).")
        required_ages = [*definition.ages_query, *definition.slope_anchors]
        anchor_age = getattr(definition, "gp_anchor_age_months", None)
        if anchor_age is not None:
            required_ages.append(anchor_age)
        sign_anchors = getattr(definition, "sign_anchor_ages", ())
        required_ages.extend(sign_anchors)
        if min(required_ages) < domain[0] or max(required_ages) > domain[1]:
            raise ValueError(f"{prefix} reference and query ages must lie in its GP domain.")

    sample_fraction = getattr(definition, "sample_fraction", 1.0)
    if not math.isfinite(sample_fraction) or not 0 < sample_fraction <= 1:
        raise ValueError(f"{prefix}.sample_fraction must be in (0, 1].")
    min_study = getattr(definition, "min_study_observations", None)
    if min_study is not None and min_study <= 0:
        raise ValueError(f"{prefix}.min_study_observations must be positive.")
    max_age = getattr(definition, "max_age_months", None)
    if max_age is not None and max_age <= 0:
        raise ValueError(f"{prefix}.max_age_months must be positive.")
    sign_anchors = getattr(definition, "sign_anchor_ages", None)
    if sign_anchors is not None and (
        len(sign_anchors) != 3
        or not all(math.isfinite(age) for age in sign_anchors)
        or tuple(sign_anchors) != tuple(sorted(sign_anchors))
    ):
        raise ValueError(f"{prefix}.sign_anchor_ages must be three ordered ages.")
    if getattr(definition, "lag_baseline", "within") not in {"within", "population"}:
        raise ValueError(f"{prefix}.lag_baseline must be 'within' or 'population'.")
    if getattr(definition, "use_cross_lag", False) and not getattr(
        definition, "use_subject_re_u", False
    ):
        raise ValueError(f"{prefix} cross-lag requires use_subject_re_u=True.")

    _validate_positive_scale_fields(definition, path=prefix)


def validate_model_registry() -> None:
    """Validate every registered specification and registry key at import time."""
    labels: set[str] = set()
    for key, definition in MODEL_REGISTRY.items():
        validate_model_definition(definition)
        if definition.gp_domain_months is None:
            raise ValueError(
                f"Registered model {definition.model_id} must declare gp_domain_months."
            )
        if key != definition.model_id.lower():
            raise ValueError(
                f"Registry key {key!r} does not match {definition.model_id!r}."
            )
        label = f"{definition.model_id}-{definition.config_name}"
        if label in labels:
            raise ValueError(f"Duplicate model output label: {label}.")
        labels.add(label)


validate_model_registry()
