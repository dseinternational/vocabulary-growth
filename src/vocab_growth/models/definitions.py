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
from dataclasses import dataclass, field
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
    eta_q_sigma: float = 0.4
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

    # -- Production ratio (q) slope priors --
    p_slope_low_q_alpha: float = 1.0
    p_slope_low_q_beta: float = 1.5
    p_slope_hi_q_alpha: float = 2.0
    p_slope_hi_q_beta: float = 1.2

    # -- Signed ratio (r) mean prior (intercept-only) --
    # The empirical signed/understood ratio is a *hump* (near zero < 24 mo, peak
    # in the preschool years, receding after), not a monotone trend. The signed
    # mean trend is INTERCEPT-ONLY (no age slope) — this structurally removes the
    # original "free slope tilts down and extrapolates to a spurious ~58% signed
    # at 12 mo" failure mode, so the level no longer needs a tight prior. A single
    # weakly-informative intercept (logit scale) lets the data set the signed
    # level and old-age tail; the GP (eta_sign, below) carries the rise-then-fall.
    intercept_sign_mu: float = math.log(0.15 / 0.85)
    """Normal mu for the signed-ratio intercept (logit scale, ~0.15)."""
    intercept_sign_sigma: float = 0.75
    """Normal sigma for the signed-ratio intercept (logit scale). Wide enough to
    span a broad signed level once combined with the GP; calibrated against the
    prior-predictive r(a) band (broad, not piling at 0 or 1)."""

    # -- Shared GP / amplitude priors --
    ell_unit_u_alpha: float = 3.0
    ell_unit_u_beta: float = 3.0
    eta_u_sigma: float = 0.4
    ell_unit_q_alpha: float = 3.0
    ell_unit_q_beta: float = 3.0
    eta_q_sigma: float = 0.4
    # Signed GP favours a shorter lengthscale (~9 mo) than U/q so the signing
    # peak can stand apart from the post-60 mo collapse to near-zero, rather than
    # being smoothed into a monotone decline. (Shorter still only adds wiggle
    # without moving the population peak past ~30 mo: the late-preschool data
    # spike is too sparse/overdispersed to pull the population ratio there.)
    ell_unit_sign_alpha: float = 2.0
    ell_unit_sign_beta: float = 5.0
    eta_sign_sigma: float = 1.0
    """HalfNormal scale for the signed-ratio GP amplitude. Loosened well above
    the spoken-ratio level so the GP has the amplitude to bend the (flat-mean)
    signed ratio into the empirical rise-then-fall hump on the logit scale."""
    ell_months_range: tuple[int, int] = (6, 18)
    n_plot: int = 500
    kappa_u: KappaPriorParams = field(default_factory=KappaPriorParams)
    kappa_s: KappaPriorParams = field(default_factory=KappaPriorParams)
    kappa_sign: KappaPriorParams = field(default_factory=KappaPriorParams)

    # -- Signed data inclusion --
    include_uk06: bool = True
    """If True (default), the uk_06 'signed' counts are included in the signed
    likelihood. uk_06 records a real signing-production count (11 obs at 60-115
    mo, often comparable to or exceeding spoken — signing implies understanding,
    so 'understands-and-signs' is a sign), not a comprehension measure. The flag
    is kept for reversibility; the open question is whether uk_06's signed counts
    are coded comparably to uk_02/04/05 (no field dictionary), not the construct."""

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

    # -- Understood (U) slope priors (seeded from VG05/VG14) --
    p_slope_low_u_alpha: float = 1.0
    p_slope_low_u_beta: float = 10.0
    p_slope_hi_u_alpha: float = 1.1
    p_slope_hi_u_beta: float = 1.1

    # -- Speak-given-understood (q) slope priors (bivariate defaults) --
    p_slope_low_q_alpha: float = 1.0
    p_slope_low_q_beta: float = 1.5
    p_slope_hi_q_alpha: float = 2.0
    p_slope_hi_q_beta: float = 1.2

    # -- Sign-given-understood (r) mean prior (intercept-only, matching VG14) --
    # The signed mean is intercept-only (no age slope): a free slope would
    # extrapolate below the data floor (< ~18 mo). A single weakly-informative
    # intercept lets the data set the grand-mean level; the study REs carry
    # between-study level differences and the GP carries the rise-then-fall hump.
    intercept_sign_mu: float = math.log(0.15 / 0.85)
    """Normal mu for the signed-ratio intercept (logit scale, ~0.15)."""
    intercept_sign_sigma: float = 0.75
    """Normal sigma for the signed-ratio intercept (logit scale)."""

    # -- Shared GP / amplitude priors (sign GP looser + shorter, per VG14) --
    ell_unit_u_alpha: float = 3.0
    ell_unit_u_beta: float = 3.0
    eta_u_sigma: float = 0.4
    ell_unit_q_alpha: float = 3.0
    ell_unit_q_beta: float = 3.0
    eta_q_sigma: float = 0.4
    ell_unit_sign_alpha: float = 2.0
    ell_unit_sign_beta: float = 5.0
    eta_sign_sigma: float = 1.0
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
    include_uk06: bool = True

    # -- nz_01 (Foster-Cohen) produced cross-tab inclusion --
    include_nz01_cells: bool = True
    """If True (default), nz_01's produced modality cross-tab (word-only / sign-only
    / both) enters via a within-produced Dirichlet-Multinomial that informs psi/q/r
    (see common_joint_modality). If False, nz_01 is excluded from VG15 entirely
    (its production-only, 675-item marginals are not comparable to the 800-item
    marginal likelihoods); the flag is kept for reversibility and for isolating
    nz_01's pull on psi."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.JOINT


# ============================================================
# Model instances
# ============================================================

VG01 = UnivariateModelDefinition(
    model_id="VG01",
    config_name="age-spoken-ds",
    banner="Fitting Model VG01: Influence of age on words spoken (A -> S)",
    population=Population.DOWN_SYNDROME,
    outcome=Outcome.SPOKEN,
    n_trials=800,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    p_slope_low_alpha=1.0,
    p_slope_low_beta=15.0,
    p_slope_hi_alpha=1.1,
    p_slope_hi_beta=1.1,
)

VG02 = UnivariateModelDefinition(
    model_id="VG02",
    config_name="age-understood-ds",
    banner="Fitting Model VG02: Influence of age on words understood (A -> U)",
    population=Population.DOWN_SYNDROME,
    outcome=Outcome.UNDERSTOOD,
    n_trials=800,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    p_slope_low_alpha=1.0,
    p_slope_low_beta=10.0,
    p_slope_hi_alpha=1.1,
    p_slope_hi_beta=1.1,
)

VG03 = UnivariateModelDefinition(
    model_id="VG03",
    config_name="age-spoken-td",
    banner="Fitting Model VG03: Influence of age on words spoken (A -> S)",
    population=Population.TYPICALLY_DEVELOPING,
    outcome=Outcome.SPOKEN,
    # Common 800-item reference inventory for TD/DS comparisons. The TD
    # loader uses WG and Oxford CDI production plus WS production-only rows.
    n_trials=800,
    slope_anchors=(12, 26),
    ages_query=[9, 12, 15, 18, 21, 24, 27, 30],
    p_slope_low_alpha=1.0,
    p_slope_low_beta=15.0,
    p_slope_hi_alpha=1.5,
    p_slope_hi_beta=1.1,
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
    # Common 800-item reference inventory for TD/DS comparisons. The TD
    # loader excludes WS comprehension because it is a production proxy.
    n_trials=800,
    slope_anchors=(12, 26),
    ages_query=[9, 12, 15, 18, 21, 24, 27, 30],
    p_slope_low_alpha=1.0,
    p_slope_low_beta=20.0,
    p_slope_hi_alpha=1.5,
    p_slope_hi_beta=1.1,
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
    n_trials=800,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    p_slope_low_u_alpha=1.0,
    p_slope_low_u_beta=10.0,
    p_slope_hi_u_alpha=1.1,
    p_slope_hi_u_beta=1.1,
)

VG07 = BivariateModelDefinition(
    model_id="VG07",
    config_name="age-understood-spoken-ds-re",
    banner=(
        "Fitting Model VG07: Joint model with study random intercepts"
        " (A -> U, A -> S, U -> S) - Down syndrome"
    ),
    population=Population.DOWN_SYNDROME,
    n_trials=800,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    p_slope_low_u_alpha=1.0,
    p_slope_low_u_beta=10.0,
    p_slope_hi_u_alpha=1.1,
    p_slope_hi_u_beta=1.1,
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
    n_trials=800,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    p_slope_low_u_alpha=1.0,
    p_slope_low_u_beta=10.0,
    p_slope_hi_u_alpha=1.1,
    p_slope_hi_u_beta=1.1,
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
    n_trials=800,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    p_slope_low_u_alpha=1.0,
    p_slope_low_u_beta=10.0,
    p_slope_hi_u_alpha=1.1,
    p_slope_hi_u_beta=1.1,
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
        "Fitting Model VG10: VG09 + tighter q anchor priors + GP anchored at"
        " reference age (A -> U, A -> S, U -> S) - Down syndrome"
    ),
    population=Population.DOWN_SYNDROME,
    n_trials=800,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    p_slope_low_u_alpha=1.0,
    p_slope_low_u_beta=10.0,
    p_slope_hi_u_alpha=1.1,
    p_slope_hi_u_beta=1.1,
    # Tighter q-anchor priors (Option A) — informed by VG07 posterior
    p_slope_low_q_alpha=3.0,
    p_slope_low_q_beta=22.0,
    p_slope_hi_q_alpha=20.0,
    p_slope_hi_q_beta=4.0,
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
    # Common 800-item reference inventory for TD/DS comparisons.
    n_trials=800,
    slope_anchors=(12, 26),
    ages_query=[9, 12, 15, 18, 21, 24, 27, 30],
    p_slope_low_alpha=1.0,
    p_slope_low_beta=15.0,
    p_slope_hi_alpha=1.5,
    p_slope_hi_beta=1.1,
    # Use all bivariate-capable rows (WG + Oxford CDI) plus WS production rows.
    # Study REs absorb between-lab variation, so subsampling is not needed.
    sample_fraction=1.0,
    # Study-level random intercepts on the spoken trajectory
    tau_study_sigma=0.5,
    # Drop datasets with <200 observations (issue #55): roughly halves the study
    # count while retaining >97% of observations.
    min_study_observations=200,
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
    # Common 800-item reference inventory for TD/DS comparisons.
    n_trials=800,
    slope_anchors=(12, 26),
    ages_query=[9, 12, 15, 18, 21, 24, 27, 30],
    p_slope_low_alpha=1.0,
    p_slope_low_beta=20.0,
    p_slope_hi_alpha=1.5,
    p_slope_hi_beta=1.1,
    # WG + Oxford CDI only (WS comprehension is a production proxy).
    # Study REs absorb between-lab variation, so subsampling is not needed.
    sample_fraction=1.0,
    # Study-level random intercepts on the understood trajectory
    tau_study_sigma=0.5,
    # Drop datasets with <200 observations (issue #55): roughly halves the study
    # count while retaining >97% of observations.
    min_study_observations=200,
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
    # Common 800-item reference inventory. Counts from WG (ceiling 396) and
    # Oxford CDI (ceiling 418) are interpreted on this shared reference scale;
    # source-form ceilings remain an interpretation caveat.
    n_trials=800,
    # Restrict to 8–18 months where WG/Oxford CDI data are dense and the WS
    # bias (production proxy comprehension) is avoided entirely.
    max_age_months=18,
    slope_anchors=(10, 16),
    ages_query=[8, 10, 12, 14, 16, 18],
    p_slope_low_u_alpha=1.0,
    p_slope_low_u_beta=15.0,
    p_slope_hi_u_alpha=2.0,
    p_slope_hi_u_beta=2.0,
    # Use all available bivariate rows in the 8–18 month window; study REs
    # absorb between-lab variation so no subsampling is required.
    sample_fraction=1.0,
    # Dataset-level study random intercepts on both trajectories
    tau_u_sigma=0.5,
    tau_q_sigma=0.5,
    # Drop datasets with <200 observations (issue #55): roughly halves the study
    # count while retaining >97% of observations.
    min_study_observations=200,
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
    n_trials=800,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    # Understood trajectory: reuse VG05's understood slope priors.
    p_slope_low_u_alpha=1.0,
    p_slope_low_u_beta=10.0,
    p_slope_hi_u_alpha=1.1,
    p_slope_hi_u_beta=1.1,
    # Spoken ratio q: bivariate defaults.
    # Signed ratio r: intercept-only mean (data-set level) + loosened GP
    #   (eta_sign=1.0) carrying the rise-then-fall hump (see dataclass).
    # uk_06 signed included by default (include_uk06=True); kappa_sign default.
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
    n_trials=800,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    # r/q/p_U priors seeded from the (uk_06-included) VG14 fit (see dataclass
    # defaults); psi ~ logNormal(0.3, 0.5) (weakly positive, spans independence);
    # study random intercepts on f_U/g/h (tau_*=0.5).
    #
    # Issue #59 — subject random intercepts throughout + VG10 stabilisation:
    # Option A (ported from VG10): tighter q anchor priors, informed by the VG07
    # posterior (mean ~0.12 at 24 mo, ~0.83 at 84 mo). The u anchors are left
    # unchanged, matching VG10. The signed mean is intercept-only, so there is no
    # signed slope anchor to tighten; intercept_sign_sigma is left at 0.75 — it is
    # already data-informed (posterior sd ~0.48 < prior in #54's freeing fix), and
    # Option D (below) removes the GP<->intercept ridge without re-tightening the
    # freed level.
    p_slope_low_q_alpha=3.0,
    p_slope_low_q_beta=22.0,
    p_slope_hi_q_alpha=20.0,
    p_slope_hi_q_beta=4.0,
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
    n_trials=800,
    slope_anchors=(24, 84),
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    p_slope_low_u_alpha=1.0,
    p_slope_low_u_beta=10.0,
    p_slope_hi_u_alpha=1.1,
    p_slope_hi_u_beta=1.1,
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
