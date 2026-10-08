# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Statistical definitions for the vocabulary growth model family.

Definitions record population, outcomes, priors and data and reporting settings.
The catalogue selects the engine. Shared engines build graphs and run fitting,
diagnostics and reports. Definition, data and executable-code checks establish
separate parts of fit compatibility.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, TypedDict

from vocab_growth.models.likelihood_utils import (
    LAG_BASELINES,
    LAG_ZERO_CLIP,
    LAG_ZERO_CONTINUITY,
    LAG_ZERO_TREATMENTS,
    SPOKEN_FALLBACK_PRODUCT,
)

ENGLISH_LANGUAGES = (
    "English (American)",
    "English (Australian)",
    "English (British)",
    "English (Irish)",
)
"""Wordbank ``language`` values treated as English — the current default scope.

The ``wordbank_child`` table now holds the full multi-language Wordbank export.
Queries restrict to these English variants by default; pass a wider ``languages``
set (or ``None`` for all languages) to the loaders to widen the scope later.
See :data:`ENGLISH_AND_ROMANCE_LANGUAGES` for the widened scope the hierarchical
typically-developing models use.
"""


ROMANCE_LANGUAGES = ("Italian", "Spanish (European)")
"""Non-English Wordbank languages admitted to the hierarchical TD reference.

Italian and European Spanish match languages in the Down syndrome pool.
Italian also provides corresponding checklist forms. European Spanish uses
different forms from es_01, so matching language does not establish matching
measurement. These choices and the excluded languages are documented in
notes/202608031500-td-romance-extension.md.

The 810-item reference denominator remains an assumption. Checks of count
distributions and form ceilings cannot establish item-level equivalence.
"""

ENGLISH_AND_ROMANCE_LANGUAGES = ENGLISH_LANGUAGES + ROMANCE_LANGUAGES
"""English, Italian and European Spanish for the hierarchical TD models.

VG03 and VG04 remain English-only because they have no study effects.
Language and dataset are largely confounded, so study effects describe dataset
differences and cannot isolate language effects. Adding another language with
an existing dataset label, such as CLEX, would pool it under that study effect.
"""

KNOWN_TD_LANGUAGES = ENGLISH_AND_ROMANCE_LANGUAGES
"""Language names a definition's ``td_languages`` may reference.

A guard against silence rather than a scientific statement: a ``language`` value that
does not match Wordbank's spelling exactly returns no rows, so a typo would shrink
the reference pool without raising anything. Widening the pool means adding to
:data:`ROMANCE_LANGUAGES` — and doing the measurement checks its docstring records —
not adding a name here.
"""


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


#: ``clamp_mean_above_hi_anchor`` value that levels off **only** the production
#: ratio ``q``, leaving the understood mean free above the high anchor.
CLAMP_Q_ONLY = "q_only"


def clamp_targets(value: bool | str) -> tuple[bool, bool]:
    """Return (clamp_understood, clamp_q) for the configured mean clamp.

    CLAMP_Q_ONLY levels only the spoken-share trend. True levels both trends,
    and False leaves both unclamped. The string extends the existing field
    without adding a key to every serialised definition that declares it.
    See notes/202608141200-clamp-q-only.md for the design rationale.
    """
    if value == CLAMP_Q_ONLY:
        return False, True
    return bool(value), bool(value)


# ============================================================
# Shared prior defaults (same default kappa shape reused by every model)
# ============================================================


@dataclass(frozen=True)
class KappaPriorParams:
    """Parameters for the dispersion (kappa) prior distributions.

    The intercept-and-slope parameterisation ``kappa(z) = kappa_min +
    exp(a_kappa - b_kappa_mag * z)``. See :class:`KappaAnchorPriorParams` for the
    two-anchor alternative and why the migrated models use it.
    """

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


@dataclass(frozen=True)
class KappaAnchorPriorParams:
    """Dispersion priors on the positive excess at two reference ages.

    The curve is kappa(z) = kappa_min + exp(a_kappa + b_kappa * z).
    Priors specify kappa_min and the excess above it at two ages in months;
    the intercept and slope follow from linear interpolation of the log excess.

    The anchors retain their age meaning when the analysis frame's age mean and
    standard deviation change. Either slope sign is possible. Between the anchors,
    the log excess interpolates; outside them it extrapolates. kappa_min is a
    lower bound throughout, approached in the direction where the excess decays.
    See notes/202608020829-kappa-and-eta-q-prior-recalibration.md.
    """

    anchor_ages: tuple[float, float]
    """Reference ages (months), ordered (young, old), for the two kappa anchors."""
    kappa_min_mu: float
    """LogNormal mu for the positive lower bound kappa_min.

    It is the old-age asymptote when concentration falls with age and the young-age
    asymptote when concentration rises.
    """
    kappa_min_sigma: float
    """LogNormal sigma for kappa_min."""
    excess_young_mu: float
    """LogNormal mu for the age term at the young anchor (kappa above the asymptote)."""
    excess_young_sigma: float
    """LogNormal sigma for the age term at the young anchor."""
    excess_old_mu: float
    """LogNormal mu for the age term at the old anchor (kappa above the asymptote)."""
    excess_old_sigma: float
    """LogNormal sigma for the age term at the old anchor."""


@dataclass(frozen=True)
class SubjectVariancePartitionParams:
    """Priors for the shared scatter budget that ``tau_subject`` and ``kappa`` split.

    Selects the reparameterisation described in
    :func:`~vocab_growth.models.gp_utils.build_variance_partition`: instead of
    giving the subject-effect scale and the young dispersion anchor independent
    priors and letting them compete, one prior goes on the **total** logit-scale
    scatter at the young anchor and one on the **share** of it that persistent
    between-child differences explain.

    Set this only where the ridge is measured. It is a graph change and it moves
    the prior, so it needs a refit and a prior-predictive check; it is not a
    drop-in for models that sample cleanly already. See
    ``notes/202608050900-td-hierarchical-geometry.md`` §7.
    """

    reference_proportion: float
    """Fixed proportion ``p0`` at which a Beta-Binomial concentration is converted
    to a logit-scale variance, ``c = 1 / (p0 * (1 - p0))``.

    Deliberately a constant rather than the fitted trajectory's value at the
    anchor age: that keeps the reparameterisation independent of the mean
    function, so the priors below do not change meaning when the trend does. Set
    it to roughly the observed proportion at the young kappa anchor."""
    total_mu: float
    """LogNormal mu for the total logit-scale scatter at the young anchor."""
    total_sigma: float
    """LogNormal sigma for the total logit-scale scatter."""
    share_alpha: float
    """Beta alpha for the subject share of that scatter."""
    share_beta: float
    """Beta beta for the subject share of that scatter."""


@dataclass(frozen=True)
class SubjectSlopePriorParams:
    """Priors for a child's correlated intercept and per-year slope.

    The non-centred graph is::

        tau0    ~ HalfNormal(tau0_sigma)
        tau1    ~ HalfNormal(tau1_sigma)
        rho_raw ~ Beta(rho_eta, rho_eta); rho01 = 2 * rho_raw - 1
        z       ~ Normal(0, 1), dims (subject_id, 2)
        L       = [[tau0, 0], [rho01 * tau1, tau1 * sqrt(1 - rho01 ** 2)]]
        b       = z @ L.T
        shift(obs) = b[subject, 0] + b[subject, 1] * (age - ref_age) / 12

    At tau1 = 0, the child block reduces to a constant intercept. At perfect
    correlation it uses one deviate with an affine age loading, which differs
    from A1's exponential scale. Whole-model nesting also requires matching covariates and dispersion.
    Tests of boundary variances need an appropriate reference distribution.

    For a 2x2 correlation matrix, the transformed Beta prior is exactly LKJ(eta).
    The explicit form preserves named scales and correlation parameters in traces.
    This object replaces a scalar tau_subj_*_sigma without adding a shared
    definition field. See notes/202608141900-child-slope-implementation-plan.md.
    """

    tau0_sigma: float
    """HalfNormal scale for child variation at the reference age."""
    tau1_sigma: float
    """HalfNormal scale for child variation in slopes, in logits per year."""
    rho_eta: float = 2.0
    """LKJ concentration for the intercept-slope correlation.

    eta = 1 is uniform on (-1, 1); larger values favour correlations near zero.
    """


@dataclass(frozen=True)
class SubjectFactorPriorParams:
    """VG22's low-rank covariance over four child effects.

    The effects are (b0u, b1u, b0q, b1q), comprehension level and rate, then
    spoken-share level and rate. With z ~ Normal(0, I_rank) and b = L z,
    their covariance is L L', which is positive semidefinite.

    Loading rows are normalised and scaled by tau_i. A numerical floor protects
    rows whose unscaled norm approaches zero.
    A lower-triangular anchor block with positive diagonal removes rotational
    redundancy. The anchor order is comprehension level, spoken-share level,
    spoken-share rate, then comprehension rate. The level correlation has the
    explicit prior described by rho_uq_eta; other correlations are derived.

    At rank one, every effect is a scaled copy of one deviate. This does not
    reproduce A1's exponential age scale. VG20's child covariance can be recovered
    with zero rate loadings at rank at least two. VG19's two full-rank 2x2 blocks
    require rank four and are not nested within the supported ranks one to three.

    Rank is specified in the definition and checked by sensitivities. The
    exploratory residual fit motivated this family but does not prove its rank or
    recoverability in the registered model. See
    notes/202608221000-four-by-four-gate1.md and the current model inventory.
    """

    rank: int
    """Number of latent dimensions. Supported ranks are one, two and three."""

    tau1_u_sigma: float
    """HalfNormal scale for the spread of per-year comprehension rates. Per year,
    not per month, for the reason ``SubjectSlopePriorParams.tau1_sigma`` gives."""

    tau1_q_sigma: float
    """HalfNormal scale for the spread of per-year production-ratio rates."""

    ref_age_months: float = 36.0
    """Age at which the two ``tau0`` scales are the between-child spreads. The
    level scales themselves are the definition's own ``tau_subj_u_sigma`` and
    ``tau_subj_q_sigma``, so this form re-uses the parent's priors rather than
    restating them."""

    rho_uq_eta: float = 2.0
    """LKJ concentration for the two level effects' correlation at rank >= 2.

    The first level direction is e_0 and the second starts with
    (rho, sqrt(1 - rho**2)). Thus (rho_uq + 1) / 2 ~ Beta(eta, eta),
    matching VG20's two-dimensional LKJ prior. Matching this prior does not match
    the models' other assumptions.

    At rank one, abs(rho_uq) = 1 and this field is unused.
    See gp_utils.build_child_factor and the prior correction in issue #266.
    """

    def __post_init__(self) -> None:
        if self.rank not in (1, 2, 3):
            raise ValueError(
                "SubjectFactorPriorParams.rank must be 1, 2 or 3: Gate 1 found "
                "rank 3 already reaches the free 4x4's likelihood exactly, so "
                f"rank 4 buys nothing. Got {self.rank!r}."
            )
        for name in ("tau1_u_sigma", "tau1_q_sigma", "rho_uq_eta"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive; got {getattr(self, name)!r}.")


@dataclass(frozen=True)
class AgeVaryingSubjectScale:
    """Proposal A1: an age-varying between-child scale, with ``kappa`` held flat.

    Supplied **in place of** a scalar ``tau_subj_*_sigma`` / ``tau_subject_sigma``.
    That is deliberate: the field it replaces already selects the subject-effect
    scale, so A1 needs no new definition field, and every existing model of record
    keeps its fingerprint (a new field would appear in every serialised definition
    of its class and invalidate every fit of it). The same trick carries
    :data:`CLAMP_Q_ONLY` on ``clamp_mean_above_hi_anchor``.

    The scale is log-linear in standardised age between two reference ages, and
    is parameterised so a constant child scale is **nested at zero**::

        tau_young       ~ HalfNormal(young_sigma)      # the record's own prior
        log_tau_ratio   ~ Normal(0, log_ratio_sigma)   # log(tau_old / tau_young)
        tau(z)          = tau_young * exp(log_tau_ratio * (z - z_young) / (z_old - z_young))

    At the young anchor the prior on the subject scale is *exactly* the prior the
    model of record places on its constant ``tau``, and ``log_tau_ratio = 0``
    gives that constant scale at every age. **It does not give the model of
    record**, because A1 also holds dispersion flat (below): at zero the variant
    is the model of record with ``b_kappa = 0``, which is a different model
    wherever the record's dispersion varies with age -- VG10, VG11, VG12 and
    VG21 among them. So an interval on the ratio that covers zero says the child
    spread need not widen *once dispersion is flat*, not that the model of
    record is adequate; the comparison the variant supports is where the age
    variation belongs. (Stated as exact nesting until 2026-09-13, when #240's
    typically developing arms were registered and the test pinning the held-flat
    dispersion made the gap visible.) A multiplicative ratio also keeps the scale positive
    without taking a logarithm of a ``HalfNormal`` that can approach zero.

    ``hold_kappa_constant`` is part of A1 rather than a separate switch: the
    proposal is that the age variation belongs on the between-child parameter,
    so it is *moved* rather than duplicated. It forces ``b_kappa = 0`` in the
    paired dispersion block — ``kappa_u`` for the understood scale, ``kappa_s``
    for the production one — leaving the dispersion *level* free.

    Structural caveat, stated where it cannot be missed: scaling a single
    per-child deviate by ``tau(age)`` imposes **perfect rank correlation of
    children across age** for that latent component. This is a structural
    restriction. The historical tracking analysis does not validate a numerical
    measurement-error correction or bound; see the September correction notice.
    This remains registered-sensitivity material rather than a model of record. See
    ``notes/202607261540-item-difficulty-and-the-aggregate-likelihood.md`` §9
    and ``notes/202608141600-rank-stability-tracking.md`` §8.
    """

    anchor_ages: tuple[float, float]
    """Reference ages (months) for the young and old ends of the scale. Set these
    to the paired ``kappa`` block's anchors so the two parameters contest the
    same span."""
    young_sigma: float
    """HalfNormal scale at the young anchor. Set it to the scalar
    ``tau_subj_*_sigma`` this object replaces, or the variant is not one-factor."""
    log_ratio_sigma: float
    """Normal scale for ``log(tau_old / tau_young)``. Zero is 'no widening'."""
    hold_kappa_constant: bool = True
    """Force the paired ``kappa`` block flat in age (level still free)."""


def subject_scale_spec(value) -> AgeVaryingSubjectScale | None:
    """Return the A1 spec a subject-scale field carries, or ``None`` if scalar.

    A low-level probe on one overloaded ``tau_subj_*_sigma`` field, and the one
    place that field's A1 reading is interpreted. It is **not** the engines'
    entry point: they ask :func:`vocab_growth.models.subject_effects.resolve`,
    which calls this and its two companions and returns a typed
    ``SubjectEffectPlan`` covering every child-effect structure at once. Use this
    directly only when you have a bare field value and no definition -- as
    ``comparison.py`` does when describing a model from its definition alone.
    """
    return value if isinstance(value, AgeVaryingSubjectScale) else None


def subject_slope_spec(value) -> SubjectSlopePriorParams | None:
    """Return the VG19 child-slope spec a subject-scale field carries, or ``None``.

    The companion to :func:`subject_scale_spec`, on the same overloaded field, and
    a low-level probe in the same sense -- :func:`subject_effects.resolve` is the
    engines' entry point. The two are mutually exclusive by construction: a field
    holds a float, an :class:`AgeVaryingSubjectScale`, or a
    :class:`SubjectSlopePriorParams`, and each selector recognises exactly one of
    them.
    """
    return value if isinstance(value, SubjectSlopePriorParams) else None


def subject_factor_spec(value) -> SubjectFactorPriorParams | None:
    """Return the VG22 low-rank factor spec, or ``None``.

    A low-level probe like its two companions, with
    :func:`subject_effects.resolve` as the engines' entry point. Unlike
    :func:`subject_scale_spec` and :func:`subject_slope_spec` this does
    **not** read an overloaded ``tau_subj_*_sigma`` field. A factor spans both
    outcomes at once, so it is not "the u scale" or "the q scale" and putting it
    in one of them would misname it; it lives on its own subclass field, the
    pattern :class:`BivariateCorrelatedSubjectREModelDefinition` established.
    """
    return value if isinstance(value, SubjectFactorPriorParams) else None


@dataclass(frozen=True)
class SingletonMarginalisationParams:
    """Integrate out the child effects that only one observation ever sees.

    Selects the likelihood in :mod:`~vocab_growth.models.subject_marginal`: a
    child assessed once has its ``delta_subject`` integrated out by
    Gauss-Hermite quadrature, while a child with repeated administrations keeps
    an explicit effect. The marginal likelihood still contains ``tau_subject``,
    so this is **not** the subject-RE removal that
    ``notes/202608050900-td-hierarchical-geometry.md`` §7 rejected: the model,
    ``kappa``'s meaning as within-child dispersion, and the posterior of every
    retained quantity are unchanged up to quadrature error. What changes is the
    sampled space, which loses the thousands of prior-dominated singleton
    dimensions that carry the funnel mass. See
    ``notes/202608231410-td-geometry-remaining-levers.md`` §3.

    It is still a graph change, so it needs a refit; and it changes what a
    marginalised row's pointwise ``log_likelihood`` and posterior predictive
    draw mean -- both become marginal rather than conditional -- so ``elpd``
    values do not compare across the change.
    """

    n_nodes: int
    """Gauss-Hermite nodes for each marginalised row.

    Required rather than defaulted, so a model's accuracy setting is written at
    its registration site. Twenty is what
    :data:`~vocab_growth.models.subject_marginal.DEFAULT_QUADRATURE_NODES`
    recommends, and the value the node-sensitivity check doubles."""

    def __post_init__(self) -> None:
        if not isinstance(self.n_nodes, int) or self.n_nodes < 2:
            raise ValueError(f"n_nodes must be an integer >= 2; got {self.n_nodes!r}.")


# ============================================================
# Univariate model definition
# ============================================================


@dataclass(frozen=True)
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
    """Number of trials on the common reference vocabulary scale."""
    slope_anchors: tuple[float, float]
    """Reference ages (months) for the slope parameterisation."""
    ages_query: tuple[int, ...]
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
    """Fraction of TD **subjects** to subsample (1.0 = no subsampling).

    Whole children are drawn and all their administrations kept
    (:func:`vocab_growth.data_utils._subsample_subjects`). Subsampling rows
    instead destroys the within-child replication that identifies a subject
    random effect; see
    ``notes/202608020829-kappa-and-eta-q-prior-recalibration.md`` §§11-12.
    """
    random_seed: int = 47
    """Random seed for TD subsampling."""

    # -- Shared priors (same across all univariate models) --
    ell_unit_alpha: float = 3.0
    ell_unit_beta: float = 3.0
    eta_sigma: float = 0.4
    ell_months_range: tuple[int, int] = (6, 18)
    n_plot: int = 500
    kappa: KappaPriorParams | KappaAnchorPriorParams = field(
        default_factory=KappaPriorParams
    )
    """Dispersion priors in the intercept-and-slope or two-anchor form."""

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
    tau_subject_sigma: float | AgeVaryingSubjectScale = 1.5
    """HalfNormal scale for the child intercept SD, in logits.

    An AgeVaryingSubjectScale instead selects the registered A1 sensitivity.
    The scale calibration is recorded in
    notes/202608020829-kappa-and-eta-q-prior-recalibration.md.
    """
    one_observation_per_subject: bool = False
    """If True, retain one reproducibly sampled administration per subject. This
    is a clustering sensitivity analysis, not the default estimand."""
    td_languages: tuple[str, ...] = ENGLISH_LANGUAGES
    """Wordbank ``language`` values the typically-developing pool draws on.

    Ignored for the Down syndrome population, whose language scope is fixed when the
    database is built. Defaults to :data:`ENGLISH_LANGUAGES`; the hierarchical
    typically-developing models set :data:`ENGLISH_AND_ROMANCE_LANGUAGES`. Changing
    this changes the data the model sees, so it is part of the model graph and a
    change requires a refit."""

    # -- GP anchor constraint (per-draw zero at reference age) --
    anchor_g_at_ref: bool = False
    """If True, constrain the GP to equal zero at the reference age for every draw."""
    gp_anchor_age_months: float | None = None
    """Reference age (months) for the GP anchor. If None, defaults to the midpoint of
    slope_anchors."""

    # -- Reporting range --
    report_max_age_understood: int | None = None
    """Highest query age (months) at which comprehension quantities are reported.

    Only meaningful on a model whose ``outcome`` is ``UNDERSTOOD``; validation
    rejects it elsewhere rather than letting it be a silent no-op. Trims the
    summary tables to where the comprehension evidence stops. Purely
    post-processing: the query grid, the model graph and the fitted trace are
    untouched, so changing this cannot move the posterior — proved by refitting
    VG10 across the change at a fixed seed and reproducing its diagnostics
    bit-for-bit. It does still require re-running the fit: the summary tables are
    written during the fit pipeline and ``--render-only`` does not regenerate
    them, and this field is part of the recorded definition, so a fit produced
    under a different value is correctly reported as stale. None reports every
    query age. See ``posterior_analysis.trim_reported_ages``."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.UNIVARIATE

    @property
    def outcome_label(self) -> str:
        return f"Words {self.outcome.value}"


@dataclass(frozen=True)
class UnivariateREModelDefinition(UnivariateModelDefinition):
    """Single-outcome random effects and optional sampling parameterisations.

    Subclass fields keep options that the random-effect engine implements off the
    plain univariate definitions. The engine reads absent optional fields with
    their documented defaults. VG17 has a separate exploratory fitting path.
    See fit_identity for the rules governing added definition fields.
    """

    centred_study_re: bool = False
    """Sample study offsets directly instead of scaling unit-scale offsets.

    Both forms give ZeroSumNormal(sigma=tau * sqrt(K/(K-1))). They preserve
    the prior but change the sampler's coordinates. The centred form is used by
    VG11 and VG12; its sampling assessment is in
    notes/202608050900-td-hierarchical-geometry.md.
    """
    subject_variance_partition: SubjectVariancePartitionParams | None = None
    """If set, sample a shared scatter budget and a subject share rather than
    giving ``tau_subject`` and the young ``kappa`` anchor competing priors.

    Requires the two-anchor ``kappa`` form (:class:`KappaAnchorPriorParams`) and
    ``use_subject_re``; validation rejects the other combinations. When set,
    ``tau_subject_sigma`` and the ``kappa`` block's ``excess_young_*`` priors are
    no longer used — the prior moves onto ``total_*`` and ``share_*`` — while
    ``tau_subject`` and ``kappa_excess_young`` remain in the trace as
    deterministics under their usual names.

    Measured on VG12 at ``test``: divergences 59 -> 14, and ``v_total`` samples
    cleanly (energy correlation -0.025). It does **not** fix the energy BFMI
    (0.203 -> 0.192): the ridge does not dissolve, it rotates, with ``share``
    inheriting the whole energy correlation at -0.737. That is the expected result
    if the cause is missing within-child replication rather than bad coordinates,
    which is what §4 of the note argues. Kept for the divergence reduction, not as
    a BFMI remedy. See :class:`SubjectVariancePartitionParams`."""
    sex_effect_sigma: float | None = None
    """Prior SD of the sex coefficient ``beta_sex``, or ``None`` for no sex term.

    The single-outcome counterpart of
    :attr:`BivariateModelDefinition.sex_effect_sigma`, which carries the full
    rationale: a girls ``+1/2`` / boys ``-1/2`` contrast on the outcome logit,
    constant in age, a child of unrecorded sex at contrast zero, and population
    trajectories reported at that midpoint with the girls' and boys' either side.
    On this class rather than :class:`UnivariateModelDefinition` because only the
    random-effect engine implements it. A single-outcome model cannot test the
    expressive-not-receptive shape the Down syndrome literature describes -- only
    the paired models can -- so VG11 and VG12 carry it for the by-sex
    predictions, not for that comparison. ``fit_identity.BACKFILL_DEFAULTS``
    records ``None``."""
    study_age_slope_sigma: float | None = None
    """Prior SD of a per-study age slope, in logits per year, or ``None`` for
    intercept-only study effects.

    A registered sensitivity, not a reporting structure (#240 item 5). Every
    random-effect model gives a study a constant offset from the population
    trajectory, which cannot represent a study whose children's vocabularies rise
    faster or slower than the pool's -- and in the typically developing pool
    studies cover very different age ranges (VG12 rests on two studies above 18
    months and one at 25), so an older-age shape can be partly a study, a
    language or a form rather than development. When set, each study also gets
    ``delta_slope[s] * (age - reference) / 12``: a zero-sum offset scaled by
    ``tau_slope ~ HalfNormal(sigma)``, measured from the GP anchor age (or the
    midpoint of the slope anchors) so the study intercept stays the offset at
    that age. The population trajectories remain at zero study effects, so the
    field changes what they are averaged over rather than how they are read.

    Nested at ``tau_slope = 0``. Added 2026-09-13 with a
    ``fit_identity.BACKFILL_DEFAULTS`` entry of ``None``, read through ``getattr``
    with that default."""


@dataclass(frozen=True)
class UnivariateMarginalisedREModelDefinition(UnivariateREModelDefinition):
    """A univariate RE model whose singleton child effects are integrated out.

    On its own subclass for the reason :class:`UnivariateREModelDefinition`
    gives: a fit is validated by comparing ``dataclasses.asdict`` field for
    field, so a definition class that gains a field invalidates every existing
    fit of that class. VG11 and VG12 are models of record fitted with every
    singleton child effect sampled explicitly; until the VG12 bench says the
    lever earns their refits, the field lives here and they stay instances of
    the parent class.

    The engine reads the field through ``getattr`` with a default, so a plain
    :class:`UnivariateREModelDefinition` still builds.
    """

    singleton_marginalisation: SingletonMarginalisationParams | None = None
    """If set, integrate out the child effects of children seen once.

    Requires ``use_subject_re``; :func:`validate_model_definition` rejects the
    combination without it, where the flag would silently do nothing."""


# ============================================================
# Bivariate model definition
# ============================================================


@dataclass(frozen=True)
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
    """Number of trials on the common reference vocabulary scale."""
    slope_anchors: tuple[float, float]
    """Reference ages (months) for the slope parameterisation."""
    ages_query: tuple[int, ...]
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
    """Fraction of TD **subjects** to subsample (1.0 = no subsampling).

    Whole children are drawn and all their administrations kept
    (:func:`vocab_growth.data_utils._subsample_subjects`). Subsampling rows
    instead destroys the within-child replication that identifies a subject
    random effect; see
    ``notes/202608020829-kappa-and-eta-q-prior-recalibration.md`` §§11-12.
    """
    random_seed: int = 47
    """Random seed for TD subsampling."""

    # -- Shared priors (same across all bivariate models) --
    ell_unit_u_alpha: float = 3.0
    ell_unit_u_beta: float = 3.0
    eta_u_sigma: float = 0.4
    ell_unit_q_alpha: float = 3.0
    ell_unit_q_beta: float = 3.0
    eta_q_sigma: float = 0.8  # Calibration history: notes/202608041730-ds-spoken-q-trajectory-prior.md.
    ell_months_range: tuple[int, int] = (6, 18)
    n_plot: int = 500
    kappa_u: KappaPriorParams | KappaAnchorPriorParams = field(
        default_factory=KappaPriorParams
    )
    kappa_s: KappaPriorParams | KappaAnchorPriorParams = field(
        default_factory=KappaPriorParams
    )

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
    #
    # Both scales are 1.5, calibrated; see `UnivariateModelDefinition
    # .tau_subject_sigma` for the evidence and why the study scales stay at 0.5.
    use_subject_re_u: bool = False
    """If True, add subject-level random intercepts on the understood trajectory."""
    tau_subj_u_sigma: float | AgeVaryingSubjectScale | SubjectSlopePriorParams = 1.5
    """HalfNormal scale for subject intercept SD on understood (logit scale).
    Estimated at 0.85 on the Down syndrome joint frame and 0.74-0.77 on the
    typically-developing ones.

    **This field is overloaded**, and the annotation says so since issue #273.
    A float is the constant between-child scale; an
    :class:`AgeVaryingSubjectScale` selects Proposal A1's age-varying scaling;
    a :class:`SubjectSlopePriorParams` selects VG19's child intercept-and-rate
    block. The overload exists because a fit is validated by comparing the
    serialised definition field for field, so a *new* field on this class would
    invalidate every existing fit of it -- six bivariate models of record. Which
    shape a definition carries is resolved once, by
    :func:`vocab_growth.models.subject_effects.resolve`, rather than tested
    inline wherever it is read.
    """
    use_subject_re_q: bool = False
    """If True, add subject-level random intercepts on the production ratio q."""
    tau_subj_q_sigma: float | AgeVaryingSubjectScale | SubjectSlopePriorParams = 1.5
    """HalfNormal scale for subject intercept SD on q (logit scale). Estimated at
    1.15 on the Down syndrome joint frame and 1.12 on VG13's.

    Overloaded exactly as ``tau_subj_u_sigma`` is; see there."""
    one_observation_per_subject: bool = False
    """If True, retain one reproducibly sampled administration per subject. This
    provides a cheap sensitivity analysis for repeated-measures dependence."""

    # -- Spoken rows with no usable understood count (issues #233, #236) --
    spoken_fallback: str = SPOKEN_FALLBACK_PRODUCT
    """How spoken rows that cannot condition on an observed understood count are
    modelled.

    The paired model is ``U ~ BB(810, p_U, kappa_U)`` then
    ``S | U ~ BB(U, q, kappa_S)``. 455 of the current frame's 1,428 spoken
    observations cannot take the second line -- 444 have no understood count and
    11 record ``spoken > understood`` -- and have always been given
    ``S ~ BB(810, p_U*q, kappa_S)`` instead, which is mean-correct but is not the
    paired model's marginal: it misses the variance, and by a signed amount that
    depends on the fitted concentrations (see
    :data:`~vocab_growth.models.likelihood_utils.SPOKEN_FALLBACK_PRODUCT`). Those
    rows are older and concentrated by study, so the approximation is not
    ignorable.

    One of :data:`~vocab_growth.models.likelihood_utils.SPOKEN_FALLBACK_TREATMENTS`,
    documented individually there. Part of the model graph: changing it requires
    a refit."""
    spoken_fallback_kappa_sigma: float = 0.5
    """Normal SD for the fallback branch's log concentration offset.

    Read only under ``spoken_fallback="separate_dispersion"``. 0.5 on the log
    scale puts an 89% prior interval of roughly [0.45, 2.2] on the multiplier,
    which spans the range a branch-specific dispersion could plausibly want
    without letting 455 rows drive it to a boundary."""

    td_languages: tuple[str, ...] = ENGLISH_LANGUAGES
    """Wordbank ``language`` values the typically-developing pool draws on.

    Ignored for the Down syndrome population, whose language scope is fixed when the
    database is built. Defaults to :data:`ENGLISH_LANGUAGES`; the hierarchical
    typically-developing models set :data:`ENGLISH_AND_ROMANCE_LANGUAGES`. Changing
    this changes the data the model sees, so it is part of the model graph and a
    change requires a refit."""

    # -- Within-child cross-lag (VG16, issue #113) --
    use_cross_lag: bool = False
    """If True, add a cross-lag: the child's prior-wave understood residual
    predicts their current production ratio q (earlier receptive -> later
    expressive). The lag source is assigned per complete (subject, age)
    administration wave (issue #242). Uses the subject understood intercept,
    so requires use_subject_re_u=True for the 'within' baseline."""
    lag_baseline: str = "within"
    """Reference level subtracted from earlier comprehension on the logit scale.

    within also subtracts the child's fitted comprehension intercept.
    population subtracts population and study terms only, so it can mix
    between-child and within-child associations.
    """
    beta_lag_mu: float = 0.0
    """Normal mean for the cross-lag coefficient beta_lag (0 = no direction imposed)."""
    beta_lag_sigma: float = 0.5
    """Normal SD for beta_lag (logit scale, weakly-informative)."""
    lag_max_gap_months: float | None = None
    """Drop a lag whose source wave is more than this many months earlier.

    ``None`` (the default) imposes no ceiling, which is the historical
    behaviour and what every fit before 2026-08-25 carries. VG16 assumes
    ``beta_lag`` is constant across the gaps it actually sees, and on the
    current frame those run 1 to 28 months (median 6): 41 of 477 lagged rows
    sit above 12 months, 9 above 18 and 4 above 24. A prospective association
    measured over two years is a different quantity from one measured over six
    months, and nothing in the model says so — which is the assumption
    [#242](https://github.com/dseinternational/vocabulary-growth/issues/242)
    asks to be checked rather than asserted."""
    lag_zero_handling: str = LAG_ZERO_CLIP
    """How a zero-count lag source is kept off the logit boundary.

    ``LAG_ZERO_CLIP`` (the default) reproduces the historical clip exactly.
    See the constants in ``likelihood_utils`` for what the alternative changes
    and why seven rows on the current frame make it a live question."""
    lag_same_form_only: bool = False
    """Drop a lag whose source wave was scored against a different checklist form.

    ``False`` (the default) is the historical behaviour: the lag predictor is
    ``logit(understood / n_trials)`` whatever form produced the count, which is
    the project's difficulty-ordering harmonisation acting directly on a
    *regressor* rather than on an outcome. A source scored on a 396-item form
    enters already deflated relative to one scored on 810, and a study intercept
    cannot absorb a form transition that happens *within* a study.

    On the 2026-09-06 frame 131 of VG16's 473 supporting rows (28%) cross a form
    ceiling between source and target. Setting this keeps 342 rows from 226
    children in all eight contributing studies -- against 80 rows from 74
    children in two studies under ``dse_native_only``, which is the other
    form-restricted check and cannot separate a scale effect from a change of
    study composition. Like ``lag_max_gap_months`` it drops the lag, not the
    row: the observation still enters both likelihoods. Counted in
    ``notes/202609071000-vg16-available-case-audit.md``
    ([#242](https://github.com/dseinternational/vocabulary-growth/issues/242))."""

    # -- GP anchor constraint (per-draw zero at reference age) --
    anchor_g_u_at_ref: bool = False
    """If True, constrain g_u to equal zero at the reference age for every draw."""
    anchor_g_q_at_ref: bool = False
    """If True, constrain g_q to equal zero at the reference age for every draw."""
    gp_anchor_age_months: float | None = None
    """Reference age (months) for the GP anchor constraint. If None, defaults to the
    midpoint of slope_anchors."""

    # -- Mean extrapolation above the high anchor --
    clamp_mean_above_hi_anchor: bool | str = False
    """Softly level the mean above the high slope anchor.

    True clamps understood and q. CLAMP_Q_ONLY clamps q alone. Resolve the value
    through clamp_targets because the sentinel string is truthy. Below the low
    anchor the line still extrapolates; the GP can still change the combined curve.
    """

    # -- Reporting range --
    report_max_age_understood: int | None = None
    """Reporting cap in months for comprehension and quantities that depend on it.

    None keeps the query grid. This changes reporting, not the likelihood, but it
    remains part of the fit definition. render-only does not regenerate fit-stage
    summary tables.
    """

    # -- Data age filtering --
    max_age_months: int | None = None
    """Upper bound on age (inclusive, months) for data loading. None = no limit."""
    exclude_us01_spoken_ceiling: bool = False
    """Exclude us_01 WS spoken counts at the 680-word ceiling.

    Retained for reversibility, and functional on a reinstated frame. It is no
    longer a *sensitivity* in its own right: those rows are masked by default, so
    on the primary frame this flag has nothing left to exclude. Use
    ``include_implausible_production`` below to interrogate that exclusion."""
    dse_native_only: bool = False
    """Restrict the pool to administrations recorded natively on the 810 reference.

    The models score every count against ``n_trials = 810``, so a 416-item Oxford
    CDI count enters on the same denominator as an 810-item DSE Checklists count.
    That harmonisation assumes the shorter form's items are the easier ones, and
    aggregate totals can still depend on item difficulties. Sufficiency for
    ability treats item difficulties as fixed; it does not prove that totals
    contain no information about them. Without linked items or respondents,
    form composition and ability distributions are hard to separate. This flag
    checks sensitivity by retaining only native forms (issue #190).

    It is the widest-scoped sensitivity in the registry, and deliberately so: on
    2026-09-15 the loader keeps 166 of 1,799 Down syndrome rows, from 129
    children across ie_01 (its 810 wave only), uk_02 (DSE form only) and uk_06,
    and VG10's prepared frame 153 of 1,707. ie_02 left the native set that day,
    when its Checklists 1 + 2 administrations were given their own 476-word
    ceiling (``data_utils.DSE_SHORT_FORM_CEILINGS``). The figures below are from
    before the ``us_03`` ingestion and before that change. Comprehension is the least affected outcome -- 252 of
    977 understood observations survive, against 264 of 1,428 spoken -- because
    the short forms are production-heavy, so expect the spoken trajectory to move
    more than the understood one.
    """
    exclude_studies: tuple[str, ...] = ()
    """Study codes to drop before fitting, for leave-one-study-out sensitivity.

    Empty (the default) admits every study the other rules keep, which is the
    historical behaviour. This exists because a pooled estimate can rest on one
    source without saying so: on the current frame ``us_01`` alone supplies 136
    of VG16's 477 lagged rows and ``it_01`` a further 106, so a cross-lag that
    does not survive dropping either is a statement about that study rather
    than about children with Down syndrome.

    Applied after the source-admissibility rules and before
    ``min_study_observations``, so a study removed here cannot change whether a
    *different* study clears the observation floor — the floor is counted per
    study, so the two are independent, and the order is fixed for
    reproducibility rather than because it changes an answer."""
    include_implausible_production: bool = False
    """Reinstate the us_01 production counts masked as implausible by default.

    The inverse sensitivity to the retired ``us01-ceiling-excluded`` variants.
    ``data_utils.mask_implausible_production_administrations`` excludes 30
    administrations matching a near-ceiling or longitudinal-collapse signature; the
    source author no longer holds the original files, so that exclusion can never
    be confirmed at source, and this flag is the only way to show what the reported
    trajectories would have been had the judgement been wrong. See
    ``notes/202607261245-edgin-duplicated-outcome-records.md``."""
    include_same_day_disagreements: bool = False
    """Reinstate the us_01 production counts masked as same-day contradictions.

    ``data_utils.mask_same_day_production_disagreements`` masks a production
    count contradicted by a same-day administration on another form (#275). Its
    own catch on the default pool is two Words & Sentences counts -- 385 and 406
    words at 23 months against same-day counts of 11 and 50 -- but it also
    re-masks six of the eleven counts ``include_implausible_production`` puts
    back, because those reinstated ceiling-region records have an observed
    same-day partner. So ``us01-implausible-reinstated`` alone reinstates five
    counts, not eleven, and cannot answer its registered question: what the
    trajectories would have been had the implausible judgement been wrong.
    ``us01-masked-production-reinstated`` sets both flags and can (#289 task
    4.3). Read ``data_utils.SAME_DAY_DISAGREEMENT_FACTOR`` before reinstating
    on its own.

    Added 2026-09-05 with a ``fit_identity.BACKFILL_DEFAULTS`` entry: every fit
    made before the field existed called the loader without the argument, whose
    default is ``False``, so a manifest that lacks the field records a fit with
    it set to ``False``. That claim is pinned in ``tests/test_fit_identity.py``
    against the loader's own signature."""
    mask_dse_short_form_comprehension: bool = False
    """Mask the comprehension counts of the DSE short forms kept on the 810 scale.

    ``ie_02`` administered DSE Checklists 1 and 2 only, and since 2026-09-15 its
    counts enter the pool as a short form with a 476-word ceiling rather than
    being masked as a partial administration
    (``data_utils.DSE_SHORT_FORM_CEILINGS``). Checklist 3's harder words matter
    for comprehension above about 300 words, so this flag masks those studies'
    ``understood`` counts and keeps everything else, for the
    ``ie02-comprehension-masked`` sensitivity arm.

    Added with a ``fit_identity.BACKFILL_DEFAULTS`` entry, on the same claim as
    ``include_same_day_disagreements``: every fit made before the field existed
    called the loader without the argument, whose default is ``False``. Pinned
    in ``tests/test_fit_identity.py`` against the loader's own signature."""

    # -- Sex as a covariate (issue #324) --
    sex_effect_sigma: float | None = None
    """Prior SD of the sex coefficients, or ``None`` for no sex term.

    When set, the engine adds ``beta_sex_u`` and ``beta_sex_q``, each
    ``Normal(0, sigma)``, multiplying a girls ``+1/2`` / boys ``-1/2`` contrast on
    the understood and production-ratio logits. Each coefficient is the
    girl-minus-boy difference in logits, and the effect is constant in age by
    design: a fixed logit shift already opens up in words and in months along a
    rising curve, which is what the literature's "growing gap" describes, and
    ``notes/202609041206-sex-differences-in-vocabulary.md`` found no age-by-sex
    interaction on the logit scale in either population.

    **A child whose sex is not recorded takes contrast zero** -- the logit
    midpoint between girls and boys -- and contributes to the trajectory as
    before, so no row is dropped. That is licensed by where the missingness
    sits: in the Down syndrome pool it is exactly study-level (eight studies
    record sex for every row, seven for none), and every model carrying this
    field also carries study intercepts, which absorb the seven studies' sex
    mix. The coefficients are therefore identified from the studies that record
    sex, on the assumption that the effect is common across studies. Of the
    typically developing frames only VG11's has unrecorded sex, measured
    2026-09-13: 778 of 14,553 children (1,449 of 18,500 rows), 757 of them
    ``Smith``'s, which records none, and 21 missing within ``Hoff`` (17 of 257)
    and ``Kalashnikova`` (4 of 1,497). Those 21 sit at the midpoint beside coded
    children of their own study, which is sound only if whether sex was recorded
    is unrelated to sex. The model pages state each fit's own coverage.

    The population trajectories these models report are evaluated at contrast
    zero, so they describe the sex-balanced midpoint in both populations rather
    than each pool's own sex mix; the girls' and boys' trajectories sit half a
    coefficient either side. The engines write both.

    Added 2026-09-13 with a ``fit_identity.BACKFILL_DEFAULTS`` entry of ``None``:
    every engine reads it through ``getattr(definition, "sex_effect_sigma",
    None)``, and until this date the only definition that set it was the
    unregistered VG20 sex-shift experiment, so no registered fit carries a sex
    term. The prior the reporting models set, 0.5, is the experiment's: the
    descriptive estimates are 0.2 to 0.35 logits, and the typically developing
    CDI norms put the whole effect well inside one such SD."""
    sex_known_only: bool = False
    """Keep only administrations whose child has a recorded sex.

    A **data** change and a sensitivity control, not the reporting design: on
    the Down syndrome pool it removes the seven studies that record no sex (711
    of VG20's 1,707 rows, ``us_03`` among them). It is what makes a sex-known
    control arm comparable with its effect arm, since both see the same rows,
    and it refuses a study that records sex for only part of its rows rather
    than turning silently into a row-level filter. Down syndrome pool only.

    Added 2026-09-13 with a ``fit_identity.BACKFILL_DEFAULTS`` entry of
    ``False``, read through ``getattr`` with that default."""
    study_age_slope_sigma: float | None = None
    """Prior SD of a per-study age slope on each outcome, in logits per year, or
    ``None`` for intercept-only study effects.

    The paired counterpart of
    :attr:`UnivariateREModelDefinition.study_age_slope_sigma`, which carries the
    rationale (#240 item 5). Here each study gets a slope on the understood
    trajectory and one on the production ratio, ``delta_u_slope`` and
    ``delta_q_slope``, with their own scales ``tau_u_slope`` and ``tau_q_slope``,
    both ``HalfNormal(sigma)``. Implemented by the bivariate random-effect engine
    only. ``fit_identity.BACKFILL_DEFAULTS`` records ``None``."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.BIVARIATE


@dataclass(frozen=True)
class BivariateCorrelatedSubjectREModelDefinition(BivariateModelDefinition):
    """Bivariate definition that also correlates the two subject random effects.

    VG20 (issue #224). The field lives on a **subclass**, not on
    ``BivariateModelDefinition``, for the reason ``UnivariateREModelDefinition``
    exists: a fit is validated by comparing the serialised definition field for
    field, so adding a field to a definition class invalidates every existing fit
    of that class — which for ``BivariateModelDefinition`` means **every instance
    of its whole tree, subclasses included**: today thirteen models, only eight of
    them direct instances. The subclasses added since this was written (VG19's
    slope, VG20's and VG23's correlation, VG22's factor) inherit the parent's fields
    and are invalidated by a change to them just the same. Both sizes are asserted
    in ``tests/test_ds_joint_shared_priors.py`` rather than restated here.

    ``None`` means "behave exactly as the parent class", so the subclass is inert
    until a definition sets the field; the engine reads it through ``getattr``.
    """

    subject_re_correlation_eta: float | None = None
    """LKJ concentration for the correlation between the two subject intercepts.

    ``None`` disables the correlation, leaving the two blocks independent as in
    VG10. When set, ``(rho_uq + 1) / 2 ~ Beta(eta, eta)``, which for a 2x2 matrix
    is exactly LKJ(eta) — so this is the standard prior, written in the one form
    that keeps ``rho_uq`` a named free variable the summaries and the recovery
    scorer can read, rather than an element of a packed Cholesky vector.

    ``eta = 1`` is uniform on (-1, 1); ``eta = 2`` puts a gentle bias toward zero
    (SD 0.45) so that a correlation has to be evidenced rather than assumed,
    which is the point of the model. The nesting is exact: at ``rho_uq = 0`` the
    graph emits what VG10's does.
    """


@dataclass(frozen=True)
class BivariateChildSlopeModelDefinition(BivariateModelDefinition):
    """Bivariate definition that gives each child a rate as well as an offset.

    VG19 (``notes/202608141900-child-slope-implementation-plan.md``). VG08-VG10
    and VG20 give each child a **constant** offset from the population
    trajectory, so three distinct quantities have to live in two parameters:
    persistent between-child differences, occasion-to-occasion movement, and
    drift. This separates the third.

    Inherits from ``BivariateModelDefinition``, **not** from
    ``BivariateCorrelatedSubjectREModelDefinition``, and that is the decision
    recorded on 2026-08-21: VG19 is gated against VG10, so VG10 is its parent and
    ``subject_re_correlation_eta`` is a field it should not be able to set.

    The two structures are not composable as written. VG20 estimates one
    correlation, between the two outcomes' constant offsets; VG19 estimates two
    different ones, between each outcome's own intercept and slope. The union is
    a 4x4 covariance with six correlations, of which the most interesting -- do
    children who gain comprehension faster also convert faster? -- is estimated
    by neither. Carrying ``rho_uq`` across while leaving that at zero would
    repeat, one level up, the assumption VG20 exists to correct. The engine
    refuses the combination as defence in depth; this base class is why it should
    never arise. See ``notes/202608211500-vg19-registration.md``.

    The slope field is supplied through the ``tau_subj_*_sigma`` seam, so setting
    neither reproduces the parent exactly.

    ``subject_slope_ref_age_months`` is the age at which ``tau0`` *is* the
    between-child spread. Centring at the design's centre is what keeps the
    intercept and the slope from trading off in the sampler, and it makes
    ``tau0`` a spread with a stated age attached rather than an extrapolated
    intercept at age zero.
    """

    subject_slope_ref_age_months: float = 36.0
    """Reference age (months) at which ``tau0`` is the between-child spread.
    36 months is the Down syndrome pool's median age."""


@dataclass(frozen=True)
class BivariateFactorSubjectREModelDefinition(BivariateModelDefinition):
    """Bivariate definition with shared factors for child levels and rates.

    The subclass keeps the factor option off other bivariate definitions.
    None retains independent constant offsets. The factor replaces the two
    outcome blocks and cannot be combined with their separate slope or correlation
    options. See SubjectFactorPriorParams for rank and nesting limits.
    """

    subject_factor: SubjectFactorPriorParams | None = None
    """Low-rank factor structure over ``(b0u, b1u, b0q, b1q)``, or ``None`` for
    the parent's two independent constant offsets.

    Requires ``use_subject_re_u`` and ``use_subject_re_q`` both set: the form is
    a joint covariance over both outcomes' effects and is not defined when only
    one outcome carries a child effect. The engine refuses the combination.
    """


# ============================================================
# Trivariate model definition
# ============================================================


@dataclass(frozen=True)
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
    """Number of trials on the common reference vocabulary scale."""
    slope_anchors: tuple[float, float]
    """Reference ages (months) for the slope parameterisation."""
    ages_query: tuple[int, ...]
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

    # Three independent signing heights define a piecewise logit-linear mean.
    # The priors favour a middle-age rise but do not require it in every draw.
    # A mental-to-chronological-age conversion informed the middle-age choice;
    # it is an assumption, not a measured peak for this sample. The literature
    # rationale is in docs/models/PRIORS.md. The Zampini cohort overlaps it_01,
    # so it cannot provide independent validation of the fitted trajectory.
    sign_anchor_ages: tuple[float, float, float] = (15.0, 36.0, 96.0)
    """Reference ages for the three signed-ratio trend anchors, in months."""
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
    eta_q_sigma: float = 0.8  # Calibration history: notes/202608041730-ds-spoken-q-trajectory-prior.md.
    # The signed GP uses a shorter-scale prior than the other trajectories.
    ell_unit_sign_alpha: float = 2.0
    ell_unit_sign_beta: float = 5.0
    eta_sign_sigma: float = 0.4
    """HalfNormal scale for smooth GP departures from the three-anchor signed trend."""
    ell_months_range: tuple[int, int] = (6, 18)
    n_plot: int = 500
    # -- Child-outcome rows with no usable understood count (issues #266, #240) --
    spoken_fallback: str = SPOKEN_FALLBACK_PRODUCT
    """How child-outcome rows that cannot condition on an observed understood
    count are modelled. One of
    :data:`~vocab_growth.models.likelihood_utils.SPOKEN_FALLBACK_TREATMENTS`,
    documented individually there, and applied to the **signed** rows as well as
    the spoken ones on this engine.

    Exposed here by issue #266 finding 8, which is explicit that the
    approximation is a methodological exposure rather than a detail: the default
    gives such a row ``BB(810, p_U*q, kappa)``, which is mean-correct but is not
    the marginal implied by the paired model, and the affected rows are older and
    clustered by study. The bivariate engines have carried the choice since #240;
    this engine hard-coded the default, so no sensitivity could be run at all.

    Part of the model graph: changing it requires a refit. Adding the field does
    **not** invalidate existing fits -- ``resolve_fallback_treatment`` returned
    this same default for every fit made before it existed, and
    :data:`~vocab_growth.models.fit_identity.BACKFILL_DEFAULTS` records that."""
    spoken_fallback_kappa_sigma: float = 0.5
    """Normal SD for the fallback branch's log concentration offset. Read only
    under ``spoken_fallback="separate_dispersion"``; see the bivariate
    definition's field of the same name for the calibration."""

    kappa_u: KappaPriorParams | KappaAnchorPriorParams = field(default_factory=KappaPriorParams)
    kappa_s: KappaPriorParams | KappaAnchorPriorParams = field(default_factory=KappaPriorParams)
    kappa_sign: KappaPriorParams | KappaAnchorPriorParams = field(default_factory=KappaPriorParams)

    # -- Signed data inclusion --
    include_uk01_signed: bool = False
    """Re-include uk_01's signed-only count as if it were total sign use.

    False by default because uk_01 excludes words that are also spoken, whereas
    the model estimand and the other sources use total signed vocabulary.  This
    switch exists only for a source-sensitivity comparison.
    """

    # -- Data age filtering --
    max_age_months: int | None = None
    """Upper bound on age (inclusive, months) for data loading. None = no limit."""

    # -- Mean extrapolation above the high anchor --
    clamp_mean_above_hi_anchor: bool | str = False
    """Softly level the mean above the high slope anchor.

    True clamps understood and q. CLAMP_Q_ONLY clamps q alone. Resolve the value
    through clamp_targets because the sentinel string is truthy. Below the low
    anchor the line still extrapolates; the GP can still change the combined curve.
    """

    # -- Reporting range --
    report_max_age_understood: int | None = None
    """Reporting cap in months for comprehension and quantities that depend on it.

    None keeps the query grid. This changes reporting, not the likelihood, but it
    remains part of the fit definition. render-only does not regenerate fit-stage
    summary tables.
    """

    report_max_age_signed: int | None = None
    """Reporting cap in months for signed counts.

    Ratios and union quantities also use the comprehension cap through reporting_ages.
    This is reporting policy, not a data exclusion or a change to the likelihood.
    """

    @property
    def model_type(self) -> ModelType:
        return ModelType.TRIVARIATE


# ============================================================
# Joint sign/speech model definition (VG15, issue #49 Option 3)
# ============================================================


@dataclass(frozen=True)
class JointModelDefinition:
    """Joint understood, spoken and signed vocabulary with sign-speech overlap.

    A Plackett odds ratio psi links spoken and signed shares within understood
    words. Four-cell sources and the produced-only source have different
    conditioning sets. Each contributing study has its own association offset.

    Optional study and child effects and GP anchors are selected by the definition.
    Child effects enter marginal counts, not cell likelihoods. See composition
    for those likelihoods and docs/models/PRIORS.md for prior assumptions.
    """

    model_id: str
    config_name: str
    banner: str
    population: Population
    n_trials: int
    slope_anchors: tuple[float, float]
    ages_query: tuple[int, ...]
    gp_domain_months: tuple[float, float] | None = None
    """Fixed HSGP age domain. ``None`` uses the observed age range; reporting
    query ages never determine the approximation domain."""

    # Understood priors match the DS joint family. See
    # notes/202608041216-ds-understood-trajectory-prior.md.
    p_slope_low_u_alpha: float = 1.5
    p_slope_low_u_beta: float = 8.0
    p_slope_hi_u_alpha: float = 3.0
    p_slope_hi_u_beta: float = 1.3

    # -- Speak-given-understood (q) slope priors (bivariate defaults) --
    p_slope_low_q_alpha: float = 1.0
    p_slope_low_q_beta: float = 1.5
    p_slope_hi_q_alpha: float = 2.0
    p_slope_hi_q_beta: float = 1.2

    # Signing uses three logit-linear anchors plus a GP. The prior favours
    # a middle-age rise but does not constrain every curve to a hump.
    # Source rationale is shared with TrivariateModelDefinition.
    sign_anchor_ages: tuple[float, float, float] = (15.0, 36.0, 96.0)
    """Outer ages and middle knot of the signed-ratio trend, in months.

    sign_peak_prior can move the middle age between the fixed outer ages. Independent
    height priors do not require the middle height to exceed the outer ones.
    """
    sign_peak_prior: tuple[float, float] | None = None
    """Beta prior on the middle signing anchor's position between the outer ages.

    None fixes the middle age at sign_anchor_ages[1]. Otherwise the position is
    sampled and converted to an age between the fixed outer anchors. The name
    refers to the trend's middle anchor, not necessarily the maximum of the
    combined trend and GP curve. Its posterior uncertainty does not by itself
    establish identification. See notes/202608060900-three-prior-conflicts.md.
    """
    p_slope_low_sign_alpha: float = 2.0
    p_slope_low_sign_beta: float = 20.0
    """Beta prior parameters for the young signed-ratio anchor."""
    p_slope_mid_sign_alpha: float = 3.0
    p_slope_mid_sign_beta: float = 4.0
    """Beta prior parameters for the middle signed-ratio anchor."""
    p_slope_hi_sign_alpha: float = 2.0
    p_slope_hi_sign_beta: float = 16.0
    """Beta prior parameters for the old signed-ratio anchor."""

    # -- Shared GP / amplitude priors (sign GP looser + shorter, per VG14) --
    ell_unit_u_alpha: float = 3.0
    ell_unit_u_beta: float = 3.0
    eta_u_sigma: float = 0.6  # aligned with the recalibrated VG02 understood trajectory
    ell_unit_q_alpha: float = 3.0
    ell_unit_q_beta: float = 3.0
    eta_q_sigma: float = 0.8  # Calibration history: notes/202608041730-ds-spoken-q-trajectory-prior.md.
    # Retain a sampled signing length scale. Dated fixed-length and no-GP
    # comparisons are in notes/202608060900-three-prior-conflicts.md, section 5b.
    # Their results do not establish identification or future fit quality.
    ell_unit_sign_alpha: float = 2.0
    ell_unit_sign_beta: float = 5.0
    eta_sign_sigma: float = 0.4
    ell_months_range: tuple[int, int] = (6, 18)
    n_plot: int = 500
    # -- Child-outcome rows with no usable understood count (issues #266, #240) --
    spoken_fallback: str = SPOKEN_FALLBACK_PRODUCT
    """Missing-parent treatment for both spoken and signed likelihoods.

    See likelihood_utils.SPOKEN_FALLBACK_TREATMENTS. The product-mean default generally
    does not match the nested model's full marginal distribution. Historical defaults
    are recorded in fit_identity.BACKFILL_DEFAULTS.
    """
    spoken_fallback_kappa_sigma: float = 0.5
    """Normal SD for the fallback branch's log concentration offset. Read only
    under ``spoken_fallback="separate_dispersion"``; see the bivariate
    definition's field of the same name for the calibration."""

    kappa_u: KappaPriorParams | KappaAnchorPriorParams = field(default_factory=KappaPriorParams)
    kappa_s: KappaPriorParams | KappaAnchorPriorParams = field(default_factory=KappaPriorParams)
    # Signing retains the legacy non-increasing concentration. The dated
    # comparison is in notes/202608060900-three-prior-conflicts.md, section 5b.
    kappa_sign: KappaPriorParams | KappaAnchorPriorParams = field(default_factory=KappaPriorParams)

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
    tau_psi_sigma: float = 1.0
    """HalfNormal scale for between-study variation in log association.

    Only studies with cell counts inform it. The wider prior reflects the dated
    audit in notes/202608121030-psi-heterogeneity-and-age-invariance.md. That audit
    does not establish age invariance or measurement equivalence. Report study
    associations and uncertainty alongside the centre.
    """

    # -- Subject-level random intercepts (VG08-VG10 pattern, issue #59) --
    #
    # All three scales are 1.5, calibrated for understood and q and widened to
    # match for the signed ratio; see `UnivariateModelDefinition.tau_subject_sigma`.
    use_subject_re_u: bool = False
    """If True, add subject-level random intercepts on the understood trajectory."""
    tau_subj_u_sigma: float = 1.5
    """HalfNormal scale for the subject intercept SD on understood (logit scale)."""
    use_subject_re_q: bool = False
    """If True, add subject-level random intercepts on the speak ratio q."""
    tau_subj_q_sigma: float = 1.5
    """HalfNormal scale for the subject intercept SD on q (logit scale)."""
    use_subject_re_sign: bool = False
    """Whether marginal signed counts include a persistent child intercept."""
    tau_subj_sign_sigma: float = 1.5
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

    # -- Mean extrapolation above the high anchor --
    clamp_mean_above_hi_anchor: bool | str = False
    """Softly level the mean above the high slope anchor.

    True clamps understood and q. CLAMP_Q_ONLY clamps q alone. Resolve the value
    through clamp_targets because the sentinel string is truthy. Below the low
    anchor the line still extrapolates; the GP can still change the combined curve.
    """

    # -- Reporting range --
    report_max_age_understood: int | None = None
    """Reporting cap in months for comprehension and quantities that depend on it.

    None keeps the query grid. This changes reporting, not the likelihood, but it
    remains part of the fit definition. render-only does not regenerate fit-stage
    summary tables.
    """

    # -- Signed data inclusion (inherits VG14's decision) --
    include_uk01_signed: bool = False
    """Re-include uk_01's signed-only field for a source-sensitivity fit."""
    exclude_us01_spoken_ceiling: bool = False
    """Exclude us_01 WS spoken counts at the 680-word ceiling.

    Retained for reversibility, and functional on a reinstated frame. It is no
    longer a *sensitivity* in its own right: those rows are masked by default, so
    on the primary frame this flag has nothing left to exclude. Use
    ``include_implausible_production`` below to interrogate that exclusion."""
    dse_native_only: bool = False
    """Restrict observations to forms recorded natively on the 810-item inventory.

    Short-form totals otherwise use the common reference scale under the project's
    item-difficulty assumption. This sensitivity changes the study and age mix as
    well as cell evidence. It cannot isolate inventory-size effects from all other
    source differences. See the data guide and prior guide.
    """
    report_max_age_signed: int | None = None
    """Reporting cap in months for signed counts.

    The signed ratio and union also depend on comprehension, so callers apply the
    tighter relevant cap through reporting_ages. Later observations still inform
    the fit. See notes/202608120030-uk07-pactds-integration-and-ds-refit.md.
    """
    include_implausible_production: bool = False
    """Reinstate the us_01 production counts masked as implausible by default.

    The inverse sensitivity to the retired ``us01-ceiling-excluded`` variants.
    ``data_utils.mask_implausible_production_administrations`` excludes 30
    administrations matching a near-ceiling or longitudinal-collapse signature; the
    source author no longer holds the original files, so that exclusion can never
    be confirmed at source, and this flag is the only way to show what the reported
    trajectories would have been had the judgement been wrong. See
    ``notes/202607261245-edgin-duplicated-outcome-records.md``."""
    include_same_day_disagreements: bool = False
    """Reinstate the us_01 production counts masked as same-day contradictions.

    ``data_utils.mask_same_day_production_disagreements`` masks a production
    count contradicted by a same-day administration on another form (#275). Its
    own catch on the default pool is two Words & Sentences counts -- 385 and 406
    words at 23 months against same-day counts of 11 and 50 -- but it also
    re-masks six of the eleven counts ``include_implausible_production`` puts
    back, because those reinstated ceiling-region records have an observed
    same-day partner. So ``us01-implausible-reinstated`` alone reinstates five
    counts, not eleven, and cannot answer its registered question: what the
    trajectories would have been had the implausible judgement been wrong.
    ``us01-masked-production-reinstated`` sets both flags and can (#289 task
    4.3). Read ``data_utils.SAME_DAY_DISAGREEMENT_FACTOR`` before reinstating
    on its own.

    Added 2026-09-05 with a ``fit_identity.BACKFILL_DEFAULTS`` entry: every fit
    made before the field existed called the loader without the argument, whose
    default is ``False``, so a manifest that lacks the field records a fit with
    it set to ``False``. That claim is pinned in ``tests/test_fit_identity.py``
    against the loader's own signature."""
    mask_dse_short_form_comprehension: bool = False
    """Mask the comprehension counts of the DSE short forms kept on the 810 scale.

    ``ie_02`` administered DSE Checklists 1 and 2 only, and since 2026-09-15 its
    counts enter the pool as a short form with a 476-word ceiling rather than
    being masked as a partial administration
    (``data_utils.DSE_SHORT_FORM_CEILINGS``). Checklist 3's harder words matter
    for comprehension above about 300 words, so this flag masks those studies'
    ``understood`` counts and keeps everything else, for the
    ``ie02-comprehension-masked`` sensitivity arm.

    Added with a ``fit_identity.BACKFILL_DEFAULTS`` entry, on the same claim as
    ``include_same_day_disagreements``: every fit made before the field existed
    called the loader without the argument, whose default is ``False``. Pinned
    in ``tests/test_fit_identity.py`` against the loader's own signature."""
    exclude_studies: tuple[str, ...] = ()
    """Study codes to drop before fitting, for leave-one-study-out sensitivity.

    Empty (the default) admits every study the other rules keep, which is what
    every joint fit before this field did. It is the same field
    ``BivariateModelDefinition`` has carried since 2026-08-25, and it exists here
    for the same reason: a pooled estimate can rest on one source without saying
    so. #297 check 5 asks it of VG25, whose sign -> speech lag draws 52 of its 191
    supporting observations from ``uk_07`` alone.

    Every row of a named study goes, marginal and cross-tabulation alike. The
    joint engine assembles its four cross-tab sources into the same frame as the
    merged view's marginals, so the filter runs on the assembled frame and
    reaches both; filtering the merged view would leave a cross-tab study's cells
    in the composition likelihood while removing its marginals. Study and subject
    codes are assigned afterwards, so no random-effect level is left empty. A
    code that matches no row is refused rather than ignored: a leave-one-study-out
    check that removes nothing cannot fail.

    Added 2026-09-13 with a ``fit_identity.BACKFILL_DEFAULTS`` entry, so a
    manifest that lacks the field records a fit with it empty. That claim is
    pinned in ``tests/test_fit_identity.py`` against the frame builder's own
    ``getattr`` default, and was verified on the VG15 and VG24 fits of record:
    both still validate, and the frame hash all three joint models share is
    unchanged by the field's addition."""

    # -- nz_01 (Foster-Cohen) produced cross-tab inclusion --
    include_nz01_cells: bool = True
    """If True (default), nz_01's produced modality cross-tab (word-only / sign-only
    / both) enters via a within-produced Dirichlet-Multinomial that informs psi/q/r
    (see common_joint_modality). If False, nz_01 is excluded from VG15 entirely
    (its production-only, 675-item marginals are not comparable to the 810-item
    marginal likelihoods); the flag is kept for reversibility and for isolating
    nz_01's pull on psi."""

    # -- uk_07 (PACT-DS) within-understood cross-tab inclusion --
    include_uk07_cells: bool = True
    """If True (default), uk_07's within-understood four-cell cross-tab (neither /
    sign-only / speech-only / both) enters the same Dirichlet-Multinomial that
    identifies psi from uk_02, roughly doubling the rows that identify it and
    extending their age span from 19-56 months out to 95 (see
    common_joint_modality).

    Unlike ``include_nz01_cells``, setting this False does **not** drop the study:
    uk_07's understood/spoken/signed marginals are on an ordinary comprehension-
    plus-production footing, so they fall back into the marginal likelihoods and
    uk_07 keeps informing U, q and r. The flag therefore isolates uk_07's pull on
    the association alone, which is what a sensitivity comparison wants."""

    # -- es_01 (Galeote) within-understood cross-tab inclusion --
    include_es01_cells: bool = True
    """Include es_01's within-understood gesture/speech partition in the cells.

    The source records item-specific symbolic gestures, including taught signs and
    spontaneous gestures. It does not establish equivalence with other sources'
    sign measurements. Study offsets do not resolve that construct question.

    If False, es_01 returns to marginal likelihoods and gestured totals still inform
    the signed trajectory. This flag checks its cell-association contribution,
    not exclusion of gesture data. See data/vocab_data_es_01.md and
    notes/202609021903-es01-gesture-construct-revisited.md.
    """

    # -- Sex as a covariate (issue #324) --
    sex_effect_sigma: float | None = None
    """Normal prior SD for each sex coefficient, or None to omit sex terms.

    Girls use +1/2, boys -1/2 and unrecorded sex zero. Zero is a logit midpoint, not
    an arithmetic probability average. The three coefficients enter marginal and
    cell likelihoods. Cell association is conditional on these sex and study terms;
    other child offsets remain excluded from cells.

    The frame builder joins cross-tab children's recorded sex from the merged view
    by study and child. fit_identity.BACKFILL_DEFAULTS records None.
    """

    @property
    def model_type(self) -> ModelType:
        return ModelType.JOINT


@dataclass(frozen=True)
class JointCorrelatedSubjectREModelDefinition(JointModelDefinition):
    """Joint definition with a correlated three-outcome child block.

    A subclass keeps this option off VG15's definition. None retains the parent's
    independent child effects. See fit_identity for added-field rules.
    """

    subject_re_correlation_eta: float | None = None
    """LKJ concentration for the three child-intercept correlations, or None.

    The block keeps the three HalfNormal scale priors and emits rho_uq, rho_u_sign
    and rho_sign_q. At identity correlation its child distribution matches
    independent intercepts.

    Equal eta does not give equal marginal correlation priors across dimensions.
    For an n by n LKJ(eta) matrix, (rho+1)/2 follows
    Beta(eta+(n-2)/2, eta+(n-2)/2). At eta=2 the correlation SD is about 0.41
    for n=3 and 0.45 for n=2.
    """


@dataclass(frozen=True)
class JointCrossLagModelDefinition(JointCorrelatedSubjectREModelDefinition):
    """Joint correlated child effects with an optional earlier-signing predictor.

    A child's prior-wave signed share, relative to the selected baseline, shifts
    the logit of the current spoken share through beta_sign_lag. The parent
    likelihood is recovered at coefficient zero.

    The correlated child block describes persistent between-child associations.
    Subtracting the fitted child signing level aims to isolate a within-child
    predictor, but it does not guarantee separation from measurement error or
    other persistent differences. The lag remains observational.
    """

    use_sign_cross_lag: bool = False
    """Use the previous signed share of comprehension to predict current q.

    Sources come from strictly earlier recorded-age waves. Selection uses the largest
    comprehension denominator within a source wave. This does not establish
    measurement equivalence across forms.

    nz_01 partitions produced words and lacks a comprehension denominator, so its
    rows supply no lag source. Their likelihood contributions remain unchanged.
    """

    sign_lag_baseline: str = "within"
    """Reference level subtracted from the earlier signed share.

    within subtracts population, study and fitted child signing terms.
    population subtracts population and study terms only. The default is within.
    These baselines define different predictors; child correlations do not by
    themselves remove persistent differences from a population lag.
    """

    sign_lag_in_cells: bool = False
    """Whether the speech lag also enters the cell likelihoods.

    False confines it to spoken marginals. With a within baseline, the predictor
    contains an estimated prior-wave child signing effect, so True carries that
    effect into cells that otherwise exclude child offsets.

    The default follows probes in notes/202609151930-vg25-lag-out-of-the-cells.md.
    The sign-lag-in-cells sensitivity uses the population baseline. Read support
    from each fit's audit rather than assuming the same rows inform both.
    """

    beta_sign_lag_mu: float = 0.0
    """Normal mean for ``beta_sign_lag`` (0 imposes no direction)."""

    beta_sign_lag_sigma: float = 0.5
    """Normal SD for ``beta_sign_lag`` (logit scale, weakly-informative).

    VG16's ``beta_lag_sigma``, deliberately, so the two coefficients are
    prior-comparable -- and paired with the same ``beta-tight`` / ``beta-wide``
    sensitivities, because symmetry is not calibration and an interval that
    excludes zero under one prior scale should be shown to under others."""

    sign_lag_max_gap_months: float | None = None
    """Drop a lag whose source wave is more than this many months earlier.

    ``None`` imposes no ceiling. VG25 assumes one coefficient across every gap it
    sees, and this pool's are at least as wide as VG16's -- which is the
    constancy assumption issue #242 asked to be checked rather than asserted, so
    the ``sign-lag-gap-12`` arm is registered with the field rather than after a
    reviewer asks for it. Like VG16's, it drops the lag, not the row."""

    sign_lag_zero_handling: str = LAG_ZERO_CLIP
    """How a boundary signed share is kept off the logit boundary.

    ``LAG_ZERO_CLIP`` clips ``signed / understood`` to ``[1e-4, 1 - 1e-4]``;
    ``LAG_ZERO_CONTINUITY`` uses ``(signed + 0.5) / (understood + 1)``, which is
    derived from the wave's own denominator rather than from a floor. This
    predictor reaches **both** boundaries where VG16's reaches only the lower
    one: a child who signs every word they understand is a real observation, and
    a clip at ``1 - 1e-4`` puts every such wave at the same +9.21 whatever its
    denominator was. See the constants in ``likelihood_utils``."""

    sign_lag_same_form_only: bool = False
    """Use a lag only when source and target have the same known inventory size.

    The restriction removes the lag, while retaining the observation. It does
    not search further back for another source, which would also change the
    age gap. See the cross-form audit in the September 2026 implementation note.
    """


# ============================================================
# Model instances
# ============================================================

_DS_GP_DOMAIN_MONTHS = (8, 115)
_TD_GP_DOMAIN_MONTHS = (8, 30)
_YOUNG_TD_GP_DOMAIN_MONTHS = (8, 18)


# ------------------------------------------------------------------
# Production-outcome dispersion, two-anchor form (2026-08-02)
# ------------------------------------------------------------------
# Marginal dispersion calibration for models without study or child effects.
# Reference-age excesses avoid a prior whose month-scale meaning changes with
# the frame's age standardisation. These priors use the fitted data.
# Estimator checks, values and limitations are recorded in
# notes/202608020829-kappa-and-eta-q-prior-recalibration.md.
_DS_SPOKEN_KAPPA = KappaAnchorPriorParams(
    # Implied b_kappa_mag: median 2.80, 5-95% [0.91, 4.67], P(kappa rising) 0.007.
    # The empirical slope is 2.78 on 25 age cells and 2.17 on the 12-cell subset
    # used earlier, so the prior brackets both readings; the legacy
    # HalfNormal(0.75) put them at prior CDF 1.00.
    anchor_ages=(18.0, 36.0),
    kappa_min_mu=math.log(3.0),
    kappa_min_sigma=0.8,
    excess_young_mu=math.log(45.0),
    excess_young_sigma=0.7,
    excess_old_mu=math.log(4.0),
    excess_old_sigma=0.7,
)

_TD_SPOKEN_KAPPA = KappaAnchorPriorParams(
    # VG03 only. Implied b_kappa_mag median 1.71 on its frame, 5-95% about
    # [0.50, 2.90], which contains both the 1.78 its own frame fits and the 1.71
    # estimated in section 17. Excess medians split the two TD frames' fitted
    # values (34.0/26.8 at 12 months, 3.07/3.47 at 20 months) from when VG11
    # shared this block; VG11 has since moved to a conditional calibration and
    # the numbers are left as they are, being within a few percent of VG03's own.
    anchor_ages=(12.0, 20.0),
    kappa_min_mu=math.log(3.0),
    kappa_min_sigma=0.8,
    excess_young_mu=math.log(30.0),
    excess_young_sigma=0.7,
    excess_old_mu=math.log(3.0),
    excess_old_sigma=0.7,
)

_DS_UNDERSTOOD_KAPPA = KappaAnchorPriorParams(
    # VG02. Fitted totals 15.4 at 18 months and 7.1 at 36, so comprehension
    # dispersion is roughly a third of spoken's at the same ages (VG01: 48 and 7)
    # and falls more gently. Implied b_kappa_mag median 0.97, 5-95%
    # [-0.67, 2.61], P(kappa rising) 0.166 — the interval reaches across zero
    # because 346 rows over 15 age cells of 15-35 observations each cannot rule
    # out a flat curve, and the freed sign is what lets the prior say so.
    # sigma 0.8 rather than the spoken blocks' 0.7: the per-cell estimates
    # scatter 3.6-16.3 around the anchors on those cell counts.
    anchor_ages=(18.0, 36.0),
    kappa_min_mu=math.log(3.0),
    kappa_min_sigma=0.8,
    excess_young_mu=math.log(11.0),
    excess_young_sigma=0.8,
    excess_old_mu=math.log(3.2),
    excess_old_sigma=0.8,
)

_TD_UNDERSTOOD_KAPPA = KappaAnchorPriorParams(
    # VG04, on its 25% subsample (1,538 rows). Fitted totals 11.8 at 12 months
    # and 11.3 at 18 — flat, which is why the anchors sit only six months apart:
    # there is no decay to span, and placing them where the data are densest
    # (n = 115 and 128) is what matters instead. Implied b_kappa_mag median 0.04,
    # 5-95% [-0.94, 1.00], P(kappa rising) 0.476.
    #
    # Cross-check on the whole marginal/conditional distinction: VG12 fits the
    # same outcome and population with random effects, and its *marginal*
    # estimate is 11.0 at 12 months against this frame's 11.8. Fit VG04's own
    # rows conditionally and they give 42.8, against VG12's 43.0. Two frames, two
    # estimators, the same answer once the specification matches the model.
    anchor_ages=(12.0, 18.0),
    kappa_min_mu=math.log(3.0),
    kappa_min_sigma=0.8,
    excess_young_mu=math.log(7.6),
    excess_young_sigma=0.7,
    excess_old_mu=math.log(7.2),
    excess_old_sigma=0.7,
)


# ------------------------------------------------------------------
# Dispersion for the random-effect models, calibrated conditionally (2026-08-02)
# ------------------------------------------------------------------
# Conditional calibration includes study and child effects because kappa
# describes residual variation after those effects. A marginal calibration
# cannot be substituted for it. Recovery and mean-model checks are recorded in
# notes/202608020829-kappa-and-eta-q-prior-recalibration.md; the registered-frame
# TD recheck is in notes/202609062330-vg11-vg13-calibration-regenerated.md.
# Those checks apply to their tested designs, not to every concentration level.
_TD_SPOKEN_KAPPA_RE = KappaAnchorPriorParams(
    # VG11. Its posterior already found 310 @ 12 mo and 50.0 @ 20 mo against this
    # calibration's 317 and 50.5 — the likelihood was overwhelming the old prior
    # rather than being distorted by it, so this change removes a prior-data
    # conflict rather than moving the fit.
    anchor_ages=(12.0, 20.0),
    kappa_min_mu=math.log(6.0),
    kappa_min_sigma=0.8,
    excess_young_mu=math.log(311.0),
    excess_young_sigma=0.7,
    excess_old_mu=math.log(44.0),
    excess_old_sigma=0.7,
)

_TD_UNDERSTOOD_KAPPA_RE = KappaAnchorPriorParams(
    # VG12 (8-25 months). Rising: 43 @ 12 mo to 66 @ 20 mo. The conditional fit
    # puts no mass on a floor (kappa_min goes to 0 with an unbounded standard
    # error, because a rising curve never reaches one inside the frame), so the
    # floor keeps the weak LogNormal(log 3, 0.8) the other blocks use and the
    # anchors carry the level.
    anchor_ages=(12.0, 20.0),
    kappa_min_mu=math.log(3.0),
    kappa_min_sigma=0.8,
    excess_young_mu=math.log(40.0),
    excess_young_sigma=0.9,
    excess_old_mu=math.log(63.0),
    excess_old_sigma=0.9,
)

_TD_UNDERSTOOD_VARIANCE_PARTITION = SubjectVariancePartitionParams(
    # Variance-budget sensitivity calibrated on the fitted data. See
    # notes/202608050900-td-hierarchical-geometry.md, section 7.
    reference_proportion=0.1041,
    # The budget changes the induced priors on scale and concentration.
    # Check their marginals against the original independent priors.
    total_mu=0.0,
    total_sigma=0.8,
    share_alpha=3.9,
    share_beta=2.1,
)

_TD_SPOKEN_VARIANCE_PARTITION = SubjectVariancePartitionParams(
    # Same budget and share priors as VG12. p0 uses the spoken mean in the
    # 11-13-month calibration band, so it changes the dispersion mapping.
    # Calibration is in sample and does not validate coverage or convergence.
    reference_proportion=0.0118,
    total_mu=0.0,
    total_sigma=0.8,
    share_alpha=3.9,
    share_beta=2.1,
)

_TD_YOUNG_UNDERSTOOD_KAPPA_RE = KappaAnchorPriorParams(
    # VG13's understood outcome (8-18 months). Here the fit *does* identify a
    # floor, at 37, and it matters: a third of the frame sits below the young
    # anchor, where a rising exponential term contributes almost nothing and the
    # floor alone sets the level. The 8-11 month cells give 23-32, consistent
    # with it. Totals: 40 @ 12 mo, 120 @ 17 mo.
    anchor_ages=(12.0, 17.0),
    kappa_min_mu=math.log(30.0),
    kappa_min_sigma=0.6,
    excess_young_mu=math.log(10.0),
    excess_young_sigma=0.9,
    excess_old_mu=math.log(90.0),
    excess_old_sigma=0.9,
)

_TD_YOUNG_Q_KAPPA_RE = KappaAnchorPriorParams(
    # VG13's production ratio, on the nested scale the engine uses: spoken out of
    # that child's own observed understood count, mean q. Falls gently, 36 to 30.
    # VG13's posterior is already at 40.4 and 29.7, so like VG11 this re-centres a
    # prior the data had overruled rather than changing the answer.
    anchor_ages=(12.0, 17.0),
    kappa_min_mu=math.log(3.0),
    kappa_min_sigma=0.8,
    excess_young_mu=math.log(33.0),
    excess_young_sigma=0.7,
    excess_old_mu=math.log(27.0),
    excess_old_sigma=0.7,
)

# VG21's understood and production-ratio kappa. Identical to the VG13 pair above
# except for the anchor ages, moved from (12, 17) to (12, 20) so the high anchor
# sits inside the wider 8-22 month window rather than five months short of its
# top. The magnitudes are deliberately NOT recalibrated -- they are inherited
# from VG13, which inherited them from VG12 -- and
# notes/202608211100-window-22-adopted.md §6 lists that as an outstanding item on
# the promotion rather than something these fits settle.
_TD_WINDOW22_UNDERSTOOD_KAPPA_RE = KappaAnchorPriorParams(
    anchor_ages=(12.0, 20.0),
    kappa_min_mu=math.log(30.0),
    kappa_min_sigma=0.6,
    excess_young_mu=math.log(10.0),
    excess_young_sigma=0.9,
    excess_old_mu=math.log(90.0),
    excess_old_sigma=0.9,
)

_TD_WINDOW22_Q_KAPPA_RE = KappaAnchorPriorParams(
    anchor_ages=(12.0, 20.0),
    kappa_min_mu=math.log(3.0),
    kappa_min_sigma=0.8,
    excess_young_mu=math.log(33.0),
    excess_young_sigma=0.7,
    excess_old_mu=math.log(27.0),
    excess_old_sigma=0.7,
)

# Shared conditional dispersion calibration for the DS joint family.
# VG14 has no study or child effects, so this is a known prior mismatch there;
# its registration records the limitation. Dated estimator checks and bias
# assessments are in notes/202608020829-kappa-and-eta-q-prior-recalibration.md.

_DS_JOINT_UNDERSTOOD_KAPPA_RE = KappaAnchorPriorParams(
# The 18/72-month anchors and floor prior follow the dated calibration in
# notes/202608191800-kappa-components-not-estimands.md. The old anchor lies
# close to the floor, so the two components are weakly separated.
# Historical simulations and fit comparisons apply to their tested settings.
    anchor_ages=(18.0, 72.0),
    kappa_min_mu=math.log(7.8),
    kappa_min_sigma=0.8,
    excess_young_mu=math.log(84.8),
    excess_young_sigma=1.0,
    excess_old_mu=math.log(6.2),
    excess_old_sigma=1.0,
)

_DS_JOINT_Q_KAPPA_RE = KappaAnchorPriorParams(
    # Calibrated on spoken counts conditional on observed comprehension.
    # The same concentration also serves marginal fallback rows, whose
    # implied variance need not equal the paired model's marginal variance.
    # The small old-anchor excess has a wider prior; see the note above.
    anchor_ages=(18.0, 72.0),
    kappa_min_mu=math.log(7.8),
    kappa_min_sigma=0.8,
    excess_young_mu=math.log(10.4),
    excess_young_sigma=1.0,
    excess_old_mu=math.log(0.6),
    excess_old_sigma=1.5,
)


# ============================================================
# Shared DS joint trajectory-prior and reporting values
# ============================================================
#
# Shared prior and reporting groups keep related DS definitions consistent.
# tests/test_ds_joint_shared_priors.py checks which registrations inherit each
# group. TD models retain their own age anchors and priors.


class _DSJointUnderstoodAnchors(TypedDict):
    """Field types for :data:`_DS_JOINT_UNDERSTOOD_ANCHORS`.

    A ``TypedDict`` rather than ``dict[str, float]`` so mypy still checks the
    splat against each definition class's own parameter types -- a plain dict
    erases them and the narrow type coverage on this module stops doing its job.
    """

    p_slope_low_u_alpha: float
    p_slope_low_u_beta: float
    p_slope_hi_u_alpha: float
    p_slope_hi_u_beta: float
    eta_u_sigma: float


_DS_JOINT_UNDERSTOOD_ANCHORS: _DSJointUnderstoodAnchors = {
    # Understood anchors. Recalibrated 2026-08-04 (see the note referenced below)
    # from Beta(1,7)/Beta(2,1.5), which left the prior median population curve
    # ~100 words below the fitted one across 24-60 months and put 80% of prior
    # mass below the frame's own median there (87% at 48 mo). 24 mo: median 76 ->
    # 108 words, against a frame median of 132 over the densest band in the pool
    # (160 rows, 156 children) and fitted anchors of 109-113 in the three models
    # that identify this parameter (VG10/VG15/VG16). 84 mo: median 475 -> 592,
    # between the four administrations observed at 78-95 mo (median 554) and
    # those same fitted anchors (658-663); the tails stay wide because the
    # evidence there is thin. Both are scale calibration on the project's own
    # frame, not an independent norm -- there is none for DS comprehension.
    #
    # This corrects the *level* only. The prior is still logit-linear in age
    # between the anchors while the trajectory is strongly concave on that scale,
    # so lifting the line to fit 24-60 raises its backward extrapolation too and
    # the 12-18 mo end gets worse, not better. eta_u absorbs the difference and
    # sat at prior CDF 0.80-0.89 across all eight DS joint models registered when
    # this was measured on 2026-08-04. The fix is a log-age mean, which is a graph
    # change and therefore a new variant; see
    # notes/202608041216-ds-understood-trajectory-prior.md.
    "p_slope_low_u_alpha": 1.5,
    "p_slope_low_u_beta": 8.0,
    "p_slope_hi_u_alpha": 3.0,
    "p_slope_hi_u_beta": 1.3,
    "eta_u_sigma": 0.6,
}


class _DSJointQAnchors(TypedDict):
    """Field types for :data:`_DS_JOINT_Q_ANCHORS`. See
    :class:`_DSJointUnderstoodAnchors` for why this is a ``TypedDict``."""

    p_slope_low_q_alpha: float
    p_slope_low_q_beta: float
    p_slope_hi_q_alpha: float
    p_slope_hi_q_beta: float


_DS_JOINT_Q_ANCHORS: _DSJointQAnchors = {
    # q anchors. Broadened from the VG07-posterior-derived Beta(3,22)/Beta(20,4)
    # to remove prior-data double-dipping; the high anchor then recalibrated
    # 2026-08-04 from Beta(3,2). q_low ~ Beta(2,12) is unchanged and well-centred
    # (fitted 0.117 at prior CDF 0.46, contraction 0.81); the high anchor was
    # carrying the whole displacement. A weighted least-squares line through the
    # directly observed spoken/understood ratio (902 rows with both outcomes,
    # 18-72 mo) implies a trend q(84) of 0.946; unweighted 0.924, 36 mo+ 0.943.
    # Beta(3,2)'s median of 0.614 left the prior trend line 1.9x too shallow,
    # putting the prior median spoken curve 12x above the fitted one at 12 mo and
    # 2.2x below it at 54. Beta(4,1.2) has median 0.805 -- deliberately short of
    # the observed extrapolation, because the last band carrying both outcomes is
    # 72 mo (n=11) and only one row has both above 78 -- with a wide lower tail
    # (5-95% 0.44-0.98). Frame calibration, not an independent norm: there is
    # none for the DS production ratio. See
    # notes/202608041730-ds-spoken-q-trajectory-prior.md.
    "p_slope_low_q_alpha": 2.0,
    "p_slope_low_q_beta": 12.0,
    "p_slope_hi_q_alpha": 4.0,
    "p_slope_hi_q_beta": 1.2,
}


class _DSJointReporting(TypedDict):
    """Field types for :data:`_DS_JOINT_REPORTING`. See
    :class:`_DSJointUnderstoodAnchors` for why this is a ``TypedDict``.

    ``clamp_mean_above_hi_anchor`` is ``bool | str`` because
    :data:`CLAMP_Q_ONLY` is a sentinel string -- see the field's own docstring."""

    report_max_age_understood: int
    clamp_mean_above_hi_anchor: bool | str


_DS_JOINT_REPORTING: _DSJointReporting = {
    # Level the mean off above the 84 mo high anchor rather than extrapolating the
    # line to the top of the 115 mo GP domain. Without it the fitted q mean alone
    # reaches 0.993 at 115 mo (P(mean > 0.99) = 0.90 across the posterior) against a
    # realised 0.842, so the GP spends -3.3 logits correcting the mean's asymptote
    # while sitting idle (+0.08) at 48 mo where the data are; understood shows the
    # same defect about 3x milder. One-sided, and the corner is rounded over about
    # +/-4 mo so the curve stays monotone -- see gp_utils.trend_and_gp.
    #
    # Comprehension reporting stops at 72 mo. Lowered from 84 on 2026-08-22 by the
    # study owner. This does not overturn the 2026-08-13 raise on its own terms: it
    # asked whether the 72-84 band is observed rather than extrapolated, and the
    # answer was and remains yes -- 24 understood rows from 18 children across five
    # studies (ie_01, uk_01, uk_06, uk_07, us_02; it was 25 rows from 20 children
    # when the decision was taken, and the uk_01 correction of 2026-08-31 removed
    # one row, which changes nothing about the argument). It applies a second and stricter
    # test that the raise did not: is the number in that band fixed by the data, or
    # by the model? It is not fixed by the data. VG19 and VG20 differ only in the
    # child-effect structure and are indistinguishable out of sample (k-fold LOSO
    # +0.93 SE), yet they put q at 0.75 against 0.85 at 72 mo and 0.83 against 0.94
    # at 84 -- gaps of 0.89 and 0.93 of VG20's own 89% ETI width. Two dozen
    # comprehension
    # observations cannot separate the two structures, so above 72 the report would
    # quote a modelling choice as a measurement. Below 60 the same comparison never
    # exceeds 0.15 interval widths, so the cap lands where the data stop determining
    # the answer rather than where they stop existing.
    #
    # The binding quantity is now q, not understood: the three models agree on the
    # understood curve to within 0.15 interval widths at every age to 84, so
    # understood alone would still support 84. It is trimmed with q because both
    # ride this one field and q is conditioned on understood -- see
    # reporting_ages.ReportedQuantity.RATIO_OF_UNDERSTOOD. Giving q its own field
    # would invalidate every fit of every definition class that declares this one,
    # which today is all twenty registered models, to express a cap that this field
    # already expresses correctly, if conservatively, for understood.
    #
    # Raise it again when the 72-84 band can *distinguish* the child structures, not
    # merely when it is populated -- so on new older-child comprehension data, and
    # by rerunning the comparison rather than recounting rows.
    #
    # Spoken keeps the full grid: 1428 rows, 95th percentile 81 mo, 59 rows at or
    # above 84, and the three models agree on its subject-marginal curve to within
    # 0.03 interval widths at every age. Reporting only -- it cannot move the
    # posterior. See notes/202608221200-reporting-source-by-quantity.md and
    # notes/202608042030-q-mean-extrapolation.md.
    "report_max_age_understood": 72,
    "clamp_mean_above_hi_anchor": CLAMP_Q_ONLY,
}


# ============================================================
# The definition union, and deriving one definition class from another
# ============================================================

#: Every definition class a registered model can be an instance of.
#:
#: Spelled once here rather than inline at :data:`MODEL_REGISTRY`, so the functions
#: that take "a definition" can say so. That matters more than it looks: mypy covers
#: four modules in this project, and an unannotated ``definition`` parameter makes
#: every field access on it ``Any`` — which is the check the narrow coverage exists
#: to provide. Only the four *base* classes are named; every derived subclass is an
#: instance of one of them.
ModelDefinition = (
    UnivariateModelDefinition
    | BivariateModelDefinition
    | TrivariateModelDefinition
    | JointModelDefinition
)


def _as_definition_subclass[Derived](
    base: ModelDefinition,
    cls: type[Derived],
    **overrides: Any,
) -> Derived:
    """Rebuild ``base`` as an instance of ``cls``, overriding named fields.

    **This is the mechanism for adding a definition field without invalidating the
    parent's fits**, and it is why several apparent smells in this file are
    deliberate. A fit is validated by comparing the serialised definition field for
    field, so a new field on an existing class invalidates every fit of that class.
    Deriving a sibling subclass instead invalidates nothing: the parent's
    serialisation is untouched, and only the new class carries the new field. VG19's
    child slope, VG20's and VG23's correlation and VG22's factor all arrived this
    way, and it is why the engines read those fields through ``getattr`` rather than
    as plain attributes.

    Shallow by design: nested prior dataclasses are shared with ``base`` rather
    than copied, so the derived definition serialises identically to its parent
    except for what is overridden here. ``dataclasses.replace`` cannot do this —
    it returns the base's own class — and ``asdict`` cannot either, because it
    recursively converts the nested prior blocks to plain dicts.

    Generic in ``cls`` so a derived definition types as its own class rather than
    as ``Any``. Without that, five of the twenty registry entries — VG19-VG23,
    including VG22, whose ``subject_factor`` is exactly the union-typed field the
    coverage exists to police — went unchecked inside the one definition module
    mypy covers.

    Lives here, above the registrations and below the definition classes, rather
    than beside the first model that happens to use it: it is a mechanism shared by
    five of them, not a VG20 detail.
    """
    values = {item.name: getattr(base, item.name) for item in fields(base)}
    values.update(overrides)
    return cls(**values)


#: Prior SD of every sex coefficient the reporting models carry (issue #324).
#:
#: ``Normal(0, 0.5)`` on the logit scale, the prior the VG20 sex-shift
#: experiment used (``notes/202609041530-vg20-sex-shift-arm.md``): the
#: descriptive girl-minus-boy estimates are 0.2 to 0.35 logits and the typically
#: developing CDI norms put the whole effect well inside one SD. One constant for
#: both populations, so the Down syndrome and typically developing coefficients
#: are estimated under the same prior.
#:
#: Carried by the models whose numbers are reported -- the Down syndrome models
#: of record (VG15, VG20, VG24, and VG25 through VG24), the typically developing
#: references (VG11, VG12, VG21, VG23) and VG21's registered successor VG26 --
#: on the study owner's decision of 2026-09-08 that predictions should be
#: available by age and sex in both populations. The development steps do not
#: carry it, so VG10 and VG13 are no longer exact nested nulls of VG20 and VG23:
#: each pair now differs by the correlation and the sex term.
_SEX_EFFECT_SIGMA = 0.5


VG01 = UnivariateModelDefinition(
    model_id="VG01",
    config_name="age-spoken-ds",
    banner="Fitting Model VG01: Influence of age on words spoken (A -> S)",
    population=Population.DOWN_SYNDROME,
    outcome=Outcome.SPOKEN,
    n_trials=810,
    slope_anchors=(24, 84),
    ages_query=(12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90,),
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
    kappa=_DS_SPOKEN_KAPPA,
)

VG02 = UnivariateModelDefinition(
    model_id="VG02",
    config_name="age-understood-ds",
    banner="Fitting Model VG02: Influence of age on words understood (A -> U)",
    population=Population.DOWN_SYNDROME,
    outcome=Outcome.UNDERSTOOD,
    n_trials=810,
    slope_anchors=(24, 84),
    ages_query=(12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90,),
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
    # Comprehension reporting stops at 72 mo, matching the joint models; lowered
    # from 84 on 2026-08-22. VG02 carries no q, so the model-dependence argument
    # that binds the joint models (see VG05) does not arise here on its own terms --
    # understood is the robust quantity, agreeing across VG10, VG19 and VG20 to
    # within 0.15 interval widths at every age to 84. It follows the joint models
    # anyway, because a single-outcome comprehension model reporting to 84 beside a
    # joint model reporting to 72 would invite the reader to prefer whichever
    # number ran further. Reporting only -- it cannot move the posterior. The
    # whole-month companion still covers the full observed span, where its n_obs
    # column records the emptiness directly. See
    # notes/202608221200-reporting-source-by-quantity.md and
    # notes/202608042030-q-mean-extrapolation.md.
    report_max_age_understood=72,
    kappa=_DS_UNDERSTOOD_KAPPA,
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
    ages_query=(9, 12, 15, 18, 21, 24, 27, 30,),
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
    # Spoken-only loading includes the WS production-only rows (see n_trials
    # note above), so the 25% subject subsample draws from the full TD spoken
    # pool and yields a frame of 4,075 rows (the figure the kappa calibration
    # table above records). An earlier rationale here claimed WS was excluded
    # and the frame was ~1,500 rows; both halves were stale (#234).
    sample_fraction=0.25,
    kappa=_TD_SPOKEN_KAPPA,
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
    ages_query=(9, 12, 15, 18, 21, 24, 27, 30,),
    # The 8-30 month HSGP domain is shared with VG03 and stays as it is.
    gp_domain_months=_TD_GP_DOMAIN_MONTHS,
    # Comprehension reporting stops at 25 months, 2026-08-17 (#228). Comprehension
    # rides only on the bivariate forms, and those stop at 25: on this model's
    # English-only pool the last 591 comprehension rows are Floccia's Oxford CDI
    # at 19-25 months, and NOTHING is observed at 26 or beyond. The query grid
    # nonetheless ran to 30, so `posterior_summary.csv` -- and the rendered report
    # table built from it -- published a 27- and a 30-month comprehension median
    # on zero observations, while this model's own figures already stopped at 25.
    # The gap between the HSGP domain and the reporting grid is exactly what this
    # field is for; the note this replaced treated keeping the domain at 30 and
    # reporting to 30 as the same decision, and they are not. Reporting only --
    # it cannot move the posterior. Separately, and not changed here: the 26 mo
    # high slope anchor VG04 shares with VG12 sits one month past the last
    # comprehension observation. That is a prior question rather than a reporting
    # one, and it is already registered as VG12's `hi-anchor-broad` sensitivity.
    report_max_age_understood=25,
    # 12 mo understood low anchor — independent Wordbank TD norm: comprehension
    # median ~84 words/810 at 12 mo. Beta(1.2, 8) matches at median ~84 (the
    # in-sample mean ~82 corroborates); the old Beta(1,20) centred it at ~28, well
    # below the norm. See docs/models/PRIORS.md, "TD anchor priors vs Wordbank
    # norms".
    p_slope_low_alpha=1.2,
    p_slope_low_beta=8.0,
    # 26 mo understood high anchor — NO independent CDI comprehension norm (WS is
    # production-only), so Beta(1.3, 1.3) is broad regularisation and a named
    # sensitivity target in PRIORS.md, not an externally anchored value. Tested
    # (#147, 2026-08-18, `vg12 hi-anchor-broad` at `rep`): reverting it to the vague
    # Beta(1.1, 1.1) moved 30 mo comprehension by 0.63 words against an 89%
    # interval 152.8 words wide. The posterior does not lean on this anchor.
    p_slope_hi_alpha=1.3,
    p_slope_hi_beta=1.3,
    eta_sigma=0.5,
    # Bumped from 0.1: total comprehension pool shrank from 16,552 to 6,134
    # after the WS exclusion; this keeps the effective training set
    # (~1,500 rows) close to the previous VG04 fit.
    sample_fraction=0.25,
    kappa=_TD_UNDERSTOOD_KAPPA,
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
    ages_query=(12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90,),
    gp_domain_months=_DS_GP_DOMAIN_MONTHS,
    # Understood trajectory anchors -- rationale at the constant.
    **_DS_JOINT_UNDERSTOOD_ANCHORS,
    # Production-ratio anchors -- rationale at the constant.
    **_DS_JOINT_Q_ANCHORS,
    # Mean clamp + comprehension reporting cap -- rationale at the constant.
    **_DS_JOINT_REPORTING,
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
    ages_query=(12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90,),
    gp_domain_months=_DS_GP_DOMAIN_MONTHS,
    # Understood trajectory anchors -- rationale at the constant.
    **_DS_JOINT_UNDERSTOOD_ANCHORS,
    # Production-ratio anchors -- rationale at the constant.
    **_DS_JOINT_Q_ANCHORS,
    tau_u_sigma=0.5,
    tau_q_sigma=0.5,
    # Mean clamp + comprehension reporting cap -- rationale at the constant.
    **_DS_JOINT_REPORTING,
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
    ages_query=(12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90,),
    gp_domain_months=_DS_GP_DOMAIN_MONTHS,
    # Understood trajectory anchors -- rationale at the constant.
    **_DS_JOINT_UNDERSTOOD_ANCHORS,
    # Production-ratio anchors -- rationale at the constant.
    **_DS_JOINT_Q_ANCHORS,
    tau_u_sigma=0.5,
    tau_q_sigma=0.5,
    use_subject_re_u=True,
    tau_subj_u_sigma=1.5,
    # Mean clamp + comprehension reporting cap -- rationale at the constant.
    **_DS_JOINT_REPORTING,
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
    ages_query=(12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90,),
    gp_domain_months=_DS_GP_DOMAIN_MONTHS,
    # Understood trajectory anchors -- rationale at the constant.
    **_DS_JOINT_UNDERSTOOD_ANCHORS,
    # Production-ratio anchors -- rationale at the constant.
    **_DS_JOINT_Q_ANCHORS,
    tau_u_sigma=0.5,
    tau_q_sigma=0.5,
    use_subject_re_u=True,
    tau_subj_u_sigma=1.5,
    use_subject_re_q=True,
    tau_subj_q_sigma=1.5,
    kappa_u=_DS_JOINT_UNDERSTOOD_KAPPA_RE,
    kappa_s=_DS_JOINT_Q_KAPPA_RE,
    # Mean clamp + comprehension reporting cap -- rationale at the constant.
    **_DS_JOINT_REPORTING,
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
    ages_query=(12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90,),
    gp_domain_months=_DS_GP_DOMAIN_MONTHS,
    # Understood trajectory anchors -- rationale at the constant.
    **_DS_JOINT_UNDERSTOOD_ANCHORS,
    # Production-ratio anchors -- rationale at the constant.
    **_DS_JOINT_Q_ANCHORS,
    tau_u_sigma=0.5,
    tau_q_sigma=0.5,
    use_subject_re_u=True,
    tau_subj_u_sigma=1.5,
    use_subject_re_q=True,
    tau_subj_q_sigma=1.5,
    # GP anchor constraint (Option D) — applied symmetrically to both trajectories
    anchor_g_u_at_ref=True,
    anchor_g_q_at_ref=True,
    gp_anchor_age_months=54.0,
    kappa_u=_DS_JOINT_UNDERSTOOD_KAPPA_RE,
    kappa_s=_DS_JOINT_Q_KAPPA_RE,
    # Mean clamp + comprehension reporting cap -- rationale at the constant.
    **_DS_JOINT_REPORTING,
)

VG11 = UnivariateREModelDefinition(
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
    ages_query=(9, 12, 15, 18, 21, 24, 27, 30,),
    gp_domain_months=_TD_GP_DOMAIN_MONTHS,
    # Spoken trajectory anchors shared with VG03 (see the note there): lower the
    # 12 mo anchor for delayed TD production and soften the 26 mo anchor. The GP
    # amplitude is NOT shared with VG03 any more -- see eta_sigma below.
    p_slope_low_alpha=1.0,
    p_slope_low_beta=30.0,
    p_slope_hi_alpha=1.3,
    p_slope_hi_beta=1.3,
    # Retain eta-wide as a sensitivity to this amplitude prior.
    # The study-owner decision and dated sampling comparisons are in
    # notes/202609161440-vg11-eta-sigma-0.4.md.
    eta_sigma=0.4,
    # Use all bivariate-capable rows (WG + Oxford CDI) plus WS production rows.
    # Use the full admitted pool; study offsets allow between-study level differences.
    sample_fraction=1.0,
    # Include the selected English, Italian and Spanish reference sources.
    # Study offsets allow level differences, not arbitrary language-by-age
    # differences or measurement equivalence. See ROMANCE_LANGUAGES.
    td_languages=ENGLISH_AND_ROMANCE_LANGUAGES,
    # Study-level random intercepts on the spoken trajectory
    tau_study_sigma=0.5,
    # Drop datasets with <200 observations (issue #55): roughly halves the study
    # count while retaining >97% of observations.
    min_study_observations=200,
    use_subject_re=True,
    tau_subject_sigma=1.5,
    # Anchor the GP at the midpoint of slope_anchors (19 months) to remove the
    # GP–intercept ridge that arises when study REs are present.
    anchor_g_at_ref=True,
    gp_anchor_age_months=19.0,
    # Centred study effects and a variance partition follow the dated
    # comparisons in notes/202608050900-td-hierarchical-geometry.md.
    # They do not guarantee convergence or resolve sparse child replication.
    centred_study_re=True,
    subject_variance_partition=_TD_SPOKEN_VARIANCE_PARTITION,
    kappa=_TD_SPOKEN_KAPPA_RE,
    # Sex as a covariate for the by-sex predictions (#324); see the constant.
    sex_effect_sigma=_SEX_EFFECT_SIGMA,
)

VG12 = UnivariateREModelDefinition(
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
    ages_query=(9, 12, 15, 18, 21, 24, 27, 30,),
    # As in VG04, observed comprehension ends at 25 months while reporting
    # reaches 30; this preserves the established 8-30 month HSGP domain.
    gp_domain_months=_TD_GP_DOMAIN_MONTHS,
    # Understood trajectory priors shared with VG04 (see the note there): the
    # 12 mo low anchor is anchored to the independent Wordbank comprehension norm
    # (~83 words), while the 26 mo high anchor has no independent CDI norm (WS is
    # production-only) and remains broad regularisation / a sensitivity target —
    # tested robust under #147 (`hi-anchor-broad`, `lo-anchor-broad` and
    # `eta-narrow` all inside the baseline 89% interval at every query age; see
    # docs/models/PRIORS.md, "Sensitivity results").
    p_slope_low_alpha=1.2,
    p_slope_low_beta=8.0,
    p_slope_hi_alpha=1.3,
    p_slope_hi_beta=1.3,
    # Keep the narrower amplitude following the dated sampling comparisons
    # in notes/202608050900-td-hierarchical-geometry.md. Their divergence counts
    # do not prove one geometric cause or establish fit quality for future runs.
    eta_sigma=0.5,
    # WG + Oxford CDI only (WS comprehension is a production proxy).
    # Use the full admitted pool; study offsets allow between-study level differences.
    sample_fraction=1.0,
    # Include the selected English, Italian and Spanish reference sources.
    # Study offsets allow level differences, not arbitrary language-by-age
    # differences or measurement equivalence. See ROMANCE_LANGUAGES.
    td_languages=ENGLISH_AND_ROMANCE_LANGUAGES,
    # Study-level random intercepts on the understood trajectory
    tau_study_sigma=0.5,
    # Drop datasets with <200 observations (issue #55): roughly halves the study
    # count while retaining >97% of observations.
    min_study_observations=200,
    use_subject_re=True,
    tau_subject_sigma=1.5,
    # Anchor the GP at the midpoint of slope_anchors (19 months).
    anchor_g_at_ref=True,
    gp_anchor_age_months=19.0,
    # Centred study effects and a variance partition follow the dated
    # comparisons in notes/202608050900-td-hierarchical-geometry.md.
    # They do not guarantee convergence or resolve sparse child replication.
    centred_study_re=True,
    subject_variance_partition=_TD_UNDERSTOOD_VARIANCE_PARTITION,
    # Comprehension reporting stops at 25 months, 2026-08-17 (#228), for the same
    # reason as VG04 and with more data behind it: comprehension rides only on the
    # bivariate forms, and in this model's widened language scope they stop at 25
    # (Floccia's Oxford CDI to 25, Caselli's Italian Words & Gestures to 24 --
    # 694 rows above 18 months and none at all above 25). The query grid ran to
    # 30, so the published report table carried a 27- and a 30-month median on no
    # observations while this model's own figures already stopped at 25.
    # Reporting only -- it cannot move the posterior.
    report_max_age_understood=25,
    kappa=_TD_UNDERSTOOD_KAPPA_RE,
    # Sex as a covariate for the by-sex predictions (#324); see the constant.
    sex_effect_sigma=_SEX_EFFECT_SIGMA,
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
    # Restrict to 8–18 months, where WG/Oxford CDI data are dense.
    #
    # This cap does NOT do the Words & Sentences work an earlier version of this
    # comment claimed for it. `load_data` selects `WORDBANK_BIVARIATE_FORMS`
    # whenever `understood` is requested, so WS is never loaded for a
    # comprehension model at any age — the form filter avoids the production-proxy
    # bias unconditionally, and the cap adds nothing to it. The cap's real
    # justification, from the July review, was that above 18 months only Oxford
    # CDI supplied bivariate rows: a single study. The Romance extension of
    # 2026-08-03 retired that by admitting Italian Words & Gestures (registered
    # 7-24), and 694 admissible administrations from two studies now sit above the
    # cap — VG12 already fits every one of them. Density above 18 months is real
    # but thin (36-163 rows a month against 499 at 18), and the Oxford CDI's
    # 418-item ceiling binds hard at 23-25 months, so the cap is defensible; it is
    # no longer self-evident. The `window-25` / `window-22` variants measure what
    # it costs. See notes/202608171500-reporting-scope-audit.md and #228.
    max_age_months=18,
    slope_anchors=(10, 16),
    ages_query=(8, 10, 12, 14, 16, 18,),
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
    # Production ratio q = P(speak | understood). A calibration proxy uses
    # the ratio of Wordbank median production to median comprehension: ~0.12 at
    # 10 mo rising to ~0.19 at 16 mo (PRIORS.md, "Production ratio q(a) from
    # norms"). The shared bivariate defaults (lo Beta(1,1.5)~0.4, hi Beta(2,1.2)
    # ~0.62) are tuned for the DS 24/84 mo window and sit ~3x above this young-TD
    # curve, compounding with U to overshoot spoken ~5x. Set window-appropriate
    # anchors at/just below the norm floor: lo Beta(1,10) (median ~0.067), hi
    # Beta(2,7) (median ~0.201). The in-sample q (~0.09 at 10 mo, ~0.23 at 16 mo)
    # is similar. A ratio of marginal medians is not a median child ratio,
    # and data overlap means this is not independent validation.
    p_slope_low_q_alpha=1.0,
    p_slope_low_q_beta=10.0,
    p_slope_hi_q_alpha=2.0,
    p_slope_hi_q_beta=7.0,
    # Keep the pre-2026-08-04 q-GP amplitude. The family default was widened to
    # 0.8 because logit(q) is S-shaped across the DS 8-115 mo range and the GP is
    # the only term that can carry that curvature; over this model's 8-18 mo
    # window only the bottom limb of that S is in view, a straight line on the
    # logit scale is adequate, and VG13 is the one model in the family whose
    # eta_q is not prior-limited (prior CDF 0.572 against 0.95-0.99 elsewhere).
    # Widening here would buy nothing and would loosen a prior the data are
    # content with. See notes/202608041730-ds-spoken-q-trajectory-prior.md.
    eta_q_sigma=0.20,
    # Use all available bivariate rows in the 8–18 month window; study REs
    # allow between-study level differences.
    sample_fraction=1.0,
    # Include the selected English, Italian and Spanish reference sources.
    # Study offsets allow level differences, not arbitrary language-by-age
    # differences or measurement equivalence. See ROMANCE_LANGUAGES.
    td_languages=ENGLISH_AND_ROMANCE_LANGUAGES,
    # Dataset-level study random intercepts on both trajectories
    tau_u_sigma=0.5,
    tau_q_sigma=0.5,
    # Drop datasets with <200 observations (issue #55): roughly halves the study
    # count while retaining >97% of observations.
    min_study_observations=200,
    use_subject_re_u=True,
    tau_subj_u_sigma=1.5,
    use_subject_re_q=True,
    tau_subj_q_sigma=1.5,
    # Anchor GPs at the midpoint of slope_anchors (13 months)
    anchor_g_u_at_ref=True,
    anchor_g_q_at_ref=True,
    gp_anchor_age_months=13.0,
    kappa_u=_TD_YOUNG_UNDERSTOOD_KAPPA_RE,
    kappa_s=_TD_YOUNG_Q_KAPPA_RE,
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
    ages_query=(12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90,),
    gp_domain_months=_DS_GP_DOMAIN_MONTHS,
    # Migrated to the two-anchor dispersion form, 2026-08-06. VG14 had been left
    # on the class-default legacy priors, where `b_kappa_mag ~ HalfNormal(0.3)`
    # forces dispersion to fall with age and caps how fast. The spoken side
    # rejected that: posterior mean 1.214, about four standard deviations beyond
    # the prior, with contraction -0.09 -- the posterior wider than the prior. A
    # parameter pinned against a boundary is not an estimate.
    #
    # These are not a new calibration. They are the same objects VG10 and VG15
    # already use, and VG14 shares their population, outcomes, slope anchors and
    # GP domain exactly, so adopting them removes a difference that was never
    # deliberate. It also brings VG14 into line with VG15, the model it is most
    # directly compared against.
    #
    # `kappa_sign` deliberately stays legacy, matching VG15: the signed block sits
    # comfortably inside its prior (CDF 0.25, contraction 0.49) and has no reason
    # to move. See notes/202608051500-report-critical-review.md section 4a.
    #
    # KNOWN MISMATCH, retained deliberately (#238): these components are the
    # CONDITIONAL calibration — kappa given study and child random effects,
    # which VG14 does not have. VG14's kappa is marginal dispersion, a different
    # estimand; the no-effects calibration on VG14's own frame targets totals of
    # roughly 13.7/3.2 (understood at 18/72 months) against these components'
    # 92.6/14.0, so the priors favour substantially less unexplained variation
    # than a calibration matching this graph would. Not recalibrated because
    # VG14 is superseded by VG15 for every reported number and supplies no
    # inferential output (docs/models/README.md role table); if VG14 is ever
    # retained for substantive use, recalibrate these first
    # (scripts/kappa_conditional_calibration.py without the RE conditioning).
    kappa_u=_DS_JOINT_UNDERSTOOD_KAPPA_RE,
    kappa_s=_DS_JOINT_Q_KAPPA_RE,
    # Understood trajectory anchors -- the reasoning is at
    # `_DS_JOINT_UNDERSTOOD_ANCHORS`, which is where VG05's copy of it went, and
    # notes/202608041216-ds-understood-trajectory-prior.md has the measurements.
    **_DS_JOINT_UNDERSTOOD_ANCHORS,
    # Production-ratio anchors -- the calibration is at `_DS_JOINT_Q_ANCHORS`,
    # which is where VG05's copy of it went, and
    # notes/202608041730-ds-spoken-q-trajectory-prior.md has the measurements.
    **_DS_JOINT_Q_ANCHORS,
    # uk_01's signed-only field is excluded from total signing by default.
    # Confirmed uk_06 totals remain included; other outcomes follow their masks.
    **_DS_JOINT_REPORTING,
    # Signed gets its own cap rather than inheriting the comprehension one, which
    # is what it did until 2026-08-13. 84 matches VG15's report_max_age_signed on
    # the same evidence, and stops the r(a) table where the r(a) figure stops.
    report_max_age_signed=84,
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
    ages_query=(12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90,),
    gp_domain_months=_DS_GP_DOMAIN_MONTHS,
    # r/q/p_U priors seeded from the (uk_06-included) VG14 fit (see dataclass
    # defaults); psi ~ logNormal(0.3, 0.5) (weakly positive, spans independence);
    # study random intercepts on f_U/g/h (tau_*=0.5).
    #
    # Issue #59 — subject random intercepts throughout + VG10 stabilisation:
    # Option A (ported from VG10), now broadened: the q anchors are
    # weakly-informative (q_low ~ Beta(2,12) at the independent TD q ~= 0.12
    # centre), replacing the VG07-posterior-derived Beta(3,22)/Beta(20,4). The
    # high anchor was recalibrated 2026-08-04 from Beta(3,2) to Beta(4,1.2)
    # alongside the rest of the DS joint family — see VG05 for the calibration and
    # notes/202608041730-ds-spoken-q-trajectory-prior.md. The u anchors are left
    # unchanged, matching VG10. The signed mean is a three-anchor hump (tent),
    # inherited from the JointModelDefinition dataclass defaults (young/peak/old
    # sign anchors + GP), so there is no monotone signed slope to tighten; the
    # anchors set the level and the GP carries smooth departures. Option D (below)
    # removes the GP<->intercept ridge.
    # Production-ratio anchors -- rationale at the constant.
    **_DS_JOINT_Q_ANCHORS,
    # Subject random intercepts on all three trajectories. Signed data has more
    # repeated-subject structure than first feared (substantial repeats across
    # uk_01/02/04/05), so the sign-subject RE is strongly data-identified — its
    # scale sat *well above* the old HalfNormal(0.5) prior (posterior 1.082 at
    # prior CDF 0.970), reflecting large between-child variation in signing, and it
    # improves out-of-sample fit. That conflict is now resolved by the family-wide
    # move to HalfNormal(1.5), which puts it at 0.53 — the signed ratio has no
    # calibration of its own, so it inherits the scale rather than being fitted to
    # one. use_subject_re_sign gates a one-line fallback to study-RE-only if a
    # future fit misbehaves. Note:
    # the four-cell DM is fed population+study marginals only, so this RE does not
    # pull the headline association psi (see the engine comment + the #59 note).
    use_subject_re_u=True,
    tau_subj_u_sigma=1.5,
    use_subject_re_q=True,
    tau_subj_q_sigma=1.5,
    use_subject_re_sign=True,
    tau_subj_sign_sigma=1.5,
    # Option D (ported from VG10): per-draw GP anchor at the reference age
    # (54 mo = midpoint of the (24, 84) anchors), applied to all three GPs to
    # remove the GP<->intercept redundancy that worsens once subject REs add
    # another level-carrying term to each predictor.
    anchor_g_u_at_ref=True,
    anchor_g_q_at_ref=True,
    anchor_g_sign_at_ref=True,
    # Peak age estimated rather than asserted, adopted 2026-08-06; see the field
    # docstring on JointModelDefinition. Beta(2, 4) puts the prior median at 40
    # months, deliberately ABOVE the 29.4 the data pull it to, so the estimate
    # moves against the prior rather than with it. Checked by the three
    # sign-peak-age-* sensitivity variants.
    sign_peak_prior=(2.0, 4.0),
    gp_anchor_age_months=54.0,
    # Understood and spoken share VG09's frame and specification, so they take
    # the same two-anchor blocks. The signed ratio stays on the legacy form:
    # nothing calibrates it, and its cross-tabulated cells are not a scale the
    # conditional estimator reproduces.
    kappa_u=_DS_JOINT_UNDERSTOOD_KAPPA_RE,
    kappa_s=_DS_JOINT_Q_KAPPA_RE,
    # Mean clamp + comprehension reporting cap -- rationale at the constant.
    **_DS_JOINT_REPORTING,
    # Signed evidence now reaches 84 months. Adopted 2026-08-07 at 60 on the same
    # argument that capped comprehension at 72 -- then 46 of 593 signed
    # observations lay above 60 and none above 72. uk_07 (PACT-DS) adds 82
    # observations at 34-95 months and uk_06's 11 at 60-115 were unmasked, taking
    # 72-84 from 0 rows to 17 and above-60 from 46 to 98, so 60 no longer marks
    # where the evidence stops -- it hid the pool's only real measurement of the
    # post-peak signing decline. Above 84 stays out: 12 observations from 10
    # children thinning to one source per band, and 84 is the trend's high anchor.
    # This caps the signed COUNTS. r and p_any are ratios of understood built
    # from the signed ratio, so they take the tighter of this cap and the
    # comprehension one (reporting_ages.max_age_for_sign_ratio) -- currently 72.
    report_max_age_signed=84,
    # Sex as a covariate for the by-sex predictions (#324); see the constant and
    # the field, which says why it reaches the cross-tab compositions. VG24 and
    # VG25 inherit it.
    sex_effect_sigma=_SEX_EFFECT_SIGMA,
)

# ============================================================
# VG16 — within-child cross-lag (issue #113): VG09 + prior understood -> current q
# ============================================================

VG16 = BivariateModelDefinition(
    model_id="VG16",
    config_name="age-understood-spoken-ds-re-subj-uq-crosslag",
    banner=(
        "Fitting Model VG16: VG09 + cross-lag (prior understood -> current q;"
        " population-relative baseline) - Down syndrome"
    ),
    population=Population.DOWN_SYNDROME,
    n_trials=810,
    slope_anchors=(24, 84),
    ages_query=(12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90,),
    gp_domain_months=_DS_GP_DOMAIN_MONTHS,
    # Understood trajectory anchors -- rationale at the constant.
    **_DS_JOINT_UNDERSTOOD_ANCHORS,
    # Production-ratio anchors -- rationale at the constant.
    **_DS_JOINT_Q_ANCHORS,
    tau_u_sigma=0.5,
    tau_q_sigma=0.5,
    use_subject_re_u=True,
    tau_subj_u_sigma=1.5,
    use_subject_re_q=True,
    tau_subj_q_sigma=1.5,
    use_cross_lag=True,
    # Headline uses the population-relative baseline. The original "bias-robust"
    # rationale — that the within-child baseline carries a short-T / Nickell /
    # errors-in-variables bias (dev: beta -0.60 [-0.85,-0.35]) — was withdrawn:
    # the negative figure was dev-tier non-convergence, and simulation from this
    # model's own posterior found no such mechanism
    # (notes/202608151500-within-child-crosslag-feasibility.md §§6-7). What
    # survives: the population-relative predictor conditions on no estimated
    # per-child quantity, and it deliberately retains persistent between-child
    # standing — so beta_lag is a history-dependent mixture of between- and
    # within-child association, not an isolated within-child dynamic
    # (notes/202608231714-vg16-statistical-model-review.md §4.1, issue #242).
    # The within-child variant remains an off-record cautionary contrast.
    lag_baseline="population",
    beta_lag_mu=0.0,
    beta_lag_sigma=0.5,
    # Option D (ported from VG10, as VG15 already does): per-draw GP anchor at
    # 54 mo (the midpoint of the (24, 84) anchors). Added 2026-08-02.
    #
    # VG16 was specified as "VG09 plus a cross-lag" and so inherited VG09's
    # *unanchored* geometry, making it the only model with subject random effects
    # on both u and q that lacked the stabilisation. It was correspondingly the
    # worst-behaved model in the family: 47 divergences (4.7% of draws, against
    # 0-0.4% everywhere else) and max scalar R-hat 1.126 on eta_u.
    #
    # The diagnosis is the understood-trajectory GP-versus-linear ridge that
    # motivated VG10. Measured on the dev traces, the anchoring is what removes
    # it — posterior correlations VG09 -> VG10: intercept_u/slope_u -0.54 ->
    # -0.27, intercept_u/eta_u -0.37 -> +0.03, intercept_u/ell_unit_u -0.47 ->
    # -0.09, while the intrinsic eta_u/ell_unit_u correlation is untouched
    # (+0.43 -> +0.46). VG10's max scalar R-hat is 1.017 against VG09's 1.252.
    #
    # See notes/202608020829-kappa-and-eta-q-prior-recalibration.md §§15-16.
    anchor_g_u_at_ref=True,
    anchor_g_q_at_ref=True,
    gp_anchor_age_months=54.0,
    kappa_u=_DS_JOINT_UNDERSTOOD_KAPPA_RE,
    kappa_s=_DS_JOINT_Q_KAPPA_RE,
    # Mean clamp + comprehension reporting cap -- rationale at the constant.
    **_DS_JOINT_REPORTING,
)

# ============================================================
# VG20 — correlated subject random effects (issue #224): VG10 + rho_uq
# ============================================================


# The correlated child block reduces to VG10's at rho_uq=0. VG20 also adds
# sex, so the registered models do not differ in only one term.
# Zero-child curves do not directly use rho; their fitted parameters can still
# move when the child likelihood changes. At fixed parameters, marginal u and
# q expectations retain their individual distributions; the product can depend
# on their correlation. Compare fitted curves and predictive checks.
VG20 = _as_definition_subclass(
    VG10,
    BivariateCorrelatedSubjectREModelDefinition,
    model_id="VG20",
    config_name="age-understood-spoken-ds-re-subj-uq-anchored-corr",
    banner=(
        "Fitting Model VG20: VG10 + correlated subject random effects on U and q"
        " (rho_uq) - Down syndrome"
    ),
    # eta=2 favours correlations near zero. Dated fitted child-deviate
    # correlations motivated the design but are not independent validation.
    # See notes/202608151120-vg16-crosslag-quantified.md.
    subject_re_correlation_eta=2.0,
    # Sex as a covariate for the by-sex predictions (#324); see the constant.
    sex_effect_sigma=_SEX_EFFECT_SIGMA,
)

VG19 = _as_definition_subclass(
    VG10,
    BivariateChildSlopeModelDefinition,
    model_id="VG19",
    config_name="age-understood-spoken-ds-re-subj-uq-anchored-slope",
    banner=(
        "Fitting Model VG19: VG10 + a child random slope on U and q"
        " - Down syndrome"
    ),
    # Gate 1 chose this structure on the fitted residuals before any of it was
    # written: a random slope is worth 2 x delta logL = 36.05 on spoken over a
    # constant intercept and survives restriction to the 334 children with
    # repeated spoken measures (20.81), while an AR(1) transient collapses to
    # zero persistence on both outcomes. See
    # notes/202608141600-rank-stability-tracking.md §10.3.
    #
    # `tau0_sigma` is VG10's own `tau_subj_*_sigma` of 1.5 on both outcomes, so
    # the intercept keeps the model of record's prior exactly and the model is
    # one-factor against it.
    #
    # `tau1_sigma = 0.5` is on the PER YEAR scale. In logit/month the ML values
    # are 0.02-ish and unreadable as a prior; per year they are 0.12-0.29, and
    # HalfNormal(0.5) has median 0.34 -- covering both comfortably while keeping
    # mass near zero, so a slope the data do not support shrinks away. The
    # nesting is exact at tau1 = 0.
    #
    # Both outcomes carry a slope because the evidence cannot say which one owns
    # the drift: it is measured on the spoken proportion, and spoken is p_u * q,
    # so the likelihood sees only the product. Expect the two tau1s to be
    # individually poorly identified and correlated, with the implied drift on
    # spoken far better determined than either alone.
    tau_subj_u_sigma=SubjectSlopePriorParams(tau0_sigma=1.5, tau1_sigma=0.5),
    tau_subj_q_sigma=SubjectSlopePriorParams(tau0_sigma=1.5, tau1_sigma=0.5),
    # Reference age for the intercept scale; slopes use years from this age.
    # Centring can reduce intercept/rate coupling but does not guarantee it.
    subject_slope_ref_age_months=36.0,
)

# Wider-window TD reference derived from VG13. The Oxford CDI ceiling at
# older ages motivated the 22-month limit. Anchors and GP priors also change;
# this is not a window-only contrast. Some priors use in-sample values.
# See notes/202608211100-window-22-adopted.md and
# notes/202608211545-window-22-prior-gate-passed.md for the dated assessments.
VG21 = _as_definition_subclass(
    VG13,
    BivariateModelDefinition,
    model_id="VG21",
    config_name="age-understood-spoken-td-re-window22",
    banner=(
        "Fitting Model VG21: Joint words understood + spoken (TD, 8-22 months) "
        "with dataset-level study random intercepts"
    ),
    max_age_months=22,
    slope_anchors=(10, 21),
    ages_query=(8, 10, 12, 14, 16, 18, 20, 22,),
    gp_domain_months=(8, 22),
    gp_anchor_age_months=15.5,
    # 21 mo in-sample medians: understood 0.359 of 810, q 0.417.
    p_slope_hi_u_alpha=2.0,
    p_slope_hi_u_beta=3.2,
    p_slope_hi_q_alpha=2.0,
    p_slope_hi_q_beta=2.6,
    # VG13 keeps 0.20 because only the bottom limb of the logit-q S-curve is in
    # its window; four more months brings enough curvature to need the family's
    # wider amplitude.
    eta_q_sigma=0.5,
    kappa_u=_TD_WINDOW22_UNDERSTOOD_KAPPA_RE,
    kappa_s=_TD_WINDOW22_Q_KAPPA_RE,
    # Sex as a covariate for the by-sex predictions (#324); see the constant.
    # VG26 inherits it.
    sex_effect_sigma=_SEX_EFFECT_SIGMA,
)


# Rank-three child-factor candidate derived from VG10. Population curves may
# change after refitting because the child structure changes the likelihood.
# The rank sensitivities compare one, two and three; exploratory residual
# likelihoods do not by themselves establish rank or parameter recovery.
# See notes/202608221000-four-by-four-gate1.md and
# notes/202608231420-vg22-factor-anchor-bimodality.md for the design history.
VG22 = _as_definition_subclass(
    VG10,
    BivariateFactorSubjectREModelDefinition,
    model_id="VG22",
    config_name="age-understood-spoken-ds-re-subj-uq-anchored-factor",
    # No rank in the banner: the sensitivity machinery appends
    # "[sensitivity: rank-N]" to this string, so a hard-coded number contradicts
    # every variant that is not the default (the stored rank-3 fit is labelled
    # "a rank-2 factor ... [sensitivity: rank-3]").
    banner=(
        "Fitting Model VG22: VG10 + a low-rank factor over the four child effects"
        " (U and q, level and rate) - Down syndrome"
    ),
    # The two LEVEL scales stay VG10's own 1.5 on both outcomes, inherited
    # through `tau_subj_u_sigma` / `tau_subj_q_sigma` rather than restated, so
    # the intercepts keep the model of record's priors exactly.
    #
    # The two RATE scales are VG19's 0.5 per year, for the reason recorded there:
    # in logit/month the ML values are 0.02-ish and unreadable as a prior, per
    # year they are 0.12-0.29, and HalfNormal(0.5) has median 0.34 -- covering
    # both while keeping mass near zero so a rate the data do not support shrinks
    # away.
    subject_factor=SubjectFactorPriorParams(
        rank=3,
        tau1_u_sigma=0.5,
        tau1_q_sigma=0.5,
        ref_age_months=36.0,
    ),
)


# Correlated TD child intercepts derived from VG13. VG23 also includes sex,
# so rho_uq = 0 alone does not reproduce the registered VG13 likelihood.
# Paired counts inform correlation, but shared reporting can also contribute.
# The model cannot separate reporter effects from vocabulary differences or
# establish upper and lower bounds on the true child correlation.
# eta = 2 matches VG20's correlation prior, not its full model or likelihood.
VG23 = _as_definition_subclass(
    VG13,
    BivariateCorrelatedSubjectREModelDefinition,
    model_id="VG23",
    config_name="age-understood-spoken-td-re-young-corr",
    banner=(
        "Fitting Model VG23: VG13 + correlated subject random effects on U and q"
        " (rho_uq) - typically developing"
    ),
    subject_re_correlation_eta=2.0,
    # Sex as a covariate for the by-sex predictions (#324); see the constant.
    sex_effect_sigma=_SEX_EFFECT_SIGMA,
)

# Correlate VG15's child intercepts while retaining its other priors.
# rho_sign_q concerns persistent signed and spoken shares across children.
# It differs from psi's association across words within an administration
# and from VG25's prospective lag coefficient.
# Child intercepts enter marginal counts, not cell likelihoods. Paired signed
# and spoken marginals therefore provide the direct correlation evidence.
# See docs/models/vg24/index.qmd and
# notes/202609041722-sign-speech-modelling-proposals.md for source support.
VG24 = _as_definition_subclass(
    VG15,
    JointCorrelatedSubjectREModelDefinition,
    model_id="VG24",
    config_name="age-joint-signspeech-ds-corr",
    banner=(
        "Fitting Model VG24: VG15 + correlated subject random effects on U, q"
        " and the signed ratio (rho_uq, rho_u_sign, rho_sign_q) - Down syndrome"
    ),
    # eta = 2, matching VG20 and VG23 so the three models' rho_uq are
    # prior-comparable. At n = 3 this is a per-correlation prior SD of 0.41
    # rather than VG20's 0.45; see the field docstring.
    subject_re_correlation_eta=2.0,
)

# VG25 adds a prior-wave signed-share predictor to VG24's correlated effects.
# The coefficient is prospective conditional association, not a causal effect.
# Development history and earlier descriptive estimates are in
# notes/202609131044-model-review-implementation.md. Publication role remains
# unclassified, so the existing validation rules retain full strictness.
VG25 = _as_definition_subclass(
    VG24,
    JointCrossLagModelDefinition,
    model_id="VG25",
    config_name="age-joint-signspeech-ds-corr-signlag",
    banner=(
        "Fitting Model VG25: VG24 + sign -> speech cross-lag (prior-wave signed"
        " share of comprehension -> current q; within-child baseline)"
        " - Down syndrome"
    ),
    use_sign_cross_lag=True,
    # The within-child baseline defines the intended conditional association.
    # `rho_sign_q` models persistent association but does not prove that all
    # persistent confounding has been removed. Registered sensitivity:
    # `sign-lag-population`.
    sign_lag_baseline="within",
    # The lag enters the spoken marginal only (2026-09-15). In the cross-tab
    # compositions the within-child baseline carried each child's estimated
    # signing intercept into likelihoods child effects are kept out of, and the
    # first rep fit was bimodal. The measurements are on the field.
    sign_lag_in_cells=False,
    beta_sign_lag_mu=0.0,
    beta_sign_lag_sigma=0.5,
    # The correction uses the source wave's denominator at both boundaries.
    # The sign-lag-clip sensitivity measures the alternative on a fitted model.
    sign_lag_zero_handling=LAG_ZERO_CONTINUITY,
)

# Add correlated child intercepts to VG21's 8-22-month specification.
# VG21's anchors and GP priors differ from VG13's, so simply widening VG23
# would not reproduce this model. The zero-correlation child block matches
# VG21's. Shared reporting and age-related measurement differences still
# limit interpretation; compare with VG23's narrower-window estimate.
# Registration does not replace VG21's reporting role.
VG26 = _as_definition_subclass(
    VG21,
    BivariateCorrelatedSubjectREModelDefinition,
    model_id="VG26",
    config_name="age-understood-spoken-td-re-window22-corr",
    banner=(
        "Fitting Model VG26: VG21 + correlated subject random effects on U and q"
        " (rho_uq) - typically developing, 8-22 months"
    ),
    subject_re_correlation_eta=2.0,
)

MODEL_REGISTRY: dict[str, ModelDefinition] = {
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
    "vg19": VG19,
    "vg20": VG20,
    "vg21": VG21,
    "vg22": VG22,
    "vg23": VG23,
    "vg24": VG24,
    "vg25": VG25,
    "vg26": VG26,
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
            # An optional scale (`float | None`, None meaning "no such term")
            # is not a scale that must be positive: `None` is its off state.
            if field_value is None and "None" in str(item.type):
                continue
            if (
                not isinstance(field_value, (int, float))
                or not math.isfinite(field_value)
                or field_value <= 0
            ):
                raise ValueError(f"{field_path} must be positive; got {field_value!r}.")


def _kappa_priors(definition):
    """Yield every kappa prior block a definition carries, whatever its arity."""
    for item in fields(definition):
        if item.name == "kappa" or item.name.startswith("kappa_"):
            value = getattr(definition, item.name)
            if isinstance(value, (KappaPriorParams, KappaAnchorPriorParams)):
                yield item.name, value


def _kappa_anchor_ages(definition) -> list[float]:
    """Every two-anchor reference age in a definition, for the GP-domain check."""
    return [
        age
        for _, kp in _kappa_priors(definition)
        if isinstance(kp, KappaAnchorPriorParams)
        for age in kp.anchor_ages
    ]


def validate_model_definition(definition) -> None:
    """Fail early when a declarative model specification is internally invalid."""
    prefix = getattr(definition, "model_id", type(definition).__name__)
    for name, kp in _kappa_priors(definition):
        if isinstance(kp, KappaAnchorPriorParams) and not (
            len(kp.anchor_ages) == 2
            and all(math.isfinite(age) for age in kp.anchor_ages)
            and kp.anchor_ages[0] < kp.anchor_ages[1]
        ):
            raise ValueError(
                f"{prefix}.{name}.anchor_ages must be ordered (young, old)."
            )
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
    report_max_u = getattr(definition, "report_max_age_understood", None)
    if report_max_u is not None:
        # A univariate spoken model has no comprehension quantity to trim, so
        # setting this there would silently do nothing.
        outcome = getattr(definition, "outcome", None)
        if outcome is not None and outcome is not Outcome.UNDERSTOOD:
            raise ValueError(
                f"{prefix}.report_max_age_understood applies to comprehension"
                f" reporting, but the model's outcome is {outcome.value}."
            )
        if report_max_u < min(definition.ages_query):
            raise ValueError(
                f"{prefix}.report_max_age_understood would report no query age."
            )
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
        required_ages.extend(_kappa_anchor_ages(definition))
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
    td_languages = getattr(definition, "td_languages", None)
    if td_languages is not None:
        if not isinstance(td_languages, tuple) or not td_languages:
            raise ValueError(
                f"{prefix}.td_languages must be a non-empty tuple of Wordbank "
                "language names."
            )
        unknown = sorted(set(td_languages) - set(KNOWN_TD_LANGUAGES))
        if unknown:
            raise ValueError(
                f"{prefix}.td_languages contains language names that are not "
                f"admitted to the reference pool: {unknown}. Add them to "
                "ROMANCE_LANGUAGES (with the measurement checks its docstring "
                "records) before referencing them here — a name that does not "
                "match a Wordbank `language` value silently yields no rows."
            )
        # Deliberately not checked here: that a model going beyond English carries a
        # study random intercept to absorb between-language variation. Every
        # definition class names its study scale differently, and this comment is the
        # one place the mapping is written down:
        #   Univariate / UnivariateRE  -> tau_study_sigma
        #   Bivariate                  -> tau_u_sigma, tau_q_sigma
        #   Joint                      -> tau_u_sigma, tau_q_sigma, tau_sign_sigma
        #   Trivariate                 -> none (VG14 carries no study intercepts)
        # Note these are the STUDY scales; the per-child ones are tau_subj_*_sigma,
        # a distinct family (see the suffix legend in vocab_growth.models). Because
        # the names differ by class, an attribute-sniffing check would quietly pass
        # for a class it does not know and give false assurance -- and it would pass
        # vacuously for Trivariate, which has no such field to find. The requirement
        # is stated on ENGLISH_AND_ROMANCE_LANGUAGES and is a review matter.
    sign_anchors = getattr(definition, "sign_anchor_ages", None)
    if sign_anchors is not None and (
        len(sign_anchors) != 3
        or not all(math.isfinite(age) for age in sign_anchors)
        # Strictly increasing, not merely sorted: the tent's segment slopes
        # divide by the anchor gaps, so a duplicated anchor is a division by
        # zero in the model graph (#238).
        or not (sign_anchors[0] < sign_anchors[1] < sign_anchors[2])
    ):
        raise ValueError(
            f"{prefix}.sign_anchor_ages must be three strictly increasing ages."
        )
    sign_peak_prior = getattr(definition, "sign_peak_prior", None)
    if sign_peak_prior is not None and (
        len(sign_peak_prior) != 2
        or not all(
            math.isfinite(parameter) and parameter > 0
            for parameter in sign_peak_prior
        )
    ):
        # The engines index this pair straight into pz.Beta(alpha, beta), which
        # accepts and then samples garbage from a non-positive parameter (#238).
        raise ValueError(
            f"{prefix}.sign_peak_prior must be two finite positive Beta "
            "parameters (alpha, beta)."
        )
    if getattr(definition, "lag_baseline", LAG_BASELINES[0]) not in LAG_BASELINES:
        raise ValueError(
            f"{prefix}.lag_baseline must be one of {LAG_BASELINES}."
        )
    # Until 2026-09-01 `lag_zero_handling` was validated nowhere at this level: its
    # only rejection was inside `cross_lag.prev_wave_lag`, reached after data
    # preparation and prior configuration had already run.
    if (
        getattr(definition, "lag_zero_handling", LAG_ZERO_CLIP)
        not in LAG_ZERO_TREATMENTS
    ):
        raise ValueError(
            f"{prefix}.lag_zero_handling must be one of {LAG_ZERO_TREATMENTS}."
        )
    if getattr(definition, "use_cross_lag", False) and not getattr(
        definition, "use_subject_re_u", False
    ):
        raise ValueError(f"{prefix} cross-lag requires use_subject_re_u=True.")

    # VG25 (#297). The same two checks for the sign -> speech lag, against the
    # same tuples, and for the same reason the understood lag has them here as
    # well as in `cross_lag`: this one fires against a definition, before any
    # data is loaded, while the engine-side check sees what the graph will
    # actually contain. The child effect this baseline is defined relative to is
    # the SIGNED-ratio one, not the understood one -- a model with
    # `use_subject_re_u` and no `use_subject_re_sign` would otherwise pass here
    # and then silently give both baselines the same value.
    if getattr(definition, "sign_lag_baseline", LAG_BASELINES[0]) not in LAG_BASELINES:
        raise ValueError(
            f"{prefix}.sign_lag_baseline must be one of {LAG_BASELINES}."
        )
    if (
        getattr(definition, "sign_lag_zero_handling", LAG_ZERO_CLIP)
        not in LAG_ZERO_TREATMENTS
    ):
        raise ValueError(
            f"{prefix}.sign_lag_zero_handling must be one of {LAG_ZERO_TREATMENTS}."
        )
    if getattr(definition, "use_sign_cross_lag", False) and not getattr(
        definition, "use_subject_re_sign", False
    ):
        raise ValueError(
            f"{prefix} sign cross-lag requires use_subject_re_sign=True."
        )

    # VG24 (#296). The joint correlated block draws a child's three deviations
    # from one joint Normal, so it is defined only when all three blocks exist.
    # Rejected here rather than built partially: a two-of-three combination has
    # a defensible reading (correlate the pair, leave the third independent) and
    # a silent one (emit a 3x3 whose third row multiplies nothing), and the
    # engine would pick the silent one. If the pairwise model is ever wanted it
    # should be a registered variant that says so, not an unvalidated flag
    # combination.
    #
    # The joint engine requires all three correlated blocks. The bivariate
    # engine also checks its resolved plan before graph construction.
    correlation_eta = getattr(definition, "subject_re_correlation_eta", None)
    if correlation_eta is not None and (
        not isinstance(correlation_eta, (int, float))
        or isinstance(correlation_eta, bool)
        or not math.isfinite(correlation_eta)
        or correlation_eta <= 0
    ):
        raise ValueError(
            f"{prefix}.subject_re_correlation_eta must be a positive finite LKJ "
            f"concentration; got {correlation_eta!r}."
        )
    if correlation_eta is not None and definition.model_type is ModelType.JOINT:
        missing_blocks = [
            name
            for name in ("use_subject_re_u", "use_subject_re_q", "use_subject_re_sign")
            if not getattr(definition, name, False)
        ]
        if missing_blocks:
            raise ValueError(
                f"{prefix}.subject_re_correlation_eta correlates all three subject "
                "random effects, so it requires "
                f"{', '.join(f'{name}=True' for name in missing_blocks)}."
            )

    marginalisation = getattr(definition, "singleton_marginalisation", None)
    if marginalisation is not None and not getattr(definition, "use_subject_re", False):
        raise ValueError(
            f"{prefix}.singleton_marginalisation integrates out child effects, but "
            "the model has no subject random effect to integrate."
        )

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
