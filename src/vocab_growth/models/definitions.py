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
"""Non-English Wordbank languages admitted to the typically-developing pool.

The Down syndrome pool is already a quarter non-English by observation — ``es_01``
(Spanish, 186 children) and ``it_01`` (Italian, 54 children) — while the
typically-developing reference was drawn from English alone. That asymmetry is what
these two languages remove: it is a defensibility argument, not a power argument, and
the added children barely move a contrast that is limited by the Down syndrome sample.

Why these two and not the wider Romance set:

- **Italian** is the only pairing that is same-language *and* same-instrument on both
  sides. ``it_01``'s two form ceilings, 408 and 670, are exactly Wordbank's Italian
  Words & Gestures and Words & Sentences item counts, so the Italian Down syndrome
  and typically-developing data are the same instrument. It is a norming sample, and
  its comprehension reaches 24 months.
- **Spanish (European)** matches ``es_01``'s language but not its instrument
  (``es_01`` uses the 651-item CDI-Down; Wordbank's Spanish forms are 309 and 594).
  It is a norming sample with clean comprehension. Note that ``es_01`` also carries
  its own 186 mental-age and sex matched typically-developing children on the *same*
  CDI-Down, which remain the better Spanish comparison for a matched analysis.

Excluded, with reasons:

- **French (French)** fails on two counts. Its Words & Gestures form carries 713 word
  items where every other Words & Gestures adaptation in Wordbank has 309-457, so it
  is a Words & Sentences-sized inventory administered at 8-16 months; and on rows with
  comprehension >= 20 (excluding all-zero rows, so this is not a low-count
  coincidence) **20.9% record comprehension exactly equal to production** — the same
  proxy-defect signature that retired VG06. The four admitted or considered Romance
  Words & Gestures forms sit at 0.000 by comparison, cleaner than English (American)
  Words & Gestures at 0.001.
- **Catalan** and **Portuguese (European)** are clean but are neither norming samples
  nor matched to any Down syndrome study, and Portuguese would have contributed 45%
  of the added observations on its own.

Two measurement checks were run before admitting these, both reported in
``notes/202608031500-td-romance-extension.md``. Ceiling exposure is no worse than the
existing pool: the Italian and Spanish Words & Gestures forms sit within 90% of their
own ceiling less often (1.5% and 2.9%) than English Words & Gestures (3.1%) or the
Oxford CDI (8.0%), and no row exceeds its own ceiling. And the fixed 810-item
denominator survives — across the 8-15 month window these forms share, language
medians align **better** on raw counts than on proportion-of-own-form (mean
coefficient of variation 0.206 against 0.244, raw tighter at 7 of 8 ages), which is
what the nesting argument predicts and the first test of that assumption across
languages rather than within English.
"""

ENGLISH_AND_ROMANCE_LANGUAGES = ENGLISH_LANGUAGES + ROMANCE_LANGUAGES
"""Widened typically-developing scope: English plus :data:`ROMANCE_LANGUAGES`.

Used by the hierarchical typically-developing models (VG11, VG12, VG13), whose
dataset random intercepts can absorb between-language variation. **Not** used by
VG03/VG04: those carry no random effects, so the between-language spread — about
±20% at matched age, with 15-month medians running 108 to 159 across candidate
languages — would be absorbed by the Beta-Binomial dispersion instead and reported as
child-level dispersion. They stay English-only as the simple baselines they are
documented to be.

Language is very nearly collinear with dataset here (each added language contributes
one dataset per form: Italian WG = Caselli, Italian WS = CLEX, Spanish = Karousou),
so a language effect cannot be separated from a sample effect and will be estimated
as between-dataset heterogeneity. Report it as such. One consequence to watch: the
``CLEX`` dataset label spans several languages in Wordbank (it supplies Italian here,
but also Croatian, Danish, Russian, Swedish and Turkish). Only its Italian rows enter
the pool today, so the study label is unambiguous — but admitting a further CLEX
language would silently pool two languages under one study intercept.
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
    """Return ``(clamp_understood, clamp_q)`` for a ``clamp_mean_above_hi_anchor``.

    The flag started as a single boolean applied to both means. Measurement on
    2026-08-14 showed the saturation it was added for is ``q``'s alone:
    extrapolating VG10's own fitted anchors past the clamp gives ``q`` 0.996 at
    115 months with ``P(mean > 0.99) = 0.999``, while the understood mean reaches
    0.962 and **never** crosses 0.99 in any posterior draw. The unclamped DS
    controls agree — VG01 and VG02 share this pool and these anchors with no
    clamp, and neither saturates nor shows a corner at 84 months.

    Because spoken is ``p_U(a) * q(a)``, clamping both means compounds: the
    spoken trajectory inherits two levelled-off factors, which is what makes the
    corner at 84 months so much sharper than either factor alone.

    ``"q_only"`` is spelled as a string rather than added as a second boolean
    field on purpose. ``fit_manifest.json`` fingerprints the definition with
    ``asdict`` and compares it as whole-object equality, so a *new* field would
    add a key to all fifteen models' definitions and invalidate every model of
    record at once. Widening this field's domain leaves ``True``/``False``
    serialising exactly as before, so only a definition that actually opts in
    changes. See ``notes/202608141200-clamp-q-only.md``.
    """
    if value == CLAMP_Q_ONLY:
        return False, True
    return bool(value), bool(value)


# ============================================================
# Shared prior defaults (same default kappa shape reused by every model)
# ============================================================


@dataclass
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


@dataclass
class KappaAnchorPriorParams:
    """Two-anchor dispersion prior: `kappa` pinned at two reference ages.

    Same curve as :class:`KappaPriorParams` — an asymptote plus an exponential
    age term, ``kappa(z) = kappa_min + exp(a_kappa + b_kappa * z)`` — but
    ``(a_kappa, b_kappa)`` are *derived* from priors on the age term at two
    reference **ages in months** instead of being given priors of their own. This
    is the same move the mean trajectory already makes through ``slope_anchors``.

    Four things follow, and all four are the reason for the change (see
    notes/202608020829-kappa-and-eta-q-prior-recalibration.md, sections 8 and 17):

    * **The priors are checkable.** ``kappa_min + excess_young`` is total `kappa`
      at ``anchor_ages[0]``, directly comparable with a per-age Beta-Binomial fit
      to the analysis frame. ``a_kappa`` is not: it is the age term at ``z = 0``,
      i.e. at whatever the *pool mean age* happens to be.
    * **They do not move when the pool does.** ``a_kappa`` is defined at the pool
      mean, so resampling or a study filter silently changes what its prior means.
      Ages do not move.
    * **The tails are interpolated, not extrapolated.** Both anchors sit inside
      the data, so the prior on `kappa` between them is a blend of two checked
      values rather than an intercept and a slope whose tails compound as
      ``exp(2b)`` at the ends of the range.
    * **The sign is free.** ``b_kappa_mag >= 0`` forces `kappa` to fall with age,
      which the typically-developing comprehension data reject. Two free anchors
      admit either direction.

    Place the anchors where the age term is about an order of magnitude above the
    floor and where it has fallen back to it: between them the exponential carries
    the curve, outside them the floor does, so both priors sit where the data
    identify them.
    """

    anchor_ages: tuple[float, float]
    """Reference ages (months), ordered (young, old), for the two kappa anchors."""
    kappa_min_mu: float
    """LogNormal mu for kappa_min, the dispersion asymptote.

    Which end of the age range it applies at follows the sign of the derived
    ``b_kappa``, so it is a floor only when dispersion falls with age. Where it
    rises — the typically-developing comprehension models — it is the young-age
    asymptote instead, and carries real weight: VG13's is 30, not 3.
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


@dataclass
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
class AgeVaryingSubjectScale:
    """Proposal A1: an age-varying between-child scale, with ``kappa`` held flat.

    Supplied **in place of** a scalar ``tau_subj_*_sigma`` / ``tau_subject_sigma``.
    That is deliberate: the field it replaces already selects the subject-effect
    scale, so A1 needs no new definition field, and the fifteen models of record
    keep their fingerprints (a new field would appear in every ``asdict`` and
    invalidate every fit). The same trick carries
    :data:`CLAMP_Q_ONLY` on ``clamp_mean_above_hi_anchor``.

    The scale is log-linear in standardised age between two reference ages, and
    is parameterised so the model of record is **nested at zero**::

        tau_young       ~ HalfNormal(young_sigma)      # the record's own prior
        log_tau_ratio   ~ Normal(0, log_ratio_sigma)   # log(tau_old / tau_young)
        tau(z)          = tau_young * exp(log_tau_ratio * (z - z_young) / (z_old - z_young))

    Two properties follow, and both are the point of the design. At the young
    anchor the prior on the subject scale is *exactly* the prior the model of
    record places on its constant ``tau``, so the variant is one-factor. And
    ``log_tau_ratio = 0`` reproduces the model of record exactly, so the
    posterior for that one parameter answers "does the spread between children
    widen with age?" as a credible interval around a null, rather than as a
    model comparison. A multiplicative ratio also keeps the scale positive
    without taking a logarithm of a ``HalfNormal`` that can approach zero.

    ``hold_kappa_constant`` is part of A1 rather than a separate switch: the
    proposal is that the age variation belongs on the between-child parameter,
    so it is *moved* rather than duplicated. It forces ``b_kappa = 0`` in the
    paired dispersion block — ``kappa_u`` for the understood scale, ``kappa_s``
    for the production one — leaving the dispersion *level* free.

    Structural caveat, stated where it cannot be missed: scaling a single
    per-child deviate by ``tau(age)`` imposes **perfect rank correlation of
    children across age**. Children never cross. That is measured, not assumed
    — about 0.75-0.83 disattenuated out to two years and 0.28 beyond — so this
    is registered-sensitivity material and not a candidate model of record. See
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

    The one place the overloaded field is interpreted. Engines call this rather
    than testing types inline, so "is this model A1?" has a single answer.
    """
    return value if isinstance(value, AgeVaryingSubjectScale) else None


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
    """Dispersion priors, in either parameterisation; only the univariate spoken
    models have migrated to the two-anchor form so far."""

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
    tau_subject_sigma: float = 1.5
    """HalfNormal scale for the subject intercept SD (logit scale).

    Calibrated, 1.5 rather than the 0.5 every model carried until section 23 of
    ``notes/202608020829-kappa-and-eta-q-prior-recalibration.md``. The conditional
    dispersion estimator reports `tau` alongside `kappa` for every pool, so this
    scale has had a calibration available since section 19 and was simply never
    read off it. It puts the subject scale between 0.74 and 1.15 across the
    family, against a HalfNormal(0.5) whose median is 0.34 — which left all
    fourteen subject-scale parameters in the registry at prior CDF 0.86 to 0.994.
    HalfNormal(1.5) has median 1.01 and lands every one of them between 0.38 and
    0.64 while keeping the mass near zero that lets a subject effect the data do
    not support shrink away. The *study* scales stay at 0.5: their posteriors sit
    at prior CDF 0.43-0.82 already and need nothing."""
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


@dataclass
class UnivariateREModelDefinition(UnivariateModelDefinition):
    """A univariate model with random effects, plus the two sampling-geometry options.

    These live on a subclass rather than on
    :class:`UnivariateModelDefinition` for a concrete reason. A fit is validated
    against the current registered definition by comparing
    ``dataclasses.asdict`` field for field, so **adding a field to a definition
    class invalidates every existing fit of that class** — including models that
    never set it. VG01-VG04 are plain univariate models with no random effects at
    all; putting these two fields on the shared base would have made four
    published models of record stale for no modelling reason, VG03 alone costing a
    2h50m refit. Only VG11, VG12 and the exploratory VG17 use the random-effect
    engine, so only they carry the fields.

    The engine reads both through ``getattr`` with a default, so a plain
    :class:`UnivariateModelDefinition` still builds — VG17 derives its definition
    from VG01 and never becomes a subclass instance.
    """

    centred_study_re: bool = False
    """If True, sample the study intercepts directly as
    ``ZeroSumNormal(sigma=tau * sqrt(K/(K-1)))`` rather than as ``tau`` times a
    unit-scale ``ZeroSumNormal``.

    Prior-preserving: scaling a zero-sum Gaussian's sigma and scaling its variate
    give the same distribution, so this changes the sampler's coordinates and
    nothing else. The non-centred form of issue #65 is the wrong side of the
    funnel trade-off once each study carries thousands of observations.

    Measured on VG12 at ``test``: ``tau`` ESS 310 -> 6,950, max R-hat 1.0133 ->
    1.0057, divergences 59 -> 31. Energy BFMI unchanged (0.203 -> 0.194), exactly
    as predicted — ``tau`` ranks 13th on energy correlation at -0.023, so this was
    never a BFMI fix. See ``notes/202608050900-td-hierarchical-geometry.md``
    §§2-3."""
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
    eta_q_sigma: float = 0.8  # widened 2026-08-04 from 0.20, itself tightened from 0.4 to curb the q-GP<->slope_q/intercept_q competition (VG09-note Option B). That tightening was mis-scoped: every DS joint model sits at prior CDF 0.95-0.99 with contraction 0.03-0.16 whether or not it has subject REs on q or the Option D anchoring, because logit(q) is S-shaped across 8-115 mo and only the GP can supply that. Short-window VG13 does not press it and keeps 0.20. See notes/202608041730-ds-spoken-q-trajectory-prior.md
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
    tau_subj_u_sigma: float = 1.5
    """HalfNormal scale for subject intercept SD on understood (logit scale).
    Estimated at 0.85 on the Down syndrome joint frame and 0.74-0.77 on the
    typically-developing ones."""
    use_subject_re_q: bool = False
    """If True, add subject-level random intercepts on the production ratio q."""
    tau_subj_q_sigma: float = 1.5
    """HalfNormal scale for subject intercept SD on q (logit scale). Estimated at
    1.15 on the Down syndrome joint frame and 1.12 on VG13's."""
    one_observation_per_subject: bool = False
    """If True, retain one reproducibly sampled administration per subject. This
    provides a cheap sensitivity analysis for repeated-measures dependence."""
    td_languages: tuple[str, ...] = ENGLISH_LANGUAGES
    """Wordbank ``language`` values the typically-developing pool draws on.

    Ignored for the Down syndrome population, whose language scope is fixed when the
    database is built. Defaults to :data:`ENGLISH_LANGUAGES`; the hierarchical
    typically-developing models set :data:`ENGLISH_AND_ROMANCE_LANGUAGES`. Changing
    this changes the data the model sees, so it is part of the model graph and a
    change requires a refit."""

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

    # -- Mean extrapolation above the high anchor --
    clamp_mean_above_hi_anchor: bool | str = False
    """Level the mean off above the high slope anchor instead of extrapolating.

    ``True`` clamps both the understood mean and ``q``; ``CLAMP_Q_ONLY``
    (``"q_only"``) clamps only ``q``. Resolve with :func:`clamp_targets`
    rather than testing truthiness -- ``"q_only"`` is truthy."""
    """If True, level the logit-linear mean off above the high anchor age instead
    of extrapolating the line. The transition is a soft minimum, so the mean stays
    differentiable and the fitted curve inherits no elbow; a hard ``min`` made the
    VG10 spoken curve briefly non-monotone at the anchor. One-sided: below the low
    anchor the line still extrapolates, which is accurate there. Applied to the
    Down syndrome models, whose GP domain runs to 115 months against a high anchor
    at 84 — see ``gp_utils.trend_and_gp`` and
    notes/202608042030-q-mean-extrapolation.md."""

    # -- Reporting range --
    report_max_age_understood: int | None = None
    """Highest query age (months) at which comprehension quantities are reported.

    Trims the understood and ``q`` summary tables and the production-ratio figure
    to where their evidence stops, leaving spoken on the full grid. Purely
    post-processing: the query grid, the model graph and the fitted trace are
    untouched, so changing this cannot move the posterior — proved by refitting
    VG10 across the change at a fixed seed and reproducing its diagnostics
    bit-for-bit. It does still require re-running the fit: the summary tables are
    written during the fit pipeline and ``--render-only`` does not regenerate
    them, and this field is part of the recorded definition, so a fit produced
    under a different value is correctly reported as stale. None reports every
    query age. See ``posterior_analysis.trim_reported_ages``."""

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
    the sufficiency result in notes/202607261540 is the proof that no aggregate
    analysis of these data can test it: a statistic sufficient for ability carries
    no information about item composition. So the assumption is not probed by
    running the model differently -- it is probed by removing the rows that need
    it, which is what this flag does (issue #190).

    It is the widest-scoped sensitivity in the registry, and deliberately so: 278
    of the merged view's 1,521 Down syndrome rows survive, from 194 children
    across ie_01 (its 810 wave only), ie_02, uk_02 (DSE form only) and uk_06,
    spanning 9-115 months. Comprehension is the least affected outcome -- 259 of
    987 understood observations survive, against 264 of 1,428 spoken -- because
    the short forms are production-heavy, so expect the spoken trajectory to move
    more than the understood one.
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

    @property
    def model_type(self) -> ModelType:
        return ModelType.BIVARIATE


@dataclass
class BivariateCorrelatedSubjectREModelDefinition(BivariateModelDefinition):
    """Bivariate definition that also correlates the two subject random effects.

    VG20 (issue #224). The field lives on a **subclass**, not on
    ``BivariateModelDefinition``, for the reason ``UnivariateREModelDefinition``
    exists: a fit is validated by comparing the serialised definition field for
    field, so adding a field to a definition class invalidates every existing fit
    of that class — here VG05, VG07-VG10 and VG16, six models of record.

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
    eta_q_sigma: float = 0.8  # widened 2026-08-04 from 0.20, itself tightened from 0.4 to curb the q-GP<->slope_q/intercept_q competition (VG09-note Option B). That tightening was mis-scoped: every DS joint model sits at prior CDF 0.95-0.99 with contraction 0.03-0.16 whether or not it has subject REs on q or the Option D anchoring, because logit(q) is S-shaped across 8-115 mo and only the GP can supply that. Short-window VG13 does not press it and keeps 0.20. See notes/202608041730-ds-spoken-q-trajectory-prior.md
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

    # -- Data age filtering --
    max_age_months: int | None = None
    """Upper bound on age (inclusive, months) for data loading. None = no limit."""

    # -- Mean extrapolation above the high anchor --
    clamp_mean_above_hi_anchor: bool | str = False
    """Level the mean off above the high slope anchor instead of extrapolating.

    ``True`` clamps both the understood mean and ``q``; ``CLAMP_Q_ONLY``
    (``"q_only"``) clamps only ``q``. Resolve with :func:`clamp_targets`
    rather than testing truthiness -- ``"q_only"`` is truthy."""
    """If True, level the logit-linear mean off above the high anchor age instead
    of extrapolating the line. The transition is a soft minimum, so the mean stays
    differentiable and the fitted curve inherits no elbow; a hard ``min`` made the
    VG10 spoken curve briefly non-monotone at the anchor. One-sided: below the low
    anchor the line still extrapolates, which is accurate there. Applied to the
    Down syndrome models, whose GP domain runs to 115 months against a high anchor
    at 84 — see ``gp_utils.trend_and_gp`` and
    notes/202608042030-q-mean-extrapolation.md."""

    # -- Reporting range --
    report_max_age_understood: int | None = None
    """Highest query age (months) at which comprehension quantities are reported.

    Trims the understood and ``q`` summary tables and the production-ratio figure
    to where their evidence stops, leaving spoken on the full grid. Signed has its
    own cap, ``report_max_age_signed``; before 2026-08-13 it did not, and the
    sign-derived figures silently borrowed this one.
    Purely post-processing: the query grid, the model graph and the fitted trace
    are untouched, so changing this cannot move the posterior — proved by
    refitting VG10 across the change at a fixed seed and reproducing its
    diagnostics bit-for-bit. It does still require re-running the fit: the
    summary tables are written during the fit pipeline and ``--render-only`` does
    not regenerate them, and this field is part of the recorded definition, so a
    fit produced under a different value is correctly reported as stale. None
    reports every query age. See ``posterior_analysis.trim_reported_ages``."""

    report_max_age_signed: int | None = None
    """Highest query age (months) at which signed quantities are reported.

    The trivariate counterpart of ``JointModelDefinition.report_max_age_signed``,
    added 2026-08-13. The cap was introduced for VG15 alone (``feat(vg15)``), so
    VG14 -- the only trivariate model, and the one whose signed results uk_07
    moved most -- never had one. Its two consequences were wrong in the same
    direction:

    * ``plot_signed_rate`` and ``plot_sign_speech_crossover`` were passed
      ``report_max_age_understood``, so the signed figures were trimmed by the
      *comprehension* cap. Raising that cap from 72 to 84 moved VG14's signed
      figures as a side effect, which nobody decided.
    * the ``r(a)`` summary table was not trimmed at all, so it ran to the top of
      the query grid at 90 while the figure beside it stopped at 72 -- the
      table/figure disagreement ``tests/test_reporting_age_caps.py`` exists to
      catch, inverted.

    Set to 84 on VG14, matching VG15 on the same evidence: uk_07 rebuilt the
    signed tail, the study owner raised VG15's cap from 60 to 84 in #212, and 84
    is the high trend anchor above which the mean is levelled off rather than
    fitted. Purely post-processing, but part of the recorded definition, so
    changing it requires a refit. None reports every query age."""

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

    # -- Understood (U) slope priors (matching VG05 and the rest of the DS joint
    # family, including the 2026-08-04 anchor recalibration: see VG05 and
    # notes/202608041216-ds-understood-trajectory-prior.md). VG15 is the only
    # model built from this dataclass, so these defaults are its anchor priors. --
    p_slope_low_u_alpha: float = 1.5
    p_slope_low_u_beta: float = 8.0
    p_slope_hi_u_alpha: float = 3.0
    p_slope_hi_u_beta: float = 1.3

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
    sign_peak_prior: tuple[float, float] | None = None
    """Beta(alpha, beta) on the signed peak's POSITION between the outer sign
    anchors, or None to fix it at ``sign_anchor_ages[1]``.

    Adopted for VG15 on 2026-08-06. With the peak fixed, `r(a)` peaked at the
    middle anchor by construction -- 77% of posterior draws within a month of it --
    so its height was estimated and its age simply asserted. "Signing peaks around
    three years" was a statement about knot placement, not a finding.

    Sampling the position rather than the age keeps ``z_low < z_mid < z_hi`` true by
    construction, which a prior on the age could not. Measured on VG15 at `test`:
    the peak age is identifiable (contraction 0.481) at 29.4 months, 89% ETI
    [23.9, 46.2], against a prior interval of [21.5, 67.5] -- and the free knot
    samples better than the fixed one (0 divergences against 2), despite making the
    GP's nuisance basis draw-dependent.

    The peak HEIGHT is unmoved (0.319 -> 0.314), so this changes shape, not level.

    Note this does not extend to VG14, whose lack of study random effects means its
    age curve must absorb between-study composition -- the reason
    notes/202606151700 found the peak age unidentifiable there. That finding stands
    for the model it was made about. See
    notes/202608060900-three-prior-conflicts.md section 5.
    """
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
    eta_q_sigma: float = 0.8  # widened 2026-08-04 from 0.20, itself tightened from 0.4 to curb the q-GP<->slope_q/intercept_q competition (VG09-note Option B). That tightening was mis-scoped: every DS joint model sits at prior CDF 0.95-0.99 with contraction 0.03-0.16 whether or not it has subject REs on q or the Option D anchoring, because logit(q) is S-shaped across 8-115 mo and only the GP can supply that. Short-window VG13 does not press it and keeps 0.20. See notes/202608041730-ds-spoken-q-trajectory-prior.md
    # `ell_unit_sign` is unidentified in VG15 (contraction 0.033) and is
    # DELIBERATELY left sampled, settled 2026-08-06. Fixing it at its prior median
    # changes nothing measurable -- a maximum median shift of 0.0023 on r(a), +0.1%
    # band width, convergence unchanged -- and removing the signed GP is worse: it
    # fails the hard convergence tier and narrows the band 63% at 96 months,
    # stripping the model's only honest signal of ignorance where signed data have
    # run out. `sign_gp_mode` in the joint engine can express both alternatives;
    # neither is an improvement. See notes/202608060900 section 5b.
    ell_unit_sign_alpha: float = 2.0
    ell_unit_sign_beta: float = 5.0
    eta_sign_sigma: float = 0.4  # reverted to standard (matches VG14): the three-anchor mean now carries the hump, so the GP only models smooth departures
    ell_months_range: tuple[int, int] = (6, 18)
    n_plot: int = 500
    kappa_u: KappaPriorParams = field(default_factory=KappaPriorParams)
    kappa_s: KappaPriorParams = field(default_factory=KappaPriorParams)
    # `kappa_sign` deliberately stays on the legacy dispersion form for VG15,
    # settled 2026-08-06. Unlike VG05/VG07/VG08/VG14 -- migrated because their
    # `b_kappa_mag_s` sat about four standard deviations beyond its prior with the
    # posterior wider than it -- the signed block is well identified (contraction
    # 0.429) at prior CDF 0.276, and the form's non-increasing-with-age constraint
    # is not binding. The asymmetry with VG14 is two separate correct calls, not an
    # inconsistency. See notes/202608060900-three-prior-conflicts.md section 5b.
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
    tau_psi_sigma: float = 1.0
    """HalfNormal sigma for the between-study SD of log psi.

    Wider than the other three (0.5) because the measured heterogeneity is wider.
    Mantel-Haenszel odds ratios over the cross-tab sources run from 0.90 (es_01) to
    about 14 (uk_07, nz_01) — roughly 2.8 on the log scale, so a between-study SD
    near 1 is what the data show, and a HalfNormal(0.5) would fight it.

    **Why this is a study term and not an age term.** The obvious alternative --
    that the association varies with age, and the sources differ only because they
    cover different ages -- was tested and rejected. uk_07's psi-informing rows have
    a median age of 60 months against 38 for uk_02 and 32 for es_01, and fitted
    without a study term age looks strong (+0.381 per year, z = +5.78). But age
    alone fits worse than study alone (weighted SSR 2688 against 1640) and adds
    nothing on top of it (1640 -> 1640); and in the 34-56 month window where all
    three within-understood sources overlap, es_01 and uk_02 are matched at exactly
    41 months and still differ six-fold (1.00 against 5.94, with uk_07 at 11.89).
    Coverage would not have supported an age term regardless: three-way overlap
    exists only at 30-60 months, so any curvature in the tails would be one study's
    intercept re-labelled as a trend. See
    notes/202608121030-psi-heterogeneity-and-age-invariance.md and
    scripts/psi_heterogeneity_audit.py.

    Only four studies inform psi, so ``tau_psi`` is weakly identified and the prior
    does real work. That is a reason to report it with its interval and treat the
    per-study values as the primary read, not a reason to pool them: pooling does
    not make the heterogeneity go away, it hides it in a headline that then moves
    with source composition (psi went 1.80 to 2.49 on adding uk_07 alone). The
    non-centred ``tau * z`` parameterisation keeps the funnel manageable at this
    group count."""

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
    """If True, add subject-level random intercepts on the sign ratio r. Signing is
    the sparsest modality, so this is gated: inspect tau_subj_sign and fall back to
    study-RE-only on r (set False) if it pins near its prior with poor diagnostics."""
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
    """Level the mean off above the high slope anchor instead of extrapolating.

    ``True`` clamps both the understood mean and ``q``; ``CLAMP_Q_ONLY``
    (``"q_only"``) clamps only ``q``. Resolve with :func:`clamp_targets`
    rather than testing truthiness -- ``"q_only"`` is truthy."""
    """If True, level the logit-linear mean off above the high anchor age instead
    of extrapolating the line. The transition is a soft minimum, so the mean stays
    differentiable and the fitted curve inherits no elbow; a hard ``min`` made the
    VG10 spoken curve briefly non-monotone at the anchor. One-sided: below the low
    anchor the line still extrapolates, which is accurate there. Applied to the
    Down syndrome models, whose GP domain runs to 115 months against a high anchor
    at 84 — see ``gp_utils.trend_and_gp`` and
    notes/202608042030-q-mean-extrapolation.md."""

    # -- Reporting range --
    report_max_age_understood: int | None = None
    """Highest query age (months) at which comprehension quantities are reported.

    Trims the understood and ``q`` summary tables and the production-ratio figure
    to where their evidence stops, leaving spoken (and signed) on the full grid.
    Purely post-processing: the query grid, the model graph and the fitted trace
    are untouched, so changing this cannot move the posterior — proved by
    refitting VG10 across the change at a fixed seed and reproducing its
    diagnostics bit-for-bit. It does still require re-running the fit: the
    summary tables are written during the fit pipeline and ``--render-only`` does
    not regenerate them, and this field is part of the recorded definition, so a
    fit produced under a different value is correctly reported as stale. None
    reports every query age. See ``posterior_analysis.trim_reported_ages``."""

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
    """Restrict the pool to administrations recorded natively on the 810 reference.

    The models score every count against ``n_trials = 810``, so a 416-item Oxford
    CDI count enters on the same denominator as an 810-item DSE Checklists count.
    That harmonisation assumes the shorter form's items are the easier ones, and
    the sufficiency result in notes/202607261540 is the proof that no aggregate
    analysis of these data can test it: a statistic sufficient for ability carries
    no information about item composition. So the assumption is not probed by
    running the model differently -- it is probed by removing the rows that need
    it, which is what this flag does (issue #190).

    It is the widest-scoped sensitivity in the registry, and deliberately so: 278
    of the merged view's 1,521 Down syndrome rows survive, from ie_01 (its 810
    wave only), ie_02, uk_02 (DSE form only) and uk_06.

    The cost is legible in what leaves. es_01, nz_01 and uk_07 are all on shorter
    forms, so the only cross-tab source left is uk_02's DSE form -- all 56 of its
    four-cell rows are that form -- and ``psi`` falls back to its single-study
    branch with no ``delta_psi`` or ``tau_psi`` to estimate. The variant therefore
    answers the denominator question and the between-study question together and
    cannot separate them. Read it against the model of record for the trajectory
    shapes, not for ``psi``.
    """
    report_max_age_signed: int | None = None
    """Highest query age (months) at which signed quantities are reported.

    The signed counterpart of ``report_max_age_understood``, and it exists for the
    same reason: a model's ``ages_query`` grid is shared by every outcome, but the
    outcomes are not observed over the same range. Signed is still the sparsest
    outcome, and the Down syndrome grid runs to 115 months, so a cap is still
    needed -- but **where** it belongs changed when uk_07 (PACT-DS) entered the
    pool on 2026-08-12.

    The cap was 60. Its justification was a count: of 593 signed observations, 46
    were above 60 months and **none above 72** -- so beyond about 60 months
    ``r(a)`` was the tent's extrapolation rather than an estimate. Two changes on
    2026-08-12 rebuilt that tail. uk_07 contributes 82 signed observations spanning
    34-95 months from 30 children, and uk_06's 11 observations at 60-115 months
    were unmasked once the source confirmed its signing field is a total (see
    ``data_utils.UNCERTAIN_SIGN_STUDIES``). The count now reads:

    ======================  ==========  =========
    band                    before      after
    ======================  ==========  =========
    60-72 months            46          69
    72-84 months            **0**       17
    84-96 months            **0**       8
    96-120 months           **0**       4
    above 60 months, total  46          98
    above 72 months, total  **0**       29
    ======================  ==========  =========

    So the cap is 84: the 72-84 band now carries 17 observations from two
    independent sources rather than nothing, and reporting it is no longer
    extrapolation. Above 84 is left out deliberately -- 12 observations from 10
    children, thinning to one source per band, is evidence but not enough to
    publish a curve on, and 84 is also the Down syndrome models' high trend anchor
    (see ``clamp_above_hi``), above which the mean is clamped rather than fitted.
    Those records still inform the fit; they are only withheld from the tables.

    This is not a cosmetic widening. At 72 months the refit puts ``r`` at 0.198
    against 0.156 before, and at 90 months 0.182 against 0.098 -- the post-peak
    decline is real but far shallower than the pre-uk_07 fits showed, and at 60
    months that finding was entirely hidden. See
    notes/202608120030-uk07-pactds-integration-and-ds-refit.md §5.

    Applies to the signed counts, the signed ratio `r`, and total expressive
    `p_any`, which is a function of the signed ratio and can only be reported where
    signed evidence reaches.

    Post-processing only, so it cannot move a posterior -- but it is part of the
    recorded definition and the tables are written during the fit, so a change
    needs a refit and `--render-only` will not pick it up. Same caveats as the
    comprehension cap; see ``posterior_analysis.trim_reported_ages``.
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
    """Whether es_01's within-understood four-cell cross-tab enters the
    Dirichlet-Multinomial that identifies psi. **Default True** since 2026-08-12,
    when psi gained a study-level term.

    es_01's cells are derivable — its fourth column is a recorded union. The
    original table's columns are TOTAL COMPREHENSIÓN, TOTAL PRODUCTION, TOTAL
    GESTURES and WORD PRODUCED + GESTURES ONLY, the last being what Galeote et al.
    (2011) call "total lexical production combining the two modalities" — so the
    four cells follow by subtraction and 185 of 186 rows yield a valid partition at
    11-71 months. They more than double the rows identifying psi and anchor it at
    the young end, where uk_02 is otherwise alone.

    It defaulted False for the nine days before that, and the reason was never the
    construct. es_01's third column scores "gestures representing specific lexical
    items", each tied to one of the 651 checklist words — a per-word lexical marker
    on an adapted CDI, structurally the same coding uk_02 and uk_07 use. It is the
    same measurement.

    The reason was that the sources already informing ``psi`` disagree about it
    substantially, and ``psi`` had nowhere to put that. Mantel-Haenszel odds ratios
    over the same cells, stratified by child:

    =======  =====  =========  =================  ==========  ===============
    source   rows   MH OR      reference set      per-child   non-vocal words
                                                  OR < 1      also spoken
    =======  =====  =========  =================  ==========  ===============
    uk_02      56     6.09     within understood     4%          50.4%
    uk_07      82    13.90     within understood    11%          72.2%
    nz_01     111    14.63     all 675 items         4%          44.8%
    es_01     185     0.90     within understood    45%          30.8%
    =======  =====  =========  =================  ==========  ===============

    Two caveats on that table. MH is a crude descriptive statistic on the observed
    cells, not ``psi`` itself, which is a population-conditioned quantity defined
    against the fitted r and q. And nz_01 has no comprehension total, so its
    "neither" cell spans all unproduced items rather than understood-but-unproduced,
    which inflates its OR — on the same data uk_07 reads 13.90 within understood and
    40.72 over all 674 items. Magnitudes are therefore only comparable within a
    reference set. The per-child sign and the share-also-spoken column need no
    "neither" cell and are comparable throughout.

    What survives every control is that es_01 sits at independence while the three
    sign sources are positive: by age band it runs 0.30-1.12 against 4.4-41.6 for
    uk_02 and 4.4-18.1 for uk_07, with no overlap in any band, and matched on
    expressive vocabulary (30-300 words) it is 1.05 against 4.80 and 9.68. On the
    conditioning-free share-also-spoken measure it is the low end of a continuous
    gradient rather than categorically apart. Either way the spread across sources
    is large and its cause is not identified. Signing instruction is the obvious
    candidate, but no source records whether its children were taught to sign, so
    that contrast is background assumption rather than measurement; and uk_07, the
    one source with experimental variation in instruction, carries the *higher*
    association in its control arm (17.93 against 11.65), which cuts against the
    mechanism rather than for it. Treat the spread as unexplained.

    That heterogeneity was disqualifying only because **``psi`` was the only latent
    in this model with no study-level term.** ``delta_u``, ``delta_q`` and ``delta_sign``
    are all study random intercepts; ``log_psi`` was a bare global scalar. So a pooled
    ``psi`` was a precision-weighted average over whichever sources happened to be in
    the pool — which is why it moved from 1.80 to 2.49 when uk_07 arrived, and why
    adding es_01's 185 rows (more than the uk_02 and uk_07 four-cell rows combined)
    would have dragged the headline toward independence as an artefact of composition
    rather than a finding.

    That is now handled: ``delta_psi`` is a zero-sum study random intercept over the
    psi-informed studies, so each source carries its own association and the reported
    population value is a shrunk centre with ``tau_psi`` quantifying the spread. With
    the heterogeneity modelled rather than averaged away, pooling these cells adds
    evidence instead of moving the headline by composition, and the flag defaults
    True. Setting it False isolates es_01's contribution. See
    data/vocab_data_es_01.md and
    notes/202608120030-uk07-pactds-integration-and-ds-refit.md."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.JOINT


# ============================================================
# Model instances
# ============================================================

_DS_GP_DOMAIN_MONTHS = (8, 115)
_TD_GP_DOMAIN_MONTHS = (8, 30)
_YOUNG_TD_GP_DOMAIN_MONTHS = (8, 18)


# ------------------------------------------------------------------
# Production-outcome dispersion, two-anchor form (2026-08-02)
# ------------------------------------------------------------------
# The spoken models were recalibrated in the legacy (a_kappa, b_kappa_mag)
# parameterisation earlier the same day and then reparameterised, because
# recalibrating that form could not be finished: no setting of its three
# parameters both admits the slope the data want and keeps young-age dispersion
# plausible, since the intercept and slope tails compound as exp(2b) at the ends
# of the age range. See notes/202608020829-kappa-and-eta-q-prior-recalibration.md
# (sections 8, 17 and 18) for the full argument and the audit trail.
#
# Both blocks below are centred on a three-parameter fit of the model's own
# dispersion curve, kappa(z) = kappa_min + exp(a - b z), to a *saturated* mean —
# a free proportion per integer-age cell, every cell with at least 15
# observations — so the dispersion estimate is not contaminated by a choice of
# mean model:
#
#   pool             n      cells  kappa_min      b     kappa at the two anchors
#   DS spoken     1,114       25       3.54    2.78     49.1 @ 18 mo, 7.7 @ 36 mo
#   TD spoken     4,075       23       3.08    1.78     37.1 @ 12 mo, 6.2 @ 20 mo   (VG03 frame)
#   TD spoken    16,235       23       3.08    1.50     29.9 @ 12 mo, 6.6 @ 20 mo   (VG11 frame)
#
# Anchors go where the age term is roughly an order of magnitude above the floor
# and where it has fallen back to it: between them the exponential carries the
# curve and outside them the floor does, so both priors sit where the data can
# identify them. The excess medians below are those fitted totals minus the
# floor, rounded.
#
# sigma 0.7 throughout (a 5-95% range of about +/- 3.2x), set so each anchor's
# range covers the spread of defensible estimates for it — the two TD frames'
# fitted values, and the per-age cells on either side of the anchor age, which
# scatter more than the smooth fit does. At the typically-developing young anchor
# the 11/12/13-month cells give total kappa of 20.3, 89.4 and 37.0, on profile
# intervals that do not overlap ([12.2, 31.2] on 86 administrations, [65.6,
# 119.1] on 162, [29.4, 45.7] on 271): the scatter is real between-study
# composition rather than noise, and no smooth curve passes through all three.
# sigma 0.6 would have put the high cell at the 96th prior percentile; 0.7 covers
# it. Erring wide is deliberate — the failure this replaces was a prior too tight
# to let the data speak (contraction 0.82, prior CDF 0.93-1.00).
#
# kappa_min is carried over unchanged (LogNormal(log 3, 0.8), median 3, 5-95%
# [0.80, 11.2]) so this is a single-factor change against the recalibrated legacy
# fits. Three independent pools put the floor at 3.08-3.54. Note that the
# anchored form leans on it harder — beyond the old anchor the floor alone sets
# the level, where before the exponential term propped it up — so its 8% of prior
# mass below kappa = 1 now shows at old ages, and tightening kappa_min_sigma is a
# candidate follow-up rather than something folded into this change.
#
# One prior per population, not one shared block: b_kappa_mag is a slope per unit
# *standardised* age, so a single prior on it is about 3.5x tighter on the DS
# pool (sd 20.8 months) than on the TD pool (sd 5.9 months) in per-month terms —
# the units problem recorded in section 15. Anchors stated in months are immune
# to it, and to the pool's age distribution moving under a resample or a study
# filter.
#
# The blocks in this section are calibrated *marginally* and so belong only to
# models with no grouping structure — VG01, VG02, VG03 and VG04, all of which run
# on the plain univariate engine and give `kappa` every source of spread to carry.
# Everything with study and subject random intercepts is calibrated conditionally
# instead; see the next section.
#
# The two comprehension blocks were added later, from the same estimator run with
# its subject and study effects switched off (scripts/kappa_conditional_calibration.py
# records which effects each pool's model has and mirrors them). Both are stable
# across every mean model tried — VG02 gives 14.8-15.4 at 18 months and 7.1-7.2 at
# 36, VG04 11.6-11.8 at 12 months and 11.1-11.4 at 18 — so the thinness of the
# Down syndrome comprehension frame (346 usable rows) does not undermine them the
# way it does the conditional fits in the next section. Nothing has to be
# separated from a random effect here, which is what that frame could not support.
#
# Two things about comprehension differ from the spoken blocks above:
#
#   * **VG04's dispersion is flat.** 11.8 at 12 months against 11.3 at 18, and
#     per-age cells scattering 5.8-15.6 with no trend across 8-24 months. Its two
#     anchors are therefore near-equal and the implied slope prior is near
#     symmetric about zero — P(kappa rising) 0.476, against 0.007 for DS spoken.
#     This is the case the legacy b_kappa_mag >= 0 could not represent at all.
#   * **The floor is not identified for either.** VG02's fitted kappa_min ranges
#     over 0.76-6.01 depending on the mean model while its anchor totals move by
#     under 4%, and VG04's curve is flat enough that any (floor, excess) split
#     reproducing the level fits equally well. Both keep the shared weak
#     LogNormal(log 3, 0.8) and let the anchors carry the level — which is the
#     ridge the two-anchor parameterisation exists to sidestep.

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
# A marginal calibration answers "how much do counts vary at this age?". A model
# carrying study and subject random intercepts has already removed most of that
# variation by the time its likelihood runs, so its kappa answers a different
# question — "how much is left once this child's own level is known?" — and the
# marginal number is a lower bound. On VG11 it was out by a factor of ten: the
# prior sat at kappa(12) = 30 while the fit went to 312, at prior CDF 1.000.
#
# scripts/kappa_conditional_calibration.py estimates the right quantity, by
# fitting the same saturated mean with the random effects present and the subject
# effect integrated out:
#
#     logit p_ij = m_c(ij) + s_k(i) + b_i,   b_i ~ N(0, tau^2)
#     y_ij       ~ BetaBinomial(N_ij, p_ij, kappa(a_ij))
#
#   pool                        n     obs/child   tau    kappa at the anchors
#   VG11 spoken            16,235       1.32     1.06    317 @ 12 mo, 50 @ 20 mo
#   VG12 understood         5,997       1.26     0.74     43 @ 12 mo, 66 @ 20 mo
#   VG13 understood         5,406       1.19     0.77     42 @ 12 mo, 124 @ 17 mo
#   VG13 q | understood     5,320       1.19     1.12     36 @ 12 mo, 30 @ 17 mo
#
# Every one is 3-10x its marginal counterpart, and two of them rise with age,
# which the legacy b_kappa <= 0 cannot represent at any setting. The medians below
# are those totals, split into a floor and an excess per anchor.
#
# Three things had to be established before reading a prior off this (section 19
# of the note has the detail, and --recover / --mean-sweep re-run the checks):
#
#  * **tau and kappa are separable here.** For a child measured once both add
#    variance to the same single number, and 84% of VG11's children are measured
#    once; what separates them is the shape of the count distribution each
#    implies, which pins a large tau but not a small one, so the children with a
#    repeat are what make the estimate precise. Simulating from a subject-heavy
#    truth and a dispersion-heavy truth on the real design returns each
#    correctly, an order of magnitude apart.
#  * **The answer does not depend on the mean model.** Saturated, spline and even
#    a linear mean agree to within a few percent on all four pools — so the gap
#    against VG12's and VG13's posteriors (both near 16) is not an artefact of
#    this estimator fitting the age curve more closely than an HSGP does.
#  * **The DS joint frame recovers only a lower bound.** Section 22 replaces
#    section 19's blanket exclusion with a measurement. Holding tau at its fitted
#    value and varying only the truth, the estimator returns kappa *below* it by
#    an amount that grows with the level — -2% at kappa(24) = 12, -4% at 41, -26%
#    at 82, -36% at 163 — because a large kappa is near-binomial and the data stop
#    distinguishing bigger from biggest. The estimates are therefore lower bounds
#    rather than noise, and the block below uses them with the bias measured at
#    the operating point divided back out and a deliberately wide sigma. tau
#    itself recovers to within 6%, which is what section 23 calibrates from.
#
# sigma is 0.7 for the spoken and ratio anchors, as in the marginal blocks, and
# 0.9 for the two understood ones. The wider setting is not caution for its own
# sake: TD understood kappa per age cell runs 19.6, 21.0, 110.7 at 14, 15 and 16
# months, so the log-linear fit is smoothing a genuinely jagged profile and the
# fitted rise should not be stated more confidently than that. Why the 16-18 month
# cells sit so far above their neighbours is not yet understood and is recorded as
# a follow-up.

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
    # Calibrated for VG12, 2026-08-05. NOT YET ATTACHED to any registered model:
    # the reparameterisation is expected to fix VG12's energy BFMI but that has not
    # been demonstrated, and attaching it is a graph change requiring a refit. See
    # notes/202608050900-td-hierarchical-geometry.md §7.1 and §9 item 1.
    #
    # p0 is the observed comprehension proportion in the 11-13 month band, 84.3 of
    # 810 items over 1,106 rows -- which independently reproduces the ~83-word
    # Wordbank 12-month norm this model's low anchor is already tied to. It gives
    # c = 1 / (p0 (1 - p0)) = 10.72.
    reference_proportion=0.1041,
    # Chosen so the induced marginals stay recognisably the current beliefs while
    # the prior moves onto the budget and the split. Against the priors these
    # replace -- tau_subject ~ HalfNormal(1.5), excess_young ~ LogNormal(log 40,
    # 0.9) -- the induced 5/50/95 are:
    #     tau_subject         0.38 / 0.79 / 1.59   (was 0.09 / 1.01 / 2.94)
    #     kappa_excess_young  7.25 / 34.7 / 212    (was 9.10 / 40.0 / 176)
    # The dispersion marginal is nearly unchanged. The subject marginal is tighter,
    # necessarily: a shared budget cannot let both parameters range over 30x
    # independently, and refusing to is the entire point.
    #
    # The share prior is deliberately NOT centred where the old priors implied
    # (median 0.79). VG12's posterior implies a share of 0.598, so centring there
    # would manufacture a prior-data conflict at prior CDF 0.04-0.12. Beta(3.9,
    # 2.1) spans 0.33-0.92 across its 5-95% range and puts that posterior at CDF
    # 0.368, with the total at CDF 0.383 -- both central, neither asserted.
    total_mu=0.0,
    total_sigma=0.8,
    share_alpha=3.9,
    share_beta=2.1,
)

_TD_SPOKEN_VARIANCE_PARTITION = SubjectVariancePartitionParams(
    # VG11's counterpart of _TD_UNDERSTOOD_VARIANCE_PARTITION. The budget and share
    # priors are *identical* to VG12's; the only model-specific input is p0, which
    # is an empirical quantity rather than a choice. That is the design working as
    # intended -- the priors are stated in units of logit-scale scatter, which is
    # comparable across outcomes, while p0 carries the outcome's level.
    #
    # p0 = 9.57/810, the observed spoken proportion in the 11-13 month band over
    # 1,177 rows. Spoken vocabulary at 12 months is tiny, so c = 1/(p0 (1-p0)) =
    # 85.65 against VG12's 10.72.
    #
    # Induced marginals against the priors they replace -- tau_subject ~
    # HalfNormal(1.5), excess_young ~ LogNormal(log 311, 0.7):
    #     tau_subject         0.38 / 0.79 / 1.59   (was 0.09 / 1.01 / 2.94)
    #     kappa_excess_young  57.9 / 277  / 1689   (was 98.3 / 311  / 984)
    # Both medians land within ~20% of the ones they replace.
    #
    # CAUTION, and it is a real one: unlike VG12 there is no VG11 posterior to
    # check the share prior against -- VG11 has never completed a fit. VG12's data
    # implied a share of 0.598 where its old priors implied 0.79, so centring on
    # the old implication would have manufactured a conflict. VG11 has an even
    # lower repeat rate (13.4% against 17.2%), hence even less information about
    # the split, so the share prior is deliberately left weak -- Beta(3.9, 2.1)
    # spans 0.33-0.91 -- rather than centred anywhere in particular. Revisit once
    # VG11 has a posterior. See notes/202608050900-td-hierarchical-geometry.md §7.
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

# Down syndrome joint frame -- VG09, VG10, VG15, VG16, the four models carrying
# subject random intercepts on *both* outcomes and therefore sharing one
# calibration target. 671 understood and 645 nested-spoken rows over 8-115
# months, 387 children at 1.73 administrations each.
#
# Section 19 left these on the legacy form because the frame failed its recovery
# check. Section 22 re-runs that check properly and reaches a different verdict.
# The failure was not scatter: the estimator is biased *downward* by a measured,
# monotone amount, so its estimates are usable as lower bounds. Correcting each
# by the bias measured at it -- kappa(24) 81.6 / 0.74 = 110 and kappa(48)
# 20.3 / 0.62 = 33 for understood, 13.8 / 0.83 = 17 and 7.6 / 0.70 = 11 for the
# ratio -- gives the medians below.
#
# What is *not* in doubt is that the legacy prior is wrong. All eight Down
# syndrome joint models put b_kappa_mag_u at prior CDF 0.993-0.9999 against
# HalfNormal(0.3), well mixed (ESS 313-3,028), and several put b_kappa_mag_s
# there too with *negative* contraction -- the posterior wider than the prior,
# the likelihood pushing outward against it. That is the pathology section 18
# built the two-anchor form to remove, and removing it needs no view on the level
# at all, since the anchored form has no slope prior to get wrong.
#
# sigma is 1.0 on all four anchors, wider than anywhere else in the family
# (0.7 spoken, 0.8-0.9 understood). That is the honest width for this frame: the
# bias correction is itself uncertain, the mean sweep moves the ratio's young
# anchor by a factor of 1.8 across spline knot counts (the understood one is
# stable to 0.4%), and 1.0 leaves the prior spanning 5-95% of 24-551 at the
# understood young anchor, which covers the uncorrected estimate, the corrected
# one and the current posterior alike. The floor keeps the shared weak default.

_DS_JOINT_UNDERSTOOD_KAPPA_RE = KappaAnchorPriorParams(
    # Totals 110 @ 24 mo and 33 @ 48 mo. The dev-config posteriors these replace
    # sat at 66 and 20 under a prior centred near 13, so this moves the prior to
    # where the data already were rather than moving the fit.
    anchor_ages=(24.0, 48.0),
    kappa_min_mu=math.log(3.0),
    kappa_min_sigma=0.8,
    excess_young_mu=math.log(106.0),
    excess_young_sigma=1.0,
    excess_old_mu=math.log(28.7),
    excess_old_sigma=1.0,
)

_DS_JOINT_Q_KAPPA_RE = KappaAnchorPriorParams(
    # The production ratio on the nested scale the engines use: spoken out of that
    # child's own observed understood count, mean q. Totals 17 @ 24 mo and 11 @ 48.
    # 469 of 1,114 spoken rows fall back to the marginal out-of-810 likelihood
    # because the understood count is missing or violated, so this calibration
    # covers the 58% on the nested scale and kappa_s governs both.
    anchor_ages=(24.0, 48.0),
    kappa_min_mu=math.log(3.0),
    kappa_min_sigma=0.8,
    excess_young_mu=math.log(12.6),
    excess_young_sigma=1.0,
    excess_old_mu=math.log(6.7),
    excess_old_sigma=1.0,
)

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
    # Comprehension reporting stops at 84 mo, matching the joint models; raised from
    # 72 on 2026-08-13 once uk_07 and the reinstated uk_06 signing rows rebuilt the
    # older tail. 38 of the 987 understood rows now sit at or above 72 (95th
    # percentile 69 mo), 25 of them from 20 children in the 72-84 band. Above 84
    # only 13 rows from 11 children remain, and 84 is the high trend anchor past
    # which the mean is levelled off rather than fitted. Reporting only -- it cannot
    # move the posterior. The whole-month companion still covers the full observed
    # span, where its n_obs column records the emptiness directly. See
    # notes/202608042030-q-mean-extrapolation.md.
    report_max_age_understood=84,
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
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
    gp_domain_months=_DS_GP_DOMAIN_MONTHS,
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
    # frame, not an independent norm — there is none for DS comprehension.
    #
    # This corrects the *level* only. The prior is still logit-linear in age
    # between the anchors while the trajectory is strongly concave on that scale,
    # so lifting the line to fit 24-60 raises its backward extrapolation too and
    # the 12-18 mo end gets worse, not better. eta_u absorbs the difference and
    # sits at prior CDF 0.80-0.89 across all eight DS joint models. The fix is a
    # log-age mean, which is a graph change and therefore a new variant; see
    # notes/202608041216-ds-understood-trajectory-prior.md.
    p_slope_low_u_alpha=1.5,
    p_slope_low_u_beta=8.0,
    p_slope_hi_u_alpha=3.0,
    p_slope_hi_u_beta=1.3,
    eta_u_sigma=0.6,
    # q anchors. Broadened from the VG07-posterior-derived Beta(3,22)/Beta(20,4)
    # to remove prior-data double-dipping; the high anchor then recalibrated
    # 2026-08-04 from Beta(3,2). q_low ~ Beta(2,12) is unchanged and well-centred
    # (fitted 0.117 at prior CDF 0.46, contraction 0.81); the high anchor was
    # carrying the whole displacement. A weighted least-squares line through the
    # directly observed spoken/understood ratio (902 rows with both outcomes,
    # 18-72 mo) implies a trend q(84) of 0.946; unweighted 0.924, 36 mo+ 0.943.
    # Beta(3,2)'s median of 0.614 left the prior trend line 1.9x too shallow,
    # putting the prior median spoken curve 12x above the fitted one at 12 mo and
    # 2.2x below it at 54. Beta(4,1.2) has median 0.805 — deliberately short of
    # the observed extrapolation, because the last band carrying both outcomes is
    # 72 mo (n=11) and only one row has both above 78 — with a wide lower tail
    # (5-95% 0.44-0.98). Frame calibration, not an independent norm: there is
    # none for the DS production ratio. See
    # notes/202608041730-ds-spoken-q-trajectory-prior.md.
    p_slope_low_q_alpha=2.0,
    p_slope_low_q_beta=12.0,
    p_slope_hi_q_alpha=4.0,
    p_slope_hi_q_beta=1.2,
    # Level the mean off above the 84 mo high anchor rather than extrapolating the
    # line to the top of the 115 mo GP domain. Without it the fitted q mean alone
    # reaches 0.993 at 115 mo (P(mean > 0.99) = 0.90 across the posterior) against a
    # realised 0.842, so the GP spends -3.3 logits correcting the mean's asymptote
    # while sitting idle (+0.08) at 48 mo where the data are; understood shows the
    # same defect about 3x milder. One-sided, and the corner is rounded over about
    # +/-4 mo so the curve stays monotone -- see gp_utils.trend_and_gp.
    # Comprehension reporting stops at 84 mo, raised from 72 on 2026-08-13 once
    # uk_07 and the reinstated uk_06 signing rows rebuilt the older tail. Understood
    # is observed on 987 rows with a 95th percentile of 69 mo; the 72-84 band now
    # carries 25 rows from 20 children across five studies (ie_01, uk_01, uk_06,
    # uk_07, us_02), so reporting it is no longer extrapolation. That is the same
    # test, and the same threshold, the study owner applied when raising
    # report_max_age_signed from 60 to 84 in #212.
    # 84 and no further, for two reasons that coincide: at or above it understood
    # has only 13 rows from 11 children, and 84 is the high trend anchor, above
    # which clamp_mean_above_hi_anchor levels the mean off rather than fitting it --
    # so anything quoted there is a levelled-off extrapolation moved by the GP alone.
    # Spoken keeps the full grid: 1428 rows, 95th percentile 81 mo, 59 rows at or
    # above 84. Reporting only -- it cannot move the posterior. See
    # notes/202608042030-q-mean-extrapolation.md.
    report_max_age_understood=84,
    clamp_mean_above_hi_anchor=CLAMP_Q_ONLY,
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
    # frame, not an independent norm — there is none for DS comprehension.
    #
    # This corrects the *level* only. The prior is still logit-linear in age
    # between the anchors while the trajectory is strongly concave on that scale,
    # so lifting the line to fit 24-60 raises its backward extrapolation too and
    # the 12-18 mo end gets worse, not better. eta_u absorbs the difference and
    # sits at prior CDF 0.80-0.89 across all eight DS joint models. The fix is a
    # log-age mean, which is a graph change and therefore a new variant; see
    # notes/202608041216-ds-understood-trajectory-prior.md.
    p_slope_low_u_alpha=1.5,
    p_slope_low_u_beta=8.0,
    p_slope_hi_u_alpha=3.0,
    p_slope_hi_u_beta=1.3,
    eta_u_sigma=0.6,
    # q anchors. Broadened from the VG07-posterior-derived Beta(3,22)/Beta(20,4)
    # to remove prior-data double-dipping; the high anchor then recalibrated
    # 2026-08-04 from Beta(3,2). q_low ~ Beta(2,12) is unchanged and well-centred
    # (fitted 0.117 at prior CDF 0.46, contraction 0.81); the high anchor was
    # carrying the whole displacement. A weighted least-squares line through the
    # directly observed spoken/understood ratio (902 rows with both outcomes,
    # 18-72 mo) implies a trend q(84) of 0.946; unweighted 0.924, 36 mo+ 0.943.
    # Beta(3,2)'s median of 0.614 left the prior trend line 1.9x too shallow,
    # putting the prior median spoken curve 12x above the fitted one at 12 mo and
    # 2.2x below it at 54. Beta(4,1.2) has median 0.805 — deliberately short of
    # the observed extrapolation, because the last band carrying both outcomes is
    # 72 mo (n=11) and only one row has both above 78 — with a wide lower tail
    # (5-95% 0.44-0.98). Frame calibration, not an independent norm: there is
    # none for the DS production ratio. See
    # notes/202608041730-ds-spoken-q-trajectory-prior.md.
    p_slope_low_q_alpha=2.0,
    p_slope_low_q_beta=12.0,
    p_slope_hi_q_alpha=4.0,
    p_slope_hi_q_beta=1.2,
    tau_u_sigma=0.5,
    tau_q_sigma=0.5,
    # Level the mean off above the 84 mo high anchor rather than extrapolating the
    # line to the top of the 115 mo GP domain. Without it the fitted q mean alone
    # reaches 0.993 at 115 mo (P(mean > 0.99) = 0.90 across the posterior) against a
    # realised 0.842, so the GP spends -3.3 logits correcting the mean's asymptote
    # while sitting idle (+0.08) at 48 mo where the data are; understood shows the
    # same defect about 3x milder. One-sided, and the corner is rounded over about
    # +/-4 mo so the curve stays monotone -- see gp_utils.trend_and_gp.
    # Comprehension reporting stops at 84 mo, raised from 72 on 2026-08-13 once
    # uk_07 and the reinstated uk_06 signing rows rebuilt the older tail. Understood
    # is observed on 987 rows with a 95th percentile of 69 mo; the 72-84 band now
    # carries 25 rows from 20 children across five studies (ie_01, uk_01, uk_06,
    # uk_07, us_02), so reporting it is no longer extrapolation. That is the same
    # test, and the same threshold, the study owner applied when raising
    # report_max_age_signed from 60 to 84 in #212.
    # 84 and no further, for two reasons that coincide: at or above it understood
    # has only 13 rows from 11 children, and 84 is the high trend anchor, above
    # which clamp_mean_above_hi_anchor levels the mean off rather than fitting it --
    # so anything quoted there is a levelled-off extrapolation moved by the GP alone.
    # Spoken keeps the full grid: 1428 rows, 95th percentile 81 mo, 59 rows at or
    # above 84. Reporting only -- it cannot move the posterior. See
    # notes/202608042030-q-mean-extrapolation.md.
    report_max_age_understood=84,
    clamp_mean_above_hi_anchor=CLAMP_Q_ONLY,
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
    # frame, not an independent norm — there is none for DS comprehension.
    #
    # This corrects the *level* only. The prior is still logit-linear in age
    # between the anchors while the trajectory is strongly concave on that scale,
    # so lifting the line to fit 24-60 raises its backward extrapolation too and
    # the 12-18 mo end gets worse, not better. eta_u absorbs the difference and
    # sits at prior CDF 0.80-0.89 across all eight DS joint models. The fix is a
    # log-age mean, which is a graph change and therefore a new variant; see
    # notes/202608041216-ds-understood-trajectory-prior.md.
    p_slope_low_u_alpha=1.5,
    p_slope_low_u_beta=8.0,
    p_slope_hi_u_alpha=3.0,
    p_slope_hi_u_beta=1.3,
    eta_u_sigma=0.6,
    # q anchors. Broadened from the VG07-posterior-derived Beta(3,22)/Beta(20,4)
    # to remove prior-data double-dipping; the high anchor then recalibrated
    # 2026-08-04 from Beta(3,2). q_low ~ Beta(2,12) is unchanged and well-centred
    # (fitted 0.117 at prior CDF 0.46, contraction 0.81); the high anchor was
    # carrying the whole displacement. A weighted least-squares line through the
    # directly observed spoken/understood ratio (902 rows with both outcomes,
    # 18-72 mo) implies a trend q(84) of 0.946; unweighted 0.924, 36 mo+ 0.943.
    # Beta(3,2)'s median of 0.614 left the prior trend line 1.9x too shallow,
    # putting the prior median spoken curve 12x above the fitted one at 12 mo and
    # 2.2x below it at 54. Beta(4,1.2) has median 0.805 — deliberately short of
    # the observed extrapolation, because the last band carrying both outcomes is
    # 72 mo (n=11) and only one row has both above 78 — with a wide lower tail
    # (5-95% 0.44-0.98). Frame calibration, not an independent norm: there is
    # none for the DS production ratio. See
    # notes/202608041730-ds-spoken-q-trajectory-prior.md.
    p_slope_low_q_alpha=2.0,
    p_slope_low_q_beta=12.0,
    p_slope_hi_q_alpha=4.0,
    p_slope_hi_q_beta=1.2,
    tau_u_sigma=0.5,
    tau_q_sigma=0.5,
    use_subject_re_u=True,
    tau_subj_u_sigma=1.5,
    # Level the mean off above the 84 mo high anchor rather than extrapolating the
    # line to the top of the 115 mo GP domain. Without it the fitted q mean alone
    # reaches 0.993 at 115 mo (P(mean > 0.99) = 0.90 across the posterior) against a
    # realised 0.842, so the GP spends -3.3 logits correcting the mean's asymptote
    # while sitting idle (+0.08) at 48 mo where the data are; understood shows the
    # same defect about 3x milder. One-sided, and the corner is rounded over about
    # +/-4 mo so the curve stays monotone -- see gp_utils.trend_and_gp.
    # Comprehension reporting stops at 84 mo, raised from 72 on 2026-08-13 once
    # uk_07 and the reinstated uk_06 signing rows rebuilt the older tail. Understood
    # is observed on 987 rows with a 95th percentile of 69 mo; the 72-84 band now
    # carries 25 rows from 20 children across five studies (ie_01, uk_01, uk_06,
    # uk_07, us_02), so reporting it is no longer extrapolation. That is the same
    # test, and the same threshold, the study owner applied when raising
    # report_max_age_signed from 60 to 84 in #212.
    # 84 and no further, for two reasons that coincide: at or above it understood
    # has only 13 rows from 11 children, and 84 is the high trend anchor, above
    # which clamp_mean_above_hi_anchor levels the mean off rather than fitting it --
    # so anything quoted there is a levelled-off extrapolation moved by the GP alone.
    # Spoken keeps the full grid: 1428 rows, 95th percentile 81 mo, 59 rows at or
    # above 84. Reporting only -- it cannot move the posterior. See
    # notes/202608042030-q-mean-extrapolation.md.
    report_max_age_understood=84,
    clamp_mean_above_hi_anchor=CLAMP_Q_ONLY,
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
    # frame, not an independent norm — there is none for DS comprehension.
    #
    # This corrects the *level* only. The prior is still logit-linear in age
    # between the anchors while the trajectory is strongly concave on that scale,
    # so lifting the line to fit 24-60 raises its backward extrapolation too and
    # the 12-18 mo end gets worse, not better. eta_u absorbs the difference and
    # sits at prior CDF 0.80-0.89 across all eight DS joint models. The fix is a
    # log-age mean, which is a graph change and therefore a new variant; see
    # notes/202608041216-ds-understood-trajectory-prior.md.
    p_slope_low_u_alpha=1.5,
    p_slope_low_u_beta=8.0,
    p_slope_hi_u_alpha=3.0,
    p_slope_hi_u_beta=1.3,
    eta_u_sigma=0.6,
    # q anchors. Broadened from the VG07-posterior-derived Beta(3,22)/Beta(20,4)
    # to remove prior-data double-dipping; the high anchor then recalibrated
    # 2026-08-04 from Beta(3,2). q_low ~ Beta(2,12) is unchanged and well-centred
    # (fitted 0.117 at prior CDF 0.46, contraction 0.81); the high anchor was
    # carrying the whole displacement. A weighted least-squares line through the
    # directly observed spoken/understood ratio (902 rows with both outcomes,
    # 18-72 mo) implies a trend q(84) of 0.946; unweighted 0.924, 36 mo+ 0.943.
    # Beta(3,2)'s median of 0.614 left the prior trend line 1.9x too shallow,
    # putting the prior median spoken curve 12x above the fitted one at 12 mo and
    # 2.2x below it at 54. Beta(4,1.2) has median 0.805 — deliberately short of
    # the observed extrapolation, because the last band carrying both outcomes is
    # 72 mo (n=11) and only one row has both above 78 — with a wide lower tail
    # (5-95% 0.44-0.98). Frame calibration, not an independent norm: there is
    # none for the DS production ratio. See
    # notes/202608041730-ds-spoken-q-trajectory-prior.md.
    p_slope_low_q_alpha=2.0,
    p_slope_low_q_beta=12.0,
    p_slope_hi_q_alpha=4.0,
    p_slope_hi_q_beta=1.2,
    tau_u_sigma=0.5,
    tau_q_sigma=0.5,
    use_subject_re_u=True,
    tau_subj_u_sigma=1.5,
    use_subject_re_q=True,
    tau_subj_q_sigma=1.5,
    kappa_u=_DS_JOINT_UNDERSTOOD_KAPPA_RE,
    kappa_s=_DS_JOINT_Q_KAPPA_RE,
    # Level the mean off above the 84 mo high anchor rather than extrapolating the
    # line to the top of the 115 mo GP domain. Without it the fitted q mean alone
    # reaches 0.993 at 115 mo (P(mean > 0.99) = 0.90 across the posterior) against a
    # realised 0.842, so the GP spends -3.3 logits correcting the mean's asymptote
    # while sitting idle (+0.08) at 48 mo where the data are; understood shows the
    # same defect about 3x milder. One-sided, and the corner is rounded over about
    # +/-4 mo so the curve stays monotone -- see gp_utils.trend_and_gp.
    # Comprehension reporting stops at 84 mo, raised from 72 on 2026-08-13 once
    # uk_07 and the reinstated uk_06 signing rows rebuilt the older tail. Understood
    # is observed on 987 rows with a 95th percentile of 69 mo; the 72-84 band now
    # carries 25 rows from 20 children across five studies (ie_01, uk_01, uk_06,
    # uk_07, us_02), so reporting it is no longer extrapolation. That is the same
    # test, and the same threshold, the study owner applied when raising
    # report_max_age_signed from 60 to 84 in #212.
    # 84 and no further, for two reasons that coincide: at or above it understood
    # has only 13 rows from 11 children, and 84 is the high trend anchor, above
    # which clamp_mean_above_hi_anchor levels the mean off rather than fitting it --
    # so anything quoted there is a levelled-off extrapolation moved by the GP alone.
    # Spoken keeps the full grid: 1428 rows, 95th percentile 81 mo, 59 rows at or
    # above 84. Reporting only -- it cannot move the posterior. See
    # notes/202608042030-q-mean-extrapolation.md.
    report_max_age_understood=84,
    clamp_mean_above_hi_anchor=CLAMP_Q_ONLY,
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
    # frame, not an independent norm — there is none for DS comprehension.
    #
    # This corrects the *level* only. The prior is still logit-linear in age
    # between the anchors while the trajectory is strongly concave on that scale,
    # so lifting the line to fit 24-60 raises its backward extrapolation too and
    # the 12-18 mo end gets worse, not better. eta_u absorbs the difference and
    # sits at prior CDF 0.80-0.89 across all eight DS joint models. The fix is a
    # log-age mean, which is a graph change and therefore a new variant; see
    # notes/202608041216-ds-understood-trajectory-prior.md.
    p_slope_low_u_alpha=1.5,
    p_slope_low_u_beta=8.0,
    p_slope_hi_u_alpha=3.0,
    p_slope_hi_u_beta=1.3,
    eta_u_sigma=0.6,
    # q anchors. Broadened from the VG07-posterior-derived Beta(3,22)/Beta(20,4)
    # to remove prior-data double-dipping; the high anchor then recalibrated
    # 2026-08-04 from Beta(3,2). q_low ~ Beta(2,12) is unchanged and well-centred
    # (fitted 0.117 at prior CDF 0.46, contraction 0.81); the high anchor was
    # carrying the whole displacement. A weighted least-squares line through the
    # directly observed spoken/understood ratio (902 rows with both outcomes,
    # 18-72 mo) implies a trend q(84) of 0.946; unweighted 0.924, 36 mo+ 0.943.
    # Beta(3,2)'s median of 0.614 left the prior trend line 1.9x too shallow,
    # putting the prior median spoken curve 12x above the fitted one at 12 mo and
    # 2.2x below it at 54. Beta(4,1.2) has median 0.805 — deliberately short of
    # the observed extrapolation, because the last band carrying both outcomes is
    # 72 mo (n=11) and only one row has both above 78 — with a wide lower tail
    # (5-95% 0.44-0.98). Frame calibration, not an independent norm: there is
    # none for the DS production ratio. See
    # notes/202608041730-ds-spoken-q-trajectory-prior.md.
    p_slope_low_q_alpha=2.0,
    p_slope_low_q_beta=12.0,
    p_slope_hi_q_alpha=4.0,
    p_slope_hi_q_beta=1.2,
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
    # Level the mean off above the 84 mo high anchor rather than extrapolating the
    # line to the top of the 115 mo GP domain. Without it the fitted q mean alone
    # reaches 0.993 at 115 mo (P(mean > 0.99) = 0.90 across the posterior) against a
    # realised 0.842, so the GP spends -3.3 logits correcting the mean's asymptote
    # while sitting idle (+0.08) at 48 mo where the data are; understood shows the
    # same defect about 3x milder. One-sided, and the corner is rounded over about
    # +/-4 mo so the curve stays monotone -- see gp_utils.trend_and_gp.
    # Comprehension reporting stops at 84 mo, raised from 72 on 2026-08-13 once
    # uk_07 and the reinstated uk_06 signing rows rebuilt the older tail. Understood
    # is observed on 987 rows with a 95th percentile of 69 mo; the 72-84 band now
    # carries 25 rows from 20 children across five studies (ie_01, uk_01, uk_06,
    # uk_07, us_02), so reporting it is no longer extrapolation. That is the same
    # test, and the same threshold, the study owner applied when raising
    # report_max_age_signed from 60 to 84 in #212.
    # 84 and no further, for two reasons that coincide: at or above it understood
    # has only 13 rows from 11 children, and 84 is the high trend anchor, above
    # which clamp_mean_above_hi_anchor levels the mean off rather than fitting it --
    # so anything quoted there is a levelled-off extrapolation moved by the GP alone.
    # Spoken keeps the full grid: 1428 rows, 95th percentile 81 mo, 59 rows at or
    # above 84. Reporting only -- it cannot move the posterior. See
    # notes/202608042030-q-mean-extrapolation.md.
    report_max_age_understood=84,
    clamp_mean_above_hi_anchor=CLAMP_Q_ONLY,
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
    # Widen the reference pool beyond English (issue: DS-TD language symmetry).
    # The DS pool is already a quarter non-English (es_01 Spanish, it_01 Italian)
    # while this reference was English-only; the study REs below absorb the
    # between-language variation. See ROMANCE_LANGUAGES for the admission criteria
    # and the two measurement checks, and note that VG03/VG04 stay English-only
    # because they carry no random effects to absorb it.
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
    # Sampling geometry, enabled 2026-08-05 after the VG12 test-config trial in
    # notes/202608050900-td-hierarchical-geometry.md §7. Centring the study block
    # took tau's ESS from 310 to 6,950 and max R-hat from 1.0133 to 1.0057; the
    # partition cut divergences from 59 to 14. Neither moves the energy BFMI --
    # that is driven by missing within-child replication and is not reparameterisable
    # away -- so this model is still expected to need the caveated publication path.
    centred_study_re=True,
    subject_variance_partition=_TD_SPOKEN_VARIANCE_PARTITION,
    kappa=_TD_SPOKEN_KAPPA_RE,
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
    # REVERTED to 0.5 on 2026-08-05, having been widened to 1.0 earlier the same
    # day. The widening was a calibration fix: at 0.5 the fitted amplitude sat at
    # prior CDF 0.913 with contraction 0.106, i.e. the model reporting its prior
    # back, the same signature that justified the 2026-08-04 DS widening.
    #
    # It was withdrawn because it cost convergence. Three rep fits isolate it:
    #     original (eta 0.5, no geometry changes)      2 divergences, BFMI 0.202
    #     centring + partition + eta 1.0              29 divergences, BFMI 0.208
    #     centring + partition + eta 0.5               2 divergences, BFMI 0.201
    # Centring and the partition cost nothing; the widening alone caused all 27.
    # It was already the only arm to raise divergences in the test-config trial
    # (76 against 59). Widening a weakly identified parameter gave it room to
    # wander: even at 1.0 it only reached prior CDF 0.810 with contraction 0.166,
    # so the calibration was not actually bought.
    #
    # Divergences bias the whole posterior, while the miscalibration is local to a
    # GP smoothing hyperparameter that is not a reported developmental quantity --
    # so the trade is not worth taking. The calibration defect is real and stands
    # recorded in notes/202608050900-td-hierarchical-geometry.md §5; fixing it
    # needs a change that identifies the amplitude rather than merely freeing it.
    eta_sigma=0.5,
    # WG + Oxford CDI only (WS comprehension is a production proxy).
    # Study REs absorb between-lab variation, so subsampling is not needed.
    sample_fraction=1.0,
    # Widen the reference pool beyond English (issue: DS-TD language symmetry).
    # The DS pool is already a quarter non-English (es_01 Spanish, it_01 Italian)
    # while this reference was English-only; the study REs below absorb the
    # between-language variation. See ROMANCE_LANGUAGES for the admission criteria
    # and the two measurement checks, and note that VG03/VG04 stay English-only
    # because they carry no random effects to absorb it.
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
    # Sampling geometry, enabled 2026-08-05 after the VG12 test-config trial in
    # notes/202608050900-td-hierarchical-geometry.md §7. Centring the study block
    # took tau's ESS from 310 to 6,950 and max R-hat from 1.0133 to 1.0057; the
    # partition cut divergences from 59 to 14. Neither moves the energy BFMI --
    # that is driven by missing within-child replication and is not reparameterisable
    # away -- so this model is still expected to need the caveated publication path.
    centred_study_re=True,
    subject_variance_partition=_TD_UNDERSTOOD_VARIANCE_PARTITION,
    kappa=_TD_UNDERSTOOD_KAPPA_RE,
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
    # absorb between-lab variation so no subsampling is required.
    sample_fraction=1.0,
    # Widen the reference pool beyond English (issue: DS-TD language symmetry).
    # The DS pool is already a quarter non-English (es_01 Spanish, it_01 Italian)
    # while this reference was English-only; the study REs below absorb the
    # between-language variation. See ROMANCE_LANGUAGES for the admission criteria
    # and the two measurement checks, and note that VG03/VG04 stay English-only
    # because they carry no random effects to absorb it.
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
    ages_query=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
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
    kappa_u=_DS_JOINT_UNDERSTOOD_KAPPA_RE,
    kappa_s=_DS_JOINT_Q_KAPPA_RE,
    # Understood trajectory: matches VG05, including the 2026-08-04 anchor
    # recalibration (Beta(1,7) -> Beta(1.5,8) at 24 mo, Beta(2,1.5) -> Beta(3,1.3)
    # at 84 mo) and eta_u at 0.6. See VG05 for the reasoning and
    # notes/202608041216-ds-understood-trajectory-prior.md for the measurements.
    p_slope_low_u_alpha=1.5,
    p_slope_low_u_beta=8.0,
    p_slope_hi_u_alpha=3.0,
    p_slope_hi_u_beta=1.3,
    eta_u_sigma=0.6,
    # Spoken ratio q. Broadened from the VG07-posterior-derived
    # Beta(3,22)/Beta(20,4); the high anchor then recalibrated 2026-08-04 from
    # Beta(3,2) to Beta(4,1.2) alongside the rest of the DS joint family. q_low ~
    # Beta(2,12) is unchanged. See VG05 for the calibration and
    # notes/202608041730-ds-spoken-q-trajectory-prior.md.
    p_slope_low_q_alpha=2.0,
    p_slope_low_q_beta=12.0,
    p_slope_hi_q_alpha=4.0,
    p_slope_hi_q_beta=1.2,
    # Signed ratio r uses the three-anchor tent + GP defined above.  uk_01's
    # signed-only field and uk_06's unverified field are excluded from the signed
    # likelihood by default; their understood/spoken observations remain.
    # Level the mean off above the 84 mo high anchor rather than extrapolating the
    # line to the top of the 115 mo GP domain. Without it the fitted q mean alone
    # reaches 0.993 at 115 mo (P(mean > 0.99) = 0.90 across the posterior) against a
    # realised 0.842, so the GP spends -3.3 logits correcting the mean's asymptote
    # while sitting idle (+0.08) at 48 mo where the data are; understood shows the
    # same defect about 3x milder. One-sided, and the corner is rounded over about
    # +/-4 mo so the curve stays monotone -- see gp_utils.trend_and_gp.
    # Comprehension reporting stops at 84 mo, raised from 72 on 2026-08-13 once
    # uk_07 and the reinstated uk_06 signing rows rebuilt the older tail. Understood
    # is observed on 987 rows with a 95th percentile of 69 mo; the 72-84 band now
    # carries 25 rows from 20 children across five studies (ie_01, uk_01, uk_06,
    # uk_07, us_02), so reporting it is no longer extrapolation. That is the same
    # test, and the same threshold, the study owner applied when raising
    # report_max_age_signed from 60 to 84 in #212.
    # 84 and no further, for two reasons that coincide: at or above it understood
    # has only 13 rows from 11 children, and 84 is the high trend anchor, above
    # which clamp_mean_above_hi_anchor levels the mean off rather than fitting it --
    # so anything quoted there is a levelled-off extrapolation moved by the GP alone.
    # Spoken keeps the full grid: 1428 rows, 95th percentile 81 mo, 59 rows at or
    # above 84. Reporting only -- it cannot move the posterior. See
    # notes/202608042030-q-mean-extrapolation.md.
    report_max_age_understood=84,
    # Signed gets its own cap rather than inheriting the comprehension one, which
    # is what it did until 2026-08-13. 84 matches VG15's report_max_age_signed on
    # the same evidence, and stops the r(a) table where the r(a) figure stops.
    report_max_age_signed=84,
    clamp_mean_above_hi_anchor=CLAMP_Q_ONLY,
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
    # centre), replacing the VG07-posterior-derived Beta(3,22)/Beta(20,4). The
    # high anchor was recalibrated 2026-08-04 from Beta(3,2) to Beta(4,1.2)
    # alongside the rest of the DS joint family — see VG05 for the calibration and
    # notes/202608041730-ds-spoken-q-trajectory-prior.md. The u anchors are left
    # unchanged, matching VG10. The signed mean is a three-anchor hump (tent),
    # inherited from the JointModelDefinition dataclass defaults (young/peak/old
    # sign anchors + GP), so there is no monotone signed slope to tighten; the
    # anchors set the level and the GP carries smooth departures. Option D (below)
    # removes the GP<->intercept ridge.
    p_slope_low_q_alpha=2.0,
    p_slope_low_q_beta=12.0,
    p_slope_hi_q_alpha=4.0,
    p_slope_hi_q_beta=1.2,
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
    # Level the mean off above the 84 mo high anchor rather than extrapolating the
    # line to the top of the 115 mo GP domain. Without it the fitted q mean alone
    # reaches 0.993 at 115 mo (P(mean > 0.99) = 0.90 across the posterior) against a
    # realised 0.842, so the GP spends -3.3 logits correcting the mean's asymptote
    # while sitting idle (+0.08) at 48 mo where the data are; understood shows the
    # same defect about 3x milder. One-sided, and the corner is rounded over about
    # +/-4 mo so the curve stays monotone -- see gp_utils.trend_and_gp.
    # Comprehension reporting stops at 84 mo, raised from 72 on 2026-08-13 once
    # uk_07 and the reinstated uk_06 signing rows rebuilt the older tail. Understood
    # is observed on 987 rows with a 95th percentile of 69 mo; the 72-84 band now
    # carries 25 rows from 20 children across five studies (ie_01, uk_01, uk_06,
    # uk_07, us_02), so reporting it is no longer extrapolation. That is the same
    # test, and the same threshold, the study owner applied when raising
    # report_max_age_signed from 60 to 84 in #212.
    # 84 and no further, for two reasons that coincide: at or above it understood
    # has only 13 rows from 11 children, and 84 is the high trend anchor, above
    # which clamp_mean_above_hi_anchor levels the mean off rather than fitting it --
    # so anything quoted there is a levelled-off extrapolation moved by the GP alone.
    # Spoken keeps the full grid: 1428 rows, 95th percentile 81 mo, 59 rows at or
    # above 84. Reporting only -- it cannot move the posterior. See
    # notes/202608042030-q-mean-extrapolation.md.
    report_max_age_understood=84,
    # Signed evidence now reaches 84 months. Adopted 2026-08-07 at 60 on the same
    # argument that capped comprehension at 72 -- then 46 of 593 signed
    # observations lay above 60 and none above 72. uk_07 (PACT-DS) adds 82
    # observations at 34-95 months and uk_06's 11 at 60-115 were unmasked, taking
    # 72-84 from 0 rows to 17 and above-60 from 46 to 98, so 60 no longer marks
    # where the evidence stops -- it hid the pool's only real measurement of the
    # post-peak signing decline. Above 84 stays out: 12 observations from 10
    # children thinning to one source per band, and 84 is the trend's high anchor.
    # Also caps p_any, a function of the signed ratio.
    report_max_age_signed=84,
    clamp_mean_above_hi_anchor=CLAMP_Q_ONLY,
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
    # frame, not an independent norm — there is none for DS comprehension.
    #
    # This corrects the *level* only. The prior is still logit-linear in age
    # between the anchors while the trajectory is strongly concave on that scale,
    # so lifting the line to fit 24-60 raises its backward extrapolation too and
    # the 12-18 mo end gets worse, not better. eta_u absorbs the difference and
    # sits at prior CDF 0.80-0.89 across all eight DS joint models. The fix is a
    # log-age mean, which is a graph change and therefore a new variant; see
    # notes/202608041216-ds-understood-trajectory-prior.md.
    p_slope_low_u_alpha=1.5,
    p_slope_low_u_beta=8.0,
    p_slope_hi_u_alpha=3.0,
    p_slope_hi_u_beta=1.3,
    eta_u_sigma=0.6,
    # q anchors. Broadened from the VG07-posterior-derived Beta(3,22)/Beta(20,4)
    # to remove prior-data double-dipping; the high anchor then recalibrated
    # 2026-08-04 from Beta(3,2). q_low ~ Beta(2,12) is unchanged and well-centred
    # (fitted 0.117 at prior CDF 0.46, contraction 0.81); the high anchor was
    # carrying the whole displacement. A weighted least-squares line through the
    # directly observed spoken/understood ratio (902 rows with both outcomes,
    # 18-72 mo) implies a trend q(84) of 0.946; unweighted 0.924, 36 mo+ 0.943.
    # Beta(3,2)'s median of 0.614 left the prior trend line 1.9x too shallow,
    # putting the prior median spoken curve 12x above the fitted one at 12 mo and
    # 2.2x below it at 54. Beta(4,1.2) has median 0.805 — deliberately short of
    # the observed extrapolation, because the last band carrying both outcomes is
    # 72 mo (n=11) and only one row has both above 78 — with a wide lower tail
    # (5-95% 0.44-0.98). Frame calibration, not an independent norm: there is
    # none for the DS production ratio. See
    # notes/202608041730-ds-spoken-q-trajectory-prior.md.
    p_slope_low_q_alpha=2.0,
    p_slope_low_q_beta=12.0,
    p_slope_hi_q_alpha=4.0,
    p_slope_hi_q_beta=1.2,
    tau_u_sigma=0.5,
    tau_q_sigma=0.5,
    use_subject_re_u=True,
    tau_subj_u_sigma=1.5,
    use_subject_re_q=True,
    tau_subj_q_sigma=1.5,
    use_cross_lag=True,
    # Headline uses the population-relative baseline: with 2-wave-dominated data the
    # pure within-child (own-intercept) baseline is biased by the short-T / Nickell
    # / errors-in-variables mechanics (dev: beta -0.60 [-0.85,-0.35], an artifact),
    # while the population-relative estimate is null (dev: +0.05 [-0.07,0.17]). The
    # within-child variant is reported as a cautionary contrast. See the scoping note.
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
    # Level the mean off above the 84 mo high anchor rather than extrapolating the
    # line to the top of the 115 mo GP domain. Without it the fitted q mean alone
    # reaches 0.993 at 115 mo (P(mean > 0.99) = 0.90 across the posterior) against a
    # realised 0.842, so the GP spends -3.3 logits correcting the mean's asymptote
    # while sitting idle (+0.08) at 48 mo where the data are; understood shows the
    # same defect about 3x milder. One-sided, and the corner is rounded over about
    # +/-4 mo so the curve stays monotone -- see gp_utils.trend_and_gp.
    # Comprehension reporting stops at 84 mo, raised from 72 on 2026-08-13 once
    # uk_07 and the reinstated uk_06 signing rows rebuilt the older tail. Understood
    # is observed on 987 rows with a 95th percentile of 69 mo; the 72-84 band now
    # carries 25 rows from 20 children across five studies (ie_01, uk_01, uk_06,
    # uk_07, us_02), so reporting it is no longer extrapolation. That is the same
    # test, and the same threshold, the study owner applied when raising
    # report_max_age_signed from 60 to 84 in #212.
    # 84 and no further, for two reasons that coincide: at or above it understood
    # has only 13 rows from 11 children, and 84 is the high trend anchor, above
    # which clamp_mean_above_hi_anchor levels the mean off rather than fitting it --
    # so anything quoted there is a levelled-off extrapolation moved by the GP alone.
    # Spoken keeps the full grid: 1428 rows, 95th percentile 81 mo, 59 rows at or
    # above 84. Reporting only -- it cannot move the posterior. See
    # notes/202608042030-q-mean-extrapolation.md.
    report_max_age_understood=84,
    clamp_mean_above_hi_anchor=CLAMP_Q_ONLY,
)

# ============================================================
# VG20 — correlated subject random effects (issue #224): VG10 + rho_uq
# ============================================================


def _as_definition_subclass(base, cls, **overrides):
    """Rebuild ``base`` as an instance of ``cls``, overriding named fields.

    Shallow by design: nested prior dataclasses are shared with ``base`` rather
    than copied, so the derived definition serialises identically to its parent
    except for what is overridden here. ``dataclasses.replace`` cannot do this —
    it returns the base's own class — and ``asdict`` cannot either, because it
    recursively converts the nested prior blocks to plain dicts.
    """
    values = {item.name: getattr(base, item.name) for item in fields(base)}
    values.update(overrides)
    return cls(**values)


# Derived from VG10 so the two differ in exactly one thing, which is what makes
# the comparison in #224 readable: VG10 is nested at rho_uq = 0, and any movement
# in the reported trajectories is a red flag rather than a benefit. Deriving
# rather than restating the priors also means VG10's anchor recalibrations cannot
# drift away from VG20's.
VG20 = _as_definition_subclass(
    VG10,
    BivariateCorrelatedSubjectREModelDefinition,
    model_id="VG20",
    config_name="age-understood-spoken-ds-re-subj-uq-anchored-corr",
    banner=(
        "Fitting Model VG20: VG10 + correlated subject random effects on U and q"
        " (rho_uq) - Down syndrome"
    ),
    # eta = 2: a gentle pull toward independence, so a correlation has to be
    # evidenced. The quantity this model exists to estimate is already known to be
    # positive and small from two independent directions -- VG16's realised
    # intercepts still correlate at +0.135 [0.087, 0.180] with its cross-lag
    # fitted, and VG10's own fitted deviations at +0.152 [0.105, 0.195] with no
    # cross-lag at all -- so a prior that made a large correlation cheap would be
    # assuming the answer. See notes/202608151120-vg16-crosslag-quantified.md.
    subject_re_correlation_eta=2.0,
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
    "vg20": VG20,
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
        # definition class names its study scale differently (tau_study_sigma,
        # tau_u_study_sigma, ...), so any attribute-sniffing check would quietly pass
        # for a class it does not know and give false assurance. The requirement is
        # stated on ENGLISH_AND_ROMANCE_LANGUAGES and is a review matter.
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
