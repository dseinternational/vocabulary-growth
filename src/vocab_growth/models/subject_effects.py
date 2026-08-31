# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve a definition's child-effect structure into one typed plan.

Five different structures can occupy the same seam in a definition, and until
this module they were detected separately, inside the PyMC context, by whichever
builder needed them:

===================  ======  ==========================================
structure            model   how the definition carries it
===================  ======  ==========================================
constant offset      VG10    ``tau_subj_*_sigma`` holds a float
variance partition   VG11    ``subject_variance_partition`` is set
age-varying scale    (A1)    ``tau_subj_*_sigma`` holds an
                             ``AgeVaryingSubjectScale``
child slope          VG19    ``tau_subj_*_sigma`` holds a
                             ``SubjectSlopePriorParams``
correlated pair      VG20    ``subject_re_correlation_eta`` is set
low-rank factor      VG22    ``subject_factor`` is set
===================  ======  ==========================================

Three of the seven are a *scalar field holding an object*. That is not an
accident and is documented where it is done: a fit is validated by comparing
the serialised definition field for field, so a new field on a shared base class
invalidates every existing fit of that class, and the overloaded field is what
let VG19 and Proposal A1 exist without a refit of VG05, VG07-VG10 and VG16. The
cost is that "what child structure does this model have?" had no single answer
— ``build_model_re`` called four different selectors, read two more fields
through ``getattr``, and interleaved five rejection rules with the graph
construction they guard.

:func:`resolve` gives that question one answer, computed **before** the model
context is entered. Every rejection happens there, once, on a definition rather
than part-way through a half-built graph. The builders then branch on a typed
plan.

Pure and PyMC-free by construction, so the whole resolution — including every
refusal — is testable without building a graph, and
``tests/test_subject_effect_plan.py`` exercises it over every registered model
and every rejected combination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

from vocab_growth.models.definitions import (
    AgeVaryingSubjectScale,
    SubjectFactorPriorParams,
    SubjectSlopePriorParams,
    SubjectVariancePartitionParams,
    subject_factor_spec,
    subject_scale_spec,
    subject_slope_spec,
)

#: ``""`` is the single-outcome engines' unsuffixed block.
UNIVARIATE_OUTCOME = ""

#: Every outcome suffix a multi-outcome engine can carry a child effect on, in
#: the order the builders create the blocks. Which of them a given definition
#: actually has is read from the definition, not assumed: the bivariate engines
#: carry ``u`` and ``q``, the joint modality engine adds ``sign``, and a plan
#: that assumed two would drop VG15's third block without saying so.
OUTCOME_SUFFIXES = ("u", "q", "sign")

#: Default reference age for a child slope, in months — the Down syndrome pool's
#: median. Mirrors ``BivariateChildSlopeModelDefinition``'s own default so a
#: definition that predates the field resolves the same way it always did.
DEFAULT_SLOPE_REF_AGE_MONTHS = 36.0


class SubjectEffectKind(Enum):
    """What kind of per-child effect one outcome carries."""

    NONE = "none"
    """No child effect: the outcome has only population and study terms."""

    CONSTANT = "constant"
    """One offset per child, constant in age. ``tau ~ HalfNormal(sigma)``."""

    VARIANCE_PARTITION = "variance_partition"
    """The child scale is a deterministic function of a shared variance budget
    (VG11, VG12), so its own ``HalfNormal`` prior never enters the model."""

    AGE_VARYING = "age_varying"
    """Proposal A1: one deviate per child, scaled by ``tau(age)``."""

    CHILD_SLOPE = "child_slope"
    """VG19: an intercept and a rate per child, correlated."""

    FACTOR = "factor"
    """VG22: this outcome's level and rate are rows of a low-rank factor
    spanning both outcomes. Set on both outcomes or neither."""


@dataclass(frozen=True)
class SubjectOutcomeEffect:
    """One outcome's child effect, fully resolved."""

    outcome: str
    """``"u"``, ``"q"``, or ``""`` for a single-outcome model."""

    kind: SubjectEffectKind

    scale_name: str
    """The emitted scale's variable name — ``tau_subject``, ``tau_subj_u`` or
    ``tau_subj_q``. Named here because every downstream reader indexes the trace
    by it, so it is part of the contract rather than a naming detail."""

    sigma: float | None = None
    """``HalfNormal`` scale for :attr:`SubjectEffectKind.CONSTANT`, and the level
    scale a factor block inherits. ``None`` where the kind places no such prior."""

    age_varying: AgeVaryingSubjectScale | None = None
    slope: SubjectSlopePriorParams | None = None

    @property
    def is_active(self) -> bool:
        return self.kind is not SubjectEffectKind.NONE


@dataclass(frozen=True)
class SubjectEffectPlan:
    """Every child-effect decision a definition implies, resolved once."""

    effects: tuple[SubjectOutcomeEffect, ...]
    """One entry per outcome the engine builds, in build order. Inactive
    outcomes are present with :attr:`SubjectEffectKind.NONE` rather than
    omitted, so a consumer indexes rather than searches."""

    correlation_eta: float | None = None
    """VG20's LKJ concentration on the two constant offsets, or ``None``."""

    factor: SubjectFactorPriorParams | None = None
    """VG22's low-rank factor over all four child effects, or ``None``."""

    slope_ref_age_months: float = DEFAULT_SLOPE_REF_AGE_MONTHS
    """Age at which a child slope's ``tau0`` is the between-child spread."""

    variance_partition: SubjectVariancePartitionParams | None = None
    """VG11/VG12's shared budget, or ``None``."""

    def __getitem__(self, outcome: str) -> SubjectOutcomeEffect:
        for effect in self.effects:
            if effect.outcome == outcome:
                return effect
        raise KeyError(
            f"No child-effect entry for outcome {outcome!r}; this plan covers "
            f"{[effect.outcome for effect in self.effects]}."
        )

    @property
    def any_active(self) -> bool:
        return any(effect.is_active for effect in self.effects)

    @property
    def kinds(self) -> frozenset[SubjectEffectKind]:
        """The distinct kinds in play, for a message or an assertion."""
        return frozenset(effect.kind for effect in self.effects if effect.is_active)


def _scale_name(outcome: str) -> str:
    return "tau_subject" if outcome == UNIVARIATE_OUTCOME else f"tau_subj_{outcome}"


def _outcome_effect(
    definition, outcome: str, *, active: bool, variance_partition
) -> SubjectOutcomeEffect:
    """Resolve one outcome's seam, before any cross-outcome rule is applied."""
    scale_name = _scale_name(outcome)
    if not active:
        return SubjectOutcomeEffect(
            outcome=outcome, kind=SubjectEffectKind.NONE, scale_name=scale_name
        )

    field = (
        "tau_subject_sigma"
        if outcome == UNIVARIATE_OUTCOME
        else f"tau_subj_{outcome}_sigma"
    )
    value = getattr(definition, field)

    age_varying = subject_scale_spec(value)
    if age_varying is not None:
        return SubjectOutcomeEffect(
            outcome=outcome,
            kind=SubjectEffectKind.AGE_VARYING,
            scale_name=scale_name,
            age_varying=age_varying,
        )

    slope = subject_slope_spec(value)
    if slope is not None:
        return SubjectOutcomeEffect(
            outcome=outcome,
            kind=SubjectEffectKind.CHILD_SLOPE,
            scale_name=scale_name,
            slope=slope,
        )

    if not isinstance(value, (int, float)):
        raise ValueError(
            f"{field} must be a number, an AgeVaryingSubjectScale or a "
            f"SubjectSlopePriorParams; got {value!r}."
        )

    if variance_partition is not None:
        # The scale becomes a function of the budget, so its own HalfNormal
        # prior is never used. Recorded as its own kind rather than as CONSTANT
        # so a report cannot describe a prior with no effect on the posterior.
        return SubjectOutcomeEffect(
            outcome=outcome,
            kind=SubjectEffectKind.VARIANCE_PARTITION,
            scale_name=scale_name,
            sigma=float(value),
        )

    return SubjectOutcomeEffect(
        outcome=outcome,
        kind=SubjectEffectKind.CONSTANT,
        scale_name=scale_name,
        sigma=float(value),
    )


def resolve(definition) -> SubjectEffectPlan:
    """The child-effect plan ``definition`` implies, with every rule applied.

    Works for both definition shapes: a single-outcome definition (``use_subject_re``)
    resolves to one entry keyed ``""``, a bivariate one (``use_subject_re_u`` /
    ``use_subject_re_q``) to two, keyed ``"u"`` and ``"q"`` in build order.

    Raises ``ValueError`` for every combination the engines refuse. Each is a
    configuration that would otherwise fit something other than what was asked
    for, silently, and each is refused here rather than part-way through
    building a graph.
    """
    outcomes = tuple(
        suffix
        for suffix in OUTCOME_SUFFIXES
        if hasattr(definition, f"use_subject_re_{suffix}")
    )
    if outcomes:
        active = {
            suffix: bool(getattr(definition, f"use_subject_re_{suffix}"))
            for suffix in outcomes
        }
    elif hasattr(definition, "use_subject_re"):
        outcomes = (UNIVARIATE_OUTCOME,)
        active = {UNIVARIATE_OUTCOME: bool(definition.use_subject_re)}
    else:
        # A definition with no child-effect seam at all -- VG14, the trivariate
        # signing baseline. An empty plan rather than one claiming a single
        # inactive outcome the model does not have.
        outcomes, active = (), {}

    partition = getattr(definition, "subject_variance_partition", None)
    if partition is not None and not any(active.values()):
        raise ValueError(
            "subject_variance_partition is set but no outcome carries a child "
            "effect: the budget it partitions has nothing to allocate."
        )

    effects = tuple(
        _outcome_effect(
            definition, outcome, active=active[outcome], variance_partition=partition
        )
        for outcome in outcomes
    )
    plan = SubjectEffectPlan(
        effects=effects,
        slope_ref_age_months=float(
            getattr(definition, "subject_slope_ref_age_months", None)
            or DEFAULT_SLOPE_REF_AGE_MONTHS
        ),
        variance_partition=partition,
    )

    plan = _with_correlation(plan, definition)
    plan = _with_factor(plan, definition)
    return plan


def _with_correlation(plan: SubjectEffectPlan, definition) -> SubjectEffectPlan:
    """Attach VG20's correlation, refusing every combination that is not it.

    Read through ``getattr`` because the field lives on a definition subclass,
    as the variance partition and the child slope do: putting it on
    ``BivariateModelDefinition`` would change the serialised definition of the
    six bivariate models of record and invalidate every one of their fits.
    """
    eta = getattr(definition, "subject_re_correlation_eta", None)
    if eta is None:
        return plan
    if not {"u", "q"} <= {effect.outcome for effect in plan.effects}:
        raise ValueError(
            "subject_re_correlation_eta is set on a definition with no u and q "
            "child-effect seams; it correlates those two blocks and there is "
            "nothing here to correlate."
        )

    if not (plan["u"].is_active and plan["q"].is_active):
        raise ValueError(
            "subject_re_correlation_eta requires use_subject_re_u=True and "
            "use_subject_re_q=True: a correlation needs both subject blocks."
        )
    if SubjectEffectKind.AGE_VARYING in plan.kinds:
        raise ValueError(
            "subject_re_correlation_eta cannot be combined with an age-varying "
            "subject scale (Proposal A1): the age-varying path scales each "
            "child's deviate per observation, so a single constant correlation "
            "between the blocks is not well defined."
        )
    if SubjectEffectKind.CHILD_SLOPE in plan.kinds:
        raise ValueError(
            "subject_re_correlation_eta cannot be combined with a child slope "
            "(VG19): each outcome then carries its own 2x2 intercept/slope "
            "covariance, so correlating the two outcomes is a 4x4 design and "
            "not this one constant. Supporting it needs its own Gate 1 rather "
            "than an implicit reading of `rho_uq` as the intercept-intercept "
            "element."
        )
    if not isinstance(eta, (int, float)) or not math.isfinite(eta) or eta <= 0:
        raise ValueError(
            f"subject_re_correlation_eta must be a positive finite number; got {eta!r}."
        )
    return replace(plan, correlation_eta=float(eta))


def _with_factor(plan: SubjectEffectPlan, definition) -> SubjectEffectPlan:
    """Attach VG22's low-rank factor, refusing every combination that is not it."""
    factor = subject_factor_spec(getattr(definition, "subject_factor", None))
    if factor is None:
        return plan
    if not {"u", "q"} <= {effect.outcome for effect in plan.effects}:
        raise ValueError(
            "subject_factor is set on a definition with no u and q child-effect "
            "seams; the factor spans those two outcomes' effects."
        )

    if not (plan["u"].is_active and plan["q"].is_active):
        raise ValueError(
            "subject_factor requires use_subject_re_u and use_subject_re_q: the "
            "form is a joint covariance over both outcomes' child effects and is "
            "undefined when only one outcome carries a child effect."
        )
    if SubjectEffectKind.AGE_VARYING in plan.kinds:
        raise ValueError(
            "subject_factor cannot be combined with an age-varying subject "
            "scale (A1): both claim the same seam, and A1's rank-one scaling is "
            "a special case of the factor form."
        )
    if SubjectEffectKind.CHILD_SLOPE in plan.kinds:
        raise ValueError(
            "subject_factor cannot be combined with a child slope (VG19): the "
            "factor already carries a rate per outcome, and supplying both "
            "would give a child two of each effect."
        )
    if plan.correlation_eta is not None:
        raise ValueError(
            "subject_factor cannot be combined with subject_re_correlation_eta "
            "(VG20): the level-level correlation that field creates is an "
            "element of the factor's own covariance, emitted as `rho_uq`."
        )
    return replace(
        plan,
        factor=factor,
        effects=tuple(
            replace(effect, kind=SubjectEffectKind.FACTOR)
            if effect.outcome in ("u", "q")
            else effect
            for effect in plan.effects
        ),
    )
