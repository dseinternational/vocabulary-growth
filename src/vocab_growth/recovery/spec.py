# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What each engine needs in order to be forward-simulated (issue #163).

A parameter-recovery check regenerates a dataset *from the model itself* at a
known parameter draw, refits, and asks whether the truth is recovered. The only
model-specific knowledge that requires is: which likelihood nodes exist, which
analysis-frame column each one fills, and in what order they must be drawn.
That knowledge is declared here as data — everything else in
:mod:`vocab_growth.recovery` is engine-agnostic.

Simulation order matters because the vocabulary likelihoods are *nested*: words
spoken and words signed are modelled conditionally on the child's observed
comprehension total, so the parent count must be drawn first and the dependent
outcomes drawn against the *simulated* parent. Each entry in
:attr:`EngineRecoverySpec.stages` is one such round; the simulator rebuilds the
model between rounds so the engine's own build code re-derives the row
denominators from the synthetic parent (see :mod:`vocab_growth.recovery.simulate`).

Row denominators and likelihood-row membership are never recomputed here. They
are read back from the built model's ``pm.Data`` containers — the same arrays the
likelihood consumes — so the simulator and the fitted model cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vocab_growth.models.catalogue import engine_for
from vocab_growth.models.definitions import MODEL_REGISTRY


@dataclass(frozen=True)
class CountOutcome:
    """A count likelihood node and the frame column its draws are written to."""

    rv_name: str
    """Observed random-variable name, e.g. ``"y_u_obs"``."""
    column: str
    """Analysis-frame column the simulated counts replace, e.g. ``"understood"``."""
    row_mask_data: str | None = None
    """``pm.Data`` 0/1 mask over ``obs_id`` selecting this node's likelihood rows.
    ``None`` means the node covers every prepared row (the univariate engine)."""


@dataclass(frozen=True)
class CompositionOutcome:
    """A Dirichlet-Multinomial cross-tab node and the cell columns it fills."""

    rv_name: str
    """Observed random-variable name, e.g. ``"cells_obs"``."""
    columns: tuple[str, ...]
    """Cell columns in the node's own cell order (must match the build's stack)."""
    row_mask_data: str
    """``pm.Data`` 0/1 mask over ``obs_id`` selecting the cross-tab rows."""
    total_column: str
    """Column holding the row total; rewritten to the drawn cell sum so the
    frame's total and its cells cannot disagree."""


@dataclass(frozen=True)
class NestedLink:
    """A child outcome modelled conditionally on a parent count.

    The simulator asserts, after simulation, that the engine rebuilt from the
    synthetic frame classifies exactly the rows the simulation drew
    conditionally, with exactly the denominators it used. That assertion is the
    guard that keeps the generating process and the fitted likelihood identical.
    """

    parent_column: str
    child_column: str
    trials_data: str
    """``pm.Data`` holding the per-row denominator for this outcome's likelihood."""
    is_conditional_data: str
    """``pm.Data`` holding the per-row 0/1 nested-vs-marginal flag."""


@dataclass(frozen=True)
class EngineRecoverySpec:
    """How to forward-simulate one engine's outcomes, in dependency order."""

    engine: str
    """Engine module basename, for provenance and error messages."""
    stages: tuple[tuple[CountOutcome | CompositionOutcome, ...], ...]
    """Ordered simulation rounds. Nodes within a round are conditionally
    independent given the parameters and the earlier rounds' draws."""
    nested_links: tuple[NestedLink, ...] = ()
    """Parent/child couplings to re-verify against the rebuilt model."""
    totals_tracking_parent: tuple[tuple[str, str], ...] = ()
    """``(total_column, parent_column)`` pairs where the total *is* the parent
    count for the rows that carry it. The uk_02 four-cell total is the
    authoritative comprehension total for its rows, so once comprehension is
    simulated the total must follow it before the cross-tab is drawn."""
    conditioned_totals: tuple[str, ...] = ()
    """Row totals the model conditions on rather than generates (the nz_01
    produced total). They are carried through unchanged, and recorded here so
    the provenance record states plainly what was held fixed."""


@dataclass(frozen=True)
class OutcomeDependentPredictor:
    """A model term whose design matrix is built from an outcome column.

    Almost every predictor here is a function of the *design* -- age, study,
    child -- so drawing every row of an outcome at once is exactly what the model
    says happens. A cross-lag is not: it reads an earlier wave's **count**, so
    the design matrix is a function of the outcome, and the order in which the
    simulator draws things starts to matter.

    Whether it matters is :func:`single_pass_is_sound`'s question, not a matter
    of declaration -- see there. What is declared here is only what the predictor
    reads and what it enters, which the engines compute inside the build and
    nothing can infer from outside it.
    """

    name: str
    source_column: str
    """The outcome column the predictor is built from."""
    consumer_rv_names: tuple[str, ...]
    """Likelihood nodes whose density the predictor shifts."""
    description: str


def outcome_dependent_predictor(definition) -> OutcomeDependentPredictor | None:
    """``definition``'s outcome-dependent predictor, if it has one.

    **A new predictor that reads an outcome column belongs here.**
    ``tests/test_recovery_wave_sequential.py`` pins the one that exists against
    the definition field that creates it, and the simulator's guard refuses any
    simulation whose predictor is not reproducible from the finished frame -- so
    an undeclared one that actually mattered would abort the run rather than
    quietly produce a wrong dataset.
    """
    if getattr(definition, "use_cross_lag", False):
        return OutcomeDependentPredictor(
            name="cross_lag",
            source_column="understood",
            consumer_rv_names=("y_s_obs",),
            description=(
                "each child's earlier-wave comprehension count, shifting the "
                "logit of their current production ratio"
            ),
        )
    if getattr(definition, "use_sign_cross_lag", False):
        # VG25 (#297), and the case `single_pass_is_sound` was written for. The
        # source is `signed`, drawn in the joint engine's SECOND stage, and the
        # consumers are drawn in that same stage -- so one pass would draw each
        # consumer against a source still holding its real study value, and the
        # wave loop is selected. Nothing here declares that; the derivation does.
        #
        # `signed` names the marginal column, and the cross-tab rows carry their
        # signed total in `signed_only` + `signed_spoken` instead. Both are drawn
        # in stage 1 -- `y_sign_obs` and `cells_obs` are in the same tuple -- so
        # either spelling selects the same behaviour, and the guard compares the
        # WHOLE predictor against the finished frame however it was assembled.
        # The consumers are named in full regardless, because
        # `single_pass_is_sound` fails toward the loop on any it cannot find and
        # a silently unrecognised name would read as a decision.
        consumers = ["y_s_obs"]
        if getattr(definition, "sign_lag_in_cells", True):
            consumers += ["cells_obs", "nz_prod_cells_obs"]
        return OutcomeDependentPredictor(
            name="sign_cross_lag",
            source_column="signed",
            consumer_rv_names=tuple(consumers),
            description=(
                "each child's earlier-wave signed share of comprehension, "
                "shifting the logit of their current production ratio"
            ),
        )
    return None


def _stage_producing_column(spec, definition, column: str) -> int | None:
    """Index of the stage that draws ``column``, or ``None`` if nothing does."""
    for index, stage in enumerate(spec.stages):
        for node in stage:
            if isinstance(node, CountOutcome):
                if outcome_column(definition, node.column) == column:
                    return index
            elif column in node.columns or column == node.total_column:
                return index
    return None


def _stage_consuming_rv(spec, rv_name: str) -> int | None:
    """Index of the stage that draws ``rv_name``, or ``None`` if nothing does."""
    for index, stage in enumerate(spec.stages):
        if any(node.rv_name == rv_name for node in stage):
            return index
    return None


def single_pass_is_sound(spec, definition, predictor) -> bool:
    """Whether every row's predictor is final before anything using it is drawn.

    The simulator already rebuilds the model between stages, so a predictor built
    from an outcome drawn in a **strictly earlier** stage is recomputed from the
    simulated value before any row that uses it is drawn -- which is a correct
    forward simulation, in one pass, with no wave loop needed.

    **This corrects the record.** VG16 was listed as unsupported on the ground
    that "single-pass simulation would fit synthetic-lag data against real-lag
    truth" (#242 item 6, #289 task 3.9, and the deferral of #297). Measured on
    2026-09-11, that is false for VG16: its lag reads ``understood``, drawn in
    stage 0, and enters ``y_s_obs``, drawn in stage 1, so at the build whose draw
    consumes the lag the predictor already matched the finished frame's in **0 of
    1,708 rows**. The blocker was a misdiagnosis, and the wave loop it asked for
    is not what VG16 needed.

    It is what a predictor reading a **same-stage** outcome needs, which is the
    shape #297's proposed VG25 has: a sign-to-speech lag reads ``signed``, drawn
    alongside ``spoken`` in the joint engine's second stage, so a single pass
    would draw the consumer against a source that is still real. There the wave
    loop is required, and this function is what selects it.
    """
    source_stage = _stage_producing_column(spec, definition, predictor.source_column)
    if source_stage is None:
        # Nothing simulates the source, so it is real data in every pass and the
        # predictor is the same one the refit will see.
        return True
    consumers = [_stage_consuming_rv(spec, rv) for rv in predictor.consumer_rv_names]
    if any(stage is None for stage in consumers):
        # A declared consumer this engine does not draw: fail toward the wave
        # loop, which is correct either way and only costs time.
        return False
    return all(source_stage < stage for stage in consumers)


# --------------------------------------------------------------------------
# Engine specifications
# --------------------------------------------------------------------------

UNIVARIATE_RE_SPEC = EngineRecoverySpec(
    engine="common_univariate_re",
    # One outcome, no nesting: the whole dataset is a single round.
    stages=((CountOutcome(rv_name="y_obs", column="__outcome__"),),),
)

BIVARIATE_RE_SPEC = EngineRecoverySpec(
    engine="common_bivariate_re",
    stages=(
        (CountOutcome(rv_name="y_u_obs", column="understood", row_mask_data="obs_u_mask"),),
        (CountOutcome(rv_name="y_s_obs", column="spoken", row_mask_data="obs_s_mask"),),
    ),
    nested_links=(
        NestedLink(
            parent_column="understood",
            child_column="spoken",
            trials_data="s_likelihood_n",
            is_conditional_data="s_is_conditional",
        ),
    ),
)

JOINT_SPEC = EngineRecoverySpec(
    engine="common_joint_modality",
    stages=(
        (CountOutcome(rv_name="y_u_obs", column="understood", row_mask_data="obs_u_mask"),),
        (
            CountOutcome(rv_name="y_s_obs", column="spoken", row_mask_data="obs_s_mask"),
            CountOutcome(rv_name="y_sign_obs", column="signed", row_mask_data="obs_sign_mask"),
            CompositionOutcome(
                rv_name="cells_obs",
                columns=("understood_only", "signed_only", "spoken_only", "signed_spoken"),
                row_mask_data="obs_cells_mask",
                total_column="cell_total",
            ),
            CompositionOutcome(
                rv_name="nz_prod_cells_obs",
                columns=("prod_signed_only", "prod_spoken_only", "prod_signed_spoken"),
                row_mask_data="obs_prod_mask",
                total_column="prod_total",
            ),
        ),
    ),
    nested_links=(
        NestedLink(
            parent_column="understood",
            child_column="spoken",
            trials_data="s_likelihood_n",
            is_conditional_data="s_is_conditional",
        ),
        NestedLink(
            parent_column="understood",
            child_column="signed",
            trials_data="sign_likelihood_n",
            is_conditional_data="sign_is_conditional",
        ),
    ),
    totals_tracking_parent=(("cell_total", "understood"),),
    conditioned_totals=("prod_total",),
)


@dataclass(frozen=True)
class RecoveryTarget:
    """A model that can be forward-simulated, with its engine's plumbing."""

    model_key: str
    spec: EngineRecoverySpec
    stages_factory: str
    """``module:function`` returning the engine's fit-pipeline stage list. Held as
    a string so this module stays import-light (the engines pull in PyMC)."""
    engine_module: str = field(default="")

    def resolve_stages(self, definition):
        """Import and call the engine's stage factory for ``definition``."""
        import importlib

        module_name, _, function_name = self.stages_factory.partition(":")
        module = importlib.import_module(module_name)
        return getattr(module, function_name)(definition)


# Models whose data-generating process this harness can reproduce exactly.
#
# VG01-VG05 and VG14 are deliberately absent: they are descriptive baselines on
# the non-RE engines, not estimands #163 gates, and adding their engines would
# widen the surface without adding evidence.
#
# Only the data-generating spec is named here. The engine's stage factory comes
# from `vocab_growth.models.catalogue`, so a model cannot be paired with another
# engine's pipeline (issue #273).
_TARGETS: dict[str, EngineRecoverySpec] = {
    "vg07": BIVARIATE_RE_SPEC,
    "vg08": BIVARIATE_RE_SPEC,
    "vg09": BIVARIATE_RE_SPEC,
    "vg10": BIVARIATE_RE_SPEC,
    "vg11": UNIVARIATE_RE_SPEC,
    "vg12": UNIVARIATE_RE_SPEC,
    "vg13": BIVARIATE_RE_SPEC,
    # VG16 runs VG10's engine and draws counts exactly as VG10 does -- the
    # cross-lag changes the q logit, not the observation nodes -- so the VG10
    # spec is correct here unchanged, as it is for VG19 to VG23 below.
    #
    # It was listed as unsupported until 2026-09-11 on a premise that measurement
    # did not support: that a single pass "would fit synthetic-lag data against
    # real-lag truth". The simulator already rebuilds between stages, and VG16's
    # lag reads `understood` (stage 0) while entering `y_s_obs` (stage 1), so the
    # lag was already recomputed from the simulated parent before the draw that
    # uses it -- 0 of 1,708 rows differing from the finished frame's lag. See
    # `single_pass_is_sound`, which now derives that rather than asserting it,
    # and refuses the case where it does not hold.
    #
    # `beta_lag` needs no scoring entry: it is a scalar free RV, so
    # `recovery/compare.py` picks it up by dimension.
    "vg16": BIVARIATE_RE_SPEC,
    "vg15": JOINT_SPEC,
    # VG19 runs VG10's engine and VG10's data-generating process. The child slope
    # changes the PRIOR on each child's effect -- one deviate becomes an
    # intercept/rate pair -- but not how counts are drawn, and the simulator
    # samples the observation nodes from the real graph at a fixed truth draw, so
    # the slope is carried without a spec change. Same reasoning as VG20 below.
    #
    # Scoring needs no entry either: `tau_subj_*_0`, `tau_subj_*_1` and
    # `tau_subj_*_rho` are scalar, so `recovery/compare.py` picks them up by
    # dimension; `tau_subj_*_rho_raw` is excluded by the `*_raw` rule; and
    # `tau_subj_*_z` carries two dims (subject_id, child_effect), so it falls
    # through both branches of `target_variables` rather than erroring.
    #
    # Registered because G3 of the VG19 plan is a recovery check, and without an
    # entry `fit_recovery.py vg19` fails with the generic "no recovery
    # specification registered". `tau1` is the parameter the check exists for:
    # the plan predicts it is weakly identified, and recovery at `test` is how
    # that is measured rather than assumed.
    "vg19": BIVARIATE_RE_SPEC,
    # VG20 runs the same engine and the same data-generating process as VG10 --
    # the correlation changes the PRIOR on the pair of subject deviates, not how
    # counts are simulated -- so the VG10 spec is correct here unchanged. It is
    # registered because #224's second gate is recovery of `rho_uq` itself, and
    # without an entry `fit_recovery.py vg20` fails with the generic "no recovery
    # specification registered" rather than running. `rho_uq` needs no scoring
    # entry of its own: it is a scalar Deterministic, so the target selection in
    # recovery/compare.py picks it up, while `rho_uq_raw` is excluded by the
    # existing `*_raw` rule as a non-centred offset with no interpretation.
    "vg20": BIVARIATE_RE_SPEC,
    # VG21, VG22 and VG23 all run the bivariate-RE engine and all draw counts the
    # same way VG10 does, so the VG10 spec is correct for each unchanged -- the same
    # argument as VG19 and VG20 above, and for the same reason: what these models
    # change is the PRIOR on the per-child effects, not the observation nodes the
    # simulator samples from. VG21 is a plain `BivariateModelDefinition` on that
    # engine, as VG13 is. VG23 shares VG20's definition class exactly.
    #
    # VG22 is the one that most needs a recovery check rather than least: a rank-k
    # factor over four child effects is the weakly identified structure the harness
    # exists to measure, and #283's rank-2/rank-3 disagreement on the spoken slope
    # scale is precisely a "is this identified?" question. Its parameters need no
    # scoring entries -- `target_variables` selects by dimension, so the scalar
    # `tau_subj_*_0` / `tau_subj_*_1` are picked up while the two-dimensional
    # `subject_factor_loadings` and `subject_factor_z` fall through both branches
    # rather than erroring, exactly as VG19's `tau_subj_*_z` does.
    "vg21": BIVARIATE_RE_SPEC,
    "vg22": BIVARIATE_RE_SPEC,
    "vg23": BIVARIATE_RE_SPEC,
    # VG24 is to VG15 what VG20 is to VG10, and the argument carries across
    # unchanged: the correlation changes the PRIOR on a child's three deviates,
    # not how any count is drawn, and the simulator samples the observation nodes
    # from the real graph at a fixed truth draw. So JOINT_SPEC is correct here
    # with no change -- the marginals, both compositions and both nested links
    # are VG15's.
    #
    # The three correlations need no scoring entries: `rho_uq`, `rho_u_sign` and
    # `rho_sign_q` are scalar Deterministics, so `recovery/compare.py` picks them
    # up by dimension, exactly as VG20's `rho_uq` is. `subject_re_corr` carries
    # two dims and falls through both branches of `target_variables` rather than
    # erroring, as VG22's `subject_factor_loadings` does.
    #
    # Registered rather than deferred because `rho_sign_q` is the whole point of
    # the model and is identified by the children carrying both a signed and a
    # spoken marginal -- a subset of the frame, not all of it. Whether that
    # subset identifies it is a recovery question, and recovery is how it gets
    # measured instead of assumed.
    "vg24": JOINT_SPEC,
    # VG25 runs VG24's engine and VG24's data-generating process: the lag shifts
    # the `q` logit, not which nodes exist or what they are drawn from, so
    # JOINT_SPEC is correct here unchanged -- the same argument VG16 carries
    # against BIVARIATE_RE_SPEC.
    #
    # What is NOT the same is the simulation order, and this is the model the
    # wave loop was built for. Its lag reads `signed`, which the joint engine
    # draws in the SAME stage as the `spoken` it shifts, so a single pass would
    # draw each consumer against a source still holding its real value.
    # `single_pass_is_sound` derives that from the stage order and selects the
    # loop; `simulate._verify_predictor_coherence` then checks, on every run,
    # that each row was drawn under the predictor the finished frame implies.
    #
    # `beta_sign_lag` needs no scoring entry: it is a scalar free RV, so
    # `recovery/compare.py` picks it up by dimension, exactly as `beta_lag` is.
    # The three correlations it inherits from VG24 are picked up the same way.
    #
    # Registered rather than deferred because gate 4 of #297 is recovery of this
    # coefficient in three designed cells -- `(beta = 0, rho != 0)`,
    # `(beta != 0, rho = 0)` and both nonzero -- whose whole purpose is to show
    # the lag can be told apart from the correlation. All three are runnable:
    # the third is a draw, and the other two are settings, which
    # `recovery.truth_overrides` supplies as `--set-truth beta_sign_lag=0` and
    # `--set-truth subject_re=independent`. The second is a transform rather
    # than a number because the correlations are deterministics read off a
    # packed Cholesky factor, so there is no correlation to set; setting the
    # factor to its diagonal zeroes all three and keeps the scales. VG25 needs
    # no comparator model for these cells -- unlike VG16, it carries both the
    # lag and the correlated block, so the three cells are settings of one
    # model.
    "vg25": JOINT_SPEC,
}

UNSUPPORTED_REASONS: dict[str, str] = {
    "vg01": "descriptive baseline on the single-outcome engine (not a gated estimand)",
    "vg02": "descriptive baseline on the single-outcome engine (not a gated estimand)",
    "vg03": "descriptive baseline on the single-outcome engine (not a gated estimand)",
    "vg04": "descriptive baseline on the single-outcome engine (not a gated estimand)",
    "vg05": "descriptive baseline on the non-RE bivariate engine (superseded by VG10)",
    "vg14": "signing baseline on the trivariate engine (superseded by VG15)",
}


def _require_complete_coverage() -> None:
    """Every registered model is either a recovery target or has a recorded reason.

    Without this, registering a model leaves it in neither table and
    `recovery_target` falls through to the generic "no recovery specification
    registered" -- which reads as a considered decision when nobody decided
    anything. That is exactly what happened to VG21, VG22 and VG23, and it is the
    failure mode the two blocks of prose above were written to avoid. This is the
    same both-directions completeness check `catalogue._catalogue` makes, for the
    same reason (#273).
    """
    registered = set(MODEL_REGISTRY)
    covered = set(_TARGETS) | set(UNSUPPORTED_REASONS)
    missing = sorted(registered - covered)
    if missing:
        raise RuntimeError(
            "Registered models absent from both _TARGETS and UNSUPPORTED_REASONS: "
            f"{', '.join(missing)}. Add a recovery spec, or a recorded reason why "
            "recovery does not apply -- not nothing."
        )
    unknown = sorted(covered - registered)
    if unknown:
        raise RuntimeError(
            "Recovery tables name models that are not registered: "
            f"{', '.join(unknown)}. Remove them, or register the model."
        )
    both = sorted(set(_TARGETS) & set(UNSUPPORTED_REASONS))
    if both:
        raise RuntimeError(
            f"Models in both _TARGETS and UNSUPPORTED_REASONS: {', '.join(both)}. "
            "A model is either supported or explained, not both."
        )


_require_complete_coverage()

# The three models #163 names as preferred/headline, in reporting order. VG20
# replaced VG10 here on 2026-08-19 when it took over as the model of record for
# the Down syndrome joint understood + spoken estimands (#224); this list tracks
# the models that carry reporting weight, so it follows the role, not the
# lineage. VG10 stays a recovery target, it is simply no longer a headline one.
HEADLINE_MODELS: tuple[str, ...] = ("vg20", "vg12", "vg15")


def supported_models() -> list[str]:
    """Recovery-capable model keys, in registry order."""
    return [key for key in MODEL_REGISTRY if key in _TARGETS]


def recovery_target(model_key: str) -> RecoveryTarget:
    """Resolve one model's recovery plumbing, or explain why it has none."""
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {model_key!r}.")
    if model_key not in _TARGETS:
        reason = UNSUPPORTED_REASONS.get(model_key, "no recovery specification registered")
        raise KeyError(
            f"Parameter recovery is not supported for {model_key!r}: {reason}. "
            f"Supported models: {', '.join(supported_models())}."
        )
    engine = engine_for(model_key)
    if engine.stages is None:
        raise KeyError(
            f"Parameter recovery is not supported for {model_key!r}: its engine "
            f"({engine.name}) exposes no stage factory, so the harness cannot "
            "substitute the simulated-frame loader for stage 0."
        )
    return RecoveryTarget(
        model_key=model_key,
        spec=_TARGETS[model_key],
        stages_factory=f"{engine.module}:{engine.stages}",
        engine_module=engine.module.rpartition(".")[2],
    )


def outcome_column(definition, column: str) -> str:
    """Resolve the ``__outcome__`` placeholder to a definition's outcome column.

    The univariate engine fits whichever single outcome its definition names
    (VG11 spoken, VG12 understood), so its spec cannot hard-code a column.
    """
    if column == "__outcome__":
        return definition.outcome.value
    return column
