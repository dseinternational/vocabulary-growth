# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Engine likelihoods, frame columns and dependency order for recovery simulation.

Nested spoken and signed outcomes use observed comprehension as a denominator.
Draw comprehension first, then rebuild the graph from simulated counts before
sampling dependent outcomes. Outcome-dependent lag predictors may also require
wave order. The simulator checks recorded denominators, row membership and
predictors against those rebuilt from the finished frame.
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
    """A parent count and its conditional child outcome.

    The simulator compares recorded denominators and conditional-row flags with
    those rebuilt from the synthetic frame. A mismatch rejects the simulation.
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
    """A predictor built from outcome counts, with its consuming likelihood nodes.

    Unlike fixed ages or study codes, earlier-wave counts change during
    simulation. single_pass_is_sound compares their producing and consuming
    stages to decide whether the simulator also needs a wave loop.
    """

    name: str
    source_column: str
    """The outcome column the predictor is built from."""
    consumer_rv_names: tuple[str, ...]
    """Likelihood nodes whose density the predictor shifts."""
    description: str


def outcome_dependent_predictor(definition) -> OutcomeDependentPredictor | None:
    """Return the definition's declared outcome-dependent predictor, if present.

    Register new outcome-based predictors here and add a state reader in
    simulate._PREDICTOR_STATE. Coherence checks cover declared predictors;
    they cannot discover a predictor omitted from this declaration.
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
        # Signing and its consumers share a stage, so single_pass_is_sound
        # selects wave order. Cross-tabs derive signed totals from their cells;
        # the coherence guard checks the complete reconstructed predictor.
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
    """Whether each predictor source is final before its consumers are drawn.

    Rebuilding between stages suffices when the source is drawn strictly before
    every consumer, as for VG16's comprehension lag. A same-stage source, as in
    VG25's signing lag, needs wave order. Sources not simulated remain fixed.
    Unknown consumers conservatively select the wave loop.
    """
    source_stage = _stage_producing_column(spec, definition, predictor.source_column)
    if source_stage is None:
        # Nothing simulates the source, so it is real data in every pass and the
        # predictor is the same one the refit will see.
        return True
    consumers = [_stage_consuming_rv(spec, rv) for rv in predictor.consumer_rv_names]
    if any(stage is None for stage in consumers):
        # Unknown consumers cannot justify the cheaper staged path.
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


# Models supported by the declared likelihood specs. Their prior and child-effect
# differences remain in the engine graph sampled at each fixed truth. Stage
# factories come from models.catalogue so simulation and refitting share an engine.
_TARGETS: dict[str, EngineRecoverySpec] = {
    "vg07": BIVARIATE_RE_SPEC,
    "vg08": BIVARIATE_RE_SPEC,
    "vg09": BIVARIATE_RE_SPEC,
    "vg10": BIVARIATE_RE_SPEC,
    "vg11": UNIVARIATE_RE_SPEC,
    "vg12": UNIVARIATE_RE_SPEC,
    "vg13": BIVARIATE_RE_SPEC,
    "vg16": BIVARIATE_RE_SPEC,
    "vg15": JOINT_SPEC,
    "vg19": BIVARIATE_RE_SPEC,
    "vg20": BIVARIATE_RE_SPEC,
    "vg21": BIVARIATE_RE_SPEC,
    "vg22": BIVARIATE_RE_SPEC,
    "vg23": BIVARIATE_RE_SPEC,
    "vg26": BIVARIATE_RE_SPEC,
    "vg24": JOINT_SPEC,
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
    """Require one recovery target or exclusion reason for every registered model.
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

# Headline selection follows the current reporting roles in the model inventory.
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
