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
# VG16 is absent for a substantive reason, not convenience. Its cross-lag
# predictor is built from each child's *earlier-wave comprehension count*, so the
# design matrix is a function of the outcome. Simulating comprehension for every
# wave at once would generate the data under real-data lags but fit it under
# synthetic-data lags, and the resulting "recovery failure" would be an artefact
# of the harness rather than of the model. A correct VG16 check needs
# wave-sequential simulation; until that exists VG16 stays unsupported.
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
}

UNSUPPORTED_REASONS: dict[str, str] = {
    "vg16": (
        "the cross-lag predictor is a function of earlier-wave comprehension, so "
        "forward simulation must proceed wave by wave; single-pass simulation "
        "would fit synthetic-lag data against real-lag truth"
    ),
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
