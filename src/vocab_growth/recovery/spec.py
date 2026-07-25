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


_UNIVARIATE_RE = (
    "vocab_growth.models.common_univariate_re:univariate_re_stages",
    UNIVARIATE_RE_SPEC,
)
_BIVARIATE_RE = (
    "vocab_growth.models.common_bivariate_re:bivariate_re_stages",
    BIVARIATE_RE_SPEC,
)
_JOINT = ("vocab_growth.models.common_joint_modality:joint_stages", JOINT_SPEC)

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
_TARGETS: dict[str, tuple[str, EngineRecoverySpec]] = {
    "vg07": _BIVARIATE_RE,
    "vg08": _BIVARIATE_RE,
    "vg09": _BIVARIATE_RE,
    "vg10": _BIVARIATE_RE,
    "vg11": _UNIVARIATE_RE,
    "vg12": _UNIVARIATE_RE,
    "vg13": _BIVARIATE_RE,
    "vg15": _JOINT,
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

# The three models #163 names as preferred/headline, in reporting order.
HEADLINE_MODELS: tuple[str, ...] = ("vg10", "vg12", "vg15")


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
    stages_factory, spec = _TARGETS[model_key]
    return RecoveryTarget(
        model_key=model_key,
        spec=spec,
        stages_factory=stages_factory,
        engine_module=spec.engine,
    )


def outcome_column(definition, column: str) -> str:
    """Resolve the ``__outcome__`` placeholder to a definition's outcome column.

    The univariate engine fits whichever single outcome its definition names
    (VG11 spoken, VG12 understood), so its spec cannot hard-code a column.
    """
    if column == "__outcome__":
        return definition.outcome.value
    return column
