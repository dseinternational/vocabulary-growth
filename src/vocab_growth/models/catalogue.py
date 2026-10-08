# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Engine and reporting declarations for every registered model.

The catalogue records each model's engine, hooks, template and reporting role.
Engine choice cannot be inferred from the definition class: VG05 and VG07
share a class but use different engines. Tests check the declarations against
wrappers, the registry and the documented roles.

These declarations stay outside the serialised statistical definition.
Executable signatures separately detect changes to engines and other package
code. Hooks are named as strings and imported on demand, so reading the
catalogue does not import the PyMC engines.
"""

from __future__ import annotations

import enum
import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from vocab_growth.models.definitions import MODEL_REGISTRY, ModelDefinition

# ============================================================
# Engines
# ============================================================


@dataclass(frozen=True)
class EngineAdapter:
    """Names and calling conventions for one fitting engine.

    Hook fields name module attributes resolved on demand. The *_call fields
    describe how to invoke those hooks; name and explanatory fields are metadata.
    """

    name: str
    """Stable engine key, used in messages and in derived dispatch tables."""

    module: str
    """Dotted module path used to resolve the declared hooks."""

    prepare: str
    """Data-preparation stage; called ``f(context, definition)``."""

    priors: str
    """Prior/hyperparameter configuration stage; called ``f(context, definition)``."""

    build: str
    """Model-build stage; called ``f(context, definition)``."""

    prior_checks: str
    """Prior-predictive stage."""

    prior_checks_call: str
    """Calling convention for the prior-predictive hook.

    outcome passes an outcome column and label; definition passes the model
    definition; context passes only the context. Both bivariate engines use
    definition, so child checks can read the selected effect structure.
    """

    frame_builder: str
    """Pure ``build_*_analysis_frame(definition)`` for the prepared-frame hash."""

    fit: str
    """Full fit pipeline; called ``f(config, definition)`` by the wrapper module."""

    stages: str | None = None
    """``f(definition) -> [(stage name, stage fn), ...]`` for the engines that
    expose their pipeline as a factory, or ``None`` where the fit function builds
    the list inline.

    The parameter-recovery harness substitutes stage 0 (data preparation) for a
    loader that injects a simulated frame and runs the identical pipeline, so a
    factory is what makes an engine recovery-capable at all.
    """

    plots: str | None = None
    """Plot stage, or ``None`` when this engine has no out-of-fit replot path."""

    plots_call: str | None = None
    """How :attr:`plots` is invoked: ``"definition"``, ``"outcome_label"`` or
    ``"context"``. ``None`` exactly when :attr:`plots` is."""

    replot_note: str | None = None
    """Why there is no replot path, when :attr:`plots` is ``None``.

    Recorded rather than left implicit: a silent skip in a diagnostic reads as
    a pass, so ``scripts/regenerate_plots.py`` prints this reason instead.
    """

    samples_extractor: str | None = None
    """Stored-draw extractor for a replot, or None to rerun predictions.

    The choice is declared rather than inferred from imports. Without an extractor,
    the predictive stage uses the recorded sampling seed. Replot checks the rerun
    draws against stored draws because seeds alone do not guarantee reproduction
    across numerical environments.
    """

    posterior_predictive: str | None = None
    """Posterior-predictive stage, re-run on replot when :attr:`samples_extractor`
    is ``None``. Required exactly when an engine supports replot and declares no
    extractor."""

    def __post_init__(self) -> None:
        """Refuse a replot declaration that cannot be carried out.

        An engine that supports replot must say how the samples are obtained --
        either a pure extractor or a posterior-predictive stage to re-run. Neither
        would make ``regenerate_plots.py`` fail at the point of use, after it had
        already staged an output directory.
        """
        if not self.supports_replot:
            return
        if self.samples_extractor is None and self.posterior_predictive is None:
            raise ValueError(
                f"engine {self.name!r} declares a plot stage but neither "
                "'samples_extractor' nor 'posterior_predictive', so a replot has no "
                "way to obtain model samples."
            )
        if self.samples_extractor is not None and self.posterior_predictive is not None:
            raise ValueError(
                f"engine {self.name!r} declares both 'samples_extractor' and "
                "'posterior_predictive'; a replot uses exactly one, so declaring "
                "both hides which."
            )

    @property
    def supports_replot(self) -> bool:
        """Whether a fitted model on this engine can have its figures redrawn."""
        return self.plots is not None

    def resolve(self, attribute: str) -> Any:
        """Resolve the hook named by attribute.

        Raise AttributeError when the field or declared hook is absent. Missing hook
        messages identify the engine and field or module. Hook return types remain
        Any because the declared stages have different signatures.
        """
        name = getattr(self, attribute)
        if name is None:
            raise AttributeError(
                f"engine {self.name!r} declares no {attribute!r}"
                + (f" ({self.replot_note})" if self.replot_note else "")
            )
        module = importlib.import_module(self.module)
        function = getattr(module, name, None)
        if function is None:
            raise AttributeError(
                f"{self.module} has no {name!r} (engine {self.name!r}, "
                f"field {attribute!r})"
            )
        return function


#: The six fitting engines, keyed by :attr:`EngineAdapter.name`.
ENGINES: dict[str, EngineAdapter] = {
    engine.name: engine
    for engine in (
        EngineAdapter(
            name="univariate",
            module="vocab_growth.models.common",
            prepare="prepare_univariate_data",
            priors="configure_univariate_priors",
            build="build_model",
            prior_checks="prior_predictive_checks",
            prior_checks_call="outcome",
            frame_builder="build_univariate_analysis_frame",
            fit="fit_single_outcome_model",
            plots="run_standard_plots",
            plots_call="outcome_label",
            samples_extractor="extract_model_samples",
        ),
        EngineAdapter(
            name="univariate_re",
            module="vocab_growth.models.common_univariate_re",
            prepare="prepare_univariate_re_data",
            priors="configure_univariate_priors",
            build="build_univariate_re_model",
            prior_checks="prior_predictive_checks",
            prior_checks_call="outcome",
            frame_builder="build_univariate_re_analysis_frame",
            fit="fit_univariate_re_model",
            stages="univariate_re_stages",
            # Replotting this engine has not been validated. Its declared
            # posterior-predictive stage differs from the common extractor.
            replot_note=(
                "no exercised replot path for the univariate random-effect "
                "engine; VG11/VG12 figures come from a refit"
            ),
        ),
        EngineAdapter(
            name="bivariate",
            module="vocab_growth.models.common_bivariate",
            prepare="prepare_bivariate_data",
            priors="configure_bivariate_priors",
            build="build_model",
            prior_checks="prior_predictive_checks",
            prior_checks_call="definition",
            frame_builder="build_bivariate_analysis_frame",
            fit="fit_bivariate_model",
            plots="run_bivariate_joint_plots",
            plots_call="definition",
            samples_extractor="extract_model_samples",
        ),
        EngineAdapter(
            name="bivariate_re",
            module="vocab_growth.models.common_bivariate_re",
            prepare="prepare_bivariate_re_data",
            priors="configure_bivariate_priors",
            build="build_model_re",
            prior_checks="prior_predictive_checks",
            prior_checks_call="definition",
            frame_builder="build_bivariate_re_analysis_frame",
            fit="fit_bivariate_re_model",
            stages="bivariate_re_stages",
            plots="run_bivariate_joint_plots",
            plots_call="definition",
            # This engine regenerates predictions when redrawing figures.
            posterior_predictive="sample_posterior_predictive",
        ),
        EngineAdapter(
            name="trivariate",
            module="vocab_growth.models.common_trivariate",
            prepare="prepare_trivariate_data",
            priors="configure_trivariate_priors",
            build="build_model",
            prior_checks="prior_predictive_checks",
            prior_checks_call="context",
            frame_builder="build_trivariate_analysis_frame",
            fit="fit_trivariate_model",
            plots="run_trivariate_plots",
            plots_call="context",
            samples_extractor="extract_model_samples",
        ),
        EngineAdapter(
            name="joint",
            module="vocab_growth.models.common_joint_modality",
            prepare="prepare_joint_data",
            priors="configure_joint_priors",
            build="build_model",
            prior_checks="prior_predictive_checks",
            prior_checks_call="context",
            frame_builder="build_joint_analysis_frame",
            fit="fit_joint_model",
            stages="joint_stages",
            plots="run_joint_plots",
            plots_call="context",
            # The joint engine builds its samples inside the posterior-predictive
            # stage, so there is no pure extractor to declare and the whole stage
            # is re-run.
            posterior_predictive="sample_posterior_predictive",
        ),
    )
}


# ============================================================
# Roles
# ============================================================


class ModelRole(enum.Enum):
    """Declared reporting purpose and the publication checks it requires.

    Roles follow docs/models/README.md and study-owner decisions. Registration
    alone does not establish a role. An unclassified model retains full checks.
    """

    MODEL_OF_RECORD = "model-of-record"
    """The current source for a stated estimand. Must be refit-current."""

    TD_REFERENCE = "td-reference"
    """A typically-developing comparison model with a distinct population role.
    It supplies the other side of a published contrast, so it must be
    refit-current too."""

    DEVELOPMENT_STEP = "development-step"
    """Retained to show how structure was added, not preferred for headline
    estimates. Not refit-current by requirement."""

    SUPERSEDED = "superseded"
    """Replaced after a documented structural or data problem. The taxonomy's
    own rule is that it "never supplies a number in the findings"."""

    UNCLASSIFIED = "unclassified"
    """No role has been decided yet. Deliberately **fails closed**: treated as
    though it were a model of record, so adding a model or leaving one
    undecided keeps today's strictness. Relaxing a model is then an explicit,
    reviewable act rather than an omission."""

    @property
    def publication_required(self) -> bool:
        """Whether a fit of this role must pass full publication validation."""
        return self in {
            ModelRole.MODEL_OF_RECORD,
            ModelRole.TD_REFERENCE,
            ModelRole.UNCLASSIFIED,
        }


# ============================================================
# Models
# ============================================================


@dataclass(frozen=True)
class RegisteredModel:
    """Everything about a model that is not part of its statistical definition."""

    model_key: str
    """Lower-case registry key (``"vg01"``), the id every script takes on the
    command line and every output directory is named from."""

    engine: EngineAdapter
    """The engine that fits it. Declared, not inferred: VG05 and VG07 share a
    definition class and run on different engines."""

    role: ModelRole
    """What the model is for, and therefore whether it must be refit-current.
    See :class:`ModelRole`; unclassified fails closed."""


    @property
    def definition(self) -> ModelDefinition:
        """The registered statistical definition, from ``definitions.py``.

        A property rather than a stored field so the catalogue cannot hold a
        second copy of a definition that has since been re-registered.
        """
        return MODEL_REGISTRY[self.model_key]

    @property
    def wrapper_module(self) -> str:
        """Dotted path of the thin ``model_vgNN`` module exposing ``fit``."""
        return f"vocab_growth.models.model_{self.model_key}"

    @property
    def report_template(self) -> str:
        """Repository-relative Quarto source copied into the fitted output."""
        return f"docs/models/{self.model_key}/index.qmd"

    def load_wrapper(self) -> ModuleType:
        """Import and return the wrapper module."""
        return importlib.import_module(self.wrapper_module)


def _catalogue() -> dict[str, RegisteredModel]:
    engine_of = {
        "vg01": "univariate",
        "vg02": "univariate",
        "vg03": "univariate",
        "vg04": "univariate",
        "vg05": "bivariate",
        "vg07": "bivariate_re",
        "vg08": "bivariate_re",
        "vg09": "bivariate_re",
        "vg10": "bivariate_re",
        "vg11": "univariate_re",
        "vg12": "univariate_re",
        "vg13": "bivariate_re",
        "vg14": "trivariate",
        "vg15": "joint",
        "vg16": "bivariate_re",
        "vg19": "bivariate_re",
        "vg20": "bivariate_re",
        "vg21": "bivariate_re",
        "vg22": "bivariate_re",
        "vg23": "bivariate_re",
        "vg24": "joint",
        "vg25": "joint",
        "vg26": "bivariate_re",
    }
    # Sourced from the roles table in ``docs/models/README.md`` and the
    # decision notes it cites -- not inferred. A model whose role that record
    # does not state is left UNCLASSIFIED, which keeps today's strictness; see
    # ``ModelRole.UNCLASSIFIED``. Classifying one is a study-owner decision and
    # belongs in the same commit as the record that justifies it.
    role_of = {
        # Roles table: model of record for the DS joint understood+spoken
        # estimands from 2026-08-19, reaffirmed through the us_03 refit by
        # ``notes/202609031930-vg20-vg22-decision.md``.
        "vg20": ModelRole.MODEL_OF_RECORD,
        # Roles table, VG14 row: "Wholly replaced by VG15 for reporting ...
        # VG15 supplies everything VG14 does".
        "vg15": ModelRole.MODEL_OF_RECORD,
        # The typically-developing side of the published DS-vs-TD contrasts:
        # ``scripts/compare_ds_td_re.py`` names vg11/vg12 as TD_KEYS and vg21
        # as JOINT_TD_KEY.
        "vg11": ModelRole.TD_REFERENCE,
        "vg12": ModelRole.TD_REFERENCE,
        "vg21": ModelRole.TD_REFERENCE,
        # Roles table: development steps, none expected to supply a reported
        # number. VG16 is a single-purpose development model whose headline is
        # withdrawn; VG10 was model of record 2026-08-05 to 2026-08-19.
        "vg05": ModelRole.DEVELOPMENT_STEP,
        "vg07": ModelRole.DEVELOPMENT_STEP,
        "vg08": ModelRole.DEVELOPMENT_STEP,
        "vg09": ModelRole.DEVELOPMENT_STEP,
        "vg10": ModelRole.DEVELOPMENT_STEP,
        "vg14": ModelRole.DEVELOPMENT_STEP,
        "vg16": ModelRole.DEVELOPMENT_STEP,
        # Role decisions are recorded in notes/202609091600-model-roles-settled.md.
        # VG24 supplies rho_sign_q; VG15 retains the signing-trajectory role.
        "vg24": ModelRole.MODEL_OF_RECORD,
        # TD reference for the between-child understood/production correlation.
        "vg23": ModelRole.TD_REFERENCE,
        # VG21 replaced this narrow-window TD comparator. The zero-correlation
        # child block matches VG23, but VG23 also adds a sex term.
        "vg13": ModelRole.SUPERSEDED,
        # The single-outcome, single-level baselines each lineage was built
        # on: VG20 carries VG01 and VG02's estimands, VG11 and VG12 are VG03
        # and VG04 rebuilt with a hierarchy.
        "vg01": ModelRole.DEVELOPMENT_STEP,
        "vg02": ModelRole.DEVELOPMENT_STEP,
        "vg03": ModelRole.DEVELOPMENT_STEP,
        "vg04": ModelRole.DEVELOPMENT_STEP,
        # Retained for child level/rate structure checks; VG20 keeps the DS role.
        "vg19": ModelRole.DEVELOPMENT_STEP,
        "vg22": ModelRole.DEVELOPMENT_STEP,
    }
    # VG25 and VG26 have no declared role. Keep full publication checks until
    # the study owner records and assigns one.
    role_of.update(
        {key: ModelRole.UNCLASSIFIED for key in MODEL_REGISTRY if key not in role_of}
    )
    unknown_roles = sorted(set(role_of) - set(MODEL_REGISTRY))
    if unknown_roles:
        raise RuntimeError(
            f"Roles declared for unregistered models: {unknown_roles}. Remove "
            "them or register the model in MODEL_REGISTRY."
        )

    missing = sorted(set(MODEL_REGISTRY) - set(engine_of))
    if missing:
        raise RuntimeError(
            f"Registered models with no catalogue entry: {missing}. Every model "
            "in MODEL_REGISTRY needs one, or its fit dispatch, frame hash, "
            "prior audit and replot path have nothing to derive from."
        )
    unknown = sorted(set(engine_of) - set(MODEL_REGISTRY))
    if unknown:
        raise RuntimeError(
            f"Catalogue entries for unregistered models: {unknown}. Remove them "
            "or register the model in MODEL_REGISTRY."
        )
    return {
        key: RegisteredModel(
            model_key=key, engine=ENGINES[engine_of[key]], role=role_of[key]
        )
        for key in MODEL_REGISTRY
    }


#: Every registered model, in ``MODEL_REGISTRY`` order.
CATALOGUE: dict[str, RegisteredModel] = _catalogue()

# Exploratory VG17/VG18 bypass the shared manifest, staged promotion and
# convergence gate. They stay outside the registered catalogue and CLI dispatch.
# Productionising either remains a statistical decision for #266.


def get(model_key: str) -> RegisteredModel:
    """The catalogue entry for ``model_key``, case-insensitively.

    Raises ``KeyError`` naming the catalogue rather than guessing an engine: an
    unregistered model routed through a plausible-looking engine is exactly the
    failure this module exists to prevent.
    """
    entry = CATALOGUE.get(model_key.lower())
    if entry is None:
        raise KeyError(
            f"{model_key!r} is not in the model catalogue. Register it in "
            "vocab_growth.models.catalogue alongside MODEL_REGISTRY."
        )
    return entry


def engine_for(model_key: str) -> EngineAdapter:
    """The engine that fits ``model_key``."""
    return get(model_key).engine


def engine_for_definition(definition) -> EngineAdapter:
    """The engine that fits ``definition``, found by its model id.

    :func:`engine_for` takes a model key, which a *definition* does not carry: a
    fold arm, a sensitivity variant and a recovery copy are all
    ``dataclasses.replace`` of a registered definition with a modified
    ``config_name``, and the key cannot be recovered from that. ``model_id`` is
    unchanged by every such copy and is unique across the catalogue, so it is
    what identifies the engine -- which is a property of the model, not of the
    arm.

    Raises rather than guessing: a definition whose ``model_id`` is not
    registered has no engine, and defaulting one would fit it with somebody
    else's builder.
    """
    model_id = getattr(definition, "model_id", None)
    for key, registered in CATALOGUE.items():
        if registered.definition.model_id == model_id:
            return engine_for(key)
    known = ", ".join(sorted(r.definition.model_id for r in CATALOGUE.values()))
    raise KeyError(
        f"No registered model has model_id {model_id!r}, so its engine is "
        f"unknown. Registered: {known}."
    )


def publication_models() -> list[str]:
    """Registry keys whose roles require current, publication-valid fits.

    This is the replication driver's default scope. It includes unclassified
    models until the study owner assigns their reporting roles.
    """
    return [key for key, model in CATALOGUE.items() if model.role.publication_required]


def models_with_role(role: ModelRole) -> list[str]:
    """Registry keys declared with ``role``, in registry order."""
    return [key for key, model in CATALOGUE.items() if model.role is role]


__all__ = [
    "CATALOGUE",
    "ENGINES",
    "EngineAdapter",
    "ModelRole",
    "RegisteredModel",
    "engine_for",
    "get",
    "models_with_role",
    "publication_models",
]
