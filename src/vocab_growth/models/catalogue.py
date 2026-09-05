# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One record per registered model, outside its statistical definition.

A model is more than the definition ``definitions.py`` holds. It also has an
engine, a wrapper module, an analysis-frame builder, a prior-predictive hook, a
plot hook and a report template -- and until this module existed each of those
lived in a separate hand-maintained table: the engine in every ``model_vgNN``
import, again in :data:`vocab_growth.analysis_frames.FRAME_BUILDERS`, again in
``scripts/regenerate_plots.py``, again in ``scripts/prior_predictive_audit.py``
and again in several test dispatch tables. Drift guards over some of the copies
are not enough. The audit script's copy was stale for six of the twenty
registered models (VG16 and VG19-VG23, all of which use ``common_bivariate_re``
while the script routed them through the plain ``common_bivariate``), and it
still produced plots -- so an audit of the wrong graph looked like a valid prior
check (issue #273).

The record is deliberately **not** part of the serialised statistical
definition. A fit is validated by comparing the manifest's recorded definition
field for field, so adding a field to a definition dataclass invalidates every
existing fit of that class. Reporting hooks remain outside that definition.
A separate executable-code signature now checks implementation changes, including
engine changes, before a fit is reused.

**Engine identity is not inferred from the definition class.** VG05 and VG07
share ``BivariateModelDefinition`` and run on different engines, so the class
cannot decide. It is declared here once, and
``tests/test_model_catalogue.py`` pins each declaration against what the
model's own wrapper module imports.

Everything is held as **strings resolved on demand**, so importing this module
costs nothing: the engines pull in PyMC, and the validators that need a frame
builder (``sync_report_figures``, ``compare_models``) must not.
"""

from __future__ import annotations

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
    """How one fitting engine's pipeline stages are named and called.

    Every field but ``name`` is an attribute of :attr:`module`, resolved on
    demand. The ``*_call`` fields record a **calling convention** rather than a
    signature, because the stages genuinely differ in what they need beyond the
    context and passing the wrong one raises only when that stage is reached --
    which, for a script that runs a single stage, is at the point of use.
    """

    name: str
    """Stable engine key, used in messages and in derived dispatch tables."""

    module: str
    """Dotted path of the engine module every other field is an attribute of."""

    prepare: str
    """Data-preparation stage; called ``f(context, definition)``."""

    priors: str
    """Prior/hyperparameter configuration stage; called ``f(context, definition)``."""

    build: str
    """Model-build stage; called ``f(context, definition)``."""

    prior_checks: str
    """Prior-predictive stage."""

    prior_checks_call: str
    """How :attr:`prior_checks` is invoked, matching the engine's own fit pipeline.

    ``"outcome"`` passes the outcome column and label positionally (the
    single-outcome stage is shared across models plotting different outcomes);
    ``"definition"`` passes the definition, which is what adds
    :mod:`vocab_growth.models.prior_child_checks`'s unseen-child figures; and
    ``"context"`` passes nothing further.

    The two bivariate engines differ here, and that difference is the point:
    ``common_bivariate_re`` passes the definition so a child-effect model's
    prior figures contain a child (issue #233), while the plain engine, which
    has no child effects, does not.
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
    """Pure ``f(trace) -> samples`` for a replot, or ``None`` to re-sample instead.

    Declared rather than probed. ``regenerate_plots.py`` used to decide this with
    ``getattr(engine_module, "extract_model_samples", None)``, which made the branch
    a function of an engine module's **import list**: ``common_bivariate_re`` imports
    the shared predictive but not the shared extractor, so its eleven models silently
    took the re-sampling path while the surrounding comment said the bivariate engines
    reused stored draws. Adding that name for an unrelated reason would have flipped
    eleven models' replot behaviour with nothing to notice.

    ``None`` means the replot re-runs :attr:`posterior_predictive`, which is seeded
    from the sampling configuration and so reproduces the stored draws. Whether an
    engine that could expose an extractor *should* is a separate decision; these
    declarations record what each engine does today.
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
        """The callable this adapter names in ``attribute``.

        Raises ``AttributeError`` naming both the engine and the field, rather
        than returning ``None`` for a caller to trip over later -- but only for a
        hook this engine declares as absent. A field name *misspelt by the caller*
        raises from the ``getattr`` below with only that name, and a declared hook
        the module does not actually define raises from ``getattr`` on the module,
        naming the module rather than the engine. Both are pinned by
        ``tests/test_model_catalogue.py``, which is what makes the declarations
        trustworthy enough for this to return ``Any``: the return type cannot be
        narrowed while the hooks have five different signatures.
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
            # No replot path. The engine re-exports `run_standard_plots` and
            # `extract_model_samples` from `common`, so one is within reach, but
            # its posterior-predictive stage is this engine's own
            # (`sample_posterior_predictive_re`) and redrawing VG11/VG12 out of
            # a fit has never been exercised. Claiming support that has not been
            # run would turn a refusal into a wrong figure.
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
            # "definition", like the RE engine: the stage's `definition` parameter
            # used to default to None so VG05 could omit it, and this convention
            # recorded that. VG05 has no child effects, so `prior_child_checks`
            # adds nothing for it and passing the definition is byte-identical --
            # one convention fewer, and no optional parameter pretending a caller
            # might not have one.
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
            # This engine imports the shared predictive from `common_bivariate` but
            # not the shared extractor, so its eleven models re-run the predictive
            # on replot. Declared, not inferred: the old `getattr` probe made the
            # branch depend on this module's import list.
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
    }
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
        key: RegisteredModel(model_key=key, engine=ENGINES[engine_of[key]])
        for key in MODEL_REGISTRY
    }


#: Every registered model, in ``MODEL_REGISTRY`` order.
CATALOGUE: dict[str, RegisteredModel] = _catalogue()

# The exploratory sign-group modules (VG17, VG18) are deliberately absent. They
# are not in MODEL_REGISTRY and carry a custom fit path that bypasses the shared
# manifest, staged promotion and convergence gate, so a catalogue entry would
# assert a supported lifecycle they do not have. Issue #273 finding 4 asked for
# that to be decided rather than left implicit; it was, on 2026-08-31, in favour
# of "explicitly exploratory and non-validatable" -- they now live in
# `vocab_growth.models.exploratory`, outside the `model_vgNN` naming convention
# `fit_model.py` resolves, so they are unreachable from here by construction.
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


__all__ = [
    "CATALOGUE",
    "ENGINES",
    "EngineAdapter",
    "RegisteredModel",
    "engine_for",
    "get",
]
