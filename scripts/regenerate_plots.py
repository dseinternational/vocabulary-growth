#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Redraw a promoted fit's figures from its saved trace, without resampling.

Figures are produced by the fit pipeline's ``Plots`` stage, so a presentation
fix -- a mislabelled axis, a missing reporting-age cap -- could previously only
reach a published model by refitting it. That is the wrong trade twice over: it
costs hours per model, and it perturbs the posterior that the report quotes, so
a cosmetic correction forces every number to be re-verified.

Nothing here touches the posterior. The trace already holds every plot-grid
deterministic the plot stage reads (``p_u_plot``, ``q_plot``, ``r_plot``, the
posterior-predictive counts and the constant-data age grids), so this script
rebuilds the fit context up to but not including sampling, loads the trace, and
re-runs the plot stage alone.

Safety, in order of importance:

* The fit is validated for ``render`` first, so a trace whose model definition,
  sampling configuration or raw-data fingerprint no longer matches the current
  registration is refused rather than redrawn.
* Where an engine re-runs the posterior predictive rather than reading the
  stored draws, the re-run must reproduce those draws exactly or the model is
  refused. The re-run is seeded, but a seed reproduces draws only on the fit's
  own numerical stack, and the companion CSVs it feeds are numbers the report
  quotes.
* Every write goes to a staging directory and is swapped in only once the
  whole redraw has succeeded, so a failure cannot leave a half-updated fit.
  That includes the prior-density figures, cross-lag audits and model graph
  that the rebuilt context writes on its way to the trace.
* ``trace.nc``, ``fit_manifest.json`` and ``fit_state.json`` are never written.
  The fit's identity and provenance are exactly what they were; only derived
  images and their companion CSVs change.

Usage: regenerate_plots.py <model_id|all> [--config rep] [--output-dir DIR]
       regenerate_plots.py vg10 --dry-run     # report what would change
"""

import argparse
import os
import shutil
import sys
from dataclasses import asdict

import arviz as az
import dse_research_utils.environment.setup as setup
import dse_research_utils.statistics.models.reporting as model_reporting
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np

import vocab_growth.reporting_ages as reporting_ages
from vocab_growth import environment as env
from vocab_growth.analysis_frames import expected_analysis_frame_hash
from vocab_growth.fit_artifacts import (
    FIT_MANIFEST_FILENAME,
    TRACE_FILENAME,
    FitValidationError,
    fit_validation_kwargs,
    read_json,
    require_full_trace,
    require_valid_fit,
    source_data_hash,
)
from vocab_growth.models import implementation_identity
from vocab_growth.models.catalogue import CATALOGUE
from vocab_growth.models.catalogue import ENGINES as CATALOGUE_ENGINES
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.reporting import console

# Artefacts a redraw replaces: the figures and CSVs the rebuilt context writes.
# Anything else it stages (the posterior-predictive stage's own trace copy) is
# discarded, and everything else in the fit directory is left alone.
PLOT_SUFFIXES = (".png", ".svg")

# Most non-reproducing variables named in a refusal before it is truncated.
_MAX_NAMED_VARIABLES = 5

# Engines whose plot stage this script knows how to drive, derived from
# `vocab_growth.models.catalogue`. A model whose engine declares no plot hook is
# skipped loudly rather than silently, because a silent skip in a diagnostic
# reads as a pass (see notes/202608061500 section 5); the engine records *why*
# it has none, and that reason is printed.
#
# `build` matters as much as `prepare`: the random-effect engines share
# `configure_bivariate_priors` and the plot stage with the plain bivariate
# engine but have their own data preparation and model build. Driving a
# subject-RE model through the plain engine happens to produce identical figures
# today -- the plot stage reads its posterior from the trace, and the only
# context-derived inputs it uses are `n_trials` and the reporting caps, which
# agree -- but that is a coincidence of the current plot code, not a guarantee,
# so each model is routed through the engine that actually fitted it. Deriving
# the routing from the catalogue is what makes that structural rather than a
# table someone has to remember to update (issue #273).
#
# `plots_call` names the plot stage's calling convention, which differs by
# engine: `definition` passes the model definition, `context` passes nothing
# beyond the context, and `outcome_label` passes the definition's outcome label
# as a keyword (the single-outcome stage is shared across models that plot
# different outcomes, so the label is not recoverable from the context).
ENGINES = {
    name: engine for name, engine in CATALOGUE_ENGINES.items() if engine.supports_replot
}

ENGINE_BY_MODEL = {
    key: model.engine.name
    for key, model in CATALOGUE.items()
    if model.engine.supports_replot
}


def _no_replot_reason(model_id: str) -> str:
    """Why ``model_id`` cannot be redrawn, from the catalogue rather than guessed."""
    model = CATALOGUE.get(model_id)
    if model is None:
        return "not a registered model"
    return model.engine.replot_note or f"engine {model.engine.name!r} declares no plot stage"


def _reporting_configuration(definition, output_root_dir: str):
    return model_reporting.ReportingConfiguration(
        model_name=definition.model_id,
        config_name=definition.config_name,
        output_root_dir=output_root_dir,
        ci_prob=0.89,
        interval_kind="eti",
    )


def _rebuild_context(model_id: str, config: str, staging_root: str, fit_dir: str):
    """Rebuild a fit context up to the plot stage, using the saved trace.

    The reporting configuration is rooted at ``staging_root`` from the start
    rather than redirected afterwards. ``prepare``, ``priors`` and ``build``
    write reporting artefacts of their own -- descriptive statistics, the
    prior-density figures, the cross-lag audits and the model graph -- and
    while this context pointed at the promoted fit they landed there before any
    staging applied. A VG24 redraw that aborted in its build on 2026-09-27 left
    those files replaced. The trace is read from ``fit_dir``, the promoted fit,
    which nothing here writes.
    """
    from vocab_growth.models.common import ModelFitContext

    definition = MODEL_REGISTRY[model_id]
    engine = ENGINES[ENGINE_BY_MODEL[model_id]]

    reporting_config = _reporting_configuration(definition, staging_root)
    context = ModelFitContext(
        reporting=reporting_config,
        sampling=sampling.get_sampling_configuration(config),
        sampling_config_name=config,
    )

    prepare = engine.resolve("prepare")
    configure = engine.resolve("priors")
    build_model = engine.resolve("build")

    prepare(context, definition)
    configure(context, definition)
    build_model(context, definition)

    # Kept conservatively. The original reason -- extract_model_samples read
    # f_obs and p_obs, which a compacted fit does not carry -- went away on
    # 2026-08-23, when the sampler stopped storing the observation-level
    # posterior and the extractors stopped reading it; regeneration from a
    # compacted fit is therefore plausible but has not been exercised, and a
    # clear refusal by name beats a KeyError partway through redrawing figures.
    require_full_trace(fit_dir, purpose=f"Plot regeneration for {model_id}")
    context.set_trace(az.from_netcdf(os.path.join(fit_dir, TRACE_FILENAME)))
    return context, definition, engine


def _predictive_group(trace):
    """The trace's ``posterior_predictive`` group as a Dataset, or ``None``."""
    if "posterior_predictive" not in trace.children:
        return None
    return trace["posterior_predictive"].to_dataset()


def _identical(stored, regenerated) -> bool:
    """Whether two draws arrays hold the same values, one chain at a time.

    The stored side is read lazily from ``trace.nc``. Its observation-level
    draws run to hundreds of megabytes per variable at reporting quality, so
    one chain is loaded at a time.
    """
    if "chain" not in stored.dims:
        return np.array_equal(stored.values, regenerated.values, equal_nan=True)
    return all(
        np.array_equal(
            stored.isel(chain=i).values,
            regenerated.isel(chain=i).values,
            equal_nan=True,
        )
        for i in range(stored.sizes["chain"])
    )


def _predictive_differences(stored, regenerated) -> list[str]:
    """The re-run posterior-predictive variables that do not reproduce the fit's.

    ``stored`` is the fit's ``posterior_predictive`` group as it was before the
    re-run, since PyMC replaces the group in place; ``regenerated`` is what the
    re-run left. Only the re-run's own variables are compared, because they are
    all that the plot stage and the calibration table read. A variable the fit
    never stored cannot be vouched for, so it counts as a difference.
    """
    if regenerated is None:
        return ["posterior_predictive (the re-run wrote no group)"]
    if stored is None:
        return ["posterior_predictive (not stored in the fit's trace)"]
    differences = []
    for name in sorted(regenerated.data_vars):
        new = regenerated[name]
        if name not in stored.data_vars:
            differences.append(f"{name} (not stored)")
            continue
        old = stored[name]
        if old.dims != new.dims or old.shape != new.shape:
            differences.append(f"{name} (shape {old.shape} -> {new.shape})")
        elif not _identical(old, new):
            differences.append(name)
    return differences


def _implementation_change(fit_dir: str) -> str:
    """What moved between the fit's recorded implementation and this checkout.

    The numerical libraries are the usual reason a seeded re-run stops
    reproducing, as VG13 showed when redrawn on a newer numpy and PyTensor. The
    signature is not the whole story, though: it omits numba, PyTensor's
    default backend, which compiles some of the draws, so a match does not rule
    the stack out.
    """
    try:
        manifest = read_json(os.path.join(fit_dir, FIT_MANIFEST_FILENAME))
    except FitValidationError:
        manifest = {}
    recorded = manifest.get("model", {}).get("implementation")
    current = implementation_identity.implementation_signature()
    if implementation_identity.matches(recorded, current):
        return "the recorded implementation signature matches this checkout"
    return implementation_identity.describe_difference(recorded, current)


def _refuse_non_reproducing(model_id: str, differences: list[str], fit_dir: str) -> None:
    shown = ", ".join(differences[:_MAX_NAMED_VARIABLES])
    if len(differences) > _MAX_NAMED_VARIABLES:
        shown += f", and {len(differences) - _MAX_NAMED_VARIABLES} more"
    console.print(
        f"[bold red]\\[refused][/bold red] {model_id}: the re-run posterior "
        f"predictive did not reproduce the fit's stored draws ({shown}). "
        "Redrawing would change companion CSVs that the report quotes. Since the "
        f"fit: {_implementation_change(fit_dir)}. Redraw with the fit's own "
        "numerical stack, or refit."
    )


def regenerate(model_id: str, config: str, dry_run: bool = False) -> bool:
    """Redraw one model's figures. Returns True if the fit was updated."""
    definition = MODEL_REGISTRY[model_id]
    output_root = env.output_root()
    canonical = _reporting_configuration(definition, output_root)
    target = canonical.output_dir

    try:
        require_valid_fit(
            target,
            **fit_validation_kwargs(
                "render",
                expected_definition=definition,
                expected_sampling_config_name=config,
                expected_sampling_parameters=asdict(
                    sampling.get_sampling_configuration(config)
                ),
                current_source_data_hash=source_data_hash(env.DATA_DIR),
                # Loader-rule drift is invisible to the raw-CSV fingerprint, and
                # replotting a fit whose frame has moved would put current-data
                # labels on a stale posterior's figures (issue #266 finding 1).
                current_analysis_frame_hash=expected_analysis_frame_hash(
                    model_id, definition
                ),
            ),
        )
    except FitValidationError as exc:
        console.print(f"[bold red]\\[invalid][/bold red] {model_id}: {exc}")
        return False

    # Stage under a throwaway *root*, so the reporting configuration derives its
    # own output directory exactly as the fit pipeline does. A failure at any
    # stage then cannot leave the promoted fit holding half a set of figures.
    # Keyed by the model id, not the label: the staged directory already repeats
    # the label, and the model graph written into it comes from graphviz `dot`,
    # which fails on paths over 260 characters (see
    # `fit_artifacts.create_staging_root`).
    staging_root = os.path.join(output_root, ".replot", model_id)
    if os.path.isdir(staging_root):
        shutil.rmtree(staging_root)
    os.makedirs(staging_root, exist_ok=True)
    staged_dir = _reporting_configuration(definition, staging_root).output_dir
    os.makedirs(staged_dir, exist_ok=True)

    try:
        context, definition, engine = _rebuild_context(
            model_id, config, staging_root, target
        )

        # Which of these an engine takes is a CATALOGUE DECLARATION, not a probe of
        # its module. `samples_extractor` names a pure `f(trace) -> samples`, so the
        # stored posterior-predictive draws are reused verbatim; where it is None the
        # engine builds its samples inside the posterior-predictive stage, which is
        # re-run and writes its trace copy into staging, never over the promoted
        # one. The univariate, bivariate and trivariate engines declare an
        # extractor; `bivariate_re` (eleven models) and `joint` do not.
        #
        # The re-run is seeded from the sampling configuration, which reproduces the
        # stored draws only on the fit's own numerical stack. VG13, fitted with
        # numpy 2.4.6 and PyTensor 3.3.0, came back from a newer stack with 17
        # companion CSVs moved by Monte Carlo noise (up to 4 words on count
        # quantiles). The draws are therefore checked against the stored ones
        # rather than assumed. A check of recorded package versions would be the
        # wrong test both ways: the signature includes dse-research-utils, whose
        # font upgrade prompted that redraw and did not move a draw, and omits
        # numba, PyTensor's default backend, which compiles some of the draws.
        if engine.samples_extractor is not None:
            context.set_model_samples(engine.resolve("samples_extractor")(context.trace))
        else:
            stored = _predictive_group(context.trace)
            engine.resolve("posterior_predictive")(context, definition)
            differences = _predictive_differences(
                stored, _predictive_group(context.trace)
            )
            if differences:
                shutil.rmtree(staging_root, ignore_errors=True)
                _refuse_non_reproducing(model_id, differences, target)
                return False

        # Staging also holds what the earlier stages wrote, so the plot stage's
        # own output is what is left over from this listing.
        before_plots = set(os.listdir(staged_dir))
        plots = engine.resolve("plots")
        plots_call = engine.plots_call
        if plots_call == "definition":
            plots(context, definition)
        elif plots_call == "outcome_label":
            # The single-outcome stage also needs to know *which* outcome, or it
            # redraws uncapped -- caught by tests/test_reporting_age_policy.py
            # when VG02's pmf/cdf came back with a 90-month column against an
            # 84-month cap.
            plots(
                context,
                outcome_label=definition.outcome_label,
                quantity=reporting_ages.quantity_for_outcome(definition.outcome),
            )
        elif plots_call == "context":
            plots(context)
        else:
            raise ValueError(f"unknown plots_call {plots_call!r} for {model_id}")
    except Exception as exc:  # noqa: BLE001 - report and keep the fit intact
        shutil.rmtree(staging_root, ignore_errors=True)
        console.print(f"[bold red]\\[failed][/bold red] {model_id}: {type(exc).__name__}: {exc}")
        return False

    staged = set(os.listdir(staged_dir))
    produced = sorted(
        f for f in staged if f.endswith(PLOT_SUFFIXES) or f.endswith(".csv")
    )
    if not staged - before_plots:
        shutil.rmtree(staging_root, ignore_errors=True)
        console.print(f"[bold red]\\[failed][/bold red] {model_id}: plot stage produced nothing")
        return False

    if dry_run:
        console.print(f"\\[dry-run] {model_id}: would replace {len(produced)} artefact(s)")
        shutil.rmtree(staging_root, ignore_errors=True)
        return False

    for name in produced:
        shutil.copy2(os.path.join(staged_dir, name), os.path.join(target, name))
    shutil.rmtree(staging_root, ignore_errors=True)
    console.print(f"[bold green]\\[done][/bold green] {model_id}: replaced {len(produced)} artefact(s)")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Model id (e.g. vg10) or 'all'.")
    parser.add_argument("--config", default="rep")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Redraw into staging and report, without replacing anything.",
    )
    args = parser.parse_args()

    env.set_output_root(args.output_dir)
    setup.init_script()
    console.print(f"\\[output] regenerating plots under {env.output_root()}")

    if args.model == "all":
        selected = [m for m in MODEL_REGISTRY if m in ENGINE_BY_MODEL]
        skipped = [m for m in MODEL_REGISTRY if m not in ENGINE_BY_MODEL]
        for model_id in sorted(skipped):
            console.print(f"\\[skip] {model_id}: {_no_replot_reason(model_id)}")
    elif args.model in ENGINE_BY_MODEL:
        selected = [args.model]
    else:
        console.print(
            f"[bold red]No regeneration path for model: {args.model}"
            f" ({_no_replot_reason(args.model)})[/bold red]"
        )
        sys.exit(1)

    updated = [m for m in selected if regenerate(m, args.config, dry_run=args.dry_run)]
    console.print(f"\n{len(updated)}/{len(selected)} model(s) updated.")
    sys.exit(0 if (updated or args.dry_run) else 1)
