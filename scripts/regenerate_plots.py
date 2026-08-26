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
* Figures are written to a staging directory and only swapped in once the whole
  stage has succeeded, so a mid-stage failure cannot leave a half-updated fit.
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

import vocab_growth.reporting_ages as reporting_ages
from vocab_growth import environment as env
from vocab_growth.analysis_frames import expected_analysis_frame_hash
from vocab_growth.fit_artifacts import (
    FitValidationError,
    fit_validation_kwargs,
    require_full_trace,
    require_valid_fit,
    source_data_hash,
)
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.reporting import console

# Artefacts the plot stage owns. Everything else in the output directory belongs
# to another stage (or to the fit's identity) and is left alone.
PLOT_SUFFIXES = (".png", ".svg")

# Engines whose plot stage this script knows how to drive. A model outside this
# map is skipped loudly rather than silently, because a silent skip in a
# diagnostic reads as a pass (see notes/202608061500 section 5).
#
# ``build`` matters as much as ``prepare``: the random-effect engines share
# ``configure_bivariate_priors`` and the plot stage with the plain bivariate
# engine but have their own data preparation and model build. Driving a
# subject-RE model through the plain engine happens to produce identical figures
# today -- the plot stage reads its posterior from the trace, and the only
# context-derived inputs it uses are ``n_trials`` and the reporting caps, which
# agree -- but that is a coincidence of the current plot code, not a guarantee,
# so each model is routed through the engine that actually fitted it.
#
# ``plots_call`` names the plot stage's calling convention, which differs by
# engine: ``definition`` passes the model definition, ``context`` passes nothing
# beyond the context, and ``outcome_label`` passes the definition's outcome label
# as a keyword (the single-outcome stage is shared across models that plot
# different outcomes, so the label is not recoverable from the context).
ENGINES = {
    "univariate": {
        "module": "vocab_growth.models.common",
        "prepare": "prepare_univariate_data",
        "priors": "configure_univariate_priors",
        "build": "build_model",
        "plots": "run_standard_plots",
        "plots_call": "outcome_label",
    },
    "bivariate": {
        "module": "vocab_growth.models.common_bivariate",
        "prepare": "prepare_bivariate_data",
        "priors": "configure_bivariate_priors",
        "build": "build_model",
        "plots": "_run_bivariate_joint_plots",
        "plots_call": "definition",
    },
    "bivariate_re": {
        "module": "vocab_growth.models.common_bivariate_re",
        "prepare": "prepare_bivariate_re_data",
        "priors": "configure_bivariate_priors",
        "build": "build_model_re",
        "plots": "_run_bivariate_joint_plots",
        "plots_call": "definition",
    },
    "trivariate": {
        "module": "vocab_growth.models.common_trivariate",
        "prepare": "prepare_trivariate_data",
        "priors": "configure_trivariate_priors",
        "build": "build_model",
        "plots": "_run_trivariate_plots",
        "plots_call": "context",
    },
    "joint": {
        "module": "vocab_growth.models.common_joint_modality",
        "prepare": "prepare_joint_data",
        "priors": "configure_joint_priors",
        "build": "build_model",
        "plots": "_run_joint_plots",
        "plots_call": "context",
    },
}

ENGINE_BY_MODEL = {
    "vg01": "univariate",
    "vg02": "univariate",
    "vg03": "univariate",
    "vg04": "univariate",
    "vg05": "bivariate",
    "vg07": "bivariate_re",
    "vg08": "bivariate_re",
    "vg09": "bivariate_re",
    "vg10": "bivariate_re",
    "vg13": "bivariate_re",
    "vg16": "bivariate_re",
    "vg19": "bivariate_re",
    "vg20": "bivariate_re",
    "vg21": "bivariate_re",
    "vg22": "bivariate_re",
    "vg23": "bivariate_re",
    "vg14": "trivariate",
    "vg15": "joint",
}


def _resolve(module_name, attr):
    import importlib

    module = importlib.import_module(module_name)
    fn = getattr(module, attr, None)
    if fn is None:
        raise AttributeError(f"{module_name} has no {attr!r}")
    return module, fn


def _rebuild_context(model_id: str, config: str, output_root_dir: str):
    """Rebuild a fit context up to the plot stage, using the saved trace."""
    from vocab_growth.models.common import ModelFitContext

    definition = MODEL_REGISTRY[model_id]
    engine = ENGINES[ENGINE_BY_MODEL[model_id]]

    reporting_config = model_reporting.ReportingConfiguration(
        model_name=definition.model_id,
        config_name=definition.config_name,
        output_root_dir=output_root_dir,
        ci_prob=0.89,
        interval_kind="eti",
    )
    context = ModelFitContext(
        reporting=reporting_config,
        sampling=sampling.get_sampling_configuration(config),
        sampling_config_name=config,
    )

    module, prepare = _resolve(engine["module"], engine["prepare"])
    _, configure = _resolve(engine["module"], engine["priors"])
    _, build_model = _resolve(engine["module"], engine["build"])

    prepare(context, definition)
    configure(context, definition)
    build_model(context, definition)

    # Kept conservatively. The original reason -- extract_model_samples read
    # f_obs and p_obs, which a compacted fit does not carry -- went away on
    # 2026-08-23, when the sampler stopped storing the observation-level
    # posterior and the extractors stopped reading it; regeneration from a
    # compacted fit is therefore plausible but has not been exercised, and a
    # clear refusal by name beats a KeyError partway through redrawing figures.
    require_full_trace(
        reporting_config.output_dir, purpose=f"Plot regeneration for {model_id}"
    )
    trace_path = os.path.join(reporting_config.output_dir, "trace.nc")
    context.set_trace(az.from_netcdf(trace_path))
    return context, definition, engine


def regenerate(model_id: str, config: str, dry_run: bool = False) -> bool:
    """Redraw one model's figures. Returns True if the fit was updated."""
    definition = MODEL_REGISTRY[model_id]
    output_root = env.output_root()
    canonical = model_reporting.ReportingConfiguration(
        model_name=definition.model_id,
        config_name=definition.config_name,
        output_root_dir=output_root,
        ci_prob=0.89,
        interval_kind="eti",
    )
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
        console.print(f"[bold red][invalid][/bold red] {model_id}: {exc}")
        return False

    # Stage under a throwaway *root*, so the reporting configuration derives its
    # own output directory exactly as the fit pipeline does. A failure mid-stage
    # then cannot leave the promoted fit holding half a set of figures.
    staging_root = os.path.join(output_root, ".replot", canonical.model_label)
    if os.path.isdir(staging_root):
        shutil.rmtree(staging_root)
    os.makedirs(staging_root, exist_ok=True)
    staged_dir = model_reporting.ReportingConfiguration(
        model_name=definition.model_id,
        config_name=definition.config_name,
        output_root_dir=staging_root,
        ci_prob=0.89,
        interval_kind="eti",
    ).output_dir
    os.makedirs(staged_dir, exist_ok=True)

    try:
        context, definition, engine = _rebuild_context(model_id, config, output_root)
        # Redirect the plot stage's writes into staging.
        context.reporting = model_reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name=definition.config_name,
            output_root_dir=staging_root,
            ci_prob=0.89,
            interval_kind="eti",
        )

        # The bivariate and trivariate engines expose a pure extractor, so the
        # stored posterior-predictive draws are reused verbatim. The joint engine
        # builds its samples inside the posterior-predictive stage, so that stage
        # is re-run -- it is seeded from the sampling configuration, so it
        # reproduces the stored draws rather than perturbing them, and it writes
        # its trace copy into staging, never over the promoted one.
        extractor = getattr(
            _resolve(engine["module"], engine["prepare"])[0],
            "extract_model_samples",
            None,
        )
        if extractor is not None:
            context.set_model_samples(extractor(context.trace))
        else:
            _, resample_pp = _resolve(engine["module"], "sample_posterior_predictive")
            resample_pp(context, definition)

        _, plots = _resolve(engine["module"], engine["plots"])
        plots_call = engine["plots_call"]
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
        console.print(f"[bold red][failed][/bold red] {model_id}: {type(exc).__name__}: {exc}")
        return False

    produced = sorted(
        f
        for f in os.listdir(staged_dir)
        if f.endswith(PLOT_SUFFIXES) or f.endswith(".csv")
    )
    if not produced:
        shutil.rmtree(staging_root, ignore_errors=True)
        console.print(f"[bold red][failed][/bold red] {model_id}: plot stage produced nothing")
        return False

    if dry_run:
        console.print(f"[dry-run] {model_id}: would replace {len(produced)} artefact(s)")
        shutil.rmtree(staging_root, ignore_errors=True)
        return False

    for name in produced:
        shutil.copy2(os.path.join(staged_dir, name), os.path.join(target, name))
    shutil.rmtree(staging_root, ignore_errors=True)
    console.print(f"[bold green][done][/bold green] {model_id}: replaced {len(produced)} artefact(s)")
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
    console.print(f"[output] regenerating plots under {env.output_root()}")

    if args.model == "all":
        selected = [m for m in MODEL_REGISTRY if m in ENGINE_BY_MODEL]
        skipped = [m for m in MODEL_REGISTRY if m not in ENGINE_BY_MODEL]
        if skipped:
            console.print(
                f"[skip] no plot-regeneration path for: {', '.join(sorted(skipped))}"
            )
    elif args.model in ENGINE_BY_MODEL:
        selected = [args.model]
    else:
        console.print(f"[bold red]No regeneration path for model: {args.model}[/bold red]")
        sys.exit(1)

    updated = [m for m in selected if regenerate(m, args.config, dry_run=args.dry_run)]
    console.print(f"\n{len(updated)}/{len(selected)} model(s) updated.")
    sys.exit(0 if (updated or args.dry_run) else 1)
