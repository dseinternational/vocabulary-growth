# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Fits the specified model to the latest data. Saves plots and data, and report to output directory.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from multiprocessing import freeze_support
from types import SimpleNamespace

import dse_research_utils.environment.setup as setup
import dse_research_utils.statistics.models.reporting as model_reporting
import dse_research_utils.statistics.models.sampling as sampling

from vocab_growth import environment as env
from vocab_growth.analysis_frames import expected_analysis_frame_hash
from vocab_growth.fit_artifacts import (
    FitValidationError,
    TracePersistence,
    configured_trace_persistence,
    fit_validation_kwargs,
    require_valid_fit,
    set_trace_persistence,
    source_data_hash,
)
from vocab_growth.models import catalogue
from vocab_growth.models.common import (
    ConvergenceGateError,
    is_reporting_quality_config,
)
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.reporting import (
    console,
    format_duration,
    key_value_table,
    pipeline_summary,
)
from vocab_growth.storage import upload_to_blob_storage, validate_fit_for_upload


def _fit_selected_models(selected, config: str):
    """Fit every selected model while collecting convergence-gate failures."""
    contexts = []
    timings: dict[str, float] = {}
    failures: dict[str, str] = {}
    for name, module in selected:
        model_started = time.perf_counter()
        try:
            contexts.append(module.fit(config))
        except ConvergenceGateError as exc:
            failures[name] = str(exc)
        finally:
            timings[name] = time.perf_counter() - model_started
    return contexts, timings, failures


def _render_output(output_dir: str, model_id: str | None = None) -> None:
    """Render one already-promoted fit without changing its lifecycle state.

    The report template is refreshed from ``docs/models/<model>/index.qmd`` first.
    The fit stage copies the template into the output directory, so without this
    ``--render-only`` would re-render whichever template was current when the fit
    ran, and a fix to the report could never reach an existing fit.

    That is not hypothetical: the soft-tier convergence callout was added to every
    model report on 2026-08-05, and VG13 — one of the three fits that actually has
    caveats to disclose — kept rendering without it. ``--render-only`` reported
    success each time. A disclosure mechanism that silently discloses nothing is
    the failure this whole path exists to prevent.

    Only the template is refreshed. The trace, the summaries and the manifest are
    untouched, and the caller has already validated the fit against the current
    registered definition, so the refreshed template is being run against a fit it
    is compatible with.
    """
    qmd_path = os.path.join(output_dir, "index.qmd")
    if model_id is not None:
        template = os.path.join(
            env.DOCS_DIR, "models", model_id.lower(), "index.qmd"
        )
        if not os.path.isfile(template):
            raise FileNotFoundError(f"Report template is missing: {template}")
        shutil.copy(template, qmd_path)
    if not os.path.isfile(qmd_path):
        raise FileNotFoundError(f"Quarto source is missing: {qmd_path}")
    # Quarto otherwise resolves the Jupyter kernel for the report's python cells
    # from PATH, which is not this interpreter when the environment's python was
    # invoked by absolute path without activation. The report then renders against
    # whichever python PATH finds and cannot open the trace this fit just wrote.
    render_env = {**os.environ, "QUARTO_PYTHON": sys.executable}
    subprocess.run(["quarto", "render", qmd_path], check=True, env=render_env)
    if not os.path.isfile(os.path.join(output_dir, "index.html")):
        raise RuntimeError("Quarto render completed without producing index.html.")


def _render_contexts(contexts):
    """Render all successful fits, collecting failures without stopping the batch."""
    timings: dict[str, float] = {}
    failures: dict[str, str] = {}
    for context in contexts:
        name = context.reporting.model_name.lower()
        render_started = time.perf_counter()
        try:
            _render_output(
                context.reporting.output_dir, context.reporting.model_name
            )
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
        finally:
            timings[name] = time.perf_counter() - render_started
    return timings, failures


def _existing_contexts(selected, config: str):
    """Resolve compatible promoted fits for a render-only retry."""
    expected_sampling = sampling.get_sampling_configuration(config)
    current_source_hash = source_data_hash(env.DATA_DIR)
    contexts = []
    failures: dict[str, str] = {}
    for name, _module in selected:
        definition = MODEL_REGISTRY[name]
        model_reporting_config = model_reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name=definition.config_name,
            output_root_dir=env.output_root(),
            ci_prob=0.89,
            interval_kind="eti",
        )
        try:
            require_valid_fit(
                model_reporting_config.output_dir,
                **fit_validation_kwargs(
                    "render",
                    expected_definition=definition,
                    expected_sampling_config_name=config,
                    expected_sampling_parameters=asdict(expected_sampling),
                    current_source_data_hash=current_source_hash,
                    # Rebuilt per definition: catches loader-rule drift the
                    # raw-CSV fingerprint cannot (issue #266 finding 1).
                    current_analysis_frame_hash=expected_analysis_frame_hash(
                        name, definition
                    ),
                ),
            )
        except FitValidationError as exc:
            failures[name] = str(exc)
        else:
            contexts.append(SimpleNamespace(reporting=model_reporting_config))
    return contexts, failures


def _publication_plan(contexts, config: str):
    """Validate the complete upload set before publishing any model."""
    expected_sampling = sampling.get_sampling_configuration(config)
    current_source_hash = source_data_hash(env.DATA_DIR)
    plan = []
    failures: dict[str, str] = {}
    for context in contexts:
        name = context.reporting.model_name.lower()
        definition = MODEL_REGISTRY[name]
        validation_kwargs = fit_validation_kwargs(
            "publish",
            expected_definition=definition,
            expected_sampling_config_name=config,
            expected_sampling_parameters=asdict(expected_sampling),
            current_source_data_hash=current_source_hash,
            current_analysis_frame_hash=expected_analysis_frame_hash(
                name, definition
            ),
        )
        try:
            validated_output = validate_fit_for_upload(
                context.reporting.output_dir, validation_kwargs
            )
        except FitValidationError as exc:
            failures[name] = str(exc)
        else:
            plan.append((context, validated_output))
    return plan, failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, default=None, help="Model id or all.")
    parser.add_argument(
        "--config",
        type=str,
        default="dev",
        help="Sampling configuration to use (e.g., dev[elopment], test, rep[orting])",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the Quarto model output after fitting",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Render an existing compatible fit without sampling again.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload model output to Azure Blob Storage",
    )
    parser.add_argument(
        "--include-traces",
        action="store_true",
        help="Include trace files (.nc) in the upload (excluded by default).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Root directory for model output (overrides "
            "$DSE_VOCAB_GROWTH_OUTPUT_DIR; default: <repo>/output). Useful for "
            "redirecting heavy traces to a scratch disk on ephemeral VMs."
        ),
    )
    parser.add_argument(
        "--trace-persistence",
        type=str,
        default=None,
        choices=[tier.value for tier in TracePersistence],
        help=(
            "How much of the trace to keep in trace.nc (overrides "
            "$DSE_VOCAB_GROWTH_TRACE_PERSISTENCE; default: full). 'compact' "
            "drops the duplicated scaled random effects, which are recomputable "
            "from the raw draws and the scales and cost nothing statistically "
            "(the observation-sized deterministics are not stored at any tier "
            "since 2026-08-23, so compact no longer has them to drop); "
            "'minimal' also drops the stored log-likelihood and posterior "
            "predictive, which forecloses recomputing LOO or a new predictive "
            "check without refitting. Does not affect the posterior."
        ),
    )

    freeze_support()

    args = parser.parse_args()

    if args.render and args.render_only:
        parser.error("Choose --render or --render-only, not both.")
    if args.upload and not (args.render or args.render_only):
        parser.error("--upload requires --render so only complete reports are published.")
    if args.upload and not is_reporting_quality_config(args.config):
        parser.error("--upload requires a reporting-quality sampling configuration.")

    # Resolve the output root before any output path is computed — and before
    # init_script(), in case script setup ever reads an output location.
    # --output-dir wins over $DSE_VOCAB_GROWTH_OUTPUT_DIR, which wins over the
    # repo-local output/ default.
    env.set_output_root(args.output_dir)
    # Same precedence for how much of each trace is kept. Setting the override
    # is not enough to validate it: with the flag omitted this only clears the
    # override, and the environment variable would not be parsed until the first
    # save — several hours into a reporting fit. Resolve it here so a bad value
    # fails at startup.
    set_trace_persistence(args.trace_persistence)
    configured_trace_persistence()

    setup.init_script()

    # Model ids are registered in lower case (see definitions.py); normalise
    # user input so "VG01", "vg01", "Vg01" and "ALL" all resolve correctly.
    args.model = args.model.lower()

    # The set of fittable models, and each one's wrapper module, come from the
    # catalogue rather than from a second hand-maintained list that can drift
    # out of sync with it (e.g. a newly added model forgetting this file). The
    # catalogue covers MODEL_REGISTRY exactly and refuses an unregistered key by
    # name rather than guessing a module.
    def _load_model_module(key: str):
        return catalogue.get(key).load_wrapper()

    if args.model == "all":
        selected = [(key, _load_model_module(key)) for key in catalogue.CATALOGUE]
    elif args.model in catalogue.CATALOGUE:
        selected = [(args.model, _load_model_module(args.model))]
    else:
        console.print(f"[bold red]Unknown model: {args.model}[/bold red]")
        sys.exit(1)

    key_value_table(
        "Run plan",
        [
            ("Models to fit", ", ".join(name for name, _ in selected)),
            ("Sampling config", args.config),
            ("Fit models", not args.render_only),
            ("Render Quarto", args.render or args.render_only),
            ("Upload to blob storage", args.upload),
            ("Include traces in upload", args.include_traces),
            ("Output root", env.output_root()),
        ],
    )

    # Disk preflight: reporting-config traces are >10 GB each, so fail fast
    # before a multi-hour sample if the volume can't hold the output.
    if not args.render_only:
        heavy = is_reporting_quality_config(args.config)
        env.preflight_disk(
            (20.0 if heavy else 2.0) * len(selected),
            env.output_root(),
            label=f"{len(selected)} fit(s) [{args.config}]",
        )

    run_started = time.perf_counter()
    if args.render_only:
        contexts, failures = _existing_contexts(selected, args.config)
        per_model_timings = {}
    else:
        contexts, per_model_timings, failures = _fit_selected_models(
            selected, args.config
        )

    render_timings: dict[str, float] = {}
    if args.render or args.render_only:
        render_timings, render_failures = _render_contexts(contexts)
        failures.update(render_failures)

    publication_plan = []
    if args.upload and not failures:
        publication_plan, publication_failures = _publication_plan(
            contexts, args.config
        )
        failures.update(publication_failures)

    if args.upload and failures:
        console.print(
            "[bold yellow]Upload skipped:[/bold yellow] at least one selected model "
            "failed fitting, validation, or rendering; no partial batch was published."
        )
    elif args.upload:
        for context, validated_output in publication_plan:
            upload_to_blob_storage(
                validated_output,
                context.reporting.model_label,
                include_traces=args.include_traces,
            )

    if len(selected) > 1 and per_model_timings:
        pipeline_summary("Run summary — all models", per_model_timings)
    if len(selected) > 1 and render_timings:
        pipeline_summary("Render summary — all models", render_timings)
    console.print(
        f"[dim]Total run wall time: "
        f"{format_duration(time.perf_counter() - run_started)}[/dim]"
    )
    if failures:
        key_value_table(
            "Fit, validation, or render failures",
            [(name, reason) for name, reason in failures.items()],
        )
        sys.exit(1)
