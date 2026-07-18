# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Fits the specified model to the latest data. Saves plots and data, and report to output directory.
"""

import argparse
import importlib
import sys
import time
from multiprocessing import freeze_support

import dse_research_utils.environment.setup as setup

from vocab_growth import environment as env
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
from vocab_growth.storage import upload_to_blob_storage


def _fit_selected_models(selected, config: str, *, render: bool = False):
    """Fit every selected model while collecting convergence-gate failures."""
    contexts = []
    timings: dict[str, float] = {}
    failures: dict[str, str] = {}
    for name, module in selected:
        model_started = time.perf_counter()
        try:
            contexts.append(module.fit(config, render=render))
        except ConvergenceGateError as exc:
            failures[name] = str(exc)
        finally:
            timings[name] = time.perf_counter() - model_started
    return contexts, timings, failures


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

    freeze_support()

    args = parser.parse_args()

    if args.upload and not args.render:
        parser.error("--upload requires --render so only complete reports are published.")

    # Resolve the output root before any output path is computed — and before
    # init_script(), in case script setup ever reads an output location.
    # --output-dir wins over $DSE_VOCAB_GROWTH_OUTPUT_DIR, which wins over the
    # repo-local output/ default.
    env.set_output_root(args.output_dir)

    setup.init_script()

    # Model ids are registered in lower case (see definitions.py); normalise
    # user input so "VG01", "vg01", "Vg01" and "ALL" all resolve correctly.
    args.model = args.model.lower()

    # Model modules follow the "model_<key>" naming convention 1:1 with
    # MODEL_REGISTRY (definitions.py), so the set of fittable models is
    # derived from the registry rather than a second, hand-maintained list
    # that can drift out of sync with it (e.g. a newly added model forgetting
    # this file).
    def _load_model_module(key: str):
        return importlib.import_module(f"vocab_growth.models.model_{key}")

    if args.model == "all":
        selected = [(key, _load_model_module(key)) for key in MODEL_REGISTRY]
    elif args.model in MODEL_REGISTRY:
        selected = [(args.model, _load_model_module(args.model))]
    else:
        console.print(f"[bold red]Unknown model: {args.model}[/bold red]")
        sys.exit(1)

    key_value_table(
        "Run plan",
        [
            ("Models to fit", ", ".join(name for name, _ in selected)),
            ("Sampling config", args.config),
            ("Render Quarto", args.render),
            ("Upload to blob storage", args.upload),
            ("Include traces in upload", args.include_traces),
            ("Output root", env.output_root()),
        ],
    )

    # Disk preflight: reporting-config traces are >10 GB each, so fail fast
    # before a multi-hour sample if the volume can't hold the output.
    heavy = is_reporting_quality_config(args.config)
    env.preflight_disk(
        (20.0 if heavy else 2.0) * len(selected),
        env.output_root(),
        label=f"{len(selected)} fit(s) [{args.config}]",
    )

    run_started = time.perf_counter()
    contexts, per_model_timings, gate_failures = _fit_selected_models(
        selected, args.config, render=args.render
    )

    if args.upload and gate_failures:
        console.print(
            "[bold yellow]Upload skipped:[/bold yellow] at least one selected model "
            "failed the convergence gate; no partial batch was published."
        )
    elif args.upload:
        for context in contexts:
            upload_to_blob_storage(
                context.reporting.output_dir,
                context.reporting.model_label,
                include_traces=args.include_traces,
            )

    if len(selected) > 1:
        pipeline_summary("Run summary — all models", per_model_timings)
    console.print(
        f"[dim]Total run wall time: "
        f"{format_duration(time.perf_counter() - run_started)}[/dim]"
    )
    if gate_failures:
        key_value_table(
            "Convergence gate failures",
            [(name, reason) for name, reason in gate_failures.items()],
        )
        sys.exit(1)
