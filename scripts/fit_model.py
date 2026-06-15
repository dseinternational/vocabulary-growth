# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Fits the specified model to the latest data. Saves plots and data, and report to output directory.
"""

import argparse
import os
import subprocess
import sys
import time
from multiprocessing import freeze_support

import dse_research_utils.environment.setup as setup

from vocab_growth.models import (
    model_vg01,
    model_vg02,
    model_vg03,
    model_vg04,
    model_vg05,
    model_vg06,
    model_vg07,
    model_vg08,
    model_vg09,
    model_vg10,
    model_vg11,
    model_vg12,
    model_vg13,
)
from vocab_growth.reporting import (
    console,
    format_duration,
    heading,
    key_value_table,
    pipeline_summary,
)
from vocab_growth.storage import upload_to_blob_storage

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

    freeze_support()

    setup.init_script()

    args = parser.parse_args()

    models = {
        "vg01": model_vg01,
        "vg02": model_vg02,
        "vg03": model_vg03,
        "vg04": model_vg04,
        "vg05": model_vg05,
        "vg06": model_vg06,
        "vg07": model_vg07,
        "vg08": model_vg08,
        "vg09": model_vg09,
        "vg10": model_vg10,
        "vg11": model_vg11,
        "vg12": model_vg12,
        "vg13": model_vg13,
    }

    if args.model == "all":
        selected = list(models.items())
    elif args.model in models:
        selected = [(args.model, models[args.model])]
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
        ],
    )

    run_started = time.perf_counter()
    per_model_timings: dict[str, float] = {}
    contexts = []
    for name, module in selected:
        model_started = time.perf_counter()
        contexts.append(module.fit(args.config))
        per_model_timings[name] = time.perf_counter() - model_started

    if args.render:
        for context in contexts:
            qmd_path = os.path.join(context.reporting.output_dir, "index.qmd")
            heading(f"Rendering Quarto output: {qmd_path}")
            subprocess.run(["quarto", "render", qmd_path], check=True)

    if args.upload:
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
