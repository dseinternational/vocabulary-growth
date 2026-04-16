# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Fits the specified model to the latest data. Saves plots and data, and report to output directory.
"""

import argparse
import os
import subprocess
import dse_research_utils.environment.setup as setup
from multiprocessing import freeze_support
from rich import print
from vocab_growth.models import (
    model_vg01,
    model_vg02,
    model_vg03,
    model_vg04,
    model_vg05,
    model_vg06,
    model_vg07,
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
    }

    if args.model == "all":
        to_fit = list(models.values())
    elif args.model in models:
        to_fit = [models[args.model]]
    else:
        print(f"Unknown model: {args.model}")
        exit(1)

    contexts = [m.fit(args.config) for m in to_fit]

    if args.render:
        for context in contexts:
            qmd_path = os.path.join(context.reporting.output_dir, "index.qmd")
            print(f"\n[bold green]Rendering Quarto output: {qmd_path}[/bold green]")
            subprocess.run(["quarto", "render", qmd_path], check=True)

    if args.upload:
        for context in contexts:
            upload_to_blob_storage(
                context.reporting.output_dir,
                context.reporting.model_label,
                include_traces=args.include_traces,
            )
