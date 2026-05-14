# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Upload model output to Azure Blob Storage.
"""

import argparse
import os

from rich import print

from vocab_growth import environment as local_env
from vocab_growth.storage import upload_to_blob_storage

MODEL_CONFIGS = {
    "vg01": ("VG01", "age-spoken-ds"),
    "vg02": ("VG02", "age-understood-ds"),
    "vg03": ("VG03", "age-spoken-td"),
    "vg04": ("VG04", "age-understood-td"),
    "vg05": ("VG05", "age-understood-spoken-ds"),
    "vg06": ("VG06", "age-understood-spoken-td"),
    "vg07": ("VG07", "age-understood-spoken-ds-re"),
    "vg08": ("VG08", "age-understood-spoken-ds-re-subj"),
    "vg09": ("VG09", "age-understood-spoken-ds-re-subj-uq"),
    "vg09b": ("VG09B", "age-understood-spoken-ds-re-subj-uq-anchored"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload model output to Azure Blob Storage."
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model id (vg01, vg02, vg03, vg04, vg05, vg06, vg07, vg08, vg09, vg09b) or 'all'.",
    )
    parser.add_argument(
        "--include-traces",
        action="store_true",
        help="Include trace files (.nc) in the upload (excluded by default).",
    )

    args = parser.parse_args()

    if args.model == "all":
        to_upload = list(MODEL_CONFIGS.items())
    elif args.model in MODEL_CONFIGS:
        to_upload = [(args.model, MODEL_CONFIGS[args.model])]
    else:
        print(f"[bold red]Unknown model: {args.model}[/bold red]")
        exit(1)

    for model_id, (model_name, config_name) in to_upload:
        model_label = f"{model_name}-{config_name}"
        output_dir = os.path.join(local_env.MODELS_OUTPUT_DIR, model_label)

        if not os.path.isdir(output_dir):
            print(
                f"[bold red]Output directory not found for {model_id}: {output_dir}[/bold red]"
            )
            print("Run fit_model.py first to generate model output.")
            exit(1)

        upload_to_blob_storage(
            output_dir, model_label, include_traces=args.include_traces
        )
