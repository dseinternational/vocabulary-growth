# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Upload model output to Azure Blob Storage.
"""

import argparse
import os

from rich import print

from vocab_growth import environment as local_env
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.storage import upload_to_blob_storage

# Registry-derived so a newly added model is uploadable without editing this
# file (key -> (model_id, config_name), matching every model's output folder
# naming "{model_id}-{config_name}").
MODEL_CONFIGS = {
    key: (d.model_id, d.config_name) for key, d in MODEL_REGISTRY.items()
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload model output to Azure Blob Storage."
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model id (vg01–vg15) or 'all'.",
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
