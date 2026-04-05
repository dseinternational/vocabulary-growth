# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import subprocess

from rich import print


def upload_to_blob_storage(output_dir: str, model_label: str) -> None:
    """Upload model output directory to Azure Blob Storage using AzCopy."""
    container_url = os.environ.get("DSERESEARCH_BLOB_CONTAINER_URL")
    if not container_url:
        print(
            "[bold red]Error: DSERESEARCH_BLOB_CONTAINER_URL environment variable is not set.[/bold red]"
        )
        print("Set it to your Azure Blob container URL, e.g.:")
        print(
            "  export DSERESEARCH_BLOB_CONTAINER_URL='https://<account>.blob.core.windows.net/<container>'"
        )
        raise RuntimeError(
            "DSERESEARCH_BLOB_CONTAINER_URL environment variable is not set."
        )

    source = output_dir.replace("\\", "/").rstrip("/") + "/*"
    destination = container_url.rstrip("/") + "/projects/vocabulary-growth/output/" + model_label + "/"

    print(f"\n[bold green]Uploading to Azure Blob Storage: {model_label}[/bold green]")
    print(f"  Source: {source}")
    print(f"  Destination: {destination}")

    try:
        subprocess.run(
            ["azcopy", "copy", source, destination, "--recursive"],
            check=True,
        )
    except FileNotFoundError:
        print(
            "[bold red]Error: `azcopy` was not found. Please install AzCopy and ensure it is available on PATH.[/bold red]"
        )
        raise
    except subprocess.CalledProcessError as error:
        print(
            f"[bold red]Error: AzCopy upload failed for {model_label} (exit code {error.returncode}).[/bold red]"
        )
        raise
    else:
        print(f"[bold green]Upload complete: {model_label}[/bold green]")
