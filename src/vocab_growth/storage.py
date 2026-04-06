# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import mimetypes
import os
import uuid
from urllib.parse import urlparse

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from rich import print


def upload_to_blob_storage(
    output_dir: str, model_label: str, *, include_traces: bool = False
) -> None:
    """Upload model output directory to Azure Blob Storage.

    Parameters
    ----------
    include_traces : bool
        If True, include NetCDF trace files (.nc). Excluded by default due to size.
    """
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

    parsed = urlparse(container_url.rstrip("/"))
    account_url = f"{parsed.scheme}://{parsed.netloc}"
    container_name = parsed.path.lstrip("/")

    run_id = uuid.uuid7()
    blob_prefix = f"projects/vocabulary-growth/output/{run_id}/{model_label}"

    print(f"\n[bold green]Uploading to Azure Blob Storage: {model_label}[/bold green]")
    print(f"  Source: {output_dir}")
    print(f"  Destination: {container_name}/{blob_prefix}/")

    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url, credential=credential)
    container_client = blob_service_client.get_container_client(container_name)

    uploaded = 0
    skipped = 0
    for root, _dirs, files in os.walk(output_dir):
        for filename in files:
            if not include_traces and filename.endswith(".nc"):
                skipped += 1
                continue

            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, output_dir).replace("\\", "/")
            blob_name = f"{blob_prefix}/{relative_path}"

            content_type, _ = mimetypes.guess_type(filename)
            if content_type is None:
                content_type = "application/octet-stream"

            with open(local_path, "rb") as f:
                container_client.upload_blob(
                    blob_name,
                    f,
                    overwrite=True,
                    content_settings=ContentSettings(content_type=content_type),
                )
            uploaded += 1

    if skipped:
        print(f"[yellow]Skipped {skipped} trace file(s) (.nc)[/yellow]")
    print(f"[bold green]Upload complete: {model_label} ({uploaded} files)[/bold green]")
