# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import mimetypes
import os
import time
import uuid
from urllib.parse import urlparse

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings

from vocab_growth.reporting import (
    console,
    format_duration,
    heading,
    key_value_table,
)


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
        console.print(
            "[bold red]Error: DSERESEARCH_BLOB_CONTAINER_URL environment variable is not set.[/bold red]"
        )
        console.print("Set it to your Azure Blob container URL, e.g.:")
        console.print(
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

    heading(f"Uploading {model_label} to Azure Blob Storage")
    key_value_table(
        "Upload target",
        [
            ("Source", output_dir),
            ("Container", container_name),
            ("Prefix", f"{blob_prefix}/"),
            ("Include traces", include_traces),
        ],
    )

    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url, credential=credential)
    container_client = blob_service_client.get_container_client(container_name)

    uploaded = 0
    skipped = 0
    bytes_sent = 0
    started = time.perf_counter()
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

            bytes_sent += os.path.getsize(local_path)
            with open(local_path, "rb") as f:
                container_client.upload_blob(
                    blob_name,
                    f,
                    overwrite=True,
                    content_settings=ContentSettings(content_type=content_type),
                )
            uploaded += 1

    key_value_table(
        f"Upload complete — {model_label}",
        [
            ("Files uploaded", uploaded),
            ("Files skipped (.nc)", skipped),
            ("Bytes uploaded", f"{bytes_sent / 1_000_000:.1f} MB"),
            ("Elapsed", format_duration(time.perf_counter() - started)),
        ],
    )
