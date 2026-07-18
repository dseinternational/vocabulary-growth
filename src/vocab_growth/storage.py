# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vocab_growth.fit_artifacts import require_valid_fit
from vocab_growth.reporting import (
    format_duration,
    heading,
    key_value_table,
)

DEFAULT_PROJECT = "vocabulary-growth"


@dataclass(frozen=True)
class ValidatedFitOutput:
    """Path token returned only after publication validation succeeds."""

    output_dir: str


def validate_fit_for_upload(
    output_dir: str,
    validation_kwargs: dict[str, Any],
) -> ValidatedFitOutput:
    """Validate once before a batch uploads any model output."""
    require_valid_fit(output_dir, **validation_kwargs)
    return ValidatedFitOutput(output_dir=output_dir)


def upload_to_blob_storage(
    validated_output: ValidatedFitOutput,
    model_label: str,
    *,
    include_traces: bool = False,
    skip: Callable[[str], bool] | None = None,
) -> str:
    """Upload model output directory to Azure Blob Storage.

    Parameters
    ----------
    include_traces : bool
        If True, include NetCDF trace files (.nc). Excluded by default due to size.
    skip : callable, optional
        Predicate called with each file's POSIX-style path relative to
        ``output_dir``; return True to skip uploading that file. Use to exclude
        unreferenced artifacts (e.g. heavy SVG figures superseded by PNGs).

    Returns
    -------
    str
        The public URL of the uploaded ``index.html`` report.
    """
    from dse_research_utils.storage.azure import upload_directory_to_blob_storage

    output_dir = validated_output.output_dir

    heading(f"Uploading {model_label} to Azure Blob Storage")
    key_value_table(
        "Upload target",
        [
            ("Source", output_dir),
            ("Project", DEFAULT_PROJECT),
            ("Include traces", include_traces),
        ],
    )

    result = upload_directory_to_blob_storage(
        output_dir,
        model_label,
        project=DEFAULT_PROJECT,
        include_traces=include_traces,
        skip=skip,
    )

    if result.report_url is None:
        raise RuntimeError(f"No index.html report was uploaded for {model_label}.")

    key_value_table(
        f"Upload complete — {model_label}",
        [
            ("Files uploaded", result.uploaded_files),
            ("Files skipped", result.skipped_files),
            ("Bytes uploaded", f"{result.bytes_uploaded / 1_000_000:.1f} MB"),
            ("Elapsed", format_duration(result.elapsed_seconds)),
            ("Prefix", result.prefix_url),
            ("Report URL", result.report_url),
        ],
    )
    return result.report_url
