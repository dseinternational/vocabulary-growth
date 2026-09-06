# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vocab_growth.fit_artifacts import require_valid_fit
from vocab_growth.publication_checks import (
    referenced_assets,
    unpublished_assets,
    verify_published,
)
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
    verify: bool = True,
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
    verify : bool
        After uploading, check that every asset the report's ``index.html``
        references was uploaded, then request the page and each asset back
        over HTTP and fail on anything that does not return 200 (#289 task
        4.10). The upload is not reported complete until it has. This is the
        check ``publish_comparison.py`` performs for the comparison book,
        after a hand-assembled upload published that book with every image
        broken; the model reports had the same gap.

    Returns
    -------
    str
        The public URL of the uploaded ``index.html`` report.

    Raises
    ------
    RuntimeError
        No ``index.html`` was uploaded; a referenced asset was left out by
        ``skip`` or the trace exclusion; or a published file did not return
        200.
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

    rows = [
        ("Files uploaded", result.uploaded_files),
        ("Files skipped", result.skipped_files),
        ("Bytes uploaded", f"{result.bytes_uploaded / 1_000_000:.1f} MB"),
        ("Elapsed", format_duration(result.elapsed_seconds)),
        ("Prefix", result.prefix_url),
        ("Report URL", result.report_url),
    ]
    if verify:
        rows.append(("Verified", _verify_report_upload(output_dir, model_label, result)))
    key_value_table(f"Upload complete — {model_label}", rows)
    return result.report_url


def _verify_report_upload(output_dir: str, model_label: str, result) -> str:
    """Every referenced asset was sent, and every published file resolves.

    Returns the one-line summary for the upload table; raises on any failure so
    a broken publication is never reported as complete.
    """
    index_html = os.path.join(output_dir, "index.html")
    missing = unpublished_assets(index_html, result.relative_paths)
    if missing:
        shown = ", ".join(missing[:8]) + (", ..." if len(missing) > 8 else "")
        raise RuntimeError(
            f"{model_label}: index.html references {len(missing)} asset(s) that "
            f"were not uploaded (excluded by the skip filter or the trace "
            f"exclusion): {shown}. The page would publish with those missing."
        )
    assets = referenced_assets(index_html)
    failures = verify_published(result.prefix_url, ["index.html", *assets])
    if failures:
        shown = ", ".join(failures[:8]) + (", ..." if len(failures) > 8 else "")
        raise RuntimeError(
            f"{model_label}: {len(failures)} of {len(assets) + 1} published files "
            f"did not return 200 under {result.prefix_url}: {shown}"
        )
    return f"index.html and all {len(assets)} referenced assets return 200"
