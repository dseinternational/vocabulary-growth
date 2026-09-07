# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vocab_growth.fit_artifacts import require_valid_fit
from vocab_growth.publication_checks import (
    check_uploaded_assets,
    describe_failures,
    inspect_report,
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
    fetch_status: Callable[[str, float], int] | None = None,
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
    fetch_status : callable, optional
        Transport for the HTTP check, receiving ``(url, timeout)`` and
        returning an integer status. Defaults to the shared bounded GET that
        refuses redirects. Supplied by the tests so the publication path can be
        exercised end to end without reaching a network.

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
        rows.append(
            (
                "Verified",
                _verify_report_upload(
                    output_dir, model_label, result, fetch_status=fetch_status
                ),
            )
        )
    key_value_table(f"Upload complete — {model_label}", rows)
    return result.report_url


def _verify_report_upload(
    output_dir: str,
    model_label: str,
    result,
    *,
    fetch_status: Callable[[str, float], int] | None = None,
) -> str:
    """Every referenced asset was sent, and every published file resolves.

    Returns the one-line summary for the upload table; raises on any failure so
    a broken publication is never reported as complete. The local and inventory
    check runs first and no request is made until it passes: a reference the
    page makes to a file that was never written, or that the skip filter or the
    trace exclusion left out, is a broken publication whatever the server would
    have answered.
    """
    index_html = os.path.join(output_dir, "index.html")
    inspection = inspect_report(index_html)
    failures = check_uploaded_assets(inspection, result.relative_paths)
    if failures:
        shown = describe_failures(failures)
        listed = ", ".join(shown[:8]) + (", ..." if len(shown) > 8 else "")
        raise RuntimeError(
            f"{model_label}: index.html requires {len(failures)} file(s) that this "
            f"upload cannot serve — missing locally, or excluded by the skip filter "
            f"or the trace exclusion: {listed}. The page would publish broken."
        )
    # The entry page first, deliberately: the shared check requests paths in the
    # order given, and a publication whose own page is unreachable is not worth
    # checking asset by asset.
    entry = inspection.pages[0]
    required = (entry, *(path for path in inspection.required_paths if path != entry))
    http_failures = verify_published(
        result.prefix_url, required, fetch_status=fetch_status
    )
    if http_failures:
        shown = describe_failures(http_failures)
        listed = ", ".join(shown[:8]) + (", ..." if len(shown) > 8 else "")
        raise RuntimeError(
            f"{model_label}: {len(http_failures)} of {len(required)} published files "
            f"did not return 200 under {result.prefix_url}: {listed}"
        )
    return f"index.html and all {len(required) - 1} referenced assets return 200"
