# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check local report assets, upload inventories and published HTTP responses.

Both model and comparison publishers use the shared
``dse_research_utils.report.assets`` parser. Navigation links count as required
assets because reports offer CSV downloads through ordinary links. Missing
files, paths outside the publication root and unsupported reference forms fail
validation. Inspection does not follow linked pages; these publishers upload
one rendered page and its assets.

HTML paths are URL-decoded once. A file literally named ``50%20.csv`` therefore
needs ``50%2520.csv`` in its ``href``. ``base href`` and ``srcset`` references are
reported as unsupported because they change how browsers resolve assets.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from dse_research_utils.report.assets import (
    AssetFailure,
    AssetInspection,
    check_uploaded_assets,
    inspect_local_assets,
    verify_published_assets,
)

__all__ = [
    "AssetFailure",
    "AssetInspection",
    "check_uploaded_assets",
    "describe_failures",
    "inspect_report",
    "local_failures",
    "present_assets",
    "referenced_assets",
    "upload_failures",
    "verify_published",
]


def inspect_report(html_path: str) -> AssetInspection:
    """Every local file the rendered page requires, with each one's state.

    The inspection root is the page's own directory, which is the directory an
    upload publishes, so the paths compare directly with the uploader's record
    of what it sent.
    """
    return inspect_local_assets(html_path, include_navigation=True)


def present_assets(inspection: AssetInspection) -> list[str]:
    """The required local files that exist, as POSIX paths relative to the root.

    Excludes the inspected page itself, which a caller publishes separately.
    Callers that copy these paths should check :func:`local_failures` first:
    this list is what *can* be copied, not a statement that nothing is missing.
    """
    present = {
        reference.relative_path
        for reference in inspection.references
        if reference.required and reference.status == "present"
    }
    return sorted(present - set(inspection.pages))


def referenced_assets(html_path: str) -> list[str]:
    """:func:`present_assets` for a page, inspecting it first."""
    return present_assets(inspect_report(html_path))


def local_failures(inspection: AssetInspection) -> tuple[AssetFailure, ...]:
    """Required references the page makes that no upload could satisfy.

    A missing file, a link that escapes the published directory, and a
    construct whose target cannot be determined (``base href``, ``srcset``) all
    land here. Passing the inspection's own required paths back as the
    prospective inventory checks the page without making any request.
    """
    return check_uploaded_assets(inspection, inspection.required_paths)


def upload_failures(
    html_path: str, published: Iterable[str]
) -> tuple[AssetFailure, ...]:
    """Local failures plus required files this upload left out.

    ``published`` carries the uploader's raw relative filenames, not URLs
    and not URL-encoded paths. Either separator is accepted: an uploader
    walking a Windows directory yields backslashes whatever platform later
    checks the record. Callers that also need the inspection should call
    :func:`inspect_report` and :func:`check_uploaded_assets` themselves rather
    than parse the page twice.
    """
    return check_uploaded_assets(inspect_report(html_path), published)


def verify_published(
    base_url: str,
    relative_paths: Iterable[str],
    *,
    timeout: float = 30.0,
    fetch_status: Callable[[str, float], int] | None = None,
) -> tuple[AssetFailure, ...]:
    """Request ``base_url/<path>`` for every path; return the ones not returning 200.

    ``base_url`` is the directory URL the files were published under, with or
    without a trailing slash. ``relative_paths`` contains raw filenames, each
    encoded exactly once. A redirect is a failure: a login page reached through
    one cannot establish that the asset is published.
    """
    return verify_published_assets(
        base_url, relative_paths, timeout=timeout, fetch_status=fetch_status
    )


def describe_failures(failures: Iterable[AssetFailure]) -> list[str]:
    """One printable line per failure: the path, why, and any HTTP status."""
    return [
        f"{failure.path} ({failure.reason}"
        + (f" {failure.status_code}" if failure.status_code is not None else "")
        + ")"
        for failure in failures
    ]
