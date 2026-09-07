# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Checks that a published page actually resolves (#289 task 4.10).

A rendered report references its figures by relative path, and an upload that
carries ``index.html`` without them publishes a page whose every image is
broken -- which is indistinguishable from a healthy one if only the page's own
URL is checked. That happened to the comparison book on 2026-09-03
(``scripts/publish_comparison.py`` records it), and nothing in the model-report
upload path checked for it either. These helpers are shared by both.

Since the shared library's 0.14.0 release the three steps are
:mod:`dse_research_utils.report.assets`, which parses the HTML rather than
matching quoted strings and reports each reference's state instead of dropping
the ones it cannot use. What stays here is the publication policy:

* **Navigation counts.** ``include_navigation=True`` is deliberate. These pages
  offer their summary tables as ordinary download links (``<a href="...csv">``),
  which the current checks already treat as required, and a link that 404s is
  the same broken publication as a missing image.
* **A missing reference is a failure, not an omission.** The previous scanner
  dropped any target that did not exist on disk, on the grounds that the render
  should have complained -- so a page referencing a figure that was never
  written published silently. :func:`local_failures` names it instead.
* **Pages are not followed.** Both publishers here upload a single rendered
  page and its assets, so ``follow_pages`` would only inspect pages nothing
  publishes as entry points. A multi-page book would need it.

Two encoding rules the previous scanner had wrong are now enforced by the
shared parser and are worth knowing before writing an artefact filename: an
``href`` is URL-decoded once, so a file literally named ``50%20.csv`` has to be
written ``50%2520.csv`` in the HTML (``50%20.csv`` asks the browser for
``50 .csv``); and ``base href``/``srcset``, which change what a browser
actually loads, are reported as unsupported rather than certified.
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

    ``published`` carries the uploader's **raw** relative filenames -- not URLs
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
