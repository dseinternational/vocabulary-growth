# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Checks that a published page actually resolves (#289 task 4.10).

A rendered report references its figures by relative path, and an upload that
carries ``index.html`` without them publishes a page whose every image is
broken -- which is indistinguishable from a healthy one if only the page's own
URL is checked. That happened to the comparison book on 2026-09-03
(``scripts/publish_comparison.py`` records it), and nothing in the model-report
upload path checked for it either. These helpers are shared by both:

* :func:`referenced_assets` derives the asset list **from the page** rather
  than from a remembered list, because a list that omits something is exactly
  the failure being guarded against.
* :func:`unpublished_assets` names the referenced assets an upload left out,
  whether by a caller's skip filter or the trace exclusion.
* :func:`verify_published` requests every published file and returns the ones
  that did not come back ``200``.
"""

from __future__ import annotations

import os
import re
import urllib.request
from collections.abc import Iterable
from urllib.parse import quote

#: ``src``/``href`` targets in rendered HTML that point at a local file.
_ASSET = re.compile(r'(?:src|href)="([^"#?][^"]*?)"')

#: Targets that are not local files, whatever the attribute.
_EXTERNAL_PREFIXES = ("http://", "https://", "//", "data:", "mailto:", "javascript:")


def referenced_assets(html_path: str) -> list[str]:
    """Every local file the rendered page references, as POSIX relative paths.

    Relative to the page's own directory, which is the directory an upload
    publishes, so the paths compare directly with the uploader's record of
    what it sent. A target that does not exist on disk is not an asset the
    page could have shipped and is left out here; the render is what should
    have complained about it.
    """
    with open(html_path, encoding="utf-8") as handle:
        html = handle.read()
    base = os.path.dirname(os.path.abspath(html_path))
    assets: set[str] = set()
    for target in _ASSET.findall(html):
        if target.startswith(_EXTERNAL_PREFIXES):
            continue
        candidate = os.path.normpath(os.path.join(base, target))
        if not os.path.isfile(candidate):
            continue
        relative = os.path.relpath(candidate, base).replace(os.sep, "/")
        if relative.startswith("../"):
            # Outside the page's directory: an upload of that directory cannot
            # carry it, so it is a link out of the page rather than an asset.
            continue
        assets.add(relative)
    return sorted(assets)


def unpublished_assets(html_path: str, published: Iterable[str]) -> list[str]:
    """Referenced assets absent from ``published`` (paths relative to the page).

    ``published`` may carry either separator: an uploader walking a Windows
    directory yields backslashes whatever platform later checks the record, so
    both are normalised to ``/`` unconditionally rather than through ``os.sep``,
    which is already ``/`` on POSIX and would leave a backslash path unmatched.
    """
    sent = {path.replace("\\", "/") for path in published}
    return [asset for asset in referenced_assets(html_path) if asset not in sent]


def verify_published(
    base_url: str, relative_paths: Iterable[str], *, timeout: float = 30.0
) -> list[str]:
    """Request ``base_url/<path>`` for every path; return the ones not returning 200.

    Each failure is ``"<status or exception> <path>"``, so the caller can print
    them as a list. ``base_url`` is the directory URL the files were published
    under, with or without a trailing slash. ``relative_paths`` contains raw
    filenames, not URL-encoded paths; each is encoded exactly once.
    """
    failures: list[str] = []
    root = base_url.rstrip("/")
    for relative in relative_paths:
        url = f"{root}/{quote(relative, safe='/')}"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                if response.status != 200:
                    failures.append(f"{response.status} {relative}")
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            failures.append(f"{type(exc).__name__} {relative}")
    return failures
