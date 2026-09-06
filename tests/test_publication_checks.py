# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A published page must resolve, figures included (#289 task 4.10).

On 2026-09-03 the comparison book was published with ``index.html`` and none
of its 24 figures, and reported as published because the page returned 200.
The model-report upload had the same gap. These tests pin the three checks
that close it: the asset list is derived from the page, an upload that left a
referenced asset out is named, and every published file is requested back.
"""

from __future__ import annotations

import functools
import http.server
import threading

import pytest

from vocab_growth.publication_checks import (
    referenced_assets,
    unpublished_assets,
    verify_published,
)

_PAGE = """<html><head>
<link rel="stylesheet" href="index_files/style.css">
<link rel="stylesheet" href="https://cdn.example.org/site.css">
</head><body>
<img src="figures/trajectory.png">
<img src="figures/missing.png">
<img src="data:image/png;base64,AAAA">
<a href="#section">anchor</a>
<a href="index.html?tab=2">query</a>
<a href="mailto:someone@example.org">mail</a>
<a href="../outside.txt">outside</a>
</body></html>
"""


@pytest.fixture
def page(tmp_path):
    (tmp_path / "index_files").mkdir()
    (tmp_path / "index_files" / "style.css").write_text("body{}", encoding="utf-8")
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "trajectory.png").write_bytes(b"png")
    (tmp_path.parent / "outside.txt").write_text("x", encoding="utf-8")
    html = tmp_path / "index.html"
    html.write_text(_PAGE, encoding="utf-8")
    return html


def test_referenced_assets_are_derived_from_the_page(page):
    assets = referenced_assets(str(page))
    # Local files that exist, relative to the page, POSIX separators. External
    # URLs, data URIs, anchors, query links, mailto and a file outside the
    # page's directory tree are not assets the upload could have carried.
    assert assets == ["figures/trajectory.png", "index_files/style.css"]


def test_unpublished_assets_names_what_the_upload_left_out(page):
    assert unpublished_assets(str(page), ["index.html", "index_files/style.css"]) == [
        "figures/trajectory.png"
    ]
    # Backslash paths from a Windows walker compare equal.
    assert unpublished_assets(
        str(page), ["index.html", "index_files\\style.css", "figures\\trajectory.png"]
    ) == []


@pytest.fixture
def served(page):
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(page.parent)
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()


def test_verify_published_requests_every_file_back(served, page):
    assets = referenced_assets(str(page))
    assert verify_published(served, ["index.html", *assets]) == []
    # A file the page references but the upload did not carry is reported by
    # path, which is the 2026-09-03 failure made visible.
    failures = verify_published(served, ["index.html", "figures/absent.png"])
    assert failures == ["HTTPError figures/absent.png"]


def test_verify_published_reports_an_unreachable_host():
    failures = verify_published("http://127.0.0.1:9/", ["index.html"], timeout=1.0)
    assert len(failures) == 1
    assert failures[0].endswith(" index.html")
