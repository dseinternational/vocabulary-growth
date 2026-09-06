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
from contextlib import nullcontext
from types import SimpleNamespace
from urllib.parse import quote

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


@pytest.fixture
def upload_report(tmp_path, monkeypatch):
    from vocab_growth.storage import ValidatedFitOutput

    names = ["figures/psi (dev).png", "tables/a+b.csv", "figures/café.png", "tables/50%20.csv", "assets/index.html"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("asset", encoding="utf-8")
    (tmp_path / "index.html").write_text(
        "".join(f'<a href="{name}">asset</a>' for name in names), encoding="utf-8"
    )
    (tmp_path / "trace.nc").write_bytes(b"excluded")
    sent, requested = [], []

    def upload_blob(name, data, **kwargs):
        sent.append(name)
        assert data.read()

    client = SimpleNamespace(get_container_client=lambda name: SimpleNamespace(upload_blob=upload_blob))
    monkeypatch.setattr("azure.storage.blob.BlobServiceClient", lambda *a, **k: client)
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", lambda: object())
    monkeypatch.setenv("DSERESEARCH_BLOB_CONTAINER_URL", "https://acct.blob.core.windows.net/reports")

    def request(url, **kwargs):
        requested.append(url)
        return nullcontext(SimpleNamespace(status=200))

    monkeypatch.setattr("urllib.request.urlopen", request)
    return ValidatedFitOutput(str(tmp_path)), names, sent, requested


def test_upload_matches_raw_names_and_requests_encoded_urls(upload_report):
    from vocab_growth.storage import upload_to_blob_storage

    output, names, sent, requested = upload_report
    report = upload_to_blob_storage(output, "model (dev)")
    prefix = report.removesuffix("index.html")
    assert "model%20%28dev%29/" in prefix
    assert requested == [report, *(prefix + quote(name, safe="/") for name in sorted(names))]
    assert len(sent) == len(names) + 1
    assert not any(name.endswith("trace.nc") for name in sent)
    assert report.endswith("/index.html") and not report.endswith("/assets/index.html")


@pytest.mark.parametrize("skipped", ["figures/psi (dev).png", "tables/a+b.csv", "figures/café.png"])
def test_upload_still_rejects_skipped_referenced_files(upload_report, skipped):
    from vocab_growth.storage import upload_to_blob_storage

    output, _, _, requested = upload_report
    with pytest.raises(RuntimeError, match="were not uploaded") as error:
        upload_to_blob_storage(output, "model", skip=lambda name: name == skipped)
    assert skipped in str(error.value)
    assert requested == []


def test_nested_index_does_not_replace_skipped_root(upload_report):
    from vocab_growth.storage import upload_to_blob_storage

    output, _, _, requested = upload_report
    with pytest.raises(RuntimeError, match="No index.html report"):
        upload_to_blob_storage(output, "model", skip=lambda name: name == "index.html")
    assert requested == []
