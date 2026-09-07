# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A published page must resolve, figures included (#289 task 4.10).

On 2026-09-03 the comparison book was published with ``index.html`` and none
of its 24 figures, and reported as published because the page returned 200.
The model-report upload had the same gap. These tests pin the three checks
that close it: the asset list is derived from the page, an upload that left a
referenced asset out is named, and every published file is requested back.

Since the shared library's 0.14.0 release the parsing is
``dse_research_utils.report.assets``, and two of these tests changed with it. A
reference to a file that is not there is now a failure rather than something
quietly left off the list; and an ``href`` is URL-decoded once, so linking a
file literally named ``50%20.csv`` as ``href="50%20.csv"`` asks the browser for
``50 .csv`` and is a broken link, not a compatibility target. Both forms are
kept below.
"""

from __future__ import annotations

import functools
import http.server
import threading
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from vocab_growth.publication_checks import (
    describe_failures,
    inspect_report,
    local_failures,
    referenced_assets,
    upload_failures,
    verify_published,
)

_PAGE = """<html><head>
<link rel="stylesheet" href="index_files/style.css">
<link rel="stylesheet" href="https://cdn.example.org/site.css">
</head><body>
<img src="figures/trajectory.png">
<img src="data:image/png;base64,AAAA">
<a href="tables/summary.csv">download</a>
<a href="#section">anchor</a>
<a href="index.html?tab=2">query</a>
<a href="mailto:someone@example.org">mail</a>
</body></html>
"""

#: The same page plus the two references no upload could satisfy: a figure that
#: was never written, and a link out of the published directory.
_BROKEN_PAGE = _PAGE.replace(
    "</body>", '<img src="figures/missing.png"><a href="../outside.txt">out</a></body>'
)


def _write_page(directory, html):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "index_files").mkdir(exist_ok=True)
    (directory / "index_files" / "style.css").write_text("body{}", encoding="utf-8")
    (directory / "figures").mkdir(exist_ok=True)
    (directory / "figures" / "trajectory.png").write_bytes(b"png")
    (directory / "tables").mkdir(exist_ok=True)
    (directory / "tables" / "summary.csv").write_text("a,b\n", encoding="utf-8")
    (directory.parent / "outside.txt").write_text("x", encoding="utf-8")
    target = directory / "index.html"
    target.write_text(html, encoding="utf-8")
    return target


@pytest.fixture
def page(tmp_path):
    return _write_page(tmp_path, _PAGE)


@pytest.fixture
def broken_page(tmp_path):
    return _write_page(tmp_path, _BROKEN_PAGE)


def test_referenced_assets_are_derived_from_the_page(page):
    assets = referenced_assets(str(page))
    # Local files that exist, relative to the page, POSIX separators. External
    # URLs, data URIs, anchors and mailto are not assets the upload could carry;
    # the query link resolves to the page itself, which the caller publishes
    # separately. The CSV download link *is* required: these reports offer their
    # summary tables that way, and a 404 there is as broken as a missing image.
    assert assets == [
        "figures/trajectory.png",
        "index_files/style.css",
        "tables/summary.csv",
    ]
    assert local_failures(inspect_report(str(page))) == ()


def test_a_reference_no_upload_could_satisfy_is_named_not_dropped(broken_page):
    """The previous scanner left both of these off the list entirely.

    A page referencing a figure that was never written then published silently,
    which is the 2026-09-03 failure with the missing file one step earlier.
    """
    failures = {failure.path: failure.reason for failure in local_failures(inspect_report(str(broken_page)))}
    assert failures == {"figures/missing.png": "missing", "../outside.txt": "outside_root"}
    # The list a publisher copies from still contains only files that exist.
    assert "figures/missing.png" not in referenced_assets(str(broken_page))


def test_upload_failures_name_what_the_upload_left_out(page):
    assert describe_failures(
        upload_failures(str(page), ["index.html", "index_files/style.css"])
    ) == ["figures/trajectory.png (not_uploaded)", "tables/summary.csv (not_uploaded)"]
    # Backslash paths from a Windows walker compare equal.
    assert (
        upload_failures(
            str(page),
            [
                "index.html",
                "index_files\\style.css",
                "figures\\trajectory.png",
                "tables\\summary.csv",
            ],
        )
        == ()
    )


def test_an_inventory_entry_cannot_hide_a_missing_local_file(broken_page):
    """Claiming to have uploaded a file that is not there is not a pass."""
    reasons = {
        failure.reason
        for failure in upload_failures(
            str(broken_page),
            [
                "index.html",
                "index_files/style.css",
                "figures/trajectory.png",
                "figures/missing.png",
                "tables/summary.csv",
            ],
        )
    }
    assert reasons == {"missing", "outside_root"}


@pytest.mark.parametrize(
    ("markup", "reason"),
    [
        ('<base href="assets/">', "unsupported_base"),
        ('<img src="figures/trajectory.png" srcset="figures/trajectory.png 2x">', "unsupported_srcset"),
        ('<img src="/figures/trajectory.png">', "root_relative_url"),
    ],
)
def test_constructs_that_change_what_a_browser_loads_are_not_certified(
    tmp_path, markup, reason
):
    """Each of these decides which file is actually requested.

    Ignoring them would let the check certify a bundle the browser does not
    load: ``base href`` re-roots every relative URL, ``srcset`` can select a
    file nothing else references, and a root-relative URL resolves at the site
    origin rather than under the upload prefix.
    """
    page = _write_page(tmp_path, _PAGE.replace("<body>", "<body>" + markup))
    assert reason in {failure.reason for failure in local_failures(inspect_report(str(page)))}


def test_a_literal_percent_in_a_filename_has_to_be_encoded_twice(tmp_path):
    """``href="50%20.csv"`` asks the browser for ``50 .csv``.

    The previous scanner matched the raw attribute against the filesystem, so
    it reported this page as complete while a reader clicking the link got a
    404. The correctly encoded form is ``50%2520.csv``; both are pinned.
    """
    (tmp_path / "50%20.csv").write_text("a,b\n", encoding="utf-8")
    mismatched = _write_page(
        tmp_path / "raw", _PAGE.replace("<body>", '<body><a href="50%20.csv">table</a>')
    )
    (tmp_path / "raw" / "50%20.csv").write_text("a,b\n", encoding="utf-8")
    assert [failure.path for failure in local_failures(inspect_report(str(mismatched)))] == ["50 .csv"]

    encoded = _write_page(
        tmp_path / "enc", _PAGE.replace("<body>", '<body><a href="50%2520.csv">table</a>')
    )
    (tmp_path / "enc" / "50%20.csv").write_text("a,b\n", encoding="utf-8")
    assert local_failures(inspect_report(str(encoded))) == ()
    assert "50%20.csv" in referenced_assets(str(encoded))


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
    assert verify_published(served, ["index.html", *assets]) == ()
    # A file the page references but the upload did not carry is reported by
    # path and status, which is the 2026-09-03 failure made visible.
    failures = verify_published(served, ["index.html", "figures/absent.png"])
    assert describe_failures(failures) == ["figures/absent.png (http_status 404)"]


def test_verify_published_reports_an_unreachable_host():
    failures = verify_published("http://127.0.0.1:9/", ["index.html"], timeout=1.0)
    assert len(failures) == 1
    assert failures[0].path == "index.html" and failures[0].status_code is None


def test_verify_published_can_be_driven_through_an_injected_transport():
    """Each distinct path is requested once, in the order given, encoded once.

    The transport is injectable so the encoding and the request order can be
    checked without a server, and so a caller can supply one that follows the
    same no-redirect rule.
    """
    requests = []

    def fetch_status(url: str, timeout: float) -> int:
        requests.append(url)
        return 404 if url.endswith("missing.png") else 200

    failures = verify_published(
        "https://example.org/report%20one/",
        ["index.html", "tables/a+b.csv", "index.html", "missing.png"],
        fetch_status=fetch_status,
    )
    assert requests == [
        "https://example.org/report%20one/index.html",
        "https://example.org/report%20one/tables/a%2Bb.csv",
        "https://example.org/report%20one/missing.png",
    ]
    assert describe_failures(failures) == ["missing.png (http_status 404)"]


@pytest.fixture
def upload_report(tmp_path, monkeypatch):
    from vocab_growth.storage import ValidatedFitOutput

    names = ["figures/psi (dev).png", "tables/a+b.csv", "figures/café.png", "tables/50%20.csv", "assets/index.html"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("asset", encoding="utf-8")
    # A literal percent sign in a filename is `%25` in an href, so the file
    # named `50%20.csv` is linked as `50%2520.csv`. The uploader still sends,
    # and the HTTP check still encodes, the raw name.
    (tmp_path / "index.html").write_text(
        "".join(
            f'<a href="{name.replace("%", "%25")}">asset</a>' for name in names
        ),
        encoding="utf-8",
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

    def fetch_status(url, timeout):
        requested.append(url)
        return 200

    return ValidatedFitOutput(str(tmp_path)), names, sent, requested, fetch_status


def test_upload_matches_raw_names_and_requests_encoded_urls(upload_report):
    from vocab_growth.storage import upload_to_blob_storage

    output, names, sent, requested, fetch_status = upload_report
    report = upload_to_blob_storage(output, "model (dev)", fetch_status=fetch_status)
    prefix = report.removesuffix("index.html")
    assert "model%20%28dev%29/" in prefix
    # The entry page is requested first, deliberately: a publication whose own
    # page is unreachable is not worth checking asset by asset.
    assert requested == [report, *(prefix + quote(name, safe="/") for name in sorted(names))]
    assert len(sent) == len(names) + 1
    assert not any(name.endswith("trace.nc") for name in sent)
    assert report.endswith("/index.html") and not report.endswith("/assets/index.html")


@pytest.mark.parametrize("skipped", ["figures/psi (dev).png", "tables/a+b.csv", "figures/café.png"])
def test_upload_still_rejects_skipped_referenced_files(upload_report, skipped):
    from vocab_growth.storage import upload_to_blob_storage

    output, _, _, requested, fetch_status = upload_report
    with pytest.raises(RuntimeError, match="this upload cannot serve") as error:
        upload_to_blob_storage(
            output, "model", skip=lambda name: name == skipped, fetch_status=fetch_status
        )
    assert skipped in str(error.value)
    assert requested == []


def test_nested_index_does_not_replace_skipped_root(upload_report):
    from vocab_growth.storage import upload_to_blob_storage

    output, _, _, requested, fetch_status = upload_report
    with pytest.raises(RuntimeError, match="No index.html report"):
        upload_to_blob_storage(
            output,
            "model",
            skip=lambda name: name == "index.html",
            fetch_status=fetch_status,
        )
    assert requested == []


def test_upload_fails_when_a_published_file_does_not_resolve(upload_report):
    """A 200 on the page says nothing about the files it needs.

    This is the 2026-09-03 failure exactly: `index.html` answered, every figure
    did not, and the publication was reported complete. Driven through the
    injected transport, so the check runs without a network.
    """
    from vocab_growth.storage import upload_to_blob_storage

    output, _, _, requested, _ = upload_report

    def fetch_status(url, timeout):
        requested.append(url)
        # The entry page answers; nothing beside it does.
        return 200 if len(requested) == 1 else 404

    with pytest.raises(RuntimeError, match="did not return 200") as error:
        upload_to_blob_storage(output, "model", fetch_status=fetch_status)
    message = str(error.value)
    assert requested[0].endswith("/index.html")
    assert "5 of 6 published files" in message
    assert "http_status 404" in message
