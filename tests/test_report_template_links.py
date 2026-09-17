# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A model report template must not link outside its own output directory.

A template is staged into ``<output-root>/models/<label>/`` and published from
there as a standalone page, so a relative link that climbs out of that directory
-- ``../vg14/index.qmd``, ``../../notes/...`` -- resolves on GitHub and in the
checkout but 404s on the published page. ``upload.py`` refuses such a page, but
only after the upload has already gone out: on 2026-09-17 VG15, VG20 and VG25
were published and then failed verification for exactly this, while the other
six publication models passed. Link to the repository by absolute GitHub URL
instead, as the notes links elsewhere in these templates already do.

This reads the template text, so it needs no fit and no render.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

MODELS = Path(__file__).parents[1] / "docs" / "models"

#: Markdown link or image targets, and HTML ``href``/``src`` attributes, that
#: start by leaving the directory the page is published from.
_ESCAPING = re.compile(r"""(?:\]\(|\b(?:href|src)=["'])(\.\./[^)"'\s]*)""")


def _templates() -> list[Path]:
    return sorted(MODELS.glob("vg*/index.qmd")) + sorted(MODELS.glob("_*.qmd"))


def escaping_links(path: Path) -> list[str]:
    """Every relative link in ``path`` that leaves its published directory."""
    found = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        found.extend(f"line {number}: {m.group(1)}" for m in _ESCAPING.finditer(line))
    return found


@pytest.mark.parametrize("path", _templates(), ids=lambda p: p.relative_to(MODELS).as_posix())
def test_template_links_stay_inside_the_published_directory(path):
    assert escaping_links(path) == []


def test_the_scan_sees_each_link_form(tmp_path):
    page = tmp_path / "index.qmd"
    page.write_text(
        "[VG14](../vg14/index.qmd) and [note](../../notes/x.md#s)\n"
        '![fig](../figure.png) <a href="../other.html">o</a>\n'
        "[ok](posterior_summary.csv) [gh](https://github.com/x/y/blob/main/notes/x.md)\n",
        encoding="utf-8",
    )
    assert escaping_links(page) == [
        "line 1: ../vg14/index.qmd",
        "line 1: ../../notes/x.md#s",
        "line 2: ../figure.png",
        "line 2: ../other.html",
    ]
