# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Give rendered DOCX files the math settings of ``docs/template.docx``.

Pandoc writes a DOCX's ``word/settings.xml`` from its own defaults and copies
only a fixed list of settings from the reference document. The math properties
(``m:mathPr``), which name the default math font, are not on that list, so Word
typesets every equation in Cambria Math whatever the template says. This copies
the template's ``m:mathPr`` into each rendered DOCX, so equations are set in
Noto Sans Math as in the HTML and PDF formats.

The report, paper and summary projects run it as a Quarto ``post-render``
script. Quarto passes the files it wrote in ``QUARTO_PROJECT_OUTPUT_FILES``, and
anything that is not a DOCX is skipped. It can also be run by hand:

    python scripts/restore_docx_math_settings.py output/report/vocabulary-growth-report.docx

It uses only the standard library, because Quarto runs it with whichever Python
it finds rather than the project environment.
"""

from __future__ import annotations

import os
import re
import sys
import zipfile
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "docs" / "template.docx"
SETTINGS = "word/settings.xml"
MATH_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/math"
MATH_PROPERTIES = re.compile(r"<m:mathPr\b.*?</m:mathPr>|<m:mathPr\b[^>]*/>", re.S)
ROOT = re.compile(r"<w:settings\b[^>]*>")

#: Elements that follow ``m:mathPr`` in the ``w:settings`` sequence (ECMA-376
#: Part 1, 17.15.1.78), in schema order. The properties go before the first one
#: present, or at the end when none is.
FOLLOWING = (
    "w:attachedSchema", "w:themeFontLang", "w:clrSchemeMapping",
    "w:doNotIncludeSubdocsInStats", "w:doNotAutoCompressPictures", "w:forceUpgrade",
    "w:captions", "w:readModeInkLockDown", "w:smartTagType", "sl:schemaLibrary",
    "w:shapeDefaults", "w:doNotEmbedSmartTags", "w:decimalSymbol", "w:listSeparator",
)


def template_math_properties(template: Path = TEMPLATE) -> str:
    with zipfile.ZipFile(template) as archive:
        settings = archive.read(SETTINGS).decode("utf-8")
    found = MATH_PROPERTIES.search(settings)
    if found is None:
        raise ValueError(f"{template} has no m:mathPr in {SETTINGS}")
    return found.group(0)


def with_math_properties(settings: str, math_properties: str) -> str:
    """``settings`` with its ``m:mathPr`` replaced by, or given, ``math_properties``."""
    if MATH_PROPERTIES.search(settings):
        return MATH_PROPERTIES.sub(lambda _: math_properties, settings, count=1)
    root = ROOT.search(settings)
    if root is None:
        raise ValueError(f"{SETTINGS} has no w:settings element")
    if "xmlns:m=" not in root.group(0):
        declared = root.group(0).replace("<w:settings", f'<w:settings xmlns:m="{MATH_NAMESPACE}"', 1)
        settings = settings[: root.start()] + declared + settings[root.end() :]
    positions = [settings.find(f"<{name}") for name in FOLLOWING]
    present = [position for position in positions if position >= 0]
    at = min(present) if present else settings.rindex("</w:settings>")
    return settings[:at] + math_properties + settings[at:]


def restore(docx: Path, math_properties: str) -> bool:
    """Rewrite ``docx`` in place; False when it already had these settings."""
    with zipfile.ZipFile(docx) as archive:
        entries = [(info, archive.read(info.filename)) for info in archive.infolist()]
    changed = False
    temporary = docx.with_name(docx.name + ".tmp")
    with zipfile.ZipFile(temporary, "w") as archive:
        for info, data in entries:
            if info.filename == SETTINGS:
                settings = data.decode("utf-8")
                updated = with_math_properties(settings, math_properties)
                changed = updated != settings
                data = updated.encode("utf-8")
            archive.writestr(info, data)
    if changed:
        temporary.replace(docx)
    else:
        temporary.unlink()
    return changed


def main(argv: list[str]) -> int:
    names = argv or os.environ.get("QUARTO_PROJECT_OUTPUT_FILES", "").splitlines()
    documents = [Path(name.strip()) for name in names if name.strip().lower().endswith(".docx")]
    if not documents:
        return 0
    math_properties = template_math_properties()
    for docx in documents:
        if restore(docx, math_properties):
            print(f"Restored the template's math settings in {docx}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
