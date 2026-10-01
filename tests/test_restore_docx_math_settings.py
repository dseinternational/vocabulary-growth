# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``scripts/restore_docx_math_settings.py`` and the DOCX template's fonts.

Pandoc copies only a fixed list of settings from the reference document, and
the math properties are not on it, so a rendered DOCX would set its equations
in Cambria Math. The post-render script copies the template's ``m:mathPr`` back.
These tests pin that the template names Noto Sans, Noto Sans Mono and Noto Sans
Math, and that
the script inserts the properties in schema order, replaces rather than
duplicates them, and leaves every other part of the package untouched.
"""

import importlib.util
import re
import sys
import zipfile
from pathlib import Path

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "restore_docx_math_settings.py"
_SPEC = importlib.util.spec_from_file_location("restore_docx_math_settings_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

# The shape of Pandoc's settings.xml: no m:mathPr, rsids then themeFontLang.
_PANDOC_SETTINGS = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<w:settings xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    '<w:zoom w:percent="130" /><w:rsids><w:rsidRoot w:val="00B061A4" /></w:rsids>'
    '<w:themeFontLang w:val="en-US" /><w:decimalSymbol w:val="." /></w:settings>'
)


def _math_font(settings: str) -> str | None:
    found = re.search(r'<m:mathFont m:val="([^"]+)"', settings)
    return found.group(1) if found else None


def test_template_names_noto_sans_mono_and_math():
    with zipfile.ZipFile(_MODULE.TEMPLATE) as archive:
        theme = archive.read("word/theme/theme1.xml").decode("utf-8")
        styles = archive.read("word/styles.xml").decode("utf-8")
        settings = archive.read("word/settings.xml").decode("utf-8")
    assert re.findall(r'<a:latin typeface="([^"]*)"', theme) == ["Noto Sans", "Noto Sans"]
    code_fonts = set(re.findall(r'<w:rFonts w:ascii="([^"]+)" w:hAnsi="([^"]+)"/>', styles))
    assert code_fonts == {("Noto Sans Mono", "Noto Sans Mono")}
    assert _math_font(settings) == "Noto Sans Math"


def test_properties_go_before_the_first_following_element():
    math_properties = _MODULE.template_math_properties()
    updated = _MODULE.with_math_properties(_PANDOC_SETTINGS, math_properties)
    assert _math_font(updated) == "Noto Sans Math"
    assert updated.index("</w:rsids>") < updated.index("<m:mathPr>") < updated.index("<w:themeFontLang")
    assert _MODULE.with_math_properties(updated, math_properties) == updated


def test_existing_properties_are_replaced_not_duplicated():
    cambria = '<m:mathPr><m:mathFont m:val="Cambria Math"/></m:mathPr>'
    settings = _PANDOC_SETTINGS.replace("<w:themeFontLang", cambria + "<w:themeFontLang")
    updated = _MODULE.with_math_properties(settings, _MODULE.template_math_properties())
    assert updated.count("<m:mathPr>") == 1
    assert _math_font(updated) == "Noto Sans Math"


def test_math_namespace_is_declared_when_missing():
    settings = _PANDOC_SETTINGS.replace(
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" ', ""
    )
    updated = _MODULE.with_math_properties(settings, _MODULE.template_math_properties())
    root = re.search(r"<w:settings\b[^>]*>", updated).group(0)
    assert f'xmlns:m="{_MODULE.MATH_NAMESPACE}"' in root


def test_main_rewrites_only_the_rendered_docx_settings(tmp_path, monkeypatch):
    docx = tmp_path / "report.docx"
    other_parts = {"[Content_Types].xml": b"<Types/>", "word/document.xml": b"<w:document/>"}
    with zipfile.ZipFile(docx, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in other_parts.items():
            archive.writestr(name, data)
        archive.writestr("word/settings.xml", _PANDOC_SETTINGS)
    listed = f"{tmp_path / 'report.html'}\n{docx}\n"
    monkeypatch.setenv("QUARTO_PROJECT_OUTPUT_FILES", listed)

    assert _MODULE.main([]) == 0
    with zipfile.ZipFile(docx) as archive:
        assert archive.namelist() == [*other_parts, "word/settings.xml"]
        assert {name: archive.read(name) for name in other_parts} == other_parts
        assert _math_font(archive.read("word/settings.xml").decode("utf-8")) == "Noto Sans Math"
    assert not _MODULE.restore(docx, _MODULE.template_math_properties())
    assert sorted(path.name for path in tmp_path.iterdir()) == ["report.docx"]
