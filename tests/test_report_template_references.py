# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve static package references in report-template Python cells.

Check package imports and attributes read from imported modules without fitting
or rendering. Dynamically constructed names are outside this check.
"""

from __future__ import annotations

import ast
import importlib
import re
import types
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
DOCS = REPO / "docs"

_CELL = re.compile(r"```\{python\}\n(.*?)```", flags=re.S)


def _templates() -> list[Path]:
    return sorted(DOCS.rglob("*.qmd"))


def _cells(path: Path):
    """``(first_line, source)`` for each Python cell, lines counted in the file."""
    text = path.read_text(encoding="utf-8")
    for match in _CELL.finditer(text):
        yield text.count("\n", 0, match.start(1)) + 1, match.group(1)


def _resolve(dotted: str, name: str):
    """``name`` read off module ``dotted``, as a module attribute or a submodule."""
    module = importlib.import_module(dotted)
    if hasattr(module, name):
        return getattr(module, name)
    return importlib.import_module(f"{dotted}.{name}")


def broken_references(path: Path) -> list[str]:
    """Every static `vocab_growth` reference in ``path`` that does not resolve."""
    problems: list[str] = []
    for first_line, source in _cells(path):
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            problems.append(f"line {first_line + (exc.lineno or 1) - 1}: cell does not parse")
            continue

        def where(node, first_line=first_line):
            return f"line {first_line + node.lineno - 1}"

        modules: dict[str, types.ModuleType] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("vocab_growth"):
                for alias in node.names:
                    try:
                        value = _resolve(node.module, alias.name)
                    except Exception:
                        problems.append(
                            f"{where(node)}: from {node.module} import {alias.name} -- no such name"
                        )
                        continue
                    if isinstance(value, types.ModuleType):
                        modules[alias.asname or alias.name] = value
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith("vocab_growth"):
                        continue
                    try:
                        module = importlib.import_module(alias.name)
                    except Exception:
                        problems.append(f"{where(node)}: import {alias.name} -- no such module")
                        continue
                    # `import a.b` binds `a`; only an aliased import binds the leaf.
                    if alias.asname:
                        modules[alias.asname] = module

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in modules
                and not hasattr(modules[node.value.id], node.attr)
            ):
                problems.append(f"{where(node)}: {node.value.id}.{node.attr} -- no such attribute")
    return sorted(set(problems))


def test_there_are_templates_to_check():
    """A glob that matched nothing would pass every template vacuously."""
    assert len(_templates()) >= 22


@pytest.mark.parametrize(
    "template", _templates(), ids=lambda path: str(path.relative_to(DOCS))
)
def test_every_package_name_a_template_uses_exists(template):
    problems = broken_references(template)
    assert not problems, (
        f"{template.relative_to(REPO)} refers to names the package no longer has:\n  "
        + "\n  ".join(problems)
    )


def test_the_check_sees_the_defect_it_was_written_for(tmp_path, monkeypatch):
    """Detect a renamed loader reached through an aliased module import."""
    stale = tmp_path / "stale.qmd"
    stale.write_text(
        "```{python}\n"
        "from vocab_growth.models import common_joint_modality as _cjm\n"
        "_four, _ = _cjm._load_uk02_four_cell()\n"
        "```\n"
        "\n"
        "```{python}\n"
        "from vocab_growth.cross_tab_sources import load_uk02_four_cell, not_a_loader\n"
        "```\n",
        encoding="utf-8",
    )
    problems = broken_references(stale)
    assert "line 3: _cjm._load_uk02_four_cell -- no such attribute" in problems
    assert (
        "line 7: from vocab_growth.cross_tab_sources import not_a_loader -- no such name"
        in problems
    )
    # The real loader on the same line is not reported.
    assert not any("import load_uk02_four_cell" in problem for problem in problems)
