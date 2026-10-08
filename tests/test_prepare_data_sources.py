# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Require exactly one us_01 source in the combined Down syndrome view.

The item-level source CSV replaces the Wordbank by-child export for this study.
Reading both could count the same children twice under different identifier
schemes, which duplicate-identifier checks would not detect. Wordbank still
supplies the typically developing pool. Read the registry through AST because
importing the preparation script would rebuild the database.
"""

import ast
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "prepare_data.py"
_REGISTRY_NAME = "_sources"


def _source_registry() -> dict[str, str]:
    """Return the ``_sources`` mapping from ``prepare_data.py`` without importing it."""
    tree = ast.parse(_SCRIPT_PATH.read_text(encoding="utf-8"), filename=str(_SCRIPT_PATH))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if _REGISTRY_NAME not in targets:
            continue
        if not isinstance(node.value, ast.Dict):
            pytest.fail(f"{_REGISTRY_NAME} in {_SCRIPT_PATH.name} is no longer a dict literal")
        return {
            ast.literal_eval(key): ast.literal_eval(value)
            for key, value in zip(node.value.keys, node.value.values, strict=True)
        }
    pytest.fail(
        f"could not find the {_REGISTRY_NAME} dataset registry in {_SCRIPT_PATH.name}; "
        "if it was renamed, re-point this guard rather than deleting it"
    )


def test_source_registry_parses_to_all_study_csvs():
    # A dynamically extended literal could leave this static read incomplete.
    # Require a substantial registry of per-study CSV paths.

    registry = _source_registry()
    assert len(registry) > 1
    assert all(path.endswith(".csv") for path in registry.values())


def _view_sql_without_comments() -> str:
    """Strip SQL comments before checking executable source references."""
    from vocab_growth.data_utils import vocab_combined_view_sql

    lines = []
    for line in vocab_combined_view_sql().split("\n"):
        head = line.split("--", 1)[0]
        if head.strip():
            lines.append(head)
    return "\n".join(lines)


def test_us_01_has_exactly_one_source():
    """Read us_01 from its source table without also including its Wordbank rows."""
    sql = _view_sql_without_comments()
    registry = _source_registry()
    has_source_csv = any(
        "us_01" in name.lower() or "us_01" in path.lower()
        for name, path in registry.items()
    )

    assert has_source_csv, (
        f"{_REGISTRY_NAME} in {_SCRIPT_PATH.name} lost its us_01 entry. us_01 is read "
        "from data/vocab_data_us_01.csv, derived by scripts/build_us01_source.py. "
        "Without it the combined view's us_01 block has no relation to read. If the "
        "intent is to go back to the Wordbank by-child export, restore the Edgin "
        "block's dataset_name/health_conditions filter in vocab_combined_view_sql at "
        "the same time -- and note that the export is age-truncated and cannot "
        "separate the four all-blank administrations it scores as zeros."
    )
    assert "wordbank_child" not in sql, (
        "vocab_combined_view_sql reads wordbank_child while a us_01 source CSV is "
        "registered. That is the double-counting configuration this guard exists to "
        "prevent: the same Edgin children would enter the DS pool twice under two "
        "disjoint identifier schemes. See this module's docstring."
    )
    assert "vocab_us_01" in sql, (
        "a us_01 source CSV is registered but vocab_combined_view_sql does not read "
        "vocab_us_01, so the loaded table is silently unused."
    )
