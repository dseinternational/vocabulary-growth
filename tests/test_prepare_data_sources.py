# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guard that us_01 is never loaded as a source CSV in ``scripts/prepare_data.py``.

us_01 (Edgin) is the one study whose rows are *derived* rather than read from a
per-study CSV: ``vocab_combined_view_sql`` filters ``wordbank_administration_data.csv``
to the American English Down syndrome subset of the Wordbank export and mints
subject identifiers as ``concat('id_', hex(hash(child_id)))``.

A ``data/vocab_data_us_01.csv`` once existed alongside the real per-study source
CSVs (deleted in the commit that added this test; recoverable from ``7eafe97``).
It carried its own pre-hashed identifiers in an unrelated scheme —
``ID_E079170FF03E6C1D``, uppercase prefix, a different hash of a different key —
which by construction cannot collide with the ``id_``-prefixed identifiers the
view mints. Adding such a file to ``_sources`` would therefore not be caught by
any duplicate-identifier check: the same Edgin children would enter the DS pool
twice as two disjoint sets of subjects, silently inflating every DS estimate and
corrupting the repeated-measures random effects.

The name ``vocab_data_us_01.csv`` matches every other study's source CSV exactly,
so this is a plausible mistake for someone extending the registry. This test is
the tripwire. If us_01 ever genuinely needs a source CSV, the fix is to remove
the Edgin block from ``vocab_combined_view_sql`` first — not to relax this test.

``scripts/prepare_data.py`` does its work at module scope (it reads the CSVs and
builds the DuckDB on import), so the registry is read statically via ``ast``
rather than by importing the module.
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
    # Guards the guard. A syntax error, a renamed registry or a non-dict value
    # already fails loudly inside _source_registry(); the risk covered here is
    # subtler. If _sources is ever refactored into a dynamically built mapping
    # (``_sources = {}`` followed by ``.update(...)``, or a seed entry plus a
    # loop), the literal still parses — but to a stub this static read cannot
    # follow. test_source_registry_has_no_us_01_entry would then pass against
    # that stub while the real registry went unchecked, so require a plausibly
    # complete registry of per-study CSV paths rather than merely a non-empty
    # one.
    registry = _source_registry()
    assert len(registry) > 1
    assert all(path.endswith(".csv") for path in registry.values())


def test_source_registry_has_no_us_01_entry():
    registry = _source_registry()
    offenders = {
        name: path
        for name, path in registry.items()
        if "us_01" in name.lower() or "us_01" in path.lower()
    }
    assert not offenders, (
        f"{_REGISTRY_NAME} in {_SCRIPT_PATH.name} gained a us_01 entry ({offenders}). "
        "us_01 rows are derived from wordbank_administration_data.csv by "
        "vocab_combined_view_sql, which mints 'id_'-prefixed subject identifiers. "
        "A us_01 source CSV carries a different, non-colliding identifier scheme, so "
        "loading one double-counts the Edgin children under two disjoint subject sets "
        "and corrupts every DS-pool estimate. See this module's docstring."
    )
