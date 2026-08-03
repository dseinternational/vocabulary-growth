# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guard that us_01 is read from exactly one source in ``scripts/prepare_data.py``.

**History.** This module was originally a blanket ban: us_01 (Edgin) was the one
study whose rows were *derived* rather than read from a per-study CSV, with
``vocab_combined_view_sql`` filtering ``wordbank_administration_data.csv`` to the
American English Down syndrome subset and minting subject identifiers as
``concat('id_', hex(hash(child_id)))``. An earlier ``data/vocab_data_us_01.csv``
(deleted in the commit that added this test; recoverable from ``7eafe97``) carried
identifiers in an unrelated scheme — ``ID_E079170FF03E6C1D``, uppercase prefix, a
different hash of a different key — which by construction could not collide with the
view's. Registering it *alongside* the view's Edgin block would therefore have escaped
every duplicate-identifier check while entering the same children twice as two
disjoint subject sets.

That original test said: "If us_01 ever genuinely needs a source CSV, the fix is to
remove the Edgin block from ``vocab_combined_view_sql`` first — not to relax this
test." That is what has since happened, for two reasons the by-child export cannot
address: it is age-truncated by Wordbank's own download page (345 Down syndrome
administrations reduced to 194), and it cannot separate the four source
administrations whose every word item is blank and which Wordbank scores as zeros.
``data/vocab_data_us_01.csv`` is now derived from the item-level contributor files by
``scripts/build_us01_source.py``, and the view's us_01 block reads *it* rather than
the export.

So the invariant is no longer "no source CSV" — it is **exactly one source**. The
hazard was never the CSV's existence; it was two relations feeding one study. This
module now checks that the CSV is registered, that the view reads it, and that the
view does not also read ``wordbank_child``. The export is still loaded for the
typically-developing pool, which is why the check is scoped to the view rather than
to the registry alone.

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


def _view_sql_without_comments() -> str:
    """The combined-view SQL with ``--`` comment text removed.

    The us_01 block documents its own history at length and names
    ``wordbank_child`` in prose, so a substring check against the raw SQL would
    read those comments as references. Only executable text may be inspected.
    """
    from vocab_growth.data_utils import vocab_combined_view_sql

    lines = []
    for line in vocab_combined_view_sql().split("\n"):
        head = line.split("--", 1)[0]
        if head.strip():
            lines.append(head)
    return "\n".join(lines)


def test_us_01_has_exactly_one_source():
    """us_01 must be read from one relation, never two.

    The hazard this guards is double-counting, not the existence of a source CSV:
    the ``vocab_us_01`` table and the ``wordbank_child`` export carry disjoint
    subject-identifier schemes, so if the combined view ever read *both* the same
    Edgin children would enter the DS pool twice as two unrelated subject sets --
    invisible to any duplicate-identifier check, and corrupting every DS estimate
    and every repeated-measures random effect.

    Which of the two is the source is a separate question, settled in favour of
    ``vocab_us_01`` (see this module's docstring). What must never hold is both.
    """
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
