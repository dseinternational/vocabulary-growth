# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check hand-written parallel runbook lists against the model registry.

Explicit model lists bypass the replication driver's default registry lookup.
Parse the published commands and prose so an omitted model or stale scope
count produces a failing check.
"""

import re
from pathlib import Path

import pytest

from vocab_growth.models.definitions import MODEL_REGISTRY

RUNBOOK = Path(__file__).resolve().parents[1] / "docs" / "runbooks" / "full-refit.md"

#: The heading whose bullets carry the split. Anchored so an unrelated
#: `-Models` example elsewhere in the runbook cannot be picked up by mistake.
SECTION = "### Parallel fitting"


def _parallel_section() -> str:
    text = RUNBOOK.read_text(encoding="utf-8")
    start = text.find(SECTION)
    assert start != -1, f"{RUNBOOK.name} no longer contains {SECTION!r}."
    end = text.find("\n### ", start + len(SECTION))
    return text[start : end if end != -1 else len(text)]


def _model_lists() -> list[list[str]]:
    """Every `-Models a,b,c` list in the parallel-fitting section, in order."""
    return [
        [key.strip() for key in match.group(1).split(",") if key.strip()]
        for match in re.finditer(r"-Models\s+([a-z0-9,]+)", _parallel_section())
    ]


def test_the_parallel_passes_cover_every_registered_model():
    lists = _model_lists()
    assert len(lists) >= 2, (
        "Expected at least two `-Models` commands in the parallel-fitting "
        f"section (a DS pool and a serial TD pass); found {len(lists)}."
    )
    covered = {key for group in lists for key in group}
    missing = sorted(set(MODEL_REGISTRY) - covered)
    assert not missing, (
        f"{RUNBOOK.name}'s parallel-fitting commands omit {missing}. A model in "
        "neither list is never queued and never reported as absent — the run "
        "reports SUCCESS having fitted a subset. Add each to the pass that "
        "matches its population, then re-read the serial-pass caveat: the "
        "full-data TD fits must not share the box."
    )


def test_the_parallel_passes_name_only_registered_models():
    covered = {key for group in _model_lists() for key in group}
    unknown = sorted(covered - set(MODEL_REGISTRY))
    assert not unknown, (
        f"{RUNBOOK.name}'s parallel-fitting commands name {unknown}, which are "
        "not in MODEL_REGISTRY. `fit_model.py` rejects an unknown key, so the "
        "pass would abort at its first launch."
    )


def test_the_parallel_passes_do_not_fit_a_model_twice():
    lists = _model_lists()
    seen: dict[str, int] = {}
    for index, group in enumerate(lists):
        for key in group:
            if key in seen:
                pytest.fail(
                    f"{key} appears in both parallel pass {seen[key]} and pass "
                    f"{index}. The second launch re-fits it or, worse, races the "
                    "first for its output directory."
                )
            seen[key] = index


def test_the_prose_headline_lists_match_their_commands():
    """Each bullet names its models in backticks before giving the command.

    The prose list is what a reader skims; the command is what they paste. They
    drifting apart is how VG22 came to be described as a DS model in one place
    and absent from the pool in the other.
    """
    section = _parallel_section()
    for bullet in re.finditer(
        r"^- \*\*(?:DS|TD) models\*\* \(`([^`]+)`\)", section, re.MULTILINE
    ):
        prose = set(bullet.group(1).split())
        command = next(
            (set(group) for group in _model_lists() if set(group) & prose), set()
        )
        assert prose == command, (
            f"The prose list {sorted(prose)} and its `-Models` command "
            f"{sorted(command)} disagree in {RUNBOOK.name}."
        )


def _scope_table_counts() -> dict[str, int]:
    """The `N today` figures from the runbook's `-Scope` table, by scope name."""
    text = RUNBOOK.read_text(encoding="utf-8")
    return {
        match.group(1): int(match.group(2))
        for match in re.finditer(
            r"^\|\s*`(publication|all)`\s*\|.*?(\d+)\s+today", text, re.MULTILINE
        )
    }


def test_the_scope_table_counts_match_the_catalogue():
    """Compare documented scope counts with the catalogue's publication and full sets."""
    from vocab_growth.models.catalogue import CATALOGUE

    roles = {
        key: str(getattr(entry.role, "value", entry.role))
        for key, entry in CATALOGUE.items()
    }
    expected = {
        "publication": sum(
            1
            for role in roles.values()
            if role in {"model-of-record", "td-reference", "unclassified"}
        ),
        "all": len(MODEL_REGISTRY),
    }
    found = _scope_table_counts()
    assert set(found) == {"publication", "all"}, (
        f"Could not parse both scope rows out of {RUNBOOK.name}; got {found}."
    )
    assert found == expected, (
        f"The `-Scope` table in {RUNBOOK.name} says {found} but the catalogue "
        f"gives {expected}. Update the table, not this test."
    )
