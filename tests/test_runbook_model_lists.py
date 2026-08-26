# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The refit runbook's parallel-pass model lists must cover the registry.

`run_replication.ps1` derives its model list from `MODEL_REGISTRY` when
`-Models` is omitted, so the default sequential path cannot go stale. The
runbook's *parallel* recipe does not use that path: it splits the registry by
hand into a Down syndrome pool and a serial typically-developing pass, and
passes each as an explicit `-Models` list. A model missing from both is never
queued, and nothing downstream notices — `validate_models` checks only the
models the run was given, so the run ends `SUCCESS` having fitted a subset.

That is exactly what happened between VG21/VG22/VG23 being registered and the
2026-08 VM run: all three were absent from both lists. The same class of defect
had already been found twice — the three agent-instruction files listing the
old model set, and the sensitivity suite's hand-maintained base-model map that
failed as a bare `KeyError: 'vg16'` — so this is the third time a hand-copied
model list has drifted from the registry, and the first time it is checked.

The lists are parsed out of the runbook rather than restated here. Restating
them would give two copies to keep in sync and a test that passes while the
document a human actually reads is wrong.
"""

import re
from pathlib import Path

import pytest

from vocab_growth.models.definitions import MODEL_REGISTRY

RUNBOOK = Path(__file__).resolve().parents[1] / "docs" / "runbooks" / "full-refit.md"

#: The heading whose bullets carry the split. Anchored so an unrelated
#: `-Models` example elsewhere in the runbook cannot be picked up by mistake.
SECTION = "### Parallel fitting on a large VM"


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
