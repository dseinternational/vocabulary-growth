# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The notes index lists every note, and lists nothing that is absent.

`notes/README.md` is the index of `notes/`: one table row per dated note,
`YYYYMMDDHHMM-slug.md`. Nothing enforced that, and ten notes had drifted out of
it by 2026-09-11 -- the oldest from 2026-08-23, three of them data-decision
records whose consequences reach the fits of record. An unindexed note is not a
cosmetic problem: the index is what a reader consults to find out whether a
question has already been settled, so a note missing from it is, in practice, a
note that was never written.

Both directions are checked. A row pointing at a file that no longer exists is
the same defect seen from the other side -- a consolidated note is deleted and
its row left behind, and the index then promises a record that is gone.

Deliberately not checked: whether a row's summary still describes its note. That
is a judgement, and a test that asserted it would either be trivial or wrong.
"""

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
NOTES = ROOT / "notes"
INDEX = NOTES / "README.md"

# A dated note. The index's own preamble states the naming rule; README.md and
# any undated document in the directory are not index entries.
NOTE_NAME = re.compile(r"^\d{12}-.+\.md$")

# A table row: "| [202609111429](202609111429-slug.md) | ... | ... |". The link
# text is the timestamp and the target is the file, so both are pinned here.
#
# The trailing letter is a real convention, not a typo: two notes written in the
# same minute are disambiguated in the link text alone -- 202608031500a and
# 202608031500b both target their own plain filename. The timestamp prefix is
# what has to agree with the file, so that is what the consistency check below
# compares.
INDEX_ROW = re.compile(r"^\| \[(\d{12})[a-z]?\]\((\d{12}-[^)]+\.md)\)")


def _note_files() -> set[str]:
    return {p.name for p in NOTES.glob("*.md") if NOTE_NAME.match(p.name)}


def _index_rows() -> list[re.Match[str]]:
    lines = INDEX.read_text(encoding="utf-8").splitlines()
    return [m for m in (INDEX_ROW.match(line) for line in lines) if m is not None]


def test_every_note_has_an_index_row():
    indexed = {m.group(2) for m in _index_rows()}
    missing = sorted(_note_files() - indexed)
    assert not missing, (
        "notes/ contains files with no row in notes/README.md: "
        + ", ".join(missing)
        + ". Add a row giving what the note is and its status -- see the table's existing entries."
    )


def test_every_index_row_points_at_a_note_that_exists():
    present = _note_files()
    dangling = sorted(m.group(2) for m in _index_rows() if m.group(2) not in present)
    assert not dangling, (
        "notes/README.md rows point at files that do not exist: "
        + ", ".join(dangling)
        + ". A consolidated note's row is removed with it; the supersession is recorded in the successor's row."
    )


def test_index_rows_are_unique_and_self_consistent():
    rows = _index_rows()

    mismatched = [f"[{m.group(1)}] -> {m.group(2)}" for m in rows if not m.group(2).startswith(m.group(1))]
    assert not mismatched, "index rows whose link text and target disagree: " + ", ".join(mismatched)

    seen: dict[str, int] = {}
    for m in rows:
        seen[m.group(2)] = seen.get(m.group(2), 0) + 1
    duplicated = sorted(name for name, count in seen.items() if count > 1)
    assert not duplicated, "notes indexed more than once: " + ", ".join(duplicated)


if __name__ == "__main__":
    # Runnable without pytest, so CI can check this on a documentation-only
    # change. `notes/*` classifies as documentation in the `changes` job, which
    # skips the test jobs -- exactly the change that adds a note and forgets its
    # row. The always-on job runs this file directly instead.
    failures = 0
    for name, case in sorted(globals().items()):
        if not name.startswith("test_") or not callable(case):
            continue
        try:
            case()
        except AssertionError as error:
            failures += 1
            print(f"FAIL {name}: {error}")
    if failures:
        raise SystemExit(1)
    print(f"notes index: {len(_note_files())} notes, all indexed, no dangling rows")
