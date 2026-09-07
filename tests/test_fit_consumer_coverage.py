# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Every script that opens a trace either validates it or is a recorded exemption.

Issue #266 finding 1 asked for the exact prepared-frame hash to be compared "for
every fit consumer", and the gap that stayed open for a month was not the
mechanism -- it was coverage. Eleven of the sixteen scripts that open a trace did
not use it, and nothing said so; the eleven were found by reading the tree, which
is exactly the check that does not run again when the seventeenth script is
added.

So the invariant is pinned here rather than swept once. A new trace-reading
script fails this test until it either validates or is added to
``EXEMPT_CONSUMERS`` with a reason -- and adding it there is a visible, arguable
edit rather than an absence.
"""

from __future__ import annotations

from pathlib import Path

from vocab_growth.fit_consumers import EXEMPT_CONSUMERS

#: Scope is ``scripts/*.py``: the supported entry points. ``scripts/experiments/``
#: is deliberately excluded and says so in its own README -- those are dated
#: records of how one number was obtained, several written against a fit that has
#: since been superseded, which is what makes them records. The two cited as live
#: evidence validate anyway (issue #266 finding 7 found them doing the opposite).
SCRIPTS = Path(__file__).parents[1] / "scripts"

#: How a script may discharge the check, and what proves it does.
#: ``fit_consumers`` is the shared helper; the rest predate it and validate
#: directly, each for a documented reason of its own.
VALIDATION_MARKERS = (
    "fit_consumers",  # the shared consumer helper
    "validate_fit_output",  # fit_model.py, loso_compare.py, resume_from_trace.py
    "require_valid_fit",  # regenerate_plots.py
    "_verified_frame",  # compare_ds_td_re.py, through report_cells
)

TRACE_MARKERS = ("from_netcdf", "trace.nc")


def _trace_reading_scripts() -> dict[str, str]:
    """Top-level scripts that open a stored trace, mapped to their source."""
    found = {}
    for path in sorted(SCRIPTS.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if any(marker in source for marker in TRACE_MARKERS):
            found[path.name] = source
    return found


def test_every_trace_reader_validates_or_is_exempt():
    unguarded = []
    for name, source in _trace_reading_scripts().items():
        if name in EXEMPT_CONSUMERS:
            continue
        if not any(marker in source for marker in VALIDATION_MARKERS):
            unguarded.append(name)
    assert not unguarded, (
        "These scripts open a fitted trace without checking it against the "
        "registered definition and the current prepared frame: "
        f"{', '.join(sorted(unguarded))}. Use "
        "vocab_growth.fit_consumers.require_current_fit (or "
        "load_validated_trace), or record an exemption with its reason in "
        "fit_consumers.EXEMPT_CONSUMERS."
    )


def test_the_exemptions_name_scripts_that_exist_and_read_a_trace():
    """An exemption for a script that no longer reads a trace is dead weight."""
    readers = _trace_reading_scripts()
    for name in EXEMPT_CONSUMERS:
        assert (SCRIPTS / name).is_file(), f"{name} is exempt but does not exist"
        assert name in readers, (
            f"{name} is recorded as an exempt trace consumer but no longer "
            "opens a trace; drop the exemption."
        )


def test_the_sixteen_are_still_sixteen():
    """A count, so a new consumer is noticed even if it happens to validate.

    Sixteen top-level scripts opened a trace when finding 1's coverage was
    enumerated on 2026-09-06. The number is not sacred -- but it moving is worth
    a moment's thought about whether the new script belongs on this list, so
    changing it here should be deliberate.
    """
    assert len(_trace_reading_scripts()) == 16
