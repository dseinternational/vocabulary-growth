# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Every comparison writer records its provenance, or is a recorded exemption.

Issue #266 finding 1 asked for comparison outputs to "record and validate all
contributing fit manifests". The mechanism landed in August and the sync
enforces it, but coverage was two scripts of thirteen: the other eleven wrote
figures and tables into the comparisons directory that ``sync_report_figures.py``
could only report as unclaimed.

Unclaimed is a warning rather than an error precisely so the ratchet could turn
one script at a time -- which means nothing fails when it stops turning. This
test is what stops it slipping back.
"""

from __future__ import annotations

from pathlib import Path

from vocab_growth.comparisons_provenance import MANIFEST_EXEMPT_SCRIPTS

SCRIPTS = Path(__file__).parents[1] / "scripts"

#: What marks a script as writing into the comparisons directory at all.
WRITER_MARKER = "comparisons_output_dir()"


def _comparison_writers() -> dict[str, str]:
    return {
        path.name: source
        for path in sorted(SCRIPTS.glob("*.py"))
        if WRITER_MARKER in (source := path.read_text(encoding="utf-8"))
    }


def test_every_comparison_writer_records_a_manifest_entry():
    unrecorded = [
        name
        for name, source in _comparison_writers().items()
        if name not in MANIFEST_EXEMPT_SCRIPTS
        and "write_comparison_manifest" not in source
    ]
    assert not unrecorded, (
        "These scripts write into the comparisons directory without recording "
        f"which fits they came from: {', '.join(sorted(unrecorded))}. Call "
        "vocab_growth.comparisons_provenance.write_comparison_manifest (with "
        "ComparisonOutputs to name what the run wrote), or record an exemption "
        "with its reason in MANIFEST_EXEMPT_SCRIPTS."
    )


def test_the_exemptions_name_scripts_that_exist_and_touch_the_directory():
    writers = _comparison_writers()
    for name in MANIFEST_EXEMPT_SCRIPTS:
        assert (SCRIPTS / name).is_file(), f"{name} is exempt but does not exist"
        assert name in writers, (
            f"{name} is recorded as an exempt comparison writer but no longer "
            "touches the comparisons directory; drop the exemption."
        )


def test_each_exemption_is_argued_rather_than_asserted():
    for script, reason in MANIFEST_EXEMPT_SCRIPTS.items():
        assert len(reason) > 80, f"{script}'s exemption is asserted, not argued"
