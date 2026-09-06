# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the fail-closed guards in ``scripts/resume_from_trace.py``.

``resume_from_trace.py`` writes a current manifest around an old posterior, so
its guards are the only thing standing between a retained trace and summaries
that describe data it was never fitted to. Until 2026-09-06 the data guard was
the **raw-CSV fingerprint alone** -- the defect
[#266](https://github.com/dseinternational/vocabulary-growth/issues/266)
finding 1 named for this script -- which is wrong in both directions:

* it cannot see a loader-rule change, because the masking and exclusion rules
  run in Python *after* the CSVs are read, so a rule change leaves the raw hash
  equal while the prepared frame drifts; and
* it fires on raw churn the model never reads, so a new Down syndrome study CSV
  would refuse a resume of a typically-developing fit whose frame is untouched.

The exact prepared-frame hash fixes both, combined with the fingerprint exactly
as ``fit_artifacts.validate_fit_output`` combines them. These tests pin that
combination, including the case a plain equality check gets wrong: a manifest
recording **no** frame hash must be refused rather than waved through.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "resume_from_trace.py"
_SPEC = importlib.util.spec_from_file_location("resume_from_trace_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

MODEL_KEY = "vg01"
CONFIG = "rep"
CURRENT_FRAME = "sha256:frame-as-the-loader-builds-it-today"
CURRENT_RAW = "sha256:raw-as-the-csvs-are-today"


@pytest.fixture
def definition():
    from vocab_growth.models.definitions import MODEL_REGISTRY

    return MODEL_REGISTRY[MODEL_KEY]


@pytest.fixture
def written_manifest(tmp_path, definition, monkeypatch):
    """Write a manifest that passes every guard, and stub the two current hashes.

    Both hashes are stubbed rather than computed: this file is about the guard's
    logic, and building a real frame would make every case a data-loading test.
    """
    from vocab_growth.fit_artifacts import normalise_for_json

    monkeypatch.setattr(
        _MODULE, "expected_analysis_frame_hash", lambda key, defn: CURRENT_FRAME
    )
    monkeypatch.setattr(_MODULE, "source_data_hash", lambda data_dir: CURRENT_RAW)

    def write(*, frame=CURRENT_FRAME, raw=CURRENT_RAW):
        import json

        payload = {
            "model": {"definition": normalise_for_json(definition)},
            "sampling": {"configuration_name": CONFIG},
            "data": {"source_data_hash": raw},
        }
        if frame is not None:
            payload["data"]["analysis_frame_hash"] = frame
        directory = tmp_path / "retained"
        directory.mkdir(exist_ok=True)
        (directory / _MODULE.FIT_MANIFEST_FILENAME).write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return str(directory)

    return write


def _verify(retained_dir, definition):
    return _MODULE._verify(retained_dir, MODEL_KEY, definition, CONFIG)


def test_a_matching_frame_and_fingerprint_verifies(written_manifest, definition):
    """The baseline: nothing has moved, so the resume proceeds."""
    manifest = _verify(written_manifest(), definition)
    assert manifest["data"]["analysis_frame_hash"] == CURRENT_FRAME


def test_a_drifted_frame_is_refused_although_the_raw_data_is_unchanged(
    written_manifest, definition
):
    """The defect this guard exists for: a loader-rule change the fingerprint cannot see.

    Masking and exclusion rules run after the CSVs are read, so this combination
    -- raw hash equal, frame hash different -- is exactly what a changed rule
    looks like, and is what the fingerprint-only guard admitted.
    """
    with pytest.raises(ValueError, match="prepared analysis frame differs"):
        _verify(written_manifest(frame="sha256:frame-under-the-old-rules"), definition)


def test_a_matching_frame_excuses_a_changed_fingerprint(written_manifest, definition):
    """Raw churn in CSVs this model never reads must not refuse the resume.

    The same excuse ``validate_fit_output`` applies, and the reason it is safe is
    the same: the fingerprint covers every CSV in ``data/`` while a model reads
    the raw data only through its own prepared frame, so a frame that still
    rebuilds identically vouches for the fit.
    """
    manifest = _verify(written_manifest(raw="sha256:a-new-study-csv-arrived"), definition)
    assert manifest["data"]["source_data_hash"] == "sha256:a-new-study-csv-arrived"


def test_both_hashes_moving_is_refused(written_manifest, definition):
    """With nothing left to vouch, the resume is refused rather than excused."""
    with pytest.raises(ValueError):
        _verify(
            written_manifest(frame="sha256:something-else", raw="sha256:also-else"),
            definition,
        )


def test_a_manifest_with_no_frame_hash_is_refused(written_manifest, definition):
    """A fit predating the frame hash is unverifiable, not assumed current.

    This is the case an equality check alone gets wrong in the dangerous
    direction: ``None != CURRENT_FRAME`` happens to refuse here, but the message
    has to say *why* -- there is nothing to compare -- because "refit" is the
    only remedy, where a drifted frame might instead be a rule change to revert.
    """
    with pytest.raises(ValueError, match="records no prepared-frame hash"):
        _verify(written_manifest(frame=None), definition)


def test_a_changed_definition_is_still_refused_first(written_manifest, definition):
    """The definition guard is unchanged by this work, and still fires."""
    import dataclasses

    other = dataclasses.replace(definition, config_name="not-the-registered-name")
    with pytest.raises(ValueError, match="model definition differs"):
        _MODULE._verify(written_manifest(), MODEL_KEY, other, CONFIG)


def test_a_changed_sampling_configuration_is_still_refused(written_manifest, definition):
    """Likewise the sampling-configuration guard."""
    with pytest.raises(ValueError, match="sampling configuration"):
        _MODULE._verify(written_manifest(), MODEL_KEY, definition, "dev")
