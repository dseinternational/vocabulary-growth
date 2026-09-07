# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The consumer-side half of issue #266 finding 1.

The finding asked for the exact prepared-frame hash to be compared "for every
fit consumer". The pipeline and the publication path did; the scripts that open
a stored trace and print a number off it did not, and a loader-rule change is
invisible to the raw-CSV fingerprint they carried, because the rules run in
Python after the CSVs are read.

These check the shared helper rather than each script: the wiring is pinned
separately in :mod:`tests.test_fit_consumer_coverage`, which is the test that
notices a *new* trace-reading script arriving unvalidated.
"""

from __future__ import annotations

import argparse
import json

import pytest

from vocab_growth import fit_consumers
from vocab_growth.fit_artifacts import FitValidationError
from vocab_growth.models.definitions import MODEL_REGISTRY


@pytest.fixture(autouse=True)
def _clear_memo():
    """The helper memoises frame and source hashes; keep tests independent."""
    fit_consumers._frame_hashes.clear()
    fit_consumers._source_hashes.clear()
    yield
    fit_consumers._frame_hashes.clear()
    fit_consumers._source_hashes.clear()


def _stub_validation(monkeypatch, errors):
    """Make ``fit_errors`` report ``errors`` without touching data or a trace."""
    monkeypatch.setattr(
        fit_consumers, "validate_fit_output", lambda *a, **k: list(errors)
    )
    monkeypatch.setattr(fit_consumers, "_current_source_hash", lambda: "sha256:src")
    monkeypatch.setattr(
        fit_consumers, "_current_frame_hash", lambda key, definition: "sha256:frame"
    )


def _fit_dir(tmp_path, name="VG16-age-understood-spoken-ds-re-subj-uq-crosslag"):
    d = tmp_path / name
    d.mkdir()
    (d / "fit_manifest.json").write_text(json.dumps({"created_at_utc": "x"}))
    return str(d)


def test_a_clean_fit_is_read_without_complaint(tmp_path, monkeypatch):
    _stub_validation(monkeypatch, [])
    assert (
        fit_consumers.require_current_fit(
            "vg16", _fit_dir(tmp_path), consumer="test"
        )
        == []
    )


def test_a_stale_fit_is_refused_by_name(tmp_path, monkeypatch):
    _stub_validation(monkeypatch, ["analysis_frame_hash differs"])
    with pytest.raises(FitValidationError) as excinfo:
        fit_consumers.require_current_fit(
            "vg16", _fit_dir(tmp_path), consumer="time_to_milestone.py"
        )
    message = str(excinfo.value)
    # The refusal has to be actionable: which script, which model, what moved,
    # and the one way past it.
    assert "time_to_milestone.py" in message
    assert "vg16" in message
    assert "analysis_frame_hash differs" in message
    assert "--allow-stale-fit" in message


def test_an_override_is_announced_rather_than_silent(tmp_path, monkeypatch, capsys):
    _stub_validation(monkeypatch, ["analysis_frame_hash differs"])
    errors = fit_consumers.require_current_fit(
        "vg16", _fit_dir(tmp_path), consumer="test", allow_stale=True
    )
    assert errors == ["analysis_frame_hash differs"]
    printed = capsys.readouterr().out
    assert "stale fit accepted" in printed
    assert "analysis_frame_hash differs" in printed


def test_a_missing_fit_directory_is_an_error_not_a_crash(tmp_path, monkeypatch):
    _stub_validation(monkeypatch, [])
    errors = fit_consumers.fit_errors("vg16", str(tmp_path / "absent"))
    assert errors and "No fitted output" in errors[0]


def test_the_frame_hash_is_computed_once_per_model(monkeypatch):
    """Eighteen models must not mean eighteen frame rebuilds."""
    calls: list[str] = []

    def _count(key, definition):
        calls.append(key)
        return "sha256:frame"

    monkeypatch.setattr(fit_consumers, "expected_analysis_frame_hash", _count)
    definition = MODEL_REGISTRY["vg16"]
    for _ in range(4):
        fit_consumers._current_frame_hash("vg16", definition)
    assert calls == ["vg16"]


# --- Directory-addressed consumers -------------------------------------------


def test_a_model_of_records_directory_resolves_to_its_key():
    definition = MODEL_REGISTRY["vg22"]
    name = f"{definition.model_id}-{definition.config_name}"
    assert fit_consumers.model_key_for_dir(f"/somewhere/models/{name}") == "vg22"
    # Trailing separators must not defeat the match.
    assert fit_consumers.model_key_for_dir(f"/somewhere/models/{name}/") == "vg22"


def test_a_variant_directory_resolves_to_no_model():
    """A sensitivity or recovery fit is *supposed* to differ from the registry."""
    definition = MODEL_REGISTRY["vg22"]
    name = f"{definition.model_id}-{definition.config_name}"
    assert fit_consumers.model_key_for_dir(f"/m/{name}-rank2") is None
    assert fit_consumers.model_key_for_dir("/m/VG20-...-recovery-r01") is None


def test_an_unvalidated_directory_says_so(tmp_path, capsys):
    errors = fit_consumers.require_current_fit_dir(
        str(tmp_path / "VG22-something-rank2"), consumer="emit_factor_correlation.py"
    )
    assert errors == []
    printed = capsys.readouterr().out
    # "not checked" and "checked and clean" must never look the same in a log.
    assert "provenance not checked" in printed
    assert "emit_factor_correlation.py" in printed


def test_a_model_of_records_directory_is_checked(tmp_path, monkeypatch):
    _stub_validation(monkeypatch, ["definition differs: kappa_u"])
    with pytest.raises(FitValidationError):
        fit_consumers.require_current_fit_dir(
            _fit_dir(tmp_path), consumer="emit_factor_correlation.py"
        )


# --- The recorded exemptions --------------------------------------------------


def test_both_exemptions_carry_a_reason():
    """An exemption that is written down can be argued with; an absence cannot."""
    assert set(fit_consumers.EXEMPT_CONSUMERS) == {
        "fit_recovery.py",
        "compact_traces.py",
    }
    for script, reason in fit_consumers.EXEMPT_CONSUMERS.items():
        assert len(reason) > 80, f"{script}'s exemption is asserted, not argued"


def test_the_shared_override_flag_is_spelled_one_way():
    parser = argparse.ArgumentParser()
    fit_consumers.add_allow_stale_argument(parser)
    args = parser.parse_args(["--allow-stale-fit"])
    assert args.allow_stale_fit is True
    assert parser.parse_args([]).allow_stale_fit is False
