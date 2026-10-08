# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate the contributing fits and data of comparison outputs.

A replaced fit must invalidate the recorded comparison. Outputs without
recorded provenance must also be reported rather than silently accepted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vocab_growth.comparisons_provenance import (
    COMPARISON_MANIFEST_FILENAME,
    validate_comparison_manifest,
    write_comparison_manifest,
)
from vocab_growth.fit_artifacts import FIT_MANIFEST_FILENAME


def _write_fit(models_dir: Path, label: str, *, created: str) -> Path:
    output_dir = models_dir / label
    output_dir.mkdir(parents=True)
    (output_dir / FIT_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "created_at_utc": created,
                "data": {
                    "analysis_frame_hash": "sha256:frame",
                    "source_data_hash": "sha256:data",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_dir


@pytest.fixture
def dirs(tmp_path):
    models_dir = tmp_path / "models"
    comparisons_dir = tmp_path / "comparisons"
    comparisons_dir.mkdir(parents=True)
    return models_dir, comparisons_dir


def test_a_recorded_comparison_validates_against_its_own_fits(dirs):
    models_dir, comparisons_dir = dirs
    fit = _write_fit(models_dir, "VG10-x", created="2026-08-01T00:00:00Z")
    (comparisons_dir / "overlay.png").touch()

    write_comparison_manifest(
        str(comparisons_dir),
        script="compare_models.py",
        contributing={"VG10-x": str(fit)},
        outputs=["overlay.png"],
    )

    errors, warnings = validate_comparison_manifest(
        str(comparisons_dir), str(models_dir)
    )
    assert errors == []
    assert warnings == []


@pytest.mark.parametrize("change, message", [
    ("output", "changed after generation"),
    ("code", "code is missing or changed"),
    ("hash", "no recorded hash"),
    ("source", "no contributing fit or data source"),
    ("unclaimed", "not claimed"),
])
def test_strict_publication_refuses_stale_or_unproven_outputs(dirs, change, message):
    models, comparisons = dirs
    output = comparisons / "summary.csv"
    output.write_text("x\n1\n", encoding="utf-8")
    write_comparison_manifest(str(comparisons), script="summary.py", contributing={},
                              outputs=[output.name], source_data_hash="current")
    manifest_path = comparisons / COMPARISON_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["scripts"]["summary.py"]
    if change == "output":
        output.write_text("x\n2\n", encoding="utf-8")
    elif change == "code":
        entry.pop("implementation")
    elif change == "hash":
        entry.pop("output_hashes")
    elif change == "source":
        entry.pop("source_data_hash")
    else:
        (comparisons / "unrecorded.csv").touch()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    errors, _ = validate_comparison_manifest(str(comparisons), str(models),
                    current_source_data_hash="current", require_publication=True)
    assert any(message in error for error in errors), errors


def test_publication_checks_contributor_quality_even_when_fingerprint_matches(dirs, monkeypatch):
    import vocab_growth.comparisons_provenance as cp

    models, comparisons = dirs
    fit = _write_fit(models, "VG10-x", created="2026-09-23")
    (comparisons / "summary.csv").touch()
    write_comparison_manifest(str(comparisons), script="summary.py",
                              contributing={fit.name: str(fit)}, outputs=["summary.csv"])
    calls = []

    def quality(path, raw_hash, **kwargs):
        calls.append((path, raw_hash))
        return ["failed hard convergence"]

    monkeypatch.setattr(cp, "_publication_fit_errors", quality)
    errors, _ = validate_comparison_manifest(str(comparisons), str(models),
                    current_source_data_hash="current", require_publication=True)
    assert calls == [(str(fit), "current")]
    assert "failed hard convergence" in errors


def test_a_refitted_contributor_invalidates_the_comparison(dirs):
    """The defect: a comparison outliving the fit it was computed from."""
    models_dir, comparisons_dir = dirs
    fit = _write_fit(models_dir, "VG10-x", created="2026-08-01T00:00:00Z")
    (comparisons_dir / "overlay.png").touch()
    write_comparison_manifest(
        str(comparisons_dir),
        script="compare_models.py",
        contributing={"VG10-x": str(fit)},
        outputs=["overlay.png"],
    )

    # The model is refitted; the comparison on disk is now stale.
    (fit / FIT_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "created_at_utc": "2026-08-26T00:00:00Z",
                "data": {
                    "analysis_frame_hash": "sha256:frame-2",
                    "source_data_hash": "sha256:data",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    errors, _ = validate_comparison_manifest(str(comparisons_dir), str(models_dir))
    assert len(errors) == 1
    assert "was refitted after this comparison was generated" in errors[0]


def test_a_vanished_contributor_is_an_error(dirs):
    models_dir, comparisons_dir = dirs
    fit = _write_fit(models_dir, "VG10-x", created="2026-08-01T00:00:00Z")
    (comparisons_dir / "overlay.png").touch()
    write_comparison_manifest(
        str(comparisons_dir),
        script="compare_models.py",
        contributing={"VG10-x": str(fit)},
        outputs=["overlay.png"],
    )
    (fit / FIT_MANIFEST_FILENAME).unlink()

    errors, _ = validate_comparison_manifest(str(comparisons_dir), str(models_dir))
    assert len(errors) == 1
    assert FIT_MANIFEST_FILENAME in errors[0]


def test_an_unclaimed_comparison_file_is_reported_not_ignored(dirs):
    """Warn about unclaimed outputs during ordinary validation.

    Strict publication checks reject them, as tested separately above.
    """
    models_dir, comparisons_dir = dirs
    fit = _write_fit(models_dir, "VG10-x", created="2026-08-01T00:00:00Z")
    (comparisons_dir / "overlay.png").touch()
    (comparisons_dir / "unrecorded.csv").touch()
    write_comparison_manifest(
        str(comparisons_dir),
        script="compare_models.py",
        contributing={"VG10-x": str(fit)},
        outputs=["overlay.png"],
    )

    errors, warnings = validate_comparison_manifest(
        str(comparisons_dir), str(models_dir)
    )
    assert errors == []
    assert len(warnings) == 1
    assert "unrecorded.csv" in warnings[0]


def test_a_missing_manifest_is_an_error_naming_the_remedy(dirs):
    models_dir, comparisons_dir = dirs
    _write_fit(models_dir, "VG10-x", created="2026-08-01T00:00:00Z")
    (comparisons_dir / "overlay.png").touch()

    errors, _ = validate_comparison_manifest(str(comparisons_dir), str(models_dir))
    assert len(errors) == 1
    assert COMPARISON_MANIFEST_FILENAME in errors[0]
    assert "regenerate" in errors[0]


def test_a_source_file_is_recorded_and_a_change_to_it_is_an_error(dirs, tmp_path):
    """Fingerprint raw inputs when a comparison reads data rather than a fit."""
    models_dir, comparisons_dir = dirs
    models_dir.mkdir()
    source = tmp_path / "data" / "vocab_data_es_01.csv"
    source.parent.mkdir()
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    (comparisons_dir / "matched_design_bands.csv").touch()

    write_comparison_manifest(
        str(comparisons_dir),
        script="compare_matched_designs.py",
        contributing={},
        outputs=["matched_design_bands.csv"],
        source_files={"es_01": str(source)},
        source_root=str(tmp_path),
    )
    recorded = json.loads(
        (comparisons_dir / COMPARISON_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    entry = recorded["scripts"]["compare_matched_designs.py"]
    assert entry["contributing_fits"] == {}
    assert entry["source_files"]["es_01"]["path"] == "data/vocab_data_es_01.csv"
    assert entry["source_files"]["es_01"]["sha256"].startswith("sha256:")

    errors, warnings = validate_comparison_manifest(
        str(comparisons_dir), str(models_dir), source_root=str(tmp_path)
    )
    assert errors == []
    assert warnings == []

    source.write_text("a,b\n1,3\n", encoding="utf-8")
    errors, _ = validate_comparison_manifest(
        str(comparisons_dir), str(models_dir), source_root=str(tmp_path)
    )
    assert len(errors) == 1
    assert "source file es_01" in errors[0]
    assert "changed after this comparison was generated" in errors[0]

    source.unlink()
    errors, _ = validate_comparison_manifest(
        str(comparisons_dir), str(models_dir), source_root=str(tmp_path)
    )
    assert len(errors) == 1
    assert "is missing" in errors[0]


def test_several_scripts_merge_into_one_manifest(dirs):
    """Each comparison script records its own entry without clobbering others."""
    models_dir, comparisons_dir = dirs
    fit_a = _write_fit(models_dir, "VG10-x", created="2026-08-01T00:00:00Z")
    fit_b = _write_fit(models_dir, "VG13-y", created="2026-08-02T00:00:00Z")
    (comparisons_dir / "a.png").touch()
    (comparisons_dir / "b.png").touch()

    write_comparison_manifest(
        str(comparisons_dir),
        script="compare_models.py",
        contributing={"VG10-x": str(fit_a)},
        outputs=["a.png"],
    )
    write_comparison_manifest(
        str(comparisons_dir),
        script="compare_ds_td.py",
        contributing={"VG13-y": str(fit_b)},
        outputs=["b.png"],
    )

    payload = json.loads(
        (comparisons_dir / COMPARISON_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert set(payload["scripts"]) == {"compare_models.py", "compare_ds_td.py"}

    errors, warnings = validate_comparison_manifest(
        str(comparisons_dir), str(models_dir)
    )
    assert errors == []
    assert warnings == []


# --- Pool-wide provenance, the snapshot, and the invocation record ------------


def test_a_pool_derived_comparison_records_the_raw_data_hash(dirs):
    """Record a pool-wide data hash when no stored reporting fit contributes."""
    models_dir, comparisons_dir = dirs
    models_dir.mkdir(parents=True)
    (comparisons_dir / "pool_descriptives.csv").write_text("a,b\n1,2\n")
    write_comparison_manifest(
        str(comparisons_dir),
        script="pool_descriptives.py",
        contributing={},
        outputs=["pool_descriptives.csv"],
        source_data_hash="sha256:pool",
    )

    errors, warnings = validate_comparison_manifest(
        str(comparisons_dir), str(models_dir), current_source_data_hash="sha256:pool"
    )
    assert errors == [] and warnings == []

    errors, _ = validate_comparison_manifest(
        str(comparisons_dir), str(models_dir), current_source_data_hash="sha256:moved"
    )
    assert len(errors) == 1
    assert "raw data changed" in errors[0]
    assert "regenerate" in errors[0]


def test_a_pool_hash_is_not_checked_when_the_caller_supplies_none(dirs):
    """Skip the raw-data hash comparison when the caller supplies no hash."""
    models_dir, comparisons_dir = dirs
    models_dir.mkdir(parents=True)
    (comparisons_dir / "pool_descriptives.csv").write_text("a\n1\n")
    write_comparison_manifest(
        str(comparisons_dir),
        script="pool_descriptives.py",
        contributing={},
        outputs=["pool_descriptives.csv"],
        source_data_hash="sha256:pool",
    )
    errors, _ = validate_comparison_manifest(str(comparisons_dir), str(models_dir))
    assert errors == []


def test_the_invocation_is_recorded_so_a_partial_run_is_legible(dirs):
    models_dir, comparisons_dir = dirs
    models_dir.mkdir(parents=True)
    (comparisons_dir / "ds_td_spoken_re_dispersion.csv").write_text("a\n1\n")
    write_comparison_manifest(
        str(comparisons_dir),
        script="compare_ds_td_re.py (spoken)",
        contributing={},
        outputs=["ds_td_spoken_re_dispersion.csv"],
        arguments=["spoken"],
    )
    payload = json.loads(
        (comparisons_dir / COMPARISON_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["scripts"]["compare_ds_td_re.py (spoken)"]["arguments"] == [
        "spoken"
    ]


def test_the_output_snapshot_names_what_the_run_actually_wrote(dirs):
    """The claim is derived from the run, not from a hand-maintained list."""
    from vocab_growth.comparisons_provenance import ComparisonOutputs

    _, comparisons_dir = dirs
    (comparisons_dir / "already_there.csv").write_text("a\n1\n")

    written = ComparisonOutputs(str(comparisons_dir))
    (comparisons_dir / "new_output.csv").write_text("b\n2\n")
    (comparisons_dir / "already_there.csv").write_text("a\n1\n2\n")  # rewritten

    assert written.written() == ["already_there.csv", "new_output.csv"]


def test_the_snapshot_ignores_the_manifest_and_the_nested_pipelines(dirs):
    from vocab_growth.comparisons_provenance import ComparisonOutputs

    _, comparisons_dir = dirs
    (comparisons_dir / "sensitivity").mkdir()
    written = ComparisonOutputs(str(comparisons_dir))
    (comparisons_dir / COMPARISON_MANIFEST_FILENAME).write_text("{}")
    (comparisons_dir / "sensitivity" / "robustness_matrix_vg10.csv").write_text("a\n")
    assert written.written() == []


# --------------------------------------------------------------------------
# Issue #362, guard (b): the generating checkout
# --------------------------------------------------------------------------


def _record(dirs, monkeypatch, *, dirty):
    import vocab_growth.comparisons_provenance as provenance

    models_dir, comparisons_dir = dirs
    fit = _write_fit(models_dir, "VG10-x", created="2026-08-01T00:00:00Z")
    (comparisons_dir / "overlay.png").touch()
    monkeypatch.setattr(
        provenance, "git_metadata", lambda root: {"commit": "abc", "branch": "main", "detached": False, "dirty": dirty}
    )
    write_comparison_manifest(
        str(comparisons_dir), script="compare_models.py",
        contributing={"VG10-x": str(fit)}, outputs=["overlay.png"],
    )
    return models_dir, comparisons_dir


def _entry(comparisons_dir: Path) -> dict:
    payload = json.loads((comparisons_dir / COMPARISON_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    return payload["scripts"]["compare_models.py"]


def test_each_entry_records_the_generating_checkout(dirs, monkeypatch):
    _, comparisons_dir = _record(dirs, monkeypatch, dirty=False)
    assert _entry(comparisons_dir)["code"] == {"commit": "abc", "branch": "main", "detached": False, "dirty": False}


def test_a_strict_sync_accepts_a_clean_entry_under_the_current_code(dirs, monkeypatch):
    models_dir, comparisons_dir = _record(dirs, monkeypatch, dirty=False)
    errors, _ = validate_comparison_manifest(str(comparisons_dir), str(models_dir), require_current_code=True)
    assert errors == []


def test_a_strict_sync_refuses_an_entry_from_a_dirty_checkout(dirs, monkeypatch):
    models_dir, comparisons_dir = _record(dirs, monkeypatch, dirty=True)
    errors, _ = validate_comparison_manifest(str(comparisons_dir), str(models_dir), require_current_code=True)
    assert any("uncommitted changes" in error for error in errors), errors
    # A provisional sync does not ask, as it does not for the fits.
    errors, _ = validate_comparison_manifest(str(comparisons_dir), str(models_dir))
    assert errors == []


def test_a_strict_sync_refuses_an_entry_with_no_recorded_checkout(dirs, monkeypatch):
    models_dir, comparisons_dir = _record(dirs, monkeypatch, dirty=False)
    path = comparisons_dir / COMPARISON_MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["scripts"]["compare_models.py"]["code"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    errors, _ = validate_comparison_manifest(str(comparisons_dir), str(models_dir), require_current_code=True)
    assert any("unrecorded provenance" in error for error in errors), errors


def test_a_strict_sync_refuses_an_entry_generated_under_other_code(dirs, monkeypatch):
    """A comparison regenerated by a development checkout with moved code."""
    models_dir, comparisons_dir = _record(dirs, monkeypatch, dirty=False)
    path = comparisons_dir / COMPARISON_MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scripts"]["compare_models.py"]["implementation"]["package"]["sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    errors, _ = validate_comparison_manifest(str(comparisons_dir), str(models_dir), require_current_code=True)
    assert any("comparison code is missing or changed" in error for error in errors), errors
