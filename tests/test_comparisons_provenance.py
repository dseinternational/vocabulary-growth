# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Provenance for cross-model comparison outputs (issue #266 finding 1).

Comparison figures and tables are derived from fitted output, but carried no
record of which fits they came from: ``sync_report_figures.py`` validated every
model directory it copied and then copied the comparisons directory wholesale,
so a comparison generated from a since-replaced fit synced as though it were
current. These tests pin the two properties that close that hole — a refitted
contributor is detected, and a comparison whose provenance was never recorded
is reported rather than passed over in silence.
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
    """Coverage is ratcheted: scripts adopt the manifest one at a time.

    An unrecorded file must be visible, but must not block the comparisons
    whose provenance *is* recorded — otherwise the first script to adopt the
    manifest breaks the sync for every other one.
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
    """A script with no contributing fit records its raw input instead (#289 4.9).

    `compare_matched_designs.py` reads one CSV and no posterior, so its four
    tables had no provenance at all and the sync warned on every one. The
    source file is fingerprinted like a fit manifest: a change stales the
    comparison, and a missing file is an error rather than a pass.
    """
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
    """Some comparisons have no contributing fit, and that must not read as none.

    ``pool_descriptives.py`` describes the data itself and ``kfold_loso.py``
    fits its own folds; neither reads a model of record. Recording nothing would
    make them indistinguishable from a script that was never wired up, which is
    the state finding 1 was about.
    """
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
    """``None`` is *not checked*, as everywhere else in this codebase."""
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
