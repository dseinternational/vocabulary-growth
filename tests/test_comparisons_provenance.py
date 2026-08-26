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
