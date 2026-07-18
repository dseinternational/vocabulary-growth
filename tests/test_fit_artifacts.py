# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for fit lifecycle, compatibility, and atomic publication."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd
import pytest

from vocab_growth import environment as env
from vocab_growth.fit_artifacts import (
    normalise_for_json,
    promote_staged_fit,
    validate_fit_output,
    write_fit_state,
)
from vocab_growth.models.common import run_fit_pipeline
from vocab_growth.models.definitions import VG01


def _write_manifest(
    output_dir: Path,
    *,
    sampling_name: str = "rep",
    dirty: bool = False,
) -> None:
    payload = {
        "model": {
            "model_id": VG01.model_id,
            "config_name": VG01.config_name,
            "definition": normalise_for_json(VG01),
        },
        "sampling": {
            "configuration_name": sampling_name,
            "parameters": asdict(sampling.get_sampling_configuration(sampling_name)),
        },
        "data": {"source_data_hash": "sha256:data"},
        "code": {"commit": "abc123", "dirty": dirty},
    }
    (output_dir / "fit_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_complete_output(output_dir: Path, *, sampling_name: str = "rep") -> None:
    output_dir.mkdir(parents=True)
    _write_manifest(output_dir, sampling_name=sampling_name)
    (output_dir / "trace.nc").touch()
    (output_dir / "index.html").touch()
    write_fit_state(
        str(output_dir),
        "complete",
        model_id=VG01.model_id,
        config_name=VG01.config_name,
        sampling_config_name=sampling_name,
    )


def test_publish_validation_accepts_complete_compatible_reporting_fit(tmp_path):
    output_dir = tmp_path / "fit"
    _write_complete_output(output_dir)

    errors = validate_fit_output(
        str(output_dir),
        expected_definition=VG01,
        expected_sampling_config_name="rep",
        expected_sampling_parameters=asdict(
            sampling.get_sampling_configuration("rep")
        ),
        expected_git={"commit": "abc123", "dirty": False},
        expected_source_data_hash="sha256:data",
        require_reporting_quality=True,
        require_rendered_report=True,
        require_clean_fit=True,
    )

    assert errors == []


def test_publish_validation_rejects_development_trace_even_when_complete(tmp_path):
    output_dir = tmp_path / "fit"
    _write_complete_output(output_dir, sampling_name="dev")

    errors = validate_fit_output(
        str(output_dir),
        require_reporting_quality=True,
        require_rendered_report=True,
        require_clean_fit=True,
    )

    assert any("not reporting-quality" in error for error in errors)


def test_trace_without_complete_state_is_not_resumable(tmp_path):
    output_dir = tmp_path / "fit"
    output_dir.mkdir()
    (output_dir / "trace.nc").touch()

    errors = validate_fit_output(str(output_dir))

    assert any("fit_state.json" in error for error in errors)
    assert any("fit_manifest.json" in error for error in errors)


def test_promote_staged_fit_replaces_canonical_only_at_completion(tmp_path):
    canonical = tmp_path / "models" / "VG01-test"
    staged = tmp_path / ".staging" / "run" / "models" / "VG01-test"
    canonical.mkdir(parents=True)
    staged.mkdir(parents=True)
    (canonical / "marker.txt").write_text("old", encoding="utf-8")
    (staged / "marker.txt").write_text("new", encoding="utf-8")

    promote_staged_fit(str(staged), str(canonical))

    assert (canonical / "marker.txt").read_text(encoding="utf-8") == "new"
    assert not staged.exists()


def test_failed_pipeline_preserves_previous_canonical_fit(tmp_path, monkeypatch):
    env.set_output_root(str(tmp_path))
    canonical = tmp_path / "models" / f"{VG01.model_id}-{VG01.config_name}"
    canonical.mkdir(parents=True)
    (canonical / "marker.txt").write_text("previous", encoding="utf-8")
    monkeypatch.setattr(
        "vocab_growth.models.common.env_info.report_environment_info", lambda: None
    )
    monkeypatch.setattr(
        "vocab_growth.models.common.package_metadata.report_package_versions",
        lambda packages: None,
    )
    monkeypatch.setattr("vocab_growth.models.common.run_banner", lambda *a, **k: None)

    def fail(_context):
        raise RuntimeError("deliberate failure")

    try:
        with pytest.raises(RuntimeError, match="deliberate failure"):
            run_fit_pipeline("dev", VG01, stages=[("Fail", fail)])
    finally:
        env.set_output_root(None)

    assert (canonical / "marker.txt").read_text(encoding="utf-8") == "previous"
    retained = list((tmp_path / "failed").glob("VG01-*-*"))
    assert len(retained) == 1
    state = json.loads((retained[0] / "fit_state.json").read_text())
    assert state["state"] == "failed"
    assert state["error"]["type"] == "RuntimeError"


def test_successful_pipeline_atomically_promotes_complete_fit(tmp_path, monkeypatch):
    env.set_output_root(str(tmp_path))
    canonical = tmp_path / "models" / f"{VG01.model_id}-{VG01.config_name}"
    canonical.mkdir(parents=True)
    (canonical / "marker.txt").write_text("previous", encoding="utf-8")
    monkeypatch.setattr(
        "vocab_growth.models.common.env_info.report_environment_info", lambda: None
    )
    monkeypatch.setattr(
        "vocab_growth.models.common.package_metadata.report_package_versions",
        lambda packages: None,
    )
    monkeypatch.setattr("vocab_growth.models.common.run_banner", lambda *a, **k: None)

    def prepare(context):
        frame = pd.DataFrame(
            {"study": ["toy", "toy"], "age": [24.0, 36.0], "spoken": [2, 12]}
        )
        context.set_model_data(
            model_data.BinomialModelData(
                X_obs=frame[["age"]].to_numpy(),
                y_obs=np.array([2, 12]),
                n_trials=810,
            ),
            frame,
        )
        Path(context.reporting.output_dir, "trace.nc").touch()

    try:
        context = run_fit_pipeline("dev", VG01, stages=[("Prepare", prepare)])
    finally:
        env.set_output_root(None)

    assert Path(context.reporting.output_dir) == canonical
    assert not (canonical / "marker.txt").exists()
    state = json.loads((canonical / "fit_state.json").read_text())
    assert state["state"] == "complete"
    assert (canonical / "fit_manifest.json").is_file()
