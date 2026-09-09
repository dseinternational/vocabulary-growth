# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for atomic report-figure cache replacement."""

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import dse_research_utils.statistics.models.sampling as sampling
import pytest

from vocab_growth import environment as env
from vocab_growth.fit_artifacts import (
    FitValidationError,
    normalise_for_json,
    write_fit_state,
)
from vocab_growth.models.catalogue import (
    ModelRole,
    models_with_role,
    publication_models,
)
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.models.implementation_identity import implementation_signature

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "sync_report_figures.py"
_SPEC = importlib.util.spec_from_file_location("sync_report_figures_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _write_output(output_root, definition, *, state):
    output_dir = (
        output_root
        / "models"
        / f"{definition.model_id}-{definition.config_name}"
    )
    output_dir.mkdir(parents=True)
    manifest = {
        "model": {"definition": normalise_for_json(definition), "implementation": implementation_signature()},
        "sampling": {
            "configuration_name": "dev",
            "parameters": asdict(sampling.get_sampling_configuration("dev")),
        },
        "data": {},
        "code": {},
    }
    (output_dir / "fit_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (output_dir / "trace.nc").touch()
    write_fit_state(
        str(output_dir),
        state,
        model_id=definition.model_id,
        config_name=definition.config_name,
        sampling_config_name="dev",
    )


def test_sync_replaces_destination_and_removes_stale_files(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "cache" / "model"
    source.mkdir()
    destination.mkdir(parents=True)
    (source / "current.svg").write_text("current", encoding="utf-8")
    (source / "trace.nc").write_text("excluded", encoding="utf-8")
    (destination / "stale.svg").write_text("stale", encoding="utf-8")

    copied = _MODULE._sync_dir(str(source), str(destination))

    assert copied == 1
    assert sorted(path.name for path in destination.iterdir()) == ["current.svg"]


def test_sync_copy_failure_preserves_previous_destination(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "cache" / "model"
    source.mkdir()
    destination.mkdir(parents=True)
    (source / "current.svg").write_text("current", encoding="utf-8")
    (destination / "previous.svg").write_text("previous", encoding="utf-8")
    monkeypatch.setattr(
        _MODULE.shutil,
        "copy2",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        _MODULE._sync_dir(str(source), str(destination))

    assert (destination / "previous.svg").read_text(encoding="utf-8") == "previous"


def test_all_model_outputs_validate_before_any_cache_is_changed(tmp_path, monkeypatch):
    # Both fits belong to publication-required models, chosen by role rather
    # than by name: a development step that fails validation is skipped by
    # role instead of blocking (next test), and when VG02 became one on
    # 2026-09-09 this test silently turned into that one.
    valid, invalid = (MODEL_REGISTRY[key] for key in publication_models()[:2])
    output_root = tmp_path / "output"
    _write_output(output_root, valid, state="complete")
    _write_output(output_root, invalid, state="running")
    sync_calls = []
    monkeypatch.setattr(_MODULE, "_sync_dir", lambda *args: sync_calls.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sync_report_figures.py",
            "--output-dir",
            str(output_root),
            "--config",
            "dev",
            "--allow-provisional",
            "--models-only",
        ],
    )

    try:
        with pytest.raises(FitValidationError, match="No report figures were changed"):
            _MODULE.main()
    finally:
        env.set_output_root(None)

    assert sync_calls == []


def test_a_model_that_supplies_no_number_is_skipped_rather_than_fatal(
    tmp_path, monkeypatch, capsys
):
    """The other half of the roles doctrine: a development step whose fit fails
    validation is reported and left out, and the models that carry a number
    still sync. Before roles were declared this was the all-or-nothing failure
    of #301, where one stale rung took every model down with it.
    """
    valid = MODEL_REGISTRY[publication_models()[0]]
    step = MODEL_REGISTRY[models_with_role(ModelRole.DEVELOPMENT_STEP)[0]]
    output_root = tmp_path / "output"
    _write_output(output_root, valid, state="complete")
    _write_output(output_root, step, state="running")
    sync_calls = []
    monkeypatch.setattr(_MODULE, "_sync_dir", lambda *args: sync_calls.append(args) or 0)
    # Past the model loop the script writes the convergence records and names
    # the figure cache; neither belongs in the checkout during a test.
    monkeypatch.setattr(_MODULE, "_write_convergence_records", lambda sources: None)
    monkeypatch.setattr(env, "REPORT_FIGS_DIR", str(tmp_path / "figures"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sync_report_figures.py",
            "--output-dir",
            str(output_root),
            "--config",
            "dev",
            "--allow-provisional",
            "--models-only",
        ],
    )

    try:
        _MODULE.main()
    finally:
        env.set_output_root(None)

    out = capsys.readouterr().out
    assert f"[skip] {step.model_id}-{step.config_name}: development-step" in out
    assert len(sync_calls) == 1
    assert sync_calls[0][0].endswith(f"{valid.model_id}-{valid.config_name}")
