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
    fit_validation_kwargs,
    normalise_for_json,
    promote_staged_fit,
    validate_fit_output,
    write_fit_state,
)
from vocab_growth.models.common import PREPARE_STAGE_NAME, run_fit_pipeline
from vocab_growth.models.definitions import VG01


def _write_manifest(
    output_dir: Path,
    *,
    sampling_name: str = "rep",
    dirty: bool = False,
    frame_hash: str | None = "sha256:frame",
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
        "data": {
            "source_data_hash": "sha256:data",
            "analysis_frame_hash": frame_hash,
        },
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


def test_a_changed_prepared_frame_invalidates_a_fit(tmp_path):
    """The defect issue #266 finding 1 names: loader-rule drift went unseen.

    The raw-CSV fingerprint cannot see it, because the masking and exclusion
    rules run in Python *after* the CSVs are read — so a rule change leaves the
    raw hash equal while the frame the model was fitted to no longer exists.
    """
    output_dir = tmp_path / "fit"
    _write_complete_output(output_dir)

    unchanged = validate_fit_output(
        str(output_dir),
        expected_source_data_hash="sha256:data",
        expected_analysis_frame_hash="sha256:frame",
    )
    assert unchanged == []

    drifted = validate_fit_output(
        str(output_dir),
        # The raw data is untouched; only the prepared frame moved.
        expected_source_data_hash="sha256:data",
        expected_analysis_frame_hash="sha256:frame-after-a-rule-change",
    )
    assert len(drifted) == 1
    assert "prepared analysis frame differs" in drifted[0]


def test_a_matching_frame_hash_excuses_a_raw_data_mismatch(tmp_path):
    """New data a model never reads must not stale its fit.

    The raw fingerprint hashes every CSV in ``data/``, so a new Down syndrome
    study CSV changes it for every model — including the typically-developing
    models, whose prepared frames contain no Down syndrome rows. The model
    consumes the raw data only through its prepared frame, so a matching exact
    frame hash vouches for the fit and the fingerprint mismatch alone is not a
    reason to refit.
    """
    output_dir = tmp_path / "fit"
    _write_complete_output(output_dir)

    excused = validate_fit_output(
        str(output_dir),
        expected_source_data_hash="sha256:data-after-a-new-ds-study",
        expected_analysis_frame_hash="sha256:frame",
    )
    assert excused == []

    # Without the frame hash to vouch, the fingerprint stays a hard failure.
    unvouched = validate_fit_output(
        str(output_dir),
        expected_source_data_hash="sha256:data-after-a-new-ds-study",
    )
    assert unvouched == ["Raw data inputs differ from those used for this fit."]

    # When both moved, both are reported: the raw change is part of the
    # diagnosis, not hidden behind the frame drift.
    both = validate_fit_output(
        str(output_dir),
        expected_source_data_hash="sha256:data-after-a-new-ds-study",
        expected_analysis_frame_hash="sha256:frame-after-a-rule-change",
    )
    assert any("Raw data inputs differ" in error for error in both)
    assert any("prepared analysis frame differs" in error for error in both)
    assert len(both) == 2

    # A manifest that predates frame-hash recording cannot vouch either: an
    # old fit must not be excused just because the caller asked for the check.
    legacy_dir = tmp_path / "legacy-fit"
    _write_complete_output(legacy_dir)
    _write_manifest(legacy_dir, frame_hash=None)
    legacy = validate_fit_output(
        str(legacy_dir),
        expected_source_data_hash="sha256:data-after-a-new-ds-study",
        expected_analysis_frame_hash="sha256:frame",
    )
    assert any("Raw data inputs differ" in error for error in legacy)


def test_frame_hash_validation_is_opt_in_for_callers_without_the_registry(tmp_path):
    """Computing the expected hash rebuilds the frame, so it is not free.

    A caller that cannot afford it (or has no definition in hand) omits it and
    keeps the previous checks; it must not silently pass a `None` comparison.
    """
    output_dir = tmp_path / "fit"
    _write_complete_output(output_dir)

    assert (
        validate_fit_output(
            str(output_dir), expected_source_data_hash="sha256:data"
        )
        == []
    )

    kwargs = fit_validation_kwargs(
        "sync",
        expected_definition=VG01,
        expected_sampling_config_name="rep",
        expected_sampling_parameters=asdict(
            sampling.get_sampling_configuration("rep")
        ),
        current_source_data_hash="sha256:data",
        current_analysis_frame_hash="sha256:frame",
    )
    assert kwargs["expected_analysis_frame_hash"] == "sha256:frame"

    without = fit_validation_kwargs(
        "sync",
        expected_definition=VG01,
        expected_sampling_config_name="rep",
        expected_sampling_parameters=asdict(
            sampling.get_sampling_configuration("rep")
        ),
        current_source_data_hash="sha256:data",
    )
    assert "expected_analysis_frame_hash" not in without

    # A provisional sync deliberately carries no data checks at all, so it must
    # not acquire one through this argument.
    provisional = fit_validation_kwargs(
        "provisional-sync",
        expected_definition=VG01,
        expected_sampling_config_name="rep",
        expected_sampling_parameters=asdict(
            sampling.get_sampling_configuration("rep")
        ),
        current_source_data_hash="sha256:data",
        current_analysis_frame_hash="sha256:frame",
    )
    assert "expected_analysis_frame_hash" not in provisional
    assert "expected_source_data_hash" not in provisional


def test_sampling_compatibility_ignores_cores_and_accepts_stronger_tuning(tmp_path):
    output_dir = tmp_path / "fit"
    _write_complete_output(output_dir)
    manifest_path = output_dir / "fit_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sampling"]["parameters"].update(
        cores=1,
        draws=8000,
        tune=12000,
        target_accept=0.99,
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    errors = validate_fit_output(
        str(output_dir),
        expected_sampling_config_name="rep",
        expected_sampling_parameters=asdict(
            sampling.get_sampling_configuration("rep")
        ),
    )

    assert errors == []


def test_sampling_compatibility_rejects_less_effort_than_named_tier(tmp_path):
    output_dir = tmp_path / "fit"
    _write_complete_output(output_dir)
    manifest_path = output_dir / "fit_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sampling"]["parameters"]["draws"] = 500
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    errors = validate_fit_output(
        str(output_dir),
        expected_sampling_parameters=asdict(
            sampling.get_sampling_configuration("rep")
        ),
    )

    assert any("draws" in error and "at least" in error for error in errors)


def test_publish_policy_does_not_require_current_commit_to_match(tmp_path):
    output_dir = tmp_path / "fit"
    _write_complete_output(output_dir)

    kwargs = fit_validation_kwargs(
        "publish",
        expected_definition=VG01,
        expected_sampling_config_name="rep",
        expected_sampling_parameters=asdict(
            sampling.get_sampling_configuration("rep")
        ),
        current_git={"commit": "later-commit", "dirty": False},
        current_source_data_hash="sha256:data",
    )

    assert "expected_git" not in kwargs
    assert validate_fit_output(str(output_dir), **kwargs) == []


def test_resume_policy_requires_matching_clean_current_checkout(tmp_path):
    output_dir = tmp_path / "fit"
    _write_complete_output(output_dir)

    kwargs = fit_validation_kwargs(
        "resume",
        expected_definition=VG01,
        expected_sampling_config_name="rep",
        expected_sampling_parameters=asdict(
            sampling.get_sampling_configuration("rep")
        ),
        current_git={"commit": "later-commit", "dirty": True},
        current_source_data_hash="sha256:data",
    )
    errors = validate_fit_output(str(output_dir), **kwargs)

    assert any("Git commit" in error for error in errors)
    assert any("current checkout is dirty" in error for error in errors)


def test_provisional_sync_skips_current_git_and_data_expectations():
    kwargs = fit_validation_kwargs(
        "provisional-sync",
        expected_definition=VG01,
        expected_sampling_config_name="dev",
        expected_sampling_parameters=asdict(
            sampling.get_sampling_configuration("dev")
        ),
    )

    assert "expected_git" not in kwargs
    assert "expected_source_data_hash" not in kwargs


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


def test_terminal_fit_state_preserves_last_running_stage(tmp_path):
    output_dir = tmp_path / "fit"
    write_fit_state(
        str(output_dir),
        "running",
        model_id=VG01.model_id,
        config_name=VG01.config_name,
        sampling_config_name="rep",
        stage="Posterior sampling",
    )
    write_fit_state(
        str(output_dir),
        "failed",
        model_id=VG01.model_id,
        config_name=VG01.config_name,
        sampling_config_name="rep",
        error=RuntimeError("stopped"),
    )

    state = json.loads((output_dir / "fit_state.json").read_text(encoding="utf-8"))
    assert state["stage"] == "Posterior sampling"


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
        context = run_fit_pipeline(
            "dev", VG01, stages=[(PREPARE_STAGE_NAME, prepare)]
        )
    finally:
        env.set_output_root(None)

    assert Path(context.reporting.output_dir) == canonical
    assert not (canonical / "marker.txt").exists()
    state = json.loads((canonical / "fit_state.json").read_text())
    assert state["state"] == "complete"
    assert (canonical / "fit_manifest.json").is_file()

# --------------------------------------------------------------------------
# The two load-bearing lines in run_fit_pipeline
# --------------------------------------------------------------------------


def test_an_unclassified_sampling_tier_is_refused_before_the_banner(tmp_path, monkeypatch):
    """`--config` has no argparse `choices`, so this is the only thing stopping an
    unknown tier from reaching the sampler and producing output with no
    convergence-gate classification. It used to be a bare
    `is_reporting_quality_config(config)` expression statement with its return value
    discarded -- one deletion from gone, with nothing pinning it at this call site.
    """
    env.set_output_root(str(tmp_path))
    banner_calls = []
    monkeypatch.setattr(
        "vocab_growth.models.common.run_banner",
        lambda *a, **k: banner_calls.append(a),
    )

    def never_called(_context):  # pragma: no cover - must not run
        raise AssertionError("the pipeline ran a stage on an unclassified tier")

    try:
        with pytest.raises(ValueError, match="no convergence-gate classification"):
            run_fit_pipeline(
                "not-a-tier", VG01, stages=[(PREPARE_STAGE_NAME, never_called)]
            )
    finally:
        env.set_output_root(None)
    assert banner_calls == [], "the tier check must run before the banner"


def test_a_pipeline_whose_first_stage_is_not_data_preparation_is_refused(
    tmp_path, monkeypatch
):
    """The manifest is written after stage 0 and needs the prepared frame.

    Before this was asserted, an engine author who put the priors stage first --
    plausible, since priors depend on no data in any engine -- got "the fit
    context for VGnn-dev has no analysis DataFrame set" from a manifest writer
    they never invoked, with nothing in `run_fit_pipeline` to explain why ordering
    mattered.
    """
    env.set_output_root(str(tmp_path))
    monkeypatch.setattr(
        "vocab_growth.models.common.env_info.report_environment_info", lambda: None
    )
    monkeypatch.setattr(
        "vocab_growth.models.common.package_metadata.report_package_versions",
        lambda packages: None,
    )
    monkeypatch.setattr("vocab_growth.models.common.run_banner", lambda *a, **k: None)

    try:
        with pytest.raises(RuntimeError, match="Stage 0 of this engine"):
            run_fit_pipeline(
                "dev", VG01, stages=[("Priors and hyperparameters", lambda ctx: None)]
            )
    finally:
        env.set_output_root(None)


def test_a_substituted_first_stage_that_loads_the_data_is_accepted(
    tmp_path, monkeypatch
):
    """The recovery harness renames stage 0, and that must stay allowed.

    `recovery/refit.py` and `scripts/experiments/vg10_under_vg20_truth.py` replace
    the engine's own "Prepare data" with a loader for a simulated frame, under a
    name that says so. An earlier version of the guard above compared stage 0's
    *name*, which rejected both -- every `scripts/fit_recovery.py` fit died after
    its loader stage had already run. The precondition the manifest writer actually
    has is that the data is loaded, whatever the stage was called.
    """
    env.set_output_root(str(tmp_path))
    monkeypatch.setattr(
        "vocab_growth.models.common.env_info.report_environment_info", lambda: None
    )
    monkeypatch.setattr(
        "vocab_growth.models.common.package_metadata.report_package_versions",
        lambda packages: None,
    )
    monkeypatch.setattr("vocab_growth.models.common.run_banner", lambda *a, **k: None)

    def load_simulated_data(context):
        # The same two-row stub the "Prepare data" test above uses; what matters
        # here is that a differently-named stage 0 sets the same two fields.
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
        # Finalisation requires it, as in the test above; nothing here reads it.
        Path(context.reporting.output_dir, "trace.nc").touch()

    reached = []
    try:
        run_fit_pipeline(
            "dev",
            VG01,
            stages=[
                ("Load simulated data", load_simulated_data),
                ("Next stage", lambda ctx: reached.append(True)),
            ],
        )
    finally:
        env.set_output_root(None)

    assert reached == [True], "the pipeline stopped after the substituted stage 0"


def test_every_stage_factory_names_its_first_stage_the_way_the_pipeline_requires():
    """The contract, checked against the engines that expose a stage factory.

    The other engines build their stage lists inline inside their fit functions,
    which cannot be called without fitting; those are covered by the two engines
    here plus `tests/test_recovery_spec.py`, which asserts the same equality for
    every recovery-capable model.
    """
    from vocab_growth.models import catalogue
    from vocab_growth.models.definitions import MODEL_REGISTRY

    checked = 0
    for key, definition in MODEL_REGISTRY.items():
        engine = catalogue.get(key).engine
        if engine.stages is None:
            continue
        first = engine.resolve("stages")(definition)[0][0]
        assert first == PREPARE_STAGE_NAME, f"{key} ({engine.name}): {first!r}"
        checked += 1
    assert checked >= 2, f"only {checked} models exercised this contract"
