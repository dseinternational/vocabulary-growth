# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify that reuse cannot hide changed inputs, missing outputs or failed fits."""

import importlib.util
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import dse_research_utils.statistics.models.sampling as sampling
import pandas as pd
import pytest

from vocab_growth import environment as env
from vocab_growth.analysis_frames import analysis_frame_hash
from vocab_growth.fit_artifacts import normalise_for_json, write_fit_state
from vocab_growth.models.definitions import VG01
from vocab_growth.models.implementation_identity import implementation_signature
from vocab_growth.recovery import resume
from vocab_growth.recovery.refit import make_recovery_definition, recovery_fit_dir
from vocab_growth.recovery.simulate import simulation_dir
from vocab_growth.workflow_cache import checkpoint_matches, record_checkpoint

ROOT = Path(__file__).parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def output_root(tmp_path):
    env.set_output_root(str(tmp_path))
    yield tmp_path
    env.set_output_root(None)


@pytest.mark.parametrize("change", ["input", "output", "missing", "malformed"])
def test_checkpoint_rejects_changes_and_missing_evidence(tmp_path, change):
    output = tmp_path / "result.csv"
    output.write_text("one")
    checkpoint = tmp_path / "checkpoint.json"
    inputs = {"seed": 1}
    record_checkpoint(checkpoint, inputs, [output])
    assert checkpoint_matches(checkpoint, inputs)
    if change == "input":
        inputs = {"seed": 2}
    elif change == "output":
        output.write_text("two")
    elif change == "missing":
        output.unlink()
    else:
        checkpoint.write_text("[]")
    assert not checkpoint_matches(checkpoint, inputs)


def test_comparison_reuses_only_the_same_invocation_and_contributors(
    output_root, monkeypatch
):
    module = _script("resume_comparison")
    comparisons = output_root / "comparisons"
    comparisons.mkdir()
    fit = output_root / "models/VG01-fit"
    fit.mkdir(parents=True)
    manifest = fit / "fit_manifest.json"
    manifest.write_text("first fit")
    calls = []
    monkeypatch.setattr(
        module, "comparison_code_signature", lambda _: {"code": "current"}
    )
    monkeypatch.setattr(module, "source_data_hash", lambda _: "data")

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["env"][env.OUTPUT_DIR_ENV_VAR] == str(output_root)
        (comparisons / "result.csv").write_text("result")
        (comparisons / "comparison_manifest.json").write_text(
            json.dumps(
                {
                    "scripts": {
                        "compare_models.py": {
                            "outputs": ["result.csv"],
                            "contributing_fits": {"VG01-fit": {}},
                            "generated": len(calls),
                        }
                    }
                }
            )
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.run_comparison("compare_models.py", [])
    assert not module.run_comparison("compare_models.py", [])
    manifest.write_text("replacement fit")
    assert module.run_comparison("compare_models.py", [])
    assert module.run_comparison("compare_models.py", ["--other-option"])
    (comparisons / "result.csv").unlink()
    assert module.run_comparison("compare_models.py", ["--other-option"])
    (fit / "fit_state.json").write_text('{"state": "failed"}')
    assert module.run_comparison("compare_models.py", ["--other-option"])
    assert not module.run_comparison("compare_models.py", ["--other-option"])
    # A fit the comparison did not record (skipped, or not yet fitted) still
    # counts: a model that appears later must not be left out of a reused table.
    other = output_root / "models/VG02-fit"
    other.mkdir()
    (other / "fit_manifest.json").write_text("new fit")
    assert module.run_comparison("compare_models.py", ["--other-option"])
    (other / "trace.nc").write_text("trace")
    assert module.run_comparison("compare_models.py", ["--other-option"])
    assert not module.run_comparison("compare_models.py", ["--other-option"])
    assert len(calls) == 7


def test_comparison_that_does_not_refresh_provenance_cannot_create_a_checkpoint(
    output_root, monkeypatch
):
    module = _script("resume_comparison")
    result = output_root / "old-result.csv"
    result.write_text("old result")
    monkeypatch.setattr(
        module,
        "comparison_inputs",
        lambda *a: ({"entry": {"generated": "old"}}, [result]),
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match="did not refresh"):
        module.run_comparison("compare_models.py", [])
    assert not (output_root / "workflow-checkpoints/compare_models.py.json").exists()


def test_changed_execution_inputs_clear_quarto_freeze_but_plain_source_changes_do_not(
    tmp_path,
):
    module = _script("render_reports")
    checkpoint = tmp_path / "report-checkpoint.json"
    frozen = tmp_path / "_freeze/chapter/execute-results.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("cached numbers")
    checkpoint.write_text(json.dumps({"inputs": {"execution": {"data": "one"}}}))
    module.invalidate_frozen_execution(
        checkpoint,
        {"execution": {"data": "one"}, "source": "new text"},
        tmp_path,
        force=False,
    )
    assert frozen.exists()
    module.invalidate_frozen_execution(
        checkpoint, {"execution": {"data": "two"}}, tmp_path, force=False
    )
    assert not frozen.exists()


def test_recovery_simulation_skips_complete_outputs_but_changed_seed_reruns(
    output_root, monkeypatch
):
    calls = []
    directory = Path(simulation_dir(VG01, 1))
    monkeypatch.setattr(
        resume, "simulation_inputs", lambda *a, **kw: {"seed": kw["random_seed"]}
    )

    def simulate(*args, **kwargs):
        calls.append(kwargs["random_seed"])
        directory.mkdir(parents=True, exist_ok=True)
        for name in (
            resume.SIMULATION_FILENAME,
            resume.SYNTHETIC_FRAME_FILENAME,
            resume.TRUTH_FILENAME,
        ):
            (directory / name).write_text(str(kwargs["random_seed"]))

    monkeypatch.setattr(resume, "simulate_replicate", simulate)
    kwargs = dict(
        replicate=1,
        definition=VG01,
        fit_definition=VG01,
        truth_source="prior",
        do_fit=False,
    )
    assert resume.run_recovery_stages("vg01", "test", random_seed=1, **kwargs) == []
    assert resume.run_recovery_stages("vg01", "test", random_seed=1, **kwargs) == [
        "simulation"
    ]
    assert resume.run_recovery_stages("vg01", "test", random_seed=2, **kwargs) == []
    assert (
        resume.run_recovery_stages("vg01", "test", random_seed=2, fresh=True, **kwargs)
        == []
    )
    assert calls == [1, 2, 2]


@pytest.mark.parametrize(
    "change",
    [
        None,
        "frame",
        "definition",
        "sampling",
        "code",
        "diagnostics",
        "state",
        "trace",
        "dirty",
    ],
)
def test_recovery_fit_reuse_keeps_the_existing_validation_gate(
    output_root, monkeypatch, change
):
    directory = Path(simulation_dir(VG01, 1))
    directory.mkdir(parents=True)
    frame = pd.DataFrame({"age": [12.0], "spoken": [10]})
    fitted = make_recovery_definition(VG01, 1, truth_definition=VG01)
    output = Path(recovery_fit_dir("vg01", 1, definition=VG01, truth_definition=VG01))
    output.mkdir(parents=True)
    payload = {
        "model": {
            "model_id": VG01.model_id,
            "definition": normalise_for_json(fitted),
            "implementation": implementation_signature(),
        },
        "sampling": {
            "configuration_name": "rep",
            "parameters": asdict(sampling.get_sampling_configuration("rep")),
        },
        "data": {
            "source_data_hash": "data",
            "analysis_frame_hash": analysis_frame_hash(frame),
        },
        "code": {"commit": "commit", "dirty": False},
    }
    if change == "definition":
        payload["model"]["definition"]["n_trials"] = 1
    elif change == "sampling":
        payload["sampling"]["parameters"]["draws"] = 1
    elif change == "code":
        payload["model"]["implementation"] = {}
    (output / "fit_manifest.json").write_text(json.dumps(payload))
    (output / "trace.nc").touch()
    (output / "diagnostics_summary.json").write_text(
        json.dumps(
            {
                "scan_completed": True,
                "max_rhat": 1.001,
                "min_ess": 1000,
                "checks": {
                    "rhat": True,
                    "ess": True,
                    "divergences": True,
                    "bfmi": True,
                },
            }
            if change != "diagnostics"
            else {}
        )
    )
    write_fit_state(
        str(output),
        "failed" if change == "state" else "complete",
        model_id=fitted.model_id,
        config_name=fitted.config_name,
        sampling_config_name="rep",
    )
    inputs = {"truth": "fixed"}
    record_checkpoint(
        output / ".recovery-checkpoint.json", inputs, [output / "fit_manifest.json"]
    )
    if change == "trace":
        (output / "trace.nc").unlink()
    if change == "frame":
        frame.loc[0, "spoken"] = 20
    monkeypatch.setattr(
        resume,
        "load_simulation",
        lambda *a, **kw: (frame, SimpleNamespace(close=lambda: None), {}),
    )
    monkeypatch.setattr(
        resume,
        "git_metadata",
        lambda _: {"commit": "commit", "dirty": change == "dirty"},
    )
    monkeypatch.setattr(resume, "source_data_hash", lambda _: "data")
    assert resume.recovery_fit_is_current(
        "vg01",
        directory,
        "rep",
        replicate=1,
        definition=VG01,
        fit_definition=VG01,
        truth_overrides=(),
        inputs=inputs,
    ) is (change is None)


def test_model_render_reuses_present_outputs_and_force_rebuilds(
    output_root, monkeypatch
):
    module = _script("fit_model")
    from vocab_growth import render_cache

    directory = output_root / "model"
    directory.mkdir()
    (directory / "index.qmd").write_text("report")
    inputs = {"template": "first"}
    monkeypatch.setattr(render_cache, "render_inputs", lambda *a: dict(inputs))
    calls = []

    def render(*args, **kwargs):
        calls.append(args)
        (directory / "index.html").write_text("rendered report")

    monkeypatch.setattr(module.subprocess, "run", render)
    module._render_output(str(directory))
    module._render_output(str(directory))
    assert len(calls) == 1
    inputs["template"] = "changed include"
    module._render_output(str(directory))
    (directory / "index.html").unlink()
    module._render_output(str(directory))
    module._render_output(str(directory), force=True)
    assert len(calls) == 4


@pytest.mark.parametrize("target", ["docs/report", "docs/comparison/index.qmd"])
def test_book_and_comparison_renders_reuse_only_complete_outputs(
    output_root, monkeypatch, target
):
    module = _script("render_reports")
    monkeypatch.setattr(env, "ROOT_DIR", str(output_root))
    directory = (
        output_root / Path(target).parent
        if target.endswith(".qmd")
        else output_root / target
    )
    directory.mkdir(parents=True)
    if target.endswith(".qmd"):
        (directory / "index.qmd").write_text("report")
        output = directory
    else:
        output = output_root / "rendered"
    inspection = {
        "config": {"project": {"output-dir": "../../rendered"}},
        "files": {"input": ["index.qmd", "results.qmd"]},
    }
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda *a, **kw: json.dumps(inspection)
    )
    inputs = {"execution": {"data": "first"}}
    monkeypatch.setattr(module, "report_inputs", lambda: json.loads(json.dumps(inputs)))
    calls = []

    def render(*args, **kwargs):
        calls.append(args)
        output.mkdir(exist_ok=True)
        for name in ("index.html", "results.html"):
            (output / name).write_text("report")
        assets = output / "results_files/figure-html"
        assets.mkdir(parents=True, exist_ok=True)
        (assets / "plot.png").write_text("plot")

    monkeypatch.setattr(module.subprocess, "run", render)
    assert module.render_report(target)
    assert not module.render_report(target)
    (output / "index.html").unlink()
    assert module.render_report(target)
    (output / "results_files/figure-html/plot.png").unlink()
    assert module.render_report(target)
    inputs["execution"]["data"] = "new comparison table"
    assert module.render_report(target)
    assert module.render_report(target, force=True)
    assert len(calls) == 5


def test_figure_preparation_reuses_only_matching_inputs_and_outputs(
    output_root, monkeypatch
):
    module = _script("prepare_report_figures")
    monkeypatch.setattr(module, "runtime_identity", lambda: {"code": "current"})
    figure = output_root / "figure.png"
    calls = []

    def generate(seed):
        calls.append(seed)
        figure.write_text(str(seed))
        return [figure]

    monkeypatch.setattr(module, "run_illustrations", generate)
    assert module.run_stage("illustrations", draws=20, seed=1)
    assert not module.run_stage("illustrations", draws=20, seed=1)
    assert module.run_stage("illustrations", draws=20, seed=2)
    figure.unlink()
    assert module.run_stage("illustrations", draws=20, seed=2)
    assert module.run_stage("illustrations", draws=20, seed=2, fresh=True)
    assert calls == [1, 2, 2, 2]


def test_failed_comparison_does_not_leave_a_reusable_checkpoint(
    output_root, monkeypatch
):
    module = _script("resume_comparison")
    checkpoint = output_root / "workflow-checkpoints/compare_models.py.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text("old checkpoint")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.run_comparison("compare_models.py", [], fresh=True)
    assert not checkpoint.exists()


def test_recovery_scoring_refuses_an_older_simulation(output_root, monkeypatch):
    module = _script("fit_recovery")
    directory = Path(simulation_dir(VG01, 1))
    directory.mkdir(parents=True)
    output = Path(recovery_fit_dir("vg01", 1, definition=VG01, truth_definition=VG01))
    output.mkdir(parents=True)
    (output / "trace.nc").touch()
    (output / "recovery_source.json").write_text(
        json.dumps({"simulation": {"seed": 1}, "source_model": {}})
    )
    closed = []
    monkeypatch.setattr(module, "available_replicates", lambda *a, **kw: [1])
    monkeypatch.setattr(module, "sampling_tier", lambda _: "test")
    monkeypatch.setattr(
        module,
        "load_simulation",
        lambda *a, **kw: (
            None,
            SimpleNamespace(close=lambda: closed.append(True)),
            {"simulation": {"seed": 2}, "model": {}},
        ),
    )
    with pytest.raises(ValueError, match="fit does not match the current simulation"):
        module._score("vg01", VG01, "vg01", config="test")
    assert closed == [True]


@pytest.mark.parametrize("current", [True, False])
def test_recovery_checks_free_space_only_before_a_required_fit(
    output_root, monkeypatch, current
):
    calls = []
    monkeypatch.setattr(
        resume, "recovery_fit_inputs", lambda *a: {"simulation": "same"}
    )
    monkeypatch.setattr(resume, "recovery_fit_is_current", lambda *a, **kw: current)
    monkeypatch.setattr(env, "preflight_disk", lambda *a, **kw: calls.append("disk"))

    def fit(*args, **kwargs):
        calls.append("fit")
        output = Path(
            recovery_fit_dir("vg01", 1, definition=VG01, truth_definition=VG01)
        )
        output.mkdir(parents=True)
        for name in (
            "fit_manifest.json",
            "fit_state.json",
            "recovery_source.json",
            "diagnostics_summary.json",
        ):
            (output / name).write_text("checked output")

    monkeypatch.setattr(resume, "fit_recovery_replicate", fit)
    reused = resume.run_recovery_stages(
        "vg01",
        "test",
        replicate=1,
        definition=VG01,
        fit_definition=VG01,
        truth_source="prior",
        do_simulate=False,
    )
    assert reused == (["fit"] if current else [])
    assert calls == ([] if current else ["disk", "fit"])
