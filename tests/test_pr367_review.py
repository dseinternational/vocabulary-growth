# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Counterexamples and workflow checks raised in PR 367 review."""

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from vocab_growth import comparison as c
from vocab_growth import comparisons_provenance as provenance
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.predictive_mixtures import conditioned_resample, importance_weights


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location("pr367_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_comparison_signature_tracks_local_dependencies_and_ignores_prose(tmp_path, monkeypatch):
    from vocab_growth import environment as env

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "producer.py").write_text('import helper\nVALUE = 1\n"attribute documentation"\n', encoding="utf-8")
    (scripts / "helper.py").write_text("import nested\nVALUE = 2\n", encoding="utf-8")
    (scripts / "nested.py").write_text("VALUE = 3\n", encoding="utf-8")
    (scripts / "upload.py").write_text("VALUE = 4\n", encoding="utf-8")
    monkeypatch.setattr(env, "ROOT_DIR", str(tmp_path))
    before = provenance.comparison_code_signature("producer.py --table")
    assert set(before["script_sources"]) == {"producer.py", "helper.py", "nested.py"}
    (scripts / "producer.py").write_text('"module prose"\nimport helper\nVALUE=1 # comment\n"revised attribute prose"\n', encoding="utf-8")
    (scripts / "upload.py").write_text("VALUE = 42\n", encoding="utf-8")
    assert provenance.comparison_code_signature("producer.py --table") == before
    (scripts / "nested.py").write_text("VALUE = 5\n", encoding="utf-8")
    after = provenance.comparison_code_signature("producer.py --table")
    assert after != before
    assert after["script_sources"]["nested.py"] != before["script_sources"]["nested.py"]


def test_comparison_signature_tracks_package_and_relative_imports(tmp_path, monkeypatch):
    from vocab_growth import environment as env

    scripts = tmp_path / "scripts"
    (scripts / "helpers").mkdir(parents=True)
    for name, source in {"producer.py": "from scripts import helpers\n",
                         "__init__.py": "VALUE=1\n",
                         "helpers/__init__.py": "from . import numerical\n",
                         "helpers/numerical.py": "VALUE=2\n"}.items():
        (scripts / name).write_text(source, encoding="utf-8")
    monkeypatch.setattr(env, "ROOT_DIR", str(tmp_path))
    assert set(provenance.comparison_code_signature("producer.py")["script_sources"]) == {
        "producer.py", "__init__.py", "helpers/__init__.py", "helpers/numerical.py"}


@pytest.mark.parametrize("label", ["", "../outside.py"])
def test_comparison_signature_refuses_invalid_generator(label):
    with pytest.raises(ValueError, match="inside scripts"):
        provenance.comparison_code_signature(label)


def test_comparison_signature_follows_generator_directory_imports(tmp_path, monkeypatch):
    from vocab_growth import environment as env

    scripts = tmp_path / "scripts"
    (scripts / "experiments").mkdir(parents=True)
    (scripts / "experiments" / "producer.py").write_text("import helper\n", encoding="utf-8")
    (scripts / "experiments" / "helper.py").write_text("VALUE=1\n", encoding="utf-8")
    monkeypatch.setattr(env, "ROOT_DIR", str(tmp_path))
    assert set(provenance.comparison_code_signature("experiments/producer.py")["script_sources"]) == {
        "experiments/producer.py", "experiments/helper.py"}


@pytest.mark.parametrize("allow_caveats", [False, True])
def test_comparison_publication_passes_config_and_caveat_policy(tmp_path, monkeypatch, allow_caveats):
    import vocab_growth.analysis_frames as frames
    import vocab_growth.fit_artifacts as artefacts

    seen = []
    monkeypatch.setattr(frames, "expected_analysis_frame_hash", lambda *args: "frame")
    monkeypatch.setattr(artefacts, "validate_fit_output", lambda path, **kwargs: seen.append(kwargs) or [])
    directory = tmp_path / f"{MODEL_REGISTRY['vg11'].model_id}-{MODEL_REGISTRY['vg11'].config_name}"
    assert provenance._publication_fit_errors(str(directory), "raw", config="rep-lite", allow_caveats=allow_caveats) == []
    assert seen[0]["expected_sampling_config_name"] == "rep-lite"
    assert seen[0]["require_clean_convergence"] is (not allow_caveats)
    assert seen[0]["require_convergence_evidence"] is True


def test_publication_cli_passes_reviewed_options(monkeypatch):
    publisher = script("publish_comparison.py")
    seen = []

    def validate(path, **kwargs):
        seen.append(kwargs)
        raise ValueError("stop before staging")

    monkeypatch.setattr(sys, "argv", ["publish_comparison.py", "--config", "rep-lite", "--allow-caveats", "--dry-run"])
    monkeypatch.setattr(publisher.env, "set_output_root", lambda _: None)
    monkeypatch.setattr(publisher, "validate_inputs", validate)
    with pytest.raises(ValueError, match="stop before staging"):
        publisher.main()
    assert seen == [{"config": "rep-lite", "allow_caveats": True}]


def _trace(tmp_path, key):
    """Extend the saved plot grid beyond the reporting ages, as real fits do."""
    directory = tmp_path / f"{MODEL_REGISTRY[key].model_id}-{MODEL_REGISTRY[key].config_name}"
    directory.mkdir()
    ages = np.array([12., 24., 60., 84., 108.])
    p = ages / 150
    def grid(x):
        return (("chain", "draw", "age"), np.asarray(x)[None, None, :])
    data = {name: grid(p) for name in ("p_plot", "p_u_plot", "p_s_plot", "p_any_plot", "p_any_indep_plot")}
    data.update({name: grid(np.repeat(.5, len(ages))) for name in ("q_plot", "r_plot", "pi_sign_only_plot", "pi_both_plot", "pi_speak_only_plot")})
    data.update({name: grid(np.log(p / (1 - p))) for name in ("f_plot", "f_u_plot")})
    data["h_plot"] = grid(np.zeros(len(ages)))
    data.update({name: (("chain", "draw", "study"), np.zeros((1, 1, 1))) for name in ("delta", "delta_u", "delta_q")})
    path = directory / "trace.nc"
    xr.DataTree.from_dict({"posterior": xr.Dataset(data), "constant_data": xr.Dataset({"X_plot": ("age", ages)})}).to_netcdf(path)
    return str(path)


def test_public_trajectory_loaders_apply_caps_before_crossings(tmp_path):
    path = _trace(tmp_path, "vg20")
    frame = pd.DataFrame({"age": [12., 108.], "study_code": [0, 0]})
    ages, u, s = c.load_population_trajectory(path, 810)
    assert ages[-1] == 72
    assert np.isnan(c.compute_q_at_U(ages, u, s, np.array([450.]))).all()
    weighted_ages, wu, _ = c.load_population_trajectory_weighted(path, 810, frame)
    assert weighted_ages[-1] == 72
    assert np.isnan(c.first_crossing_age(wu, weighted_ages, 450.)).all()
    spoken_ages, spoken = c.load_weighted_outcome_trajectory(path, 810, frame, "spoken")
    assert spoken_ages[-1] == max(MODEL_REGISTRY["vg20"].ages_query)
    assert spoken.shape[-1] == len(spoken_ages)
    sign_path = _trace(tmp_path, "vg15")
    sign_ages, values = c.load_sign_speech_trajectory(sign_path, 810)
    assert sign_ages[-1] == 72
    assert np.isnan(c.first_crossing_age(values["any"], sign_ages, 450.)).all()
    any_ages, _ = c.load_p_any_trajectory(sign_path, 810)
    assert any_ages[-1] == 72
    td_path = _trace(tmp_path, "vg12")
    assert c.load_univariate_trajectory(td_path, 810)[0][-1] == 25
    assert c.load_univariate_trajectory_weighted(td_path, 810, frame)[0][-1] == 25


def test_unregistered_trace_requires_explicit_reporting_definition(tmp_path):
    with pytest.raises(ValueError, match="supply a definition"):
        c._trajectory_definition(str(tmp_path / "trace.nc"))


def test_icc_bootstrap_refits_on_all_children_then_selects_repeats(monkeypatch):
    from types import SimpleNamespace

    rank = script("experiments/rank_stability.py")
    rng = np.random.default_rng(392)
    ids = np.r_[np.repeat(np.arange(8), 3), np.arange(8, 28)]
    age = rng.uniform(12, 50, len(ids))
    values = .02 * age + rng.normal(size=28)[ids] + rng.normal(size=len(ids)) * .2
    values[24:] += .15 * age[24:]  # singleton adjustment differs from repeats
    frame = pd.DataFrame({"child": ids, "age": age, "study": "one", "logit": values})
    design = rank._design(age, frame.study)
    frame["resid"] = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
    repeats = frame.groupby("child").filter(lambda g: len(g) > 1)
    expected = rank.icc(repeats)
    # An identity resample must reproduce the point estimator, including its
    # singleton-based first-stage adjustment. This is not a coverage assertion.
    monkeypatch.setattr(rank, "RNG", SimpleNamespace(choice=lambda values, **kwargs: values))
    lo, hi = rank.icc_ci(frame, n_boot=1)
    assert lo == pytest.approx(expected)
    assert hi == pytest.approx(expected)


def test_degenerate_importance_weights_warn_and_are_reported():
    logw = np.full((500, 2), -1000.)
    logw[0, 0] = 0.
    logw[:, 1] = 0.
    with pytest.warns(RuntimeWarning, match="ESS as low as 1.0"):
        draws, ess = conditioned_resample(np.tile(np.arange(500)[:, None], (1, 2)), logw,
                                          np.random.default_rng(5), return_ess=True)
    np.testing.assert_allclose(ess, [1., 500.])
    assert np.all(draws[:, 0] == 0)
    with warnings.catch_warnings(record=True) as caught:
        weights, ess = importance_weights(np.zeros(500))
    assert not caught
    assert ess == pytest.approx(500)
    assert weights.sum() == pytest.approx(1)


def test_expressive_self_check_exercises_the_nested_helper(monkeypatch):
    expressive = script("compare_ds_td_expressive.py")

    def obsolete(*args, **kwargs):
        raise AssertionError("obsolete approximation used")

    monkeypatch.setattr(expressive.C, "fraction_below_reference_percentile", obsolete)
    expressive._verify()


def test_nested_experiment_outputs_do_not_poison_publication(tmp_path):
    comparisons = tmp_path / "comparisons"
    experiment = comparisons / "experiments" / "age_at_word_count"
    experiment.mkdir(parents=True)
    (experiment / "age_at_word_count_vg20.csv").write_text("x\n1\n", encoding="utf-8")
    (comparisons / "summary.csv").write_text("x\n1\n", encoding="utf-8")
    provenance.write_comparison_manifest(str(comparisons), script="pool_descriptives.py",
                                         contributing={}, outputs=["summary.csv"], source_data_hash="raw")
    errors, warnings = provenance.validate_comparison_manifest(str(comparisons), str(tmp_path / "models"),
                        current_source_data_hash="raw", require_publication=True)
    assert errors == warnings == []
    assert json.loads((comparisons / "comparison_manifest.json").read_text())["scripts"]
