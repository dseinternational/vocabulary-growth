# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Controlled pairing and coverage checks for sensitivity comparisons."""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from dse_research_utils.statistics.models.sampling import get_sampling_configuration
from test_sensitivity_compare import _write_gate_payload

from vocab_growth import analysis_frames
from vocab_growth.fit_artifacts import normalise_for_json
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.models.implementation_identity import implementation_signature
from vocab_growth.sensitivity.compare import (
    compare_dirs,
    coverage_report,
    load_headlines,
    pairing_errors,
    required_quantities,
    summarise,
)
from vocab_growth.sensitivity.registry import build_variant


def _pair(tmp_path, monkeypatch, key="vg23", name="eta-flat"):
    base = MODEL_REGISTRY[key]
    variant, = build_variant(key, name)

    def frame_hash(model_key, definition):
        return "restricted-frame" if getattr(definition, "dse_native_only", False) else "full-frame"

    monkeypatch.setattr(analysis_frames, "expected_analysis_frame_hash", frame_hash)
    signature = implementation_signature()
    paths = []
    for label, definition in (("base", base), ("variant", variant)):
        directory = tmp_path / label
        directory.mkdir()
        payload = {
            "model": {"definition": normalise_for_json(definition), "implementation": signature},
            "data": {"analysis_frame_hash": frame_hash(key, definition)},
            "sampling": {"configuration_name": "dev", "parameters": asdict(get_sampling_configuration("dev"))},
            "code": {"dirty": False, "commit": "abc123"},
        }
        (directory / "fit_manifest.json").write_text(json.dumps(payload))
        (directory / "fit_state.json").write_text(json.dumps({"state": "complete"}))
        (directory / "trace.nc").touch()
        _write_gate_payload(directory)
        paths.append(directory)
    return paths


def _mutate(directory, mutate):
    path = directory / "fit_manifest.json"
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload))


def test_prior_pair_requires_the_current_prepared_frame(tmp_path, monkeypatch):
    base, var = _pair(tmp_path, monkeypatch)
    assert pairing_errors(base, var, "vg23", "eta-flat") == []
    _mutate(var, lambda m: m["data"].update(analysis_frame_hash="unrelated-frame"))
    assert any("prepared analysis frame" in error for error in pairing_errors(base, var, "vg23", "eta-flat"))


def test_registered_data_restriction_uses_its_own_expected_frame(tmp_path, monkeypatch):
    base, var = _pair(tmp_path, monkeypatch, "vg10", "dse-native-only")
    assert pairing_errors(base, var, "vg10", "dse-native-only") == []
    _mutate(var, lambda m: m["data"].update(analysis_frame_hash="full-frame"))
    assert pairing_errors(base, var, "vg10", "dse-native-only")


@pytest.mark.parametrize("change,fragment", [
    (lambda m: m["model"]["definition"].update(subject_re_correlation_eta=7), "definition differs"),
    (lambda m: m["model"].pop("implementation"), "implementation signature"),
    (lambda m: m["code"].update(dirty=True), "dirty"),
    (lambda m: m["sampling"]["parameters"].update(draws=1), "draws"),
])
def test_pairing_checks_values_code_and_provenance(tmp_path, monkeypatch, change, fragment):
    base, var = _pair(tmp_path, monkeypatch)
    _mutate(var, change)
    assert any(fragment in error for error in pairing_errors(base, var, "vg23", "eta-flat"))


def test_missing_manifest_and_incomplete_baseline_are_unverifiable(tmp_path, monkeypatch):
    base, var = _pair(tmp_path, monkeypatch)
    (var / "fit_manifest.json").unlink()
    (base / "fit_state.json").write_text(json.dumps({"state": "sampling"}))
    errors = pairing_errors(base, var, "vg23", "eta-flat")
    assert any("variant" in error for error in errors)
    assert any("baseline" in error for error in errors)


def _count(directory, suffix, median, *, prefixed):
    prefix = f"Ey_{suffix}" if prefixed else "Ey"
    pd.DataFrame({"age_months": [30.5], f"{prefix}_median": [median],
                  f"{prefix}_ci_lo": [90], f"{prefix}_ci_hi": [110]}).to_csv(
        directory / f"posterior_summary_{suffix}.csv", index=False)


def test_both_count_dialects_include_understood_spoken_and_signed(tmp_path):
    base, var = tmp_path / "base", tmp_path / "variant"
    base.mkdir(), var.mkdir()
    for suffix in ("u", "s", "sign"):
        _count(base, suffix, 100, prefixed=False)
        _count(var, suffix, 200, prefixed=True)
    comparison = compare_dirs(base, var)
    assert set(comparison.quantity) == {"Ey_understood", "Ey_spoken", "Ey_signed"}
    assert not comparison.within_baseline_ci.any()
    assert set(comparison.age_months) == {30.5}
    assert coverage_report(base, var) == (3, 3, [])


def test_missing_from_both_fits_still_fails_expected_coverage(tmp_path):
    base, var = tmp_path / "base", tmp_path / "variant"
    base.mkdir(), var.mkdir()
    _count(base, "u", 100, prefixed=True)
    _count(var, "u", 100, prefixed=True)
    required = required_quantities("vg15", MODEL_REGISTRY["vg15"])
    coverage = coverage_report(base, var, required=required)
    assert {"Ey_spoken", "Ey_signed", "psi"}.issubset(coverage[2])
    _write_gate_payload(base), _write_gate_payload(var)
    row = summarise(compare_dirs(base, var), var, "v", baseline_dir=base,
                    validation_errors=[], coverage=coverage)
    assert row["status"] == "partial-coverage"


@pytest.mark.parametrize("parameter", ["rho_uq", "tau_subj_u_1", "subject_variance_share"])
def test_parameter_shift_is_checked_when_trajectories_stay_fixed(tmp_path, parameter):
    base, var = tmp_path / "base", tmp_path / "variant"
    base.mkdir(), var.mkdir()
    for directory, estimate in ((base, 0.5), (var, 0.9)):
        _count(directory, "u", 100, prefixed=True)
        _write_gate_payload(directory)
        pd.DataFrame({"mean": [estimate], "eti89_lb": [0.4], "eti89_ub": [0.6]},
                     index=[parameter]).to_csv(directory / "diagnostics.csv")
    comparison = compare_dirs(base, var)
    row = summarise(comparison, var, "v", baseline_dir=base,
                    validation_errors=[], coverage=coverage_report(base, var))
    assert row["verdict"] == f"sensitive: {parameter}"
    assert comparison.set_index("quantity").loc[parameter, "estimate_kind"] == "mean"


def test_baseline_convergence_and_missing_pairing_checks_prevent_robustness(tmp_path):
    base, var = tmp_path / "base", tmp_path / "variant"
    base.mkdir(), var.mkdir()
    for directory in (base, var):
        _count(directory, "u", 100, prefixed=True)
        _write_gate_payload(directory)
    comparison = compare_dirs(base, var)
    coverage = coverage_report(base, var)
    assert summarise(comparison, var, "v", baseline_dir=base, coverage=coverage)["status"] == "unverified-pairing"
    for payload, status in [
        ({"passed": False, "checks": {"rhat": False, "ess": True}}, "non-converged"),
        ({"passed": False, "divergences": 3, "checks": {"rhat": True, "ess": True, "divergences": False, "bfmi": True, "diagnostics_assessable": True}}, "converged-with-caveats"),
    ]:
        _write_gate_payload(base, **payload)
        row = summarise(comparison, var, "v", baseline_dir=base,
                        validation_errors=[], coverage=coverage)
        assert row["status"] == status
        assert not row["verdict"].startswith("robust")


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_count_summary_cannot_disappear_from_coverage(tmp_path, bad):
    _count(tmp_path, "u", bad, prefixed=True)
    with pytest.raises(ValueError, match="non-finite"):
        load_headlines(tmp_path)


def test_structural_schema_covers_the_model_defining_parameters():
    for key, names in {
        "vg23": {"rho_uq"}, "vg16": {"beta_lag"},
        "vg19": {"tau_subj_u_1", "tau_subj_q_1", "tau_subj_u_rho"},
        "vg22": {"tau_subj_u_1", "tau_subj_q_1", "rho_uq"},
        "vg11": {"subject_variance_share", "v_total"},
    }.items():
        assert names.issubset(required_quantities(key, MODEL_REGISTRY[key]))


def test_cli_uses_pairing_and_required_quantity_checks(tmp_path, monkeypatch):
    import runpy
    import sys
    from pathlib import Path

    from vocab_growth import environment as env

    base, var = _pair(tmp_path, monkeypatch)
    model_root = tmp_path / "models"
    model_root.mkdir()
    definition = MODEL_REGISTRY["vg23"]
    variant, = build_variant("vg23", "eta-flat")
    directories = []
    for directory, record in ((base, definition), (var, variant)):
        destination = model_root / f"{record.model_id}-{record.config_name}"
        directory.rename(destination)
        directories.append(destination)
        for suffix in ("u", "s"):
            _count(destination, suffix, 100, prefixed=False)
        pd.DataFrame({"age_months": [30.5], "q_median": [0.5],
                      "q_ci_lo": [0.4], "q_ci_hi": [0.6]}).to_csv(
            destination / "posterior_summary_q.csv", index=False)
        parameters = sorted(required_quantities("vg23", record) - {"Ey_understood", "Ey_spoken", "q"})
        pd.DataFrame({"mean": 0.5, "eti89_lb": 0.4, "eti89_ub": 0.6},
                     index=parameters).to_csv(destination / "diagnostics.csv")
    monkeypatch.setattr(env, "models_output_dir", lambda: str(model_root))
    monkeypatch.setattr(env, "comparisons_output_dir", lambda: str(tmp_path / "comparisons"))
    monkeypatch.setattr(env, "output_root", lambda: str(tmp_path))
    output = tmp_path / "matrix.csv"
    script = Path(__file__).parents[1] / "scripts" / "compare_sensitivity.py"
    monkeypatch.setattr(sys, "argv", [str(script), "vg23", "--variant", "eta-flat", "--out", str(output)])
    runpy.run_path(str(script), run_name="__main__")
    row = pd.read_csv(output).iloc[0]
    assert row["verdict"].startswith("robust")
    _mutate(directories[1], lambda m: m["data"].update(analysis_frame_hash="wrong"))
    runpy.run_path(str(script), run_name="__main__")
    assert pd.read_csv(output).iloc[0]["status"] == "unverified-pairing"
