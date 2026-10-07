# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Protect a fit of record when development and reporting share an output root.

Check compatibility before fitting and again before promoting the result.
"""

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
    REPLACE_MODEL_OF_RECORD_ENV_VAR,
    model_of_record_replacement_refusal,
    normalise_for_json,
    write_fit_state,
)
from vocab_growth.models.common import PREPARE_STAGE_NAME, run_fit_pipeline
from vocab_growth.models.definitions import VG01
from vocab_growth.models.implementation_identity import implementation_signature

CANONICAL_NAME = f"{VG01.model_id}-{VG01.config_name}"


def _fit_of_record(
    output_dir: Path,
    *,
    tier: str = "rep",
    dirty: bool = False,
    state: str = "complete",
    signature: dict | None = None,
    diagnostics: dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": {
            "model_id": VG01.model_id,
            "config_name": VG01.config_name,
            "definition": normalise_for_json(VG01),
            "implementation": signature or implementation_signature(),
        },
        "sampling": {
            "configuration_name": tier,
            "parameters": asdict(sampling.get_sampling_configuration(tier)),
        },
        "code": {"commit": "0123456789abcdef", "dirty": dirty},
    }
    (output_dir / "fit_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output_dir / "trace.nc").touch()
    (output_dir / "diagnostics_summary.json").write_text(json.dumps(diagnostics or {
        "scan_completed": True, "max_rhat": 1.001, "min_ess": 1000,
        "checks": {"rhat": True, "ess": True, "divergences": True, "bfmi": True},
    }))
    write_fit_state(
        str(output_dir), state, model_id=VG01.model_id,
        config_name=VG01.config_name, sampling_config_name=tier,
    )


def _other_signature() -> dict:
    signature = implementation_signature()
    return {**signature, "sha256": "0" * 64, "sources": {**signature["sources"], "edited.py": "x"}}


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
    monkeypatch.delenv(REPLACE_MODEL_OF_RECORD_ENV_VAR, raising=False)


def _refusal(directory: Path, tier: str = "rep") -> str | None:
    return model_of_record_replacement_refusal(
        str(directory), sampling_config_name=tier,
        current_implementation=implementation_signature(),
    )


def test_a_fit_under_other_code_is_refused_and_says_what_moved(tmp_path):
    canonical = tmp_path / "models" / CANONICAL_NAME
    _fit_of_record(canonical, signature=_other_signature())
    refusal = _refusal(canonical)
    assert refusal is not None
    assert "different executable-code signature" in refusal
    assert "edited.py" in refusal
    assert "--replace-model-of-record" in refusal


def test_a_lower_tier_fit_is_refused_even_under_the_same_code(tmp_path):
    canonical = tmp_path / "models" / CANONICAL_NAME
    _fit_of_record(canonical)
    refusal = _refusal(canonical, "dev")
    assert refusal is not None and "a dev fit would replace" in refusal


def test_a_reporting_fit_under_the_same_code_may_replace_it(tmp_path):
    """An escalation rung within a cycle replaces a fit made by the same code."""
    canonical = tmp_path / "models" / CANONICAL_NAME
    _fit_of_record(canonical)
    assert _refusal(canonical) is None


@pytest.mark.parametrize("existing", [
    {"tier": "dev"},
    {"dirty": True},
    {"state": "failed"},
    {"diagnostics": {"scan_completed": True, "max_rhat": 1.2, "min_ess": 1000,
                     "checks": {"rhat": False, "ess": True}}},
])
def test_only_a_fit_of_record_is_protected(tmp_path, existing):
    canonical = tmp_path / "models" / CANONICAL_NAME
    _fit_of_record(canonical, signature=_other_signature(), **existing)
    assert _refusal(canonical) is None


def test_an_accepted_exception_still_counts_as_a_fit_of_record(tmp_path):
    canonical = tmp_path / "models" / CANONICAL_NAME
    _fit_of_record(canonical, signature=_other_signature(), diagnostics={
        "scan_completed": True, "max_rhat": 1.012, "min_ess": 900,
        "rhat_failing": ["ell"], "checks": {"rhat": False, "ess": True},
        "accepted_rhat_exception": {"parameters": ["ell"], "decided": "test"},
    })
    assert _refusal(canonical) is not None


def test_variant_recovery_and_experiment_directories_are_not_protected(tmp_path):
    other = tmp_path / "models" / f"{CANONICAL_NAME}-recovery-r01"
    _fit_of_record(other, signature=_other_signature())
    assert _refusal(other) is None


def test_the_override_allows_a_deliberate_refit(tmp_path, monkeypatch):
    canonical = tmp_path / "models" / CANONICAL_NAME
    _fit_of_record(canonical, signature=_other_signature())
    monkeypatch.setenv(REPLACE_MODEL_OF_RECORD_ENV_VAR, "1")
    assert _refusal(canonical) is None


def _quiet_pipeline(monkeypatch) -> None:
    monkeypatch.setattr("vocab_growth.models.common.env_info.report_environment_info", lambda: None)
    monkeypatch.setattr(
        "vocab_growth.models.common.package_metadata.report_package_versions", lambda packages: None
    )
    monkeypatch.setattr("vocab_growth.models.common.run_banner", lambda *a, **k: None)


def _prepare(context) -> None:
    frame = pd.DataFrame({"study": ["toy", "toy"], "age": [24.0, 36.0], "spoken": [2, 12]})
    context.set_model_data(
        model_data.BinomialModelData(X_obs=frame[["age"]].to_numpy(), y_obs=np.array([2, 12]), n_trials=810),
        frame,
    )
    Path(context.reporting.output_dir, "trace.nc").touch()


def test_the_pipeline_refuses_at_launch_before_any_stage_runs(tmp_path, monkeypatch):
    env.set_output_root(str(tmp_path))
    canonical = tmp_path / "models" / CANONICAL_NAME
    _fit_of_record(canonical, signature=_other_signature())
    before = (canonical / "fit_manifest.json").read_bytes()
    _quiet_pipeline(monkeypatch)
    ran = []
    try:
        with pytest.raises(RuntimeError, match="Refusing to replace"):
            run_fit_pipeline("dev", VG01, stages=[(PREPARE_STAGE_NAME, lambda ctx: ran.append(1))])
    finally:
        env.set_output_root(None)
    assert ran == []
    assert (canonical / "fit_manifest.json").read_bytes() == before
    assert not (tmp_path / ".staging").exists() or not any((tmp_path / ".staging").iterdir())


def test_a_fit_of_record_that_appears_while_sampling_is_not_replaced(tmp_path, monkeypatch):
    """The second check, at promotion; the finished fit is kept under failed/."""
    env.set_output_root(str(tmp_path))
    canonical = tmp_path / "models" / CANONICAL_NAME
    _quiet_pipeline(monkeypatch)

    def prepare_then_record(context):
        _prepare(context)
        _fit_of_record(canonical, signature=_other_signature())

    try:
        with pytest.raises(RuntimeError, match="Refusing to replace"):
            run_fit_pipeline("dev", VG01, stages=[(PREPARE_STAGE_NAME, prepare_then_record)])
    finally:
        env.set_output_root(None)
    manifest = json.loads((canonical / "fit_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sampling"]["configuration_name"] == "rep"
    retained = list((tmp_path / "failed").glob(f"{CANONICAL_NAME}-*"))
    assert len(retained) == 1 and (retained[0] / "trace.nc").is_file()
