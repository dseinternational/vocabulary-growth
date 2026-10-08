# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check data and definition guards when reusing a retained posterior.

Raw fingerprints cannot detect loader-rule changes and can change for sources
a model does not use. Prepared-frame hashes check the actual fitting rows.
These data checks complement definition and executable-implementation checks.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "resume_from_trace.py"
_SPEC = importlib.util.spec_from_file_location("resume_from_trace_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

MODEL_KEY = "vg01"
CONFIG = "rep"
CURRENT_FRAME = "sha256:frame-as-the-loader-builds-it-today"
CURRENT_RAW = "sha256:raw-as-the-csvs-are-today"


@pytest.fixture
def definition():
    from vocab_growth.models.definitions import MODEL_REGISTRY

    return MODEL_REGISTRY[MODEL_KEY]


@pytest.fixture
def written_manifest(tmp_path, definition, monkeypatch):
    """Write a manifest that passes every guard, and stub the two current hashes.

    Both hashes are stubbed rather than computed: this file is about the guard's
    logic, and building a real frame would make every case a data-loading test.
    """
    from dataclasses import asdict

    from vocab_growth.fit_artifacts import normalise_for_json

    monkeypatch.setattr(
        _MODULE, "expected_analysis_frame_hash", lambda key, defn: CURRENT_FRAME
    )
    monkeypatch.setattr(_MODULE, "source_data_hash", lambda data_dir: CURRENT_RAW)

    def write(*, frame=CURRENT_FRAME, raw=CURRENT_RAW):
        import json

        payload = {
            "model": {"definition": normalise_for_json(definition),
                      "implementation": _MODULE.implementation_signature()},
            "sampling": {"configuration_name": CONFIG,
                         "parameters": asdict(_MODULE.sampling.get_sampling_configuration(CONFIG))},
            "data": {"source_data_hash": raw},
        }
        if frame is not None:
            payload["data"]["analysis_frame_hash"] = frame
        directory = tmp_path / "retained"
        directory.mkdir(exist_ok=True)
        (directory / _MODULE.FIT_MANIFEST_FILENAME).write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return str(directory)

    return write


def _verify(retained_dir, definition):
    return _MODULE._verify(retained_dir, MODEL_KEY, definition, CONFIG)


def test_a_matching_frame_and_fingerprint_verifies(written_manifest, definition):
    """The baseline: nothing has moved, so the resume proceeds."""
    manifest = _verify(written_manifest(), definition)
    assert manifest["data"]["analysis_frame_hash"] == CURRENT_FRAME


def test_a_drifted_frame_is_refused_although_the_raw_data_is_unchanged(
    written_manifest, definition
):
    """The defect this guard exists for: a loader-rule change the fingerprint cannot see.

    Masking and exclusion rules run after the CSVs are read, so this combination
    -- raw hash equal, frame hash different -- is exactly what a changed rule
    looks like, and is what the fingerprint-only guard admitted.
    """
    with pytest.raises(ValueError, match="prepared analysis frame differs"):
        _verify(written_manifest(frame="sha256:frame-under-the-old-rules"), definition)


def test_a_matching_frame_excuses_a_changed_fingerprint(written_manifest, definition):
    """Accept raw-source changes when the model's prepared frame still matches.

    This establishes data compatibility only; other guards check the definition
    and executable implementation.
    """
    manifest = _verify(written_manifest(raw="sha256:a-new-study-csv-arrived"), definition)
    assert manifest["data"]["source_data_hash"] == "sha256:a-new-study-csv-arrived"


def test_both_hashes_moving_is_refused(written_manifest, definition):
    """With nothing left to vouch, the resume is refused rather than excused."""
    with pytest.raises(ValueError):
        _verify(
            written_manifest(frame="sha256:something-else", raw="sha256:also-else"),
            definition,
        )


def test_a_manifest_with_no_frame_hash_is_refused(written_manifest, definition):
    """Reject a missing prepared-frame hash because data compatibility cannot be verified."""
    with pytest.raises(ValueError, match="records no prepared-frame hash"):
        _verify(written_manifest(frame=None), definition)


def test_a_changed_definition_is_still_refused_first(written_manifest, definition):
    """The definition guard is unchanged by this work, and still fires."""
    import dataclasses

    other = dataclasses.replace(definition, config_name="not-the-registered-name")
    with pytest.raises(ValueError, match="model definition differs"):
        _MODULE._verify(written_manifest(), MODEL_KEY, other, CONFIG)


def test_a_changed_sampling_configuration_is_still_refused(written_manifest, definition):
    """Likewise the sampling-configuration guard."""
    with pytest.raises(ValueError, match="sampling configuration"):
        _MODULE._verify(written_manifest(), MODEL_KEY, definition, "dev")


@pytest.mark.parametrize("implementation", [None, {"schema_version": 1, "package_ast_sha256": "changed"}])
def test_missing_or_different_implementation_is_refused(written_manifest, definition, implementation):
    import json

    directory = written_manifest()
    path = Path(directory) / _MODULE.FIT_MANIFEST_FILENAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["model"]["implementation"] = implementation
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="implementation"):
        _verify(directory, definition)


def test_missing_sampling_parameters_are_refused(written_manifest, definition):
    import json

    directory = written_manifest()
    path = Path(directory) / _MODULE.FIT_MANIFEST_FILENAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["sampling"]["parameters"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sampling parameters"):
        _verify(directory, definition)


def test_loader_preserves_original_sampling_provenance(tmp_path, monkeypatch):
    import json
    from dataclasses import asdict
    from types import SimpleNamespace

    import xarray as xr

    sampling = asdict(_MODULE.sampling.get_sampling_configuration("dev"))
    retained = {"sampling": {"configuration_name": "dev", "parameters": sampling},
                "code": {"commit": "original", "dirty": True}, "runtime": {"host": "original"}}
    path = tmp_path / _MODULE.FIT_MANIFEST_FILENAME
    path.write_text(json.dumps({"code": {"commit": "new"}, "runtime": {"host": "new"}}), encoding="utf-8")
    trace = xr.DataTree.from_dict({"posterior": xr.Dataset({"x": (("chain", "draw"), [[1.]])})})
    monkeypatch.setattr(_MODULE.xr, "open_datatree", lambda _: trace)
    loaded = []
    context = SimpleNamespace(model=SimpleNamespace(free_RVs=[SimpleNamespace(name="x")], coords={}),
                              reporting=SimpleNamespace(output_dir=str(tmp_path)), set_trace=loaded.append)
    _MODULE._loader_stage("unused.nc", retained, implementation_change_reason="Reporting correction only")(context)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == [trace]
    assert asdict(context.sampling) == sampling
    assert saved["sampling"] == retained["sampling"]
    assert saved["code"] == {"commit": "new"}
    assert saved["runtime"] == {"host": "new"}
    assert saved["artefacts"]["retained_sampling_manifest"] == retained
    assert saved["artefacts"]["implementation_change_review"]["reason"] == "Reporting correction only"


@pytest.mark.parametrize("change", ["code", "packages", "missing", "frame"])
def test_reviewed_resume_only_relaxes_code_identity(written_manifest, definition, change):
    import json

    directory = written_manifest()
    path = Path(directory) / _MODULE.FIT_MANIFEST_FILENAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    implementation = manifest["model"]["implementation"]
    implementation["sha256"] = "older-code"
    if change == "packages":
        implementation["packages"]["numpy"] = {"version": "0.0"}
    elif change == "missing":
        manifest["model"]["implementation"] = None
    elif change == "frame":
        manifest["data"]["analysis_frame_hash"] = "older-frame"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="implementation"):
        _verify(directory, definition)
    if change == "code":
        assert _MODULE._verify(directory, MODEL_KEY, definition, CONFIG,
                               implementation_change_reason="Reviewed reporting-only change") == manifest
    else:
        with pytest.raises(ValueError, match="implementation|prepared analysis frame"):
            _MODULE._verify(directory, MODEL_KEY, definition, CONFIG,
                            implementation_change_reason="Reviewed reporting-only change")
