# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fits must track code changes independently of documentation commits."""

import json

import pytest
from test_fit_artifacts import _write_complete_output

from vocab_growth.fit_artifacts import fit_validation_kwargs, validate_fit_output
from vocab_growth.models import implementation_identity as identity
from vocab_growth.models.definitions import VG01


def test_comments_docstrings_and_whitespace_do_not_change_executable_identity():
    original = '"module"\nclass A:\n    "class"\n    def f(self, x):\n        "method"\n        return x + 1\n'
    prose = original.replace('"module"', '"new module"').replace('"class"', '"new class"').replace('"method"', '"new method"') + '\n# comment\n'
    assert identity.executable_source(original) == identity.executable_source(prose)
    assert identity.executable_source(original) != identity.executable_source(original.replace('x + 1', 'x + 2'))


def test_signature_covers_new_helpers_and_ignores_external_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(identity, "PACKAGE_ROOT", tmp_path)
    path = tmp_path / "engine.py"
    path.write_text('def likelihood(x):\n    return x * 2\n')
    before = identity.implementation_signature()
    (tmp_path / "report.qmd").write_text('A revised report.')
    assert identity.implementation_signature() == before
    path.write_text('def likelihood(x):\n    return x * 3\n')
    assert identity.implementation_signature() != before
    path.write_text('def likelihood(x):\n    return x * 2\n')
    (tmp_path / "helper.py").write_text('EPSILON = 1e-9\n')
    assert identity.implementation_signature() != before


def _kwargs_for(purpose):
    return fit_validation_kwargs(
        purpose, expected_definition=VG01, expected_sampling_config_name="rep",
        expected_sampling_parameters={}, current_git={"commit": "abc123", "dirty": False},
        current_source_data_hash="sha256:data")


@pytest.mark.parametrize("purpose", ["sync", "publish", "resume"])
def test_syndicating_a_fit_requires_the_implementation_signature(purpose, tmp_path):
    directory = tmp_path / "fit"
    _write_complete_output(directory)
    kwargs = _kwargs_for(purpose)
    assert "expected_implementation" in kwargs
    path = directory / "fit_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["model"].pop("implementation", None)
    path.write_text(json.dumps(manifest))
    assert any("implementation signature" in e for e in validate_fit_output(str(directory), **kwargs))
    manifest["model"]["implementation"] = kwargs["expected_implementation"]
    path.write_text(json.dumps(manifest))
    assert not any("implementation signature" in e for e in validate_fit_output(str(directory), **kwargs))
    manifest["model"]["implementation"] = {"sha256": "old-code"}
    path.write_text(json.dumps(manifest))
    assert any("implementation signature" in e for e in validate_fit_output(str(directory), **kwargs))


@pytest.mark.parametrize("purpose", ["render", "provisional-sync"])
def test_reading_a_fit_back_does_not_require_the_implementation_signature(purpose):
    """The signature covers the whole package, including plotting and reporting.

    Requiring it to re-render an existing fit, or to sync provisionally in the
    checkout where code is being edited, would make any code change enough to
    make every fit unusable for its own report.
    """
    assert "expected_implementation" not in _kwargs_for(purpose)


def test_the_signature_is_never_inferred_from_the_current_code(tmp_path):
    """``None`` means not checked, as it does for every other ``expected_*``.

    ``scripts/loso_compare.py`` and :mod:`vocab_growth.recovery.simulate` pass a
    definition to read a model of record back, not to publish from it; inferring
    the signature there would refuse every fit made before it existed.
    """
    directory = tmp_path / "fit"
    _write_complete_output(directory)
    path = directory / "fit_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["model"].pop("implementation", None)
    path.write_text(json.dumps(manifest))
    errors = validate_fit_output(str(directory), expected_definition=VG01)
    assert not any("implementation signature" in e for e in errors)


def test_a_mismatch_names_the_module_or_library_that_moved(tmp_path):
    directory = tmp_path / "fit"
    _write_complete_output(directory)
    expected = identity.implementation_signature()
    stale = json.loads(json.dumps(expected))
    stale["sha256"] = "old-code"
    stale["sources"]["models/common.py"] = "old-hash"
    stale["packages"]["numpy"] = {"version": "0.0.1", "commit": None}
    path = directory / "fit_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["model"]["implementation"] = stale
    path.write_text(json.dumps(manifest))
    (error,) = [
        e
        for e in validate_fit_output(
            str(directory), expected_definition=VG01, expected_implementation=expected
        )
        if "implementation signature" in e
    ]
    assert "numpy 0.0.1 ->" in error
    assert "models/common.py" in error
