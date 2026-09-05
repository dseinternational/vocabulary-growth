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


@pytest.mark.parametrize("purpose", ["render", "sync", "publish", "resume", "provisional-sync"])
def test_every_reuse_policy_requires_the_implementation_signature(purpose, tmp_path):
    directory = tmp_path / "fit"
    _write_complete_output(directory)
    kwargs = fit_validation_kwargs(
        purpose, expected_definition=VG01, expected_sampling_config_name="rep",
        expected_sampling_parameters={}, current_git={"commit": "abc123", "dirty": False},
        current_source_data_hash="sha256:data")
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
