# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for :mod:`vocab_growth.environment` output-root resolution (issue #105).

The output root is resolved at call time with precedence
``set_output_root()`` override > ``$DSE_VOCAB_GROWTH_OUTPUT_DIR`` > repo-local
``output/``. The report figure cache (``docs/report/figures/``) is deliberately
kept in the checkout and must never follow a redirected root.
"""

from __future__ import annotations

import os

import pytest

import vocab_growth.environment as env


@pytest.fixture(autouse=True)
def _isolate_output_root(monkeypatch):
    """Start each test with no env var and no override; always restore after."""
    monkeypatch.delenv(env.OUTPUT_DIR_ENV_VAR, raising=False)
    env.set_output_root(None)
    yield
    env.set_output_root(None)


def test_default_is_repo_local_output():
    assert env.output_root() == os.path.join(env.ROOT_DIR, "output")


def test_models_and_comparisons_are_subdirs_of_root():
    env.set_output_root("/scratch/vg")
    assert env.models_output_dir() == os.path.join("/scratch/vg", "models")
    assert env.comparisons_output_dir() == os.path.join("/scratch/vg", "comparisons")


def test_env_var_redirects_root(monkeypatch):
    monkeypatch.setenv(env.OUTPUT_DIR_ENV_VAR, "/scratch/vg")
    assert env.output_root() == "/scratch/vg"
    assert env.models_output_dir() == os.path.join("/scratch/vg", "models")


def test_override_takes_precedence_over_env_var(monkeypatch):
    monkeypatch.setenv(env.OUTPUT_DIR_ENV_VAR, "/scratch/vg")
    env.set_output_root("/cli/dir")
    assert env.output_root() == "/cli/dir"


def test_clearing_override_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv(env.OUTPUT_DIR_ENV_VAR, "/scratch/vg")
    env.set_output_root("/cli/dir")
    env.set_output_root(None)
    assert env.output_root() == "/scratch/vg"


def test_override_is_expanded_and_absolute():
    env.set_output_root("~/foo/../foo")
    assert env.output_root() == os.path.abspath(os.path.expanduser("~/foo/../foo"))


def test_legacy_attributes_resolve_dynamically():
    # The historical module constants are now PEP 562 shims that honour overrides.
    env.set_output_root("/scratch/vg")
    assert env.OUTPUT_DIR == env.output_root() == "/scratch/vg"
    assert env.MODELS_OUTPUT_DIR == env.models_output_dir()
    assert env.COMPARISONS_OUTPUT_DIR == env.comparisons_output_dir()


def test_unknown_attribute_still_raises():
    with pytest.raises(AttributeError):
        _ = env.DOES_NOT_EXIST


def test_report_figs_dir_stays_repo_local(monkeypatch):
    monkeypatch.setenv(env.OUTPUT_DIR_ENV_VAR, "/scratch/vg")
    assert env.REPORT_FIGS_DIR.startswith(env.ROOT_DIR)
    assert "/scratch/vg" not in env.REPORT_FIGS_DIR
