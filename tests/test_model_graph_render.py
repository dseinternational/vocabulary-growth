# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""`render_model_graph` must skip a missing `dot` quietly and report a failing one.

The blanket `except Exception` this replaces existed for the first case and
silently swallowed the second, which is how a sensitivity variant lost its
`gp_model_graph.svg` unnoticed on 2026-09-06: the path was 262 characters and
graphviz, which is not long-path-aware, could not write it.
"""

from __future__ import annotations

import graphviz
import pytest

from vocab_growth.models import common


class _Digraph:
    def __init__(self, exc):
        self._exc = exc

    def render(self, **_kwargs):
        raise self._exc


def _drive(monkeypatch, exc, tmp_path):
    monkeypatch.setattr(
        common.pymc_utils, "model_to_graphviz", lambda _model: _Digraph(exc)
    )
    common.render_model_graph(object(), str(tmp_path))


def test_a_missing_dot_binary_is_a_quiet_skip(monkeypatch, tmp_path, capsys):
    _drive(monkeypatch, graphviz.ExecutableNotFound(["dot"]), tmp_path)
    out = capsys.readouterr().out
    assert "Skipped model graph" in out
    assert "Model graph failed" not in out


def test_a_dot_that_runs_and_fails_is_reported_with_the_path_length(
    monkeypatch, tmp_path, capsys
):
    exc = graphviz.backend.execute.CalledProcessError(1, ["dot", "-Tsvg"])
    _drive(monkeypatch, exc, tmp_path)
    out = capsys.readouterr().out.replace("\n", "")

    assert "Model graph failed" in out, "a write failure must not be a quiet skip"
    assert "Skipped model graph" not in out
    assert "MAX_PATH" in out, "the message should name the usual Windows cause"
    expected = len(str(tmp_path / "gp_model_graph")) + 4
    assert str(expected) in out, "the target's character count should be printed"


def test_the_render_failure_is_still_not_fatal(monkeypatch, tmp_path):
    """The figure is optional; losing it must never lose the fit."""
    exc = graphviz.backend.execute.CalledProcessError(1, ["dot"])
    try:
        _drive(monkeypatch, exc, tmp_path)
    except Exception as raised:  # pragma: no cover - the assertion is the point
        pytest.fail(f"render_model_graph must not propagate: {raised!r}")
