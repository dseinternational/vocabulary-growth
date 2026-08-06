# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for convergence-aware multi-model fitting in ``fit_model.py``."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from vocab_growth.models.common import ConvergenceGateError

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "fit_model.py"
_SPEC = importlib.util.spec_from_file_location("fit_model_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_batch_continues_after_convergence_failure():
    calls: list[str] = []

    def fit_ok(name):
        def fit(config):
            calls.append(f"{name}:{config}")
            return name

        return SimpleNamespace(fit=fit)

    def fit_fails(config):
        calls.append(f"bad:{config}")
        raise ConvergenceGateError("did not converge")

    selected = [
        ("first", fit_ok("first")),
        ("bad", SimpleNamespace(fit=fit_fails)),
        ("last", fit_ok("last")),
    ]

    contexts, timings, failures = _MODULE._fit_selected_models(selected, "rep")

    assert calls == ["first:rep", "bad:rep", "last:rep"]
    assert contexts == ["first", "last"]
    assert set(timings) == {"first", "bad", "last"}
    assert failures == {"bad": "did not converge"}


def test_batch_render_continues_after_one_model_fails(monkeypatch):
    calls: list[str] = []
    contexts = [
        SimpleNamespace(
            reporting=SimpleNamespace(model_name=name, output_dir=f"/{name}")
        )
        for name in ("first", "bad", "last")
    ]

    def render(output_dir, model_id=None):
        # model_id is passed so _render_output can refresh the report template
        # from docs/models/<model>/index.qmd before rendering; asserted below
        # because dropping it silently reverts --render-only to re-rendering
        # whatever template was current when the fit ran.
        calls.append((output_dir, model_id))
        if output_dir == "/bad":
            raise RuntimeError("quarto failed")

    monkeypatch.setattr(_MODULE, "_render_output", render)

    timings, failures = _MODULE._render_contexts(contexts)

    assert calls == [("/first", "first"), ("/bad", "bad"), ("/last", "last")]
    assert set(timings) == {"first", "bad", "last"}
    assert failures == {"bad": "RuntimeError: quarto failed"}


def test_render_pins_quarto_python_to_the_fitting_interpreter(monkeypatch, tmp_path):
    (tmp_path / "index.qmd").write_text("")
    (tmp_path / "index.html").write_text("")
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)
    monkeypatch.setenv("QUARTO_PYTHON", "/usr/bin/python3")
    monkeypatch.setenv("DSE_RENDER_ENV_SENTINEL", "inherited")

    _MODULE._render_output(str(tmp_path))

    assert captured["command"] == ["quarto", "render", str(tmp_path / "index.qmd")]
    env = captured["env"]
    # An inherited QUARTO_PYTHON is the failure mode, not an override to respect:
    # the report must be rendered by the interpreter that produced the fit.
    assert env["QUARTO_PYTHON"] == sys.executable
    assert env["DSE_RENDER_ENV_SENTINEL"] == "inherited"
