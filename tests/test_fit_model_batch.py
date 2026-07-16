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
