# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for sensitivity-runner coverage."""

import importlib.util
import sys
from pathlib import Path

from vocab_growth.models.common_bivariate_re import fit_bivariate_re_model

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "fit_sensitivity.py"
_SPEC = importlib.util.spec_from_file_location("fit_sensitivity_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_vg13_single_administration_variant_has_runner():
    assert _MODULE._RUNNER_BY_KEY["vg13"] is fit_bivariate_re_model
