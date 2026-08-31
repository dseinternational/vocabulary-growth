# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for sensitivity-runner coverage.

The runner a variant is fitted through used to be a hand-maintained table in
``scripts/fit_sensitivity.py``, duplicated again in ``scripts/refit_hightune.py``
and again as engine identity in three other places. It had gone stale: variants
were registered for VG16, VG21 and VG23 while neither script had a runner for
them, so ``fit_sensitivity.py vg16 lag-gap-12`` exited with "No sensitivity
variants for model: vg16" against a registry that holds five (issue #273).

Both scripts now derive the set from the variant registry and the runner from
:mod:`vocab_growth.models.catalogue`, which is what these tests pin.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from vocab_growth.models.catalogue import engine_for
from vocab_growth.models.common_bivariate_re import fit_bivariate_re_model
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.sensitivity.registry import VARIANTS


def _load(name: str):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_MODULE = _load("fit_sensitivity")
_HIGHTUNE = _load("refit_hightune")

_MODELS_WITH_VARIANTS = sorted({model_key for model_key, _ in VARIANTS})


def test_vg13_single_administration_variant_has_runner():
    assert _MODULE._runner("vg13") is fit_bivariate_re_model


@pytest.mark.parametrize("model_key", _MODELS_WITH_VARIANTS)
def test_every_model_with_variants_is_reachable(model_key):
    """A registered variant nobody can fit is a variant that does not exist.

    This is the check the hand-maintained table had no equivalent of: VG16's
    five variants, VG21's and VG23's one each were unreachable from both
    scripts.
    """
    assert model_key in _MODULE._MODELS_WITH_VARIANTS
    assert model_key in _HIGHTUNE._models_with_variants()
    assert callable(_MODULE._runner(model_key))


@pytest.mark.parametrize("model_key", _MODELS_WITH_VARIANTS)
def test_the_runner_is_the_engine_that_fits_the_model_of_record(model_key):
    """A variant is the model of record with one field overridden.

    Fitting it through a different engine would compare a variant of one graph
    against a model of record built from another, which is the failure the
    catalogue exists to make impossible.
    """
    assert model_key in MODEL_REGISTRY
    assert _MODULE._runner(model_key) is engine_for(model_key).resolve("fit")


def test_the_two_scripts_agree_on_which_models_have_variants():
    assert _MODULE._MODELS_WITH_VARIANTS == _HIGHTUNE._models_with_variants()
