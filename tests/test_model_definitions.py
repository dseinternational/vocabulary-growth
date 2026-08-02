# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for declarative model specification validation."""

from dataclasses import replace

import pytest

from vocab_growth.models.definitions import (
    MODEL_REGISTRY,
    VG01,
    VG16,
    KappaAnchorPriorParams,
    KappaPriorParams,
    _kappa_priors,
    validate_model_definition,
    validate_model_registry,
)


def test_all_registered_model_definitions_are_valid():
    assert MODEL_REGISTRY
    validate_model_registry()


@pytest.mark.parametrize("definition", MODEL_REGISTRY.values(), ids=MODEL_REGISTRY)
def test_each_registered_model_definition_validates(definition):
    validate_model_definition(definition)


def test_definition_rejects_unsorted_query_ages():
    invalid = replace(VG01, ages_query=[24, 12])

    with pytest.raises(ValueError, match="sorted"):
        validate_model_definition(invalid)


@pytest.mark.parametrize("config_name", ["../escape", "nested/path", r"nested\path", ".."])
def test_definition_rejects_unsafe_output_path_label(config_name):
    invalid = replace(VG01, config_name=config_name)

    with pytest.raises(ValueError, match="path-safe"):
        validate_model_definition(invalid)


def test_definition_rejects_reference_age_outside_explicit_gp_domain():
    invalid = replace(VG01, gp_domain_months=(20, 80))

    with pytest.raises(ValueError, match="must lie in its GP domain"):
        validate_model_definition(invalid)


def test_cross_lag_requires_understood_subject_effect():
    invalid = replace(VG16, use_subject_re_u=False)

    with pytest.raises(ValueError, match="requires use_subject_re_u"):
        validate_model_definition(invalid)


def test_definition_rejects_unordered_kappa_anchors():
    invalid = replace(VG01, kappa=replace(VG01.kappa, anchor_ages=(36.0, 18.0)))

    with pytest.raises(ValueError, match=r"kappa\.anchor_ages must be ordered"):
        validate_model_definition(invalid)


def test_definition_rejects_kappa_anchor_outside_the_gp_domain():
    # The anchors are reference ages like slope_anchors, so they are held to the
    # same rule: a prior stated at an age the model never sees is not checkable.
    invalid = replace(VG01, kappa=replace(VG01.kappa, anchor_ages=(4.0, 36.0)))

    with pytest.raises(ValueError, match="must lie in its GP domain"):
        validate_model_definition(invalid)


@pytest.mark.parametrize("model_id", ["VG01", "VG03", "VG11"])
def test_migrated_models_use_the_two_anchor_kappa_form(model_id):
    kappa = MODEL_REGISTRY[model_id.lower()].kappa

    assert isinstance(kappa, KappaAnchorPriorParams)


@pytest.mark.parametrize(
    "definition",
    [d for k, d in MODEL_REGISTRY.items() if k not in {"vg01", "vg03", "vg11"}],
    ids=[k for k in MODEL_REGISTRY if k not in {"vg01", "vg03", "vg11"}],
)
def test_unmigrated_models_keep_the_legacy_kappa_form(definition):
    # The two-anchor priors are calibrated for spoken counts out of 810. A model
    # picking them up by accident would inherit a dispersion prior for a
    # different quantity, so the split is asserted rather than left to review.
    for name, kappa in _kappa_priors(definition):
        assert isinstance(kappa, KappaPriorParams), name
