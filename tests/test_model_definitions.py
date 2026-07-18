# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for declarative model specification validation."""

from dataclasses import replace

import pytest

from vocab_growth.models.definitions import (
    MODEL_REGISTRY,
    VG01,
    VG16,
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


def test_definition_rejects_reference_age_outside_explicit_gp_domain():
    invalid = replace(VG01, gp_domain_months=(20, 80))

    with pytest.raises(ValueError, match="must lie in its GP domain"):
        validate_model_definition(invalid)


def test_cross_lag_requires_understood_subject_effect():
    invalid = replace(VG16, use_subject_re_u=False)

    with pytest.raises(ValueError, match="requires use_subject_re_u"):
        validate_model_definition(invalid)
