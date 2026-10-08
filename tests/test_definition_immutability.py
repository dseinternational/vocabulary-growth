# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Keep registered definitions and their fixed collections immutable.

Definitions and nested prior blocks are shared across fits and variants.
Freezing both prevents one mutation from changing several models. Hashable
definitions also support cached child-effect plans. Tuple and list fields
serialise to the same JSON arrays.
"""

from __future__ import annotations

import dataclasses

import pytest

from vocab_growth.models.definitions import (
    MODEL_REGISTRY,
    KappaAnchorPriorParams,
    KappaPriorParams,
    SubjectVariancePartitionParams,
)

_MODEL_KEYS = sorted(MODEL_REGISTRY)

#: Every prior block a definition can nest. All must be frozen: a mutable one
#: shared by reference between a base and its subclass-derived children is a
#: single edit away from changing several models at once.
_PRIOR_BLOCKS = (KappaPriorParams, KappaAnchorPriorParams, SubjectVariancePartitionParams)


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_registered_definitions_are_frozen(model_key):
    definition = MODEL_REGISTRY[model_key]
    assert dataclasses.is_dataclass(definition)
    assert type(definition).__dataclass_params__.frozen, (
        f"{model_key}'s definition class {type(definition).__name__} is mutable"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        definition.n_trials = 1


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_no_registered_definition_holds_a_mutable_collection(model_key):
    """A frozen dataclass holding a list is frozen in name only."""
    definition = MODEL_REGISTRY[model_key]
    mutable = {
        field.name: type(getattr(definition, field.name)).__name__
        for field in dataclasses.fields(definition)
        if isinstance(getattr(definition, field.name), (list, dict, set))
    }
    assert not mutable, (
        f"{model_key} holds mutable collections: {mutable}. Use a tuple; "
        "`normalise_for_json` renders both as a JSON array, so the recorded "
        "definition is unchanged."
    )


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_query_grids_are_tuples(model_key):
    assert isinstance(MODEL_REGISTRY[model_key].ages_query, tuple)


@pytest.mark.parametrize("block", _PRIOR_BLOCKS)
def test_every_nested_prior_block_type_is_frozen(block):
    assert block.__dataclass_params__.frozen, f"{block.__name__} is mutable"


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_every_nested_dataclass_a_definition_carries_is_frozen(model_key):
    """Checked over the actual instances, not a list of types kept up to date."""
    definition = MODEL_REGISTRY[model_key]
    for field in dataclasses.fields(definition):
        value = getattr(definition, field.name)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            assert type(value).__dataclass_params__.frozen, (
                f"{model_key}.{field.name} is a mutable "
                f"{type(value).__name__}; it may be shared by reference with "
                "another definition."
            )


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_definitions_are_hashable(model_key):
    """Frozen plus eq gives a hash, which the plan resolution caches on."""
    definition = MODEL_REGISTRY[model_key]
    assert hash(definition) == hash(definition)
    assert len({definition, definition}) == 1


def test_replace_still_builds_a_variant():
    """Freezing must not break the one supported way to derive a definition."""
    base = MODEL_REGISTRY["vg10"]
    variant = dataclasses.replace(base, config_name=f"{base.config_name}-probe")
    assert variant.config_name.endswith("-probe")
    assert base.config_name == MODEL_REGISTRY["vg10"].config_name
    assert type(variant) is type(base)


def test_the_serialised_definition_is_unaffected_by_the_container_type():
    """Serialise tuple and list query grids to the same definition payload."""
    from vocab_growth.fit_artifacts import normalise_for_json

    base = MODEL_REGISTRY["vg10"]
    as_list = dataclasses.replace(base, ages_query=list(base.ages_query))
    assert normalise_for_json(as_list) == normalise_for_json(base)
    assert isinstance(normalise_for_json(base)["ages_query"], list)
