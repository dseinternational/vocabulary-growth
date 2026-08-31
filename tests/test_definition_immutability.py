# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Registered definitions and their fixed collections are immutable.

Issue #273. Twenty definitions are module-level singletons shared by every fit,
every sensitivity variant, every recovery replicate and every validator in the
same process, and until this they were mutable dataclasses holding mutable
lists. Nothing was observed mutating one; the point is that nothing *can*, so a
future edit to a shared definition fails at the write rather than as an
unexplained difference between two fits of the same model.

Two consequences worth stating. ``_as_definition_subclass`` shares nested prior
blocks with its base by reference — VG20, VG22 and VG23 all carry VG10's or
VG13's kappa objects — and freezing those blocks is what makes the sharing safe
rather than merely untested; the sensitivity override code has carried a comment
about exactly this aliasing risk since it was written. And a frozen definition
is hashable, so it can be a dictionary key or go in a set, which the plan
resolution in ``SubjectEffectPlan`` relies on for caching.

Serialisation is deliberately unaffected: ``normalise_for_json`` renders a tuple
and a list as the same JSON array, so freezing the classes and tupling the query
grids left every registered model's recorded definition byte-identical and no
fitted output was invalidated.
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
    """The claim that freezing invalidated no fitted output, as a test.

    A tuple and a list both render as a JSON array, so a fit recorded before
    this change still compares equal to the definition registered after it.
    """
    from vocab_growth.fit_artifacts import normalise_for_json

    base = MODEL_REGISTRY["vg10"]
    as_list = dataclasses.replace(base, ages_query=list(base.ages_query))
    assert normalise_for_json(as_list) == normalise_for_json(base)
    assert isinstance(normalise_for_json(base)["ages_query"], list)
