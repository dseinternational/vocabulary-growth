# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``clamp_mean_above_hi_anchor`` applies to two means, and they are separable.

The flag levels a trajectory's mean off above the high slope anchor instead of
extrapolating the logit-linear trend to the top of the GP domain. It was a
single boolean applied to *both* the understood mean and the production ratio
``q``; because spoken is ``p_U(a) * q(a)``, that compounds, and the spoken
trajectory inherits two levelled factors.

Measurement on 2026-08-14 (``notes/202608141200-clamp-q-only.md``) showed the
saturation the flag was added for is ``q``'s alone, so ``"q_only"`` was added as
a third value. Two things about it are easy to get wrong and are pinned here.

* **It is truthy.** ``if definition.clamp_mean_above_hi_anchor:`` clamps both
  means under ``"q_only"``, silently doing the opposite of what was asked.
  Engines must resolve through :func:`clamp_targets`.
* **It must not change how ``True``/``False`` serialise.** The manifest
  fingerprints the definition with ``asdict`` and compares whole-object
  equality, so any drift there invalidates every model of record at once.
"""

import dataclasses
import json

import pytest

from vocab_growth.fit_artifacts import normalise_for_json
from vocab_growth.models.definitions import (
    CLAMP_Q_ONLY,
    MODEL_REGISTRY,
    clamp_targets,
)
from vocab_growth.sensitivity.registry import build_variant


def test_clamp_targets_resolves_all_three_values():
    assert clamp_targets(True) == (True, True)
    assert clamp_targets(False) == (False, False)
    assert clamp_targets(CLAMP_Q_ONLY) == (False, True)


def test_q_only_is_truthy_which_is_why_the_resolver_exists():
    """The trap: a bare truth test clamps both means under ``"q_only"``."""
    assert bool(CLAMP_Q_ONLY) is True
    understood, q = clamp_targets(CLAMP_Q_ONLY)
    assert understood is False and q is True


def test_true_and_false_still_serialise_as_booleans():
    """Widening the field's type must not move any existing model's fingerprint."""
    for value in (True, False):
        assert json.dumps(normalise_for_json(value)) == json.dumps(value)


@pytest.mark.parametrize("model_id", sorted(MODEL_REGISTRY))
def test_registered_definitions_serialise_the_clamp_as_a_bool(model_id):
    """No model of record opts into ``"q_only"``; it is a sensitivity variant only.

    If one ever does, this test should be updated deliberately -- the change
    invalidates that model and demands a refit.
    """
    payload = normalise_for_json(MODEL_REGISTRY[model_id])
    if "clamp_mean_above_hi_anchor" not in payload:
        pytest.skip(f"{model_id} has no clamp field")
    assert isinstance(payload["clamp_mean_above_hi_anchor"], bool), (
        f"{model_id} serialises the clamp as "
        f"{payload['clamp_mean_above_hi_anchor']!r}, which changes its fingerprint"
    )


def test_the_variant_clamps_q_only_and_leaves_the_baseline_alone():
    base = MODEL_REGISTRY["vg10"]
    variant = build_variant("vg10", "clamp-q-only")[0]

    assert clamp_targets(base.clamp_mean_above_hi_anchor) == (True, True)
    assert clamp_targets(variant.clamp_mean_above_hi_anchor) == (False, True)
    assert variant.config_name.endswith("clamp-q-only")
    # make_variant must not mutate the registered definition.
    assert base.clamp_mean_above_hi_anchor is True


def test_the_variant_changes_only_the_clamp():
    """A one-factor variant: everything else must match the model of record."""
    base = MODEL_REGISTRY["vg10"]
    variant = build_variant("vg10", "clamp-q-only")[0]
    ignore = {"clamp_mean_above_hi_anchor", "config_name", "banner"}
    differing = [
        f.name
        for f in dataclasses.fields(base)
        if f.name not in ignore
        and getattr(base, f.name) != getattr(variant, f.name)
    ]
    assert not differing, f"variant also changes: {differing}"
