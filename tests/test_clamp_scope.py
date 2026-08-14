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


# Every DS joint model adopted CLAMP_Q_ONLY on 2026-08-14. Pinning the set makes
# a model silently joining or leaving it visible, because either way it is a
# definition change that invalidates that model and demands a refit.
CLAMP_Q_ONLY_MODELS = {"vg05", "vg07", "vg08", "vg09", "vg10", "vg14", "vg15", "vg16"}


@pytest.mark.parametrize("model_id", sorted(MODEL_REGISTRY))
def test_the_clamp_scope_of_every_model_is_the_one_recorded(model_id):
    payload = normalise_for_json(MODEL_REGISTRY[model_id])
    if "clamp_mean_above_hi_anchor" not in payload:
        pytest.skip(f"{model_id} has no clamp field")
    value = payload["clamp_mean_above_hi_anchor"]
    if model_id in CLAMP_Q_ONLY_MODELS:
        assert value == CLAMP_Q_ONLY, (
            f"{model_id} should clamp q only; got {value!r}"
        )
        assert clamp_targets(value) == (False, True)
    else:
        assert isinstance(value, bool) and value is False, (
            f"{model_id} is not in the q-only set but serialises {value!r}"
        )


def test_the_variant_is_now_the_inverse_of_the_adopted_behaviour():
    """`clamp-q-only` became the model of record, so the variant restores `True`."""
    base = MODEL_REGISTRY["vg10"]
    variant = build_variant("vg10", "clamp-both")[0]

    assert clamp_targets(base.clamp_mean_above_hi_anchor) == (False, True)
    assert clamp_targets(variant.clamp_mean_above_hi_anchor) == (True, True)
    assert variant.config_name.endswith("clamp-both")
    # make_variant must not mutate the registered definition.
    assert base.clamp_mean_above_hi_anchor == CLAMP_Q_ONLY


def test_the_variant_changes_only_the_clamp():
    """A one-factor variant: everything else must match the model of record."""
    base = MODEL_REGISTRY["vg10"]
    variant = build_variant("vg10", "clamp-both")[0]
    ignore = {"clamp_mean_above_hi_anchor", "config_name", "banner"}
    differing = [
        f.name
        for f in dataclasses.fields(base)
        if f.name not in ignore
        and getattr(base, f.name) != getattr(variant, f.name)
    ]
    assert not differing, f"variant also changes: {differing}"
