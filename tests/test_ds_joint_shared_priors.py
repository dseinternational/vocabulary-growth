# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Which models share each DS-joint prior group, pinned rather than described.

Seven registrations restated the same eleven trajectory-prior and reporting field
values, each above its own verbatim copy of the rationale — 386 lines of literal
repetition, whose prose carried model counts that had gone stale in four separate
places ("all seventeen models" against twenty registered; "six models of record"
for a class tree holding twelve). Those counts *are* the argument a maintainer
weighs when deciding whether a new field is affordable, so they are asserted here
instead of written out: a count in a comment drifts silently, a count in a test
fails.

The values are now splatted from three module-level dicts. That keeps every
serialised field value byte-identical, so no fit is invalidated — this file's job
is to make the *next* divergence visible.
"""

from __future__ import annotations

import dataclasses

import pytest

from vocab_growth.models import definitions as D
from vocab_growth.models.definitions import MODEL_REGISTRY

#: Models expected to carry the shared DS-joint understood/q anchors and reporting
#: values. VG05, VG07-VG10, VG14 and VG16 splat them; VG19, VG20, VG22 inherit
#: through `_as_definition_subclass`; VG15 shares the q and reporting groups and
#: takes the understood anchors from its class defaults, which match.
_DS_JOINT_MODELS = frozenset(
    {"vg05", "vg07", "vg08", "vg09", "vg10", "vg14", "vg15", "vg16", "vg19", "vg20", "vg22"}
)

#: The typically-developing bivariate models, which deliberately carry *different*
#: anchors — they are the reason the groups cannot simply be class defaults.
_TD_BIVARIATE_MODELS = frozenset({"vg13", "vg21", "vg23"})

_GROUPS = {
    "understood": D._DS_JOINT_UNDERSTOOD_ANCHORS,
    "q": D._DS_JOINT_Q_ANCHORS,
    "reporting": D._DS_JOINT_REPORTING,
}


@pytest.mark.parametrize("group", sorted(_GROUPS))
def test_every_ds_joint_model_carries_the_group_verbatim(group):
    """The splat must reach every model the rationale claims it covers."""
    for key in sorted(_DS_JOINT_MODELS):
        definition = MODEL_REGISTRY[key]
        for name, expected in _GROUPS[group].items():
            assert getattr(definition, name) == expected, (
                f"{key}.{name} is {getattr(definition, name)!r}, not the shared "
                f"{expected!r}. Either the model has diverged deliberately — in which "
                f"case remove it from _DS_JOINT_MODELS and say why — or a splat was "
                f"dropped."
            )


@pytest.mark.parametrize("group", sorted(_GROUPS))
def test_the_td_bivariate_models_do_not_share_the_group(group):
    """The TD models are the contrast case, and must stay one.

    If a TD model ever matched the DS group on every field, the group would be a
    class default rather than a shared constant, and this file would be describing
    a distinction that no longer exists.
    """
    fields = _GROUPS[group]
    for key in sorted(_TD_BIVARIATE_MODELS):
        definition = MODEL_REGISTRY[key]
        present = {n: getattr(definition, n) for n in fields if hasattr(definition, n)}
        assert present != dict(fields), (
            f"{key} matches the shared DS {group} group on every field. If that is "
            "intended, the group is a class default and this test is wrong."
        )


def test_the_ds_joint_model_set_is_exactly_the_models_sharing_the_kappa_block():
    """The anchor groups and the kappa block should cover the same family.

    They are separate constants, so nothing forces it; a divergence means one
    recalibration reached a model the other did not, which is exactly what
    deriving-rather-than-restating exists to prevent. The kappa block is the
    narrower of the two — VG05, VG07 and VG08 predate it — so this asserts
    containment rather than equality, and names the difference.
    """
    kappa_sharers = {
        key
        for key, d in MODEL_REGISTRY.items()
        if getattr(d, "kappa_u", None) is D._DS_JOINT_UNDERSTOOD_KAPPA_RE
    }
    assert kappa_sharers, "no model shares _DS_JOINT_UNDERSTOOD_KAPPA_RE"
    assert kappa_sharers <= _DS_JOINT_MODELS, (
        f"models on the shared kappa block but not the shared anchors: "
        f"{sorted(kappa_sharers - _DS_JOINT_MODELS)}"
    )
    # The three that carry the anchors but not the kappa block, recorded so the
    # difference is deliberate rather than noticed later.
    assert sorted(_DS_JOINT_MODELS - kappa_sharers) == ["vg05", "vg07", "vg08"]


# --------------------------------------------------------------------------
# The counts the removed prose asserted
# --------------------------------------------------------------------------


def test_the_bivariate_class_tree_size():
    """`definitions.py` said "six models of record" for this tree; it holds twelve.

    The number matters because it is the refit bill for adding a field to
    `BivariateModelDefinition`, and a maintainer reading the old comment would have
    budgeted half of it.
    """
    tree = sorted(
        k for k, d in MODEL_REGISTRY.items()
        if isinstance(d, D.BivariateModelDefinition)
    )
    assert len(tree) == 12, tree
    assert tree == [
        "vg05", "vg07", "vg08", "vg09", "vg10", "vg13",
        "vg16", "vg19", "vg20", "vg21", "vg22", "vg23",
    ]

    # The other half of the same claim: the class docstring says "twelve models,
    # only eight of them direct instances", and it was the direct-instance count
    # that was mistaken for the refit bill.
    direct = sorted(
        k for k, d in MODEL_REGISTRY.items()
        if type(d) is D.BivariateModelDefinition
    )
    assert direct == [
        "vg05", "vg07", "vg08", "vg09", "vg10", "vg13", "vg16", "vg21",
    ], direct


def test_the_mean_clamp_field_is_declared_by_fourteen_of_the_twenty():
    """`clamp_targets`' docstring gives this as the refit bill for widening it.

    It was written as "fifteen", then briefly as "all twenty" -- which is the reach
    of `report_max_age_understood` (the test below), not of this field. The six
    univariate models do not declare it, which is why `common_univariate_re` reads
    it through `getattr`.
    """
    declaring = sorted(
        k for k, d in MODEL_REGISTRY.items()
        if "clamp_mean_above_hi_anchor" in {f.name for f in dataclasses.fields(d)}
    )
    assert declaring == [
        "vg05", "vg07", "vg08", "vg09", "vg10", "vg13", "vg14",
        "vg15", "vg16", "vg19", "vg20", "vg21", "vg22", "vg23",
    ], declaring
    assert len(declaring) == 14, len(declaring)
    # Stated in the docstring as a rule over classes, so check that shape too.
    assert {type(MODEL_REGISTRY[k]).__name__ for k in declaring} == {
        "BivariateModelDefinition",
        "BivariateChildSlopeModelDefinition",
        "BivariateCorrelatedSubjectREModelDefinition",
        "BivariateFactorSubjectREModelDefinition",
        "TrivariateModelDefinition",
        "JointModelDefinition",
    }


def test_every_registered_class_declares_the_comprehension_cap_field():
    """The cap comment said giving `q` its own field would invalidate "seventeen".

    It is every registered model, because every definition class declares
    `report_max_age_understood`. Stated as a rule here so the number cannot go
    stale again.
    """
    without = sorted(
        k for k, d in MODEL_REGISTRY.items()
        if "report_max_age_understood" not in {f.name for f in dataclasses.fields(d)}
    )
    assert without == [], without
    assert len(MODEL_REGISTRY) == 20, len(MODEL_REGISTRY)


def test_the_shared_kappa_block_covers_eight_definitions():
    """Two comments in `definitions.py` said "four" and then listed six."""
    sharers = sorted(
        k for k, d in MODEL_REGISTRY.items()
        if getattr(d, "kappa_u", None) is D._DS_JOINT_UNDERSTOOD_KAPPA_RE
    )
    assert sharers == ["vg09", "vg10", "vg14", "vg15", "vg16", "vg19", "vg20", "vg22"]
