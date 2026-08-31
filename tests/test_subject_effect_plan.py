# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One resolution of a definition's child-effect structure (issue #273).

Five structures can occupy the same seam, three of them through a scalar field
that holds an object instead. Until this, "what child structure does this model
have?" was answered by four selector calls, two ``getattr`` reads and five
rejection rules interleaved with graph construction inside the PyMC context --
so a refusal fired part-way through a half-built model, and the rules could only
be tested by building one.

``subject_effects.resolve`` is a pure function of the definition, so everything
here runs without PyMC and without data: the resolution for every registered
model, and every combination the engines refuse.

That the resolution did not change any model's graph is the separate claim, and
``tests/test_graph_equivalence.py`` is where it is checked.
"""

from __future__ import annotations

import pytest

from vocab_growth.models.definitions import (
    MODEL_REGISTRY,
    VG10,
    VG13,
    VG15,
    VG19,
    VG20,
    VG22,
    AgeVaryingSubjectScale,
    BivariateCorrelatedSubjectREModelDefinition,
    BivariateFactorSubjectREModelDefinition,
    SubjectFactorPriorParams,
    SubjectSlopePriorParams,
    _as_definition_subclass,
)
from vocab_growth.models.subject_effects import (
    SubjectEffectKind,
    resolve,
)

_MODEL_KEYS = sorted(MODEL_REGISTRY)

#: What each registered model resolves to: ``{outcome: kind}``, then the joint
#: structures. Written out rather than derived, because a table computed from
#: the same code it checks would agree with any answer.
EXPECTED = {
    "vg01": ({"": "none"}, None, None, False),
    "vg02": ({"": "none"}, None, None, False),
    "vg03": ({"": "none"}, None, None, False),
    "vg04": ({"": "none"}, None, None, False),
    "vg05": ({"u": "none", "q": "none"}, None, None, False),
    "vg07": ({"u": "none", "q": "none"}, None, None, False),
    # VG08 gives a child effect to comprehension only.
    "vg08": ({"u": "constant", "q": "none"}, None, None, False),
    "vg09": ({"u": "constant", "q": "constant"}, None, None, False),
    "vg10": ({"u": "constant", "q": "constant"}, None, None, False),
    # VG11 and VG12 reparameterise the child scale into a shared budget, so the
    # HalfNormal their definitions still carry never enters the model.
    "vg11": ({"": "variance_partition"}, None, None, True),
    "vg12": ({"": "variance_partition"}, None, None, True),
    "vg13": ({"u": "constant", "q": "constant"}, None, None, False),
    # VG14 has no child-effect seam at all.
    "vg14": ({}, None, None, False),
    # VG15's third block is signing. A plan that assumed two outcomes dropped it.
    "vg15": ({"u": "constant", "q": "constant", "sign": "constant"}, None, None, False),
    "vg16": ({"u": "constant", "q": "constant"}, None, None, False),
    "vg19": ({"u": "child_slope", "q": "child_slope"}, None, None, False),
    "vg20": ({"u": "constant", "q": "constant"}, 2.0, None, False),
    "vg21": ({"u": "constant", "q": "constant"}, None, None, False),
    "vg22": ({"u": "factor", "q": "factor"}, None, 3, False),
    "vg23": ({"u": "constant", "q": "constant"}, 2.0, None, False),
}


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_every_registered_model_resolves_as_expected(model_key):
    kinds, eta, rank, partitioned = EXPECTED[model_key]
    plan = resolve(MODEL_REGISTRY[model_key])
    assert {e.outcome: e.kind.value for e in plan.effects} == kinds
    assert plan.correlation_eta == eta
    assert (plan.factor.rank if plan.factor else None) == rank
    assert (plan.variance_partition is not None) is partitioned


def test_the_expectations_cover_the_registry():
    assert sorted(EXPECTED) == _MODEL_KEYS


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_the_emitted_scale_names_are_the_ones_consumers_read(model_key):
    """Every downstream reader indexes the trace by these, so they are a contract."""
    plan = resolve(MODEL_REGISTRY[model_key])
    for effect in plan.effects:
        expected = "tau_subject" if effect.outcome == "" else f"tau_subj_{effect.outcome}"
        assert effect.scale_name == expected


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_resolution_is_deterministic_and_the_plan_is_immutable(model_key):
    import dataclasses

    definition = MODEL_REGISTRY[model_key]
    assert resolve(definition) == resolve(definition)
    plan = resolve(definition)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.correlation_eta = 1.0


def test_indexing_an_absent_outcome_names_what_the_plan_covers():
    plan = resolve(VG10)
    with pytest.raises(KeyError, match="sign"):
        plan["sign"]


def test_an_inactive_outcome_is_present_rather_than_omitted():
    """A consumer indexes rather than searches, so VG08's q must have an entry."""
    plan = resolve(MODEL_REGISTRY["vg08"])
    assert plan["q"].kind is SubjectEffectKind.NONE
    assert not plan["q"].is_active
    assert plan["u"].is_active
    assert plan.any_active


# --- the refusals, each against a definition that really carries it ------------


@pytest.mark.parametrize("drop", ["u", "q"])
def test_a_correlation_needs_both_blocks(drop):
    definition = _as_definition_subclass(
        VG20,
        BivariateCorrelatedSubjectREModelDefinition,
        model_id="VGXX",
        **{f"use_subject_re_{drop}": False},
    )
    with pytest.raises(ValueError, match="requires use_subject_re_u"):
        resolve(definition)


@pytest.mark.parametrize("side", ["u", "q"])
def test_a_correlation_refuses_an_age_varying_scale(side):
    definition = _as_definition_subclass(
        VG20,
        BivariateCorrelatedSubjectREModelDefinition,
        model_id="VGXX",
        **{
            f"tau_subj_{side}_sigma": AgeVaryingSubjectScale(
                young_sigma=1.5, log_ratio_sigma=0.5, anchor_ages=(24.0, 60.0)
            )
        },
    )
    with pytest.raises(ValueError, match="age-varying"):
        resolve(definition)


@pytest.mark.parametrize("side", ["u", "q"])
def test_a_correlation_refuses_a_child_slope(side):
    definition = _as_definition_subclass(
        VG20,
        BivariateCorrelatedSubjectREModelDefinition,
        model_id="VGXX",
        **{
            f"tau_subj_{side}_sigma": SubjectSlopePriorParams(
                tau0_sigma=1.5, tau1_sigma=0.5, rho_eta=2.0
            )
        },
    )
    with pytest.raises(ValueError, match="child slope"):
        resolve(definition)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), "2"])
def test_a_correlation_refuses_a_non_positive_eta(bad):
    definition = _as_definition_subclass(
        VG10,
        BivariateCorrelatedSubjectREModelDefinition,
        model_id="VGXX",
        subject_re_correlation_eta=bad,
    )
    with pytest.raises(ValueError, match="positive finite"):
        resolve(definition)


@pytest.mark.parametrize("drop", ["u", "q"])
def test_a_factor_needs_both_blocks(drop):
    definition = _as_definition_subclass(
        VG22,
        BivariateFactorSubjectREModelDefinition,
        model_id="VGXX",
        **{f"use_subject_re_{drop}": False},
    )
    with pytest.raises(ValueError, match="subject_factor requires"):
        resolve(definition)


@pytest.mark.parametrize("side", ["u", "q"])
def test_a_factor_refuses_an_age_varying_scale(side):
    definition = _as_definition_subclass(
        VG22,
        BivariateFactorSubjectREModelDefinition,
        model_id="VGXX",
        **{
            f"tau_subj_{side}_sigma": AgeVaryingSubjectScale(
                young_sigma=1.5, log_ratio_sigma=0.5, anchor_ages=(24.0, 60.0)
            )
        },
    )
    with pytest.raises(ValueError, match="age-varying"):
        resolve(definition)


@pytest.mark.parametrize("side", ["u", "q"])
def test_a_factor_refuses_a_child_slope(side):
    definition = _as_definition_subclass(
        VG22,
        BivariateFactorSubjectREModelDefinition,
        model_id="VGXX",
        **{
            f"tau_subj_{side}_sigma": SubjectSlopePriorParams(
                tau0_sigma=1.5, tau1_sigma=0.5, rho_eta=2.0
            )
        },
    )
    with pytest.raises(ValueError, match="child slope"):
        resolve(definition)


def test_a_variance_partition_needs_a_child_effect():
    import dataclasses

    from vocab_growth.models.definitions import (
        VG11,
        UnivariateREModelDefinition,
    )

    definition = dataclasses.replace(VG11, use_subject_re=False)
    assert isinstance(definition, UnivariateREModelDefinition)
    with pytest.raises(ValueError, match="nothing to allocate"):
        resolve(definition)


def test_an_overloaded_scale_field_refuses_an_unrecognised_object():
    """The field holds a float, an A1 scale or a child slope, and nothing else."""
    import dataclasses

    definition = dataclasses.replace(VG10, tau_subj_u_sigma=object())
    with pytest.raises(ValueError, match="must be a number"):
        resolve(definition)


# --- the combinations that are refused, but only when actually reachable -------


def test_the_factor_refuses_a_correlation_it_would_double_count():
    """VG22 and VG20 both claim the level-level correlation."""
    definition = _as_definition_subclass(
        VG22,
        BivariateFactorSubjectREModelDefinition,
        model_id="VGXX",
    )
    # The two fields live on sibling subclasses, so a definition carrying both
    # has to be constructed rather than registered; the rule exists because
    # nothing in the type system prevents it.
    object.__setattr__(definition, "subject_re_correlation_eta", 2.0)
    with pytest.raises(ValueError, match="cannot be combined with"):
        resolve(definition)


def test_a_third_outcome_keeps_its_own_kind_under_a_factor():
    """The factor spans u and q; a signing block is not one of its rows."""
    definition = _as_definition_subclass(
        VG15,
        type(VG15),
        model_id="VGXX",
    )
    object.__setattr__(
        definition,
        "subject_factor",
        SubjectFactorPriorParams(rank=2, tau1_u_sigma=0.5, tau1_q_sigma=0.5),
    )
    plan = resolve(definition)
    assert plan["u"].kind is SubjectEffectKind.FACTOR
    assert plan["q"].kind is SubjectEffectKind.FACTOR
    assert plan["sign"].kind is SubjectEffectKind.CONSTANT


def test_the_child_slope_reference_age_comes_from_the_definition():
    assert resolve(VG19).slope_ref_age_months == VG19.subject_slope_ref_age_months
    # A definition with no such field falls back to the documented default.
    assert resolve(VG13).slope_ref_age_months == 36.0
