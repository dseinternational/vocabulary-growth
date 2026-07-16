# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the prior-sensitivity override + registry tooling (issue #89).

These are pure/fast (no data, no sampling): they pin that a variant applies the
requested overrides, isolates its output via a suffixed ``config_name``, and
never mutates the committed model definitions (including the nested kappa
priors, which must be fresh objects rather than aliases of the base's).
"""

import dataclasses

import pytest

from vocab_growth.models.definitions import VG10, VG11, VG12, VG13, VG15
from vocab_growth.sensitivity.overrides import make_variant, replace_kappa
from vocab_growth.sensitivity.registry import VARIANTS, build_variant, variants_for


def test_make_variant_suffixes_config_name_and_leaves_base_untouched():
    base_low = (VG10.p_slope_low_q_alpha, VG10.p_slope_low_q_beta)
    v = make_variant(VG10, config_suffix="q-broad", scalar_over={
        "p_slope_low_q_alpha": 1.0, "p_slope_low_q_beta": 1.5})
    assert v.config_name == f"{VG10.config_name}-q-broad"
    assert v.model_id == VG10.model_id
    assert (v.p_slope_low_q_alpha, v.p_slope_low_q_beta) == (1.0, 1.5)
    # Base instance is untouched.
    assert (VG10.p_slope_low_q_alpha, VG10.p_slope_low_q_beta) == base_low
    assert VG10.config_name == "age-understood-spoken-ds-re-subj-uq-anchored"


def test_make_variant_nested_kappa_is_fresh_not_aliased():
    base_sigma = VG10.kappa_s.kappa_min_sigma
    v = make_variant(VG10, config_suffix="kappa-broadfloor", kappa_over={
        "kappa_u": {"kappa_min_sigma": 1.0}, "kappa_s": {"kappa_min_sigma": 1.0}})
    assert v.kappa_s.kappa_min_sigma == 1.0
    assert v.kappa_u.kappa_min_sigma == 1.0
    # The base's kappa objects are neither mutated nor shared with the variant.
    assert VG10.kappa_s.kappa_min_sigma == base_sigma
    assert v.kappa_s is not VG10.kappa_s
    assert v.kappa_u is not VG10.kappa_u


def test_make_variant_rejects_bad_input():
    with pytest.raises(ValueError):
        make_variant(VG10, config_suffix="")  # empty suffix
    with pytest.raises(TypeError):
        make_variant(VG10, config_suffix="x", scalar_over={"not_a_field": 1.0})
    with pytest.raises(ValueError):
        make_variant(VG10, config_suffix="x", kappa_over={"kappa_u": {"nope": 1.0}})
    with pytest.raises(ValueError):
        make_variant(VG10, config_suffix="x", kappa_over={"no_such_kappa": {"a_kappa_mu": 0.0}})


def test_replace_kappa_overrides_only_named_fields():
    kp = VG15.kappa_sign
    new = replace_kappa(kp, kappa_min_sigma=1.0)
    assert new.kappa_min_sigma == 1.0
    assert new.a_kappa_mu == kp.a_kappa_mu  # untouched
    assert new is not kp


def test_registry_counts_and_models():
    # 27 §7 targets + 7 Target-8 young-age anchor variants (#146), two
    # source-harmonisation variants, and three repeated-measures sensitivities.
    assert len(VARIANTS) == 39
    assert len(variants_for("vg10")) == 10
    assert len(variants_for("vg11")) == 5
    assert len(variants_for("vg12")) == 4
    assert len(variants_for("vg13")) == 1
    assert len(variants_for("vg15")) == 19


def test_td_models_account_for_repeated_children_by_default():
    assert VG11.use_subject_re
    assert VG12.use_subject_re
    assert VG13.use_subject_re_u
    assert VG13.use_subject_re_q

    (single_vg13,) = build_variant("vg13", "single-admin")
    assert single_vg13.one_observation_per_subject
    assert not single_vg13.use_subject_re_u
    assert not single_vg13.use_subject_re_q


def test_build_variant_all_and_named():
    all_vg15 = build_variant("vg15", "all")
    assert len(all_vg15) == 19
    # All distinct config_names, all still VG15.
    assert len({d.config_name for d in all_vg15}) == 19
    assert all(d.model_id == "VG15" for d in all_vg15)
    # psi-neutral applies both hyperparameters.
    (psi,) = build_variant("vg15", "psi-neutral")
    assert (psi.log_psi_mu, psi.log_psi_sigma) == (0.0, 0.5)


def test_build_variant_rejects_unknown():
    with pytest.raises(KeyError):
        build_variant("vg99", "q-broad")
    with pytest.raises(KeyError):
        build_variant("vg10", "no-such-variant")


def test_variants_are_single_factor_or_documented_pairs():
    # Every variant produces a definition whose model_type matches its base
    # (sanity that replace preserved the class), and changes at least one field.
    for (model_key, name) in VARIANTS:
        (v,) = build_variant(model_key, name)
        base = {
            "vg10": VG10,
            "vg11": VG11,
            "vg12": VG12,
            "vg13": VG13,
            "vg15": VG15,
        }[model_key]
        assert v.model_type == base.model_type
        assert dataclasses.asdict(v) != dataclasses.asdict(base)
