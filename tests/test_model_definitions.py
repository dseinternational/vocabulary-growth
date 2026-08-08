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


# Which outcomes carry the two-anchor form, and why. Anchored priors are stated
# as a dispersion at a named age for a named quantity, so one picked up by the
# wrong pool is a prior for something else entirely — the split is asserted here
# rather than left to review. A model reaches this set only once an empirical
# calibration exists for its own outcome and frame:
#
#   VG01-VG04                 marginal calibration (no grouping to condition on)
#   VG11, VG12, VG13          conditional calibration (study + subject intercepts)
#   VG09, VG10, VG15, VG16    conditional calibration on the DS joint frame, which
#                             yields a lower bound rather than a point estimate
#
# The four Down syndrome joint models are the ones carrying subject intercepts on
# *both* outcomes, so they share one calibration target. VG15 anchors only its
# understood and spoken outcomes; its signed ratio has no calibration and stays
# on the legacy form, which is also the mixed case no other registered model
# exercises.
#
# Everything else stays on the legacy form, and each for its own reason.
# VG05 carries no random effects and VG07 only study ones, so neither can use the
# conditional numbers below; VG08 has a subject effect on understood but not on
# `q`, so it would need one of each. All three are lineage steps in the
# VG05 -> VG07 -> VG08 -> VG09 -> VG10 sequence, where changing a prior partway
# would confound the contrast the sequence exists to show. That is a deliberate
# exclusion, and it is why their `b_kappa_mag_s` sitting four standard deviations
# beyond its prior is disclosed rather than fixed -- they supply no reported
# number (see docs/models/README.md, model roles).
#
# VG14 was excluded on the stated grounds that its "frame is the signing subset,
# not this one". That was wrong: VG14, VG15 and VG10 all fit the same 1,349-row
# frame, and VG15 already uses these exact prior objects. VG14 was migrated on
# 2026-08-06 -- see notes/202608051500-report-critical-review.md section 4a. The
# real blocker was never the frame, it was that `common_trivariate` accepted only
# the legacy form until the same date.
#
# VG17/VG18 have had neither calibration.
_ANCHORED_OUTCOMES = {
    "vg01": {"kappa"},
    "vg02": {"kappa"},
    "vg03": {"kappa"},
    "vg04": {"kappa"},
    "vg09": {"kappa_u", "kappa_s"},
    "vg10": {"kappa_u", "kappa_s"},
    "vg11": {"kappa"},
    "vg12": {"kappa"},
    "vg13": {"kappa_u", "kappa_s"},
    "vg14": {"kappa_u", "kappa_s"},
    "vg15": {"kappa_u", "kappa_s"},
    "vg16": {"kappa_u", "kappa_s"},
}


@pytest.mark.parametrize("model_id", sorted(_ANCHORED_OUTCOMES), ids=str)
def test_calibrated_models_use_the_two_anchor_kappa_form(model_id):
    expected = _ANCHORED_OUTCOMES[model_id]
    anchored = {
        name
        for name, kappa in _kappa_priors(MODEL_REGISTRY[model_id])
        if isinstance(kappa, KappaAnchorPriorParams)
    }

    assert anchored == expected


@pytest.mark.parametrize(
    "model_id", [k for k in MODEL_REGISTRY if k not in _ANCHORED_OUTCOMES], ids=str
)
def test_uncalibrated_models_keep_the_legacy_kappa_form(model_id):
    for name, kappa in _kappa_priors(MODEL_REGISTRY[model_id]):
        assert isinstance(kappa, KappaPriorParams), f"{model_id}.{name}"
