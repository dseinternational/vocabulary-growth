# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check the A1 age-varying child scale and preserve the scalar-scale path.

The sensitivity variant changes the graph through an existing definition
field. Registered models must retain their scalar scales and serialised
definitions.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from vocab_growth.models.definitions import (
    MODEL_REGISTRY,
    AgeVaryingSubjectScale,
    subject_scale_spec,
    subject_slope_spec,
)
from vocab_growth.sensitivity.registry import VARIANTS, build_variant

A1_VARIANT = ("vg10", "a1-tau-age-varying")

SUBJECT_SCALE_FIELDS = ("tau_subject_sigma", "tau_subj_u_sigma", "tau_subj_q_sigma")


def test_a1_is_registered_where_a_decision_put_it():
    """Register A1 only for models with a supported child-effect structure."""
    registered = [key for key in VARIANTS if key[1] == "a1-tau-age-varying"]
    assert registered == [
        A1_VARIANT,
        ("vg11", "a1-tau-age-varying"),
        ("vg12", "a1-tau-age-varying"),
        ("vg21", "a1-tau-age-varying"),
    ]


def test_no_model_of_record_carries_an_age_varying_scale():
    """Keep A1 as a sensitivity structure rather than a registered default.

    A1 multiplies one child deviate by a positive age-varying scale, so ranks
    of child offsets cannot change with age. A random intercept and slope can
    allow crossing. These are different structures, not interchangeable priors.
    """
    offenders = []
    for key, definition in MODEL_REGISTRY.items():
        for field in SUBJECT_SCALE_FIELDS:
            value = getattr(definition, field, None)
            if value is None:
                continue
            if subject_scale_spec(value) is not None:
                offenders.append(f"{key}.{field}")
            elif subject_slope_spec(value) is not None:
                continue
            else:
                assert isinstance(value, float), f"{key}.{field} is {value!r}"
    assert not offenders, f"models of record carrying A1: {offenders}"


def test_variant_keeps_the_record_prior_at_the_young_anchor():
    """Keep the scale prior unchanged at the young anchor.

    Changing ``young_sigma`` would alter both the age dependence and the
    starting scale prior.
    """
    base = MODEL_REGISTRY["vg10"]
    variant = build_variant(*A1_VARIANT)[0]
    for field in ("tau_subj_u_sigma", "tau_subj_q_sigma"):
        spec = subject_scale_spec(getattr(variant, field))
        assert spec is not None
        assert spec.young_sigma == getattr(base, field)


def test_variant_anchors_match_the_paired_kappa_blocks():
    """Use the same reference ages for child scale and count concentration.

    This makes their age dependence comparable in the sensitivity analysis.
    """
    variant = build_variant(*A1_VARIANT)[0]
    pairs = (("tau_subj_u_sigma", "kappa_u"), ("tau_subj_q_sigma", "kappa_s"))
    for scale_field, kappa_field in pairs:
        spec = subject_scale_spec(getattr(variant, scale_field))
        assert spec.anchor_ages == getattr(variant, kappa_field).anchor_ages


def test_variant_holds_both_kappa_blocks_flat():
    """Move age variation from concentration to the child-effect scale."""
    variant = build_variant(*A1_VARIANT)[0]
    for field in ("tau_subj_u_sigma", "tau_subj_q_sigma"):
        assert subject_scale_spec(getattr(variant, field)).hold_kappa_constant


def test_subject_scale_spec_ignores_scalars():
    """Recognise the structured scale and leave scalar scales unchanged."""
    assert subject_scale_spec(1.5) is None
    assert subject_scale_spec(0.0) is None
    spec = AgeVaryingSubjectScale(
        anchor_ages=(24.0, 48.0), young_sigma=1.5, log_ratio_sigma=0.5
    )
    assert subject_scale_spec(spec) is spec


def test_scale_closure_is_the_record_at_ratio_zero():
    """``log_ratio = 0`` must reproduce a constant scale exactly."""
    z_young, z_old = -1.0, 0.5
    tau_young = 1.3

    def tau_of_z(z, log_ratio):
        return tau_young * np.exp(log_ratio * (z - z_young) / (z_old - z_young))

    grid = np.linspace(-2.0, 2.0, 9)
    assert np.allclose(tau_of_z(grid, 0.0), tau_young)
    # The ratio at the old anchor is exp(log_ratio).
    assert np.isclose(tau_of_z(z_old, 0.4) / tau_young, np.exp(0.4))


def test_flat_kappa_needs_the_two_anchor_form():
    """Reject a constant-concentration request for the legacy parameterisation."""
    from vocab_growth.models.common import build_kappa_for_config

    legacy = dataclasses.replace(
        MODEL_REGISTRY["vg10"], config_name="probe-legacy"
    )
    config = type("Cfg", (), {"kappa_anchored_u": None})()
    with pytest.raises(ValueError, match="two-anchor kappa form"):
        build_kappa_for_config(
            config, X_obs_mean=30.0, X_obs_std=10.0, suffix="_u", hold_constant=True
        )
    del legacy
