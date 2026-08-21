# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Proposal A1: the age-varying subject scale, and what it must not disturb.

A1 is a *graph* change carried on an existing definition field, exactly as
``CLAMP_Q_ONLY`` is. That buys the fifteen models of record their fingerprints,
and costs a standing obligation: the scalar path must stay untouched. These
tests pin both halves — that the variant does what it claims, and that nothing
else in the family can see it.
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


def test_a1_is_registered_on_vg10_alone():
    """A1 is a diagnostic on one model. A second registration is a decision."""
    registered = [key for key in VARIANTS if key[1] == "a1-tau-age-varying"]
    assert registered == [A1_VARIANT]


def test_no_model_of_record_carries_an_age_varying_scale():
    """The guard that keeps A1 out of the published models.

    If a subject-scale field in the registry ever becomes an
    ``AgeVaryingSubjectScale``, a model of record has silently adopted a
    structure with a measured-false rank-correlation assumption behind it: A1
    scales ONE per-child deviate by ``tau(age)``, which forces children never to
    cross, and the disattenuated rank correlation is about 0.75-0.83 out to two
    years and 0.28 beyond.

    A :class:`SubjectSlopePriorParams` is admissible where A1 is not, and the
    reason is exactly that assumption. A child slope draws ``(b0, b1)`` from a
    2x2 joint with ``rho01`` **estimated**, so A1 is its special case at
    ``rho01 = 1`` -- the slope frees the constraint rather than imposing it.
    Freeing it costs 6.28 on 1 df on the repeats-only production fit, which is
    the measurement that made VG19 a slope rather than a scaled deviate.
    Registered 2026-08-21; see notes/202608141900-child-slope-implementation-plan.md.
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
    """One factor: only the *constancy* of the scale changes, not its prior.

    ``young_sigma`` must equal the scalar it replaces. If it drifts, the variant
    conflates "the scale varies with age" with "the scale has a different prior",
    and its result attributes to A1 something A1 did not do.
    """
    base = MODEL_REGISTRY["vg10"]
    variant = build_variant(*A1_VARIANT)[0]
    for field in ("tau_subj_u_sigma", "tau_subj_q_sigma"):
        spec = subject_scale_spec(getattr(variant, field))
        assert spec is not None
        assert spec.young_sigma == getattr(base, field)


def test_variant_anchors_match_the_paired_kappa_blocks():
    """`tau` and `kappa` must contest the same span, or neither answers the other.

    The whole diagnostic is "how much of kappa's decline is misattributed
    widening". That is only a well-posed question if the two parameters vary over
    identical reference ages.
    """
    variant = build_variant(*A1_VARIANT)[0]
    pairs = (("tau_subj_u_sigma", "kappa_u"), ("tau_subj_q_sigma", "kappa_s"))
    for scale_field, kappa_field in pairs:
        spec = subject_scale_spec(getattr(variant, scale_field))
        assert spec.anchor_ages == getattr(variant, kappa_field).anchor_ages


def test_variant_holds_both_kappa_blocks_flat():
    """A1 *moves* the age variation; it does not add a second copy of it."""
    variant = build_variant(*A1_VARIANT)[0]
    for field in ("tau_subj_u_sigma", "tau_subj_q_sigma"):
        assert subject_scale_spec(getattr(variant, field)).hold_kappa_constant


def test_subject_scale_spec_ignores_scalars():
    """The overloaded field's single interpreter, pinned against the float path."""
    assert subject_scale_spec(1.5) is None
    assert subject_scale_spec(0.0) is None
    spec = AgeVaryingSubjectScale(
        anchor_ages=(24.0, 48.0), young_sigma=1.5, log_ratio_sigma=0.5
    )
    assert subject_scale_spec(spec) is spec


def test_scale_closure_is_the_record_at_ratio_zero():
    """``log_ratio = 0`` must reproduce a constant scale exactly.

    Nesting is what makes the posterior for one parameter an answer rather than a
    model comparison, so it is worth checking arithmetically rather than trusting
    the algebra in the docstring.
    """
    z_young, z_old = -1.0, 0.5
    tau_young = 1.3

    def tau_of_z(z, log_ratio):
        return tau_young * np.exp(log_ratio * (z - z_young) / (z_old - z_young))

    grid = np.linspace(-2.0, 2.0, 9)
    assert np.allclose(tau_of_z(grid, 0.0), tau_young)
    # And at the old anchor the ratio is exactly exp(log_ratio), by construction.
    assert np.isclose(tau_of_z(z_old, 0.4) / tau_young, np.exp(0.4))


def test_flat_kappa_needs_the_two_anchor_form():
    """A legacy-kappa model must refuse A1 rather than silently ignore it.

    Silently ignoring would produce a variant that believes it has switched off
    the dispersion trajectory while still fitting one — a failure that looks
    exactly like a pass.
    """
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
