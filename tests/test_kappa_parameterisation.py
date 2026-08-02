# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Config-level contract for the two kappa parameterisations.

``tests/test_gp_utils.py`` pins the anchored builder's algebra. This module
covers the layer above it: that a model configuration carries exactly one
dispersion form, that the engines dispatch on it correctly, and that the
property motivating the change actually holds — a prior stated at an age means
the same thing whatever the pool's age distribution turns out to be.
"""

import math
import types

import numpy as np
import preliz as pz
import pymc as pm
import pytest

from vocab_growth.models import common
from vocab_growth.models.common import (
    AnchoredKappaPriors,
    ModelConfiguration,
    build_kappa_for_config,
    kappa_prior_rows,
)
from vocab_growth.models.common_bivariate import BivariateModelConfiguration
from vocab_growth.models.definitions import (
    VG01,
    VG02,
    VG03,
    VG09,
    VG10,
    VG11,
    VG13,
    VG16,
    KappaAnchorPriorParams,
    KappaPriorParams,
)

_ANCHORED = AnchoredKappaPriors(
    kappa_min_dist=pz.LogNormal(mu=math.log(3.0), sigma=0.8),
    excess_young_dist=pz.LogNormal(mu=math.log(30.0), sigma=0.7),
    excess_old_dist=pz.LogNormal(mu=math.log(3.0), sigma=0.7),
    anchor_ages=(12.0, 20.0),
)


def _config(**kappa_fields) -> ModelConfiguration:
    return ModelConfiguration(
        slope_anchors=(12, 26),
        ell_months_range=(6, 18),
        p_slope_low_dist=pz.Beta(alpha=1, beta=30),
        p_slope_hi_dist=pz.Beta(alpha=1.3, beta=1.3),
        ell_unit_dist=pz.Beta(alpha=2, beta=2),
        eta_dist=pz.HalfNormal(sigma=0.5),
        n_plot=10,
        ages_query=[12, 24],
        **kappa_fields,
    )


_LEGACY_FIELDS = {
    "kappa_min_dist": pz.LogNormal(mu=math.log(3.0), sigma=0.8),
    "a_kappa_dist": pz.Normal(mu=math.log(8.0), sigma=0.75),
    "b_kappa_mag_dist": pz.HalfNormal(sigma=0.75),
}


# --- the configuration carries exactly one form ------------------------------


def test_config_accepts_the_legacy_triple():
    assert _config(**_LEGACY_FIELDS).kappa_anchored is None


def test_config_accepts_the_anchored_form():
    assert _config(kappa_anchored=_ANCHORED).kappa_min_dist is None


def test_config_rejects_a_partial_legacy_triple():
    partial = dict(_LEGACY_FIELDS)
    del partial["b_kappa_mag_dist"]

    with pytest.raises(ValueError, match="needs all of"):
        _config(**partial)


def test_config_rejects_mixing_the_two_forms():
    with pytest.raises(ValueError, match="cannot be combined"):
        _config(kappa_anchored=_ANCHORED, **_LEGACY_FIELDS)


def test_config_rejects_no_kappa_at_all():
    with pytest.raises(ValueError, match="needs all of"):
        _config()


# --- the engines dispatch on it ----------------------------------------------


def _free_rvs(config):
    with pm.Model() as model:
        build_kappa_for_config(config, X_obs_mean=19.6, X_obs_std=5.9)
    return {v.name for v in model.free_RVs}


def test_dispatch_builds_the_legacy_variables():
    assert _free_rvs(_config(**_LEGACY_FIELDS)) == {
        "kappa_min", "a_kappa", "b_kappa_mag"
    }


def test_dispatch_builds_the_anchored_variables():
    assert _free_rvs(_config(kappa_anchored=_ANCHORED)) == {
        "kappa_min", "kappa_excess_young", "kappa_excess_old"
    }


@pytest.mark.parametrize("mean,sd", [(19.6, 5.9), (33.0, 15.0)])
def test_dispatch_places_the_anchors_at_their_ages(mean, sd):
    """The ages, not the z positions, are what the definition states.

    Point-mass priors make the identity exact: whatever the standardisation,
    kappa at 12 and 20 months must come out as the floor plus the excess drawn
    for that anchor.
    """
    pinned = AnchoredKappaPriors(
        kappa_min_dist=pz.LogNormal(mu=math.log(3.0), sigma=1e-9),
        excess_young_dist=pz.LogNormal(mu=math.log(30.0), sigma=1e-9),
        excess_old_dist=pz.LogNormal(mu=math.log(3.0), sigma=1e-9),
        anchor_ages=(12.0, 20.0),
    )
    with pm.Model():
        kappa_of_z = build_kappa_for_config(
            _config(kappa_anchored=pinned), X_obs_mean=mean, X_obs_std=sd
        )
        at_12 = float(kappa_of_z((12.0 - mean) / sd).eval())
        at_20 = float(kappa_of_z((20.0 - mean) / sd).eval())

    assert np.isclose(at_12, 33.0)
    assert np.isclose(at_20, 6.0)


def test_prior_rows_name_the_form_in_use():
    legacy = [name for name, _ in kappa_prior_rows(_config(**_LEGACY_FIELDS))]
    anchored = [name for name, _ in kappa_prior_rows(_config(kappa_anchored=_ANCHORED))]

    assert legacy == ["kappa_min", "a_kappa", "b_kappa_mag"]
    # the age is part of what the prior means, so it appears in the label
    assert anchored == [
        "kappa_min", "kappa_excess_young (12 mo)", "kappa_excess_old (20 mo)"
    ]


def test_derived_rows_are_empty_for_the_legacy_form():
    rows = common.kappa_anchor_derived_rows(
        _config(**_LEGACY_FIELDS), X_obs_mean=19.6, X_obs_std=5.9
    )

    assert rows == []


def test_derived_rows_report_the_anchor_z_positions():
    rows = dict(
        common.kappa_anchor_derived_rows(
            _config(kappa_anchored=_ANCHORED), X_obs_mean=20.0, X_obs_std=5.0
        )
    )

    assert rows["Kappa anchors (months)"] == (12.0, 20.0)
    assert np.allclose(rows["Kappa anchors (z-score)"], (-1.6, 0.0))


# --- configure_univariate_priors picks the form up from the definition -------


def _configure(definition, monkeypatch):
    monkeypatch.setattr(common, "_plot_and_print_dist", lambda *a, **k: None)
    captured = []
    context = types.SimpleNamespace(set_model_config=captured.append)
    common.configure_univariate_priors(context, definition)
    (config,) = captured
    return config


def test_configure_gives_vg01_the_anchored_form(monkeypatch):
    config = _configure(VG01, monkeypatch)

    assert config.kappa_anchored is not None
    assert config.kappa_anchored.anchor_ages == VG01.kappa.anchor_ages


def test_configure_leaves_vg02_on_the_legacy_form(monkeypatch):
    config = _configure(VG02, monkeypatch)

    assert config.kappa_anchored is None
    assert config.b_kappa_mag_dist is not None


# --- the property that motivates the change ----------------------------------


_POOLS = ((19.6, 5.9), (33.0, 15.0))  # two deliberately different age distributions
_AGES = np.array([8.0, 12.0, 18.0, 24.0, 30.0])


def _kappa_curve_through_the_builder(kappa_fields, mean, sd):
    """kappa at `_AGES` under a pool's standardisation, via the real dispatch."""
    with pm.Model():
        kappa_of_z = build_kappa_for_config(
            _config(**kappa_fields), X_obs_mean=mean, X_obs_std=sd
        )
        return np.asarray(kappa_of_z((_AGES - mean) / sd).eval())


def _pinned(definition):
    """The model's own anchor medians as point masses, so draws are deterministic."""
    kp = definition.kappa
    return {
        "kappa_anchored": AnchoredKappaPriors(
            kappa_min_dist=pz.LogNormal(mu=kp.kappa_min_mu, sigma=1e-9),
            excess_young_dist=pz.LogNormal(mu=kp.excess_young_mu, sigma=1e-9),
            excess_old_dist=pz.LogNormal(mu=kp.excess_old_mu, sigma=1e-9),
            anchor_ages=kp.anchor_ages,
        )
    }


@pytest.mark.parametrize("definition", [VG01, VG03, VG11], ids=["VG01", "VG03", "VG11"])
def test_anchored_prior_at_an_age_is_free_of_the_pool_standardisation(definition):
    """The whole point: resampling the pool must not move what the prior says.

    ``a_kappa`` is the age term at ``z = 0``, i.e. at the pool's mean age, so a
    subsample or a study filter silently redefines it. Under the two-anchor form
    the interpolation weight is ``(age - young) / (old - young)`` in *months* —
    the standardisation cancels — so kappa at any age is identical under any
    standardisation. Driven through ``build_kappa_for_config`` rather than
    recomputed here, so it tests the graph the engines actually emit.
    """
    curves = [
        _kappa_curve_through_the_builder(_pinned(definition), mean, sd)
        for mean, sd in _POOLS
    ]

    # Exact in real arithmetic; in float64 the round trip through z leaves ~1e-9
    # relative error, amplified a little where an age sits outside the anchors.
    np.testing.assert_allclose(curves[0], curves[1], rtol=1e-7)


def test_legacy_prior_at_an_age_does_depend_on_the_standardisation():
    # The contrast the test above is asserting against: a_kappa is pinned at
    # z = 0, so the same prior describes a different age in a different pool.
    # b_kappa_mag is pinned with a LogNormal rather than the production
    # HalfNormal only because HalfNormal cannot be concentrated away from zero,
    # and a slope of zero would make both curves flat and the test vacuous.
    pinned_legacy = {
        "kappa_min_dist": pz.LogNormal(mu=math.log(3.0), sigma=1e-9),
        "a_kappa_dist": pz.Normal(mu=math.log(8.0), sigma=1e-9),
        "b_kappa_mag_dist": pz.LogNormal(mu=math.log(1.5), sigma=1e-9),
    }
    curves = [
        _kappa_curve_through_the_builder(pinned_legacy, mean, sd) for mean, sd in _POOLS
    ]

    assert not np.allclose(curves[0], curves[1])


def test_the_two_kappa_prior_classes_stay_distinct():
    assert not isinstance(VG01.kappa, KappaPriorParams)
    assert isinstance(VG01.kappa, KappaAnchorPriorParams)
    assert isinstance(VG02.kappa, KappaPriorParams)


# --- the joint engines carry one form per outcome -----------------------------


def _bivariate_config(**kappa_fields) -> BivariateModelConfiguration:
    return BivariateModelConfiguration(
        slope_anchors=(10, 16),
        ell_months_range=(6, 18),
        p_slope_low_u_dist=pz.Beta(alpha=1, beta=15),
        p_slope_hi_u_dist=pz.Beta(alpha=2, beta=6),
        ell_unit_u_dist=pz.Beta(alpha=3, beta=3),
        eta_u_dist=pz.HalfNormal(sigma=0.4),
        p_slope_low_q_dist=pz.Beta(alpha=1, beta=10),
        p_slope_hi_q_dist=pz.Beta(alpha=2, beta=7),
        ell_unit_q_dist=pz.Beta(alpha=3, beta=3),
        eta_q_dist=pz.HalfNormal(sigma=0.2),
        n_plot=10,
        ages_query=[12, 17],
        **kappa_fields,
    )


_LEGACY_U = {
    "kappa_min_u_dist": pz.LogNormal(mu=math.log(5.0), sigma=0.6),
    "a_kappa_u_dist": pz.Normal(mu=math.log(8.0), sigma=1.0),
    "b_kappa_mag_u_dist": pz.HalfNormal(sigma=0.3),
}
_LEGACY_S = {
    "kappa_min_s_dist": pz.LogNormal(mu=math.log(5.0), sigma=0.6),
    "a_kappa_s_dist": pz.Normal(mu=math.log(8.0), sigma=1.0),
    "b_kappa_mag_s_dist": pz.HalfNormal(sigma=0.3),
}


def test_a_joint_model_may_anchor_one_outcome_and_not_the_other():
    """VG13 anchors both and the DS joint models anchor neither, but nothing in
    the configuration ties the two outcomes together — so the mixed case has to
    work rather than merely not be exercised."""
    config = _bivariate_config(kappa_anchored_u=_ANCHORED, **_LEGACY_S)

    with pm.Model() as model:
        build_kappa_for_config(
            config, X_obs_mean=13.0, X_obs_std=2.8, suffix="_u"
        )
        build_kappa_for_config(
            config, X_obs_mean=13.0, X_obs_std=2.8, suffix="_s"
        )

    assert {v.name for v in model.free_RVs} == {
        "kappa_min_u", "kappa_excess_young_u", "kappa_excess_old_u",
        "kappa_min_s", "a_kappa_s", "b_kappa_mag_s",
    }


def test_a_joint_outcome_cannot_mix_the_two_forms():
    with pytest.raises(ValueError, match="kappa_anchored_u cannot be combined"):
        _bivariate_config(kappa_anchored_u=_ANCHORED, **_LEGACY_U, **_LEGACY_S)


def test_a_joint_outcome_cannot_be_half_specified():
    partial = dict(_LEGACY_S)
    del partial["b_kappa_mag_s_dist"]

    with pytest.raises(ValueError, match="kappa_s form needs all of"):
        _bivariate_config(kappa_anchored_u=_ANCHORED, **partial)


@pytest.mark.parametrize("mean,sd", [(13.0, 2.8), (28.0, 12.0)])
def test_joint_anchors_land_at_their_ages_whatever_the_pool(mean, sd):
    """Same invariance as the single-outcome case, through the suffixed path."""
    pinned = AnchoredKappaPriors(
        kappa_min_dist=pz.LogNormal(mu=math.log(3.0), sigma=1e-9),
        excess_young_dist=pz.LogNormal(mu=math.log(30.0), sigma=1e-9),
        excess_old_dist=pz.LogNormal(mu=math.log(3.0), sigma=1e-9),
        anchor_ages=(12.0, 20.0),
    )
    config = _bivariate_config(kappa_anchored_u=pinned, **_LEGACY_S)

    with pm.Model():
        kappa_of_z = build_kappa_for_config(
            config, X_obs_mean=mean, X_obs_std=sd, suffix="_u"
        )
        at_12 = float(kappa_of_z((12.0 - mean) / sd).eval())
        at_20 = float(kappa_of_z((20.0 - mean) / sd).eval())

    assert np.isclose(at_12, 33.0)
    assert np.isclose(at_20, 6.0)


def test_derived_rows_are_labelled_per_outcome():
    config = _bivariate_config(kappa_anchored_u=_ANCHORED, **_LEGACY_S)

    rows = dict(
        common.kappa_anchor_derived_rows(
            config, X_obs_mean=20.0, X_obs_std=5.0, suffix="_u"
        )
    )
    assert rows["Kappa anchors_u (months)"] == (12.0, 20.0)
    # the legacy outcome contributes nothing to the table
    assert common.kappa_anchor_derived_rows(
        config, X_obs_mean=20.0, X_obs_std=5.0, suffix="_s"
    ) == []


def test_vg13_anchors_both_outcomes_and_the_ds_joint_models_anchor_neither():
    assert isinstance(VG13.kappa_u, KappaAnchorPriorParams)
    assert isinstance(VG13.kappa_s, KappaAnchorPriorParams)
    # The DS joint frame is too thin for a stable conditional estimate (see
    # scripts/kappa_conditional_calibration.py --mean-sweep), so VG09/VG10/VG16
    # deliberately keep the legacy priors.
    for definition in (VG09, VG10, VG16):
        assert isinstance(definition.kappa_u, KappaPriorParams), definition.model_id
        assert isinstance(definition.kappa_s, KappaPriorParams), definition.model_id
