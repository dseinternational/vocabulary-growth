# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The Gompertz-mean experiment harness (#330): its mean, its graph and its output root.

``scripts/experiments/gompertz_mean_arm.py`` swaps a registered engine's
trend-plus-HSGP mean for a free-asymptote Gompertz and claims to hold everything
else fixed. Three things have to be true for its comparison to mean anything:

* the Gompertz it builds is the Gompertz -- right values, increasing, bounded by
  its asymptote, and evaluated at ages in months rather than standardised ages;
* the experiment graph differs from the registered one **only** in the
  mean-function nodes, checked on the named free random variables, the
  deterministics and the likelihood terms of each model it supports;
* it cannot write into a canonical ``models/`` directory.

The graphs are built on the small synthetic frame of ``tests/support`` (about a
second each), so nothing here needs the prepared database.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from support.synthetic_graphs import build_synthetic_model, synthetic_frame

SCRIPT = Path(__file__).parents[1] / "scripts" / "experiments" / "gompertz_mean_arm.py"


@pytest.fixture(scope="module")
def gm():
    spec = importlib.util.spec_from_file_location("gompertz_mean_arm_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The mean function
# ---------------------------------------------------------------------------

#: (A, k_g, T_i): Day et al.'s typically-developing production values on the
#: 810-item scale, our free-asymptote Down syndrome comprehension fit, and a
#: steep curve that saturates inside the age range.
PARAMETER_SETS = [(680 / 810, 0.161, 23.3), (0.72, 0.043, 37.1), (0.5, 0.3, 10.0)]


@pytest.mark.parametrize(("a_share", "k_g", "t_i"), PARAMETER_SETS)
def test_share_at_known_ages(gm, a_share, k_g, t_i):
    # At T_i the inner exponential is 1, so the curve is at A/e -- the property
    # the parameter-free test of notes/202609101230 §3 rests on.
    assert gm.gompertz_share(t_i, a_share, k_g, t_i) == pytest.approx(a_share / np.e)
    # One growth time later the inner exponential is 1/e.
    assert gm.gompertz_share(t_i + 1 / k_g, a_share, k_g, t_i) == pytest.approx(
        a_share * np.exp(-np.exp(-1.0))
    )
    # ln 2 / k_g earlier it is 2.
    assert gm.gompertz_share(t_i - np.log(2) / k_g, a_share, k_g, t_i) == pytest.approx(
        a_share * np.exp(-2.0)
    )


def test_day_et_al_250_words_at_t_i(gm):
    # The reconstruction check in notes/202609101230 §1: with A = 680 words the
    # curve passes 250 words at T_i, because 680 / e = 250.2.
    words = 810 * gm.gompertz_share(23.3, 680 / 810, 0.161, 23.3)
    assert words == pytest.approx(250.2, abs=0.05)


@pytest.mark.parametrize(("a_share", "k_g", "t_i"), PARAMETER_SETS)
def test_increasing_and_bounded_by_the_asymptote(gm, a_share, k_g, t_i):
    ages = np.linspace(0.0, 240.0, 4001)
    share = gm.gompertz_share(ages, a_share, k_g, t_i)
    logit = gm.gompertz_logit(ages, a_share, k_g, t_i)
    # Non-decreasing up to rounding once the curve has saturated.
    assert np.all(np.diff(share) >= -1e-15)
    assert np.all(np.diff(logit) >= -1e-12)
    # Strictly increasing wherever floating point can still see it move: the
    # logit approaches logit(A), a constant, so far past T_i the steps vanish.
    resolvable = ages < t_i + 20.0 / k_g
    assert np.all(np.diff(logit[resolvable]) > 0.0)
    assert np.all(share > 0.0)
    assert np.all(share <= a_share)


@pytest.mark.parametrize(("a_share", "k_g", "t_i"), PARAMETER_SETS)
def test_asymptote(gm, a_share, k_g, t_i):
    late = t_i + 60.0 / k_g
    assert gm.gompertz_share(late, a_share, k_g, t_i) == pytest.approx(a_share, rel=1e-12)
    assert gm.gompertz_logit(late, a_share, k_g, t_i) == pytest.approx(
        np.log(a_share / (1 - a_share)), rel=1e-9
    )
    assert gm.gompertz_share(t_i - 5.0 / k_g, a_share, k_g, t_i) < 1e-60


def test_logit_is_the_logit_of_the_share_and_finite_in_the_tails(gm):
    ages = np.array([6.0, 12.0, 23.3, 36.0, 60.0])
    share = gm.gompertz_share(ages, 0.84, 0.161, 23.3)
    np.testing.assert_allclose(
        gm.gompertz_logit(ages, 0.84, 0.161, 23.3), np.log(share / (1 - share)), rtol=1e-10
    )
    # A share that underflows to zero in probability is still a finite logit,
    # so the likelihood's clip sees a number rather than -inf.
    deep = gm.gompertz_logit(np.array([-200.0]), 0.7, 0.08, 60.0)
    assert np.isfinite(deep).all() and deep[0] < -1e9


def test_pytensor_form_matches_numpy(gm):
    import pytensor.tensor as pt

    ages = np.linspace(8.0, 115.0, 50)
    graph = gm.gompertz_logit_pt(pt.as_tensor_variable(ages), 0.72, 0.043, 37.1)
    np.testing.assert_allclose(graph.eval(), gm.gompertz_logit(ages, 0.72, 0.043, 37.1), rtol=1e-12)


def test_age_scale_is_recovered_exactly_from_the_anchors(gm):
    class Grid:
        sa_z = (24.0 - 36.4) / 19.7
        sb_z = (84.0 - 36.4) / 19.7

    mean, std = gm.age_scale_from_anchors((24, 84), Grid)
    assert mean == pytest.approx(36.4, rel=1e-12)
    assert std == pytest.approx(19.7, rel=1e-12)


# ---------------------------------------------------------------------------
# The graph differs from the registered one only in the mean function
# ---------------------------------------------------------------------------

#: The registered mean's stored deterministics, before the outcome suffix: the
#: trend's slope and intercept, the length scale, the HSGP's unit draw on the
#: grid and its scaled form.
MEAN_DETERMINISTICS = ("slope", "intercept", "ell", "g_unit", "g")


def _registered_mean_free_rvs(names, suffix, gm):
    exact = {f"{prefix}{suffix}" for prefix in gm.MEAN_FUNCTION_PREFIXES}
    return {n for n in names if n in exact or n.startswith(f"g_unit{suffix}_")}


def _build(gm, curve, arm, ratio_mean, tmp_path, monkeypatch):
    from vocab_growth.models.catalogue import engine_for_definition

    definition = gm.arm_definition(curve, arm, ratio_mean)
    engine = engine_for_definition(definition)
    with gm.arm_context(definition, curve, arm, ratio_mean) as calls:
        context = build_synthetic_model(
            definition, engine, output_dir=str(tmp_path / arm), monkeypatch=monkeypatch
        )
    return definition, context, dict(calls)


CASES = [("vg12", "gompertz"), ("vg11", "gompertz"), ("vg20", "gompertz"), ("vg20", "flexible")]


@pytest.mark.parametrize(("curve", "ratio_mean"), CASES)
def test_experiment_graph_differs_only_in_the_mean_function(gm, curve, ratio_mean, tmp_path, monkeypatch):
    _, registered_ctx, _ = _build(gm, curve, "baseline", ratio_mean, tmp_path, monkeypatch)
    _, experiment_ctx, calls = _build(gm, curve, "gompertz", ratio_mean, tmp_path, monkeypatch)
    registered, experiment = registered_ctx.model, experiment_ctx.model

    slots = gm.mean_slots(curve, ratio_mean)
    swapped = [suffix for suffix, slot in slots.items() if slot is not None]
    kept = [suffix for suffix, slot in slots.items() if slot is None]
    # Every mean the engine builds went through the patch exactly once.
    assert calls == dict.fromkeys(slots, 1)

    reg_free = [rv.name for rv in registered.free_RVs]
    exp_free = [rv.name for rv in experiment.free_RVs]
    removed = set(reg_free) - set(exp_free)
    added = set(exp_free) - set(reg_free)

    expected_removed = set().union(*(_registered_mean_free_rvs(reg_free, s, gm) for s in swapped))
    assert expected_removed, "the registered graph should carry a trend + HSGP mean to remove"
    assert removed == expected_removed
    assert added == {f"{p}{s}" for p in gm.GOMPERTZ_PARAMETERS for s in swapped}
    # A kept ratio mean keeps its trend + HSGP parameters.
    for suffix in kept:
        assert _registered_mean_free_rvs(exp_free, suffix, gm) == _registered_mean_free_rvs(reg_free, suffix, gm)

    # Every other free variable is the same, in the same order and with the same dims.
    shared_reg = [n for n in reg_free if n not in removed]
    shared_exp = [n for n in exp_free if n not in added]
    assert shared_reg == shared_exp
    for name in shared_reg:
        assert registered.named_vars_to_dims.get(name) == experiment.named_vars_to_dims.get(name), name
        assert registered[name].shape.eval().tolist() == experiment[name].shape.eval().tolist(), name

    # The likelihood terms are the same variables.
    assert [rv.name for rv in registered.observed_RVs] == [rv.name for rv in experiment.observed_RVs]

    # Deterministics: only the mean's own disappear, nothing new appears.
    reg_det = {d.name for d in registered.deterministics}
    exp_det = {d.name for d in experiment.deterministics}
    assert exp_det <= reg_det
    assert reg_det - exp_det == {f"{p}{s}" for p in MEAN_DETERMINISTICS for s in swapped} & reg_det


def test_experiment_latent_is_the_gompertz_at_ages_in_months(gm, tmp_path, monkeypatch):
    from vocab_growth.models.build_utils import standardize_ages

    definition, context, _ = _build(gm, "vg12", "gompertz", "gompertz", tmp_path, monkeypatch)
    model = context.model
    point = model.initial_point()
    point.update({"k_g_log__": np.log(0.13), "T_i": 17.0, "A_share_logodds__": np.log(0.65 / 0.35)})
    # Evaluated at the value variables, not by drawing the random variables.
    (latent,) = model.replace_rvs_by_values([model["f_all"]])
    f_all = model.compile_fn(latent, inputs=model.value_vars, on_unused_input="ignore")(point)

    frame = synthetic_frame(definition)
    mean, std = standardize_ages(frame["age"].to_numpy().reshape(-1, 1))[:2]
    ages = np.asarray(model["X_all_z"].get_value())[:, 0] * std + mean
    np.testing.assert_allclose(f_all, gm.gompertz_logit(ages, 0.65, 0.13, 17.0), rtol=1e-9)


def test_patch_is_removed_after_the_build(gm, tmp_path, monkeypatch):
    from vocab_growth.models import common_bivariate_re, gp_utils

    _build(gm, "vg20", "gompertz", "gompertz", tmp_path, monkeypatch)
    assert common_bivariate_re.trend_and_gp is gp_utils.trend_and_gp


# ---------------------------------------------------------------------------
# Output root
# ---------------------------------------------------------------------------


def test_experiment_root_sits_beside_models_of_the_canonical_root(gm, tmp_path):
    canonical = tmp_path / "shared"
    root = gm.resolve_experiment_root(str(canonical), [str(canonical)])
    assert Path(root) == canonical / "experiments" / "gompertz-mean"


@pytest.mark.parametrize("inside", ["models", "models/VG20-age-understood-spoken", "models/experiments"])
def test_refuses_a_destination_inside_canonical_models(gm, tmp_path, inside):
    canonical = tmp_path / "shared"
    with pytest.raises(ValueError, match="canonical models directory"):
        gm.resolve_experiment_root(str(canonical / inside), [str(tmp_path / "elsewhere"), str(canonical)])


def test_a_sibling_whose_name_starts_with_models_is_not_inside_it(gm, tmp_path):
    canonical = tmp_path / "shared"
    root = gm.resolve_experiment_root(str(canonical / "models-scratch"), [str(canonical)])
    assert Path(root) == canonical / "models-scratch" / "experiments" / "gompertz-mean"


def test_install_refuses_the_configured_shared_root(gm, tmp_path, monkeypatch):
    from vocab_growth import environment as env

    shared = tmp_path / "shared"
    monkeypatch.setenv(env.OUTPUT_DIR_ENV_VAR, str(shared))
    with pytest.raises(SystemExit, match="canonical models directory"):
        gm.install_output_root(str(shared / "models"))
    # Refused before the override is set, so nothing downstream resolves there.
    assert Path(env.models_output_dir()) == shared / "models"


def test_install_sets_the_experiment_root(gm, tmp_path, monkeypatch):
    from vocab_growth import environment as env

    shared = tmp_path / "shared"
    monkeypatch.setenv(env.OUTPUT_DIR_ENV_VAR, str(shared))
    try:
        root = gm.install_output_root(None)
        assert Path(root) == shared / "experiments" / "gompertz-mean"
        assert Path(env.models_output_dir()) == shared / "experiments" / "gompertz-mean" / "models"
    finally:
        env.set_output_root(None)
