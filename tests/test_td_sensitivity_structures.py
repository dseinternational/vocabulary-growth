# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The two structures #240's typically developing variants needed from the engines.

- **Per-study age slopes** (item 5): a zero-sum slope per study, in logits per
  year from the GP anchor age, entering the observation logits only and nested
  at zero, on both random-effect engines.
- **Proposal A1 on the univariate random-effect engine** (item 1): the child
  scale varies log-linearly in age with dispersion held flat. Under the variance
  partition VG11 and VG12 carry, the young-anchor scale is the partition's own, so
  the only prior the variant adds is the ratio.

Graph builds on the synthetic frame, so marked slow with the graph suite.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from vocab_growth.models.definitions import (
    VG12,
    VG21,
    AgeVaryingSubjectScale,
    SingletonMarginalisationParams,
    UnivariateMarginalisedREModelDefinition,
    _as_definition_subclass,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def sg():
    from support import synthetic_graphs

    return synthetic_graphs


def _fn(model, names):
    import pytensor

    outputs = model.replace_rvs_by_values([model[name] for name in names])
    return pytensor.function(model.value_vars, outputs, on_unused_input="ignore")


def _call(model, fn, point):
    return fn(*[point[v.name] for v in model.value_vars])


@pytest.mark.parametrize(
    ("key", "deterministic", "slopes"),
    [
        ("vg12", "f_obs", ("delta_slope_raw",)),
        ("vg21", "h_obs", ("delta_q_slope_raw",)),
    ],
)
def test_study_slopes_shift_each_row_by_its_studys_slope(sg, tmp_path, monkeypatch, key, deterministic, slopes):
    from scipy.special import logit

    from vocab_growth.models.catalogue import get

    entry = get(key)
    definition = dataclasses.replace(entry.definition, study_age_slope_sigma=0.5)
    built = sg.build_synthetic_model(
        definition, entry.engine, output_dir=str(tmp_path), monkeypatch=monkeypatch
    )
    model = built.model
    frame = built.analysis_df
    reference = definition.gp_anchor_age_months
    years = (frame["age"].to_numpy() - reference) / 12.0
    study = frame["study_code"].to_numpy()

    if key == "vg12":
        outputs = ["p_obs"]
    else:
        outputs = ["q_obs"]
    fn = _fn(model, outputs)
    point = model.initial_point()
    (base,) = _call(model, fn, point)

    raw_name = slopes[0]
    scale_name = "tau_slope" if key == "vg12" else "tau_q_slope"
    # The ZeroSumNormal is sampled through a transform, so its value variable
    # carries a suffix; move it in that space.
    (value_name,) = [name for name in point if name.startswith(raw_name)]
    raw = np.array(point[value_name], dtype=float)
    raw = raw + np.linspace(-0.3, 0.3, raw.size).reshape(raw.shape)
    moved = dict(point, **{value_name: raw})
    (after,) = _call(model, fn, moved)

    # The ZeroSumNormal's value space is K - 1 dimensional; read the realised
    # study slopes back from the graph rather than reconstructing the transform.
    offsets_fn = _fn(model, [raw_name.removesuffix("_raw")])
    (before_offsets,) = _call(model, offsets_fn, point)
    (after_offsets,) = _call(model, offsets_fn, moved)
    np.testing.assert_allclose(after_offsets.sum(), 0.0, atol=1e-9)

    change = logit(after) - logit(base)
    expected = (after_offsets - before_offsets)[study] * years
    np.testing.assert_allclose(change, expected, atol=1e-8)
    assert scale_name in {rv.name for rv in model.free_RVs}
    # The reported population trajectory is at zero study effects.
    assert deterministic in model.named_vars


def test_study_slopes_do_not_reach_the_population_grids(sg, tmp_path, monkeypatch):
    from vocab_growth.models.catalogue import get

    entry = get("vg21")
    base = sg.build_synthetic_model(
        entry.definition, entry.engine, output_dir=str(tmp_path / "base"), monkeypatch=monkeypatch
    )
    sloped = sg.build_synthetic_model(
        dataclasses.replace(entry.definition, study_age_slope_sigma=0.5),
        entry.engine,
        output_dir=str(tmp_path / "sloped"),
        monkeypatch=monkeypatch,
    )
    fp_base = sg.graph_fingerprint(base.model)
    fp = sg.graph_fingerprint(sloped.model)
    added = [name for name in fp["deterministics"] if name not in fp_base["deterministics"]]
    assert added == ["delta_u_slope", "delta_q_slope"]
    # Everything the base reports is still there, in its order.
    assert [n for n in fp["deterministics"] if n not in added] == fp_base["deterministics"]

    # And by value: moving every study's slope moves the observation rows but
    # leaves the population grids the report reads exactly where they were.
    model = sloped.model
    grids = ["p_u_query", "q_query", "p_u_plot", "q_plot"]
    observed = ["p_u_obs", "q_obs"]
    fn = _fn(model, grids + observed)
    point = model.initial_point()
    moved = dict(point)
    for raw_name in ("delta_u_slope_raw", "delta_q_slope_raw"):
        (value_name,) = [name for name in point if name.startswith(raw_name)]
        value = np.array(point[value_name], dtype=float)
        moved[value_name] = value + np.linspace(-0.4, 0.4, value.size).reshape(value.shape)
    before = _call(model, fn, point)
    after = _call(model, fn, moved)
    for name, b, a in zip(grids, before[: len(grids)], after[: len(grids)], strict=True):
        np.testing.assert_array_equal(a, b, err_msg=name)
    for name, b, a in zip(observed, before[len(grids):], after[len(grids):], strict=True):
        assert not np.allclose(a, b), name


def _a1(anchor_ages):
    return AgeVaryingSubjectScale(anchor_ages=anchor_ages, young_sigma=1.5, log_ratio_sigma=0.5)


def test_a1_under_the_partition_adds_only_the_ratio_and_holds_kappa_flat(sg, tmp_path, monkeypatch):
    from vocab_growth.models.catalogue import get

    entry = get("vg12")
    base = sg.build_synthetic_model(
        entry.definition, entry.engine, output_dir=str(tmp_path / "base"), monkeypatch=monkeypatch
    )
    varying = dataclasses.replace(
        entry.definition, tau_subject_sigma=_a1(entry.definition.kappa.anchor_ages)
    )
    built = sg.build_synthetic_model(
        varying, entry.engine, output_dir=str(tmp_path / "a1"), monkeypatch=monkeypatch
    )
    base_rvs = sg.graph_fingerprint(base.model)["free_RVs"]
    rvs = sg.graph_fingerprint(built.model)["free_RVs"]
    assert [n for n in rvs if n not in base_rvs] == ["log_tau_subject_ratio"]
    # Held flat, the old excess is no longer a free anchor.
    assert sorted(set(base_rvs) - set(rvs)) == ["kappa_excess_old"]

    model = built.model
    fn = _fn(model, ["kappa_young", "kappa_old", "tau_subject", "tau_subject_old"])
    point = dict(model.initial_point(), log_tau_subject_ratio=np.array(0.4))
    kappa_young, kappa_old, tau_young, tau_old = _call(model, fn, point)
    assert kappa_young == pytest.approx(kappa_old)
    assert tau_old == pytest.approx(tau_young * np.exp(0.4))


def test_a1_at_a_zero_ratio_gives_each_child_the_constant_scale(sg, tmp_path, monkeypatch):
    from scipy.special import logit

    from vocab_growth.models.catalogue import get

    entry = get("vg12")
    varying = dataclasses.replace(
        entry.definition, tau_subject_sigma=_a1(entry.definition.kappa.anchor_ages)
    )
    built = sg.build_synthetic_model(
        varying, entry.engine, output_dir=str(tmp_path), monkeypatch=monkeypatch
    )
    model = built.model
    fn = _fn(model, ["p_obs", "tau_subject"])
    point = model.initial_point()
    deviates = np.linspace(-1.0, 1.0, np.asarray(point["delta_subject_raw"]).size)
    point = dict(point, delta_subject_raw=deviates, log_tau_subject_ratio=np.array(0.0))
    (at_zero, tau_young) = _call(model, fn, point)
    (no_child, _) = _call(model, fn, dict(point, delta_subject_raw=np.zeros_like(deviates)))
    # At a zero ratio every row of a child is shifted by the young-anchor scale
    # times that child's deviate, whatever the row's age: VG12's constant scale.
    subject = built.analysis_df["subject_code"].to_numpy()
    np.testing.assert_allclose(
        logit(at_zero) - logit(no_child), float(tau_young) * deviates[subject], atol=1e-8
    )
    # A non-zero ratio then rescales each row by its own age.
    (widened, _) = _call(model, fn, dict(point, log_tau_subject_ratio=np.array(0.6)))
    change = logit(widened) - logit(at_zero)
    assert not np.allclose(change, change[0])


def test_a1_refuses_singleton_marginalisation(sg, tmp_path, monkeypatch):
    from vocab_growth.models.catalogue import get

    entry = get("vg12")
    definition = _as_definition_subclass(
        dataclasses.replace(
            VG12, tau_subject_sigma=_a1(VG12.kappa.anchor_ages)
        ),
        UnivariateMarginalisedREModelDefinition,
        singleton_marginalisation=SingletonMarginalisationParams(n_nodes=20),
    )
    with pytest.raises(ValueError, match="singleton marginalisation"):
        sg.build_synthetic_model(
            definition, entry.engine, output_dir=str(tmp_path), monkeypatch=monkeypatch
        )


def test_the_non_re_bivariate_engine_refuses_study_slopes():
    from vocab_growth.models.common_bivariate import build_model_graph
    from vocab_growth.models.definitions import VG05

    with pytest.raises(ValueError, match="study_age_slope_sigma"):
        build_model_graph(None, dataclasses.replace(VG05, study_age_slope_sigma=0.5))


def test_the_registered_arms_are_one_factor(sg, tmp_path, monkeypatch):
    from vocab_growth.sensitivity.registry import build_variant

    changed = {}
    for key, name in (
        ("vg12", "study-age-slopes"),
        ("vg12", "a1-tau-age-varying"),
        ("vg21", "study-age-slopes"),
        ("vg21", "eta-q-wide"),
    ):
        (variant,) = build_variant(key, name)
        base = {"vg12": VG12, "vg21": VG21}[key]
        changed[(key, name)] = {
            f.name
            for f in dataclasses.fields(base)
            if getattr(base, f.name) != getattr(variant, f.name)
        } - {"config_name", "banner"}
    assert changed == {
        ("vg12", "study-age-slopes"): {"study_age_slope_sigma"},
        ("vg12", "a1-tau-age-varying"): {"tau_subject_sigma"},
        ("vg21", "study-age-slopes"): {"study_age_slope_sigma"},
        ("vg21", "eta-q-wide"): {"eta_q_sigma"},
    }
