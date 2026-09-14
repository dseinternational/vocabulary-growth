# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Sex as a covariate in the reporting models (issue #324).

The design pinned here, in the order a reader of the change needs it:

- **Which models carry it.** The reporting models -- the Down syndrome models of
  record and the typically developing references, plus VG21's registered
  successor and VG24's -- and no development step.
- **How a row is coded.** Girls ``+1/2``, boys ``-1/2``, a child of unrecorded sex
  ``0``, resolved per child, with a child carrying two values refused.
- **What a graph gains.** Exactly the coefficients, entering the logits they
  name, and nothing for a model without the field. On the joint engine the
  coefficients also reach the cross-tab compositions, unlike the child effects.
- **What the frames do.** No row is dropped by the covariate; a frame without it
  is unchanged; the cross-tab rows find their child's sex in the merged view.
- **What the fit writes.** By-sex tables from paired draws.
"""

from __future__ import annotations

import dataclasses
import os

import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.pymc_utils as pymc_utils
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd
import pytest

from vocab_growth.models import sex_covariate
from vocab_growth.models.definitions import (
    _SEX_EFFECT_SIGMA,
    MODEL_REGISTRY,
    VG05,
    VG07,
    VG12,
    VG15,
    VG20,
)
from vocab_growth.models.observation_arrays import SEX_CONTRAST, sex_contrast_codes

REPORTING_MODELS = {"vg11", "vg12", "vg15", "vg20", "vg21", "vg23", "vg24", "vg25", "vg26"}


# ---------------------------------------------------------------------------
# Which models carry it
# ---------------------------------------------------------------------------


def test_exactly_the_reporting_models_carry_the_covariate():
    carrying = {
        key
        for key, definition in MODEL_REGISTRY.items()
        if sex_covariate.sex_effect_sigma(definition) is not None
    }
    assert carrying == REPORTING_MODELS
    assert {
        sex_covariate.sex_effect_sigma(MODEL_REGISTRY[key]) for key in carrying
    } == {_SEX_EFFECT_SIGMA}
    # The restriction is a sensitivity control; no registered model is restricted.
    assert not [
        key for key, d in MODEL_REGISTRY.items() if sex_covariate.sex_known_only(d)
    ]


def test_the_readers_default_to_no_sex_term():
    """A definition with neither field -- every one before #324 -- has no sex term."""
    from types import SimpleNamespace

    bare = SimpleNamespace()
    assert sex_covariate.sex_effect_sigma(bare) is None
    assert sex_covariate.sex_known_only(bare) is False
    assert sex_covariate.needs_sex_column(bare) is False


# ---------------------------------------------------------------------------
# How a row is coded
# ---------------------------------------------------------------------------


def _frame(sex, subjects=("1", "1", "2", "3")):
    return pd.DataFrame(
        {"study": ["a", "a", "b", "b"], "subject_id": list(subjects), "sex": sex}
    )


def test_the_contrast_is_half_a_unit_each_way():
    codes = sex_contrast_codes(_frame(["F", "F", "M", "F"]))
    np.testing.assert_array_equal(codes, [0.5, 0.5, -0.5, 0.5])
    assert SEX_CONTRAST == {"F": 0.5, "M": -0.5}
    assert dict(sex_covariate.SEX_LEVELS) == {"girls": 0.5, "boys": -0.5}


def test_a_child_of_unrecorded_sex_sits_at_zero_only_when_allowed():
    frame = _frame(["F", "F", None, "M"])
    np.testing.assert_array_equal(
        sex_contrast_codes(frame, allow_unknown=True), [0.5, 0.5, 0.0, -0.5]
    )
    with pytest.raises(ValueError, match="no recorded sex"):
        sex_contrast_codes(frame)


def test_sex_is_resolved_per_child():
    # A child's blank row takes the value recorded on their other row.
    frame = _frame(["F", None, None, "M"])
    np.testing.assert_array_equal(
        sex_contrast_codes(frame, allow_unknown=True), [0.5, 0.5, 0.0, -0.5]
    )
    # And that fill is what lets the restricted arm accept the same frame's
    # child 1, whose sex is known.
    np.testing.assert_array_equal(
        sex_contrast_codes(_frame(["F", None, "M", "M"])), [0.5, 0.5, -0.5, -0.5]
    )


def test_conflicting_or_unknown_codes_are_refused():
    with pytest.raises(ValueError, match="more than one sex value"):
        sex_contrast_codes(_frame(["F", "M", "M", "F"]), allow_unknown=True)
    with pytest.raises(ValueError, match="Unexpected sex codes"):
        sex_contrast_codes(_frame(["F", "F", "Male", "F"]), allow_unknown=True)
    with pytest.raises(KeyError, match="no `sex` column"):
        sex_contrast_codes(_frame(["F", "F", "M", "F"]).drop(columns="sex"))


def test_the_bivariate_engine_without_random_effects_refuses_the_field():
    from vocab_growth.models.common_bivariate import build_model_graph

    with pytest.raises(ValueError, match="does not implement"):
        build_model_graph(None, dataclasses.replace(VG05, sex_effect_sigma=0.5))


# ---------------------------------------------------------------------------
# What the fit writes
# ---------------------------------------------------------------------------


def test_the_by_sex_tables_are_paired_and_labelled(tmp_path):
    rng = np.random.default_rng(3)
    ages = np.array([24.0, 36.0, 48.0])
    logits = rng.normal(-1.0, 0.2, size=(3, 400)) + np.array([[0.0], [0.5], [1.0]])
    beta = rng.normal(0.3, 0.05, size=400)
    population = {
        level: sex_covariate.shifted(logits, beta, contrast)
        for level, contrast in sex_covariate.SEX_LEVELS
    }
    counts = {level: (p * 810).round().astype(int) for level, p in population.items()}

    table = sex_covariate.write_probability_by_sex(
        str(tmp_path), "u", ages, population, n_trials=810, ci_prob=0.89,
        predictive=counts, subject_marginal=population, max_age_months=36,
    )
    assert (tmp_path / "posterior_summary_u_by_sex.csv").is_file()
    assert list(table.columns[:2]) == ["sex", "age_months"]
    assert table["sex"].tolist() == ["girls", "girls", "boys", "boys"]  # 48 trimmed
    for column in ("Y_median", "p_population_median", "p_subject_marginal_median"):
        assert column in table.columns
    girls = table[table["sex"] == "girls"]["Ey_median"].to_numpy()
    boys = table[table["sex"] == "boys"]["Ey_median"].to_numpy()
    assert (girls > boys).all()

    # A single-outcome model's table sits beside its `posterior_summary.csv`.
    sex_covariate.write_probability_by_sex(
        str(tmp_path), "", ages, population, n_trials=810, ci_prob=0.89
    )
    assert (tmp_path / "posterior_summary_by_sex.csv").is_file()

    rates = sex_covariate.write_rate_by_sex(
        str(tmp_path), "q", ages, population, ci_prob=0.89
    )
    assert "q_median" in rates.columns and not [c for c in rates.columns if c.startswith("Ey")]

    difference = sex_covariate.difference_rows(
        ages, "understood", population["girls"], population["boys"],
        n_trials=810, ci_prob=0.89,
    )
    paired = np.median((population["girls"] - population["boys"]) * 810, axis=1)
    np.testing.assert_allclose(difference["Ey_difference_median"], paired)
    assert (difference["P_girls_gt_boys"] == 1.0).all()

    coefficients = sex_covariate.write_coefficients(
        str(tmp_path), {"beta_sex_u": beta}, ci_prob=0.89
    )
    assert coefficients.loc[0, "P_positive"] == pytest.approx(float(np.mean(beta > 0)))


def test_the_report_section_reads_the_tables_or_says_there_are_none(tmp_path, capsys):
    from vocab_growth.report_cells import render_sex_section

    render_sex_section(str(tmp_path))
    assert "carries no sex covariate" in capsys.readouterr().out

    ages = np.array([24.0, 36.0])
    beta = np.full(200, 0.3)
    logits = np.zeros((2, 200))
    population = {
        level: sex_covariate.shifted(logits, beta, contrast)
        for level, contrast in sex_covariate.SEX_LEVELS
    }
    sex_covariate.write_coefficients(str(tmp_path), {"beta_sex_u": beta}, ci_prob=0.89)
    sex_covariate.write_differences(
        str(tmp_path),
        [sex_covariate.difference_rows(
            ages, "understood", population["girls"], population["boys"],
            n_trials=810, ci_prob=0.89,
        )],
    )
    render_sex_section(str(tmp_path))
    out = capsys.readouterr().out
    assert "Girl–boy difference, understood" in out
    assert "| understood | 24 |" in out
    assert "How sex enters this model" in out
    # No manifest here, so the callout says why it cannot describe coverage.
    assert "Where this fit's frame records sex is not shown" in out


def test_the_new_child_table_sets_girls_beside_boys(tmp_path, capsys):
    """One row per age with both sexes, labelled by the single outcome, not a block each."""
    from vocab_growth.report_cells import render_sex_section

    beta = np.full(200, 0.3)
    sex_covariate.write_coefficients(str(tmp_path), {"beta_sex": beta}, ci_prob=0.89)
    logits = np.zeros((2, 200))
    population = {
        level: sex_covariate.shifted(logits, beta, contrast)
        for level, contrast in sex_covariate.SEX_LEVELS
    }
    sex_covariate.write_differences(
        str(tmp_path),
        [sex_covariate.difference_rows(
            np.array([24.0, 36.0]), "spoken", population["girls"], population["boys"],
            n_trials=810, ci_prob=0.89,
        )],
    )
    pd.DataFrame(
        {
            "sex": ["girls", "girls", "boys", "boys"],
            "age_months": [24.0, 36.0, 24.0, 36.0],
            "Y_median": [40.0, 90.0, 30.0, 80.0],
            "Y_ci_lo": [2.0, 10.0, 1.0, 8.0],
            "Y_ci_hi": [200.0, 400.0, 180.0, 380.0],
        }
    ).to_csv(tmp_path / "posterior_summary_by_sex.csv", index=False)

    render_sex_section(str(tmp_path))
    out = capsys.readouterr().out
    assert "| Outcome | Age (months) | A new girl | Nine in ten girls | A new boy | Nine in ten boys |" in out
    assert "| spoken | 24 | 40 | 2–200 | 30 | 1–180 |" in out
    assert "| spoken | 36 | 90 | 10–400 | 80 | 8–380 |" in out
    # The caption names the outcome the one coefficient is on, and no ratio.
    assert "the odds that a given word is said" in out
    assert "for a ratio" not in out


def test_word_differences_never_print_a_negative_zero():
    from vocab_growth.report_cells import _signed_words, _word_digits

    assert _signed_words(-0.3, 0) == "0"
    assert _signed_words(0.04, 1) == "0"
    assert _signed_words(0.26, 1) == "+0.3"
    assert _signed_words(-12.4, 0) == "-12"
    # A row of fractions keeps a decimal; a row reaching ten words drops it.
    assert _word_digits(0.7, 0.6, 0.05, 0.3) == 1
    assert _word_digits(92.0, 82.0, -0.3, 21.0) == 0


def _coverage_frame(rows):
    return pd.DataFrame(rows, columns=["study", "subject_id", "sex"])


def test_the_coverage_callout_separates_whole_study_from_within_study_gaps():
    from vocab_growth.report_cells import sex_coverage_sentences

    complete = _coverage_frame([("a", 1, "F"), ("a", 1, "F"), ("b", 2, "M")])
    assert sex_coverage_sentences(complete) == (
        "Every child in this fit's frame has sex recorded (1 girls, 1 boys), so the "
        "coefficients rest on all 2 children."
    )

    mixed = _coverage_frame(
        [
            ("a", 1, "F"), ("a", 2, "M"),
            ("b", 1, None), ("b", 2, None),  # a study that records none
            ("c", 1, "F"), ("c", 2, None), ("c", 2, None),  # one child missing, seen twice
        ]
    )
    text = sex_coverage_sentences(mixed)
    assert "Sex is recorded for 3 of this fit's 6 children (2 girls, 1 boys), from 2 of its 3 studies." in text
    assert "The study that records none (`b`; 2 children) leans on its study effect" in text
    assert "Sex is missing for some children only in `c` (1 of 2)" in text

    assert "no per-child sex column" in sex_coverage_sentences(complete.drop(columns="sex"))


# ---------------------------------------------------------------------------
# What a graph gains (synthetic builds; slow with the graph suite)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_graphs():
    from support import synthetic_graphs as sg

    return sg


def _values(model, names, point):
    import pytensor

    outputs = model.replace_rvs_by_values([model[name] for name in names])
    fn = pytensor.function(model.value_vars, outputs, on_unused_input="ignore")
    return fn(*[point[v.name] for v in model.value_vars])


@pytest.mark.slow
@pytest.mark.parametrize(
    ("key", "added", "shifted"),
    [
        ("vg20", ["beta_sex_u", "beta_sex_q"], {"beta_sex_u": "p_u_obs", "beta_sex_q": "q_obs"}),
        ("vg12", ["beta_sex"], {"beta_sex": "p_obs"}),
        (
            "vg15",
            ["beta_sex_u", "beta_sex_q", "beta_sex_sign"],
            {"beta_sex_u": "p_u_obs", "beta_sex_q": "q_obs", "beta_sex_sign": "r_obs"},
        ),
    ],
)
def test_the_graph_gains_the_coefficients_and_nothing_else(
    synthetic_graphs, tmp_path, monkeypatch, key, added, shifted
):
    from scipy.special import logit

    from vocab_growth.models.catalogue import get

    model_entry = get(key)
    with_sex = synthetic_graphs.build_synthetic_model(
        model_entry.definition, model_entry.engine,
        output_dir=str(tmp_path / "with"), monkeypatch=monkeypatch,
    )
    without = synthetic_graphs.build_synthetic_model(
        dataclasses.replace(model_entry.definition, sex_effect_sigma=None),
        model_entry.engine, output_dir=str(tmp_path / "without"), monkeypatch=monkeypatch,
    )
    fp = synthetic_graphs.graph_fingerprint(with_sex.model)
    ref = synthetic_graphs.graph_fingerprint(without.model)
    assert [n for n in fp["free_RVs"] if n not in ref["free_RVs"]] == added
    assert [n for n in fp["free_RVs"] if n not in added] == ref["free_RVs"]
    assert fp["deterministics"] == ref["deterministics"]
    assert fp["observed_RVs"] == ref["observed_RVs"]

    model = with_sex.model
    x_sex = model["x_sex"].get_value()
    assert set(np.unique(x_sex)) == {-0.5, 0.0, 0.5}
    point = model.initial_point()
    for coefficient, deterministic in shifted.items():
        (base,) = _values(model, [deterministic], point)
        moved = dict(point, **{coefficient: np.array(0.8)})
        (after,) = _values(model, [deterministic], moved)
        change = logit(after) - logit(base)
        np.testing.assert_allclose(change[x_sex > 0], 0.4, atol=1e-8)
        np.testing.assert_allclose(change[x_sex < 0], -0.4, atol=1e-8)
        np.testing.assert_allclose(change[x_sex == 0], 0.0, atol=1e-8)


@pytest.mark.slow
def test_the_joint_coefficients_reach_the_compositions(
    synthetic_graphs, tmp_path, monkeypatch
):
    """Unlike the child effects, which the cell likelihood is built to exclude."""
    from vocab_growth.models.catalogue import get

    entry = get("vg15")
    built = synthetic_graphs.build_synthetic_model(
        entry.definition, entry.engine, output_dir=str(tmp_path), monkeypatch=monkeypatch
    )
    model = built.model
    x_sex = model["x_sex"].get_value()
    point = model.initial_point()
    (base,) = _values(model, ["pi_cells_obs"], point)
    (after,) = _values(model, ["pi_cells_obs"], dict(point, beta_sex_q=np.array(0.8)))
    moved_rows = ~np.isclose(base, after).all(axis=1)
    assert moved_rows[x_sex != 0].all()
    assert not moved_rows[x_sex == 0].any()


# ---------------------------------------------------------------------------
# What the predictive stage draws (prior draws standing in as a posterior)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("child_effects", [True, False], ids=["child-effects", "single-admin"])
def test_the_by_sex_predictive_shares_one_new_child(tmp_path, monkeypatch, child_effects):
    import pymc as pm
    import xarray as xr
    from support import synthetic_graphs as sg

    from vocab_growth.models.common import ModelFitContext
    from vocab_growth.models.common_bivariate import (
        configure_bivariate_priors,
        posterior_summary,
        sample_posterior_predictive,
    )
    from vocab_growth.models.common_bivariate_re import build_model_re

    monkeypatch.setattr(
        pymc_utils, "model_to_graphviz", lambda model: sg._NoopDigraph(), raising=False
    )
    definition = dataclasses.replace(
        VG07,
        use_subject_re_u=child_effects,
        use_subject_re_q=child_effects,
        sex_effect_sigma=0.5,
    )
    n = 24
    ages = np.linspace(10.0, 90.0, n)
    frame = pd.DataFrame(
        {
            "age": ages,
            "understood": np.round(ages * 5.0),
            "spoken": np.round(ages * 3.0),
            "study": ["a"] * 12 + ["b"] * 12,
            "study_code": [0] * 12 + [1] * 12,
            "subject_id": np.repeat(np.arange(12), 2).astype(str),
            "subject_code": np.repeat(np.arange(12), 2),
            "sex": np.repeat([("F", "M", None)[i % 3] for i in range(12)], 2),
        }
    )
    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name="TEST_SEX", config_name="test", output_root_dir=str(tmp_path),
            ci_prob=0.89, interval_kind="eti",
        ),
        sampling=sampling.get_sampling_configuration("test"),
    )
    os.makedirs(context.reporting.output_dir, exist_ok=True)
    context.set_model_data(
        model_data.BinomialModelData(
            X_obs=ages.reshape(-1, 1), y_obs=np.zeros(n, dtype=int),
            n_trials=definition.n_trials,
        ),
        frame,
    )
    configure_bivariate_priors(context, definition)
    build_model_re(context, definition)
    with context.model:
        prior = pm.sample_prior_predictive(draws=6, random_seed=0)

    def dataset(node):
        return node if isinstance(node, xr.Dataset) else node.to_dataset()

    context.set_trace(
        xr.DataTree.from_dict(
            {
                "posterior": dataset(prior["prior"]),
                "constant_data": dataset(prior["constant_data"]),
                "observed_data": dataset(prior["observed_data"]),
            }
        )
    )
    sample_posterior_predictive(context, definition)
    posterior_summary(context)

    predictive = dataset(context.trace["posterior_predictive"])
    beta_u = dataset(context.trace["posterior"])["beta_sex_u"].values
    from scipy.special import logit

    girls = logit(predictive["p_u_query_subject_marginal_girls"].values)
    boys = logit(predictive["p_u_query_subject_marginal_boys"].values)
    # One child per draw, shifted a whole coefficient apart: the two levels are
    # paired, not two independent children.
    np.testing.assert_allclose(
        girls - boys, np.broadcast_to(beta_u[..., None], girls.shape), atol=1e-6
    )
    # And the same child as the existing new child of unrecorded sex, which sits
    # at the midpoint: a girl is that child half a coefficient up.
    unrecorded = logit(predictive["p_u_query_subject_marginal"].values)
    np.testing.assert_allclose(
        girls - unrecorded, np.broadcast_to(0.5 * beta_u[..., None], girls.shape), atol=1e-6
    )
    np.testing.assert_allclose(
        predictive["p_s_query_subject_marginal_girls"].values,
        predictive["p_u_query_subject_marginal_girls"].values
        * predictive["q_query_subject_marginal_girls"].values,
        atol=0,
    )
    for name in (
        "posterior_summary_u_by_sex.csv",
        "posterior_summary_s_by_sex.csv",
        "posterior_summary_q_by_sex.csv",
        "posterior_summary_sex_difference.csv",
        "posterior_summary_sex_effect.csv",
    ):
        assert os.path.isfile(os.path.join(context.reporting.output_dir, name)), name
    # The by-sex tables carry "new child" columns exactly when the pooled ones do:
    # without child effects they would only repeat the population columns.
    for suffix in ("u", "s"):
        pooled = pd.read_csv(
            os.path.join(context.reporting.output_dir, f"posterior_summary_{suffix}.csv")
        )
        by_sex = pd.read_csv(
            os.path.join(context.reporting.output_dir, f"posterior_summary_{suffix}_by_sex.csv")
        )
        pooled_marginal = any("subject_marginal" in column for column in pooled.columns)
        by_sex_marginal = any("subject_marginal" in column for column in by_sex.columns)
        assert pooled_marginal is child_effects
        assert by_sex_marginal is child_effects


@pytest.mark.slow
@pytest.mark.parametrize("age_varying", [False, True], ids=["constant-scale", "a1"])
def test_the_univariate_by_sex_predictive_shares_one_new_child(tmp_path, monkeypatch, age_varying):
    """VG12's engine pairs girls and boys too, on both of its new-child paths."""
    import pymc as pm
    import xarray as xr
    from scipy.special import logit
    from support import synthetic_graphs as sg

    from vocab_growth.models.catalogue import get
    from vocab_growth.models.common_univariate_re import (
        posterior_summary_re,
        sample_posterior_predictive_re,
    )
    from vocab_growth.models.definitions import AgeVaryingSubjectScale

    entry = get("vg12")
    definition = entry.definition
    if age_varying:
        definition = dataclasses.replace(
            definition,
            tau_subject_sigma=AgeVaryingSubjectScale(
                anchor_ages=definition.kappa.anchor_ages, young_sigma=1.5, log_ratio_sigma=0.5
            ),
        )
    context = sg.build_synthetic_model(
        definition, entry.engine, output_dir=str(tmp_path), monkeypatch=monkeypatch
    )
    with context.model:
        prior = pm.sample_prior_predictive(draws=6, random_seed=0)

    def dataset(node):
        return node if isinstance(node, xr.Dataset) else node.to_dataset()

    context.set_trace(
        xr.DataTree.from_dict(
            {
                "posterior": dataset(prior["prior"]),
                "constant_data": dataset(prior["constant_data"]),
                "observed_data": dataset(prior["observed_data"]),
            }
        )
    )
    # The A1 path draws one standard deviate per new child and scales it by age;
    # the engine selects it on the age-varying scale's query deterministic.
    assert ("tau_subject_query" in context.model_variables) is age_varying
    sample_posterior_predictive_re(context, definition)
    predictive = dataset(context.trace["posterior_predictive"])
    beta = dataset(context.trace["posterior"])["beta_sex"].values
    girls = logit(predictive["p_query_subject_marginal_girls"].values)
    boys = logit(predictive["p_query_subject_marginal_boys"].values)
    np.testing.assert_allclose(
        girls - boys, np.broadcast_to(beta[..., None], girls.shape), atol=1e-6
    )
    posterior_summary_re(context, definition)
    by_sex = pd.read_csv(os.path.join(context.reporting.output_dir, "posterior_summary_by_sex.csv"))
    assert {"Y_median", "Y_ci_lo", "Y_ci_hi"} <= set(by_sex.columns)
    assert set(by_sex["sex"]) == {"girls", "boys"}


# ---------------------------------------------------------------------------
# What the frames do (against the prepared database)
# ---------------------------------------------------------------------------


def _hash(frame):
    from vocab_growth.analysis_frames import analysis_frame_hash

    return analysis_frame_hash(frame)


@pytest.mark.parametrize("key", ["vg20", "vg12", "vg15"])
def test_the_covariate_drops_no_row_and_adds_only_its_column(require_prepared_data, key):
    from vocab_growth.analysis_frames import build_analysis_frame

    definition = MODEL_REGISTRY[key]
    frame, _ = build_analysis_frame(key, definition)
    plain, _ = build_analysis_frame(key, dataclasses.replace(definition, sex_effect_sigma=None))
    assert "sex" in frame.columns and "sex" not in plain.columns
    assert _hash(frame.drop(columns="sex")) == _hash(plain)
    assert set(frame["sex"].dropna().unique()) <= set(SEX_CONTRAST)


def test_the_restricted_arm_drops_the_studies_that_record_no_sex(require_prepared_data):
    from vocab_growth.analysis_frames import build_analysis_frame

    full, _ = build_analysis_frame("vg20", VG20)
    restricted, info = build_analysis_frame(
        "vg20", dataclasses.replace(VG20, sex_known_only=True)
    )
    assert restricted["sex"].notna().all()
    assert info["sex_unknown_rows_excluded"] > 0
    assert len(restricted) < len(full)
    # us_03 joined the studies without sex on 2026-09-06.
    assert not set(restricted["study"]) & {
        "ie_01", "it_01", "nz_01", "uk_03", "uk_04", "us_02", "us_03"
    }
    assert (restricted.groupby(["study", "subject_id"])["sex"].nunique() == 1).all()


def test_the_cross_tab_rows_take_their_childs_sex(require_prepared_data):
    from vocab_growth.analysis_frames import build_analysis_frame

    frame, _ = build_analysis_frame("vg15", VG15)
    coverage = frame.groupby("study")["sex"].apply(lambda s: s.notna().mean())
    for study in ("uk_02", "uk_07", "es_01"):
        assert coverage[study] == 1.0, study
    assert coverage["nz_01"] == 0.0


def test_the_wordbank_loader_recodes_sex_only_on_request(require_prepared_data):
    from vocab_growth import data_utils

    columns = ["age", "understood", "study", "subject_id"]
    plain = data_utils.load_data(
        data_utils.Population.TYPICALLY_DEVELOPING, columns, languages=VG12.td_languages
    )
    with_sex = data_utils.load_data(
        data_utils.Population.TYPICALLY_DEVELOPING,
        columns + ["sex"],
        languages=VG12.td_languages,
    )
    assert "sex" not in plain.columns
    pd.testing.assert_frame_equal(with_sex[columns], plain)
    assert set(with_sex["sex"].dropna().unique()) == {"F", "M"}
