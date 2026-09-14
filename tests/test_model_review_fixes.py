# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check the model behaviours identified in the September code review."""

from dataclasses import replace

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
from pytensor.graph.replace import graph_replace
from scipy.special import logit
from support import synthetic_graphs

from vocab_growth.models.catalogue import get
from vocab_growth.models.common_bivariate import _child_slope_offsets
from vocab_growth.models.definitions import VG19
from vocab_growth.models.study_effects import zero_sum_study_offsets

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("reference_age", [0.0, 24.0, 36.0])
def test_fitted_and_predictive_child_slopes_use_the_same_age(
    reference_age, tmp_path, monkeypatch
):
    original = synthetic_graphs.synthetic_frame

    def frame(definition, n_rows=48):
        data = original(definition, n_rows)
        data.loc[0:1, "age"] = [24.0, 36.0]
        return data

    monkeypatch.setattr(synthetic_graphs, "synthetic_frame", frame)
    definition = replace(
        VG19, subject_slope_ref_age_months=reference_age, ages_query=(24, 36), n_plot=12
    )
    context = synthetic_graphs.build_synthetic_model(
        definition,
        get("vg19").engine,
        output_dir=str(tmp_path),
        monkeypatch=monkeypatch,
    )
    model = context.model
    children = len(model.coords["subject_id"])
    probability = graph_replace(
        model["p_u_obs"],
        {
            model["f_u_all"]: pt.constant(np.zeros(len(model.coords["all_id"]))),
            model["delta_u"]: pt.constant(np.zeros(len(model.coords["study_id"]))),
            model["b0_tau_subj_u"]: pt.constant(np.zeros(children)),
            model["b1_tau_subj_u"]: pt.constant(np.ones(children)),
        },
    ).eval()[:2]
    _, predictive_years = _child_slope_offsets(context, definition)
    expected = (np.array([24.0, 36.0]) - reference_age) / 12
    np.testing.assert_allclose(logit(probability), expected, atol=1e-12)
    np.testing.assert_allclose(predictive_years.eval(), expected, atol=1e-12)


@pytest.mark.parametrize(
    "key, centred", [("vg20", False), ("vg12", True), ("vg12", False)]
)
def test_a_single_study_builds_with_zero_study_offset(
    key, centred, tmp_path, monkeypatch
):
    original = synthetic_graphs.synthetic_frame

    def frame(definition, n_rows=48):
        data = original(definition, n_rows)
        data["study"] = "only_study"
        data["study_code"] = 0
        return data

    monkeypatch.setattr(synthetic_graphs, "synthetic_frame", frame)
    record = get(key)
    settings = {"n_plot": 12}
    if key == "vg12":
        settings["centred_study_re"] = centred
    context = synthetic_graphs.build_synthetic_model(
        replace(record.definition, **settings),
        record.engine,
        output_dir=str(tmp_path),
        monkeypatch=monkeypatch,
    )
    model = context.model
    names = ("delta_u", "delta_q") if key == "vg20" else ("delta",)
    for name in names:
        np.testing.assert_array_equal(model[name].eval(), [0.0])
    assert np.isfinite(model.compile_logp()(model.initial_point()))


@pytest.mark.parametrize("fallback", ["product_marginal", "paired_only"])
def test_bivariate_reference_still_includes_studies_without_one_outcome(
    fallback, tmp_path, monkeypatch
):
    original = synthetic_graphs.synthetic_frame

    def frame(definition, n_rows=48):
        data = original(definition, n_rows)
        data.loc[data.study_code == 3, "spoken"] = np.nan
        data.loc[data.study_code == 2, "understood"] = np.nan
        return data

    monkeypatch.setattr(synthetic_graphs, "synthetic_frame", frame)
    record = get("vg20")
    context = synthetic_graphs.build_synthetic_model(
        replace(record.definition, spoken_fallback=fallback, n_plot=12),
        record.engine,
        output_dir=str(tmp_path),
        monkeypatch=monkeypatch,
    )
    draws = pm.draw(context.model["delta_q"], draws=8, random_seed=12)
    assert draws.shape == (8, 4)
    np.testing.assert_allclose(draws.sum(axis=1), 0.0, atol=1e-12)
    assert np.any(draws[:, 3] != 0)


def test_outcome_specific_reference_sets_uninformed_studies_to_zero():
    with pm.Model(coords={"study_id": range(4)}):
        offsets = zero_sum_study_offsets(
            "delta", scale=1.0, n_studies=4, informed=np.array([0, 2])
        )
    draws = pm.draw(offsets, draws=8, random_seed=12)
    np.testing.assert_array_equal(draws[:, [1, 3]], 0.0)
    np.testing.assert_allclose(draws[:, [0, 2]].sum(axis=1), 0.0, atol=1e-12)
