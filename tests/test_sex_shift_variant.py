# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The exploratory sex-shift variant of VG20 (issue #295).

``BivariateSexShiftModelDefinition`` is unregistered and driven only by
``scripts/experiments/vg20_sex_arm.py``, but the seam it opens runs through the
engine that fits eleven registered models. Two properties carry the design and
are pinned here:

- **The seam is inert unless the field is set.** A definition of the subclass
  with neither field set must build VG20's graph exactly -- the same free
  variables in the same order, the same deterministics, the same fixed-point log
  probability -- and no registered model may gain either field.
- **When set, it adds exactly two coefficients and the contrast that multiplies
  them**, with the stated sign: girls ``+1/2``, boys ``-1/2``, and the shift
  entering both logits.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pandas as pd
import pytest

from vocab_growth.models import definitions as D
from vocab_growth.models.definitions import (
    MODEL_REGISTRY,
    VG20,
    BivariateSexShiftModelDefinition,
    _as_definition_subclass,
)
from vocab_growth.models.observation_arrays import SEX_CONTRAST, sex_contrast_codes

SEX_FIELDS = {"sex_known_only", "sex_effect_sigma"}


def test_no_registered_model_carries_the_sex_fields():
    """Putting either field on a registered class would invalidate its fits."""
    for key, definition in MODEL_REGISTRY.items():
        names = {f.name for f in dataclasses.fields(definition)}
        assert not (names & SEX_FIELDS), key
        assert not isinstance(definition, BivariateSexShiftModelDefinition), key


def test_the_inert_variant_is_vg20_plus_two_defaults():
    variant = _as_definition_subclass(VG20, BivariateSexShiftModelDefinition)
    assert isinstance(variant, D.BivariateCorrelatedSubjectREModelDefinition)
    for f in dataclasses.fields(VG20):
        assert getattr(variant, f.name) == getattr(VG20, f.name), f.name
    assert variant.sex_known_only is False
    assert variant.sex_effect_sigma is None
    assert {f.name for f in dataclasses.fields(variant)} - {f.name for f in dataclasses.fields(VG20)} == SEX_FIELDS
    # An optional scale at `None` is its off state, not a scale that must be
    # positive: the inert and control arms must validate.
    D.validate_model_definition(variant)
    D.validate_model_definition(
        _as_definition_subclass(VG20, BivariateSexShiftModelDefinition, sex_known_only=True)
    )


def test_a_coefficient_without_the_restriction_is_refused_at_definition_time():
    with pytest.raises(ValueError, match="needs sex_known_only"):
        _as_definition_subclass(VG20, BivariateSexShiftModelDefinition, sex_effect_sigma=0.5)


def _frame(sex):
    return pd.DataFrame(
        {
            "study": ["a", "a", "b", "b"],
            "subject_id": ["1", "1", "2", "3"],
            "sex": sex,
        }
    )


def test_sex_contrast_is_half_a_unit_each_way():
    codes = sex_contrast_codes(_frame(["F", "F", "M", "F"]))
    np.testing.assert_array_equal(codes, [0.5, 0.5, -0.5, 0.5])
    assert SEX_CONTRAST == {"F": 0.5, "M": -0.5}


def test_sex_contrast_refuses_missing_unknown_or_inconsistent_values():
    with pytest.raises(ValueError, match="no recorded sex"):
        sex_contrast_codes(_frame(["F", None, "M", "F"]))
    with pytest.raises(ValueError, match="Unexpected sex codes"):
        sex_contrast_codes(_frame(["F", "F", "boy", "F"]))
    with pytest.raises(ValueError, match="more than one sex value"):
        sex_contrast_codes(_frame(["F", "M", "M", "F"]))
    with pytest.raises(KeyError, match="no `sex` column"):
        sex_contrast_codes(_frame(["F", "F", "M", "F"]).drop(columns="sex"))


# ---------------------------------------------------------------------------
# Graph checks: one synthetic build each, so marked slow with the graph suite.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_graphs():
    from support import synthetic_graphs as sg

    return sg


@pytest.fixture(scope="module")
def vg20_engine():
    from vocab_growth.models.catalogue import get

    return get("vg20").engine


@pytest.mark.slow
def test_the_inert_variant_builds_vg20_graph_exactly(
    synthetic_graphs, vg20_engine, tmp_path, monkeypatch
):
    reference = synthetic_graphs.build_synthetic_model(
        VG20, vg20_engine, output_dir=str(tmp_path / "vg20"), monkeypatch=monkeypatch
    )
    variant = _as_definition_subclass(
        VG20, BivariateSexShiftModelDefinition, config_name="sex-inert"
    )
    built = synthetic_graphs.build_synthetic_model(
        variant, vg20_engine, output_dir=str(tmp_path / "inert"), monkeypatch=monkeypatch
    )
    assert synthetic_graphs.graph_fingerprint(built.model) == synthetic_graphs.graph_fingerprint(
        reference.model
    )
    assert synthetic_graphs.fixed_point_logp(built.model) == pytest.approx(
        synthetic_graphs.fixed_point_logp(reference.model), rel=1e-9
    )


def _build_effect_arm(synthetic_graphs, engine, output_dir, monkeypatch, **overrides):
    import dse_research_utils.statistics.models.data as model_data
    import dse_research_utils.statistics.models.pymc_utils as pymc_utils
    from dse_research_utils.statistics.models import reporting, sampling

    from vocab_growth.models.common import ModelFitContext

    monkeypatch.setattr(
        pymc_utils, "model_to_graphviz", lambda model: synthetic_graphs._NoopDigraph(), raising=False
    )
    definition = _as_definition_subclass(
        VG20, BivariateSexShiftModelDefinition, config_name="sex-shift", **overrides
    )
    frame = synthetic_graphs.synthetic_frame(definition)
    frame["sex"] = np.where(frame["subject_code"] % 2 == 0, "F", "M")
    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name="synthetic",
            output_root_dir=output_dir,
            ci_prob=0.89,
            interval_kind="eti",
        ),
        sampling=sampling.SamplingConfiguration(
            draws=4, tune=4, chains=1, cores=1, target_accept=0.8, random_seed=11
        ),
    )
    os.makedirs(context.reporting.output_dir, exist_ok=True)
    context.set_model_data(
        model_data.BinomialModelData(
            X_obs=frame["age"].to_numpy().reshape(-1, 1),
            y_obs=frame["understood"].to_numpy().astype(int),
            n_trials=definition.n_trials,
        ),
        frame,
    )
    engine.resolve("priors")(context, definition)
    engine.resolve("build")(context, definition)
    return context


@pytest.mark.slow
def test_the_effect_arm_adds_exactly_two_coefficients_and_shifts_both_logits(
    synthetic_graphs, vg20_engine, tmp_path, monkeypatch
):
    import pytensor
    from scipy.special import logit

    reference = synthetic_graphs.build_synthetic_model(
        VG20, vg20_engine, output_dir=str(tmp_path / "vg20"), monkeypatch=monkeypatch
    )
    built = _build_effect_arm(
        synthetic_graphs,
        vg20_engine,
        str(tmp_path / "sex"),
        monkeypatch,
        sex_known_only=True,
        sex_effect_sigma=0.5,
    )
    ref_fp = synthetic_graphs.graph_fingerprint(reference.model)
    fp = synthetic_graphs.graph_fingerprint(built.model)
    added = [name for name in fp["free_RVs"] if name not in ref_fp["free_RVs"]]
    assert added == ["beta_sex_u", "beta_sex_q"]
    assert [n for n in fp["free_RVs"] if n not in added] == ref_fp["free_RVs"]
    assert fp["deterministics"] == ref_fp["deterministics"]
    assert fp["observed_RVs"] == ref_fp["observed_RVs"]

    model = built.model
    x_sex = model["x_sex"].get_value()
    assert set(np.unique(x_sex)) == {-0.5, 0.5}
    point = model.initial_point()
    # The deterministics are graphs over the random variables; evaluate them as
    # functions of the value variables, or the call draws from the priors and
    # the two evaluations agree by accident.
    outputs = model.replace_rvs_by_values([model["p_u_obs"], model["q_obs"]])
    fn = pytensor.function(model.value_vars, outputs, on_unused_input="ignore")
    p0, q0 = fn(*[point[v.name] for v in model.value_vars])
    shifted = dict(point, beta_sex_u=np.array(1.0), beta_sex_q=np.array(0.4))
    p1, q1 = fn(*[shifted[v.name] for v in model.value_vars])
    du = logit(p1) - logit(p0)
    dq = logit(q1) - logit(q0)
    np.testing.assert_allclose(du[x_sex > 0], 0.5, atol=1e-9)
    np.testing.assert_allclose(du[x_sex < 0], -0.5, atol=1e-9)
    np.testing.assert_allclose(dq[x_sex > 0], 0.2, atol=1e-9)
    np.testing.assert_allclose(dq[x_sex < 0], -0.2, atol=1e-9)


@pytest.mark.slow
def test_a_coefficient_without_the_restriction_is_refused(
    synthetic_graphs, vg20_engine, tmp_path, monkeypatch
):
    with pytest.raises(ValueError, match="needs sex_known_only"):
        _build_effect_arm(
            synthetic_graphs, vg20_engine, str(tmp_path / "bad"), monkeypatch, sex_effect_sigma=0.5
        )


# ---------------------------------------------------------------------------
# Frame checks against the prepared database.
# ---------------------------------------------------------------------------


def test_the_restricted_frame_carries_sex_for_every_row(require_prepared_data):
    from vocab_growth.analysis_frames import analysis_frame_hash
    from vocab_growth.models.common_bivariate_re import (
        build_bivariate_re_analysis_frame,
    )

    reference, _ = build_bivariate_re_analysis_frame(VG20)
    inert = _as_definition_subclass(VG20, BivariateSexShiftModelDefinition, config_name="inert")
    inert_frame, _ = build_bivariate_re_analysis_frame(inert)
    assert analysis_frame_hash(inert_frame) == analysis_frame_hash(reference)

    restricted = _as_definition_subclass(
        VG20, BivariateSexShiftModelDefinition, config_name="known", sex_known_only=True
    )
    frame, info = build_bivariate_re_analysis_frame(restricted)
    assert "sex" in frame.columns
    assert frame["sex"].notna().all()
    assert set(frame["sex"].unique()) <= set(SEX_CONTRAST)
    assert info["sex_unknown_rows_excluded"] > 0
    assert len(frame) < len(reference)
    # Sex is recorded in eight of the fourteen Down syndrome studies.
    assert not set(frame["study"]) & {"ie_01", "it_01", "nz_01", "uk_03", "uk_04", "us_02"}
    # And it is a child-level covariate.
    assert (frame.groupby(["study", "subject_id"])["sex"].nunique() == 1).all()
