# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute the teaching route and the builders' separate reporting interface."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from support.synthetic_graphs import synthetic_frame

from vocab_growth.models.catalogue import get
from vocab_growth.models.common_bivariate import unseen_child_offsets
from vocab_growth.models.subject_effects import resolve

pytestmark = [pytest.mark.slow, pytest.mark.emits_reporting_artefacts]

_PATH = Path(__file__).resolve().parents[1] / "examples" / "model_walkthrough.py"
_SPEC = importlib.util.spec_from_file_location("model_walkthrough", _PATH)
walkthrough = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(walkthrough)


@pytest.mark.parametrize("key", ["vg01", "vg05", "vg07", "vg11", "vg14", "vg15"])
def test_each_engine_builds_without_report_files(key, tmp_path, capsys):
    definition = replace(get(key).definition, n_plot=12)
    context, details = walkthrough.build_example(
        definition,
        synthetic_frame(definition, n_rows=12),
        output_dir=tmp_path,
    )
    assert context.model.observed_RVs
    assert details.tables
    assert not context.plots
    assert not list(tmp_path.rglob("*"))
    assert capsys.readouterr().out == ""


def test_the_complete_student_example_executes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = walkthrough.run_walkthrough()
    assert set(result["contexts"]) == {"VG01", "VG05", "VG07", "VG10", "VG20"}
    assert result["prior"].prior["p_query"].shape[1] == 8
    assert result["missing_parent"].spoken_spec.n_marginal == 1
    np.testing.assert_array_equal(result["one_study"].model["delta_u"].eval(), [0.0])
    assert result["reference_age"] == 0.0
    np.testing.assert_allclose(result["cells"], [0.2, 0.3, 0.4, 0.1])
    assert not list(tmp_path.rglob("*"))


def test_a_missing_slope_variable_does_not_select_constant_offsets(tmp_path):
    definition = replace(get("vg19").definition, n_plot=12)
    context, _ = walkthrough.build_example(
        definition, walkthrough.small_frame(), output_dir=tmp_path
    )
    context.model_variables.pop("tau_subj_u_1")
    with context.model, pytest.raises(ValueError, match="tau_subj_u_1"):
        unseen_child_offsets(context, definition, resolve(definition), "u")


def test_count_variance_matches_the_mean_concentration_formula():
    table = walkthrough.count_variation()
    np.testing.assert_allclose(table["mean"], 20.0)
    expected = [16.0, *(100 * 0.2 * 0.8 * (100 + k) / (1 + k) for k in (2, 20, 200))]
    np.testing.assert_allclose(table["variance"], expected)
