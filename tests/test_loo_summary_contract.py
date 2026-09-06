# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LOO writers and report decisions preserve unavailable diagnostics (#309)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dse_research_utils.statistics.loo import loo_summary_row

from vocab_growth.models.common import emit_loo_summary
from vocab_growth.report_cells import render_loo_section


def _loo(values, threshold=0.7):
    return SimpleNamespace(
        elpd=-10.0, se=1.0, p=2.0, pareto_k=xr.DataArray(values),
        good_k=threshold, n_data_points=len(values), n_samples=1000,
    )


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf])
def test_nonfinite_thresholds_are_rejected_before_writing(tmp_path, threshold):
    loo = _loo([0.0, 0.2], threshold)
    with pytest.raises(ValueError, match="finite"):
        emit_loo_summary({"all": loo}, {}, str(tmp_path))
    assert not (tmp_path / "loo_summary.csv").exists()
    with pytest.raises(ValueError, match="finite"):
        loo_summary_row(_loo([0.1]), label="all", k_threshold=threshold)


def test_zero_threshold_and_nonfinite_counts_reach_report(tmp_path, capsys):
    values = [-0.1, 0.0, 0.2, np.nan, np.inf, -np.inf]
    row = emit_loo_summary({"all": _loo(values, 0.0)}, {}, str(tmp_path)).iloc[0]
    assert row["good_k_threshold"] == 0.0
    assert row["pareto_k_good"] == 2
    assert row["pareto_k_bad"] == 1
    assert row["pareto_k_nonfinite"] == 3
    assert row["pareto_k_unusable"] == 4
    render_loo_section(str(tmp_path))
    text = capsys.readouterr().out
    assert "4 of 6 observations (67%)" in text
    assert "every one" not in text


@pytest.mark.parametrize("script", ["loo_compare", "loso_compare"])
def test_comparison_rows_keep_union_without_double_counting(script, capsys):
    spec = importlib.util.spec_from_file_location(script, Path(__file__).parents[1] / "scripts" / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    # Some scripts declare dataclasses, which resolve their module by name.
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    builder = module._loo_summary_row if script == "loo_compare" else module._summary_row
    row = builder("all", _loo([0.1, 0.8, np.nan, np.inf, -np.inf]), reff=0.8)
    assert row["pareto_k_gt_0.7"] == 2
    assert row["pareto_k_nonfinite"] == 3
    assert row["pareto_k_unusable"] == 4
    if script == "loo_compare":
        row = module._loo_summary_row("all", _loo([0.1, np.nan]))
        module._warn_if_unusable("all", row)
        assert "50%" in capsys.readouterr().out


def test_legacy_report_counts_missing_values_as_unusable(tmp_path, capsys):
    table = emit_loo_summary({"all": _loo([0.1, np.nan, -np.inf])}, {}, str(tmp_path))
    table.drop(columns=["pareto_k_nonfinite", "pareto_k_unusable"]).to_csv(tmp_path / "loo_summary.csv", index=False)
    render_loo_section(str(tmp_path))
    assert "2 of 3 observations (67%)" in capsys.readouterr().out


def test_missing_counts_cannot_produce_reliable_verdict(tmp_path, capsys):
    table = emit_loo_summary({"all": _loo([0.1])}, {}, str(tmp_path))
    table["pareto_k_good"] = np.nan
    table["pareto_k_unusable"] = np.nan
    table.to_csv(tmp_path / "loo_summary.csv", index=False)
    render_loo_section(str(tmp_path))
    assert "reliability cannot be assessed" in capsys.readouterr().out


def test_empty_loo_cannot_produce_reliable_verdict(tmp_path, capsys):
    emit_loo_summary({"all": _loo([])}, {}, str(tmp_path))
    render_loo_section(str(tmp_path))
    assert "reliability cannot be assessed" in capsys.readouterr().out


def test_interval_wrapper_uses_same_finite_draws_for_median_and_bounds():
    from vocab_growth.intervals import summarise

    samples = np.array([[1.0, 2.0, 3.0, np.inf, np.nan]])
    got = summarise(samples, np.array([12.0]))
    expected = summarise(samples[:, :3], np.array([12.0]))
    pd.testing.assert_frame_equal(got, expected)
