# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for parameter-recovery scoring (issue #163).

Data-free and sampling-free, so these run in CI. The behaviour that matters most
is the *gate*: a recovery fit that did not converge, or whose convergence was
never recorded, must never be reported as having recovered its parameters — a
truth sitting outside the posterior of an unconverged fit is sampler noise, not
evidence about identifiability.
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from vocab_growth.recovery.compare import (
    aggregate_table,
    interval_kind_for_target,
    is_excluded_target,
    pooled_row,
    recovery_table,
    summarise,
    target_variables,
)


def _posterior(**variables) -> xr.Dataset:
    """A posterior with one chain and however many draws each variable supplies."""
    data = {}
    for name, values in variables.items():
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            data[name] = (("chain", "draw"), array[None, :])
        else:
            dim = "query_id" if name.endswith("_query") else "study_id"
            data[name] = (("chain", "draw", dim), array[None, :, :])
    return xr.Dataset(data)


def _truth(**variables) -> xr.Dataset:
    data = {}
    for name, value in variables.items():
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            data[name] = (("chain", "draw"), array.reshape(1, 1))
        else:
            dim = "query_id" if name.endswith("_query") else "study_id"
            data[name] = (("chain", "draw", dim), array[None, None, :])
    return xr.Dataset(data)


def _write_diagnostics(dirpath, max_rhat, min_ess):
    pd.DataFrame(
        {"r_hat": [1.0, max_rhat], "ess_bulk": [5000.0, min_ess]},
        index=["intercept", "slope"],
    ).to_csv(dirpath / "diagnostics.csv")


def _write_gate_payload(dirpath, **overrides):
    """A ``diagnostics_summary.json`` in the shape the fit pipeline writes."""
    payload = {
        "passed": True,
        "checks": {
            "rhat": True,
            "ess": True,
            "divergences": True,
            "bfmi": True,
            "diagnostics_assessable": True,
        },
        "divergences": 0,
        "max_rhat": 1.004,
        "min_ess": 1500.0,
        "bfmi_per_chain": [0.9, 0.85],
        "rhat_failing": [],
        "ess_failing": [],
        "unassessable_parameters": [],
        "thresholds": {"rhat_max": 1.01, "ess_threshold": 400, "bfmi_threshold": 0.3},
    }
    payload.update(overrides)
    (dirpath / "diagnostics_summary.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_score_places_a_central_truth_inside_both_intervals():
    draws = np.linspace(-3.0, 3.0, 2001)
    table = recovery_table(_truth(slope=0.0), _posterior(slope=draws))

    row = table.iloc[0]
    assert row["quantity"] == "slope"
    # bool(...) because a DataFrame stores these as numpy booleans.
    assert bool(row["within_ci50"])
    assert bool(row["within_ci89"])
    assert abs(row["z"]) < 0.01
    # A truth at the centre of a symmetric posterior sits at the median quantile.
    assert row["truth_quantile"] == pytest.approx(0.5, abs=0.01)


def test_score_places_an_extreme_truth_outside_the_outer_interval():
    draws = np.linspace(-3.0, 3.0, 2001)
    table = recovery_table(_truth(slope=2.9), _posterior(slope=draws))

    row = table.iloc[0]
    assert not bool(row["within_ci50"])
    assert not bool(row["within_ci89"])
    assert row["z"] < 0
    assert row["truth_quantile"] > 0.95


def test_truth_quantile_is_the_fraction_of_draws_below_the_truth():
    draws = np.arange(100.0)
    table = recovery_table(_truth(slope=25.0), _posterior(slope=draws))
    # 25 of 100 draws are strictly below 25, plus half of the single tie.
    assert table.iloc[0]["truth_quantile"] == pytest.approx(0.255)


def test_query_grid_rows_are_labelled_with_query_ages():
    draws = np.tile(np.linspace(0.1, 0.3, 101)[:, None], (1, 3))
    table = recovery_table(
        _truth(p_query=[0.2, 0.2, 0.2]),
        _posterior(p_query=draws),
        query_ages=np.array([12, 24, 36]),
    )
    assert table["quantity"].tolist() == ["p_query"] * 3
    assert table["index"].tolist() == ["12", "24", "36"]
    assert table["dimension"].tolist() == ["query_id"] * 3


def test_target_selection_excludes_non_estimands():
    draws = np.linspace(0.0, 1.0, 11)
    grid = np.tile(draws[:, None], (1, 2))
    posterior = _posterior(
        p_query=grid,
        f_query=grid,          # logit-scale duplicate of p_query
        z_query=grid,          # standardised age grid: design, not a parameter
        delta=grid,            # study effects: element-wise target
        delta_raw=grid,        # non-centred helper
        slope=draws,
    )
    truth = _truth(
        p_query=[0.5, 0.5],
        f_query=[0.0, 0.0],
        z_query=[0.0, 0.0],
        delta=[0.0, 0.0],
        delta_raw=[0.0, 0.0],
        slope=0.5,
    )

    elementwise, aggregate = target_variables(posterior, truth)
    assert set(elementwise) == {"p_query", "delta", "slope"}
    assert aggregate == []
    assert is_excluded_target("f_u_query")
    assert is_excluded_target("h_query")
    assert is_excluded_target("z_obs")
    assert is_excluded_target("delta_subject_raw")
    assert not is_excluded_target("p_u_query")
    assert not is_excluded_target("kappa_u_query")


def test_grid_valued_dispersion_uses_the_same_interval_kind_as_the_scalar():
    # The project reports the skewed estimands with highest-density intervals; a
    # grid-valued dispersion is the same estimand and must not silently switch.
    assert interval_kind_for_target("kappa") == "hdi"
    assert interval_kind_for_target("kappa_u_query") == "hdi"
    assert interval_kind_for_target("kappa_sign_plot") == "hdi"
    assert interval_kind_for_target("psi") == "hdi"
    assert interval_kind_for_target("conc") == "hdi"
    assert interval_kind_for_target("q_query") == "eti"
    assert interval_kind_for_target("p_u_query") == "eti"


def test_aggregate_table_summarises_high_dimensional_effects():
    rng = np.random.default_rng(0)
    n_subjects, n_draws = 40, 200
    true_effects = rng.normal(0.0, 0.5, size=n_subjects)
    # Posteriors centred on the truth, so coverage should be near nominal.
    draws = true_effects[None, :] + rng.normal(0.0, 0.2, size=(n_draws, n_subjects))
    posterior = xr.Dataset(
        {"delta_subject": (("chain", "draw", "subject_id"), draws[None, :, :])}
    )
    truth = xr.Dataset(
        {"delta_subject": (("chain", "draw", "subject_id"), true_effects[None, None, :])}
    )

    elementwise, aggregate = target_variables(posterior, truth)
    assert elementwise == []
    assert aggregate == ["delta_subject"]

    table = aggregate_table(truth, posterior)
    row = table.iloc[0]
    assert row["quantity"] == "delta_subject"
    assert row["n_elements"] == n_subjects
    assert row["coverage_ci89"] > 0.7
    assert row["truth_vs_posterior_mean_correlation"] > 0.8
    # Per-child effects are not reported one by one.
    assert len(recovery_table(truth, posterior)) == 0


def test_non_converged_fit_is_never_reported_as_recovered(tmp_path):
    _write_diagnostics(tmp_path, max_rhat=1.05, min_ess=120.0)
    table = pd.DataFrame(
        [
            {"quantity": "slope", "index": "", "z": 0.1, "within_ci50": True, "within_ci89": True},
            {"quantity": "q_query", "index": "24", "z": 0.2, "within_ci50": True, "within_ci89": True},
        ]
    )

    row = summarise(table, str(tmp_path), label="r01", truth_source="posterior")
    assert row["converged"] is False
    assert row["verdict"] == "NON-CONVERGED (not assessed)"
    # Every target was covered, and it still must not read as recovered.
    assert row["coverage_ci89"] == 1.0


def test_missing_diagnostics_are_reported_as_unverified(tmp_path):
    table = pd.DataFrame(
        [{"quantity": "slope", "index": "", "z": 0.1, "within_ci50": True, "within_ci89": True}]
    )
    row = summarise(table, str(tmp_path), label="r01", truth_source="prior")
    assert row["converged"] is None
    assert "UNVERIFIED" in row["verdict"]


def test_converged_fit_reports_the_quantities_that_missed(tmp_path):
    _write_gate_payload(tmp_path)
    table = pd.DataFrame(
        [
            {"quantity": "slope", "index": "", "z": 0.3, "within_ci50": True, "within_ci89": True},
            {"quantity": "q_query", "index": "24", "z": 5.5, "within_ci50": False, "within_ci89": False},
            {"quantity": "tau", "index": "", "z": 0.1, "within_ci50": True, "within_ci89": True},
        ]
    )

    row = summarise(table, str(tmp_path), label="r02", truth_source="posterior")
    assert row["converged"] is True
    assert row["caveats"] == ""
    assert row["verdict"] == "not recovered: q_query"
    assert row["n_targets"] == 3
    assert row["n_within_ci89"] == 2
    assert row["worst_quantity"] == "q_query[24]"


def test_converged_fit_with_large_z_but_full_coverage_is_flagged(tmp_path):
    _write_gate_payload(tmp_path)
    table = pd.DataFrame(
        [{"quantity": "kappa_u_query", "index": "24", "z": 6.0, "within_ci50": True, "within_ci89": True}]
    )
    row = summarise(table, str(tmp_path), label="r03", truth_source="posterior")
    assert row["verdict"].startswith("recovered, but |z| up to")


def test_clean_payload_is_required_for_a_recovered_verdict(tmp_path):
    _write_gate_payload(tmp_path)
    table = pd.DataFrame(
        [{"quantity": "slope", "index": "", "z": 0.1, "within_ci50": True, "within_ci89": True}]
    )
    row = summarise(table, str(tmp_path), label="r04", truth_source="posterior")
    assert row["converged"] is True
    assert row["caveats"] == ""
    assert row["verdict"] == "recovered (every target within its 89% interval)"


def test_caveated_payload_is_never_scored_recovered(tmp_path):
    # Hard tier passed, but the sampler recorded divergent transitions: the
    # soft tier failed and "recovered" is reserved for a clean payload.
    _write_gate_payload(
        tmp_path,
        passed=False,
        checks={
            "rhat": True, "ess": True, "divergences": False, "bfmi": True,
            "diagnostics_assessable": True,
        },
        divergences=5,
    )
    table = pd.DataFrame(
        [{"quantity": "slope", "index": "", "z": 0.1, "within_ci50": True, "within_ci89": True}]
    )
    row = summarise(table, str(tmp_path), label="r05", truth_source="posterior")
    assert row["converged"] is True
    assert row["verdict"].startswith("converged with caveats")
    assert "divergent" in row["caveats"]


def test_csv_fallback_is_flagged_and_never_scored_recovered(tmp_path):
    # A pre-payload fit has only the rounded, scalars-only diagnostics.csv; the
    # fallback verdict is caveated so the replicate cannot read as clean.
    _write_diagnostics(tmp_path, max_rhat=1.005, min_ess=2000.0)
    table = pd.DataFrame(
        [{"quantity": "slope", "index": "", "z": 0.1, "within_ci50": True, "within_ci89": True}]
    )
    row = summarise(table, str(tmp_path), label="r06", truth_source="posterior")
    assert row["converged"] is True
    assert row["verdict"].startswith("converged with caveats")
    assert "diagnostics.csv" in row["caveats"]


def test_pooled_row_counts_only_confirmed_converged_replicates():
    summaries = [
        {
            "replicate": "r01", "truth_source": "posterior", "converged": True,
            "max_rhat": 1.004, "min_ess": 1500.0, "n_targets": 100,
            "n_within_ci89": 90, "coverage_ci89": 0.9, "coverage_ci50": 0.5,
            "max_abs_z": 2.0, "quantities_outside_ci89": "q_query",
        },
        {
            "replicate": "r02", "truth_source": "posterior", "converged": False,
            "max_rhat": 1.05, "min_ess": 100.0, "n_targets": 100,
            "n_within_ci89": 40, "coverage_ci89": 0.4, "coverage_ci50": 0.2,
            "max_abs_z": 9.0, "quantities_outside_ci89": "tau",
        },
    ]

    row = pooled_row(summaries)
    # The unconverged replicate contributes nothing to the pooled numbers.
    assert row["n_targets"] == 100
    assert row["n_within_ci89"] == 90
    assert row["coverage_ci89"] == pytest.approx(0.9)
    assert row["max_abs_z"] == 2.0
    assert row["quantities_outside_ci89"] == "q_query"
    assert "1 of 2 replicates assessed" in row["replicate"]
    assert "indicative" in row["verdict"]
