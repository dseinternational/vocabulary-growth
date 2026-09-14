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


def _write_manifest(dirpath, tier):
    (dirpath / "fit_manifest.json").write_text(
        json.dumps({"sampling": {"configuration_name": tier}}), encoding="utf-8"
    )


def test_summarise_records_the_sampling_tier_from_the_manifest(tmp_path):
    """The matrix has to say what each replicate was sampled at (#289 task 4.7)."""
    _write_gate_payload(tmp_path)
    table = pd.DataFrame(
        [{"quantity": "slope", "index": "", "z": 0.1, "within_ci50": True, "within_ci89": True}]
    )
    row = summarise(table, str(tmp_path), label="r01", truth_source="posterior")
    assert row["sampling_configuration"] is None

    _write_manifest(tmp_path, "rep")
    row = summarise(table, str(tmp_path), label="r01", truth_source="posterior")
    assert row["sampling_configuration"] == "rep"


def _summary(label, tier, *, converged=True):
    return {
        "replicate": label, "truth_source": "posterior", "sampling_configuration": tier,
        "converged": converged, "max_rhat": 1.004, "min_ess": 1500.0, "n_targets": 100,
        "n_within_ci89": 90, "coverage_ci89": 0.9, "coverage_ci50": 0.5,
        "max_abs_z": 2.0, "quantities_outside_ci89": "q_query",
    }


def test_pooled_row_refuses_to_pool_across_sampling_tiers():
    """The 2026-09-03 defect: a `rep` run's two replicates pooled with a stale
    `test`-tier fit in the third directory, and the comparison book rendered
    three replicates of three. The refusal keeps the POOLED prefix (so the
    book's filter still drops it) and says "not assessed" (so its counter does
    not count it)."""
    row = pooled_row([_summary("r01", "rep"), _summary("r02", "rep"), _summary("r03", "test")])
    assert row["replicate"].startswith("POOLED")
    assert "refused" in row["replicate"]
    assert row["verdict"].startswith("NOT POOLED (not assessed)")
    assert "rep: r01, r02" in row["verdict"]
    assert "test: r03" in row["verdict"]
    assert row["n_targets"] == 0
    assert row["converged"] is None
    assert np.isnan(row["coverage_ci89"])

    # A non-converged replicate at the other tier still stops the pooling: the
    # label "n of m assessed" would otherwise count it as a replicate of this run.
    row = pooled_row([_summary("r01", "rep"), _summary("r02", "test", converged=False)])
    assert "refused" in row["replicate"]


def test_pooled_row_pools_one_tier_and_records_it():
    row = pooled_row([_summary("r01", "rep"), _summary("r02", "rep")])
    assert row["replicate"] == "POOLED (2 of 2 replicates assessed)"
    assert row["sampling_configuration"] == "rep"
    assert row["n_targets"] == 200

    # Summaries that never recorded a tier still pool, as before the column existed.
    row = pooled_row([_summary("r01", None), _summary("r02", None)])
    assert row["replicate"] == "POOLED (2 of 2 replicates assessed)"
    assert row["sampling_configuration"] is None


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


# ----------------------------------------------------------------------------
# The total spread (#229 option 4, #289 task 4.12)
# ----------------------------------------------------------------------------
def _query_dataset(n_chain, n_draw, n_age, *, seed, spoken=True):
    """Posterior-shaped query-grid inputs for a VG20-like model."""
    rng = np.random.default_rng(seed)

    def grid(lo, hi):
        return (("chain", "draw", "query_id"), rng.uniform(lo, hi, size=(n_chain, n_draw, n_age)))

    def scalar(lo, hi):
        return (("chain", "draw"), rng.uniform(lo, hi, size=(n_chain, n_draw)))

    data = {
        "p_u_query": grid(0.05, 0.6),
        "kappa_u_query": grid(10.0, 60.0),
        "tau_subj_u": scalar(0.6, 1.0),
    }
    if spoken:
        data.update({
            "q_query": grid(0.02, 0.5),
            "kappa_s_query": grid(5.0, 40.0),
            "tau_subj_q": scalar(0.9, 1.4),
            "rho_uq": scalar(0.2, 0.5),
        })
    return xr.Dataset(data)


def test_total_spread_is_derived_on_the_query_grid_by_the_comparison_function():
    """The recovery quantity is the comparison's own computation, arranged draw by
    draw on the query grid. The maths is tested in test_comparison.py; this tests
    that the reshaping and assignment keep chain, draw and age aligned."""
    from vocab_growth import comparison
    from vocab_growth.models.definitions import MODEL_REGISTRY
    from vocab_growth.recovery.compare import with_total_spread

    definition = MODEL_REGISTRY["vg20"]
    ages = np.asarray(definition.ages_query, dtype=float)
    dataset = _query_dataset(2, 3, ages.size, seed=1)
    out = with_total_spread(dataset, definition)

    for name in ("total_spread_words_u_query", "total_spread_words_s_query"):
        assert out[name].dims == ("chain", "draw", "query_id")
    flat = {name: dataset[name].values.reshape(6, -1).squeeze() for name in dataset.data_vars}
    for name in ("tau_subj_u", "tau_subj_q", "rho_uq"):
        flat[name] = dataset[name].values.reshape(6)
    plan = comparison.total_spread_plan(definition, "spoken", "query")
    want = comparison.total_spread_from_values(plan, flat, ages, 810).sd_words
    np.testing.assert_allclose(out["total_spread_words_s_query"].values.reshape(6, -1), want, rtol=1e-12)
    # Every draw and age is a spread in words, and a positive one.
    assert np.all(out["total_spread_words_u_query"].values > 0)


def test_total_spread_rows_are_scored_at_the_query_ages():
    """The derived variable sits on query_id, so target selection picks it up."""
    from vocab_growth.models.definitions import MODEL_REGISTRY
    from vocab_growth.recovery.compare import with_total_spread

    definition = MODEL_REGISTRY["vg12"]
    ages = np.asarray(definition.ages_query, dtype=float)
    rng = np.random.default_rng(2)
    truth = xr.Dataset({
        "p_query": (("chain", "draw", "query_id"), rng.uniform(0.1, 0.5, size=(1, 1, ages.size))),
        "kappa_query": (("chain", "draw", "query_id"), np.full((1, 1, ages.size), 30.0)),
        "tau_subject": (("chain", "draw"), np.array([[0.7]])),
    })
    posterior = xr.Dataset({
        "p_query": (("chain", "draw", "query_id"), np.tile(truth["p_query"].values, (1, 200, 1))),
        "kappa_query": (("chain", "draw", "query_id"), rng.uniform(25.0, 35.0, size=(1, 200, ages.size))),
        "tau_subject": (("chain", "draw"), rng.uniform(0.6, 0.8, size=(1, 200))),
    })
    table = recovery_table(
        with_total_spread(truth, definition),
        with_total_spread(posterior, definition),
        query_ages=ages,
    )
    rows = table[table["quantity"] == "total_spread_words_query"]
    assert rows["index"].tolist() == [f"{a:g}" for a in ages]
    assert rows["within_ci89"].all()


def test_compare_replicate_scores_the_total_spread_from_a_stored_trace(tmp_path):
    """End to end: the derived spread joins the graph's own targets, scored at every
    query age, and a caller without a definition keeps the previous target set."""
    from vocab_growth.models.definitions import MODEL_REGISTRY
    from vocab_growth.recovery.compare import compare_replicate

    definition = MODEL_REGISTRY["vg12"]
    ages = np.asarray(definition.ages_query, dtype=float)
    rng = np.random.default_rng(8)
    p_true = rng.uniform(0.1, 0.5, size=ages.size)
    dims = ("chain", "draw", "query_id")
    truth = xr.DataTree.from_dict({"posterior": xr.Dataset({
        "p_query": (dims, p_true.reshape(1, 1, -1)),
        "kappa_query": (dims, np.full((1, 1, ages.size), 30.0)),
        "tau_subject": (("chain", "draw"), np.array([[0.7]])),
    })})
    posterior = xr.Dataset({
        "p_query": (dims, np.tile(p_true, (2, 100, 1)) * rng.uniform(0.98, 1.02, size=(2, 100, 1))),
        "kappa_query": (dims, rng.uniform(27.0, 33.0, size=(2, 100, ages.size))),
        "tau_subject": (("chain", "draw"), rng.uniform(0.65, 0.75, size=(2, 100))),
    })
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    trace = fit_dir / "trace.nc"
    xr.DataTree.from_dict({"posterior": posterior}).to_netcdf(str(trace))
    _write_gate_payload(fit_dir)

    table, _aggregates, summary = compare_replicate(
        truth, str(trace), str(fit_dir), label="r01", truth_source="posterior",
        query_ages=ages, definition=definition,
    )
    assert set(table["quantity"]) == {
        "p_query", "kappa_query", "tau_subject", "total_spread_words_query"
    }
    assert (table["quantity"] == "total_spread_words_query").sum() == ages.size
    assert summary["n_targets"] == len(table)

    without, _, _ = compare_replicate(
        truth, str(trace), str(fit_dir), label="r01", truth_source="posterior",
        query_ages=ages,
    )
    assert "total_spread_words_query" not in set(without["quantity"])

    # A truth on a different query grid -- same length, one age moved -- has no
    # age at which both spreads describe the same child, so it is left out
    # rather than derived at the fitted model's ages.
    from dataclasses import replace

    moved = replace(definition, ages_query=(*definition.ages_query[:-1], definition.ages_query[-1] - 1))
    other_grid, _, _ = compare_replicate(
        truth, str(trace), str(fit_dir), label="r01", truth_source="posterior",
        query_ages=ages, definition=definition, truth_definition=moved,
    )
    assert "total_spread_words_query" not in set(other_grid["quantity"])


def test_total_spread_is_not_derived_where_the_quadrature_does_not_apply():
    """The joint engine and VG22's factor keep the target set they had."""
    from vocab_growth.models.definitions import MODEL_REGISTRY
    from vocab_growth.recovery.compare import total_spread_targets, with_total_spread

    for key in ("vg15", "vg22", "vg14"):
        assert total_spread_targets(MODEL_REGISTRY[key]) == ()
    dataset = _query_dataset(1, 2, 3, seed=3)
    assert with_total_spread(dataset, MODEL_REGISTRY["vg22"]) is dataset
    assert [name for _, name in total_spread_targets(MODEL_REGISTRY["vg11"])] == [
        "total_spread_words_query"
    ]


def test_total_spread_refuses_a_dataset_missing_an_input():
    """A missing input is an error, not a quietly smaller target set."""
    from vocab_growth.models.definitions import MODEL_REGISTRY
    from vocab_growth.recovery.compare import with_total_spread

    definition = MODEL_REGISTRY["vg20"]
    dataset = _query_dataset(1, 2, len(definition.ages_query), seed=4, spoken=False)
    with pytest.raises(KeyError, match="q_query"):
        with_total_spread(dataset, definition)
