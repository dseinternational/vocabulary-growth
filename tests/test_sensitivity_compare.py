# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the prior-sensitivity baseline/variant comparison (issue #89 §7).

Pure/fast (no sampling). The load-bearing regression here is the VG15 shape:
``compare_dirs`` emits ``within_baseline_hdi = None`` for the HDI-less series
(``Ey_any``, ``P_psi_gt_1``), which makes the column object dtype (mixed Python
bools + None) even after ``dropna``. ``summarise`` must coerce before inverting
the mask — on object dtype ``~True``/``~False`` are the ints -2/-1 and ``.loc``
raises ``KeyError``, aborting ``scripts/compare_sensitivity.py`` for vg15.
VG10/VG11 frames are pure bool and never hit this.
"""

import json

import pandas as pd

from vocab_growth.sensitivity.compare import (
    compare_dirs,
    coverage_report,
    diagnostics_gate,
    summarise,
)


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


def test_summarise_tolerates_mixed_bool_none_ci_column(tmp_path):
    # Hand-built frame in the VG15 shape: object-dtype within_baseline_ci
    # mixing True/False/None (the interval-less P_psi_gt_1 / four-cell rows).
    comparison = pd.DataFrame([
        {"quantity": "q", "age_months": 30, "delta": -0.05, "within_baseline_ci": True},
        {"quantity": "psi", "age_months": -1, "delta": 0.4, "within_baseline_ci": False},
        {"quantity": "four_cell", "age_months": 30, "delta": 20.0, "within_baseline_ci": None},
        {"quantity": "P_psi_gt_1", "age_months": -1, "delta": -0.01, "within_baseline_ci": None},
    ])
    assert comparison["within_baseline_ci"].dtype == object  # the failing shape

    row = summarise(comparison, str(tmp_path), label="vg15-variant")
    assert row["n_checked"] == 2
    assert row["n_within_ci"] == 1
    assert row["quantities_outside_ci"] == "psi"
    assert row["status"] == "unverified-pairing"
    # Unchecked (None) rows still count towards the magnitude summary.
    assert row["max_abs_delta"] == 20.0


def _write_vg15_outputs(dirpath, q_median, p_any_median, ey_any_median, psi_median, p_psi_gt_1):
    pd.DataFrame({
        "age_months": [30], "q_median": [q_median],
        "q_ci_lo": [0.4], "q_ci_hi": [0.6],
    }).to_csv(dirpath / "posterior_summary_q.csv", index=False)
    pd.DataFrame({
        "age_months": [30], "p_any_median": [p_any_median],
        "p_any_ci_lo": [0.2], "p_any_ci_hi": [0.4],
        "Ey_any_median": [ey_any_median],
        "Ey_any_ci_lo": [ey_any_median - 30.0], "Ey_any_ci_hi": [ey_any_median + 30.0],
    }).to_csv(dirpath / "posterior_summary_p_any.csv", index=False)
    pd.DataFrame({
        "psi_median": [psi_median], "psi_ci_lo": [1.2], "psi_ci_hi": [1.9],
        "P_psi_gt_1": [p_psi_gt_1],
    }).to_csv(dirpath / "posterior_summary_psi.csv", index=False)


def test_compare_dirs_then_summarise_vg15_shape(tmp_path):
    base_dir, var_dir = tmp_path / "base", tmp_path / "var"
    base_dir.mkdir(), var_dir.mkdir()
    _write_gate_payload(base_dir)
    _write_vg15_outputs(base_dir, q_median=0.5, p_any_median=0.3,
                        ey_any_median=100.0, psi_median=1.5, p_psi_gt_1=0.99)
    # q shifts outside the baseline 89% interval; p_any, Ey_any and psi stay within
    # (base Ey_any interval is [90, 150]; the variant median 120 sits inside it).
    _write_vg15_outputs(var_dir, q_median=0.7, p_any_median=0.35,
                        ey_any_median=120.0, psi_median=1.6, p_psi_gt_1=0.98)
    _write_gate_payload(var_dir)

    comparison = compare_dirs(str(base_dir), str(var_dir))
    by_qty = comparison.set_index("quantity")["within_baseline_ci"]
    # Ey_any now carries an interval and is assessed; only the interval-less
    # P_psi_gt_1 row stays None, so the column is still object dtype.
    assert by_qty["Ey_any"] is True
    assert by_qty["P_psi_gt_1"] is None
    assert by_qty["q"] is False
    assert by_qty["p_any"] is True
    assert by_qty["psi"] is True

    row = summarise(comparison, str(var_dir), label="vg15-variant",
                    baseline_dir=str(base_dir), validation_errors=[],
                    coverage=coverage_report(str(base_dir), str(var_dir)))
    assert row["converged"] is True
    assert row["status"] == "compared"
    assert row["caveats"] == ""
    assert row["n_checked"] == 4
    assert row["n_within_ci"] == 3
    assert row["quantities_outside_ci"] == "q"
    assert row["verdict"] == "sensitive: q"


# --- The VG16 cross-lag coefficient (issue #242) -------------------------------


def _write_grid_outputs(dirpath, *, gap_ages, gap_offset, q_ages):
    """A query-grid series (`q`) and the plot-grid `gap` series, both readable."""
    dirpath.mkdir(exist_ok=True)
    pd.DataFrame({
        "age_months": gap_ages,
        "gap_median": 2.0 * gap_ages + gap_offset,
        "ci_lo": 2.0 * gap_ages - 5.0,
        "ci_hi": 2.0 * gap_ages + 5.0,
    }).to_csv(dirpath / "comprehension_production_gap.csv", index=False)
    pd.DataFrame({
        "age_months": q_ages,
        "q_median": 0.5,
        "q_ci_lo": 0.4,
        "q_ci_hi": 0.6,
    }).to_csv(dirpath / "posterior_summary_q.csv", index=False)


def test_plot_grid_series_are_compared_as_curves_inside_the_variant_support(tmp_path):
    """#289 task 4.2: a restricted pool gets a different linspace for `gap`.

    Matched on exact ages, VG10 `dse-native-only` shared 39 of 335 baseline
    rows and was reported as partial coverage with nothing to compare. The
    variant's curve is now interpolated onto the baseline's plot ages inside
    the variant's own support, while the query-grid series still match
    exactly, so a genuinely narrower support still counts against coverage.
    """
    import numpy as np

    base_dir, var_dir = tmp_path / "base", tmp_path / "var"
    base_gap = np.linspace(8.0, 72.0, 65)
    var_gap = np.linspace(9.0, 71.9, 57)  # a restricted pool's own linspace
    q_ages = [12.0, 18.0, 24.0, 30.0]
    _write_grid_outputs(base_dir, gap_ages=base_gap, gap_offset=0.0, q_ages=q_ages)
    _write_grid_outputs(var_dir, gap_ages=var_gap, gap_offset=1.0, q_ages=q_ages[:-1])
    # Exact matching would keep only the ages that coincide by accident.
    assert len(set(base_gap) & set(var_gap)) <= 2

    baseline_rows, shared_rows, missing = coverage_report(str(base_dir), str(var_dir))
    inside = int(((base_gap >= 9.0) & (base_gap <= 71.9)).sum())
    assert baseline_rows == 65 + 4
    # The plot-grid series is covered inside the variant's support; the
    # query-grid series loses exactly the age the variant does not report.
    assert shared_rows == inside + 3
    assert missing == []

    comparison = compare_dirs(str(base_dir), str(var_dir))
    gap = comparison[comparison["quantity"] == "gap"]
    assert len(gap) == inside
    assert gap["age_months"].min() >= 9.0
    # A linear curve interpolates exactly: the variant sits 1.0 above the
    # baseline at every compared age, inside the baseline's interval.
    assert np.allclose(gap["delta"], 1.0)
    assert gap["within_baseline_ci"].all()
    q = comparison[comparison["quantity"] == "q"]
    assert sorted(q["age_months"]) == q_ages[:-1]


def _write_diagnostics(dirpath, *, beta_lag=None, r_hat=1.005):
    """A diagnostics.csv in the shape every fit writes: params in column 0."""
    index = ["tau_u", "kappa_min_u"]
    mean = [0.4, 20.0]
    lb = [0.2, 15.0]
    ub = [0.6, 25.0]
    if beta_lag is not None:
        index.append("beta_lag")
        mean.append(beta_lag[0])
        lb.append(beta_lag[1])
        ub.append(beta_lag[2])
    pd.DataFrame(
        {
            "mean": mean, "sd": [0.1] * len(index),
            "eti89_lb": lb, "eti89_ub": ub,
            "ess_bulk": [1000.0] * len(index), "ess_tail": [900.0] * len(index),
            "r_hat": [r_hat] * len(index),
        },
        index=index,
    ).to_csv(dirpath / "diagnostics.csv")


def test_load_parameters_reads_the_cross_lag_mean_from_diagnostics(tmp_path):
    from vocab_growth.sensitivity.compare import load_parameters

    _write_diagnostics(tmp_path, beta_lag=(0.199, 0.089, 0.311))
    row = load_parameters(str(tmp_path))["beta_lag"].iloc[0]
    assert row["estimate"] == 0.199
    assert row["estimate_kind"] == "mean"
    assert row["ci_lo"] == 0.089
    assert row["ci_hi"] == 0.311


def test_load_parameters_omits_absent_cross_lag(tmp_path):
    """Every bivariate fit writes diagnostics.csv; only VG16 carries beta_lag."""
    from vocab_growth.sensitivity.compare import load_parameters

    _write_diagnostics(tmp_path, beta_lag=None)
    assert "beta_lag" not in load_parameters(str(tmp_path))
    assert load_parameters(str(tmp_path / "does-not-exist")) == {}


def test_a_moved_cross_lag_is_scored_rather_than_ignored(tmp_path):
    """The defect this closes: a variant that halves beta_lag while leaving the
    trajectories alone was previously scored **robust**, because beta_lag was in
    no compared series. VG16 supplies no other reported number."""
    base_dir, var_dir = tmp_path / "base", tmp_path / "var"
    base_dir.mkdir(), var_dir.mkdir()
    _write_gate_payload(base_dir)
    _write_vg15_outputs(base_dir, q_median=0.5, p_any_median=0.3,
                        ey_any_median=100.0, psi_median=1.5, p_psi_gt_1=0.99)
    _write_vg15_outputs(var_dir, q_median=0.5, p_any_median=0.3,
                        ey_any_median=100.0, psi_median=1.5, p_psi_gt_1=0.99)
    _write_diagnostics(base_dir, beta_lag=(0.20, 0.09, 0.31))
    _write_diagnostics(var_dir, beta_lag=(0.05, 0.01, 0.12))   # outside [0.09, 0.31]

    comparison = compare_dirs(str(base_dir), str(var_dir))
    by_qty = comparison.set_index("quantity")["within_baseline_ci"]
    assert by_qty["beta_lag"] is False
    assert by_qty["q"] is True          # every trajectory is unmoved

    row = summarise(comparison, str(var_dir), label="vg16-conditional-only",
                    baseline_dir=str(base_dir), validation_errors=[],
                    coverage=coverage_report(str(base_dir), str(var_dir)))
    assert "beta_lag" in row["quantities_outside_ci"]
    assert row["verdict"].startswith("sensitive")


def test_beta_lag_within_the_baseline_interval_scores_robust(tmp_path):
    base_dir, var_dir = tmp_path / "base", tmp_path / "var"
    base_dir.mkdir(), var_dir.mkdir()
    _write_gate_payload(base_dir)
    _write_vg15_outputs(base_dir, q_median=0.5, p_any_median=0.3,
                        ey_any_median=100.0, psi_median=1.5, p_psi_gt_1=0.99)
    _write_vg15_outputs(var_dir, q_median=0.5, p_any_median=0.3,
                        ey_any_median=100.0, psi_median=1.5, p_psi_gt_1=0.99)
    _write_diagnostics(base_dir, beta_lag=(0.20, 0.09, 0.31))
    _write_diagnostics(var_dir, beta_lag=(0.22, 0.10, 0.33))

    comparison = compare_dirs(str(base_dir), str(var_dir))
    assert comparison.set_index("quantity")["within_baseline_ci"]["beta_lag"] is True


# --- The diagnostics gate reads the canonical payload (issue #266, finding 7d) --


def test_diagnostics_gate_prefers_the_payload_and_reports_soft_caveats(tmp_path):
    _write_gate_payload(
        tmp_path,
        passed=False,
        checks={
            "rhat": True, "ess": True, "divergences": False, "bfmi": False,
            "diagnostics_assessable": True,
        },
        divergences=12,
        bfmi_per_chain=[0.2, 0.9],
    )
    # A contradictory diagnostics.csv proves the payload, not the CSV, is read.
    pd.DataFrame({"r_hat": [1.2], "ess_bulk": [10.0], "ess_tail": [10.0]}).to_csv(
        tmp_path / "diagnostics.csv", index=False
    )

    gate = diagnostics_gate(str(tmp_path))
    converged, max_rhat, min_ess = gate  # triple unpacking stays supported
    assert converged is True             # the hard R-hat/ESS tier passed
    assert gate.clean is False           # but the payload is not clean
    assert max_rhat == 1.004
    assert min_ess == 1500.0
    assert any("divergent" in caveat for caveat in gate.caveats)
    assert any("BFMI" in caveat for caveat in gate.caveats)


def test_diagnostics_gate_hard_failure_in_the_payload_is_non_converged(tmp_path):
    _write_gate_payload(
        tmp_path,
        passed=False,
        checks={
            "rhat": False, "ess": True, "divergences": True, "bfmi": True,
            "diagnostics_assessable": True,
        },
        max_rhat=1.08,
        rhat_failing=["eta"],
    )
    gate = diagnostics_gate(str(tmp_path))
    assert gate.converged is False
    assert gate.clean is False


def test_diagnostics_gate_flags_unassessable_parameters_as_a_hard_failure(tmp_path):
    _write_gate_payload(
        tmp_path,
        passed=False,
        checks={
            "rhat": True, "ess": True, "divergences": True, "bfmi": True,
            "diagnostics_assessable": False,
        },
        unassessable_parameters=["tau_subj_q"],
    )
    gate = diagnostics_gate(str(tmp_path))
    assert gate.converged is False
    assert gate.clean is False
    assert any("could not be assessed" in caveat for caveat in gate.caveats)


def test_diagnostics_gate_falls_back_to_the_csv_and_says_so(tmp_path):
    pd.DataFrame({"r_hat": [1.005], "ess_bulk": [1000.0], "ess_tail": [900.0]}).to_csv(
        tmp_path / "diagnostics.csv", index=False
    )
    gate = diagnostics_gate(str(tmp_path))
    assert gate.converged is True
    assert gate.clean is None
    assert gate.source == "diagnostics.csv"
    assert any("diagnostics.csv" in caveat for caveat in gate.caveats)


def test_summarise_reports_converged_with_caveats_rather_than_robust(tmp_path):
    base_dir, var_dir = tmp_path / "base", tmp_path / "var"
    base_dir.mkdir(), var_dir.mkdir()
    _write_gate_payload(base_dir)
    _write_vg15_outputs(base_dir, q_median=0.5, p_any_median=0.3,
                        ey_any_median=100.0, psi_median=1.5, p_psi_gt_1=0.99)
    _write_vg15_outputs(var_dir, q_median=0.5, p_any_median=0.3,
                        ey_any_median=100.0, psi_median=1.5, p_psi_gt_1=0.99)
    _write_gate_payload(
        var_dir,
        passed=False,
        checks={
            "rhat": True, "ess": True, "divergences": False, "bfmi": True,
            "diagnostics_assessable": True,
        },
        divergences=3,
    )

    comparison = compare_dirs(str(base_dir), str(var_dir))
    row = summarise(comparison, str(var_dir), label="vg15-variant",
                    baseline_dir=str(base_dir), validation_errors=[],
                    coverage=coverage_report(str(base_dir), str(var_dir)))
    assert row["converged"] is True
    assert row["status"] == "converged-with-caveats"
    assert "not scored robust" in row["verdict"]
    assert "divergent" in row["caveats"]


def test_summarise_keeps_robust_for_a_clean_payload(tmp_path):
    base_dir, var_dir = tmp_path / "base", tmp_path / "var"
    base_dir.mkdir(), var_dir.mkdir()
    _write_gate_payload(base_dir)
    _write_vg15_outputs(base_dir, q_median=0.5, p_any_median=0.3,
                        ey_any_median=100.0, psi_median=1.5, p_psi_gt_1=0.99)
    _write_vg15_outputs(var_dir, q_median=0.5, p_any_median=0.3,
                        ey_any_median=100.0, psi_median=1.5, p_psi_gt_1=0.99)
    _write_gate_payload(var_dir)

    comparison = compare_dirs(str(base_dir), str(var_dir))
    row = summarise(comparison, str(var_dir), label="vg15-variant",
                    baseline_dir=str(base_dir), validation_errors=[],
                    coverage=coverage_report(str(base_dir), str(var_dir)))
    assert row["status"] == "compared"
    assert row["caveats"] == ""
    assert row["verdict"] == "robust (all within baseline 89% interval)"
