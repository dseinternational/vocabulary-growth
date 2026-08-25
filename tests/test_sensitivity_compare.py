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

import pandas as pd

from vocab_growth.sensitivity.compare import compare_dirs, summarise


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
    assert row["verdict"] == "sensitive: psi"
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
    _write_vg15_outputs(base_dir, q_median=0.5, p_any_median=0.3,
                        ey_any_median=100.0, psi_median=1.5, p_psi_gt_1=0.99)
    # q shifts outside the baseline 89% interval; p_any, Ey_any and psi stay within
    # (base Ey_any interval is [90, 150]; the variant median 120 sits inside it).
    _write_vg15_outputs(var_dir, q_median=0.7, p_any_median=0.35,
                        ey_any_median=120.0, psi_median=1.6, p_psi_gt_1=0.98)
    pd.DataFrame({
        "r_hat": [1.005], "ess_bulk": [1000.0], "ess_tail": [900.0],
    }).to_csv(var_dir / "diagnostics.csv", index=False)

    comparison = compare_dirs(str(base_dir), str(var_dir))
    by_qty = comparison.set_index("quantity")["within_baseline_ci"]
    # Ey_any now carries an interval and is assessed; only the interval-less
    # P_psi_gt_1 row stays None, so the column is still object dtype.
    assert by_qty["Ey_any"] is True
    assert by_qty["P_psi_gt_1"] is None
    assert by_qty["q"] is False
    assert by_qty["p_any"] is True
    assert by_qty["psi"] is True

    row = summarise(comparison, str(var_dir), label="vg15-variant")
    assert row["converged"] is True
    assert row["n_checked"] == 4
    assert row["n_within_ci"] == 3
    assert row["quantities_outside_ci"] == "q"
    assert row["verdict"] == "sensitive: q"


# --- The VG16 cross-lag coefficient (issue #242) -------------------------------


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


def test_load_beta_lag_reads_the_scalar_from_diagnostics(tmp_path):
    from vocab_growth.sensitivity.compare import load_beta_lag

    _write_diagnostics(tmp_path, beta_lag=(0.199, 0.089, 0.311))
    assert load_beta_lag(str(tmp_path)) == {
        "beta_lag_median": 0.199,
        "beta_lag_ci_lo": 0.089,
        "beta_lag_ci_hi": 0.311,
    }


def test_load_beta_lag_is_none_without_a_cross_lag(tmp_path):
    """Every bivariate fit writes diagnostics.csv; only VG16 carries beta_lag."""
    from vocab_growth.sensitivity.compare import load_beta_lag

    _write_diagnostics(tmp_path, beta_lag=None)
    assert load_beta_lag(str(tmp_path)) is None
    assert load_beta_lag(str(tmp_path / "does-not-exist")) is None


def test_a_moved_cross_lag_is_scored_rather_than_ignored(tmp_path):
    """The defect this closes: a variant that halves beta_lag while leaving the
    trajectories alone was previously scored **robust**, because beta_lag was in
    no compared series. VG16 supplies no other reported number."""
    base_dir, var_dir = tmp_path / "base", tmp_path / "var"
    base_dir.mkdir(), var_dir.mkdir()
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

    row = summarise(comparison, str(var_dir), label="vg16-conditional-only")
    assert "beta_lag" in row["quantities_outside_ci"]
    assert row["verdict"].startswith("sensitive")


def test_beta_lag_within_the_baseline_interval_scores_robust(tmp_path):
    base_dir, var_dir = tmp_path / "base", tmp_path / "var"
    base_dir.mkdir(), var_dir.mkdir()
    _write_vg15_outputs(base_dir, q_median=0.5, p_any_median=0.3,
                        ey_any_median=100.0, psi_median=1.5, p_psi_gt_1=0.99)
    _write_vg15_outputs(var_dir, q_median=0.5, p_any_median=0.3,
                        ey_any_median=100.0, psi_median=1.5, p_psi_gt_1=0.99)
    _write_diagnostics(base_dir, beta_lag=(0.20, 0.09, 0.31))
    _write_diagnostics(var_dir, beta_lag=(0.22, 0.10, 0.33))

    comparison = compare_dirs(str(base_dir), str(var_dir))
    assert comparison.set_index("quantity")["within_baseline_ci"]["beta_lag"] is True
