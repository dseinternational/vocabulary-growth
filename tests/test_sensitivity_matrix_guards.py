# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The three ways a sensitivity comparison can be confidently wrong.

Every one of these produced a well-formed matrix row on 2026-08-16 — none
produced an error, a blank, or a missing file. A harness that reads two
directories of CSVs will compare whatever it finds, so the guards have to be
positive checks rather than the absence of a crash. See
``notes/202608142000-refit-run-record-and-disk-failure.md`` §7.

A fourth was found on 2026-09-06 and is guarded at the end of this file: a
targeted rerun retains the rows it did not recompute, and a retained row was
scored against whatever baseline existed at the time. After a refit the matrix
therefore presents verdicts against two different baselines side by side. That
one is not a bad row either -- it is a true record, mislabelled by its
neighbours.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from vocab_growth.sensitivity.compare import (
    MIN_COVERAGE,
    STALE_BASELINE_STATUS,
    coverage_report,
    failed_fit_dir,
    merge_retained_rows,
    summarise,
    summarise_absent,
)


def _row(quantity, base, var, lo, hi, within):
    """One ``compare_dirs`` row, in the schema ``compare_dirs`` actually emits."""
    return {
        "quantity": quantity, "age_months": 12.0,
        "base_estimate": base, "var_estimate": var, "estimate_kind": "median",
        "delta": var - base, "base_ci_lo": lo, "base_ci_hi": hi,
        "within_baseline_ci": within, "interval_kind": "eti",
    }


def _write_fit(root, name, definition, series):
    """A minimal fit directory: a manifest and one headline series."""
    d = root / name
    d.mkdir()
    (d / "fit_manifest.json").write_text(
        json.dumps({
            "created_at_utc": "2026-08-16T00:00:00+00:00",
            "model": {"definition": definition},
        })
    )
    ages, medians = series
    pd.DataFrame({
        "age_months": ages,
        "Ey_median": medians,
        "Ey_ci_lo": [m - 1.0 for m in medians],
        "Ey_ci_hi": [m + 1.0 for m in medians],
    }).to_csv(d / "posterior_summary.csv", index=False)
    return d


BASE_DEF = {
    "model_id": "VG10",
    "config_name": "base",
    "banner": "…",
    "eta_sigma": 0.5,
    "clamp_mean_above_hi_anchor": "q_only",
}


# ---------------------------------------------------------------- stale pairing


def test_a_stale_pairing_is_never_reported_as_robust(tmp_path):
    """The live case: the baseline was refitted under CLAMP_Q_ONLY mid-run.

    ``pairing_errors`` is what detects this now — it validates each fit against
    the definition the registry currently builds, values and all, rather than
    diffing the two manifests for unexpected field names. The guard kept here is
    the consequence: whatever the containment says, an unsound pairing must not
    reach a robustness verdict, and the reason must reach the matrix row.
    """
    comparison = pd.DataFrame([_row("Ey", 10.0, 10.0, 9.0, 11.0, True)])
    _write_clean_gate_payload(tmp_path)
    row = summarise(
        comparison, str(tmp_path), "v",
        baseline_dir=str(tmp_path), coverage=(1, 1, []),
        validation_errors=["variant: the model definition differs from the current "
                           "registered definition: clamp_mean_above_hi_anchor."],
    )
    assert row["status"] == "unverified-pairing"
    assert "robust" not in row["verdict"]
    assert "clamp_mean_above_hi_anchor" in row["verdict"]


# ------------------------------------------------------------------- coverage


def test_coverage_uses_the_comparison_own_matching_rule(tmp_path):
    """Plot-grid series only align when the two fits share an age range.

    ``gap`` is a linspace over the observed span, so a pool-restricting variant
    gets different ages and the intersection collapses. Measuring coverage any
    other way overstates what the comparison actually paired up.
    """
    base = _write_fit(tmp_path, "base", BASE_DEF, ([8.0, 8.5, 9.0, 9.5], [1.0, 2.0, 3.0, 4.0]))
    variant = _write_fit(
        tmp_path, "variant", BASE_DEF, ([8.0, 8.4, 8.8, 9.2], [1.0, 2.0, 3.0, 4.0])
    )
    baseline_rows, shared_rows, missing = coverage_report(base, variant)
    assert baseline_rows == 4
    assert shared_rows == 1  # only 8.0 coincides
    assert missing == []


def test_collapsed_coverage_is_not_assessed():
    comparison = pd.DataFrame([_row("gap", 10.0, 99.0, 9.0, 11.0, False)])
    row = summarise(comparison, "/nonexistent", "v", baseline_dir="/nonexistent",
                    validation_errors=[], coverage=(355, 3, []))
    assert row["status"] == "partial-coverage"
    assert row["coverage"] < MIN_COVERAGE
    # The delta is large and outside the interval, but on 3 of 355 points that
    # is not a sensitivity finding.
    assert "sensitive" not in row["verdict"]


def _write_clean_gate_payload(dirpath):
    """A cleanly passing ``diagnostics_summary.json`` — what "robust" needs."""
    (dirpath / "diagnostics_summary.json").write_text(
        json.dumps({
            "passed": True,
            "checks": {
                "rhat": True, "ess": True, "divergences": True, "bfmi": True,
                "diagnostics_assessable": True,
            },
            "divergences": 0,
            "max_rhat": 1.004,
            "min_ess": 1500.0,
            "bfmi_per_chain": [0.9, 0.85],
            "rhat_failing": [],
            "ess_failing": [],
            "unassessable_parameters": [],
            "thresholds": {
                "rhat_max": 1.01, "ess_threshold": 400, "bfmi_threshold": 0.3,
            },
        }),
        encoding="utf-8",
    )


def test_full_coverage_reaches_a_real_verdict(tmp_path):
    comparison = pd.DataFrame([_row("Ey", 10.0, 10.0, 9.0, 11.0, True)])
    _write_clean_gate_payload(tmp_path)
    row = summarise(comparison, str(tmp_path), "v", coverage=(1, 1, []),
                    baseline_dir=str(tmp_path), validation_errors=[])
    assert row["status"] == "compared"
    assert row["verdict"].startswith("robust")

    # Without any recorded convergence gate the containment is still reported,
    # but "robust" is reserved for a fit with a cleanly passing payload.
    row = summarise(comparison, "/nonexistent", "v", coverage=(1, 1, []))
    assert row["status"] == "unverified-pairing"
    assert not row["verdict"].startswith("robust")


def test_unchecked_coverage_says_so_rather_than_blaming_the_gate(tmp_path):
    """Both gates clean, no coverage supplied: name the input that is missing."""
    comparison = pd.DataFrame([_row("Ey", 10.0, 10.0, 9.0, 11.0, True)])
    _write_clean_gate_payload(tmp_path)
    row = summarise(comparison, str(tmp_path), "v",
                    baseline_dir=str(tmp_path), validation_errors=[])
    assert row["status"] == "unverified-coverage"
    assert "COVERAGE NOT CHECKED" in row["verdict"]
    assert "convergence" not in row["verdict"]


# -------------------------------------------------------- variants that vanish


def test_a_failed_fit_is_found_under_the_failed_root(tmp_path):
    failed = tmp_path / "failed"
    failed.mkdir()
    (failed / "VG11-age-spoken-td-re-anchor-broad-20260815T154328Z").mkdir()
    found = failed_fit_dir(str(failed), "VG11", "age-spoken-td-re-anchor-broad")
    assert found is not None and found.endswith("20260815T154328Z")


def test_the_most_recent_failed_fit_wins(tmp_path):
    failed = tmp_path / "failed"
    failed.mkdir()
    for stamp in ("20260814T000000Z", "20260815T154328Z"):
        (failed / f"VG11-cfg-{stamp}").mkdir()
    assert failed_fit_dir(str(failed), "VG11", "cfg").endswith("20260815T154328Z")


def test_a_missing_failed_root_is_not_an_error(tmp_path):
    assert failed_fit_dir(str(tmp_path / "nope"), "VG11", "cfg") is None


@pytest.mark.parametrize("status", ["not-fitted", "failed"])
def test_an_absent_variant_is_a_row_not_an_omission(status):
    """A matrix that silently drops what it could not assess reads as coverage."""
    row = summarise_absent("anchor-broad", status, "reason")
    assert row["variant"] == "anchor-broad"
    assert row["status"] == status
    assert row["n_checked"] == 0
    assert row["verdict"] == "reason"


# --- A carried-over row scored against a superseded baseline (issue #266) ---


def _matrix_row(variant, *, status="compared", baseline="2026-09-01T00:00:00+00:00",
                verdict="robust"):
    return {
        "variant": variant,
        "status": status,
        "verdict": verdict,
        "max_abs_delta": 1.0,
        "baseline_fit_utc": baseline,
        "variant_fit_utc": "2026-09-01T00:00:00+00:00",
    }


def test_a_retained_row_against_a_superseded_baseline_is_marked():
    """The concrete defect: two verdicts side by side, scored against two baselines.

    ``robustness_matrix_vg10.csv`` held exactly this on 2026-09-06 — one row
    against the pre-``us_03`` fit and two against the refitted one, with nothing
    in the presented columns to tell them apart.
    """
    previous = pd.DataFrame([_matrix_row("us01-implausible-reinstated")])
    recomputed = pd.DataFrame(
        [_matrix_row("dse-native-only", baseline="2026-09-06T11:57:26+00:00")]
    )
    merged = merge_retained_rows(
        previous, recomputed, baseline_fit_utc="2026-09-06T11:57:26+00:00"
    )
    retained = merged[merged["variant"] == "us01-implausible-reinstated"].iloc[0]
    assert retained["status"] == STALE_BASELINE_STATUS
    assert "STALE BASELINE" in retained["verdict"]
    # The baseline it actually used is named, not merely flagged.
    assert "2026-09-01T00:00:00+00:00" in retained["verdict"]
    # ... and the recomputed row is untouched.
    current = merged[merged["variant"] == "dse-native-only"].iloc[0]
    assert current["status"] == "compared"
    assert current["verdict"] == "robust"


def test_a_retained_row_against_the_same_baseline_is_left_alone():
    baseline = "2026-09-06T11:57:26+00:00"
    previous = pd.DataFrame([_matrix_row("no-us01", baseline=baseline)])
    recomputed = pd.DataFrame([_matrix_row("dse-native-only", baseline=baseline)])
    merged = merge_retained_rows(previous, recomputed, baseline_fit_utc=baseline)
    retained = merged[merged["variant"] == "no-us01"].iloc[0]
    assert retained["status"] == "compared"
    assert retained["verdict"] == "robust"


@pytest.mark.parametrize("status", ["not-fitted", "failed", "invalid-summary"])
def test_only_a_comparison_can_be_stale_not_an_absence(status):
    """A new baseline does not change "nobody ran it".

    ``summarise_absent`` rows carry ``baseline_fit_utc`` like any other, but
    their verdict is a statement about the variant rather than about a
    comparison, so re-marking them would be a false report of staleness.
    """
    previous = pd.DataFrame([_matrix_row("no-us01", status=status,
                                         verdict="NOT ASSESSED: no fit")])
    recomputed = pd.DataFrame(
        [_matrix_row("dse-native-only", baseline="2026-09-06T11:57:26+00:00")]
    )
    merged = merge_retained_rows(
        previous, recomputed, baseline_fit_utc="2026-09-06T11:57:26+00:00"
    )
    retained = merged[merged["variant"] == "no-us01"].iloc[0]
    assert retained["status"] == status
    assert retained["verdict"] == "NOT ASSESSED: no fit"


def test_the_retained_rows_numbers_survive_the_mark():
    """Marking, not dropping: the row is a true record of a comparison made."""
    previous = pd.DataFrame([_matrix_row("us01-implausible-reinstated")])
    merged = merge_retained_rows(
        previous,
        pd.DataFrame([_matrix_row("dse-native-only", baseline="2026-09-06T00:00:00+00:00")]),
        baseline_fit_utc="2026-09-06T00:00:00+00:00",
    )
    retained = merged[merged["variant"] == "us01-implausible-reinstated"].iloc[0]
    assert retained["max_abs_delta"] == 1.0
    assert retained["variant_fit_utc"] == "2026-09-01T00:00:00+00:00"


def test_the_registered_order_is_restored_after_a_merge():
    previous = pd.DataFrame([_matrix_row("no-us01"), _matrix_row("lag-gap-12")])
    recomputed = pd.DataFrame([_matrix_row("dse-native-only")])
    merged = merge_retained_rows(
        previous,
        recomputed,
        baseline_fit_utc="2026-09-01T00:00:00+00:00",
        order=["lag-gap-12", "dse-native-only", "no-us01"],
    )
    assert list(merged["variant"]) == ["lag-gap-12", "dse-native-only", "no-us01"]


def test_a_matrix_with_no_variant_column_is_replaced_not_merged():
    """An unreadable standing matrix must not be silently half-merged."""
    recomputed = pd.DataFrame([_matrix_row("dse-native-only")])
    merged = merge_retained_rows(
        pd.DataFrame({"something_else": [1]}),
        recomputed,
        baseline_fit_utc="2026-09-01T00:00:00+00:00",
    )
    assert list(merged["variant"]) == ["dse-native-only"]
