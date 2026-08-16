# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The three ways a sensitivity comparison can be confidently wrong.

Every one of these produced a well-formed matrix row on 2026-08-16 — none
produced an error, a blank, or a missing file. A harness that reads two
directories of CSVs will compare whatever it finds, so the guards have to be
positive checks rather than the absence of a crash. See
``notes/202608142000-refit-run-record-and-disk-failure.md`` §7.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from vocab_growth.sensitivity.compare import (
    MIN_COVERAGE,
    coverage_report,
    definition_mismatch,
    failed_fit_dir,
    summarise,
    summarise_absent,
)


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


def test_a_variant_differing_only_in_its_overrides_is_a_sound_pairing(tmp_path):
    base = _write_fit(tmp_path, "base", BASE_DEF, ([12, 24], [10.0, 20.0]))
    variant = _write_fit(
        tmp_path,
        "variant",
        {**BASE_DEF, "config_name": "base-narrow", "banner": "x", "eta_sigma": 0.4},
        ([12, 24], [10.0, 20.0]),
    )
    assert definition_mismatch(base, variant, {"eta_sigma"}) == []


def test_a_definition_change_outside_the_overrides_is_caught_by_name(tmp_path):
    """The live case: the baseline was refitted under CLAMP_Q_ONLY mid-run."""
    base = _write_fit(tmp_path, "base", BASE_DEF, ([12, 24], [10.0, 20.0]))
    variant = _write_fit(
        tmp_path,
        "variant",
        {**BASE_DEF, "config_name": "base-narrow", "eta_sigma": 0.4,
         "clamp_mean_above_hi_anchor": True},
        ([12, 24], [10.0, 20.0]),
    )
    assert definition_mismatch(base, variant, {"eta_sigma"}) == [
        "clamp_mean_above_hi_anchor"
    ]


def test_an_unverifiable_pairing_is_not_reported_as_a_mismatch(tmp_path):
    """No manifest means "cannot check", which is not the same as "differs"."""
    base = tmp_path / "base"
    base.mkdir()
    variant = tmp_path / "variant"
    variant.mkdir()
    assert definition_mismatch(str(base), str(variant), {"eta_sigma"}) == []


def test_a_stale_pairing_is_never_reported_as_robust():
    comparison = pd.DataFrame([{
        "quantity": "Ey", "age_months": 12, "base_median": 10.0,
        "var_median": 10.0, "delta": 0.0, "base_ci_lo": 9.0, "base_ci_hi": 11.0,
        "within_baseline_ci": True, "interval_kind": "eti",
    }])
    row = summarise(comparison, "/nonexistent", "v", mismatch=["clamp_mean_above_hi_anchor"])
    assert row["status"] == "stale-pairing"
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
    comparison = pd.DataFrame([{
        "quantity": "gap", "age_months": 12, "base_median": 10.0,
        "var_median": 99.0, "delta": 89.0, "base_ci_lo": 9.0, "base_ci_hi": 11.0,
        "within_baseline_ci": False, "interval_kind": "eti",
    }])
    row = summarise(comparison, "/nonexistent", "v", mismatch=[], coverage=(355, 3, []))
    assert row["status"] == "partial-coverage"
    assert row["coverage"] < MIN_COVERAGE
    # The delta is large and outside the interval, but on 3 of 355 points that
    # is not a sensitivity finding.
    assert "sensitive" not in row["verdict"]


def test_full_coverage_reaches_a_real_verdict():
    comparison = pd.DataFrame([{
        "quantity": "Ey", "age_months": 12, "base_median": 10.0,
        "var_median": 10.0, "delta": 0.0, "base_ci_lo": 9.0, "base_ci_hi": 11.0,
        "within_baseline_ci": True, "interval_kind": "eti",
    }])
    row = summarise(comparison, "/nonexistent", "v", mismatch=[], coverage=(1, 1, []))
    assert row["status"] == "compared"
    assert row["verdict"].startswith("robust")


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
